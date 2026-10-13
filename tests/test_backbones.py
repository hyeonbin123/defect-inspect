# [파일 역할] 고정(frozen) 패치 특징 추출기 모듈 `defect_inspect.backbones`를 검증하는 테스트 묶음이다. 대상은
# 입력 정규화(to_tensor), WideResNet-50-2 패치 특징(P0), DINOv2 ViT 패치 토큰(D-S, 지도 학습 특징), 7단계
# 이미지별 중심화(CentredPatches), 이름으로 추출기를 고르는 make_extractor다.
# [왜 필요한가] PatchCore의 이상 점수는 이 특징 사이의 거리라서, 특징 정의가 docs/experiments.md에 등록한
# 것(1단계 P0: layer2·layer3, 3×3 평균 풀링, layer3을 layer2 격자로 bilinear 확대, 1,536차원 / 2단계 D-S:
# register 4개 ViT-S의 마지막 패치 토큰, L2 정규화)과 어긋나면 모든 단계의 수치가 달라진다.
# [설계] 사전학습 가중치를 내려받지 않는다. pretrained=False의 무작위 가중치로 모양·계산 경로를 보고, 가중치
# 적재는 monkeypatch로 다운로드 함수를 가짜로 바꿔 확인한다(네트워크·GPU 없이 돈다).

import numpy as np
import pytest

# torch·torchvision은 train 그룹에만 있으므로 없으면 파일 전체를 건너뛴다(CI는 기본 패키지만 설치한다). 그래서
# 아래 import에 E402 예외 표시가 붙어 있다.
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch.nn.functional as F  # noqa: E402

from defect_inspect import backbones  # noqa: E402
from defect_inspect.backbones import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    DinoV2Patches,
    WideResNetPatches,
    make_extractor,
    to_tensor,
)


# 무작위 가중치 WRN-50 추출기를 파일 안에서 한 번만 만들어 여러 테스트가 같이 쓴다(scope="module"). 만드는
# 비용이 커서 테스트마다 새로 만들지 않는다. manual_seed로 가중치를 고정한다.
@pytest.fixture(scope="module")
def wrn():
    torch.manual_seed(0)
    return WideResNetPatches(pretrained=False)


# 확인: uint8 [B, S, S, 3] 이미지가 ImageNet 평균·표준편차로 정규화된 [B, 3, S, S] 텐서가 되고, 값이 float64로
# 직접 계산한 것과 같으며, 입력 배열은 바뀌지 않는다(전처리 규칙: 백본의 ImageNet 평균·표준편차).
def test_to_tensor_normalises_and_reorders():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(2, 5, 5, 3), dtype=np.uint8)
    x = to_tensor(images, "cpu", torch.float32)
    assert x.shape == (2, 3, 5, 5)
    assert x.dtype == torch.float32
    expected = (images.astype(np.float64) / 255.0 - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    np.testing.assert_allclose(x.numpy(), expected.transpose(0, 3, 1, 2), atol=1e-6)
    # The input array is left untouched.
    assert images.dtype == np.uint8


# 확인: 요청한 dtype(fp16)으로 내보내되 fp32 결과와 fp16 반올림 범위 안에서 같고, uint8이 아니거나 배치 축이
# 없는 입력은 ValueError다. to_tensor는 fp32로 정규화한 뒤 마지막에 한 번만 바꾸므로 fp16 경로에서 정밀도를
# 추가로 잃지 않아야 한다.
def test_to_tensor_dtype_and_validation():
    images = np.full((1, 4, 4, 3), 128, dtype=np.uint8)
    half = to_tensor(images, torch.device("cpu"), torch.float16)
    assert half.dtype == torch.float16
    full = to_tensor(images, "cpu", torch.float32)
    np.testing.assert_allclose(half.float().numpy(), full.numpy(), atol=2e-3)
    with pytest.raises(ValueError):
        to_tensor(images.astype(np.float32), "cpu", torch.float32)
    with pytest.raises(ValueError):
        to_tensor(images[0], "cpu", torch.float32)


# 확인: 슬라이스로 만든 비연속(non-contiguous) 배열도 연속 사본과 같은 결과를 낸다. 구현은 ascontiguousarray로
# 연속 배열로 바꾼 뒤 torch로 옮기므로, 호출자가 배열 일부를 잘라 넘겨도 결과가 같아야 한다.
def test_to_tensor_accepts_non_contiguous_input():
    rng = np.random.default_rng(1)
    images = rng.integers(0, 256, size=(4, 6, 6, 3), dtype=np.uint8)
    np.testing.assert_array_equal(
        to_tensor(images[::2], "cpu", torch.float32).numpy(),
        to_tensor(images[::2].copy(), "cpu", torch.float32).numpy(),
    )


# 확인: WRN-50 추출기의 이름·차원(512 + 1024 = 1,536)과 출력 모양이다. 64px 입력이면 layer2 격자가 8×8이므로
# [B, 8, 8, 1536]이다(256px이면 32×32, 이미지당 1,024패치).
def test_wrn_output_shape_and_attributes(wrn):
    assert wrn.name == "wrn50"
    assert wrn.dim == 1536
    x = torch.randn(2, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        out = wrn(x)
    assert out.shape == (2, 8, 8, 1536)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


# 확인: .train()을 불러도 eval 모드가 유지되고(BatchNorm 통계 고정), 모든 파라미터가 학습되지 않으며, layer4와
# 분류기(fc)는 아예 갖고 있지 않다. state_dict 키는 torchvision 이름 그대로라 사전학습 가중치가 일대일로
# 들어간다.
# 왜: BatchNorm이 train 모드로 바뀌면 배치 구성에 따라 특징이 달라져 뱅크와 질의가 같은 특징 공간이 아니게
# 된다.
def test_wrn_is_frozen_and_stops_at_layer3(wrn):
    assert not wrn.training
    wrn.train()
    assert not wrn.training and not wrn.bn1.training and not wrn.layer3[0].bn1.training
    assert all(not p.requires_grad for p in wrn.parameters())
    assert not hasattr(wrn, "layer4") and not hasattr(wrn, "fc")
    # State dict keys keep the torchvision names, so the pretrained weights map one to one.
    keys = set(wrn.state_dict())
    assert "conv1.weight" in keys and "layer3.5.conv3.weight" in keys
    assert not any(k.startswith(("layer4", "fc")) for k in keys)


# 확인: pretrained=True면 IMAGENET1K_V1 가중치를 해시 검사(check_hash)와 함께 받고, TORCH_HOME이 없으면 저장
# 위치를 프로젝트의 data/torch/checkpoints로 둔다. TORCH_HOME이 있으면 torch 기본 위치를 따른다.
# 왜: 등록한 P0 구성의 가중치가 IMAGENET1K_V1이고, 내려받은 것은 홈 디렉터리가 아니라 data/ 아래에 둔다는
# 프로젝트 규칙이 있다(_checkpoint_dir docstring).
# monkeypatch: torchvision의 load_state_dict_from_url을 무작위 가중치 state_dict를 돌려주는 가짜로 바꿔
# 네트워크 없이 호출 인자와 적재 결과만 확인한다.
def test_wrn_pretrained_loads_v1_weights_into_the_data_directory(monkeypatch):
    # No network: the torchvision download helper is replaced by one that hands back known weights.
    from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
    from torchvision.models import _api as tv_api

    from defect_inspect import paths

    reference = wide_resnet50_2(weights=None)
    calls = []

    def fake_loader(url, *args, **kwargs):
        calls.append((url, args, kwargs))
        return reference.state_dict()

    monkeypatch.setattr(tv_api, "load_state_dict_from_url", fake_loader)
    monkeypatch.delenv("TORCH_HOME", raising=False)
    model = WideResNetPatches(pretrained=True)
    url, args, kwargs = calls[0]
    assert url == Wide_ResNet50_2_Weights.IMAGENET1K_V1.url and args == ()
    assert kwargs["check_hash"] is True
    assert kwargs["model_dir"] == str(paths.DATA / "torch" / "checkpoints")
    assert torch.equal(model.conv1.weight, reference.conv1.weight)
    assert torch.equal(model.layer3[5].conv3.weight, reference.layer3[5].conv3.weight)
    assert torch.equal(model.layer2[0].bn1.running_var, reference.layer2[0].bn1.running_var)
    assert not model.training and all(not p.requires_grad for p in model.parameters())

    # An explicit TORCH_HOME is respected (torch's own default directory).
    monkeypatch.setenv("TORCH_HOME", "somewhere")
    assert backbones._checkpoint_dir() is None


# 확인: 출력의 앞 512차원은 3×3 평균 풀링한 layer2, 뒤 1,024차원은 3×3 평균 풀링한 뒤 layer2 격자로 bilinear
# 확대한 layer3이다. forward hook으로 중간 출력을 잡아 직접 계산한 값과 비교한다.
# 왜: 1단계 P0 규칙의 패치 특징 정의를 그대로 따르는지 보는 테스트다. 모서리 값 검사는 패딩이 0으로 채워지고
# 9로 나누는지(count_include_pad 기본값)까지 확인한다.
def test_wrn_features_are_pooled_layer2_then_upsampled_layer3(wrn):
    captured = {}
    hooks = [
        wrn.layer2.register_forward_hook(lambda m, i, o: captured.__setitem__("l2", o)),
        wrn.layer3.register_forward_hook(lambda m, i, o: captured.__setitem__("l3", o)),
    ]
    x = torch.randn(1, 3, 64, 64, generator=torch.Generator().manual_seed(1))
    try:
        with torch.no_grad():
            out = wrn(x)
    finally:
        for hook in hooks:
            hook.remove()
    assert captured["l2"].shape == (1, 512, 8, 8)
    assert captured["l3"].shape == (1, 1024, 4, 4)
    pooled2 = F.avg_pool2d(captured["l2"], 3, 1, 1)
    pooled3 = F.avg_pool2d(captured["l3"], 3, 1, 1)
    up3 = F.interpolate(pooled3, size=(8, 8), mode="bilinear", align_corners=False)
    torch.testing.assert_close(out[..., :512], pooled2.permute(0, 2, 3, 1))
    torch.testing.assert_close(out[..., 512:], up3.permute(0, 2, 3, 1))
    # 3x3 average with zero padding: a corner value is the sum of its 2x2 neighbourhood over 9.
    corner = captured["l2"][0, :, :2, :2].sum(dim=(1, 2)) / 9.0
    torch.testing.assert_close(out[0, 0, 0, :512], corner, rtol=1e-4, atol=1e-6)


# 확인: 같은 입력에 같은 출력이 나오고, 한 이미지의 특징이 같은 배치의 다른 이미지에 좌우되지 않는다(eval 모드
# BatchNorm). 배치 크기가 달라도 뱅크와 질의 특징이 같아야 한다.
def test_wrn_is_deterministic_in_eval_mode(wrn):
    x = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        a, b = wrn(x), wrn(x)
        single = wrn(x[:1])
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    # Eval-mode BatchNorm: an image's features do not depend on the rest of the batch.
    torch.testing.assert_close(a[:1], single, rtol=1e-4, atol=1e-5)


# 확인: DINOv2 추출기가 class 토큰 1개와 register 토큰 4개를 버리고 남은 패치 토큰을 래스터 순서로 [B, H, W,
# 384]에 놓으며, 각 토큰이 L2 정규화돼 길이가 1이다. 또 timm이 위치 임베딩을 입력 크기(28px → 2×2)에 맞춰
# 만든다.
# 왜: 2단계 D-S 규칙(register 4개 ViT-S, 마지막 층 패치 토큰, L2 정규화)을 따르는지 확인한다. 위치 임베딩이
# 기본 크기 그대로면 448px 입력의 32×32 격자와 맞지 않는다.
def test_dinov2_patches_are_normalised_tokens():
    pytest.importorskip("timm")
    torch.manual_seed(0)
    model = DinoV2Patches(img_size=28, pretrained=False)
    assert model.name == "dinov2_vits14"
    assert model.dim == 384
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())
    x = torch.randn(2, 3, 28, 28, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        out = model(x)
        tokens = model.model.forward_features(x)
    assert out.shape == (2, 2, 2, 384)
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(2, 2, 2), atol=1e-5, rtol=0)
    # One class token and four register tokens are dropped; the rest are the patches in raster order.
    assert tokens.shape == (2, 5 + 4, 384)
    expected = F.normalize(tokens[:, 5:], dim=-1).reshape(2, 2, 2, 384)
    torch.testing.assert_close(out, expected)
    # The model is built for this input size: a 2x2 position embedding, not timm's default 37x37.
    assert model.img_size == 28
    assert tuple(model.model.patch_embed.img_size) == (28, 28)
    assert model.model.pos_embed.shape == (1, 4, 384)


# 확인: make_extractor·DinoV2Patches가 timm.create_model에 올바른 아키텍처 이름, pretrained, num_classes=0,
# img_size를 넘긴다. 클래스 기본값은 사전학습 가중치·448px(D-S 구성)이다.
# monkeypatch: create_model을 감싸 인자를 기록하고 실제 생성은 pretrained=False로 해서 가중치를 내려받지
# 않는다.
def test_dinov2_arguments_reach_timm(monkeypatch):
    timm = pytest.importorskip("timm")
    real_create = timm.create_model
    calls = []

    def fake_create(arch, *args, **kwargs):
        calls.append((arch, args, dict(kwargs)))
        # No network: build the same model with random weights.
        return real_create(arch, *args, **{**kwargs, "pretrained": False})

    monkeypatch.setattr(timm, "create_model", fake_create)
    model = make_extractor("dinov2_vitb14", img_size=42, pretrained=True)
    arch, args, kwargs = calls[0]
    assert len(calls) == 1 and arch == "vit_base_patch14_reg4_dinov2" and args == ()
    assert kwargs["pretrained"] is True and kwargs["num_classes"] == 0 and kwargs["img_size"] == 42
    assert model.model.pos_embed.shape == (1, 9, 768)

    make_extractor("dinov2_vits14", img_size=28, pretrained=False)
    assert calls[1][0] == "vit_small_patch14_reg4_dinov2" and calls[1][2]["pretrained"] is False
    # The class default is pretrained weights at 448 px.
    DinoV2Patches()
    assert calls[2][0] == "vit_small_patch14_reg4_dinov2"
    assert calls[2][2]["pretrained"] is True and calls[2][2]["img_size"] == 448


# 확인: make_extractor("wrn50")가 pretrained 인자를 그대로 전달한다. False면 다운로드를 부르지 않고, True거나
# 생략하면 V1 가중치 URL로 한 번 부른다(기본값은 사전학습 가중치).
def test_make_extractor_passes_pretrained_to_the_wide_resnet(monkeypatch):
    from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
    from torchvision.models import _api as tv_api

    reference = wide_resnet50_2(weights=None)
    urls = []

    def fake_loader(url, *args, **kwargs):
        urls.append(url)
        return reference.state_dict()

    monkeypatch.setattr(tv_api, "load_state_dict_from_url", fake_loader)
    random = make_extractor("wrn50", img_size=256, pretrained=False)
    assert urls == []
    assert not torch.equal(random.conv1.weight, reference.conv1.weight)
    loaded = make_extractor("wrn50", img_size=256, pretrained=True)
    assert urls == [Wide_ResNet50_2_Weights.IMAGENET1K_V1.url]
    assert torch.equal(loaded.conv1.weight, reference.conv1.weight)
    # Pretrained weights are the default.
    make_extractor("wrn50", img_size=256)
    assert len(urls) == 2


# 확인: 입력 크기가 패치 크기 14의 배수가 아니면 ValueError다. 나머지가 생기면 토큰 격자와 이미지가
# 어긋난다(D-S 448, DM 392처럼 등록한 ViT 입력 크기는 모두 14의 배수다).
def test_dinov2_rejects_sizes_that_are_not_multiples_of_14():
    with pytest.raises(ValueError):
        DinoV2Patches(img_size=30, pretrained=False)
    with pytest.raises(ValueError):
        make_extractor("dinov2_vits14", img_size=100, pretrained=False)


# 확인: 짧은 이름 세 가지(wrn50, dinov2_vits14, dinov2_vitb14)가 맞는 클래스·차원(1,536, 384, 768)으로
# 만들어지고, 모르는 이름은 ValueError다. 42px ViT-S 입력이 3×3 격자가 되는지도 본다.
def test_make_extractor_names():
    pytest.importorskip("timm")
    assert set(backbones.DINOV2_ARCHS) == {"dinov2_vits14", "dinov2_vitb14"}
    wide = make_extractor("wrn50", img_size=256, pretrained=False)
    assert isinstance(wide, WideResNetPatches) and wide.name == "wrn50" and wide.dim == 1536
    small = make_extractor("dinov2_vits14", img_size=42, pretrained=False)
    assert isinstance(small, DinoV2Patches) and small.name == "dinov2_vits14" and small.dim == 384
    with torch.no_grad():
        assert small(torch.zeros(1, 3, 42, 42)).shape == (1, 3, 3, 384)
    base = make_extractor("dinov2_vitb14", img_size=28, pretrained=False)
    assert base.name == "dinov2_vitb14" and base.dim == 768
    with pytest.raises(ValueError):
        make_extractor("resnet18", img_size=256, pretrained=False)


# 중심화 테스트용 가짜 추출기. 고정된 2×2×2 특징 격자에 '입력 평균'만큼의 이미지별 오프셋을 더한다. 이미지
# 전체에 고르게 걸린 밝기 이동을 흉내 낸 것이다.
class _Shift(torch.nn.Module):
    """Stand-in extractor: a fixed grid of features plus a per-image offset taken from the input."""

    name = "shift"
    dim = 2

    def forward(self, x):
        base = torch.arange(8, dtype=torch.float32).view(1, 2, 2, 2)
        offset = x.mean(dim=(1, 2, 3)).view(-1, 1, 1, 1)
        return (base + offset).to(x.dtype)


# 확인: CentredPatches가 이미지마다 패치 격자 평균 벡터를 빼서, 오프셋만 다른 두 이미지가 같은 중심화 격자를
# 내고 채널별 평균이 0이 된다. 출력은 fp32이고 eval 모드가 유지된다.
# 왜: 7단계 E1(이미지별 특징 중심화)은 '이미지 전체에 고르게 걸린 밝기 이동을 지우려는 것'으로
# 등록됐다(docs/experiments.md 7단계 후보). 그 의도대로 이미지별 상수 이동이 사라지는지 본다.
def test_centred_patches_remove_each_images_mean_feature():
    from defect_inspect.backbones import CentredPatches

    centred = CentredPatches(_Shift())
    assert centred.name == "shift-centred" and centred.dim == 2
    x = torch.stack([torch.zeros(3, 4, 4), torch.full((3, 4, 4), 5.0)])
    with torch.no_grad():
        out = centred(x)
    assert out.shape == (2, 2, 2, 2) and out.dtype == torch.float32
    # The per-image offset is gone: both images give the same centred grid, with zero mean per channel.
    assert torch.allclose(out[0], out[1])
    assert torch.allclose(out.mean(dim=(1, 2)), torch.zeros(2, 2))
    base = torch.arange(8, dtype=torch.float32).view(2, 2, 2)
    assert torch.allclose(out[0], base - base.mean(dim=(0, 1), keepdim=True))
    # Frozen like every extractor: no trainable parameter, and train() keeps eval mode.
    assert not centred.train().training


# 확인: make_extractor(centre=True)가 같은 WRN-50을 CentredPatches로 감싸 이름에 '-centred'를 붙이고, 출력이
# 감싸지 않은 추출기 출력에서 이미지별 평균을 뺀 것과 같다. 7단계 설정 p0-c, d-s-c는
# run_patchcore·run_perturb·run_m2ad에서 centre=cfg.centre로 이 경로를 거친다.
def test_make_extractor_wraps_in_centring_on_request():
    from defect_inspect.backbones import CentredPatches

    torch.manual_seed(0)
    plain = make_extractor("wrn50", img_size=64, pretrained=False)
    torch.manual_seed(0)
    centred = make_extractor("wrn50", img_size=64, pretrained=False, centre=True)
    assert isinstance(centred, CentredPatches) and centred.name == "wrn50-centred" and centred.dim == 1536
    assert all(not p.requires_grad for p in centred.parameters())
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        a, b = plain(x), centred(x)
    assert torch.allclose(b, a - a.mean(dim=(1, 2), keepdim=True), atol=1e-5)
