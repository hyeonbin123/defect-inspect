# [파일 역할] 6단계 Dinomaly ViT-S(DMS) 모델을 위해 anomalib의 DinomalyModel에 가한
# 변경(`defect_inspect.dinomaly_model`)을 검증하는 테스트 묶음이다.
# [왜 필요한가] 이 모듈의 모든 변경은 '출력을 바꾸지 않는다'는 전제로 들어갔다(모듈 docstring,
# docs/experiments.md 6단계 공통 구성). (1) 키를 토큰 수로 나누는 선형 어텐션(scaled)은 수식상 같은 출력을
# 내면서 fp16에서 넘치던 합만 줄여야 하고, (2) 인코더를 10블록으로 자르고 입력 크기를 고정해도 점수·맵이 비트
# 단위로 같아야 하며, (3) ONNX로 내보내는 ServingGraph는 모델의 추론 경로와 run_dinomaly.predict가 저장하는
# 값과 같아야 한다. 6단계 규칙에 '무작위 가중치로 비트 단위까지 같은 것을 테스트로 확인'한다고 적혀 있는 그
# 테스트다.
# [흐름] 어텐션 세 가지(scaled·fp32·교체) → 고정 인코더 → 내보내는 그래프.

import copy

import numpy as np
import pytest

# torch·anomalib·timm은 train/dinomaly 그룹에만 있어 없으면 파일 전체를 건너뛴다(CI는 기본 패키지만 설치).
torch = pytest.importorskip("torch")
pytest.importorskip("anomalib")
pytest.importorskip("timm")

import torch.nn.functional as F  # noqa: E402
from anomalib.models.image.dinomaly.components import LinearAttention  # noqa: E402

from defect_inspect import dinomaly_model, run_dinomaly  # noqa: E402
from defect_inspect.backbones import to_tensor  # noqa: E402
from defect_inspect.dinomaly_model import (  # noqa: E402
    Fp32LinearAttention,
    ScaledLinearAttention,
    ServingGraph,
    attention_mode,
    fix_encoder,
    set_attention,
)

# 56px = 14px 패치 4×4 격자. CPU에서 빨리 돌도록 아주 작은 입력을 쓴다.
SIZE = 56  # a 4x4 patch grid: small enough for CPU tests


# 사전학습 가중치를 내려받지 않게 timm.create_model을 pretrained=False로 바꾸는 fixture.
# 또 anomalib는 모델을 만들 때 timm의 resample_abs_pos_embed를 antialias를 끈 함수로 바꿔 두는데, mp.setattr로
# 지금 값을 등록해 두면 context가 끝날 때 원래 함수로 되돌아간다. 이 파일이 다른 테스트 파일(예:
# DinoV2Patches를 쓰는 test_backbones)에 전역 변경을 남기지 않게 하려는 것이다.
@pytest.fixture(scope="module")
def no_download():
    """`timm.create_model` without pretrained weights; anomalib's pos-embed patch is undone afterwards."""
    import timm
    import timm.models.vision_transformer as vit

    with pytest.MonkeyPatch.context() as mp:
        real_create = timm.create_model

        def create_without_download(name, *args, **kwargs):
            kwargs["pretrained"] = False
            return real_create(name, *args, **kwargs)

        mp.setattr(timm, "create_model", create_without_download)
        mp.setattr(vit, "resample_abs_pos_embed", vit.resample_abs_pos_embed)
        yield


# ViT-S 인코더 DinomalyModel을 run_dinomaly.build_model로 만든다. 디코더 초기값이 전역 torch 생성기에서
# 나오므로 seed를 먼저 고정한다. options는 6단계 구성의 img_size·encoder_blocks·context_recentering 같은
# 인자다.
def _small(seed=0, **options):
    torch.manual_seed(seed)
    return run_dinomaly.build_model(run_dinomaly.ENCODER_S, **options).eval()


# seed로 고정한 무작위 uint8 이미지 [n, size, size, 3].
def _images(n, seed=0, size=SIZE):
    return np.random.default_rng(seed).integers(0, 256, (n, size, size, 3), dtype=np.uint8)


# anomalib LinearAttention 한 층(ViT-S 폭 384, 헤드 6). qkv 가중치를 4배로 키워 갓 초기화한 층보다 큰 활성값을
# 만든다(학습된 디코더처럼 합이 커지는 상황을 흉내 낸다).
def _attention(seed=0, dim=384, heads=6):
    torch.manual_seed(seed)
    layer = LinearAttention(dim, num_heads=heads, qkv_bias=True)
    with torch.no_grad():
        layer.qkv.weight.mul_(4.0)  # larger activations than a fresh layer, like a trained decoder
    return layer.eval()


# ---------------------------------------------------------------- attention


# 확인: ScaledLinearAttention이 원래 LinearAttention과 같은 출력을 낸다(fp64는 1e-12, fp32는 1e-5 상대 오차).
# 돌려주는 kv(디코더 블록은 쓰지 않음)만 1/토큰 수로 작아진다.
# 왜: 출력은 (q·(kᵀv)) / (q·Σk)라 키에 곱한 1/N이 분자·분모에서 약분된다(docs/experiments.md 6단계 관문 2).
# 405토큰은 280px 입력(20×20 패치 + 앞 토큰 5개)이다. 같은 객체의 __class__만 바꿔 같은 가중치로 비교한다.
@pytest.mark.parametrize(("dtype", "rtol"), [(torch.float64, 1e-12), (torch.float32, 1e-5)])
def test_scaled_attention_gives_the_output_of_the_original(dtype, rtol):
    layer = _attention().to(dtype)
    x = torch.randn(2, 405, 384, dtype=dtype, generator=torch.Generator().manual_seed(1)) * 3
    ref_out, ref_kv = layer(x)
    layer.__class__ = ScaledLinearAttention
    out, kv = layer(x)
    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=rtol * float(ref_out.abs().max()))
    # The key-value product it hands back (unused by the decoder block) is scaled by 1 / tokens.
    torch.testing.assert_close(kv * 405, ref_kv, rtol=rtol, atol=rtol * float(ref_kv.abs().max()))


# 확인: 원래 어텐션을 fp16으로 돌리면 kv 합이 inf가 되거나 출력이 유한하지 않은 입력에서, scaled 어텐션은
# 유한한 출력을 내고 fp64 기준과 2% 안에서 같다.
# 왜: 2단계 DM(ViT-B 392px)이 fp16 학습 15스텝째 NaN이 난 원인이 선형 어텐션 합의 fp16 넘침(65,504 초과)으로
# 기록돼 있다. 789토큰은 392px 입력(28×28 + 5)이다. 단, 이것은 무작위 가중치로 만든 합성 점검이고, 실제 DM이
# 패치로 fp16 학습이 되는지는 재지 않았다(README 6단계 결과).
def test_scaled_attention_keeps_the_fp16_sums_in_range():
    layer = _attention().double()
    with torch.no_grad():
        layer.qkv.weight.mul_(3.0)
    x = torch.randn(1, 789, 384, dtype=torch.float64, generator=torch.Generator().manual_seed(2)) * 4
    reference, _ = layer(x)
    half = copy.deepcopy(layer).half()
    out, kv = half(x.half())
    # The unscaled sums leave the fp16 range: the plain layer cannot give a finite result.
    assert float(kv.float().abs().max()) == float("inf") or not bool(torch.isfinite(out).all())
    half.__class__ = ScaledLinearAttention
    out, kv = half(x.half())
    assert bool(torch.isfinite(out).all()) and bool(torch.isfinite(kv).all())
    torch.testing.assert_close(out.double(), reference, rtol=2e-2, atol=2e-2 * float(reference.abs().max()))


# 확인: Fp32LinearAttention은 autocast 안에서도 fp32로 계산한다. 같은 autocast에서 원래 층은 저정밀도를
# 따르지만, fp32 층은 입력이 들어올 때 한 번 반올림된 것 말고는 fp32 계산 결과와 비트 단위로 같다.
# 왜: 6단계 관문 2 (나)에서 scaled가 실패하면 '선형 어텐션만 fp32(autocast 끔)'로 학습하는 대안이 등록돼 있다.
# GPU 없이 CPU autocast로 확인하려고 bf16을 쓴다.
def test_fp32_attention_ignores_autocast():
    layer = _attention()
    x = torch.randn(2, 100, 384, generator=torch.Generator().manual_seed(3))
    reference, _ = layer(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        low, _ = layer(x)
        layer.__class__ = Fp32LinearAttention
        out, kv = layer(x.bfloat16())
    assert low.dtype == torch.bfloat16  # the plain layer follows autocast
    assert out.dtype == torch.float32 and kv.dtype == torch.float32
    # The input was rounded to bf16 on the way in; nothing after that is.
    expected, _ = LinearAttention.forward(layer, x.bfloat16().float())
    assert torch.equal(out, expected)
    assert float((out - reference).abs().max()) < 0.05 * float(reference.abs().max())


# 확인: set_attention이 디코더 모든 블록의 어텐션 클래스만 바꾸고 state_dict 키와 파라미터 객체는 그대로 둔다.
# attention_mode는 모드를 읽고, 모르는 모드는 ValueError, 블록마다 모드가 다르면 'different' ValueError다.
# 왜: 클래스만 바꾸므로 같은 체크포인트(model.pt)를 어느 어텐션 모드로도 그대로 읽을 수 있어야 한다. 블록이
# 섞이면 어떤 모드로 학습·평가했는지 기록이 틀어진다.
def test_set_attention_switches_every_decoder_block_and_keeps_the_parameters(no_download):
    model = _small()
    keys = list(model.state_dict())
    params = [id(p) for p in model.parameters()]
    assert attention_mode(model) == "original"
    for mode, cls in (("scaled", ScaledLinearAttention), ("fp32", Fp32LinearAttention)):
        set_attention(model, mode)
        assert attention_mode(model) == mode
        assert all(type(block.attn) is cls for block in model.decoder)
    set_attention(model, "original")
    assert all(type(block.attn) is LinearAttention for block in model.decoder)
    assert list(model.state_dict()) == keys and [id(p) for p in model.parameters()] == params
    with pytest.raises(ValueError):
        set_attention(model, "fast")
    model.decoder[0].attn.__class__ = ScaledLinearAttention
    with pytest.raises(ValueError, match="different"):
        attention_mode(model)


# ---------------------------------------------------------------- the fixed encoder


# 확인: fix_encoder(img_size=56, blocks=10) 뒤에도 인코더 특징, 이미지 점수, 이상 맵이 고정 전과 비트 단위로
# 같다. 블록 수와 위치 임베딩 크기(37×37 → 4×4)가 바뀌고, 다른 크기(70px)를 넣으면 'fixed to 56x56'
# ValueError다.
# 왜: 6단계 규칙은 '읽지 않는 블록 10·11을 자르고 위치 임베딩을 한 번만 보간해도 출력이 같다'를 전제로 CPU
# 지연을 줄였다. 동적 경로가 이미지마다 부르는 같은 보간 함수로 한 번 계산하므로 같아야 하고, 고정한 임베딩을
# 다시 보간하지 않게 다른 크기는 거부한다.
def test_fix_encoder_keeps_the_outputs_and_refuses_other_sizes(no_download):
    model = _small()
    x = to_tensor(_images(2), "cpu", torch.float32)
    with torch.no_grad():
        before = model(x)
        features_before = model.encoder(x)
    vit = model.encoder.feature_extractor
    assert len(vit.blocks) == 12 and vit.pos_embed.shape[1] == 37 * 37

    fix_encoder(model, img_size=SIZE, blocks=10)
    assert len(vit.blocks) == 10 and vit.pos_embed.shape[1] == (SIZE // 14) ** 2
    assert dinomaly_model.encoder_blocks(model) == 10 and model.fixed_img_size == SIZE
    with torch.no_grad():
        after = model(x)
        features_after = model.encoder(x)
    # Blocks 10 and 11 are never read, and the embedding is resampled by the same function: same bits.
    assert list(features_after) == list(features_before)
    assert all(torch.equal(features_after[k], features_before[k]) for k in features_before)
    assert torch.equal(after.pred_score, before.pred_score)
    assert torch.equal(after.anomaly_map, before.anomaly_map)

    with pytest.raises(ValueError, match="fixed to 56x56"):
        model(to_tensor(_images(1, size=70), "cpu", torch.float32))


# 확인: 읽는 마지막 층(블록 9)보다 적은 블록(9개), 원래보다 많은 블록(13개), 14의 배수가 아닌 크기, 0 크기는
# ValueError이고, 오류가 나면 모델이 바뀌지 않는다(블록 12개 그대로).
def test_fix_encoder_rejects_bad_arguments(no_download):
    model = _small()
    for kwargs in ({"blocks": 9}, {"blocks": 13}, {"img_size": 50}, {"img_size": 0}):
        with pytest.raises(ValueError):
            fix_encoder(model, **kwargs)
    assert len(model.encoder.feature_extractor.blocks) == 12


# ---------------------------------------------------------------- the exported graph


# 확인: ONNX로 내보내는 ServingGraph의 (점수, 256×256 맵)이 모델 forward의 pred_score, anomaly_map을 256으로
# 줄인 것과 같고, run_dinomaly.predict가 채점·저장하는 값(맵은 fp16 반올림)과도 같다. CAR(Context-Aware
# Recentering)을 켠 경우와 끈 경우 모두 본다.
# 왜: 서빙 그래프는 상위 1% 평균을 전체 정렬 대신 topk로 계산하는 등 구현이 다르므로, torch에서 잰 dev·test
# 점수와 CPU 서빙 점수가 같은 정의인지 확인해야 한다. 디코더 가중치를 조금 흔들어 초기값(출력이 거의 0)에서만
# 맞는 우연을 피한다.
@pytest.mark.parametrize("recentering", [False, True])
def test_serving_graph_is_the_inference_path_of_the_model(no_download, recentering):
    options = {"img_size": SIZE, "encoder_blocks": 10, "context_recentering": recentering}
    model = _small(**options)
    with torch.no_grad():  # a decoder that is not at its initial weights
        for p in model.decoder.parameters():
            p.add_(torch.randn(p.shape, generator=torch.Generator().manual_seed(4)) * 0.02)
    x = to_tensor(_images(3, seed=5), "cpu", torch.float32)
    with torch.no_grad():
        out = model(x)
        score, amap = ServingGraph(model).eval()(x)
    assert score.shape == (3,) and amap.shape == (3, 256, 256)
    torch.testing.assert_close(score, out.pred_score, rtol=1e-6, atol=0)
    expected = F.interpolate(out.anomaly_map, size=(256, 256), mode="bilinear", align_corners=False)[:, 0]
    assert torch.equal(amap, expected)
    # The same as what run_dinomaly.predict scores and stores (its maps are rounded to fp16).
    result = run_dinomaly.predict(model, _images(3, seed=5), batch_size=2, device="cpu", amp=False)
    np.testing.assert_allclose(result.image_scores, score.numpy(), rtol=1e-6)
    np.testing.assert_array_equal(result.maps, amap.numpy().astype(np.float16))
