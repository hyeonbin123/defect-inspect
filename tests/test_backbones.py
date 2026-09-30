import numpy as np
import pytest

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


@pytest.fixture(scope="module")
def wrn():
    torch.manual_seed(0)
    return WideResNetPatches(pretrained=False)


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


def test_to_tensor_accepts_non_contiguous_input():
    rng = np.random.default_rng(1)
    images = rng.integers(0, 256, size=(4, 6, 6, 3), dtype=np.uint8)
    np.testing.assert_array_equal(
        to_tensor(images[::2], "cpu", torch.float32).numpy(),
        to_tensor(images[::2].copy(), "cpu", torch.float32).numpy(),
    )


def test_wrn_output_shape_and_attributes(wrn):
    assert wrn.name == "wrn50"
    assert wrn.dim == 1536
    x = torch.randn(2, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        out = wrn(x)
    assert out.shape == (2, 8, 8, 1536)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


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


def test_wrn_is_deterministic_in_eval_mode(wrn):
    x = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        a, b = wrn(x), wrn(x)
        single = wrn(x[:1])
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    # Eval-mode BatchNorm: an image's features do not depend on the rest of the batch.
    torch.testing.assert_close(a[:1], single, rtol=1e-4, atol=1e-5)


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


def test_dinov2_rejects_sizes_that_are_not_multiples_of_14():
    with pytest.raises(ValueError):
        DinoV2Patches(img_size=30, pretrained=False)
    with pytest.raises(ValueError):
        make_extractor("dinov2_vits14", img_size=100, pretrained=False)


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
