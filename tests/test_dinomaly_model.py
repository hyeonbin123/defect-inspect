import copy

import numpy as np
import pytest

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

SIZE = 56  # a 4x4 patch grid: small enough for CPU tests


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


def _small(seed=0, **options):
    torch.manual_seed(seed)
    return run_dinomaly.build_model(run_dinomaly.ENCODER_S, **options).eval()


def _images(n, seed=0, size=SIZE):
    return np.random.default_rng(seed).integers(0, 256, (n, size, size, 3), dtype=np.uint8)


def _attention(seed=0, dim=384, heads=6):
    torch.manual_seed(seed)
    layer = LinearAttention(dim, num_heads=heads, qkv_bias=True)
    with torch.no_grad():
        layer.qkv.weight.mul_(4.0)  # larger activations than a fresh layer, like a trained decoder
    return layer.eval()


# ---------------------------------------------------------------- attention


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


def test_fix_encoder_rejects_bad_arguments(no_download):
    model = _small()
    for kwargs in ({"blocks": 9}, {"blocks": 13}, {"img_size": 50}, {"img_size": 0}):
        with pytest.raises(ValueError):
            fix_encoder(model, **kwargs)
    assert len(model.encoder.feature_extractor.blocks) == 12


# ---------------------------------------------------------------- the exported graph


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
