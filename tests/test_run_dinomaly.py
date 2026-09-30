import contextlib
import copy
import json
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("anomalib")

import torch.nn.functional as F  # noqa: E402
from anomalib.data import InferenceBatch  # noqa: E402
from anomalib.models.image.dinomaly import components  # noqa: E402

from defect_inspect import run_dinomaly  # noqa: E402
from defect_inspect.backbones import to_tensor  # noqa: E402
from defect_inspect.run_dinomaly import batch_order, predict, train  # noqa: E402
from defect_inspect.splits import SEALED_ROLES, ManifestRow, SealedTestError, write_manifest  # noqa: E402

SIZE = 32


class TinyModel(torch.nn.Module):
    """Stand-in with Dinomaly's call convention: a frozen encoder and a trainable decoder.

    Training mode returns a scalar loss and needs `global_step`; eval mode returns an InferenceBatch
    whose map is the mean absolute value of the normalised image (so it depends on the input only).
    """

    def __init__(self, nan_at: int | None = None, map_dims: int = 4) -> None:
        super().__init__()
        self.encoder = torch.nn.Conv2d(3, 4, kernel_size=4, stride=4)
        self.decoder = torch.nn.Conv2d(4, 4, kernel_size=1)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.nan_at = nan_at
        self.map_dims = map_dims
        self.calls: list[tuple] = []

    def forward(self, batch, global_step=None):
        self.calls.append((self.training, global_step, tuple(batch.shape), batch.dtype))
        if self.training:
            if global_step is None:
                raise ValueError("global_step must be provided during training")
            feats = self.encoder(batch)
            loss = 100.0 * (self.decoder(feats) - feats).pow(2).mean()
            return loss * float("nan") if global_step == self.nan_at else loss
        anomaly_map = batch.abs().mean(dim=1, keepdim=True)
        score = anomaly_map.flatten(1).amax(dim=1)
        if self.map_dims == 3:
            anomaly_map = anomaly_map[:, 0]
        return InferenceBatch(pred_score=score, anomaly_map=anomaly_map)


class DropoutModel(TinyModel):
    """TinyModel whose training loss goes through dropout, like the bottleneck of Dinomaly."""

    def __init__(self) -> None:
        super().__init__()
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, batch, global_step=None):
        if not self.training:
            return super().forward(batch, global_step)
        feats = self.encoder(batch)
        return 100.0 * (self.decoder(self.drop(feats)) - feats).pow(2).mean()


def _images(n: int, seed: int = 0, size: int = SIZE) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (n, size, size, 3), dtype=np.uint8)


def _reference_train(model, images, *, steps: int, batch_size: int, seed: int) -> list[float]:
    """The fp32 training of anomalib's Lightning module written out step by step; returns the losses.

    Built from anomalib's TRAINING_CONFIG and components only: it shares no code with
    `run_dinomaly.train` apart from the batch order.
    """
    from anomalib.models.image.dinomaly.lightning_model import TRAINING_CONFIG

    order = batch_order(len(images), steps, batch_size, seed)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = components.StableAdamW([{"params": params}], **TRAINING_CONFIG["optimizer"])
    scheduler = components.WarmCosineScheduler(
        optimizer, **{**TRAINING_CONFIG["scheduler"], "total_iters": steps}
    )
    torch.manual_seed(seed)
    model.train()
    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        loss = model(to_tensor(images[order[step]], "cpu", torch.float32), global_step=step)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, TRAINING_CONFIG["trainer"]["gradient_clip_val"])
        optimizer.step()
        scheduler.step()
        losses.append(float(loss))
    return losses


# ---------------------------------------------------------------- batch_order


def test_batch_order_shape_and_determinism():
    order = batch_order(50, 20, 16, seed=3)
    assert order.shape == (20, 16) and order.dtype == np.int64
    assert order.min() >= 0 and order.max() < 50
    assert np.array_equal(order, batch_order(50, 20, 16, seed=3))
    assert not np.array_equal(order, batch_order(50, 20, 16, seed=4))
    # The first batches do not depend on how many steps follow.
    assert np.array_equal(order[:7], batch_order(50, 7, 16, seed=3))
    assert batch_order(50, 0, 16, seed=0).shape == (0, 16)


@pytest.mark.parametrize(("n", "batch"), [(50, 16), (20, 16), (16, 16), (17, 16), (64, 16), (31, 5), (9, 8)])
def test_batch_order_is_a_stream_of_permutations_without_repeats_in_a_batch(n, batch):
    for seed in range(5):
        steps = 4 * n  # many epoch boundaries inside batches
        order = batch_order(n, steps, batch, seed=seed)
        for row in order:
            assert len(set(row.tolist())) == batch
        stream = order.reshape(-1)
        for epoch in range(len(stream) // n):
            assert np.array_equal(np.sort(stream[epoch * n : (epoch + 1) * n]), np.arange(n))


def test_batch_order_repairs_an_epoch_boundary():
    # 20 images in batches of 16: the second batch takes 4 images of epoch 0 and 12 of epoch 1. A plain
    # concatenation of permutations repeats an image there for most seeds.
    rng = np.random.default_rng(0)
    plain = np.concatenate([rng.permutation(20), rng.permutation(20)])[:32].reshape(2, 16)
    assert len(set(plain[1].tolist())) < 16  # this seed does collide, so the repair is exercised
    order = batch_order(20, 2, 16, seed=0)
    assert np.array_equal(order[0], plain[0])
    assert len(set(order[1].tolist())) == 16


def test_batch_order_with_fewer_images_than_a_batch():
    order = batch_order(5, 6, 8, seed=0)
    assert order.shape == (6, 8)
    stream = order.reshape(-1)
    for epoch in range(len(stream) // 5):
        assert np.array_equal(np.sort(stream[epoch * 5 : (epoch + 1) * 5]), np.arange(5))
    counts = np.bincount(stream, minlength=5)
    assert counts.max() - counts.min() <= 1


def test_batch_order_rejects_bad_arguments():
    for args in [(0, 5, 4), (10, -1, 4), (10, 5, 0)]:
        with pytest.raises(ValueError):
            batch_order(*args, seed=0)


# ---------------------------------------------------------------- train


def test_training_constants_match_anomalib():
    from anomalib.models.image.dinomaly import lightning_model

    config = lightning_model.TRAINING_CONFIG
    assert run_dinomaly.OPTIMIZER == config["optimizer"]
    scheduler = dict(config["scheduler"])
    assert scheduler.pop("total_iters") == run_dinomaly.STEPS == config["trainer"]["max_steps"]
    assert run_dinomaly.SCHEDULER == scheduler
    assert run_dinomaly.GRAD_CLIP == config["trainer"]["gradient_clip_val"]
    assert run_dinomaly.IMG_SIZE == lightning_model.DEFAULT_CROP_SIZE
    assert lightning_model.Dinomaly.__init__.__defaults__[0] == run_dinomaly.ENCODER


def test_train_logs_and_follows_the_schedule(tmp_path):
    torch.manual_seed(0)
    model = TinyModel()
    frozen = model.encoder.weight.clone()
    start = model.decoder.weight.detach().clone()
    images = _images(10)
    log_path = tmp_path / "train_log.jsonl"
    steps = 150
    log = train(
        model, images, steps=steps, batch_size=4, device="cpu", amp=True, log_every=1, log_path=log_path
    )

    assert [e["step"] for e in log] == list(range(1, steps + 1))
    assert all(set(e) == {"step", "loss", "lr", "seconds", "vram_mb", "skipped"} for e in log)
    assert all(np.isfinite(e["loss"]) and e["vram_mb"] == 0.0 and e["skipped"] == 0 for e in log)
    seconds = [e["seconds"] for e in log]
    assert seconds == sorted(seconds)
    # Warm-up (100 steps from 0 to 2e-3), then a cosine from 2e-3 towards 2e-4 over the other 50 steps.
    cosine = 2e-4 + 0.5 * (2e-3 - 2e-4) * (1 + np.cos(np.pi * np.arange(50) / 50))
    expected = np.concatenate([np.linspace(0.0, 2e-3, 100), cosine])
    assert np.allclose([e["lr"] for e in log], expected, rtol=1e-12, atol=0)
    assert log[0]["lr"] == 0.0 and log[99]["lr"] == log[100]["lr"] == 2e-3

    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert lines == log
    assert b"\r" not in log_path.read_bytes()

    # The model saw every step once, in training mode, with normalised fp32 batches in batch_order.
    assert [c[:2] for c in model.calls] == [(True, step) for step in range(steps)]
    assert all(c[2] == (4, 3, SIZE, SIZE) and c[3] == torch.float32 for c in model.calls)
    assert model.training
    assert torch.equal(model.encoder.weight, frozen)
    assert not torch.equal(model.decoder.weight, start)
    assert log[-1]["loss"] < log[0]["loss"]


def test_train_feeds_the_batches_of_batch_order():
    seen = []

    class Recorder(TinyModel):
        def forward(self, batch, global_step=None):
            seen.append(batch.clone())
            return super().forward(batch, global_step)

    images = _images(9, seed=1)
    train(Recorder(), images, steps=5, batch_size=4, device="cpu", seed=7)
    order = batch_order(9, 5, 4, seed=7)
    for step, batch in enumerate(seen):
        assert torch.equal(batch, to_tensor(images[order[step]], "cpu", torch.float32))


def test_train_uses_the_optimiser_scheduler_and_clipping_of_anomalib(monkeypatch):
    events: list[str] = []
    made: dict = {}

    class SpyOptimizer(components.StableAdamW):
        def __init__(self, params, **kwargs):
            params = list(params)
            made["params"], made["optimizer"] = params, kwargs
            super().__init__(params, **kwargs)

        def step(self, closure=None):
            events.append("optimizer")
            return super().step(closure)

    class SpyScheduler(components.WarmCosineScheduler):
        def __init__(self, optimizer, **kwargs):
            made["scheduler"] = kwargs
            super().__init__(optimizer, **kwargs)

        def step(self, *args, **kwargs):
            events.append("scheduler")
            return super().step(*args, **kwargs)

    clips = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def spy_clip(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        before = float(torch.sqrt(sum(p.grad.pow(2).sum() for p in parameters)))
        result = real_clip(parameters, max_norm, *args, **kwargs)
        after = float(torch.sqrt(sum(p.grad.pow(2).sum() for p in parameters)))
        clips.append((max_norm, [id(p) for p in parameters], before, after))
        events.append("clip")
        return result

    monkeypatch.setattr(components, "StableAdamW", SpyOptimizer)
    monkeypatch.setattr(components, "WarmCosineScheduler", SpyScheduler)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy_clip)

    torch.manual_seed(0)
    model = TinyModel()
    steps = 6
    train(model, _images(8), steps=steps, batch_size=4, device="cpu")

    trainable = [id(p) for p in model.decoder.parameters()]
    assert made["optimizer"] == run_dinomaly.OPTIMIZER
    assert [id(p) for group in made["params"] for p in group["params"]] == trainable
    assert made["scheduler"] == {"total_iters": steps, **run_dinomaly.SCHEDULER}
    # Every iteration: clip, then the optimiser, then the scheduler.
    assert [e for e in events if e != "scheduler"] == ["clip", "optimizer"] * steps
    assert events[-3 * steps :] == ["clip", "optimizer", "scheduler"] * steps
    assert len(clips) == steps
    for max_norm, ids, _before, after in clips:
        assert max_norm == 0.1 and ids == trainable
        assert after <= 0.1 * (1 + 1e-5)
    assert max(before for _, _, before, _ in clips) > 0.1  # the clipping really had something to cut


def test_train_is_deterministic_for_a_seed():
    def run(seed, noise):
        torch.manual_seed(0)
        model = DropoutModel()
        torch.manual_seed(noise)  # what the process drew before the call must not matter
        log = train(model, _images(12), steps=8, batch_size=4, device="cpu", seed=seed, log_every=1)
        return [e["loss"] for e in log], model.decoder.weight.detach().clone()

    losses_a, weight_a = run(0, noise=11)
    losses_b, weight_b = run(0, noise=22)
    losses_c, _ = run(1, noise=11)
    assert losses_a == losses_b and torch.equal(weight_a, weight_b)
    assert losses_a != losses_c


def test_train_matches_a_plain_reference_loop():
    # Gradients that pile up over the steps, another dropout stream or another order of clipping and
    # stepping all end in other weights than the loop written out in `_reference_train`.
    images = _images(10, seed=4)
    torch.manual_seed(0)
    ours = DropoutModel()
    reference = copy.deepcopy(ours)
    start = ours.decoder.weight.detach().clone()
    torch.manual_seed(999)  # train seeds the dropout stream itself
    log = train(ours, images, steps=12, batch_size=4, device="cpu", seed=3, log_every=1)
    losses = _reference_train(reference, images, steps=12, batch_size=4, seed=3)

    assert [e["loss"] for e in log] == losses
    assert torch.equal(ours.decoder.weight, reference.decoder.weight)
    assert torch.equal(ours.decoder.bias, reference.decoder.bias)
    assert not torch.equal(ours.decoder.weight, start)


def test_train_clips_after_unscaling_with_a_working_grad_scaler(monkeypatch):
    # The fp16 path multiplies the loss by the scale of a GradScaler (65536 at the start). The clipping
    # has to see the gradients after `unscale_`, or every step is cut down to a norm of 0.1 / 65536.
    real_scaler = torch.amp.GradScaler
    real_clip = torch.nn.utils.clip_grad_norm_
    force = {"on": False}
    made: list[tuple] = []
    events: list[str] = []
    norms: list[float] = []

    class SpyScaler(real_scaler):
        def __init__(self, device="cuda", **kwargs):
            made.append((device, dict(kwargs)))
            if force["on"]:  # a scaler that really scales, on the CPU
                device, kwargs = "cpu", {**kwargs, "enabled": True}
            super().__init__(device, **kwargs)

        def unscale_(self, optimizer):
            events.append("unscale")
            return super().unscale_(optimizer)

        def step(self, optimizer, *args, **kwargs):
            events.append("step")
            return super().step(optimizer, *args, **kwargs)

        def update(self, *args, **kwargs):
            events.append("update")
            return super().update(*args, **kwargs)

    def spy_clip(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        norms.append(float(torch.sqrt(sum(p.grad.pow(2).sum() for p in parameters))))
        events.append("clip")
        return real_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.amp, "GradScaler", SpyScaler)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy_clip)
    steps = 6

    def run(scaling):
        force["on"] = scaling
        made.clear()
        events.clear()
        norms.clear()
        torch.manual_seed(0)
        model = TinyModel()
        log = train(model, _images(8), steps=steps, batch_size=4, device="cpu", amp=True, log_every=1)
        assert events == ["unscale", "clip", "step", "update"] * steps
        assert log[-1]["skipped"] == 0
        return [e["loss"] for e in log], list(norms), model.decoder.weight.detach().clone()

    plain = run(scaling=False)
    assert made == [("cuda", {"enabled": False})]  # on the CPU `amp` does not apply: plain fp32
    scaled = run(scaling=True)
    # Scaling by a power of two and back is exact: the clipping saw the same gradients both times.
    assert max(plain[1]) > 0.1 and scaled[1] == plain[1]
    assert scaled[0] == plain[0] and torch.equal(scaled[2], plain[2])


def test_fp16_autocast_applies_on_cuda_only():
    assert run_dinomaly.amp_applied(True, "cuda") and run_dinomaly.amp_applied(True, "cuda:0")
    assert not run_dinomaly.amp_applied(True, "cpu") and not run_dinomaly.amp_applied(False, "cuda")
    context = run_dinomaly._autocast(torch.device("cuda"), True)
    assert isinstance(context, torch.autocast)
    assert context.device == "cuda" and context.fast_dtype == torch.float16
    for device, amp in (("cpu", True), ("cuda", False), ("cpu", False)):
        assert isinstance(run_dinomaly._autocast(torch.device(device), amp), contextlib.nullcontext)


def test_train_logs_every_n_steps_and_the_last_step_and_appends(tmp_path):
    log_path = tmp_path / "log.jsonl"
    log = train(TinyModel(), _images(8), steps=12, batch_size=4, device="cpu", log_every=5, log_path=log_path)
    assert [e["step"] for e in log] == [5, 10, 12]
    train(TinyModel(), _images(8), steps=3, batch_size=4, device="cpu", log_every=5, log_path=log_path)
    steps = [json.loads(line)["step"] for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert steps == [5, 10, 12, 3]


def test_train_flushes_each_log_line(tmp_path):
    log_path = tmp_path / "log.jsonl"
    visible = []

    class Watcher(TinyModel):
        def forward(self, batch, global_step=None):
            # What another process would see of the log while this step is running.
            text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
            visible.append(len(text.splitlines()))
            return super().forward(batch, global_step)

    train(Watcher(), _images(8), steps=5, batch_size=4, device="cpu", log_every=1, log_path=log_path)
    assert visible == [0, 1, 2, 3, 4]


def test_train_stops_on_a_non_finite_loss(tmp_path):
    log_path = tmp_path / "log.jsonl"
    model = TinyModel(nan_at=3)
    with pytest.raises(FloatingPointError, match=r"step 4 of 10 \(global_step=3\)"):
        train(model, _images(8), steps=10, batch_size=4, device="cpu", log_every=1, log_path=log_path)
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 3
    assert torch.isfinite(model.decoder.weight).all()


def test_train_rejects_bad_input():
    with pytest.raises(ValueError):
        train(TinyModel(), _images(8).astype(np.float32), steps=2, batch_size=4, device="cpu")
    with pytest.raises(ValueError):
        train(TinyModel(), _images(8), steps=0, batch_size=4, device="cpu")
    frozen = TinyModel()
    for param in frozen.parameters():
        param.requires_grad = False
    with pytest.raises(ValueError, match="no trainable"):
        train(frozen, _images(8), steps=2, batch_size=4, device="cpu")


# ---------------------------------------------------------------- predict


def test_predict_shapes_order_and_resizing():
    model = TinyModel().train()
    images = _images(7, seed=2)
    result = predict(model, images, batch_size=3, device="cpu")
    assert result.image_scores.shape == (7,) and result.image_scores.dtype == np.float32
    assert result.maps.shape == (7, 256, 256) and result.maps.dtype == np.float16
    assert not model.training
    assert all(c[0] is False and c[1] is None for c in model.calls)
    assert [c[2][0] for c in model.calls] == [3, 3, 1]

    with torch.no_grad():
        out = model(to_tensor(images, "cpu", torch.float32))
        resized = F.interpolate(out.anomaly_map, size=(256, 256), mode="bilinear", align_corners=False)
    assert np.array_equal(result.image_scores, out.pred_score.numpy())
    assert np.array_equal(result.maps, resized[:, 0].to(torch.float16).numpy())


def test_predict_keeps_a_map_that_already_has_the_size():
    model = TinyModel(map_dims=3)  # maps without the channel axis are accepted too
    images = _images(3, seed=3)
    result = predict(model, images, batch_size=2, device="cpu", map_size=SIZE)
    with torch.no_grad():
        out = model(to_tensor(images, "cpu", torch.float32))
    assert result.maps.shape == (3, SIZE, SIZE)
    assert np.array_equal(result.maps, out.anomaly_map.to(torch.float16).numpy())
    # The brightest or darkest pixel of each image carries the score.
    assert np.allclose(result.maps.reshape(3, -1).max(axis=1), result.image_scores, rtol=1e-3)


def test_predict_empty_and_bad_input():
    empty = predict(TinyModel(), _images(0), device="cpu")
    assert empty.image_scores.shape == (0,) and empty.maps.shape == (0, 256, 256)
    with pytest.raises(ValueError):
        predict(TinyModel(), _images(2)[..., :1], device="cpu")
    with pytest.raises(ValueError):
        predict(TinyModel(), _images(2), batch_size=0, device="cpu")


def test_predict_raises_on_non_finite_scores():
    class Broken(TinyModel):
        def forward(self, batch, global_step=None):
            out = super().forward(batch, global_step)
            return InferenceBatch(pred_score=out.pred_score * float("nan"), anomaly_map=out.anomaly_map)

    with pytest.raises(FloatingPointError):
        predict(Broken(), _images(2), device="cpu")


# ---------------------------------------------------------------- the real model, random weights


def test_real_model_builds_trains_and_predicts_without_pretrained_weights(monkeypatch, tmp_path):
    timm = pytest.importorskip("timm")
    import timm.models.vision_transformer as vit

    real_create = timm.create_model
    requested = []

    def create_without_download(name, *args, **kwargs):
        requested.append((name, kwargs.get("pretrained")))
        kwargs["pretrained"] = False  # tests never download DINOv2 weights
        return real_create(name, *args, **kwargs)

    monkeypatch.setattr(timm, "create_model", create_without_download)
    # anomalib replaces this timm function for the whole process when it builds a ViT encoder;
    # registering it here makes monkeypatch put the original back after the test.
    monkeypatch.setattr(vit, "resample_abs_pos_embed", vit.resample_abs_pos_embed)

    encoder = "vit_small_patch14_reg4_dinov2"
    torch.manual_seed(0)
    model = run_dinomaly.build_model(encoder)
    assert requested == [(encoder, True)]  # the real run asks for the pretrained encoder

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(name.startswith(("bottleneck.", "decoder.")) for name in trainable)
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert all(p.dtype == torch.float32 for p in model.parameters())
    for module in (model.bottleneck, model.decoder):
        assert all(p.requires_grad for p in module.parameters())
        for layer in module.modules():
            if isinstance(layer, torch.nn.Linear):
                weight = layer.weight.detach()
                assert float(weight.abs().max()) <= 0.03
                assert 0.005 < float(weight.std()) < 0.015
                assert layer.bias is None or not layer.bias.any()
            elif isinstance(layer, torch.nn.LayerNorm):
                assert bool((layer.weight == 1).all()) and not layer.bias.any()

    images = _images(8, seed=5, size=56)  # a 4x4 patch grid
    reference = copy.deepcopy(model)
    frozen = {name: p.clone() for name, p in model.encoder.named_parameters()}
    before = run_dinomaly.trainable_state(model)
    log = train(model, images, steps=3, batch_size=4, device="cpu", log_every=1)
    assert len(log) == 3 and all(np.isfinite(e["loss"]) for e in log)
    assert all(torch.equal(p, frozen[name]) for name, p in model.encoder.named_parameters())
    after = run_dinomaly.trainable_state(model)
    assert set(after) == trainable
    assert sum(not torch.equal(after[name], before[name]) for name in after) > len(after) // 2
    # The same steps written out with anomalib's pieces: the same losses and weights, bit for bit.
    assert _reference_train(reference, images, steps=3, batch_size=4, seed=0) == [e["loss"] for e in log]
    expected = run_dinomaly.trainable_state(reference)
    assert all(torch.equal(after[name], expected[name]) for name in after)

    result = predict(model, images, batch_size=5, device="cpu")
    assert result.image_scores.shape == (8,) and result.maps.shape == (8, 256, 256)
    assert np.isfinite(result.image_scores).all() and np.isfinite(result.maps).all()
    assert result.maps.min() > -1e-3 and result.maps.max() < 2 + 1e-3  # 1 - cosine similarity

    # The saved file holds the trainable part only; with the same encoder it reproduces the scores.
    path = tmp_path / "model.pt"
    run_dinomaly.save_model(path, model, encoder=encoder, steps=3, amp=False)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert set(saved) == {"encoder", "steps", "amp", "state"} and set(saved["state"]) == trainable
    torch.manual_seed(0)  # the same random "pretrained" encoder as above
    loaded, meta = run_dinomaly.load_model(path)
    assert meta == {"encoder": encoder, "steps": 3, "amp": False} and not loaded.training
    again = predict(loaded, images, batch_size=5, device="cpu")
    assert np.array_equal(again.image_scores, result.image_scores)
    assert np.array_equal(again.maps, result.maps)

    saved["state"].pop(sorted(trainable)[0])
    torch.save(saved, path)
    with pytest.raises(ValueError, match="trainable parameters"):
        run_dinomaly.load_model(path)


# ---------------------------------------------------------------- evaluation and the CLI


class FakeCache:
    """In-memory stand-in for ImageCache: defects carry a bright square that no normal image has."""

    size = SIZE

    def __init__(self, manifest, ledger=None):
        rng = np.random.default_rng(0)
        self._images, self._masks, self._role = {}, {}, {}
        self._ledger = ledger
        self.requested: list[str] = []
        for row in manifest:
            img = rng.integers(100, 140, (SIZE, SIZE, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            if row.label == "anomaly":
                img[8:16, 8:16] = 255
                mask[64:128, 64:128] = 1
            self._images[row.image] = img
            self._masks[row.image] = mask
            self._role[row.image] = row.role

    def _note(self, rows):
        if self._ledger is not None and any(self._role[r.image] in SEALED_ROLES for r in rows):
            # The read of sealed images must already be on record.
            assert self._ledger.exists() and self._ledger.read_text(encoding="utf-8").strip()
        self.requested += [r.image for r in rows]

    def images(self, rows):
        self._note(rows)
        return np.stack([self._images[r.image] for r in rows])

    def masks(self, rows):
        self._note(rows)
        return np.stack([self._masks[r.image] for r in rows])


def _manifest():
    rows = []
    for i in range(40):
        rows.append(ManifestRow(f"toy/n{i}.JPG", "", "toy", "normal", "pool_normal", i % 5, ""))
    for i in range(6):
        kind = "hole" if i % 2 else "hole|scratch"
        rows.append(ManifestRow(f"toy/d{i}.JPG", f"toy/d{i}.png", "toy", "anomaly", "dev_defect", -1, kind))
    for i in range(10):
        rows.append(ManifestRow(f"toy/tn{i}.JPG", "", "toy", "normal", "test_normal", -1, ""))
    for i in range(4):
        mask = f"toy/td{i}.png"
        rows.append(ManifestRow(f"toy/td{i}.JPG", mask, "toy", "anomaly", "test_defect", -1, "hole"))
    return rows


COMMON_KEYS = {
    "eval_images",
    "eval_labels",
    "eval_defect_types",
    "eval_score",
    "cal_score",
    "pixel_auroc",
    "aupro",
    "pro_edges",
    "pro_normal",
    "pro_components",
    "pro_component_image",
}


def test_eval_category_dev_writes_the_common_format(tmp_path):
    manifest = _manifest()
    cache = FakeCache(manifest)
    info = run_dinomaly.eval_category(
        TinyModel(), "dev", "toy", manifest, cache, "cpu", False, tmp_path, batch_size=5
    )
    assert (info["category"], info["eval_normal"], info["eval_defect"], info["cal"]) == ("toy", 8, 6, 0)
    assert not any(image.startswith(("toy/tn", "toy/td")) for image in cache.requested)

    with np.load(tmp_path / "toy.npz") as z:
        assert COMMON_KEYS <= set(z.files)
        assert z["eval_images"].tolist() == [f"toy/n{i}.JPG" for i in range(0, 40, 5)] + [
            f"toy/d{i}.JPG" for i in range(6)
        ]
        assert z["eval_labels"].dtype == np.int8 and z["eval_labels"].tolist() == [0] * 8 + [1] * 6
        assert z["eval_defect_types"].tolist() == [""] * 8 + ["hole|scratch", "hole"] * 3
        assert z["eval_score"].dtype == np.float32 and z["eval_score"].shape == (14,)
        assert z["cal_score"].dtype == np.float32 and z["cal_score"].shape == (0,)
        assert z["cal_images"].shape == (0,)
        # The stand-in scores by brightness: every defect is above every normal, also per pixel.
        assert z["eval_score"][8:].min() > z["eval_score"][:8].max()
        assert z["pixel_auroc"].dtype == np.float64 and float(z["pixel_auroc"]) > 0.95
        assert z["aupro"].dtype == np.float64 and float(z["aupro"]) > 0.8
        assert z["pro_edges"].shape == (2001,) and z["pro_normal"].shape == (14, 2000)
        assert z["pro_components"].shape == (6, 2000)
        assert z["pro_component_image"].tolist() == list(range(8, 14))
    maps = np.load(tmp_path / "toy_maps.npy")
    assert maps.shape == (14, 256, 256) and maps.dtype == np.float16
    assert maps[8:, 64:128, 64:128].min() > maps[:8].max()


def test_eval_category_test_protocol_needs_permission(tmp_path):
    manifest = _manifest()
    cache = FakeCache(manifest)
    with pytest.raises(SealedTestError):
        run_dinomaly.eval_category(TinyModel(), "test", "toy", manifest, cache, "cpu", False, tmp_path)
    assert cache.requested == [] and not (tmp_path / "toy.npz").exists()

    info = run_dinomaly.eval_category(TinyModel(), "test", "toy", manifest, cache, "cpu", True, tmp_path)
    assert (info["eval_normal"], info["eval_defect"], info["cal"]) == (10, 4, 8)
    with np.load(tmp_path / "toy.npz") as z:
        assert z["eval_images"].tolist() == [f"toy/tn{i}.JPG" for i in range(10)] + [
            f"toy/td{i}.JPG" for i in range(4)
        ]
        # Thresholds come from the fold-0 normals, which are not part of the training pool.
        assert z["cal_images"].tolist() == [f"toy/n{i}.JPG" for i in range(0, 40, 5)]
        assert z["cal_score"].shape == (8,) and z["cal_score"].dtype == np.float32
        assert z["eval_score"][10:].min() > max(z["eval_score"][:10].max(), z["cal_score"].max())


def test_cli_refuses_test_protocol_without_flags(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(run_dinomaly.paths, "TEST_LEDGER", ledger)
    monkeypatch.setattr(run_dinomaly, "ImageCache", None)  # nothing may be opened before the refusal
    base = ["eval", "--protocol", "test", "--out", str(tmp_path / "out")]
    for extra in ([], ["--allow-test"], ["--stage", "2"]):
        with pytest.raises(SystemExit):
            run_dinomaly.main(base + extra)
    assert not ledger.exists() and not (tmp_path / "out").exists()
    with pytest.raises(SystemExit):
        run_dinomaly.main(["eval", "--out", str(tmp_path / "out")])  # the protocol must be named


@pytest.fixture
def toy_project(tmp_path, monkeypatch):
    """The CLI wired to a toy manifest, a fake cache, the stand-in model and a ledger under tmp_path."""
    manifest = _manifest()
    ledger = tmp_path / "ledger.jsonl"
    cache = FakeCache(manifest, ledger=ledger)
    built = []
    initial = []

    def fake_cache(root, size):
        assert size == run_dinomaly.IMG_SIZE == 392
        return cache

    def fake_build(encoder_name=run_dinomaly.ENCODER):
        built.append((encoder_name, TinyModel()))
        initial.append(built[-1][1].decoder.weight.detach().clone())
        return built[-1][1]

    write_manifest(manifest, tmp_path / "visa.csv")
    monkeypatch.setattr(run_dinomaly.paths, "VISA_MANIFEST", tmp_path / "visa.csv")
    monkeypatch.setattr(run_dinomaly.paths, "TEST_LEDGER", ledger)
    monkeypatch.setattr(run_dinomaly.paths, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(run_dinomaly, "ImageCache", fake_cache)
    monkeypatch.setattr(run_dinomaly, "build_model", fake_build)
    monkeypatch.setattr(run_dinomaly, "git_commit", lambda: "abc1234")
    return {
        "cache": cache,
        "built": built,
        "initial": initial,
        "ledger": ledger,
        "outputs": tmp_path / "outputs",
    }


TRAIN_ARGS = ["train", "--steps", "6", "--batch-size", "4", "--log-every", "2", "--device", "cpu"]
TOY = ["--categories", "toy"]


def test_cli_train_then_eval_dev(toy_project, capsys):
    outputs = toy_project["outputs"]
    torch.manual_seed(123)
    run_dinomaly.main(TRAIN_ARGS + TOY)

    # Trained on the dev normal pool (folds 1-4) only: no fold-0 normal, no defect, no test image.
    pool = [f"toy/n{i}.JPG" for i in range(40) if i % 5 != 0]
    assert toy_project["cache"].requested == pool
    train_dir = outputs / "dm"
    info = json.loads((train_dir / "train.json").read_text(encoding="utf-8"))
    assert info["method"] == "dinomaly" and info["commit"] == "abc1234" and info["device"] == "cpu"
    assert info["config"]["encoder"] == run_dinomaly.ENCODER and info["config"]["steps"] == 6
    # `--no-amp` was not given, but fp16 autocast only exists on CUDA: the record says what really ran.
    assert info["config"]["batch_size"] == 4 and info["config"]["amp"] is False
    assert info["config"]["img_size"] == 392 and info["config"]["grad_clip_norm"] == 0.1
    assert info["train_images"] == 32 and info["train_images_by_category"] == {"toy": 32}
    log = [json.loads(line) for line in (train_dir / "train_log.jsonl").read_text("utf-8").splitlines()]
    assert [e["step"] for e in log] == [2, 4, 6] and info["final_loss"] == log[-1]["loss"]
    saved = torch.load(train_dir / "model.pt", map_location="cpu", weights_only=True)
    assert saved["encoder"] == run_dinomaly.ENCODER and saved["steps"] == 6 and saved["amp"] is False
    assert set(saved["state"]) == {"decoder.weight", "decoder.bias"}
    trained = toy_project["built"][0][1]
    assert torch.equal(saved["state"]["decoder.weight"], trained.decoder.weight.detach())
    assert not list(train_dir.glob("*.part"))

    # A finished training run is not overwritten by accident.
    with pytest.raises(SystemExit):
        run_dinomaly.main(TRAIN_ARGS + TOY)
    assert len(toy_project["built"]) == 1

    capsys.readouterr()
    run_dinomaly.main(["eval", "--protocol", "dev", "--device", "cpu", "--batch-size", "5"] + TOY)
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [p["category"] for p in printed] == ["toy"]
    eval_dir = outputs / "dm-dev"
    run = json.loads((eval_dir / "run.json").read_text(encoding="utf-8"))
    assert set(run) == {"method", "protocol", "commit", "device", "config", "categories", "total_s"}
    assert (run["method"], run["protocol"], run["commit"], run["device"]) == (
        "dinomaly",
        "dev",
        "abc1234",
        "cpu",
    )
    assert run["config"]["encoder"] == run_dinomaly.ENCODER and run["config"]["train_steps"] == 6
    assert run["config"]["model"] == str(train_dir / "model.pt") and len(run["config"]["model_sha256"]) == 64
    assert run["config"]["amp"] is False and run["config"]["train_amp"] is False
    assert run["config"]["score_precision"] == "fp32"
    assert run["categories"][0]["category"] == "toy" and run["categories"][0]["eval_defect"] == 6
    # The evaluation used the saved weights, not the fresh ones of the rebuilt model.
    name, evaluated = toy_project["built"][1]
    assert name == run_dinomaly.ENCODER
    assert torch.equal(evaluated.decoder.weight.detach(), saved["state"]["decoder.weight"])
    with np.load(eval_dir / "toy.npz") as z:
        assert COMMON_KEYS <= set(z.files)
        assert z["cal_score"].shape == (0,) and z["eval_score"].shape == (14,)
    assert np.load(eval_dir / "toy_maps.npy").shape == (14, 256, 256)
    assert not toy_project["ledger"].exists()

    with pytest.raises(SystemExit):
        run_dinomaly.main(["eval", "--protocol", "dev", "--device", "cpu"] + TOY)
    run_dinomaly.main(["eval", "--protocol", "dev", "--device", "cpu", "--overwrite"] + TOY)


def test_cli_eval_test_protocol_records_the_read_first(toy_project):
    outputs = toy_project["outputs"]
    run_dinomaly.main(TRAIN_ARGS + TOY + ["--out", str(outputs / "m")])
    model = ["--model", str(outputs / "m" / "model.pt")]
    common = ["eval", "--protocol", "test", "--device", "cpu"] + TOY + model

    # A missing model file stops the run before anything is recorded or read.
    toy_project["cache"].requested.clear()
    with pytest.raises(SystemExit):
        run_dinomaly.main(common[:-1] + [str(outputs / "nowhere.pt"), "--allow-test", "--stage", "2"])
    assert not toy_project["ledger"].exists() and toy_project["cache"].requested == []

    # FakeCache asserts that the ledger line exists before it hands out a sealed image.
    run_dinomaly.main(common + ["--allow-test", "--stage", "2", "--note", "toy"])
    lines = toy_project["ledger"].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert (entry["stage"], entry["config"], entry["commit"], entry["note"]) == ("2", "dm", "abc1234", "toy")
    run = json.loads((outputs / "dm-test" / "run.json").read_text(encoding="utf-8"))
    assert run["protocol"] == "test" and run["categories"][0]["cal"] == 8
    with np.load(outputs / "dm-test" / "toy.npz") as z:
        assert z["eval_labels"].tolist() == [0] * 10 + [1] * 4
        assert z["cal_score"].shape == (8,)


def test_cli_rejects_unknown_categories(toy_project):
    with pytest.raises(SystemExit):
        run_dinomaly.main(TRAIN_ARGS + ["--categories", "nothing"])
    assert toy_project["built"] == [] and not (toy_project["outputs"] / "dm").exists()


def test_cli_rejects_repeated_categories(toy_project):
    outputs = toy_project["outputs"]
    with pytest.raises(SystemExit):
        run_dinomaly.main(TRAIN_ARGS + ["--categories", "toy", "toy"])  # would train on every image twice
    assert toy_project["built"] == [] and toy_project["cache"].requested == []
    assert not (outputs / "dm").exists()

    run_dinomaly.main(TRAIN_ARGS + TOY)
    toy_project["cache"].requested.clear()
    with pytest.raises(SystemExit):
        run_dinomaly.main(["eval", "--protocol", "dev", "--device", "cpu", "--categories", "toy", "toy"])
    assert toy_project["cache"].requested == [] and not (outputs / "dm-dev").exists()


def test_cli_train_seed_fixes_the_initial_weights_and_the_result(toy_project):
    outputs = toy_project["outputs"]
    states = {}
    for name, seed, noise in (("a", 0, 11), ("b", 0, 22), ("c", 1, 11)):
        torch.manual_seed(noise)  # what the process drew before the run must not matter
        run_dinomaly.main(TRAIN_ARGS + TOY + ["--seed", str(seed), "--out", str(outputs / name)])
        saved = torch.load(outputs / name / "model.pt", map_location="cpu", weights_only=True)
        states[name] = saved["state"]
        info = json.loads((outputs / name / "train.json").read_text(encoding="utf-8"))
        assert info["config"]["seed"] == seed
    init_a, init_b, init_c = toy_project["initial"]
    assert torch.equal(init_a, init_b) and not torch.equal(init_a, init_c)
    assert all(torch.equal(states["a"][key], states["b"][key]) for key in states["a"])
    assert not torch.equal(states["a"]["decoder.weight"], states["c"]["decoder.weight"])


def test_cli_eval_overwrite_clears_the_results_of_the_earlier_run(toy_project, monkeypatch):
    outputs = toy_project["outputs"]
    run_dinomaly.main(TRAIN_ARGS + TOY)
    base = ["eval", "--protocol", "dev", "--device", "cpu"] + TOY
    eval_dir = outputs / "dm-dev"
    run_dinomaly.main(base)
    # As if the earlier run had covered one more category: a reader that lists *.npz would mix the runs.
    shutil.copy(eval_dir / "toy.npz", eval_dir / "gone.npz")
    shutil.copy(eval_dir / "toy_maps.npy", eval_dir / "gone_maps.npy")
    (eval_dir / "notes.txt").write_text("not a result file", encoding="utf-8")
    run_dinomaly.main(base + ["--overwrite"])
    assert sorted(p.name for p in eval_dir.iterdir()) == ["notes.txt", "run.json", "toy.npz", "toy_maps.npy"]

    # Result files without run.json (a run that broke off) are not mixed into a new run either, and
    # nothing is deleted without --overwrite.
    partial = outputs / "partial"
    partial.mkdir()
    shutil.copy(eval_dir / "toy.npz", partial / "gone.npz")
    toy_project["cache"].requested.clear()
    with pytest.raises(SystemExit):
        run_dinomaly.main(base + ["--out", str(partial)])
    assert [p.name for p in partial.iterdir()] == ["gone.npz"] and toy_project["cache"].requested == []
    run_dinomaly.main(base + ["--out", str(partial), "--overwrite"])
    assert sorted(p.name for p in partial.iterdir()) == ["run.json", "toy.npz", "toy_maps.npy"]

    # A redo that breaks half-way must not look like the finished run it was replacing.
    def broken(*args, **kwargs):
        raise RuntimeError("stopped")

    monkeypatch.setattr(run_dinomaly, "eval_category", broken)
    with pytest.raises(RuntimeError, match="stopped"):
        run_dinomaly.main(base + ["--overwrite"])
    assert [p.name for p in eval_dir.iterdir()] == ["notes.txt"]


def test_cli_eval_precision_follows_the_training_run(toy_project, monkeypatch):
    outputs = toy_project["outputs"]
    run_dinomaly.main(TRAIN_ARGS + TOY)
    saved = torch.load(outputs / "dm" / "model.pt", map_location="cpu", weights_only=True)
    asked = []
    real_predict = run_dinomaly.predict

    def spy(model, images, **kwargs):
        asked.append(kwargs["amp"])
        return real_predict(model, images, **kwargs)

    monkeypatch.setattr(run_dinomaly, "predict", spy)

    def evaluate(train_amp, *flags):
        """Score with a model file that says it was trained with `train_amp` (None: it says nothing)."""
        asked.clear()
        path = outputs / "model-under-test.pt"
        content = {key: value for key, value in saved.items() if key != "amp"}
        if train_amp is not None:
            content["amp"] = train_amp
        torch.save(content, path)
        out = outputs / "precision"
        command = ["eval", "--protocol", "dev", "--device", "cpu", "--overwrite"] + TOY
        run_dinomaly.main(command + ["--model", str(path), "--out", str(out), *flags])
        config = json.loads((out / "run.json").read_text(encoding="utf-8"))["config"]
        assert len(set(asked)) == 1
        return asked[0], config

    asked_amp, config = evaluate(False)
    assert asked_amp is False and config["train_amp"] is False
    assert config["amp"] is False and config["score_precision"] == "fp32"
    # A model trained in fp16 is scored in fp16 unless told otherwise. The CPU cannot meet that
    # request, and run.json records what really ran.
    asked_amp, config = evaluate(True)
    assert asked_amp is True and config["train_amp"] is True
    assert config["amp"] is False and config["score_precision"] == "fp32"
    assert evaluate(True, "--no-amp")[0] is False
    assert evaluate(False, "--amp")[0] is True
    # A file that does not say how it was trained: the precision has to be named.
    with pytest.raises(SystemExit):
        evaluate(None)
    assert asked == []
    asked_amp, config = evaluate(None, "--no-amp")
    assert asked_amp is False and config["train_amp"] is None
    assert evaluate(None, "--amp")[0] is True
