"""Dinomaly with anomalib's model code and a plain training loop: one model for all categories.

`train` fits the bottleneck and the decoder on the normal pool of the dev protocol (folds 1-4) and writes
`model.pt`, `train_log.jsonl` and `train.json`. `eval` scores the evaluation set of a protocol with that
model and writes the common run format: `run.json`, `<category>.npz` and `<category>_maps.npy`. It scores
in the precision the model was trained in (fp16 autocast or fp32) unless `--amp` / `--no-amp` says
otherwise, and records the precision that really ran in `run.json`.

`--config` picks a registered configuration: `dm` (stage 2, the default) or one of the ViT-S models of
stage 6 (`dms-<size>`, `dms-<size>-car`). `parity` and `gate` are the two checks of the stage 6 fp16
gate: the scaled attention against the original in fp32 (any device), and fp16 training against fp32
for 60 steps (CUDA only).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import paths
from .cache import ImageCache
from .download import sha256_file
from .ledger import git_commit, record_test_access
from .metrics import aupro, pixel_auroc, pro_histograms
from .splits import ManifestRow, read_manifest, select
from .visa import CATEGORIES

if TYPE_CHECKING:
    import torch

    from .patchcore import ScoreResult

ENCODER = "vit_base_patch14_reg4_dinov2"
ENCODER_S = "vit_small_patch14_reg4_dinov2"
IMG_SIZE = 392
STEPS = 5000
BATCH = 16
# The values of anomalib's TRAINING_CONFIG (dinomaly/lightning_model.py); a test keeps them in step.
OPTIMIZER = {"lr": 2e-3, "betas": (0.9, 0.999), "weight_decay": 1e-4, "amsgrad": True, "eps": 1e-8}
SCHEDULER = {"base_value": 2e-3, "final_value": 2e-4, "warmup_iters": 100}
GRAD_CLIP = 0.1
PRO_BINS = 2000
NAME = "dm"  # method name in docs/experiments.md; also the config label in the test ledger
ATTENTION_MODES = ("original", "scaled", "fp32")  # see dinomaly_model.py
# Stage 6 fp16 gate (docs/experiments.md): the scaled attention must match the original in fp32 within
# PARITY_TOL; fp16 training must stay finite, skip no step and keep every GATE_WINDOW-step mean loss
# within GATE_RTOL of the fp32 run.
PARITY_TOL = 1e-5
GATE_STEPS = 60
GATE_WINDOW = 10
GATE_RTOL = 0.03


@dataclass(frozen=True)
class DinomalyConfig:
    """A registered Dinomaly configuration (docs/experiments.md: `dm` in stage 2, `dms-*` in stage 6)."""

    name: str
    encoder: str
    img_size: int
    steps: int
    batch_size: int = BATCH
    encoder_blocks: int | None = None  # keep the first blocks of the encoder only (None: all)
    fixed_size: bool = False  # resample the position embedding once, for img_size
    context_recentering: bool = False  # Dinomaly2's Context-Aware Recentering (CAR)
    attention: str = "original"  # decoder attention unless the command line or the gate says otherwise

    def build_options(self, attention: str | None = None) -> dict:
        """Keyword arguments of `build_model` beyond the encoder name (empty for `dm`)."""
        options: dict = {}
        if self.fixed_size:
            options["img_size"] = self.img_size
        if self.encoder_blocks is not None:
            options["encoder_blocks"] = self.encoder_blocks
        if self.context_recentering:
            options["context_recentering"] = True
        mode = attention or self.attention
        if mode not in ATTENTION_MODES:
            raise ValueError(f"attention must be one of {ATTENTION_MODES}, got {mode!r}")
        if mode != "original":
            options["attention"] = mode
        return options


def _dms(size: int, car: bool = False) -> DinomalyConfig:
    return DinomalyConfig(
        name=f"dms-{size}" + ("-car" if car else ""),
        encoder=ENCODER_S,
        img_size=size,
        steps=10_000,
        encoder_blocks=10,
        fixed_size=True,
        context_recentering=car,
        attention="scaled",
    )


CONFIGS: dict[str, DinomalyConfig] = {
    c.name: c
    for c in (
        DinomalyConfig(NAME, ENCODER, IMG_SIZE, STEPS),
        _dms(252),
        _dms(280),
        _dms(294),
        _dms(308),
        _dms(252, car=True),
        _dms(280, car=True),
    )
}


def _keep_out(perm: np.ndarray, blocked: np.ndarray, head: int) -> None:
    """Swap every member of `blocked` out of `perm[:head]` (in place) with the first later free entries."""
    is_blocked = np.isin(perm, blocked)
    clash = np.flatnonzero(is_blocked[:head])
    free = np.flatnonzero(~is_blocked[head:])[: len(clash)] + head
    perm[clash], perm[free] = perm[free], perm[clash]


def batch_order(n_images: int, steps: int, batch_size: int, seed: int) -> np.ndarray:
    """Image indices of every training batch, int64 [steps, batch_size].

    Epochs of seeded permutations of all images are concatenated and cut into batches. Where a batch
    spans two epochs, images that the old epoch ended with are swapped further back in the new one, so
    that a batch never holds the same image twice (unless there are fewer images than a batch holds).
    The first rows do not depend on `steps`.
    """
    if n_images < 1 or batch_size < 1 or steps < 0:
        raise ValueError(
            f"need n_images >= 1, batch_size >= 1, steps >= 0; got {n_images}, {batch_size}, {steps}"
        )
    rng = np.random.default_rng(seed)
    total = steps * batch_size
    stream = np.empty(total, dtype=np.int64)
    pos = 0
    while pos < total:
        perm = rng.permutation(n_images)
        carried = pos % batch_size  # images of the previous epoch that are already in the open batch
        if carried and n_images >= batch_size:
            _keep_out(perm, stream[pos - carried : pos], batch_size - carried)
        take = min(n_images, total - pos)
        stream[pos : pos + take] = perm[:take]
        pos += take
    return stream.reshape(steps, batch_size)


def build_model(
    encoder_name: str = ENCODER,
    *,
    img_size: int | None = None,
    encoder_blocks: int | None = None,
    context_recentering: bool = False,
    attention: str = "original",
) -> torch.nn.Module:
    """anomalib's DinomalyModel, set up as its Lightning module does: fp32, frozen encoder, fresh decoder.

    Only the bottleneck and the decoder train. Their Linear layers start from a truncated normal (std
    0.01, cut at +-0.03) with zero bias, their LayerNorms from weight 1 and bias 0. These weights come
    from the global torch generator: call `torch.manual_seed` first to fix them.

    The keyword options are those of the stage 6 configurations (`DinomalyConfig.build_options`): a
    fixed input size and fewer encoder blocks (`dinomaly_model.fix_encoder`), Context-Aware Recentering,
    and the decoder attention mode. None of them draws from the torch generator.
    """
    import torch
    from anomalib.models.image.dinomaly.torch_model import DinomalyModel

    model = DinomalyModel(encoder_name=encoder_name, use_context_recentering=context_recentering).float()
    if img_size is not None or encoder_blocks is not None or attention != "original":
        from .dinomaly_model import fix_encoder, set_attention

        if img_size is not None or encoder_blocks is not None:
            fix_encoder(model, img_size=img_size, blocks=encoder_blocks)
        set_attention(model, attention)
    for param in model.parameters():
        param.requires_grad = False
    for module in (model.bottleneck, model.decoder):
        for param in module.parameters():
            param.requires_grad = True
        for layer in module.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.trunc_normal_(layer.weight, std=0.01, a=-0.03, b=0.03)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, torch.nn.LayerNorm):
                torch.nn.init.constant_(layer.bias, 0)
                torch.nn.init.constant_(layer.weight, 1.0)
    return model


def _check_images(images: np.ndarray, batch_size: int) -> None:
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError(f"expected uint8 images [N, S, S, 3], got {images.dtype} {images.shape}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")


def amp_applied(amp: bool, device: str | torch.device) -> bool:
    """Whether `amp` really runs fp16 on `device`: autocast is only used on CUDA, the CPU stays fp32."""
    import torch

    return bool(amp) and torch.device(device).type == "cuda"


def _autocast(device: torch.device, amp: bool) -> contextlib.AbstractContextManager:
    """fp16 autocast on CUDA when `amp` (this GPU has no bf16), plain fp32 otherwise."""
    import torch

    if amp_applied(amp, device):
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def train(
    model: torch.nn.Module,
    images: np.ndarray,
    *,
    steps: int = STEPS,
    batch_size: int = BATCH,
    device: str = "cuda",
    amp: bool = True,
    seed: int = 0,
    log_every: int = 50,
    log_path: Path | None = None,
) -> list[dict]:
    """Train the parameters of `model` that require gradients on uint8 images [N, S, S, 3].

    Follows anomalib's Lightning module: loss = `model(batch, global_step=step)` in training mode,
    StableAdamW, warm-up + cosine schedule stepped every iteration, gradient norm clipped to 0.1. `amp`
    (CUDA only) runs the forward pass under fp16 autocast with a GradScaler. A non-finite loss raises
    FloatingPointError. Returns one entry per `log_every` steps (and for the last step) and, with
    `log_path`, appends the same entries as JSON lines, flushed one by one.
    """
    import torch
    from anomalib.models.image.dinomaly.components import StableAdamW, WarmCosineScheduler

    from .backbones import to_tensor

    _check_images(images, batch_size)
    if steps < 1 or log_every < 1:
        raise ValueError(f"steps and log_every must be at least 1, got {steps} and {log_every}")
    dev = torch.device(device)
    use_amp = amp_applied(amp, dev)
    order = batch_order(len(images), steps, batch_size, seed)

    torch.manual_seed(seed)  # dropout masks of the bottleneck
    model = model.to(dev).train()
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("the model has no trainable parameters")
    optimizer = StableAdamW([{"params": params}], **OPTIMIZER)
    scheduler = WarmCosineScheduler(optimizer, total_iters=steps, **SCHEDULER)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    log: list[dict] = []
    skipped = 0
    with contextlib.ExitStack() as stack:
        handle = None
        if log_path is not None:
            handle = stack.enter_context(open(log_path, "a", encoding="utf-8", newline="\n"))
        started = time.perf_counter()
        for step in range(steps):
            batch = to_tensor(images[order[step]], dev, torch.float32)
            with _autocast(dev, use_amp):
                loss = model(batch, global_step=step)
            value = float(loss.detach())
            if not math.isfinite(value):
                raise FloatingPointError(
                    f"non-finite loss ({value}) at step {step + 1} of {steps} (global_step={step})"
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # The scaler leaves the weights alone and lowers its scale when a gradient is not finite.
            skipped += int(scaler.get_scale() < scale)
            lr = float(optimizer.param_groups[0]["lr"])
            scheduler.step()

            if (step + 1) % log_every == 0 or step + 1 == steps:
                vram_mb = 0.0
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                    vram_mb = torch.cuda.max_memory_allocated(dev) / 2**20
                entry = {
                    "step": step + 1,
                    "loss": value,
                    "lr": lr,
                    "seconds": round(time.perf_counter() - started, 3),
                    "vram_mb": round(vram_mb, 1),
                    "skipped": skipped,
                }
                log.append(entry)
                if handle is not None:
                    handle.write(json.dumps(entry) + "\n")
                    handle.flush()
    return log


def predict(
    model: torch.nn.Module,
    images: np.ndarray,
    *,
    batch_size: int = 32,
    device: str = "cuda",
    amp: bool = True,
    map_size: int = 256,
) -> ScoreResult:
    """Score uint8 images [N, S, S, 3] with the model in eval mode.

    Image score = `pred_score` of the model's inference output; map = its `anomaly_map`, resized
    bilinearly to `map_size` when it has another size. Non-finite scores or maps raise FloatingPointError.

    With `amp` on CUDA the model smooths and averages its score map in fp16, so the image scores come
    out rounded to fp16 (steps of 1.2e-4 between 0.125 and 0.25) and many images tie. They are stored
    as float32 either way; `amp=False` gives scores with fp32 precision.
    """
    import torch
    import torch.nn.functional as F

    from .backbones import to_tensor
    from .patchcore import ScoreResult

    _check_images(images, batch_size)
    dev = torch.device(device)
    n = len(images)
    image_scores = np.empty(n, dtype=np.float32)
    maps = np.empty((n, map_size, map_size), dtype=np.float16)
    if n == 0:
        return ScoreResult(image_scores, maps)

    model = model.to(dev).eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = to_tensor(images[start : start + batch_size], dev, torch.float32)
            b = batch.shape[0]
            with _autocast(dev, amp):
                out = model(batch)
            scores = out.pred_score.float().reshape(-1)
            batch_maps = out.anomaly_map.float()
            if batch_maps.ndim == 3:
                batch_maps = batch_maps.unsqueeze(1)
            if scores.shape[0] != b or batch_maps.ndim != 4 or batch_maps.shape[:2] != (b, 1):
                raise ValueError(
                    f"expected {b} scores and maps [{b}, 1, H, W], got {tuple(out.pred_score.shape)} "
                    f"and {tuple(out.anomaly_map.shape)}"
                )
            if tuple(batch_maps.shape[-2:]) != (map_size, map_size):
                batch_maps = F.interpolate(
                    batch_maps, size=(map_size, map_size), mode="bilinear", align_corners=False
                )
            batch_maps = batch_maps[:, 0].to(torch.float16)
            if not bool(torch.isfinite(scores).all()) or not bool(torch.isfinite(batch_maps).all()):
                raise FloatingPointError(f"non-finite scores or maps in images {start}..{start + b - 1}")
            image_scores[start : start + b] = scores.cpu().numpy()
            maps[start : start + b] = batch_maps.cpu().numpy()
    return ScoreResult(image_scores, maps)


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """CPU copies of the parameters that train (for Dinomaly: the bottleneck and the decoder), by name."""
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def save_model(
    path: Path,
    model: torch.nn.Module,
    *,
    encoder: str,
    steps: int,
    amp: bool,
    config: str | None = None,
    options: dict | None = None,
) -> None:
    """Write the trainable state with the encoder name and step count (the frozen encoder is not stored).

    `amp` is the precision the training really ran in (True: fp16 autocast); `eval` scores in the same
    precision unless told otherwise. `config` (a CONFIGS name) and `options` (the `build_model` keyword
    arguments) are stored when given, so that `load_model` rebuilds the same model.
    """
    import torch

    path = Path(path)
    part = path.with_name(path.name + ".part")
    saved = {"encoder": encoder, "steps": int(steps), "amp": bool(amp), "state": trainable_state(model)}
    if config is not None:
        saved["config"] = str(config)
        saved["options"] = dict(options or {})
    torch.save(saved, part)
    os.replace(part, path)


def load_model(path: Path) -> tuple[torch.nn.Module, dict]:
    """Rebuild the model of a `save_model` file on the CPU: (model in eval mode, {"encoder", "steps", "amp"}).

    "amp" is None for a file that does not record the precision it was trained in. Files that record a
    configuration add "config" and "options" to the dictionary; the model is built with those options.
    """
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=True)
    options = dict(saved.get("options") or {})
    model = build_model(saved["encoder"], **options)
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(saved["state"]) != expected:
        odd = sorted(set(saved["state"]) ^ expected)
        raise ValueError(f"{path} does not hold the trainable parameters of {saved['encoder']}: {odd[:3]}")
    model.load_state_dict(saved["state"], strict=False)
    amp = saved.get("amp")
    meta = {
        "encoder": saved["encoder"],
        "steps": int(saved["steps"]),
        "amp": None if amp is None else bool(amp),
    }
    if "config" in saved:
        meta["config"] = saved["config"]
        meta["options"] = options
    return model.eval(), meta


# ---------------------------------------------------------------- the stage 6 fp16 gate


def attention_parity(
    build: Callable[[str], torch.nn.Module],
    images: np.ndarray,
    *,
    steps: int = 3,
    batch_size: int = 8,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Gate 2a: the scaled attention against the original, both fp32, from the same initial weights.

    `build(attention)` must return a fresh model; it is called right after `torch.manual_seed(seed)`.
    Compares the eval-mode scores and maps on the first batch of `batch_order`, then the losses of
    `steps` training steps (fp32, the batches of `batch_order`). Passes when every relative loss and
    score difference and every absolute map difference is at most PARITY_TOL.
    """
    import torch

    from .backbones import to_tensor

    _check_images(images, batch_size)
    torch.manual_seed(seed)
    reference = build("original")
    torch.manual_seed(seed)
    patched = build("scaled")
    first = images[batch_order(len(images), 1, batch_size, seed)[0]]
    x = to_tensor(first, device, torch.float32)
    with torch.no_grad():
        a = reference.to(device).eval()(x)
        b = patched.to(device).eval()(x)
    score_rel = float(((a.pred_score - b.pred_score).abs() / a.pred_score.abs().clamp_min(1e-12)).max())
    map_abs = float((a.anomaly_map - b.anomaly_map).abs().max())

    common = {"steps": steps, "batch_size": batch_size, "device": device, "amp": False, "seed": seed}
    log_a = train(reference, images, log_every=1, **common)
    log_b = train(patched, images, log_every=1, **common)
    loss_rel = [
        abs(ea["loss"] - eb["loss"]) / max(abs(ea["loss"]), 1e-12)
        for ea, eb in zip(log_a, log_b, strict=True)
    ]
    state_a, state_b = trainable_state(reference), trainable_state(patched)
    weight_abs = max(float((state_a[k] - state_b[k]).abs().max()) for k in state_a)
    passed = max(loss_rel) <= PARITY_TOL and score_rel <= PARITY_TOL and map_abs <= PARITY_TOL
    return {
        "passed": bool(passed),
        "tolerance": PARITY_TOL,
        "batch_size": batch_size,
        "steps": steps,
        "score_rel_diff": score_rel,
        "map_abs_diff": map_abs,
        "loss_rel_diff": loss_rel,
        "losses_original": [e["loss"] for e in log_a],
        "losses_scaled": [e["loss"] for e in log_b],
        "weight_abs_diff_after": weight_abs,
    }


def gate_verdict(
    reference: list[float],
    losses: list[float],
    *,
    skipped: int,
    window: int = GATE_WINDOW,
    rtol: float = GATE_RTOL,
) -> dict:
    """Gate 2b on per-step losses: fp16 (`losses`) against fp32 (`reference`) over the same batches.

    Passes when there are as many fp16 losses as reference losses, all finite, no step was skipped by
    the gradient scaler, and the mean loss of every `window` consecutive steps is within `rtol` of the
    reference's mean over the same steps.
    """
    if not reference or len(reference) % window:
        raise ValueError(f"need a reference of a positive multiple of {window} steps, got {len(reference)}")
    out: dict = {"skipped": int(skipped), "window": window, "rtol": rtol}
    if len(losses) != len(reference):
        return {**out, "passed": False, "reason": f"{len(losses)} of {len(reference)} steps ran"}
    values = np.asarray(losses, dtype=np.float64)
    if not np.isfinite(values).all():
        return {**out, "passed": False, "reason": "a loss is not finite"}
    ref = np.asarray(reference, dtype=np.float64).reshape(-1, window).mean(axis=1)
    got = values.reshape(-1, window).mean(axis=1)
    rel = np.abs(got - ref) / np.abs(ref)
    out["window_rel_diff"] = [float(v) for v in rel]
    if skipped:
        return {**out, "passed": False, "reason": f"the gradient scaler skipped {skipped} steps"}
    if float(rel.max()) > rtol:
        return {**out, "passed": False, "reason": f"a window mean differs by {float(rel.max()):.2%}"}
    return {**out, "passed": True, "reason": ""}


def _gate_run(build, attention: str, amp: bool, images, *, steps, batch_size, device, seed, log_path) -> dict:
    import torch

    log_path.unlink(missing_ok=True)
    torch.manual_seed(seed)
    model = build(attention)
    started = time.perf_counter()
    error = ""
    try:
        train(
            model,
            images,
            steps=steps,
            batch_size=batch_size,
            device=device,
            amp=amp,
            seed=seed,
            log_every=1,
            log_path=log_path,
        )
    except FloatingPointError as err:
        error = str(err)
    log = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    return {
        "attention": attention,
        "amp": amp,
        "error": error,
        "losses": [e["loss"] for e in log],
        "skipped": log[-1]["skipped"] if log else 0,
        "peak_vram_mb": log[-1]["vram_mb"] if log else 0.0,
        "seconds": round(time.perf_counter() - started, 1),
        "train_seconds_at_end": log[-1]["seconds"] if log else 0.0,
    }


def run_gate(
    build,
    images: np.ndarray,
    *,
    out_dir: Path,
    steps: int = GATE_STEPS,
    batch_size: int = BATCH,
    device: str = "cuda",
    seed: int = 0,
) -> dict:
    """Gate 2b: fp32 with the original attention, then fp16 with the scaled attention, then (only if that
    fails) fp16 with the attention in fp32. The first fp16 variant that passes is the choice; when none
    passes, training runs in fp32 with the original attention.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    common = {"steps": steps, "batch_size": batch_size, "device": device, "seed": seed}
    reference = _gate_run(
        build, "original", False, images, log_path=out_dir / "log_fp32_original.jsonl", **common
    )
    if reference["error"] or len(reference["losses"]) != steps:
        raise FloatingPointError(f"the fp32 reference run failed: {reference['error']}")
    variants = []
    choice = {"amp": False, "attention": "original"}
    for attention in ("scaled", "fp32"):
        log_path = out_dir / f"log_fp16_{attention}.jsonl"
        run = _gate_run(build, attention, True, images, log_path=log_path, **common)
        run["verdict"] = gate_verdict(reference["losses"], run["losses"], skipped=run["skipped"])
        if run["error"]:
            run["verdict"] = {**run["verdict"], "passed": False, "reason": run["error"]}
        variants.append(run)
        if run["verdict"]["passed"]:
            choice = {"amp": True, "attention": attention}
            break
    return {"reference": reference, "variants": variants, "choice": choice}


def eval_category(
    model: torch.nn.Module,
    protocol: str,
    category: str,
    manifest: list[ManifestRow],
    cache: ImageCache,
    device: str,
    allow_test: bool,
    out_dir: Path,
    *,
    batch_size: int = 32,
    amp: bool = True,
) -> dict:
    """Score one category's evaluation set and write `<category>.npz` and `<category>_maps.npy`."""
    import torch

    eval_normal = select(
        manifest, protocol=protocol, part="eval_normal", category=category, allow_test=allow_test
    )
    eval_defect = select(
        manifest, protocol=protocol, part="eval_defect", category=category, allow_test=allow_test
    )
    eval_rows = eval_normal + eval_defect
    labels = np.array([0] * len(eval_normal) + [1] * len(eval_defect), dtype=np.int8)
    # Thresholds come from the fold-0 normals, which the model never trained on. In the dev protocol
    # those are the evaluation normals themselves, so there is nothing left to calibrate on.
    cal_rows: list[ManifestRow] = []
    if protocol == "test":
        cal_rows = select(manifest, protocol="dev", part="eval_normal", category=category)

    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    timing: dict[str, float] = {}

    t0 = time.perf_counter()
    cal_score = np.empty(0, dtype=np.float32)
    if cal_rows:
        cal = predict(model, cache.images(cal_rows), batch_size=batch_size, device=device, amp=amp)
        cal_score = cal.image_scores
        del cal
    timing["cal_score_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = predict(model, cache.images(eval_rows), batch_size=batch_size, device=device, amp=amp)
    masks = cache.masks(eval_rows).astype(bool)
    timing["eval_score_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    maps = result.maps.astype(np.float32)
    px_auroc = pixel_auroc(maps, masks)
    au_pro = aupro(maps, masks)
    hist = pro_histograms(maps, masks, bins=PRO_BINS)
    timing["pixel_metrics_s"] = time.perf_counter() - t0

    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if cuda else 0.0
    np.savez(
        out_dir / f"{category}.npz",
        eval_images=np.array([r.image for r in eval_rows]),
        eval_labels=labels,
        eval_defect_types=np.array([r.defect_types for r in eval_rows]),
        eval_score=result.image_scores,
        cal_score=cal_score,
        cal_images=np.array([r.image for r in cal_rows], dtype=np.str_),
        pixel_auroc=np.float64(px_auroc),
        aupro=np.float64(au_pro),
        pro_edges=hist.edges,
        pro_normal=hist.normal,
        pro_components=hist.components,
        pro_component_image=hist.component_image,
    )
    np.save(out_dir / f"{category}_maps.npy", result.maps)
    return {
        "category": category,
        "eval_normal": len(eval_normal),
        "eval_defect": len(eval_defect),
        "cal": len(cal_rows),
        "peak_vram_mb": round(peak_mb, 1),
        "timing": {k: round(v, 2) for k, v in timing.items()},
    }


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _check_categories(
    parser: argparse.ArgumentParser, manifest: list[ManifestRow], wanted: list[str]
) -> None:
    known = {r.category for r in manifest}
    unknown = [c for c in wanted if c not in known]
    if unknown or not wanted:
        parser.error(f"unknown or missing categories {unknown}; the manifest has {sorted(known)}")
    repeated = sorted({c for c in wanted if wanted.count(c) > 1})
    if repeated:  # training would see those images twice, evaluation would list the category twice
        parser.error(f"categories named more than once: {repeated}")


def _result_files(out_dir: Path) -> list[Path]:
    """What an evaluation leaves in its folder: run.json, `<category>.npz` and `<category>_maps.npy`."""
    if not out_dir.is_dir():
        return []
    found = [out_dir / "run.json", *out_dir.glob("*.npz"), *out_dir.glob("*_maps.npy")]
    return sorted(p for p in found if p.is_file())


def _dev_pool(manifest: list[ManifestRow], categories: list[str]) -> tuple[list[ManifestRow], dict[str, int]]:
    """The normal pool of the dev protocol (folds 1-4) of `categories`: fold 0 stays out for thresholds."""
    counts = {}
    rows: list[ManifestRow] = []
    for category in categories:
        pool = select(manifest, protocol="dev", part="pool_normal", category=category)
        counts[category] = len(pool)
        rows += pool
    return rows, counts


def _read_gate(parser: argparse.ArgumentParser, path: Path) -> tuple[dict, dict]:
    """(choice, record) of a `gate` result file: the precision and attention to train with."""
    try:
        with open(path, encoding="utf-8") as f:
            gate = json.load(f)
        choice = gate["choice"]
        amp, attention = bool(choice["amp"]), str(choice["attention"])
    except (OSError, ValueError, KeyError, TypeError) as err:
        parser.error(f"{path} is not a gate result ({err})")
    if attention not in ATTENTION_MODES:
        parser.error(f"{path} chooses an unknown attention {attention!r}")
    record = {"file": str(path), "sha256": sha256_file(path), "config": gate.get("config"), **choice}
    return {"amp": amp, "attention": attention}, record


def _train_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    import torch

    cfg = CONFIGS[args.config]
    out_dir = args.out or paths.OUTPUTS / cfg.name
    steps = cfg.steps if args.steps is None else args.steps
    batch_size = cfg.batch_size if args.batch_size is None else args.batch_size
    if (out_dir / "train.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished training run; pass --overwrite to redo it")
    if steps < 1 or batch_size < 1 or args.log_every < 1:
        parser.error("--steps, --batch-size and --log-every must be at least 1")
    if args.encoder is not None and args.encoder != cfg.encoder and cfg.name != NAME:
        parser.error(f"--encoder cannot change the encoder of the registered config {cfg.name!r}")
    encoder = args.encoder or cfg.encoder
    gate_record = None
    want_amp, attention = not args.no_amp, args.attention or cfg.attention
    if args.precision_from is not None:
        if args.no_amp or args.attention is not None:
            parser.error("--precision-from sets the precision and the attention: drop --no-amp / --attention")
        choice, gate_record = _read_gate(parser, args.precision_from)
        want_amp, attention = choice["amp"], choice["attention"]
    options = cfg.build_options(attention)

    commit = git_commit()
    manifest = read_manifest(paths.VISA_MANIFEST)
    _check_categories(parser, manifest, args.categories)
    cache = ImageCache(paths.CACHE, cfg.img_size)
    rows, counts = _dev_pool(manifest, args.categories)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.json", "train_log.jsonl", "model.pt"):  # leftovers of an earlier or broken run
        (out_dir / name).unlink(missing_ok=True)

    started = time.perf_counter()
    images = cache.images(rows)
    amp = amp_applied(want_amp, args.device)  # what really runs: there is no fp16 on the CPU
    torch.manual_seed(args.seed)  # initial weights of the bottleneck and the decoder
    model = build_model(encoder, **options)
    log = train(
        model,
        images,
        steps=steps,
        batch_size=batch_size,
        device=args.device,
        amp=amp,
        seed=args.seed,
        log_every=args.log_every,
        log_path=out_dir / "train_log.jsonl",
    )
    save_model(
        out_dir / "model.pt", model, encoder=encoder, steps=steps, amp=amp, config=cfg.name, options=options
    )
    config = {
        "name": cfg.name,
        "encoder": encoder,
        "img_size": cfg.img_size,
        "steps": steps,
        "batch_size": batch_size,
        "amp": amp,
        "seed": args.seed,
        "attention": attention,
        "options": options,
        "optimizer": {"name": "StableAdamW", **OPTIMIZER},
        "scheduler": {"name": "WarmCosineScheduler", "total_iters": steps, **SCHEDULER},
        "grad_clip_norm": GRAD_CLIP,
    }
    if gate_record is not None:
        config["gate"] = gate_record
    info = {
        "method": "dinomaly",
        "commit": commit,
        "device": args.device,
        "config": config,
        "train_images": len(rows),
        "train_images_by_category": counts,
        "final_loss": log[-1]["loss"],
        "skipped_steps": log[-1]["skipped"],
        "peak_vram_mb": log[-1]["vram_mb"],
        "train_s": log[-1]["seconds"],
        "total_s": round(time.perf_counter() - started, 1),
    }
    _write_json(out_dir / "train.json", info)
    print(json.dumps(info, ensure_ascii=False), flush=True)


def _eval_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.protocol == "test" and (not args.allow_test or not args.stage):
        parser.error("--protocol test needs --allow-test and --stage")
    cfg = CONFIGS[args.config]
    out_dir = args.out or paths.OUTPUTS / f"{cfg.name}-{args.protocol}"
    if (out_dir / "run.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")
    if _result_files(out_dir) and not args.overwrite:
        parser.error(f"{out_dir} holds result files of an unfinished run; pass --overwrite to replace them")
    model_path = args.model or paths.OUTPUTS / cfg.name / "model.pt"
    if not model_path.exists():
        parser.error(f"{model_path} does not exist: run the train command first")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    commit = git_commit()
    manifest = read_manifest(paths.VISA_MANIFEST)
    _check_categories(parser, manifest, args.categories)
    model, saved = load_model(model_path)
    # A file without a config is a stage 2 model (dm).
    if saved.get("config", NAME) != cfg.name:
        parser.error(f"{model_path} holds a {saved.get('config', NAME)!r} model: pass --config to match it")
    cache = ImageCache(paths.CACHE, cfg.img_size)
    # Score in the precision the model was trained in unless a flag says otherwise: fp16 autocast
    # rounds the image scores to fp16, which makes ties among the normals that fix the thresholds.
    want_amp = saved["amp"] if args.amp is None else args.amp
    if want_amp is None:
        parser.error(f"{model_path} does not record the precision it was trained in: pass --amp or --no-amp")
    amp = amp_applied(want_amp, args.device)
    model = model.to(args.device)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Nothing of an earlier run may stay: readers take every *.npz of the folder as part of this run,
    # and a redo that breaks off must not look finished.
    for path in _result_files(out_dir):
        path.unlink()
    if args.protocol == "test":
        # Everything that can fail without the test images has been done: the read starts here.
        record_test_access(
            paths.TEST_LEDGER, stage=args.stage, config=cfg.name, note=args.note, commit=commit
        )

    summaries = []
    started = time.perf_counter()
    for category in args.categories:
        t0 = time.perf_counter()
        info = eval_category(
            model,
            args.protocol,
            category,
            manifest,
            cache,
            args.device,
            args.allow_test,
            out_dir,
            batch_size=args.batch_size,
            amp=want_amp,
        )
        info["total_s"] = round(time.perf_counter() - t0, 1)
        summaries.append(info)
        print(json.dumps(info, ensure_ascii=False), flush=True)

    config = {
        "name": cfg.name,
        "encoder": saved["encoder"],
        "img_size": cfg.img_size,
        "train_steps": saved["steps"],
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "train_amp": saved["amp"],
        "amp": amp,  # what really ran (fp16 autocast exists on CUDA only), not the flag
        "score_precision": "fp16" if amp else "fp32",
        "batch_size": args.batch_size,
        "map_size": 256,
        "calibration": "fold-0 normals (hold-out)" if args.protocol == "test" else "none",
    }
    if "options" in saved:
        config["options"] = saved["options"]
    run = {
        "method": "dinomaly",
        "protocol": args.protocol,
        "commit": commit,
        "device": args.device,
        "config": config,
        "categories": summaries,
        "total_s": round(time.perf_counter() - started, 1),
    }
    _write_json(out_dir / "run.json", run)


def _builder(cfg: DinomalyConfig) -> Callable[[str], torch.nn.Module]:
    """`build(attention)` for the gate checks: a fresh model of `cfg` with that decoder attention."""

    def build(attention: str) -> torch.nn.Module:
        return build_model(cfg.encoder, **cfg.build_options(attention))

    return build


def _check_run(parser, args, steps_flag: int, batch_flag: int) -> list[ManifestRow]:
    if steps_flag < 1 or batch_flag < 1:
        parser.error("--steps and --batch-size must be at least 1")
    manifest = read_manifest(paths.VISA_MANIFEST)
    _check_categories(parser, manifest, args.categories)
    return _dev_pool(manifest, args.categories)[0]


def _parity_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    cfg = CONFIGS[args.config]
    out = args.out or paths.OUTPUTS / f"gate-{cfg.name}" / "parity.json"
    rows = _check_run(parser, args, args.steps, args.batch_size)
    images = ImageCache(paths.CACHE, cfg.img_size).images(rows)
    started = time.perf_counter()
    result = attention_parity(
        _builder(cfg),
        images,
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    record = {
        "check": "parity",
        "config": cfg.name,
        "commit": git_commit(),
        "device": args.device,
        "seed": args.seed,
        "train_images": len(rows),
        **result,
        "seconds": round(time.perf_counter() - started, 1),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out, record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if not result["passed"]:
        raise SystemExit(3)


def _gate_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not amp_applied(True, args.device):
        parser.error("the fp16 gate needs a CUDA device (fp16 autocast runs on CUDA only)")
    cfg = CONFIGS[args.config]
    out_dir = args.out or paths.OUTPUTS / f"gate-{cfg.name}"
    if (out_dir / "gate.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a gate result; pass --overwrite to redo it")
    rows = _check_run(parser, args, args.steps, args.batch_size)
    if args.steps % GATE_WINDOW:
        parser.error(f"--steps must be a multiple of {GATE_WINDOW}")
    (out_dir / "gate.json").unlink(missing_ok=True)
    images = ImageCache(paths.CACHE, cfg.img_size).images(rows)
    started = time.perf_counter()
    result = run_gate(
        _builder(cfg),
        images,
        out_dir=out_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    record = {
        "check": "gate",
        "config": cfg.name,
        "commit": git_commit(),
        "device": args.device,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "train_images": len(rows),
        **result,
        "total_s": round(time.perf_counter() - started, 1),
    }
    _write_json(out_dir / "gate.json", record)
    print(json.dumps({"choice": result["choice"], "total_s": record["total_s"]}), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    config_help = "registered configuration (docs/experiments.md)"

    fit = commands.add_parser("train", help="train one model on the dev normal pool of all categories")
    fit.add_argument("--config", choices=sorted(CONFIGS), default=NAME, help=config_help)
    fit.add_argument("--steps", type=int, default=None, help="default: the config's")
    fit.add_argument("--batch-size", type=int, default=None, help="default: the config's")
    fit.add_argument("--no-amp", action="store_true", help="train in fp32 instead of fp16 autocast")
    fit.add_argument("--attention", choices=ATTENTION_MODES, default=None, help="default: the config's")
    fit.add_argument(
        "--precision-from",
        type=Path,
        default=None,
        help="gate.json whose choice sets precision and attention",
    )
    fit.add_argument("--encoder", default=None, help="timm name of the DINOv2 encoder (dm only)")
    fit.add_argument("--out", type=Path, default=None, help="default: outputs/<config>")
    fit.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--seed", type=int, default=0, help="initial weights, batch order and dropout")
    fit.add_argument("--log-every", type=int, default=50)
    fit.add_argument("--overwrite", action="store_true")

    score = commands.add_parser("eval", help="score the evaluation set of a protocol")
    score.add_argument("--config", choices=sorted(CONFIGS), default=NAME, help=config_help)
    score.add_argument("--protocol", choices=["dev", "test"], required=True)
    score.add_argument("--model", type=Path, default=None, help="default: outputs/<config>/model.pt")
    score.add_argument("--out", type=Path, default=None, help="default: outputs/<config>-<protocol>")
    score.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    score.add_argument("--device", default="cuda")
    score.add_argument("--batch-size", type=int, default=32)
    score.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="score under fp16 autocast (--amp) or in fp32 (--no-amp); default: as the model was trained",
    )
    score.add_argument("--allow-test", action="store_true", help="read the sealed test set (logged)")
    score.add_argument(
        "--stage", default="", help="stage label for the test ledger (required with --allow-test)"
    )
    score.add_argument("--note", default="")
    score.add_argument("--overwrite", action="store_true")

    parity = commands.add_parser("parity", help="gate 2a: scaled against original attention in fp32")
    parity.add_argument("--config", choices=sorted(CONFIGS), default="dms-280", help=config_help)
    parity.add_argument("--steps", type=int, default=3)
    parity.add_argument("--batch-size", type=int, default=8)
    parity.add_argument("--device", default="cpu")
    parity.add_argument("--seed", type=int, default=0)
    parity.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    parity.add_argument("--out", type=Path, default=None, help="default: outputs/gate-<config>/parity.json")

    gate = commands.add_parser("gate", help="gate 2b: fp16 training against fp32 (CUDA)")
    gate.add_argument("--config", choices=sorted(CONFIGS), default="dms-280", help=config_help)
    gate.add_argument("--steps", type=int, default=GATE_STEPS)
    gate.add_argument("--batch-size", type=int, default=BATCH)
    gate.add_argument("--device", default="cuda")
    gate.add_argument("--seed", type=int, default=0)
    gate.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    gate.add_argument("--out", type=Path, default=None, help="default: outputs/gate-<config>")
    gate.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)
    commands_by_name = {
        "train": _train_command,
        "eval": _eval_command,
        "parity": _parity_command,
        "gate": _gate_command,
    }
    commands_by_name[args.command](args, parser)


if __name__ == "__main__":
    main()
