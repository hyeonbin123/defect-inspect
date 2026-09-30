"""Dinomaly with anomalib's model code and a plain training loop: one model for all categories.

`train` fits the bottleneck and the decoder on the normal pool of the dev protocol (folds 1-4) and writes
`model.pt`, `train_log.jsonl` and `train.json`. `eval` scores the evaluation set of a protocol with that
model and writes the common run format: `run.json`, `<category>.npz` and `<category>_maps.npy`. It scores
in the precision the model was trained in (fp16 autocast or fp32) unless `--amp` / `--no-amp` says
otherwise, and records the precision that really ran in `run.json`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
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
IMG_SIZE = 392
STEPS = 5000
BATCH = 16
# The values of anomalib's TRAINING_CONFIG (dinomaly/lightning_model.py); a test keeps them in step.
OPTIMIZER = {"lr": 2e-3, "betas": (0.9, 0.999), "weight_decay": 1e-4, "amsgrad": True, "eps": 1e-8}
SCHEDULER = {"base_value": 2e-3, "final_value": 2e-4, "warmup_iters": 100}
GRAD_CLIP = 0.1
PRO_BINS = 2000
NAME = "dm"  # method name in docs/experiments.md; also the config label in the test ledger


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


def build_model(encoder_name: str = ENCODER) -> torch.nn.Module:
    """anomalib's DinomalyModel, set up as its Lightning module does: fp32, frozen encoder, fresh decoder.

    Only the bottleneck and the decoder train. Their Linear layers start from a truncated normal (std
    0.01, cut at +-0.03) with zero bias, their LayerNorms from weight 1 and bias 0. These weights come
    from the global torch generator: call `torch.manual_seed` first to fix them.
    """
    import torch
    from anomalib.models.image.dinomaly.torch_model import DinomalyModel

    model = DinomalyModel(encoder_name=encoder_name).float()
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


def save_model(path: Path, model: torch.nn.Module, *, encoder: str, steps: int, amp: bool) -> None:
    """Write the trainable state with the encoder name and step count (the frozen encoder is not stored).

    `amp` is the precision the training really ran in (True: fp16 autocast); `eval` scores in the same
    precision unless told otherwise.
    """
    import torch

    path = Path(path)
    part = path.with_name(path.name + ".part")
    saved = {"encoder": encoder, "steps": int(steps), "amp": bool(amp), "state": trainable_state(model)}
    torch.save(saved, part)
    os.replace(part, path)


def load_model(path: Path) -> tuple[torch.nn.Module, dict]:
    """Rebuild the model of a `save_model` file on the CPU: (model in eval mode, {"encoder", "steps", "amp"}).

    "amp" is None for a file that does not record the precision it was trained in.
    """
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(saved["encoder"])
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
    return model.eval(), meta


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


def _train_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    import torch

    out_dir = args.out or paths.OUTPUTS / NAME
    if (out_dir / "train.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished training run; pass --overwrite to redo it")
    if args.steps < 1 or args.batch_size < 1 or args.log_every < 1:
        parser.error("--steps, --batch-size and --log-every must be at least 1")

    commit = git_commit()
    manifest = read_manifest(paths.VISA_MANIFEST)
    _check_categories(parser, manifest, args.categories)
    cache = ImageCache(paths.CACHE, IMG_SIZE)
    # The normal pool of the dev protocol in both protocols: fold 0 stays out for thresholds.
    counts = {}
    rows: list[ManifestRow] = []
    for category in args.categories:
        pool = select(manifest, protocol="dev", part="pool_normal", category=category)
        counts[category] = len(pool)
        rows += pool

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.json", "train_log.jsonl", "model.pt"):  # leftovers of an earlier or broken run
        (out_dir / name).unlink(missing_ok=True)

    started = time.perf_counter()
    images = cache.images(rows)
    amp = amp_applied(not args.no_amp, args.device)  # what really runs: there is no fp16 on the CPU
    torch.manual_seed(args.seed)  # initial weights of the bottleneck and the decoder
    model = build_model(args.encoder)
    log = train(
        model,
        images,
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
        amp=amp,
        seed=args.seed,
        log_every=args.log_every,
        log_path=out_dir / "train_log.jsonl",
    )
    save_model(out_dir / "model.pt", model, encoder=args.encoder, steps=args.steps, amp=amp)
    info = {
        "method": "dinomaly",
        "commit": commit,
        "device": args.device,
        "config": {
            "name": NAME,
            "encoder": args.encoder,
            "img_size": IMG_SIZE,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "amp": amp,
            "seed": args.seed,
            "optimizer": {"name": "StableAdamW", **OPTIMIZER},
            "scheduler": {"name": "WarmCosineScheduler", "total_iters": args.steps, **SCHEDULER},
            "grad_clip_norm": GRAD_CLIP,
        },
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
    out_dir = args.out or paths.OUTPUTS / f"{NAME}-{args.protocol}"
    if (out_dir / "run.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")
    if _result_files(out_dir) and not args.overwrite:
        parser.error(f"{out_dir} holds result files of an unfinished run; pass --overwrite to replace them")
    model_path = args.model or paths.OUTPUTS / NAME / "model.pt"
    if not model_path.exists():
        parser.error(f"{model_path} does not exist: run the train command first")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    commit = git_commit()
    manifest = read_manifest(paths.VISA_MANIFEST)
    _check_categories(parser, manifest, args.categories)
    cache = ImageCache(paths.CACHE, IMG_SIZE)
    model, saved = load_model(model_path)
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
        record_test_access(paths.TEST_LEDGER, stage=args.stage, config=NAME, note=args.note, commit=commit)

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

    run = {
        "method": "dinomaly",
        "protocol": args.protocol,
        "commit": commit,
        "device": args.device,
        "config": {
            "name": NAME,
            "encoder": saved["encoder"],
            "img_size": IMG_SIZE,
            "train_steps": saved["steps"],
            "model": str(model_path),
            "model_sha256": sha256_file(model_path),
            "train_amp": saved["amp"],
            "amp": amp,  # what really ran (fp16 autocast exists on CUDA only), not the flag
            "score_precision": "fp16" if amp else "fp32",
            "batch_size": args.batch_size,
            "map_size": 256,
            "calibration": "fold-0 normals (hold-out)" if args.protocol == "test" else "none",
        },
        "categories": summaries,
        "total_s": round(time.perf_counter() - started, 1),
    }
    _write_json(out_dir / "run.json", run)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("train", help="train one model on the dev normal pool of all categories")
    fit.add_argument("--steps", type=int, default=STEPS)
    fit.add_argument("--batch-size", type=int, default=BATCH)
    fit.add_argument("--no-amp", action="store_true", help="train in fp32 instead of fp16 autocast")
    fit.add_argument("--encoder", default=ENCODER, help="timm name of the DINOv2 encoder")
    fit.add_argument("--out", type=Path, default=None, help=f"default: outputs/{NAME}")
    fit.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--seed", type=int, default=0, help="initial weights, batch order and dropout")
    fit.add_argument("--log-every", type=int, default=50)
    fit.add_argument("--overwrite", action="store_true")

    score = commands.add_parser("eval", help="score the evaluation set of a protocol")
    score.add_argument("--protocol", choices=["dev", "test"], required=True)
    score.add_argument("--model", type=Path, default=None, help=f"default: outputs/{NAME}/model.pt")
    score.add_argument("--out", type=Path, default=None, help=f"default: outputs/{NAME}-<protocol>")
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

    args = parser.parse_args(argv)
    if args.command == "train":
        _train_command(args, parser)
    else:
        _eval_command(args, parser)


if __name__ == "__main__":
    main()
