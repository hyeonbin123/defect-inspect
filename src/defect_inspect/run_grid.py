"""One PatchCore setting (backbone + input size) at several coreset ratios, with cross-fitted scores.

Greedy k-center selection is nested: the bank for a smaller ratio is a prefix of the bank built for the
largest ratio. So the expensive selection runs once per bank (the full bank and one bank per fold), and
every ratio only re-scores. Each ratio gets its own folder in the `run_patchcore` layout, readable by
`compare.load_patchcore`.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import paths
from .cache import ImageCache
from .configs import PatchCoreConfig
from .ledger import git_commit, record_test_access
from .metrics import aupro, pixel_auroc, pro_histograms
from .splits import PARTS, ManifestRow, pool_folds, read_manifest, select
from .visa import CATEGORIES

PRO_BINS = 2000
BACKBONES = ("wrn50", "dinov2_vits14", "dinov2_vitb14")


def ratio_name(ratio: float) -> str:
    """Folder name of a ratio: `r0.1`, `r0.01`, `r0.001`."""
    return f"r{float(ratio)!r}"


def ratio_rows(n_features: int, ratio: float) -> int:
    """Bank rows for `ratio` of `n_features` patch features: the same rounding as `patchcore.build_bank`."""
    return n_features if ratio >= 1 else max(1, round(ratio * n_features))


def setting_name(backbone: str, size: int) -> str:
    return f"{backbone}-{size}"


def _is_ratio_folder(name: str) -> bool:
    try:
        ratio = float(name[1:])
    except ValueError:
        return False
    return 0 < ratio <= 1 and ratio_name(ratio) == name


def _remove_if_empty(folder: Path) -> None:
    try:
        folder.rmdir()
    except OSError:
        pass  # something that is not a grid output is in there: keep the folder


def clear_outputs(out_dir: Path) -> None:
    """Delete what earlier calls wrote to `out_dir`: run.json, the ratio folders' runs, the saved banks.

    A call owns its output folder: results of other ratios or categories left by an earlier (or an
    interrupted) call would otherwise still load as a finished run through `compare.load_patchcore`.
    Only files with the names this module writes are deleted. The top-level run.json goes first, so the
    folder stops counting as finished before anything else changes.
    """
    (out_dir / "run.json").unlink(missing_ok=True)
    for folder in sorted(p for p in out_dir.glob("r*") if p.is_dir() and _is_ratio_folder(p.name)):
        for path in [folder / "run.json", *sorted(folder.glob("*.npz"))]:
            path.unlink(missing_ok=True)
        _remove_if_empty(folder)
    bank_dir = out_dir / "banks"
    if bank_dir.is_dir():
        for pattern in ("*.json", "*_full.npy", "*_minus_fold_*.npy"):
            for path in sorted(bank_dir.glob(pattern)):
                path.unlink(missing_ok=True)
        _remove_if_empty(bank_dir)


def run_category(
    cfg: PatchCoreConfig,
    ratios: list[float],
    protocol: str,
    category: str,
    manifest: list[ManifestRow],
    cache: ImageCache,
    extractor,
    device: str,
    allow_test: bool,
    out_dir: Path,
    save_banks: bool = False,
) -> dict[float, dict]:
    """Build the banks at `cfg.coreset_ratio` (the largest ratio) and score every ratio in `ratios`."""
    import torch

    from .patchcore import build_bank, collect_features, score_images

    if max(ratios) != cfg.coreset_ratio:
        raise ValueError("cfg.coreset_ratio must be the largest ratio")

    def score(bank, images):
        return score_images(
            extractor,
            bank,
            images,
            batch_size=cfg.batch_size,
            device=device,
            reweight_k=cfg.reweight_k,
            sigma=cfg.sigma,
        )

    pool = select(manifest, protocol=protocol, part="pool_normal", category=category)
    eval_normal = select(
        manifest, protocol=protocol, part="eval_normal", category=category, allow_test=allow_test
    )
    eval_defect = select(
        manifest, protocol=protocol, part="eval_defect", category=category, allow_test=allow_test
    )
    eval_rows = eval_normal + eval_defect
    labels = np.array([0] * len(eval_normal) + [1] * len(eval_defect), dtype=np.int8)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    pool_images = cache.images(pool)
    feats, (grid_h, grid_w) = collect_features(
        extractor, pool_images, batch_size=cfg.batch_size, device=device
    )
    dim = feats.shape[1]
    per_image = grid_h * grid_w
    feats = feats.view(len(pool), per_image, dim)
    folds = np.array([r.fold for r in pool], dtype=np.int64)

    # The selections: one for the full pool, one per fold without that fold. N = patches they chose from.
    full_bank = build_bank(feats.reshape(-1, dim), cfg.coreset_ratio, seed=cfg.seed, device=device)
    n_full = len(pool) * per_image
    fold_banks: dict[int, tuple[torch.Tensor, int]] = {}
    for fold in pool_folds(protocol):
        keep = torch.from_numpy(folds != fold)
        bank = build_bank(feats[keep].reshape(-1, dim), cfg.coreset_ratio, seed=cfg.seed, device=device)
        fold_banks[fold] = (bank, int(keep.sum()) * per_image)
    del feats
    banks_s = time.perf_counter() - t0

    if save_banks:
        bank_dir = out_dir / "banks"
        bank_dir.mkdir(parents=True, exist_ok=True)
        np.save(bank_dir / f"{category}_full.npy", full_bank.numpy())
        sources = {"full": n_full}
        for fold, (bank, n) in fold_banks.items():
            np.save(bank_dir / f"{category}_minus_fold_{fold}.npy", bank.numpy())
            sources[f"minus_fold_{fold}"] = n
        with open(bank_dir / f"{category}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump({"largest_ratio": cfg.coreset_ratio, "n_features": sources}, f, indent=2)
            f.write("\n")

    eval_images = cache.images(eval_rows)
    masks = cache.masks(eval_rows).astype(bool)
    infos: dict[float, dict] = {}
    for ratio in sorted(ratios, reverse=True):
        t0 = time.perf_counter()
        rows_full = ratio_rows(n_full, ratio)
        res = score(full_bank[:rows_full], eval_images)
        oof = np.full(len(pool), np.nan, dtype=np.float32)
        for fold, (bank, n) in fold_banks.items():
            held = np.flatnonzero(folds == fold)
            oof[held] = score(bank[: ratio_rows(n, ratio)], pool_images[held]).image_scores
        if np.isnan(oof).any():
            raise RuntimeError("cross-fitting did not cover every pool image")

        maps = res.maps.astype(np.float32)
        hist = pro_histograms(maps, masks, bins=PRO_BINS)
        ratio_dir = out_dir / ratio_name(ratio)
        ratio_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            ratio_dir / f"{category}.npz",
            eval_images=np.array([r.image for r in eval_rows]),
            eval_labels=labels,
            eval_defect_types=np.array([r.defect_types for r in eval_rows]),
            eval_score_full=res.image_scores,
            pool_images=np.array([r.image for r in pool]),
            pool_folds=folds,
            pool_score_oof=oof,
            pixel_auroc=np.float64(pixel_auroc(maps, masks)),
            aupro=np.float64(aupro(maps, masks)),
            pro_edges=hist.edges,
            pro_normal=hist.normal,
            pro_components=hist.components,
            pro_component_image=hist.component_image,
            bank_rows=np.int64(rows_full),
        )
        infos[ratio] = {
            "category": category,
            "pool": len(pool),
            "eval_normal": len(eval_normal),
            "eval_defect": len(eval_defect),
            "grid": [grid_h, grid_w],
            "dim": int(dim),
            "bank_rows": {"full": rows_full},
            "score_s": round(time.perf_counter() - t0, 2),
            "banks_s": round(banks_s, 2),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else 0.0,
        }
    return infos


def _ratio_config(cfg: PatchCoreConfig, ratio: float) -> dict:
    return {
        "name": f"{setting_name(cfg.backbone, cfg.img_size)}-{ratio_name(ratio)}",
        "backbone": cfg.backbone,
        "img_size": cfg.img_size,
        "coreset_ratio": ratio,
        "reweight_k": cfg.reweight_k,
        "sigma": cfg.sigma,
        "batch_size": cfg.batch_size,
        "seed": cfg.seed,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--size", type=int, required=True, help="input size (a multiple of 14 for DINOv2)")
    parser.add_argument("--protocol", choices=["dev", "test"], required=True)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.01, 0.001])
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument(
        "--out", type=Path, default=None, help="default: outputs/grid-<backbone>-<size>-<protocol>"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-banks", action="store_true", help="keep the full and the minus-fold banks")
    parser.add_argument("--allow-test", action="store_true", help="read the sealed test set (logged)")
    parser.add_argument(
        "--stage", default="", help="stage label for the test ledger (required with --allow-test)"
    )
    parser.add_argument("--note", default="")
    parser.add_argument(
        "--overwrite", action="store_true", help="redo a finished run (its outputs are deleted first)"
    )
    args = parser.parse_args(argv)

    from .backbones import make_extractor

    ratios = sorted(set(args.ratios), reverse=True)
    if any(not 0 < r <= 1 for r in ratios):
        parser.error("ratios must be in (0, 1]")
    if args.backbone.startswith("dinov2") and args.size % 14:
        parser.error("DINOv2 input sizes must be multiples of 14")
    if args.protocol == "test" and (not args.allow_test or not args.stage):
        parser.error("--protocol test needs --allow-test and --stage")
    setting = setting_name(args.backbone, args.size)
    out_dir = args.out or paths.OUTPUTS / f"grid-{setting}-{args.protocol}"
    if (out_dir / "run.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")

    # Everything that can fail without touching a test image comes before the ledger line.
    manifest = read_manifest(paths.VISA_MANIFEST)
    for category in args.categories:
        if args.categories.count(category) > 1:
            parser.error(f"--categories lists {category} twice")
        empty = [
            part
            for part in PARTS  # manifest rows only: no image is read here
            if not select(
                manifest, protocol=args.protocol, part=part, category=category, allow_test=args.allow_test
            )
        ]
        if empty:
            parser.error(f"unknown or empty category {category!r}: the manifest has no {', '.join(empty)}")
    try:
        cache = ImageCache(paths.CACHE, args.size)
    except FileNotFoundError:
        command = f"python -m defect_inspect.cache --size {args.size}"
        parser.error(f"no {args.size} px image cache: build it with `{command}`")
    cfg = PatchCoreConfig(name=setting, backbone=args.backbone, img_size=args.size, coreset_ratio=ratios[0])
    extractor = make_extractor(cfg.backbone, img_size=cfg.img_size).to(args.device).eval()
    commit = git_commit()
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_outputs(out_dir)
    if args.protocol == "test":
        record_test_access(
            paths.TEST_LEDGER, stage=args.stage, config=f"grid-{setting}", note=args.note, commit=commit
        )

    per_ratio: dict[float, list[dict]] = {r: [] for r in ratios}
    started = time.perf_counter()
    for category in args.categories:
        t0 = time.perf_counter()
        infos = run_category(
            cfg,
            ratios,
            args.protocol,
            category,
            manifest,
            cache,
            extractor,
            args.device,
            args.allow_test,
            out_dir,
            save_banks=args.save_banks,
        )
        for ratio, info in infos.items():
            per_ratio[ratio].append(info)
        line = {"category": category, "seconds": round(time.perf_counter() - t0, 1)}
        line["banks_s"] = infos[ratios[0]]["banks_s"]
        print(json.dumps(line), flush=True)

    def dump(path: Path, payload: dict) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")

    for ratio in ratios:
        dump(
            out_dir / ratio_name(ratio) / "run.json",
            {
                "config": _ratio_config(cfg, ratio),
                "protocol": args.protocol,
                "commit": commit,
                "device": args.device,
                "categories": per_ratio[ratio],
            },
        )
    dump(
        out_dir / "run.json",
        {
            "setting": setting,
            "ratios": ratios,
            "protocol": args.protocol,
            "commit": commit,
            "device": args.device,
            "categories": list(args.categories),
            "save_banks": bool(args.save_banks),
            "total_s": round(time.perf_counter() - started, 1),
        },
    )


if __name__ == "__main__":
    main()
