"""Build PatchCore memory banks for one protocol and score every set the analysis needs.

Per category this writes `<out>/<category>.npz` (image scores under the full bank, the hold-out bank,
resubstitution and cross-fitting, plus pixel metrics and PRO histograms) and `<out>/<category>_maps.npy`
(score maps of the evaluation set under the full bank). Tables are made by `defect_inspect.analyze`.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import paths
from .cache import ImageCache
from .configs import PatchCoreConfig, get_config
from .ledger import git_commit, record_test_access
from .metrics import aupro, pixel_auroc, pro_histograms
from .splits import ManifestRow, holdout_fold, pool_folds, read_manifest, select
from .visa import CATEGORIES

PRO_BINS = 2000


def run_category(
    cfg: PatchCoreConfig,
    protocol: str,
    category: str,
    manifest: list[ManifestRow],
    cache: ImageCache,
    extractor,
    device: str,
    allow_test: bool,
    out_dir: Path,
    save_bank: bool = False,
) -> dict:
    import torch

    from .patchcore import build_bank, collect_features, score_images

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
    timing: dict[str, float] = {}

    t0 = time.perf_counter()
    pool_images = cache.images(pool)
    feats, (grid_h, grid_w) = collect_features(
        extractor, pool_images, batch_size=cfg.batch_size, device=device
    )
    dim = feats.shape[1]
    feats = feats.view(len(pool), grid_h * grid_w, dim)
    folds = np.array([r.fold for r in pool], dtype=np.int64)
    timing["features_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    full_bank = build_bank(feats.reshape(-1, dim), cfg.coreset_ratio, seed=cfg.seed, device=device)
    timing["full_bank_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    resub = score(full_bank, pool_images).image_scores
    timing["resub_score_s"] = time.perf_counter() - t0

    # Cross-fitting: every pool image is scored by a bank that never saw its fold.
    t0 = time.perf_counter()
    oof = np.full(len(pool), np.nan, dtype=np.float32)
    holdout_bank = None
    bank_rows = {"full": int(full_bank.shape[0])}
    for fold in pool_folds(protocol):
        keep = torch.from_numpy(folds != fold)
        bank = build_bank(feats[keep].reshape(-1, dim), cfg.coreset_ratio, seed=cfg.seed, device=device)
        held = np.flatnonzero(folds == fold)
        oof[held] = score(bank, pool_images[held]).image_scores
        bank_rows[f"minus_fold_{fold}"] = int(bank.shape[0])
        if fold == holdout_fold(protocol):
            holdout_bank = bank
    timing["crossfit_s"] = time.perf_counter() - t0
    if holdout_bank is None or np.isnan(oof).any():
        raise RuntimeError("cross-fitting did not cover every pool image")
    del feats, pool_images

    t0 = time.perf_counter()
    eval_images = cache.images(eval_rows)
    masks = cache.masks(eval_rows).astype(bool)
    res_full = score(full_bank, eval_images)
    res_holdout = score(holdout_bank, eval_images)
    timing["eval_score_s"] = time.perf_counter() - t0
    timing["eval_images"] = len(eval_rows)

    t0 = time.perf_counter()
    maps = res_full.maps.astype(np.float32)
    px_auroc = pixel_auroc(maps, masks)
    au_pro = aupro(maps, masks)
    hist = pro_histograms(maps, masks, bins=PRO_BINS)
    timing["pixel_metrics_s"] = time.perf_counter() - t0

    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
    np.savez(
        out_dir / f"{category}.npz",
        eval_images=np.array([r.image for r in eval_rows]),
        eval_labels=labels,
        eval_defect_types=np.array([r.defect_types for r in eval_rows]),
        eval_score_full=res_full.image_scores,
        eval_score_holdout=res_holdout.image_scores,
        pool_images=np.array([r.image for r in pool]),
        pool_folds=folds,
        pool_score_resub=resub,
        pool_score_oof=oof,
        pixel_auroc=np.float64(px_auroc),
        aupro=np.float64(au_pro),
        pro_edges=hist.edges,
        pro_normal=hist.normal,
        pro_components=hist.components,
        pro_component_image=hist.component_image,
    )
    np.save(out_dir / f"{category}_maps.npy", res_full.maps)
    if save_bank:
        # The full bank in selection order (fp16): later stages score new conditions against it.
        np.save(out_dir / f"{category}_bank.npy", full_bank.numpy())
    return {
        "category": category,
        "pool": len(pool),
        "eval_normal": len(eval_normal),
        "eval_defect": len(eval_defect),
        "grid": [grid_h, grid_w],
        "dim": int(dim),
        "bank_rows": bank_rows,
        "peak_vram_mb": round(peak_mb, 1),
        "timing": {k: round(v, 2) for k, v in timing.items()},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="p0")
    parser.add_argument("--protocol", choices=["dev", "test"], default="dev")
    parser.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    parser.add_argument("--out", type=Path, default=None, help="default: outputs/<config>-<protocol>")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-test", action="store_true", help="read the sealed test set (logged)")
    parser.add_argument(
        "--stage", default="", help="stage label for the test ledger (required with --allow-test)"
    )
    parser.add_argument("--note", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-bank", action="store_true", help="keep each category's full memory bank")
    args = parser.parse_args(argv)

    from .backbones import make_extractor

    cfg = get_config(args.config)
    if args.protocol == "test":
        if not args.allow_test or not args.stage:
            parser.error("--protocol test needs --allow-test and --stage")
    out_dir = args.out or paths.OUTPUTS / f"{cfg.name}-{args.protocol}"
    if (out_dir / "run.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(paths.VISA_MANIFEST)
    cache = ImageCache(paths.CACHE, cfg.img_size)
    commit = git_commit()
    if args.protocol == "test":
        record_test_access(
            paths.TEST_LEDGER, stage=args.stage, config=cfg.name, note=args.note, commit=commit
        )

    extractor = make_extractor(cfg.backbone, img_size=cfg.img_size).to(args.device).eval()
    summaries = []
    started = time.perf_counter()
    for category in args.categories:
        t0 = time.perf_counter()
        info = run_category(
            cfg,
            args.protocol,
            category,
            manifest,
            cache,
            extractor,
            args.device,
            args.allow_test,
            out_dir,
            save_bank=args.save_bank,
        )
        info["total_s"] = round(time.perf_counter() - t0, 1)
        summaries.append(info)
        print(json.dumps(info, ensure_ascii=False), flush=True)

    run = {
        "config": asdict(cfg),
        "protocol": args.protocol,
        "commit": commit,
        # Coreset selection differs between devices at float rounding level,
        # so banks that are compared must be built on the same device.
        "device": args.device,
        "categories": summaries,
        "total_s": round(time.perf_counter() - started, 1),
    }
    with open(out_dir / "run.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
