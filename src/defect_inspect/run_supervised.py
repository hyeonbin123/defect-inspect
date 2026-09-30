"""Train the supervised patch head for every (category, k, seed) and score the evaluation set.

Features come from the frozen DINOv2 ViT-S/14 at 448 px and are extracted once per category. Every
(k, seed) writes `<out>/k{k}-s{seed}/<category>.npz` (common stage 2 format plus the training defects and
the selected epoch) and `<category>_maps.npy`; `<out>/run.json` lists every run and marks the whole
invocation as finished. An output directory only ever holds the results of one invocation: a redo
(`--overwrite`) first removes what the earlier one wrote. Rules: docs/experiments.md, section
"2단계 / 지도 학습".
"""

import argparse
import json
import re
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from . import paths
from .cache import ImageCache
from .ledger import git_commit, record_test_access
from .metrics import aupro, pixel_auroc, pro_histograms
from .splits import K_VALUES, SEEDS, ManifestRow, label_subset, read_manifest, select
from .visa import CATEGORIES

METHOD = "supervised"
BACKBONE = "dinov2_vits14"
IMG_SIZE = 448
PRO_BINS = 2000
EXTRACT_BATCH = 32
_RUN_DIR_NAME = re.compile(r"k\d+-s\d+")


def run_dir(out_dir: Path, k: int, seed: int) -> Path:
    return Path(out_dir) / f"k{k}-s{seed}"


def previous_results(out_dir: Path) -> list[Path]:
    """Files an earlier invocation left in the `k{k}-s{seed}` folders of `out_dir`, in sorted order.

    Only what this module writes counts: `run.json`, `<category>.npz` and `<category>_maps.npy` directly
    inside such a folder.
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return []
    found: list[Path] = []
    for folder in sorted(out_dir.iterdir()):
        if not folder.is_dir() or not _RUN_DIR_NAME.fullmatch(folder.name):
            continue
        for path in sorted(folder.iterdir()):
            is_result = path.name == "run.json" or path.suffix == ".npz" or path.name.endswith("_maps.npy")
            if is_result and path.is_file():
                found.append(path)
    return found


def clear_previous_results(out_dir: Path) -> int:
    """Remove `<out_dir>/run.json` and every file of `previous_results`; returns how many of the latter.

    The analysis reads `k{k}-s{seed}/<category>.npz` without looking at a run.json, so results of an
    earlier invocation (another commit, other ks, seeds or categories) must not stay next to the new
    ones. Folders that are empty afterwards are removed; any other file is left alone.
    """
    out_dir = Path(out_dir)
    # First, so that a directory that is being cleared or redone never looks finished.
    (out_dir / "run.json").unlink(missing_ok=True)
    found = previous_results(out_dir)
    for path in found:
        path.unlink()
    for folder in sorted({path.parent for path in found}):
        if not any(folder.iterdir()):
            folder.rmdir()
    return len(found)


def label_pool(manifest: list[ManifestRow], category: str) -> list[ManifestRow]:
    """The category's label pool in manifest order (the only source of training defects)."""
    return [r for r in manifest if r.category == category and r.role == "label_pool"]


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run_category(
    protocol: str,
    category: str,
    manifest: list[ManifestRow],
    cache: ImageCache,
    extractor,
    device: str,
    allow_test: bool,
    out_dir: Path,
    *,
    ks: tuple[int, ...] = K_VALUES,
    seeds: tuple[int, ...] = SEEDS,
    train_cfg=None,
) -> tuple[dict, list[dict]]:
    """Extract the features of one category once, then train and evaluate every (k, seed).

    Returns the category info and one info dict per run. `train_cfg` (default `TrainConfig()`) is used
    with its seed replaced by the run's seed.
    """
    import torch

    from . import supervised

    base_cfg = supervised.TrainConfig() if train_cfg is None else train_cfg
    # The evaluation parts come first: without permission the test protocol stops here, before any work.
    eval_normal = select(
        manifest, protocol=protocol, part="eval_normal", category=category, allow_test=allow_test
    )
    eval_defect = select(
        manifest, protocol=protocol, part="eval_defect", category=category, allow_test=allow_test
    )
    eval_rows = eval_normal + eval_defect
    eval_labels = np.array([0] * len(eval_normal) + [1] * len(eval_defect), dtype=np.int8)
    # Training normals are folds 1-4 and validation is fold 0 + dev defects in BOTH protocols.
    train_normal = select(manifest, protocol="dev", part="pool_normal", category=category)
    val_normal = select(manifest, protocol="dev", part="eval_normal", category=category)
    val_defect = select(manifest, protocol="dev", part="eval_defect", category=category)
    val_rows = val_normal + val_defect
    val_labels = np.array([0] * len(val_normal) + [1] * len(val_defect), dtype=np.int8)
    pool_rows = label_pool(manifest, category)
    pool_index = {r.image: i for i, r in enumerate(pool_rows)}
    subsets = {(k, seed): label_subset(manifest, category, k, seed) for k in ks for seed in seeds}

    on_cuda = torch.device(device).type == "cuda"
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()

    def features(rows: list[ManifestRow]) -> torch.Tensor:
        return supervised.extract(extractor, cache.images(rows), batch_size=EXTRACT_BATCH, device=device)

    t0 = time.perf_counter()
    normal_feats = features(train_normal)
    val_feats = features(val_rows)
    pool_feats = features(pool_rows) if pool_rows else normal_feats[:0]
    if protocol == "dev":
        if eval_rows != val_rows:
            raise RuntimeError("the dev evaluation set must be the validation set")
        eval_feats = val_feats
    else:
        eval_feats = features(eval_rows)
    features_s = time.perf_counter() - t0
    grid_h, grid_w, dim = (int(v) for v in normal_feats.shape[1:])
    if grid_h != grid_w:
        raise ValueError(f"expected a square patch grid, got {grid_h}x{grid_w}")

    normal_targets = supervised.patch_targets(cache.masks(train_normal), grid_h)
    pool_targets = (
        supervised.patch_targets(cache.masks(pool_rows), grid_h) if pool_rows else normal_targets[:0]
    )
    eval_masks = cache.masks(eval_rows).astype(bool)

    runs: list[dict] = []
    for k in ks:
        for seed in seeds:
            started = time.perf_counter()
            defects = subsets[(k, seed)]
            index = np.array([pool_index[r.image] for r in defects], dtype=np.int64)
            train_feats = torch.cat([normal_feats, pool_feats[torch.from_numpy(index)]])
            train_targets = np.concatenate([normal_targets, pool_targets[index]])
            cfg = replace(base_cfg, seed=seed)

            t0 = time.perf_counter()
            result = supervised.train_head(
                train_feats, train_targets, val_feats, val_labels, cfg, device=device
            )
            train_s = time.perf_counter() - t0
            del train_feats

            t0 = time.perf_counter()
            scored = supervised.predict(result.head, eval_feats, device=device)
            if protocol == "test":
                # Thresholds are fixed on the fold-0 normals, which the head was not trained on.
                cal_score = supervised.predict(
                    result.head, val_feats[: len(val_normal)], device=device
                ).image_scores
            else:
                # In dev the fold-0 normals are the evaluation normals themselves.
                cal_score = np.empty(0, dtype=np.float32)
            predict_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            maps = scored.maps.astype(np.float32)
            px_auroc = pixel_auroc(maps, eval_masks)
            au_pro = aupro(maps, eval_masks)
            hist = pro_histograms(maps, eval_masks, bins=PRO_BINS)
            del maps
            metrics_s = time.perf_counter() - t0

            target_dir = run_dir(out_dir, k, seed)
            target_dir.mkdir(parents=True, exist_ok=True)
            val_auroc = [np.nan if h["val_auroc"] is None else h["val_auroc"] for h in result.history]
            # Compressed: the PRO histograms are large and almost empty (read back with np.load as usual).
            np.savez_compressed(
                target_dir / f"{category}.npz",
                eval_images=np.array([r.image for r in eval_rows]),
                eval_labels=eval_labels,
                eval_defect_types=np.array([r.defect_types for r in eval_rows]),
                eval_score=scored.image_scores,
                cal_score=cal_score,
                pixel_auroc=np.float64(px_auroc),
                aupro=np.float64(au_pro),
                pro_edges=hist.edges,
                pro_normal=hist.normal,
                pro_components=hist.components,
                pro_component_image=hist.component_image,
                train_defect_images=np.array([r.image for r in defects], dtype=np.str_),
                train_defect_types=np.array([r.defect_types for r in defects], dtype=np.str_),
                best_epoch=np.int64(result.best_epoch),
                best_val_auroc=np.float64(result.best_val_auroc),
                history_loss=np.array([h["loss"] for h in result.history], dtype=np.float64),
                history_val_auroc=np.array(val_auroc, dtype=np.float64),
            )
            np.save(target_dir / f"{category}_maps.npy", scored.maps)
            info = {
                "category": category,
                "k": int(k),
                "seed": int(seed),
                "best_epoch": int(result.best_epoch),
                "best_val_auroc": float(result.best_val_auroc),
                "n_train_normal": len(train_normal),
                "n_train_defect": len(defects),
                "pos_weight": supervised.pos_weight(train_targets, cfg.max_pos_weight),
                "final_loss": float(result.history[-1]["loss"]),
                "train_s": round(train_s, 2),
                "predict_s": round(predict_s, 2),
                "metrics_s": round(metrics_s, 2),
                "seconds": round(time.perf_counter() - started, 2),
            }
            runs.append(info)
            print(json.dumps(info, ensure_ascii=False), flush=True)

    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if on_cuda else 0.0
    category_info = {
        "category": category,
        "train_normal": len(train_normal),
        "val_normal": len(val_normal),
        "val_defect": len(val_defect),
        "label_pool": len(pool_rows),
        "eval_normal": len(eval_normal),
        "eval_defect": len(eval_defect),
        "grid": [grid_h, grid_w],
        "dim": dim,
        "features_s": round(features_s, 2),
        "peak_vram_mb": round(peak_mb, 1),
    }
    return category_info, runs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", choices=["dev", "test"], default="dev")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument("--ks", nargs="+", type=int, default=list(K_VALUES), help="training defects")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--out", type=Path, default=None, help="default: outputs/sup-<protocol>")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-test", action="store_true", help="read the sealed test set (logged)")
    parser.add_argument(
        "--stage", default="", help="stage label for the test ledger (required with --allow-test)"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.protocol == "test" and (not args.allow_test or not args.stage):
        parser.error("--protocol test needs --allow-test and --stage")
    for name in ("categories", "ks", "seeds"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            parser.error(f"--{name} lists a value more than once: {values}")
    out_dir = args.out or paths.OUTPUTS / f"sup-{args.protocol}"
    finished = out_dir / "run.json"
    if finished.exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")
    if previous_results(out_dir) and not args.overwrite:
        parser.error(
            f"{out_dir} holds results of an unfinished run; pass --overwrite to remove them and start again"
        )

    manifest = read_manifest(paths.VISA_MANIFEST)
    for category in args.categories:
        if not any(r.category == category for r in manifest):
            parser.error(f"category {category!r} is not in {paths.VISA_MANIFEST}")
        n_pool = len(label_pool(manifest, category))
        if min(args.ks) < 0 or max(args.ks) > n_pool:
            parser.error(f"--ks must be within 0..{n_pool} (the label pool of {category})")

    import torch

    from .backbones import make_extractor
    from .supervised import TrainConfig

    # Everything that can fail without touching the test set comes before the ledger entry.
    cache = ImageCache(paths.CACHE, IMG_SIZE)
    extractor = make_extractor(BACKBONE, img_size=IMG_SIZE).to(args.device).eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    # An interrupted redo must not look finished, and no file of the earlier invocation may be taken
    # for a result of this one (the checks above only let this happen with --overwrite).
    removed = clear_previous_results(out_dir)
    if removed:
        print(json.dumps({"removed_previous_results": removed}), flush=True)
    commit = git_commit()
    torch_threads = int(torch.get_num_threads())
    config = {
        "backbone": BACKBONE,
        "img_size": IMG_SIZE,
        "ks": list(args.ks),
        "seeds": list(args.seeds),
        "train": asdict(TrainConfig()),  # "seed" is replaced by the seed of each run
    }
    if args.protocol == "test":
        record_test_access(
            paths.TEST_LEDGER,
            stage=args.stage,
            config=METHOD,
            note=f"ks={list(args.ks)} seeds={list(args.seeds)} categories={len(args.categories)}",
            commit=commit,
        )

    started = time.perf_counter()
    categories: list[dict] = []
    runs: list[dict] = []
    for category in args.categories:
        t0 = time.perf_counter()
        info, category_runs = run_category(
            args.protocol,
            category,
            manifest,
            cache,
            extractor,
            args.device,
            args.allow_test,
            out_dir,
            ks=tuple(args.ks),
            seeds=tuple(args.seeds),
        )
        info["total_s"] = round(time.perf_counter() - t0, 1)
        categories.append(info)
        runs.extend(category_runs)
        print(json.dumps(info, ensure_ascii=False), flush=True)

    header = {
        "method": METHOD,
        "protocol": args.protocol,
        "commit": commit,
        "device": args.device,
        # Training is bit-reproducible only on the CPU with the same number of threads; see `train_head`.
        "torch_threads": torch_threads,
    }
    # Every k{k}-s{seed} directory is a run directory of the common format on its own.
    for k in args.ks:
        for seed in args.seeds:
            own = [r for r in runs if r["k"] == k and r["seed"] == seed]
            sub = {
                **header,
                "config": {**config, "k": int(k), "seed": int(seed)},
                "categories": own,
                "total_s": round(sum(r["seconds"] for r in own), 1),
            }
            _write_json(run_dir(out_dir, k, seed) / "run.json", sub)
    run = {
        **header,
        "config": config,
        "categories": categories,
        "runs": runs,
        "total_s": round(time.perf_counter() - started, 1),
    }
    _write_json(finished, run)


if __name__ == "__main__":
    main()
