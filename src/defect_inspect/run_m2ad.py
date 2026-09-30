"""PatchCore inspectors on M2AD: one per (category, view), under synthetic and real illumination changes.

Per inspector this writes `<out>/<category>_<view>.npz`: the scores of the test specimens under the
reference illumination (S), the 15 synthetic conditions applied to it (P) and the nine other real
illuminations (R), the cross-fitted threshold, and the scores and thresholds after recalibration with
normal images under each new illumination. Tables are made by `defect_inspect.analyze_m2ad`.
Rules: docs/experiments.md, stage 3-B.
"""

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import paths
from .calibrate import Threshold, conformal_threshold
from .conditions import apply_condition, condition_names
from .configs import PatchCoreConfig, get_config
from .ledger import git_commit, record_test_access
from .m2ad import (
    CATEGORIES,
    ILLUMINATIONS,
    N_FOLDS,
    RECAL_SIZES,
    REFERENCE,
    VIEWS,
    M2adCache,
    M2adRow,
    read_meta,
    recal_specimens,
    specimen_folds,
)
from .splits import SealedTestError

ALPHA = 0.05
METHODS = ("p0", "d-s")
# The registered code check: one inspector, the reference condition and one real illumination.
CHECK_INSPECTOR = ("Motor", "000")
CHECK_ILLUMINATIONS = ("02",)
# Score maps are not kept here, and image scores do not depend on the size the maps are resized to.
_MAP_SIZE = 32


def real_illuminations() -> list[str]:
    return [light for light in ILLUMINATIONS if light != REFERENCE]


def condition_list(check: bool = False) -> list[str]:
    """`S`, then `P:<name>` for the 15 synthetic conditions, then `R:<LL>` for illuminations 02..10."""
    if check:
        return ["S", *(f"R:{light}" for light in CHECK_ILLUMINATIONS)]
    synthetic = condition_names(include_clean=False)
    return ["S", *(f"P:{name}" for name in synthetic), *(f"R:{light}" for light in real_illuminations())]


def recal_key_list(recal_sizes: Sequence[int] = RECAL_SIZES) -> list[str]:
    """`<LL>:<n>` for every new illumination and recalibration size, illuminations first."""
    return [f"{light}:{n}" for light in real_illuminations() for n in recal_sizes]


def inspector_rows(
    rows: Sequence[M2adRow], category: str, view: str, split: str, illumination: str
) -> list[M2adRow]:
    """The images one inspector sees under one illumination: one per specimen, sorted by specimen name."""
    wanted = (category, view, split, illumination)
    picked = sorted(
        (r for r in rows if (r.category, r.view, r.split, r.illumination) == wanted), key=lambda r: r.specimen
    )
    names = [r.specimen for r in picked]
    if len(set(names)) != len(names):
        raise ValueError(f"{wanted}: a specimen has more than one image")
    return picked


def _same_specimens(picked: list[M2adRow], names: list[str], what: str) -> None:
    if [r.specimen for r in picked] != names:
        raise ValueError(f"{what}: the specimens differ from those under illumination {REFERENCE}")


def crossfit(
    build: Callable, score: Callable, feats, images: np.ndarray, folds: np.ndarray
) -> tuple[object, np.ndarray, Threshold, dict[str, int]]:
    """Bank of all images, their cross-fitted scores and the conformal threshold from those scores.

    `feats` is [M, P, D] (torch), `images` uint8 [M, S, S, 3], `folds` int [M]. Every image is scored by a
    bank built without its fold; images of one specimen share a fold, so they are held out together.
    """
    import torch

    dim = feats.shape[2]
    bank = build(feats.reshape(-1, dim))
    oof = np.full(len(images), np.nan, dtype=np.float32)
    bank_rows = {"full": int(bank.shape[0])}
    for fold in range(N_FOLDS):
        held = np.flatnonzero(folds == fold)
        if held.size == 0:
            continue
        fold_bank = build(feats[torch.from_numpy(folds != fold)].reshape(-1, dim))
        oof[held] = score(fold_bank, images[held])
        bank_rows[f"minus_fold_{fold}"] = int(fold_bank.shape[0])
    if np.isnan(oof).any():
        raise RuntimeError("cross-fitting did not cover every image")
    return bank, oof, conformal_threshold(oof, ALPHA), bank_rows


def _threshold_info(thr: Threshold) -> dict:
    return {"value": thr.value, "rank": thr.rank, "n": thr.n, "guaranteed": thr.guaranteed}


def run_inspector(
    cfg: PatchCoreConfig,
    category: str,
    view: str,
    rows: Sequence[M2adRow],
    cache: M2adCache,
    extractor,
    device: str,
    out_dir: Path,
    *,
    allow_test: bool = False,
    check: bool = False,
    recal_sizes: Sequence[int] = RECAL_SIZES,
) -> dict:
    """Build, calibrate, score and recalibrate the inspector of one (category, view); write its npz.

    Test images are read only with `allow_test=True`, or by the registered check (`check=True` on
    CHECK_INSPECTOR: conditions S and R:02, no recalibration).
    """
    import torch

    from .patchcore import build_bank, collect_features, score_images

    if allow_test is not True and not (check and (category, view) == CHECK_INSPECTOR):
        raise SealedTestError(
            "M2AD test images are read only by the registered check (--check: "
            f"{CHECK_INSPECTOR[0]} {CHECK_INSPECTOR[1]}, S and R:02) or with allow_test=True (--allow-test)"
        )

    def score(bank, images: np.ndarray) -> np.ndarray:
        return score_images(
            extractor,
            bank,
            images,
            batch_size=cfg.batch_size,
            device=device,
            reweight_k=cfg.reweight_k,
            sigma=cfg.sigma,
            map_size=_MAP_SIZE,
        ).image_scores

    def build(features):
        return build_bank(features, cfg.coreset_ratio, seed=cfg.seed, device=device)

    def features(images: np.ndarray):
        feats, (grid_h, grid_w) = collect_features(
            extractor, images, batch_size=cfg.batch_size, device=device
        )
        return feats.view(len(images), grid_h * grid_w, feats.shape[1]), (grid_h, grid_w)

    started = time.perf_counter()
    on_cuda = str(device).startswith("cuda")
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()

    # 1-2. Reference bank and cross-fitted threshold from the train specimens under the reference light.
    train_ref = inspector_rows(rows, category, view, "train", REFERENCE)
    if not train_ref:
        raise ValueError(f"no train images for {category} {view} under illumination {REFERENCE}")
    train_names = [r.specimen for r in train_ref]
    fold_of = specimen_folds(train_names)
    folds = np.array([fold_of[name] for name in train_names], dtype=np.int64)
    ref_images = cache.images(train_ref)
    ref_feats, grid = features(ref_images)
    bank, cal_score, thr, bank_rows = crossfit(build, score, ref_feats, ref_images, folds)

    # 3. Conditions on the test specimens (the only place where test images are read).
    test_ref = inspector_rows(rows, category, view, "test", REFERENCE)
    if not test_ref:
        raise ValueError(f"no test images for {category} {view} under illumination {REFERENCE}")
    test_names = [r.specimen for r in test_ref]
    test_ref_images = cache.images(test_ref)
    ref_labels = np.array([r.label for r in test_ref], dtype=np.int8)

    conditions = condition_list(check)
    recal_keys = [] if check else recal_key_list(recal_sizes)
    n_test = len(test_ref)
    scores = np.empty((len(conditions), n_test), dtype=np.float32)
    labels = np.empty((len(conditions), n_test), dtype=np.int8)
    recal_scores = np.empty((len(recal_keys), n_test), dtype=np.float32)
    recal_labels = np.empty((len(recal_keys), n_test), dtype=np.int8)
    recal_thresholds = np.empty(len(recal_keys), dtype=np.float64)
    recal_info: dict[str, dict] = {}

    for k, condition in enumerate(conditions):
        kind, _, arg = condition.partition(":")
        if kind == "S":
            images, shown = test_ref_images, ref_labels
        elif kind == "P":
            images, shown = apply_condition(test_ref_images, arg), ref_labels
        else:
            test_rows = inspector_rows(rows, category, view, "test", arg)
            _same_specimens(test_rows, test_names, f"{category} {view} test, illumination {arg}")
            images = cache.images(test_rows)
            shown = np.array([r.label for r in test_rows], dtype=np.int8)
        scores[k] = score(bank, images)
        labels[k] = shown
        if kind != "R" or check:
            continue

        # 4. Recalibration under the new illumination `arg`: its train images join the bank and the
        # cross-fitting. A specimen's reference and new images share a fold.
        new_rows = inspector_rows(rows, category, view, "train", arg)
        _same_specimens(new_rows, train_names, f"{category} {view} train, illumination {arg}")
        new_images = cache.images(new_rows)
        new_feats, _ = features(new_images)
        for n in recal_sizes:
            chosen = recal_specimens(train_names, n)
            pick = np.array([train_names.index(name) for name in chosen], dtype=np.int64)
            bank_n, oof_n, thr_n, rows_n = crossfit(
                build,
                score,
                torch.cat([ref_feats, new_feats[torch.from_numpy(pick)]]),
                np.concatenate([ref_images, new_images[pick]]),
                np.concatenate([folds, folds[pick]]),
            )
            key = f"{arg}:{n}"
            j = recal_keys.index(key)
            recal_scores[j] = score(bank_n, images)
            recal_labels[j] = shown
            recal_thresholds[j] = thr_n.value
            recal_info[key] = {"images": len(oof_n), "bank_rows": rows_n, "threshold": _threshold_info(thr_n)}
            del bank_n

    np.savez(
        out_dir / f"{category}_{view}.npz",
        specimens=np.array(test_names),
        object_anomaly=np.array([r.object_anomaly for r in test_ref], dtype=np.int8),
        conditions=np.array(conditions),
        scores=scores,
        labels=labels,
        threshold=np.float64(thr.value),
        cal_score=cal_score,
        recal_keys=np.array(recal_keys, dtype=str),
        recal_scores=recal_scores,
        recal_thresholds=recal_thresholds,
        recal_labels=recal_labels,
    )
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if on_cuda else 0.0
    return {
        "category": category,
        "view": view,
        "train_specimens": len(train_names),
        "test_specimens": n_test,
        "test_anomalous_specimens": int(sum(r.object_anomaly for r in test_ref)),
        "grid": list(grid),
        "dim": int(ref_feats.shape[2]),
        "folds": {name: int(fold_of[name]) for name in train_names},
        "bank_rows": bank_rows,
        "threshold": _threshold_info(thr),
        "recal": recal_info,
        "peak_vram_mb": round(peak_mb, 1),
        "seconds": round(time.perf_counter() - started, 1),
    }


def _make_extractor(cfg: PatchCoreConfig, device: str):
    from .backbones import make_extractor

    return make_extractor(cfg.backbone, img_size=cfg.img_size).to(device).eval()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=METHODS, required=True, help="PatchCore config")
    parser.add_argument(
        "--check",
        action="store_true",
        help="registered code check: the Motor 000 inspector with S and R:02 only, into <out>-check",
    )
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--out", type=Path, default=None, help="default: outputs/m2ad-<method>")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-test", action="store_true", help="read the M2AD test images (logged)")
    parser.add_argument(
        "--stage", default="", help="stage label for the test ledger (required with --allow-test)"
    )
    parser.add_argument("--note", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jsons", type=Path, default=None, help="default: data/raw/m2ad/jsons.zip")
    parser.add_argument("--cache", type=Path, default=None, help="cache directory (data/cache)")
    args = parser.parse_args(argv)

    cfg = get_config(args.method)
    if not args.check and (not args.allow_test or not args.stage):
        parser.error("the full run reads the M2AD test images: it needs --allow-test and --stage")
    out_dir = args.out or paths.OUTPUTS / f"m2ad-{args.method}"
    if args.check:
        out_dir = out_dir.with_name(out_dir.name + "-check")
    if (out_dir / "run.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")

    # Everything that can fail without touching a test image comes before the ledger line.
    rows = read_meta(args.jsons or paths.RAW / "m2ad" / "jsons.zip")
    cache = M2adCache(args.cache or paths.CACHE, cfg.img_size)
    extractor = _make_extractor(cfg, args.device)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    if args.check:
        inspectors = [CHECK_INSPECTOR]
    else:
        inspectors = [
            (category, view) for category in CATEGORIES if category in args.categories for view in VIEWS
        ]
        record_test_access(
            paths.TEST_LEDGER, stage=args.stage, config=f"m2ad-{args.method}", note=args.note, commit=commit
        )
    # With --overwrite: an interrupted rerun must not look like the finished run it replaces. Only now,
    # after the ledger line: a rerun that could not be recorded reads nothing and leaves that run as it is.
    (out_dir / "run.json").unlink(missing_ok=True)

    summaries = []
    started = time.perf_counter()
    for category, view in inspectors:
        info = run_inspector(
            cfg,
            category,
            view,
            rows,
            cache,
            extractor,
            args.device,
            out_dir,
            allow_test=args.allow_test and not args.check,
            check=args.check,
        )
        summaries.append(info)
        print(json.dumps({k: v for k, v in info.items() if k not in ("folds", "recal")}), flush=True)

    run = {
        "method": args.method,
        "config": asdict(cfg),
        "commit": commit,
        # Coreset selection differs between devices at float rounding level.
        "device": args.device,
        "check": args.check,
        "alpha": ALPHA,
        "reference": REFERENCE,
        "conditions": condition_list(args.check),
        "recal_keys": [] if args.check else recal_key_list(),
        "inspectors": summaries,
        "total_s": round(time.perf_counter() - started, 1),
    }
    with open(out_dir / "run.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
