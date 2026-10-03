"""Stage 7: multi-illumination enrolment and per-image feature centring for the M2AD inspectors.

Candidates (per method, P0 or D-S): e0 as in stage 3-B (bank and threshold from the 30 train specimens
under I01), e1 with per-image feature centring, e2 enrolled under I01 and one of two illumination groups
(two arms that swap the groups, so every illumination is unseen once), e3 both.

`val` reads train specimens only: outer specimen folds, a bank and a nested cross-fitted threshold from
the other folds, scores of the held-out specimens under all ten illuminations. `test` reads the sealed
M2AD test images (logged): e0 and the candidate `val` picked, and with --loop the closed-loop re-enrolment
arm on e0. Tables and verdicts: `defect_inspect.analyze_m2ad_enrol`. Rules: docs/experiments.md, stage 7.
"""

import argparse
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import paths
from .configs import PatchCoreConfig, get_config
from .ledger import git_commit, record_test_access
from .m2ad import (
    CATEGORIES,
    ILLUMINATIONS,
    N_FOLDS,
    REFERENCE,
    VIEWS,
    M2adCache,
    M2adRow,
    illumination_groups,
    read_meta,
    recal_specimens,
    specimen_folds,
)
from .run_m2ad import _MAP_SIZE, _same_specimens, _threshold_info, inspector_rows
from .run_m2ad import crossfit as _crossfit
from .splits import SealedTestError, path_key

METHODS = ("p0", "d-s")
CANDIDATES = ("e0", "e1", "e2", "e3")  # in order of simplicity (ties in `val` go to the earlier one)
CENTRE = {"e0": False, "e1": True, "e2": False, "e3": True}
MULTI = {"e0": False, "e1": False, "e2": True, "e3": True}
# Closed-loop arm (test only, on e0): n unlabelled images under the new illumination, k of them defects.
LOOP_N = 20
LOOP_TRIM = 0.2  # share of the batch dropped by the robust filter: the highest within-batch scores
LOOP_ARMS = (("trim", 0), ("trim", 1), ("trim", 2), ("none", 2))
INSPECTORS = tuple(f"{category}_{view}" for category in CATEGORIES for view in VIEWS)


def arms(candidate: str) -> list[tuple[str, tuple[str, ...]]]:
    """(arm, enrolled illuminations): one arm `ref` (I01 only) or the arms `A` and `B` (I01 + a group)."""
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate {candidate!r}; known: {CANDIDATES}")
    if not MULTI[candidate]:
        return [("ref", (REFERENCE,))]
    a, b = illumination_groups()
    return [("A", (REFERENCE, *a)), ("B", (REFERENCE, *b))]


def enrolled_matrix(candidate: str) -> np.ndarray:
    """bool [n_arms, 10]: which illuminations each arm enrols (columns in ILLUMINATIONS order)."""
    return np.array([[light in enrolled for light in ILLUMINATIONS] for _, enrolled in arms(candidate)])


def loop_keys() -> list[str]:
    """`<LL>:<filter>:<k>` for every new illumination and closed-loop arm, illuminations first."""
    return [f"{light}:{filt}:{k}" for light in ILLUMINATIONS if light != REFERENCE for filt, k in LOOP_ARMS]


def centred(extractor):
    from .backbones import CentredPatches

    return CentredPatches(extractor)


class Tools:
    """Bank building, scoring, cross-fitting and feature extraction with the options of one config."""

    def __init__(self, cfg: PatchCoreConfig, extractor, device: str):
        self.cfg, self.extractor, self.device = cfg, extractor, device

    def build(self, features):
        from .patchcore import build_bank

        return build_bank(features, self.cfg.coreset_ratio, seed=self.cfg.seed, device=self.device)

    def score(self, bank, images: np.ndarray) -> np.ndarray:
        from .patchcore import score_images

        return score_images(
            self.extractor,
            bank,
            images,
            batch_size=self.cfg.batch_size,
            device=self.device,
            reweight_k=self.cfg.reweight_k,
            sigma=self.cfg.sigma,
            map_size=_MAP_SIZE,
        ).image_scores

    def features(self, images: np.ndarray):
        from .patchcore import collect_features

        feats, (grid_h, grid_w) = collect_features(
            self.extractor, images, batch_size=self.cfg.batch_size, device=self.device
        )
        return feats.view(len(images), grid_h * grid_w, feats.shape[1])

    def crossfit(self, features, images: np.ndarray, folds: np.ndarray):
        """`run_m2ad.crossfit` with this config: (bank of all, cross-fitted scores, threshold, bank rows)."""
        return _crossfit(self.build, self.score, features, images, folds)


def _train_rows(rows: Sequence[M2adRow], category: str, view: str) -> tuple[dict, list[str]]:
    """Train rows of one inspector under every illumination (same specimens, sorted by name)."""
    train = {light: inspector_rows(rows, category, view, "train", light) for light in ILLUMINATIONS}
    names = [r.specimen for r in train[REFERENCE]]
    if not names:
        raise ValueError(f"no train images for {category} {view} under illumination {REFERENCE}")
    for light, picked in train.items():
        _same_specimens(picked, names, f"{category} {view} train, illumination {light}")
    return train, names


def _test_rows(rows: Sequence[M2adRow], category: str, view: str) -> tuple[dict, list[str]]:
    test = {light: inspector_rows(rows, category, view, "test", light) for light in ILLUMINATIONS}
    names = [r.specimen for r in test[REFERENCE]]
    if not names:
        raise ValueError(f"no test images for {category} {view} under illumination {REFERENCE}")
    for light, picked in test.items():
        _same_specimens(picked, names, f"{category} {view} test, illumination {light}")
    return test, names


def _peak_reset(device: str) -> bool:
    import torch

    on_cuda = str(device).startswith("cuda")
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    return on_cuda


def _peak_mb(on_cuda: bool) -> float:
    import torch

    return round(torch.cuda.max_memory_allocated() / 2**20, 1) if on_cuda else 0.0


def validate_inspector(
    tools: Tools,
    candidate: str,
    category: str,
    view: str,
    rows: Sequence[M2adRow],
    cache: M2adCache,
    out_dir: Path,
) -> dict:
    """Validation of one candidate on one inspector, train specimens only; writes its npz.

    For every arm and outer fold f: bank = the arm's illuminations of the specimens outside f, threshold =
    conformal threshold of their scores cross-fitted over the remaining folds (nested), and the specimens
    of f are scored under all ten illuminations by that bank.
    """
    import torch

    started = time.perf_counter()
    on_cuda = _peak_reset(tools.device)
    train, names = _train_rows(rows, category, view)
    fold_of = specimen_folds(names)
    folds = np.array([fold_of[name] for name in names], dtype=np.int64)
    arm_list = arms(candidate)
    needed = sorted({light for _, enrolled in arm_list for light in enrolled})
    images = {light: cache.images(train[light]) for light in ILLUMINATIONS}
    feats = {light: tools.features(images[light]) for light in needed}

    n_lights, n_spec = len(ILLUMINATIONS), len(names)
    scores = np.full((len(arm_list), n_lights, n_spec), np.nan, dtype=np.float32)
    thresholds = np.full((len(arm_list), N_FOLDS), np.nan, dtype=np.float64)
    cal_n = np.zeros((len(arm_list), N_FOLDS), dtype=np.int64)
    bank_rows: dict[str, dict] = {}
    for a, (arm, enrolled) in enumerate(arm_list):
        for f in range(N_FOLDS):
            held = np.flatnonzero(folds == f)
            if held.size == 0:
                continue
            keep = folds != f
            keep_t = torch.from_numpy(keep)
            sub_feats = torch.cat([feats[light][keep_t] for light in enrolled])
            sub_images = np.concatenate([images[light][keep] for light in enrolled])
            sub_folds = np.concatenate([folds[keep]] * len(enrolled))
            bank, _, thr, rows_f = tools.crossfit(sub_feats, sub_images, sub_folds)
            thresholds[a, f], cal_n[a, f] = thr.value, thr.n
            bank_rows[f"{arm}:{f}"] = rows_f
            shown = np.concatenate([images[light][held] for light in ILLUMINATIONS])
            scores[a][:, held] = tools.score(bank, shown).reshape(n_lights, held.size)
            del bank
    if np.isnan(scores).any() or np.isnan(thresholds).any():
        raise RuntimeError("validation did not cover every specimen and fold")

    np.savez(
        out_dir / f"{category}_{view}_{candidate}.npz",
        specimens=np.array(names),
        folds=folds,
        lights=np.array(ILLUMINATIONS),
        arms=np.array([arm for arm, _ in arm_list]),
        enrolled=enrolled_matrix(candidate),
        scores=scores,
        thresholds=thresholds,
        cal_n=cal_n,
    )
    return {
        "category": category,
        "view": view,
        "candidate": candidate,
        "specimens": n_spec,
        "arms": {arm: list(enrolled) for arm, enrolled in arm_list},
        "bank_rows": bank_rows,
        "peak_vram_mb": _peak_mb(on_cuda),
        "seconds": round(time.perf_counter() - started, 1),
    }


def sealed_inspector(
    tools: Tools,
    candidate: str,
    category: str,
    view: str,
    rows: Sequence[M2adRow],
    cache: M2adCache,
    out_dir: Path,
    *,
    allow_test: bool = False,
) -> dict:
    """One candidate on one inspector against the sealed test specimens; writes its npz.

    Per arm: bank of the arm's illuminations of all train specimens, conformal threshold of their
    scores cross-fitted over the specimen folds, test specimens scored under every illumination
    (one call of all test specimens per illumination, as in stage 3-B).
    """
    import torch

    if allow_test is not True:
        raise SealedTestError("the M2AD test images are read only with allow_test=True (--allow-test)")
    started = time.perf_counter()
    on_cuda = _peak_reset(tools.device)
    train, names = _train_rows(rows, category, view)
    fold_of = specimen_folds(names)
    folds = np.array([fold_of[name] for name in names], dtype=np.int64)
    arm_list = arms(candidate)
    needed = sorted({light for _, enrolled in arm_list for light in enrolled})
    train_images = {light: cache.images(train[light]) for light in needed}
    feats = {light: tools.features(train_images[light]) for light in needed}

    test, test_names = _test_rows(rows, category, view)
    labels = np.array([[r.label for r in test[light]] for light in ILLUMINATIONS], dtype=np.int8)
    scores = np.full((len(arm_list), len(ILLUMINATIONS), len(test_names)), np.nan, dtype=np.float32)
    thresholds = np.full(len(arm_list), np.nan, dtype=np.float64)
    cal_scores, info_arms = {}, {}
    test_images = {}
    for a, (arm, enrolled) in enumerate(arm_list):
        bank, oof, thr, bank_rows = tools.crossfit(
            torch.cat([feats[light] for light in enrolled]),
            np.concatenate([train_images[light] for light in enrolled]),
            np.concatenate([folds] * len(enrolled)),
        )
        thresholds[a] = thr.value
        cal_scores[f"cal_score_{arm}"] = oof
        info_arms[arm] = {
            "enrolled": list(enrolled),
            "threshold": _threshold_info(thr),
            "bank_rows": bank_rows,
        }
        for li, light in enumerate(ILLUMINATIONS):
            if light not in test_images:
                test_images[light] = cache.images(test[light])
            scores[a, li] = tools.score(bank, test_images[light])
        del bank
    if np.isnan(scores).any():
        raise RuntimeError("not every test image was scored")

    np.savez(
        out_dir / f"{category}_{view}_{candidate}.npz",
        specimens=np.array(test_names),
        object_anomaly=np.array([r.object_anomaly for r in test[REFERENCE]], dtype=np.int8),
        lights=np.array(ILLUMINATIONS),
        arms=np.array([arm for arm, _ in arm_list]),
        enrolled=enrolled_matrix(candidate),
        scores=scores,
        labels=labels,
        thresholds=thresholds,
        **cal_scores,
    )
    return {
        "category": category,
        "view": view,
        "candidate": candidate,
        "train_specimens": len(names),
        "test_specimens": len(test_names),
        "folds": {name: int(fold_of[name]) for name in names},
        "arms": info_arms,
        "peak_vram_mb": _peak_mb(on_cuda),
        "seconds": round(time.perf_counter() - started, 1),
    }


def loop_inspector(
    tools: Tools,
    category: str,
    view: str,
    rows: Sequence[M2adRow],
    cache: M2adCache,
    out_dir: Path,
    *,
    allow_test: bool = False,
) -> dict:
    """Closed-loop re-enrolment on e0 for one inspector; writes `<category>_<view>_loop.npz`.

    Under each new illumination L and arm (filter, k): a batch of LOOP_N unlabelled images = the first
    LOOP_N - k train specimens in SHA-256 order (normal) and the first k test specimens with a visible
    defect under L, in SHA-256 order of the specimen name. `trim` scores every batch image with a bank of
    the other batch folds (specimen folds of the batch) and drops the ceil(LOOP_TRIM * LOOP_N) highest;
    `none` keeps all. The kept images join the I01 bank and the cross-fitting (train specimens keep their
    fold, defect specimens get folds by their own SHA-256 rank). Test images under L are then scored; the
    enrolled defect specimens are excluded (label -1) from the detection rate.
    """
    import torch

    if allow_test is not True:
        raise SealedTestError("the closed-loop arm reads M2AD test images: allow_test=True (--allow-test)")
    started = time.perf_counter()
    on_cuda = _peak_reset(tools.device)
    train, names = _train_rows(rows, category, view)
    fold_of = specimen_folds(names)
    folds = np.array([fold_of[name] for name in names], dtype=np.int64)
    ref_images = cache.images(train[REFERENCE])
    ref_feats = tools.features(ref_images)
    test, test_names = _test_rows(rows, category, view)
    n_drop = math.ceil(LOOP_TRIM * LOOP_N)

    keys = loop_keys()
    scores = np.full((len(keys), len(test_names)), np.nan, dtype=np.float32)
    labels = np.zeros((len(keys), len(test_names)), dtype=np.int8)
    thresholds = np.full(len(keys), np.nan, dtype=np.float64)
    info: dict[str, dict] = {}
    for light in ILLUMINATIONS:
        if light == REFERENCE:
            continue
        test_images = cache.images(test[light])
        shown = np.array([r.label for r in test[light]], dtype=np.int8)
        defect_order = sorted((r for r in test[light] if r.label == 1), key=lambda r: path_key(r.specimen))
        for filt, k in LOOP_ARMS:
            normal_names = recal_specimens(names, LOOP_N - k)
            normal_rows = [train[light][names.index(name)] for name in normal_names]
            defect_rows = defect_order[:k]
            if len(defect_rows) < k:
                raise ValueError(f"{category} {view} I{light}: fewer than {k} visible defects to mix in")
            batch_rows = normal_rows + defect_rows
            batch_names = [r.specimen for r in batch_rows]
            batch_images = cache.images(batch_rows)
            batch_feats = tools.features(batch_images)
            is_defect = np.array([False] * len(normal_rows) + [True] * len(defect_rows))
            if filt == "trim":
                within = specimen_folds(batch_names)
                within_folds = np.array([within[name] for name in batch_names], dtype=np.int64)
                _, batch_oof, _, _ = tools.crossfit(batch_feats, batch_images, within_folds)
                dropped = np.argsort(-batch_oof, kind="stable")[:n_drop]
            elif filt == "none":
                dropped = np.zeros(0, dtype=np.int64)
            else:
                raise ValueError(f"unknown filter {filt!r}")
            kept = np.setdiff1d(np.arange(len(batch_rows)), dropped)
            defect_fold = specimen_folds([r.specimen for r in defect_rows]) if defect_rows else {}
            batch_folds = np.array(
                [defect_fold[name] if name in defect_fold else fold_of[name] for name in batch_names],
                dtype=np.int64,
            )
            bank, oof, thr, bank_rows = tools.crossfit(
                torch.cat([ref_feats, batch_feats[torch.from_numpy(kept)]]),
                np.concatenate([ref_images, batch_images[kept]]),
                np.concatenate([folds, batch_folds[kept]]),
            )
            key = f"{light}:{filt}:{k}"
            j = keys.index(key)
            scores[j] = tools.score(bank, test_images)
            labels[j] = shown
            enrolled_defects = {r.specimen for r in defect_rows}
            labels[j, [i for i, name in enumerate(test_names) if name in enrolled_defects]] = -1
            thresholds[j] = thr.value
            info[key] = {
                "batch": len(batch_rows),
                "defects_in_batch": int(is_defect.sum()),
                "dropped_defects": int(is_defect[dropped].sum()),
                "dropped_normals": int((~is_defect[dropped]).sum()),
                "enrolled": len(oof),
                "threshold": _threshold_info(thr),
                "bank_rows": bank_rows,
            }
            del bank
    np.savez(
        out_dir / f"{category}_{view}_loop.npz",
        specimens=np.array(test_names),
        object_anomaly=np.array([r.object_anomaly for r in test[REFERENCE]], dtype=np.int8),
        loop_keys=np.array(keys),
        scores=scores,
        labels=labels,
        thresholds=thresholds,
    )
    return {
        "category": category,
        "view": view,
        "loop": info,
        "peak_vram_mb": _peak_mb(on_cuda),
        "seconds": round(time.perf_counter() - started, 1),
    }


def _make_extractor(cfg: PatchCoreConfig, device: str):
    from .backbones import make_extractor

    return make_extractor(cfg.backbone, img_size=cfg.img_size).to(device).eval()


def read_pick(val_report: Path, method: str) -> str:
    """The candidate `analyze_m2ad_enrol val` picked for `method`."""
    with open(val_report, encoding="utf-8") as f:
        report = json.load(f)
    try:
        pick = report["methods"][method]["pick"]
    except (KeyError, TypeError):
        raise ValueError(f"{val_report} has no pick for {method!r}") from None
    if pick not in CANDIDATES:
        raise ValueError(f"{val_report}: unknown pick {pick!r} for {method!r}")
    return pick


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _inspectors(names: Sequence[str]) -> list[tuple[str, str]]:
    unknown = [name for name in names if name not in INSPECTORS]
    if unknown:
        raise ValueError(f"unknown inspectors {unknown}; known: {list(INSPECTORS)}")
    return [tuple(name.split("_")) for name in INSPECTORS if name in names]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("val", "test"):
        p = sub.add_parser(name)
        p.add_argument("--method", choices=METHODS, required=True)
        p.add_argument("--out", type=Path, default=None, help=f"default: outputs/m2ad-enrol-{name}-<method>")
        p.add_argument("--device", default="cuda")
        p.add_argument("--overwrite", action="store_true")
        p.add_argument("--jsons", type=Path, default=None, help="default: data/raw/m2ad/jsons.zip")
        p.add_argument("--cache", type=Path, default=None, help="cache directory (data/cache)")
    val = sub.choices["val"]
    val.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    val.add_argument("--inspectors", nargs="+", default=list(INSPECTORS), help="e.g. Motor_000")
    test = sub.choices["test"]
    test.add_argument("--pick-from", type=Path, required=True, help="reports/stage7/val.json")
    test.add_argument("--loop", action="store_true", help="also run the closed-loop arm on e0")
    test.add_argument("--allow-test", action="store_true", help="read the M2AD test images (logged)")
    test.add_argument("--stage", default="", help="stage label for the test ledger")
    test.add_argument("--note", default="")
    args = parser.parse_args(argv)

    cfg = get_config(args.method)
    out_dir = args.out or paths.OUTPUTS / f"m2ad-enrol-{args.command}-{args.method}"
    if (out_dir / "run.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")
    if args.command == "test":
        if not args.allow_test or not args.stage:
            parser.error("test reads the M2AD test images: it needs --allow-test and --stage")
        try:
            pick = read_pick(args.pick_from, args.method)
        except (OSError, ValueError) as err:
            parser.error(str(err))
        candidates = list(dict.fromkeys(["e0", pick]))
        inspectors = _inspectors(INSPECTORS)
    else:
        candidates = list(args.candidates)
        try:
            inspectors = _inspectors(args.inspectors)
        except ValueError as err:
            parser.error(str(err))

    # Everything that can fail without touching a test image comes before the ledger line.
    rows = read_meta(args.jsons or paths.RAW / "m2ad" / "jsons.zip")
    cache = M2adCache(args.cache or paths.CACHE, cfg.img_size)
    plain = _make_extractor(cfg, args.device)
    extractors = {False: plain, True: centred(plain).to(args.device).eval()}
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    if args.command == "test":
        note = args.note or f"stage 7 candidates {candidates}" + (", closed loop on e0" if args.loop else "")
        record_test_access(
            paths.TEST_LEDGER, stage=args.stage, config=f"m2ad-enrol-{args.method}", note=note, commit=commit
        )
    (out_dir / "run.json").unlink(missing_ok=True)

    summaries = []
    started = time.perf_counter()
    for candidate in candidates:
        tools = Tools(cfg, extractors[CENTRE[candidate]], args.device)
        for category, view in inspectors:
            if args.command == "val":
                info = validate_inspector(tools, candidate, category, view, rows, cache, out_dir)
            else:
                info = sealed_inspector(
                    tools, candidate, category, view, rows, cache, out_dir, allow_test=args.allow_test
                )
            summaries.append(info)
            print(json.dumps({k: info[k] for k in ("category", "view", "candidate", "seconds")}), flush=True)
    loop = []
    if args.command == "test" and args.loop:
        tools = Tools(cfg, extractors[False], args.device)
        for category, view in inspectors:
            info = loop_inspector(tools, category, view, rows, cache, out_dir, allow_test=args.allow_test)
            loop.append(info)
            print(json.dumps({"loop": f"{category}_{view}", "seconds": info["seconds"]}), flush=True)

    a, b = illumination_groups()
    _write_json(
        out_dir / "run.json",
        {
            "command": args.command,
            "method": args.method,
            "config": asdict(cfg),
            "commit": commit,
            # Coreset selection differs between devices at float rounding level.
            "device": args.device,
            "candidates": candidates,
            "inspectors": [f"{category}_{view}" for category, view in inspectors],
            "groups": {"A": list(a), "B": list(b)},
            "loop": {"n": LOOP_N, "trim": LOOP_TRIM, "arms": [list(arm) for arm in LOOP_ARMS]}
            if loop
            else None,
            "runs": summaries,
            "loop_runs": loop,
            "total_s": round(time.perf_counter() - started, 1),
        },
    )


if __name__ == "__main__":
    main()
