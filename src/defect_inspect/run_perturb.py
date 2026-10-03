"""Score the evaluation set of a finished run again under the synthetic conditions of stage 3.

Nothing is trained or calibrated here: the memory bank (PatchCore) or the model (Dinomaly) and the scores
the thresholds are fixed from all come from the source run. Per category this writes
`<out>/<category>.npz` with the image scores under every condition (clean first). Tables are made by
`defect_inspect.analyze_perturb`.
"""

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import paths
from .cache import ImageCache
from .conditions import CLEAN, apply_condition, condition_names
from .configs import get_config, same_config
from .download import sha256_file
from .ledger import git_commit, record_test_access
from .splits import ManifestRow, read_manifest, select
from .visa import CATEGORIES

# p0-c and d-s-c: the stage 7 (E1) centred variants, dev protocol only by the rules.
PATCHCORE_METHODS = ("p0", "d-s", "p0-c", "d-s-c")
DINOMALY = "dm"
METHODS = (*PATCHCORE_METHODS, DINOMALY)
# The clean scores must reproduce the source run's scores within this relative difference.
CLEAN_RTOL = 1e-3

Scorer = Callable[[np.ndarray], np.ndarray]  # uint8 images [N, S, S, 3] -> image scores [N]


class CleanScoreMismatch(RuntimeError):
    """The clean scores differ from the source run: its thresholds do not belong to what is scored here."""


@dataclass(frozen=True)
class Source:
    """What a perturbation run takes from one category of the run whose thresholds it reuses."""

    eval_images: np.ndarray  # str [N], image paths of the evaluation set
    eval_score: np.ndarray  # float32 [N], scores of the evaluated model on the unchanged images
    cal_score: np.ndarray  # float32, scores the thresholds are fixed from (may be empty)


def source_keys(method: str) -> tuple[str, str]:
    """npz keys of (evaluation scores, calibration scores) in the source run of `method`."""
    if method == DINOMALY:
        return "eval_score", "cal_score"  # common stage 2 format
    return "eval_score_full", "pool_score_oof"  # run_patchcore: full bank, cross-fitted pool scores


def load_source(source_dir: Path, category: str, method: str) -> Source:
    """One category of the source run, checked as far as the file alone allows (no image is needed)."""
    score_key, cal_key = source_keys(method)
    path = Path(source_dir) / f"{category}.npz"
    with np.load(path) as z:
        missing = [k for k in ("eval_images", score_key, cal_key) if k not in z.files]
        if missing:
            raise KeyError(f"{path} has no {missing}: it is not a finished {method} run")
        source = Source(
            eval_images=z["eval_images"],
            eval_score=z[score_key].astype(np.float32),
            cal_score=z[cal_key].astype(np.float32),
        )
    n = len(source.eval_images)
    if source.eval_score.shape != (n,) or source.cal_score.ndim != 1:
        raise ValueError(
            f"{path}: {score_key} {source.eval_score.shape} and {cal_key} {source.cal_score.shape} are "
            f"not one score per image for {n} evaluation images and a flat list of calibration scores"
        )
    if not (np.isfinite(source.eval_score).all() and np.isfinite(source.cal_score).all()):
        raise ValueError(f"{path}: {score_key} or {cal_key} is not finite")
    return source


def evaluation_rows(
    protocol: str, category: str, manifest: list[ManifestRow], source: Source, allow_test: bool
) -> tuple[list[ManifestRow], list[ManifestRow]]:
    """The (normal, defect) evaluation rows of a category, which must be the images `source` scored.

    Reads the manifest only, so a source that does not fit is found before any image is touched.
    """
    eval_normal = select(
        manifest, protocol=protocol, part="eval_normal", category=category, allow_test=allow_test
    )
    eval_defect = select(
        manifest, protocol=protocol, part="eval_defect", category=category, allow_test=allow_test
    )
    rows = eval_normal + eval_defect
    if not rows:
        raise ValueError(f"{category}: the manifest has no evaluation images for the {protocol} protocol")
    if [r.image for r in rows] != [str(image) for image in source.eval_images]:
        raise ValueError(
            f"{category}: the source run did not evaluate the {protocol} evaluation set of this manifest"
        )
    return eval_normal, eval_defect


def max_rel_diff(scores: np.ndarray, reference: np.ndarray) -> float:
    """Largest `|score - reference| / |reference|` over the images (0.0 for an empty set)."""
    a = np.asarray(scores, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"score shapes differ: {a.shape} and {b.shape}")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), np.finfo(np.float64).tiny)))


def patchcore_scorer(cfg, extractor, bank, device: str) -> Scorer:
    """Image scores against a saved memory bank, with the batch size and scoring options of `cfg`."""
    from .patchcore import score_images

    def score(images: np.ndarray) -> np.ndarray:
        return score_images(
            extractor,
            bank,
            images,
            batch_size=cfg.batch_size,
            device=device,
            reweight_k=cfg.reweight_k,
            sigma=cfg.sigma,
        ).image_scores

    return score


def _dinomaly():
    """`run_dinomaly`, imported only when the dm method is used (building its model needs anomalib)."""
    from . import run_dinomaly

    return run_dinomaly


def load_dinomaly(path: Path):
    """The model written by `run_dinomaly train`, on the CPU in eval mode."""
    dm = _dinomaly()
    loader = getattr(dm, "load_model", None)
    if loader is not None:
        return loader(path)[0]
    # The file layout of contract-stage2: the trainable state with the encoder name and step count.
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = dm.build_model(saved["encoder"])
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(saved["state"]) != expected:
        odd = sorted(set(saved["state"]) ^ expected)
        raise ValueError(f"{path} does not hold the trainable parameters of {saved['encoder']}: {odd[:3]}")
    model.load_state_dict(saved["state"], strict=False)
    return model.eval()


def dinomaly_scorer(model, device: str, batch_size: int, amp: bool) -> Scorer:
    predict = _dinomaly().predict

    def score(images: np.ndarray) -> np.ndarray:
        return predict(model, images, batch_size=batch_size, device=device, amp=amp).image_scores

    return score


def select_conditions(requested: list[str] | None = None) -> list[str]:
    """The conditions to score, in the registered order: all of them, or `requested` (clean included)."""
    names = condition_names()
    if requested is None:
        return names
    unknown = sorted(set(requested) - set(names))
    if unknown:
        raise ValueError(f"unknown conditions {unknown}; known: {names}")
    if CLEAN not in requested:
        raise ValueError(f"the {CLEAN!r} condition is needed: it checks the source run's scores")
    return [name for name in names if name in requested]


def run_category(
    protocol: str,
    category: str,
    manifest: list[ManifestRow],
    cache: ImageCache,
    score: Scorer,
    source: Source,
    allow_test: bool,
    out_dir: Path,
    *,
    rtol: float = CLEAN_RTOL,
    conditions: list[str] | None = None,
) -> dict:
    """Score one category's evaluation set under every condition and write `<category>.npz`.

    `conditions` defaults to all of them; a subset keeps the registered order and must hold the clean
    condition. That comes first and must reproduce `source.eval_score` (CleanScoreMismatch otherwise,
    before any other condition is scored).
    """
    eval_normal, eval_defect = evaluation_rows(protocol, category, manifest, source, allow_test)
    rows = eval_normal + eval_defect
    names = [r.image for r in rows]
    labels = np.array([0] * len(eval_normal) + [1] * len(eval_defect), dtype=np.int8)
    conditions = select_conditions(conditions)
    if conditions[0] != CLEAN:
        raise RuntimeError("the clean condition must come first")

    images = cache.images(rows)
    scores = np.empty((len(conditions), len(rows)), dtype=np.float32)
    clean_diff = float("nan")
    perturb_s = score_s = 0.0
    for k, name in enumerate(conditions):
        t0 = time.perf_counter()
        changed = apply_condition(images, name)
        t1 = time.perf_counter()
        result = np.asarray(score(changed), dtype=np.float32)
        t2 = time.perf_counter()
        perturb_s += t1 - t0
        score_s += t2 - t1
        if result.shape != (len(rows),):
            raise ValueError(f"{category}/{name}: expected {len(rows)} scores, got shape {result.shape}")
        if not np.isfinite(result).all():
            raise FloatingPointError(f"{category}/{name}: non-finite image scores")
        scores[k] = result
        if name == CLEAN:
            clean_diff = max_rel_diff(result, source.eval_score)
            if not clean_diff < rtol:
                raise CleanScoreMismatch(
                    f"{category}: the clean scores differ from the source run by up to {clean_diff:.3g} "
                    f"(relative; limit {rtol:g}). The bank or model, or the images, are not the ones the "
                    "source run's thresholds were fixed with."
                )

    np.savez(
        out_dir / f"{category}.npz",
        eval_images=np.array(names),
        eval_labels=labels,
        conditions=np.array(conditions),
        scores=scores,
        cal_score=np.asarray(source.cal_score, dtype=np.float32),
    )
    return {
        "category": category,
        "eval_normal": len(eval_normal),
        "eval_defect": len(eval_defect),
        "cal": int(len(source.cal_score)),
        "clean_max_rel_diff": clean_diff,
        "timing": {"perturb_s": round(perturb_s, 2), "score_s": round(score_s, 2)},
    }


def _bank_rows(path: Path) -> int:
    """Number of rows of a saved memory bank, read from the file header only."""
    bank = np.load(path, mmap_mode="r")
    rows = int(bank.shape[0])
    del bank  # unmaps: Windows cannot replace a file that is still mapped
    return rows


def _same_dir(a: Path, b: Path) -> bool:
    """Whether two paths name one directory (also through `..`, links or another spelling of the case)."""
    if a.exists() and b.exists():
        return a.samefile(b)
    return a.resolve() == b.resolve()


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--protocol", choices=["dev", "test"], required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="finished run whose bank or model scores are reused (default: outputs/<method>-<protocol>)",
    )
    parser.add_argument("--model", type=Path, default=None, help="dm only (default: outputs/dm/model.pt)")
    parser.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    parser.add_argument("--out", type=Path, default=None, help="default: outputs/perturb-<method>-<protocol>")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-test", action="store_true", help="read the sealed test set (logged)")
    parser.add_argument(
        "--stage", default="", help="stage label for the test ledger (required with --allow-test)"
    )
    parser.add_argument("--note", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--conditions", nargs="+", default=None, help="a subset in the registered order (clean included)"
    )
    args = parser.parse_args(argv)

    if args.protocol == "test" and (not args.allow_test or not args.stage):
        parser.error("--protocol test needs --allow-test and --stage")
    try:
        conditions = select_conditions(args.conditions)
    except ValueError as err:
        parser.error(str(err))
    if args.model is not None and args.method != DINOMALY:
        parser.error("--model is only used with --method dm")
    source_dir = args.source or paths.OUTPUTS / f"{args.method}-{args.protocol}"
    out_dir = args.out or paths.OUTPUTS / f"perturb-{args.method}-{args.protocol}"
    if _same_dir(out_dir, source_dir):
        # Both runs write run.json and <category>.npz: the source run would be replaced file by file.
        parser.error(f"--out is the source run {source_dir}: the perturbation run would overwrite it")
    if (out_dir / "run.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")
    if not (source_dir / "run.json").exists():
        parser.error(f"{source_dir} holds no finished run (run.json is missing); see --source")
    with open(source_dir / "run.json", encoding="utf-8") as f:
        source_meta = json.load(f)
    if source_meta.get("protocol") != args.protocol:
        parser.error(
            f"{source_dir} is a {source_meta.get('protocol')!r} run: thresholds of the {args.protocol} "
            "protocol are needed"
        )

    commit = git_commit()
    manifest = read_manifest(paths.VISA_MANIFEST)
    known = {r.category for r in manifest}
    unknown = [c for c in args.categories if c not in known]
    if unknown or not args.categories:
        parser.error(f"unknown or missing categories {unknown}; the manifest has {sorted(known)}")
    absent = [c for c in args.categories if not (source_dir / f"{c}.npz").exists()]
    if absent:
        parser.error(f"{source_dir} has no scores for {absent}")
    # Every category's source scores and image list, not only the first one's: none of this needs an
    # image, so a source that does not fit must stop the run before a test read is recorded or made.
    sources: dict[str, Source] = {}
    for category in args.categories:
        try:
            sources[category] = load_source(source_dir, category, args.method)
            evaluation_rows(args.protocol, category, manifest, sources[category], args.allow_test)
        except (KeyError, ValueError) as err:
            parser.error(str(err.args[0]))

    import torch

    source_config = source_meta.get("config") or {}
    if args.method == DINOMALY:
        dm = _dinomaly()
        img_size = int(dm.IMG_SIZE)
        model_path = args.model or paths.OUTPUTS / DINOMALY / "model.pt"
        if not model_path.exists():
            parser.error(f"{model_path} does not exist: run `python -m defect_inspect.run_dinomaly train`")
        model_sha256 = sha256_file(model_path)
        if source_config.get("model_sha256") not in (None, model_sha256):
            parser.error(f"{model_path} is not the model that {source_dir} was scored with (sha256 differs)")
        # Same batches and precision as the source run, so that the clean scores can be compared.
        batch_size = int(source_config.get("batch_size", 32))
        amp = bool(source_config.get("amp", True))
        model = load_dinomaly(model_path).to(args.device)
        scorer = dinomaly_scorer(model, args.device, batch_size, amp)
        config = {
            "name": DINOMALY,
            "img_size": img_size,
            "model": str(model_path),
            "model_sha256": model_sha256,
            "batch_size": batch_size,
            "amp": amp,
        }

        def scorer_for(category: str) -> Scorer:
            return scorer
    else:
        cfg = get_config(args.method)
        img_size = cfg.img_size
        if not same_config(source_config, cfg):
            parser.error(
                f"{source_dir} was not made with the registered config {cfg.name!r}: {source_config}"
            )
        no_bank = [c for c in args.categories if not (source_dir / f"{c}_bank.npy").exists()]
        if no_bank:
            parser.error(
                f"{source_dir} has no memory bank (<category>_bank.npy) for {no_bank}: make the source run "
                f"with `python -m defect_inspect.run_patchcore --config {cfg.name} --protocol "
                f"{args.protocol} --save-bank`"
            )
        # A bank left over from another run is caught here, before the test images are read.
        recorded = {
            info.get("category"): (info.get("bank_rows") or {}).get("full")
            for info in source_meta.get("categories", [])
        }
        stale = [
            c
            for c in args.categories
            if recorded.get(c) is not None and _bank_rows(source_dir / f"{c}_bank.npy") != recorded[c]
        ]
        if stale:
            parser.error(
                f"the memory bank of {stale} in {source_dir} does not have the number of rows that its "
                "run.json records: it was written by another run"
            )
        from .backbones import make_extractor

        extractor = make_extractor(cfg.backbone, img_size=cfg.img_size, centre=cfg.centre)
        extractor = extractor.to(args.device).eval()
        config = asdict(cfg)

        def scorer_for(category: str) -> Scorer:
            bank = torch.from_numpy(np.load(source_dir / f"{category}_bank.npy"))
            return patchcore_scorer(cfg, extractor, bank, args.device)

    cache = ImageCache(paths.CACHE, img_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").unlink(missing_ok=True)  # a run that is being redone is not a finished run
    if args.protocol == "test":
        # The source run (config, protocol, every category's scores, image list and bank or the model)
        # has been checked without an image: the read starts here.
        record_test_access(
            paths.TEST_LEDGER,
            stage=args.stage,
            config=f"perturb-{args.method}",
            note=args.note,
            commit=commit,
        )

    cuda = torch.device(args.device).type == "cuda"
    summaries = []
    started = time.perf_counter()
    for category in args.categories:
        t0 = time.perf_counter()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        info = run_category(
            args.protocol,
            category,
            manifest,
            cache,
            scorer_for(category),
            sources[category],
            args.allow_test,
            out_dir,
            conditions=conditions,
        )
        info["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else 0.0
        info["seconds"] = round(time.perf_counter() - t0, 1)
        summaries.append(info)
        print(json.dumps(info, ensure_ascii=False), flush=True)

    run = {
        "method": args.method,
        "protocol": args.protocol,
        "commit": commit,
        "device": args.device,
        "source": str(source_dir),
        "source_commit": source_meta.get("commit"),
        "source_device": source_meta.get("device"),
        "config": config,
        "conditions": conditions,
        "clean_rtol": CLEAN_RTOL,
        "categories": summaries,
        "total_s": round(time.perf_counter() - started, 1),
    }
    _write_json(out_dir / "run.json", run)


if __name__ == "__main__":
    main()
