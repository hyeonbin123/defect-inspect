"""Score the evaluation set and the cross-fitted pool with the exported CPU pipeline.

The torch pipeline (GPU fp16) produced the banks and the thresholds of a `run_grid --save-banks` run. This
run repeats the scoring with onnxruntime on the CPU (fp32 and INT8 models) against the same banks, so
that the drift of scores, thresholds and false alarm rates caused by the export can be measured.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import paths
from .cache import ImageCache
from .inspector import PRECISIONS, Inspector
from .ledger import git_commit, record_test_access
from .run_grid import ratio_name, ratio_rows
from .splits import ManifestRow, pool_folds, read_manifest, select


def run_category(
    session,
    meta: dict,
    grid_dir: Path,
    ratio: float,
    protocol: str,
    category: str,
    manifest: list[ManifestRow],
    cache: ImageCache,
    allow_test: bool,
) -> dict[str, np.ndarray]:
    """Evaluation scores under the full bank prefix and cross-fitted pool scores, all from `session`."""
    pool = select(manifest, protocol=protocol, part="pool_normal", category=category)
    eval_normal = select(
        manifest, protocol=protocol, part="eval_normal", category=category, allow_test=allow_test
    )
    eval_defect = select(
        manifest, protocol=protocol, part="eval_defect", category=category, allow_test=allow_test
    )
    eval_rows = eval_normal + eval_defect
    folds = np.array([r.fold for r in pool], dtype=np.int64)
    with open(grid_dir / "banks" / f"{category}.json", encoding="utf-8") as f:
        sources = json.load(f)["n_features"]

    def inspector_for(bank_name: str) -> Inspector:
        bank = np.load(grid_dir / "banks" / f"{category}_{bank_name}.npy")
        return Inspector(session, bank[: ratio_rows(sources[bank_name], ratio)], meta)

    full = inspector_for("full")
    eval_scores = full.scores(cache.images(eval_rows))
    oof = np.full(len(pool), np.nan, dtype=np.float32)
    pool_images = cache.images(pool)
    for fold in pool_folds(protocol):
        held = np.flatnonzero(folds == fold)
        oof[held] = inspector_for(f"minus_fold_{fold}").scores(pool_images[held])
    if np.isnan(oof).any():
        raise RuntimeError("cross-fitting did not cover every pool image")

    with np.load(grid_dir / ratio_name(ratio) / f"{category}.npz") as z:
        if list(z["eval_images"]) != [r.image for r in eval_rows] or list(z["pool_images"]) != [
            r.image for r in pool
        ]:
            raise ValueError(f"{category}: the grid run scored other images")
        torch_eval, torch_oof = z["eval_score_full"], z["pool_score_oof"]
    return {
        "eval_images": np.array([r.image for r in eval_rows]),
        "eval_labels": np.array([0] * len(eval_normal) + [1] * len(eval_defect), dtype=np.int8),
        "eval_score_full": eval_scores,
        "pool_images": np.array([r.image for r in pool]),
        "pool_folds": folds,
        "pool_score_oof": oof,
        "torch_eval_score": torch_eval,
        "torch_pool_score_oof": torch_oof,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", type=Path, required=True, help="a run_grid --save-banks run directory")
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--artifacts", type=Path, required=True, help="artifact set built by export.py")
    parser.add_argument("--precision", nargs="+", default=["fp32", "int8"], choices=sorted(PRECISIONS))
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--out", type=Path, default=None, help="default: outputs/export-<artifact set>-<protocol>"
    )
    parser.add_argument("--allow-test", action="store_true", help="read the sealed test set (logged)")
    parser.add_argument(
        "--stage", default="", help="stage label for the test ledger (required with --allow-test)"
    )
    parser.add_argument("--note", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    import onnxruntime as ort

    with open(args.grid / "run.json", encoding="utf-8") as f:
        grid = json.load(f)
    protocol = grid["protocol"]
    if protocol == "test" and (not args.allow_test or not args.stage):
        parser.error("the grid run is a test-protocol run: --allow-test and --stage are needed")
    if not grid.get("save_banks"):
        parser.error(f"{args.grid} was run without --save-banks")
    categories = args.categories or grid["categories"]
    name = args.artifacts.name
    out_dir = args.out or paths.OUTPUTS / f"export-{name}-{protocol}"
    if (out_dir / "run.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")

    # Everything that can fail without touching a test image comes before the ledger line.
    manifest = read_manifest(paths.VISA_MANIFEST)
    metas, sessions = {}, {}
    for category in categories:
        with open(args.artifacts / category / "meta.json", encoding="utf-8") as f:
            metas[category] = json.load(f)
    size = metas[categories[0]]["img_size"]
    cache = ImageCache(paths.CACHE, size)
    for precision in args.precision:
        options = ort.SessionOptions()
        if args.threads is not None:
            options.intra_op_num_threads = args.threads
        model = args.artifacts / PRECISIONS[precision]
        sessions[precision] = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
    commit = git_commit()
    if protocol == "test":
        record_test_access(
            paths.TEST_LEDGER, stage=args.stage, config=f"export-{name}", note=args.note, commit=commit
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").unlink(missing_ok=True)

    started = time.perf_counter()
    timings = []
    for precision in args.precision:
        (out_dir / precision).mkdir(parents=True, exist_ok=True)
        for category in categories:
            t0 = time.perf_counter()
            arrays = run_category(
                sessions[precision],
                metas[category],
                args.grid,
                args.ratio,
                protocol,
                category,
                manifest,
                cache,
                args.allow_test,
            )
            np.savez(out_dir / precision / f"{category}.npz", **arrays)
            line = {
                "precision": precision,
                "category": category,
                "seconds": round(time.perf_counter() - t0, 1),
            }
            line["images"] = int(len(arrays["eval_images"]) + len(arrays["pool_images"]))
            timings.append(line)
            print(json.dumps(line), flush=True)
    run = {
        "artifacts": name,
        "grid": args.grid.name,
        "ratio": args.ratio,
        "protocol": protocol,
        "precisions": list(args.precision),
        "categories": list(categories),
        "threads": args.threads,
        "commit": commit,
        "runs": timings,
        "total_s": round(time.perf_counter() - started, 1),
    }
    with open(out_dir / "run.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
