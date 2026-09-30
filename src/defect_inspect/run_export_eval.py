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
from .inspector import PRECISIONS, Inspector, check_meta, check_model_io
from .ledger import git_commit, record_test_access
from .run_grid import ratio_name, ratio_rows
from .splits import ManifestRow, pool_folds, read_manifest, select

# Artifact meta values that must equal the grid run's: they decide the scores of both pipelines.
_CONFIG_KEYS = ("backbone", "img_size", "reweight_k", "sigma")


def _read_json(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as err:
        raise ValueError(f"{path} is missing or not readable JSON ({err})") from err


def _npy_shape(path: Path) -> tuple[int, ...]:
    """Shape of a .npy file from its header; the data is not read."""
    try:
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            read_header = (
                np.lib.format.read_array_header_1_0
                if version == (1, 0)
                else np.lib.format.read_array_header_2_0
            )
            return tuple(read_header(f)[0])
    except (OSError, ValueError) as err:
        raise ValueError(f"{path} is missing or not a readable array ({err})") from err


def _bank_names(protocol: str) -> list[str]:
    return ["full", *(f"minus_fold_{fold}" for fold in pool_folds(protocol))]


def _rows(
    manifest: list[ManifestRow], protocol: str, category: str, allow_test: bool
) -> tuple[list[ManifestRow], list[ManifestRow], list[ManifestRow]]:
    """(pool normals, evaluation normals, evaluation defects) of a category, in manifest order."""
    pool = select(manifest, protocol=protocol, part="pool_normal", category=category)
    eval_normal = select(
        manifest, protocol=protocol, part="eval_normal", category=category, allow_test=allow_test
    )
    eval_defect = select(
        manifest, protocol=protocol, part="eval_defect", category=category, allow_test=allow_test
    )
    return pool, eval_normal, eval_defect


def _check_same_images(npz_path: Path, category: str, pool: list[ManifestRow], eval_rows: list[ManifestRow]):
    """The grid run's torch scores (evaluation, cross-fitted pool), checked to be of these images."""
    try:
        with np.load(npz_path) as z:
            eval_images, pool_images = z["eval_images"].tolist(), z["pool_images"].tolist()
            torch_eval, torch_oof = z["eval_score_full"], z["pool_score_oof"]
    except (OSError, KeyError, ValueError) as err:
        raise ValueError(f"{npz_path} is missing or not a grid run's score file ({err!r})") from err
    if eval_images != [r.image for r in eval_rows] or pool_images != [r.image for r in pool]:
        raise ValueError(f"{category}: the grid run scored other images than the manifest selects")
    if len(torch_eval) != len(eval_rows) or len(torch_oof) != len(pool):
        raise ValueError(f"{category}: {npz_path.name} has scores for another number of images")
    return torch_eval, torch_oof


def check_inputs(
    grid_dir: Path,
    grid: dict,
    ratio: float,
    artifacts: Path,
    precisions: list[str],
    categories: list[str],
    manifest: list[ManifestRow],
    allow_test: bool,
) -> dict[str, dict]:
    """Check everything a run needs, without reading an image; returns the artifact meta per category.

    Raises `ValueError` naming the first problem: a ratio the grid run does not have, missing models,
    banks or score files, artifacts that were built from another setting or ratio, image lists that
    differ from the grid run's. A test-protocol run calls this before it writes the ledger line.
    """
    protocol = grid["protocol"]
    if ratio_name(ratio) not in {ratio_name(r) for r in grid["ratios"]}:
        raise ValueError(f"ratio {ratio:g} is not one of the grid run's ratios {grid['ratios']}")
    unknown = [c for c in categories if c not in grid["categories"]]
    if unknown:
        raise ValueError(f"the grid run has no categories {unknown}")
    for precision in precisions:
        model = artifacts / PRECISIONS[precision]
        if not model.is_file():
            hint = ""
            if precision == "int8":
                reason = None
                if (artifacts / "artifacts.json").is_file():
                    reason = _read_json(artifacts / "artifacts.json").get("int8", {}).get("reason")
                hint = f" (export: {reason})" if reason else ""
                hint += "; pass --precision fp32 to score the fp32 model only"
            raise ValueError(f"the artifact set has no {model.name}{hint}")

    ratio_dir = grid_dir / ratio_name(ratio)
    ratio_run = _read_json(ratio_dir / "run.json")
    config = ratio_run["config"]
    infos = {info["category"]: info for info in ratio_run["categories"]}
    metas = {}
    for category in categories:
        if category not in infos:
            raise ValueError(f"{ratio_dir / 'run.json'} has no category {category}")
        meta = _read_json(artifacts / category / "meta.json")
        try:
            check_meta(meta)
        except ValueError as err:
            raise ValueError(f"{artifacts / category / 'meta.json'}: {err}") from err
        expected = {key: config[key] for key in _CONFIG_KEYS}
        expected.update(dim=infos[category]["dim"], grid=list(infos[category]["grid"]))
        found = {key: list(meta[key]) if key == "grid" else meta[key] for key in expected}
        if found != expected:
            differing = {k: (found[k], expected[k]) for k in expected if found[k] != expected[k]}
            raise ValueError(
                f"{category}: the artifact was built from another setting than the grid run at ratio "
                f"{ratio:g} (artifact, grid run): {differing}"
            )
        source = meta.get("source")
        if source is not None:
            if ratio_name(source.get("ratio", ratio)) != ratio_name(ratio):
                raise ValueError(
                    f"{category}: the artifact was built at ratio {source['ratio']:g}, not {ratio:g}"
                )
            if source.get("protocol", protocol) != protocol:
                raise ValueError(
                    f"{category}: the artifact was built from a {source['protocol']} run, "
                    f"the grid run is a {protocol} run"
                )

        sources = _read_json(grid_dir / "banks" / f"{category}.json")["n_features"]
        for bank_name in _bank_names(protocol):
            if bank_name not in sources:
                raise ValueError(f"banks/{category}.json does not list the bank {bank_name}")
            path = grid_dir / "banks" / f"{category}_{bank_name}.npy"
            shape = _npy_shape(path)
            rows = ratio_rows(sources[bank_name], ratio)
            if len(shape) != 2 or shape[1] != meta["dim"] or shape[0] < rows:
                raise ValueError(
                    f"{path} has shape {shape}; ratio {ratio:g} needs {rows} rows of dim {meta['dim']}"
                )

        pool, eval_normal, eval_defect = _rows(manifest, protocol, category, allow_test)
        _check_same_images(ratio_dir / f"{category}.npz", category, pool, eval_normal + eval_defect)
        metas[category] = meta
    return metas


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
    pool, eval_normal, eval_defect = _rows(manifest, protocol, category, allow_test)
    eval_rows = eval_normal + eval_defect
    folds = np.array([r.fold for r in pool], dtype=np.int64)
    # Before any image is read: the torch scores this run is compared with must be of the same images.
    torch_eval, torch_oof = _check_same_images(
        grid_dir / ratio_name(ratio) / f"{category}.npz", category, pool, eval_rows
    )
    sources = _read_json(grid_dir / "banks" / f"{category}.json")["n_features"]

    def inspector_for(bank_name: str) -> Inspector:
        bank = np.load(grid_dir / "banks" / f"{category}_{bank_name}.npy")
        rows = ratio_rows(sources[bank_name], ratio)
        if bank.shape[0] < rows:  # a ratio above the grid's largest would silently use the whole bank
            raise ValueError(
                f"{category}_{bank_name}.npy has {bank.shape[0]} rows, ratio {ratio:g} needs {rows}"
            )
        return Inspector(session, bank[:rows], meta)

    full = inspector_for("full")  # built before the first image is read
    eval_scores = full.scores(cache.images(eval_rows))
    del full
    oof = np.full(len(pool), np.nan, dtype=np.float32)
    pool_images = cache.images(pool)
    for fold in pool_folds(protocol):
        held = np.flatnonzero(folds == fold)
        oof[held] = inspector_for(f"minus_fold_{fold}").scores(pool_images[held])
    if np.isnan(oof).any():
        raise RuntimeError("cross-fitting did not cover every pool image")

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

    try:
        grid = _read_json(args.grid / "run.json")
    except ValueError as err:
        parser.error(str(err))
    protocol = grid["protocol"]
    if protocol == "test" and (not args.allow_test or not args.stage):
        parser.error("the grid run is a test-protocol run: --allow-test and --stage are needed")
    if not grid.get("save_banks"):
        parser.error(f"{args.grid} was run without --save-banks")
    categories = args.categories or grid["categories"]
    precisions = list(dict.fromkeys(args.precision))
    name = args.artifacts.name
    out_dir = args.out or paths.OUTPUTS / f"export-{name}-{protocol}"
    if (out_dir / "run.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")

    # Everything that can fail without touching a test image comes before the ledger line: the ratio,
    # the files of the grid run and of the artifact set, the image lists, the models and the cache.
    manifest = read_manifest(paths.VISA_MANIFEST)
    sessions = {}
    try:
        metas = check_inputs(
            args.grid, grid, args.ratio, args.artifacts, precisions, categories, manifest, args.allow_test
        )
        for precision in precisions:
            options = ort.SessionOptions()
            if args.threads is not None:
                options.intra_op_num_threads = args.threads
            model = args.artifacts / PRECISIONS[precision]
            try:
                sessions[precision] = ort.InferenceSession(
                    str(model), sess_options=options, providers=["CPUExecutionProvider"]
                )
            except Exception as err:  # onnxruntime has its own exception types for files it cannot load
                raise ValueError(f"{model} is not a loadable ONNX model: {err}") from err
            for category in categories:
                try:
                    check_model_io(sessions[precision], metas[category])
                except ValueError as err:
                    raise ValueError(f"{model.name} does not fit the artifact of {category}: {err}") from err
    except ValueError as err:
        parser.error(str(err))
    size = metas[categories[0]]["img_size"]
    try:
        cache = ImageCache(paths.CACHE, size)
    except FileNotFoundError:
        command = f"python -m defect_inspect.cache --size {size}"
        parser.error(f"no {size} px image cache: build it with `{command}`")
    commit = git_commit()
    if protocol == "test":
        record_test_access(
            paths.TEST_LEDGER, stage=args.stage, config=f"export-{name}", note=args.note, commit=commit
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").unlink(missing_ok=True)

    started = time.perf_counter()
    timings = []
    for precision in precisions:
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
        "precisions": precisions,
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
