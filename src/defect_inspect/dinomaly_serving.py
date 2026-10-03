"""The CPU pipeline of a Dinomaly model: ONNX export, scoring with onnxruntime, thresholds for the service.

`export` writes an artifact set: `model_fp32.onnx` (image in; `score` and `map` out, see
`dinomaly_model.ServingGraph`), optionally `model_int8_dynamic.onnx` (dynamic INT8 quantization of the
MatMul nodes), and per category a `meta.json` (`kind: reconstruction`). A trained model comes from
`run_dinomaly train`; `--untrained` exports a freshly initialised model of a config, which is enough for
the latency gate of stage 6 (the latency does not depend on the weights). The thresholds of a fresh export
are placeholders (`calibration.strategy: none`) until `calibrate` sets them from a `score` run.

`score` scores the evaluation set of a protocol (and, in the test protocol, the fold-0 normals that fix
the hold-out thresholds) with onnxruntime on the CPU, optionally under one synthetic condition of stage 3
applied to the evaluation images. It writes `<category>.npz` in the common format of stage 2 (no maps)
and `run.json`. Only `export` needs torch.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import paths
from .cache import ImageCache
from .calibrate import conformal_threshold
from .conditions import CLEAN, apply_condition, condition_names
from .download import sha256_file
from .inspector import (
    ARTIFACT_VERSION,
    RECONSTRUCTION,
    RECONSTRUCTION_PRECISIONS,
    ReconstructionInspector,
    check_reconstruction_meta,
    preprocess,
)
from .ledger import git_commit, record_test_access
from .splits import ManifestRow, read_manifest, select
from .visa import CATEGORIES

MAP_SIZE = 256
PARITY_TOL = 1e-3  # largest torch-onnxruntime difference accepted at export (score relative, map absolute)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _read_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as err:
        raise ValueError(f"{path} is missing or not readable JSON ({err})") from err


# ---------------------------------------------------------------- export (torch)


def export_onnx(model, img_size: int, path: Path, *, opset: int = 18, probes: int = 2) -> dict:
    """Export `model` (a DinomalyModel on the CPU, fp32) as a `ServingGraph` for one image of `img_size`.

    Checks onnxruntime against torch on `probes` random images (seed 0) and raises when the score
    differs by more than PARITY_TOL (relative) or the map by more than PARITY_TOL (absolute). Returns
    the largest differences.
    """
    import onnx
    import torch

    from .dinomaly_model import ServingGraph

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = ServingGraph(model.to("cpu").float().eval()).eval()
    example = torch.zeros(1, 3, img_size, img_size, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            graph,
            (example,),
            str(path),
            input_names=["image"],
            output_names=["score", "map"],
            opset_version=opset,
            dynamo=False,
        )
    onnx.checker.check_model(str(path))
    session = _session(path)
    rng = np.random.default_rng(0)
    score_rel = map_abs = 0.0
    for _ in range(probes):
        image = rng.integers(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        x = preprocess(image, img_size)
        with torch.no_grad():
            ref_score, ref_map = (t.numpy() for t in graph(torch.from_numpy(x)))
        score, amap = session.run(["score", "map"], {"image": x})
        score_rel = max(
            score_rel, float(np.max(np.abs(score - ref_score) / np.maximum(np.abs(ref_score), 1e-12)))
        )
        map_abs = max(map_abs, float(np.max(np.abs(amap - ref_map))))
    parity = {"score_rel_diff": score_rel, "map_abs_diff": map_abs, "probes": probes, "tolerance": PARITY_TOL}
    if score_rel > PARITY_TOL or map_abs > PARITY_TOL:
        raise RuntimeError(f"ONNX export differs from torch: {parity}")
    return parity


def _session(path: Path, threads: int | None = None):
    import onnxruntime as ort

    options = ort.SessionOptions()
    if threads is not None:
        options.intra_op_num_threads = int(threads)
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def quantize_dynamic_matmul(fp32_path: Path, out_path: Path) -> Path:
    """Dynamic INT8 quantization of the MatMul nodes (weights int8, activations quantized per call)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(fp32_path), str(out_path), op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8
    )
    return Path(out_path)


def build_artifacts(
    model,
    *,
    name: str,
    img_size: int,
    out_dir: Path,
    source: dict,
    categories: list[str],
    int8_dynamic: bool = False,
) -> dict:
    """Export `model` and write the meta of every category with a placeholder threshold."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    fp32 = out_dir / RECONSTRUCTION_PRECISIONS["fp32"]
    parity = export_onnx(model, img_size, fp32)
    info: dict = {
        "export_s": round(time.perf_counter() - t0, 1),
        "parity": parity,
        "int8_dynamic": {"available": False, "reason": "not requested"},
    }
    int8_path = out_dir / RECONSTRUCTION_PRECISIONS["int8-dynamic"]
    int8_path.unlink(missing_ok=True)  # an older quantized model does not belong to the new fp32 model
    if int8_dynamic:
        quantize_dynamic_matmul(fp32, int8_path)
        info["int8_dynamic"] = {"available": True, "op_types": ["MatMul"], "weight_type": "QInt8"}
    commit = git_commit()
    model_sha256 = sha256_file(fp32)
    for category in categories:
        meta = {
            "version": ARTIFACT_VERSION,
            "kind": RECONSTRUCTION,
            "name": name,
            "category": category,
            "img_size": int(img_size),
            "map_size": MAP_SIZE,
            "threshold": 0.0,
            "calibration": {"strategy": "none"},
            "commit": commit,
            "source": {**source, "onnx_sha256": model_sha256},
        }
        check_reconstruction_meta(meta)
        _write_json(out_dir / category / "meta.json", meta)
    _write_json(
        out_dir / "artifacts.json",
        {"name": name, "img_size": int(img_size), "commit": commit, "source": source, **info},
    )
    return info


def _export_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    import torch

    from . import run_dinomaly

    cfg = run_dinomaly.CONFIGS[args.config]
    if args.untrained == (args.model is not None):
        parser.error("pass either --model (a trained model) or --untrained")
    out_dir = args.out or paths.ROOT / "artifacts" / (cfg.name + ("-untrained" if args.untrained else ""))
    if args.untrained:
        torch.manual_seed(args.seed)
        model = run_dinomaly.build_model(cfg.encoder, **cfg.build_options()).eval()
        source = {"config": cfg.name, "untrained": True, "seed": args.seed, "options": cfg.build_options()}
    else:
        if not args.model.exists():
            parser.error(f"{args.model} does not exist")
        model, saved = run_dinomaly.load_model(args.model)
        if saved.get("config", run_dinomaly.NAME) != cfg.name:
            parser.error(f"{args.model} holds a {saved.get('config', run_dinomaly.NAME)!r} model")
        source = {
            "config": cfg.name,
            "untrained": False,
            "model": str(args.model),
            "model_sha256": sha256_file(args.model),
            "train_steps": saved["steps"],
            "train_amp": saved["amp"],
            "options": saved.get("options", {}),
        }
    info = build_artifacts(
        model,
        name=cfg.name,
        img_size=cfg.img_size,
        out_dir=out_dir,
        source=source,
        categories=args.categories,
        int8_dynamic=args.int8_dynamic,
    )
    print(json.dumps({"out": str(out_dir), **info}, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- scoring (onnxruntime only)


def _rows(
    manifest: list[ManifestRow], protocol: str, category: str, allow_test: bool
) -> tuple[list[ManifestRow], list[ManifestRow], list[ManifestRow]]:
    """(evaluation normals, evaluation defects, calibration normals) of a category."""
    eval_normal = select(
        manifest, protocol=protocol, part="eval_normal", category=category, allow_test=allow_test
    )
    eval_defect = select(
        manifest, protocol=protocol, part="eval_defect", category=category, allow_test=allow_test
    )
    # Hold-out thresholds come from the fold-0 normals, which no model was trained on. In the dev
    # protocol those are the evaluation normals themselves.
    cal = (
        select(manifest, protocol="dev", part="eval_normal", category=category) if protocol == "test" else []
    )
    return eval_normal, eval_defect, cal


def score_category(
    inspector: ReconstructionInspector,
    protocol: str,
    category: str,
    manifest: list[ManifestRow],
    cache,
    allow_test: bool,
    out_dir: Path,
    *,
    condition: str = CLEAN,
) -> dict:
    """Score one category with the CPU pipeline and write `<category>.npz` (common format, no maps)."""
    eval_normal, eval_defect, cal_rows = _rows(manifest, protocol, category, allow_test)
    rows = eval_normal + eval_defect
    labels = np.array([0] * len(eval_normal) + [1] * len(eval_defect), dtype=np.int8)
    t0 = time.perf_counter()
    cal_score = inspector.scores(cache.images(cal_rows)) if cal_rows else np.empty(0, dtype=np.float32)
    t1 = time.perf_counter()
    images = apply_condition(cache.images(rows), condition) if rows else np.empty((0, 1, 1, 3), np.uint8)
    eval_score = inspector.scores(images)
    t2 = time.perf_counter()
    np.savez(
        out_dir / f"{category}.npz",
        eval_images=np.array([r.image for r in rows]),
        eval_labels=labels,
        eval_defect_types=np.array([r.defect_types for r in rows]),
        eval_score=eval_score,
        cal_score=cal_score,
        cal_images=np.array([r.image for r in cal_rows], dtype=np.str_),
    )
    return {
        "category": category,
        "eval_normal": len(eval_normal),
        "eval_defect": len(eval_defect),
        "cal": len(cal_rows),
        "threshold_in_meta": inspector.threshold,
        "timing": {"cal_score_s": round(t1 - t0, 2), "eval_score_s": round(t2 - t1, 2)},
    }


def _score_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.protocol == "test" and (not args.allow_test or not args.stage):
        parser.error("--protocol test needs --allow-test and --stage")
    if args.condition not in condition_names():
        parser.error(f"unknown condition {args.condition!r}; one of {condition_names()}")
    art = args.artifacts
    try:
        info = _read_json(art / "artifacts.json")
    except ValueError as err:
        parser.error(str(err))
    suffix = "" if args.condition == CLEAN else f"-{args.condition}"
    out_dir = args.out or paths.OUTPUTS / f"{info['name']}-onnx-{args.precision}-{args.protocol}{suffix}"
    if (out_dir / "run.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")
    model_path = art / RECONSTRUCTION_PRECISIONS[args.precision]
    if not model_path.exists():
        parser.error(f"{model_path} does not exist")
    manifest = read_manifest(paths.VISA_MANIFEST)
    known = {r.category for r in manifest}
    bad = [c for c in args.categories if c not in known or not (art / c / "meta.json").exists()]
    if bad or not args.categories or len(set(args.categories)) != len(args.categories):
        parser.error(f"categories {bad or args.categories} are unknown, repeated or not in {art}")
    # Every inspector is built before an image is read: a broken artifact stops the run unrecorded.
    try:
        inspectors = {
            c: ReconstructionInspector.load(art / c, precision=args.precision, threads=args.threads)
            for c in args.categories
        }
    except ValueError as err:
        parser.error(str(err))
    sizes = {i.size for i in inspectors.values()}
    if len(sizes) != 1:
        parser.error(f"the categories of {art} have different input sizes {sorted(sizes)}")
    commit = git_commit()
    cache = ImageCache(paths.CACHE, sizes.pop())
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in [out_dir / "run.json", *out_dir.glob("*.npz")]:
        path.unlink(missing_ok=True)  # an earlier run's files must not mix into this one
    if args.protocol == "test":
        record_test_access(
            paths.TEST_LEDGER,
            stage=args.stage,
            config=f"{info['name']}-onnx-{args.precision}" + suffix,
            note=args.note,
            commit=commit,
        )
    summaries = []
    started = time.perf_counter()
    for category in args.categories:
        summary = score_category(
            inspectors[category],
            args.protocol,
            category,
            manifest,
            cache,
            args.allow_test,
            out_dir,
            condition=args.condition,
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    run = {
        "method": "dinomaly-onnx",
        "protocol": args.protocol,
        "commit": commit,
        "config": {
            "name": info["name"],
            "img_size": info["img_size"],
            "artifacts": str(art),
            "precision": args.precision,
            "onnx_sha256": sha256_file(model_path),
            "source": info.get("source"),
            "threads": args.threads,
            "condition": args.condition,
            "calibration": "fold-0 normals (hold-out)" if args.protocol == "test" else "none",
        },
        "categories": summaries,
        "total_s": round(time.perf_counter() - started, 1),
    }
    _write_json(out_dir / "run.json", run)


# ---------------------------------------------------------------- thresholds for the service


def _calibrate_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    art = args.artifacts
    try:
        run = _read_json(args.scores / "run.json")
    except ValueError as err:
        parser.error(str(err))
    if run.get("protocol") != "test" or run.get("config", {}).get("condition") != CLEAN:
        parser.error("thresholds come from a clean test-protocol score run (its fold-0 normal scores)")
    fp32 = art / RECONSTRUCTION_PRECISIONS["fp32"]
    if run["config"].get("precision") != "fp32" or run["config"].get("onnx_sha256") != sha256_file(fp32):
        parser.error(f"{args.scores} was not scored with the fp32 model of {art}")
    written = {}
    for summary in run["categories"]:
        category = summary["category"]
        with np.load(args.scores / f"{category}.npz") as z:
            cal = z["cal_score"]
        thr = conformal_threshold(cal, args.alpha)
        meta = _read_json(art / category / "meta.json")
        meta.update(
            threshold=float(thr.value),
            alpha=float(args.alpha),
            calibration={"strategy": "holdout", "n": int(thr.n), "guaranteed": bool(thr.guaranteed)},
        )
        meta["source"] = {**meta.get("source", {}), "scores": str(args.scores)}
        check_reconstruction_meta(meta)
        _write_json(art / category / "meta.json", meta)
        written[category] = meta["threshold"]
    print(json.dumps(written, ensure_ascii=False), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    exp = commands.add_parser("export", help="export a Dinomaly model as an artifact set")
    exp.add_argument("--config", required=True, help="a run_dinomaly config name, e.g. dms-280")
    exp.add_argument("--model", type=Path, default=None, help="model.pt of run_dinomaly train")
    exp.add_argument("--untrained", action="store_true", help="a freshly initialised model (latency only)")
    exp.add_argument("--seed", type=int, default=0, help="initial weights of --untrained")
    exp.add_argument("--int8-dynamic", action="store_true", help="also write the dynamic INT8 model")
    exp.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    exp.add_argument("--out", type=Path, default=None, help="default: artifacts/<config>[-untrained]")

    score = commands.add_parser("score", help="score a protocol with onnxruntime on the CPU")
    score.add_argument("--artifacts", type=Path, required=True)
    score.add_argument("--protocol", choices=["dev", "test"], required=True)
    score.add_argument("--precision", choices=sorted(RECONSTRUCTION_PRECISIONS), default="fp32")
    score.add_argument(
        "--condition", default=CLEAN, help="a stage 3 condition, applied to the evaluation images"
    )
    score.add_argument(
        "--threads", type=int, default=None, help="onnxruntime intra-op threads (default: its own)"
    )
    score.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    score.add_argument("--out", type=Path, default=None)
    score.add_argument("--allow-test", action="store_true", help="read the sealed test set (logged)")
    score.add_argument(
        "--stage", default="", help="stage label for the test ledger (required with --allow-test)"
    )
    score.add_argument("--note", default="")
    score.add_argument("--overwrite", action="store_true")

    cal = commands.add_parser("calibrate", help="hold-out thresholds from a clean test-protocol score run")
    cal.add_argument("--artifacts", type=Path, required=True)
    cal.add_argument("--scores", type=Path, required=True)
    cal.add_argument("--alpha", type=float, default=0.05)

    args = parser.parse_args(argv)
    {"export": _export_command, "score": _score_command, "calibrate": _calibrate_command}[args.command](
        args, parser
    )


if __name__ == "__main__":
    main()
