"""Stage 6: the CPU latency gate, the pick of the DMS model on dev, and H16 / H17 on the sealed test.

Rules as registered in docs/experiments.md (stage 6):

- gate: an input size stays a training candidate when its untrained model's CPU latency (resize +
  inference, onnxruntime FP32) is at most 200 ms. Trained sizes: those of 252/280/308 that pass; when
  308 fails and 294 passes, 294 takes its place. The CAR model goes with 280, or with 252 when 280 fails.
- pick: among the trained models (all within the budget), the highest dev mean image AUROC (torch,
  fp32 scoring); ties go to the smaller input.
- H16 (as H15): (a) CPU latency of the trained pick <= 200 ms, (b) pooled false alarm rate of the ONNX
  FP32 pipeline at its own hold-out thresholds (alpha 5%) <= 6%, (c) its mean image AUROC at most 2.0
  points below DM's (stage 2, 96.79). All three: supported.
- H17: ONNX FP32 AUROC of the pick minus that of the serving PatchCore (stage 4, ONNX FP32), paired
  stratified bootstrap: supported when the interval excludes 0 from above and the difference is at least
  1.0 point; rejected when the interval lies below 0; otherwise undecided.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import compare, paths
from .calibrate import conformal_threshold

CPU_BUDGET_MS = 200.0
GATE_SIZES = (252, 280, 294, 308)
H16_FPR_MAX = 0.06
H16_AUROC_MARGIN = 0.02
H17_MIN_EFFECT = 0.01
ALPHA = 0.05


def gate_latency(
    latency: dict, name: str, suffix: str = "-untrained", precision: str = "fp32"
) -> float | None:
    """Registered CPU latency (resize + inference) of entry `<name><suffix>`, or None when not measured."""
    entry = latency.get(name + suffix, {}).get(f"cpu_{precision}")
    return None if entry is None else float(entry["total_with_resize_ms"])


def gate(latency: dict, budget_ms: float = CPU_BUDGET_MS) -> dict:
    """Which DMS models to train, from the latency of the untrained models (`dms-<size>-untrained`)."""
    ms = {size: gate_latency(latency, f"dms-{size}") for size in GATE_SIZES}
    missing = [size for size, value in ms.items() if value is None]
    if missing:
        raise ValueError(f"the latency file has no untrained entry for sizes {missing}")
    passed = [size for size in GATE_SIZES if ms[size] <= budget_ms]
    train = [size for size in (252, 280, 308) if size in passed]
    if 308 not in passed and 294 in passed:
        train.append(294)
    car = 280 if 280 in passed else (252 if 252 in passed else None)
    models = [f"dms-{size}" for size in sorted(train)]
    if car is not None:
        models.append(f"dms-{car}-car")
    return {
        "budget_ms": budget_ms,
        "cpu_ms": {f"dms-{size}": ms[size] for size in GATE_SIZES},
        "passed": [f"dms-{size}" for size in passed],
        "train": models,
        "blocked": not models,
    }


def _size(name: str) -> int:
    return int(name.split("-")[1])


def choose(rows: list[dict], budget_ms: float = CPU_BUDGET_MS) -> dict | None:
    """The pick among rows with `name`, `cpu_ms` and `macro_image_auroc` (None: nothing within budget)."""
    within = [r for r in rows if r.get("cpu_ms") is not None and r["cpu_ms"] <= budget_ms]
    if not within:
        return None
    best = min(within, key=lambda r: (-r["macro_image_auroc"], _size(r["name"])))
    return {
        "name": best["name"],
        "macro_image_auroc": best["macro_image_auroc"],
        "cpu_ms": best["cpu_ms"],
        "candidates": [r["name"] for r in within],
    }


def load_patchcore_cpu(run_dir: Path, name: str = "serving-patchcore") -> compare.MethodRun:
    """The ONNX FP32 scores of the stage 4 serving PatchCore (`run_export_eval`, one precision folder)."""
    run = compare.MethodRun(name, "crossfit")
    for path in sorted(Path(run_dir).glob("*.npz")):
        with np.load(path) as z:
            run.cats[path.stem] = {
                "eval_images": z["eval_images"],
                "eval_labels": z["eval_labels"],
                "eval_score": z["eval_score_full"],
                "cal_score": z["pool_score_oof"],
            }
    if not run.cats:
        raise ValueError(f"{run_dir} holds no category scores")
    return run


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dev_report(run_dirs: list[Path], latency: dict) -> dict:
    """Dev table of the trained candidates, the pick, and the CAR pair difference (no verdict)."""
    runs = {}
    for run_dir in run_dirs:
        name = _read_json(Path(run_dir) / "run.json")["config"]["name"]
        runs[name] = compare.load_common(Path(run_dir), name)
    names = list(runs)
    compare.check_same_images(list(runs.values()))
    boot = compare.Bootstrap(runs[names[0]])
    rows = []
    for name, run in runs.items():
        summary = compare.summarize(run, boot)
        rows.append(
            {
                "name": name,
                "macro_image_auroc": summary["macro_image_auroc"],
                "macro_image_auroc_ci": summary["macro_image_auroc_ci"],
                "macro_aupro": summary["macro_aupro"],
                "pooled_fpr_at_tpr95": summary["pooled_fpr_at_tpr95"],
                "cpu_ms": gate_latency(latency, name),
                "per_category_auroc": {c: v["image_auroc"] for c, v in summary["per_category"].items()},
            }
        )
    report = {"n_boot": boot.n_boot, "rows": rows, "pick": choose(rows), "gate": gate(latency)}
    for name in names:
        if name.endswith("-car") and name[: -len("-car")] in runs:
            base = name[: -len("-car")]
            point = compare.macro_auroc(runs[name]) - compare.macro_auroc(runs[base])
            samples = boot.macro_auroc(runs[name]) - boot.macro_auroc(runs[base])
            report["car_effect"] = {"pair": [name, base], **compare.difference(point, samples)}
    return report


def _pooled_rates(run: compare.MethodRun, thresholds: dict[str, float]) -> dict:
    fp = tp = n_neg = n_pos = 0
    for c, cat in run.cats.items():
        labels = cat["eval_labels"]
        scores = cat["eval_score"]
        fp += int((scores[labels == 0] > thresholds[c]).sum())
        tp += int((scores[labels == 1] > thresholds[c]).sum())
        n_neg += int((labels == 0).sum())
        n_pos += int((labels == 1).sum())
    return {"fpr": fp / n_neg, "tpr": tp / n_pos, "fp": fp, "n_neg": n_neg, "tp": tp, "n_pos": n_pos}


def h16(onnx_summary: dict, cpu_ms: float | None, dm_auroc: float) -> dict:
    """The registered three conditions of H16 on point estimates."""
    fpr = onnx_summary["fixed_threshold"]["fpr"]
    auroc = onnx_summary["macro_image_auroc"]
    conditions = {
        "a_cpu_ms": {"value": cpu_ms, "met": cpu_ms is not None and cpu_ms <= CPU_BUDGET_MS},
        "b_fpr": {"value": fpr, "met": fpr <= H16_FPR_MAX},
        "c_auroc_vs_dm": {"value": auroc - dm_auroc, "met": auroc >= dm_auroc - H16_AUROC_MARGIN},
    }
    failed = [k for k, v in conditions.items() if not v["met"]]
    return {"conditions": conditions, "failed": failed, "verdict": "지지" if not failed else "기각"}


def h17(point: float, samples: np.ndarray) -> dict:
    lo, hi = compare._ci(samples)
    if lo > 0 and point >= H17_MIN_EFFECT:
        verdict = "지지"
    elif hi < 0:
        verdict = "기각"
    else:
        verdict = "판정 불가"
    return {"diff": float(point), "ci": [lo, hi], "verdict": verdict}


def sealed_report(
    *,
    torch_dir: Path,
    onnx_dir: Path,
    patchcore_dir: Path,
    cpu_ms: float | None,
    dm_auroc: float,
) -> dict:
    """H16 and H17 for the pick, with the numbers registered to go next to them."""
    name = _read_json(Path(onnx_dir) / "run.json")["config"]["name"]
    onnx = compare.load_common(Path(onnx_dir), f"{name}-onnx-fp32")
    torch_run = compare.load_common(Path(torch_dir), f"{name}-torch")
    patchcore = load_patchcore_cpu(Path(patchcore_dir))
    compare.check_same_images([onnx, torch_run, patchcore])
    boot = compare.Bootstrap(onnx)
    onnx_summary = {
        "macro_image_auroc": compare.macro_auroc(onnx),
        "macro_image_auroc_ci": compare._ci(boot.macro_auroc(onnx)),
        "fixed_threshold": boot.pooled_rates(onnx, ALPHA),
    }
    torch_summary = compare.summarize(torch_run, boot)
    point = compare.macro_auroc(onnx) - compare.macro_auroc(patchcore)
    samples = boot.macro_auroc(onnx) - boot.macro_auroc(patchcore)
    torch_thresholds = {
        c: conformal_threshold(cat["cal_score"], ALPHA).value for c, cat in torch_run.cats.items()
    }
    rel = np.concatenate(
        [
            np.abs(onnx.cats[c]["eval_score"] - torch_run.cats[c]["eval_score"])
            / np.maximum(np.abs(torch_run.cats[c]["eval_score"]), 1e-12)
            for c in onnx.cats
        ]
    )
    return {
        "name": name,
        "alpha": ALPHA,
        "n_boot": boot.n_boot,
        "onnx_fp32": onnx_summary,
        "torch": {
            k: torch_summary[k]
            for k in (
                "macro_image_auroc",
                "macro_image_auroc_ci",
                "macro_aupro",
                "macro_aupro_ci",
                "fixed_threshold",
            )
        },
        "torch_per_category": torch_summary["per_category"],
        "pooled_fpr_at_tpr95_torch": torch_summary["pooled_fpr_at_tpr95"],
        "onnx_at_torch_thresholds": _pooled_rates(onnx, torch_thresholds),
        "score_rel_diff_onnx_vs_torch": {"median": float(np.median(rel)), "max": float(np.max(rel))},
        "serving_patchcore_auroc": compare.macro_auroc(patchcore),
        "dm_auroc": dm_auroc,
        "cpu_ms": cpu_ms,
        "H16": h16(onnx_summary, cpu_ms, dm_auroc),
        "H17": h17(point, samples),
    }


def supplement_report(clean_dir: Path, condition_dirs: list[Path], int8_dir: Path | None = None) -> dict:
    """Dev, no verdict: conditions at thresholds from the clean fold-0 normal scores, and dynamic INT8.

    In the dev protocol the fold-0 normals are the evaluation normals, so the clean row's false alarm
    rate is about 5% by construction; the rows under a condition show how many of the same normals cross
    those thresholds once the images change.
    """
    clean = compare.load_common(Path(clean_dir), "clean")
    thresholds = {}
    for c, cat in clean.cats.items():
        thresholds[c] = conformal_threshold(cat["eval_score"][cat["eval_labels"] == 0], ALPHA).value
    rows = [
        {
            "condition": "clean",
            "macro_image_auroc": compare.macro_auroc(clean),
            **_pooled_rates(clean, thresholds),
        }
    ]
    for run_dir in condition_dirs:
        condition = _read_json(Path(run_dir) / "run.json")["config"]["condition"]
        run = compare.load_common(Path(run_dir), condition)
        compare.check_same_images([clean, run])
        rows.append(
            {
                "condition": condition,
                "macro_image_auroc": compare.macro_auroc(run),
                **_pooled_rates(run, thresholds),
            }
        )
    report: dict = {"thresholds": "clean fold-0 normal scores (in-sample in the dev protocol)", "rows": rows}
    if int8_dir is not None:
        int8 = compare.load_common(Path(int8_dir), "int8-dynamic")
        compare.check_same_images([clean, int8])
        boot = compare.Bootstrap(clean)
        point = compare.macro_auroc(int8) - compare.macro_auroc(clean)
        samples = boot.macro_auroc(int8) - boot.macro_auroc(clean)
        report["int8_dynamic"] = {
            "macro_image_auroc": compare.macro_auroc(int8),
            "fp32_macro_image_auroc": compare.macro_auroc(clean),
            **compare.difference(point, samples),
        }
    return report


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    default_latency = paths.REPORTS / "stage6" / "latency.json"

    g = commands.add_parser("gate", help="training candidates from the untrained latency")
    g.add_argument("--latency", type=Path, default=default_latency)
    g.add_argument("--out", type=Path, default=paths.REPORTS / "stage6" / "gate.json")

    d = commands.add_parser("dev", help="dev table, pick and CAR difference")
    d.add_argument(
        "--runs", type=Path, nargs="+", required=True, help="run_dinomaly eval --protocol dev runs"
    )
    d.add_argument("--latency", type=Path, default=default_latency)
    d.add_argument("--out", type=Path, default=paths.REPORTS / "stage6" / "dev.json")

    t = commands.add_parser("test", help="H16 and H17 of the pick")
    t.add_argument("--torch", type=Path, required=True, help="run_dinomaly eval --protocol test run")
    t.add_argument("--onnx", type=Path, required=True, help="dinomaly_serving score --protocol test run")
    t.add_argument(
        "--patchcore",
        type=Path,
        default=paths.OUTPUTS / "export-serving-wrn50-256-r0.01-test" / "fp32",
        help="stage 4 CPU FP32 scores of the serving PatchCore",
    )
    t.add_argument("--latency", type=Path, default=default_latency)
    t.add_argument("--latency-key", required=True, help="entry of the trained pick, e.g. dms-280")
    t.add_argument("--stage2", type=Path, default=paths.REPORTS / "stage2" / "compare-test.json")
    t.add_argument("--out", type=Path, default=paths.REPORTS / "stage6" / "test.json")

    s = commands.add_parser("supplement", help="dev conditions and dynamic INT8 of the pick (no verdict)")
    s.add_argument("--clean", type=Path, required=True)
    s.add_argument("--conditions", type=Path, nargs="*", default=[])
    s.add_argument("--int8", type=Path, default=None)
    s.add_argument("--out", type=Path, default=paths.REPORTS / "stage6" / "dev-supplement.json")

    args = parser.parse_args(argv)
    if args.command == "gate":
        report = gate(_read_json(args.latency))
    elif args.command == "dev":
        report = dev_report(args.runs, _read_json(args.latency))
    elif args.command == "test":
        stage2 = _read_json(args.stage2)
        dm_auroc = float(stage2["unsupervised"][stage2["reference_unsupervised"]]["macro_image_auroc"])
        latency = _read_json(args.latency)
        report = sealed_report(
            torch_dir=args.torch,
            onnx_dir=args.onnx,
            patchcore_dir=args.patchcore,
            cpu_ms=gate_latency(latency, args.latency_key, suffix=""),
            dm_auroc=dm_auroc,
        )
    else:
        report = supplement_report(args.clean, args.conditions, args.int8)
    _write(args.out, report)
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
