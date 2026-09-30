"""How much the exported CPU pipeline drifts from the torch pipeline, and whether INT8 needs new thresholds.

Pipelines: `torch` (GPU fp16, where banks and thresholds come from), `fp32` and `int8` (onnxruntime on the
CPU). For each: accuracy, and the pooled false alarm and detection rates under thresholds calibrated
with a named pipeline's cross-fitted scores.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import compare, paths
from .calibrate import conformal_threshold

ALPHA = 0.05
MIN_EFFECT_FPR = 0.02


def load(out_dir: Path) -> dict:
    with open(out_dir / "run.json", encoding="utf-8") as f:
        meta = json.load(f)
    pipelines: dict[str, dict[str, dict[str, np.ndarray]]] = {"torch": {}}
    for precision in meta["precisions"]:
        pipelines[precision] = {}
        for category in meta["categories"]:
            with np.load(out_dir / precision / f"{category}.npz") as z:
                arrays = {k: z[k] for k in z.files}
            pipelines[precision][category] = {
                "eval_images": arrays["eval_images"],
                "eval_labels": arrays["eval_labels"],
                "eval_score": arrays["eval_score_full"],
                "cal_score": arrays["pool_score_oof"],
            }
            pipelines["torch"][category] = {
                "eval_images": arrays["eval_images"],
                "eval_labels": arrays["eval_labels"],
                "eval_score": arrays["torch_eval_score"],
                "cal_score": arrays["torch_pool_score_oof"],
            }
    return {"meta": meta, "pipelines": pipelines}


def _rates(
    boot: compare.Bootstrap, scores: dict[str, dict], thresholds_from: dict[str, dict], alpha: float
) -> dict[str, np.ndarray | float]:
    """Pooled FPR/TPR of `scores` under thresholds calibrated on `thresholds_from`, with bootstrap samples."""
    fp = tp = n_neg = n_pos = 0
    fp_boot = np.zeros(boot.n_boot)
    tp_boot = np.zeros(boot.n_boot)
    for i, c in enumerate(boot.names):
        thr = conformal_threshold(thresholds_from[c]["cal_score"], alpha).value
        labels = scores[c]["eval_labels"]
        neg, pos = scores[c]["eval_score"][labels == 0], scores[c]["eval_score"][labels == 1]
        f_neg, f_pos = neg > thr, pos > thr
        neg_idx, pos_idx = boot.indices(i)
        fp += int(f_neg.sum())
        tp += int(f_pos.sum())
        n_neg += len(neg)
        n_pos += len(pos)
        fp_boot += f_neg[neg_idx].sum(axis=1)
        tp_boot += f_pos[pos_idx].sum(axis=1)
    return {"fpr": fp / n_neg, "tpr": tp / n_pos, "fpr_boot": fp_boot / n_neg, "tpr_boot": tp_boot / n_pos}


def _cell(r: dict) -> dict:
    return {
        "fpr": float(r["fpr"]),
        "fpr_ci": compare._ci(r["fpr_boot"]),
        "tpr": float(r["tpr"]),
        "tpr_ci": compare._ci(r["tpr_boot"]),
    }


def build_report(run: dict, alpha: float = ALPHA) -> dict:
    pipelines = run["pipelines"]
    runs = {name: compare.MethodRun(name, "crossfit", cats) for name, cats in pipelines.items()}
    compare.check_same_images(list(runs.values()))
    boot = compare.Bootstrap(runs["torch"])
    base_auroc = compare.macro_auroc(runs["torch"])
    base_auroc_boot = boot.macro_auroc(runs["torch"])

    rows = []
    for name, cats in pipelines.items():
        auroc_boot = boot.macro_auroc(runs[name])
        own = _rates(boot, cats, cats, alpha)
        row = {
            "pipeline": name,
            "macro_image_auroc": compare.macro_auroc(runs[name]),
            "macro_image_auroc_ci": compare._ci(auroc_boot),
            "own_thresholds": _cell(own),
        }
        if name != "torch":
            row["vs_torch_auroc"] = compare.difference(
                row["macro_image_auroc"] - base_auroc, auroc_boot - base_auroc_boot
            )
            at_torch = _rates(boot, cats, pipelines["torch"], alpha)
            torch_own = _rates(boot, pipelines["torch"], pipelines["torch"], alpha)
            row["torch_thresholds"] = _cell(at_torch)
            row["d_fpr_vs_torch_at_torch_thresholds"] = compare.difference(
                at_torch["fpr"] - torch_own["fpr"],
                at_torch["fpr_boot"] - torch_own["fpr_boot"],
                MIN_EFFECT_FPR,
            )
            rel = [
                np.abs(cats[c]["eval_score"] - pipelines["torch"][c]["eval_score"])
                / np.maximum(np.abs(pipelines["torch"][c]["eval_score"]), 1e-12)
                for c in cats
            ]
            row["score_rel_diff_vs_torch"] = {
                "median": float(np.median(np.concatenate(rel))),
                "max": float(np.max(np.concatenate(rel))),
            }
        rows.append(row)

    report = {
        "alpha": alpha,
        "n_boot": boot.n_boot,
        "rows": rows,
        **{k: run["meta"][k] for k in ("artifacts", "ratio", "protocol")},
    }
    if "fp32" in pipelines and "int8" in pipelines:
        report["int8_at_fp32_thresholds"] = int8_shift(boot, pipelines, alpha)
    return report


def int8_shift(boot: compare.Bootstrap, pipelines: dict, alpha: float = ALPHA) -> dict:
    """INT8 scores under thresholds calibrated with the fp32 CPU pipeline, against fp32 under the same ones.

    Registered claim: the pooled false alarm rate moves by more than 2 points. Supported when the paired
    interval excludes 0 and the change is at least 2 points; rejected when the whole interval lies inside
    (-2, +2) points; undecided otherwise.
    """
    fp32 = _rates(boot, pipelines["fp32"], pipelines["fp32"], alpha)
    int8 = _rates(boot, pipelines["int8"], pipelines["fp32"], alpha)
    recal = _rates(boot, pipelines["int8"], pipelines["int8"], alpha)
    diff = int8["fpr"] - fp32["fpr"]
    lo, hi = compare._ci(int8["fpr_boot"] - fp32["fpr_boot"])
    if (lo > 0 or hi < 0) and abs(diff) >= MIN_EFFECT_FPR:
        verdict = "지지"
    elif -MIN_EFFECT_FPR < lo and hi < MIN_EFFECT_FPR:
        verdict = "기각"
    else:
        verdict = "판정 불가"
    return {
        "fp32": _cell(fp32),
        "int8_at_fp32_thresholds": _cell(int8),
        "int8_recalibrated": _cell(recal),
        "d_fpr": float(diff),
        "d_fpr_ci": [lo, hi],
        "d_tpr": float(int8["tpr"] - fp32["tpr"]),
        "d_tpr_ci": compare._ci(int8["tpr_boot"] - fp32["tpr_boot"]),
        "verdict": verdict,
    }


def _pct(x: float) -> str:
    return f"{x * 100:.2f}"


def _span(ci: list[float]) -> str:
    return f"[{_pct(ci[0])}, {_pct(ci[1])}]"


def format_report(report: dict) -> str:
    lines = [
        f"{report['artifacts']} ratio {report['ratio']:g} protocol {report['protocol']}",
        "| pipeline | image AUROC | FPR, own thresholds | detection | FPR, torch thresholds "
        "| score drift (median / max) |",
        "|---|---|---|---|---|---|",
    ]
    for r in report["rows"]:
        own = r["own_thresholds"]
        cells = [
            r["pipeline"],
            f"{_pct(r['macro_image_auroc'])} {_span(r['macro_image_auroc_ci'])}",
            f"{_pct(own['fpr'])} {_span(own['fpr_ci'])}",
            f"{_pct(own['tpr'])} {_span(own['tpr_ci'])}",
        ]
        if "torch_thresholds" in r:
            drift = r["score_rel_diff_vs_torch"]
            cells += [
                f"{_pct(r['torch_thresholds']['fpr'])} {_span(r['torch_thresholds']['fpr_ci'])}",
                f"{drift['median'] * 100:.3f}% / {drift['max'] * 100:.2f}%",
            ]
        else:
            cells += ["-", "-"]
        lines.append("| " + " | ".join(cells) + " |")
    shift = report.get("int8_at_fp32_thresholds")
    if shift:
        lines += [
            "",
            f"INT8 under fp32 thresholds: FPR {_pct(shift['fp32']['fpr'])} -> "
            f"{_pct(shift['int8_at_fp32_thresholds']['fpr'])} "
            f"({shift['d_fpr'] * 100:+.2f}%p {_span(shift['d_fpr_ci'])}), "
            f"after recalibration {_pct(shift['int8_recalibrated']['fpr'])}; verdict: {shift['verdict']}",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="a run_export_eval output directory")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = build_report(load(args.run_dir))
    print(format_report(report))
    if not args.no_write:
        out_dir = paths.REPORTS / "stage4"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"export-{report['protocol']}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
