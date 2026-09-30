"""Tables for a PatchCore run: accuracy, threshold calibration strategies, finite-sample curve.

Reads the per-category files written by `run_patchcore` and writes a JSON report plus a per-image score
CSV, so that every number can be recomputed without the images.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from . import paths
from .calibrate import conformal_threshold, pooled_fpr_band, subsample_fpr_curve
from .metrics import ProHistograms, aupro_bootstrap, aupro_from_histograms, auroc, fpr_at_tpr
from .splits import holdout_fold
from .stats import (
    clopper_pearson,
    macro_auroc_bootstrap,
    percentile_ci,
    pooled_rate_bootstrap,
    stratified_indices,
)

N_BOOT = 2000
SEED = 0
ALPHAS = (0.05, 0.01)
STRATEGIES = ("resubstitution", "holdout", "crossfit")
# Reference simulation for the finite-sample curve: exchangeable scores with the same pool/test sizes.
REF_REALISATIONS = 100
REF_DRAWS = 300


def load_run(run_dir: Path) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    with open(run_dir / "run.json", encoding="utf-8") as f:
        meta = json.load(f)
    cats = {}
    for info in meta["categories"]:
        with np.load(run_dir / f"{info['category']}.npz") as z:
            cats[info["category"]] = {k: z[k] for k in z.files}
    return meta, cats


def _split(cat: dict[str, np.ndarray], key: str) -> tuple[np.ndarray, np.ndarray]:
    labels = cat["eval_labels"]
    return cat[key][labels == 0], cat[key][labels == 1]


def _ci(samples: np.ndarray) -> list[float]:
    lo, hi = percentile_ci(samples)
    return [float(lo), float(hi)]


def _histograms(cat: dict[str, np.ndarray]) -> ProHistograms:
    return ProHistograms(
        edges=cat["pro_edges"],
        normal=cat["pro_normal"],
        components=cat["pro_components"],
        component_image=cat["pro_component_image"],
    )


def accuracy(cats: dict[str, dict[str, np.ndarray]], score_key: str = "eval_score_full") -> dict:
    """Image AUROC, AUPRO, pixel AUROC per category and their macro averages; pooled FPR at 95% TPR."""
    names = list(cats)
    neg = [_split(cats[c], score_key)[0] for c in names]
    pos = [_split(cats[c], score_key)[1] for c in names]
    per_cat = {}
    for c, n, p in zip(names, neg, pos, strict=True):
        fpr, _, fp = fpr_at_tpr(n, p, 0.95)
        per_cat[c] = {
            "n_normal": len(n),
            "n_defect": len(p),
            "image_auroc": float(auroc(n, p)),
            "aupro": float(cats[c]["aupro"]),
            "pixel_auroc": float(cats[c]["pixel_auroc"]),
            "fpr_at_tpr95": float(fpr),
            "fp_at_tpr95": int(fp),
        }
    macro_auroc = float(np.mean([v["image_auroc"] for v in per_cat.values()]))
    macro_aupro = float(np.mean([v["aupro"] for v in per_cat.values()]))
    pooled_fp = sum(v["fp_at_tpr95"] for v in per_cat.values())
    pooled_n = sum(v["n_normal"] for v in per_cat.values())

    auroc_boot = macro_auroc_bootstrap(neg, pos, n_boot=N_BOOT, seed=SEED)

    # Same stratified draws (normals then defects of each category) for the other statistics.
    sizes = [s for n, p in zip(neg, pos, strict=True) for s in (len(n), len(p))]
    draws = stratified_indices(sizes, n_boot=N_BOOT, seed=SEED)
    aupro_boot = np.zeros(N_BOOT)
    aupro_binned = []
    fpr95_fp = np.zeros(N_BOOT)
    for i, c in enumerate(names):
        neg_idx, pos_idx = draws[2 * i], draws[2 * i + 1]
        n_neg, n_pos = len(neg[i]), len(pos[i])
        hist = _histograms(cats[c])
        aupro_binned.append(aupro_from_histograms(hist))
        # counts[b, j] = how often evaluation image j (normals first, then defects) is drawn in resample b
        counts = np.zeros((N_BOOT, n_neg + n_pos), dtype=np.int64)
        rows = np.arange(N_BOOT)[:, None]
        np.add.at(counts, (rows, neg_idx), 1)
        np.add.at(counts, (rows, n_neg + pos_idx), 1)
        aupro_boot += aupro_bootstrap(hist, counts) / len(names)
        for b in range(N_BOOT):
            fpr95_fp[b] += fpr_at_tpr(neg[i][neg_idx[b]], pos[i][pos_idx[b]], 0.95)[2]
    return {
        "per_category": per_cat,
        "macro_image_auroc": macro_auroc,
        "macro_image_auroc_ci": _ci(auroc_boot),
        "macro_aupro": macro_aupro,
        "macro_aupro_ci": _ci(aupro_boot),
        "macro_aupro_binned": float(np.mean(aupro_binned)),
        "macro_pixel_auroc": float(np.mean([v["pixel_auroc"] for v in per_cat.values()])),
        "pooled_fpr_at_tpr95": pooled_fp / pooled_n,
        "pooled_fpr_at_tpr95_ci": _ci(fpr95_fp / pooled_n),
    }


def _strategy_inputs(cat: dict[str, np.ndarray], strategy: str, protocol: str) -> tuple[np.ndarray, str]:
    """Calibration scores and the evaluation score column the strategy's deployed model produces."""
    if strategy == "resubstitution":
        return cat["pool_score_resub"], "eval_score_full"
    if strategy == "holdout":
        return cat["pool_score_oof"][cat["pool_folds"] == holdout_fold(protocol)], "eval_score_holdout"
    if strategy == "crossfit":
        return cat["pool_score_oof"], "eval_score_full"
    raise ValueError(strategy)


def calibration(cats: dict[str, dict[str, np.ndarray]], protocol: str, alpha: float) -> dict:
    """Actual pooled false positive and detection rates of thresholds fixed from normal scores only."""
    out: dict[str, dict] = {}
    flags: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for strategy in STRATEGIES:
        fp_flags, tp_flags, n_cal, per_cat = [], [], [], {}
        guaranteed = 0
        for c, cat in cats.items():
            cal, key = _strategy_inputs(cat, strategy, protocol)
            thr = conformal_threshold(cal, alpha)
            neg, pos = _split(cat, key)
            fp_flags.append(neg > thr.value)
            tp_flags.append(pos > thr.value)
            n_cal.append(len(cal))
            guaranteed += int(thr.guaranteed)
            per_cat[c] = {
                "threshold": float(thr.value),
                "n_cal": len(cal),
                "guaranteed": bool(thr.guaranteed),
                "fp": int(fp_flags[-1].sum()),
                "n_normal": len(neg),
                "tp": int(tp_flags[-1].sum()),
                "n_defect": len(pos),
            }
        fp, n_neg = sum(int(f.sum()) for f in fp_flags), sum(len(f) for f in fp_flags)
        tp, n_pos = sum(int(f.sum()) for f in tp_flags), sum(len(f) for f in tp_flags)
        band = pooled_fpr_band(n_cal, [len(f) for f in fp_flags], alpha)
        out[strategy] = {
            "fpr": fp / n_neg,
            "fp": fp,
            "n_normal": n_neg,
            "fpr_ci": _ci(pooled_rate_bootstrap(fp_flags, n_boot=N_BOOT, seed=SEED)),
            "fpr_clopper_pearson": [float(x) for x in clopper_pearson(fp, n_neg)],
            "theory_band": [float(band[0]), float(band[1])],
            "theory_mean": float(band[2]),
            "tpr": tp / n_pos,
            "tp": tp,
            "n_defect": n_pos,
            "tpr_ci": _ci(pooled_rate_bootstrap(tp_flags, n_boot=N_BOOT, seed=SEED)),
            "categories_guaranteed": guaranteed,
            "n_cal_total": int(sum(n_cal)),
            "per_category": per_cat,
        }
        flags[strategy] = (fp_flags, tp_flags)

    # Paired difference (same images, same draws): cross-fitting minus hold-out detection rate.
    diff = pooled_rate_bootstrap(flags["crossfit"][1], n_boot=N_BOOT, seed=SEED) - pooled_rate_bootstrap(
        flags["holdout"][1], n_boot=N_BOOT, seed=SEED
    )
    out["tpr_crossfit_minus_holdout"] = {
        "point": out["crossfit"]["tpr"] - out["holdout"]["tpr"],
        "ci": _ci(diff),
    }
    return out


def hypotheses(cal: dict, alpha: float = 0.05) -> dict[str, str]:
    """Verdicts of stage 1 H1-H4 exactly as registered in docs/experiments.md."""

    def three_way(support: bool, reject: bool) -> str:
        return "지지" if support else "기각" if reject else "판정 불가"

    resub, hold, cross = cal["resubstitution"], cal["holdout"], cal["crossfit"]
    lo, hi = hold["theory_band"]
    if lo <= hold["fpr"] <= hi:
        h2 = "지지"
    else:
        h2 = "기각 (이론 구간보다 " + ("높음" if hold["fpr"] > hi else "낮음") + ")"
    d_lo, d_hi = cal["tpr_crossfit_minus_holdout"]["ci"]
    return {
        "H1": three_way(resub["fpr_ci"][0] >= 2 * alpha, resub["fpr_ci"][1] < 2 * alpha),
        "H2": h2,
        "H3": three_way(cross["fpr_ci"][1] <= alpha, cross["fpr_ci"][0] > alpha),
        "H4": three_way(d_lo > -0.02, d_hi < -0.02),
    }


def _curve_means(curves: list[list[dict]]) -> dict[int, float]:
    """Mean over categories of each category's mean false positive rate, per calibration size n."""
    by_n: dict[int, list[float]] = {}
    for curve in curves:
        for r in curve:
            by_n.setdefault(int(r["n"]), []).append(float(r["mean"]))
    return {n: float(np.mean(v)) for n, v in by_n.items()}


def finite_sample(cats: dict[str, dict[str, np.ndarray]], alpha: float) -> list[dict]:
    """Test false positive rate against calibration size n, averaged over categories.

    Each category's curve subsamples its cross-fitted scores (a finite pool) and applies the thresholds
    to its evaluation normals (a finite set), so the Beta law of the theory does not describe its spread
    or its mean exactly. The comparable reference is a simulation of the same procedure on exchangeable
    (iid uniform) scores with the same pool and evaluation sizes: if the observed mean over categories
    falls outside the reference interval, the deviation is more than finite-sample wobble.
    """
    observed, sizes = [], []
    for cat in cats.values():
        neg, _ = _split(cat, "eval_score_full")
        observed.append(subsample_fpr_curve(cat["pool_score_oof"], neg, alpha))
        sizes.append((len(cat["pool_score_oof"]), len(neg)))

    rng = np.random.default_rng(SEED)
    reference: dict[int, list[float]] = {}
    for r in range(REF_REALISATIONS):
        curves = [
            subsample_fpr_curve(rng.random(pool), rng.random(m), alpha, draws=REF_DRAWS, seed=r)
            for pool, m in sizes
        ]
        for n, value in _curve_means(curves).items():
            reference.setdefault(n, []).append(value)

    rows: dict[int, list[dict]] = {}
    for curve in observed:
        for r in curve:
            rows.setdefault(int(r["n"]), []).append(r)
    keys = ("mean", "p5", "p95", "theory_mean", "theory_p5", "theory_p95")
    out = []
    for n, rs in sorted(rows.items()):
        ref = np.array(reference[n])
        lo, hi = percentile_ci(ref)
        row = {
            "n": n,
            "categories": len(rs),
            "guaranteed": all(bool(r["guaranteed"]) for r in rs),
            **{k: float(np.mean([r[k] for r in rs])) for k in keys},
            "reference_mean": float(ref.mean()),
            "reference_interval": [float(lo), float(hi)],
        }
        row["within_reference"] = bool(lo <= row["mean"] <= hi)
        out.append(row)
    return out


def write_scores_csv(cats: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(
            [
                "category",
                "set",
                "image",
                "label",
                "fold",
                "score_full",
                "score_holdout",
                "score_resub",
                "score_oof",
            ]
        )
        for c, cat in cats.items():
            for i, image in enumerate(cat["eval_images"]):
                label = "anomaly" if cat["eval_labels"][i] else "normal"
                full, hold = cat["eval_score_full"][i], cat["eval_score_holdout"][i]
                w.writerow([c, "eval", image, label, "", f"{full:.6f}", f"{hold:.6f}", "", ""])
            for i, image in enumerate(cat["pool_images"]):
                resub, oof = cat["pool_score_resub"][i], cat["pool_score_oof"][i]
                w.writerow(
                    [
                        c,
                        "pool",
                        image,
                        "normal",
                        int(cat["pool_folds"][i]),
                        "",
                        "",
                        f"{resub:.6f}",
                        f"{oof:.6f}",
                    ]
                )


def build_report(run_dir: Path) -> dict:
    meta, cats = load_run(run_dir)
    protocol = meta["protocol"]
    report = {
        "config": meta["config"],
        "protocol": protocol,
        "commit": meta["commit"],
        "n_boot": N_BOOT,
        "accuracy": accuracy(cats),
        "accuracy_holdout_model": {
            "macro_image_auroc": float(
                np.mean([auroc(*_split(cat, "eval_score_holdout")) for cat in cats.values()])
            )
        },
        "calibration": {str(a): calibration(cats, protocol, a) for a in ALPHAS},
        "finite_sample": {str(a): finite_sample(cats, a) for a in ALPHAS},
    }
    report["hypotheses"] = hypotheses(report["calibration"]["0.05"], 0.05)
    return report


def _pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}"


def _span(ci: list[float], digits: int = 2) -> str:
    return f"[{_pct(ci[0], digits)}, {_pct(ci[1], digits)}]"


def format_report(report: dict) -> str:
    acc = report["accuracy"]
    lines = [
        f"config {report['config']['name']}  protocol {report['protocol']}  commit {report['commit']}",
        "",
        "| category | normal | defect | image AUROC | AUPRO | pixel AUROC | FPR@TPR95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for c, v in acc["per_category"].items():
        cells = [
            c,
            str(v["n_normal"]),
            str(v["n_defect"]),
            _pct(v["image_auroc"]),
            _pct(v["aupro"]),
            _pct(v["pixel_auroc"]),
            _pct(v["fpr_at_tpr95"], 1),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    cells = [
        "**mean / pooled**",
        "",
        "",
        f"{_pct(acc['macro_image_auroc'])} {_span(acc['macro_image_auroc_ci'])}",
        f"{_pct(acc['macro_aupro'])} {_span(acc['macro_aupro_ci'])}",
        _pct(acc["macro_pixel_auroc"]),
        f"{_pct(acc['pooled_fpr_at_tpr95'], 1)} {_span(acc['pooled_fpr_at_tpr95_ci'], 1)}",
    ]
    lines += ["| " + " | ".join(cells) + " |", ""]
    header = ["strategy", "n cal", "actual FPR", "bootstrap CI", "Clopper-Pearson", "theory band"]
    header += ["detection rate", "CI"]
    for alpha, cal in report["calibration"].items():
        lines += [
            f"target FPR {float(alpha) * 100:g}%",
            "| " + " | ".join(header) + " |",
            "|" + "---|" * len(header),
        ]
        for s in STRATEGIES:
            v = cal[s]
            cells = [
                s,
                str(v["n_cal_total"]),
                f"{_pct(v['fpr'])}% ({v['fp']}/{v['n_normal']})",
                _span(v["fpr_ci"]),
                _span(v["fpr_clopper_pearson"]),
                _span(v["theory_band"]),
                f"{_pct(v['tpr'])}% ({v['tp']}/{v['n_defect']})",
                _span(v["tpr_ci"]),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        d = cal["tpr_crossfit_minus_holdout"]
        lines += [
            f"detection rate, crossfit - holdout: {d['point'] * 100:+.2f}%p "
            f"[{d['ci'][0] * 100:+.2f}, {d['ci'][1] * 100:+.2f}]",
            "",
        ]
    lines.append("hypotheses (alpha 5%): " + ", ".join(f"{k} {v}" for k, v in report["hypotheses"].items()))
    lines.append("")
    for alpha, rows in report["finite_sample"].items():
        lines += [
            f"finite-sample curve, target {float(alpha) * 100:g}% (mean over categories)",
            "| n | guaranteed | mean FPR | reference mean | reference 95% | within | Beta mean |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            cells = [
                str(r["n"]),
                str(r["guaranteed"]),
                _pct(r["mean"]) + "%",
                _pct(r["reference_mean"]) + "%",
                _span(r["reference_interval"]),
                "yes" if r["within_reference"] else "no",
                _pct(r["theory_mean"]) + "%",
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--stage", default="stage1", help="sub-folder of reports/ to write to")
    parser.add_argument("--no-write", action="store_true", help="print only")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Korean verdict words on a cp949 console
    report = build_report(args.run_dir)
    print(format_report(report))
    if args.no_write:
        return
    out_dir = paths.REPORTS / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report['config']['name']}-{report['protocol']}"
    with open(out_dir / f"{stem}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    _, cats = load_run(args.run_dir)
    write_scores_csv(cats, out_dir / f"{stem}-scores.csv")


if __name__ == "__main__":
    main()
