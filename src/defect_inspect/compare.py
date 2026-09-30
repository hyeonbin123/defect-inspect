"""Stage 2 comparison: unsupervised methods against each other and against the supervised head.

Every method is loaded into one shape (per category: evaluation labels, image scores, scores of held-out
normals for the threshold, pixel metrics), then compared with the same stratified bootstrap draws so that
differences are paired. Hypotheses H5-H8 are judged exactly as registered in docs/experiments.md.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import paths
from .calibrate import conformal_threshold
from .metrics import ProHistograms, aupro_bootstrap, auroc, fpr_at_tpr
from .stats import percentile_ci, stratified_indices

N_BOOT = 2000
SEED = 0
ALPHA = 0.05
MIN_EFFECT_AUROC = 0.01
MIN_UNSEEN_PER_CATEGORY = 3
MIN_UNSEEN_POOLED = 30


@dataclass
class MethodRun:
    name: str
    calibration: str  # how cal_score was obtained: "crossfit", "holdout" or "none"
    cats: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def load_patchcore(run_dir: Path, name: str | None = None) -> MethodRun:
    """A `run_patchcore` run: the full-bank model with its cross-fitted calibration scores."""
    with open(run_dir / "run.json", encoding="utf-8") as f:
        meta = json.load(f)
    run = MethodRun(name or meta["config"]["name"], "crossfit")
    for info in meta["categories"]:
        cat = _load_npz(run_dir / f"{info['category']}.npz")
        cat["eval_score"] = cat["eval_score_full"]
        cat["cal_score"] = cat["pool_score_oof"]
        run.cats[info["category"]] = cat
    return run


def load_common(run_dir: Path, name: str, categories: list[str] | None = None) -> MethodRun:
    """A run in the common stage 2 format (Dinomaly, one supervised (k, seed) folder)."""
    files = sorted(p for p in run_dir.glob("*.npz"))
    names = categories or [p.stem for p in files]
    run = MethodRun(name, "holdout")
    for c in names:
        run.cats[c] = _load_npz(run_dir / f"{c}.npz")
    if all(len(cat["cal_score"]) == 0 for cat in run.cats.values()):
        run.calibration = "none"
    return run


def _neg_pos(cat: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    labels = cat["eval_labels"]
    return cat["eval_score"][labels == 0], cat["eval_score"][labels == 1]


def check_same_images(runs: list[MethodRun]) -> list[str]:
    """All runs must score the same evaluation images in the same order; returns the category list."""
    names = list(runs[0].cats)
    for run in runs[1:]:
        if list(run.cats) != names:
            raise ValueError(f"{run.name} has categories {list(run.cats)}, expected {names}")
        for c in names:
            if not np.array_equal(run.cats[c]["eval_images"], runs[0].cats[c]["eval_images"]):
                raise ValueError(f"{run.name} and {runs[0].name} evaluate different images in {c}")
    return names


class Bootstrap:
    """One set of stratified draws (normals and defects of each category) shared by every statistic."""

    def __init__(self, reference: MethodRun, n_boot: int | None = None, seed: int = SEED):
        n_boot = n_boot or N_BOOT
        self.names = list(reference.cats)
        self.n_boot = n_boot
        self.sizes = []
        for c in self.names:
            labels = reference.cats[c]["eval_labels"]
            self.sizes += [int((labels == 0).sum()), int((labels == 1).sum())]
        self.draws = stratified_indices(self.sizes, n_boot=n_boot, seed=seed)

    def indices(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return self.draws[2 * i], self.draws[2 * i + 1]

    def macro_auroc(self, run: MethodRun, keep_pos: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Bootstrap samples of the mean over categories of the image AUROC.

        `keep_pos[c]` (bool over the category's defects) restricts the defects; categories it leaves out
        entirely are skipped, and a resample that draws none of the kept defects skips that category.
        """
        total = np.zeros(self.n_boot)
        count = np.zeros(self.n_boot)
        for i, c in enumerate(self.names):
            if keep_pos is not None and c not in keep_pos:
                continue
            neg, pos = _neg_pos(run.cats[c])
            neg_idx, pos_idx = self.indices(i)
            for b in range(self.n_boot):
                p_idx = pos_idx[b]
                if keep_pos is not None:
                    p_idx = p_idx[keep_pos[c][p_idx]]
                    if len(p_idx) == 0:
                        continue
                total[b] += auroc(neg[neg_idx[b]], pos[p_idx])
                count[b] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, total / count, np.nan)

    def macro_aupro(self, run: MethodRun) -> np.ndarray:
        out = np.zeros(self.n_boot)
        rows = np.arange(self.n_boot)[:, None]
        for i, c in enumerate(self.names):
            cat = run.cats[c]
            hist = ProHistograms(
                edges=cat["pro_edges"],
                normal=cat["pro_normal"],
                components=cat["pro_components"],
                component_image=cat["pro_component_image"],
            )
            neg_idx, pos_idx = self.indices(i)
            n_neg, n_pos = self.sizes[2 * i], self.sizes[2 * i + 1]
            counts = np.zeros((self.n_boot, n_neg + n_pos), dtype=np.int64)
            np.add.at(counts, (rows, neg_idx), 1)
            np.add.at(counts, (rows, n_neg + pos_idx), 1)
            out += aupro_bootstrap(hist, counts) / len(self.names)
        return out

    def pooled_rates(self, run: MethodRun, alpha: float) -> dict | None:
        """Pooled false positive and detection rates of per-category thresholds fixed from `cal_score`."""
        if run.calibration == "none":
            return None
        fp = tp = n_neg = n_pos = 0
        fp_boot = np.zeros(self.n_boot)
        tp_boot = np.zeros(self.n_boot)
        guaranteed = 0
        for i, c in enumerate(self.names):
            cat = run.cats[c]
            thr = conformal_threshold(cat["cal_score"], alpha)
            guaranteed += int(thr.guaranteed)
            neg, pos = _neg_pos(cat)
            f_neg, f_pos = neg > thr.value, pos > thr.value
            neg_idx, pos_idx = self.indices(i)
            fp += int(f_neg.sum())
            tp += int(f_pos.sum())
            n_neg += len(neg)
            n_pos += len(pos)
            fp_boot += f_neg[neg_idx].sum(axis=1)
            tp_boot += f_pos[pos_idx].sum(axis=1)
        return {
            "calibration": run.calibration,
            "n_cal": int(sum(len(cat["cal_score"]) for cat in run.cats.values())),
            "categories_guaranteed": guaranteed,
            "fpr": fp / n_neg,
            "fpr_ci": _ci(fp_boot / n_neg),
            "tpr": tp / n_pos,
            "tpr_ci": _ci(tp_boot / n_pos),
        }


def _ci(samples: np.ndarray) -> list[float]:
    lo, hi = percentile_ci(samples)
    return [float(lo), float(hi)]


def macro_auroc(run: MethodRun, keep_pos: dict[str, np.ndarray] | None = None) -> float:
    values = []
    for c, cat in run.cats.items():
        neg, pos = _neg_pos(cat)
        if keep_pos is not None:
            if c not in keep_pos:
                continue
            pos = pos[keep_pos[c]]
        values.append(auroc(neg, pos))
    return float(np.mean(values)) if values else float("nan")


def summarize(run: MethodRun, boot: Bootstrap) -> dict:
    per_cat = {}
    fp95 = n_neg = 0
    for c, cat in run.cats.items():
        neg, pos = _neg_pos(cat)
        fpr, _, fp = fpr_at_tpr(neg, pos, 0.95)
        fp95 += fp
        n_neg += len(neg)
        per_cat[c] = {
            "image_auroc": float(auroc(neg, pos)),
            "aupro": float(cat["aupro"]),
            "pixel_auroc": float(cat["pixel_auroc"]),
            "fpr_at_tpr95": float(fpr),
        }
    return {
        "name": run.name,
        "macro_image_auroc": float(np.mean([v["image_auroc"] for v in per_cat.values()])),
        "macro_image_auroc_ci": _ci(boot.macro_auroc(run)),
        "macro_aupro": float(np.mean([v["aupro"] for v in per_cat.values()])),
        "macro_aupro_ci": _ci(boot.macro_aupro(run)),
        "macro_pixel_auroc": float(np.mean([v["pixel_auroc"] for v in per_cat.values()])),
        "pooled_fpr_at_tpr95": fp95 / n_neg,
        "fixed_threshold": boot.pooled_rates(run, ALPHA),
        "per_category": per_cat,
    }


def difference(point: float, samples: np.ndarray, min_effect: float = MIN_EFFECT_AUROC) -> dict:
    """Paired difference with its interval and the registered three-way wording."""
    lo, hi = _ci(samples)
    if not (lo > 0 or hi < 0):
        verdict = "판정 불가"
    elif abs(point) >= min_effect:
        verdict = "차이 있음"
    else:
        verdict = "차이 작음"
    return {"diff": float(point), "ci": [lo, hi], "verdict": verdict}


def unseen_masks(sup: MethodRun) -> tuple[dict[str, np.ndarray], int]:
    """Per category: which test defects share no defect type with the run's training defects.

    Categories with fewer than MIN_UNSEEN_PER_CATEGORY such defects are left out. Also returns the number
    of unseen defects kept over all categories.
    """
    masks, total = {}, 0
    for c, cat in sup.cats.items():
        seen: set[str] = set()
        for entry in cat["train_defect_types"]:
            seen.update(t for t in str(entry).split("|") if t)
        types = cat["eval_defect_types"][cat["eval_labels"] == 1]
        mask = np.array([not (set(str(t).split("|")) & seen) for t in types], dtype=bool)
        if mask.sum() >= MIN_UNSEEN_PER_CATEGORY:
            masks[c] = mask
            total += int(mask.sum())
    return masks, total


def supervised_vs_reference(
    sup_runs: dict[tuple[int, int], MethodRun], reference: MethodRun, boot: Bootstrap
) -> dict[int, dict]:
    """Per k: supervised (mean over seeds) minus the reference method, on all defects and on unseen types."""
    ref_point = macro_auroc(reference)
    ref_boot = boot.macro_auroc(reference)
    out = {}
    for k in sorted({k for k, _ in sup_runs}):
        seeds = sorted(s for kk, s in sup_runs if kk == k)
        points = [macro_auroc(sup_runs[k, s]) for s in seeds]
        boots = np.mean([boot.macro_auroc(sup_runs[k, s]) for s in seeds], axis=0)
        row = {
            "k": k,
            "labels_with_validation": k + 20,
            "seeds": seeds,
            "supervised_macro_image_auroc": float(np.mean(points)),
            "supervised_per_seed": [float(p) for p in points],
            "supervised_ci": _ci(boots),
            "reference_macro_image_auroc": ref_point,
            "all_defects": difference(float(np.mean(points)) - ref_point, boots - ref_boot),
        }

        # Unseen defect types: the subset depends on the run, the reference is scored on the same subset.
        sup_pts, ref_pts, sup_b, ref_b, kept, cats_used = [], [], [], [], [], []
        for s in seeds:
            masks, total = unseen_masks(sup_runs[k, s])
            kept.append(total)
            cats_used.append(len(masks))
            if not masks:
                continue
            sup_pts.append(macro_auroc(sup_runs[k, s], masks))
            ref_pts.append(macro_auroc(reference, masks))
            sup_b.append(boot.macro_auroc(sup_runs[k, s], masks))
            ref_b.append(boot.macro_auroc(reference, masks))
        unseen = {"unseen_defects_mean": float(np.mean(kept)), "categories_mean": float(np.mean(cats_used))}
        if np.mean(kept) >= MIN_UNSEEN_POOLED and sup_pts:
            diff_boot = np.nanmean(np.array(sup_b) - np.array(ref_b), axis=0)
            unseen.update(
                supervised=float(np.mean(sup_pts)),
                reference=float(np.mean(ref_pts)),
                **difference(float(np.mean(sup_pts) - np.mean(ref_pts)), diff_boot),
            )
            unseen["judged"] = True
        else:
            unseen["judged"] = False
        row["unseen_types"] = unseen
        out[k] = row
    return out


def hypotheses(sup: dict[int, dict], pairs: dict[str, dict]) -> dict[str, str]:
    """H5-H8 as registered. `pairs` holds the unsupervised differences against P0 by method name."""

    def lower(row: dict) -> str:  # supervised below the reference
        d = row["all_defects"]
        if d["ci"][1] < 0 and abs(d["diff"]) >= MIN_EFFECT_AUROC:
            return "지지"
        return "기각" if d["ci"][0] > 0 else "판정 불가"

    def higher(row: dict) -> str:
        d = row["all_defects"]
        if d["ci"][0] > 0 and abs(d["diff"]) >= MIN_EFFECT_AUROC:
            return "지지"
        return "기각" if d["ci"][1] < 0 else "판정 불가"

    out = {}
    for k in (5, 10):
        if k in sup:
            out[f"H5 (k={k})"] = lower(sup[k])
    if 40 in sup:
        out["H6 (k=40)"] = higher(sup[40])
    crossing = [k for k, row in sorted(sup.items()) if higher(row) == "지지"]
    out["교차점"] = f"k = {crossing[0]}" if crossing else "k = 40까지 교차 없음"
    judged = {k: row["unseen_types"] for k, row in sup.items() if row["unseen_types"]["judged"]}
    if not judged:
        out["H7"] = "판정 불가 (보지 못한 유형의 결함이 기준 장수에 못 미침)"
    elif all(u["ci"][1] <= 0 for u in judged.values()):
        out["H7"] = "지지"
    elif any(u["ci"][0] > 0 for u in judged.values()):
        out["H7"] = "기각 (k = " + ", ".join(str(k) for k, u in judged.items() if u["ci"][0] > 0) + ")"
    else:
        out["H7"] = "판정 불가"
    for name, d in pairs.items():
        if d["verdict"] == "차이 있음":
            out[f"H8 ({name} > P0)"] = "지지" if d["diff"] > 0 else "기각"
        else:
            out[f"H8 ({name} > P0)"] = d["verdict"]
    return out


def build_report(
    unsupervised: list[MethodRun],
    sup_runs: dict[tuple[int, int], MethodRun],
    reference_name: str,
    baseline_name: str = "p0",
) -> dict:
    runs = unsupervised + list(sup_runs.values())
    check_same_images(runs)
    boot = Bootstrap(unsupervised[0])
    by_name = {r.name: r for r in unsupervised}
    summaries = {r.name: summarize(r, boot) for r in unsupervised}
    pairs = {}
    if baseline_name in by_name:
        base_boot = boot.macro_auroc(by_name[baseline_name])
        base_point = macro_auroc(by_name[baseline_name])
        for r in unsupervised:
            if r.name != baseline_name:
                pairs[r.name] = difference(macro_auroc(r) - base_point, boot.macro_auroc(r) - base_boot)
    report = {
        "n_boot": N_BOOT,
        "alpha": ALPHA,
        "unsupervised": summaries,
        "vs_baseline": {"baseline": baseline_name, "pairs": pairs},
        "reference_unsupervised": reference_name,
    }
    if sup_runs:
        sup = supervised_vs_reference(sup_runs, by_name[reference_name], boot)
        report["supervised"] = {str(k): v for k, v in sup.items()}
        report["supervised_runs"] = {
            f"k{k}-s{s}": summarize(run, boot) for (k, s), run in sorted(sup_runs.items())
        }
        report["hypotheses"] = hypotheses(sup, pairs)
    return report


def _pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}"


def _span(ci: list[float], digits: int = 2) -> str:
    return f"[{_pct(ci[0], digits)}, {_pct(ci[1], digits)}]"


def format_report(report: dict) -> str:
    lines = [
        "| method | image AUROC | AUPRO | pixel AUROC | FPR@TPR95 | calibration | actual FPR (5%) "
        "| detection |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in report["unsupervised"].values():
        fixed = s["fixed_threshold"]
        cells = [
            s["name"],
            f"{_pct(s['macro_image_auroc'])} {_span(s['macro_image_auroc_ci'])}",
            f"{_pct(s['macro_aupro'])} {_span(s['macro_aupro_ci'])}",
            _pct(s["macro_pixel_auroc"]),
            _pct(s["pooled_fpr_at_tpr95"], 1),
        ]
        if fixed:
            cells += [
                f"{fixed['calibration']} (n={fixed['n_cal']})",
                f"{_pct(fixed['fpr'])} {_span(fixed['fpr_ci'])}",
                f"{_pct(fixed['tpr'])} {_span(fixed['tpr_ci'])}",
            ]
        else:
            cells += ["-", "-", "-"]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    base = report["vs_baseline"]["baseline"]
    for name, d in report["vs_baseline"]["pairs"].items():
        lines.append(f"{name} - {base}: {d['diff'] * 100:+.2f}%p {_span(d['ci'])} {d['verdict']}")
    if "supervised" in report:
        lines += [
            "",
            f"supervised head against {report['reference_unsupervised']} (image AUROC, mean over categories)",
            "| k | labels incl. validation | supervised | difference | verdict | unseen-type defects "
            "| difference | verdict |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in report["supervised"].values():
            d, u = row["all_defects"], row["unseen_types"]
            cells = [
                str(row["k"]),
                str(row["labels_with_validation"]),
                f"{_pct(row['supervised_macro_image_auroc'])} {_span(row['supervised_ci'])}",
                f"{d['diff'] * 100:+.2f}%p {_span(d['ci'])}",
                d["verdict"],
                f"{u['unseen_defects_mean']:.0f}",
            ]
            cells += (
                [f"{u['diff'] * 100:+.2f}%p {_span(u['ci'])}", u["verdict"]] if u["judged"] else ["-", "-"]
            )
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", "hypotheses: " + "; ".join(f"{k}: {v}" for k, v in report["hypotheses"].items())]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", choices=["dev", "test"], required=True)
    parser.add_argument("--patchcore", nargs="*", default=["p0", "d-s"], help="PatchCore config names")
    parser.add_argument("--dinomaly", action="store_true", help="include outputs/dm-<protocol>")
    parser.add_argument("--supervised", action="store_true", help="include outputs/sup-<protocol>")
    parser.add_argument("--ks", nargs="*", type=int, default=[5, 10, 20, 40])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--reference", default=None, help="reference unsupervised method (dev default: best macro AUROC)"
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    if args.protocol == "test" and not args.reference:
        parser.error("--protocol test needs --reference (the unsupervised method chosen on dev)")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    runs = [load_patchcore(paths.OUTPUTS / f"{name}-{args.protocol}") for name in args.patchcore]
    names = list(runs[0].cats)
    if args.dinomaly:
        runs.append(load_common(paths.OUTPUTS / f"dm-{args.protocol}", "dm", names))
    sup_runs = {}
    if args.supervised:
        for k in args.ks:
            for s in args.seeds:
                folder = paths.OUTPUTS / f"sup-{args.protocol}" / f"k{k}-s{s}"
                sup_runs[k, s] = load_common(folder, f"sup-k{k}-s{s}", names)
    reference = args.reference or max(runs, key=macro_auroc).name
    report = build_report(runs, sup_runs, reference)
    report["protocol"] = args.protocol
    report["reference_chosen_by"] = "argument" if args.reference else f"best macro AUROC on {args.protocol}"
    print(format_report(report))
    if not args.no_write:
        out_dir = paths.REPORTS / "stage2"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"compare-{args.protocol}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
