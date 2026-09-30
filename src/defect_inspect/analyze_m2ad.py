"""Stage 3-B tables and verdicts: M2AD inspectors under synthetic and real illumination changes.

Rates are pooled over the inspectors (one per category and view). Intervals come from a cluster bootstrap
whose unit is the specimen, with the same draws for every condition and statistic, so differences are
paired. H11 and H12 are judged exactly as registered in docs/experiments.md.
"""

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import paths
from .metrics import auroc
from .stats import percentile_ci, stratified_indices

N_BOOT = 2000
SEED = 0
BASE = "S"  # the reference condition every other one is compared with
H12_N = 30
H12_LIMIT = 0.10
JUDGED_METHOD = "p0"
# Pooled rates are ratios of integers, so an interval end that equals the bound of a rule exactly comes out
# a few 1e-17 off in floats. The tolerance keeps "<= limit" and "< 0" what they are in exact arithmetic;
# it is far below the smallest step such an end can take (about 1e-5: one image in a thousand, times the
# weight the percentile interpolation gives to the neighbouring resample).
_TOL = 1e-9

# One condition (or one recalibrated illumination) as seen by every inspector, in inspector order:
# (scores [n_specimens], labels [n_specimens], threshold).
Cell = list[tuple[np.ndarray, np.ndarray, float]]


def _validate(run: dict) -> None:
    inspectors = run["inspectors"]
    if not inspectors:
        raise ValueError("the run has no inspectors")
    first = next(iter(inspectors.values()))
    by_category: dict[str, dict] = {}
    for name, arr in inspectors.items():
        for key in ("conditions", "recal_keys"):
            if arr[key].tolist() != first[key].tolist():
                raise ValueError(f"{name}: {key} differ between inspectors")
        n = len(arr["specimens"])
        if arr["scores"].shape != (len(arr["conditions"]), n) or arr["labels"].shape != arr["scores"].shape:
            raise ValueError(f"{name}: unexpected shape of scores or labels")
        if arr["recal_scores"].shape != (len(arr["recal_keys"]), n):
            raise ValueError(f"{name}: unexpected shape of recal_scores")
        same = by_category.setdefault(arr["category"], arr)
        if same["specimens"].tolist() != arr["specimens"].tolist() or not np.array_equal(
            same["object_anomaly"], arr["object_anomaly"]
        ):
            raise ValueError(f"{name}: specimens differ from the other views of {arr['category']}")
        # Normal specimens are label 0 in every image, anomalous ones 1 or -1: resampling relies on it.
        normal = arr["object_anomaly"] == 0
        for labels in (arr["labels"], arr["recal_labels"]):
            if (labels[:, normal] != 0).any() or not np.isin(labels[:, ~normal], (1, -1)).all():
                raise ValueError(f"{name}: labels do not agree with object_anomaly")


def load(run_dir: Path) -> dict:
    """A `run_m2ad` run: {"meta": run.json, "inspectors": {"<category>_<view>": arrays}} (validated)."""
    run_dir = Path(run_dir)
    with open(run_dir / "run.json", encoding="utf-8") as f:
        meta = json.load(f)
    inspectors = {}
    for info in meta["inspectors"]:
        name = f"{info['category']}_{info['view']}"
        with np.load(run_dir / f"{name}.npz") as z:
            arrays = {k: z[k] for k in z.files}
        arrays["category"], arrays["view"] = info["category"], info["view"]
        inspectors[name] = arrays
    run = {"meta": meta, "inspectors": inspectors}
    _validate(run)
    return run


def conditions(run: dict) -> list[str]:
    return next(iter(run["inspectors"].values()))["conditions"].tolist()


def recal_keys(run: dict) -> list[str]:
    return next(iter(run["inspectors"].values()))["recal_keys"].tolist()


def condition_cell(run: dict, name: str) -> Cell:
    cell = []
    for arr in run["inspectors"].values():
        k = arr["conditions"].tolist().index(name)
        cell.append((arr["scores"][k], arr["labels"][k], float(arr["threshold"])))
    return cell


def recal_cell(run: dict, key: str) -> Cell:
    cell = []
    for arr in run["inspectors"].values():
        k = arr["recal_keys"].tolist().index(key)
        cell.append((arr["recal_scores"][k], arr["recal_labels"][k], float(arr["recal_thresholds"][k])))
    return cell


class ClusterBootstrap:
    """Specimen-level draws shared by every condition and statistic.

    Within each category the normal and the anomalous test specimens are resampled separately with
    replacement (one generator; categories in order, normals then anomalous). A drawn specimen brings its
    images from all views, so `weights[category]` [n_boot, n_specimens] counts how often each was drawn.
    """

    def __init__(self, run: dict, n_boot: int | None = None, seed: int = SEED):
        self.n_boot = int(n_boot or N_BOOT)
        inspectors = run["inspectors"]
        self.names = list(inspectors)
        self.category = {name: arr["category"] for name, arr in inspectors.items()}
        self.categories = list(dict.fromkeys(self.category.values()))
        groups, sizes = [], []
        for category in self.categories:
            arr = next(a for a in inspectors.values() if a["category"] == category)
            anomalous = np.asarray(arr["object_anomaly"]) != 0
            groups.append((np.flatnonzero(~anomalous), np.flatnonzero(anomalous)))
            sizes += [len(group) for group in groups[-1]]
        draws = stratified_indices(sizes, n_boot=self.n_boot, seed=seed)
        rows = np.arange(self.n_boot)[:, None]
        self.weights: dict[str, np.ndarray] = {}
        for i, category in enumerate(self.categories):
            weights = np.zeros((self.n_boot, sum(len(group) for group in groups[i])))
            for members, draw in zip(groups[i], draws[2 * i : 2 * i + 2], strict=True):
                if members.size:
                    np.add.at(weights, (rows, members[draw]), 1.0)
            self.weights[category] = weights

    def counts(self, cells: Sequence[Cell]) -> tuple[np.ndarray, np.ndarray]:
        """(false positives, normal images, detected defects, defect images) summed over `cells`.

        Returns the observed counts [4] and their bootstrap samples [n_boot, 4]. Rule: `score > threshold`.
        """
        point = np.zeros(4)
        samples = np.zeros((self.n_boot, 4))
        for cell in cells:
            for name, (scores, labels, threshold) in zip(self.names, cell, strict=True):
                flagged = np.asarray(scores, dtype=np.float64) > threshold
                normal, defect = labels == 0, labels == 1
                columns = np.stack([flagged & normal, normal, flagged & defect, defect], axis=1)
                columns = columns.astype(np.float64)
                point += columns.sum(axis=0)
                samples += self.weights[self.category[name]] @ columns
        return point, samples

    def auroc(self, cell: Cell) -> tuple[float, np.ndarray]:
        """Mean over inspectors of the image AUROC on labelled images: observed value and samples.

        One rule for both: an inspector without a defect image (or without a normal one) has no AUROC and
        is left out of the mean, in the observed data and in every resample that leaves it none. The
        value is nan only when no inspector has an AUROC.
        """
        values = []
        total = np.zeros(self.n_boot)
        count = np.zeros(self.n_boot)
        for name, (scores, labels, _) in zip(self.names, cell, strict=True):
            s = np.asarray(scores, dtype=np.float64)
            neg, pos = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
            if neg.size == 0 or pos.size == 0:
                continue
            values.append(auroc(s[neg], s[pos]))
            # AUROC of a resample = weighted share of (defect, normal) pairs the defect wins (ties half).
            wins = (s[pos][:, None] > s[neg][None, :]) + 0.5 * (s[pos][:, None] == s[neg][None, :])
            weights = self.weights[self.category[name]]
            w_pos, w_neg = weights[:, pos], weights[:, neg]
            pairs = w_pos.sum(axis=1) * w_neg.sum(axis=1)
            ok = pairs > 0
            total[ok] += ((w_pos[ok] @ wins) * w_neg[ok]).sum(axis=1) / pairs[ok]
            count[ok] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            samples = np.where(count > 0, total / count, np.nan)
        return float(np.mean(values)) if values else float("nan"), samples


def _ratio(num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _ci(samples: np.ndarray) -> list[float]:
    lo, hi = percentile_ci(samples)
    return [float(lo), float(hi)]


def _rates(boot: ClusterBootstrap, cells: Sequence[Cell]) -> dict:
    """Pooled false positive and detection rates over `cells`, observed and resampled."""
    point, samples = boot.counts(cells)
    return {
        "false_positives": int(point[0]),
        "n_normal": int(point[1]),
        "detected": int(point[2]),
        "n_defect": int(point[3]),
        "fpr": float(_ratio(point[0], point[1])),
        "fpr_samples": _ratio(samples[:, 0], samples[:, 1]),
        "tpr": float(_ratio(point[2], point[3])),
        "tpr_samples": _ratio(samples[:, 2], samples[:, 3]),
    }


def _condition_stats(run: dict, boot: ClusterBootstrap) -> dict[str, dict]:
    stats = {}
    for name in conditions(run):
        cell = condition_cell(run, name)
        entry = _rates(boot, [cell])
        entry["auroc"], entry["auroc_samples"] = boot.auroc(cell)
        stats[name] = entry
    if BASE not in stats:
        raise ValueError(f"the run has no condition {BASE!r}")
    return stats


def condition_table(run: dict, n_boot: int | None = None, seed: int = SEED) -> list[dict]:
    """One row per condition: pooled FPR and detection rate, mean AUROC, and the paired differences to S."""
    boot = ClusterBootstrap(run, n_boot, seed)
    stats = _condition_stats(run, boot)
    base = stats[BASE]
    table = []
    for name, entry in stats.items():
        row = {"condition": name}
        row |= {k: entry[k] for k in ("false_positives", "n_normal", "detected", "n_defect")}
        for key in ("fpr", "tpr", "auroc"):
            row[key] = entry[key]
            row[f"{key}_ci"] = _ci(entry[f"{key}_samples"])
        if name != BASE:
            for key in ("fpr", "tpr", "auroc"):
                row[f"d_{key}"] = entry[key] - base[key]
                row[f"d_{key}_ci"] = _ci(entry[f"{key}_samples"] - base[f"{key}_samples"])
        table.append(row)
    return table


def h11(run: dict, n_boot: int | None = None, seed: int = SEED) -> dict:
    """H11: the largest synthetic FPR increase is below half the mean real-illumination FPR increase.

    Statistic = max over the P conditions of (FPR - FPR_S) minus 0.5 x mean over the R conditions of
    (FPR - FPR_S); the max is taken again inside every resample. Supported if the interval lies below 0,
    rejected if it lies above 0, undecided otherwise.
    """
    boot = ClusterBootstrap(run, n_boot, seed)
    stats = _condition_stats(run, boot)
    synthetic = [name for name in stats if name.startswith("P:")]
    real = [name for name in stats if name.startswith("R:")]
    if not synthetic or not real:
        raise ValueError("H11 needs synthetic (P) and real (R) conditions")
    base = stats[BASE]
    d_point = {name: stats[name]["fpr"] - base["fpr"] for name in synthetic + real}
    d_samples = {name: stats[name]["fpr_samples"] - base["fpr_samples"] for name in synthetic + real}
    top = max(synthetic, key=lambda name: d_point[name])  # the first one on ties
    mean_real = float(np.mean([d_point[name] for name in real]))
    max_samples = np.max(np.stack([d_samples[name] for name in synthetic]), axis=0)
    mean_samples = np.mean(np.stack([d_samples[name] for name in real]), axis=0)
    lo, hi = _ci(max_samples - 0.5 * mean_samples)
    # An end that is 0 up to float rounding is 0: it neither lies below nor above.
    verdict = "지지" if hi < -_TOL else "기각" if lo > _TOL else "판정 불가"
    return {
        "statistic": d_point[top] - 0.5 * mean_real,
        "ci": [lo, hi],
        "max_synthetic_condition": top,
        "max_synthetic_increase": d_point[top],
        "max_synthetic_increase_ci": _ci(max_samples),
        "mean_real_increase": mean_real,
        "mean_real_increase_ci": _ci(mean_samples),
        "n_synthetic": len(synthetic),
        "n_real": len(real),
        "verdict": verdict,
    }


def _recal_lights(run: dict, n: int) -> list[str]:
    return [key.split(":")[0] for key in recal_keys(run) if int(key.split(":")[1]) == n]


def h12(
    run: dict, n: int = H12_N, limit: float = H12_LIMIT, n_boot: int | None = None, seed: int = SEED
) -> dict:
    """H12: after recalibration with n specimens the FPR pooled over the new illuminations is <= limit.

    Supported if the interval's upper end <= limit, rejected if its lower end > limit, undecided otherwise.
    """
    lights = _recal_lights(run, n)
    if not lights:
        raise ValueError(f"the run has no recalibration with n = {n}")
    boot = ClusterBootstrap(run, n_boot, seed)
    rates = _rates(boot, [recal_cell(run, f"{light}:{n}") for light in lights])
    lo, hi = _ci(rates["fpr_samples"])
    verdict = "지지" if hi <= limit + _TOL else "기각" if lo > limit + _TOL else "판정 불가"
    return {
        "n": n,
        "limit": limit,
        "illuminations": lights,
        "false_positives": rates["false_positives"],
        "n_normal": rates["n_normal"],
        "fpr": rates["fpr"],
        "fpr_ci": [lo, hi],
        "verdict": verdict,
    }


def recal_table(run: dict, n_boot: int | None = None, seed: int = SEED) -> list[dict]:
    """Pooled FPR and detection rate over the real illuminations for n = 0 (no recalibration) and each n.

    Rows with n > 0 also carry the paired differences to n = 0.
    """
    boot = ClusterBootstrap(run, n_boot, seed)
    sizes = sorted({int(key.split(":")[1]) for key in recal_keys(run)})
    real = [name.split(":")[1] for name in conditions(run) if name.startswith("R:")]
    table: list[dict] = []
    base = None
    for n in [0, *sizes]:
        if n == 0:
            lights = real
            cells = [condition_cell(run, f"R:{light}") for light in lights]
        else:
            lights = _recal_lights(run, n)
            cells = [recal_cell(run, f"{light}:{n}") for light in lights]
        rates = _rates(boot, cells)
        row = {"n": n, "illuminations": lights}
        row |= {k: rates[k] for k in ("false_positives", "n_normal", "detected", "n_defect")}
        for key in ("fpr", "tpr"):
            row[key] = rates[key]
            row[f"{key}_ci"] = _ci(rates[f"{key}_samples"])
        if n == 0:
            base = rates
        elif lights == real:
            for key in ("fpr", "tpr"):
                row[f"d_{key}"] = rates[key] - base[key]
                row[f"d_{key}_ci"] = _ci(rates[f"{key}_samples"] - base[f"{key}_samples"])
        table.append(row)
    return table


def inspector_table(run: dict) -> dict[str, dict]:
    """Supplementary, no verdicts: per inspector its threshold and each condition's counts and AUROC."""
    out = {}
    for name, arr in run["inspectors"].items():
        threshold = float(arr["threshold"])
        per_condition = {}
        for k, condition in enumerate(arr["conditions"].tolist()):
            scores, labels = arr["scores"][k].astype(np.float64), arr["labels"][k]
            flagged = scores > threshold
            per_condition[condition] = {
                "false_positives": int((flagged & (labels == 0)).sum()),
                "n_normal": int((labels == 0).sum()),
                "detected": int((flagged & (labels == 1)).sum()),
                "n_defect": int((labels == 1).sum()),
                "auroc": float(auroc(scores[labels == 0], scores[labels == 1])),
            }
        out[name] = {"threshold": threshold, "n_cal": int(len(arr["cal_score"])), "conditions": per_condition}
    return out


def build_report(runs: dict[str, dict], n_boot: int | None = None, seed: int = SEED) -> dict:
    """Tables of every method; H11 and H12 are judged on JUDGED_METHOD only."""
    report: dict = {
        "n_boot": int(n_boot or N_BOOT),
        "seed": seed,
        "judged_method": JUDGED_METHOD,
        "methods": {},
    }
    for name, run in runs.items():
        meta = run["meta"]
        entry = {
            "commit": meta.get("commit"),
            "device": meta.get("device"),
            "check": bool(meta.get("check", False)),
            "inspectors": list(run["inspectors"]),
            "conditions": condition_table(run, n_boot, seed),
            "recal": recal_table(run, n_boot, seed),
            "per_inspector": inspector_table(run),
        }
        has_real = any(c.startswith("R:") for c in conditions(run))
        if has_real and any(c.startswith("P:") for c in conditions(run)):
            entry["h11"] = h11(run, n_boot, seed)
        if _recal_lights(run, H12_N):
            entry["h12"] = h12(run, H12_N, H12_LIMIT, n_boot, seed)
        if name != JUDGED_METHOD:
            # Reported next to the judged method; no verdict is attached to these numbers.
            for key in ("h11", "h12"):
                entry.get(key, {}).pop("verdict", None)
        report["methods"][name] = entry
    judged = report["methods"].get(JUDGED_METHOD, {})
    report["hypotheses"] = {
        key.upper(): judged[key]["verdict"] for key in ("h11", "h12") if "verdict" in judged.get(key, {})
    }
    return report


def strict_json(value):
    """Copy of a report in which every non-finite float is None, so the file is strict JSON.

    A rate without images or an AUROC without defect images is nan in the report; `json` would write the
    token NaN, which other JSON readers reject.
    """
    if isinstance(value, dict):
        return {key: strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}"


def _span(ci: list[float], digits: int = 1) -> str:
    return f"[{_pct(ci[0], digits)}, {_pct(ci[1], digits)}]"


def _diff(row: dict, key: str) -> str:
    if f"d_{key}" not in row:
        return "-"
    return f"{row[f'd_{key}'] * 100:+.1f} {_span(row[f'd_{key}_ci'])}"


def format_report(report: dict) -> str:
    lines = []
    for name, entry in report["methods"].items():
        lines += [
            f"## {name} ({len(entry['inspectors'])} inspectors, commit {entry['commit']})",
            "| condition | normal | defect | FPR % | detection % | AUROC % | dFPR %p | d detection %p "
            "| dAUROC %p |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for row in entry["conditions"]:
            cells = [
                row["condition"],
                str(row["n_normal"]),
                str(row["n_defect"]),
                f"{_pct(row['fpr'])} {_span(row['fpr_ci'])}",
                f"{_pct(row['tpr'])} {_span(row['tpr_ci'])}",
                f"{_pct(row['auroc'])} {_span(row['auroc_ci'])}",
                _diff(row, "fpr"),
                _diff(row, "tpr"),
                _diff(row, "auroc"),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines += [
            "",
            "real illuminations pooled, by recalibration size n (0 = not recalibrated)",
            "| n | normal | defect | FPR % | detection % | dFPR %p | d detection %p |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in entry["recal"]:
            cells = [
                str(row["n"]),
                str(row["n_normal"]),
                str(row["n_defect"]),
                f"{_pct(row['fpr'])} {_span(row['fpr_ci'])}",
                f"{_pct(row['tpr'])} {_span(row['tpr_ci'])}",
                _diff(row, "fpr"),
                _diff(row, "tpr"),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        if "h11" in entry:
            h = entry["h11"]
            lines.append(
                "H11 statistic (max synthetic increase - half the mean real increase): "
                f"{h['statistic'] * 100:+.1f}%p {_span(h['ci'])}; "
                f"max synthetic {h['max_synthetic_condition']} {h['max_synthetic_increase'] * 100:+.1f}%p, "
                f"mean real {h['mean_real_increase'] * 100:+.1f}%p"
                + (f" -> {h['verdict']}" if "verdict" in h else "")
            )
        if "h12" in entry:
            h = entry["h12"]
            lines.append(
                f"H12 pooled FPR after recalibration with n = {h['n']}: "
                f"{_pct(h['fpr'])}% {_span(h['fpr_ci'])} ({h['false_positives']}/{h['n_normal']}), "
                f"limit {_pct(h['limit'])}%" + (f" -> {h['verdict']}" if "verdict" in h else "")
            )
        lines.append("")
    if report["hypotheses"]:
        verdicts = "; ".join(f"{k}: {v}" for k, v in report["hypotheses"].items())
        lines.append(f"hypotheses ({report['judged_method']}): {verdicts}")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="*", default=["p0", "d-s"], help="runs outputs/m2ad-<method>")
    parser.add_argument(
        "--check", action="store_true", help="read the code-check runs (outputs/m2ad-<method>-check); no file"
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    suffix = "-check" if args.check else ""
    runs = {name: load(paths.OUTPUTS / f"m2ad-{name}{suffix}") for name in args.methods}
    report = build_report(runs)
    print(format_report(report), end="")
    if not (args.no_write or args.check):
        out_dir = paths.REPORTS / "stage3"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "m2ad.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(strict_json(report), f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")


if __name__ == "__main__":
    main()
