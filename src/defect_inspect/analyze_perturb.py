"""Stage 3-A analysis: accuracy and fixed-threshold rates under the synthetic conditions, against clean.

Reads the runs of `defect_inspect.run_perturb`. Thresholds are fixed once per category from the source
run's calibration scores and never recomputed. Every condition is resampled with the same stratified
bootstrap draws, so the differences to the clean condition are paired. H9 is judged exactly as registered
in docs/experiments.md.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from . import paths
from .calibrate import conformal_threshold
from .conditions import CLEAN
from .metrics import auroc
from .stats import percentile_ci, stratified_indices
from .visa import CATEGORIES  # the full set, in the order the registered bootstrap walks through

N_BOOT = 2000
SEED = 0
ALPHA = 0.05
MAX_AUROC_DROP = 0.01  # H9: the mean image AUROC falls by less than 1.0 point
METHODS = ("p0", "d-s", "dm")
H9_METHOD = "p0"
_STATS = ("auroc", "fpr", "tpr")


def load(run_dir: Path) -> dict:
    """A `run_perturb` run: {"meta": run.json, "cats": {category: arrays}}, categories in run order."""
    run_dir = Path(run_dir)
    with open(run_dir / "run.json", encoding="utf-8") as f:
        meta = json.load(f)
    cats = {}
    for info in meta["categories"]:
        with np.load(run_dir / f"{info['category']}.npz") as z:
            cats[info["category"]] = {k: z[k] for k in z.files}
    return {"meta": meta, "cats": cats}


def _condition_names(cats: dict[str, dict[str, np.ndarray]]) -> list[str]:
    """The condition list shared by every category, after checking the shapes of the score arrays."""
    if not cats:
        raise ValueError("the run has no categories")
    names: list[str] | None = None
    for c, cat in cats.items():
        mine = [str(name) for name in cat["conditions"]]
        if names is None:
            names = mine
        elif mine != names:
            raise ValueError(f"{c} lists other conditions than {next(iter(cats))}")
        if cat["scores"].shape != (len(mine), len(cat["eval_labels"])):
            raise ValueError(f"{c}: scores {cat['scores'].shape} do not match conditions x images")
        if not np.isfinite(cat["scores"]).all():
            raise ValueError(f"{c}: scores are not finite")
    assert names is not None
    if names.count(CLEAN) != 1:
        raise ValueError(f"the run needs exactly one {CLEAN!r} condition, got {names}")
    return names


def _thresholds(cats: dict[str, dict[str, np.ndarray]], alpha: float) -> dict | None:
    """Per-category conformal thresholds from `cal_score`; None when no category has calibration scores."""
    empty = [c for c, cat in cats.items() if len(cat["cal_score"]) == 0]
    if len(empty) == len(cats):
        return None
    if empty:
        raise ValueError(f"calibration scores are missing for {empty} only: the run is inconsistent")
    return {c: conformal_threshold(cat["cal_score"], alpha) for c, cat in cats.items()}


def _auroc_rows(neg: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """`metrics.auroc` of every resample: neg [B, n_neg], pos [B, n_pos] -> [B] (nan without both classes)."""
    n_neg, n_pos = neg.shape[1], pos.shape[1]
    if n_neg == 0 or n_pos == 0:
        return np.full(neg.shape[0], np.nan)
    ranks = rankdata(np.concatenate([neg, pos], axis=1), axis=1)  # average ranks within each resample
    u = ranks[:, n_neg:].sum(axis=1) - n_pos * (n_pos + 1) / 2
    return u / (n_neg * n_pos)


def _ci(samples: np.ndarray) -> list[float]:
    lo, hi = percentile_ci(samples)
    return [float(lo), float(hi)]


def _rate(count: float | np.ndarray, total: int) -> float | np.ndarray:
    return count / total if total else count * float("nan")


def condition_table(run: dict, alpha: float = 0.05, n_boot: int | None = None, seed: int = 0) -> list[dict]:
    """One row per condition: mean image AUROC and the pooled rates at the fixed thresholds.

    Rows hold `condition`, `macro_image_auroc`, `fpr`, `tpr` (pooled over categories, rule
    `score > threshold`; None when the run has no calibration scores) with `_ci` intervals, and every row
    but the clean one the differences to clean `d_auroc`, `d_fpr`, `d_tpr` with `_ci`. The bootstrap
    resamples the normals and the defects of each category separately (groups in category order) and
    uses the same draws for every condition.
    """
    n_boot = n_boot or N_BOOT
    cats = run["cats"]
    names = _condition_names(cats)
    clean = names.index(CLEAN)
    thresholds = _thresholds(cats, alpha)

    masks, sizes = [], []
    for cat in cats.values():
        labels = cat["eval_labels"]
        masks.append((labels == 0, labels == 1))
        sizes += [int((labels == 0).sum()), int((labels == 1).sum())]
    draws = stratified_indices(sizes, n_boot=n_boot, seed=seed)
    n_neg, n_pos = sum(sizes[0::2]), sum(sizes[1::2])

    points: list[dict[str, float]] = []
    boots: list[dict[str, np.ndarray]] = []
    for k in range(len(names)):
        per_cat = []
        auc_boot = np.zeros(n_boot)
        fp = tp = 0
        fp_boot = np.zeros(n_boot)
        tp_boot = np.zeros(n_boot)
        for i, (c, cat) in enumerate(cats.items()):
            scores = cat["scores"][k].astype(np.float64)
            neg, pos = scores[masks[i][0]], scores[masks[i][1]]
            neg_idx, pos_idx = draws[2 * i], draws[2 * i + 1]
            per_cat.append(auroc(neg, pos))
            auc_boot += _auroc_rows(neg[neg_idx], pos[pos_idx])
            if thresholds is not None:
                # Fixed once from the source run's calibration scores, the same for every condition.
                f_neg, f_pos = neg > thresholds[c].value, pos > thresholds[c].value
                fp += int(f_neg.sum())
                tp += int(f_pos.sum())
                fp_boot += f_neg[neg_idx].sum(axis=1)
                tp_boot += f_pos[pos_idx].sum(axis=1)
        point = {"auroc": float(np.mean(per_cat))}
        boot = {"auroc": auc_boot / len(cats)}
        if thresholds is not None:
            point.update(fpr=float(_rate(fp, n_neg)), tpr=float(_rate(tp, n_pos)))
            boot.update(fpr=_rate(fp_boot, n_neg), tpr=_rate(tp_boot, n_pos))
        points.append(point)
        boots.append(boot)

    rows = []
    for k, name in enumerate(names):
        point, boot = points[k], boots[k]
        row: dict = {
            "condition": name,
            "macro_image_auroc": point["auroc"],
            "macro_image_auroc_ci": _ci(boot["auroc"]),
        }
        for key in ("fpr", "tpr"):
            row[key] = point.get(key)
            row[f"{key}_ci"] = _ci(boot[key]) if key in boot else None
        if k != clean:
            for key in _STATS:
                if key in point:
                    row[f"d_{key}"] = point[key] - points[clean][key]
                    row[f"d_{key}_ci"] = _ci(boot[key] - boots[clean][key])  # paired: same draws
                else:
                    row[f"d_{key}"] = row[f"d_{key}_ci"] = None
        rows.append(row)
    return rows


def h9(table: list[dict], alpha: float = 0.05) -> dict:
    """H9 as registered, on point estimates: is there a condition that accuracy does not reveal?

    The conditions whose mean image AUROC falls by less than 1.0 point (`d_auroc > -0.01`) while the
    pooled false positive rate is at least twice the target (`fpr >= 2 * alpha`). Supported if there is
    at least one, rejected otherwise.
    """
    found = []
    for row in table:
        if "d_auroc" not in row:  # the clean row
            continue
        if row["fpr"] is None:
            raise ValueError("H9 needs fixed thresholds, but the run has no calibration scores")
        if row["d_auroc"] > -MAX_AUROC_DROP and row["fpr"] >= 2 * alpha:
            found.append(row["condition"])
    return {"verdict": "지지" if found else "기각", "conditions": found}


def describe(run: dict, alpha: float = ALPHA) -> dict:
    """Where a run came from and how many images and calibration scores stand behind its table."""
    meta, cats = run["meta"], run["cats"]
    thresholds = _thresholds(cats, alpha)
    diffs = [info.get("clean_max_rel_diff") for info in meta.get("categories", [])]
    return {
        "commit": meta.get("commit"),
        "device": meta.get("device"),
        "source": meta.get("source"),
        "source_commit": meta.get("source_commit"),
        "categories": list(cats),
        "n_normal": int(sum((cat["eval_labels"] == 0).sum() for cat in cats.values())),
        "n_defect": int(sum((cat["eval_labels"] == 1).sum() for cat in cats.values())),
        "n_cal": int(sum(len(cat["cal_score"]) for cat in cats.values())),
        "categories_guaranteed": (
            None if thresholds is None else int(sum(t.guaranteed for t in thresholds.values()))
        ),
        "clean_max_rel_diff": max((d for d in diffs if d is not None), default=None),
    }


def build_report(
    run_dirs: dict[str, Path], alpha: float = ALPHA, n_boot: int | None = None, seed: int = SEED
) -> dict:
    """Tables of every method (name -> run directory) and the H9 verdict, computed on `p0` only.

    The methods must cover the same categories in the same order: their tables stand side by side.
    `categories` is that list and `complete` says whether it is the full set `CATEGORIES`, which the
    conclusions are drawn over. A run on part of the categories (a code check) gets its tables but no
    H9 verdict (`h9` is None).
    """
    loaded = {}
    for method, run_dir in run_dirs.items():
        run = load(run_dir)
        if run["meta"].get("method") != method:
            raise ValueError(f"{run_dir} is a run of {run['meta'].get('method')!r}, not of {method!r}")
        loaded[method] = run
    covered = {method: list(run["cats"]) for method, run in loaded.items()}
    categories = next(iter(covered.values()), [])
    if any(cats != categories for cats in covered.values()):
        raise ValueError(f"the runs cover different categories and cannot be put side by side: {covered}")
    complete = categories == list(CATEGORIES)

    tables, runs = {}, {}
    for method, run in loaded.items():
        tables[method] = condition_table(run, alpha, n_boot, seed)
        runs[method] = {"protocol": run["meta"].get("protocol"), **describe(run, alpha)}
    verdict = None
    if H9_METHOD in tables and complete:
        verdict = {"method": H9_METHOD, **h9(tables[H9_METHOD], alpha)}
    return {
        "alpha": alpha,
        "n_boot": n_boot or N_BOOT,
        "seed": seed,
        "categories": categories,
        "complete": complete,
        "runs": runs,
        "tables": tables,
        "h9": verdict,
    }


def format_coverage(report: dict) -> str:
    """What the tables of a report stand on: the categories, and the images behind each method."""
    categories = report["categories"]
    lines = [f"categories ({len(categories)} of {len(CATEGORIES)}): {', '.join(categories)}"]
    for method, info in report["runs"].items():
        lines.append(
            f"{method}: {info['n_normal']} normal and {info['n_defect']} defect images, "
            f"{info['n_cal']} calibration scores"
        )
    if not report["complete"]:
        lines.append(f"partial run: H9 is not judged (it is registered on all {len(CATEGORIES)} categories)")
    return "\n".join(lines)


def _value(x: float | None, ci: list[float] | None) -> str:
    if x is None or ci is None:
        return "-"
    return f"{x * 100:.2f} [{ci[0] * 100:.2f}, {ci[1] * 100:.2f}]"


def _change(row: dict, key: str) -> str:
    x, ci = row.get(f"d_{key}"), row.get(f"d_{key}_ci")
    if x is None or ci is None:
        return "-"
    return f"{x * 100:+.2f} [{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}]"


def format_tables(tables: dict[str, list[dict]], verdict: dict | None) -> str:
    """Markdown: one table per method (values in %, changes against clean in points), then H9."""
    lines = []
    for method, table in tables.items():
        lines += [
            f"method {method}",
            "| condition | image AUROC | change | pooled FPR | change | pooled detection | change |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in table:
            cells = [
                row["condition"],
                _value(row["macro_image_auroc"], row["macro_image_auroc_ci"]),
                _change(row, "auroc"),
                _value(row["fpr"], row["fpr_ci"]),
                _change(row, "fpr"),
                _value(row["tpr"], row["tpr_ci"]),
                _change(row, "tpr"),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    if verdict is not None:
        name = f" ({verdict['method']})" if "method" in verdict else ""
        found = ", ".join(verdict["conditions"]) or "none"
        lines.append(f"H9{name}: {verdict['verdict']} (conditions: {found})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", choices=["dev", "test"], required=True)
    parser.add_argument(
        "--methods", nargs="*", choices=METHODS, default=list(METHODS), help="runs to tabulate"
    )
    parser.add_argument("--no-write", action="store_true", help="print only")
    args = parser.parse_args(argv)
    if not args.methods:
        parser.error("--methods needs at least one method")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Korean verdict words on a cp949 console

    run_dirs = {m: paths.OUTPUTS / f"perturb-{m}-{args.protocol}" for m in args.methods}
    missing = [str(d) for d in run_dirs.values() if not (d / "run.json").exists()]
    if missing:
        parser.error(f"no finished run in {missing}: run defect_inspect.run_perturb first, or pass --methods")
    report = {"protocol": args.protocol, **build_report(run_dirs)}
    wrong = [m for m, info in report["runs"].items() if info["protocol"] != args.protocol]
    if wrong:
        parser.error(f"the runs of {wrong} were not made with the {args.protocol} protocol")
    if args.protocol == "test" and not report["complete"]:
        # The sealed-test tables are the result, and results are stated over every category.
        parser.error(
            f"the runs cover {report['categories']}, but the test protocol is reported over all "
            f"categories in the registered order {list(CATEGORIES)}: redo run_perturb without --categories"
        )
    print(format_coverage(report))
    print()
    print(format_tables(report["tables"], report["h9"]))
    if args.no_write:
        return
    out_dir = paths.REPORTS / "stage3"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"perturb-{args.protocol}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
