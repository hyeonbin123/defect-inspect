"""Stage 7 tables and verdicts: the validation pick, H18 and H19 on the sealed M2AD test, the closed-loop
arm, and the VisA dev table of the feature centring (E1). Rules: docs/experiments.md, stage 7.

Rates are pooled over the inspectors. Intervals come from bootstraps whose unit is the specimen (M2AD)
or the image within category and class (VisA dev), with the same draws for everything compared, so
differences are paired.
"""

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import paths
from .analyze_m2ad import ClusterBootstrap, strict_json
from .analyze_perturb import _auroc_rows
from .analyze_perturb import load as load_perturb
from .calibrate import conformal_threshold
from .m2ad import ILLUMINATIONS, REFERENCE
from .run_m2ad_enrol import CANDIDATES, LOOP_ARMS
from .stats import percentile_ci, stratified_indices

N_BOOT = 2000
SEED = 0
ALPHA = 0.05
JUDGED_METHOD = "p0"
H18_MARGIN = -0.50  # supported when the interval of (pick - e0) unseen FPR lies below -50 points
GUARD = -0.10  # "improvement" only if the I01 detection rate falls by no more than 10 points
H19_LIMIT = 0.06
VISA_CONDITIONS = ("clean", "brightness-3", "gamma-3")
_TOL = (
    1e-9  # pooled rates are ratios of integers: keeps an end equal to a bound where exact arithmetic has it
)


def _ci(samples: np.ndarray) -> list[float]:
    lo, hi = percentile_ci(samples)
    return [float(lo), float(hi)]


def _ratio(num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(strict_json(report), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def _load_npzs(run_dir: Path, meta: dict, suffixes: list[str]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for suffix in suffixes:
        out[suffix] = {}
        for name in meta["inspectors"]:
            with np.load(Path(run_dir) / f"{name}_{suffix}.npz") as z:
                out[suffix][name] = {k: z[k] for k in z.files}
    return out


def _read_meta(run_dir: Path) -> dict:
    with open(Path(run_dir) / "run.json", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- validation (train specimens only)


def load_val(run_dir: Path) -> dict:
    """A `run_m2ad_enrol val` run: {"meta", "cands": {candidate: {inspector: arrays}}}."""
    meta = _read_meta(run_dir)
    if meta.get("command") != "val":
        raise ValueError(f"{run_dir} is not a validation run")
    return {"meta": meta, "cands": _load_npzs(run_dir, meta, meta["candidates"])}


def _val_weights(run: dict, n_boot: int, seed: int) -> dict[str, np.ndarray]:
    """Specimen draws per category ([n_boot, n_specimens] counts), shared by every candidate."""
    first = next(iter(run["cands"].values()))
    names: dict[str, list[str]] = {}
    for inspector, arr in first.items():
        category = inspector.split("_")[0]
        mine = arr["specimens"].tolist()
        if names.setdefault(category, mine) != mine:
            raise ValueError(f"{inspector}: train specimens differ from the other views of {category}")
    for cand in run["cands"].values():
        for inspector, arr in cand.items():
            if arr["specimens"].tolist() != names[inspector.split("_")[0]]:
                raise ValueError(f"{inspector}: specimens differ between candidates")
    categories = list(names)
    draws = stratified_indices([len(names[c]) for c in categories], n_boot=n_boot, seed=seed)
    weights = {}
    rows = np.arange(n_boot)[:, None]
    for category, draw in zip(categories, draws, strict=True):
        w = np.zeros((n_boot, len(names[category])))
        np.add.at(w, (rows, draw), 1.0)
        weights[category] = w
    return weights


def val_candidate(cand: dict[str, dict], weights: dict[str, np.ndarray]) -> dict:
    """Unseen and seen illumination false alarm counts of one candidate, pooled, with bootstrap samples."""
    n_boot = next(iter(weights.values())).shape[0]
    point = {"unseen": np.zeros(2), "seen": np.zeros(2)}
    samples = {"unseen": np.zeros((n_boot, 2)), "seen": np.zeros((n_boot, 2))}
    per_light = {light: np.zeros(2) for light in ILLUMINATIONS}
    for inspector, arr in cand.items():
        if arr["lights"].tolist() != list(ILLUMINATIONS):
            raise ValueError(f"{inspector}: unexpected illumination order")
        w = weights[inspector.split("_")[0]]
        thresholds = arr["thresholds"][:, arr["folds"]]  # [arms, specimens]
        for a in range(len(arr["arms"])):
            for li, light in enumerate(ILLUMINATIONS):
                flagged = (arr["scores"][a, li].astype(np.float64) > thresholds[a]).astype(np.float64)
                cols = np.stack([flagged, np.ones_like(flagged)], axis=1)
                kind = "seen" if arr["enrolled"][a, li] else "unseen"
                point[kind] += cols.sum(axis=0)
                samples[kind] += w @ cols
                if kind == "unseen":
                    per_light[light] += cols.sum(axis=0)
    out = {}
    for kind in ("unseen", "seen"):
        out[kind] = {
            "false_positives": int(point[kind][0]),
            "n_normal": int(point[kind][1]),
            "fpr": float(_ratio(point[kind][0], point[kind][1])),
            "fpr_ci": _ci(_ratio(samples[kind][:, 0], samples[kind][:, 1])),
        }
    out["unseen_by_light"] = {
        light: {"false_positives": int(c[0]), "n_normal": int(c[1])} for light, c in per_light.items() if c[1]
    }
    return out


def choose(rows: dict[str, dict]) -> str:
    """Lowest pooled unseen FPR (exact ratio); ties go to the simpler candidate (CANDIDATES order)."""
    ranked = sorted(
        rows,
        key=lambda c: (
            Fraction(rows[c]["unseen"]["false_positives"], rows[c]["unseen"]["n_normal"]),
            CANDIDATES.index(c),
        ),
    )
    return ranked[0]


def build_val_report(runs: dict[str, dict], n_boot: int | None = None, seed: int = SEED) -> dict:
    n_boot = int(n_boot or N_BOOT)
    report: dict = {"n_boot": n_boot, "seed": seed, "methods": {}}
    for method, run in runs.items():
        weights = _val_weights(run, n_boot, seed)
        rows = {c: val_candidate(run["cands"][c], weights) for c in run["cands"]}
        n = {rows[c]["unseen"]["n_normal"] for c in rows}
        if len(n) != 1:
            raise ValueError(f"{method}: candidates were scored on different unseen images: {n}")
        report["methods"][method] = {
            "commit": run["meta"].get("commit"),
            "device": run["meta"].get("device"),
            "groups": run["meta"].get("groups"),
            "inspectors": run["meta"]["inspectors"],
            "candidates": rows,
            "pick": choose(rows),
        }
    return report


def format_val(report: dict) -> str:
    lines = []
    for method, entry in report["methods"].items():
        lines += [
            f"## {method} validation (commit {entry['commit']}): pick {entry['pick']}",
            "| candidate | unseen FPR % | seen FPR % |",
            "|---|---|---|",
        ]
        for cand, row in entry["candidates"].items():
            u, s = row["unseen"], row["seen"]
            lines.append(
                f"| {cand} | {u['fpr'] * 100:.1f} [{u['fpr_ci'][0] * 100:.1f}, {u['fpr_ci'][1] * 100:.1f}] "
                f"({u['false_positives']}/{u['n_normal']}) | {s['fpr'] * 100:.1f} "
                f"({s['false_positives']}/{s['n_normal']}) |"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- sealed test


def load_test(run_dir: Path) -> dict:
    """A `run_m2ad_enrol test` run: {"meta", "cands": {candidate: {inspector: arrays}}, "loop": ...}."""
    meta = _read_meta(run_dir)
    if meta.get("command") != "test":
        raise ValueError(f"{run_dir} is not a test run")
    cands = _load_npzs(run_dir, meta, meta["candidates"])
    loop = _load_npzs(run_dir, meta, ["loop"])["loop"] if meta.get("loop") else None
    return {"meta": meta, "cands": cands, "loop": loop}


def _bootstrap(run: dict, n_boot: int | None, seed: int) -> ClusterBootstrap:
    e0 = run["cands"]["e0"]
    for cand in run["cands"].values():
        for name, arr in cand.items():
            if arr["specimens"].tolist() != e0[name]["specimens"].tolist():
                raise ValueError(f"{name}: test specimens differ between candidates")
    shell = {
        "inspectors": {
            name: {"category": name.split("_")[0], "object_anomaly": arr["object_anomaly"]}
            for name, arr in e0.items()
        }
    }
    return ClusterBootstrap(shell, n_boot, seed)


def _arm_cell(cand: dict[str, dict], names: list[str], light: str, want_enrolled: bool) -> list:
    """Cells (one per arm that has `light` enrolled, or not) in inspector order."""
    li = ILLUMINATIONS.index(light)
    first = cand[names[0]]
    cells = []
    for a in range(len(first["arms"])):
        if bool(first["enrolled"][a, li]) != want_enrolled:
            continue
        cells.append(
            [
                (cand[n]["scores"][a, li], cand[n]["labels"][li], float(cand[n]["thresholds"][a]))
                for n in names
            ]
        )
    return cells


def _rates(boot: ClusterBootstrap, cells: list) -> dict:
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


def _public(rates: dict) -> dict:
    out = {k: v for k, v in rates.items() if not k.endswith("_samples")}
    for key in ("fpr", "tpr"):
        out[f"{key}_ci"] = _ci(rates[f"{key}_samples"])
    return out


def candidate_rates(run: dict, cand_name: str, boot: ClusterBootstrap) -> dict:
    """Unseen, seen and I01 rates of one candidate, a per-illumination table and the mean unseen AUROC."""
    cand = run["cands"][cand_name]
    names = boot.names
    real = [light for light in ILLUMINATIONS if light != REFERENCE]
    unseen_cells = [cell for light in real for cell in _arm_cell(cand, names, light, False)]
    seen_cells = [cell for light in ILLUMINATIONS for cell in _arm_cell(cand, names, light, True)]
    ref_cells = _arm_cell(cand, names, REFERENCE, True)
    per_light = {}
    aurocs, auroc_samples = [], []
    for light in ILLUMINATIONS:
        for enrolled in (False, True):
            for k, cell in enumerate(_arm_cell(cand, names, light, enrolled)):
                key = f"{light}:{'seen' if enrolled else 'unseen'}:{k}"
                rates = _public(_rates(boot, [cell]))
                rates["auroc"], samples = boot.auroc(cell)
                per_light[key] = rates
                if not enrolled:
                    aurocs.append(rates["auroc"])
                    auroc_samples.append(samples)
    return {
        "unseen": _rates(boot, unseen_cells),
        "seen": _rates(boot, seen_cells),
        "reference": _rates(boot, ref_cells),
        "unseen_auroc_mean": float(np.mean(aurocs)) if aurocs else float("nan"),
        "unseen_auroc_mean_ci": _ci(np.mean(np.stack(auroc_samples), axis=0)) if aurocs else None,
        "per_light": per_light,
    }


def h18_h19(e0: dict, pick: dict, pick_name: str) -> dict:
    """H18 (unseen FPR vs e0, with the I01 detection guard) and H19 (seen FPR of the pick)."""
    d_fpr = pick["unseen"]["fpr"] - e0["unseen"]["fpr"]
    lo, hi = _ci(pick["unseen"]["fpr_samples"] - e0["unseen"]["fpr_samples"])
    if pick_name == "e0":
        h18 = "기각"
    else:
        h18 = "지지" if hi < H18_MARGIN - _TOL else "기각" if lo > H18_MARGIN + _TOL else "판정 불가"
    d_ref = pick["reference"]["tpr"] - e0["reference"]["tpr"]
    guard_ok = bool(d_ref >= GUARD - _TOL)
    seen = pick["seen"]["fpr"]
    return {
        "pick": pick_name,
        "d_unseen_fpr": d_fpr,
        "d_unseen_fpr_ci": [lo, hi],
        "margin": H18_MARGIN,
        "H18": h18,
        "d_reference_tpr": d_ref,
        "d_reference_tpr_ci": _ci(pick["reference"]["tpr_samples"] - e0["reference"]["tpr_samples"]),
        "guard": GUARD,
        "guard_ok": guard_ok,
        "improved": bool(h18 == "지지" and guard_ok),
        "d_unseen_tpr": pick["unseen"]["tpr"] - e0["unseen"]["tpr"],
        "d_unseen_tpr_ci": _ci(pick["unseen"]["tpr_samples"] - e0["unseen"]["tpr_samples"]),
        "seen_fpr": seen,
        "seen_fpr_ci": _ci(pick["seen"]["fpr_samples"]),
        "h19_limit": H19_LIMIT,
        "H19": "지지" if seen <= H19_LIMIT + _TOL else "기각",
    }


def loop_table(run: dict, boot: ClusterBootstrap) -> list[dict]:
    """Closed loop on e0: pooled over the nine new illuminations, one row per (filter, k)."""
    loop = run["loop"]
    names = boot.names
    info = {item["category"] + "_" + item["view"]: item["loop"] for item in run["meta"]["loop_runs"]}
    table = []
    for filt, k in LOOP_ARMS:
        cells, mixed, dropped, dropped_normals = [], 0, 0, 0
        for light in ILLUMINATIONS:
            if light == REFERENCE:
                continue
            key = f"{light}:{filt}:{k}"
            j = loop[names[0]]["loop_keys"].tolist().index(key)
            cells.append(
                [(loop[n]["scores"][j], loop[n]["labels"][j], float(loop[n]["thresholds"][j])) for n in names]
            )
            for n in names:
                mixed += info[n][key]["defects_in_batch"]
                dropped += info[n][key]["dropped_defects"]
                dropped_normals += info[n][key]["dropped_normals"]
        row = {"filter": filt, "k": k, "defects_in_batches": mixed, "dropped_defects": dropped}
        row["dropped_normals"] = dropped_normals
        row |= _public(_rates(boot, cells))
        table.append(row)
    return table


def stage3b_match(run: dict, ref_dir: Path) -> dict | None:
    """e0 against the stage 3-B run of the same method: largest score difference and equal thresholds."""
    ref_dir = Path(ref_dir)
    if not (ref_dir / "run.json").exists():
        return None
    worst, same_thr = 0.0, True
    for name, arr in run["cands"]["e0"].items():
        with np.load(ref_dir / f"{name}.npz") as z:
            if "specimens" in z.files and z["specimens"].tolist() != arr["specimens"].tolist():
                raise ValueError(f"{ref_dir / name}: other test specimens than this run")
            conditions = z["conditions"].tolist()
            for li, light in enumerate(ILLUMINATIONS):
                k = conditions.index("S" if light == REFERENCE else f"R:{light}")
                diff = np.abs(arr["scores"][0, li].astype(np.float64) - z["scores"][k].astype(np.float64))
                worst = max(worst, float(diff.max()))
            same_thr &= float(z["threshold"]) == float(arr["thresholds"][0])
    return {"max_abs_score_diff": worst, "thresholds_equal": bool(same_thr)}


def build_test_report(runs: dict[str, dict], n_boot: int | None = None, seed: int = SEED) -> dict:
    report: dict = {
        "n_boot": int(n_boot or N_BOOT),
        "seed": seed,
        "judged_method": JUDGED_METHOD,
        "methods": {},
    }
    for method, run in runs.items():
        boot = _bootstrap(run, n_boot, seed)
        cands = run["meta"]["candidates"]
        pick_name = cands[-1]
        rates = {c: candidate_rates(run, c, boot) for c in cands}
        entry = {
            "commit": run["meta"].get("commit"),
            "device": run["meta"].get("device"),
            "candidates": {
                c: {
                    "unseen": _public(r["unseen"]),
                    "seen": _public(r["seen"]),
                    "reference": _public(r["reference"]),
                    "unseen_auroc_mean": r["unseen_auroc_mean"],
                    "unseen_auroc_mean_ci": r["unseen_auroc_mean_ci"],
                    "per_light": r["per_light"],
                }
                for c, r in rates.items()
            },
            "verdicts": h18_h19(rates["e0"], rates[pick_name], pick_name),
            "stage3b_match": stage3b_match(run, paths.OUTPUTS / f"m2ad-{method}"),
        }
        if run["loop"] is not None:
            entry["loop"] = loop_table(run, boot)
        if method != JUDGED_METHOD:
            for key in ("H18", "H19"):
                entry["verdicts"].pop(key)
            entry["verdicts"].pop("improved")
        report["methods"][method] = entry
    judged = report["methods"].get(JUDGED_METHOD, {}).get("verdicts", {})
    report["hypotheses"] = {k: judged[k] for k in ("H18", "H19") if k in judged}
    return report


def format_test(report: dict) -> str:
    lines = []
    for method, entry in report["methods"].items():
        v = entry["verdicts"]
        lines.append(f"## {method} test (commit {entry['commit']}), pick {v['pick']}")
        for cand, row in entry["candidates"].items():
            u, s, r = row["unseen"], row["seen"], row["reference"]
            lines.append(
                f"{cand}: unseen FPR {u['fpr'] * 100:.1f}% ({u['false_positives']}/{u['n_normal']}), "
                f"detection {u['tpr'] * 100:.1f}%, AUROC {row['unseen_auroc_mean'] * 100:.1f}; "
                f"seen FPR {s['fpr'] * 100:.1f}%; "
                f"I01 FPR {r['fpr'] * 100:.1f}% detection {r['tpr'] * 100:.1f}%"
            )
        lo, hi = v["d_unseen_fpr_ci"]
        lines.append(
            f"unseen FPR pick - e0: {v['d_unseen_fpr'] * 100:+.1f}%p [{lo * 100:.1f}, {hi * 100:.1f}]"
            + (f" -> H18 {v['H18']}" if "H18" in v else "")
        )
        lines.append(
            f"I01 detection pick - e0: {v['d_reference_tpr'] * 100:+.1f}%p (guard {v['guard'] * 100:.0f}%p: "
            f"{'ok' if v['guard_ok'] else 'fails'})"
            + (f", improved: {v['improved']}" if "improved" in v else "")
        )
        lines.append(f"seen FPR {v['seen_fpr'] * 100:.1f}%" + (f" -> H19 {v['H19']}" if "H19" in v else ""))
        for row in entry.get("loop", []):
            lines.append(
                f"loop {row['filter']} k={row['k']}: FPR {row['fpr'] * 100:.1f}% "
                f"detection {row['tpr'] * 100:.1f}%"
                f" (defects dropped {row['dropped_defects']}/{row['defects_in_batches']})"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- VisA dev: centring (E1)


def _visa_arrays(run: dict, conditions: tuple[str, ...]) -> dict:
    """Per category: labels, the scores of `conditions` [n_cond, n] and the conformal threshold."""
    out = {}
    for category, cat in run["cats"].items():
        names = [str(c) for c in cat["conditions"]]
        missing = [c for c in conditions if c not in names]
        if missing:
            raise ValueError(f"{category}: the run has no {missing}")
        out[category] = {
            "images": [str(x) for x in cat["eval_images"]],
            "labels": cat["eval_labels"],
            "scores": np.stack([cat["scores"][names.index(c)] for c in conditions]).astype(np.float64),
            "threshold": conformal_threshold(cat["cal_score"], ALPHA).value,
        }
    return out


def visa_dev_table(
    e0_run: dict,
    e1_run: dict,
    conditions: tuple[str, ...] = VISA_CONDITIONS,
    n_boot: int | None = None,
    seed: int = SEED,
) -> list[dict]:
    """Per condition: mean image AUROC, pooled FPR and detection rate of E0 and E1 at their own thresholds,
    and the paired differences E1 - E0 (stratified bootstrap, normals and defects per category)."""
    n_boot = int(n_boot or N_BOOT)
    a, b = _visa_arrays(e0_run, conditions), _visa_arrays(e1_run, conditions)
    if list(a) != list(b):
        raise ValueError("the two runs cover other categories")
    sizes = []
    for category in a:
        if a[category]["images"] != b[category]["images"] or not np.array_equal(
            a[category]["labels"], b[category]["labels"]
        ):
            raise ValueError(f"{category}: the two runs scored other images")
        labels = a[category]["labels"]
        sizes += [int((labels == 0).sum()), int((labels == 1).sum())]
    draws = stratified_indices(sizes, n_boot=n_boot, seed=seed)

    def stats(arrays: dict, k: int) -> dict:
        auroc_samples, auroc_point = [], []
        fp = np.zeros(n_boot)
        tp = np.zeros(n_boot)
        fp_point = tp_point = n_neg = n_pos = 0
        for c, cat in enumerate(arrays.values()):
            s = cat["scores"][k]
            neg, pos = s[cat["labels"] == 0], s[cat["labels"] == 1]
            ineg, ipos = draws[2 * c], draws[2 * c + 1]
            auroc_samples.append(_auroc_rows(neg[ineg], pos[ipos]))
            auroc_point.append(_auroc_rows(neg[None, :], pos[None, :])[0])
            thr = cat["threshold"]
            fp += (neg[ineg] > thr).sum(axis=1)
            tp += (pos[ipos] > thr).sum(axis=1)
            fp_point += int((neg > thr).sum())
            tp_point += int((pos > thr).sum())
            n_neg += neg.size
            n_pos += pos.size
        return {
            "auroc": float(np.mean(auroc_point)),
            "auroc_samples": np.mean(np.stack(auroc_samples), axis=0),
            "fpr": fp_point / n_neg,
            "fpr_samples": fp / n_neg,
            "tpr": tp_point / n_pos,
            "tpr_samples": tp / n_pos,
        }

    table = []
    base = {}
    for k, condition in enumerate(conditions):
        s0, s1 = stats(a, k), stats(b, k)
        if k == 0:
            base = {"e0": s0, "e1": s1}
        row = {"condition": condition}
        for name, s in (("e0", s0), ("e1", s1)):
            for key in ("auroc", "fpr", "tpr"):
                row[f"{name}_{key}"] = s[key]
                row[f"{name}_{key}_ci"] = _ci(s[f"{key}_samples"])
            if k > 0:
                row[f"{name}_d_auroc_vs_clean"] = s["auroc"] - base[name]["auroc"]
                row[f"{name}_d_auroc_vs_clean_ci"] = _ci(s["auroc_samples"] - base[name]["auroc_samples"])
        for key in ("auroc", "fpr", "tpr"):
            row[f"d_{key}"] = s1[key] - s0[key]
            row[f"d_{key}_ci"] = _ci(s1[f"{key}_samples"] - s0[f"{key}_samples"])
        table.append(row)
    return table


def rerun_check(source_dir: Path, reference_dir: Path) -> dict | None:
    """Largest relative difference of the full-bank evaluation scores between a rerun and an older run."""
    source_dir, reference_dir = Path(source_dir), Path(reference_dir)
    if not (reference_dir / "run.json").exists() or source_dir.resolve() == reference_dir.resolve():
        return None
    worst = 0.0
    for path in sorted(source_dir.glob("*.npz")):
        if path.name.endswith("_maps.npy") or not (reference_dir / path.name).exists():
            continue
        with np.load(path) as new, np.load(reference_dir / path.name) as old:
            a, b = new["eval_score_full"].astype(np.float64), old["eval_score_full"].astype(np.float64)
            worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))))
    return {"reference": str(reference_dir), "max_rel_diff": worst}


def build_visa_report(
    pairs: dict[str, tuple[Path, Path]], n_boot: int | None = None, seed: int = SEED
) -> dict:
    report: dict = {"n_boot": int(n_boot or N_BOOT), "seed": seed, "alpha": ALPHA, "methods": {}}
    for method, (e0_dir, e1_dir) in pairs.items():
        e0_run, e1_run = load_perturb(e0_dir), load_perturb(e1_dir)
        if e0_run["meta"].get("protocol") != "dev" or e1_run["meta"].get("protocol") != "dev":
            raise ValueError("the centring table is read on the dev protocol only")
        report["methods"][method] = {
            "e0": str(e0_dir),
            "e1": str(e1_dir),
            "e0_commit": e0_run["meta"].get("commit"),
            "e1_commit": e1_run["meta"].get("commit"),
            "table": visa_dev_table(e0_run, e1_run, n_boot=n_boot, seed=seed),
        }
        source = e0_run["meta"].get("source")
        if source:
            report["methods"][method]["e0_rerun_check"] = rerun_check(
                Path(source), paths.OUTPUTS / f"{method}-dev"
            )
    return report


def format_visa(report: dict) -> str:
    lines = []
    for method, entry in report["methods"].items():
        lines += [
            f"## {method}: E0 {entry['e0_commit']}, E1 {entry['e1_commit']}",
            "| condition | E0 AUROC | E1 AUROC | dAUROC %p | E0 FPR | E1 FPR | dFPR %p |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in entry["table"]:
            lo, hi = row["d_auroc_ci"]
            flo, fhi = row["d_fpr_ci"]
            lines.append(
                f"| {row['condition']} | {row['e0_auroc'] * 100:.2f} | {row['e1_auroc'] * 100:.2f} | "
                f"{row['d_auroc'] * 100:+.2f} [{lo * 100:.2f}, {hi * 100:.2f}] | {row['e0_fpr'] * 100:.1f} | "
                f"{row['e1_fpr'] * 100:.1f} | {row['d_fpr'] * 100:+.1f} [{flo * 100:.1f}, {fhi * 100:.1f}] |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("val", "test"):
        p = sub.add_parser(name)
        p.add_argument("--methods", nargs="+", default=["p0", "d-s"])
        p.add_argument("--no-write", action="store_true")
    visa = sub.add_parser("visa-dev")
    visa.add_argument("--no-write", action="store_true")
    visa.add_argument(
        "--pairs",
        nargs="+",
        default=[
            "p0=perturb-p0-dev7:perturb-p0-c-dev",
            "d-s=perturb-d-s-dev:perturb-d-s-c-dev",
        ],
        help="method=<E0 perturb run>:<E1 perturb run>, directories under outputs/",
    )
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    out_dir = paths.REPORTS / "stage7"
    if args.command == "val":
        runs = {m: load_val(paths.OUTPUTS / f"m2ad-enrol-val-{m}") for m in args.methods}
        report = build_val_report(runs)
        print(format_val(report))
        target = out_dir / "val.json"
    elif args.command == "test":
        runs = {m: load_test(paths.OUTPUTS / f"m2ad-enrol-test-{m}") for m in args.methods}
        report = build_test_report(runs)
        print(format_test(report))
        target = out_dir / "test.json"
    else:
        pairs = {}
        for item in args.pairs:
            method, _, dirs = item.partition("=")
            e0, _, e1 = dirs.partition(":")
            pairs[method] = (paths.OUTPUTS / e0, paths.OUTPUTS / e1)
        report = build_visa_report(pairs)
        print(format_visa(report))
        target = out_dir / "visa-dev.json"
    if not args.no_write:
        _write(target, report)


if __name__ == "__main__":
    main()
