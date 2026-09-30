"""Table over PatchCore settings and coreset ratios: accuracy, fixed-threshold rates, bank size, latency."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import compare, paths
from .run_grid import ratio_name


def load_grid(out_dir: Path) -> dict[float, compare.MethodRun]:
    """Every ratio folder of one `run_grid` call, largest ratio first."""
    with open(out_dir / "run.json", encoding="utf-8") as f:
        meta = json.load(f)
    return {
        float(r): compare.load_patchcore(out_dir / ratio_name(r), name=f"{meta['setting']}-{ratio_name(r)}")
        for r in sorted(meta["ratios"], reverse=True)
    }


def grid_table(runs: dict[str, dict[float, compare.MethodRun]], alpha: float = 0.05) -> list[dict]:
    """One row per (setting, ratio). All runs must score the same evaluation images (same bootstrap draws)."""
    flat = [run for ratios in runs.values() for run in ratios.values()]
    compare.check_same_images(flat)
    boot = compare.Bootstrap(flat[0])
    rows = []
    for setting, ratios in runs.items():
        largest = max(ratios)
        base_point = compare.macro_auroc(ratios[largest])
        base_boot = boot.macro_auroc(ratios[largest])
        for ratio in sorted(ratios, reverse=True):
            run = ratios[ratio]
            summary = compare.summarize(run, boot)
            fixed = boot.pooled_rates(run, alpha)
            row = {
                "key": run.name,
                "setting": setting,
                "ratio": ratio,
                "bank_rows_mean": float(np.mean([int(cat["bank_rows"]) for cat in run.cats.values()])),
                "macro_image_auroc": summary["macro_image_auroc"],
                "macro_image_auroc_ci": summary["macro_image_auroc_ci"],
                "macro_aupro": summary["macro_aupro"],
                "macro_aupro_ci": summary["macro_aupro_ci"],
                "pooled_fpr_at_tpr95": summary["pooled_fpr_at_tpr95"],
                "fpr": fixed["fpr"],
                "fpr_ci": fixed["fpr_ci"],
                "tpr": fixed["tpr"],
                "tpr_ci": fixed["tpr_ci"],
            }
            if ratio != largest:
                row["vs_largest_ratio"] = compare.difference(
                    compare.macro_auroc(run) - base_point, boot.macro_auroc(run) - base_boot
                )
            rows.append(row)
    return rows


def merge_latency(rows: list[dict], latency: dict) -> None:
    """Attach the latency entry with the row's key (`<backbone>-<size>-r<ratio>`), when there is one."""
    for row in rows:
        if row["key"] in latency:
            row["latency"] = latency[row["key"]]


def _pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}"


def _span(ci: list[float]) -> str:
    return f"[{_pct(ci[0])}, {_pct(ci[1])}]"


def format_table(rows: list[dict]) -> str:
    header = [
        "setting",
        "ratio",
        "bank rows",
        "image AUROC",
        "vs largest",
        "AUPRO",
        "actual FPR (5%)",
        "detection",
    ]
    header += ["CPU ms"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        d = r.get("vs_largest_ratio")
        cpu = r.get("latency", {}).get("cpu_fp32", {}).get("total_ms")
        cells = [
            r["setting"],
            f"{r['ratio']:g}",
            f"{r['bank_rows_mean']:.0f}",
            f"{_pct(r['macro_image_auroc'])} {_span(r['macro_image_auroc_ci'])}",
            f"{d['diff'] * 100:+.2f}%p {d['verdict']}" if d else "-",
            _pct(r["macro_aupro"]),
            f"{_pct(r['fpr'])} {_span(r['fpr_ci'])}",
            f"{_pct(r['tpr'])} {_span(r['tpr_ci'])}",
            f"{cpu:.0f}" if cpu is not None else "-",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", choices=["dev", "test"], required=True)
    parser.add_argument("--settings", nargs="+", required=True, help="e.g. wrn50-256 dinov2_vits14-392")
    parser.add_argument("--latency", type=Path, default=None, help="default: reports/stage4/latency.json")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    runs = {s: load_grid(paths.OUTPUTS / f"grid-{s}-{args.protocol}") for s in args.settings}
    rows = grid_table(runs)
    latency_path = args.latency or paths.REPORTS / "stage4" / "latency.json"
    if latency_path.exists():
        with open(latency_path, encoding="utf-8") as f:
            merge_latency(rows, json.load(f))
    print(format_table(rows))
    if not args.no_write:
        out_dir = paths.REPORTS / "stage4"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"grid-{args.protocol}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(
                {"protocol": args.protocol, "alpha": 0.05, "rows": rows}, f, ensure_ascii=False, indent=2
            )
            f.write("\n")


if __name__ == "__main__":
    main()
