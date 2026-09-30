"""Table over PatchCore settings and coreset ratios: accuracy, fixed-threshold rates, bank size, latency."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import compare, paths
from .run_grid import ratio_name

# The stage 4 grid and the CPU budget of the serving choice, as registered in docs/experiments.md.
SETTINGS = ("wrn50-256", "wrn50-384", "dinov2_vits14-252", "dinov2_vits14-392", "dinov2_vits14-448")
CPU_BUDGET_MS = 200.0


class _SharedBootstrap(compare.Bootstrap):
    """`compare.Bootstrap` that keeps the macro AUROC samples of every run it has resampled.

    The interval of a row (`compare.summarize`) and the paired differences need the same samples; the
    resampling is the slow part of the table, so each run goes through it once.
    """

    def __init__(self, reference: compare.MethodRun):
        super().__init__(reference)
        self._auroc: dict[int, tuple[compare.MethodRun, np.ndarray]] = {}

    def macro_auroc(
        self, run: compare.MethodRun, keep_pos: dict[str, np.ndarray] | None = None
    ) -> np.ndarray:
        if keep_pos is not None:
            return super().macro_auroc(run, keep_pos)
        if id(run) not in self._auroc:  # the run is stored with its samples, so the id cannot be reused
            self._auroc[id(run)] = (run, super().macro_auroc(run))
        return self._auroc[id(run)][1]


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
    boot = _SharedBootstrap(flat[0])
    rows = []
    for setting, ratios in runs.items():
        largest = max(ratios)
        base_point = compare.macro_auroc(ratios[largest])
        base_boot = boot.macro_auroc(ratios[largest])
        for ratio in sorted(ratios, reverse=True):
            run = ratios[ratio]
            summary = compare.summarize(run, boot)  # its fixed-threshold rates are at compare.ALPHA
            fixed = summary["fixed_threshold"] if alpha == compare.ALPHA else boot.pooled_rates(run, alpha)
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
    """Attach the latency entry with the row's key (`<backbone>-<size>-r<ratio>`), when there is one.

    `cpu_ms` is the registered CPU latency of a row: the time to resize a camera-sized image to the input
    size plus the FP32 onnxruntime inference (medians, as `bench` writes them). An entry without CPU
    timings (GPU only) gets no `cpu_ms`; CPU timings without the resize time are an error, because the
    inference time alone must not be compared with the latency budget.
    """
    for row in rows:
        entry = latency.get(row["key"])
        if entry is None:
            continue
        row["latency"] = entry
        if "cpu_fp32" not in entry:
            continue
        if "resize_ms" not in entry:
            raise ValueError(f"latency entry {row['key']} has cpu_fp32 but no resize_ms: run bench again")
        row["cpu_ms"] = float(entry["resize_ms"]) + float(entry["cpu_fp32"]["total_ms"])


def choose_serving(rows: list[dict], budget_ms: float = CPU_BUDGET_MS) -> dict | None:
    """The serving configuration by the registered rule (docs/experiments.md, stage 4), from `cpu_ms`.

    Among the rows whose CPU latency (resize + inference) is within the budget: the highest macro image
    AUROC, ties to the smaller bank. When no row is within the budget: the fastest row, and
    `within_budget` is False (H15 is then rejected). Rows without a CPU latency cannot be chosen and are
    listed under `unmeasured`. None when no row has a CPU latency.
    """
    measured = [r for r in rows if "cpu_ms" in r]
    if not measured:
        return None
    within = [r for r in measured if r["cpu_ms"] <= budget_ms]
    if within:
        best = min(within, key=lambda r: (-r["macro_image_auroc"], r["bank_rows_mean"]))
    else:
        best = min(measured, key=lambda r: r["cpu_ms"])
    return {
        "key": best["key"],
        "setting": best["setting"],
        "ratio": best["ratio"],
        "cpu_ms": best["cpu_ms"],
        "macro_image_auroc": best["macro_image_auroc"],
        "budget_ms": budget_ms,
        "within_budget": bool(within),
        "candidates": [r["key"] for r in within],
        "unmeasured": [r["key"] for r in rows if "cpu_ms" not in r],
    }


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
        "CPU ms (resize + inference)",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        d = r.get("vs_largest_ratio")
        cpu = r.get("cpu_ms")
        cells = [
            r["setting"],
            f"{r['ratio']:g}",
            f"{r['bank_rows_mean']:.0f}",
            f"{_pct(r['macro_image_auroc'])} {_span(r['macro_image_auroc_ci'])}",
            f"{d['diff'] * 100:+.2f}%p {d['verdict']}" if d else "-",
            _pct(r["macro_aupro"]),
            f"{_pct(r['fpr'])} {_span(r['fpr_ci'])}",
            f"{_pct(r['tpr'])} {_span(r['tpr_ci'])}",
            f"{cpu:.1f}" if cpu is not None else "-",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_serving(serving: dict) -> str:
    if serving["within_budget"]:
        reason = f"highest image AUROC among {len(serving['candidates'])} rows within the budget"
    else:
        reason = "no row is within the budget, so the fastest one"
    text = (
        f"serving configuration: {serving['key']} ({serving['cpu_ms']:.1f} ms on the CPU, budget "
        f"{serving['budget_ms']:g} ms; {reason})"
    )
    if serving["unmeasured"]:
        text += f"\nno CPU latency, not considered: {', '.join(serving['unmeasured'])}"
    return text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", choices=["dev", "test"], required=True)
    parser.add_argument(
        "--settings",
        nargs="+",
        default=None,
        help="e.g. wrn50-256 dinov2_vits14-392 (default: the registered settings with a finished grid run)",
    )
    parser.add_argument("--latency", type=Path, default=None, help="default: reports/stage4/latency.json")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    def run_dir(setting: str) -> Path:
        return paths.OUTPUTS / f"grid-{setting}-{args.protocol}"

    finished = [s for s in args.settings or SETTINGS if (run_dir(s) / "run.json").exists()]
    if args.settings is None:
        if not finished:
            parser.error(f"no finished grid run of the {args.protocol} protocol under {paths.OUTPUTS}")
        skipped = [s for s in SETTINGS if s not in finished]
        if skipped:
            print(f"no finished grid run, left out: {', '.join(skipped)}", file=sys.stderr)
    elif len(finished) != len(args.settings):
        missing = [str(run_dir(s)) for s in args.settings if s not in finished]
        parser.error(f"no finished grid run in: {', '.join(missing)}")
    if args.latency is not None and not args.latency.exists():
        parser.error(f"latency file not found: {args.latency}")

    runs = {s: load_grid(run_dir(s)) for s in finished}
    rows = grid_table(runs)
    latency_path = args.latency or paths.REPORTS / "stage4" / "latency.json"
    if latency_path.exists():
        with open(latency_path, encoding="utf-8") as f:
            merge_latency(rows, json.load(f))
    report = {"protocol": args.protocol, "alpha": compare.ALPHA, "rows": rows}
    print(format_table(rows))
    # The serving configuration is chosen on the dev protocol only; test numbers never choose anything.
    serving = choose_serving(rows) if args.protocol == "dev" else None
    if serving is not None:
        report["serving"] = serving
        print("\n" + format_serving(serving))
    if not args.no_write:
        out_dir = paths.REPORTS / "stage4"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"grid-{args.protocol}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
