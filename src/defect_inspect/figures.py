"""Figures of README.md, redrawn from the committed report files.

    python -m defect_inspect.figures [--reports reports] [--out docs/figures] [--only NAME ...]
    python -m defect_inspect.figures --examples RUN_DIR [--allow-test] [--out docs/figures] [--name examples]

Each `fig_<name>` takes report data that is already loaded and returns a matplotlib Figure. `main` loads
the report files, skips the figures whose inputs are missing, writes one PNG per figure and exits with an
error at the end when a report could not be read or drawn (the other figures are still written). With
`--examples`, `main` draws only the panel of example inspections of one evaluation run (`examples`) from
the scores and maps the run saved, and exits with code 2 when the run cannot give it. Nothing here
reads the clock or draws random numbers, and the PNG metadata carries no version string, so two runs on
the same reports write identical bytes. matplotlib (dependency group `figures`) is imported inside the
functions: importing this module needs the base packages only.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import math
import sys
import textwrap
import zipfile
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

import numpy as np

from . import ledger, paths
from .analyze_grid import CPU_BUDGET_MS
from .analyze_m2ad import BASE as M2AD_BASE
from .cache import MASK_SIZE, ImageCache
from .calibrate import conformal_threshold
from .compare import ALPHA, MethodRun, load_common, load_patchcore
from .conditions import CLEAN, LEVELS, REFERENCE_SIZE, strength
from .configs import CONFIGS
from .m2ad import REFERENCE as M2AD_REFERENCE
from .run_dinomaly import IMG_SIZE as DINOMALY_SIZE
from .run_dinomaly import NAME as DINOMALY
from .splits import SEALED_ROLES, ManifestRow, SealedTestError, read_manifest

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

DPI = 150
WIDTH = 12.0  # inches: 1800 px at DPI
SURFACE = "#ffffff"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, WASH = "#e1e0d9", "#c3c2b7", "#f0efec"
SMALL = 8.5  # font size of notes inside the panels

# Label, colour and marker of each method, the same in every figure. The four colours were checked as a
# set for colour-vision deficiencies; the marker shape repeats the identity without colour.
METHODS: dict[str, tuple[str, str, str]] = {
    "p0": ("PatchCore WRN-50", "#2a78d6", "o"),
    "d-s": ("PatchCore DINOv2 ViT-S", "#eb6834", "s"),
    "dm": ("Dinomaly", "#1baf7a", "^"),
}
SUPERVISED = ("Supervised head", "#4a3aa7", "D")
STRATEGIES = {"resubstitution": "Resubstitution", "holdout": "Hold-out", "crossfit": "Cross-fit"}
PROTOCOLS = {"test": "VisA test split", "dev": "VisA dev split"}
BACKBONES = {"wrn50": ("WRN-50", "#2a78d6"), "dinov2_vits14": ("DINOv2 ViT-S", "#eb6834")}
_SPARE_COLOURS = ("#1baf7a", "#4a3aa7")  # backbones that BACKBONES does not know, in order of appearance
_KINDS = {
    "brightness": "brightness",
    "gamma": "gamma",
    "blur": "blur σ",
    "shift": "shift (px)",
    "jpeg": "JPEG quality",
}
# The kinds whose strength `conditions.strength` scales with the input size (blur σ and shift): their
# ticks show the value on the 256 px grid, and a footnote says what the other input sizes got.
_SCALED = tuple(
    kind
    for kind in LEVELS
    if strength(f"{kind}-1", 2 * REFERENCE_SIZE) != strength(f"{kind}-1", REFERENCE_SIZE)
)
# Input size of each method in stage 3 (the images were perturbed at this size).
INPUT_SIZES = {**{name: cfg.img_size for name, cfg in CONFIGS.items()}, DINOMALY: DINOMALY_SIZE}
_GROUP_GAP = 0.6  # extra space between two groups of conditions, in condition slots

_RC = {
    "font.family": "DejaVu Sans",  # ships with matplotlib: the same glyphs on every machine
    "font.size": 10,
    "text.color": INK,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    "axes.labelsize": 9.5,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
}


@contextlib.contextmanager
def _style() -> Iterator[None]:
    """matplotlib defaults plus `_RC`, whatever rc file the user has; the previous settings come back."""
    import matplotlib

    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        matplotlib.rcParams.update(_RC)
        yield


def _styled(func: Callable[..., Figure]) -> Callable[..., Figure]:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _style():
            return func(*args, **kwargs)

    return wrapper


def _figure(height: float, width: float = WIDTH) -> Figure:
    """An empty figure on the Agg canvas, made without pyplot (no global state, no GUI backend)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, height), dpi=DPI, layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def _suptitle(fig: Figure, text: str) -> None:
    fig.suptitle(text, x=0.008, ha="left", fontsize=12, fontweight="bold")


def _method(name: str) -> tuple[str, str, str]:
    """(label, colour, marker) of a method; an unknown one keeps its name and gets a neutral colour."""
    return METHODS.get(name, (name, MUTED, "o"))


def _at_alpha(block: dict, alpha: float):
    """The entry of a report block whose keys are target rates written as strings ("0.05")."""
    for key, value in block.items():
        if math.isclose(float(key), alpha):
            return value
    raise KeyError(f"no entry for the target rate {alpha} among {list(block)}")


def _signed(x: float, digits: int = 1) -> str:
    """`+1.5` or `-1.5` with the minus sign the axis ticks use."""
    return f"{x:+.{digits}f}".replace("-", "−")


def _target(ax: Axes, alpha: float, *, outside: bool = False, label: bool = True) -> None:
    """Dashed line at the target false-alarm rate (the axis is in percent), labelled at its right end."""
    y = 100 * alpha
    ax.axhline(y, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1.5, gid="target")
    if not label:
        return
    ax.annotate(
        f"target {y:g}%",
        (1.0, y),
        xycoords=ax.get_yaxis_transform(),
        xytext=(4, 0) if outside else (-3, 3),
        textcoords="offset points",
        ha="left" if outside else "right",
        va="center" if outside else "bottom",
        fontsize=SMALL,
        color=INK_2,
        annotation_clip=False,
    )


def _points(
    ax: Axes,
    x: Sequence[float],
    values: Sequence[float],
    intervals: Sequence[Sequence[float]] | None,
    style: tuple[str, str, str],
    *,
    gid: str,
    connect: bool = False,
    scale: float = 100.0,
):
    """Markers (gid `gid`) with their intervals (gid `gid:ci`); fractions are drawn in percent."""
    label, colour, marker = style
    if intervals is not None:
        ci = scale * np.asarray(intervals, dtype=float).reshape(-1, 2)
        ax.vlines(x, ci[:, 0], ci[:, 1], color=colour, linewidth=1.3, zorder=3, gid=f"{gid}:ci")
    (line,) = ax.plot(
        x,
        scale * np.asarray(values, dtype=float),
        linestyle="-" if connect else "none",
        linewidth=1.6,
        color=colour,
        marker=marker,
        markersize=6.5,
        markeredgecolor=SURFACE,
        markeredgewidth=0.9,
        zorder=4,
        gid=gid,
        label=label,
    )
    return line


def _offsets(n: int, step: float) -> np.ndarray:
    """Sideways offsets that put n series next to each other around one x position."""
    return (np.arange(n) - (n - 1) / 2) * step


def _condition(name: str) -> tuple[str, str]:
    """(group label, tick label) of a condition name of the perturbation or the M2AD report.

    A strength that was scaled to the input size is shown on the 256 px grid, its group marked with `*`.
    """
    if name == CLEAN:
        return "clean", ""
    if name == M2AD_BASE:
        return "reference", M2AD_REFERENCE  # the lighting whose images the thresholds were fixed on
    if name.startswith("R:"):
        return "real lighting", name[2:]
    kind, _, level = name.removeprefix("P:").rpartition("-")
    if kind in LEVELS and level in {"1", "2", "3"}:
        mark = "*" if kind in _SCALED else ""
        return f"{_KINDS.get(kind, kind)}{mark}", f"{LEVELS[kind][int(level) - 1]:g}"
    return name, ""


def _scaled_note(methods: Sequence[str]) -> str:
    """Footnote of the `*` groups: the strengths are on the 256 px grid; what every other input size got."""
    scaled = [m for m in methods if INPUT_SIZES.get(m) != REFERENCE_SIZE]
    at_256 = [_method(m)[0] for m in methods if m not in scaled]
    note = f"* blur σ and shift in pixels of a {REFERENCE_SIZE} px image"
    if at_256:
        note += f", the input of {' and '.join(at_256)}"
    if not scaled:
        return note
    parts = []
    for method in scaled:
        label, size = _method(method)[0], INPUT_SIZES.get(method)
        if size is None:
            parts.append(f"{label}: input size not known")
            continue
        factor = strength("blur-1", size) / strength("blur-1", REFERENCE_SIZE)
        shifts = ", ".join(f"{strength(f'shift-{level}', size):g}" for level in (1, 2, 3))
        parts.append(f"{label} ({size} px): σ ×{factor:.3g}, shift {shifts} px")
    return f"{note}; other input sizes got them scaled:\n" + ";   ".join(parts)


def _condition_axis(
    ax: Axes, names: Sequence[str], *, labels: bool = True, methods: Sequence[str] = ()
) -> np.ndarray:
    """Put the conditions on the x axis, group by group with a gap and a hairline between the groups.

    Ticks show the strength (or the lighting id); the group name stands under the middle tick of its
    group. When a group is on the 256 px grid (`*`), the x label says what the input sizes of `methods`
    got. Returns the x position of every condition.
    """
    parsed = [_condition(name) for name in names]
    x: list[float] = []
    groups: list[list[int]] = []
    pos = 0.0
    for i, (group, _) in enumerate(parsed):
        if i and group != parsed[i - 1][0]:
            ax.axvline(pos - 0.5 + _GROUP_GAP / 2, color=GRID, linewidth=0.8, zorder=0.5)
            pos += _GROUP_GAP
            groups.append([])
        elif not i:
            groups.append([])
        groups[-1].append(i)
        x.append(pos)
        pos += 1.0
    # Every label has two lines (a single-line label would sit a little higher than the others).
    ticks = [f"{tick}\n " for _, tick in parsed]
    for members in groups:
        middle = members[len(members) // 2]
        ticks[middle] = f"{parsed[middle][1]}\n{parsed[middle][0]}"
    ax.set_xticks(x, ticks if labels else [""] * len(x))
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(x[0] - 0.75, x[-1] + 0.75)
    if labels and any(group.endswith("*") for group, _ in parsed):
        ax.set_xlabel(_scaled_note(methods), loc="left", fontsize=SMALL, labelpad=6)
    return np.asarray(x)


def _interval_handle() -> Line2D:
    """Legend entry of the vertical lines through the markers."""
    from matplotlib.lines import Line2D

    return Line2D(
        [],
        [],
        linestyle="none",
        marker="|",
        markersize=11,
        markeredgewidth=1.3,
        color=INK_2,
        label="95% bootstrap interval",
    )


def _same_conditions(tables: dict[str, list[dict]]) -> list[str]:
    """The condition list that the rows of every method share (ValueError when they differ)."""
    listed = {method: [row["condition"] for row in rows] for method, rows in tables.items()}
    names = next(iter(listed.values()))
    if any(mine != names for mine in listed.values()):
        raise ValueError(f"the methods list different conditions: {listed}")
    return names


@_styled
def fig_calibration(report: dict, alpha: float = 0.05) -> Figure:
    """Actual pooled false-alarm rate of the three threshold procedures at one target rate.

    Input: a stage 1 report (`analyze.build_report`). Three panels: all procedures on a 0-100% axis;
    hold-out and cross-fit enlarged, with the range that theory expects of the hold-out threshold
    (`theory_band`); and the rate against the calibration size (`finite_sample`) next to the interval
    simulated for exchangeable scores.
    """
    cal = _at_alpha(report["calibration"], alpha)
    curve = _at_alpha(report["finite_sample"], alpha)
    label, colour, _ = _method(report["config"]["name"])
    protocol = PROTOCOLS.get(report["protocol"], report["protocol"])
    fig = _figure(4.6)
    ax_all, ax_zoom, ax_n = fig.subplots(1, 3, width_ratios=[1.0, 0.95, 1.3])
    _suptitle(
        fig,
        f"Actual false-alarm rate of thresholds set for {100 * alpha:g}% "
        f"({label}, {protocol}, {cal['holdout']['n_normal']:,} normal images)",
    )

    def bars(ax: Axes, strategies: Sequence[str], digits: int) -> None:
        x = np.arange(len(strategies))
        rate = np.array([100 * cal[s]["fpr"] for s in strategies])
        ci = 100 * np.array([cal[s]["fpr_ci"] for s in strategies])
        drawn = ax.bar(x, rate, width=0.5, color=colour, zorder=2)
        for patch, s in zip(drawn, strategies, strict=True):
            patch.set_gid(f"bar:{s}")
        ax.vlines(x, ci[:, 0], ci[:, 1], color=INK, linewidth=1.3, zorder=3, gid="ci")
        for xi, r, hi in zip(x, rate, ci[:, 1], strict=True):
            ax.annotate(
                f"{r:.{digits}f}%",
                (xi, hi),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_xticks(x, [f"{STRATEGIES[s]}\nn = {cal[s]['n_cal_total']:,}" for s in strategies])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, len(strategies) - 0.4)
        # `n_cal_total` is the sum over the categories; the right panel's n is per category.
        ax.set_xlabel("n: calibration scores of all categories together")
        _target(ax, alpha, outside=True)

    bars(ax_all, list(STRATEGIES), 1)
    ax_all.set_ylim(0, 100)
    ax_all.set_title("Three ways to set the threshold")
    ax_all.set_ylabel("pooled false-alarm rate (%)")

    zoomed = ["holdout", "crossfit"]
    bars(ax_zoom, zoomed, 2)
    lo, hi = (100 * v for v in cal["holdout"]["theory_band"])
    at = zoomed.index("holdout") + 0.37
    ax_zoom.plot(
        [at, at],
        [lo, hi],
        color=INK_2,
        linewidth=1.3,
        marker="_",
        markersize=8,
        zorder=3,
        gid="theory:holdout",
        label="hold-out: 95% range expected in theory",
    )
    top = max(hi, 100 * alpha, *(100 * cal[s]["fpr_ci"][1] for s in zoomed))
    ax_zoom.set_ylim(0, 1.45 * top)
    ax_zoom.set_title("Hold-out and cross-fit, enlarged")
    # Name the whiskers of the bars too, so that the theory range is not taken for them (or vice versa).
    whisker = _interval_handle()
    whisker.set(color=INK, label="on the bars: 95% bootstrap interval")
    ax_zoom.legend(
        handles=[whisker, *ax_zoom.get_legend_handles_labels()[0]],
        loc="upper left",
        handlelength=1.2,
        borderaxespad=0.2,
    )

    x = np.arange(len(curve))
    ref = 100 * np.array([row["reference_interval"] for row in curve])
    ax_n.fill_between(
        x,
        ref[:, 0],
        ref[:, 1],
        color=INK_2,
        alpha=0.16,
        linewidth=0,
        zorder=1,
        gid="reference:band",
        label="simulated for exchangeable scores:\n95% interval and mean",
    )
    ax_n.plot(
        x,
        [100 * row["reference_mean"] for row in curve],
        color=INK_2,
        linewidth=1.0,
        zorder=2,
        gid="reference:mean",
    )
    ax_n.plot(
        x,
        [100 * row["mean"] for row in curve],
        linestyle="none",
        color=colour,
        marker="o",
        markersize=7,
        markeredgecolor=SURFACE,
        markeredgewidth=0.9,
        zorder=4,
        gid="observed",
        label=f"observed, mean over {curve[0]['categories']} categories",
    )
    _target(ax_n, alpha)
    ax_n.set_xticks(x, [str(row["n"]) for row in curve])
    ax_n.set_xlim(-0.5, len(curve) - 0.5)
    ax_n.set_ylim(0, 1.5 * max(100 * alpha, float(ref.max()), *(100 * row["mean"] for row in curve)))
    ax_n.set_title("Threshold from only n calibration images")
    ax_n.set_xlabel("calibration images per category (n)")
    ax_n.set_ylabel("mean false-alarm rate over categories (%)")
    ax_n.legend(loc="upper left", handlelength=1.2, borderaxespad=0.2)
    return fig


@_styled
def fig_label_curve(report: dict) -> Figure:
    """Macro image AUROC of the supervised head against the defect labels per category.

    Input: a stage 2 comparison report (`compare.build_report`). Left: the supervised mean over seeds
    with its interval and the single seeds, over the unsupervised methods drawn as horizontal lines with
    interval bands. Right: supervised against the reference method on the defects whose type was not
    among the labels, for every k where that comparison was judged.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rows = sorted(report["supervised"].values(), key=lambda row: row["k"])
    reference = report["reference_unsupervised"]
    sup_label, sup_colour, sup_marker = SUPERVISED
    fig = _figure(4.9)
    ax, ax_unseen = fig.subplots(1, 2, width_ratios=[1.85, 1.0])
    protocol = PROTOCOLS.get(report.get("protocol"), report.get("protocol"))
    _suptitle(fig, f"How many defect labels until supervised training wins ({protocol})")

    x = np.arange(len(rows))
    handles, extent = [], []
    for name, summary in report["unsupervised"].items():
        label, colour, _ = _method(name)
        value = 100 * summary["macro_image_auroc"]
        lo, hi = (100 * v for v in summary["macro_image_auroc_ci"])
        ax.axhspan(lo, hi, color=colour, alpha=0.13, linewidth=0, zorder=1, gid=f"band:{name}")
        ax.axhline(value, color=colour, linewidth=1.8, zorder=2, gid=f"line:{name}")
        ax.annotate(
            f"{value:.1f}",
            (1.0, value),
            xycoords=ax.get_yaxis_transform(),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            annotation_clip=False,
        )
        handles.append(
            (Line2D([], [], color=colour, linewidth=1.8), Patch(color=colour, alpha=0.13, linewidth=0), label)
        )
        extent += [lo, hi]

    seeds = [(xi + dx, 100 * v) for xi, row in zip(x, rows, strict=True) for dx, v in _seed_dots(row)]
    ax.plot(
        [p[0] for p in seeds],
        [p[1] for p in seeds],
        linestyle="none",
        marker="o",
        markersize=3.2,
        color=sup_colour,
        alpha=0.55,
        markeredgewidth=0,
        zorder=3,
        gid="supervised:seeds",
    )
    line = _points(
        ax,
        x,
        [row["supervised_macro_image_auroc"] for row in rows],
        [row["supervised_ci"] for row in rows],
        SUPERVISED,
        gid="supervised",
        connect=True,
    )
    extent += [100 * v for row in rows for v in (*row["supervised_ci"], *row["supervised_per_seed"])]
    ax.set_xticks(x, [f"{row['k']}\n({row['labels_with_validation']})" for row in rows])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.45, len(rows) - 0.55)
    ax.set_ylim(math.floor(min(extent) - 0.5), min(100.0, math.ceil(max(extent) + 0.5)))
    ax.set_xlabel("defect labels per category k (in brackets: with the labels used for validation)")
    ax.set_ylabel("macro image AUROC (%)")
    ax.set_title("All test defects")
    n_seeds = len(rows[0]["seeds"])
    fig.legend(
        [line, *[(h[0], h[1]) for h in handles]],
        [f"{sup_label}: mean of {n_seeds} seeds, 95% CI (small dots: seeds)", *[h[2] for h in handles]],
        loc="outside lower left",
        ncols=4,
        handlelength=1.6,
        columnspacing=1.6,
    )

    ref_label, ref_colour, ref_marker = _method(reference)
    judged = [row for row in rows if row["unseen_types"].get("judged")]
    ax_unseen.set_title("Defect types missing from the labels")
    ax_unseen.set_ylabel("macro image AUROC on these defects (%)")
    ax_unseen.tick_params(axis="x", length=0)
    if not judged:
        ax_unseen.set_xticks([])
        ax_unseen.text(
            0.5,
            0.5,
            "too few such defects at every k",
            transform=ax_unseen.transAxes,
            ha="center",
            va="center",
            color=INK_2,
        )
        return fig
    xu = np.arange(len(judged))
    sup = np.array([100 * row["unseen_types"]["supervised"] for row in judged])
    ref = np.array([100 * row["unseen_types"]["reference"] for row in judged])
    ax_unseen.vlines(xu, np.minimum(sup, ref), np.maximum(sup, ref), color=AXIS, linewidth=1.6, zorder=2)
    common = {"linestyle": "none", "markersize": 8, "markeredgecolor": SURFACE, "markeredgewidth": 0.9}
    ax_unseen.plot(xu, ref, marker=ref_marker, color=ref_colour, zorder=4, gid="unseen:reference", **common)
    ax_unseen.plot(xu, sup, marker=sup_marker, color=sup_colour, zorder=4, gid="unseen:supervised", **common)
    for xi, s, r, row in zip(xu, sup, ref, judged, strict=True):
        unseen = row["unseen_types"]
        lo, hi = (100 * v for v in unseen["ci"])
        ax_unseen.annotate(
            f"{_signed(100 * unseen['diff'])} pt\n[{_signed(lo)}, {_signed(hi)}]",
            (xi, (s + r) / 2),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    # Name the two markers once, at the first k.
    for value, text in ((ref[0], ref_label), (sup[0], sup_label)):
        ax_unseen.annotate(
            text,
            (xu[0], value),
            xytext=(-9, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=9,
            color=INK_2,
        )
    # Each seed labels other defects, so the unseen subset differs by seed: its size is a mean over seeds,
    # written with a decimal unless it is whole.
    counts = [_mean_count(row["unseen_types"]["unseen_defects_mean"]) for row in judged]
    ax_unseen.set_xticks(xu, [f"k = {row['k']}\n{n} defects" for row, n in zip(judged, counts, strict=True)])
    ax_unseen.set_xlabel(f"defects of these types: mean over the {n_seeds} seeds")
    ax_unseen.set_xlim(-1.2, len(judged) - 0.05)
    span = float(max(sup.max(), ref.max()) - min(sup.min(), ref.min())) or 1.0
    ax_unseen.set_ylim(min(sup.min(), ref.min()) - 0.35 * span, max(sup.max(), ref.max()) + 0.35 * span)
    return fig


def _mean_count(value: float) -> str:
    """A mean of counts: `60` when it is whole, `116.7` when it is not (never rounded to a count)."""
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"


def _seed_dots(row: dict) -> list[tuple[float, float]]:
    """(x offset, AUROC) of the single seeds of one k, spread a little to the right of the mean."""
    values = row["supervised_per_seed"]
    return [(0.10 + 0.045 * i, v) for i, v in enumerate(values)]


@_styled
def fig_m2ad_conditions(report: dict, alpha: float = 0.05, methods: Sequence[str] = ("p0", "d-s")) -> Figure:
    """False-alarm rate of the M2AD inspectors under every condition, and after recalibration.

    Input: the stage 3 M2AD report (`analyze_m2ad.build_report`); `alpha` is the target rate the
    thresholds were set for (the report does not store it). Left: the reference lighting, the synthetic
    conditions and the real lightings, one marker per method with its interval. Right: the pooled rate
    over the real lightings after recalibrating with n normal images of the new lighting.
    """
    from matplotlib.lines import Line2D

    runs = {m: report["methods"][m] for m in methods if m in report["methods"]}
    if not runs:
        raise ValueError(f"the report has none of the methods {list(methods)}")
    names = _same_conditions({m: run["conditions"] for m, run in runs.items()})
    fig = _figure(4.9)
    # 25 conditions on the left: its slots must hold labels like "1.25" side by side with a space between.
    ax, ax_recal = fig.subplots(1, 2, width_ratios=[4.4, 1.0])
    inspectors = len(next(iter(runs.values()))["inspectors"])
    _suptitle(fig, f"M2AD: false alarms when the capture condition changes ({inspectors} inspectors)")

    x = _condition_axis(ax, names, methods=list(runs))
    handles = []
    # Wide enough apart that the markers of two methods at one condition do not overlap.
    for dx, (method, run) in zip(_offsets(len(runs), 0.38), runs.items(), strict=True):
        rows = run["conditions"]
        handles.append(
            _points(
                ax,
                x + dx,
                [row["fpr"] for row in rows],
                [row["fpr_ci"] for row in rows],
                _method(method),
                gid=f"fpr:{method}",
            )
        )
    # The target is named in the legend: a label at the end of the line would take room from the slots.
    _target(ax, alpha, label=False)
    handles += [
        _interval_handle(),
        Line2D([], [], color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)), label=f"target {100 * alpha:g}%"),
    ]
    ax.set_ylim(-3, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("false-alarm rate on normal images (%)")
    ax.set_title(
        f"Thresholds fixed on the reference lighting {M2AD_REFERENCE}: "
        "synthetic changes (by strength), real lightings (by id)"
    )
    fig.legend(handles=handles, loc="outside lower left", ncols=len(handles), handlelength=1.2)

    sizes = [row["n"] for row in next(iter(runs.values()))["recal"]]
    xr = np.arange(len(sizes))
    for dx, (method, run) in zip(_offsets(len(runs), 0.46), runs.items(), strict=True):
        rows = run["recal"]
        if [row["n"] for row in rows] != sizes:
            raise ValueError(f"the methods were recalibrated with different sizes: {method} has other n")
        _points(
            ax_recal,
            xr + dx,
            [row["fpr"] for row in rows],
            [row["fpr_ci"] for row in rows],
            _method(method),
            gid=f"recal:{method}",
        )
        for xi, row in zip(xr + dx, rows, strict=True):
            ax_recal.annotate(
                f"{100 * row['fpr']:.1f}",
                (xi, 100 * max(row["fpr"], row["fpr_ci"][1])),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=SMALL,
            )
    # Linear below 1%, logarithmic above: 97% and 4% are both readable, and a rate of 0 still has a place.
    ax_recal.set_yscale("symlog", linthresh=1.0, linscale=0.35)
    # Room below 0 for the whole marker of a rate of 0, above 100% for the value over its marker.
    ax_recal.set_ylim(-0.3, 180)
    ax_recal.set_yticks([0, 1, 2, 5, 10, 20, 50, 100], ["0", "1", "2", "5", "10", "20", "50", "100"])
    ax_recal.minorticks_off()
    _target(ax_recal, alpha, label=False)  # the 5 tick and the legend name the line
    ax_recal.set_xticks(xr, [str(n) for n in sizes])
    ax_recal.tick_params(axis="x", length=0)
    ax_recal.set_xlim(-0.75, len(sizes) - 0.25)
    ax_recal.set_xlabel("recalibration images (n)")
    ax_recal.set_ylabel("pooled rate, real lightings (%, log scale)")
    ax_recal.set_title("After recalibration")
    return fig


@_styled
def fig_perturb(report: dict) -> Figure:
    """Accuracy change and fixed-threshold false-alarm rate under the synthetic conditions.

    Input: the stage 3 perturbation report (`analyze_perturb.build_report`). Top: change of the macro
    image AUROC against the clean condition, in points. Bottom: pooled false-alarm rate at the thresholds
    fixed on clean calibration scores (a method without calibration scores has no markers there). The
    conditions that H9 names (accuracy barely moves, false alarms at least twice the target) are shaded.
    """
    from matplotlib.patches import Patch

    tables = report["tables"]
    alpha = report["alpha"]
    names = _same_conditions(tables)
    fig = _figure(6.6)
    ax_auc, ax_fpr = fig.subplots(2, 1, sharex=True, height_ratios=[1.0, 1.1])
    protocol = PROTOCOLS.get(report.get("protocol"), report.get("protocol"))
    _suptitle(fig, f"Synthetic capture changes: accuracy and false alarms at a fixed threshold ({protocol})")

    _condition_axis(ax_auc, names, labels=False)
    x = _condition_axis(ax_fpr, names, methods=list(tables))
    handles = []
    for dx, (method, table) in zip(_offsets(len(tables), 0.25), tables.items(), strict=True):
        style = _method(method)
        changed = [(xi, row) for xi, row in zip(x, table, strict=True) if row.get("d_auroc") is not None]
        handles.append(
            _points(
                ax_auc,
                [xi + dx for xi, _ in changed],
                [row["d_auroc"] for _, row in changed],
                [row["d_auroc_ci"] for _, row in changed],
                style,
                gid=f"d_auroc:{method}",
            )
        )
        rated = [(xi, row) for xi, row in zip(x, table, strict=True) if row["fpr"] is not None]
        if rated:
            _points(
                ax_fpr,
                [xi + dx for xi, _ in rated],
                [row["fpr"] for _, row in rated],
                [row["fpr_ci"] for _, row in rated],
                style,
                gid=f"fpr:{method}",
            )
    handles.append(_interval_handle())
    labels = [h.get_label() for h in handles]
    verdict = report.get("h9")
    if verdict and verdict.get("conditions"):
        for name in verdict["conditions"]:
            at = x[names.index(name)]
            for axis in (ax_auc, ax_fpr):
                axis.axvspan(at - 0.5, at + 0.5, color=WASH, linewidth=0, zorder=0, gid=f"h9:{name}")
        handles.append(Patch(color=WASH, linewidth=0))
        labels.append(
            f"{_method(verdict['method'])[0]}: AUROC falls by less than 1 point, "
            f"false alarms at least {200 * alpha:g}%"
        )
    ax_auc.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax_auc.margins(y=0.12)  # keeps the markers at 0 clear of the panel edge
    ax_auc.set_ylabel("change in macro image AUROC (points)")
    ax_auc.set_title("Change in accuracy against the clean images")
    _target(ax_fpr, alpha, outside=True)
    ax_fpr.set_ylim(bottom=0)
    ax_fpr.set_ylabel("pooled false-alarm rate (%)")
    ax_fpr.set_title("False alarms at the threshold fixed on clean images")
    fig.legend(
        handles, labels, loc="outside lower left", ncols=len(handles), handlelength=1.2, columnspacing=1.2
    )
    return fig


def _setting(setting: str) -> tuple[str, str]:
    """`wrn50-256` -> (backbone, input size)."""
    backbone, _, size = setting.rpartition("-")
    return (backbone, size) if backbone else (setting, "")


@_styled
def fig_grid(grid: dict, latency: dict | None = None) -> Figure:
    """CPU latency per image against the dev macro image AUROC of every grid configuration.

    Input: the stage 4 grid report (`analyze_grid`) and the latency file (`bench`). The timings merged
    into the grid rows come first, because the serving choice was made on them; the latency file only
    adds the rows and precisions that the report has no timing for. One marker per (setting, coreset
    ratio, precision): x is resize + inference, for FP32 the row's registered `cpu_ms`, otherwise the
    timing's `total_with_resize_ms` (or `resize_ms` + `total_ms`); y is the row's AUROC, which the torch
    pipeline scored, so an INT8 marker shows the latency gain only. Rows without a CPU timing are left
    out. The budget is a vertical line and the chosen serving configuration is ringed.
    """
    from matplotlib.lines import Line2D

    points = []  # (setting, ratio, precision, ms, auroc in %, key)
    for row in grid["rows"]:
        entry = {**(latency or {}).get(row["key"], {}), **(row.get("latency") or {})}
        for name, timing in entry.items():
            if not name.startswith("cpu_") or not isinstance(timing, dict):
                continue  # "cpu" is the processor name, "gpu_fp16" another device
            precision = name[4:]
            if precision == "fp32" and "cpu_ms" in row:
                ms = float(row["cpu_ms"])
            elif "total_with_resize_ms" in timing:
                ms = float(timing["total_with_resize_ms"])
            else:
                ms = float(entry["resize_ms"]) + float(timing["total_ms"])
            points.append(
                (row["setting"], row["ratio"], precision, ms, 100 * row["macro_image_auroc"], row["key"])
            )
    if not points:
        raise ValueError("no grid row has a CPU latency")
    serving = grid.get("serving")
    budget = float(serving["budget_ms"]) if serving else CPU_BUDGET_MS
    protocol = PROTOCOLS.get(grid.get("protocol"), grid.get("protocol"))

    fig = _figure(5.4)
    ax = fig.subplots()
    _suptitle(fig, f"PatchCore configurations: CPU latency against accuracy ({protocol})")
    ratios = sorted({p[1] for p in points}, reverse=True)
    sizes = {r: s for r, s in zip(ratios, np.linspace(9.5, 5.0, len(ratios)), strict=True)}
    precisions = sorted({p[2] for p in points}, key=lambda p: (p != "fp32", p))
    settings = list(dict.fromkeys(p[0] for p in points))
    spare = iter(_SPARE_COLOURS)
    colours: dict[str, tuple[str, str]] = {}
    markers: dict[str, str] = {}
    for setting in settings:
        backbone, _ = _setting(setting)
        if backbone not in colours:
            colours[backbone] = BACKBONES.get(backbone) or (backbone, next(spare, MUTED))
        of_backbone = [s for s in settings if _setting(s)[0] == backbone]
        markers[setting] = "osD^v"[of_backbone.index(setting) % 5]

    for setting in settings:
        backbone, size = _setting(setting)
        colour = colours[backbone][1]
        for precision in precisions:
            mine = sorted((p for p in points if p[0] == setting and p[2] == precision), key=lambda p: -p[1])
            if not mine:
                continue
            solid = precision == "fp32"
            ax.plot(
                [p[3] for p in mine],
                [p[4] for p in mine],
                color=colour,
                linewidth=1.2,
                linestyle="-" if solid else (0, (2, 2)),
                alpha=0.55,
                zorder=2,
            )
            for p in mine:
                ax.plot(
                    [p[3]],
                    [p[4]],
                    linestyle="none",
                    marker=markers[setting],
                    markersize=sizes[p[1]],
                    color=colour,
                    markerfacecolor=colour if solid else SURFACE,
                    markeredgecolor=SURFACE if solid else colour,
                    markeredgewidth=0.9 if solid else 1.5,
                    zorder=4,
                    gid=f"point:{p[5]}:{precision}",
                )
        # The name stands next to the slowest FP32 marker of the setting (any marker if it has no FP32).
        own = [p for p in points if p[0] == setting]
        slowest = max([p for p in own if p[2] == "fp32"] or own, key=lambda p: p[3])
        ringed = bool(serving) and slowest[5] == serving["key"] and slowest[2] == "fp32"
        ax.annotate(
            f"{colours[backbone][0]} {size} px" if size else colours[backbone][0],
            (slowest[3], slowest[4]),
            xytext=(14 if ringed else 9, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_2,
        )

    ax.axvline(budget, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1.5, gid="budget")
    ax.annotate(
        f"budget {budget:g} ms",
        (budget, 1.0),
        xycoords=ax.get_xaxis_transform(),
        xytext=(4, -3),
        textcoords="offset points",
        va="top",
        fontsize=SMALL,
        color=INK_2,
    )
    if serving:
        chosen = [p for p in points if p[5] == serving["key"] and p[2] == "fp32"]
        for p in chosen:
            ax.plot(
                [p[3]],
                [p[4]],
                linestyle="none",
                marker="o",
                markersize=17,
                markerfacecolor="none",
                markeredgecolor=INK,
                markeredgewidth=1.4,
                zorder=5,
                gid="serving",
            )
            ax.annotate(
                f"serving: {p[3]:.0f} ms, AUROC {p[4]:.1f}",
                (p[3], p[4]),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_xscale("log")
    ms = [p[3] for p in points] + [budget]
    ax.set_xlim(min(ms) / 1.35, max(ms) * 2.1)
    ax.xaxis.set_major_formatter("{x:g}")
    ax.xaxis.set_minor_formatter(_log_minor_label)
    ax.tick_params(axis="x", which="both", length=3)
    ax.set_xlabel("CPU latency per image: resize + inference (ms, log scale)")
    ax.set_ylabel("macro image AUROC (%)")
    ax.margins(y=0.12)

    key = [
        Line2D([], [], linestyle="none", marker="o", markersize=sizes[r], color=INK_2, markeredgewidth=0)
        for r in ratios
    ]
    key_labels = [f"coreset {100 * r:g}%" for r in ratios]
    if precisions != ["fp32"]:
        for precision in precisions:
            solid = precision == "fp32"
            key.append(
                Line2D(
                    [],
                    [],
                    linestyle="-" if solid else (0, (2, 2)),
                    linewidth=1.2,
                    marker="o",
                    markersize=7,
                    color=INK_2,
                    markerfacecolor=INK_2 if solid else SURFACE,
                    markeredgecolor=SURFACE if solid else INK_2,
                    markeredgewidth=0.9 if solid else 1.5,
                )
            )
            key_labels.append(
                "FP32" if solid else f"{precision.upper()}: latency only, accuracy not scored again"
            )
    fig.legend(key, key_labels, loc="outside lower left", ncols=len(key), handlelength=2.2)
    return fig


def _log_minor_label(x: float, pos: int | None = None) -> str:
    """Label the 2, 3 and 5 of every decade of a log axis; the other minor ticks stay bare."""
    if x <= 0:
        return ""
    mantissa = x / 10 ** math.floor(math.log10(x))
    return f"{x:g}" if any(math.isclose(mantissa, m) for m in (2, 3, 5)) else ""


@_styled
def fig_examples(rows: list[dict], *, title: str | None = None, caption: str | None = None) -> Figure:
    """One line of panels per example: the image, its ground-truth outline, and the score map over it.

    Each row has `image` (HxWx3 uint8), `mask` (HxW bool or None: no outline is drawn), `map` (hxw
    finite float, stretched over the image), `title` and `threshold` (float or None). The colour scale
    of a row spans its map and its threshold: a map that stays below the threshold does not light up, a
    map above it everywhere lights up everywhere. The threshold is marked on the colour bar and drawn
    as a contour where the map crosses it. `title` goes over the figure, `caption` in small type under it.
    """
    from matplotlib import patheffects

    if not rows:
        raise ValueError("need at least one example")
    fig = _figure(3.3 * len(rows) + 0.1, width=10.4)
    if title:
        _suptitle(fig, title)
    if caption:
        fig.supxlabel(caption, x=0.008, ha="left", fontsize=SMALL, color=INK_2, linespacing=1.5)
    grid = fig.subplots(len(rows), 3, squeeze=False)
    headers = ("", "ground-truth outline", "anomaly map")
    outline = [patheffects.withStroke(linewidth=3.2, foreground=INK)]
    for line, row in zip(grid, rows, strict=True):
        image = np.asarray(row["image"])
        scores = np.asarray(row["map"], dtype=np.float64)
        height, width = image.shape[:2]
        if scores.ndim != 2 or not np.isfinite(scores).all():
            raise ValueError(f"the map of {row['title']!r} must be a 2-d array of finite scores")
        if row["mask"] is not None and np.shape(row["mask"]) != (height, width):
            raise ValueError(f"the mask of {row['title']!r} is not the size of its image {(height, width)}")
        extent = (-0.5, width - 0.5, height - 0.5, -0.5)
        threshold = row.get("threshold")
        low, high = float(scores.min()), float(scores.max())
        if threshold is not None:
            threshold = float(threshold)
        vmin, vmax = (low, high) if threshold is None else (min(low, threshold), max(high, threshold))
        if vmax <= vmin:
            vmax = vmin + 1.0
        for ax, header in zip(line, headers, strict=True):
            ax.imshow(image, extent=extent, interpolation="nearest")
            ax.set_axis_off()
            if header:
                ax.set_title(header, fontweight="normal", color=INK_2, fontsize=9.5)
            else:
                ax.set_title(row["title"])
        if row["mask"] is not None:
            drawn = line[1].contour(
                np.asarray(row["mask"], dtype=np.float64), levels=[0.5], colors=[SURFACE], linewidths=1.4
            )
            drawn.set_path_effects(outline)
        else:
            line[1].text(
                0.03,
                0.04,
                "normal image: no defect mask",
                transform=line[1].transAxes,
                fontsize=SMALL,
                color=SURFACE,
                path_effects=[patheffects.withStroke(linewidth=2.5, foreground=INK)],
            )
        # Low scores stay transparent, so the image is visible where nothing is flagged.
        opacity = 0.75 * np.clip((scores - vmin) / (vmax - vmin), 0.0, 1.0)
        heat = line[2].imshow(
            scores,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            alpha=opacity,
            extent=extent,
            interpolation="bilinear",
        )
        bar = fig.colorbar(heat, ax=line[2], fraction=0.046, pad=0.03)
        bar.ax.set_label("colorbar")
        bar.solids.set_alpha(1.0)
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=8, length=2)
        bar.set_label(
            "anomaly score" if threshold is None else "anomaly score (line: threshold)",
            fontsize=SMALL,
            color=INK_2,
        )
        if threshold is not None:
            bar.ax.axhline(threshold, color=SURFACE, linewidth=2.6)
            bar.ax.axhline(threshold, color=INK, linewidth=1.2, gid="threshold")
            # A contour needs a map of at least 2 x 2 scores.
            if low < threshold < high and min(scores.shape) >= 2:
                ys = np.linspace(-0.5, height - 0.5, scores.shape[0] * 2 + 1)[1::2]
                xs = np.linspace(-0.5, width - 0.5, scores.shape[1] * 2 + 1)[1::2]
                flagged = line[2].contour(
                    xs, ys, scores, levels=[threshold], colors=[SURFACE], linewidths=1.2
                )
                flagged.set_path_effects(outline)
    return fig


# Name -> (input files under the reports folder, function that gets the loaded files in that order).
FIGURES: dict[str, tuple[tuple[str, ...], Callable[..., Figure]]] = {
    "calibration": (("stage1/p0-test.json",), fig_calibration),
    "label_curve": (("stage2/compare-test.json",), fig_label_curve),
    "m2ad_conditions": (("stage3/m2ad.json",), fig_m2ad_conditions),
    "perturb": (("stage3/perturb-test.json",), fig_perturb),
    "grid": (("stage4/grid-dev.json", "stage4/latency.json"), fig_grid),
}


def save(fig: Figure, path: Path) -> Path:
    """Write the figure as a PNG whose bytes depend on the figure only (no software version in it)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _style():
        fig.savefig(path, format="png", dpi=DPI, metadata={"Software": None})
    return path


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---- example inspections of an evaluation run ------------------------------------------------------------

EXAMPLE_GROUPS = ("detected defect", "missed defect", "false alarm")
PER_GROUP = 2  # rows per group
EXAMPLES_STAGE = "5"  # stage label of the test ledger line
EXAMPLES_NOTE = "README example panel: images and masks of the picked rows read for a figure only, no scores"
DATA_CREDIT = "Images: VisA (Zou et al., ECCV 2022), CC BY 4.0"
# Where `compare` takes the calibration scores of each kind of run from.
_CALIBRATION_SOURCE = {
    "crossfit": "the cross-fitted (out-of-fold) scores of the normal pool",
    "holdout": "the scores of held-out normal images",
}
_EVAL_KEYS = ("eval_images", "eval_labels", "eval_defect_types", "eval_score")
_TITLE_CHARS = 34  # a row title is wrapped to stay over the first panel of its row
_TITLE_LINES = 3  # beyond this, a long list of defect types is shortened ("+ 3 more")
_CAPTION_CHARS = 165


class ExamplesError(ValueError):
    """The run folder cannot give the example panel: incomplete, of an unknown kind or without thresholds."""


@dataclass(frozen=True)
class Example:
    """One evaluation image picked for the panel; `index` is its row in its category's arrays and maps."""

    group: str
    category: str
    index: int
    image: str
    label: int
    defect_types: str
    margin: float  # image score / threshold of its category

    def _parts(self, keep: int | None = None) -> tuple[str, str]:
        """(group, category and kind; score/threshold); only the first `keep` defect types are named."""
        if self.label == 1:
            types = [t for t in self.defect_types.split("|") if t] or ["defect"]
            named = types if keep is None else types[:keep]
            kind = " + ".join(named) + (
                f" + {len(types) - len(named)} more" if len(named) < len(types) else ""
            )
        else:
            kind = "normal"
        return f"{self.group}: {self.category}, {kind},", f"score/threshold = {_ratio(self.margin)}"

    @property
    def title(self) -> str:
        return " ".join(self._parts())

    def _wrap(self, width: int, keep: int | None) -> list[str]:
        head, tail = self._parts(keep)
        lines = textwrap.wrap(head, width, break_long_words=False, break_on_hyphens=False)
        if len(lines[-1]) + 1 + len(tail) <= width:
            lines[-1] += f" {tail}"
        else:
            lines.append(tail)
        return lines

    def wrapped_title(self, width: int = _TITLE_CHARS, max_lines: int = _TITLE_LINES) -> str:
        """`title` in lines of at most `width` characters; "score/threshold = x" is never split.

        When the title takes more than `max_lines` lines, the defect types after the first ones that fit
        are counted ("+ 3 more") instead of named.
        """
        lines = self._wrap(width, None)
        n_types = len(self.defect_types.split("|"))
        for keep in range(n_types - 1, 0, -1):
            if len(lines) <= max_lines:
                break
            lines = self._wrap(width, keep)
        return "\n".join(lines)


def _ratio(margin: float) -> str:
    """Two decimals, or as many more as it takes to keep a value that is not 1 off "1.00" (its side of 1)."""
    if margin == 1.0:
        return "1.00"
    for digits in range(2, 18):  # 16 decimals tell any float64 from 1
        text = f"{margin:.{digits}f}"
        if float(text) != 1.0:
            return text
    return repr(margin)


def _one_per_category(ordered: list[Example], n: int) -> list[Example]:
    taken: list[Example] = []
    for example in ordered:
        if len(taken) == n:
            break
        if all(example.category != other.category for other in taken):
            taken.append(example)
    return taken


def pick_examples(
    cats: dict[str, dict[str, np.ndarray]], thresholds: dict[str, float], per_group: int = PER_GROUP
) -> dict[str, list[Example]]:
    """The rows of the panel, by a fixed rule that looks at the scores only.

    margin = image score / threshold of its category. An image is flagged when score > threshold (the
    decision rule of `calibrate`), so a flagged image has margin > 1. Groups, in this order:

    - detected defect (defect, flagged): the `per_group` whose margin is closest to the median margin
      of all detected defects of the run (typical hits);
    - missed defect (defect, not flagged): the `per_group` with the lowest margin;
    - false alarm (normal, flagged): the `per_group` with the highest margin.

    At most one image per category in each group; ties are broken by image path. A group with fewer
    candidates keeps what it has. `cats` holds `eval_images`, `eval_labels` (0 or 1),
    `eval_defect_types` and `eval_score` per category; every threshold must be positive.
    """
    found: dict[str, list[Example]] = {group: [] for group in EXAMPLE_GROUPS}
    for category, cat in cats.items():
        threshold = float(thresholds[category])
        if not (math.isfinite(threshold) and threshold > 0):
            raise ExamplesError(f"the threshold of {category} is {threshold}: score/threshold needs it > 0")
        scores = np.asarray(cat["eval_score"], dtype=np.float64)
        labels = np.asarray(cat["eval_labels"])
        if not np.isfinite(scores).all():
            raise ExamplesError(f"the evaluation scores of {category} are not all finite")
        if not np.isin(labels, (0, 1)).all():
            raise ExamplesError(f"the evaluation labels of {category} are not all 0 or 1")
        for i, (score, label) in enumerate(zip(scores, labels, strict=True)):
            flagged = bool(score > threshold)
            if label == 1:
                group = EXAMPLE_GROUPS[0] if flagged else EXAMPLE_GROUPS[1]
            elif flagged:
                group = EXAMPLE_GROUPS[2]
            else:
                continue
            found[group].append(
                Example(
                    group=group,
                    category=category,
                    index=i,
                    image=str(cat["eval_images"][i]),
                    label=int(label),
                    defect_types=str(cat["eval_defect_types"][i]),
                    margin=float(score / threshold),
                )
            )
    detected = found[EXAMPLE_GROUPS[0]]
    median = float(np.median([e.margin for e in detected])) if detected else 0.0
    keys: dict[str, Callable[[Example], tuple[float, str]]] = {
        EXAMPLE_GROUPS[0]: lambda e: (abs(e.margin - median), e.image),
        EXAMPLE_GROUPS[1]: lambda e: (e.margin, e.image),
        EXAMPLE_GROUPS[2]: lambda e: (-e.margin, e.image),
    }
    return {
        group: _one_per_category(sorted(found[group], key=keys[group]), per_group) for group in EXAMPLE_GROUPS
    }


def example_thresholds(run: MethodRun, alpha: float = ALPHA) -> dict[str, float]:
    """Per-category threshold as `compare` fixes it: the conformal threshold of the run's `cal_score`."""
    empty = [c for c, cat in run.cats.items() if len(cat["cal_score"]) == 0]
    if empty:
        raise ExamplesError(
            f"{run.name} has no calibration scores for {len(empty)} of {len(run.cats)} categories "
            f"({', '.join(empty)}), so there is no threshold to draw. A dev-protocol Dinomaly run keeps "
            "none (its held-out normals are its evaluation normals): use a run that has them"
        )
    try:
        return {c: conformal_threshold(cat["cal_score"], alpha).value for c, cat in run.cats.items()}
    except ValueError as err:
        raise ExamplesError(f"the calibration scores of {run.name} give no threshold: {err}") from err


def _run_meta(run_dir: Path) -> dict:
    path = run_dir / "run.json"
    if not path.is_file():
        raise ExamplesError(f"{run_dir} is not a finished evaluation run: it has no run.json")
    try:
        meta = _load(path)
    except (OSError, ValueError) as err:  # a JSONDecodeError is a ValueError
        raise ExamplesError(f"{path} cannot be read: {type(err).__name__}: {err}") from err
    if not isinstance(meta, dict):
        raise ExamplesError(f"{path} is not the run.json of an evaluation run")
    return meta


def _run_categories(meta: dict, path: Path) -> list[str]:
    """Category names of run.json; each becomes a file name in the run folder, so it must be a plain name."""
    try:
        names = [info["category"] for info in meta["categories"]]
    except (KeyError, TypeError) as err:
        raise ExamplesError(f"{path} does not list its categories as an evaluation run does") from err
    if not names:
        raise ExamplesError(f"{path} lists no categories")
    for name in names:
        plain = isinstance(name, str) and name not in ("", ".", "..")
        if not (plain and PurePosixPath(name).name == name == PureWindowsPath(name).name):
            raise ExamplesError(f"{path} lists {name!r}, which is not a plain category name")
    twice = [name for name, n in Counter(names).items() if n > 1]
    if twice:
        raise ExamplesError(f"{path} lists {', '.join(twice)} twice")
    return names


# Errors of np.load on a damaged file: a truncated .npz is no zip file, an empty file has no header.
_READ_ERRORS = (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile)


def _npz_arrays(path: Path, keys: Sequence[str]) -> dict[str, np.ndarray]:
    """Only `keys` of an .npz file: the other arrays in it are not read."""
    try:
        with np.load(path) as z:
            arrays = {key: z[key] for key in keys if key in z.files}
    except _READ_ERRORS as err:
        raise ExamplesError(f"{path} cannot be read: {type(err).__name__}: {err}") from err
    missing = [key for key in keys if key not in arrays]
    if missing:
        raise ExamplesError(f"{path} has no {', '.join(missing)}")
    return arrays


def _calibration_images_key(meta: dict) -> str:
    """Key of the images behind `cal_score`; a `run_patchcore` run (no "method") calls them its pool."""
    return "cal_images" if meta.get("method") else "pool_images"


def _check_scored_images(
    run_dir: Path, names: list[str], cal_key: str, sealed: bool
) -> tuple[dict[str, ManifestRow], dict[str, dict[str, np.ndarray]]]:
    """Check every image the run scored against the manifest before any score is read.

    Only the image lists and the evaluation labels are read. Every evaluation and calibration image must
    be a manifest image of its category, with the manifest's label (calibration images are normal), and
    no evaluation image may be listed twice. A run that is not a test-protocol run must hold no sealed
    test image in either list, drawn or not: its scores would set a threshold or move the picks.
    Returns the manifest rows by image and the arrays read, by category.
    """
    try:
        manifest = read_manifest(paths.VISA_MANIFEST)
    except (OSError, ValueError) as err:
        raise ExamplesError(f"the manifest {paths.VISA_MANIFEST} cannot be read: {err}") from err
    by_image = {row.image: row for row in manifest}
    known = {row.category for row in manifest}
    for category in names:
        if category not in known:
            raise ExamplesError(f"{category} is not a category of the manifest {paths.VISA_MANIFEST}")
    lists = {}
    for category in names:
        path = run_dir / f"{category}.npz"
        arrays = _npz_arrays(path, ("eval_images", "eval_labels", cal_key))
        evaluated, labels, calibration = arrays["eval_images"], arrays["eval_labels"], arrays[cal_key]
        if evaluated.ndim != 1 or calibration.ndim != 1 or labels.shape != evaluated.shape:
            shapes = {key: values.shape for key, values in arrays.items()}
            raise ExamplesError(f"{path}: {shapes} are not lists, one label per evaluation image")
        images = [str(image) for image in evaluated]
        scored = [*zip(images, labels.tolist(), strict=True), *((str(image), 0) for image in calibration)]
        for image, label in scored:
            row = by_image.get(image)
            if row is None:
                raise ExamplesError(
                    f"{path} lists {image}, which is not in the manifest {paths.VISA_MANIFEST}"
                )
            if row.role in SEALED_ROLES and not sealed:
                raise SealedTestError(
                    f"{path} scores {image}, a sealed test image, although the run is not a test-protocol run"
                )
            if row.category != category or (row.label == "anomaly") != (label == 1):
                raise ExamplesError(
                    f"the manifest has {image} as {row.label} of {row.category}; the run has it as "
                    f"label {label} of {category}"
                )
        twice = [image for image, n in Counter(images).items() if n > 1]
        if twice:
            raise ExamplesError(f"{path} lists {', '.join(twice)} twice in eval_images")
        lists[category] = arrays
    return by_image, lists


def _load_scores(run_dir: Path, meta: dict, names: list[str]) -> MethodRun:
    """The run as `compare` loads it: a `run_patchcore` run (no "method" in run.json) or the common format."""
    try:
        if meta.get("method"):
            config = meta.get("config") or {}
            run = load_common(run_dir, str(config.get("name") or meta["method"]), names)
        else:
            run = load_patchcore(run_dir)
    except (AttributeError, KeyError, *_READ_ERRORS) as err:
        raise ExamplesError(
            f"{run_dir} is not a complete evaluation run: {type(err).__name__}: {err}"
        ) from err
    if list(run.cats) != names:
        raise ExamplesError(f"{run_dir}: the categories read {list(run.cats)} are not those of run.json")
    for category, cat in run.cats.items():
        missing = [key for key in (*_EVAL_KEYS, "cal_score") if key not in cat]
        if missing:
            raise ExamplesError(f"{run_dir / f'{category}.npz'} has no {', '.join(missing)}")
        sizes = {key: len(cat[key]) for key in _EVAL_KEYS}
        if len(set(sizes.values())) != 1:
            raise ExamplesError(
                f"{run_dir / f'{category}.npz'}: evaluation arrays of different lengths {sizes}"
            )
    return run


def _check_maps(path: Path, n: int) -> None:
    """The maps file holds one 2-d map per evaluation image (only its header is read)."""
    try:
        maps = np.load(path, mmap_mode="r")
    except _READ_ERRORS as err:
        raise ExamplesError(f"{path} cannot be read: {type(err).__name__}: {err}") from err
    shape = getattr(maps, "shape", None)  # an .npz under this name loads as an NpzFile
    del maps
    if shape is None or len(shape) != 3 or shape[0] != n:
        raise ExamplesError(f"{path} has shape {shape}: expected {n} maps, one per evaluation image")


def _read_map(path: Path, index: int) -> np.ndarray:
    maps = np.load(path, mmap_mode="r")
    row = np.array(maps[index], dtype=np.float64)
    del maps  # Windows keeps a mapped file locked
    return row


def _examples_title(label: str, protocol: str, picks: dict[str, list[Example]]) -> str:
    title = f"Example inspections: {label} on the {PROTOCOLS[protocol]}"
    short = [(group, len(picks[group])) for group in EXAMPLE_GROUPS if len(picks[group]) < PER_GROUP]
    if short:
        counts = ", ".join(f"{n} {group}{'' if n == 1 else 's'}" for group, n in short)
        title += f"\nFewer than {PER_GROUP} found (one per category at most): {counts}"
    return title


def _examples_caption(label: str, calibration: str) -> str:
    source = _CALIBRATION_SOURCE.get(calibration, f"its {calibration} calibration scores")
    lines = [
        f"{DATA_CREDIT}. Method: {label}. Threshold per category: split-conformal at a {ALPHA:.0%} target "
        f"false-alarm rate, from {source}. An image is flagged when its score/threshold is above 1.",
        "Rows picked by a fixed rule, at most one image per category in each group, ties broken by path: "
        "the detected defects whose score/threshold is closest to the median of all detected defects, the "
        "missed defects with the lowest and the false alarms with the highest score/threshold.",
        "The line on a map marks pixels above the image threshold; the image score is not the map's "
        "maximum, so near the threshold the line and the decision can disagree.",
    ]
    return "\n".join(textwrap.fill(line, _CAPTION_CHARS, break_on_hyphens=False) for line in lines)


def examples(run_dir: Path, out: Path, name: str = "examples", *, allow_test: bool = False) -> Path:
    """Draw the example inspections of an evaluation run and write `<out>/<name>.png`.

    `run_dir` is an evaluation folder: run.json, `<category>.npz` and `<category>_maps.npy` (one map per
    evaluation image, in `eval_images` order). Scores are loaded as `compare` loads them and each
    category gets the threshold `compare` fixes at `compare.ALPHA`: cross-fit for a PatchCore run,
    hold-out for a run in the common format (Dinomaly). The rows follow the rule of `pick_examples`:
    two detected defects nearest the median score/threshold of all detected defects, the two missed
    defects with the lowest and the two false alarms with the highest score/threshold, at most one image
    per category in each group, ties broken by image path; a group that has fewer keeps what it has and
    the figure title says so.

    A test-protocol run is refused unless `allow_test is True`. Before any score is read, every image the
    run scored (evaluation and calibration) is checked against the manifest: a run that is not a
    test-protocol run holding a sealed test image anywhere is refused. Images and masks come from the
    256 px cache (`paths.CACHE`) through the manifest rows of the picked images; for a test-protocol run
    one line goes to the test ledger after every check that can be made without them and before the
    first image is read. The maps of the picked rows are read after the images (a map shows where the
    defect of its image is). Raises ExamplesError or SealedTestError; then no PNG is written, and the
    ledger line only when images were read.
    """
    run_dir, out = Path(run_dir), Path(out)
    meta = _run_meta(run_dir)
    protocol = meta.get("protocol")
    if not isinstance(protocol, str) or protocol not in PROTOCOLS:
        raise ExamplesError(f"{run_dir / 'run.json'} names no known protocol: {protocol!r}")
    sealed = protocol == "test"
    if sealed and allow_test is not True:  # not truthiness, as in splits.select
        raise SealedTestError(
            f"{run_dir} is a test-protocol run: its images are the sealed test set. Pass --allow-test "
            "(the read is logged in the test ledger) to draw them"
        )
    names = _run_categories(meta, run_dir / "run.json")
    wanted = [run_dir / f"{c}{suffix}" for c in names for suffix in (".npz", "_maps.npy")]
    missing = [path.name for path in wanted if not path.is_file()]
    if missing:
        raise ExamplesError(f"{run_dir} is incomplete: it has no {', '.join(missing)}")
    cal_key = _calibration_images_key(meta)
    by_image, checked = _check_scored_images(run_dir, names, cal_key, sealed)
    run = _load_scores(run_dir, meta, names)
    for category, cat in run.cats.items():  # the second read must hold the images the first one checked
        if any(not np.array_equal(cat.get(key), checked[category][key]) for key in ("eval_images", cal_key)):
            raise ExamplesError(f"{run_dir / f'{category}.npz'} changed while it was being read")
    thresholds = example_thresholds(run)
    picks = pick_examples(run.cats, thresholds)
    chosen = [example for group in EXAMPLE_GROUPS for example in picks[group]]
    if not chosen:
        raise ExamplesError(f"no image of {run_dir} falls in any group")
    for category in dict.fromkeys(example.category for example in chosen):
        _check_maps(run_dir / f"{category}_maps.npy", len(run.cats[category]["eval_images"]))
    rows = [by_image[example.image] for example in chosen]
    try:
        cache = ImageCache(paths.CACHE, MASK_SIZE)
    except FileNotFoundError as err:
        raise ExamplesError(
            f"no {MASK_SIZE} px image cache in {paths.CACHE}: build it with "
            f"`python -m defect_inspect.cache --size {MASK_SIZE}`"
        ) from err
    except ValueError as err:
        raise ExamplesError(f"the image cache in {paths.CACHE} cannot be used: {err}") from err
    try:
        try:
            cache._rows(rows)  # index lookup only: a missing image stops the run before the ledger line
        except KeyError as err:
            raise ExamplesError(
                f"{paths.CACHE}: {err.args[0]} (rebuild the cache from the manifest)"
            ) from None
        if sealed:
            ledger.record_test_access(
                paths.TEST_LEDGER, stage=EXAMPLES_STAGE, config=f"examples-{run.name}", note=EXAMPLES_NOTE
            )
        images, masks = cache.images(rows), cache.masks(rows)
    finally:
        cache.close()
    panel = []
    for example, image, mask in zip(chosen, images, masks, strict=True):
        path = run_dir / f"{example.category}_maps.npy"
        try:
            scores = _read_map(path, example.index)
        except _READ_ERRORS as err:
            raise ExamplesError(f"{path} cannot be read: {type(err).__name__}: {err}") from err
        if not np.isfinite(scores).all():
            raise ExamplesError(f"{path}: the map of {example.image} (row {example.index}) is not all finite")
        panel.append(
            {
                "image": image,
                "mask": mask.astype(bool) if example.label == 1 else None,
                "map": scores,
                "title": example.wrapped_title(),
                "threshold": thresholds[example.category],
            }
        )
    label = _method(run.name)[0]
    fig = fig_examples(
        panel,
        title=_examples_title(label, protocol, picks),
        caption=_examples_caption(label, run.calibration),
    )
    return save(fig, out / f"{name}.png")


def _examples_main(args: argparse.Namespace) -> None:
    try:
        path = examples(args.examples, args.out, args.name, allow_test=args.allow_test)
    except (ExamplesError, SealedTestError) as err:
        print(f"examples not drawn: {err}", file=sys.stderr)
        raise SystemExit(2) from None
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports", type=Path, default=paths.REPORTS, help="folder of the report files")
    parser.add_argument("--out", type=Path, default=paths.ROOT / "docs" / "figures", help="PNG folder")
    parser.add_argument(
        "--only", nargs="+", choices=list(FIGURES), default=None, metavar="NAME", help="default: all figures"
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="draw only the example inspections of this evaluation run folder",
    )
    parser.add_argument(
        "--allow-test", action="store_true", help="with --examples: read sealed test images (logged)"
    )
    parser.add_argument(
        "--name", default=None, help="with --examples: PNG name without .png (default examples)"
    )
    args = parser.parse_args(argv)
    if args.examples is None and (args.allow_test or args.name is not None):
        parser.error("--allow-test and --name go with --examples")
    if args.examples is not None and args.only is not None:
        parser.error("--only names report figures; it does not go with --examples")
    if args.name is not None and (not args.name or Path(args.name).name != args.name):
        parser.error(f"--name must be a plain file name, got {args.name!r}")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # output paths outside the console code page
    if args.examples is not None:
        args.name = args.name or "examples"
        _examples_main(args)
        return
    failed = []
    for name in args.only or FIGURES:
        inputs, build = FIGURES[name]
        files = [args.reports / rel for rel in inputs]
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            print(f"skipped {name}: no {', '.join(missing)}")
            continue
        # A report that cannot be read or drawn stops its own figure only; the run fails at the end.
        try:
            path = save(build(*(_load(f) for f in files)), args.out / f"{name}.png")
        except Exception as err:
            print(
                f"failed {name} ({', '.join(map(str, files))}): {type(err).__name__}: {err}", file=sys.stderr
            )
            failed.append(name)
            continue
        print(f"wrote {path}")
    if failed:
        raise SystemExit(f"figures not written: {', '.join(failed)}")


if __name__ == "__main__":
    main()
