"""figures on synthetic reports and on the committed ones (needs the `figures` dependency group)."""

import copy
import io
import json
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

from defect_inspect import analyze_grid, analyze_perturb, figures, paths  # noqa: E402
from defect_inspect.conditions import condition_names  # noqa: E402
from defect_inspect.run_grid import ratio_name  # noqa: E402
from defect_inspect.visa import CATEGORIES  # noqa: E402

PERTURBED = condition_names(include_clean=False)
M2AD_CONDITIONS = ["S", *[f"P:{name}" for name in PERTURBED], *[f"R:{i:02d}" for i in range(2, 11)]]
SHIFTED = "brightness-2"  # every score moves up: the ranking stays, the fixed thresholds are overrun


def _artists(fig, gid):
    return fig.findobj(lambda artist: artist.get_gid() == gid)


def _one(fig, gid):
    found = _artists(fig, gid)
    assert len(found) == 1, f"{len(found)} artists with gid {gid!r}"
    return found[0]


def _x(fig, gid):
    return list(_one(fig, gid).get_xdata())


def _y(fig, gid):
    return list(_one(fig, gid).get_ydata())


def _whiskers(fig, gid):
    """(x, low, high) of every interval line drawn under `gid`."""
    return [(seg[0, 0], seg[0, 1], seg[1, 1]) for seg in _one(fig, gid).get_segments()]


def _notes(ax):
    """Text -> anchor point of the annotations of an axes."""
    return {t.get_text(): tuple(t.xy) for t in ax.texts if hasattr(t, "xy")}


def _texts(fig):
    return [t.get_text() for t in fig.findobj(lambda artist: hasattr(artist, "get_text"))]


def _png(fig, path):
    return figures.save(fig, path).read_bytes()


def _marks_inside(fig):
    """Every line, whole marker, interval, band and bar lies inside the view of its axes (none is cut off).

    Artist data does not change when the view limits cut a mark off, so the check is made in pixels.
    """
    with figures._style():
        fig.canvas.draw()
    for ax in fig.axes:
        box = ax.bbox
        for artist in [*ax.lines, *ax.collections, *ax.patches]:
            room = -1.0  # pixels between a point and the edge; a line may touch it
            if isinstance(artist, Line2D):
                points = artist.get_transform().transform(artist.get_xydata())
                if artist.get_marker() not in ("", " ", "None", "none", None):
                    room = artist.get_markersize() * fig.dpi / 72 / 2 - 1.0  # the whole marker
            elif isinstance(artist, Patch):
                ext = artist.get_extents()
                points = np.array([[ext.x0, ext.y0], [ext.x1, ext.y1]])
            else:
                vertices = [p.vertices for p in artist.get_paths() if len(p.vertices)]
                if not vertices:
                    continue
                points = artist.get_transform().transform(np.concatenate(vertices))
            inside = (
                (points[:, 0] >= box.x0 + room)
                & (points[:, 0] <= box.x1 - room)
                & (points[:, 1] >= box.y0 + room)
                & (points[:, 1] <= box.y1 - room)
            )
            where = ax.get_title(loc="left") or ax.get_label()
            assert inside.all(), f"{artist.get_gid() or artist} is cut off in the axes {where!r}"


# ---- calibration -----------------------------------------------------------------------------------------


def _calibration_report():
    def strategy(fpr, half, n_cal, band):
        return {
            "fpr": fpr,
            "fpr_ci": [fpr - half, fpr + half],
            "n_cal_total": n_cal,
            "n_normal": 1000,
            "theory_band": band,
        }

    def curve(scale):
        return [
            {
                "n": n,
                "categories": 3,
                "mean": scale * mean,
                "reference_mean": scale * ref,
                "reference_interval": [scale * (ref - 0.01), scale * (ref + 0.01)],
            }
            for n, mean, ref in ((20, 0.041, 0.048), (50, 0.033, 0.039), (100, 0.040, 0.050))
        ]

    return {
        "config": {"name": "p0"},
        "protocol": "test",
        "calibration": {
            "0.05": {
                "resubstitution": strategy(0.80, 0.02, 900, [0.04, 0.06]),
                "holdout": strategy(0.055, 0.01, 180, [0.035, 0.065]),
                "crossfit": strategy(0.04, 0.008, 900, [0.04, 0.06]),
            },
            "0.01": {
                "resubstitution": strategy(0.70, 0.02, 900, [0.006, 0.014]),
                "holdout": strategy(0.012, 0.004, 180, [0.004, 0.018]),
                "crossfit": strategy(0.008, 0.003, 900, [0.006, 0.014]),
            },
        },
        "finite_sample": {"0.05": curve(1.0), "0.01": curve(0.2)},
    }


def test_calibration_draws_the_rates_of_the_report_in_percent():
    fig = figures.fig_calibration(_calibration_report())
    assert len(fig.axes) == 3
    full, zoom, by_n = fig.axes
    assert fig.get_suptitle() == (
        "Actual false-alarm rate of thresholds set for 5% (PatchCore WRN-50, VisA test split, "
        "1,000 normal images)"
    )
    # Every procedure on the full axis; hold-out and cross-fit once more on the enlarged one.
    assert [p.get_gid() for p in full.patches] == ["bar:resubstitution", "bar:holdout", "bar:crossfit"]
    assert [p.get_height() for p in full.patches] == pytest.approx([80.0, 5.5, 4.0])
    assert [p.get_x() + p.get_width() / 2 for p in full.patches] == pytest.approx(full.get_xticks())
    assert [p.get_gid() for p in zoom.patches] == ["bar:holdout", "bar:crossfit"]
    assert [p.get_height() for p in zoom.patches] == pytest.approx([5.5, 4.0])
    assert full.get_ylim() == (0.0, 100.0) and zoom.get_ylim()[1] < 12.0
    assert [t.get_text() for t in full.get_xticklabels()] == [
        "Resubstitution\nn = 900",
        "Hold-out\nn = 180",
        "Cross-fit\nn = 900",
    ]
    # The bootstrap intervals are the whiskers of the bars, in both panels.
    (whiskers,) = full.findobj(lambda a: a.get_gid() == "ci")
    assert [(s[0, 0], s[0, 1], s[1, 1]) for s in whiskers.get_segments()] == [
        pytest.approx((0, 78.0, 82.0)),
        pytest.approx((1, 4.5, 6.5)),
        pytest.approx((2, 3.2, 4.8)),
    ]
    (whiskers,) = zoom.findobj(lambda a: a.get_gid() == "ci")
    assert [(s[0, 0], s[0, 1], s[1, 1]) for s in whiskers.get_segments()] == [
        pytest.approx((0, 4.5, 6.5)),
        pytest.approx((1, 3.2, 4.8)),
    ]
    # The value over each whisker, and the target named at the right end of its line.
    assert _notes(full) == {
        "80.0%": pytest.approx((0, 82.0)),
        "5.5%": pytest.approx((1, 6.5)),
        "4.0%": pytest.approx((2, 4.8)),
        "target 5%": pytest.approx((1.0, 5.0)),
    }
    assert _notes(zoom) == {
        "5.50%": pytest.approx((0, 6.5)),
        "4.00%": pytest.approx((1, 4.8)),
        "target 5%": pytest.approx((1.0, 5.0)),
    }
    # One target line per panel, at 5%.
    targets = _artists(fig, "target")
    assert len(targets) == 3
    assert all(list(line.get_ydata()) == pytest.approx([5.0, 5.0]) for line in targets)
    # The theory range of the hold-out threshold, next to its bar.
    assert _y(fig, "theory:holdout") == pytest.approx([3.5, 6.5])
    assert _one(fig, "theory:holdout").axes is zoom
    # The finite-sample curve: observed means, and the simulated mean inside its interval, n by n.
    assert _x(fig, "observed") == pytest.approx(by_n.get_xticks())
    assert _y(fig, "observed") == pytest.approx([4.1, 3.3, 4.0])
    assert _y(fig, "reference:mean") == pytest.approx([4.8, 3.9, 5.0])
    band = np.concatenate([p.vertices for p in _one(fig, "reference:band").get_paths()])
    for x, (lo, hi) in zip(by_n.get_xticks(), ((3.8, 5.8), (2.9, 4.9), (4.0, 6.0)), strict=True):
        at = band[np.isclose(band[:, 0], x), 1]
        assert (at.min(), at.max()) == pytest.approx((lo, hi))
    assert [t.get_text() for t in by_n.get_xticklabels()] == ["20", "50", "100"]
    assert _notes(by_n) == {"target 5%": pytest.approx((1.0, 5.0))}
    _marks_inside(fig)


def test_calibration_at_another_target_rate():
    fig = figures.fig_calibration(_calibration_report(), alpha=0.01)
    assert [p.get_height() for p in fig.axes[0].patches] == pytest.approx([70.0, 1.2, 0.8])
    assert all(list(line.get_ydata()) == pytest.approx([1.0, 1.0]) for line in _artists(fig, "target"))
    assert _y(fig, "observed") == pytest.approx([0.82, 0.66, 0.8])
    assert "set for 1% (" in fig.get_suptitle() and "target 1%" in _notes(fig.axes[0])
    _marks_inside(fig)
    with pytest.raises(KeyError, match="0.1"):
        figures.fig_calibration(_calibration_report(), alpha=0.1)


# ---- label curve -----------------------------------------------------------------------------------------

MEANS = (0.95, 0.97, 0.98, 0.985)  # supervised AUROC at k = 5, 10, 20, 40


def _compare_report(judged=(5, 10)):
    def unsupervised(value):
        return {"macro_image_auroc": value, "macro_image_auroc_ci": [value - 0.01, value + 0.01]}

    supervised = {}
    for k, mean in zip((5, 10, 20, 40), MEANS, strict=True):
        unseen = {"unseen_defects_mean": 2.0, "categories_mean": 1.0, "judged": False}
        if k in judged:
            unseen = {
                "unseen_defects_mean": 116.67,
                "categories_mean": 9.0,
                "supervised": mean - 0.02,
                "reference": 0.975,
                "diff": mean - 0.02 - 0.975,
                "ci": [mean - 0.02 - 0.975 - 0.015, mean - 0.02 - 0.975 + 0.015],
                "judged": True,
            }
        # Keyed by str(k) and listed out of order, as a JSON object may be.
        supervised[str(k)] = {
            "k": k,
            "labels_with_validation": k + 20,
            "seeds": [0, 1, 2],
            "supervised_macro_image_auroc": mean,
            "supervised_per_seed": [mean - 0.002, mean, mean + 0.002],
            "supervised_ci": [mean - 0.005, mean + 0.005],
            "unseen_types": unseen,
        }
    return {
        "protocol": "test",
        "unsupervised": {"p0": unsupervised(0.89), "d-s": unsupervised(0.94), "dm": unsupervised(0.968)},
        "reference_unsupervised": "dm",
        "supervised": dict(reversed(supervised.items())),
    }


def test_label_curve_draws_the_supervised_curve_over_the_unsupervised_lines():
    fig = figures.fig_label_curve(_compare_report())
    assert len(fig.axes) == 2
    curve, unseen = fig.axes
    # Sorted by k whatever the order of the report, each k at its own tick.
    ticks = curve.get_xticks()
    assert [t.get_text() for t in curve.get_xticklabels()] == ["5\n(25)", "10\n(30)", "20\n(40)", "40\n(60)"]
    assert _x(fig, "supervised") == pytest.approx(ticks)
    assert _y(fig, "supervised") == pytest.approx([100 * m for m in MEANS])
    assert _whiskers(fig, "supervised:ci") == [
        pytest.approx((x, 100 * m - 0.5, 100 * m + 0.5)) for x, m in zip(ticks, MEANS, strict=True)
    ]
    # The single seeds of a k stand a little to the right of its mean.
    seeds = _one(fig, "supervised:seeds")
    xs, ys = np.asarray(seeds.get_xdata()), np.asarray(seeds.get_ydata())
    assert len(xs) == 12
    for x, m in zip(ticks, MEANS, strict=True):
        mine = (xs > x) & (xs < x + 0.3)
        assert sorted(ys[mine]) == pytest.approx([100 * m - 0.2, 100 * m, 100 * m + 0.2])
    # The unsupervised methods: line, interval band and the value at the right end.
    for name, value in (("p0", 89.0), ("d-s", 94.0), ("dm", 96.8)):
        assert _y(fig, f"line:{name}") == pytest.approx([value, value])
        band = _one(fig, f"band:{name}").get_extents().transformed(curve.transData.inverted())
        assert (band.y0, band.y1) == pytest.approx((value - 1.0, value + 1.0))
    assert _notes(curve) == {
        "89.0": pytest.approx((1.0, 89.0)),
        "94.0": pytest.approx((1.0, 94.0)),
        "96.8": pytest.approx((1.0, 96.8)),
    }
    legend = [t.get_text() for t in fig.legends[0].get_texts()]
    assert legend[1:] == ["PatchCore WRN-50", "PatchCore DINOv2 ViT-S", "Dinomaly"]
    assert legend[0].startswith("Supervised head: mean of 3 seeds")
    # Unseen defect types: the judged k only, supervised against the reference method (Dinomaly).
    uticks = unseen.get_xticks()
    assert [t.get_text() for t in unseen.get_xticklabels()] == ["k = 5\n117 defects", "k = 10\n117 defects"]
    assert _x(fig, "unseen:supervised") == pytest.approx(uticks)
    assert _x(fig, "unseen:reference") == pytest.approx(uticks)
    assert _y(fig, "unseen:supervised") == pytest.approx([93.0, 95.0])
    assert _y(fig, "unseen:reference") == pytest.approx([97.5, 97.5])
    reference = _one(fig, "unseen:reference")
    assert (reference.get_color(), reference.get_marker()) == figures.METHODS["dm"][1:]
    # The difference with its interval between the two markers of its k; the markers named at the first.
    assert _notes(unseen) == {
        "−4.5 pt\n[−6.0, −3.0]": pytest.approx((uticks[0], 95.25)),
        "−2.5 pt\n[−4.0, −1.0]": pytest.approx((uticks[1], 96.25)),
        "Dinomaly": pytest.approx((uticks[0], 97.5)),
        "Supervised head": pytest.approx((uticks[0], 93.0)),
    }
    _marks_inside(fig)


def test_label_curve_without_a_judged_subset_keeps_the_panel_and_says_so():
    fig = figures.fig_label_curve(_compare_report(judged=()))
    assert len(fig.axes) == 2
    assert not _artists(fig, "unseen:supervised")
    assert "too few such defects at every k" in _texts(fig)
    assert _y(fig, "supervised") == pytest.approx([95.0, 97.0, 98.0, 98.5])
    _marks_inside(fig)


# ---- M2AD ------------------------------------------------------------------------------------------------


def _m2ad_report(methods=("p0", "d-s")):
    out = {}
    for method, base in zip(methods, (0.02, 0.04), strict=False):
        conditions = []
        for i, name in enumerate(M2AD_CONDITIONS):
            fpr = min(1.0, base + 0.04 * i)
            conditions.append({"condition": name, "fpr": fpr, "fpr_ci": [max(0.0, fpr - 0.01), fpr]})
        recal = [
            {"n": n, "fpr": fpr, "fpr_ci": [fpr / 2, min(1.0, 1.5 * fpr)]}
            for n, fpr in ((0, 0.9 + base), (8, 0.0), (30, 0.03 + base))
        ]
        out[method] = {"inspectors": ["a_000", "a_120"], "conditions": conditions, "recal": recal}
    return {"methods": out}


def test_m2ad_conditions_are_grouped_and_every_rate_is_drawn():
    report = _m2ad_report()
    fig = figures.fig_m2ad_conditions(report)
    assert len(fig.axes) == 2
    ax, recal = fig.axes
    # The two methods stand side by side at each condition: marker and interval at the condition's tick.
    ticks, rticks = ax.get_xticks(), recal.get_xticks()
    sides, rsides = figures._offsets(2), figures._offsets(2, 0.38)
    for (method, run), dx, rdx in zip(report["methods"].items(), sides, rsides, strict=True):
        line = _one(fig, f"fpr:{method}")
        assert line.axes is ax
        assert _x(fig, f"fpr:{method}") == pytest.approx(ticks + dx)
        assert _y(fig, f"fpr:{method}") == pytest.approx([100 * row["fpr"] for row in run["conditions"]])
        assert _whiskers(fig, f"fpr:{method}:ci") == [
            pytest.approx((x + dx, 100 * row["fpr_ci"][0], 100 * row["fpr_ci"][1]))
            for x, row in zip(ticks, run["conditions"], strict=True)
        ]
        assert _one(fig, f"recal:{method}").axes is recal
        assert _x(fig, f"recal:{method}") == pytest.approx(rticks + rdx)
        assert _y(fig, f"recal:{method}") == pytest.approx([100 * row["fpr"] for row in run["recal"]])
        assert _whiskers(fig, f"recal:{method}:ci") == [
            pytest.approx((x + rdx, 100 * row["fpr_ci"][0], 100 * row["fpr_ci"][1]))
            for x, row in zip(rticks, run["recal"], strict=True)
        ]
    assert sides[0] < 0 < sides[1]  # WRN-50 on the left
    # The recalibrated rate written over its marker and interval.
    assert [(t.get_text(), *t.xy) for t in recal.texts] == [
        ("92.0", pytest.approx(rsides[0]), pytest.approx(100.0)),
        ("0.0", pytest.approx(1 + rsides[0]), pytest.approx(0.0)),
        ("5.0", pytest.approx(2 + rsides[0]), pytest.approx(7.5)),
        ("94.0", pytest.approx(rsides[1]), pytest.approx(100.0)),
        ("0.0", pytest.approx(1 + rsides[1]), pytest.approx(0.0)),
        ("7.0", pytest.approx(2 + rsides[1]), pytest.approx(10.5)),
    ]
    # The seven groups are set apart.
    steps = np.diff(ticks)
    assert np.isclose(steps, 1.0).sum() == 18 and (steps > 1.5).sum() == 6
    assert [int(i) for i in np.flatnonzero(steps > 1.5)] == [0, 3, 6, 9, 12, 15]
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels[0] == "S\nreference"
    assert labels[1:4] == ["0.9\n ", "0.8\nbrightness", "0.7\n "]
    assert labels[13:16] == ["90\n ", "70\nJPEG quality", "50\n "]
    assert labels[16] == "02\n " and labels[20] == "06\nreal lighting" and labels[24] == "10\n "
    assert {"gamma", "blur σ", "shift (px)"} <= {label.split("\n")[1] for label in labels}
    # The target line in both panels, named once; a rate of 0 has a place on the recalibration axis.
    targets = _artists(fig, "target")
    assert {line.axes for line in targets} == {ax, recal}
    assert all(list(line.get_ydata()) == pytest.approx([5.0, 5.0]) for line in targets)
    assert _notes(ax) == {"target 5%": pytest.approx((1.0, 5.0))}
    assert np.isfinite(recal.transData.transform((1.0, 0.0))).all()
    assert [t.get_text() for t in recal.get_xticklabels()] == ["0", "8", "30"]
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [
        "PatchCore WRN-50",
        "PatchCore DINOv2 ViT-S",
    ]
    _marks_inside(fig)


def test_m2ad_takes_the_methods_the_report_has_and_checks_their_conditions():
    fig = figures.fig_m2ad_conditions(_m2ad_report(methods=("p0",)), alpha=0.1)
    assert len(fig.axes) == 2 and not _artists(fig, "fpr:d-s")
    assert _x(fig, "fpr:p0") == pytest.approx(fig.axes[0].get_xticks())  # alone: on the tick
    assert all(list(line.get_ydata()) == pytest.approx([10.0, 10.0]) for line in _artists(fig, "target"))
    _marks_inside(fig)
    with pytest.raises(ValueError, match="none of the methods"):
        figures.fig_m2ad_conditions(_m2ad_report(methods=("dm",)))
    report = _m2ad_report()
    report["methods"]["d-s"]["conditions"].pop()
    with pytest.raises(ValueError, match="different conditions"):
        figures.fig_m2ad_conditions(report)


# ---- perturbation ----------------------------------------------------------------------------------------


def _perturb_run(root, method, seed, cal=60, protocol="test"):
    """A small `run_perturb` folder over all categories; `SHIFTED` adds 2 to every score."""
    names = condition_names()
    rng = np.random.default_rng(seed)
    run_dir = root / f"perturb-{method}-{protocol}"
    run_dir.mkdir(parents=True)
    for category in CATEGORIES:
        n_neg, n_pos = 30, 10
        clean = np.concatenate([rng.normal(0, 1, n_neg), rng.normal(3, 1, n_pos)])
        scores = np.tile(np.round(clean * 1024) / 1024, (len(names), 1))
        scores[names.index(SHIFTED)] += 2.0
        scores[names.index("blur-3"), n_neg:] -= 2.5
        np.savez(
            run_dir / f"{category}.npz",
            eval_images=np.array([f"{category}/{i}.JPG" for i in range(n_neg + n_pos)]),
            eval_labels=np.array([0] * n_neg + [1] * n_pos, dtype=np.int8),
            conditions=np.array(names),
            scores=scores.astype(np.float32),
            cal_score=rng.normal(0, 1, cal).astype(np.float32),
        )
    meta = {
        "method": method,
        "protocol": protocol,
        "conditions": names,
        "categories": [{"category": c} for c in CATEGORIES],
    }
    (run_dir / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    return run_dir


def _perturb_report(root, cal=None):
    """The report `analyze_perturb` writes, from synthetic runs; `cal`: calibration scores per method."""
    cal = {"p0": 60, "d-s": 60} if cal is None else cal
    run_dirs = {
        method: _perturb_run(root, method, seed=i, cal=n) for i, (method, n) in enumerate(cal.items())
    }
    return {"protocol": "test", **analyze_perturb.build_report(run_dirs, n_boot=40)}


def test_perturb_shows_accuracy_change_and_fixed_threshold_rate(tmp_path):
    report = _perturb_report(tmp_path)
    assert report["h9"] == {"method": "p0", "verdict": "지지", "conditions": [SHIFTED]}
    fig = figures.fig_perturb(report)
    assert len(fig.axes) == 2
    ax_auc, ax_fpr = fig.axes
    # Clean first, then the 15 conditions; the accuracy change has no clean marker.
    ticks = ax_fpr.get_xticks()
    assert len(ticks) == 16 and list(ax_auc.get_xticks()) == pytest.approx(ticks)
    for (method, table), dx in zip(report["tables"].items(), figures._offsets(2, 0.25), strict=True):
        assert [row["condition"] for row in table[1:]] == PERTURBED
        changes = _one(fig, f"d_auroc:{method}")
        assert changes.axes is ax_auc
        assert _x(fig, f"d_auroc:{method}") == pytest.approx(ticks[1:] + dx)
        assert _y(fig, f"d_auroc:{method}") == pytest.approx([100 * row["d_auroc"] for row in table[1:]])
        assert _whiskers(fig, f"d_auroc:{method}:ci") == [
            pytest.approx((x + dx, 100 * row["d_auroc_ci"][0], 100 * row["d_auroc_ci"][1]))
            for x, row in zip(ticks[1:], table[1:], strict=True)
        ]
        rates = _one(fig, f"fpr:{method}")
        assert rates.axes is ax_fpr
        assert _x(fig, f"fpr:{method}") == pytest.approx(ticks + dx)  # with clean
        assert _y(fig, f"fpr:{method}") == pytest.approx([100 * row["fpr"] for row in table])
        assert _whiskers(fig, f"fpr:{method}:ci") == [
            pytest.approx((x + dx, 100 * row["fpr_ci"][0], 100 * row["fpr_ci"][1]))
            for x, row in zip(ticks, table, strict=True)
        ]
    # The shifted condition keeps its accuracy and overruns the threshold; the figure shades it in both rows.
    k = PERTURBED.index(SHIFTED)
    assert _y(fig, "d_auroc:p0")[k] == pytest.approx(0.0) and _y(fig, "fpr:p0")[k + 1] > 50.0
    shaded = _artists(fig, f"h9:{SHIFTED}")
    assert {patch.axes for patch in shaded} == {ax_auc, ax_fpr}
    assert all(p.get_x() + p.get_width() / 2 == pytest.approx(ticks[k + 1]) for p in shaded)
    assert not _artists(fig, "h9:blur-3")
    # The target line belongs to the rate panel only.
    (target,) = _artists(fig, "target")
    assert target.axes is ax_fpr and list(target.get_ydata()) == pytest.approx([5.0, 5.0])
    assert _notes(ax_fpr) == {"target 5%": pytest.approx((1.0, 5.0))}
    labels = [t.get_text() for t in ax_fpr.get_xticklabels()]
    assert labels[0] == "\nclean" and labels[1:4] == ["0.9\n ", "0.8\nbrightness", "0.7\n "]
    assert all(t.get_text() == "" for t in ax_auc.get_xticklabels())
    legend = [t.get_text() for t in fig.legends[0].get_texts()]
    assert legend[:2] == ["PatchCore WRN-50", "PatchCore DINOv2 ViT-S"]
    assert legend[2] == "PatchCore WRN-50: AUROC falls by less than 1 point, false alarms at least 10%"
    _marks_inside(fig)


def test_perturb_method_without_calibration_scores_has_no_rate_markers(tmp_path):
    report = _perturb_report(tmp_path, cal={"p0": 60, "dm": 0})
    assert report["tables"]["dm"][0]["fpr"] is None
    fig = figures.fig_perturb(report)
    ticks = fig.axes[1].get_xticks()
    assert _x(fig, "d_auroc:dm") == pytest.approx(ticks[1:] + figures._offsets(2, 0.25)[1])
    assert not _artists(fig, "fpr:dm")
    assert _x(fig, "fpr:p0") == pytest.approx(ticks + figures._offsets(2, 0.25)[0])
    _marks_inside(fig)
    # A partial run is not judged: nothing is shaded and the legend has the methods only.
    report["h9"] = None
    fig = figures.fig_perturb(report)
    assert not _artists(fig, f"h9:{SHIFTED}")
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["PatchCore WRN-50", "Dinomaly"]


# ---- grid ------------------------------------------------------------------------------------------------


def _grid_inputs(protocol="dev", budget_ms=analyze_grid.CPU_BUDGET_MS):
    """Grid rows and a latency file put together by `analyze_grid`, as its `main` does."""
    rows, latency = [], {}
    for setting, ms, auroc in (("wrn50-256", 60.0, 0.90), ("dinov2_vits14-392", 400.0, 0.95)):
        for ratio, slower, drop in ((0.1, 3.0, 0.0), (0.01, 1.0, 0.004)):
            key = f"{setting}-{ratio_name(ratio)}"
            rows.append(
                {
                    "key": key,
                    "setting": setting,
                    "ratio": ratio,
                    "bank_rows_mean": 1e5 * ratio,
                    "macro_image_auroc": auroc - drop,
                }
            )
            latency[key] = {"cpu": "some cpu", "resize_ms": 2.0, "cpu_fp32": {"total_ms": ms * slower}}
            if setting == "wrn50-256":
                latency[key]["cpu_int8"] = {"total_ms": ms * slower / 2}
    # A row timed on the GPU only has no CPU latency and no marker.
    gpu_only = f"wrn50-384-{ratio_name(0.1)}"
    rows.append(
        {
            "key": gpu_only,
            "setting": "wrn50-384",
            "ratio": 0.1,
            "bank_rows_mean": 2e4,
            "macro_image_auroc": 0.93,
        }
    )
    latency[gpu_only] = {"gpu_fp16": {"total_ms": 9.0}}
    # The file the grid report was made from stays apart from the one returned (as on disk).
    analyze_grid.merge_latency(rows, copy.deepcopy(latency))
    grid = {"protocol": protocol, "alpha": 0.05, "rows": rows}
    if protocol == "dev":
        grid["serving"] = analyze_grid.choose_serving(rows, budget_ms=budget_ms)
    return grid, latency


SLOW, FAST = f"wrn50-256-{ratio_name(0.1)}", f"wrn50-256-{ratio_name(0.01)}"


def test_grid_puts_every_timed_configuration_at_its_latency_and_accuracy():
    grid, latency = _grid_inputs()
    assert grid["serving"]["key"] == SLOW  # 182 ms: the best within 200 ms
    fig = figures.fig_grid(grid, latency)
    assert len(fig.axes) == 1
    (ax,) = fig.axes
    assert ax.get_xscale() == "log"
    for row in grid["rows"][:4]:
        point = _one(fig, f"point:{row['key']}:fp32")
        assert list(point.get_xdata()) == pytest.approx([row["cpu_ms"]])
        assert list(point.get_ydata()) == pytest.approx([100 * row["macro_image_auroc"]])
    assert _x(fig, f"point:{FAST}:fp32") == pytest.approx([62.0])
    # INT8: its own latency at the accuracy of the row.
    assert _x(fig, f"point:{FAST}:int8") == pytest.approx([32.0])
    assert _y(fig, f"point:{FAST}:int8") == pytest.approx([89.6])
    assert not _artists(fig, f"point:dinov2_vits14-392-{ratio_name(0.1)}:int8")
    assert not [a for a in fig.findobj() if str(a.get_gid()).startswith("point:wrn50-384")]
    # Marker size is the coreset ratio, the same as in the key.
    key = fig.legends[0].legend_handles
    big, small = (_one(fig, f"point:{k}:fp32").get_markersize() for k in (SLOW, FAST))
    assert big > small and (big, small) == pytest.approx((key[0].get_markersize(), key[1].get_markersize()))
    # The budget line and the ring around the serving configuration.
    assert _x(fig, "budget") == pytest.approx([200.0, 200.0])
    assert _x(fig, "serving") == pytest.approx([182.0]) and _y(fig, "serving") == pytest.approx([90.0])
    notes = _notes(ax)
    assert notes["serving: 182 ms, AUROC 90.0"] == pytest.approx((182.0, 90.0))
    assert notes["budget 200 ms"] == pytest.approx((200.0, 1.0))
    # Each setting named next to its slowest FP32 marker.
    assert notes["WRN-50 256 px"] == pytest.approx((182.0, 90.0))
    assert notes["DINOv2 ViT-S 392 px"] == pytest.approx((1202.0, 95.0))
    legend = [t.get_text() for t in fig.legends[0].get_texts()]
    assert legend[:3] == ["coreset 10%", "coreset 1%", "FP32"] and legend[3].startswith("INT8")
    _marks_inside(fig)
    # Without the latency file the entries merged into the rows give the same picture.
    merged = figures.fig_grid(grid)
    assert _x(merged, f"point:{FAST}:int8") == pytest.approx([32.0])


def test_grid_takes_the_budget_and_the_choice_of_the_report():
    grid, latency = _grid_inputs(budget_ms=150.0)
    assert grid["serving"]["key"] == FAST  # 182 ms is over 150 ms
    fig = figures.fig_grid(grid, latency)
    assert _x(fig, "budget") == pytest.approx([150.0, 150.0])
    assert _x(fig, "serving") == pytest.approx([62.0])
    assert "budget 150 ms" in _texts(fig) and "serving: 62 ms, AUROC 89.6" in _texts(fig)
    _marks_inside(fig)


def test_grid_draws_the_timings_the_serving_choice_was_made_on():
    grid, latency = _grid_inputs()
    # bench ran again after analyze_grid: the file has other timings than the report.
    newer = copy.deepcopy(latency)
    newer[SLOW]["cpu_fp32"]["total_ms"] = 398.0
    newer[FAST]["cpu_int8"]["total_ms"] = 97.0
    fig = figures.fig_grid(grid, newer)
    assert _x(fig, f"point:{SLOW}:fp32") == pytest.approx([182.0])
    assert _x(fig, f"point:{FAST}:int8") == pytest.approx([32.0])  # the report has INT8 timings too
    assert _x(fig, "serving") == pytest.approx([182.0]) and "serving: 182 ms, AUROC 90.0" in _texts(fig)
    # The file fills in what the report has no timing for: a whole row, or a precision timed later.
    bare = {
        **grid,
        "rows": [{k: v for k, v in r.items() if k not in ("latency", "cpu_ms")} for r in grid["rows"]],
    }
    for row in grid["rows"]:
        row.get("latency", {}).pop("cpu_int8", None)
    newer[FAST]["cpu_int8"]["total_with_resize_ms"] = 35.0  # bench writes the sum with its own resize
    assert not _artists(figures.fig_grid(grid), f"point:{FAST}:int8")
    for report in (grid, bare):
        fig = figures.fig_grid(report, newer)
        assert _x(fig, f"point:{FAST}:fp32") == pytest.approx([62.0])
        assert _x(fig, f"point:{FAST}:int8") == pytest.approx([35.0])
    assert _x(fig, f"point:{SLOW}:fp32") == pytest.approx([400.0])  # no registered cpu_ms in `bare`


def test_grid_fp32_marker_is_the_registered_cpu_latency():
    # bench timed FP32 with a 2 ms resize, then INT8 alone with a 10 ms resize, which replaced `resize_ms`;
    # analyze_grid registered 10 + 180 ms for FP32, and the serving choice was made on that.
    _, latency = _grid_inputs()
    latency[SLOW]["resize_ms"] = 10.0
    latency[SLOW]["cpu_fp32"]["total_with_resize_ms"] = 182.0
    latency[SLOW]["cpu_int8"]["total_with_resize_ms"] = 100.0
    rows = [{k: v for k, v in r.items() if k not in ("latency", "cpu_ms")} for r in _grid_inputs()[0]["rows"]]
    analyze_grid.merge_latency(rows, latency)
    grid = {"protocol": "dev", "rows": rows, "serving": analyze_grid.choose_serving(rows)}
    assert grid["serving"]["key"] == SLOW and grid["serving"]["cpu_ms"] == pytest.approx(190.0)
    fig = figures.fig_grid(grid, latency)
    assert _x(fig, f"point:{SLOW}:fp32") == pytest.approx([190.0]) and _x(fig, "serving") == pytest.approx(
        [190.0]
    )
    assert _x(fig, f"point:{SLOW}:int8") == pytest.approx([100.0])
    assert "serving: 190 ms, AUROC 90.0" in _texts(fig)


def test_grid_names_a_setting_timed_in_int8_only():
    grid, latency = _grid_inputs(protocol="test")
    for row in grid["rows"]:
        row.pop("latency", None)
        row.pop("cpu_ms", None)
    for key in (SLOW, FAST):
        del latency[key]["cpu_fp32"]
    fig = figures.fig_grid(grid, latency)
    assert not _artists(fig, f"point:{SLOW}:fp32")
    assert _x(fig, f"point:{SLOW}:int8") == pytest.approx([92.0])
    assert _notes(fig.axes[0])["WRN-50 256 px"] == pytest.approx((92.0, 90.0))
    _marks_inside(fig)
    # Nothing but INT8: the key still says what the hollow markers are.
    for key in latency:
        latency[key].pop("cpu_fp32", None)
    fig = figures.fig_grid(grid, latency)
    assert [t.get_text() for t in fig.legends[0].get_texts()][-1].startswith("INT8")


def test_grid_without_a_serving_choice_or_without_cpu_timings():
    grid, latency = _grid_inputs(protocol="test")
    fig = figures.fig_grid(grid, latency)
    assert not _artists(fig, "serving")
    assert _x(fig, "budget") == pytest.approx([analyze_grid.CPU_BUDGET_MS] * 2)
    for row in grid["rows"]:
        row.pop("latency", None)
    with pytest.raises(ValueError, match="CPU latency"):
        figures.fig_grid(grid, {})


# ---- examples --------------------------------------------------------------------------------------------


def _example_rows():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (64, 48, 3), dtype=np.uint8)
    mask = np.zeros((64, 48), dtype=bool)
    mask[20:32, 12:28] = True
    scores = np.zeros((16, 12))  # one score per 4 x 4 pixels
    scores[5:8, 3:7] = 2.0  # the cells over the mask
    return [
        {"image": image, "mask": mask, "map": scores, "title": "defect", "threshold": 1.0},
        {"image": image, "mask": None, "map": 0.2 * scores, "title": "normal", "threshold": 1.0},
        {"image": image, "mask": mask, "map": scores + 1.0, "title": "no threshold", "threshold": None},
        {
            "image": image,
            "mask": mask,
            "map": scores + 3.0,
            "title": "all flagged",
            "threshold": np.float32(1),
        },
    ]


def _bounds(contours):
    vertices = np.concatenate([p.vertices for p in contours.get_paths()])
    return (*vertices.min(axis=0), *vertices.max(axis=0))


def test_examples_draw_three_panels_and_one_colour_scale_per_row():
    rows = _example_rows()
    fig = figures.fig_examples(rows)
    panels = [ax for ax in fig.axes if ax.get_label() != "colorbar"]
    bars = [ax for ax in fig.axes if ax.get_label() == "colorbar"]
    assert len(panels) == 12 and len(bars) == 4
    assert [panels[3 * i].get_title(loc="left") for i in range(4)] == [row["title"] for row in rows]
    assert [ax.get_title(loc="left") for ax in panels[1:3]] == ["ground-truth outline", "anomaly map"]
    for i, row in enumerate(rows):
        for ax in panels[3 * i : 3 * i + 3]:
            assert np.array_equal(ax.images[0].get_array(), row["image"])
            assert ax.get_xlim() == (-0.5, 47.5) and ax.get_ylim() == (63.5, -0.5)  # image coordinates
        # The score map lies over the whole image although it is coarser.
        heat = panels[3 * i + 2].images[1]
        assert tuple(heat.get_extent()) == (-0.5, 47.5, 63.5, -0.5)
        assert np.array_equal(heat.get_array(), row["map"])
        assert len(panels[3 * i].images) == 1 and len(panels[3 * i + 1].images) == 1
    # The scale of a row spans its map and its threshold.
    clims = [panels[3 * i + 2].images[1].get_clim() for i in range(4)]
    assert clims == [(0.0, 2.0), (0.0, 1.0), (1.0, 3.0), (1.0, 5.0)]
    marks = _artists(fig, "threshold")
    assert [line.axes for line in marks] == [bars[0], bars[1], bars[3]]
    assert all(list(line.get_ydata()) == [1.0, 1.0] for line in marks)
    # Low scores stay transparent; a map above its threshold everywhere is coloured everywhere.
    for i, peak in ((0, 0.75), (2, 0.75), (3, 0.75)):
        alpha = panels[3 * i + 2].images[1].get_alpha()
        assert alpha.max() == pytest.approx(peak)
        assert np.all(alpha[rows[i]["map"] == rows[i]["map"].min()] == (0.375 if i == 3 else 0.0))
    # The mask outline only where there is a mask, along the mask; the normal row says so instead.
    assert [len(panels[3 * i + 1].collections) for i in range(4)] == [1, 0, 1, 1]
    assert _bounds(panels[1].collections[0]) == pytest.approx((11.5, 19.5, 27.5, 31.5))
    assert "normal image: no defect mask" in [t.get_text() for t in panels[4].texts]
    # The threshold contour only where the map crosses the threshold, in image coordinates: the flagged
    # cells are the ones over the mask.
    assert [len(panels[3 * i + 2].collections) for i in range(4)] == [1, 0, 0, 0]
    contour = panels[2].collections[0]
    assert list(contour.levels) == [1.0]
    assert _bounds(contour) == pytest.approx((11.5, 19.5, 27.5, 31.5))
    _marks_inside(fig)
    with pytest.raises(ValueError, match="at least one"):
        figures.fig_examples([])


def test_examples_refuse_maps_they_cannot_draw_and_masks_of_another_size():
    (row, *_) = _example_rows()
    for bad in (np.nan, np.inf):
        scores = row["map"].copy()
        scores[0, 0] = bad
        with pytest.raises(ValueError, match="finite"):
            figures.fig_examples([{**row, "map": scores}])
    with pytest.raises(ValueError, match="2-d"):
        figures.fig_examples([{**row, "map": np.zeros((2, 2, 2))}])
    with pytest.raises(ValueError, match="size of its image"):
        figures.fig_examples([{**row, "mask": np.zeros((32, 24), dtype=bool)}])
    # A map one score high crosses the threshold but gives no contour.
    fig = figures.fig_examples([{**row, "map": np.array([[0.0, 0.5, 2.0, 0.1]])}])
    assert not fig.axes[2].collections and len(_artists(fig, "threshold")) == 1


# ---- committed reports, determinism, CLI -----------------------------------------------------------------


@pytest.mark.parametrize("name", ["calibration", "label_curve", "m2ad_conditions"])
def test_committed_reports_give_figures(name, tmp_path):
    inputs, build = figures.FIGURES[name]
    files = [paths.REPORTS / rel for rel in inputs]
    if not all(f.exists() for f in files):
        pytest.skip(f"no {files}")
    loaded = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    fig = build(*loaded)
    path = figures.save(fig, tmp_path / f"{name}.png")
    with Image.open(path) as image:
        assert image.format == "PNG" and image.width == 1800 and 500 < image.height < 1200
    _marks_inside(fig)


def test_the_same_report_gives_identical_bytes(tmp_path):
    report = _m2ad_report()
    fig = figures.fig_m2ad_conditions(report)
    first = _png(fig, tmp_path / "a.png")
    assert first[:8] == b"\x89PNG\r\n\x1a\n"
    assert _png(fig, tmp_path / "b.png") == first  # saved twice
    assert _png(figures.fig_m2ad_conditions(report), tmp_path / "c.png") == first  # built twice
    with Image.open(tmp_path / "a.png") as image:
        assert "Software" not in image.info  # no matplotlib version in the file
    rows = _example_rows()
    assert _png(figures.fig_examples(rows), tmp_path / "d.png") == _png(
        figures.fig_examples(rows), tmp_path / "e.png"
    )


def test_building_a_figure_leaves_the_global_settings_alone():
    import matplotlib

    before = dict(matplotlib.rcParams)
    figures.fig_calibration(_calibration_report())
    assert dict(matplotlib.rcParams) == before


def test_the_users_settings_do_not_reach_the_figures(tmp_path):
    import matplotlib

    # What a matplotlibrc file or the calling code may have set.
    theirs = {
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.5,
        "savefig.transparent": True,
        "lines.linewidth": 4.0,
        "lines.markersize": 20.0,
        "ytick.major.size": 12.0,
        "axes.xmargin": 0.3,
        "image.origin": "lower",
        "image.interpolation": "bicubic",
        "text.antialiased": False,
    }
    report, rows = _calibration_report(), _example_rows()
    plain = [_png(figures.fig_calibration(report), tmp_path / "a.png")]
    plain.append(_png(figures.fig_examples(rows), tmp_path / "b.png"))
    with matplotlib.rc_context(theirs):
        built = [figures.fig_calibration(report), figures.fig_examples(rows)]
        assert [_png(fig, tmp_path / f"c{i}.png") for i, fig in enumerate(built)] == plain
    built = [figures.fig_calibration(report), figures.fig_examples(rows)]
    with matplotlib.rc_context(theirs):
        assert [_png(fig, tmp_path / f"d{i}.png") for i, fig in enumerate(built)] == plain


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_cli_writes_what_it_has_inputs_for_and_names_what_it_skips(tmp_path, capsys):
    reports, out = tmp_path / "reports", tmp_path / "out"
    _write_json(reports / "stage1" / "p0-test.json", _calibration_report())
    _write_json(reports / "stage3" / "m2ad.json", _m2ad_report())
    grid, _ = _grid_inputs()
    _write_json(reports / "stage4" / "grid-dev.json", grid)  # the latency file is missing
    figures.main(["--reports", str(reports), "--out", str(out)])
    assert sorted(p.name for p in out.iterdir()) == ["calibration.png", "m2ad_conditions.png"]
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        f"wrote {out / 'calibration.png'}",
        f"skipped label_curve: no {reports / 'stage2' / 'compare-test.json'}",
        f"wrote {out / 'm2ad_conditions.png'}",
        f"skipped perturb: no {reports / 'stage3' / 'perturb-test.json'}",
        f"skipped grid: no {reports / 'stage4' / 'latency.json'}",
    ]
    # A second run overwrites the files with the same bytes.
    first = (out / "calibration.png").read_bytes()
    figures.main(["--reports", str(reports), "--out", str(out), "--only", "calibration", "perturb"])
    assert (out / "calibration.png").read_bytes() == first
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2 and lines[0].startswith("wrote ") and lines[1].startswith("skipped perturb: no ")


def test_cli_writes_every_figure_when_all_inputs_are_there(tmp_path, capsys):
    reports, out = tmp_path / "reports", tmp_path / "deep" / "out"
    grid, latency = _grid_inputs()
    _write_json(reports / "stage1" / "p0-test.json", _calibration_report())
    _write_json(reports / "stage2" / "compare-test.json", _compare_report())
    _write_json(reports / "stage3" / "m2ad.json", _m2ad_report())
    _write_json(reports / "stage3" / "perturb-test.json", _perturb_report(tmp_path / "runs"))
    _write_json(reports / "stage4" / "grid-dev.json", grid)
    _write_json(reports / "stage4" / "latency.json", latency)
    figures.main(["--reports", str(reports), "--out", str(out)])
    assert sorted(p.stem for p in out.iterdir()) == sorted(figures.FIGURES)
    assert "skipped" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        figures.main(["--reports", str(reports), "--out", str(out), "--only", "examples"])


def test_cli_draws_the_other_figures_when_a_report_is_broken(tmp_path, capsys):
    reports, out = tmp_path / "reports", tmp_path / "out"
    old = _calibration_report()
    del old["finite_sample"]  # a report of an older schema
    _write_json(reports / "stage1" / "p0-test.json", old)
    _write_json(reports / "stage2" / "compare-test.json", _compare_report())
    cut = reports / "stage3" / "m2ad.json"
    _write_json(cut, _m2ad_report())
    cut.write_text(cut.read_text(encoding="utf-8")[:200], encoding="utf-8")  # a write that was interrupted
    with pytest.raises(SystemExit, match="^figures not written: calibration, m2ad_conditions$"):
        figures.main(["--reports", str(reports), "--out", str(out)])
    assert sorted(p.name for p in out.iterdir()) == ["label_curve.png"]
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        f"wrote {out / 'label_curve.png'}",
        f"skipped perturb: no {reports / 'stage3' / 'perturb-test.json'}",
        f"skipped grid: no {reports / 'stage4' / 'grid-dev.json'}, {reports / 'stage4' / 'latency.json'}",
    ]
    errors = captured.err.splitlines()
    assert len(errors) == 2
    assert (
        errors[0] == f"failed calibration ({reports / 'stage1' / 'p0-test.json'}): KeyError: 'finite_sample'"
    )
    assert errors[1].startswith(f"failed m2ad_conditions ({cut}): JSONDecodeError: ")


def test_cli_prints_paths_the_console_code_page_cannot_encode(tmp_path, monkeypatch):
    reports, out = tmp_path / "reports", tmp_path / "figé中"
    _write_json(reports / "stage1" / "p0-test.json", _calibration_report())
    _write_json(reports / "stage3" / "m2ad.json", _m2ad_report())
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp949")  # stdout redirected on a Korean Windows
    monkeypatch.setattr(sys, "stdout", console)
    figures.main(["--reports", str(reports), "--out", str(out), "--only", "calibration", "m2ad_conditions"])
    console.flush()
    assert sorted(p.name for p in out.iterdir()) == ["calibration.png", "m2ad_conditions.png"]
    assert console.buffer.getvalue().decode("utf-8").splitlines() == [
        f"wrote {out / 'calibration.png'}",
        f"wrote {out / 'm2ad_conditions.png'}",
    ]


def test_the_module_imports_without_matplotlib():
    # `None` in sys.modules makes every `import matplotlib` fail, as where the group is not installed.
    code = (
        "import sys; sys.modules['matplotlib'] = None; "
        "from defect_inspect import figures; print(len(figures.FIGURES))"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "5"
