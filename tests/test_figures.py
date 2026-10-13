"""figures on synthetic reports and on the committed ones (needs the `figures` dependency group)."""

# [파일 역할] README 그림을 그리는 `defect_inspect.figures`를 검증하는 테스트 묶음이다. 리포트 다섯 개로
# 그리는 그림(calibration, label_curve, m2ad_conditions, perturb, grid), 평가 실행 하나에서 뽑는 실패 사례
# 패널(examples), 그리고 CLI를 다룬다.
# [왜 필요한가] README의 그림은 측정 결과를 사람에게 보여 주는 마지막 단계(5단계)라, 그림의 점·구간이 리포트
# 수치와 다르면 문서가 틀린 말을 하게 된다. 그래서 (1) 각 점의 좌표가 리포트 값의 % 환산과 정확히 같은지
# artist의 gid로 찾아 확인하고, (2) 표시가 축 밖으로 잘리거나 글자가 겹치거나 그림 밖으로 나가지 않는지 픽셀
# 단위로 재며, (3) 같은 리포트면 PNG 바이트가 같고 사용자의 matplotlib 설정이 결과를 바꾸지 않는지 본다. 실패
# 사례 패널은 봉인 테스트 이미지를 읽으므로, --allow-test와 장부 기록 순서, dev 실행에 섞인 봉인 이미지 거부를
# 따로 시험한다.
# [설계] 합성 리포트와 작은 가짜 실행·캐시를 tmp_path에 만들고, 저장소에 커밋된 실제 리포트가 있으면
# 그것으로도 그려 본다. matplotlib는 figures 의존성 그룹에만 있어 없으면 파일 전체를 건너뛴다.

import copy
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# matplotlib(figures 그룹)가 없는 환경(CI 기본 설치)에서는 이 파일 전체를 건너뛴다. 그래서 아래 import에 E402
# 예외 표시가 붙는다.
pytest.importorskip("matplotlib")

from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

# analyze_grid·analyze_perturb는 실제 분석 코드로 그림의 입력 리포트를 만들려고, cache·ledger·paths·splits는
# 실패 사례 패널의 가짜 캐시·장부·매니페스트를 꾸미려고, compare는 점수 파일 읽기를 엿보려고 가져온다.
from defect_inspect import analyze_grid, analyze_perturb, cache, compare, figures, ledger, paths  # noqa: E402
from defect_inspect.conditions import condition_names  # noqa: E402
from defect_inspect.run_grid import ratio_name  # noqa: E402
from defect_inspect.splits import ManifestRow, SealedTestError, read_manifest, write_manifest  # noqa: E402
from defect_inspect.visa import CATEGORIES  # noqa: E402

# PERTURBED: 3단계 합성 교란 15개 조건. M2AD_CONDITIONS: M2AD 리포트의 조건 순서(기준 조명 S, 합성 교란 P:*,
# 실제 조명 R:02~R:10). SHIFTED: 모든 점수를 올려 순위는 그대로이고 고정 임계값만 넘치게 만드는 조건(H9가 찾는
# 고장).
PERTURBED = condition_names(include_clean=False)
M2AD_CONDITIONS = ["S", *[f"P:{name}" for name in PERTURBED], *[f"R:{i:02d}" for i in range(2, 11)]]
SHIFTED = "brightness-2"  # every score moves up: the ranking stays, the fixed thresholds are overrun
# The note under a condition axis with WRN-50 (256 px input) and DINOv2 ViT-S (448 px input).
SCALED_NOTE = (
    "* blur σ and shift in pixels of a 256 px image, the input of PatchCore WRN-50; "
    "other input sizes got them scaled:\nPatchCore DINOv2 ViT-S (448 px): σ ×1.75, shift 2, 4, 7 px"
)


# 그림 안에서 gid(그림 코드가 붙인 이름표)가 같은 artist를 모두 찾는다. 이하 _one·_x·_y·_whiskers는 이
# 이름표로 점·선·구간을 하나 집어 좌표를 꺼내는 도우미다.
def _artists(fig, gid):
    return fig.findobj(lambda artist: artist.get_gid() == gid)


# gid가 정확히 하나여야 한다(같은 이름이 두 번 그려지는 실수도 잡는다).
def _one(fig, gid):
    found = _artists(fig, gid)
    assert len(found) == 1, f"{len(found)} artists with gid {gid!r}"
    return found[0]


# 그 artist의 x 데이터.
def _x(fig, gid):
    return list(_one(fig, gid).get_xdata())


# 그 artist의 y 데이터.
def _y(fig, gid):
    return list(_one(fig, gid).get_ydata())


# 구간 세로선(vlines) 묶음에서 선마다 (x, 아래, 위)를 꺼낸다. 리포트의 구간과 그대로 비교한다.
def _whiskers(fig, gid):
    """(x, low, high) of every interval line drawn under `gid`."""
    return [(seg[0, 0], seg[0, 1], seg[1, 1]) for seg in _one(fig, gid).get_segments()]


# 축 안 주석 글자 → 그 글자가 가리키는 좌표. 값 표시가 맞는 점 위에 붙었는지 본다.
def _notes(ax):
    """Text -> anchor point of the annotations of an axes."""
    return {t.get_text(): tuple(t.xy) for t in ax.texts if hasattr(t, "xy")}


# 그림 안 모든 글자.
def _texts(fig):
    return [t.get_text() for t in fig.findobj(lambda artist: hasattr(artist, "get_text"))]


# figures.save로 PNG를 쓰고 그 바이트를 돌려준다(바이트 동일성 비교용).
def _png(fig, path):
    return figures.save(fig, path).read_bytes()


# 선·마커 전체·구간·띠·막대가 모두 자기 축 안에 있는지(잘려 보이지 않는지) 픽셀 좌표로 확인한다. 축 범위가
# 표시를 잘라도 artist 데이터는 그대로라 데이터로는 알 수 없기 때문이다. 마커는 반지름만큼 여유를 둔다.
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


# 그림 스타일 안에서 한 번 그려(글자 크기가 정해지도록) 렌더러를 돌려준다.
def _drawn(fig):
    with figures._style():
        fig.canvas.draw()
    return fig.canvas.get_renderer()


# x 눈금 글자가 이웃과 최소한 공백 한 칸만큼 떨어져 있는지 줄마다 잰다(강도 줄끼리, 묶음 이름 줄끼리). 조건이
# 25개인 M2AD 축처럼 칸이 좁은 그림에서 글자가 겹치는 회귀를 막는다.
def _tick_labels_apart(fig):
    """On every axes, neighbouring x tick labels keep at least a space between them, line by line.

    A tick label of two lines ("1.25\\ngamma") is centred on its tick line by line, so the lines are
    measured one by one: a strength against the next strength, a group name against the next group name.
    """
    renderer = _drawn(fig)
    for ax in fig.axes:
        labels = [t for t in ax.get_xticklabels() if t.get_visible() and t.get_text().strip()]
        if not labels:
            continue
        font = labels[0].get_fontproperties()

        def width(text, font=font):
            return renderer.get_text_width_height_descent(text, font, ismath=False)[0]

        space = width("a b") - width("ab")
        rows = {}
        for label in labels:
            box = label.get_window_extent(renderer)
            centre = (box.x0 + box.x1) / 2
            for i, line in enumerate(label.get_text().split("\n")):
                if line.strip():
                    rows.setdefault(i, []).append((centre - width(line) / 2, centre + width(line) / 2, line))
        for row in rows.values():
            row.sort()
            for (_, right, left_text), (left, _, right_text) in zip(row, row[1:], strict=False):
                gap = left - right
                assert gap >= space, (
                    f"{left_text!r} and {right_text!r} are {gap:.1f} px apart (a space: {space:.1f})"
                )


# 같은 조건 위에 나란히 놓인 방법들의 마커가 흰 테두리까지 포함해 서로 겹치지 않는지 확인한다.
def _markers_apart(fig, gids):
    """At every x position, the markers of neighbouring series stand side by side without overlapping."""
    _drawn(fig)
    lines = [_one(fig, gid) for gid in gids]
    for a, b in zip(lines, lines[1:], strict=False):
        xa = a.axes.transData.transform(a.get_xydata())[:, 0]
        xb = b.axes.transData.transform(b.get_xydata())[:, 0]
        # Half the width of each marker, with the white edge drawn around it.
        room = sum(line.get_markersize() + line.get_markeredgewidth() for line in (a, b)) / 2 * fig.dpi / 72
        assert np.min(np.abs(xb - xa)) >= room, f"{a.get_gid()} and {b.get_gid()} overlap"


# 제목·축 이름·주석·범례 글자가 모두 그림 경계 안에 있는지 확인한다(PNG에서 잘려 나가지 않게).
def _texts_inside(fig):
    """Every text (titles, labels, notes, legend entries) lies inside the figure."""
    renderer = _drawn(fig)
    box = fig.bbox
    for text in fig.findobj(
        lambda artist: hasattr(artist, "get_text") and hasattr(artist, "get_window_extent")
    ):
        if not text.get_visible() or not text.get_text().strip():
            continue
        ext = text.get_window_extent(renderer)
        inside = box.x0 <= ext.x0 and ext.x1 <= box.x1 and box.y0 <= ext.y0 and ext.y1 <= box.y1
        assert inside, f"{text.get_text()!r} sticks out of the figure: {ext}"


# ---- calibration -----------------------------------------------------------------------------------------


# 1단계 리포트(analyze.build_report) 모양의 합성 데이터. 목표 5%와 1%마다 세 보정 방식(재대입, 홀드아웃, 교차
# 적합)의 실제 오검출률·구간·보정 점수 수·이론 구간과, 보정 표본 수 n에 따른 관측 곡선과 기준 모의
# 구간(finite_sample)을 담는다.
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


# 확인: calibration 그림의 세 패널이 리포트 값을 %로 그대로 그린다. 왼쪽은 세 방식 막대(0~100% 축), 가운데는
# 홀드아웃·교차 적합을 확대해 홀드아웃의 이론 구간(theory_band)을 옆에 두고, 오른쪽은 n별 관측 평균과 교환
# 가능한 점수 모의의 평균·구간 띠다. 막대 위 부트스트랩 구간, 값 표시, 패널마다 목표선, 축 이름(막대의 n은
# 범주 합계, 오른쪽 n은 범주당), 범례에서 '부트스트랩 구간'과 '이론 구간'이 구별되는지까지 본다.
# 왜: README의 '재대입 86%, 홀드아웃 5.6%(이론 구간 안), 교차 적합 4.0%'를 보여 주는 그림이다. 두 종류의
# 구간을 혼동하게 그리면 H2 판정을 잘못 읽게 된다.
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
    # The whiskers of the bars are named next to the theory range, so neither is taken for the other; the
    # n under the bars is a total over the categories, the n of the right panel is per category.
    assert [t.get_text() for t in zoom.get_legend().get_texts()] == [
        "on the bars: 95% bootstrap interval",
        "hold-out: 95% range expected in theory",
    ]
    whisker, theory = zoom.get_legend().legend_handles
    assert (whisker.get_marker(), whisker.get_color()) == ("|", figures.INK)
    assert theory.get_color() == _one(fig, "theory:holdout").get_color()
    assert full.get_xlabel() == zoom.get_xlabel() == "n: calibration scores of all categories together"
    assert by_n.get_xlabel() == "calibration images per category (n)"
    _marks_inside(fig)
    _tick_labels_apart(fig)
    _texts_inside(fig)


# 확인: alpha=0.01이면 1% 블록 값과 1% 목표선·제목으로 그리고, 리포트에 없는 목표(10%)는 그 값을 담은
# KeyError다.
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

# 합성 지도 학습 평균 AUROC.
MEANS = (0.95, 0.97, 0.98, 0.985)  # supervised AUROC at k = 5, 10, 20, 40


# 2단계 비교 리포트(compare.build_report) 모양의 합성 데이터. 비지도 세 방법의 AUROC와 구간, k마다 지도 학습의
# 시드 평균·시드별 값·구간, 그리고 judged에 든 k에만 '학습에서 보지 못한 결함 유형' 비교를 넣는다. 보지 못한
# 결함 수는 시드 평균이라 k = 5는 116.67처럼 정수가 아니다(커밋된 리포트와 같은 상황).
def _compare_report(judged=(5, 10)):
    def unsupervised(value):
        return {"macro_image_auroc": value, "macro_image_auroc_ci": [value - 0.01, value + 0.01]}

    supervised = {}
    # Means over the seeds, whose subsets differ: 116.67 at k = 5 (as in the committed report), whole at 10.
    unseen_defects = {5: 350 / 3, 10: 60.0}
    for k, mean in zip((5, 10, 20, 40), MEANS, strict=True):
        unseen = {"unseen_defects_mean": 2.0, "categories_mean": 1.0, "judged": False}
        if k in judged:
            unseen = {
                "unseen_defects_mean": unseen_defects[k],
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


# 확인: label_curve 그림 왼쪽은 리포트 순서와 상관없이 k 순으로 정렬한 지도 학습 평균·구간과 그 옆의 시드별
# 점, 그리고 비지도 방법의 수평선·구간 띠·오른쪽 끝 값이다. 눈금은 'k (검증 포함 라벨 수)'로 적는다. 오른쪽은
# 판정된 k만, 보지 못한 결함 유형에서 지도 학습과 기준 방법(Dinomaly, 왼쪽과 같은 색·모양)을 나란히 두고
# 차이와 구간을 두 마커 사이에 적는다. 보지 못한 결함 수는 시드 평균이라 정수가 아니면 소수로 적는다.
# 왜: README의 질문 1 그림이다. 라벨 비용(k + 20)과 '보지 못한 유형에서는 뒤집힌다'는 결론이 그림에서 읽혀야
# 한다.
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
    # The number of unseen defects is a mean over the seeds: not written as a count unless it is whole.
    assert [t.get_text() for t in unseen.get_xticklabels()] == ["k = 5\n116.7 defects", "k = 10\n60 defects"]
    assert unseen.get_xlabel() == "defects of these types: mean over the 3 seeds"
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
    _tick_labels_apart(fig)
    _texts_inside(fig)


# 확인: 어느 k도 보지 못한 유형 비교가 판정되지 않았으면 오른쪽 패널을 비우는 대신 '그런 결함이 너무 적다'고
# 적고, 왼쪽 곡선은 그대로 그린다.
def test_label_curve_without_a_judged_subset_keeps_the_panel_and_says_so():
    fig = figures.fig_label_curve(_compare_report(judged=()))
    assert len(fig.axes) == 2
    assert not _artists(fig, "unseen:supervised")
    assert "too few such defects at every k" in _texts(fig)
    assert _y(fig, "supervised") == pytest.approx([95.0, 97.0, 98.0, 98.5])
    _marks_inside(fig)


# ---- M2AD ------------------------------------------------------------------------------------------------


# 3-B M2AD 리포트 모양의 합성 데이터. 방법마다 조건 25개의 오검출률·구간과, 새 조명 정상 n장(0, 8, 30)으로
# 재보정한 뒤의 오검출률을 담는다.
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


# 확인: m2ad_conditions 그림 왼쪽은 두 방법을 각 조건 눈금 양옆에 나란히(WRN-50이 왼쪽) 그리고, 조건을 일곱
# 묶음(기준 조명, 밝기, 감마, 흐림, 이동, JPEG, 실제 조명)으로 간격을 두어 나눈다. 기준 조건은 '01
# reference'로, 흐림·이동은 256px 격자 값에 *를 붙이고 축 아래 주석에 448px 입력이 실제로 받은 값을 적는다.
# 오른쪽은 재보정 n별 오검출률과 값 표시다. 목표선은 두 패널 모두에 있고, 선 끝에 글자를 붙이지 않고 범례로
# 이름을 준다(왼쪽 25칸의 자리를 아끼려고). 오검출률 0도 재보정 축에 자리가 있다.
# 왜: README의 '조명이 실제로 바뀌면' 그림이다. '고정 임계값'이라는 말은 왼쪽 패널에만 해당하고 오른쪽은 다시
# 잡은 임계값이라 제목을 따로 단다.
def test_m2ad_conditions_are_grouped_and_every_rate_is_drawn():
    report = _m2ad_report()
    fig = figures.fig_m2ad_conditions(report)
    assert len(fig.axes) == 2
    ax, recal = fig.axes
    # The two methods stand side by side at each condition: marker and interval at the condition's tick.
    ticks, rticks = ax.get_xticks(), recal.get_xticks()
    sides, rsides = figures._offsets(2, 0.38), figures._offsets(2, 0.46)
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
    # The reference condition is lighting 01, the one the thresholds were fixed on (as 02-10 are lightings).
    assert labels[0] == "01\nreference"
    assert labels[1:4] == ["0.9\n ", "0.8\nbrightness", "0.7\n "]
    assert labels[13:16] == ["90\n ", "70\nJPEG quality", "50\n "]
    assert labels[16] == "02\n " and labels[20] == "06\nreal lighting" and labels[24] == "10\n "
    # Blur and shift are on the 256 px grid: their groups are starred, and the note under the axis says
    # what the 448 px input of DINOv2 got (conditions.strength: shift rounded, blur scaled).
    assert labels[7:13] == ["0.5\n ", "1\nblur σ*", "1.5\n ", "1\n ", "2\nshift (px)*", "4\n "]
    assert "gamma" in labels[5]
    assert ax.get_xlabel() == SCALED_NOTE
    # The target line in both panels, named in the legend; a rate of 0 has a place on the recalibration axis.
    targets = _artists(fig, "target")
    assert {line.axes for line in targets} == {ax, recal}
    assert all(list(line.get_ydata()) == pytest.approx([5.0, 5.0]) for line in targets)
    assert not ax.texts  # no label at the end of the line: it would take room from the 25 slots
    assert np.isfinite(recal.transData.transform((1.0, 0.0))).all()
    assert [t.get_text() for t in recal.get_xticklabels()] == ["0", "8", "30"]
    # The legend names the vertical lines too.
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [
        "PatchCore WRN-50",
        "PatchCore DINOv2 ViT-S",
        "95% bootstrap interval",
        "target 5%",
    ]
    # "Fixed thresholds" is said of the left panel only: the right one shows recalibrated thresholds.
    assert fig.get_suptitle() == "M2AD: false alarms when the capture condition changes (2 inspectors)"
    assert ax.get_title(loc="left").startswith("Thresholds fixed on the reference lighting 01: ")
    assert recal.get_title(loc="left") == "After recalibration"
    _marks_inside(fig)
    _tick_labels_apart(fig)
    _markers_apart(fig, ["fpr:p0", "fpr:d-s"])
    _markers_apart(fig, ["recal:p0", "recal:d-s"])
    _texts_inside(fig)


# 확인: 리포트에 있는 방법만 그리고(하나면 눈금 위에), 목표를 10%로 주면 목표선과 범례가 따라 바뀌며, 256px
# 입력뿐이면 '크기 환산' 문구를 붙이지 않는다. 요청한 방법이 하나도 없거나 방법마다 조건 목록이 다르면
# ValueError다.
def test_m2ad_takes_the_methods_the_report_has_and_checks_their_conditions():
    fig = figures.fig_m2ad_conditions(_m2ad_report(methods=("p0",)), alpha=0.1)
    assert len(fig.axes) == 2 and not _artists(fig, "fpr:d-s")
    assert _x(fig, "fpr:p0") == pytest.approx(fig.axes[0].get_xticks())  # alone: on the tick
    assert all(list(line.get_ydata()) == pytest.approx([10.0, 10.0]) for line in _artists(fig, "target"))
    assert [t.get_text() for t in fig.legends[0].get_texts()][-1] == "target 10%"
    # Only the 256 px input: nothing was scaled.
    assert (
        fig.axes[0].get_xlabel()
        == "* blur σ and shift in pixels of a 256 px image, the input of PatchCore WRN-50"
    )
    _marks_inside(fig)
    with pytest.raises(ValueError, match="none of the methods"):
        figures.fig_m2ad_conditions(_m2ad_report(methods=("dm",)))
    report = _m2ad_report()
    report["methods"]["d-s"]["conditions"].pop()
    with pytest.raises(ValueError, match="different conditions"):
        figures.fig_m2ad_conditions(report)


# ---- perturbation ----------------------------------------------------------------------------------------


# 12개 범주 전체의 작은 run_perturb 실행 폴더를 만든다. SHIFTED는 모든 점수에 +2(순위 그대로, 임계값 넘침),
# blur-3은 결함 점수만 −2.5(정확도 하락)다.
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


# 합성 실행들로 실제 analyze_perturb.build_report를 돌려 그림 입력 리포트를 만든다(부트스트랩 40번). 그림과
# 분석 코드가 같은 리포트 형식을 쓰는지도 함께 확인된다.
def _perturb_report(root, cal=None):
    """The report `analyze_perturb` writes, from synthetic runs; `cal`: calibration scores per method."""
    cal = {"p0": 60, "d-s": 60} if cal is None else cal
    run_dirs = {
        method: _perturb_run(root, method, seed=i, cal=n) for i, (method, n) in enumerate(cal.items())
    }
    return {"protocol": "test", **analyze_perturb.build_report(run_dirs, n_boot=40)}


# 확인: perturb 그림 위 패널은 15개 조건의 AUROC 변화(교란 없음 제외), 아래 패널은 교란 없음을 포함한 16개
# 조건의 고정 임계값 오검출률과 구간이다. H9가 이름 붙인 조건(SHIFTED)은 두 패널 모두 음영으로 표시하고 다른
# 조건(blur-3)은 표시하지 않는다. 목표선은 아래 패널에만, 눈금 글자는 아래 패널에만 있고, 범례 마지막 줄이 H9
# 조건을 설명한다.
# 왜: README의 '촬영 조건이 조금 바뀌면' 그림으로, 'AUROC는 그대로인데 오검출률만 뛴다'(H9 지지)를 한눈에 보여
# 주는 것이 목적이다.
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
    assert labels[7:13] == ["0.5\n ", "1\nblur σ*", "1.5\n ", "1\n ", "2\nshift (px)*", "4\n "]
    assert all(t.get_text() == "" for t in ax_auc.get_xticklabels())
    # The starred strengths are on the 256 px grid; the note says what the larger inputs got.
    assert ax_auc.get_xlabel() == ""
    assert ax_fpr.get_xlabel() == SCALED_NOTE
    legend = [t.get_text() for t in fig.legends[0].get_texts()]
    assert legend[:3] == ["PatchCore WRN-50", "PatchCore DINOv2 ViT-S", "95% bootstrap interval"]
    assert legend[3] == "PatchCore WRN-50: AUROC falls by less than 1 point, false alarms at least 10%"
    _marks_inside(fig)
    _tick_labels_apart(fig)
    for gid in ("d_auroc", "fpr"):
        _markers_apart(fig, [f"{gid}:p0", f"{gid}:d-s"])
    _texts_inside(fig)


# 확인: 보정 점수가 없는 방법은 AUROC 변화만 그리고 오검출률 마커를 그리지 않는다. H9가 판정되지 않은
# 리포트(일부 범주 실행)면 음영도 범례 줄도 없다. Dinomaly(392px)가 받은 흐림·이동 값이 주석에 적힌다.
def test_perturb_method_without_calibration_scores_has_no_rate_markers(tmp_path):
    report = _perturb_report(tmp_path, cal={"p0": 60, "dm": 0})
    assert report["tables"]["dm"][0]["fpr"] is None
    fig = figures.fig_perturb(report)
    ticks = fig.axes[1].get_xticks()
    assert _x(fig, "d_auroc:dm") == pytest.approx(ticks[1:] + figures._offsets(2, 0.25)[1])
    assert not _artists(fig, "fpr:dm")
    assert _x(fig, "fpr:p0") == pytest.approx(ticks + figures._offsets(2, 0.25)[0])
    _marks_inside(fig)
    # A partial run is not judged: nothing is shaded and the legend has the methods and intervals only.
    report["h9"] = None
    fig = figures.fig_perturb(report)
    assert not _artists(fig, f"h9:{SHIFTED}")
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [
        "PatchCore WRN-50",
        "Dinomaly",
        "95% bootstrap interval",
    ]
    assert fig.axes[1].get_xlabel().endswith("scaled:\nDinomaly (392 px): σ ×1.53, shift 2, 3, 6 px")


# 확인: 세 방법(가장 긴 범례와 주석)이어도 주석 둘째 줄이 두 입력 크기의 환산을 모두 적고, 눈금 글자·마커가
# 겹치지 않으며 모든 글자가 그림 안에 있다.
def test_perturb_of_three_methods_keeps_its_legend_and_note_inside_the_figure(tmp_path):
    report = _perturb_report(tmp_path, cal={"p0": 60, "d-s": 60, "dm": 60})
    assert report["h9"]["conditions"] == [SHIFTED]  # the longest legend: three methods and the H9 shading
    fig = figures.fig_perturb(report)
    assert fig.axes[1].get_xlabel().splitlines()[1] == (
        "PatchCore DINOv2 ViT-S (448 px): σ ×1.75, shift 2, 4, 7 px;   "
        "Dinomaly (392 px): σ ×1.53, shift 2, 3, 6 px"
    )
    _tick_labels_apart(fig)
    for gid in ("d_auroc", "fpr"):
        _markers_apart(fig, [f"{gid}:p0", f"{gid}:d-s", f"{gid}:dm"])
    _texts_inside(fig)


# 확인: 방법별 입력 크기 표(INPUT_SIZES)가 등록 구성과 같고(7단계 p0-c·d-s-c 포함), 크기에 따라 강도가 바뀌는
# 교란이 흐림과 이동뿐이며, 크기를 모르는 방법은 주석에 'input size not known'으로 적힌다. 그림 주석이
# conditions.strength의 실제 환산을 따르는지 고정한다.
def test_scaled_strengths_follow_the_input_size_of_each_method():
    # The input sizes of stage 3, and the conditions that conditions.strength scales with them.
    # p0-c and d-s-c: the stage 7 centred variants, same input sizes as their base configs.
    assert figures.INPUT_SIZES == {"p0": 256, "d-s": 448, "d-b": 448, "dm": 392, "p0-c": 256, "d-s-c": 448}
    assert figures._SCALED == ("blur", "shift")
    assert [figures._condition(f"P:{kind}-1")[0] for kind in ("gamma", "blur", "shift")] == [
        "gamma",
        "blur σ*",
        "shift (px)*",
    ]
    note = figures._scaled_note(["d-s", "mystery"])
    assert note == (
        "* blur σ and shift in pixels of a 256 px image; other input sizes got them scaled:\n"
        "PatchCore DINOv2 ViT-S (448 px): σ ×1.75, shift 2, 4, 7 px;   mystery: input size not known"
    )


# ---- grid ------------------------------------------------------------------------------------------------


# 4단계 격자 리포트와 지연 파일을 analyze_grid의 main처럼 만든다. 설정 두 개 × 코어셋 두 비율의 CPU FP32
# 지연(WRN-50은 INT8도)과, GPU로만 잰 행 하나를 넣는다. 리포트에 합쳐 넣은 지연과 디스크의 지연 파일은 따로
# 둔다(나중에 파일만 바뀐 상황을 시험하려고). dev면 서빙 구성 선택도 넣는다.
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


# WRN-50 256px의 코어셋 10%(느림, 182ms) 행과 1%(빠름, 62ms) 행의 키.
SLOW, FAST = f"wrn50-256-{ratio_name(0.1)}", f"wrn50-256-{ratio_name(0.01)}"


# 확인: grid 그림은 로그 x축에 각 (설정, 비율, 정밀도)의 CPU 지연(리사이즈 포함)과 dev AUROC를 찍는다. INT8은
# 자기 지연과 그 행의 AUROC로, GPU로만 잰 행은 찍지 않는다. 마커 크기는 코어셋 비율이고, 예산선(200ms), 서빙
# 구성의 고리와 설명, 설정 이름표가 붙는다. 지연 파일 없이 리포트에 합친 값만으로도 같은 그림이 나온다.
# 왜: README의 '200ms 안에 넣으려면' 그림이다. 규칙대로 고른 서빙 구성이 어느 점인지가 그림에서 정확히 보여야
# 한다.
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


# 확인: 예산선과 서빙 고리는 리포트에 기록된 예산(150ms)과 선택을 따른다. 그림이 선택을 다시 계산하지 않는다.
def test_grid_takes_the_budget_and_the_choice_of_the_report():
    grid, latency = _grid_inputs(budget_ms=150.0)
    assert grid["serving"]["key"] == FAST  # 182 ms is over 150 ms
    fig = figures.fig_grid(grid, latency)
    assert _x(fig, "budget") == pytest.approx([150.0, 150.0])
    assert _x(fig, "serving") == pytest.approx([62.0])
    assert "budget 150 ms" in _texts(fig) and "serving: 62 ms, AUROC 89.6" in _texts(fig)
    _marks_inside(fig)


# 확인: analyze_grid 뒤에 bench를 다시 돌려 지연 파일이 바뀌어도, 그림은 리포트에 합쳐진(선택에 쓰인) 값을
# 먼저 쓴다. 지연 파일은 리포트에 없는 행이나 나중에 잰 정밀도만 채운다.
# 왜: 4단계에서 한가할 때 다시 잰 지연으로 서빙 구성을 골랐다(규칙 변경 6). 그림이 다른 측정값을 보여 주면 '왜
# 이 구성을 골랐는지'가 그림과 어긋난다.
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


# 확인: FP32 마커의 x는 analyze_grid가 등록한 cpu_ms(리사이즈 10 + 180 = 190ms)이고, 나중에 INT8만 다시 재면서
# 바뀐 resize_ms로 다시 더한 값이 아니다. INT8 마커는 자기 total_with_resize_ms를 쓴다.
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


# 확인: FP32 지연이 없고 INT8만 있는 설정도 INT8 마커 옆에 이름표를 붙이고, INT8뿐이어도 범례가 속 빈 마커가
# INT8임을 설명한다.
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


# 확인: test 리포트처럼 서빙 선택이 없으면 고리를 그리지 않고 기본 예산선만 그린다. CPU 지연이 하나도 없으면
# 'CPU latency' ValueError다(빈 그림을 만들지 않는다).
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


# fig_examples에 직접 넘기는 패널 행 네 개: 결함(마스크 있음, 맵이 임계값을 넘는 칸이 마스크 위), 정상(마스크
# 없음), 임계값 없음, 맵 전체가 임계값 위. 맵은 4×4 픽셀마다 점수 하나라 이미지보다 거칠다.
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


# 윤곽선 모음의 꼭짓점 경계 (x 최소, y 최소, x 최대, y 최대).
def _bounds(contours):
    vertices = np.concatenate([p.vertices for p in contours.get_paths()])
    return (*vertices.min(axis=0), *vertices.max(axis=0))


# 확인: 행마다 이미지, 정답 윤곽, 이상 맵 세 패널과 색 막대 하나가 있다. 이미지 좌표계를 쓰고, 거친 맵도
# 이미지 전체에 펼친다. 색 범위는 그 행의 맵과 임계값을 함께 덮고, 임계값은 색 막대에 선으로, 맵에는 넘는 곳의
# 윤곽선으로 표시한다. 낮은 점수는 투명하고 맵 전체가 임계값 위면 전부 색이 칠해진다. 마스크가 없는 정상 행은
# 윤곽 대신 '결함 마스크 없음'을 적는다.
# 왜: README 실패 사례 그림에서 '어디가 이상하다고 봤는지'와 '정답 위치'를 같은 좌표로 비교할 수 있어야 한다.
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


# 확인: NaN·무한대가 든 맵, 2차원이 아닌 맵, 이미지와 크기가 다른 마스크는 ValueError다. 높이 1인 맵은
# 임계값을 넘어도 윤곽선을 만들 수 없으므로 윤곽 없이 색 막대 표시만 남는다.
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


# 확인: 저장소에 커밋된 실제 리포트(reports/)가 있으면 그것으로 그려도 폭 1800px PNG가 나오고 표시·글자가
# 잘리거나 겹치지 않는다. 리포트가 없으면 건너뛴다. 합성 데이터에서만 통과하는 그림 코드를 막는다.
@pytest.mark.parametrize("name", ["calibration", "label_curve", "m2ad_conditions", "perturb"])
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
    _tick_labels_apart(fig)
    _texts_inside(fig)


# 확인: 같은 리포트로 두 번 저장하거나 두 번 만들어도 PNG 바이트가 같고, PNG 메타데이터에 matplotlib
# 버전(Software)이 없다. 실패 사례 패널도 같다.
# 왜: 그림을 다시 그렸을 때 내용이 바뀌지 않았으면 git diff에 나타나지 않아야 한다(모듈 docstring: 시계·난수를
# 쓰지 않고 버전 문자열을 넣지 않는다).
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


# 확인: 그림을 만든 뒤 전역 matplotlib 설정(rcParams)이 그대로다. 그림 스타일은 rc_context 안에서만 적용된다.
def test_building_a_figure_leaves_the_global_settings_alone():
    import matplotlib

    before = dict(matplotlib.rcParams)
    figures.fig_calibration(_calibration_report())
    assert dict(matplotlib.rcParams) == before


# 확인: 사용자의 matplotlibrc나 호출 코드가 바꿔 둔 설정(bbox tight, 투명 배경, 선 두께, 이미지 원점 등)이
# 있어도, 만들 때든 저장할 때든 PNG가 기본 상태와 바이트 단위로 같다. _style이 매번 기본값으로 되돌린 뒤
# 프로젝트 스타일만 얹기 때문이다.
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


# 리포트 JSON을 UTF-8로 쓴다(상위 폴더 포함).
def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# 확인: CLI는 입력 리포트가 있는 그림만 쓰고, 없는 그림은 빠진 파일 경로와 함께 'skipped'로 알린다(grid는 지연
# 파일이 없어 건너뜀). 다시 돌리면 같은 바이트로 덮어쓰고, --only로 고른 그림만 다룬다.
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


# 확인: 입력이 모두 있으면 FIGURES의 그림 다섯 개를 모두 쓰고(없는 출력 폴더도 만든다), --only examples는
# 리포트 그림이 아니라서 거부한다.
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


# 확인: 옛 형식 리포트(KeyError)나 쓰다 끊긴 JSON(JSONDecodeError)이 있어도 다른 그림은 그리고, 실패한 그림과
# 원인을 stderr에 적은 뒤 마지막에 'figures not written: …'으로 실패 종료한다. 리포트 하나 때문에 나머지
# 그림이 갱신되지 않거나, 실패가 조용히 묻히지 않게 한다.
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


# 확인: 한국어 Windows처럼 stdout이 cp949로 리디렉션돼 있어도, cp949로 표현할 수 없는 글자(é, 中)가 든 경로를
# 출력하다 멈추지 않는다. main이 stdout을 UTF-8로 다시 설정하기 때문이다.
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


# ---- example inspections of an evaluation run --------------------------------------------------------------

# 실패 사례 패널은 256px 캐시의 이미지와 마스크를 쓴다. 아래 EXAMPLE_CATS·EXPECTED_EXAMPLES는 선택 규칙의
# 결과를 손으로 계산할 수 있게 고른 점수와 그 기대 결과다.
SIZE = cache.MASK_SIZE
# category -> (threshold its calibration scores give, [(image, defect types, image score)] in run order).
# Detected margins 1.75, 2.25, 8 (cat_a) and 1.25, 1.5, 16 (cat_b): median 2.0, with the two cat_a hits
# 0.25 away from it on either side. cat_a/Anomaly/002 is listed before 001, so only the path breaks the tie.
# The scores equal to the threshold are not flagged (score > threshold is the rule).
EXAMPLE_CATS = {
    "cat_a": (
        1.0,
        [
            ("cat_a/Normal/000.JPG", "", 0.5),
            ("cat_a/Normal/001.JPG", "", 1.0),
            ("cat_a/Normal/002.JPG", "", 3.0),
            ("cat_a/Normal/003.JPG", "", 5.0),
            ("cat_a/Anomaly/000.JPG", "crack", 0.25),
            ("cat_a/Anomaly/002.JPG", "scratch", 1.75),
            ("cat_a/Anomaly/001.JPG", "crack|scratch", 2.25),
            ("cat_a/Anomaly/003.JPG", "crack", 8.0),
            ("cat_a/Anomaly/004.JPG", "hole", 0.125),
        ],
    ),
    "cat_b": (
        2.0,
        [
            ("cat_b/Normal/000.JPG", "", 1.0),
            ("cat_b/Normal/001.JPG", "", 8.0),
            ("cat_b/Anomaly/000.JPG", "bent", 1.0),
            ("cat_b/Anomaly/001.JPG", "bent", 2.5),
            ("cat_b/Anomaly/002.JPG", "dent", 2.0),
            ("cat_b/Anomaly/003.JPG", "dent", 3.0),
            ("cat_b/Anomaly/004.JPG", "bent|dent", 32.0),
        ],
    ),
}
# (image, row title, threshold) of the six rows the rule picks, in panel order.
EXPECTED_EXAMPLES = [
    ("cat_a/Anomaly/001.JPG", "detected defect: cat_a, crack + scratch, score/threshold = 2.25", 1.0),
    ("cat_b/Anomaly/003.JPG", "detected defect: cat_b, dent, score/threshold = 1.50", 2.0),
    ("cat_a/Anomaly/004.JPG", "missed defect: cat_a, hole, score/threshold = 0.12", 1.0),
    ("cat_b/Anomaly/000.JPG", "missed defect: cat_b, bent, score/threshold = 0.50", 2.0),
    ("cat_a/Normal/003.JPG", "false alarm: cat_a, normal, score/threshold = 5.00", 1.0),
    ("cat_b/Normal/001.JPG", "false alarm: cat_b, normal, score/threshold = 4.00", 2.0),
]


N_CAL = 19  # calibration scores per category: the conformal rank at 5% is the 19th, the largest
# Keys the driver reads from each .npz before any score: image lists and evaluation labels.
NAME_KEYS = {
    "common": ("eval_images", "eval_labels", "cal_images"),
    "patchcore": ("eval_images", "eval_labels", "pool_images"),
}


# 경로로 결함 이미지인지 가린다.
def _is_defect(image):
    return "/Anomaly/" in image


# 범주별 보정 이미지 이름 19개(평가 이미지와 다른 학습 풀 정상).
def _cal_images(category):
    return [f"{category}/cal/{i:03d}.JPG" for i in range(N_CAL)]


# EXAMPLE_CATS로 완성된 평가 실행 폴더를 만든다. common은 Dinomaly 공통 형식(eval_score, cal_score,
# cal_images), patchcore는 run_patchcore 형식(여러 보정 방식의 점수가 같이 있고 교차 적합 점수
# pool_score_oof만 임계값에 써야 함)이다. 보정 점수 19개의 최댓값이 정해 둔 임계값이 된다. scores로 일부
# 이미지 점수를 바꿀 수 있다.
def _write_examples_run(root, fmt="common", protocol="dev", scores=None):
    """A finished evaluation run of EXAMPLE_CATS: the common format (Dinomaly) or a run_patchcore run.

    Map i of a category is filled with i, so a drawn map tells which evaluation row it came from.
    """
    run_dir = root / "run"
    run_dir.mkdir(parents=True)
    categories = []
    for category, (threshold, items) in EXAMPLE_CATS.items():
        images = np.array([image for image, _, _ in items])
        labels = np.array([int(_is_defect(image)) for image in images], dtype=np.int8)
        score = np.array([(scores or {}).get(image, s) for image, _, s in items], dtype=np.float32)
        # The largest of the N_CAL scores is the threshold.
        cal = (threshold * np.linspace(0.0, 1.0, N_CAL)).astype(np.float32)
        if fmt == "common":
            arrays = {"eval_score": score, "cal_score": cal, "cal_images": np.array(_cal_images(category))}
        else:  # other strategies' scores too: only the cross-fitted ones may decide
            arrays = {
                "eval_score_full": score,
                "eval_score_holdout": score[::-1].copy(),
                "pool_images": np.array(_cal_images(category)),
                "pool_score_oof": cal,
                "pool_score_resub": cal / 4,
                "pool_folds": np.arange(len(cal)) % 5,
            }
        np.savez(
            run_dir / f"{category}.npz",
            eval_images=images,
            eval_labels=labels,
            eval_defect_types=np.array([types for _, types, _ in items]),
            **arrays,
        )
        maps = np.stack([np.full((SIZE, SIZE), i, dtype=np.float16) for i in range(len(items))])
        np.save(run_dir / f"{category}_maps.npy", maps)
        categories.append({"category": category, "eval_normal": int((labels == 0).sum())})
    meta = {"protocol": protocol, "categories": categories}
    if fmt == "common":
        meta |= {"method": "dinomaly", "config": {"name": "dm"}}
    else:
        meta |= {"config": {"name": "p0", "backbone": "wrn50"}}
    _write_json(run_dir / "run.json", meta)
    return run_dir


# 가짜 매니페스트와 256px 캐시를 만드는 fixture(프로토콜을 받는 함수를 돌려준다). 평가 이미지는 프로토콜에
# 맞는 역할(dev면 학습 풀·dev 결함, test면 봉인 테스트)을 받고, 실행과 다른 순서·다른 행을 섞는다. 캐시 이미지
# j는 값 j + 1로 채우고 결함 j는 4j행부터 마스크 띠를 둬, 그려진 이미지가 어느 행에서 왔는지 값으로 알 수
# 있다. 장부는 임시 파일로 돌린다.
@pytest.fixture
def fake_data(tmp_path, monkeypatch):
    """A manifest and a 256 px cache of the EXAMPLE_CATS images (other rows and another order than the run).

    The evaluation images get the roles of the protocol; the calibration images are pool normals, after
    the others. Cache image j is filled with j + 1; defect j has a mask band at rows 4j..4j+8. The test
    ledger goes to a temporary file. Call the returned function with the protocol; it returns the
    manifest rows.
    """
    monkeypatch.setattr(paths, "TEST_LEDGER", tmp_path / "ledger" / "test_ledger.jsonl")
    monkeypatch.setattr(ledger, "git_commit", lambda root=None: "abc1234")

    def build(protocol="dev"):
        roles = ("pool_normal", "dev_defect") if protocol == "dev" else ("test_normal", "test_defect")
        rows = [ManifestRow("other/Normal/000.JPG", "", "other", "normal", "pool_normal", 1, "")]
        for category in reversed(list(EXAMPLE_CATS)):
            for image, types, _ in reversed(EXAMPLE_CATS[category][1]):
                if _is_defect(image):
                    mask = image.replace("Images", "Masks").replace(".JPG", ".png")
                    rows.append(ManifestRow(image, mask, category, "anomaly", roles[1], -1, types))
                else:
                    fold = 0 if protocol == "dev" else -1
                    rows.append(ManifestRow(image, "", category, "normal", roles[0], fold, ""))
        for category in EXAMPLE_CATS:
            rows += [
                ManifestRow(im, "", category, "normal", "pool_normal", 1, "") for im in _cal_images(category)
            ]
        manifest = tmp_path / "data" / "visa.csv"
        write_manifest(rows, manifest)
        out = tmp_path / "data" / "cache"
        out.mkdir(parents=True)
        index = "".join(f"{row.image}\n" for row in rows)
        for path in (cache.index_path(out, SIZE), cache.masks_index_path(out)):
            path.write_text(index, encoding="utf-8")
        images = np.zeros((len(rows), SIZE, SIZE, 3), dtype=np.uint8)
        masks = np.zeros((len(rows), SIZE, SIZE), dtype=np.uint8)
        for j, row in enumerate(rows):
            images[j] = j + 1
            if row.mask:
                masks[j, 4 * j : 4 * j + 8, 16:32] = 1
        np.save(cache.images_path(out, SIZE), images)
        np.save(cache.masks_path(out), masks)
        monkeypatch.setattr(paths, "VISA_MANIFEST", manifest)
        monkeypatch.setattr(paths, "CACHE", out)
        return rows

    return build


# 캐시 이미지·마스크와 점수 맵을 읽을 때마다 (무엇을, 어떤 행을, 그때 장부에 있던 줄)을 기록하는 엿보기
# fixture. '장부 기록 뒤에만 이미지를 읽는다'를 순서로 확인한다.
@pytest.fixture
def reads(monkeypatch):
    """Every read of cached images, masks and score maps, with the ledger lines that existed at the time."""
    seen = []

    def ledger_lines():
        path = paths.TEST_LEDGER
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    for method in ("images", "masks"):
        original = getattr(cache.ImageCache, method)

        def spy(self, rows, _original=original, _method=method):
            seen.append((_method, [row.image for row in rows], ledger_lines()))
            return _original(self, rows)

        monkeypatch.setattr(cache.ImageCache, method, spy)
    read_map = figures._read_map

    def map_spy(path, index):
        seen.append(("map", [f"{path.name}:{index}"], ledger_lines()))
        return read_map(path, index)

    monkeypatch.setattr(figures, "_read_map", map_spy)
    return seen


# fig_examples 호출을 가로채 (패널 행, 키워드 인자, 그림)을 기록한다. 어떤 행이 어떤 제목·임계값으로
# 그려졌는지 본다.
@pytest.fixture
def drawn(monkeypatch):
    """(rows, keyword arguments, figure) of every call of fig_examples."""
    calls = []
    original = figures.fig_examples

    def spy(rows, **kwargs):
        fig = original(rows, **kwargs)
        calls.append((rows, kwargs, fig))
        return fig

    monkeypatch.setattr(figures, "fig_examples", spy)
    return calls


# 실행의 .npz 파일 읽기를 기록한다(일부 키만 읽는 _npz_arrays와 전부 읽는 compare._load_npz). 봉인 이미지를
# 거부할 때 점수를 한 개도 읽지 않았는지 확인하는 데 쓴다.
@pytest.fixture
def npz_reads(monkeypatch):
    """Every read of a run's .npz files: (file name, keys read); "all" for compare's loaders."""
    seen = []
    some, every = figures._npz_arrays, compare._load_npz

    def some_spy(path, keys):
        seen.append((Path(path).name, tuple(keys)))
        return some(path, keys)

    def every_spy(path):
        seen.append((Path(path).name, "all"))
        return every(path)

    monkeypatch.setattr(figures, "_npz_arrays", some_spy)
    monkeypatch.setattr(compare, "_load_npz", every_spy)
    return seen


# 범주 .npz를 읽어 change가 dict를 고치게 한 뒤 다시 저장한다(손상·변형 실행을 만드는 도우미).
def _rewrite(run_dir, category, change):
    """Load every array of a category's .npz, let `change` edit the dict in place, save it again."""
    path = run_dir / f"{category}.npz"
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    change(arrays)
    np.savez(path, **arrays)


# 범주의 평가 배열 전부와 맵에 결함 행을 덧붙인다(봉인 이미지를 몰래 섞는 상황 등).
def _add_eval_rows(run_dir, category, images, score):
    """Append defects ("crack", one score) to a category's evaluation: every eval_* array and the maps."""
    n = len(images)

    def change(arrays):
        extra = {
            "eval_images": np.array(images),
            "eval_labels": np.ones(n, dtype=np.int8),
            "eval_defect_types": np.array(["crack"] * n),
        }
        extra |= {key: np.full(n, score, np.float32) for key in arrays if key.startswith("eval_score")}
        for key, values in extra.items():
            arrays[key] = np.concatenate([arrays[key], values])

    _rewrite(run_dir, category, change)
    maps = np.load(run_dir / f"{category}_maps.npy")
    np.save(run_dir / f"{category}_maps.npy", np.concatenate([maps, np.zeros((n, SIZE, SIZE), np.float16)]))


# 가짜 매니페스트에 행을 더한다.
def _add_manifest_rows(rows):
    write_manifest(read_manifest(paths.VISA_MANIFEST) + rows, paths.VISA_MANIFEST)


# 'missed defect' Example 하나를 만든다(제목 형식 시험용).
def _example(category, image, margin, types="crack"):
    return figures.Example("missed defect", category, 0, image, 1, types, margin)


# 확인: 선택 규칙이 비(score/threshold)가 같은 후보를 이미지 경로 순으로 가르고, 한 묶음 안에서 범주당 한 장만
# 고른다(범주 이름 순서가 섞여 있어도). per_group을 늘리면 범주마다 하나씩 더 나온다. 임계값이 0이면 비를 낼
# 수 없어, 점수에 NaN이 있으면 ExamplesError다.
# 왜: README 실패 사례는 '미리 정한 규칙으로 고른 예시'다. 동점 처리까지 정해져 있어야 같은 실행에서 늘 같은
# 그림이 나온다.
def test_examples_breaks_ties_by_path_and_takes_one_image_per_category():
    cats = {}
    for category in ("z", "y", "x"):  # not in path order
        cats[category] = {
            "eval_images": np.array([f"{category}/2.JPG", f"{category}/1.JPG", f"{category}/0.JPG"]),
            "eval_labels": np.array([1, 1, 0], dtype=np.int8),
            "eval_defect_types": np.array(["crack", "dent", ""]),
            "eval_score": np.array([0.5, 0.5, 0.25], dtype=np.float32),  # two missed defects with one margin
        }
    picks = figures.pick_examples(cats, {"x": 1.0, "y": 1.0, "z": 1.0})
    assert [(e.category, e.image, e.index) for e in picks["missed defect"]] == [
        ("x", "x/1.JPG", 1),
        ("y", "y/1.JPG", 1),
    ]
    assert picks["detected defect"] == [] and picks["false alarm"] == []
    assert figures.pick_examples(cats, {"x": 1.0, "y": 1.0, "z": 1.0}, per_group=5)["missed defect"] == [
        figures.Example("missed defect", c, 1, f"{c}/1.JPG", 1, "dent", 0.5) for c in ("x", "y", "z")
    ]
    with pytest.raises(figures.ExamplesError, match="threshold of x"):
        figures.pick_examples(cats, {"x": 0.0, "y": 1.0, "z": 1.0})
    cats["y"]["eval_score"] = np.array([np.nan, 0.5, 0.25], dtype=np.float32)
    with pytest.raises(figures.ExamplesError, match="not all finite"):
        figures.pick_examples(cats, {"x": 1.0, "y": 1.0, "z": 1.0})


# 확인: 행 제목은 '묶음: 범주, 결함 유형, score/threshold = 값'이다. 줄바꿈은 34자·3줄 안에 들도록 하되 비
# 표시는 쪼개지 않고, 유형이 많으면 뒤쪽을 '+ n more'로 줄인다. 비는 소수 둘째 자리로 쓰되, 1이 아닌 값이
# 반올림으로 '1.00'이 되면 판정(1 초과면 불량)과 반대로 읽히므로 1과 구별될 때까지 자릿수를 늘린다.
def test_example_titles_keep_the_ratio_whole_and_never_round_it_to_the_other_side_of_one():
    long_types = "|".join(
        ["breakage down the middle", "color spot", "other", "small cracks", "small scratches"]
    )
    example = _example("macaroni2", "m/1.JPG", 1.2345, long_types)
    assert example.title == (
        "missed defect: macaroni2, breakage down the middle + color spot + other + small cracks + small "
        "scratches, score/threshold = 1.23"
    )
    lines = example.wrapped_title().split("\n")
    assert len(lines) <= 3 and all(len(line) <= 34 for line in lines)
    assert lines[-1].endswith("score/threshold = 1.23") and "more," in " ".join(lines)
    short = _example("pcb3", "p/1.JPG", 0.75, "melt|missing|scratch")
    assert short.wrapped_title().replace("\n", " ") == short.title  # fits: nothing left out
    assert _example("a", "a/1.JPG", 0.999).title.endswith("= 0.999")
    assert _example("a", "a/1.JPG", 1.0004).title.endswith("= 1.0004")
    assert _example("a", "a/1.JPG", 0.125).title.endswith("= 0.12")
    normal = figures.Example("false alarm", "a", 3, "a/n.JPG", 0, "", 2.5)
    assert normal.title == "false alarm: a, normal, score/threshold = 2.50"
    # As many digits as it takes, however close to 1.
    assert _example("a", "a/1.JPG", float(np.nextafter(1.0, 0.0))).title.endswith("= 0.9999999999999999")
    assert _example("a", "a/1.JPG", float(np.nextafter(1.0, 2.0))).title.endswith("= 1.0000000000000002")
    assert _example("a", "a/1.JPG", 1.0).title.endswith("= 1.00")


# 확인: 임계값 바로 아래(float32 한 칸) 점수는 놓친 결함이고 비가 0.9999999로, 임계값과 같은 점수도 '점수 >
# 임계값' 규칙상 놓친 결함이고 비가 정확히 1.00으로 적힌다. 검출 묶음에는 들어가지 않는다.
def test_example_margins_next_to_one_show_their_side_of_the_threshold():
    threshold = np.float32(1.3)
    below = np.nextafter(threshold, np.float32(0.0))  # one float32 step under the threshold
    cats = {
        category: {
            "eval_images": np.array([f"{category}/0.JPG"]),
            "eval_labels": np.array([1], dtype=np.int8),
            "eval_defect_types": np.array(["crack"]),
            "eval_score": np.array([score], dtype=np.float32),
        }
        for category, score in (("a", below), ("b", threshold))
    }
    picks = figures.pick_examples(cats, {"a": float(threshold), "b": float(threshold)})
    # Flagged means score > threshold (calibrate's rule): a score equal to it is missed, at exactly 1.
    assert [e.title for e in picks["missed defect"]] == [
        "missed defect: a, crack, score/threshold = 0.9999999",
        "missed defect: b, crack, score/threshold = 1.00",
    ]
    assert picks["detected defect"] == []


# 확인(정상 경로): dev 실행으로 패널을 그리면 장부를 쓰지 않고 PNG 하나를 쓴다. 고른 여섯 행이 규칙의
# 기대값(검출 2: 중앙값에 가까운 비, 놓침 2: 가장 낮은 비, 오검출 2: 가장 높은 비, 묶음 안 범주당 한 장)과
# 같고, 각 행의 이미지·마스크(결함만)·맵(그 평가 행의 것)이 캐시와 실행에서 정확히 왔다. 제목은 방법과 분할을,
# 설명은 데이터 출처(VisA CC BY 4.0)·임계값 출처(Dinomaly는 홀드아웃, PatchCore는 교차 적합)·판정
# 규칙·윤곽선의 한계를 적고, 모두 그림 안에 있다.
@pytest.mark.parametrize("fmt", ["common", "patchcore"])
def test_examples_draw_the_rows_the_rule_picks_with_their_image_mask_and_map(
    fmt, tmp_path, fake_data, reads, drawn, capsys
):
    rows = fake_data("dev")
    run_dir = _write_examples_run(tmp_path, fmt)
    out = tmp_path / "figs"
    figures.main(["--examples", str(run_dir), "--out", str(out)])
    assert capsys.readouterr().out.splitlines() == [f"wrote {out / 'examples.png'}"]
    assert (out / "examples.png").exists() and not paths.TEST_LEDGER.exists()  # a dev run is not logged
    ((panel, kwargs, fig),) = drawn
    assert [row["title"].replace("\n", " ") for row in panel] == [title for _, title, _ in EXPECTED_EXAMPLES]
    assert [row["threshold"] for row in panel] == [thr for _, _, thr in EXPECTED_EXAMPLES]
    cache_row = {row.image: j for j, row in enumerate(rows)}
    for row, (image, _, _) in zip(panel, EXPECTED_EXAMPLES, strict=True):
        j = cache_row[image]
        assert row["image"].shape == (SIZE, SIZE, 3) and np.all(row["image"] == j + 1)
        if _is_defect(image):
            assert row["mask"].dtype == bool and np.array_equal(
                np.flatnonzero(row["mask"].any(axis=1)), np.arange(4 * j, 4 * j + 8)
            )
        else:
            assert row["mask"] is None
        category = image.split("/")[0]
        index = [item[0] for item in EXAMPLE_CATS[category][1]].index(image)
        assert np.array_equal(row["map"], np.full((SIZE, SIZE), float(index)))
    # At most one image per category in each group.
    shown = [row["title"].replace("\n", " ").split(", ")[0].split(": ") for row in panel]
    assert [tuple(pair) for pair in shown] == [
        (group, category) for group in figures.EXAMPLE_GROUPS for category in ("cat_a", "cat_b")
    ]
    method = "Dinomaly" if fmt == "common" else "PatchCore WRN-50"
    source = "held-out normal images" if fmt == "common" else "cross-fitted (out-of-fold) scores"
    assert kwargs["title"] == f"Example inspections: {method} on the VisA dev split"
    assert fig.get_suptitle() == kwargs["title"]
    caption = fig.get_supxlabel().replace("\n", " ")
    assert caption.startswith(f"Images: VisA (Zou et al., ECCV 2022), CC BY 4.0. Method: {method}.")
    assert source in caption and "5% target false-alarm rate" in caption
    # The decision rule (a score equal to the threshold is not flagged), and what the line on a map is.
    assert "An image is flagged when its score/threshold is above 1." in caption
    assert "the line and the decision can disagree" in caption
    # The figure title, the caption and the row titles lie inside the figure (the panels have no ticks).
    renderer = _drawn(fig)
    titles = [ax.title for ax in fig.axes if ax.get_label() != "colorbar"]
    for text in [fig._suptitle, fig._supxlabel, *titles]:
        box = text.get_window_extent(renderer)
        assert (
            fig.bbox.x0 <= box.x0 and box.x1 <= fig.bbox.x1 and fig.bbox.y0 <= box.y0 <= box.y1 <= fig.bbox.y1
        )
    with Image.open(out / "examples.png") as image:
        assert image.format == "PNG" and image.width <= 1800


# 확인: 오검출이 하나도 없고 놓친 결함이 한 범주에만 있으면, 있는 만큼만 그리고 제목 둘째 줄에 모자란 묶음과
# 개수를 적는다(빈 자리를 다른 규칙으로 채우지 않는다).
def test_examples_say_in_the_title_when_a_group_is_short(tmp_path, fake_data, drawn):
    fake_data("dev")
    # No false alarm left; the missed defects are all in cat_a, which gives one row.
    calm = {"cat_a/Normal/002.JPG": 0.5, "cat_a/Normal/003.JPG": 0.5, "cat_b/Normal/001.JPG": 0.5}
    run_dir = _write_examples_run(
        tmp_path, scores={**calm, "cat_b/Anomaly/000.JPG": 6.0, "cat_b/Anomaly/002.JPG": 5.0}
    )
    figures.examples(run_dir, tmp_path / "figs")
    ((panel, kwargs, _),) = drawn
    assert [row["title"].split(":")[0] for row in panel] == [
        "detected defect",
        "detected defect",
        "missed defect",
    ]
    assert kwargs["title"] == (
        "Example inspections: Dinomaly on the VisA dev split\n"
        "Fewer than 2 found (one per category at most): 1 missed defect, 0 false alarms"
    )


# 확인: test 절차 실행은 --allow-test 없이는 코드 2로 거부하고, 이미지를 읽지도 장부를 쓰지도 출력 폴더를
# 만들지도 않는다. allow_test는 정확히 True일 때만 열리며 'yes', 1, 'true' 같은 참 같은 값은
# SealedTestError다(splits.select와 같은 방식).
def test_examples_of_a_test_run_are_refused_without_allow_test(tmp_path, fake_data, reads, capsys):
    fake_data("test")
    run_dir = _write_examples_run(tmp_path, protocol="test")
    out = tmp_path / "figs"
    with pytest.raises(SystemExit) as stop:
        figures.main(["--examples", str(run_dir), "--out", str(out)])
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("examples not drawn: ") and "sealed test set" in err and "--allow-test" in err
    assert reads == [] and not paths.TEST_LEDGER.exists() and not out.exists()
    # Only True opens the seal, as in splits.select.
    for value in ("yes", 1, "true"):
        with pytest.raises(SealedTestError):
            figures.examples(run_dir, out, allow_test=value)
    assert reads == [] and not paths.TEST_LEDGER.exists()


# 확인: run.json은 dev라고 하지만 매니페스트상 봉인 테스트 이미지를 채점한 실행은 거부한다. 첫 범주의 이미지
# 목록 키만 읽은 시점에서 멈춰, 봉인 이미지의 점수를 한 개도 읽지 않는다.
def test_examples_refuse_sealed_images_in_a_run_that_says_dev(tmp_path, fake_data, reads, npz_reads, capsys):
    fake_data("test")  # the manifest has the images as sealed test images
    run_dir = _write_examples_run(tmp_path, protocol="dev")
    with pytest.raises(SystemExit) as stop:
        figures.main(["--examples", str(run_dir), "--out", str(tmp_path / "figs")])
    assert stop.value.code == 2 and "sealed test image" in capsys.readouterr().err
    assert reads == [] and not paths.TEST_LEDGER.exists()
    # Refused on the image lists of the first category: no score of a sealed image was read.
    assert npz_reads == [("cat_a.npz", NAME_KEYS["common"])]


# 확인: 그려지지 않을 행에 봉인 이미지가 숨어 있어도 거부한다. 평가 행에 섞인 봉인 결함 세 장(비 16이라
# 뽑히지는 않지만 검출 비의 중앙값을 움직인다)이나, 임계값을 정하는 보정 이미지로 쓰인 봉인 정상이 그 경우다.
# 두 형식 모두 이미지 목록만 읽고 멈춘다.
# 왜: 봉인 이미지의 점수가 그림에 드러나지 않아도 임계값이나 선택을 바꾸면 봉인 테스트를 본
# 셈이다(_check_scored_images docstring).
@pytest.mark.parametrize("fmt", ["common", "patchcore"])
@pytest.mark.parametrize("where", ["eval", "cal"])
def test_examples_refuse_a_dev_run_with_sealed_images_among_the_rows_not_drawn(
    where, fmt, tmp_path, fake_data, reads, npz_reads, capsys
):
    # Sealed images whose scores would steer the panel without being drawn: three detected test defects
    # (margin 16: never picked, but they move the median of the detected margins), or test normals as
    # the calibration images whose scores set the threshold of cat_a.
    fake_data("dev")
    run_dir = _write_examples_run(tmp_path, fmt)
    if where == "eval":
        sealed = [f"cat_a/Anomaly/9{i:02d}.JPG" for i in range(3)]
        mask = "cat_a/Anomaly/mask.png"
        _add_manifest_rows(
            [ManifestRow(im, mask, "cat_a", "anomaly", "test_defect", -1, "crack") for im in sealed]
        )
        _add_eval_rows(run_dir, "cat_a", sealed, 16.0)
    else:
        sealed = [f"cat_a/Normal/9{i:02d}.JPG" for i in range(N_CAL)]
        _add_manifest_rows([ManifestRow(im, "", "cat_a", "normal", "test_normal", -1, "") for im in sealed])
        _rewrite(run_dir, "cat_a", lambda arrays: arrays.update({NAME_KEYS[fmt][2]: np.array(sealed)}))
    out = tmp_path / "figs"
    with pytest.raises(SystemExit) as stop:
        figures.main(["--examples", str(run_dir), "--out", str(out)])
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("examples not drawn: ") and f"{sealed[0]}, a sealed test image" in err
    assert reads == [] and not paths.TEST_LEDGER.exists() and not out.exists()
    assert npz_reads == [("cat_a.npz", NAME_KEYS[fmt])]


# 확인: run.json의 범주 이름은 그대로 실행 폴더의 파일 이름이 되므로, 경로 이동(../, ..\, 드라이브 C:), 빈
# 이름, 숫자, 중복, 매니페스트에 없는 범주를 .npz를 하나도 열기 전에 거부한다. 옆에 둔 test 실행 파일을 dev
# 실행 이름으로 끌어다 쓰는 길을 막는다.
@pytest.mark.parametrize(
    ("names", "message"),
    [
        (["cat_a", "../../t/run/cat_b"], "'../../t/run/cat_b', which is not a plain category name"),
        (["cat_a", "..\\..\\t\\run\\cat_b"], "which is not a plain category name"),
        (["cat_a", "C:cat_b"], "'C:cat_b', which is not a plain category name"),
        ([".."], "'..', which is not a plain category name"),
        ([""], "'', which is not a plain category name"),
        ([3], "3, which is not a plain category name"),
        (["cat_a", "cat_b", "cat_a"], "lists cat_a twice"),
        (["cat_a", "cat_b", "cat_c"], "cat_c is not a category of the manifest"),
    ],
)
def test_examples_take_only_plain_category_names_of_the_manifest(
    names, message, tmp_path, fake_data, npz_reads, capsys
):
    fake_data("dev")
    _write_examples_run(tmp_path / "t", protocol="test")  # a test run next to the dev run
    run_dir = _write_examples_run(tmp_path / "d")
    for suffix in (".npz", "_maps.npy"):  # cat_c: a copy of cat_a under a name the manifest does not have
        (run_dir / f"cat_c{suffix}").write_bytes((run_dir / f"cat_a{suffix}").read_bytes())
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    meta["categories"] = [{"category": name} for name in names]
    _write_json(run_dir / "run.json", meta)
    with pytest.raises(SystemExit) as stop:
        figures.main(["--examples", str(run_dir), "--out", str(tmp_path / "figs")])
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("examples not drawn: ") and message in err
    assert npz_reads == []  # refused before any .npz is opened, here or in the test run next to it


# 확인: --allow-test로 test 실행을 그리면 장부에 stage 5, config examples-<이름>, '그림용, 점수 없음' 설명 한
# 줄을 남기고, 이미지·마스크·맵 읽기가 모두 그 줄이 생긴 뒤에 일어난다(읽기는 이미지 1번, 마스크 1번, 맵 6번).
# 왜: 봉인 테스트를 읽을 때마다 먼저 장부에 남긴다는 규칙(docs/experiments.md 봉인 테스트)이 README 그림을
# 만들 때도 지켜져야 한다.
@pytest.mark.parametrize("fmt", ["common", "patchcore"])
def test_examples_of_a_test_run_write_the_ledger_line_before_reading_an_image(
    fmt, tmp_path, fake_data, reads, drawn
):
    fake_data("test")
    run_dir = _write_examples_run(tmp_path, fmt, protocol="test")
    figures.main(["--examples", str(run_dir), "--out", str(tmp_path / "figs"), "--allow-test"])
    lines = paths.TEST_LEDGER.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    name = "dm" if fmt == "common" else "p0"
    assert (entry["stage"], entry["config"], entry["commit"]) == ("5", f"examples-{name}", "abc1234")
    assert "figure only" in entry["note"] and "no scores" in entry["note"]
    assert [kind for kind, _, _ in reads] == ["images", "masks", *["map"] * 6]
    assert all(seen == lines for _, _, seen in reads)  # every read came after the line
    assert reads[0][1] == [image for image, _, _ in EXPECTED_EXAMPLES]
    ((_, kwargs, _),) = drawn
    assert kwargs["title"].endswith("on the VisA test split")


# 확인: 고른 이미지가 캐시 색인에 없으면 장부를 쓰기 전에 멈춘다(색인 조회만으로 확인). 실제로 이미지를 읽지
# 않았는데 읽었다는 장부 줄이 남지 않게 한다.
def test_examples_of_a_test_run_find_every_picked_image_in_the_cache_before_the_ledger_line(
    tmp_path, fake_data, reads, capsys
):
    rows = fake_data("test")
    missing = EXPECTED_EXAMPLES[0][0]
    index = "".join(f"{row.image}\n" for row in rows).replace(missing, "cat_a/Anomaly/xxx.JPG")
    for path in (cache.index_path(paths.CACHE, SIZE), cache.masks_index_path(paths.CACHE)):
        path.write_text(index, encoding="utf-8")
    run_dir = _write_examples_run(tmp_path, protocol="test")
    out = tmp_path / "figs"
    with pytest.raises(SystemExit) as stop:
        figures.main(["--examples", str(run_dir), "--out", str(out), "--allow-test"])
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("examples not drawn: ") and f"image not in the cache index: {missing}" in err
    assert reads == [] and not paths.TEST_LEDGER.exists() and not out.exists()


# 확인: 고른 행의 맵에 NaN이 있으면 그 파일과 'not all finite'를 알리며 멈추고 PNG를 쓰지 않는다. 맵은 결함
# 위치를 보여 주므로 이미지 다음에, 장부 줄 뒤에 읽는다. 그래서 test 실행은 이미 이미지를 읽었으므로 장부 줄이
# 남는 것이 맞다.
@pytest.mark.parametrize("protocol", ["dev", "test"])
def test_examples_refuse_a_picked_map_that_is_not_finite(protocol, tmp_path, fake_data, reads, capsys):
    fake_data(protocol)
    run_dir = _write_examples_run(tmp_path, protocol=protocol)
    maps = np.load(run_dir / "cat_a_maps.npy")
    maps[6, 10, 10] = np.nan  # cat_a/Anomaly/001.JPG, the first row of the panel
    np.save(run_dir / "cat_a_maps.npy", maps)
    out = tmp_path / "figs"
    with pytest.raises(SystemExit) as stop:
        figures.main(
            ["--examples", str(run_dir), "--out", str(out), *(["--allow-test"] * (protocol == "test"))]
        )
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("examples not drawn: ") and "cat_a_maps.npy" in err and "not all finite" in err
    assert not out.exists()
    # A map shows where the defect of its image is, so it is read after the images, behind the ledger line:
    # a test run that gets this far has read its images and keeps its line.
    assert [kind for kind, _, _ in reads] == ["images", "masks", "map"]
    assert paths.TEST_LEDGER.exists() == (protocol == "test")
    lines = paths.TEST_LEDGER.read_text(encoding="utf-8").splitlines() if protocol == "test" else []
    assert all(seen == lines for _, _, seen in reads) and len(lines) == (protocol == "test")


# 확인: 같은 실행으로 두 번 그리면 PNG가 바이트 단위로 같고 버전 메타데이터가 없다.
def test_examples_png_is_the_same_bytes_every_time(tmp_path, fake_data):
    fake_data("dev")
    run_dir = _write_examples_run(tmp_path)
    figures.main(["--examples", str(run_dir), "--out", str(tmp_path / "a"), "--name", "panel"])
    figures.main(["--examples", str(run_dir), "--out", str(tmp_path / "b"), "--name", "panel"])
    first = (tmp_path / "a" / "panel.png").read_bytes()
    assert first[:8] == b"\x89PNG\r\n\x1a\n" and (tmp_path / "b" / "panel.png").read_bytes() == first
    with Image.open(tmp_path / "a" / "panel.png") as image:
        assert "Software" not in image.info


# 이하 손상 함수들: 실행 폴더를 하나씩 망가뜨린다. _drop은 파일 하나를 지운다.
def _drop(run_dir, name):
    (run_dir / name).unlink()


# cat_b.npz에서 평가 점수 키를 뺀다.
def _no_eval_score(run_dir, name):
    with np.load(run_dir / "cat_b.npz") as z:
        arrays = {k: z[k] for k in z.files if k != "eval_score"}
    np.savez(run_dir / "cat_b.npz", **arrays)


# cat_b의 보정 점수를 비운다(임계값을 정할 수 없는 실행).
def _no_calibration(run_dir, name):
    with np.load(run_dir / "cat_b.npz") as z:
        arrays = {k: z[k] for k in z.files}
    arrays["cal_score"], arrays["cal_images"] = np.empty(0, np.float32), np.empty(0, np.str_)
    np.savez(run_dir / "cat_b.npz", **arrays)


# 맵 수를 평가 이미지 수보다 적게 만든다.
def _short_maps(run_dir, name):
    np.save(run_dir / "cat_a_maps.npy", np.zeros((3, SIZE, SIZE), dtype=np.float16))


# run.json에서 protocol을 지운다.
def _no_protocol(run_dir, name):
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    del meta["protocol"]
    _write_json(run_dir / "run.json", meta)


# protocol을 문자열이 아닌 값(리스트, dict)으로 바꾸는 손상 함수를 만든다.
def _protocol(value):
    def damage(run_dir, name):
        meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        meta["protocol"] = value
        _write_json(run_dir / "run.json", meta)

    return damage


# 파일을 200바이트로 자른다.
def _truncate(run_dir, name):  # a copy or a rewrite cut short
    (run_dir / name).write_bytes((run_dir / name).read_bytes()[:200])


# 파일을 빈 파일로 만든다.
def _empty(run_dir, name):
    (run_dir / name).write_bytes(b"")


# 보정 이미지 목록 키를 뺀다(어떤 이미지로 임계값을 정했는지 확인할 수 없게).
def _no_cal_images(run_dir, name):
    _rewrite(run_dir, "cat_b", lambda arrays: arrays.pop("cal_images"))


# 같은 평가 이미지를 한 번 더 넣는다.
def _eval_row_twice(run_dir, name):  # cat_a/Anomaly/000.JPG again, now detected
    _add_eval_rows(run_dir, "cat_a", ["cat_a/Anomaly/000.JPG"], 2.0)


# 확인: 불완전하거나 손상된 실행(파일 없음, 키 없음, 보정 점수 없음, 맵 수 불일치, 모르는 프로토콜, 잘린·빈
# 파일, 중복 평가 행)은 모두 원인을 담은 메시지와 코드 2로 끝나고, 이미지를 읽거나 출력 폴더를 만들지 않는다.
# 깨진 실행으로 틀린 그림이 만들어지지 않게 한다.
@pytest.mark.parametrize(
    ("damage", "name", "message"),
    [
        (_drop, "cat_b_maps.npy", "is incomplete: it has no cat_b_maps.npy"),
        (_drop, "cat_a.npz", "is incomplete: it has no cat_a.npz"),
        (_drop, "run.json", "has no run.json"),
        (_no_eval_score, "", "cat_b.npz has no eval_score"),
        (_no_calibration, "", "no calibration scores for 1 of 2 categories (cat_b)"),
        (_no_cal_images, "", "cat_b.npz has no cal_images"),
        (_short_maps, "", "expected 9 maps, one per evaluation image"),
        (_no_protocol, "", "names no known protocol: None"),
        (_protocol(["test"]), "", "names no known protocol: ['test']"),
        (_protocol({"test": 1}), "", "names no known protocol: {'test': 1}"),
        (_truncate, "cat_a.npz", "cat_a.npz cannot be read: BadZipFile"),
        (_empty, "cat_b.npz", "cat_b.npz cannot be read: EOFError"),
        (_empty, "cat_a_maps.npy", "cat_a_maps.npy cannot be read: EOFError"),
        (_eval_row_twice, "", "cat_a.npz lists cat_a/Anomaly/000.JPG twice in eval_images"),
    ],
)
def test_examples_of_an_incomplete_run_exit_with_code_2(
    damage, name, message, tmp_path, fake_data, reads, capsys
):
    fake_data("dev")
    run_dir = _write_examples_run(tmp_path)
    damage(run_dir, name)
    out = tmp_path / "figs"
    with pytest.raises(SystemExit) as stop:
        figures.main(["--examples", str(run_dir), "--out", str(out)])
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("examples not drawn: ") and message in err
    assert reads == [] and not out.exists()


# 확인: --allow-test와 --name은 --examples와만 쓸 수 있고, --examples와 --only는 함께 쓸 수 없으며, --name은
# 경로가 아닌 파일 이름이어야 한다. 잘못 조합하면 아무것도 쓰지 않고 코드 2로 끝난다.
def test_examples_options_only_go_with_examples(tmp_path, capsys):
    for argv in (
        ["--allow-test"],
        ["--name", "x"],
        ["--examples", str(tmp_path), "--only", "calibration"],
        ["--examples", str(tmp_path), "--name", "sub/x"],
    ):
        with pytest.raises(SystemExit) as stop:
            figures.main(["--reports", str(tmp_path), "--out", str(tmp_path / "out"), *argv])
        assert stop.value.code == 2
    assert not (tmp_path / "out").exists()


# 확인: matplotlib가 없는 환경에서도 figures 모듈을 import할 수 있다(별도 프로세스에서 sys.modules에 None을
# 넣어 흉내 낸다). matplotlib는 그림 함수 안에서만 import하므로, 모듈 docstring대로 이 모듈을 import하는 데는
# 기본 패키지만 필요하다(CI는 기본 패키지만 설치한다).
def test_the_module_imports_without_matplotlib():
    # `None` in sys.modules makes every `import matplotlib` fail, as where the group is not installed.
    code = (
        "import sys; sys.modules['matplotlib'] = None; "
        "from defect_inspect import figures; print(len(figures.FIGURES))"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "5"
