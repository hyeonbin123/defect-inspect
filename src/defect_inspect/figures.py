"""README.md의 그림을 커밋된 리포트 파일에서 다시 그린다.

    python -m defect_inspect.figures [--reports reports] [--out docs/figures] [--only NAME ...]
    python -m defect_inspect.figures --examples RUN_DIR [--allow-test] [--out docs/figures] [--name examples]

`fig_<이름>` 함수는 이미 읽은 리포트 데이터를 받아 matplotlib Figure를 돌려준다. `main`은 리포트 파일을 읽고,
입력이 없는 그림은 건너뛰고, 그림마다 PNG 하나를 쓴다. 리포트를 읽거나 그리지 못한 것이 있으면 다른 그림은
그대로 쓰고 끝에 오류로 끝난다. `--examples`를 주면 `main`은 평가 실행 하나가 저장한 점수와 맵으로 예시 검사
패널(`examples`)만 그리고, 그 실행으로 패널을 만들 수 없으면 코드 2로 끝난다. 여기서는 시계를 읽거나 난수를
뽑지 않고 PNG 메타데이터에 버전 문자열을 넣지 않으므로, 같은 리포트로 두 번 그리면 같은 바이트가 나온다.
matplotlib(`figures` 의존성 그룹)은 함수 안에서 import하므로, 이 모듈을 import하는 데는 기본 패키지만 있으면
된다.
"""

# [파일 역할] README 그림(docs/figures/*.png)을 커밋된 리포트 파일(reports/)에서 다시 그리는 모듈이다. 리포트
# 그림 다섯 장(calibration, label_curve, m2ad_conditions, perturb, grid)과, 저장된 평가 실행에서
# 검출·놓침·오검출 예를 고르는 예시 패널(examples-dinomaly, examples-patchcore)을 만든다.
# [왜 필요한가] README 수치와 그림이 같은 리포트에서 나와야 문서와 그림이 어긋나지 않는다. 그림을 손으로
# 고치지 않고 이 명령으로 언제든 다시 그린다(README 재현 명령의 'README 그림'). 커밋 aad3ce1에는 그려진 수치를
# 리포트와 572번 대조해 불일치가 없었다는 기록이 있다.
# [설계 이유] (1) 결정성: 시계·난수를 쓰지 않고 PNG 메타데이터에서 소프트웨어 버전을 빼서, 같은 리포트로 두 번
# 그리면 같은 바이트가 나온다(docstring). (2) pyplot 전역 상태 대신 Agg 캔버스에 직접 Figure를 만들어 GUI
# 백엔드가 필요 없다. (3) matplotlib은 figures 의존성 그룹이고 함수 안에서 import해서, 기본 패키지만으로 이
# 모듈을 import할 수 있다. (4) 예시 패널은 봉인 테스트 이미지를 읽을 수 있으므로 --allow-test와 장부 기록,
# 매니페스트 대조를 거친다.
# [흐름] 공통 스타일·도우미 → fig_* 그림 함수(리포트 dict → Figure) → FIGURES 표와 save → 예시
# 패널(pick_examples, examples) → main.
# [연결] 리포트는 analyze(1단계), compare(2단계), analyze_m2ad·analyze_perturb(3단계),
# analyze_grid·bench(4단계)가 만든다. 예시 패널은 compare의 load_common·load_patchcore로 점수를 읽고 cache로
# 이미지·마스크를 읽는다.

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

# 프로젝트 모듈에서 가져오는 것: 장부와 경로, 4단계 CPU 예산(CPU_BUDGET_MS), M2AD 기준 조건 이름,
# 이미지·마스크 캐시(MASK_SIZE = 256), conformal 임계값, compare의 실행 읽기 함수와 목표 α, 3단계 교란
# 정의(강도 표와 크기 환산), 구성 표(방법별 입력 크기), Dinomaly 입력 크기와 이름, 매니페스트와 봉인 역할.
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

# matplotlib 타입은 타입 힌트에만 쓰므로 실행 시에는 import하지 않는다(matplotlib 없이도 모듈 import 가능).
if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

# 그림 공통 크기·색: 150 DPI, 가로 12인치(1,800px). 흰 배경과 먹색 계열 글자·축·격자 색을 상수로 두어 모든
# 그림이 같은 모양이 되게 한다.
DPI = 150
WIDTH = 12.0  # 인치: DPI 150에서 1,800px
SURFACE = "#ffffff"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, WASH = "#e1e0d9", "#c3c2b7", "#f0efec"
SMALL = 8.5  # 패널 안 메모 글자 크기

# 방법마다 라벨·색·마커를 하나로 고정해 모든 그림에서 같은 방법이 같은 모양으로 나온다. 색각 이상에서도
# 구별되도록 네 색을 한 묶음으로 점검했고, 마커 모양으로도 색 없이 방법을 구별할 수 있다.
METHODS: dict[str, tuple[str, str, str]] = {
    "p0": ("PatchCore WRN-50", "#2a78d6", "o"),
    "d-s": ("PatchCore DINOv2 ViT-S", "#eb6834", "s"),
    "dm": ("Dinomaly", "#1baf7a", "^"),
}
# 지도 학습 헤드의 표식, 임계값 보정 방식 이름(재대입·홀드아웃·교차 적합), 절차 이름, 백본별 라벨·색(4단계
# 격자 그림용).
SUPERVISED = ("Supervised head", "#4a3aa7", "D")
STRATEGIES = {"resubstitution": "Resubstitution", "holdout": "Hold-out", "crossfit": "Cross-fit"}
PROTOCOLS = {"test": "VisA test split", "dev": "VisA dev split"}
BACKBONES = {"wrn50": ("WRN-50", "#2a78d6"), "dinov2_vits14": ("DINOv2 ViT-S", "#eb6834")}
_SPARE_COLOURS = ("#1baf7a", "#4a3aa7")  # BACKBONES에 없는 백본에 나오는 순서대로 주는 색
# 교란 종류의 축 라벨. 흐림 σ와 이동은 픽셀 단위다.
_KINDS = {
    "brightness": "brightness",
    "gamma": "gamma",
    "blur": "blur σ",
    "shift": "shift (px)",
    "jpeg": "JPEG quality",
}
# _SCALED: 입력 크기에 따라 강도가 바뀌는 교란(흐림 σ, 이동)을 conditions.strength로 직접 알아낸다(512px과
# 256px의 값이 다르면 크기 환산 대상). 이 교란의 눈금은 256 격자 값으로 보이고 각주에 다른 입력 크기의 실제
# 값을 적는다. 커밋 aad3ce1에서 읽기 쉽게 고친 부분이다.
_SCALED = tuple(
    kind
    for kind in LEVELS
    if strength(f"{kind}-1", 2 * REFERENCE_SIZE) != strength(f"{kind}-1", REFERENCE_SIZE)
)
# 3단계의 방법별 입력 크기(P0 256, D-S 448, Dinomaly 392). 3단계 교란은 이 크기로 줄인 이미지에 걸었다.
INPUT_SIZES = {**{name: cfg.img_size for name, cfg in CONFIGS.items()}, DINOMALY: DINOMALY_SIZE}
_GROUP_GAP = 0.6  # 조건 묶음 사이에 더 두는 간격(조건 한 칸 단위)

# matplotlib 설정 묶음. 글꼴은 matplotlib에 들어 있는 DejaVu Sans로 고정해 어느 PC에서나 같은 글리프로
# 그린다(그래야 PNG 바이트가 같다). 위·오른쪽 축선을 없애고 y축 격자만 둔다.
_RC = {
    "font.family": "DejaVu Sans",  # matplotlib에 들어 있어 어느 PC에서나 같은 글리프
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


# 그림이 실행 환경 설정에 따라 달라지지 않게 한다.
@contextlib.contextmanager
def _style() -> Iterator[None]:
    """사용자의 rc 파일이 무엇이든 matplotlib 기본값 + `_RC`로 그리고, 끝나면 이전 설정으로 되돌린다."""
    import matplotlib

    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        matplotlib.rcParams.update(_RC)
        yield


# 그림 함수를 _style() 안에서 실행하게 감싸는 데코레이터. 모든 fig_* 함수에 붙는다.
def _styled(func: Callable[..., Figure]) -> Callable[..., Figure]:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _style():
            return func(*args, **kwargs)

    return wrapper


# 테스트·서버 환경에서도 같은 결과가 나온다. layout="constrained"로 범례·제목이 겹치지 않게 자동 배치한다.
def _figure(height: float, width: float = WIDTH) -> Figure:
    """pyplot 없이 Agg 캔버스에 붙은 빈 Figure를 만든다(전역 상태도 GUI 백엔드도 없음)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, height), dpi=DPI, layout="constrained")
    FigureCanvasAgg(fig)
    return fig


# 그림 전체 제목을 왼쪽 정렬로 단다.
def _suptitle(fig: Figure, text: str) -> None:
    fig.suptitle(text, x=0.008, ha="left", fontsize=12, fontweight="bold")


def _method(name: str) -> tuple[str, str, str]:
    """방법의 (라벨, 색, 마커). 모르는 방법은 이름을 그대로 쓰고 중립 회색을 받는다."""
    return METHODS.get(name, (name, MUTED, "o"))


# 키를 부동소수 비교(isclose)로 찾으므로 문자열 표기 차이("0.05"와 "0.050")에 흔들리지 않는다.
def _at_alpha(block: dict, alpha: float):
    """키가 문자열로 적힌 목표 비율("0.05")인 리포트 블록에서 `alpha`의 항목을 찾는다."""
    for key, value in block.items():
        if math.isclose(float(key), alpha):
            return value
    raise KeyError(f"no entry for the target rate {alpha} among {list(block)}")


def _signed(x: float, digits: int = 1) -> str:
    """부호를 붙인 수 `+1.5` 또는 `−1.5`. 마이너스는 축 눈금과 같은 유니코드 기호(−)다."""
    return f"{x:+.{digits}f}".replace("-", "−")


# 라벨은 'target 5%' 같은 글이다.
def _target(ax: Axes, alpha: float, *, outside: bool = False, label: bool = True) -> None:
    """목표 오검출률에 점선을 긋고(축은 퍼센트 단위) 오른쪽 끝에 라벨을 단다."""
    y = 100 * alpha
    ax.axhline(y, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1.5, gid="target")
    if not label:
        return
    # 라벨은 y축 변환(x는 축 비율, y는 데이터 값)으로 오른쪽 끝에 붙이고, outside면 축 밖에 둔다.
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


# 구간은 95% 구간이다. gid를 붙여 테스트가 그려진 요소를 찾아 값이 리포트와 같은지 확인할 수 있게 한다.
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
    """점(gid `gid`)과 그 구간 세로선(gid `gid:ci`)을 그린다. 비율(0~1)은 퍼센트로 바꿔 그린다."""
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
    """한 x 위치를 가운데로 두고 n개 계열을 나란히 놓는 좌우 간격."""
    return (np.arange(n) - (n - 1) / 2) * step


# clean, 기준 조명(I01), 실제 조명(R:<id>), 합성 교란(종류-강도)을 구별한다.
def _condition(name: str) -> tuple[str, str]:
    """교란·M2AD 리포트의 조건 이름을 (묶음 라벨, 눈금 라벨)로 바꾼다.

    입력 크기에 맞게 환산한 강도는 256px 격자 값으로 보이고, 그 묶음에는 `*`를 붙인다.
    """
    if name == CLEAN:
        return "clean", ""
    if name == M2AD_BASE:
        return "reference", M2AD_REFERENCE  # 임계값을 정한 이미지의 조명
    if name.startswith("R:"):
        return "real lighting", name[2:]
    # 합성 교란 이름('blur-2' 또는 M2AD의 'P:blur-2')을 종류와 강도로 나눠, 눈금에는 강도 표의 256 격자 값을
    # 적는다.
    kind, _, level = name.removeprefix("P:").rpartition("-")
    if kind in LEVELS and level in {"1", "2", "3"}:
        mark = "*" if kind in _SCALED else ""
        return f"{_KINDS.get(kind, kind)}{mark}", f"{LEVELS[kind][int(level) - 1]:g}"
    return name, ""


# 다른 입력 크기의 실제 값(σ 배율, 이동 픽셀)은 conditions.strength로 계산한다. 예를 들어 448px D-S의 1px
# 이동은 반올림으로 2px이 된다.
def _scaled_note(methods: Sequence[str]) -> str:
    """`*` 묶음의 각주: 강도는 256px 격자 값이고, 다른 입력 크기는 실제로 얼마를 받았는지 적는다."""
    scaled = [m for m in methods if INPUT_SIZES.get(m) != REFERENCE_SIZE]
    at_256 = [_method(m)[0] for m in methods if m not in scaled]
    note = f"* blur σ and shift in pixels of a {REFERENCE_SIZE} px image"
    if at_256:
        note += f", the input of {' and '.join(at_256)}"
    if not scaled:
        return note
    # 입력 크기가 256이 아닌 방법마다 σ 배율과 강도 1~3의 실제 이동 픽셀을 계산해 적는다.
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


# 묶음 사이 간격은 _GROUP_GAP이다.
def _condition_axis(
    ax: Axes, names: Sequence[str], *, labels: bool = True, methods: Sequence[str] = ()
) -> np.ndarray:
    """x축에 조건을 묶음별로 놓는다. 묶음 사이에는 간격과 가는 세로선을 넣는다.

    눈금에는 강도(또는 조명 id)를 적고, 묶음 이름은 그 묶음 가운데 눈금 아래에 둔다. 256px 격자 값(`*`)인
    묶음이 있으면 x축 라벨에 `methods`의 입력 크기가 실제로 얼마를 받았는지 적는다. 각 조건의 x 위치를
    돌려준다.
    """
    # 묶음이 바뀔 때마다 위치에 간격을 더하고 경계선을 긋는다. groups는 묶음별 조건 인덱스 목록이다.
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
    # 모든 눈금 라벨을 두 줄로 맞춘다. 한 줄짜리 라벨은 다른 라벨보다 조금 위에 놓여 줄이 어긋나 보이기
    # 때문이다.
    ticks = [f"{tick}\n " for _, tick in parsed]
    # 묶음 가운데 조건의 눈금에만 둘째 줄로 묶음 이름을 넣는다.
    for members in groups:
        middle = members[len(members) // 2]
        ticks[middle] = f"{parsed[middle][1]}\n{parsed[middle][0]}"
    ax.set_xticks(x, ticks if labels else [""] * len(x))
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(x[0] - 0.75, x[-1] + 0.75)
    if labels and any(group.endswith("*") for group, _ in parsed):
        ax.set_xlabel(_scaled_note(methods), loc="left", fontsize=SMALL, labelpad=6)
    return np.asarray(x)


# 범례 항목 이름은 '95% bootstrap interval'이다(커밋 aad3ce1: 부트스트랩 수염을 범례에 이름 붙임).
def _interval_handle() -> Line2D:
    """점을 지나는 세로선(구간)이 무엇인지 밝히는 범례 항목: 세로선 모양의 빈 선."""
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


# 다르면 같은 x 위치에 다른 조건이 그려지므로 멈춘다.
def _same_conditions(tables: dict[str, list[dict]]) -> list[str]:
    """모든 방법의 행이 같은 순서로 공유하는 조건 목록을 돌려준다(다르면 ValueError)."""
    listed = {method: [row["condition"] for row in rows] for method, rows in tables.items()}
    names = next(iter(listed.values()))
    if any(mine != names for mine in listed.values()):
        raise ValueError(f"the methods list different conditions: {listed}")
    return names


# 그림 calibration.png(README '임계값을 정상 이미지만으로 정하는 세 가지 방법'). 왼쪽 패널의 재대입은 86%다.
@_styled
def fig_calibration(report: dict, alpha: float = 0.05) -> Figure:
    """목표 비율 하나에서 세 임계값 방식의 실제 합산 오검출률을 그린다.

    입력: 1단계 리포트(`analyze.build_report`). 패널 셋: 세 방식 모두를 0~100% 축에, 홀드아웃과 교차
    적합을 확대해 홀드아웃 임계값에 대해 이론이 기대하는 범위(`theory_band`)와 함께, 그리고 보정 표본 수에
    따른 비율(`finite_sample`)을 교환 가능한 점수로 모의한 구간 옆에.
    """
    # 리포트에서 목표 α의 보정 결과(방식별 실제 오검출률과 구간)와 보정 표본 수 곡선(finite_sample)을 꺼낸다.
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

    # 막대(실제 오검출률), 부트스트랩 구간 수염, 값 라벨을 그리는 패널 공용 도우미. x 라벨의 n은 12개 범주
    # 보정 점수의 합이다.
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
        # `n_cal_total`은 범주를 모두 더한 값이다. 오른쪽 패널의 n은 범주당 값이다.
        ax.set_xlabel("n: calibration scores of all categories together")
        _target(ax, alpha, outside=True)

    # 왼쪽 패널: 세 방식을 0~100% 축에. 재대입(86%)과 나머지 둘의 차이를 한눈에 보인다.
    bars(ax_all, list(STRATEGIES), 1)
    ax_all.set_ylim(0, 100)
    ax_all.set_title("Three ways to set the threshold")
    ax_all.set_ylabel("pooled false-alarm rate (%)")

    # 가운데 패널: 홀드아웃과 교차 적합만 확대해 소수 둘째 자리까지 적는다.
    zoomed = ["holdout", "crossfit"]
    bars(ax_zoom, zoomed, 2)
    # 홀드아웃 막대 옆에 이론 구간(교환 가능하면 실제 오검출률이 들어와야 하는 95% 범위, experiments.md
    # '임계값 보정')을 따로 그린다. README: 실제 5.6%가 이론 구간 3.6~6.3% 안이다.
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
    # 막대의 수염(부트스트랩 구간)에도 범례 이름을 붙여 이론 구간 선과 구별해, 하나를 다른 것으로 읽지 않게
    # 한다.
    whisker = _interval_handle()
    whisker.set(color=INK, label="on the bars: 95% bootstrap interval")
    ax_zoom.legend(
        handles=[whisker, *ax_zoom.get_legend_handles_labels()[0]],
        loc="upper left",
        handlelength=1.2,
        borderaxespad=0.2,
    )

    # 오른쪽 패널: 보정 표본 n마다 관측한 범주 평균 오검출률과, 같은 표본 크기로 모의한 기준 구간·평균. 비교
    # 기준을 Beta 분위가 아니라 같은 크기의 모의로 바꾼 이유는 규칙 변경 2에 있다(부분표본이라 관측 폭이
    # Beta보다 좁다).
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


# 그림 label_curve.png(README '결함 라벨 몇 장부터 지도 학습이 이기는가'). 오른쪽 패널의 기준 방법은
# Dinomaly이고, 그 값이 H7의 근거다.
@_styled
def fig_label_curve(report: dict) -> Figure:
    """범주당 결함 라벨 수 k에 따른 지도 학습 헤드의 평균 이미지 AUROC를 그린다.

    입력: 2단계 비교 리포트(`compare.build_report`). 왼쪽: 지도 학습의 시드 평균과 구간, 시드별 값을,
    가로선과 구간 띠로 그린 비지도 방법들 위에. 오른쪽: 결함 유형이 라벨에 없던 결함에서 지도 학습과 기준
    방법을 비교한 값을, 그 비교를 판정한 k마다.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # 지도 학습 결과를 k 순으로 정렬하고, 기준 비지도 방법 이름을 꺼낸다.
    rows = sorted(report["supervised"].values(), key=lambda row: row["k"])
    reference = report["reference_unsupervised"]
    sup_label, sup_colour, sup_marker = SUPERVISED
    fig = _figure(4.9)
    ax, ax_unseen = fig.subplots(1, 2, width_ratios=[1.85, 1.0])
    protocol = PROTOCOLS.get(report.get("protocol"), report.get("protocol"))
    _suptitle(fig, f"How many defect labels until supervised training wins ({protocol})")

    x = np.arange(len(rows))
    handles, extent = [], []
    # 비지도 방법은 k와 무관하므로 가로선(점 추정)과 반투명 띠(95% 구간)로 그리고, 오른쪽 끝에 값을 적는다.
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

    # 시드별 값은 평균 점 오른쪽에 조금씩 비켜 작은 점으로 찍어, 시드 사이 흔들림을 보여 준다.
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
    # 시드 평균과 95% 구간을 선으로 이어 k에 따른 추세를 보인다.
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
    # x 눈금은 k와 괄호 안의 '검증 포함 라벨 수'(k + 20). 라벨 비용은 검증 20장까지 센다(experiments.md '라벨
    # 회계').
    ax.set_xticks(x, [f"{row['k']}\n({row['labels_with_validation']})" for row in rows])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.45, len(rows) - 0.55)
    ax.set_ylim(math.floor(min(extent) - 0.5), min(100.0, math.ceil(max(extent) + 0.5)))
    ax.set_xlabel("defect labels per category k (in brackets: with the labels used for validation)")
    ax.set_ylabel("macro image AUROC (%)")
    ax.set_title("All test defects")
    n_seeds = len(rows[0]["seeds"])
    # 범례는 그림 아래에 한 줄로: 지도 학습(시드 평균·구간·시드 점)과 비지도 방법들(선 + 띠).
    fig.legend(
        [line, *[(h[0], h[1]) for h in handles]],
        [f"{sup_label}: mean of {n_seeds} seeds, 95% CI (small dots: seeds)", *[h[2] for h in handles]],
        loc="outside lower left",
        ncols=4,
        handlelength=1.6,
        columnspacing=1.6,
    )

    # 오른쪽 패널: 판정된(남은 결함이 충분한) k만 그린다. 모든 k가 판정 불가면 '너무 적다'는 안내만 적는다(H7
    # 규칙: 남은 결함 합이 30장 미만인 k는 판정하지 않음).
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
    # 판정된 k마다 기준 비지도와 지도 학습 점을 세로선으로 잇는다.
    xu = np.arange(len(judged))
    sup = np.array([100 * row["unseen_types"]["supervised"] for row in judged])
    ref = np.array([100 * row["unseen_types"]["reference"] for row in judged])
    ax_unseen.vlines(xu, np.minimum(sup, ref), np.maximum(sup, ref), color=AXIS, linewidth=1.6, zorder=2)
    common = {"linestyle": "none", "markersize": 8, "markeredgecolor": SURFACE, "markeredgewidth": 0.9}
    ax_unseen.plot(xu, ref, marker=ref_marker, color=ref_colour, zorder=4, gid="unseen:reference", **common)
    ax_unseen.plot(xu, sup, marker=sup_marker, color=sup_colour, zorder=4, gid="unseen:supervised", **common)
    # 두 점 사이에 차이(지도 − 기준, %p)와 95% 구간을 적는다. README: k = 5에서 −3.5%p, k = 10에서 −2.1%p.
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
    # 두 마커의 이름은 첫 k에서 한 번만 적는다.
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
    # 시드마다 학습에 쓴 결함이 달라 '못 본 유형' 부분집합의 크기도 시드마다 다르다. 그래서 장수는 시드
    # 평균이고, 정수가 아니면 소수 한 자리로 적는다.
    counts = [_mean_count(row["unseen_types"]["unseen_defects_mean"]) for row in judged]
    ax_unseen.set_xticks(xu, [f"k = {row['k']}\n{n} defects" for row, n in zip(judged, counts, strict=True)])
    ax_unseen.set_xlabel(f"defects of these types: mean over the {n_seeds} seeds")
    ax_unseen.set_xlim(-1.2, len(judged) - 0.05)
    span = float(max(sup.max(), ref.max()) - min(sup.min(), ref.min())) or 1.0
    ax_unseen.set_ylim(min(sup.min(), ref.min()) - 0.35 * span, max(sup.max(), ref.max()) + 0.35 * span)
    return fig


def _mean_count(value: float) -> str:
    """개수의 평균을 문자열로: 정수면 `60`, 아니면 `116.7`. 개수처럼 보이게 정수로 반올림하지 않는다."""
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"


# 평균 점 오른쪽으로 0.10부터 조금씩 비킨다.
def _seed_dots(row: dict) -> list[tuple[float, float]]:
    """k 하나의 시드별 (x 비킴, AUROC). 평균 점에서 오른쪽으로 조금씩 비켜 놓는다."""
    values = row["supervised_per_seed"]
    return [(0.10 + 0.045 * i, v) for i, v in enumerate(values)]


# 그림 m2ad_conditions.png(README '조명이 실제로 바뀌면'). 합성 교란은 15조건, 실제 조명은 9조건, 재보정 장수
# n은 8과 30이다.
@_styled
def fig_m2ad_conditions(report: dict, alpha: float = 0.05, methods: Sequence[str] = ("p0", "d-s")) -> Figure:
    """M2AD 검사기들의 조건별 오검출률과 재보정 뒤의 오검출률을 그린다.

    입력: 3단계 M2AD 리포트(`analyze_m2ad.build_report`). `alpha`는 임계값을 정한 목표 비율이다(리포트에
    저장되지 않아 인자로 받는다). 왼쪽: 기준 조명, 합성 조건, 실제 조명마다 방법별 점 하나와 구간. 오른쪽:
    새 조명의 정상 n장으로 재보정한 뒤 실제 조명들을 합친 비율.
    """
    from matplotlib.lines import Line2D

    # 리포트에 있는 방법만 그리고, 방법들의 조건 목록이 같은지 확인한다.
    runs = {m: report["methods"][m] for m in methods if m in report["methods"]}
    if not runs:
        raise ValueError(f"the report has none of the methods {list(methods)}")
    names = _same_conditions({m: run["conditions"] for m, run in runs.items()})
    fig = _figure(4.9)
    # 왼쪽 패널은 조건이 25개라 눈금 라벨('1.25' 등)이 옆과 띄어 들어가도록 넓게 잡는다.
    ax, ax_recal = fig.subplots(1, 2, width_ratios=[4.4, 1.0])
    inspectors = len(next(iter(runs.values()))["inspectors"])
    _suptitle(fig, f"M2AD: false alarms when the capture condition changes ({inspectors} inspectors)")

    # 왼쪽 패널: 조건 25개(기준, 합성 15, 실제 조명 9)를 묶음별로 놓고 방법마다 조금 비켜 점과 구간을 찍는다.
    x = _condition_axis(ax, names, methods=list(runs))
    handles = []
    # 한 조건에서 두 방법의 마커가 겹치지 않을 만큼 띄운다.
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
    # 조건 칸이 좁아 선 끝 라벨 대신 범례에 목표선을 넣는다. 선 끝 라벨은 조건 칸의 자리를 빼앗는다.
    _target(ax, alpha, label=False)
    handles += [
        _interval_handle(),
        Line2D([], [], color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)), label=f"target {100 * alpha:g}%"),
    ]
    # 오검출률은 0~100%라 축을 조금 넉넉히 고정해 0%와 100% 점이 잘리지 않게 한다.
    ax.set_ylim(-3, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("false-alarm rate on normal images (%)")
    ax.set_title(
        f"Thresholds fixed on the reference lighting {M2AD_REFERENCE}: "
        "synthetic changes (by strength), real lightings (by id)"
    )
    fig.legend(handles=handles, loc="outside lower left", ncols=len(handles), handlelength=1.2)

    # 오른쪽 패널: 재보정 장수 n별 점. 두 방법의 n 목록이 다르면 같은 x에 다른 조건이 그려지므로 멈춘다. 점
    # 위에 값을 적는다.
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
    # symlog 축: 1% 아래는 선형, 위는 로그라서 97%와 4%가 한 축에서 함께 읽히고 0%도 자리가 있다. 재보정
    # 전후 오검출률이 수십 배 차이 나기 때문이다(README: 96.9% → 4.4%).
    ax_recal.set_yscale("symlog", linthresh=1.0, linscale=0.35)
    # 0 아래에는 비율 0인 마커가 통째로 들어갈 자리를, 100% 위에는 마커 위에 적는 값의 자리를 둔다.
    ax_recal.set_ylim(-0.3, 180)
    ax_recal.set_yticks([0, 1, 2, 5, 10, 20, 50, 100], ["0", "1", "2", "5", "10", "20", "50", "100"])
    ax_recal.minorticks_off()
    _target(ax_recal, alpha, label=False)  # 눈금 5와 범례가 이 선의 이름을 대신한다
    ax_recal.set_xticks(xr, [str(n) for n in sizes])
    ax_recal.tick_params(axis="x", length=0)
    ax_recal.set_xlim(-0.75, len(sizes) - 0.25)
    ax_recal.set_xlabel("recalibration images (n)")
    ax_recal.set_ylabel("pooled rate, real lightings (%, log scale)")
    ax_recal.set_title("After recalibration")
    return fig


# 그림 perturb.png(README '촬영 조건이 조금 바뀌면'). H9가 지목한 조건은 AUROC가 1%p 미만으로 떨어지는데
# 오검출률은 목표의 두 배 이상인 것이다. 'AUROC로 감시하면 이 고장은 보이지 않는다'(README)를 한눈에 보이려는
# 그림이다.
@_styled
def fig_perturb(report: dict) -> Figure:
    """합성 교란 조건에서의 정확도 변화와 고정 임계값의 오검출률을 그린다.

    입력: 3단계 교란 리포트(`analyze_perturb.build_report`). 위: 교란 없는 조건 대비 평균 이미지 AUROC
    변화(%p). 아래: 교란 없는 보정 점수로 고정한 임계값에서의 합산 오검출률(보정 점수가 없는 방법은 이
    패널에 점이 없다). H9가 지목한 조건(정확도는 거의 그대로, 오검출률은 목표의 두 배 이상)은 옅은 배경으로
    칠한다.
    """
    from matplotlib.patches import Patch

    # 방법별 조건 표, 목표 α, 조건 목록을 꺼낸다. 두 패널은 x축을 같이 쓴다.
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
    # 방법마다: 위 패널에는 AUROC 변화(교란 없는 조건은 변화가 없어 제외), 아래 패널에는 고정 임계값의
    # 오검출률을 그린다.
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
        # 보정 점수가 없는 방법은 오검출률 칸이 None이라 아래 패널에 점을 찍지 않는다.
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
    # H9 판정이 지목한 조건을 두 패널에 같은 배경으로 칠하고 범례에 기준(AUROC 1점 미만 하락, 오검출 2α
    # 이상)을 적는다.
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
    # 위 패널에 변화 0 기준선을 긋는다.
    ax_auc.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax_auc.margins(y=0.12)  # 0에 있는 마커가 패널 가장자리에 붙지 않게 한다
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
    """설정 이름을 나눈다: `wrn50-256` -> (백본, 입력 크기)."""
    backbone, _, size = setting.rpartition("-")
    return (backbone, size) if backbone else (setting, "")


# 그림 grid.png(README 'CPU 200ms 안에 넣으려면'). 예산은 200ms다.
@_styled
def fig_grid(grid: dict, latency: dict | None = None) -> Figure:
    """격자의 모든 구성에서 장당 CPU 지연과 dev 평균 이미지 AUROC를 점으로 그린다.

    입력: 4단계 격자 리포트(`analyze_grid`)와 지연 파일(`bench`). 서빙 선택이 격자 행에 합쳐진 지연으로
    이루어졌으므로 그 값을 먼저 쓰고, 지연 파일은 리포트에 지연이 없는 행과 정밀도만 채운다. (설정, 코어셋
    비율, 정밀도)마다 점 하나: x는 리사이즈 + 추론으로, FP32는 행에 등록된 `cpu_ms`, 그 밖에는 측정의
    `total_with_resize_ms`(없으면 `resize_ms` + `total_ms`)다. y는 행의 AUROC인데 torch 파이프라인이 채점한
    값이라, INT8 점은 지연 이득만 보여 준다. CPU 지연이 없는 행은 빠진다. 예산은 세로선, 고른 서빙 구성은
    동그라미로 표시한다.
    """
    from matplotlib.lines import Line2D

    # 점 목록 만들기: 지연 파일 항목에 격자 행 자체의 latency를 덮어써(행 값 우선) 정밀도별(cpu_fp32,
    # cpu_int8) 지연을 모은다. FP32는 등록된 cpu_ms, 나머지는 리사이즈 포함 합계를 쓴다.
    points = []  # (설정, 비율, 정밀도, ms, AUROC %, 키)
    for row in grid["rows"]:
        entry = {**(latency or {}).get(row["key"], {}), **(row.get("latency") or {})}
        for name, timing in entry.items():
            if not name.startswith("cpu_") or not isinstance(timing, dict):
                continue  # "cpu"는 프로세서 이름, "gpu_fp16"은 다른 장치
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
    # 예산은 리포트의 서빙 선택 값(없으면 analyze_grid의 CPU_BUDGET_MS = 200ms)을 쓴다.
    serving = grid.get("serving")
    budget = float(serving["budget_ms"]) if serving else CPU_BUDGET_MS
    protocol = PROTOCOLS.get(grid.get("protocol"), grid.get("protocol"))

    fig = _figure(5.4)
    ax = fig.subplots()
    _suptitle(fig, f"PatchCore configurations: CPU latency against accuracy ({protocol})")
    # 코어셋 비율은 마커 크기(큰 비율이 큰 점), 백본은 색, 같은 백본 안의 입력 크기는 마커 모양, 정밀도는
    # 채움(FP32 채움, INT8 속 빈 점선)으로 구별한다.
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

    # 설정마다 정밀도별로 비율 순서대로 점을 선으로 잇는다. 같은 설정의 코어셋 10%·1%·0.1%가 한 선 위에 놓여,
    # 뱅크를 줄일 때 지연과 정확도가 함께 어떻게 움직이는지 보인다.
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
        # 설정 이름은 그 설정에서 가장 느린 FP32 점(FP32가 없으면 아무 점) 옆에 적고, 그 점이 서빙
        # 동그라미면 라벨을 조금 더 띄운다.
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

    # 예산 200ms 세로 점선과 라벨.
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
    # 고른 서빙 구성(README: wrn50-256 코어셋 1%, dev 뱅크 184ms)을 큰 빈 동그라미로 감싸고 지연과 AUROC를
    # 적는다.
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
    # 지연이 84ms~2,326ms로 넓어 x축을 로그로 한다(experiments.md 4단계 표). 보조 눈금 2, 3, 5에도 숫자를 적어
    # 읽기 쉽게 한다.
    ax.set_xscale("log")
    ms = [p[3] for p in points] + [budget]
    ax.set_xlim(min(ms) / 1.35, max(ms) * 2.1)
    ax.xaxis.set_major_formatter("{x:g}")
    ax.xaxis.set_minor_formatter(_log_minor_label)
    ax.tick_params(axis="x", which="both", length=3)
    ax.set_xlabel("CPU latency per image: resize + inference (ms, log scale)")
    ax.set_ylabel("macro image AUROC (%)")
    ax.margins(y=0.12)

    # 범례: 코어셋 비율별 마커 크기, 그리고 정밀도가 둘 이상이면 FP32와 INT8('지연만, 정확도는 다시 재지
    # 않음') 표식.
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
    """로그축 보조 눈금 포매터: 각 십진 구간의 2, 3, 5에만 숫자를 적고 나머지는 비워 둔다."""
    if x <= 0:
        return ""
    mantissa = x / 10 ** math.floor(math.log10(x))
    return f"{x:g}" if any(math.isclose(mantissa, m) for m in (2, 3, 5)) else ""


# 예시 패널 그림이다. 호출 위치: examples().
@_styled
def fig_examples(rows: list[dict], *, title: str | None = None, caption: str | None = None) -> Figure:
    """예시마다 패널 한 줄을 그린다: 이미지, 정답 윤곽, 그 위에 겹친 점수 맵.

    각 행에는 `image`(HxWx3 uint8), `mask`(HxW bool, None이면 윤곽을 그리지 않음), `map`(hxw 유한 float,
    이미지 위로 늘림), `title`, `threshold`(float 또는 None)가 있다. 한 줄의 색 범위는 그 맵과 임계값을 함께
    덮는다: 임계값 아래에 머무는 맵은 밝아지지 않고, 모든 곳에서 넘는 맵은 모든 곳이 밝아진다. 임계값은 색
    막대에 표시하고, 맵이 임계값을 가로지르는 곳에 등고선으로 그린다. `title`은 그림 위에, `caption`은 그림
    아래에 작은 글자로 넣는다.
    """
    from matplotlib import patheffects

    if not rows:
        raise ValueError("need at least one example")
    # 예시 수만큼 줄을 만든다(줄당 높이 3.3인치, 세 칸).
    fig = _figure(3.3 * len(rows) + 0.1, width=10.4)
    if title:
        _suptitle(fig, title)
    if caption:
        fig.supxlabel(caption, x=0.008, ha="left", fontsize=SMALL, color=INK_2, linespacing=1.5)
    grid = fig.subplots(len(rows), 3, squeeze=False)
    headers = ("", "ground-truth outline", "anomaly map")
    outline = [patheffects.withStroke(linewidth=3.2, foreground=INK)]
    # 줄마다 입력을 점검한다: 맵은 유한한 2차원 배열, 마스크는 이미지와 같은 크기.
    for line, row in zip(grid, rows, strict=True):
        image = np.asarray(row["image"])
        scores = np.asarray(row["map"], dtype=np.float64)
        height, width = image.shape[:2]
        if scores.ndim != 2 or not np.isfinite(scores).all():
            raise ValueError(f"the map of {row['title']!r} must be a 2-d array of finite scores")
        if row["mask"] is not None and np.shape(row["mask"]) != (height, width):
            raise ValueError(f"the mask of {row['title']!r} is not the size of its image {(height, width)}")
        # imshow의 extent를 픽셀 중심 좌표로 맞춰, 해상도가 다른 맵(예: 256)을 이미지 위에 정확히 늘려 겹친다.
        extent = (-0.5, width - 0.5, height - 0.5, -0.5)
        # 색 범위(vmin, vmax)는 맵의 최소·최대와 임계값을 모두 포함하게 잡는다. 범위가 0이면 1을 더해 나눗셈이
        # 깨지지 않게 한다.
        threshold = row.get("threshold")
        low, high = float(scores.min()), float(scores.max())
        if threshold is not None:
            threshold = float(threshold)
        vmin, vmax = (low, high) if threshold is None else (min(low, threshold), max(high, threshold))
        if vmax <= vmin:
            vmax = vmin + 1.0
        # 세 칸 모두 같은 이미지를 바탕으로 깔고 축을 숨긴다. 첫 칸 제목은 예시 제목, 나머지는 칸 이름.
        for ax, header in zip(line, headers, strict=True):
            ax.imshow(image, extent=extent, interpolation="nearest")
            ax.set_axis_off()
            if header:
                ax.set_title(header, fontweight="normal", color=INK_2, fontsize=9.5)
            else:
                ax.set_title(row["title"])
        # 가운데 칸: 결함 이미지는 마스크 0.5 등고선으로 정답 윤곽을 흰 선(검은 테두리)으로 그리고, 정상
        # 이미지는 '마스크 없음'이라고 적는다.
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
        # 점수가 낮은 곳은 투명하게 두어 이상으로 표시되지 않은 곳에서는 원본 이미지가 보이게 한다. 불투명도는
        # 정규화한 점수에 비례(최대 0.75)한다.
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
        # 색 막대는 불투명하게 그려(맵의 투명도와 무관하게) 점수 눈금을 읽을 수 있게 한다.
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
        # 색 막대에 임계값 선을 긋고, 맵이 임계값을 가로지르면 그 등고선을 맵 위에 그린다. 등고선 좌표는 맵
        # 칸의 중심이 이미지 좌표에 오도록 계산한다.
        if threshold is not None:
            bar.ax.axhline(threshold, color=SURFACE, linewidth=2.6)
            bar.ax.axhline(threshold, color=INK, linewidth=1.2, gid="threshold")
            # 등고선을 그리려면 맵이 적어도 2 x 2 점수여야 한다.
            if low < threshold < high and min(scores.shape) >= 2:
                ys = np.linspace(-0.5, height - 0.5, scores.shape[0] * 2 + 1)[1::2]
                xs = np.linspace(-0.5, width - 0.5, scores.shape[1] * 2 + 1)[1::2]
                flagged = line[2].contour(
                    xs, ys, scores, levels=[threshold], colors=[SURFACE], linewidths=1.2
                )
                flagged.set_path_effects(outline)
    return fig


# 리포트 그림 표: 그림 이름 → (reports/ 아래 입력 파일들, 그 파일들을 순서대로 받는 그림 함수). main이 이 표를
# 돌며 입력이 있는 그림만 그린다.
FIGURES: dict[str, tuple[tuple[str, ...], Callable[..., Figure]]] = {
    "calibration": (("stage1/p0-test.json",), fig_calibration),
    "label_curve": (("stage2/compare-test.json",), fig_label_curve),
    "m2ad_conditions": (("stage3/m2ad.json",), fig_m2ad_conditions),
    "perturb": (("stage3/perturb-test.json",), fig_perturb),
    "grid": (("stage4/grid-dev.json", "stage4/latency.json"), fig_grid),
}


# metadata의 Software를 None으로 지워 matplotlib 버전 문자열이 들어가지 않게 한다(같은 리포트 → 같은 파일).
def save(fig: Figure, path: Path) -> Path:
    """그림을 바이트가 그림 내용에만 의존하는 PNG로 쓴다(소프트웨어 버전을 넣지 않음)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _style():
        fig.savefig(path, format="png", dpi=DPI, metadata={"Software": None})
    return path


# 리포트 JSON을 읽는다.
def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---- 평가 실행의 예시 검사 -------------------------------------------------------------------------------

# 예시 패널 설정: 세 묶음(검출한 결함, 놓친 결함, 오검출) × 묶음당 2장. 봉인 테스트 실행이면 장부에 stage
# "5"(README·그림 단계)와 '점수 없이 그림용 이미지·마스크만 읽음' 메모로 남긴다(reports/test_ledger.jsonl의
# examples-dm, examples-p0 줄). 이미지 출처 표기(VisA, CC BY 4.0)는 라이선스 조건이라 캡션에 넣는다.
EXAMPLE_GROUPS = ("detected defect", "missed defect", "false alarm")
PER_GROUP = 2  # 묶음당 줄 수
EXAMPLES_STAGE = "5"  # 테스트 장부 줄에 적는 단계 이름
EXAMPLES_NOTE = "README example panel: images and masks of the picked rows read for a figure only, no scores"
DATA_CREDIT = "Images: VisA (Zou et al., ECCV 2022), CC BY 4.0"
# 캡션에 적을 임계값 출처, 곧 `compare`가 실행 종류마다 보정 점수를 가져오는 곳: PatchCore 실행은 정상 풀의
# 교차 적합 점수, Dinomaly(공통 형식) 실행은 빼 둔 정상의 점수.
_CALIBRATION_SOURCE = {
    "crossfit": "the cross-fitted (out-of-fold) scores of the normal pool",
    "holdout": "the scores of held-out normal images",
}
# 예시를 고르는 데 필요한 평가 배열 키와, 제목·캡션 줄바꿈 폭.
_EVAL_KEYS = ("eval_images", "eval_labels", "eval_defect_types", "eval_score")
_TITLE_CHARS = 34  # 줄 제목이 그 줄의 첫 패널 위에 머물도록 줄바꿈하는 폭
_TITLE_LINES = 3  # 이보다 길면 결함 유형 목록을 줄인다("+ 3 more")
_CAPTION_CHARS = 165


# main에서 종료 코드 2로 바뀐다.
class ExamplesError(ValueError):
    """실행 폴더로 예시 패널을 만들 수 없다: 불완전하거나, 모르는 형식이거나, 임계값이 없다."""


# 패널에 고른 평가 이미지 하나: 묶음, 범주, 범주 배열 안의 행 번호(index, 맵을 읽을 때 씀), 이미지 경로, 라벨,
# 결함 유형, margin(이미지 점수 ÷ 범주 임계값, 1보다 크면 불량 판정).
@dataclass(frozen=True)
class Example:
    """패널에 고른 평가 이미지 하나. `index`는 그 범주의 배열과 맵에서의 행 번호다."""

    group: str
    category: str
    index: int
    image: str
    label: int
    defect_types: str
    margin: float  # 이미지 점수 ÷ 그 범주의 임계값

    # 나머지 결함 유형은 '+ n more'로 센다.
    def _parts(self, keep: int | None = None) -> tuple[str, str]:
        """제목 조각 두 개: ('묶음: 범주, 결함 유형,', 'score/threshold = 값'). 유형은 앞 `keep`개만."""
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

    # 제목을 width 글자로 줄바꿈하되, 'score/threshold = 값'은 쪼개지 않고 마지막 줄에 붙거나 새 줄로 간다.
    def _wrap(self, width: int, keep: int | None) -> list[str]:
        head, tail = self._parts(keep)
        lines = textwrap.wrap(head, width, break_long_words=False, break_on_hyphens=False)
        if len(lines[-1]) + 1 + len(tail) <= width:
            lines[-1] += f" {tail}"
        else:
            lines.append(tail)
        return lines

    # 결함 유형을 하나씩 줄여 가며 다시 줄바꿈해, 결함 유형이 많은 이미지(VisA는 한 이미지에 여러 유형이
    # 있다)의 제목이 첫 칸을 넘지 않게 한다.
    def wrapped_title(self, width: int = _TITLE_CHARS, max_lines: int = _TITLE_LINES) -> str:
        """`title`을 한 줄 `width` 글자 이하로 줄바꿈한다. "score/threshold = x"는 쪼개지 않는다.

        제목이 `max_lines` 줄을 넘으면, 들어가는 앞쪽 결함 유형만 이름을 적고 나머지는 개수("+ 3 more")로
        센다.
        """
        lines = self._wrap(width, None)
        n_types = len(self.defect_types.split("|"))
        for keep in range(n_types - 1, 0, -1):
            if len(lines) <= max_lines:
                break
            lines = self._wrap(width, keep)
        return "\n".join(lines)


# 1이 아닌 값이 '1.00'으로 보이면 판정 방향(1보다 큰지)이 거짓으로 보이기 때문이다.
def _ratio(margin: float) -> str:
    """margin을 소수 둘째 자리로 적되, 1이 아닌 값이 1의 어느 쪽인지 보일 때까지 자릿수를 늘린다."""
    if margin == 1.0:
        return "1.00"
    for digits in range(2, 18):  # 소수 16자리면 어떤 float64든 1과 구별된다
        text = f"{margin:.{digits}f}"
        if float(text) != 1.0:
            return text
    return repr(margin)


# 정렬된 후보에서 범주가 겹치지 않게 앞에서부터 n개를 고른다(한 묶음 안에서 같은 범주는 한 장까지, README
# '실패 사례' 규칙).
def _one_per_category(ordered: list[Example], n: int) -> list[Example]:
    taken: list[Example] = []
    for example in ordered:
        if len(taken) == n:
            break
        if all(example.category != other.category for other in taken):
            taken.append(example)
    return taken


# README '실패 사례'에 같은 규칙이 적혀 있다. 그림을 보고 고르지 않고 점수만으로 정하므로 같은 실행에서는 항상
# 같은 예가 나온다(README '미리 정한 규칙으로 고른 예시').
def pick_examples(
    cats: dict[str, dict[str, np.ndarray]], thresholds: dict[str, float], per_group: int = PER_GROUP
) -> dict[str, list[Example]]:
    """점수만 보는 고정 규칙으로 패널 행을 고른다.

    margin = 이미지 점수 ÷ 그 범주의 임계값. score > threshold(`calibrate`의 판정 규칙)이면 불량 판정이므로
    불량 판정 이미지는 margin > 1이다. 묶음은 이 순서다:

    - 검출한 결함(결함, 불량 판정): margin이 실행의 모든 검출 결함 margin 중앙값에 가장 가까운
      `per_group`장(전형적인 검출)
    - 놓친 결함(결함, 정상 판정): margin이 가장 낮은 `per_group`장
    - 오검출(정상, 불량 판정): margin이 가장 높은 `per_group`장

    각 묶음에서 범주당 한 장까지이고, 동점은 이미지 경로로 가른다. 후보가 모자란 묶음은 있는 만큼만 둔다.
    `cats`는 범주마다 `eval_images`, `eval_labels`(0 또는 1), `eval_defect_types`, `eval_score`를 가지고,
    모든 임계값은 양수여야 한다.
    """
    found: dict[str, list[Example]] = {group: [] for group in EXAMPLE_GROUPS}
    # 임계값은 양수여야 margin이 의미가 있다. 점수가 유한하고 라벨이 0/1인지 확인한 뒤 이미지마다 묶음을
    # 정한다(정상이고 통과한 이미지는 어느 묶음에도 들지 않는다).
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
    # 묶음마다 정렬 키: 검출은 중앙값과의 거리, 놓침은 margin 오름차순, 오검출은 margin 내림차순. 모두 이미지
    # 경로를 두 번째 키로 둔다.
    detected = found[EXAMPLE_GROUPS[0]]
    median = float(np.median([e.margin for e in detected])) if detected else 0.0
    keys: dict[str, Callable[[Example], tuple[float, str]]] = {
        EXAMPLE_GROUPS[0]: lambda e: (abs(e.margin - median), e.image),
        EXAMPLE_GROUPS[1]: lambda e: (e.margin, e.image),
        EXAMPLE_GROUPS[2]: lambda e: (-e.margin, e.image),
    }
    # 묶음마다 정렬한 뒤 범주당 한 장씩 per_group개를 고른다.
    return {
        group: _one_per_category(sorted(found[group], key=keys[group]), per_group) for group in EXAMPLE_GROUPS
    }


# dev 절차 Dinomaly 실행은 빼 둔 정상이 곧 평가 정상이라 보정 점수가 없으므로 거절하고 이유를 알려 준다.
def example_thresholds(run: MethodRun, alpha: float = ALPHA) -> dict[str, float]:
    """`compare`와 같은 방식의 범주별 임계값: 실행의 `cal_score`에서 목표 α의 conformal 임계값."""
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


# 평가 폴더의 run.json을 읽는다. 없거나 깨졌으면 '완성된 평가 실행이 아니다'로 알린다.
def _run_meta(run_dir: Path) -> dict:
    path = run_dir / "run.json"
    if not path.is_file():
        raise ExamplesError(f"{run_dir} is not a finished evaluation run: it has no run.json")
    try:
        meta = _load(path)
    except (OSError, ValueError) as err:  # JSONDecodeError도 ValueError다
        raise ExamplesError(f"{path} cannot be read: {type(err).__name__}: {err}") from err
    if not isinstance(meta, dict):
        raise ExamplesError(f"{path} is not the run.json of an evaluation run")
    return meta


# 경로 구분자·'..' 같은 것이 섞인 이름은 폴더 밖 파일을 읽게 할 수 있어 거절한다(POSIX·Windows 경로 모두
# 확인). 중복도 거절한다.
def _run_categories(meta: dict, path: Path) -> list[str]:
    """run.json의 범주 이름 목록. 이름이 실행 폴더의 파일 이름(<범주>.npz)이 되므로 순수한 이름이어야 한다."""
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


# 손상된 파일에서 np.load가 낼 수 있는 예외들(잘린 .npz는 zip이 아니고, 빈 파일은 헤더가 없다). 한곳에 모아
# 같은 방식으로 ExamplesError로 바꾼다.
_READ_ERRORS = (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile)


# 키가 없으면 무엇이 빠졌는지 알린다.
def _npz_arrays(path: Path, keys: Sequence[str]) -> dict[str, np.ndarray]:
    """.npz 파일에서 `keys`만 읽는다. 파일 안의 다른(큰) 배열은 읽지 않는다."""
    try:
        with np.load(path) as z:
            arrays = {key: z[key] for key in keys if key in z.files}
    except _READ_ERRORS as err:
        raise ExamplesError(f"{path} cannot be read: {type(err).__name__}: {err}") from err
    missing = [key for key in keys if key not in arrays]
    if missing:
        raise ExamplesError(f"{path} has no {', '.join(missing)}")
    return arrays


# 공통 형식 실행(run.json에 method가 있음)은 cal_images, run_patchcore 실행은 pool_images(교차 적합 풀)다.
def _calibration_images_key(meta: dict) -> str:
    """`cal_score`를 낸 이미지 목록의 키. `run_patchcore` 실행("method" 없음)은 그것을 풀이라 부른다."""
    return "cal_images" if meta.get("method") else "pool_images"


# 봉인 규칙을 어기면 SealedTestError로 거절한다.
def _check_scored_images(
    run_dir: Path, names: list[str], cal_key: str, sealed: bool
) -> tuple[dict[str, ManifestRow], dict[str, dict[str, np.ndarray]]]:
    """점수를 하나도 읽기 전에, 실행이 채점한 모든 이미지를 매니페스트와 대조한다.

    이미지 목록과 평가 라벨만 읽는다. 모든 평가·보정 이미지는 그 범주의 매니페스트 이미지여야 하고
    매니페스트의 라벨과 맞아야 하며(보정 이미지는 정상), 평가 이미지가 두 번 나오면 안 된다. test 절차가
    아닌 실행은 어느 목록에도 봉인 테스트 이미지를 가지면 안 된다(그려지든 아니든): 그 점수가 임계값을
    정하거나 고르는 예를 바꾸기 때문이다. 이미지별 매니페스트 행과, 범주별로 읽은 배열을 돌려준다.
    """
    try:
        manifest = read_manifest(paths.VISA_MANIFEST)
    except (OSError, ValueError) as err:
        raise ExamplesError(f"the manifest {paths.VISA_MANIFEST} cannot be read: {err}") from err
    # 이미지 경로 → 매니페스트 행, 그리고 매니페스트에 있는 범주 집합. run.json의 범주가 모두 매니페스트
    # 범주인지 먼저 확인한다.
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
        # 평가 이미지는 자기 라벨로, 보정 이미지는 정상(0)으로 묶어 한꺼번에 매니페스트와 대조한다.
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


# 범주 순서가 run.json과 같은지, 평가 배열 길이가 서로 같은지 확인한다.
def _load_scores(run_dir: Path, meta: dict, names: list[str]) -> MethodRun:
    """`compare`와 같은 방식으로 실행을 읽는다: `run_patchcore` 실행("method" 없음) 또는 공통 형식."""
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


# 헤더는 메모리 맵으로 본다.
def _check_maps(path: Path, n: int) -> None:
    """맵 파일에 평가 이미지마다 2차원 맵이 하나씩 있는지 확인한다(헤더만 읽는다)."""
    try:
        maps = np.load(path, mmap_mode="r")
    except _READ_ERRORS as err:
        raise ExamplesError(f"{path} cannot be read: {type(err).__name__}: {err}") from err
    shape = getattr(maps, "shape", None)  # 이 이름의 .npz는 NpzFile로 읽혀 shape이 없으므로 걸러진다
    del maps
    if shape is None or len(shape) != 3 or shape[0] != n:
        raise ExamplesError(f"{path} has shape {shape}: expected {n} maps, one per evaluation image")


# 맵 파일에서 한 행만 메모리 맵으로 읽어 float64로 복사한다. Windows는 매핑된 파일을 잠그므로 바로 매핑을
# 놓는다.
def _read_map(path: Path, index: int) -> np.ndarray:
    maps = np.load(path, mmap_mode="r")
    row = np.array(maps[index], dtype=np.float64)
    del maps  # Windows는 매핑된 파일을 잠가 둔다
    return row


# 패널 제목. 범주당 한 장 규칙 때문에 묶음이 2장보다 적게 찼으면 그 사실을 제목에 적는다.
def _examples_title(label: str, protocol: str, picks: dict[str, list[Example]]) -> str:
    title = f"Example inspections: {label} on the {PROTOCOLS[protocol]}"
    short = [(group, len(picks[group])) for group in EXAMPLE_GROUPS if len(picks[group]) < PER_GROUP]
    if short:
        counts = ", ".join(f"{n} {group}{'' if n == 1 else 's'}" for group, n in short)
        title += f"\nFewer than {PER_GROUP} found (one per category at most): {counts}"
    return title


# 패널 캡션: 이미지 출처(VisA, CC BY 4.0), 방법, 임계값 출처(목표 5% split-conformal), 고르는 규칙, 그리고 '맵
# 위 선은 픽셀 임계값 초과 영역이고 이미지 점수는 맵 최댓값이 아니라서 임계값 근처에서는 선과 판정이 다를 수
# 있다'는 주의.
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


# README의 examples-dinomaly.png, examples-patchcore.png를 만든다. 순서가 핵심이다: (1) run.json·절차
# 확인(test면 allow_test 필수) → (2) 파일 존재 → (3) 채점된 모든 이미지를 매니페스트와 대조 → (4) 점수 읽기와
# 임계값·예 고르기 → (5) 고른 범주의 맵 파일 점검과 캐시 색인 확인 → (6) test면 장부 기록 → (7) 고른
# 이미지·마스크만 캐시에서 읽기 → (8) 맵 읽기와 그리기. 이미지 없이 할 수 있는 점검을 모두 장부 기록 전에
# 해서, 실패한 실행이 장부에 줄을 남기지 않게 한다.
def examples(run_dir: Path, out: Path, name: str = "examples", *, allow_test: bool = False) -> Path:
    """평가 실행 하나의 예시 검사 패널을 그려 `<out>/<name>.png`로 쓴다.

    `run_dir`은 평가 폴더다: run.json, `<범주>.npz`, `<범주>_maps.npy`(평가 이미지마다 맵 하나,
    `eval_images` 순서). 점수는 `compare`가 읽는 방식으로 읽고, 범주마다 `compare`가 `compare.ALPHA`에서
    정하는 임계값을 쓴다: PatchCore 실행은 교차 적합, 공통 형식(Dinomaly) 실행은 홀드아웃. 행은
    `pick_examples` 규칙을 따른다: 모든 검출 결함의 score/threshold 중앙값에 가장 가까운 검출 결함 2장,
    score/threshold가 가장 낮은 놓친 결함 2장, 가장 높은 오검출 2장. 각 묶음에서 범주당 한 장까지이고 동점은
    이미지 경로로 가른다. 모자란 묶음은 있는 만큼만 두고 그림 제목에 그렇다고 적는다.

    test 절차 실행은 `allow_test is True`가 아니면 거절한다. 점수를 하나라도 읽기 전에, 실행이 채점한 모든
    이미지(평가와 보정)를 매니페스트와 대조한다: test 절차가 아닌 실행이 어디든 봉인 테스트 이미지를 가지면
    거절한다. 이미지와 마스크는 고른 이미지의 매니페스트 행을 거쳐 256px 캐시(`paths.CACHE`)에서 읽는다.
    test 절차 실행이면 이미지 없이 할 수 있는 점검을 모두 마친 뒤, 첫 이미지를 읽기 전에 테스트 장부에 한
    줄을 남긴다. 고른 행의 맵은 이미지 다음에 읽는다(맵은 그 이미지의 결함 위치를 보여 주기 때문이다).
    ExamplesError나 SealedTestError를 낸다. 그때 PNG는 쓰지 않고, 장부 줄은 이미지를 읽은 경우에만 남는다.
    """
    run_dir, out = Path(run_dir), Path(out)
    meta = _run_meta(run_dir)
    protocol = meta.get("protocol")
    if not isinstance(protocol, str) or protocol not in PROTOCOLS:
        raise ExamplesError(f"{run_dir / 'run.json'} names no known protocol: {protocol!r}")
    sealed = protocol == "test"
    # allow_test는 진릿값이 아니라 정확히 True일 때만 허락한다(splits.select와 같은 규칙). 실수로 넘긴 다른
    # 값이 봉인을 열지 않게 한다.
    if sealed and allow_test is not True:
        raise SealedTestError(
            f"{run_dir} is a test-protocol run: its images are the sealed test set. Pass --allow-test "
            "(the read is logged in the test ledger) to draw them"
        )
    # 범주마다 점수 파일(.npz)과 맵 파일(_maps.npy)이 모두 있는지 확인한다.
    names = _run_categories(meta, run_dir / "run.json")
    wanted = [run_dir / f"{c}{suffix}" for c in names for suffix in (".npz", "_maps.npy")]
    missing = [path.name for path in wanted if not path.is_file()]
    if missing:
        raise ExamplesError(f"{run_dir} is incomplete: it has no {', '.join(missing)}")
    cal_key = _calibration_images_key(meta)
    by_image, checked = _check_scored_images(run_dir, names, cal_key, sealed)
    run = _load_scores(run_dir, meta, names)
    # 대조할 때 읽은 이미지 목록과 점수를 읽을 때 다시 읽은 목록이 같은지 확인한다. 두 번 읽는 사이에 파일이
    # 바뀌었으면 검사한 것과 다른 내용을 쓰게 되므로 멈춘다.
    for category, cat in run.cats.items():
        if any(not np.array_equal(cat.get(key), checked[category][key]) for key in ("eval_images", cal_key)):
            raise ExamplesError(f"{run_dir / f'{category}.npz'} changed while it was being read")
    # compare와 같은 임계값으로 예를 고르고, 하나도 없으면 멈춘다.
    thresholds = example_thresholds(run)
    picks = pick_examples(run.cats, thresholds)
    chosen = [example for group in EXAMPLE_GROUPS for example in picks[group]]
    if not chosen:
        raise ExamplesError(f"no image of {run_dir} falls in any group")
    for category in dict.fromkeys(example.category for example in chosen):
        _check_maps(run_dir / f"{category}_maps.npy", len(run.cats[category]["eval_images"]))
    # 고른 이미지의 매니페스트 행으로 256px 캐시에서 이미지와 마스크를 읽는다(점수 맵·픽셀 지표와 같은 256
    # 격자).
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
            # 색인 조회만 먼저 해서, 캐시에 없는 이미지가 있으면 장부 기록 전에 멈춘다.
            cache._rows(rows)
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
        # 메모리 맵을 닫아 캐시 파일이 잠긴 채 남지 않게 한다.
        cache.close()
    # 고른 예마다 맵을 읽어 패널 행을 만든다. 마스크는 결함 이미지만 넘기고, 임계값은 그 범주의 것이다.
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
    # 제목(방법, 절차, 모자란 묶음)과 캡션(출처, 임계값, 규칙)을 붙여 그리고 저장한다.
    label = _method(run.name)[0]
    fig = fig_examples(
        panel,
        title=_examples_title(label, protocol, picks),
        caption=_examples_caption(label, run.calibration),
    )
    return save(fig, out / f"{name}.png")


# --examples 처리: 실패하면 이유를 stderr에 쓰고 종료 코드 2로 끝낸다(docstring: 실행이 패널을 줄 수 없을 때
# 2).
def _examples_main(args: argparse.Namespace) -> None:
    try:
        path = examples(args.examples, args.out, args.name, allow_test=args.allow_test)
    except (ExamplesError, SealedTestError) as err:
        print(f"examples not drawn: {err}", file=sys.stderr)
        raise SystemExit(2) from None
    print(f"wrote {path}")


# 명령줄 진입점. 인자 없이 부르면 리포트 그림 다섯 장을 docs/figures/에 다시 그린다. 예시 패널은 `python -m
# defect_inspect.figures --examples outputs/dm-test --allow-test --name examples-dinomaly`처럼 따로
# 부른다(README). --allow-test와 --name은 --examples와만, --only는 리포트 그림에만 쓴다.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports", type=Path, default=paths.REPORTS, help="리포트 파일 폴더")
    parser.add_argument("--out", type=Path, default=paths.ROOT / "docs" / "figures", help="PNG를 쓸 폴더")
    parser.add_argument(
        "--only", nargs="+", choices=list(FIGURES), default=None, metavar="NAME", help="기본값: 모든 그림"
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="이 평가 실행 폴더의 예시 검사 패널만 그린다",
    )
    parser.add_argument(
        "--allow-test", action="store_true", help="--examples와 함께: 봉인 테스트 이미지를 읽는다(장부 기록)"
    )
    parser.add_argument("--name", default=None, help="--examples와 함께: .png를 뺀 PNG 이름(기본값 examples)")
    args = parser.parse_args(argv)
    # 인자 조합 점검: --allow-test·--name은 --examples와만, --only는 리포트 그림에만 쓰고, --name은 경로가
    # 아닌 파일 이름이어야 한다.
    if args.examples is None and (args.allow_test or args.name is not None):
        parser.error("--allow-test and --name go with --examples")
    if args.examples is not None and args.only is not None:
        parser.error("--only names report figures; it does not go with --examples")
    if args.name is not None and (not args.name or Path(args.name).name != args.name):
        parser.error(f"--name must be a plain file name, got {args.name!r}")
    # Windows 콘솔 코드 페이지 밖의 문자(한글 경로 등)가 든 경로를 출력해도 깨지지 않게 표준 출력을 UTF-8로
    # 바꾼다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.examples is not None:
        args.name = args.name or "examples"
        _examples_main(args)
        return
    # 입력 리포트가 없는 그림은 건너뛰고, 읽거나 그리다 실패한 그림은 그 그림만 멈춘 뒤 나머지를 계속 그린다.
    # 실패가 있으면 끝에서 오류 종료한다.
    failed = []
    for name in args.only or FIGURES:
        inputs, build = FIGURES[name]
        files = [args.reports / rel for rel in inputs]
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            print(f"skipped {name}: no {', '.join(missing)}")
            continue
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
