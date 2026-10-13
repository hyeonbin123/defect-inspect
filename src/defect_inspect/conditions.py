"""The synthetic capture-condition changes of stage 3, as registered in docs/experiments.md.

Strengths are given on the 256 grid. Length-like strengths (blur sigma, shift distance) are scaled by
`size / 256` so that an image at another input size gets the same physical change.
"""

# [파일 역할] 3단계의 합성 교란 표(밝기·감마·흐림·이동·JPEG 5종 × 강도 3)를 코드로 옮긴 것이다.
# docs/experiments.md 3단계 "합성 교란 (공통 정의)" 표와 값·순서가 같다.
# [설계 이유] 방법마다 입력 크기가 다르다(P0 256, DMS 280, DM 392, D-S 448px). 강도를 256 격자 기준으로 정하고
# 길이 단위(흐림 σ, 이동 거리)만 S/256배 하면, 어느 입력 크기에서도 원본 사진 위에서는 같은 크기의 변화가
# 된다. 밝기 배율·감마·JPEG 품질은 길이가 아니라서 그대로 쓴다.
# [연결] 조건 이름과 강도는 여기서 정하고, 실제 픽셀 연산은 perturb.py가 한다. run_perturb(3-A VisA),
# run_m2ad(3-B의 합성 조건), dinomaly_serving score --condition(6단계 dev 보조)이 apply_condition으로 이미지를
# 바꾸고, analyze_perturb와 figures가 이름·강도를 읽는다.

# math는 이동 거리 반올림(floor)에, perturb는 실제 교란 함수에 쓴다.
import math

import numpy as np

from . import perturb

# 교란 없는 조건의 이름. 분석은 이 조건을 기준으로 각 교란 조건의 AUROC·오검출률 변화를 낸다.
CLEAN = "clean"
# 교란 종류별 강도 1·2·3의 값(256 격자 기준). 밝기는 픽셀 값 배율, 감마는 255·(x/255)^g의 g(1보다 크면
# 어두워짐), 흐림은 가우시안 σ(px), 이동은 오른쪽 아래로 옮기는 픽셀 수, JPEG는 저장 품질이다. dict 순서가 곧
# 등록 순서라 condition_names가 이 순서로 조건 이름을 만든다.
LEVELS: dict[str, tuple[float, float, float]] = {
    "brightness": (0.9, 0.8, 0.7),
    "gamma": (1.1, 1.25, 1.5),
    "blur": (0.5, 1.0, 1.5),
    "shift": (1, 2, 4),
    "jpeg": (90, 70, 50),
}
# 강도를 정의한 기준 격자 크기. 길이 단위 강도는 size / REFERENCE_SIZE 배로 바꾼다.
REFERENCE_SIZE = 256


# 등록 순서대로 조건 이름 목록을 만든다: clean, brightness-1, ..., jpeg-3(모두 16개). 명령행에서 받은 조건
# 이름을 검사하거나 모든 조건을 차례로 돌릴 때 쓴다. include_clean=False면 교란 15개만 준다(run_m2ad의 합성
# 조건 P).
def condition_names(include_clean: bool = True) -> list[str]:
    """`clean`, then `<kind>-<level>` for every kind and level 1..3, in the registered order."""
    names = [f"{kind}-{level}" for kind in LEVELS for level in (1, 2, 3)]
    return [CLEAN, *names] if include_clean else names


# "brightness-3" 같은 이름을 (종류, 강도 번호)로 나눈다. 마지막 '-'를 기준으로 자르고(rpartition), 표에 없는
# 종류나 1~3이 아닌 강도면 ValueError를 낸다.
def parse(name: str) -> tuple[str, int]:
    kind, _, level = name.rpartition("-")
    if kind not in LEVELS or level not in {"1", "2", "3"}:
        raise ValueError(f"unknown condition {name!r}")
    return kind, int(level)


# 조건 이름과 입력 크기에서 perturb.apply에 넘길 실제 강도 값을 계산한다.
# - 흐림: σ에 size/256을 곱한다(소수 그대로).
# - 이동: size/256을 곱한 뒤 정수 픽셀로 반올림한다. 예를 들어 shift-1은 280px에서 1px(256 격자로 0.91px),
#   392·448px에서 2px다(docs/experiments.md 6단계 정정). 이동 거리가 크기마다 반올림되므로 방법끼리의 실제
#   이동량은 조금씩 다르다.
# - 나머지(밝기, 감마, JPEG): 크기와 무관하게 표의 값을 그대로 쓴다.
def strength(name: str, size: int) -> float:
    """The value passed to `perturb.apply` for condition `name` at input size `size`."""
    kind, level = parse(name)
    value = LEVELS[kind][level - 1]
    if kind == "blur":
        return value * size / REFERENCE_SIZE
    if kind == "shift":
        # Python의 round()는 .5를 짝수 쪽으로 보내므로(1.5 → 2, 2.5 → 2), floor(x + 0.5)로 .5를 항상 올리는
        # 반올림을 쓴다.
        return float(math.floor(value * size / REFERENCE_SIZE + 0.5))
    return float(value)


# uint8 이미지 배열(한 장 [S, S, 3] 또는 여러 장 [N, S, S, 3])에 조건을 적용한 새 배열을 돌려준다. 입력 크기
# S는 images.shape[-2](정사각형이라 너비 = 높이)에서 읽어 강도를 맞춘다.
# clean도 복사본을 돌려준다. perturb의 함수들처럼 항상 새 배열을 주어, 호출한 쪽이 결과를 고쳐도 원본 캐시
# 배열이 바뀌지 않게 한다.
def apply_condition(images: np.ndarray, name: str) -> np.ndarray:
    """uint8 images [N, S, S, 3] (or [S, S, 3]) under condition `name`; `clean` returns a copy."""
    if name == CLEAN:
        return images.copy()
    kind, _ = parse(name)
    return perturb.apply(images, kind, strength(name, images.shape[-2]))
