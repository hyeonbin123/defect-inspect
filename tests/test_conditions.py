# [파일 역할] 3단계 합성 교란 조건표를 코드로 옮긴 `defect_inspect.conditions`를 검증하는 테스트 묶음이다.
# [왜 필요한가] 교란 이름(밝기·감마·흐림·이동·JPEG × 강도 1~3)과 강도값은 docs/experiments.md 3단계 '합성 교란
# (공통 정의)' 표에 측정 전에 등록한 것이다. 강도는 256 격자 기준이고 길이 단위(흐림 σ, 이동 거리)는 입력 크기
# S에 맞춰 S÷256을 곱한다(이동은 반올림). 이 환산이 틀리면 크기가 다른 방법(P0 256px, D-S 448px, DM 392px)이
# 서로 다른 물리적 교란을 받게 된다.
# [연결] conditions는 실제 픽셀 변환을 `defect_inspect.perturb`에 맡기므로, 여기서는 이름 → (종류, 강도)
# 해석과 perturb 호출이 같은 결과를 내는지 본다.

import numpy as np
import pytest

from defect_inspect import conditions, perturb


# 확인: 조건 이름이 clean + 15개(종류 5 × 강도 3)이고 등록한 순서를 따르며, LEVELS 값이 등록표와 같다.
# 왜: run_perturb·run_m2ad·dinomaly_serving이 이 목록으로 채점할 조건을 정하고 figures가 LEVELS로 그림을
# 그린다. 이름이나 값이 바뀌면 이미 잰 결과와 새 결과가 어긋난다.
def test_names_follow_the_registered_table():
    names = conditions.condition_names()
    assert names[0] == "clean" and len(names) == 16
    assert names[1:4] == ["brightness-1", "brightness-2", "brightness-3"]
    assert names[-3:] == ["jpeg-1", "jpeg-2", "jpeg-3"]
    assert conditions.condition_names(include_clean=False) == names[1:]
    assert conditions.LEVELS == {
        "brightness": (0.9, 0.8, 0.7),
        "gamma": (1.1, 1.25, 1.5),
        "blur": (0.5, 1.0, 1.5),
        "shift": (1, 2, 4),
        "jpeg": (90, 70, 50),
    }


# 확인: strength가 흐림 σ는 S/256을 곱하고, 이동 거리는 S/256을 곱한 뒤 반올림(0.5는 올림)하며,
# 밝기·감마·JPEG는 크기와 무관하게 그대로 둔다.
# 왜: 448px의 shift-1(1.75 → 2)과 shift-2(3.5 → 4)처럼 경계값 반올림이 실제 이동 픽셀 수를 정한다.
# docs/experiments.md 6단계의 2026-10-04 정정(280px에서 shift-1은 1px 이동, 392px에서는 2px)처럼 이 값은 결과
# 해석에 직접 쓰인다.
@pytest.mark.parametrize(
    ("name", "size", "expected"),
    [
        ("brightness-2", 448, 0.8),
        ("gamma-3", 392, 1.5),
        ("jpeg-1", 448, 90),
        ("blur-1", 256, 0.5),
        ("blur-2", 448, 1.75),
        ("blur-3", 392, 1.5 * 392 / 256),
        ("shift-1", 256, 1),
        ("shift-1", 448, 2),  # 1.75를 반올림해 2
        ("shift-2", 448, 4),  # 3.5는 0.5라 올림해 4
        ("shift-3", 448, 7),
        ("shift-1", 392, 2),  # 1.53
        ("shift-2", 392, 3),  # 3.06
        ("shift-3", 392, 6),  # 6.125
    ],
)
def test_strength_scales_lengths_with_the_input_size(name, size, expected):
    assert conditions.strength(name, size) == pytest.approx(expected)


# 확인: 강도 번호 밖(0, 4), 없는 종류(noise), 강도가 빠진 이름, 'clean-1', 빈 문자열은 모두 ValueError다.
# 오타가 조용히 다른 조건으로 바뀌지 않게 막는다.
@pytest.mark.parametrize("name", ["clean-1", "blur-0", "blur-4", "noise-1", "blur", ""])
def test_unknown_conditions_are_rejected(name):
    with pytest.raises(ValueError):
        conditions.parse(name)


# 확인: apply_condition이 같은 강도로 perturb 함수를 부른 결과와 비트 단위로 같고, clean은 입력과 값은 같지만
# 다른 배열(사본)이다.
# 왜: clean이 원본을 그대로 돌려주면 뒤에서 배열을 고칠 때 원본 이미지까지 바뀐다. 64px 입력으로 길이 환산(256
# 기준의 1/4)도 함께 확인한다.
def test_apply_matches_perturb_and_clean_is_a_copy():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, (3, 64, 64, 3), dtype=np.uint8)
    clean = conditions.apply_condition(images, "clean")
    assert np.array_equal(clean, images) and clean is not images
    assert np.array_equal(conditions.apply_condition(images, "brightness-3"), perturb.brightness(images, 0.7))
    assert np.array_equal(conditions.apply_condition(images, "gamma-1"), perturb.gamma(images, 1.1))
    # 64px 입력이라 길이(흐림 σ, 이동 거리)가 256 격자 값의 4분의 1로 준다.
    assert np.array_equal(conditions.apply_condition(images, "blur-2"), perturb.gaussian_blur(images, 0.25))
    assert np.array_equal(conditions.apply_condition(images, "shift-3"), perturb.shift(images, 1, 1))
    assert np.array_equal(conditions.apply_condition(images, "shift-1"), images)  # 0.25px는 0으로 반올림
    assert np.array_equal(conditions.apply_condition(images, "jpeg-2"), perturb.jpeg(images, 70))
    assert np.array_equal(conditions.apply_condition(images[0], "gamma-2"), perturb.gamma(images[0], 1.25))


# 확인: 모든 조건이 모양과 uint8 형식을 지키고, clean이 아닌 조건은 256px 이미지를 실제로 바꾼다(교란이 아무
# 일도 안 하는 조건이 없다).
def test_every_condition_keeps_shape_and_dtype():
    images = np.random.default_rng(1).integers(0, 256, (2, 256, 256, 3), dtype=np.uint8)
    for name in conditions.condition_names():
        out = conditions.apply_condition(images, name)
        assert out.shape == images.shape and out.dtype == np.uint8
        if name != "clean":
            assert not np.array_equal(out, images)
