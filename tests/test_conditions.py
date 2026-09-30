import numpy as np
import pytest

from defect_inspect import conditions, perturb


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
        ("shift-1", 448, 2),  # 1.75 rounds to 2
        ("shift-2", 448, 4),  # 3.5 rounds half up
        ("shift-3", 448, 7),
        ("shift-1", 392, 2),  # 1.53
        ("shift-2", 392, 3),  # 3.06
        ("shift-3", 392, 6),  # 6.125
    ],
)
def test_strength_scales_lengths_with_the_input_size(name, size, expected):
    assert conditions.strength(name, size) == pytest.approx(expected)


@pytest.mark.parametrize("name", ["clean-1", "blur-0", "blur-4", "noise-1", "blur", ""])
def test_unknown_conditions_are_rejected(name):
    with pytest.raises(ValueError):
        conditions.parse(name)


def test_apply_matches_perturb_and_clean_is_a_copy():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, (3, 64, 64, 3), dtype=np.uint8)
    clean = conditions.apply_condition(images, "clean")
    assert np.array_equal(clean, images) and clean is not images
    assert np.array_equal(conditions.apply_condition(images, "brightness-3"), perturb.brightness(images, 0.7))
    assert np.array_equal(conditions.apply_condition(images, "gamma-1"), perturb.gamma(images, 1.1))
    # 64 px input: lengths shrink to a quarter of the 256-grid value.
    assert np.array_equal(conditions.apply_condition(images, "blur-2"), perturb.gaussian_blur(images, 0.25))
    assert np.array_equal(conditions.apply_condition(images, "shift-3"), perturb.shift(images, 1, 1))
    assert np.array_equal(conditions.apply_condition(images, "shift-1"), images)  # 0.25 px rounds to no shift
    assert np.array_equal(conditions.apply_condition(images, "jpeg-2"), perturb.jpeg(images, 70))
    assert np.array_equal(conditions.apply_condition(images[0], "gamma-2"), perturb.gamma(images[0], 1.25))


def test_every_condition_keeps_shape_and_dtype():
    images = np.random.default_rng(1).integers(0, 256, (2, 256, 256, 3), dtype=np.uint8)
    for name in conditions.condition_names():
        out = conditions.apply_condition(images, name)
        assert out.shape == images.shape and out.dtype == np.uint8
        if name != "clean":
            assert not np.array_equal(out, images)
