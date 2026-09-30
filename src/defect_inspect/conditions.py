"""The synthetic capture-condition changes of stage 3, as registered in docs/experiments.md.

Strengths are given on the 256 grid. Length-like strengths (blur sigma, shift distance) are scaled by
`size / 256` so that an image at another input size gets the same physical change.
"""

import math

import numpy as np

from . import perturb

CLEAN = "clean"
LEVELS: dict[str, tuple[float, float, float]] = {
    "brightness": (0.9, 0.8, 0.7),
    "gamma": (1.1, 1.25, 1.5),
    "blur": (0.5, 1.0, 1.5),
    "shift": (1, 2, 4),
    "jpeg": (90, 70, 50),
}
REFERENCE_SIZE = 256


def condition_names(include_clean: bool = True) -> list[str]:
    """`clean`, then `<kind>-<level>` for every kind and level 1..3, in the registered order."""
    names = [f"{kind}-{level}" for kind in LEVELS for level in (1, 2, 3)]
    return [CLEAN, *names] if include_clean else names


def parse(name: str) -> tuple[str, int]:
    kind, _, level = name.rpartition("-")
    if kind not in LEVELS or level not in {"1", "2", "3"}:
        raise ValueError(f"unknown condition {name!r}")
    return kind, int(level)


def strength(name: str, size: int) -> float:
    """The value passed to `perturb.apply` for condition `name` at input size `size`."""
    kind, level = parse(name)
    value = LEVELS[kind][level - 1]
    if kind == "blur":
        return value * size / REFERENCE_SIZE
    if kind == "shift":
        return float(math.floor(value * size / REFERENCE_SIZE + 0.5))
    return float(value)


def apply_condition(images: np.ndarray, name: str) -> np.ndarray:
    """uint8 images [N, S, S, 3] (or [S, S, 3]) under condition `name`; `clean` returns a copy."""
    if name == CLEAN:
        return images.copy()
    kind, _ = parse(name)
    return perturb.apply(images, kind, strength(name, images.shape[-2]))
