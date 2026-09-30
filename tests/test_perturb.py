"""Synthetic perturbations: identities, value ranges and dtypes, shift direction, batch consistency."""

from __future__ import annotations

import io
import warnings

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from defect_inspect import perturb

S = 24


def _noise(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)


def _smooth(n: int = 3, size: int = 32) -> np.ndarray:
    """Smooth colour gradients, different per image and channel."""
    y, x = np.mgrid[0:size, 0:size] / (size - 1)
    images = np.empty((n, size, size, 3), dtype=np.uint8)
    for i in range(n):
        for c in range(3):
            images[i, :, :, c] = np.rint(
                255 * (0.5 + 0.5 * np.sin(2.0 * x * (i + 1) + 3.0 * y * (c + 1))) * 0.9
            )
    return images


def _all_values() -> np.ndarray:
    """A [16, 16, 3] image containing every uint8 value in every channel."""
    return np.repeat(np.arange(256, dtype=np.uint8).reshape(16, 16, 1), 3, axis=2)


# (kind, value) pairs that are not identities
CASES = [
    ("brightness", 0.6),
    ("brightness", 1.4),
    ("gamma", 0.7),
    ("gamma", 1.5),
    ("blur", 1.5),
    ("shift", 2),
    ("shift", -3),
    ("jpeg", 50),
]


# ---------------------------------------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(S, S, 3), (4, S, S, 3)])
def test_identity_parameters_return_an_equal_copy(shape: tuple[int, ...]) -> None:
    images = _noise(shape)
    original = images.copy()
    results = [
        perturb.brightness(images, 1.0),
        perturb.gamma(images, 1.0),
        perturb.gaussian_blur(images, 0.0),
        perturb.shift(images, 0, 0),
        perturb.apply(images, "brightness", 1),
        perturb.apply(images, "gamma", 1),
        perturb.apply(images, "blur", 0),
        perturb.apply(images, "shift", 0),
    ]
    for out in results:
        assert out.dtype == np.uint8
        assert out.shape == images.shape
        assert np.array_equal(out, original)
        assert not np.shares_memory(out, images)
    assert np.array_equal(images, original)


def test_identity_holds_for_every_pixel_value() -> None:
    img = _all_values()
    assert np.array_equal(perturb.brightness(img, 1.0), img)
    assert np.array_equal(perturb.gamma(img, 1.0), img)


def test_shift_mask_identity() -> None:
    masks = (_noise((3, 10, 12), seed=1) > 128).astype(np.uint8)
    out = perturb.shift_mask(masks, 0, 0)
    assert np.array_equal(out, masks)
    assert not np.shares_memory(out, masks)


# ---------------------------------------------------------------------------------------------------------
# Shapes, dtypes, ranges, inputs left alone
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("kind", "value"), CASES)
@pytest.mark.parametrize("shape", [(S, S, 3), (3, S, S, 3), (2, 12, 20, 3)])
def test_output_shape_dtype_and_input_untouched(kind: str, value: float, shape: tuple[int, ...]) -> None:
    images = _noise(shape, seed=2)
    original = images.copy()
    out = perturb.apply(images, kind, value)
    assert out.dtype == np.uint8
    assert out.shape == shape
    assert np.array_equal(images, original)
    assert not np.array_equal(out, original)


def test_brightness_values() -> None:
    img = np.array([0, 1, 3, 100, 200, 255], dtype=np.uint8).reshape(1, 6, 1).repeat(3, axis=2)
    assert perturb.brightness(img, 1.5)[0, :, 0].tolist() == [0, 2, 4, 150, 255, 255]  # 1.5 -> 2, 4.5 -> 4
    assert perturb.brightness(img, 0.5)[0, :, 0].tolist() == [0, 0, 2, 50, 100, 128]  # halves round to even
    assert perturb.brightness(img, 0.0).max() == 0
    assert perturb.brightness(img, 1000.0)[0, :, 0].tolist() == [0, 255, 255, 255, 255, 255]


def test_brightness_direction_and_monotonicity() -> None:
    img = _all_values()
    darker = perturb.brightness(img, 0.7).astype(int)
    brighter = perturb.brightness(img, 1.3).astype(int)
    assert np.all(darker <= img) and darker.mean() < img.mean()
    assert np.all(brighter >= img) and brighter.mean() > img.mean()
    assert brighter.max() == 255
    for out in (darker, brighter):  # order of pixel values is preserved
        assert np.all(np.diff(out[:, :, 0].ravel()) >= 0)
    expected = np.clip(np.rint(np.arange(256) * 1.3), 0, 255)
    assert np.array_equal(brighter[:, :, 1].ravel(), expected)


def test_gamma_values_and_direction() -> None:
    img = _all_values()
    dark = perturb.gamma(img, 2.0).astype(int)
    bright = perturb.gamma(img, 0.5).astype(int)
    for out in (dark, bright):
        flat = out[:, :, 0].ravel()
        assert flat[0] == 0 and flat[255] == 255  # end points are fixed
        assert np.all(np.diff(flat) >= 0)
    assert np.all(dark <= img) and dark.mean() < img.mean()  # g > 1 darkens
    assert np.all(bright >= img) and bright.mean() > img.mean()  # g < 1 brightens
    assert dark[:, :, 0].ravel()[128] == 64  # 255 * (128 / 255) ** 2 = 64.25
    assert bright[:, :, 0].ravel()[64] == 128  # 255 * sqrt(64 / 255) = 127.75
    expected = np.rint(255.0 * (np.arange(256) / 255.0) ** 2.0)
    assert np.array_equal(dark[:, :, 2].ravel(), expected)


def test_blur_matches_scipy_per_channel() -> None:
    images = _noise((2, S, S, 3), seed=3)
    out = perturb.gaussian_blur(images, 1.2)
    for i in range(2):
        for c in range(3):
            ref = ndimage.gaussian_filter(images[i, :, :, c].astype(np.float64), sigma=1.2, mode="nearest")
            assert np.array_equal(out[i, :, :, c], np.clip(np.rint(ref), 0, 255).astype(np.uint8))


def test_blur_smooths_and_keeps_constants() -> None:
    images = _noise((2, S, S, 3), seed=4)
    out = perturb.gaussian_blur(images, 2.0)
    assert out.std() < 0.5 * images.std()
    assert abs(out.mean() - images.mean()) < 3.0
    assert perturb.gaussian_blur(images, 4.0).std() < out.std()
    for value in (0, 37, 255):  # replicated edges: a constant image stays constant, corners included
        const = np.full((S, S, 3), value, dtype=np.uint8)
        assert np.array_equal(perturb.gaussian_blur(const, 3.0), const)


def test_blur_does_not_mix_channels_or_images() -> None:
    batch = np.zeros((2, S, S, 3), dtype=np.uint8)
    batch[1] = 255  # a black and a white image
    assert np.array_equal(perturb.gaussian_blur(batch, 3.0), batch)
    red = np.zeros((S, S, 3), dtype=np.uint8)
    red[8:16, 8:16, 0] = 255
    out = perturb.gaussian_blur(red, 2.0)
    assert out[:, :, 0].max() < 255 and out[4, 12, 0] > 0  # spread within the red channel
    assert out[:, :, 1:].max() == 0


def test_jpeg_is_lossy_close_and_deterministic() -> None:
    images = _smooth()
    high = perturb.jpeg(images, 95)
    low = perturb.jpeg(images, 10)
    err_high = np.abs(high.astype(int) - images).mean()
    err_low = np.abs(low.astype(int) - images).mean()
    assert 0 < err_high < 3.0
    assert err_high < err_low < 30.0
    assert np.array_equal(low, perturb.jpeg(images, 10))
    assert np.array_equal(perturb.apply(images, "jpeg", 10.0), low)


def _pil_round_trip(img: np.ndarray, quality: int, **save_options: object) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=quality, **save_options)
    buf.seek(0)
    with Image.open(buf) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8)


@pytest.mark.parametrize("quality", [5, 30, 50, 75, 90, 100])
def test_jpeg_is_the_plain_pil_round_trip(quality: int) -> None:
    # Pins the encoder settings: PIL defaults, i.e. 4:2:0 chroma subsampling at every quality. A change
    # such as subsampling=0 (4:4:4) would silently alter what a given JPEG strength means.
    images = np.concatenate([_smooth(2, 32), _noise((2, 32, 32, 3), seed=9)])
    out = perturb.jpeg(images, quality)
    for i, img in enumerate(images):
        assert np.array_equal(out[i], _pil_round_trip(img, quality)), i
        assert np.array_equal(perturb.jpeg(img, quality), out[i]), i
        assert np.array_equal(_pil_round_trip(img, quality, subsampling="4:2:0"), out[i]), i
        assert not np.array_equal(_pil_round_trip(img, quality, subsampling=0), out[i]), i
    # Not square, odd sizes: still the plain round trip.
    odd = _noise((13, 21, 3), seed=10)
    assert np.array_equal(perturb.jpeg(odd, quality), _pil_round_trip(odd, quality))


def test_jpeg_blurs_colour_more_than_a_444_encoding() -> None:
    # What 4:2:0 means for the image: a one-pixel colour checkerboard of equal luma loses its chroma.
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:32, 0:32]
    img[(yy + xx) % 2 == 0] = (200, 60, 60)
    img[(yy + xx) % 2 == 1] = (60, 110, 200)
    err_default = np.abs(perturb.jpeg(img, 95).astype(int) - img).mean()
    err_444 = np.abs(_pil_round_trip(img, 95, subsampling=0).astype(int) - img).mean()
    assert err_default > 3 * err_444


# ---------------------------------------------------------------------------------------------------------
# Shift
# ---------------------------------------------------------------------------------------------------------


def _hot(y: int, x: int, size: int = 12) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[y, x] = (10, 20, 30)
    return img


@pytest.mark.parametrize(("dx", "dy"), [(2, 0), (0, 3), (2, 3), (-4, 0), (0, -1), (-2, 5), (1, -3)])
def test_shift_direction(dx: int, dy: int) -> None:
    # Positive dx moves content right (larger column index), positive dy moves it down (larger row index).
    y, x = 5, 6
    out = perturb.shift(_hot(y, x), dx, dy)
    assert np.array_equal(out, _hot(y + dy, x + dx))

    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[y, x] = 1
    moved = perturb.shift_mask(mask, dx, dy)
    assert moved.sum() == 1 and moved[y + dy, x + dx] == 1


def test_shift_replicates_edges() -> None:
    rows, cols = np.mgrid[0:6, 0:8]
    img = np.stack([cols * 10, rows * 10, cols + rows], axis=-1).astype(np.uint8)

    right = perturb.shift(img, 3, 0)
    assert np.array_equal(right[:, 3:], img[:, :-3])
    assert np.array_equal(right[:, :3], np.repeat(img[:, :1], 3, axis=1))

    up = perturb.shift(img, 0, -2)
    assert np.array_equal(up[:-2], img[2:])
    assert np.array_equal(up[-2:], np.repeat(img[-1:], 2, axis=0))

    # A shift larger than the image leaves only the replicated edge.
    assert np.array_equal(perturb.shift(img, 100, 0), np.repeat(img[:, :1], 8, axis=1))
    assert np.array_equal(perturb.shift(img, 0, -100), np.repeat(img[-1:], 6, axis=0))


def test_shift_is_separable_and_matches_roll_inside() -> None:
    img = _noise((S, S, 3), seed=5)
    out = perturb.shift(img, 2, -3)
    assert np.array_equal(out, perturb.shift(perturb.shift(img, 2, 0), 0, -3))
    rolled = np.roll(img, shift=(-3, 2), axis=(0, 1))
    assert np.array_equal(out[:-3, 2:], rolled[:-3, 2:])
    assert np.array_equal(perturb.apply(img, "shift", 2.0), perturb.shift(img, 2, 2))
    assert np.array_equal(perturb.apply(img, "shift", -2.9), perturb.shift(img, -2, -2))  # int() truncates


def test_shift_mask_zero_fills_and_follows_the_image() -> None:
    mask = np.ones((6, 8), dtype=np.uint8)
    out = perturb.shift_mask(mask, 3, -2)
    expected = np.zeros((6, 8), dtype=np.uint8)
    expected[:4, 3:] = 1
    assert np.array_equal(out, expected)
    assert perturb.shift_mask(mask, 8, 0).sum() == 0
    assert perturb.shift_mask(mask, 0, -6).sum() == 0
    assert perturb.shift_mask(mask, -7, 5).sum() == 1

    # Same translation as `shift`: a mask drawn into an image lands on the shifted mask.
    rng = np.random.default_rng(6)
    masks = np.zeros((3, S, S), dtype=np.uint8)
    masks[:, 6:-6, 6:-6] = rng.integers(0, 2, size=(3, S - 12, S - 12))
    images = np.repeat(masks[..., None] * 255, 3, axis=-1).astype(np.uint8)
    for dx, dy in [(2, 2), (-3, 1), (0, -4), (5, 0)]:
        moved = perturb.shift(images, dx, dy)
        assert np.array_equal(moved[..., 0] > 0, perturb.shift_mask(masks, dx, dy) > 0)


@pytest.mark.parametrize("dtype", [np.uint8, np.bool_, np.float32])
def test_shift_mask_shapes_and_dtypes(dtype: type) -> None:
    masks = (_noise((4, 10, 14), seed=7) > 100).astype(dtype)
    original = masks.copy()
    out = perturb.shift_mask(masks, 2, -1)
    assert out.dtype == masks.dtype
    assert out.shape == masks.shape
    assert np.array_equal(masks, original)
    for i in range(4):
        assert np.array_equal(out[i], perturb.shift_mask(masks[i], 2, -1))
    assert np.array_equal(out[:, :-1, 2:], masks[:, 1:, :-2])
    assert not out[:, -1, :].any() and not out[:, :, :2].any()


# ---------------------------------------------------------------------------------------------------------
# [S, S, 3] and [N, S, S, 3] give the same result
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("kind", "value"), CASES)
def test_single_image_and_batch_agree(kind: str, value: float) -> None:
    batch = np.concatenate([_noise((2, 32, 32, 3), seed=8), _smooth(2, 32)])
    out = perturb.apply(batch, kind, value)
    for i in range(len(batch)):
        assert np.array_equal(out[i], perturb.apply(batch[i], kind, value)), (kind, i)
    assert np.array_equal(perturb.apply(batch[1][None], kind, value)[0], out[1])


def test_apply_dispatches_to_the_named_function() -> None:
    images = _smooth(2, 16)
    assert np.array_equal(perturb.apply(images, "brightness", 1.2), perturb.brightness(images, 1.2))
    assert np.array_equal(perturb.apply(images, "gamma", 0.8), perturb.gamma(images, 0.8))
    assert np.array_equal(perturb.apply(images, "blur", 1.0), perturb.gaussian_blur(images, 1.0))
    assert np.array_equal(perturb.apply(images, "shift", 2), perturb.shift(images, 2, 2))
    assert np.array_equal(perturb.apply(images, "jpeg", 30), perturb.jpeg(images, 30))
    with pytest.raises(ValueError):
        perturb.apply(images, "contrast", 1.0)


# ---------------------------------------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------------------------------------


def test_rejects_wrong_dtype_and_shape() -> None:
    with pytest.raises(TypeError):
        perturb.brightness(np.zeros((S, S, 3), dtype=np.float32), 1.0)
    with pytest.raises(TypeError):
        perturb.shift(np.zeros((S, S, 3), dtype=np.int64), 1, 1)
    for bad in [(S, S), (S, S, 1), (2, 2, S, S, 3), (3, S, S)]:
        with pytest.raises(ValueError):
            perturb.gaussian_blur(np.zeros(bad, dtype=np.uint8), 1.0)
    with pytest.raises(ValueError):
        perturb.shift_mask(np.zeros(5, dtype=np.uint8), 1, 1)


def test_rejects_bad_parameters() -> None:
    img = np.zeros((S, S, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        perturb.brightness(img, -0.1)
    with pytest.raises(ValueError):
        perturb.gamma(img, 0.0)
    with pytest.raises(ValueError):
        perturb.gaussian_blur(img, -1.0)
    with pytest.raises(ValueError):
        perturb.jpeg(img, 0)
    with pytest.raises(ValueError):
        perturb.jpeg(img, 101)
    with pytest.raises(ValueError):
        perturb.brightness(img, float("nan"))


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -float("inf"), np.float32("inf"), np.inf])
def test_rejects_non_finite_brightness_and_blur(bad: float) -> None:
    # inf used to get through brightness (0 * inf = nan cast to uint8, with RuntimeWarnings) and to
    # surface as an OverflowError from inside scipy for the blur.
    images = _noise((2, S, S, 3), seed=11)
    images[0, 0, 0] = 0  # a zero pixel is what made 0 * inf undefined
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="factor"):
            perturb.brightness(images, bad)
        with pytest.raises(ValueError, match="sigma"):
            perturb.gaussian_blur(images, bad)
        with pytest.raises(ValueError, match="factor"):
            perturb.apply(images, "brightness", bad)
        with pytest.raises(ValueError, match="sigma"):
            perturb.apply(images, "blur", bad)


def test_large_finite_parameters_still_work() -> None:
    images = _noise((S, S, 3), seed=12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.array_equal(perturb.brightness(images, 1e300), np.where(images > 0, 255, 0))
        wide = perturb.gaussian_blur(images, 50.0)
        assert wide.dtype == np.uint8 and wide.shape == images.shape
        assert wide.std() < 0.6 * images.std()
