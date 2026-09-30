"""Synthetic image perturbations on uint8 RGB arrays shaped [S, S, 3] or [N, S, S, 3].

Every function returns a new uint8 array of the same shape and never modifies its input.
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image
from scipy import ndimage

KINDS = ("brightness", "gamma", "blur", "shift", "jpeg")


def _check_images(images: np.ndarray) -> np.ndarray:
    arr = np.asarray(images)
    if arr.dtype != np.uint8:
        raise TypeError(f"images must be uint8, got {arr.dtype}")
    if arr.ndim not in (3, 4) or arr.shape[-1] != 3:
        raise ValueError(f"images must be [S, S, 3] or [N, S, S, 3], got shape {arr.shape}")
    return arr


def _lookup(images: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Apply a pointwise map given as 256 float values (rounded half to even, clipped to 0..255)."""
    lut = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    return lut[_check_images(images)]


def brightness(images: np.ndarray, factor: float) -> np.ndarray:
    """Multiply pixel values by ``factor``, clip to 0..255 and round."""
    factor = float(factor)
    if not (math.isfinite(factor) and factor >= 0):
        raise ValueError(f"factor must be finite and non-negative, got {factor}")
    return _lookup(images, np.arange(256, dtype=np.float64) * factor)


def gamma(images: np.ndarray, g: float) -> np.ndarray:
    """Gamma curve ``255 * (x / 255) ** g`` (g > 1 darkens, g < 1 brightens)."""
    if not g > 0:
        raise ValueError(f"g must be positive, got {g}")
    return _lookup(images, 255.0 * (np.arange(256, dtype=np.float64) / 255.0) ** float(g))


def gaussian_blur(images: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur over the two spatial axes of each image and channel, edges replicated."""
    arr = _check_images(images)
    sigma = float(sigma)
    if not (math.isfinite(sigma) and sigma >= 0):
        raise ValueError(f"sigma must be finite and non-negative, got {sigma}")
    if sigma == 0:
        return arr.copy()
    flat = arr.reshape((-1, *arr.shape[-3:]))
    out = np.empty_like(flat)
    for i, img in enumerate(flat):
        # sigma 0 on the channel axis: channels are filtered independently.
        blurred = ndimage.gaussian_filter(img.astype(np.float64), sigma=(sigma, sigma, 0), mode="nearest")
        out[i] = np.clip(np.rint(blurred), 0, 255).astype(np.uint8)
    return out.reshape(arr.shape)


def shift(images: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate by whole pixels (positive dx: right, positive dy: down), replicating the edges."""
    arr = _check_images(images)
    height, width = arr.shape[-3], arr.shape[-2]
    rows = np.clip(np.arange(height) - int(dy), 0, height - 1)
    cols = np.clip(np.arange(width) - int(dx), 0, width - 1)
    return np.take(np.take(arr, rows, axis=-3), cols, axis=-2)


def shift_mask(masks: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate [..., H, W] masks like ``shift``; the vacated area becomes 0."""
    arr = np.asarray(masks)
    if arr.ndim < 2:
        raise ValueError(f"masks must be [..., H, W], got shape {arr.shape}")
    dx, dy = int(dx), int(dy)
    height, width = arr.shape[-2], arr.shape[-1]
    out = np.zeros_like(arr)
    if abs(dx) >= width or abs(dy) >= height:
        return out
    dst_rows = slice(max(dy, 0), height + min(dy, 0))
    src_rows = slice(max(-dy, 0), height - max(dy, 0))
    dst_cols = slice(max(dx, 0), width + min(dx, 0))
    src_cols = slice(max(-dx, 0), width - max(dx, 0))
    out[..., dst_rows, dst_cols] = arr[..., src_rows, src_cols]
    return out


def _jpeg_one(img: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8)


def jpeg(images: np.ndarray, quality: int) -> np.ndarray:
    """JPEG encode/decode round trip with PIL at the given quality (PIL defaults otherwise)."""
    arr = _check_images(images)
    quality = int(quality)
    if not 1 <= quality <= 100:
        raise ValueError(f"quality must be in 1..100, got {quality}")
    flat = arr.reshape((-1, *arr.shape[-3:]))
    out = np.empty_like(flat)
    for i, img in enumerate(flat):
        out[i] = _jpeg_one(img, quality)
    return out.reshape(arr.shape)


def apply(images: np.ndarray, kind: str, value: float) -> np.ndarray:
    """Dispatch by name: brightness | gamma | blur | shift | jpeg.

    ``shift`` moves by (dx=int(value), dy=int(value)); ``jpeg`` uses quality int(value).
    """
    if kind == "brightness":
        return brightness(images, float(value))
    if kind == "gamma":
        return gamma(images, float(value))
    if kind == "blur":
        return gaussian_blur(images, float(value))
    if kind == "shift":
        return shift(images, int(value), int(value))
    if kind == "jpeg":
        return jpeg(images, int(value))
    raise ValueError(f"unknown perturbation {kind!r}; expected one of {KINDS}")
