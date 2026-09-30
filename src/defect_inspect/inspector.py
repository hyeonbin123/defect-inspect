"""The serving-side pipeline: ONNX Runtime features + numpy PatchCore scoring, without torch.

Banks and thresholds come from the torch pipeline (`patchcore.score_images`, GPU fp16). This module
reproduces the same arithmetic in numpy so that the service can run on a CPU with onnxruntime only:
features rounded to float16, bank centring, nearest bank row, the PatchCore re-weighting, and the score
map (bilinear upsample, Gaussian blur, resize to the 256 grid).
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .calibrate import conformal_threshold

ARTIFACT_VERSION = 1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MAP_SIZE = 256
# Largest [query rows, bank rows] block built at once while searching the bank (2**27 fp32 = 512 MB).
_MAX_DIST_ELEMS = 1 << 27
META_KEYS = (
    "version",
    "name",
    "category",
    "backbone",
    "img_size",
    "grid",
    "dim",
    "reweight_k",
    "sigma",
    "threshold",
)
PRECISIONS = {"fp32": "model_fp32.onnx", "int8": "model_int8.onnx"}


@dataclass
class Inspection:
    score: float
    threshold: float
    is_defect: bool
    heatmap: np.ndarray | None  # float32 [256, 256] on the score scale, or None when not requested


def preprocess(image: Image.Image | np.ndarray, size: int) -> np.ndarray:
    """An image -> ImageNet-normalised float32 [1, 3, size, size].

    PIL images (any size) are converted to RGB and resized with BICUBIC, like the resize cache. A uint8
    array must already be [size, size, 3].
    """
    if isinstance(image, Image.Image):
        if image.size != (size, size) or image.mode != "RGB":
            image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.uint8)
    else:
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.shape != (size, size, 3):
            raise ValueError(f"expected a uint8 array [{size}, {size}, 3], got {array.dtype} {array.shape}")
    x = array.astype(np.float32) / np.float32(255.0)
    x = (x - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def nearest(query: np.ndarray, bank: np.ndarray, bank_sq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Euclidean distance to, and index of, the nearest bank row for each query row (both centred, fp32).

    The search uses the expanded form in blocks; the distance to the row it finds is recomputed by
    direct subtraction, so it does not suffer from cancellation near zero.
    """
    n = query.shape[0]
    step = max(1, _MAX_DIST_ELEMS // bank.shape[0])
    dist = np.empty(n, dtype=np.float32)
    index = np.empty(n, dtype=np.int64)
    for start in range(0, n, step):
        q = query[start : start + step]
        block = bank_sq[None, :] - 2.0 * (q @ bank.T)
        found = block.argmin(axis=1)
        index[start : start + step] = found
        dist[start : start + step] = np.linalg.norm(q - bank[found], axis=1)
    return dist, index


def image_score(
    patch_scores: np.ndarray,
    nearest_index: np.ndarray,
    query: np.ndarray,
    bank: np.ndarray,
    bank_sq: np.ndarray,
    reweight_k: int = 9,
) -> float:
    """PatchCore image score of one image: the largest patch score, re-weighted by its bank neighbourhood."""
    p_star = int(patch_scores.argmax())
    s_star = float(patch_scores[p_star])
    if reweight_k <= 1:
        return s_star
    k = min(reweight_k, bank.shape[0])
    m_star = int(nearest_index[p_star])
    block = bank_sq - 2.0 * (bank @ bank[m_star])
    block[m_star] = -np.inf  # m* itself comes first, even if the bank has duplicates of it
    support = np.argpartition(block, k - 1)[:k] if k < bank.shape[0] else np.arange(bank.shape[0])
    support = support[np.argsort(block[support], kind="stable")]
    dist = np.linalg.norm(query[p_star][None, :] - bank[support], axis=1).astype(np.float64)
    weights = np.exp(dist - dist.max())
    weight = 1.0 - weights[0] / weights.sum()
    return float(weight * s_star)


def _bilinear_axis(n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source indices and weight of torch's bilinear interpolation with `align_corners=False`."""
    src = (np.arange(n_out, dtype=np.float64) + 0.5) * (n_in / n_out) - 0.5
    src = np.clip(src, 0.0, None)
    lo = np.minimum(np.floor(src).astype(np.int64), n_in - 1)
    hi = np.minimum(lo + 1, n_in - 1)
    return lo, hi, (src - lo).astype(np.float32)


def resize_bilinear(a: np.ndarray, size: int) -> np.ndarray:
    """[H, W] float32 -> [size, size], as torch's bilinear interpolation without antialiasing."""
    lo_r, hi_r, w_r = _bilinear_axis(a.shape[0], size)
    lo_c, hi_c, w_c = _bilinear_axis(a.shape[1], size)
    rows = a[lo_r] * (1.0 - w_r)[:, None] + a[hi_r] * w_r[:, None]
    return (rows[:, lo_c] * (1.0 - w_c)[None, :] + rows[:, hi_c] * w_c[None, :]).astype(np.float32)


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    """Normalised kernel of size `2 * int(4 * sigma + 0.5) + 1` (a single 1 if sigma <= 0)."""
    if sigma <= 0:
        return np.ones(1, dtype=np.float32)
    radius = int(4 * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return (kernel / kernel.sum()).astype(np.float32)


def score_map(
    patch_scores: np.ndarray, size: int, sigma: float = 4.0, map_size: int = MAP_SIZE
) -> np.ndarray:
    """Patch scores [H, W] -> float32 map [map_size, map_size]: upsample to `size`, blur, resize."""
    out = resize_bilinear(patch_scores.astype(np.float32), size)
    kernel = gaussian_kernel1d(sigma)
    if kernel.size > 1:
        if kernel.size // 2 >= size:
            raise ValueError(f"blur radius {kernel.size // 2} needs a map larger than {size}")
        # scipy "mirror" = reflect without repeating the edge sample = torch's "reflect" padding.
        out = ndimage.correlate1d(out, kernel, axis=0, mode="mirror")
        out = ndimage.correlate1d(out, kernel, axis=1, mode="mirror")
    if size != map_size:
        out = resize_bilinear(out, map_size)
    return out.astype(np.float32)


class Inspector:
    """One category's inspector: an ONNX Runtime session, a memory bank and a threshold."""

    def __init__(self, session, bank: np.ndarray, meta: dict):
        missing = [k for k in META_KEYS if k not in meta]
        if missing:
            raise ValueError(f"artifact meta lacks {missing}")
        if meta["version"] != ARTIFACT_VERSION:
            raise ValueError(f"artifact version {meta['version']} is not {ARTIFACT_VERSION}")
        if bank.ndim != 2 or bank.shape[0] == 0 or bank.shape[1] != meta["dim"]:
            raise ValueError(f"bank shape {bank.shape} does not match dim {meta['dim']}")
        self.session = session
        self.meta = dict(meta)
        self.size = int(meta["img_size"])
        self.grid = (int(meta["grid"][0]), int(meta["grid"][1]))
        self.reweight_k = int(meta["reweight_k"])
        self.sigma = float(meta["sigma"])
        self.threshold = float(meta["threshold"])
        self.bank_rows = int(bank.shape[0])
        # One centred fp32 copy. Centring leaves distances unchanged and keeps the expanded form accurate.
        bank32 = bank.astype(np.float32)
        self._centre = bank32.mean(axis=0, keepdims=True)
        self._bank = bank32 - self._centre
        self._bank_sq = np.square(self._bank).sum(axis=1)

    @classmethod
    def load(cls, artifact_dir: Path, *, precision: str = "fp32", threads: int | None = None) -> "Inspector":
        import onnxruntime as ort

        artifact_dir = Path(artifact_dir)
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(PRECISIONS)}, got {precision!r}")
        model = artifact_dir / PRECISIONS[precision]
        if not model.exists():
            # An artifact set keeps one model file next to the per-category folders.
            model = artifact_dir.parent / PRECISIONS[precision]
        for path in (artifact_dir / "meta.json", artifact_dir / "bank.npy", model):
            if not path.exists():
                raise ValueError(f"not an inspector artifact: {path} is missing")
        with open(artifact_dir / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        options = ort.SessionOptions()
        if threads is not None:
            options.intra_op_num_threads = int(threads)
        session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
        inspector = cls(session, np.load(artifact_dir / "bank.npy"), meta)
        inspector.precision = precision
        return inspector

    precision = "fp32"

    def features(self, image: Image.Image | np.ndarray) -> np.ndarray:
        """Patch features float32 [H * W, D] of one image, rounded through float16 like the torch pipeline."""
        out = self.session.run(["features"], {"image": preprocess(image, self.size)})[0]
        if out.shape != (1, *self.grid, self._bank.shape[1]):
            raise ValueError(
                f"model returned features {out.shape}, expected {(1, *self.grid, self._bank.shape[1])}"
            )
        feats = out[0].reshape(-1, out.shape[-1])
        if not np.isfinite(feats).all():
            raise FloatingPointError("the model produced non-finite features")
        return feats.astype(np.float16).astype(np.float32)

    def score_features(self, feats: np.ndarray) -> tuple[float, np.ndarray]:
        """Image score and patch scores [H, W] of one image's features."""
        query = feats.astype(np.float32) - self._centre
        dist, index = nearest(query, self._bank, self._bank_sq)
        score = image_score(dist, index, query, self._bank, self._bank_sq, self.reweight_k)
        return score, dist.reshape(self.grid)

    def inspect(self, image: Image.Image | np.ndarray, *, heatmap: bool = True) -> Inspection:
        score, patch_scores = self.score_features(self.features(image))
        heat = score_map(patch_scores, self.size, self.sigma) if heatmap else None
        return Inspection(
            score=score, threshold=self.threshold, is_defect=score > self.threshold, heatmap=heat
        )

    def scores(self, images: np.ndarray) -> np.ndarray:
        """Image scores float32 [N] of uint8 images [N, S, S, 3] that are already at the input size."""
        out = np.empty(len(images), dtype=np.float32)
        for i, image in enumerate(images):
            out[i] = self.score_features(self.features(image))[0]
        return out

    def set_threshold(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError("the threshold must be finite")
        self.threshold = float(value)
        self.meta["threshold"] = float(value)

    def calibrate(self, images: Sequence[Image.Image | np.ndarray], alpha: float = 0.05) -> float:
        """Set the threshold from normal images of the current condition (conformal rank at `alpha`)."""
        if len(images) == 0:
            raise ValueError("calibration needs at least one normal image")
        scores = np.array([self.score_features(self.features(image))[0] for image in images])
        thr = conformal_threshold(scores, alpha)
        self.set_threshold(thr.value)
        self.meta["alpha"] = float(alpha)
        self.meta["calibration"] = {
            "strategy": "holdout",
            "n": len(images),
            "guaranteed": bool(thr.guaranteed),
        }
        return self.threshold


def save_artifact(
    artifact_dir: Path, *, onnx_fp32: Path, bank: np.ndarray, meta: dict, onnx_int8: Path | None = None
) -> None:
    """Write an artifact directory: the ONNX model(s), the bank and meta.json."""
    import shutil

    meta = {"version": ARTIFACT_VERSION, **meta}
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"artifact meta lacks {missing}")
    if bank.ndim != 2 or bank.shape[1] != meta["dim"]:
        raise ValueError(f"bank shape {bank.shape} does not match dim {meta['dim']}")
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    targets = [(Path(onnx_fp32), artifact_dir / PRECISIONS["fp32"])]
    if onnx_int8 is not None:
        targets.append((Path(onnx_int8), artifact_dir / PRECISIONS["int8"]))
    for source, target in targets:
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
    np.save(artifact_dir / "bank.npy", bank)
    with open(artifact_dir / "meta.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")
