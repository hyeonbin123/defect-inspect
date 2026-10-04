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
PATCHCORE = "patchcore"


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
    # Rounded to float32 like the torch pipeline's scores: thresholds are float32 scores, and a verdict
    # must not depend on the digits float64 keeps beyond them.
    return float(np.float32(weight * s_star))


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


def _is_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_finite_number(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:  # an integer beyond the float range
        return False


def check_meta(meta: dict) -> None:
    """Raise ValueError unless `meta` has every key of an artifact with a usable value."""
    if not isinstance(meta, dict):
        raise ValueError(f"artifact meta must be a JSON object, got {type(meta).__name__}")
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"artifact meta lacks {missing}")
    if not _is_int(meta["version"]) or meta["version"] != ARTIFACT_VERSION:
        raise ValueError(f"artifact version {meta['version']!r} is not {ARTIFACT_VERSION}")
    for key in ("img_size", "dim"):
        if not _is_int(meta[key]) or meta[key] <= 0:
            raise ValueError(f"artifact meta: {key} must be a positive integer, got {meta[key]!r}")
    grid = meta["grid"]
    if not isinstance(grid, (list, tuple)) or len(grid) != 2 or not all(_is_int(g) and g > 0 for g in grid):
        raise ValueError(f"artifact meta: grid must be two positive integers [H, W], got {grid!r}")
    if not _is_int(meta["reweight_k"]) or meta["reweight_k"] < 0:
        raise ValueError(
            f"artifact meta: reweight_k must be a non-negative integer, got {meta['reweight_k']!r}"
        )
    if not _is_finite_number(meta["sigma"]) or meta["sigma"] < 0:
        raise ValueError(f"artifact meta: sigma must be a finite number >= 0, got {meta['sigma']!r}")
    radius = int(4 * float(meta["sigma"]) + 0.5)  # the kernel radius of `gaussian_kernel1d`
    if radius >= meta["img_size"]:
        raise ValueError(
            f"artifact meta: sigma {meta['sigma']!r} gives a blur radius of {radius}, which needs an "
            f"input larger than {meta['img_size']}"
        )
    if not _is_finite_number(meta["threshold"]):
        raise ValueError(f"artifact meta: threshold must be a finite number, got {meta['threshold']!r}")


def check_model_io(session, meta: dict) -> None:
    """Raise ValueError unless the session's declared input and output fit `meta`.

    The model must take `image` [1, 3, S, S] and return `features` [1, H, W, dim]. Dimensions the model
    leaves symbolic are not compared.
    """
    size, (grid_h, grid_w), dim = meta["img_size"], meta["grid"], meta["dim"]
    wanted = (
        ("input", session.get_inputs(), "image", [1, 3, size, size]),
        ("output", session.get_outputs(), "features", [1, grid_h, grid_w, dim]),
    )
    for kind, nodes, name, want in wanted:
        shapes = {node.name: node.shape for node in nodes}
        if name not in shapes:
            raise ValueError(f"the model has no {kind} named {name!r} (it has {sorted(shapes)})")
        shape = list(shapes[name])
        if len(shape) != 4 or any(_is_int(a) and a != b for a, b in zip(shape, want, strict=True)):
            raise ValueError(f"the model's {kind} {name!r} is {shape}, the artifact meta needs {want}")


def _check_bank(bank: np.ndarray, dim: int) -> np.ndarray:
    """The bank as one fp32 array [R, dim], or ValueError."""
    bank = np.asarray(bank)
    if bank.ndim != 2 or bank.shape[0] == 0 or bank.shape[1] != dim:
        raise ValueError(f"bank shape {bank.shape} does not match dim {dim}")
    if not np.issubdtype(bank.dtype, np.floating):
        raise ValueError(f"the bank must be a float array, got {bank.dtype}")
    with np.errstate(over="ignore"):
        bank32 = bank.astype(np.float32)
    if not np.isfinite(bank32).all():
        raise ValueError("the bank has values that are not finite in fp32")
    return bank32


class Inspector:
    """One category's inspector: an ONNX Runtime session, a memory bank and a threshold.

    A score that is not finite raises `FloatingPointError` instead of reaching a verdict: `nan > threshold`
    is false, so it would otherwise pass as normal.
    """

    kind = PATCHCORE
    precision = "fp32"
    threads: int | None = None  # intra-op threads the session was created with; None = onnxruntime's default

    def __init__(self, session, bank: np.ndarray, meta: dict):
        check_meta(meta)
        bank32 = _check_bank(bank, meta["dim"])
        if hasattr(session, "get_inputs") and hasattr(session, "get_outputs"):
            check_model_io(session, meta)
        self.session = session
        self.meta = dict(meta)
        self.size = int(meta["img_size"])
        self.grid = (int(meta["grid"][0]), int(meta["grid"][1]))
        self.reweight_k = int(meta["reweight_k"])
        self.sigma = float(meta["sigma"])
        self.threshold = float(meta["threshold"])
        self.bank_rows = int(bank32.shape[0])
        # One centred fp32 copy. Centring leaves distances unchanged and keeps the expanded form accurate.
        self._centre = bank32.mean(axis=0, keepdims=True)
        self._bank = bank32 - self._centre
        self._bank_sq = np.square(self._bank).sum(axis=1)

    @classmethod
    def load(cls, artifact_dir: Path, *, precision: str = "fp32", threads: int | None = None) -> "Inspector":
        """An inspector from an artifact folder. Anything wrong with the artifact is a `ValueError`."""
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
        try:
            with open(artifact_dir / "meta.json", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError) as err:
            raise ValueError(f"{artifact_dir / 'meta.json'} is not readable JSON: {err}") from err
        try:
            check_meta(meta)  # before the model is opened, so that the message names the bad value
        except ValueError as err:
            raise ValueError(f"{artifact_dir / 'meta.json'}: {err}") from err
        try:
            bank = np.load(artifact_dir / "bank.npy", allow_pickle=False)
        except (OSError, ValueError, EOFError) as err:
            raise ValueError(f"{artifact_dir / 'bank.npy'} is not a readable array: {err}") from err
        options = ort.SessionOptions()
        if threads is not None:
            options.intra_op_num_threads = int(threads)
        try:
            session = ort.InferenceSession(
                str(model), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as err:  # onnxruntime has its own exception types for files it cannot load
            raise ValueError(f"{model} is not a loadable ONNX model: {err}") from err
        try:
            inspector = cls(session, bank, meta)
        except ValueError as err:
            raise ValueError(f"{artifact_dir} ({model.name}): {err}") from err
        inspector.precision = precision
        inspector.threads = None if threads is None else int(threads)
        return inspector

    def features(self, image: Image.Image | np.ndarray) -> np.ndarray:
        """Patch features float32 [H * W, D] of one image, rounded through float16 like the torch pipeline."""
        out = self.session.run(["features"], {"image": preprocess(image, self.size)})[0]
        if out.shape != (1, *self.grid, self._bank.shape[1]):
            raise ValueError(
                f"model returned features {out.shape}, expected {(1, *self.grid, self._bank.shape[1])}"
            )
        feats = out[0].reshape(-1, out.shape[-1])
        with np.errstate(over="ignore"):
            rounded = feats.astype(np.float16)
        # Checked after the cast, like patchcore._patch_features: a value beyond fp16 becomes inf there.
        if not np.isfinite(rounded).all():
            if not np.isfinite(feats).all():
                raise FloatingPointError("the model produced non-finite features")
            raise FloatingPointError("patch features overflowed fp16")
        return rounded.astype(np.float32)

    def score_features(self, feats: np.ndarray) -> tuple[float, np.ndarray]:
        """Image score and patch scores [H, W] of one image's features."""
        feats = np.asarray(feats, dtype=np.float32)
        if not np.isfinite(feats).all():
            raise FloatingPointError("patch features are not finite")
        query = feats - self._centre
        dist, index = nearest(query, self._bank, self._bank_sq)
        score = image_score(dist, index, query, self._bank, self._bank_sq, self.reweight_k)
        if not math.isfinite(score):
            raise FloatingPointError("the anomaly score is not finite")
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
        if not _is_finite_number(value):
            raise ValueError(f"the threshold must be a finite number, got {value!r}")
        self.threshold = float(value)
        self.meta["threshold"] = float(value)

    def calibrate(self, images: Sequence[Image.Image | np.ndarray], alpha: float = 0.05) -> float:
        """Set the threshold from normal images of the current condition (conformal rank at `alpha`)."""
        if len(images) == 0:
            raise ValueError("calibration needs at least one normal image")
        scores = np.array(
            [self.score_features(self.features(image))[0] for image in images], dtype=np.float32
        )
        thr = conformal_threshold(scores, alpha)
        self.set_threshold(thr.value)
        self.meta["alpha"] = float(alpha)
        self.meta["calibration"] = {
            "strategy": "holdout",
            "n": len(images),
            "guaranteed": bool(thr.guaranteed),
        }
        return self.threshold


RECONSTRUCTION = "reconstruction"
# Model files of a reconstruction (Dinomaly) artifact set; INT8 is dynamic quantization of MatMul only.
RECONSTRUCTION_PRECISIONS = {"fp32": "model_fp32.onnx", "int8-dynamic": "model_int8_dynamic.onnx"}
RECONSTRUCTION_META_KEYS = ("version", "kind", "name", "category", "img_size", "map_size", "threshold")


def check_reconstruction_meta(meta: dict) -> None:
    """Raise ValueError unless `meta` describes a usable reconstruction artifact."""
    if not isinstance(meta, dict):
        raise ValueError(f"artifact meta must be a JSON object, got {type(meta).__name__}")
    missing = [k for k in RECONSTRUCTION_META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"artifact meta lacks {missing}")
    if meta["kind"] != RECONSTRUCTION:
        raise ValueError(f"artifact kind {meta['kind']!r} is not {RECONSTRUCTION!r}")
    if not _is_int(meta["version"]) or meta["version"] != ARTIFACT_VERSION:
        raise ValueError(f"artifact version {meta['version']!r} is not {ARTIFACT_VERSION}")
    for key in ("img_size", "map_size"):
        if not _is_int(meta[key]) or meta[key] <= 0:
            raise ValueError(f"artifact meta: {key} must be a positive integer, got {meta[key]!r}")
    if not _is_finite_number(meta["threshold"]):
        raise ValueError(f"artifact meta: threshold must be a finite number, got {meta['threshold']!r}")


def artifact_kind(artifact_dir: Path) -> str:
    """ "patchcore" or "reconstruction", from the `kind` of the folder's meta.json (absent: patchcore)."""
    try:
        with open(Path(artifact_dir) / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError) as err:
        raise ValueError(f"{Path(artifact_dir) / 'meta.json'} is not readable JSON: {err}") from err
    return str(meta.get("kind", PATCHCORE)) if isinstance(meta, dict) else PATCHCORE


class ReconstructionInspector:
    """One category's inspector for a model that scores by itself (Dinomaly): no memory bank.

    The ONNX model takes `image` [1, 3, S, S] (ImageNet-normalised) and returns `score` [1] (the image
    score: the mean of the top 1% of the blurred map) and `map` [1, M, M] (the anomaly map). The model is
    shared by all categories; the threshold is per category. A score that is not finite raises
    `FloatingPointError` instead of reaching a verdict.
    """

    kind = RECONSTRUCTION
    precision = "fp32"
    threads: int | None = None
    bank_rows = 0

    def __init__(self, session, meta: dict):
        check_reconstruction_meta(meta)
        self.size = int(meta["img_size"])
        self.map_size = int(meta["map_size"])
        if hasattr(session, "get_inputs") and hasattr(session, "get_outputs"):
            self._check_io(session)
        self.session = session
        self.meta = dict(meta)
        self.threshold = float(meta["threshold"])

    def _check_io(self, session) -> None:
        inputs = {node.name: list(node.shape) for node in session.get_inputs()}
        outputs = {node.name: list(node.shape) for node in session.get_outputs()}
        wanted = (
            ("input", inputs, "image", [1, 3, self.size, self.size]),
            ("output", outputs, "score", [1]),
            ("output", outputs, "map", [1, self.map_size, self.map_size]),
        )
        for kind, shapes, name, want in wanted:
            if name not in shapes:
                raise ValueError(f"the model has no {kind} named {name!r} (it has {sorted(shapes)})")
            shape = shapes[name]
            if len(shape) != len(want) or any(
                _is_int(a) and a != b for a, b in zip(shape, want, strict=True)
            ):
                raise ValueError(f"the model's {kind} {name!r} is {shape}, the artifact meta needs {want}")

    @classmethod
    def load(
        cls, artifact_dir: Path, *, precision: str = "fp32", threads: int | None = None
    ) -> "ReconstructionInspector":
        """An inspector from a category folder (the model may sit in its parent folder)."""
        import onnxruntime as ort

        artifact_dir = Path(artifact_dir)
        if precision not in RECONSTRUCTION_PRECISIONS:
            raise ValueError(
                f"precision must be one of {sorted(RECONSTRUCTION_PRECISIONS)}, got {precision!r}"
            )
        model = artifact_dir / RECONSTRUCTION_PRECISIONS[precision]
        if not model.exists():
            model = artifact_dir.parent / RECONSTRUCTION_PRECISIONS[precision]
        for path in (artifact_dir / "meta.json", model):
            if not path.exists():
                raise ValueError(f"not a reconstruction artifact: {path} is missing")
        try:
            with open(artifact_dir / "meta.json", encoding="utf-8") as f:
                meta = json.load(f)
            check_reconstruction_meta(meta)
        except (OSError, ValueError) as err:
            raise ValueError(f"{artifact_dir / 'meta.json'}: {err}") from err
        options = ort.SessionOptions()
        if threads is not None:
            options.intra_op_num_threads = int(threads)
        try:
            session = ort.InferenceSession(
                str(model), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as err:  # onnxruntime has its own exception types for files it cannot load
            raise ValueError(f"{model} is not a loadable ONNX model: {err}") from err
        try:
            inspector = cls(session, meta)
        except ValueError as err:
            raise ValueError(f"{artifact_dir} ({model.name}): {err}") from err
        inspector.precision = precision
        inspector.threads = None if threads is None else int(threads)
        return inspector

    def run(self, image: Image.Image | np.ndarray) -> tuple[float, np.ndarray]:
        """(image score, float32 map [M, M]) of one image."""
        score, amap = self.session.run(["score", "map"], {"image": preprocess(image, self.size)})
        score = np.asarray(score, dtype=np.float32).reshape(-1)
        amap = np.asarray(amap, dtype=np.float32)
        if score.shape != (1,) or amap.shape != (1, self.map_size, self.map_size):
            raise ValueError(
                f"model returned score {score.shape} and map {amap.shape}, "
                f"expected (1,) and (1, {self.map_size}, {self.map_size})"
            )
        if not np.isfinite(score).all():
            raise FloatingPointError("the anomaly score is not finite")
        return float(score[0]), amap[0]

    def inspect(self, image: Image.Image | np.ndarray, *, heatmap: bool = True) -> Inspection:
        score, amap = self.run(image)
        return Inspection(
            score=score,
            threshold=self.threshold,
            is_defect=score > self.threshold,
            heatmap=amap if heatmap else None,
        )

    def scores(self, images: np.ndarray) -> np.ndarray:
        """Image scores float32 [N] of uint8 images [N, S, S, 3] that are already at the input size."""
        out = np.empty(len(images), dtype=np.float32)
        for i, image in enumerate(images):
            out[i] = self.run(image)[0]
        return out

    def set_threshold(self, value: float) -> None:
        if not _is_finite_number(value):
            raise ValueError(f"the threshold must be a finite number, got {value!r}")
        self.threshold = float(value)
        self.meta["threshold"] = float(value)

    def calibrate(self, images: Sequence[Image.Image | np.ndarray], alpha: float = 0.05) -> float:
        """Set the threshold from normal images of the current condition (conformal rank at `alpha`)."""
        if len(images) == 0:
            raise ValueError("calibration needs at least one normal image")
        scores = np.array([self.run(image)[0] for image in images], dtype=np.float32)
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
    check_meta(meta)
    _check_bank(bank, meta["dim"])
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
