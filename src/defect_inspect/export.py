"""Export a patch-feature extractor to ONNX, quantize it to INT8, and build inspector artifacts.

The ONNX model has input `image` float32 [1, 3, S, S] (ImageNet-normalised) and output `features`
float32 [1, H, W, D]: exactly what the torch extractors in `backbones.py` return. An artifact set is a
folder with the model file(s) shared by all categories and one sub-folder per category holding its memory
bank and `meta.json` (threshold, grid, scoring options).
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from . import paths
from .calibrate import conformal_threshold
from .inspector import ARTIFACT_VERSION, PRECISIONS, check_meta, preprocess
from .ledger import git_commit
from .run_grid import ratio_name, ratio_rows


def _cpu_fp32(extractor):
    """`extractor` on the CPU in fp32 and eval mode: a copy unless it already is (the caller's stays put)."""
    import torch

    tensors = [*extractor.parameters(), *extractor.buffers()]
    ready = not extractor.training and all(
        t.device.type == "cpu" and (not t.is_floating_point() or t.dtype == torch.float32) for t in tensors
    )
    return extractor if ready else copy.deepcopy(extractor).to("cpu").float().eval()


def export_onnx(extractor, img_size: int, path: Path, *, opset: int = 18) -> Path:
    """Export `extractor` (eval mode, fp32, CPU) for a fixed batch of one image and check it against torch.

    Uses the TorchScript-based exporter (`dynamo=False`): it handles both the WideResNet and the timm
    DINOv2 extractor. The WideResNet graph goes through onnxruntime's static quantization without
    changes; the DINOv2 graph does not (see `quantize_static_int8`).
    Raises if the onnxruntime output differs from torch by more than 1e-3 (max abs) on one random image.
    The caller's module is not moved or converted: a copy is exported when it is not on the CPU in fp32.
    """
    import onnx
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extractor = _cpu_fp32(extractor)
    example = torch.zeros(1, 3, img_size, img_size, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            extractor,
            (example,),
            str(path),
            input_names=["image"],
            output_names=["features"],
            opset_version=opset,
            dynamo=False,
        )
    onnx.checker.check_model(str(path))
    rng = np.random.default_rng(0)
    probe = rng.integers(0, 256, (1, img_size, img_size, 3), dtype=np.uint8)
    parity = check_parity(extractor, path, probe)
    if parity["max_abs"] > 1e-3:
        raise RuntimeError(f"ONNX export differs from torch: {parity}")
    return path


def _session(path: Path, threads: int | None = None):
    import onnxruntime as ort

    options = ort.SessionOptions()
    if threads is not None:
        options.intra_op_num_threads = int(threads)
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def check_parity(extractor, onnx_path: Path, images: np.ndarray) -> dict:
    """Difference between torch fp32 features and onnxruntime features on uint8 images [N, S, S, 3]."""
    import torch

    session = _session(onnx_path)
    extractor = _cpu_fp32(extractor)
    max_abs, sum_abs, count, scale = 0.0, 0.0, 0, 0.0
    for image in images:
        x = preprocess(image, image.shape[0])
        with torch.no_grad():
            ref = extractor(torch.from_numpy(x)).numpy()
        out = session.run(["features"], {"image": x})[0]
        diff = np.abs(out - ref)
        max_abs = max(max_abs, float(diff.max()))
        sum_abs += float(diff.sum())
        count += diff.size
        scale = max(scale, float(np.abs(ref).max()))
    return {"max_abs": max_abs, "mean_abs": sum_abs / max(count, 1), "rel": max_abs / max(scale, 1e-12)}


class _Reader:
    """Calibration images for onnxruntime's static quantization, preprocessed like the service does."""

    def __init__(self, images: np.ndarray):
        self._images = iter(images)

    def get_next(self):
        image = next(self._images, None)
        return None if image is None else {"image": preprocess(image, image.shape[0])}


def quantize_static_int8(
    fp32_path: Path, int8_path: Path, calibration_images: np.ndarray, *, per_channel: bool = True
) -> Path:
    """Static INT8 quantization (QDQ, MinMax calibration, activations uint8, weights int8).

    Works for the WideResNet export. Raises `NotImplementedError` with the reason when onnxruntime's
    pre-processing cannot handle the graph, and writes no model then. That is the case for the DINOv2
    export (onnxruntime 1.30): symbolic shape inference stops at the Expand node of the class and register
    tokens. There is no fallback, because skipping that step only lets the quantization finish: the
    features of the pretrained ViT-S/14 at 252 px then correlate 0.07 with the fp32 ones.
    """
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    fp32_path, int8_path = Path(fp32_path), Path(int8_path)
    if not fp32_path.is_file():
        raise FileNotFoundError(f"no fp32 model at {fp32_path}")
    if len(calibration_images) == 0:
        raise ValueError("static quantization needs at least one calibration image")
    prepared = int8_path.with_suffix(".prep.onnx")
    try:
        try:
            quant_pre_process(str(fp32_path), str(prepared))
        except (OSError, MemoryError):
            raise  # the machine, not the graph: not a reason to go on without INT8
        except Exception as err:  # onnxruntime's shape inference fails with whatever its internals raise
            raise NotImplementedError(
                f"static INT8 quantization is not available for {fp32_path.name}: onnxruntime's "
                f"pre-processing failed ({type(err).__name__}: {err})"
            ) from err
        quantize_static(
            str(prepared),
            str(int8_path),
            _Reader(calibration_images),
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )
    finally:
        prepared.unlink(missing_ok=True)
    return int8_path


def build_artifacts(
    grid_dir: Path,
    ratio: float,
    out_dir: Path,
    *,
    alpha: float = 0.05,
    int8: bool = True,
    calibration_per_category: int = 8,
) -> dict:
    """Artifacts for every category of a `run_grid --save-banks` run at one coreset ratio.

    The threshold of each category is the conformal threshold of the run's cross-fitted scores (torch
    pipeline). INT8 calibration uses the first pool normals of every category in the run. When the model
    cannot be quantized (`NotImplementedError`, e.g. DINOv2) the fp32 artifacts are still complete: the
    set then has no `model_int8.onnx` and `artifacts.json` says why under `int8`.
    """
    from .backbones import make_extractor
    from .cache import ImageCache
    from .splits import read_manifest, select

    grid_dir, out_dir = Path(grid_dir), Path(out_dir)
    with open(grid_dir / "run.json", encoding="utf-8") as f:
        grid = json.load(f)
    if not grid.get("save_banks"):
        raise ValueError(f"{grid_dir} was run without --save-banks")
    with open(grid_dir / ratio_name(ratio) / "run.json", encoding="utf-8") as f:
        ratio_run = json.load(f)
    cfg = ratio_run["config"]
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    extractor = make_extractor(cfg["backbone"], img_size=cfg["img_size"])
    fp32 = export_onnx(extractor, cfg["img_size"], out_dir / PRECISIONS["fp32"])
    info = {
        "export_s": round(time.perf_counter() - t0, 1),
        "int8": {"available": False, "reason": "not requested"},
        "categories": [],
    }

    manifest = read_manifest(paths.VISA_MANIFEST)
    int8_path = out_dir / PRECISIONS["int8"]
    # A quantized model left by an earlier export into this folder does not belong to the new fp32 model.
    int8_path.unlink(missing_ok=True)
    if int8:
        cache = ImageCache(paths.CACHE, cfg["img_size"])
        rows = []
        for category in grid["categories"]:
            pool = select(manifest, protocol="dev", part="pool_normal", category=category)
            rows += pool[:calibration_per_category]
        t0 = time.perf_counter()
        try:
            quantize_static_int8(fp32, int8_path, cache.images(rows))
        except NotImplementedError as err:
            int8_path.unlink(missing_ok=True)
            info["int8"] = {"available": False, "reason": str(err)}
        else:
            info["int8"] = {"available": True}
            info["quantize_s"] = round(time.perf_counter() - t0, 1)
            info["calibration_images"] = len(rows)

    commit = git_commit()
    for cat_info in ratio_run["categories"]:
        category = cat_info["category"]
        with open(grid_dir / "banks" / f"{category}.json", encoding="utf-8") as f:
            n_full = json.load(f)["n_features"]["full"]
        bank = np.load(grid_dir / "banks" / f"{category}_full.npy")[: ratio_rows(n_full, ratio)]
        with np.load(grid_dir / ratio_name(ratio) / f"{category}.npz") as z:
            thr = conformal_threshold(z["pool_score_oof"], alpha)
        meta = {
            "version": ARTIFACT_VERSION,
            "name": cfg["name"],
            "category": category,
            "backbone": cfg["backbone"],
            "img_size": cfg["img_size"],
            "grid": cat_info["grid"],
            "dim": cat_info["dim"],
            "reweight_k": cfg["reweight_k"],
            "sigma": cfg["sigma"],
            "threshold": float(thr.value),
            "alpha": alpha,
            "calibration": {"strategy": "crossfit", "n": int(thr.n), "guaranteed": bool(thr.guaranteed)},
            "commit": commit,
            "source": {"grid": grid_dir.name, "ratio": ratio, "protocol": grid["protocol"]},
        }
        check_meta(meta)
        cat_dir = out_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        np.save(cat_dir / "bank.npy", bank)
        with open(cat_dir / "meta.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            f.write("\n")
        info["categories"].append(
            {"category": category, "bank_rows": int(bank.shape[0]), "threshold": meta["threshold"]}
        )
    with open(out_dir / "artifacts.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"config": cfg, "commit": commit, **info}, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return info


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", type=Path, required=True, help="a run_grid --save-banks run directory")
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="artifact set directory, e.g. artifacts/<name>"
    )
    parser.add_argument("--no-int8", action="store_true")
    parser.add_argument("--calibration-per-category", type=int, default=8)
    args = parser.parse_args(argv)
    info = build_artifacts(
        args.grid,
        args.ratio,
        args.out,
        int8=not args.no_int8,
        calibration_per_category=args.calibration_per_category,
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
