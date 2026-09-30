"""Single-image latency of the inspection path: onnxruntime on the CPU, and the torch pipeline on the GPU."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import paths
from .inspector import Inspector, score_map


def _stats(ms: list[float]) -> dict:
    values = np.asarray(ms, dtype=np.float64)
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95))}


def time_inspector(inspector: Inspector, images: np.ndarray, *, warmup: int = 5, repeats: int = 50) -> dict:
    """Latency per stage of the CPU path on uint8 images [N, S, S, 3] (cycled), in milliseconds.

    `backbone_ms` covers normalisation and the ONNX model, `search_ms` the bank search and the image
    score, `map_ms` the heatmap. The first `warmup` calls are not timed.
    """
    if len(images) == 0:
        raise ValueError("need at least one image")
    backbone, search, heat, total = [], [], [], []
    for i in range(warmup + repeats):
        image = images[i % len(images)]
        t0 = time.perf_counter()
        feats = inspector.features(image)
        t1 = time.perf_counter()
        _, patch_scores = inspector.score_features(feats)
        t2 = time.perf_counter()
        score_map(patch_scores, inspector.size, inspector.sigma)
        t3 = time.perf_counter()
        if i >= warmup:
            backbone.append((t1 - t0) * 1e3)
            search.append((t2 - t1) * 1e3)
            heat.append((t3 - t2) * 1e3)
            total.append((t3 - t0) * 1e3)
    return {
        "backbone_ms": _stats(backbone)["median"],
        "search_ms": _stats(search)["median"],
        "map_ms": _stats(heat)["median"],
        "total_ms": _stats(total)["median"],
        "total_p95_ms": _stats(total)["p95"],
        "repeats": repeats,
        "bank_rows": inspector.bank_rows,
        "precision": inspector.precision,
    }


def time_torch(
    extractor,
    bank,
    images: np.ndarray,
    *,
    device: str = "cuda",
    warmup: int = 5,
    repeats: int = 50,
    **score_kwargs,
) -> dict:
    """Latency of `patchcore.score_images` on one image at a time (fp16 on CUDA), in milliseconds."""
    import torch

    from .patchcore import score_images

    total = []
    for i in range(warmup + repeats):
        image = images[i % len(images)][None]
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        score_images(extractor, bank, image, batch_size=1, device=device, **score_kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            total.append((time.perf_counter() - t0) * 1e3)
    stats = _stats(total)
    return {"total_ms": stats["median"], "total_p95_ms": stats["p95"], "repeats": repeats, "device": device}


def time_resize(size: int, source: tuple[int, int] = (1500, 1000), repeats: int = 20) -> float:
    """Median milliseconds to resize a camera-sized RGB image to the input size (PIL BICUBIC)."""
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (source[1], source[0], 3), dtype=np.uint8))
    ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        image.resize((size, size), Image.Resampling.BICUBIC)
        ms.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ms))


def cpu_name() -> str:
    name = platform.processor()
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return name


def merge_into(path: Path, key: str, entry: dict) -> dict:
    """Add `entry` under `key` to the JSON file at `path` (created if missing); returns the whole content."""
    data = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    data[key] = {**data.get(key, {}), **entry}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, required=True, help="artifact set directory")
    parser.add_argument("--category", default="pcb1")
    parser.add_argument("--precision", nargs="+", default=["fp32"], choices=["fp32", "int8"])
    parser.add_argument(
        "--threads", type=int, default=None, help="onnxruntime intra-op threads (default: its own)"
    )
    parser.add_argument("--images", type=int, default=50)
    parser.add_argument("--gpu", action="store_true", help="also time the torch fp16 pipeline on the GPU")
    parser.add_argument("--key", required=True, help="entry name in the latency file, e.g. wrn50-256-r0.01")
    parser.add_argument("--out", type=Path, default=None, help="default: reports/stage4/latency.json")
    args = parser.parse_args(argv)

    from .cache import ImageCache
    from .splits import read_manifest, select

    art_dir = args.artifacts / args.category
    with open(art_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    manifest = read_manifest(paths.VISA_MANIFEST)
    rows = select(manifest, protocol="dev", part="pool_normal", category=args.category)[: args.images]
    images = ImageCache(paths.CACHE, meta["img_size"]).images(rows)

    entry: dict = {
        "category": args.category,
        "cpu": cpu_name(),
        "threads": args.threads,
        "resize_ms": time_resize(meta["img_size"]),
    }
    for precision in args.precision:
        inspector = Inspector.load(art_dir, precision=precision, threads=args.threads)
        entry[f"cpu_{precision}"] = time_inspector(inspector, images, repeats=len(images))
    if args.gpu:
        import torch

        from .backbones import make_extractor

        extractor = make_extractor(meta["backbone"], img_size=meta["img_size"]).to("cuda").eval()
        bank = torch.from_numpy(np.load(art_dir / "bank.npy"))
        entry["gpu_fp16"] = time_torch(
            extractor, bank, images, reweight_k=meta["reweight_k"], sigma=meta["sigma"], repeats=len(images)
        )
        entry["gpu"] = torch.cuda.get_device_name(0)
    data = merge_into(args.out or paths.REPORTS / "stage4" / "latency.json", args.key, entry)
    print(json.dumps({args.key: data[args.key]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
