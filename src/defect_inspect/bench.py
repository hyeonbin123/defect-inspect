"""Single-image latency of the inspection path: onnxruntime on the CPU, and the torch pipeline on the GPU.

The CPU latency that the serving budget is judged on is resize + inference: `resize_ms` (a camera-sized
photo down to the input size) plus the median of the inference path. Every `cpu_<precision>` entry of the
latency file carries that sum as `total_with_resize_ms` (`analyze_grid` adds the same two numbers).

A reconstruction (Dinomaly) artifact has no bank: its inference path is the normalisation and one ONNX
call that returns the score and the map (`time_reconstruction`).
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import paths
from .inspector import (
    RECONSTRUCTION,
    RECONSTRUCTION_PRECISIONS,
    Inspector,
    ReconstructionInspector,
    score_map,
)

RESIZE_SOURCE = (1500, 1000)  # width, height of the photo whose resize is added to the CPU latency


def _stats(ms: list[float]) -> dict:
    values = np.asarray(ms, dtype=np.float64)
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95))}


def time_inspector(inspector: Inspector, images: np.ndarray, *, warmup: int = 5, repeats: int = 50) -> dict:
    """Latency per stage of the CPU path on uint8 images [N, S, S, 3] (cycled), in milliseconds.

    `backbone_ms` covers normalisation and the ONNX model, `search_ms` the bank search and the image
    score, `map_ms` the heatmap. The first `warmup` calls are not timed; `*_ms` are medians over the
    `repeats` timed calls. The resize of the original photo is not part of it (see `time_resize`).
    """
    if len(images) == 0:
        raise ValueError("need at least one image")
    if warmup < 0 or repeats < 1:
        raise ValueError(f"need warmup >= 0 and repeats >= 1, got {warmup} and {repeats}")
    stages: dict[str, list[float]] = {"backbone": [], "search": [], "map": [], "total": []}
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
            stages["backbone"].append((t1 - t0) * 1e3)
            stages["search"].append((t2 - t1) * 1e3)
            stages["map"].append((t3 - t2) * 1e3)
            stages["total"].append((t3 - t0) * 1e3)
    out: dict = {}
    for name, ms in stages.items():
        stats = _stats(ms)
        out[f"{name}_ms"] = stats["median"]
        out[f"{name}_p95_ms"] = stats["p95"]
    out.update(
        repeats=repeats,
        bank_rows=inspector.bank_rows,
        precision=inspector.precision,
        threads=getattr(inspector, "threads", None),  # None = onnxruntime's default
    )
    return out


def time_reconstruction(inspector, images: np.ndarray, *, warmup: int = 5, repeats: int = 50) -> dict:
    """Latency of a reconstruction inspector on uint8 images [N, S, S, 3] (cycled), in milliseconds.

    One call covers the normalisation and the ONNX model, which returns the score and the map. The
    first `warmup` calls are not timed; `total_ms` is the median over `repeats` timed calls.
    """
    if len(images) == 0:
        raise ValueError("need at least one image")
    if warmup < 0 or repeats < 1:
        raise ValueError(f"need warmup >= 0 and repeats >= 1, got {warmup} and {repeats}")
    total = []
    for i in range(warmup + repeats):
        image = images[i % len(images)]
        t0 = time.perf_counter()
        inspector.run(image)
        t1 = time.perf_counter()
        if i >= warmup:
            total.append((t1 - t0) * 1e3)
    stats = _stats(total)
    return {
        "total_ms": stats["median"],
        "total_p95_ms": stats["p95"],
        "repeats": repeats,
        "kind": RECONSTRUCTION,
        "precision": inspector.precision,
        "threads": getattr(inspector, "threads", None),
    }


class _TorchScorer:
    """`patchcore.score_images` for one image at a time, with the bank prepared once.

    `score_images` copies the bank to the device, centres it and computes its row norms on every call.
    A service does that once, and its cost grows with the bank, so timing `score_images` per image would
    charge every image for it. The per-image work here is the same sequence of patchcore helpers.
    """

    def __init__(
        self, extractor, bank, *, device: str = "cuda", reweight_k: int = 9, sigma: float = 4.0, map_size=256
    ):
        import torch

        from . import patchcore

        if bank.ndim != 2 or bank.shape[0] == 0:
            raise ValueError(f"expected a non-empty bank [R, D], got {tuple(bank.shape)}")
        self._patchcore = patchcore
        self.device = torch.device(device)
        self.extractor = extractor.to(self.device).eval()
        self.bank = bank.to(device=self.device, dtype=torch.float32, copy=True)
        self.centre = self.bank.mean(dim=0, keepdim=True)
        self.bank -= self.centre
        self.bank_sq = torch.linalg.vector_norm(self.bank, dim=1).square_()
        self.kernel = patchcore.gaussian_kernel1d(sigma).to(self.device)
        self.reweight_k = reweight_k
        self.map_size = map_size

    def __call__(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        """uint8 [S, S, 3] -> (image score, float16 map [map_size, map_size]), both back on the host."""
        import torch
        import torch.nn.functional as F

        pc = self._patchcore
        with torch.no_grad():
            feats = pc._patch_features(self.extractor, image[None], self.device)
            _, h, w, dim = feats.shape
            if dim != self.bank.shape[1]:
                raise ValueError(f"feature dim {dim} does not match the bank ({self.bank.shape[1]})")
            query = feats.reshape(h * w, dim).to(torch.float32).sub_(self.centre)
            dist, nearest = pc._nearest(query, self.bank, self.bank_sq)
            patch_scores = dist.view(1, h * w)
            score = pc._image_scores(
                patch_scores,
                nearest.view(1, h * w),
                query.view(1, h * w, dim),
                self.bank,
                self.bank_sq,
                self.reweight_k,
            )
            size = (image.shape[0], image.shape[1])
            maps = F.interpolate(
                patch_scores.view(1, 1, h, w), size=size, mode="bilinear", align_corners=False
            )
            maps = pc._blur(maps, self.kernel)
            if size != (self.map_size, self.map_size):
                maps = F.interpolate(
                    maps, size=(self.map_size, self.map_size), mode="bilinear", align_corners=False
                )
            return float(score.cpu().numpy()[0]), maps[0, 0].to(torch.float16).cpu().numpy()


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
    """Single-image latency of the torch pipeline (fp16 on CUDA), in milliseconds.

    The work per image is that of `patchcore.score_images` (upload, features, search, image score, map,
    results back on the host). The bank is moved to the device and centred once, before the clock
    starts; that one-off cost is reported as `bank_setup_ms`.
    """
    import torch

    if len(images) == 0:
        raise ValueError("need at least one image")
    if warmup < 0 or repeats < 1:
        raise ValueError(f"need warmup >= 0 and repeats >= 1, got {warmup} and {repeats}")
    on_cuda = torch.device(device).type == "cuda"

    def clock() -> float:
        if on_cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    t0 = clock()
    scorer = _TorchScorer(extractor, bank, device=device, **score_kwargs)
    setup_ms = (clock() - t0) * 1e3
    total = []
    for i in range(warmup + repeats):
        image = images[i % len(images)]
        t0 = clock()
        scorer(image)
        t1 = clock()
        if i >= warmup:
            total.append((t1 - t0) * 1e3)
    stats = _stats(total)
    return {
        "total_ms": stats["median"],
        "total_p95_ms": stats["p95"],
        "bank_setup_ms": setup_ms,
        "repeats": repeats,
        "bank_rows": int(bank.shape[0]),
        "device": str(device),
    }


def time_resize(size: int, source: tuple[int, int] = RESIZE_SOURCE, repeats: int = 20) -> float:
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


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def merge_into(path: Path, key: str, entry: dict) -> dict:
    """Add `entry` under `key` to the JSON file at `path` (created if missing); returns the whole content."""
    data = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    data[key] = {**data.get(key, {}), **entry}
    _write_json(path, data)
    return data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, required=True, help="artifact set directory")
    parser.add_argument("--category", default="pcb1")
    parser.add_argument(
        "--precision",
        nargs="+",
        default=["fp32"],
        choices=sorted({"fp32", "int8", *RECONSTRUCTION_PRECISIONS}),
        help="fp32 and int8 (PatchCore artifacts) or fp32 and int8-dynamic (reconstruction artifacts)",
    )
    parser.add_argument(
        "--threads", type=int, default=None, help="onnxruntime intra-op threads (default: its own)"
    )
    parser.add_argument("--images", type=int, default=50)
    parser.add_argument("--gpu", action="store_true", help="also time the torch fp16 pipeline on the GPU")
    parser.add_argument("--key", required=True, help="entry name in the latency file, e.g. wrn50-256-r0.01")
    parser.add_argument("--out", type=Path, default=None, help="default: reports/stage4/latency.json")
    args = parser.parse_args(argv)
    latency_path = args.out or paths.REPORTS / "stage4" / "latency.json"

    from .cache import ImageCache
    from .splits import read_manifest, select

    art_dir = args.artifacts / args.category
    with open(art_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    manifest = read_manifest(paths.VISA_MANIFEST)
    rows = select(manifest, protocol="dev", part="pool_normal", category=args.category)[: args.images]
    images = ImageCache(paths.CACHE, meta["img_size"]).images(rows)

    reconstruction = meta.get("kind") == RECONSTRUCTION
    allowed = set(RECONSTRUCTION_PRECISIONS) if reconstruction else {"fp32", "int8"}
    if not set(args.precision) <= allowed:
        parser.error(f"precisions of this artifact: {sorted(allowed)}")
    if reconstruction and args.gpu:
        parser.error("--gpu times the PatchCore torch pipeline; a reconstruction artifact has none")

    resize_ms = time_resize(meta["img_size"])
    entry: dict = {
        "category": args.category,
        "cpu": cpu_name(),
        "threads": args.threads,
        "resize_ms": resize_ms,
        "resize_source": list(RESIZE_SOURCE),
    }
    if reconstruction:
        entry["artifacts"] = str(args.artifacts)
        entry["source"] = meta.get("source")
        for precision in args.precision:
            inspector = ReconstructionInspector.load(art_dir, precision=precision, threads=args.threads)
            timing = time_reconstruction(inspector, images, repeats=len(images))
            timing["total_with_resize_ms"] = resize_ms + timing["total_ms"]
            entry[f"cpu_{precision}"] = timing
        data = merge_into(latency_path, args.key, entry)
        print(json.dumps({args.key: data[args.key]}, ensure_ascii=False, indent=2))
        return
    for precision in args.precision:
        inspector = Inspector.load(art_dir, precision=precision, threads=args.threads)
        timing = time_inspector(inspector, images, repeats=len(images))
        # The registered CPU latency: the photo's resize plus the inference path.
        timing["total_with_resize_ms"] = resize_ms + timing["total_ms"]
        entry[f"cpu_{precision}"] = timing
    if args.gpu:
        import torch

        from .backbones import make_extractor

        extractor = make_extractor(meta["backbone"], img_size=meta["img_size"]).to("cuda").eval()
        bank = torch.from_numpy(np.load(art_dir / "bank.npy"))
        entry["gpu_fp16"] = time_torch(
            extractor, bank, images, reweight_k=meta["reweight_k"], sigma=meta["sigma"], repeats=len(images)
        )
        entry["gpu"] = torch.cuda.get_device_name(0)
    data = merge_into(latency_path, args.key, entry)
    print(json.dumps({args.key: data[args.key]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
