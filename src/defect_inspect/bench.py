"""Single-image latency of the inspection path: onnxruntime on the CPU, and the torch pipeline on the GPU.

The CPU latency that the serving budget is judged on is resize + inference: `resize_ms` (a camera-sized
photo down to the input size) plus the median of the inference path. Every `cpu_<precision>` entry of the
latency file carries that sum as `total_with_resize_ms` (`analyze_grid` adds the same two numbers).

A reconstruction (Dinomaly) artifact has no bank: its inference path is the normalisation and one ONNX
call that returns the score and the map (`time_reconstruction`).
"""

# [파일 역할] 검사 경로의 한 장당 지연을 재는 4·6단계 측정 모듈이다(`python -m defect_inspect.bench`). 서빙과
# 같은 onnxruntime CPU 경로(inspector.py)와, 비교용 torch GPU 경로를 잰다.
# [측정 규칙] 범주 pcb1의 dev 풀 정상 50장(봉인 테스트 아님)을 한 장씩 넣고, 워밍업 5회 뒤 50회의 중앙값을
# 쓴다. CPU 지연의 판정값은 "원본 크기(1500×1000) 사진을 입력 크기로 줄이는 시간 + 추론
# 중앙값"이다(docs/experiments.md 4단계 격자, 6단계 관문 1). CPU가 한가할 때 재야 한다는 조건(규칙 변경 6)은
# 이 코드가 확인하지 않으므로 실행하는 쪽이 지킨다.
# [설계 이유] PatchCore는 단계별(백본, 최근접 탐색, 맵)로 나눠 잰다. 4단계에서 WideResNet 10% 뱅크의 지연
# 대부분(82~91%)이 최근접 탐색이라는 것을 이 분해로 보였고, 그래서 코어셋 비율이 지연을 좌우했다.
# Dinomaly(복원 방식)는 뱅크가 없고 ONNX 호출 한 번이 점수와 맵을 함께 내므로 전체만 잰다.
# [연결] 결과는 reports/stage4/latency.json(기본) 또는 `--out`으로 reports/stage6/latency.json에 키별로 합쳐
# 쓴다. analyze_grid와 analyze_stage6이 이 파일을 읽어 판정하고, runtime_compare가 time_resize와 cpu_name을
# 가져다 쓴다.

import argparse
import json
import platform
import time
from pathlib import Path

# numpy는 중앙값·백분위 계산과 가짜 사진 생성에, PIL은 리사이즈 시간 측정에 쓴다. inspector에서는 서빙과
# 똑같은 CPU 검사기(PatchCore용 Inspector, Dinomaly용 ReconstructionInspector)와 맵 계산 함수를 가져와,
# 서비스가 실제로 도는 코드 그대로의 시간을 잰다.
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

# 카메라 원본 크기의 가정(가로 1500, 세로 1000, VisA 원본 약 1,300~1,560 × 960~1,180px과 비슷한 크기). 이
# 사진을 입력 크기로 줄이는 시간을 CPU 지연에 더하는 것이 등록된 규칙이다.
RESIZE_SOURCE = (1500, 1000)  # width, height of the photo whose resize is added to the CPU latency


# 밀리초 목록의 중앙값과 95백분위. 평균 대신 중앙값을 써서 가끔 튀는 한 번(다른 프로그램의 끼어듦 등)에 덜
# 흔들리게 한다.
def _stats(ms: list[float]) -> dict:
    values = np.asarray(ms, dtype=np.float64)
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95))}


# PatchCore CPU 검사기의 단계별 지연. backbone_ms는 정규화 + ONNX 백본, search_ms는 뱅크 최근접 탐색 + 이미지
# 점수, map_ms는 히트맵 계산이다. 이미지 목록을 돌려 쓰고(i % N), 앞의 warmup회는 세지 않는다(첫 호출의 메모리
# 할당·스레드 풀 준비 시간을 빼려는 것). 결과에 뱅크 행 수·정밀도·스레드 수를 같이 남긴다. 리사이즈는 여기
# 들어가지 않고 main이 time_resize 값을 더한다.
# 호출 위치: main. 4단계 표의 "리사이즈 7.7 + 백본 91.1 + 최근접 탐색 83.7 + 맵 2.0ms" 같은 분해가 이 값이다.
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


# Dinomaly(복원 방식) 검사기의 지연. inspector.run 한 번이 정규화 + ONNX 호출(점수와 256×256 맵)이라 전체
# 시간만 잰다. 6단계 관문 1(학습 전 모델)과 학습 뒤 아티팩트의 지연(dms-280-car FP32 추론 141.7ms + 리사이즈
# 8.0ms = 149.7ms)이 이 함수로 잰 값이다. 호출 위치: main. runtime_compare는 같은 방식으로 원시 값을 직접
# 잰다.
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


# GPU torch 경로를 한 장씩 재기 위한 채점기. patchcore.score_images는 부를 때마다 뱅크를 장치로 옮기고
# 중심화하고 행 노름을 계산하는데, 서비스라면 이것을 한 번만 한다. 그 비용은 뱅크가 클수록 커지므로 이미지마다
# 매기면 지연이 부풀려진다(docstring). 그래서 뱅크 준비는 생성자에서 한 번 하고, 이미지마다의 일은
# score_images와 같은 patchcore 내부 함수 순서를 그대로 부른다.
class _TorchScorer:
    """`patchcore.score_images` for one image at a time, with the bank prepared once.

    `score_images` copies the bank to the device, centres it and computes its row norms on every call.
    A service does that once, and its cost grows with the bank, so timing `score_images` per image would
    charge every image for it. The per-image work here is the same sequence of patchcore helpers.
    """

    def __init__(
        self, extractor, bank, *, device: str = "cuda", reweight_k: int = 9, sigma: float = 4.0, map_size=256
    ):
        # torch와 patchcore는 GPU 측정을 할 때만 불러온다. 이 모듈의 CPU 측정은 torch 없이(serve 의존성만으로)
        # 돌아야 하기 때문이다.
        import torch

        from . import patchcore

        if bank.ndim != 2 or bank.shape[0] == 0:
            raise ValueError(f"expected a non-empty bank [R, D], got {tuple(bank.shape)}")
        self._patchcore = patchcore
        self.device = torch.device(device)
        self.extractor = extractor.to(self.device).eval()
        # 뱅크를 fp32로 장치에 복사하고 평균을 빼 둔다(질의도 같은 평균을 뺀다). 거리는 그대로이고 ||a||² +
        # ||b||² − 2a·b 전개식의 반올림 오차가 줄어든다. 행 노름 제곱도 미리 계산해 매 탐색에 다시 쓴다.
        self.bank = bank.to(device=self.device, dtype=torch.float32, copy=True)
        self.centre = self.bank.mean(dim=0, keepdim=True)
        self.bank -= self.centre
        self.bank_sq = torch.linalg.vector_norm(self.bank, dim=1).square_()
        self.kernel = patchcore.gaussian_kernel1d(sigma).to(self.device)
        self.reweight_k = reweight_k
        self.map_size = map_size

    # 한 장 채점: 특징 추출 → 뱅크 평균 빼기 → 최근접 탐색 → PatchCore 재가중 이미지 점수 → 패치 점수를 입력
    # 크기로 키우고 가우시안 흐림 → 256×256 맵. 결과를 호스트(CPU 메모리)로 가져오는 것까지 포함해, 서비스가
    # 실제로 기다리는 시간을 잰다.
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


# GPU torch 파이프라인(fp16)의 한 장당 지연. 뱅크 준비 시간은 bank_setup_ms로 따로 적고 지연에서는 뺀다. 4단계
# 표의 "GPU fp16 ms"(예: wrn50-256 코어셋 1% 10.8ms)가 이 값이다. 호출 위치: main(`--gpu`).
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

    # CUDA 연산은 비동기라 그냥 시각을 읽으면 GPU 일이 끝나기 전 시각이 된다. 시각을 읽기 전에
    # torch.cuda.synchronize로 GPU 작업이 끝나기를 기다린다.
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


# 1500×1000 RGB 사진(시드 0의 무작위 픽셀)을 입력 크기로 PIL BICUBIC 리사이즈하는 시간의 중앙값(20회).
# 서비스와 캐시가 쓰는 것과 같은 리사이즈다. 4·6단계에서 약 8ms였다.
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


# 측정한 CPU 이름을 결과에 남긴다. Windows에서 platform.processor()는 "AMD64 Family ..." 같은 일반 문자열만
# 주므로, 레지스트리의 ProcessorNameString(예: AMD Ryzen 5 3600)을 읽고 실패하면 그 문자열을 그대로 쓴다.
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


# 지연 파일(JSON)에 key 항목을 합쳐 쓴다. 파일 하나에 여러 구성(키)의 결과가 쌓이고, 같은 키도 기존 필드를
# 지우지 않고 새 필드로 덮어쓴다. 그래서 CPU 측정과 GPU 측정을 따로 돌려도 한 항목에 모인다.
def merge_into(path: Path, key: str, entry: dict) -> dict:
    """Add `entry` under `key` to the JSON file at `path` (created if missing); returns the whole content."""
    data = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    data[key] = {**data.get(key, {}), **entry}
    _write_json(path, data)
    return data


# 명령행 입구. 흐름: 아티팩트의 범주 폴더(meta.json)를 읽음 → 같은 범주의 dev 풀 정상 앞 N장을 리사이즈
# 캐시에서 가져옴 → 아티팩트 종류(meta의 kind)에 맞는 정밀도인지 확인 → 리사이즈 시간 측정 → 정밀도마다 CPU
# 지연을 재고 total_with_resize_ms(판정값)를 더함 → (PatchCore이고 --gpu면) GPU 지연 → latency 파일에 합쳐 씀.
# 정밀도: PatchCore 아티팩트는 fp32·int8(정적 양자화), 복원 아티팩트는 fp32·int8-dynamic(MatMul만 동적
# 양자화)이다.
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
    # 지연용 이미지는 dev 풀 정상(학습 풀, 봉인 테스트 아님)이므로 장부 기록 없이 읽는다. 뱅크가 범주마다 달라
    # PatchCore 지연은 범주에 따라 다르고, 측정은 pcb1 하나로 정했다(4단계 규칙).
    rows = select(manifest, protocol="dev", part="pool_normal", category=args.category)[: args.images]
    images = ImageCache(paths.CACHE, meta["img_size"]).images(rows)

    reconstruction = meta.get("kind") == RECONSTRUCTION
    allowed = set(RECONSTRUCTION_PRECISIONS) if reconstruction else {"fp32", "int8"}
    if not set(args.precision) <= allowed:
        parser.error(f"precisions of this artifact: {sorted(allowed)}")
    # --gpu는 PatchCore의 torch 파이프라인을 재는 옵션이라 복원 아티팩트에는 쓸 수 없다.
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
