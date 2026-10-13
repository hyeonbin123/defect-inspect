"""Export a patch-feature extractor to ONNX, quantize it to INT8, and build inspector artifacts.

The ONNX model has input `image` float32 [1, 3, S, S] (ImageNet-normalised) and output `features`
float32 [1, H, W, D]: exactly what the torch extractors in `backbones.py` return. An artifact set is a
folder with the model file(s) shared by all categories and one sub-folder per category holding its memory
bank and `meta.json` (threshold, grid, scoring options).
"""

# [파일 역할] 4단계(지연 예산과 내보내기)의 PatchCore 서빙 아티팩트를 만든다. 패치 특징 추출기를 ONNX로
# 내보내고, 가능하면 INT8로 정적 양자화하고, run_grid가 저장한 범주별 메모리 뱅크와 교차 적합 임계값을 범주
# 폴더(bank.npy, meta.json)로 묶는다.
# [왜 필요한가] 질문 4('CPU에서 장당 200ms 안에 들면서 오검출률 기준을 지키는 구성이 남는가')를 재려면 GPU
# torch 파이프라인과 같은 점수를 내는 CPU 파이프라인(onnxruntime, torch 없음)이 있어야 한다. 아티팩트 폴더
# 하나에 모델과 범주별 뱅크·임계값을 모아 두면 검사 서비스(inspector.py, service.py)와 지연 측정(bench.py),
# CPU 재채점(run_export_eval.py)이 같은 파일을 읽는다.
# [흐름] run_grid --save-banks 실행 폴더 읽기 → 백본을 ONNX(FP32)로 내보내고 torch와 출력 대조 → (선택) dev 풀
# 정상으로 INT8 정적 양자화 → 범주마다 뱅크를 코어셋 비율만큼 잘라 저장하고, 교차 적합 점수로 conformal
# 임계값을 계산해 meta.json에 적음 → artifacts.json 요약.
# [연결] 실행: `python -m defect_inspect.export --grid outputs/grid-<설정>-<절차> --ratio 0.01 --out
# artifacts/<이름>`(README 재현 명령 4단계). 6단계 Dinomaly ViT-S 아티팩트는 dinomaly_serving.py가 따로
# 만든다.

import argparse
import copy
import json
import time
from pathlib import Path

# torch·onnx·onnxruntime은 함수 안에서 늦게 import한다. torch 없이 도는 CI(.github/workflows/ci.yml)에서도
# 이 모듈을 import할 수 있게 하려는 것이다.
import numpy as np

# conformal_threshold: 정상 점수 n개와 목표 α로 임계값을 정하는 보정 함수(calibrate.py). inspector의
# ARTIFACT_VERSION·PRECISIONS(정밀도별 모델 파일 이름)·check_meta·preprocess를 그대로 써서, 만드는 쪽과 읽는
# 쪽(서비스)이 같은 형식과 같은 전처리를 쓰게 한다. ratio_name·ratio_rows: run_grid와 같은 폴더 이름·뱅크 행
# 수 규칙.
from . import paths
from .calibrate import conformal_threshold
from .inspector import ARTIFACT_VERSION, PRECISIONS, check_meta, preprocess
from .ledger import git_commit
from .run_grid import ratio_name, ratio_rows


# 내보내기와 대조는 CPU fp32 eval 모드 모델로 해야 하므로, 이미 그 상태면 그대로 쓰고 아니면 깊은 복사본을
# 만들어 옮긴다. 호출한 쪽의 모델(예: GPU fp16)을 몰래 옮기거나 바꾸지 않기 위해서다.
def _cpu_fp32(extractor):
    """`extractor` on the CPU in fp32 and eval mode: a copy unless it already is (the caller's stays put)."""
    import torch

    tensors = [*extractor.parameters(), *extractor.buffers()]
    ready = not extractor.training and all(
        t.device.type == "cpu" and (not t.is_floating_point() or t.dtype == torch.float32) for t in tensors
    )
    return extractor if ready else copy.deepcopy(extractor).to("cpu").float().eval()


# 추출기를 배치 1, 고정 입력 크기 S의 ONNX로 내보낸다. 입력 이름 image, 출력 이름 features로 고정해 서빙
# 코드가 이름으로 부른다.
# 왜 dynamo=False(TorchScript 기반 내보내기)인가: WideResNet과 timm DINOv2 추출기를 모두 처리하기
# 때문이다(docstring). 내보낸 뒤 onnx.checker로 구조를 점검하고, seed 0 무작위 이미지 한 장으로 torch 출력과
# onnxruntime 출력의 최대 절대 차이가 1e-3을 넘으면 실패로 본다.
# 호출 위치: build_artifacts, tests/test_export.py.
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
    # 대조용 이미지는 seed를 고정한 numpy Generator로 만들어 결정적이다. 값이 실제 사진은 아니지만 그래프가
    # 같은 계산을 하는지 보기에는 충분하다.
    rng = np.random.default_rng(0)
    probe = rng.integers(0, 256, (1, img_size, img_size, 3), dtype=np.uint8)
    parity = check_parity(extractor, path, probe)
    if parity["max_abs"] > 1e-3:
        raise RuntimeError(f"ONNX export differs from torch: {parity}")
    return path


# onnxruntime CPU 추론 세션을 만든다. threads를 주면 연산 내부 스레드 수를 고정하고, 주지 않으면 onnxruntime
# 기본값을 쓴다(4단계 지연 규칙: 스레드 수는 onnxruntime 기본값).
def _session(path: Path, threads: int | None = None):
    import onnxruntime as ort

    options = ort.SessionOptions()
    if threads is not None:
        options.intra_op_num_threads = int(threads)
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


# 같은 uint8 이미지들에 대해 torch fp32 특징과 onnxruntime 특징의 차이(최대 절대, 평균 절대, 최대값 대비
# 상대)를 잰다. 전처리는 서비스와 같은 inspector.preprocess를 써서 실제 서빙 입력과 같은 값으로 대조한다.
# 호출 위치: export_onnx(내보낸 직후 검증), tests/test_export.py.
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


# onnxruntime 정적 양자화의 보정 데이터 공급자(CalibrationDataReader 인터페이스). get_next가 호출될 때마다
# 이미지 한 장을 서비스와 같은 전처리로 바꿔 주고, 다 쓰면 None을 돌려준다. 보정 입력 분포가 실제 서빙 입력과
# 같아야 활성값 범위(MinMax)가 맞게 잡힌다.
class _Reader:
    """Calibration images for onnxruntime's static quantization, preprocessed like the service does."""

    def __init__(self, images: np.ndarray):
        self._images = iter(images)

    def get_next(self):
        image = next(self._images, None)
        return None if image is None else {"image": preprocess(image, image.shape[0])}


# INT8 정적 양자화: QDQ 형식, MinMax 보정, 활성값 uint8, 가중치 int8(채널별). 4단계 규칙(experiments.md 'test
# 절차': INT8은 QDQ, MinMax, 범주마다 dev 풀 정상 8장씩 96장으로 보정)을 그대로 옮긴 설정이다.
# DINOv2 내보내기는 onnxruntime 전처리(기호 형상 추론)가 클래스·레지스터 토큰의 Expand 노드에서 멈춘다. 그
# 단계를 건너뛰면 양자화는 끝나지만 특징이 FP32와 상관 0.07로 사실상 무관해지므로, 대체 경로 없이
# NotImplementedError로 알리고 모델을 남기지 않는다(규칙 변경 5).
# 결과(README): WideResNet INT8은 점수가 중앙값 24% 달라져 FP32 임계값을 그대로 쓰면 정상의 68%를 불량으로
# 판정했다(H14 지지).
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
    # 전처리한 중간 모델은 임시 파일(.prep.onnx)이고 finally에서 항상 지운다.
    prepared = int8_path.with_suffix(".prep.onnx")
    try:
        try:
            quant_pre_process(str(fp32_path), str(prepared))
        # 디스크·메모리 문제는 그래프 탓이 아니므로 그대로 올려 보낸다. 'INT8 없이 계속'으로 처리하면 기계
        # 문제를 숨기게 된다.
        except (OSError, MemoryError):
            raise  # the machine, not the graph: not a reason to go on without INT8
        # onnxruntime 형상 추론은 내부에서 어떤 예외든 낼 수 있어 넓게 잡고, 이유를 담은 NotImplementedError로
        # 바꿔 호출한 쪽(build_artifacts)이 'INT8 불가'로 기록하게 한다.
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


# run_grid --save-banks 실행 하나에서 한 코어셋 비율의 아티팩트 세트를 만든다. 입력: 격자 실행 폴더, 비율,
# 출력 폴더, 목표 오검출률 alpha(기본 5%), INT8 여부, 범주당 INT8 보정 장수(기본 8). 출력: artifacts.json에
# 쓰는 요약 dict.
# 임계값은 그 실행의 교차 적합 점수(torch 파이프라인)로 정한 conformal 임계값이다. 교차 적합은 1단계에서 목표
# 5%에 실제 4.0%로 목표를 넘지 않는 보수적인 방식으로 확인됐다(README). INT8이 안 되면(DINOv2) FP32
# 아티팩트만으로도 완전하고, artifacts.json의 int8에 이유가 남는다.
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
    # torch가 필요한 백본, 캐시, 매니페스트 도구는 이 함수가 실제로 돌 때만 import한다.
    from .backbones import make_extractor
    from .cache import ImageCache
    from .splits import read_manifest, select

    grid_dir, out_dir = Path(grid_dir), Path(out_dir)
    with open(grid_dir / "run.json", encoding="utf-8") as f:
        grid = json.load(f)
    # 뱅크를 저장하지 않은 격자 실행에서는 아티팩트를 만들 수 없으므로 일찍 멈춘다.
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

    # INT8 보정 이미지를 고를 분할 매니페스트(manifests/visa.csv)를 읽는다.
    manifest = read_manifest(paths.VISA_MANIFEST)
    int8_path = out_dir / PRECISIONS["int8"]
    # INT8 모델은 FP32 모델에서 만들어지므로, 이전 내보내기가 같은 폴더에 남긴 int8 파일은 새 FP32 모델과 짝이
    # 맞지 않는다. 먼저 지우고, 양자화가 성공했을 때만 다시 생긴다.
    # A quantized model left by an earlier export into this folder does not belong to the new fp32 model.
    int8_path.unlink(missing_ok=True)
    if int8:
        # INT8 보정 이미지: 격자 실행의 범주마다 dev 절차 학습 풀 정상(pool_normal)에서 선택 순서의 앞
        # calibration_per_category장(기본 8장, 12개 범주 96장). 봉인 테스트 이미지는 쓰지 않는다. 리사이즈
        # 캐시에서 읽어 입력 크기가 학습·채점과 같다.
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

    # 아티팩트를 만든 코드 버전을 meta.json과 artifacts.json에 남긴다.
    commit = git_commit()
    for cat_info in ratio_run["categories"]:
        category = cat_info["category"]
        # 격자 실행은 10% 코어셋 뱅크 하나만 저장한다. 탐욕적 k-center 선택은 앞에서부터 포함 관계라, 앞부분
        # ratio_rows(n_full, ratio)행만 잘라 1%·0.1% 뱅크를 얻는다(experiments.md 4단계 '격자').
        with open(grid_dir / "banks" / f"{category}.json", encoding="utf-8") as f:
            n_full = json.load(f)["n_features"]["full"]
        bank = np.load(grid_dir / "banks" / f"{category}_full.npy")[: ratio_rows(n_full, ratio)]
        # pool_score_oof: 교차 적합 점수. 정상 풀의 각 이미지를 '그 겹을 뺀 뱅크'로 매긴 점수라 자기 자신이
        # 뱅크에 들어간 재대입 편향이 없다(README: 재대입으로 잡으면 정상의 86%가 불량). 이 점수로 목표
        # alpha의 conformal 임계값을 정한다.
        with np.load(grid_dir / ratio_name(ratio) / f"{category}.npz") as z:
            thr = conformal_threshold(z["pool_score_oof"], alpha)
        # meta.json: 서비스가 이 범주를 채점하는 데 필요한 모든 값(입력 크기, 패치 격자, 특징 차원, 재가중
        # 이웃 수, 맵 σ, 임계값과 목표 α, 보정 방식·표본 수·보장 여부)과 출처(격자 폴더, 비율, 절차, 커밋)를
        # 적는다. 저장 전에 inspector.check_meta로 서비스가 읽을 수 있는 형식인지 확인한다.
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
        # JSON은 UTF-8, 줄바꿈 LF로 써서 운영체제와 상관없이 같은 파일이 나오게 한다.
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


# 명령줄 진입점. --grid(격자 실행 폴더), --ratio(코어셋 비율), --out(아티팩트 폴더)은 필수이고, --no-int8은
# DINOv2처럼 양자화가 안 되는 설정에 쓴다(README: DINOv2는 export --no-int8).
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
