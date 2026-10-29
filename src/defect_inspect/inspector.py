"""서빙 쪽 파이프라인: torch 없이 ONNX Runtime 특징 + numpy PatchCore 채점.

뱅크와 임계값은 torch 파이프라인(`patchcore.score_images`, GPU fp16)에서 온다. 이 모듈은 같은 계산을 numpy로
다시 구현해 서비스가 onnxruntime만으로 CPU에서 돌 수 있게 한다: float16으로 반올림한 특징, 뱅크 중심화,
최근접 뱅크 행, PatchCore 재가중, 점수 맵(bilinear 확대, 가우시안 흐림, 256 격자로 리사이즈).
"""

# [파일 역할] 검사 서비스가 쓰는 torch 없는 추론 경로다. 범주 하나를 맡는 검사기 두 종류를 둔다:
# Inspector(4단계 PatchCore: onnxruntime 특징 + numpy 최근접 탐색 채점)와 ReconstructionInspector(6단계
# Dinomaly ViT-S: 점수와 맵을 ONNX 모델이 직접 냄). 아티팩트 형식 점검(check_meta 등)과 임계값
# 재보정(calibrate)도 여기 있다.
# [왜 필요한가] 질문 4는 'CPU에서 장당 200ms'이고, 서비스 Docker 이미지는 torch 없이 onnxruntime만
# 넣는다(README 설치: serve 그룹에는 torch가 없다). 그래서 GPU torch 파이프라인(patchcore.score_images)의
# 계산(fp16 반올림, 뱅크 중심화, 최근접 행, 재가중, 점수 맵)을 numpy로 똑같이 다시 구현해, 서빙 점수가 실험
# 점수와 같게 한다. 4단계 test에서 onnxruntime FP32 점수는 GPU 점수와 중앙값 0.012% 차이였고 GPU 임계값을
# 그대로 써도 오검출률이 같았다(README).
# [주요 구성] 전처리(preprocess) → PatchCore 수학(nearest, image_score, resize_bilinear, gaussian_kernel1d,
# score_map) → 아티팩트·모델 점검(check_meta, check_model_io, _check_bank) → Inspector → 재구성
# 아티팩트(check_reconstruction_meta, artifact_kind, ReconstructionInspector) → save_artifact.
# [연결] service.py(load_inspectors가 meta의 kind로 클래스를 고름), bench.py(지연 측정),
# run_export_eval.py(4단계 CPU 재채점), dinomaly_serving.py(6단계 채점·보정), runtime_compare.py(OpenVINO
# 대조), export.py(형식 상수·전처리 재사용).

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# numpy: 최근접 탐색과 점수 계산. PIL: 업로드 사진 디코딩 결과를 RGB로 바꾸고 리사이즈. scipy.ndimage: 점수
# 맵의 분리형 가우시안 흐림(correlate1d). torch는 쓰지 않는다.
import numpy as np
from PIL import Image
from scipy import ndimage

# conformal_threshold: 정상 점수로 목표 오검출률 α의 임계값을 정하는 함수. 서비스의 /calibrate와 같은 규칙을
# 쓴다.
from .calibrate import conformal_threshold

# ARTIFACT_VERSION: 아티팩트 meta.json 형식 버전. 다른 버전이면 읽지 않는다. IMAGENET_MEAN/STD: 백본의
# ImageNet 정규화 값(experiments.md '전처리'). MAP_SIZE: 점수 맵을 맞추는 256 격자(픽셀 지표와 같은 격자).
ARTIFACT_VERSION = 1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MAP_SIZE = 256
# 뱅크를 탐색할 때 한 번에 만드는 [쿼리 행, 뱅크 행] 블록의 최대 크기(2**27개 fp32 = 512MB). torch
# 쪽(patchcore._MAX_DIST_ELEMS)과 같은 제한이다.
_MAX_DIST_ELEMS = 1 << 27
# PatchCore 아티팩트 meta.json에 반드시 있어야 하는 키. export.build_artifacts가 쓰고 check_meta가 확인한다.
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
# 정밀도별 PatchCore 모델 파일 이름. 아티팩트 세트 폴더 하나에 모든 범주가 같이 쓰는 모델 파일이 있고, 범주
# 폴더에는 뱅크와 meta.json만 있다.
PRECISIONS = {"fp32": "model_fp32.onnx", "int8": "model_int8.onnx"}
PATCHCORE = "patchcore"


# 검사 한 번의 결과: 이상 점수, 그때의 임계값, 불량 여부(score > threshold), 히트맵(요청했을 때만 256×256,
# 점수와 같은 척도). service.py가 이 값을 응답으로 바꾼다.
@dataclass
class Inspection:
    score: float
    threshold: float
    is_defect: bool
    heatmap: np.ndarray | None  # 점수 척도의 float32 [256, 256], 요청하지 않았으면 None


# 리사이즈 캐시(실험)와 같은 방식이라 서빙 입력이 실험 입력과 같아진다(experiments.md '전처리': 비율을
# 무시하고 정사각형으로 BICUBIC). export.py의 대조·INT8 보정, bench.py, 서비스가 모두 이 함수를 써서 전처리가
# 한 곳에만 있다.
def preprocess(image: Image.Image | np.ndarray, size: int) -> np.ndarray:
    """이미지를 모델 입력, 곧 ImageNet 정규화한 float32 [1, 3, size, size]로 바꾼다.

    PIL 이미지(업로드 사진, 아무 크기)는 리사이즈 캐시처럼 RGB로 바꾸고 BICUBIC으로 줄인다. uint8
    배열(캐시에서 읽은 이미지)은 이미 [size, size, 3]이어야 한다.
    """
    if isinstance(image, Image.Image):
        if image.size != (size, size) or image.mode != "RGB":
            image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.uint8)
    else:
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.shape != (size, size, 3):
            raise ValueError(f"expected a uint8 array [{size}, {size}, 3], got {array.dtype} {array.shape}")
    # 0~1로 나눈 뒤 채널별 평균을 빼고 표준편차로 나누고, HWC를 CHW로 바꿔 배치 축을 붙인다.
    # ascontiguousarray로 onnxruntime이 복사 없이 읽을 수 있는 연속 메모리로 만든다.
    x = array.astype(np.float32) / np.float32(255.0)
    x = (x - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


# patchcore._nearest의 numpy 판이다. 탐색의 전개식은 ||b||² - 2q·b다.
# 4단계 dev 지연 분해에서 WideResNet 10% 뱅크는 이 최근접 탐색이 CPU 지연의 82~91%였다(experiments.md 4단계).
# 그래서 코어셋을 1%로 줄이는 것이 지연을 줄이는 주된 방법이었다.
def nearest(query: np.ndarray, bank: np.ndarray, bank_sq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """쿼리 행마다 가장 가까운 뱅크 행까지의 유클리드 거리와 그 인덱스(둘 다 중심화한 fp32).

    탐색은 블록 단위로 전개식을 쓰고, 찾은 행까지의 거리는 q - b로 직접 다시 계산한다. 그래서 0 근처에서
    큰 수끼리 빼는 오차(상쇄)를 겪지 않는다.
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


# patchcore._image_scores의 이미지 한 장 판. 재가중은 가장 큰 패치 점수 s*에 1 - softmax(이웃 거리)[m*]를
# 곱한다(reweight_k=9가 P0 기본값).
# 차이점: torch topk 대신 argpartition으로 k개만 고른 뒤 거리 순으로 안정 정렬한다. -inf로 둔 m*가 맨 앞에
# 와서 softmax의 0번 칸이 m*가 되는 점은 torch 쪽과 같다.
def image_score(
    patch_scores: np.ndarray,
    nearest_index: np.ndarray,
    query: np.ndarray,
    bank: np.ndarray,
    bank_sq: np.ndarray,
    reweight_k: int = 9,
) -> float:
    """이미지 한 장의 PatchCore 이미지 점수: 가장 큰 패치 점수를 그 뱅크 이웃으로 재가중한 값."""
    p_star = int(patch_scores.argmax())
    s_star = float(patch_scores[p_star])
    if reweight_k <= 1:
        return s_star
    k = min(reweight_k, bank.shape[0])
    m_star = int(nearest_index[p_star])
    block = bank_sq - 2.0 * (bank @ bank[m_star])
    block[m_star] = -np.inf  # 뱅크에 m*와 같은 행이 있어도 m* 자신이 맨 앞에 온다
    support = np.argpartition(block, k - 1)[:k] if k < bank.shape[0] else np.arange(bank.shape[0])
    support = support[np.argsort(block[support], kind="stable")]
    # softmax를 직접 쓰지 않고 exp(d - max d)로 계산한다. 최댓값을 빼도 비율은 같고, 거리가 커도 exp가 넘치지
    # 않는다.
    dist = np.linalg.norm(query[p_star][None, :] - bank[support], axis=1).astype(np.float64)
    weights = np.exp(dist - dist.max())
    weight = 1.0 - weights[0] / weights.sum()
    # 임계값은 torch 파이프라인의 float32 점수에서 나왔으므로, 서빙 점수도 float32로 반올림해 float64의 남는
    # 자릿수 때문에 임계값 근처 판정이 갈리지 않게 한다.
    return float(np.float32(weight * s_star))


# 출력 픽셀 중심을 원본 좌표로 옮겨 (i + 0.5) × (n_in / n_out) - 0.5, 음수는 0으로 자른다. 아래쪽·위쪽 이웃
# 인덱스와 보간 가중치를 돌려준다.
def _bilinear_axis(n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """torch bilinear 보간(`align_corners=False`)과 같은 원본 인덱스와 가중치를 계산한다."""
    src = (np.arange(n_out, dtype=np.float64) + 0.5) * (n_in / n_out) - 0.5
    src = np.clip(src, 0.0, None)
    lo = np.minimum(np.floor(src).astype(np.int64), n_in - 1)
    hi = np.minimum(lo + 1, n_in - 1)
    return lo, hi, (src - lo).astype(np.float32)


# 행 방향 보간 후 열 방향 보간(분리형)이다. torch 파이프라인의 점수 맵과 같은 값을 내도록 torch의 좌표
# 규칙(_bilinear_axis)을 그대로 옮겼다.
def resize_bilinear(a: np.ndarray, size: int) -> np.ndarray:
    """[H, W] float32 -> [size, size]. torch의 안티앨리어싱 없는 bilinear 보간과 같다."""
    lo_r, hi_r, w_r = _bilinear_axis(a.shape[0], size)
    lo_c, hi_c, w_c = _bilinear_axis(a.shape[1], size)
    rows = a[lo_r] * (1.0 - w_r)[:, None] + a[hi_r] * w_r[:, None]
    return (rows[:, lo_c] * (1.0 - w_c)[None, :] + rows[:, hi_c] * w_c[None, :]).astype(np.float32)


# patchcore.gaussian_kernel1d와 같은 크기 규칙(반지름 int(4σ + 0.5))이다.
def gaussian_kernel1d(sigma: float) -> np.ndarray:
    """크기 `2 * int(4 * sigma + 0.5) + 1`의 정규화한 가우시안 커널(sigma <= 0이면 1 하나)."""
    if sigma <= 0:
        return np.ones(1, dtype=np.float32)
    radius = int(4 * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return (kernel / kernel.sum()).astype(np.float32)


# torch 쪽 score_images의 맵 단계와 같은 순서다. Inspector.inspect가 heatmap=True일 때 부르고, bench.py도
# 지연 측정에 쓴다.
def score_map(
    patch_scores: np.ndarray, size: int, sigma: float = 4.0, map_size: int = MAP_SIZE
) -> np.ndarray:
    """패치 점수 [H, W] -> float32 맵 [map_size, map_size]: `size`로 확대, 가우시안 흐림, 리사이즈."""
    out = resize_bilinear(patch_scores.astype(np.float32), size)
    kernel = gaussian_kernel1d(sigma)
    if kernel.size > 1:
        # mirror(=torch reflect) 패딩은 반지름이 맵 크기보다 작아야 하므로 미리 막는다.
        if kernel.size // 2 >= size:
            raise ValueError(f"blur radius {kernel.size // 2} needs a map larger than {size}")
        # scipy "mirror"는 가장자리 표본을 반복하지 않는 반사로, torch의 "reflect" 패딩과 같다.
        out = ndimage.correlate1d(out, kernel, axis=0, mode="mirror")
        out = ndimage.correlate1d(out, kernel, axis=1, mode="mirror")
    if size != map_size:
        out = resize_bilinear(out, map_size)
    return out.astype(np.float32)


# JSON에서 읽은 값이 정수인지 확인한다. bool은 int의 하위 클래스라 true/false가 1/0으로 통과하지 않게 따로
# 뺀다.
def _is_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


# 유한한 수인지 확인한다(bool 제외). float 범위를 넘는 큰 정수는 float 변환에서 OverflowError가 나므로 False로
# 처리한다.
def _is_finite_number(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:  # float 범위를 넘는 정수
        return False


# 키 누락, 버전, 양의 정수(img_size, dim, grid), 음수가 아닌 reweight_k·sigma, 유한한 threshold를 확인한다.
# 잘못된 아티팩트를 서비스 시작 때 바로 거절해, 요청을 처리하다 이상한 값으로 판정하는 일을 막는다.
# 호출 위치: export.build_artifacts(쓰기 전), Inspector 생성자, Inspector.load, save_artifact,
# run_export_eval.
def check_meta(meta: dict) -> None:
    """PatchCore 아티팩트 `meta`에 모든 키가 쓸 수 있는 값으로 있지 않으면 ValueError를 낸다."""
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
    # 가우시안 커널 반지름이 입력 크기 이상이면 score_map의 반사 패딩이 실패하므로, 요청 때가 아니라 읽을 때
    # 거절한다.
    radius = int(4 * float(meta["sigma"]) + 0.5)  # `gaussian_kernel1d`의 커널 반지름
    if radius >= meta["img_size"]:
        raise ValueError(
            f"artifact meta: sigma {meta['sigma']!r} gives a blur radius of {radius}, which needs an "
            f"input larger than {meta['img_size']}"
        )
    if not _is_finite_number(meta["threshold"]):
        raise ValueError(f"artifact meta: threshold must be a finite number, got {meta['threshold']!r}")


# 다른 설정의 모델 파일이 섞여 들어오는 것을 막는다.
def check_model_io(session, meta: dict) -> None:
    """ONNX 세션이 선언한 입력·출력 이름과 모양이 `meta`와 맞지 않으면 ValueError를 낸다.

    모델은 `image` [1, 3, S, S]를 받아 `features` [1, H, W, dim]을 돌려줘야 한다. 모델이 기호로 남긴
    차원(정수가 아닌 값)은 비교하지 않는다.
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


# 뱅크가 [R, dim]의 비어 있지 않은 실수 배열이고 fp32로 바꿨을 때 모두 유한한지 확인한다. 뱅크는 fp16으로
# 저장되지만 거리 계산은 fp32로 한다(torch 쪽과 같음).
def _check_bank(bank: np.ndarray, dim: int) -> np.ndarray:
    """메모리 뱅크를 fp32 배열 [R, dim] 하나로 돌려주고, 쓸 수 없으면 ValueError를 낸다."""
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


# 세션은 모든 범주가 함께 쓰는 모델이다.
# 호출 위치: service.load_inspectors, bench.py(Inspector.load), run_export_eval(범주·비율마다 생성),
# service.offline_inspector(모델 파일 없이 도는 합성 demo 범주).
class Inspector:
    """PatchCore 범주 하나의 CPU 검사기: ONNX Runtime 세션, 메모리 뱅크, 임계값.

    점수가 유한하지 않으면 판정하지 않고 `FloatingPointError`를 낸다. `nan > threshold`는 거짓이라, 막지
    않으면 NaN 점수가 정상으로 통과한다.
    """

    # 클래스 속성: kind는 /categories 응답에 나가는 모델 종류, precision은 읽은 모델 정밀도, threads는 세션을
    # 만들 때 준 연산 스레드 수(None이면 onnxruntime 기본값).
    kind = PATCHCORE
    precision = "fp32"
    threads: int | None = None

    def __init__(self, session, bank: np.ndarray, meta: dict):
        check_meta(meta)
        bank32 = _check_bank(bank, meta["dim"])
        # 테스트는 가짜 세션(run만 있는 객체)을 넘길 수 있으므로, 입출력 정보를 가진 진짜 세션일 때만 모델
        # 입출력을 점검한다.
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
        # torch 쪽 score_images와 같은 중심화: 뱅크 평균을 뺀 fp32 사본 하나를 두고, 쿼리도 같은 centre를
        # 뺀다. 중심화해도 거리는 같고 전개식은 정확해진다. 행별 제곱 노름은 한 번만 계산한다.
        self._centre = bank32.mean(axis=0, keepdims=True)
        self._bank = bank32 - self._centre
        self._bank_sq = np.square(self._bank).sum(axis=1)

    # 범주 폴더에 모델 파일이 없으면 부모(아티팩트 세트) 폴더의 공용 모델을 쓴다. meta.json → bank.npy → 모델
    # 순으로 읽는다. check_meta를 모델을 열기 전에 불러, 오류 메시지가 onnxruntime 오류가 아니라 잘못된 meta
    # 값을 가리키게 한다. np.load는 allow_pickle=False로 피클 실행을 막는다.
    @classmethod
    def load(cls, artifact_dir: Path, *, precision: str = "fp32", threads: int | None = None) -> "Inspector":
        """아티팩트 범주 폴더에서 검사기를 만든다. 아티팩트가 무엇이 잘못됐든 `ValueError` 하나로 알린다."""
        import onnxruntime as ort

        artifact_dir = Path(artifact_dir)
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(PRECISIONS)}, got {precision!r}")
        model = artifact_dir / PRECISIONS[precision]
        if not model.exists():
            # 아티팩트 세트는 범주 폴더들 옆에 모델 파일 하나를 둔다.
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
            check_meta(meta)  # 모델을 열기 전에 불러 메시지가 잘못된 값을 가리키게 한다
        except ValueError as err:
            raise ValueError(f"{artifact_dir / 'meta.json'}: {err}") from err
        # meta에 모델 지문이 적혀 있으면 대조한다. 4단계 PatchCore 아티팩트는 적지 않았으므로 보통 건너뛴다.
        try:
            check_model_sha256(model, meta, precision)
        except ValueError as err:
            raise ValueError(f"{artifact_dir} ({model.name}): {err}") from err
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
        except Exception as err:  # onnxruntime은 읽지 못하는 파일에 자체 예외 형식을 낸다
            raise ValueError(f"{model} is not a loadable ONNX model: {err}") from err
        try:
            inspector = cls(session, bank, meta)
        except ValueError as err:
            raise ValueError(f"{artifact_dir} ({model.name}): {err}") from err
        inspector.precision = precision
        inspector.threads = None if threads is None else int(threads)
        return inspector

    # torch 파이프라인은 특징을 fp16으로 계산·저장하므로(patchcore._patch_features), 같은 반올림을 거쳐야
    # 뱅크와 같은 값의 격자에서 거리를 잰다.
    def features(self, image: Image.Image | np.ndarray) -> np.ndarray:
        """이미지 한 장의 패치 특징 float32 [H * W, D]. torch 파이프라인처럼 float16을 거쳐 반올림한다."""
        out = self.session.run(["features"], {"image": preprocess(image, self.size)})[0]
        if out.shape != (1, *self.grid, self._bank.shape[1]):
            raise ValueError(
                f"model returned features {out.shape}, expected {(1, *self.grid, self._bank.shape[1])}"
            )
        feats = out[0].reshape(-1, out.shape[-1])
        # fp16 범위를 넘는 값은 캐스트에서 inf가 되므로 경고 대신 캐스트 뒤에 직접 검사한다
        # (patchcore._patch_features와 같음). 원래 값이 비정상인지(모델 문제), 캐스트에서 넘쳤는지 구분해
        # 메시지를 다르게 낸다.
        with np.errstate(over="ignore"):
            rounded = feats.astype(np.float16)
        if not np.isfinite(rounded).all():
            if not np.isfinite(feats).all():
                raise FloatingPointError("the model produced non-finite features")
            raise FloatingPointError("patch features overflowed fp16")
        return rounded.astype(np.float32)

    # 쿼리 중심화 → 최근접 거리(패치 점수) → 재가중 이미지 점수. 특징을 따로 받는 것은 bench.py가 단계별
    # 시간(백본, 탐색, 맵)을 나눠 재기 위해서다.
    def score_features(self, feats: np.ndarray) -> tuple[float, np.ndarray]:
        """이미지 한 장의 특징으로 이미지 점수와 패치 점수 [H, W]를 계산한다."""
        feats = np.asarray(feats, dtype=np.float32)
        if not np.isfinite(feats).all():
            raise FloatingPointError("patch features are not finite")
        query = feats - self._centre
        dist, index = nearest(query, self._bank, self._bank_sq)
        score = image_score(dist, index, query, self._bank, self._bank_sq, self.reweight_k)
        if not math.isfinite(score):
            raise FloatingPointError("the anomaly score is not finite")
        return score, dist.reshape(self.grid)

    # 사진 한 장을 검사한다. heatmap=False면 맵 계산을 건너뛴다(서비스 /calibrate는 점수만 필요해 이 옵션을
    # 쓴다). 판정 규칙은 score > threshold.
    def inspect(self, image: Image.Image | np.ndarray, *, heatmap: bool = True) -> Inspection:
        score, patch_scores = self.score_features(self.features(image))
        heat = score_map(patch_scores, self.size, self.sigma) if heatmap else None
        return Inspection(
            score=score, threshold=self.threshold, is_defect=score > self.threshold, heatmap=heat
        )

    # run_export_eval이 봉인 테스트와 교차 적합 풀을 CPU 파이프라인으로 다시 채점할 때 쓴다. 서빙과 같은 한
    # 장씩 경로로 계산한다.
    def scores(self, images: np.ndarray) -> np.ndarray:
        """이미 입력 크기인 uint8 이미지 [N, S, S, 3]의 이미지 점수 float32 [N]만 낸다."""
        out = np.empty(len(images), dtype=np.float32)
        for i, image in enumerate(images):
            out[i] = self.score_features(self.features(image))[0]
        return out

    # 임계값을 바꾸고 meta에도 반영한다(서비스의 /calibrate·DELETE /calibrate가 쓴다). 유한한 수만 받는다.
    def set_threshold(self, value: float) -> None:
        if not _is_finite_number(value):
            raise ValueError(f"the threshold must be a finite number, got {value!r}")
        self.threshold = float(value)
        self.meta["threshold"] = float(value)

    # 뱅크는 그대로다. meta에 alpha와 보정 기록(holdout, 장수, 보장 여부)을 남긴다.
    # 왜 필요한가: 고정 임계값은 촬영 조건과 한 묶음이다. 조명이 바뀐 M2AD에서 새 조명의 정상 30장으로 다시
    # 잡으면 오검출률이 4.4%로 돌아왔다(README 3단계).
    # 호출 위치: service.offline_inspector, tests/test_inspector.py. 서비스의 /calibrate는 같은 규칙을 업로드
    # 처리 중에 직접 적용한다.
    def calibrate(self, images: Sequence[Image.Image | np.ndarray], alpha: float = 0.05) -> float:
        """현재 촬영 조건의 정상 이미지들로 임계값만 다시 잡는다(`alpha`의 conformal 순위)."""
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


# 6단계 Dinomaly(복원 오차로 점수를 내는 모델) 아티팩트의 종류 이름. 메모리 뱅크가 없고, 범주 폴더에는
# meta.json(임계값)만 있다.
RECONSTRUCTION = "reconstruction"
# 재구성(Dinomaly) 아티팩트 세트의 모델 파일 이름. INT8은 MatMul만 동적 양자화한 것이다. README: 동적
# INT8은 모델 경로 123ms로 빨라지지만 dev AUROC가 1.4%p 낮고, 아티팩트 임계값은 FP32 점수로 정한 것이라 서빙은
# FP32다.
RECONSTRUCTION_PRECISIONS = {"fp32": "model_fp32.onnx", "int8-dynamic": "model_int8_dynamic.onnx"}
RECONSTRUCTION_META_KEYS = ("version", "kind", "name", "category", "img_size", "map_size", "threshold")
# 정밀도 → meta.json `source`에서 그 정밀도 모델 파일의 SHA-256을 적는 키. `dinomaly_serving export`가 FP32
# 모델의 해시를 `onnx_sha256`에, 동적 INT8을 만들면 그 해시를 `onnx_int8_dynamic_sha256`에 적는다. 4단계
# PatchCore 아티팩트(`defect_inspect.export`)는 해시를 적지 않았으므로 이 표의 키가 meta에 없고, 그때는 대조를
# 건너뛴다(옛 아티팩트도 그대로 읽히게).
MODEL_SHA256_KEYS = {"fp32": "onnx_sha256", "int8-dynamic": "onnx_int8_dynamic_sha256"}
# SHA-256 16진수 표기에 쓰는 글자(소문자만).
SHA256_HEX_DIGITS = frozenset("0123456789abcdef")


# SHA-256 지문의 16진수 표기(소문자 64자리)인지 본다. meta 기록의 검사(check_model_sha256)와 배포 설정의 기대
# 지문 읽기(service.parse_expected_sha256)가 같은 규칙을 쓰게 하려고 한곳에 둔다. 대문자는 받지 않는다: meta
# 기록은 쓰는 쪽(dinomaly_serving export)이 늘 소문자로 적고, 배포 설정은 읽을 때 소문자로 바꾼 뒤 이 검사를
# 거친다. 입력이 문자열이 아니면(빈 값, 숫자 등) False다.
def is_sha256_hex(value) -> bool:
    """`value`가 소문자 64자리 16진수 문자열이면 True."""
    return isinstance(value, str) and len(value) == 64 and all(c in SHA256_HEX_DIGITS for c in value)


# 모델 파일의 SHA-256(소문자 16진수). digests는 실제 경로 → 지문 캐시로, 12개 범주가 같이 쓰는 파일을 한 번만
# 해시하려는 것이다(이 함수가 채운다). meta 대조(check_model_sha256)와 배포 설정의 기대 지문 대조
# (service.load_inspectors)가 같은 캐시를 써서, 둘 다 해도 파일마다 한 번만 읽는다. 경로는 resolve()한 실제
# 경로로 키를 만들어, 범주 폴더와 상위 폴더에서 같은 파일을 가리키는 경로가 하나로 묶이게 한다.
def model_file_sha256(model: Path, digests: dict[Path, str] | None = None) -> str:
    """모델 파일의 SHA-256 16진수 지문. `digests`가 있으면 그 캐시를 먼저 보고 새로 낸 값을 채운다."""
    from .download import sha256_file

    path = Path(model).resolve()
    if digests is not None and path in digests:
        return digests[path]
    actual = sha256_file(path)
    if digests is not None:
        digests[path] = actual
    return actual


# 모델 파일이 meta.json에 적힌 지문과 같은지 확인한다. 왜: 서비스는 모델(ONNX)과 임계값(meta.json)을 서로 다른
# 파일에서 읽는다. 누가 모델 파일만 다른 학습본으로 바꾸거나 파일이 깨지면, 서비스는 아무 오류 없이 다른
# 모델의 점수를 이 임계값과 비교해 조용히 틀린 판정을 낸다(임계값은 그 모델의 점수 분포로 정한 값이다). 서비스
# 교체 점검(docs/experiments.md)에서는 이 대조를 손으로 했고, 이제 시작할 때마다 코드가 한다. 입력: 모델 파일
# 경로, 그 범주의 meta, 정밀도. digests는 같은 파일을 여러 범주가 같이 쓸 때 해시를 한 번만 계산하려는
# 캐시(실제 경로 → 16진수 해시)이고, 이 함수가 채운다. 134MB FP32 모델의 해시는 HDD에서 처음 약 2초, 그 뒤 OS
# 캐시로 1초 안쪽이다. 출력: 대조했으면 그 해시, meta에 기록이 없어 건너뛰었으면 None. 다르면 두 해시의 앞
# 12자리를 담은 ValueError로 서비스 시작을 멈춘다(서비스는 이 오류를 범주 폴더 이름과 함께 다시 낸다). 주의:
# 기록이 문자열이 아니거나 64자리 16진수가 아니면 깨진 meta로 보고 멈춘다(빈 값으로 대조를 피하지 못하게).
def check_model_sha256(
    model: Path, meta: dict, precision: str, digests: dict[Path, str] | None = None
) -> str | None:
    """`meta["source"]`에 이 정밀도 모델의 SHA-256이 적혀 있으면 `model` 파일과 대조한다.

    같으면 그 해시를, 기록이 없으면 None을 돌려준다. 다르거나 기록이 해시 형식이 아니면 ValueError를 낸다.
    """
    key = MODEL_SHA256_KEYS.get(precision)
    source = meta.get("source") if isinstance(meta, dict) else None
    if key is None or not isinstance(source, dict) or key not in source:
        return None
    expected = source[key]
    if not is_sha256_hex(expected):
        raise ValueError(f"artifact meta: source.{key} is not a SHA-256 hex digest: {expected!r}")
    actual = model_file_sha256(model, digests)
    if actual != expected:
        raise ValueError(
            f"{Path(model).name} does not match the artifact meta: sha256 {actual[:12]}... on disk, "
            f"{expected[:12]}... in source.{key} (the thresholds were set for another model file)"
        )
    return actual


# 점검 항목: 키, kind, 버전, 양의 정수 img_size·map_size, 유한한 threshold.
def check_reconstruction_meta(meta: dict) -> None:
    """`meta`가 쓸 수 있는 재구성 아티팩트를 설명하지 않으면 ValueError를 낸다."""
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


# 결과로 어느 검사기 클래스로 열지 정한다. 4단계 PatchCore 아티팩트는 kind 키 없이 만들어졌으므로, 옛
# 아티팩트도 그대로 읽힌다(README: 4단계 PatchCore 아티팩트도 그대로 읽는다).
# 호출 위치: service.load_inspectors.
def artifact_kind(artifact_dir: Path) -> str:
    """범주 폴더 meta.json의 `kind`로 "patchcore" 또는 "reconstruction"을 돌려준다(없으면 patchcore)."""
    try:
        with open(Path(artifact_dir) / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError) as err:
        raise ValueError(f"{Path(artifact_dir) / 'meta.json'} is not readable JSON: {err}") from err
    return str(meta.get("kind", PATCHCORE)) if isinstance(meta, dict) else PATCHCORE


# 서비스의 모델은 Dinomaly ViT-S(6단계 dms-280-car)다. 왜: 6단계에서 이 모델이 CPU 150ms, 실제 오검출률 4.3%,
# AUROC 96.5로 세 기준을 모두 지켜(H16 지지) 2026-10-04에 서비스 모델로 바뀌었다(README '검사 서비스').
# 호출 위치: service.load_inspectors, bench.py, dinomaly_serving.py, runtime_compare.py(OpenVINO 세션을 끼워
# 넣어 같은 클래스로 비교).
class ReconstructionInspector:
    """스스로 점수를 내는 모델(Dinomaly)의 범주 하나 CPU 검사기: 메모리 뱅크가 없다.

    ONNX 모델은 `image` [1, 3, S, S](ImageNet 정규화)를 받아 `score` [1](이미지 점수: 흐린 맵의 상위 1%
    평균)과 `map` [1, M, M](이상 맵)을 돌려준다. 모델은 모든 범주가 함께 쓰고 임계값만 범주별이다. 점수나
    맵이 유한하지 않으면 판정하지 않고 `FloatingPointError`를 낸다.
    """

    kind = RECONSTRUCTION
    precision = "fp32"
    threads: int | None = None
    # 뱅크가 없으므로 0. 서비스가 두 검사기를 같은 방식으로 설명할 수 있게 같은 속성 이름을 둔다.
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

    # 모델 입출력 점검: 입력 image [1, 3, S, S], 출력 score [1]과 map [1, M, M]. 기호 차원은 비교하지 않는다.
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

    # 잘못된 것은 모두 ValueError로 알린다(Inspector.load와 같은 방식).
    @classmethod
    def load(
        cls, artifact_dir: Path, *, precision: str = "fp32", threads: int | None = None
    ) -> "ReconstructionInspector":
        """범주 폴더에서 재구성 검사기를 연다(모델 파일은 부모 폴더에 있을 수 있다)."""
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
        # 모델 파일이 meta에 적힌 지문과 다르면 세션을 만들기 전에 멈춘다(check_model_sha256 주석).
        try:
            check_model_sha256(model, meta, precision)
        except ValueError as err:
            raise ValueError(f"{artifact_dir} ({model.name}): {err}") from err
        options = ort.SessionOptions()
        if threads is not None:
            options.intra_op_num_threads = int(threads)
        try:
            session = ort.InferenceSession(
                str(model), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as err:  # onnxruntime은 읽지 못하는 파일에 자체 예외 형식을 낸다
            raise ValueError(f"{model} is not a loadable ONNX model: {err}") from err
        try:
            inspector = cls(session, meta)
        except ValueError as err:
            raise ValueError(f"{artifact_dir} ({model.name}): {err}") from err
        inspector.precision = precision
        inspector.threads = None if threads is None else int(threads)
        return inspector

    # 출력 모양이 meta와 다르거나 점수·맵이 유한하지 않으면 판정 전에 멈춘다. 맵도 검사하는 이유: 맵은 그대로
    # 서비스 히트맵이 되는데, NaN은 0~1로 자르고 uint8로 바꾸는 동안 0(검정)이 되어 '점수 0'처럼 조용히
    # 그려지고, 맵이 깨진 실행은 점수도 믿을 수 없다. 연구 경로 run_dinomaly.predict가 점수와 맵을 함께
    # 검사하는 것과 같은 규칙이다.
    def run(self, image: Image.Image | np.ndarray) -> tuple[float, np.ndarray]:
        """모델을 한 번 돌려 이미지 한 장의 (이미지 점수, float32 맵 [M, M])을 얻는다.

        점수나 맵이 유한하지 않으면 `FloatingPointError`를 낸다.
        """
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
        if not np.isfinite(amap).all():
            raise FloatingPointError("the anomaly map is not finite")
        return float(score[0]), amap[0]

    # Inspector.inspect와 같은 결과 형식. 맵은 모델이 이미 계산했으므로 heatmap=False여도 계산량은 같고
    # 응답에서만 뺀다.
    def inspect(self, image: Image.Image | np.ndarray, *, heatmap: bool = True) -> Inspection:
        score, amap = self.run(image)
        return Inspection(
            score=score,
            threshold=self.threshold,
            is_defect=score > self.threshold,
            heatmap=amap if heatmap else None,
        )

    def scores(self, images: np.ndarray) -> np.ndarray:
        """이미 입력 크기인 uint8 이미지 [N, S, S, 3]의 이미지 점수 float32 [N]만 낸다."""
        out = np.empty(len(images), dtype=np.float32)
        for i, image in enumerate(images):
            out[i] = self.run(image)[0]
        return out

    # Inspector와 같은 임계값 변경 규칙.
    def set_threshold(self, value: float) -> None:
        if not _is_finite_number(value):
            raise ValueError(f"the threshold must be a finite number, got {value!r}")
        self.threshold = float(value)
        self.meta["threshold"] = float(value)

    # Inspector.calibrate와 같은 규칙으로 임계값만 다시 잡는다. 모델은 그대로라 README의 'POST /calibrate는 그
    # 범주의 임계값만 다시 잡는다(모델은 그대로)'에 해당한다.
    def calibrate(self, images: Sequence[Image.Image | np.ndarray], alpha: float = 0.05) -> float:
        """현재 촬영 조건의 정상 이미지들로 임계값만 다시 잡는다(`alpha`의 conformal 순위)."""
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


# 단일 범주 아티팩트 폴더다. 쓰기 전에 check_meta와 _check_bank로 형식을 확인하고, 원본과 대상이 같은 파일이면
# 복사하지 않는다. 이 저장소에서는 테스트만 쓴다(실제 4단계 아티팩트는 export.build_artifacts가 만든다). 두
# 함수가 쓰는 bank.npy·meta.json이 바이트까지 같다는 것은 tests/test_run_export_eval.py가 확인한다.
def save_artifact(
    artifact_dir: Path, *, onnx_fp32: Path, bank: np.ndarray, meta: dict, onnx_int8: Path | None = None
) -> None:
    """아티팩트 폴더를 쓴다: ONNX 모델(들), 뱅크, meta.json."""
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
