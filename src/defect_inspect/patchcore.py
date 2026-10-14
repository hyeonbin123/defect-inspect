"""PatchCore: 패치 특징, 코어셋 메모리 뱅크, 최근접 이웃 거리로 매기는 이상 점수."""

# [파일 역할] 직접 구현한 PatchCore의 핵심 계산(GPU, torch)이다. 정상 이미지의 패치 특징을
# 모으고(collect_features), 탐욕적 k-center 코어셋으로 메모리 뱅크를 줄이고(build_bank), 테스트 이미지의 각
# 패치가 뱅크의 가장 가까운 정상 패치에서 얼마나 먼지로 이상 점수와 점수 맵을 만든다(score_images).
# [왜 필요한가] PatchCore는 학습이 없는 비지도 이상 탐지 기준선(1단계 P0, 2단계 D-S)이고, 4단계 격자·3단계
# 교란·M2AD·7단계 조명 등록이 모두 이 함수들을 다시 쓴다. 라이브러리 대신 직접 구현한 것은 docs/experiments.md
# '실행 예산'의 '줄이지 않는 것(직접 구현한 PatchCore·지표·보정 코드와 그 테스트)'에 적혀 있고, anomalib
# PatchCore와 dev에서 세 범주 모두 2%p 안으로 맞는지 대조했다(1단계 '구현 점검').
# [주요 구성] 특징 추출(fp16) → 코어셋 뱅크 → 블록 단위 최근접 탐색 → 이미지 점수 재가중 → 점수 맵(bilinear
# 확대, 가우시안 σ, 256 격자).
# [연결] run_patchcore, run_grid, run_m2ad, run_m2ad_enrol, run_perturb가 함수 안에서 늦게 import한다(torch가
# 없는 CI에서 기본 모듈이 깨지지 않게). CPU 서빙용 numpy/onnxruntime 경로는 inspector.py가 같은 계산을 다시
# 구현한다.

from __future__ import annotations

import contextlib
from dataclasses import dataclass

# torch: GPU 특징 추출과 거리 계산. 이 모듈은 모듈 수준에서 torch를 가져오므로 train 의존성 그룹이 있어야
# import된다. 다른 기본 패키지 모듈은 torch를 모듈 수준에서 가져오지 않으므로(CI는 torch 없이 돈다,
# .github/workflows/ci.yml) 이 파일을 쓰는 쪽이 함수 안에서 import한다.
import numpy as np
import torch
import torch.nn.functional as F

# to_tensor: uint8 [N, S, S, 3]을 정규화된 [N, 3, S, S] 텐서로 바꾸는 백본 공용 도우미. kcenter_greedy: 코어셋
# 선택(coreset.py).
from defect_inspect.backbones import to_tensor
from defect_inspect.coreset import kcenter_greedy

# 뱅크를 탐색할 때 한 번에 만드는 [쿼리 행, 뱅크 행] 거리 블록의 최대 원소 수(2**27개, fp32로 512MB).
# RTX 2080 Ti(11GB)에서 뱅크가 커져도 한 블록이 이 크기를 넘지 않게 쿼리를 잘라 처리한다.
_MAX_DIST_ELEMS = 1 << 27


# score_images의 결과 묶음. image_scores는 이미지당 점수 하나(float32), maps는 256×256 점수 맵(float16으로
# 저장 용량을 반으로 줄인다). run_dinomaly와 supervised도 같은 형식을 써서 뒤의 분석 코드가 방법과 무관하게
# 같은 구조를 읽는다.
@dataclass
class ScoreResult:
    image_scores: np.ndarray  # float32 [N]
    maps: np.ndarray  # float16 [N, map_size, map_size]


# 입력이 [N, S, S, 3] 모양이고 배치 크기가 1 이상인지 확인한다.
def _check_images(images: np.ndarray, batch_size: int) -> None:
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"expected images [N, S, S, 3], got {images.shape}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")


# CUDA에서는 fp16 autocast를 켜고 CPU에서는 아무것도 하지 않는 문맥을 돌려준다. RTX 2080 Ti에는 bf16이 없어
# 혼합 정밀도는 fp16만 쓴다. P0 구성도 fp16 추론으로 정해져 있다(experiments.md 1단계).
def _autocast(device: torch.device) -> contextlib.AbstractContextManager:
    """CUDA에서는 fp16 autocast를 켜고, 그 밖의 장치에서는 그냥 fp32로 계산한다."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


# 한 배치의 uint8 이미지를 백본에 넣어 패치 특징 [B, H, W, D]를 얻는다. 결과를 fp16으로 맞춰, 뱅크를 만들 때와
# 채점할 때 같은 반올림 값을 보게 한다.
# fp16 범위를 넘어 inf/NaN이 생기면 조용히 이상한 점수가 나오지 않게 FloatingPointError로 멈춘다.
def _patch_features(extractor: torch.nn.Module, images: np.ndarray, device: torch.device) -> torch.Tensor:
    """uint8 이미지 한 배치 -> 장치 위의 fp16 패치 특징 [B, H, W, D]."""
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    with _autocast(device):
        feats = extractor(to_tensor(images, device, dtype))
    # 뒤의 모든 단계(뱅크 행이든 쿼리든)가 똑같이 fp16으로 반올림된 값을 보게 한다.
    feats = feats.to(torch.float16)
    if not bool(torch.isfinite(feats).all()):
        raise FloatingPointError("patch features overflowed fp16")
    return feats


# 정상 이미지들의 패치 특징을 모두 모아 CPU fp16 텐서 [N*H*W, D]로 돌려준다(뱅크 재료). 행 순서는 이미지 순,
# 그 안에서 패치 격자의 행 우선 순이다.
# 출력 텐서를 첫 배치에서 한 번만 할당하고 배치마다 채워 넣어, 리스트에 모았다가 이어 붙일 때 생기는 두 배
# 메모리 정점을 피한다(정점이 특징 한 벌 크기에 머문다).
# 호출 위치: run_patchcore, run_grid, run_m2ad, run_m2ad_enrol, supervised(지도 학습 헤드의 고정 특징).
@torch.no_grad()
def collect_features(
    extractor: torch.nn.Module,
    images: np.ndarray,
    *,
    batch_size: int = 32,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, tuple[int, int]]:
    """uint8 이미지 [N, S, S, 3] -> (CPU fp16 패치 특징 [N * H * W, D], 패치 격자 (H, W)).

    행은 이미지 순서대로 놓이고, 한 이미지 안에서는 패치 격자의 행 우선 순서를 따른다.
    """
    _check_images(images, batch_size)
    device = torch.device(device)
    n = len(images)
    # 이미지가 없으면 특징 차원(extractor.dim)만 맞춘 빈 텐서를 돌려준다.
    if n == 0:
        return torch.empty((0, int(extractor.dim)), dtype=torch.float16), (0, 0)
    extractor = extractor.to(device).eval()
    out: torch.Tensor | None = None
    grid = (0, 0)
    pos = 0
    for start in range(0, n, batch_size):
        feats = _patch_features(extractor, images[start : start + batch_size], device)
        b, h, w, dim = feats.shape
        if out is None:
            # 한 번만 할당하므로 메모리 정점이 특징 한 벌 크기에 머문다.
            grid = (h, w)
            out = torch.empty((n * h * w, dim), dtype=torch.float16)
        out[pos : pos + b * h * w] = feats.reshape(b * h * w, dim).cpu()
        pos += b * h * w
    assert out is not None and pos == out.shape[0]
    return out, grid


# 패치 특징에서 탐욕적 k-center 코어셋을 골라 메모리 뱅크를 만든다. ratio는 남길 비율(P0 기본값 10%, 4단계
# 격자는 0.1/0.01/0.001), seed는 시작점과 투영의 시드다.
# 선택 순서대로 저장하므로 bank[:m]이 곧 크기 m의 코어셋이다. 그래서 4단계 격자에서는 10% 뱅크 하나로 1%·0.1%
# 뱅크를 앞부분만 잘라 얻는다(run_grid의 같은 반올림 함수 참고).
# 주의: 코어셋 선택은 CPU와 GPU에서 반올림 수준으로 달라질 수 있어 모든 뱅크를 GPU에서 만든다(experiments.md
# 0단계).
def build_bank(
    features: torch.Tensor, ratio: float = 0.1, *, seed: int = 0, device: str | torch.device = "cuda"
) -> torch.Tensor:
    """`features` [N, D]의 탐욕적 k-center 코어셋을 선택 순서대로 CPU fp16 텐서로 돌려준다.

    `max(1, round(ratio * N))`행을 남긴다(`ratio >= 1`이면 전부). `bank[:m]`이 곧 크기 m의 코어셋이다.
    """
    n = features.shape[0]
    # 남길 행 수: 비율 × 전체를 반올림하되 최소 1행. ratio가 1 이상이면 전부 남긴다.
    n_select = n if ratio >= 1 else max(1, round(ratio * n))
    index = kcenter_greedy(features, n_select, seed=seed, device=device)
    return features[index].to(torch.float16)


# 점수 맵을 부드럽게 할 1차원 가우시안 커널(합 1로 정규화). 반지름 int(4σ + 0.5)까지 잘라 쓰는데,
# scipy.ndimage의 기본 truncate=4.0과 같은 크기 규칙이다. bench.py는 이 함수를 그대로 쓰고, CPU 서빙
# 경로(inspector.py)는 같은 크기 규칙의 numpy 커널을 따로 두어 두 경로의 점수 맵이 맞게 한다.
def gaussian_kernel1d(sigma: float) -> torch.Tensor:
    """합이 1인 fp32 가우시안 커널. 길이는 `2 * int(4 * sigma + 0.5) + 1`(sigma <= 0이면 1 하나)."""
    if sigma <= 0:
        return torch.ones(1, dtype=torch.float32)
    radius = int(4 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return (kernel / kernel.sum()).to(torch.float32)


# 가로·세로로 나눠 두 번 1차원 합성곱하는 분리형 가우시안 흐림(2차원 커널보다 계산이 적다). 가장자리는 reflect
# 패딩으로 처리한다.
# reflect 패딩은 패딩 폭이 맵 크기보다 작아야 하므로, 커널 반지름이 맵보다 크면 ValueError로 막는다.
def _blur(maps: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """[B, 1, H, W] 맵에 reflect 패딩으로 분리형 가우시안 흐림을 준다."""
    pad = kernel.numel() // 2
    if pad == 0:
        return maps
    if pad >= min(maps.shape[-2:]):
        raise ValueError(f"blur radius {pad} needs a map larger than {tuple(maps.shape[-2:])}")
    maps = F.pad(maps, (pad, pad, pad, pad), mode="reflect")
    maps = F.conv2d(maps, kernel.view(1, 1, -1, 1))
    return F.conv2d(maps, kernel.view(1, 1, 1, -1))


# 쿼리 패치마다 뱅크에서 가장 가까운 행과 그 유클리드 거리를 구한다(PatchCore의 패치 점수).
# 방식: 전개식 ||q-b||² = ||q||² - 2q·b + ||b||²로 거리 행렬을 행렬곱(addmm) 한 번에 계산해 최근접 인덱스를
# 찾고, 찾은 행과의 거리만 q - b로 다시 직접 계산한다. 전개식은 0 근처에서 큰 수끼리 빼며 정밀도를
# 잃으므로(아래 docstring의 '0 근처 상쇄'), 순위는 전개식으로 빠르게, 실제 거리는 정확한 식으로 낸다.
# 메모리: 블록 원소 수가 _MAX_DIST_ELEMS를 넘지 않도록 쿼리를 step개씩 자른다.
def _nearest(
    query: torch.Tensor, bank: torch.Tensor, bank_sq: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """쿼리 행마다 가장 가까운 뱅크 행까지의 유클리드 거리와 그 행 번호(fp32, 한 장치 위).

    탐색은 원소 `_MAX_DIST_ELEMS`개 이하의 블록으로 나눠 전개식으로 한다. 찾은 행까지의 거리는 직접 다시
    계산하므로, 거리가 0에 가까울 때 큰 수끼리 빼며 생기는 상쇄 오차를 겪지 않는다.
    """
    n_query = query.shape[0]
    step = max(1, _MAX_DIST_ELEMS // bank.shape[0])
    dist = torch.empty(n_query, dtype=torch.float32, device=query.device)
    index = torch.empty(n_query, dtype=torch.int64, device=query.device)
    for start in range(0, n_query, step):
        q = query[start : start + step]
        # addmm(bank_sq, q, bank.T, alpha=-2)는 ||b||² - 2q·b를 계산한다. 이 값은 제곱 거리와 같은 순서로 뱅크
        # 행을 줄 세운다. ||q||²는 쿼리 행마다 상수라 argmin 결과에 영향이 없어 빼고 계산한다.
        block = torch.addmm(bank_sq.unsqueeze(0), q, bank.T, alpha=-2.0)
        nearest = block.argmin(dim=1)
        # 큰 거리 블록을 바로 놓아 다음 블록을 만들 때 GPU 메모리 정점이 두 배가 되지 않게 한다.
        del block
        index[start : start + step] = nearest
        dist[start : start + step] = torch.linalg.vector_norm(q - bank[nearest], dim=1)
    return dist, index


# PatchCore 논문의 이미지 점수: 가장 이상한 패치의 점수 s*에 가중치 w를 곱한다.
# w = 1 - softmax(거리)[m*]: 그 패치(feat_star)와 최근접 뱅크 행 m*, 그리고 m*의 뱅크 안 이웃들 사이의 거리에
# softmax를 취해 m* 몫을 1에서 뺀다. m*가 이웃들보다 유난히 가까우면(뱅크의 드문 영역) 점수를 덜 깎고,
# 이웃들과 거리가 비슷하면(촘촘한 정상 영역) 더 깎는다.
# reweight_k는 이웃 수(P0는 9, experiments.md 1단계 '재가중(그 패치의 최근접 뱅크 패치의 이웃 9개 기준)'). 1
# 이하이면 재가중 없이 s*만 쓴다.
def _image_scores(
    patch_scores: torch.Tensor,
    nearest: torch.Tensor,
    query: torch.Tensor,
    bank: torch.Tensor,
    bank_sq: torch.Tensor,
    reweight_k: int,
) -> torch.Tensor:
    """PatchCore 이미지 점수: 가장 큰 패치 점수를 뱅크 안 이웃 분포로 재가중한 값.

    `patch_scores`와 `nearest`는 [B, P], `query`는 [B, P, D] 모양이다.
    """
    s_star, p_star = patch_scores.max(dim=1)
    if reweight_k <= 1:
        return s_star
    # 뱅크 행 수가 k보다 적으면(아주 작은 뱅크) 있는 행만큼만 이웃으로 쓴다.
    k = min(reweight_k, bank.shape[0])
    rows = torch.arange(patch_scores.shape[0], device=patch_scores.device)
    feat_star = query[rows, p_star]  # [B, D]: 이미지마다 가장 이상한 패치
    m_star = nearest[rows, p_star]  # [B]: 그 패치의 최근접 뱅크 행
    # m*와 뱅크 전체 행 사이의 (순위용) 전개식 거리 [B, R]. 같은 값의 중복 행이 있어도 m* 자신이 첫 번째
    # 이웃이 되도록 그 칸을 -inf로 둔다. 그래서 softmax의 0번 칸이 항상 m*다.
    block = torch.addmm(bank_sq.unsqueeze(0), bank[m_star], bank.T, alpha=-2.0)  # [B, R]
    block[rows, m_star] = float("-inf")  # 뱅크에 m*와 같은 행이 있어도 m* 자신이 맨 앞에 온다
    support = block.topk(k, dim=1, largest=False).indices  # [B, k], 가까운 순
    # 가중치에 쓰는 거리는 전개식이 아니라 가장 이상한 패치 feat_star와 이웃 행들 사이의 직접 거리다.
    dist = torch.linalg.vector_norm(feat_star.unsqueeze(1) - bank[support], dim=2)  # [B, k]
    weight = 1.0 - torch.softmax(dist, dim=1)[:, 0]
    return weight * s_star


# 이미지들을 메모리 뱅크와 비교해 이미지 점수와 점수 맵을 낸다. PatchCore 채점의 공용 진입점이다.
# 입력: uint8 이미지 [N, S, S, 3], 뱅크 [R, D]. 출력: ScoreResult(이미지 점수 float32 [N], 맵 float16 [N, 256,
# 256]).
# 기본값(reweight_k=9, sigma=4.0, map_size=256)은 P0 구성(experiments.md 1단계: 가우시안 σ=4, 픽셀 지표는 모든
# 구성에서 256×256 격자)과 같다.
# 호출 위치: run_patchcore, run_grid, run_m2ad, run_m2ad_enrol, run_perturb. 테스트(tests/test_export.py 등)는
# CPU 서빙 경로의 점수를 이 함수 결과와 대조한다.
@torch.no_grad()
def score_images(
    extractor: torch.nn.Module,
    bank: torch.Tensor,
    images: np.ndarray,
    *,
    batch_size: int = 32,
    device: str | torch.device = "cuda",
    reweight_k: int = 9,
    sigma: float = 4.0,
    map_size: int = 256,
) -> ScoreResult:
    """uint8 이미지 [N, S, S, 3]를 메모리 뱅크 [R, D]와 비교해 채점한다.

    패치 점수 = 가장 가까운 뱅크 행까지의 유클리드 거리. 이미지 점수 = 재가중한 가장 큰 패치 점수.
    점수 맵 = 패치 점수를 S로 bilinear 확대하고 가우시안(`sigma`)으로 흐린 뒤 `map_size`로 크기를 맞춘 것.
    """
    _check_images(images, batch_size)
    if bank.ndim != 2 or bank.shape[0] == 0:
        raise ValueError(f"expected a non-empty bank [R, D], got {tuple(bank.shape)}")
    device = torch.device(device)
    n = len(images)
    image_scores = np.empty(n, dtype=np.float32)
    maps = np.empty((n, map_size, map_size), dtype=np.float16)
    if n == 0:
        return ScoreResult(image_scores, maps)

    extractor = extractor.to(device).eval()
    # 뱅크를 장치에 fp32 사본 하나로 올리고 평균 벡터를 빼서 중심화한다. 모든 행과 쿼리에서 같은 벡터를
    # 빼므로 거리는 그대로이고, 탐색에 쓰는 전개식이 정확하게 유지된다(ReLU 뒤 특징은 모든 행이 큰 공통
    # 오프셋을 가져 그대로 두면 큰 수 뺄셈 오차가 커진다). 쿼리도 아래에서 같은 centre를 뺀다.
    bank_dev = bank.to(device=device, dtype=torch.float32, copy=True)
    centre = bank_dev.mean(dim=0, keepdim=True)
    bank_dev -= centre
    # 뱅크 각 행의 제곱 노름은 배치마다 바뀌지 않으므로 한 번만 계산해 _nearest와 _image_scores에 넘긴다.
    bank_sq = torch.linalg.vector_norm(bank_dev, dim=1).square_()
    kernel = gaussian_kernel1d(sigma).to(device)
    size = (images.shape[1], images.shape[2])

    for start in range(0, n, batch_size):
        feats = _patch_features(extractor, images[start : start + batch_size], device)
        b, h, w, dim = feats.shape
        # 다른 백본·층으로 만든 뱅크를 실수로 넣으면 차원이 달라진다. 조용히 틀린 점수를 내지 않게 막는다.
        if dim != bank_dev.shape[1]:
            raise ValueError(f"feature dim {dim} does not match the bank ({bank_dev.shape[1]})")
        # 배치의 모든 패치를 [B*H*W, D] 쿼리로 펼치고 fp32로 올려 뱅크와 같은 centre를 뺀다. feats는 바로 지워
        # GPU 메모리를 아낀다.
        query = feats.reshape(b * h * w, dim).to(torch.float32).sub_(centre)
        del feats
        dist, nearest = _nearest(query, bank_dev, bank_sq)
        patch_scores = dist.view(b, h * w)
        scores = _image_scores(
            patch_scores, nearest.view(b, h * w), query.view(b, h * w, dim), bank_dev, bank_sq, reweight_k
        )
        image_scores[start : start + b] = scores.cpu().numpy()

        # 점수 맵: 패치 격자(예: 32×32)의 패치 점수를 입력 크기 S로 bilinear 확대한 뒤 가우시안 흐림을 주고,
        # S가 256이 아니면 256 격자로 다시 맞춘다. 흐림을 입력 크기에서 주므로 σ의 단위는 입력 픽셀이다.
        batch_maps = F.interpolate(
            patch_scores.view(b, 1, h, w), size=size, mode="bilinear", align_corners=False
        )
        batch_maps = _blur(batch_maps, kernel)
        if size != (map_size, map_size):
            batch_maps = F.interpolate(
                batch_maps, size=(map_size, map_size), mode="bilinear", align_corners=False
            )
        maps[start : start + b] = batch_maps[:, 0].to(torch.float16).cpu().numpy()
    return ScoreResult(image_scores, maps)
