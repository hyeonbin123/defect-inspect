"""Greedy k-center coreset selection for PatchCore memory banks."""

# [파일 역할] PatchCore 메모리 뱅크를 줄이는 탐욕적 k-center 코어셋 선택을 직접 구현한 모듈이다. 정상 이미지의
# 패치 특징 수십만 개 가운데, 지금까지 고른 점들에서 가장 먼 점을 하나씩 더해 가는 방식으로 일부(예: 10%)만
# 남긴다. 이렇게 고른 점들은 전체 특징 공간을 고르게 덮으므로, 뱅크가 작아져도 "가장 가까운 정상 패치까지의
# 거리"가 크게 달라지지 않는다.
# [설계 이유] (1) 선택 거리 계산만 128차원 무작위 투영으로 줄인다(1단계 P0 규칙: 선택할 때만 128차원 투영,
# seed 0, 시작점도 seed 0). 뱅크에는 원래 차원의 특징이 들어간다. (2) 고른 순서대로 인덱스를 돌려주므로 앞
# m개가 곧 크기 m의 코어셋이다. 4단계 격자는 10% 뱅크를 한 번 만들고 앞부분을 잘라 1%, 0.1% 뱅크로
# 쓴다(docs/experiments.md 4단계). (3) 같은 장치에서는 비트 단위로 재현되지만 CPU와 GPU 사이에서는 부동소수
# 반올림으로 동점 처리가 달라질 수 있어, 비교할 뱅크는 같은 장치에서 만든다(experiments.md 0단계).
# [연결] patchcore.build_bank가 kcenter_greedy를 불러 뱅크를 만든다. 0단계 실측으로 RTX 2080 Ti에서 491,520 →
# 49,152 패치 선택에 31초(선택 1번에 약 0.6ms)였다.

from __future__ import annotations

import math

import torch

# _PROJECT_CHUNK: 특징을 장치로 옮기며 투영할 때 한 번에 처리하는 행 수. 원본 특징([N, D], 예: WRN-50은 패치당
# 1,536차원)은 CPU에 두고 조각씩만 장치에 올려, 장치 메모리에는 [N, proj_dim] 투영 결과만 남게 한다.
# Rows projected per step: bounds what is on the device besides the [N, proj_dim] projection.
_PROJECT_CHUNK = 32768


# 선택에 쓸 fp32 [N, d] 배열을 만든다. proj가 None이면(특징 차원이 proj_dim 이하) 투영 없이 fp32로만 바꿔
# 옮긴다. 입력이 fp16이어도 거리 계산은 fp32로 한다.
def _project(features: torch.Tensor, proj: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    """Return the fp32 [N, d] array the selection runs on, moving `features` to the device in chunks."""
    n = features.shape[0]
    out_dim = features.shape[1] if proj is None else proj.shape[1]
    # 결과 배열을 장치에 한 번 잡아 두고 조각마다 채워 넣는다. 조각 단위라 중간 텐서가 _PROJECT_CHUNK 행을
    # 넘지 않는다.
    z = torch.empty((n, out_dim), dtype=torch.float32, device=device)
    for start in range(0, n, _PROJECT_CHUNK):
        chunk = features[start : start + _PROJECT_CHUNK].to(device=device, dtype=torch.float32)
        z[start : start + _PROJECT_CHUNK] = chunk if proj is None else chunk @ proj
    return z


# 탐욕적 k-center의 본 반복문. 각 점의 "이미 고른 중심들까지의 최소 거리"(min_dist)를 유지하고, 매번 그 값이
# 가장 큰 점을 다음 중심으로 고른다. 출력은 고른 순서대로의 int64 인덱스다.
# 성능: docstring대로 중심 인덱스를 파이썬 정수로 꺼내지 않고(.item() 없이) 1원소 텐서로만 다룬다. CUDA에서 매
# 반복 값을 읽어 오면 동기화 비용이 계산보다 크기 때문이다. dist·min_dist도 미리 잡아 두고 out=·제자리
# 연산으로 재사용해 반복마다 새 메모리를 잡지 않는다.
def _select(z: torch.Tensor, z_sq: torch.Tensor, first: torch.Tensor, n_select: int) -> torch.Tensor:
    """Greedy k-center loop over centred points `z` [N, d] with squared norms `z_sq` [N].

    `first` is the int64 [1] index of the first centre; everything is on one device, the result too.
    The current centre stays a 1-element device tensor and is only used through index_select /
    index_copy_: indexing with it (or `.item()`) would read a value back from the device in every
    iteration, and on CUDA that synchronisation costs more than the arithmetic of the iteration.
    """
    n = z.shape[0]
    selected = torch.empty(n_select, dtype=torch.int64, device=z.device)
    # 아직 중심이 없으므로 모든 점의 최소 거리를 무한대로 시작한다. taken(-1)은 이미 고른 점의 min_dist를
    # 덮어쓸 값이다.
    min_dist = torch.full((n,), float("inf"), dtype=torch.float32, device=z.device)
    dist = torch.empty(n, dtype=torch.float32, device=z.device)
    taken = torch.full((1,), -1.0, dtype=torch.float32, device=z.device)
    centre = first
    # 반복: 현재 중심을 기록 → 새 중심까지의 거리로 min_dist 갱신 → min_dist가 가장 큰 점을 다음 중심으로.
    # 마지막 중심을 기록한 뒤에는 거리 계산이 필요 없어 바로 끝낸다.
    for i in range(n_select):
        selected[i : i + 1] = centre
        if i + 1 == n_select:
            break
        # addmv로 z_sq - 2·(z·c)를 한 번에 계산하고 다음 줄에서 ||c||²를 더한다. 반올림 때문에 아주 작은
        # 음수가 나올 수 있어 0으로 자른다. 순위만 쓰므로 제곱근은 구하지 않는다.
        # Squared distance of every point to the last centre: ||z||^2 + ||c||^2 - 2 z.c
        torch.addmv(z_sq, z, z.index_select(0, centre)[0], alpha=-2.0, out=dist)
        dist.add_(z_sq.index_select(0, centre)).clamp_(min=0.0)
        torch.minimum(min_dist, dist, out=min_dist)
        # 고른 점의 min_dist를 -1로 두어 argmax에 다시 뽑히지 않게 한다. 중복 점이 많아 남은 점의 거리가 모두
        # 0이 되어도 같은 점을 두 번 고르지 않는다.
        # A chosen point is never chosen again, even among exact duplicates.
        min_dist.index_copy_(0, centre, taken)
        centre = torch.argmax(min_dist, dim=0, keepdim=True)
    return selected


# 코어셋 선택의 공개 함수. 입력: [N, D] 특징(CPU, fp16 또는 fp32), 고를 개수 n_select. 출력: 고른 순서대로의
# CPU int64 인덱스.
# 학습이 아니므로 no_grad로 자동 미분 기록을 끈다. 투영 행렬과 시작점은 torch.Generator(seed)로 만들어, 모듈
# 수준 난수를 쓰지 않는 결정적 선택이 된다. 호출 위치: patchcore.build_bank.
@torch.no_grad()
def kcenter_greedy(
    features: torch.Tensor,
    n_select: int,
    *,
    seed: int = 0,
    proj_dim: int = 128,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Greedy k-center selection over the rows of `features` [N, D] (CPU, fp16 or fp32).

    Returns int64 indices in selection order, so the first m entries equal the result for
    `n_select=m`. Only the [N, proj_dim] projection and a few [N] vectors live on `device`.
    Non-finite features raise ValueError. The selection is reproducible on one device; another
    device may break near-ties differently, so build banks that are compared on the same device.
    """
    # 입력 검사: 2차원이 아니거나 비었으면, 또는 고를 개수가 1보다 작으면 바로 오류를 낸다. n_select가 N보다
    # 크면 N개(전부)로 줄인다.
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"expected a non-empty [N, D] tensor, got {tuple(features.shape)}")
    if n_select < 1:
        raise ValueError(f"n_select must be at least 1, got {n_select}")
    device = torch.device(device)
    n, dim = features.shape
    n_select = min(int(n_select), n)

    # 무작위 투영: D차원 특징이 proj_dim(128)보다 크면 표준정규 행렬을 √proj_dim으로 나눈 것으로 곱한다. 이런
    # 무작위 투영은 점들 사이의 거리를 대략 보존하므로(Johnson-Lindenstrauss), 선택 계산을 1,536차원 대신
    # 128차원에서 할 수 있다. 같은 생성기에서 이어서 시작점 하나를 뽑는다.
    generator = torch.Generator().manual_seed(seed)
    proj = None
    if dim > proj_dim:
        proj = torch.randn(dim, proj_dim, generator=generator) / math.sqrt(proj_dim)
        proj = proj.to(device)
    first = torch.randint(n, (1,), generator=generator)

    z = _project(features, proj, device)
    # 모든 점에서 평균을 빼도 점 사이 거리는 같다. 대신 노름이 작아져, ||z||² + ||c||² - 2z·c처럼 큰 수끼리
    # 빼는 전개식의 fp32 반올림 오차가 줄어든다.
    # Centring leaves distances unchanged and keeps the expanded form in the loop accurate in fp32.
    z -= z.mean(dim=0, keepdim=True)
    z_sq = torch.linalg.vector_norm(z, dim=1).square_()
    # One NaN or inf anywhere reaches every row through the mean. Left alone, the loop's argmax
    # would return the same indices again and again instead of failing.
    if not bool(torch.isfinite(z_sq).all()):
        raise ValueError("features contain non-finite values")

    return _select(z, z_sq, first.to(device), n_select).cpu()
