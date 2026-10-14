# [파일 역할] PatchCore 메모리 뱅크를 줄이는 탐욕적 k-center 코어셋
# 선택(`defect_inspect.coreset.kcenter_greedy`)을 검증하는 테스트 묶음이다.
# [왜 필요한가] 코어셋은 직접 구현했고(docs/plan.md: PatchCore 직접 구현), 4단계 격자는 10% 뱅크의 앞부분을
# 잘라 1%, 0.1% 뱅크로 쓴다. 그래서 (1) 느리지만 단순한 float64 기준 구현과 같은 순서로 고르는지, (2) 앞 m개가
# 곧 크기 m의 결과인지(포함 관계), (3) seed로 결정되는지, (4) 비정상 입력에서 조용히 틀린 답을 내지 않는지를
# 확인한다.
# [흐름] 기준 구현 대조(투영 없음/있음, 경계 차원) → 포함 관계·전부 선택·결정성 → 입력 보존·fp16 →
# 중복점·군집·큰 오프셋(수치 안정성) → 조각 투영 → 비유한값 거부 → CUDA 동기화(선택형 GPU 테스트) → 인자 검사.

import math
import os

import numpy as np
import pytest

# coreset은 torch를 쓰므로 torch가 없는 CI 환경(.github/workflows/ci.yml)에서는 이 파일 전체를 건너뛴다.
# 그래서 아래 프로젝트 import가 importorskip 뒤에 온다.
torch = pytest.importorskip("torch")

from defect_inspect import coreset  # noqa: E402
from defect_inspect.coreset import kcenter_greedy  # noqa: E402


# float64로 실제 거리 차를 직접 제곱해 더하므로 전개식(||a||² + ||b||² − 2ab)의 fp32 반올림 문제가 없다. 이미
# 고른 점의 최소 거리를 −1로 두어 다시 뽑지 않는 규칙은 실제 구현과 같다.
def reference_kcenter(points: np.ndarray, first: int, n_select: int) -> list[int]:
    """비교 기준이 되는 가장 단순한 float64 탐욕적 k-center."""
    chosen = [first]
    min_dist = np.full(len(points), np.inf)
    while len(chosen) < n_select:
        dist = ((points - points[chosen[-1]]) ** 2).sum(axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[chosen] = -1.0
        chosen.append(int(np.argmax(min_dist)))
    return chosen


# seed로 고정한 표준정규 특징 [n, dim]. 모듈 수준 난수를 쓰지 않아 테스트가 결정적이다.
def random_features(n: int, dim: int, seed: int) -> torch.Tensor:
    return torch.randn(n, dim, generator=torch.Generator().manual_seed(seed))


# 확인: 특징 차원(8)이 proj_dim(128) 이하라 투영 없이 고를 때, 결과가 int64 [n_select]이고 기준 구현과 같은
# 순서다. 시작점은 같은 seed의 Generator에서 처음 뽑은 값이다.
def test_matches_reference_without_projection():
    feats = random_features(200, 8, seed=1)
    for seed in (0, 3):
        first = int(torch.randint(200, (1,), generator=torch.Generator().manual_seed(seed)))
        got = kcenter_greedy(feats, 40, seed=seed)
        assert got.dtype == torch.int64 and got.shape == (40,)
        assert got.tolist() == reference_kcenter(feats.double().numpy(), first, 40)


# 확인: 차원이 proj_dim보다 크면 같은 Generator에서 투영 행렬을 먼저 뽑고 그다음 시작점을 뽑으며, 투영한
# 공간에서 기준 구현과 같은 순서로 고른다.
# 왜: 난수를 뽑는 순서가 바뀌면 같은 seed라도 다른 뱅크가 나와 이미 기록한 실행과 재현이 어긋난다. +5.0으로
# 원점에서 떨어뜨린 점은 구현 안의 평균 빼기가 결과를 바꾸지 않음도 함께 본다.
def test_matches_reference_with_projection():
    feats = random_features(300, 32, seed=2) + 5.0
    seed, proj_dim = 7, 4
    generator = torch.Generator().manual_seed(seed)
    proj = torch.randn(32, proj_dim, generator=generator) / math.sqrt(proj_dim)
    first = int(torch.randint(300, (1,), generator=generator))
    projected = (feats.double() @ proj.double()).numpy()
    got = kcenter_greedy(feats, 30, seed=seed, proj_dim=proj_dim)
    assert got.tolist() == reference_kcenter(projected, first, 30)
    # 투영은 점 사이의 기하를 바꾼다: 투영하지 않은 원래 공간에서 고르면 순서가 달라진다.
    assert got.tolist() != kcenter_greedy(feats, 30, seed=seed, proj_dim=32).tolist()


# 확인: 투영은 차원이 proj_dim보다 '클 때만' 일어난다(같으면 투영 없음). 경계 한 칸 차이로 난수 소비 순서와
# 시작점이 달라지므로 두 경우를 모두 기준 구현과 맞춘다.
def test_projection_starts_strictly_above_proj_dim():
    # D == proj_dim: 투영하지 않고, 첫 중심은 Generator에서 처음 뽑은 값이다.
    seed, proj_dim = 4, 8
    feats = random_features(200, proj_dim, seed=10)
    first = int(torch.randint(200, (1,), generator=torch.Generator().manual_seed(seed)))
    got = kcenter_greedy(feats, 40, seed=seed, proj_dim=proj_dim)
    assert got.tolist() == reference_kcenter(feats.double().numpy(), first, 40)
    # D == proj_dim + 1: 투영하며, 첫 중심은 투영 행렬을 뽑은 다음에 뽑는다.
    wide = random_features(200, proj_dim + 1, seed=10)
    generator = torch.Generator().manual_seed(seed)
    proj = torch.randn(proj_dim + 1, proj_dim, generator=generator) / math.sqrt(proj_dim)
    first = int(torch.randint(200, (1,), generator=generator))
    got = kcenter_greedy(wide, 40, seed=seed, proj_dim=proj_dim)
    assert got.tolist() == reference_kcenter((wide.double() @ proj.double()).numpy(), first, 40)


# 확인: n_select = m의 결과가 n_select = 60 결과의 앞 m개와 같다(포함 관계).
# 왜: 4단계 격자는 10% 뱅크를 한 번 고르고 앞부분을 잘라 1%, 0.1% 뱅크로 쓴다(docs/experiments.md 4단계). 이
# 성질이 깨지면 잘라 쓴 뱅크가 그 비율로 직접 고른 뱅크와 달라진다.
def test_selection_is_nested():
    feats = random_features(150, 12, seed=3)
    full = kcenter_greedy(feats, 60, seed=0, proj_dim=6)
    for m in (1, 2, 17, 59):
        assert kcenter_greedy(feats, m, seed=0, proj_dim=6).tolist() == full[:m].tolist()


# 확인: 전부 고르면 0..N−1의 순열이 되고, N보다 많이 요청하면 N개로 줄며, 앞부분은 적게 고른 결과와 같다.
def test_selecting_everything_returns_a_permutation_with_the_same_start():
    feats = random_features(50, 5, seed=4)
    some = kcenter_greedy(feats, 10, seed=2)
    everything = kcenter_greedy(feats, 50, seed=2)
    more = kcenter_greedy(feats, 500, seed=2)
    assert sorted(everything.tolist()) == list(range(50))
    assert more.tolist() == everything.tolist()
    assert everything[:10].tolist() == some.tolist()


# 확인: 같은 seed면 같은 결과이고, seed를 바꾸면 시작점이 실제로 달라진다(seed가 무시되지 않는다). 분할·표본
# 추출은 결정적이어야 한다는 프로젝트 규칙을 지키는지 본다.
def test_deterministic_in_seed():
    feats = random_features(120, 20, seed=5)
    a = kcenter_greedy(feats, 25, seed=11, proj_dim=8)
    b = kcenter_greedy(feats, 25, seed=11, proj_dim=8)
    assert a.tolist() == b.tolist()
    firsts = {int(kcenter_greedy(feats, 1, seed=s, proj_dim=8)) for s in range(8)}
    assert len(firsts) > 1


# 확인: 호출자의 특징 텐서가 바뀌지 않고, fp16 입력은 같은 값의 fp32 입력과 똑같이 고른다.
# 왜: 구현은 내부에서 평균을 빼는데(z -= mean), 투영 없이 fp32 입력이 들어오면 같은 저장 공간을 제자리에서
# 고칠 위험이 있다. patchcore.collect_features가 패치 특징을 CPU fp16으로 모아 넘기므로, fp16은 저장 형식일 뿐
# 선택 결과를 바꾸지 않아야 한다.
def test_input_is_not_modified_and_fp16_is_accepted():
    feats = random_features(80, 16, seed=6)
    half = feats.to(torch.float16)
    before = half.clone()
    got = kcenter_greedy(half, 20, seed=0, proj_dim=4)
    assert torch.equal(half, before)
    # fp16 입력은 저장 형식일 뿐이라, 같은 값을 fp32로 담은 입력과 똑같이 고른다.
    assert got.tolist() == kcenter_greedy(half.float(), 20, seed=0, proj_dim=4).tolist()
    # 투영 없음 + fp32 입력: 호출자의 텐서가 제자리에서 중심화(평균 빼기)되면 안 된다.
    full = feats.clone()
    kcenter_greedy(full, 20, seed=0, proj_dim=16)
    assert torch.equal(full, feats)


# 확인: 모든 점이 같아 거리가 전부 0이어도 같은 점을 두 번 고르지 않는다. 두 값만 있으면 두 번째 중심은 다른
# 무리에서 나온다.
# 왜: 고른 점의 최소 거리를 −1로 덮는 처리가 없으면 argmax가 같은 인덱스를 되풀이해 뱅크에 중복이 생긴다(정상
# 이미지의 배경 패치는 거의 같은 특징이 많다).
def test_exact_duplicates_are_never_selected_twice():
    feats = torch.ones(12, 3)
    got = kcenter_greedy(feats, 12, seed=0)
    assert sorted(got.tolist()) == list(range(12))
    # 서로 다른 값 두 개: 두 번째로 고른 점은 첫 점과 다른 무리에서 나와야 한다.
    feats = torch.cat([torch.zeros(6, 3), torch.ones(6, 3)])
    got = kcenter_greedy(feats, 12, seed=0)
    assert sorted(got.tolist()) == list(range(12))
    assert (got[0] < 6) != (got[1] < 6)


# 확인: 멀리 떨어진 군집 5개에서 5개를 고르면 군집마다 하나씩 고른다. k-center가 특징 공간을 고르게 덮는다는
# 성질(코어셋을 쓰는 이유)을 직접 본다.
def test_covers_well_separated_clusters():
    centres = torch.tensor([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0], [50.0, 300.0]])
    labels = torch.arange(200) % 5
    feats = centres[labels] + 0.1 * random_features(200, 2, seed=7)
    got = kcenter_greedy(feats, 5, seed=0)
    assert sorted(labels[got].tolist()) == [0, 1, 2, 3, 4]


# 확인: 모든 점을 +1000만큼 옮겨도 원래 점들과 같은 순서로 고른다.
# 왜: 구현은 거리를 ||z||² + ||c||² − 2z·c 전개식으로 계산하는데, 원점에서 먼 점은 큰 수끼리 빼서 fp32
# 정밀도를 잃는다. 구현이 먼저 평균을 빼(중심화) 이 문제를 피하는지 확인한다.
def test_large_offset_does_not_break_the_expanded_distance():
    # 원점에서 멀면 단순 전개식 ||a||^2 + ||b||^2 - 2ab는 fp32에서 정밀도를 모두 잃는다.
    feats = random_features(100, 6, seed=8)
    shifted = feats + 1000.0
    first = int(torch.randint(100, (1,), generator=torch.Generator().manual_seed(0)))
    expected = reference_kcenter(feats.double().numpy(), first, 30)
    assert kcenter_greedy(shifted, 30, seed=0).tolist() == expected


# 확인: 투영을 조각 단위로 하는 경계(_PROJECT_CHUNK)를 7행으로 줄여도 결과가 같다. monkeypatch로 모듈 상수를
# 테스트 동안만 바꿔, 큰 데이터 없이도 조각 경계 처리(마지막 덜 찬 조각 포함)를 거치게 한다.
def test_projection_is_chunked(monkeypatch):
    feats = random_features(90, 24, seed=9)
    whole = kcenter_greedy(feats, 20, seed=0, proj_dim=8)
    monkeypatch.setattr(coreset, "_PROJECT_CHUNK", 7)
    assert kcenter_greedy(feats, 20, seed=0, proj_dim=8).tolist() == whole.tolist()


# 확인: NaN·±inf가 한 칸이라도 있으면 n_select와 투영 여부, fp16 저장과 상관없이 'non-finite' ValueError를
# 낸다.
# 왜: 평균 빼기를 거치며 비유한값이 모든 행으로 퍼지고, 그대로 두면 argmax가 같은 인덱스를 되풀이해 오류 없이
# 망가진 뱅크가 만들어진다.
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_features_are_rejected(bad):
    # 이 검사가 없으면 NaN 행 하나 때문에 선택 반복문이 같은 인덱스를 되풀이해 돌려준다.
    feats = random_features(100, 8, seed=12)
    feats[7, 3] = bad
    for n_select in (1, 10, 100):
        with pytest.raises(ValueError, match="non-finite"):
            kcenter_greedy(feats, n_select, seed=0)
    # 무작위 투영을 거치고 fp16으로 저장된 경우도 마찬가지다.
    wide = random_features(100, 16, seed=12).to(torch.float16)
    wide[99, 0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        kcenter_greedy(wide, 10, seed=0, proj_dim=4)


# 확인: fp16 최댓값(65504)으로 채운 1,536차원 특징도 정상 처리한다. 비유한값 검사가 '큰 유한값'까지 잘못
# 거부하지 않는지, fp32 제곱 노름이 넘치지 않는지 본다(WRN-50 패치 특징이 1,536차원이다).
def test_large_finite_features_are_accepted():
    # 모든 칸이 fp16 최댓값이어도 fp32 제곱 노름이 넘치기에는 아직 한참 모자란다.
    feats = torch.full((20, 1536), 65504.0, dtype=torch.float16)
    feats[::2] = -65504.0
    got = kcenter_greedy(feats, 20, seed=0)
    assert sorted(got.tolist()) == list(range(20))
    assert (got[0] % 2) != (got[1] % 2)


# 확인(선택형 GPU 테스트): CUDA에서 선택 반복문이 장치→호스트 동기화를 한 번도 하지 않는다.
# torch.cuda.set_sync_debug_mode("error")가 그런 읽기를 모두 오류로 바꿔 준다.
# 왜: 반복마다 장치에서 값을 읽어 오면(`.item()`, 0차원 텐서로 인덱싱) 그 반복의 계산 자체보다 동기화 비용이
# 커진다(_select docstring). 기본 테스트는 GPU를 건드리면 안 되므로(다른 프로젝트와 GPU를 나눠 씀)
# DEFECT_INSPECT_GPU_TESTS=1일 때만 돈다.
@pytest.mark.skipif(
    os.environ.get("DEFECT_INSPECT_GPU_TESTS") != "1" or not torch.cuda.is_available(),
    reason="opt-in GPU test: set DEFECT_INSPECT_GPU_TESTS=1 (the default test run must not touch the GPU)",
)
def test_selection_loop_does_not_synchronise_on_cuda():
    n, n_select = 512, 64
    z = random_features(n, 16, seed=13).to("cuda")
    z -= z.mean(dim=0, keepdim=True)
    z_sq = torch.linalg.vector_norm(z, dim=1).square_()
    first = torch.tensor([5], device="cuda")
    torch.cuda.set_sync_debug_mode("error")
    try:
        picked = coreset._select(z, z_sq, first, n_select)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert picked.device.type == "cuda" and picked.dtype == torch.int64
    picked = picked.tolist()
    assert picked[0] == 5 and len(set(picked)) == n_select


# 확인: 점이 하나면 그 점만 돌려주고, 빈 입력·1차원 입력·n_select 0은 ValueError다.
def test_single_point_and_invalid_arguments():
    assert kcenter_greedy(torch.zeros(1, 4), 3).tolist() == [0]
    with pytest.raises(ValueError):
        kcenter_greedy(torch.zeros(0, 4), 1)
    with pytest.raises(ValueError):
        kcenter_greedy(torch.zeros(4), 1)
    with pytest.raises(ValueError):
        kcenter_greedy(torch.zeros(4, 4), 0)
