# [파일 역할] 직접 구현한 PatchCore(src/defect_inspect/patchcore.py)의 torch 경로를 검증한다: 패치 특징
# 모으기(collect_features), 코어셋 메모리 뱅크(build_bank), 채점(score_images: 최근접 거리, 재가중 이미지
# 점수, 점수 맵).
# [왜 필요한가] PatchCore는 학습 없이 '가장 가까운 정상 패치까지의 거리'로 점수를 매기므로, 수식이 조금만
# 틀려도 모든 단계의 기준선(P0) 수치가 틀린다. 실제 백본 대신 평균 풀링만 하는 가짜 추출기를 써서 기대값을
# numpy로 독립 계산하고, fp32 전개식의 상쇄 오차·동점·중복 행 같은 수치 함정을 일부러 만들어 확인한다. 실제
# 백본으로 anomalib과 대조한 점검은 docs/experiments.md 1단계 '구현 점검'에 따로 있다.
# [실행 조건] torch와 torchvision이 필요해(train 그룹), 없으면 파일 전체를 건너뛴다(CI). 모든 계산은
# device="cpu"로 해 GPU 없이 돈다.

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

# torch가 없는 환경(CI는 기본·serve 그룹만 설치, .github/workflows/ci.yml)에서는 이 파일을 실패가 아니라
# 건너뜀으로 처리한다. 아래 import들이 이 줄 뒤에 와야 하므로 줄 끝에 E402(모듈 첫머리가 아닌 import) 예외
# 표시를 달았다.
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch.nn.functional as F  # noqa: E402

from defect_inspect import patchcore  # noqa: E402
from defect_inspect.backbones import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from defect_inspect.coreset import kcenter_greedy  # noqa: E402
from defect_inspect.patchcore import (  # noqa: E402
    ScoreResult,
    build_bank,
    collect_features,
    gaussian_kernel1d,
    score_images,
)


# 실제 백본(WideResNet, DINOv2) 대신 쓰는 가짜 특징 추출기. 정규화된 이미지를 grid×grid로 평균 풀링해 [B,
# grid, grid, 3] 패치 특징을 낸다. 실제 추출기처럼 name과 dim 속성을 가진다(collect_features가 빈 입력일 때
# dim을 읽는다). 결과를 numpy로 쉽게 다시 계산할 수 있어 기대값을 독립적으로 만들 수 있다.
class PoolExtractor(torch.nn.Module):
    """Fake extractor: average-pools the normalised image into a [B, grid, grid, 3] feature grid."""

    name = "pool"
    dim = 3

    def __init__(self, grid: int = 4) -> None:
        super().__init__()
        self.grid = grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, self.grid).permute(0, 2, 3, 1)


# PoolExtractor 출력을 numpy로 다시 계산한 기대값. 파이프라인과 같이 fp16으로 한 번 반올림한 뒤 float64로
# 돌려준다.
def pooled_features(images: np.ndarray, grid: int) -> np.ndarray:
    """What PoolExtractor returns, computed with numpy: float64 [N, grid * grid, 3] after fp16 rounding."""
    n, size = images.shape[0], images.shape[1]
    x = (
        images.astype(np.float32) / np.float32(255.0) - np.array(IMAGENET_MEAN, dtype=np.float32)
    ) / np.array(IMAGENET_STD, dtype=np.float32)
    block = size // grid
    x = x.reshape(n, grid, block, grid, block, 3).mean(axis=(2, 4), dtype=np.float32)
    return x.reshape(n, grid * grid, 3).astype(np.float16).astype(np.float64)


# score_images가 실제로 쓰는 질의 특징(fp16 반올림된 값)을 float64 [N, 패치 수, D]로 꺼낸다. numpy 기준 계산이
# 같은 입력에서 출발하게 하려는 것이다.
def query_features(extractor: torch.nn.Module, images: np.ndarray) -> np.ndarray:
    """The fp16 features score_images works with, as float64 [N, patches, D]."""
    feats, (h, w) = collect_features(extractor, images, device="cpu")
    return feats.double().numpy().reshape(len(images), h * w, -1)


# seed로 결정되는 무작위 uint8 이미지 [n, size, size, 3].
def random_images(n: int, size: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(n, size, size, 3), dtype=np.uint8)


# 모든 (질의, 뱅크 행) 거리를 직접 계산하는 최근접 기준 구현(float64, 전개식 없음).
def nearest_numpy(query: np.ndarray, bank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dist = np.sqrt(((query[:, None, :] - bank[None, :, :]) ** 2).sum(axis=2))
    index = dist.argmin(axis=1)
    return dist[np.arange(len(query)), index], index


# PatchCore 이미지 점수의 단계별 numpy 기준 구현: 가장 큰 패치 점수 s*의 패치 p*, 그 최근접 뱅크 행 m*, m*에
# 가까운 뱅크 행 k개(m*가 맨 앞이어야 함), 그 k개까지의 거리 d로 가중치 1 - softmax(d)[0]를 만들어 s*에
# 곱한다(experiments.md 1단계 P0: PatchCore 논문의 재가중, 이웃 9개).
def image_score_numpy(feats: np.ndarray, bank: np.ndarray, reweight_k: int) -> float:
    """PatchCore image score for one image, written out step by step in float64."""
    dist, index = nearest_numpy(feats, bank)
    p_star = int(dist.argmax())
    s_star = dist[p_star]
    if reweight_k <= 1:
        return float(s_star)
    m_star = index[p_star]
    to_m_star = np.sqrt(((bank - bank[m_star]) ** 2).sum(axis=1))
    support = np.argsort(to_m_star, kind="stable")[: min(reweight_k, len(bank))]
    assert support[0] == m_star
    d = np.sqrt(((bank[support] - feats[p_star]) ** 2).sum(axis=1))
    softmax = np.exp(d - d.max())
    softmax /= softmax.sum()
    return float((1.0 - softmax[0]) * s_star)


# collect_features의 출력 계약을 본다: 행은 이미지 순서, 그 안에서 패치 격자의 행 우선 순서, 모양 [N·H·W, D],
# fp16, CPU. 값이 numpy 기대값과 같고, 배치 크기를 바꿔도 결과가 같아야 한다. 마지막에 이미지 3의 패치 (1,
# 2)를 손으로 계산해 행 순서를 직접 확인한다. 뱅크와 교차 적합 코드는 이 순서로 행을 이미지별로 나누므로
# 순서가 틀리면 결과가 섞인다.
def test_collect_features_shape_order_and_values():
    images = random_images(5, 8, seed=0)
    extractor = PoolExtractor(grid=4)
    feats, grid = collect_features(extractor, images, batch_size=2, device="cpu")
    assert grid == (4, 4)
    assert feats.shape == (5 * 16, 3)
    assert feats.dtype == torch.float16
    assert feats.device.type == "cpu"
    expected = pooled_features(images, 4).reshape(-1, 3)
    np.testing.assert_allclose(feats.double().numpy(), expected, atol=2e-3)
    # Batching does not change anything.
    whole, _ = collect_features(extractor, images, batch_size=32, device="cpu")
    assert torch.equal(whole, feats)
    # Row order: image, then row-major patches. Patch (1, 2) of image 3 is the mean of its 2x2 block.
    block = images[3, 2:4, 4:6].reshape(-1, 3).astype(np.float64).mean(axis=0) / 255.0
    manual = (block - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    np.testing.assert_allclose(feats[3 * 16 + 1 * 4 + 2].double().numpy(), manual, atol=2e-3)


# 이미지 0장이면 [0, dim] 빈 특징과 격자 (0, 0)을 돌려주고, 4차원이 아닌 입력이나 batch_size 0은 ValueError다.
def test_collect_features_empty_and_invalid():
    extractor = PoolExtractor()
    feats, grid = collect_features(extractor, np.zeros((0, 8, 8, 3), dtype=np.uint8), device="cpu")
    assert feats.shape == (0, 3) and feats.dtype == torch.float16 and grid == (0, 0)
    with pytest.raises(ValueError):
        collect_features(extractor, np.zeros((8, 8, 3), dtype=np.uint8), device="cpu")
    with pytest.raises(ValueError):
        collect_features(extractor, random_images(2, 8, seed=0), batch_size=0, device="cpu")


# 특징이 fp16 범위를 넘으면(여기서는 100만 배) 조용히 inf로 저장하지 않고 FloatingPointError를 내는지 본다.
# 뱅크와 질의는 모두 fp16으로 반올림한 값을 쓰므로 넘침은 모든 거리를 망가뜨린다. 서빙 쪽 inspector도 같은
# 오류를 낸다.
def test_collect_features_rejects_fp16_overflow():
    class Huge(PoolExtractor):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return super().forward(x) * 1e6

    with pytest.raises(FloatingPointError):
        collect_features(Huge(), random_images(1, 8, seed=0), device="cpu")


# 출력에 드롭아웃을 붙인 추출기. train 모드면 출력이 매번 달라지고 eval 모드에서만 결정적이다.
class DropoutExtractor(PoolExtractor):
    """PoolExtractor followed by dropout: its output is only deterministic in eval mode."""

    def __init__(self, grid: int = 4) -> None:
        super().__init__(grid)
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(super().forward(x))


# collect_features와 score_images가 추출기를 eval 모드로 바꾸는지 본다. 학습 모드로 넘겨진
# 추출기(드롭아웃·BatchNorm이 있는 실제 백본)를 그대로 쓰면 특징이 무작위로 흔들려 뱅크와 점수가 재현되지
# 않는다. 먼저 train 모드에서 드롭아웃이 실제로 출력을 바꾼다는 것을 확인해 테스트가 헛돌지 않게 한다.
def test_extractor_is_switched_to_eval_mode():
    train, test = random_images(3, 8, seed=19), random_images(2, 8, seed=20)
    plain, _ = collect_features(PoolExtractor(), train, device="cpu")
    # Sanity: left in train mode, the dropout layer does change the features.
    noisy = DropoutExtractor().train()
    with torch.no_grad():
        assert not torch.equal(noisy(torch.ones(1, 3, 8, 8)), PoolExtractor()(torch.ones(1, 3, 8, 8)))
    feats, _ = collect_features(noisy, train, device="cpu")
    assert torch.equal(feats, plain)

    bank = build_bank(plain, 0.25, device="cpu")
    expected = score_images(PoolExtractor(), bank, test, device="cpu", sigma=0.0, map_size=8)
    got = score_images(DropoutExtractor().train(), bank, test, device="cpu", sigma=0.0, map_size=8)
    np.testing.assert_array_equal(got.image_scores, expected.image_scores)
    np.testing.assert_array_equal(got.maps, expected.maps)


# build_bank가 비율만큼 행을 고르고(200행의 10% = 20행), fp16 CPU 텐서로, kcenter_greedy의 선택 순서대로
# 돌려주는지 본다. 핵심은 '작은 비율의 뱅크 = 큰 비율 뱅크의 앞부분'(중첩)이다. 4단계 격자(run_grid)는 가장 큰
# 비율로 한 번만 고르고 작은 비율은 앞부분을 잘라 쓰므로, 이 성질이 깨지면 비율별 결과가 다른 선택에서 나온
# 것이 된다. 아주 작은 비율은 최소 1행, 1 이상은 전부(같은 행 집합), fp32 입력도 fp16 뱅크가 되어야 한다.
def test_build_bank_size_order_and_nesting():
    feats = torch.randn(200, 6, generator=torch.Generator().manual_seed(0)).to(torch.float16)
    bank = build_bank(feats, 0.1, seed=0, device="cpu")
    assert bank.shape == (20, 6)
    assert bank.dtype == torch.float16 and bank.device.type == "cpu"
    assert torch.equal(bank, feats[kcenter_greedy(feats, 20, seed=0)])
    # Selection order: a smaller ratio is a prefix of a larger one.
    big = build_bank(feats, 0.5, seed=0, device="cpu")
    assert big.shape == (100, 6)
    assert torch.equal(big[:20], bank)
    assert torch.equal(build_bank(feats, 0.01, seed=0, device="cpu"), big[:2])
    # At least one row, and everything for ratio >= 1.
    assert build_bank(feats, 1e-6, seed=0, device="cpu").shape == (1, 6)
    for ratio in (1.0, 2.5):
        everything = build_bank(feats, ratio, seed=0, device="cpu")
        assert everything.shape == (200, 6)
        assert torch.equal(everything[:100], big)
        assert sorted(map(tuple, everything.tolist())) == sorted(map(tuple, feats.tolist()))
    # fp32 features still give an fp16 bank.
    assert build_bank(feats.float(), 0.1, seed=0, device="cpu").dtype == torch.float16


# 뱅크 행 수는 비율 × 특징 수를 반올림(내림·올림이 아님)하고 최소 1행이다. run_grid.ratio_rows가 같은 규칙으로
# 앞부분 길이를 계산하므로 둘이 어긋나면 안 된다.
def test_build_bank_rounds_the_row_count_to_nearest():
    feats = torch.randn(96, 6, generator=torch.Generator().manual_seed(2)).to(torch.float16)
    # 0.3 * 96 = 28.8 -> 29 (truncating would give 28).
    assert build_bank(feats, 0.3, seed=0, device="cpu").shape == (29, 6)
    # 0.1 * 94 = 9.4 -> 9 (rounding up would give 10).
    assert build_bank(feats[:94], 0.1, seed=0, device="cpu").shape == (9, 6)
    # 0.1 * 4 = 0.4 -> 0, raised to the minimum of one row.
    assert build_bank(feats[:4], 0.1, seed=0, device="cpu").shape == (1, 6)


# NaN이 섞인 특징으로 뱅크를 만들면 kcenter_greedy가 ValueError를 낸다. 깨진 행이 뱅크에 들어가면 모든 거리가
# NaN이 될 수 있다.
def test_build_bank_rejects_non_finite_features():
    feats = torch.randn(40, 6, generator=torch.Generator().manual_seed(3))
    feats[11, 2] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        build_bank(feats, 0.25, seed=0, device="cpu")


# seed가 실제로 선택(시작점)에 쓰이는지 본다. seed 0~5에서 첫 행이 모두 같으면 seed가 무시되고 있다는 뜻이다.
def test_build_bank_uses_the_seed():
    feats = torch.randn(64, 4, generator=torch.Generator().manual_seed(1))
    first_rows = {tuple(build_bank(feats, 0.1, seed=s, device="cpu")[0].tolist()) for s in range(6)}
    assert len(first_rows) > 1


# score_images 결과의 형식: 이미지 점수 float32 [N], 맵 float16 [N, map_size, map_size], 모두 유한하고 양수.
# 기본 구성(σ 4, 이웃 9, 256 맵)이 작은 32px 입력에서도 도는지 확인한다.
def test_score_result_shapes_and_dtypes():
    extractor = PoolExtractor(grid=4)
    train, test = random_images(6, 32, seed=0), random_images(3, 32, seed=1)
    feats, _ = collect_features(extractor, train, device="cpu")
    bank = build_bank(feats, 0.25, device="cpu")
    result = score_images(extractor, bank, test, batch_size=2, device="cpu", sigma=1.0, map_size=16)
    assert isinstance(result, ScoreResult)
    assert result.image_scores.shape == (3,) and result.image_scores.dtype == np.float32
    assert result.maps.shape == (3, 16, 16) and result.maps.dtype == np.float16
    assert np.isfinite(result.image_scores).all() and np.isfinite(result.maps).all()
    assert (result.image_scores > 0).all() and (result.maps > 0).all()
    # The default configuration (sigma 4, 9 neighbours, 256 maps) runs on a 32 px input as well.
    default = score_images(extractor, bank, test, device="cpu")
    assert default.maps.shape == (3, 256, 256)


# 이미지 0장이면 빈 결과를 돌려주고, 빈 뱅크나 특징 차원(3)과 다른 뱅크(5)는 ValueError다.
def test_score_images_empty_input():
    extractor = PoolExtractor()
    bank = torch.zeros(4, 3, dtype=torch.float16)
    result = score_images(extractor, bank, np.zeros((0, 32, 32, 3), dtype=np.uint8), device="cpu", map_size=8)
    assert result.image_scores.shape == (0,) and result.maps.shape == (0, 8, 8)
    with pytest.raises(ValueError):
        score_images(extractor, torch.zeros(0, 3), random_images(1, 32, seed=0), device="cpu")
    with pytest.raises(ValueError):
        score_images(extractor, torch.zeros(4, 5), random_images(1, 32, seed=0), device="cpu")


# 뱅크를 만든 이미지 자체를 채점하면 모든 패치가 뱅크에 있으므로 점수와 맵이 0이어야 한다. 재가중을 켜든(9)
# 끄든(1) 같다.
def test_bank_members_score_zero():
    extractor = PoolExtractor(grid=4)
    images = random_images(4, 32, seed=2)
    feats, _ = collect_features(extractor, images, device="cpu")
    bank = build_bank(feats, 1.0, device="cpu")
    for reweight_k in (1, 9):
        result = score_images(
            extractor, bank, images, device="cpu", reweight_k=reweight_k, sigma=1.0, map_size=32
        )
        np.testing.assert_allclose(result.image_scores, 0.0, atol=1e-6)
        np.testing.assert_allclose(result.maps.astype(np.float32), 0.0, atol=1e-6)


# _nearest의 수치 정확도를 본다. ReLU 뒤 특징은 원점에서 멀리 있어(+50 오프셋) ||b||^2 - 2q·b 전개식만 쓰면 0
# 근처 거리가 반올림 잡음이 된다. 구현은 뱅크를 중심화하고, 찾은 행까지 거리를 직접 뺄셈으로 다시 계산한다. 그
# 결과 1e-3 크기의 작은 거리도 5% 안으로 맞고, 뱅크 행을 그대로 질의하면 정확히 0이어야 한다.
def test_nearest_is_accurate_near_zero_for_large_norms():
    # Post-ReLU features sit far from the origin; the expanded form alone would return noise here.
    generator = torch.Generator().manual_seed(0)
    bank = torch.randn(500, 64, generator=generator) + 50.0
    offset = 1e-3 * torch.randn(20, 64, generator=generator)
    query = bank[100:120] + offset
    expected = (query.double() - bank[100:120].double()).norm(dim=1)
    centre = bank.mean(dim=0, keepdim=True)
    centred = bank - centre
    bank_sq = torch.linalg.vector_norm(centred, dim=1).square()
    dist, index = patchcore._nearest(query - centre, centred, bank_sq)
    assert index.tolist() == list(range(100, 120))
    np.testing.assert_allclose(dist.numpy(), expected.numpy(), rtol=0.05)
    assert float(dist.max()) < 0.02
    # Exact members give exactly zero.
    dist, index = patchcore._nearest(centred[:50].clone(), centred, bank_sq)
    assert index.tolist() == list(range(50))
    assert float(dist.abs().max()) == 0.0


# 4px 이미지를 4×4 격자로 나눠 패치 하나 = 화소 하나로 만들고 흐림을 끄면(σ = 0), 점수 맵이 곧 패치별 최근접
# 거리 격자다. 이것을 numpy 전수 계산과 맞추고, reweight_k = 1이면 이미지 점수가 가장 큰 패치 점수와 같은지
# 본다. 맵은 fp16으로 저장되므로 허용 오차를 그만큼 둔다.
def test_patch_scores_match_brute_force():
    # One patch per pixel and no blur: the map is the grid of nearest-neighbour distances itself.
    extractor = PoolExtractor(grid=4)
    train, test = random_images(6, 4, seed=3), random_images(5, 4, seed=4)
    feats, grid = collect_features(extractor, train, device="cpu")
    assert grid == (4, 4)
    bank = build_bank(feats, 0.3, device="cpu")
    result = score_images(
        extractor, bank, test, batch_size=2, device="cpu", reweight_k=1, sigma=0.0, map_size=4
    )
    bank64 = bank.double().numpy()
    queries = query_features(extractor, test)
    for i in range(len(test)):
        expected, _ = nearest_numpy(queries[i], bank64)
        np.testing.assert_allclose(
            result.maps[i].astype(np.float64).reshape(-1), expected, rtol=2e-3, atol=1e-4
        )
        # reweight_k <= 1: the image score is the largest patch score.
        np.testing.assert_allclose(result.image_scores[i], expected.max(), rtol=1e-5)


# 재가중 이미지 점수를 numpy 기준(image_score_numpy)과 맞춘다. k는 기본값 9부터 재가중이 꺼지는 1·0까지 본다.
# k > 1이면 가중치가 1보다 작으므로 재가중 점수는 항상 재가중 전 점수보다 작아야 한다.
@pytest.mark.parametrize("reweight_k", [9, 3, 2, 1, 0])
def test_image_score_reweighting_matches_numpy(reweight_k):
    extractor = PoolExtractor(grid=4)
    train, test = random_images(8, 8, seed=5), random_images(7, 8, seed=6)
    feats, _ = collect_features(extractor, train, device="cpu")
    bank = build_bank(feats, 0.25, device="cpu")
    assert bank.shape[0] == 32
    result = score_images(extractor, bank, test, batch_size=3, device="cpu", reweight_k=reweight_k, sigma=0.0)
    bank64 = bank.double().numpy()
    queries = query_features(extractor, test)
    expected = [image_score_numpy(queries[i], bank64, reweight_k) for i in range(len(test))]
    np.testing.assert_allclose(result.image_scores, expected, rtol=1e-4)
    if reweight_k > 1:
        plain = [image_score_numpy(queries[i], bank64, 1) for i in range(len(test))]
        assert (result.image_scores < np.array(plain)).all()


# 뱅크가 k(9)보다 작은 5행일 때 k를 뱅크 크기로 줄여 쓰는지(min(k, R)) 본다. topk가 뱅크보다 큰 k로 불리면
# 오류가 난다.
def test_reweighting_with_a_bank_smaller_than_k():
    extractor = PoolExtractor(grid=4)
    train, test = random_images(2, 8, seed=7), random_images(4, 8, seed=8)
    feats, _ = collect_features(extractor, train, device="cpu")
    bank = build_bank(feats, 5 / 32, device="cpu")
    assert bank.shape[0] == 5
    result = score_images(extractor, bank, test, device="cpu", reweight_k=9, sigma=0.0)
    bank64 = bank.double().numpy()
    queries = query_features(extractor, test)
    expected = [image_score_numpy(queries[i], bank64, 9) for i in range(len(test))]
    np.testing.assert_allclose(result.image_scores, expected, rtol=1e-4)


# fp32 전개식이 m*와 다른 행 b를 구별하지 못하는 경우를 일부러 만든다(아래 영어 주석의 2^24 계산). 그래도
# 구현은 m* 자리를 -inf로 두어 지지 집합의 맨 앞에 m*를 놓으므로, 가중치는 b까지 거리(4)가 아니라 m*까지
# 거리(3)로 계산되어야 한다. m*가 뱅크의 0번이든 1번이든 같아야 한다.
@pytest.mark.parametrize("m_index", [0, 1])
def test_reweighting_forces_the_nearest_row_first_when_the_search_cannot_separate_it(m_index):
    # m* = (4096, 0) and b = (4096, 1) are indistinguishable to the fp32 expanded form around m*:
    # ||b||^2 = 2**24 + 1 rounds to 2**24 and m*.b = 2**24, so both rows get exactly -2**24. The
    # weight must still use the distance to m* (3), not to b (4), wherever m* sits in the bank.
    m, b, far = [4096.0, 0.0], [4096.0, 1.0], [0.0, 0.0]
    bank = torch.tensor([m, b, far] if m_index == 0 else [b, m, far])
    bank_sq = torch.linalg.vector_norm(bank, dim=1).square_()
    assert bank_sq[0] == bank_sq[1] == 2.0**24
    query = torch.tensor([[[4096.0, -3.0]]])  # one image, one patch
    patch_scores = torch.tensor([[3.0]])
    nearest = torch.tensor([[m_index]])
    score = patchcore._image_scores(patch_scores, nearest, query, bank, bank_sq, reweight_k=2)
    # softmax([3, 4])[0] = 1 / (1 + e); with b first it would be e / (1 + e).
    assert float(score[0]) == pytest.approx(3.0 * (1.0 - 1.0 / (1.0 + np.e)), rel=1e-6)


# 뱅크의 모든 행이 세 번씩 들어 있으면 m*의 이웃은 자기 복사본 셋과 그다음 다른 행이다. 복사본도 각각 한
# 행으로 세어야 하고, 탐색이 어느 복사본을 돌려주든 가중치가 같아야 한다. 기대값은 거리 0인 복사본 셋 + 네
# 번째로 가까운 행으로 직접 만든다.
def test_reweighting_counts_duplicated_bank_rows_separately():
    # The bank holds every row three times: the support of m* is its three copies, then the next
    # distinct row. Whichever copy the search returns, the weight is the same.
    extractor = PoolExtractor(grid=4)
    train, test = random_images(2, 8, seed=9), random_images(3, 8, seed=10)
    feats, _ = collect_features(extractor, train, device="cpu")
    base = build_bank(feats, 0.25, device="cpu")
    bank = torch.cat([base, base, base])
    result = score_images(extractor, bank, test, device="cpu", reweight_k=4, sigma=0.0)
    bank64 = bank.double().numpy()
    queries = query_features(extractor, test)
    for i in range(len(test)):
        dist, index = nearest_numpy(queries[i], bank64)
        p_star = int(dist.argmax())
        to_m_star = np.sqrt(((bank64 - bank64[index[p_star]]) ** 2).sum(axis=1))
        fourth = np.sort(to_m_star)[3]
        # Three copies of m* at distance 0, then the next distinct row.
        d = np.array(
            [dist[p_star]] * 3 + [np.sqrt(((queries[i][p_star] - bank64[to_m_star == fourth][0]) ** 2).sum())]
        )
        weight = 1.0 - np.exp(d[0]) / np.exp(d).sum()
        np.testing.assert_allclose(result.image_scores[i], weight * dist[p_star], rtol=1e-4)


# 메모리 상한(_MAX_DIST_ELEMS, 실제 2^27 원소)을 monkeypatch로 200까지 줄여 질의를 2행씩 블록으로 처리해도,
# 1로 줄여 한 행씩 처리해도 점수와 맵이 같아야 한다. 큰 뱅크에서 GPU 메모리를 넘기지 않으려고 블록으로 나누는
# 코드가 결과를 바꾸지 않는지 확인한다.
def test_chunked_search_gives_the_same_scores(monkeypatch):
    extractor = PoolExtractor(grid=4)
    train, test = random_images(10, 8, seed=11), random_images(6, 8, seed=12)
    feats, _ = collect_features(extractor, train, device="cpu")
    bank = build_bank(feats, 0.5, device="cpu")
    whole = score_images(extractor, bank, test, device="cpu", sigma=0.0, map_size=8)
    # 80 bank rows and a budget of 200 values: two query rows per block.
    monkeypatch.setattr(patchcore, "_MAX_DIST_ELEMS", 200)
    chunked = score_images(extractor, bank, test, device="cpu", sigma=0.0, map_size=8)
    np.testing.assert_allclose(chunked.image_scores, whole.image_scores, rtol=1e-6)
    np.testing.assert_allclose(chunked.maps.astype(np.float32), whole.maps.astype(np.float32), rtol=1e-3)
    # A budget smaller than one bank row still works, one query row at a time.
    monkeypatch.setattr(patchcore, "_MAX_DIST_ELEMS", 1)
    single = score_images(extractor, bank, test, device="cpu", sigma=0.0, map_size=8)
    np.testing.assert_allclose(single.image_scores, whole.image_scores, rtol=1e-6)


# batch_size 1과 5가 같은 점수와 맵을 내는지 본다. 배치 경계에서 결과를 잘못 이어 붙이는 회귀를 막는다.
def test_batch_size_does_not_change_scores():
    extractor = PoolExtractor(grid=4)
    train, test = random_images(6, 32, seed=13), random_images(5, 32, seed=14)
    feats, _ = collect_features(extractor, train, device="cpu")
    bank = build_bank(feats, 0.25, device="cpu")
    a = score_images(extractor, bank, test, batch_size=1, device="cpu", sigma=2.0, map_size=32)
    b = score_images(extractor, bank, test, batch_size=5, device="cpu", sigma=2.0, map_size=32)
    np.testing.assert_allclose(a.image_scores, b.image_scores, rtol=1e-6)
    np.testing.assert_allclose(a.maps.astype(np.float32), b.maps.astype(np.float32), rtol=1e-3)


# score_images는 뱅크를 중심화하는데, 넘겨받은 텐서를 제자리에서 바꾸면 같은 뱅크로 다음 채점을 할 때 결과가
# 달라진다. fp32 뱅크(장치 이동 시 복사가 일어나지 않을 수 있는 경우)를 넘겨도 원본이 그대로인지 본다. 구현은
# copy=True로 복사본을 만든다.
def test_score_images_does_not_modify_the_bank():
    extractor = PoolExtractor(grid=4)
    feats, _ = collect_features(extractor, random_images(4, 8, seed=15), device="cpu")
    bank = build_bank(feats, 0.5, device="cpu").float()
    before = bank.clone()
    score_images(extractor, bank, random_images(2, 8, seed=16), device="cpu", sigma=0.0)
    assert torch.equal(bank, before)


# 가우시안 커널: 크기 2·int(4σ+0.5)+1(σ 4면 33), 합 1, 좌우 대칭, 값이 정의식과 같다. 반경은 4σ를 반올림한
# 값이라 σ 0.4는 반경 2, 0.125는 1이다. σ ≤ 0이면 흐림 없음([1]). 서빙 쪽 inspector.gaussian_kernel1d도 같은
# 규칙이어야 맵이 같다.
def test_gaussian_kernel1d():
    kernel = gaussian_kernel1d(4.0)
    assert kernel.shape == (33,) and kernel.dtype == torch.float32
    assert float(kernel.sum()) == pytest.approx(1.0, abs=1e-6)
    assert torch.equal(kernel, kernel.flip(0))
    x = np.arange(-16, 17, dtype=np.float64)
    expected = np.exp(-(x**2) / (2 * 4.0**2))
    np.testing.assert_allclose(kernel.numpy(), expected / expected.sum(), rtol=1e-6)
    assert gaussian_kernel1d(1.0).shape == (9,)
    assert gaussian_kernel1d(0.6).shape == (5,)
    # The radius is 4 * sigma rounded to nearest, not truncated: 1.6 -> 2 and 0.5 -> 1.
    assert gaussian_kernel1d(0.4).shape == (5,)
    assert gaussian_kernel1d(0.125).shape == (3,)
    assert gaussian_kernel1d(0.0).tolist() == [1.0]
    assert gaussian_kernel1d(-1.0).tolist() == [1.0]


# 패치 하나 = 화소 하나(grid 32, 32px)로 만들면 확대가 항등이라 맵은 흐린 거리 격자 그 자체다. 이것을 scipy
# gaussian_filter(mode="mirror" = torch reflect 패딩, truncate=4로 같은 반경)와 맞춰 분리형 흐림과 패딩이
# 맞는지 본다.
def test_map_blur_matches_scipy():
    # One patch per pixel: upsampling is the identity, so the map is the blurred distance grid.
    extractor = PoolExtractor(grid=32)
    train, test = random_images(2, 32, seed=17), random_images(2, 32, seed=18)
    feats, grid = collect_features(extractor, train, device="cpu")
    assert grid == (32, 32)
    bank = build_bank(feats, 0.05, device="cpu")
    sharp = score_images(extractor, bank, test, device="cpu", sigma=0.0, map_size=32).maps.astype(np.float64)
    for sigma in (1.0, 2.0):
        blurred = score_images(extractor, bank, test, device="cpu", sigma=sigma, map_size=32).maps
        for i in range(len(test)):
            # scipy "mirror" is torch "reflect"; truncate=4 gives the same radius int(4 * sigma + 0.5).
            expected = gaussian_filter(sharp[i], sigma=sigma, mode="mirror", truncate=4.0)
            np.testing.assert_allclose(blurred[i].astype(np.float64), expected, rtol=3e-3, atol=1e-4)


# 흐림 반경(σ 4면 16)이 맵 크기(8px)보다 크면 torch reflect 패딩이 불가능하므로 명확한 ValueError를 내야 한다.
def test_blur_needs_a_map_larger_than_its_radius():
    extractor = PoolExtractor(grid=4)
    bank = torch.zeros(4, 3, dtype=torch.float16)
    with pytest.raises(ValueError):
        score_images(extractor, bank, random_images(1, 8, seed=0), device="cpu", sigma=4.0)


# 균일한 회색 학습 이미지 2장과, 격자 칸 (row, col) 하나만 밝게(250) 칠한 시험 이미지 1장을 만든다. 이상
# 위치와 점수를 손으로 계산할 수 있게 하려는 것이다.
def hot_patch_images(row: int, col: int, size: int = 32, grid: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Uniform grey training images and one test image with a single bright block."""
    train = np.full((2, size, size, 3), 100, dtype=np.uint8)
    test = np.full((1, size, size, 3), 100, dtype=np.uint8)
    block = size // grid
    test[0, row * block : (row + 1) * block, col * block : (col + 1) * block] = 250
    return train, test


# 밝은 칸 하나가 있을 때 맵의 최댓값이 그 칸 가운데에 오고(흐림이 위치를 옮기지 않음), 다른 칸은 뱅크와 같아
# 먼 곳은 0으로 사라지는지 본다. 이미지 점수는 회색과의 차이(채널당 150/255를 ImageNet 표준편차로 나눈 값)의
# 유클리드 거리와 같아야 한다. σ = 0이면 bilinear 확대 때문에 맵 최댓값이 패치 점수의 (1 - 1/16)^2배가 된다.
@pytest.mark.parametrize("sigma", [0.0, 1.0, 2.0])
def test_single_hot_patch_keeps_the_maximum_location(sigma):
    extractor = PoolExtractor(grid=4)
    row, col = 1, 2
    train, test = hot_patch_images(row, col)
    feats, _ = collect_features(extractor, train, device="cpu")
    bank = build_bank(feats, 1.0, device="cpu")
    result = score_images(extractor, bank, test, device="cpu", reweight_k=1, sigma=sigma, map_size=32)
    heat = result.maps[0].astype(np.float32)
    # Patch (row, col) covers pixels 8*row..8*row+7; bilinear upsampling peaks at its two centre pixels.
    peak_r, peak_c = np.unravel_index(int(heat.argmax()), heat.shape)
    assert peak_r in (8 * row + 3, 8 * row + 4)
    assert peak_c in (8 * col + 3, 8 * col + 4)
    # Every other patch equals the bank exactly, so the map fades to zero away from the hot block.
    assert heat[28:, :8].max() < 1e-3 * heat.max()
    # The hot patch differs from grey by 150/255 per channel, scaled by the ImageNet std.
    expected = np.sqrt((((250 - 100) / 255.0 / np.array(IMAGENET_STD)) ** 2).sum())
    np.testing.assert_allclose(result.image_scores[0], expected, rtol=2e-3)
    assert heat.max() <= expected * 1.002
    if sigma == 0.0:
        np.testing.assert_allclose(heat.max(), expected * (1 - 1 / 16) ** 2, rtol=2e-3)


# 입력 크기(32)와 다른 map_size(16)를 주면, 32 맵을 torch bilinear로 16에 맞춘 것과 같고 최댓값 위치도 따라
# 옮겨지는지 본다. 픽셀 지표는 모든 구성에서 256×256 격자로 계산하므로(experiments.md 전처리) 이 크기 조정이
# 맞아야 한다. 입력보다 큰 맵(48)도 된다.
def test_map_is_resized_to_map_size():
    extractor = PoolExtractor(grid=4)
    row, col = 1, 2
    train, test = hot_patch_images(row, col)
    feats, _ = collect_features(extractor, train, device="cpu")
    bank = build_bank(feats, 1.0, device="cpu")
    full = score_images(extractor, bank, test, device="cpu", sigma=1.0, map_size=32).maps
    half = score_images(extractor, bank, test, device="cpu", sigma=1.0, map_size=16).maps
    assert half.shape == (1, 16, 16)
    expected = F.interpolate(
        torch.from_numpy(full.astype(np.float32))[:, None],
        size=(16, 16),
        mode="bilinear",
        align_corners=False,
    )[:, 0].numpy()
    np.testing.assert_allclose(half.astype(np.float32), expected, rtol=2e-3, atol=1e-4)
    peak_r, peak_c = np.unravel_index(int(half[0].argmax()), (16, 16))
    assert peak_r in (4 * row + 1, 4 * row + 2) and peak_c in (4 * col + 1, 4 * col + 2)
    # Larger than the input works too.
    assert score_images(extractor, bank, test, device="cpu", sigma=1.0, map_size=48).maps.shape == (1, 48, 48)
