import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

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


class PoolExtractor(torch.nn.Module):
    """Fake extractor: average-pools the normalised image into a [B, grid, grid, 3] feature grid."""

    name = "pool"
    dim = 3

    def __init__(self, grid: int = 4) -> None:
        super().__init__()
        self.grid = grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, self.grid).permute(0, 2, 3, 1)


def pooled_features(images: np.ndarray, grid: int) -> np.ndarray:
    """What PoolExtractor returns, computed with numpy: float64 [N, grid * grid, 3] after fp16 rounding."""
    n, size = images.shape[0], images.shape[1]
    x = (
        images.astype(np.float32) / np.float32(255.0) - np.array(IMAGENET_MEAN, dtype=np.float32)
    ) / np.array(IMAGENET_STD, dtype=np.float32)
    block = size // grid
    x = x.reshape(n, grid, block, grid, block, 3).mean(axis=(2, 4), dtype=np.float32)
    return x.reshape(n, grid * grid, 3).astype(np.float16).astype(np.float64)


def query_features(extractor: torch.nn.Module, images: np.ndarray) -> np.ndarray:
    """The fp16 features score_images works with, as float64 [N, patches, D]."""
    feats, (h, w) = collect_features(extractor, images, device="cpu")
    return feats.double().numpy().reshape(len(images), h * w, -1)


def random_images(n: int, size: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(n, size, size, 3), dtype=np.uint8)


def nearest_numpy(query: np.ndarray, bank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dist = np.sqrt(((query[:, None, :] - bank[None, :, :]) ** 2).sum(axis=2))
    index = dist.argmin(axis=1)
    return dist[np.arange(len(query)), index], index


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


def test_collect_features_empty_and_invalid():
    extractor = PoolExtractor()
    feats, grid = collect_features(extractor, np.zeros((0, 8, 8, 3), dtype=np.uint8), device="cpu")
    assert feats.shape == (0, 3) and feats.dtype == torch.float16 and grid == (0, 0)
    with pytest.raises(ValueError):
        collect_features(extractor, np.zeros((8, 8, 3), dtype=np.uint8), device="cpu")
    with pytest.raises(ValueError):
        collect_features(extractor, random_images(2, 8, seed=0), batch_size=0, device="cpu")


def test_collect_features_rejects_fp16_overflow():
    class Huge(PoolExtractor):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return super().forward(x) * 1e6

    with pytest.raises(FloatingPointError):
        collect_features(Huge(), random_images(1, 8, seed=0), device="cpu")


class DropoutExtractor(PoolExtractor):
    """PoolExtractor followed by dropout: its output is only deterministic in eval mode."""

    def __init__(self, grid: int = 4) -> None:
        super().__init__(grid)
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(super().forward(x))


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


def test_build_bank_rounds_the_row_count_to_nearest():
    feats = torch.randn(96, 6, generator=torch.Generator().manual_seed(2)).to(torch.float16)
    # 0.3 * 96 = 28.8 -> 29 (truncating would give 28).
    assert build_bank(feats, 0.3, seed=0, device="cpu").shape == (29, 6)
    # 0.1 * 94 = 9.4 -> 9 (rounding up would give 10).
    assert build_bank(feats[:94], 0.1, seed=0, device="cpu").shape == (9, 6)
    # 0.1 * 4 = 0.4 -> 0, raised to the minimum of one row.
    assert build_bank(feats[:4], 0.1, seed=0, device="cpu").shape == (1, 6)


def test_build_bank_rejects_non_finite_features():
    feats = torch.randn(40, 6, generator=torch.Generator().manual_seed(3))
    feats[11, 2] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        build_bank(feats, 0.25, seed=0, device="cpu")


def test_build_bank_uses_the_seed():
    feats = torch.randn(64, 4, generator=torch.Generator().manual_seed(1))
    first_rows = {tuple(build_bank(feats, 0.1, seed=s, device="cpu")[0].tolist()) for s in range(6)}
    assert len(first_rows) > 1


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


def test_score_images_empty_input():
    extractor = PoolExtractor()
    bank = torch.zeros(4, 3, dtype=torch.float16)
    result = score_images(extractor, bank, np.zeros((0, 32, 32, 3), dtype=np.uint8), device="cpu", map_size=8)
    assert result.image_scores.shape == (0,) and result.maps.shape == (0, 8, 8)
    with pytest.raises(ValueError):
        score_images(extractor, torch.zeros(0, 3), random_images(1, 32, seed=0), device="cpu")
    with pytest.raises(ValueError):
        score_images(extractor, torch.zeros(4, 5), random_images(1, 32, seed=0), device="cpu")


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


def test_batch_size_does_not_change_scores():
    extractor = PoolExtractor(grid=4)
    train, test = random_images(6, 32, seed=13), random_images(5, 32, seed=14)
    feats, _ = collect_features(extractor, train, device="cpu")
    bank = build_bank(feats, 0.25, device="cpu")
    a = score_images(extractor, bank, test, batch_size=1, device="cpu", sigma=2.0, map_size=32)
    b = score_images(extractor, bank, test, batch_size=5, device="cpu", sigma=2.0, map_size=32)
    np.testing.assert_allclose(a.image_scores, b.image_scores, rtol=1e-6)
    np.testing.assert_allclose(a.maps.astype(np.float32), b.maps.astype(np.float32), rtol=1e-3)


def test_score_images_does_not_modify_the_bank():
    extractor = PoolExtractor(grid=4)
    feats, _ = collect_features(extractor, random_images(4, 8, seed=15), device="cpu")
    bank = build_bank(feats, 0.5, device="cpu").float()
    before = bank.clone()
    score_images(extractor, bank, random_images(2, 8, seed=16), device="cpu", sigma=0.0)
    assert torch.equal(bank, before)


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


def test_blur_needs_a_map_larger_than_its_radius():
    extractor = PoolExtractor(grid=4)
    bank = torch.zeros(4, 3, dtype=torch.float16)
    with pytest.raises(ValueError):
        score_images(extractor, bank, random_images(1, 8, seed=0), device="cpu", sigma=4.0)


def hot_patch_images(row: int, col: int, size: int = 32, grid: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Uniform grey training images and one test image with a single bright block."""
    train = np.full((2, size, size, 3), 100, dtype=np.uint8)
    test = np.full((1, size, size, 3), 100, dtype=np.uint8)
    block = size // grid
    test[0, row * block : (row + 1) * block, col * block : (col + 1) * block] = 250
    return train, test


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
