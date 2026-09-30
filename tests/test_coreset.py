import math
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from defect_inspect import coreset  # noqa: E402
from defect_inspect.coreset import kcenter_greedy  # noqa: E402


def reference_kcenter(points: np.ndarray, first: int, n_select: int) -> list[int]:
    """Plain float64 greedy k-center."""
    chosen = [first]
    min_dist = np.full(len(points), np.inf)
    while len(chosen) < n_select:
        dist = ((points - points[chosen[-1]]) ** 2).sum(axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[chosen] = -1.0
        chosen.append(int(np.argmax(min_dist)))
    return chosen


def random_features(n: int, dim: int, seed: int) -> torch.Tensor:
    return torch.randn(n, dim, generator=torch.Generator().manual_seed(seed))


def test_matches_reference_without_projection():
    feats = random_features(200, 8, seed=1)
    for seed in (0, 3):
        first = int(torch.randint(200, (1,), generator=torch.Generator().manual_seed(seed)))
        got = kcenter_greedy(feats, 40, seed=seed)
        assert got.dtype == torch.int64 and got.shape == (40,)
        assert got.tolist() == reference_kcenter(feats.double().numpy(), first, 40)


def test_matches_reference_with_projection():
    feats = random_features(300, 32, seed=2) + 5.0
    seed, proj_dim = 7, 4
    generator = torch.Generator().manual_seed(seed)
    proj = torch.randn(32, proj_dim, generator=generator) / math.sqrt(proj_dim)
    first = int(torch.randint(300, (1,), generator=generator))
    projected = (feats.double() @ proj.double()).numpy()
    got = kcenter_greedy(feats, 30, seed=seed, proj_dim=proj_dim)
    assert got.tolist() == reference_kcenter(projected, first, 30)
    # The projection changes the geometry: selecting in the full space gives a different order.
    assert got.tolist() != kcenter_greedy(feats, 30, seed=seed, proj_dim=32).tolist()


def test_projection_starts_strictly_above_proj_dim():
    # D == proj_dim: no projection, and the first centre is the generator's first draw.
    seed, proj_dim = 4, 8
    feats = random_features(200, proj_dim, seed=10)
    first = int(torch.randint(200, (1,), generator=torch.Generator().manual_seed(seed)))
    got = kcenter_greedy(feats, 40, seed=seed, proj_dim=proj_dim)
    assert got.tolist() == reference_kcenter(feats.double().numpy(), first, 40)
    # D == proj_dim + 1: projected, and the first centre is drawn after the projection matrix.
    wide = random_features(200, proj_dim + 1, seed=10)
    generator = torch.Generator().manual_seed(seed)
    proj = torch.randn(proj_dim + 1, proj_dim, generator=generator) / math.sqrt(proj_dim)
    first = int(torch.randint(200, (1,), generator=generator))
    got = kcenter_greedy(wide, 40, seed=seed, proj_dim=proj_dim)
    assert got.tolist() == reference_kcenter((wide.double() @ proj.double()).numpy(), first, 40)


def test_selection_is_nested():
    feats = random_features(150, 12, seed=3)
    full = kcenter_greedy(feats, 60, seed=0, proj_dim=6)
    for m in (1, 2, 17, 59):
        assert kcenter_greedy(feats, m, seed=0, proj_dim=6).tolist() == full[:m].tolist()


def test_selecting_everything_returns_a_permutation_with_the_same_start():
    feats = random_features(50, 5, seed=4)
    some = kcenter_greedy(feats, 10, seed=2)
    everything = kcenter_greedy(feats, 50, seed=2)
    more = kcenter_greedy(feats, 500, seed=2)
    assert sorted(everything.tolist()) == list(range(50))
    assert more.tolist() == everything.tolist()
    assert everything[:10].tolist() == some.tolist()


def test_deterministic_in_seed():
    feats = random_features(120, 20, seed=5)
    a = kcenter_greedy(feats, 25, seed=11, proj_dim=8)
    b = kcenter_greedy(feats, 25, seed=11, proj_dim=8)
    assert a.tolist() == b.tolist()
    firsts = {int(kcenter_greedy(feats, 1, seed=s, proj_dim=8)) for s in range(8)}
    assert len(firsts) > 1


def test_input_is_not_modified_and_fp16_is_accepted():
    feats = random_features(80, 16, seed=6)
    half = feats.to(torch.float16)
    before = half.clone()
    got = kcenter_greedy(half, 20, seed=0, proj_dim=4)
    assert torch.equal(half, before)
    # fp16 input is only a storage format: it selects like the same values held in fp32.
    assert got.tolist() == kcenter_greedy(half.float(), 20, seed=0, proj_dim=4).tolist()
    # No projection and fp32 input: the caller's tensor must not be centred in place.
    full = feats.clone()
    kcenter_greedy(full, 20, seed=0, proj_dim=16)
    assert torch.equal(full, feats)


def test_exact_duplicates_are_never_selected_twice():
    feats = torch.ones(12, 3)
    got = kcenter_greedy(feats, 12, seed=0)
    assert sorted(got.tolist()) == list(range(12))
    # Two distinct values: the second pick must come from the other group.
    feats = torch.cat([torch.zeros(6, 3), torch.ones(6, 3)])
    got = kcenter_greedy(feats, 12, seed=0)
    assert sorted(got.tolist()) == list(range(12))
    assert (got[0] < 6) != (got[1] < 6)


def test_covers_well_separated_clusters():
    centres = torch.tensor([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0], [50.0, 300.0]])
    labels = torch.arange(200) % 5
    feats = centres[labels] + 0.1 * random_features(200, 2, seed=7)
    got = kcenter_greedy(feats, 5, seed=0)
    assert sorted(labels[got].tolist()) == [0, 1, 2, 3, 4]


def test_large_offset_does_not_break_the_expanded_distance():
    # Far from the origin the naive ||a||^2 + ||b||^2 - 2ab loses all precision in fp32.
    feats = random_features(100, 6, seed=8)
    shifted = feats + 1000.0
    first = int(torch.randint(100, (1,), generator=torch.Generator().manual_seed(0)))
    expected = reference_kcenter(feats.double().numpy(), first, 30)
    assert kcenter_greedy(shifted, 30, seed=0).tolist() == expected


def test_projection_is_chunked(monkeypatch):
    feats = random_features(90, 24, seed=9)
    whole = kcenter_greedy(feats, 20, seed=0, proj_dim=8)
    monkeypatch.setattr(coreset, "_PROJECT_CHUNK", 7)
    assert kcenter_greedy(feats, 20, seed=0, proj_dim=8).tolist() == whole.tolist()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_features_are_rejected(bad):
    # Without the check a NaN row makes the loop return the same index over and over.
    feats = random_features(100, 8, seed=12)
    feats[7, 3] = bad
    for n_select in (1, 10, 100):
        with pytest.raises(ValueError, match="non-finite"):
            kcenter_greedy(feats, n_select, seed=0)
    # The same behind the random projection, with fp16 storage.
    wide = random_features(100, 16, seed=12).to(torch.float16)
    wide[99, 0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        kcenter_greedy(wide, 10, seed=0, proj_dim=4)


def test_large_finite_features_are_accepted():
    # The largest fp16 value in every entry is still far from overflowing the fp32 squared norms.
    feats = torch.full((20, 1536), 65504.0, dtype=torch.float16)
    feats[::2] = -65504.0
    got = kcenter_greedy(feats, 20, seed=0)
    assert sorted(got.tolist()) == list(range(20))
    assert (got[0] % 2) != (got[1] % 2)


@pytest.mark.skipif(
    os.environ.get("DEFECT_INSPECT_GPU_TESTS") != "1" or not torch.cuda.is_available(),
    reason="opt-in GPU test: set DEFECT_INSPECT_GPU_TESTS=1 (the default test run must not touch the GPU)",
)
def test_selection_loop_does_not_synchronise_on_cuda():
    # Reading a value back from the device in every iteration (`.item()`, indexing by a 0-dim
    # tensor) costs more than the arithmetic of the iteration itself. Sync debug mode turns any
    # such read into an error.
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


def test_single_point_and_invalid_arguments():
    assert kcenter_greedy(torch.zeros(1, 4), 3).tolist() == [0]
    with pytest.raises(ValueError):
        kcenter_greedy(torch.zeros(0, 4), 1)
    with pytest.raises(ValueError):
        kcenter_greedy(torch.zeros(4), 1)
    with pytest.raises(ValueError):
        kcenter_greedy(torch.zeros(4, 4), 0)
