import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch.nn.functional as F  # noqa: E402

from defect_inspect import supervised  # noqa: E402
from defect_inspect.backbones import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from defect_inspect.metrics import auroc  # noqa: E402
from defect_inspect.patchcore import ScoreResult, collect_features  # noqa: E402
from defect_inspect.supervised import (  # noqa: E402
    GRID,
    SegHead,
    TrainConfig,
    extract,
    patch_targets,
    pos_weight,
    predict,
    steps_per_epoch,
    train_head,
)

G, DIM = 8, 6  # patch grid and feature dimension of the synthetic data


def planted(n_normal: int, n_defect: int, seed: int, strength: float = 6.0):
    """Gaussian noise features; defect images get `strength` added to channel 0 in one 2x2 block.

    Returns (fp16 features [N, G, G, DIM], bool targets [N, G, G], labels [N]), normals first.
    """
    n = n_normal + n_defect
    feats = torch.randn(n, G, G, DIM, generator=torch.Generator().manual_seed(seed))
    targets = np.zeros((n, G, G), dtype=bool)
    rng = np.random.default_rng(seed)
    for i in range(n_normal, n):
        row, col = rng.integers(0, G - 1, size=2)
        targets[i, row : row + 2, col : col + 2] = True
    feats[..., 0] += strength * torch.from_numpy(targets)
    labels = np.array([0] * n_normal + [1] * n_defect, dtype=np.int8)
    return feats.to(torch.float16), targets, labels


FAST = TrainConfig(epochs=10, batch_size=16, eval_every=5, seed=0)


def fit(cfg: TrainConfig = FAST, n_normal: int = 24, n_defect: int = 8):
    train_feats, train_targets, _ = planted(n_normal, n_defect, seed=1)
    val_feats, _, val_labels = planted(12, 6, seed=2)
    result = train_head(train_feats, train_targets, val_feats, val_labels, cfg, device="cpu")
    return result, val_feats, val_labels


def pass_through_head(dim: int, bias: float = -1.0) -> SegHead:
    """Eval-mode head with hand-set weights: logit = relu(channel 0) / (1 + eps) + bias."""
    head = SegHead(dim=dim, hidden=4)
    with torch.no_grad():
        for param in head.parameters():
            param.zero_()
        head.net[0].weight[0, 0, 1, 1] = 1.0
        head.net[3].weight[0, 0, 1, 1] = 1.0
        head.net[1].weight.fill_(1.0)
        head.net[4].weight.fill_(1.0)
        head.net[6].weight[0, 0, 0, 0] = 1.0
        head.net[6].bias[0] = bias
    return head.eval()


def pass_through_logits(feats: torch.Tensor, bias: float = -1.0) -> np.ndarray:
    # Two eval-mode BatchNorm layers with fresh statistics each divide by sqrt(1 + 1e-5).
    return np.maximum(feats[..., 0].double().numpy(), 0.0) / (1.0 + 1e-5) + bias


# --- head ---------------------------------------------------------------------------------------


def test_seg_head_layers():
    assert GRID == 32
    head = SegHead()
    kinds = [type(layer).__name__ for layer in head.net]
    assert kinds == ["Conv2d", "BatchNorm2d", "ReLU", "Conv2d", "BatchNorm2d", "ReLU", "Conv2d"]
    first, second, last = head.net[0], head.net[3], head.net[6]
    assert (first.in_channels, first.out_channels, first.kernel_size, first.padding) == (
        384,
        256,
        (3, 3),
        (1, 1),
    )
    assert (second.in_channels, second.out_channels, second.kernel_size, second.padding) == (
        256,
        256,
        (3, 3),
        (1, 1),
    )
    assert (last.in_channels, last.out_channels, last.kernel_size) == (256, 1, (1, 1))
    assert head.net[1].num_features == head.net[4].num_features == 256


def test_seg_head_takes_channels_last_and_keeps_the_grid():
    torch.manual_seed(0)
    head = SegHead(dim=5, hidden=7).eval()
    feats = torch.randn(2, 3, 6, 5)  # H != W, so a swapped axis would show
    with torch.no_grad():
        logits = head(feats)
        assert logits.shape == (2, 3, 6)
        assert torch.equal(logits, head.net(feats.permute(0, 3, 1, 2))[:, 0])
        # Two 3x3 convolutions: a patch only influences logits at most two cells away.
        changed = feats.clone()
        changed[0, 1, 5] += 3.0
        diff = (head(changed) - logits).abs()
    assert float(diff[0, 1, 5]) > 0
    assert float(diff[0, :, :3].max()) == 0.0
    assert float(diff[1].max()) == 0.0
    with pytest.raises(ValueError):
        head(torch.zeros(3, 6, 5))


def test_pass_through_head_is_what_the_tests_assume():
    feats = torch.randn(3, 4, 4, 2, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        logits = pass_through_head(2)(feats)
    np.testing.assert_allclose(logits.numpy(), pass_through_logits(feats), rtol=1e-5, atol=1e-6)


# --- targets ------------------------------------------------------------------------------------


def test_patch_targets_is_a_max_pool():
    masks = np.zeros((3, 256, 256), dtype=np.uint8)
    masks[0, 0, 0] = 1  # a single pixel is enough for its patch
    masks[1, 7, 8] = 1  # row 7 is still patch row 0, column 8 is patch column 1
    masks[1, 255, 255] = 1
    masks[2, 100:120, 30:41] = 1  # rows 96..127 -> patches 12..14, columns 24..47 -> patches 3..5
    targets = patch_targets(masks)
    assert targets.shape == (3, 32, 32) and targets.dtype == np.bool_
    assert np.argwhere(targets[0]).tolist() == [[0, 0]]
    assert np.argwhere(targets[1]).tolist() == [[0, 1], [31, 31]]
    expected = np.zeros((32, 32), dtype=bool)
    expected[12:15, 3:6] = True
    np.testing.assert_array_equal(targets[2], expected)
    pooled = F.max_pool2d(torch.from_numpy(masks).float()[:, None], kernel_size=8)[:, 0].numpy() > 0
    np.testing.assert_array_equal(targets, pooled)


def test_patch_targets_grid_dtypes_and_errors():
    rng = np.random.default_rng(0)
    masks = (rng.random((4, 256, 256)) < 0.001).astype(np.uint8)
    for grid in (4, 16, 64, 256):
        pooled = F.max_pool2d(torch.from_numpy(masks).float()[:, None], kernel_size=256 // grid)[:, 0]
        np.testing.assert_array_equal(patch_targets(masks, grid), pooled.numpy() > 0)
    np.testing.assert_array_equal(patch_targets(masks.astype(bool)), patch_targets(masks))
    # Any value above zero counts as a defect.
    np.testing.assert_array_equal(patch_targets(masks * 7), patch_targets(masks))
    assert patch_targets(np.zeros((0, 256, 256), dtype=np.uint8)).shape == (0, 32, 32)
    with pytest.raises(ValueError):
        patch_targets(np.zeros((256, 256), dtype=np.uint8))
    with pytest.raises(ValueError):
        patch_targets(np.zeros((1, 256, 128), dtype=np.uint8))
    with pytest.raises(ValueError):
        patch_targets(np.zeros((1, 256, 256), dtype=np.uint8), grid=30)


def test_pos_weight_is_the_capped_ratio_over_the_whole_set():
    targets = np.zeros((4, 8, 8), dtype=bool)
    assert pos_weight(targets, 100.0) == 1.0  # no positive patch
    targets[0, :2, :4] = True  # 8 positive, 248 negative
    assert pos_weight(targets, 100.0) == pytest.approx(248 / 8)
    assert pos_weight(targets, 10.0) == 10.0
    targets[:] = True
    assert pos_weight(targets, 100.0) == 0.0


# --- batches and flips --------------------------------------------------------------------------


def test_steps_per_epoch_drops_a_single_image_batch_only():
    assert steps_per_epoch(64, 32) == 2
    assert steps_per_epoch(65, 32) == 2  # the 65th image alone is dropped
    assert steps_per_epoch(66, 32) == 3
    assert steps_per_epoch(20, 32) == 1
    assert steps_per_epoch(1, 32) == 0
    assert steps_per_epoch(3, 1) == 3  # batch size 1 has no partial batch


def coordinate_features(n: int, h: int, w: int):
    """fp16 features whose channels are (image id, row, column); the target marks cell (0, 0)."""
    feats = torch.zeros(n, h, w, 3)
    feats[..., 0] = torch.arange(n).view(n, 1, 1)
    feats[..., 1] = torch.arange(h).view(1, h, 1)
    feats[..., 2] = torch.arange(w).view(1, 1, w)
    targets = torch.zeros(n, h, w, dtype=torch.bool)
    targets[:, 0, 0] = True
    return feats.to(torch.float16), targets


def test_flipped_batch_flips_the_named_axis_of_both_tensors():
    feats, targets = coordinate_features(4, 2, 3)
    index = torch.tensor([3, 0, 2, 1])
    flips = torch.tensor([[False, False], [True, False], [False, True], [True, True]])
    out, out_targets = supervised._flipped_batch(feats, targets, index, flips)
    assert out.shape == (4, 2, 3, 3) and out.dtype == torch.float16
    assert out_targets.shape == (4, 2, 3) and out_targets.dtype == torch.bool
    assert out[:, 0, 0, 0].tolist() == [3, 0, 2, 1]
    # Column 0 of `flips` mirrors left-right (W), column 1 up-down (H).
    assert torch.equal(out[0], feats[3])
    assert torch.equal(out[1], feats[0].flip(1))
    assert torch.equal(out[2], feats[2].flip(0))
    assert torch.equal(out[3], feats[1].flip(0).flip(1))
    assert np.argwhere(out_targets.numpy()).tolist() == [[0, 0, 0], [1, 0, 2], [2, 1, 0], [3, 1, 2]]
    # The inputs are left alone.
    again, _ = coordinate_features(4, 2, 3)
    assert torch.equal(feats, again)


def test_epoch_batches_keep_features_and_targets_together():
    n, h, w = 401, 2, 3
    feats, targets = coordinate_features(n, h, w)
    generator = torch.Generator().manual_seed(0)
    batches = list(supervised._epoch_batches(feats, targets, 16, generator))
    assert [len(f) for f, _ in batches] == [16] * 25  # 401 = 25 * 16 + 1: the single image is dropped
    out = torch.cat([f for f, _ in batches]).float()
    out_targets = torch.cat([t for _, t in batches])
    # Wherever the target ends up, the features of the original cell (0, 0) are there too.
    assert torch.equal(out_targets, (out[..., 1] == 0) & (out[..., 2] == 0))
    ids = out[:, 0, 0, 0].long().tolist()
    assert len(set(ids)) == 400 and ids != sorted(ids)
    assert bool((out[..., 0] == out[:, :1, :1, 0]).all())  # no image was mixed with another
    h_flip = (out[:, 0, 0, 2] == w - 1).numpy()
    v_flip = (out[:, 0, 0, 1] == h - 1).numpy()
    # Probability 0.5 each, drawn independently: all four combinations occur about equally often.
    for count in (h_flip.sum(), v_flip.sum()):
        assert 160 <= count <= 240
    for a in (False, True):
        for b in (False, True):
            assert 60 <= np.count_nonzero((h_flip == a) & (v_flip == b)) <= 140

    # Same generator state, same epoch; the next epoch of the same generator differs.
    repeat = list(supervised._epoch_batches(feats, targets, 16, torch.Generator().manual_seed(0)))
    assert len(repeat) == len(batches)
    for (feats_a, targets_a), (feats_b, targets_b) in zip(batches, repeat, strict=True):
        assert torch.equal(feats_a, feats_b) and torch.equal(targets_a, targets_b)
    second = list(supervised._epoch_batches(feats, targets, 16, generator))
    assert not torch.equal(second[0][0], batches[0][0])


@pytest.mark.parametrize(
    ("n", "batch_size", "sizes"), [(34, 16, [16, 16, 2]), (5, 8, [5]), (3, 1, [1, 1, 1])]
)
def test_epoch_batches_keep_the_last_partial_batch(n, batch_size, sizes):
    feats, targets = coordinate_features(n, 2, 2)
    batches = list(supervised._epoch_batches(feats, targets, batch_size, torch.Generator().manual_seed(1)))
    assert [len(f) for f, _ in batches] == sizes
    assert len(batches) == steps_per_epoch(n, batch_size)
    ids = torch.cat([f for f, _ in batches])[:, 0, 0, 0].long().tolist()
    assert sorted(ids) == list(range(n))


# --- training -----------------------------------------------------------------------------------


def test_order_and_flips_come_from_one_generator_seeded_with_the_config(monkeypatch):
    seen = []
    real = supervised._epoch_batches

    def spy(feats, targets, batch_size, generator):
        seen.append((batch_size, generator.initial_seed(), generator.device.type, id(generator)))
        return real(feats, targets, batch_size, generator)

    monkeypatch.setattr(supervised, "_epoch_batches", spy)
    fit(TrainConfig(epochs=3, batch_size=16, eval_every=5, seed=7))
    assert [entry[:3] for entry in seen] == [(16, 7, "cpu")] * 3
    assert len({entry[3] for entry in seen}) == 1  # one generator carried through the epochs


def test_training_is_deterministic_for_a_seed():
    first, _, _ = fit()
    torch.randn(5)  # the state of the global generator between two runs does not matter
    second, _, _ = fit()
    assert first.history == second.history
    assert (first.best_epoch, first.best_val_auroc) == (second.best_epoch, second.best_val_auroc)
    state_a, state_b = first.head.state_dict(), second.head.state_dict()
    assert all(torch.equal(state_a[name], state_b[name]) for name in state_a)
    # Another seed gives another initialisation, order and flips.
    other, _, _ = fit(TrainConfig(epochs=10, batch_size=16, eval_every=5, seed=1))
    assert not torch.equal(other.head.net[0].weight, first.head.net[0].weight)
    assert other.history != first.history


def test_head_separates_planted_defects():
    train_feats, train_targets, _ = planted(24, 8, seed=1)
    before = train_feats.clone()
    val_feats, _, val_labels = planted(12, 6, seed=2)
    result = train_head(train_feats, train_targets, val_feats, val_labels, FAST, device="cpu")
    assert isinstance(result.head, SegHead) and not result.head.training
    assert result.head.net[0].in_channels == DIM and result.head.net[0].out_channels == 256
    assert result.best_val_auroc >= 0.98
    assert result.history[-1]["loss"] < 0.5 * result.history[0]["loss"]
    # The stored features (fp16, CPU) are not touched by the flips or the casts.
    assert train_feats.dtype == torch.float16 and torch.equal(train_feats, before)
    # The returned head is the one that reached the reported validation AUROC.
    scores = predict(result.head, val_feats, device="cpu", map_size=G).image_scores
    assert auroc(scores[val_labels == 0], scores[val_labels == 1]) == pytest.approx(result.best_val_auroc)
    # It also separates images it has never seen.
    test_feats, _, test_labels = planted(20, 10, seed=3)
    scores = predict(result.head, test_feats, device="cpu", map_size=G).image_scores
    assert auroc(scores[test_labels == 0], scores[test_labels == 1]) >= 0.98


def test_head_is_back_in_train_mode_after_every_validation(monkeypatch):
    modes = []

    class RecordingHead(SegHead):
        def forward(self, feats):
            modes.append((torch.is_grad_enabled(), self.training))
            return super().forward(feats)

    monkeypatch.setattr(supervised, "SegHead", RecordingHead)
    rising = iter([0.6, 0.7, 0.8, 0.9])  # the last epoch is the best one
    monkeypatch.setattr(supervised, "auroc", lambda neg, pos: next(rising))
    # 32 training images in batches of 16: two steps per epoch. 18 validation images: one batch.
    result, _, _ = fit(TrainConfig(epochs=4, batch_size=16, eval_every=1))
    # Training batches (gradients on) see the head in train mode in every epoch, not only before the
    # first validation pass; validation batches (gradients off) see it in eval mode.
    assert modes == ([(True, True)] * 2 + [(False, False)]) * 4
    assert result.best_epoch == 4 and not result.head.training
    # BatchNorm statistics: updated by every training batch up to the selected epoch, by nothing else.
    for layer in (result.head.net[1], result.head.net[4]):
        assert int(layer.num_batches_tracked) == 4 * steps_per_epoch(32, 16)


def test_history_and_evaluation_epochs():
    result, _, _ = fit(TrainConfig(epochs=10, batch_size=16, eval_every=4))
    assert [h["epoch"] for h in result.history] == list(range(1, 11))
    assert all(set(h) == {"epoch", "loss", "val_auroc"} for h in result.history)
    assert all(math.isfinite(h["loss"]) and h["loss"] > 0 for h in result.history)
    # Every 4 epochs and at the last epoch.
    evaluated = [h["epoch"] for h in result.history if h["val_auroc"] is not None]
    assert evaluated == [4, 8, 10]
    assert result.best_epoch in evaluated
    best = max(h["val_auroc"] for h in result.history if h["val_auroc"] is not None)
    assert result.best_val_auroc == best
    assert result.history[result.best_epoch - 1]["val_auroc"] == best


@pytest.mark.parametrize(("n_normal", "steps"), [(24, 2), (25, 2), (26, 3)])
def test_adamw_with_cosine_annealing_over_all_steps(monkeypatch, n_normal, steps):
    created = []

    class RecordingAdamW(torch.optim.AdamW):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            self.kwargs, self.lrs = kwargs, []
            created.append(self)

        def step(self, closure=None):
            self.lrs.append(self.param_groups[0]["lr"])
            return super().step(closure)

    monkeypatch.setattr(torch.optim, "AdamW", RecordingAdamW)
    cfg = TrainConfig(epochs=3, batch_size=16, eval_every=5, lr=2e-3, weight_decay=0.05)
    fit(cfg, n_normal=n_normal, n_defect=8)  # 32, 33 or 34 images
    (optimizer,) = created
    assert optimizer.kwargs == {"lr": 2e-3, "weight_decay": 0.05}
    total = 3 * steps
    assert len(optimizer.lrs) == total
    # No warm-up: the first step uses the full rate, and the curve reaches 0 right after the last one.
    expected = [2e-3 * 0.5 * (1.0 + math.cos(math.pi * t / total)) for t in range(total)]
    np.testing.assert_allclose(optimizer.lrs, expected, rtol=1e-9)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)


def test_loss_is_weighted_bce_on_the_patch_grid(monkeypatch):
    calls = []
    real = F.binary_cross_entropy_with_logits

    def spy(logits, target, **kwargs):
        calls.append((tuple(logits.shape), target.dtype, float(kwargs["pos_weight"])))
        assert set(kwargs) == {"pos_weight"}  # mean reduction
        assert set(target.unique().tolist()) <= {0.0, 1.0}
        return real(logits, target, **kwargs)

    monkeypatch.setattr(supervised.F, "binary_cross_entropy_with_logits", spy)
    train_feats, train_targets, _ = planted(24, 8, seed=1)
    val_feats, _, val_labels = planted(12, 6, seed=2)
    n_pos = int(train_targets.sum())
    ratio = (train_targets.size - n_pos) / n_pos  # 8 images x 4 patches of 32 x 64: 63
    assert ratio == pytest.approx(63.0)
    cfg = TrainConfig(epochs=2, batch_size=16, eval_every=5)
    train_head(train_feats, train_targets, val_feats, val_labels, cfg, device="cpu")
    assert calls == [((16, G, G), torch.float32, ratio)] * 4  # the same weight in every batch

    calls.clear()
    capped = TrainConfig(epochs=1, batch_size=16, eval_every=5, max_pos_weight=20.0)
    train_head(train_feats, train_targets, val_feats, val_labels, capped, device="cpu")
    assert {c[2] for c in calls} == {20.0}

    calls.clear()
    no_defects = np.zeros_like(train_targets)
    train_head(train_feats, no_defects, val_feats, val_labels, capped, device="cpu")
    assert {c[2] for c in calls} == {1.0}


@pytest.mark.parametrize(
    ("values", "eval_every", "best_epoch"),
    [
        ([0.5, 0.9, 0.9, 0.7], 1, 3),  # a tie keeps the later epoch
        ([0.8, 0.8, 0.8, 0.8], 1, 4),
        ([0.5, 0.6, 0.7, 0.8], 1, 4),
        ([0.9, 0.8, 0.7, 0.6], 1, 1),  # a lower value never replaces
        ([0.9, 0.9], 2, 4),  # evaluated at epochs 2 and 4 only
        ([0.9, 0.95], 3, 4),  # evaluated at epoch 3 and at the last epoch
    ],
)
def test_best_epoch_rule(monkeypatch, values, eval_every, best_epoch):
    seen = []
    pending = list(values)

    def fake_auroc(neg, pos):
        seen.append(np.concatenate([neg, pos]))
        return pending.pop(0)

    monkeypatch.setattr(supervised, "auroc", fake_auroc)
    cfg = TrainConfig(epochs=4, batch_size=16, eval_every=eval_every)
    result, val_feats, val_labels = fit(cfg)
    assert not pending
    assert result.best_epoch == best_epoch
    assert result.best_val_auroc == max(values)
    evaluated = [h["epoch"] for h in result.history if h["val_auroc"] is not None]
    assert [h["val_auroc"] for h in result.history if h["val_auroc"] is not None] == values
    # The head holds the weights of the best epoch: it reproduces the scores seen at that evaluation.
    scores = predict(result.head, val_feats, device="cpu", map_size=G).image_scores
    which = evaluated.index(best_epoch)
    np.testing.assert_allclose(scores, seen[which], rtol=1e-5, atol=1e-6)
    for other, other_scores in enumerate(seen):
        if other != which:
            assert not np.allclose(scores, other_scores, rtol=1e-3, atol=1e-4)
    # Validation scores arrive as normals first, then defects.
    assert all(len(s) == len(val_labels) for s in seen)


def test_train_head_rejects_unusable_inputs():
    train_feats, train_targets, _ = planted(6, 2, seed=1)
    val_feats, _, val_labels = planted(4, 2, seed=2)
    cfg = TrainConfig(epochs=1, batch_size=4, eval_every=1)
    with pytest.raises(ValueError, match="both normal and defect"):
        train_head(train_feats, train_targets, val_feats[:4], val_labels[:4], cfg, device="cpu")
    with pytest.raises(ValueError, match="both normal and defect"):
        train_head(train_feats, train_targets, val_feats[4:], val_labels[4:], cfg, device="cpu")
    with pytest.raises(ValueError, match="at least two images"):
        train_head(train_feats[:1], train_targets[:1], val_feats, val_labels, cfg, device="cpu")
    with pytest.raises(ValueError, match="targets"):
        train_head(train_feats, train_targets[:, :4], val_feats, val_labels, cfg, device="cpu")
    with pytest.raises(ValueError, match="validation features"):
        train_head(train_feats, train_targets, val_feats[..., :3], val_labels, cfg, device="cpu")
    with pytest.raises(ValueError, match="validation labels"):
        train_head(train_feats, train_targets, val_feats, val_labels[:3], cfg, device="cpu")
    with pytest.raises(ValueError, match="at least 1"):
        train_head(train_feats, train_targets, val_feats, val_labels, TrainConfig(epochs=0), device="cpu")


def test_train_config_defaults_are_the_registered_ones():
    cfg = TrainConfig()
    assert (cfg.epochs, cfg.batch_size, cfg.lr, cfg.weight_decay) == (60, 32, 1e-3, 1e-4)
    assert (cfg.eval_every, cfg.max_pos_weight, cfg.seed) == (5, 100.0, 0)
    with pytest.raises(AttributeError):
        cfg.seed = 1


# --- prediction ---------------------------------------------------------------------------------


def test_predict_scores_and_maps_match_the_definition():
    feats = torch.randn(5, 4, 4, 3, generator=torch.Generator().manual_seed(0)).to(torch.float16)
    head = pass_through_head(3)
    logits = pass_through_logits(feats)
    result = predict(head, feats, batch_size=2, device="cpu", map_size=4)
    assert isinstance(result, ScoreResult)
    assert result.image_scores.shape == (5,) and result.image_scores.dtype == np.float32
    assert result.maps.shape == (5, 4, 4) and result.maps.dtype == np.float16
    # Image score: the largest patch logit (not a probability).
    np.testing.assert_allclose(result.image_scores, logits.reshape(5, -1).max(axis=1), rtol=1e-5)
    # Map at the grid size: the logits themselves (no sigmoid, so float16 cannot saturate at 1.0).
    np.testing.assert_allclose(result.maps.astype(np.float64), logits, rtol=2e-3, atol=2e-3)
    # Larger maps: bilinear upsampling of the logits with align_corners=False.
    big = predict(head, feats, device="cpu", map_size=16).maps
    expected = F.interpolate(
        torch.from_numpy(logits).float()[:, None], size=(16, 16), mode="bilinear", align_corners=False
    )[:, 0].numpy()
    np.testing.assert_allclose(big.astype(np.float64), expected, rtol=2e-3, atol=2e-3)
    assert predict(head, feats, device="cpu").maps.shape == (5, 256, 256)  # default map size


def test_predict_batching_empty_input_and_eval_mode():
    feats = torch.randn(7, 4, 4, 3, generator=torch.Generator().manual_seed(1)).to(torch.float16)
    torch.manual_seed(0)
    head = SegHead(dim=3, hidden=8).train()
    whole = predict(head, feats, batch_size=64, device="cpu", map_size=8)
    assert not head.training  # BatchNorm must use its running statistics
    single = predict(head, feats, batch_size=1, device="cpu", map_size=8)
    np.testing.assert_allclose(single.image_scores, whole.image_scores, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        single.maps.astype(np.float32), whole.maps.astype(np.float32), rtol=2e-3, atol=1e-6
    )
    empty = predict(head, feats[:0], device="cpu", map_size=8)
    assert empty.image_scores.shape == (0,) and empty.maps.shape == (0, 8, 8)
    with pytest.raises(ValueError):
        predict(head, feats[0], device="cpu")
    with pytest.raises(ValueError):
        predict(head, feats, batch_size=0, device="cpu")


def test_predict_raises_on_non_finite_logits():
    head = pass_through_head(3)
    feats = torch.zeros(2, 4, 4, 3)
    feats[1, 2, 2, 0] = float("inf")
    with pytest.raises(FloatingPointError):
        predict(head, feats, device="cpu", map_size=4)


def test_map_peaks_at_the_planted_patch():
    result, _, _ = fit()
    block = 8  # map pixels per patch at map_size 64
    for row, col in [(1, 5), (5, 2), (6, 6)]:
        feats = torch.randn(1, G, G, DIM, generator=torch.Generator().manual_seed(10 * row + col))
        feats[0, row : row + 2, col : col + 2, 0] += 6.0
        scored = predict(result.head, feats.to(torch.float16), device="cpu", map_size=64)
        heat = scored.maps[0].astype(np.float32)
        peak_r, peak_c = np.unravel_index(int(heat.argmax()), heat.shape)
        assert row * block <= peak_r < (row + 2) * block
        assert col * block <= peak_c < (col + 2) * block
        # Inside the planted block the logits are clearly positive; two patches away and beyond they
        # are much lower.
        inside = heat[row * block : (row + 2) * block, col * block : (col + 2) * block]
        assert inside.mean() > 1.0
        far = np.ones_like(heat, dtype=bool)
        top, left = max(0, (row - 2) * block), max(0, (col - 2) * block)
        far[top : (row + 4) * block, left : (col + 4) * block] = False
        assert inside.mean() - heat[far].mean() > 1.0
        assert scored.image_scores[0] > 0  # the largest logit is on the defect side


# --- features -----------------------------------------------------------------------------------


class PoolExtractor(torch.nn.Module):
    """Fake extractor: average-pools the normalised image into a [B, 2, 4, 3] feature grid."""

    name = "pool"
    dim = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, (2, 4)).permute(0, 2, 3, 1)


def test_extract_returns_cpu_fp16_grids():
    images = np.random.default_rng(0).integers(0, 256, size=(5, 8, 8, 3), dtype=np.uint8)
    extractor = PoolExtractor()
    feats = extract(extractor, images, batch_size=2, device="cpu")
    assert feats.shape == (5, 2, 4, 3) and feats.dtype == torch.float16 and feats.device.type == "cpu"
    flat, grid = collect_features(extractor, images, batch_size=2, device="cpu")
    assert grid == (2, 4)
    assert torch.equal(feats.reshape(-1, 3), flat)
    # Cell (1, 2) of image 3 covers rows 4..7 and columns 4..5.
    block = images[3, 4:8, 4:6].reshape(-1, 3).astype(np.float64).mean(axis=0) / 255.0
    manual = (block - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    np.testing.assert_allclose(feats[3, 1, 2].double().numpy(), manual, atol=2e-3)
    assert extract(extractor, images[:0], device="cpu").shape == (0, 0, 0, 3)
