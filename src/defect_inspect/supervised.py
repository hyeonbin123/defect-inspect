"""Supervised patch head on frozen patch features: targets, training and prediction.

The rules (head, loss, schedule, augmentation, model selection) are fixed in docs/experiments.md,
section "2단계 / 지도 학습".
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from defect_inspect.metrics import auroc
from defect_inspect.patchcore import ScoreResult, collect_features

GRID = 32
# Batch size of the validation passes inside `train_head` (the default of `predict`).
_VAL_BATCH = 64


class SegHead(torch.nn.Module):
    """Conv3x3-BN-ReLU twice, then a 1x1 convolution: one defect logit per patch."""

    def __init__(self, dim: int = 384, hidden: int = 256) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(dim, hidden, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Channels-last features [B, H, W, D] -> logits [B, H, W]."""
        if feats.ndim != 4:
            raise ValueError(f"expected features [B, H, W, D], got {tuple(feats.shape)}")
        return self.net(feats.permute(0, 3, 1, 2)).squeeze(1)


def patch_targets(masks: np.ndarray, grid: int = GRID) -> np.ndarray:
    """{0, 1} masks [N, S, S] -> bool [N, grid, grid]: a patch is a defect if any of its pixels is."""
    masks = np.asarray(masks)
    if masks.ndim != 3 or masks.shape[1] != masks.shape[2]:
        raise ValueError(f"expected square masks [N, S, S], got {masks.shape}")
    n, side = masks.shape[0], masks.shape[1]
    if grid < 1 or side % grid != 0:
        raise ValueError(f"grid {grid} does not divide the mask side {side}")
    block = side // grid
    defect = masks if masks.dtype == np.bool_ else masks != 0
    return defect.reshape(n, grid, block, grid, block).any(axis=(2, 4))


def pos_weight(targets: np.ndarray, max_pos_weight: float) -> float:
    """min(negative patches / positive patches, max_pos_weight); 1.0 when there is no positive patch."""
    targets = np.asarray(targets)
    n_pos = int(np.count_nonzero(targets))
    if n_pos == 0:
        return 1.0
    return float(min((targets.size - n_pos) / n_pos, max_pos_weight))


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    eval_every: int = 5
    max_pos_weight: float = 100.0
    seed: int = 0


@dataclass
class TrainResult:
    head: SegHead  # loaded with the best state, in eval mode
    best_epoch: int  # 1-based
    best_val_auroc: float
    history: list[dict]  # one {"epoch", "loss", "val_auroc"} per epoch; val_auroc is None when not evaluated


def steps_per_epoch(n_images: int, batch_size: int) -> int:
    """Batches in one epoch: the last partial batch is kept unless it has a single image."""
    full, partial = divmod(n_images, batch_size)
    return full + (1 if partial > 1 else 0)


def _flipped_batch(
    feats: torch.Tensor, targets: torch.Tensor, index: torch.Tensor, flips: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows `index` of `feats` [N, H, W, D] and `targets` [N, H, W], flipped per image.

    `flips` is bool [B, 2]: column 0 flips left-right (the W axis), column 1 up-down (the H axis).
    One gather per tensor, with the same indices, so features and targets always move together.
    """
    h, w = feats.shape[1], feats.shape[2]
    rows = torch.arange(h).expand(len(index), h)
    cols = torch.arange(w).expand(len(index), w)
    rows = torch.where(flips[:, 1:2], h - 1 - rows, rows)
    cols = torch.where(flips[:, 0:1], w - 1 - cols, cols)
    pick = (index[:, None, None], rows[:, :, None], cols[:, None, :])
    return feats[pick], targets[pick]


def _epoch_batches(
    feats: torch.Tensor, targets: torch.Tensor, batch_size: int, generator: torch.Generator
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """One epoch: a permutation of the images cut into batches, each image flipped at random.

    Both the order and the flips (probability 0.5 each, independent) are drawn from `generator`.
    """
    n = feats.shape[0]
    order = torch.randperm(n, generator=generator)
    for start in range(0, n, batch_size):
        index = order[start : start + batch_size]
        if len(index) == 1 and batch_size > 1:
            break  # a last batch of one image is dropped
        flips = torch.rand((len(index), 2), generator=generator) < 0.5
        yield _flipped_batch(feats, targets, index, flips)


@torch.no_grad()
def _max_logits(head: SegHead, feats: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    """Image scores (largest patch logit, float32) with the head in eval mode."""
    head.eval()
    scores = np.empty(feats.shape[0], dtype=np.float32)
    for start in range(0, feats.shape[0], batch_size):
        batch = feats[start : start + batch_size].to(device).to(torch.float32)
        scores[start : start + len(batch)] = head(batch).flatten(1).amax(dim=1).cpu().numpy()
    return scores


def _check_training_inputs(
    train_feats: torch.Tensor,
    train_targets: np.ndarray,
    val_feats: torch.Tensor,
    val_labels: np.ndarray,
    cfg: TrainConfig,
) -> None:
    if train_feats.ndim != 4 or val_feats.ndim != 4:
        raise ValueError(
            f"expected features [N, H, W, D], got {tuple(train_feats.shape)} and {tuple(val_feats.shape)}"
        )
    if tuple(train_targets.shape) != tuple(train_feats.shape[:3]):
        raise ValueError(
            f"targets {train_targets.shape} do not match the features {tuple(train_feats.shape)}"
        )
    if val_feats.shape[1:] != train_feats.shape[1:]:
        raise ValueError(
            f"validation features {tuple(val_feats.shape)} do not match {tuple(train_feats.shape)}"
        )
    if val_labels.shape != (val_feats.shape[0],):
        raise ValueError(f"expected {val_feats.shape[0]} validation labels, got shape {val_labels.shape}")
    if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.eval_every < 1:
        raise ValueError(f"epochs, batch_size and eval_every must be at least 1, got {cfg}")


def train_head(
    train_feats: torch.Tensor,
    train_targets: np.ndarray,
    val_feats: torch.Tensor,
    val_labels: np.ndarray,
    cfg: TrainConfig,
    *,
    device: str | torch.device = "cuda",
) -> TrainResult:
    """Train a `SegHead` on patch features and keep the epoch with the best validation image AUROC.

    `train_feats` [N, H, W, D] and `val_feats` [M, H, W, D] stay on the CPU (fp16) and go to the device
    one batch at a time as fp32. `train_targets` is bool [N, H, W], `val_labels` is 0/1 [M]. The
    validation AUROC is measured every `cfg.eval_every` epochs and at the last epoch; a later epoch
    replaces the best one only when it is strictly better.

    Reproducibility: order, flips and initial weights depend only on `cfg.seed`. On the CPU the same
    inputs and seed give bit-identical results for the same number of threads
    (`torch.get_num_threads()`); another thread count, or a GPU, changes the arithmetic at float
    rounding level, and training amplifies that (a validation AUROC can move in the third decimal, and
    with it the selected epoch). Results are then comparable, not identical.
    """
    device = torch.device(device)
    train_targets = np.asarray(train_targets)
    val_labels = np.asarray(val_labels)
    _check_training_inputs(train_feats, train_targets, val_feats, val_labels, cfg)
    is_defect = val_labels != 0
    if is_defect.all() or not is_defect.any():
        raise ValueError("the validation set needs both normal and defect images")
    n_steps = cfg.epochs * steps_per_epoch(train_feats.shape[0], cfg.batch_size)
    if n_steps == 0:
        raise ValueError("training needs at least two images")

    weight = torch.tensor(pos_weight(train_targets, cfg.max_pos_weight), dtype=torch.float32, device=device)
    targets = torch.from_numpy(np.ascontiguousarray(train_targets != 0))
    # Order and flips are drawn on the CPU, so they are the same on every device.
    generator = torch.Generator().manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)  # weight initialisation
    head = SegHead(dim=int(train_feats.shape[-1])).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=0.0)

    best_epoch, best_auroc, best_state = 0, float("-inf"), None
    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        head.train()
        loss_sum, seen = 0.0, 0
        for feats, target in _epoch_batches(train_feats, targets, cfg.batch_size, generator):
            # Moved as fp16 and cast on the device: half the bytes cross over.
            feats = feats.to(device).to(torch.float32)
            target = target.to(device).to(torch.float32)
            loss = F.binary_cross_entropy_with_logits(head(feats), target, pos_weight=weight)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss in epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(feats)
            seen += len(feats)
        entry: dict = {"epoch": epoch, "loss": loss_sum / seen, "val_auroc": None}
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            scores = _max_logits(head, val_feats, _VAL_BATCH, device)
            value = float(auroc(scores[~is_defect], scores[is_defect]))
            entry["val_auroc"] = value
            if value >= best_auroc:  # ties keep the later epoch (rule change 4)
                best_epoch, best_auroc = epoch, value
                best_state = {name: tensor.detach().clone() for name, tensor in head.state_dict().items()}
        history.append(entry)

    if best_state is None:
        raise FloatingPointError("the validation AUROC was never a number")
    head.load_state_dict(best_state)
    head.eval()
    return TrainResult(head=head, best_epoch=best_epoch, best_val_auroc=best_auroc, history=history)


@torch.no_grad()
def predict(
    head: SegHead,
    feats: torch.Tensor,
    *,
    batch_size: int = 64,
    device: str | torch.device = "cuda",
    map_size: int = 256,
) -> ScoreResult:
    """Score patch features [N, H, W, D] with a trained head (switched to eval mode).

    Image score = largest patch logit (float32). Map = the logits upsampled bilinearly
    (align_corners=False) to `map_size`, float16. Logits rather than probabilities: a float16 probability
    saturates at 1.0 above a logit of about 8.3 and would tie every confident pixel (rule change 4).
    """
    if feats.ndim != 4:
        raise ValueError(f"expected features [N, H, W, D], got {tuple(feats.shape)}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    device = torch.device(device)
    n = feats.shape[0]
    image_scores = np.empty(n, dtype=np.float32)
    maps = np.empty((n, map_size, map_size), dtype=np.float16)
    if n == 0:
        return ScoreResult(image_scores, maps)
    head = head.to(device).eval()
    for start in range(0, n, batch_size):
        batch = feats[start : start + batch_size].to(device).to(torch.float32)
        logits = head(batch)
        image_scores[start : start + len(batch)] = logits.flatten(1).amax(dim=1).cpu().numpy()
        batch_maps = F.interpolate(
            logits.unsqueeze(1),
            size=(map_size, map_size),
            mode="bilinear",
            align_corners=False,
        )
        maps[start : start + len(batch)] = batch_maps[:, 0].to(torch.float16).cpu().numpy()
    if not np.isfinite(image_scores).all():
        raise FloatingPointError("the head produced non-finite logits")
    return ScoreResult(image_scores, maps)


def extract(
    extractor: torch.nn.Module,
    images: np.ndarray,
    *,
    batch_size: int = 32,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """uint8 images [N, S, S, 3] -> CPU fp16 patch features [N, H, W, D]."""
    feats, (grid_h, grid_w) = collect_features(extractor, images, batch_size=batch_size, device=device)
    return feats.view(len(images), grid_h, grid_w, feats.shape[1])
