"""Changes to anomalib's DinomalyModel for the stage 6 models, and the graph that is exported for the CPU.

Needs torch and anomalib (the `train` and `dinomaly` groups). Every change keeps the model's outputs:

- `fix_encoder` keeps only the encoder blocks the model reads (the last target layer is block 9; anomalib
  runs all 12 blocks of the ViT) and resamples the position embedding once for a fixed input size, so
  that no interpolation runs per image. The same resampling function runs as in the dynamic path.
- `ScaledLinearAttention` divides the keys by the number of tokens. The output of linear attention is
  `(q @ (k^T v)) / (q . sum(k))`: the factor cancels, but the sums that overflowed fp16 in stage 2
  (`work/dinomaly_nan_probe_cpu.json`: q.kv up to 89,101) shrink by that factor.
- `Fp32LinearAttention` runs the attention in fp32 with autocast switched off (the fallback of the
  fp16 gate in docs/experiments.md, stage 6).
- `ServingGraph` is the inference path of `DinomalyModel.forward` plus the map resize of
  `run_dinomaly.predict`: image in, (image score, 256x256 map) out.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from anomalib.models.image.dinomaly.components import LinearAttention
from anomalib.models.image.dinomaly.torch_model import DEFAULT_MAX_RATIO, DEFAULT_RESIZE_SIZE

ATTENTION_MODES = ("original", "scaled", "fp32")
MAP_SIZE = 256


class ScaledLinearAttention(LinearAttention):
    """anomalib's LinearAttention with the keys divided by the number of tokens.

    The output is the same up to rounding; the key-value product it returns as its second value (which
    the decoder block does not use) is the original one divided by the number of tokens.
    """

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The body of anomalib 2.6.2's LinearAttention.forward (Apache-2.0) with `k / seq_len`.
        batch_size, seq_len, embed_dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, seq_len, 3, self.num_heads, embed_dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.elu(q) + 1.0
        k = (F.elu(k) + 1.0) / seq_len
        kv = torch.matmul(k.transpose(-2, -1), v)
        k_sum = k.sum(dim=-2, keepdim=True)
        z = 1.0 / torch.sum(q * k_sum, dim=-1, keepdim=True)
        x = torch.matmul(q, kv) * z
        x = x.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, kv


class Fp32LinearAttention(LinearAttention):
    """anomalib's LinearAttention computed in fp32, outside any autocast region."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=x.device.type, enabled=False):
            return super().forward(x.float())


_ATTENTION = {"original": LinearAttention, "scaled": ScaledLinearAttention, "fp32": Fp32LinearAttention}


def set_attention(model: torch.nn.Module, mode: str) -> None:
    """Switch the attention of every decoder block to `mode` (the parameters stay the same objects)."""
    if mode not in _ATTENTION:
        raise ValueError(f"attention must be one of {ATTENTION_MODES}, got {mode!r}")
    for block in model.decoder:
        if not isinstance(block.attn, LinearAttention):
            raise TypeError(f"decoder attention is {type(block.attn).__name__}, not a LinearAttention")
        block.attn.__class__ = _ATTENTION[mode]


def attention_mode(model: torch.nn.Module) -> str:
    """The mode of the decoder's attention (all blocks must agree)."""
    modes = {next(k for k, cls in _ATTENTION.items() if type(b.attn) is cls) for b in model.decoder}
    if len(modes) != 1:
        raise ValueError(f"decoder blocks use different attention modes: {sorted(modes)}")
    return modes.pop()


def _size_guard(img_size: int):
    def check(module: torch.nn.Module, args: tuple) -> None:
        size = tuple(args[0].shape[-2:])
        if size != (img_size, img_size):
            raise ValueError(
                f"this encoder is fixed to {img_size}x{img_size} inputs, got {size[0]}x{size[1]}"
            )

    return check


def fix_encoder(model: torch.nn.Module, *, img_size: int | None = None, blocks: int | None = None) -> None:
    """Keep the first `blocks` encoder blocks and fix the input size to `img_size` (both optional).

    Fixing the size resamples the position embedding once with the function the dynamic path calls on
    every image (anomalib switches its antialiasing off), so the outputs do not change. A fixed encoder
    refuses other input sizes instead of resampling the already resampled embedding.
    """
    vit = model.encoder.feature_extractor
    if blocks is not None:
        needed = max(model.target_layers) + 1
        if not needed <= blocks <= len(vit.blocks):
            raise ValueError(f"blocks must be between {needed} and {len(vit.blocks)}, got {blocks}")
        vit.blocks = vit.blocks[:blocks]
    if img_size is not None:
        patch = int(model.encoder.patch_size)
        if img_size <= 0 or img_size % patch != 0:
            raise ValueError(f"img_size must be a positive multiple of {patch}, got {img_size}")
        vit.set_input_size(img_size=(img_size, img_size))
        model.encoder.register_forward_pre_hook(_size_guard(img_size))
        model.fixed_img_size = img_size


def encoder_blocks(model: torch.nn.Module) -> int:
    return len(model.encoder.feature_extractor.blocks)


class ServingGraph(torch.nn.Module):
    """Normalised image [B, 3, S, S] -> (image score [B], map [B, 256, 256]) of a Dinomaly model.

    The score is that of `DinomalyModel.forward` in eval mode: the mean of the top 1% of the blurred
    256x256 map (top-k instead of a full sort; the same values are averaged). The map is the unblurred
    map resized to 256x256, which is what `run_dinomaly.predict` stores for the pixel metrics.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        if DEFAULT_RESIZE_SIZE != MAP_SIZE:
            raise RuntimeError(f"anomalib scores on a {DEFAULT_RESIZE_SIZE} map, this graph on {MAP_SIZE}")
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        en, de = self.model.get_encoder_decoder_outputs(x)
        amap, _ = self.model.calculate_anomaly_maps(en, de, out_size=(x.shape[2], x.shape[3]))
        amap = F.interpolate(amap, size=(MAP_SIZE, MAP_SIZE), mode="bilinear", align_corners=False)
        flat = self.model.gaussian_blur(amap).flatten(1)
        k = int(flat.shape[1] * DEFAULT_MAX_RATIO)
        score = torch.topk(flat, k, dim=1).values.mean(dim=1)
        return score, amap[:, 0]
