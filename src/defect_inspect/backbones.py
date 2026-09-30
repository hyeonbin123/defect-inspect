"""Frozen patch-feature extractors: a batch of images in, a grid of patch features out.

Every extractor has `name` and `dim`, stays in eval mode and returns `[B, H, W, dim]`. Callers run it
under `torch.no_grad()`, with `torch.autocast("cuda", dtype=torch.float16)` on CUDA and fp32 on CPU.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

from defect_inspect import paths

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DINOV2_PATCH = 14
# Short extractor name -> timm architecture.
DINOV2_ARCHS = {
    "dinov2_vits14": "vit_small_patch14_reg4_dinov2",
    "dinov2_vitb14": "vit_base_patch14_reg4_dinov2",
}


def to_tensor(images: np.ndarray, device: str | torch.device, dtype: torch.dtype) -> torch.Tensor:
    """uint8 images [B, S, S, 3] -> ImageNet-normalised [B, 3, S, S] of `dtype` on `device`."""
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError(f"expected uint8 [B, S, S, 3], got {images.dtype} {images.shape}")
    x = torch.from_numpy(np.ascontiguousarray(images)).to(device)
    # Normalise in fp32 and cast once at the end, so fp16 inputs do not lose precision on the way.
    x = x.permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
    return x.sub_(mean).div_(std).to(dtype).contiguous()


def _checkpoint_dir() -> str | None:
    """Where torchvision checkpoints are downloaded: torch's default if TORCH_HOME is set, else data/torch.

    Without TORCH_HOME torch would use the home directory; downloads of this project belong under data/.
    """
    if os.environ.get("TORCH_HOME"):
        return None
    return str(paths.DATA / "torch" / "checkpoints")


class _FrozenExtractor(torch.nn.Module):
    """Base class: parameters never train and the module cannot leave eval mode."""

    name: str
    dim: int

    def train(self, mode: bool = True) -> _FrozenExtractor:
        # BatchNorm statistics must stay fixed, so `.train()` is a no-op.
        return super().train(False)

    def _freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()


class WideResNetPatches(_FrozenExtractor):
    """torchvision WideResNet-50-2 up to layer3: layer2 and layer3 features on the layer2 grid."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        net = wide_resnet50_2(weights=None)
        if pretrained:
            weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
            state = weights.get_state_dict(progress=False, check_hash=True, model_dir=_checkpoint_dir())
            net.load_state_dict(state)
        # layer4 and the classifier are never run, so they are not kept.
        self.conv1, self.bn1, self.relu, self.maxpool = net.conv1, net.bn1, net.relu, net.maxpool
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3
        self.name = "wrn50"
        self.dim = 512 + 1024
        self._freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        f2 = self.layer2(self.layer1(x))  # [B, 512, H, W]
        f3 = self.layer3(f2)  # [B, 1024, H/2, W/2]
        f2 = F.avg_pool2d(f2, kernel_size=3, stride=1, padding=1)
        f3 = F.avg_pool2d(f3, kernel_size=3, stride=1, padding=1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1).permute(0, 2, 3, 1).contiguous()


class DinoV2Patches(_FrozenExtractor):
    """timm DINOv2 ViT: L2-normalised patch tokens of the final norm layer."""

    def __init__(
        self, arch: str = "vit_small_patch14_reg4_dinov2", img_size: int = 448, pretrained: bool = True
    ) -> None:
        super().__init__()
        if img_size <= 0 or img_size % DINOV2_PATCH != 0:
            raise ValueError(f"img_size must be a positive multiple of {DINOV2_PATCH}, got {img_size}")
        import timm  # only this extractor needs it

        # timm resamples the position embedding to the requested input size.
        self.model = timm.create_model(arch, pretrained=pretrained, num_classes=0, img_size=img_size)
        self.img_size = img_size
        self.name = next((short for short, full in DINOV2_ARCHS.items() if full == arch), arch)
        self.dim = int(self.model.embed_dim)
        self._freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2] // DINOV2_PATCH, x.shape[-1] // DINOV2_PATCH
        # Drop the class and register tokens.
        tokens = self.model.forward_features(x)[:, self.model.num_prefix_tokens :]
        if tokens.shape[1] != h * w:
            raise ValueError(f"expected {h * w} patch tokens, got {tokens.shape[1]}")
        tokens = F.normalize(tokens.float(), dim=-1)
        return tokens.reshape(x.shape[0], h, w, tokens.shape[-1])


def make_extractor(name: str, *, img_size: int, pretrained: bool = True) -> torch.nn.Module:
    """Build an extractor by short name: "wrn50", "dinov2_vits14" or "dinov2_vitb14"."""
    if name == "wrn50":
        # Fully convolutional: the grid follows the input size, so img_size is not needed here.
        return WideResNetPatches(pretrained=pretrained)
    if name in DINOV2_ARCHS:
        return DinoV2Patches(DINOV2_ARCHS[name], img_size=img_size, pretrained=pretrained)
    raise ValueError(f"unknown extractor {name!r}; expected one of {['wrn50', *DINOV2_ARCHS]}")
