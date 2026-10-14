"""고정된 패치 특징 추출기: 이미지 배치를 넣으면 패치 특징 격자가 나온다.

모든 추출기는 `name`과 `dim`을 가지고, eval 모드에 머물며, `[B, H, W, dim]`을 돌려준다. 호출하는 쪽은
`torch.no_grad()` 아래에서 돌리고, CUDA에서는 `torch.autocast("cuda", dtype=torch.float16)`, CPU에서는
fp32로 돌린다.
"""

# [파일 역할] PatchCore와 지도 학습 비교군이 쓰는 "고정된(학습하지 않는) 패치 특징 추출기"를 만드는 모듈이다.
# 이미지 배치를 넣으면 [B, H, W, dim] 패치 특징 격자를 돌려준다.
# [주요 구성] WideResNetPatches(1단계 P0: WideResNet-50-2의 layer2·layer3), DinoV2Patches(2단계 D-S: DINOv2
# ViT-S/14의 마지막 층 패치 토큰), CentredPatches(7단계 E1: 이미지별 평균을 뺀 특징), 이름으로 고르는
# make_extractor, 공용 정규화 to_tensor.
# [설계 이유] PatchCore는 사전학습 백본의 특징을 그대로 메모리 뱅크에 모으는 방법이라 학습이
# 없다(docs/plan.md). 그래서 추출기는 파라미터를 얼리고 항상 eval 모드에 묶어 둔다. 특징을 뽑는
# 층·풀링·정규화는 docs/experiments.md 1·2단계 규칙에 미리 적은 문헌 기본값 그대로다.
# [연결] run_patchcore, run_grid, run_perturb, run_m2ad, run_m2ad_enrol, run_supervised, export(ONNX
# 내보내기), bench(GPU 지연)가 make_extractor를 쓰고, patchcore.py와 run_dinomaly.py가 to_tensor를 쓴다.
# torch가 필요한 모듈이라 서빙 쪽(inspector.py)은 이 파일을 쓰지 않고 같은 정규화를 numpy로 따로 한다.

from __future__ import annotations

# os는 TORCH_HOME 환경 변수 확인에, numpy는 입력 배열 검사에, torchvision은 WideResNet-50-2 구조와 ImageNet
# 가중치에 쓴다. timm(DINOv2)은 DinoV2Patches 안에서만 불러온다. paths는 가중치를 받을 data/ 폴더 위치를 준다.
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

from defect_inspect import paths

# ImageNet 채널 평균·표준편차. 두 백본 모두 ImageNet 정규화로 사전학습되어 입력도 같은 방식으로
# 맞춘다(docs/experiments.md 공통 규칙 "전처리").
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# DINOv2 ViT/14의 패치 한 변(px). 입력 한 변이 이 값의 배수여야 패치 격자로 나누어떨어진다. 그래서 4단계
# 격자는 256·384 대신 252·392px를 썼다.
DINOV2_PATCH = 14
# 추출기의 짧은 이름 -> timm 구조 이름.
DINOV2_ARCHS = {
    "dinov2_vits14": "vit_small_patch14_reg4_dinov2",
    "dinov2_vitb14": "vit_base_patch14_reg4_dinov2",
}


# 입력은 리사이즈 캐시의 배열 그대로다. 입력 형식이 다르면 바로 ValueError를 낸다.
# 호출 위치: patchcore의 _patch_features(특징 추출), run_dinomaly(학습·채점·대조).
def to_tensor(images: np.ndarray, device: str | torch.device, dtype: torch.dtype) -> torch.Tensor:
    """uint8 이미지 [B, S, S, 3]을 ImageNet 정규화한 [B, 3, S, S] 텐서(`dtype`, `device` 위)로 바꾼다."""
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError(f"expected uint8 [B, S, S, 3], got {images.dtype} {images.shape}")
    x = torch.from_numpy(np.ascontiguousarray(images)).to(device)
    # 정규화는 fp32로 하고 마지막에 한 번만 dtype(fp16 등)으로 바꾼다. fp16에서 나누기·빼기를 하면 그 과정에서
    # 정밀도를 잃기 때문이다.
    x = x.permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
    return x.sub_(mean).div_(std).to(dtype).contiguous()


# TORCH_HOME이 있으면 None을 돌려 torch 기본 위치(TORCH_HOME 아래)에 맡기고, 없으면 data/torch/checkpoints를
# 쓴다.
def _checkpoint_dir() -> str | None:
    """torchvision 가중치를 받을 폴더: TORCH_HOME이 있으면 torch 기본 위치, 없으면 data/torch.

    TORCH_HOME이 없으면 torch는 홈 폴더를 쓴다. 이 프로젝트가 내려받는 것은 data/ 아래에 둔다.
    """
    if os.environ.get("TORCH_HOME"):
        return None
    return str(paths.DATA / "torch" / "checkpoints")


# name(구성·기록에 쓰는 짧은 이름)과 dim(패치 특징 차원)을 모든 추출기가 가진다.
class _FrozenExtractor(torch.nn.Module):
    """추출기 공통 바탕 클래스: 파라미터는 학습되지 않고, .train()을 불러도 eval 모드에서 벗어나지 않는다."""

    name: str
    dim: int

    # nn.Module.train을 덮어써 항상 eval(False)로 둔다. WideResNet의 BatchNorm이 train 모드가 되면 배치 통계로
    # 정규화해 같은 이미지라도 배치 구성에 따라 특징이 달라지고, 이동 평균도 바뀌어 버린다.
    def train(self, mode: bool = True) -> _FrozenExtractor:
        # BatchNorm 통계가 고정되어 있어야 하므로 `.train()`은 아무것도 하지 않는다.
        return super().train(False)

    # 모든 파라미터의 기울기 계산을 끄고 eval 모드로 둔다. 각 추출기의 __init__ 마지막에 부른다.
    def _freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()


# 1단계 P0의 특징 추출기(docs/experiments.md 1단계 "구성 P0"). torchvision wide_resnet50_2(IMAGENET1K_V1)의
# layer2(512채널, 입력의 1/8 격자)와 layer3(1024채널, 1/16 격자) 출력을 각각 3×3 평균 풀링하고, layer3을
# layer2 격자로 bilinear 확대해 이어 붙인다. 256px 입력이면 32×32 격자, 패치당 1,536차원이다.
# PatchCore 논문은 얕은 층은 문맥이 부족하고 깊은 층은 ImageNet 분류 쪽으로 치우친다는 이유로 중간 층을 쓰고,
# 이 프로젝트는 그 기본값을 그대로 따른다. 3×3 이웃 평균은 패치마다 주변 문맥을 조금 섞어 준다. 3-A에서 P0이
# 256 격자 기준 1~2px 이동에 흔들리지 않은 것을 이 풀링과 stride 8 이상의 CNN 특징 덕분으로
# 본다(docs/experiments.md 3-A 읽는 법).
class WideResNetPatches(_FrozenExtractor):
    """torchvision WideResNet-50-2를 layer3까지 쓴다: layer2와 layer3 특징을 layer2 격자에 맞춰 잇는다."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        # 구조만 먼저 만들고, pretrained면 torchvision의 ImageNet 가중치를 해시 확인(check_hash)과 함께 받아
        # 넣는다. 테스트는 pretrained=False로 내려받기 없이 구조만 쓴다.
        net = wide_resnet50_2(weights=None)
        if pretrained:
            weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
            state = weights.get_state_dict(progress=False, check_hash=True, model_dir=_checkpoint_dir())
            net.load_state_dict(state)
        # layer4와 분류기는 돌리지 않으므로 보관하지 않는다.
        self.conv1, self.bn1, self.relu, self.maxpool = net.conv1, net.bn1, net.relu, net.maxpool
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3
        self.name = "wrn50"
        self.dim = 512 + 1024
        self._freeze()

    # 정규화된 [B, 3, S, S] → [B, S/8, S/8, 1536]. layer4와 분류기는 돌리지 않는다. 마지막에 채널을 맨 뒤로
    # 옮겨(permute) 다른 추출기와 같은 [B, H, W, D] 모양으로 맞춘다.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        f2 = self.layer2(self.layer1(x))  # [B, 512, H, W]
        f3 = self.layer3(f2)  # [B, 1024, H/2, W/2]
        f2 = F.avg_pool2d(f2, kernel_size=3, stride=1, padding=1)
        f3 = F.avg_pool2d(f3, kernel_size=3, stride=1, padding=1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1).permute(0, 2, 3, 1).contiguous()


# 2단계 D-S의 특징 추출기(docs/experiments.md 2단계). timm의 DINOv2 ViT(register 토큰 4개)에서 마지막 norm
# 층의 패치 토큰을 L2 정규화해 쓴다. 448px 입력이면 32×32 격자로 P0과 패치 수가 같다. 이웃 풀링은 하지 않는다.
# 3-A·3-B 결과로는 이 특징이 밝기 배율 변화에는 거의 흔들리지 않지만 1~2px 위치 어긋남에는 크게
# 무너졌다(README "촬영 조건이 조금 바뀌면").
class DinoV2Patches(_FrozenExtractor):
    """timm DINOv2 ViT: 마지막 norm 층의 패치 토큰을 L2 정규화해 쓴다."""

    def __init__(
        self, arch: str = "vit_small_patch14_reg4_dinov2", img_size: int = 448, pretrained: bool = True
    ) -> None:
        super().__init__()
        # 입력이 14의 배수가 아니면 패치 격자가 맞지 않으므로 만들 때 바로 거절한다. timm은
        # DinoV2Patches에서만 필요해서 여기서 불러온다(WideResNet만 쓰는 실행은 timm 없이 돈다).
        if img_size <= 0 or img_size % DINOV2_PATCH != 0:
            raise ValueError(f"img_size must be a positive multiple of {DINOV2_PATCH}, got {img_size}")
        import timm  # 이 추출기만 필요로 한다

        # num_classes=0으로 분류 머리를 없애고, img_size를 주면 timm이 위치 임베딩을 그 입력 크기에 맞게
        # 다시 표집한 모델을 만든다. name은 timm 이름을 짧은 이름(dinov2_vits14 등)으로 되돌린 것이고, dim은
        # 임베딩 차원(ViT-S 384, ViT-B 768)이다.
        self.model = timm.create_model(arch, pretrained=pretrained, num_classes=0, img_size=img_size)
        self.img_size = img_size
        self.name = next((short for short, full in DINOV2_ARCHS.items() if full == arch), arch)
        self.dim = int(self.model.embed_dim)
        self._freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 앞쪽 접두 토큰(클래스 토큰 1개 + register 4개)을 버리고 패치 토큰만 남긴다. 개수가 h×w와 다르면 입력
        # 크기가 모델과 맞지 않는 것이라 오류를 낸다. L2 정규화는 fp32로 해서 fp16 autocast에서도 노름 계산이
        # 정확하도록 한다.
        h, w = x.shape[-2] // DINOV2_PATCH, x.shape[-1] // DINOV2_PATCH
        tokens = self.model.forward_features(x)[:, self.model.num_prefix_tokens :]
        if tokens.shape[1] != h * w:
            raise ValueError(f"expected {h * w} patch tokens, got {tokens.shape[1]}")
        tokens = F.normalize(tokens.float(), dim=-1)
        return tokens.reshape(x.shape[0], h, w, tokens.shape[-1])


# 이미지 전체에 고르게 걸린 밝기 이동을 지우려는 것이고, 제품 전체 색이 바뀌는 결함도 같이 지울 수
# 있다(docs/experiments.md 7단계 후보).
# 추출기 자체를 감싸므로 뱅크를 만들 때와 질의를 채점할 때, 교차 적합 때 모두 같은 변환을 거친다. 결과: M2AD
# 실제 조명에서는 P0의 오검출을 줄이지 못했고(검증 95.8% → 95.4%), VisA dev 밝기 0.7에서는 38.9% → 14.0%로
# 줄였다.
# 호출 위치: make_extractor(centre=True), run_m2ad_enrol(중심화 없는 추출기 하나를 만들어 두고 같은 객체를
# 감싸 E1·E3에 씀).
class CentredPatches(_FrozenExtractor):
    """7단계 E1(이미지별 특징 중심화): 다른 추출기의 패치 특징에서 그 이미지의 패치 평균 특징을 뺀다.

    평균은 한 이미지의 패치 격자에 대해 fp32로 구하므로, 뱅크와 질의가 같은 변환을 거친다. 학습하는 것은
    없다.
    """

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.name = f"{inner.name}-centred"
        self.dim = int(inner.dim)
        self._freeze()

    # 평균은 한 이미지의 패치 격자(H, W 축)에 대해 fp32로 구한다. 이미지마다 따로 빼므로 배치 구성과 무관하다.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.inner(x).float()  # [B, H, W, D]
        return feats - feats.mean(dim=(1, 2), keepdim=True)


# configs.PatchCoreConfig의 backbone·img_size·centre를 그대로 받으면 되도록 했다. 모르는 이름이면 ValueError.
def make_extractor(
    name: str, *, img_size: int, pretrained: bool = True, centre: bool = False
) -> torch.nn.Module:
    """짧은 이름("wrn50", "dinov2_vits14", "dinov2_vitb14")으로 추출기를 만든다.

    `centre=True`면 CentredPatches(7단계 E1)로 감싼다.
    """
    if name == "wrn50":
        # 완전 합성곱이라 격자가 입력 크기를 따라가므로 여기서는 img_size가 필요 없다.
        extractor = WideResNetPatches(pretrained=pretrained)
    elif name in DINOV2_ARCHS:
        extractor = DinoV2Patches(DINOV2_ARCHS[name], img_size=img_size, pretrained=pretrained)
    else:
        raise ValueError(f"unknown extractor {name!r}; expected one of {['wrn50', *DINOV2_ARCHS]}")
    return CentredPatches(extractor) if centre else extractor
