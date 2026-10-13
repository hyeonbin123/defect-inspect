"""Dinomaly with anomalib's model code and a plain training loop: one model for all categories.

`train` fits the bottleneck and the decoder on the normal pool of the dev protocol (folds 1-4) and writes
`model.pt`, `train_log.jsonl` and `train.json`. `eval` scores the evaluation set of a protocol with that
model and writes the common run format: `run.json`, `<category>.npz` and `<category>_maps.npy`. It scores
in the precision the model was trained in (fp16 autocast or fp32) unless `--amp` / `--no-amp` says
otherwise, and records the precision that really ran in `run.json`.

`--config` picks a registered configuration: `dm` (stage 2, the default) or one of the ViT-S models of
stage 6 (`dms-<size>`, `dms-<size>-car`). `parity` and `gate` are the two checks of the stage 6 fp16
gate: the scaled attention against the original in fp32 (any device), and fp16 training against fp32
for 60 steps (CUDA only).
"""

# [파일 역할] Dinomaly 학습·평가 스크립트. anomalib 2.6.2의 모델 코드(DinomalyModel)만 쓰고 학습 루프는 직접
# 쓴다. 명령은 네 개다: train(12개 범주를 한 모델로 학습), eval(평가 세트 채점과 공통 결과 형식 저장),
# parity(6단계 관문 2 (가): 패치 어텐션과 원본 어텐션의 fp32 대조), gate(관문 2 (나): fp16 학습이 fp32와 같은
# 곡선을 내는지).
# [왜 필요한가] 2단계에서 비지도 방법 중 가장 강한 기준(DM, ViT-B 392px)이 필요했고(README: AUROC 96.8),
# 6단계에서는 CPU 서빙을 위해 인코더를 ViT-S로 줄인 모델(dms-*)을 같은 코드로 학습·평가해야 했다. 모델 구현은
# anomalib 것을 쓰되 학습 루프는 직접 써서(docstring), 배치 순서·정밀도·관문 검사를 직접 통제하고,
# 옵티마이저·스케줄러·클리핑은 anomalib Lightning 모듈의 설정과 맞춘다.
# [주요 구성] 구성 표(DinomalyConfig, CONFIGS) → 결정적 배치 순서(batch_order) → 모델 만들기(build_model) →
# 학습(train)·채점(predict) → 저장·불러오기(save_model, load_model) → 관문 검사(attention_parity,
# gate_verdict, run_gate) → 범주 평가(eval_category) → 명령 처리(_train_command 등, main).
# [연결] 모델 수정(인코더 자르기, 패치 어텐션)은 dinomaly_model.py, ONNX 서빙은
# dinomaly_serving.py(build_model과 CONFIGS를 씀), dev 선택과 test 판정은 analyze_stage6.py, 2단계 비교는
# compare.py가 이 실행의 출력을 읽는다.

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# 프로젝트 모듈: 리사이즈 캐시, sha256_file(모델·관문 파일의 해시를 기록에 남김), 장부, 픽셀 지표, 분할 선택.
# torch·anomalib은 기본 패키지 규칙(모듈 수준에서 torch를 import하지 않음)에 따라 함수 안에서 가져오고, 타입
# 힌트용으로만 TYPE_CHECKING 아래에서 가져온다.
import numpy as np

from . import paths
from .cache import ImageCache
from .download import sha256_file
from .ledger import git_commit, record_test_access
from .metrics import aupro, pixel_auroc, pro_histograms
from .splits import ManifestRow, read_manifest, select
from .visa import CATEGORIES

if TYPE_CHECKING:
    import torch

    from .patchcore import ScoreResult

# 인코더(timm 이름): 2단계 DM은 DINOv2 ViT-B/14 register 4개, 6단계 DMS는 ViT-S/14 register 4개(D-S
# PatchCore와 같은 가중치라 새로 받을 것이 없다). IMG_SIZE 392는 DM 입력이다. 공식 전처리(448로 줄이고 가운데
# 392 자르기) 대신 전체 이미지를 392로 줄여, 픽셀 지표를 다른 방법과 같은 영역에서 잰다(experiments.md 2단계).
# STEPS·BATCH: DM의 등록값 5,000스텝·배치 16. 실제 DM은 fp16에서 NaN이 나서 fp32·배치 8로 학습했다(--no-amp
# --batch-size 8, README 재현 명령).
ENCODER = "vit_base_patch14_reg4_dinov2"
ENCODER_S = "vit_small_patch14_reg4_dinov2"
IMG_SIZE = 392
STEPS = 5000
BATCH = 16
# anomalib Lightning 모듈의 학습 설정을 그대로 옮긴 값: StableAdamW 학습률 2e-3에서 2e-4로(warmup 100),
# gradient clip 0.1. 테스트가 anomalib 값과 같은지 확인한다(기존 영어 주석).
# The values of anomalib's TRAINING_CONFIG (dinomaly/lightning_model.py); a test keeps them in step.
OPTIMIZER = {"lr": 2e-3, "betas": (0.9, 0.999), "weight_decay": 1e-4, "amsgrad": True, "eps": 1e-8}
SCHEDULER = {"base_value": 2e-3, "final_value": 2e-4, "warmup_iters": 100}
GRAD_CLIP = 0.1
PRO_BINS = 2000
# 2단계 구성 이름이자 장부의 config 값. ATTENTION_MODES: 디코더 선형 어텐션 방식. original(anomalib 그대로),
# scaled(키를 토큰 수 N으로 나누는 fp16용 패치), fp32(어텐션만 fp32로 계산하는 대안).
NAME = "dm"  # method name in docs/experiments.md; also the config label in the test ledger
ATTENTION_MODES = ("original", "scaled", "fp32")  # see dinomaly_model.py
# 6단계 관문 2의 기준값(experiments.md 6단계 '관문 2'): (가) 패치 어텐션과 원본의 fp32 출력·손실 차이가 1e-5
# 이하, (나) fp16 60스텝이 모두 유한하고 건너뛴 스텝이 0이며 10스텝 구간 평균 손실이 fp32 기준과 상대 3% 이내.
# 한 스텝 단위가 아니라 10스텝 구간 평균으로 정한 것은 배치마다 손실이 흔들리기 때문이다.
# Stage 6 fp16 gate (docs/experiments.md): the scaled attention must match the original in fp32 within
# PARITY_TOL; fp16 training must stay finite, skip no step and keep every GATE_WINDOW-step mean loss
# within GATE_RTOL of the fp32 run.
PARITY_TOL = 1e-5
GATE_STEPS = 60
GATE_WINDOW = 10
GATE_RTOL = 0.03


# 등록된 Dinomaly 구성 하나. 2단계 dm(ViT-B, 392px, 5,000스텝)과 6단계 dms-*(ViT-S, 252~308px, 10,000스텝,
# 인코더 10블록, 입력 크기 고정, CAR 선택, 패치 어텐션)를 같은 형식으로 둔다. frozen이라 실행 중 실수로 바뀌지
# 않는다.
@dataclass(frozen=True)
class DinomalyConfig:
    """A registered Dinomaly configuration (docs/experiments.md: `dm` in stage 2, `dms-*` in stage 6)."""

    name: str
    encoder: str
    img_size: int
    steps: int
    batch_size: int = BATCH
    # encoder_blocks: 인코더 앞쪽 블록만 남긴다. anomalib은 12블록을 다 돌리지만 읽는 층이 2~9라 10, 11번 블록
    # 출력은 쓰이지 않는다(출력이 같음을 테스트로 확인, experiments.md 6단계). fixed_size: 위치 임베딩을
    # img_size로 한 번만 보간해 둔다. context_recentering: Dinomaly2의 CAR(인코더 특징에서 클래스 토큰을 빼고
    # 복원).
    encoder_blocks: int | None = None  # keep the first blocks of the encoder only (None: all)
    fixed_size: bool = False  # resample the position embedding once, for img_size
    context_recentering: bool = False  # Dinomaly2's Context-Aware Recentering (CAR)
    attention: str = "original"  # decoder attention unless the command line or the gate says otherwise

    # 이 구성으로 build_model에 넘길 키워드 인자를 만든다(dm은 빈 dict). attention을 주면 구성의 기본 어텐션
    # 대신 그것을 쓴다(관문 검사와 --precision-from이 쓴다). 저장 파일에 이 dict를 그대로 남겨 load_model이
    # 같은 모델을 다시 만든다.
    def build_options(self, attention: str | None = None) -> dict:
        """Keyword arguments of `build_model` beyond the encoder name (empty for `dm`)."""
        options: dict = {}
        # 옵션은 기본값과 다를 때만 넣는다. 그래서 dm은 빈 dict라 anomalib 원래 모델 그대로 만들어진다.
        if self.fixed_size:
            options["img_size"] = self.img_size
        if self.encoder_blocks is not None:
            options["encoder_blocks"] = self.encoder_blocks
        if self.context_recentering:
            options["context_recentering"] = True
        # 인자로 받은 어텐션이 구성 기본값보다 우선한다. original이면 옵션을 넣지 않는다.
        mode = attention or self.attention
        if mode not in ATTENTION_MODES:
            raise ValueError(f"attention must be one of {ATTENTION_MODES}, got {mode!r}")
        if mode != "original":
            options["attention"] = mode
        return options


# 6단계 DMS 구성을 만드는 도우미: ViT-S, 10,000스텝(논문의 VisA 설정, 배치 16 × 10,000 = 16만 장), 인코더
# 10블록, 입력 크기 고정, 패치 어텐션(scaled).
def _dms(size: int, car: bool = False) -> DinomalyConfig:
    return DinomalyConfig(
        name=f"dms-{size}" + ("-car" if car else ""),
        encoder=ENCODER_S,
        img_size=size,
        steps=10_000,
        encoder_blocks=10,
        fixed_size=True,
        context_recentering=car,
        attention="scaled",
    )


# 등록된 구성 표. 키가 --config 값이다. 6단계 후보는 252, 280, 294(관문 1에서 308이 빠질 때만), 308과 CAR 두
# 개(252-car는 280이 빠질 때의 대체)다. 실제로는 dms-252, 280, 308, 280-car를 학습했고 dev에서 dms-280-car(dev
# 96.98)가 뽑혔다(experiments.md 6단계).
CONFIGS: dict[str, DinomalyConfig] = {
    c.name: c
    for c in (
        DinomalyConfig(NAME, ENCODER, IMG_SIZE, STEPS),
        _dms(252),
        _dms(280),
        _dms(294),
        _dms(308),
        _dms(252, car=True),
        _dms(280, car=True),
    )
}


# perm 앞쪽 head칸에 들어 있는 blocked 원소를, 뒤쪽에서 처음 나오는 blocked가 아닌 원소와 자리 바꿈한다(제자리
# 수정). batch_order가 에폭 경계의 배치에서 같은 이미지가 두 번 들어가지 않게 할 때 쓴다.
def _keep_out(perm: np.ndarray, blocked: np.ndarray, head: int) -> None:
    """Swap every member of `blocked` out of `perm[:head]` (in place) with the first later free entries."""
    is_blocked = np.isin(perm, blocked)
    clash = np.flatnonzero(is_blocked[:head])
    free = np.flatnonzero(~is_blocked[head:])[: len(clash)] + head
    perm[clash], perm[free] = perm[free], perm[clash]


# 학습 배치마다의 이미지 인덱스 [steps, batch_size]를 seed로 결정적으로 만든다. 전체 이미지의 무작위
# 순열(에폭)을 이어 붙여 배치로 자르고, 배치가 두 에폭에 걸치면 이전 에폭 끝 이미지를 새 에폭에서 뒤로 보내 한
# 배치에 같은 이미지가 두 번 들어가지 않게 한다.
# 앞쪽 행은 steps에 의존하지 않으므로, 관문 검사(60스텝)와 실제 학습(10,000스텝)이 같은 배치 순서로
# 시작한다(experiments.md 관문 2 (나): '실제 학습과 같은 배치 순서').
# numpy Generator(seed)만 써서 결정적이다.
def batch_order(n_images: int, steps: int, batch_size: int, seed: int) -> np.ndarray:
    """Image indices of every training batch, int64 [steps, batch_size].

    Epochs of seeded permutations of all images are concatenated and cut into batches. Where a batch
    spans two epochs, images that the old epoch ended with are swapped further back in the new one, so
    that a batch never holds the same image twice (unless there are fewer images than a batch holds).
    The first rows do not depend on `steps`.
    """
    if n_images < 1 or batch_size < 1 or steps < 0:
        raise ValueError(
            f"need n_images >= 1, batch_size >= 1, steps >= 0; got {n_images}, {batch_size}, {steps}"
        )
    # 전체 스트림(steps × batch_size개)을 seed 하나의 Generator로 만든 에폭 순열들로 채운다.
    rng = np.random.default_rng(seed)
    total = steps * batch_size
    stream = np.empty(total, dtype=np.int64)
    pos = 0
    while pos < total:
        perm = rng.permutation(n_images)
        # carried: 열려 있는 배치에 이미 들어간 이전 에폭 이미지 수. 그 이미지들이 새 에폭 앞부분(배치의
        # 나머지 칸)에 다시 나오지 않게 뒤로 보낸다. 이미지 수가 배치보다 적으면 중복을 피할 수 없으므로 하지
        # 않는다.
        carried = pos % batch_size  # images of the previous epoch that are already in the open batch
        if carried and n_images >= batch_size:
            _keep_out(perm, stream[pos - carried : pos], batch_size - carried)
        # 마지막 에폭은 남은 칸만큼만 잘라 쓴다.
        take = min(n_images, total - pos)
        stream[pos : pos + take] = perm[:take]
        pos += take
    return stream.reshape(steps, batch_size)


# anomalib DinomalyModel을 Lightning 모듈과 같은 방식으로 준비한다: fp32, 인코더 고정, 병목과 디코더만 학습.
# 선형층은 절단 정규분포(std 0.01, ±0.03), 편향 0, LayerNorm은 가중치 1·편향 0으로 초기화한다. 초기값은 전역
# torch 난수를 쓰므로 호출 전에 torch.manual_seed로 고정한다(docstring).
# 6단계 옵션(입력 크기 고정·인코더 블록 줄이기·CAR·어텐션 방식)은 dinomaly_model.fix_encoder와
# set_attention으로 적용하며, 이 옵션들은 난수를 쓰지 않아 같은 seed면 같은 초기값이 된다.
# 호출 위치: _train_command, load_model, _builder(관문), dinomaly_serving(--untrained 내보내기).
def build_model(
    encoder_name: str = ENCODER,
    *,
    img_size: int | None = None,
    encoder_blocks: int | None = None,
    context_recentering: bool = False,
    attention: str = "original",
) -> torch.nn.Module:
    """anomalib's DinomalyModel, set up as its Lightning module does: fp32, frozen encoder, fresh decoder.

    Only the bottleneck and the decoder train. Their Linear layers start from a truncated normal (std
    0.01, cut at +-0.03) with zero bias, their LayerNorms from weight 1 and bias 0. These weights come
    from the global torch generator: call `torch.manual_seed` first to fix them.

    The keyword options are those of the stage 6 configurations (`DinomalyConfig.build_options`): a
    fixed input size and fewer encoder blocks (`dinomaly_model.fix_encoder`), Context-Aware Recentering,
    and the decoder attention mode. None of them draws from the torch generator.
    """
    import torch
    from anomalib.models.image.dinomaly.torch_model import DinomalyModel

    # anomalib 모델을 fp32로 만든다. CAR은 anomalib의 use_context_recentering 옵션 그대로다.
    model = DinomalyModel(encoder_name=encoder_name, use_context_recentering=context_recentering).float()
    # 6단계 옵션이 있을 때만 dinomaly_model을 가져와 인코더를 고정 크기·10블록으로 고치고 디코더 어텐션을
    # 바꾼다. dm은 이 분기를 타지 않는다.
    if img_size is not None or encoder_blocks is not None or attention != "original":
        from .dinomaly_model import fix_encoder, set_attention

        if img_size is not None or encoder_blocks is not None:
            fix_encoder(model, img_size=img_size, blocks=encoder_blocks)
        set_attention(model, attention)
    # 먼저 모든 파라미터를 고정하고, 병목과 디코더만 다시 학습 가능으로 바꾼 뒤 초기화한다. 사전학습 인코더는
    # 그대로 두는 Dinomaly의 설계다(docs/plan.md: DINOv2 인코더를 고정하고 디코더가 특징을 복원하게 학습).
    for param in model.parameters():
        param.requires_grad = False
    for module in (model.bottleneck, model.decoder):
        for param in module.parameters():
            param.requires_grad = True
        for layer in module.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.trunc_normal_(layer.weight, std=0.01, a=-0.03, b=0.03)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, torch.nn.LayerNorm):
                torch.nn.init.constant_(layer.bias, 0)
                torch.nn.init.constant_(layer.weight, 1.0)
    return model


# 입력이 uint8 [N, S, S, 3]이고 배치 크기가 1 이상인지 확인한다.
def _check_images(images: np.ndarray, batch_size: int) -> None:
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError(f"expected uint8 images [N, S, S, 3], got {images.dtype} {images.shape}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")


# amp 플래그가 실제로 fp16으로 도는지: autocast는 CUDA에서만 쓰므로 CPU에서는 항상 False다. 기록(run.json,
# train.json)에는 플래그가 아니라 실제로 돈 정밀도를 남긴다.
def amp_applied(amp: bool, device: str | torch.device) -> bool:
    """Whether `amp` really runs fp16 on `device`: autocast is only used on CUDA, the CPU stays fp32."""
    import torch

    return bool(amp) and torch.device(device).type == "cuda"


# CUDA이고 amp면 fp16 autocast, 아니면 아무것도 하지 않는 문맥. RTX 2080 Ti에는 bf16이 없어 fp16만 쓴다.
def _autocast(device: torch.device, amp: bool) -> contextlib.AbstractContextManager:
    """fp16 autocast on CUDA when `amp` (this GPU has no bf16), plain fp32 otherwise."""
    import torch

    if amp_applied(amp, device):
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


# 학습 가능한 파라미터를 uint8 이미지로 학습한다. anomalib Lightning 모듈을 따른다: 학습 모드에서 손실 =
# model(batch, global_step=step), StableAdamW, 매 반복마다 warmup + cosine 스케줄, gradient norm 0.1로 자르기.
# amp(CUDA)면 fp16 autocast와 GradScaler를 쓴다.
# 손실이 유한하지 않으면 FloatingPointError로 멈춘다(2단계 DM은 fp16에서 15번째 스텝에 NaN, experiments.md).
# log_every 스텝마다(그리고 마지막 스텝) 기록을 남기고, log_path가 있으면 JSON 줄로 하나씩 flush해 오래 걸리는
# 학습의 진행을 밖에서 볼 수 있게 한다.
# 호출 위치: _train_command, attention_parity, _gate_run.
def train(
    model: torch.nn.Module,
    images: np.ndarray,
    *,
    steps: int = STEPS,
    batch_size: int = BATCH,
    device: str = "cuda",
    amp: bool = True,
    seed: int = 0,
    log_every: int = 50,
    log_path: Path | None = None,
) -> list[dict]:
    """Train the parameters of `model` that require gradients on uint8 images [N, S, S, 3].

    Follows anomalib's Lightning module: loss = `model(batch, global_step=step)` in training mode,
    StableAdamW, warm-up + cosine schedule stepped every iteration, gradient norm clipped to 0.1. `amp`
    (CUDA only) runs the forward pass under fp16 autocast with a GradScaler. A non-finite loss raises
    FloatingPointError. Returns one entry per `log_every` steps (and for the last step) and, with
    `log_path`, appends the same entries as JSON lines, flushed one by one.
    """
    import torch
    from anomalib.models.image.dinomaly.components import StableAdamW, WarmCosineScheduler

    from .backbones import to_tensor

    _check_images(images, batch_size)
    if steps < 1 or log_every < 1:
        raise ValueError(f"steps and log_every must be at least 1, got {steps} and {log_every}")
    dev = torch.device(device)
    use_amp = amp_applied(amp, dev)
    # 배치 순서는 학습 전에 한 번에 정한다. 병목 dropout 마스크는 torch 난수라 seed로 따로 고정한다.
    order = batch_order(len(images), steps, batch_size, seed)

    torch.manual_seed(seed)  # dropout masks of the bottleneck
    # 학습 모드(병목 dropout 켬)로 장치에 올리고, requires_grad인 파라미터(병목·디코더)만 옵티마이저에 넘긴다.
    model = model.to(dev).train()
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("the model has no trainable parameters")
    # anomalib Lightning 모듈과 같은 옵티마이저(StableAdamW)와 스케줄러(warmup 100 뒤 cosine, total_iters =
    # 전체 스텝)를 쓴다.
    optimizer = StableAdamW([{"params": params}], **OPTIMIZER)
    scheduler = WarmCosineScheduler(optimizer, total_iters=steps, **SCHEDULER)
    # GradScaler는 fp16에서 작은 기울기가 0으로 사라지지 않게 손실을 키웠다가 되돌린다. amp가 아니면
    # enabled=False라 아무 일도 하지 않는다.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    log: list[dict] = []
    skipped = 0
    # 로그 파일은 있을 때만 열고, 예외(NaN 등)로 끝나도 ExitStack이 닫는다. 그때까지 쓴 줄은 이미 flush돼
    # 남는다.
    with contextlib.ExitStack() as stack:
        handle = None
        if log_path is not None:
            handle = stack.enter_context(open(log_path, "a", encoding="utf-8", newline="\n"))
        started = time.perf_counter()
        for step in range(steps):
            # 배치를 GPU로 올려 ImageNet 정규화한 fp32 텐서로 만든다(정밀도 변환은 autocast가 맡는다).
            batch = to_tensor(images[order[step]], dev, torch.float32)
            # anomalib DinomalyModel은 학습 모드에서 손실을 바로 돌려준다. global_step은 anomalib Lightning
            # 모듈이 넘기는 것과 같은 값이다. 순전파만 autocast 안에서 하고 역전파는 밖에서 한다.
            with _autocast(dev, use_amp):
                loss = model(batch, global_step=step)
            # 손실이 유한한지 역전파 전에 확인해, NaN이 나면 그 스텝에서 바로 멈추고 몇 번째 스텝인지 알린다.
            value = float(loss.detach())
            if not math.isfinite(value):
                raise FloatingPointError(
                    f"non-finite loss ({value}) at step {step + 1} of {steps} (global_step={step})"
                )
            # 역전파 → 기울기 되돌리기(unscale) → norm 자르기 → 스텝 순서. unscale을 먼저 해야 클리핑이 실제
            # 기울기 크기 기준으로 된다.
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # GradScaler는 기울기에 inf/NaN이 있으면 그 스텝의 가중치 갱신을 건너뛰고 스케일을 낮춘다.
            # 스케일이 줄었는지로 건너뛴 스텝을 세어 관문 (나)의 '건너뛴 스텝 0' 조건에 쓴다.
            # The scaler leaves the weights alone and lowers its scale when a gradient is not finite.
            skipped += int(scaler.get_scale() < scale)
            # 이번 스텝에 실제로 쓴 학습률을 기록한 뒤 스케줄러를 한 칸 진행한다.
            lr = float(optimizer.param_groups[0]["lr"])
            scheduler.step()

            # 기록 항목: 스텝, 손실, 학습률, 경과 시간, 최대 VRAM, 지금까지 건너뛴 스텝 수. VRAM은 공유
            # 메모리로 넘치면 20배쯤 느려지므로 기록한다.
            if (step + 1) % log_every == 0 or step + 1 == steps:
                vram_mb = 0.0
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                    vram_mb = torch.cuda.max_memory_allocated(dev) / 2**20
                entry = {
                    "step": step + 1,
                    "loss": value,
                    "lr": lr,
                    "seconds": round(time.perf_counter() - started, 3),
                    "vram_mb": round(vram_mb, 1),
                    "skipped": skipped,
                }
                log.append(entry)
                if handle is not None:
                    handle.write(json.dumps(entry) + "\n")
                    handle.flush()
    return log


# 학습한 모델로 이미지를 채점한다. 이미지 점수는 모델 추론 출력의 pred_score(흐린 맵의 상위 1% 평균, 구현
# 기본값), 맵은 anomaly_map을 256으로 맞춘 것이다. 유한하지 않으면 FloatingPointError.
# fp16(amp)으로 채점하면 점수가 fp16으로 반올림돼(0.125~0.25에서 1.2e-4 간격) 정상끼리 동점이
# 많아진다(docstring). 그래서 6단계 dev·test 채점은 학습이 fp16이어도 fp32(--no-amp)로 한다(experiments.md
# 6단계: 서빙도 FP32).
def predict(
    model: torch.nn.Module,
    images: np.ndarray,
    *,
    batch_size: int = 32,
    device: str = "cuda",
    amp: bool = True,
    map_size: int = 256,
) -> ScoreResult:
    """Score uint8 images [N, S, S, 3] with the model in eval mode.

    Image score = `pred_score` of the model's inference output; map = its `anomaly_map`, resized
    bilinearly to `map_size` when it has another size. Non-finite scores or maps raise FloatingPointError.

    With `amp` on CUDA the model smooths and averages its score map in fp16, so the image scores come
    out rounded to fp16 (steps of 1.2e-4 between 0.125 and 0.25) and many images tie. They are stored
    as float32 either way; `amp=False` gives scores with fp32 precision.
    """
    import torch
    import torch.nn.functional as F

    from .backbones import to_tensor
    from .patchcore import ScoreResult

    _check_images(images, batch_size)
    dev = torch.device(device)
    n = len(images)
    image_scores = np.empty(n, dtype=np.float32)
    maps = np.empty((n, map_size, map_size), dtype=np.float16)
    if n == 0:
        return ScoreResult(image_scores, maps)

    # 평가 모드(dropout 끔)와 no_grad로 배치 단위 채점한다. 입력은 fp32 텐서이고 amp일 때만 autocast가
    # fp16으로 계산한다.
    model = model.to(dev).eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = to_tensor(images[start : start + batch_size], dev, torch.float32)
            b = batch.shape[0]
            with _autocast(dev, amp):
                out = model(batch)
            # 모델 출력 모양을 확인한다. 맵이 [B, H, W]면 채널 축을 붙여 [B, 1, H, W]로 맞추고, 점수 수나 맵
            # 모양이 예상과 다르면 멈춘다(예상과 다른 출력 형식을 조용히 쓰지 않게).
            scores = out.pred_score.float().reshape(-1)
            batch_maps = out.anomaly_map.float()
            if batch_maps.ndim == 3:
                batch_maps = batch_maps.unsqueeze(1)
            if scores.shape[0] != b or batch_maps.ndim != 4 or batch_maps.shape[:2] != (b, 1):
                raise ValueError(
                    f"expected {b} scores and maps [{b}, 1, H, W], got {tuple(out.pred_score.shape)} "
                    f"and {tuple(out.anomaly_map.shape)}"
                )
            # 모델 맵이 256이 아니면 bilinear로 256 격자에 맞춘다(모든 방법의 픽셀 지표를 같은 격자에서 잼).
            if tuple(batch_maps.shape[-2:]) != (map_size, map_size):
                batch_maps = F.interpolate(
                    batch_maps, size=(map_size, map_size), mode="bilinear", align_corners=False
                )
            # 맵은 float16으로 저장해 용량을 줄인다(PatchCore의 ScoreResult와 같은 형식).
            batch_maps = batch_maps[:, 0].to(torch.float16)
            if not bool(torch.isfinite(scores).all()) or not bool(torch.isfinite(batch_maps).all()):
                raise FloatingPointError(f"non-finite scores or maps in images {start}..{start + b - 1}")
            image_scores[start : start + b] = scores.cpu().numpy()
            maps[start : start + b] = batch_maps.cpu().numpy()
    return ScoreResult(image_scores, maps)


# 학습되는 파라미터(병목과 디코더)만 이름별 CPU 복사본으로 모은다. 고정 인코더는 사전학습 가중치에서 다시 만들
# 수 있으므로 저장하지 않아 파일이 작다.
def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """CPU copies of the parameters that train (for Dinomaly: the bottleneck and the decoder), by name."""
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


# 학습 상태를 인코더 이름·스텝 수·실제 학습 정밀도(amp)·구성 이름·build_model 옵션과 함께 저장한다. .part로
# 쓰고 os.replace로 바꿔, 쓰다 끊긴 파일이 완성된 model.pt로 보이지 않게 한다.
def save_model(
    path: Path,
    model: torch.nn.Module,
    *,
    encoder: str,
    steps: int,
    amp: bool,
    config: str | None = None,
    options: dict | None = None,
) -> None:
    """Write the trainable state with the encoder name and step count (the frozen encoder is not stored).

    `amp` is the precision the training really ran in (True: fp16 autocast); `eval` scores in the same
    precision unless told otherwise. `config` (a CONFIGS name) and `options` (the `build_model` keyword
    arguments) are stored when given, so that `load_model` rebuilds the same model.
    """
    import torch

    path = Path(path)
    part = path.with_name(path.name + ".part")
    saved = {"encoder": encoder, "steps": int(steps), "amp": bool(amp), "state": trainable_state(model)}
    if config is not None:
        saved["config"] = str(config)
        saved["options"] = dict(options or {})
    torch.save(saved, part)
    os.replace(part, path)


# save_model 파일로 CPU에서 모델을 다시 만든다(eval 모드). 저장된 옵션으로 build_model을 부르고, 저장된 학습
# 파라미터 이름 집합이 그 모델의 것과 정확히 같은지 확인한 뒤 불러온다. weights_only=True로 임의 객체
# 역직렬화를 막는다.
# amp가 기록되지 않은 옛 파일은 None을 돌려줘, eval이 --amp/--no-amp를 요구하게 한다.
# 호출 위치: _eval_command, dinomaly_serving(학습한 모델을 ONNX로 내보낼 때).
def load_model(path: Path) -> tuple[torch.nn.Module, dict]:
    """Rebuild the model of a `save_model` file on the CPU: (model in eval mode, {"encoder", "steps", "amp"}).

    "amp" is None for a file that does not record the precision it was trained in. Files that record a
    configuration add "config" and "options" to the dictionary; the model is built with those options.
    """
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=True)
    options = dict(saved.get("options") or {})
    model = build_model(saved["encoder"], **options)
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(saved["state"]) != expected:
        odd = sorted(set(saved["state"]) ^ expected)
        raise ValueError(f"{path} does not hold the trainable parameters of {saved['encoder']}: {odd[:3]}")
    # strict=False인 이유: 파일에는 학습 파라미터만 있고 고정 인코더 가중치는 없다. 빠진 키가 인코더뿐인지는
    # 바로 위 이름 집합 비교로 이미 확인했다.
    model.load_state_dict(saved["state"], strict=False)
    amp = saved.get("amp")
    meta = {
        "encoder": saved["encoder"],
        "steps": int(saved["steps"]),
        "amp": None if amp is None else bool(amp),
    }
    if "config" in saved:
        meta["config"] = saved["config"]
        meta["options"] = options
    return model.eval(), meta


# ---------------------------------------------------------------- the stage 6 fp16 gate


# 관문 2 (가): 같은 초기값에서 만든 패치 어텐션(scaled) 모델과 원본 모델을 fp32로 비교한다. 첫 배치의 추론
# 점수(상대)·맵(절대) 차이와 학습 steps스텝(기본 3)의 손실 상대 차이가 모두 PARITY_TOL(1e-5) 이하이면 통과.
# 왜: 패치는 키 k를 토큰 수 N으로 나누는 것이라 출력 (q·(kᵀv)) / (q·Σk)에서 N이 약분되어 수식상 같고 합만
# 1/N로 줄어든다(experiments.md 관문 2). 수식상 같다는 것을 실제 계산으로 확인해야 fp16 학습 결과를 원본
# 모델의 결과로 볼 수 있다. 결과: 점수 차 0, 맵 2.4e-7, 손실 3스텝 차이 0으로 통과(experiments.md 관문 2 (가)
# 결과).
def attention_parity(
    build: Callable[[str], torch.nn.Module],
    images: np.ndarray,
    *,
    steps: int = 3,
    batch_size: int = 8,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Gate 2a: the scaled attention against the original, both fp32, from the same initial weights.

    `build(attention)` must return a fresh model; it is called right after `torch.manual_seed(seed)`.
    Compares the eval-mode scores and maps on the first batch of `batch_order`, then the losses of
    `steps` training steps (fp32, the batches of `batch_order`). Passes when every relative loss and
    score difference and every absolute map difference is at most PARITY_TOL.
    """
    import torch

    from .backbones import to_tensor

    _check_images(images, batch_size)
    # 두 모델을 같은 seed 직후에 만들어 초기 가중치를 같게 한다.
    torch.manual_seed(seed)
    reference = build("original")
    torch.manual_seed(seed)
    patched = build("scaled")
    # 비교 배치는 실제 학습의 첫 배치(batch_order의 첫 행)다.
    first = images[batch_order(len(images), 1, batch_size, seed)[0]]
    x = to_tensor(first, device, torch.float32)
    with torch.no_grad():
        a = reference.to(device).eval()(x)
        b = patched.to(device).eval()(x)
    # 점수는 상대 차이, 맵은 절대 차이로 잰다(규칙의 기준과 같은 단위).
    score_rel = float(((a.pred_score - b.pred_score).abs() / a.pred_score.abs().clamp_min(1e-12)).max())
    map_abs = float((a.anomaly_map - b.anomaly_map).abs().max())

    # 두 모델을 같은 배치·같은 seed로 fp32(amp=False) 학습해 스텝별 손실을 비교한다. 학습 뒤 가중치 최대
    # 차이도 참고로 남긴다.
    common = {"steps": steps, "batch_size": batch_size, "device": device, "amp": False, "seed": seed}
    log_a = train(reference, images, log_every=1, **common)
    log_b = train(patched, images, log_every=1, **common)
    loss_rel = [
        abs(ea["loss"] - eb["loss"]) / max(abs(ea["loss"]), 1e-12)
        for ea, eb in zip(log_a, log_b, strict=True)
    ]
    state_a, state_b = trainable_state(reference), trainable_state(patched)
    weight_abs = max(float((state_a[k] - state_b[k]).abs().max()) for k in state_a)
    passed = max(loss_rel) <= PARITY_TOL and score_rel <= PARITY_TOL and map_abs <= PARITY_TOL
    return {
        "passed": bool(passed),
        "tolerance": PARITY_TOL,
        "batch_size": batch_size,
        "steps": steps,
        "score_rel_diff": score_rel,
        "map_abs_diff": map_abs,
        "loss_rel_diff": loss_rel,
        "losses_original": [e["loss"] for e in log_a],
        "losses_scaled": [e["loss"] for e in log_b],
        "weight_abs_diff_after": weight_abs,
    }


# 관문 2 (나)의 판정: fp16 손실 목록을 같은 배치의 fp32 기준 손실과 비교한다. 통과 조건은 (1) 스텝 수가 같고
# (2) 모두 유한하고 (3) GradScaler가 건너뛴 스텝이 0이고 (4) window(10)스텝 구간 평균이 모두 기준과 상대
# rtol(3%) 이내. 실패하면 이유를 reason에 적는다.
def gate_verdict(
    reference: list[float],
    losses: list[float],
    *,
    skipped: int,
    window: int = GATE_WINDOW,
    rtol: float = GATE_RTOL,
) -> dict:
    """Gate 2b on per-step losses: fp16 (`losses`) against fp32 (`reference`) over the same batches.

    Passes when there are as many fp16 losses as reference losses, all finite, no step was skipped by
    the gradient scaler, and the mean loss of every `window` consecutive steps is within `rtol` of the
    reference's mean over the same steps.
    """
    # 기준 손실 수가 구간 크기의 배수여야 구간 평균을 낼 수 있다.
    if not reference or len(reference) % window:
        raise ValueError(f"need a reference of a positive multiple of {window} steps, got {len(reference)}")
    out: dict = {"skipped": int(skipped), "window": window, "rtol": rtol}
    if len(losses) != len(reference):
        return {**out, "passed": False, "reason": f"{len(losses)} of {len(reference)} steps ran"}
    values = np.asarray(losses, dtype=np.float64)
    if not np.isfinite(values).all():
        return {**out, "passed": False, "reason": "a loss is not finite"}
    # 손실을 window개씩 묶어 구간 평균을 내고 기준 대비 상대 차이를 구한다. 건너뛴 스텝 검사가 구간 비교보다
    # 먼저다.
    ref = np.asarray(reference, dtype=np.float64).reshape(-1, window).mean(axis=1)
    got = values.reshape(-1, window).mean(axis=1)
    rel = np.abs(got - ref) / np.abs(ref)
    out["window_rel_diff"] = [float(v) for v in rel]
    if skipped:
        return {**out, "passed": False, "reason": f"the gradient scaler skipped {skipped} steps"}
    if float(rel.max()) > rtol:
        return {**out, "passed": False, "reason": f"a window mean differs by {float(rel.max()):.2%}"}
    return {**out, "passed": True, "reason": ""}


# 관문 실행 하나: 같은 seed로 모델을 만들어 steps스텝 학습하고, JSON 줄 로그에서 손실·건너뛴 스텝·VRAM·시간을
# 읽어 돌려준다. 손실이 NaN이 되면 예외 대신 error에 남겨 판정이 '실패'로 기록되게 한다.
def _gate_run(build, attention: str, amp: bool, images, *, steps, batch_size, device, seed, log_path) -> dict:
    import torch

    # train이 로그를 덧붙이기로 쓰므로 이전 로그를 먼저 지운다.
    log_path.unlink(missing_ok=True)
    torch.manual_seed(seed)
    model = build(attention)
    started = time.perf_counter()
    error = ""
    try:
        train(
            model,
            images,
            steps=steps,
            batch_size=batch_size,
            device=device,
            amp=amp,
            seed=seed,
            log_every=1,
            log_path=log_path,
        )
    except FloatingPointError as err:
        error = str(err)
    # 학습이 NaN으로 멈췄어도 그때까지 flush된 로그로 손실 수·건너뛴 스텝을 판정에 넘긴다.
    log = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    return {
        "attention": attention,
        "amp": amp,
        "error": error,
        "losses": [e["loss"] for e in log],
        "skipped": log[-1]["skipped"] if log else 0,
        "peak_vram_mb": log[-1]["vram_mb"] if log else 0.0,
        "seconds": round(time.perf_counter() - started, 1),
        "train_seconds_at_end": log[-1]["seconds"] if log else 0.0,
    }


# 관문 2 (나) 순서: fp32 + 원본 어텐션(기준) → fp16 + 패치 어텐션 → (실패할 때만) fp16 + 어텐션만 fp32. 처음
# 통과한 fp16 변형을 고르고, 모두 실패하면 fp32 + 원본 어텐션으로 학습한다. 선택은 gate.json의 choice에 남고
# train --precision-from이 그대로 읽는다.
# 결과(experiments.md): 첫 변형(fp16 + 패치)이 통과했고 구간 평균 차이는 최대 2.64%였다.
def run_gate(
    build,
    images: np.ndarray,
    *,
    out_dir: Path,
    steps: int = GATE_STEPS,
    batch_size: int = BATCH,
    device: str = "cuda",
    seed: int = 0,
) -> dict:
    """Gate 2b: fp32 with the original attention, then fp16 with the scaled attention, then (only if that
    fails) fp16 with the attention in fp32. The first fp16 variant that passes is the choice; when none
    passes, training runs in fp32 with the original attention.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    common = {"steps": steps, "batch_size": batch_size, "device": device, "seed": seed}
    # 기준(fp32 + 원본 어텐션)이 실패하면 비교 자체가 안 되므로 예외로 멈춘다.
    reference = _gate_run(
        build, "original", False, images, log_path=out_dir / "log_fp32_original.jsonl", **common
    )
    if reference["error"] or len(reference["losses"]) != steps:
        raise FloatingPointError(f"the fp32 reference run failed: {reference['error']}")
    variants = []
    choice = {"amp": False, "attention": "original"}
    # fp16 변형을 정해진 순서로 시도하고 처음 통과한 것에서 멈춘다. 하나도 통과하지 못하면 choice는 처음
    # 값(fp32 + 원본)으로 남는다.
    for attention in ("scaled", "fp32"):
        log_path = out_dir / f"log_fp16_{attention}.jsonl"
        run = _gate_run(build, attention, True, images, log_path=log_path, **common)
        run["verdict"] = gate_verdict(reference["losses"], run["losses"], skipped=run["skipped"])
        if run["error"]:
            run["verdict"] = {**run["verdict"], "passed": False, "reason": run["error"]}
        variants.append(run)
        if run["verdict"]["passed"]:
            choice = {"amp": True, "attention": attention}
            break
    return {"reference": reference, "variants": variants, "choice": choice}


# 범주 하나의 평가 세트를 채점해 <범주>.npz(점수, 라벨, 결함 유형, 보정 점수, 픽셀 지표, PRO 히스토그램)와
# <범주>_maps.npy(256 맵, float16)를 쓴다. 다른 방법과 같은 공통 결과 형식이라 analyze·compare·figures가
# 그대로 읽는다.
def eval_category(
    model: torch.nn.Module,
    protocol: str,
    category: str,
    manifest: list[ManifestRow],
    cache: ImageCache,
    device: str,
    allow_test: bool,
    out_dir: Path,
    *,
    batch_size: int = 32,
    amp: bool = True,
) -> dict:
    """Score one category's evaluation set and write `<category>.npz` and `<category>_maps.npy`."""
    import torch

    eval_normal = select(
        manifest, protocol=protocol, part="eval_normal", category=category, allow_test=allow_test
    )
    eval_defect = select(
        manifest, protocol=protocol, part="eval_defect", category=category, allow_test=allow_test
    )
    eval_rows = eval_normal + eval_defect
    labels = np.array([0] * len(eval_normal) + [1] * len(eval_defect), dtype=np.int8)
    # 임계값은 학습에 쓰지 않은 겹 0 정상의 점수로 정한다(홀드아웃, experiments.md 2단계). Dinomaly는 학습이
    # 비싸 겹 1~4로 한 번만 학습하므로 PatchCore처럼 교차 적합을 할 수 없다. test 절차에서는 dev의
    # eval_normal(= 겹 0 정상)을 보정용으로 따로 채점하고, dev 절차에서는 겹 0 정상이 곧 평가 정상이라 따로
    # 채점하지 않는다.
    # Thresholds come from the fold-0 normals, which the model never trained on. In the dev protocol
    # those are the evaluation normals themselves, so there is nothing left to calibrate on.
    cal_rows: list[ManifestRow] = []
    if protocol == "test":
        cal_rows = select(manifest, protocol="dev", part="eval_normal", category=category)

    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    timing: dict[str, float] = {}

    t0 = time.perf_counter()
    # 보정 점수(test 절차만): 겹 0 정상을 같은 모델·같은 정밀도로 채점한다. 범주별 단계 시간도 남긴다.
    cal_score = np.empty(0, dtype=np.float32)
    if cal_rows:
        cal = predict(model, cache.images(cal_rows), batch_size=batch_size, device=device, amp=amp)
        cal_score = cal.image_scores
        del cal
    timing["cal_score_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    # 평가 세트를 채점하고 256 격자 마스크를 읽는다.
    result = predict(model, cache.images(eval_rows), batch_size=batch_size, device=device, amp=amp)
    masks = cache.masks(eval_rows).astype(bool)
    timing["eval_score_s"] = time.perf_counter() - t0

    # 맵을 float32로 올려 픽셀 AUROC, AUPRO, 부트스트랩용 PRO 히스토그램을 계산한다.
    t0 = time.perf_counter()
    maps = result.maps.astype(np.float32)
    px_auroc = pixel_auroc(maps, masks)
    au_pro = aupro(maps, masks)
    hist = pro_histograms(maps, masks, bins=PRO_BINS)
    timing["pixel_metrics_s"] = time.perf_counter() - t0

    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if cuda else 0.0
    # 공통 결과 형식: eval_score(평가 점수), cal_score·cal_images(임계값용 겹 0 정상), 픽셀 지표, PRO
    # 히스토그램. 맵은 따로 <범주>_maps.npy로 저장해 figures의 예시 패널이 필요한 행만 메모리 맵으로 읽는다.
    np.savez(
        out_dir / f"{category}.npz",
        eval_images=np.array([r.image for r in eval_rows]),
        eval_labels=labels,
        eval_defect_types=np.array([r.defect_types for r in eval_rows]),
        eval_score=result.image_scores,
        cal_score=cal_score,
        cal_images=np.array([r.image for r in cal_rows], dtype=np.str_),
        pixel_auroc=np.float64(px_auroc),
        aupro=np.float64(au_pro),
        pro_edges=hist.edges,
        pro_normal=hist.normal,
        pro_components=hist.components,
        pro_component_image=hist.component_image,
    )
    np.save(out_dir / f"{category}_maps.npy", result.maps)
    return {
        "category": category,
        "eval_normal": len(eval_normal),
        "eval_defect": len(eval_defect),
        "cal": len(cal_rows),
        "peak_vram_mb": round(peak_mb, 1),
        "timing": {k: round(v, 2) for k, v in timing.items()},
    }


# JSON을 UTF-8·LF로 쓰는 도우미.
def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


# --categories가 매니페스트에 있는 범주인지, 비어 있지 않은지, 중복이 없는지 확인한다. 중복이 있으면 학습은 그
# 이미지를 두 번 보고 평가는 범주를 두 번 적게 된다(기존 주석).
def _check_categories(
    parser: argparse.ArgumentParser, manifest: list[ManifestRow], wanted: list[str]
) -> None:
    known = {r.category for r in manifest}
    unknown = [c for c in wanted if c not in known]
    if unknown or not wanted:
        parser.error(f"unknown or missing categories {unknown}; the manifest has {sorted(known)}")
    repeated = sorted({c for c in wanted if wanted.count(c) > 1})
    if repeated:  # training would see those images twice, evaluation would list the category twice
        parser.error(f"categories named more than once: {repeated}")


# 평가 폴더에 남는 결과 파일 목록(run.json, *.npz, *_maps.npy). 다시 돌릴 때 지우거나, 끝나지 않은 실행의
# 흔적이 있는지 볼 때 쓴다.
def _result_files(out_dir: Path) -> list[Path]:
    """What an evaluation leaves in its folder: run.json, `<category>.npz` and `<category>_maps.npy`."""
    if not out_dir.is_dir():
        return []
    found = [out_dir / "run.json", *out_dir.glob("*.npz"), *out_dir.glob("*_maps.npy")]
    return sorted(p for p in found if p.is_file())


# 학습 이미지: 범주마다 dev 절차의 정상 풀(겹 1~4, 모두 4,615장)을 모은다. 겹 0은 임계값용으로 남긴다. 범주별
# 장수도 함께 돌려준다.
def _dev_pool(manifest: list[ManifestRow], categories: list[str]) -> tuple[list[ManifestRow], dict[str, int]]:
    """The normal pool of the dev protocol (folds 1-4) of `categories`: fold 0 stays out for thresholds."""
    counts = {}
    rows: list[ManifestRow] = []
    for category in categories:
        pool = select(manifest, protocol="dev", part="pool_normal", category=category)
        counts[category] = len(pool)
        rows += pool
    return rows, counts


# gate.json에서 학습할 정밀도(amp)와 어텐션을 읽고, 파일 경로·sha256·구성을 기록용으로 묶는다. 학습 기록에
# 관문 파일의 해시가 남아 어느 관문 결과로 학습했는지 추적된다.
def _read_gate(parser: argparse.ArgumentParser, path: Path) -> tuple[dict, dict]:
    """(choice, record) of a `gate` result file: the precision and attention to train with."""
    try:
        with open(path, encoding="utf-8") as f:
            gate = json.load(f)
        choice = gate["choice"]
        amp, attention = bool(choice["amp"]), str(choice["attention"])
    except (OSError, ValueError, KeyError, TypeError) as err:
        parser.error(f"{path} is not a gate result ({err})")
    if attention not in ATTENTION_MODES:
        parser.error(f"{path} chooses an unknown attention {attention!r}")
    record = {"file": str(path), "sha256": sha256_file(path), "config": gate.get("config"), **choice}
    return {"amp": amp, "attention": attention}, record


# train 명령: 구성(--config)으로 학습해 outputs/<구성>/에 model.pt, train_log.jsonl, train.json을 쓴다.
# 예(README 6단계): `run_dinomaly train --config dms-280-car --precision-from outputs/gate-dms-280/gate.json`.
# 2단계 DM은 `train --no-amp --batch-size 8`로 학습했다.
# --precision-from은 정밀도와 어텐션을 관문 결과로 정하므로 --no-amp/--attention과 같이 쓸 수 없다. 등록된 dms
# 구성은 --encoder로 인코더를 바꿀 수 없다.
def _train_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    import torch

    cfg = CONFIGS[args.config]
    out_dir = args.out or paths.OUTPUTS / cfg.name
    # --steps·--batch-size를 주지 않으면 구성 값을 쓴다.
    steps = cfg.steps if args.steps is None else args.steps
    batch_size = cfg.batch_size if args.batch_size is None else args.batch_size
    if (out_dir / "train.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished training run; pass --overwrite to redo it")
    if steps < 1 or batch_size < 1 or args.log_every < 1:
        parser.error("--steps, --batch-size and --log-every must be at least 1")
    if args.encoder is not None and args.encoder != cfg.encoder and cfg.name != NAME:
        parser.error(f"--encoder cannot change the encoder of the registered config {cfg.name!r}")
    encoder = args.encoder or cfg.encoder
    gate_record = None
    want_amp, attention = not args.no_amp, args.attention or cfg.attention
    if args.precision_from is not None:
        if args.no_amp or args.attention is not None:
            parser.error("--precision-from sets the precision and the attention: drop --no-amp / --attention")
        choice, gate_record = _read_gate(parser, args.precision_from)
        want_amp, attention = choice["amp"], choice["attention"]
    options = cfg.build_options(attention)

    # 커밋을 먼저 기록하고 매니페스트·범주·캐시를 확인한 뒤 학습 이미지(dev 정상 풀)를 고른다.
    commit = git_commit()
    manifest = read_manifest(paths.VISA_MANIFEST)
    _check_categories(parser, manifest, args.categories)
    cache = ImageCache(paths.CACHE, cfg.img_size)
    rows, counts = _dev_pool(manifest, args.categories)

    out_dir.mkdir(parents=True, exist_ok=True)
    # 이전·중단된 실행의 파일을 지운다. train.json은 끝에 쓰므로 이것이 있으면 완성된 학습이다.
    for name in ("train.json", "train_log.jsonl", "model.pt"):  # leftovers of an earlier or broken run
        (out_dir / name).unlink(missing_ok=True)

    started = time.perf_counter()
    # 학습 이미지 전체를 한 번 메모리로 읽어 두고 배치마다 인덱스로 꺼낸다.
    images = cache.images(rows)
    # 기록할 정밀도는 실제로 돌 정밀도(CPU면 fp32)다. 초기 가중치는 seed로 고정한 뒤 모델을 만든다.
    amp = amp_applied(want_amp, args.device)  # what really runs: there is no fp16 on the CPU
    torch.manual_seed(args.seed)  # initial weights of the bottleneck and the decoder
    model = build_model(encoder, **options)
    log = train(
        model,
        images,
        steps=steps,
        batch_size=batch_size,
        device=args.device,
        amp=amp,
        seed=args.seed,
        log_every=args.log_every,
        log_path=out_dir / "train_log.jsonl",
    )
    save_model(
        out_dir / "model.pt", model, encoder=encoder, steps=steps, amp=amp, config=cfg.name, options=options
    )
    # train.json의 config: 구성·인코더·입력 크기·스텝·배치·실제
    # 정밀도·seed·어텐션·옵션·옵티마이저·스케줄러·클리핑, 그리고 관문을 썼으면 관문 파일 기록. 결과를 다시
    # 만들 수 있게 학습 조건을 모두 남긴다.
    config = {
        "name": cfg.name,
        "encoder": encoder,
        "img_size": cfg.img_size,
        "steps": steps,
        "batch_size": batch_size,
        "amp": amp,
        "seed": args.seed,
        "attention": attention,
        "options": options,
        "optimizer": {"name": "StableAdamW", **OPTIMIZER},
        "scheduler": {"name": "WarmCosineScheduler", "total_iters": steps, **SCHEDULER},
        "grad_clip_norm": GRAD_CLIP,
    }
    if gate_record is not None:
        config["gate"] = gate_record
    info = {
        "method": "dinomaly",
        "commit": commit,
        "device": args.device,
        "config": config,
        "train_images": len(rows),
        "train_images_by_category": counts,
        "final_loss": log[-1]["loss"],
        "skipped_steps": log[-1]["skipped"],
        "peak_vram_mb": log[-1]["vram_mb"],
        "train_s": log[-1]["seconds"],
        "total_s": round(time.perf_counter() - started, 1),
    }
    _write_json(out_dir / "train.json", info)
    print(json.dumps(info, ensure_ascii=False), flush=True)


# eval 명령: 저장한 모델로 절차(dev/test)의 평가 세트를 채점해 outputs/<구성>-<절차>/에 공통 결과 형식을 쓴다.
# test는 --allow-test와 --stage가 있어야 하고 장부에 한 줄 남긴다. 예(README 6단계): `run_dinomaly eval
# --config dms-280-car --protocol test --allow-test --stage 6 --no-amp`.
def _eval_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.protocol == "test" and (not args.allow_test or not args.stage):
        parser.error("--protocol test needs --allow-test and --stage")
    cfg = CONFIGS[args.config]
    out_dir = args.out or paths.OUTPUTS / f"{cfg.name}-{args.protocol}"
    if (out_dir / "run.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")
    if _result_files(out_dir) and not args.overwrite:
        parser.error(f"{out_dir} holds result files of an unfinished run; pass --overwrite to replace them")
    # 모델 기본 경로는 outputs/<구성>/model.pt다.
    model_path = args.model or paths.OUTPUTS / cfg.name / "model.pt"
    if not model_path.exists():
        parser.error(f"{model_path} does not exist: run the train command first")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    commit = git_commit()
    manifest = read_manifest(paths.VISA_MANIFEST)
    _check_categories(parser, manifest, args.categories)
    model, saved = load_model(model_path)
    # 모델 파일의 구성과 --config가 다르면(예: dms 모델을 dm 설정의 입력 크기 캐시로 채점) 멈춘다.
    # A file without a config is a stage 2 model (dm).
    if saved.get("config", NAME) != cfg.name:
        parser.error(f"{model_path} holds a {saved.get('config', NAME)!r} model: pass --config to match it")
    cache = ImageCache(paths.CACHE, cfg.img_size)
    # 기본은 학습 정밀도대로 채점하지만, fp16 채점은 점수를 반올림해 임계값을 정하는 정상 점수에 동점을
    # 만든다(기존 영어 주석). 6단계는 그래서 --no-amp로 fp32 채점을 했다.
    # Score in the precision the model was trained in unless a flag says otherwise: fp16 autocast
    # rounds the image scores to fp16, which makes ties among the normals that fix the thresholds.
    want_amp = saved["amp"] if args.amp is None else args.amp
    if want_amp is None:
        parser.error(f"{model_path} does not record the precision it was trained in: pass --amp or --no-amp")
    amp = amp_applied(want_amp, args.device)
    model = model.to(args.device)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 읽는 쪽은 폴더의 모든 *.npz를 이 실행의 일부로 보므로 이전 결과를 남김없이 지운다. run.json(완성 표시)은
    # 마지막에 다시 쓴다.
    # Nothing of an earlier run may stay: readers take every *.npz of the folder as part of this run,
    # and a redo that breaks off must not look finished.
    for path in _result_files(out_dir):
        path.unlink()
    if args.protocol == "test":
        # Everything that can fail without the test images has been done: the read starts here.
        record_test_access(
            paths.TEST_LEDGER, stage=args.stage, config=cfg.name, note=args.note, commit=commit
        )

    summaries = []
    started = time.perf_counter()
    # 범주마다 채점·저장하고 요약을 한 줄씩 출력한다. run.json은 모든 범주가 끝난 뒤 마지막에 쓴다.
    for category in args.categories:
        t0 = time.perf_counter()
        info = eval_category(
            model,
            args.protocol,
            category,
            manifest,
            cache,
            args.device,
            args.allow_test,
            out_dir,
            batch_size=args.batch_size,
            amp=want_amp,
        )
        info["total_s"] = round(time.perf_counter() - t0, 1)
        summaries.append(info)
        print(json.dumps(info, ensure_ascii=False), flush=True)

    # run.json의 config: 모델 경로와 sha256(어느 모델로 채점했는지 고정), 학습 정밀도와 실제 채점 정밀도, 맵
    # 크기, 보정 방식(test는 겹 0 홀드아웃).
    config = {
        "name": cfg.name,
        "encoder": saved["encoder"],
        "img_size": cfg.img_size,
        "train_steps": saved["steps"],
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "train_amp": saved["amp"],
        "amp": amp,  # what really ran (fp16 autocast exists on CUDA only), not the flag
        "score_precision": "fp16" if amp else "fp32",
        "batch_size": args.batch_size,
        "map_size": 256,
        "calibration": "fold-0 normals (hold-out)" if args.protocol == "test" else "none",
    }
    if "options" in saved:
        config["options"] = saved["options"]
    run = {
        "method": "dinomaly",
        "protocol": args.protocol,
        "commit": commit,
        "device": args.device,
        "config": config,
        "categories": summaries,
        "total_s": round(time.perf_counter() - started, 1),
    }
    _write_json(out_dir / "run.json", run)


# 관문 검사용 build(attention): 구성은 그대로 두고 디코더 어텐션만 바꾼 새 모델을 만드는 함수를 돌려준다.
def _builder(cfg: DinomalyConfig) -> Callable[[str], torch.nn.Module]:
    """`build(attention)` for the gate checks: a fresh model of `cfg` with that decoder attention."""

    def build(attention: str) -> torch.nn.Module:
        return build_model(cfg.encoder, **cfg.build_options(attention))

    return build


# parity·gate 명령 공통 점검: 스텝·배치 크기, 범주 확인 뒤 dev 정상 풀(학습과 같은 이미지)을 돌려준다. 봉인
# 테스트는 읽지 않는다.
def _check_run(parser, args, steps_flag: int, batch_flag: int) -> list[ManifestRow]:
    if steps_flag < 1 or batch_flag < 1:
        parser.error("--steps and --batch-size must be at least 1")
    manifest = read_manifest(paths.VISA_MANIFEST)
    _check_categories(parser, manifest, args.categories)
    return _dev_pool(manifest, args.categories)[0]


# parity 명령(관문 2 (가), 기본 CPU): attention_parity 결과를 outputs/gate-<구성>/parity.json에 쓰고, 통과하지
# 못하면 종료 코드 3으로 끝내 스크립트에서 실패를 알아챌 수 있게 한다. 예: `run_dinomaly parity --config
# dms-280 --device cpu`.
def _parity_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    cfg = CONFIGS[args.config]
    out = args.out or paths.OUTPUTS / f"gate-{cfg.name}" / "parity.json"
    rows = _check_run(parser, args, args.steps, args.batch_size)
    # 학습과 같은 dev 정상 풀 이미지로 비교한다(봉인 테스트를 읽지 않음).
    images = ImageCache(paths.CACHE, cfg.img_size).images(rows)
    started = time.perf_counter()
    result = attention_parity(
        _builder(cfg),
        images,
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    record = {
        "check": "parity",
        "config": cfg.name,
        "commit": git_commit(),
        "device": args.device,
        "seed": args.seed,
        "train_images": len(rows),
        **result,
        "seconds": round(time.perf_counter() - started, 1),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out, record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if not result["passed"]:
        raise SystemExit(3)


# gate 명령(관문 2 (나), CUDA 전용): fp16 autocast는 CUDA에서만 돌므로 CPU면 거절한다. run_gate 결과를
# outputs/gate-<구성>/gate.json에 쓴다. 스텝 수는 구간(10)의 배수여야 한다.
def _gate_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not amp_applied(True, args.device):
        parser.error("the fp16 gate needs a CUDA device (fp16 autocast runs on CUDA only)")
    cfg = CONFIGS[args.config]
    out_dir = args.out or paths.OUTPUTS / f"gate-{cfg.name}"
    if (out_dir / "gate.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a gate result; pass --overwrite to redo it")
    rows = _check_run(parser, args, args.steps, args.batch_size)
    if args.steps % GATE_WINDOW:
        parser.error(f"--steps must be a multiple of {GATE_WINDOW}")
    # 옛 gate.json을 먼저 지워, 중간에 끊긴 관문이 이전 선택을 남겨 train --precision-from이 그것을 읽는 일을
    # 막는다.
    (out_dir / "gate.json").unlink(missing_ok=True)
    images = ImageCache(paths.CACHE, cfg.img_size).images(rows)
    started = time.perf_counter()
    result = run_gate(
        _builder(cfg),
        images,
        out_dir=out_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    record = {
        "check": "gate",
        "config": cfg.name,
        "commit": git_commit(),
        "device": args.device,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "train_images": len(rows),
        **result,
        "total_s": round(time.perf_counter() - started, 1),
    }
    _write_json(out_dir / "gate.json", record)
    print(json.dumps({"choice": result["choice"], "total_s": record["total_s"]}), flush=True)


# 명령줄 진입점: train, eval, parity, gate 네 하위 명령을 만들고 이름으로 처리 함수를 고른다.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    config_help = "registered configuration (docs/experiments.md)"

    # train 인자: 구성, 스텝·배치(기본은 구성 값), 정밀도(--no-amp 또는 --precision-from), 어텐션, seed(초기
    # 가중치·배치 순서·dropout을 모두 정함).
    fit = commands.add_parser("train", help="train one model on the dev normal pool of all categories")
    fit.add_argument("--config", choices=sorted(CONFIGS), default=NAME, help=config_help)
    fit.add_argument("--steps", type=int, default=None, help="default: the config's")
    fit.add_argument("--batch-size", type=int, default=None, help="default: the config's")
    fit.add_argument("--no-amp", action="store_true", help="train in fp32 instead of fp16 autocast")
    fit.add_argument("--attention", choices=ATTENTION_MODES, default=None, help="default: the config's")
    fit.add_argument(
        "--precision-from",
        type=Path,
        default=None,
        help="gate.json whose choice sets precision and attention",
    )
    fit.add_argument("--encoder", default=None, help="timm name of the DINOv2 encoder (dm only)")
    fit.add_argument("--out", type=Path, default=None, help="default: outputs/<config>")
    fit.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--seed", type=int, default=0, help="initial weights, batch order and dropout")
    fit.add_argument("--log-every", type=int, default=50)
    fit.add_argument("--overwrite", action="store_true")

    # eval 인자: --amp/--no-amp를 주지 않으면 모델이 학습된 정밀도로 채점한다. test 절차는 --allow-test와
    # --stage가 필요하다.
    score = commands.add_parser("eval", help="score the evaluation set of a protocol")
    score.add_argument("--config", choices=sorted(CONFIGS), default=NAME, help=config_help)
    score.add_argument("--protocol", choices=["dev", "test"], required=True)
    score.add_argument("--model", type=Path, default=None, help="default: outputs/<config>/model.pt")
    score.add_argument("--out", type=Path, default=None, help="default: outputs/<config>-<protocol>")
    score.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    score.add_argument("--device", default="cuda")
    score.add_argument("--batch-size", type=int, default=32)
    score.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="score under fp16 autocast (--amp) or in fp32 (--no-amp); default: as the model was trained",
    )
    score.add_argument("--allow-test", action="store_true", help="read the sealed test set (logged)")
    score.add_argument(
        "--stage", default="", help="stage label for the test ledger (required with --allow-test)"
    )
    score.add_argument("--note", default="")
    score.add_argument("--overwrite", action="store_true")

    # parity 기본값은 관문 2 (가) 규칙 그대로다: dms-280, 3스텝, 배치 8, CPU.
    parity = commands.add_parser("parity", help="gate 2a: scaled against original attention in fp32")
    parity.add_argument("--config", choices=sorted(CONFIGS), default="dms-280", help=config_help)
    parity.add_argument("--steps", type=int, default=3)
    parity.add_argument("--batch-size", type=int, default=8)
    parity.add_argument("--device", default="cpu")
    parity.add_argument("--seed", type=int, default=0)
    parity.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    parity.add_argument("--out", type=Path, default=None, help="default: outputs/gate-<config>/parity.json")

    # gate 기본값은 관문 2 (나) 규칙 그대로다: dms-280, 60스텝, 배치 16, CUDA.
    gate = commands.add_parser("gate", help="gate 2b: fp16 training against fp32 (CUDA)")
    gate.add_argument("--config", choices=sorted(CONFIGS), default="dms-280", help=config_help)
    gate.add_argument("--steps", type=int, default=GATE_STEPS)
    gate.add_argument("--batch-size", type=int, default=BATCH)
    gate.add_argument("--device", default="cuda")
    gate.add_argument("--seed", type=int, default=0)
    gate.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    gate.add_argument("--out", type=Path, default=None, help="default: outputs/gate-<config>")
    gate.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)
    commands_by_name = {
        "train": _train_command,
        "eval": _eval_command,
        "parity": _parity_command,
        "gate": _gate_command,
    }
    commands_by_name[args.command](args, parser)


if __name__ == "__main__":
    main()
