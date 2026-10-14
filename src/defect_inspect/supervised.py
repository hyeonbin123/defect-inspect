"""고정한 패치 특징 위의 지도 학습 패치 헤드: 정답 만들기, 학습, 예측.

규칙(헤드, 손실, 스케줄, 증강, 모델 선택)은 docs/experiments.md의 "2단계 / 지도 학습" 절에 고정되어 있다.
"""

# [파일 역할] 2단계 질문 1("결함 라벨 몇 장부터 지도 학습이 비지도 이상 탐지를 이기는가")의 지도 학습 쪽 모델.
# 고정한 DINOv2 ViT-S/14 패치 특징(448px 입력 → 32×32×384) 위에 작은 합성곱 헤드(SegHead)를 올려, 패치마다
# 결함인지 맞히도록 학습하고 이미지 점수·점수 맵을 낸다.
# [설계 이유] 백본은 학습하지 않고 특징을 한 번만 뽑아 두므로, 범주 12 × k 4 × 시드 3 = 144회 학습을 짧게 돌릴
# 수 있다. 헤드 구조·손실·스케줄·증강·에폭 선택은 docs/experiments.md "2단계 / 지도 학습"에 미리 고정한
# 그대로다.
# [주요 구성] SegHead(헤드) → patch_targets·pos_weight(정답과 양성 가중치) → TrainConfig·train_head(학습과
# 검증 AUROC로 에폭 선택) → predict(점수·맵) → extract(특징 추출).
# [연결] run_supervised.py가 범주마다 이 모듈을 불러 학습·채점한다. torch를 모듈 수준에서 가져오므로 기본
# 패키지(CI)에서는 import하지 않고, run_supervised가 함수 안에서 늦게 가져온다.

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

# torch: 헤드 학습(AdamW, cosine 스케줄, BCE 손실)과 GPU 추론. numpy: 마스크·정답 배열과 점수 배열.
import numpy as np
import torch
import torch.nn.functional as F

# metrics.auroc: 검증 이미지 AUROC(에폭 선택 기준). patchcore의 ScoreResult는 PatchCore와 같은 결과
# 형식(이미지 점수 + 점수 맵)을 쓰기 위해, collect_features는 같은 특징 추출 경로를 다시 쓰기 위해 가져온다.
from defect_inspect.metrics import auroc
from defect_inspect.patchcore import ScoreResult, collect_features

# GRID: 패치 격자 한 변(448px ÷ 패치 14px = 32). 정답 마스크도 이 격자로 줄인다.
GRID = 32
# `train_head` 안에서 검증 채점에 쓰는 배치 크기(`predict`의 기본값과 같다).
_VAL_BATCH = 64


# 패치 격자 위의 분할 헤드: Conv3×3(384→256)-BN-ReLU 두 번, 그다음 Conv1×1(→1)로 패치마다 결함 로짓 하나를
# 낸다. 3×3 합성곱이라 이웃 패치의 문맥도 함께 본다(규칙에 고정한 구조).
class SegHead(torch.nn.Module):
    """Conv3x3-BN-ReLU 두 번, 그다음 1x1 합성곱: 패치마다 결함 로짓 하나."""

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

    # 특징은 [B, H, W, D](채널 마지막)로 저장돼 있으므로 Conv2d가 받는 [B, D, H, W]로 바꿔 넣고, 채널 축 1개를
    # 없애 [B, H, W] 로짓으로 돌려준다.
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """채널 마지막 특징 [B, H, W, D] -> 로짓 [B, H, W]."""
        if feats.ndim != 4:
            raise ValueError(f"expected features [B, H, W, D], got {tuple(feats.shape)}")
        return self.net(feats.permute(0, 3, 1, 2)).squeeze(1)


# 픽셀 마스크를 패치 정답으로 바꾼다. 256 격자 마스크를 8×8 블록으로 나눠 픽셀 하나라도 결함이면 그 패치를
# 결함으로 본다(= 8×8 최대 풀링, 규칙의 "정답" 항목). 작은 결함이 축소로 사라지지 않게 any를 쓴다.
# 호출 위치: run_supervised.run_category(정상 학습 이미지와 라벨 풀).
def patch_targets(masks: np.ndarray, grid: int = GRID) -> np.ndarray:
    """{0, 1} 마스크 [N, S, S] -> bool [N, grid, grid]: 픽셀 하나라도 결함이면 그 패치는 결함이다."""
    masks = np.asarray(masks)
    if masks.ndim != 3 or masks.shape[1] != masks.shape[2]:
        raise ValueError(f"expected square masks [N, S, S], got {masks.shape}")
    n, side = masks.shape[0], masks.shape[1]
    if grid < 1 or side % grid != 0:
        raise ValueError(f"grid {grid} does not divide the mask side {side}")
    block = side // grid
    defect = masks if masks.dtype == np.bool_ else masks != 0
    # [N, S, S]를 [N, 격자, 블록, 격자, 블록]으로 다시 보고 블록 축(2, 4)에서 any를 취하면 반복문 없이 블록별
    # 최대 풀링이 된다.
    return defect.reshape(n, grid, block, grid, block).any(axis=(2, 4))


# BCE 손실의 양성(결함 패치) 가중치 = min(음성 패치 수 ÷ 양성 패치 수, 상한 100). 결함 패치는 전체의 아주 작은
# 비율이라 그대로 두면 헤드가 "전부 정상"으로 기운다. 비율만큼 결함 패치를 키우되, 결함이 아주 작을 때
# 가중치가 폭주하지 않게 100에서 자른다(규칙의 손실 항목). 결함 패치가 없으면 1.0이다.
def pos_weight(targets: np.ndarray, max_pos_weight: float) -> float:
    """min(음성 패치 수 / 양성 패치 수, max_pos_weight). 양성 패치가 없으면 1.0."""
    targets = np.asarray(targets)
    n_pos = int(np.count_nonzero(targets))
    if n_pos == 0:
        return 1.0
    return float(min((targets.size - n_pos) / n_pos, max_pos_weight))


# 학습 설정(규칙에 고정한 값): 60에폭, 배치 32, AdamW 학습률 1e-3·weight decay 1e-4, 검증은 5에폭마다, 양성
# 가중치 상한 100. seed는 run_supervised가 실행(k, 시드)마다 바꿔 넣는다.
@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    eval_every: int = 5
    max_pos_weight: float = 100.0
    seed: int = 0


# train_head의 결과: 가장 좋은 에폭의 가중치를 실은 헤드(eval 모드), 그 에폭 번호(1부터)와 검증 AUROC, 에폭별
# 손실·검증 AUROC 기록.
@dataclass
class TrainResult:
    head: SegHead  # 가장 좋은 상태를 실은 eval 모드 헤드
    best_epoch: int  # 1부터 센다
    best_val_auroc: float
    history: list[dict]  # 에폭마다 {"epoch", "loss", "val_auroc"} 하나. 검증하지 않은 에폭은 val_auroc가 None


# 한 에폭의 배치 수. 마지막 배치가 한 장뿐이면 버린다(_epoch_batches와 같은 규칙, 배치 크기가 1이면 버리지
# 않음). train_head가 cosine 스케줄의 전체 스텝 수(T_max)를 실제 스텝 수와 정확히 맞추려고 따로 계산한다.
def steps_per_epoch(n_images: int, batch_size: int) -> int:
    """한 에폭의 배치 수: 마지막 덜 찬 배치는 이미지가 한 장뿐일 때만 버린다."""
    full, partial = divmod(n_images, batch_size)
    return full + (1 if partial > 1 else 0)


# 증강: 특징 격자 단위로 이미지마다 좌우·상하 뒤집기(규칙). 이미지가 아니라 이미 뽑은 특징을 뒤집으므로 백본을
# 다시 돌리지 않아도 된다. 특징과 정답을 같은 인덱스 한 번의 gather로 뽑아 둘이 어긋날 수 없게 한다.
def _flipped_batch(
    feats: torch.Tensor, targets: torch.Tensor, index: torch.Tensor, flips: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`feats` [N, H, W, D]와 `targets` [N, H, W]에서 `index` 행을 골라 이미지마다 뒤집는다.

    `flips`는 bool [B, 2]: 열 0은 좌우(W 축), 열 1은 상하(H 축) 뒤집기다. 텐서마다 같은 인덱스로 gather를 한
    번씩 하므로 특징과 정답이 늘 함께 움직인다.
    """
    h, w = feats.shape[1], feats.shape[2]
    rows = torch.arange(h).expand(len(index), h)
    cols = torch.arange(w).expand(len(index), w)
    # 뒤집을 이미지는 행(또는 열) 인덱스를 h-1-i로 거꾸로 바꾼다. 아래 pick은 (이미지, 행, 열) 인덱스를
    # 브로드캐스트해 [B, H, W]를 한 번에 고르는 고급 인덱싱이다.
    rows = torch.where(flips[:, 1:2], h - 1 - rows, rows)
    cols = torch.where(flips[:, 0:1], w - 1 - cols, cols)
    # 인덱스는 CPU에서 뽑고, gather는 특징이 있는 장치에서 한다.
    index, rows, cols = index.to(feats.device), rows.to(feats.device), cols.to(feats.device)
    pick = (index[:, None, None], rows[:, :, None], cols[:, None, :])
    return feats[pick], targets[pick]


# 한 에폭의 배치를 차례로 내준다. 순서(randperm)와 뒤집기(각 0.5 확률)를 모두 같은 generator에서 뽑으므로
# 시드가 같으면 같은 순서·같은 증강이 나온다.
def _epoch_batches(
    feats: torch.Tensor, targets: torch.Tensor, batch_size: int, generator: torch.Generator
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """한 에폭: 이미지 순열을 배치로 자르고, 이미지마다 무작위로 뒤집는다.

    순서와 뒤집기(각각 확률 0.5, 서로 독립)를 모두 `generator`에서 뽑는다.
    """
    n = feats.shape[0]
    order = torch.randperm(n, generator=generator)
    for start in range(0, n, batch_size):
        index = order[start : start + batch_size]
        if len(index) == 1 and batch_size > 1:
            break  # 한 장뿐인 마지막 배치는 버린다
        flips = torch.rand((len(index), 2), generator=generator) < 0.5
        yield _flipped_batch(feats, targets, index, flips)


# 검증용 이미지 점수: 패치 로짓의 최댓값(규칙의 이미지 점수 정의). eval 모드로 바꿔 BatchNorm이 학습 중 쌓은
# 이동 평균을 쓰게 하고, 기울기를 계산하지 않는다.
@torch.no_grad()
def _max_logits(head: SegHead, feats: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    """eval 모드 헤드로 낸 이미지 점수(가장 큰 패치 로짓, float32)."""
    head.eval()
    scores = np.empty(feats.shape[0], dtype=np.float32)
    for start in range(0, feats.shape[0], batch_size):
        batch = feats[start : start + batch_size].to(device).to(torch.float32)
        scores[start : start + len(batch)] = head(batch).flatten(1).amax(dim=1).cpu().numpy()
    return scores


# 학습 전에 모양을 모두 확인한다: 특징 4차원, 정답이 특징 격자와 같은 모양, 검증 특징이 학습 특징과 같은
# 격자·차원, 검증 라벨 수, 설정값이 1 이상. 학습을 몇 분 돌린 뒤가 아니라 시작 전에 멈추게 한다.
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


# 헤드 하나를 학습하고 검증 이미지 AUROC가 가장 높은 에폭의 가중치를 고른다. 입력은 학습 특징·정답(정상 겹 1~4
# + 결함 k장)과 검증 특징·라벨(겹 0 정상 + dev 결함 20장). 검증은 5에폭마다와 마지막 에폭에 한다.
# 호출 위치: run_supervised.run_category(실행마다 한 번).
def train_head(
    train_feats: torch.Tensor,
    train_targets: np.ndarray,
    val_feats: torch.Tensor,
    val_labels: np.ndarray,
    cfg: TrainConfig,
    *,
    device: str | torch.device = "cuda",
) -> TrainResult:
    """패치 특징으로 `SegHead`를 학습하고 검증 이미지 AUROC가 가장 좋은 에폭을 남긴다.

    `train_feats` [N, H, W, D]와 `val_feats` [M, H, W, D]는 CPU fp16 텐서다. CUDA면 장치로 한 번만 옮기고(범주
    하나가 fp16으로 수백 MB) 배치는 장치 안에서 고른다. 스텝마다 호스트에서 장치로 복사하면 실행 하나가 몇
    초가 아니라 몇 분 걸렸다. 배치는 fp32로 바꿔 계산한다. `train_targets`는 bool [N, H, W], `val_labels`는
    0/1 [M]이다. 검증 AUROC는 `cfg.eval_every` 에폭마다와 마지막 에폭에 재고, 뒤의 에폭이 지금까지의 최고와
    같거나 더 높으면 그 에폭으로 바꾼다(동점이면 뒤의 에폭, docs/experiments.md 규칙 변경 4 (가)).

    재현성: 순서, 뒤집기, 초기 가중치는 `cfg.seed`에만 달려 있다. CPU에서는 입력과 시드가 같고 스레드 수
    (`torch.get_num_threads()`)가 같으면 비트 단위로 같은 결과가 나온다. 스레드 수가 다르거나 GPU면 계산이
    부동소수 반올림 수준에서 달라지고 학습이 그 차이를 키운다(검증 AUROC가 소수 셋째 자리에서 움직이고, 그에
    따라 고른 에폭도 바뀔 수 있다). 그때 결과는 비교할 만하지만 똑같지는 않다.
    """
    device = torch.device(device)
    train_targets = np.asarray(train_targets)
    val_labels = np.asarray(val_labels)
    _check_training_inputs(train_feats, train_targets, val_feats, val_labels, cfg)
    # 검증 세트에 정상과 결함이 둘 다 있어야 AUROC가 정의된다. 학습 장수가 1장 이하면 스텝이 0이 되므로 역시
    # 멈춘다.
    is_defect = val_labels != 0
    if is_defect.all() or not is_defect.any():
        raise ValueError("the validation set needs both normal and defect images")
    n_steps = cfg.epochs * steps_per_epoch(train_feats.shape[0], cfg.batch_size)
    if n_steps == 0:
        raise ValueError("training needs at least two images")

    # 양성 가중치는 학습 세트 전체의 패치 비율로 한 번만 정한다(배치마다 바꾸지 않음).
    weight = torch.tensor(pos_weight(train_targets, cfg.max_pos_weight), dtype=torch.float32, device=device)
    targets = torch.from_numpy(np.ascontiguousarray(train_targets != 0))
    # GPU면 특징 전체를 한 번만 옮겨 두고 배치는 GPU 안에서 고른다. docstring대로 스텝마다 호스트→GPU 복사를
    # 하면 실행이 몇 초에서 몇 분으로 늘었다.
    if device.type == "cuda":
        train_feats, targets, val_feats = train_feats.to(device), targets.to(device), val_feats.to(device)
    # 순서와 뒤집기는 CPU 생성기에서 뽑으므로 어느 장치에서든 같다.
    generator = torch.Generator().manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)  # 가중치 초기화용
    # AdamW + cosine 감소(전체 스텝 n_steps 동안 학습률을 0까지 줄임, 스텝마다 scheduler.step). 규칙의 최적화
    # 설정 그대로다.
    head = SegHead(dim=int(train_feats.shape[-1])).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=0.0)

    best_epoch, best_auroc, best_state = 0, float("-inf"), None
    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        head.train()
        loss_sum, seen = 0.0, 0
        for feats, target in _epoch_batches(train_feats, targets, cfg.batch_size, generator):
            # fp16으로 옮긴 뒤 장치에서 fp32로 바꾼다: 건너가는 바이트가 절반이다.
            feats = feats.to(device).to(torch.float32)
            target = target.to(device).to(torch.float32)
            loss = F.binary_cross_entropy_with_logits(head(feats), target, pos_weight=weight)
            # 손실이 NaN·inf가 되면 그 뒤의 학습과 에폭 선택이 모두 무의미하므로 즉시 예외로 멈춘다(조용히
            # 이상한 헤드를 저장하지 않게).
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss in epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # 에폭 손실은 배치 크기로 가중한 평균이다(마지막 배치가 작아도 공정하게).
            loss_sum += loss.item() * len(feats)
            seen += len(feats)
        entry: dict = {"epoch": epoch, "loss": loss_sum / seen, "val_auroc": None}
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            scores = _max_logits(head, val_feats, _VAL_BATCH, device)
            value = float(auroc(scores[~is_defect], scores[is_defect]))
            entry["val_auroc"] = value
            # 같거나 더 높으면 뒤의 에폭으로 바꾼다(>=). 쉬운 범주에서는 검증 AUROC가 5에폭 만에 1.0에 닿아 그
            # 뒤가 전부 동점이 되는데, 앞선 에폭을 고르면 덜 학습된 헤드가 뽑혀 지도 학습 쪽이
            # 불리해진다(docs/experiments.md 규칙 변경 4 (가)). 가중치는 clone으로 복사해 이후 학습이 덮어쓰지
            # 않게 한다.
            if value >= best_auroc:  # 동점이면 뒤의 에폭을 남긴다(규칙 변경 4)
                best_epoch, best_auroc = epoch, value
                best_state = {name: tensor.detach().clone() for name, tensor in head.state_dict().items()}
        history.append(entry)

    # 검증 AUROC가 한 번도 숫자가 아니었으면(NaN과의 비교는 늘 거짓) 고를 에폭이 없으므로 멈춘다.
    if best_state is None:
        raise FloatingPointError("the validation AUROC was never a number")
    head.load_state_dict(best_state)
    head.eval()
    return TrainResult(head=head, best_epoch=best_epoch, best_val_auroc=best_auroc, history=history)


# 학습한 헤드로 특징을 채점해 ScoreResult(이미지 점수, 256×256 점수 맵)를 돌려준다. 맵을 확률이 아니라
# 로짓으로 두는 이유는 docstring대로 float16 확률이 로짓 약 8.3 이상에서 1.0으로 포화해 픽셀 지표에 동점이
# 생기기 때문이다(규칙 변경 4 (나)). 로짓과 확률은 순서가 같아 순위 기반 지표의 정의는 같다.
# 호출 위치: run_supervised.run_category(평가 세트와, test 절차의 임계값용 겹 0 정상).
@torch.no_grad()
def predict(
    head: SegHead,
    feats: torch.Tensor,
    *,
    batch_size: int = 64,
    device: str | torch.device = "cuda",
    map_size: int = 256,
) -> ScoreResult:
    """학습한 헤드(eval 모드로 바꿈)로 패치 특징 [N, H, W, D]를 채점한다.

    이미지 점수 = 가장 큰 패치 로짓(float32). 맵 = 로짓을 `map_size`로 bilinear 확대(align_corners=False)한
    float16. 확률이 아니라 로짓을 쓴다: float16 확률은 로짓 약 8.3 이상에서 1.0으로 포화해, 확신하는 픽셀이
    모두 동점이 되기 때문이다(규칙 변경 4).
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
        # 32×32 로짓을 픽셀 지표용 256 격자로 bilinear 확대한다(모든 구성의 픽셀 지표를 256×256 격자로
        # 계산하는 공통 규칙). 메모리를 줄이려고 float16으로 저장한다.
        batch_maps = F.interpolate(
            logits.unsqueeze(1),
            size=(map_size, map_size),
            mode="bilinear",
            align_corners=False,
        )
        maps[start : start + len(batch)] = batch_maps[:, 0].to(torch.float16).cpu().numpy()
    # 점수에 NaN·inf가 섞이면 AUROC와 임계값이 조용히 틀어지므로 결과를 돌려주기 전에 멈춘다.
    if not np.isfinite(image_scores).all():
        raise FloatingPointError("the head produced non-finite logits")
    return ScoreResult(image_scores, maps)


# 이미지(uint8 [N, S, S, 3])에서 백본 패치 특징을 뽑아 [N, H, W, D] 모양의 CPU fp16 텐서로 돌려준다.
# patchcore.collect_features가 [N·H·W, D]로 펼쳐 주는 것을 이미지·격자 모양으로 되돌릴 뿐이라 추가 복사가
# 없다(view).
# 호출 위치: run_supervised.run_category, tests/test_run_supervised.py.
def extract(
    extractor: torch.nn.Module,
    images: np.ndarray,
    *,
    batch_size: int = 32,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """uint8 이미지 [N, S, S, 3] -> CPU fp16 패치 특징 [N, H, W, D]."""
    feats, (grid_h, grid_w) = collect_features(extractor, images, batch_size=batch_size, device=device)
    return feats.view(len(images), grid_h, grid_w, feats.shape[1])
