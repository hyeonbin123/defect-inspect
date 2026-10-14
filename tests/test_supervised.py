# [파일 역할] 2단계 지도 학습 헤드 모듈 `defect_inspect.supervised`를 검증하는 테스트 묶음이다. 헤드
# 구조(SegHead), 패치 정답(patch_targets), 양성 가중치, 배치 나누기와 뒤집기 증강, 학습 루프(train_head: 시드,
# 최적화기, 손실, 에폭 선택), 예측(predict: 이미지 점수와 맵), 특징 추출(extract)을 확인한다.
# [왜 필요한가] 지도 학습의 규칙(헤드, 손실, 일정, 증강, 모델 선택)은 측정 전에 docs/experiments.md '2단계 /
# 지도 학습'과 규칙 변경 4에 고정했다. 이 파일은 그 규칙을 한 줄씩 코드로 확인해, 결과(결함 몇 장부터 지도
# 학습이 이기는가)가 적어 둔 절차로 나왔음을 보장한다.
# [설계 메모] DINOv2 특징 대신 작은 합성 특징(8x8 격자, 6차원 잡음에 결함 칸만 채널 0을 키움)과 손으로
# 가중치를 정한 헤드를 써서 CPU에서 빠르게, 기대값을 손으로 계산할 수 있게 한다.

import math

import numpy as np
import pytest

# supervised는 torch를, 그 import 경로의 backbones는 torchvision을 모듈 수준에서 쓰므로 둘 중 하나라도 없으면
# 파일 전체를 건너뛴다(CI 기본 환경).
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch.nn.functional as F  # noqa: E402

# backbones의 ImageNet 평균·표준편차는 특징 추출 기대값을 손으로 계산하는 데, metrics.auroc는 검증 AUROC를
# 다시 계산하는 데, patchcore의 ScoreResult·collect_features는 예측 결과 형식과 특징 추출 경로를 대조하는 데
# 쓴다.
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

# 합성 데이터의 패치 격자(8x8)와 특징 차원(6).
G, DIM = 8, 6


# 결함을 '심은' 합성 특징을 만든다. 정상은 순수 잡음이고, 결함 이미지는 무작위 위치의 2x2 칸에서 채널 0만
# strength만큼 커진다. 정답 위치를 알고 있으므로 학습이 결함을 찾는지, 맵이 그 자리에서 가장 높은지를 확인할
# 수 있다.
def planted(n_normal: int, n_defect: int, seed: int, strength: float = 6.0):
    """가우스 잡음 특징. 결함 이미지는 2x2 칸 하나에서 채널 0에 `strength`를 더한다.

    (fp16 특징 [N, G, G, DIM], bool 정답 [N, G, G], 라벨 [N])을 돌려주며, 정상이 먼저 온다.
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


# 테스트용 짧은 학습 구성(10에폭, 배치 16, 5에폭마다 검증). 규칙의 기본값은 60에폭·배치 32다.
FAST = TrainConfig(epochs=10, batch_size=16, eval_every=5, seed=0)


# 합성 학습 세트(시드 1)와 검증 세트(시드 2, 정상 12 + 결함 6)로 CPU에서 헤드를 학습해 결과와 검증 데이터를
# 돌려준다.
def fit(cfg: TrainConfig = FAST, n_normal: int = 24, n_defect: int = 8):
    train_feats, train_targets, _ = planted(n_normal, n_defect, seed=1)
    val_feats, _, val_labels = planted(12, 6, seed=2)
    result = train_head(train_feats, train_targets, val_feats, val_labels, cfg, device="cpu")
    return result, val_feats, val_labels


# 가중치를 손으로 정한 헤드: 채널 0을 ReLU 두 번 거쳐 그대로 내보내고 bias를 더한다. 학습 없이 로짓을 정확히
# 알 수 있어 predict의 점수·맵 정의를 숫자로 확인하는 데 쓴다.
def pass_through_head(dim: int, bias: float = -1.0) -> SegHead:
    """손으로 가중치를 정한 평가 모드 헤드: logit = relu(채널 0) / (1 + eps) + bias."""
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


# pass_through_head가 내야 하는 로짓의 기대값(numpy로 직접 계산).
def pass_through_logits(feats: torch.Tensor, bias: float = -1.0) -> np.ndarray:
    # 새 통계(평균 0, 분산 1)를 가진 평가 모드 BatchNorm 두 층이 각각 sqrt(1 + 1e-5)로 나눈다.
    return np.maximum(feats[..., 0].double().numpy(), 0.0) / (1.0 + 1e-5) + bias


# --- 헤드 ---------------------------------------------------------------------------------------


# 헤드 구조가 등록 규칙과 같은지: Conv3x3(384→256)-BN-ReLU-Conv3x3(256→256)-BN-ReLU-Conv1x1(256→1), 패딩
# 1(격자 크기 유지). 정답 격자는 32x32(GRID)다.
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


# 헤드가 채널이 마지막인 특징 [B, H, W, D]를 받아 격자 모양 그대로 로짓 [B, H, W]를 내는지 본다. H와 W를
# 다르게 해 축이 바뀌면 드러나게 하고, 3x3 합성곱 두 번이라 한 패치의 변화가 두 칸 밖까지만 퍼지며 다른
# 이미지에는 영향이 없는지도 확인한다. 4차원이 아닌 입력은 ValueError다.
def test_seg_head_takes_channels_last_and_keeps_the_grid():
    torch.manual_seed(0)
    head = SegHead(dim=5, hidden=7).eval()
    feats = torch.randn(2, 3, 6, 5)  # H != W라 축이 뒤바뀌면 드러난다
    with torch.no_grad():
        logits = head(feats)
        assert logits.shape == (2, 3, 6)
        assert torch.equal(logits, head.net(feats.permute(0, 3, 1, 2))[:, 0])
        # 3x3 합성곱 두 번: 한 패치는 최대 두 칸 떨어진 로짓까지만 영향을 준다.
        changed = feats.clone()
        changed[0, 1, 5] += 3.0
        diff = (head(changed) - logits).abs()
    assert float(diff[0, 1, 5]) > 0
    assert float(diff[0, :, :3].max()) == 0.0
    assert float(diff[1].max()) == 0.0
    with pytest.raises(ValueError):
        head(torch.zeros(3, 6, 5))


# 다른 테스트가 기대는 손 설정 헤드가 실제로 기대 로짓을 내는지 먼저 확인한다(도구 자체의 검증).
def test_pass_through_head_is_what_the_tests_assume():
    feats = torch.randn(3, 4, 4, 2, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        logits = pass_through_head(2)(feats)
    np.testing.assert_allclose(logits.numpy(), pass_through_logits(feats), rtol=1e-5, atol=1e-6)


# --- 정답 ---------------------------------------------------------------------------------------


# 패치 정답이 256 마스크를 8x8 블록으로 최대 풀링한 32x32라는 규칙을 확인한다: 픽셀 하나만 결함이어도 그
# 패치는 결함이고, 블록 경계의 좌표가 맞는 패치로 가며, 결과가 torch max_pool2d와 같다.
def test_patch_targets_is_a_max_pool():
    masks = np.zeros((3, 256, 256), dtype=np.uint8)
    masks[0, 0, 0] = 1  # 픽셀 하나만으로도 그 패치는 결함이다
    masks[1, 7, 8] = 1  # 행 7은 여전히 패치 행 0, 열 8은 패치 열 1
    masks[1, 255, 255] = 1
    masks[2, 100:120, 30:41] = 1  # 행 96..127 -> 패치 12..14, 열 24..47 -> 패치 3..5
    targets = patch_targets(masks)
    assert targets.shape == (3, 32, 32) and targets.dtype == np.bool_
    assert np.argwhere(targets[0]).tolist() == [[0, 0]]
    assert np.argwhere(targets[1]).tolist() == [[0, 1], [31, 31]]
    expected = np.zeros((32, 32), dtype=bool)
    expected[12:15, 3:6] = True
    np.testing.assert_array_equal(targets[2], expected)
    pooled = F.max_pool2d(torch.from_numpy(masks).float()[:, None], kernel_size=8)[:, 0].numpy() > 0
    np.testing.assert_array_equal(targets, pooled)


# 다른 격자 크기에서도 최대 풀링과 같고, bool·0/1·0보다 큰 임의 값 마스크가 같은 결과를 낸다. 빈 입력은 빈
# 격자, 정사각형이 아니거나 격자가 변을 나누지 못하면 ValueError다.
def test_patch_targets_grid_dtypes_and_errors():
    rng = np.random.default_rng(0)
    masks = (rng.random((4, 256, 256)) < 0.001).astype(np.uint8)
    for grid in (4, 16, 64, 256):
        pooled = F.max_pool2d(torch.from_numpy(masks).float()[:, None], kernel_size=256 // grid)[:, 0]
        np.testing.assert_array_equal(patch_targets(masks, grid), pooled.numpy() > 0)
    np.testing.assert_array_equal(patch_targets(masks.astype(bool)), patch_targets(masks))
    # 0보다 큰 값은 모두 결함으로 센다.
    np.testing.assert_array_equal(patch_targets(masks * 7), patch_targets(masks))
    assert patch_targets(np.zeros((0, 256, 256), dtype=np.uint8)).shape == (0, 32, 32)
    with pytest.raises(ValueError):
        patch_targets(np.zeros((256, 256), dtype=np.uint8))
    with pytest.raises(ValueError):
        patch_targets(np.zeros((1, 256, 128), dtype=np.uint8))
    with pytest.raises(ValueError):
        patch_targets(np.zeros((1, 256, 256), dtype=np.uint8), grid=30)


# 양성 가중치 = min(학습 세트 전체의 음성 패치 수 / 양성 패치 수, 상한). 양성이 없으면 1.0, 모두 양성이면
# 0이다(규칙: min(음성 ÷ 양성, 100)).
def test_pos_weight_is_the_capped_ratio_over_the_whole_set():
    targets = np.zeros((4, 8, 8), dtype=bool)
    assert pos_weight(targets, 100.0) == 1.0  # 양성 패치 없음
    targets[0, :2, :4] = True  # 양성 8, 음성 248
    assert pos_weight(targets, 100.0) == pytest.approx(248 / 8)
    assert pos_weight(targets, 10.0) == 10.0
    targets[:] = True
    assert pos_weight(targets, 100.0) == 0.0


# --- 배치와 뒤집기 ------------------------------------------------------------------------------


# 에폭당 배치 수: 마지막 배치가 한 장뿐일 때만 버리고, 두 장 이상이면 남긴다. 배치 크기 1이면 버릴 것이 없다.
# 이 값이 코사인 일정의 전체 스텝 수가 된다.
def test_steps_per_epoch_drops_a_single_image_batch_only():
    assert steps_per_epoch(64, 32) == 2
    assert steps_per_epoch(65, 32) == 2  # 혼자 남은 65번째 이미지는 버린다
    assert steps_per_epoch(66, 32) == 3
    assert steps_per_epoch(20, 32) == 1
    assert steps_per_epoch(1, 32) == 0
    assert steps_per_epoch(3, 1) == 3  # 배치 크기 1에는 부분 배치가 없다


# 채널에 (이미지 번호, 행, 열)을 적은 특징과 (0, 0) 칸만 표시한 정답을 만든다. 뒤집기·섞기 뒤에 각 값이 어디서
# 왔는지 읽어 낼 수 있게 하려는 것이다.
def coordinate_features(n: int, h: int, w: int):
    """채널이 (이미지 번호, 행, 열)인 fp16 특징. 정답은 (0, 0) 칸을 표시한다."""
    feats = torch.zeros(n, h, w, 3)
    feats[..., 0] = torch.arange(n).view(n, 1, 1)
    feats[..., 1] = torch.arange(h).view(1, h, 1)
    feats[..., 2] = torch.arange(w).view(1, 1, w)
    targets = torch.zeros(n, h, w, dtype=torch.bool)
    targets[:, 0, 0] = True
    return feats.to(torch.float16), targets


# _flipped_batch가 고른 이미지를 뽑아 flips의 열 0이면 좌우(W), 열 1이면 상하(H)로 뒤집고, 특징과 정답을 같은
# 인덱스로 함께 옮기는지 확인한다. 입력 텐서는 바뀌지 않는다.
def test_flipped_batch_flips_the_named_axis_of_both_tensors():
    feats, targets = coordinate_features(4, 2, 3)
    index = torch.tensor([3, 0, 2, 1])
    flips = torch.tensor([[False, False], [True, False], [False, True], [True, True]])
    out, out_targets = supervised._flipped_batch(feats, targets, index, flips)
    assert out.shape == (4, 2, 3, 3) and out.dtype == torch.float16
    assert out_targets.shape == (4, 2, 3) and out_targets.dtype == torch.bool
    assert out[:, 0, 0, 0].tolist() == [3, 0, 2, 1]
    # `flips`의 열 0은 좌우(W), 열 1은 상하(H)로 뒤집는다.
    assert torch.equal(out[0], feats[3])
    assert torch.equal(out[1], feats[0].flip(1))
    assert torch.equal(out[2], feats[2].flip(0))
    assert torch.equal(out[3], feats[1].flip(0).flip(1))
    assert np.argwhere(out_targets.numpy()).tolist() == [[0, 0, 0], [1, 0, 2], [2, 1, 0], [3, 1, 2]]
    # 입력은 바뀌지 않는다.
    again, _ = coordinate_features(4, 2, 3)
    assert torch.equal(feats, again)


# 한 에폭의 배치: 401장을 16장씩 나누면 마지막 한 장은 버려진다. 정답이 간 자리에 원래 (0, 0) 칸의 특징도 함께
# 가고(특징·정답 어긋남 없음), 이미지끼리 섞이지 않으며, 순서는 섞인다. 좌우·상하 뒤집기는 각각 확률 0.5로
# 독립이라 네 조합이 고르게 나온다.
# 같은 생성기 상태면 같은 에폭이 다시 나오고, 같은 생성기로 다음 에폭을 뽑으면 달라진다.
def test_epoch_batches_keep_features_and_targets_together():
    n, h, w = 401, 2, 3
    feats, targets = coordinate_features(n, h, w)
    generator = torch.Generator().manual_seed(0)
    batches = list(supervised._epoch_batches(feats, targets, 16, generator))
    assert [len(f) for f, _ in batches] == [16] * 25  # 401 = 25 * 16 + 1: 한 장 남은 것은 버린다
    out = torch.cat([f for f, _ in batches]).float()
    out_targets = torch.cat([t for _, t in batches])
    # 정답이 어디로 가든 원래 (0, 0) 칸의 특징도 그 자리에 있다.
    assert torch.equal(out_targets, (out[..., 1] == 0) & (out[..., 2] == 0))
    ids = out[:, 0, 0, 0].long().tolist()
    assert len(set(ids)) == 400 and ids != sorted(ids)
    assert bool((out[..., 0] == out[:, :1, :1, 0]).all())  # 다른 이미지와 섞인 이미지가 없다
    h_flip = (out[:, 0, 0, 2] == w - 1).numpy()
    v_flip = (out[:, 0, 0, 1] == h - 1).numpy()
    # 각각 확률 0.5로 독립적으로 뽑는다: 네 조합이 거의 같은 빈도로 나온다.
    for count in (h_flip.sum(), v_flip.sum()):
        assert 160 <= count <= 240
    for a in (False, True):
        for b in (False, True):
            assert 60 <= np.count_nonzero((h_flip == a) & (v_flip == b)) <= 140

    # 같은 생성기 상태면 같은 에폭이 나오고, 같은 생성기의 다음 에폭은 다르다.
    repeat = list(supervised._epoch_batches(feats, targets, 16, torch.Generator().manual_seed(0)))
    assert len(repeat) == len(batches)
    for (feats_a, targets_a), (feats_b, targets_b) in zip(batches, repeat, strict=True):
        assert torch.equal(feats_a, feats_b) and torch.equal(targets_a, targets_b)
    second = list(supervised._epoch_batches(feats, targets, 16, generator))
    assert not torch.equal(second[0][0], batches[0][0])


# 두 장 이상인 마지막 부분 배치는 남기고, 배치 수가 steps_per_epoch와 같으며, 모든 이미지가 한 번씩 나온다.
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


# --- 학습 ---------------------------------------------------------------------------------------


# 순서와 뒤집기가 구성의 seed로 만든 CPU 생성기 하나에서 에폭 내내 이어져 나오는지 확인한다. CPU 생성기를
# 쓰므로 장치가 달라도 순서와 뒤집기가 같다(train_head 주석).
def test_order_and_flips_come_from_one_generator_seeded_with_the_config(monkeypatch):
    seen = []
    real = supervised._epoch_batches

    def spy(feats, targets, batch_size, generator):
        seen.append((batch_size, generator.initial_seed(), generator.device.type, id(generator)))
        return real(feats, targets, batch_size, generator)

    monkeypatch.setattr(supervised, "_epoch_batches", spy)
    fit(TrainConfig(epochs=3, batch_size=16, eval_every=5, seed=7))
    assert [entry[:3] for entry in seen] == [(16, 7, "cpu")] * 3
    assert len({entry[3] for entry in seen}) == 1  # 에폭 내내 생성기 하나를 이어 쓴다


# 같은 시드면 학습 기록, 고른 에폭, 가중치가 비트 단위로 같고, 전역 난수 상태와 무관하다. 다른 시드면
# 초기화·순서·뒤집기가 달라진다(CPU·같은 스레드 수에서의 재현성, train_head docstring).
def test_training_is_deterministic_for_a_seed():
    first, _, _ = fit()
    torch.randn(5)  # 두 실행 사이 전역 생성기의 상태는 상관없다
    second, _, _ = fit()
    assert first.history == second.history
    assert (first.best_epoch, first.best_val_auroc) == (second.best_epoch, second.best_val_auroc)
    state_a, state_b = first.head.state_dict(), second.head.state_dict()
    assert all(torch.equal(state_a[name], state_b[name]) for name in state_a)
    # 다른 시드는 다른 초기화, 순서, 뒤집기를 낸다.
    other, _, _ = fit(TrainConfig(epochs=10, batch_size=16, eval_every=5, seed=1))
    assert not torch.equal(other.head.net[0].weight, first.head.net[0].weight)
    assert other.history != first.history


# 학습이 실제로 작동하는지: 심은 결함을 검증 AUROC 0.98 이상으로 가르고 손실이 절반 아래로 준다. 저장된
# 특징(fp16, CPU)은 뒤집기나 형 변환으로 바뀌지 않는다. 돌려받은 헤드가 보고된 검증 AUROC를 그대로 재현하고,
# 보지 못한 이미지도 가른다.
def test_head_separates_planted_defects():
    train_feats, train_targets, _ = planted(24, 8, seed=1)
    before = train_feats.clone()
    val_feats, _, val_labels = planted(12, 6, seed=2)
    result = train_head(train_feats, train_targets, val_feats, val_labels, FAST, device="cpu")
    assert isinstance(result.head, SegHead) and not result.head.training
    assert result.head.net[0].in_channels == DIM and result.head.net[0].out_channels == 256
    assert result.best_val_auroc >= 0.98
    assert result.history[-1]["loss"] < 0.5 * result.history[0]["loss"]
    # 저장된 특징(fp16, CPU)은 뒤집기나 형 변환으로 바뀌지 않는다.
    assert train_feats.dtype == torch.float16 and torch.equal(train_feats, before)
    # 돌려받은 헤드는 보고된 검증 AUROC에 도달한 바로 그 헤드다.
    scores = predict(result.head, val_feats, device="cpu", map_size=G).image_scores
    assert auroc(scores[val_labels == 0], scores[val_labels == 1]) == pytest.approx(result.best_val_auroc)
    # 보지 못한 이미지도 가른다.
    test_feats, _, test_labels = planted(20, 10, seed=3)
    scores = predict(result.head, test_feats, device="cpu", map_size=G).image_scores
    assert auroc(scores[test_labels == 0], scores[test_labels == 1]) >= 0.98


# 검증 뒤 헤드가 다시 학습 모드로 돌아오는지 확인한다. 학습 배치는 매 에폭 기울기 켜짐·학습 모드, 검증 배치는
# 기울기 꺼짐·평가 모드여야 한다. 고른 에폭의 BatchNorm 통계는 그 에폭까지의 학습 배치로만 갱신된다(검증이
# 통계를 바꾸지 않음). auroc를 점점 오르는 가짜 값으로 바꿔 마지막 에폭이 뽑히게 한다.
def test_head_is_back_in_train_mode_after_every_validation(monkeypatch):
    modes = []

    class RecordingHead(SegHead):
        def forward(self, feats):
            modes.append((torch.is_grad_enabled(), self.training))
            return super().forward(feats)

    monkeypatch.setattr(supervised, "SegHead", RecordingHead)
    rising = iter([0.6, 0.7, 0.8, 0.9])  # 마지막 에폭이 최고
    monkeypatch.setattr(supervised, "auroc", lambda neg, pos: next(rising))
    # 학습 이미지 32장을 16장씩: 에폭당 두 스텝. 검증 이미지 18장: 배치 하나.
    result, _, _ = fit(TrainConfig(epochs=4, batch_size=16, eval_every=1))
    # 학습 배치(기울기 켜짐)는 첫 검증 전만이 아니라 모든 에폭에서 헤드를 학습 모드로 보고, 검증
    # 배치(기울기 꺼짐)는 평가 모드로 본다.
    assert modes == ([(True, True)] * 2 + [(False, False)]) * 4
    assert result.best_epoch == 4 and not result.head.training
    # BatchNorm 통계: 고른 에폭까지의 모든 학습 배치로만 갱신되고, 다른 것으로는 바뀌지 않는다.
    for layer in (result.head.net[1], result.head.net[4]):
        assert int(layer.num_batches_tracked) == 4 * steps_per_epoch(32, 16)


# 학습 기록은 에폭마다 하나(epoch, loss, val_auroc)이고, 검증은 eval_every 배수와 마지막 에폭에서만 한다(4, 8,
# 10). 고른 에폭은 검증한 에폭 가운데 최고값의 에폭이다.
def test_history_and_evaluation_epochs():
    result, _, _ = fit(TrainConfig(epochs=10, batch_size=16, eval_every=4))
    assert [h["epoch"] for h in result.history] == list(range(1, 11))
    assert all(set(h) == {"epoch", "loss", "val_auroc"} for h in result.history)
    assert all(math.isfinite(h["loss"]) and h["loss"] > 0 for h in result.history)
    # 4에폭마다, 그리고 마지막 에폭에서.
    evaluated = [h["epoch"] for h in result.history if h["val_auroc"] is not None]
    assert evaluated == [4, 8, 10]
    assert result.best_epoch in evaluated
    best = max(h["val_auroc"] for h in result.history if h["val_auroc"] is not None)
    assert result.best_val_auroc == best
    assert result.history[result.best_epoch - 1]["val_auroc"] == best


# 최적화기와 일정: 구성의 학습률·weight decay로 AdamW를 만들고, 전체 스텝(에폭 × 에폭당 배치) 동안 워밍업 없이
# 코사인으로 0까지 줄인다. 이미지 수를 32·33·34장으로 바꿔 한 장짜리 마지막 배치를 버리는 규칙이 스텝 수에
# 반영되는지도 본다.
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
    fit(cfg, n_normal=n_normal, n_defect=8)  # 32, 33, 34장
    (optimizer,) = created
    assert optimizer.kwargs == {"lr": 2e-3, "weight_decay": 0.05}
    total = 3 * steps
    assert len(optimizer.lrs) == total
    # 워밍업 없음: 첫 스텝은 학습률 전체를 쓰고, 곡선은 마지막 스텝 바로 뒤에 0에 닿는다.
    expected = [2e-3 * 0.5 * (1.0 + math.cos(math.pi * t / total)) for t in range(total)]
    np.testing.assert_allclose(optimizer.lrs, expected, rtol=1e-9)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)


# 손실이 패치 격자 위의 BCE(평균, 로짓 입력)이고 양성 가중치가 학습 세트 전체에서 한 번 계산돼 모든 배치에
# 같게 쓰이는지 확인한다. 여기서 비율은 63이다. 상한(20)을 넘으면 상한으로 자르고, 정답에 양성이 없으면
# 1.0이다.
def test_loss_is_weighted_bce_on_the_patch_grid(monkeypatch):
    calls = []
    real = F.binary_cross_entropy_with_logits

    def spy(logits, target, **kwargs):
        calls.append((tuple(logits.shape), target.dtype, float(kwargs["pos_weight"])))
        assert set(kwargs) == {"pos_weight"}  # reduction은 기본값(평균)
        assert set(target.unique().tolist()) <= {0.0, 1.0}
        return real(logits, target, **kwargs)

    monkeypatch.setattr(supervised.F, "binary_cross_entropy_with_logits", spy)
    train_feats, train_targets, _ = planted(24, 8, seed=1)
    val_feats, _, val_labels = planted(12, 6, seed=2)
    n_pos = int(train_targets.sum())
    ratio = (train_targets.size - n_pos) / n_pos  # 32장 x 64패치 중 양성 8장 x 4패치: 비율 63
    assert ratio == pytest.approx(63.0)
    cfg = TrainConfig(epochs=2, batch_size=16, eval_every=5)
    train_head(train_feats, train_targets, val_feats, val_labels, cfg, device="cpu")
    assert calls == [((16, G, G), torch.float32, ratio)] * 4  # 모든 배치에 같은 가중치

    calls.clear()
    capped = TrainConfig(epochs=1, batch_size=16, eval_every=5, max_pos_weight=20.0)
    train_head(train_feats, train_targets, val_feats, val_labels, capped, device="cpu")
    assert {c[2] for c in calls} == {20.0}

    calls.clear()
    no_defects = np.zeros_like(train_targets)
    train_head(train_feats, no_defects, val_feats, val_labels, capped, device="cpu")
    assert {c[2] for c in calls} == {1.0}


# 에폭 선택 규칙: 검증 AUROC가 같으면 뒤의 에폭을 고르고(규칙 변경 4), 더 낮은 값은 바꾸지 않으며, 검증은
# eval_every 배수와 마지막 에폭에서만 한다. 돌려받은 헤드가 고른 에폭의 가중치를 가지고 있어 그때의 검증
# 점수를 그대로 재현하는지, 다른 에폭의 점수와는 다른지도 본다.
@pytest.mark.parametrize(
    ("values", "eval_every", "best_epoch"),
    [
        ([0.5, 0.9, 0.9, 0.7], 1, 3),  # 동점이면 뒤 에폭
        ([0.8, 0.8, 0.8, 0.8], 1, 4),
        ([0.5, 0.6, 0.7, 0.8], 1, 4),
        ([0.9, 0.8, 0.7, 0.6], 1, 1),  # 더 낮은 값은 바꾸지 않는다
        ([0.9, 0.9], 2, 4),  # 에폭 2와 4에서만 검증
        ([0.9, 0.95], 3, 4),  # 에폭 3과 마지막 에폭에서 검증
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
    # 헤드는 최고 에폭의 가중치를 가진다: 그 검증에서 본 점수를 그대로 재현한다.
    scores = predict(result.head, val_feats, device="cpu", map_size=G).image_scores
    which = evaluated.index(best_epoch)
    np.testing.assert_allclose(scores, seen[which], rtol=1e-5, atol=1e-6)
    for other, other_scores in enumerate(seen):
        if other != which:
            assert not np.allclose(scores, other_scores, rtol=1e-3, atol=1e-4)
    # 검증 점수는 정상 먼저, 그다음 결함 순서로 들어온다.
    assert all(len(s) == len(val_labels) for s in seen)


# 쓸 수 없는 입력은 학습 전에 거절한다: 검증에 정상이나 결함 한쪽만 있음(AUROC 정의 불가), 학습 이미지 한
# 장(배치가 없음), 정답·특징 모양 불일치, 검증 특징 차원 불일치, 검증 라벨 수 불일치, 에폭 0.
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


# TrainConfig 기본값이 등록 규칙(60에폭, 배치 32, 학습률 1e-3, weight decay 1e-4, 5에폭마다 검증, 양성 가중치
# 상한 100, 시드 0)과 같고, frozen이라 바꿀 수 없는지 확인한다.
def test_train_config_defaults_are_the_registered_ones():
    cfg = TrainConfig()
    assert (cfg.epochs, cfg.batch_size, cfg.lr, cfg.weight_decay) == (60, 32, 1e-3, 1e-4)
    assert (cfg.eval_every, cfg.max_pos_weight, cfg.seed) == (5, 100.0, 0)
    with pytest.raises(AttributeError):
        cfg.seed = 1


# --- 예측 ---------------------------------------------------------------------------------------


# predict의 정의: 이미지 점수는 패치 로짓의 최댓값(float32), 맵은 로짓 자체를 bilinear(align_corners=False)로
# 키운 float16이다. 확률 대신 로짓을 쓰는 이유는 float16 확률이 로짓 약 8.3 위에서 1.0으로 포화해 픽셀 지표에
# 동점이 생기기 때문이다(규칙 변경 4). 기본 맵 크기는 256이다.
def test_predict_scores_and_maps_match_the_definition():
    feats = torch.randn(5, 4, 4, 3, generator=torch.Generator().manual_seed(0)).to(torch.float16)
    head = pass_through_head(3)
    logits = pass_through_logits(feats)
    result = predict(head, feats, batch_size=2, device="cpu", map_size=4)
    assert isinstance(result, ScoreResult)
    assert result.image_scores.shape == (5,) and result.image_scores.dtype == np.float32
    assert result.maps.shape == (5, 4, 4) and result.maps.dtype == np.float16
    # 이미지 점수: 가장 큰 패치 로짓(확률이 아니다).
    np.testing.assert_allclose(result.image_scores, logits.reshape(5, -1).max(axis=1), rtol=1e-5)
    # 격자 크기의 맵: 로짓 그 자체(sigmoid가 없어 float16이 1.0에서 포화할 수 없다).
    np.testing.assert_allclose(result.maps.astype(np.float64), logits, rtol=2e-3, atol=2e-3)
    # 더 큰 맵: 로짓을 align_corners=False로 bilinear 업샘플링한다.
    big = predict(head, feats, device="cpu", map_size=16).maps
    expected = F.interpolate(
        torch.from_numpy(logits).float()[:, None], size=(16, 16), mode="bilinear", align_corners=False
    )[:, 0].numpy()
    np.testing.assert_allclose(big.astype(np.float64), expected, rtol=2e-3, atol=2e-3)
    assert predict(head, feats, device="cpu").maps.shape == (5, 256, 256)  # 기본 맵 크기


# predict는 헤드를 평가 모드로 바꿔(BatchNorm이 누적 통계를 쓰게) 배치 크기와 무관하게 같은 결과를 낸다. 빈
# 입력은 빈 결과, 4차원이 아닌 입력과 배치 크기 0은 ValueError다.
def test_predict_batching_empty_input_and_eval_mode():
    feats = torch.randn(7, 4, 4, 3, generator=torch.Generator().manual_seed(1)).to(torch.float16)
    torch.manual_seed(0)
    head = SegHead(dim=3, hidden=8).train()
    whole = predict(head, feats, batch_size=64, device="cpu", map_size=8)
    assert not head.training  # BatchNorm이 누적 통계를 써야 한다
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


# 로짓에 inf가 생기면 조용히 저장하지 않고 FloatingPointError로 멈춘다.
def test_predict_raises_on_non_finite_logits():
    head = pass_through_head(3)
    feats = torch.zeros(2, 4, 4, 3)
    feats[1, 2, 2, 0] = float("inf")
    with pytest.raises(FloatingPointError):
        predict(head, feats, device="cpu", map_size=4)


# 학습한 헤드의 맵이 심은 결함 자리에서 가장 높은지 확인한다. 맵 크기 64에서 패치 하나는 8픽셀이고, 정점이
# 심은 2x2 블록 안에 있으며, 블록 안 평균 로짓이 양수이고 두 패치 넘게 떨어진 곳보다 분명히 높다. 픽셀
# 지표(AUPRO, 픽셀 AUROC)가 이 맵에서 계산되므로 위치가 맞아야 한다.
def test_map_peaks_at_the_planted_patch():
    result, _, _ = fit()
    block = 8  # map_size 64에서 패치 하나의 맵 픽셀 수
    for row, col in [(1, 5), (5, 2), (6, 6)]:
        feats = torch.randn(1, G, G, DIM, generator=torch.Generator().manual_seed(10 * row + col))
        feats[0, row : row + 2, col : col + 2, 0] += 6.0
        scored = predict(result.head, feats.to(torch.float16), device="cpu", map_size=64)
        heat = scored.maps[0].astype(np.float32)
        peak_r, peak_c = np.unravel_index(int(heat.argmax()), heat.shape)
        assert row * block <= peak_r < (row + 2) * block
        assert col * block <= peak_c < (col + 2) * block
        # 심은 블록 안의 로짓은 분명히 양수이고, 두 패치 넘게 떨어진 곳에서는 훨씬 낮다.
        inside = heat[row * block : (row + 2) * block, col * block : (col + 2) * block]
        assert inside.mean() > 1.0
        far = np.ones_like(heat, dtype=bool)
        top, left = max(0, (row - 2) * block), max(0, (col - 2) * block)
        far[top : (row + 4) * block, left : (col + 4) * block] = False
        assert inside.mean() - heat[far].mean() > 1.0
        assert scored.image_scores[0] > 0  # 가장 큰 로짓이 결함 쪽에 있다


# --- 특징 추출 ----------------------------------------------------------------------------------


# 정규화한 이미지를 2x4 격자로 평균 풀링하는 가짜 특징 추출기. 칸 하나의 값을 손으로 계산할 수 있다.
class PoolExtractor(torch.nn.Module):
    """가짜 추출기: 정규화한 이미지를 [B, 2, 4, 3] 특징 격자로 평균 풀링한다."""

    name = "pool"
    dim = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, (2, 4)).permute(0, 2, 3, 1)


# extract가 collect_features와 같은 값을 [N, H, W, D] CPU fp16 격자로 돌려주는지, 칸 하나의 값이 그 영역
# 픽셀을 ImageNet 평균·표준편차로 정규화해 평균한 값과 같은지 확인한다. 빈 입력도 처리한다.
def test_extract_returns_cpu_fp16_grids():
    images = np.random.default_rng(0).integers(0, 256, size=(5, 8, 8, 3), dtype=np.uint8)
    extractor = PoolExtractor()
    feats = extract(extractor, images, batch_size=2, device="cpu")
    assert feats.shape == (5, 2, 4, 3) and feats.dtype == torch.float16 and feats.device.type == "cpu"
    flat, grid = collect_features(extractor, images, batch_size=2, device="cpu")
    assert grid == (2, 4)
    assert torch.equal(feats.reshape(-1, 3), flat)
    # 이미지 3의 (1, 2) 칸은 행 4..7, 열 4..5를 덮는다.
    block = images[3, 4:8, 4:6].reshape(-1, 3).astype(np.float64).mean(axis=0) / 255.0
    manual = (block - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    np.testing.assert_allclose(feats[3, 1, 2].double().numpy(), manual, atol=2e-3)
    assert extract(extractor, images[:0], device="cpu").shape == (0, 0, 0, 3)
