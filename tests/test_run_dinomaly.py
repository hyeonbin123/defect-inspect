# [파일 역할] Dinomaly 학습·평가 스크립트 src/defect_inspect/run_dinomaly.py를 검증한다. anomalib의 모델
# 코드만 쓰고 학습 루프는 직접 쓴 구조라, 그 루프가 anomalib Lightning 모듈과 같은 학습을 하는지, 평가 결과가
# 공통 형식으로 저장되는지, CLI가 봉인 테스트 규칙을 지키는지, 6단계 구성(dms-*)과 fp16 관문 검사가
# 등록대로인지를 본다.
# [왜 필요한가] 2단계 DM(README AUROC 96.8)과 6단계 서빙 모델 dms-280-car(CPU 150ms, AUROC 96.5)가 모두 이
# 스크립트로 학습·평가됐다. 학습 루프를 직접 쓴 이상, 옵티마이저·스케줄러·클리핑 순서나 GradScaler 처리, 배치
# 순서, 난수 고정이 조금만 달라도 '문헌 기본값으로 고정한 구성'이 아니게 된다. 또 6단계는 fp16 학습 가능
# 여부를 관문(패치 어텐션 대조, fp16 대 fp32 곡선)으로 정했으므로 그 판정 함수가 규칙(docs/experiments.md
# 6단계 관문 2)과 같아야 한다.
# [주요 구성] batch_order → train(가짜 모델 TinyModel로 일정·순서·결정성·anomalib 기준 루프와 비트 단위
# 일치·fp16 클리핑) → predict → 실제 DinomalyModel(무작위 가중치, 다운로드 없음) → eval_category와 CLI(가짜
# 캐시·매니페스트·장부) → 6단계 구성·관문.
# [실행 조건] torch와 anomalib(dinomaly 그룹)이 있어야 돈다. 모든 계산은 CPU이고 사전학습 가중치를 내려받지
# 않는다.

import contextlib
import copy
import json
import shutil

import numpy as np
import pytest

# torch·anomalib이 없는 환경(CI)에서는 파일 전체를 건너뛴다. 아래 import는 이 줄 뒤에 와야 해서 E402 예외
# 표시를 달았다. anomalib의 InferenceBatch(추론 출력 형식)와 components(StableAdamW, WarmCosineScheduler)는
# 가짜 모델과 기준 학습 루프를 anomalib과 같은 부품으로 만들려고 가져온다.
torch = pytest.importorskip("torch")
pytest.importorskip("anomalib")

import torch.nn.functional as F  # noqa: E402
from anomalib.data import InferenceBatch  # noqa: E402
from anomalib.models.image.dinomaly import components  # noqa: E402

from defect_inspect import run_dinomaly  # noqa: E402
from defect_inspect.backbones import to_tensor  # noqa: E402
from defect_inspect.run_dinomaly import batch_order, predict, train  # noqa: E402
from defect_inspect.splits import SEALED_ROLES, ManifestRow, SealedTestError, write_manifest  # noqa: E402

# 가짜 모델·가짜 캐시 이미지의 한 변(32px).
SIZE = 32


# DinomalyModel과 호출 규약이 같은 작은 가짜 모델. 인코더(합성곱)는 고정, 디코더는 학습 가능하다. 학습
# 모드에서는 global_step을 받아야 하고 '디코더 출력과 인코더 특징의 차이' 손실을 내며, nan_at 스텝에서는 NaN
# 손실을 낸다. 평가 모드에서는 정규화 이미지의 절댓값 평균을 맵으로, 그 최댓값을 점수로 담은 InferenceBatch를
# 돌려준다(입력에만 의존). calls에 (학습 모드 여부, global_step, 배치 모양, dtype)을 남겨 학습 루프가 무엇을
# 넘겼는지 확인한다. map_dims = 3이면 채널 축 없는 맵을 낸다.
class TinyModel(torch.nn.Module):
    """Dinomaly와 호출 규약이 같은 가짜 모델: 인코더는 고정, 디코더는 학습 가능하다.

    학습 모드는 스칼라 손실을 돌려주고 `global_step`이 있어야 한다. 평가 모드는 정규화 이미지의 절댓값
    평균을 맵으로 담은 InferenceBatch를 돌려준다(그래서 입력에만 의존한다).
    """

    def __init__(self, nan_at: int | None = None, map_dims: int = 4) -> None:
        super().__init__()
        self.encoder = torch.nn.Conv2d(3, 4, kernel_size=4, stride=4)
        self.decoder = torch.nn.Conv2d(4, 4, kernel_size=1)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.nan_at = nan_at
        self.map_dims = map_dims
        self.calls: list[tuple] = []

    def forward(self, batch, global_step=None):
        self.calls.append((self.training, global_step, tuple(batch.shape), batch.dtype))
        if self.training:
            if global_step is None:
                raise ValueError("global_step must be provided during training")
            feats = self.encoder(batch)
            loss = 100.0 * (self.decoder(feats) - feats).pow(2).mean()
            return loss * float("nan") if global_step == self.nan_at else loss
        anomaly_map = batch.abs().mean(dim=1, keepdim=True)
        score = anomaly_map.flatten(1).amax(dim=1)
        if self.map_dims == 3:
            anomaly_map = anomaly_map[:, 0]
        return InferenceBatch(pred_score=score, anomaly_map=anomaly_map)


# 학습 손실이 드롭아웃을 거치는 가짜 모델. Dinomaly 병목의 드롭아웃처럼 torch 난수를 쓰므로, 학습이 드롭아웃
# 난수까지 seed로 고정하는지 확인하는 데 쓴다.
class DropoutModel(TinyModel):
    """Dinomaly의 병목처럼 학습 손실이 드롭아웃을 거치는 TinyModel."""

    def __init__(self) -> None:
        super().__init__()
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, batch, global_step=None):
        if not self.training:
            return super().forward(batch, global_step)
        feats = self.encoder(batch)
        return 100.0 * (self.decoder(self.drop(feats)) - feats).pow(2).mean()


# seed로 정해지는 무작위 uint8 이미지 [n, size, size, 3].
def _images(n: int, seed: int = 0, size: int = SIZE) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (n, size, size, 3), dtype=np.uint8)


# 기준 학습 루프: anomalib의 TRAINING_CONFIG와 부품(StableAdamW, WarmCosineScheduler, gradient clip)만으로
# fp32 학습을 한 스텝씩 풀어 쓴 것이다. run_dinomaly.train과는 배치 순서 외에 코드를 공유하지 않으므로, 두
# 결과가 비트 단위로 같으면 직접 쓴 루프가 Lightning 모듈의 학습과 같다는 근거가 된다. 순서는 zero_grad → 손실
# → backward → 클리핑 → 옵티마이저 → 스케줄러다.
def _reference_train(model, images, *, steps: int, batch_size: int, seed: int) -> list[float]:
    """anomalib Lightning 모듈의 fp32 학습을 한 스텝씩 풀어 쓴 것. 손실 목록을 돌려준다.

    anomalib의 TRAINING_CONFIG와 부품만으로 만들어, 배치 순서 말고는 `run_dinomaly.train`과 공유하는
    코드가 없다.
    """
    from anomalib.models.image.dinomaly.lightning_model import TRAINING_CONFIG

    order = batch_order(len(images), steps, batch_size, seed)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = components.StableAdamW([{"params": params}], **TRAINING_CONFIG["optimizer"])
    scheduler = components.WarmCosineScheduler(
        optimizer, **{**TRAINING_CONFIG["scheduler"], "total_iters": steps}
    )
    torch.manual_seed(seed)
    model.train()
    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        loss = model(to_tensor(images[order[step]], "cpu", torch.float32), global_step=step)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, TRAINING_CONFIG["trainer"]["gradient_clip_val"])
        optimizer.step()
        scheduler.step()
        losses.append(float(loss))
    return losses


# ---------------------------------------------------------------- batch_order(배치 순서)


# batch_order가 [steps, batch] int64 인덱스를 내고, 같은 seed면 같고 다른 seed면 다르며, 앞쪽 행이 전체 스텝
# 수와 무관한지 본다. 마지막 성질 덕분에 60스텝 관문 검사와 10,000스텝 실제 학습이 같은 배치 순서로
# 시작한다(experiments.md 관문 2 (나) '실제 학습과 같은 배치 순서').
def test_batch_order_shape_and_determinism():
    order = batch_order(50, 20, 16, seed=3)
    assert order.shape == (20, 16) and order.dtype == np.int64
    assert order.min() >= 0 and order.max() < 50
    assert np.array_equal(order, batch_order(50, 20, 16, seed=3))
    assert not np.array_equal(order, batch_order(50, 20, 16, seed=4))
    # 앞쪽 배치는 뒤에 몇 스텝이 이어지는지와 무관하다.
    assert np.array_equal(order[:7], batch_order(50, 7, 16, seed=3))
    assert batch_order(50, 0, 16, seed=0).shape == (0, 16)


# 배치 순서가 '에폭마다 모든 이미지를 한 번씩 쓰는 순열의 연속'이면서 한 배치 안에 같은 이미지가 두 번
# 들어가지 않는지, 이미지 수와 배치 크기 조합을 바꿔 가며 본다. 스텝을 이미지 수의 4배로 잡아 배치 중간에 에폭
# 경계가 많이 생기게 한다.
@pytest.mark.parametrize(("n", "batch"), [(50, 16), (20, 16), (16, 16), (17, 16), (64, 16), (31, 5), (9, 8)])
def test_batch_order_is_a_stream_of_permutations_without_repeats_in_a_batch(n, batch):
    for seed in range(5):
        steps = 4 * n  # 배치 안에 에폭 경계가 많이 생긴다
        order = batch_order(n, steps, batch, seed=seed)
        for row in order:
            assert len(set(row.tolist())) == batch
        stream = order.reshape(-1)
        for epoch in range(len(stream) // n):
            assert np.array_equal(np.sort(stream[epoch * n : (epoch + 1) * n]), np.arange(n))


# 순열을 그냥 이어 붙이면 에폭 경계에 걸친 배치에서 같은 이미지가 두 번 나오는 seed가 실제로 있음을 먼저
# 보이고, batch_order는 첫 배치는 그대로 두면서 둘째 배치의 중복을 없애는지 확인한다(_keep_out이 새 에폭
# 앞쪽의 겹치는 이미지를 뒤로 보냄).
def test_batch_order_repairs_an_epoch_boundary():
    # 이미지 20장을 16장씩: 둘째 배치는 에폭 0의 4장과 에폭 1의 12장을 담는다. 순열을 그냥 이어 붙이면
    # 대부분의 seed에서 이 배치에 같은 이미지가 두 번 들어간다.
    rng = np.random.default_rng(0)
    plain = np.concatenate([rng.permutation(20), rng.permutation(20)])[:32].reshape(2, 16)
    assert len(set(plain[1].tolist())) < 16  # 이 seed는 실제로 겹치므로 고치는 경로를 탄다
    order = batch_order(20, 2, 16, seed=0)
    assert np.array_equal(order[0], plain[0])
    assert len(set(order[1].tolist())) == 16


# 이미지가 배치보다 적으면 한 배치 안 중복은 피할 수 없지만, 여전히 에폭 단위로 모든 이미지를 한 번씩 쓰고
# 이미지별 사용 횟수가 최대 1밖에 차이 나지 않아야 한다.
def test_batch_order_with_fewer_images_than_a_batch():
    order = batch_order(5, 6, 8, seed=0)
    assert order.shape == (6, 8)
    stream = order.reshape(-1)
    for epoch in range(len(stream) // 5):
        assert np.array_equal(np.sort(stream[epoch * 5 : (epoch + 1) * 5]), np.arange(5))
    counts = np.bincount(stream, minlength=5)
    assert counts.max() - counts.min() <= 1


# 이미지 0장, 음수 스텝, 배치 0은 거절한다.
def test_batch_order_rejects_bad_arguments():
    for args in [(0, 5, 4), (10, -1, 4), (10, 5, 0)]:
        with pytest.raises(ValueError):
            batch_order(*args, seed=0)


# ---------------------------------------------------------------- train(학습 루프)


# 직접 옮겨 적은 학습 상수(옵티마이저, 스케줄러, 스텝 수, 클리핑, 입력 크기, 기본 인코더)가 설치된 anomalib의
# TRAINING_CONFIG·기본값과 같은지 본다. 2단계 DM은 '구현 기본값으로 고정'한 구성이라(experiments.md 2단계),
# anomalib을 올리거나 상수를 고칠 때 어긋나면 여기서 드러난다.
def test_training_constants_match_anomalib():
    from anomalib.models.image.dinomaly import lightning_model

    config = lightning_model.TRAINING_CONFIG
    assert run_dinomaly.OPTIMIZER == config["optimizer"]
    scheduler = dict(config["scheduler"])
    assert scheduler.pop("total_iters") == run_dinomaly.STEPS == config["trainer"]["max_steps"]
    assert run_dinomaly.SCHEDULER == scheduler
    assert run_dinomaly.GRAD_CLIP == config["trainer"]["gradient_clip_val"]
    assert run_dinomaly.IMG_SIZE == lightning_model.DEFAULT_CROP_SIZE
    assert lightning_model.Dinomaly.__init__.__defaults__[0] == run_dinomaly.ENCODER


# 150스텝 학습에서 기록·학습률 일정·모델 호출을 한꺼번에 확인한다. 기록 항목 키, 모든 손실 유한, CPU라 VRAM 0,
# 건너뛴 스텝 0, 시간 단조 증가. 학습률은 warmup 100스텝 동안 0→2e-3 직선, 그 뒤 50스텝은 2e-3→2e-4
# 코사인이어야 하고(기록되는 값은 그 스텝에 실제로 쓴 학습률), JSONL 파일 내용이 반환값과 같고 CR이 없어야
# 한다.
# 모델은 모든 스텝을 학습 모드·올바른 global_step·정규화된 fp32 배치로 한 번씩 보고, 고정 인코더는 그대로,
# 디코더는 바뀌며, 손실은 줄어야 한다. amp=True를 줬지만 CPU라 fp32로 돈다.
def test_train_logs_and_follows_the_schedule(tmp_path):
    torch.manual_seed(0)
    model = TinyModel()
    frozen = model.encoder.weight.clone()
    start = model.decoder.weight.detach().clone()
    images = _images(10)
    log_path = tmp_path / "train_log.jsonl"
    steps = 150
    log = train(
        model, images, steps=steps, batch_size=4, device="cpu", amp=True, log_every=1, log_path=log_path
    )

    assert [e["step"] for e in log] == list(range(1, steps + 1))
    assert all(set(e) == {"step", "loss", "lr", "seconds", "vram_mb", "skipped"} for e in log)
    assert all(np.isfinite(e["loss"]) and e["vram_mb"] == 0.0 and e["skipped"] == 0 for e in log)
    seconds = [e["seconds"] for e in log]
    assert seconds == sorted(seconds)
    # warmup(100스텝 동안 0에서 2e-3까지), 그다음 나머지 50스텝은 2e-3에서 2e-4 쪽으로 코사인.
    cosine = 2e-4 + 0.5 * (2e-3 - 2e-4) * (1 + np.cos(np.pi * np.arange(50) / 50))
    expected = np.concatenate([np.linspace(0.0, 2e-3, 100), cosine])
    assert np.allclose([e["lr"] for e in log], expected, rtol=1e-12, atol=0)
    assert log[0]["lr"] == 0.0 and log[99]["lr"] == log[100]["lr"] == 2e-3

    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert lines == log
    assert b"\r" not in log_path.read_bytes()

    # 모델은 모든 스텝을 학습 모드로 한 번씩, batch_order 순서의 정규화된 fp32 배치로 보았다.
    assert [c[:2] for c in model.calls] == [(True, step) for step in range(steps)]
    assert all(c[2] == (4, 3, SIZE, SIZE) and c[3] == torch.float32 for c in model.calls)
    assert model.training
    assert torch.equal(model.encoder.weight, frozen)
    assert not torch.equal(model.decoder.weight, start)
    assert log[-1]["loss"] < log[0]["loss"]


# train이 스텝마다 batch_order의 그 행에 해당하는 이미지들을 같은 정규화로 넘기는지 모델 입력을 기록해
# 비교한다.
def test_train_feeds_the_batches_of_batch_order():
    seen = []

    class Recorder(TinyModel):
        def forward(self, batch, global_step=None):
            seen.append(batch.clone())
            return super().forward(batch, global_step)

    images = _images(9, seed=1)
    train(Recorder(), images, steps=5, batch_size=4, device="cpu", seed=7)
    order = batch_order(9, 5, 4, seed=7)
    for step, batch in enumerate(seen):
        assert torch.equal(batch, to_tensor(images[order[step]], "cpu", torch.float32))


# 옵티마이저·스케줄러·클리핑을 감시용 클래스로 바꿔, anomalib 설정값 그대로 만들어지는지, 학습 가능한
# 파라미터(디코더)만 옵티마이저에 들어가는지, 매 반복이 '클리핑 → 옵티마이저 → 스케줄러' 순서인지, 클리핑이
# norm 0.1로 실제로 자를 만큼 큰 기울기를 받았는지 본다. train은 이 부품들을 호출할 때 anomalib 모듈에서
# 가져오므로 monkeypatch가 먹힌다.
def test_train_uses_the_optimiser_scheduler_and_clipping_of_anomalib(monkeypatch):
    events: list[str] = []
    made: dict = {}

    class SpyOptimizer(components.StableAdamW):
        def __init__(self, params, **kwargs):
            params = list(params)
            made["params"], made["optimizer"] = params, kwargs
            super().__init__(params, **kwargs)

        def step(self, closure=None):
            events.append("optimizer")
            return super().step(closure)

    class SpyScheduler(components.WarmCosineScheduler):
        def __init__(self, optimizer, **kwargs):
            made["scheduler"] = kwargs
            super().__init__(optimizer, **kwargs)

        def step(self, *args, **kwargs):
            events.append("scheduler")
            return super().step(*args, **kwargs)

    clips = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def spy_clip(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        before = float(torch.sqrt(sum(p.grad.pow(2).sum() for p in parameters)))
        result = real_clip(parameters, max_norm, *args, **kwargs)
        after = float(torch.sqrt(sum(p.grad.pow(2).sum() for p in parameters)))
        clips.append((max_norm, [id(p) for p in parameters], before, after))
        events.append("clip")
        return result

    monkeypatch.setattr(components, "StableAdamW", SpyOptimizer)
    monkeypatch.setattr(components, "WarmCosineScheduler", SpyScheduler)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy_clip)

    torch.manual_seed(0)
    model = TinyModel()
    steps = 6
    train(model, _images(8), steps=steps, batch_size=4, device="cpu")

    trainable = [id(p) for p in model.decoder.parameters()]
    assert made["optimizer"] == run_dinomaly.OPTIMIZER
    assert [id(p) for group in made["params"] for p in group["params"]] == trainable
    assert made["scheduler"] == {"total_iters": steps, **run_dinomaly.SCHEDULER}
    # 매 반복: 클리핑, 옵티마이저, 스케줄러 순서.
    assert [e for e in events if e != "scheduler"] == ["clip", "optimizer"] * steps
    assert events[-3 * steps :] == ["clip", "optimizer", "scheduler"] * steps
    assert len(clips) == steps
    for max_norm, ids, _before, after in clips:
        assert max_norm == 0.1 and ids == trainable
        assert after <= 0.1 * (1 + 1e-5)
    assert max(before for _, _, before, _ in clips) > 0.1  # 클리핑이 실제로 자를 것이 있었다


# 학습이 seed 하나로 완전히 결정되는지 본다. 호출 전에 프로세스가 어떤 난수를 썼든(noise) 같은 seed면 손실과
# 가중치가 같고, seed가 다르면 달라야 한다. 드롭아웃 모델을 써서 드롭아웃 난수까지 seed로 고정되는지 확인한다.
def test_train_is_deterministic_for_a_seed():
    def run(seed, noise):
        torch.manual_seed(0)
        model = DropoutModel()
        torch.manual_seed(noise)  # 호출 전에 프로세스가 뽑은 난수는 결과를 바꾸면 안 된다
        log = train(model, _images(12), steps=8, batch_size=4, device="cpu", seed=seed, log_every=1)
        return [e["loss"] for e in log], model.decoder.weight.detach().clone()

    losses_a, weight_a = run(0, noise=11)
    losses_b, weight_b = run(0, noise=22)
    losses_c, _ = run(1, noise=11)
    assert losses_a == losses_b and torch.equal(weight_a, weight_b)
    assert losses_a != losses_c


# run_dinomaly.train과 anomalib 부품만으로 쓴 기준 루프(_reference_train)를 같은 초기 가중치에서 돌려 손실
# 목록과 디코더 가중치가 비트 단위로 같은지 본다. 기울기가 스텝마다 쌓이거나, 드롭아웃 난수 흐름이 다르거나,
# 클리핑과 스텝 순서가 다르면 모두 다른 가중치가 나오므로 학습 루프 전체의 동등성을 확인하는 테스트다.
def test_train_matches_a_plain_reference_loop():
    images = _images(10, seed=4)
    torch.manual_seed(0)
    ours = DropoutModel()
    reference = copy.deepcopy(ours)
    start = ours.decoder.weight.detach().clone()
    torch.manual_seed(999)  # train이 드롭아웃 난수 흐름을 스스로 고정한다
    log = train(ours, images, steps=12, batch_size=4, device="cpu", seed=3, log_every=1)
    losses = _reference_train(reference, images, steps=12, batch_size=4, seed=3)

    assert [e["loss"] for e in log] == losses
    assert torch.equal(ours.decoder.weight, reference.decoder.weight)
    assert torch.equal(ours.decoder.bias, reference.decoder.bias)
    assert not torch.equal(ours.decoder.weight, start)


# fp16 경로에서 GradScaler가 손실을 키우므로(처음 65536배), 클리핑은 unscale_ 뒤의 실제 기울기를 봐야 한다.
# 그렇지 않으면 매 스텝이 norm 0.1/65536으로 깎인다. CPU에서도 실제로 스케일하는 GradScaler를
# 강제로 만들어, 스케일한 실행과 하지 않은 실행의 클리핑 입력·손실·가중치가 똑같은지 본다(2의 거듭제곱으로
# 곱했다 나누는 것은 정확하므로 같아야 한다). 매 스텝 순서는 unscale → clip → step → update이고, 건너뛴 스텝은
# 0이어야 한다. 강제하지 않으면 CPU에서는 enabled=False인 스케일러가 만들어진다.
def test_train_clips_after_unscaling_with_a_working_grad_scaler(monkeypatch):
    real_scaler = torch.amp.GradScaler
    real_clip = torch.nn.utils.clip_grad_norm_
    force = {"on": False}
    made: list[tuple] = []
    events: list[str] = []
    norms: list[float] = []

    class SpyScaler(real_scaler):
        def __init__(self, device="cuda", **kwargs):
            made.append((device, dict(kwargs)))
            if force["on"]:  # CPU에서도 실제로 스케일하는 스케일러
                device, kwargs = "cpu", {**kwargs, "enabled": True}
            super().__init__(device, **kwargs)

        def unscale_(self, optimizer):
            events.append("unscale")
            return super().unscale_(optimizer)

        def step(self, optimizer, *args, **kwargs):
            events.append("step")
            return super().step(optimizer, *args, **kwargs)

        def update(self, *args, **kwargs):
            events.append("update")
            return super().update(*args, **kwargs)

    def spy_clip(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        norms.append(float(torch.sqrt(sum(p.grad.pow(2).sum() for p in parameters))))
        events.append("clip")
        return real_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.amp, "GradScaler", SpyScaler)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy_clip)
    steps = 6

    def run(scaling):
        force["on"] = scaling
        made.clear()
        events.clear()
        norms.clear()
        torch.manual_seed(0)
        model = TinyModel()
        log = train(model, _images(8), steps=steps, batch_size=4, device="cpu", amp=True, log_every=1)
        assert events == ["unscale", "clip", "step", "update"] * steps
        assert log[-1]["skipped"] == 0
        return [e["loss"] for e in log], list(norms), model.decoder.weight.detach().clone()

    plain = run(scaling=False)
    assert made == [("cuda", {"enabled": False})]  # CPU에서는 `amp`가 적용되지 않아 그냥 fp32
    scaled = run(scaling=True)
    # 2의 거듭제곱으로 곱했다가 나누는 것은 정확하므로 클리핑은 두 번 모두 같은 기울기를 봤다.
    assert max(plain[1]) > 0.1 and scaled[1] == plain[1]
    assert scaled[0] == plain[0] and torch.equal(scaled[2], plain[2])


# fp16 autocast는 CUDA에서 amp일 때만 켜지고(RTX 2080 Ti에는 bf16이 없어 fp16만 쓴다), CPU나 amp 끔에서는
# 아무것도 하지 않는 문맥이어야 한다. 기록에 '실제로 돈 정밀도'를 남기는 근거가 이 함수다.
def test_fp16_autocast_applies_on_cuda_only():
    assert run_dinomaly.amp_applied(True, "cuda") and run_dinomaly.amp_applied(True, "cuda:0")
    assert not run_dinomaly.amp_applied(True, "cpu") and not run_dinomaly.amp_applied(False, "cuda")
    context = run_dinomaly._autocast(torch.device("cuda"), True)
    assert isinstance(context, torch.autocast)
    assert context.device == "cuda" and context.fast_dtype == torch.float16
    for device, amp in (("cpu", True), ("cuda", False), ("cpu", False)):
        assert isinstance(run_dinomaly._autocast(torch.device(device), amp), contextlib.nullcontext)


# log_every = 5로 12스텝이면 5, 10과 마지막 12스텝이 기록되고, 같은 로그 파일에 두 번째 학습이 이어
# 붙는지(덮어쓰지 않는지) 본다.
def test_train_logs_every_n_steps_and_the_last_step_and_appends(tmp_path):
    log_path = tmp_path / "log.jsonl"
    log = train(TinyModel(), _images(8), steps=12, batch_size=4, device="cpu", log_every=5, log_path=log_path)
    assert [e["step"] for e in log] == [5, 10, 12]
    train(TinyModel(), _images(8), steps=3, batch_size=4, device="cpu", log_every=5, log_path=log_path)
    steps = [json.loads(line)["step"] for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert steps == [5, 10, 12, 3]


# 로그 줄을 쓸 때마다 flush하는지 본다. 모델의 forward 안에서 로그 파일을 읽어 보면, 이전 스텝까지의 줄이 이미
# 파일에 보여야 한다. 오래 걸리는 GPU 학습은 백그라운드로 돌리고 다른 프로세스가 로그로 진행을 감시한다.
def test_train_flushes_each_log_line(tmp_path):
    log_path = tmp_path / "log.jsonl"
    visible = []

    class Watcher(TinyModel):
        def forward(self, batch, global_step=None):
            # 이 스텝이 도는 동안 다른 프로세스가 볼 수 있는 로그 내용.
            text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
            visible.append(len(text.splitlines()))
            return super().forward(batch, global_step)

    train(Watcher(), _images(8), steps=5, batch_size=4, device="cpu", log_every=1, log_path=log_path)
    assert visible == [0, 1, 2, 3, 4]


# 손실이 NaN이 되면 몇 번째 스텝인지 담은 FloatingPointError로 바로 멈추고, 그 전 3스텝의 기록만 남으며,
# 가중치는 NaN으로 갱신되지 않아야 한다. 2단계 DM이 fp16에서 15번째 스텝에 NaN이 난 것처럼(experiments.md)
# 학습이 깨지면 조용히 이어 가지 않게 한다.
def test_train_stops_on_a_non_finite_loss(tmp_path):
    log_path = tmp_path / "log.jsonl"
    model = TinyModel(nan_at=3)
    with pytest.raises(FloatingPointError, match=r"step 4 of 10 \(global_step=3\)"):
        train(model, _images(8), steps=10, batch_size=4, device="cpu", log_every=1, log_path=log_path)
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 3
    assert torch.isfinite(model.decoder.weight).all()


# uint8이 아닌 이미지, 0스텝, 학습 가능한 파라미터가 없는 모델은 거절한다.
def test_train_rejects_bad_input():
    with pytest.raises(ValueError):
        train(TinyModel(), _images(8).astype(np.float32), steps=2, batch_size=4, device="cpu")
    with pytest.raises(ValueError):
        train(TinyModel(), _images(8), steps=0, batch_size=4, device="cpu")
    frozen = TinyModel()
    for param in frozen.parameters():
        param.requires_grad = False
    with pytest.raises(ValueError, match="no trainable"):
        train(frozen, _images(8), steps=2, batch_size=4, device="cpu")


# ---------------------------------------------------------------- predict(추론)


# predict가 모델을 평가 모드로 바꾸고(global_step 없음), 배치를 3, 3, 1로 나눠 돌리며, 점수는 pred_score
# 그대로 float32, 맵은 bilinear(align_corners=False)로 256에 맞춘 뒤 float16으로 저장하는지 직접 계산과 비트
# 단위로 맞춘다. 픽셀 지표는 모든 구성에서 256 격자로 잰다(experiments.md 전처리).
def test_predict_shapes_order_and_resizing():
    model = TinyModel().train()
    images = _images(7, seed=2)
    result = predict(model, images, batch_size=3, device="cpu")
    assert result.image_scores.shape == (7,) and result.image_scores.dtype == np.float32
    assert result.maps.shape == (7, 256, 256) and result.maps.dtype == np.float16
    assert not model.training
    assert all(c[0] is False and c[1] is None for c in model.calls)
    assert [c[2][0] for c in model.calls] == [3, 3, 1]

    with torch.no_grad():
        out = model(to_tensor(images, "cpu", torch.float32))
        resized = F.interpolate(out.anomaly_map, size=(256, 256), mode="bilinear", align_corners=False)
    assert np.array_equal(result.image_scores, out.pred_score.numpy())
    assert np.array_equal(result.maps, resized[:, 0].to(torch.float16).numpy())


# 채널 축 없는 [B, H, W] 맵도 받아들이고, 맵이 이미 map_size면 크기를 바꾸지 않는지 본다. 가짜 모델은 맵
# 최댓값이 점수이므로 저장된 맵(fp16)의 최댓값이 점수와 맞아야 한다.
def test_predict_keeps_a_map_that_already_has_the_size():
    model = TinyModel(map_dims=3)  # 채널 축 없는 맵도 받아들인다
    images = _images(3, seed=3)
    result = predict(model, images, batch_size=2, device="cpu", map_size=SIZE)
    with torch.no_grad():
        out = model(to_tensor(images, "cpu", torch.float32))
    assert result.maps.shape == (3, SIZE, SIZE)
    assert np.array_equal(result.maps, out.anomaly_map.to(torch.float16).numpy())
    # 이미지마다 가장 밝거나 가장 어두운 화소가 점수를 결정한다.
    assert np.allclose(result.maps.reshape(3, -1).max(axis=1), result.image_scores, rtol=1e-3)


# 이미지 0장이면 빈 결과, 채널이 3이 아니거나 배치 0이면 ValueError다.
def test_predict_empty_and_bad_input():
    empty = predict(TinyModel(), _images(0), device="cpu")
    assert empty.image_scores.shape == (0,) and empty.maps.shape == (0, 256, 256)
    with pytest.raises(ValueError):
        predict(TinyModel(), _images(2)[..., :1], device="cpu")
    with pytest.raises(ValueError):
        predict(TinyModel(), _images(2), batch_size=0, device="cpu")


# 점수가 NaN이면 FloatingPointError로 멈춘다. NaN 점수는 임계값 비교에서 '정상'으로 통과하므로 저장하면 안
# 된다.
def test_predict_raises_on_non_finite_scores():
    class Broken(TinyModel):
        def forward(self, batch, global_step=None):
            out = super().forward(batch, global_step)
            return InferenceBatch(pred_score=out.pred_score * float("nan"), anomaly_map=out.anomaly_map)

    with pytest.raises(FloatingPointError):
        predict(Broken(), _images(2), device="cpu")


# ---------------------------------------------------------------- 실제 모델, 무작위 가중치


# 가짜 모델이 아니라 anomalib의 실제 DinomalyModel(ViT-S 인코더)로 만들기·학습·채점·저장·불러오기를 CPU에서
# 확인한다. timm.create_model을 감싸 사전학습 가중치를 내려받지 않게 하되, 실제 실행이 pretrained=True를
# 요청한다는 것은 기록으로 확인한다.
# build_model이 Lightning 모듈처럼 준비하는지 본다: 학습 파라미터는 병목·디코더뿐, 인코더 고정, fp32, 선형층은
# 절단 정규분포(std 0.01, ±0.03)·편향 0, LayerNorm은 1·0. 56px(패치 14라 4×4 격자) 이미지로 3스텝 학습하면
# 인코더는 그대로, 학습 파라미터 절반 넘게 바뀌고, 기준 루프와 손실·가중치가 비트 단위로 같아야 한다. 맵은 1 -
# 코사인 유사도라 0~2 범위다.
# 저장 파일에는 학습 파라미터만 들어 있고(고정 인코더는 다시 만들 수 있음), 같은 seed로 같은 무작위 인코더를
# 만들면 불러온 모델이 같은 점수·맵을 내야 한다. 파라미터 하나가 빠진 파일은 거절한다.
def test_real_model_builds_trains_and_predicts_without_pretrained_weights(monkeypatch, tmp_path):
    timm = pytest.importorskip("timm")
    import timm.models.vision_transformer as vit

    real_create = timm.create_model
    requested = []

    def create_without_download(name, *args, **kwargs):
        requested.append((name, kwargs.get("pretrained")))
        kwargs["pretrained"] = False  # 테스트는 DINOv2 가중치를 절대 내려받지 않는다
        return real_create(name, *args, **kwargs)

    monkeypatch.setattr(timm, "create_model", create_without_download)
    # anomalib은 ViT 인코더를 만들 때 timm의 이 함수를 프로세스 전체에서 바꿔 끼운다. 같은 값으로
    # monkeypatch에 등록해 두면 테스트가 끝날 때 원래 함수로 되돌려져 다른 테스트에 영향이 남지 않는다.
    monkeypatch.setattr(vit, "resample_abs_pos_embed", vit.resample_abs_pos_embed)

    encoder = "vit_small_patch14_reg4_dinov2"
    torch.manual_seed(0)
    model = run_dinomaly.build_model(encoder)
    assert requested == [(encoder, True)]  # 실제 실행은 사전학습 인코더를 요청한다

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(name.startswith(("bottleneck.", "decoder.")) for name in trainable)
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert all(p.dtype == torch.float32 for p in model.parameters())
    for module in (model.bottleneck, model.decoder):
        assert all(p.requires_grad for p in module.parameters())
        for layer in module.modules():
            if isinstance(layer, torch.nn.Linear):
                weight = layer.weight.detach()
                assert float(weight.abs().max()) <= 0.03
                assert 0.005 < float(weight.std()) < 0.015
                assert layer.bias is None or not layer.bias.any()
            elif isinstance(layer, torch.nn.LayerNorm):
                assert bool((layer.weight == 1).all()) and not layer.bias.any()

    images = _images(8, seed=5, size=56)  # 4x4 패치 격자
    reference = copy.deepcopy(model)
    frozen = {name: p.clone() for name, p in model.encoder.named_parameters()}
    before = run_dinomaly.trainable_state(model)
    log = train(model, images, steps=3, batch_size=4, device="cpu", log_every=1)
    assert len(log) == 3 and all(np.isfinite(e["loss"]) for e in log)
    assert all(torch.equal(p, frozen[name]) for name, p in model.encoder.named_parameters())
    after = run_dinomaly.trainable_state(model)
    assert set(after) == trainable
    assert sum(not torch.equal(after[name], before[name]) for name in after) > len(after) // 2
    # anomalib 부품으로 풀어 쓴 같은 스텝: 손실과 가중치가 비트 단위로 같다.
    assert _reference_train(reference, images, steps=3, batch_size=4, seed=0) == [e["loss"] for e in log]
    expected = run_dinomaly.trainable_state(reference)
    assert all(torch.equal(after[name], expected[name]) for name in after)

    result = predict(model, images, batch_size=5, device="cpu")
    assert result.image_scores.shape == (8,) and result.maps.shape == (8, 256, 256)
    assert np.isfinite(result.image_scores).all() and np.isfinite(result.maps).all()
    assert result.maps.min() > -1e-3 and result.maps.max() < 2 + 1e-3  # 1 - 코사인 유사도

    # 저장 파일에는 학습 파라미터만 있고, 같은 인코더와 함께 쓰면 같은 점수를 다시 낸다.
    path = tmp_path / "model.pt"
    run_dinomaly.save_model(path, model, encoder=encoder, steps=3, amp=False)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert set(saved) == {"encoder", "steps", "amp", "state"} and set(saved["state"]) == trainable
    torch.manual_seed(0)  # 위와 같은 무작위 '사전학습' 인코더
    loaded, meta = run_dinomaly.load_model(path)
    assert meta == {"encoder": encoder, "steps": 3, "amp": False} and not loaded.training
    again = predict(loaded, images, batch_size=5, device="cpu")
    assert np.array_equal(again.image_scores, result.image_scores)
    assert np.array_equal(again.maps, result.maps)

    saved["state"].pop(sorted(trainable)[0])
    torch.save(saved, path)
    with pytest.raises(ValueError, match="trainable parameters"):
        run_dinomaly.load_model(path)


# ---------------------------------------------------------------- 평가와 명령줄(CLI)


# 평가·CLI 테스트용 가짜 캐시. 결함 이미지에만 흰 사각형과 마스크를 넣어 가짜 모델(밝기 점수)이 결함을 확실히
# 높게 매기게 한다. ledger를 주면 봉인 테스트 이미지(SEALED_ROLES)를 내주기 전에 장부에 이미 줄이 있는지
# 단언하므로, '읽기 전에 기록' 규칙을 캐시 쪽에서 강제한다. requested에 요청된 이미지 이름을 순서대로 남긴다.
class FakeCache:
    """ImageCache 대신 쓰는 메모리 안 가짜: 결함 이미지에만 정상에 없는 밝은 사각형이 있다."""

    size = SIZE

    def __init__(self, manifest, ledger=None):
        rng = np.random.default_rng(0)
        self._images, self._masks, self._role = {}, {}, {}
        self._ledger = ledger
        self.requested: list[str] = []
        for row in manifest:
            img = rng.integers(100, 140, (SIZE, SIZE, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            if row.label == "anomaly":
                img[8:16, 8:16] = 255
                mask[64:128, 64:128] = 1
            self._images[row.image] = img
            self._masks[row.image] = mask
            self._role[row.image] = row.role

    def _note(self, rows):
        if self._ledger is not None and any(self._role[r.image] in SEALED_ROLES for r in rows):
            # 봉인 이미지 읽기는 이미 장부에 기록돼 있어야 한다.
            assert self._ledger.exists() and self._ledger.read_text(encoding="utf-8").strip()
        self.requested += [r.image for r in rows]

    def images(self, rows):
        self._note(rows)
        return np.stack([self._images[r.image] for r in rows])

    def masks(self, rows):
        self._note(rows)
        return np.stack([self._masks[r.image] for r in rows])


# 범주 toy 하나의 가짜 매니페스트: 정상 풀 40장(겹 0~4), dev 결함 6장(결함 유형 'hole|scratch'와 'hole'을
# 번갈아), test 정상 10장, test 결함 4장.
def _manifest():
    rows = []
    for i in range(40):
        rows.append(ManifestRow(f"toy/n{i}.JPG", "", "toy", "normal", "pool_normal", i % 5, ""))
    for i in range(6):
        kind = "hole" if i % 2 else "hole|scratch"
        rows.append(ManifestRow(f"toy/d{i}.JPG", f"toy/d{i}.png", "toy", "anomaly", "dev_defect", -1, kind))
    for i in range(10):
        rows.append(ManifestRow(f"toy/tn{i}.JPG", "", "toy", "normal", "test_normal", -1, ""))
    for i in range(4):
        mask = f"toy/td{i}.png"
        rows.append(ManifestRow(f"toy/td{i}.JPG", mask, "toy", "anomaly", "test_defect", -1, "hole"))
    return rows


# 평가 결과 npz에 반드시 있어야 하는 키. compare.py가 Dinomaly·지도 학습 결과를 읽는 2단계 공통 형식(eval_*,
# cal_score)과 픽셀 지표·PRO 히스토그램이다.
COMMON_KEYS = {
    "eval_images",
    "eval_labels",
    "eval_defect_types",
    "eval_score",
    "cal_score",
    "pixel_auroc",
    "aupro",
    "pro_edges",
    "pro_normal",
    "pro_components",
    "pro_component_image",
}


# dev 절차의 eval_category가 겹 0 정상 8장 + dev 결함 6장을 평가 세트로 쓰고 봉인 이미지를 요청하지 않는지,
# 공통 형식 키와 dtype을 갖춰 저장하는지 본다. dev에서는 겹 0 정상이 곧 평가 정상이라 보정 점수(cal_score)는
# 비어 있다. 결함 유형 문자열이 그대로 남고, 가짜 모델에서는 결함이 모두 정상보다 높아 픽셀 AUROC·AUPRO가
# 높으며, PRO 히스토그램은 2,000칸이고 결함 영역 6개가 각자 이미지 번호를 가진다. 256 맵도 float16으로 따로
# 저장된다.
def test_eval_category_dev_writes_the_common_format(tmp_path):
    manifest = _manifest()
    cache = FakeCache(manifest)
    info = run_dinomaly.eval_category(
        TinyModel(), "dev", "toy", manifest, cache, "cpu", False, tmp_path, batch_size=5
    )
    assert (info["category"], info["eval_normal"], info["eval_defect"], info["cal"]) == ("toy", 8, 6, 0)
    assert not any(image.startswith(("toy/tn", "toy/td")) for image in cache.requested)

    with np.load(tmp_path / "toy.npz") as z:
        assert COMMON_KEYS <= set(z.files)
        assert z["eval_images"].tolist() == [f"toy/n{i}.JPG" for i in range(0, 40, 5)] + [
            f"toy/d{i}.JPG" for i in range(6)
        ]
        assert z["eval_labels"].dtype == np.int8 and z["eval_labels"].tolist() == [0] * 8 + [1] * 6
        assert z["eval_defect_types"].tolist() == [""] * 8 + ["hole|scratch", "hole"] * 3
        assert z["eval_score"].dtype == np.float32 and z["eval_score"].shape == (14,)
        assert z["cal_score"].dtype == np.float32 and z["cal_score"].shape == (0,)
        assert z["cal_images"].shape == (0,)
        # 가짜 모델은 밝기로 채점하므로 모든 결함이 모든 정상보다 높다(픽셀 단위로도).
        assert z["eval_score"][8:].min() > z["eval_score"][:8].max()
        assert z["pixel_auroc"].dtype == np.float64 and float(z["pixel_auroc"]) > 0.95
        assert z["aupro"].dtype == np.float64 and float(z["aupro"]) > 0.8
        assert z["pro_edges"].shape == (2001,) and z["pro_normal"].shape == (14, 2000)
        assert z["pro_components"].shape == (6, 2000)
        assert z["pro_component_image"].tolist() == list(range(8, 14))
    maps = np.load(tmp_path / "toy_maps.npy")
    assert maps.shape == (14, 256, 256) and maps.dtype == np.float16
    assert maps[8:, 64:128, 64:128].min() > maps[:8].max()


# test 절차는 allow_test 없이는 SealedTestError로 멈추고 아무 이미지도 요청하지 않는다. 허용하면 봉인
# 테스트(정상 10, 결함 4)를 평가하고, 임계값용 보정 점수는 학습에 쓰지 않은 겹 0 정상 8장으로 따로
# 채점한다(홀드아웃, experiments.md 2단계). Dinomaly는 학습이 비싸 PatchCore처럼 교차 적합을 하지 않기
# 때문이다.
def test_eval_category_test_protocol_needs_permission(tmp_path):
    manifest = _manifest()
    cache = FakeCache(manifest)
    with pytest.raises(SealedTestError):
        run_dinomaly.eval_category(TinyModel(), "test", "toy", manifest, cache, "cpu", False, tmp_path)
    assert cache.requested == [] and not (tmp_path / "toy.npz").exists()

    info = run_dinomaly.eval_category(TinyModel(), "test", "toy", manifest, cache, "cpu", True, tmp_path)
    assert (info["eval_normal"], info["eval_defect"], info["cal"]) == (10, 4, 8)
    with np.load(tmp_path / "toy.npz") as z:
        assert z["eval_images"].tolist() == [f"toy/tn{i}.JPG" for i in range(10)] + [
            f"toy/td{i}.JPG" for i in range(4)
        ]
        # 임계값은 학습 풀에 들지 않는 겹 0 정상에서 나온다.
        assert z["cal_images"].tolist() == [f"toy/n{i}.JPG" for i in range(0, 40, 5)]
        assert z["cal_score"].shape == (8,) and z["cal_score"].dtype == np.float32
        assert z["eval_score"][10:].min() > max(z["eval_score"][:10].max(), z["cal_score"].max())


# eval --protocol test는 --allow-test와 --stage가 모두 있어야 한다. ImageCache를 None으로 바꿔 두어, 거절하기
# 전에 캐시를 열려고 하면 바로 오류가 나게 한다. 장부와 출력 폴더는 생기지 않아야 하고, 절차를 밝히지 않은
# eval도 거절한다.
def test_cli_refuses_test_protocol_without_flags(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(run_dinomaly.paths, "TEST_LEDGER", ledger)
    monkeypatch.setattr(run_dinomaly, "ImageCache", None)  # 거절 전에는 아무것도 열면 안 된다
    base = ["eval", "--protocol", "test", "--out", str(tmp_path / "out")]
    for extra in ([], ["--allow-test"], ["--stage", "2"]):
        with pytest.raises(SystemExit):
            run_dinomaly.main(base + extra)
    assert not ledger.exists() and not (tmp_path / "out").exists()
    with pytest.raises(SystemExit):
        run_dinomaly.main(["eval", "--out", str(tmp_path / "out")])  # 절차를 밝혀야 한다


# CLI를 가짜 매니페스트(CSV로 실제 저장)·가짜 캐시·가짜 모델·임시 장부로 연결한 환경. build_model을 가로채
# 만들어진 모델·옵션·초기 가중치를 기록하고, 캐시를 연 크기도 기록한다. git_commit은 고정값을 돌려준다.
@pytest.fixture
def toy_project(tmp_path, monkeypatch):
    """가짜 매니페스트·가짜 캐시·가짜 모델과 tmp_path 아래 장부에 연결한 CLI 환경."""
    manifest = _manifest()
    ledger = tmp_path / "ledger.jsonl"
    cache = FakeCache(manifest, ledger=ledger)
    built = []
    initial = []
    sizes = []
    options = []

    def fake_cache(root, size):
        sizes.append(size)
        return cache

    def fake_build(encoder_name=run_dinomaly.ENCODER, **build_options):
        built.append((encoder_name, TinyModel()))
        options.append(build_options)
        initial.append(built[-1][1].decoder.weight.detach().clone())
        return built[-1][1]

    write_manifest(manifest, tmp_path / "visa.csv")
    monkeypatch.setattr(run_dinomaly.paths, "VISA_MANIFEST", tmp_path / "visa.csv")
    monkeypatch.setattr(run_dinomaly.paths, "TEST_LEDGER", ledger)
    monkeypatch.setattr(run_dinomaly.paths, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(run_dinomaly, "ImageCache", fake_cache)
    monkeypatch.setattr(run_dinomaly, "build_model", fake_build)
    monkeypatch.setattr(run_dinomaly, "git_commit", lambda: "abc1234")
    return {
        "cache": cache,
        "built": built,
        "initial": initial,
        "ledger": ledger,
        "outputs": tmp_path / "outputs",
        "sizes": sizes,
        "options": options,
    }


# CLI 학습을 빠르게 돌리는 공통 인자(6스텝, 배치 4, 2스텝마다 기록, CPU)와 범주 지정.
TRAIN_ARGS = ["train", "--steps", "6", "--batch-size", "4", "--log-every", "2", "--device", "cpu"]
TOY = ["--categories", "toy"]


# train → eval(dev) 흐름 전체를 본다. 기본 구성 dm은 392px 캐시와 옵션 없는 anomalib 모델을 쓰고, 학습은 dev
# 정상 풀(겹 1~4)만 읽어야 한다(겹 0 정상·결함·봉인 이미지 없음). train.json에는 실제로 돈 정밀도(CPU라 amp
# False)와 등록 상수가 남고, 저장 파일에는 학습 파라미터만 들어 있으며 .part 파일이 남지 않는다. 끝난 학습은
# 실수로 덮어쓰지 않는다.
# eval은 run.json에 모델 경로와 sha256, 학습·채점 정밀도를 남기고, 다시 만든 모델의 새 초기값이 아니라 저장된
# 가중치로 채점해야 한다. dev는 장부를 쓰지 않고, 끝난 평가는 --overwrite 없이 다시 하지 않는다.
def test_cli_train_then_eval_dev(toy_project, capsys):
    outputs = toy_project["outputs"]
    torch.manual_seed(123)
    run_dinomaly.main(TRAIN_ARGS + TOY)
    # 2단계 모델: 392px 캐시와 옵션 없는 anomalib 모델.
    assert toy_project["sizes"] == [run_dinomaly.IMG_SIZE] == [392] and toy_project["options"] == [{}]

    # dev 정상 풀(겹 1~4)로만 학습한다: 겹 0 정상, 결함, 테스트 이미지는 없다.
    pool = [f"toy/n{i}.JPG" for i in range(40) if i % 5 != 0]
    assert toy_project["cache"].requested == pool
    train_dir = outputs / "dm"
    info = json.loads((train_dir / "train.json").read_text(encoding="utf-8"))
    assert info["method"] == "dinomaly" and info["commit"] == "abc1234" and info["device"] == "cpu"
    assert info["config"]["encoder"] == run_dinomaly.ENCODER and info["config"]["steps"] == 6
    # `--no-amp`를 주지 않았지만 fp16 autocast는 CUDA에만 있으므로, 기록은 실제로 돈 정밀도를 적는다.
    assert info["config"]["batch_size"] == 4 and info["config"]["amp"] is False
    assert info["config"]["img_size"] == 392 and info["config"]["grad_clip_norm"] == 0.1
    assert info["train_images"] == 32 and info["train_images_by_category"] == {"toy": 32}
    log = [json.loads(line) for line in (train_dir / "train_log.jsonl").read_text("utf-8").splitlines()]
    assert [e["step"] for e in log] == [2, 4, 6] and info["final_loss"] == log[-1]["loss"]
    saved = torch.load(train_dir / "model.pt", map_location="cpu", weights_only=True)
    assert saved["encoder"] == run_dinomaly.ENCODER and saved["steps"] == 6 and saved["amp"] is False
    assert set(saved["state"]) == {"decoder.weight", "decoder.bias"}
    trained = toy_project["built"][0][1]
    assert torch.equal(saved["state"]["decoder.weight"], trained.decoder.weight.detach())
    assert not list(train_dir.glob("*.part"))

    # 끝난 학습은 실수로 덮어쓰지 않는다.
    with pytest.raises(SystemExit):
        run_dinomaly.main(TRAIN_ARGS + TOY)
    assert len(toy_project["built"]) == 1

    capsys.readouterr()
    run_dinomaly.main(["eval", "--protocol", "dev", "--device", "cpu", "--batch-size", "5"] + TOY)
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [p["category"] for p in printed] == ["toy"]
    eval_dir = outputs / "dm-dev"
    run = json.loads((eval_dir / "run.json").read_text(encoding="utf-8"))
    assert set(run) == {"method", "protocol", "commit", "device", "config", "categories", "total_s"}
    assert (run["method"], run["protocol"], run["commit"], run["device"]) == (
        "dinomaly",
        "dev",
        "abc1234",
        "cpu",
    )
    assert run["config"]["encoder"] == run_dinomaly.ENCODER and run["config"]["train_steps"] == 6
    assert run["config"]["model"] == str(train_dir / "model.pt") and len(run["config"]["model_sha256"]) == 64
    assert run["config"]["amp"] is False and run["config"]["train_amp"] is False
    assert run["config"]["score_precision"] == "fp32"
    assert run["categories"][0]["category"] == "toy" and run["categories"][0]["eval_defect"] == 6
    # 평가는 다시 만든 모델의 새 초기값이 아니라 저장된 가중치를 썼다.
    name, evaluated = toy_project["built"][1]
    assert name == run_dinomaly.ENCODER
    assert torch.equal(evaluated.decoder.weight.detach(), saved["state"]["decoder.weight"])
    with np.load(eval_dir / "toy.npz") as z:
        assert COMMON_KEYS <= set(z.files)
        assert z["cal_score"].shape == (0,) and z["eval_score"].shape == (14,)
    assert np.load(eval_dir / "toy_maps.npy").shape == (14, 256, 256)
    assert not toy_project["ledger"].exists()

    with pytest.raises(SystemExit):
        run_dinomaly.main(["eval", "--protocol", "dev", "--device", "cpu"] + TOY)
    run_dinomaly.main(["eval", "--protocol", "dev", "--device", "cpu", "--overwrite"] + TOY)


# 봉인 테스트 평가의 순서를 본다. 모델 파일이 없으면 장부 기록·이미지 읽기 전에 멈춘다. 정상 실행에서는
# FakeCache가 봉인 이미지를 내줄 때마다 장부 줄이 이미 있는지 단언하므로 '기록 먼저'가 강제되고, 장부에는
# stage·config(dm)·커밋·메모가 담긴 한 줄만 생긴다. test 결과에는 겹 0 보정 점수 8개가 함께 저장된다.
def test_cli_eval_test_protocol_records_the_read_first(toy_project):
    outputs = toy_project["outputs"]
    run_dinomaly.main(TRAIN_ARGS + TOY + ["--out", str(outputs / "m")])
    model = ["--model", str(outputs / "m" / "model.pt")]
    common = ["eval", "--protocol", "test", "--device", "cpu"] + TOY + model

    # 모델 파일이 없으면 아무것도 기록하거나 읽기 전에 멈춘다.
    toy_project["cache"].requested.clear()
    with pytest.raises(SystemExit):
        run_dinomaly.main(common[:-1] + [str(outputs / "nowhere.pt"), "--allow-test", "--stage", "2"])
    assert not toy_project["ledger"].exists() and toy_project["cache"].requested == []

    # FakeCache는 봉인 이미지를 내주기 전에 장부 줄이 있는지 단언한다.
    run_dinomaly.main(common + ["--allow-test", "--stage", "2", "--note", "toy"])
    lines = toy_project["ledger"].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert (entry["stage"], entry["config"], entry["commit"], entry["note"]) == ("2", "dm", "abc1234", "toy")
    run = json.loads((outputs / "dm-test" / "run.json").read_text(encoding="utf-8"))
    assert run["protocol"] == "test" and run["categories"][0]["cal"] == 8
    with np.load(outputs / "dm-test" / "toy.npz") as z:
        assert z["eval_labels"].tolist() == [0] * 10 + [1] * 4
        assert z["cal_score"].shape == (8,)


# 매니페스트에 없는 범주는 모델을 만들기 전에 거절하고 출력 폴더도 만들지 않는다.
def test_cli_rejects_unknown_categories(toy_project):
    with pytest.raises(SystemExit):
        run_dinomaly.main(TRAIN_ARGS + ["--categories", "nothing"])
    assert toy_project["built"] == [] and not (toy_project["outputs"] / "dm").exists()


# 같은 범주를 두 번 적으면 학습은 그 이미지를 두 번 보고 평가는 범주를 두 번 적게 되므로, train과 eval 모두
# 아무것도 읽기 전에 거절해야 한다.
def test_cli_rejects_repeated_categories(toy_project):
    outputs = toy_project["outputs"]
    with pytest.raises(SystemExit):
        run_dinomaly.main(TRAIN_ARGS + ["--categories", "toy", "toy"])  # 모든 이미지로 두 번씩 학습하게 된다
    assert toy_project["built"] == [] and toy_project["cache"].requested == []
    assert not (outputs / "dm").exists()

    run_dinomaly.main(TRAIN_ARGS + TOY)
    toy_project["cache"].requested.clear()
    with pytest.raises(SystemExit):
        run_dinomaly.main(["eval", "--protocol", "dev", "--device", "cpu", "--categories", "toy", "toy"])
    assert toy_project["cache"].requested == [] and not (outputs / "dm-dev").exists()


# CLI의 --seed가 초기 가중치와 학습 결과를 모두 고정하는지 본다. 실행 전 프로세스 난수 상태(noise)가 달라도
# seed가 같으면 초기값·최종 가중치가 같고, seed가 다르면 달라야 한다. seed는 train.json에도 남는다.
def test_cli_train_seed_fixes_the_initial_weights_and_the_result(toy_project):
    outputs = toy_project["outputs"]
    states = {}
    for name, seed, noise in (("a", 0, 11), ("b", 0, 22), ("c", 1, 11)):
        torch.manual_seed(noise)  # 실행 전에 프로세스가 뽑은 난수는 결과를 바꾸면 안 된다
        run_dinomaly.main(TRAIN_ARGS + TOY + ["--seed", str(seed), "--out", str(outputs / name)])
        saved = torch.load(outputs / name / "model.pt", map_location="cpu", weights_only=True)
        states[name] = saved["state"]
        info = json.loads((outputs / name / "train.json").read_text(encoding="utf-8"))
        assert info["config"]["seed"] == seed
    init_a, init_b, init_c = toy_project["initial"]
    assert torch.equal(init_a, init_b) and not torch.equal(init_a, init_c)
    assert all(torch.equal(states["a"][key], states["b"][key]) for key in states["a"])
    assert not torch.equal(states["a"]["decoder.weight"], states["c"]["decoder.weight"])


# 평가 폴더를 읽는 쪽은 폴더의 모든 *.npz를 그 실행의 일부로 보므로, 다시 평가할 때 이전 실행의 결과 파일(다른
# 범주 gone.npz 등)을 남김없이 지워야 한다. 결과 파일이 아닌 notes.txt는 남긴다. run.json 없이 결과만 남은
# 폴더(중단된 실행)는 --overwrite 없이는 건드리지 않고 거절하며, 다시 하다 중간에 깨지면 run.json도 결과도
# 없는 상태가 되어 끝난 실행처럼 보이지 않아야 한다.
def test_cli_eval_overwrite_clears_the_results_of_the_earlier_run(toy_project, monkeypatch):
    outputs = toy_project["outputs"]
    run_dinomaly.main(TRAIN_ARGS + TOY)
    base = ["eval", "--protocol", "dev", "--device", "cpu"] + TOY
    eval_dir = outputs / "dm-dev"
    run_dinomaly.main(base)
    # 이전 실행이 범주를 하나 더 다룬 것처럼 만든다: *.npz를 나열하는 쪽은 두 실행을 섞게 된다.
    shutil.copy(eval_dir / "toy.npz", eval_dir / "gone.npz")
    shutil.copy(eval_dir / "toy_maps.npy", eval_dir / "gone_maps.npy")
    (eval_dir / "notes.txt").write_text("not a result file", encoding="utf-8")
    run_dinomaly.main(base + ["--overwrite"])
    assert sorted(p.name for p in eval_dir.iterdir()) == ["notes.txt", "run.json", "toy.npz", "toy_maps.npy"]

    # run.json 없는 결과 파일(중단된 실행)도 새 실행에 섞이지 않고, --overwrite 없이는 아무것도 지우지
    # 않는다.
    partial = outputs / "partial"
    partial.mkdir()
    shutil.copy(eval_dir / "toy.npz", partial / "gone.npz")
    toy_project["cache"].requested.clear()
    with pytest.raises(SystemExit):
        run_dinomaly.main(base + ["--out", str(partial)])
    assert [p.name for p in partial.iterdir()] == ["gone.npz"] and toy_project["cache"].requested == []
    run_dinomaly.main(base + ["--out", str(partial), "--overwrite"])
    assert sorted(p.name for p in partial.iterdir()) == ["run.json", "toy.npz", "toy_maps.npy"]

    # 다시 하다 중간에 깨진 실행이 바꾸려던 끝난 실행처럼 보이면 안 된다.
    def broken(*args, **kwargs):
        raise RuntimeError("stopped")

    monkeypatch.setattr(run_dinomaly, "eval_category", broken)
    with pytest.raises(RuntimeError, match="stopped"):
        run_dinomaly.main(base + ["--overwrite"])
    assert [p.name for p in eval_dir.iterdir()] == ["notes.txt"]


# eval의 채점 정밀도 규칙을 본다. 기본은 모델 파일에 기록된 학습 정밀도를 따르고(fp16으로 학습했으면 fp16
# 요청), --amp/--no-amp가 있으면 그것을 따른다. CPU는 fp16을 할 수 없으므로 run.json에는 실제로 돈 fp32가
# 남는다. 학습 정밀도를 기록하지 않은 옛 파일은 플래그 없이는 거절한다. 6단계 채점은 fp16이 점수를 반올림해
# 정상끼리 동점을 만들기 때문에 --no-amp(fp32)로 했다(run_dinomaly predict docstring, experiments.md 6단계).
def test_cli_eval_precision_follows_the_training_run(toy_project, monkeypatch):
    outputs = toy_project["outputs"]
    run_dinomaly.main(TRAIN_ARGS + TOY)
    saved = torch.load(outputs / "dm" / "model.pt", map_location="cpu", weights_only=True)
    asked = []
    real_predict = run_dinomaly.predict

    def spy(model, images, **kwargs):
        asked.append(kwargs["amp"])
        return real_predict(model, images, **kwargs)

    monkeypatch.setattr(run_dinomaly, "predict", spy)

    # 저장 파일의 amp 기록만 바꾼 복사본으로 eval을 돌려, predict가 받은 amp 값과 run.json의 config를
    # 돌려준다.
    def evaluate(train_amp, *flags):
        """`train_amp`로 학습했다고 적힌 모델 파일로 채점한다(None이면 아무것도 적혀 있지 않다)."""
        asked.clear()
        path = outputs / "model-under-test.pt"
        content = {key: value for key, value in saved.items() if key != "amp"}
        if train_amp is not None:
            content["amp"] = train_amp
        torch.save(content, path)
        out = outputs / "precision"
        command = ["eval", "--protocol", "dev", "--device", "cpu", "--overwrite"] + TOY
        run_dinomaly.main(command + ["--model", str(path), "--out", str(out), *flags])
        config = json.loads((out / "run.json").read_text(encoding="utf-8"))["config"]
        assert len(set(asked)) == 1
        return asked[0], config

    asked_amp, config = evaluate(False)
    assert asked_amp is False and config["train_amp"] is False
    assert config["amp"] is False and config["score_precision"] == "fp32"
    # fp16으로 학습한 모델은 따로 말하지 않으면 fp16으로 채점한다. CPU는 그 요청을 들어줄 수 없고,
    # run.json은 실제로 돈 정밀도를 기록한다.
    asked_amp, config = evaluate(True)
    assert asked_amp is True and config["train_amp"] is True
    assert config["amp"] is False and config["score_precision"] == "fp32"
    assert evaluate(True, "--no-amp")[0] is False
    assert evaluate(False, "--amp")[0] is True
    # 어떻게 학습했는지 적혀 있지 않은 파일: 정밀도를 직접 밝혀야 한다.
    with pytest.raises(SystemExit):
        evaluate(None)
    assert asked == []
    asked_amp, config = evaluate(None, "--no-amp")
    assert asked_amp is False and config["train_amp"] is None
    assert evaluate(None, "--amp")[0] is True


# ---------------------------------------------------------------- 6단계: 구성과 관문


# 등록된 구성 표가 experiments.md 2·6단계와 같은지 고정한다. dm은 2단계 등록값(ViT-B, 392px, 5,000스텝, 배치
# 16) 그대로이고 build_model 옵션이 없어야 2단계 결과와 같은 모델이 된다. dms-*는 ViT-S, 이름의 크기,
# 10,000스텝, 배치 16, 인코더 10블록, 입력 크기 고정, 이름이 -car로 끝날 때만 CAR, 기본 어텐션은 scaled(키를
# 토큰 수로 나누는 패치)다. 어텐션을 original로 바꾸면 옵션에서 빠지고 fp32로 바꿀 수 있으며, 모르는 방식은
# 거절한다.
def test_registered_configs():
    configs = run_dinomaly.CONFIGS
    assert set(configs) == {"dm", "dms-252", "dms-280", "dms-294", "dms-308", "dms-252-car", "dms-280-car"}
    dm = configs["dm"]
    # 2단계는 그대로다: 등록 당시의 상수이고 build_model 옵션이 없다.
    assert (dm.encoder, dm.img_size, dm.steps, dm.batch_size) == (run_dinomaly.ENCODER, 392, 5000, 16)
    assert dm.build_options() == {} and dm.build_options("original") == {}
    for name, cfg in configs.items():
        if name == "dm":
            continue
        size = int(name.split("-")[1])
        assert cfg.encoder == run_dinomaly.ENCODER_S == "vit_small_patch14_reg4_dinov2"
        assert (cfg.img_size, cfg.steps, cfg.batch_size, cfg.encoder_blocks) == (size, 10_000, 16, 10)
        assert cfg.fixed_size and cfg.context_recentering == name.endswith("-car")
        expected = {"img_size": size, "encoder_blocks": 10, "attention": "scaled"}
        if cfg.context_recentering:
            expected["context_recentering"] = True
        assert cfg.build_options() == expected
        assert "attention" not in cfg.build_options("original")
        assert cfg.build_options("fp32")["attention"] == "fp32"
    with pytest.raises(ValueError):
        configs["dms-280"].build_options("fast")


# 관문 2 (나)의 판정 함수를 규칙(10스텝 구간 평균 손실이 fp32 기준과 상대 3% 이내, 모두 유한, 건너뛴 스텝 0,
# 스텝 수 같음)대로 시험한다. 구간 안 한 스텝이 튀어도 구간 평균이 3% 안이면 통과하고, 한 구간이라도 4% 차이,
# 건너뛴 스텝 1, NaN, 스텝 부족(이유에 '14 of 30')이면 실패다. 기준 길이가 구간의 배수가 아니면 ValueError다.
def test_gate_verdict():
    ref = [1.0] * 30
    ok = run_dinomaly.gate_verdict(ref, [1.02] * 30, skipped=0)
    assert ok["passed"] and ok["window_rel_diff"] == pytest.approx([0.02] * 3)
    # 구간 안 한 스텝이 튀어도 구간 평균이 3% 안이면 통과한다.
    noisy = [1.0] * 30
    noisy[4] = 1.25
    assert run_dinomaly.gate_verdict(ref, noisy, skipped=0)["passed"]
    assert not run_dinomaly.gate_verdict(ref, [1.0] * 20 + [1.04] * 10, skipped=0)["passed"]
    assert not run_dinomaly.gate_verdict(ref, [1.0] * 30, skipped=1)["passed"]
    assert not run_dinomaly.gate_verdict(ref, [1.0] * 29 + [float("nan")], skipped=0)["passed"]
    short = run_dinomaly.gate_verdict(ref, [1.0] * 14, skipped=0)
    assert not short["passed"] and "14 of 30" in short["reason"]
    with pytest.raises(ValueError):
        run_dinomaly.gate_verdict([1.0] * 25, [1.0] * 25, skipped=0)


# 관문 2 (가)와 6단계 모델 저장을 실제 DinomalyModel(무작위 가중치, 56px)로 본다. attention_parity는 같은
# 초기값에서 만든 원본 어텐션과 패치 어텐션 모델의 fp32 점수·맵·2스텝 손실 차이가 1e-5 이하일 때 통과해야
# 한다(패치는 수식상 출력이 같다, experiments.md 관문 2).
# 이어 dms-252-car 구성으로 만든 모델이 CAR·scaled 어텐션·인코더 10블록인지 확인하고, 2스텝 학습 후
# save_model에 구성 이름과 옵션을 함께 저장하면 load_model이 같은 옵션(고정 입력 크기 포함)으로 같은 모델을
# 다시 만들어 같은 점수를 내는지 본다. ONNX 서빙(dinomaly_serving)이 이 load_model로 학습한 모델을 읽는다.
def test_attention_parity_and_save_load_of_a_stage6_model(monkeypatch, tmp_path):
    timm = pytest.importorskip("timm")
    import timm.models.vision_transformer as vit

    from defect_inspect.dinomaly_model import attention_mode

    real_create = timm.create_model

    def create_without_download(name, *args, **kwargs):
        kwargs["pretrained"] = False
        return real_create(name, *args, **kwargs)

    monkeypatch.setattr(timm, "create_model", create_without_download)
    monkeypatch.setattr(vit, "resample_abs_pos_embed", vit.resample_abs_pos_embed)
    cfg = run_dinomaly.CONFIGS["dms-252-car"]
    size = 56
    options = {**cfg.build_options(), "img_size": size}

    def build(attention):
        return run_dinomaly.build_model(cfg.encoder, **{**options, "attention": attention})

    images = _images(12, seed=7, size=size)
    result = run_dinomaly.attention_parity(build, images, steps=2, batch_size=4, device="cpu")
    assert result["passed"] and len(result["loss_rel_diff"]) == 2
    assert max(result["loss_rel_diff"]) <= 1e-5 and result["map_abs_diff"] <= 1e-5
    assert result["losses_original"] == pytest.approx(result["losses_scaled"], rel=1e-5)

    torch.manual_seed(0)
    model = run_dinomaly.build_model(cfg.encoder, **options)
    assert model.use_context_recentering and attention_mode(model) == "scaled"
    assert len(model.encoder.feature_extractor.blocks) == 10
    train(model, images, steps=2, batch_size=4, device="cpu", log_every=1)
    path = tmp_path / "model.pt"
    run_dinomaly.save_model(
        path, model, encoder=cfg.encoder, steps=2, amp=False, config=cfg.name, options=options
    )
    torch.manual_seed(0)  # 같은 무작위 '사전학습' 인코더
    loaded, meta = run_dinomaly.load_model(path)
    assert meta == {"encoder": cfg.encoder, "steps": 2, "amp": False, "config": cfg.name, "options": options}
    assert loaded.use_context_recentering and attention_mode(loaded) == "scaled"
    assert loaded.fixed_img_size == size and len(loaded.encoder.feature_extractor.blocks) == 10
    a = predict(model, images[:3], batch_size=3, device="cpu", amp=False)
    b = predict(loaded, images[:3], batch_size=3, device="cpu", amp=False)
    assert np.array_equal(a.image_scores, b.image_scores)


# CLI로 6단계 구성(dms-280)을 학습·평가한다. 280px 캐시와 구성 옵션으로 모델을 만들고, train.json·model.pt에
# 구성 이름·옵션이 남아야 한다. dm 평가가 dms 모델 파일을, dms 평가가 dm 모델 파일을 잘못 집으면(입력
# 크기·인코더가 다름) 거절한다. 봉인 평가(--no-amp, stage 6)는 run.json에 옵션과 fp32 채점을 남기고 장부
# config를 dms-280으로 쓴다.
def test_cli_trains_and_evaluates_a_stage6_config(toy_project, capsys):
    outputs = toy_project["outputs"]
    run_dinomaly.main(TRAIN_ARGS + TOY + ["--config", "dms-280"])
    assert toy_project["sizes"] == [280]
    assert toy_project["options"] == [{"img_size": 280, "encoder_blocks": 10, "attention": "scaled"}]
    info = json.loads((outputs / "dms-280" / "train.json").read_text(encoding="utf-8"))
    cfg = info["config"]
    assert (cfg["name"], cfg["encoder"], cfg["img_size"]) == ("dms-280", run_dinomaly.ENCODER_S, 280)
    assert cfg["steps"] == 6 and cfg["attention"] == "scaled" and "gate" not in cfg
    saved = torch.load(outputs / "dms-280" / "model.pt", map_location="cpu", weights_only=True)
    assert saved["config"] == "dms-280" and saved["options"] == toy_project["options"][0]

    # dm 평가는 6단계 모델을 집을 수 없고, 반대도 마찬가지다.
    stage6_model = ["--model", str(outputs / "dms-280" / "model.pt")]
    with pytest.raises(SystemExit):
        run_dinomaly.main(["eval", "--protocol", "dev", "--device", "cpu"] + TOY + stage6_model)
    run_dinomaly.main(TRAIN_ARGS + TOY)
    dm_model = ["--model", str(outputs / "dm" / "model.pt")]
    with pytest.raises(SystemExit):
        run_dinomaly.main(
            ["eval", "--protocol", "dev", "--device", "cpu", "--config", "dms-280"] + TOY + dm_model
        )
    capsys.readouterr()

    base = ["eval", "--protocol", "test", "--device", "cpu", "--config", "dms-280", "--no-amp"] + TOY
    run_dinomaly.main(base + ["--allow-test", "--stage", "6"])
    run = json.loads((outputs / "dms-280-test" / "run.json").read_text(encoding="utf-8"))
    assert run["config"]["name"] == "dms-280" and run["config"]["img_size"] == 280
    assert run["config"]["options"] == toy_project["options"][0]
    assert run["config"]["score_precision"] == "fp32"
    assert toy_project["sizes"][-1] == 280
    line = json.loads(toy_project["ledger"].read_text(encoding="utf-8").splitlines()[-1])
    assert line["stage"] == "6" and line["config"] == "dms-280"


# --precision-from gate.json이 관문 결과의 정밀도·어텐션을 학습에 쓰고, 관문 파일 경로와 sha256을 train.json에
# 남기는지 본다(어느 관문 결과로 학습했는지 추적). 관문은 dms-280으로 돌리고 같은 결과로 다른 크기(여기서는
# dms-308)를 학습하는 것이 README의 실제 절차다. fp16을 골랐어도 CPU에서는 실제 정밀도 fp32가 기록된다.
# 관문 파일과 --no-amp/--attention을 같이 주면 서로 충돌하므로 거절하고, choice가 없는 관문 파일도 거절한다.
# 등록된 dms 구성의 인코더는 --encoder로 바꿀 수 없다.
def test_cli_train_takes_precision_and_attention_from_a_gate_file(toy_project, tmp_path):
    outputs = toy_project["outputs"]
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"config": "dms-280", "choice": {"amp": True, "attention": "fp32"}}), "utf-8")
    base = TRAIN_ARGS + TOY + ["--config", "dms-308"]
    run_dinomaly.main(base + ["--precision-from", str(gate)])
    info = json.loads((outputs / "dms-308" / "train.json").read_text(encoding="utf-8"))
    assert info["config"]["attention"] == "fp32" and toy_project["options"][-1]["attention"] == "fp32"
    assert info["config"]["gate"]["file"] == str(gate) and len(info["config"]["gate"]["sha256"]) == 64
    assert info["config"]["amp"] is False  # fp16을 요청했지만 CPU에는 fp16 autocast가 없다

    again = base + ["--overwrite", "--precision-from", str(gate)]
    for extra in (["--no-amp"], ["--attention", "scaled"]):
        with pytest.raises(SystemExit):
            run_dinomaly.main(again + extra)
    gate.write_text("{}", "utf-8")
    with pytest.raises(SystemExit):
        run_dinomaly.main(again)
    with pytest.raises(SystemExit):  # 등록된 6단계 구성의 인코더는 고정
        run_dinomaly.main(TRAIN_ARGS + TOY + ["--config", "dms-252", "--encoder", run_dinomaly.ENCODER])


# 관문 2 (나)는 fp16 학습을 재는 것이라 fp16 autocast가 없는 CPU에서는 아무것도 읽지 않고 거절해야 한다.
def test_cli_gate_needs_cuda(toy_project):
    with pytest.raises(SystemExit):
        run_dinomaly.main(["gate", "--device", "cpu"] + TOY)
    assert toy_project["sizes"] == []


# run_gate의 선택 순서를 실제 학습 없이(_gate_run을 가짜로) 본다. fp32 + 원본 어텐션 기준 → fp16 + scaled →
# (실패할 때만) fp16 + 어텐션 fp32 순으로 돌고, 처음 통과한 fp16 변형을 고른다. 여기서는 scaled가 뒤 구간에서
# 20% 벗어나 실패하고 fp32 어텐션이 선택된다. fp16 변형이 모두 NaN으로 실패하면 fp32 + 원본 어텐션으로
# 학습하도록 고른다. 실제 결과는 첫 변형(fp16 + scaled)이 통과했다(experiments.md 관문 2 (나)).
def test_run_gate_picks_the_first_fp16_variant_that_passes(tmp_path, monkeypatch):
    calls = []

    def fake_run(build, attention, amp, images, *, log_path, **common):
        calls.append((attention, amp))
        losses = [1.0] * 20
        if attention == "scaled":
            losses = [1.0] * 10 + [1.2] * 10  # fp32에서 너무 멀다
        return {"attention": attention, "amp": amp, "error": "", "losses": losses, "skipped": 0}

    monkeypatch.setattr(run_dinomaly, "_gate_run", fake_run)
    result = run_dinomaly.run_gate(None, None, out_dir=tmp_path, steps=20)
    assert calls == [("original", False), ("scaled", True), ("fp32", True)]
    assert result["choice"] == {"amp": True, "attention": "fp32"}
    assert [v["verdict"]["passed"] for v in result["variants"]] == [False, True]

    def failing(build, attention, amp, images, *, log_path, **common):
        losses, error = ([1.0] * 5, "non-finite loss") if amp else ([1.0] * 20, "")
        return {"attention": attention, "amp": amp, "error": error, "losses": losses, "skipped": 0}

    monkeypatch.setattr(run_dinomaly, "_gate_run", failing)
    result = run_dinomaly.run_gate(None, None, out_dir=tmp_path, steps=20)
    assert result["choice"] == {"amp": False, "attention": "original"}
    assert all(not v["verdict"]["passed"] for v in result["variants"])
