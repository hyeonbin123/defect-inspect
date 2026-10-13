# [파일 역할] 3-B단계(M2AD 실제 조명 변화) 실행 스크립트 `defect_inspect.run_m2ad`를 검증하는 테스트 묶음이다.
# 검사기 하나((범주, 시점)마다 PatchCore 하나)가 만드는 npz의 모양·순서, 교차 적합 임계값, 조건 S/P/R 채점, 새
# 조명 재보정, 봉인(테스트 이미지 접근 허락)과 장부 기록 순서를 확인한다.
# [왜 필요한가] 3-B 전체 측정은 규칙상 한 번만 돌리고 장부에 남긴다(docs/experiments.md 3-B). 그래서 실제 M2AD
# 데이터와 GPU 없이, 작은 가짜 백본(GridMean)과 가짜 캐시(FakeCache)로 같은 코드 경로를 돌려 실행 전에 오류를
# 잡는다.
# [주요 구성] 가짜 데이터·백본 도우미 → 조건 목록·npz 모양 → 임계값·라벨·점수 → 재보정과 교차 적합의 누수 검사
# → 봉인 → CLI(장부 기록 순서, --check, --overwrite).
# [설계 메모] 기대값은 run_m2ad를 거치지 않고 patchcore 함수를 직접 불러 계산한 점수(direct_scores)와
# 비교한다. 같은 코드를 두 번 부르는 것이 아니라 '어떤 이미지와 어떤 뱅크로 채점했는가'를 독립적으로 다시 세워
# 보는 방식이다.

import hashlib
import json

import numpy as np
import pytest

# run_m2ad의 채점은 torch(PatchCore 특징·뱅크)가 필요하다. CI는 torch 없이 돌므로(.github/workflows/ci.yml)
# torch가 없으면 이 파일 전체를 건너뛴다. 그래서 아래 프로젝트 import는 이 줄 뒤에 온다.
torch = pytest.importorskip("torch")

# ledger·paths는 장부 경로와 커밋 조회를 테스트용으로 바꾸려고, calibrate·conditions·m2ad는 기대값(임계값,
# 교란 이미지, 겹·재보정 시편)을 실행과 따로 계산하려고 가져온다. SealedTestError는 허락 없이 테스트 이미지를
# 읽을 때 나야 하는 예외다.
from defect_inspect import ledger, paths, run_m2ad  # noqa: E402
from defect_inspect.calibrate import conformal_threshold  # noqa: E402
from defect_inspect.conditions import apply_condition, condition_names  # noqa: E402
from defect_inspect.configs import PatchCoreConfig  # noqa: E402
from defect_inspect.m2ad import ILLUMINATIONS, VIEWS, M2adRow, recal_specimens, specimen_folds  # noqa: E402
from defect_inspect.splits import SealedTestError  # noqa: E402

# 가짜 데이터의 크기. 16px 이미지를 GridMean이 2x2 격자로 줄이므로 이미지 한 장은 패치 4개(뱅크 4행)가 된다.
# 학습 시편 30개는 실제 3-B 규칙(학습 시편 30개로 뱅크와 5겹 교차 적합)과 같은 수다.
SIZE = 16
PATCHES = 4  # GridMean gives a 2x2 grid
N_TRAIN = 30
# 테스트 시편: 정상 4개(홀수 번호, 학습 시편 짝수 번호와 겹치지 않음)와 불량 5개. 불량 가운데 하나는 어느
# 조명에서도 결함이 안 보이는 시편(라벨 -1로 제외), 하나는 조명 05에서만 안 보이는 시편이다. 실제 M2AD 라벨
# 규칙(보이지 않거나 판별 불가인 결함 이미지는 뺀다)을 흉내 내 라벨이 '보여 준 이미지'에서 오는지 확인하려는
# 장치다.
# SAME_AS_REFERENCE: 조명 03의 가짜 이미지를 기준 조명 01과 똑같게 만들어, 같은 시편의 두 이미지가 겹을 나눠
# 새면 점수가 0이 되는 상황을 일부러 만든다.
NORMAL = [f"{2 * i + 1:03d}" for i in range(4)]
ANOMALOUS = ["hole_1_000", "hole_1_001", "hole_1_002", "scratch_1_003", "scratch_2_004"]
NEVER_VISIBLE = "hole_1_001"  # excluded everywhere
HIDDEN_UNDER_05 = "hole_1_002"  # excluded under illumination 05 only
SAME_AS_REFERENCE = "03"  # the fake images under this illumination equal those under 01


# 실제 백본(WideResNet, DINOv2) 대신 쓰는 작은 특징 추출기. 2x2 칸마다 평균 색(3차원)을 패치 특징으로 낸다.
# 학습 가중치가 없어 결정적이고 CPU에서 빠르므로, 점수를 손으로 다시 계산해 비교할 수 있다. name·dim 속성은
# patchcore 코드가 백본에서 읽는 값을 흉내 낸다.
class GridMean(torch.nn.Module):
    """Tiny stand-in for a backbone: the mean colour of each cell of a 2x2 grid."""

    name = "gridmean"
    dim = 3

    def forward(self, x):
        return torch.nn.functional.avg_pool2d(x.float(), SIZE // 2).permute(0, 2, 3, 1)


# M2AD 메타데이터(read_meta 결과)를 흉내 낸 행 목록을 만든다. 시점 3개 × 조명 10개마다 학습 정상 30장, 테스트
# 정상 4장, 테스트 불량 5장이다. 마지막에 경로의 SHA-256 순으로 섞어, run_m2ad가 입력 행 순서에 기대지 않고
# 스스로 시편 이름순으로 정렬하는지도 함께 확인한다.
def make_rows(categories=("Motor",)) -> list[M2adRow]:
    rows = []
    for category in categories:
        for view in VIEWS:
            for light in ILLUMINATIONS:
                for i in range(N_TRAIN):
                    name = f"{2 * i:03d}"
                    path = f"{category}/Good/{name}/A{view}_I{light}.png"
                    rows.append(M2adRow(category, "train", name, view, light, path, 0, 0))
                for name in NORMAL:
                    path = f"{category}/Good/{name}/A{view}_I{light}.png"
                    rows.append(M2adRow(category, "test", name, view, light, path, 0, 0))
                for name in ANOMALOUS:
                    label = 1
                    if name == NEVER_VISIBLE or (name == HIDDEN_UNDER_05 and light == "05"):
                        label = -1
                    path = f"{category}/NG/{name}/A{view}_I{light}.png"
                    rows.append(M2adRow(category, "test", name, view, light, path, 1, label))
    # The run must not depend on the order in which the rows arrive.
    return sorted(rows, key=lambda r: hashlib.sha256(r.img_path.encode()).hexdigest())


# M2adCache를 대신하는 가짜 캐시. 이미지 내용을 행에서 결정적으로 만든다: 시편·시점별 무작위 잡음(같은 시편은
# 조명이 달라도 같은 잡음), 조명마다 8단계씩 더 어둡게, 보이는 결함(label 1)은 왼쪽 위 칸을 255로 밝게 칠한다.
# reads에는 (분할, 조명들, 장수)를 남겨 어떤 이미지를 몇 번 읽었는지 검사하고, on_test_read 훅으로 테스트
# 이미지를 읽는 순간을 잡아 장부 줄이 그보다 먼저 있었는지 확인한다.
class FakeCache:
    """Images made from the row: specimen noise, an offset per illumination, a bright cell for defects."""

    size = SIZE

    def __init__(self):
        self.reads: list[tuple[str, str, int]] = []
        self.on_test_read = None

    def _image(self, row: M2adRow) -> np.ndarray:
        # 잡음 시드를 (범주, 시편, 시점)의 SHA-256에서 만든다. 조명은 키에 넣지 않아, 같은 시편의 조명별
        # 이미지는 밝기만 다르고 무늬는 같다(실제 데이터에서 같은 물체를 조명만 바꿔 찍은 것과 같은 구조).
        key = f"{row.category}/{row.specimen}/{row.view}"
        rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
        img = rng.integers(100, 140, (SIZE, SIZE, 3), dtype=np.uint8)
        light = ILLUMINATIONS.index(row.illumination)
        if row.illumination != SAME_AS_REFERENCE:
            img = img - np.uint8(8 * light)  # every new illumination is darker by eight more levels
        if row.label == 1:
            img[:8, :8] = 255  # a bright cell that no normal image has
        return img

    def images(self, rows):
        # 테스트 분할 이미지를 읽으려는 순간 훅을 먼저 부른다. CLI 테스트가 이 시점에 장부 줄 수를 세어 '장부
        # 기록이 테스트 이미지 읽기보다 먼저'라는 규칙을 확인한다.
        splits = {r.split for r in rows}
        if "test" in splits and self.on_test_read is not None:
            self.on_test_read()
        for split in sorted(splits):
            lights = sorted({r.illumination for r in rows if r.split == split})
            self.reads.append((split, ",".join(lights), len(rows)))
        return np.stack([self._image(r) for r in rows])


# 테스트용 PatchCore 구성. coreset_ratio=1.0이라 뱅크가 모든 패치를 그대로 가진다. 그래서 자기 자신(또는
# 똑같은 이미지)이 뱅크에 남아 있으면 점수가 정확히 0이 되고, 교차 적합의 누수를 점수 0으로 알아챌 수 있다.
CFG = PatchCoreConfig(
    name="toy", backbone="gridmean", img_size=SIZE, coreset_ratio=1.0, sigma=1.0, batch_size=16
)


# 모든 테스트에 자동으로 붙는 fixture. 장부 경로를 임시 폴더로 돌려 실제 reports/test_ledger.jsonl을 건드리지
# 않게 하고, git 커밋 조회를 고정값으로 바꿔 git 상태와 무관하게 기록 내용을 비교할 수 있게 한다. run_m2ad는
# git_commit을 자기 이름공간으로 가져오므로 두 모듈 모두 바꾼다.
@pytest.fixture(autouse=True)
def temporary_ledger(tmp_path, monkeypatch):
    path = tmp_path / "reports" / "test_ledger.jsonl"
    monkeypatch.setattr(paths, "TEST_LEDGER", path)
    monkeypatch.setattr(ledger, "git_commit", lambda root=None: "abc1234")
    monkeypatch.setattr(run_m2ad, "git_commit", lambda root=None: "abc1234")
    return path


# Motor 검사기 하나를 run_inspector로 돌리고 (요약 dict, npz 배열 dict)를 돌려준다. npz를 with로 열어 배열을
# 꺼낸 뒤 닫아, Windows에서 파일이 잠긴 채 남지 않게 한다.
def _run(tmp_path, view="000", cache=None, cfg=CFG, **kwargs):
    cache = cache or FakeCache()
    info = run_m2ad.run_inspector(
        cfg, "Motor", view, make_rows(), cache, GridMean(), "cpu", tmp_path, **kwargs
    )
    with np.load(tmp_path / f"Motor_{view}.npz") as z:
        return info, {k: z[k] for k in z.files}


# 허락(allow_test=True)을 준 전체 실행 한 번의 결과를 모듈 안 여러 테스트가 나눠 쓴다. 조건 25개와 재보정
# 18개를 도는 실행이라 테스트마다 다시 돌리면 느려서 module 범위로 둔다.
@pytest.fixture(scope="module")
def full(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("m2ad-full")
    cache = FakeCache()
    info, arrays = _run(tmp_path, cache=cache, allow_test=True)
    return info, arrays, cache


# 장부 파일(JSON Lines)을 읽어 줄마다 dict로 돌려준다. 파일이 없으면 빈 목록(아직 한 줄도 쓰지 않음).
def ledger_entries(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# run_m2ad.inspector_rows를 쓰지 않고 같은 조건의 행을 직접 골라 시편 이름순으로 정렬한다. 기대값을 실행
# 코드와 독립적으로 세우기 위한 도우미다.
def pick(rows, view, split, light, category="Motor") -> list[M2adRow]:
    """The rows of one (category, view, split, illumination) by specimen name, stated without the run."""
    wanted = (category, view, split, light)
    return sorted(
        (r for r in rows if (r.category, r.view, r.split, r.illumination) == wanted), key=lambda r: r.specimen
    )


# 기대값 계산기: bank_rows로 뱅크를 만들고 shown_rows를 채점한 이미지 점수를 patchcore 함수로 바로 계산한다.
# run_m2ad의 점수가 '이 뱅크로 이 이미지를 채점한 값'과 같은지 비교하는 데 쓴다.
def direct_scores(bank_rows, shown_rows, cfg=CFG) -> np.ndarray:
    """Image scores of `shown_rows` against a bank of `bank_rows`, straight from the patchcore functions."""
    from defect_inspect.patchcore import build_bank, collect_features, score_images

    cache = FakeCache()
    feats, _ = collect_features(GridMean(), cache.images(bank_rows), batch_size=cfg.batch_size, device="cpu")
    bank = build_bank(feats, cfg.coreset_ratio, seed=cfg.seed, device="cpu")
    return score_images(
        GridMean(),
        bank,
        cache.images(shown_rows),
        batch_size=cfg.batch_size,
        device="cpu",
        reweight_k=cfg.reweight_k,
        sigma=cfg.sigma,
    ).image_scores


# 조건 목록과 재보정 키의 순서를 확인한다. S 하나, 합성 교란 15개(conditions 모듈의 등록 순서), 실제 조명
# 02~10 아홉 개로 25개이고, --check는 S와 R:02뿐이다. 재보정 키는 조명별로 n = 8, 30 순서다. 이 목록이 npz의
# conditions·recal_keys 배열과 run.json에 그대로 저장되고 analyze_m2ad가 그 이름으로 행을 찾으므로, 등록된
# 조건(3-B 규칙)이 빠지거나 바뀌는 회귀를 여기서 막는다.
def test_condition_and_recalibration_order():
    conditions = run_m2ad.condition_list()
    assert len(conditions) == 25 and conditions[0] == "S"
    assert conditions[1:16] == [f"P:{name}" for name in condition_names(include_clean=False)]
    assert conditions[16:] == [f"R:{light}" for light in ILLUMINATIONS[1:]]
    assert run_m2ad.condition_list(check=True) == ["S", "R:02"]
    keys = run_m2ad.recal_key_list()
    assert len(keys) == 18 and keys[:4] == ["02:8", "02:30", "03:8", "03:30"] and keys[-1] == "10:30"


# npz의 키·모양·dtype과 요약 dict를 확인한다. 시편은 이름순(정상 4개가 먼저), 점수는 float32, 임계값은 float64
# 스칼라, 보정 점수는 학습 시편 30개분이다. 저장 형식이 분석 코드가 기대하는 것과 어긋나는 회귀를 막는다.
def test_full_run_shapes_and_order(full):
    info, z, _ = full
    n_test = len(NORMAL) + len(ANOMALOUS)
    assert z["specimens"].tolist() == sorted(NORMAL + ANOMALOUS)
    assert z["object_anomaly"].tolist() == [0] * len(NORMAL) + [1] * len(ANOMALOUS)
    assert z["object_anomaly"].dtype == np.int8
    assert z["conditions"].tolist() == run_m2ad.condition_list()
    assert z["scores"].shape == (25, n_test) and z["scores"].dtype == np.float32
    assert z["labels"].shape == (25, n_test) and z["labels"].dtype == np.int8
    assert z["threshold"].shape == () and z["threshold"].dtype == np.float64
    assert z["cal_score"].shape == (N_TRAIN,) and z["cal_score"].dtype == np.float32
    assert z["recal_keys"].tolist() == run_m2ad.recal_key_list()
    assert z["recal_scores"].shape == (18, n_test) and z["recal_scores"].dtype == np.float32
    assert z["recal_thresholds"].shape == (18,) and z["recal_thresholds"].dtype == np.float64
    assert z["recal_labels"].shape == (18, n_test) and z["recal_labels"].dtype == np.int8
    assert np.isfinite(z["scores"]).all() and np.isfinite(z["recal_scores"]).all()
    assert (info["category"], info["view"]) == ("Motor", "000")
    assert (info["train_specimens"], info["test_specimens"], info["test_anomalous_specimens"]) == (30, 9, 5)
    assert info["grid"] == [2, 2] and info["dim"] == 3
    assert set(info["recal"]) == set(run_m2ad.recal_key_list())


# 임계값이 교차 적합 점수에서 등각(conformal) 규칙으로 나왔는지 확인한다. n = 30, α = 5%면 l = ceil(31 × 0.95)
# = 30이라 가장 큰 보정 점수가 임계값이 된다. 겹은 m2ad.specimen_folds(시편 이름 SHA-256 순, 5겹)와 같아야
# 한다.
# 보정 점수가 모두 0보다 크다는 것은 어느 이미지도 자기를 가진 뱅크로 채점되지 않았다는 뜻이다(뱅크 전체를
# 유지하므로 새면 정확히 0). 겹 뱅크 행 수(시편 24개 × 패치 4)로 한 겹씩 빠졌는지도 본다.
def test_threshold_is_cross_fitted_over_specimens(full):
    info, z, _ = full
    thr = conformal_threshold(z["cal_score"], 0.05)
    assert float(z["threshold"]) == thr.value
    assert info["threshold"] == {"value": thr.value, "rank": 30, "n": 30, "guaranteed": True}
    train = [f"{2 * i:03d}" for i in range(N_TRAIN)]
    assert info["folds"] == specimen_folds(train)
    # With the whole bank kept, an image scored by a bank that holds it would score exactly 0.
    assert z["cal_score"].min() > 0
    assert info["bank_rows"] == {"full": 30 * PATCHES} | {f"minus_fold_{f}": 24 * PATCHES for f in range(5)}


# 라벨이 '그 조건에서 보여 준 이미지'의 라벨인지 확인한다. 조명 05에서만 안 보이는 결함은 R:05와 05 재보정
# 키에서만 -1이고, 합성 교란(P)은 기준 조명 이미지를 변형한 것이므로 기준 라벨을 그대로 쓴다. 조명마다 보이는
# 결함이 다르다는 M2AD 라벨 규칙(3-B)이 조건별로 적용되는지 지키는 테스트다.
def test_labels_come_from_the_image_shown(full):
    _, z, _ = full
    names = z["specimens"].tolist()
    conditions = z["conditions"].tolist()
    base = np.array([0] * len(NORMAL) + [1, -1, 1, 1, 1], dtype=np.int8)
    assert np.array_equal(z["labels"][0], base)
    for k, condition in enumerate(conditions):
        expected = base.copy()
        if condition == "R:05":
            expected[names.index(HIDDEN_UNDER_05)] = -1
        assert np.array_equal(z["labels"][k], expected), condition
    # Synthetic conditions show the illumination-01 image, so they keep its labels.
    assert (z["labels"][1:16] == base).all()
    for j, key in enumerate(z["recal_keys"].tolist()):
        expected = base.copy()
        if key.startswith("05:"):
            expected[names.index(HIDDEN_UNDER_05)] = -1
        assert np.array_equal(z["recal_labels"][j], expected), key


# 가짜 데이터에서 점수가 예상대로 움직이는지 본다: 보이는 결함은 임계값을 넘고 정상보다 5배 넘게 높으며, 안
# 보이는 결함은 정상처럼 낮다. 16px에서 이동 1(256 기준)은 반올림으로 0px라 점수가 그대로이고, 밝기 교란은
# 점수를 바꾼다. 조명 03은 기준과 같아 점수가 같고, 더 어두운 조명일수록 정상 점수 평균이 커진다.
def test_scores_separate_defects_and_conditions(full):
    _, z, _ = full
    conditions = z["conditions"].tolist()
    s = z["scores"][0]
    threshold = float(z["threshold"])
    visible = z["labels"][0] == 1
    assert s[visible].min() > threshold > 0
    assert s[visible].min() > 5 * s[z["labels"][0] == 0].max()
    # The never-visible defect has no bright cell: it scores like a normal image.
    assert s[z["specimens"].tolist().index(NEVER_VISIBLE)] < s[visible].min() / 5
    # A translation of less than half a pixel at this size is the identity.
    assert np.array_equal(z["scores"][conditions.index("P:shift-1")], s)
    assert not np.allclose(z["scores"][conditions.index("P:brightness-3")], s)
    # The fake illumination 03 equals the reference, the others are darker and darker.
    assert np.array_equal(z["scores"][conditions.index("R:03")], s)
    normal = z["labels"][0] == 0
    darker = [z["scores"][conditions.index(f"R:{light}")][normal].mean() for light in ("02", "06", "10")]
    assert threshold < darker[0] < darker[1] < darker[2]


# 합성 교란 조건(P)이 '기준 조명 테스트 이미지에 교란을 준 것'을 기준 뱅크로 채점한 값인지, 교란 세 개로 직접
# 계산해 비교한다(3-B 규칙: P는 I01에 합성 교란 15조건).
def test_synthetic_conditions_are_applied_to_the_reference_test_images(tmp_path, full, monkeypatch):
    _, z, _ = full
    from defect_inspect.patchcore import build_bank, collect_features, score_images

    rows = make_rows()
    cache = FakeCache()
    train = run_m2ad.inspector_rows(rows, "Motor", "000", "train", "01")
    test = run_m2ad.inspector_rows(rows, "Motor", "000", "test", "01")
    feats, _ = collect_features(GridMean(), cache.images(train), batch_size=16, device="cpu")
    bank = build_bank(feats, 1.0, seed=0, device="cpu")
    for name in ("brightness-3", "gamma-2", "jpeg-3"):
        images = apply_condition(cache.images(test), name)
        direct = score_images(GridMean(), bank, images, batch_size=16, device="cpu", sigma=1.0).image_scores
        k = z["conditions"].tolist().index(f"P:{name}")
        np.testing.assert_allclose(z["scores"][k], direct, rtol=1e-5)


# 재보정을 확인한다. 새 조명 L, n장마다: 보정 이미지 수와 뱅크 행 수가 30 + n장분이고, 겹 뱅크는 기준 이미지와
# 새 이미지를 같은 겹 규칙으로 함께 뺀 크기다. 새 조명 정상은 뱅크에 이웃이 생겨 점수가 재보정 전 최솟값의
# 절반 아래로 내려가고, 결함은 여전히 새 임계값을 넘는다.
# 마지막으로 조명 10에서 재보정 전에는 정상 전부가 불량이었다가 n = 30이면 75% 이상이 임계값 아래로
# 돌아오는지, 새 조명의 학습 이미지를 조명마다 한 번(30장)만 읽는지(n마다 다시 읽지 않음) 본다.
def test_recalibration_bank_contains_the_new_illumination(full):
    info, z, cache = full
    conditions = z["conditions"].tolist()
    keys = z["recal_keys"].tolist()
    normal = z["labels"][0] == 0
    train = [f"{2 * i:03d}" for i in range(N_TRAIN)]
    folds = specimen_folds(train)
    for light in ("04", "07", "10"):
        before = z["scores"][conditions.index(f"R:{light}")]
        for n in (8, 30):
            key = f"{light}:{n}"
            j = keys.index(key)
            rec = info["recal"][key]
            assert rec["images"] == 30 + n and rec["threshold"]["n"] == 30 + n
            assert rec["bank_rows"]["full"] == (30 + n) * PATCHES
            # 겹 f에서 빠지는 이미지 수 = 그 겹의 기준 학습 시편 수 + 그 겹에 속한 재보정 시편 수(새 조명
            # 이미지도 시편의 겹을 따른다).
            held = {
                f: sum(folds[s] == f for s in train) + sum(folds[s] == f for s in recal_specimens(train, n))
                for f in range(5)
            }
            assert {f: rec["bank_rows"][f"minus_fold_{f}"] for f in range(5)} == {
                f: (30 + n - held[f]) * PATCHES for f in range(5)
            }
            assert float(z["recal_thresholds"][j]) == rec["threshold"]["value"]
            # Normal images under the new illumination now have neighbours in the bank.
            assert z["recal_scores"][j][normal].max() < before[normal].min() / 2
            # Defects are still far from it.
            visible = z["recal_labels"][j] == 1
            assert z["recal_scores"][j][visible].min() > float(z["recal_thresholds"][j])
    # n = 30 brings every normal back under the threshold; without recalibration all were flagged.
    j = keys.index("10:30")
    assert (z["scores"][conditions.index("R:10")][normal] > float(z["threshold"])).all()
    assert (z["recal_scores"][j][normal] <= float(z["recal_thresholds"][j])).mean() >= 0.75
    # The train images of every new illumination were read once, 30 at a time.
    train_reads = [r for r in cache.reads if r[0] == "train"]
    assert train_reads == [("train", "01", 30)] + [("train", light, 30) for light in ILLUMINATIONS[1:]]


# 재보정 점수가 '기준 학습 30장 + 새 조명 학습 n장' 뱅크로 '새 조명의 테스트 이미지'를 채점한 값과 같은지
# 확인한다. 이어서 기준 조명 테스트 이미지나 기준 뱅크만으로는 같은 값이 나오지 않음을 보여, 엉뚱한 이미지나
# 뱅크로 채점하는 버그가 통과하지 못하게 한다.
def test_recalibrated_scores_are_those_of_the_new_illumination_test_images(full):
    _, z, _ = full
    rows = make_rows()
    keys = z["recal_keys"].tolist()
    ref_train, ref_test = pick(rows, "000", "train", "01"), pick(rows, "000", "test", "01")
    train = [r.specimen for r in ref_train]
    # 05 hides one defect that the reference illumination shows; 07 only changes the brightness.
    for light in ("05", "07"):
        new_train, new_test = pick(rows, "000", "train", light), pick(rows, "000", "test", light)
        for n in (8, 30):
            chosen = set(recal_specimens(train, n))
            bank_rows = ref_train + [r for r in new_train if r.specimen in chosen]
            assert len(bank_rows) == 30 + n
            got = z["recal_scores"][keys.index(f"{light}:{n}")]
            np.testing.assert_allclose(got, direct_scores(bank_rows, new_test), rtol=1e-5)
            # Neither the reference test images nor the reference bank give these numbers.
            # 가짜 이미지는 조명마다 밝기만 일정하게 빼므로, 07에서 n = 30이면 기준 테스트 이미지도 같은
            # 점수를 받을 수 있다. 그래서 이 부정 확인은 구별이 되는 조합(n = 8 또는 결함이 하나 가려지는
            # 05)에서만 한다.
            if n == 8 or light == "05":
                assert not np.allclose(got, direct_scores(bank_rows, ref_test), rtol=1e-3)
            assert not np.allclose(got, direct_scores(ref_train, new_test), rtol=1e-3)


# 검사기가 자기 시점(여기서는 120)의 이미지만 쓰는지 확인한다. 조명을 두 개로 줄여(monkeypatch) 빠르게 돌린 뒤
# 보정 점수, 조건 S·R:02 점수, 재보정 점수를 모두 시점 120의 행으로 직접 계산한 값과 비교한다. 마지막 두 줄은
# 같은 시편의 시점 000 이미지로 계산하면 값이 달라짐을 보여, 시점을 섞는 버그를 잡는다.
def test_an_inspector_uses_the_images_of_its_own_view(tmp_path, monkeypatch):
    monkeypatch.setattr(run_m2ad, "ILLUMINATIONS", ("01", "02"))
    info, z = _run(tmp_path, view="120", allow_test=True)
    assert info["view"] == "120" and z["conditions"].tolist()[-1] == "R:02"
    assert z["recal_keys"].tolist() == ["02:8", "02:30"]
    rows = make_rows()
    ref_train, ref_test = pick(rows, "120", "train", "01"), pick(rows, "120", "test", "01")
    new_train, new_test = pick(rows, "120", "train", "02"), pick(rows, "120", "test", "02")
    train = [r.specimen for r in ref_train]

    # Threshold: every train image is scored by the bank of the other folds, all from view 120.
    # 교차 적합 기대값: 겹마다 나머지 겹의 이미지로 뱅크를 만들어 그 겹의 이미지를 채점한다.
    fold_of = specimen_folds(train)
    expected = np.empty(N_TRAIN, dtype=np.float32)
    for fold in range(5):
        held = [i for i, r in enumerate(ref_train) if fold_of[r.specimen] == fold]
        rest = [r for r in ref_train if fold_of[r.specimen] != fold]
        expected[held] = direct_scores(rest, [ref_train[i] for i in held])
    np.testing.assert_allclose(z["cal_score"], expected, rtol=1e-5)
    assert float(z["threshold"]) == conformal_threshold(z["cal_score"], 0.05).value

    np.testing.assert_allclose(z["scores"][0], direct_scores(ref_train, ref_test), rtol=1e-5)
    np.testing.assert_allclose(z["scores"][-1], direct_scores(ref_train, new_test), rtol=1e-5)
    assert z["labels"][-1].tolist() == [r.label for r in new_test]
    for j, n in enumerate((8, 30)):
        chosen = set(recal_specimens(train, n))
        bank_rows = ref_train + [r for r in new_train if r.specimen in chosen]
        np.testing.assert_allclose(z["recal_scores"][j], direct_scores(bank_rows, new_test), rtol=1e-5)
    # The same specimens seen from view 000 give other numbers, as test images and as bank images.
    other_test, other_train = pick(rows, "000", "test", "02"), pick(rows, "000", "train", "02")
    assert not np.allclose(z["scores"][-1], direct_scores(ref_train, other_test), rtol=1e-3)
    assert not np.allclose(z["recal_scores"][1], direct_scores(ref_train + other_train, new_test), rtol=1e-3)


# 구성(PatchCoreConfig)의 값이 모든 뱅크 생성·채점·특징 추출 호출에 전달되는지 확인한다. patchcore 함수 세
# 개를 감싸 호출 인자를 기록하는 스파이로 바꾸고(run_inspector가 함수 안에서 patchcore를 import하므로 모듈
# 속성을 바꾸면 잡힌다), 기본값과 다른 값들을 넣어 기본값이 몰래 쓰이면 드러나게 한다.
# 호출 횟수도 센다: 뱅크 18개(기준 1 + 재보정 2, 각각 전체 1 + 겹 5), 채점 22 + 2 × 6번, 특징 추출 2번(기준
# 학습, 새 조명 학습).
def test_config_values_reach_every_bank_and_every_scoring_call(tmp_path, monkeypatch):
    from dataclasses import replace

    from defect_inspect import patchcore

    # None of these is the default of PatchCoreConfig or of the patchcore functions.
    cfg = PatchCoreConfig(
        name="toy",
        backbone="gridmean",
        img_size=SIZE,
        coreset_ratio=0.25,
        reweight_k=3,
        sigma=2.0,
        batch_size=7,
        seed=5,
    )
    built, scored, collected = [], [], []
    real_build, real_score, real_collect = (
        patchcore.build_bank,
        patchcore.score_images,
        patchcore.collect_features,
    )

    def build(features, ratio=0.1, **kwargs):
        built.append((ratio, kwargs))
        return real_build(features, ratio, **kwargs)

    def score(extractor, bank, images, **kwargs):
        scored.append(kwargs)
        return real_score(extractor, bank, images, **kwargs)

    def collect(extractor, images, **kwargs):
        collected.append(kwargs)
        return real_collect(extractor, images, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(run_m2ad, "ILLUMINATIONS", ("01", "02"))
        patch.setattr(patchcore, "build_bank", build)
        patch.setattr(patchcore, "score_images", score)
        patch.setattr(patchcore, "collect_features", collect)
        _, z = _run(tmp_path, cfg=cfg, allow_test=True)
    # Reference bank and two recalibrated ones, each with its five fold banks.
    assert len(built) == 18 and all(call == (0.25, {"seed": 5, "device": "cpu"}) for call in built)
    # 5 folds + S + 15 P + R:02, then per recalibration 5 folds + the new test images.
    assert len(scored) == 22 + 2 * 6
    for kwargs in scored:
        picked = (kwargs["batch_size"], kwargs["reweight_k"], kwargs["sigma"], kwargs["device"])
        assert picked == (7, 3, 2.0, "cpu")
    assert len(collected) == 2 and all(k == {"batch_size": 7, "device": "cpu"} for k in collected)

    # The numbers are those of a direct call with the config's values, and the values matter.
    # 기록된 인자뿐 아니라 숫자도 확인한다: 같은 구성으로 직접 계산한 점수와 같고,
    # seed·reweight_k·coreset_ratio 중 하나만 바꿔도 달라진다(값이 실제로 결과에 영향을 준다는 뜻).
    rows = make_rows()
    ref_train, ref_test = pick(rows, "000", "train", "01"), pick(rows, "000", "test", "01")
    np.testing.assert_allclose(z["scores"][0], direct_scores(ref_train, ref_test, cfg), rtol=1e-5)
    for other in (replace(cfg, seed=0), replace(cfg, reweight_k=9), replace(cfg, coreset_ratio=0.1)):
        assert not np.allclose(z["scores"][0], direct_scores(ref_train, ref_test, other), rtol=1e-3)


# 재보정 교차 적합에서 한 시편의 기준 이미지와 새 조명 이미지가 같은 겹으로 함께 빠지는지 확인한다. 조명 03의
# 가짜 이미지는 기준과 똑같으므로, 둘이 다른 겹에 들어가면 서로를 뱅크에서 찾아 점수 60개가 모두 0이 되고
# 임계값도 0이 된다. n = 60이면 순위 l = ceil(61 × 0.95) = 58이다.
def test_reference_and_new_images_of_a_specimen_are_held_out_together(full):
    info, z, _ = full
    keys = z["recal_keys"].tolist()
    # Under the fake illumination 03 the new image of a specimen equals its reference image. If one of
    # them stayed in the bank while the other was scored, all 60 cross-fitted scores (and so the
    # threshold) would be exactly 0, because the whole bank is kept.
    assert info["recal"]["03:30"]["threshold"]["rank"] == 58
    assert float(z["recal_thresholds"][keys.index("03:30")]) > 0


# crossfit을 감싸는 스파이로 세 번의 호출(기준 30장, 재보정 n = 8과 30)의 입력을 붙잡아, 이미지·특징·겹 배열의
# i번째가 같은 시편을 가리키는지(기준 30장 뒤에 재보정 시편 순서) 확인한다. 순서가 어긋나면 엉뚱한 겹으로 빠져
# 누수나 잘못된 임계값이 생긴다.
def test_recalibration_crossfit_gets_aligned_inputs_and_leaks_nothing(tmp_path, monkeypatch):
    from defect_inspect.patchcore import collect_features

    monkeypatch.setattr(run_m2ad, "ILLUMINATIONS", ("01", SAME_AS_REFERENCE))
    calls = []
    real_crossfit = run_m2ad.crossfit

    def spying(build, score, feats, images, folds):
        result = real_crossfit(build, score, feats, images, folds)
        calls.append((feats, images, folds, result[1]))
        return result

    monkeypatch.setattr(run_m2ad, "crossfit", spying)
    _, z = _run(tmp_path, allow_test=True)

    rows = make_rows()
    cache = FakeCache()
    ref_train = pick(rows, "000", "train", "01")
    new_train = {r.specimen: r for r in pick(rows, "000", "train", SAME_AS_REFERENCE)}
    train = [r.specimen for r in ref_train]
    fold_of = specimen_folds(train)
    assert [len(images) for _, images, _, _ in calls] == [30, 38, 60]
    for (feats, images, folds, oof), n in zip(calls, (0, 8, 30), strict=True):
        added = recal_specimens(train, n)
        # Image i, feature i and fold i belong to one specimen: the 30 reference images, then the new ones.
        assert np.array_equal(images, cache.images(ref_train + [new_train[name] for name in added]))
        direct, _ = collect_features(GridMean(), images, batch_size=16, device="cpu")
        assert torch.equal(feats.reshape(-1, feats.shape[-1]), direct)
        assert folds.tolist() == [fold_of[name] for name in train + added]
        # The new image of a specimen equals its reference image under this fake illumination, and the
        # whole bank is kept: a score of 0 (below 1e-3 after fp16 rounding, see the crossfit test) would
        # mean that one of the two stayed in the bank while the other was scored. Every one of the
        # cross-fitted scores must be clear of that.
        assert oof.shape == (30 + n,) and oof.min() > 5e-3
    assert np.array_equal(calls[0][3], z["cal_score"])


# crossfit 함수만 따로 확인한다. 10장 가운데 뒤 5장이 앞 5장의 복사본이다. 복사본을 같은 겹에 두면 어느
# 이미지도 자기(또는 복사본)를 가진 뱅크로 채점되지 않아 점수가 0보다 크고 복사본끼리 점수가 같다. 다른 겹에
# 두면 복사본이 뱅크에 남아 점수가 거의 0이 된다(누수를 실제로 감지할 수 있다는 대조군). 어떤 겹에도 들지 않는
# 이미지(겹 번호 7)가 있으면 RuntimeError로 멈춰야 한다.
def test_crossfit_holds_out_by_fold():
    from defect_inspect.patchcore import build_bank, collect_features, score_images

    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, (10, SIZE, SIZE, 3), dtype=np.uint8)
    images[5:] = images[:5]  # image i + 5 is a copy of image i
    feats, _ = collect_features(GridMean(), images, batch_size=4, device="cpu")
    feats = feats.view(10, PATCHES, 3)
    calls = []

    def build(features):
        calls.append(features.shape[0])
        return build_bank(features, 1.0, seed=0, device="cpu")

    def score(bank, batch):
        extractor = GridMean()
        return score_images(extractor, bank, batch, batch_size=4, device="cpu", sigma=1.0).image_scores

    together = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    bank, oof, thr, bank_rows = run_m2ad.crossfit(build, score, feats, images, together)
    assert calls == [40, 32, 32, 32, 32, 32] and bank.shape == (40, 3)
    assert bank_rows == {"full": 40} | {f"minus_fold_{f}": 32 for f in range(5)}
    assert oof.shape == (10,) and thr.value == conformal_threshold(oof, 0.05).value
    assert oof.min() > 0.01  # no image was scored by a bank that holds it or its copy
    assert np.array_equal(oof[:5], oof[5:])  # copies in the same fold get the same score
    # With the copies in different folds every image finds itself in the bank.
    apart = np.array([0, 1, 2, 3, 4, 1, 2, 3, 4, 0])
    _, leaked, _, _ = run_m2ad.crossfit(build, score, feats, images, apart)
    assert np.abs(leaked).max() < 1e-3
    with pytest.raises(RuntimeError):
        run_m2ad.crossfit(build, score, feats, images, np.array([0, 1, 2, 3, 7, 0, 1, 2, 3, 7]))


# 코어셋 비율 0.25가 실제로 뱅크 크기를 4분의 1로 줄이는지(전체 뱅크와 겹 뱅크 모두) --check 실행으로
# 확인한다.
def test_coreset_ratio_of_the_config_is_used(tmp_path):
    cfg = PatchCoreConfig(
        name="toy", backbone="gridmean", img_size=SIZE, coreset_ratio=0.25, sigma=1.0, batch_size=16
    )
    info, _ = _run(tmp_path, cfg=cfg, check=True)
    assert info["bank_rows"]["full"] == 30 * PATCHES // 4
    assert info["bank_rows"]["minus_fold_0"] == 24 * PATCHES // 4


# 같은 입력으로 다시 돌리면 npz의 모든 배열이 비트 단위로 같아야 한다. 분할·표본 추출이 결정적이어야 한다는
# 규칙(docs/experiments.md 3-B: 겹과 재보정 시편은 시편 이름의 SHA-256 순)을 이 실행 경로에서 지킨다.
def test_run_is_deterministic(tmp_path, full):
    _, z, _ = full
    _, again = _run(tmp_path, allow_test=True)
    assert z.keys() == again.keys()
    for key in z:
        assert np.array_equal(z[key], again[key]), key


# 등록된 코드 점검(--check)은 조건 S와 R:02만, 재보정 없이 돌린다(3-B: M2AD에는 검증 분할이 없어 Motor 000
# 검사기 하나로 점검). 같은 검사기의 전체 실행과 임계값·점수가 같아야 하고, 테스트 이미지는 두 조건분만 읽어야
# 한다.
def test_check_runs_one_inspector_without_recalibration(tmp_path, full):
    _, z, _ = full
    cache = FakeCache()
    info, c = _run(tmp_path, cache=cache, check=True)
    assert c["conditions"].tolist() == ["S", "R:02"]
    assert c["scores"].shape == (2, 9) and c["labels"].shape == (2, 9)
    assert c["recal_keys"].shape == (0,) and c["recal_thresholds"].shape == (0,)
    assert c["recal_scores"].shape == (0, 9) and c["recal_labels"].shape == (0, 9)
    assert info["recal"] == {}
    # Same inspector, same numbers as in the full run.
    assert float(c["threshold"]) == float(z["threshold"])
    assert np.array_equal(c["scores"][0], z["scores"][0])
    assert np.array_equal(c["scores"][1], z["scores"][z["conditions"].tolist().index("R:02")])
    # Only the two conditions of the check were read from the test split, and only reference train images.
    assert cache.reads == [("train", "01", 30), ("test", "01", 9), ("test", "02", 9)]


# 봉인: allow_test가 정확히 True가 아니면("yes" 같은 참 값도 안 됨) SealedTestError로 멈추고, --check 예외는
# Motor 000 검사기 하나에만 적용된다. 멈출 때는 이미지를 한 장도 읽지 않고 npz도 남기지 않아야 한다.
def test_test_images_need_permission(tmp_path):
    cache = FakeCache()
    with pytest.raises(SealedTestError):
        _run(tmp_path, cache=cache)
    with pytest.raises(SealedTestError):
        _run(tmp_path, cache=cache, allow_test="yes")
    # The check exempts one inspector only.
    with pytest.raises(SealedTestError):
        _run(tmp_path, view="120", cache=cache, check=True)
    assert cache.reads == [] and not list(tmp_path.glob("*.npz"))
    _run(tmp_path, view="120", cache=cache, check=True, allow_test=True)


# 조명마다 시편 구성이 같아야 한다는 검사. 조명 02의 테스트 시편 하나를 빼면 ValueError가 나고, 행이 없는
# 범주(Bird)는 '학습 이미지 없음'으로 멈춘다. 시편이 어긋난 채 진행하면 조건별 점수 행이 서로 다른 시편을
# 가리키게 된다.
def test_specimens_must_match_across_illuminations(tmp_path):
    rows = [
        r for r in make_rows() if not (r.split == "test" and r.illumination == "02" and r.specimen == "001")
    ]
    with pytest.raises(ValueError, match="illumination 02"):
        run_m2ad.run_inspector(
            CFG, "Motor", "000", rows, FakeCache(), GridMean(), "cpu", tmp_path, allow_test=True
        )
    with pytest.raises(ValueError, match="no train images"):
        run_m2ad.run_inspector(
            CFG, "Bird", "000", rows, FakeCache(), GridMean(), "cpu", tmp_path, allow_test=True
        )


# ----------------------------------------------------------------------------------------------------- CLI


# CLI(main) 테스트용 준비. 메타데이터 읽기, 캐시, 구성, 백본 생성을 모두 가짜로 바꿔 데이터 파일과 GPU 없이
# main을 돌린다. 범주는 Motor와 Bird 두 개를 넣어 범주 선택도 확인한다.
@pytest.fixture
def cli(tmp_path, monkeypatch, temporary_ledger):
    cache = FakeCache()
    rows = make_rows(("Motor", "Bird"))
    monkeypatch.setattr(run_m2ad, "read_meta", lambda path: rows)
    monkeypatch.setattr(run_m2ad, "M2adCache", lambda out_dir, size: cache)
    monkeypatch.setattr(run_m2ad, "get_config", lambda name: CFG)
    monkeypatch.setattr(run_m2ad, "_make_extractor", lambda cfg, device: GridMean())
    return cache, ["--method", "p0", "--device", "cpu", "--out", str(tmp_path / "m2ad-p0")]


# --check는 허락 없이 돌고 장부에 줄을 쓰지 않으며, 결과를 '<out>-check' 폴더에 따로 둔다(본 실행 폴더를
# 만들지 않음). run.json의 메타 정보와 표준 출력의 요약 줄을 확인하고, 끝난 실행은 --overwrite 없이 다시 돌지
# 않는지도 본다.
def test_cli_check_needs_no_permission_and_writes_no_ledger_line(tmp_path, cli, temporary_ledger, capsys):
    cache, argv = cli
    run_m2ad.main([*argv, "--check"])
    out_dir = tmp_path / "m2ad-p0-check"
    assert sorted(p.name for p in out_dir.iterdir()) == ["Motor_000.npz", "run.json"]
    assert not (tmp_path / "m2ad-p0").exists()
    assert not temporary_ledger.exists()
    meta = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["method"] == "p0" and meta["check"] is True and meta["device"] == "cpu"
    assert meta["commit"] == "abc1234" and meta["conditions"] == ["S", "R:02"] and meta["recal_keys"] == []
    assert [(i["category"], i["view"]) for i in meta["inspectors"]] == [("Motor", "000")]
    assert meta["inspectors"][0]["seconds"] >= 0 and meta["inspectors"][0]["bank_rows"]["full"] == 120
    assert json.loads(capsys.readouterr().out.splitlines()[0])["view"] == "000"
    # A finished run is not redone without --overwrite.
    with pytest.raises(SystemExit):
        run_m2ad.main([*argv, "--check"])
    run_m2ad.main([*argv, "--check", "--overwrite"])


# --check에 --allow-test와 --stage를 같이 줘도 점검으로만 돌아야 한다: run_inspector에 allow_test=False,
# check=True로 넘어가고 장부 줄도 본 실행 폴더도 없다. 플래그 조합 실수로 점검이 봉인 테스트 전체를 읽는 일을
# 막는다.
def test_cli_check_stays_the_check_when_the_full_run_flags_are_given(
    tmp_path, cli, temporary_ledger, monkeypatch
):
    cache, argv = cli
    calls = []
    real_inspector = run_m2ad.run_inspector

    def watching(cfg, category, view, *args, **kwargs):
        calls.append((category, view, kwargs["allow_test"], kwargs["check"]))
        return real_inspector(cfg, category, view, *args, **kwargs)

    monkeypatch.setattr(run_m2ad, "run_inspector", watching)
    run_m2ad.main([*argv, "--check", "--allow-test", "--stage", "3-m2ad"])
    # One inspector, without the permission of the full run, no ledger line, only the check's reads.
    assert calls == [("Motor", "000", False, True)]
    assert not temporary_ledger.exists() and not (tmp_path / "m2ad-p0").exists()
    assert cache.reads == [("train", "01", 30), ("test", "01", 9), ("test", "02", 9)]


# 본 실행은 --allow-test와 --stage가 둘 다 있어야 한다. 하나라도 빠지면 argparse 오류(SystemExit)로 멈추고,
# 이미지 읽기·장부·출력 폴더가 하나도 생기지 않아야 한다.
def test_cli_full_run_needs_the_flags(tmp_path, cli, temporary_ledger):
    cache, argv = cli
    with pytest.raises(SystemExit):
        run_m2ad.main(argv)
    with pytest.raises(SystemExit):
        run_m2ad.main([*argv, "--allow-test"])
    with pytest.raises(SystemExit):
        run_m2ad.main([*argv, "--stage", "3-m2ad"])
    assert cache.reads == [] and not temporary_ledger.exists()
    assert not (tmp_path / "m2ad-p0").exists()


# 본 실행이 장부에 줄을 정확히 한 번, 테스트 이미지를 처음 읽기 전에 쓰는지 확인한다(on_test_read 훅이 읽을
# 때마다 장부 줄 수를 기록하고 그 값이 모두 1이어야 한다). 두 범주 × 세 시점의 npz가 모두 생기고, 범주가 다른
# 검사기는 점수를 공유하지 않는지도 본다.
def test_cli_full_run_records_the_ledger_line_first(tmp_path, cli, temporary_ledger, monkeypatch):
    cache, argv = cli
    # Three illuminations keep the six inspectors quick; the full condition list is tested above.
    monkeypatch.setattr(run_m2ad, "ILLUMINATIONS", ("01", "02", "03"))
    seen = []
    cache.on_test_read = lambda: seen.append(len(ledger_entries(temporary_ledger)))
    run_m2ad.main([*argv, "--allow-test", "--stage", "3-m2ad", "--note", "full"])
    entries = ledger_entries(temporary_ledger)
    assert len(entries) == 1
    assert {k: v for k, v in entries[0].items() if k != "time"} == {
        "commit": "abc1234",
        "stage": "3-m2ad",
        "config": "m2ad-p0",
        "note": "full",
    }
    assert seen and set(seen) == {1}  # the line was there before every read of test images
    out_dir = tmp_path / "m2ad-p0"
    inspectors = [f"{category}_{view}" for category in ("Motor", "Bird") for view in VIEWS]
    assert sorted(p.name for p in out_dir.iterdir()) == sorted(
        [f"{i}.npz" for i in inspectors] + ["run.json"]
    )
    meta = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert [f"{i['category']}_{i['view']}" for i in meta["inspectors"]] == inspectors
    assert meta["check"] is False and meta["config"]["name"] == "toy" and meta["alpha"] == 0.05
    assert meta["conditions"] == ["S", *(f"P:{n}" for n in condition_names(False)), "R:02", "R:03"]
    assert meta["recal_keys"] == ["02:8", "02:30", "03:8", "03:30"]
    for name in inspectors:
        with np.load(out_dir / f"{name}.npz") as z:
            assert z["conditions"].tolist() == meta["conditions"]
            assert z["recal_keys"].tolist() == meta["recal_keys"]
            assert z["scores"].shape == (18, 9) and z["recal_scores"].shape == (4, 9)
    # The two categories are different objects: their inspectors do not share scores.
    with np.load(out_dir / "Motor_120.npz") as a, np.load(out_dir / "Bird_120.npz") as b:
        assert not np.array_equal(a["scores"], b["scores"])


# 캐시가 없으면 실제 M2adCache가 만드는 방법(m2ad --size 256)을 알려 주는 FileNotFoundError로 멈추고, 장부
# 줄과 출력 폴더를 남기지 않아야 한다. main이 테스트 이미지와 무관하게 실패할 수 있는 일을 장부 기록보다 먼저
# 하도록 짠 순서를 지킨다(테스트를 읽지도 않았는데 읽은 것으로 기록되는 일을 막음).
def test_cli_stops_before_the_ledger_when_the_cache_is_missing(tmp_path, monkeypatch, temporary_ledger):
    rows = make_rows()
    monkeypatch.setattr(run_m2ad, "read_meta", lambda path: rows)
    argv = ["--method", "p0", "--device", "cpu", "--out", str(tmp_path / "out"), "--cache", str(tmp_path)]
    with pytest.raises(FileNotFoundError, match="defect_inspect.m2ad --size 256"):
        run_m2ad.main([*argv, "--allow-test", "--stage", "3-m2ad"])
    assert not temporary_ledger.exists() and not (tmp_path / "out").exists()


# --categories로 한 범주만 돌리기, 끝난 실행 덮어쓰기 거부, 중간에 죽은 재실행 처리를 확인한다. --overwrite
# 재실행이 도중에 죽으면 run.json이 지워진 채 남아 끝난 실행처럼 보이지 않아야 하고, 그 재실행도 테스트를 읽을
# 수 있었으므로 장부 줄은 2개가 된다. 없는 범주(Car)는 argparse 선택지에서 거절된다.
def test_cli_one_category_and_overwrite(tmp_path, cli, temporary_ledger, monkeypatch):
    cache, argv = cli
    monkeypatch.setattr(run_m2ad, "ILLUMINATIONS", ("01", "02"))
    full = [*argv, "--allow-test", "--stage", "3-m2ad", "--categories", "Bird"]
    run_m2ad.main(full)
    out_dir = tmp_path / "m2ad-p0"
    meta = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert [(i["category"], i["view"]) for i in meta["inspectors"]] == [("Bird", view) for view in VIEWS]
    assert not list(out_dir.glob("Motor_*.npz"))
    with pytest.raises(SystemExit):
        run_m2ad.main(full)
    assert len(ledger_entries(temporary_ledger)) == 1

    # A rerun that dies half-way leaves no run.json behind, so it cannot pass for the finished run.
    def dying(*args, **kwargs):
        raise RuntimeError("stopped")

    monkeypatch.setattr(run_m2ad, "run_inspector", dying)
    with pytest.raises(RuntimeError, match="stopped"):
        run_m2ad.main([*full, "--overwrite"])
    assert not (out_dir / "run.json").exists()
    assert len(ledger_entries(temporary_ledger)) == 2  # the rerun was allowed to read test images
    with pytest.raises(SystemExit):
        run_m2ad.main([*argv, "--allow-test", "--stage", "3-m2ad", "--categories", "Car"])


# 장부 잠금을 얻지 못해 기록이 실패하면(TimeoutError) 재실행은 시작하지 않아야 하고, 이전에 끝난 run.json과
# npz는 그대로 남아야 한다. main이 run.json을 지우는 일을 장부 기록 '뒤'에 하도록 한 이유(기록 못 한 재실행이
# 끝난 결과를 망가뜨리지 않게)를 지키는 테스트다.
def test_cli_overwrite_keeps_the_finished_run_when_the_ledger_cannot_be_written(
    tmp_path, cli, temporary_ledger, monkeypatch
):
    cache, argv = cli
    out_dir = tmp_path / "m2ad-p0"
    out_dir.mkdir()
    (out_dir / "run.json").write_text('{"finished": true}\n', encoding="utf-8")
    (out_dir / "Motor_000.npz").write_bytes(b"old scores")
    started = []

    def locked(*args, **kwargs):
        raise TimeoutError("could not take the ledger lock")

    monkeypatch.setattr(run_m2ad, "record_test_access", locked)
    monkeypatch.setattr(run_m2ad, "run_inspector", lambda *args, **kwargs: started.append(args[1:3]))
    with pytest.raises(TimeoutError, match="ledger lock"):
        run_m2ad.main([*argv, "--allow-test", "--stage", "3-m2ad", "--overwrite"])
    # The rerun never started: no inspector, no read, and the finished run can still be analysed.
    assert started == [] and cache.reads == [] and not temporary_ledger.exists()
    assert (out_dir / "run.json").read_text(encoding="utf-8") == '{"finished": true}\n'
    assert (out_dir / "Motor_000.npz").read_bytes() == b"old scores"
