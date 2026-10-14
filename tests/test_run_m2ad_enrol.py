# [파일 역할] 7단계(조명 변화 대응) 실행 스크립트 `defect_inspect.run_m2ad_enrol`을 검증하는 테스트 묶음이다.
# 후보 E0~E3(그대로, 이미지별 특징 중심화, 여러 조명 등록, 둘 다)의 팔 구성, 학습 시편만 쓰는 중첩 검증(val),
# 봉인 test 채점, 경보 뒤 자동 재등록(폐루프), CLI의 장부 기록과 분석 모듈 연결을 확인한다.
# [왜 필요한가] 7단계는 검증 결과로 후보를 고른 뒤 봉인 M2AD test를 방법마다 한 번만
# 읽는다(docs/experiments.md 7단계). 그래서 GPU와 실제 데이터 없이 가짜 백본·가짜 캐시로 같은 코드 경로를 미리
# 돌려 검증이 test 이미지를 건드리지 않는지, E0이 3-B와 같은 점수를 내는지 등을 먼저 확인한다.
# [주요 구성] 가짜 데이터 도우미 → 팔·등록 행렬 → 검증(학습 이미지만, 중첩 교차 적합, 등록·중심화 효과) →
# test(봉인, E0 = 3-B 재현, 다중 등록, 폐루프) → CLI(val, test, read_pick).
# [연결] 기대값은 run_m2ad.crossfit·run_inspector와 analyze_m2ad_enrol을 직접 불러 맞춰 본다.
# test_run_m2ad.py와 같은 방식의 가짜 데이터(GridMean, FakeCache)를 쓰지만 조명 효과와 결함 모양이 조금
# 다르다.

import hashlib
import json

import numpy as np
import pytest

# 뱅크·채점이 torch를 쓰므로 torch가 없는 CI 환경(.github/workflows/ci.yml)에서는 파일 전체를 건너뛴다.
torch = pytest.importorskip("torch")

# run_m2ad는 3-B 실행과 점수를 대조하고 교차 적합 기대값을 다시 계산하려고, analyze_m2ad_enrol은 실행 결과를
# 분석 모듈이 그대로 읽는지 확인하려고 가져온다. m2ad의 겹·재보정 시편·조명 묶음 함수와 splits.path_key는
# 기대값을 실행과 따로 계산하는 데 쓴다.
from defect_inspect import analyze_m2ad_enrol, ledger, paths, run_m2ad, run_m2ad_enrol  # noqa: E402
from defect_inspect.calibrate import conformal_threshold  # noqa: E402
from defect_inspect.configs import PatchCoreConfig  # noqa: E402
from defect_inspect.m2ad import (  # noqa: E402
    ILLUMINATIONS,
    VIEWS,
    M2adRow,
    illumination_groups,
    recal_specimens,
    specimen_folds,
)
from defect_inspect.splits import SealedTestError, path_key  # noqa: E402

# 가짜 데이터 크기: 16px 이미지(GridMean이 2x2 패치로 줄임), 학습 시편 30개(3-B·7단계 규칙과 같은 수), 테스트
# 정상 4개와 불량 5개. NEVER_VISIBLE 불량은 모든 조명에서 라벨 -1(보이지 않아 제외)이다.
SIZE = 16
N_TRAIN = 30
NORMAL = [f"{2 * i + 1:03d}" for i in range(4)]
ANOMALOUS = ["hole_1_000", "hole_1_001", "hole_1_002", "scratch_1_003", "scratch_2_004"]
NEVER_VISIBLE = "hole_1_001"


# 실제 백본 대신 쓰는 작은 특징 추출기: 2x2 칸마다 평균 색(3차원). 학습 가중치가 없어 결정적이고 CPU에서
# 빠르다.
class GridMean(torch.nn.Module):
    """백본 대신 쓰는 작은 추출기: 2x2 격자의 칸마다 평균색을 낸다."""

    name = "gridmean"
    dim = 3

    def forward(self, x):
        return torch.nn.functional.avg_pool2d(x.float(), SIZE // 2).permute(0, 2, 3, 1)


# M2AD 메타데이터를 흉내 낸 행 목록(시점 3 × 조명 10 × 학습 30 + 테스트 9). 경로의 SHA-256 순으로 섞어 실행
# 코드가 입력 순서에 기대지 않는지도 함께 본다.
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
                    label = -1 if name == NEVER_VISIBLE else 1
                    path = f"{category}/NG/{name}/A{view}_I{light}.png"
                    rows.append(M2adRow(category, "test", name, view, light, path, 1, label))
    return sorted(rows, key=lambda r: hashlib.sha256(r.img_path.encode()).hexdigest())


# M2adCache를 대신하는 가짜 캐시. splits에 읽은 분할 이름을 쌓아, 검증이 학습 이미지만 읽는지와 CLI가 허락
# 전에 아무것도 읽지 않는지 확인한다.
class FakeCache:
    """시편별 잡음, 조명마다 고른 어두워짐, 결함에는 시편마다 다른 색의 칸."""

    size = SIZE

    def __init__(self):
        self.splits: list[str] = []

    def _image(self, row: M2adRow) -> np.ndarray:
        key = f"{row.category}/{row.specimen}/{row.view}"
        rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
        img = rng.integers(100, 140, (SIZE, SIZE, 3), dtype=np.uint8)
        # 조명마다 이미지 전체를 8단계씩 고르게 어둡게 한다(조명 01은 그대로). 고른 밝기 이동은 이미지별
        # 중심화(E1)가 정확히 지울 수 있는 변화라서 E1 효과를 확인하는 데 쓴다.
        # 보이는 결함은 왼쪽 위 칸을 시편 잡음 생성기에서 뽑은 색으로 칠한다. 초록 채널이 0인 색이라 정상
        # 잡음(100~140)과 뚜렷이 다르고, 색은 시편마다 다르다.
        img = img - np.uint8(8 * ILLUMINATIONS.index(row.illumination))
        if row.label == 1:
            img[:8, :8] = rng.integers(0, 256, 3, dtype=np.uint8) // 2 * np.array([2, 0, 1], dtype=np.uint8)
        return img

    def images(self, rows):
        self.splits += sorted({r.split for r in rows})
        return np.stack([self._image(r) for r in rows])


# 테스트용 구성. coreset_ratio=1.0이라 뱅크가 모든 패치를 가져, 자기 이미지가 뱅크에 남으면 점수가 0이
# 된다(누수 감지).
CFG = PatchCoreConfig(
    name="toy", backbone="gridmean", img_size=SIZE, coreset_ratio=1.0, sigma=1.0, batch_size=16
)


# run_m2ad_enrol.Tools를 테스트 구성으로 만든다. centre=True면 실행 코드와 같은 centred()로 GridMean을 감싸
# E1·E3의 이미지별 중심화를 흉내 낸다.
def tools(centre=False):
    extractor = run_m2ad_enrol.centred(GridMean()) if centre else GridMean()
    return run_m2ad_enrol.Tools(CFG, extractor, "cpu")


# 모든 테스트에서 장부를 임시 경로로 돌리고 git 커밋 조회를 고정값으로 바꾼다. 실제
# reports/test_ledger.jsonl을 건드리지 않고, 기록 내용을 git 상태와 무관하게 비교하려는 것이다.
@pytest.fixture(autouse=True)
def temporary_ledger(tmp_path, monkeypatch):
    path = tmp_path / "reports" / "test_ledger.jsonl"
    monkeypatch.setattr(paths, "TEST_LEDGER", path)
    monkeypatch.setattr(ledger, "git_commit", lambda root=None: "abc1234")
    monkeypatch.setattr(run_m2ad_enrol, "git_commit", lambda root=None: "abc1234")
    return path


# npz를 열어 배열을 dict로 꺼낸 뒤 닫는다(Windows에서 열린 파일이 잠긴 채 남지 않게).
def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# 실행 코드(inspector_rows)를 쓰지 않고 같은 조건의 행을 직접 골라 시편 이름순으로 정렬한다. 기대값을
# 독립적으로 세우기 위한 도우미다.
def pick(rows, view, split, light, category="Motor"):
    wanted = (category, view, split, light)
    return sorted(
        (r for r in rows if (r.category, r.view, r.split, r.illumination) == wanted), key=lambda r: r.specimen
    )


# ---------------------------------------------------------------- 팔(arm) 구성


# 후보별 팔 구성을 확인한다. E0·E1은 기준 조명 I01 하나(팔 ref), E2·E3는 I01 + 묶음 A, I01 + 묶음 B 두 팔이다.
# 등록 행렬에서 I01은 모든 팔에 있고 나머지 아홉 조명은 정확히 한 팔에만 있어야, 각 조명이 다른 팔에서 한 번씩
# '못 본 조명'으로 재진다(7단계 조명 묶음 규칙).
# 폐루프 키는 조명 9개 × 팔 4개 = 36개이고, 후보 순서(E0 < E1 < E2 < E3)는 검증에서 동점일 때 단순한 쪽을
# 고르는 규칙에 쓰이므로 함께 고정한다.
def test_arms_and_enrolment():
    a, b = illumination_groups()
    assert run_m2ad_enrol.arms("e0") == run_m2ad_enrol.arms("e1") == [("ref", ("01",))]
    assert run_m2ad_enrol.arms("e2") == run_m2ad_enrol.arms("e3") == [("A", ("01", *a)), ("B", ("01", *b))]
    m = run_m2ad_enrol.enrolled_matrix("e2")
    assert m.shape == (2, 10) and m[:, 0].all()
    # 기준이 아닌 조명은 정확히 한 팔에만 등록되므로 다른 팔에서는 못 본 조명이다.
    assert (m[:, 1:].sum(axis=0) == 1).all()
    assert run_m2ad_enrol.enrolled_matrix("e0").tolist() == [[True] + [False] * 9]
    with pytest.raises(ValueError):
        run_m2ad_enrol.arms("e4")
    keys = run_m2ad_enrol.loop_keys()
    assert len(keys) == 36 and keys[:4] == ["02:trim:0", "02:trim:1", "02:trim:2", "02:none:2"]
    assert run_m2ad_enrol.CANDIDATES == ("e0", "e1", "e2", "e3")


# ---------------------------------------------------------------- 검증(validation)


# E0, E1, E2의 검증을 Motor 000 검사기에서 한 번씩 돌려 모듈 안 테스트들이 나눠 쓴다(테스트마다 다시 돌리면
# 느리다). E3 검증은 이 fixture에 없고 CLI 테스트에서 돈다.
@pytest.fixture(scope="module")
def val_runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("val")
    rows = make_rows()
    cache = FakeCache()
    infos = {}
    for candidate in ("e0", "e1", "e2"):
        t = tools(run_m2ad_enrol.CENTRE[candidate])
        infos[candidate] = run_m2ad_enrol.validate_inspector(t, candidate, "Motor", "000", rows, cache, out)
    return out, rows, cache, infos


# 검증(val)은 학습 분할만 읽어야 한다. 검증 결과로 후보를 고른 뒤에야 봉인 test를 읽는다는 7단계 규칙을 코드
# 수준에서 지키는 테스트다. E2 요약의 팔 구성도 확인한다.
def test_validation_reads_train_images_only(val_runs):
    _, _, cache, infos = val_runs
    assert set(cache.splits) == {"train"}
    assert infos["e2"]["arms"] == {
        "A": ["01", *illumination_groups()[0]],
        "B": ["01", *illumination_groups()[1]],
    }


# 검증이 시편 겹 위에서 '중첩'으로 도는지 확인한다. 바깥 겹 f마다 뱅크는 겹 f 밖 시편의 등록 조명 이미지로
# 만들고, 임계값은 같은 이미지의 나머지 4겹 교차 적합 점수로 정하며, 겹 f 시편은 열 조명 모두를 그 뱅크로
# 채점한다(평가 시편이 뱅크에도 임계값에도 들어가지 않음).
# 팔 B, 겹 2 하나를 손으로 다시 만들어 임계값·보정 장수·점수가 실행 결과와 같은지 비교한다.
def test_validation_is_nested_over_specimen_folds(val_runs):
    out, rows, _, _ = val_runs
    z = load(out / "Motor_000_e2.npz")
    names = [f"{2 * i:03d}" for i in range(N_TRAIN)]
    assert z["specimens"].tolist() == names
    fold_of = specimen_folds(names)
    assert z["folds"].tolist() == [fold_of[n] for n in names]
    assert z["arms"].tolist() == ["A", "B"] and z["lights"].tolist() == list(ILLUMINATIONS)
    assert z["scores"].shape == (2, 10, N_TRAIN) and z["thresholds"].shape == (2, 5)
    folds = z["folds"]
    # 팔 B, 겹 2를 손으로 다시 만든다: 다른 시편들의 등록 조명 이미지, 그 4겹의 교차 적합, 그리고 그
    # 이미지 전부로 만든 뱅크로 채점한 빼 둔 시편들.
    _, enrolled = run_m2ad_enrol.arms("e2")[1]
    t = tools()
    keep = folds != 2
    imgs = {light: FakeCache().images(pick(rows, "000", "train", light)) for light in ILLUMINATIONS}
    feats = torch.cat([t.features(imgs[light][keep]) for light in enrolled])
    bank, oof, thr, _ = run_m2ad.crossfit(
        t.build,
        t.score,
        feats,
        np.concatenate([imgs[light][keep] for light in enrolled]),
        np.concatenate([folds[keep]] * len(enrolled)),
    )
    assert z["thresholds"][1, 2] == thr.value == conformal_threshold(oof, 0.05).value
    assert z["cal_n"][1, 2] == len(oof) == len(enrolled) * keep.sum()
    held = np.flatnonzero(folds == 2)
    for li, light in enumerate(ILLUMINATIONS):
        np.testing.assert_allclose(z["scores"][1, li, held], t.score(bank, imgs[light][held]), rtol=1e-6)


# 가짜 데이터에서 두 대응책이 의도대로 작동하는지 본다. E0은 못 본 조명의 정상을 모두 불량으로
# 판정하고(오검출률 1.0), 고른 어두워짐을 지우는 E1은 0.2 이하로 낮춘다. E1에서 조명 06(인덱스 5)의 점수가
# 조명 01과 거의 같다는 것은 중심화가 밝기 이동을 지웠다는 뜻이다. E2는 등록한 조명(팔 A의 07, 팔 B의 05)에서
# E0이 모두 불량으로 낸 것을 대부분 정상으로 되돌린다.
def test_enrolment_and_centring_remove_the_false_alarms_of_unseen_light(val_runs):
    out, _, _, _ = val_runs

    # 못 본 조명의 오검출률: 시편마다 자기 바깥 겹 임계값과 비교해 불량 표시를 만들고, 등록 행렬이 거짓인 (팔,
    # 조명) 칸만 모아 평균한다. 검증의 학습 시편은 모두 정상이라 불량 표시 비율이 곧 오검출률이다.
    def unseen_fpr(candidate):
        z = load(out / f"Motor_000_{candidate}.npz")
        flagged = z["scores"] > z["thresholds"][:, None, z["folds"]]
        unseen = ~z["enrolled"][:, :, None].repeat(N_TRAIN, axis=2)
        return flagged[unseen].mean()

    # 어두운 이미지는 I01 패치 뱅크에서 멀다. 중심화는 고른 어두워짐을 완전히 지운다.
    assert unseen_fpr("e0") == 1.0
    assert unseen_fpr("e1") <= 0.2
    z1 = load(out / "Motor_000_e1.npz")
    np.testing.assert_allclose(z1["scores"][0, 5], z1["scores"][0, 0], rtol=1e-4)
    # 등록한 조명에서 찍은 다른 시편들을 담은 뱅크는 빼 둔 시편의 그 조명 이미지도 덮는다.
    z0, z2 = load(out / "Motor_000_e0.npz"), load(out / "Motor_000_e2.npz")
    for a, light in ((0, "07"), (1, "05")):
        li = ILLUMINATIONS.index(light)
        assert z2["enrolled"][a, li]
        assert (z0["scores"][0, li] > z0["thresholds"][0, z0["folds"]]).all()
        assert (z2["scores"][a, li] > z2["thresholds"][a, z2["folds"]]).mean() <= 0.2


# ---------------------------------------------------------------- 봉인 test


# 봉인: allow_test=True 없이 test 채점(sealed_inspector)과 폐루프(loop_inspector)를 부르면 SealedTestError로
# 멈춰야 한다.
def test_the_test_run_needs_permission(tmp_path):
    with pytest.raises(SealedTestError):
        run_m2ad_enrol.sealed_inspector(tools(), "e0", "Motor", "000", make_rows(), FakeCache(), tmp_path)
    with pytest.raises(SealedTestError):
        run_m2ad_enrol.loop_inspector(tools(), "Motor", "000", make_rows(), FakeCache(), tmp_path)


# E0이 3-B와 같은 절차이므로 같은 입력에서 3-B 실행(run_m2ad.run_inspector, 재보정 없음)과 임계값·보정
# 점수·조명별 점수·라벨이 정확히 같아야 한다. 7단계 규칙은 실제 실행에서도 이 일치(stage3b_match)를 적도록
# 했고, 이 테스트는 코드가 그 전제를 지키는지 미리 확인한다.
def test_e0_reproduces_the_stage_3b_scores(tmp_path):
    rows = make_rows()
    info = run_m2ad_enrol.sealed_inspector(
        tools(), "e0", "Motor", "000", rows, FakeCache(), tmp_path, allow_test=True
    )
    z = load(tmp_path / "Motor_000_e0.npz")
    ref = tmp_path / "ref"
    ref.mkdir()
    run_m2ad.run_inspector(
        CFG, "Motor", "000", rows, FakeCache(), GridMean(), "cpu", ref, allow_test=True, recal_sizes=()
    )
    old = load(ref / "Motor_000.npz")
    assert z["specimens"].tolist() == old["specimens"].tolist()
    assert z["thresholds"].tolist() == [float(old["threshold"])]
    assert np.array_equal(z["cal_score_ref"], old["cal_score"])
    conditions = old["conditions"].tolist()
    for li, light in enumerate(ILLUMINATIONS):
        k = conditions.index("S" if light == "01" else f"R:{light}")
        assert np.array_equal(z["scores"][0, li], old["scores"][k]), light
        assert np.array_equal(z["labels"][li], old["labels"][k]), light
    assert info["arms"]["ref"]["threshold"]["n"] == N_TRAIN


# 여러 조명 등록(E2)의 test 실행: 점수 모양은 (팔 2, 조명 10, 테스트 시편 9), 팔마다 임계값 하나다. 보정 점수
# 수는 팔 A가 30 × 5(I01 + 묶음 A 4개), 팔 B가 30 × 6(I01 + 묶음 B 5개)이다. 다른 시점(120)으로 돌려 시점
# 처리도 함께 본다.
def test_multi_enrolment_test_run(tmp_path):
    rows = make_rows()
    info = run_m2ad_enrol.sealed_inspector(
        tools(), "e2", "Motor", "120", rows, FakeCache(), tmp_path, allow_test=True
    )
    z = load(tmp_path / "Motor_120_e2.npz")
    assert z["scores"].shape == (2, 10, 9) and z["thresholds"].shape == (2,)
    assert len(z["cal_score_A"]) == N_TRAIN * 5 and len(z["cal_score_B"]) == N_TRAIN * 6
    assert info["arms"]["B"]["enrolled"] == ["01", *illumination_groups()[1]]
    assert z["object_anomaly"].tolist() == [0] * 4 + [1] * 5


# 폐루프 자동 재등록(E0, 7단계)을 확인한다. 새 조명 L과 팔 (거르기, k)마다 무라벨 묶음 20장 = 학습 시편
# SHA-256 순 앞 20 - k개(정상) + 보이는 불량 시편 앞 k개다. 등록에 쓰인 불량 시편은 검출률에서 빠지도록 라벨
# -1이 되고(늘 안 보이는 결함 하나와 합쳐 k + 1개), trim은 묶음 안 점수가 높은 4장(ceil(0.2 × 20))을 버려 결함
# k장을 모두 거르고 정상 4 - k장을 버린다. none은 20장을 모두 더한다.
# 마지막으로 정상 부분이 앞에서부터 자른 포함 관계인지, 05:trim:2 재등록 뒤 새 조명 정상의 75% 이상이 임계값
# 아래로 돌아오는지 본다.
def test_closed_loop_drops_the_mixed_in_defects(tmp_path):
    rows = make_rows()
    cache = FakeCache()
    info = run_m2ad_enrol.loop_inspector(tools(), "Motor", "000", rows, cache, tmp_path, allow_test=True)
    z = load(tmp_path / "Motor_000_loop.npz")
    keys = run_m2ad_enrol.loop_keys()
    assert z["loop_keys"].tolist() == keys and z["scores"].shape == (36, 9)
    names = z["specimens"].tolist()
    visible = sorted((n for n in ANOMALOUS if n != NEVER_VISIBLE), key=path_key)
    for key in keys:
        light, filt, k = key.split(":")
        k = int(k)
        entry = info["loop"][key]
        assert entry["batch"] == 20 and entry["defects_in_batch"] == k
        labels = z["labels"][keys.index(key)]
        # 등록에 쓰인 불량 시편(SHA-256 순으로 보이는 것 앞 k개)은 검출률에서 빠진다.
        for i, name in enumerate(names):
            if name in visible[:k]:
                assert labels[i] == -1
        assert (labels == -1).sum() == k + 1  # 끝내 안 보이는 결함 하나를 더해
        if filt == "trim":
            # 결함 칸은 묶음 안에서 점수가 가장 높으므로 거르기가 결함부터 버린다.
            assert entry["dropped_defects"] == k and entry["dropped_normals"] == 4 - k
            assert entry["enrolled"] == N_TRAIN + 16
        else:
            assert entry["dropped_defects"] == 0 and entry["enrolled"] == N_TRAIN + 20
    # 묶음의 정상 부분은 SHA-256 순 학습 시편 앞 20 - k개다(앞에서부터 자른 포함 관계).
    assert (
        recal_specimens([f"{2 * i:03d}" for i in range(N_TRAIN)], 18)
        == recal_specimens([f"{2 * i:03d}" for i in range(N_TRAIN)], 20)[:18]
    )
    # 재등록하면 새 조명의 정상이 임계값 아래로 돌아온다.
    j = keys.index("05:trim:2")
    normal = z["labels"][j] == 0
    assert (z["scores"][j][normal] <= z["thresholds"][j]).mean() >= 0.75


# ---------------------------------------------------------------- 명령줄(CLI)


# CLI(main) 테스트 준비. 메타데이터·캐시·백본·구성을 가짜로 바꾸고, 검사기를 두 개로 줄이고, 출력 폴더를 임시
# 경로로 돌린다. 돌려주는 cache로 무엇을 읽었는지 확인한다.
@pytest.fixture
def project(tmp_path, monkeypatch):
    rows = make_rows()
    cache = FakeCache()
    monkeypatch.setattr(run_m2ad_enrol, "read_meta", lambda path: rows)
    monkeypatch.setattr(run_m2ad_enrol, "M2adCache", lambda root, size: cache)
    monkeypatch.setattr(run_m2ad_enrol, "_make_extractor", lambda cfg, device: GridMean())
    monkeypatch.setattr(run_m2ad_enrol, "get_config", lambda name: CFG)
    monkeypatch.setattr(run_m2ad_enrol, "INSPECTORS", ("Motor_000", "Motor_120"))
    monkeypatch.setattr(paths, "OUTPUTS", tmp_path / "outputs")
    return cache


# val 명령: 고른 후보(E0, E3) × 검사기 2개의 npz와 run.json(조명 묶음 A = 03, 07, 08, 09 포함)을 쓰고, 학습
# 이미지만 읽으며 장부에 줄을 쓰지 않는다. 끝난 실행은 --overwrite 없이 다시 돌지 않는다.
# 이어서 analyze_m2ad_enrol이 이 결과를 그대로 읽어, 못 본 조명 정상 수(조명 9 × 시편 30 × 검사기 2)를 맞게
# 세고 오검출이 적은 E3를 고르는지 확인한다(실행과 분석의 파일 형식이 어긋나는 회귀를 막는다).
def test_cli_validation(project, tmp_path, temporary_ledger):
    run_m2ad_enrol.main(["val", "--method", "p0", "--device", "cpu", "--candidates", "e0", "e3"])
    out = tmp_path / "outputs" / "m2ad-enrol-val-p0"
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["command"] == "val" and run["candidates"] == ["e0", "e3"]
    assert run["inspectors"] == ["Motor_000", "Motor_120"] and run["groups"]["A"] == ["03", "07", "08", "09"]
    assert sorted(p.name for p in out.glob("*.npz")) == [
        "Motor_000_e0.npz",
        "Motor_000_e3.npz",
        "Motor_120_e0.npz",
        "Motor_120_e3.npz",
    ]
    assert set(project.splits) == {"train"} and not temporary_ledger.exists()
    with pytest.raises(SystemExit):
        run_m2ad_enrol.main(["val", "--method", "p0", "--device", "cpu"])  # 끝난 실행은 그대로 둔다

    # 분석은 쓰인 그대로의 실행을 읽고, 못 본 조명 오검출이 적은 후보를 고른다.
    report = analyze_m2ad_enrol.build_val_report({"p0": analyze_m2ad_enrol.load_val(out)}, n_boot=20)
    rows = report["methods"]["p0"]["candidates"]
    assert rows["e0"]["unseen"]["n_normal"] == rows["e3"]["unseen"]["n_normal"] == 9 * N_TRAIN * 2
    assert rows["e0"]["unseen"]["fpr"] == 1.0 and rows["e3"]["unseen"]["fpr"] < 0.5
    assert report["methods"]["p0"]["pick"] == "e3"


# test 명령: --allow-test가 없거나 검증 리포트(val.json)가 없으면 argparse 오류로 멈추고, 그때까지 이미지를
# 읽지도 장부를 쓰지도 않아야 한다. 정상 실행은 장부에 줄을 하나 남기고, E0과 검증에서 고른 후보(E2)를 같은
# 실행에서 재며, --loop면 검사기마다 폐루프 npz를 쓴다. 마지막으로 analyze_m2ad_enrol의 test 리포트가 고른
# 후보와 폐루프 팔 순서를 그대로 읽는지 본다.
def test_cli_test_records_the_read_and_runs_e0_and_the_pick(project, tmp_path, temporary_ledger):
    val = tmp_path / "val.json"
    base = ["test", "--method", "p0", "--device", "cpu", "--pick-from", str(val)]
    with pytest.raises(SystemExit):
        run_m2ad_enrol.main(base)  # --allow-test 없음
    with pytest.raises(SystemExit):
        run_m2ad_enrol.main(base + ["--allow-test", "--stage", "7-m2ad"])  # 검증 리포트 없음
    val.write_text(json.dumps({"methods": {"p0": {"pick": "e2"}}}), encoding="utf-8")
    assert project.splits == [] and not temporary_ledger.exists()
    run_m2ad_enrol.main(base + ["--allow-test", "--stage", "7-m2ad", "--loop"])
    entries = [json.loads(line) for line in temporary_ledger.read_text(encoding="utf-8").splitlines()]
    assert [(e["stage"], e["config"], e["commit"]) for e in entries] == [
        ("7-m2ad", "m2ad-enrol-p0", "abc1234")
    ]
    out = tmp_path / "outputs" / "m2ad-enrol-test-p0"
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["candidates"] == ["e0", "e2"] and run["loop"]["n"] == 20
    assert len(run["loop_runs"]) == 2
    assert (out / "Motor_120_loop.npz").exists() and (out / "Motor_000_e2.npz").exists()

    report = analyze_m2ad_enrol.build_test_report({"p0": analyze_m2ad_enrol.load_test(out)}, n_boot=20)
    entry = report["methods"]["p0"]
    assert entry["verdicts"]["pick"] == "e2" and entry["candidates"]["e0"]["unseen"]["fpr"] == 1.0
    assert [(row["filter"], row["k"]) for row in entry["loop"]] == [tuple(a) for a in run["loop"]["arms"]]
    assert all(row["dropped_defects"] == row["defects_in_batches"] for row in entry["loop"][:3])


# 검증 리포트에서 방법별 선택 후보를 읽는 함수. 그 방법의 선택이 없거나 알 수 없는 후보 이름이면 ValueError로
# 멈춰, 엉뚱한 후보로 봉인 test를 읽는 일을 막는다.
def test_read_pick(tmp_path):
    path = tmp_path / "val.json"
    path.write_text(json.dumps({"methods": {"p0": {"pick": "e1"}}}), encoding="utf-8")
    assert run_m2ad_enrol.read_pick(path, "p0") == "e1"
    with pytest.raises(ValueError):
        run_m2ad_enrol.read_pick(path, "d-s")
    path.write_text(json.dumps({"methods": {"p0": {"pick": "e9"}}}), encoding="utf-8")
    with pytest.raises(ValueError):
        run_m2ad_enrol.read_pick(path, "p0")
