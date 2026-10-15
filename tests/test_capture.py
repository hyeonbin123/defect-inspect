# [파일 역할] 9단계 촬영 통계 모듈 `defect_inspect.capture`를 검증한다: 특징 11개가 뜻대로 움직이는지(흐림은
# 초점 통계를 낮추고, 밝기는 평균을 옮기고, 검은 배경은 전경 비율로 잡힌다), 점수 모형(바닥값·수축)이 거의
# 변하지 않는 변수에 과민하지 않은지, 교차 적합이 자기 사진을 빼는지, 게이트·스트림·판정 표가 규칙대로 세는지.
# [왜 필요한가] 9단계는 봉인 데이터 없이 CPU로 한 번 재는 측정이다(docs/experiments.md 9단계). 실제 데이터로
# 돌리기 전에 작은 합성 자료로 같은 코드 경로를 점검해, 결과 수치가 코드 오류가 아니라 데이터에서 나왔음을
# 보장한다.
# [연결] drift(p-value, 마팅게일), conditions(교란), calibrate(conformal 임계값)는 실행 코드와 같은 것을 쓴다.
# torch가 필요 없어 CI에서도 돈다.

import json

import numpy as np
import pytest

from defect_inspect import capture, drift
from defect_inspect.calibrate import conformal_threshold
from defect_inspect.conditions import CLEAN, apply_condition

# 교란 강도는 256 격자 기준이라(흐림 σ는 크기에 비례) 256px 합성 사진을 쓴다.
SIZE = 256


# 결정적인 합성 사진: 회색 바탕에 줄무늬(초점 통계가 0이 아니게)와 시드별 잡음.
def _photos(n: int, seed: int = 0, base: int = 120) -> np.ndarray:
    rng = np.random.default_rng(seed)
    images = rng.integers(base - 10, base + 10, (n, SIZE, SIZE, 3)).astype(np.int16)
    images[:, :, ::4] += 40
    return np.clip(images, 0, 255).astype(np.uint8)


def test_features_move_the_way_the_rule_says():
    photos = _photos(8)
    feats = capture.capture_features(photos)
    assert feats.shape == (8, len(capture.FEATURES)) and np.isfinite(feats).all()
    lap = capture.FEATURES.index("log_laplacian_var")
    # 흐림은 초점(라플라시안 분산)을 낮춘다.
    blurred = capture.capture_features(apply_condition(photos, "blur-3"))
    assert (blurred[:, lap] < feats[:, lap]).all()
    # 밝기 0.7배는 채널 평균을 낮춘다.
    darker = capture.capture_features(apply_condition(photos, "brightness-3"))
    assert (darker[:, :3] < feats[:, :3]).all()
    # 검은 배경 위의 작은 물체: 전경 비율은 물체 넓이, 전경 평균은 물체 밝기.
    black = np.zeros((1, SIZE, SIZE, 3), np.uint8)
    black[0, :32, :32] = 200
    f = capture.capture_features(black)[0]
    assert f[capture.FEATURES.index("foreground_share")] == pytest.approx(32 * 32 / SIZE**2)
    assert f[capture.FEATURES.index("foreground_y_mean")] == pytest.approx(200, abs=0.5)
    # 포화 비율: 어느 채널이든 250 이상.
    white = np.full((1, SIZE, SIZE, 3), 100, np.uint8)
    white[0, 0, :, 2] = 255
    assert capture.capture_features(white)[0, capture.FEATURES.index("saturated_share")] == pytest.approx(
        1 / SIZE
    )
    # 64장씩 나눠 계산해도 한 번에 계산한 것과 같다.
    many = _photos(70, seed=3)
    assert np.array_equal(capture.capture_features(many), capture.capture_features(many[:70]))
    with pytest.raises(ValueError, match="uint8"):
        capture.capture_features(many.astype(np.float32))


# 바닥값 덕분에 모든 사진에서 같은 값인 변수(분산 0)가 있어도 모형을 맞출 수 있고, 그 변수가 아주 조금 바뀐
# 사진에 무한대 점수를 주지 않는다. 표본이 2장보다 적으면 멈춘다.
def test_the_model_tolerates_constant_features_and_scores_outliers_higher():
    rng = np.random.default_rng(0)
    feats = rng.normal(100, 2, (50, len(capture.FEATURES)))
    feats[:, capture.FEATURES.index("foreground_share")] = 1.0  # VisA처럼 늘 1
    model = capture.CaptureModel.fit(feats)
    assert np.isfinite(model.precision).all()
    typical = model.score(feats[:5])
    nudged = feats[:1].copy()
    nudged[0, capture.FEATURES.index("foreground_share")] = 0.9995
    assert np.isfinite(model.score(nudged)).all()
    far = feats[:1].copy()
    far[0, 0] += 40
    assert model.score(far)[0] > typical.max()
    with pytest.raises(ValueError, match="at least two"):
        capture.CaptureModel.fit(feats[:1])


# 교차 적합: 겹마다 그 겹을 뺀 사진으로 맞춘 점수라서, 자기 자신을 넣고 맞춘 점수(재대입)보다 크거나 같다.
def test_crossfit_scores_leave_each_fold_out():
    rng = np.random.default_rng(1)
    feats = rng.normal(0, 1, (40, len(capture.FEATURES)))
    folds = np.arange(40) % 4
    oof = capture.crossfit_scores(feats, folds)
    for fold in range(4):
        held = folds == fold
        expected = capture.CaptureModel.fit(feats[~held]).score(feats[held])
        assert np.allclose(oof[held], expected)
    resub = capture.CaptureModel.fit(feats).score(feats)
    assert oof.mean() > resub.mean()


# 게이트 표: 범주마다 교차 적합 점수로 정한 1% 임계값보다 큰 사진을 센다. 흐린 사진은 거의 다 걸리고, 같은
# 촬영의 정상은 거의 안 걸린다.
def _visa_like(seed: int) -> dict:
    pool = _photos(80, seed=seed)
    evals = np.concatenate([_photos(30, seed=seed + 100), _photos(10, seed=seed + 200)])
    labels = np.array([0] * 30 + [1] * 10, dtype=np.int8)
    evals[30:, :24, :24] = 255  # 작은 결함
    data = {
        "pool": capture.capture_features(pool),
        "pool_folds": np.arange(80) % 4 + 1,
        "eval_images": [f"img{i}" for i in range(40)],
        "labels": labels,
        "eval": {name: capture.capture_features(apply_condition(evals, name)) for name in (CLEAN, "blur-2")},
    }
    return capture.visa_scores(data)


def test_gate_table_counts_gated_images_per_condition():
    scores = {"a": _visa_like(0), "b": _visa_like(1)}
    for s in scores.values():
        assert s["gate"].value == conformal_threshold(s["cal"], capture.ALPHA_GATE).value
    table = capture.gate_table(scores)
    assert table["blur-2"]["normal"]["n"] == 60 and table[CLEAN]["defect"]["n"] == 20
    assert table["blur-2"]["normal"]["rate"] > 0.9
    assert table[CLEAN]["normal"]["rate"] < 0.1
    manual = sum(
        int((s["eval"]["blur-2"][s["labels"] == 0] > s["gate"].value).sum()) for s in scores.values()
    )
    assert table["blur-2"]["normal"]["gated"] == manual
    verdict = capture.h26(table | {"blur-2": table["blur-2"]})
    assert (
        verdict["verdict"] in ("지지", "기각")
        and verdict["blur_normal_gated"] == table["blur-2"]["normal"]["rate"]
    )


# 같은 사진 번호 스트림을 두 감지기에 쓰는 도우미: 정상 번호 전부와 결함 번호 k개가 스트림마다 한 번씩 나온다.
def test_index_streams_hold_every_normal_once_and_k_defects():
    rng = drift.rng_for("test")
    streams, k = capture._index_streams(20, 5, 0.10, 7, rng)
    assert k == drift.n_mixed(20, 0.10, 5) and streams.shape == (7, 20 + k)
    for row in streams:
        assert sorted(row[row < 20]) == list(range(20))
        assert ((row >= 20) & (row < 25)).sum() == k and len(set(row)) == len(row)


# M2AD 표: 조명 L만 크게 다르면 대부분 20장 안에 경보하고, 같은 조명이면 변화 전·후 모두 조용하다. 보정 집합은
# 겹 f 밖 시편의 I01 교차 적합 점수다.
def test_m2ad_tables_alarm_on_a_changed_light_and_count_within_20():
    from defect_inspect.m2ad import ILLUMINATIONS, REFERENCE

    rng = np.random.default_rng(5)
    folds = np.arange(30) % 5
    feats = {light: rng.normal(0, 1, (30, len(capture.FEATURES))) for light in ILLUMINATIONS}
    feats["02"][:, 0] += 30  # 조명 02만 크게 바뀜
    data = capture.m2ad_scores({"folds": folds, "features": feats})
    for fold, cal in data["cal"].items():
        assert np.array_equal(cal, data["scores"][REFERENCE][folds != fold])
    tables = capture.m2ad_tables({"Motor_000": data}, n_streams=50)
    assert tables["delay"]["02"]["within_20_share"] > 0.9
    assert tables["delay"]["03"]["within_20_share"] < 0.5
    assert tables["gate"]["02"]["rate"] == 1.0
    assert set(tables["delay"]) == {light for light in ILLUMINATIONS if light != REFERENCE}


def test_h27_needs_both_parts():
    m2ad = {"delay": {f"{i:02d}": {"within_20_share": 0.6 if i < 9 else 0.1} for i in range(2, 11)}}
    fa = {"0.10": {"capture": {"rate": 0.005}}}
    assert capture.h27(fa, m2ad)["verdict"] == "지지"  # 일곱 조명(02~08)이 50% 이상
    assert capture.h27({"0.10": {"capture": {"rate": 0.02}}}, m2ad)["verdict"] == "기각"
    m2ad["delay"]["08"]["within_20_share"] = 0.4
    assert capture.h27(fa, m2ad)["verdict"] == "기각"


# 서빙 모델 표는 저장된 6단계 점수의 사진 순서가 평가 순서와 같아야 하고, 흐림 σ 1.0의 정상 오검출률이 기록과
# 다르면 표를 내지 않는다.
def test_dms_table_checks_the_record_and_the_image_order(tmp_path):
    scores = {"a": _visa_like(0)}
    s = scores["a"]
    normal = s["labels"] == 0
    for name, run in capture.DMS_RUNS.items():
        (tmp_path / run).mkdir()
        score = np.where(normal, 0.1, 0.9).astype(np.float32)
        if name == "blur-2":
            score = score.copy()
            score[np.flatnonzero(normal)[:9]] = 0.5  # 정상 30장 가운데 9장이 임계값을 넘는다 = 30%
        np.savez(
            tmp_path / run / "a.npz",
            eval_images=np.array(s["eval_images"]),
            eval_labels=s["labels"],
            eval_score=score,
        )
    out = capture.dms_table(scores, tmp_path)
    assert out["check"]["blur-2_normal_flagged"] == 0.3 and out["check"]["ok"] is False
    assert "blur-2" not in out
    np.savez(
        tmp_path / capture.DMS_RUNS["clean"] / "a.npz",
        eval_images=np.array(s["eval_images"][::-1]),
        eval_labels=s["labels"],
        eval_score=np.zeros(40, np.float32),
    )
    with pytest.raises(ValueError, match="images differ"):
        capture.dms_table(scores, tmp_path)


def test_plain_turns_numpy_and_nan_into_json():
    value = {"a": np.float32(1.5), "b": [np.int64(2), float("nan")], "c": np.array([1, 2])}
    assert json.loads(json.dumps(capture._plain(value))) == {"a": 1.5, "b": [2, None], "c": [1, 2]}
