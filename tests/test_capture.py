# [파일 역할] 9단계 촬영 통계 모듈 `defect_inspect.capture`를 검증한다: 특징 11개가 뜻대로 움직이는지(흐림은
# 초점 통계를 낮추고, 밝기는 평균을 옮기고, 검은 배경은 전경 비율로 잡힌다), 점수 모형(바닥값·수축)이 거의
# 변하지 않는 변수에 과민하지 않은지, 교차 적합이 자기 사진을 빼는지, 게이트·스트림·판정 표가 규칙대로 세는지.
# [왜 필요한가] 9단계는 봉인 데이터 없이 CPU로 한 번 재는 측정이다(docs/experiments.md 9단계). 실제 데이터로
# 돌리기 전에 작은 합성 자료로 같은 코드 경로를 점검해, 결과 수치가 코드 오류가 아니라 데이터에서 나왔음을
# 보장한다.
# [연결] drift(p-value, 마팅게일), conditions(교란), calibrate(conformal 임계값)는 실행 코드와 같은 것을 쓴다.
# torch가 필요 없어 CI에서도 돈다.
# [9단계 보충] 결함 마스크 밖 화소만으로 같은 통계를 내는 마스크 맞춤 채점(`capture --masked`)도 여기서 본다:
# 빈 마스크면 9단계 통계·게이트와 비트 단위로 같고, 밝은 사각형을 가리면 배경 값이 나오고, 대조 정상의 마스크
# 배정이 경로 SHA-256 순으로 결정적이고, 해석 규칙과 9단계 기록 대조가 규칙(docs/experiments.md)대로
# 동작하는지.

import hashlib
import json

import numpy as np
import pytest

from defect_inspect import capture, drift
from defect_inspect.calibrate import conformal_threshold
from defect_inspect.conditions import CLEAN, apply_condition
from defect_inspect.splits import ManifestRow

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


# ---------------------------------------------------------------- 9단계 보충: 결함 마스크 밖 촬영 통계


# 마스크가 비어 있으면(모든 화소를 쓰면) 마스크 통계가 9단계 통계와 비트 단위로 같다. 64장 묶음 경계를 넘는
# 70장, 채널 합이 2^24를 넘어 float32 합이 정확하지 않은 밝은 280px 사진, 검은 배경이 절반인
# 사진(전경 비율 < 1)으로 모든 변수의 계산 경로를 지난다.
def test_an_empty_mask_gives_the_stage9_statistics_bit_for_bit():
    photos = _photos(70, seed=3)
    keep = np.ones((SIZE, SIZE), dtype=bool)
    assert np.array_equal(capture.capture_features(photos, keep=keep), capture.capture_features(photos))
    rng = np.random.default_rng(4)
    bright = rng.integers(200, 256, (5, 280, 280, 3)).astype(np.uint8)
    bright[1, :140] = 0
    keep280 = capture.outside_pixels(np.zeros((256, 256), np.uint8), 280)
    assert keep280.shape == (280, 280) and keep280.dtype == bool and keep280.all()
    assert np.array_equal(capture.capture_features(bright, keep=keep280), capture.capture_features(bright))


# 밝은 사각형을 마스크로 가리면 마스크 밖 통계는 배경 값이다. 필터는 이미지 전체에 걸리므로 팽창 없이 가리면
# 사각형 바로 옆 화소의 라플라시안·기울기가 통계에 남고, 반경 4px(9×9 정사각형) 팽창은 그것까지 뺀다.
def test_masked_statistics_leave_out_a_bright_square():
    mask = np.zeros((256, 256), np.uint8)
    mask[100:120, 60:90] = 1
    square = ~capture.outside_pixels(mask, 280, radius=0)
    h, w = int(square.any(axis=1).sum()), int(square.any(axis=0).sum())
    assert square.sum() == h * w  # NEAREST 확대는 직사각형을 직사각형으로 키운다
    assert 21 <= h <= 23 and 32 <= w <= 34  # 20·30칸 × 280/256
    image = np.full((1, 280, 280, 3), 100, np.uint8)
    image[0][square] = 255
    keep = capture.outside_pixels(mask, 280)
    assert (~keep).sum() == (h + 2 * capture.DILATION) * (w + 2 * capture.DILATION)
    f = dict(zip(capture.FEATURES, capture.capture_features(image, keep=keep)[0], strict=True))
    assert f["mean_r"] == f["mean_g"] == f["mean_b"] == 100.0
    for name in ("y_p01", "y_p99", "foreground_y_mean"):
        assert f[name] == pytest.approx(100.0, abs=1e-3)
    for name in ("y_std", "log_laplacian_var", "log_gradient_mean"):
        assert f[name] == pytest.approx(0.0, abs=1e-3)
    assert f["saturated_share"] == 0.0 and f["foreground_share"] == 1.0
    whole = dict(zip(capture.FEATURES, capture.capture_features(image)[0], strict=True))
    assert whole["mean_r"] > 100.5 and whole["saturated_share"] > 0 and whole["log_laplacian_var"] > 1
    bare = dict(zip(capture.FEATURES, capture.capture_features(image, keep=~square)[0], strict=True))
    assert bare["mean_r"] == 100.0 and bare["log_laplacian_var"] > 0.1 and bare["log_gradient_mean"] > 0.1


# 마스크·keep 입력 검사와 가장자리: 마스크가 이미지 전체면 쓸 화소가 없어 멈추고, keep이 bool이 아니거나
# 크기가 다르면 멈춘다(결함 마스크(1 = 결함)를 keep(True = 쓸 화소)으로 잘못 넘기는 실수를 막는다). 모서리 칸
# 하나는 팽창 뒤 이미지 안쪽으로만 4px 넓어진다.
def test_outside_pixels_and_keep_are_checked():
    assert not capture.outside_pixels(np.ones((256, 256), np.uint8), 280).any()
    photos = _photos(2)
    with pytest.raises(ValueError, match="no pixels"):
        capture.capture_features(photos, keep=np.zeros((SIZE, SIZE), bool))
    with pytest.raises(ValueError, match="keep"):
        capture.capture_features(photos, keep=np.ones((SIZE, SIZE), np.uint8))
    with pytest.raises(ValueError, match="keep"):
        capture.capture_features(photos, keep=np.ones((SIZE, SIZE + 1), bool))
    with pytest.raises(ValueError, match="mask"):
        capture.outside_pixels(np.zeros((2, 256, 256), np.uint8), 280)
    corner = np.zeros((256, 256), np.uint8)
    corner[0, 0] = 1
    cell = ~capture.outside_pixels(corner, 280, radius=0)
    h, w = int(cell.any(axis=1).sum()), int(cell.any(axis=0).sum())
    grown = ~capture.outside_pixels(corner, 280)
    assert grown.sum() == (h + capture.DILATION) * (w + capture.DILATION) and grown[0, 0]


# 대조 정상의 마스크 배정: 정상과 결함을 각각 경로 SHA-256 순으로 줄 세워 i번째 정상에 (i mod D)번째 결함을
# 준다. 입력 순서를 바꿔도 이름끼리의 짝은 같고, 결함마다 받는 정상 수는 많아야 1장 차이다.
def test_control_masks_follow_the_sha256_order():
    normals = [f"pcb2/Data/Images/Normal/{i:04d}.JPG" for i in range(13)]
    defects = [f"pcb2/Data/Images/Anomaly/{j:03d}.JPG" for j in range(5)]
    assign = capture.control_assignment(normals, defects)

    def sha(name: str) -> str:
        return hashlib.sha256(name.encode("utf-8")).hexdigest()

    ranked_normals, ranked_defects = sorted(normals, key=sha), sorted(defects, key=sha)
    expected = {n: ranked_defects[i % 5] for i, n in enumerate(ranked_normals)}
    assert {n: defects[j] for n, j in zip(normals, assign, strict=True)} == expected
    shuffled_n, shuffled_d = normals[::-1], defects[2:] + defects[:2]
    again = capture.control_assignment(shuffled_n, shuffled_d)
    assert {n: shuffled_d[j] for n, j in zip(shuffled_n, again, strict=True)} == expected
    assert sorted(np.bincount(assign, minlength=5)) == [2, 2, 3, 3, 3]
    with pytest.raises(ValueError, match="defect"):
        capture.control_assignment(normals, [])


# 작은 합성 범주: 정상 풀 80장(겹 1~4), 겹 0 정상 30장, 밝은 8×8 결함이 있는 결함 사진 6장, 빈 256 격자
# 마스크.
def _category(seed: int, size: int = 48) -> dict:
    rng = np.random.default_rng(seed)

    def photos(n: int) -> np.ndarray:
        x = rng.integers(110, 130, (n, size, size, 3)).astype(np.int16)
        x[:, :, ::4] += 40
        return np.clip(x, 0, 255).astype(np.uint8)

    defect = photos(6)
    defect[:, 8:16, 8:16] = 255
    return {
        "pool_images": photos(80),
        "pool_folds": np.arange(80) % 4 + 1,
        "normal_images": photos(30),
        "normal_names": [f"n{seed}-{i:02d}" for i in range(30)],
        "defect_images": defect,
        "defect_names": [f"d{seed}-{j}" for j in range(6)],
        "defect_types": ["spot"] * 6,
        "masks": np.zeros((6, 256, 256), np.uint8),
    }


# 9단계 채점(visa_scores) 그대로: 정상 풀 교차 적합 1% 임계값과 풀 전체 모형의 점수.
def _stage9(data: dict) -> dict:
    n_normal, n_defect = len(data["normal_names"]), len(data["defect_names"])
    evals = np.concatenate([data["normal_images"], data["defect_images"]])
    return capture.visa_scores(
        {
            "pool": capture.capture_features(data["pool_images"]),
            "pool_folds": data["pool_folds"],
            "eval_images": data["normal_names"] + data["defect_names"],
            "labels": np.array([0] * n_normal + [1] * n_defect, dtype=np.int8),
            "eval": {CLEAN: capture.capture_features(evals)},
        }
    )


# 빈 마스크로 마스크 맞춤 채점을 돌리면 9단계 채점과 같은 임계값·점수가 나와 게이트 결과가 같다(회귀). 결함을
# 덮는 마스크를 주면 그 결함 사진의 점수는 마스크 없는 점수보다 작아진다(결함 화소가 빠진다).
def test_empty_masks_reproduce_the_stage9_gate():
    data = _category(0)
    s = capture.masked_scores(data)
    ref = _stage9(data)
    clean = ref["eval"][CLEAN]
    assert s["unmasked"]["gate"].value == ref["gate"].value
    assert np.array_equal(s["unmasked"]["normal_score"], clean[:30])
    assert np.array_equal(s["unmasked"]["defect_score"], clean[30:])
    assert all(g.value == ref["gate"].value for g in s["masked"]["gate"])
    assert np.array_equal(s["masked"]["defect_score"], clean[30:])
    assert np.array_equal(s["control"]["score"], clean[:30])
    assert np.array_equal(s["masked"]["defect_z"], s["unmasked"]["defect_z"])
    assert (s["masked"]["kept_share"] == 1.0).all()
    assert np.array_equal(
        s["control"]["mask"], capture.control_assignment(data["normal_names"], data["defect_names"])
    )

    covered = _category(0)
    covered["masks"][:, 40:88, 40:88] = 1  # 48px의 8~15 화소에 닿는 256 격자 칸
    m = capture.masked_scores(covered)
    assert (m["masked"]["kept_share"] < 1).all()
    assert (m["masked"]["defect_score"] < m["unmasked"]["defect_score"]).all()


# 해석 규칙(측정 전에 고정): 두 범주 다 10/20 이상이면 촬영 조건 차이 가능성, 다 3/20 이하면 결함 영역 때문,
# 한쪽 10/20 이상·다른 쪽 3/20 이하면 범주마다, 그 밖은 가르지 못함. 대조 정상 게이트가 3%를 넘거나 9단계
# 게이트 수가 다시 나오지 않으면 결론을 내지 않는다(3%와 같으면 넘은 것이 아니다).
def test_masked_reading_follows_the_rule():
    def read(pcb2, pipe, control=0.01, ok=True):
        return capture.masked_reading({"pcb2": [pcb2, 20], "pipe_fryum": [pipe, 20]}, control, ok)

    outside = read(10, 14)
    assert outside["overall"] == "결함 밖 영역도 정상과 촬영 통계가 다르다(촬영 조건 차이 가능성)"
    assert outside["per_category"] == {"pcb2": "결함 밖도 다름", "pipe_fryum": "결함 밖도 다름"}
    assert read(3, 0)["overall"] == "게이트는 결함 영역 때문이었다"
    split = read(12, 2)
    assert split["overall"] == "범주마다 갈림"
    assert split["per_category"] == {"pcb2": "결함 밖도 다름", "pipe_fryum": "결함 영역 때문"}
    undecided = read(9, 12)
    assert undecided["overall"] == "가르지 못했다" and undecided["per_category"]["pcb2"] == "가르지 못함"
    assert read(4, 3)["overall"] == "가르지 못했다"
    assert read(10, 14, control=0.03)["overall"].startswith("결함 밖")
    stopped = read(10, 14, control=0.0301)
    assert stopped["overall"].startswith("결론 없음") and stopped["per_category"] is None
    assert read(10, 14, ok=False)["overall"].startswith("결론 없음")
    assert capture.masked_reading({"pcb2": [10, 20]}, 0.01, True)["overall"].startswith("해석 없음")


# 리포트: 범주별·합산 게이트 수가 점수와 임계값에서 다시 센 값과 같고, 9단계 기록(per_category의 [게이트 수,
# 장수])과 다르면 대조가 실패해 결론을 내지 않는다. 결과는 JSON으로 쓸 수 있다.
def test_masked_tables_count_gates_and_check_the_stage9_record():
    per = {"pcb2": capture.masked_scores(_category(1)), "pipe_fryum": capture.masked_scores(_category(2))}

    def unmasked_counts(s: dict) -> dict:
        gate = s["unmasked"]["gate"].value
        return {
            "normal": [int((s["unmasked"]["normal_score"] > gate).sum()), len(s["unmasked"]["normal_score"])],
            "defect": [int((s["unmasked"]["defect_score"] > gate).sum()), len(s["unmasked"]["defect_score"])],
        }

    recorded = {"visa": {"gate": {CLEAN: {"per_category": {c: unmasked_counts(s) for c, s in per.items()}}}}}
    report = capture.masked_tables(per, recorded)
    assert report["stage9_check"]["ok"] is True
    gates = [g.value for g in per["pcb2"]["masked"]["gate"]]
    defect_hits = int((per["pcb2"]["masked"]["defect_score"] > np.array(gates)).sum())
    assert report["gate"]["masked"]["per_category"]["pcb2"]["defect"] == [defect_hits, 6]
    control = per["pcb2"]["control"]
    control_hits = int((control["score"] > np.array(gates)[control["mask"]]).sum())
    assert report["gate"]["masked"]["per_category"]["pcb2"]["control"] == [control_hits, 30]
    pooled = report["gate"]["masked"]["control"]
    assert pooled["n"] == 60 and pooled["rate"] == pooled["gated"] / 60
    assert report["reading"]["control_rate"] == pooled["rate"]
    assert len(report["defects"]) == 12 and report["defects"][0]["category"] == "pcb2"
    assert set(report["z_mean"]["pcb2"]) == {"masked", "unmasked"}
    assert list(report["z_mean"]["pcb2"]["masked"]) == list(capture.FEATURES)
    json.dumps(capture._plain(report))
    recorded["visa"]["gate"][CLEAN]["per_category"]["pcb2"]["defect"][0] += 1
    bad = capture.masked_tables(per, recorded)
    assert bad["stage9_check"]["ok"] is False and bad["stage9_check"]["mismatched"] == ["pcb2"]
    assert bad["reading"]["overall"].startswith("결론 없음")


# 명령줄 `--masked`: 가짜 캐시와 매니페스트로 끝까지 돌려 리포트를 쓴다. 봉인 test 행의 이미지는 가짜 캐시에
# 없어서, 실행이 test 행을 읽으면 KeyError로 실패한다.
class _FakeCache:
    def __init__(self, images: dict[str, np.ndarray], masks: dict[str, np.ndarray]):
        self._images, self._masks = images, masks

    def images(self, rows) -> np.ndarray:
        return np.stack([self._images[r.image] for r in rows])

    def masks(self, rows) -> np.ndarray:
        return np.stack([self._masks.get(r.image, np.zeros((256, 256), np.uint8)) for r in rows])


def test_masked_main_writes_the_report_from_dev_rows_only(tmp_path, monkeypatch):
    import defect_inspect.cache
    import defect_inspect.splits

    manifest, images, masks = [], {}, {}
    for seed, category in enumerate(("pcb2", "pipe_fryum")):
        data = _category(seed + 5)
        for i, image in enumerate(data["pool_images"]):
            name = f"{category}/pool/{i:03d}.JPG"
            manifest.append(
                ManifestRow(name, "", category, "normal", "pool_normal", int(data["pool_folds"][i]), "")
            )
            images[name] = image
        for name, image in zip(data["normal_names"], data["normal_images"], strict=True):
            manifest.append(ManifestRow(f"{category}/{name}", "", category, "normal", "pool_normal", 0, ""))
            images[f"{category}/{name}"] = image
        for name, image in zip(data["defect_names"], data["defect_images"], strict=True):
            row = ManifestRow(
                f"{category}/{name}", f"{category}/{name}.png", category, "anomaly", "dev_defect", -1, "spot"
            )
            manifest.append(row)
            images[row.image], masks[row.image] = image, np.zeros((256, 256), np.uint8)
        manifest.append(ManifestRow(f"{category}/test.JPG", "", category, "normal", "test_normal", -1, ""))
    fake = _FakeCache(images, masks)
    per = {
        c: capture.masked_scores(capture.masked_category(fake, manifest, c)) for c in ("pcb2", "pipe_fryum")
    }
    first = capture.masked_tables(per, None)
    recorded = {"visa": {"gate": {CLEAN: {"per_category": first["gate"]["unmasked"]["per_category"]}}}}
    record_path, out = tmp_path / "capture.json", tmp_path / "capture-masked.json"
    record_path.write_text(json.dumps(capture._plain(recorded)), encoding="utf-8")
    monkeypatch.setattr(defect_inspect.cache, "ImageCache", lambda out_dir, size: fake)
    monkeypatch.setattr(defect_inspect.splits, "read_manifest", lambda path: manifest)
    capture.main(
        ["--masked", "--categories", "pcb2", "pipe_fryum", "--recorded", str(record_path), "--out", str(out)]
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["stage9_check"]["ok"] is True and report["dilation"] == capture.DILATION
    assert report["gate"]["masked"]["defect"]["n"] == 12 and report["gate"]["masked"]["control"]["n"] == 60
    assert report["reading"]["overall"] and "seconds" in report
