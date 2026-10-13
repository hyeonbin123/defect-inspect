# [파일 역할] 6단계 분석 모듈(`defect_inspect.analyze_stage6`)을 검증하는 테스트 묶음이다. 학습 전 CPU 지연
# 관문(gate), dev에서 서빙 모델 고르기(choose, dev_report), 봉인 테스트의 H16·H17 판정(sealed_report), 판정
# 없는 dev 보조 표(supplement_report)를 다룬다.
# [왜 필요한가] 6단계의 모든 선택과 판정은 docs/experiments.md 6단계에 측정 전에 등록됐다: 200ms 이하만 학습,
# 308이 빠지면 294가 대신, CAR은 280(안 되면 252)과 함께, dev AUROC가 가장 높은 모델(동점이면 작은 입력),
# H16은 세 조건(CPU ≤ 200ms, 오검출률 ≤ 6%, DM보다 2.0%p 넘게 낮지 않음)의 점 추정값, H17은 짝지은 부트스트랩
# 구간. 경계값 하나만 틀려도 학습할 모델이나 서비스 교체 결정이 달라진다.
# [설계] 지연 파일과 실행 폴더(npz, run.json)를 합성으로 만들어, 실제 측정값 없이 규칙의 각 분기와 경계(정확히
# 200ms, 0.0001 차이)를 짚는다.

import json

import numpy as np
import pytest

from defect_inspect import analyze_stage6 as a6
from defect_inspect.calibrate import conformal_threshold
from defect_inspect.metrics import pro_histograms

# 합성 범주 두 개.
CATS = ("alpha", "beta")


# 지연 리포트(reports/stage6/latency.json) 모양의 dict를 만든다. 인자 이름 s252처럼 크기를 받아
# 'dms-252-untrained' 항목의 리사이즈 포함 CPU FP32 지연으로 넣는다.
def _latency(**ms):
    """Latency file entries `dms-<size>-untrained` with the given registered CPU latency."""
    return {
        f"dms-{size}-untrained": {"cpu_fp32": {"total_with_resize_ms": value}}
        for size, value in ((int(k[1:]), v) for k, v in ms.items())
    }


# 확인: 관문 1의 학습 대상 규칙을 다섯 경우로 본다.
# - 모두 200ms 이하: 252·280·308과 dms-280-car
# - 308만 넘고 294가 통과: 294가 308 자리에 들어간다
# - 294·308 모두 넘음: 252·280과 280-car
# - 280이 200.5ms로 넘음: 252와 CAR을 252로 옮긴 dms-252-car
# - 280이 정확히 200.0ms: '이하'이므로 통과한다
# 왜: 경계와 대체 규칙(docs/experiments.md 6단계 관문 1 판정)이 그대로 코드에 있는지 확인한다. 이 결과가
# 학습할 모델(GPU 약 15분씩)을 정한다.
@pytest.mark.parametrize(
    ("ms", "train"),
    [
        (
            {"s252": 120, "s280": 150, "s294": 170, "s308": 190},
            ["dms-252", "dms-280", "dms-308", "dms-280-car"],
        ),
        (
            {"s252": 120, "s280": 150, "s294": 195, "s308": 205},
            ["dms-252", "dms-280", "dms-294", "dms-280-car"],
        ),
        ({"s252": 120, "s280": 150, "s294": 201, "s308": 230}, ["dms-252", "dms-280", "dms-280-car"]),
        ({"s252": 180, "s280": 200.5, "s294": 220, "s308": 240}, ["dms-252", "dms-252-car"]),
        ({"s252": 200, "s280": 200.0, "s294": 220, "s308": 240}, ["dms-252", "dms-280", "dms-280-car"]),
    ],
)
def test_gate_picks_the_registered_training_set(ms, train):
    result = a6.gate(_latency(**ms))
    assert result["train"] == train and not result["blocked"]
    assert result["cpu_ms"]["dms-252"] == ms["s252"]


# 확인: 252까지 모두 200ms를 넘으면 학습할 것이 없어 blocked가 되고(규칙: 252도 통과하지 못하면 이 단계를
# 멈춘다), 네 크기 가운데 하나라도 지연 기록이 없으면 그 크기(294)를 적은 ValueError로 멈춘다. 재지 않은
# 크기를 '통과 못함'으로 조용히 처리하지 않는다.
def test_gate_blocks_when_nothing_fits_and_needs_every_size():
    result = a6.gate(_latency(s252=201, s280=230, s294=240, s308=250))
    assert result["blocked"] and result["train"] == [] and result["passed"] == []
    with pytest.raises(ValueError, match="294"):
        a6.gate(_latency(s252=120, s280=150, s308=190))


# 확인: 서빙 모델 선택은 CPU 200ms 이하인 후보 가운데 dev 평균 이미지 AUROC가 가장 높은 것이고, 동점이면
# 입력이 작은 쪽이다. 예산을 넘은 dms-308은 AUROC가 높아도 후보에서 빠지고, 예산 안 후보가 없으면 None이다.
# 왜: 6단계 dev 선택 규칙 그대로다. 실제로 dev에서 dms-280-car(96.98)가 뽑혔다(README 6단계).
def test_choose_takes_the_best_dev_auroc_within_budget_and_the_smaller_input_on_ties():
    rows = [
        {"name": "dms-252", "cpu_ms": 120.0, "macro_image_auroc": 0.93},
        {"name": "dms-280", "cpu_ms": 150.0, "macro_image_auroc": 0.95},
        {"name": "dms-280-car", "cpu_ms": 150.0, "macro_image_auroc": 0.95},
        {"name": "dms-308", "cpu_ms": 201.0, "macro_image_auroc": 0.99},
    ]
    pick = a6.choose(rows)
    assert pick["name"] == "dms-280" and pick["candidates"] == ["dms-252", "dms-280", "dms-280-car"]
    rows[2]["macro_image_auroc"] = 0.951
    assert a6.choose(rows)["name"] == "dms-280-car"
    rows[0]["name"], rows[0]["macro_image_auroc"] = "dms-294", 0.951  # same AUROC, larger input
    assert a6.choose(rows)["name"] == "dms-280-car"
    assert a6.choose([{"name": "dms-308", "cpu_ms": 230.0, "macro_image_auroc": 0.99}]) is None


# 확인: H16은 세 조건 모두의 점 추정값으로 판정한다. 200.1ms, 오검출률 6.1%, AUROC가 DM 96.79 − 2.0%p =
# 94.79보다 0.01%p 낮은 경우는 세 조건이 모두 실패로 적히고 '기각'이며, 지연을 재지 않았으면(None) (가) 조건이
# 실패다. H17은 구간 하한이 0보다 크고 차이가 1.0%p 이상이면 '지지', 하한이 0보다 크지만 차이가 1.0%p보다
# 작거나 구간이 0을 포함하면 '판정 불가', 상한이 0보다 작으면 '기각'이다.
# 왜: H17은 공통 규칙의 '차이 작음' 대신 지지/기각/판정 불가 세 갈래로 등록됐다. 경계 근처 값으로 부등호
# 방향(이하/미만)을 확인한다.
def test_h16_and_h17_rules():
    good = {"macro_image_auroc": 0.95, "fixed_threshold": {"fpr": 0.05}}
    assert a6.h16(good, 199.0, 0.9679)["verdict"] == "지지"
    result = a6.h16({"macro_image_auroc": 0.9478, "fixed_threshold": {"fpr": 0.061}}, 200.1, 0.9679)
    assert result["verdict"] == "기각" and result["failed"] == ["a_cpu_ms", "b_fpr", "c_auroc_vs_dm"]
    assert a6.h16(good, None, 0.9679)["failed"] == ["a_cpu_ms"]
    rng = np.random.default_rng(0)
    assert a6.h17(0.05, 0.05 + rng.normal(0, 0.005, 2000))["verdict"] == "지지"
    assert a6.h17(0.005, 0.005 + rng.normal(0, 0.001, 2000))["verdict"] == "판정 불가"  # above 0, too small
    assert a6.h17(0.01, 0.01 + rng.normal(0, 0.02, 2000))["verdict"] == "판정 불가"
    assert a6.h17(-0.03, -0.03 + rng.normal(0, 0.005, 2000))["verdict"] == "기각"


# 실행 폴더 하나를 합성한다. 범주마다 정상 60장 N(0,1), 결함 20장 N(2,1) + shift 점수와 보정 점수
# 40개(cal=False면 비움)를 npz로, protocol·이름·조건을 run.json으로 쓴다. pc=True면 4단계 PatchCore CPU
# 형식(eval_score_full, pool_score_oof)으로 쓰고, with_pixels=True면 AUPRO·픽셀 AUROC·PRO 히스토그램도 넣는다.
# 같은 seed면 같은 점수가 나와 'torch와 ONNX 점수가 같은' 상황을 만들 수 있다.
def _write_run(
    out, name, *, protocol, shift=0.0, seed=0, with_pixels=False, condition="clean", cal=True, pc=False
):
    """A run in the common format (or the stage 4 CPU format with `pc`): normals ~N(0,1), defects ~N(2,1)."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for c in CATS:
        neg = rng.normal(0, 1, 60).astype(np.float32)
        pos = (rng.normal(2, 1, 20) + shift).astype(np.float32)
        images = np.array([f"{c}/n{i}" for i in range(60)] + [f"{c}/d{i}" for i in range(20)])
        labels = np.array([0] * 60 + [1] * 20, dtype=np.int8)
        score = np.concatenate([neg, pos])
        cal_score = rng.normal(0, 1, 40).astype(np.float32) if cal else np.empty(0, np.float32)
        arrays = {"eval_images": images, "eval_labels": labels}
        if pc:
            arrays.update(eval_score_full=score, pool_score_oof=cal_score)
        else:
            arrays.update(eval_score=score, cal_score=cal_score)
        if with_pixels:
            maps = np.random.default_rng(99).random((80, 8, 8)).astype(np.float32)
            masks = np.zeros((80, 8, 8), dtype=bool)
            masks[60:, 2:5, 2:5] = True
            maps[60:, 2:5, 2:5] += 0.5
            hist = pro_histograms(maps, masks, bins=50)
            arrays.update(
                aupro=np.float64(0.8),
                pixel_auroc=np.float64(0.9),
                pro_edges=hist.edges,
                pro_normal=hist.normal,
                pro_components=hist.components,
                pro_component_image=hist.component_image,
            )
        np.savez(out / f"{c}.npz", **arrays)
    (out / "run.json").write_text(
        json.dumps({"protocol": protocol, "config": {"name": name, "condition": condition}}), encoding="utf-8"
    )
    return out


# 확인: dev 표가 후보마다 AUROC·지연을 담고, 예산 안에서 AUROC가 가장 높은 dms-280-car를 고르며, 지연 파일로
# 다시 계산한 관문 결과(308이 230ms라 294가 대신)를 함께 싣는다. CAR 모델과 같은 크기의 CAR 없는 모델이 둘 다
# 있으면 짝지은 차이와 구간(car_effect)을 판정 없이 적는다.
# 왜: 6단계 결과의 'CAR을 켜면 dev AUROC +0.8%p [+0.1, +1.6]'(판정하지 않는 보조 수치)이 이 키에서 나온다.
def test_dev_report_tables_the_candidates_and_the_car_pair(tmp_path):
    runs = [
        _write_run(tmp_path / "a", "dms-252", protocol="dev", seed=1, with_pixels=True, cal=False),
        _write_run(tmp_path / "b", "dms-280", protocol="dev", seed=1, shift=0.5, with_pixels=True, cal=False),
        _write_run(
            tmp_path / "c", "dms-280-car", protocol="dev", seed=1, shift=1.5, with_pixels=True, cal=False
        ),
    ]
    latency = _latency(s252=120, s280=150, s294=170, s308=230)
    latency["dms-280-car-untrained"] = {"cpu_fp32": {"total_with_resize_ms": 151.0}}
    report = a6.dev_report(runs, latency)
    assert [r["name"] for r in report["rows"]] == ["dms-252", "dms-280", "dms-280-car"]
    assert report["rows"][1]["macro_image_auroc"] > report["rows"][0]["macro_image_auroc"]
    assert report["pick"]["name"] == "dms-280-car" and report["rows"][0]["cpu_ms"] == 120
    assert report["gate"]["train"] == ["dms-252", "dms-280", "dms-294", "dms-280-car"]
    car = report["car_effect"]
    assert car["pair"] == ["dms-280-car", "dms-280"] and car["diff"] > 0 and len(car["ci"]) == 2


# 확인: 봉인 테스트 리포트가 ONNX FP32 점수로 H16·H17을 판정하고, 서빙 PatchCore(4단계 CPU 형식)와 비교하며,
# torch와 ONNX 점수의 상대 차이(같은 seed라 0)를 적는다. torch 지표에 12개 범주 평균 픽셀
# AUROC(macro_pixel_auroc)가 들어 있고, torch 임계값을 ONNX 점수에 그대로 쓴 오검출률이 ONNX 자신의 임계값과
# 같다(점수와 보정 정상이 같으므로).
# 왜: 판정은 서빙과 같은 ONNX CPU 점수로 한다는 규칙(6단계 봉인 테스트 2.)을 지키는지 본다. 평균 픽셀 AUROC
# 단언은 2026-10-04 검토에서 빠진 것이 발견돼 추가됐다(docs/experiments.md 6단계 보충).
def test_sealed_report_judges_h16_and_h17(tmp_path):
    torch_dir = _write_run(tmp_path / "t", "dms-280", protocol="test", seed=2, shift=2.0, with_pixels=True)
    onnx_dir = _write_run(tmp_path / "o", "dms-280", protocol="test", seed=2, shift=2.0)
    pc_dir = _write_run(tmp_path / "p", "x", protocol="test", seed=3, pc=True)
    report = a6.sealed_report(
        torch_dir=torch_dir, onnx_dir=onnx_dir, patchcore_dir=pc_dir, cpu_ms=150.0, dm_auroc=0.9
    )
    assert report["name"] == "dms-280" and report["H17"]["verdict"] == "지지"
    assert report["onnx_fp32"]["macro_image_auroc"] > report["serving_patchcore_auroc"]
    assert report["score_rel_diff_onnx_vs_torch"]["max"] == 0.0
    # The registered torch metrics include the mean pixel AUROC next to the per-category values.
    per_cat = [v["pixel_auroc"] for v in report["torch_per_category"].values()]
    assert report["torch"]["macro_pixel_auroc"] == pytest.approx(np.mean(per_cat))
    assert report["torch"]["macro_pixel_auroc"] == pytest.approx(0.9)
    # The same scores and calibration normals: the torch thresholds give the ONNX pipeline's own rates.
    own = report["onnx_fp32"]["fixed_threshold"]
    assert report["onnx_at_torch_thresholds"]["fpr"] == pytest.approx(own["fpr"])
    h16 = report["H16"]
    assert h16["conditions"]["a_cpu_ms"]["met"] and h16["conditions"]["c_auroc_vs_dm"]["met"]
    assert h16["conditions"]["b_fpr"]["met"] == (own["fpr"] <= 0.06)
    with np.load(onnx_dir / "alpha.npz") as z:
        thr = conformal_threshold(z["cal_score"], 0.05).value
        assert z["eval_score"][:60][z["eval_score"][:60] > thr].size <= 60


# 확인: dev 보조 표는 교란 없는 실행의 겹 0 정상 점수로 임계값을 정하고, 교란 실행(shift-1)에도 같은 임계값을
# 쓴다. 교란 쪽 정상 점수를 모두 1.5 올리면 오검출률이 30%를 넘는다. 교란 없는 칸은 dev에서 같은 정상으로
# 임계값을 정하므로 정의상 5% 이하다. 동적 INT8 실행은 FP32와의 AUROC 차이로 적는다.
# 왜: 6단계 '판정 없이 함께 적는 것'의 dev 교란 규칙(임계값은 교란 없는 겹 0 정상 점수)을 따르는지 본다. 교란
# 실행마다 임계값을 다시 잡으면 교란의 영향이 지워진다.
def test_supplement_report_uses_the_clean_normals_for_thresholds(tmp_path):
    clean = _write_run(tmp_path / "clean", "dms-280", protocol="dev", seed=4, cal=False)
    shifted = _write_run(
        tmp_path / "shift", "dms-280", protocol="dev", seed=4, cal=False, condition="shift-1"
    )
    # Move every normal score up: more normals cross the clean thresholds.
    for c in CATS:
        with np.load(shifted / f"{c}.npz") as z:
            arrays = {k: z[k] for k in z.files}
        arrays["eval_score"] = arrays["eval_score"] + np.where(arrays["eval_labels"] == 0, 1.5, 0).astype(
            np.float32
        )
        np.savez(shifted / f"{c}.npz", **arrays)
    int8 = _write_run(tmp_path / "int8", "dms-280", protocol="dev", seed=4, cal=False, shift=-0.2)
    report = a6.supplement_report(clean, [shifted], int8)
    rows = {r["condition"]: r for r in report["rows"]}
    assert rows["clean"]["fpr"] <= 0.05 and rows["shift-1"]["fpr"] > 0.3
    assert report["int8_dynamic"]["diff"] < 0
