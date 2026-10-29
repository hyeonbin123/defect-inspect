# [파일 역할] 6단계 분석 모듈(`defect_inspect.analyze_stage6`)을 검증하는 테스트 묶음이다. 학습 전 CPU 지연
# 관문(gate), dev에서 서빙 모델 고르기(choose, dev_report), 봉인 테스트의 H16·H17 판정(sealed_report), 판정
# 없는 dev 보조 표(supplement_report)를 다룬다.
# [왜 필요한가] 6단계의 모든 선택과 판정은 docs/experiments.md 6단계에 측정 전에 등록됐다: 200ms 이하만 학습,
# 308이 빠지면 294가 대신, CAR은 280(안 되면 252)과 함께, dev AUROC가 가장 높은 모델(동점이면 작은 입력),
# H16은 세 조건(CPU ≤ 200ms, 오검출률 ≤ 6%, DM보다 2.0%p 넘게 낮지 않음)의 점 추정값, H17은 짝지은 부트스트랩
# 구간. 경계값 하나만 틀려도 학습할 모델이나 서비스 교체 결정이 달라진다.
# [설계] 지연 파일과 실행 폴더(npz, run.json)를 합성으로 만들어, 실제 측정값 없이 규칙의 각 분기와 경계(정확히
# 200ms, 0.0001 차이)를 짚는다. 6단계 보충(학습 시드 분산, seeds)은 가짜 모델 파일·학습 기록(train.json)·dev
# 채점 폴더로 시드별 요약, seed 짝 CAR 효과의 t 구간(scipy와 대조), 해석 문장 선택, 등록한 줄이는 순서와 기록
# 대조(멈춤 조건)를 본다.

import hashlib
import json

import numpy as np
import pytest
from scipy import stats as scipy_stats

from defect_inspect import analyze_stage6 as a6
from defect_inspect import compare
from defect_inspect.calibrate import conformal_threshold
from defect_inspect.metrics import pro_histograms

# 합성 범주 두 개.
CATS = ("alpha", "beta")


# 지연 리포트(reports/stage6/latency.json) 모양의 dict를 만든다. 인자 이름 s252처럼 크기를 받아
# 'dms-252-untrained' 항목의 리사이즈 포함 CPU FP32 지연으로 넣는다.
def _latency(**ms):
    """주어진 등록 CPU 지연을 담은 지연 파일 항목 `dms-<size>-untrained`."""
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
    rows[0]["name"], rows[0]["macro_image_auroc"] = "dms-294", 0.951  # AUROC는 같고 입력이 더 크다
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
    assert a6.h17(0.005, 0.005 + rng.normal(0, 0.001, 2000))["verdict"] == "판정 불가"  # 0보다 크지만 작다
    assert a6.h17(0.01, 0.01 + rng.normal(0, 0.02, 2000))["verdict"] == "판정 불가"
    assert a6.h17(-0.03, -0.03 + rng.normal(0, 0.005, 2000))["verdict"] == "기각"


# 실행 폴더 하나를 합성한다. 범주마다 정상 60장 N(0,1), 결함 20장 N(2,1) + shift 점수와 보정 점수
# 40개(cal=False면 비움)를 npz로, protocol·이름·조건을 run.json으로 쓴다. pc=True면 4단계 PatchCore CPU
# 형식(eval_score_full, pool_score_oof)으로 쓰고, with_pixels=True면 AUPRO·픽셀 AUROC·PRO 히스토그램도 넣는다.
# 같은 seed면 같은 점수가 나와 'torch와 ONNX 점수가 같은' 상황을 만들 수 있다.
def _write_run(
    out, name, *, protocol, shift=0.0, seed=0, with_pixels=False, condition="clean", cal=True, pc=False
):
    """공통 형식(`pc`면 4단계 CPU 형식)의 실행: 정상 ~N(0,1), 결함 ~N(2,1)."""
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
    # 등록한 torch 지표에는 범주별 값 옆에 평균 픽셀 AUROC도 들어 있다.
    per_cat = [v["pixel_auroc"] for v in report["torch_per_category"].values()]
    assert report["torch"]["macro_pixel_auroc"] == pytest.approx(np.mean(per_cat))
    assert report["torch"]["macro_pixel_auroc"] == pytest.approx(0.9)
    # 점수와 보정 정상이 같다: torch 임계값으로 ONNX 파이프라인 자신의 비율이 그대로 나온다.
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
    # 정상 점수를 모두 올린다: 교란 없는 임계값을 넘는 정상이 늘어난다.
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


# --- 6단계 보충: 학습 시드 분산 (docs/experiments.md "6단계 보충") ---


# 6단계 학습 기록(train.json)과 같은 모양. 구성 기록 가운데 seed만 seed마다 다르다(관문 파일 경로는 같게
# 둔다).
def _train_record(name, seed):
    options = {"img_size": 280, "encoder_blocks": 10, "attention": "scaled"}
    if name.endswith("-car"):
        options["context_recentering"] = True
    gate = {"file": "outputs/gate-dms-280/gate.json", "sha256": "7b" * 32, "amp": True, "attention": "scaled"}
    config = {
        "name": name,
        "encoder": "vit_small_patch14_reg4_dinov2",
        "img_size": 280,
        "steps": 10000,
        "batch_size": 16,
        "amp": True,
        "seed": seed,
        "attention": "scaled",
        "options": options,
        "optimizer": {"name": "StableAdamW", "lr": 0.002},
        "scheduler": {"name": "WarmCosineScheduler", "total_iters": 10000, "warmup_iters": 100},
        "grad_clip_norm": 0.1,
        "gate": gate,
    }
    return {
        "method": "dinomaly",
        "commit": "abc1234",
        "device": "cuda",
        "config": config,
        "train_images": 4615,
        "train_images_by_category": {"alpha": 2000, "beta": 2615},
        "final_loss": 0.01 + seed / 1000,
        "skipped_steps": 0,
        "peak_vram_mb": 1600.0 + seed,
        "train_s": 870.0,
    }


# seed 하나의 가짜 학습 폴더(model.pt, train.json)와 그 모델의 dev 채점 폴더(npz, run.json). 폴더 이름은
# 등록한 배치를 따른다: seed 0은 <구성>과 <구성>-dev(6단계 폴더), seed s는 <구성>-s<s>와 <구성>-s<s>-dev.
# run.json에는 eval 명령처럼 모델 경로·sha256과 채점 정밀도를 적는다. 같은 score_seed면 정상 점수가 같고 결함
# 점수만 shift 만큼 다르다.
def _seed_run(root, name, seed, *, shift, score_seed):
    model_dir = root / (name if seed == 0 else f"{name}-s{seed}")
    model_dir.mkdir(parents=True)
    model = model_dir / "model.pt"
    model.write_bytes(f"{name} seed {seed}".encode())
    (model_dir / "train.json").write_text(json.dumps(_train_record(name, seed)), encoding="utf-8")
    run_dir = _write_run(
        root / f"{model_dir.name}-dev",
        name,
        protocol="dev",
        seed=score_seed,
        shift=shift,
        with_pixels=True,
        cal=False,
    )
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    meta["commit"] = "def5678"
    sha = hashlib.sha256(model.read_bytes()).hexdigest()
    meta["config"].update(model=str(model), model_sha256=sha, amp=False, score_precision="fp32")
    (run_dir / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    return run_dir


# 등록한 여덟 채점(두 구성 × seed 0~3). 같은 seed의 두 구성은 정상 점수가 같고 CAR 쪽 결함 점수만 0.6 높다.
def _seed_runs(root):
    return {
        (name, s): _seed_run(root, name, s, shift=shift, score_seed=10 + s)
        for s in a6.SEEDS
        for name, shift in (("dms-280-car", 0.6), ("dms-280", 0.0))
    }


# reports/stage6/dev.json 모양의 기록: seed 0 두 채점을 6단계 코드(compare.summarize)로 요약한 행, 선택 차이
# gap을 만드는 dms-308 행, 같은 두 모델의 CAR 차이.
def _dev_record(runs, gap=0.0049):
    rows = []
    for name in a6.SEED_CONFIGS:
        run = compare.load_common(runs[(name, 0)], name)
        summary = compare.summarize(run, compare.Bootstrap(run, n_boot=20))
        rows.append(
            {k: summary[k] for k in ("name", "macro_image_auroc", "macro_aupro", "pooled_fpr_at_tpr95")}
        )
    rows.append({"name": "dms-308", "macro_image_auroc": rows[0]["macro_image_auroc"] - gap})
    diff = rows[0]["macro_image_auroc"] - rows[1]["macro_image_auroc"]
    return {
        "rows": rows,
        "car_effect": {"pair": list(a6.SEED_CONFIGS), "diff": diff, "ci": [diff - 0.01, diff]},
    }


# 채점 폴더(<이름>-dev)에 짝인 학습 폴더(<이름>).
def _model_dir(run_dir):
    return run_dir.with_name(run_dir.name.removesuffix("-dev"))


def _rewrite_json(path, change):
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


# 확인: 부트스트랩 없이 내는 점 요약(평균 이미지 AUROC, AUPRO, 합산 FPR@TPR95, 범주별 AUROC)이 6단계 dev 표를
# 만든 compare.summarize의 같은 값과 같다.
# 왜: seed 0 대조는 이 요약이 reports/stage6/dev.json을 다시 내는지로 기록을 확인한다. 정의가 어긋나면 멀쩡한
# seed 0 기록에서 멈추거나 새 seed가 다른 정의로 요약된다.
def test_point_summary_matches_the_stage6_summary(tmp_path):
    run_dir = _write_run(
        tmp_path / "r", "dms-280", protocol="dev", seed=5, shift=0.3, with_pixels=True, cal=False
    )
    run = compare.load_common(run_dir, "dms-280")
    full = compare.summarize(run, compare.Bootstrap(run, n_boot=20))
    point = a6.point_summary(run)
    for key in ("macro_image_auroc", "macro_aupro", "pooled_fpr_at_tpr95"):
        assert point[key] == full[key]
    assert point["per_category_auroc"] == {c: v["image_auroc"] for c, v in full["per_category"].items()}


# 확인: t 구간(평균 ± t₀.₉₇₅(n − 1) · s / √n)이 scipy의 두 경로(ttest_1samp의 신뢰구간, t.interval)와 같고,
# 표준편차는 n − 1로 나눈 표본 표준편차이며, 값이 하나뿐이면 멈춘다.
# 왜: 등록한 CAR 효과 구간이 이 식이다(자유도 = seed 짝 수 − 1). n으로 나누거나 정규 분위수 1.96을 쓰면 seed
# 4개에서 구간이 실제보다 좁아진다.
def test_t_interval_matches_scipy():
    d = [0.004, 0.011, -0.002, 0.007]
    got = a6.t_interval(d)
    assert got["n"] == 4 and got["df"] == 3 and got["mean"] == pytest.approx(np.mean(d))
    assert got["sd"] == pytest.approx(np.std(d, ddof=1), rel=1e-12)
    assert got["t"] == pytest.approx(scipy_stats.t.ppf(0.975, 3), rel=1e-12)
    ci = scipy_stats.ttest_1samp(d, 0.0).confidence_interval(0.95)
    assert got["ci"] == pytest.approx([ci.low, ci.high], rel=1e-9)
    lo, hi = scipy_stats.t.interval(0.95, 3, loc=np.mean(d), scale=scipy_stats.sem(d))
    assert got["ci"] == pytest.approx([lo, hi], rel=1e-9)
    assert a6.t_interval(d[:3])["df"] == 2
    with pytest.raises(ValueError, match="two"):
        a6.t_interval([0.01])


# 확인: 여덟 채점으로 구성마다 seed 4개의 AUROC 평균·표본 표준편차·최솟값·최댓값을, seed 짝 CAR 효과 d_s와 그
# t(자유도 3) 구간을 내고, seed 0 대조(dev.json의 두 행)와 선택 차이(dms-280-car − dms-308)를 싣는다. 채점마다
# 학습 기록(손실·VRAM)과 모델 sha256이 남는다. 관문 파일 경로만 다른 학습 기록은 같은 설정으로 본다.
# 왜: 6단계 보충의 지표 정의 그대로다. 짝 차이는 같은 seed의 두 구성끼리 뺀다(구성 평균끼리 빼면 점 추정은
# 같아도 구간이 seed 짝을 쓰지 않는다). 관문 파일은 sha256으로 대조하고, 경로 문자열은 셸에 따라 구분자가
# 달라진다.
def test_seeds_report_summarizes_the_seed_spread_and_the_paired_car_effect(tmp_path):
    runs = _seed_runs(tmp_path)
    dev = _dev_record(runs)
    _rewrite_json(
        _model_dir(runs[("dms-280", 1)]) / "train.json",
        lambda d: d["config"]["gate"].update(file="outputs\\gate-dms-280\\gate.json"),
    )
    report = a6.seeds_report(list(runs.values()), dev)
    auroc = {key: compare.macro_auroc(compare.load_common(path, key[0])) for key, path in runs.items()}
    for name in a6.SEED_CONFIGS:
        values = [auroc[(name, s)] for s in a6.SEEDS]
        row = report["per_config"][name]
        assert row["seeds"] == [0, 1, 2, 3] and row["auroc"] == pytest.approx(values)
        assert row["mean"] == pytest.approx(np.mean(values))
        assert row["sd"] == pytest.approx(np.std(values, ddof=1))
        assert row["min"] == pytest.approx(min(values)) and row["max"] == pytest.approx(max(values))
    d = [auroc[("dms-280-car", s)] - auroc[("dms-280", s)] for s in a6.SEEDS]
    car = report["car_effect"]
    assert car["seeds"] == [0, 1, 2, 3] and car["d"] == pytest.approx(d) and car["df"] == 3
    ci = scipy_stats.ttest_1samp(d, 0.0).confidence_interval(0.95)
    assert car["ci"] == pytest.approx([ci.low, ci.high])
    assert car["d"][0] == pytest.approx(dev["car_effect"]["diff"])  # seed 0의 d = 6단계 기록
    assert all(check["equal"] for check in report["seed0_check"].values())
    sd_car = report["per_config"]["dms-280-car"]["sd"]
    assert report["selection"]["gap"] == pytest.approx(0.0049)
    assert report["selection"]["ratio"] == pytest.approx(0.0049 / sd_car)
    assert not report["reduced"] and report["dropped"] == []
    first = report["runs"][0]
    assert (first["config"], first["seed"]) == ("dms-280-car", 0) and first["train"]["peak_vram_mb"] == 1600.0
    assert len({r["model_sha256"] for r in report["runs"]}) == 8
    assert set(report["interpretation"]) == {"car", "selection"}


# 확인: 등록한 줄이는 순서의 두 조합(dms-280 seed 3을 뺀 7개, 두 seed 3을 뺀 6개)은 받아서 줄인 사실을 적고,
# CAR 효과는 두 구성이 모두 있는 seed만 쓴다(자유도 = 짝 수 − 1). 그 밖의 조합(dms-280-car seed 1을 뺌)과 같은
# 구성·seed가 두 번 나오는 묶음에서는 멈춘다.
# 왜: 규칙은 줄이는 순서를 측정 전에 하나로 정했다. 결과를 보고 고른 seed 묶음이 들어오면 시드 분산이
# 골라진다.
def test_seeds_report_takes_only_the_registered_reductions(tmp_path):
    runs = _seed_runs(tmp_path)
    dev = _dev_record(runs)
    report = a6.seeds_report([p for k, p in runs.items() if k != ("dms-280", 3)], dev)
    assert report["reduced"] and report["dropped"] == [["dms-280", 3]]
    assert "reduced" in report["interpretation"]
    assert report["per_config"]["dms-280-car"]["seeds"] == [0, 1, 2, 3]
    assert report["per_config"]["dms-280"]["seeds"] == [0, 1, 2]
    assert report["car_effect"]["seeds"] == [0, 1, 2] and report["car_effect"]["df"] == 2
    six = a6.seeds_report([p for k, p in runs.items() if k[1] != 3], dev)
    assert six["dropped"] == [["dms-280-car", 3], ["dms-280", 3]]
    with pytest.raises(ValueError, match="registered"):
        a6.seeds_report([p for k, p in runs.items() if k != ("dms-280-car", 1)], dev)
    with pytest.raises(ValueError, match="twice"):
        a6.seeds_report([*runs.values(), runs[("dms-280", 2)]], dev)


# 기록 대조를 깨는 방법들. 정상 묶음(runs)과 6단계 기록(dev)을 고쳐, 분석이 멈춰야 하는 상황을 하나씩 만든다.
def _break_seed0(runs, dev):  # seed 0 채점이 6단계 기록을 다시 내지 못한다
    dev["rows"][0]["macro_image_auroc"] += 1e-9


def _break_model(runs, dev):  # 채점 뒤 모델 파일이 바뀌었다(다시 학습)
    (_model_dir(runs[("dms-280", 2)]) / "model.pt").write_bytes(b"retrained")


def _break_optimizer(runs, dev):  # 학습률이 seed 0과 다르다
    path = _model_dir(runs[("dms-280-car", 1)]) / "train.json"
    _rewrite_json(path, lambda d: d["config"]["optimizer"].update(lr=0.001))


def _break_gate(runs, dev):  # 다른 관문 파일(sha256)로 정밀도를 정했다
    path = _model_dir(runs[("dms-280", 3)]) / "train.json"
    _rewrite_json(path, lambda d: d["config"]["gate"].update(sha256="00" * 32))


def _break_steps(runs, dev):  # seed 0부터 6단계 설정(10,000스텝)이 아니다
    _rewrite_json(_model_dir(runs[("dms-280", 0)]) / "train.json", lambda d: d["config"].update(steps=5000))


def _break_protocol(runs, dev):  # 봉인 테스트 채점이 섞였다
    _rewrite_json(runs[("dms-280-car", 2)] / "run.json", lambda d: d.update(protocol="test"))


def _break_precision(runs, dev):  # fp16으로 채점했다(--no-amp 빠짐)
    _rewrite_json(runs[("dms-280", 1)] / "run.json", lambda d: d["config"].update(score_precision="fp16"))


def _break_name(runs, dev):  # 모델 옆 학습 기록이 다른 구성의 것이다
    path = _model_dir(runs[("dms-280-car", 3)]) / "train.json"
    _rewrite_json(path, lambda d: d["config"].update(name="dms-252"))


def _break_images(runs, dev):  # 한 채점이 다른 dev 이미지를 채점했다
    path = runs[("dms-280-car", 2)] / "alpha.npz"
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["eval_images"] = np.array([f"other/{i}" for i in range(len(arrays["eval_images"]))])
    np.savez(path, **arrays)


def _break_same_model(runs, dev):  # 두 seed 폴더가 같은 모델 파일을 담았다
    model = _model_dir(runs[("dms-280-car", 3)]) / "model.pt"
    model.write_bytes((_model_dir(runs[("dms-280-car", 2)]) / "model.pt").read_bytes())
    sha = hashlib.sha256(model.read_bytes()).hexdigest()
    _rewrite_json(runs[("dms-280-car", 3)] / "run.json", lambda d: d["config"].update(model_sha256=sha))


def _break_unfinished(runs, dev):  # 채점이 끝나지 않았다(run.json 없음)
    (runs[("dms-280", 3)] / "run.json").unlink()


# 확인: 기록 대조(규칙의 1~4) 가운데 하나라도 어긋나면 결과를 내지 않고 무엇이 어긋났는지 적은 ValueError로
# 멈춘다.
# 왜: seed 분산은 '같은 설정에서 seed만 바꾼 학습'끼리의 흔들림이어야 한다. 설정이 다른 학습, 바뀐 모델, fp16
# 채점, 다른 이미지가 섞이면 그 차이가 시드 분산으로 읽힌다.
@pytest.mark.parametrize(
    ("breaker", "message"),
    [
        (_break_seed0, "dev.json"),
        (_break_model, "sha256"),
        (_break_optimizer, "seed 0"),
        (_break_gate, "seed 0"),
        (_break_steps, "10000"),
        (_break_protocol, "not a dev"),
        (_break_precision, "fp32"),
        (_break_name, "dms-252"),
        (_break_images, "different images"),
        (_break_same_model, "same model"),
        (_break_unfinished, "run.json"),
    ],
)
def test_seeds_report_stops_when_a_record_check_fails(tmp_path, breaker, message):
    runs = _seed_runs(tmp_path)
    dev = _dev_record(runs)
    breaker(runs, dev)
    with pytest.raises(ValueError, match=message):
        a6.seeds_report(list(runs.values()), dev)


# 확인: 해석 규칙 (가)의 문장 선택. d의 t 구간 하한 > 0이면 '넘는다', 상한 < 0이면 '반대 방향으로 낮다', 0을
# 포함하면(끝이 정확히 0인 경우도) '구별되지 않는다'. 문장에는 seed 수·평균·자유도·구간이 들어간다.
# 왜: 규칙은 결과마다 README에 적을 문장을 측정 전에 정했다. 경계(0)에서 부등호 방향이 바뀌면 다른 문장이
# 적힌다.
@pytest.mark.parametrize(
    ("ci", "expected", "phrase"),
    [
        ([0.001, 0.015], "exceeds", "시드 잡음을 넘는다"),
        ([-0.02, -0.001], "below", "반대 방향"),
        ([-0.004, 0.012], "indistinguishable", "구별되지 않는다"),
        ([0.0, 0.01], "indistinguishable", "구별되지 않는다"),
    ],
)
def test_interpretation_picks_the_registered_car_sentence(ci, expected, phrase):
    car = {"n": 4, "df": 3, "mean": (ci[0] + ci[1]) / 2, "ci": ci}
    result = a6.interpret_seeds(car, sd_car=0.003, gap=0.0049, d0=0.008)["car"]
    assert result["result"] == expected and phrase in result["sentence"]
    assert "seed 4개" in result["sentence"] and "자유도 3" in result["sentence"]
    assert f"{ci[1] * 100:+.2f}" in result["sentence"]


# 확인: 해석 규칙 (나)의 문장 선택. dms-280-car seed 표준편차가 선택 차이 이상이면(같을 때 포함) '시드 잡음
# 수준', 작으면 '표준편차의 x배'(x = 차이 / 표준편차)이고, 두 문장 모두 dms-308을 반복하지 않은 근사임을
# 적는다. 표준편차가 0이면 멈춘다.
# 왜: '이상'과 '작으면'의 경계, x의 정의가 규칙과 같아야 한다. 표준편차 0은 서로 다른 학습에서 나올 수 없는
# 값이다.
def test_interpretation_picks_the_registered_selection_sentence():
    car = {"n": 4, "df": 3, "mean": 0.005, "ci": [-0.001, 0.011]}
    same = a6.interpret_seeds(car, sd_car=0.0049, gap=0.0049, d0=0.008)["selection"]
    assert same["result"] == "noise" and "시드 잡음 수준" in same["sentence"] and "근사" in same["sentence"]
    small = a6.interpret_seeds(car, sd_car=0.002, gap=0.0049, d0=0.008)["selection"]
    assert small["result"] == "multiple" and small["ratio"] == pytest.approx(0.0049 / 0.002)
    assert f"{0.0049 / 0.002:.1f}배" in small["sentence"] and "근사" in small["sentence"]
    with pytest.raises(ValueError, match="zero"):
        a6.interpret_seeds(car, sd_car=0.0, gap=0.0049, d0=0.008)


# 확인: 측정 전 점검(check_seed0_runs)은 seed 0 두 채점만 기록 대조(채점 형식, 모델 sha256, 학습 기록의
# 구성·seed 0·6단계 설정)와 dev.json 재현으로 확인해 기록값과 다시 낸 값을 돌려주고, 기록과 다르거나 seed 0
# 채점이 아니면 멈춘다.
# 왜: GPU 학습(약 1.6시간)을 시작하기 전에, seed 0 쪽 문제로 마지막 분석이 멈추는 일을 막는다.
def test_check_seed0_runs_before_training(tmp_path):
    runs = _seed_runs(tmp_path)
    dev = _dev_record(runs)
    dev_path = tmp_path / "dev.json"
    dev_path.write_text(json.dumps(dev), encoding="utf-8")
    seed0 = [runs[(name, 0)] for name in a6.SEED_CONFIGS]
    checked = a6.check_seed0_runs(seed0, dev_path)
    assert set(checked) == set(a6.SEED_CONFIGS) and all(c["equal"] for c in checked.values())
    with pytest.raises(ValueError, match="seed 0"):
        a6.check_seed0_runs([runs[("dms-280-car", 1)], runs[("dms-280", 0)]], dev_path)
    dev["rows"][1]["pooled_fpr_at_tpr95"] += 0.01
    dev_path.write_text(json.dumps(dev), encoding="utf-8")
    with pytest.raises(ValueError, match="dev.json"):
        a6.check_seed0_runs(seed0, dev_path)


# 확인: 새 seed 채점 하나의 점검(check_seed_run)은 기록 대조와 seed 0(기본 <구성>-dev)과의 학습 설정·이미지
# 대조를 하고 학습 기록 요약만 돌려준다(지표 없음). seed 0이 다른 구성이거나, seed 0 모델을 다시 채점한
# 것이거나, 설정이 다르면 멈춘다.
# 왜: 측정 스크립트가 학습마다 이 점검을 불러, 설정이 어긋난 학습을 마지막 분석(약 1.6시간 뒤)에서야 알게 되는
# 일을 막는다. 진행 중에 AUROC를 보며 진행하지 않도록 지표는 돌려주지 않는다.
def test_check_seed_run_compares_one_new_seed_with_seed0(tmp_path, monkeypatch):
    runs = _seed_runs(tmp_path)
    got = a6.check_seed_run(runs[("dms-280", 2)], runs[("dms-280", 0)])
    assert (got["config"], got["seed"], got["peak_vram_mb"]) == ("dms-280", 2, 1602.0)
    assert "macro_image_auroc" not in got
    monkeypatch.setattr(a6.paths, "OUTPUTS", tmp_path)
    assert a6.check_seed_run(runs[("dms-280-car", 3)])["seed"] == 3
    with pytest.raises(ValueError, match="seed 0 scoring"):
        a6.check_seed_run(runs[("dms-280", 2)], runs[("dms-280-car", 0)])
    with pytest.raises(ValueError, match="same sha256"):
        a6.check_seed_run(runs[("dms-280", 0)], runs[("dms-280", 0)])
    _break_optimizer(runs, None)
    with pytest.raises(ValueError, match="differs from seed 0"):
        a6.check_seed_run(runs[("dms-280-car", 1)])


# 확인: 기본 채점 폴더 배치(seed_run_dirs: 6단계 폴더 두 개, 이어서 seed마다 <구성>-s<s>-dev 두 개)와, seeds
# 명령이 인자 없이(기본 폴더) 또는 --runs로 준 채점에서 리포트를 만들어 --out에 쓰는 것.
# 왜: 측정 스크립트가 줄이지 않았을 때 인자 없이 부르는 경로이고, 줄였으면 --runs로 남은 폴더만 준다.
def test_seeds_command_writes_the_report(tmp_path, monkeypatch):
    expected = [tmp_path / "dms-280-car-dev", tmp_path / "dms-280-dev"]
    expected += [tmp_path / f"{c}-s{s}-dev" for s in (1, 2, 3) for c in a6.SEED_CONFIGS]
    assert a6.seed_run_dirs(tmp_path) == expected
    runs = _seed_runs(tmp_path)
    dev_path = tmp_path / "dev.json"
    dev_path.write_text(json.dumps(_dev_record(runs)), encoding="utf-8")
    out = tmp_path / "seeds.json"
    a6.main(["seeds", "--runs", *map(str, runs.values()), "--dev-report", str(dev_path), "--out", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["car_effect"]["df"] == 3 and report["interpretation"]["car"]["sentence"]
    monkeypatch.setattr(a6.paths, "OUTPUTS", tmp_path)
    out.unlink()
    a6.main(["seeds", "--dev-report", str(dev_path), "--out", str(out)])
    assert json.loads(out.read_text(encoding="utf-8")) == report
