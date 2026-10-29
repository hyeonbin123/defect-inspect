"""6단계: CPU 지연 관문, dev에서의 DMS 모델 선택, 봉인 테스트의 H16·H17 판정, 학습 시드 분산(보충).

docs/experiments.md(6단계)에 등록한 규칙:

- 관문(gate): 학습 전 모델의 CPU 지연(리사이즈 + 추론, onnxruntime FP32)이 200ms 이하인 입력 크기만
  학습 후보로 남는다. 학습하는 크기는 252/280/308 가운데 통과한 것이고, 308이 실패하고 294가 통과하면
  294가 그 자리를 대신한다. CAR 모델은 280으로, 280이 실패하면 252로 학습한다.
- 선택(pick): 학습한 모델(모두 예산 안) 가운데 dev 평균 이미지 AUROC(torch, fp32 채점)가 가장 높은 것.
  같으면 입력이 작은 쪽.
- H16(H15와 같은 형식): (가) 학습한 선택 모델의 CPU 지연 ≤ 200ms, (나) ONNX FP32 파이프라인 자신의
  홀드아웃 임계값(alpha 5%)에서 합산 오검출률 ≤ 6%, (다) 평균 이미지 AUROC가 DM(2단계, 96.79)보다 2.0%p
  넘게 낮지 않음. 셋 다 만족하면 지지.
- H17: 선택 모델의 ONNX FP32 AUROC − 서빙 PatchCore(4단계, ONNX FP32)의 AUROC, 짝지은 층화 부트스트랩.
  구간이 0보다 위에 있고(0을 포함하지 않고) 차이가 1.0%p 이상이면 지지, 구간이 0보다 아래면 기각, 그 밖에는
  판정 불가.

docs/experiments.md("6단계 보충: 학습 시드 분산", 2026-10-30 측정 전)에 등록한 규칙(판정 없음, dev만):

- dms-280-car와 dms-280을 seed 0~3으로 학습한 dev 채점(seed 0은 6단계 것)에서 구성마다 평균 이미지 AUROC의
  seed 평균과 표본 표준편차, seed 짝 CAR 효과(켬 − 끔)의 t 구간(자유도 = 짝 수 − 1)을 낸다.
- 해석 (가): 구간 하한 > 0이면 'CAR 효과는 시드 잡음을 넘는다', 상한 < 0이면 '반대 방향으로 낮다', 그 밖에는
  '구별되지 않는다'. (나): dms-280-car seed 표준편차가 6단계 선택 차이(dms-280-car − dms-308, 0.4926%p)
  이상이면 '선택 차이는 시드 잡음 수준', 작으면 '선택 차이는 표준편차의 x배'.
"""

# [파일 역할] 6단계(CPU 서빙용 Dinomaly ViT-S) 분석 모듈이다. 하위 명령 다섯 개 가운데 앞의 넷이 6단계의
# 순서와 같고, 마지막 하나는 6단계 보충이다.
# - gate: 학습 전 모델의 CPU 지연으로 학습할 크기를 정한다(관문 1) → reports/stage6/gate.json.
# - dev: 학습한 후보들의 dev 표와 선택, CAR 효과(판정 없음) → dev.json.
# - test: 고른 모델 하나의 봉인 테스트 결과로 H16·H17을 판정한다 → test.json.
# - supplement: dev 교란(1px 이동, 흐림 σ 1.0)과 동적 INT8(판정 없음) → dev-supplement.json.
# - seeds: 6단계 보충, dms-280-car·dms-280의 학습 시드 분산(판정 없음, dev만) → seeds.json.
# [왜 필요한가] 4단계 PatchCore 서빙 구성이 H15(CPU 200ms + 오검출률 + 정확도)를 지키지 못해, 같은 형식의
# H16으로 DMS가 대체할 수 있는지 잰다. 규칙은 docstring과 docs/experiments.md 6단계에 학습 전에 등록했다.
# [결과] 학습 후보 dms-252/280/308/280-car, 선택 dms-280-car(dev AUROC 96.98). H16 지지(CPU 149.7ms, 오검출률
# 4.26%, AUROC 96.47로 DM보다 0.32%p 낮음), H17 지지(서빙 PatchCore보다 +6.43%p [+5.10, +7.79]).
# [연결] 지연은 bench(reports/stage6/latency.json), dev·test 점수는 run_dinomaly eval(torch)과
# dinomaly_serving score(ONNX), 비교 대상 PatchCore는 run_export_eval의 4단계 CPU FP32 점수, DM 기준값은 2단계
# compare 리포트에서 읽는다. 부트스트랩과 요약은 compare 모듈 것을 쓴다.

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# scipy.stats.t: 6단계 보충의 seed 짝 CAR 효과 구간에 쓰는 t 분포 분위수.
from scipy import stats as scipy_stats

# compare는 실행 읽기·짝지은 부트스트랩·요약을, conformal_threshold는 torch 임계값과 dev 보조 표의 임계값
# 계산을 맡는다. sha256_file은 6단계 보충에서 채점한 모델 파일이 지금도 같은지 대조하고, auroc·fpr_at_tpr는
# 부트스트랩 없는 seed별 요약을 낸다.
from . import compare, paths
from .calibrate import conformal_threshold
from .download import sha256_file
from .metrics import auroc, fpr_at_tpr

# 등록된 판정 상수. CPU 예산 200ms, 관문 1에서 잰 입력 크기 네 개, H16 (나)의 오검출률 상한 6%(1단계 이론 구간
# 상한 5.9%를 올린 값, H15와 같음), H16 (다)의 DM 대비 허용 차이 2.0%p, H17의 최소 효과 1.0%p, 목표 오검출률
# 5%.
CPU_BUDGET_MS = 200.0
GATE_SIZES = (252, 280, 294, 308)
H16_FPR_MAX = 0.06
H16_AUROC_MARGIN = 0.02
H17_MIN_EFFECT = 0.01
ALPHA = 0.05

# 6단계 보충(학습 시드 분산)의 등록 상수. SEED_CONFIGS는 (CAR 켬, CAR 끔) 순서이고 CAR 효과는 앞 − 뒤다. seed
# 0은 6단계 모델과 dev 채점을 그대로 쓰고 seed 1~3을 새로 학습한다. 학습 설정은 6단계와 같아야 한다(10,000스텝
# × 배치 16, fp16 autocast).
SEED_CONFIGS = ("dms-280-car", "dms-280")
SEEDS = (0, 1, 2, 3)
SEED_STEPS = 10_000
SEED_BATCH = 16
# 받는 채점 묶음: 여덟 개 전부와, 측정 전에 정한 줄이는 순서(dms-280 seed 3 → dms-280-car seed 3)대로 뺀 두
# 묶음. 그 밖의 조합은 결과를 보고 고른 seed가 섞일 수 있어 받지 않는다.
_FULL_RUN_SET = frozenset((name, seed) for name in SEED_CONFIGS for seed in SEEDS)
SEED_RUN_SETS = (
    _FULL_RUN_SET,
    _FULL_RUN_SET - {("dms-280", 3)},
    _FULL_RUN_SET - {("dms-280", 3), ("dms-280-car", 3)},
)
# seed 0 대조: seed 0 두 채점의 세 값이 6단계 dev 기록(reports/stage6/dev.json)과 이 차이 안이어야 한다. 같은
# 점수 파일을 같은 정의로 다시 계산하므로 실제로는 비트 단위로 같다(규칙을 적기 전에 확인). 1e-12는 합산 순서
# 같은 부동소수점 차이만 넘기고, 다른 점수 파일이면 걸린다.
SEED0_KEYS = ("macro_image_auroc", "macro_aupro", "pooled_fpr_at_tpr95")
SEED0_TOL = 1e-12
# 해석 (나)의 6단계 선택 차이는 seed 0 dev에서 고른 dms-280-car와 다음 후보 dms-308의 평균 AUROC 차이다.
SELECTION_RUNNER_UP = "dms-308"
DEV_REPORT = paths.REPORTS / "stage6" / "dev.json"
# 학습 설정 대조에서 구성 기록(config) 밖에서 함께 맞춰 보는 학습 기록 항목.
_SETTING_KEYS = ("device", "train_images", "train_images_by_category")


# 값은 bench가 쓴 total_with_resize_ms다. 기본 접미사 -untrained는 관문 1의 학습 전 모델, 빈 접미사는 학습한
# 아티팩트다.
def gate_latency(
    latency: dict, name: str, suffix: str = "-untrained", precision: str = "fp32"
) -> float | None:
    """지연 파일에서 `<name><suffix>` 항목의 등록된 CPU 지연(리사이즈 + 추론)을 읽는다. 재지 않았으면 None."""
    entry = latency.get(name + suffix, {}).get(f"cpu_{precision}")
    return None if entry is None else float(entry["total_with_resize_ms"])


# 관문 1 판정. 200ms 이하인 크기만 학습한다: 252·280·308 가운데 통과한 것, 308이 빠지고 294가 통과하면 294가
# 308 자리에 들어간다(최대 3개). CAR 모델은 280이 통과하면 280, 아니면 252로. 아무것도 학습할 수 없으면
# blocked로 표시해 단계를 멈춘다. 결과: 네 크기 모두 통과(부하 상태에서 138.8~197.5ms)해 dms-252, 280, 308,
# 280-car를 학습했다.
def gate(latency: dict, budget_ms: float = CPU_BUDGET_MS) -> dict:
    """학습 전 모델(`dms-<크기>-untrained`)의 지연으로 어떤 DMS 모델을 학습할지 정한다."""
    ms = {size: gate_latency(latency, f"dms-{size}") for size in GATE_SIZES}
    missing = [size for size, value in ms.items() if value is None]
    if missing:
        raise ValueError(f"the latency file has no untrained entry for sizes {missing}")
    passed = [size for size in GATE_SIZES if ms[size] <= budget_ms]
    train = [size for size in (252, 280, 308) if size in passed]
    if 308 not in passed and 294 in passed:
        train.append(294)
    # CAR 모델의 크기: 280이 통과하면 280, 아니면 252, 둘 다 실패면 CAR 모델을 학습하지 않는다.
    car = 280 if 280 in passed else (252 if 252 in passed else None)
    models = [f"dms-{size}" for size in sorted(train)]
    if car is not None:
        models.append(f"dms-{car}-car")
    return {
        "budget_ms": budget_ms,
        "cpu_ms": {f"dms-{size}": ms[size] for size in GATE_SIZES},
        "passed": [f"dms-{size}" for size in passed],
        "train": models,
        "blocked": not models,
    }


# 모델 이름(dms-280, dms-280-car)에서 입력 크기를 읽는다.
def _size(name: str) -> int:
    return int(name.split("-")[1])


# dev 선택 규칙: CPU 지연(관문 1 값) 200ms 이하 가운데 dev 평균 이미지 AUROC가 가장 높은 것, 같으면 입력이
# 작은 쪽. 정렬 키 (−AUROC, 크기)로 한 번에 표현한다. dev 결함으로 고르므로 "라벨 20장 사용"이다(6단계 규칙).
def choose(rows: list[dict], budget_ms: float = CPU_BUDGET_MS) -> dict | None:
    """`name`, `cpu_ms`, `macro_image_auroc`를 가진 행들에서 하나를 고른다(예산 안이 없으면 None)."""
    within = [r for r in rows if r.get("cpu_ms") is not None and r["cpu_ms"] <= budget_ms]
    if not within:
        return None
    best = min(within, key=lambda r: (-r["macro_image_auroc"], _size(r["name"])))
    return {
        "name": best["name"],
        "macro_image_auroc": best["macro_image_auroc"],
        "cpu_ms": best["cpu_ms"],
        "candidates": [r["name"] for r in within],
    }


# H17의 비교 대상이다. 서빙 구성은 wrn50-256 코어셋 1%이고, compare.MethodRun으로 읽는다. 보정은 교차 적합
# 점수다.
def load_patchcore_cpu(run_dir: Path, name: str = "serving-patchcore") -> compare.MethodRun:
    """4단계 서빙 PatchCore의 ONNX FP32 점수(`run_export_eval`의 정밀도 폴더 하나)를 읽는다."""
    run = compare.MethodRun(name, "crossfit")
    for path in sorted(Path(run_dir).glob("*.npz")):
        with np.load(path) as z:
            run.cats[path.stem] = {
                "eval_images": z["eval_images"],
                "eval_labels": z["eval_labels"],
                "eval_score": z["eval_score_full"],
                "cal_score": z["pool_score_oof"],
            }
    if not run.cats:
        raise ValueError(f"{run_dir} holds no category scores")
    return run


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# 표의 칸: 후보마다 평균 이미지 AUROC와 구간, AUPRO, 합산 FPR@TPR95, 관문 1 지연, 범주별 AUROC. 모든 후보가
# 같은 dev 이미지인지 확인하고 부트스트랩 한 벌을 공유한다. 관문 판정도 같이 적는다. CAR 차이는 car_effect로
# 적는다(결과 +0.80%p [+0.07, +1.57]).
def dev_report(run_dirs: list[Path], latency: dict) -> dict:
    """학습한 후보들의 dev 표, 선택, 같은 크기 CAR 켬·끔 쌍의 차이(판정 없음)를 만든다."""
    runs = {}
    for run_dir in run_dirs:
        # 실행 이름은 run.json의 설정 이름(dms-280 등)이고, 관문 1 지연 항목(`<이름>-untrained`)의 키와
        # 맞는다.
        name = _read_json(Path(run_dir) / "run.json")["config"]["name"]
        runs[name] = compare.load_common(Path(run_dir), name)
    names = list(runs)
    compare.check_same_images(list(runs.values()))
    boot = compare.Bootstrap(runs[names[0]])
    rows = []
    for name, run in runs.items():
        summary = compare.summarize(run, boot)
        rows.append(
            {
                "name": name,
                "macro_image_auroc": summary["macro_image_auroc"],
                "macro_image_auroc_ci": summary["macro_image_auroc_ci"],
                "macro_aupro": summary["macro_aupro"],
                "pooled_fpr_at_tpr95": summary["pooled_fpr_at_tpr95"],
                "cpu_ms": gate_latency(latency, name),
                "per_category_auroc": {c: v["image_auroc"] for c, v in summary["per_category"].items()},
            }
        )
    report = {"n_boot": boot.n_boot, "rows": rows, "pick": choose(rows), "gate": gate(latency)}
    for name in names:
        # CAR 켬·끔 쌍(예: dms-280-car와 dms-280)이 둘 다 있으면 같은 추출로 켬 − 끔 차이를 낸다.
        if name.endswith("-car") and name[: -len("-car")] in runs:
            base = name[: -len("-car")]
            point = compare.macro_auroc(runs[name]) - compare.macro_auroc(runs[base])
            samples = boot.macro_auroc(runs[name]) - boot.macro_auroc(runs[base])
            report["car_effect"] = {"pair": [name, base], **compare.difference(point, samples)}
    return report


# 주어진 범주별 임계값에서의 합산 오검출률·검출률(구간 없이 점 추정과 개수만). torch 임계값을 ONNX 점수에
# 그대로 쓸 때와 dev 보조 표에서 쓴다.
def _pooled_rates(run: compare.MethodRun, thresholds: dict[str, float]) -> dict:
    fp = tp = n_neg = n_pos = 0
    for c, cat in run.cats.items():
        labels = cat["eval_labels"]
        scores = cat["eval_score"]
        fp += int((scores[labels == 0] > thresholds[c]).sum())
        tp += int((scores[labels == 1] > thresholds[c]).sum())
        n_neg += int((labels == 0).sum())
        n_pos += int((labels == 1).sum())
    return {"fpr": fp / n_neg, "tpr": tp / n_pos, "fp": fp, "n_neg": n_neg, "tp": tp, "n_pos": n_pos}


# 조건: (가) 학습한 아티팩트의 CPU 지연 ≤ 200ms, (나) ONNX CPU FP32 자신의 홀드아웃 임계값에서 합산 오검출률
# ≤ 6%, (다) ONNX CPU FP32 평균 AUROC ≥ DM − 2.0%p. 셋 다 만족하면 지지, 하나라도 어긋나면 기각하고 어긋난
# 조건을 적는다. 지연을 재지 않았으면 (가)를 만족하지 않은 것으로 본다.
def h16(onnx_summary: dict, cpu_ms: float | None, dm_auroc: float) -> dict:
    """등록한 H16의 세 조건을 점 추정값으로 판정한다."""
    fpr = onnx_summary["fixed_threshold"]["fpr"]
    auroc = onnx_summary["macro_image_auroc"]
    conditions = {
        "a_cpu_ms": {"value": cpu_ms, "met": cpu_ms is not None and cpu_ms <= CPU_BUDGET_MS},
        "b_fpr": {"value": fpr, "met": fpr <= H16_FPR_MAX},
        "c_auroc_vs_dm": {"value": auroc - dm_auroc, "met": auroc >= dm_auroc - H16_AUROC_MARGIN},
    }
    failed = [k for k, v in conditions.items() if not v["met"]]
    return {"conditions": conditions, "failed": failed, "verdict": "지지" if not failed else "기각"}


# H17 판정: 선택 모델 − 서빙 PatchCore(둘 다 ONNX CPU FP32)의 AUROC 차이. 짝지은 구간 하한 > 0이고 차이 ≥
# 1.0%p면 지지, 상한 < 0이면 기각, 나머지 판정 불가.
def h17(point: float, samples: np.ndarray) -> dict:
    lo, hi = compare._ci(samples)
    if lo > 0 and point >= H17_MIN_EFFECT:
        verdict = "지지"
    elif hi < 0:
        verdict = "기각"
    else:
        verdict = "판정 불가"
    return {"diff": float(point), "ci": [lo, hi], "verdict": verdict}


# 판정은 규칙대로 ONNX CPU FP32 점수로 하고, torch(GPU fp32) 실행에서는 AUPRO·픽셀
# AUROC·FPR@TPR95 같은 맵 지표를 가져온다(ONNX 채점은 맵을 저장하지 않는다). 세 실행(ONNX, torch, PatchCore)이
# 같은 봉인 테스트 이미지인지 확인한 뒤 ONNX 기준으로 부트스트랩 한 벌을 만든다.
# 함께 적는 것(판정 없음): torch 점수로 정한 임계값을 ONNX 점수에 그대로 썼을 때의 비율, ONNX와 torch의 이미지
# 점수 상대 차이(결과: 중앙값 1.2e-6, 최대 7.3e-6). 2026-10-04 검토 지적으로 torch의 평균 픽셀
# AUROC(macro_pixel_auroc)를 더했다(결과 99.140).
def sealed_report(
    *,
    torch_dir: Path,
    onnx_dir: Path,
    patchcore_dir: Path,
    cpu_ms: float | None,
    dm_auroc: float,
) -> dict:
    """봉인 테스트 리포트: 선택 모델의 H16·H17 판정과, 그 옆에 함께 적기로 등록한 수치."""
    name = _read_json(Path(onnx_dir) / "run.json")["config"]["name"]
    onnx = compare.load_common(Path(onnx_dir), f"{name}-onnx-fp32")
    torch_run = compare.load_common(Path(torch_dir), f"{name}-torch")
    patchcore = load_patchcore_cpu(Path(patchcore_dir))
    compare.check_same_images([onnx, torch_run, patchcore])
    boot = compare.Bootstrap(onnx)
    onnx_summary = {
        "macro_image_auroc": compare.macro_auroc(onnx),
        "macro_image_auroc_ci": compare._ci(boot.macro_auroc(onnx)),
        "fixed_threshold": boot.pooled_rates(onnx, ALPHA),
    }
    torch_summary = compare.summarize(torch_run, boot)
    point = compare.macro_auroc(onnx) - compare.macro_auroc(patchcore)
    samples = boot.macro_auroc(onnx) - boot.macro_auroc(patchcore)
    # torch 실행 자신의 겹 0 홀드아웃 점수로 정한 임계값. 이것을 ONNX 점수에 그대로 써도 같은 판정이 나오는지
    # 본다(결과: 오검출률 4.26%, 검출률 83.33%로 같았다).
    torch_thresholds = {
        c: conformal_threshold(cat["cal_score"], ALPHA).value for c, cat in torch_run.cats.items()
    }
    # ONNX와 torch의 이미지 점수 상대 차이. 분모가 0에 가까운 경우는 1e-12로 막는다.
    rel = np.concatenate(
        [
            np.abs(onnx.cats[c]["eval_score"] - torch_run.cats[c]["eval_score"])
            / np.maximum(np.abs(torch_run.cats[c]["eval_score"]), 1e-12)
            for c in onnx.cats
        ]
    )
    return {
        "name": name,
        "alpha": ALPHA,
        "n_boot": boot.n_boot,
        "onnx_fp32": onnx_summary,
        "torch": {
            k: torch_summary[k]
            for k in (
                "macro_image_auroc",
                "macro_image_auroc_ci",
                "macro_aupro",
                "macro_aupro_ci",
                "macro_pixel_auroc",
                "fixed_threshold",
            )
        },
        "torch_per_category": torch_summary["per_category"],
        "pooled_fpr_at_tpr95_torch": torch_summary["pooled_fpr_at_tpr95"],
        "onnx_at_torch_thresholds": _pooled_rates(onnx, torch_thresholds),
        "score_rel_diff_onnx_vs_torch": {"median": float(np.median(rel)), "max": float(np.max(rel))},
        "serving_patchcore_auroc": compare.macro_auroc(patchcore),
        "dm_auroc": dm_auroc,
        "cpu_ms": cpu_ms,
        "H16": h16(onnx_summary, cpu_ms, dm_auroc),
        "H17": h17(point, samples),
    }


# 교란 조건 실행은 shift-1, blur-2다. 결과: 흐림 σ 1.0에서 28.50%. 동적 INT8 실행이 있으면 FP32 대비 AUROC
# 차이와 짝지은 구간을 적는다(결과 −1.44%p).
def supplement_report(clean_dir: Path, condition_dirs: list[Path], int8_dir: Path | None = None) -> dict:
    """dev 보조 표(판정 없음): 교란 없는 겹 0 정상 점수로 정한 임계값에서의 교란 조건 비율과 동적 INT8.

    dev 절차에서는 겹 0 정상이 곧 평가 정상이라, 교란 없는 행의 오검출률은 만드는 방식상 5% 안팎이다.
    교란 조건 행은 이미지가 바뀐 뒤 같은 정상 가운데 몇 장이 그 임계값을 넘는지를 보여 준다.
    """
    clean = compare.load_common(Path(clean_dir), "clean")
    thresholds = {}
    for c, cat in clean.cats.items():
        # dev 절차 실행에는 따로 보정 점수가 없고 겹 0 정상이 곧 평가 정상이라, 교란 없는 실행의 평가 정상
        # 점수로 임계값을 정한다(표본 안 임계값).
        thresholds[c] = conformal_threshold(cat["eval_score"][cat["eval_labels"] == 0], ALPHA).value
    rows = [
        {
            "condition": "clean",
            "macro_image_auroc": compare.macro_auroc(clean),
            **_pooled_rates(clean, thresholds),
        }
    ]
    for run_dir in condition_dirs:
        condition = _read_json(Path(run_dir) / "run.json")["config"]["condition"]
        run = compare.load_common(Path(run_dir), condition)
        compare.check_same_images([clean, run])
        rows.append(
            {
                "condition": condition,
                "macro_image_auroc": compare.macro_auroc(run),
                **_pooled_rates(run, thresholds),
            }
        )
    report: dict = {"thresholds": "clean fold-0 normal scores (in-sample in the dev protocol)", "rows": rows}
    if int8_dir is not None:
        int8 = compare.load_common(Path(int8_dir), "int8-dynamic")
        compare.check_same_images([clean, int8])
        boot = compare.Bootstrap(clean)
        point = compare.macro_auroc(int8) - compare.macro_auroc(clean)
        samples = boot.macro_auroc(int8) - boot.macro_auroc(clean)
        report["int8_dynamic"] = {
            "macro_image_auroc": compare.macro_auroc(int8),
            "fp32_macro_image_auroc": compare.macro_auroc(clean),
            **compare.difference(point, samples),
        }
    return report


# 6단계 보충의 dev 채점 폴더 배치. seed 0은 6단계 폴더(outputs/<구성>-dev)를 그대로 쓰고, seed s는 측정
# 스크립트가 outputs/<구성>-s<s>-dev에 쓴다. 순서는 seed 0 두 개, 이어서 seed마다 (CAR 켬, 끔)이다.
def seed_run_dirs(outputs: Path | None = None) -> list[Path]:
    """등록한 여덟 dev 채점 폴더(`outputs`를 주지 않으면 paths.OUTPUTS 아래)."""
    root = paths.OUTPUTS if outputs is None else Path(outputs)
    dirs = [root / f"{name}-dev" for name in SEED_CONFIGS]
    return dirs + [root / f"{name}-s{seed}-dev" for seed in SEEDS[1:] for name in SEED_CONFIGS]


# 부트스트랩 없는 dev 요약. 정의는 6단계 dev 표를 만든 compare.summarize와 같다: 범주별 이미지 AUROC의 단순
# 평균, npz에 저장된 범주별 AUPRO의 평균, 범주마다 결함 95%를 잡는 임계값의 오검출 수를 더해 정상 전체 수로
# 나눈 FPR@TPR95. 이 보충의 지표는 seed 사이 흔들림이라 seed마다 2,000번 재표집을 돌릴 필요가 없고, 정의가
# 같아서 seed 0 대조에 그대로 쓸 수 있다(같은 값인지 테스트로 확인한다).
def point_summary(run: compare.MethodRun) -> dict:
    """평균 이미지 AUROC, 평균 AUPRO, 합산 FPR@TPR95, 범주별 이미지 AUROC(점 추정값만)."""
    per_category = {}
    aupro = []
    fp95 = n_neg = 0
    for c, cat in run.cats.items():
        labels = cat["eval_labels"]
        neg, pos = cat["eval_score"][labels == 0], cat["eval_score"][labels == 1]
        _, _, fp = fpr_at_tpr(neg, pos, 0.95)
        fp95 += fp
        n_neg += len(neg)
        per_category[c] = float(auroc(neg, pos))
        aupro.append(float(cat["aupro"]))
    return {
        "macro_image_auroc": float(np.mean(list(per_category.values()))),
        "macro_aupro": float(np.mean(aupro)),
        "pooled_fpr_at_tpr95": fp95 / n_neg,
        "per_category_auroc": per_category,
    }


# 등록한 CAR 효과 구간: 평균 ± t₀.₉₇₅(n − 1) · s / √n. s는 n − 1로 나눈 표본 표준편차, 분위수는
# scipy.stats.t다. seed 4개면 자유도 3이라 t₀.₉₇₅ = 3.18이다(정규 근사 1.96을 쓰면 구간이 실제보다 좁아진다).
def t_interval(values, conf: float = 0.95) -> dict:
    """값들의 개수, 자유도, 평균, 표본 표준편차, t 분위수와 평균의 `conf` 신뢰구간."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError(f"a t interval needs at least two values, got {n}")
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    t = float(scipy_stats.t.ppf(0.5 + conf / 2, n - 1))
    half = t * sd / math.sqrt(n)
    return {"n": n, "df": n - 1, "mean": mean, "sd": sd, "t": t, "ci": [mean - half, mean + half]}


# 측정 전에 고정한 해석 문장(docs/experiments.md "6단계 보충"의 해석 규칙). 판정이 아니라 README 한계 줄에
# 날짜를 붙여 적을 문장을 고르는 것이다.
# (가) CAR 효과: d의 t 구간 하한 > 0이면 넘는다, 상한 < 0이면 반대 방향으로 낮다, 0을 포함하면(끝이 정확히 0인
#     경우도) 구별되지 않는다.
# (나) 6단계 선택: dms-280-car seed 표준편차 ≥ 선택 차이면 시드 잡음 수준, 작으면 차이가 표준편차의 x배(x =
#     차이 / 표준편차). dms-308은 seed를 반복하지 않으므로 두 문장 모두 근사라고 적는다. 표준편차가 0(또는
#     NaN)이면 서로 다른 학습에서 나올 수 없는 값이라 멈춘다(같은 모델을 여러 번 채점한 경우 등).
def interpret_seeds(car: dict, *, sd_car: float, gap: float, d0: float) -> dict:
    """CAR 효과(`car`: n·df·mean·ci), dms-280-car seed 표준편차, 선택 차이, seed 0의 d로 두 문장을 고른다."""
    lo, hi = car["ci"]
    numbers = (
        f"seed {car['n']}개 짝 차이 평균 {car['mean'] * 100:+.2f}%p, "
        f"t(자유도 {car['df']}) 95% 구간 [{lo * 100:+.2f}, {hi * 100:+.2f}]%p"
    )
    if lo > 0:
        kind, text = "exceeds", "CAR 효과는 학습 시드 잡음을 넘는다"
    elif hi < 0:
        kind, text = (
            "below",
            f"CAR을 켠 쪽이 학습 시드 잡음을 넘게 낮다(seed 0의 {d0 * 100:+.2f}%p와 반대 방향)",
        )
    else:
        kind, text = "indistinguishable", "CAR 효과는 학습 시드 잡음과 구별되지 않는다"
    if not sd_car > 0:
        raise ValueError(
            f"the dms-280-car seed standard deviation is {sd_car!r}: zero spread over separate trainings"
        )
    ratio = gap / sd_car
    gap_text = f"6단계 선택 차이(seed 0 dev의 dms-280-car − {SELECTION_RUNNER_UP}, {gap * 100:.2f}%p)"
    sd_text = f"dms-280-car seed 표준편차 {sd_car * 100:.2f}%p"
    approx = f"{SELECTION_RUNNER_UP}은 seed를 반복하지 않아 dms-280-car의 표준편차로 본 근사"
    if sd_car >= gap:
        selection = {"result": "noise", "sentence": f"{gap_text}는 시드 잡음 수준이다: {sd_text}({approx})"}
    else:
        selection = {"result": "multiple", "sentence": f"{gap_text}는 {sd_text}의 {ratio:.1f}배다({approx})"}
    return {
        "car": {"result": kind, "sentence": f"{text}: {numbers}"},
        "selection": {**selection, "ratio": ratio},
    }


# 리포트에 적는 경로: 저장소 안이면 저장소 기준 상대 경로(이 PC의 절대 경로를 커밋하지 않는다), 밖이면 그대로.
def _rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(paths.ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


# 채점 하나의 기록 대조(규칙의 기록 대조 2와 3 앞부분): dev 절차를 fp32로 채점했는지, run.json에 적힌 모델
# sha256이 지금 모델 파일과 같은지(채점 뒤 다시 학습하지 않았는지), 모델 옆 train.json이 같은 구성·6단계
# 설정(10,000스텝, 배치 16, fp16)·등록한 seed인지 본다. train 명령은 시작할 때 이전 기록을 지우고 모델 다음에
# train.json을 마지막으로 쓰므로, sha256이 맞는 모델 옆에 train.json이 있으면 둘은 같은 학습의 것이다.
# run.json의 모델 경로가 상대 경로면(eval에 --model을 상대 경로로 준 경우) 저장소 루트 기준으로 읽는다(측정
# 스크립트는 루트에서 돈다).
def _seed_record(run_dir: Path) -> dict:
    """dev 채점 폴더 하나의 구성, seed, 모델 경로·sha256, 학습 기록, 채점 커밋. 기록이 어긋나면 ValueError."""
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        raise ValueError(f"{run_dir} holds no finished dev scoring (run.json)")
    meta = _read_json(run_path)
    config = meta.get("config", {})
    name = config.get("name")
    if meta.get("protocol") != "dev":
        raise ValueError(f"{run_dir} is a {meta.get('protocol')!r} scoring, not a dev one")
    if config.get("score_precision") != "fp32":
        raise ValueError(f"{run_dir} was scored in {config.get('score_precision')!r}, not fp32 (--no-amp)")
    if name not in SEED_CONFIGS:
        raise ValueError(f"{run_dir} scores {name!r}, not one of {list(SEED_CONFIGS)}")
    model = Path(config.get("model", ""))
    if not model.is_absolute():
        model = paths.ROOT / model
    if not model.is_file():
        raise ValueError(f"{model}, the model scored in {run_dir}, is missing")
    sha = sha256_file(model)
    if sha != config.get("model_sha256"):
        raise ValueError(
            f"{model} no longer has the sha256 recorded in {run_path} (retrained after scoring?)"
        )
    train_path = model.parent / "train.json"
    if not train_path.is_file():
        raise ValueError(f"{train_path} is missing: the scored model has no finished training record")
    train = _read_json(train_path)
    setting = train.get("config", {})
    if setting.get("name") != name:
        raise ValueError(f"{train_path} trained {setting.get('name')!r}, but {run_dir} scores {name!r}")
    stage6_setting = (SEED_STEPS, SEED_BATCH, True)
    if (setting.get("steps"), setting.get("batch_size"), setting.get("amp")) != stage6_setting:
        raise ValueError(
            f"{train_path} is not the stage 6 setting ({SEED_STEPS} steps, batch {SEED_BATCH}, fp16): "
            f"{setting.get('steps')} steps, batch {setting.get('batch_size')}, amp {setting.get('amp')}"
        )
    seed = setting.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed not in SEEDS:
        raise ValueError(f"{train_path} has seed {seed!r}, not one of {list(SEEDS)}")
    return {
        "config": name,
        "seed": seed,
        "run_dir": run_dir,
        "model": model,
        "model_sha256": sha,
        "train": train,
        "eval_commit": meta.get("commit"),
    }


# 학습 설정 대조(규칙의 기록 대조 3 뒷부분): 구성 기록 전체와 장치·학습 이미지 수가 같은 구성 seed 0의 기록과
# 같아야 한다. 빼는 것은 seed와 관문 파일 경로(gate.file)뿐이다. 경로 문자열은 부른 셸에 따라 구분자가 달라질
# 수 있고, 같은 관문 파일인지는 함께 기록된 sha256으로 본다. 결과는 다른 항목 이름들(같으면 빈 목록)이다.
def _setting_diff(reference: dict, train: dict) -> list[str]:
    def setting(record: dict) -> dict:
        config = {k: v for k, v in record.get("config", {}).items() if k != "seed"}
        if isinstance(config.get("gate"), dict):
            config["gate"] = {k: v for k, v in config["gate"].items() if k != "file"}
        return config

    a, b = setting(reference), setting(train)
    diff = [f"config.{k}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    return diff + [k for k in _SETTING_KEYS if reference.get(k) != train.get(k)]


# seed 0 대조(규칙의 기록 대조 1): seed 0 채점의 점 요약이 6단계 dev 기록의 같은 구성 행과 SEED0_TOL 안에서
# 같아야 한다. 어긋나면(NaN 포함) seed 0 쪽 점수 파일이 6단계 기록의 것이 아니므로 멈춘다. 결과는 구성별
# 기록값과 다시 낸 값.
def _check_seed0(summaries: dict[str, dict], dev_report: dict) -> dict:
    rows = {row["name"]: row for row in dev_report.get("rows", [])}
    checked = {}
    for name, summary in summaries.items():
        if name not in rows:
            raise ValueError(f"the stage 6 dev record (reports/stage6/dev.json) has no {name} row")
        values = {}
        for key in SEED0_KEYS:
            recorded, got = float(rows[name][key]), float(summary[key])
            if not abs(got - recorded) <= SEED0_TOL:
                raise ValueError(
                    f"seed 0 of {name} does not reproduce reports/stage6/dev.json: {key} {got!r} "
                    f"against the recorded {recorded!r}"
                )
            values[key] = {"recorded": recorded, "reproduced": got}
        checked[name] = {"equal": True, **values}
    return checked


# 측정 스크립트가 GPU 학습 전에 부르는 점검: seed 0 두 채점만 기록 대조와 6단계 dev 기록 재현을 확인한다(새
# seed 폴더가 없어도 된다). 저장된 점수와 모델·학습 기록 파일만 읽고, 모델을 불러 돌리지 않는다.
def check_seed0_runs(run_dirs: list[Path] | None = None, dev_report: Path | None = None) -> dict:
    """seed 0 두 채점의 기록 대조와 dev.json 재현. 어긋나면 ValueError, 맞으면 구성별 기록값과 다시 낸 값."""
    run_dirs = seed_run_dirs()[: len(SEED_CONFIGS)] if run_dirs is None else run_dirs
    records = [_seed_record(Path(d)) for d in run_dirs]
    found = sorted((r["config"], r["seed"]) for r in records)
    if found != sorted((name, 0) for name in SEED_CONFIGS):
        raise ValueError(f"expected the seed 0 scorings of {list(SEED_CONFIGS)}, got {found}")
    summaries = {r["config"]: point_summary(compare.load_common(r["run_dir"], r["config"])) for r in records}
    return _check_seed0(summaries, _read_json(DEV_REPORT if dev_report is None else Path(dev_report)))


# 측정 스크립트가 새 seed의 채점 하나가 끝날 때마다 부르는 점검: 그 채점의 기록 대조, 학습 설정이 같은 구성
# seed 0(기본 outputs/<구성>-dev)과 같은지, 같은 dev 이미지인지. 마지막 분석과 같은 대조를 미리 해, 설정이
# 어긋나면 남은 학습을 돌리기 전에 멈춘다. 지표는 돌려주지 않고(결과를 보며 진행하지 않도록) 학습 기록만
# 돌려준다.
def check_seed_run(run_dir: Path, seed0_dir: Path | None = None) -> dict:
    """새 seed 채점 하나의 기록 대조. 어긋나면 ValueError, 맞으면 구성·seed·모델 sha256과 학습 기록 요약."""
    record = _seed_record(Path(run_dir))
    seed0_dir = paths.OUTPUTS / f"{record['config']}-dev" if seed0_dir is None else Path(seed0_dir)
    reference = _seed_record(seed0_dir)
    if (reference["config"], reference["seed"]) != (record["config"], 0):
        raise ValueError(f"{seed0_dir} is not the seed 0 scoring of {record['config']}")
    if reference["model_sha256"] == record["model_sha256"]:
        raise ValueError(f"{run_dir} scores the seed 0 model of {record['config']} (same sha256)")
    diff = _setting_diff(reference["train"], record["train"])
    if diff:
        raise ValueError(
            f"{run_dir}: the training record differs from seed 0 of {record['config']} in {diff}"
        )
    compare.check_same_images(
        [compare.load_common(reference["run_dir"], "seed0"), compare.load_common(record["run_dir"], "new")]
    )
    train = record["train"]
    return {
        "config": record["config"],
        "seed": record["seed"],
        "model_sha256": record["model_sha256"],
        **{k: train.get(k) for k in ("device", "final_loss", "skipped_steps", "peak_vram_mb", "train_s")},
    }


# 6단계 보충 리포트. 순서: 채점마다 기록 대조 → 같은 구성·seed 중복과 등록한 묶음인지 → 모델 파일이 모두
# 다른지 → 학습 설정이 같은 구성 seed 0과 같은지 → 모두 같은 dev 이미지인지 → seed 0 대조 → 요약. 하나라도
# 어긋나면 ValueError로 멈추고 아무것도 내지 않는다(규칙의 멈춤 조건). 판정은 없고, 해석 문장만 등록한
# 규칙대로 고른다. CAR 효과는 두 구성이 모두 있는 seed끼리의 짝 차이다(줄인 묶음이면 자유도가 그만큼 준다).
def seeds_report(run_dirs: list[Path], dev_report: dict) -> dict:
    """dms-280-car·dms-280의 seed별 dev AUROC 분산, seed 짝 CAR 효과의 t 구간, 6단계 선택 차이와의 비교."""
    records = [_seed_record(Path(d)) for d in run_dirs]
    keys = [(r["config"], r["seed"]) for r in records]
    twice = sorted({k for k in keys if keys.count(k) > 1})
    if twice:
        raise ValueError(f"these config and seed pairs appear twice: {twice}")
    if frozenset(keys) not in SEED_RUN_SETS:
        raise ValueError(
            f"the scorings {sorted(keys)} are not a registered run set "
            "(all eight, all but dms-280 seed 3, or all but both seed 3 runs)"
        )
    shas = [r["model_sha256"] for r in records]
    if len(set(shas)) != len(shas):
        raise ValueError("two scorings come from the same model file (same sha256)")
    order = sorted(keys, key=lambda k: (SEED_CONFIGS.index(k[0]), k[1]))
    by_key = {(r["config"], r["seed"]): r for r in records}
    for key in order:
        diff = _setting_diff(by_key[(key[0], 0)]["train"], by_key[key]["train"])
        if diff:
            raise ValueError(
                f"{by_key[key]['run_dir']}: the training record differs from seed 0 of {key[0]} in {diff}"
            )
    runs = {key: compare.load_common(by_key[key]["run_dir"], f"{key[0]}-s{key[1]}") for key in order}
    compare.check_same_images(list(runs.values()))
    summaries = {key: point_summary(run) for key, run in runs.items()}
    seed0 = _check_seed0({name: summaries[(name, 0)] for name in SEED_CONFIGS}, dev_report)

    # 구성마다 seed별 평균 이미지 AUROC와 그 평균·표본 표준편차(n − 1)·최솟값·최댓값. AUPRO와 FPR@TPR95는
    # 참고.
    per_config = {}
    for name in SEED_CONFIGS:
        seeds = [s for c, s in order if c == name]
        values = [summaries[(name, s)]["macro_image_auroc"] for s in seeds]
        entry = {
            "seeds": seeds,
            "auroc": values,
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "min": min(values),
            "max": max(values),
        }
        for label, key in (("aupro", "macro_aupro"), ("fpr_at_tpr95", "pooled_fpr_at_tpr95")):
            other = [summaries[(name, s)][key] for s in seeds]
            entry[label] = {
                "values": other,
                "mean": float(np.mean(other)),
                "sd": float(np.std(other, ddof=1)),
            }
        per_config[name] = entry

    # seed 짝 CAR 효과 d_s = AUROC(켬, s) − AUROC(끔, s)와 t 구간. 6단계 기록의 seed 0 차이와 구간도 옆에
    # 둔다.
    car_name, base_name = SEED_CONFIGS
    paired = [s for s in SEEDS if (car_name, s) in by_key and (base_name, s) in by_key]
    d = [
        summaries[(car_name, s)]["macro_image_auroc"] - summaries[(base_name, s)]["macro_image_auroc"]
        for s in paired
    ]
    recorded_car = dev_report.get("car_effect", {})
    car = {
        "pair": list(SEED_CONFIGS),
        "seeds": paired,
        "d": d,
        **t_interval(d),
        "stage6_seed0": {"diff": recorded_car.get("diff"), "ci": recorded_car.get("ci")},
    }

    # 6단계 선택 차이(seed 0 dev의 dms-280-car − dms-308)는 기록된 dev.json 값(반올림 전)으로 낸다.
    rows = {row["name"]: row for row in dev_report.get("rows", [])}
    if SELECTION_RUNNER_UP not in rows:
        raise ValueError(f"the stage 6 dev record has no {SELECTION_RUNNER_UP} row")
    gap = float(rows[car_name]["macro_image_auroc"]) - float(rows[SELECTION_RUNNER_UP]["macro_image_auroc"])
    sd_car = per_config[car_name]["sd"]
    interpretation = interpret_seeds(car, sd_car=sd_car, gap=gap, d0=d[0])
    dropped = sorted(_FULL_RUN_SET - set(keys), key=lambda k: (SEED_CONFIGS.index(k[0]), k[1]))
    if dropped:
        listed = ", ".join(f"{c} seed {s}" for c, s in dropped)
        interpretation["reduced"] = f"줄인 조합: 등록한 줄이는 순서대로 {listed}을 뺐다"

    out_runs = []
    for key in order:
        rec, train = by_key[key], by_key[key]["train"]
        out_runs.append(
            {
                "config": key[0],
                "seed": key[1],
                "run": _rel(rec["run_dir"]),
                "model": _rel(rec["model"]),
                "model_sha256": rec["model_sha256"],
                "train_commit": train.get("commit"),
                "eval_commit": rec["eval_commit"],
                "train": {
                    k: train.get(k) for k in ("final_loss", "skipped_steps", "peak_vram_mb", "train_s")
                },
                **summaries[key],
            }
        )
    return {
        "configs": list(SEED_CONFIGS),
        "registered_seeds": list(SEEDS),
        "reduced": bool(dropped),
        "dropped": [[c, s] for c, s in dropped],
        "seed0_check": seed0,
        "per_config": per_config,
        "car_effect": car,
        "selection": {
            "pair": [car_name, SELECTION_RUNNER_UP],
            "gap": gap,
            "sd_car": sd_car,
            "ratio": interpretation["selection"]["ratio"],
        },
        "interpretation": interpretation,
        "runs": out_runs,
    }


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


# 명령행 입구: 하위 명령마다 입력을 읽어 리포트를 만들고 파일과 표준 출력에 쓴다. test는 2단계 compare
# 리포트에서 기준 비지도 방법(DM)의 평균 AUROC를 읽고, --latency-key로 학습한 선택 모델의 지연 항목을 찾는다.
# 모두 저장된 점수만 읽으므로 이 모듈은 봉인 테스트 이미지를 읽지 않는다(seeds는 sha256 대조를 위해 모델
# 파일과 학습 기록도 읽지만 모델을 돌리지 않는다).
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    default_latency = paths.REPORTS / "stage6" / "latency.json"

    g = commands.add_parser("gate", help="학습 전 모델의 지연으로 학습 후보를 정한다")
    g.add_argument("--latency", type=Path, default=default_latency)
    g.add_argument("--out", type=Path, default=paths.REPORTS / "stage6" / "gate.json")

    d = commands.add_parser("dev", help="dev 표, 선택, CAR 차이")
    d.add_argument(
        "--runs", type=Path, nargs="+", required=True, help="run_dinomaly eval --protocol dev 실행 폴더들"
    )
    d.add_argument("--latency", type=Path, default=default_latency)
    d.add_argument("--out", type=Path, default=paths.REPORTS / "stage6" / "dev.json")

    t = commands.add_parser("test", help="선택 모델의 H16과 H17")
    t.add_argument("--torch", type=Path, required=True, help="run_dinomaly eval --protocol test 실행")
    t.add_argument("--onnx", type=Path, required=True, help="dinomaly_serving score --protocol test 실행")
    t.add_argument(
        "--patchcore",
        type=Path,
        default=paths.OUTPUTS / "export-serving-wrn50-256-r0.01-test" / "fp32",
        help="서빙 PatchCore의 4단계 CPU FP32 점수",
    )
    t.add_argument("--latency", type=Path, default=default_latency)
    t.add_argument("--latency-key", required=True, help="학습한 선택 모델의 지연 항목 이름, 예: dms-280")
    t.add_argument("--stage2", type=Path, default=paths.REPORTS / "stage2" / "compare-test.json")
    t.add_argument("--out", type=Path, default=paths.REPORTS / "stage6" / "test.json")

    s = commands.add_parser("supplement", help="선택 모델의 dev 교란 조건과 동적 INT8(판정 없음)")
    s.add_argument("--clean", type=Path, required=True)
    s.add_argument("--conditions", type=Path, nargs="*", default=[])
    s.add_argument("--int8", type=Path, default=None)
    s.add_argument("--out", type=Path, default=paths.REPORTS / "stage6" / "dev-supplement.json")

    # seeds: 6단계 보충. 기본 채점 폴더는 등록한 여덟 개(seed_run_dirs)이고, 줄이는 순서대로 뺐으면 남은
    # 폴더만 --runs로 준다. 등록한 묶음이 아니거나 기록 대조가 어긋나면 리포트를 쓰지 않고 멈춘다.
    sp = commands.add_parser(
        "seeds",
        help="6단계 보충: dms-280-car·dms-280의 학습 시드 분산(dev만, 판정 없음)",
        description=(
            "dms-280-car와 dms-280을 seed 0~3으로 학습한 dev 채점에서 구성마다 평균 이미지 AUROC의 "
            "seed 평균과 표본 표준편차, seed 짝 CAR 효과의 t 구간을 내고, 측정 전에 정한 해석 문장을 고른다. "
            "seed 0은 6단계 채점(outputs/<구성>-dev)이고 reports/stage6/dev.json을 다시 내야 한다. 저장된 "
            "점수와 모델·학습 기록 파일만 읽는다(봉인 테스트 없음)."
        ),
    )
    sp.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        default=None,
        help="run_dinomaly eval --protocol dev 채점 폴더들. 기본값: outputs/<구성>-dev와 "
        "outputs/<구성>-s<s>-dev(s = 1, 2, 3) 여덟 개. 줄였으면 등록한 순서(dms-280 seed 3 → "
        "dms-280-car seed 3)대로 뺀 폴더만 준다",
    )
    sp.add_argument(
        "--dev-report", type=Path, default=DEV_REPORT, help="seed 0 대조와 선택 차이를 읽을 6단계 dev 기록"
    )
    sp.add_argument(
        "--out", type=Path, default=paths.REPORTS / "stage6" / "seeds.json", help="리포트 경로(커밋한다)"
    )

    args = parser.parse_args(argv)
    if args.command == "gate":
        report = gate(_read_json(args.latency))
    elif args.command == "dev":
        report = dev_report(args.runs, _read_json(args.latency))
    elif args.command == "test":
        stage2 = _read_json(args.stage2)
        # H16 (다)의 기준: 2단계 compare 리포트에 적힌 기준 비지도 방법(DM)의 평균 이미지 AUROC(96.79).
        dm_auroc = float(stage2["unsupervised"][stage2["reference_unsupervised"]]["macro_image_auroc"])
        latency = _read_json(args.latency)
        report = sealed_report(
            torch_dir=args.torch,
            onnx_dir=args.onnx,
            patchcore_dir=args.patchcore,
            cpu_ms=gate_latency(latency, args.latency_key, suffix=""),
            dm_auroc=dm_auroc,
        )
    elif args.command == "supplement":
        report = supplement_report(args.clean, args.conditions, args.int8)
    else:
        report = seeds_report(args.runs or seed_run_dirs(), _read_json(args.dev_report))
    _write(args.out, report)
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
