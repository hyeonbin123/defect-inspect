"""6단계: CPU 지연 관문, dev에서의 DMS 모델 선택, 봉인 테스트의 H16·H17 판정.

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
"""

# [파일 역할] 6단계(CPU 서빙용 Dinomaly ViT-S) 분석 모듈이다. 하위 명령 네 개가 6단계의 순서와 같다.
# - gate: 학습 전 모델의 CPU 지연으로 학습할 크기를 정한다(관문 1) → reports/stage6/gate.json.
# - dev: 학습한 후보들의 dev 표와 선택, CAR 효과(판정 없음) → dev.json.
# - test: 고른 모델 하나의 봉인 테스트 결과로 H16·H17을 판정한다 → test.json.
# - supplement: dev 교란(1px 이동, 흐림 σ 1.0)과 동적 INT8(판정 없음) → dev-supplement.json.
# [왜 필요한가] 4단계 PatchCore 서빙 구성이 H15(CPU 200ms + 오검출률 + 정확도)를 지키지 못해, 같은 형식의
# H16으로 DMS가 대체할 수 있는지 잰다. 규칙은 docstring과 docs/experiments.md 6단계에 학습 전에 등록했다.
# [결과] 학습 후보 dms-252/280/308/280-car, 선택 dms-280-car(dev AUROC 96.98). H16 지지(CPU 149.7ms, 오검출률
# 4.26%, AUROC 96.47로 DM보다 0.32%p 낮음), H17 지지(서빙 PatchCore보다 +6.43%p [+5.10, +7.79]).
# [연결] 지연은 bench(reports/stage6/latency.json), dev·test 점수는 run_dinomaly eval(torch)과
# dinomaly_serving score(ONNX), 비교 대상 PatchCore는 run_export_eval의 4단계 CPU FP32 점수, DM 기준값은 2단계
# compare 리포트에서 읽는다. 부트스트랩과 요약은 compare 모듈 것을 쓴다.

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# compare는 실행 읽기·짝지은 부트스트랩·요약을, conformal_threshold는 torch 임계값과 dev 보조 표의 임계값
# 계산을 맡는다.
from . import compare, paths
from .calibrate import conformal_threshold

# 등록된 판정 상수. CPU 예산 200ms, 관문 1에서 잰 입력 크기 네 개, H16 (나)의 오검출률 상한 6%(1단계 이론 구간
# 상한 5.9%를 올린 값, H15와 같음), H16 (다)의 DM 대비 허용 차이 2.0%p, H17의 최소 효과 1.0%p, 목표 오검출률
# 5%.
CPU_BUDGET_MS = 200.0
GATE_SIZES = (252, 280, 294, 308)
H16_FPR_MAX = 0.06
H16_AUROC_MARGIN = 0.02
H17_MIN_EFFECT = 0.01
ALPHA = 0.05


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


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


# 명령행 입구: 하위 명령마다 입력을 읽어 리포트를 만들고 파일과 표준 출력에 쓴다. test는 2단계 compare
# 리포트에서 기준 비지도 방법(DM)의 평균 AUROC를 읽고, --latency-key로 학습한 선택 모델의 지연 항목을 찾는다.
# 모두 저장된 점수만 읽으므로 이 모듈은 봉인 테스트 이미지를 읽지 않는다.
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
    else:
        report = supplement_report(args.clean, args.conditions, args.int8)
    _write(args.out, report)
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
