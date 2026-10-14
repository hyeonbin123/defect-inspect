"""내보낸 CPU 파이프라인이 torch 파이프라인에서 얼마나 벗어나는지, INT8이 임계값을 새로 잡아야 하는지 잰다.

파이프라인은 `torch`(GPU fp16, 뱅크와 임계값을 만든 경로), `fp32`와 `int8`(CPU의 onnxruntime)이다. 각각에
대해 정확도와, 지정한 파이프라인의 교차 적합 점수로 보정한 임계값에서의 합산 오검출률·검출률을 낸다.
"""

# [파일 역할] 4단계에서 PatchCore를 CPU용으로 내보낸 뒤, 점수가 GPU torch 파이프라인과 얼마나 달라지는지,
# INT8이 임계값을 다시 잡아야 하는지를 표로 만드는 분석 모듈이다(`python -m defect_inspect.analyze_export
# outputs/export-...-test`).
# [비교하는 파이프라인] torch(GPU fp16, 뱅크와 임계값을 만든 원래 경로), fp32·int8(onnxruntime CPU).
# 파이프라인마다 정확도, 자기 교차 적합 점수로 정한 임계값의 합산 오검출률·검출률을 내고, CPU 쪽은 torch
# 임계값을 그대로 썼을 때와 torch 대비 점수 차이도 낸다. 그리고 가설 H14(FP32 임계값을 INT8 점수에 그대로 쓰면
# 오검출률이 2.0%p 넘게 달라진다)를 판정한다.
# [결과] FP32 CPU는 torch와 AUROC가 같고 torch 임계값을 그대로 써도 오검출률 4.18%로 같았다(점수 상대 차이
# 중앙값 0.012%). INT8은 FP32 임계값을 쓰면 4.18% → 68.48%로 H14 지지, 다시 잡으면 3.77%지만 AUROC가 7.1%p
# 떨어져 쓰지 않았다(docs/experiments.md 4단계 test 결과).
# [연결] 입력은 run_export_eval의 출력 폴더, 출력은 reports/stage4/export-<절차>.json. 부트스트랩·차이 판정은
# compare 모듈 것을 그대로 쓴다.

# Fraction은 H14의 2.0%p 기준을 오검출 "수"로 바꿔 정확히 비교하는 데 쓴다(shift_verdict 참고).
import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import compare, paths
from .calibrate import conformal_threshold

# ALPHA: 목표 오검출률 5%. MIN_EFFECT_FPR: 합산 오검출률의 최소 효과 2.0%p(공통 규칙, H14의 기준).
ALPHA = 0.05
MIN_EFFECT_FPR = 0.02


# run_export_eval 출력을 파이프라인별 공통 모양으로 읽는다. 정밀도 폴더(fp32/, int8/)의 npz에는 CPU
# 점수(eval_score_full, pool_score_oof)와 함께 같은 이미지의 torch 점수(torch_eval_score,
# torch_pool_score_oof)가 들어 있어, torch 파이프라인도 같은 파일에서 만든다. 보정 점수는 모두 교차 적합
# 점수다.
def load(out_dir: Path) -> dict:
    with open(out_dir / "run.json", encoding="utf-8") as f:
        meta = json.load(f)
    pipelines: dict[str, dict[str, dict[str, np.ndarray]]] = {"torch": {}}
    for precision in meta["precisions"]:
        pipelines[precision] = {}
        for category in meta["categories"]:
            with np.load(out_dir / precision / f"{category}.npz") as z:
                arrays = {k: z[k] for k in z.files}
            pipelines[precision][category] = {
                "eval_images": arrays["eval_images"],
                "eval_labels": arrays["eval_labels"],
                "eval_score": arrays["eval_score_full"],
                "cal_score": arrays["pool_score_oof"],
            }
            pipelines["torch"][category] = {
                "eval_images": arrays["eval_images"],
                "eval_labels": arrays["eval_labels"],
                "eval_score": arrays["torch_eval_score"],
                "cal_score": arrays["torch_pool_score_oof"],
            }
    return {"meta": meta, "pipelines": pipelines}


# 두 인자를 다르게 주면 "다른 파이프라인의 임계값을 그대로 쓰면"을 잴 수 있다.
def _rates(
    boot: compare.Bootstrap, scores: dict[str, dict], thresholds_from: dict[str, dict], alpha: float
) -> dict[str, np.ndarray | float]:
    """`thresholds_from`으로 보정한 범주별 임계값에서 `scores`의 합산 FPR·TPR과 부트스트랩 표본을 낸다."""
    fp = tp = n_neg = n_pos = 0
    fp_boot = np.zeros(boot.n_boot)
    tp_boot = np.zeros(boot.n_boot)
    for i, c in enumerate(boot.names):
        # 임계값은 thresholds_from 파이프라인의 교차 적합 점수에서 정하고, 표시는 scores 파이프라인의 평가
        # 점수에 한다(규칙: 점수 > 임계값이면 불량).
        thr = conformal_threshold(thresholds_from[c]["cal_score"], alpha).value
        labels = scores[c]["eval_labels"]
        neg, pos = scores[c]["eval_score"][labels == 0], scores[c]["eval_score"][labels == 1]
        f_neg, f_pos = neg > thr, pos > thr
        neg_idx, pos_idx = boot.indices(i)
        fp += int(f_neg.sum())
        tp += int(f_pos.sum())
        n_neg += len(neg)
        n_pos += len(pos)
        fp_boot += f_neg[neg_idx].sum(axis=1)
        tp_boot += f_pos[pos_idx].sum(axis=1)
    return {
        "fpr": fp / n_neg,
        "tpr": tp / n_pos,
        "fpr_boot": fp_boot / n_neg,
        "tpr_boot": tp_boot / n_pos,
        # 오검출률의 바탕이 되는 오검출 수(fp, fp_boot)도 돌려준다. 최소 효과와의 비교를 정확한 정수로 하기
        # 위해서다.
        "fp": fp,
        "fp_boot": fp_boot,
        "n_neg": n_neg,
    }


# 리포트 칸 하나: 오검출률·검출률과 각 95% 구간.
def _cell(r: dict) -> dict:
    return {
        "fpr": float(r["fpr"]),
        "fpr_ci": compare._ci(r["fpr_boot"]),
        "tpr": float(r["tpr"]),
        "tpr_ci": compare._ci(r["tpr_boot"]),
    }


# 리포트 조립. 세 파이프라인이 같은 이미지를 채점했는지 확인하고, torch를 기준으로 부트스트랩 한 벌을 만든다.
# 파이프라인마다 평균 AUROC와 구간, 자기 임계값의 비율을 적고, CPU 파이프라인은 추가로 (1) torch 대비 AUROC
# 차이, (2) torch 임계값을 그대로 쓴 비율과 그때 오검출률이 torch 자신과 얼마나 다른지, (3) torch 대비 이미지
# 점수의 상대 차이(중앙값, 최대)를 적는다. fp32와 int8이 모두 있으면 H14를 판정한다.
def build_report(run: dict, alpha: float = ALPHA) -> dict:
    pipelines = run["pipelines"]
    runs = {name: compare.MethodRun(name, "crossfit", cats) for name, cats in pipelines.items()}
    compare.check_same_images(list(runs.values()))
    boot = compare.Bootstrap(runs["torch"])
    base_auroc = compare.macro_auroc(runs["torch"])
    base_auroc_boot = boot.macro_auroc(runs["torch"])

    rows = []
    for name, cats in pipelines.items():
        auroc_boot = boot.macro_auroc(runs[name])
        # 자기 임계값: 파이프라인마다 자기 교차 적합 점수로 임계값을 정했을 때의 비율.
        own = _rates(boot, cats, cats, alpha)
        row = {
            "pipeline": name,
            "macro_image_auroc": compare.macro_auroc(runs[name]),
            "macro_image_auroc_ci": compare._ci(auroc_boot),
            "own_thresholds": _cell(own),
        }
        if name != "torch":
            row["vs_torch_auroc"] = compare.difference(
                row["macro_image_auroc"] - base_auroc, auroc_boot - base_auroc_boot
            )
            # torch 임계값을 CPU 점수에 그대로 쓴 경우와 torch 자신의 경우를 같은 추출로 비교한다. 4단계에서
            # FP32는 4.18%로 같았고, INT8은 크게 달라졌다.
            at_torch = _rates(boot, cats, pipelines["torch"], alpha)
            torch_own = _rates(boot, pipelines["torch"], pipelines["torch"], alpha)
            row["torch_thresholds"] = _cell(at_torch)
            row["d_fpr_vs_torch_at_torch_thresholds"] = compare.difference(
                at_torch["fpr"] - torch_own["fpr"],
                at_torch["fpr_boot"] - torch_own["fpr_boot"],
                MIN_EFFECT_FPR,
            )
            # torch 대비 이미지 점수의 상대 차이(중앙값, 최대). 4단계 FP32는 0.012%, 2.6%, INT8은 24.3%,
            # 165%였다.
            rel = [
                np.abs(cats[c]["eval_score"] - pipelines["torch"][c]["eval_score"])
                / np.maximum(np.abs(pipelines["torch"][c]["eval_score"]), 1e-12)
                for c in cats
            ]
            row["score_rel_diff_vs_torch"] = {
                "median": float(np.median(np.concatenate(rel))),
                "max": float(np.max(np.concatenate(rel))),
            }
        rows.append(row)

    report = {
        "alpha": alpha,
        "n_boot": boot.n_boot,
        "rows": rows,
        **{k: run["meta"][k] for k in ("artifacts", "ratio", "protocol")},
        # INT8 점수가 없는 실행(예: 모델을 양자화하지 못함. DINOv2 설정은 INT8 모델을 만들 수 없다, 규칙
        # 변경 5)이면 False가 되어, H14를 판정하지 않았다는 표시가 남는다.
        "int8_scored": "int8" in pipelines,
    }
    if "fp32" in pipelines and "int8" in pipelines:
        report["int8_at_fp32_thresholds"] = int8_shift(boot, pipelines, alpha)
    return report


def shift_verdict(fp_diff: int, ci_counts: list[float], n_neg: int) -> str:
    """등록한 H14 규칙을, 합산 정상 `n_neg`장 가운데 오검출 장수로 적용한다.

    짝지은 구간이 0을 포함하지 않고 변화가 2.0%p 이상이면 지지("지지"), 구간 전체가 (−2.0, +2.0)%p 안에
    엄격히 들어가면 기각("기각"), 그 밖에는 판정 불가("판정 불가")다. 2.0%p를 정상 수 × 0.02(Fraction으로
    정확히 계산)인 장수로 바꿔 비교하므로 기준과의 비교가 정확하다. 두 비율의 차로 계산하면 12/100 − 10/100이
    부동소수로 0.01999...가 되어 "2.0%p 이상"을 놓친다.
    """
    # 2.0%p × 정상 수 = 기준이 되는 오검출 장수(정확한 분수). 아래 비교는 모두 장수 단위다.
    margin = Fraction(str(MIN_EFFECT_FPR)) * n_neg
    lo, hi = ci_counts
    if (lo > 0 or hi < 0) and abs(fp_diff) >= margin:
        return "지지"
    if -margin < lo and hi < margin:
        return "기각"
    return "판정 불가"


def int8_shift(boot: compare.Bootstrap, pipelines: dict, alpha: float = ALPHA) -> dict:
    """fp32 CPU 파이프라인으로 보정한 임계값을 INT8 점수와 fp32 점수에 똑같이 적용해 비교한다(H14의 수치).

    등록한 주장은 합산 오검출률이 2%p 넘게 움직인다는 것이다(`shift_verdict`로 판정). 오검출 수 차이와
    짝지은 구간을 낸다. 임계값은 fp32 교차 적합 점수로 정한 것을 그대로 쓰고, INT8 자신의 교차 적합 점수로
    다시 보정했을 때의 비율과 검출률 변화는 판정 없이 옆에 적는다.
    """
    fp32 = _rates(boot, pipelines["fp32"], pipelines["fp32"], alpha)
    int8 = _rates(boot, pipelines["int8"], pipelines["fp32"], alpha)
    recal = _rates(boot, pipelines["int8"], pipelines["int8"], alpha)
    n_neg = fp32["n_neg"]
    fp_diff = int8["fp"] - fp32["fp"]
    lo, hi = compare._ci(int8["fp_boot"] - fp32["fp_boot"])
    return {
        "fp32": _cell(fp32),
        "int8_at_fp32_thresholds": _cell(int8),
        "int8_recalibrated": _cell(recal),
        "d_fpr": fp_diff / n_neg,
        "d_fpr_ci": [lo / n_neg, hi / n_neg],
        "d_tpr": float(int8["tpr"] - fp32["tpr"]),
        "d_tpr_ci": compare._ci(int8["tpr_boot"] - fp32["tpr_boot"]),
        "verdict": shift_verdict(fp_diff, [lo, hi], n_neg),
    }


def _pct(x: float) -> str:
    return f"{x * 100:.2f}"


def _span(ci: list[float]) -> str:
    return f"[{_pct(ci[0])}, {_pct(ci[1])}]"


# 콘솔용 표: 파이프라인별 AUROC, 자기 임계값의 오검출률·검출률, torch 임계값의 오검출률, torch 대비 점수 차이,
# 그리고 H14 한 줄.
def format_report(report: dict) -> str:
    lines = [
        f"{report['artifacts']} ratio {report['ratio']:g} protocol {report['protocol']}",
        "| pipeline | image AUROC | FPR, own thresholds | detection | FPR, torch thresholds "
        "| score drift (median / max) |",
        "|---|---|---|---|---|---|",
    ]
    for r in report["rows"]:
        own = r["own_thresholds"]
        cells = [
            r["pipeline"],
            f"{_pct(r['macro_image_auroc'])} {_span(r['macro_image_auroc_ci'])}",
            f"{_pct(own['fpr'])} {_span(own['fpr_ci'])}",
            f"{_pct(own['tpr'])} {_span(own['tpr_ci'])}",
        ]
        if "torch_thresholds" in r:
            drift = r["score_rel_diff_vs_torch"]
            cells += [
                f"{_pct(r['torch_thresholds']['fpr'])} {_span(r['torch_thresholds']['fpr_ci'])}",
                f"{drift['median'] * 100:.3f}% / {drift['max'] * 100:.2f}%",
            ]
        else:
            cells += ["-", "-"]
        lines.append("| " + " | ".join(cells) + " |")
    shift = report.get("int8_at_fp32_thresholds")
    if shift:
        lines += [
            "",
            f"INT8 under fp32 thresholds: FPR {_pct(shift['fp32']['fpr'])} -> "
            f"{_pct(shift['int8_at_fp32_thresholds']['fpr'])} "
            f"({shift['d_fpr'] * 100:+.2f}%p {_span(shift['d_fpr_ci'])}), "
            f"after recalibration {_pct(shift['int8_recalibrated']['fpr'])}; verdict: {shift['verdict']}",
        ]
    elif not report.get("int8_scored", True):
        lines += ["", "INT8 was not scored in this run: no verdict on the INT8 threshold shift."]
    return "\n".join(lines)


# 명령행 입구. 저장된 점수만 읽고 reports/stage4/export-<절차>.json을 쓴다.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="run_export_eval의 출력 폴더")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = build_report(load(args.run_dir))
    print(format_report(report))
    if not args.no_write:
        out_dir = paths.REPORTS / "stage4"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"export-{report['protocol']}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
