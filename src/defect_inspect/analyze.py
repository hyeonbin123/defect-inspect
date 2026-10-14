"""PatchCore 실행 하나의 표를 만든다: 정확도, 임계값 보정 방식 비교, 유한 표본 곡선.

`run_patchcore`가 범주마다 쓴 파일을 읽어 JSON 리포트와 이미지별 점수 CSV를 쓴다. 점수 CSV가 있으므로
모든 수치를 이미지 없이 다시 계산할 수 있다.
"""

# [파일 역할] PatchCore 실행 하나(run_patchcore의 출력 폴더)를 읽어 1단계 표를 만드는 분석 모듈이다(`python -m
# defect_inspect.analyze outputs/p0-test --stage stage1`). 세 부분으로 나뉜다.
# - accuracy: 범주별·평균 이미지 AUROC, AUPRO, 픽셀 AUROC, 합산 FPR@TPR95와 부트스트랩 구간.
# - calibration: 질문 2 앞부분. 정상 점수만으로 임계값을 정하는 세 방식(재대입, 홀드아웃, 교차 적합)의 실제
#   합산 오검출률·검출률과 이론 구간, 가설 H1~H4 판정.
# - finite_sample: 보정 표본 수 n에 따른 실제 오검출률(판정 없는 보조 표)과, 같은 절차를 교환 가능한 점수에
#   돌린 기준 모의.
# [설계 이유] 리포트 JSON과 함께 이미지별 점수 CSV를 남겨, 이미지 없이도 모든 수치를 다시 계산할 수 있게
# 한다(docstring). 봉인 테스트 결과(reports/stage1/p0-test.json, p0-test-scores.csv)가 커밋되어 있는 이유다.
# [연결] 입력: run_patchcore의 run.json과 범주별 npz. 계산: calibrate(임계값·이론), metrics(지표),
# stats(부트스트랩). figures가 이 리포트로 README 그림을 그린다.

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# calibrate에서는 conformal 임계값과 이론 구간·표본 수 곡선을, metrics에서는 AUROC·AUPRO(히스토그램
# 부트스트랩)·FPR@TPR95를, splits에서는 홀드아웃 겹 번호를, stats에서는 Clopper-Pearson 구간과 층화
# 부트스트랩을 가져온다.
from . import paths
from .calibrate import conformal_threshold, pooled_fpr_band, subsample_fpr_curve
from .metrics import ProHistograms, aupro_bootstrap, aupro_from_histograms, auroc, fpr_at_tpr
from .splits import holdout_fold
from .stats import (
    clopper_pearson,
    macro_auroc_bootstrap,
    percentile_ci,
    pooled_rate_bootstrap,
    stratified_indices,
)

# N_BOOT·SEED: 부트스트랩 2,000번, seed 0(공통 규칙). ALPHAS: 목표 오검출률 5%(주)와 1%(판정에 쓰지 않는
# 참고). STRATEGIES: 임계값 보정 방식 세 가지(1단계 표).
# REF_REALISATIONS·REF_DRAWS: 표본 수 곡선의 기준 모의를 100번 실현하고, 실현마다 300번 추출한다(1단계 규칙,
# 규칙 변경 2). 기준 모의는 실제와 같은 풀·테스트 크기를 가진 교환 가능한 점수로 같은 절차를 돌린 것이다.
N_BOOT = 2000
SEED = 0
ALPHAS = (0.05, 0.01)
STRATEGIES = ("resubstitution", "holdout", "crossfit")
REF_REALISATIONS = 100
REF_DRAWS = 300


# 실행 폴더의 run.json(구성, 절차, 커밋, 범주 목록)과 범주별 npz 배열을 읽는다.
def load_run(run_dir: Path) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    with open(run_dir / "run.json", encoding="utf-8") as f:
        meta = json.load(f)
    cats = {}
    for info in meta["categories"]:
        with np.load(run_dir / f"{info['category']}.npz") as z:
            cats[info["category"]] = {k: z[k] for k in z.files}
    return meta, cats


# 범주 하나에서 지정한 점수 열을 정상과 결함으로 나눈다.
def _split(cat: dict[str, np.ndarray], key: str) -> tuple[np.ndarray, np.ndarray]:
    labels = cat["eval_labels"]
    return cat[key][labels == 0], cat[key][labels == 1]


def _ci(samples: np.ndarray) -> list[float]:
    lo, hi = percentile_ci(samples)
    return [float(lo), float(hi)]


# 실행이 저장한 AUPRO 히스토그램을 ProHistograms로 묶는다. 이미지를 다시 읽지 않고 AUPRO와 그 부트스트랩을
# 계산하기 위한 것이다.
def _histograms(cat: dict[str, np.ndarray]) -> ProHistograms:
    return ProHistograms(
        edges=cat["pro_edges"],
        normal=cat["pro_normal"],
        components=cat["pro_components"],
        component_image=cat["pro_component_image"],
    )


# 정확도 표. 평가 점수는 전체 뱅크 점수(eval_score_full)가 기본이다. 이미지 AUROC는 범주별로 구해 12개 범주
# 단순 평균을 대표값으로, FPR@TPR95는 범주별 오검출 수를 합쳐 정상 전체로 나눈 합산 값으로 낸다(공통 규칙
# "지표").
def accuracy(cats: dict[str, dict[str, np.ndarray]], score_key: str = "eval_score_full") -> dict:
    """범주별 이미지 AUROC·AUPRO·픽셀 AUROC와 그 범주 평균, 그리고 TPR 95%에서의 합산 FPR을 낸다."""
    names = list(cats)
    neg = [_split(cats[c], score_key)[0] for c in names]
    pos = [_split(cats[c], score_key)[1] for c in names]
    per_cat = {}
    for c, n, p in zip(names, neg, pos, strict=True):
        fpr, _, fp = fpr_at_tpr(n, p, 0.95)
        per_cat[c] = {
            "n_normal": len(n),
            "n_defect": len(p),
            "image_auroc": float(auroc(n, p)),
            "aupro": float(cats[c]["aupro"]),
            "pixel_auroc": float(cats[c]["pixel_auroc"]),
            "fpr_at_tpr95": float(fpr),
            "fp_at_tpr95": int(fp),
        }
    macro_auroc = float(np.mean([v["image_auroc"] for v in per_cat.values()]))
    macro_aupro = float(np.mean([v["aupro"] for v in per_cat.values()]))
    # 합산 FPR@TPR95 = 범주별 오검출 수의 합 / 범주별 정상 수의 합(범주 평균이 아니라 합산, 공통 규칙).
    pooled_fp = sum(v["fp_at_tpr95"] for v in per_cat.values())
    pooled_n = sum(v["n_normal"] for v in per_cat.values())

    # 평균 AUROC의 구간. 아래 다른 통계도 같은 크기·같은 시드의 층화 추출을 써서 같은 재표집을 공유한다.
    auroc_boot = macro_auroc_bootstrap(neg, pos, n_boot=N_BOOT, seed=SEED)

    # 다른 통계에 쓸 층화 추출이다. 범주마다 정상, 그다음 결함 순서로 크기를 늘어놓는다.
    sizes = [s for n, p in zip(neg, pos, strict=True) for s in (len(n), len(p))]
    draws = stratified_indices(sizes, n_boot=N_BOOT, seed=SEED)
    aupro_boot = np.zeros(N_BOOT)
    aupro_binned = []
    fpr95_fp = np.zeros(N_BOOT)
    for i, c in enumerate(names):
        neg_idx, pos_idx = draws[2 * i], draws[2 * i + 1]
        n_neg, n_pos = len(neg[i]), len(pos[i])
        hist = _histograms(cats[c])
        # aupro_binned는 히스토그램으로 다시 계산한 AUPRO다. 저장된 정확한 AUPRO와 나란히 적어, 구간 계산에
        # 쓰는 히스토그램 근사가 점 추정값과 얼마나 다른지 확인할 수 있게 한다.
        aupro_binned.append(aupro_from_histograms(hist))
        # counts[b, j] = 재표집 b에서 평가 이미지 j(정상 먼저, 그다음 결함)가 뽑힌 횟수
        counts = np.zeros((N_BOOT, n_neg + n_pos), dtype=np.int64)
        rows = np.arange(N_BOOT)[:, None]
        np.add.at(counts, (rows, neg_idx), 1)
        np.add.at(counts, (rows, n_neg + pos_idx), 1)
        aupro_boot += aupro_bootstrap(hist, counts) / len(names)
        for b in range(N_BOOT):
            # 재표집마다 범주의 FPR@TPR95 임계값을 다시 정하고(테스트 결함으로 정하는 지표라 재표집마다
            # 달라진다) 그 오검출 수를 더한다.
            fpr95_fp[b] += fpr_at_tpr(neg[i][neg_idx[b]], pos[i][pos_idx[b]], 0.95)[2]
    return {
        "per_category": per_cat,
        "macro_image_auroc": macro_auroc,
        "macro_image_auroc_ci": _ci(auroc_boot),
        "macro_aupro": macro_aupro,
        "macro_aupro_ci": _ci(aupro_boot),
        "macro_aupro_binned": float(np.mean(aupro_binned)),
        "macro_pixel_auroc": float(np.mean([v["pixel_auroc"] for v in per_cat.values()])),
        "pooled_fpr_at_tpr95": pooled_fp / pooled_n,
        "pooled_fpr_at_tpr95_ci": _ci(fpr95_fp / pooled_n),
    }


# 보정 방식마다 (보정 점수, 실제로 운영할 모델의 평가 점수 열)을 고른다(docs/experiments.md 1단계 표).
# - 재대입: 뱅크를 만든 바로 그 이미지의 점수(pool_score_resub) + 전체 뱅크.
# - 홀드아웃: 뱅크에서 뺀 겹(test 절차 겹 0, dev 절차 겹 1)의 점수 + 그 겹을 뺀 뱅크(eval_score_holdout).
# - 교차 적합: 겹마다 그 겹을 뺀 뱅크로 매긴 점수를 모두 모은 것(pool_score_oof) + 전체 뱅크.
# 보정 점수와 평가 점수가 같은 모델에서 나와야 임계값이 그 모델의 운영 오검출률을 맞춘다.
def _strategy_inputs(cat: dict[str, np.ndarray], strategy: str, protocol: str) -> tuple[np.ndarray, str]:
    """보정 점수와, 그 방식으로 실제 운영할 모델이 내는 평가 점수 열의 이름을 돌려준다."""
    if strategy == "resubstitution":
        return cat["pool_score_resub"], "eval_score_full"
    if strategy == "holdout":
        return cat["pool_score_oof"][cat["pool_folds"] == holdout_fold(protocol)], "eval_score_holdout"
    if strategy == "crossfit":
        return cat["pool_score_oof"], "eval_score_full"
    raise ValueError(strategy)


# 목표 alpha에서 세 보정 방식의 실제 성능. 범주마다 임계값을 정해 테스트 정상(오검출)과 결함(검출)을 표시하고,
# 12개 범주를 합쳐 비율을 낸다. 오검출률에는 세 가지 구간을 붙인다: 이미지 부트스트랩 구간, Clopper-Pearson
# 정확 구간, 그리고 교환 가능할 때의 이론 구간(pooled_fpr_band, Beta-Binomial 모의). 결과: 재대입 86.0%,
# 홀드아웃 5.56%(이론 구간 3.56~6.34 안), 교차 적합 3.98%(README).
def calibration(cats: dict[str, dict[str, np.ndarray]], protocol: str, alpha: float) -> dict:
    """정상 점수만으로 고정한 임계값의 실제 합산 오검출률과 검출률을 낸다."""
    out: dict[str, dict] = {}
    flags: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for strategy in STRATEGIES:
        fp_flags, tp_flags, n_cal, per_cat = [], [], [], {}
        guaranteed = 0
        for c, cat in cats.items():
            cal, key = _strategy_inputs(cat, strategy, protocol)
            thr = conformal_threshold(cal, alpha)
            neg, pos = _split(cat, key)
            # 규칙: 점수 > 임계값이면 불량. 이미지별 표시(불리언)를 범주별로 모아 두고, 아래에서 합산 비율과
            # 부트스트랩 구간을 낸다.
            fp_flags.append(neg > thr.value)
            tp_flags.append(pos > thr.value)
            n_cal.append(len(cal))
            guaranteed += int(thr.guaranteed)
            per_cat[c] = {
                "threshold": float(thr.value),
                "n_cal": len(cal),
                "guaranteed": bool(thr.guaranteed),
                "fp": int(fp_flags[-1].sum()),
                "n_normal": len(neg),
                "tp": int(tp_flags[-1].sum()),
                "n_defect": len(pos),
            }
        fp, n_neg = sum(int(f.sum()) for f in fp_flags), sum(len(f) for f in fp_flags)
        tp, n_pos = sum(int(f.sum()) for f in tp_flags), sum(len(f) for f in tp_flags)
        # 이론 구간: 범주별 보정 점수 수와 테스트 정상 수로 Beta-Binomial 합산 오검출률을 100,000번 모의한
        # 가운데 95%.
        band = pooled_fpr_band(n_cal, [len(f) for f in fp_flags], alpha)
        out[strategy] = {
            "fpr": fp / n_neg,
            "fp": fp,
            "n_normal": n_neg,
            "fpr_ci": _ci(pooled_rate_bootstrap(fp_flags, n_boot=N_BOOT, seed=SEED)),
            "fpr_clopper_pearson": [float(x) for x in clopper_pearson(fp, n_neg)],
            "theory_band": [float(band[0]), float(band[1])],
            "theory_mean": float(band[2]),
            "tpr": tp / n_pos,
            "tp": tp,
            "n_defect": n_pos,
            "tpr_ci": _ci(pooled_rate_bootstrap(tp_flags, n_boot=N_BOOT, seed=SEED)),
            "categories_guaranteed": guaranteed,
            "n_cal_total": int(sum(n_cal)),
            "per_category": per_cat,
        }
        flags[strategy] = (fp_flags, tp_flags)

    # H4용: 교차 적합 − 홀드아웃의 검출률 차이. 두 호출이 같은 결함 이미지에 같은 시드로 같은 재표집을 쓰므로
    # 짝지은 차이가 된다.
    diff = pooled_rate_bootstrap(flags["crossfit"][1], n_boot=N_BOOT, seed=SEED) - pooled_rate_bootstrap(
        flags["holdout"][1], n_boot=N_BOOT, seed=SEED
    )
    out["tpr_crossfit_minus_holdout"] = {
        "point": out["crossfit"]["tpr"] - out["holdout"]["tpr"],
        "ci": _ci(diff),
    }
    return out


# 1단계 가설 판정(등록한 문장 그대로, α = 5%).
# - H1 재대입 오검출률 ≥ 목표의 두 배: 구간 하한 ≥ 10%면 지지, 상한 < 10%면 기각.
# - H2 홀드아웃이 이론 구간 안: 점 추정값이 구간 안이면 지지, 밖이면 기각하고 방향을 적는다.
# - H3 교차 적합이 목표를 넘지 않음: 상한 ≤ 5%면 지지, 하한 > 5%면 기각.
# - H4 교차 적합 검출률이 홀드아웃보다 낮지 않음: 차이 하한 > −2.0%p면 지지, 상한 < −2.0%p면 기각(결과: 판정
#   불가).
def hypotheses(cal: dict, alpha: float = 0.05) -> dict[str, str]:
    """1단계 가설 H1~H4를 docs/experiments.md에 등록한 문장 그대로 판정한다."""

    def three_way(support: bool, reject: bool) -> str:
        return "지지" if support else "기각" if reject else "판정 불가"

    resub, hold, cross = cal["resubstitution"], cal["holdout"], cal["crossfit"]
    # H2는 점 추정값이 이론 구간 안인지로 판정하고, 밖이면 높은 쪽인지 낮은 쪽인지 적는다. 나머지는 부트스트랩
    # 구간과 기준값을 비교한다.
    lo, hi = hold["theory_band"]
    if lo <= hold["fpr"] <= hi:
        h2 = "지지"
    else:
        h2 = "기각 (이론 구간보다 " + ("높음" if hold["fpr"] > hi else "낮음") + ")"
    d_lo, d_hi = cal["tpr_crossfit_minus_holdout"]["ci"]
    return {
        "H1": three_way(resub["fpr_ci"][0] >= 2 * alpha, resub["fpr_ci"][1] < 2 * alpha),
        "H2": h2,
        "H3": three_way(cross["fpr_ci"][1] <= alpha, cross["fpr_ci"][0] > alpha),
        "H4": three_way(d_lo > -0.02, d_hi < -0.02),
    }


def _curve_means(curves: list[list[dict]]) -> dict[int, float]:
    """범주별 곡선을 보정 표본 수 n마다 합친다: 각 범주의 평균 오검출률을 범주에 걸쳐 단순 평균한다."""
    by_n: dict[int, list[float]] = {}
    for curve in curves:
        for r in curve:
            by_n.setdefault(int(r["n"]), []).append(float(r["mean"]))
    return {n: float(np.mean(v)) for n, v in by_n.items()}


# 보정 표본 수 n(20, 50, 100, 200)에 따른 실제 오검출률. 범주마다 교차 적합 점수에서 n개를 뽑아 임계값을
# 정하고 그 범주의 평가 정상에 적용한다.
# 기준 모의(규칙 변경 2)는 100번 돌려 범주 평균의 2.5~97.5 백분위를 기준 구간으로 삼는다. 결과: 목표 5%에서
# n ≥ 100이면 관측이 기준 구간보다 낮아, 교차 적합 점수가 보수 쪽으로 치우쳐 있음을 보였다.
def finite_sample(cats: dict[str, dict[str, np.ndarray]], alpha: float) -> list[dict]:
    """보정 표본 수 n에 따른 테스트 오검출률을 범주에 걸쳐 평균해 낸다.

    범주마다 곡선은 교차 적합 점수(크기가 정해진 풀)에서 부분표본을 뽑아 임계값을 정하고, 그 범주의 평가
    정상(고정된 집합)에 적용해 얻는다. 그래서 이론의 Beta 분포가 곡선의 퍼짐이나 평균을 정확히 설명하지
    못한다. 비교할 수 있는 기준은 같은 풀 크기·평가 크기를 가진 교환 가능한 점수(iid 균등 난수)에 같은 절차를
    돌린 모의다. 범주 평균한 관측값이 이 기준 구간 밖이면, 그 벗어남은 유한 표본의 흔들림보다 크다.
    """
    observed, sizes = [], []
    for cat in cats.values():
        neg, _ = _split(cat, "eval_score_full")
        # 관측: 범주마다 교차 적합 점수 풀에서 n개 부분표본을 뽑아 임계값을 정하고, 그 범주의 평가 정상(전체
        # 뱅크 점수)에 적용한다.
        observed.append(subsample_fpr_curve(cat["pool_score_oof"], neg, alpha))
        sizes.append((len(cat["pool_score_oof"]), len(neg)))

    # 기준 모의: 실현 r마다 범주별로 균등 난수 풀과 평가 정상을 새로 만들고, 추출 시드도 r로 바꿔 독립인 실현
    # 100개를 만든다.
    rng = np.random.default_rng(SEED)
    reference: dict[int, list[float]] = {}
    for r in range(REF_REALISATIONS):
        curves = [
            subsample_fpr_curve(rng.random(pool), rng.random(m), alpha, draws=REF_DRAWS, seed=r)
            for pool, m in sizes
        ]
        for n, value in _curve_means(curves).items():
            reference.setdefault(n, []).append(value)

    rows: dict[int, list[dict]] = {}
    for curve in observed:
        for r in curve:
            rows.setdefault(int(r["n"]), []).append(r)
    keys = ("mean", "p5", "p95", "theory_mean", "theory_p5", "theory_p95")
    out = []
    for n, rs in sorted(rows.items()):
        ref = np.array(reference[n])
        lo, hi = percentile_ci(ref)
        row = {
            "n": n,
            "categories": len(rs),
            "guaranteed": all(bool(r["guaranteed"]) for r in rs),
            **{k: float(np.mean([r[k] for r in rs])) for k in keys},
            "reference_mean": float(ref.mean()),
            "reference_interval": [float(lo), float(hi)],
        }
        # 관측 평균이 기준 모의 구간 안이면 유한 표본의 흔들림으로 설명되고, 밖이면 치우침이다.
        row["within_reference"] = bool(lo <= row["mean"] <= hi)
        out.append(row)
    return out


# 이미지별 점수 CSV(리포트 옆 `<구성>-<절차>-scores.csv`). 평가 이미지는 전체 뱅크·홀드아웃 뱅크 점수를, 정상
# 풀 이미지는 겹 번호와 재대입·교차 적합 점수를 적는다. 이 파일만으로 세 보정 방식을 이미지 없이 다시 계산할
# 수 있다.
def write_scores_csv(cats: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(
            [
                "category",
                "set",
                "image",
                "label",
                "fold",
                "score_full",
                "score_holdout",
                "score_resub",
                "score_oof",
            ]
        )
        for c, cat in cats.items():
            for i, image in enumerate(cat["eval_images"]):
                label = "anomaly" if cat["eval_labels"][i] else "normal"
                full, hold = cat["eval_score_full"][i], cat["eval_score_holdout"][i]
                w.writerow([c, "eval", image, label, "", f"{full:.6f}", f"{hold:.6f}", "", ""])
            for i, image in enumerate(cat["pool_images"]):
                resub, oof = cat["pool_score_resub"][i], cat["pool_score_oof"][i]
                w.writerow(
                    [
                        c,
                        "pool",
                        image,
                        "normal",
                        int(cat["pool_folds"][i]),
                        "",
                        "",
                        f"{resub:.6f}",
                        f"{oof:.6f}",
                    ]
                )


# 리포트 조립: 정확도(전체 뱅크), 홀드아웃 뱅크의 평균 AUROC(참고), 목표 5%·1%의 보정 표, 표본 수 곡선, 그리고
# 5%에서의 H1~H4 판정.
def build_report(run_dir: Path) -> dict:
    meta, cats = load_run(run_dir)
    protocol = meta["protocol"]
    report = {
        "config": meta["config"],
        "protocol": protocol,
        "commit": meta["commit"],
        "n_boot": N_BOOT,
        "accuracy": accuracy(cats),
        # 홀드아웃 방식이 실제로 운영하는 모델(겹 하나를 뺀 뱅크)의 평균 AUROC. 보정 방식마다 운영하는 뱅크가
        # 다르다는 것을 참고로 남긴다.
        "accuracy_holdout_model": {
            "macro_image_auroc": float(
                np.mean([auroc(*_split(cat, "eval_score_holdout")) for cat in cats.values()])
            )
        },
        "calibration": {str(a): calibration(cats, protocol, a) for a in ALPHAS},
        "finite_sample": {str(a): finite_sample(cats, a) for a in ALPHAS},
    }
    report["hypotheses"] = hypotheses(report["calibration"]["0.05"], 0.05)
    return report


def _pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}"


def _span(ci: list[float], digits: int = 2) -> str:
    return f"[{_pct(ci[0], digits)}, {_pct(ci[1], digits)}]"


# 콘솔용 마크다운 표(정확도 표, 목표별 보정 표, 가설, 표본 수 곡선). docs/experiments.md 1단계 결과 표의
# 원본이다.
def format_report(report: dict) -> str:
    acc = report["accuracy"]
    lines = [
        f"config {report['config']['name']}  protocol {report['protocol']}  commit {report['commit']}",
        "",
        "| category | normal | defect | image AUROC | AUPRO | pixel AUROC | FPR@TPR95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for c, v in acc["per_category"].items():
        cells = [
            c,
            str(v["n_normal"]),
            str(v["n_defect"]),
            _pct(v["image_auroc"]),
            _pct(v["aupro"]),
            _pct(v["pixel_auroc"]),
            _pct(v["fpr_at_tpr95"], 1),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    cells = [
        "**mean / pooled**",
        "",
        "",
        f"{_pct(acc['macro_image_auroc'])} {_span(acc['macro_image_auroc_ci'])}",
        f"{_pct(acc['macro_aupro'])} {_span(acc['macro_aupro_ci'])}",
        _pct(acc["macro_pixel_auroc"]),
        f"{_pct(acc['pooled_fpr_at_tpr95'], 1)} {_span(acc['pooled_fpr_at_tpr95_ci'], 1)}",
    ]
    lines += ["| " + " | ".join(cells) + " |", ""]
    header = ["strategy", "n cal", "actual FPR", "bootstrap CI", "Clopper-Pearson", "theory band"]
    header += ["detection rate", "CI"]
    for alpha, cal in report["calibration"].items():
        lines += [
            f"target FPR {float(alpha) * 100:g}%",
            "| " + " | ".join(header) + " |",
            "|" + "---|" * len(header),
        ]
        for s in STRATEGIES:
            v = cal[s]
            cells = [
                s,
                str(v["n_cal_total"]),
                f"{_pct(v['fpr'])}% ({v['fp']}/{v['n_normal']})",
                _span(v["fpr_ci"]),
                _span(v["fpr_clopper_pearson"]),
                _span(v["theory_band"]),
                f"{_pct(v['tpr'])}% ({v['tp']}/{v['n_defect']})",
                _span(v["tpr_ci"]),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        d = cal["tpr_crossfit_minus_holdout"]
        lines += [
            f"detection rate, crossfit - holdout: {d['point'] * 100:+.2f}%p "
            f"[{d['ci'][0] * 100:+.2f}, {d['ci'][1] * 100:+.2f}]",
            "",
        ]
    lines.append("hypotheses (alpha 5%): " + ", ".join(f"{k} {v}" for k, v in report["hypotheses"].items()))
    lines.append("")
    for alpha, rows in report["finite_sample"].items():
        lines += [
            f"finite-sample curve, target {float(alpha) * 100:g}% (mean over categories)",
            "| n | guaranteed | mean FPR | reference mean | reference 95% | within | Beta mean |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            cells = [
                str(r["n"]),
                str(r["guaranteed"]),
                _pct(r["mean"]) + "%",
                _pct(r["reference_mean"]) + "%",
                _span(r["reference_interval"]),
                "yes" if r["within_reference"] else "no",
                _pct(r["theory_mean"]) + "%",
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


# 명령행 입구. 저장된 점수만 읽으므로 봉인 테스트 실행을 분석해도 이미지를 다시 읽지 않는다.
# reports/<stage>/에 `<구성>-<절차>.json`과 점수 CSV를 쓴다.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--stage", default="stage1", help="리포트를 쓸 reports/ 아래 하위 폴더")
    parser.add_argument("--no-write", action="store_true", help="파일을 쓰지 않고 표만 출력한다")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # cp949 콘솔에서도 한국어 판정 문구가 깨지지 않게 한다
    report = build_report(args.run_dir)
    print(format_report(report))
    if args.no_write:
        return
    out_dir = paths.REPORTS / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report['config']['name']}-{report['protocol']}"
    with open(out_dir / f"{stem}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    # 이미지별 점수 CSV를 쓰려고 배열을 다시 읽는다.
    _, cats = load_run(args.run_dir)
    write_scores_csv(cats, out_dir / f"{stem}-scores.csv")


if __name__ == "__main__":
    main()
