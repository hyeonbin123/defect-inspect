"""Stage 2 comparison: unsupervised methods against each other and against the supervised head.

Every method is loaded into one shape (per category: evaluation labels, image scores, scores of held-out
normals for the threshold, pixel metrics), then compared with the same stratified bootstrap draws so that
differences are paired. Hypotheses H5-H8 are judged exactly as registered in docs/experiments.md.
"""

# [파일 역할] 2단계 비교 분석이다(`python -m defect_inspect.compare`). 비지도 방법(P0, D-S, DM)끼리, 그리고
# 결함 라벨 k장으로 학습한 지도 학습 헤드와 기준 비지도 방법을 비교해 가설 H5~H8을 등록한 규칙대로 판정한다.
# 질문 1("결함 라벨 몇 장부터 지도 학습이 이기나")의 답이 여기서 나온다.
# [설계 이유] 모든 방법을 같은 모양(범주마다 평가 라벨, 이미지 점수, 임계값용 정상 점수, 픽셀 지표)으로 읽고,
# 모든 통계에 같은 부트스트랩 추출(범주 안 정상·결함 층화, 2,000번, seed 0)을 써서 방법 사이의 차이가 짝지어진
# 비교가 되게 한다(공통 규칙 "신뢰구간과 판정"). 그래서 두 방법의 차이를 재표집마다 같은 이미지 묶음 위에서 잴
# 수 있다.
# [연결] 입력은 run_patchcore(P0, D-S), run_dinomaly(DM), run_supervised(sup-<절차>/k<k>-s<시드>)의 실행
# 폴더이고, 결과는 reports/stage2/compare-<절차>.json이다. MethodRun, load_common, Bootstrap, summarize,
# difference는 analyze_grid, analyze_export, analyze_stage6이 그대로 가져다 쓴다.

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# metrics는 이미지 AUROC, AUPRO(히스토그램으로 부트스트랩까지), FPR@TPR95를, stats는 층화 부트스트랩 인덱스와
# 백분위 구간을 준다. 지표 코드는 scikit-learn·전수 계산과 대조하는 단위 테스트로 확인했다(1단계 구현 점검).
from . import paths
from .calibrate import conformal_threshold
from .metrics import ProHistograms, aupro_bootstrap, auroc, fpr_at_tpr
from .stats import percentile_ci, stratified_indices

# 판정 상수(docs/experiments.md 공통 규칙과 2단계 규칙).
# - N_BOOT, SEED: 부트스트랩 2,000번, seed 0. ALPHA: 고정 임계값의 목표 오검출률 5%.
# - MIN_EFFECT_AUROC: "차이 있음"의 최소 효과(이미지 AUROC 1.0%p).
# - MIN_UNSEEN_PER_CATEGORY, MIN_UNSEEN_POOLED: H7(보지 못한 결함 유형)에서 남은 결함이 3장 미만인 범주는
#   빼고, 12개 범주 합이(시드 평균) 30장 미만인 k는 판정하지 않는다.
N_BOOT = 2000
SEED = 0
ALPHA = 0.05
MIN_EFFECT_AUROC = 0.01
MIN_UNSEEN_PER_CATEGORY = 3
MIN_UNSEEN_POOLED = 30


# 비교할 방법 하나의 실행 결과. calibration은 cal_score를 어떻게 얻었는지(crossfit: PatchCore 교차 적합,
# holdout: 겹 0 정상, none: 없음)이고, cats는 범주 이름 → npz 배열들이다.
@dataclass
class MethodRun:
    name: str
    calibration: str  # how cal_score was obtained: "crossfit", "holdout" or "none"
    cats: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# run_patchcore 실행을 읽는다. 평가 점수는 전체 뱅크 점수(eval_score_full), 보정 점수는 교차 적합
# 점수(pool_score_oof)로 이름을 맞춘다. 2단계 규칙상 PatchCore 계열의 임계값은 교차 적합으로 정한다.
def load_patchcore(run_dir: Path, name: str | None = None) -> MethodRun:
    """A `run_patchcore` run: the full-bank model with its cross-fitted calibration scores."""
    with open(run_dir / "run.json", encoding="utf-8") as f:
        meta = json.load(f)
    run = MethodRun(name or meta["config"]["name"], "crossfit")
    for info in meta["categories"]:
        cat = _load_npz(run_dir / f"{info['category']}.npz")
        cat["eval_score"] = cat["eval_score_full"]
        cat["cal_score"] = cat["pool_score_oof"]
        run.cats[info["category"]] = cat
    return run


# 2단계 공통 형식(eval_*, cal_score 키를 가진 npz)의 실행을 읽는다. Dinomaly, 지도 학습의 (k, 시드) 폴더 하나,
# ONNX 채점 결과가 이 형식이다. 기본은 홀드아웃이고, 모든 범주의 cal_score가 비어 있으면(dev 채점 등) 고정
# 임계값을 계산하지 않도록 none으로 둔다.
def load_common(run_dir: Path, name: str, categories: list[str] | None = None) -> MethodRun:
    """A run in the common stage 2 format (Dinomaly, one supervised (k, seed) folder)."""
    files = sorted(p for p in run_dir.glob("*.npz"))
    names = categories or [p.stem for p in files]
    run = MethodRun(name, "holdout")
    for c in names:
        run.cats[c] = _load_npz(run_dir / f"{c}.npz")
    if all(len(cat["cal_score"]) == 0 for cat in run.cats.values()):
        run.calibration = "none"
    return run


# 범주 하나의 평가 점수를 정상(라벨 0)과 결함(라벨 1)으로 나눈다.
def _neg_pos(cat: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    labels = cat["eval_labels"]
    return cat["eval_score"][labels == 0], cat["eval_score"][labels == 1]


# 짝지은 비교의 전제 확인: 모든 실행이 같은 범주를 같은 순서로, 범주마다 같은 평가 이미지를 같은 순서로
# 채점했는지 본다. 다르면 같은 부트스트랩 인덱스가 다른 이미지를 가리키게 되므로 바로 오류를 낸다.
def check_same_images(runs: list[MethodRun]) -> list[str]:
    """All runs must score the same evaluation images in the same order; returns the category list."""
    names = list(runs[0].cats)
    for run in runs[1:]:
        if list(run.cats) != names:
            raise ValueError(f"{run.name} has categories {list(run.cats)}, expected {names}")
        for c in names:
            if not np.array_equal(run.cats[c]["eval_images"], runs[0].cats[c]["eval_images"]):
                raise ValueError(f"{run.name} and {runs[0].name} evaluate different images in {c}")
    return names


# 모든 통계가 함께 쓰는 층화 부트스트랩 추출 한 벌. 기준 실행의 범주별 정상 수·결함 수로
# stats.stratified_indices를 한 번 불러, 범주 i의 정상 인덱스 draws[2i]와 결함 인덱스 draws[2i+1]([n_boot,
# 크기])을 만든다. 같은 크기와 시드면 항상 같은 인덱스라, 어느 방법에 적용해도 같은 이미지 묶음을 뽑는다.
class Bootstrap:
    """One set of stratified draws (normals and defects of each category) shared by every statistic."""

    def __init__(self, reference: MethodRun, n_boot: int | None = None, seed: int = SEED):
        n_boot = n_boot or N_BOOT
        self.names = list(reference.cats)
        self.n_boot = n_boot
        self.sizes = []
        for c in self.names:
            labels = reference.cats[c]["eval_labels"]
            self.sizes += [int((labels == 0).sum()), int((labels == 1).sum())]
        self.draws = stratified_indices(self.sizes, n_boot=n_boot, seed=seed)

    def indices(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return self.draws[2 * i], self.draws[2 * i + 1]

    # 재표집마다 12개 범주 이미지 AUROC의 단순 평균(대표값)을 구해 [n_boot] 표본을 돌려준다. keep_pos를 주면
    # 그 범주에서 남길 결함만 쓴다(H7의 보지 못한 유형). 남길 결함이 없는 범주는 건너뛰고, 어떤 재표집에서
    # 남길 결함이 하나도 뽑히지 않으면 그 재표집에서만 그 범주를 뺀다.
    def macro_auroc(self, run: MethodRun, keep_pos: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Bootstrap samples of the mean over categories of the image AUROC.

        `keep_pos[c]` (bool over the category's defects) restricts the defects; categories it leaves out
        entirely are skipped, and a resample that draws none of the kept defects skips that category.
        """
        total = np.zeros(self.n_boot)
        count = np.zeros(self.n_boot)
        for i, c in enumerate(self.names):
            if keep_pos is not None and c not in keep_pos:
                continue
            neg, pos = _neg_pos(run.cats[c])
            neg_idx, pos_idx = self.indices(i)
            for b in range(self.n_boot):
                p_idx = pos_idx[b]
                if keep_pos is not None:
                    p_idx = p_idx[keep_pos[c][p_idx]]
                    if len(p_idx) == 0:
                        continue
                # 재표집 b에서 뽑힌 정상과 (남길) 결함으로 범주 AUROC를 구해 더하고, 더한 범주 수(count)로
                # 나눠 평균한다.
                total[b] += auroc(neg[neg_idx[b]], pos[p_idx])
                count[b] += 1
        # 어떤 재표집에서 모든 범주가 빠졌으면(count 0) 0으로 나누지 않고 nan으로 둔다. 구간
        # 계산(percentile_ci)은 nan을 빼고 한다.
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, total / count, np.nan)

    # 재표집마다의 평균 AUPRO. 픽셀 점수를 다시 다루지 않고, 실행이 저장한 히스토그램(정상 픽셀 점수 분포와
    # 결함 영역별 분포)에 "각 이미지가 몇 번 뽑혔는지" 가중치를 곱해 다시 적분한다. 2,000번 재표집을 픽셀
    # 수만큼 반복하지 않아도 된다.
    def macro_aupro(self, run: MethodRun) -> np.ndarray:
        out = np.zeros(self.n_boot)
        rows = np.arange(self.n_boot)[:, None]
        for i, c in enumerate(self.names):
            cat = run.cats[c]
            hist = ProHistograms(
                edges=cat["pro_edges"],
                normal=cat["pro_normal"],
                components=cat["pro_components"],
                component_image=cat["pro_component_image"],
            )
            neg_idx, pos_idx = self.indices(i)
            n_neg, n_pos = self.sizes[2 * i], self.sizes[2 * i + 1]
            counts = np.zeros((self.n_boot, n_neg + n_pos), dtype=np.int64)
            # counts[b, j] = 재표집 b에서 평가 이미지 j(정상 다음 결함 순서)가 뽑힌 횟수. 복원 추출이라 같은
            # 인덱스가 여러 번 나오므로, 중복도 모두 더하는 np.add.at을 쓴다.
            np.add.at(counts, (rows, neg_idx), 1)
            np.add.at(counts, (rows, n_neg + pos_idx), 1)
            out += aupro_bootstrap(hist, counts) / len(self.names)
        return out

    # 고정 임계값의 합산 오검출률과 검출률. 범주마다 cal_score로 conformal 임계값(목표 alpha)을 정하고, 점수 >
    # 임계값을 불량으로 보아 12개 범주의 오검출·검출 수를 더해 비율을 낸다(공통 규칙: 합산). 구간은 같은
    # 부트스트랩 인덱스로 센 수에서 낸다. 보장 조건을 채운 범주 수도 적는다. 보정 점수가 없는 실행이면 None.
    def pooled_rates(self, run: MethodRun, alpha: float) -> dict | None:
        """Pooled false positive and detection rates of per-category thresholds fixed from `cal_score`."""
        if run.calibration == "none":
            return None
        fp = tp = n_neg = n_pos = 0
        fp_boot = np.zeros(self.n_boot)
        tp_boot = np.zeros(self.n_boot)
        guaranteed = 0
        for i, c in enumerate(self.names):
            cat = run.cats[c]
            thr = conformal_threshold(cat["cal_score"], alpha)
            guaranteed += int(thr.guaranteed)
            neg, pos = _neg_pos(cat)
            f_neg, f_pos = neg > thr.value, pos > thr.value
            neg_idx, pos_idx = self.indices(i)
            fp += int(f_neg.sum())
            tp += int(f_pos.sum())
            n_neg += len(neg)
            n_pos += len(pos)
            # f_neg[neg_idx]는 [n_boot, 정상 수] 모양이라, 행마다 더하면 재표집마다의 오검출 수가 된다. 범주를
            # 돌며 더해 합산 값을 만든다.
            fp_boot += f_neg[neg_idx].sum(axis=1)
            tp_boot += f_pos[pos_idx].sum(axis=1)
        return {
            "calibration": run.calibration,
            "n_cal": int(sum(len(cat["cal_score"]) for cat in run.cats.values())),
            "categories_guaranteed": guaranteed,
            "fpr": fp / n_neg,
            "fpr_ci": _ci(fp_boot / n_neg),
            "tpr": tp / n_pos,
            "tpr_ci": _ci(tp_boot / n_pos),
        }


# 부트스트랩 표본의 95% 백분위 구간(2.5~97.5)을 [하한, 상한] 목록으로. nan은 빼고 계산한다.
def _ci(samples: np.ndarray) -> list[float]:
    lo, hi = percentile_ci(samples)
    return [float(lo), float(hi)]


# 재표집 없이 원래 평가 세트에서 구한 평균 이미지 AUROC(점 추정값). keep_pos 의미는 Bootstrap.macro_auroc와
# 같다.
def macro_auroc(run: MethodRun, keep_pos: dict[str, np.ndarray] | None = None) -> float:
    values = []
    for c, cat in run.cats.items():
        neg, pos = _neg_pos(cat)
        if keep_pos is not None:
            if c not in keep_pos:
                continue
            pos = pos[keep_pos[c]]
        values.append(auroc(neg, pos))
    return float(np.mean(values)) if values else float("nan")


# 방법 하나의 요약: 평균 이미지 AUROC·AUPRO와 구간, 평균 픽셀 AUROC, 합산 FPR@TPR95, 목표 5% 고정 임계값의
# 합산 오검출률·검출률, 범주별 수치. FPR@TPR95는 범주마다 결함의 95%를 잡는 가장 높은 임계값에서의 오검출 수를
# 12개 범주에서 더해 정상 전체 수로 나눈다. 테스트 라벨로 임계값을 정하므로 운영 수치가 아니라 분리도의
# 요약이다(공통 규칙 "지표").
def summarize(run: MethodRun, boot: Bootstrap) -> dict:
    per_cat = {}
    fp95 = n_neg = 0
    for c, cat in run.cats.items():
        neg, pos = _neg_pos(cat)
        # 범주마다 결함 95%를 잡는 임계값에서의 오검출 수를 모아 두었다가, 아래에서 정상 전체 수로 나눠 합산
        # FPR@TPR95를 만든다.
        fpr, _, fp = fpr_at_tpr(neg, pos, 0.95)
        fp95 += fp
        n_neg += len(neg)
        per_cat[c] = {
            "image_auroc": float(auroc(neg, pos)),
            "aupro": float(cat["aupro"]),
            "pixel_auroc": float(cat["pixel_auroc"]),
            "fpr_at_tpr95": float(fpr),
        }
    return {
        "name": run.name,
        "macro_image_auroc": float(np.mean([v["image_auroc"] for v in per_cat.values()])),
        "macro_image_auroc_ci": _ci(boot.macro_auroc(run)),
        "macro_aupro": float(np.mean([v["aupro"] for v in per_cat.values()])),
        "macro_aupro_ci": _ci(boot.macro_aupro(run)),
        "macro_pixel_auroc": float(np.mean([v["pixel_auroc"] for v in per_cat.values()])),
        "pooled_fpr_at_tpr95": fp95 / n_neg,
        "fixed_threshold": boot.pooled_rates(run, ALPHA),
        "per_category": per_cat,
    }


# 짝지은 차이와 등록된 세 갈래 문구. 구간이 0을 포함하면 "판정 불가", 포함하지 않고 크기가 최소 효과 이상이면
# "차이 있음", 최소 효과보다 작으면 "차이 작음"이다(공통 규칙).
def difference(point: float, samples: np.ndarray, min_effect: float = MIN_EFFECT_AUROC) -> dict:
    """Paired difference with its interval and the registered three-way wording."""
    lo, hi = _ci(samples)
    if not (lo > 0 or hi < 0):
        verdict = "판정 불가"
    elif abs(point) >= min_effect:
        verdict = "차이 있음"
    else:
        verdict = "차이 작음"
    return {"diff": float(point), "ci": [lo, hi], "verdict": verdict}


# H7용: 지도 학습 실행(k, 시드)마다, 학습에 쓴 결함 k장의 유형과 하나도 겹치지 않는 테스트 결함을 표시한다.
# 유형은 "a|b"처럼 '|'로 이어진 문자열이다(이미지 하나에 유형이 여럿일 수 있음, VisA image_anno.csv). 그런
# 결함이 3장 미만인 범주는 빼고, 남은 결함의 총수도 돌려준다.
def unseen_masks(sup: MethodRun) -> tuple[dict[str, np.ndarray], int]:
    """Per category: which test defects share no defect type with the run's training defects.

    Categories with fewer than MIN_UNSEEN_PER_CATEGORY such defects are left out. Also returns the number
    of unseen defects kept over all categories.
    """
    masks, total = {}, 0
    for c, cat in sup.cats.items():
        seen: set[str] = set()
        for entry in cat["train_defect_types"]:
            seen.update(t for t in str(entry).split("|") if t)
        types = cat["eval_defect_types"][cat["eval_labels"] == 1]
        mask = np.array([not (set(str(t).split("|")) & seen) for t in types], dtype=bool)
        if mask.sum() >= MIN_UNSEEN_PER_CATEGORY:
            masks[c] = mask
            total += int(mask.sum())
    return masks, total


# k마다 지도 학습(시드 3개 평균) − 기준 비지도 방법의 차이. 시드 평균은 점 추정값과 부트스트랩 표본 모두에서
# 낸다(같은 추출이라 표본끼리 평균해도 짝이 유지된다). labels_with_validation은 학습 k장 + 검증 20장(dev
# 결함)으로 라벨 비용을 적는다(공통 규칙 "라벨 회계").
def supervised_vs_reference(
    sup_runs: dict[tuple[int, int], MethodRun], reference: MethodRun, boot: Bootstrap
) -> dict[int, dict]:
    """Per k: supervised (mean over seeds) minus the reference method, on all defects and on unseen types."""
    ref_point = macro_auroc(reference)
    ref_boot = boot.macro_auroc(reference)
    out = {}
    for k in sorted({k for k, _ in sup_runs}):
        seeds = sorted(s for kk, s in sup_runs if kk == k)
        points = [macro_auroc(sup_runs[k, s]) for s in seeds]
        # 시드 3개의 재표집 AUROC를 재표집마다 평균한다. 기준 방법의 같은 재표집 값을 빼면 짝지은 차이 표본이
        # 된다.
        boots = np.mean([boot.macro_auroc(sup_runs[k, s]) for s in seeds], axis=0)
        row = {
            "k": k,
            "labels_with_validation": k + 20,
            "seeds": seeds,
            "supervised_macro_image_auroc": float(np.mean(points)),
            "supervised_per_seed": [float(p) for p in points],
            "supervised_ci": _ci(boots),
            "reference_macro_image_auroc": ref_point,
            "all_defects": difference(float(np.mean(points)) - ref_point, boots - ref_boot),
        }

        # 보지 못한 유형의 부분집합은 시드마다(학습 결함이 달라) 다르므로, 시드마다 그 부분집합에서 지도
        # 학습과 기준 방법을 함께 채점하고 차이를 시드 평균한다. 합계가 30장 이상일 때만 판정한다(k = 20은
        # 8장, 40은 0장이라 판정하지 않았다).
        # Unseen defect types: the subset depends on the run, the reference is scored on the same subset.
        sup_pts, ref_pts, sup_b, ref_b, kept, cats_used = [], [], [], [], [], []
        for s in seeds:
            masks, total = unseen_masks(sup_runs[k, s])
            kept.append(total)
            cats_used.append(len(masks))
            if not masks:
                continue
            sup_pts.append(macro_auroc(sup_runs[k, s], masks))
            ref_pts.append(macro_auroc(reference, masks))
            sup_b.append(boot.macro_auroc(sup_runs[k, s], masks))
            ref_b.append(boot.macro_auroc(reference, masks))
        unseen = {"unseen_defects_mean": float(np.mean(kept)), "categories_mean": float(np.mean(cats_used))}
        if np.mean(kept) >= MIN_UNSEEN_POOLED and sup_pts:
            # 시드마다의 짝지은 차이 표본을 평균한다. 어떤 재표집에서 남길 결함이 하나도 뽑히지 않은 시드는
            # nan이라 nanmean으로 그 시드만 뺀다.
            diff_boot = np.nanmean(np.array(sup_b) - np.array(ref_b), axis=0)
            unseen.update(
                supervised=float(np.mean(sup_pts)),
                reference=float(np.mean(ref_pts)),
                **difference(float(np.mean(sup_pts) - np.mean(ref_pts)), diff_boot),
            )
            unseen["judged"] = True
        else:
            unseen["judged"] = False
        row["unseen_types"] = unseen
        out[k] = row
    return out


# H5~H8과 교차점을 등록한 문장 그대로 판정한다.
# - H5(k = 5, 10): 지도 학습이 낮다. 상한 < 0이고 크기 ≥ 1.0%p면 지지, 하한 > 0이면 기각, 나머지 판정 불가.
# - H6(k = 40): 지도 학습이 높다. 하한 > 0이고 크기 ≥ 1.0%p면 지지, 상한 < 0이면 기각.
# - 교차점: H6 기준으로 "높다"가 지지되는 가장 작은 k(결과: k = 20).
# - H7: 판정할 수 있는 모든 k에서 상한 ≤ 0이면 지지, 하한 > 0인 k가 있으면 기각.
# - H8: D-S와 DM이 각각 P0보다 높은지("차이 있음"이고 양수면 지지).
def hypotheses(sup: dict[int, dict], pairs: dict[str, dict]) -> dict[str, str]:
    """H5-H8 as registered. `pairs` holds the unsupervised differences against P0 by method name."""

    def lower(row: dict) -> str:  # supervised below the reference
        d = row["all_defects"]
        if d["ci"][1] < 0 and abs(d["diff"]) >= MIN_EFFECT_AUROC:
            return "지지"
        return "기각" if d["ci"][0] > 0 else "판정 불가"

    def higher(row: dict) -> str:
        d = row["all_defects"]
        if d["ci"][0] > 0 and abs(d["diff"]) >= MIN_EFFECT_AUROC:
            return "지지"
        return "기각" if d["ci"][1] < 0 else "판정 불가"

    out = {}
    for k in (5, 10):
        if k in sup:
            out[f"H5 (k={k})"] = lower(sup[k])
    if 40 in sup:
        out["H6 (k=40)"] = higher(sup[40])
    # 교차점: k를 작은 것부터 보며 H6과 같은 기준("지도 학습이 높다" 지지)을 처음 만족하는 k.
    crossing = [k for k, row in sorted(sup.items()) if higher(row) == "지지"]
    out["교차점"] = f"k = {crossing[0]}" if crossing else "k = 40까지 교차 없음"
    judged = {k: row["unseen_types"] for k, row in sup.items() if row["unseen_types"]["judged"]}
    if not judged:
        out["H7"] = "판정 불가 (보지 못한 유형의 결함이 기준 장수에 못 미침)"
    elif all(u["ci"][1] <= 0 for u in judged.values()):
        out["H7"] = "지지"
    elif any(u["ci"][0] > 0 for u in judged.values()):
        out["H7"] = "기각 (k = " + ", ".join(str(k) for k, u in judged.items() if u["ci"][0] > 0) + ")"
    else:
        out["H7"] = "판정 불가"
    for name, d in pairs.items():
        if d["verdict"] == "차이 있음":
            out[f"H8 ({name} > P0)"] = "지지" if d["diff"] > 0 else "기각"
        else:
            out[f"H8 ({name} > P0)"] = d["verdict"]
    return out


# 리포트 조립: 같은 이미지인지 확인 → 첫 비지도 실행으로 부트스트랩 한 벌 생성 → 비지도 방법 요약 → 기준선(P0)
# 대비 차이 → (지도 학습이 있으면) k별 비교, 실행별 요약, 가설 판정.
def build_report(
    unsupervised: list[MethodRun],
    sup_runs: dict[tuple[int, int], MethodRun],
    reference_name: str,
    baseline_name: str = "p0",
) -> dict:
    runs = unsupervised + list(sup_runs.values())
    check_same_images(runs)
    # 부트스트랩은 첫 비지도 실행의 범주별 정상·결함 수로 만든다. 모든 실행이 같은 이미지를 같은 순서로
    # 채점했으므로(check_same_images) 어느 실행으로 만들어도 같은 추출이다.
    boot = Bootstrap(unsupervised[0])
    by_name = {r.name: r for r in unsupervised}
    summaries = {r.name: summarize(r, boot) for r in unsupervised}
    pairs = {}
    if baseline_name in by_name:
        base_boot = boot.macro_auroc(by_name[baseline_name])
        base_point = macro_auroc(by_name[baseline_name])
        for r in unsupervised:
            if r.name != baseline_name:
                pairs[r.name] = difference(macro_auroc(r) - base_point, boot.macro_auroc(r) - base_boot)
    report = {
        "n_boot": N_BOOT,
        "alpha": ALPHA,
        "unsupervised": summaries,
        "vs_baseline": {"baseline": baseline_name, "pairs": pairs},
        "reference_unsupervised": reference_name,
    }
    if sup_runs:
        sup = supervised_vs_reference(sup_runs, by_name[reference_name], boot)
        report["supervised"] = {str(k): v for k, v in sup.items()}
        report["supervised_runs"] = {
            f"k{k}-s{s}": summarize(run, boot) for (k, s), run in sorted(sup_runs.items())
        }
        report["hypotheses"] = hypotheses(sup, pairs)
    return report


def _pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}"


def _span(ci: list[float], digits: int = 2) -> str:
    return f"[{_pct(ci[0], digits)}, {_pct(ci[1], digits)}]"


# 콘솔용 마크다운 표: 비지도 방법 표, P0 대비 차이, 지도 학습 k별 표, 가설 판정 한 줄. docs/experiments.md
# 2단계 결과 표의 원본이다.
def format_report(report: dict) -> str:
    lines = [
        "| method | image AUROC | AUPRO | pixel AUROC | FPR@TPR95 | calibration | actual FPR (5%) "
        "| detection |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in report["unsupervised"].values():
        fixed = s["fixed_threshold"]
        cells = [
            s["name"],
            f"{_pct(s['macro_image_auroc'])} {_span(s['macro_image_auroc_ci'])}",
            f"{_pct(s['macro_aupro'])} {_span(s['macro_aupro_ci'])}",
            _pct(s["macro_pixel_auroc"]),
            _pct(s["pooled_fpr_at_tpr95"], 1),
        ]
        if fixed:
            cells += [
                f"{fixed['calibration']} (n={fixed['n_cal']})",
                f"{_pct(fixed['fpr'])} {_span(fixed['fpr_ci'])}",
                f"{_pct(fixed['tpr'])} {_span(fixed['tpr_ci'])}",
            ]
        else:
            cells += ["-", "-", "-"]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    base = report["vs_baseline"]["baseline"]
    for name, d in report["vs_baseline"]["pairs"].items():
        lines.append(f"{name} - {base}: {d['diff'] * 100:+.2f}%p {_span(d['ci'])} {d['verdict']}")
    if "supervised" in report:
        lines += [
            "",
            f"supervised head against {report['reference_unsupervised']} (image AUROC, mean over categories)",
            "| k | labels incl. validation | supervised | difference | verdict | unseen-type defects "
            "| difference | verdict |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in report["supervised"].values():
            d, u = row["all_defects"], row["unseen_types"]
            cells = [
                str(row["k"]),
                str(row["labels_with_validation"]),
                f"{_pct(row['supervised_macro_image_auroc'])} {_span(row['supervised_ci'])}",
                f"{d['diff'] * 100:+.2f}%p {_span(d['ci'])}",
                d["verdict"],
                f"{u['unseen_defects_mean']:.0f}",
            ]
            cells += (
                [f"{u['diff'] * 100:+.2f}%p {_span(u['ci'])}", u["verdict"]] if u["judged"] else ["-", "-"]
            )
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", "hypotheses: " + "; ".join(f"{k}: {v}" for k, v in report["hypotheses"].items())]
    return "\n".join(lines)


# 명령행 입구. 기준 비지도 방법은 dev에서는 평균 이미지 AUROC가 가장 높은 것으로 자동으로 정하고(2단계 규칙,
# dev 결함 20장 사용), test에서는 dev에서 고른 방법을 --reference로 반드시 준다(테스트 수치로 기준을 고르지
# 않게 함). 결과: dev에서 DM이 기준으로 뽑혔다. 저장된 점수만 읽으므로 봉인 테스트를 다시 읽지 않는다(README).
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", choices=["dev", "test"], required=True)
    parser.add_argument("--patchcore", nargs="*", default=["p0", "d-s"], help="PatchCore config names")
    parser.add_argument("--dinomaly", action="store_true", help="include outputs/dm-<protocol>")
    parser.add_argument("--supervised", action="store_true", help="include outputs/sup-<protocol>")
    parser.add_argument("--ks", nargs="*", type=int, default=[5, 10, 20, 40])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--reference", default=None, help="reference unsupervised method (dev default: best macro AUROC)"
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    if args.protocol == "test" and not args.reference:
        parser.error("--protocol test needs --reference (the unsupervised method chosen on dev)")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # 실행 폴더 이름 규칙: PatchCore는 outputs/<구성>-<절차>, Dinomaly는 outputs/dm-<절차>, 지도 학습은
    # outputs/sup-<절차>/k<k>-s<시드>.
    runs = [load_patchcore(paths.OUTPUTS / f"{name}-{args.protocol}") for name in args.patchcore]
    names = list(runs[0].cats)
    if args.dinomaly:
        runs.append(load_common(paths.OUTPUTS / f"dm-{args.protocol}", "dm", names))
    sup_runs = {}
    if args.supervised:
        for k in args.ks:
            for s in args.seeds:
                folder = paths.OUTPUTS / f"sup-{args.protocol}" / f"k{k}-s{s}"
                sup_runs[k, s] = load_common(folder, f"sup-k{k}-s{s}", names)
    # --reference가 없으면(dev) 비지도 실행 가운데 평균 이미지 AUROC가 가장 높은 것을 기준 방법으로 고른다.
    reference = args.reference or max(runs, key=macro_auroc).name
    report = build_report(runs, sup_runs, reference)
    report["protocol"] = args.protocol
    report["reference_chosen_by"] = "argument" if args.reference else f"best macro AUROC on {args.protocol}"
    print(format_report(report))
    if not args.no_write:
        out_dir = paths.REPORTS / "stage2"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"compare-{args.protocol}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
