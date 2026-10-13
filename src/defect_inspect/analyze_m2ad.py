"""Stage 3-B tables and verdicts: M2AD inspectors under synthetic and real illumination changes.

Rates are pooled over the inspectors (one per category and view). Intervals come from a cluster bootstrap
whose unit is the specimen, with the same draws for every condition and statistic, so differences are
paired. H11 and H12 are judged exactly as registered in docs/experiments.md.
"""

# [파일 역할] 3-B(M2AD 실제 조명 변화) 분석 모듈이다(`python -m defect_inspect.analyze_m2ad --methods p0
# d-s`). run_m2ad가 저장한 검사기 6개(Motor·Bird × 시점 000·120·240)의 점수로 조건별 표(S: 기준 조명 I01, P:
# I01에 합성 교란 15가지, R: 실제 조명 I02~I10)와 재보정 표를 만들고, 질문 3의 가설 H11·H12를 판정한다.
# [설계 이유] M2AD에서 독립 표본은 이미지가 아니라 시편이다(같은 시편을 여러 시점·조명으로 찍음). 그래서
# 구간은 시편 단위 군집 부트스트랩으로 낸다: 범주 안에서 정상 시편 20개와 불량 시편 50개를 따로 복원 추출하고,
# 뽑힌 시편은 모든 시점의 이미지를 함께 데려온다. 모든 조건·통계에 같은 추출을 써서 조건 S와의 차이가
# 짝지어진다(3-B 규칙).
# [결과] H11 기각(합성 최댓값 +82.5%p가 실제 조명 평균 +95.3%p의 절반보다 크다), H12 지지(새 조명 정상
# 30장으로 재보정하면 4.4% [1.3, 8.8]).
# [연결] 입력은 outputs/m2ad-<방법>, 출력은 reports/stage3/m2ad.json이다. ClusterBootstrap과 strict_json은
# analyze_m2ad_enrol(7단계)과 drift가, BASE는 figures가 가져다 쓴다.

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

# metrics.auroc는 검사기별 AUROC, stats는 층화 추출 인덱스와 백분위 구간을 준다.
from . import paths
from .metrics import auroc
from .stats import percentile_ci, stratified_indices

# 판정 상수. 부트스트랩 2,000번(seed 0), 기준 조건 이름 "S"(기준 조명 I01의 테스트 이미지), H12의 재보정 장수
# n = 30과 상한 10%, 가설을 판정하는 방법 P0(D-S는 판정 없이 나란히 싣는다).
N_BOOT = 2000
SEED = 0
BASE = "S"  # the reference condition every other one is compared with
H12_N = 30
H12_LIMIT = 0.10
JUDGED_METHOD = "p0"
# _TOL: 정수의 비인 비율이 규칙의 경계값과 정확히 같을 때 부동소수 오차(1e-17 수준)로 판정이 뒤집히지 않게
# 두는 허용치(코드 주석).
# Pooled rates are ratios of integers, so an interval end that equals the bound of a rule exactly comes out
# a few 1e-17 off in floats. The tolerance keeps "<= limit" and "< 0" what they are in exact arithmetic;
# it is far below the smallest step such an end can take (about 1e-5: one image in a thousand, times the
# weight the percentile interpolation gives to the neighbouring resample).
_TOL = 1e-9

# Cell: 조건 하나(또는 재보정한 조명 하나)를 검사기 순서대로 본 것. 검사기마다 (시편별 점수, 시편별 라벨,
# 임계값)이다. 라벨은 0 정상, 1 결함, -1 평가에서 뺌(불량 시편이지만 그 시점·조명에서 결함이 보이지 않거나
# 판별 불가로 표시된 이미지, m2ad.label_of).
# One condition (or one recalibrated illumination) as seen by every inspector, in inspector order:
# (scores [n_specimens], labels [n_specimens], threshold).
Cell = list[tuple[np.ndarray, np.ndarray, float]]


# 실행이 분석의 전제를 지키는지 확인한다: 모든 검사기가 같은 조건·재보정 목록을 가지고, 배열 모양이 (조건 수,
# 시편 수)이며, 같은 범주의 시점들이 같은 시편 목록과 불량 여부를 공유하고, 정상 시편은 모든 이미지가 라벨 0,
# 불량 시편은 1 또는 -1이다. 군집 부트스트랩이 시편 단위로 뽑으므로 이 전제가 깨지면 구간이 틀린다.
def _validate(run: dict) -> None:
    inspectors = run["inspectors"]
    if not inspectors:
        raise ValueError("the run has no inspectors")
    first = next(iter(inspectors.values()))
    by_category: dict[str, dict] = {}
    for name, arr in inspectors.items():
        for key in ("conditions", "recal_keys"):
            if arr[key].tolist() != first[key].tolist():
                raise ValueError(f"{name}: {key} differ between inspectors")
        n = len(arr["specimens"])
        if arr["scores"].shape != (len(arr["conditions"]), n) or arr["labels"].shape != arr["scores"].shape:
            raise ValueError(f"{name}: unexpected shape of scores or labels")
        if arr["recal_scores"].shape != (len(arr["recal_keys"]), n):
            raise ValueError(f"{name}: unexpected shape of recal_scores")
        same = by_category.setdefault(arr["category"], arr)
        if same["specimens"].tolist() != arr["specimens"].tolist() or not np.array_equal(
            same["object_anomaly"], arr["object_anomaly"]
        ):
            raise ValueError(f"{name}: specimens differ from the other views of {arr['category']}")
        # Normal specimens are label 0 in every image, anomalous ones 1 or -1: resampling relies on it.
        normal = arr["object_anomaly"] == 0
        for labels in (arr["labels"], arr["recal_labels"]):
            if (labels[:, normal] != 0).any() or not np.isin(labels[:, ~normal], (1, -1)).all():
                raise ValueError(f"{name}: labels do not agree with object_anomaly")


# run_m2ad 실행을 읽는다. 검사기 이름은 `<범주>_<시점>`이고 범주·시점을 배열 dict에 같이 넣은 뒤 검사한다.
def load(run_dir: Path) -> dict:
    """A `run_m2ad` run: {"meta": run.json, "inspectors": {"<category>_<view>": arrays}} (validated)."""
    run_dir = Path(run_dir)
    with open(run_dir / "run.json", encoding="utf-8") as f:
        meta = json.load(f)
    inspectors = {}
    for info in meta["inspectors"]:
        name = f"{info['category']}_{info['view']}"
        with np.load(run_dir / f"{name}.npz") as z:
            arrays = {k: z[k] for k in z.files}
        arrays["category"], arrays["view"] = info["category"], info["view"]
        inspectors[name] = arrays
    run = {"meta": meta, "inspectors": inspectors}
    _validate(run)
    return run


# 조건 이름 목록(S, P:<교란>, R:<조명>)과 재보정 키 목록(<조명>:<n>). 검사가 모든 검사기에서 같음을 보장하므로
# 첫 검사기에서 읽는다.
def conditions(run: dict) -> list[str]:
    return next(iter(run["inspectors"].values()))["conditions"].tolist()


def recal_keys(run: dict) -> list[str]:
    return next(iter(run["inspectors"].values()))["recal_keys"].tolist()


# 조건 하나의 Cell. 임계값은 검사기마다 I01로 정한 원래 값(고정 임계값) 그대로다.
def condition_cell(run: dict, name: str) -> Cell:
    cell = []
    for arr in run["inspectors"].values():
        k = arr["conditions"].tolist().index(name)
        cell.append((arr["scores"][k], arr["labels"][k], float(arr["threshold"])))
    return cell


# 재보정 키 하나(<조명>:<n>)의 Cell. 그 조명의 정상 n장을 뱅크에 더하고 다시 정한 임계값과, 그 뱅크로 다시
# 매긴 점수를 쓴다.
def recal_cell(run: dict, key: str) -> Cell:
    cell = []
    for arr in run["inspectors"].values():
        k = arr["recal_keys"].tolist().index(key)
        cell.append((arr["recal_scores"][k], arr["recal_labels"][k], float(arr["recal_thresholds"][k])))
    return cell


# 시편 단위 군집 부트스트랩. 범주마다 정상 시편과 불량 시편을 따로 복원 추출하고(생성기 하나, 범주 순서, 정상
# 다음 불량), 결과를 "재표집 b에서 시편 j가 몇 번 뽑혔나"의 가중치 행렬 weights[범주] [n_boot, 시편 수]로
# 둔다. 같은 범주의 세 시점 검사기가 같은 가중치를 쓰므로 한 시편의 여러 이미지가 함께 움직인다.
class ClusterBootstrap:
    """Specimen-level draws shared by every condition and statistic.

    Within each category the normal and the anomalous test specimens are resampled separately with
    replacement (one generator; categories in order, normals then anomalous). A drawn specimen brings its
    images from all views, so `weights[category]` [n_boot, n_specimens] counts how often each was drawn.
    """

    def __init__(self, run: dict, n_boot: int | None = None, seed: int = SEED):
        self.n_boot = int(n_boot or N_BOOT)
        inspectors = run["inspectors"]
        self.names = list(inspectors)
        self.category = {name: arr["category"] for name, arr in inspectors.items()}
        self.categories = list(dict.fromkeys(self.category.values()))
        groups, sizes = [], []
        for category in self.categories:
            arr = next(a for a in inspectors.values() if a["category"] == category)
            anomalous = np.asarray(arr["object_anomaly"]) != 0
            groups.append((np.flatnonzero(~anomalous), np.flatnonzero(anomalous)))
            sizes += [len(group) for group in groups[-1]]
        # 범주마다 (정상 시편, 불량 시편) 두 묶음의 크기로 층화 추출 한 벌을 만든다(범주 순서, 정상 다음
        # 불량).
        draws = stratified_indices(sizes, n_boot=self.n_boot, seed=seed)
        rows = np.arange(self.n_boot)[:, None]
        self.weights: dict[str, np.ndarray] = {}
        for i, category in enumerate(self.categories):
            weights = np.zeros((self.n_boot, sum(len(group) for group in groups[i])))
            for members, draw in zip(groups[i], draws[2 * i : 2 * i + 2], strict=True):
                if members.size:
                    # 뽑힌 시편 번호마다 횟수를 더해 가중치 행렬을 채운다. 복원 추출의 중복도 모두 센다.
                    np.add.at(weights, (rows, members[draw]), 1.0)
            self.weights[category] = weights

    # 여러 Cell을 합친 (오검출 수, 정상 이미지 수, 검출 수, 결함 이미지 수)의 관측값과 재표집 값. 시편별 0/1
    # 열을 만든 뒤 가중치 행렬과 곱하면 재표집마다의 합계가 행렬 곱 한 번으로 나온다. 라벨 -1 이미지는 어느
    # 열에도 들어가지 않아 자동으로 빠진다.
    def counts(self, cells: Sequence[Cell]) -> tuple[np.ndarray, np.ndarray]:
        """(false positives, normal images, detected defects, defect images) summed over `cells`.

        Returns the observed counts [4] and their bootstrap samples [n_boot, 4]. Rule: `score > threshold`.
        """
        point = np.zeros(4)
        samples = np.zeros((self.n_boot, 4))
        for cell in cells:
            for name, (scores, labels, threshold) in zip(self.names, cell, strict=True):
                flagged = np.asarray(scores, dtype=np.float64) > threshold
                normal, defect = labels == 0, labels == 1
                columns = np.stack([flagged & normal, normal, flagged & defect, defect], axis=1)
                columns = columns.astype(np.float64)
                point += columns.sum(axis=0)
                # [n_boot, 시편] @ [시편, 4] = 재표집마다의 네 합계. 같은 범주의 다른 시점 검사기도 같은
                # 가중치를 써서 시편이 함께 움직인다.
                samples += self.weights[self.category[name]] @ columns
        return point, samples

    # 검사기별 이미지 AUROC의 평균(관측값과 재표집 표본). 정상이나 결함 이미지가 없는 검사기는 관측과 재표집
    # 모두에서 평균에서 뺀다(같은 규칙).
    def auroc(self, cell: Cell) -> tuple[float, np.ndarray]:
        """Mean over inspectors of the image AUROC on labelled images: observed value and samples.

        One rule for both: an inspector without a defect image (or without a normal one) has no AUROC and
        is left out of the mean, in the observed data and in every resample that leaves it none. The
        value is nan only when no inspector has an AUROC.
        """
        values = []
        total = np.zeros(self.n_boot)
        count = np.zeros(self.n_boot)
        for name, (scores, labels, _) in zip(self.names, cell, strict=True):
            s = np.asarray(scores, dtype=np.float64)
            neg, pos = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
            if neg.size == 0 or pos.size == 0:
                continue
            values.append(auroc(s[neg], s[pos]))
            # 재표집 AUROC = (결함, 정상) 쌍 가운데 결함 점수가 큰 쌍의 가중 비율(동점은 0.5). 쌍마다의 승패
            # 행렬 wins를 한 번 만들고, 재표집마다 두 시편의 뽑힌 횟수를 곱한 가중치로 평균한다. 재표집마다
            # 정렬을 다시 하지 않아도 된다.
            # AUROC of a resample = weighted share of (defect, normal) pairs the defect wins (ties half).
            wins = (s[pos][:, None] > s[neg][None, :]) + 0.5 * (s[pos][:, None] == s[neg][None, :])
            weights = self.weights[self.category[name]]
            w_pos, w_neg = weights[:, pos], weights[:, neg]
            # 재표집마다의 (결함, 정상) 이미지 쌍 수. 0인 재표집에서는 이 검사기를 평균에서 뺀다(관측값과 같은
            # 규칙).
            pairs = w_pos.sum(axis=1) * w_neg.sum(axis=1)
            ok = pairs > 0
            total[ok] += ((w_pos[ok] @ wins) * w_neg[ok]).sum(axis=1) / pairs[ok]
            count[ok] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            samples = np.where(count > 0, total / count, np.nan)
        return float(np.mean(values)) if values else float("nan"), samples


# 분모가 0인 칸은 0으로 나누지 않고 nan으로 둔다(이미지가 없는 조건).
def _ratio(num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _ci(samples: np.ndarray) -> list[float]:
    lo, hi = percentile_ci(samples)
    return [float(lo), float(hi)]


# Cell 목록을 합친 합산 오검출률·검출률(관측값, 재표집 표본, 개수).
def _rates(boot: ClusterBootstrap, cells: Sequence[Cell]) -> dict:
    """Pooled false positive and detection rates over `cells`, observed and resampled."""
    point, samples = boot.counts(cells)
    return {
        "false_positives": int(point[0]),
        "n_normal": int(point[1]),
        "detected": int(point[2]),
        "n_defect": int(point[3]),
        "fpr": float(_ratio(point[0], point[1])),
        "fpr_samples": _ratio(samples[:, 0], samples[:, 1]),
        "tpr": float(_ratio(point[2], point[3])),
        "tpr_samples": _ratio(samples[:, 2], samples[:, 3]),
    }


# 모든 조건의 비율과 평균 AUROC를 같은 부트스트랩으로 계산한다. 기준 조건 S가 없으면 차이를 낼 수 없어 오류.
def _condition_stats(run: dict, boot: ClusterBootstrap) -> dict[str, dict]:
    stats = {}
    for name in conditions(run):
        cell = condition_cell(run, name)
        entry = _rates(boot, [cell])
        entry["auroc"], entry["auroc_samples"] = boot.auroc(cell)
        stats[name] = entry
    if BASE not in stats:
        raise ValueError(f"the run has no condition {BASE!r}")
    return stats


# 조건마다 한 행: 합산 오검출률·검출률·평균 AUROC와 구간, 그리고 S 대비 차이와 짝지은 구간.
# docs/experiments.md 3-B의 P0·D-S 표가 이 값이다(예: 실제 조명에서 오검출률 증가 평균 +95.3%p).
def condition_table(run: dict, n_boot: int | None = None, seed: int = SEED) -> list[dict]:
    """One row per condition: pooled FPR and detection rate, mean AUROC, and the paired differences to S."""
    boot = ClusterBootstrap(run, n_boot, seed)
    stats = _condition_stats(run, boot)
    base = stats[BASE]
    table = []
    for name, entry in stats.items():
        row = {"condition": name}
        row |= {k: entry[k] for k in ("false_positives", "n_normal", "detected", "n_defect")}
        for key in ("fpr", "tpr", "auroc"):
            row[key] = entry[key]
            row[f"{key}_ci"] = _ci(entry[f"{key}_samples"])
        if name != BASE:
            for key in ("fpr", "tpr", "auroc"):
                row[f"d_{key}"] = entry[key] - base[key]
                row[f"d_{key}_ci"] = _ci(entry[f"{key}_samples"] - base[f"{key}_samples"])
        table.append(row)
    return table


# H11: 합성 교란 15조건 가운데 가장 큰 오검출률 증가(P − S)가 실제 조명 아홉 조건의 증가(R − S) 평균의
# 절반보다 작다. 통계량 = 합성 최댓값 − 0.5 × 실제 평균이고, 최댓값은 재표집마다 다시 취한다(어느 조건이
# 최대인지도 재표집마다 달라질 수 있으므로). 구간 상한 < 0이면 지지, 하한 > 0이면 기각, 나머지 판정 불가.
# 결과: +34.9%p [33.6, 36.1]로 기각.
def h11(run: dict, n_boot: int | None = None, seed: int = SEED) -> dict:
    """H11: the largest synthetic FPR increase is below half the mean real-illumination FPR increase.

    Statistic = max over the P conditions of (FPR - FPR_S) minus 0.5 x mean over the R conditions of
    (FPR - FPR_S); the max is taken again inside every resample. Supported if the interval lies below 0,
    rejected if it lies above 0, undecided otherwise.
    """
    boot = ClusterBootstrap(run, n_boot, seed)
    stats = _condition_stats(run, boot)
    synthetic = [name for name in stats if name.startswith("P:")]
    real = [name for name in stats if name.startswith("R:")]
    if not synthetic or not real:
        raise ValueError("H11 needs synthetic (P) and real (R) conditions")
    base = stats[BASE]
    d_point = {name: stats[name]["fpr"] - base["fpr"] for name in synthetic + real}
    d_samples = {name: stats[name]["fpr_samples"] - base["fpr_samples"] for name in synthetic + real}
    # 점 추정: 증가가 가장 큰 합성 조건(동점이면 앞의 것)과 실제 조명 아홉 조건의 증가 평균.
    top = max(synthetic, key=lambda name: d_point[name])  # the first one on ties
    mean_real = float(np.mean([d_point[name] for name in real]))
    # 재표집마다 합성 15조건의 증가 최댓값을 다시 취하고, 실제 9조건의 증가 평균을 낸다. 최댓값을 고정한
    # 조건으로 구간을 내면 "가장 큰 것을 골랐다"는 선택의 흔들림이 빠지기 때문이다.
    max_samples = np.max(np.stack([d_samples[name] for name in synthetic]), axis=0)
    mean_samples = np.mean(np.stack([d_samples[name] for name in real]), axis=0)
    lo, hi = _ci(max_samples - 0.5 * mean_samples)
    # An end that is 0 up to float rounding is 0: it neither lies below nor above.
    verdict = "지지" if hi < -_TOL else "기각" if lo > _TOL else "판정 불가"
    return {
        "statistic": d_point[top] - 0.5 * mean_real,
        "ci": [lo, hi],
        "max_synthetic_condition": top,
        "max_synthetic_increase": d_point[top],
        "max_synthetic_increase_ci": _ci(max_samples),
        "mean_real_increase": mean_real,
        "mean_real_increase_ci": _ci(mean_samples),
        "n_synthetic": len(synthetic),
        "n_real": len(real),
        "verdict": verdict,
    }


# 재보정 장수 n으로 다시 잰 조명 목록.
def _recal_lights(run: dict, n: int) -> list[str]:
    return [key.split(":")[0] for key in recal_keys(run) if int(key.split(":")[1]) == n]


# H12: 새 조명마다 그 조명의 정상 n장(학습 시편 SHA-256 순 앞 n개)을 뱅크에 더하고 임계값을 다시 잡은 뒤, 아홉
# 조명을 합친 오검출률(정상 1,080장)이 10% 이하인가. 구간 상한 ≤ 10%면 지지, 하한 > 10%면 기각. 경계 비교에는
# _TOL을 더한다.
def h12(
    run: dict, n: int = H12_N, limit: float = H12_LIMIT, n_boot: int | None = None, seed: int = SEED
) -> dict:
    """H12: after recalibration with n specimens the FPR pooled over the new illuminations is <= limit.

    Supported if the interval's upper end <= limit, rejected if its lower end > limit, undecided otherwise.
    """
    lights = _recal_lights(run, n)
    if not lights:
        raise ValueError(f"the run has no recalibration with n = {n}")
    boot = ClusterBootstrap(run, n_boot, seed)
    rates = _rates(boot, [recal_cell(run, f"{light}:{n}") for light in lights])
    lo, hi = _ci(rates["fpr_samples"])
    # 구간 끝이 상한과 정확히 같은 경우를 부동소수 오차 없이 "이하"로 보도록 _TOL을 더해 비교한다.
    verdict = "지지" if hi <= limit + _TOL else "기각" if lo > limit + _TOL else "판정 불가"
    return {
        "n": n,
        "limit": limit,
        "illuminations": lights,
        "false_positives": rates["false_positives"],
        "n_normal": rates["n_normal"],
        "fpr": rates["fpr"],
        "fpr_ci": [lo, hi],
        "verdict": verdict,
    }


# 재보정 장수별 표: n = 0(재보정 없음, 실제 조명 R 조건 그대로)과 각 n(8, 30)의 합산 오검출률·검출률. 같은
# 조명 집합이면 n = 0 대비 차이와 구간도 붙인다. 결과: P0 96.9% → n = 8에서 3.5%, n = 30에서 4.4%.
def recal_table(run: dict, n_boot: int | None = None, seed: int = SEED) -> list[dict]:
    """Pooled FPR and detection rate over the real illuminations for n = 0 (no recalibration) and each n.

    Rows with n > 0 also carry the paired differences to n = 0.
    """
    boot = ClusterBootstrap(run, n_boot, seed)
    sizes = sorted({int(key.split(":")[1]) for key in recal_keys(run)})
    real = [name.split(":")[1] for name in conditions(run) if name.startswith("R:")]
    table: list[dict] = []
    base = None
    for n in [0, *sizes]:
        if n == 0:
            lights = real
            cells = [condition_cell(run, f"R:{light}") for light in lights]
        else:
            lights = _recal_lights(run, n)
            cells = [recal_cell(run, f"{light}:{n}") for light in lights]
        rates = _rates(boot, cells)
        row = {"n": n, "illuminations": lights}
        row |= {k: rates[k] for k in ("false_positives", "n_normal", "detected", "n_defect")}
        for key in ("fpr", "tpr"):
            row[key] = rates[key]
            row[f"{key}_ci"] = _ci(rates[f"{key}_samples"])
        if n == 0:
            base = rates
        # 재보정한 조명 집합이 n = 0과 같은 아홉 조명일 때만 짝지은 차이를 낸다(집합이 다르면 같은 이미지의
        # 비교가 아니다).
        elif lights == real:
            for key in ("fpr", "tpr"):
                row[f"d_{key}"] = rates[key] - base[key]
                row[f"d_{key}_ci"] = _ci(rates[f"{key}_samples"] - base[f"{key}_samples"])
        table.append(row)
    return table


# 보조 표(판정 없음): 검사기마다 임계값, 보정 점수 수, 조건별 오검출·검출 수와 AUROC. 합산 수치가 어느
# 검사기에서 나오는지 볼 때 쓴다.
def inspector_table(run: dict) -> dict[str, dict]:
    """Supplementary, no verdicts: per inspector its threshold and each condition's counts and AUROC."""
    out = {}
    for name, arr in run["inspectors"].items():
        threshold = float(arr["threshold"])
        per_condition = {}
        for k, condition in enumerate(arr["conditions"].tolist()):
            scores, labels = arr["scores"][k].astype(np.float64), arr["labels"][k]
            flagged = scores > threshold
            per_condition[condition] = {
                "false_positives": int((flagged & (labels == 0)).sum()),
                "n_normal": int((labels == 0).sum()),
                "detected": int((flagged & (labels == 1)).sum()),
                "n_defect": int((labels == 1).sum()),
                "auroc": float(auroc(scores[labels == 0], scores[labels == 1])),
            }
        out[name] = {"threshold": threshold, "n_cal": int(len(arr["cal_score"])), "conditions": per_condition}
    return out


# 방법마다 조건 표, 재보정 표, 검사기 표를 만들고, 조건이 갖춰져 있으면 H11·H12 수치를 계산한다. 판정
# 문구(verdict)는 P0에만 남기고 다른 방법(D-S)은 같은 수치를 판정 없이 싣는다(3-B 규칙). 코드 점검 실행이면
# check 표시가 남는다.
def build_report(runs: dict[str, dict], n_boot: int | None = None, seed: int = SEED) -> dict:
    """Tables of every method; H11 and H12 are judged on JUDGED_METHOD only."""
    report: dict = {
        "n_boot": int(n_boot or N_BOOT),
        "seed": seed,
        "judged_method": JUDGED_METHOD,
        "methods": {},
    }
    for name, run in runs.items():
        meta = run["meta"]
        entry = {
            "commit": meta.get("commit"),
            "device": meta.get("device"),
            "check": bool(meta.get("check", False)),
            "inspectors": list(run["inspectors"]),
            "conditions": condition_table(run, n_boot, seed),
            "recal": recal_table(run, n_boot, seed),
            "per_inspector": inspector_table(run),
        }
        has_real = any(c.startswith("R:") for c in conditions(run))
        if has_real and any(c.startswith("P:") for c in conditions(run)):
            entry["h11"] = h11(run, n_boot, seed)
        if _recal_lights(run, H12_N):
            entry["h12"] = h12(run, H12_N, H12_LIMIT, n_boot, seed)
        if name != JUDGED_METHOD:
            # Reported next to the judged method; no verdict is attached to these numbers.
            for key in ("h11", "h12"):
                entry.get(key, {}).pop("verdict", None)
        report["methods"][name] = entry
    judged = report["methods"].get(JUDGED_METHOD, {})
    report["hypotheses"] = {
        key.upper(): judged[key]["verdict"] for key in ("h11", "h12") if "verdict" in judged.get(key, {})
    }
    return report


# 리포트의 유한하지 않은 float(nan, inf)를 None으로 바꾼 사본. 표준 JSON에는 NaN 토큰이 없어 다른 프로그램이
# 파일을 읽지 못하기 때문이다(docstring). drift와 analyze_m2ad_enrol도 리포트를 쓸 때 이 함수를 쓴다.
def strict_json(value):
    """Copy of a report in which every non-finite float is None, so the file is strict JSON.

    A rate without images or an AUROC without defect images is nan in the report; `json` would write the
    token NaN, which other JSON readers reject.
    """
    if isinstance(value, dict):
        return {key: strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}"


def _span(ci: list[float], digits: int = 1) -> str:
    return f"[{_pct(ci[0], digits)}, {_pct(ci[1], digits)}]"


def _diff(row: dict, key: str) -> str:
    if f"d_{key}" not in row:
        return "-"
    return f"{row[f'd_{key}'] * 100:+.1f} {_span(row[f'd_{key}_ci'])}"


# 콘솔용 표: 방법마다 조건 표와 재보정 표, H11·H12 한 줄씩, 마지막에 판정 방법의 가설 요약.
def format_report(report: dict) -> str:
    lines = []
    for name, entry in report["methods"].items():
        lines += [
            f"## {name} ({len(entry['inspectors'])} inspectors, commit {entry['commit']})",
            "| condition | normal | defect | FPR % | detection % | AUROC % | dFPR %p | d detection %p "
            "| dAUROC %p |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for row in entry["conditions"]:
            cells = [
                row["condition"],
                str(row["n_normal"]),
                str(row["n_defect"]),
                f"{_pct(row['fpr'])} {_span(row['fpr_ci'])}",
                f"{_pct(row['tpr'])} {_span(row['tpr_ci'])}",
                f"{_pct(row['auroc'])} {_span(row['auroc_ci'])}",
                _diff(row, "fpr"),
                _diff(row, "tpr"),
                _diff(row, "auroc"),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines += [
            "",
            "real illuminations pooled, by recalibration size n (0 = not recalibrated)",
            "| n | normal | defect | FPR % | detection % | dFPR %p | d detection %p |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in entry["recal"]:
            cells = [
                str(row["n"]),
                str(row["n_normal"]),
                str(row["n_defect"]),
                f"{_pct(row['fpr'])} {_span(row['fpr_ci'])}",
                f"{_pct(row['tpr'])} {_span(row['tpr_ci'])}",
                _diff(row, "fpr"),
                _diff(row, "tpr"),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        if "h11" in entry:
            h = entry["h11"]
            lines.append(
                "H11 statistic (max synthetic increase - half the mean real increase): "
                f"{h['statistic'] * 100:+.1f}%p {_span(h['ci'])}; "
                f"max synthetic {h['max_synthetic_condition']} {h['max_synthetic_increase'] * 100:+.1f}%p, "
                f"mean real {h['mean_real_increase'] * 100:+.1f}%p"
                + (f" -> {h['verdict']}" if "verdict" in h else "")
            )
        if "h12" in entry:
            h = entry["h12"]
            lines.append(
                f"H12 pooled FPR after recalibration with n = {h['n']}: "
                f"{_pct(h['fpr'])}% {_span(h['fpr_ci'])} ({h['false_positives']}/{h['n_normal']}), "
                f"limit {_pct(h['limit'])}%" + (f" -> {h['verdict']}" if "verdict" in h else "")
            )
        lines.append("")
    if report["hypotheses"]:
        verdicts = "; ".join(f"{k}: {v}" for k, v in report["hypotheses"].items())
        lines.append(f"hypotheses ({report['judged_method']}): {verdicts}")
    return "\n".join(lines).rstrip() + "\n"


# 명령행 입구. --check면 코드 점검 실행(outputs/m2ad-<방법>-check, Motor 시점 000의 S와 I02만)을 읽고 파일은
# 쓰지 않는다. 저장된 점수만 읽으므로 M2AD 테스트 이미지를 다시 읽지 않는다.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="*", default=["p0", "d-s"], help="runs outputs/m2ad-<method>")
    parser.add_argument(
        "--check", action="store_true", help="read the code-check runs (outputs/m2ad-<method>-check); no file"
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    suffix = "-check" if args.check else ""
    runs = {name: load(paths.OUTPUTS / f"m2ad-{name}{suffix}") for name in args.methods}
    report = build_report(runs)
    print(format_report(report), end="")
    if not (args.no_write or args.check):
        out_dir = paths.REPORTS / "stage3"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "m2ad.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(strict_json(report), f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")


if __name__ == "__main__":
    main()
