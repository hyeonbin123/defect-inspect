"""3-B단계 표와 판정: 합성 교란과 실제 조명 변화 아래에서 M2AD 검사기들을 잰다.

비율은 검사기(범주·시점마다 하나)를 모두 합쳐 낸다. 구간은 시편을 단위로 하는 군집 부트스트랩에서 나오고,
모든 조건과 통계에 같은 추출을 쓰므로 차이가 짝지어진다. H11과 H12는 docs/experiments.md에 등록한 그대로
판정한다.
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
BASE = "S"  # 다른 모든 조건을 비교하는 기준 조건
H12_N = 30
H12_LIMIT = 0.10
JUDGED_METHOD = "p0"
# _TOL: 판정 경계의 부동소수 허용치. 합산 비율은 정수의 비라서, 규칙의 경계값과 정확히 같은 구간 끝이
# 부동소수로 계산하면 1e-17 몇 배만큼 어긋난다. 이 허용치로 "<= limit"과 "< 0"이 정확한 산술에서와 같은 뜻을
# 지키게 한다. 그런 구간 끝이 움직일 수 있는 가장 작은 걸음(약 1e-5: 천 장 가운데 한 장 × 백분위 보간이 이웃
# 재표집에 주는 가중치)보다 훨씬 작다.
_TOL = 1e-9

# Cell: 조건 하나(또는 재보정한 조명 하나)를 모든 검사기가 본 것, 검사기 순서대로. 검사기마다 (점수
# [n_specimens], 라벨 [n_specimens], 임계값)이다. 라벨은 0 정상, 1 결함, -1 평가에서 뺌(불량 시편이지만 그
# 시점·조명에서 결함이 보이지 않거나 판별 불가로 표시된 이미지, m2ad.label_of).
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
        # 정상 시편은 모든 이미지가 라벨 0, 불량 시편은 1 또는 -1이어야 한다. 재표집이 이 전제에 기댄다.
        normal = arr["object_anomaly"] == 0
        for labels in (arr["labels"], arr["recal_labels"]):
            if (labels[:, normal] != 0).any() or not np.isin(labels[:, ~normal], (1, -1)).all():
                raise ValueError(f"{name}: labels do not agree with object_anomaly")


# 범주·시점을 배열 dict에 같이 넣은 뒤 검사한다.
def load(run_dir: Path) -> dict:
    """`run_m2ad` 실행을 읽어 검사한 결과: {"meta": run.json, "inspectors": {"<범주>_<시점>": 배열}}."""
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


class ClusterBootstrap:
    """모든 조건과 통계가 함께 쓰는 시편 단위 추출(군집 부트스트랩).

    범주마다 테스트의 정상 시편과 불량 시편을 따로 복원 추출한다(생성기 하나, 범주 순서, 정상 다음 불량).
    뽑힌 시편은 모든 시점의 이미지를 함께 데려오므로, `weights[범주]` [n_boot, n_specimens]는 재표집 b에서
    시편 j가 몇 번 뽑혔는지를 센다. 같은 범주의 세 시점 검사기가 같은 가중치를 쓰므로 한 시편의 여러 이미지가
    함께 움직인다.
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

    # 시편별 0/1 열을 만든 뒤 가중치 행렬과 곱하면 재표집마다의 합계가 행렬 곱 한 번으로 나온다. 라벨 -1
    # 이미지는 어느 열에도 들어가지 않아 자동으로 빠진다.
    def counts(self, cells: Sequence[Cell]) -> tuple[np.ndarray, np.ndarray]:
        """`cells`를 합친 (오검출 수, 정상 이미지 수, 검출한 결함 수, 결함 이미지 수)를 센다.

        관측 개수 [4]와 그 부트스트랩 표본 [n_boot, 4]를 돌려준다. 규칙: `score > threshold`이면 불량.
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

    def auroc(self, cell: Cell) -> tuple[float, np.ndarray]:
        """라벨이 있는 이미지에서 잰 검사기별 이미지 AUROC의 평균: 관측값과 재표집 표본.

        관측과 재표집에 같은 규칙을 쓴다. 결함 이미지(또는 정상 이미지)가 없는 검사기는 AUROC가 없으므로
        평균에서 뺀다. 관측 데이터에서도, 그런 이미지를 하나도 남기지 않은 재표집에서도 마찬가지다. AUROC가
        있는 검사기가 하나도 없을 때만 값이 nan이다.
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


def _rates(boot: ClusterBootstrap, cells: Sequence[Cell]) -> dict:
    """`cells`를 합친 합산 오검출률·검출률을 관측값, 재표집 표본, 개수와 함께 낸다."""
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


# docs/experiments.md 3-B의 P0·D-S 표가 이 값이다(예: 실제 조명에서 오검출률 증가 평균 +95.3%p).
def condition_table(run: dict, n_boot: int | None = None, seed: int = SEED) -> list[dict]:
    """조건마다 한 행: 합산 FPR·검출률·평균 AUROC와 구간, 그리고 S 대비 차이와 짝지은 구간."""
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


# 합성은 15조건, 실제 조명은 아홉 조건이다. 최댓값을 재표집마다 다시 취하는 것은 어느 조건이 최대인지도
# 재표집마다 달라질 수 있기 때문이다. 결과: +34.9%p [33.6, 36.1]로 기각.
def h11(run: dict, n_boot: int | None = None, seed: int = SEED) -> dict:
    """H11: 합성 교란의 가장 큰 FPR 증가가 실제 조명 FPR 증가 평균의 절반보다 작다.

    통계량 = P 조건들의 (FPR − FPR_S) 최댓값 − 0.5 × R 조건들의 (FPR − FPR_S) 평균이고, 최댓값은 재표집마다
    다시 취한다. 구간 전체가 0보다 아래면 지지, 위면 기각, 그 밖에는 판정 불가다.
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
    top = max(synthetic, key=lambda name: d_point[name])  # 동점이면 앞의 것
    mean_real = float(np.mean([d_point[name] for name in real]))
    # 재표집마다 합성 15조건의 증가 최댓값을 다시 취하고, 실제 9조건의 증가 평균을 낸다. 최댓값을 고정한
    # 조건으로 구간을 내면 "가장 큰 것을 골랐다"는 선택의 흔들림이 빠지기 때문이다.
    max_samples = np.max(np.stack([d_samples[name] for name in synthetic]), axis=0)
    mean_samples = np.mean(np.stack([d_samples[name] for name in real]), axis=0)
    lo, hi = _ci(max_samples - 0.5 * mean_samples)
    # 부동소수 반올림 범위 안에서 0인 구간 끝은 0으로 본다: 0보다 아래도 위도 아니다.
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


# 재보정은 새 조명마다 그 조명의 정상 n장(학습 시편 SHA-256 순 앞 n개)을 뱅크에 더하고 임계값을 다시 잡는
# 것이다. 아홉 조명을 합치면 정상 1,080장이다. 경계 비교에는 _TOL을 더한다.
def h12(
    run: dict, n: int = H12_N, limit: float = H12_LIMIT, n_boot: int | None = None, seed: int = SEED
) -> dict:
    """H12: 시편 n개로 재보정한 뒤 새 조명들을 합친 FPR이 limit(10%) 이하다.

    구간 상한 ≤ limit이면 지지, 하한 > limit이면 기각, 그 밖에는 판정 불가다.
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


# 재보정 장수 n은 8과 30이다. 결과: P0 96.9% → n = 8에서 3.5%, n = 30에서 4.4%.
def recal_table(run: dict, n_boot: int | None = None, seed: int = SEED) -> list[dict]:
    """실제 조명들을 합친 FPR·검출률을 n = 0(재보정 없음, R 조건 그대로)과 각 n마다 낸다.

    n > 0인 행에는 조명 집합이 n = 0과 같을 때 n = 0 대비 짝지은 차이와 구간도 붙인다.
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


# 합산 수치가 어느 검사기에서 나오는지 볼 때 쓴다.
def inspector_table(run: dict) -> dict[str, dict]:
    """판정 없는 보조 표: 검사기마다 임계값, 보정 점수 수, 조건별 오검출·검출 수와 AUROC."""
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


# 방법마다 조건 표, 재보정 표, 검사기 표를 만들고, 조건이 갖춰져 있으면 H11·H12 수치를 계산한다. 다른
# 방법(D-S)은 같은 수치를 판정 없이 싣는다(3-B 규칙). 코드 점검 실행이면 check 표시가 남는다.
def build_report(runs: dict[str, dict], n_boot: int | None = None, seed: int = SEED) -> dict:
    """모든 방법의 표를 만든다. H11과 H12의 판정 문구(verdict)는 JUDGED_METHOD(P0)에만 남긴다."""
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
            # 판정하는 방법 옆에 나란히 싣는 수치라 판정 문구를 붙이지 않는다.
            for key in ("h11", "h12"):
                entry.get(key, {}).pop("verdict", None)
        report["methods"][name] = entry
    judged = report["methods"].get(JUDGED_METHOD, {})
    report["hypotheses"] = {
        key.upper(): judged[key]["verdict"] for key in ("h11", "h12") if "verdict" in judged.get(key, {})
    }
    return report


# drift와 analyze_m2ad_enrol도 리포트를 쓸 때 이 함수를 쓴다.
def strict_json(value):
    """유한하지 않은 float(nan, inf)를 모두 None으로 바꾼 리포트 사본을 만든다. 파일이 엄격한 JSON이 된다.

    이미지가 없는 비율이나 결함 이미지가 없는 AUROC는 리포트에서 nan이다. 그대로 두면 `json`이 NaN 토큰을
    쓰는데, 표준 JSON에는 이 토큰이 없어 다른 JSON 읽기 프로그램이 파일을 거부한다.
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
    parser.add_argument("--methods", nargs="*", default=["p0", "d-s"], help="읽을 실행 outputs/m2ad-<방법>")
    parser.add_argument(
        "--check",
        action="store_true",
        help="코드 점검 실행(outputs/m2ad-<방법>-check)을 읽는다. 파일은 쓰지 않는다",
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
