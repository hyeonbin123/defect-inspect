"""9단계: 결함과 무관한 촬영 통계로 장별 품질 게이트와 변화 감지를 잰다(CPU만, 봉인 데이터 없음).

`python -m defect_inspect.capture` → `reports/stage9/capture.json`. 규칙은 docs/experiments.md 9단계.
`python -m defect_inspect.capture --masked` → `reports/stage9/capture-masked.json`: 9단계 보충(결함 마스크 밖
촬영 통계). 규칙은 docs/experiments.md "9단계 보충".
"""

# [파일 역할] 이미지 전체의 촬영 통계 11개(채널 평균, 휘도 표준편차·백분위, 초점(라플라시안 분산), 기울기
# 크기, 포화·전경 비율, 전경 밝기)를 뽑고, 범주(M2AD는 검사기)마다 정상 사진으로 맞춘 수축 공분산 Mahalanobis
# 거리를 "이상한 촬영" 점수로 쓴다. 그 점수로 (가) 장마다 "재촬영 필요"를 가리는 품질 게이트와 (나) 7단계와
# 같은 마팅게일 변화 감지기를 만들어, VisA dev와 M2AD 학습 시편에서 잰다.
# [왜 필요한가] 7단계의 감지기는 모델 점수의 p-value를 써서 결함이 늘어난 것에도 경보했다(결함 10%면 스트림의
# 34~61%, README 한계). 또 서빙 모델은 흐림 σ 1.0에서 정상의 28.5%를 불량으로 판정했다(6단계 보조). 결함은
# 대개 국소적이라 전역 통계를 거의 바꾸지 않고 조명·초점·노출·압축은 크게 바꾼다는 가정으로, 점수와 무관한
# 신호를 따로 둔다. 방법 자체는 오래된 고전 기법이다(experiments.md 9단계).
# [설계 이유] 변수는 측정 전에 11개로 고정했다. 거의 변하지 않는 변수(예: VisA의 전경 비율 ≈ 1)가 작은
# 흔들림에 과민해지지 않게 변수별 척도에 바닥값을 두고, 표본 수(범주당 240~484장, M2AD 검사기당 24장)에 비해
# 변수가 많아 공분산 역행렬이 불안정하지 않게 단위행렬 쪽으로 λ = 0.2만큼 수축한다. 보정 점수는 자기 사진이
# 평균· 공분산에 들어가 작게 나오지 않도록 교차 적합으로 낸다(PatchCore 교차 적합과 같은 구조).
# [흐름] capture_features(사진 → 11개) → CaptureModel.fit/score → crossfit_scores → VisA: gate_table,
# dms_table, fa_table, perturb_delay_table / M2AD: m2ad_tables → build_report → main.
# [9단계 보충] 9단계는 교란 없는 dev 결함의 20%(pcb2·pipe_fryum에 몰림)를 게이트했지만, 결함 영역 때문인지
# 결함 사진의 촬영 조건이 달랐기 때문인지 가르지 않았다. --masked는 결함 마스크 밖 화소만으로 같은 통계를
# 내고, 마스크마다 정상 풀에도 같은 마스크를 씌워 모형·게이트를 다시 맞추는 "마스크 맞춤" 채점으로 이를
# 가린다. 흐름: outside_pixels(마스크 → 쓸 화소) → capture_features(keep=...) → masked_scores →
# masked_tables(9단계 기록 대조, 해석 규칙 masked_reading) → run_masked.
# [연결] conditions.apply_condition(3단계 교란), calibrate.conformal_threshold(게이트 임계값),
# drift(p-value, 마팅게일, 스트림 도우미), cache·m2ad(리사이즈 캐시), splits(dev 절차 행 고르기, path_key).
# torch는 쓰지 않는다.

import argparse
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from . import drift, paths
from .calibrate import Threshold, conformal_threshold
from .conditions import CLEAN, apply_condition, condition_names
from .splits import path_key

# 변수 이름(순서가 곧 특징 벡터의 열 순서). 규칙의 "촬영 통계 11개"와 같다.
FEATURES = (
    "mean_r",
    "mean_g",
    "mean_b",
    "y_std",
    "y_p01",
    "y_p99",
    "log_laplacian_var",
    "log_gradient_mean",
    "saturated_share",
    "foreground_share",
    "foreground_y_mean",
)
# 변수별 척도의 바닥값. 밝기 단위(0~255) 0.5, 로그 단위 0.01, 비율 0.001(규칙). 표준편차가 이보다 작으면 이
# 값으로 나눈다.
FLOORS = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.01, 0.01, 0.001, 0.001, 0.5])
# SHRINK: 공분산을 단위행렬 쪽으로 섞는 비율 λ. ALPHA_GATE: 품질 게이트의 목표 거절률 1%.
SHRINK = 0.2
ALPHA_GATE = 0.01
# 전경 문턱(Y > 20)과 포화 문턱(어느 채널이든 250 이상). 휘도 가중치는 ITU-R BT.601.
FOREGROUND_LEVEL = 20.0
SATURATION_LEVEL = 250
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
# 메모리를 묶으려고 이미지를 64장씩 나눠 계산한다(280px float32 64장 ≈ 20MB씩).
_CHUNK = 64
# 스트림 수: VisA 허위 경보는 범주마다 1,000개, 교란 지연은 200개, M2AD는 검사기·조명마다 1,000개(규칙).
N_FA_STREAMS = 1000
N_DELAY_STREAMS = 200
N_M2AD_STREAMS = 1000
# H26·H27의 기준(규칙): 흐림 σ 1.0 정상 게이트 90% 이상, dev 결함 게이트 2% 이하, 결함 10% 허위 경보 1% 이하,
# 아홉 조명 가운데 7개 이상에서 변화 뒤 20장 안 경보 50% 이상.
H26_BLUR = "blur-2"
H26_MIN_GATED = 0.90
H26_MAX_DEFECT = 0.02
H27_RATE = 0.10
H27_MAX_FA = 0.01
H27_WITHIN = 20
H27_MIN_SHARE = 0.50
H27_MIN_LIGHTS = 7
# 6단계 dev 보조 표에 쓴 서빙 모델 실행(교란 없음, 흐림 σ 1.0, 이동 1px)과 P0 dev 실행. 이 단계는 저장된
# 점수만 읽는다.
DMS_RUNS = {
    "clean": "dms-280-car-onnx-fp32-dev",
    "blur-2": "dms-280-car-onnx-fp32-dev-blur-2",
    "shift-1": "dms-280-car-onnx-fp32-dev-shift-1",
}
P0_DEV_RUN = "p0-dev"
DMS_ALPHA = 0.05
# 6단계 dev 보조 표에 적힌 흐림 σ 1.0의 정상 오검출률. 이 값이 다시 나와야 서빙 모델 칸을 낸다(규칙).
DMS_BLUR_RECORD = 0.2850


# ---------------------------------------------------------------- 촬영 통계


# 휘도 Y = 0.299R + 0.587G + 0.114B를 float32 [N, S, S]로 만든다.
def luminance(images: np.ndarray) -> np.ndarray:
    return (np.asarray(images, dtype=np.float32) @ LUMA).astype(np.float32)


# 이미지 묶음(같은 크기)의 촬영 통계 [N, 11]. 라플라시안과 Sobel은 이미지 축(1, 2)으로만 걸어 묶음 축이 섞이지
# 않게 하고, 가장자리는 반사(mode="reflect")로 채운다. 이미지가 0장이면 [0, 11].
# keep(bool [S, S], 묶음의 모든 사진에 같은 것)을 주면 필터는 그대로 이미지 전체에 걸고 평균·표준편차·분산·
# 백분위·비율만 keep이 True인 화소로 낸다(9단계 보충의 마스크 밖 통계). 평균·표준편차·분산은 numpy 줄임의
# where 인자로 내서 "고른 화소만의 통계"의 정의를 numpy에 맡기고(합과 개수를 손으로 나누지 않는다), 백분위는
# where를 받지 않아 keep 화소만 골라 낸다. keep이 None이면 where가 numpy 기본값 True이고 백분위는 모든 화소라
# 9단계 측정 때와 같은 호출이다. 모든 화소가 True인 keep도 None과 비트 단위로 같은 값을 낸다(테스트로 확인).
def _chunk_features(images: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(images)
    if x.ndim != 4 or x.shape[-1] != 3 or x.dtype != np.uint8:
        raise ValueError(f"expected uint8 images [N, S, S, 3], got {x.dtype} {x.shape}")
    n = x.shape[0]
    out = np.empty((n, len(FEATURES)), dtype=np.float64)
    if n == 0:
        return out
    if keep is None:
        where_rgb = where_y = where_flat = True
        pick: slice | np.ndarray = slice(None)
        n_kept = x.shape[1] * x.shape[2]
    else:
        where_rgb, where_y, where_flat = keep[None, :, :, None], keep[None], keep.reshape(1, -1)
        pick = keep.reshape(-1)
        n_kept = int(keep.sum())
    rgb = x.astype(np.float32)
    y = luminance(x)
    out[:, 0:3] = rgb.mean(axis=(1, 2), where=where_rgb)
    out[:, 3] = y.std(axis=(1, 2), where=where_y)
    flat = y.reshape(n, -1)
    kept = flat[:, pick]
    out[:, 4] = np.percentile(kept, 1, axis=1)
    out[:, 5] = np.percentile(kept, 99, axis=1)
    # 라플라시안 [0, 1, 0; 1, -4, 1; 0, 1, 0] = 두 축 [1, -2, 1]의 합. 분산이 클수록 초점이 맞은 사진이다.
    lap = ndimage.correlate1d(y, [1.0, -2.0, 1.0], axis=1, mode="reflect") + ndimage.correlate1d(
        y, [1.0, -2.0, 1.0], axis=2, mode="reflect"
    )
    out[:, 6] = np.log1p(lap.reshape(n, -1).var(axis=1, where=where_flat))
    # Sobel 3×3: 한 축으로 [-1, 0, 1] 미분, 다른 축으로 [1, 2, 1] 평활. 기울기 크기의 평균.
    gx = ndimage.correlate1d(
        ndimage.correlate1d(y, [-1.0, 0.0, 1.0], axis=2, mode="reflect"),
        [1.0, 2.0, 1.0],
        axis=1,
        mode="reflect",
    )
    gy = ndimage.correlate1d(
        ndimage.correlate1d(y, [-1.0, 0.0, 1.0], axis=1, mode="reflect"),
        [1.0, 2.0, 1.0],
        axis=2,
        mode="reflect",
    )
    out[:, 7] = np.log1p(np.sqrt(gx * gx + gy * gy).reshape(n, -1).mean(axis=1, where=where_flat))
    out[:, 8] = (x >= SATURATION_LEVEL).any(axis=3).reshape(n, -1).mean(axis=1, where=where_flat)
    # 전경 비율의 분모는 쓰는 화소 수, 전경 평균은 쓰는 화소 가운데 전경인 것의 평균이다.
    fg = y > FOREGROUND_LEVEL
    if keep is not None:
        fg &= keep
    count = fg.reshape(n, -1).sum(axis=1)
    out[:, 9] = count / n_kept
    total = np.where(fg, y, 0.0).reshape(n, -1).sum(axis=1)
    out[:, 10] = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    return out


# keep은 bool만 받는다: 결함 마스크(1 = 결함)를 keep(True = 쓸 화소)으로 잘못 넘기면 뜻이 뒤집히므로, 정수
# 배열은 거절해 호출하는 쪽이 outside_pixels로 바꾸게 한다. 쓸 화소가 하나도 없으면 통계가 정의되지 않아
# 멈춘다.
def capture_features(images: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """uint8 이미지 [N, S, S, 3]의 촬영 통계 float64 [N, 11](열 순서는 FEATURES).

    keep(bool [S, S])을 주면 필터는 이미지 전체에 걸고 통계는 keep이 True인 화소로만 낸다.
    """
    x = np.asarray(images)
    if x.ndim != 4:
        raise ValueError(f"expected uint8 images [N, S, S, 3], got shape {x.shape}")
    if keep is not None:
        keep = np.asarray(keep)
        if keep.dtype != np.bool_ or keep.shape != x.shape[1:3]:
            raise ValueError(
                f"keep must be a bool array of shape {x.shape[1:3]}, got {keep.dtype} {keep.shape}"
            )
        if not keep.any():
            raise ValueError("no pixels left outside the mask")
    parts = [_chunk_features(x[i : i + _CHUNK], keep) for i in range(0, x.shape[0], _CHUNK)]
    return np.concatenate(parts) if parts else np.empty((0, len(FEATURES)))


# ---------------------------------------------------------------- 점수 모형


# 정상 사진의 촬영 통계로 맞춘 모형: 평균, 변수별 척도(바닥값 적용), 수축 공분산의 역행렬. 점수는 표준화한
# 벡터의 Mahalanobis 거리다. frozen이라 맞춘 뒤 바뀌지 않는다.
@dataclass(frozen=True)
class CaptureModel:
    mean: np.ndarray
    scale: np.ndarray
    precision: np.ndarray  # (1 - λ)R + λI 의 역행렬

    # 표본이 2장보다 적으면 공분산을 낼 수 없으므로 멈춘다. 공분산은 표준화한 z에서 ddof=1로 낸다.
    @classmethod
    def fit(
        cls, features: np.ndarray, *, shrink: float = SHRINK, floors: np.ndarray = FLOORS
    ) -> "CaptureModel":
        """정상 사진의 촬영 통계 [n, d]로 모형을 맞춘다(n ≥ 2)."""
        f = np.asarray(features, dtype=np.float64)
        if f.ndim != 2 or f.shape[1] != len(floors):
            raise ValueError(f"expected features [n, {len(floors)}], got {f.shape}")
        if f.shape[0] < 2:
            raise ValueError("fitting needs at least two images")
        if not 0.0 <= shrink <= 1.0:
            raise ValueError(f"shrink must be in [0, 1], got {shrink}")
        mean = f.mean(axis=0)
        scale = np.maximum(f.std(axis=0, ddof=1), floors)
        z = (f - mean) / scale
        cov = np.atleast_2d(np.cov(z, rowvar=False))
        sigma = (1.0 - shrink) * cov + shrink * np.eye(cov.shape[0])
        return cls(mean=mean, scale=scale, precision=np.linalg.inv(sigma))

    def score(self, features: np.ndarray) -> np.ndarray:
        """촬영 통계 [n, d]의 점수 [n] = √(zᵀ Σ⁻¹ z). 클수록 맞춘 사진들과 다른 촬영이다."""
        z = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", z, self.precision, z), 0.0))


# 겹마다 그 겹을 뺀 사진으로 모형을 맞춰 그 겹을 채점한다. 사진이 자기 모형에 들어가 점수가 작게 나오는 것을
# 막는다(규칙: 보정 점수는 교차 적합).
def crossfit_scores(features: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """사진마다 자기 겹을 뺀 사진으로 맞춘 모형의 점수 [n]."""
    f = np.asarray(features, dtype=np.float64)
    folds = np.asarray(folds)
    out = np.full(len(f), np.nan)
    for fold in np.unique(folds):
        held = folds == fold
        out[held] = CaptureModel.fit(f[~held]).score(f[held])
    return out


# ---------------------------------------------------------------- VisA dev


# 범주 하나의 dev 자료: 정상 풀(겹 1~4)의 통계와 겹 번호, 평가 사진(겹 0 정상 + dev 결함)의 경로·라벨과
# 조건마다 통계. 평가 사진에는 조건을 리사이즈한 뒤에 준다(3-A와 같다). 봉인 test 행은 고르지
# 않는다(splits.select가 dev 절차에서 test 역할을 내주지 않는다).
def visa_category(cache, manifest, category: str, conditions: Sequence[str]) -> dict:
    from .splits import select

    pool = select(manifest, protocol="dev", part="pool_normal", category=category)
    normal = select(manifest, protocol="dev", part="eval_normal", category=category)
    defect = select(manifest, protocol="dev", part="eval_defect", category=category)
    eval_rows = normal + defect
    images = cache.images(eval_rows)
    return {
        "pool": capture_features(cache.images(pool)),
        "pool_folds": np.array([r.fold for r in pool], dtype=np.int64),
        "eval_images": [r.image for r in eval_rows],
        "labels": np.array([0] * len(normal) + [1] * len(defect), dtype=np.int8),
        "eval": {name: capture_features(apply_condition(images, name)) for name in conditions},
    }


# 범주마다 교차 적합 보정 점수, 평가 점수(조건별), 게이트 임계값을 만든다.
def visa_scores(data: dict) -> dict:
    model = CaptureModel.fit(data["pool"])
    cal = crossfit_scores(data["pool"], data["pool_folds"])
    gate = conformal_threshold(cal, ALPHA_GATE)
    return {
        "cal": cal,
        "gate": gate,
        "eval": {name: model.score(f) for name, f in data["eval"].items()},
        "labels": data["labels"],
        "eval_images": data["eval_images"],
    }


def _rate(num: int, den: int) -> float:
    return num / den if den else float("nan")


# 조건마다 정상·결함의 게이트 비율(12개 범주 합산)과 범주별 수. H26은 이 표의 흐림 σ 1.0 정상과 교란 없는
# 결함.
def gate_table(scores: dict[str, dict]) -> dict:
    """{조건: {"normal": {...}, "defect": {...}, "per_category": {...}}}."""
    table = {}
    names = list(next(iter(scores.values()))["eval"])
    for name in names:
        totals = {"normal": [0, 0], "defect": [0, 0]}
        per_category = {}
        for category, s in scores.items():
            gated = s["eval"][name] > s["gate"].value
            row = {}
            for label, key in ((0, "normal"), (1, "defect")):
                mask = s["labels"] == label
                g, n = int(gated[mask].sum()), int(mask.sum())
                totals[key][0] += g
                totals[key][1] += n
                row[key] = [g, n]
            per_category[category] = row
        table[name] = {key: {"gated": g, "n": n, "rate": _rate(g, n)} for key, (g, n) in totals.items()} | {
            "per_category": per_category
        }
    return table


# 게이트 임계값의 기록(범주마다 값·순위·n·보장 여부).
def gate_thresholds(scores: dict[str, dict]) -> dict:
    return {
        c: {
            "value": s["gate"].value,
            "rank": s["gate"].rank,
            "n": s["gate"].n,
            "guaranteed": s["gate"].guaranteed,
        }
        for c, s in scores.items()
    }


# 서빙 모델(dms-280-car, 6단계 dev 보조 점수)과 게이트를 함께 본 표. 서빙 모델 임계값은 6단계 보조 표와 같이
# 교란 없는 겹 0 정상 점수의 목표 5% 임계값이다. 저장된 점수의 사진 순서가 이 단계의 평가 순서와 같은지
# 확인한다. 먼저 흐림 σ 1.0의 정상 오검출률이 기록(28.50%)과 같은지 본다. 다르면 이 표를 내지 않는다(규칙).
def dms_table(scores: dict[str, dict], outputs: Path) -> dict:
    stored = {}
    for name, run in DMS_RUNS.items():
        stored[name] = {}
        for category in scores:
            with np.load(Path(outputs) / run / f"{category}.npz") as z:
                stored[name][category] = {
                    "images": z["eval_images"].tolist(),
                    "labels": z["eval_labels"],
                    "score": z["eval_score"],
                }
    thresholds = {}
    for category, s in scores.items():
        clean = stored["clean"][category]
        for name in DMS_RUNS:
            if stored[name][category]["images"] != s["eval_images"]:
                raise ValueError(f"{DMS_RUNS[name]}/{category}: images differ from the dev evaluation set")
        thresholds[category] = conformal_threshold(clean["score"][clean["labels"] == 0], DMS_ALPHA).value
    flagged_blur = sum(
        int((stored["blur-2"][c]["score"][s["labels"] == 0] > thresholds[c]).sum()) for c, s in scores.items()
    )
    n_normal = sum(int((s["labels"] == 0).sum()) for s in scores.values())
    check = round(flagged_blur / n_normal, 4)
    out: dict = {
        "check": {"blur-2_normal_flagged": check, "record": DMS_BLUR_RECORD, "ok": check == DMS_BLUR_RECORD}
    }
    if not out["check"]["ok"]:
        return out
    for name in DMS_RUNS:
        cells = {}
        for label, key in ((0, "normal"), (1, "defect")):
            flagged = gated = both = n = 0
            for category, s in scores.items():
                mask = s["labels"] == label
                f = stored[name][category]["score"][mask] > thresholds[category]
                g = s["eval"][name][mask] > s["gate"].value
                flagged += int(f.sum())
                gated += int(g.sum())
                both += int((f & ~g).sum())
                n += int(mask.sum())
            cells[key] = {
                "n": n,
                "flagged_without_gate": _rate(flagged, n),
                "gated": _rate(gated, n),
                "flagged_after_gate": _rate(both, n),
            }
        out[name] = cells
    return out


# 같은 스트림 위에서 두 감지기(촬영 통계, P0 점수)를 돌릴 수 있게, 점수 대신 "사진 번호"로 스트림을 만든다.
# 정상 번호는 0..n-1, 결함 번호는 n..n+d-1이다. drift._segment가 점수 배열을 섞는 것과 같은 방식이다.
def _index_streams(n_normal: int, n_defect: int, rate: float, n_streams: int, rng) -> tuple[np.ndarray, int]:
    normal = np.arange(n_normal, dtype=np.float64)
    defect = np.arange(n_normal, n_normal + n_defect, dtype=np.float64)
    k = drift.n_mixed(n_normal, rate, n_defect)
    rows = [drift._segment(rng, normal, defect, k)[0] for _ in range(n_streams)]
    return np.stack(rows).astype(np.int64), k


# P0 dev 실행의 점수(평가 = 전체 뱅크, 보정 = 정상 풀 교차 적합)를 읽고 사진 순서를 확인한다.
def _p0_dev(outputs: Path, category: str, images: list[str]) -> dict:
    with np.load(Path(outputs) / P0_DEV_RUN / f"{category}.npz") as z:
        if z["eval_images"].tolist() != images:
            raise ValueError(f"{P0_DEV_RUN}/{category}: images differ from the dev evaluation set")
        return {
            "eval": z["eval_score_full"].astype(np.float64),
            "cal": z["pool_score_oof"].astype(np.float64),
        }


# VisA dev 허위 경보: 결함 섞임 0/5/10%마다 범주별 1,000개 스트림, 두 감지기를 같은 스트림·같은 θ로 돌린다.
def fa_table(scores: dict[str, dict], outputs: Path, n_streams: int = N_FA_STREAMS) -> dict:
    table = {}
    for rate in drift.RATES:
        alarms = {"capture": [], "p0_score": []}
        mixed = {}
        for category, s in scores.items():
            normal_idx = np.flatnonzero(s["labels"] == 0)
            defect_idx = np.flatnonzero(s["labels"] == 1)
            order = np.concatenate([normal_idx, defect_idx])
            cap = s["eval"][CLEAN][order]
            p0 = _p0_dev(outputs, category, s["eval_images"])
            p0_scores = p0["eval"][order]
            rng = drift.rng_for(f"capture-fa|{category}|{rate}")
            streams, k = _index_streams(len(normal_idx), len(defect_idx), rate, n_streams, rng)
            theta = drift._theta(rng, streams.shape)
            alarms["capture"].append(
                drift.first_alarm(drift.inductive_pvalues(cap[streams], s["cal"], theta))
            )
            alarms["p0_score"].append(
                drift.first_alarm(drift.inductive_pvalues(p0_scores[streams], p0["cal"], theta))
            )
            mixed[category] = k
        table[f"{rate:.2f}"] = {
            "mixed_defects": mixed,
            **{name: drift.summarise_fa(np.concatenate(a)) for name, a in alarms.items()},
        }
    return table


# VisA dev 교란 지연(판정 없음): 조건마다 범주별 200개 스트림(정상·결함을 반씩 나눠 앞은 교란 없음, 뒤는
# 교란).
def perturb_delay_table(scores: dict[str, dict], n_streams: int = N_DELAY_STREAMS) -> dict:
    table = {}
    names = [n for n in next(iter(scores.values()))["eval"] if n != CLEAN]
    for name in names:
        parts = []
        for category, s in scores.items():
            rng = drift.rng_for(f"capture-delay|{name}|{category}")
            streams, change = drift.split_change_streams(
                s["eval"][CLEAN], s["eval"][name], s["labels"], 0.0, n_streams, rng
            )
            theta = drift._theta(rng, streams.shape)
            parts.append((drift.first_alarm(drift.inductive_pvalues(streams, s["cal"], theta)), change))
        table[name] = drift.pool_delays(parts)
    return table


# ---------------------------------------------------------------- M2AD 학습 시편


# 검사기 하나의 촬영 통계(조명마다 [30, 11])와 시편 겹. 학습 시편만 읽는다.
def m2ad_inspector(rows, cache, category: str, view: str) -> dict:
    from .m2ad import ILLUMINATIONS, REFERENCE, specimen_folds
    from .run_m2ad import _same_specimens, inspector_rows

    train = {light: inspector_rows(rows, category, view, "train", light) for light in ILLUMINATIONS}
    names = [r.specimen for r in train[REFERENCE]]
    for light, picked in train.items():
        _same_specimens(picked, names, f"{category} {view} train, illumination {light}")
    fold_of = specimen_folds(names)
    return {
        "folds": np.array([fold_of[n] for n in names], dtype=np.int64),
        "features": {light: capture_features(cache.images(train[light])) for light in ILLUMINATIONS},
    }


# 시편 겹 교차 적합: 겹 f 시편의 사진(어느 조명이든)은 겹 f를 뺀 I01 사진으로 맞춘 모형으로 채점한다. 보정
# 집합(겹 f 시편의 p-value용)은 겹 f 밖 시편의 I01 교차 적합 점수다(규칙).
def m2ad_scores(data: dict) -> dict:
    from .m2ad import REFERENCE

    folds = data["folds"]
    ref = data["features"][REFERENCE]
    scores = {light: np.full(len(folds), np.nan) for light in data["features"]}
    for fold in np.unique(folds):
        held = folds == fold
        model = CaptureModel.fit(ref[~held])
        for light, f in data["features"].items():
            scores[light][held] = model.score(f[held])
    cal = {int(fold): scores[REFERENCE][folds != fold] for fold in np.unique(folds)}
    return {"folds": folds, "scores": scores, "cal": cal}


# 스트림마다 p-value: 사진이 속한 겹의 보정 집합으로 귀납형 p-value를 낸다(겹마다 따로 계산).
def _fold_pvalues(values: np.ndarray, folds: np.ndarray, cal: dict[int, np.ndarray], theta: np.ndarray):
    p = np.empty_like(values, dtype=np.float64)
    for fold, c in cal.items():
        mask = folds == fold
        p[mask] = drift.inductive_pvalues(values[mask], c, theta[mask])
    return p


# M2AD: 조명마다 게이트 비율(판정 없음)과 I01 → 조명 L 스트림의 경보. 스트림은 I01 30장(무작위 순서) 뒤에 같은
# 시편의 조명 L 30장(무작위 순서). "변화 뒤 20장 안 경보"는 첫 경보가 변화 뒤 1~20번째 사진에서 난 스트림의
# 비율이고, 변화 전에 경보한 스트림은 넣지 않는다(규칙).
def m2ad_tables(per_inspector: dict[str, dict], n_streams: int = N_M2AD_STREAMS) -> dict:
    from .m2ad import ILLUMINATIONS, REFERENCE

    gate, delay = {}, {}
    thresholds = {}
    for name, s in per_inspector.items():
        thr = conformal_threshold(s["scores"][REFERENCE], ALPHA_GATE)
        thresholds[name] = {"value": thr.value, "rank": thr.rank, "n": thr.n, "guaranteed": thr.guaranteed}
    for light in ILLUMINATIONS:
        gated = sum(
            int((s["scores"][light] > thresholds[n]["value"]).sum()) for n, s in per_inspector.items()
        )
        total = sum(len(s["folds"]) for s in per_inspector.values())
        gate[light] = {"gated": gated, "n": total, "rate": _rate(gated, total)}
        if light == REFERENCE:
            continue
        parts, early, within, streams_total = [], 0, 0, 0
        for name, s in per_inspector.items():
            rng = drift.rng_for(f"capture-m2ad|{name}|{light}")
            n = len(s["folds"])
            pre = np.stack([rng.permutation(n) for _ in range(n_streams)])
            post = np.stack([rng.permutation(n) for _ in range(n_streams)])
            values = np.concatenate([s["scores"][REFERENCE][pre], s["scores"][light][post]], axis=1)
            folds = np.concatenate([s["folds"][pre], s["folds"][post]], axis=1)
            theta = drift._theta(rng, values.shape)
            alarm = drift.first_alarm(_fold_pvalues(values, folds, s["cal"], theta))
            parts.append((alarm, n))
            early += int(((alarm >= 0) & (alarm < n)).sum())
            within += int(((alarm >= n) & (alarm < n + H27_WITHIN)).sum())
            streams_total += len(alarm)
        summary = drift.pool_delays(parts)
        summary["within_20_share"] = within / streams_total
        summary["early_alarms"] = early
        delay[light] = summary
    return {"gate_thresholds": thresholds, "gate": gate, "delay": delay}


# ---------------------------------------------------------------- 판정과 리포트


def h26(gate: dict) -> dict:
    blur = gate[H26_BLUR]["normal"]["rate"]
    defect = gate[CLEAN]["defect"]["rate"]
    ok = blur >= H26_MIN_GATED and defect <= H26_MAX_DEFECT
    return {
        "blur_normal_gated": blur,
        "clean_defect_gated": defect,
        "min_blur": H26_MIN_GATED,
        "max_defect": H26_MAX_DEFECT,
        "verdict": "지지" if ok else "기각",
    }


def h27(fa: dict, m2ad: dict) -> dict:
    rate = fa[f"{H27_RATE:.2f}"]["capture"]["rate"]
    shares = {light: row["within_20_share"] for light, row in m2ad["delay"].items()}
    passing = sum(share >= H27_MIN_SHARE for share in shares.values())
    ok = rate <= H27_MAX_FA and passing >= H27_MIN_LIGHTS
    return {
        "false_alarm_at_10pct": rate,
        "max_false_alarm": H27_MAX_FA,
        "within_20_share": shares,
        "lights_passing": passing,
        "min_lights": H27_MIN_LIGHTS,
        "verdict": "지지" if ok else "기각",
    }


# NaN·무한대를 JSON의 null로 바꾸고 numpy 값을 파이썬 값으로 바꾼다.
def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.integer | np.bool_):
        return value.item()
    if isinstance(value, float | np.floating):
        v = float(value)
        return v if math.isfinite(v) else None
    return value


def build_report(visa: dict[str, dict], m2ad: dict[str, dict], outputs: Path) -> dict:
    gate = gate_table(visa)
    fa = fa_table(visa, outputs)
    m2 = m2ad_tables(m2ad)
    return {
        "features": list(FEATURES),
        "floors": FLOORS.tolist(),
        "shrink": SHRINK,
        "alpha_gate": ALPHA_GATE,
        "visa": {
            "gate_thresholds": gate_thresholds(visa),
            "gate": gate,
            "with_serving_model": dms_table(visa, outputs),
            "false_alarms": fa,
            "perturb_delay": perturb_delay_table(visa),
        },
        "m2ad": m2,
        "hypotheses": {"H26": h26(gate), "H27": h27(fa, m2)},
    }


# ---------------------------------------------------------------- 9단계 보충: 결함 마스크 밖 촬영 통계

# DILATION: 결함 마스크 팽창의 체비셰프 반경(9×9 정사각형 구조 요소). 280px 이미지는 원본을 BICUBIC(축소할 때
# 안티앨리어싱)으로 줄인 것이라 결함 화소가 280px 격자에서 앞뒤 2px까지 번지고, 256 격자 마스크를 280px로 키울
# 때 위치가 1px 이하로 어긋나며, 3×3 라플라시안·Sobel이 1px을 더 본다(2 + 1 + 1). 측정 전에 고정했다.
DILATION = 4
# 대조 정상(겹 0)의 합산 게이트 비율이 이보다 크면 마스크 맞춤 절차가 어긋난 것으로 보고 결론을 내지 않는다
# (설계값 α_gate = 1%, 9단계 마스크 없는 값 1.1%). 같으면 넘은 것이 아니다.
CONTROL_MAX_GATE = 0.03
# 해석 규칙의 두 범주와 기준(분자, 분모): 20장 중 10장 이상이면 "결함 밖도 다름", 3장 이하면 "결함 영역 때문".
# 비율을 정수 곱으로 비교해 부동소수 경계 문제를 없애고, 장수가 20이 아닌 점검 실행에서도 같은 비율로 쓴다.
READ_CATEGORIES = ("pcb2", "pipe_fryum")
READ_OUTSIDE_MIN = (10, 20)
READ_REGION_MAX = (3, 20)
# 마스크 맞춤 리포트, 그리고 같은 실행에서 다시 낸 마스크 없는 게이트 수를 대조할 9단계 기록.
MASKED_REPORT = paths.REPORTS / "stage9" / "capture-masked.json"
STAGE9_REPORT = paths.REPORTS / "stage9" / "capture.json"


# 256 격자 결함 마스크 [h, w] → 통계에 쓸 화소 bool [size, size](팽창한 결함 마스크 밖이 True). 크기는 PIL
# NEAREST로 맞춰 칸 값을 그대로 옮기고(리사이즈 캐시와 같은 PIL), 팽창은 (2·radius + 1)² 정사각형으로 해서
# 결함 칸에서 체비셰프 거리 radius 안의 화소를 모두 뺀다. 이미지 밖은 결함이 아닌 것으로 본다(가장자리에서는
# 안쪽으로만 넓어진다). 마스크 값 > 0이 결함이다. radius = 0은 팽창 없이 확대만 한다(테스트에서 팽창의 몫을
# 보는 데 쓴다).
def outside_pixels(mask: np.ndarray, size: int, radius: int = DILATION) -> np.ndarray:
    """결함 마스크 밖 화소 bool [size, size]: 팽창한 결함 영역이 False, 나머지가 True."""
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"expected a single defect mask [h, w], got shape {m.shape}")
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    binary = Image.fromarray(np.where(m > 0, 255, 0).astype(np.uint8))
    defect = np.asarray(binary.resize((size, size), Image.Resampling.NEAREST)) > 0
    if radius > 0 and defect.any():
        defect = ndimage.binary_dilation(
            defect, structure=np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        )
    return ~defect


# 대조 정상의 마스크 배정(규칙): 겹 0 정상과 dev 결함을 각각 경로의 SHA-256(splits.path_key) 순으로 줄 세워
# i번째 정상에 (i mod D)번째 결함의 마스크를 준다. 결정적이고 입력 순서와 무관하며, 결함마다 받는 정상 수는
# 많아야 1장 차이다. 출력은 정상의 입력 순서대로, 배정된 결함의 defect_images 안 위치다.
def control_assignment(normal_images: Sequence[str], defect_images: Sequence[str]) -> np.ndarray:
    """정상마다 배정된 결함 번호 int64 [len(normal_images)]."""
    if len(defect_images) == 0:
        raise ValueError("need at least one defect mask to assign")
    normal_rank = sorted(range(len(normal_images)), key=lambda i: path_key(normal_images[i]))
    defect_rank = sorted(range(len(defect_images)), key=lambda j: path_key(defect_images[j]))
    out = np.empty(len(normal_images), dtype=np.int64)
    for rank, i in enumerate(normal_rank):
        out[i] = defect_rank[rank % len(defect_rank)]
    return out


# 범주 하나의 보충 자료: 정상 풀(겹 1~4) 사진과 겹 번호, 겹 0 정상 사진, dev 결함 사진과 그 256 격자 마스크.
# 9단계 visa_category와 같은 dev 절차 행만 고른다(splits.select는 dev 절차에서 봉인 test 행을 내주지 않는다).
# 교란은 주지 않는다(교란 없음만 본다).
def masked_category(cache, manifest, category: str) -> dict:
    from .splits import select

    pool = select(manifest, protocol="dev", part="pool_normal", category=category)
    normal = select(manifest, protocol="dev", part="eval_normal", category=category)
    defect = select(manifest, protocol="dev", part="eval_defect", category=category)
    return {
        "pool_images": cache.images(pool),
        "pool_folds": np.array([r.fold for r in pool], dtype=np.int64),
        "normal_images": cache.images(normal),
        "normal_names": [r.image for r in normal],
        "defect_images": cache.images(defect),
        "defect_names": [r.image for r in defect],
        "defect_types": [r.defect_types for r in defect],
        "masks": cache.masks(defect),
    }


# 9단계와 같은 맞춤: 풀 전체로 맞춘 모형(평가 사진 채점용)과 겹 교차 적합 점수의 α_gate conformal 임계값.
def _fit_gate(features: np.ndarray, folds: np.ndarray) -> tuple[CaptureModel, Threshold]:
    return CaptureModel.fit(features), conformal_threshold(crossfit_scores(features, folds), ALPHA_GATE)


# 변수별 표준화 값 z = (x − μ) / s(점수 모형이 쓰는 것과 같은 값) [n, 11].
def _z(model: CaptureModel, features: np.ndarray) -> np.ndarray:
    return (np.asarray(features, dtype=np.float64) - model.mean) / model.scale


# 범주 하나의 마스크 맞춤 채점.
# (1) 마스크 없음: 9단계 visa_scores와 같은 구성(풀 전체 모형, 겹 교차 적합 1% 임계값)으로 겹 0 정상과 결함을
#     채점한다. 9단계 기록과 대조하고, 마스크 맞춤 값과 나란히 싣는 데 쓴다.
# (2) 마스크 맞춤: 결함마다 그 마스크 밖 화소로 정상 풀 전부의 통계를 다시 내고, 같은 방식으로 모형·임계값을
#     다시 맞춰 그 결함 사진과, 그 마스크를 배정받은 겹 0 정상(대조)을 채점한다. 결함 사진에서만 화소를 빼면
#     화소 집합이 달라진 것만으로 통계가 바뀌므로 정상 풀에도 같은 마스크를 씌운다(규칙). 결함 하나마다 정상
#     풀 전체(240~483장)의 통계를 다시 내므로 범주 하나에 1~2분 걸린다(필터는 마스크와 무관하지만 9단계 함수를
#     그대로 쓰려고 다시 계산한다).
def masked_scores(data: dict, radius: int = DILATION) -> dict:
    """{"unmasked": {...}, "masked": {...}, "control": {...}, "defect_names", "defect_types", "size"}."""
    pool, folds = data["pool_images"], np.asarray(data["pool_folds"])
    normal, defect = data["normal_images"], data["defect_images"]
    n_normal, n_defect = len(data["normal_names"]), len(data["defect_names"])
    size = int(pool.shape[1])
    model, gate = _fit_gate(capture_features(pool), folds)
    eval_f = capture_features(np.concatenate([normal, defect]))
    eval_s = model.score(eval_f)
    unmasked = {
        "gate": gate,
        "normal_score": eval_s[:n_normal],
        "defect_score": eval_s[n_normal:],
        "defect_z": _z(model, eval_f[n_normal:]),
    }
    assign = control_assignment(data["normal_names"], data["defect_names"])
    gates = []
    scores, kept = np.empty(n_defect), np.empty(n_defect)
    zs = np.empty((n_defect, len(FEATURES)))
    control = np.full(n_normal, np.nan)
    for j in range(n_defect):
        keep = outside_pixels(data["masks"][j], size, radius)
        m_model, m_gate = _fit_gate(capture_features(pool, keep), folds)
        f = capture_features(defect[j : j + 1], keep)
        gates.append(m_gate)
        scores[j] = m_model.score(f)[0]
        zs[j] = _z(m_model, f)[0]
        kept[j] = keep.mean()
        mine = assign == j
        if mine.any():
            control[mine] = m_model.score(capture_features(normal[mine], keep))
    return {
        "unmasked": unmasked,
        "masked": {"gate": gates, "defect_score": scores, "defect_z": zs, "kept_share": kept},
        "control": {"mask": assign, "score": control},
        "defect_names": list(data["defect_names"]),
        "defect_types": list(data["defect_types"]),
        "size": size,
    }


def _count(flags: np.ndarray) -> list[int]:
    return [int(np.sum(flags)), int(len(flags))]


def _summary(count: Sequence[int]) -> dict:
    return {"gated": count[0], "n": count[1], "rate": _rate(count[0], count[1])}


# 같은 실행에서 다시 낸 마스크 없는 게이트 수(범주별 정상·결함의 [게이트 수, 장수])가 9단계 기록
# (visa.gate.clean.per_category)과 같은지 본다. 실행한 범주만 비교한다. 기록이 없으면(테스트) ok는 None이다.
def _stage9_check(unmasked: dict, recorded: dict | None) -> dict:
    if recorded is None:
        return {"ok": None, "mismatched": []}
    stored = recorded["visa"]["gate"][CLEAN]["per_category"]
    mismatched = [c for c, row in unmasked.items() if stored.get(c) != row]
    return {"ok": not mismatched, "mismatched": mismatched}


# 해석 규칙(측정 전에 고정, 판정 없음): pcb2와 pipe_fryum의 마스크 맞춤 결함 게이트 수 [게이트 수, 장수]로
# 읽는다. 9단계 게이트 수가 다시 나오지 않거나(check_ok False) 대조 정상 게이트가 3%를 넘으면(NaN도) 결론을
# 내지 않는다.
def masked_reading(counts: dict[str, Sequence[int]], control_rate: float, check_ok: bool | None) -> dict:
    """{"overall", "per_category", "counts", "control_rate", "control_max", "stage9_ok"}."""
    out = {
        "counts": {c: list(v) for c, v in counts.items()},
        "control_rate": control_rate,
        "control_max": CONTROL_MAX_GATE,
        "stage9_ok": check_ok,
        "per_category": None,
    }
    if check_ok is False:
        return out | {"overall": "결론 없음: 마스크 없는 9단계 게이트 수가 다시 나오지 않았다"}
    if not control_rate <= CONTROL_MAX_GATE:
        return out | {"overall": "결론 없음: 대조 정상의 게이트 비율이 3%를 넘었다"}
    if any(c not in counts for c in READ_CATEGORIES):
        return out | {"overall": "해석 없음: pcb2와 pipe_fryum이 이 실행에 다 들어 있지 않다"}
    labels = {}
    for c in READ_CATEGORIES:
        gated, n = counts[c]
        if gated * READ_OUTSIDE_MIN[1] >= READ_OUTSIDE_MIN[0] * n:
            labels[c] = "결함 밖도 다름"
        elif gated * READ_REGION_MAX[1] <= READ_REGION_MAX[0] * n:
            labels[c] = "결함 영역 때문"
        else:
            labels[c] = "가르지 못함"
    found = set(labels.values())
    if found == {"결함 밖도 다름"}:
        overall = "결함 밖 영역도 정상과 촬영 통계가 다르다(촬영 조건 차이 가능성)"
    elif found == {"결함 영역 때문"}:
        overall = "게이트는 결함 영역 때문이었다"
    elif found == {"결함 밖도 다름", "결함 영역 때문"}:
        overall = "범주마다 갈림"
    else:
        overall = "가르지 못했다"
    return out | {"per_category": labels, "overall": overall}


# 범주별 결과를 리포트로 모은다: 마스크 없음(9단계)과 마스크 맞춤의 게이트 수(범주별 [게이트 수, 장수]와
# 합산), 대조 정상의 게이트 수, 변수별 z 평균(결함 사진 평균), 마스크 밖 화소 비율, 결함마다의 줄, 9단계 기록
# 대조, 해석.
def masked_tables(per_category: dict[str, dict], recorded: dict | None) -> dict:
    unmasked_pc, masked_pc, thresholds = {}, {}, {}
    totals = {key: [0, 0] for key in ("normal", "defect", "masked_defect", "control")}
    z_mean, kept_share, defects = {}, {}, []
    for category, s in per_category.items():
        u, m, c = s["unmasked"], s["masked"], s["control"]
        gate_values = np.array([g.value for g in m["gate"]])
        u_normal = u["normal_score"] > u["gate"].value
        u_defect = u["defect_score"] > u["gate"].value
        m_defect = m["defect_score"] > gate_values
        control = c["score"] > gate_values[c["mask"]]
        unmasked_pc[category] = {"normal": _count(u_normal), "defect": _count(u_defect)}
        masked_pc[category] = {
            "defect": _count(m_defect),
            "control": _count(control),
            "pool": m["gate"][0].n,
            "gates_guaranteed": int(sum(g.guaranteed for g in m["gate"])),
        }
        thresholds[category] = {
            "value": u["gate"].value,
            "rank": u["gate"].rank,
            "n": u["gate"].n,
            "guaranteed": u["gate"].guaranteed,
        }
        for key, flags in (
            ("normal", u_normal),
            ("defect", u_defect),
            ("masked_defect", m_defect),
            ("control", control),
        ):
            totals[key][0] += int(flags.sum())
            totals[key][1] += len(flags)
        z_mean[category] = {
            "masked": dict(zip(FEATURES, m["defect_z"].mean(axis=0), strict=True)),
            "unmasked": dict(zip(FEATURES, u["defect_z"].mean(axis=0), strict=True)),
        }
        kept_share[category] = {
            "median": float(np.median(m["kept_share"])),
            "min": float(m["kept_share"].min()),
        }
        for j, name in enumerate(s["defect_names"]):
            defects.append(
                {
                    "category": category,
                    "image": name,
                    "defect_types": s["defect_types"][j],
                    "kept_share": m["kept_share"][j],
                    "unmasked": {"score": u["defect_score"][j], "gated": bool(u_defect[j])},
                    "masked": {
                        "score": m["defect_score"][j],
                        "gate": gate_values[j],
                        "gated": bool(m_defect[j]),
                    },
                }
            )
    check = _stage9_check(unmasked_pc, recorded)
    control_rate = _rate(*totals["control"])
    reading_counts = {c: masked_pc[c]["defect"] for c in READ_CATEGORIES if c in masked_pc}
    return {
        "features": list(FEATURES),
        "floors": FLOORS.tolist(),
        "shrink": SHRINK,
        "alpha_gate": ALPHA_GATE,
        "size": next(iter(per_category.values()))["size"],
        "dilation": DILATION,
        "control_max_gate": CONTROL_MAX_GATE,
        "stage9_check": check,
        "gate": {
            "unmasked": {
                "normal": _summary(totals["normal"]),
                "defect": _summary(totals["defect"]),
                "per_category": unmasked_pc,
                "thresholds": thresholds,
            },
            "masked": {
                "defect": _summary(totals["masked_defect"]),
                "control": _summary(totals["control"]),
                "per_category": masked_pc,
            },
        },
        "z_mean": z_mean,
        "kept_share": kept_share,
        "reading": masked_reading(reading_counts, control_rate, check["ok"]),
        "defects": defects,
    }


# 명령줄 --masked의 본체: 범주마다 자료를 읽어 마스크 맞춤 채점을 하고 리포트를 만든다. dev 절차 행만 읽으므로
# 장부에 남기지 않는다.
def run_masked(cache, manifest, categories: Sequence[str], recorded: dict | None, started: float) -> dict:
    per = {}
    for category in categories:
        per[category] = masked_scores(masked_category(cache, manifest, category))
        print(f"[masked] {category} {time.perf_counter() - started:.0f} s", flush=True)
    return masked_tables(per, recorded)


# 명령줄 진입점. VisA 280px 캐시와 M2AD 256px 캐시에서 dev 절차·학습 시편만 읽는다. 봉인 테스트 행은 고르지
# 않으므로 장부에 남기지 않는다. --masked는 VisA dev만 읽는다(M2AD 캐시는 열지 않는다).
def main(argv: list[str] | None = None) -> None:
    from .cache import ImageCache
    from .m2ad import CATEGORIES, VIEWS, M2adCache, read_meta
    from .splits import read_manifest
    from .visa import CATEGORIES as VISA_CATEGORIES

    parser = argparse.ArgumentParser(
        description=(
            "9단계: 촬영 통계 품질 게이트와 변화 감지(CPU, 봉인 데이터 없음). "
            "--masked는 9단계 보충: dev 결함 마스크 밖 화소만으로 같은 통계를 내는 마스크 맞춤 게이트."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="리포트 경로(기본: reports/stage9/capture.json, --masked면 reports/stage9/capture-masked.json)",
    )
    parser.add_argument("--visa-size", type=int, default=280, help="VisA 리사이즈 캐시 크기(서빙 모델 입력)")
    parser.add_argument("--m2ad-size", type=int, default=256, help="M2AD 리사이즈 캐시 크기")
    parser.add_argument(
        "--categories", nargs="+", default=list(VISA_CATEGORIES), help="VisA 범주(점검용으로 줄일 때만)"
    )
    parser.add_argument(
        "--masked",
        action="store_true",
        help=(
            "9단계 보충만 돈다: 결함마다 팽창한 마스크 밖 화소로 정상 풀의 모형·게이트를 다시 맞춰 그 결함과 "
            "대조 정상을 채점한다(VisA dev만, M2AD는 읽지 않는다). 규칙은 docs/experiments.md '9단계 보충'"
        ),
    )
    parser.add_argument(
        "--recorded",
        type=Path,
        default=STAGE9_REPORT,
        help=(
            "--masked에서 같은 실행의 마스크 없는 게이트 수를 대조할 9단계 리포트(범주별 수가 다르면 결론을 "
            "내지 않는다)"
        ),
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    manifest = read_manifest(paths.VISA_MANIFEST)
    cache = ImageCache(paths.CACHE, args.visa_size)
    if args.masked:
        recorded = json.loads(Path(args.recorded).read_text(encoding="utf-8"))
        report = run_masked(cache, manifest, args.categories, recorded, started)
        out, summary = args.out or MASKED_REPORT, report["reading"]
    else:
        conditions = condition_names(include_clean=True)
        visa = {}
        for category in args.categories:
            visa[category] = visa_scores(visa_category(cache, manifest, category, conditions))
            print(f"[visa] {category} {time.perf_counter() - started:.0f} s", flush=True)
        rows = read_meta(paths.RAW / "m2ad" / "jsons.zip")
        m_cache = M2adCache(paths.CACHE, args.m2ad_size)
        m2ad = {}
        for category in CATEGORIES:
            for view in VIEWS:
                m2ad[f"{category}_{view}"] = m2ad_scores(m2ad_inspector(rows, m_cache, category, view))
        print(f"[m2ad] {time.perf_counter() - started:.0f} s", flush=True)
        report = build_report(visa, m2ad, paths.OUTPUTS)
        out, summary = args.out or paths.REPORTS / "stage9" / "capture.json", report["hypotheses"]
    report["seconds"] = round(time.perf_counter() - started, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_plain(report), f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(_plain(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
