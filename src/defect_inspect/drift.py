"""이상 점수 흐름에서 라벨 없이 변화를 감지한다(7단계, 판정 없이 보고).

새 이미지의 점수마다 conformal p-value를 만든다: 귀납형(정상 이미지의 고정된 보정 점수 집합과 비교) 또는
전도형(스트림이 지금까지 보인 점수, 자기 포함과 비교). p-value를 거듭제곱 베팅 함수를 쓰는 jumper test
martingale에 넣고, 자본이 1/DELTA에 닿으면 경보한다. 점수가 교환 가능하면 전도형 p-value는 서로 독립인
균등분포라, Ville 부등식으로 스트림 전체에서 허위 경보가 한 번이라도 날 확률이 DELTA 이하로 묶인다.
귀납형은 같은 보정 집합을 공유하므로 이 보장이 없다.

`python -m defect_inspect.drift`는 저장된 점수만 읽고(이미지·GPU 없음) reports/stage7/drift.json을 쓴다.
규칙은 docs/experiments.md 7단계에 있다.
"""

# [파일 역할] 7단계 "라벨 없는 변화 감지" 모듈이다. 검사기가 내는 이상 점수의 흐름만 보고, 촬영 조건이 바뀌어
# 고정 임계값이 무너졌다는 것을 결함 라벨 없이 알아챌 수 있는지 잰다(판정 없음).
# [왜 필요한가] 3-B에서 조명이 바뀌면 오검출률이 96.9%가 되고, 새 조명의 정상 8장으로 재보정하면 3.5%로
# 돌아왔다(H12). 남은 문제는 "언제 재보정해야 하는지"를 라벨 없이 아는 장치다(docs/experiments.md 7단계 도입).
# [방법] 위 docstring의 방법이다. δ = 1%라 자본이 1/δ = 100에 닿으면 경보한다(Vovk et al. 2021).
# [흐름] 저장된 점수만 읽는다(이미지·GPU 없음, 봉인 테스트를 다시 읽지 않아 장부 줄도 없음). (1) VisA 봉인
# test 정상 스트림의 허위 경보율(결함 0/5/10% 섞음), (2) VisA 합성 교란 전후 스트림의 경보 지연, (3) M2AD 실제
# 조명 I01 → 다른 조명 스트림의 경보 지연을 계산해 reports/stage7/drift.json에 쓴다.
# [결과 요약] 전도형 허위 경보 0.22~0.28%(결함 10%에서도 0.19~0.22%). 귀납형은 M2AD 새 조명에서 P0가 6~7장
# 만에 경보했지만 결함이 10% 섞인 VisA 스트림의 34~61%에서도 경보했다(README 7단계).

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

# numpy로 스트림 수천 개를 배열 하나로 묶어 한꺼번에 계산한다. path_key(문자열의 SHA-256)는 설정 문자열에서
# 난수 시드를 만드는 데 쓴다.
import numpy as np

from . import paths
from .splits import path_key

# 마팅게일 설정(docs/experiments.md 7단계, 실제 점수를 보기 전에 합성 스트림으로 고름).
# - EPSILONS: 거듭제곱 베팅 함수 ε·p^(ε−1)의 ε 네 개. 이 함수는 p가 균등분포일 때 기댓값이 1이라(∫ε·p^(ε−1)dp
#   = 1) 정상일 때는 자본이 늘지 않고, 작은 p가 이어지면 빠르게 커진다. ε = 1은 항상 1이라 "걸지 않음"이다.
#   원래 Simple Jumper의 1 + ε(p − 1/2)는 한 장에 최대 1.5배라 합성 조건에서 경보까지 중앙값 18~24장이 걸렸고,
#   이 집합은 4~7장이었다.
# - JUMP(J = 0.01): 매 베팅 전에 전체 자본의 1%를 함수들에 고르게 다시 나눈다.
# - DELTA(δ = 1%): 자본이 1/δ = 100에 닿으면 경보. 허위 경보율의 설계 목표다.
EPSILONS = (1.0, 0.5, 0.2, 0.05)
JUMP = 0.01
DELTA = 0.01
# RATES: 스트림에 섞는 결함 비율(0/5/10%). 결함이 섞인 스트림의 경보는 "결함이 이어서 나오는 것도 변화로
# 잡는다"는 한계를 보여 준다. 스트림 수: 허위 경보는 범주마다 1,000개, VisA 지연은 범주마다 200개, M2AD 지연은
# 검사기마다 1,000개(규칙). _CHUNK: 전도형 p-value의 쌍 비교를 스트림 100개씩 나눠 [100, T, T] 크기 배열로
# 메모리를 묶는다.
RATES = (0.0, 0.05, 0.10)
DETECTORS = ("inductive", "transductive")
N_FA_STREAMS = 1000
N_DELAY_STREAMS = 200
N_M2AD_STREAMS = 1000
_CHUNK = 100


# 설정 문자열은 예를 들어 "fa|p0|candle|0.05"이고, 그 SHA-256 앞 16자리를 시드로 쓴다. 스트림 순서, 섞을 결함,
# θ가 모두 이 생성기에서 나와 결과가 결정적이고, 설정마다 시드가 달라 서로 겹치지 않는다.
def rng_for(key: str) -> np.random.Generator:
    """`key`의 SHA-256으로 시드를 정한 생성기. 스트림 묶음마다 고정된 자기 시드를 가진다."""
    return np.random.default_rng(int(path_key(key)[:16], 16))


# θ는 동점을 무작위로 쪼개는 데 쓴다. 0을 빼서 베팅 함수의 p^(ε−1)이 무한대가 되는 것을 막는다.
def _theta(rng: np.random.Generator, shape) -> np.ndarray:
    """(0, 1] 위의 균등분포 θ를 뽑는다: p-value가 정확히 0이 되지 않는다."""
    return 1.0 - rng.random(shape)


# 귀납형 conformal p-value. 점수가 클수록 이상하므로 큰 점수일수록 p가 작다. 보정 점수는 그 검사기의 임계값을
# 정한 바로 그 점수다(P0·D-S: 학습 풀 교차 적합, DM·DMS: 겹 0 홀드아웃, M2AD: 학습 시편 30장의 교차 적합).
# 정렬한 보정 점수에 searchsorted 두 번으로 "큰 수"와 "같은 수"를 한꺼번에 센다.
def inductive_pvalues(scores: np.ndarray, cal: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """`scores`의 원소마다 p = (#{cal > s} + theta * (#{cal = s} + 1)) / (n + 1)을 계산한다."""
    cal = np.sort(np.asarray(cal, dtype=np.float64))
    if cal.size == 0:
        raise ValueError("inductive p-values need calibration scores")
    s = np.asarray(scores, dtype=np.float64)
    right = np.searchsorted(cal, s, side="right")
    left = np.searchsorted(cal, s, side="left")
    return ((cal.size - right) + theta * (right - left + 1)) / (cal.size + 1)


# 전도형 conformal p-value. 보정 집합이 필요 없지만
# 변화 전 기준 구간이 길어야 한다(M2AD처럼 변화 전 20장뿐이면 늦다).
# 구현: 하삼각 행렬(past[t, i] = i ≤ t)로 모든 (t, i) 쌍을 한 번에 비교해 "지난 점수 중 큰 수"와 "같은 수"를
# 센다. 스트림 100개씩 나눠 메모리를 묶는다.
def transductive_pvalues(scores: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """스트림(행)마다 p_t = (#{i <= t: s_i > s_t} + theta_t * #{i <= t: s_i = s_t}) / t (t는 1부터).

    t번째 점수를 그때까지 본 점수(자기 포함) 가운데의 순위로 바꾸는 것이다.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"expected scores [streams, time], got {s.shape}")
    n_streams, length = s.shape
    past = np.tril(np.ones((length, length), dtype=bool))  # [t, i]: i <= t
    out = np.empty_like(s)
    for start in range(0, n_streams, _CHUNK):
        block = s[start : start + _CHUNK]
        later, earlier = block[:, :, None], block[:, None, :]
        greater = ((earlier > later) & past).sum(axis=2)
        equal = ((earlier == later) & past).sum(axis=2)
        out[start : start + _CHUNK] = (greater + theta[start : start + _CHUNK] * equal) / np.arange(
            1, length + 1
        )
    return out


# 위치는 0부터 센다. 자본은 베팅 함수들에 1/k씩 고르게 나눠 시작하고, 매 관측 전에 J만큼 다시 고르게 나눈 뒤
# 각 함수의 배수를 곱한다. 모든 스트림을 배열로 한꺼번에 갱신한다.
def first_alarm(
    pvalues: np.ndarray, epsilons: Sequence[float] = EPSILONS, jump: float = JUMP, delta: float = DELTA
) -> np.ndarray:
    """jumper 마팅게일이 1/delta에 처음 닿는 관측의 위치를 돌려준다(끝내 닿지 않으면 -1).

    자본은 베팅 함수 eps * p**(eps - 1)들에 고르게 나눠 시작한다. 매 베팅 전에 전체의 `jump` 몫을 다시
    고르게 나눈다(Vovk et al. 2021, Simple Jumper). 그래서 오래 조용한 동안 잃은 함수도 변화 뒤에 빨리
    다시 이길 수 있다. 행은 서로 독립인 스트림이다.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"expected p-values [streams, time], got {p.shape}")
    # p가 (0, 1] 밖이면 베팅 배수가 정의되지 않으므로 입력을 거절한다. ε도 (0, 1]이어야 기댓값 1인 베팅
    # 함수다.
    if not ((p > 0) & (p <= 1)).all():
        raise ValueError("p-values must lie in (0, 1]")
    eps = np.asarray(epsilons, dtype=np.float64)[None, :]
    if not ((eps > 0) & (eps <= 1)).all():
        raise ValueError("every epsilon must lie in (0, 1]")
    k = eps.shape[1]
    capital = np.full((p.shape[0], k), 1.0 / k)
    alarm = np.full(p.shape[0], -1, dtype=np.int64)
    live = np.ones(p.shape[0], dtype=bool)
    threshold = 1.0 / delta
    for t in range(p.shape[1]):
        if not live.any():
            break
        # 아직 경보가 나지 않은 스트림만 갱신한다. 경보가 난 스트림은 live에서 빠져 첫 경보만 세고, 그 행의
        # 자본은 더 갱신하지 않는다.
        c = capital[live]
        c = (1.0 - jump) * c + jump * c.sum(axis=1, keepdims=True) / k
        c = c * eps * p[live, t : t + 1] ** (eps - 1.0)
        capital[live] = c
        hit = np.flatnonzero(live)[c.sum(axis=1) >= threshold]
        alarm[hit] = t
        live[hit] = False
    return alarm


# 계산은 round(rate / (1 − rate) × n)이다. 그 범주의 결함 수(VisA test는 40장)를 넘으면 그 수로 자른다. 그래서
# 결함 10%는 실제로 10%보다 조금 낮은 범주가 있다(7단계 결과 표 아래 설명).
def n_mixed(n_normal: int, rate: float, available: int) -> int:
    """정상 `n_normal`장에 섞을 결함 수: 결함이 스트림의 `rate`가 되게 한다(있는 결함 수로 상한)."""
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"rate must be in [0, 1), got {rate}")
    want = math.floor(rate / (1.0 - rate) * n_normal + 0.5)
    return min(want, int(available))


def _segment(rng: np.random.Generator, normal: np.ndarray, defect: np.ndarray, k: int) -> tuple:
    """`normal` 점수 전부와 `defect` 점수 k개를 무작위 순서로 섞은 한 구간: (점수, 결함 여부)."""
    chosen = rng.permutation(defect.size)[:k]
    scores = np.concatenate([normal, defect[chosen]])
    is_defect = np.concatenate([np.zeros(normal.size, bool), np.ones(k, bool)])
    order = rng.permutation(scores.size)
    return scores[order], is_defect[order]


# 같은 스트림, 같은 θ를 쓰는 것은 둘의 차이가 p-value 방식에서만 나오게 하려는 것이다(규칙). 보정 점수가
# 없으면 전도형만 돈다.
def detect(scores: np.ndarray, cal: np.ndarray | None, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """같은 스트림과 같은 무작위화로 두 감지기의 첫 경보 위치를 구한다."""
    theta = _theta(rng, scores.shape)
    out = {"transductive": first_alarm(transductive_pvalues(scores, theta))}
    if cal is not None:
        out["inductive"] = first_alarm(inductive_pvalues(scores, cal, theta))
    return out


# 정상만 있으면 변화가 없는 스트림이므로, 여기서 나는 경보는 모두 허위 경보다.
def false_alarm_streams(
    normal: np.ndarray, defect: np.ndarray, rate: float, n_streams: int, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """허위 경보용 [n_streams, T]: 범주의 정상 점수 전부 한 번씩 + 결함 k개, 스트림마다 다른 무작위 순서."""
    normal, defect = np.asarray(normal, np.float64), np.asarray(defect, np.float64)
    k = n_mixed(normal.size, rate, defect.size)
    rows = [_segment(rng, normal, defect, k)[0] for _ in range(n_streams)]
    return np.stack(rows), k


# M2AD에서는 변화 전 구간이 I01의 정상 + 결함, 변화 뒤 구간이 새 조명의 정상 + 결함이다. 같은 시편이 두
# 조명에서 한 번씩 나온다.
def change_streams(
    pre: tuple[np.ndarray, np.ndarray],
    post: tuple[np.ndarray, np.ndarray],
    rate: float,
    n_streams: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, tuple[int, int]]:
    """변화 지연용 [n_streams, T]: 변화 전 구간 뒤에 변화 뒤 구간을 붙인다. 각 구간은 (정상, 결함)을 섞는다.

    스트림, 변화 뒤 첫 관측의 위치, 구간별로 섞은 결함 수 (전, 후)를 돌려준다.
    """
    (pre_n, pre_d), (post_n, post_d) = [
        (np.asarray(a, np.float64), np.asarray(b, np.float64)) for a, b in (pre, post)
    ]
    k_pre = n_mixed(pre_n.size, rate, pre_d.size)
    k_post = n_mixed(post_n.size, rate, post_d.size)
    rows = [
        np.concatenate([_segment(rng, pre_n, pre_d, k_pre)[0], _segment(rng, post_n, post_d, k_post)[0]])
        for _ in range(n_streams)
    ]
    return np.stack(rows), pre_n.size + k_pre, (k_pre, k_post)


# VisA 합성 교란용이다. 같은 이미지가 양쪽에 나오지 않아 "교란 전후의 같은 사진"이 감지를 쉽게 만드는 일이
# 없다.
def split_change_streams(
    clean: np.ndarray,
    changed: np.ndarray,
    labels: np.ndarray,
    rate: float,
    n_streams: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """이미지 집합 하나로 변화 지연용 스트림을 만든다: 스트림마다 정상(과 결함)을 무작위로 반씩 나눠, 한
    절반은 변화 전에 교란 없는 점수로, 나머지 절반은 변화 뒤에 교란 조건의 점수로 늘어놓는다.

    어떤 이미지도 두 구간에 함께 나오지 않는다. 스트림과 변화 뒤 첫 관측의 위치를 돌려준다.
    """
    clean, changed = np.asarray(clean, np.float64), np.asarray(changed, np.float64)
    normal, defect = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    half_n, half_d = normal.size // 2, defect.size // 2
    k_pre = n_mixed(half_n, rate, half_d)
    k_post = n_mixed(normal.size - half_n, rate, defect.size - half_d)
    rows = []
    for _ in range(n_streams):
        n_perm, d_perm = rng.permutation(normal), rng.permutation(defect)
        pre = _segment(rng, clean[n_perm[:half_n]], clean[d_perm[:half_d]], k_pre)[0]
        post = _segment(rng, changed[n_perm[half_n:]], changed[d_perm[half_d:]], k_post)[0]
        rows.append(np.concatenate([pre, post]))
    return np.stack(rows), half_n + k_pre


# 허위 경보 요약: 스트림 수, 경보가 한 번이라도 난 스트림 수와 그 비율(허위 경보율의 정의).
def summarise_fa(alarm: np.ndarray) -> dict:
    return {"streams": int(alarm.size), "alarms": int((alarm >= 0).sum()), "rate": float((alarm >= 0).mean())}


def delays(alarm: np.ndarray, change: int) -> tuple[int, np.ndarray]:
    """(변화 전에 난 조기 경보 수, 나머지 스트림의 지연)을 돌려준다.

    지연은 경보가 난 장까지 포함한 변화 뒤 관측 수다(1 = 변화 뒤 첫 장에서 경보). 변화 뒤에 끝내 경보가
    없는 스트림은 무한대다."""
    alarm = np.asarray(alarm)
    early = (alarm >= 0) & (alarm < change)
    rest = alarm[~early]
    return int(early.sum()), np.where(rest >= change, rest - change + 1, np.inf).astype(np.float64)


# 요약 칸: 조기 경보 비율, 변화 뒤 구간 안에서 경보한 비율, 지연의 중앙값과 90백분위. 절반 넘게 경보가 없으면
# 중앙값이 None이 된다(README 표의 "-"). 백분위는 method="higher"로 실제 스트림 값 하나를 고르게 해, 무한대와
# 유한값 사이를 보간하지 않는다.
def pool_delays(parts: Sequence[tuple[np.ndarray, int]]) -> dict:
    """여러 범주·검사기의 스트림(각자 변화 위치를 가짐)을 합쳐 지연을 요약한다.

    중앙값과 90백분위는 경보 없는 스트림을 무한히 늦은 것으로 센다. 값이 그런 스트림에 떨어지면 None이다.
    """
    early, rest = 0, []
    for alarm, change in parts:
        e, d = delays(alarm, change)
        early += e
        rest.append(d)
    late = np.concatenate(rest) if rest else np.zeros(0)
    streams = early + late.size

    def finite_or_none(x: float) -> float | None:
        return float(x) if math.isfinite(x) else None

    return {
        "streams": streams,
        "early_share": early / streams if streams else float("nan"),
        "alarmed_share": float(np.isfinite(late).mean()) if late.size else float("nan"),
        "median_delay": finite_or_none(np.median(late)) if late.size else None,
        "p90_delay": finite_or_none(np.percentile(late, 90, method="higher")) if late.size else None,
    }


# ---------------------------------------------------------------- 저장된 점수


# 저장된 점수의 출처(봉인 test 실행 폴더, 테스트 점수 키, 보정 점수 키). P0·D-S는 전체 뱅크
# 점수(eval_score_full)와 교차 적합 점수(pool_score_oof), DM과 DMS(dms-280-car, ONNX CPU)는 평가 점수와 겹 0
# 홀드아웃 점수를 쓴다. 각 검사기의 임계값을 정한 그 점수가 귀납형의 보정 집합이다. 교란 지연은 3-A 실행이
# 있는 세 방법, M2AD 지연은 3-B의 두 방법만 잰다.
VISA_SOURCES = {
    # 방법: (실행 폴더, 테스트 점수 키, 보정 점수 키)
    "p0": ("p0-test", "eval_score_full", "pool_score_oof"),
    "d-s": ("d-s-test", "eval_score_full", "pool_score_oof"),
    "dm": ("dm-test", "eval_score", "cal_score"),
    "dms-280-car": ("dms-280-car-onnx-fp32-test", "eval_score", "cal_score"),
}
PERTURB_METHODS = ("p0", "d-s", "dm")
M2AD_METHODS = ("p0", "d-s")


# 한 방법의 VisA test 실행에서 범주별 (점수, 라벨, 보정 점수)를 읽는다. 점수 키가 없는 npz(다른 형식의 파일)는
# 건너뛰고, 하나도 없으면 실행을 먼저 하라는 뜻으로 FileNotFoundError를 낸다.
def load_visa(method: str, outputs: Path) -> dict[str, dict[str, np.ndarray]]:
    run, score_key, cal_key = VISA_SOURCES[method]
    out = {}
    for path in sorted((Path(outputs) / run).glob("*.npz")):
        with np.load(path) as z:
            if score_key not in z.files:
                continue
            out[path.stem] = {"scores": z[score_key], "labels": z["eval_labels"], "cal": z[cal_key]}
    if not out:
        raise FileNotFoundError(f"no saved {method} test scores in {Path(outputs) / run}")
    return out


# 결함 비율 × 범주마다 스트림 1,000개를 만들고, 12개 범주를 합친 허위 경보율과 범주별 값, 섞은 결함 수를
# 적는다. 귀납형의 DM·DMS 허위 경보가 결함 0%에서도 1.5~1.8%로 목표를
# 넘었고 대부분 pcb1에서 나왔다(7단계 결과).
def fa_table(method: str, outputs: Path, n_streams: int = N_FA_STREAMS) -> dict:
    """허위 경보 표: 한 방법의 저장된 VisA test 점수로 만든 스트림에서 감지기·결함 비율별 허위 경보."""
    cats = load_visa(method, outputs)
    table: dict = {}
    for rate in RATES:
        alarms = {d: [] for d in DETECTORS}
        per_category: dict = {}
        mixed = {}
        for category, arr in cats.items():
            labels = arr["labels"]
            rng = rng_for(f"fa|{method}|{category}|{rate}")
            streams, k = false_alarm_streams(
                arr["scores"][labels == 0], arr["scores"][labels == 1], rate, n_streams, rng
            )
            mixed[category] = k
            found = detect(streams, arr["cal"], rng)
            per_category[category] = {d: summarise_fa(found[d])["rate"] for d in DETECTORS}
            for d in DETECTORS:
                alarms[d].append(found[d])
        table[f"{rate:.2f}"] = {d: summarise_fa(np.concatenate(alarms[d])) for d in DETECTORS} | {
            "mixed_defects": mixed,
            "per_category": per_category,
        }
    return table


# 점수는 3-A의 test 교란 실행 `perturb-<방법>-test`에서 읽는다. 조건 목록의 첫 항목(clean)은 기준이라 빼고,
# 범주마다 200개 스트림을 split_change_streams로 만든다.
def perturb_delay_table(method: str, outputs: Path, n_streams: int = N_DELAY_STREAMS) -> dict:
    """교란 없는 점수에서 합성 교란 15조건 각각의 점수로 바뀐 뒤의 경보 지연 표(VisA test)."""
    run = Path(outputs) / f"perturb-{method}-test"
    cats = {}
    for path in sorted(run.glob("*.npz")):
        with np.load(path) as z:
            cats[path.stem] = {k: z[k] for k in ("conditions", "scores", "eval_labels", "cal_score")}
    if not cats:
        raise FileNotFoundError(f"no perturbation run in {run}")
    conditions = [str(c) for c in next(iter(cats.values()))["conditions"]]
    table: dict = {}
    for condition in conditions[1:]:
        table[condition] = {}
        for rate in RATES:
            pooled = {d: [] for d in DETECTORS}
            for category, arr in cats.items():
                names = [str(c) for c in arr["conditions"]]
                rng = rng_for(f"visa-delay|{method}|{category}|{condition}|{rate}")
                streams, change = split_change_streams(
                    arr["scores"][names.index("clean")],
                    arr["scores"][names.index(condition)],
                    arr["eval_labels"],
                    rate,
                    n_streams,
                    rng,
                )
                found = detect(streams, arr["cal_score"], rng)
                for d in DETECTORS:
                    pooled[d].append((found[d], change))
            table[condition][f"{rate:.2f}"] = {d: pool_delays(pooled[d]) for d in DETECTORS}
    return table


# 점수는 3-B 실행 `m2ad-<방법>`에서 읽는다. 조건 이름이 "S"(기준 조명 I01)와
# "R:<조명>"(실제 조명)으로 저장되어 있어, 검사기마다 S 구간 뒤에 R 구간을 이어 붙인다. 보정 점수는 그
# 검사기의 학습 시편 교차 적합 점수다.
def m2ad_delay_table(method: str, outputs: Path, n_streams: int = N_M2AD_STREAMS) -> dict:
    """I01에서 다른 실제 조명 각각으로 바뀐 뒤의 경보 지연 표(3-B 저장 점수)."""
    run = Path(outputs) / f"m2ad-{method}"
    with open(run / "run.json", encoding="utf-8") as f:
        meta = json.load(f)
    inspectors = {}
    for info in meta["inspectors"]:
        name = f"{info['category']}_{info['view']}"
        with np.load(run / f"{name}.npz") as z:
            inspectors[name] = {k: z[k] for k in ("conditions", "scores", "labels", "cal_score")}
    lights = [
        c.split(":")[1] for c in next(iter(inspectors.values()))["conditions"].tolist() if c.startswith("R:")
    ]
    table: dict = {}
    for light in lights:
        table[light] = {}
        for rate in RATES:
            pooled = {d: [] for d in DETECTORS}
            for name, arr in inspectors.items():
                conditions = arr["conditions"].tolist()
                s_ref, s_new = conditions.index("S"), conditions.index(f"R:{light}")
                pre = (
                    arr["scores"][s_ref][arr["labels"][s_ref] == 0],
                    arr["scores"][s_ref][arr["labels"][s_ref] == 1],
                )
                post = (
                    arr["scores"][s_new][arr["labels"][s_new] == 0],
                    arr["scores"][s_new][arr["labels"][s_new] == 1],
                )
                rng = rng_for(f"m2ad-delay|{method}|{name}|{light}|{rate}")
                streams, change, _ = change_streams(pre, post, rate, n_streams, rng)
                found = detect(streams, arr["cal_score"], rng)
                for d in DETECTORS:
                    pooled[d].append((found[d], change))
            table[light][f"{rate:.2f}"] = {d: pool_delays(pooled[d]) for d in DETECTORS}
    return table


# 설정값과 세 표(허위 경보, VisA 지연, M2AD 지연)를 하나의 리포트 dict로 모은다.
def build_report(
    outputs: Path,
    fa_streams: int = N_FA_STREAMS,
    delay_streams: int = N_DELAY_STREAMS,
    m2ad_streams: int = N_M2AD_STREAMS,
) -> dict:
    return {
        "settings": {
            "epsilons": list(EPSILONS),
            "jump": JUMP,
            "delta": DELTA,
            "rates": list(RATES),
            "fa_streams_per_category": fa_streams,
            "delay_streams_per_category": delay_streams,
            "m2ad_streams_per_inspector": m2ad_streams,
        },
        "false_alarms": {m: fa_table(m, outputs, fa_streams) for m in VISA_SOURCES},
        "visa_delay": {m: perturb_delay_table(m, outputs, delay_streams) for m in PERTURB_METHODS},
        "m2ad_delay": {m: m2ad_delay_table(m, outputs, m2ad_streams) for m in M2AD_METHODS},
    }


# 콘솔에 찍을 요약: 방법별 허위 경보율과, 결함 0%의 지연 중앙값(귀납형/전도형).
def format_report(report: dict) -> str:
    lines = ["false alarms on VisA test normals (share of streams with any alarm):"]
    for method, table in report["false_alarms"].items():
        cells = [
            f"{rate}: ind {row['inductive']['rate'] * 100:.2f}% / "
            f"trans {row['transductive']['rate'] * 100:.2f}%"
            for rate, row in table.items()
        ]
        lines.append(f"  {method}: " + "; ".join(cells))
    for key, title in (("m2ad_delay", "M2AD illuminations"), ("visa_delay", "VisA conditions")):
        lines.append(f"median delay after the change ({title}, rate 0.00, ind / trans):")
        for method, table in report[key].items():
            cells = [
                f"{cond} {row['0.00']['inductive']['median_delay']}/"
                f"{row['0.00']['transductive']['median_delay']}"
                for cond, row in table.items()
            ]
            lines.append(f"  {method}: " + ", ".join(cells))
    return "\n".join(lines) + "\n"


# 명령행 입구. 리포트를 만들어 요약을 출력하고 reports/stage7/drift.json에 쓴다(--no-write면 쓰지 않음).
# 스트림이 없는 칸의 nan처럼 유한하지 않은 값은 표준 JSON에 쓸 수 없으므로 analyze_m2ad.strict_json으로
# None으로 바꾼 뒤 allow_nan=False로 쓴다. Windows 콘솔에서도 출력이 깨지지 않게 stdout을 UTF-8로 바꾼다.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = build_report(paths.OUTPUTS)
    print(format_report(report), end="")
    if not args.no_write:
        from .analyze_m2ad import strict_json

        out = paths.REPORTS / "stage7" / "drift.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(strict_json(report), f, ensure_ascii=False, indent=1, allow_nan=False)
            f.write("\n")


if __name__ == "__main__":
    main()
