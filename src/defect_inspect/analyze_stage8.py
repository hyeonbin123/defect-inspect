"""8단계 표와 판정: 임계값만 다시 잡기(8-A), 폐루프 원인 분해(8-B), 서빙 모델 구성의 M2AD(8-C).

`python -m defect_inspect.analyze_stage8 calibrate|loop|dms-train|dms-test` → `reports/stage8/<이름>.json`.
비율은 검사기 6개를 합쳐 내고, 구간은 3-B와 같은 시편 단위 군집 부트스트랩(2,000번, seed 0)이다. 비교하는
수치는 모두 같은 추출을 쓴다. 규칙은 docs/experiments.md 8단계.
"""

# [파일 역할] 8단계 분석 모듈이다. 세 실행기(run_m2ad_threshold, run_m2ad_loop, run_m2ad_dinomaly)의 저장물을
# 읽어 표를 만들고 H23·H24·H25를 등록한 규칙대로 판정한다.
# [설계 이유] M2AD의 독립 표본은 시편이므로 3-B의 ClusterBootstrap을 그대로 쓴다. 같은 seed·같은 검사기 순서로
# 만들면 3-B 표와 같은 재표집이 나와, 3-B 사람 재보정 행과의 차이도 같은 추출로 낼 수 있다. 임계값을 정하는
# 규칙(conformal, 목표 5%)은 calibrate.conformal_threshold 하나만 쓴다.
# [흐름] calibrate: 3-B 실행 + 8-A 실행 → 팔(T8, T19, T30)마다 조명별 임계값 → 합산 비율 → H23. 팔마다
# 검사기·조명 평균 이미지 AUROC도 같은 추출로 낸다(판정 없음).
# loop: 3-B 실행 + 7단계 폐루프 + 8-B 실행 → 재현 관문 → 팔별 표 → H25와 차이.
# dms-train: 8-C 학습 기록과 학습 시편 교차 적합 점수 → 팔별 등록·못 본 조명 오검출률(판정 없음).
# dms-test: 8-C test 점수 → 7단계와 같은 표 → H24, 임계값만 다시 잡기, 폐루프(판정 없음).
# [연결] analyze_m2ad(ClusterBootstrap, load, strict_json), analyze_m2ad_enrol(비율 도우미, H18과 같은
# 판정선), m2ad(조명 묶음, 앞 n개 시편), calibrate(conformal 임계값).

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import paths
from .analyze_m2ad import ClusterBootstrap, condition_cell, recal_cell, strict_json
from .analyze_m2ad import load as load_3b
from .analyze_m2ad_enrol import _ci, _public, _rates
from .calibrate import conformal_threshold
from .m2ad import ILLUMINATIONS, REFERENCE, illumination_groups, recal_specimens
from .splits import path_key

# 판정 상수(규칙). H23: T30 − 뱅크+임계값 n = 30 검출률 차이의 기준선 −20%p. H24: E2 − E0 못 본 조명
# 오검출률의 기준선 −50%p와 I01 검출률 가드 −10%p(H18과 같다). H25·H12형 상한 10%.
N_BOOT = 2000
SEED = 0
ALPHA = 0.05
H23_MARGIN = -0.20
H24_MARGIN = -0.50
GUARD = -0.10
LIMIT = 0.10
JUDGED_METHOD = "p0"
REL_TOL = 1e-6
_TOL = 1e-9  # 합산 비율은 정수의 비라 경계와 같은 구간 끝을 부동소수에서도 경계에 둔다
THRESHOLD_ARMS = (8, 19, 30)
NEW_LIGHTS = tuple(light for light in ILLUMINATIONS if light != REFERENCE)
DMS_ARMS = ("e0", "e2-a", "e2-b")
DMS_RECAL = (19, 30)
DMS_LOOP_N = 20
DMS_LOOP_K = (0, 2)
HUMAN_M = 4
TRIM = 0.2


# 리포트를 엄격한 JSON(NaN·무한대는 null)으로 쓴다. 폴더가 없으면 만든다.
def _write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(strict_json(report), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


# run.json·리포트 같은 JSON 파일 하나를 읽는다.
def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# 구간 상한·하한을 기준선과 비교하는 판정(지지·기각·판정 불가). "작으면 좋다" 쪽 판정만 쓴다.
def _verdict_below(lo: float, hi: float, margin: float) -> str:
    if hi < margin - _TOL:
        return "지지"
    if lo > margin + _TOL:
        return "기각"
    return "판정 불가"


# 상한이 limit 이하면 지지, 하한이 limit보다 크면 기각(H12·H25형).
def _verdict_limit(lo: float, hi: float, limit: float) -> str:
    if hi <= limit + _TOL:
        return "지지"
    if lo > limit + _TOL:
        return "기각"
    return "판정 불가"


# 두 비율 묶음의 차이(관측값과 같은 추출의 구간).
def _diff(a: dict, b: dict, key: str) -> dict:
    return {"diff": a[key] - b[key], "ci": _ci(a[f"{key}_samples"] - b[f"{key}_samples"])}


# 임계값 기록을 모은다: 보장된 칸 수, 순위·n의 범위.
def _threshold_summary(thresholds: list) -> dict:
    return {
        "cells": len(thresholds),
        "guaranteed": int(sum(t.guaranteed for t in thresholds)),
        "n": sorted({int(t.n) for t in thresholds}),
        "rank": sorted({int(t.rank) for t in thresholds}),
    }


# 조명 칸 여러 개의 이미지 AUROC를 한 값으로 모은다. 칸(조명)마다 ClusterBootstrap.auroc로 검사기 평균 AUROC를
# 내고 조명끼리 단순 평균한다. 재표집 표본도 같은 추출 안에서 조명 평균을 낸다(7단계·8-C의 못 본 조명 AUROC
# 평균과 같은 정의). 입력: 칸 목록(검사기마다 점수, 라벨, 임계값). 출력: {"auroc": 관측값, "auroc_samples":
# [n_boot]}. AUROC는 임계값을 쓰지 않으므로 점수와 라벨이 같은 칸이면 임계값 규칙이 달라도 값이 같다. 어떤
# 조명 칸에 AUROC가 없으면(정상이나 결함 이미지가 하나도 없음) 그 평균은 nan이고, 구간은 nan 표본을 빼고 낸다.
def _mean_auroc(boot: ClusterBootstrap, cells: list) -> dict:
    values, samples = [], []
    for cell in cells:
        value, draws = boot.auroc(cell)
        values.append(value)
        samples.append(draws)
    return {"auroc": float(np.mean(values)), "auroc_samples": np.mean(np.stack(samples), axis=0)}


# ---------------------------------------------------------------- 8-A


# 8-A 실행 읽기. 동일성 관문이 모든 검사기에서 통과했고 점검용(--no-gate-stop) 실행이 아닌지 확인한다.
def load_threshold_run(run_dir: Path) -> dict:
    meta = _read_json(Path(run_dir) / "run.json")
    if meta.get("no_gate_stop"):
        raise ValueError(f"{run_dir} is a code-path check (--no-gate-stop), not a measurement")
    arrays = {}
    for info in meta["inspectors"]:
        if not info["gate"]["ok"]:
            raise ValueError(f"{run_dir}: the identity gate failed for {info['category']} {info['view']}")
        name = f"{info['category']}_{info['view']}"
        with np.load(Path(run_dir) / f"{name}.npz") as z:
            arrays[name] = {k: z[k] for k in z.files}
    return {"meta": meta, "inspectors": arrays}


# 팔 Tn의 조명 L 칸: 검사기마다 (3-B 전체 E0 뱅크의 L test 점수, 라벨, L 학습 시편 앞 n개 점수로 정한 임계값).
def threshold_cell(ref: dict, run: dict, light: str, n: int) -> tuple[list, list]:
    cell, thresholds = [], []
    for name, arr in ref["inspectors"].items():
        k = arr["conditions"].tolist().index(f"R:{light}")
        new = run["inspectors"][name]
        names = new["specimens"].tolist()
        idx = [names.index(s) for s in recal_specimens(names, n)]
        li = new["lights"].tolist().index(light)
        thr = conformal_threshold(new["cal_scores"][li, idx], ALPHA)
        cell.append((arr["scores"][k], arr["labels"][k], float(thr.value)))
        thresholds.append(thr)
    return cell, thresholds


# 방법 하나의 8-A 표. 팔 T8·T19·T30(임계값만), 3-B의 뱅크+임계값 재보정(bank8, bank30), 재보정 없음(none)을
# 같은 부트스트랩 추출로 내고, H23(T30 − bank30 검출률)과 T30 오검출률 상한이 10% 이하인지를 적는다. 동일성
# 관문 기록도 같이 싣는다.
# 팔마다 붙는 검사기·조명 평균 이미지 AUROC(auroc, auroc_ci)와 H23 옆의 같은 추출 AUROC 차이
# (d_auroc_t30_bank30)는 판정 없는 보조 수치다. H23의 검출률 차이에는 분리도(AUROC) 차이와 운영점(임계값 순위:
# T30은 30장 중 30번째, bank30은 60장 중 58번째) 차이가 함께 들어 있어, 팔의 분리도를 따로 보려고 측정 뒤에
# 더했다. 3-B 실행에 저장된 점수만 쓰므로 봉인 이미지를 다시 읽지 않고 장부 줄도 생기지 않는다. 임계값만 팔은
# test 점수가 none(3-B E0 점수)과 같아 AUROC도 none과 같다. 새 키는 각 묶음의 끝에 붙여 기존 키의 값과 순서는
# 그대로다(커밋한 리포트를 다시 만들어도 새 키만 는다).
def calibrate_method(ref: dict, run: dict, boot: ClusterBootstrap) -> dict:
    arms, per_light, aurocs = {}, {}, {}
    for n in THRESHOLD_ARMS:
        cells, thresholds = [], []
        for light in NEW_LIGHTS:
            cell, thr = threshold_cell(ref, run, light, n)
            cells.append(cell)
            thresholds += thr
            per_light[f"T{n}:{light}"] = _public(_rates(boot, [cell]))
        arms[f"T{n}"] = _rates(boot, cells) | {"thresholds": _threshold_summary(thresholds)}
        aurocs[f"T{n}"] = _mean_auroc(boot, cells)
    for n in (8, 30):
        cells = [recal_cell(ref, f"{light}:{n}") for light in NEW_LIGHTS]
        arms[f"bank{n}"] = _rates(boot, cells)
        aurocs[f"bank{n}"] = _mean_auroc(boot, cells)
        for light, cell in zip(NEW_LIGHTS, cells, strict=True):
            per_light[f"bank{n}:{light}"] = _public(_rates(boot, [cell]))
    none_cells = [condition_cell(ref, f"R:{light}") for light in NEW_LIGHTS]
    arms["none"] = _rates(boot, none_cells)
    aurocs["none"] = _mean_auroc(boot, none_cells)
    d_tpr = _diff(arms["T30"], arms["bank30"], "tpr")
    lo, hi = d_tpr["ci"]
    t30_fpr_ci = _ci(arms["T30"]["fpr_samples"])
    public_auroc = {k: {"auroc": a["auroc"], "auroc_ci": _ci(a["auroc_samples"])} for k, a in aurocs.items()}
    return {
        "arms": {k: _public(v) | public_auroc[k] for k, v in arms.items()},
        "per_light": per_light,
        "h23": {
            "d_tpr": d_tpr["diff"],
            "d_tpr_ci": d_tpr["ci"],
            "margin": H23_MARGIN,
            "verdict": _verdict_below(lo, hi, H23_MARGIN),
            "t30_fpr": arms["T30"]["fpr"],
            "t30_fpr_ci": t30_fpr_ci,
            "t30_fpr_upper_at_most_10pct": bool(t30_fpr_ci[1] <= LIMIT + _TOL),
            "d_fpr_t30_bank30": _diff(arms["T30"], arms["bank30"], "fpr"),
            "d_auroc_t30_bank30": _diff(aurocs["T30"], aurocs["bank30"], "auroc"),
        },
        "gate": [
            info["gate"] | {"inspector": f"{info['category']}_{info['view']}"}
            for info in run["meta"]["inspectors"]
        ],
    }


# 8-A 리포트: 방법마다 3-B 실행(outputs/m2ad-<방법>)과 8-A 실행(outputs/m2ad-threshold-<방법>)을 읽어
# calibrate_method로 표를 만든다. 판정은 P0만 남기고 D-S는 수치만 둔다(규칙).
def calibrate_report(methods: list[str], n_boot: int | None = None, seed: int = SEED) -> dict:
    report: dict = {
        "n_boot": int(n_boot or N_BOOT),
        "seed": seed,
        "judged_method": JUDGED_METHOD,
        "methods": {},
    }
    for method in methods:
        ref = load_3b(paths.OUTPUTS / f"m2ad-{method}")
        run = load_threshold_run(paths.OUTPUTS / f"m2ad-threshold-{method}")
        if list(run["inspectors"]) != list(ref["inspectors"]):
            raise ValueError(f"{method}: the 8-A run and the 3-B run have different inspectors")
        boot = ClusterBootstrap(ref, n_boot, seed)
        entry = calibrate_method(ref, run, boot)
        entry["commit"], entry["device"] = run["meta"].get("commit"), run["meta"].get("device")
        if method != JUDGED_METHOD:
            entry["h23"].pop("verdict")
        report["methods"][method] = entry
    judged = report["methods"].get(JUDGED_METHOD, {}).get("h23", {})
    report["hypotheses"] = {"H23": judged["verdict"]} if "verdict" in judged else {}
    return report


# ---------------------------------------------------------------- 8-B


# 8-B 실행 읽기: run.json(팔 목록, 검사기별 묶음 기록)과 검사기별 npz(점수, 라벨, 두 임계값).
def load_loop_run(run_dir: Path) -> dict:
    meta = _read_json(Path(run_dir) / "run.json")
    arrays = {}
    for info in meta["inspectors"]:
        name = f"{info['category']}_{info['view']}"
        with np.load(Path(run_dir) / f"{name}.npz") as z:
            arrays[name] = {k: z[k] for k in z.files}
    return {
        "meta": meta,
        "inspectors": arrays,
        "info": {f"{i['category']}_{i['view']}": i["arms"] for i in meta["inspectors"]},
    }


# 재현 관문: (1) R(trim, k = 0, n = 20)의 점수·mixed 임계값이 7단계 저장값과, (2) C0(n = 30)의 점수·mixed
# 임계값이 3-B 재보정 n = 30과 같아야 한다(최대 상대 차이 1e-6, 임계값 같음).
def loop_gates(run: dict, ref: dict, stage7_dir: Path) -> dict:
    def rel(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))

    worst7 = worst3b = 0.0
    same7 = same3b = True
    for name, arr in run["inspectors"].items():
        keys = arr["arm_keys"].tolist()
        with np.load(Path(stage7_dir) / f"{name}_loop.npz") as z:
            if z["specimens"].tolist() != arr["specimens"].tolist():
                raise ValueError(f"{name}: test specimens differ from stage 7")
            keys7 = z["loop_keys"].tolist()
            for light in NEW_LIGHTS:
                j, j7 = keys.index(f"{light}:20:0:trim"), keys7.index(f"{light}:trim:0")
                worst7 = max(worst7, rel(arr["scores"][j], z["scores"][j7]))
                same7 &= float(arr["thresholds_mixed"][j]) == float(z["thresholds"][j7])
        b = ref["inspectors"][name]
        rkeys = b["recal_keys"].tolist()
        for light in NEW_LIGHTS:
            j, jb = keys.index(f"{light}:30:0:none"), rkeys.index(f"{light}:30")
            worst3b = max(worst3b, rel(arr["scores"][j], b["recal_scores"][jb]))
            same3b &= float(arr["thresholds_mixed"][j]) == float(b["recal_thresholds"][jb])
    return {
        "stage7_trim": {"max_rel_diff": worst7, "thresholds_equal": bool(same7)},
        "stage3b_recal30": {"max_rel_diff": worst3b, "thresholds_equal": bool(same3b)},
        "tolerance": REL_TOL,
        "ok": bool(worst7 <= REL_TOL and same7 and worst3b <= REL_TOL and same3b),
    }


# 팔 하나(n, k, 등록, 임계값 방식)의 아홉 조명 합산 비율과 묶음 기록의 합.
def loop_arm(run: dict, boot: ClusterBootstrap, n: int, k: int, enrol: str, kind: str) -> dict:
    cells = []
    totals = {"defects_in_batch": 0, "dropped_defects": 0, "dropped_normals": 0, "human_labels": 0}
    guaranteed, ranks = 0, set()
    for light in NEW_LIGHTS:
        key = f"{light}:{n}:{k}:{enrol}"
        cell = []
        for name, arr in run["inspectors"].items():
            j = arr["arm_keys"].tolist().index(key)
            cell.append((arr["scores"][j], arr["labels"][j], float(arr[f"thresholds_{kind}"][j])))
            info = run["info"][name][key]
            for field in totals:
                totals[field] += int(info[field])
            thr = info[f"threshold_{kind}"]
            guaranteed += int(thr["guaranteed"])
            ranks.add((int(thr["rank"]), int(thr["n"])))
        cells.append(cell)
    return _rates(boot, cells) | totals | {"guaranteed_cells": guaranteed, "rank_n": sorted(ranks)}


# 8-B 리포트. 시편 목록이 3-B와 같은지 확인하고 재현 관문을 먼저 본다. 관문이 실패하면 표를 내지 않는다(규칙).
# 팔 이름 대응: C0 = none + mixed, C1 = none + cond, C2 = label + cond, R = trim + mixed(n = 20, k = 0). 같은
# 추출로 차이 (가)~(라)를 내고 H25(C2(30, 2) 오검출률 상한 10%)를 판정한다.
def loop_report(n_boot: int | None = None, seed: int = SEED) -> dict:
    ref = load_3b(paths.OUTPUTS / "m2ad-p0")
    run = load_loop_run(paths.OUTPUTS / "m2ad-loop-p0")
    if list(run["inspectors"]) != list(ref["inspectors"]):
        raise ValueError("the 8-B run and the 3-B run have different inspectors")
    for name, arr in run["inspectors"].items():
        if arr["specimens"].tolist() != ref["inspectors"][name]["specimens"].tolist():
            raise ValueError(f"{name}: test specimens differ from stage 3-B")
    gates = loop_gates(run, ref, paths.OUTPUTS / "m2ad-enrol-test-p0")
    report: dict = {
        "n_boot": int(n_boot or N_BOOT),
        "seed": seed,
        "gates": gates,
        "commit": run["meta"].get("commit"),
    }
    if not gates["ok"]:
        report["note"] = "reproduction gate failed: no tables (rule)"
        return report
    boot = ClusterBootstrap(ref, n_boot, seed)
    arms = {}
    for n, k, enrol in (tuple(a) for a in run["meta"]["arms"]):
        for kind in ("mixed", "cond"):
            arms[f"{n}:{k}:{enrol}:{kind}"] = loop_arm(run, boot, n, k, enrol, kind)
    named = {
        "R": arms["20:0:trim:mixed"],
        "C0(20)": arms["20:0:none:mixed"],
        "C0(30)": arms["30:0:none:mixed"],
        "C1(20,0)": arms["20:0:none:cond"],
        "C1(20,2)": arms["20:2:none:cond"],
        "C1(30,0)": arms["30:0:none:cond"],
        "C1(30,2)": arms["30:2:none:cond"],
        "C2(20,1)": arms["20:1:label:cond"],
        "C2(20,2)": arms["20:2:label:cond"],
        "C2(30,1)": arms["30:1:label:cond"],
        "C2(30,2)": arms["30:2:label:cond"],
        "C2(30,3)": arms["30:3:label:cond"],
    }
    human30 = _rates(boot, [recal_cell(ref, f"{light}:30") for light in NEW_LIGHTS])
    human8 = _rates(boot, [recal_cell(ref, f"{light}:8") for light in NEW_LIGHTS])
    c2 = named["C2(30,2)"]
    lo, hi = _ci(c2["fpr_samples"])
    report["arms"] = {key: _public(value) for key, value in arms.items()}
    report["named"] = {key: _public(value) for key, value in named.items()}
    report["reference"] = {"human_recal_8": _public(human8), "human_recal_30": _public(human30)}
    report["differences"] = {
        "C0(20)-R": {
            "fpr": _diff(named["C0(20)"], named["R"], "fpr"),
            "tpr": _diff(named["C0(20)"], named["R"], "tpr"),
        },
        "C1(20,0)-C0(20)": {
            "fpr": _diff(named["C1(20,0)"], named["C0(20)"], "fpr"),
            "tpr": _diff(named["C1(20,0)"], named["C0(20)"], "tpr"),
        },
        "C1(30,0)-C0(30)": {
            "fpr": _diff(named["C1(30,0)"], named["C0(30)"], "fpr"),
            "tpr": _diff(named["C1(30,0)"], named["C0(30)"], "tpr"),
        },
        "C2(20,2)-C1(20,2)": {
            "fpr": _diff(named["C2(20,2)"], named["C1(20,2)"], "fpr"),
            "tpr": _diff(named["C2(20,2)"], named["C1(20,2)"], "tpr"),
        },
        "C2(30,2)-C1(30,2)": {
            "fpr": _diff(named["C2(30,2)"], named["C1(30,2)"], "fpr"),
            "tpr": _diff(named["C2(30,2)"], named["C1(30,2)"], "tpr"),
        },
        "C2(30,2)-human30": {"fpr": _diff(c2, human30, "fpr"), "tpr": _diff(c2, human30, "tpr")},
    }
    report["h25"] = {
        "fpr": c2["fpr"],
        "fpr_ci": [lo, hi],
        "tpr": c2["tpr"],
        "tpr_ci": _ci(c2["tpr_samples"]),
        "limit": LIMIT,
        "verdict": _verdict_limit(lo, hi, LIMIT),
    }
    report["hypotheses"] = {"H25": report["h25"]["verdict"]}
    return report


# ---------------------------------------------------------------- 8-C


# 8-C 실행기의 출력 폴더(outputs/m2ad-dms).
def _dms_dir() -> Path:
    return paths.OUTPUTS / "m2ad-dms"


# 팔의 학습 시편 교차 적합 점수와 검사기별 임계값(등록 조명 점수 전부로 목표 5%).
def dms_train_scores(out_dir: Path, arm: str) -> dict:
    with np.load(Path(out_dir) / arm / "train_scores.npz") as z:
        data = {k: z[k] for k in z.files}
    enrolled = data["enrolled"].astype(bool)
    data["thresholds"] = [
        conformal_threshold(data["scores"][i][enrolled].reshape(-1), ALPHA)
        for i in range(len(data["inspectors"]))
    ]
    return data


# 8-C 학습 시편 기록(판정 없음, test를 읽기 전에 커밋한다): 팔마다 모델 6개의 학습 기록(마지막 손실, 건너뛴
# 스텝, 최대 VRAM, 시간), 검사기별 임계값, 등록 조명·못 본 조명의 교차 적합 오검출률과 조명별 수. 등록 조명
# 오검출률은 임계값을 정한 바로 그 점수라 목표 5% 근처가 되는 것이 당연하고, 못 본 조명 수치가 정보다.
def dms_train_report(out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir or _dms_dir())
    report: dict = {"arms": {}}
    for arm in DMS_ARMS:
        data = dms_train_scores(out_dir, arm)
        models = {}
        for path in sorted((out_dir / arm).glob("*.json")):
            if path.name == "train_scores.json":
                continue
            info = _read_json(path)
            models[info["model"]] = {
                k: info[k] for k in ("train_images", "final_loss", "skipped_steps", "peak_vram_mb", "train_s")
            }
        enrolled = data["enrolled"].astype(bool)
        per_light = {}
        seen = [0, 0]
        unseen = [0, 0]
        for li, light in enumerate(data["lights"].tolist()):
            fp = sum(int((data["scores"][i, li] > t.value).sum()) for i, t in enumerate(data["thresholds"]))
            n = int(data["scores"][:, li].size)
            per_light[light] = {"false_positives": fp, "n": n, "fpr": fp / n, "enrolled": bool(enrolled[li])}
            target = seen if enrolled[li] else unseen
            target[0] += fp
            target[1] += n
        report["arms"][arm] = {
            "models": models,
            "thresholds": [
                {
                    "inspector": str(name),
                    "value": t.value,
                    "rank": t.rank,
                    "n": t.n,
                    "guaranteed": t.guaranteed,
                }
                for name, t in zip(data["inspectors"], data["thresholds"], strict=True)
            ],
            "enrolled_fpr": seen[0] / seen[1] if seen[1] else math.nan,
            "enrolled_counts": seen,
            "unseen_fpr": unseen[0] / unseen[1] if unseen[1] else math.nan,
            "unseen_counts": unseen,
            "per_light": per_light,
        }
    return report


# test 저장물을 3-B 실행의 검사기 순서로 읽는다. 시편 목록이 3-B와 같아야 같은 부트스트랩을 쓸 수 있다.
def load_dms_test(out_dir: Path, ref: dict) -> dict:
    test = {}
    for arm in DMS_ARMS:
        test[arm] = {}
        for name, b in ref["inspectors"].items():
            with np.load(Path(out_dir) / "test" / f"{arm}_{name}.npz") as z:
                arr = {k: z[k] for k in z.files}
            if arr["specimens"].tolist() != b["specimens"].tolist():
                raise ValueError(f"{arm} {name}: test specimens differ from stage 3-B")
            test[arm][name] = arr
    return test


# 조명 L을 재는 팔: E0은 늘 e0. E2는 L을 등록하지 않은 팔(못 본 조명) 또는 등록한 팔(I01은 두 팔 모두).
def _e2_arms(light: str, enrolled: bool) -> list[str]:
    a, b = illumination_groups()
    if light == REFERENCE:
        return ["e2-a", "e2-b"]
    in_a = light in a
    if enrolled:
        return ["e2-a"] if in_a else ["e2-b"]
    return ["e2-b"] if in_a else ["e2-a"]


# 팔 하나·조명 하나의 Cell(검사기마다 test 점수, 라벨, 임계값). 임계값을 따로 주지 않으면 그 팔의 학습 시편
# 교차 적합 점수로 정한 검사기별 임계값을 쓴다.
def _cell(
    test: dict, train: dict, arm: str, light: str, names: list[str], thresholds: dict | None = None
) -> list:
    li = ILLUMINATIONS.index(light)
    cell = []
    for i, name in enumerate(names):
        arr = test[arm][name]
        thr = thresholds[name] if thresholds is not None else float(train[arm]["thresholds"][i].value)
        cell.append((arr["scores"][li], arr["labels"][li], thr))
    return cell


# 후보(e0 또는 두 팔을 합친 e2)의 test 수치. 7단계 candidate_rates와 같은 정의다: 못 본 조명(I01 제외 아홉
# 조명, E2는 그 조명을 등록하지 않은 팔), 등록 조명(E2는 두 팔의 등록 조명, I01은 두 팔 모두), 기준 조명 I01.
def _candidate(test, train, boot, names, candidate: str) -> dict:
    unseen, seen, ref_cells, per_light, aurocs, auroc_samples = [], [], [], {}, [], []
    for light in ILLUMINATIONS:
        for enrolled in (False, True):
            if candidate == "e0":
                arms = ["e0"] if (light == REFERENCE) == enrolled else []
            else:
                arms = _e2_arms(light, enrolled) if (light != REFERENCE or enrolled) else []
            for arm in arms:
                cell = _cell(test, train, arm, light, names)
                (seen if enrolled else unseen).append(cell)
                if light == REFERENCE:
                    ref_cells.append(cell)
                rates = _public(_rates(boot, [cell]))
                rates["auroc"], samples = boot.auroc(cell)
                per_light[f"{light}:{'seen' if enrolled else 'unseen'}:{arm}"] = rates
                if not enrolled:
                    aurocs.append(rates["auroc"])
                    auroc_samples.append(samples)
    return {
        "unseen": _rates(boot, unseen),
        "seen": _rates(boot, seen),
        "reference": _rates(boot, ref_cells),
        "unseen_auroc_mean": float(np.mean(aurocs)),
        "unseen_auroc_mean_ci": _ci(np.mean(np.stack(auroc_samples), axis=0)),
        "per_light": per_light,
    }


# 임계값만 다시 잡기(판정 없음): 못 본 조명 L마다 L 학습 시편 앞 n개의 교차 적합 점수로 임계값.
def _dms_recal(test, train, boot, names, candidate: str, n: int) -> dict:
    cells = []
    for light in NEW_LIGHTS:
        arm = "e0" if candidate == "e0" else _e2_arms(light, False)[0]
        tr = train[arm]
        li = tr["lights"].tolist().index(light)
        thresholds = {}
        for i, name in enumerate(names):
            specimens = tr["specimens"][i].tolist()
            idx = [specimens.index(s) for s in recal_specimens(specimens, n)]
            thresholds[name] = float(conformal_threshold(tr["scores"][i, li, idx], ALPHA).value)
        cells.append(_cell(test, train, arm, light, names, thresholds))
    return _rates(boot, cells)


# 폐루프(E0, 임계값만, 판정 없음): 무라벨 20장 = 학습 시편 앞 20 − k개의 교차 적합 점수 + 결함이 보이는 test
# 불량 시편 앞 k개의 전체 모델 점수. none / trim(상위 ⌈0.2n⌉장 버림) / label(상위 4장 중 결함만 뺌).
def _dms_loop(test, train, boot, names, k: int, enrol: str) -> dict:
    cells = []
    dropped_defects = 0
    for light in NEW_LIGHTS:
        li = ILLUMINATIONS.index(light)
        tr = train["e0"]
        cell = []
        for i, name in enumerate(names):
            arr = test["e0"][name]
            specimens = tr["specimens"][i].tolist()
            normal_idx = [specimens.index(s) for s in recal_specimens(specimens, DMS_LOOP_N - k)]
            labels = arr["labels"][li].copy()
            test_names = arr["specimens"].tolist()
            visible = sorted(
                (s for s, lab in zip(test_names, labels, strict=True) if lab == 1), key=path_key
            )[:k]
            if len(visible) < k:
                raise ValueError(f"{name} I{light}: fewer than {k} visible defects to mix in")
            batch = np.concatenate(
                [tr["scores"][i, li, normal_idx], [arr["scores"][li][test_names.index(s)] for s in visible]]
            ).astype(np.float64)
            is_defect = np.array([False] * len(normal_idx) + [True] * len(visible))
            order = np.argsort(-batch, kind="stable")
            if enrol == "none":
                dropped = np.zeros(0, dtype=np.int64)
            elif enrol == "trim":
                dropped = order[: math.ceil(TRIM * len(batch))]
            else:
                top = order[:HUMAN_M]
                dropped = top[is_defect[top]]
            dropped_defects += int(is_defect[dropped].sum())
            kept = np.setdiff1d(np.arange(len(batch)), dropped)
            thr = conformal_threshold(batch[kept], ALPHA)
            for s in visible:
                labels[test_names.index(s)] = -1
            cell.append((arr["scores"][li], labels, float(thr.value)))
        cells.append(cell)
    return _rates(boot, cells) | {"dropped_defects": dropped_defects}


# 8-C test 리포트: 후보 표, H24(E2 − E0 못 본 조명 오검출률, I01 검출률 가드), 판정 없는 임계값만 다시
# 잡기(T19, T30)와 E0 폐루프, 그리고 7단계 P0·D-S 수치를 나란히. 부트스트랩은 3-B 실행과 같은 추출이다.
def dms_test_report(out_dir: Path | None = None, n_boot: int | None = None, seed: int = SEED) -> dict:
    out_dir = Path(out_dir or _dms_dir())
    ref = load_3b(paths.OUTPUTS / "m2ad-p0")
    names = list(ref["inspectors"])
    train = {arm: dms_train_scores(out_dir, arm) for arm in DMS_ARMS}
    for arm in DMS_ARMS:
        if train[arm]["inspectors"].tolist() != names:
            raise ValueError(f"{arm}: inspector order differs from stage 3-B")
    test = load_dms_test(out_dir, ref)
    boot = ClusterBootstrap(ref, n_boot, seed)
    cands = {c: _candidate(test, train, boot, names, c) for c in ("e0", "e2")}
    e0, e2 = cands["e0"], cands["e2"]
    d = _diff(e2["unseen"], e0["unseen"], "fpr")
    lo, hi = d["ci"]
    d_ref = _diff(e2["reference"], e0["reference"], "tpr")
    verdict = _verdict_below(lo, hi, H24_MARGIN)
    guard_ok = bool(d_ref["diff"] >= GUARD - _TOL)
    recal = {
        f"{c}:T{n}": _public(_dms_recal(test, train, boot, names, c, n))
        for c in ("e0", "e2")
        for n in DMS_RECAL
    }
    loop = {
        f"{enrol}:{k}": _public(_dms_loop(test, train, boot, names, k, enrol))
        for k in DMS_LOOP_K
        for enrol in ("none", "trim", "label")
    }
    stage7 = (
        _read_json(paths.REPORTS / "stage7" / "test.json")
        if (paths.REPORTS / "stage7" / "test.json").exists()
        else None
    )
    side = None
    if stage7 is not None:
        side = {
            method: {
                c: {"unseen_fpr": v["unseen"]["fpr"], "reference_tpr": v["reference"]["tpr"]}
                for c, v in entry["candidates"].items()
            }
            for method, entry in stage7["methods"].items()
        }
    return {
        "n_boot": int(n_boot or N_BOOT),
        "seed": seed,
        "candidates": {
            c: {
                "unseen": _public(v["unseen"]),
                "seen": _public(v["seen"]),
                "reference": _public(v["reference"]),
                "unseen_auroc_mean": v["unseen_auroc_mean"],
                "unseen_auroc_mean_ci": v["unseen_auroc_mean_ci"],
                "per_light": v["per_light"],
            }
            for c, v in cands.items()
        },
        "h24": {
            "d_unseen_fpr": d["diff"],
            "d_unseen_fpr_ci": d["ci"],
            "margin": H24_MARGIN,
            "verdict": verdict,
            "d_reference_tpr": d_ref["diff"],
            "d_reference_tpr_ci": d_ref["ci"],
            "guard": GUARD,
            "guard_ok": guard_ok,
            "improved": bool(verdict == "지지" and guard_ok),
            "d_unseen_tpr": _diff(e2["unseen"], e0["unseen"], "tpr"),
        },
        "threshold_only": recal,
        "closed_loop_e0": loop,
        "stage7_side_by_side": side,
        "hypotheses": {"H24": verdict},
    }


# 명령줄 진입점. 하위 명령마다 저장물을 읽어 reports/stage8/<이름>.json을 쓰고 판정을 출력한다.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="8단계 표와 판정(8-A calibrate, 8-B loop, 8-C dms-train·dms-test)."
    )
    parser.add_argument("command", choices=("calibrate", "loop", "dms-train", "dms-test"))
    parser.add_argument("--methods", nargs="+", default=["p0", "d-s"], help="calibrate에서 볼 방법")
    parser.add_argument("--out", type=Path, default=None, help="기본값: reports/stage8/<이름>.json")
    args = parser.parse_args(argv)
    names = {"calibrate": "calibrate", "loop": "loop", "dms-train": "dms-train", "dms-test": "dms-test"}
    if args.command == "calibrate":
        report = calibrate_report(args.methods)
    elif args.command == "loop":
        report = loop_report()
    elif args.command == "dms-train":
        report = dms_train_report()
    else:
        report = dms_test_report()
    out = args.out or paths.REPORTS / "stage8" / f"{names[args.command]}.json"
    _write(out, report)
    print(json.dumps(strict_json(report.get("hypotheses", {})), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
