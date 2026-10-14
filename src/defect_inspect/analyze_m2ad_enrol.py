"""7단계 표와 판정을 만든다: 검증 선택, 봉인 M2AD test의 H18·H19, 폐루프 팔, 특징 중심화(E1)의 VisA dev 표.

규칙은 docs/experiments.md 7단계에 있다. 비율은 검사기를 모두 합쳐 낸다. 구간은 부트스트랩에서 나오는데,
추출 단위는 M2AD에서는 시편, VisA dev에서는 범주·클래스 안의 이미지다. 비교하는 모든 것에 같은 추출을
쓰므로 차이가 짝지어진다.
"""

# [파일 역할] 7단계 "조명 변화 대응" 분석 모듈이다. 하위 명령 세 개가 7단계의 순서와 같다.
# - val: M2AD 학습 시편만으로 한 검증. 후보(E0 그대로, E1 이미지별 중심화, E2 조명 묶음 등록, E3 둘 다)마다 못
#   본 조명의 합산 오검출률을 내고 방법별로 하나를 고른다 → reports/stage7/val.json.
# - test: 봉인 M2AD test에서 E0과 고른 후보를 비교해 H18·H19를 판정하고, 경보 뒤 자동 재등록(폐루프) 표를
#   만든다 → test.json.
# - visa-dev: E1(중심화)이 VisA dev의 밝기 교란에 듣는지 보는 판정 없는 점검 → visa-dev.json.
# [설계 이유] 후보는 학습 시편 검증으로만 고르고(결함 라벨 0장), 고른 결과를 커밋·push한 뒤에 봉인 test를
# 읽는다(7단계 규칙). 비교하는 모든 수치에 같은 부트스트랩 추출을 써서 차이가 짝지어진다(M2AD는 시편 단위,
# VisA dev는 범주 안 정상·결함 층화).
# [결과] 검증 선택: P0는 E2(53.4%, E3와 동점이라 단순한 쪽), D-S는 E3(11.4%). test: H18 기각(P0 96.9% → 55.6%,
# −41.3%p [−45.2, −37.1]로 −50%p 기준에 못 미침), H19 지지(등록한 조명 3.3%). 폐루프는 오검출률을 약
# 9.4~9.9%로 되돌렸다(README 7단계).
# [연결] 입력은 run_m2ad_enrol(val/test)과 run_perturb(VisA dev)의 출력이다. 시편 부트스트랩과 strict_json은
# analyze_m2ad, 행 단위 AUROC는 analyze_perturb, 후보·폐루프 팔 이름은 run_m2ad_enrol, 조명 목록은 m2ad에서
# 가져온다.

# Fraction은 후보 선택에서 오검출 비율을 정확한 분수로 비교하는 데 쓴다(동점 판단이 부동소수 반올림에 흔들리지
# 않게).
import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import paths
from .analyze_m2ad import ClusterBootstrap, strict_json
from .analyze_perturb import _auroc_rows
from .analyze_perturb import load as load_perturb
from .calibrate import conformal_threshold
from .m2ad import ILLUMINATIONS, REFERENCE
from .run_m2ad_enrol import CANDIDATES, LOOP_ARMS
from .stats import percentile_ci, stratified_indices

# 판정 상수(docs/experiments.md 7단계).
# - JUDGED_METHOD: H18·H19는 P0만 판정하고 D-S는 같은 수치를 판정 없이 싣는다(D-S는 재보정 없는 오검출률이
#   57.6%라 −50%p 기준에 닿기 어렵다).
# - H18_MARGIN: 못 본 조명 오검출률 차이(선택 − E0)의 구간 상한이 −50%p보다 작으면 지지, 하한이 −50%p보다 크면
#   기각.
# - GUARD: 공동 보고 가드. 기준 조명 I01의 검출률이 10%p 넘게 떨어지면 H18이 지지돼도 "개선"이라고 적지
#   않는다.
# - H19_LIMIT: 등록한 조명의 합산 오검출률 상한 6%.
# - VISA_CONDITIONS: E1 VisA dev 점검의 세 조건(교란 없음, 밝기 0.7, 감마 1.5). _TOL: 정수 비율이 경계와 같을
#   때의 반올림 허용치.
N_BOOT = 2000
SEED = 0
ALPHA = 0.05
JUDGED_METHOD = "p0"
H18_MARGIN = -0.50
GUARD = -0.10
H19_LIMIT = 0.06
VISA_CONDITIONS = ("clean", "brightness-3", "gamma-3")
_TOL = 1e-9  # 합산 비율은 정수의 비다. 정확한 산술에서 경계와 같은 구간 끝을 부동소수에서도 경계에 둔다


# 작은 도우미들: 95% 백분위 구간, 분모 0이면 nan인 비율, 유한하지 않은 값을 None으로 바꿔 쓰는 JSON 저장, 실행
# 폴더의 후보·검사기별 npz 읽기(`<검사기>_<후보>.npz`), run.json 읽기.
def _ci(samples: np.ndarray) -> list[float]:
    lo, hi = percentile_ci(samples)
    return [float(lo), float(hi)]


def _ratio(num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(strict_json(report), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def _load_npzs(run_dir: Path, meta: dict, suffixes: list[str]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for suffix in suffixes:
        out[suffix] = {}
        for name in meta["inspectors"]:
            with np.load(Path(run_dir) / f"{name}_{suffix}.npz") as z:
                out[suffix][name] = {k: z[k] for k in z.files}
    return out


def _read_meta(run_dir: Path) -> dict:
    with open(Path(run_dir) / "run.json", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 검증 (학습 시편만)


# command가 val이 아니면 다른 종류의 실행을 잘못 넘긴 것이라 거절한다.
def load_val(run_dir: Path) -> dict:
    """`run_m2ad_enrol val` 실행을 읽는다: {"meta", "cands": {후보: {검사기: 배열}}}."""
    meta = _read_meta(run_dir)
    if meta.get("command") != "val":
        raise ValueError(f"{run_dir} is not a validation run")
    return {"meta": meta, "cands": _load_npzs(run_dir, meta, meta["candidates"])}


# 검증용 부트스트랩 가중치. 학습 시편은 모두 정상이라 범주마다 시편 30개를 한 묶음으로 복원 추출하고, "재표집
# b에서 시편 j가 몇 번 뽑혔나" 행렬을 만든다. 같은 범주의 시점들, 그리고 모든 후보가 같은 시편 목록이어야 같은
# 가중치를 공유할 수 있어 먼저 확인한다. 이 구간은 선택에 쓰지 않는 참고다(규칙).
def _val_weights(run: dict, n_boot: int, seed: int) -> dict[str, np.ndarray]:
    """범주마다 시편 추출([n_boot, n_specimens] 횟수)을 만든다. 모든 후보가 이것을 함께 쓴다."""
    first = next(iter(run["cands"].values()))
    names: dict[str, list[str]] = {}
    for inspector, arr in first.items():
        category = inspector.split("_")[0]
        mine = arr["specimens"].tolist()
        if names.setdefault(category, mine) != mine:
            raise ValueError(f"{inspector}: train specimens differ from the other views of {category}")
    for cand in run["cands"].values():
        for inspector, arr in cand.items():
            if arr["specimens"].tolist() != names[inspector.split("_")[0]]:
                raise ValueError(f"{inspector}: specimens differ between candidates")
    categories = list(names)
    draws = stratified_indices([len(names[c]) for c in categories], n_boot=n_boot, seed=seed)
    weights = {}
    rows = np.arange(n_boot)[:, None]
    for category, draw in zip(categories, draws, strict=True):
        w = np.zeros((n_boot, len(names[category])))
        np.add.at(w, (rows, draw), 1.0)
        weights[category] = w
    return weights


# 결과에는 합산 오검출(수, 비율, 구간)과 조명별 못 본 조명 오검출 수가 들어간다. 임계값은 바깥 겹마다
# 다르므로(평가 시편이 뱅크에도 임계값에도 들어가지 않는 중첩 교차 적합) thresholds[arm, folds]로 시편마다
# 자기 겹의 임계값을 고른다. enrolled[arm, 조명]이 참이면 그 팔에서 등록한 조명(seen), 거짓이면 못 본
# 조명(unseen)이다.
def val_candidate(cand: dict[str, dict], weights: dict[str, np.ndarray]) -> dict:
    """후보 하나의 검증 결과: 못 본 조명과 등록한 조명의 합산 오검출 수를 부트스트랩 표본과 함께 낸다."""
    n_boot = next(iter(weights.values())).shape[0]
    point = {"unseen": np.zeros(2), "seen": np.zeros(2)}
    samples = {"unseen": np.zeros((n_boot, 2)), "seen": np.zeros((n_boot, 2))}
    per_light = {light: np.zeros(2) for light in ILLUMINATIONS}
    for inspector, arr in cand.items():
        if arr["lights"].tolist() != list(ILLUMINATIONS):
            raise ValueError(f"{inspector}: unexpected illumination order")
        w = weights[inspector.split("_")[0]]
        thresholds = arr["thresholds"][:, arr["folds"]]  # [팔 수, 시편 수]
        for a in range(len(arr["arms"])):
            for li, light in enumerate(ILLUMINATIONS):
                flagged = (arr["scores"][a, li].astype(np.float64) > thresholds[a]).astype(np.float64)
                cols = np.stack([flagged, np.ones_like(flagged)], axis=1)
                kind = "seen" if arr["enrolled"][a, li] else "unseen"
                point[kind] += cols.sum(axis=0)
                # [n_boot, 시편] @ [시편, 2] = 재표집마다의 (오검출 수, 정상 수). 못 본 조명이면 조명별
                # 합계에도 더한다.
                samples[kind] += w @ cols
                if kind == "unseen":
                    per_light[light] += cols.sum(axis=0)
    out = {}
    for kind in ("unseen", "seen"):
        out[kind] = {
            "false_positives": int(point[kind][0]),
            "n_normal": int(point[kind][1]),
            "fpr": float(_ratio(point[kind][0], point[kind][1])),
            "fpr_ci": _ci(_ratio(samples[kind][:, 0], samples[kind][:, 1])),
        }
    out["unseen_by_light"] = {
        light: {"false_positives": int(c[0]), "n_normal": int(c[1])} for light, c in per_light.items() if c[1]
    }
    return out


# P0에서 E2와 E3가 865장으로 같아 E2가 뽑혔다.
def choose(rows: dict[str, dict]) -> str:
    """선택 규칙: 못 본 조명 합산 FPR(Fraction으로 정확한 분수 비교)이 가장 낮은 후보를 고른다.

    같으면 단순한 쪽(CANDIDATES 순서 E0 < E1 < E2 < E3)을 고른다.
    """
    ranked = sorted(
        rows,
        key=lambda c: (
            Fraction(rows[c]["unseen"]["false_positives"], rows[c]["unseen"]["n_normal"]),
            CANDIDATES.index(c),
        ),
    )
    return ranked[0]


# 방법마다 후보별 검증 결과와 선택을 모은다. 후보들이 같은 수의 못 본 조명 이미지(1,620장)로 채점되지 않았으면
# 비교가 공정하지 않으므로 오류를 낸다.
def build_val_report(runs: dict[str, dict], n_boot: int | None = None, seed: int = SEED) -> dict:
    n_boot = int(n_boot or N_BOOT)
    report: dict = {"n_boot": n_boot, "seed": seed, "methods": {}}
    for method, run in runs.items():
        weights = _val_weights(run, n_boot, seed)
        rows = {c: val_candidate(run["cands"][c], weights) for c in run["cands"]}
        n = {rows[c]["unseen"]["n_normal"] for c in rows}
        if len(n) != 1:
            raise ValueError(f"{method}: candidates were scored on different unseen images: {n}")
        report["methods"][method] = {
            "commit": run["meta"].get("commit"),
            "device": run["meta"].get("device"),
            "groups": run["meta"].get("groups"),
            "inspectors": run["meta"]["inspectors"],
            "candidates": rows,
            "pick": choose(rows),
        }
    return report


# 콘솔용 검증 표(후보별 못 본 조명·등록 조명 오검출률). docs/experiments.md 7단계 검증 표의 원본이다.
def format_val(report: dict) -> str:
    lines = []
    for method, entry in report["methods"].items():
        lines += [
            f"## {method} validation (commit {entry['commit']}): pick {entry['pick']}",
            "| candidate | unseen FPR % | seen FPR % |",
            "|---|---|---|",
        ]
        for cand, row in entry["candidates"].items():
            u, s = row["unseen"], row["seen"]
            lines.append(
                f"| {cand} | {u['fpr'] * 100:.1f} [{u['fpr_ci'][0] * 100:.1f}, {u['fpr_ci'][1] * 100:.1f}] "
                f"({u['false_positives']}/{u['n_normal']}) | {s['fpr'] * 100:.1f} "
                f"({s['false_positives']}/{s['n_normal']}) |"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- 봉인 테스트


# 후보(E0과 선택 후보) 점수와, --loop로 돌렸으면 폐루프 팔 점수(`<검사기>_loop.npz`)까지 읽는다.
def load_test(run_dir: Path) -> dict:
    """`run_m2ad_enrol test` 실행을 읽는다: {"meta", "cands": {후보: {검사기: 배열}}, "loop": ...}."""
    meta = _read_meta(run_dir)
    if meta.get("command") != "test":
        raise ValueError(f"{run_dir} is not a test run")
    cands = _load_npzs(run_dir, meta, meta["candidates"])
    loop = _load_npzs(run_dir, meta, ["loop"])["loop"] if meta.get("loop") else None
    return {"meta": meta, "cands": cands, "loop": loop}


# 3-B와 같은 시편 단위 군집 부트스트랩(analyze_m2ad.ClusterBootstrap)을 만든다. 그 클래스가 필요로 하는
# 것(검사기별 범주와 시편의 불량 여부)만 담은 껍데기 run을 넘긴다. 모든 후보가 같은 test 시편을 채점했는지
# 먼저 확인한다.
def _bootstrap(run: dict, n_boot: int | None, seed: int) -> ClusterBootstrap:
    e0 = run["cands"]["e0"]
    for cand in run["cands"].values():
        for name, arr in cand.items():
            if arr["specimens"].tolist() != e0[name]["specimens"].tolist():
                raise ValueError(f"{name}: test specimens differ between candidates")
    shell = {
        "inspectors": {
            name: {"category": name.split("_")[0], "object_anomaly": arr["object_anomaly"]}
            for name, arr in e0.items()
        }
    }
    return ClusterBootstrap(shell, n_boot, seed)


# E2·E3는 팔이 둘(A: I01 + 묶음 A 등록, B: I01 + 묶음 B 등록)이라, 못 본 조명은 그 조명을 등록하지 않은 팔의
# 점수와 임계값으로 잰다. want_enrolled로 등록한 팔 쪽(seen)과 등록하지 않은 팔 쪽(unseen)을 고른다. E0·E1은
# 팔이 하나이고 I01만 등록했다.
def _arm_cell(cand: dict[str, dict], names: list[str], light: str, want_enrolled: bool) -> list:
    """후보 하나, 조명 하나의 Cell 목록: `light`를 등록한(또는 등록하지 않은) 팔마다 하나, 검사기 순서."""
    li = ILLUMINATIONS.index(light)
    first = cand[names[0]]
    cells = []
    for a in range(len(first["arms"])):
        if bool(first["enrolled"][a, li]) != want_enrolled:
            continue
        cells.append(
            [
                (cand[n]["scores"][a, li], cand[n]["labels"][li], float(cand[n]["thresholds"][a]))
                for n in names
            ]
        )
    return cells


# Cell 목록을 합친 합산 오검출률·검출률(관측값, 재표집 표본, 개수). _public은 리포트에 쓸 때 재표집 표본을
# 빼고 구간만 남긴다.
def _rates(boot: ClusterBootstrap, cells: list) -> dict:
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


def _public(rates: dict) -> dict:
    out = {k: v for k, v in rates.items() if not k.endswith("_samples")}
    for key in ("fpr", "tpr"):
        out[f"{key}_ci"] = _ci(rates[f"{key}_samples"])
    return out


# 못 본 조명(I01을 뺀 아홉 조명 × 그 조명을 등록하지 않은 팔)은 1,080장, 등록한 조명은 E2·E3에서 두 팔의 등록
# 조명 1,320장(I01은 두 팔에서 한 번씩)이고, 기준 조명 I01은 E2·E3에서 두 팔 합산이다.
def candidate_rates(run: dict, cand_name: str, boot: ClusterBootstrap) -> dict:
    """후보 하나의 test 수치: 못 본 조명·등록한 조명·I01의 비율, 조명·팔별 표, 못 본 조명 AUROC의 평균."""
    cand = run["cands"][cand_name]
    names = boot.names
    real = [light for light in ILLUMINATIONS if light != REFERENCE]
    # unseen: I01을 뺀 아홉 조명 × 그 조명을 등록하지 않은 팔. seen: 모든 조명 × 그 조명을 등록한 팔(I01은 두
    # 팔 모두). ref: 기준 조명 I01(등록한 팔 전부).
    unseen_cells = [cell for light in real for cell in _arm_cell(cand, names, light, False)]
    seen_cells = [cell for light in ILLUMINATIONS for cell in _arm_cell(cand, names, light, True)]
    ref_cells = _arm_cell(cand, names, REFERENCE, True)
    per_light = {}
    aurocs, auroc_samples = [], []
    for light in ILLUMINATIONS:
        for enrolled in (False, True):
            for k, cell in enumerate(_arm_cell(cand, names, light, enrolled)):
                key = f"{light}:{'seen' if enrolled else 'unseen'}:{k}"
                rates = _public(_rates(boot, [cell]))
                rates["auroc"], samples = boot.auroc(cell)
                per_light[key] = rates
                if not enrolled:
                    aurocs.append(rates["auroc"])
                    auroc_samples.append(samples)
    return {
        "unseen": _rates(boot, unseen_cells),
        "seen": _rates(boot, seen_cells),
        "reference": _rates(boot, ref_cells),
        "unseen_auroc_mean": float(np.mean(aurocs)) if aurocs else float("nan"),
        "unseen_auroc_mean_ci": _ci(np.mean(np.stack(auroc_samples), axis=0)) if aurocs else None,
        "per_light": per_light,
    }


# H18·H19 판정(P0).
# - H18: 못 본 조명 오검출률 차이(선택 − E0)의 짝지은 구간. 상한 < −50%p면 지지, 하한 > −50%p면 기각, 나머지
#   판정 불가. 선택이 E0이면 기각이다.
# - 가드: 같은 추출의 I01 검출률 차이가 −10%p 이상일 때만 improved(개선)로 적는다. 브리프는 못 본 조명
#   검출률로 가드를 두려 했지만, E0이 못 본 조명에서 거의 모두 불량으로 판정해 검출률이 98.2%라 어떤 후보도
#   "개선"이 될 수 없어 I01 검출률로 바꿨다(규칙의 "브리프에서 바꾼 것" 1).
# - H19: 선택 후보의 등록 조명 합산 오검출률 ≤ 6%면 지지.
# 못 본 조명 검출률 차이는 판정 없이 함께 적는다.
def h18_h19(e0: dict, pick: dict, pick_name: str) -> dict:
    """H18(e0 대비 못 본 조명 FPR, I01 검출률 가드 포함)과 H19(선택 후보의 등록 조명 FPR)를 판정한다."""
    d_fpr = pick["unseen"]["fpr"] - e0["unseen"]["fpr"]
    lo, hi = _ci(pick["unseen"]["fpr_samples"] - e0["unseen"]["fpr_samples"])
    # 선택이 E0이면 비교할 다른 후보가 없으므로 규칙대로 기각이다. 아니면 구간 끝을 −50%p와 비교한다(경계는
    # _TOL로 오차 없이).
    if pick_name == "e0":
        h18 = "기각"
    else:
        h18 = "지지" if hi < H18_MARGIN - _TOL else "기각" if lo > H18_MARGIN + _TOL else "판정 불가"
    d_ref = pick["reference"]["tpr"] - e0["reference"]["tpr"]
    guard_ok = bool(d_ref >= GUARD - _TOL)
    seen = pick["seen"]["fpr"]
    return {
        "pick": pick_name,
        "d_unseen_fpr": d_fpr,
        "d_unseen_fpr_ci": [lo, hi],
        "margin": H18_MARGIN,
        "H18": h18,
        "d_reference_tpr": d_ref,
        "d_reference_tpr_ci": _ci(pick["reference"]["tpr_samples"] - e0["reference"]["tpr_samples"]),
        "guard": GUARD,
        "guard_ok": guard_ok,
        "improved": bool(h18 == "지지" and guard_ok),
        "d_unseen_tpr": pick["unseen"]["tpr"] - e0["unseen"]["tpr"],
        "d_unseen_tpr_ci": _ci(pick["unseen"]["tpr_samples"] - e0["unseen"]["tpr_samples"]),
        "seen_fpr": seen,
        "seen_fpr_ci": _ci(pick["seen"]["fpr_samples"]),
        "h19_limit": H19_LIMIT,
        "H19": "지지" if seen <= H19_LIMIT + _TOL else "기각",
    }


# 팔은 (거르기 trim 또는 거르지 않음 none, 섞은 결함 k = 0/1/2)다. 오검출률·검출률과 함께 실행 기록(run.json의
# loop_runs)에서 묶음에 섞인 결함 수, 거르기가 버린 결함·정상 수를 더한다. 결과: 거르기 k = 2에서 오검출률
# 9.8%, 검출률 63.5%, 섞인 결함의 59%를 버렸다. 거르지 않으면 검출률이 34.3%로 떨어졌다.
def loop_table(run: dict, boot: ClusterBootstrap) -> list[dict]:
    """e0의 폐루프 자동 재등록 표: 아홉 새 조명을 합쳐 (거르기, k)마다 한 행을 만든다."""
    loop = run["loop"]
    names = boot.names
    # run.json에 검사기마다 기록된 폐루프 묶음 정보(키 `<조명>:<거르기>:<k>`별로 섞인 결함 수, 거르기가 버린
    # 결함·정상 수).
    info = {item["category"] + "_" + item["view"]: item["loop"] for item in run["meta"]["loop_runs"]}
    table = []
    for filt, k in LOOP_ARMS:
        cells, mixed, dropped, dropped_normals = [], 0, 0, 0
        for light in ILLUMINATIONS:
            if light == REFERENCE:
                continue
            key = f"{light}:{filt}:{k}"
            # loop npz의 행 순서는 loop_keys 목록을 따르므로, 키로 행 번호를 찾아 점수·라벨·다시 잡은 임계값을
            # 꺼낸다.
            j = loop[names[0]]["loop_keys"].tolist().index(key)
            cells.append(
                [(loop[n]["scores"][j], loop[n]["labels"][j], float(loop[n]["thresholds"][j])) for n in names]
            )
            for n in names:
                mixed += info[n][key]["defects_in_batch"]
                dropped += info[n][key]["dropped_defects"]
                dropped_normals += info[n][key]["dropped_normals"]
        row = {"filter": filt, "k": k, "defects_in_batches": mixed, "dropped_defects": dropped}
        row["dropped_normals"] = dropped_normals
        row |= _public(_rates(boot, cells))
        table.append(row)
    return table


# E0은 3-B와 같은 절차이므로, 같은 장치에서 점수와 임계값이 3-B 실행(outputs/m2ad-<방법>)과 같아야 한다.
# 검사기·조명마다 최대 절대 점수 차이와 임계값이 같은지 적는다(규칙의 stage3b_match). 결과: 두 방법 모두 차이
# 0, 임계값 같음. 3-B 실행이 없으면 None.
def stage3b_match(run: dict, ref_dir: Path) -> dict | None:
    """e0을 같은 방법의 3-B 실행과 비교한다: 가장 큰 점수 차이와 임계값이 같은지."""
    ref_dir = Path(ref_dir)
    if not (ref_dir / "run.json").exists():
        return None
    worst, same_thr = 0.0, True
    for name, arr in run["cands"]["e0"].items():
        with np.load(ref_dir / f"{name}.npz") as z:
            if "specimens" in z.files and z["specimens"].tolist() != arr["specimens"].tolist():
                raise ValueError(f"{ref_dir / name}: other test specimens than this run")
            conditions = z["conditions"].tolist()
            for li, light in enumerate(ILLUMINATIONS):
                # 3-B 실행은 I01을 조건 S로, 다른 조명을 R:<조명>으로 저장했으므로 이름을 맞춰 같은 조명끼리
                # 비교한다. E0의 팔은 하나(행 0)다.
                k = conditions.index("S" if light == REFERENCE else f"R:{light}")
                diff = np.abs(arr["scores"][0, li].astype(np.float64) - z["scores"][k].astype(np.float64))
                worst = max(worst, float(diff.max()))
            same_thr &= float(z["threshold"]) == float(arr["thresholds"][0])
    return {"max_abs_score_diff": worst, "thresholds_equal": bool(same_thr)}


# test 리포트. 실행의 후보 목록 마지막이 선택 후보다(선택이 E0이면 E0 하나뿐). 후보별 수치, H18·H19와 가드,
# 3-B 일치 확인, 폐루프 표를 모으고, P0가 아닌 방법은 판정 문구(H18, H19, improved)를 지워 수치만 남긴다.
def build_test_report(runs: dict[str, dict], n_boot: int | None = None, seed: int = SEED) -> dict:
    report: dict = {
        "n_boot": int(n_boot or N_BOOT),
        "seed": seed,
        "judged_method": JUDGED_METHOD,
        "methods": {},
    }
    for method, run in runs.items():
        boot = _bootstrap(run, n_boot, seed)
        cands = run["meta"]["candidates"]
        # 실행의 후보 목록은 E0 다음 검증에서 고른 후보 순서이므로 마지막이 선택 후보다.
        pick_name = cands[-1]
        rates = {c: candidate_rates(run, c, boot) for c in cands}
        entry = {
            "commit": run["meta"].get("commit"),
            "device": run["meta"].get("device"),
            "candidates": {
                c: {
                    "unseen": _public(r["unseen"]),
                    "seen": _public(r["seen"]),
                    "reference": _public(r["reference"]),
                    "unseen_auroc_mean": r["unseen_auroc_mean"],
                    "unseen_auroc_mean_ci": r["unseen_auroc_mean_ci"],
                    "per_light": r["per_light"],
                }
                for c, r in rates.items()
            },
            "verdicts": h18_h19(rates["e0"], rates[pick_name], pick_name),
            "stage3b_match": stage3b_match(run, paths.OUTPUTS / f"m2ad-{method}"),
        }
        if run["loop"] is not None:
            entry["loop"] = loop_table(run, boot)
        if method != JUDGED_METHOD:
            for key in ("H18", "H19"):
                entry["verdicts"].pop(key)
            entry["verdicts"].pop("improved")
        report["methods"][method] = entry
    judged = report["methods"].get(JUDGED_METHOD, {}).get("verdicts", {})
    report["hypotheses"] = {k: judged[k] for k in ("H18", "H19") if k in judged}
    return report


# 콘솔용 test 요약(후보별 비율, H18 차이와 판정, 가드, H19, 폐루프 팔).
def format_test(report: dict) -> str:
    lines = []
    for method, entry in report["methods"].items():
        v = entry["verdicts"]
        lines.append(f"## {method} test (commit {entry['commit']}), pick {v['pick']}")
        for cand, row in entry["candidates"].items():
            u, s, r = row["unseen"], row["seen"], row["reference"]
            lines.append(
                f"{cand}: unseen FPR {u['fpr'] * 100:.1f}% ({u['false_positives']}/{u['n_normal']}), "
                f"detection {u['tpr'] * 100:.1f}%, AUROC {row['unseen_auroc_mean'] * 100:.1f}; "
                f"seen FPR {s['fpr'] * 100:.1f}%; "
                f"I01 FPR {r['fpr'] * 100:.1f}% detection {r['tpr'] * 100:.1f}%"
            )
        lo, hi = v["d_unseen_fpr_ci"]
        lines.append(
            f"unseen FPR pick - e0: {v['d_unseen_fpr'] * 100:+.1f}%p [{lo * 100:.1f}, {hi * 100:.1f}]"
            + (f" -> H18 {v['H18']}" if "H18" in v else "")
        )
        lines.append(
            f"I01 detection pick - e0: {v['d_reference_tpr'] * 100:+.1f}%p (guard {v['guard'] * 100:.0f}%p: "
            f"{'ok' if v['guard_ok'] else 'fails'})"
            + (f", improved: {v['improved']}" if "improved" in v else "")
        )
        lines.append(f"seen FPR {v['seen_fpr'] * 100:.1f}%" + (f" -> H19 {v['H19']}" if "H19" in v else ""))
        for row in entry.get("loop", []):
            lines.append(
                f"loop {row['filter']} k={row['k']}: FPR {row['fpr'] * 100:.1f}% "
                f"detection {row['tpr'] * 100:.1f}%"
                f" (defects dropped {row['dropped_defects']}/{row['defects_in_batches']})"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- VisA dev: 중심화 (E1)


# 임계값은 그 실행 자신의 교차 적합 점수로 정한다(목표 5%). 필요한 조건이 없으면 오류.
def _visa_arrays(run: dict, conditions: tuple[str, ...]) -> dict:
    """VisA dev 교란 실행에서 범주마다 라벨, `conditions`의 점수 [n_cond, n], conformal 임계값을 꺼낸다."""
    out = {}
    for category, cat in run["cats"].items():
        names = [str(c) for c in cat["conditions"]]
        missing = [c for c in conditions if c not in names]
        if missing:
            raise ValueError(f"{category}: the run has no {missing}")
        out[category] = {
            "images": [str(x) for x in cat["eval_images"]],
            "labels": cat["eval_labels"],
            "scores": np.stack([cat["scores"][names.index(c)] for c in conditions]).astype(np.float64),
            "threshold": conformal_threshold(cat["cal_score"], ALPHA).value,
        }
    return out


# E1 VisA dev 점검 표(판정 없음, 선택에 쓰지 않음). 두 실행이 같은 이미지를 같은 순서로 채점했는지 먼저
# 확인한다. 교란 없는 조건의 E1 − E0 AUROC 차이가 "결함 없는 조건에서의 정확도 손실"이다(스켑틱 지적). 결과:
# P0 밝기 0.7에서 오검출률 38.9% → 14.0%, 교란 없음 AUROC −0.75%p [−1.77, +0.20].
def visa_dev_table(
    e0_run: dict,
    e1_run: dict,
    conditions: tuple[str, ...] = VISA_CONDITIONS,
    n_boot: int | None = None,
    seed: int = SEED,
) -> list[dict]:
    """조건마다 E0과 E1의 평균 이미지 AUROC, 자기 임계값에서의 합산 FPR·검출률, E1 − E0의 짝지은 차이를 낸다.

    구간은 범주마다 정상과 결함을 따로 뽑는 층화 부트스트랩으로 낸다."""
    n_boot = int(n_boot or N_BOOT)
    a, b = _visa_arrays(e0_run, conditions), _visa_arrays(e1_run, conditions)
    if list(a) != list(b):
        raise ValueError("the two runs cover other categories")
    sizes = []
    for category in a:
        if a[category]["images"] != b[category]["images"] or not np.array_equal(
            a[category]["labels"], b[category]["labels"]
        ):
            raise ValueError(f"{category}: the two runs scored other images")
        labels = a[category]["labels"]
        sizes += [int((labels == 0).sum()), int((labels == 1).sum())]
    # 두 실행(E0, E1)과 세 조건 모두에 같은 추출을 써서, E1 − E0 차이와 조건 − 교란 없음 차이가 모두
    # 짝지어진다.
    draws = stratified_indices(sizes, n_boot=n_boot, seed=seed)

    # 조건 k에서 한 실행의 점 추정값과 재표집 표본(AUROC는 범주 평균, 비율은 합산). AUROC 점 추정도 같은
    # _auroc_rows 함수를 1행으로 불러 계산 방식을 맞춘다.
    def stats(arrays: dict, k: int) -> dict:
        auroc_samples, auroc_point = [], []
        fp = np.zeros(n_boot)
        tp = np.zeros(n_boot)
        fp_point = tp_point = n_neg = n_pos = 0
        for c, cat in enumerate(arrays.values()):
            s = cat["scores"][k]
            neg, pos = s[cat["labels"] == 0], s[cat["labels"] == 1]
            ineg, ipos = draws[2 * c], draws[2 * c + 1]
            auroc_samples.append(_auroc_rows(neg[ineg], pos[ipos]))
            auroc_point.append(_auroc_rows(neg[None, :], pos[None, :])[0])
            thr = cat["threshold"]
            fp += (neg[ineg] > thr).sum(axis=1)
            tp += (pos[ipos] > thr).sum(axis=1)
            fp_point += int((neg > thr).sum())
            tp_point += int((pos > thr).sum())
            n_neg += neg.size
            n_pos += pos.size
        return {
            "auroc": float(np.mean(auroc_point)),
            "auroc_samples": np.mean(np.stack(auroc_samples), axis=0),
            "fpr": fp_point / n_neg,
            "fpr_samples": fp / n_neg,
            "tpr": tp_point / n_pos,
            "tpr_samples": tp / n_pos,
        }

    table = []
    base = {}
    for k, condition in enumerate(conditions):
        s0, s1 = stats(a, k), stats(b, k)
        # 첫 조건(clean)의 값을 기준으로 저장해 두고, 뒤 조건에서 교란 없음 대비 AUROC 변화를 낸다.
        if k == 0:
            base = {"e0": s0, "e1": s1}
        row = {"condition": condition}
        for name, s in (("e0", s0), ("e1", s1)):
            for key in ("auroc", "fpr", "tpr"):
                row[f"{name}_{key}"] = s[key]
                row[f"{name}_{key}_ci"] = _ci(s[f"{key}_samples"])
            if k > 0:
                row[f"{name}_d_auroc_vs_clean"] = s["auroc"] - base[name]["auroc"]
                row[f"{name}_d_auroc_vs_clean_ci"] = _ci(s["auroc_samples"] - base[name]["auroc_samples"])
        for key in ("auroc", "fpr", "tpr"):
            row[f"d_{key}"] = s1[key] - s0[key]
            row[f"d_{key}_ci"] = _ci(s1[f"{key}_samples"] - s0[f"{key}_samples"])
        table.append(row)
    return table


# P0는 dev 뱅크를 저장해 두지 않아 7단계에서 outputs/p0-dev7로 다시 만들었고, 원래 outputs/p0-dev와 같은
# 점수인지 확인하는 데 쓴다(결과: 차이 0). 같은 폴더이거나 비교할 실행이 없으면 None.
def rerun_check(source_dir: Path, reference_dir: Path) -> dict | None:
    """다시 만든 실행과 예전 실행의 전체 뱅크 평가 점수(eval_score_full)의 최대 상대 차이를 낸다."""
    source_dir, reference_dir = Path(source_dir), Path(reference_dir)
    if not (reference_dir / "run.json").exists() or source_dir.resolve() == reference_dir.resolve():
        return None
    worst = 0.0
    for path in sorted(source_dir.glob("*.npz")):
        if path.name.endswith("_maps.npy") or not (reference_dir / path.name).exists():
            continue
        with np.load(path) as new, np.load(reference_dir / path.name) as old:
            a, b = new["eval_score_full"].astype(np.float64), old["eval_score_full"].astype(np.float64)
            worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))))
    return {"reference": str(reference_dir), "max_rel_diff": worst}


# 방법마다 (E0 실행, E1 실행) 쌍으로 표를 만든다. dev 절차 실행만 받는다. E0 교란 실행의 원본(source)이 있으면
# 그 원본과 outputs/<방법>-dev를 rerun_check로 비교한다.
def build_visa_report(
    pairs: dict[str, tuple[Path, Path]], n_boot: int | None = None, seed: int = SEED
) -> dict:
    report: dict = {"n_boot": int(n_boot or N_BOOT), "seed": seed, "alpha": ALPHA, "methods": {}}
    for method, (e0_dir, e1_dir) in pairs.items():
        e0_run, e1_run = load_perturb(e0_dir), load_perturb(e1_dir)
        if e0_run["meta"].get("protocol") != "dev" or e1_run["meta"].get("protocol") != "dev":
            raise ValueError("the centring table is read on the dev protocol only")
        report["methods"][method] = {
            "e0": str(e0_dir),
            "e1": str(e1_dir),
            "e0_commit": e0_run["meta"].get("commit"),
            "e1_commit": e1_run["meta"].get("commit"),
            "table": visa_dev_table(e0_run, e1_run, n_boot=n_boot, seed=seed),
        }
        source = e0_run["meta"].get("source")
        if source:
            report["methods"][method]["e0_rerun_check"] = rerun_check(
                Path(source), paths.OUTPUTS / f"{method}-dev"
            )
    return report


# 콘솔용 E1 표(조건별 E0·E1 AUROC와 오검출률, 차이와 구간).
def format_visa(report: dict) -> str:
    lines = []
    for method, entry in report["methods"].items():
        lines += [
            f"## {method}: E0 {entry['e0_commit']}, E1 {entry['e1_commit']}",
            "| condition | E0 AUROC | E1 AUROC | dAUROC %p | E0 FPR | E1 FPR | dFPR %p |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in entry["table"]:
            lo, hi = row["d_auroc_ci"]
            flo, fhi = row["d_fpr_ci"]
            lines.append(
                f"| {row['condition']} | {row['e0_auroc'] * 100:.2f} | {row['e1_auroc'] * 100:.2f} | "
                f"{row['d_auroc'] * 100:+.2f} [{lo * 100:.2f}, {hi * 100:.2f}] | {row['e0_fpr'] * 100:.1f} | "
                f"{row['e1_fpr'] * 100:.1f} | {row['d_fpr'] * 100:+.1f} [{flo * 100:.1f}, {fhi * 100:.1f}] |"
            )
        lines.append("")
    return "\n".join(lines)


# 명령행 입구. val·test는 outputs/m2ad-enrol-<val|test>-<방법>을, visa-dev는 `방법=<E0 교란 실행>:<E1 교란
# 실행>` 쌍(기본 p0: perturb-p0-dev7 대 perturb-p0-c-dev, d-s: perturb-d-s-dev 대 perturb-d-s-c-dev)을 읽어
# reports/stage7/에 쓴다. 저장된 점수만 읽으므로 이 모듈은 이미지를 읽지 않는다.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("val", "test"):
        p = sub.add_parser(name)
        p.add_argument("--methods", nargs="+", default=["p0", "d-s"])
        p.add_argument("--no-write", action="store_true")
    visa = sub.add_parser("visa-dev")
    visa.add_argument("--no-write", action="store_true")
    visa.add_argument(
        "--pairs",
        nargs="+",
        default=[
            "p0=perturb-p0-dev7:perturb-p0-c-dev",
            "d-s=perturb-d-s-dev:perturb-d-s-c-dev",
        ],
        help="방법=<E0 교란 실행>:<E1 교란 실행>. 실행은 outputs/ 아래 폴더 이름",
    )
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    out_dir = paths.REPORTS / "stage7"
    if args.command == "val":
        runs = {m: load_val(paths.OUTPUTS / f"m2ad-enrol-val-{m}") for m in args.methods}
        report = build_val_report(runs)
        print(format_val(report))
        target = out_dir / "val.json"
    elif args.command == "test":
        runs = {m: load_test(paths.OUTPUTS / f"m2ad-enrol-test-{m}") for m in args.methods}
        report = build_test_report(runs)
        print(format_test(report))
        target = out_dir / "test.json"
    else:
        pairs = {}
        for item in args.pairs:
            # `방법=<E0 실행>:<E1 실행>` 문자열을 나눠 outputs/ 아래 두 폴더로 바꾼다.
            method, _, dirs = item.partition("=")
            e0, _, e1 = dirs.partition(":")
            pairs[method] = (paths.OUTPUTS / e0, paths.OUTPUTS / e1)
        report = build_visa_report(pairs)
        print(format_visa(report))
        target = out_dir / "visa-dev.json"
    if not args.no_write:
        _write(target, report)


if __name__ == "__main__":
    main()
