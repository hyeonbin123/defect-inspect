"""3-A단계 분석: 합성 교란 조건에서의 정확도와 고정 임계값의 비율을 교란 없는 조건과 비교한다.

`defect_inspect.run_perturb`의 실행을 읽는다. 임계값은 원본 실행의 보정 점수로 범주마다 한 번만 정하고 다시
계산하지 않는다. 모든 조건을 같은 층화 부트스트랩 추출로 재표집하므로 교란 없는 조건과의 차이가 짝지어진다.
H9는 docs/experiments.md에 등록한 그대로 판정한다.
"""

# [파일 역할] 3-A(VisA 합성 교란) 분석 모듈이다(`python -m defect_inspect.analyze_perturb --protocol test`).
# run_perturb가 저장한 조건별 점수(교란 없음 + 교란 15조건)를 읽어, 방법(P0, D-S, DM)마다 조건별 평균 이미지
# AUROC와 고정 임계값의 합산 오검출률·검출률, 교란 없는 조건과의 차이를 표로 만들고 가설 H9를 판정한다.
# [핵심 설계] 임계값은 원래 test 실행(P0은 1단계, D-S·DM은 2단계)의 보정 점수로 범주마다 한 번만 정하고
# 조건마다 다시 계산하지 않는다. "정상 이미지로 고정해 둔 임계값이 촬영 조건이 바뀌면 어떻게 되는가"를 재려는
# 것이다. 모든 조건을 같은 층화 부트스트랩 추출로 재표집해, 교란 없는 조건과의 차이가 짝지어진다(docstring).
# [결과] H9 지지: 밝기 0.8·0.7에서 P0의 AUROC는 그대로(+0.33, +0.50%p)인데 오검출률이 15.2%, 41.4%였다.
# AUROC로 감시하면 이 고장이 보이지 않는다(README).
# [연결] 입력은 outputs/perturb-<방법>-<절차>, 출력은 reports/stage3/perturb-<절차>.json이다. drift.py는 같은
# 실행의 점수를 변화 감지 지연에 다시 쓴다.

import argparse
import json
import sys
from pathlib import Path

# scipy의 rankdata는 재표집 2,000개의 AUROC를 행 단위로 한꺼번에 계산하는 데(평균 순위로 동점 처리) 쓴다.
import numpy as np
from scipy.stats import rankdata

from . import paths
from .calibrate import conformal_threshold
from .conditions import CLEAN
from .metrics import auroc
from .stats import percentile_ci, stratified_indices
from .visa import CATEGORIES  # 범주 전체 집합. 등록한 부트스트랩이 도는 순서 그대로다

# N_BOOT·SEED·ALPHA: 공통 규칙의 부트스트랩 2,000번, seed 0, 목표 오검출률 5%. MAX_AUROC_DROP: H9의 "AUROC가
# 1.0%p 미만으로 떨어진다" 기준. H9_METHOD: H9는 P0에서만 판정하고, D-S와 DM의 표는 판정 없이 나란히
# 싣는다(3-A 규칙). _STATS: 교란 없는 조건과의 차이를 낼 통계 세 가지.
N_BOOT = 2000
SEED = 0
ALPHA = 0.05
MAX_AUROC_DROP = 0.01
METHODS = ("p0", "d-s", "dm")
H9_METHOD = "p0"
_STATS = ("auroc", "fpr", "tpr")


# 범주별 npz에는 conditions(조건 이름 목록), scores([조건, 이미지]), eval_labels, cal_score가 들어 있다.
def load(run_dir: Path) -> dict:
    """`run_perturb` 실행 하나를 읽는다: {"meta": run.json, "cats": {범주: 배열}}, 범주는 실행 순서."""
    run_dir = Path(run_dir)
    with open(run_dir / "run.json", encoding="utf-8") as f:
        meta = json.load(f)
    cats = {}
    for info in meta["categories"]:
        with np.load(run_dir / f"{info['category']}.npz") as z:
            cats[info["category"]] = {k: z[k] for k in z.files}
    return {"meta": meta, "cats": cats}


# 모든 범주가 같은 조건 목록을 같은 순서로 가졌는지, 점수 배열 모양이 (조건 수, 이미지 수)이고 유한한지,
# clean이 정확히 하나인지 확인한다. 하나라도 어긋나면 표가 잘못된 짝을 비교하게 되므로 바로 오류를 낸다.
def _condition_names(cats: dict[str, dict[str, np.ndarray]]) -> list[str]:
    """점수 배열 모양을 확인한 뒤, 모든 범주가 함께 쓰는 조건 목록을 돌려준다."""
    if not cats:
        raise ValueError("the run has no categories")
    names: list[str] | None = None
    for c, cat in cats.items():
        mine = [str(name) for name in cat["conditions"]]
        if names is None:
            names = mine
        elif mine != names:
            raise ValueError(f"{c} lists other conditions than {next(iter(cats))}")
        if cat["scores"].shape != (len(mine), len(cat["eval_labels"])):
            raise ValueError(f"{c}: scores {cat['scores'].shape} do not match conditions x images")
        if not np.isfinite(cat["scores"]).all():
            raise ValueError(f"{c}: scores are not finite")
    assert names is not None
    if names.count(CLEAN) != 1:
        raise ValueError(f"the run needs exactly one {CLEAN!r} condition, got {names}")
    return names


# 보정 점수는 P0·D-S는 교차 적합, DM은 홀드아웃 점수다. 모든 범주에 보정 점수가 없으면(코드 점검용 실행)
# None을 돌려 비율 칸을 비우고, 일부 범주만 없으면 실행이 일관되지 않은 것이라 오류를 낸다.
def _thresholds(cats: dict[str, dict[str, np.ndarray]], alpha: float) -> dict | None:
    """원래 실행의 `cal_score`로 범주별 conformal 임계값(목표 alpha)을 정한다. 보정 점수가 없으면 None."""
    empty = [c for c, cat in cats.items() if len(cat["cal_score"]) == 0]
    if len(empty) == len(cats):
        return None
    if empty:
        raise ValueError(f"calibration scores are missing for {empty} only: the run is inconsistent")
    return {c: conformal_threshold(cat["cal_score"], alpha) for c, cat in cats.items()}


# Mann-Whitney U 통계량으로 계산한다: 행(재표집)마다 정상·결함 점수를 합쳐 평균 순위를 매기고, 결함 순위
# 합에서 n_pos(n_pos+1)/2를 빼 n_neg × n_pos로 나눈다. metrics.auroc와 같은 값(동점은 평균 순위)이지만 2,000번
# 반복문 없이 배열 연산 한 번이다.
def _auroc_rows(neg: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """재표집마다의 `metrics.auroc`: neg [B, n_neg], pos [B, n_pos] -> [B] (한쪽 클래스가 없으면 nan)."""
    n_neg, n_pos = neg.shape[1], pos.shape[1]
    if n_neg == 0 or n_pos == 0:
        return np.full(neg.shape[0], np.nan)
    ranks = rankdata(np.concatenate([neg, pos], axis=1), axis=1)  # 재표집(행)마다 평균 순위
    u = ranks[:, n_neg:].sum(axis=1) - n_pos * (n_pos + 1) / 2
    return u / (n_neg * n_pos)


def _ci(samples: np.ndarray) -> list[float]:
    lo, hi = percentile_ci(samples)
    return [float(lo), float(hi)]


# 개수 / 전체. 전체가 0이면 0으로 나누지 않고 nan을 돌려준다(배열이면 원소마다 nan).
def _rate(count: float | np.ndarray, total: int) -> float | np.ndarray:
    return count / total if total else count * float("nan")


# 흐름: 범주별 정상·결함 마스크와 크기로 층화 추출 한 벌을 만든다 → 조건마다 범주별 AUROC 평균(점 추정)과
# 재표집 AUROC를 더하고, 고정 임계값으로 오검출·검출을 세어 합산 비율과 재표집 비율을 만든다 → 교란 없는
# 조건이 아닌 행에는 차이(조건 − clean)와 짝지은 구간을 붙인다. 이미지가 조건 사이에서 같고 추출도 같으므로
# 차이의 구간이 짝지어진다(3-A 한계 절).
def condition_table(run: dict, alpha: float = 0.05, n_boot: int | None = None, seed: int = 0) -> list[dict]:
    """조건마다 한 행: 평균 이미지 AUROC와 고정 임계값에서의 합산 비율.

    행에는 `condition`, `macro_image_auroc`, `fpr`, `tpr`(범주를 합친 값, 규칙 `score > threshold`, 실행에
    보정 점수가 없으면 None)과 각 `_ci` 구간이 들어간다. clean을 뺀 모든 행에는 clean 대비 차이 `d_auroc`,
    `d_fpr`, `d_tpr`와 `_ci`도 들어간다. 부트스트랩은 범주마다 정상과 결함을 따로 재표집하고(묶음은 범주
    순서), 모든 조건에 같은 추출을 쓴다.
    """
    n_boot = n_boot or N_BOOT
    cats = run["cats"]
    names = _condition_names(cats)
    clean = names.index(CLEAN)
    thresholds = _thresholds(cats, alpha)

    masks, sizes = [], []
    for cat in cats.values():
        labels = cat["eval_labels"]
        masks.append((labels == 0, labels == 1))
        sizes += [int((labels == 0).sum()), int((labels == 1).sum())]
    # 범주 순서대로 (정상 수, 결함 수) 묶음의 층화 추출 한 벌. 모든 조건이 이 추출을 같이 쓴다.
    draws = stratified_indices(sizes, n_boot=n_boot, seed=seed)
    n_neg, n_pos = sum(sizes[0::2]), sum(sizes[1::2])

    points: list[dict[str, float]] = []
    boots: list[dict[str, np.ndarray]] = []
    for k in range(len(names)):
        per_cat = []
        auc_boot = np.zeros(n_boot)
        fp = tp = 0
        fp_boot = np.zeros(n_boot)
        tp_boot = np.zeros(n_boot)
        for i, (c, cat) in enumerate(cats.items()):
            scores = cat["scores"][k].astype(np.float64)
            neg, pos = scores[masks[i][0]], scores[masks[i][1]]
            neg_idx, pos_idx = draws[2 * i], draws[2 * i + 1]
            per_cat.append(auroc(neg, pos))
            # neg[neg_idx]는 [n_boot, 정상 수] 모양이라 재표집 2,000개의 범주 AUROC가 한 번에 나온다. 범주마다
            # 더한 뒤 아래에서 범주 수로 나눠 평균한다.
            auc_boot += _auroc_rows(neg[neg_idx], pos[pos_idx])
            if thresholds is not None:
                # 임계값은 원본 실행의 보정 점수로 한 번 정한 값이라 모든 조건에서 같고, 규칙대로 점수 >
                # 임계값만 불량이다.
                f_neg, f_pos = neg > thresholds[c].value, pos > thresholds[c].value
                fp += int(f_neg.sum())
                tp += int(f_pos.sum())
                fp_boot += f_neg[neg_idx].sum(axis=1)
                tp_boot += f_pos[pos_idx].sum(axis=1)
        # 조건 하나의 점 추정값과 재표집 표본. 비율은 12개 범주의 오검출·검출 수를 합쳐 정상·결함 전체 수로
        # 나눈 합산 값이다.
        point = {"auroc": float(np.mean(per_cat))}
        boot = {"auroc": auc_boot / len(cats)}
        if thresholds is not None:
            point.update(fpr=float(_rate(fp, n_neg)), tpr=float(_rate(tp, n_pos)))
            boot.update(fpr=_rate(fp_boot, n_neg), tpr=_rate(tp_boot, n_pos))
        points.append(point)
        boots.append(boot)

    rows = []
    for k, name in enumerate(names):
        point, boot = points[k], boots[k]
        row: dict = {
            "condition": name,
            "macro_image_auroc": point["auroc"],
            "macro_image_auroc_ci": _ci(boot["auroc"]),
        }
        for key in ("fpr", "tpr"):
            row[key] = point.get(key)
            row[f"{key}_ci"] = _ci(boot[key]) if key in boot else None
        if k != clean:
            for key in _STATS:
                if key in point:
                    row[f"d_{key}"] = point[key] - points[clean][key]
                    row[f"d_{key}_ci"] = _ci(boot[key] - boots[clean][key])  # 같은 추출이라 짝지은 차이
                else:
                    row[f"d_{key}"] = row[f"d_{key}_ci"] = None
        rows.append(row)
    return rows


# 해당 조건을 모두 적는다. 고정 임계값이 없는 실행이면 판정할 수 없어 오류를 낸다.
def h9(table: list[dict], alpha: float = 0.05) -> dict:
    """등록한 H9를 점 추정값으로 판정한다: 정확도로는 드러나지 않는 조건이 있는가?

    찾는 조건은 평균 이미지 AUROC가 1.0%p 미만으로 떨어지는데(`d_auroc > -0.01`) 합산 오검출률이 목표의
    두 배 이상인(`fpr >= 2 * alpha`, 10%) 것이다. 하나라도 있으면 지지, 없으면 기각이다.
    """
    found = []
    for row in table:
        if "d_auroc" not in row:  # clean 행
            continue
        if row["fpr"] is None:
            raise ValueError("H9 needs fixed thresholds, but the run has no calibration scores")
        if row["d_auroc"] > -MAX_AUROC_DROP and row["fpr"] >= 2 * alpha:
            found.append(row["condition"])
    return {"verdict": "지지" if found else "기각", "conditions": found}


# 적는 것: 실행 커밋·장치, 원본 실행과 그 커밋, 범주, 정상·결함·보정 점수 수, 보장 조건을 채운 범주 수, 그리고
# clean_max_rel_diff(교란 없는 조건을 다시 채점한 점수와 원래 실행 점수의 최대 상대 차이. 3-A에서 모두
# 0이었다).
def describe(run: dict, alpha: float = ALPHA) -> dict:
    """실행이 어디서 왔고, 그 표 뒤에 이미지와 보정 점수가 몇 개 있는지 적는다."""
    meta, cats = run["meta"], run["cats"]
    thresholds = _thresholds(cats, alpha)
    diffs = [info.get("clean_max_rel_diff") for info in meta.get("categories", [])]
    return {
        "commit": meta.get("commit"),
        "device": meta.get("device"),
        "source": meta.get("source"),
        "source_commit": meta.get("source_commit"),
        "categories": list(cats),
        "n_normal": int(sum((cat["eval_labels"] == 0).sum() for cat in cats.values())),
        "n_defect": int(sum((cat["eval_labels"] == 1).sum() for cat in cats.values())),
        "n_cal": int(sum(len(cat["cal_score"]) for cat in cats.values())),
        "categories_guaranteed": (
            None if thresholds is None else int(sum(t.guaranteed for t in thresholds.values()))
        ),
        "clean_max_rel_diff": max((d for d in diffs if d is not None), default=None),
    }


# 실행 폴더의 method가 이름과 맞는지도 확인한다.
def build_report(
    run_dirs: dict[str, Path], alpha: float = ALPHA, n_boot: int | None = None, seed: int = SEED
) -> dict:
    """모든 방법(이름 -> 실행 폴더)의 표와 H9 판정을 만든다. H9는 `p0`에서만 계산한다.

    방법들의 표를 나란히 놓으므로 모든 방법이 같은 범주를 같은 순서로 다뤄야 한다. `categories`는 그
    목록이고, `complete`는 그것이 결론을 내리는 전체 집합 `CATEGORIES`(12개 범주, 등록 순서)인지를 말한다.
    일부 범주만 돈 실행(코드 점검)은 표만 내고 H9는 판정하지 않는다(`h9`가 None).
    """
    loaded = {}
    for method, run_dir in run_dirs.items():
        run = load(run_dir)
        if run["meta"].get("method") != method:
            raise ValueError(f"{run_dir} is a run of {run['meta'].get('method')!r}, not of {method!r}")
        loaded[method] = run
    covered = {method: list(run["cats"]) for method, run in loaded.items()}
    categories = next(iter(covered.values()), [])
    if any(cats != categories for cats in covered.values()):
        raise ValueError(f"the runs cover different categories and cannot be put side by side: {covered}")
    # 결론은 12개 범주 전체로만 내므로, 범주 목록과 순서가 등록된 CATEGORIES와 같을 때만 complete다(순서가
    # 같아야 부트스트랩 추출도 등록대로 같다).
    complete = categories == list(CATEGORIES)

    tables, runs = {}, {}
    for method, run in loaded.items():
        tables[method] = condition_table(run, alpha, n_boot, seed)
        runs[method] = {"protocol": run["meta"].get("protocol"), **describe(run, alpha)}
    verdict = None
    if H9_METHOD in tables and complete:
        verdict = {"method": H9_METHOD, **h9(tables[H9_METHOD], alpha)}
    return {
        "alpha": alpha,
        "n_boot": n_boot or N_BOOT,
        "seed": seed,
        "categories": categories,
        "complete": complete,
        "runs": runs,
        "tables": tables,
        "h9": verdict,
    }


# 부분 실행이면 H9를 판정하지 않았다고 적는다.
def format_coverage(report: dict) -> str:
    """리포트의 표가 기대는 것(범주, 방법별 이미지·보정 점수 수)을 사람이 읽을 문장으로 쓴다."""
    categories = report["categories"]
    lines = [f"categories ({len(categories)} of {len(CATEGORIES)}): {', '.join(categories)}"]
    for method, info in report["runs"].items():
        lines.append(
            f"{method}: {info['n_normal']} normal and {info['n_defect']} defect images, "
            f"{info['n_cal']} calibration scores"
        )
    if not report["complete"]:
        lines.append(f"partial run: H9 is not judged (it is registered on all {len(CATEGORIES)} categories)")
    return "\n".join(lines)


def _value(x: float | None, ci: list[float] | None) -> str:
    if x is None or ci is None:
        return "-"
    return f"{x * 100:.2f} [{ci[0] * 100:.2f}, {ci[1] * 100:.2f}]"


def _change(row: dict, key: str) -> str:
    x, ci = row.get(f"d_{key}"), row.get(f"d_{key}_ci")
    if x is None or ci is None:
        return "-"
    return f"{x * 100:+.2f} [{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}]"


# docs/experiments.md 3-A 결과 표의 원본이다.
def format_tables(tables: dict[str, list[dict]], verdict: dict | None) -> str:
    """마크다운: 방법마다 표 하나(값은 %, clean 대비 변화는 %p와 구간), 그다음 H9 한 줄."""
    lines = []
    for method, table in tables.items():
        lines += [
            f"method {method}",
            "| condition | image AUROC | change | pooled FPR | change | pooled detection | change |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in table:
            cells = [
                row["condition"],
                _value(row["macro_image_auroc"], row["macro_image_auroc_ci"]),
                _change(row, "auroc"),
                _value(row["fpr"], row["fpr_ci"]),
                _change(row, "fpr"),
                _value(row["tpr"], row["tpr_ci"]),
                _change(row, "tpr"),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    if verdict is not None:
        name = f" ({verdict['method']})" if "method" in verdict else ""
        found = ", ".join(verdict["conditions"]) or "none"
        lines.append(f"H9{name}: {verdict['verdict']} (conditions: {found})")
    return "\n".join(lines)


# 명령행 입구. outputs/perturb-<방법>-<절차>를 읽는다. 실행의 절차가 요청과 다르면 멈추고, test 절차는 12개
# 범주 전체가 아니면 멈춘다(결과는 모든 범주로 말한다는 규칙). 저장된 점수만 읽으므로 봉인 테스트 이미지를
# 다시 읽지 않는다.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", choices=["dev", "test"], required=True)
    parser.add_argument(
        "--methods", nargs="*", choices=METHODS, default=list(METHODS), help="표로 만들 방법의 실행"
    )
    parser.add_argument("--no-write", action="store_true", help="파일을 쓰지 않고 표만 출력한다")
    args = parser.parse_args(argv)
    if not args.methods:
        parser.error("--methods needs at least one method")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # cp949 콘솔에서도 한국어 판정 문구가 깨지지 않게 한다

    run_dirs = {m: paths.OUTPUTS / f"perturb-{m}-{args.protocol}" for m in args.methods}
    missing = [str(d) for d in run_dirs.values() if not (d / "run.json").exists()]
    if missing:
        parser.error(f"no finished run in {missing}: run defect_inspect.run_perturb first, or pass --methods")
    report = {"protocol": args.protocol, **build_report(run_dirs)}
    # 요청한 절차(dev/test)와 다른 절차로 만든 실행이 섞이면 멈춘다.
    wrong = [m for m, info in report["runs"].items() if info["protocol"] != args.protocol]
    if wrong:
        parser.error(f"the runs of {wrong} were not made with the {args.protocol} protocol")
    if args.protocol == "test" and not report["complete"]:
        # 봉인 테스트의 표가 곧 결과이고, 결과는 모든 범주에 대해 말한다.
        parser.error(
            f"the runs cover {report['categories']}, but the test protocol is reported over all "
            f"categories in the registered order {list(CATEGORIES)}: redo run_perturb without --categories"
        )
    print(format_coverage(report))
    print()
    print(format_tables(report["tables"], report["h9"]))
    if args.no_write:
        return
    out_dir = paths.REPORTS / "stage3"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"perturb-{args.protocol}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
