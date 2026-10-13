"""Table over PatchCore settings and coreset ratios: accuracy, fixed-threshold rates, bank size, latency."""

# [파일 역할] 4단계 격자 표와 서빙 구성 선택을 만드는 분석 모듈이다(`python -m defect_inspect.analyze_grid
# --protocol dev`). PatchCore 다섯 설정(백본 × 입력 크기) × 코어셋 세 비율(10%, 1%, 0.1%)마다 dev 정확도, 고정
# 임계값의 오검출률·검출률, 뱅크 크기, CPU 지연을 한 표에 모으고, 등록한 규칙대로 서빙 구성을 고른다.
# [선택 규칙] CPU 지연(리사이즈 + FP32 추론)이 200ms 이하인 행 가운데 dev 평균 이미지 AUROC가 가장 높은 것,
# 같으면 뱅크가 작은 쪽. 200ms 이하가 없으면 가장 빠른 것을 고르고 H15는 기각이다(docs/experiments.md 4단계).
# 결과: wrn50-256 코어셋 1%(한가할 때 184.2ms, dev AUROC 91.44). 서빙 구성은 dev로만 고르고 test 수치로는
# 아무것도 고르지 않는다.
# [연결] 입력은 run_grid 실행 폴더(outputs/grid-<설정>-<절차>/<비율>)와 bench가 쓴 지연
# 파일(reports/stage4/latency.json), 출력은 reports/stage4/grid-<절차>.json이다. 부트스트랩·요약·차이는
# compare 모듈 것을 쓴다.

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# compare는 실행 읽기·부트스트랩·요약·차이 판정을, run_grid.ratio_name은 비율 폴더 이름 규칙을 준다(run_grid가
# 쓴 이름과 같아야 해서 가져다 쓴다).
from . import compare, paths
from .run_grid import ratio_name

# SETTINGS: 등록한 다섯 설정 이름(<백본>-<입력 크기>). DINOv2는 입력이 14의 배수여야 해서 252·392px, 448은
# 2단계 D-S와 같은 설정이다. CPU_BUDGET_MS: 질문 4의 CPU 예산(장당 200ms).
# The stage 4 grid and the CPU budget of the serving choice, as registered in docs/experiments.md.
SETTINGS = ("wrn50-256", "wrn50-384", "dinov2_vits14-252", "dinov2_vits14-392", "dinov2_vits14-448")
CPU_BUDGET_MS = 200.0


# 실행마다 평균 AUROC 부트스트랩 표본을 한 번만 계산해 기억해 두는 Bootstrap. 행의 구간(compare.summarize
# 안)과 최대 비율 대비 차이가 같은 표본을 필요로 하는데, 재표집이 표의 느린 부분이라 실행마다 한 번만 돌게
# 한다(docstring). keep_pos를 주는 호출은 기억하지 않는다.
class _SharedBootstrap(compare.Bootstrap):
    """`compare.Bootstrap` that keeps the macro AUROC samples of every run it has resampled.

    The interval of a row (`compare.summarize`) and the paired differences need the same samples; the
    resampling is the slow part of the table, so each run goes through it once.
    """

    def __init__(self, reference: compare.MethodRun):
        super().__init__(reference)
        self._auroc: dict[int, tuple[compare.MethodRun, np.ndarray]] = {}

    def macro_auroc(
        self, run: compare.MethodRun, keep_pos: dict[str, np.ndarray] | None = None
    ) -> np.ndarray:
        if keep_pos is not None:
            return super().macro_auroc(run, keep_pos)
        # 실행 객체의 id를 키로 쓴다. 객체 자체도 값에 같이 넣어 두므로 그 객체가 사라져 id가 다른 객체에 다시
        # 쓰이는 일이 없다(코드 주석).
        if id(run) not in self._auroc:  # the run is stored with its samples, so the id cannot be reused
            self._auroc[id(run)] = (run, super().macro_auroc(run))
        return self._auroc[id(run)][1]


# run_grid 실행 하나(설정 하나)의 비율 폴더들을 큰 비율부터 읽는다. 각 폴더는 run_patchcore와 같은 형식이라
# compare.load_patchcore로 읽고, 이름은 `<설정>-r<비율>`(지연 파일의 키와 같은 형식)이다.
def load_grid(out_dir: Path) -> dict[float, compare.MethodRun]:
    """Every ratio folder of one `run_grid` call, largest ratio first."""
    with open(out_dir / "run.json", encoding="utf-8") as f:
        meta = json.load(f)
    return {
        float(r): compare.load_patchcore(out_dir / ratio_name(r), name=f"{meta['setting']}-{ratio_name(r)}")
        for r in sorted(meta["ratios"], reverse=True)
    }


# (설정, 비율)마다 한 행: 범주 평균 뱅크 행 수, 평균 이미지 AUROC·AUPRO와 구간, 합산 FPR@TPR95, 교차 적합
# 임계값(목표 5%)의 합산 오검출률·검출률. 같은 설정 안에서는 가장 큰 비율(10%) 대비 AUROC 차이와 판정("차이
# 있음/작음/판정 불가")도 붙인다. 이것이 H13(코어셋 10% → 1%에서 1.0%p 넘게 떨어지지 않음)의 근거 수치다. 모든
# 실행이 같은 dev 이미지를 채점했는지 먼저 확인하고 부트스트랩 한 벌을 공유한다.
def grid_table(runs: dict[str, dict[float, compare.MethodRun]], alpha: float = 0.05) -> list[dict]:
    """One row per (setting, ratio). All runs must score the same evaluation images (same bootstrap draws)."""
    flat = [run for ratios in runs.values() for run in ratios.values()]
    compare.check_same_images(flat)
    boot = _SharedBootstrap(flat[0])
    rows = []
    for setting, ratios in runs.items():
        largest = max(ratios)
        base_point = compare.macro_auroc(ratios[largest])
        base_boot = boot.macro_auroc(ratios[largest])
        for ratio in sorted(ratios, reverse=True):
            run = ratios[ratio]
            # summarize의 고정 임계값 비율은 compare.ALPHA(5%) 기준이라, 다른 alpha를 요청했을 때만 따로
            # 계산한다.
            summary = compare.summarize(run, boot)  # its fixed-threshold rates are at compare.ALPHA
            fixed = summary["fixed_threshold"] if alpha == compare.ALPHA else boot.pooled_rates(run, alpha)
            row = {
                "key": run.name,
                "setting": setting,
                "ratio": ratio,
                "bank_rows_mean": float(np.mean([int(cat["bank_rows"]) for cat in run.cats.values()])),
                "macro_image_auroc": summary["macro_image_auroc"],
                "macro_image_auroc_ci": summary["macro_image_auroc_ci"],
                "macro_aupro": summary["macro_aupro"],
                "macro_aupro_ci": summary["macro_aupro_ci"],
                "pooled_fpr_at_tpr95": summary["pooled_fpr_at_tpr95"],
                "fpr": fixed["fpr"],
                "fpr_ci": fixed["fpr_ci"],
                "tpr": fixed["tpr"],
                "tpr_ci": fixed["tpr_ci"],
            }
            if ratio != largest:
                row["vs_largest_ratio"] = compare.difference(
                    compare.macro_auroc(run) - base_point, boot.macro_auroc(run) - base_boot
                )
            rows.append(row)
    return rows


# 지연 파일에서 행의 키와 같은 항목을 붙이고, 등록된 CPU 지연 cpu_ms = 리사이즈 + FP32 추론 중앙값을 계산한다.
# 추론 시간만으로 예산과 비교하면 안 되므로, CPU 측정이 있는데 리사이즈 시간이 없으면 오류를 낸다(규칙 변경 5
# 때 지연 표가 리사이즈를 빼고 있던 것을 고친 기록이 있다). GPU만 잰 항목에는 cpu_ms를 붙이지 않는다.
def merge_latency(rows: list[dict], latency: dict) -> None:
    """Attach the latency entry with the row's key (`<backbone>-<size>-r<ratio>`), when there is one.

    `cpu_ms` is the registered CPU latency of a row: the time to resize a camera-sized image to the input
    size plus the FP32 onnxruntime inference (medians, as `bench` writes them). An entry without CPU
    timings (GPU only) gets no `cpu_ms`; CPU timings without the resize time are an error, because the
    inference time alone must not be compared with the latency budget.
    """
    for row in rows:
        entry = latency.get(row["key"])
        if entry is None:
            continue
        row["latency"] = entry
        if "cpu_fp32" not in entry:
            continue
        if "resize_ms" not in entry:
            raise ValueError(f"latency entry {row['key']} has cpu_fp32 but no resize_ms: run bench again")
        # 등록된 CPU 지연 = 리사이즈 중앙값 + FP32 추론 중앙값.
        row["cpu_ms"] = float(entry["resize_ms"]) + float(entry["cpu_fp32"]["total_ms"])


# 등록한 규칙으로 서빙 구성을 고른다. min의 정렬 키 (−AUROC, 뱅크 행 수)로 "AUROC 최고, 같으면 작은 뱅크"를 한
# 번에 표현한다. 예산 안에 든 후보 목록과 지연을 재지 않아 고를 수 없던 행도 기록한다. 지연을 처음에는 부하
# 상태에서 쟀고, 규칙 변경 6 뒤 한가할 때 다시 잰 값으로 골랐다. 부하 상태의 값으로는
# dinov2_vits14-252-r0.01이 뽑혔다(두 측정에서 고른 구성이 다르다는 것도 기록).
def choose_serving(rows: list[dict], budget_ms: float = CPU_BUDGET_MS) -> dict | None:
    """The serving configuration by the registered rule (docs/experiments.md, stage 4), from `cpu_ms`.

    Among the rows whose CPU latency (resize + inference) is within the budget: the highest macro image
    AUROC, ties to the smaller bank. When no row is within the budget: the fastest row, and
    `within_budget` is False (H15 is then rejected). Rows without a CPU latency cannot be chosen and are
    listed under `unmeasured`. None when no row has a CPU latency.
    """
    measured = [r for r in rows if "cpu_ms" in r]
    if not measured:
        return None
    within = [r for r in measured if r["cpu_ms"] <= budget_ms]
    if within:
        best = min(within, key=lambda r: (-r["macro_image_auroc"], r["bank_rows_mean"]))
    else:
        best = min(measured, key=lambda r: r["cpu_ms"])
    return {
        "key": best["key"],
        "setting": best["setting"],
        "ratio": best["ratio"],
        "cpu_ms": best["cpu_ms"],
        "macro_image_auroc": best["macro_image_auroc"],
        "budget_ms": budget_ms,
        "within_budget": bool(within),
        "candidates": [r["key"] for r in within],
        "unmeasured": [r["key"] for r in rows if "cpu_ms" not in r],
    }


def _pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}"


def _span(ci: list[float]) -> str:
    return f"[{_pct(ci[0])}, {_pct(ci[1])}]"


# 콘솔용 마크다운 격자 표. docs/experiments.md 4단계 dev 격자 표의 원본이다.
def format_table(rows: list[dict]) -> str:
    header = [
        "setting",
        "ratio",
        "bank rows",
        "image AUROC",
        "vs largest",
        "AUPRO",
        "actual FPR (5%)",
        "detection",
        "CPU ms (resize + inference)",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        d = r.get("vs_largest_ratio")
        cpu = r.get("cpu_ms")
        cells = [
            r["setting"],
            f"{r['ratio']:g}",
            f"{r['bank_rows_mean']:.0f}",
            f"{_pct(r['macro_image_auroc'])} {_span(r['macro_image_auroc_ci'])}",
            f"{d['diff'] * 100:+.2f}%p {d['verdict']}" if d else "-",
            _pct(r["macro_aupro"]),
            f"{_pct(r['fpr'])} {_span(r['fpr_ci'])}",
            f"{_pct(r['tpr'])} {_span(r['tpr_ci'])}",
            f"{cpu:.1f}" if cpu is not None else "-",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# 서빙 선택 한 줄 설명(예산 안 최고 AUROC인지, 예산 안이 없어 가장 빠른 것인지).
def format_serving(serving: dict) -> str:
    if serving["within_budget"]:
        reason = f"highest image AUROC among {len(serving['candidates'])} rows within the budget"
    else:
        reason = "no row is within the budget, so the fastest one"
    text = (
        f"serving configuration: {serving['key']} ({serving['cpu_ms']:.1f} ms on the CPU, budget "
        f"{serving['budget_ms']:g} ms; {reason})"
    )
    if serving["unmeasured"]:
        text += f"\nno CPU latency, not considered: {', '.join(serving['unmeasured'])}"
    return text


# 명령행 입구. --settings를 주지 않으면 등록한 다섯 설정 가운데 끝난 실행만 쓰고 빠진 설정을 알린다.
# --settings를 주면 하나라도 없을 때 멈춘다(4단계 test는 서빙 설정 wrn50-256 하나만 돌렸다).
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", choices=["dev", "test"], required=True)
    parser.add_argument(
        "--settings",
        nargs="+",
        default=None,
        help="e.g. wrn50-256 dinov2_vits14-392 (default: the registered settings with a finished grid run)",
    )
    parser.add_argument("--latency", type=Path, default=None, help="default: reports/stage4/latency.json")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    def run_dir(setting: str) -> Path:
        return paths.OUTPUTS / f"grid-{setting}-{args.protocol}"

    # run.json이 있는(끝까지 돈) 실행만 표에 넣는다.
    finished = [s for s in args.settings or SETTINGS if (run_dir(s) / "run.json").exists()]
    if args.settings is None:
        if not finished:
            parser.error(f"no finished grid run of the {args.protocol} protocol under {paths.OUTPUTS}")
        skipped = [s for s in SETTINGS if s not in finished]
        if skipped:
            print(f"no finished grid run, left out: {', '.join(skipped)}", file=sys.stderr)
    elif len(finished) != len(args.settings):
        missing = [str(run_dir(s)) for s in args.settings if s not in finished]
        parser.error(f"no finished grid run in: {', '.join(missing)}")
    if args.latency is not None and not args.latency.exists():
        parser.error(f"latency file not found: {args.latency}")

    runs = {s: load_grid(run_dir(s)) for s in finished}
    rows = grid_table(runs)
    # 지연 파일이 있으면 행에 붙인다. 기본 파일 reports/stage4/latency.json에는 규칙 변경 6에 따라 CPU가
    # 한가할 때 다시 잰 값이 들어 있다(처음 잰 부하 상태 값은 latency-loaded.json).
    latency_path = args.latency or paths.REPORTS / "stage4" / "latency.json"
    if latency_path.exists():
        with open(latency_path, encoding="utf-8") as f:
            merge_latency(rows, json.load(f))
    report = {"protocol": args.protocol, "alpha": compare.ALPHA, "rows": rows}
    print(format_table(rows))
    # 서빙 선택은 dev 절차에서만 한다. test 실행의 표는 선택에 쓰지 않는다(봉인 테스트 수치로 구성을 고르지
    # 않는다는 규칙).
    # The serving configuration is chosen on the dev protocol only; test numbers never choose anything.
    serving = choose_serving(rows) if args.protocol == "dev" else None
    if serving is not None:
        report["serving"] = serving
        print("\n" + format_serving(serving))
    if not args.no_write:
        out_dir = paths.REPORTS / "stage4"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"grid-{args.protocol}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
