"""합성 격자 실행으로 analyze_grid를 시험한다(numpy만 쓴다: 이 파일은 torch 없는 CI 환경에서 돈다).

`grid_table`의 모든 수치를 등록한 규칙에서 다시 계산한다. 그 코드는 난수 추출 방식(seed 0, 범주마다 정상
다음 결함 순서로 한 생성기에서 뽑기) 말고는 compare.py와 아무것도 공유하지 않는다.
"""

# [파일 역할] analyze_grid.py(4단계: PatchCore 설정 × 코어셋 비율 격자 표, CPU 지연 붙이기, 서빙 구성 선택)를
# 검증하는 테스트 묶음. numpy만 써서 torch 없는 CI에서도 돈다(docstring).
# [왜 필요한가] 4단계의 서빙 구성(wrn50-256 코어셋 1%)과 H13·H15 판정이 이 표에서 나온다. 표의 모든 수치를
# 등록 규칙에서 바로 다시 계산하는 독립 구현(_reference)과 비교하고, 같은 부트스트랩 추출 공유, 지연 =
# 리사이즈 + 추론, 서빙 선택 규칙(예산 200ms, AUROC 최고, 같으면 작은 뱅크), test 절차에서는 고르지 않는다는
# 규칙을 확인한다.
# [구성] write_grid가 run_grid와 같은 폴더 구조의 합성 실행을 만들고, 앞쪽은 표 계산, 가운데는 지연·서빙 선택,
# 뒤쪽은 명령행(main) 테스트다.

import json
import math
from fractions import Fraction

import numpy as np
import pytest

# compare는 부트스트랩 횟수를 줄이고 요약 함수와 비교하려고, paths는 출력·리포트 폴더를 임시 폴더로 바꾸려고
# 가져온다.
from defect_inspect import analyze_grid, compare, paths

# 합성 격자의 크기: 부트스트랩 200번, 범주 두 개(평가 정상·결함·풀 정상 장수가 서로 다름), 비율 세 개(10%, 1%,
# 0.1%).
N_BOOT = 200
CATS = {"a": (31, 12, 90), "b": (45, 10, 130)}  # 평가 정상, 결함, 풀 정상
RATIOS = [0.1, 0.01, 0.001]


# run_grid 형식의 폴더(outputs/grid-<설정>-<절차>/r<비율>/<범주>.npz, 비율 폴더별 run.json, 맨 위 run.json)를
# 만든다. 비율이 작을수록 결함 점수를 내리고 잡음을 키워 정상과 결함이 덜 갈리게 해서, "작은 비율 − 가장 큰
# 비율" 차이가 음수로 나오게 한다. 점수를 소수 한 자리로 반올림해 동점을 많이 만들어 동점 처리 규칙까지
# 시험한다. listed로 맨 위 run.json에 적을 비율을 따로 줄 수 있다(남은 옛 폴더 시험용).
def write_grid(root, setting, ratios, seed, cats=CATS, listed=None, protocol="dev"):
    """run_grid 구조의 폴더. 비율이 작을수록 정상과 결함이 덜 갈린다."""
    out = root / f"grid-{setting}-{protocol}"
    rng = np.random.default_rng(seed)
    base = {
        c: (rng.normal(0, 1, n), rng.normal(1.5, 1, p), rng.normal(0, 1, m)) for c, (n, p, m) in cats.items()
    }
    for k, ratio in enumerate(sorted(ratios, reverse=True)):
        folder = out / f"r{ratio!r}"
        folder.mkdir(parents=True)
        noise = np.random.default_rng(1000 * seed + k)
        for c, (n, p, m) in cats.items():
            neg, pos, cal = base[c]
            # 소수 한 자리: 정상, 결함, 임계값 사이에 동점이 많이 생긴다.
            scores = np.round(np.concatenate([neg, pos - 0.3 * k]) + noise.normal(0, 0.2 * k, n + p), 1)
            np.savez(
                folder / f"{c}.npz",
                eval_images=np.array([f"{c}/e{i}.JPG" for i in range(n + p)]),
                eval_labels=np.array([0] * n + [1] * p, dtype=np.int8),
                eval_defect_types=np.array([""] * n + ["hole"] * p),
                eval_score_full=scores.astype(np.float32),
                pool_images=np.array([f"{c}/p{i}.JPG" for i in range(m)]),
                pool_folds=np.arange(m) % 4 + 1,
                pool_score_oof=np.round(cal + noise.normal(0, 0.2 * k, m), 1).astype(np.float32),
                pixel_auroc=np.float64(0.9),
                aupro=np.float64(0.8 - 0.05 * k),
                pro_edges=np.linspace(0, 1, 11),
                pro_normal=noise.integers(1, 9, (n + p, 10)),
                pro_components=noise.integers(1, 9, (p, 10)),
                pro_component_image=np.arange(n, n + p),
                bank_rows=np.int64(round(ratio * m * 100)),
            )
        with open(folder / "run.json", "w", encoding="utf-8") as f:
            json.dump({"config": {"name": "x"}, "categories": [{"category": c} for c in cats]}, f)
    with open(out / "run.json", "w", encoding="utf-8") as f:
        json.dump(
            {"setting": setting, "ratios": listed or sorted(ratios, reverse=True), "protocol": protocol}, f
        )
    return out


# 검증용 AUROC: 모든 (결함, 정상) 쌍에서 결함이 크면 1, 같으면 0.5(동점 평균 순위 규칙과 같음).
def _auroc(neg, pos):
    d = pos[:, None].astype(np.float64) - neg[None, :]
    return ((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size


# 표의 수치를 등록 규칙에서 직접 다시 계산하는 독립 구현(compare.py와 공유하는 것은 난수 추출 방식뿐: seed 0,
# 범주마다 정상 다음 결함 순서로 한 생성기에서 뽑기).
# - 평균 이미지 AUROC와 구간, 고정 임계값(l = ceil((n+1)(1−α))번째 교차 적합 점수, Fraction으로 정확히)의 합산
#   오검출률·검출률과 구간
# - FPR@TPR95: 결함의 95% 이상을 잡는 가장 높은 임계값(점수 ≥ 임계값)에서의 정상 오검출 합산
# - 가장 큰 비율 대비 AUROC 차이와 같은 추출로 낸 짝지은 구간
def _reference(run_dir, ratios, alpha):
    data = {r: {c: dict(np.load(run_dir / f"r{r!r}" / f"{c}.npz")) for c in CATS} for r in ratios}
    rng = np.random.default_rng(0)
    draws = {
        c: (rng.integers(0, n, (N_BOOT, n)), rng.integers(0, p, (N_BOOT, p))) for c, (n, p, _) in CATS.items()
    }
    out, boots = {}, {}
    for ratio, cats in data.items():
        point, boot = [], np.zeros(N_BOOT)
        fp = tp = n_neg = n_pos = fp95 = 0
        fp_b, tp_b = np.zeros(N_BOOT), np.zeros(N_BOOT)
        for c, z in cats.items():
            neg = z["eval_score_full"][z["eval_labels"] == 0]
            pos = z["eval_score_full"][z["eval_labels"] == 1]
            ni, pi = draws[c]
            point.append(_auroc(neg, pos))
            boot += np.array([_auroc(neg[ni[b]], pos[pi[b]]) for b in range(N_BOOT)]) / len(cats)
            cal = np.sort(z["pool_score_oof"].astype(np.float64))
            thr = cal[math.ceil((len(cal) + 1) * (1 - Fraction(str(alpha)))) - 1]
            fp, tp = fp + (neg > thr).sum(), tp + (pos > thr).sum()
            n_neg, n_pos = n_neg + len(neg), n_pos + len(pos)
            fp_b += (neg > thr)[ni].sum(axis=1)
            tp_b += (pos > thr)[pi].sum(axis=1)
            # 결함의 95%를 여전히 잡는 가장 높은 임계값. 판정 규칙은 점수 >= 임계값.
            t95 = np.sort(pos)[::-1][math.ceil(Fraction("0.95") * len(pos)) - 1]
            fp95 += (neg >= t95).sum()
        boots[ratio] = boot
        out[ratio] = {
            "macro_image_auroc": np.mean(point),
            "macro_image_auroc_ci": np.percentile(boot, [2.5, 97.5]),
            "pooled_fpr_at_tpr95": fp95 / n_neg,
            "fpr": fp / n_neg,
            "fpr_ci": np.percentile(fp_b / n_neg, [2.5, 97.5]),
            "tpr": tp / n_pos,
            "tpr_ci": np.percentile(tp_b / n_pos, [2.5, 97.5]),
            "bank_rows_mean": np.mean([int(z["bank_rows"]) for z in cats.values()]),
            "macro_aupro": np.mean([float(z["aupro"]) for z in cats.values()]),
        }
    top = max(ratios)
    for ratio in ratios:
        if ratio != top:
            out[ratio]["diff"] = out[ratio]["macro_image_auroc"] - out[top]["macro_image_auroc"]
            out[ratio]["diff_ci"] = np.percentile(boots[ratio] - boots[top], [2.5, 97.5])
    return out


# 설정 두 개(wrn50-256, dinov2_vits14-392)의 합성 격자를 만든다. 부트스트랩 횟수는 테스트용으로 줄인다.
@pytest.fixture
def grids(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "N_BOOT", N_BOOT)
    return {
        "wrn50-256": write_grid(tmp_path, "wrn50-256", RATIOS, 1),
        "dinov2_vits14-392": write_grid(tmp_path, "dinov2_vits14-392", RATIOS, 2),
    }


# grid_table의 모든 칸이 독립 구현과 1e-12 안에서 같은지 본다. 행 순서(설정 순, 큰 비율부터)와 행
# 키("<설정>-r<비율>", 지연 파일의 키와 같은 형식)도 확인한다. α = 0.2는 compare.ALPHA(5%)가 아닐 때 따로
# 계산하는 경로를 시험한다. 가장 큰 비율 행에는 차이가 없고, 나머지 행은 같은 추출로 낸 차이와 구간을 가져야
# 한다(H13의 근거 수치). 마지막 줄은 fpr과 tpr이 우연히 같아 서로 바뀐 것을 못 알아채는 일이 없게 확인한다.
@pytest.mark.parametrize("alpha", [0.05, 0.2])
def test_rows_match_the_registered_rules(grids, alpha):
    runs = {s: analyze_grid.load_grid(d) for s, d in grids.items()}
    rows = analyze_grid.grid_table(runs, alpha=alpha)
    assert [(r["setting"], r["ratio"]) for r in rows] == [(s, r) for s in grids for r in RATIOS]
    assert [r["key"] for r in rows[:3]] == ["wrn50-256-r0.1", "wrn50-256-r0.01", "wrn50-256-r0.001"]
    exact = ("macro_image_auroc", "macro_image_auroc_ci", "pooled_fpr_at_tpr95", "fpr", "fpr_ci", "tpr")
    for setting, run_dir in grids.items():
        want = _reference(run_dir, RATIOS, alpha)
        for row in (r for r in rows if r["setting"] == setting):
            ref = want[row["ratio"]]
            for key in (*exact, "tpr_ci"):
                np.testing.assert_allclose(row[key], ref[key], rtol=0, atol=1e-12, err_msg=key)
            assert row["bank_rows_mean"] == ref["bank_rows_mean"]
            assert row["macro_aupro"] == pytest.approx(ref["macro_aupro"])
            if row["ratio"] == 0.1:
                assert "vs_largest_ratio" not in row
            else:  # 같은 추출로 낸 "작은 비율 − 가장 큰 비율"
                d = row["vs_largest_ratio"]
                assert d["diff"] == pytest.approx(ref["diff"], abs=1e-12) and d["diff"] < 0
                np.testing.assert_allclose(d["ci"], ref["diff_ci"], rtol=0, atol=1e-12)
    assert rows[0]["fpr"] != rows[0]["tpr"]  # 같은 숫자끼리 바뀌면 독립 구현이 알아채지 못한다


# 격자 표의 행이 2단계 비교에 쓰는 compare.summarize와 같은 숫자를 싣는지 본다. 단계마다 같은 지표를 다른
# 코드로 계산해 어긋나는 일을 막는다.
def test_rows_carry_the_numbers_of_compare_summarize(grids):
    runs = {s: analyze_grid.load_grid(d) for s, d in grids.items()}
    rows = analyze_grid.grid_table(runs)
    flat = [run for ratios in runs.values() for run in ratios.values()]
    boot = compare.Bootstrap(flat[0])
    assert [r["key"] for r in rows] == [run.name for run in flat]
    for row, run in zip(rows, flat, strict=True):
        summary = compare.summarize(run, boot)
        for key in ("macro_image_auroc", "macro_image_auroc_ci", "macro_aupro", "macro_aupro_ci"):
            assert row[key] == summary[key], key
        assert row["pooled_fpr_at_tpr95"] == summary["pooled_fpr_at_tpr95"]
        for key in ("fpr", "fpr_ci", "tpr", "tpr_ci"):
            assert row[key] == summary["fixed_threshold"][key], key


# 설정 순서를 뒤집어 넣어도 표가 똑같아야 한다. 모든 행이 하나의 부트스트랩 추출을 공유한다는 뜻이다(행끼리
# 구간을 비교할 수 있게).
def test_every_row_uses_the_same_draws(grids):
    runs = {s: analyze_grid.load_grid(d) for s, d in grids.items()}
    forward = {r["key"]: r for r in analyze_grid.grid_table(runs)}
    backward = {r["key"]: r for r in analyze_grid.grid_table({s: runs[s] for s in reversed(list(runs))})}
    assert forward == backward


# 실행마다 평균 AUROC 재표집이 한 번만 일어나는지 센다(monkeypatch로 compare.Bootstrap.macro_auroc를 감싸
# 호출을 기록). 행의 구간과 짝지은 차이가 같은 표본을 쓰고, 느린 재표집을 반복하지 않는다는 _SharedBootstrap의
# 약속을 지킨다.
def test_each_run_is_resampled_once(grids, monkeypatch):
    """행의 구간과 짝지은 차이가 평균 AUROC의 부트스트랩 하나를 같이 쓴다."""
    runs = {"wrn50-256": analyze_grid.load_grid(grids["wrn50-256"])}
    calls = []
    real = compare.Bootstrap.macro_auroc

    def counting(self, run, keep_pos=None):
        calls.append(run.name)
        return real(self, run, keep_pos)

    monkeypatch.setattr(compare.Bootstrap, "macro_auroc", counting)
    rows = analyze_grid.grid_table(runs)
    assert sorted(calls) == sorted(r["key"] for r in rows)


# 같은 실행을 두 비율에 넣으면 차이 0, 구간 [0, 0], 판정은 "판정 불가"(구간이 0을 포함)여야 한다.
def test_identical_runs_have_no_difference(grids):
    run = analyze_grid.load_grid(grids["wrn50-256"])[0.1]
    d = analyze_grid.grid_table({"twin": {0.1: run, 0.01: run}})[1]["vs_largest_ratio"]
    assert d["diff"] == 0 and d["ci"] == [0.0, 0.0] and d["verdict"] == "판정 불가"


# 다른 이미지(여기서는 범주가 하나뿐인 실행)를 채점한 실행이 섞이면 같은 추출을 쓸 수 없으므로 ValueError로
# 거절해야 한다.
def test_runs_over_other_images_are_refused(grids, tmp_path):
    other = write_grid(tmp_path, "wrn50-384", [0.1], 3, cats={"a": CATS["a"]})
    runs = {
        "wrn50-256": analyze_grid.load_grid(grids["wrn50-256"]),
        "wrn50-384": analyze_grid.load_grid(other),
    }
    with pytest.raises(ValueError, match="categories"):
        analyze_grid.grid_table(runs)


# load_grid는 맨 위 run.json에 적힌 비율만 읽어야 한다. 이전 호출이 남긴 r0.05 폴더(옛 결과)를 끼워 넣지
# 않는지, run.json이 없는(끝나지 않은) 실행은 FileNotFoundError로 거절하는지 본다.
def test_load_grid_reads_only_what_the_last_call_wrote(tmp_path):
    out = write_grid(tmp_path, "wrn50-256", [0.1, 0.05, 0.01], 3, listed=[0.1, 0.01])  # r0.05는 옛 결과
    runs = analyze_grid.load_grid(out)
    assert list(runs) == [0.1, 0.01] and runs[0.01].name == "wrn50-256-r0.01"
    (out / "run.json").unlink()  # 끝나지 않은 호출에는 run.json이 없다
    with pytest.raises(FileNotFoundError):
        analyze_grid.load_grid(out)


# bench가 쓰는 지연 파일 모양의 합성 예: r0.01은 CPU·GPU 둘 다, r0.001은 GPU만, 비율 없는 키는 어느 행과도
# 맞지 않는다.
LATENCY = {
    "wrn50-256-r0.01": {"resize_ms": 31.0, "cpu_fp32": {"total_ms": 180.0}, "gpu_fp16": {"total_ms": 9.0}},
    "wrn50-256-r0.001": {"resize_ms": 31.0, "gpu_fp16": {"total_ms": 8.0}},
    "wrn50-256": {"resize_ms": 1.0, "cpu_fp32": {"total_ms": 1.0}},  # 비율 없음: 어느 행의 키도 아니다
}


# 등록된 CPU 지연 cpu_ms = 리사이즈 + FP32 추론(31 + 180 = 211ms)인지, 지연 항목이 없는 행과 GPU만 잰 행에는
# cpu_ms를 붙이지 않는지 본다. 규칙 변경 5 때 지연 표가 리사이즈를 빼고 있던 버그를 고친 것의 회귀 테스트다.
def test_cpu_latency_is_resize_plus_inference():
    rows = [{"key": "wrn50-256-r0.1"}, {"key": "wrn50-256-r0.01"}, {"key": "wrn50-256-r0.001"}]
    analyze_grid.merge_latency(rows, LATENCY)
    assert "latency" not in rows[0] and "cpu_ms" not in rows[0]
    assert rows[1]["latency"] == LATENCY["wrn50-256-r0.01"] and rows[1]["cpu_ms"] == 211.0
    # 이 키는 GPU만 쟀다: 보고할 CPU 지연이 없다.
    assert rows[2]["latency"]["gpu_fp16"]["total_ms"] == 8.0 and "cpu_ms" not in rows[2]


# CPU 측정이 있는데 리사이즈 시간이 없으면 추론만으로 예산과 비교하게 되므로 오류로 멈춰야 한다.
def test_cpu_entry_without_the_resize_time_is_refused():
    with pytest.raises(ValueError, match="resize_ms"):
        analyze_grid.merge_latency([{"key": "k-r0.1"}], {"k-r0.1": {"cpu_fp32": {"total_ms": 120.0}}})


# 콘솔 표의 마지막 열 이름이 "리사이즈 + 추론"임을 밝히고, 지연이 있는 행만 숫자, 없는 행은 "-"이며, 모든 줄의
# 칸 수가 같은지(마크다운 표가 깨지지 않는지) 본다.
def test_table_shows_resize_plus_inference(grids):
    rows = analyze_grid.grid_table({"wrn50-256": analyze_grid.load_grid(grids["wrn50-256"])})
    analyze_grid.merge_latency(rows, LATENCY)
    lines = analyze_grid.format_table(rows).splitlines()
    assert lines[0].split(" | ")[-1] == "CPU ms (resize + inference) |"
    cells = {line.split(" | ")[1]: line.split(" | ")[-1] for line in lines[2:]}
    assert cells == {"0.1": "- |", "0.01": "211.0 |", "0.001": "- |"}
    assert all(line.count("|") == lines[0].count("|") for line in lines)


# 서빙 선택 테스트용 최소 행(키, 설정, 비율, AUROC, 뱅크 행 수, 선택적으로 CPU 지연).
def _row(key, auroc, bank_rows, cpu_ms=None):
    setting, ratio = key.rsplit("-r", 1)
    row = {"key": key, "setting": setting, "ratio": float(ratio)}
    row.update(macro_image_auroc=auroc, bank_rows_mean=float(bank_rows))
    if cpu_ms is not None:
        row["cpu_ms"] = cpu_ms
    return row


# 서빙 선택 규칙(docs/experiments.md 4단계)을 확인한다.
# - 가장 정확한 행도 예산(200ms)을 넘으면 후보가 아니다. 정확히 200ms는 예산 안이다
# - AUROC가 같으면 뱅크가 작은 쪽, 행 순서를 뒤집어도 같은 선택
# - 지연을 재지 않은 행은 고를 수 없고 unmeasured로 기록
# - 예산 안이 하나도 없으면 가장 빠른 행을 고르고 within_budget이 False(H15 기각의 경우), 지연이 있는 행이
#   없으면 None
def test_serving_choice_follows_the_registered_rule():
    rows = [
        _row("a-r0.1", 0.95, 1000, 200.5),  # 가장 정확하지만 예산을 넘는다
        _row("a-r0.01", 0.93, 100, 200.0),  # 정확히 예산이면 예산 안으로 친다
        _row("b-r0.1", 0.93, 50, 120.0),  # a-r0.01과 AUROC가 같다: 뱅크가 작은 쪽이 이긴다
        _row("b-r0.01", 0.90, 5, 60.0),
        _row("c-r0.1", 0.99, 10),  # CPU 지연 없음(예: 내보내기 실패): 후보가 아니다
    ]
    pick = analyze_grid.choose_serving(rows)
    assert pick["key"] == "b-r0.1" and pick["setting"] == "b" and pick["ratio"] == 0.1
    assert pick["within_budget"] is True and pick["budget_ms"] == 200.0 and pick["cpu_ms"] == 120.0
    assert pick["candidates"] == ["a-r0.01", "b-r0.1", "b-r0.01"] and pick["unmeasured"] == ["c-r0.1"]
    assert analyze_grid.choose_serving(list(reversed(rows)))["key"] == "b-r0.1"

    # 예산 안이 하나도 없다: 가장 빠른 구성을 고른다(이때 H15는 기각).
    pick = analyze_grid.choose_serving(rows, budget_ms=50.0)
    assert pick["key"] == "b-r0.01" and pick["within_budget"] is False and pick["candidates"] == []
    assert analyze_grid.choose_serving([_row("c-r0.1", 0.99, 10)]) is None


# 명령행 테스트용 작업 폴더: paths.OUTPUTS·REPORTS를 임시 폴더로 바꾸고, 등록 설정 두 개와 등록되지 않은
# 설정(wrn50-32) 하나의 dev 격자를 만든다.
@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(paths, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(compare, "N_BOOT", 50)
    write_grid(tmp_path / "outputs", "dinov2_vits14-392", [0.1, 0.01], 2)
    write_grid(tmp_path / "outputs", "wrn50-256", [0.1, 0.01], 1)
    write_grid(tmp_path / "outputs", "wrn50-32", [0.1], 3)  # 등록하지 않은 설정
    return tmp_path


# main이 쓴 reports/stage4/grid-<절차>.json을 읽는다.
def _report(world, protocol="dev"):
    with open(world / "reports" / "stage4" / f"grid-{protocol}.json", encoding="utf-8") as f:
        return json.load(f)


# --settings 없이 부르면 등록한 다섯 설정 가운데 끝난 실행만 등록 순서대로(폴더 순서가 아니라) 쓰고, 빠진
# 설정은 표준 오류로 알리며, 등록되지 않은 설정(wrn50-32)은 조용히 무시하는지 본다. 지연 파일이 붙고, CPU
# 지연이 있는 행이 하나뿐이고 예산을 넘으므로 그 행이 "예산 밖 가장 빠른 것"으로 골라져야 한다.
def test_cli_defaults_to_the_registered_settings_that_were_run(world, capsys):
    (world / "reports" / "stage4").mkdir(parents=True)
    with open(world / "reports" / "stage4" / "latency.json", "w", encoding="utf-8") as f:
        json.dump(LATENCY, f)
    analyze_grid.main(["--protocol", "dev"])
    out, err = capsys.readouterr()
    report = _report(world)
    keys = ["wrn50-256-r0.1", "wrn50-256-r0.01", "dinov2_vits14-392-r0.1", "dinov2_vits14-392-r0.01"]
    assert [r["key"] for r in report["rows"]] == keys  # 폴더 순서가 아니라 등록 순서
    assert report["protocol"] == "dev" and report["alpha"] == 0.05
    assert report["rows"][1]["cpu_ms"] == 211.0 and "cpu_ms" not in report["rows"][0]
    # CPU 지연이 있는 행은 하나뿐이고, 그 행은 예산을 넘는다.
    serving = report["serving"]
    assert serving["key"] == "wrn50-256-r0.01" and serving["within_budget"] is False
    assert serving["unmeasured"] == [k for k in keys if k != "wrn50-256-r0.01"]
    assert "CPU ms (resize + inference)" in out and "211.0" in out and "wrn50-256-r0.01" in out
    for skipped in ("wrn50-384", "dinov2_vits14-252", "dinov2_vits14-448"):
        assert skipped in err
    assert "wrn50-32" not in out + err


# --settings로 고른 설정만 표에 나오고, --no-write면 리포트 폴더를 만들지 않으며, 지연 파일이 없으면 서빙 선택
# 항목이 없어야 한다.
def test_cli_with_explicit_settings_and_without_latency(world, capsys):
    analyze_grid.main(["--protocol", "dev", "--settings", "wrn50-32", "--no-write"])
    out = capsys.readouterr().out
    assert out.count("| wrn50-32 |") == 1 and "wrn50-256" not in out
    assert not (world / "reports").exists()
    analyze_grid.main(["--protocol", "dev", "--settings", "wrn50-32"])
    report = _report(world)
    assert [r["key"] for r in report["rows"]] == ["wrn50-32-r0.1"] and "serving" not in report


# 끝난 실행이 하나도 없는 절차, --settings에 실행이 없는 설정이 섞인 경우, 지정한 지연 파일이 없는 경우는 모두
# 명령행 오류로 멈춰야 한다(빠진 수치로 조용히 표를 만들지 않게).
def test_cli_refuses_missing_runs_and_a_missing_latency_file(world):
    with pytest.raises(SystemExit):  # test 절차의 격자 실행이 하나도 없다
        analyze_grid.main(["--protocol", "test", "--no-write"])
    with pytest.raises(SystemExit):
        analyze_grid.main(["--protocol", "dev", "--settings", "wrn50-256", "wrn50-384", "--no-write"])
    with pytest.raises(SystemExit):
        analyze_grid.main(["--protocol", "dev", "--latency", str(world / "nowhere.json"), "--no-write"])


# test 절차의 표에는 지연이 붙더라도 서빙 구성을 고르지 않아야 한다. 봉인 테스트 수치로 구성을 고르지 않는다는
# 규칙을 코드가 지키는지 확인한다.
def test_cli_does_not_choose_a_serving_configuration_on_the_test_protocol(world, capsys):
    write_grid(world / "outputs", "wrn50-256", [0.1, 0.01], 1, protocol="test")
    latency = world / "latency.json"
    with open(latency, "w", encoding="utf-8") as f:
        json.dump(LATENCY, f)
    analyze_grid.main(["--protocol", "test", "--latency", str(latency)])
    report = _report(world, "test")
    assert report["rows"][1]["cpu_ms"] == 211.0 and "serving" not in report
    assert "serving" not in capsys.readouterr().out
