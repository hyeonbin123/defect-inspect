# [파일 역할] 4단계 격자 실행 스크립트 src/defect_inspect/run_grid.py를 검증한다. PatchCore 설정 하나를 코어셋
# 비율 여러 개로 채점하면서, 비싼 코어셋 선택은 가장 큰 비율로 한 번만 하고 작은 비율은 그 뱅크의 앞부분을
# 잘라 쓰는 구조다.
# [왜 필요한가] 질문 4(CPU 200ms 안에 남는 구성)의 격자는 다섯 설정 × 세 비율을 이 스크립트로 재고 서빙 구성을
# 고른다(docs/experiments.md 4단계). '앞부분 자르기'가 실제로 그 비율로 새로 고른 뱅크와 같아야 비율별 수치를
# 믿을 수 있고, 가장 큰 비율 결과는 1단계 run_patchcore와 같아야 단계 사이 수치가 이어진다. 또 test 절차는
# 봉인 테스트를 읽으므로, 장부 기록 순서와 '실패하면 기록하지 않음', 이전 출력 정리 규칙을 CLI 수준에서
# 확인한다.
# [방법] 실제 백본·데이터 없이 4×4 격자 평균색 추출기(GridMean)와 가짜 캐시·매니페스트로 CPU에서 돌린다.
# torch가 필요하므로 없으면 건너뛴다.

import json

import numpy as np
import pytest

# torch가 없는 CI에서는 파일 전체를 건너뛴다. 아래 import는 이 줄 뒤에 와야 해서 E402 예외 표시를 달았다.
# analyze_grid·compare·run_patchcore는 격자 출력이 그 도구들과 호환되는지 확인하려고 가져온다.
torch = pytest.importorskip("torch")

from defect_inspect import analyze_grid, compare, paths, run_grid, run_patchcore  # noqa: E402
from defect_inspect.configs import PatchCoreConfig  # noqa: E402
from defect_inspect.patchcore import build_bank, collect_features, score_images  # noqa: E402
from defect_inspect.splits import ManifestRow, SealedTestError  # noqa: E402

# 입력 32px, 4×4 격자라 이미지당 패치 16개. RATIOS는 실제 격자(0.1, 0.01, 0.001)보다 큰 비율로 잡아 작은
# 풀에서도 뱅크 행 수가 비율마다 달라지게 한다.
SIZE = 32
RATIOS = [0.5, 0.25, 0.1]


# 백본 대신 쓰는 작은 추출기: 8px 칸마다 평균색을 내 [B, 4, 4, 3] 특징을 만든다. name·dim 속성은 실제 추출기와
# 같은 형태다.
class GridMean(torch.nn.Module):
    """백본 대신 쓰는 작은 추출기: 4x4 격자의 칸마다 평균색을 낸다."""

    name = "gridmean"
    dim = 3

    def forward(self, x):
        return torch.nn.functional.avg_pool2d(x.float(), SIZE // 4).permute(0, 2, 3, 1)


# ImageCache 대신 쓰는 가짜 캐시. 매니페스트 행마다 회색 잡음 이미지를 만들고, 결함 행은 흰 사각형과 256 격자
# 마스크를 넣는다. images·masks가 불릴 때마다 읽은 행의 역할(pool_normal, test_normal 등)을 log에 남겨, CLI가
# 언제 어떤 이미지(특히 봉인 테스트)를 읽었는지 장부 기록과 순서를 비교할 수 있게 한다.
class FakeCache:
    size = SIZE

    def __init__(self, manifest, log=None):
        rng = np.random.default_rng(0)
        self._images, self._masks = {}, {}
        self._role = {row.image: row.role for row in manifest}
        self.log = [] if log is None else log  # 호출 순서대로 ("images" | "masks", 읽은 역할들)
        for row in manifest:
            img = rng.integers(100, 140, (SIZE, SIZE, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            if row.label == "anomaly":
                img[8:16, 8:16] = 255
                mask[64:128, 64:128] = 1
            self._images[row.image] = img
            self._masks[row.image] = mask

    def images(self, rows):
        self.log.append(("images", sorted({self._role[r.image] for r in rows})))
        return np.stack([self._images[r.image] for r in rows])

    def masks(self, rows):
        self.log.append(("masks", sorted({self._role[r.image] for r in rows})))
        return np.stack([self._masks[r.image] for r in rows])


# 범주마다 정상 풀 40장(겹 0~4에 8장씩), dev 결함 6장, test 정상 10장, test 결함 4장인 가짜 매니페스트. dev
# 절차에서는 겹 0 정상 8장 + dev 결함이 평가 세트, 겹 1~4의 32장이 뱅크 재료가 된다.
def _manifest(categories=("toy",)):
    rows = []
    for c in categories:
        for i in range(40):
            rows.append(ManifestRow(f"{c}/n{i}.JPG", "", c, "normal", "pool_normal", i % 5, ""))
        for i in range(6):
            rows.append(ManifestRow(f"{c}/d{i}.JPG", f"{c}/d{i}.png", c, "anomaly", "dev_defect", -1, "hole"))
        for i in range(10):
            rows.append(ManifestRow(f"{c}/tn{i}.JPG", "", c, "normal", "test_normal", -1, ""))
        for i in range(4):
            rows.append(
                ManifestRow(f"{c}/td{i}.JPG", f"{c}/td{i}.png", c, "anomaly", "test_defect", -1, "hole")
            )
    return rows


# 작은 입력에 맞춘 구성: 흐림 σ 1(32px 맵이라 기본 σ 4는 크다), 재가중 k 3, 배치 8.
def _cfg(ratio):
    return PatchCoreConfig(
        name="toy",
        backbone="gridmean",
        img_size=SIZE,
        coreset_ratio=ratio,
        sigma=1.0,
        batch_size=8,
        reweight_k=3,
    )


# 범주 toy 하나로 run_category를 돌리는 도우미. 구성의 coreset_ratio는 RATIOS의 최댓값이어야 한다.
def _run(tmp_path, protocol="dev", allow_test=False, save_banks=False):
    manifest = _manifest()
    return run_grid.run_category(
        _cfg(max(RATIOS)),
        RATIOS,
        protocol,
        "toy",
        manifest,
        FakeCache(manifest),
        GridMean(),
        "cpu",
        allow_test,
        tmp_path,
        save_banks=save_banks,
    )


# 비율 폴더 이름(r0.1 등)과 비율별 뱅크 행 수 계산을 본다. ratio_rows는 patchcore.build_bank와 같은
# 반올림(가장 가까운 정수, 최소 1행, 비율 1은 전부)이어야 '앞부분 자르기 = 새로 고르기'가 성립한다.
def test_ratio_names_and_rows():
    assert [run_grid.ratio_name(r) for r in (0.1, 0.01, 0.001, 1)] == ["r0.1", "r0.01", "r0.001", "r1.0"]
    assert run_grid.ratio_rows(512, 0.5) == 256
    assert run_grid.ratio_rows(96, 0.3) == 29  # build_bank처럼 28.8을 반올림
    assert run_grid.ratio_rows(94, 0.1) == 9
    assert run_grid.ratio_rows(4, 0.001) == 1
    assert run_grid.ratio_rows(7, 1.0) == 7


# 격자의 가장 큰 비율 결과가 같은 구성으로 1단계 실행 스크립트(run_patchcore.run_category)를 돌린 결과와
# 같은지 본다. 평가 이미지·라벨·점수, 교차 적합 점수, 겹, 픽셀 지표가 같아야 한다. 같은 출력 형식이라
# compare·analyze가 두 결과를 섞어 비교하므로, 계산이 달라지면 단계 사이 수치가 이어지지 않는다.
def test_largest_ratio_equals_run_patchcore(tmp_path):
    _run(tmp_path / "grid")
    manifest = _manifest()
    (tmp_path / "pc").mkdir()
    run_patchcore.run_category(
        _cfg(0.5), "dev", "toy", manifest, FakeCache(manifest), GridMean(), "cpu", False, tmp_path / "pc"
    )
    with np.load(tmp_path / "grid" / "r0.5" / "toy.npz") as g, np.load(tmp_path / "pc" / "toy.npz") as p:
        np.testing.assert_array_equal(g["eval_images"], p["eval_images"])
        np.testing.assert_array_equal(g["eval_labels"], p["eval_labels"])
        np.testing.assert_allclose(g["eval_score_full"], p["eval_score_full"], rtol=1e-6)
        np.testing.assert_allclose(g["pool_score_oof"], p["pool_score_oof"], rtol=1e-6)
        np.testing.assert_array_equal(g["pool_folds"], p["pool_folds"])
        assert float(g["aupro"]) == pytest.approx(float(p["aupro"]))
        assert float(g["pixel_auroc"]) == pytest.approx(float(p["pixel_auroc"]))


# 작은 비율의 점수가 그 비율로 처음부터 뱅크를 새로 만들어 채점한 것과 같은지 본다(docstring). 평가 세트
# 점수와 뱅크 행 수, 그리고 교차 적합 점수(겹 2 이미지를 '겹 2를 뺀 풀'로 새로 만든 뱅크로 채점)까지 맞춘다.
# 탐욕적 k-center 선택의 중첩 성질에 기대는 설계가 실제로 맞는지 확인하는 핵심 테스트다.
@pytest.mark.parametrize("ratio", RATIOS)
def test_smaller_ratios_equal_a_fresh_selection(tmp_path, ratio):
    """가장 큰 뱅크의 앞부분이 그 비율로 처음부터 만든 뱅크와 같은 점수를 낸다."""
    infos = _run(tmp_path)
    manifest = _manifest()
    cache, ext, cfg = FakeCache(manifest), GridMean(), _cfg(ratio)
    pool = [r for r in manifest if r.role == "pool_normal" and r.fold != 0]
    eval_rows = [r for r in manifest if r.role == "pool_normal" and r.fold == 0]
    eval_rows += [r for r in manifest if r.role == "dev_defect"]
    feats, _ = collect_features(ext, cache.images(pool), batch_size=8, device="cpu")
    bank = build_bank(feats, ratio, seed=0, device="cpu")
    assert infos[ratio]["bank_rows"]["full"] == bank.shape[0] == run_grid.ratio_rows(len(pool) * 16, ratio)
    direct = score_images(
        ext,
        bank,
        cache.images(eval_rows),
        batch_size=8,
        device="cpu",
        reweight_k=cfg.reweight_k,
        sigma=cfg.sigma,
    )
    with np.load(tmp_path / run_grid.ratio_name(ratio) / "toy.npz") as z:
        np.testing.assert_allclose(z["eval_score_full"], direct.image_scores, rtol=1e-6)
        assert int(z["bank_rows"]) == bank.shape[0]

        # 이 비율의 교차 적합: 겹 2 이미지를 겹 2를 빼고 새로 만든 뱅크로 채점한다.
        rest = [r for r in pool if r.fold != 2]
        held = [r for r in pool if r.fold == 2]
        feats, _ = collect_features(ext, cache.images(rest), batch_size=8, device="cpu")
        fold_bank = build_bank(feats, ratio, seed=0, device="cpu")
        oof = score_images(
            ext,
            fold_bank,
            cache.images(held),
            batch_size=8,
            device="cpu",
            reweight_k=cfg.reweight_k,
            sigma=cfg.sigma,
        )
        np.testing.assert_allclose(z["pool_score_oof"][z["pool_folds"] == 2], oof.image_scores, rtol=1e-6)


# 비율이 실제로 결과를 바꾸는지(모든 비율이 같은 뱅크를 쓰는 회귀가 없는지) 본다. 작은 뱅크는 정상 패치를 덜
# 덮으므로 정상의 교차 적합 점수 평균이 내려가지 않아야 한다. 평가 실행(run_dinomaly 등)이 남기는 점수 맵
# 파일(*_maps.npy)을 격자는 저장하지 않는다는 것도 확인한다.
def test_smaller_banks_change_the_scores(tmp_path):
    _run(tmp_path)
    with np.load(tmp_path / "r0.5" / "toy.npz") as a, np.load(tmp_path / "r0.1" / "toy.npz") as b:
        assert not np.allclose(a["eval_score_full"], b["eval_score_full"])
        assert int(a["bank_rows"]) > int(b["bank_rows"])
        # 작은 뱅크는 정상 패치를 덜 덮으므로 정상 점수가 내려가지 않는다.
        assert b["pool_score_oof"].mean() >= a["pool_score_oof"].mean()
    assert not list(tmp_path.glob("**/*_maps.npy"))


# --save-banks로 저장한 뱅크만으로 모든 비율의 점수를 다시 만들 수 있는지 본다. 4단계는 저장한 뱅크로 ONNX
# 아티팩트를 만들고(export) CPU 경로로 다시 채점하므로(run_export_eval) 필요하다. banks/<범주>.json의 원래
# 패치 수(전체 32장 × 16, 겹 하나를 뺀 24장 × 16)로 ratio_rows를 계산해 앞부분을 자르면, 평가 점수와 겹별 교차
# 적합 점수가 저장된 결과와 같아야 한다. 뱅크는 fp16이고 겹 1~4 뱅크 파일이 모두 있어야 한다.
def test_saved_banks_reproduce_every_ratio(tmp_path):
    _run(tmp_path, save_banks=True)
    manifest = _manifest()
    cache, ext, cfg = FakeCache(manifest), GridMean(), _cfg(0.5)
    with open(tmp_path / "banks" / "toy.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["largest_ratio"] == 0.5
    assert meta["n_features"]["full"] == 32 * 16 and meta["n_features"]["minus_fold_3"] == 24 * 16
    full = torch.from_numpy(np.load(tmp_path / "banks" / "toy_full.npy"))
    assert full.dtype == torch.float16
    eval_rows = [r for r in manifest if r.role == "pool_normal" and r.fold == 0]
    eval_rows += [r for r in manifest if r.role == "dev_defect"]
    for ratio in RATIOS:
        rows = run_grid.ratio_rows(meta["n_features"]["full"], ratio)
        res = score_images(
            ext,
            full[:rows],
            cache.images(eval_rows),
            batch_size=8,
            device="cpu",
            reweight_k=cfg.reweight_k,
            sigma=1.0,
        )
        with np.load(tmp_path / run_grid.ratio_name(ratio) / "toy.npz") as z:
            np.testing.assert_allclose(z["eval_score_full"], res.image_scores, rtol=1e-6)
    assert sorted(p.name for p in (tmp_path / "banks").glob("toy_minus_fold_*.npy")) == [
        f"toy_minus_fold_{f}.npy" for f in (1, 2, 3, 4)
    ]

    # 풀 이미지의 교차 적합 점수는 자기 겹을 빼고 만들어 저장한 뱅크에서 나온다.
    pool = [r for r in manifest if r.role == "pool_normal" and r.fold != 0]
    for fold in (1, 2, 3, 4):
        bank = torch.from_numpy(np.load(tmp_path / "banks" / f"toy_minus_fold_{fold}.npy"))
        n_source = meta["n_features"][f"minus_fold_{fold}"]
        assert bank.dtype == torch.float16 and bank.shape[0] == run_grid.ratio_rows(n_source, 0.5)
        held = cache.images([r for r in pool if r.fold == fold])
        for ratio in RATIOS:
            res = score_images(
                ext,
                bank[: run_grid.ratio_rows(n_source, ratio)],
                held,
                batch_size=8,
                device="cpu",
                reweight_k=cfg.reweight_k,
                sigma=1.0,
            )
            with np.load(tmp_path / run_grid.ratio_name(ratio) / "toy.npz") as z:
                oof = z["pool_score_oof"][z["pool_folds"] == fold]
                np.testing.assert_allclose(oof, res.image_scores, rtol=1e-6)


# test 절차는 allow_test 없이는 SealedTestError로 멈추고, 허용하면 정상 풀 전체(겹 0~4, 40장)를 뱅크에 쓰고
# 봉인 테스트(정상 10, 결함 4)로 평가하는지 본다.
def test_test_protocol_is_sealed(tmp_path):
    with pytest.raises(SealedTestError):
        _run(tmp_path, protocol="test")
    infos = _run(tmp_path, protocol="test", allow_test=True)
    assert infos[0.5]["pool"] == 40 and infos[0.5]["eval_normal"] == 10 and infos[0.5]["eval_defect"] == 4


# 구성의 coreset_ratio가 비율 목록의 최댓값이 아니면 거절한다. 더 작은 비율로 고른 뱅크에서 큰 비율의 앞부분을
# 잘라낼 수는 없기 때문이다.
def test_largest_ratio_must_be_the_config_ratio(tmp_path):
    manifest = _manifest()
    with pytest.raises(ValueError, match="largest ratio"):
        run_grid.run_category(
            _cfg(0.25),
            RATIOS,
            "dev",
            "toy",
            manifest,
            FakeCache(manifest),
            GridMean(),
            "cpu",
            False,
            tmp_path,
        )


# CLI 실행 중 일어난 장부 쓰기와 캐시 읽기를 순서대로 담는 목록. 장부가 이미지 읽기보다 먼저인지 비교할 때
# 쓴다.
@pytest.fixture
def calls():
    """CLI 실행의 장부 쓰기와 캐시 읽기를 일어난 순서대로 담는다."""
    return []


# CLI(main)를 데이터·GPU 없이 돌리기 위한 환경. 출력·리포트·장부 경로를 tmp_path로 바꾸고,
# 매니페스트·캐시·백본 생성을 가짜로 바꾸며, 장부 기록을 감싸 calls에 남긴다. 매니페스트에는 결함이 없는 범주
# bare를 더해 '평가할 것이 없는 범주' 거절을 시험한다. compare.N_BOOT를 100으로 줄여 analyze_grid의
# 부트스트랩을 빠르게 한다.
@pytest.fixture
def cli(tmp_path, monkeypatch, calls):
    manifest = _manifest(("toy", "two"))
    # 정상 풀은 있지만 평가할 결함이 없는 범주.
    manifest += [
        ManifestRow(f"bare/n{i}.JPG", "", "bare", "normal", "pool_normal", i % 5, "") for i in range(10)
    ]
    real_record = run_grid.record_test_access

    def record(*args, **kwargs):
        calls.append(("ledger", kwargs["config"]))
        return real_record(*args, **kwargs)

    monkeypatch.setattr(paths, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(paths, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(paths, "TEST_LEDGER", tmp_path / "reports" / "test_ledger.jsonl")
    monkeypatch.setattr(run_grid, "read_manifest", lambda path: manifest)
    monkeypatch.setattr(run_grid, "ImageCache", lambda out_dir, size: FakeCache(manifest, log=calls))
    monkeypatch.setattr(run_grid, "record_test_access", record)
    monkeypatch.setattr("defect_inspect.backbones.make_extractor", lambda name, img_size: GridMean())
    monkeypatch.setattr(compare, "N_BOOT", 100)
    return tmp_path


# CLI 공통 인자. 백본 이름은 wrn50이지만 make_extractor를 GridMean으로 바꿔 두었다. BASE는 두 범주·두 비율
# 실행이다.
SETTING = ["--backbone", "wrn50", "--size", "32", "--device", "cpu"]
BASE = [*SETTING, "--categories", "toy", "two", "--ratios", "0.5", "0.1"]


# CLI가 최상위 run.json(설정, 비율, 절차)과 비율마다 run_patchcore 형식의 폴더를 쓰고,
# compare.load_patchcore가 그 폴더를 읽는지 본다. 각 비율 폴더의 config는 가장 큰 비율이 아니라 자기 비율을
# 적어야 하고, 고정값(재가중 9, σ 4, 배치 32, seed 0 = P0 문헌 기본값)과 커밋·뱅크 행 수가 맞아야 한다. dev
# 실행은 장부를 쓰지 않는다.
# 이미 끝난 실행은 --overwrite 없이 다시 쓰지 않고, 마지막으로 analyze_grid가 이 출력으로
# 리포트(grid-dev.json, 가장 큰 비율 대비 차이 포함)를 만드는지 확인한다.
def test_cli_writes_a_run_per_ratio_that_compare_can_read(cli, capsys):
    run_grid.main([*BASE, "--protocol", "dev"])
    out = cli / "outputs" / "grid-wrn50-32-dev"
    with open(out / "run.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["setting"] == "wrn50-32" and meta["ratios"] == [0.5, 0.1] and meta["protocol"] == "dev"
    run = compare.load_patchcore(out / "r0.1")
    assert run.name == "wrn50-32-r0.1" and list(run.cats) == ["toy", "two"]
    assert not (cli / "reports" / "test_ledger.jsonl").exists()

    for ratio in (0.5, 0.1):  # 비율 폴더마다 가장 큰 비율이 아니라 자기 비율을 적는다
        folder = out / run_grid.ratio_name(ratio)
        with open(folder / "run.json", encoding="utf-8") as f:
            ratio_meta = json.load(f)
        config = ratio_meta["config"]
        assert config["name"] == f"wrn50-32-r{ratio}" and config["coreset_ratio"] == ratio
        assert config["backbone"] == "wrn50" and config["img_size"] == 32
        fixed = {"reweight_k": 9, "sigma": 4.0, "batch_size": 32, "seed": 0}
        assert {k: config[k] for k in fixed} == fixed
        assert ratio_meta["protocol"] == "dev" and ratio_meta["device"] == "cpu"
        assert ratio_meta["commit"] == meta["commit"]
        assert [info["category"] for info in ratio_meta["categories"]] == ["toy", "two"]
        for info in ratio_meta["categories"]:
            with np.load(folder / f"{info['category']}.npz") as z:
                assert info["bank_rows"]["full"] == int(z["bank_rows"]) == run_grid.ratio_rows(32 * 16, ratio)

    with pytest.raises(SystemExit):  # 끝난 실행은 조용히 덮어쓰지 않는다
        run_grid.main([*BASE, "--protocol", "dev"])
    run_grid.main([*BASE, "--protocol", "dev", "--overwrite"])

    capsys.readouterr()
    analyze_grid.main(["--protocol", "dev", "--settings", "wrn50-32"])
    text = capsys.readouterr().out
    assert "wrn50-32" in text and "vs largest" in text
    with open(cli / "reports" / "stage4" / "grid-dev.json", encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    assert [r["key"] for r in rows] == ["wrn50-32-r0.5", "wrn50-32-r0.1"]
    assert "vs_largest_ratio" not in rows[0] and "vs_largest_ratio" in rows[1]
    assert rows[0]["bank_rows_mean"] == 32 * 16 * 0.5


# test 절차는 --allow-test와 --stage가 둘 다 있어야 하고, 하나라도 빠지면 아무것도 읽지 않고 장부도 쓰지
# 않는다. 둘 다 있으면 장부에 정확히 한 줄(stage 4, config grid-wrn50-32)이 생기고, 그 줄이 어떤 이미지
# 읽기보다 먼저여야 한다(calls의 첫 항목). 이미 끝난 test 실행은 --overwrite 없이 다시 읽지 않아 장부 줄도
# 늘지 않는다. 봉인 테스트를 읽을 때마다 기록한다는 규칙(docs/experiments.md 봉인 테스트)을 CLI에서 확인한다.
def test_cli_test_protocol_needs_flags_and_writes_the_ledger(cli, calls):
    ledger = cli / "reports" / "test_ledger.jsonl"
    for extra in ([], ["--allow-test"], ["--stage", "4"]):
        with pytest.raises(SystemExit):
            run_grid.main([*BASE, "--protocol", "test", *extra])
    assert not ledger.exists() and not calls
    run_grid.main([*BASE, "--protocol", "test", "--allow-test", "--stage", "4"])
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["stage"] == "4" and entry["config"] == "grid-wrn50-32"
    # 장부 줄은 어떤 이미지든 처음 읽기 전에 이미 있고, 그 한 줄뿐이다.
    assert calls[0] == ("ledger", "grid-wrn50-32") and [c for c in calls if c[0] == "ledger"] == calls[:1]
    assert ("images", ["test_defect", "test_normal"]) in calls[1:]

    calls.clear()
    with pytest.raises(SystemExit):  # 끝난 test 실행은 --overwrite 없이 다시 읽지 않는다
        run_grid.main([*BASE, "--protocol", "test", "--allow-test", "--stage", "4"])
    assert not calls and len(ledger.read_text(encoding="utf-8").splitlines()) == 1


# dev 절차가 읽는 이미지 역할이 정상 풀과 dev 결함뿐인지(test_normal·test_defect를 읽지 않는지) 캐시 읽기
# 기록으로 확인한다.
def test_cli_dev_protocol_reads_no_sealed_image(cli, calls):
    run_grid.main([*BASE, "--protocol", "dev"])
    assert {role for _, roles in calls for role in roles} == {"pool_normal", "dev_defect"}
    assert not (cli / "reports" / "test_ledger.jsonl").exists()


# 범주 인자 오류(오타, 평가할 결함이 없는 범주, 같은 범주 두 번)를 장부 기록과 출력 폴더 생성 전에 잡는지
# test·dev 두 절차에서 본다. 실패한 실행이 '봉인 테스트를 읽었다'는 기록만 남기면 장부가 틀리기 때문이다. 오류
# 메시지에 문제 이름이 들어 있어야 한다.
@pytest.mark.parametrize(
    ("categories", "named"),
    [
        (["toy", "tyo"], "tyo"),  # 오타
        (["bare"], "eval_defect"),  # 정상 풀은 있지만 평가할 것이 없음
        (["toy", "toy"], "twice"),
    ],
)
def test_cli_checks_the_categories_before_the_ledger(cli, calls, capsys, categories, named):
    args = [*SETTING, "--ratios", "0.5", "--categories", *categories]
    for protocol in (["--protocol", "test", "--allow-test", "--stage", "4"], ["--protocol", "dev"]):
        with pytest.raises(SystemExit):
            run_grid.main([*args, *protocol])
        assert named in capsys.readouterr().err
    assert not calls and not (cli / "reports" / "test_ledger.jsonl").exists()
    assert not (cli / "outputs").exists()


# 이전 출력을 지우다 실패하면(파일이 다른 곳에서 열려 있음을 PermissionError로 흉내 냄) 장부 줄 없이 멈추는지
# 본다. main은 clear_outputs를 장부 기록 전에 부른다.
def test_cli_clears_old_outputs_before_the_ledger(cli, calls, monkeypatch):
    """지울 수 없는 이전 출력(사용 중인 파일)은 장부 줄 없이 실행을 멈춘다."""

    def in_use(out_dir):
        raise PermissionError(f"{out_dir} is in use")

    monkeypatch.setattr(run_grid, "clear_outputs", in_use)
    with pytest.raises(PermissionError):
        run_grid.main([*BASE, "--protocol", "test", "--allow-test", "--stage", "4"])
    assert not calls and not (cli / "reports" / "test_ledger.jsonl").exists()


# 폴더 아래 모든 파일의 상대 경로 목록(정렬, / 구분). 실행 뒤 남은 파일을 정확히 비교하는 데 쓴다.
def _files(folder):
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


# --overwrite가 이전 실행이 쓴 것(최상위·비율 run.json, 범주 npz, 저장한 뱅크)만 지우고 다시 쓰는지 본다. 이전
# 실행에만 있던 비율(r0.25)이나 범주(two)가 남으면 compare.load_patchcore가 완성된 결과처럼 읽기 때문이다.
# 반대로 이 모듈이 쓰지 않은 파일(notes.txt, reports/ 아래 npz, r로 시작하지만 비율 폴더가 아닌 r0.5x)은
# 남겨야 한다.
def test_cli_overwrite_removes_what_the_earlier_call_wrote(cli):
    out = cli / "outputs" / "grid-wrn50-32-dev"
    first = [*SETTING, "--protocol", "dev", "--categories", "toy", "two", "--ratios", "0.5", "0.25", "0.1"]
    run_grid.main([*first, "--save-banks"])
    assert (out / "r0.25" / "two.npz").exists() and (out / "banks" / "two_minus_fold_1.npy").exists()
    # 격자 출력이 아닌 파일은 남는다. 정리되는 폴더 안이나 이름이 "r"로 시작할 뿐인 폴더 안에서도 그렇다.
    kept = ["notes.txt", "r0.5/notes.txt", "banks/notes.txt", "reports/toy.npz", "r0.5x/run.json"]
    for name in kept:
        (out / name).parent.mkdir(exist_ok=True)
        (out / name).write_text("keep", encoding="utf-8")

    second = [*SETTING, "--protocol", "dev", "--categories", "toy", "--ratios", "0.5", "0.1", "--overwrite"]
    run_grid.main(second)
    written = ["run.json", "r0.5/run.json", "r0.5/toy.npz", "r0.1/run.json", "r0.1/toy.npz"]
    assert _files(out) == sorted(written + kept)
    assert not (out / "r0.25").exists()  # 낡은 비율 폴더가 더는 끝난 실행으로 읽히지 않는다
    assert list(compare.load_patchcore(out / "r0.5").cats) == ["toy"]


# --overwrite 실행이 두 번째 범주에서 끊기면(KeyboardInterrupt) 어느 수준에도 run.json이 남지 않아,
# analyze_grid.load_grid와 compare.load_patchcore가 이 폴더를 완성된 실행으로 읽지 않는지 본다. main은 시작할
# 때 최상위 run.json부터 지우고, 끝날 때 맨 마지막에 쓴다. 다음 호출은 완성된 실행이 없으므로 --overwrite 없이
# 깨끗한 폴더에서 시작해야 한다.
def test_interrupted_overwrite_does_not_leave_a_finished_looking_run(cli, monkeypatch):
    out = cli / "outputs" / "grid-wrn50-32-dev"
    run_grid.main([*BASE, "--protocol", "dev"])
    real, seen = run_grid.run_category, []

    def interrupted(*args, **kwargs):
        seen.append(args[3])
        if len(seen) == 2:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(run_grid, "run_category", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_grid.main([*BASE, "--protocol", "dev", "--overwrite"])
    assert seen == ["toy", "two"]
    assert _files(out) == ["r0.1/toy.npz", "r0.5/toy.npz"]  # 어느 수준에도 run.json이 없다
    with pytest.raises(FileNotFoundError):
        analyze_grid.load_grid(out)
    with pytest.raises(FileNotFoundError):
        compare.load_patchcore(out / "r0.5")

    # 다음 호출은 깨끗한 폴더에서 시작한다. 끝난 실행이 없으므로 --overwrite가 필요 없다.
    monkeypatch.setattr(run_grid, "run_category", real)
    run_grid.main([*SETTING, "--protocol", "dev", "--categories", "two", "--ratios", "0.5"])
    assert _files(out) == ["r0.5/run.json", "r0.5/two.npz", "run.json"]


# DINOv2는 패치 크기 14의 배수 입력만 받으므로 256을 거절하고, 비율은 (0, 1] 밖(0, 1.5, 음수, nan)을
# 거절하는지 본다. 거절한 경우 출력 폴더를 만들지 않는다.
def test_cli_rejects_bad_arguments(cli, capsys):
    with pytest.raises(SystemExit):
        run_grid.main(["--backbone", "dinov2_vits14", "--size", "256", "--protocol", "dev", *BASE[6:]])
    assert "multiples of 14" in capsys.readouterr().err
    for ratio in ("0", "1.5", "-0.1", "nan"):
        with pytest.raises(SystemExit):
            run_grid.main([*SETTING, "--categories", "toy", "--protocol", "dev", "--ratios", ratio])
        assert "ratios must be in (0, 1]" in capsys.readouterr().err
    assert not (cli / "outputs").exists()
