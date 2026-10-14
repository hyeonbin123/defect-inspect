# [파일 역할] 1·2단계 PatchCore 실행 스크립트 `defect_inspect.run_patchcore`를 검증하는 테스트 묶음이다. 범주
# 하나의 실행(run_category)이 분석에 필요한 점수 세트(전체 뱅크·홀드아웃 뱅크로 매긴 평가 점수, 재대입·교차
# 적합 보정 점수, 픽셀 지표, PRO 히스토그램, 점수 맵)를 모두 쓰는지, 그 결과를 analyze가 그대로 읽는지, 뱅크
# 저장과 봉인 관문이 맞는지 확인한다.
# [왜 필요한가] 1단계의 임계값 보정 비교(재대입·홀드아웃·교차 적합, docs/experiments.md 1단계)는 이 파일이
# 만드는 점수 세트에 기대고, test 절차는 봉인 테스트를 한 번만 읽는다. 그래서 VisA 데이터와 GPU 없이 가짜
# 매니페스트·캐시·백본으로 전체 경로를 먼저 돌려 본다.
# [설계 메모] 가짜 데이터는 정상 풀 40장(겹 0~4에 8장씩), dev 결함 6장, 봉인 테스트 정상 10장·결함 4장이다.
# 결함은 밝은 사각형과 그에 맞는 마스크라 점수와 픽셀 지표가 분명히 갈린다.

import json

import numpy as np
import pytest

# PatchCore 특징·뱅크가 torch를 쓰므로 torch가 없는 CI(.github/workflows/ci.yml)에서는 파일 전체를 건너뛴다.
torch = pytest.importorskip("torch")

# analyze는 실행 결과를 분석 모듈이 끝까지 읽는지 확인하려고, ManifestRow·SealedTestError는 가짜 매니페스트를
# 만들고 봉인 예외를 기다리려고 가져온다.
from defect_inspect import analyze, run_patchcore  # noqa: E402
from defect_inspect.configs import PatchCoreConfig  # noqa: E402
from defect_inspect.splits import ManifestRow, SealedTestError  # noqa: E402

# 가짜 이미지 크기 32px. GridMean이 4x4 격자로 줄여 이미지당 패치 16개가 된다.
SIZE = 32


# 실제 백본(WideResNet-50 등) 대신 쓰는 작은 특징 추출기: 4x4 칸마다 평균 색(3차원). 학습 가중치가 없어
# 결정적이고 CPU에서 빠르다.
class GridMean(torch.nn.Module):
    """백본 대신 쓰는 작은 추출기: 4x4 격자의 칸마다 평균색을 낸다."""

    name = "gridmean"
    dim = 3

    def forward(self, x):
        return torch.nn.functional.avg_pool2d(x.float(), SIZE // 4).permute(0, 2, 3, 1)


# ImageCache를 대신하는 가짜 캐시. 매니페스트 행마다 고정 시드 잡음 이미지를 만들고, 결함 행에는 이미지에 밝은
# 사각형을, 256x256 마스크에는 같은 위치(이미지 8~16px = 마스크 64~128px)를 표시한다. 픽셀 지표는 모든
# 구성에서 256 격자로 계산한다는 규칙(docs/experiments.md 전처리)에 맞춰 마스크를 256 크기로 둔다.
class FakeCache:
    size = SIZE

    def __init__(self, manifest):
        rng = np.random.default_rng(0)
        self._images, self._masks = {}, {}
        for row in manifest:
            img = rng.integers(100, 140, (SIZE, SIZE, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            if row.label == "anomaly":
                img[8:16, 8:16] = 255  # 어떤 정상 이미지에도 없는 밝은 사각형
                mask[64:128, 64:128] = 1
            self._images[row.image] = img
            self._masks[row.image] = mask

    def images(self, rows):
        return np.stack([self._images[r.image] for r in rows])

    def masks(self, rows):
        return np.stack([self._masks[r.image] for r in rows])


# 가짜 매니페스트: 범주 toy 하나에 정상 풀 40장(겹 = 순번 % 5), dev 결함 6장, 봉인 테스트 정상 10장과 결함
# 4장. 실제 매니페스트와 같은 역할 이름을 써서 splits.select가 그대로 동작하게 한다.
def _manifest():
    rows = []
    for i in range(40):
        rows.append(ManifestRow(f"toy/n{i}.JPG", "", "toy", "normal", "pool_normal", i % 5, ""))
    for i in range(6):
        rows.append(ManifestRow(f"toy/d{i}.JPG", f"toy/d{i}.png", "toy", "anomaly", "dev_defect", -1, "hole"))
    for i in range(10):
        rows.append(ManifestRow(f"toy/tn{i}.JPG", "", "toy", "normal", "test_normal", -1, ""))
    for i in range(4):
        mask = f"toy/td{i}.png"
        rows.append(ManifestRow(f"toy/td{i}.JPG", mask, "toy", "anomaly", "test_defect", -1, "hole"))
    return rows


# 테스트용 구성: 코어셋 50%(뱅크 행 수가 패치 수의 절반이 되는지 확인), 맵 σ 1, 배치 8.
CFG = PatchCoreConfig(
    name="toy", backbone="gridmean", img_size=SIZE, coreset_ratio=0.5, sigma=1.0, batch_size=8
)


# 가짜 데이터로 run_category를 CPU에서 한 번 돌리고 요약 dict를 돌려준다.
def _run(tmp_path, protocol, allow_test=False, save_bank=False):
    manifest = _manifest()
    return run_patchcore.run_category(
        CFG,
        protocol,
        "toy",
        manifest,
        FakeCache(manifest),
        GridMean(),
        "cpu",
        allow_test,
        tmp_path,
        save_bank,
    )


# dev 절차 실행이 모든 점수 세트를 쓰는지 확인한다. dev 정상 풀은 겹 1~4의 32장, 평가는 겹 0 정상 8장 + dev
# 결함 6장이다. 뱅크 행 수는 32장 × 패치 16 × 코어셋 0.5이고, 겹 뱅크는 풀의 네 겹마다 하나씩이다.
# 결함 점수가 두 모델(전체 뱅크, 홀드아웃 뱅크) 모두에서 정상보다 높고, 재대입 점수 평균이 교차 적합보다
# 낮다(자기 패치가 든 뱅크로 매기면 더 정상처럼 보인다는 1단계 H1의 바탕). 점수 맵은 256x256 fp16으로
# 저장된다.
def test_dev_run_writes_every_score_set(tmp_path):
    info = _run(tmp_path, "dev")
    assert (info["pool"], info["eval_normal"], info["eval_defect"]) == (32, 8, 6)
    assert info["grid"] == [4, 4] and info["dim"] == 3
    assert set(info["bank_rows"]) == {"full", "minus_fold_1", "minus_fold_2", "minus_fold_3", "minus_fold_4"}
    assert info["bank_rows"]["full"] == 32 * 16 // 2

    with np.load(tmp_path / "toy.npz") as z:
        assert z["eval_labels"].tolist() == [0] * 8 + [1] * 6
        assert set(z["pool_folds"].tolist()) == {1, 2, 3, 4}
        assert np.isfinite(z["pool_score_oof"]).all() and np.isfinite(z["pool_score_resub"]).all()
        # 결함(밝은 사각형)은 두 모델 모두에서 정상보다 훨씬 높게 채점된다.
        for key in ("eval_score_full", "eval_score_holdout"):
            assert z[key][8:].min() > z[key][:8].max()
        # 자기 패치가 든 뱅크로 채점한 이미지는 빼 두고 채점할 때보다 더 정상처럼 보인다.
        assert z["pool_score_resub"].mean() < z["pool_score_oof"].mean()
        assert float(z["pixel_auroc"]) > 0.9
        assert z["pro_components"].shape[0] == 6
    maps = np.load(tmp_path / "toy_maps.npy")
    assert maps.shape == (14, 256, 256) and maps.dtype == np.float16


# 실행 결과를 analyze.build_report가 끝까지 읽는지 확인한다. run.json을 손으로 만들고, 부트스트랩 횟수를
# 100으로 줄여(monkeypatch) 빠르게 돌린다. 홀드아웃 보정 장수는 dev 홀드아웃 겹(겹 1)의 8장, 교차 적합은 풀
# 전체 32장이어야 한다.
def test_dev_run_feeds_the_analysis(tmp_path, monkeypatch):
    info = _run(tmp_path, "dev")
    meta = {"config": {"name": "toy"}, "protocol": "dev", "commit": "test", "categories": [info]}
    (tmp_path / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr(analyze, "N_BOOT", 100)
    report = analyze.build_report(tmp_path)
    assert report["accuracy"]["macro_image_auroc"] == 1.0
    assert report["calibration"]["0.05"]["holdout"]["n_cal_total"] == 8
    assert report["calibration"]["0.05"]["crossfit"]["n_cal_total"] == 32


# --save-bank로 저장한 전체 뱅크(fp16, 코어셋 선택 순서)로 dev 결함을 다시 채점하면 실행이 기록한 점수와
# 같아야 한다. 3단계 교란 실행(run_perturb)이 이 저장 뱅크를 읽어 새 조건을 채점하므로, 저장한 뱅크가 실제로
# 쓴 뱅크와 같은지가 중요하다. 저장하는 뱅크는 전체 뱅크 하나뿐이고 겹 뱅크(minus_fold_*)는 남기지 않으므로,
# 실행 폴더의 .npy 파일은 이 범주의 점수 맵과 그 뱅크 둘이어야 한다.
def test_saved_bank_reproduces_the_scores(tmp_path):
    from defect_inspect.patchcore import score_images

    info = _run(tmp_path, "dev", save_bank=True)
    bank = np.load(tmp_path / "toy_bank.npy")
    assert bank.dtype == np.float16 and bank.shape == (info["bank_rows"]["full"], 3)
    manifest = _manifest()
    rows = [r for r in manifest if r.role == "dev_defect"]
    images = FakeCache(manifest).images(rows)
    res = score_images(
        GridMean(), torch.from_numpy(bank), images, batch_size=CFG.batch_size, device="cpu", sigma=CFG.sigma
    )
    with np.load(tmp_path / "toy.npz") as z:
        np.testing.assert_allclose(res.image_scores, z["eval_score_full"][8:], rtol=1e-5)
    # 전체 뱅크만 남기고 그 옆에 겹 뱅크는 쓰지 않는다.
    assert sorted(p.name for p in tmp_path.glob("*.npy")) == ["toy_bank.npy", "toy_maps.npy"]


# 뱅크 저장은 선택 사항이라 기본 실행에서는 뱅크 파일을 남기지 않는다.
def test_bank_is_not_saved_by_default(tmp_path):
    _run(tmp_path, "dev")
    assert not (tmp_path / "toy_bank.npy").exists()


# test 절차는 allow_test=True 없이는 SealedTestError로 멈추고, 허락하면 정상 풀 40장(겹 0 포함), 봉인 테스트
# 정상 10장·결함 4장으로 돈다.
def test_test_protocol_needs_permission(tmp_path):
    with pytest.raises(SealedTestError):
        _run(tmp_path, "test")
    info = _run(tmp_path, "test", allow_test=True)
    assert (info["pool"], info["eval_normal"], info["eval_defect"]) == (40, 10, 4)
    assert "minus_fold_0" in info["bank_rows"]


# CLI에서 --protocol test는 --allow-test와 --stage가 모두 있어야 한다. 하나라도 없으면 매니페스트를 읽거나
# 장부를 쓰기 전에 argparse 오류로 멈춘다.
def test_cli_refuses_test_protocol_without_flags(tmp_path):
    with pytest.raises(SystemExit):
        run_patchcore.main(["--protocol", "test", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        run_patchcore.main(["--protocol", "test", "--allow-test", "--out", str(tmp_path)])
