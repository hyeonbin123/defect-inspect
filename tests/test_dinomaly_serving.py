# [파일 역할] Dinomaly 모델의 CPU 서빙 파이프라인(`defect_inspect.dinomaly_serving`)의 세 명령 export(ONNX
# 내보내기와 아티팩트 세트 쓰기), score(onnxruntime CPU 채점), calibrate(서비스용 홀드아웃 임계값)를 검증하는
# 테스트 묶음이다.
# [왜 필요한가] 6단계 서빙 모델(dms-280-car)은 torch에서 학습·평가하고 onnxruntime CPU로 서빙한다. (1) 내보낸
# ONNX가 torch 그래프와 같은 점수·맵을 내는지, (2) score가 봉인 테스트를 --allow-test와 --stage가 있을 때만,
# 장부(ledger)에 먼저 기록한 뒤 읽는지(다시 채점할 때 장부 기록이 실패하면 예전 결과를 지우지 않는지 포함),
# (3) 합성 교란 조건이 평가 이미지에만 걸리고 임계값용 겹 0 정상에는 걸리지 않는지, (4) calibrate가
# '교란 없는 test 절차 채점'의 겹 0 점수로만 임계값을 정하는지를 확인한다.
# [설계] 무작위 가중치 28px 모델을 한 번 내보내 모든 테스트가 같이 쓰고, 매니페스트·캐시·장부·출력 경로는
# tmp_path와 monkeypatch로 가짜를 꽂아 실제 데이터와 reports/test_ledger.jsonl을 건드리지 않는다.

import json

import numpy as np
import pytest

# export에는 torch·anomalib·timm·onnx가, 채점에는 onnxruntime이 필요하다. 하나라도 없으면 파일 전체를
# 건너뛴다.
torch = pytest.importorskip("torch")
pytest.importorskip("anomalib")
pytest.importorskip("timm")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from defect_inspect import dinomaly_serving, run_dinomaly  # noqa: E402
from defect_inspect.calibrate import conformal_threshold  # noqa: E402
from defect_inspect.conditions import apply_condition  # noqa: E402
from defect_inspect.dinomaly_model import ServingGraph  # noqa: E402
from defect_inspect.download import sha256_file  # noqa: E402
from defect_inspect.inspector import ReconstructionInspector, preprocess  # noqa: E402
from defect_inspect.splits import SEALED_ROLES, ManifestRow, write_manifest  # noqa: E402

# 28px = 14px 패치 2×2 격자. 내보내기와 채점을 CPU에서 빠르게 돌리려는 최소 크기다.
SIZE = 28


# 6단계 구성과 같은 옵션(입력 크기 고정, 인코더 10블록, scaled 어텐션)을 쓴다. 내보내기와 INT8 양자화가
# 느려서 module 범위로 파일에서 한 번만 내보낸다.
# - timm.create_model을 pretrained=False로 바꿔 내려받기를 막고, anomalib가 바꿔 두는 resample_abs_pos_embed를
#   context가 끝날 때 원래대로 되돌린다.
# - git_commit을 고정값으로 바꿔 기록되는 commit을 확인할 수 있게 한다.
# - 디코더 가중치를 흔들어 초기값(출력이 거의 0)에서만 맞는 우연을 피한다.
# - 범주 두 개(toy, other)와 동적 INT8 모델까지 쓴다.
@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """디코더를 흔든 무작위 가중치 ViT-S Dinomaly를 이 모듈에서 한 번만 내보내는 fixture."""
    import timm
    import timm.models.vision_transformer as vit

    with pytest.MonkeyPatch.context() as mp:
        real_create = timm.create_model

        def create_without_download(name, *args, **kwargs):
            kwargs["pretrained"] = False
            return real_create(name, *args, **kwargs)

        mp.setattr(timm, "create_model", create_without_download)
        mp.setattr(vit, "resample_abs_pos_embed", vit.resample_abs_pos_embed)
        mp.setattr(dinomaly_serving, "git_commit", lambda: "abc1234")
        torch.manual_seed(0)
        options = {"img_size": SIZE, "encoder_blocks": 10, "attention": "scaled"}
        model = run_dinomaly.build_model(run_dinomaly.ENCODER_S, **options).eval()
        with torch.no_grad():
            for p in model.decoder.parameters():
                p.add_(torch.randn(p.shape, generator=torch.Generator().manual_seed(1)) * 0.05)
        out = tmp_path_factory.mktemp("art") / "dms-toy"
        info = dinomaly_serving.build_artifacts(
            model,
            name="dms-toy",
            img_size=SIZE,
            out_dir=out,
            source={"config": "dms-toy", "untrained": True},
            categories=["toy", "other"],
            int8_dynamic=True,
        )
        yield {"model": model, "out": out, "info": info}


# seed로 고정한 무작위 uint8 이미지 [n, 28, 28, 3].
def _images(n, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (n, SIZE, SIZE, 3), dtype=np.uint8)


# 확인: export가 FP32·동적 INT8 ONNX, artifacts.json, 범주별 meta.json(kind=reconstruction, 256 맵, 아직 보정
# 안 된 임계값 0과 strategy none, ONNX sha256)을 쓰고, 내보낼 때 잰 torch-onnxruntime 차이가 허용치(1e-3)
# 안이다. 이어서 서비스가 쓰는 ReconstructionInspector로 읽은 점수·맵이 torch ServingGraph와 같고, INT8 모델도
# 유한한 점수를 낸다.
# 왜: 6단계 관문 1 규칙이 내보낼 때 점수 상대 1e-3, 맵 절대 1e-3 이하를 요구한다. meta의 ONNX sha256은 나중에
# calibrate가 '같은 모델로 채점한 점수인지' 확인할 때 쓴다.
def test_export_writes_an_artifact_set_that_matches_torch(exported):
    out, info = exported["out"], exported["info"]
    assert (out / "model_fp32.onnx").exists() and (out / "model_int8_dynamic.onnx").exists()
    assert info["parity"]["score_rel_diff"] <= 1e-3 and info["parity"]["map_abs_diff"] <= 1e-3
    assert info["int8_dynamic"]["available"] and info["int8_dynamic"]["op_types"] == ["MatMul"]
    record = json.loads((out / "artifacts.json").read_text(encoding="utf-8"))
    assert record["name"] == "dms-toy" and record["img_size"] == SIZE and record["commit"] == "abc1234"
    meta = json.loads((out / "toy" / "meta.json").read_text(encoding="utf-8"))
    assert meta["kind"] == "reconstruction" and meta["map_size"] == 256 and meta["img_size"] == SIZE
    assert meta["threshold"] == 0.0 and meta["calibration"] == {"strategy": "none"}
    assert len(meta["source"]["onnx_sha256"]) == 64 and meta["source"]["untrained"] is True
    # 두 모델 파일의 지문이 meta에 따로 적히고, 서비스가 시작할 때 대조하는 값과 실제 파일 해시가 같다.
    assert meta["source"]["onnx_sha256"] == sha256_file(out / "model_fp32.onnx")
    assert meta["source"]["onnx_int8_dynamic_sha256"] == sha256_file(out / "model_int8_dynamic.onnx")

    graph = ServingGraph(exported["model"]).eval()
    insp = ReconstructionInspector.load(out / "toy")
    for image in _images(3, seed=4):
        score, amap = insp.run(image)
        with torch.no_grad():
            ref_score, ref_map = graph(torch.from_numpy(preprocess(image, SIZE)))
        assert score == pytest.approx(float(ref_score[0]), rel=1e-4)
        assert amap.shape == (256, 256) and float(np.abs(amap - ref_map[0].numpy()).max()) < 1e-4
    quantized = ReconstructionInspector.load(out / "toy", precision="int8-dynamic")
    scores = quantized.scores(_images(3, seed=4))
    assert quantized.precision == "int8-dynamic" and np.isfinite(scores).all()


# 리사이즈 캐시(ImageCache) 대신 쓰는 가짜. 매니페스트 행마다 무작위 이미지를 만들고(결함 행에는 흰 사각형을
# 넣어 점수가 달라지게 함), 봉인 테스트 역할(SEALED_ROLES)의 이미지를 요청받으면 장부 파일에 이미 줄이 있는지
# 단언한다.
# 왜: '봉인 테스트 이미지는 장부에 기록한 뒤에만 읽는다'는 규칙(docs/experiments.md 봉인 테스트)을 실제 순서로
# 검사하려는 것이다. requested에는 요청된 이미지 경로가 쌓여 dev 실행이 테스트 이미지를 건드리지 않았는지 볼
# 수 있다.
class FakeCache:
    """매니페스트 행별 이미지를 내주되, 봉인 이미지는 장부에 줄이 생긴 뒤에만 내주는지 단언한다."""

    def __init__(self, manifest, ledger):
        rng = np.random.default_rng(0)
        self._images = {}
        self._role = {}
        self._ledger = ledger
        self.requested = []
        for row in manifest:
            img = rng.integers(100, 140, (SIZE, SIZE, 3), dtype=np.uint8)
            if row.label == "anomaly":
                img[4:12, 4:12] = 255
            self._images[row.image] = img
            self._role[row.image] = row.role

    def images(self, rows):
        if any(self._role[r.image] in SEALED_ROLES for r in rows):
            assert self._ledger.exists() and self._ledger.read_text(encoding="utf-8").strip()
        self.requested += [r.image for r in rows]
        return np.stack([self._images[r.image] for r in rows])


# 범주 두 개의 작은 매니페스트. 범주마다 학습 풀 정상 20장(겹 = i % 5이므로 겹 0은 n0·n5·n10·n15), dev 결함
# 3장, 봉인 테스트 정상 5장·결함 2장이다.
def _manifest():
    rows = []
    for category in ("toy", "other"):
        for i in range(20):
            rows.append(ManifestRow(f"{category}/n{i}.JPG", "", category, "normal", "pool_normal", i % 5, ""))
        for i in range(3):
            rows.append(
                ManifestRow(f"{category}/d{i}.JPG", "m.png", category, "anomaly", "dev_defect", -1, "hole")
            )
        for i in range(5):
            rows.append(ManifestRow(f"{category}/tn{i}.JPG", "", category, "normal", "test_normal", -1, ""))
        for i in range(2):
            rows.append(
                ManifestRow(f"{category}/td{i}.JPG", "m.png", category, "anomaly", "test_defect", -1, "hole")
            )
    return rows


# 프로젝트 경로를 tmp_path로 돌리는 fixture. 매니페스트 CSV, 장부, outputs 경로, ImageCache, git_commit을
# monkeypatch로 바꿔 실제 저장소 파일을 건드리지 않는다. sizes에는 캐시를 연 입력 크기가 기록된다.
@pytest.fixture
def project(tmp_path, monkeypatch):
    manifest = _manifest()
    ledger = tmp_path / "ledger.jsonl"
    cache = FakeCache(manifest, ledger)
    sizes = []

    def fake_cache(root, size):
        sizes.append(size)
        return cache

    write_manifest(manifest, tmp_path / "visa.csv")
    monkeypatch.setattr(dinomaly_serving.paths, "VISA_MANIFEST", tmp_path / "visa.csv")
    monkeypatch.setattr(dinomaly_serving.paths, "TEST_LEDGER", ledger)
    monkeypatch.setattr(dinomaly_serving.paths, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(dinomaly_serving, "ImageCache", fake_cache)
    monkeypatch.setattr(dinomaly_serving, "git_commit", lambda: "abc1234")
    return {"cache": cache, "ledger": ledger, "outputs": tmp_path / "outputs", "sizes": sizes}


# 확인(흐름 전체): dev 채점 → test 채점의 플래그 검사와 장부 → 교란 조건 → 임계값 보정을 한 번에 따라간다.
# - dev: 장부를 쓰지 않고, 평가 세트가 겹 0 정상 + dev 결함이며, 보정 점수는 비어 있고(dev에서는 겹 0 정상이
#   곧 평가 정상이라 따로 모으지 않는다), 점수가 서비스 inspector 점수와 같고, 테스트 이미지(파일 이름이 t로
#   시작)를 한 장도 요청하지 않는다.
# - test: --stage 없이 --allow-test만 주면 멈추고 장부도 생기지 않는다. 둘 다 주면 장부에 stage·config 한 줄을
#   남기고, 보정 점수(cal)는 겹 0 정상, 평가는 봉인 테스트다.
# - 교란: brightness-3을 주면 평가 점수만 바뀌고 보정 점수는 교란 없는 실행과 같다. 모르는 조건은 거부한다.
# - calibrate: dev 실행과 교란 실행으로는 임계값을 만들 수 없고, 교란 없는 test 실행에서만 겹 0 점수로
#   conformal 임계값(목표 5%)을 meta에 쓴다. 겹 0이 4장뿐이라 (n+1)(1−α) 순위가 n을 넘어 최댓값을 쓰고
#   guaranteed가 False다.
# 왜: 서비스 임계값은 학습에 쓰지 않은 정상(겹 0)으로 정한다는 6단계 규칙(README 검사 서비스 절)이 코드로
# 강제되는지 본다.
def test_score_cli_dev_test_conditions_and_calibrate(exported, project, tmp_path):
    # module fixture의 아티팩트를 복사해 쓰므로 calibrate가 meta.json을 고쳐도 다른 테스트에 영향이 없다.
    import shutil

    art = tmp_path / "dms-toy"
    shutil.copytree(exported["out"], art)
    base = ["score", "--artifacts", str(art), "--categories", "toy", "other"]
    dinomaly_serving.main(base + ["--protocol", "dev"])
    dev_dir = project["outputs"] / "dms-toy-onnx-fp32-dev"
    run = json.loads((dev_dir / "run.json").read_text(encoding="utf-8"))
    assert run["protocol"] == "dev" and run["config"]["condition"] == "clean" and run["commit"] == "abc1234"
    assert project["sizes"] == [SIZE] and not project["ledger"].exists()
    insp = ReconstructionInspector.load(art / "toy")
    with np.load(dev_dir / "toy.npz") as z:
        assert z["eval_images"].tolist() == [f"toy/n{i}.JPG" for i in range(0, 20, 5)] + [
            f"toy/d{i}.JPG" for i in range(3)
        ]
        assert z["eval_labels"].tolist() == [0] * 4 + [1] * 3 and z["cal_score"].shape == (0,)
        rows = [r for r in _manifest() if r.image in set(z["eval_images"].tolist())]
        expected = insp.scores(project["cache"].images(rows))
        np.testing.assert_array_equal(z["eval_score"], expected)
    assert not any("/t" in image for image in project["cache"].requested)

    with pytest.raises(SystemExit):  # 봉인 테스트는 두 플래그가 모두 있어야 한다
        dinomaly_serving.main(base + ["--protocol", "test", "--allow-test"])
    assert not project["ledger"].exists()
    dinomaly_serving.main(base + ["--protocol", "test", "--allow-test", "--stage", "6"])
    line = json.loads(project["ledger"].read_text(encoding="utf-8").splitlines()[-1])
    assert line["stage"] == "6" and line["config"] == "dms-toy-onnx-fp32"
    test_dir = project["outputs"] / "dms-toy-onnx-fp32-test"
    with np.load(test_dir / "toy.npz") as z:
        assert z["cal_images"].tolist() == [f"toy/n{i}.JPG" for i in range(0, 20, 5)]  # 겹 0
        assert z["eval_images"].tolist()[:5] == [f"toy/tn{i}.JPG" for i in range(5)]
        cal_score = z["cal_score"]
        clean_eval = z["eval_score"]

    # 교란 조건은 평가 이미지만 바꾸고, 임계값용 보정 점수는 교란 없는 정상의 점수 그대로다.
    dinomaly_serving.main(
        base + ["--protocol", "test", "--allow-test", "--stage", "6", "--condition", "brightness-3"]
    )
    with np.load(project["outputs"] / "dms-toy-onnx-fp32-test-brightness-3" / "toy.npz") as z:
        np.testing.assert_array_equal(z["cal_score"], cal_score)
        assert not np.array_equal(z["eval_score"], clean_eval)
        test_rows = [r for r in _manifest() if r.category == "toy" and r.role in SEALED_ROLES]
        blurred = apply_condition(project["cache"].images(test_rows), "brightness-3")
        np.testing.assert_array_equal(z["eval_score"], insp.scores(blurred))
    with pytest.raises(SystemExit):
        dinomaly_serving.main(base + ["--protocol", "dev", "--condition", "fog-1"])

    # 서비스용 임계값: 교란 없는 test 실행의 겹 0 정상 점수로만 홀드아웃 보정한다.
    for run_dir in (dev_dir, project["outputs"] / "dms-toy-onnx-fp32-test-brightness-3"):
        with pytest.raises(SystemExit):
            dinomaly_serving.main(["calibrate", "--artifacts", str(art), "--scores", str(run_dir)])
    dinomaly_serving.main(["calibrate", "--artifacts", str(art), "--scores", str(test_dir)])
    meta = json.loads((art / "toy" / "meta.json").read_text(encoding="utf-8"))
    thr = conformal_threshold(cal_score, 0.05)
    assert meta["threshold"] == float(thr.value) and meta["alpha"] == 0.05
    assert meta["calibration"] == {"strategy": "holdout", "n": 4, "guaranteed": False}
    assert ReconstructionInspector.load(art / "toy").threshold == meta["threshold"]


# 봉인 테스트 채점을 --overwrite로 다시 할 때도 장부에 먼저 기록하고 그 뒤에 예전 결과를 지운다(run_m2ad·
# run_dinomaly eval과 같은 순서). 장부 기록이 실패하면(다른 실행이 잠금을 쥐고 있어 10초 안에 얻지 못한
# TimeoutError) 재실행은 봉인 이미지를 한 장도 읽지 않고 멈추고, 완성된 예전 실행의 run.json·npz가 바이트
# 그대로 남는다. calibrate가 서비스 임계값을 정할 때 읽는 것이 이 파일들이라, 장부 기록 실패 한 번으로 그
# 근거가 사라지면 안 된다. 장부에 기록되면 예전 결과(다른 범주의 gone.npz 포함)는 채점 전에 지워진다.
def test_score_overwrite_keeps_the_earlier_test_run_when_the_ledger_fails(
    exported, project, tmp_path, monkeypatch
):
    import shutil

    art = tmp_path / "dms-toy"
    shutil.copytree(exported["out"], art)
    base = ["score", "--artifacts", str(art), "--categories", "toy", "other"]
    base += ["--protocol", "test", "--allow-test", "--stage", "6"]
    dinomaly_serving.main(base)
    test_dir = project["outputs"] / "dms-toy-onnx-fp32-test"
    finished = {p.name: p.read_bytes() for p in test_dir.iterdir()}
    assert sorted(finished) == ["other.npz", "run.json", "toy.npz"]
    ledger = project["ledger"].read_bytes()
    real = dinomaly_serving.record_test_access

    def locked(*args, **kwargs):
        raise TimeoutError("could not take the ledger lock within 10 s: another run is writing the ledger")

    monkeypatch.setattr(dinomaly_serving, "record_test_access", locked)
    project["cache"].requested.clear()
    with pytest.raises(TimeoutError):
        dinomaly_serving.main(base + ["--overwrite"])
    assert {p.name: p.read_bytes() for p in test_dir.iterdir()} == finished
    assert project["ledger"].read_bytes() == ledger and project["cache"].requested == []

    monkeypatch.setattr(dinomaly_serving, "record_test_access", real)
    shutil.copy(test_dir / "toy.npz", test_dir / "gone.npz")
    dinomaly_serving.main(base + ["--overwrite"])
    assert sorted(p.name for p in test_dir.iterdir()) == ["other.npz", "run.json", "toy.npz"]
    assert len(project["ledger"].read_text(encoding="utf-8").splitlines()) == 2


# 확인: export는 --model(학습한 모델)과 --untrained(지연 관문용 초기값 모델) 가운데 정확히 하나만 받는다. 둘
# 다 없거나 둘 다 주면 출력 폴더를 만들기 전에 멈춘다.
def test_export_cli_needs_exactly_one_model_source(tmp_path):
    with pytest.raises(SystemExit):
        dinomaly_serving.main(["export", "--config", "dms-280", "--out", str(tmp_path / "a")])
    with pytest.raises(SystemExit):
        dinomaly_serving.main(
            ["export", "--config", "dms-280", "--untrained", "--model", "m.pt", "--out", str(tmp_path / "a")]
        )
    assert not (tmp_path / "a").exists()
