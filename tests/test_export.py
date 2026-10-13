# [파일 역할] 4단계 PatchCore 서빙 경로의 내보내기 모듈(`defect_inspect.export`)과, 그 결과를 읽는 서비스 쪽
# `inspector.Inspector`를 함께 검증하는 테스트 묶음이다. 패치 특징 추출기를 ONNX로 내보내기, torch와의 차이
# 확인, 정적 INT8 양자화, 아티팩트 폴더 저장·읽기와 meta 검사를 다룬다.
# [왜 필요한가] 서비스는 torch 없이 onnxruntime + numpy로 PatchCore 점수를 다시 계산한다(inspector 모듈
# docstring). 그래서 (1) ONNX 특징이 torch 특징과 같은지, (2) numpy 경로의 이미지 점수·256 격자 맵이 torch의
# patchcore.score_images와 같은지, (3) 깨진 아티팩트나 meta가 모델과 맞지 않는 경우를 조용히 넘기지 않고
# ValueError로 막는지가 중요하다. DINOv2 그래프는 onnxruntime 정적 양자화 전처리가 실패한다는
# 사실(docs/experiments.md 규칙 변경 5)도 여기서 고정한다.
# [설계] 사전학습 백본 대신 작은 합성곱 추출기(TinyPatches)를 써서 CPU에서 빠르게 돈다. 실제 WRN-50·DINOv2
# 내보내기는 무작위 가중치·작은 입력으로 한 번씩만 본다.

import numpy as np
import pytest

# 내보내기에는 torch·onnx, 채점에는 onnxruntime이 필요하다. 없으면 파일 전체를 건너뛴다(CI는 기본 패키지만
# 설치).
torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from defect_inspect import export, inspector  # noqa: E402
from defect_inspect.backbones import WideResNetPatches  # noqa: E402
from defect_inspect.inspector import Inspector  # noqa: E402
from defect_inspect.patchcore import build_bank, collect_features, score_images  # noqa: E402

# 가짜 추출기의 특징 차원. 작게 둬서 뱅크와 거리 계산이 빠르다.
DIM = 12


# backbones.py 추출기와 같은 인터페이스(name, dim, [B, 3, S, S] → [B, S/8, S/8, DIM])를 가진 작은 합성곱
# 추출기. 생성자에서 seed를 고정해 몇 번 만들어도 같은 가중치가 된다(torch 모델과 내보낸 ONNX를 따로 만들어
# 비교할 수 있게).
class TinyPatches(torch.nn.Module):
    """A small conv extractor with the interface of backbones.py: [B, 3, S, S] -> [B, S/8, S/8, DIM]."""

    name = "tiny"
    dim = DIM

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(8, DIM, 3, stride=4, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.nn.functional.avg_pool2d(self.conv2(x), 3, 1, 1)
        return x.permute(0, 2, 3, 1).contiguous()


# seed로 고정한 중간 밝기(60~199) 무작위 이미지. defect=True면 왼쪽 위 1/4 칸을 흰색으로 칠해 점수가 높게
# 나오는 '결함' 이미지를 만든다.
def _images(n, size, seed=0, defect=False):
    rng = np.random.default_rng(seed)
    images = rng.integers(60, 200, (n, size, size, 3), dtype=np.uint8)
    if defect:
        q = size // 4
        images[:, q : 2 * q, q : 2 * q] = 255
    return images


# Inspector가 요구하는 meta 키(버전, 이름, 범주, 백본, 입력 크기, 격자, 차원, 재가중 k, 맵 σ, 임계값)를 채운
# 기본 meta.
def _meta(size, grid, **extra):
    meta = {
        "version": inspector.ARTIFACT_VERSION,
        "name": "tiny",
        "category": "toy",
        "backbone": "tiny",
        "img_size": size,
        "grid": [grid, grid],
        "dim": DIM,
        "reweight_k": 3,
        "sigma": 1.0,
        "threshold": 0.0,
    }
    meta.update(extra)
    return meta


# TinyPatches를 64px와 256px로 한 번씩 내보내 파일 안의 테스트들이 같이 쓴다(module 범위, 내보내기 비용을 한
# 번만 낸다).
@pytest.fixture(scope="module")
def tiny_onnx(tmp_path_factory):
    out = {}
    for size in (64, 256):
        path = tmp_path_factory.mktemp(f"onnx{size}") / "model_fp32.onnx"
        export.export_onnx(TinyPatches(), size, path)
        out[size] = path
    return out


# 확인: 내보낸 ONNX의 특징이 torch 특징과 1e-4 안에서 같고, 입력 이름 image [1, 3, 64, 64], 출력 이름 features
# [1, 8, 8, DIM]으로 크기가 고정돼 있다. 서비스가 이 이름과 모양을 그대로 쓴다.
def test_export_matches_the_torch_extractor(tiny_onnx):
    parity = export.check_parity(TinyPatches(), tiny_onnx[64], _images(4, 64, seed=1))
    assert parity["max_abs"] < 1e-4 and parity["rel"] < 1e-4
    session = export._session(tiny_onnx[64])
    assert [i.name for i in session.get_inputs()] == ["image"]
    assert [o.name for o in session.get_outputs()] == ["features"]
    assert session.get_inputs()[0].shape == [1, 3, 64, 64]
    assert session.get_outputs()[0].shape == [1, 8, 8, DIM]


# 확인: onnxruntime + numpy 경로(Inspector)의 이미지 점수가 torch 경로(patchcore.score_images, CPU)와 상대
# 1e-3 안에서 같고, 256 격자 히트맵이 맵 값 범위의 1% 안에서 같다. 정상 5장과 결함 5장, 64px와 256px(맵 크기
# 조정이 있는 경우와 없는 경우)를 모두 본다.
# 왜: 임계값은 torch 경로 점수로 정하고 서비스는 CPU 경로로 채점하므로 두 경로의 점수가 같아야 임계값을 그대로
# 쓸 수 있다(README: CPU로 옮겨도 점수는 같다).
@pytest.mark.parametrize("size", [64, 256])
def test_inspector_reproduces_the_torch_pipeline(tiny_onnx, size):
    """Image scores and 256-grid maps of the numpy/onnxruntime path against patchcore.score_images on CPU."""
    ext = TinyPatches()
    train = _images(10, size, seed=2)
    feats, grid = collect_features(ext, train, batch_size=4, device="cpu")
    bank = build_bank(feats, 0.25, seed=0, device="cpu")
    images = np.concatenate([_images(5, size, seed=3), _images(5, size, seed=4, defect=True)])
    ref = score_images(ext, bank, images, batch_size=4, device="cpu", reweight_k=3, sigma=1.0, map_size=256)

    insp = Inspector(export._session(tiny_onnx[size]), bank.numpy(), _meta(size, grid[0]))
    scores = insp.scores(images)
    np.testing.assert_allclose(scores, ref.image_scores, rtol=1e-3)
    span = float(ref.maps.astype(np.float32).max() - ref.maps.astype(np.float32).min())
    for i in (0, 7):
        heat = insp.inspect(images[i]).heatmap
        assert np.abs(heat - ref.maps[i].astype(np.float32)).max() < 1e-2 * span


# 확인: 아티팩트를 두 가지 배치로 읽을 수 있다. 한 폴더에 모델·뱅크·meta가 다 있는 배치와, 모델 하나를 상위
# 폴더에 두고 범주 폴더에는 뱅크·meta만 두는 세트 배치다. 두 경우 점수가 같고, 임계값·뱅크 행 수·정밀도·스레드
# 설정이 읽히며, 없는 INT8 모델을 요청하면 'missing' ValueError다.
# 왜: export.build_artifacts는 모델 파일을 모든 범주가 같이 쓰는 세트 배치로 만든다(모듈 docstring).
def test_inspector_loads_a_saved_artifact_set(tiny_onnx, tmp_path):
    ext = TinyPatches()
    feats, grid = collect_features(ext, _images(6, 64, seed=5), batch_size=4, device="cpu")
    bank = build_bank(feats, 0.5, seed=0, device="cpu").numpy()
    meta = {k: v for k, v in _meta(64, grid[0], threshold=1.25).items() if k != "version"}
    # Single-folder layout.
    inspector.save_artifact(tmp_path / "one", onnx_fp32=tiny_onnx[64], bank=bank, meta=meta)
    a = Inspector.load(tmp_path / "one", threads=1)
    # Set layout: the model next to the category folders.
    (tmp_path / "set" / "toy").mkdir(parents=True)
    (tmp_path / "set" / "model_fp32.onnx").write_bytes(tiny_onnx[64].read_bytes())
    for name in ("bank.npy", "meta.json"):
        (tmp_path / "set" / "toy" / name).write_bytes((tmp_path / "one" / name).read_bytes())
    b = Inspector.load(tmp_path / "set" / "toy")
    images = _images(3, 64, seed=6)
    np.testing.assert_array_equal(a.scores(images), b.scores(images))
    assert a.threshold == 1.25 and a.bank_rows == bank.shape[0] and a.precision == "fp32"
    assert a.threads == 1 and b.threads is None  # None = onnxruntime's default
    with pytest.raises(ValueError, match="missing"):
        Inspector.load(tmp_path / "one", precision="int8")


# 확인: 정적 INT8 양자화(QDQ, MinMax 보정)가 읽을 수 있는 모델을 만들고, 중간 파일(.prep.onnx)을 남기지
# 않으며, INT8 특징이 FP32와 다르지만 상관이 0.98을 넘는다. 4단계 규칙의 INT8 방식 그대로다.
def test_int8_quantization_gives_a_loadable_model_close_to_fp32(tiny_onnx, tmp_path):
    int8 = export.quantize_static_int8(tiny_onnx[64], tmp_path / "model_int8.onnx", _images(16, 64, seed=7))
    assert int8.exists() and not (tmp_path / "model_int8.prep.onnx").exists()
    images = _images(4, 64, seed=8)
    fp32_session, int8_session = export._session(tiny_onnx[64]), export._session(int8)
    a = np.concatenate(
        [fp32_session.run(["features"], {"image": inspector.preprocess(im, 64)})[0] for im in images]
    )
    b = np.concatenate(
        [int8_session.run(["features"], {"image": inspector.preprocess(im, 64)})[0] for im in images]
    )
    assert a.shape == b.shape and not np.array_equal(a, b)
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.98


# 확인: FP32 모델 파일이 없으면 FileNotFoundError, 보정 이미지가 0장이면 'calibration' ValueError이고, 어느
# 경우에도 INT8 파일을 만들지 않는다.
def test_quantization_needs_a_model_and_calibration_images(tiny_onnx, tmp_path):
    with pytest.raises(FileNotFoundError):
        export.quantize_static_int8(tmp_path / "nothing.onnx", tmp_path / "int8.onnx", _images(2, 64))
    with pytest.raises(ValueError, match="calibration"):
        export.quantize_static_int8(tiny_onnx[64], tmp_path / "int8.onnx", _images(0, 64))
    assert not (tmp_path / "int8.onnx").exists()


# 확인: onnxruntime 전처리(quant_pre_process)가 그래프 때문에 실패하면 원인 예외 이름과 메시지를 담은
# NotImplementedError로 바꾸고, 모델과 중간 파일을 하나도 남기지 않는다. 반대로 디스크 부족 같은 OSError는
# 그대로 올린다.
# 왜: build_artifacts는 NotImplementedError만 'INT8을 쓸 수 없음'으로 기록하고 FP32 아티팩트를 완성한다. 기계
# 문제까지 그렇게 낮춰 처리하면 INT8이 조용히 빠진다. monkeypatch로 전처리 함수를 일부러 실패하게 만든다.
def test_a_graph_the_preprocessing_cannot_handle_is_refused_with_the_reason(tiny_onnx, tmp_path, monkeypatch):
    from onnxruntime.quantization import shape_inference

    def broken(*args, **kwargs):
        raise TypeError("object of type 'NoneType' has no len()")

    monkeypatch.setattr(shape_inference, "quant_pre_process", broken)
    with pytest.raises(NotImplementedError, match="pre-processing failed .*TypeError.*NoneType"):
        export.quantize_static_int8(tiny_onnx[64], tmp_path / "model_int8.onnx", _images(2, 64))
    assert list(tmp_path.iterdir()) == []  # neither a model nor the intermediate file

    def disk_full(*args, **kwargs):
        raise OSError("no space left on device")

    # A failure of the machine is not "this graph cannot be quantized": it must not be downgraded.
    monkeypatch.setattr(shape_inference, "quant_pre_process", disk_full)
    with pytest.raises(OSError, match="no space"):
        export.quantize_static_int8(tiny_onnx[64], tmp_path / "model_int8.onnx", _images(2, 64))


# 확인: DINOv2 ViT-S 추출기(무작위 가중치, 56px)도 ONNX로 내보내지고 torch와 같으며 출력이 [1, 4, 4, 384]다.
# 정적 INT8은 전처리 실패로 거부되거나(onnxruntime 1.30: class·register 토큰의 Expand에서 실패) 쓸 수 있는
# 모델을 내야 하며, 어느 쪽이든 중간 파일이 남지 않는다.
# 왜: 규칙 변경 5에 'DINOv2 내보내기는 INT8로 양자화할 수 없다'가 기록돼 있다. onnxruntime이 바뀌어 양자화가
# 되더라도 테스트가 깨지지 않게 두 결과를 모두 받는다.
def test_dinov2_exports_and_its_int8_quantization_is_refused_or_usable(tmp_path):
    """The ViT graph must export; onnxruntime's static quantization either refuses it or gives a model."""
    pytest.importorskip("timm")
    from defect_inspect.backbones import make_extractor

    ext = make_extractor("dinov2_vits14", img_size=56, pretrained=False)
    path = export.export_onnx(ext, 56, tmp_path / "model_fp32.onnx")
    parity = export.check_parity(ext, path, _images(2, 56, seed=10))
    assert parity["max_abs"] < 1e-4
    session = export._session(path)
    assert session.get_inputs()[0].shape == [1, 3, 56, 56]
    assert session.get_outputs()[0].shape == [1, 4, 4, 384]
    int8 = tmp_path / "model_int8.onnx"
    try:
        export.quantize_static_int8(path, int8, _images(4, 56, seed=11))
    except NotImplementedError as err:
        # onnxruntime 1.30: symbolic shape inference fails at the Expand of the class/register tokens.
        assert "pre-processing failed" in str(err) and not int8.exists()
    else:
        out = export._session(int8).run(["features"], {"image": inspector.preprocess(_images(1, 56)[0], 56)})
        assert out[0].shape == (1, 4, 4, 384) and np.isfinite(out[0]).all()
    assert sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".onnx" and "prep" in p.name) == []


# 확인: fp16·train 모드 모듈을 내보내도 호출자의 모듈은 train 모드·fp16 그대로이고, 내보낸 것은 그 모듈의 fp32
# 판이다. 이미 CPU·fp32·eval인 모듈은 복사하지 않고 그대로 쓴다.
# 왜: export가 모듈을 제자리에서 .float().eval()로 바꾸면 GPU fp16으로 이어서 쓰려던 호출자가 영향을 받는다.
# 반대로 큰 백본을 매번 복사하면 메모리를 낭비하므로 _cpu_fp32는 필요할 때만 deepcopy한다.
def test_export_and_parity_leave_the_callers_module_alone(tmp_path):
    ext = TinyPatches().half()
    ext.train()
    path = export.export_onnx(ext, 64, tmp_path / "model.onnx")
    export.check_parity(ext, path, _images(1, 64))
    assert ext.training and all(p.dtype == torch.float16 for p in ext.parameters())
    # The exported model is the fp32 version of it.
    reference = TinyPatches()
    reference.load_state_dict({k: v.float() for k, v in ext.state_dict().items()})
    assert export.check_parity(reference, path, _images(2, 64, seed=3))["max_abs"] < 1e-4
    # A module that is ready is used as it is (no copy of a large backbone).
    ready = TinyPatches().eval()
    assert export._cpu_fp32(ready) is ready and export._cpu_fp32(ext) is not ext


# 확인: meta의 입력 크기·격자·차원이 ONNX 모델의 선언된 입력·출력 모양과 다르면 "the model's ..." ValueError로
# 읽기를 거부한다. 손으로 만든 세션으로 Inspector를 직접 만들 때도 같은 검사를 거치고, 잘린 모델 파일은
# onnxruntime 예외가 아니라 'not a loadable ONNX model' ValueError가 된다.
# 왜: 모양이 맞지 않는 아티팩트는 첫 요청에서야 알 수 없는 오류를 내거나 틀린 점수를 낼 수 있다. 서비스가
# 시작할 때 막는다. pytest.fail은 예외가 나지 않았을 때 어떤 경우였는지 메시지로 남긴다.
def test_load_checks_the_meta_against_the_models_declared_shapes(tiny_onnx, tmp_path):
    ext = TinyPatches()
    feats, grid = collect_features(ext, _images(6, 64, seed=5), batch_size=4, device="cpu")
    bank = build_bank(feats, 0.5, seed=0, device="cpu").numpy()
    good = {k: v for k, v in _meta(64, grid[0]).items() if k != "version"}
    bad = {
        "img_size 96 with the 64 px model": {"img_size": 96, "grid": [12, 12]},
        "grid 4x4 with the 8x8 model": {"grid": [4, 4]},
    }
    for i, (name, change) in enumerate(bad.items()):
        art = tmp_path / f"art{i}"
        inspector.save_artifact(art, onnx_fp32=tiny_onnx[64], bank=bank, meta={**good, **change})
        with pytest.raises(ValueError, match="the model's"):
            Inspector.load(art)
            pytest.fail(f"loaded an artifact with {name}")
    art = tmp_path / "dim"
    inspector.save_artifact(art, onnx_fp32=tiny_onnx[64], bank=bank[:, :5], meta={**good, "dim": 5})
    with pytest.raises(ValueError, match="the model's output"):
        Inspector.load(art)
    # The session built by hand goes through the same check.
    with pytest.raises(ValueError, match="the model's"):
        Inspector(export._session(tiny_onnx[64]), bank, _meta(64, 4))
    # A broken model file is a ValueError too, not an onnxruntime exception.
    art = tmp_path / "broken"
    inspector.save_artifact(art, onnx_fp32=tiny_onnx[64], bank=bank, meta=good)
    (art / "model_fp32.onnx").write_bytes(tiny_onnx[64].read_bytes()[:200])
    with pytest.raises(ValueError, match="not a loadable ONNX model"):
        Inspector.load(art)


# 확인: 실제 서빙 백본인 WRN-50(무작위 가중치, 64px) 추출기가 ONNX로 내보내지고 torch와 1e-3 안에서 같으며
# 출력이 [1, 8, 8, 1536]이다. 4단계 서빙 구성(wrn50-256)과 같은 그래프 구조를 작은 입력으로 확인한다.
def test_wide_resnet_export_at_a_small_size(tmp_path):
    ext = WideResNetPatches(pretrained=False)
    path = export.export_onnx(ext, 64, tmp_path / "wrn.onnx")
    parity = export.check_parity(ext, path, _images(1, 64, seed=9))
    assert parity["max_abs"] < 1e-3
    session = export._session(path)
    assert session.get_outputs()[0].shape == [1, 8, 8, 1536]
