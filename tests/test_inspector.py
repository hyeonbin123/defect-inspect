# [파일 역할] src/defect_inspect/inspector.py의 PatchCore 서빙 경로(Inspector와 numpy 도우미 함수)를 검증하는
# 테스트 묶음이다. ReconstructionInspector(Dinomaly)는 tests/test_reconstruction_inspector.py가 따로 맡는다.
# [왜 필요한가] 메모리 뱅크와 임계값은 torch 파이프라인(patchcore.score_images, GPU fp16)에서 만들어지고,
# 서비스는 torch 없이 onnxruntime + numpy로 같은 계산을 다시 한다. 두 경로의 산술(fp16 반올림, 뱅크 중심화,
# 최근접 탐색, 재가중, 점수 맵)이 조금이라도 다르면 GPU 점수로 정한 임계값이 CPU 서비스에서 다른 판정을 낸다.
# 그래서 각 단계를 정의식·torch 규칙과 맞춰 보고, 잘못된 아티팩트·NaN 점수가 판정에 닿지 않는지도 본다. 실제
# torch 파이프라인과 점수·맵을 직접 대조하는 테스트는 torch가 필요해 tests/test_export.py에 있다.
# [주요 구성] 가짜 ONNX 세션(FakeSession과 파생 클래스)으로 실제 모델 없이 돌린다. 앞부분은 순수
# 함수(preprocess, nearest, image_score, resize_bilinear, score_map), 가운데는 Inspector의 판정·보정·유한성
# 검사, 뒷부분은 meta·뱅크·모델 입출력·파일 적재 검증과 torch 비의존 확인이다.

import json
import warnings
from types import SimpleNamespace

# numpy로 기대값을 직접 계산하고, PIL은 preprocess가 PIL 이미지를 받는 경로(크기 변환·회색조 변환)를
# 시험하려고 가져온다.
import numpy as np
import pytest
from PIL import Image

# 모듈 자체(inspector)는 함수·상수를 부르고 monkeypatch로 블록 크기를 바꾸려고, Inspector 클래스는 짧게 쓰려고
# 따로 가져온다.
from defect_inspect import inspector
from defect_inspect.inspector import Inspector

# 가짜 모델의 패치 격자(4×4)와 입력 한 변(32px). 실제 아티팩트(256px 등)보다 훨씬 작게 잡아 CPU에서 빨리 돈다.
# 32/4 = 8px 칸 하나가 패치 하나다.
GRID = 4
SIZE = 32


# onnxruntime 세션 대신 쓰는 가짜 모델. run(names, feeds)만 같은 모양으로 흉내 내며, 각 격자 칸의 평균
# 색(정규화된 RGB 3개)을 3차원 패치 특징으로 돌려준다. 출력 모양은 실제 내보낸 모델과 같은 [1, H, W, dim]이다.
# get_inputs/get_outputs가 없으므로 Inspector는 선언된 입출력 모양 검사(check_model_io)를 건너뛴다. 그 검사는
# 아래 DescribedSession으로 따로 시험한다. calls는 호출 횟수를 세지만 이 파일의 테스트는 읽지 않는다.
class FakeSession:
    """onnxruntime 세션 대신 쓰는 가짜 모델: 격자 칸마다 평균 색을 3차원 패치 특징으로 낸다."""

    def __init__(self, grid=GRID):
        self.grid = grid
        self.calls = 0

    def run(self, names, feeds):
        assert names == ["features"]
        x = feeds["image"]
        self.calls += 1
        b, c, h, w = x.shape
        cell = h // self.grid
        # [B, C, H, W]를 칸 단위로 쪼개 칸마다 평균을 낸 뒤 channels-last [B, grid, grid, C]로 바꾼다.
        out = x.reshape(b, c, self.grid, cell, self.grid, cell).mean(axis=(3, 5))
        return [np.ascontiguousarray(out.transpose(0, 2, 3, 1)).astype(np.float32)]


# check_meta를 통과하는 기본 meta(META_KEYS 전부)를 만들고, 키워드로 준 값만 덮어쓴다. 잘못된 값 하나만 바꿔
# 넣는 검증 테스트에서 쓴다.
def _meta(**extra):
    meta = {
        "version": inspector.ARTIFACT_VERSION,
        "name": "toy",
        "category": "toy",
        "backbone": "gridmean",
        "img_size": SIZE,
        "grid": [GRID, GRID],
        "dim": 3,
        "reweight_k": 3,
        "sigma": 1.0,
        "threshold": 0.5,
    }
    meta.update(extra)
    return meta


# 회색 잡음(100~139) uint8 이미지 n장. defect=True면 8~16px 구역(4×4 격자의 (1,1) 칸)을 흰색으로 칠해 '결함'을
# 만든다. seed를 받은 Generator만 써서 결정적이다.
def _images(n, seed=0, defect=False):
    rng = np.random.default_rng(seed)
    images = rng.integers(100, 140, (n, SIZE, SIZE, 3), dtype=np.uint8)
    if defect:
        images[:, 8:16, 8:16] = 255
    return images


# 정상 이미지들의 패치 특징을 모두 모아 fp16 메모리 뱅크로 만든다. 실제 뱅크도 torch 쪽에서 fp16으로
# 저장되므로(patchcore.build_bank) 같은 형식으로 맞춘다. 코어셋 선택은 하지 않고 전부 넣는다.
def _bank(images):
    session = FakeSession()
    feats = [
        session.run(["features"], {"image": inspector.preprocess(im, SIZE)})[0].reshape(-1, 3)
        for im in images
    ]
    return np.concatenate(feats).astype(np.float16)


# preprocess가 정의대로 '/255 → ImageNet 평균·표준편차 정규화 → [1, 3, S, S] float32'를 만드는지 확인한다.
# 왜 중요한가: 정규화가 학습·내보내기 때와 다르면 같은 사진에서 특징이 달라져 점수 전체가 틀어진다. 크기가
# 맞는 PIL 이미지와 배열이 같은 텐서를 내는지, 다른 크기는 리사이즈 캐시와 같은 BICUBIC으로 줄이는지, 회색조
# 이미지도 RGB로 바꿔 받는지, 크기·dtype이 틀린 배열은 거절하는지까지 본다.
def test_preprocess_matches_the_definition():
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)
    x = inspector.preprocess(array, SIZE)
    assert x.shape == (1, 3, SIZE, SIZE) and x.dtype == np.float32
    mean, std = np.array(inspector.IMAGENET_MEAN), np.array(inspector.IMAGENET_STD)
    expected = ((array / 255.0 - mean) / std).transpose(2, 0, 1)
    np.testing.assert_allclose(x[0], expected, rtol=1e-5, atol=1e-6)
    # 크기가 맞는 PIL 이미지는 같은 텐서를 내고, 다른 크기는 BICUBIC으로 리사이즈한다.
    np.testing.assert_array_equal(inspector.preprocess(Image.fromarray(array), SIZE), x)
    big = Image.fromarray(rng.integers(0, 256, (90, 120, 3), dtype=np.uint8))
    resized = np.asarray(big.resize((SIZE, SIZE), Image.Resampling.BICUBIC))
    np.testing.assert_array_equal(inspector.preprocess(big, SIZE), inspector.preprocess(resized, SIZE))
    grey = Image.fromarray(array[..., 0])
    assert inspector.preprocess(grey, SIZE).shape == (1, 3, SIZE, SIZE)
    with pytest.raises(ValueError):
        inspector.preprocess(array[:16], SIZE)
    with pytest.raises(ValueError):
        inspector.preprocess(array.astype(np.float32), SIZE)


# nearest(전개식 ||b||^2 - 2q·b로 찾은 최근접 행)가 전수 비교(brute force)와 같은 인덱스·거리를 내는지 본다.
# 뱅크에 +5.0을 더한 것은 모든 특징이 큰 공통 오프셋을 가진 상황(ReLU 뒤 특징, patchcore.score_images 주석)을
# 흉내 낸 것이고, 그래서 중심화한 뒤 넘긴다. 뱅크에 있는 행을 그대로 질의하면 거리가 정확히 0이어야 한다. 찾은
# 행까지 거리를 직접 뺄셈으로 다시 계산하는 설계 덕분에 전개식의 상쇄 오차가 남지 않는지를 확인하는 것이다.
def test_nearest_matches_brute_force_and_is_exact_for_bank_members():
    rng = np.random.default_rng(1)
    bank = rng.normal(size=(200, 16)).astype(np.float32) + 5.0
    query = np.concatenate([rng.normal(size=(40, 16)).astype(np.float32) + 5.0, bank[:7]])
    centre = bank.mean(axis=0, keepdims=True)
    b, q = bank - centre, query - centre
    dist, index = inspector.nearest(q, b, np.square(b).sum(axis=1))
    brute = np.linalg.norm(query[:, None, :] - bank[None, :, :], axis=2)
    np.testing.assert_array_equal(index, brute.argmin(axis=1))
    np.testing.assert_allclose(dist, brute.min(axis=1), rtol=1e-5, atol=1e-6)
    assert np.all(dist[-7:] == 0.0)


# 메모리를 아끼려고 질의를 블록으로 나눠 처리해도(실제 상한 _MAX_DIST_ELEMS = 2^27 원소 = 512MB) 결과가 같은지
# 본다. monkeypatch로 상한을 50×4로 줄여 블록 4행씩, 33행이라 마지막 블록이 덜 찬 경우까지 강제로 만든다. 블록
# 경계에서 인덱스를 잘못 이어 붙이는 회귀를 막는다.
def test_nearest_in_blocks_gives_the_same_result(monkeypatch):
    rng = np.random.default_rng(2)
    bank = rng.normal(size=(50, 8)).astype(np.float32)
    query = rng.normal(size=(33, 8)).astype(np.float32)
    sq = np.square(bank).sum(axis=1)
    whole = inspector.nearest(query, bank, sq)
    monkeypatch.setattr(inspector, "_MAX_DIST_ELEMS", 50 * 4)  # 블록마다 질의 4행
    blocks = inspector.nearest(query, bank, sq)
    np.testing.assert_array_equal(whole[1], blocks[1])
    np.testing.assert_allclose(whole[0], blocks[0], rtol=1e-6)


# image_score가 PatchCore 이미지 점수 정의를 따르는지 본다: 가장 큰 패치 점수 s*에, 그 패치의 최근접 뱅크 행
# m*와 m*에 가까운 뱅크 행 k개(m* 포함, m*가 맨 앞)까지의 거리 d로 만든 가중치 1 - softmax(d)[0]를 곱한다. k =
# 1이면 재가중 없이 s*, k = 500은 뱅크(60행)보다 커서 뱅크 전체를 쓰는 경계 사례다. 기대값은 거리 정렬로 따로
# 계산해 구현(argpartition + 전개식)과 독립적으로 맞춘다.
@pytest.mark.parametrize("k", [1, 2, 9, 500])
def test_image_score_is_the_reweighted_largest_patch_score(k):
    rng = np.random.default_rng(3)
    bank = rng.normal(size=(60, 6)).astype(np.float32)
    query = rng.normal(size=(16, 6)).astype(np.float32) * 1.5
    sq = np.square(bank).sum(axis=1)
    dist, index = inspector.nearest(query, bank, sq)
    score = inspector.image_score(dist, index, query, bank, sq, reweight_k=k)

    p = int(dist.argmax())
    if k <= 1:
        expected = dist[p]
    else:
        m = index[p]
        to_m = np.linalg.norm(bank - bank[m], axis=1)
        support = np.argsort(to_m, kind="stable")[: min(k, len(bank))]
        assert support[0] == m
        d = np.linalg.norm(query[p] - bank[support], axis=1).astype(np.float64)
        expected = (1.0 - np.exp(d[0]) / np.exp(d).sum()) * dist[p]
    assert score == pytest.approx(float(expected), rel=1e-5)


# 뱅크에 똑같은 행이 두 개 있을 때 m*가 어느 쪽이든 같은 점수가 나오는지 본다. 구현은 m* 자리를 -inf로 두어
# 항상 맨 앞에 오게 한다(torch 쪽 patchcore._image_scores와 같은 규칙). 지지 집합은 두 중복 행이고 질의까지
# 거리가 둘 다 1이라 가중치는 1 - 1/2, 점수는 0.5다.
def test_image_score_puts_m_star_first_among_duplicates():
    bank = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    sq = np.square(bank).sum(axis=1)
    dist = np.array([1.0], dtype=np.float32)
    for m in (0, 1):
        score = inspector.image_score(dist, np.array([m]), query, bank, sq, reweight_k=2)
        # 지지 집합 = 두 중복 행이고 질의까지 거리가 둘 다 1이라 가중치는 1 - 1/2.
        assert score == pytest.approx(0.5)


# numpy resize_bilinear가 torch F.interpolate(mode="bilinear", align_corners=False)와 같은 공식인지 손으로
# 계산한 값으로 확인한다: 같은 크기면 그대로, 상수 맵은 상수, 2→4 확대는 원본 좌표 (i+0.5)·(n_in/n_out)-0.5를
# 0과 끝에서 잘라 쓰고, 4→2 축소는 안티에일리어싱 없이 두 점만 뽑는다. torch 쪽 점수
# 맵(patchcore.score_images)과 서비스 히트맵이 같아야 하므로 중요하다.
def test_resize_bilinear_matches_the_align_corners_false_formula():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_array_equal(inspector.resize_bilinear(a[:3, :3], 3), a[:3, :3])
    np.testing.assert_allclose(inspector.resize_bilinear(np.full((5, 5), 2.5, np.float32), 17), 2.5)
    # 한 축으로 2 -> 4: 원본 좌표 -0.25(0으로 잘림), 0.25, 0.75, 1.25(1로 잘림).
    up = inspector.resize_bilinear(np.array([[0.0, 4.0], [0.0, 4.0]], dtype=np.float32), 4)
    np.testing.assert_allclose(up[0], [0.0, 1.0, 3.0, 4.0])
    # 안티에일리어싱 없는 축소: 4 -> 2는 좌표 0.5와 2.5의 값을 뽑는다.
    down = inspector.resize_bilinear(np.tile(np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32), (4, 1)), 2)
    np.testing.assert_allclose(down[0], [0.5, 2.5])


# 가우시안 커널 크기 규칙 2·int(4σ+0.5)+1(torch 쪽 patchcore.gaussian_kernel1d와 같음)과 정규화, σ ≤ 0이면
# [1](흐림 없음)인지 본다. 이어서 score_map(확대 → 흐림 → 256 격자로 크기 조정)이 뜨거운 패치 위치를 지키는지,
# 상수 맵을 상수로 두는지(정규화된 커널 + mirror 패딩), σ=0이면 단순 확대와 같은지, 흐림 반경이 맵보다 크면
# ValueError인지 확인한다. 마지막 검사는 torch의 reflect 패딩이 반경 < 맵 크기를 요구하는 것과 같은 조건이다.
def test_gaussian_kernel_and_score_map():
    assert inspector.gaussian_kernel1d(4.0).shape == (33,)
    assert inspector.gaussian_kernel1d(0.4).shape == (5,)  # 반경 int(4 * 0.4 + 0.5) = 2
    assert inspector.gaussian_kernel1d(0.0).tolist() == [1.0]
    assert inspector.gaussian_kernel1d(1.0).sum() == pytest.approx(1.0)

    patch = np.zeros((4, 4), dtype=np.float32)
    patch[1, 2] = 8.0
    out = inspector.score_map(patch, SIZE, sigma=1.0, map_size=64)
    assert out.shape == (64, 64) and out.dtype == np.float32
    r, c = np.unravel_index(int(out.argmax()), out.shape)
    assert 16 <= r < 32 and 32 <= c < 48  # 64 격자에서 뜨거운 패치의 칸
    np.testing.assert_allclose(
        inspector.score_map(np.full((4, 4), 3.0, np.float32), SIZE, 1.0, 64), 3.0, rtol=1e-5
    )
    unblurred = inspector.score_map(patch, SIZE, sigma=0.0, map_size=SIZE)
    np.testing.assert_allclose(unblurred, inspector.resize_bilinear(patch, SIZE))
    with pytest.raises(ValueError):
        inspector.score_map(patch, 8, sigma=4.0)


# Inspector를 끝까지 돌리는 통합 테스트다. 흰 사각형을 넣은 결함 이미지가 모든 정상보다 높은 점수를 받고,
# 임계값을 정상 최댓값으로 두면 결함으로 판정되며, 히트맵(256×256 float32)의 최댓값이 사각형 근처에 있는지
# 본다. heatmap=False면 히트맵을 만들지 않는다.
# 마지막 두 줄은 판정 규칙 'score > threshold'(calibrate.py 모듈 docstring)를 확인한다. 점수가 임계값과 정확히
# 같으면 정상이다.
def test_inspector_scores_defects_above_normals_and_flags_them():
    bank = _bank(_images(12))
    insp = Inspector(FakeSession(), bank, _meta())
    normal = insp.scores(_images(6, seed=5))
    defect = insp.scores(_images(6, seed=6, defect=True))
    assert normal.dtype == np.float32 and defect.min() > normal.max()
    insp.set_threshold(float(normal.max()))
    result = insp.inspect(_images(1, seed=7, defect=True)[0])
    assert result.is_defect and result.score > result.threshold
    assert result.heatmap.shape == (256, 256) and result.heatmap.dtype == np.float32
    r, c = np.unravel_index(int(result.heatmap.argmax()), result.heatmap.shape)
    assert 48 <= r < 144 and 48 <= c < 144  # 밝은 사각형(256 격자의 행·열 64..128) 근처
    clean = insp.inspect(_images(1, seed=8)[0], heatmap=False)
    assert not clean.is_defect and clean.heatmap is None
    # 임계값과 정확히 같은 점수는 결함이 아니다(규칙: score > threshold).
    insp.set_threshold(clean.score)
    assert not insp.inspect(_images(1, seed=8)[0], heatmap=False).is_defect


# 뱅크를 만든 이미지 자체를 채점하면 모든 패치가 뱅크에 있으므로 점수가 정확히 0이어야 한다(reweight_k=1로
# 재가중을 꺼서 s* 그대로 비교). 또 features가 torch 파이프라인처럼 fp16으로 한 번 반올림된 값을 돌려주는지
# 본다. 뱅크가 fp16이므로 질의도 같은 반올림을 거쳐야 같은 이미지가 거리 0이 된다.
def test_a_bank_member_scores_zero_and_features_are_rounded_to_fp16():
    images = _images(3)
    insp = Inspector(FakeSession(), _bank(images), _meta(reweight_k=1))
    np.testing.assert_array_equal(insp.scores(images), 0.0)
    feats = insp.features(images[0])
    np.testing.assert_array_equal(feats, feats.astype(np.float16).astype(np.float32))


# calibrate가 정상 이미지 점수로 split-conformal 임계값을 잡는지 본다. n = 40, α = 0.05면 순위 ceil(41·0.95) =
# 39번째 작은 점수가 임계값이고, meta에 보정 기록(strategy, n, guaranteed)이 남는다. n = 5면 필요한 순위 6이
# n을 넘어 최댓값을 쓰고 guaranteed = False가 된다(calibrate.conformal_threshold 규칙).
# 뒤쪽은 set_threshold 입력 검증이다. NaN·inf·None·문자열·bool은 거절하고 기존 임계값을 그대로 둔다.
# np.float32를 받아도 meta에는 파이썬 float로 저장해, 서비스가 meta를 JSON으로 내보낼 때 numpy 형식이 섞이지
# 않게 한다. 서비스의 POST /calibrate는 같은 규칙과 기록을 service.py에서 따로 적용한다.
def test_calibrate_sets_the_conformal_threshold():
    insp = Inspector(FakeSession(), _bank(_images(12)), _meta())
    normals = list(_images(40, seed=9))
    scores = np.sort(insp.scores(np.stack(normals)))
    value = insp.calibrate(normals, alpha=0.05)
    assert value == pytest.approx(float(scores[38]))  # 순위 ceil(41 * 0.95) = 39
    assert insp.threshold == value and insp.meta["threshold"] == value
    assert insp.meta["calibration"] == {"strategy": "holdout", "n": 40, "guaranteed": True}
    insp.calibrate(normals[:5], alpha=0.05)
    assert insp.threshold == pytest.approx(float(insp.scores(np.stack(normals[:5])).max()))
    assert insp.meta["calibration"]["guaranteed"] is False
    with pytest.raises(ValueError):
        insp.calibrate([])
    kept = insp.threshold
    for bad in (float("nan"), float("inf"), None, "0.5", True):
        with pytest.raises(ValueError):
            insp.set_threshold(bad)
    assert insp.threshold == kept and insp.meta["threshold"] == kept
    insp.set_threshold(np.float32(1.5))
    assert insp.threshold == 1.5 and isinstance(insp.meta["threshold"], float)


# 생성자가 meta(키 누락, 버전 불일치)와 뱅크(차원 불일치, 빈 뱅크)를 검사하는지 본다. 마지막은
# FakeSession(grid=2)처럼 선언 모양이 없는 모델이 meta와 다른 격자를 낼 때, 생성 시점이 아니라 features 호출
# 시점의 실제 출력 모양 검사에서 잡히는지 확인한다.
def test_meta_and_bank_are_validated():
    bank = _bank(_images(4))
    with pytest.raises(ValueError, match="lacks"):
        Inspector(FakeSession(), bank, {k: v for k, v in _meta().items() if k != "sigma"})
    with pytest.raises(ValueError, match="version"):
        Inspector(FakeSession(), bank, _meta(version=99))
    with pytest.raises(ValueError, match="bank shape"):
        Inspector(FakeSession(), bank[:, :2], _meta())
    with pytest.raises(ValueError, match="bank shape"):
        Inspector(FakeSession(), bank[:0], _meta())
    wrong_grid = Inspector(FakeSession(grid=2), bank, _meta())
    with pytest.raises(ValueError, match="features"):
        wrong_grid.features(_images(1)[0])


# 첫 특징값 하나만 원하는 값(fp16 범위를 넘는 큰 값, NaN, inf)으로 바꿔 내는 가짜 모델. 모델 출력이 깨졌을
# 때의 처리를 시험하려고 쓴다.
class SpikeSession(FakeSession):
    """첫 특징값이 `value`(fp16 범위를 넘는 값, 또는 유한하지 않은 값)인 모델."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def run(self, names, feeds):
        out = super().run(names, feeds)
        out[0][0, 0, 0, 0] = self.value
        return out


# 특징값이 fp16 최댓값(65504)을 넘으면(여기서는 1e5) fp16 반올림에서 inf가 되므로, torch 쪽
# patchcore._patch_features처럼 'overflowed fp16' FloatingPointError를 내는지 본다.
# features·inspect·scores·calibrate 네 진입점 모두 확인한다.
# warnings를 오류로 바꾼 것은 numpy의 overflow RuntimeWarning이 새어 나오지 않는지(구현이 np.errstate로
# 막는지)도 함께 보려는 것이다. 보정이 실패하면 임계값은 그대로여야 하고, 정확히 65504는 통과해야
# 한다(경계값).
def test_features_beyond_fp16_raise_like_the_torch_pipeline():
    # patchcore._patch_features는 fp16으로 바꾼 뒤 검사한다. numpy 경로도 경고 없이 같은 검사를 한다.
    insp = Inspector(SpikeSession(1e5), _bank(_images(4)), _meta())
    image = _images(1, seed=3)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for call in (
            lambda: insp.features(image),
            lambda: insp.inspect(image),
            lambda: insp.scores(image[None]),
            lambda: insp.calibrate([image]),
        ):
            with pytest.raises(FloatingPointError, match="overflowed fp16"):
                call()
    assert insp.threshold == 0.5  # 보정이 실패하면 임계값은 그대로다
    # fp16 최댓값(65504)은 여전히 통과한다.
    assert np.isfinite(Inspector(SpikeSession(65504.0), _bank(_images(4)), _meta()).features(image)).all()


# NaN·±inf 특징이 판정까지 가지 않고 FloatingPointError가 되는지 본다. NaN > threshold는 False라서 막지 않으면
# 깨진 출력이 '정상'으로 통과한다(Inspector docstring). 모델 밖에서 받은 특징을 score_features에 직접 넣는
# 경로도 같은 검사를 받는지 확인한다.
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_features_never_give_a_verdict(value):
    insp = Inspector(SpikeSession(value), _bank(_images(4)), _meta())
    image = _images(1, seed=3)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(FloatingPointError):
            insp.inspect(image)
        with pytest.raises(FloatingPointError):
            insp.scores(image[None])
        # 밖에서 받은 특징도 같은 검사를 받는다. 막지 않으면 NaN > threshold가 거짓이라 '정상'으로 읽힌다.
        feats = Inspector(FakeSession(), _bank(_images(4)), _meta()).features(image)
        feats[5, 1] = value
        with pytest.raises(FloatingPointError):
            insp.score_features(feats)


# 오프라인 채점(scores), 서비스 판정(inspect), 보정(calibrate)이 모두 float32로 반올림한 같은 점수를 쓰는지
# 본다. image_score는 float64로 계산한 뒤 float32로 반올림하는데, 이 반올림이 없으면 임계값 자리의 이미지가
# float64의 남은 자릿수 때문에 판정이 뒤집힐 수 있다.
# 각 이미지마다 자기 점수를 임계값으로 두고 inspect하면 점수가 비트 단위로 같고 결함이 아니어야 한다. 보정
# 결과도 float32 점수 중 하나와 정확히 같아야 한다.
def test_inspect_scores_and_calibrate_use_the_same_float32_score():
    """임계값은 float32 점수에서 나오므로, 서비스도 같은 float32 값으로 판정해야 한다."""
    insp = Inspector(FakeSession(), _bank(_images(10)), _meta(reweight_k=9))
    images = _images(40, seed=11)
    offline = insp.scores(images)
    assert offline.dtype == np.float32 and len(set(offline.tolist())) > 30
    for image, score in zip(images, offline, strict=True):
        insp.set_threshold(float(score))
        served = insp.inspect(image, heatmap=False)
        assert served.score == float(score)
        assert not served.is_defect  # 규칙은 score > threshold이고, 이 점수가 곧 임계값이다
    value = insp.calibrate(list(images), alpha=0.05)
    assert value == float(np.sort(offline)[38])  # float32 점수 중 정확히 하나, 순위 ceil(41 * 0.95)
    assert value == float(np.float32(value))


# check_meta가 거절해야 하는 meta 값 목록(키는 테스트 id). 숫자 자리에 문자열·None·bool이 오거나, 양수여야 할
# 값이 0·음수·소수이거나, 격자 길이가 2가 아니거나, NaN·inf인 경우다. bool은 파이썬에서 int의 하위형이라 따로
# 막아야 한다. sigma 8.0은 흐림 반경 int(32.5) = 32가 32px 입력 이상이 되어 점수 맵을 만들 수 없는 경우다.
BAD_META = {
    "threshold NaN": {"threshold": float("nan")},
    "threshold inf": {"threshold": float("inf")},
    "threshold None": {"threshold": None},
    "threshold text": {"threshold": "0.5"},
    "threshold bool": {"threshold": True},
    "grid one entry": {"grid": [GRID]},
    "grid three entries": {"grid": [GRID, GRID, GRID]},
    "grid zero": {"grid": [0, 0]},
    "grid negative": {"grid": [GRID, -GRID]},
    "grid fractional": {"grid": [4.5, 4]},
    "grid not a list": {"grid": GRID},
    "img_size zero": {"img_size": 0},
    "img_size text": {"img_size": str(SIZE)},
    "img_size fractional": {"img_size": 32.5},
    "dim zero": {"dim": 0},
    "dim text": {"dim": "3"},
    "reweight_k negative": {"reweight_k": -3},
    "reweight_k fractional": {"reweight_k": 2.5},
    "sigma NaN": {"sigma": float("nan")},
    "sigma negative": {"sigma": -1.0},
    "sigma text": {"sigma": "1"},
    "sigma too large for the input": {"sigma": 8.0},  # 32px 맵에 흐림 반경 32
    "version as text": {"version": str(inspector.ARTIFACT_VERSION)},
}


# 잘못된 meta 값마다 생성자, check_meta, save_artifact 세 곳이 모두 ValueError를 내고, save_artifact는 검사를
# 먼저 해서 meta.json을 아예 쓰지 않는지 본다. 잘못된 아티팩트가 디스크에 남아 나중에 서비스가 읽는 일을
# 막는다.
@pytest.mark.parametrize("change", BAD_META.values(), ids=BAD_META.keys())
def test_bad_meta_values_are_refused_with_a_value_error(change, tmp_path):
    bank = _bank(_images(4))
    with pytest.raises(ValueError):
        Inspector(FakeSession(), bank, _meta(**change))
    with pytest.raises(ValueError):
        inspector.check_meta(_meta(**change))
    # 잘못된 meta는 디스크에도 쓰이지 않는다.
    model = tmp_path / "source.onnx"
    model.write_bytes(b"x")
    with pytest.raises(ValueError):
        inspector.save_artifact(tmp_path / "art", onnx_fp32=model, bank=bank, meta=_meta(**change))
    assert not (tmp_path / "art" / "meta.json").exists()


# 경계에 있지만 올바른 값은 받아들이는지 본다(검증이 지나치게 엄격하지 않은지): 임계값 0·음수, sigma 0(흐림
# 없음), sigma 7.8(반경 31 < 32), reweight_k 0(재가중 없음), 격자 튜플, numpy 정수 img_size. 받아들인 뒤에는
# 파이썬 int·float로 바뀌어 있어야 한다.
def test_valid_meta_variants_are_accepted():
    bank = _bank(_images(4))
    for change in (
        {"threshold": 0},
        {"threshold": -1.5},
        {"sigma": 0},
        {"sigma": 7.8},  # 흐림 반경 int(31.7) = 31 < 32
        {"reweight_k": 0},
        {"grid": (GRID, GRID)},
        {"img_size": np.int64(SIZE)},
    ):
        insp = Inspector(FakeSession(), bank, _meta(**change))
        assert insp.size == SIZE and insp.grid == (GRID, GRID)
        assert isinstance(insp.threshold, float) and isinstance(insp.sigma, float)


# 뱅크에 NaN·inf가 있거나, 1차원이거나, 실수가 아닌 배열이면 생성 시점에 거절하는지 본다. 깨진 뱅크는 모든
# 거리를 NaN으로 만들어 판정을 무의미하게 하므로 적재할 때 막는다.
def test_a_bank_with_non_finite_values_is_refused():
    bank = _bank(_images(4)).astype(np.float32)
    for value in (np.nan, np.inf):
        bad = bank.copy()
        bad[3, 1] = value
        with pytest.raises(ValueError, match="finite"):
            Inspector(FakeSession(), bad, _meta())
    with pytest.raises(ValueError, match="bank"):
        Inspector(FakeSession(), bank[:, 0], _meta())
    with pytest.raises(ValueError, match="bank"):
        Inspector(FakeSession(), np.array([["a", "b", "c"]]), _meta())


# onnxruntime처럼 get_inputs/get_outputs로 입출력 이름과 모양(name, shape)을 선언하는 가짜 세션. 이 두
# 메서드가 있으면 Inspector가 check_model_io로 meta와 맞춰 본다.
class DescribedSession(FakeSession):
    """onnxruntime처럼 입력과 출력을 선언하는 가짜 세션."""

    def __init__(self, inputs, outputs):
        super().__init__()
        self._inputs, self._outputs = inputs, outputs

    def get_inputs(self):
        return [SimpleNamespace(name=n, shape=s) for n, s in self._inputs]

    def get_outputs(self):
        return [SimpleNamespace(name=n, shape=s) for n, s in self._outputs]


# 모델이 선언한 입력 image [1, 3, S, S]와 출력 features [1, H, W, dim]이 meta와 맞지 않으면 생성 시점에
# 거절하는지 본다. 범주 폴더 옆에 모델 파일 하나를 두고 여러 범주가 같이 쓰는 구조라, 다른 구성의 모델·meta가
# 섞이는 경우를 첫 요청 전에 잡으려는 것이다. 문자열(심볼릭)이나 None 차원은 비교하지 않는다.
# 반복문 안의 pytest.fail은 생성자가 예외 없이 지나갔을 때만 실행되어 어느 사례가 통과해 버렸는지 이름을
# 남긴다.
def test_the_declared_model_shapes_must_fit_the_meta():
    bank = _bank(_images(4))
    image, features = ("image", [1, 3, SIZE, SIZE]), ("features", [1, GRID, GRID, 3])
    Inspector(DescribedSession([image], [features]), bank, _meta())
    # 심볼릭(str)이나 알 수 없는(None) 차원은 비교하지 않는다.
    Inspector(
        DescribedSession([("image", ["batch", 3, SIZE, SIZE])], [("features", [None, GRID, GRID, 3])]),
        bank,
        _meta(),
    )
    bad = {
        "another input size": ([("image", [1, 3, 64, 64])], [features]),
        "another grid": ([image], [("features", [1, 8, 8, 3])]),
        "another dim": ([image], [("features", [1, GRID, GRID, 5])]),
        "a fixed batch of two": ([("image", [2, 3, SIZE, SIZE])], [("features", [2, GRID, GRID, 3])]),
        "channels last": ([("image", [1, SIZE, SIZE, 3])], [features]),
        "three dimensions": ([("image", [3, SIZE, SIZE])], [features]),
        "another input name": ([("x", [1, 3, SIZE, SIZE])], [features]),
        "another output name": ([image], [("out", [1, GRID, GRID, 3])]),
    }
    for name, (inputs, outputs) in bad.items():
        with pytest.raises(ValueError, match="model"):
            Inspector(DescribedSession(inputs, outputs), bank, _meta())
            pytest.fail(f"accepted {name}")
    with pytest.raises(ValueError, match="model"):
        inspector.check_model_io(DescribedSession([image], [("features", [1, 8, 8, 3])]), _meta())


# Inspector.load가 깨진 파일(ONNX가 아닌 모델, 잘못된 JSON, 객체가 아닌 JSON, 잘못된 임계값, 읽을 수 없는
# bank.npy)을 모두 ValueError 한 종류로 바꾸고, 메시지에 원인 파일·값을 담는지 본다. onnxruntime이 없는
# 환경(serve 그룹 미설치)에서는 건너뛴다.
# meta는 모델을 열기 전에 검사하므로, threshold가 None인 경우 메시지가 모델이 아니라 threshold를 가리켜야
# 한다.
def test_load_turns_broken_files_into_value_errors(tmp_path):
    pytest.importorskip("onnxruntime")
    art = tmp_path / "art"
    art.mkdir()
    np.save(art / "bank.npy", _bank(_images(4)))
    (art / "meta.json").write_text(json.dumps(_meta()), encoding="utf-8")
    (art / "model_fp32.onnx").write_bytes(b"not a real model")
    with pytest.raises(ValueError, match="ONNX"):
        Inspector.load(art)
    (art / "meta.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="meta.json"):
        Inspector.load(art)
    (art / "meta.json").write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(ValueError, match="meta.json"):
        Inspector.load(art)
    # meta는 모델을 열기 전에 검사하므로, 메시지는 모델이 아니라 잘못된 값을 가리킨다.
    (art / "meta.json").write_text(json.dumps(_meta(threshold=None)), encoding="utf-8")
    with pytest.raises(ValueError, match="threshold"):
        Inspector.load(art)
    (art / "meta.json").write_text(json.dumps(_meta()), encoding="utf-8")
    (art / "bank.npy").write_bytes(b"not an array")
    with pytest.raises(ValueError, match="bank.npy"):
        Inspector.load(art)


# save_artifact가 문서화된 세 파일(bank.npy, meta.json, model_fp32.onnx)만 쓰고, meta에 version을 자동으로
# 넣으며, 뱅크를 그대로 저장하는지 본다. 키가 빠진 meta는 거절하고, load는 없는 폴더(missing)와 모르는
# 정밀도(fp8)를 거절해야 한다.
def test_save_artifact_writes_the_documented_files(tmp_path):
    bank = _bank(_images(4))
    model = tmp_path / "source.onnx"
    model.write_bytes(b"not a real model")
    meta = {k: v for k, v in _meta().items() if k != "version"}
    inspector.save_artifact(tmp_path / "art", onnx_fp32=model, bank=bank, meta=meta)
    assert sorted(p.name for p in (tmp_path / "art").iterdir()) == [
        "bank.npy",
        "meta.json",
        "model_fp32.onnx",
    ]
    saved = json.loads((tmp_path / "art" / "meta.json").read_text(encoding="utf-8"))
    assert saved["version"] == inspector.ARTIFACT_VERSION and saved["threshold"] == 0.5
    np.testing.assert_array_equal(np.load(tmp_path / "art" / "bank.npy"), bank)
    with pytest.raises(ValueError, match="lacks"):
        inspector.save_artifact(tmp_path / "bad", onnx_fp32=model, bank=bank, meta={"name": "x"})
    with pytest.raises(ValueError, match="missing"):
        Inspector.load(tmp_path / "nowhere")
    with pytest.raises(ValueError, match="precision"):
        Inspector.load(tmp_path / "art", precision="fp8")


# inspector 모듈을 import해도 torch가 딸려 오지 않는지 본다. 검사 서비스와 Docker 이미지는 torch 없이
# onnxruntime만으로 돌고(README 검사 서비스 절), CI도 torch 없이 돈다(.github/workflows/ci.yml). 같은 pytest
# 프로세스에서 다른 테스트가 이미 torch를 불러왔을 수 있어서 새 파이썬 프로세스(subprocess)에서 확인한다.
def test_module_does_not_import_torch():
    import subprocess
    import sys

    code = "import sys, defect_inspect.inspector; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
