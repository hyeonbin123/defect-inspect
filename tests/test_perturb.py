"""합성 교란 테스트: 항등, 값 범위와 dtype, 이동 방향, 한 장과 묶음의 일치."""

# [파일 역할] 3단계(조건 변화)의 합성 교란 함수(src/defect_inspect/perturb.py: 밝기 배율, 감마, 가우시안 흐림,
# 평행 이동, JPEG)를 검증한다.
# [왜 필요한가] 3단계는 같은 교란을 세 방법(P0, D-S, DM)과 서로 다른 입력 크기에 똑같이 걸어 오검출률 변화를
# 비교한다(docs/experiments.md '합성 교란 (공통 정의)'). 교란이 정의와 다르거나(반올림, 이동 방향, JPEG 인코더
# 설정), 입력을 몰래 바꾸거나, 한 장·묶음 처리에서 결과가 다르면 방법 사이 비교가 무너진다. README의
# 결론('밝기만 바꿔 보는 점검으로는 실제 조명 변화를 어림할 수 없다' 등)도 이 함수들의 정의 위에 서 있다.
# [주요 구성] 항등(교란 없음) → 모양·dtype·입력 보존·값 범위 → 교란별 값 검증(밝기·감마·흐림·JPEG) → 이동
# 방향과 가장자리 처리, 마스크 이동 → 한 장과 묶음의 일치, 이름 분기 → 입력·매개변수 검증.

from __future__ import annotations

import io
import warnings

# PIL은 JPEG 왕복의 기준 결과를 직접 만들 때, scipy.ndimage는 흐림의 기준 결과를 만들 때 쓴다.
import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from defect_inspect import perturb

# 테스트 이미지 한 변(24px). 실제 입력 크기(256 등)보다 작게 잡아 빨리 돈다.
S = 24


# seed로 정해지는 무작위 uint8 배열. 모든 값이 고르게 나와 교란이 '바뀌었는지'를 확인하기 좋다.
def _noise(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)


# 이미지·채널마다 다른 매끄러운 사인 그라데이션. 실제 사진처럼 이웃 화소가 비슷해 JPEG 손실이 작게 나오는
# 입력이다(잡음 이미지는 JPEG 오차가 커서 품질 비교가 흐려진다).
def _smooth(n: int = 3, size: int = 32) -> np.ndarray:
    """이미지·채널마다 다른 매끄러운 색 그라데이션."""
    y, x = np.mgrid[0:size, 0:size] / (size - 1)
    images = np.empty((n, size, size, 3), dtype=np.uint8)
    for i in range(n):
        for c in range(3):
            images[i, :, :, c] = np.rint(
                255 * (0.5 + 0.5 * np.sin(2.0 * x * (i + 1) + 3.0 * y * (c + 1))) * 0.9
            )
    return images


# 0~255 모든 값을 한 번씩 담은 16×16 이미지(세 채널 같음). 화소별 변환표(LUT)를 쓰는 밝기·감마가 모든
# 입력값에서 맞는지 한 번에 확인할 수 있다.
def _all_values() -> np.ndarray:
    """모든 채널에 uint8 값 전부를 하나씩 담은 [16, 16, 3] 이미지."""
    return np.repeat(np.arange(256, dtype=np.uint8).reshape(16, 16, 1), 3, axis=2)


# 항등이 아닌 (교란, 값) 조합. 어둡게·밝게, 감마 양방향, 흐림, 양·음 이동, JPEG를 고루 넣어 공통 성질 테스트에
# 쓴다.
CASES = [
    ("brightness", 0.6),
    ("brightness", 1.4),
    ("gamma", 0.7),
    ("gamma", 1.5),
    ("blur", 1.5),
    ("shift", 2),
    ("shift", -3),
    ("jpeg", 50),
]


# ---------------------------------------------------------------------------------------------------------
# 항등(교란 없음)
# ---------------------------------------------------------------------------------------------------------


# 교란 없음에 해당하는 값(밝기 1, 감마 1, 흐림 σ 0, 이동 0)이 입력과 똑같은 결과를 내되 같은 메모리를 공유하지
# 않는 새 배열인지 본다. 모듈 docstring의 약속('항상 새 배열, 입력은 바꾸지 않음')이고, '교란 없음' 조건이
# 실제 교란 조건과 같은 코드 경로를 지나도 기준선과 같아야 한다. apply 경로도 함께 확인한다.
@pytest.mark.parametrize("shape", [(S, S, 3), (4, S, S, 3)])
def test_identity_parameters_return_an_equal_copy(shape: tuple[int, ...]) -> None:
    images = _noise(shape)
    original = images.copy()
    results = [
        perturb.brightness(images, 1.0),
        perturb.gamma(images, 1.0),
        perturb.gaussian_blur(images, 0.0),
        perturb.shift(images, 0, 0),
        perturb.apply(images, "brightness", 1),
        perturb.apply(images, "gamma", 1),
        perturb.apply(images, "blur", 0),
        perturb.apply(images, "shift", 0),
    ]
    for out in results:
        assert out.dtype == np.uint8
        assert out.shape == images.shape
        assert np.array_equal(out, original)
        assert not np.shares_memory(out, images)
    assert np.array_equal(images, original)


# 밝기 1·감마 1이 0~255 모든 값에서 정확히 항등인지 본다. 변환표의 반올림이 한 값이라도 바꾸면 '교란 없음'이
# 기준선과 달라진다.
def test_identity_holds_for_every_pixel_value() -> None:
    img = _all_values()
    assert np.array_equal(perturb.brightness(img, 1.0), img)
    assert np.array_equal(perturb.gamma(img, 1.0), img)


# 마스크 이동 0도 같은 값의 새 배열을 돌려주는지 본다.
def test_shift_mask_identity() -> None:
    masks = (_noise((3, 10, 12), seed=1) > 128).astype(np.uint8)
    out = perturb.shift_mask(masks, 0, 0)
    assert np.array_equal(out, masks)
    assert not np.shares_memory(out, masks)


# ---------------------------------------------------------------------------------------------------------
# 모양, dtype, 값 범위, 입력 보존
# ---------------------------------------------------------------------------------------------------------


# 모든 교란이 세 가지 모양(한 장, 묶음, 정사각형이 아닌 묶음)에서 uint8·같은 모양을 돌려주고, 입력을 바꾸지
# 않으며, 실제로 결과를 바꾸는지(헛돌지 않는지) 본다.
@pytest.mark.parametrize(("kind", "value"), CASES)
@pytest.mark.parametrize("shape", [(S, S, 3), (3, S, S, 3), (2, 12, 20, 3)])
def test_output_shape_dtype_and_input_untouched(kind: str, value: float, shape: tuple[int, ...]) -> None:
    images = _noise(shape, seed=2)
    original = images.copy()
    out = perturb.apply(images, kind, value)
    assert out.dtype == np.uint8
    assert out.shape == shape
    assert np.array_equal(images, original)
    assert not np.array_equal(out, original)


# 밝기 배율의 정확한 값: 곱한 뒤 0~255로 자르고 반올림하며, 반올림은 0.5를 짝수 쪽으로 보낸다(np.rint, 1.5→2,
# 4.5→4, 127.5→128). 배율 0은 전부 검정, 아주 큰 배율은 0만 남기고 255다.
def test_brightness_values() -> None:
    img = np.array([0, 1, 3, 100, 200, 255], dtype=np.uint8).reshape(1, 6, 1).repeat(3, axis=2)
    assert perturb.brightness(img, 1.5)[0, :, 0].tolist() == [0, 2, 4, 150, 255, 255]  # 1.5 -> 2, 4.5 -> 4
    assert perturb.brightness(img, 0.5)[0, :, 0].tolist() == [0, 0, 2, 50, 100, 128]  # .5는 짝수로 반올림
    assert perturb.brightness(img, 0.0).max() == 0
    assert perturb.brightness(img, 1000.0)[0, :, 0].tolist() == [0, 255, 255, 255, 255, 255]


# 배율 < 1은 모든 화소를 어둡게(또는 그대로), > 1은 밝게 하고, 화소값의 순서를 뒤집지 않는지(단조) 본다. 밝은
# 쪽은 255에서 잘린다. 강도 표의 밝기(0.9, 0.8, 0.7)는 모두 어둡게 하는 쪽이다.
def test_brightness_direction_and_monotonicity() -> None:
    img = _all_values()
    darker = perturb.brightness(img, 0.7).astype(int)
    brighter = perturb.brightness(img, 1.3).astype(int)
    assert np.all(darker <= img) and darker.mean() < img.mean()
    assert np.all(brighter >= img) and brighter.mean() > img.mean()
    assert brighter.max() == 255
    for out in (darker, brighter):  # 화소값의 순서가 유지된다
        assert np.all(np.diff(out[:, :, 0].ravel()) >= 0)
    expected = np.clip(np.rint(np.arange(256) * 1.3), 0, 255)
    assert np.array_equal(brighter[:, :, 1].ravel(), expected)


# 감마 곡선 255·(x/255)^g: 양 끝(0, 255)은 고정이고 단조이며, g > 1은 어둡게, g < 1은 밝게 한다. 128과 64에서
# 손 계산 값과 맞추고 전체를 정의식과 비교한다. 밝기 배율과 달리 밝은 쪽은 거의 그대로 두고 중간을 바꾸는
# 교란이라는 점이 드러난다.
def test_gamma_values_and_direction() -> None:
    img = _all_values()
    dark = perturb.gamma(img, 2.0).astype(int)
    bright = perturb.gamma(img, 0.5).astype(int)
    for out in (dark, bright):
        flat = out[:, :, 0].ravel()
        assert flat[0] == 0 and flat[255] == 255  # 양 끝은 고정
        assert np.all(np.diff(flat) >= 0)
    assert np.all(dark <= img) and dark.mean() < img.mean()  # g > 1은 어둡게
    assert np.all(bright >= img) and bright.mean() > img.mean()  # g < 1은 밝게
    assert dark[:, :, 0].ravel()[128] == 64  # 255 * (128 / 255) ** 2 = 64.25
    assert bright[:, :, 0].ravel()[64] == 128  # 255 * sqrt(64 / 255) = 127.75
    expected = np.rint(255.0 * (np.arange(256) / 255.0) ** 2.0)
    assert np.array_equal(dark[:, :, 2].ravel(), expected)


# 흐림이 채널마다 scipy gaussian_filter(mode="nearest": 가장자리 복제)를 걸고 반올림·자르기한 것과 정확히
# 같은지 본다.
def test_blur_matches_scipy_per_channel() -> None:
    images = _noise((2, S, S, 3), seed=3)
    out = perturb.gaussian_blur(images, 1.2)
    for i in range(2):
        for c in range(3):
            ref = ndimage.gaussian_filter(images[i, :, :, c].astype(np.float64), sigma=1.2, mode="nearest")
            assert np.array_equal(out[i, :, :, c], np.clip(np.rint(ref), 0, 255).astype(np.uint8))


# 흐림은 잡음의 표준편차를 크게 줄이고 평균은 거의 그대로 두며, σ가 클수록 더 매끄럽다. 가장자리를 복제하므로
# 상수 이미지는 모서리까지 그대로여야 한다(0으로 채우는 패딩이면 테두리가 어두워진다).
def test_blur_smooths_and_keeps_constants() -> None:
    images = _noise((2, S, S, 3), seed=4)
    out = perturb.gaussian_blur(images, 2.0)
    assert out.std() < 0.5 * images.std()
    assert abs(out.mean() - images.mean()) < 3.0
    assert perturb.gaussian_blur(images, 4.0).std() < out.std()
    for value in (0, 37, 255):  # 가장자리 복제: 상수 이미지는 모서리까지 상수로 남는다
        const = np.full((S, S, 3), value, dtype=np.uint8)
        assert np.array_equal(perturb.gaussian_blur(const, 3.0), const)


# 흐림이 채널 축과 이미지 축으로 번지지 않는지 본다. 검정·흰색 이미지 묶음은 그대로여야 하고, 빨강 채널의
# 사각형은 빨강 채널 안에서만 퍼져야 한다(구현의 sigma=(σ, σ, 0)).
def test_blur_does_not_mix_channels_or_images() -> None:
    batch = np.zeros((2, S, S, 3), dtype=np.uint8)
    batch[1] = 255  # 검정 이미지 한 장과 흰 이미지 한 장
    assert np.array_equal(perturb.gaussian_blur(batch, 3.0), batch)
    red = np.zeros((S, S, 3), dtype=np.uint8)
    red[8:16, 8:16, 0] = 255
    out = perturb.gaussian_blur(red, 2.0)
    assert out[:, :, 0].max() < 255 and out[4, 12, 0] > 0  # 빨강 채널 안에서만 퍼진다
    assert out[:, :, 1:].max() == 0


# JPEG는 손실이 있지만(품질 95에서도 오차 > 0) 원본과 가깝고, 품질이 낮을수록 오차가 크며, 같은 입력에 항상
# 같은 결과를 낸다. apply에 실수로 품질을 줘도 정수로 바꿔 같은 결과를 낸다.
def test_jpeg_is_lossy_close_and_deterministic() -> None:
    images = _smooth()
    high = perturb.jpeg(images, 95)
    low = perturb.jpeg(images, 10)
    err_high = np.abs(high.astype(int) - images).mean()
    err_low = np.abs(low.astype(int) - images).mean()
    assert 0 < err_high < 3.0
    assert err_high < err_low < 30.0
    assert np.array_equal(low, perturb.jpeg(images, 10))
    assert np.array_equal(perturb.apply(images, "jpeg", 10.0), low)


# PIL로 JPEG 저장 → 다시 읽기를 직접 하는 기준 함수. save_options로 서브샘플링 같은 인코더 설정을 바꿔 비교할
# 수 있다.
def _pil_round_trip(img: np.ndarray, quality: int, **save_options: object) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=quality, **save_options)
    buf.seek(0)
    with Image.open(buf) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8)


# perturb.jpeg가 PIL 기본 설정(모든 품질에서 4:2:0 색차 서브샘플링)의 단순 왕복과 같은지 여러 품질에서
# 고정한다. 인코더 설정이 바뀌면 같은 'JPEG 품질 70'이 다른 교란이 되어 이전 결과와 비교할 수 없으므로,
# 4:2:0을 명시한 결과와는 같고 4:4:4(subsampling=0)와는 달라야 한다. 정사각형이 아닌 홀수 크기도 확인한다.
@pytest.mark.parametrize("quality", [5, 30, 50, 75, 90, 100])
def test_jpeg_is_the_plain_pil_round_trip(quality: int) -> None:
    images = np.concatenate([_smooth(2, 32), _noise((2, 32, 32, 3), seed=9)])
    out = perturb.jpeg(images, quality)
    for i, img in enumerate(images):
        assert np.array_equal(out[i], _pil_round_trip(img, quality)), i
        assert np.array_equal(perturb.jpeg(img, quality), out[i]), i
        assert np.array_equal(_pil_round_trip(img, quality, subsampling="4:2:0"), out[i]), i
        assert not np.array_equal(_pil_round_trip(img, quality, subsampling=0), out[i]), i
    # 정사각형이 아닌 홀수 크기도 단순 왕복과 같다.
    odd = _noise((13, 21, 3), seed=10)
    assert np.array_equal(perturb.jpeg(odd, quality), _pil_round_trip(odd, quality))


# 4:2:0이 이미지에 실제로 어떤 뜻인지 보여 주는 테스트다. 밝기는 비슷하고 색만 다른 한 화소 체크무늬는 색차를
# 절반 해상도로 줄이면 색이 뭉개지므로, 기본 설정의 오차가 4:4:4보다 3배 넘게 커야 한다.
def test_jpeg_blurs_colour_more_than_a_444_encoding() -> None:
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:32, 0:32]
    img[(yy + xx) % 2 == 0] = (200, 60, 60)
    img[(yy + xx) % 2 == 1] = (60, 110, 200)
    err_default = np.abs(perturb.jpeg(img, 95).astype(int) - img).mean()
    err_444 = np.abs(_pil_round_trip(img, 95, subsampling=0).astype(int) - img).mean()
    assert err_default > 3 * err_444


# ---------------------------------------------------------------------------------------------------------
# 이동(shift)
# ---------------------------------------------------------------------------------------------------------


# (y, x) 한 화소만 색이 있는 검정 이미지. 이동 방향을 화소 위치로 확인하는 데 쓴다.
def _hot(y: int, x: int, size: int = 12) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[y, x] = (10, 20, 30)
    return img


# 이동 방향 규약: dx > 0은 오른쪽(열 번호 증가), dy > 0은 아래(행 번호 증가)다. 강도 표의 '오른쪽 아래로'
# 이동이 이 규약을 따른다. 같은 이동을 마스크에 걸면 표시 화소가 같은 곳으로 가야 한다.
@pytest.mark.parametrize(("dx", "dy"), [(2, 0), (0, 3), (2, 3), (-4, 0), (0, -1), (-2, 5), (1, -3)])
def test_shift_direction(dx: int, dy: int) -> None:
    y, x = 5, 6
    out = perturb.shift(_hot(y, x), dx, dy)
    assert np.array_equal(out, _hot(y + dy, x + dx))

    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[y, x] = 1
    moved = perturb.shift_mask(mask, dx, dy)
    assert moved.sum() == 1 and moved[y + dy, x + dx] == 1


# 이동으로 빈 쪽은 가장자리 줄을 복제해 채운다(0으로 채우면 원래 없던 검은 띠가 생겨 그 자체가 이상으로
# 잡힌다). 이미지보다 큰 이동은 가장자리 줄만 남는다.
def test_shift_replicates_edges() -> None:
    rows, cols = np.mgrid[0:6, 0:8]
    img = np.stack([cols * 10, rows * 10, cols + rows], axis=-1).astype(np.uint8)

    right = perturb.shift(img, 3, 0)
    assert np.array_equal(right[:, 3:], img[:, :-3])
    assert np.array_equal(right[:, :3], np.repeat(img[:, :1], 3, axis=1))

    up = perturb.shift(img, 0, -2)
    assert np.array_equal(up[:-2], img[2:])
    assert np.array_equal(up[-2:], np.repeat(img[-1:], 2, axis=0))

    # 이미지보다 큰 이동은 복제한 가장자리 줄만 남긴다.
    assert np.array_equal(perturb.shift(img, 100, 0), np.repeat(img[:, :1], 8, axis=1))
    assert np.array_equal(perturb.shift(img, 0, -100), np.repeat(img[-1:], 6, axis=0))


# 가로·세로 이동을 나눠 해도 같고, 가장자리를 뺀 안쪽은 np.roll과 같은지 본다. apply의 shift는 dx = dy =
# value로 대각선 이동이며, 값은 int()로 0 쪽으로 잘린다(-2.9 → -2). 입력 크기에 맞춘 반올림은 apply가 아니라
# conditions.py가 먼저 한다.
def test_shift_is_separable_and_matches_roll_inside() -> None:
    img = _noise((S, S, 3), seed=5)
    out = perturb.shift(img, 2, -3)
    assert np.array_equal(out, perturb.shift(perturb.shift(img, 2, 0), 0, -3))
    rolled = np.roll(img, shift=(-3, 2), axis=(0, 1))
    assert np.array_equal(out[:-3, 2:], rolled[:-3, 2:])
    assert np.array_equal(perturb.apply(img, "shift", 2.0), perturb.shift(img, 2, 2))
    assert np.array_equal(perturb.apply(img, "shift", -2.9), perturb.shift(img, -2, -2))  # int()는 버림


# 마스크 이동은 빈 곳을 0(정상)으로 채운다. 마스크까지 가장자리를 복제하면 없던 결함이 생기기 때문이다. 크기
# 이상 이동은 빈 마스크가 된다. 뒤쪽은 마스크를 그린 이미지를 shift한 결과와 shift_mask 결과가 같은 위치에
# 오는지 확인해 두 함수의 이동 규약이 같음을 보인다(가장자리에서 6px 떨어진 곳에만 그려 복제 차이가 끼지 않게
# 한다).
def test_shift_mask_zero_fills_and_follows_the_image() -> None:
    mask = np.ones((6, 8), dtype=np.uint8)
    out = perturb.shift_mask(mask, 3, -2)
    expected = np.zeros((6, 8), dtype=np.uint8)
    expected[:4, 3:] = 1
    assert np.array_equal(out, expected)
    assert perturb.shift_mask(mask, 8, 0).sum() == 0
    assert perturb.shift_mask(mask, 0, -6).sum() == 0
    assert perturb.shift_mask(mask, -7, 5).sum() == 1

    # `shift`와 같은 이동: 이미지에 그린 마스크가 이동한 마스크 위치에 온다.
    rng = np.random.default_rng(6)
    masks = np.zeros((3, S, S), dtype=np.uint8)
    masks[:, 6:-6, 6:-6] = rng.integers(0, 2, size=(3, S - 12, S - 12))
    images = np.repeat(masks[..., None] * 255, 3, axis=-1).astype(np.uint8)
    for dx, dy in [(2, 2), (-3, 1), (0, -4), (5, 0)]:
        moved = perturb.shift(images, dx, dy)
        assert np.array_equal(moved[..., 0] > 0, perturb.shift_mask(masks, dx, dy) > 0)


# shift_mask가 uint8·bool·float32 마스크와 [N, H, W] 묶음을 받아 dtype·모양을 유지하고, 묶음 결과가 한 장씩
# 처리한 것과 같으며, 입력을 바꾸지 않는지 본다.
@pytest.mark.parametrize("dtype", [np.uint8, np.bool_, np.float32])
def test_shift_mask_shapes_and_dtypes(dtype: type) -> None:
    masks = (_noise((4, 10, 14), seed=7) > 100).astype(dtype)
    original = masks.copy()
    out = perturb.shift_mask(masks, 2, -1)
    assert out.dtype == masks.dtype
    assert out.shape == masks.shape
    assert np.array_equal(masks, original)
    for i in range(4):
        assert np.array_equal(out[i], perturb.shift_mask(masks[i], 2, -1))
    assert np.array_equal(out[:, :-1, 2:], masks[:, 1:, :-2])
    assert not out[:, -1, :].any() and not out[:, :, :2].any()


# ---------------------------------------------------------------------------------------------------------
# [S, S, 3]과 [N, S, S, 3]이 같은 결과를 낸다
# ---------------------------------------------------------------------------------------------------------


# [N, S, S, 3] 묶음을 한 번에 교란한 결과가 한 장씩 교란한 결과와 같은지 모든 교란에서 본다. 실행 스크립트는
# 묶음으로 교란하므로, 이미지끼리 섞이거나(흐림) 순서가 바뀌면 안 된다.
@pytest.mark.parametrize(("kind", "value"), CASES)
def test_single_image_and_batch_agree(kind: str, value: float) -> None:
    batch = np.concatenate([_noise((2, 32, 32, 3), seed=8), _smooth(2, 32)])
    out = perturb.apply(batch, kind, value)
    for i in range(len(batch)):
        assert np.array_equal(out[i], perturb.apply(batch[i], kind, value)), (kind, i)
    assert np.array_equal(perturb.apply(batch[1][None], kind, value)[0], out[1])


# apply가 이름별로 맞는 함수를 부르는지, 모르는 이름(contrast)은 ValueError인지 본다. conditions.py가 이
# 이름으로 교란을 고른다.
def test_apply_dispatches_to_the_named_function() -> None:
    images = _smooth(2, 16)
    assert np.array_equal(perturb.apply(images, "brightness", 1.2), perturb.brightness(images, 1.2))
    assert np.array_equal(perturb.apply(images, "gamma", 0.8), perturb.gamma(images, 0.8))
    assert np.array_equal(perturb.apply(images, "blur", 1.0), perturb.gaussian_blur(images, 1.0))
    assert np.array_equal(perturb.apply(images, "shift", 2), perturb.shift(images, 2, 2))
    assert np.array_equal(perturb.apply(images, "jpeg", 30), perturb.jpeg(images, 30))
    with pytest.raises(ValueError):
        perturb.apply(images, "contrast", 1.0)


# ---------------------------------------------------------------------------------------------------------
# 입력 검증
# ---------------------------------------------------------------------------------------------------------


# uint8이 아닌 이미지는 TypeError, [S, S, 3]·[N, S, S, 3]이 아닌 모양은 ValueError다. float 이미지가 들어오면
# 변환표 인덱싱과 반올림 규칙이 깨지므로 일찍 막는다. 1차원 마스크도 거절한다.
def test_rejects_wrong_dtype_and_shape() -> None:
    with pytest.raises(TypeError):
        perturb.brightness(np.zeros((S, S, 3), dtype=np.float32), 1.0)
    with pytest.raises(TypeError):
        perturb.shift(np.zeros((S, S, 3), dtype=np.int64), 1, 1)
    for bad in [(S, S), (S, S, 1), (2, 2, S, S, 3), (3, S, S)]:
        with pytest.raises(ValueError):
            perturb.gaussian_blur(np.zeros(bad, dtype=np.uint8), 1.0)
    with pytest.raises(ValueError):
        perturb.shift_mask(np.zeros(5, dtype=np.uint8), 1, 1)


# 의미 없는 매개변수를 거절한다: 음수 밝기, 감마 0, 음수 흐림, JPEG 품질 1~100 밖, NaN 밝기.
def test_rejects_bad_parameters() -> None:
    img = np.zeros((S, S, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        perturb.brightness(img, -0.1)
    with pytest.raises(ValueError):
        perturb.gamma(img, 0.0)
    with pytest.raises(ValueError):
        perturb.gaussian_blur(img, -1.0)
    with pytest.raises(ValueError):
        perturb.jpeg(img, 0)
    with pytest.raises(ValueError):
        perturb.jpeg(img, 101)
    with pytest.raises(ValueError):
        perturb.brightness(img, float("nan"))


# 무한대·NaN 밝기와 흐림을 경고 없이 ValueError로 거절하는지 본다. 예전에는 inf 밝기가 0 × inf = nan을
# uint8로 바꾸며 경고(RuntimeWarning)만 내고 통과했고, 흐림은 scipy 안에서 OverflowError가 났다. 경고를
# 오류로 바꿔 그런 경고가 다시 새지 않는지도 함께 확인한다.
@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -float("inf"), np.float32("inf"), np.inf])
def test_rejects_non_finite_brightness_and_blur(bad: float) -> None:
    images = _noise((2, S, S, 3), seed=11)
    images[0, 0, 0] = 0  # 0 * inf를 정의되지 않게 만든 것이 값 0인 화소다
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="factor"):
            perturb.brightness(images, bad)
        with pytest.raises(ValueError, match="sigma"):
            perturb.gaussian_blur(images, bad)
        with pytest.raises(ValueError, match="factor"):
            perturb.apply(images, "brightness", bad)
        with pytest.raises(ValueError, match="sigma"):
            perturb.apply(images, "blur", bad)


# 반대로 매우 크지만 유한한 값은 정상 처리해야 한다(검증이 지나치지 않은지): 밝기 1e300은 0이 아닌 화소를 모두
# 255로, σ 50 흐림은 이미지 크기보다 넓어도 uint8 결과를 낸다.
def test_large_finite_parameters_still_work() -> None:
    images = _noise((S, S, 3), seed=12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.array_equal(perturb.brightness(images, 1e300), np.where(images > 0, 255, 0))
        wide = perturb.gaussian_blur(images, 50.0)
        assert wide.dtype == np.uint8 and wide.shape == images.shape
        assert wide.std() < 0.6 * images.std()
