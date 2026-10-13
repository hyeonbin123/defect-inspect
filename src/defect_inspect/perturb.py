"""Synthetic image perturbations on uint8 RGB arrays shaped [S, S, 3] or [N, S, S, 3].

Every function returns a new uint8 array of the same shape and never modifies its input.
"""

# [파일 역할] 3단계(조건 변화)에서 쓰는 합성 교란 다섯 종류(밝기 배율, 감마, 가우시안 흐림, 평행 이동, JPEG
# 압축)를 uint8 RGB 배열에 거는 순수 함수 모음이다.
# [왜 필요한가] 촬영 조건이 조금 바뀌었을 때 AUROC는 그대로인데 고정 임계값의 오검출률만 뛰는지(질문 2 뒷부분,
# H9), 그리고 합성 교란이 실제 조명 변화(M2AD)를 대신할 수 있는지(질문 3)를 재려면 같은 교란을 모든 방법·모든
# 입력 크기에 똑같이 걸어야 한다. AGPL 증강 라이브러리를 쓰지 않으므로(docs/plan.md '하지 않는 것')
# PIL/numpy/scipy로 직접 만든다.
# [설계 이유] 모든 함수가 입력을 바꾸지 않고 새 uint8 배열을 돌려준다. 강도 표(docs/experiments.md 3단계 '합성
# 교란 (공통 정의)')와 입력 크기에 맞춘 환산은 conditions.py가 맡고, 이 파일은 '주어진 값으로 교란을 거는
# 일'만 한다.
# [연결] conditions.apply_condition이 apply()를 부르고, run_perturb·run_m2ad·dinomaly_serving 등은
# conditions를 거쳐 이 함수들을 쓴다.

from __future__ import annotations

import io
import math

# numpy: 픽셀 값 변환표(LUT)와 배열 인덱싱. PIL: JPEG 인코딩·디코딩 왕복. scipy.ndimage: 가장자리를 복제하는
# 가우시안 필터.
import numpy as np
from PIL import Image
from scipy import ndimage

# 지원하는 교란 이름. conditions.LEVELS의 키와 같은 순서이고, apply()가 모르는 이름을 받으면 이 목록을 오류
# 메시지에 보여 준다.
KINDS = ("brightness", "gamma", "blur", "shift", "jpeg")


# 입력이 uint8이고 [S, S, 3] 또는 [N, S, S, 3] 모양인지 확인한다. float 이미지가 섞여 들어오면 LUT 인덱싱과
# 반올림 규칙이 깨지므로 일찍 막는다.
def _check_images(images: np.ndarray) -> np.ndarray:
    arr = np.asarray(images)
    if arr.dtype != np.uint8:
        raise TypeError(f"images must be uint8, got {arr.dtype}")
    if arr.ndim not in (3, 4) or arr.shape[-1] != 3:
        raise ValueError(f"images must be [S, S, 3] or [N, S, S, 3], got shape {arr.shape}")
    return arr


# 픽셀마다 같은 함수를 적용하는 교란(밝기, 감마)을 256칸 변환표 하나로 처리한다. 값 0~255마다 결과를 한 번만
# 계산하고 배열 인덱싱(lut[images])으로 전체에 적용하므로 이미지 수와 무관하게 계산이 싸다.
# np.rint는 0.5를 짝수 쪽으로 반올림한다(docstring의 round half to even).
def _lookup(images: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Apply a pointwise map given as 256 float values (rounded half to even, clipped to 0..255)."""
    lut = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    return lut[_check_images(images)]


# 밝기 배율 교란: 픽셀 값에 factor를 곱한다(강도 표 0.9, 0.8, 0.7 = 10~30% 어둡게). 노출이 바뀐 상황을 흉내
# 낸다. 1보다 크면 밝아지고 255에서 잘린다.
# README 결과: WideResNet PatchCore는 20~30% 어둡게 했을 때 AUROC는 그대로인데 오검출률이 15%, 41%로 뛰었다.
def brightness(images: np.ndarray, factor: float) -> np.ndarray:
    """Multiply pixel values by ``factor``, clip to 0..255 and round."""
    factor = float(factor)
    # NaN·무한대·음수 배율은 의미 있는 교란이 아니므로 거절한다.
    if not (math.isfinite(factor) and factor >= 0):
        raise ValueError(f"factor must be finite and non-negative, got {factor}")
    return _lookup(images, np.arange(256, dtype=np.float64) * factor)


# 감마 교란: 255 * (x / 255) ** g. 밝기 배율과 달리 어두운 쪽과 밝은 쪽을 다르게 바꿔 조명 곡선이 달라진
# 상황을 흉내 낸다(강도 표 1.1, 1.25, 1.5로 모두 어둡게).
def gamma(images: np.ndarray, g: float) -> np.ndarray:
    """Gamma curve ``255 * (x / 255) ** g`` (g > 1 darkens, g < 1 brightens)."""
    if not g > 0:
        raise ValueError(f"g must be positive, got {g}")
    return _lookup(images, 255.0 * (np.arange(256, dtype=np.float64) / 255.0) ** float(g))


# 가우시안 흐림 교란: 초점이 조금 흐려진 상황을 흉내 낸다. sigma는 이미 입력 크기에 맞춰 환산된
# 값(conditions.strength가 S÷256을 곱함)으로 받는다.
# mode="nearest"로 가장자리 픽셀을 복제해 테두리가 검게 번지지 않게 한다.
def gaussian_blur(images: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur over the two spatial axes of each image and channel, edges replicated."""
    arr = _check_images(images)
    sigma = float(sigma)
    if not (math.isfinite(sigma) and sigma >= 0):
        raise ValueError(f"sigma must be finite and non-negative, got {sigma}")
    # sigma 0은 '교란 없음'이므로 필터를 돌리지 않고 복사본만 돌려준다(입력을 바꾸지 않는 규칙 유지).
    if sigma == 0:
        return arr.copy()
    # [S, S, 3]과 [N, S, S, 3]을 같은 코드로 다루려고 [N, S, S, 3]으로 펼친 뒤 이미지마다 거른다. float64로
    # 계산하고 반올림·자르기로 uint8로 되돌린다.
    flat = arr.reshape((-1, *arr.shape[-3:]))
    out = np.empty_like(flat)
    for i, img in enumerate(flat):
        # sigma 0 on the channel axis: channels are filtered independently.
        blurred = ndimage.gaussian_filter(img.astype(np.float64), sigma=(sigma, sigma, 0), mode="nearest")
        out[i] = np.clip(np.rint(blurred), 0, 255).astype(np.uint8)
    return out.reshape(arr.shape)


# 평행 이동 교란: 정수 픽셀만큼 오른쪽(dx>0)·아래(dy>0)로 민다. 비는 쪽은 가장자리 줄을 복제해 채운다(빈 곳을
# 0으로 채우면 원래 없던 검은 띠가 생긴다). 지그 위치가 조금 어긋난 상황을 흉내 낸다.
# 구현: 각 출력 행·열이 읽을 원본 인덱스를 (i - d)로 계산하고 0..크기-1로 잘라 np.take로 가져온다. 잘린
# 인덱스가 가장자리 복제가 된다.
# README 결과: DINOv2 특징을 쓰는 두 방법은 256px 기준 1~2px 어긋남에 크게 무너졌다.
def shift(images: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate by whole pixels (positive dx: right, positive dy: down), replicating the edges."""
    arr = _check_images(images)
    height, width = arr.shape[-3], arr.shape[-2]
    rows = np.clip(np.arange(height) - int(dy), 0, height - 1)
    cols = np.clip(np.arange(width) - int(dx), 0, width - 1)
    return np.take(np.take(arr, rows, axis=-3), cols, axis=-2)


# 결함 마스크를 shift와 같은 방향으로 옮기되, 비는 영역은 0(정상)으로 채운다. 마스크는 가장자리를 복제하면
# 없던 결함이 생기므로 이미지와 다르게 처리한다.
# 이 저장소에서는 tests/test_perturb.py만 이 함수를 부른다(3-A는 픽셀 지표를 재지 않는다는 규칙,
# docs/experiments.md).
def shift_mask(masks: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate [..., H, W] masks like ``shift``; the vacated area becomes 0."""
    arr = np.asarray(masks)
    if arr.ndim < 2:
        raise ValueError(f"masks must be [..., H, W], got shape {arr.shape}")
    dx, dy = int(dx), int(dy)
    height, width = arr.shape[-2], arr.shape[-1]
    out = np.zeros_like(arr)
    # 이동 거리가 크기 이상이면 원본이 하나도 남지 않으므로 전부 0인 마스크를 돌려준다.
    if abs(dx) >= width or abs(dy) >= height:
        return out
    # 겹치는 영역만 복사한다: 양수 이동이면 원본 앞쪽을 출력 뒤쪽에, 음수 이동이면 그 반대로 붙인다. 행과 열에
    # 같은 규칙을 쓴다.
    dst_rows = slice(max(dy, 0), height + min(dy, 0))
    src_rows = slice(max(-dy, 0), height - max(dy, 0))
    dst_cols = slice(max(dx, 0), width + min(dx, 0))
    src_cols = slice(max(-dx, 0), width - max(dx, 0))
    out[..., dst_rows, dst_cols] = arr[..., src_rows, src_cols]
    return out


# 이미지 한 장을 메모리 버퍼에 JPEG로 저장했다가 다시 읽어 압축 손실만 남긴다. 디스크를 쓰지 않으므로
# HDD에서도 빠르다. np.ascontiguousarray는 슬라이스된 배열도 PIL이 받을 수 있게 연속 메모리로 만든다.
def _jpeg_one(img: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8)


# JPEG 압축 교란(강도 표 품질 90, 70, 50). 전송·저장 과정의 압축 손실을 흉내 낸다. 품질 외 설정은 PIL
# 기본값이라 결과가 PIL 버전에 묶인다.
def jpeg(images: np.ndarray, quality: int) -> np.ndarray:
    """JPEG encode/decode round trip with PIL at the given quality (PIL defaults otherwise)."""
    arr = _check_images(images)
    quality = int(quality)
    if not 1 <= quality <= 100:
        raise ValueError(f"quality must be in 1..100, got {quality}")
    flat = arr.reshape((-1, *arr.shape[-3:]))
    out = np.empty_like(flat)
    for i, img in enumerate(flat):
        out[i] = _jpeg_one(img, quality)
    return out.reshape(arr.shape)


# 이름으로 교란 함수를 고르는 진입점. conditions.apply_condition이 크기 환산을 마친 값(value)과 함께 부른다.
# shift는 dx = dy = value로 오른쪽 아래 대각선으로 민다(강도 표의 '오른쪽 아래로').
def apply(images: np.ndarray, kind: str, value: float) -> np.ndarray:
    """Dispatch by name: brightness | gamma | blur | shift | jpeg.

    ``shift`` moves by (dx=int(value), dy=int(value)); ``jpeg`` uses quality int(value).
    """
    if kind == "brightness":
        return brightness(images, float(value))
    if kind == "gamma":
        return gamma(images, float(value))
    if kind == "blur":
        return gaussian_blur(images, float(value))
    if kind == "shift":
        return shift(images, int(value), int(value))
    if kind == "jpeg":
        return jpeg(images, int(value))
    raise ValueError(f"unknown perturbation {kind!r}; expected one of {KINDS}")
