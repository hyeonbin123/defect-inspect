"""업로드 이미지의 구조 사전 검사: Pillow가 이미지를 열기 전에 바이트열을 직접 훑어, 알려진 증폭 경로의 구조
항목이 상한을 넘으면 거절한다.

`check_image_structure(data)`는 PNG·JPEG·TIFF·BMP의 구조 항목을 세어 형식별 상한을 넘으면
`ImageStructureError`를 낸다. 세는 것은 TIFF의 IFD 항목·스트립·타일 표·숫자 배열(BYTE형으로 적었어도 Pillow와
libtiff가 원소마다 도는 태그 포함)과 따라가는 IFD 수, JPEG의 첫 스캔 앞 마커 세그먼트·머리 길이·쓰레기
바이트와 스캔(SOS) 수, PNG의 청크 수와 압축을 푸는 청크(iCCP·zTXt·iTXt) 수, RLE로 압축한 BMP의 파일 크기다.
상한 안이면 아무것도 돌려주지 않고 바이트열을 바꾸지 않는다. WebP와 알아볼 수 없는 형식은 건드리지
않는다(Pillow가 판단한다).

왜 필요한가: 서비스는 업로드를 파일 크기(20MiB)와 화소 수(2,500만)로 묶지만, Pillow는 화소 수와 무관한 구조
항목마다 일을 한다. 무압축 TIFF는 StripOffsets 원소마다 `_Tile` 하나를, 하위 IFD의 숫자 배열은 원소마다 파이썬
수를 만든다. JPEG는 첫 스캔 앞의 마커마다 객체를 쌓고, 점진 JPEG는 스캔마다 libjpeg가 그림 전체를 돈다. PNG는
청크마다 객체를 쌓고 압축 청크마다 1MiB까지 풀며, BMP RLE는 파이썬 디코더가 명령마다 한 바퀴를 돈다. 그래서 폭
1 × 높이 N 같은 그림이나 작은 파일 하나가 작업 스레드 하나를 수 초~수십 초(외삽하면 수십 분) 잡고 메모리를
수백 MiB~수 GiB 쓸 수 있었다(측정은 docs/security.md). 화소 수 검사는 `Image.open`이 끝난 뒤라 늦다.

설계: (1) 경로마다 Pillow 13.0.0과 그 밑의 libjpeg-turbo·libtiff가 읽는 규칙을 따라가며 센다. 근거가 된 소스
위치는 각 상수와 함수의 주석에 있다(행 번호는 Pillow 13.0.0 기준). 어느 쪽을 읽을지 이 검사가 정확히 가를 수
없으면 더 넓게 센다(예: 같은 가리킴 태그가 여럿이면 모두 따라간다). (2) 거절만 한다. 통과한 바이트열은 그대로
Pillow에 넘어가므로 디코딩 결과가 고치기 전과 같다. 무압축 TIFF를 libtiff로 읽게 하는
`TiffImagePlugin.READ_LIBTIFF`는 스트립 표 문제를 없애지만 YCbCr TIFF를 RGB 모드로 읽어 결과가 달라져서 쓰지
않았다. (3) 읽을 수 없는 구조(잘림, 범위 밖 오프셋, 길이 없는 마커)에서는 멈추고 통과시킨다. Pillow도 같은
자리에서 오류를 내므로 같은 400이 되고, Pillow가 받아 주는 조금 어긋난 파일(IEND 없는 PNG 등)을 이 검사가 새로
거절하지 않는다. 상한을 넘었다는 관찰만 거절의 근거다. (4) 파서가 새 공격면이 되지 않게 모든 반복에 한도가
있어서 파일이 얼마나 크든 검사 비용은 한도에 묶인다(TIFF는 IFD 16개 × 항목 1,024개, JPEG는 세그먼트를 첫 스캔
앞뒤로 1,000개씩, PNG는 청크 11만 개 이하). 바이트열 읽기는 모두 범위를 확인한 뒤에 한다.

한계: 이 검사가 막는 것은 위에 적은 알려진 증폭 경로뿐이다. 요청 하나가 쓰는 CPU 시간과 메모리를 직접 묶는
장치는 없고, 이 검사가 그것을 대신하지 않는다. 상한 안에서도 비용은 0이 아니다(형식별로 상한 안에서 가장 비싼
입력을 고친 코드로 다시 잰 값은 docs/security.md). 같은 종류의 경로가 더 있을 수 있어 근본 대책은 디코딩을
작업 프로세스로 분리해 시간·메모리 한도에서 끝내는 것인데, 그 일은 아직 하지 않았다. 한도를 넘으면 `_decode`가
기존의 '읽을 수 없는 이미지'와 같은 400으로 바꾼다.
"""

import re
import struct
from collections import deque

# [파일 역할] service._decode가 Image.open 전에 부르는 구조 사전 검사. FastAPI에 기대지 않아 기본 패키지만으로
# 시험할 수 있다.
# [입력·출력] 업로드 바이트열 → 상한 안이면 None, 넘으면 ImageStructureError(메시지는 로그·시험용이고
# 클라이언트에 나가지 않는다).
# [연결] service._decode(호출, 예외를 400으로 바꿈), tests/test_image_precheck.py(상한 경계, 정상 파일 통과,
# Pillow 표·동작과의 일치).


class ImageStructureError(ValueError):
    """업로드의 구조 항목 수가 서비스의 상한을 넘는다(알려진 증폭 경로의 입력)."""


# --- TIFF 상한 -------------------------------------------------------------------------------------
# 정상 TIFF에서 센 값(실제 파일과 Pillow·OpenCV(libtiff)·tifffile이 저장한 2,400만 화소 파일, 표는
# docs/security.md): 첫 IFD 항목 10~18개, 줄마다 스트립이면 높이만큼(4,000개), 타일은 256이면 384개·64이면
# 5,922개·가장 작은 16이면 93,750개(2,500만 화소 정사각이면 97,969개), 그 밖의 숫자 배열 원소 합 10~18개(8비트
# 팔레트 그림의 COLORMAP은 768개), IFD는 첫 IFD·Exif·GPS·Interop 넷까지.
# - MAX_TIFF_ENTRIES: IFD 하나의 항목 수. 고전 TIFF의 개수 칸은 16비트(65,535)이고 BigTIFF는 64비트라 파일이
#   허락하는 만큼(20MiB에 100만 개) 적을 수 있다. Pillow는 항목마다 3~5μs를 쓰고 IFD를 여러 번 읽는다(첫 IFD는
#   열 때 두 번, 디코딩 뒤 한 번 더).
# - MAX_TIFF_STRIPS: StripOffsets·StripByteCounts·TileOffsets·TileByteCounts 하나의 원소 수. Pillow는 무압축
#   TIFF에서 StripOffsets(타일이면 TileOffsets) 원소마다 타일 하나를 만들어 디코딩 때 하나씩 디코더를 만든다
#   (TiffImagePlugin.py 1674행 `for offset in offsets`. 원소당 약 300바이트, 8.6μs: 260만 개에서
#   22초·837MiB, 상한 끝인 10만 개에 숫자 합계 끝까지 더해도 0.80초·+31MiB). 압축 TIFF는 libtiff가 네 표를
#   원소마다 8바이트 정수 배열로 바꾼다. 원소 수는 그림의 높이(줄마다 스트립 하나일 때)나 타일 수라서 화소 수
#   한도와 무관하게 커질 수 있다. 줄마다 스트립이어도 보통 높이만큼(수천 개)이라 25배 넘게 남지만, 2,500만
#   화소를 가장 작은 16×16 타일로 깔면 97,969개라 여유가 2%뿐이다(가늘고 긴 그림을 16×16 타일로 저장하면 넘을
#   수 있다: docs/security.md의 한계).
# - MAX_TIFF_VALUES: 위 표를 뺀 숫자 배열(SHORT·LONG·RATIONAL·DOUBLE 등)의 원소 수 합계(첫 IFD와 따라간 하위
#   IFD 전체). Pillow는 하위 IFD의 모든 항목을 파이썬 수의 튜플로 풀고, RATIONAL은 원소마다 IFDRational을
#   만든다(원소당 약 150바이트, 3.7μs: 20MiB의 RATIONAL 배열 하나가 CPU 10초·+397MiB). 정상 최대치(팔레트
#   768, 나머지 수십 개)의 85배다. BYTE·ASCII·UNDEFINED 값은 Pillow가 bytes·str 하나로 두므로 세지 않지만,
#   BYTE형으로 적은 몇 태그는 Pillow·libtiff가 그 bytes를 원소마다 돌므로 센다(아래 _TIFF_BYTE_WALKED_TAGS).
# - MAX_TIFF_IFDS: 따라가는 IFD 수(첫 IFD 포함). 정상 파일은 넷이다. 넘으면 멈추지 않고 거절한다: 멈추면 미끼
#   가리킴 여럿으로 한도를 채우고 진짜 가리킴을 그 뒤에 둘 수 있다.
MAX_TIFF_ENTRIES = 1_024
MAX_TIFF_STRIPS = 100_000
MAX_TIFF_VALUES = 65_536
MAX_TIFF_IFDS = 16

# --- JPEG 상한 -------------------------------------------------------------------------------------
# 정상 JPEG의 머리(SOI부터 첫 SOS까지)에서 센 값(실제 파일과 Pillow·OpenCV(libjpeg)가 저장한 파일): 세그먼트
# 7~12개(JFIF, Exif, DQT, SOF, DHT 등), 머리 길이 240바이트~22.8KB. ICC 프로파일은 65,519바이트씩 나뉘어
# 들어가므로 1.5MB짜리 프로파일을 넣어도 세그먼트 32개, 머리 길이 1.5MB다. 점진 JPEG의 스캔은 libjpeg의 기본
# 순서(Pillow·OpenCV가 쓴다)가 YCbCr 10개·회색 6개·CMYK 18개이고, 실제 사진 10장을 여러 설정으로 저장한 것의
# 최대도 18개였다.
# - MAX_JPEG_SEGMENTS: 첫 SOS 전의 마커 세그먼트 수(채움·이스케이프 마커 포함). Pillow는 세그먼트마다
#   바이트열과 튜플을 쌓고(약 130바이트, 1.5μs: 520만 개에서 CPU 7.9초·+676MiB) Exif 세그먼트를 이어 붙일
#   때마다 지금까지의 Exif 전체를 복사한다(세그먼트 수 × 길이에 비례). 첫 SOS 뒤의 세그먼트(스캔·DHT 등, 정상
#   점진 JPEG에서 수십 개)도 이 검사의 반복 수를 묶으려고 같은 상한으로 따로 센다.
# - MAX_JPEG_HEADER_BYTES: 첫 SOS 세그먼트가 끝나는 위치. Photoshop 자원(APP13)처럼 세그먼트 안을 파이썬으로
#   훑는 처리는 바이트당 약 0.1μs라 머리 길이에 비례한다(2MiB에서 0.25초). 실제 파일 최대 22.8KB의 90배이고
#   1.5MB짜리 ICC 프로파일을 넣은 파일도 들어간다(그 파일에는 1.4배 남는다).
# - MAX_JPEG_STRAY_BYTES: 세그먼트 사이에 끼어든 쓰레기·채움 바이트 합계. Pillow는 쓰레기 바이트를 파이썬
#   반복으로 한 바이트씩 건너뛴다(바이트당 약 0.1μs: 20MiB에서 2초). 정상 파일에는 없다.
# - MAX_JPEG_SCANS: 스캔(SOS) 수(첫 SOS 포함). libjpeg는 점진 JPEG의 스캔마다 그림 전체의 MCU를 돈다(자료가
#   모자라면 0으로 채운다). 5000×5000에서 스캔 하나가 CPU 약 1ms(YCbCr 4:2:0)~7ms(CMYK)이고 스캔 수에
#   선형이라, 상한 없이는 0.57MiB 파일이 37초를 썼다. 정상 최대 18의 5.5배인 100으로 두었다(5000×5000 CMYK
#   점진 JPEG이 스캔 18개면 0.61초, 100개면 1.17초). libtiff도 TIFF
#   안의 JPEG에 같은 기본 상한(100)을 둔다(tif_jpeg.c의 TIFFjpeg_progress_monitor: 실행해 보면 'Scan number
#   100 exceeds maximum scans (100)'으로 멈춘다). 이 경로는 사전 검사를 넣기 전부터 있던 문제다.
MAX_JPEG_SEGMENTS = 1_000
MAX_JPEG_HEADER_BYTES = 2 * 1024 * 1024
MAX_JPEG_STRAY_BYTES = 64 * 1024
MAX_JPEG_SCANS = 100

# --- PNG 상한 --------------------------------------------------------------------------------------
# 정상 PNG에서 센 값(실제 파일과 Pillow·OpenCV(libpng)가 저장한 파일): IDAT 밖의 청크 2~5개(IHDR, gAMA, pHYs,
# tEXt 등). IDAT는 인코더가 8KiB(libpng 기본)나 64KiB(Pillow)씩 끊으므로 20MiB 파일에서 320~2,560개이고,
# 1KiB로 쪼개면 20,480개다. 압축을 푸는 청크(iCCP·zTXt·iTXt)는 실제 PNG 46개에서 최대 2개였다.
# - MAX_PNG_CHUNKS_BEFORE_DATA: 첫 IDAT 앞의 청크 수(`Image.open`이 읽는 부분). Pillow는 알 수 없는 청크(두
#   번째 글자가 소문자인 것)를 청크마다 `private_chunks`에 쌓는다(약 120바이트, 4μs: 170만 개에서
#   5.4초·+207MiB).
# - MAX_PNG_OTHER_CHUNKS: IDAT·fdAT 밖의 청크 총수(IEND까지, 앞뒤 합계). 디코딩이 끝난 뒤에도 Pillow는
#   IEND까지 청크를 읽는다. APNG는 프레임마다 fcTL 하나가 더해진다.
# - MAX_PNG_DATA_CHUNKS: 이미지 데이터 조각(IDAT·fdAT·DDAT) 수. 조각마다 파이썬 반복을 한 번 도는 데 3.4μs를
#   쓴다(빈 조각 170만 개에서 5.7초).
# - MAX_PNG_INFLATING_CHUNKS: 압축을 푸는 청크(iCCP·zTXt·iTXt) 수(위치와 종류 무관 합계). Pillow는 이 청크를
#   하나에 1MiB까지 따로 푼다(PngImagePlugin.py 154~160행 `_safe_zlib_decompress`, 433~459행 iCCP, 619~652행
#   zTXt, 654~695행 iTXt). 텍스트 메모리 합계(64MiB, 412~419행)는 iCCP에는 없고, zTXt는 키가 있을 때만(645행),
#   iTXt는 UTF-8로 풀릴 때만(693행) 잡힌다. 그래서 청크 하나 약 1KB가 1MB로 풀리는 입력을 청크 수 상한 안에서
#   되풀이하면 CPU가 청크 수에 비례했다(iCCP 9,999개, 9.6MiB가 25초). 0 바이트 열이면 청크 하나에 약
#   2.5ms라 64개에서 0.16초다. 청크를 작은 deflate 블록 열로 채우면 zlib의 바이트당 비용(약 0.15초/MiB)이
#   파일 크기에 비례해 든다(18MiB에서 2.6초. IDAT에 넣어도 같다): 구조 항목 수로 묶을 수 없는 디코더 자체의
#   비용이다. 상한 64는 실제 파일 최대 2의 32배이고, Pillow로 iCCP 1개 + 압축 텍스트 20개를 넣은 파일도
#   들어간다.
MAX_PNG_CHUNKS_BEFORE_DATA = 1_000
MAX_PNG_OTHER_CHUNKS = 10_000
MAX_PNG_DATA_CHUNKS = 100_000
MAX_PNG_INFLATING_CHUNKS = 64

# --- BMP 상한 --------------------------------------------------------------------------------------
# - MAX_BMP_RLE_BYTES: RLE로 압축한 BMP(압축 칸 1 = RLE8, 2 = RLE4)의 파일 크기. Pillow의 RLE 디코더는 파이썬
#   반복이라(BmpImagePlugin.py 331~405행 BmpRleDecoder) 명령 두 바이트마다 한 바퀴를 돈다. 줄 끝(00 00)이나
#   이동 0(00 02 00 00)처럼 화소를 늘리지 않는 명령은 파일 끝까지 돌 수 있어 CPU가 그림 크기가 아니라 파일
#   크기에 비례한다(1×1 그림에 19.8MiB가 5초, MiB당 약 0.21~0.25초). 정상 RLE 디코딩도 MiB당 약 0.16초라
#   이것은 디코더 자체의 비용이고, 크기로만 묶을 수 있다. 4MiB에서 0.97~1.11초다. 실제 사진 10장을 256색으로
#   줄여 표준 RLE8로 저장하면 2,000픽셀 폭(260만~470만 화소)에서 0.15~3.23MiB였다(2MiB면 셋이 막힌다). 그보다
#   큰 RLE BMP는 거절된다(docs/security.md의 한계).
MAX_BMP_RLE_BYTES = 4 * 1024 * 1024


# 형식 판별은 Pillow가 쓰는 머리 바이트(각 플러그인의 _accept)와 같다. 이 네 형식이 아니면 그대로 돌려보낸다.
def check_image_structure(data: bytes) -> None:
    """`data`가 PNG·JPEG·TIFF·BMP이면 구조 항목 수를 세어, 상한을 넘으면 `ImageStructureError`를 낸다.

    입력은 `bytes`다. 상한 안이면 `None`을 돌려주고 아무것도 바꾸지 않는다. 형식을 알아볼 수 없거나(WebP
    포함) 구조가 잘려 더 읽을 수 없으면 그 자리에서 멈추고 통과시킨다(그 뒤의 판단은 Pillow가 한다). 한도에
    걸리지 않는 한 예외를 내지 않도록 모든 바이트 읽기는 범위를 먼저 확인한다.
    """
    if data.startswith(_PNG_SIGNATURE):
        _check_png(data)
    elif data.startswith(_JPEG_SIGNATURE):
        _check_jpeg(data)
    elif data[:4] in _TIFF_PREFIXES:
        _check_tiff(data)
    elif data.startswith(_BMP_SIGNATURE):
        _check_bmp(data)


# ==================================================================================================
# PNG
# ==================================================================================================
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# Pillow의 PngImageFile.load_read가 이어 읽는 이미지 데이터 청크(IDAT, APNG의 fdAT, 옛 MNG의 DDAT).
_PNG_DATA_CHUNKS = frozenset({b"IDAT", b"fdAT", b"DDAT"})
# Pillow가 청크마다 zlib로 푸는 청크(MAX_PNG_INFLATING_CHUNKS 주석). iTXt는 압축 표시가 켜졌을 때만 풀지만 이
# 검사는 표시와 무관하게 센다(더 넓게 센다).
_PNG_INFLATING_CHUNKS = frozenset({b"iCCP", b"zTXt", b"iTXt"})
# Pillow의 PngImagePlugin.is_cid와 같은 식: 청크 종류는 영숫자·밑줄 네 글자여야 하고, 아니면 Pillow는 첫 IDAT
# 앞에서 오류를 내고 뒤에서는 읽기를 멈춘다. 이 검사도 거기서 멈춘다.
_PNG_CHUNK_TYPE = re.compile(rb"\w\w\w\w")
# 청크 머리의 앞 4바이트는 자료 길이(큰 쪽 바이트 순서)이고, 이어지는 4바이트가 청크 종류다.
_PNG_LENGTH = struct.Struct(">I")


# 청크 사슬을 처음부터 IEND까지(또는 읽을 수 없는 곳까지) 따라가며 센다. 청크 하나는 헤더 8바이트 + 자료 + CRC
# 4바이트다. CRC는 확인하지 않는다: Pillow도 첫 IDAT 뒤의 CRC는 읽기만 하고(앞쪽은 확인해 오류를 낸다) 틀려도
# 그 오류는 Pillow가 낸다. IEND가 없거나 마지막 청크가 잘린 PNG를 Pillow는 받아 주므로(load_end가 읽기
# 오류에서 멈춘다) 이 검사도 거절하지 않는다. 첫 IDAT 뒤의 청크도 Pillow의 load_end(1047~1088행)가 같은
# 처리기로 읽으므로 압축 청크는 위치와 무관하게 센다. 한도를 넘는 순간 멈춘다.
def _check_png(data: bytes) -> None:
    size = len(data)
    pos = len(_PNG_SIGNATURE)
    before_data = others = data_chunks = inflating = 0
    seen_data = False
    while pos + 8 <= size:
        (length,) = _PNG_LENGTH.unpack_from(data, pos)
        kind = data[pos + 4 : pos + 8]
        if not _PNG_CHUNK_TYPE.match(kind):
            return
        if kind in _PNG_DATA_CHUNKS:
            seen_data = True
            data_chunks += 1
            if data_chunks > MAX_PNG_DATA_CHUNKS:
                raise ImageStructureError(f"PNG has more than {MAX_PNG_DATA_CHUNKS} image data chunks")
        elif kind == b"IEND":
            return
        else:
            others += 1
            if others > MAX_PNG_OTHER_CHUNKS:
                raise ImageStructureError(
                    f"PNG has more than {MAX_PNG_OTHER_CHUNKS} chunks besides the image data"
                )
            if not seen_data:
                before_data += 1
                if before_data > MAX_PNG_CHUNKS_BEFORE_DATA:
                    raise ImageStructureError(
                        f"PNG has more than {MAX_PNG_CHUNKS_BEFORE_DATA} chunks before the image data"
                    )
            if kind in _PNG_INFLATING_CHUNKS:
                inflating += 1
                if inflating > MAX_PNG_INFLATING_CHUNKS:
                    raise ImageStructureError(
                        f"PNG has more than {MAX_PNG_INFLATING_CHUNKS} compressed profile or text chunks"
                    )
        pos += 12 + length


# ==================================================================================================
# JPEG
# ==================================================================================================
_JPEG_SIGNATURE = b"\xff\xd8\xff"
# 연속한 0xFF(채움 바이트)를 한 번에 건너뛰는 식. Pillow는 채움 바이트마다 파이썬 반복을 한 번 돈다.
_JPEG_FILL_RUN = re.compile(rb"\xff*")
# 첫 SOS 뒤에서 libjpeg가 처리하는 마커 가운데 스캔 수를 세는 데 필요한 것: EOI(0xD9), SOS(0xDA), 그리고 길이
# 칸이 있는 세그먼트(SOFn 0xC0~0xC7·0xC9~0xCF, DHT 0xC4, DAC 0xCC, DQT 0xDB, DNL 0xDC, DRI 0xDD, DHP 0xDE, EXP
# 0xDF, APPn 0xE0~0xEF, COM 0xFE). 앞의 0xFF가 마커 접두다. 빠진 것은 길이 없는 마커(RST0~7 0xD0~0xD7, TEM
# 0x01)와 libjpeg가 오류로 멈추는 마커(0x02~0xBF, JPG 0xC8, SOI 0xD8, JPGn 0xF0~0xFD)다. 0x00·0xFF는 마커가
# 아니다(아래 _check_jpeg_scans 주석).
_JPEG_SCAN_WALK = re.compile(rb"\xff[\xc0-\xc7\xc9-\xcf\xd9-\xdf\xe0-\xef\xfe]")


# Pillow의 JpegImagePlugin.MARKER 표에서 핸들러가 None인 마커(길이 칸이 없다): JPG(0xC8), RST0~7과
# SOI·EOI(0xD0~0xD9), JPG0~13(0xF0~0xFD). 0xC0~0xFE의 나머지는 모두 표에 있고 길이 칸(2바이트, 칸 자신 포함)이
# 뒤따른다. 0xC0 아래의 값은 표에 없어 Pillow가 "no marker found"로 오류를 낸다(tests/test_image_precheck.py가
# 이 구분을 Pillow의 표와 대조한다).
def _jpeg_marker_has_length(marker: int) -> bool:
    return not (marker == 0xC8 or 0xD0 <= marker <= 0xD9 or 0xF0 <= marker <= 0xFD)


# SOI 뒤 첫 SOS(스캔 시작)까지의 세그먼트를 Pillow의 JpegImageFile._open(JpegImagePlugin.py 356~413행)과 같은
# 규칙으로 따라간다:
# - 처음 세 바이트(0xFFD8FF)는 이미 읽은 것이고 셋째 바이트(FF)가 첫 마커의 접두다.
# - 마커 접두(FF) 다음 바이트가 마커 번호다. FF가 더 이어지면 채움 바이트고(마지막 FF가 접두), 00이면
#   이스케이프된 FF라 건너뛴다.
# - 길이 칸이 있는 마커는 칸 값 - 2바이트의 자료를 읽고(칸 값이 2 이하면 자료 없음) 넘어간다. 길이 칸이 없는
#   마커는 바로 다음 바이트로 간다.
# - 마커 뒤 FF가 아닌 바이트는 쓰레기라 다음 FF까지 한 바이트씩 건너뛴다(Pillow는 파이썬 반복, 여기서는 한
#   번에 찾되 개수를 센다).
# - SOS(0xDA)를 읽으면 Pillow의 머리 읽기가 끝나고, 그 뒤의 스캔은 _check_jpeg_scans가 libjpeg의 규칙으로
#   센다. 파일이 끝나거나 자료가 파일 밖으로 나가거나 알 수 없는 마커(< 0xC0)가 나오면 Pillow가 오류를 내거나
#   읽기를 끝내는 자리라 여기서도 멈춘다.
def _check_jpeg(data: bytes) -> None:
    size = len(data)
    pos = len(_JPEG_SIGNATURE)
    after_marker = False  # 방금 마커(또는 이스케이프)를 처리했으면 True: 다음 FF까지 쓰레기를 건너뛴다
    segments = stray = 0
    while True:
        if after_marker:
            window = pos + MAX_JPEG_STRAY_BYTES - stray + 1
            found = data.find(b"\xff", pos, window)
            if found < 0:
                if window > size:  # 파일 끝까지 쓰레기가 상한 안이다: Pillow는 끝까지 건너뛰고 읽기를 마친다
                    return
                raise ImageStructureError(
                    f"JPEG has more than {MAX_JPEG_STRAY_BYTES} stray bytes between segments"
                )
            stray += found - pos
            pos = found + 1
        if pos >= size:
            return
        window = pos + MAX_JPEG_STRAY_BYTES - stray + 1
        fill_end = _JPEG_FILL_RUN.match(data, pos, window).end()
        if fill_end >= window:
            raise ImageStructureError(
                f"JPEG has more than {MAX_JPEG_STRAY_BYTES} stray bytes between segments"
            )
        stray += fill_end - pos
        pos = fill_end
        if pos >= size:
            return
        marker = data[pos]
        pos += 1
        segments += 1
        if segments > MAX_JPEG_SEGMENTS:
            raise ImageStructureError(
                f"JPEG has more than {MAX_JPEG_SEGMENTS} marker segments before the scan"
            )
        after_marker = True
        if marker == 0x00:
            continue
        if marker < 0xC0:
            return
        if _jpeg_marker_has_length(marker):
            if pos + 2 > size:
                return
            length = (data[pos] << 8) | data[pos + 1]
            pos += 2
            if length > 2:
                pos += length - 2
                if pos > size:
                    return
        if pos > MAX_JPEG_HEADER_BYTES:
            raise ImageStructureError(f"JPEG header is longer than {MAX_JPEG_HEADER_BYTES} bytes")
        if marker == 0xDA:
            _check_jpeg_scans(data, pos)
            return


# 첫 SOS 세그먼트가 끝난 pos부터 스캔(SOS) 수를 센다(첫 SOS 포함). 화소 디코딩은 Pillow가 아니라
# libjpeg-turbo가 파일 처음부터 하고(Pillow의 타일 오프셋 0, JpegImagePlugin.py 398~400행), libjpeg는 EOI를
# 만나거나 오류가 날 때까지 스캔을 차례로 처리한다. 그래서 libjpeg가 마커를 알아보는 규칙(jdhuff.c의
# jpeg_fill_bit_buffer, jdmarker.c의 next_marker·read_markers)을 따른다:
# - 엔트로피 부호화 자료 안의 0xFF 바이트는 늘 뒤에 0x00을 덧붙여 적는다(JPEG 규격의 바이트 채움, byte
#   stuffing). 그래서 'FF 다음이 00도 FF도 아닌 바이트'인 자리는 마커뿐이고, 자료 안의 바이트를 마커로 잘못
#   세지 않는다. FF가 여럿 이어진 뒤 마커 번호가 오면(채움 바이트) 마지막 FF가 접두이고, libjpeg도 그렇게
#   읽는다.
# - 길이 칸이 있는 세그먼트(DHT·DQT·APPn·COM 등)는 libjpeg처럼 길이만큼 건너뛰므로 그 자료 안의 FF DA를 세지
#   않는다. 길이가 맞지 않는 DHT·DQT·SOS·DRI에서는 libjpeg가 오류로 멈추므로 그 뒤를 더 세어도 손해가 없다.
# - RST0~7·TEM은 길이 없는 마커라 건너뛴다. libjpeg가 오류로 멈추는 마커(0x02~0xBF, 0xC8, SOI, 0xF0~0xFD)도
#   건너뛰고 계속 센다(멈춘 뒤의 스캔은 처리되지 않으니 더 넓게 세는 쪽이다).
# - EOI에서 멈춘다. EOI 뒤에 붙은 자료(동영상이 붙은 사진, 두 번째 이미지 등)는 libjpeg가 읽지 않으므로 세지
#   않는다. EOI 없이 파일이 끝나도 libjpeg는 그때까지의 스캔을 모두 처리한 뒤에야 잘림을 알리므로 끝까지 센다.
# 첫 SOS 뒤의 세그먼트 수도 MAX_JPEG_SEGMENTS로 묶어 이 반복이 파일 크기와 무관하게 끝나게 한다. 마커 찾기는
# 정규식 search 한 번(C 수준)이라 엔트로피 자료가 길어도 파이썬 반복이 늘지 않는다.
def _check_jpeg_scans(data: bytes, pos: int) -> None:
    size = len(data)
    scans = 1
    segments = 0
    while True:
        found = _JPEG_SCAN_WALK.search(data, pos)
        if found is None:
            return
        marker = data[found.end() - 1]
        pos = found.end()
        if marker == 0xD9:
            return
        segments += 1
        if segments > MAX_JPEG_SEGMENTS:
            raise ImageStructureError(
                f"JPEG has more than {MAX_JPEG_SEGMENTS} marker segments after the first scan"
            )
        if marker == 0xDA:
            scans += 1
            if scans > MAX_JPEG_SCANS:
                raise ImageStructureError(f"JPEG has more than {MAX_JPEG_SCANS} scans")
        if pos + 2 > size:
            return
        # 길이 칸(칸 자신 포함)만큼 건너뛴다. 칸 값이 2보다 작으면 libjpeg는 칸 두 바이트만 읽는다.
        pos += max((data[pos] << 8) | data[pos + 1], 2)


# ==================================================================================================
# TIFF
# ==================================================================================================
# Pillow의 TiffImagePlugin.PREFIXES와 같은 여섯 가지 머리(바이트 순서 II/MM, 버전 42(고전)·43(BigTIFF), 잘못
# 적힌 머리 두 가지도 받는다).
_TIFF_PREFIXES = (b"MM\x00\x2a", b"II\x2a\x00", b"MM\x2a\x00", b"II\x00\x2a", b"MM\x00\x2b", b"II\x2b\x00")
# Pillow가 항목 값을 파이썬 수의 튜플로 푸는 형식 번호(`ImageFileDirectory_v2._load_dispatch`에서
# BYTE(1)·ASCII(2)·UNDEFINED(7)를 뺀 것: SHORT·LONG·RATIONAL·SBYTE·SSHORT·SLONG·SRATIONAL·FLOAT·DOUBLE·IFD·
# LONG8). 그 밖의 형식 번호는 Pillow가 항목을 버린다.
_TIFF_NUMERIC_TYPES = frozenset({3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 16})
_TIFF_BYTE = 1
# BYTE형(1)으로 적었을 때 값이 bytes 하나로 남지만(TiffImagePlugin.py 815~817행 load_byte, 757~769행
# _setitem의 BYTE 분기) 서비스의 디코딩 경로가 그 bytes를 원소마다 도는 첫 IFD 태그. 설치된 Pillow 13.0.0의
# 소스에서 찾은 것과 실측(survey 스크립트, 태그마다 BYTE형 10만·40만 개로 열고 RGB로 바꾸기):
# - StripOffsets(273)·TileOffsets(324): _setup 1657·1662행에서 꺼내 1674행 `for offset in offsets`로
#   바이트마다 _Tile을 만든다(40만 개에서 2.75초·89MiB, 타일 0.92초).
# - StripOffsets·StripByteCounts(279)·TileOffsets·TileByteCounts(325): 압축 TIFF는 _load_libtiff(1375행)가
#   libtiff에 넘기고, libtiff는 네 표를 원소마다 8바이트 정수 배열로 바꾼다(BYTE형 200만 개에서 +17~32MiB, C
#   수준).
# - ColorMap(320): 팔레트 그림이면 1714행 `tuple(b // 256 for b in ...)`이 원소마다 파이썬 반복을 돈다.
# - SampleFormat(339): 1518행 `max(sample_format) == min(sample_format)`이 C 수준으로 원소마다 돈다(2,000만
#   개에서 0.30초).
# - ExtraSamples(338): 평면 배치 2이면 1538행 `max(extra_tuple)`이 원소마다 돈다.
# 확인했지만 돌지 않는 것: BitsPerSample(258, 자르기·len만 하고 튜플 키 조회에서 실패), YCbCrSubSampling(530,
# 로그 인자로만 꺼냄), XMP(700)·ICC(34675) 등(정규식·대입만, C 수준이고 파일 크기에 묶임), 하위 IFD 안의 BYTE
# 값(Image.Exif._get_ifd_dict가 dict로 담을 뿐 돌지 않음). ASCII·UNDEFINED는 Pillow가 원소 하나짜리 튜플로
# 싸서 돌지 않는다.
_TIFF_BYTE_WALKED_TAGS = frozenset({273, 279, 324, 325, 320, 338, 339})
# 정수 하나로 풀리는 형식과 그 크기(바이트). 하위 IFD를 가리키는 항목의 값을 읽는 데 쓴다. Pillow는 이 값으로
# fp.seek를 부르므로 정수 형식만 따라간다(BYTE는 bytes, RATIONAL은 IFDRational, FLOAT·DOUBLE은 float이라
# seek가 TypeError를 내고 Image.Exif._get_ifd_dict(Image.py 4112~4126행)가 None으로 넘긴다).
_TIFF_INTEGER_WIDTHS = {
    3: ("H", 2),
    4: ("L", 4),
    6: ("b", 1),
    8: ("h", 2),
    9: ("l", 4),
    13: ("L", 4),
    16: ("Q", 8),
}
# 스트립·타일 표: StripOffsets(273), StripByteCounts(279), TileOffsets(324), TileByteCounts(325).
_TIFF_TABLE_TAGS = frozenset({273, 279, 324, 325})
# 하위 IFD를 가리키는 항목: ExifIFD(34665), GPSInfoIFD(34853), InteroperabilityIFD(40965). Pillow는 디코딩이
# 끝난 뒤(load_end, TiffImagePlugin.py 1358~1373행, 쪽이 하나일 때만) getexif()로 첫 IFD를 다시 읽고 이 세
# 무리를 get_ifd로 연다(Image.py 4231~4342행): Exif·GPS는 첫 IFD의 가리킴(4239행), Interop은 첫 IFD에 40965가
# 있을 때 Exif IFD 안의 가리킴(4245~4247·4332행)을 따라가 그 IFD의 모든 항목을 푼다. 무리마다 한 번씩 열어서
# 세 가리킴이 같은 IFD를 가리키면 세 번 푼다. 가리킴 태그는 Pillow의 태그 표에 개수 1로 정의돼 있어(TiffTags:
# 첫 IFD와 Exif 무리 모두) 값이 여럿이면 경고만 내고 첫 값을 쓴다(TiffImagePlugin.py 757~777행). 같은 태그가
# 되풀이되면 처음으로 쓸 수 있는 항목을 쓰는데, 형식이 지원되지 않거나 태그 자료 예산(파일 크기)을 넘은 항목은
# 버린다(922~947행). 서비스 경로에서 Pillow가 여는 IFD는 이것이 전부다: getexif·exif_transpose(ImageOps.py
# 745~803행)는 더 열지 않고, 다음 IFD(둘째 쪽)는 seek·n_frames에서만, SubIFDs(330)는
# ImageFile.get_child_images에서만 읽으며 서비스는 둘 다 부르지 않는다.
_TIFF_SUB_IFD_TAGS = frozenset({34665, 34853, 40965})


# 머리를 읽어 첫 IFD를 찾아 항목을 세고, 가리킴을 따라가며 하위 IFD의 항목도 센다(_TIFF_SUB_IFD_TAGS 주석의
# Pillow 규칙). Pillow가 어느 가리킴을 쓸지 이 검사가 가를 수 없는 곳은 더 넓게 센다:
# - 가리킴은 센 모든 IFD에서 모은다(Interop은 Exif IFD 안에 있다). Pillow가 따라가지 않는 자리(첫 IFD의 40965,
#   GPS·Interop IFD 안의 가리킴)도 따라간다.
# - 같은 가리킴 태그가 여럿이면 정수로 읽히는 것을 모두 따라간다. 처음 것만 따라가면 Pillow가 버리는 미끼를
#   앞에 두고 진짜를 뒤에 둘 수 있다.
# - 값의 개수와 무관하게 첫 값을 따라간다(Pillow와 같다).
# - (가리킴 태그, 위치) 쌍마다 한 번씩 센다: Pillow가 무리마다 따로 열기 때문이다. 같은 쌍은 한 번만 센다.
# - 첫 IFD는 표(273·279·324·325)를 따로 세지만, 가리킴이 첫 IFD 자신을 가리키면 Pillow가 그 IFD를 하위 IFD로서
#   모두 풀므로 그때는 표도 숫자 합계에 든다.
# - 따라간 IFD 수가 MAX_TIFF_IFDS를 넘으면 거절한다.
# 다음 IFD(여러 쪽 TIFF의 둘째 쪽 이후)는 따라가지 않는다: Pillow는 열 때 첫 쪽만 읽고 다음 쪽 오프셋만
# 기억한다. 하위 IFD 가리킴이 깨진 파일은 Pillow가 경고만 내고 받아 주지만 가리킨 곳이 항목 수 상한을 넘는
# 표로 읽히면 이 검사는 거절한다(가리킴이 진짜인 파일과 가를 방법이 없고, 가리킴을 믿지 않으면 큰 표를 하위
# IFD에 숨길 수 있다). 머리 해석은 Pillow와 같다: data[2] == 43일 때만 BigTIFF(머리 16바이트, 오프셋·개수
# 8바이트, 항목 20바이트)로 읽는다. 그래서 big-endian BigTIFF (MM\x00\x2b)는 Pillow가 고전 TIFF로 읽는 것과
# 똑같이 이 검사도 고전 TIFF로 읽는다(첫 IFD 오프셋이 엉뚱한 값이 되어 Pillow가 거절한다).
def _check_tiff(data: bytes) -> None:
    size = len(data)
    endian = "<" if data[:2] == b"II" else ">"
    bigtiff = data[2] == 43
    if size < (16 if bigtiff else 8):
        return
    # 고전 TIFF: 개수 2바이트, 항목 12바이트(태그 2·형식 2·개수 4·값 4), 값 칸 4바이트. BigTIFF: 개수 8바이트,
    # 항목 20바이트(…·개수 8·값 8), 값 칸 8바이트.
    offset_format = endian + ("Q" if bigtiff else "L")
    count_format = endian + ("Q" if bigtiff else "H")
    count_size, entry_size, value_size = (8, 20, 8) if bigtiff else (2, 12, 4)
    entry = struct.Struct(endian + ("HHQ" if bigtiff else "HHL"))
    value_at = entry_size - value_size
    (first_ifd,) = struct.unpack_from(offset_format, data, 8 if bigtiff else 4)
    values = 0  # 숫자 배열 원소 수 합계(첫 IFD와 따라간 하위 IFD 전체)

    # 가리킴 항목의 첫 값. Pillow처럼 값 전체(형식 크기 × 개수)가 값 칸에 들어가면 칸에서, 아니면 칸이
    # 가리키는 곳에서 읽는다. 읽을 수 없거나 음수면 None(Pillow도 그런 값으로는 하위 IFD를 열지 못한다: 음수는
    # seek가 ValueError를 내서 디코딩이 실패한다).
    def integer_value(pos: int, kind: int, count: int) -> int | None:
        spec = _TIFF_INTEGER_WIDTHS.get(kind)
        if spec is None or count < 1:
            return None
        fmt, width = spec
        at = pos + value_at
        if width * count > value_size:
            (at,) = struct.unpack_from(offset_format, data, at)
        if at + width > size:
            return None
        (value,) = struct.unpack_from(endian + fmt, data, at)
        return value if value >= 0 else None

    # IFD 하나의 항목을 세고, 그 IFD에 있는 가리킴(태그, 위치)을 모두 돌려준다.
    def scan(offset: int, is_first: bool) -> list[tuple[int, int]]:
        nonlocal values
        pointers: list[tuple[int, int]] = []
        if offset + count_size > size:
            return pointers
        (entries,) = struct.unpack_from(count_format, data, offset)
        if entries > MAX_TIFF_ENTRIES:
            raise ImageStructureError(f"TIFF directory has more than {MAX_TIFF_ENTRIES} entries")
        pos = offset + count_size
        for _ in range(entries):
            if pos + entry_size > size:
                break  # 항목 표가 잘렸다: Pillow도 여기서 읽기를 멈춘다
            tag, kind, count = entry.unpack_from(data, pos)
            # 같은 태그가 되풀이돼도 모두 센다. Pillow는 첫 항목을 쓸 수 없을 때(모르는 형식 등) 다음 항목을
            # 쓰므로 어느 쪽을 쓸지 이쪽에서 가리지 않는다. BYTE형은 Pillow·libtiff가 원소마다 도는 첫 IFD
            # 태그일 때만 센다(_TIFF_BYTE_WALKED_TAGS).
            walked = is_first and kind == _TIFF_BYTE and tag in _TIFF_BYTE_WALKED_TAGS
            if kind in _TIFF_NUMERIC_TYPES or walked:
                if is_first and tag in _TIFF_TABLE_TAGS:
                    if count > MAX_TIFF_STRIPS:
                        raise ImageStructureError(f"TIFF strip or tile table (tag {tag}) has {count} entries")
                else:
                    values += count
                    if values > MAX_TIFF_VALUES:
                        raise ImageStructureError(
                            f"TIFF has more than {MAX_TIFF_VALUES} numeric values to decode"
                        )
            if tag in _TIFF_SUB_IFD_TAGS:
                target = integer_value(pos, kind, count)
                if target is not None:
                    pointers.append((tag, target))
            pos += entry_size
        return pointers

    # 첫 IFD에서 시작해 가리킴을 차례로 따라간다(너비 우선). 같은 (태그, 위치) 쌍은 한 번만 연다.
    pending = deque(scan(first_ifd, is_first=True))
    opened: set[tuple[int, int]] = set()
    while pending:
        pointer = pending.popleft()
        if pointer in opened:
            continue
        opened.add(pointer)
        if len(opened) + 1 > MAX_TIFF_IFDS:
            raise ImageStructureError(f"TIFF points to more than {MAX_TIFF_IFDS} directories")
        pending.extend(scan(pointer[1], is_first=False))


# ==================================================================================================
# BMP
# ==================================================================================================
_BMP_SIGNATURE = b"BM"
# Pillow의 BmpImageFile._bitmap(BmpImagePlugin.py 117~128행)이 압축 칸(정보 머리의 16번째 바이트, 파일 오프셋
# 30)을 읽는 정보 머리 크기: BITMAPINFOHEADER(40), BITMAPV2INFOHEADER(52), BITMAPV3INFOHEADER(56), OS/2 2판의
# 머리(64), BITMAPV4HEADER(108), BITMAPV5HEADER(124). 12(BITMAPCOREHEADER)는 압축 칸이 없어 Pillow가
# 무압축으로 읽고, 그 밖의 크기는 Pillow가 거절한다.
_BMP_INFO_HEADER_SIZES = frozenset({40, 52, 56, 64, 108, 124})
# Pillow가 파이썬 디코더 bmp_rle(BmpRleDecoder)로 푸는 압축 번호: RLE8(1)·RLE4(2)(257~261행).
_BMP_RLE_COMPRESSIONS = frozenset({1, 2})


# RLE로 압축한 BMP가 MAX_BMP_RLE_BYTES보다 크면 거절한다. 파일 머리(14바이트) 뒤의 정보 머리 크기(4바이트)와
# 압축 칸(오프셋 30, 4바이트, 작은 쪽 바이트 순서)만 읽는다. 크기는 파일 전체 길이로 잰다(RLE 자료는 그보다
# 짧으니 더 넓게 센다). 무압축·비트필드 BMP는 C 디코더라 화소 수에 비례하므로 건드리지 않는다.
def _check_bmp(data: bytes) -> None:
    if len(data) <= MAX_BMP_RLE_BYTES or len(data) < 34:
        return
    (header_size,) = struct.unpack_from("<I", data, 14)
    if header_size not in _BMP_INFO_HEADER_SIZES:
        return
    (compression,) = struct.unpack_from("<I", data, 30)
    if compression in _BMP_RLE_COMPRESSIONS:
        raise ImageStructureError(f"BMP with RLE compression is larger than {MAX_BMP_RLE_BYTES} bytes")
