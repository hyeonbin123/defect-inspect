# [파일 역할] 업로드 구조 사전 검사(`defect_inspect.image_precheck`)와, 그것을 `Image.open` 앞에서 부르는
# `service._decode`를 검증한다.
# [왜 필요한가] Pillow의 `Image.open`과 디코딩은 화소 수와 무관한 구조 항목마다 일을 한다. 무압축 TIFF의
# 스트립 표, TIFF의 Exif·GPS·Interop 하위 IFD의 숫자 배열, JPEG의 마커와 점진 JPEG의 스캔, PNG의 청크와 압축을
# 푸는 청크, BMP RLE의 명령이 그렇다. 그래서 파일 크기(20MiB)·화소 수(2,500만) 한도 안의 파일 하나가 작업
# 스레드를 CPU 수 초에서 수십 초 잡고 메모리를 수백 MiB 썼다(화소 수 검사는 `Image.open`이 끝난 뒤라 늦다). 이
# 파일은 (1) 알려진 증폭 경로의 입력이 열리기 전에 400으로 끝나는 것, (2) 상한이 정상 파일을 건드리지 않는 것,
# (3) 검사가 Pillow(와 그 밑의 libjpeg·libtiff)가 읽는 규칙을 따라가는 것, (4) 그 규칙의 전제(Pillow가 무엇을
# 원소마다 돌고 어떤 가리킴을 따라가는지)가 설치된 Pillow에서도 참인 것을 지킨다.
# [주요 구성] 상수 표가 설치된 Pillow와 맞는지 → 정상 파일(여러 인코더, 구조가 많은 정상 파일)이 통과 → 폭탄
# 입력이 거절되고 싸게 끝남 → 상한 경계(상한 개수는 통과, 하나 더는 거절) → Pillow의 읽기 규칙과의
# 일치(잘렸거나 엇나간 구조는 멈추고 통과시켜 Pillow에 맡김, 가리킴은 Pillow가 따라가는 것을 모두 따라감) →
# 어떤 바이트열에도 ImageStructureError 말고는 예외를 내지 않음 → 서비스 통합(FastAPI가 있을 때): 400,
# Image.open 미호출, 정상 파일의 화소는 Pillow로 직접 연 것과 같음. 큰 파일은 저장소에 넣지 않고 시험 안에서
# 바이트로 만든다(폭탄 하나가 20MiB 안팎이다).

import functools
import io
import re
import struct
import time
import tracemalloc
import zlib

import numpy as np
import pytest
from PIL import (
    BmpImagePlugin,
    Image,
    ImageFile,
    JpegImagePlugin,
    PngImagePlugin,
    TiffImagePlugin,
    TiffTags,
    UnidentifiedImageError,
)

from defect_inspect import image_precheck as pre

ERROR_DETAIL = "the upload is not a readable image"
TOKEN = "test-admin-token"


# ==================================================================================================
# 입력 제작 도우미
# ==================================================================================================
# TIFF 형식 번호 -> 값 하나의 struct 서식(RATIONAL은 분자·분모 두 개).
_TIFF_FORMATS = {
    1: "B",
    2: "B",
    3: "H",
    4: "L",
    5: "LL",
    6: "b",
    7: "B",
    8: "h",
    9: "l",
    10: "ll",
    11: "f",
    12: "d",
    13: "L",
    16: "Q",
}


# 같은 값 count개를 TIFF 형식 kind의 배열 바이트로 만든다(값은 RATIONAL이면 (분자, 분모) 쌍).
def _repeat(endian: str, kind: int, value, count: int) -> bytes:
    values = value if isinstance(value, tuple) else (value,)
    return struct.pack(endian + _TIFF_FORMATS[kind], *values) * count


# TIFF 하나를 만든다. ifds는 IFD 목록이고 각 IFD는 (태그, 형식, 개수, 값 바이트)의 목록이다. 값 바이트는
# 개수와 맞지 않아도 된다(깨진 파일을 만들 때). 값 자리에 ("IFD", k)를 주면 k번째 IFD의 오프셋을 LONG 값으로
# 넣고, ("IFD", k, "H")처럼 struct 서식을 더하면 그 형식의 정수로 넣는다(값 칸보다 크면 칸 밖에 두고 칸에는 그
# 위치를 적는다). ("IFD", k, "L", 2)처럼 넷째 값을 주면 같은 오프셋을 그 수만큼 되풀이해 넣는다(개수 2 이상인
# 가리킴: 항목의 개수 칸은 부르는 쪽이 같은 수로 적는다). 머리 바로 뒤에 화소 데이터를 두므로 화소의 오프셋은
# 머리 크기(고전 8, BigTIFF 16)다. chain=True면 IFD를 차례로 이어(첫 IFD의 next가 둘째) 여러 쪽 TIFF로 만든다.
def _tiff(
    ifds, pixel: bytes = b"\x80", *, big: bool = False, endian: str = "<", chain: bool = False
) -> bytes:
    head = 16 if big else 8
    count_size, entry_size, next_size, inline = (8, 20, 8, 8) if big else (2, 12, 4, 4)
    offset_format = endian + ("Q" if big else "L")
    cursor = head + len(pixel)
    where = []
    for entries in ifds:
        where.append(cursor)
        cursor += count_size + len(entries) * entry_size + next_size
        for _, _, _, payload in entries:
            width = (
                struct.calcsize("=" + payload[2]) if isinstance(payload, tuple) and len(payload) > 2 else 4
            )
            copies = payload[3] if isinstance(payload, tuple) and len(payload) > 3 else 1
            size = len(payload) if isinstance(payload, bytes) else width * copies
            if size > inline:
                cursor += size + (size & 1)
    out = bytearray(b"II" if endian == "<" else b"MM")
    out += struct.pack(endian + "HHH", 43, 8, 0) if big else struct.pack(endian + "H", 42)
    out += struct.pack(offset_format, where[0]) + pixel
    for k, entries in enumerate(ifds):
        out += struct.pack(endian + ("Q" if big else "H"), len(entries))
        spill = bytearray()
        spill_at = len(out) + len(entries) * entry_size + next_size
        for tag, kind, count, payload in entries:
            if isinstance(payload, tuple):
                pointer = struct.pack(endian + (payload[2] if len(payload) > 2 else "L"), where[payload[1]])
                pointer *= payload[3] if len(payload) > 3 else 1
                if len(pointer) <= inline:
                    value = pointer.ljust(inline, b"\0")
                else:
                    value = struct.pack(offset_format, spill_at + len(spill))
                    spill += pointer + (b"\0" if len(pointer) & 1 else b"")
            elif len(payload) <= inline:
                value = payload.ljust(inline, b"\0")
            else:
                value = struct.pack(offset_format, spill_at + len(spill))
                spill += payload + (b"\0" if len(payload) & 1 else b"")
            out += struct.pack(endian + "HH", tag, kind) + struct.pack(offset_format, count) + value
        out += struct.pack(offset_format, where[k + 1] if chain and k + 1 < len(ifds) else 0) + spill
    return bytes(out)


# 8비트 회색 무압축 TIFF의 IFD 항목. 스트립 오프셋은 모두 화소 한 곳(머리 직후)을 가리킨다.
def _gray_entries(endian="<", *, width=1, height=1, strips=1, with_counts=True, head=8, extra=()):
    entries = [
        (256, 4, 1, struct.pack(endian + "L", width)),
        (257, 4, 1, struct.pack(endian + "L", height)),
        (258, 3, 1, struct.pack(endian + "H", 8)),
        (259, 3, 1, struct.pack(endian + "H", 1)),
        (262, 3, 1, struct.pack(endian + "H", 1)),
        (273, 4, strips, _repeat(endian, 4, head, strips)),
        (277, 3, 1, struct.pack(endian + "H", 1)),
        (278, 4, 1, struct.pack(endian + "L", 1)),
    ]
    if with_counts:
        entries.append((279, 4, strips, _repeat(endian, 4, 1, strips)))
    return sorted([*entries, *extra], key=lambda entry: entry[0])


# 줄마다 스트립 하나(또는 rows줄마다 하나)인 실제 8비트 회색 무압축 TIFF: 화소가 진짜로 들어 있고 스트립마다
# 오프셋이 다르다.
def _striped_tiff(pixels: np.ndarray, rows: int = 1, endian: str = "<") -> bytes:
    height, width = pixels.shape
    starts = list(range(0, height, rows))
    offsets = struct.pack(endian + f"{len(starts)}L", *(8 + start * width for start in starts))
    counts = struct.pack(endian + f"{len(starts)}L", *(min(rows, height - start) * width for start in starts))
    entries = _gray_entries(endian, width=width, height=height, strips=len(starts))
    entries = [e for e in entries if e[0] not in (273, 278, 279)]
    entries += [
        (273, 4, len(starts), offsets),
        (278, 4, 1, struct.pack(endian + "L", rows)),
        (279, 4, len(starts), counts),
    ]
    return _tiff([sorted(entries, key=lambda e: e[0])], pixels.tobytes(), endian=endian)


# 타일 TIFF(무압축 8비트 회색, 타일 side×side, 그림 크기는 side의 배수): 타일마다 자기 화소 블록을 따로
# 가진다.
def _tiled_tiff(pixels: np.ndarray, side: int) -> bytes:
    height, width = pixels.shape
    tiles = [
        pixels[y : y + side, x : x + side].tobytes()
        for y in range(0, height, side)
        for x in range(0, width, side)
    ]
    offsets = [8 + side * side * i for i in range(len(tiles))]
    entries = [
        (256, 4, 1, struct.pack("<L", width)),
        (257, 4, 1, struct.pack("<L", height)),
        (258, 3, 1, struct.pack("<H", 8)),
        (259, 3, 1, struct.pack("<H", 1)),
        (262, 3, 1, struct.pack("<H", 1)),
        (277, 3, 1, struct.pack("<H", 1)),
        (322, 3, 1, struct.pack("<H", side)),
        (323, 3, 1, struct.pack("<H", side)),
        (324, 4, len(tiles), struct.pack(f"<{len(tiles)}L", *offsets)),
        (325, 4, len(tiles), struct.pack(f"<{len(tiles)}L", *([side * side] * len(tiles)))),
    ]
    return _tiff([entries], b"".join(tiles))


# JPEG 마커 세그먼트 하나: FF + 마커 + 2바이트 길이(길이 칸 자신 포함) + 자료.
def _jpeg_segment(marker: int, payload: bytes = b"") -> bytes:
    return b"\xff" + bytes([marker]) + struct.pack(">H", len(payload) + 2) + payload


# 만들어 둔 JPEG의 SOI 바로 뒤에 바이트를 끼운다.
def _jpeg_with(base: bytes, inserted: bytes) -> bytes:
    return base[:2] + inserted + base[2:]


# PNG 청크 하나(길이 + 종류 + 자료 + CRC).
def _png_chunk(kind: bytes, payload: bytes = b"") -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


_PNG_AFTER_IHDR = 8 + 12 + 13  # 서명 + IHDR 청크(길이 4·종류 4·자료 13·CRC 4)


# PNG의 IHDR 바로 뒤(첫 IDAT 앞)와 IEND 바로 앞에 청크를 끼운다.
def _png_with(base: bytes, before: bytes = b"", after: bytes = b"") -> bytes:
    return base[:_PNG_AFTER_IHDR] + before + base[_PNG_AFTER_IHDR:-12] + after + base[-12:]


# 첫 IDAT들의 자료를 piece바이트씩의 IDAT 여럿으로 다시 쪼갠다(화소는 그대로).
def _png_resplit(png: bytes, piece: int) -> bytes:
    out, data, pos = bytearray(png[:8]), bytearray(), 8
    while pos < len(png):
        (length,) = struct.unpack_from(">I", png, pos)
        kind, payload = png[pos + 4 : pos + 8], png[pos + 8 : pos + 8 + length]
        if kind == b"IDAT":
            data += payload
        else:
            for i in range(0, len(data), piece):
                out += _png_chunk(b"IDAT", bytes(data[i : i + piece]))
            data = bytearray()
            out += _png_chunk(kind, payload)
        pos += 12 + length
    return bytes(out)


# 압축을 푸는 PNG 청크 하나(iCCP·zTXt·iTXt). 자료는 0 바이트 size개를 deflate한 것이라 청크는 작아도 Pillow는
# 청크마다 size바이트를 푼다. zTXt의 키가 비면 Pillow의 텍스트 메모리 합계(64MiB)에도 잡히지 않는다.
def _inflating_chunk(kind: bytes, size: int = 1000, key: bytes = b"k") -> bytes:
    blob = zlib.compress(b"\x00" * size, 9)
    if kind == b"iCCP":
        return _png_chunk(kind, key + b"\x00\x00" + blob)
    if kind == b"zTXt":
        return _png_chunk(kind, key + b"\x00\x00" + blob)
    # iTXt: 압축 표시 1, 압축 방식 0, 언어·번역 키 없음
    return _png_chunk(kind, key + b"\x00\x01\x00\x00\x00" + blob)


# 점진(progressive) JPEG 하나. 같은 변 길이·모드는 한 번만 만든다(5000×5000은 만드는 데 0.3초쯤 걸린다).
@functools.cache
def _progressive_jpeg(side: int, mode: str = "RGB") -> bytes:
    return _encode(Image.new(mode, (side, side), 90), "JPEG", progressive=True, quality=75)


# 엔트로피 자료가 끝나는 곳: 0xFF 다음이 0x00(채움)·RST(0xD0~0xD7)·0xFF가 아닌 첫 자리.
_NEXT_MARKER = re.compile(rb"\xff[^\x00\xd0-\xd7\xff]")


# 점진 JPEG의 첫 스캔이 끝난 자리에 첫 스캔의 SOS 머리 + 자료 1바이트(짧은 스캔)를 n번 끼운다. libjpeg는
# 스캔마다 그림 전체의 MCU를 돌므로(자료가 모자라면 0으로 채운다) 비용이 스캔 수 × 화소 수에 비례한다.
def _with_extra_scans(jpeg: bytes, n: int) -> bytes:
    first = jpeg.index(b"\xff\xda")
    header = jpeg[first : first + 2 + ((jpeg[first + 2] << 8) | jpeg[first + 3])]
    end = _NEXT_MARKER.search(jpeg, first + len(header)).start()
    return jpeg[:end] + (header + b"\x00") * n + jpeg[end:]


# BMP 하나: 파일 머리 14바이트 + 정보 머리(header 바이트, 40·52·56·64·108·124는 BITMAPINFOHEADER 꼴, 12는 압축
# 칸이 없는 BITMAPCOREHEADER) + 회색이 아닌 팔레트(모드 P) + 화소 자료. compression 1은 RLE8, 2는 RLE4.
def _bmp(
    width: int, height: int, pixels: bytes, *, bits: int = 8, compression: int = 1, header: int = 40
) -> bytes:
    colors = 1 << bits
    if header == 12:
        info = struct.pack("<IHHHH", 12, width, height, 1, bits)
        palette = b"".join(bytes((i, 255 - i, (i * 7) & 255)) for i in range(colors))
    else:
        info = struct.pack(
            "<IiiHHIIiiII", header, width, height, 1, bits, compression, len(pixels), 2835, 2835, colors, 0
        )
        info += b"\x00" * (header - 40)
        palette = b"".join(bytes((i, 255 - i, (i * 7) & 255, 0)) for i in range(colors))
    offset = 14 + len(info) + len(palette)
    return b"BM" + struct.pack("<IHHI", offset + len(pixels), 0, 0, offset) + info + palette + pixels


# 정상 RLE8·RLE4 자료(아래 줄부터, 같은 값이 이어지는 구간마다 부호화 명령 하나, 줄 끝 00 00, 끝 00 01).
def _rle(indices: np.ndarray, *, rle4: bool = False) -> bytes:
    out = bytearray()
    for row in indices[::-1].tolist():
        i = 0
        while i < len(row):
            j = i
            while j + 1 < len(row) and row[j + 1] == row[i] and j - i < 254:
                j += 1
            out += bytes((j - i + 1, (row[i] << 4) | row[i] if rle4 else row[i]))
            i = j + 1
        out += b"\x00\x00"
    return bytes(out + b"\x00\x01")


# 1×1 RLE8 BMP로 길이가 정확히 total바이트인 것: 줄 끝(00 00)만 되풀이한다. Pillow의 파이썬 RLE 디코더는 이
# 명령마다 한 바퀴를 돌지만 화소는 늘지 않아 파일 끝까지 돈다(파일 크기에 비례하는 CPU).
def _rle_bmp_of_length(total: int, compression: int = 1) -> bytes:
    bits = 8 if compression == 1 else 4
    head = len(_bmp(1, 1, b"", bits=bits, compression=compression))
    pairs, odd = divmod(total - head - 2, 2)
    return _bmp(1, 1, b"\x00\x00" * pairs + b"\x00\x01" + b"\x00" * odd, bits=bits, compression=compression)


def _picture(seed: int = 0, shape=(48, 64)) -> np.ndarray:
    """부드러운 바탕에 약한 잡음을 얹은 작은 컬러 그림(인코더마다 다르게 저장해 디코딩 결과를 견준다)."""
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    base = np.stack([x * 3, y * 4, (x + y) * 2], axis=-1) % 256
    noise = np.random.default_rng(seed).integers(0, 8, base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _encode(image, fmt: str, **options) -> bytes:
    image = image if isinstance(image, Image.Image) else Image.fromarray(image)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **options)
    return buffer.getvalue()


# 구조가 정상 범위에서 많은 파일들을 포함한 정상 입력 모음. 이름 -> 바이트. 모두 Pillow가 열고 RGB로 바꿀 수
# 있어야 한다(아래 시험이 확인).
@functools.cache
def _legit_files() -> dict[str, bytes]:
    pic = _picture()
    gray = pic[..., 0]
    files: dict[str, bytes] = {}
    # PNG: 압축 수준, 팔레트, 회색, 알파(불투명), 텍스트 청크 여럿, 작은 IDAT 조각 여럿, 애니메이션
    files["png"] = _encode(pic, "PNG")
    files["png-level0"] = _encode(pic, "PNG", compress_level=0)
    files["png-palette"] = _encode(Image.fromarray(pic).quantize(32), "PNG")
    files["png-gray"] = _encode(gray, "PNG")
    info = PngImagePlugin.PngInfo()
    for i in range(200):
        info.add_text(f"key{i}", f"value{i}")
    files["png-200-text-chunks"] = _encode(pic, "PNG", pnginfo=info)
    files["png-thousands-of-idat-pieces"] = _png_resplit(
        _encode(_picture(1, (200, 256)), "PNG", compress_level=0), 64
    )
    frames = [Image.fromarray(_picture(i)) for i in range(30)]
    files["apng-30-frames"] = _encode(frames[0], "PNG", save_all=True, append_images=frames[1:])
    files["png-without-iend"] = files["png"][:-12]
    # 압축을 푸는 청크가 많은 정상 PNG: ICC 프로파일(iCCP) 하나 + 압축 텍스트 zTXt 10개·iTXt 10개(실제
    # 파일에서 센 최대는 2개였다)
    packed = PngImagePlugin.PngInfo()
    for i in range(10):
        packed.add_text(f"zkey{i}", "z" * 500, zip=True)
        packed.add_itxt(f"ikey{i}", "i" * 500, zip=True)
    files["png-icc-and-20-compressed-text-chunks"] = _encode(
        pic, "PNG", pnginfo=packed, icc_profile=b"\x00" * 3000
    )
    # JPEG: 기본·점진·최적화, 회색, CMYK, 주석, ICC(여러 APP2 조각), EXIF, XMP, Photoshop 자원
    files["jpeg"] = _encode(pic, "JPEG", quality=90)
    files["jpeg-progressive"] = _encode(pic, "JPEG", progressive=True, optimize=True)
    files["jpeg-gray"] = _encode(gray, "JPEG")
    files["jpeg-cmyk"] = _encode(Image.fromarray(pic).convert("CMYK"), "JPEG")
    files["jpeg-comment"] = _encode(pic, "JPEG", comment=b"inspection line 3")
    files["jpeg-icc-1.5MB-in-23-segments"] = _encode(pic, "JPEG", icc_profile=b"\x00" * 1_500_000)
    exif = Image.Exif()
    exif[0x010F] = "camera maker"
    exif[0x0112] = 1
    files["jpeg-exif"] = _encode(pic, "JPEG", exif=exif)
    xmp = b"http://ns.adobe.com/xap/1.0/\x00" + b"<x:xmpmeta>" + b"a" * 60_000 + b"</x:xmpmeta>"
    photoshop = (
        b"Photoshop 3.0\x00"
        + (b"8BIM" + struct.pack(">H", 0x0400) + b"\x00\x00" + struct.pack(">I", 4) + b"abcd") * 200
    )
    files["jpeg-xmp-and-photoshop"] = _jpeg_with(
        files["jpeg"], _jpeg_segment(0xE1, xmp) + _jpeg_segment(0xED, photoshop)
    )
    files["jpeg-100-comments"] = _jpeg_with(files["jpeg"], _jpeg_segment(0xFE, b"note") * 100)
    # 점진 JPEG의 스캔 수: libjpeg(Pillow·OpenCV)의 기본 순서는 YCbCr 10, 회색 6, CMYK 18이다
    files["jpeg-progressive-cmyk-18-scans"] = _encode(
        Image.fromarray(pic).convert("CMYK"), "JPEG", progressive=True
    )
    files["jpeg-progressive-gray-6-scans"] = _encode(gray, "JPEG", progressive=True)
    # EOI 뒤에 붙은 자료(동영상이 붙은 사진처럼)에 FF DA가 많아도 스캔이 아니다: libjpeg는 EOI에서 멈춘다
    files["jpeg-with-data-after-the-end-of-image"] = files["jpeg"] + (b"\xff\xda\x00\x08" + bytes(6)) * 300
    # TIFF: 압축 방식, 색 모드, 여러 쪽, 줄마다 스트립(3,000줄), 16×16 타일, 큰 쪽 바이트 순서, BigTIFF, Exif
    # 하위 IFD
    for name, options in {
        "raw": {},
        "packbits": {"compression": "packbits"},
        "lzw": {"compression": "tiff_lzw"},
        "deflate": {"compression": "tiff_adobe_deflate"},
        "jpeg": {"compression": "jpeg"},
    }.items():
        files[f"tiff-{name}"] = _encode(pic, "TIFF", **options)
    files["tiff-gray"] = _encode(gray, "TIFF")
    files["tiff-palette"] = _encode(Image.fromarray(pic).quantize(16), "TIFF")  # COLORMAP(SHORT 768개)
    files["tiff-1bit-group4"] = _encode(Image.fromarray(gray).convert("1"), "TIFF", compression="group4")
    files["tiff-cmyk"] = _encode(Image.fromarray(pic).convert("CMYK"), "TIFF")
    files["tiff-ycbcr"] = _encode(Image.fromarray(pic).convert("YCbCr"), "TIFF")
    files["tiff-two-pages"] = _encode(
        Image.fromarray(pic), "TIFF", save_all=True, append_images=[Image.fromarray(gray)]
    )
    tall = np.random.default_rng(3).integers(0, 256, (3000, 40), dtype=np.uint8)
    files["tiff-big-endian-small"] = _tiff([_gray_entries(">")], endian=">")
    files["tiff-3000-strips"] = _striped_tiff(tall, 1)
    files["tiff-3000-strips-big-endian"] = _striped_tiff(tall, 1, ">")
    files["tiff-16px-tiles"] = _tiled_tiff(
        np.random.default_rng(4).integers(0, 256, (160, 192), dtype=np.uint8), 16
    )
    exif_ifd = [
        (0x829A, 5, 1, struct.pack("<LL", 1, 125)),
        (0x829D, 5, 1, struct.pack("<LL", 28, 10)),
        (0x9000, 7, 4, b"0230"),
    ]
    exif_ifd += [(0xA000 + i, 3, 1, struct.pack("<H", i)) for i in range(30)]
    gps_ifd = [
        (1, 2, 2, b"N\0"),
        (2, 5, 3, _repeat("<", 5, (37, 1), 3)),
        (3, 2, 2, b"E\0"),
        (4, 5, 3, _repeat("<", 5, (127, 1), 3)),
    ]
    files["tiff-exif-and-gps-sub-ifds"] = _tiff(
        [_gray_entries(extra=[(34665, 4, 1, ("IFD", 1)), (34853, 4, 1, ("IFD", 2))]), exif_ifd, gps_ifd]
    )
    one = _gray_entries(head=16)
    files["bigtiff"] = _tiff([one], big=True)
    files["tiff-chain-of-10-pages"] = _tiff([_gray_entries()] + [_gray_entries()] * 9, chain=True)
    # BMP: 무압축(Pillow가 저장한 것)과 상한보다 작은 RLE8·RLE4(Pillow는 RLE로 저장하지 못해 시험 안에서
    # 만든다)
    files["bmp"] = _encode(pic, "BMP")
    stripes = np.add.outer(np.arange(48) // 2, np.arange(64) // 5)
    files["bmp-rle8"] = _bmp(64, 48, _rle((stripes * 7 % 256).astype(np.uint8)))
    files["bmp-rle4"] = _bmp(64, 48, _rle((stripes % 16).astype(np.uint8), rle4=True), bits=4, compression=2)
    # 이 검사가 건드리지 않는 형식
    files["webp"] = _encode(pic, "WEBP", lossless=True)
    return files


# 폭탄 입력: 작은 구조 항목이 수백만 개 든 20MiB 안팎의 파일(서비스가 받는 크기 그대로다). 이름 -> 바이트를
# 만드는 함수. 각 주석은 고치기 전의 비용이다(이 PC, Windows, Pillow 13.0.0).
def _bombs() -> dict:
    return {
        # StripOffsets 260만 개(폭 1 × 높이 260만, 줄마다 스트립): 고치기 전에 CPU 22초·+837MiB
        "tiff-2.6M-strips": lambda: _tiff([_gray_entries(height=2_600_000, strips=2_600_000)]),
        # StripByteCounts를 뺀 변형: Pillow는 받는다. 같은 20MiB에 스트립이 두 배(520만 개)
        "tiff-5.2M-strips-without-counts": lambda: _tiff(
            [_gray_entries(height=5_200_000, strips=5_200_000, with_counts=False)]
        ),
        # 타일 표 300만 개
        "tiff-3M-tiles": lambda: _tiff(
            [_gray_entries(extra=[(324, 4, 3_000_000, _repeat("<", 4, 8, 3_000_000))])]
        ),
        # Exif 하위 IFD의 RATIONAL 배열 260만 개: 열 때가 아니라 디코딩이 끝난 뒤 Pillow가 푼다(고치기 전에
        # CPU 10초·+397MiB)
        "tiff-2.6M-rationals-in-the-exif-ifd": lambda: _tiff(
            [
                _gray_entries(extra=[(34665, 4, 1, ("IFD", 1))]),
                [(0x9999, 5, 2_600_000, _repeat("<", 5, (1, 3), 2_600_000))],
            ]
        ),
        # 첫 IFD의 SHORT 배열 1,000만 개(YCbCrSubSampling 자리): 파이썬 수 1,000만 개의 튜플
        "tiff-10M-shorts": lambda: _tiff(
            [_gray_entries(extra=[(530, 3, 10_000_000, _repeat("<", 3, 1000, 10_000_000))])]
        ),
        # BigTIFF는 항목 개수 칸이 64비트라 파일이 허락하는 만큼(20MiB에 100만 개) 항목을 적을 수 있다. 20만
        # 개
        "bigtiff-200k-entries": lambda: _tiff(
            [_gray_entries(head=16, extra=[(1000 + i % 30000, 1, 1, b"\x01") for i in range(200_000)])],
            big=True,
        ),
        # 길이 0인 APP5 마커(0xFFE50002) 520만 개
        "jpeg-5.2M-markers": lambda: _jpeg_with(_encode(_picture(), "JPEG"), _jpeg_segment(0xE5) * 5_200_000),
        # 세그먼트 사이의 쓰레기 바이트 2,000만 개(Pillow는 한 바이트씩 파이썬 반복으로 건너뛴다)
        "jpeg-20MB-of-stray-bytes": lambda: (
            b"\xff\xd8\xff\x00" + b"A" * 20_000_000 + _encode(_picture(), "JPEG")[2:]
        ),
        # 채움 바이트(FF) 2,000만 개
        "jpeg-20MB-of-fill-bytes": lambda: (
            b"\xff\xd8" + b"\xff" * 20_000_000 + _encode(_picture(), "JPEG")[2:]
        ),
        # 첫 SOS까지가 3MiB(Photoshop 자원 세그먼트 48개): 세그먼트 안을 파이썬으로 훑는 처리는 길이에
        # 비례한다
        "jpeg-3MB-header": lambda: _jpeg_with(
            _encode(_picture(), "JPEG"),
            _jpeg_segment(0xED, b"Photoshop 3.0\x00" + b"8BIM\x04\x00\x00\x00\x00\x00\x00\x00" * 5_400) * 48,
        ),
        # 알 수 없는(소문자 둘째 글자) 사설 청크 170만 개를 첫 IDAT 앞에
        "png-1.7M-private-chunks-before-idat": lambda: _png_with(
            _encode(_picture(), "PNG"), before=_png_chunk(b"abCD") * 1_700_000
        ),
        # 같은 청크를 IDAT 뒤 IEND 앞에(디코딩이 끝난 뒤 Pillow가 IEND까지 읽는다)
        "png-1.7M-private-chunks-after-idat": lambda: _png_with(
            _encode(_picture(), "PNG"), after=_png_chunk(b"abCD") * 1_700_000
        ),
        # 빈 IDAT 170만 개(객체는 남기지 않지만 조각마다 파이썬 반복을 돈다)
        "png-1.7M-empty-idat-chunks": lambda: _png_with(
            _encode(_picture(), "PNG"), before=_png_chunk(b"IDAT") * 1_700_000
        ),
        # 아래는 독립 검토가 첫 사전 검사를 우회한 모양을 크기 그대로 만든 것이다(고치기 전의 사전 검사는
        # 통과시켰다). StripOffsets를 BYTE형 2,000만 개로: Pillow는 bytes 하나로 두었다가 바이트마다 _Tile을
        # 만든다(20만 개에서 1.6초. 이 크기면 약 130초·4.4GiB로 외삽)
        "tiff-20M-strip-offsets-typed-byte": lambda: _tiff(
            [
                sorted(
                    [e for e in _gray_entries(height=2, with_counts=False) if e[0] != 273]
                    + [(273, 1, 20_000_000, bytes(20_000_000))],
                    key=lambda e: e[0],
                )
            ]
        ),
        # Exif 가리킴의 개수를 2로: Pillow는 첫 값을 따라가 RATIONAL 260만 개를 푼다(고치기 전 CPU 8~10초)
        "tiff-2.6M-rationals-behind-a-two-value-exif-pointer": lambda: _tiff(
            [
                _gray_entries(extra=[(34665, 4, 2, ("IFD", 1, "L", 2))]),
                [(0x9999, 5, 2_600_000, _repeat("<", 5, (1, 3), 2_600_000))],
            ]
        ),
        # 첫 IFD에는 가짜 Interop 가리킴, Exif IFD 안에 진짜 Interop 가리킴: Pillow는 Exif IFD 안의 것을
        # 읽는다
        "tiff-2.6M-rationals-behind-the-interop-pointer-in-the-exif-ifd": lambda: _tiff(
            [
                _gray_entries(extra=[(34665, 4, 1, ("IFD", 1)), (40965, 4, 1, ("IFD", 3))]),
                [(40965, 4, 1, ("IFD", 2))],
                [(0x9999, 5, 2_600_000, _repeat("<", 5, (1, 3), 2_600_000))],
                [(0x9998, 3, 1, struct.pack("<H", 1))],
            ]
        ),
        # IDAT 뒤의 iCCP 9,999개(청크 하나 약 1KB가 1MB로 풀린다. 고치기 전 CPU 25초)
        "png-9999-iccp-chunks-after-idat": lambda: _png_with(
            _encode(_picture(), "PNG"), after=_inflating_chunk(b"iCCP", 1_000_000) * 9_999
        ),
        # 키가 빈 zTXt 9,999개(Pillow의 텍스트 메모리 합계에 잡히지 않는다. 1,000개에서 3초)
        "png-9999-empty-key-ztxt-chunks": lambda: _png_with(
            _encode(_picture(), "PNG"), after=_inflating_chunk(b"zTXt", 1_000_000, key=b"") * 9_999
        ),
        # 5000×5000 점진 JPEG에 짧은 스캔 130만 개(스캔마다 그림 전체의 MCU를 돈다. 3만 개에서 37초, 이 크기는
        # 약 30분으로 외삽)
        "jpeg-1.3M-progressive-scans": lambda: _with_extra_scans(_progressive_jpeg(5000), 1_300_000),
        # 1×1 RLE8 BMP에 줄 끝 명령 1,040만 개(파이썬 RLE 디코더가 명령마다 한 바퀴. 고치기 전 CPU 5초)
        "bmp-rle8-10.4M-end-of-line-commands": lambda: _bmp(1, 1, b"\x00\x00" * 10_400_000 + b"\x00\x01"),
    }


# 호출이 싸게 끝나는지 본다: 쓴 CPU 시간과 파이썬 메모리 최고치가 20MiB짜리 파일을 훑는 것으로는 터무니없이 큰
# 값 아래여야 한다(고치기 전에 이 입력들은 `Image.open`과 디코딩에서 8~22초·0.2~0.8GiB를 썼다). 시간은
# tracemalloc을 끈 첫 호출에서 프로세스의 CPU 시간으로(다른 프로그램이 CPU를 나눠 써도 흔들리지 않게,
# tracemalloc이 파이썬 코드를 몇 배 느리게 만드는 것도 피하려고), 메모리는 켠 둘째 호출에서 잰다. 입력
# 바이트열은 호출 앞에서 만들므로 그 메모리는 들어가지 않는다. 호출은 두 번 불린다.
def _assert_cheap(call, *, seconds: float, megabytes: float) -> None:
    began = time.process_time()
    call()
    elapsed = time.process_time() - began
    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        call()
        peak = tracemalloc.get_traced_memory()[1] - before
    finally:
        if started:
            tracemalloc.stop()
    assert elapsed < seconds, f"used {elapsed:.2f}s of CPU"
    assert peak < megabytes * 2**20, f"peaked at {peak / 2**20:.1f}MiB"


# ==================================================================================================
# 1. 상수 표가 설치된 Pillow와 맞는다
# ==================================================================================================
# 이 검사는 Pillow가 읽는 순서와 규칙을 따라가므로, 따라가는 표가 Pillow의 것과 달라지면(Pillow를 올렸을 때)
# 조용히 어긋난다. 그 표들을 Pillow의 것과 대조해 어긋나면 이 시험이 먼저 실패하게 한다.
def test_the_format_tables_mirror_the_installed_pillow():
    assert set(pre._TIFF_PREFIXES) == set(TiffImagePlugin.PREFIXES)
    dispatch = TiffImagePlugin.ImageFileDirectory_v2._load_dispatch
    # BYTE·ASCII·UNDEFINED는 풀 때 객체 하나(bytes·str)로 남는다. 그중 BYTE를 Pillow가 나중에 원소마다 도는
    # 태그는 따로 센다(아래 전제 시험과 BYTE 시험)
    assert pre._TIFF_NUMERIC_TYPES == set(dispatch) - {1, 2, 7}
    for kind, (fmt, width) in pre._TIFF_INTEGER_WIDTHS.items():
        assert dispatch[kind][0] == width == struct.calcsize("=" + fmt)
    assert pre._TIFF_SUB_IFD_TAGS == set(TiffTags.TAGS_V2_GROUPS)
    assert pre._TIFF_TABLE_TAGS == {TiffImagePlugin.STRIPOFFSETS, TiffImagePlugin.STRIPBYTECOUNTS, 324, 325}
    assert pre._PNG_SIGNATURE == PngImagePlugin._MAGIC
    for kind in (b"IDAT", b"abCD", b"tEXt", b"a_1Z", b"ID T", b"ID\xe4T", b"\xff\xff\xff\xff", b"12 3"):
        assert bool(pre._PNG_CHUNK_TYPE.match(kind)) == bool(PngImagePlugin.is_cid(kind)), kind
    # JPEG: 0xC0~0xFE는 모두 Pillow의 마커 표에 있고, 길이 칸이 있는 것은 핸들러가 있는 것과 같다. 0xC0 아래는
    # 표에 없다.
    for second in range(0x100):
        entry = JpegImagePlugin.MARKER.get(0xFF00 | second)
        assert (entry is not None) == (0xC0 <= second <= 0xFE), hex(second)
        if entry is not None:
            assert (entry[2] is not None) == pre._jpeg_marker_has_length(second), hex(second)


# 고친 규칙이 기대는 Pillow의 동작이 설치된 Pillow에서도 참인지 본다(Pillow를 올려 동작이 바뀌면 여기서 먼저
# 실패한다).
# - TIFF: BYTE형 값은 bytes 하나로 저장되고(그래서 원소마다 도는 곳에서는 바이트마다 돈다), 원소마다 도는 태그
#   표는 Pillow의 태그 번호와 같다. 하위 IFD 가리킴 태그는 개수 1로 정의돼 있어 값이 여럿이면 Pillow가 첫 값을
#   쓴다(Exif IFD 안의 Interop 가리킴도 같다).
# - PNG: 압축을 푸는 세 청크 모두 Pillow에 처리기가 있고 청크 하나를 1MiB까지 푼다.
# - BMP: 압축 칸을 읽는 정보 머리 크기와 RLE 압축 번호가 Pillow와 같고, RLE는 파이썬 디코더가 푼다.
def test_the_premises_of_the_new_rules_hold_in_the_installed_pillow():
    directory = TiffImagePlugin.ImageFileDirectory_v2()
    directory.tagtype[TiffImagePlugin.STRIPOFFSETS] = TiffTags.BYTE
    directory[TiffImagePlugin.STRIPOFFSETS] = b"\x08\x09\x0a"
    assert directory[TiffImagePlugin.STRIPOFFSETS] == b"\x08\x09\x0a"  # 튜플이 아니라 bytes 하나
    assert pre._TIFF_BYTE_WALKED_TAGS == {
        TiffImagePlugin.STRIPOFFSETS,
        TiffImagePlugin.STRIPBYTECOUNTS,
        TiffImagePlugin.TILEOFFSETS,
        325,  # TileByteCounts(Pillow에 상수 이름이 없다)
        TiffImagePlugin.COLORMAP,
        TiffImagePlugin.SAMPLEFORMAT,
        TiffImagePlugin.EXTRASAMPLES,
    }
    for tag in pre._TIFF_SUB_IFD_TAGS:
        assert TiffTags.lookup(tag).length == 1, tag
    assert TiffTags.lookup(40965, 34665).length == 1  # Exif IFD 안의 Interop 가리킴
    assert all(hasattr(PngImagePlugin.PngStream, f"chunk_{k.decode()}") for k in pre._PNG_INFLATING_CHUNKS)
    assert PngImagePlugin.MAX_TEXT_CHUNK == 1 << 20
    accepted = {size for size in range(256) if BmpImagePlugin._dib_accept(struct.pack("<I", size))}
    assert pre._BMP_INFO_HEADER_SIZES == accepted - {12}  # 12(BITMAPCOREHEADER)에는 압축 칸이 없다
    compressions = BmpImagePlugin.BmpImageFile.COMPRESSIONS
    assert pre._BMP_RLE_COMPRESSIONS == {compressions["RLE8"], compressions["RLE4"]}
    assert issubclass(Image.DECODERS["bmp_rle"], ImageFile.PyDecoder)


# ==================================================================================================
# 2. 정상 파일은 통과한다
# ==================================================================================================
@pytest.mark.parametrize("name", sorted(_legit_files()))
def test_ordinary_files_pass_and_still_open_as_before(name):
    data = _legit_files()[name]
    pre.check_image_structure(data)
    # 모음의 모든 파일이 실제로 열린다(검사가 통과시킨 것이 쓸모없는 파일이 아니다)
    Image.open(io.BytesIO(data)).convert("RGB")


# ==================================================================================================
# 3. 폭탄 입력은 거절되고 검사가 싸게 끝난다
# ==================================================================================================
@pytest.mark.parametrize("name", sorted(_bombs()))
def test_structure_bombs_are_refused_cheaply(name):
    data = _bombs()[name]()
    assert len(data) < 21 * 2**20  # 서비스가 받는 크기 안(파일 한도 20MiB + 여유)이다

    def check():
        with pytest.raises(pre.ImageStructureError):
            pre.check_image_structure(data)

    _assert_cheap(check, seconds=2.0, megabytes=8)


# ==================================================================================================
# 4. 상한 경계: 상한 개수까지는 통과하고 하나 더는 거절한다 (상한을 시험 안에서 작은 값으로 바꿔 본다)
# ==================================================================================================
def _passes(data: bytes) -> bool:
    try:
        pre.check_image_structure(data)
    except pre.ImageStructureError:
        return False
    return True


# 개수 n일 때 만든 입력이 상한 limit에서 n == limit이면 통과, n == limit + 1이면 거절인지 본다.
def _inclusive(monkeypatch, name: str, limit: int, build) -> None:
    monkeypatch.setattr(pre, name, limit)
    assert _passes(build(limit)), f"{name}={limit}: exactly the limit must pass"
    assert not _passes(build(limit + 1)), f"{name}={limit}: one more must be refused"


# 항목이 정확히 n개인 첫 IFD(기본 항목에 사설 BYTE 항목을 채운다).
def _tiff_with_entries(n: int, *, endian: str = "<", big: bool = False) -> bytes:
    base = _gray_entries(endian, head=16 if big else 8)
    return _tiff([base + [(1000 + i, 1, 1, b"\x01") for i in range(n - len(base))]], endian=endian, big=big)


@pytest.mark.parametrize(
    ("endian", "big"), [("<", False), (">", False), ("<", True)], ids=["II", "MM", "BigTIFF"]
)
def test_the_tiff_entry_limit_is_inclusive(monkeypatch, endian, big):
    _inclusive(monkeypatch, "MAX_TIFF_ENTRIES", 40, lambda n: _tiff_with_entries(n, endian=endian, big=big))


# 첫 IFD 항목 중 숫자 배열 합계에 드는 원소 수(스트립·타일 표 네 가지는 따로 센다). 폭·높이 같은 개수 1짜리
# 숫자 항목도 Pillow가 푸는 값이라 센다.
def _counted(entries) -> int:
    return sum(
        c
        for tag, kind, c, _ in entries
        if kind in pre._TIFF_NUMERIC_TYPES and tag not in pre._TIFF_TABLE_TAGS
    )


# 스트립·타일 표 네 가지(273, 279, 324, 325) 각각의 원소 수 상한. 273은 스트립 수 자체이고 나머지는 표 하나를
# 더한다.
@pytest.mark.parametrize("tag", [273, 279, 324, 325])
def test_the_strip_and_tile_table_limit_is_inclusive_for_all_four_tables(monkeypatch, tag):
    def build(n: int) -> bytes:
        if tag == 273:
            return _tiff([_gray_entries(strips=n)])
        return _tiff([_gray_entries(with_counts=False, extra=[(tag, 4, n, _repeat("<", 4, 1, n))])])

    _inclusive(monkeypatch, "MAX_TIFF_STRIPS", 10, build)


# 숫자 배열 합계 상한: 원소 수는 형식과 무관하게 개수로 세고(RATIONAL 하나가 원소 하나), 표 네 가지는 따로
# 세므로 합계에 들지 않는다. 바이트·문자열· UNDEFINED·모르는 형식은 아무리 길어도 세지 않는다. 한 파일의 첫
# IFD와 하위 IFD를 합쳐 센다.
@pytest.mark.parametrize("kind", [3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 16])
def test_the_numeric_value_limit_is_inclusive_for_every_decoded_type(monkeypatch, kind):
    value = (1, 3) if kind in (5, 10) else 1
    used = _counted(_gray_entries())  # 기본 항목이 이미 쓰는 원소 수

    def build(n: int) -> bytes:  # 숫자 원소 합계가 정확히 n이 되게 항목 하나를 더한다
        extra = (0x9999, kind, n - used, _repeat("<", kind, value, n - used))
        return _tiff([_gray_entries(extra=[extra])])

    _inclusive(monkeypatch, "MAX_TIFF_VALUES", 50, build)


# BYTE·ASCII·UNDEFINED 값은 Pillow가 풀 때 bytes·str 하나로 두므로 대부분의 태그에서는 원소마다 일을 하지
# 않는다. 그래서 그런 태그는 아무리 길어도 세지 않는다(XMP·Photoshop 자원처럼 정상 파일에서도 수 MB인 BYTE
# 값이 있다). 그러나 BYTE형으로 적은 몇 태그는 Pillow·libtiff가 그 bytes를 바이트마다
# 돈다(_TIFF_BYTE_WALKED_TAGS, 근거는 image_precheck.py의 주석). 고치기 전에는 그 태그도 세지 않아 BYTE형
# StripOffsets 20만 개(0.19MiB)가 검사를 통과하고 `_decode`에서 1.6초를 썼다. ASCII·UNDEFINED로 적은 표는
# Pillow가 원소 하나짜리 튜플로 싸서 돌지 않으므로 세지 않고, 실제로 싸다(아래에서 Pillow로 확인한다).
# 17·18·99는 이 Pillow가 버리는 형식이다.
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_byte_values_count_only_for_the_tags_pillow_walks_element_by_element(monkeypatch):
    monkeypatch.setattr(pre, "MAX_TIFF_VALUES", 10)
    for tag in (0x9999, 700, 34377, 34675):  # 사설 태그, XMP, Photoshop 자원, ICC 프로파일
        for kind in (1, 2, 7, 17, 18, 99):
            huge = (tag, kind, 5_000_000, b"\x00" * 16)
            assert _passes(_tiff([_gray_entries(extra=[huge])])), (tag, kind)
    # 스트립 표는 합계에 들지 않는다: 표 네 개가 각각 합계 상한보다 길어도 표 상한 안이면 통과한다
    monkeypatch.setattr(pre, "MAX_TIFF_STRIPS", 100)
    tables = [(tag, 4, 80, _repeat("<", 4, 1, 80)) for tag in (279, 324, 325)]
    assert _passes(_tiff([_gray_entries(strips=80, with_counts=False, extra=tables)]))
    # BYTE형으로 적은, Pillow가 원소마다 도는 태그는 센다: 표 네 가지는 표 상한, 나머지는 숫자 합계에 든다
    for tag in (273, 279, 324, 325):
        base = [e for e in _gray_entries(height=2, with_counts=False) if e[0] != tag]
        table = sorted([*base, (tag, 1, 101, bytes(101))], key=lambda e: e[0])
        assert not _passes(_tiff([table])), tag
    for tag in (320, 338, 339):
        assert not _passes(_tiff([_gray_entries(extra=[(tag, 1, 11, bytes(11))])])), tag
    # ASCII·UNDEFINED로 적은 스트립·타일 표는 세지 않는다. Pillow도 돌지 않아 크기를 키워도 열고 바꾸는 데
    # 싸다
    monkeypatch.setattr(pre, "MAX_TIFF_STRIPS", 100_000)
    for tag in (273, 324):
        for kind in (2, 7):
            base = [e for e in _gray_entries(height=2, with_counts=False) if e[0] not in (273, 278)]
            base += [(278, 4, 1, struct.pack("<L", 1))]
            if tag == 324:
                base += [(322, 3, 1, struct.pack("<H", 16)), (323, 3, 1, struct.pack("<H", 16))]
            data = _tiff([sorted([*base, (tag, kind, 200_000, b"\x01" * 200_000)], key=lambda e: e[0])])
            assert _passes(data), (tag, kind)

            def open_and_convert(data=data):
                try:
                    Image.open(io.BytesIO(data)).convert("RGB")
                except (OSError, ValueError, TypeError, SyntaxError):
                    pass  # 원소 하나짜리 튜플이라 오프셋이 엉뚱해 실패할 수 있다(서비스에서는 400)

            _assert_cheap(open_and_convert, seconds=1.0, megabytes=8)


# BYTE형 표·값의 상한 경계: 표 네 가지(273, 279, 324, 325)는 원소 수가 표 상한, 원소마다 도는 나머지 셋(320
# ColorMap, 338 ExtraSamples, 339 SampleFormat)은 숫자 합계 상한에 든다. 첫 IFD에서만 센다(하위 IFD의 BYTE
# 값은 Pillow가 bytes 하나로 두고 돌지 않는다).
@pytest.mark.parametrize("tag", [273, 279, 324, 325, 320, 338, 339])
def test_byte_typed_values_that_pillow_walks_are_limited_inclusively(monkeypatch, tag):
    if tag in pre._TIFF_TABLE_TAGS:

        def build(n: int) -> bytes:
            base = [e for e in _gray_entries(height=2, with_counts=False) if e[0] != tag]
            return _tiff([sorted([*base, (tag, 1, n, bytes(n))], key=lambda e: e[0])])

        _inclusive(monkeypatch, "MAX_TIFF_STRIPS", 10, build)
    else:
        used = _counted(_gray_entries())

        def build(n: int) -> bytes:
            return _tiff([_gray_entries(extra=[(tag, 1, n - used, bytes(n - used))])])

        _inclusive(monkeypatch, "MAX_TIFF_VALUES", 50, build)
    # 같은 BYTE 값이 하위 IFD에 있으면 세지 않는다
    monkeypatch.setattr(pre, "MAX_TIFF_VALUES", 50)
    sub = [(tag, 1, 5_000, bytes(5_000))]
    assert _passes(_tiff([_gray_entries(extra=[(34665, 4, 1, ("IFD", 1))]), sub]))


# 같은 태그가 되풀이돼도 모두 센다: Pillow는 첫 항목을 쓸 수 없으면(모르는 형식 등) 다음 항목을 쓰므로, 첫
# 항목이 작거나 모르는 형식이라고 해서 뒤의 큰 표를 놓치면 안 된다.
def test_a_repeated_tag_cannot_hide_a_large_table_behind_an_unusable_first_entry(monkeypatch):
    monkeypatch.setattr(pre, "MAX_TIFF_STRIPS", 10)
    # Pillow는 모르는 형식(99)의 항목을 버리고, 같은 태그의 다음 항목을 쓴다
    unusable_first = (273, 99, 1, b"\x00" * 4)
    big_second = (273, 4, 50, _repeat("<", 4, 8, 50))
    entries = [e for e in _gray_entries() if e[0] != 273]
    assert not _passes(_tiff([sorted([*entries, unusable_first, big_second], key=lambda e: e[0])]))


# 하위 IFD(Exif·GPS·Interop)는 열 때가 아니라 디코딩이 끝난 뒤 Pillow가 항목마다 푼다. 가리킴이 들어 있으면 그
# IFD의 항목과 숫자 배열도 센다.
@pytest.mark.parametrize("tag", [34665, 34853, 40965])
@pytest.mark.parametrize(
    ("kind", "width", "big"),
    [
        (3, "H", False),
        (4, "L", False),
        (13, "L", False),
        (9, "l", False),
        (8, "h", False),
        (16, "Q", False),
        (16, "Q", True),
    ],
    ids=["SHORT", "LONG", "IFD", "SLONG", "SSHORT", "LONG8-spilled", "LONG8-BigTIFF"],
)
def test_sub_ifds_are_counted_however_the_pointer_is_stored(monkeypatch, tag, kind, width, big):
    monkeypatch.setattr(pre, "MAX_TIFF_VALUES", 30)
    head = 16 if big else 8
    main = _gray_entries(head=head, extra=[(tag, kind, 1, ("IFD", 1, width))])
    used = _counted(main)  # 첫 IFD가 쓰는 원소 수(가리킴 항목 자신 포함)

    def build(n: int) -> bytes:  # 하위 IFD의 RATIONAL n개를 더해 합계가 used + n
        sub = [(0x9999, 5, n, _repeat("<", 5, (1, 3), n))]
        return _tiff([main, sub], big=big)

    assert _passes(build(30 - used)) and not _passes(build(31 - used))


# SBYTE 가리킴은 오프셋이 127 이하일 때만 쓸 수 있다: 항목이 둘뿐인 아주 작은 첫 IFD 바로 뒤의 하위 IFD라야
# 한다.
def test_a_signed_byte_pointer_is_followed_when_the_offset_fits(monkeypatch):
    monkeypatch.setattr(pre, "MAX_TIFF_VALUES", 30)
    main = [(256, 4, 1, struct.pack("<L", 1)), (34665, 6, 1, ("IFD", 1, "b"))]
    used = _counted(main)

    def build(n: int) -> bytes:
        return _tiff([main, [(0x9999, 5, n, _repeat("<", 5, (1, 3), n))]])

    assert _passes(build(30 - used)) and not _passes(build(31 - used))


def test_every_sub_ifd_is_counted_once_and_in_the_same_total(monkeypatch):
    monkeypatch.setattr(pre, "MAX_TIFF_VALUES", 100)

    def sub(n: int) -> list:  # SHORT n개만 든 하위 IFD
        return [(0x9999, 3, n, _repeat("<", 3, 7, n))]

    pointers = [(34665, 4, 1, ("IFD", 1)), (34853, 4, 1, ("IFD", 2)), (40965, 4, 1, ("IFD", 3))]
    room = 100 - _counted(_gray_entries(extra=pointers))  # 하위 IFD 셋에 남은 원소 수
    # 셋이 합쳐 room개면 통과, 하나 더면 거절한다(IFD마다가 아니라 합계)
    assert _passes(
        _tiff([_gray_entries(extra=pointers), sub(room // 3), sub(room // 3), sub(room - 2 * (room // 3))])
    )
    assert not _passes(
        _tiff(
            [_gray_entries(extra=pointers), sub(room // 3), sub(room // 3), sub(room - 2 * (room // 3) + 1)]
        )
    )
    # 같은 가리킴이 되풀이돼도 (가리킴 태그, 위치) 쌍마다 한 번만 따라간다(되풀이로 검사 비용을 키우지 못하고,
    # Pillow도 가리킴 무리마다 한 번만 푼다): 같은 태그로 두 번 가리킨 IFD를 한 번만 센다
    twice = [(34665, 4, 1, ("IFD", 1)), (34665, 4, 1, ("IFD", 1))]
    assert _passes(_tiff([_gray_entries(extra=twice), sub(100 - _counted(_gray_entries(extra=twice)))]))


# Pillow가 따라갈 수 없는 가리킴은 따라가지 않는다: BYTE(bytes라 seek하지 못한다)와 RATIONAL(IFDRational)이다.
# 개수가 2 이상인 정수 가리킴은 여기에 들지 않는다. 고치기 전의 이 시험은 개수 2짜리를 '따라가지 않는다'고
# 적었지만, 그 값 8이 하위 IFD가 아니라 화소 자료를 가리켜서 싸게 끝났을 뿐이다. 하위 IFD 가리킴 태그는
# Pillow의 태그 표에 개수 1로 정의돼 있어 값이 여럿이면 Pillow가 경고만 내고 첫 값을 쓴다(TiffImagePlugin의
# _setitem). 그래서 개수 2짜리 가리킴이 진짜 하위 IFD를 가리키면 Pillow는 그 IFD를 풀고, 이 검사도 따라가서
# 거절해야 한다(아래 둘째 부분). Pillow가 깨진 가리킴·개수 초과에 내는 UserWarning은 일부러 만든 입력의 결과라
# 걸러 낸다.
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_pointers_are_followed_exactly_when_pillow_can_follow_them(monkeypatch):
    sub = [(0x9999, 3, 400_000, _repeat("<", 3, 1000, 400_000))]  # 열면 SHORT 40만 개
    for pointer in (
        (34665, 1, 1, ("IFD", 1, "B")),  # BYTE: 바이트열이라 Pillow가 seek하지 못한다
        (34665, 5, 1, struct.pack("<LL", 8, 1)),  # RATIONAL
    ):
        data = _tiff([_gray_entries(extra=[pointer]), sub])
        assert _passes(data), pointer
        # Pillow도 따라가지 않으므로 열고 RGB로 바꿔도 싸다(따라갔다면 SHORT 40만 개를 풀어 파이썬 메모리가
        # 수십 MiB가 된다)
        _assert_cheap(lambda data=data: Image.open(io.BytesIO(data)).convert("RGB"), seconds=5.0, megabytes=8)
    # 개수 2짜리 LONG 가리킴의 두 값이 모두 진짜 하위 IFD를 가리킨다: 검사는 거절하고, Pillow는 첫 값을 따라가
    # 그 IFD를 푼다(작은 하위 IFD로 Pillow 쪽을 확인한다)
    two_values = (34665, 4, 2, ("IFD", 1, "L", 2))
    assert not _passes(_tiff([_gray_entries(extra=[two_values]), sub]))
    monkeypatch.setattr(pre, "MAX_TIFF_VALUES", 30)
    small = [(0x9999, 3, 40, _repeat("<", 3, 7, 40))]
    data = _tiff([_gray_entries(extra=[two_values]), small])
    assert not _passes(data)
    image = Image.open(io.BytesIO(data))
    image.load()
    assert len(image.getexif().get_ifd(34665)[0x9999]) == 40
    # 음수 오프셋(SLONG -1)은 Pillow의 seek가 ValueError를 내서 디코딩이 실패한다. 따라갈 곳이 없으니 이
    # 검사는 건드리지 않는다.
    negative = _tiff([_gray_entries(extra=[(34665, 9, 1, struct.pack("<l", -1))]), sub])
    assert _passes(negative)
    with pytest.raises(ValueError, match="negative seek"):
        Image.open(io.BytesIO(negative)).convert("RGB")


# 가리킴 값의 개수가 2 이상이어도 정수 형식이면 첫 값을 따라간다(형식 여섯 가지 × 가리킴 태그 셋). 독립 검토가
# Exif·GPS 가리킴에서 형식마다 '검사는 통과, Pillow는 따라감'을 재현한 조합이다. Interop 가리킴은 Pillow가
# Exif IFD 안에서만 읽지만 검사는 어느 IFD에 있든 따라간다(더 넓게 센다).
@pytest.mark.parametrize("tag", [34665, 34853, 40965])
@pytest.mark.parametrize(("kind", "width"), [(3, "H"), (4, "L"), (13, "L"), (9, "l"), (8, "h"), (16, "Q")])
def test_a_pointer_with_several_values_is_followed_for_every_integer_type(monkeypatch, tag, kind, width):
    monkeypatch.setattr(pre, "MAX_TIFF_VALUES", 30)
    main = _gray_entries(extra=[(tag, kind, 3, ("IFD", 1, width, 3))])
    used = _counted(main)

    def build(n: int) -> bytes:
        return _tiff([main, [(0x9999, 5, n, _repeat("<", 5, (1, 3), n))]])

    assert _passes(build(30 - used)) and not _passes(build(31 - used))


# Pillow는 Interop 가리킴을 첫 IFD가 아니라 Exif IFD 안에서 읽는다(Image.Exif.get_ifd). 첫 IFD에는 40965가
# '있기만' 하면 된다. 그래서 첫 IFD의 40965는 아무 데나 가리키게 하고 Exif IFD 안의 40965가 큰 IFD를 가리키게
# 하면, 첫 IFD의 가리킴만 보던 고치기 전의 검사는 통과시켰다. 이 검사는 센 모든 IFD에서 가리킴을 모은다.
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_interop_pointer_inside_the_exif_ifd_is_followed(monkeypatch):
    decoy = [(0x9998, 3, 1, struct.pack("<H", 1))]

    def build(n: int, count: int = 1) -> bytes:
        main = _gray_entries(extra=[(34665, 4, 1, ("IFD", 1)), (40965, 4, 1, ("IFD", 3))])
        exif = [(40965, 4, count, ("IFD", 2, "L", count))]
        return _tiff([main, exif, [(0x9999, 3, n, _repeat("<", 3, 7, n))], decoy])

    monkeypatch.setattr(pre, "MAX_TIFF_VALUES", 60)
    used = _counted(_gray_entries(extra=[(34665, 4, 1, ("IFD", 1)), (40965, 4, 1, ("IFD", 3))])) + 1 + 1
    assert _passes(build(60 - used)) and not _passes(build(61 - used))
    assert not _passes(build(61 - used, count=2))  # Exif IFD 안의 가리킴도 개수와 무관하게 첫 값
    # Pillow도 Exif IFD 안의 가리킴을 따라간다
    image = Image.open(io.BytesIO(build(20)))
    image.load()
    assert len(image.getexif().get_ifd(40965)[0x9999]) == 20


# 같은 가리킴 태그가 여럿이면 Pillow는 처음으로 '쓸 수 있는' 항목을 쓴다. 앞 항목이 지원하지 않는 형식이거나
# 태그 자료 예산(파일 크기)을 넘어 Pillow가 버리면 뒤 항목이 진짜가 된다. 검사는 어느 쪽인지 가리지 않고
# 정수로 읽히는 가리킴을 모두 따라간다(처음 것만 따라가면 앞에 미끼를 두어 피할 수 있다).
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_every_candidate_pointer_is_followed(monkeypatch):
    monkeypatch.setattr(pre, "MAX_TIFF_VALUES", 30)
    small = [(0x9999, 3, 40, _repeat("<", 3, 7, 40))]
    decoy = [(0x9998, 3, 1, struct.pack("<H", 1))]
    # (가) 앞 항목은 Pillow가 버리는 형식(99), 뒤 항목이 진짜
    data = _tiff([_gray_entries(extra=[(34665, 99, 1, ("IFD", 2)), (34665, 4, 1, ("IFD", 1))]), small, decoy])
    assert not _passes(data)
    # (나) 앞 항목(개수 1인 LONG8, 값 8바이트라 칸 밖)은 미끼 IFD를 가리키지만 그 자료가 태그 자료 예산을 넘어
    # Pillow가 버린다: 앞의 XMP 항목이 파일 길이에 가까운 자료를 이미 썼다. 뒤 항목(LONG, 값 칸 안)이 진짜다.
    # 태그마다 처음 것만 따라가던 고치기 전의 검사는 미끼만 보고 통과시켰다.
    entries = [(700, 7, 16, bytes(16)), (34665, 16, 1, ("IFD", 2, "Q")), (34665, 4, 1, ("IFD", 1))]
    data = bytearray(_tiff([_gray_entries(extra=entries), small, decoy]))
    (first,) = struct.unpack_from("<L", data, 4)
    (entry_count,) = struct.unpack_from("<H", data, first)
    for i in range(entry_count):
        at = first + 2 + 12 * i
        if struct.unpack_from("<H", data, at)[0] == 700:
            # XMP 자료 = 파일 앞 len - 4바이트(다른 자료와 겹쳐도 된다)
            struct.pack_into("<LL", data, at + 4, len(data) - 4, 0)
    data = bytes(data)
    assert not _passes(data)
    image = Image.open(io.BytesIO(data))  # Pillow는 뒤 항목을 따라가 SHORT 40개를 푼다
    image.load()
    assert len(image.getexif().get_ifd(34665)[0x9999]) == 40


# 한 IFD를 Exif·GPS·Interop 가리킴이 모두 가리키면 Pillow는 그 IFD를 무리마다 한 번씩, 세 번 푼다(get_ifd가
# 무리마다 따로 읽는다). 그래서 (가리킴 태그, 위치) 쌍마다 세고 같은 쌍은 한 번만 센다. 가리킴이 첫 IFD 자신을
# 가리키면 Pillow는 첫 IFD를 하위 IFD로서 다시 풀며 스트립 표도 숫자 튜플로 푼다: 그때는 표도 숫자 합계에
# 든다.
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_directory_reached_through_several_pointers_is_counted_each_time(monkeypatch):
    monkeypatch.setattr(pre, "MAX_TIFF_VALUES", 100)
    pointers = [(34665, 4, 1, ("IFD", 1)), (34853, 4, 1, ("IFD", 1)), (40965, 4, 1, ("IFD", 1))]
    used = _counted(_gray_entries(extra=pointers))

    def shared(n: int) -> bytes:  # 세 가리킴이 모두 SHORT n개(+ Interop 가리킴 하나)인 IFD를 가리킨다
        return _tiff([_gray_entries(extra=pointers), [(0x9999, 3, n, _repeat("<", 3, 7, n)), pointers[2]]])

    room = (100 - used) // 3 - 1  # 세 번 센 합계가 정확히 상한이 되는 n
    assert used + 3 * (room + 1) == 100
    assert _passes(shared(room)) and not _passes(shared(room + 1))
    image = Image.open(io.BytesIO(shared(10)))
    image.load()
    exif = image.getexif()
    assert all(len(exif.get_ifd(tag)[0x9999]) == 10 for tag in (34665, 34853, 40965))
    # 첫 IFD 자신을 가리키면 표(StripOffsets·StripByteCounts 각 30개)가 숫자 합계에 든다
    monkeypatch.setattr(pre, "MAX_TIFF_VALUES", 60)
    itself = _gray_entries(strips=30, extra=[(34665, 4, 1, ("IFD", 0))])
    assert _passes(_tiff([_gray_entries(strips=30)]))
    assert not _passes(_tiff([itself]))


# 따라가는 IFD 수에도 상한이 있다(첫 IFD 포함). 상한을 넘으면 멈추지 않고 거절한다: 멈추면 미끼 가리킴
# 여럿으로 상한을 채우고 진짜 가리킴을 그 뒤에 둘 수 있기 때문이다. 정상 파일은 첫 IFD·Exif·GPS·Interop
# 넷이다.
def test_the_directory_limit_is_inclusive(monkeypatch):
    def build(n: int) -> bytes:  # 첫 IFD + 서로 다른 하위 IFD n - 1개
        pointers = [(34665, 4, 1, ("IFD", k)) for k in range(1, n)]
        subs = [[(0x9998, 3, 1, struct.pack("<H", k))] for k in range(1, n)]
        return _tiff([_gray_entries(extra=pointers), *subs])

    _inclusive(monkeypatch, "MAX_TIFF_IFDS", 5, build)


def test_a_second_page_is_not_scanned_because_pillow_opens_only_the_first():
    pages = [_gray_entries(), _gray_entries(strips=2_000_000)]  # 둘째 쪽이 스트립 200만 개
    data = _tiff(pages, chain=True)
    assert _passes(data)
    # Pillow는 첫 쪽만 열고 둘째 쪽은 건드리지 않는다: 싸다
    _assert_cheap(lambda: Image.open(io.BytesIO(data)).convert("RGB"), seconds=5.0, megabytes=8)
    # 사슬이 길어도(IFD 10만 개) 검사는 사슬을 따라가지 않는다
    assert _passes(_tiff([_gray_entries()] + [[(256, 4, 1, struct.pack("<L", 1))]] * 100_000, chain=True))


def test_the_jpeg_segment_limit_counts_the_scan_marker_and_is_inclusive(monkeypatch):
    # 마커 세그먼트 수 = 끼운 것 + 첫 SOS(SOI는 머리 세 바이트라 세지 않는다)
    def build(n: int) -> bytes:
        return b"\xff\xd8" + _jpeg_segment(0xE5) * (n - 1) + _jpeg_segment(0xDA, b"\x00")

    _inclusive(monkeypatch, "MAX_JPEG_SEGMENTS", 10, build)


def test_the_jpeg_header_limit_is_the_end_of_the_first_scan_header(monkeypatch):
    # 머리 길이 = SOI 2 + APP1(4 + n) + SOS(4 + 1) = n + 11
    monkeypatch.setattr(pre, "MAX_JPEG_HEADER_BYTES", 100)

    def build(n: int) -> bytes:
        return b"\xff\xd8" + _jpeg_segment(0xE1, b"x" * n) + _jpeg_segment(0xDA, b"\x00")

    assert _passes(build(89)) and not _passes(build(90))
    # 첫 SOS가 아예 없어도 머리가 상한을 넘으면 거절한다
    assert not _passes(b"\xff\xd8" + _jpeg_segment(0xE1, b"x" * 200) + _jpeg_segment(0xE1, b"x"))


def test_the_jpeg_stray_byte_limit_counts_junk_and_fill_bytes(monkeypatch):
    monkeypatch.setattr(pre, "MAX_JPEG_STRAY_BYTES", 5)
    scan = _jpeg_segment(0xDA, b"\x00")

    def junk(n: int) -> bytes:  # 0xFF00(이스케이프) 다음에 마커가 아닌 바이트 n개
        return b"\xff\xd8\xff\x00" + b"A" * n + scan

    def fill(n: int) -> bytes:  # 마커 접두 0xFF 뒤에 0xFF가 n개 더 이어진 채움 바이트
        return b"\xff\xd8\xff" + b"\xff" * n + scan[1:]

    assert _passes(junk(5)) and not _passes(junk(6))
    assert _passes(fill(5)) and not _passes(fill(6))
    # 쓰레기만 이어지다 파일이 끝나도 상한을 넘으면 거절하고, 상한 안이면 파일 끝까지 쓰레기여도 통과시킨다
    # (그 뒤는 Pillow가 판단한다)
    only_junk = b"\xff\xd8\xff\x00" + b"A" * 1000
    assert not _passes(only_junk)
    monkeypatch.setattr(pre, "MAX_JPEG_STRAY_BYTES", 2000)
    assert _passes(only_junk)
    # 파일 끝까지 쓰레기인 경우의 경계도 정확하다: 상한 개수까지는 통과, 하나 더는 거절
    monkeypatch.setattr(pre, "MAX_JPEG_STRAY_BYTES", 5)
    assert _passes(b"\xff\xd8\xff\x00" + b"A" * 5) and not _passes(b"\xff\xd8\xff\x00" + b"A" * 6)


# 스캔(SOS) 수의 상한(첫 SOS 포함). libjpeg는 점진 JPEG의 스캔마다 그림 전체의 MCU를 돌므로(자료가 모자라면
# 0으로 채운다) 비용이 스캔 수 × 화소 수에 비례한다. 첫 SOS까지만 보던 고치기 전의 검사는 그 뒤를 보지
# 않았다(이 문제는 사전 검사를 넣기 전부터 있었다).
def test_the_jpeg_scan_limit_is_inclusive(monkeypatch):
    def build(n: int) -> bytes:  # SOS n개(첫 SOS 포함) 뒤 EOI
        return b"\xff\xd8" + _jpeg_segment(0xDA, b"\x00") * n + b"\xff\xd9"

    _inclusive(monkeypatch, "MAX_JPEG_SCANS", 5, build)
    # 실제 점진 JPEG(YCbCr, 스캔 10개)에 짧은 스캔을 끼워도 같다
    monkeypatch.setattr(pre, "MAX_JPEG_SCANS", 12)
    base = _progressive_jpeg(64)
    assert _passes(_with_extra_scans(base, 2)) and not _passes(_with_extra_scans(base, 3))


# 첫 SOS 뒤의 스캔은 libjpeg가 마커를 읽는 규칙(jdmarker.c의 next_marker·read_markers, jdhuff.c의
# jpeg_fill_bit_buffer)으로 센다: 엔트로피 자료 안의 FF는 00이 덧붙어(byte stuffing) 마커가 아니고, 길이 칸이
# 있는 세그먼트의 자료는 길이만큼 건너뛰며, RST·TEM은 길이 없는 마커이고, 채움 바이트(FF FF …)가 앞에 붙어도
# 마커이며, EOI에서 멈춘다(EOI 뒤에 붙은 자료는 libjpeg가 읽지 않는다).
def test_jpeg_scans_are_counted_by_the_marker_rules_of_libjpeg(monkeypatch):
    monkeypatch.setattr(pre, "MAX_JPEG_SCANS", 3)
    sos = _jpeg_segment(0xDA, b"\x00")
    head = b"\xff\xd8" + sos  # 첫 SOS
    assert _passes(head + b"\x12\xff\x00\xda" * 50 + b"\xff\xd9")  # FF 00 DA는 자료다
    segments = _jpeg_segment(0xFE, b"\xff\xda" * 100) + _jpeg_segment(0xE1, b"\xff\xda" * 10)
    assert _passes(head + segments + b"\xff\xd9")  # COM·APP1 안의 FF DA는 세그먼트 자료다
    assert _passes(head + b"\xff\xd9" + sos * 100)  # EOI 뒤
    more = head + b"\xff\xd0" + sos + b"\xff\x01" + b"\xff\xff\xff" + sos[1:]  # RST0, SOS, TEM, 채움 + SOS
    assert _passes(more + b"\xff\xd9") and not _passes(more + sos + b"\xff\xd9")


# 첫 SOS 뒤의 세그먼트도 검사의 반복 수를 묶으려고 MAX_JPEG_SEGMENTS로 센다(정상 점진 JPEG는 스캔과 DHT를 합쳐
# 수십 개). 상한을 넘으면 거절한다.
def test_jpeg_segments_after_the_first_scan_are_limited(monkeypatch):
    def build(n: int) -> bytes:  # 첫 SOS 뒤에 COM 세그먼트 n개
        return b"\xff\xd8" + _jpeg_segment(0xDA, b"\x00") + _jpeg_segment(0xFE, b"c") * n + b"\xff\xd9"

    _inclusive(monkeypatch, "MAX_JPEG_SEGMENTS", 10, build)


def _bare_png(before: int = 0, after: int = 0, data: int = 1) -> bytes:
    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"abCD") * before
        + _png_chunk(b"IDAT", b"\x00") * data
        + _png_chunk(b"abCD") * after
        + _png_chunk(b"IEND")
    )


def test_the_png_limits_are_inclusive_and_separate(monkeypatch):
    # IHDR도 첫 IDAT 앞의 청크다: 앞 한도 5이면 사설 청크는 4개까지
    _inclusive(monkeypatch, "MAX_PNG_CHUNKS_BEFORE_DATA", 5, lambda n: _bare_png(before=n - 1))
    # 뒤쪽(IDAT 다음) 청크는 앞 한도에 들지 않는다. IDAT·fdAT 밖의 청크 합계(IHDR 포함, IEND 제외) 한도 8
    monkeypatch.setattr(pre, "MAX_PNG_CHUNKS_BEFORE_DATA", 1000)
    _inclusive(monkeypatch, "MAX_PNG_OTHER_CHUNKS", 8, lambda n: _bare_png(before=2, after=n - 3))
    # 이미지 데이터 조각(IDAT)은 따로 센다: 사설 청크 한도와 무관하다
    monkeypatch.setattr(pre, "MAX_PNG_OTHER_CHUNKS", 3)
    _inclusive(monkeypatch, "MAX_PNG_DATA_CHUNKS", 7, lambda n: _bare_png(data=n))
    # APNG의 프레임 데이터 조각(fdAT)과 옛 DDAT도 이미지 데이터 조각이다
    for kind in (b"fdAT", b"DDAT"):
        body = _bare_png(data=0).replace(
            _png_chunk(b"IEND"), _png_chunk(kind, b"\x00" * 4) * 8 + _png_chunk(b"IEND")
        )
        monkeypatch.setattr(pre, "MAX_PNG_DATA_CHUNKS", 8)
        assert _passes(body), kind
        monkeypatch.setattr(pre, "MAX_PNG_DATA_CHUNKS", 7)
        assert not _passes(body), kind


# 압축을 푸는 청크(iCCP·zTXt·iTXt)는 위치(첫 IDAT 앞뒤)와 종류를 가리지 않고 합쳐 센다. Pillow는 이 청크를
# 하나에 1MiB까지 풀고(청크마다 따로), 텍스트 메모리 합계(64MiB)는 iCCP와 키가 빈 zTXt, UTF-8이 아닌 iTXt에
# 걸리지 않는다. 고치기 전에는 청크 수 상한(전체 1만 개) 안에서 iCCP 9,999개(9.6MiB)가 CPU 25초를 썼다.
def test_png_chunks_that_inflate_are_limited_wherever_they_are(monkeypatch):
    kinds = (b"iCCP", b"zTXt", b"iTXt")

    def build(n: int) -> bytes:
        chunks = [_inflating_chunk(kinds[i % 3], 10) for i in range(n)]
        return _png_with(_bare_png(), before=b"".join(chunks[: n // 2]), after=b"".join(chunks[n // 2 :]))

    _inclusive(monkeypatch, "MAX_PNG_INFLATING_CHUNKS", 6, build)
    # 압축하지 않은 텍스트(tEXt)는 이 상한에 들지 않는다
    monkeypatch.setattr(pre, "MAX_PNG_INFLATING_CHUNKS", 6)
    assert _passes(_png_with(_bare_png(), after=_png_chunk(b"tEXt", b"k\x00v") * 50))


# RLE BMP(압축 1 = RLE8, 2 = RLE4)는 파일 크기에 상한을 둔다. Pillow의 RLE
# 디코더(BmpImagePlugin.BmpRleDecoder)는 파이썬 반복이라 명령 두 바이트마다 한 바퀴를 돌고, 줄 끝·이동 0처럼
# 화소를 늘리지 않는 명령은 파일 끝까지 돌 수 있다(파일 크기에 비례하는 CPU). 압축 칸은 Pillow처럼 정보 머리가
# BITMAPINFOHEADER 꼴(40·52·56·64·108·124바이트)일 때만 읽는다.
def test_the_rle_bmp_size_limit_is_inclusive_and_only_for_rle(monkeypatch):
    for compression in (1, 2):
        _inclusive(monkeypatch, "MAX_BMP_RLE_BYTES", 3000, lambda n, c=compression: _rle_bmp_of_length(n, c))
    monkeypatch.setattr(pre, "MAX_BMP_RLE_BYTES", 3000)
    for header in (40, 52, 56, 64, 108, 124):
        assert not _passes(_bmp(1, 1, b"\x00\x00" * 2000 + b"\x00\x01", header=header)), header
    # 무압축 BMP는 상한보다 커도 통과한다(화소 수에 비례하는 C 디코더)
    raw = _encode(np.zeros((100, 100, 3), np.uint8), "BMP")
    assert len(raw) > 3000 and _passes(raw)
    # 압축 칸이 없는 BITMAPCOREHEADER(12바이트)는 Pillow가 무압축으로 읽는다: 압축 칸 자리(30)의 값이 1이어도
    # 통과
    core = bytearray(_bmp(1, 1, b"\x01" * 5000, header=12))
    core[30:34] = struct.pack("<I", 1)
    assert _passes(bytes(core))
    # Pillow가 모르는 머리 크기(200)는 Pillow가 열 때 거절하므로 건드리지 않는다
    unknown = _bmp(1, 1, b"\x00\x00" * 2000 + b"\x00\x01", header=200)
    assert _passes(unknown)
    with pytest.raises(OSError, match="Unsupported BMP header type"):
        Image.open(io.BytesIO(unknown))


# ==================================================================================================
# 5. Pillow의 읽기 규칙과의 일치: 읽을 수 없는 구조에서는 멈추고 통과시킨다(그 뒤는 Pillow가 판단한다)
# ==================================================================================================
# 일부러 깨 놓은 TIFF에 Pillow가 내는 UserWarning(Corrupt EXIF data 등)은 시험 대상의 결과라 걸러 낸다.
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_tiff_structures_pillow_cannot_read_are_passed_on(monkeypatch):
    monkeypatch.setattr(pre, "MAX_TIFF_ENTRIES", 10)
    big_endian_bigtiff = b"MM\x00\x2b\x00\x08\x00\x00" + struct.pack(">Q", 16) + b"\x00" * 64
    # Pillow는 data[2] == 43일 때만 BigTIFF로 읽는다. 이것은 고전 TIFF로 읽혀 첫 IFD 오프셋이 0x00080000이
    # 되고 파일 밖이라 거절된다(Pillow의 일).
    assert _passes(big_endian_bigtiff)
    with pytest.raises(UnidentifiedImageError):  # Pillow는 이 파일을 열지 못한다
        Image.open(io.BytesIO(big_endian_bigtiff))
    assert _passes(b"II\x2a\x00" + struct.pack("<L", 10_000) + b"\x00" * 100)  # 첫 IFD가 파일 밖
    assert _passes(b"II\x2a\x00")  # 머리가 잘림
    assert _passes(b"II\x2b\x00\x08\x00\x00\x00")  # BigTIFF 머리가 잘림
    full = _tiff([_gray_entries()])
    for cut in range(8, len(full)):
        _passes(full[:cut])  # 어디서 잘려도 예외 없이 끝난다(거절 여부는 상관없다)
    # 개수 칸이 상한을 넘는다고 적었으면 표가 실제로는 잘려 있어도 거절한다: 정상 파일의 IFD가 적을 수 없는
    # 값이다
    assert not _passes(b"II\x2a\x00" + struct.pack("<L", 8) + struct.pack("<H", 500))
    # 표가 개수 칸보다 짧게 잘렸으면(개수는 상한 안) 읽을 수 있는 데까지 세고 통과시킨다
    assert _passes(b"II\x2a\x00" + struct.pack("<L", 8) + struct.pack("<H", 5) + b"\x00" * 20)


def test_jpeg_structures_pillow_cannot_read_are_passed_on():
    assert _passes(b"\xff\xd8\xff")  # 머리 세 바이트뿐
    assert _passes(b"\xff\xd8\xff\xe0")  # 마커 번호만 있고 길이가 없다
    assert _passes(b"\xff\xd8\xff\xe0\x00")  # 길이 칸이 한 바이트뿐이다
    assert _passes(b"\xff\xd8\xff\xe1\xff\xff" + b"x" * 10)  # 자료가 파일 밖으로 나간다
    # 알 수 없는 마커(0x01): Pillow가 "no marker found"로 오류를 내는 자리라 그 뒤를 보지 않는다
    assert _passes(b"\xff\xd8\xff\x01" + b"\xff\xe5\x00\x02" * 100_000)
    # SOS가 없다: 길이 칸이 없는 마커(EOI)는 자료를 읽지 않는다
    assert _passes(b"\xff\xd8\xff\xe0\x00\x02" * 3 + b"\xff\xd9" * 50)
    # 길이 칸 값이 2 미만이면 자료를 읽지 않는다(Pillow의 _safe_read가 빈 바이트열을 돌려준다)
    assert _passes(b"\xff\xd8\xff\xe0\x00\x00" + b"\xff\xe0\x00\x01" * 50 + b"\xff\xda\x00\x02")


def test_png_structures_pillow_stops_at_end_the_scan():
    chunk = _png_chunk(b"abCD")
    body = _bare_png(before=0)
    start = body.index(_png_chunk(b"IDAT", b"\x00"))
    # 청크 종류가 영숫자 네 글자가 아니면 Pillow는 첫 IDAT 앞에서 오류를 내고 뒤에서는 읽기를 멈춘다. 그 뒤의
    # 폭탄은 보지 않는다.
    bad = struct.pack(">I", 0) + b"ab D" + b"\x00" * 4
    assert _passes(body[:start] + bad + chunk * (pre.MAX_PNG_OTHER_CHUNKS + 5))
    with pytest.raises(UnidentifiedImageError):  # Pillow는 열자마자 오류를 낸다
        Image.open(io.BytesIO(body[:start] + bad + chunk * 3))
    # 길이가 파일을 벗어난 청크, 헤더가 8바이트에 못 미치는 끝, IEND 없는 끝에서 멈춘다
    assert _passes(body[:start] + struct.pack(">I", 2**31) + b"abCD" + chunk * (pre.MAX_PNG_OTHER_CHUNKS + 5))
    assert _passes(body[:-12] + b"\x00\x00\x00")
    assert _passes(body[:-12])
    # IEND 뒤의 바이트는 Pillow도 읽지 않는다
    assert _passes(body + chunk * (pre.MAX_PNG_OTHER_CHUNKS + 5))


# ==================================================================================================
# 6. 어떤 바이트열에도 ImageStructureError 말고는 예외를 내지 않고, 오래 걸리지 않는다
# ==================================================================================================
def _small_samples() -> list[bytes]:
    names = [
        "png",
        "png-200-text-chunks",
        "apng-30-frames",
        "jpeg",
        "jpeg-progressive",
        "jpeg-xmp-and-photoshop",
        "jpeg-100-comments",
        "tiff-raw",
        "tiff-lzw",
        "tiff-two-pages",
        "tiff-exif-and-gps-sub-ifds",
        "tiff-big-endian-small",
        "bigtiff",
        "tiff-16px-tiles",
        "jpeg-progressive-cmyk-18-scans",
        "png-icc-and-20-compressed-text-chunks",
        "bmp-rle8",
    ]
    return [_legit_files()[name] for name in names]


# 잘라 볼 위치: 앞 3,000바이트와 뒤 3,000바이트는 전부, 가운데는 7바이트마다.
def _cut_points(length: int) -> list[int]:
    return sorted(
        {*range(min(length, 3000)), *range(max(0, length - 3000), length + 1), *range(0, length, 7)}
    )


# 정상 파일을 어디서 잘라도(모든 위치), 그리고 무작위로 바이트를 바꿔도 검사는 ImageStructureError 말고는
# 예외를 내지 않는다. 읽기마다 범위를 먼저 확인하는지 보는 시험이다. 바꾼 파일은 상한을 넘는 값(예: IFD 개수
# 65,535)이 되기도 해서 거절은 나올 수 있다.
def test_truncated_and_mutated_files_never_raise_anything_else():
    rng = np.random.default_rng(7)
    began = time.process_time()
    for sample in _small_samples():
        for cut in _cut_points(len(sample)):
            _passes(sample[:cut])
        for _ in range(300):
            mutated = bytearray(sample)
            for position in rng.integers(0, len(mutated), size=int(rng.integers(1, 6))):
                mutated[int(position)] = int(rng.integers(0, 256))
            _passes(bytes(mutated))
    assert time.process_time() - began < 30


# ==================================================================================================
# 7. 서비스 통합 (FastAPI가 있을 때): 400, Image.open을 부르지 않음, 정상 파일의 화소는 Pillow로 연 것과 같다
# ==================================================================================================
@pytest.fixture
def service():
    pytest.importorskip("fastapi")
    pytest.importorskip("multipart")
    from defect_inspect import service as module

    return module


# 서비스 시험 도우미의 것과 같은 모양: 모델 입력 크기로 줄인 RGB 그림의 화소.
def _expected_pixels(data: bytes, size: int) -> np.ndarray:
    expected = Image.open(io.BytesIO(data)).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(expected)


# 서비스 안의 폭탄은 20MiB급을 그대로 쓰지 않고 상한의 3~20배로 줄인 것을 쓴다(고치기 전 코드에서도 이 시험이
# 메모리를 터뜨리지 않고 실패하게).
def _service_bombs() -> dict:
    return {
        "tiff-strips": lambda: _tiff([_gray_entries(height=400_000, strips=400_000)]),
        "tiff-strips-without-counts": lambda: _tiff(
            [_gray_entries(height=800_000, strips=800_000, with_counts=False)]
        ),
        "tiff-exif-rationals": lambda: _tiff(
            [
                _gray_entries(extra=[(34665, 4, 1, ("IFD", 1))]),
                [(0x9999, 5, 300_000, _repeat("<", 5, (1, 3), 300_000))],
            ]
        ),
        "jpeg-markers": lambda: _jpeg_with(_encode(_picture(), "JPEG"), _jpeg_segment(0xE5) * 50_000),
        "png-private-chunks": lambda: _png_with(
            _encode(_picture(), "PNG"), before=_png_chunk(b"abCD") * 50_000
        ),
        # 아래는 독립 검토가 첫 사전 검사를 우회한 다섯 갈래를 줄인 크기로 만든 것이다. 고치기 전의 검사는
        # 모두 통과시켰고, 괄호는 그때 `_decode`가 쓴 CPU다(이 PC, Pillow 13.0.0).
        # BYTE형 StripOffsets 20만 개, 0.19MiB(1.6초)와 같은 꼴의 TileOffsets
        "tiff-strip-offsets-typed-byte": lambda: _tiff(
            [
                sorted(
                    [e for e in _gray_entries(height=2, with_counts=False) if e[0] != 273]
                    + [(273, 1, 200_000, bytes(200_000))],
                    key=lambda e: e[0],
                )
            ]
        ),
        "tiff-tile-offsets-typed-byte": lambda: _tiff(
            [
                sorted(
                    [e for e in _gray_entries(with_counts=False) if e[0] not in (273, 278)]
                    + [(322, 3, 1, struct.pack("<H", 16)), (323, 3, 1, struct.pack("<H", 16))]
                    + [(324, 1, 200_000, bytes(200_000))],
                    key=lambda e: e[0],
                )
            ]
        ),
        # 값이 둘인 Exif 가리킴이 RATIONAL 30만 개의 하위 IFD를 가리킨다, 2.3MiB(0.9초)
        "tiff-exif-pointer-with-two-values": lambda: _tiff(
            [
                _gray_entries(extra=[(34665, 4, 2, ("IFD", 1, "L", 2))]),
                [(0x9999, 5, 300_000, _repeat("<", 5, (1, 3), 300_000))],
            ]
        ),
        # Exif IFD 안의 Interop 가리킴(첫 IFD의 40965는 미끼)이 RATIONAL 30만 개를 가리킨다
        "tiff-interop-pointer-in-the-exif-ifd": lambda: _tiff(
            [
                _gray_entries(extra=[(34665, 4, 1, ("IFD", 1)), (40965, 4, 1, ("IFD", 3))]),
                [(40965, 4, 1, ("IFD", 2))],
                [(0x9999, 5, 300_000, _repeat("<", 5, (1, 3), 300_000))],
                [(0x9998, 3, 1, struct.pack("<H", 1))],
            ]
        ),
        # IDAT 뒤 iCCP 1,000개와 키가 빈 zTXt 1,000개, 각 0.96MiB(2.4초, 3.0초)
        "png-iccp-chunks-after-idat": lambda: _png_with(
            _encode(_picture(), "PNG"), after=_inflating_chunk(b"iCCP", 1_000_000) * 1_000
        ),
        "png-empty-key-ztxt-chunks": lambda: _png_with(
            _encode(_picture(), "PNG"), after=_inflating_chunk(b"zTXt", 1_000_000, key=b"") * 1_000
        ),
        # 5000×5000 점진 JPEG + 짧은 스캔 2,000개, 0.17MiB(2.8초). 사전 검사를 넣기 전부터 있던 경로다
        "jpeg-progressive-with-2000-scans": lambda: _with_extra_scans(_progressive_jpeg(5000), 2_000),
        # 줄 끝 명령만 8MiB인 1×1 RLE8 BMP(1.7초, 끝에 읽을 수 없는 이미지로 400이 되지만 그 전에 CPU를 쓴다)
        "bmp-rle8-end-of-line-commands": lambda: _rle_bmp_of_length(8 * 2**20),
    }


@pytest.mark.parametrize("name", sorted(_service_bombs()))
def test_decode_refuses_a_structure_bomb_before_pillow_opens_it(service, monkeypatch, name):
    from fastapi import HTTPException

    data = _service_bombs()[name]()
    opened = []
    real_open = service.Image.open
    monkeypatch.setattr(
        service.Image, "open", lambda *args, **kwargs: opened.append(1) or real_open(*args, **kwargs)
    )

    def decode():
        with pytest.raises(HTTPException) as refused:
            service._decode(data, 64)
        assert refused.value.status_code == 400 and refused.value.detail == ERROR_DETAIL

    _assert_cheap(decode, seconds=2.0, megabytes=16)
    assert opened == []  # Pillow가 열지 않았다: 구조 항목마다 객체를 만들 기회가 없었다


# 점검에서 20MiB 안팎으로 재현한 네 가지 모양을 실제 크기 그대로 `_decode`에 넣는다(위 시험의 줄인 입력과 같은
# 종류). 고치기 전에는 각각 22초·+837MiB, 44초·+1.7GiB(외삽), 7.9초·+676MiB, 5.4초·+207MiB를 쓰고 400 없이
# 끝났다.
@pytest.mark.parametrize(
    "name",
    [
        "tiff-2.6M-strips",
        "tiff-5.2M-strips-without-counts",
        "jpeg-5.2M-markers",
        "png-1.7M-private-chunks-before-idat",
    ],
)
def test_decode_refuses_the_full_size_structure_bombs_quickly(service, name):
    from fastapi import HTTPException

    data = _bombs()[name]()
    assert len(data) < service.MAX_UPLOAD_BYTES  # 서비스가 받는 파일 크기 안이다

    def decode():
        with pytest.raises(HTTPException) as refused:
            service._decode(data, 64)
        assert refused.value.status_code == 400 and refused.value.detail == ERROR_DETAIL

    _assert_cheap(decode, seconds=2.0, megabytes=16)


def test_a_structure_bomb_is_a_400_for_inspect_and_calibrate(service):
    from fastapi.testclient import TestClient

    client = TestClient(service.create_offline_app(admin_token=TOKEN), base_url="http://127.0.0.1")
    bomb = _service_bombs()["tiff-strips"]()
    response = client.post(
        "/inspect", files={"image": ("x.tif", bomb, "image/tiff")}, data={"category": "demo"}
    )
    assert response.status_code == 400 and response.json() == {"detail": ERROR_DETAIL}
    assert response.headers["x-content-type-options"] == "nosniff"
    ok = _encode(np.full((64, 64, 3), 128, np.uint8), "PNG")
    files = [("images", (f"{i}.png", ok, "image/png")) for i in range(19)] + [
        ("images", ("bomb.tif", bomb, "image/tiff"))
    ]
    response = client.post(
        "/calibrate", files=files, data={"category": "demo"}, headers={"X-Admin-Token": TOKEN}
    )
    assert response.status_code == 400 and response.json() == {"detail": ERROR_DETAIL}


@pytest.mark.parametrize("name", sorted(_legit_files()))
def test_decode_gives_the_same_pixels_as_pillow_for_ordinary_files(service, name):
    data = _legit_files()[name]
    decoded = service._decode(data, 32)
    assert decoded.mode == "RGB" and decoded.size == (32, 32)
    assert np.array_equal(np.asarray(decoded), _expected_pixels(data, 32)), name
    full = service._decode(data)  # 리사이즈 없이도 같은 그림이다
    assert full.mode == "RGB" and np.array_equal(
        np.asarray(full), np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    )


# 잘렸거나 깨진 구조는 400이다. 사전 검사가 멈추고 Pillow가 거절하든 검사가 거절하든 서비스의 답은 같은
# 400 하나이고, 어느 쪽에서도 500이나 다른 예외가 나오지 않는다. 형식마다 머리·표·마커·청크가 잘린 것과
# 오프셋·길이·종류가 틀린 것을 본다.
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cut_and_broken_structures_are_a_400(service):
    from fastapi import HTTPException

    files = _legit_files()
    tiff = bytearray(_tiff([_gray_entries()]))
    stale_offset = bytearray(tiff)
    stale_offset[4:8] = struct.pack("<L", 10_000)  # 첫 IFD가 파일 밖을 가리킨다
    png = files["png-200-text-chunks"]
    cases = {
        "tiff cut inside the header": bytes(tiff[:6]),
        "tiff cut before the directory": bytes(tiff[:9]),
        "tiff cut inside the entry table": bytes(tiff[: 9 + 2 + 12 * 3]),
        "tiff directory outside the file": bytes(stale_offset),
        "tiff big-endian BigTIFF header": b"MM\x00\x2b\x00\x08\x00\x00"
        + struct.pack(">Q", 16)
        + b"\x00" * 64,
        "jpeg cut inside the first segment": files["jpeg"][:12],
        "jpeg segment longer than the file": b"\xff\xd8\xff\xe1\xff\xff" + b"x" * 10,
        "jpeg unknown marker": b"\xff\xd8\xff\x01" + b"\x00" * 20,
        "png cut inside the header chunk": png[:20],
        "png cut inside a text chunk": png[:200],
        "png chunk type that is not a name": png[:33]
        + struct.pack(">I", 0)
        + b"ab D"
        + b"\x00" * 4
        + png[33:],
        "png chunk longer than the file": png[:33] + struct.pack(">I", 2**31) + b"abCD" + png[33:],
    }
    for name, data in cases.items():
        with pytest.raises(HTTPException) as refused:
            service._decode(data, 64)
        assert refused.value.status_code == 400 and refused.value.detail == ERROR_DETAIL, name


# 잘린 파일의 결과(그림이거나 400)가 Pillow를 직접 쓸 때와 같다: 이 검사가 잘린 파일을 새로 거절하지 않는다.
# 잘린 TIFF에 Pillow가 내는 UserWarning(Corrupt EXIF data 등)은 시험 대상의 결과라 걸러 낸다.
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_truncated_files_end_the_same_way_as_with_pillow_alone(service):
    from fastapi import HTTPException

    def pillow_alone(data: bytes) -> bool:
        try:
            Image.open(io.BytesIO(data), formats=service.IMAGE_FORMATS).convert("RGB")
        except Exception:  # 어떤 디코더 오류든 서비스는 400으로 답한다
            return False
        return True

    for name in (
        "png",
        "png-200-text-chunks",
        "jpeg",
        "jpeg-icc-1.5MB-in-23-segments",
        "tiff-raw",
        "tiff-lzw",
        "tiff-exif-and-gps-sub-ifds",
        "tiff-3000-strips",
        "bigtiff",
        "apng-30-frames",
        "jpeg-progressive-cmyk-18-scans",
        "bmp-rle8",
    ):
        data = _legit_files()[name]
        for cut in sorted({*range(0, min(len(data), 400), 7), *range(0, len(data), max(1, len(data) // 25))}):
            piece = data[:cut]
            try:
                service._decode(piece, 32)
                served = True
            except HTTPException as error:
                assert error.status_code == 400 and error.detail in (
                    ERROR_DETAIL,
                    "image has too many pixels",
                )
                served = False
            assert served == pillow_alone(piece), (name, cut)
