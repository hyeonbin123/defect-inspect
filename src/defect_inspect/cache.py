"""Resized image and mask cache: one sequential pass over the tar into a few large .npy files."""

# [파일 역할] VisA tar를 처음부터 끝까지 한 번 순서대로 읽어, 입력 크기 S로 줄인 이미지
# 배열(`visa_<S>_images.npy`)과 256×256 결함 마스크 배열(`visa_masks_256.npy`), 각 배열의 행 순서
# 파일(index.csv)을 만드는 0단계 리사이즈 캐시 모듈이다. 모든 실행(PatchCore, Dinomaly, 지도 학습, 교란, 지연
# 측정)은 tar 대신 이 캐시에서 이미지를 읽는다.
# [왜 필요한가] tar를 작은 파일 만 장으로 풀어 놓고 매번 읽으면 느리다. 큰 배열 파일 몇
# 개로 만들어 두면 메모리 맵으로 필요한 행만 빠르게 읽는다. 0단계 실측: 256px 2.1GB, 392px 5.0GB, 448px 6.5GB,
# 마스크 0.7GB이고 448px 캐시를 만드는 데 178초였다(docs/experiments.md 0단계).
# [전처리 규칙] 이미지는 가로세로 비율을 무시하고 PIL BICUBIC으로 S×S(축소 시 안티앨리어싱 포함), 마스크는
# 0보다 큰 값을 결함으로 보고 BOX 필터로 256×256으로 줄인 뒤 0보다 크면 결함이다(공통 규칙 "전처리").
# [봉인 테스트] 캐시는 테스트 이미지까지 배열로 옮겨야 해서 `--allow-test` 없이 돌지만, 만들 때마다 장부에
# `stage: cache`로 남긴다(규칙 변경 기록 1). 캐시에서 봉인 테스트를 막는 것은 splits.select의 몫이다.
# [연결] run_patchcore, run_grid, run_perturb, run_dinomaly, run_export_eval, export, bench, dinomaly_serving,
# figures가 ImageCache로 읽는다. M2AD 캐시는 m2ad.py에 따로 있다.

# tarfile은 tar를 압축 풀지 않고 순서대로 읽는 데, ThreadPoolExecutor와 deque는 디코딩을 여러 스레드로 돌리며
# 쓰기 순서를 관리하는 데, re는 다른 크기의 캐시 index 파일 이름을 찾는 데 쓴다.
import argparse
import io
import os
import re
import tarfile
import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# numpy는 .npy 배열 파일 생성과 메모리 맵 읽기에, PIL은 JPEG·PNG 디코딩과 리사이즈에 쓴다. ManifestRow는
# splits.py의 매니페스트 한 행(이미지 경로, 마스크 경로, 역할)이다.
import numpy as np
from PIL import Image

from defect_inspect.splits import ManifestRow

# MASK_SIZE: 픽셀 지표를 모든 구성에서 같은 256×256 격자로 재므로 마스크는 입력 크기와 상관없이 256으로 한
# 벌만 만든다. _PART: 만드는 중인 파일에 붙이는 접미사. 다 만든 뒤에야 최종 이름으로 바꾼다.
MASK_SIZE = 256
_PART = ".part"


# 캐시 파일 이름 규칙. 이미지 배열과 그 행 순서(index)는 크기마다 하나씩, 마스크 배열과 그 index는 모든 크기가
# 함께 쓴다.
def images_path(out_dir: Path, size: int) -> Path:
    return Path(out_dir) / f"visa_{size}_images.npy"


def index_path(out_dir: Path, size: int) -> Path:
    return Path(out_dir) / f"visa_{size}_index.csv"


def masks_path(out_dir: Path) -> Path:
    return Path(out_dir) / f"visa_masks_{MASK_SIZE}.npy"


def masks_index_path(out_dir: Path) -> Path:
    # Row order of the mask array, kept next to it so that other sizes can check it before reusing it.
    return Path(out_dir) / f"visa_masks_{MASK_SIZE}_index.csv"


# 이미지 한 장을 RGB uint8 [S, S, 3]으로 줄인다. 원본(약 1,300~1,560 × 960~1,180px)을 정사각형으로 늘이거나
# 줄여 비율은 맞추지 않는다. BICUBIC은 축소할 때 PIL이 안티앨리어싱을 적용한다.
def resize_image(img: Image.Image, size: int) -> np.ndarray:
    """RGB uint8 [size, size, 3]; aspect ratio ignored, bicubic (antialiased when shrinking)."""
    return np.asarray(img.convert("RGB").resize((size, size), Image.Resampling.BICUBIC), dtype=np.uint8)


# 마스크 한 장을 {0, 1} [256, 256]으로 줄인다. 먼저 0보다 큰 값을 255로 이진화하고, BOX 필터(상자 안 평균)로
# 줄인 뒤 0보다 크면 1로 둔다. 상자 안에 결함 픽셀이 하나라도 있으면 결함이 되므로, 몇 픽셀짜리 작은
# 결함(마카로니의 잔금 등)이 축소로 사라지지 않는다. 채널이 둘 이상인 마스크는 형식이 다른 것이라 거절한다.
def resize_mask(mask: Image.Image, size: int = MASK_SIZE) -> np.ndarray:
    """uint8 {0, 1} [size, size]: 1 where any source pixel of the box is a defect (value > 0)."""
    source = np.asarray(mask)
    if source.ndim != 2:
        raise ValueError(f"expected a single-channel mask, got mode {mask.mode}")
    binary = Image.fromarray(np.where(source > 0, 255, 0).astype(np.uint8))
    small = binary.resize((size, size), Image.Resampling.BOX)
    return (np.asarray(small) > 0).astype(np.uint8)


# tar에서 꺼낸 바이트를 디코딩해 줄이는 작업 단위. 스레드 풀에서 돈다(PIL의 디코딩·리사이즈는 GIL을 풀어
# 스레드로도 병렬이 된다).
def _decode_image(data: bytes, size: int) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as img:
        return resize_image(img, size)


def _decode_mask(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as mask:
        return resize_mask(mask)


# index 파일: 배열의 i번째 행이 어느 이미지인지를 한 줄에 하나씩 적은 텍스트. 매니페스트의 이미지 경로 순서와
# 같아야 그 캐시를 믿고 쓴다.
def _read_index(path: Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()


def _write_index(path: Path, images: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(f"{image}\n" for image in images))


# 배열 전체를 읽지 않고 메모리 맵으로 모양만 확인한다. 바로 del로 맵을 풀어 Windows에서 파일이 잡혀 있지 않게
# 한다.
def _npy_shape(path: Path) -> tuple[int, ...]:
    array = np.load(path, mmap_mode="r")
    shape = array.shape
    del array
    return shape


# 같은 행 순서로 만든 완성된 마스크 배열이 이미 있는지 본다. 있으면 새 크기의 캐시를 만들 때 마스크를 다시
# 만들지 않고 재사용한다.
def _masks_match(out_dir: Path, index: list[str]) -> bool:
    """True when a finished mask array with exactly this row order is already in `out_dir`."""
    if not (masks_path(out_dir).exists() and masks_index_path(out_dir).exists()):
        return False
    if _read_index(masks_index_path(out_dir)) != index:
        return False
    return _npy_shape(masks_path(out_dir)) == (len(index), MASK_SIZE, MASK_SIZE)


# 주어진 크기의 캐시가 다 만들어져 있고 매니페스트 순서를 따르는지 본다. main은 이것이 참이면(그리고 --force가
# 아니면) tar를 다시 읽지 않는다.
def cache_is_complete(out_dir: Path, size: int, manifest: list[ManifestRow]) -> bool:
    """True when images, masks and both indexes exist for `size` and follow the manifest order."""
    index = [r.image for r in manifest]
    if not (images_path(out_dir, size).exists() and index_path(out_dir, size).exists()):
        return False
    if _read_index(index_path(out_dir, size)) != index or not _masks_match(out_dir, index):
        return False
    return _npy_shape(images_path(out_dir, size)) == (len(index), size, size, 3)


# 큰 .npy 파일을 한 행씩 채우는 도우미. open_memmap으로 헤더와 0으로 채운 파일을 만든 뒤 맵은 바로 풀고, 일반
# 파일 핸들로 행 위치(offset + row × 행 바이트)에 찾아가 쓴다. 메모리 맵을 계속 열어 두면 Windows에서 이름
# 바꾸기·지우기가 막히므로, 닫는 시점을 확실히 정할 수 있는 파일 핸들을 쓴다(docstring). 행을 순서와 상관없이
# 쓸 수 있어 tar 순서대로 오는 이미지를 매니페스트 순서 자리에 바로 넣는다.
class _RowWriter:
    """A .npy file created with `open_memmap`, then filled row by row through a plain file handle.

    The handle (unlike a live memory map) can be closed at a known point, which Windows needs before
    the file is renamed or removed.
    """

    def __init__(self, path: Path, shape: tuple[int, ...]):
        self.path = path
        self.row_shape = shape[1:]
        self.row_bytes = int(np.prod(self.row_shape))
        created = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)
        self.offset = int(created.offset)
        del created  # unmaps; the file is zero-filled, so rows that are never written stay 0
        self.file = open(path, "r+b")

    # 행 하나를 제자리에 쓴다. 모양이나 dtype이 다르면 잘못된 행이 조용히 섞이지 않게 ValueError를 낸다.
    def write(self, row: int, array: np.ndarray) -> None:
        if array.shape != self.row_shape or array.dtype != np.uint8:
            raise ValueError(f"row {row}: expected uint8 {self.row_shape}, got {array.dtype} {array.shape}")
        self.file.seek(self.offset + row * self.row_bytes)
        self.file.write(np.ascontiguousarray(array).data)

    def close(self) -> None:
        self.file.close()


# tar 안 이름이 "./"로 시작하면 떼어 매니페스트 경로와 같은 형식으로 맞춘다.
def _member_name(name: str) -> str:
    return name[2:] if name.startswith("./") else name


# tar를 한 번 순서대로 읽으며 필요한 멤버만 꺼내 디코딩을 스레드 풀에 맡기고, 결과는 이 스레드에서 차례로
# 파일에 쓴다. 디스크에서 앞뒤로 건너뛰지 않고 한 방향으로만 읽는 것이 핵심이다.
# - 메모리: 대기 중인 작업을 workers × 4개로 묶어(max_pending), 원본 바이트와 디코딩한 배열이 메모리에 무한히
#   쌓이지 않게 한다.
# - 필요한 것을 다 읽으면 tar 끝까지 가지 않고 멈춘다.
# - 다 읽은 뒤에도 매니페스트에 있는데 tar에 없던 멤버가 남으면 FileNotFoundError를 낸다(빈 0 행이 그대로 남는
#   것을 막음).
def _fill(
    tar_path: Path,
    size: int,
    image_rows: dict[str, int],
    mask_rows: dict[str, int],
    images: _RowWriter,
    masks: _RowWriter | None,
    workers: int,
    progress: bool,
) -> None:
    """Read wanted members in tar order, decode them in a thread pool, write rows from this thread."""
    total = len(image_rows) + len(mask_rows)
    max_pending = max(1, workers) * 4  # bounds the encoded bytes and decoded arrays held in memory
    pending: deque = deque()
    done = 0
    started = time.perf_counter()

    # 대기열이 limit보다 길면 가장 오래된 작업부터 결과를 기다려 쓴다(앞에서부터 꺼내므로 쓰기 순서가 제출
    # 순서와 같다). 500개마다 진행 상황을 출력한다.
    def drain(limit: int) -> None:
        nonlocal done
        while len(pending) > limit:
            writer, row, future = pending.popleft()
            writer.write(row, future.result())
            done += 1
            if progress and (done % 500 == 0 or done == total):
                elapsed = time.perf_counter() - started
                print(f"[cache {size}] {done:,}/{total:,} members, {elapsed:.0f} s", flush=True)

    # tar 파일은 1MB 버퍼로 열고 "r:"(압축 없음) 모드로 스트림처럼 읽는다.
    with (
        open(tar_path, "rb", buffering=1 << 20) as raw,
        tarfile.open(fileobj=raw, mode="r:") as tar,
        ThreadPoolExecutor(max_workers=max(1, workers)) as pool,
    ):
        for member in tar:
            if not (image_rows or mask_rows):
                break  # everything wanted has been read
            if not member.isfile():
                continue
            name = _member_name(member.name)
            if name in image_rows:
                row = image_rows.pop(name)
                data = tar.extractfile(member).read()
                pending.append((images, row, pool.submit(_decode_image, data, size)))
            elif name in mask_rows:
                row = mask_rows.pop(name)
                data = tar.extractfile(member).read()
                pending.append((masks, row, pool.submit(_decode_mask, data)))
            else:
                continue
            drain(max_pending)
        drain(0)

    missing = sorted([*image_rows, *mask_rows])
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest members are not in {tar_path}: {missing[:3]}")


# 이번 빌드가 바꿀 파일 쌍(배열, index)을 정한다. 마스크가 재사용 가능하면 이미지 쌍만, 아니면 마스크 쌍을
# 앞에 더한다.
def _targets(out_dir: Path, size: int, index: list[str]) -> tuple[bool, list[tuple[Path, Path]]]:
    """(reuse_masks, [(array, its index file)]): the files a build for `size` replaces, masks first."""
    reuse_masks = _masks_match(out_dir, index)
    pairs = [(images_path(out_dir, size), index_path(out_dir, size))]
    if not reuse_masks:
        pairs.insert(0, (masks_path(out_dir), masks_index_path(out_dir)))
    return reuse_masks, pairs


# 바꿀 파일이 다른 프로세스(닫지 않은 ImageCache의 메모리 맵 등)에 잡혀 있으면 몇 분짜리 tar 읽기를 시작하기
# 전에 바로 PermissionError를 낸다. 파일 이름을 자기 자신으로 바꿔 보는 것은 아무것도 바꾸지 않으면서 같은
# 권한이 필요한 시험이다.
def _check_replaceable(pairs: list[tuple[Path, Path]]) -> None:
    """Raise PermissionError now, not after the build, if a file the build replaces is held open.

    Windows refuses to replace a file that is memory-mapped (an `ImageCache` that was not closed).
    Renaming a file onto itself needs the same access and changes nothing.
    """
    for path in (path for pair in pairs for path in pair):
        if path.exists():
            try:
                os.rename(path, path)
            except PermissionError as err:
                raise PermissionError(
                    f"{path} is in use and cannot be replaced: close every ImageCache on it, then rebuild"
                ) from err


# 다 만든 `.part` 파일을 (배열, index) 쌍마다 최종 이름으로 옮긴다. index 파일이 "옆의 배열은 이 순서다"라는
# 보증이므로, 옛 index를 먼저 치우고 배열을 바꾼 뒤 새 index를 둔다. 어디서 끊겨도 맞지 않는 배열 옆에 index가
# 남지 않는다. 배열 교체 자체가 실패하면 옛 index를 되돌려 이전 캐시를 계속 쓸 수 있게 한다.
def _publish(pairs: list[tuple[Path, Path]], parts: dict[Path, Path]) -> None:
    """Move the finished ".part" files into place, one (array, index) pair at a time.

    An index file vouches for the array next to it. The old index is therefore taken away before
    the array changes and the new one appears only after the new array is in place: whatever
    interrupts this, no index is left next to an array it does not describe. When the array cannot
    be replaced at all, the old index is put back, so the previous cache stays usable.
    """
    for array, index in pairs:
        previous = index.read_bytes() if index.exists() else None
        index.unlink(missing_ok=True)
        try:
            os.replace(parts[array], array)
        except OSError:
            if previous is not None:
                index.write_bytes(previous)  # os.replace failed, so the old array is untouched
            raise
        os.replace(parts[index], index)


# 캐시를 만드는 본 함수. 매니페스트에 같은 이미지가 두 번 있거나 비었으면 거절하고, 모든 파일을 `.part`
# 이름으로 쓴 뒤 끝에 한꺼번에 옮긴다. 실패하면(Ctrl+C 같은 BaseException 포함) `.part` 파일을 지워 반쯤 만든
# 캐시가 남지 않게 한다.
# tar에서 테스트 이미지를 읽는 유일한 코드지만 장부 기록은 여기서 하지 않고 main이 이 함수를 부르기 전에
# 한다(docstring).
def build_cache(
    tar_path: Path,
    manifest: list[ManifestRow],
    size: int,
    out_dir: Path,
    *,
    workers: int = 8,
    progress: bool = True,
) -> None:
    """Build the image array for `size` (and the 256x256 masks unless a matching array exists).

    This is the only code that reads test images from the tar; it does not write the test ledger
    itself (`main` does, before it calls this). Everything is written under ".part" names and
    renamed at the end, each index file after its array; a failed build removes its ".part" files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = [r.image for r in manifest]
    if len(set(index)) != len(index):
        raise ValueError("manifest lists an image more than once")
    n = len(index)
    if n == 0:
        raise ValueError("manifest is empty")
    reuse_masks, pairs = _targets(out_dir, size, index)
    _check_replaceable(pairs)  # before the pass over the tar, which takes minutes
    image_rows = {image: i for i, image in enumerate(index)}
    mask_rows = {} if reuse_masks else {r.mask: i for i, r in enumerate(manifest) if r.mask}
    parts = {final: final.with_name(final.name + _PART) for pair in pairs for final in pair}

    # 안쪽 try/finally는 성공·실패와 관계없이 파일 핸들을 닫는다(Windows에서 닫지 않으면 이름을 바꾸거나 지울
    # 수 없다). 바깥 try는 실패하면 `.part`를 지운다.
    images = masks = None
    try:
        try:
            images = _RowWriter(parts[images_path(out_dir, size)], (n, size, size, 3))
            if not reuse_masks:
                masks = _RowWriter(parts[masks_path(out_dir)], (n, MASK_SIZE, MASK_SIZE))
            _fill(Path(tar_path), size, image_rows, mask_rows, images, masks, workers, progress)
        finally:
            for writer in (images, masks):
                if writer is not None:
                    writer.close()
        for _, index_file in pairs:
            _write_index(parts[index_file], index)
        _publish(pairs, parts)
    except BaseException:
        for part in parts.values():
            part.unlink(missing_ok=True)
        raise


# 메모리 맵 배열에서 원하는 행들을 메모리로 복사한다. 요청 순서 그대로 읽으면 디스크에서 파일 위치를
# 오가므로, 행 번호를 오름차순으로 정렬해 파일 앞에서 뒤로 256행씩 읽고 결과는 원래 순서 자리에 넣는다.
def _take(source: np.ndarray, rows: np.ndarray, chunk: int = 256) -> np.ndarray:
    """In-memory copy of `source[rows]`, read in ascending file order in small chunks."""
    out = np.empty((len(rows), *source.shape[1:]), dtype=source.dtype)
    order = np.argsort(rows, kind="stable")
    for start in range(0, len(rows), chunk):
        sel = order[start : start + chunk]
        out[sel] = source[rows[sel]]
    return out


# 완성된 캐시의 읽기 전용 접근자. 이미지·마스크 배열을 메모리 맵으로 열고, 매니페스트 행(ManifestRow)을 주면
# 그 순서대로 이미지 [N, S, S, 3]이나 마스크 [N, 256, 256]을 돌려준다. 이미지 index와 마스크 index가
# 다르면(다른 매니페스트로 만든 마스크) 열지 않는다.
# 봉인 테스트를 막지 않는다. 어떤 행을 읽을지는 splits.select가 --allow-test를 보고 정한다(docstring).
class ImageCache:
    """Read access to a finished cache. It does not enforce the test seal: `splits.select` does."""

    def __init__(self, out_dir: Path, size: int):
        out_dir = Path(out_dir)
        self.size = int(size)
        index = _read_index(index_path(out_dir, self.size))
        if _read_index(masks_index_path(out_dir)) != index:
            raise ValueError(f"masks in {out_dir} were built from a different manifest: rebuild the cache")
        self._row = {image: i for i, image in enumerate(index)}
        self._images = np.load(images_path(out_dir, self.size), mmap_mode="r")
        self._masks = np.load(masks_path(out_dir), mmap_mode="r")
        if self._images.shape != (len(index), self.size, self.size, 3):
            raise ValueError(f"unexpected image array shape {self._images.shape}")
        if self._masks.shape != (len(index), MASK_SIZE, MASK_SIZE):
            raise ValueError(f"unexpected mask array shape {self._masks.shape}")

    def __len__(self) -> int:
        return len(self._row)

    def _rows(self, rows: Sequence[ManifestRow]) -> np.ndarray:
        try:
            return np.fromiter((self._row[r.image] for r in rows), dtype=np.int64, count=len(rows))
        except KeyError as err:
            raise KeyError(f"image not in the cache index: {err.args[0]}") from None

    def images(self, rows: Sequence[ManifestRow]) -> np.ndarray:
        """uint8 [len(rows), size, size, 3], in the order of `rows`."""
        return _take(self._images, self._rows(rows))

    def masks(self, rows: Sequence[ManifestRow]) -> np.ndarray:
        """uint8 {0, 1} [len(rows), 256, 256], in the order of `rows` (zeros for normal images)."""
        return _take(self._masks, self._rows(rows))

    # 메모리 맵을 놓는다. 같은 프로세스에서 캐시를 다시 만들기 전에 불러야 Windows가 파일 교체를
    # 허락한다(_check_replaceable의 오류 메시지가 가리키는 것).
    def close(self) -> None:
        """Drop the memory maps (Windows cannot replace a file that is still mapped)."""
        self._images = self._masks = None


# 지금 마스크 배열에 기대는 다른 크기의 완성된 캐시를 찾는다. 다른 매니페스트로 마스크를 다시 만들면 그
# 캐시들의 index와 마스크 index가 어긋나 ImageCache가 열리지 않으므로, main이 미리 경고하는 데 쓴다.
def _sizes_using_masks(out_dir: Path, size: int) -> list[int]:
    """Other image sizes in `out_dir` whose finished cache relies on the current mask array."""
    if not masks_index_path(out_dir).exists():
        return []
    mask_index = _read_index(masks_index_path(out_dir))
    sizes = []
    for path in Path(out_dir).glob("visa_*_index.csv"):
        match = re.fullmatch(r"visa_(\d+)_index\.csv", path.name)
        if match and int(match[1]) != size and _read_index(path) == mask_index:
            sizes.append(int(match[1]))
    return sorted(sizes)


# 명령행 입구(`python -m defect_inspect.cache --size 256` 등). 흐름: 이미 완성됐으면 건너뜀 → 마스크를 새로
# 만들면 다른 크기 캐시가 못 쓰게 되는 경우 --force 없이는 멈춤 → 파일 잠금 확인 → 매니페스트에 봉인 테스트
# 행이 있으면 장부에 먼저 기록 → build_cache → 결과 파일 크기와 모양 출력. 장부 줄이 생기므로 실행 뒤 작업
# 트리가 dirty가 된다.
def main(argv: list[str] | None = None) -> int:
    from defect_inspect import ledger, paths
    from defect_inspect.splits import SEALED_ROLES, read_manifest

    parser = argparse.ArgumentParser(description="Build the resized VisA cache from the tar.")
    parser.add_argument("--size", type=int, default=256, help="side of the square images")
    parser.add_argument("--workers", type=int, default=8, help="decoder threads")
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even if the cache is complete, or if new masks would invalidate other sizes",
    )
    parser.add_argument("--tar", type=Path, default=paths.VISA_TAR, help="VisA tar file")
    parser.add_argument("--manifest", type=Path, default=paths.VISA_MANIFEST, help="split manifest CSV")
    parser.add_argument("--out", type=Path, default=paths.CACHE, help="cache directory")
    args = parser.parse_args(argv)

    manifest = read_manifest(args.manifest)
    if not args.force and cache_is_complete(args.out, args.size, manifest):
        print(f"cache for size {args.size} is already complete in {args.out} (use --force to rebuild)")
    else:
        index = [r.image for r in manifest]
        reuse_masks, pairs = _targets(args.out, args.size, index)
        stranded = [] if reuse_masks or args.force else _sizes_using_masks(args.out, args.size)
        if stranded and _read_index(masks_index_path(args.out)) != index:
            sizes = ", ".join(str(s) for s in stranded)
            print(
                f"ERROR: the masks in {args.out} were built from another manifest (or row order). "
                f"Rebuilding them would make the finished cache of size {sizes} unusable; nothing was "
                "done. Pass --force to rebuild anyway, then rebuild those sizes."
            )
            return 1
        _check_replaceable(pairs)
        # 읽기 전에 기록한다: 기록이 실패하면 테스트 이미지를 읽지 않은 채 멈춘다. config 이름은
        # `cache-<크기>`이고, 점수나 지표를 내지 않는 캐시 생성임을 note에 적는다.
        if any(r.role in SEALED_ROLES for r in manifest):
            # The sealed test images are about to be decoded: that is a read, and every read is
            # recorded before it happens (docs/experiments.md).
            ledger.record_test_access(
                paths.TEST_LEDGER,
                stage="cache",
                config=f"cache-{args.size}",
                note="resize cache build: pixels copied into arrays, no scores or metrics",
            )
            print(f"recorded the read of the sealed test images in {paths.TEST_LEDGER}")
        build_cache(args.tar, manifest, args.size, args.out, workers=args.workers)
    for path in (images_path(args.out, args.size), masks_path(args.out)):
        print(f"{path}  {path.stat().st_size:,} bytes  shape {_npy_shape(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
