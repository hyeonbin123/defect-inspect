"""Resized image and mask cache: one sequential pass over the tar into a few large .npy files."""

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

import numpy as np
from PIL import Image

from defect_inspect.splits import ManifestRow

MASK_SIZE = 256
_PART = ".part"


def images_path(out_dir: Path, size: int) -> Path:
    return Path(out_dir) / f"visa_{size}_images.npy"


def index_path(out_dir: Path, size: int) -> Path:
    return Path(out_dir) / f"visa_{size}_index.csv"


def masks_path(out_dir: Path) -> Path:
    return Path(out_dir) / f"visa_masks_{MASK_SIZE}.npy"


def masks_index_path(out_dir: Path) -> Path:
    # Row order of the mask array, kept next to it so that other sizes can check it before reusing it.
    return Path(out_dir) / f"visa_masks_{MASK_SIZE}_index.csv"


def resize_image(img: Image.Image, size: int) -> np.ndarray:
    """RGB uint8 [size, size, 3]; aspect ratio ignored, bicubic (antialiased when shrinking)."""
    return np.asarray(img.convert("RGB").resize((size, size), Image.Resampling.BICUBIC), dtype=np.uint8)


def resize_mask(mask: Image.Image, size: int = MASK_SIZE) -> np.ndarray:
    """uint8 {0, 1} [size, size]: 1 where any source pixel of the box is a defect (value > 0)."""
    source = np.asarray(mask)
    if source.ndim != 2:
        raise ValueError(f"expected a single-channel mask, got mode {mask.mode}")
    binary = Image.fromarray(np.where(source > 0, 255, 0).astype(np.uint8))
    small = binary.resize((size, size), Image.Resampling.BOX)
    return (np.asarray(small) > 0).astype(np.uint8)


def _decode_image(data: bytes, size: int) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as img:
        return resize_image(img, size)


def _decode_mask(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as mask:
        return resize_mask(mask)


def _read_index(path: Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()


def _write_index(path: Path, images: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(f"{image}\n" for image in images))


def _npy_shape(path: Path) -> tuple[int, ...]:
    array = np.load(path, mmap_mode="r")
    shape = array.shape
    del array
    return shape


def _masks_match(out_dir: Path, index: list[str]) -> bool:
    """True when a finished mask array with exactly this row order is already in `out_dir`."""
    if not (masks_path(out_dir).exists() and masks_index_path(out_dir).exists()):
        return False
    if _read_index(masks_index_path(out_dir)) != index:
        return False
    return _npy_shape(masks_path(out_dir)) == (len(index), MASK_SIZE, MASK_SIZE)


def cache_is_complete(out_dir: Path, size: int, manifest: list[ManifestRow]) -> bool:
    """True when images, masks and both indexes exist for `size` and follow the manifest order."""
    index = [r.image for r in manifest]
    if not (images_path(out_dir, size).exists() and index_path(out_dir, size).exists()):
        return False
    if _read_index(index_path(out_dir, size)) != index or not _masks_match(out_dir, index):
        return False
    return _npy_shape(images_path(out_dir, size)) == (len(index), size, size, 3)


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

    def write(self, row: int, array: np.ndarray) -> None:
        if array.shape != self.row_shape or array.dtype != np.uint8:
            raise ValueError(f"row {row}: expected uint8 {self.row_shape}, got {array.dtype} {array.shape}")
        self.file.seek(self.offset + row * self.row_bytes)
        self.file.write(np.ascontiguousarray(array).data)

    def close(self) -> None:
        self.file.close()


def _member_name(name: str) -> str:
    return name[2:] if name.startswith("./") else name


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

    def drain(limit: int) -> None:
        nonlocal done
        while len(pending) > limit:
            writer, row, future = pending.popleft()
            writer.write(row, future.result())
            done += 1
            if progress and (done % 500 == 0 or done == total):
                elapsed = time.perf_counter() - started
                print(f"[cache {size}] {done:,}/{total:,} members, {elapsed:.0f} s", flush=True)

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


def _targets(out_dir: Path, size: int, index: list[str]) -> tuple[bool, list[tuple[Path, Path]]]:
    """(reuse_masks, [(array, its index file)]): the files a build for `size` replaces, masks first."""
    reuse_masks = _masks_match(out_dir, index)
    pairs = [(images_path(out_dir, size), index_path(out_dir, size))]
    if not reuse_masks:
        pairs.insert(0, (masks_path(out_dir), masks_index_path(out_dir)))
    return reuse_masks, pairs


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


def _take(source: np.ndarray, rows: np.ndarray, chunk: int = 256) -> np.ndarray:
    """In-memory copy of `source[rows]`, read in ascending file order in small chunks."""
    out = np.empty((len(rows), *source.shape[1:]), dtype=source.dtype)
    order = np.argsort(rows, kind="stable")
    for start in range(0, len(rows), chunk):
        sel = order[start : start + chunk]
        out[sel] = source[rows[sel]]
    return out


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

    def close(self) -> None:
        """Drop the memory maps (Windows cannot replace a file that is still mapped)."""
        self._images = self._masks = None


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
