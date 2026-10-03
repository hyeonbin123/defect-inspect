"""M2AD (Motor and Bird, three views, ten illuminations): metadata, specimen folds and the resize cache.

Rules are registered in docs/experiments.md, stage 3-B. The cache build is the only code that reads M2AD
images from the zips; it copies test images too, so `main` records it in the test ledger first.
"""

import argparse
import json
import time
import zipfile
from collections import Counter, deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import cache as _cache
from .splits import path_key

CATEGORIES = ("Motor", "Bird")
VIEWS = ("000", "120", "240")
ILLUMINATIONS = ("01", "02", "03", "04", "05", "06", "07", "08", "09", "10")
REFERENCE = "01"
N_FOLDS = 5
RECAL_SIZES = (8, 30)
SPLITS = ("train", "test")
META_MEMBER = "meta_unsupervised.json"


@dataclass(frozen=True)
class M2adRow:
    category: str
    split: str  # "train" | "test"
    specimen: str  # object_name: one physical object
    view: str
    illumination: str
    img_path: str  # member name inside <category>.zip
    object_anomaly: int  # 1 when the specimen is defective
    label: int  # 0 normal, 1 defect, -1 excluded


def label_of(object_anomaly: int, image_anomaly: int, detectable: object) -> int:
    """Registered label rule: 0 for a normal specimen, 1 for a visible and detectable defect, else -1."""
    if object_anomaly == 0:
        return 0
    if image_anomaly == 1 and detectable is True:
        return 1
    return -1


def _row(category: str, split: str, entry: dict) -> M2adRow:
    object_anomaly, image_anomaly = entry["object_anomaly"], entry["image_anomaly"]
    detectable = entry["detectable"]
    where = f"{split}/{category}: {entry.get('img_path')!r}"
    if entry["illumination"] not in ILLUMINATIONS:
        raise ValueError(f"{where}: unknown illumination {entry['illumination']!r}")
    for name, value in (("object_anomaly", object_anomaly), ("image_anomaly", image_anomaly)):
        if isinstance(value, bool) or value not in (0, 1):
            raise ValueError(f"{where}: {name} must be 0 or 1, got {value!r}")
    if not (isinstance(detectable, bool) or detectable == ""):
        raise ValueError(f'{where}: detectable must be "" or a bool, got {detectable!r}')
    if object_anomaly == 0 and image_anomaly != 0:
        raise ValueError(f"{where}: a normal specimen has image_anomaly = {image_anomaly}")
    return M2adRow(
        category=category,
        split=split,
        specimen=str(entry["object_name"]),
        view=entry["view"],
        illumination=entry["illumination"],
        img_path=entry["img_path"],
        object_anomaly=int(object_anomaly),
        label=label_of(object_anomaly, image_anomaly, detectable),
    )


def _sort_key(row: M2adRow) -> tuple:
    # Categories and splits in their declared order (Motor before Bird, train before test).
    return (CATEGORIES.index(row.category), SPLITS.index(row.split), row.specimen, row.view, row.illumination)


def _check(rows: list[M2adRow]) -> None:
    keys = Counter((r.category, r.split, r.specimen, r.view, r.illumination) for r in rows)
    repeated = sorted(key for key, n in keys.items() if n > 1)
    if repeated:
        raise ValueError(f"image listed more than once: {repeated[:3]}")
    paths_seen = Counter(r.img_path for r in rows)
    repeated_paths = sorted(p for p, n in paths_seen.items() if n > 1)
    if repeated_paths:
        raise ValueError(f"img_path listed more than once: {repeated_paths[:3]}")
    defective_train = sorted({r.img_path for r in rows if r.split == "train" and r.object_anomaly != 0})
    if defective_train:
        raise ValueError(f"train rows must be normal specimens: {defective_train[:3]}")
    status: dict[tuple[str, str], set[tuple[str, int]]] = {}
    for r in rows:
        status.setdefault((r.category, r.specimen), set()).add((r.split, r.object_anomaly))
    mixed = sorted(key for key, seen in status.items() if len(seen) > 1)
    if mixed:
        raise ValueError(f"specimen in both splits or with a changing object_anomaly: {mixed[:3]}")


def read_meta(jsons_zip: Path) -> list[M2adRow]:
    """Rows of CATEGORIES x VIEWS (all ten illuminations) from `meta_unsupervised.json` in `jsons_zip`.

    Sorted by (category, split, specimen, view, illumination), categories and splits in declared order.
    """
    with zipfile.ZipFile(jsons_zip) as z:
        members = [name for name in z.namelist() if name.rsplit("/", 1)[-1] == META_MEMBER]
        if len(members) != 1:
            raise FileNotFoundError(f"expected one {META_MEMBER} in {jsons_zip}, found {members}")
        meta = json.loads(z.read(members[0]).decode("utf-8"))
    rows: list[M2adRow] = []
    for split in SPLITS:
        for category in CATEGORIES:
            try:
                entries = meta[split][category]
            except KeyError:
                raise ValueError(f"{jsons_zip}: no entries for {split}/{category}") from None
            rows += [_row(category, split, entry) for entry in entries if entry["view"] in VIEWS]
    rows.sort(key=_sort_key)
    _check(rows)
    return rows


def _ranked(specimens: Sequence[str]) -> list[str]:
    """Distinct specimen names in the order of the SHA-256 hex of the name."""
    names = set(specimens)
    if not all(isinstance(name, str) for name in names):
        raise TypeError("specimen names must be str")
    return sorted(names, key=path_key)


def specimen_folds(specimens: Sequence[str]) -> dict[str, int]:
    """specimen -> fold: rank in SHA-256 order modulo N_FOLDS (the scheme of `splits`)."""
    return {name: rank % N_FOLDS for rank, name in enumerate(_ranked(specimens))}


def recal_specimens(specimens: Sequence[str], n: int) -> list[str]:
    """The first `n` specimens in SHA-256 order (nested across n)."""
    ordered = _ranked(specimens)
    if not 0 <= n <= len(ordered):
        raise ValueError(f"n={n} is outside the {len(ordered)} specimens")
    return ordered[:n]


def illumination_groups() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Stage 7: the nine non-reference illuminations in SHA-256 order of their two-digit label, cut into
    the first four (group A) and the other five (group B). Each group is returned in label order."""
    ranked = sorted((light for light in ILLUMINATIONS if light != REFERENCE), key=path_key)
    return tuple(sorted(ranked[:4])), tuple(sorted(ranked[4:]))


def count_table(rows: Sequence[M2adRow]) -> str:
    """Image counts per (category, split, label) and, for the test split, per view and illumination."""
    lines = [f"{'category':<10}{'split':<7}{'specimens':>10}{'normal':>8}{'defect':>8}{'excluded':>9}"]
    for category in CATEGORIES:
        for split in SPLITS:
            part = [r for r in rows if r.category == category and r.split == split]
            by_label = Counter(r.label for r in part)
            specimens = len({r.specimen for r in part})
            lines.append(
                f"{category:<10}{split:<7}{specimens:>10}{by_label[0]:>8}{by_label[1]:>8}{by_label[-1]:>9}"
            )
    lines.append("test images as normal/defect/excluded, per (category, view) and illumination:")
    lines.append(f"{'':<12}" + "".join(f"{f'I{light}':>10}" for light in ILLUMINATIONS))
    for category in CATEGORIES:
        for view in VIEWS:
            cells = []
            for light in ILLUMINATIONS:
                by_label = Counter(
                    r.label
                    for r in rows
                    if (r.category, r.split, r.view, r.illumination) == (category, "test", view, light)
                )
                cells.append(f"{by_label[0]}/{by_label[1]}/{by_label[-1]}")
            lines.append(f"{category + ' ' + view:<12}" + "".join(f"{c:>10}" for c in cells))
    return "\n".join(lines)


def images_path(out_dir: Path, size: int) -> Path:
    return Path(out_dir) / f"m2ad_{size}_images.npy"


def index_path(out_dir: Path, size: int) -> Path:
    return Path(out_dir) / f"m2ad_{size}_index.csv"


def cache_is_complete(out_dir: Path, size: int, rows: Sequence[M2adRow]) -> bool:
    """True when the image array and its index exist for `size` and follow the order of `rows`."""
    if not (images_path(out_dir, size).exists() and index_path(out_dir, size).exists()):
        return False
    index = [r.img_path for r in rows]
    if _cache._read_index(index_path(out_dir, size)) != index:
        return False
    return _cache._npy_shape(images_path(out_dir, size)) == (len(index), size, size, 3)


def _members(zip_path: Path, wanted: dict[str, int]) -> list[zipfile.ZipInfo]:
    """The wanted members of one zip in the order they lie in the file; all of them must be there."""
    with zipfile.ZipFile(zip_path) as z:
        found = {info.filename: info for info in z.infolist() if info.filename in wanted}
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise FileNotFoundError(f"{len(missing)} images are not in {zip_path}: {missing[:3]}")
    return sorted(found.values(), key=lambda info: info.header_offset)


def _plan(zip_dir: Path, rows: Sequence[M2adRow]) -> list[tuple[Path, list[zipfile.ZipInfo], dict[str, int]]]:
    """What a build reads: per `<zip_dir>/<category>.zip` its wanted members in file order and their rows.

    Central directories only, no image is read: a missing zip or member (FileNotFoundError) and unusable
    rows (ValueError) show up here, so `main` calls this before it writes the ledger line.
    """
    index = [r.img_path for r in rows]
    if not index:
        raise ValueError("no rows to cache")
    if len(set(index)) != len(index):
        raise ValueError("rows list an image more than once")
    wanted: dict[str, dict[str, int]] = {}
    for i, row in enumerate(rows):
        wanted.setdefault(row.category, {})[row.img_path] = i
    plan = []
    for category, members in wanted.items():
        zip_path = Path(zip_dir) / f"{category}.zip"
        plan.append((zip_path, _members(zip_path, members), members))
    return plan


def _fill(
    plan: list[tuple[Path, list[zipfile.ZipInfo], dict[str, int]]],
    size: int,
    writer: _cache._RowWriter,
    workers: int,
    progress: bool,
) -> None:
    """Read each zip front to back, decode in a thread pool, write rows from this thread."""
    total = sum(len(infos) for _, infos, _ in plan)
    max_pending = max(1, workers) * 4  # bounds the encoded bytes and decoded arrays held in memory
    pending: deque = deque()
    done = 0
    started = time.perf_counter()

    def drain(limit: int) -> None:
        nonlocal done
        while len(pending) > limit:
            row, future = pending.popleft()
            writer.write(row, future.result())
            done += 1
            if progress and (done % 500 == 0 or done == total):
                elapsed = time.perf_counter() - started
                print(f"[m2ad cache {size}] {done:,}/{total:,} images, {elapsed:.0f} s", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for zip_path, infos, wanted in plan:
            with zipfile.ZipFile(zip_path) as z:
                for info in infos:
                    data = z.read(info)
                    pending.append((wanted[info.filename], pool.submit(_cache._decode_image, data, size)))
                    drain(max_pending)
        drain(0)


def build_cache(
    zip_dir: Path,
    rows: Sequence[M2adRow],
    size: int,
    out_dir: Path,
    *,
    workers: int = 8,
    progress: bool = False,
) -> None:
    """Write `m2ad_{size}_images.npy` (uint8 [len(rows), size, size, 3]) and its index, in `rows` order.

    Only the members named by `rows` are read, from `<zip_dir>/<category>.zip`, in the order they lie in
    each zip (sequential on an HDD). It does not write the test ledger itself (`main` does, before it calls
    this). Everything is written under ".part" names and renamed at the end, the index after the array.
    """
    out_dir = Path(out_dir)
    # Central directories only: a missing zip or member stops the build before anything is decoded.
    plan = _plan(zip_dir, rows)
    index = [r.img_path for r in rows]

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [(images_path(out_dir, size), index_path(out_dir, size))]
    _cache._check_replaceable(pairs)
    parts = {final: final.with_name(final.name + _cache._PART) for pair in pairs for final in pair}
    writer = None
    try:
        try:
            writer = _cache._RowWriter(parts[images_path(out_dir, size)], (len(index), size, size, 3))
            _fill(plan, size, writer, workers, progress)
        finally:
            if writer is not None:
                writer.close()
        _cache._write_index(parts[index_path(out_dir, size)], index)
        _cache._publish(pairs, parts)
    except BaseException:
        for part in parts.values():
            part.unlink(missing_ok=True)
        raise


class M2adCache:
    """Read access to a finished M2AD cache. It does not enforce the test seal: `run_m2ad` does."""

    def __init__(self, out_dir: Path, size: int):
        out_dir = Path(out_dir)
        self.size = int(size)
        if not (images_path(out_dir, self.size).exists() and index_path(out_dir, self.size).exists()):
            raise FileNotFoundError(
                f"no M2AD cache of size {self.size} in {out_dir}: build it with "
                f"`python -m defect_inspect.m2ad --size {self.size}`"
            )
        index = _cache._read_index(index_path(out_dir, self.size))
        self._row = {img_path: i for i, img_path in enumerate(index)}
        self._images = np.load(images_path(out_dir, self.size), mmap_mode="r")
        if self._images.shape != (len(index), self.size, self.size, 3):
            raise ValueError(f"unexpected image array shape {self._images.shape}")

    def __len__(self) -> int:
        return len(self._row)

    def images(self, rows: Sequence[M2adRow]) -> np.ndarray:
        """uint8 [len(rows), size, size, 3], in the order of `rows`."""
        try:
            index = np.fromiter((self._row[r.img_path] for r in rows), dtype=np.int64, count=len(rows))
        except KeyError as err:
            raise KeyError(f"image not in the cache index: {err.args[0]}") from None
        return _cache._take(self._images, index)

    def close(self) -> None:
        """Drop the memory map (Windows cannot replace a file that is still mapped)."""
        self._images = None


def main(argv: list[str] | None = None) -> int:
    from . import ledger, paths

    parser = argparse.ArgumentParser(description="Build the resized M2AD cache from the zips.")
    parser.add_argument("--size", type=int, default=256, help="side of the square images")
    parser.add_argument("--workers", type=int, default=8, help="decoder threads")
    parser.add_argument("--force", action="store_true", help="rebuild even if the cache is complete")
    parser.add_argument("--zips", type=Path, default=None, help="folder with <category>.zip (data/raw/m2ad)")
    parser.add_argument("--jsons", type=Path, default=None, help="jsons.zip (default: in the zip folder)")
    parser.add_argument("--out", type=Path, default=None, help="cache directory (data/cache)")
    args = parser.parse_args(argv)
    zip_dir = args.zips or paths.RAW / "m2ad"
    out_dir = args.out or paths.CACHE

    rows = read_meta(args.jsons or zip_dir / "jsons.zip")
    print(count_table(rows))
    if not args.force and cache_is_complete(out_dir, args.size, rows):
        print(f"M2AD cache for size {args.size} is already complete in {out_dir} (use --force to rebuild)")
    else:
        # Everything that can fail without reading an image comes before the ledger line: a build that
        # finds a zip or a member missing, or cannot replace the old files, leaves no entry.
        _plan(zip_dir, rows)
        _cache._check_replaceable([(images_path(out_dir, args.size), index_path(out_dir, args.size))])
        if any(r.split == "test" for r in rows):
            # Test images are about to be decoded: every read is recorded before it happens.
            ledger.record_test_access(
                paths.TEST_LEDGER,
                stage="cache",
                config=f"m2ad-cache-{args.size}",
                note="M2AD resize cache build: pixels copied into arrays, no scores or metrics",
            )
            print(f"recorded the read of the M2AD test images in {paths.TEST_LEDGER}")
        build_cache(zip_dir, rows, args.size, out_dir, workers=args.workers, progress=True)
    path = images_path(out_dir, args.size)
    print(f"{path}  {path.stat().st_size:,} bytes  shape {_cache._npy_shape(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
