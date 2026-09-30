"""Read the VisA split and annotation CSVs straight from the tar (no extraction, no image decoding)."""

import csv
import functools
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path

CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)
SPLIT_NAMES = ("1cls", "2cls_fewshot", "2cls_highshot")
SPLIT_HEADER = ["object", "split", "label", "image", "mask"]
ANNO_HEADER = ["image", "label", "mask"]


@dataclass(frozen=True)
class VisaRow:
    category: str
    split: str  # "train" | "test"
    label: str  # "normal" | "anomaly"
    image: str  # member name of the image, e.g. candle/Data/Images/Normal/0836.JPG
    mask: str  # member name of the mask, "" for normal images


def _member_name(name: str) -> str:
    return name[2:] if name.startswith("./") else name


@functools.lru_cache(maxsize=4)
def _scan_csv_members(tar_path: str, size: int, mtime_ns: int) -> dict[str, str]:
    """Text of every CSV member, from one pass over the tar headers.

    The split CSVs sit at the very end of the real tar, so a pass is a full scan of ~12,000 headers.
    The result is cached per (path, size, mtime) so that one process scans the tar only once.
    """
    wanted = {f"split_csv/{name}.csv" for name in SPLIT_NAMES} | {f"{c}/image_anno.csv" for c in CATEGORIES}
    found: dict[str, str] = {}
    with tarfile.open(tar_path, "r:") as tar:
        for member in tar:
            name = _member_name(member.name)
            if not member.isfile() or not name.lower().endswith(".csv"):
                continue
            found[name] = tar.extractfile(member).read().decode("utf-8-sig")
            if wanted <= found.keys():
                break
    return found


def _csv_members(tar_path: Path) -> dict[str, str]:
    stat = Path(tar_path).stat()
    return _scan_csv_members(str(Path(tar_path).resolve()), stat.st_size, stat.st_mtime_ns)


def _rows(text: str, header: list[str], member: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != header:
        raise ValueError(f"{member}: expected header {header}, got {reader.fieldnames}")
    return list(reader)


def read_split_csv(tar_path: Path, name: str = "2cls_highshot") -> list[VisaRow]:
    """Rows of `split_csv/<name>.csv` in file order."""
    member = f"split_csv/{name}.csv"
    members = _csv_members(tar_path)
    if member not in members:
        raise KeyError(f"{member} not found in {tar_path}")
    rows = []
    for r in _rows(members[member], SPLIT_HEADER, member):
        # csv.DictReader gives None for cells a short row lacks and files extra cells under the key None.
        if None in r or any(r[key] is None for key in SPLIT_HEADER[:-1]):
            raise ValueError(
                f"{member}: expected {len(SPLIT_HEADER)} columns (the mask may be left out): {r}"
            )
        # A normal row without its trailing comma has no mask cell at all: that is an empty mask.
        row = VisaRow(r["object"], r["split"], r["label"], r["image"], r["mask"] or "")
        if row.split not in ("train", "test") or row.label not in ("normal", "anomaly"):
            raise ValueError(f"{member}: unexpected split/label in {r}")
        if (row.label == "anomaly") != bool(row.mask):
            raise ValueError(f"{member}: anomaly rows need a mask and normal rows must not have one: {r}")
        rows.append(row)
    return rows


def read_defect_types(tar_path: Path) -> dict[str, tuple[str, ...]]:
    """Image path -> sorted, distinct defect type names, for anomaly images (`<category>/image_anno.csv`)."""
    types: dict[str, tuple[str, ...]] = {}
    for member, text in _csv_members(tar_path).items():
        if member.count("/") != 1 or not member.endswith("/image_anno.csv"):
            continue
        for r in _rows(text, ANNO_HEADER, member):
            if r["label"].strip() == "normal":
                continue
            names = {name.strip() for name in r["label"].split(",")} - {""}
            types[r["image"]] = tuple(sorted(names))
    return types
