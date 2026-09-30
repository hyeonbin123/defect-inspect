"""Deterministic split manifest for VisA (rules in docs/experiments.md) and the sealed-test gate."""

import argparse
import csv
import hashlib
from collections import Counter
from dataclasses import astuple, dataclass, fields
from pathlib import Path

from defect_inspect.visa import CATEGORIES, VisaRow

N_FOLDS = 5
N_DEV_DEFECTS = 20
K_VALUES = (5, 10, 20, 40)
SEEDS = (0, 1, 2)
ROLES = ("pool_normal", "dev_defect", "label_pool", "test_normal", "test_defect")
SEALED_ROLES = ("test_normal", "test_defect")  # the sealed test set
PROTOCOLS = ("dev", "test")
PARTS = ("pool_normal", "eval_normal", "eval_defect")
# Totals of the official 2cls_highshot split under these rules (docs/experiments.md).
EXPECTED_TOTALS = {
    "pool_normal": 5773,
    "dev_defect": 240,
    "label_pool": 480,
    "test_normal": 3848,
    "test_defect": 480,
}


class SealedTestError(RuntimeError):
    """Raised when the sealed test set is requested without `allow_test=True`."""


def path_key(path: str, salt: str = "") -> str:
    """Sort key: SHA-256 hex of the path, or of "<salt>:<path>" when a salt is given.

    The salt must be a string (`str(seed)`): an integer 0 would otherwise read as "no salt".
    """
    if not isinstance(path, str) or not isinstance(salt, str):
        raise TypeError(f"path and salt must be str, got {type(path).__name__} and {type(salt).__name__}")
    text = path if salt == "" else f"{salt}:{path}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ManifestRow:
    image: str
    mask: str
    category: str
    label: str  # "normal" | "anomaly"
    role: str  # one of ROLES
    fold: int  # 0..4 for pool_normal, else -1
    defect_types: str  # "|".join(types), "" for normal


MANIFEST_HEADER = [f.name for f in fields(ManifestRow)]


def _category_order(category: str) -> tuple[int, str]:
    # Known categories in CATEGORIES order; anything else (synthetic data) after them, by name.
    return (CATEGORIES.index(category), "") if category in CATEGORIES else (len(CATEGORIES), category)


def build_manifest(rows: list[VisaRow], defect_types: dict[str, tuple[str, ...]]) -> list[ManifestRow]:
    """Assign a role (and a fold for pool normals) to every row of the official 2cls_highshot split."""
    images = Counter(r.image for r in rows)
    repeated = sorted(image for image, n in images.items() if n > 1)
    if repeated:
        raise ValueError(f"image listed more than once: {repeated[:3]}")

    manifest: list[ManifestRow] = []
    for category in sorted({r.category for r in rows}, key=_category_order):
        by_group: dict[tuple[str, str], list[VisaRow]] = {}
        for r in rows:
            if r.category == category:
                if r.split not in ("train", "test") or r.label not in ("normal", "anomaly"):
                    raise ValueError(f"unexpected split/label: {r}")
                by_group.setdefault((r.split, r.label), []).append(r)

        assigned: list[tuple[VisaRow, str, int]] = []
        pool = sorted(by_group.get(("train", "normal"), []), key=lambda r: path_key(r.image))
        assigned += [(r, "pool_normal", rank % N_FOLDS) for rank, r in enumerate(pool)]
        defects = sorted(by_group.get(("train", "anomaly"), []), key=lambda r: path_key(r.image))
        assigned += [(r, "dev_defect", -1) for r in defects[:N_DEV_DEFECTS]]
        assigned += [(r, "label_pool", -1) for r in defects[N_DEV_DEFECTS:]]
        assigned += [(r, "test_normal", -1) for r in by_group.get(("test", "normal"), [])]
        assigned += [(r, "test_defect", -1) for r in by_group.get(("test", "anomaly"), [])]

        assigned.sort(key=lambda item: (ROLES.index(item[1]), item[0].image))
        for r, role, fold in assigned:
            types = "|".join(defect_types.get(r.image, ())) if r.label == "anomaly" else ""
            manifest.append(ManifestRow(r.image, r.mask, category, r.label, role, fold, types))
    return manifest


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    """Write the manifest as UTF-8 CSV with LF line endings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(MANIFEST_HEADER)
        writer.writerows(astuple(r) for r in rows)


def read_manifest(path: Path) -> list[ManifestRow]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != MANIFEST_HEADER:
            raise ValueError(f"{path}: expected header {MANIFEST_HEADER}, got {reader.fieldnames}")
        return [
            ManifestRow(
                r["image"], r["mask"], r["category"], r["label"], r["role"], int(r["fold"]), r["defect_types"]
            )
            for r in reader
        ]


def label_subset(manifest: list[ManifestRow], category: str, k: int, seed: int) -> list[ManifestRow]:
    """First k images of the category's label pool in the order of seed `seed` (nested across k)."""
    pool = [r for r in manifest if r.category == category and r.role == "label_pool"]
    if not 0 <= k <= len(pool):
        raise ValueError(f"k={k} is outside the label pool of {category} ({len(pool)} images)")
    pool.sort(key=lambda r: path_key(r.image, salt=str(seed)))
    return pool[:k]


def pool_folds(protocol: str) -> tuple[int, ...]:
    """Folds whose normals form the normal pool: dev keeps fold 0 for evaluation."""
    if protocol == "dev":
        return tuple(range(1, N_FOLDS))
    if protocol == "test":
        return tuple(range(N_FOLDS))
    raise ValueError(f"unknown protocol: {protocol!r} (expected one of {PROTOCOLS})")


def holdout_fold(protocol: str) -> int:
    """Fold left out of the bank by the hold-out calibration: the first fold of the pool."""
    return pool_folds(protocol)[0]


def select(
    manifest: list[ManifestRow],
    *,
    protocol: str,
    part: str,
    category: str | None = None,
    allow_test: bool = False,
) -> list[ManifestRow]:
    """Rows of one part of a protocol, in manifest order. The test eval parts need `allow_test=True`."""
    folds = pool_folds(protocol)  # also validates the protocol
    if part not in PARTS:
        raise ValueError(f"unknown part: {part!r} (expected one of {PARTS})")
    if part == "pool_normal":
        role, wanted_folds = "pool_normal", folds
    elif protocol == "dev":
        role, wanted_folds = ("pool_normal", (0,)) if part == "eval_normal" else ("dev_defect", (-1,))
    else:
        if allow_test is not True:  # not truthiness: "false", "0" or 1 must not open the seal
            raise SealedTestError(
                f"{part} of the test protocol is the sealed test set; pass allow_test=True (--allow-test)"
            )
        role, wanted_folds = ("test_normal" if part == "eval_normal" else "test_defect"), (-1,)
    return [
        r
        for r in manifest
        if r.role == role and r.fold in wanted_folds and (category is None or r.category == category)
    ]


def role_counts(manifest: list[ManifestRow]) -> dict[str, dict[str, int]]:
    """category -> role -> number of images (categories in manifest order)."""
    counts: dict[str, dict[str, int]] = {}
    for r in manifest:
        per_role = counts.setdefault(r.category, dict.fromkeys(ROLES, 0))
        per_role[r.role] += 1
    return counts


def format_counts(manifest: list[ManifestRow]) -> str:
    """Per-category table of role counts plus the size of the dev evaluation normals (fold 0)."""
    counts = role_counts(manifest)
    fold0 = Counter(r.category for r in manifest if r.role == "pool_normal" and r.fold == 0)
    columns = [*ROLES, "fold0_normal"]
    lines = [f"{'category':<12}" + "".join(f"{c:>13}" for c in columns)]
    for category, per_role in counts.items():
        values = [per_role[role] for role in ROLES] + [fold0[category]]
        lines.append(f"{category:<12}" + "".join(f"{v:>13,}" for v in values))
    totals = [sum(per_role[role] for per_role in counts.values()) for role in ROLES] + [sum(fold0.values())]
    lines.append(f"{'total':<12}" + "".join(f"{v:>13,}" for v in totals))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from defect_inspect import paths
    from defect_inspect.visa import read_defect_types, read_split_csv

    parser = argparse.ArgumentParser(description="Build the VisA split manifest from the tar.")
    parser.add_argument("--tar", type=Path, default=paths.VISA_TAR, help="VisA tar file")
    parser.add_argument("--out", type=Path, default=paths.VISA_MANIFEST, help="manifest CSV to write")
    parser.add_argument(
        "--no-verify", action="store_true", help="skip the check against the official VisA counts"
    )
    args = parser.parse_args(argv)

    rows = read_split_csv(args.tar, "2cls_highshot")
    manifest = build_manifest(rows, read_defect_types(args.tar))
    print(format_counts(manifest))

    if not args.no_verify:
        totals = dict.fromkeys(ROLES, 0)
        for r in manifest:
            totals[r.role] += 1
        untyped = [r.image for r in manifest if r.label == "anomaly" and not r.defect_types]
        categories = tuple(role_counts(manifest))
        if totals != EXPECTED_TOTALS or untyped or categories != CATEGORIES:
            print(f"ERROR: manifest does not match the official split, nothing written. totals={totals}")
            print(f"  categories={categories}, anomalies without a defect type: {len(untyped)}")
            return 1

    write_manifest(manifest, args.out)
    print(f"wrote {len(manifest):,} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
