"""Locations of data, caches, outputs and committed artefacts."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Downloads and caches can live on another disk: set DEFECT_INSPECT_DATA to move them.
DATA = Path(os.environ.get("DEFECT_INSPECT_DATA") or ROOT / "data")
RAW = DATA / "raw"
CACHE = DATA / "cache"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
MANIFESTS = ROOT / "manifests"
VISA_TAR = RAW / "VisA_20220922.tar"
VISA_MANIFEST = MANIFESTS / "visa.csv"
TEST_LEDGER = REPORTS / "test_ledger.jsonl"
