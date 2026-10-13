"""Locations of data, caches, outputs and committed artefacts."""

# [파일 역할] 데이터·캐시·실행 결과·커밋하는 리포트의 위치를 한곳에 모아 둔 경로 상수 모듈이다.
# [왜 필요한가] 다운로드(download), 캐시 생성(cache, m2ad), 실행 스크립트(run_*), 분석(analyze_*),
# 그림(figures)이 모두 같은 폴더 규칙을 써야 하므로 경로를 여기서만 정한다. 폴더 규칙은 내려받은 것과
# 캐시는 data/, 실행 결과는 outputs/(둘 다 .gitignore에 있어 커밋하지 않는다), 커밋하는 리포트는 reports/,
# 분할 매니페스트는 manifests/이다.
# [연결] 거의 모든 run_*·analyze_* 모듈이 `from . import paths`로 가져와 기본 인자값으로 쓴다.

import os
from pathlib import Path

# ROOT: 저장소 루트. 이 파일이 src/defect_inspect/paths.py에 있으므로 두 단계 위(parents[2])가 루트다. 설치
# 위치가 아니라 소스 위치 기준이라 uv의 편집 가능 설치에서도 저장소 폴더를 가리킨다.
ROOT = Path(__file__).resolve().parents[2]
# Downloads and caches can live on another disk: set DEFECT_INSPECT_DATA to move them.
# 환경 변수가 비어 있으면(빈 문자열 포함) 저장소 안 data/를 쓴다. `or`로 처리해 빈 값도 기본값으로 떨어진다.
DATA = Path(os.environ.get("DEFECT_INSPECT_DATA") or ROOT / "data")
# RAW: 내려받은 원본(VisA tar, M2AD zip), CACHE: 한 번 읽어 만든 리사이즈 배열 캐시. 데이터 디스크가 HDD라
# tar를 풀어 두고 매번 읽지 않고 큰 배열 파일로 한 번 옮겨 둔다(cache.py와 m2ad.build_cache의 docstring).
# OUTPUTS(실행 결과·특징·모델)는 커밋하지 않고, REPORTS(요약 리포트)와 MANIFESTS(분할 매니페스트)는 커밋한다.
RAW = DATA / "raw"
CACHE = DATA / "cache"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
MANIFESTS = ROOT / "manifests"
# VISA_TAR: 공식 S3에서 받는 VisA tar 하나(1.93GB, download.py가 sha256으로 검증). VISA_MANIFEST: splits.py가
# 만드는 결정적 분할 매니페스트(10,821행, 커밋됨).
# TEST_LEDGER: 봉인 테스트를 읽을 때마다 한 줄씩 남기는 장부(ledger.record_test_access가 씀). 저장소에
# 커밋되어 언제 무엇이 봉인 테스트를 봤는지 공개 기록으로 남는다.
VISA_TAR = RAW / "VisA_20220922.tar"
VISA_MANIFEST = MANIFESTS / "visa.csv"
TEST_LEDGER = REPORTS / "test_ledger.jsonl"
