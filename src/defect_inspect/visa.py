"""Read the VisA split and annotation CSVs straight from the tar (no extraction, no image decoding)."""

# [파일 역할] VisA 원본 tar(VisA_20220922.tar) 안에서 공식 분할 CSV(split_csv/*.csv)와 범주별 결함 유형
# CSV(<범주>/image_anno.csv)만 꺼내 파이썬 객체로 돌려준다.
# [왜 필요한가] 데이터 디스크가 HDD라 tar를 작은 파일 약 1만 2천 개로 풀어 두고 매번 읽으면 느리다. 그래서
# tar를 풀지 않고 CSV 멤버의 텍스트만 읽는다(이미지는 cache.py가 tar를 한 번 순서대로 읽어 리사이즈 캐시로
# 만든다). 이미지는 여기서 디코딩하지 않고 경로 문자열만 넘기며, 어느 이미지가 어느 역할(정상 풀, dev 결함,
# 봉인 테스트 등)인지는 splits.py가 정한다.
# [흐름] _scan_csv_members가 tar 헤더를 한 번 훑어 필요한 CSV 텍스트를 모은다(프로세스당 한 번, 캐시) →
# read_split_csv가 분할 CSV를 검증하며 VisaRow 목록으로, read_defect_types가 결함 유형 표로 바꾼다.
# [연결] splits.main이 두 읽기 함수를 불러 manifests/visa.csv를 만든다. CATEGORIES는
# run_patchcore·run_perturb·analyze_perturb 등 여러 모듈이 범주 목록의 기준으로 가져다 쓴다.

# 표준 라이브러리만 쓴다(기본 패키지만으로 도는 코드). tarfile은 풀지 않고 멤버를 읽고, io·csv는 메모리 안의
# 텍스트를 CSV로 파싱하며, functools.lru_cache는 tar 훑기 결과를 캐시한다.
import csv
import functools
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path

# VisA의 12개 범주(알파벳 순). 이 튜플 순서가 프로젝트 전체의 범주 순서 기준이다. splits._category_order가
# 매니페스트 행을 이 순서로 놓고, splits.main은 매니페스트의 범주 순서가 이것과 같은지 검사한다.
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
# tar의 split_csv 폴더에 있는 공식 분할 세 가지. 이 프로젝트는 2cls_highshot만 분할의 출발점으로
# 쓴다(docs/experiments.md 공통 규칙). 1cls의 공개 수치는 봉인 테스트와 겹쳐 재현하지 않는다(같은 문서의 봉인
# 테스트 절).
# SPLIT_HEADER, ANNO_HEADER: 기대하는 열 이름. _rows가 실제 헤더와 비교해 다르면 멈춘다. 데이터 판이 바뀌어
# 열이 달라졌을 때 조용히 엉뚱한 칸을 읽는 것을 막는다.
SPLIT_NAMES = ("1cls", "2cls_fewshot", "2cls_highshot")
SPLIT_HEADER = ["object", "split", "label", "image", "mask"]
ANNO_HEADER = ["image", "label", "mask"]


# 분할 CSV 한 행. frozen=True라 만든 뒤 바꿀 수 없고 해시할 수 있다. image·mask는 디스크 경로가 아니라 tar
# 안의 멤버 이름이다(캐시를 만들 때 이 이름으로 tar에서 꺼낸다).
@dataclass(frozen=True)
class VisaRow:
    category: str
    split: str  # "train" | "test"
    label: str  # "normal" | "anomaly"
    image: str  # member name of the image, e.g. candle/Data/Images/Normal/0836.JPG
    mask: str  # member name of the mask, "" for normal images


# tar 멤버 이름 앞의 "./"를 떼어 CSV에 적힌 경로(예: split_csv/2cls_highshot.csv)와 같은 모양으로 맞춘다.
# tar를 만든 방식에 따라 "./"가 붙을 수 있다.
def _member_name(name: str) -> str:
    return name[2:] if name.startswith("./") else name


# tar 안의 CSV 멤버 이름 → 텍스트 표를 만든다. 인자가 Path가 아니라 (문자열 경로, 크기, 수정 시각)인 이유는
# lru_cache의 키로 쓰기 위해서다. 같은 파일이면 캐시를 다시 쓰고, 파일이 바뀌어 크기나 mtime이 달라지면 새로
# 훑는다. 최근 tar 4개까지 기억한다.
# 호출 위치: _csv_members에서만 부른다.
@functools.lru_cache(maxsize=4)
def _scan_csv_members(tar_path: str, size: int, mtime_ns: int) -> dict[str, str]:
    """Text of every CSV member, from one pass over the tar headers.

    The split CSVs sit at the very end of the real tar, so a pass is a full scan of ~12,000 headers.
    The result is cached per (path, size, mtime) so that one process scans the tar only once.
    """
    # 찾아야 할 CSV: 분할 CSV 3개와 범주별 image_anno.csv 12개. 이것을 다 모으면 남은 헤더를 보지 않고 멈춘다.
    wanted = {f"split_csv/{name}.csv" for name in SPLIT_NAMES} | {f"{c}/image_anno.csv" for c in CATEGORIES}
    found: dict[str, str] = {}
    # "r:"는 압축 없는 tar로 연다는 뜻이다. 멤버를 순서대로 훑고, CSV가 아닌 멤버(이미지와 마스크)는 내용을
    # 읽지 않고 헤더만 지나간다.
    with tarfile.open(tar_path, "r:") as tar:
        for member in tar:
            name = _member_name(member.name)
            if not member.isfile() or not name.lower().endswith(".csv"):
                continue
            # utf-8-sig로 디코딩해 파일 앞에 BOM이 있어도 첫 열 이름이 "\ufeffobject"처럼 깨지지 않게 한다.
            found[name] = tar.extractfile(member).read().decode("utf-8-sig")
            if wanted <= found.keys():
                break
    return found


# Path를 받아 stat로 크기·mtime을 구하고, resolve로 같은 파일의 다른 경로 표기를 하나로 맞춘 뒤 캐시된 훑기를
# 부른다.
def _csv_members(tar_path: Path) -> dict[str, str]:
    stat = Path(tar_path).stat()
    return _scan_csv_members(str(Path(tar_path).resolve()), stat.st_size, stat.st_mtime_ns)


# CSV 텍스트를 dict 행 목록으로 바꾼다. newline=""는 csv 모듈이 줄 끝(\r\n, \n)을 직접 다루게 하는 권장
# 설정이다. 헤더가 기대와 다르면 member 이름을 넣은 ValueError로 멈춘다.
def _rows(text: str, header: list[str], member: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != header:
        raise ValueError(f"{member}: expected header {header}, got {reader.fieldnames}")
    return list(reader)


# 공식 분할 CSV를 VisaRow 목록으로 읽는다(파일 순서 그대로). 행마다 세 가지를 검사한다: 열 수, split·label 값,
# 이상 행과 마스크의 짝.
# 왜 이렇게 엄격한가: 이 행들이 봉인 테스트의 경계(공식 test = 봉인 테스트)를 정하므로, 잘못 읽힌 행이 조용히
# 다른 역할로 섞이면 안 된다.
# 호출 위치: splits.main(2cls_highshot), tests/test_visa.py.
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
        # split은 train/test, label은 normal/anomaly만 허용한다. 그 밖의 값은 분할 규칙이 다루지 않으므로
        # 멈춘다.
        if row.split not in ("train", "test") or row.label not in ("normal", "anomaly"):
            raise ValueError(f"{member}: unexpected split/label in {r}")
        # 이상 행에는 픽셀 마스크가 반드시 있고 정상 행에는 없어야 한다. 마스크 유무로 라벨을 한 번 더 교차
        # 확인한다.
        if (row.label == "anomaly") != bool(row.mask):
            raise ValueError(f"{member}: anomaly rows need a mask and normal rows must not have one: {r}")
        rows.append(row)
    return rows


# 이상 이미지 경로 → 결함 유형 이름 튜플. 2단계에서 "학습에서 보지 못한 결함 유형"만 따로 모아 지도 학습과
# 비지도 방법을 비교하는 데 쓴다(docs/experiments.md 0단계, 39종).
# 호출 위치: splits.main이 매니페스트의 defect_types 열을 채울 때 부른다.
def read_defect_types(tar_path: Path) -> dict[str, tuple[str, ...]]:
    """Image path -> sorted, distinct defect type names, for anomaly images (`<category>/image_anno.csv`)."""
    types: dict[str, tuple[str, ...]] = {}
    for member, text in _csv_members(tar_path).items():
        # 범주 폴더 바로 아래의 image_anno.csv만 읽는다(분할 CSV나 더 깊은 경로의 CSV는 건너뜀).
        if member.count("/") != 1 or not member.endswith("/image_anno.csv"):
            continue
        for r in _rows(text, ANNO_HEADER, member):
            # 정상 행은 유형이 없으므로 건너뛴다. 이상 행의 label 칸은 "weird candle wick, chunk of wax
            # missing"처럼 쉼표로 유형 여러 개를 담을 수 있어, 쉼표로 나누고 앞뒤 공백과 빈 이름을 버린 뒤
            # 정렬한다(같은 이미지면 늘 같은 튜플).
            if r["label"].strip() == "normal":
                continue
            names = {name.strip() for name in r["label"].split(",")} - {""}
            types[r["image"]] = tuple(sorted(names))
    return types
