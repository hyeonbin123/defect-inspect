"""Deterministic split manifest for VisA (rules in docs/experiments.md) and the sealed-test gate."""

# [파일 역할] VisA 공식 2cls_highshot 분할의 모든 이미지에 역할(정상 풀과 겹 번호, dev 결함, 라벨 풀, 봉인
# 테스트 정상·결함)을 정해 매니페스트(manifests/visa.csv)로 남기고, 실행 코드가 "절차(dev/test)의 어느 부분"을
# 달라고 하면 그 행만 골라 준다. 봉인 테스트를 여는 관문(select의 allow_test)도 여기 있다.
# [설계 이유] 분할은 결정적이어야 한다는 원칙에 따라 난수 대신 이미지 경로의 SHA-256 순서로만 나눈다.
# 그래서 언제 어디서 다시 만들어도 같은 매니페스트가 나오고, 그 결과를 저장소에 커밋해 둔다. 분할 규칙의
# 원문은 docs/experiments.md "분할" 절이다.
# [주요 구성] path_key(정렬 키) → build_manifest(역할 배정) → write/read_manifest(CSV 입출력) →
# label_subset(지도 학습의 k장), pool_folds·holdout_fold(절차별 겹), select(절차·부분별 행 선택과 봉인 관문) →
# main(tar에서 매니페스트 생성과 공식 장수 검증).
# [연결] run_patchcore·run_grid·run_supervised·run_dinomaly·export 등 거의 모든 실행 모듈이 read_manifest와
# select로 이미지를 받는다. path_key는 m2ad·drift·run_m2ad_enrol도 결정적 순서·시드로 쓴다.

import argparse
import csv
import hashlib
from collections import Counter
from dataclasses import astuple, dataclass, fields
from pathlib import Path

# visa.py에서 범주 순서 기준(CATEGORIES)과 분할 CSV 한 행의 형식(VisaRow)을 가져온다.
from defect_inspect.visa import CATEGORIES, VisaRow

# 분할 규칙의 상수(docs/experiments.md 분할 절).
# - N_FOLDS: 학습 풀 정상을 나누는 겹 수. 교차 적합 보정과 dev 평가(겹 0)에 쓴다
# - N_DEV_DEFECTS: 범주마다 학습 풀 결함 60장 중 dev 평가용으로 떼는 장수. 나머지 40장이 라벨 풀이다
# - K_VALUES, SEEDS: 지도 학습이 쓰는 결함 라벨 장수 k와 시드(run_supervised의 기본값)
# - ROLES: 매니페스트의 역할 다섯 가지, SEALED_ROLES: 그중 봉인 테스트(공식 test 4,328장)
# - PROTOCOLS: dev(코드 점검, 몇 번이든)와 test(단계 끝에 한 번), PARTS: select가 돌려주는 부분 세 가지
N_FOLDS = 5
N_DEV_DEFECTS = 20
K_VALUES = (5, 10, 20, 40)
SEEDS = (0, 1, 2)
ROLES = ("pool_normal", "dev_defect", "label_pool", "test_normal", "test_defect")
SEALED_ROLES = ("test_normal", "test_defect")  # the sealed test set
PROTOCOLS = ("dev", "test")
PARTS = ("pool_normal", "eval_normal", "eval_defect")
# Totals of the official 2cls_highshot split under these rules (docs/experiments.md).
# main이 만든 매니페스트를 이 장수와 비교해 하나라도 다르면 쓰지 않는다(0단계 분할 표와 같은 값).
EXPECTED_TOTALS = {
    "pool_normal": 5773,
    "dev_defect": 240,
    "label_pool": 480,
    "test_normal": 3848,
    "test_defect": 480,
}


# 봉인 테스트를 허락 없이 읽으려 할 때 내는 전용 예외. 일반 ValueError와 구분해 잡을 수 있게 따로
# 둔다(figures.py가 ExamplesError와 함께 잡고, run_m2ad·run_m2ad_enrol도 같은 예외로 M2AD 테스트를 막는다).
class SealedTestError(RuntimeError):
    """Raised when the sealed test set is requested without `allow_test=True`."""


# 결정적 정렬 키: 경로(또는 "시드:경로")의 SHA-256 16진수. 난수 생성기 대신 이 키로 섞으면 실행 환경이나
# 순서와 무관하게 같은 결과가 나온다.
# salt를 문자열로만 받는 이유는 docstring대로 정수 0이 "salt 없음"과 섞이지 않게 하려는 것이다. 타입이 틀리면
# TypeError로 바로 멈춘다.
# 호출 위치: 이 파일(build_manifest, label_subset), m2ad.py, drift.py(시드 생성), run_m2ad_enrol.py.
def path_key(path: str, salt: str = "") -> str:
    """Sort key: SHA-256 hex of the path, or of "<salt>:<path>" when a salt is given.

    The salt must be a string (`str(seed)`): an integer 0 would otherwise read as "no salt".
    """
    if not isinstance(path, str) or not isinstance(salt, str):
        raise TypeError(f"path and salt must be str, got {type(path).__name__} and {type(salt).__name__}")
    text = path if salt == "" else f"{salt}:{path}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# 매니페스트 한 행. 필드 순서가 곧 CSV 열 순서다(MANIFEST_HEADER). fold는 정상 풀만 0~4이고 나머지는 -1,
# defect_types는 결함 유형 이름을 "|"로 이은 문자열이다(CSV 한 칸에 넣기 위해).
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


# 범주 정렬 키. VisA 범주는 CATEGORIES 순서로, 그 밖의 이름(테스트용 합성 데이터)은 그 뒤에 이름 순으로
# 놓는다.
def _category_order(category: str) -> tuple[int, str]:
    # Known categories in CATEGORIES order; anything else (synthetic data) after them, by name.
    return (CATEGORIES.index(category), "") if category in CATEGORIES else (len(CATEGORIES), category)


# 공식 분할 행 전체에 역할을 배정한다. 입력은 visa.read_split_csv의 행과 visa.read_defect_types의 결함 유형
# 표, 출력은 범주 순 → 역할 순 → 경로 순으로 정렬된 ManifestRow 목록이다.
# 호출 위치: main, tests/test_splits.py.
def build_manifest(rows: list[VisaRow], defect_types: dict[str, tuple[str, ...]]) -> list[ManifestRow]:
    """Assign a role (and a fold for pool normals) to every row of the official 2cls_highshot split."""
    # 같은 이미지가 두 번 나오면 두 역할에 동시에 들어갈 수 있어(예: 학습과 봉인 테스트) 처음부터 거절한다.
    images = Counter(r.image for r in rows)
    repeated = sorted(image for image, n in images.items() if n > 1)
    if repeated:
        raise ValueError(f"image listed more than once: {repeated[:3]}")

    manifest: list[ManifestRow] = []
    for category in sorted({r.category for r in rows}, key=_category_order):
        # 범주 하나의 행을 (split, label) 네 묶음으로 나눈다. 예상 밖 값은 여기서도 멈춘다.
        by_group: dict[tuple[str, str], list[VisaRow]] = {}
        for r in rows:
            if r.category == category:
                if r.split not in ("train", "test") or r.label not in ("normal", "anomaly"):
                    raise ValueError(f"unexpected split/label: {r}")
                by_group.setdefault((r.split, r.label), []).append(r)

        assigned: list[tuple[VisaRow, str, int]] = []
        # 학습 풀 정상: SHA-256 키 순으로 줄 세운 순번을 5로 나눈 나머지가 겹 번호다. 순번을 5로 돌려 나누므로
        # 겹 크기가 거의 같고, 해시 순서라 파일 번호 순으로 한 겹에 몰리지 않는다.
        pool = sorted(by_group.get(("train", "normal"), []), key=lambda r: path_key(r.image))
        assigned += [(r, "pool_normal", rank % N_FOLDS) for rank, r in enumerate(pool)]
        # 학습 풀 결함: 같은 키 순으로 앞 20장은 dev 결함(구성 확인·선택용), 나머지는 라벨 풀(지도 학습의
        # k장만 여기서 뽑음)이다. 둘을 나눠야 지도 학습의 검증 세트와 학습 세트가 겹치지 않는다.
        defects = sorted(by_group.get(("train", "anomaly"), []), key=lambda r: path_key(r.image))
        assigned += [(r, "dev_defect", -1) for r in defects[:N_DEV_DEFECTS]]
        assigned += [(r, "label_pool", -1) for r in defects[N_DEV_DEFECTS:]]
        # 공식 test는 그대로 봉인 테스트 정상·결함이 된다.
        assigned += [(r, "test_normal", -1) for r in by_group.get(("test", "normal"), [])]
        assigned += [(r, "test_defect", -1) for r in by_group.get(("test", "anomaly"), [])]

        # 출력 순서를 역할 → 경로로 고정해, 매니페스트 CSV가 다시 만들어도 줄 단위로 똑같게 한다(커밋된 파일과
        # diff가 생기지 않게).
        assigned.sort(key=lambda item: (ROLES.index(item[1]), item[0].image))
        for r, role, fold in assigned:
            types = "|".join(defect_types.get(r.image, ())) if r.label == "anomaly" else ""
            manifest.append(ManifestRow(r.image, r.mask, category, r.label, role, fold, types))
    return manifest


# 매니페스트를 CSV로 쓴다. newline="\n"과 lineterminator로 OS와 관계없이 LF로 써서, Windows에서 다시 만들어도
# 커밋된 manifests/visa.csv와 줄 끝 차이가 생기지 않게 한다.
def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    """Write the manifest as UTF-8 CSV with LF line endings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(MANIFEST_HEADER)
        writer.writerows(astuple(r) for r in rows)


# 커밋된 매니페스트 CSV를 ManifestRow 목록으로 읽는다. 헤더가 다르면 멈추고, fold는 int로 되돌린다.
# 호출 위치: run_patchcore, run_grid, run_supervised, export, bench, cache, dinomaly_serving, figures 등
# 이미지를 고르는 모든 실행 모듈.
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


# 지도 학습에 줄 결함 k장을 범주의 라벨 풀에서 고른다. 시드마다 "시드:경로"의 SHA-256 순서로 줄 세우고 앞에서
# k장을 자르므로, 같은 시드에서 k=5 집합은 k=10 집합에 포함된다(포함 관계, docs/experiments.md 분할 절). k를
# 늘릴 때 앞의 표본은 그대로 두고 더하기만 하므로, k 사이 비교에 표본 교체의 영향이 섞이지 않는다.
# 호출 위치: run_supervised.py.
def label_subset(manifest: list[ManifestRow], category: str, k: int, seed: int) -> list[ManifestRow]:
    """First k images of the category's label pool in the order of seed `seed` (nested across k)."""
    pool = [r for r in manifest if r.category == category and r.role == "label_pool"]
    if not 0 <= k <= len(pool):
        raise ValueError(f"k={k} is outside the label pool of {category} ({len(pool)} images)")
    pool.sort(key=lambda r: path_key(r.image, salt=str(seed)))
    return pool[:k]


# 절차별 정상 풀의 겹: dev는 겹 0을 평가용으로 빼고 겹 1~4, test는 겹 0~4 전부다. 알 수 없는 절차 이름은
# ValueError로 막는다.
def pool_folds(protocol: str) -> tuple[int, ...]:
    """Folds whose normals form the normal pool: dev keeps fold 0 for evaluation."""
    if protocol == "dev":
        return tuple(range(1, N_FOLDS))
    if protocol == "test":
        return tuple(range(N_FOLDS))
    raise ValueError(f"unknown protocol: {protocol!r} (expected one of {PROTOCOLS})")


# 홀드아웃 보정에서 뱅크에서 빼고 임계값을 정할 때만 쓰는 겹. 풀의 첫 겹이라 test 절차에서는 겹 0, dev
# 절차에서는 겹 1이다(docs/experiments.md 1단계).
# 호출 위치: run_patchcore.py(그 겹을 뺀 뱅크), analyze.py(그 겹의 점수로 임계값).
def holdout_fold(protocol: str) -> int:
    """Fold left out of the bank by the hold-out calibration: the first fold of the pool."""
    return pool_folds(protocol)[0]


# 절차(dev/test)와 부분(pool_normal/eval_normal/eval_defect)으로 매니페스트에서 행을 고른다. 모든 실행 코드가
# 이미지를 이 함수로만 받으므로, 봉인 테스트를 여는 관문이 한 곳에 모인다.
# - pool_normal: 절차의 정상 풀(겹 목록은 pool_folds)
# - dev의 eval_normal/eval_defect: 겹 0 정상 / dev 결함
# - test의 eval_normal/eval_defect: 봉인 테스트 정상·결함. allow_test=True가 아니면 SealedTestError
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
        # 참·거짓 판정이 아니라 True 객체인지 비교한다. 설정 파일이나 인자에서 문자열 "false", "0", 정수 1
        # 같은 값이 넘어와도 봉인이 열리지 않게 하려는 것이다.
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


# 범주 → 역할 → 장수 표. format_counts의 출력과 main의 범주 순서 검사에 쓴다.
def role_counts(manifest: list[ManifestRow]) -> dict[str, dict[str, int]]:
    """category -> role -> number of images (categories in manifest order)."""
    counts: dict[str, dict[str, int]] = {}
    for r in manifest:
        per_role = counts.setdefault(r.category, dict.fromkeys(ROLES, 0))
        per_role[r.role] += 1
    return counts


# main이 출력하는 범주별 장수 표. dev 평가 정상(겹 0) 장수를 열 하나로 더 붙인다.
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


# 명령행 진입점(python -m defect_inspect.splits). tar에서 분할 CSV와 결함 유형을 읽어 매니페스트를 만들고,
# 공식 장수와 맞는지 확인한 뒤 쓴다. paths와 visa 읽기 함수는 이 함수 안에서만 필요해서 함수 안에서
# import한다.
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

    # 검증 세 가지: 역할별 합계가 EXPECTED_TOTALS와 같은지, 결함 유형이 비어 있는 이상 이미지가 없는지(2단계
    # "보지 못한 유형" 분석에 필요), 범주 순서가 CATEGORIES와 같은지. 하나라도 틀리면 파일을 쓰지 않고 1을
    # 돌려준다. --no-verify를 주면 이 검사를 건너뛴다(tests/test_splits.py의 작은 합성 tar가 이렇게 돈다).
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
