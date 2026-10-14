"""M2AD(Motor와 Bird, 시점 3개, 조명 10가지): 메타데이터, 시편 단위 겹 나누기, 리사이즈 캐시.

규칙은 docs/experiments.md 3-B단계에 등록했다. 캐시 만들기는 zip에서 M2AD 이미지를 읽는 유일한 코드다.
테스트 이미지도 옮기므로 `main`이 먼저 테스트 장부에 기록한다.
"""

# [파일 역할] M2AD 데이터(실제로 조명을 바꿔 찍은 이상 탐지 데이터)의 메타데이터 읽기, 라벨 규칙, 시편 단위 겹
# 나누기, 리사이즈 캐시 만들기·읽기를 맡는다.
# [왜 필요한가] 질문 3('합성 교란이 실제 조명 변화를 대신하는가', 3-B)과 7단계(조명 등록·변화 감지)는 같은
# 시편을 여러 조명에서 찍은 데이터가 있어야 잴 수 있다. M2AD는 범주마다 zip이 수십 GB라(Motor 13.9GB, Bird
# 14.4GB, experiments.md 3-B) HDD에서 매번 읽지 않도록, 필요한 이미지만 zip 안의 순서대로 한 번 읽어 큰 배열
# 캐시로 옮긴다(build_cache docstring).
# [설계 이유] 독립 표본은 이미지가 아니라 시편(물리적 물체 하나)이다. 같은 시편의 시점·조명 이미지는 서로
# 비슷하므로 겹 나누기·재보정 표본·부트스트랩을 모두 시편 단위로 한다. 시편 순서는 VisA 분할과 같은 SHA-256
# 순서로 정해 결정적이다(docs/experiments.md 3-B: 시편 이름의 SHA-256 순).
# [흐름] read_meta(jsons.zip의 meta_unsupervised.json → 행 목록, 라벨 규칙 적용, 일관성 점검) →
# specimen_folds·recal_specimens·illumination_groups(결정적 시편·조명 나누기) → build_cache(zip을 파일
# 순서대로 읽어 디코딩·리사이즈 → .npy) → M2adCache(읽기).
# [연결] run_m2ad(3-B), run_m2ad_enrol(7단계), analyze_m2ad_enrol, figures가 이 모듈의 상수·함수를 가져다
# 쓴다. 캐시 쓰기 도구(_RowWriter, _publish 등)는 VisA 캐시(cache.py)의 것을 그대로 쓴다.

import argparse
import json
import time
import zipfile
from collections import Counter, deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# cache: VisA 리사이즈 캐시의 쓰기·읽기 도구(같은 BICUBIC 리사이즈, .part 파일 후 교체, 메모리 맵 읽기)를
# 재사용한다. path_key: 문자열의 SHA-256 정렬 키(VisA 분할과 같은 결정적 순서).
import numpy as np

from . import cache as _cache
from .splits import path_key

# 실험에 쓰는 M2AD 범주 둘. 내려받는 파일이 가장 작은 두 범주로 골랐고 성능을 보고 고르지
# 않았다(experiments.md 3-B). 순서가 정렬 순서다(Motor 먼저).
CATEGORIES = ("Motor", "Bird")
# 같은 간격의 세 시점. (범주, 시점)마다 검사기를 하나씩 만들어 모두 6개다.
VIEWS = ("000", "120", "240")
# 열 가지 조명 I01~I10. REFERENCE(I01)는 뱅크를 만드는 기준 조명이고 나머지 아홉이 '실제 조명 변화' 조건이다.
# I01~I10이 물리적으로 어떤 조명인지는 확인하지 못했다(experiments.md 3-B 한계).
ILLUMINATIONS = ("01", "02", "03", "04", "05", "06", "07", "08", "09", "10")
REFERENCE = "01"
# N_FOLDS: 학습 시편 30개를 시편 단위 5겹으로 나눠 교차 적합 임계값을 정한다. RECAL_SIZES: 새 조명 재보정에
# 쓰는 학습 시편 수(8, 30). 3단계 계획의 32장을 M2AD 학습 시편이 범주마다 30개뿐이라 30으로 고쳤다(규칙 변경
# 3).
N_FOLDS = 5
RECAL_SIZES = (8, 30)
# M2AD에는 공식 검증 분할이 없고 train(정상 시편만)과 test만 있다. META_MEMBER: jsons.zip 안에서 읽는 비지도
# 설정 메타데이터 파일 이름.
SPLITS = ("train", "test")
META_MEMBER = "meta_unsupervised.json"


# M2AD 이미지 한 장의 메타데이터 행. specimen(object_name)은 물리적 물체 하나이고, object_anomaly는 그 시편이
# 불량인지, label은 아래 label_of 규칙으로 정한 이 이미지의 평가 라벨이다(-1은 평가에서 뺌). frozen이라 해시
# 가능하고 실수로 바뀌지 않는다.
@dataclass(frozen=True)
class M2adRow:
    category: str
    split: str  # "train" | "test"
    specimen: str  # object_name: 물리적 물체 하나
    view: str
    illumination: str
    img_path: str  # <범주>.zip 안의 멤버 이름
    object_anomaly: int  # 시편이 불량이면 1
    label: int  # 0 정상, 1 결함, -1 평가에서 뺌


# 결함이 보인다는 것은 image_anomaly = 1, 판별 가능은 detectable이 True라는 뜻이다. 나머지는 그 시점에서
# 결함이 안 보이거나 판별 불가로 표시된 것이다. 결함이 안 보이는 사진을 '결함'으로 세면 어떤 검사기도 잡을 수
# 없는 표본이 검출률을 깎으므로 뺀다.
def label_of(object_anomaly: int, image_anomaly: int, detectable: object) -> int:
    """등록한 라벨 규칙(experiments.md 3-B): 정상 시편은 0, 보이고 판별 가능한 결함은 1, 나머지는 -1."""
    if object_anomaly == 0:
        return 0
    if image_anomaly == 1 and detectable is True:
        return 1
    return -1


# 메타데이터 항목 하나를 M2adRow로 바꾸면서 값이 규칙대로인지 확인한다: 조명 이름이 알려진 것인지, 0/1
# 플래그가 정말 0이나 1인지(bool은 거절), detectable이 빈 문자열이나 bool인지, 정상 시편인데 image_anomaly가
# 1인 모순은 없는지. 이상한 메타데이터를 조용히 다른 라벨로 바꾸지 않고 멈춘다.
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


# 행 정렬 키: 범주·분할은 선언 순서, 그 안에서 시편·시점·조명 순. 캐시 배열의 행 순서가 이 순서로 고정돼 다시
# 만들어도 같다.
def _sort_key(row: M2adRow) -> tuple:
    # 범주와 분할은 선언 순서(Motor 다음 Bird, train 다음 test).
    return (CATEGORIES.index(row.category), SPLITS.index(row.split), row.specimen, row.view, row.illumination)


# 메타데이터 전체의 일관성 점검: 같은 (범주, 분할, 시편, 시점, 조명)이나 같은 파일 경로가 두 번 나오지 않는지,
# train 행이 모두 정상 시편인지, 한 시편이 두 분할에 걸치거나 불량 여부가 바뀌지 않는지. 시편 단위로 겹을
# 나누므로 시편이 train과 test에 함께 있으면 평가가 새어 나간다.
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


# 점검까지 마친 행을 돌려준다. 실험에 쓰지 않는 시점은 여기서 걸러진다.
# 호출 위치: main(캐시 생성), run_m2ad, run_m2ad_enrol.
def read_meta(jsons_zip: Path) -> list[M2adRow]:
    """`jsons_zip` 안의 `meta_unsupervised.json`에서 CATEGORIES x VIEWS의 행(조명 10가지 모두)을 읽는다.

    (범주, 분할, 시편, 시점, 조명) 순으로 정렬하고, 범주와 분할은 선언 순서를 따른다.
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


# 이름에 숨은 순서(번호 등)와 무관한 결정적 순서를 얻으려는 것으로, VisA 분할(splits.py)과 같은 방식이다.
def _ranked(specimens: Sequence[str]) -> list[str]:
    """중복 없는 시편 이름을 이름의 SHA-256 16진수 순으로 늘어놓는다."""
    names = set(specimens)
    if not all(isinstance(name, str) for name in names):
        raise TypeError("specimen names must be str")
    return sorted(names, key=path_key)


# 겹 번호는 0~4다. 교차 적합 임계값을 시편 단위로 나눠, 한 시편의 여러 시점·조명 이미지가 뱅크와 보정 점수에
# 동시에 들어가지 않게 한다(experiments.md 3-B '임계값').
def specimen_folds(specimens: Sequence[str]) -> dict[str, int]:
    """시편 -> 겹 번호: SHA-256 순위를 N_FOLDS로 나눈 나머지(`splits`와 같은 방식)."""
    return {name: rank % N_FOLDS for rank, name in enumerate(_ranked(specimens))}


# 앞에서 자르므로 n = 8의 시편은 n = 30에 모두 포함된다(포함 관계).
def recal_specimens(specimens: Sequence[str], n: int) -> list[str]:
    """새 조명 재보정에 쓸 학습 시편: SHA-256 순서의 앞 `n`개(n끼리 포함 관계)."""
    ordered = _ranked(specimens)
    if not 0 <= n <= len(ordered):
        raise ValueError(f"n={n} is outside the {len(ordered)} specimens")
    return ordered[:n]


# 결과는 A = 03, 07, 08, 09, B = 02, 04, 05, 06, 10(experiments.md 7단계 '조명 묶음'). 한 묶음을 I01과 함께
# 등록하고 다른 묶음을 '못 본 조명'으로 잰 뒤 서로 바꾸므로 아홉 조명이 모두 한 번씩 못 본 조명이 된다.
# 호출 위치: run_m2ad_enrol.
def illumination_groups() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """7단계 조명 묶음: 기준(I01)이 아닌 아홉 조명을 두 자리 이름("02"~"10")의 SHA-256 순으로 늘어놓아,
    앞 4개를 묶음 A, 나머지 5개를 묶음 B로 자른다. 각 묶음은 이름 순으로 돌려준다."""
    ranked = sorted((light for light in ILLUMINATIONS if light != REFERENCE), key=path_key)
    return tuple(sorted(ranked[:4])), tuple(sorted(ranked[4:]))


# 캐시를 만들 때 출력한다. (범주, 분할)별 칸에는 시편 수도 들어간다. 라벨 규칙 적용 결과를 눈으로 확인하는
# 용도다.
def count_table(rows: Sequence[M2adRow]) -> str:
    """(범주, 분할, 라벨)별 이미지 장수와, test 분할의 시점·조명별 정상/결함/제외 장수 표를 만든다."""
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


# 캐시 파일 이름 규칙: 크기마다 이미지 배열(m2ad_<size>_images.npy)과 행 순서 색인(m2ad_<size>_index.csv) 한
# 쌍. P0는 256, D-S는 448 캐시를 쓴다(README 재현 명령).
def images_path(out_dir: Path, size: int) -> Path:
    return Path(out_dir) / f"m2ad_{size}_images.npy"


def index_path(out_dir: Path, size: int) -> Path:
    return Path(out_dir) / f"m2ad_{size}_index.csv"


# main이 이미 완성된 캐시를 다시 만들지 않게(그리고 장부에 불필요한 줄을 남기지 않게) 쓴다. 배열 모양은
# 전체를 읽지 않고 헤더만 본다(_npy_shape).
def cache_is_complete(out_dir: Path, size: int, rows: Sequence[M2adRow]) -> bool:
    """`size`의 이미지 배열과 색인이 있고 `rows` 순서를 따르면(배열 모양도 맞으면) True."""
    if not (images_path(out_dir, size).exists() and index_path(out_dir, size).exists()):
        return False
    index = [r.img_path for r in rows]
    if _cache._read_index(index_path(out_dir, size)) != index:
        return False
    return _cache._npy_shape(images_path(out_dir, size)) == (len(index), size, size, 3)


# 놓인 순서는 header_offset으로 정한다. 하나라도 없으면 FileNotFoundError. 파일 순서대로 읽으면 HDD에서
# 앞뒤로 찾아다니지 않고 순차로 읽게 된다.
def _members(zip_path: Path, wanted: dict[str, int]) -> list[zipfile.ZipInfo]:
    """zip 하나에서 원하는 멤버들을 파일 안에 놓인 순서대로 돌려준다. 모두 있어야 한다."""
    with zipfile.ZipFile(zip_path) as z:
        found = {info.filename: info for info in z.infolist() if info.filename in wanted}
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise FileNotFoundError(f"{len(missing)} images are not in {zip_path}: {missing[:3]}")
    return sorted(found.values(), key=lambda info: info.header_offset)


def _plan(zip_dir: Path, rows: Sequence[M2adRow]) -> list[tuple[Path, list[zipfile.ZipInfo], dict[str, int]]]:
    """캐시 만들기 계획: `<zip_dir>/<범주>.zip`마다 읽을 멤버(파일 순서)와 그 멤버가 갈 캐시 행 번호.

    zip의 중앙 디렉터리만 읽고 이미지는 읽지 않는다. 그래서 zip·멤버 누락(FileNotFoundError)과 쓸 수 없는
    행(ValueError)이 테스트 이미지를 읽기 전에 여기서 드러나고, `main`은 장부 줄을 쓰기 전에 이것을 부른다.
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


# 디코딩과 리사이즈를 스레드 풀에서 병렬로 한다. 읽기는 순차(HDD에 유리), 계산은 병렬, 쓰기는 한 곳이라 쓰기
# 경합이 없다.
def _fill(
    plan: list[tuple[Path, list[zipfile.ZipInfo], dict[str, int]]],
    size: int,
    writer: _cache._RowWriter,
    workers: int,
    progress: bool,
) -> None:
    """계획대로 zip을 앞에서 뒤로 읽고, 스레드 풀에서 디코딩하고, 행은 이 스레드에서만 쓴다."""
    total = sum(len(infos) for _, infos, _ in plan)
    # 대기 중인 작업 수를 작업자 수의 4배로 묶어, 읽어 둔 압축 바이트와 디코딩된 배열이 메모리에 무한정 쌓이지
    # 않게 한다.
    max_pending = max(1, workers) * 4
    pending: deque = deque()
    done = 0
    started = time.perf_counter()

    # 대기열이 limit보다 길면 먼저 넣은 작업부터 결과를 기다려 해당 행에 쓴다. 들어간 순서대로 꺼내므로 결과
    # 순서가 보장되고, 500장마다 진행 상황을 찍는다.
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


# ".part" 이름으로 쓰고 끝에 바꾸는 것은, 중간에 끊겨도 옛 캐시나 반쯤 쓴 파일이 '완성된 캐시'로 보이지 않게
# 하려는 것이다(cache._publish docstring).
def build_cache(
    zip_dir: Path,
    rows: Sequence[M2adRow],
    size: int,
    out_dir: Path,
    *,
    workers: int = 8,
    progress: bool = False,
) -> None:
    """M2AD 캐시 `m2ad_{size}_images.npy`(uint8 [len(rows), size, size, 3])와 색인을 `rows` 순서로 쓴다.

    `rows`가 가리키는 멤버만 `<zip_dir>/<범주>.zip`에서, 각 zip 안에 놓인 순서대로 읽는다(HDD에서 순차).
    테스트 장부 기록은 여기서 하지 않는다(`main`이 이 함수를 부르기 전에 한다). 모든 파일을 ".part" 이름으로
    쓰고 끝에 이름을 바꾸며, 색인은 배열 다음에 옮긴다.
    """
    out_dir = Path(out_dir)
    # 중앙 디렉터리만 본다: zip이나 멤버가 없으면 아무것도 디코딩하기 전에 빌드가 멈춘다.
    plan = _plan(zip_dir, rows)
    index = [r.img_path for r in rows]

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [(images_path(out_dir, size), index_path(out_dir, size))]
    # 바꿀 파일이 다른 곳에서 메모리 맵으로 열려 있으면(Windows에서는 교체 불가) 오래 걸리는 빌드를 하기 전에
    # 지금 PermissionError로 멈춘다.
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
    # Ctrl+C(KeyboardInterrupt)까지 포함해 어떤 이유로 끊겨도 .part 파일을 지워 쓰다 만 파일이 남지 않게 한다.
    except BaseException:
        for part in parts.values():
            part.unlink(missing_ok=True)
        raise


# 배열을 메모리 맵(mmap_mode="r")으로 열어 필요한 행만 디스크에서 읽는다.
class M2adCache:
    """완성된 M2AD 캐시의 읽기 전용 접근자. 봉인 테스트를 막지는 않는다: 그것은 `run_m2ad`의 몫이다."""

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

    # 행 번호를 오름차순으로 나눠 읽는 cache._take를 써서 HDD에서 순차에 가깝게 읽는다. 색인에 없는 이미지는
    # KeyError로 알린다.
    def images(self, rows: Sequence[M2adRow]) -> np.ndarray:
        """uint8 [len(rows), size, size, 3]을 `rows` 순서대로 돌려준다."""
        try:
            index = np.fromiter((self._row[r.img_path] for r in rows), dtype=np.int64, count=len(rows))
        except KeyError as err:
            raise KeyError(f"image not in the cache index: {err.args[0]}") from None
        return _cache._take(self._images, index)

    # 캐시를 다시 만들기 전에 닫아야 한다.
    def close(self) -> None:
        """메모리 맵을 놓는다(Windows는 아직 맵으로 열린 파일을 교체하지 못한다)."""
        self._images = None


# 명령줄 진입점: `python -m defect_inspect.m2ad --size 256`(README 3단계). data/raw/m2ad의 <범주>.zip과
# jsons.zip에서 캐시를 만든다. 이미 완성됐으면 --force 없이는 다시 만들지 않는다.
def main(argv: list[str] | None = None) -> int:
    from . import ledger, paths

    parser = argparse.ArgumentParser(description="M2AD zip에서 리사이즈 캐시를 만든다.")
    parser.add_argument("--size", type=int, default=256, help="정사각형 이미지의 한 변(px)")
    parser.add_argument("--workers", type=int, default=8, help="디코딩 스레드 수")
    parser.add_argument("--force", action="store_true", help="캐시가 이미 완성됐어도 다시 만든다")
    parser.add_argument("--zips", type=Path, default=None, help="<범주>.zip이 있는 폴더(data/raw/m2ad)")
    parser.add_argument("--jsons", type=Path, default=None, help="jsons.zip(기본값: zip 폴더 안)")
    parser.add_argument("--out", type=Path, default=None, help="캐시 폴더(data/cache)")
    args = parser.parse_args(argv)
    zip_dir = args.zips or paths.RAW / "m2ad"
    out_dir = args.out or paths.CACHE

    rows = read_meta(args.jsons or zip_dir / "jsons.zip")
    print(count_table(rows))
    if not args.force and cache_is_complete(out_dir, args.size, rows):
        print(f"M2AD cache for size {args.size} is already complete in {out_dir} (use --force to rebuild)")
    else:
        # 이미지를 읽지 않고도 실패할 수 있는 점검(zip·멤버 누락, 옛 파일 교체 가능 여부)을 장부 기록 전에
        # 모두 해서, 실패한 빌드가 장부에 줄을 남기지 않게 한다.
        _plan(zip_dir, rows)
        _cache._check_replaceable([(images_path(out_dir, args.size), index_path(out_dir, args.size))])
        # 캐시는 test 이미지도 배열로 옮기므로 봉인 규칙의 예외(점수·지표를 내지 않음)로 돌리되, 읽기 전에
        # 장부에 stage "cache"로 남긴다(experiments.md 공통 규칙 '봉인 테스트', 규칙 변경 1과 같은 처리).
        if any(r.split == "test" for r in rows):
            # 이제 테스트 이미지를 디코딩한다: 모든 읽기는 일어나기 전에 기록한다.
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
