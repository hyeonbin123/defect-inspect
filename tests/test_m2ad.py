# [파일 역할] M2AD(같은 시편을 조명 10가지로 찍은 데이터) 모듈 src/defect_inspect/m2ad.py를 검증한다:
# 메타데이터 읽기와 라벨 규칙, 시편 단위 겹·재보정 시편·조명 묶음, zip에서 리사이즈 캐시 만들기·읽기, 그리고
# 캐시를 만들기 전에 봉인 테스트 장부에 기록하는 CLI.
# [왜 필요한가] 3-B(합성 교란 대 실제 조명)와 7단계(조명 등록·변화 감지)의 모든 수치가 이 행 목록·라벨·겹
# 위에서 나온다. 라벨 규칙(결함이 보이고 판별 가능한 이미지만 결함)이나 시편 순서가 문서(docs/experiments.md
# 3-B, 7단계)와 다르면 규칙과 다른 실험이 된다. 캐시는 test 이미지를 배열로 옮기므로 장부 기록이 빠지거나,
# 실패한 빌드가 장부에 줄을 남기면 봉인 기록이 틀어진다.
# [방법] 실제 데이터(범주당 약 14GB) 없이, 같은 구조의 가짜 메타데이터와 작은 PNG를 담은 zip을 tmp_path에
# 만들어 돌린다. 장부는 autouse fixture로 임시 파일로 바꾼다.

import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

# cache는 디코딩 함수(_decode_image)를 가짜로 바꾸고 기대 이미지를 같은 리사이즈로 만들려고, ledger·paths는
# 장부 경로와 커밋 값을 바꾸려고 모듈째 가져온다.
from defect_inspect import cache, ledger, m2ad, paths
from defect_inspect.m2ad import (
    ILLUMINATIONS,
    VIEWS,
    M2adCache,
    M2adRow,
    build_cache,
    read_meta,
    recal_specimens,
    specimen_folds,
)

# 캐시 이미지 한 변(16px). 실제는 256(P0)·448(D-S)이지만 테스트는 작게 잡는다.
SIZE = 16


# 모든 테스트에 자동으로 적용되는 fixture. 장부 경로(paths.TEST_LEDGER)를 tmp_path 아래로 바꾸고 git_commit을
# 고정값으로 바꾼다. 테스트가 실수로 저장소의 진짜 reports/test_ledger.jsonl에 줄을 쓰면 봉인 기록이
# 오염되므로 막는다.
@pytest.fixture(autouse=True)
def temporary_ledger(tmp_path, monkeypatch):
    """`m2ad.main`은 test 이미지 읽기를 장부에 기록하므로, 어떤 테스트도 진짜 장부에 쓰면 안 된다."""
    path = tmp_path / "reports" / "test_ledger.jsonl"
    monkeypatch.setattr(paths, "TEST_LEDGER", path)
    monkeypatch.setattr(ledger, "git_commit", lambda root=None: "abc1234")
    return path


# meta_unsupervised.json 항목 하나를 실제 M2AD와 같은 키로 만든다. 경로 형식은
# <범주>/<폴더>/<시편>/A<시점>_I<조명>.png다. m2ad가 읽지 않는 키(cls_name, mask_path 등)도 넣어 실제 파일
# 모양을 흉내 낸다.
def entry(category, folder, specimen, view, light, object_anomaly=0, image_anomaly=0, detectable=""):
    return {
        "img_path": f"{category}/{folder}/{specimen}/A{view}_I{light}.png",
        "view": view,
        "illumination": light,
        "object_name": specimen,
        "object_anomaly": object_anomaly,
        "image_anomaly": image_anomaly,
        "cls_name": category,
        "mask_path": "",
        "seg_path": "",
        "detectable": detectable,
    }


# 가짜 메타데이터 전체. 쓰지 않는 범주(Car)와 쓰지 않는 시점(030)을 일부러 넣어 걸러지는지 보고, docstring의
# 세 불량 시편으로 라벨 규칙의 모든 갈래(결함, 판별 불가, 안 보임)를 만든다. 항목 순서를 뒤집어 넣어 파일
# 순서와 무관하게 정렬되는지도 시험한다.
def make_meta(train=("000", "003"), normal=("001",), lights=ILLUMINATIONS, views=(*VIEWS, "030")) -> dict:
    """남길 범주 둘과 무시해야 할 범주 하나. 시점 030도 걸러져야 한다.

    남는 범주마다 test 분할에는 정상 시편과 불량 시편 셋이 있다: `hole_1_000`(모든 시점에서 보이고
    판별 가능), `scratch_1_001`(시점 120에서 판별 불가, 시점 240에서 안 보임), `dent_1_002`(어느
    시점에서도 안 보임).
    """
    meta: dict = {"train": {}, "test": {}}
    for category in ("Bird", "Car", "Motor"):
        train_rows, test_rows = [], []
        for view in views:
            for light in lights:
                train_rows += [entry(category, "Good", s, view, light) for s in train]
                test_rows += [entry(category, "Good", s, view, light) for s in normal]
                test_rows.append(entry(category, "NG", "hole_1_000", view, light, 1, 1, True))
                if view == "120":
                    test_rows.append(entry(category, "NG", "scratch_1_001", view, light, 1, 1, False))
                elif view == "240":
                    test_rows.append(entry(category, "NG", "scratch_1_001", view, light, 1, 0, ""))
                else:
                    test_rows.append(entry(category, "NG", "scratch_1_001", view, light, 1, 1, True))
                test_rows.append(entry(category, "NG", "dent_1_002", view, light, 1, 0, ""))
        meta["train"][category] = train_rows[::-1]  # 파일 안 순서가 결과를 바꾸면 안 된다
        meta["test"][category] = test_rows[::-1]
    return meta


# 메타데이터를 jsons.zip에 쓴다. 이름이 다른 JSON(meta_other.json)도 같이 넣어 정확한 이름의 멤버 하나만
# 읽는지 확인한다.
def write_jsons(tmp_path, meta, member=m2ad.META_MEMBER):
    path = tmp_path / "jsons.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(member, json.dumps(meta))
        z.writestr("meta_other.json", "{}")
    return path


# 멤버 이름의 SHA-256으로 정해지는 작은 PNG 바이트. 5×5 무작위 색을 40px로 키워 이미지마다 다르지만 재현
# 가능하므로, 캐시의 각 행이 올바른 이미지에서 왔는지 expected_image와 비교할 수 있다.
def png_bytes(name: str, side: int = 40) -> bytes:
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, size=(5, 5, 3), dtype=np.uint8)
    img = Image.fromarray(coarse).resize((side, side), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


# 범주별 <범주>.zip을 만든다. 원하지 않는 멤버(쓰지 않는 시점, GT 마스크)와 폴더 항목도 넣고, 멤버를 이름의
# 해시 순으로(reverse면 거꾸로) 넣어 zip 안의 순서가 행 순서·이름 순서와 다르게 한다. skip으로 일부 멤버를 빼
# 누락 상황을 만든다.
def write_zips(zip_dir, rows, *, reverse=False, skip=()):
    """`rows`의 모든 이미지에 원하지 않는 멤버까지 더해 담은 `<category>.zip`."""
    zip_dir.mkdir(parents=True, exist_ok=True)
    for category in {r.category for r in rows}:
        names = [r.img_path for r in rows if r.category == category and r.img_path not in skip]
        names += [f"{category}/Good/000/A030_I01.png", f"{category}/GT/hole_1_000/A000_mask.png"]
        names.sort(key=lambda name: hashlib.sha256(name.encode()).hexdigest(), reverse=reverse)
        with zipfile.ZipFile(zip_dir / f"{category}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"{category}/", b"")
            for name in names:
                z.writestr(name, png_bytes(name))


# main은 캐시를 새로 만들 때 zip 세 개의 SHA-256을 기록한 값(m2ad.ZIP_SHA256)과 대조한다. 가짜 zip으로 정상
# 경로를 돌리려면 기대 지문을 그 가짜 파일들의 것으로 바꿔야 하므로, 지금 디스크에 있는 가짜 파일의 해시를
# 계산해 그 자리에 넣는다. 돌려준 dict는 테스트가 일부를 틀린 값으로 바꿔 불일치를 만들 때 쓴다.
def pin_zip_hashes(monkeypatch, zip_dir, jsons) -> dict:
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    pinned = {
        "jsons.zip": digest(jsons),
        **{f"{c}.zip": digest(zip_dir / f"{c}.zip") for c in m2ad.CATEGORIES},
    }
    monkeypatch.setattr(m2ad, "ZIP_SHA256", pinned)
    return pinned


# 조명 2개(01, 02)만 있는 작은 행 목록. 캐시 테스트를 빨리 돌리려는 것이다.
def small_rows(tmp_path):
    meta = make_meta(lights=("01", "02"))
    # 여기서는 조명이 둘뿐이라 `read_meta`에 조명이 모두 있는지 확인하라고 하지 않는다.
    return read_meta(write_jsons(tmp_path, meta))


# --------------------------------------------------------------------------------------------- 메타데이터


# read_meta가 등록한 두 범주(Motor, Bird)와 세 시점만 남기고(Car와 시점 030 제외), 열 가지 조명을 모두 읽어
# 기대한 행 수를 내는지 본다.
def test_read_meta_keeps_two_categories_and_three_views(tmp_path):
    rows = read_meta(write_jsons(tmp_path, make_meta()))
    assert {r.category for r in rows} == {"Motor", "Bird"}
    assert {r.view for r in rows} == set(VIEWS)
    assert {r.illumination for r in rows} == set(ILLUMINATIONS)
    # 범주 2 x 시점 3 x 조명 10 x (학습 시편 2 + 정상 시편 1 + 불량 시편 3)
    assert len(rows) == 2 * 3 * 10 * 6
    assert all(isinstance(r, M2adRow) for r in rows)


# 행 순서가 (범주 선언 순서, 분할 train→test, 시편, 시점, 조명)인지 본다. 캐시 배열의 행 순서가 이 순서로
# 고정되므로, 메타데이터 파일 안의 순서(여기서는 뒤집혀 있음)가 바뀌어도 같은 캐시가 나와야 한다.
def test_read_meta_order(tmp_path):
    rows = read_meta(write_jsons(tmp_path, make_meta()))
    keys = [
        (("Motor", "Bird").index(r.category), ("train", "test").index(r.split), r.specimen, r.view)
        for r in rows
    ]
    assert keys == sorted(keys)
    first = rows[:11]
    assert [r.illumination for r in first[:10]] == list(ILLUMINATIONS)
    assert (first[0].category, first[0].split, first[0].specimen, first[0].view) == (
        "Motor",
        "train",
        "000",
        "000",
    )
    assert first[10].view == "120"
    assert rows[0].img_path == "Motor/Good/000/A000_I01.png"


# 등록한 라벨 규칙(experiments.md 3-B)을 시편·시점별로 확인한다: 정상 시편은 두 분할 모두 0, 결함이 보이고
# detectable이 True면 1, detectable False나 결함이 안 보이는 시점(image_anomaly 0)은 -1(평가 제외). dent처럼
# 어느 시점에서도 결함이 안 보이는 시편은 불량 시편이지만 모두 제외된다. train에는 0만 있어야 한다.
def test_label_rule(tmp_path):
    rows = read_meta(write_jsons(tmp_path, make_meta()))
    by = {(r.category, r.specimen, r.view, r.illumination): r for r in rows}
    for category in ("Motor", "Bird"):
        for light in ILLUMINATIONS:
            # 정상 시편은 두 분할 모두 라벨 0이다.
            assert by[category, "000", "000", light].label == 0
            assert by[category, "001", "240", light].label == 0
            assert by[category, "001", "240", light].object_anomaly == 0
            # 보이고 판별 가능한 결함.
            assert by[category, "hole_1_000", "120", light].label == 1
            assert by[category, "scratch_1_001", "000", light].label == 1
            # detectable False면 제외. 결함이 안 보이는 경우(detectable "")도 제외.
            assert by[category, "scratch_1_001", "120", light].label == -1
            assert by[category, "scratch_1_001", "240", light].label == -1
            assert by[category, "dent_1_002", "000", light].label == -1
            assert by[category, "dent_1_002", "000", light].object_anomaly == 1
    assert {r.label for r in rows if r.split == "train"} == {0}


# label_of의 진리표. 불량 시편이라도 결함이 보이고(image_anomaly 1) 판별 가능(True)해야만 1이다.
@pytest.mark.parametrize(
    ("object_anomaly", "image_anomaly", "detectable", "expected"),
    [(0, 0, "", 0), (1, 1, True, 1), (1, 1, False, -1), (1, 0, "", -1), (1, 1, "", -1), (1, 0, True, -1)],
)
def test_label_of(object_anomaly, image_anomaly, detectable, expected):
    assert m2ad.label_of(object_anomaly, image_anomaly, detectable) == expected


# detectable은 JSON true(파이썬 True)일 때만 결함으로 센다. 1이나 "True" 같은 참으로 읽히는 값이 결함 라벨을
# 만들면 메타데이터 오류가 조용히 라벨로 바뀐다.
def test_label_of_needs_a_real_true():
    # JSON `true`가 아닌 참 같은 값은 결함 라벨을 만들지 않는다.
    assert m2ad.label_of(1, 1, 1) == -1
    assert m2ad.label_of(1, 1, "True") == -1


# 메타 파일이 zip 안 하위 폴더(jsons/)에 있어도 파일 이름으로 찾는지 본다.
def test_read_meta_finds_the_member_in_a_folder(tmp_path):
    path = write_jsons(tmp_path, make_meta(), member="jsons/" + m2ad.META_MEMBER)
    assert len(read_meta(path)) == 360


# 메타데이터 일관성 점검이 문제마다 알맞은 오류를 내는지 본다: 같은 이미지 중복, train에 불량 시편, 정상
# 시편인데 image_anomaly 1, 모르는 조명(11), detectable이 ""·bool이 아님, 한 시편이 두 분할에 걸침, 범주 항목
# 없음, 메타 파일 없음. 쓰지 않는 시점(030)의 중복은 걸러진 뒤라 문제가 아니다.
# 특히 '두 분할에 걸친 시편'은 학습 시편이 평가에 섞여 새는 것이라 반드시 막아야 한다.
def test_read_meta_rejects_bad_metadata(tmp_path):
    meta = make_meta()
    kept = next(e for e in meta["test"]["Motor"] if e["view"] == "000")
    meta["test"]["Motor"].append(dict(kept))
    with pytest.raises(ValueError, match="more than once"):
        read_meta(write_jsons(tmp_path, meta))

    # 쓰지 않는 시점에서 같은 항목이 다시 나오는 것은 무시한다.
    meta = make_meta()
    unused = next(e for e in meta["test"]["Motor"] if e["view"] == "030")
    meta["test"]["Motor"].append(dict(unused))
    assert len(read_meta(write_jsons(tmp_path, meta))) == 360

    meta = make_meta()
    meta["train"]["Bird"].append(entry("Bird", "Good", "099", "000", "01", 1, 1, True))
    with pytest.raises(ValueError, match="train rows"):
        read_meta(write_jsons(tmp_path, meta))

    meta = make_meta()
    meta["test"]["Bird"].append(entry("Bird", "Good", "098", "000", "01", 0, 1, True))
    with pytest.raises(ValueError, match="normal specimen"):
        read_meta(write_jsons(tmp_path, meta))

    meta = make_meta()
    meta["test"]["Bird"].append(entry("Bird", "NG", "x_1_009", "000", "11", 1, 1, True))
    with pytest.raises(ValueError, match="illumination"):
        read_meta(write_jsons(tmp_path, meta))

    meta = make_meta()
    meta["test"]["Bird"].append(entry("Bird", "NG", "x_1_009", "000", "01", 1, 1, "yes"))
    with pytest.raises(ValueError, match="detectable"):
        read_meta(write_jsons(tmp_path, meta))

    meta = make_meta()
    meta["test"]["Motor"].append(entry("Motor", "Again", "000", "000", "01"))  # 학습 시편
    with pytest.raises(ValueError, match="both splits"):
        read_meta(write_jsons(tmp_path, meta))

    meta = make_meta()
    del meta["test"]["Motor"]
    with pytest.raises(ValueError, match="test/Motor"):
        read_meta(write_jsons(tmp_path, meta))

    with zipfile.ZipFile(tmp_path / "empty.zip", "w") as z:
        z.writestr("readme.txt", "x")
    with pytest.raises(FileNotFoundError):
        read_meta(tmp_path / "empty.zip")


# ----------------------------------------------------------------------------------------------- 시편 순서


# 기준 구현: 중복을 뺀 이름을 UTF-8 SHA-256 16진수 순으로 정렬한다(splits.path_key와 같은 방식이어야 함).
def sha_order(names):
    return sorted(set(names), key=lambda name: hashlib.sha256(name.encode("utf-8")).hexdigest())


# 시편 겹 번호가 SHA-256 순위 mod 5인지 본다. 해시 순서가 이름 순서와 실제로 다른지 먼저 확인하고(같으면
# 테스트가 헛돈다), 30개면 겹마다 6개씩 고르게 나뉘며, 입력 순서나 이미지마다 반복된 시편 이름이 결과를 바꾸지
# 않아야 한다.
def test_folds_follow_the_sha256_rank():
    names = [f"{i:03d}" for i in range(0, 60, 2)]
    folds = specimen_folds(names)
    ordered = sha_order(names)
    assert ordered != sorted(names)  # 해시 순서는 이름 순서와 다르다
    assert folds == {name: rank % 5 for rank, name in enumerate(ordered)}
    assert sorted(np.bincount(list(folds.values())).tolist()) == [6, 6, 6, 6, 6]
    # 입력 순서와 반복된 이름(이미지마다 하나씩)은 결과를 바꾸지 않는다.
    assert specimen_folds(names[::-1] + names) == folds


# 재보정 시편이 같은 SHA-256 순서의 앞 n개이고, n = 8이 n = 30의 앞부분(중첩)인지 본다. 3-B는 새 조명의 정상
# 8장·30장으로 다시 보정한 결과를 비교하므로(README), 8장이 30장의 부분 집합이어야 표본 차이만 남는다. 앞
# 8개는 겹 0~4를 고루 덮는다. 범위 밖 n은 거절한다.
def test_recal_specimens_are_the_first_in_the_same_order():
    names = [f"{i:03d}" for i in range(30)]
    ordered = sha_order(names)
    assert recal_specimens(names, 8) == ordered[:8]
    assert recal_specimens(names[::-1], 30) == ordered
    assert recal_specimens(names, 0) == []
    assert recal_specimens(names, 8) == recal_specimens(names, 30)[:8]  # 중첩
    # 앞 8개 시편이 모든 겹을 덮는다.
    folds = specimen_folds(names)
    assert [folds[s] for s in recal_specimens(names, 8)] == [0, 1, 2, 3, 4, 0, 1, 2]
    with pytest.raises(ValueError):
        recal_specimens(names, 31)
    with pytest.raises(ValueError):
        recal_specimens(names, -1)


# 시편 이름이 문자열이 아니면 TypeError다. 해시 순서가 문자열 인코딩에 의존하므로 숫자가 섞이면 순서가
# 정의되지 않는다.
def test_specimen_names_must_be_strings():
    with pytest.raises(TypeError):
        specimen_folds([1, 2, 3])


# ---------------------------------------------------------------------------------------------------- 캐시


# 멤버 이름의 PNG를 VisA 캐시와 같은 리사이즈(cache.resize_image)로 줄인 기대 이미지.
def expected_image(name: str) -> np.ndarray:
    with Image.open(io.BytesIO(png_bytes(name))) as img:
        return cache.resize_image(img, SIZE)


# 등록한 상수(범주 둘, 시점 셋, 조명 I01~I10, 기준 조명 I01, 5겹, 재보정 8·30장)가 experiments.md 3-B와 같은지
# 고정한다. 무심코 바꾸면 등록한 실험과 다른 것을 재게 된다.
def test_constants_are_the_registered_ones():
    assert m2ad.CATEGORIES == ("Motor", "Bird") and m2ad.VIEWS == ("000", "120", "240")
    assert m2ad.ILLUMINATIONS == tuple(f"{i:02d}" for i in range(1, 11))
    assert m2ad.REFERENCE == "01" and m2ad.N_FOLDS == 5 and m2ad.RECAL_SIZES == (8, 30)


# 캐시를 만들 때 대조하는 zip 지문(experiments.md 공통 규칙 '데이터'에 적은 값)을 고정한다. 2026-10-29에 이
# PC의 data/raw/m2ad에서 계산했고 Hugging Face 데이터셋 파일 목록의 LFS sha256과 같았다. 대조하는 파일은
# jsons.zip과 실험에 쓰는 두 범주의 zip뿐이다.
def test_zip_hashes_are_the_recorded_ones():
    assert m2ad.ZIP_SHA256 == {
        "jsons.zip": "bbeaa75c231c981e443c818859a701fcc0daf1769065abd91631f7e676725980",
        "Motor.zip": "eeffdb29450b5ffc5cb9bbc85afa5010ee633eefb87c4515c0f20b74f93e3e0f",
        "Bird.zip": "b5bb22a976440d8739e84724eb864b290f23eee053511ab47f8a4c200fabace7",
    }
    assert set(m2ad.ZIP_SHA256) == {"jsons.zip", *(f"{c}.zip" for c in m2ad.CATEGORIES)}


# 캐시 만들기의 핵심 성질을 본다. zipfile.ZipFile.read를 감시해, 각 zip을 놓인 순서(header_offset
# 오름차순)대로 앞에서 뒤로 읽는지(HDD 순차 읽기), 원하는 멤버만 한 번씩 읽는지, 그 순서가 이름 순서와 다른지
# 확인한다. zip 안 순서를 정방향·역방향으로 바꿔도 캐시 배열과 색인은 행 순서대로 똑같아야 하고, .part 파일이
# 남지 않으며, 행마다 올바른 이미지가 들어 있어야 한다. 디코딩 횟수로 원하지 않는 멤버를 디코딩하지 않았는지도
# 센다. 마지막으로 cache_is_complete가 같은 행 순서·크기에서만 True인지 본다.
def test_build_cache_follows_the_row_order_not_the_member_order(tmp_path, monkeypatch):
    rows = small_rows(tmp_path)
    decoded = []
    real_decode = cache._decode_image

    def counting(data, size):
        decoded.append(len(data))
        return real_decode(data, size)

    monkeypatch.setattr(cache, "_decode_image", counting)
    reads: list[tuple[str, int, str]] = []
    real_read = zipfile.ZipFile.read

    def spying_read(self, name, pwd=None):
        info = name if isinstance(name, zipfile.ZipInfo) else self.getinfo(name)
        reads.append((self.filename, info.header_offset, info.filename))
        return real_read(self, name, pwd)

    arrays = []
    for i, reverse in enumerate((False, True)):
        zip_dir, out = tmp_path / f"zips{i}", tmp_path / f"cache{i}"
        write_zips(zip_dir, rows, reverse=reverse)
        reads.clear()
        with monkeypatch.context() as patch:
            patch.setattr(zipfile.ZipFile, "read", spying_read)
            build_cache(zip_dir, rows, SIZE, out, workers=3)
        # 각 zip은 앞에서 뒤로(HDD에서 순차로) 읽고 그 순서는 이름 순서와 다르며, 원하는 멤버만 각각 한 번씩
        # 읽는다.
        assert sorted(name for _, _, name in reads) == sorted(r.img_path for r in rows)
        for category in ("Motor", "Bird"):
            of_zip = [
                (offset, name) for path, offset, name in reads if path == str(zip_dir / f"{category}.zip")
            ]
            assert len(of_zip) == sum(r.category == category for r in rows)
            assert [offset for offset, _ in of_zip] == sorted(offset for offset, _ in of_zip)
            assert [name for _, name in of_zip] != sorted(name for _, name in of_zip)
        index = m2ad.index_path(out, SIZE).read_text(encoding="utf-8").splitlines()
        assert index == [r.img_path for r in rows]
        arrays.append(np.load(m2ad.images_path(out, SIZE)))
        assert arrays[-1].shape == (len(rows), SIZE, SIZE, 3) and arrays[-1].dtype == np.uint8
        assert not list(out.glob("*.part"))
    assert np.array_equal(arrays[0], arrays[1])
    assert len(decoded) == 2 * len(rows)  # 원하는 멤버만 디코딩한다
    for i in (0, 7, len(rows) - 1):
        assert np.array_equal(arrays[0][i], expected_image(rows[i].img_path))
    assert m2ad.cache_is_complete(tmp_path / "cache0", SIZE, rows)
    assert not m2ad.cache_is_complete(tmp_path / "cache0", SIZE, rows[::-1])
    assert not m2ad.cache_is_complete(tmp_path / "cache0", SIZE + 1, rows)


# 행 일부(시점 120만)를 거꾸로 된 순서로 캐시해도, M2adCache가 행 객체로 이미지를 찾아 요청한 순서(중복
# 포함)대로 돌려주는지 본다. 캐시에 없는 행은 KeyError다.
def test_build_cache_of_a_subset_in_another_order(tmp_path):
    rows = small_rows(tmp_path)
    write_zips(tmp_path / "zips", rows)
    subset = [r for r in rows if r.view == "120"][::-1]
    build_cache(tmp_path / "zips", subset, SIZE, tmp_path / "cache")
    store = M2adCache(tmp_path / "cache", SIZE)
    assert len(store) == len(subset)
    picked = [subset[5], subset[0], subset[5]]
    images = store.images(picked)
    assert images.shape == (3, SIZE, SIZE, 3)
    for image, row in zip(images, picked, strict=True):
        assert np.array_equal(image, expected_image(row.img_path))
    with pytest.raises(KeyError):
        store.images([r for r in rows if r.view == "000"][:1])
    store.close()


# zip에 멤버 하나가 없으면 이미지를 하나도 디코딩하기 전에(중앙 디렉터리만 보고) FileNotFoundError로 멈추고
# 출력 폴더에 아무것도 남기지 않는지 본다. 긴 빌드가 끝 무렵에 실패하는 일을 막는다. zip 폴더 자체가 없어도
# 같은 오류다.
def test_build_cache_stops_before_decoding_when_a_member_is_missing(tmp_path, monkeypatch):
    rows = small_rows(tmp_path)
    gone = [r.img_path for r in rows if r.category == "Bird"][3]
    write_zips(tmp_path / "zips", rows, skip={gone})

    def no_decoding(data, size):
        raise AssertionError("nothing may be decoded")

    monkeypatch.setattr(cache, "_decode_image", no_decoding)
    with pytest.raises(FileNotFoundError, match="Bird"):
        build_cache(tmp_path / "zips", rows, SIZE, tmp_path / "cache")
    assert not (tmp_path / "cache").exists() or not list((tmp_path / "cache").iterdir())
    with pytest.raises(FileNotFoundError):
        build_cache(tmp_path / "nowhere", rows, SIZE, tmp_path / "cache")


# 디코딩 도중(5번째 이미지) 실패하면 .part 파일까지 모두 지워 반쯤 쓴 캐시가 '완성된 캐시'로 남지 않는지 본다.
def test_failed_build_leaves_no_files(tmp_path, monkeypatch):
    rows = small_rows(tmp_path)
    write_zips(tmp_path / "zips", rows)
    calls = []

    def broken(data, size):
        calls.append(1)
        if len(calls) == 5:
            raise OSError("broken image")
        return np.zeros((size, size, 3), dtype=np.uint8)

    monkeypatch.setattr(cache, "_decode_image", broken)
    with pytest.raises(OSError, match="broken image"):
        build_cache(tmp_path / "zips", rows, SIZE, tmp_path / "cache", workers=1)
    assert not list((tmp_path / "cache").iterdir())


# 빈 행 목록과 같은 이미지를 두 번 담은 행 목록은 거절한다. 중복 행은 색인에서 이미지를 찾을 때 모호해진다.
def test_build_cache_rejects_empty_and_repeated_rows(tmp_path):
    rows = small_rows(tmp_path)
    write_zips(tmp_path / "zips", rows)
    with pytest.raises(ValueError):
        build_cache(tmp_path / "zips", [], SIZE, tmp_path / "cache")
    with pytest.raises(ValueError):
        build_cache(tmp_path / "zips", [rows[0], rows[0]], SIZE, tmp_path / "cache")


# 캐시가 없으면 오류 메시지에 만드는 명령(python -m defect_inspect.m2ad --size N)을 알려 주는지 본다.
def test_cache_names_a_missing_build(tmp_path):
    with pytest.raises(FileNotFoundError, match="defect_inspect.m2ad --size 24"):
        M2adCache(tmp_path, 24)


# --------------------------------------------------------------------------------------------- 명령줄(CLI)


# 임시 장부의 줄들을 dict 목록으로 읽는다(파일이 없으면 빈 목록).
def ledger_entries(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# CLI가 test 이미지를 읽기 전에 장부에 stage "cache" 한 줄을 남기는지 본다. build_cache를 감싸 호출되는 순간의
# 장부 줄 수를 기록해 '기록 → 빌드' 순서를 확인한다. 이미 완성된 캐시는 아무것도 읽지 않으므로 줄이 늘지 않고,
# --force로 다시 만들면 두 번째 줄이 생겨야 한다. 장부는 봉인 테스트 이미지를 읽을 때마다 남긴다는
# 규칙을 캐시 빌드에도 적용한 것이다(docs/experiments.md 봉인 테스트 절의 예외).
# zip 지문은 가짜 파일의 것으로 맞춰 두고(pin_zip_hashes) sha256_file 호출을 센다: 캐시를 만들 때마다 세
# 파일을 한 번씩 대조하고, 이미 완성된 캐시는 zip을 해시하지 않아야 한다(실제 zip은 30GB라 매번 읽지 않는다).
def test_main_records_the_read_before_it_builds(tmp_path, capsys, monkeypatch, temporary_ledger):
    jsons = write_jsons(tmp_path, make_meta(lights=("01", "02")))
    rows = read_meta(jsons)
    write_zips(tmp_path / "zips", rows)
    pin_zip_hashes(monkeypatch, tmp_path / "zips", jsons)
    argv = ["--size", str(SIZE), "--zips", str(tmp_path / "zips"), "--jsons", str(jsons)]
    argv += ["--out", str(tmp_path / "cache"), "--workers", "2"]
    real_build, real_sha256 = m2ad.build_cache, m2ad.sha256_file
    recorded_before_build = []
    hashed = []

    def watching_build(*args, **kwargs):
        recorded_before_build.append(len(ledger_entries(temporary_ledger)))
        return real_build(*args, **kwargs)

    def counting_sha256(path, *args, **kwargs):
        hashed.append(path.name)
        return real_sha256(path, *args, **kwargs)

    monkeypatch.setattr(m2ad, "build_cache", watching_build)
    monkeypatch.setattr(m2ad, "sha256_file", counting_sha256)
    assert m2ad.main(argv) == 0
    entries = ledger_entries(temporary_ledger)
    assert len(entries) == 1 and recorded_before_build == [1]
    assert (entries[0]["stage"], entries[0]["config"]) == ("cache", f"m2ad-cache-{SIZE}")
    assert entries[0]["commit"] == "abc1234"
    assert entries[0]["note"] == "M2AD resize cache build: pixels copied into arrays, no scores or metrics"
    assert sorted(hashed) == ["Bird.zip", "Motor.zip", "jsons.zip"]
    out = capsys.readouterr().out
    assert str(temporary_ledger) in out and f"({len(rows)}, {SIZE}, {SIZE}, 3)" in out
    assert m2ad.cache_is_complete(tmp_path / "cache", SIZE, rows)

    # 완성된 캐시는 건너뛴다: 아무것도 읽지 않고(zip 해시도 하지 않고) 기록도 하지 않는다.
    assert m2ad.main(argv) == 0
    assert len(ledger_entries(temporary_ledger)) == 1 and recorded_before_build == [1] and len(hashed) == 3
    # --force는 이미지를 다시 읽으므로 zip을 다시 대조하고 두 번째 줄이 생긴다.
    assert m2ad.main([*argv, "--force"]) == 0
    assert len(ledger_entries(temporary_ledger)) == 2 and recorded_before_build == [1, 2] and len(hashed) == 6


# 캐시를 새로 만들 때 zip 하나의 지문이 기록과 다르면(다른 판의 파일, 내려받기 손상) 장부에 줄을 쓰기 전에,
# 이미지를 하나도 디코딩하기 전에 ValueError로 멈추는지 본다. 메시지에는 파일, 기대·실제 해시, 우회 옵션이
# 들어간다. --skip-hash-check는 대조를 건너뛰고 캐시를 만들되, 장부 메모에 '지문을 대조하지 않았다'를 남겨
# 나중에 그 캐시가 검증되지 않은 zip에서 나왔음을 알 수 있게 한다. jsons.zip이 다르면 큰 zip을 해시하기 전에
# 멈춘다(맨 먼저 대조한다).
def test_main_stops_before_the_ledger_when_a_zip_hash_differs(tmp_path, monkeypatch, temporary_ledger):
    jsons = write_jsons(tmp_path, make_meta(lights=("01", "02")))
    rows = read_meta(jsons)
    write_zips(tmp_path / "zips", rows)
    pinned = pin_zip_hashes(monkeypatch, tmp_path / "zips", jsons)
    argv = ["--size", str(SIZE), "--zips", str(tmp_path / "zips"), "--jsons", str(jsons)]
    argv += ["--out", str(tmp_path / "cache"), "--workers", "2"]

    def no_decoding(data, size):
        raise AssertionError("nothing may be decoded")

    monkeypatch.setattr(cache, "_decode_image", no_decoding)
    monkeypatch.setattr(m2ad, "ZIP_SHA256", {**pinned, "Bird.zip": "0" * 64})
    with pytest.raises(ValueError, match="Bird.zip") as raised:
        m2ad.main(argv)
    message = str(raised.value)
    assert "0" * 64 in message and pinned["Bird.zip"] in message and "--skip-hash-check" in message
    assert not temporary_ledger.exists()
    assert not (tmp_path / "cache").exists() or not list((tmp_path / "cache").iterdir())

    real_sha256 = m2ad.sha256_file
    hashed = []

    def counting_sha256(path, *args, **kwargs):
        hashed.append(path.name)
        return real_sha256(path, *args, **kwargs)

    monkeypatch.setattr(m2ad, "sha256_file", counting_sha256)
    monkeypatch.setattr(m2ad, "ZIP_SHA256", {**pinned, "jsons.zip": "f" * 64})
    with pytest.raises(ValueError, match="jsons.zip"):
        m2ad.main(argv)
    assert hashed == ["jsons.zip"] and not temporary_ledger.exists()

    # 대조를 건너뛰면 만들고, 장부 메모에 남긴다.
    hashed.clear()
    monkeypatch.setattr(cache, "_decode_image", lambda data, size: np.zeros((size, size, 3), dtype=np.uint8))
    assert m2ad.main([*argv, "--skip-hash-check"]) == 0
    entries = ledger_entries(temporary_ledger)
    assert len(entries) == 1 and hashed == []
    assert "zip sha256 not checked (--skip-hash-check)" in entries[0]["note"]
    assert m2ad.cache_is_complete(tmp_path / "cache", SIZE, rows)


# 반대로 이미지를 읽지 못하고 실패하는 빌드(zip 없음, 멤버 누락)는 장부에 줄을 남기지 않아야 한다. 읽지 않은
# 것을 읽었다고 기록하면 장부가 틀린다. main은 이미지를 읽지 않는 점검(_plan, 교체 가능 여부)을 장부 기록 전에
# 하므로, 디코딩 함수가 불리면 실패하도록 바꿔 두고 확인한다. 멤버를 되돌리면 같은 명령이 한 줄만 남기고
# 성공한다.
def test_main_records_nothing_when_the_zips_cannot_be_read(tmp_path, monkeypatch, temporary_ledger):
    jsons = write_jsons(tmp_path, make_meta(lights=("01", "02")))
    rows = read_meta(jsons)
    argv = ["--size", str(SIZE), "--jsons", str(jsons), "--out", str(tmp_path / "cache")]

    def no_decoding(data, size):
        raise AssertionError("nothing may be decoded")

    monkeypatch.setattr(cache, "_decode_image", no_decoding)
    # zip이 하나도 없으면 test 이미지를 읽을 수 없으므로 장부에 줄이 생기지 않는다.
    with pytest.raises(FileNotFoundError):
        m2ad.main([*argv, "--zips", str(tmp_path / "nowhere")])
    assert not temporary_ledger.exists()
    # 원하는 멤버 하나가 zip에 없다: 장부 줄을 쓰기 전에 중앙 디렉터리에서 찾아낸다.
    gone = [r.img_path for r in rows if r.category == "Bird"][3]
    write_zips(tmp_path / "zips", rows, skip={gone})
    with pytest.raises(FileNotFoundError, match="1 images are not in"):
        m2ad.main([*argv, "--zips", str(tmp_path / "zips")])
    assert not temporary_ledger.exists()
    assert not (tmp_path / "cache").exists() or not list((tmp_path / "cache").iterdir())
    # 멤버를 되돌리면 같은 명령이 캐시를 만들고 읽기를 한 번 기록한다(지문은 다시 쓴 가짜 zip의 것으로
    # 맞춘다).
    write_zips(tmp_path / "zips", rows)
    pin_zip_hashes(monkeypatch, tmp_path / "zips", jsons)
    monkeypatch.setattr(cache, "_decode_image", lambda data, size: np.zeros((size, size, 3), dtype=np.uint8))
    assert m2ad.main([*argv, "--zips", str(tmp_path / "zips")]) == 0
    assert len(ledger_entries(temporary_ledger)) == 1


# 캐시를 만들 때 출력하는 장수 표가 라벨 규칙 결과를 맞게 세는지 본다. 범주·분할별 시편 수와 정상/결함/제외
# 장수, test의 (범주, 시점) × 조명별 '정상/결함/제외' 칸을 손 계산(주석)과 맞춘다.
def test_count_table_lists_every_label(tmp_path):
    rows = read_meta(write_jsons(tmp_path, make_meta()))
    table = m2ad.count_table(rows).splitlines()
    # Motor test: 정상 시편 1개(30장), 결함은 hole 30 + scratch 10장, 제외는 scratch 20 + dent 30장.
    assert table[1].split() == ["Motor", "train", "2", "60", "0", "0"]
    assert table[2].split() == ["Motor", "test", "4", "30", "40", "50"]
    assert table[4].split() == ["Bird", "test", "4", "30", "40", "50"]
    motor_000 = next(line for line in table if line.startswith("Motor 000"))
    assert motor_000.split()[2:] == ["1/2/1"] * 10
    motor_120 = next(line for line in table if line.startswith("Motor 120"))
    assert motor_120.split()[2:] == ["1/1/2"] * 10


# 7단계 조명 묶음이 I01을 뺀 아홉 조명 이름의 SHA-256 순서로 앞 4개(A)와 나머지 5개(B)인지 본다. 등록한 결과(A
# = 03, 07, 08, 09 / B = 02, 04, 05, 06, 10, experiments.md 7단계)를 그대로 적어 고정하고, 두 묶음이 기준 조명
# 없이 아홉 조명을 빠짐없이 나누는지 확인한다.
def test_illumination_groups_follow_the_sha256_order_of_the_labels():
    a, b = m2ad.illumination_groups()
    ranked = sorted((f"{i:02d}" for i in range(2, 11)), key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    assert a == tuple(sorted(ranked[:4])) and b == tuple(sorted(ranked[4:]))
    # 등록한 분할(docs/experiments.md 7단계)을 그대로 적는다.
    assert a == ("03", "07", "08", "09") and b == ("02", "04", "05", "06", "10")
    assert m2ad.REFERENCE not in a + b and sorted(a + b) == [f"{i:02d}" for i in range(2, 11)]
