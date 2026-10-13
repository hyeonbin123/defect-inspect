# [파일 역할] VisA 분할 모듈 `defect_inspect.splits`를 검증하는 테스트 묶음이다. 정렬 키(path_key), 역할
# 배정(정상 풀과 겹, dev 결함, 라벨 풀, 봉인 테스트), 매니페스트 CSV 입출력, 지도 학습의 k장
# 뽑기(label_subset), 절차별 겹과 행 선택(select), 봉인 관문, tar에서 매니페스트를 만드는 main, 그리고 커밋된
# manifests/visa.csv가 규칙을 지키는지를 확인한다.
# [왜 필요한가] 이 프로젝트의 모든 측정은 이 분할 위에서 돈다. 분할이 결정적이지 않거나(같은 입력에서 다른
# 결과), 한 이미지가 두 역할에 들어가거나, 봉인 테스트가 허락 없이 열리면 결과 전체를 믿을 수
# 없다(docs/experiments.md 분할·봉인 테스트 절). 그래서 규칙을 문장 하나하나 테스트로 옮겨 둔다.
# [설계 메모] 작은 합성 분할(범주 3개, 장수를 일부러 다르게)로 규칙을 확인하고, 실제 매니페스트가 있으면 공식
# 장수(10,821행)와 '자기 경로로 다시 만들면 같은 결과'까지 확인한다. torch 없이 기본 패키지만으로 돌아
# CI에서도 실행된다.

import hashlib
import io
import tarfile
from collections import Counter, defaultdict

import pytest

# paths는 커밋된 매니페스트 위치를, visa의 CATEGORIES·VisaRow는 범주 순서 기준과 분할 CSV 한 행의 형식을
# 쓰려고 가져온다.
from defect_inspect import paths, splits
from defect_inspect.splits import ManifestRow, SealedTestError
from defect_inspect.visa import CATEGORIES, VisaRow

# 합성 분할의 범주별 장수. pcb1과 candle은 학습 결함이 60장(실제 VisA와 같은 수)이라 dev 결함 20장 + 라벨 풀
# 40장이 되고, capsules는 학습 결함이 27장뿐이라 라벨 풀이 7장이 되어 'k가 라벨 풀보다 큰 경우'를 시험할 수
# 있다. 입력 순서와 범주 순서(CATEGORIES)가 다르게 섞여 있다.
# Synthetic split: (train normal, train anomaly, test normal, test anomaly) per category.
SIZES = {"pcb1": (23, 60, 11, 40), "candle": (31, 60, 9, 40), "capsules": (5, 27, 3, 4)}


# 합성 분할 행과 결함 유형 표를 만든다. 결함 유형은 세 장에 한 장꼴로 두 유형(bent, melt)을 가져 '|'로 이어
# 붙이는 경우를 만든다. 마지막에 MD5 순으로 섞는 것은 공식 CSV가 정렬돼 있지 않은 것을 흉내 내 코드가 입력
# 순서에 기대지 못하게 하려는 것이다.
def make_rows() -> tuple[list[VisaRow], dict[str, tuple[str, ...]]]:
    rows, types = [], {}
    for category, (train_n, train_a, test_n, test_a) in SIZES.items():
        for i in range(train_n + test_n):
            split = "train" if i < train_n else "test"
            rows.append(VisaRow(category, split, "normal", f"{category}/Data/Images/Normal/{i:04d}.JPG", ""))
        for i in range(train_a + test_a):
            split = "train" if i < train_a else "test"
            image = f"{category}/Data/Images/Anomaly/{i:03d}.JPG"
            rows.append(
                VisaRow(category, split, "anomaly", image, f"{category}/Data/Masks/Anomaly/{i:03d}.png")
            )
            types[image] = ("bent", "melt") if i % 3 == 0 else ("scratch",)
    # The official CSV is not sorted: shuffle deterministically so the code cannot rely on input order.
    rows.sort(key=lambda r: hashlib.md5(r.image.encode()).hexdigest())
    return rows, types


# 합성 분할로 만든 매니페스트를 모듈 안 테스트들이 나눠 쓴다.
@pytest.fixture(scope="module")
def manifest() -> list[ManifestRow]:
    rows, types = make_rows()
    return splits.build_manifest(rows, types)


# 기대값 계산용 SHA-256 16진수(splits.path_key와 따로 계산해 비교한다).
def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# 정렬 키가 경로의 SHA-256, salt가 있으면 'salt:경로'의 SHA-256인지 확인한다. salt "0"은 salt 없음과 다른 키를
# 내야 한다(시드 0의 라벨 순서가 겹 순서와 같아지지 않게).
def test_path_key_is_sha256_of_path_or_salted_path():
    path = "candle/Data/Images/Normal/0836.JPG"
    assert splits.path_key(path) == sha(path)
    assert splits.path_key(path, salt="2") == sha(f"2:{path}")
    assert splits.path_key(path, salt="0") == sha(f"0:{path}") != splits.path_key(path)
    assert splits.path_key(path, "") == splits.path_key(path, salt="") == sha(path)
    assert splits.path_key("a", "0") == sha("0:a")


# 문자열이 아닌 salt(정수, None, False, 실수, 바이트)는 TypeError로 거절한다. 정수 시드 0은 거짓 값이라
# '없음'으로 읽히면 겹을 정하는 salt 없는 키가 조용히 나오기 때문이다(기존 주석). 경로가 문자열이 아니어도
# 거절한다.
@pytest.mark.parametrize("salt", [0, 1, None, False, 0.0, b"0"])
def test_path_key_rejects_salts_that_are_not_strings(salt):
    # An integer seed 0 is falsy: it must not silently give the unsalted key used for folds.
    with pytest.raises(TypeError, match="salt"):
        splits.path_key("candle/Data/Images/Anomaly/000.JPG", salt)
    with pytest.raises(TypeError):
        splits.path_key(None)


# 시드 0으로 뽑은 라벨 순서가 '0:경로' 키 순서이고, salt 없는 키 순서와 다르다는 것을 실제 label_subset으로
# 확인한다.
def test_label_subset_with_seed_zero_is_salted(manifest):
    pool = [r.image for r in manifest if r.category == "pcb1" and r.role == "label_pool"]
    got = [r.image for r in splits.label_subset(manifest, "pcb1", 40, 0)]
    assert got == sorted(pool, key=lambda image: sha(f"0:{image}")) != sorted(pool, key=sha)


# 분할 규칙의 상수(겹 5개, dev 결함 20장, k = 5·10·20·40, 시드 0·1·2, 역할 이름과 봉인 역할)가 문서에 적힌
# 값과 같은지 고정한다. 이 값이 바뀌면 커밋된 매니페스트와 결과가 모두 달라진다.
def test_constants():
    assert splits.N_FOLDS == 5 and splits.N_DEV_DEFECTS == 20
    assert splits.K_VALUES == (5, 10, 20, 40) and splits.SEEDS == (0, 1, 2)
    assert splits.ROLES == ("pool_normal", "dev_defect", "label_pool", "test_normal", "test_defect")
    assert splits.SEALED_ROLES == ("test_normal", "test_defect")


# 범주별 역할 장수: 학습 정상은 모두 정상 풀, 학습 결함은 앞 20장(모자라면 전부)이 dev 결함이고 나머지가 라벨
# 풀, 공식 test는 봉인 테스트다. 범주 순서는 입력 순서가 아니라 CATEGORIES 순서다.
def test_role_counts(manifest):
    counts = splits.role_counts(manifest)
    assert list(counts) == ["candle", "capsules", "pcb1"]  # CATEGORIES order, not input order
    for category, (train_n, train_a, test_n, test_a) in SIZES.items():
        assert counts[category] == {
            "pool_normal": train_n,
            "dev_defect": min(20, train_a),
            "label_pool": train_a - min(20, train_a),
            "test_normal": test_n,
            "test_defect": test_a,
        }
    assert counts["capsules"]["label_pool"] == 7


# 모든 이미지가 정확히 한 번, 한 역할에만 들어가는지 확인한다. 공식 train과 test가 섞이지 않고(공식 test만
# 봉인 역할), 정상 라벨은 정상 풀이나 테스트 정상에만 간다. 학습 데이터와 봉인 테스트가 겹치는 누수를 막는
# 가장 기본적인 검사다.
def test_no_image_appears_in_two_roles(manifest):
    rows, _ = make_rows()
    images = [r.image for r in manifest]
    assert len(images) == len(set(images)) == len(rows)
    roles = defaultdict(set)
    for r in manifest:
        roles[r.image].add(r.role)
    assert all(len(found) == 1 for found in roles.values())
    # Official train and test never mix.
    official = {r.image: r.split for r in rows}
    for r in manifest:
        assert (official[r.image] == "test") == r.role.startswith("test_")
        assert (r.label == "normal") == (r.role in ("pool_normal", "test_normal"))


# 정상 풀의 겹 번호가 SHA-256 순위를 5로 나눈 나머지이고, 겹 크기 차이가 1 이하인지 본다. 정상 풀이 아닌 행의
# 겹은 -1이다.
def test_folds_follow_hash_rank_and_are_balanced(manifest):
    for category in SIZES:
        pool = [r for r in manifest if r.category == category and r.role == "pool_normal"]
        ranked = sorted(pool, key=lambda r: sha(r.image))
        assert [r.fold for r in ranked] == [i % 5 for i in range(len(ranked))]
        sizes = Counter(r.fold for r in pool)
        assert set(sizes) == {0, 1, 2, 3, 4}
        assert max(sizes.values()) - min(sizes.values()) <= 1
    assert all(r.fold == -1 for r in manifest if r.role != "pool_normal")


# dev 결함이 학습 결함을 SHA-256 순으로 정렬한 앞 20장이고, 나머지가 라벨 풀인지 확인한다. 둘이 겹치지 않아야
# 지도 학습의 검증 세트와 학습 세트가 섞이지 않는다.
def test_dev_defects_are_the_first_twenty_by_hash(manifest):
    rows, _ = make_rows()
    for category in SIZES:
        train_defects = [
            r.image for r in rows if r.category == category and r.split == "train" and r.label == "anomaly"
        ]
        ranked = sorted(train_defects, key=sha)
        dev = {r.image for r in manifest if r.category == category and r.role == "dev_defect"}
        pool = {r.image for r in manifest if r.category == category and r.role == "label_pool"}
        assert dev == set(ranked[:20])
        assert pool == set(ranked[20:])


# 매니페스트 행 순서가 범주(CATEGORIES 순) → 역할 → 경로이고, 결함 행은 마스크 경로와 '|'로 이은 결함 유형을,
# 정상 행은 빈 값을 가지는지 본다. 순서가 고정돼야 다시 만든 CSV가 커밋된 파일과 줄 단위로 같다.
def test_manifest_order_and_fields(manifest):
    def order(r: ManifestRow):
        return (CATEGORIES.index(r.category), splits.ROLES.index(r.role), r.image)

    assert manifest == sorted(manifest, key=order)
    _, types = make_rows()
    for r in manifest:
        if r.label == "anomaly":
            assert r.mask == r.image.replace("Images", "Masks").replace(".JPG", ".png")
            assert r.defect_types == "|".join(types[r.image])
        else:
            assert r.mask == "" and r.defect_types == ""
    assert any("|" in r.defect_types for r in manifest)


# 입력 행을 뒤집어 넣어도 같은 매니페스트가 나와야 한다(결정적 분할).
def test_build_is_independent_of_input_order(manifest):
    rows, types = make_rows()
    assert splits.build_manifest(rows[::-1], types) == manifest


# 같은 이미지가 두 번 나오면 두 역할에 동시에 들어갈 수 있으므로 처음부터 거절한다.
def test_build_rejects_repeated_images():
    rows, types = make_rows()
    with pytest.raises(ValueError, match="more than once"):
        splits.build_manifest([*rows, rows[0]], types)


# 매니페스트를 쓰고 다시 읽으면 같은 행이 나오고, 파일은 정해진 헤더와 LF 줄 끝만 쓴다(Windows에서 다시
# 만들어도 커밋된 파일과 줄 끝 차이가 생기지 않게).
def test_manifest_round_trip_with_lf(manifest, tmp_path):
    path = tmp_path / "manifests" / "visa.csv"
    splits.write_manifest(manifest, path)
    raw = path.read_bytes()
    assert b"\r" not in raw
    assert raw.startswith(b"image,mask,category,label,role,fold,defect_types\n")
    assert raw.count(b"\n") == len(manifest) + 1
    assert splits.read_manifest(path) == manifest


# 헤더가 다른 CSV는 조용히 엉뚱한 열을 읽지 않고 ValueError로 거절한다.
def test_read_manifest_rejects_other_headers(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("image,mask\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="header"):
        splits.read_manifest(path)


# 지도 학습의 k장 규칙: 시드마다 '시드:경로' 키 순서의 앞 k장이고, k를 늘려도 앞의 표본은 그대로인 포함
# 관계이며, 모두 그 범주의 라벨 풀에서만 나온다. 시드가 다르면 순서가 다르고, k = 0은 빈 목록이다. 포함 관계
# 덕분에 k 사이의 비교에 표본 교체의 영향이 섞이지 않는다(docs/experiments.md 분할 절).
def test_label_subset_is_nested_seeded_and_inside_the_label_pool(manifest):
    pool = {r.image for r in manifest if r.category == "pcb1" and r.role == "label_pool"}
    assert len(pool) == 40
    for seed in splits.SEEDS:
        ordered = sorted(pool, key=lambda image: sha(f"{seed}:{image}"))
        previous: list[str] = []
        for k in splits.K_VALUES:
            subset = splits.label_subset(manifest, "pcb1", k, seed)
            images = [r.image for r in subset]
            assert images == ordered[:k]
            assert images[: len(previous)] == previous
            assert all(r.role == "label_pool" and r.category == "pcb1" for r in subset)
            previous = images
    first = [[r.image for r in splits.label_subset(manifest, "pcb1", 5, seed)] for seed in splits.SEEDS]
    assert first[0] != first[1] != first[2]
    assert splits.label_subset(manifest, "pcb1", 40, 0) != splits.label_subset(manifest, "pcb1", 40, 1)
    assert splits.label_subset(manifest, "pcb1", 0, 0) == []


# 라벨 풀보다 큰 k(capsules는 7장)나 음수 k는 ValueError다.
def test_label_subset_rejects_k_beyond_the_pool(manifest):
    with pytest.raises(ValueError):
        splits.label_subset(manifest, "capsules", 10, 0)  # only 7 in this pool
    with pytest.raises(ValueError):
        splits.label_subset(manifest, "pcb1", -1, 0)


# 절차별 정상 풀의 겹: dev는 겹 0을 평가용으로 빼고 1~4, test는 0~4 전부다. 홀드아웃 보정에서 빼는 겹은 풀의
# 첫 겹(dev 1, test 0)이다(docs/experiments.md 1단계). 모르는 절차 이름은 ValueError다.
def test_pool_and_holdout_folds():
    assert splits.pool_folds("dev") == (1, 2, 3, 4)
    assert splits.pool_folds("test") == (0, 1, 2, 3, 4)
    assert splits.holdout_fold("dev") == 1
    assert splits.holdout_fold("test") == 0
    for fn in (splits.pool_folds, splits.holdout_fold):
        with pytest.raises(ValueError):
            fn("val")


# dev 절차의 세 부분: 정상 풀은 겹 1~4, 평가 정상은 겹 0, 평가 결함은 dev 결함이다. 풀과 평가 정상이 겹치지
# 않고, dev 절차에서는 봉인 테스트나 라벨 풀이 나올 수 없다.
def test_select_dev_protocol(manifest):
    pool = splits.select(manifest, protocol="dev", part="pool_normal")
    eval_normal = splits.select(manifest, protocol="dev", part="eval_normal")
    eval_defect = splits.select(manifest, protocol="dev", part="eval_defect")
    assert pool == [r for r in manifest if r.role == "pool_normal" and r.fold in (1, 2, 3, 4)]
    assert eval_normal == [r for r in manifest if r.role == "pool_normal" and r.fold == 0]
    assert eval_defect == [r for r in manifest if r.role == "dev_defect"]
    assert not {r.image for r in pool} & {r.image for r in eval_normal}
    # Nothing from the sealed test set or the label pool can come out of the dev protocol.
    assert {r.role for r in pool + eval_normal + eval_defect} == {"pool_normal", "dev_defect"}


# test 절차의 정상 풀은 허락 없이 받을 수 있지만, 평가 정상·결함(봉인 테스트)은 allow_test 없이, 범주를
# 좁혀도, False를 줘도 SealedTestError다. 이 예외는 RuntimeError의 하위 클래스다.
def test_select_test_protocol_is_sealed(manifest):
    pool = splits.select(manifest, protocol="test", part="pool_normal")
    assert pool == [r for r in manifest if r.role == "pool_normal"]
    for part in ("eval_normal", "eval_defect"):
        with pytest.raises(SealedTestError):
            splits.select(manifest, protocol="test", part=part)
        with pytest.raises(SealedTestError):
            splits.select(manifest, protocol="test", part=part, category="pcb1")
        with pytest.raises(SealedTestError):
            splits.select(manifest, protocol="test", part=part, allow_test=False)
    assert issubclass(SealedTestError, RuntimeError)


# 봉인은 정확히 True 객체일 때만 열린다. 설정 파일이나 환경 변수에서 온 문자열("false", "0", "true"), 정수 1,
# 실수, None, 리스트 같은 값은 참·거짓과 상관없이 모두 거절한다. 봉인이 없는 부분(정상 풀, dev 평가)은 이 값과
# 무관하게 동작한다.
@pytest.mark.parametrize(
    "allow_test", ["false", "0", "no", "true", "True", 1, 1.0, None, 0, [True], object()]
)
def test_seal_opens_only_for_the_literal_true(manifest, allow_test):
    # A truthy value from a config file or an environment variable ("false", "0") must not open it.
    for part in ("eval_normal", "eval_defect"):
        with pytest.raises(SealedTestError):
            splits.select(manifest, protocol="test", part=part, allow_test=allow_test)
        with pytest.raises(SealedTestError):
            splits.select(manifest, protocol="test", part=part, category="pcb1", allow_test=allow_test)
    # The parts that are not sealed do not care.
    assert splits.select(manifest, protocol="test", part="pool_normal", allow_test=allow_test)
    assert splits.select(manifest, protocol="dev", part="eval_normal", allow_test=allow_test)


# allow_test=True면 봉인 테스트 정상·결함이 나오고, dev 절차의 결과는 allow_test에 따라 바뀌지 않는다.
def test_select_test_protocol_with_allow_test(manifest):
    eval_normal = splits.select(manifest, protocol="test", part="eval_normal", allow_test=True)
    eval_defect = splits.select(manifest, protocol="test", part="eval_defect", allow_test=True)
    assert eval_normal == [r for r in manifest if r.role == "test_normal"]
    assert eval_defect == [r for r in manifest if r.role == "test_defect"]
    # allow_test does not change what the dev protocol returns.
    assert splits.select(manifest, protocol="dev", part="eval_defect", allow_test=True) == [
        r for r in manifest if r.role == "dev_defect"
    ]


# 범주를 좁혀 고른 결과를 범주 순으로 이으면 전체 결과와 같고, 순서는 매니페스트 순서를 그대로 따른다. 없는
# 범주는 빈 목록이다.
def test_select_category_filter_keeps_manifest_order(manifest):
    for protocol in ("dev", "test"):
        for part in ("pool_normal", "eval_normal", "eval_defect"):
            everything = splits.select(manifest, protocol=protocol, part=part, allow_test=True)
            assert everything == sorted(everything, key=manifest.index)
            per_category = [
                splits.select(manifest, protocol=protocol, part=part, category=c, allow_test=True)
                for c in ("candle", "capsules", "pcb1")
            ]
            assert all(r.category == "pcb1" for r in per_category[2]) and per_category[2]
            assert [r for rows in per_category for r in rows] == everything
    assert splits.select(manifest, protocol="dev", part="pool_normal", category="fryum") == []


# 모르는 절차·부분 이름은 ValueError다. test 절차에서도 모르는 부분은 봉인 예외가 아니라 ValueError로 먼저
# 거절한다.
def test_select_rejects_unknown_names(manifest):
    with pytest.raises(ValueError):
        splits.select(manifest, protocol="val", part="pool_normal")
    with pytest.raises(ValueError):
        splits.select(manifest, protocol="dev", part="test_normal")
    # An unknown part is a ValueError even where the seal would otherwise apply.
    with pytest.raises(ValueError):
        splits.select(manifest, protocol="test", part="label_pool")


# VisA tar를 흉내 낸 작은 tar를 만든다: 공식 분할 CSV(split_csv/2cls_highshot.csv)와 범주별
# image_anno.csv(결함 유형을 따옴표 안에 쉼표로 이음)만 넣고, 실제 파일처럼 CRLF 줄 끝을 쓴다.
def make_tar(path, rows: list[VisaRow], types: dict[str, tuple[str, ...]]) -> None:
    split = "object,split,label,image,mask\r\n" + "".join(
        f"{r.category},{r.split},{r.label},{r.image},{r.mask}\r\n" for r in rows
    )
    members = {"split_csv/2cls_highshot.csv": split}
    for category in SIZES:
        lines = ["image,label,mask"]
        for r in rows:
            if r.category == category:
                label = '"' + ",".join(types[r.image]) + '"' if r.label == "anomaly" else "normal"
                lines.append(f"{r.image},{label},{r.mask}")
        members[f"{category}/image_anno.csv"] = "\r\n".join(lines) + "\r\n"
    with tarfile.open(path, "w") as tar:
        for name, text in members.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


# main이 tar에서 매니페스트를 만들어 쓰고 범주별 장수 표를 출력하는지 확인한다. 합성 분할은 공식 장수와
# 다르므로 --no-verify로 검사를 끈다.
def test_main_writes_manifest_from_tar(manifest, tmp_path, capsys):
    rows, types = make_rows()
    tar_path = tmp_path / "visa.tar"
    make_tar(tar_path, rows, types)
    out = tmp_path / "out" / "visa.csv"
    assert splits.main(["--tar", str(tar_path), "--out", str(out), "--no-verify"]) == 0
    assert splits.read_manifest(out) == manifest
    printed = capsys.readouterr().out
    assert "pool_normal" in printed and "capsules" in printed and "total" in printed


# 검사를 켜면 공식 장수(EXPECTED_TOTALS)와 다른 매니페스트는 쓰지 않고 1을 돌려주며 ERROR를 출력한다.
def test_main_refuses_to_write_when_counts_are_not_the_official_ones(tmp_path, capsys):
    rows, types = make_rows()
    tar_path = tmp_path / "visa.tar"
    make_tar(tar_path, rows, types)
    out = tmp_path / "visa.csv"
    assert splits.main(["--tar", str(tar_path), "--out", str(out)]) == 1
    assert not out.exists()
    assert "ERROR" in capsys.readouterr().out


# 커밋된 manifests/visa.csv(있을 때만)가 규칙을 지키는지 확인한다: LF 줄 끝, 역할별 공식 장수와 전체 10,821행,
# CATEGORIES 순서, 범주마다 dev 결함 20·라벨 풀 40·테스트 결함 40장, 정상 풀과 테스트 정상 장수 범위, 겹 크기
# 균형과 겹 0 장수(60~121장, dev 평가 정상), 모든 결함의 유형. 마지막으로 매니페스트의 경로만으로 다시 만들면
# 같은 배정이 나와, 파일이 규칙에서 나온 것임을 보인다.
def test_committed_manifest_follows_the_rules():
    """The committed manifest (when present) has the documented counts and reproduces from its paths."""
    if not paths.VISA_MANIFEST.exists():
        pytest.skip("manifests/visa.csv has not been built")
    manifest = splits.read_manifest(paths.VISA_MANIFEST)
    assert b"\r" not in paths.VISA_MANIFEST.read_bytes()
    assert Counter(r.role for r in manifest) == splits.EXPECTED_TOTALS
    assert len({r.image for r in manifest}) == len(manifest) == 10821
    counts = splits.role_counts(manifest)
    assert tuple(counts) == CATEGORIES
    for category, per_role in counts.items():
        assert (per_role["dev_defect"], per_role["label_pool"], per_role["test_defect"]) == (20, 40, 40)
        assert 300 <= per_role["pool_normal"] <= 604 and 200 <= per_role["test_normal"] <= 402
        folds = Counter(r.fold for r in manifest if r.category == category and r.role == "pool_normal")
        assert max(folds.values()) - min(folds.values()) <= 1
        assert 60 <= folds[0] <= 121
    assert all(r.defect_types for r in manifest if r.label == "anomaly")
    # Rebuilding from the manifest's own paths gives the same assignment.
    rows = [
        VisaRow(r.category, "test" if r.role.startswith("test_") else "train", r.label, r.image, r.mask)
        for r in manifest
    ]
    types = {r.image: tuple(r.defect_types.split("|")) for r in manifest if r.label == "anomaly"}
    assert splits.build_manifest(rows, types) == manifest
