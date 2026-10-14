# [파일 역할] VisA tar를 한 번 순서대로 읽어 리사이즈한 이미지·마스크를 큰 .npy 배열로 만드는 캐시
# 모듈(`defect_inspect.cache`)을 검증하는 테스트 묶음이다. 리사이즈 규칙, 캐시 만들기·읽기, 다른 크기 사이의
# 마스크 재사용, 봉인 테스트 읽기 장부 기록, 실패·중단·파일 잠금에서 기존 캐시를 지키는 동작을 다룬다.
# [왜 필요한가] 데이터 디스크가 HDD라 tar를 작은 파일로 풀어 매번 읽지 않고 한 번 순서대로 읽어 큰 배열로
# 만든다(docs/experiments.md 0단계의 캐시 항목). 모든 단계가 이 캐시에서 이미지를 꺼내므로 행 순서·리사이즈가
# 틀리면 모든 수치가 틀린다. 전처리 규칙(정사각형 BICUBIC, 마스크는 BOX 필터 후 0 초과면 결함,
# docs/experiments.md 전처리)과, 캐시 생성은 봉인 테스트 이미지를 옮기므로 만들 때마다 장부에 stage cache로
# 남긴다는 예외 규칙(규칙 변경 1)을 고정한다. 또 Windows는 메모리 맵으로 열린 파일을 바꾸지 못하므로,
# 실패하거나 중단돼도 '색인 파일이 다른 배열을 설명하는' 상태가 남지 않아야 한다.
# [설계] 실제 1.93GB tar 대신 작은 JPEG·PNG 몇 장으로 tar를 만들고, 구성원 순서를 매니페스트 순서와
# 다르게(SHA-256 순) 넣는다. 장부·디코더·os.rename·os.replace를 monkeypatch로 바꿔 실패 지점을 하나씩 만든다.

import hashlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# ledger·paths는 장부 경로와 commit을 가짜로 바꾸려고, ManifestRow·write_manifest는 작은 매니페스트를 만들려고
# 가져온다.
from defect_inspect import cache, ledger, paths
from defect_inspect.cache import MASK_SIZE, ImageCache, build_cache, resize_image, resize_mask
from defect_inspect.splits import ManifestRow, write_manifest

# 테스트용 이미지 한 변(실제는 256, 448 등). 마스크는 크기와 상관없이 늘 256×256이다(MASK_SIZE).
SIZE = 32


# cache.main은 봉인 테스트를 읽을 때 장부에 줄을 쓴다. 모든 테스트에서 장부 경로를 tmp_path로 돌리고
# git_commit을 고정값으로 바꿔, 실제 reports/test_ledger.jsonl에 테스트가 줄을 남기는 일이 절대 없게
# 한다(autouse).
@pytest.fixture(autouse=True)
def temporary_ledger(tmp_path, monkeypatch):
    """`cache.main`은 봉인 테스트 읽기를 기록한다: 어떤 테스트도 실제 장부에 써서는 안 된다."""
    path = tmp_path / "reports" / "test_ledger.jsonl"
    monkeypatch.setattr(paths, "TEST_LEDGER", path)
    monkeypatch.setattr(ledger, "git_commit", lambda root=None: "abc1234")
    return path


# seed마다 다른 작은 JPEG. 거친 잡음을 키워 부드럽게 만들어 JPEG 압축 손상이 작게 남게 한다(이미지마다 내용이
# 확실히 다르도록).
def jpeg_bytes(seed: int, width: int = 90, height: int = 60) -> bytes:
    rng = np.random.default_rng(seed)
    # 부드러운 내용(확대한 잡음)이라 JPEG 압축 손상이 작게 남는다.
    coarse = rng.integers(0, 256, size=(6, 9, 3), dtype=np.uint8)
    img = Image.fromarray(coarse).resize((width, height), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


# 8×8 결함 하나가 있는 PNG 마스크. VisA 마스크처럼 255가 아니라 클래스 번호(1~3)로 채워, '0보다 크면 결함'
# 규칙을 시험한다.
def mask_bytes(seed: int, width: int = 90, height: int = 60) -> bytes:
    rng = np.random.default_rng(seed)
    array = np.zeros((height, width), dtype=np.uint8)
    y, x = int(rng.integers(0, height - 8)), int(rng.integers(0, width - 8))
    array[y : y + 8, x : x + 8] = 1 + seed % 3  # 255가 아니라 클래스 번호
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


# 작은 VisA 흉내 tar와 매니페스트를 만든다. 정상 7장(짝수 번째는 학습 풀, 홀수는 봉인 테스트), 결함 4장(dev
# 결함과 봉인 테스트 결함이 섞임)과 마스크, 캐시가 건너뛰어야 할 관계없는 파일(image_anno.csv,
# LICENSE-DATASET)과 디렉터리 항목이 들어 있다. tar 안 순서를 이름의 SHA-256 순으로 섞어, 캐시가 tar 순서가
# 아니라 매니페스트 순서로 행을 놓는지 확인할 수 있다.
def make_dataset(tmp_path, n_normal: int = 7, n_defect: int = 4, prefix: str = ""):
    """멤버 순서가 매니페스트 순서와 다른 작은 tar와 파일 바이트들.

    `prefix`는 tar 안의 모든 멤버 이름 앞에 붙인다(매니페스트에는 붙이지 않는다).
    """
    files: dict[str, bytes] = {"pcb1/image_anno.csv": b"image,label,mask\n", "LICENSE-DATASET": b"x"}
    manifest: list[ManifestRow] = []
    for i in range(n_normal):
        image = f"pcb1/Data/Images/Normal/{i:04d}.JPG"
        files[image] = jpeg_bytes(i)
        role = "pool_normal" if i % 2 == 0 else "test_normal"
        manifest.append(ManifestRow(image, "", "pcb1", "normal", role, i % 5 if i % 2 == 0 else -1, ""))
    for i in range(n_defect):
        image, mask = f"pcb1/Data/Images/Anomaly/{i:03d}.JPG", f"pcb1/Data/Masks/Anomaly/{i:03d}.png"
        files[image] = jpeg_bytes(100 + i)
        files[mask] = mask_bytes(i)
        role = "dev_defect" if i % 2 == 0 else "test_defect"
        manifest.append(ManifestRow(image, mask, "pcb1", "anomaly", role, -1, "melt"))
    tar_path = tmp_path / "visa.tar"
    with tarfile.open(tar_path, "w") as tar:
        directory = tarfile.TarInfo("pcb1")
        directory.type = tarfile.DIRTYPE
        tar.addfile(directory)
        for name in sorted(files, key=lambda n: hashlib.sha256(n.encode()).hexdigest()):
            info = tarfile.TarInfo(prefix + name)
            info.size = len(files[name])
            tar.addfile(info, io.BytesIO(files[name]))
    return tar_path, manifest, files


# 캐시 폴더의 파일 이름 → 내용. 실패 전후로 비교해 '아무것도 바뀌지 않았다'를 확인한다.
def snapshot(out_dir) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in out_dir.iterdir()}


# 남아 있는 모든 색인 파일(visa_*_index.csv)이 옆 배열의 행을 정확히 설명하는지 원본 바이트에서 다시 계산해
# 확인한다. 실패·중단 테스트의 핵심 불변식인 '색인 파일이 있으면 그 배열은 믿을 수 있다'를 검사한다.
def check_indexes_describe_their_arrays(out_dir, manifest, files) -> None:
    """남아 있는 모든 색인 파일 옆에는 정확히 그 행들을 담은 배열이 있다(어긋난 옛 짝이 없다)."""
    mask_of = {r.image: r.mask for r in manifest}
    for index_file in out_dir.glob("visa_*_index.csv"):
        listed = index_file.read_text(encoding="utf-8").splitlines()
        if index_file.name == f"visa_masks_{MASK_SIZE}_index.csv":
            array = np.load(out_dir / f"visa_masks_{MASK_SIZE}.npy")
            want = [
                expected_mask(files[mask_of[image]])
                if mask_of[image]
                else np.zeros((MASK_SIZE, MASK_SIZE), np.uint8)
                for image in listed
            ]
        else:
            size = int(index_file.name.split("_")[1])
            array = np.load(out_dir / f"visa_{size}_images.npy")
            want = [expected_image(files[image], size) for image in listed]
        assert len(array) == len(listed), index_file.name
        for row, expected in zip(array, want, strict=True):
            assert np.array_equal(row, expected), index_file.name


# 원본 바이트를 캐시와 같은 함수로 리사이즈한 기대값.
def expected_image(data: bytes, size: int) -> np.ndarray:
    return resize_image(Image.open(io.BytesIO(data)), size)


def expected_mask(data: bytes) -> np.ndarray:
    return resize_mask(Image.open(io.BytesIO(data)))


# 확인: 가로세로 비율을 무시하고 정사각형 RGB uint8로 PIL BICUBIC 리사이즈를 하며, 줄일 때 이웃 픽셀을
# 평균한다(안티앨리어싱). 1픽셀 줄무늬를 1/4로 줄이면 흑백이 번갈아 남지 않고 회색으로 고르게 나온다(전처리
# 규칙).
def test_resize_image_is_square_rgb_bicubic():
    rng = np.random.default_rng(0)
    source = Image.fromarray(rng.integers(0, 256, size=(40, 70, 3), dtype=np.uint8))
    out = resize_image(source, 16)
    assert out.shape == (16, 16, 3) and out.dtype == np.uint8
    assert np.array_equal(out, np.asarray(source.resize((16, 16), Image.Resampling.BICUBIC)))
    # 줄일 때 원본 픽셀 하나를 고르지 않고 이웃 픽셀을 평균한다(안티앨리어싱).
    stripes = np.zeros((64, 64, 3), dtype=np.uint8)
    stripes[:, ::2] = 255
    small = resize_image(Image.fromarray(stripes), 16)
    assert 100 < small.mean() < 155 and small.std() < 10


# 확인: 흑백(L)과 RGBA 이미지도 RGB로 바꿔 3채널로 내보낸다. 투명도 채널은 버린다.
def test_resize_image_converts_other_modes_to_rgb():
    grey = Image.fromarray(np.full((20, 30), 77, dtype=np.uint8))
    out = resize_image(grey, 8)
    assert out.shape == (8, 8, 3) and (out == 77).all()
    rgba = Image.new("RGBA", (30, 20), (10, 20, 30, 0))
    assert (resize_image(rgba, 8) == np.array([10, 20, 30], dtype=np.uint8)).all()


# 확인: 원본 크기(1500×1100) 마스크의 결함 픽셀 하나가 모서리·가운데 어디에 있든 256 격자에서 정확히 한 칸으로
# 남고, 그 칸은 원래 픽셀의 중심을 담은 상자다.
# 왜: 전처리 규칙은 '작은 결함이 축소로 사라지지 않게' BOX 필터 후 0보다 크면 결함으로 한다. 평균 후 반올림
# 같은 다른 방식이면 1픽셀 결함이 사라져 픽셀 지표가 달라진다.
def test_resize_mask_keeps_single_pixel_defects():
    height, width = 1100, 1500
    for y, x in [(0, 0), (1099, 1499), (0, 1499), (547, 733), (5, 6), (1098, 3)]:
        source = np.zeros((height, width), dtype=np.uint8)
        source[y, x] = 3  # 255가 아니라 클래스 번호
        out = resize_mask(Image.fromarray(source))
        assert out.shape == (MASK_SIZE, MASK_SIZE) and out.dtype == np.uint8
        assert set(np.unique(out)) == {0, 1}
        assert out.sum() == 1
        # 원본 픽셀이 속하는 상자는 그 픽셀의 중심을 담은 상자다.
        assert out[int((y + 0.5) * MASK_SIZE / height), int((x + 0.5) * MASK_SIZE / width)] == 1


# 확인: 정수배 축소에서는 '상자 안에 결함 픽셀이 하나라도 있으면 1'(max 풀링)과 같다. 결함이 없으면 0, 전부
# 결함이면 1이다.
def test_resize_mask_matches_any_pooling_on_integer_factors():
    rng = np.random.default_rng(1)
    source = (rng.random((64, 96)) < 0.02).astype(np.uint8) * rng.integers(
        1, 5, size=(64, 96), dtype=np.uint8
    )
    out = resize_mask(Image.fromarray(source), size=16)
    blocks = source.reshape(16, 4, 16, 6).max(axis=(1, 3)) > 0
    assert np.array_equal(out, blocks.astype(np.uint8))
    assert resize_mask(Image.fromarray(np.zeros((64, 96), dtype=np.uint8)), size=16).sum() == 0
    assert resize_mask(Image.fromarray(np.full((64, 96), 1, dtype=np.uint8)), size=16).all()


# 확인: 여러 채널 마스크는 ValueError다. 마스크 형식이 예상과 다르면 조용히 한 채널만 쓰지 않는다.
def test_resize_mask_rejects_multichannel_masks():
    with pytest.raises(ValueError):
        resize_mask(Image.new("RGB", (8, 8)))


# 확인: 캐시 폴더에 이미지 배열·색인, 마스크 배열·색인 네 파일만 생기고(.part 없음), 색인은 LF 줄바꿈의
# 매니페스트 순서이며, 행마다 이미지가 원본을 리사이즈한 값과 같고, 결함 행 마스크는 0/1, 정상 행 마스크는
# 모두 0이다. 행이 서로 다른지도 확인해 모든 행에 같은 이미지가 들어가는 실수를 막는다.
def test_build_cache_files_and_contents(tmp_path):
    tar_path, manifest, files = make_dataset(tmp_path)
    out_dir = tmp_path / "cache"
    build_cache(tar_path, manifest, SIZE, out_dir, workers=3, progress=False)

    names = sorted(p.name for p in out_dir.iterdir())
    assert names == [
        f"visa_{SIZE}_images.npy",
        f"visa_{SIZE}_index.csv",
        "visa_masks_256.npy",
        "visa_masks_256_index.csv",
    ]
    index_raw = (out_dir / f"visa_{SIZE}_index.csv").read_bytes()
    assert b"\r" not in index_raw
    assert index_raw.decode("utf-8").splitlines() == [r.image for r in manifest]

    images = np.load(out_dir / f"visa_{SIZE}_images.npy")
    masks = np.load(out_dir / "visa_masks_256.npy")
    assert images.shape == (len(manifest), SIZE, SIZE, 3) and images.dtype == np.uint8
    assert masks.shape == (len(manifest), MASK_SIZE, MASK_SIZE) and masks.dtype == np.uint8
    for i, row in enumerate(manifest):
        assert np.array_equal(images[i], expected_image(files[row.image], SIZE))
        if row.mask:
            assert np.array_equal(masks[i], expected_mask(files[row.mask]))
            assert masks[i].any() and masks[i].max() == 1
        else:
            assert not masks[i].any()
    # 행끼리 정말 다르다(모든 행에 같은 이미지가 들어가는 실수를 막는다).
    assert len({images[i].tobytes() for i in range(len(manifest))}) == len(manifest)


# 확인: ImageCache가 요청한 행 순서 그대로(순서가 섞이고 같은 행이 반복돼도) 이미지·마스크를 돌려주고, 돌려준
# 것은 메모리 맵이 아니라 고칠 수 있는 사본이라 고쳐도 캐시가 바뀌지 않는다. 빈 요청은 빈 배열, 캐시에 없는
# 이미지는 'not in the cache' KeyError다.
def test_image_cache_returns_copies_in_row_order(tmp_path):
    tar_path, manifest, files = make_dataset(tmp_path)
    out_dir = tmp_path / "cache"
    build_cache(tar_path, manifest, SIZE, out_dir, workers=2, progress=False)
    store = ImageCache(out_dir, SIZE)
    assert store.size == SIZE and len(store) == len(manifest)

    rows = [manifest[9], manifest[0], manifest[10], manifest[3], manifest[9]]  # 순서 자유, 반복 허용
    images, masks = store.images(rows), store.masks(rows)
    assert images.shape == (5, SIZE, SIZE, 3) and images.dtype == np.uint8
    assert masks.shape == (5, MASK_SIZE, MASK_SIZE) and masks.dtype == np.uint8
    assert type(images) is np.ndarray and type(masks) is np.ndarray
    for i, row in enumerate(rows):
        assert np.array_equal(images[i], expected_image(files[row.image], SIZE))
        want = expected_mask(files[row.mask]) if row.mask else np.zeros((MASK_SIZE, MASK_SIZE), np.uint8)
        assert np.array_equal(masks[i], want)
    images[:] = 0  # 메모리 사본: 고칠 수 있고, 캐시 자체는 그대로다
    assert np.array_equal(store.images(rows[:1])[0], expected_image(files[rows[0].image], SIZE))

    assert store.images([]).shape == (0, SIZE, SIZE, 3)
    assert store.masks([]).shape == (0, MASK_SIZE, MASK_SIZE)
    with pytest.raises(KeyError, match="not in the cache"):
        store.images(
            [ManifestRow("pcb1/Data/Images/Normal/9999.JPG", "", "pcb1", "normal", "pool_normal", 0, "")]
        )
    store.close()


# 확인: 같은 매니페스트로 다른 크기(48) 캐시를 만들 때는 마스크를 다시 디코딩하지 않고(디코더를 실패하게
# 바꿔도 통과) 기존 마스크 파일을 그대로 쓴다. 두 크기 캐시가 모두 완성 상태이고 마스크가 같다.
# 왜: 마스크는 크기와 무관하게 256 격자이고(전처리 규칙), 6단계에서 252/280/294/308처럼 크기별 캐시를 여러 개
# 만들었다. 매번 다시 만들면 tar를 읽는 시간이 늘고 다른 크기 캐시와 어긋날 위험도 생긴다.
def test_masks_are_reused_by_other_sizes(tmp_path, monkeypatch):
    tar_path, manifest, files = make_dataset(tmp_path)
    out_dir = tmp_path / "cache"
    build_cache(tar_path, manifest, SIZE, out_dir, progress=False)
    assert cache.cache_is_complete(out_dir, SIZE, manifest)
    assert not cache.cache_is_complete(out_dir, 48, manifest)

    def no_masks(data):
        raise AssertionError("masks must not be decoded again")

    monkeypatch.setattr(cache, "_decode_mask", no_masks)
    masks_before = (out_dir / "visa_masks_256.npy").read_bytes()
    build_cache(tar_path, manifest, 48, out_dir, progress=False)
    assert (out_dir / "visa_masks_256.npy").read_bytes() == masks_before
    assert cache.cache_is_complete(out_dir, 48, manifest) and cache.cache_is_complete(out_dir, SIZE, manifest)

    big, small = ImageCache(out_dir, 48), ImageCache(out_dir, SIZE)
    assert big.images(manifest).shape == (len(manifest), 48, 48, 3)
    assert np.array_equal(big.images(manifest[:2])[1], expected_image(files[manifest[1].image], 48))
    assert np.array_equal(big.masks(manifest), small.masks(manifest))
    big.close()
    small.close()


# 확인: 매니페스트 행 순서가 바뀌면 기존 캐시는 완성이 아니고, 새 크기 캐시를 만들 때 마스크도 새 순서로 다시
# 만든다. 그러면 옛 크기 캐시는 마스크 순서와 어긋나므로 열 때 'different manifest' ValueError로 거부한다(틀린
# 마스크를 돌려주는 대신).
def test_masks_are_rebuilt_when_the_manifest_order_changes(tmp_path):
    tar_path, manifest, files = make_dataset(tmp_path)
    out_dir = tmp_path / "cache"
    build_cache(tar_path, manifest, SIZE, out_dir, progress=False)
    reordered = manifest[::-1]
    assert not cache.cache_is_complete(out_dir, SIZE, reordered)
    build_cache(tar_path, reordered, 48, out_dir, progress=False)

    store = ImageCache(out_dir, 48)
    assert np.array_equal(store.masks(reordered[:1])[0], expected_mask(files[reordered[0].mask]))
    store.close()
    # 옛 크기 캐시는 이제 마스크 순서와 어긋나고, 틀린 마스크를 돌려주는 대신 그렇다고 알린다.
    with pytest.raises(ValueError, match="different manifest"):
        ImageCache(out_dir, SIZE)


# 확인: tar에 없는 이미지가 매니페스트에 있으면 FileNotFoundError(그 이름 포함)를 내고 캐시 폴더에 아무
# 파일(.part 포함)도 남기지 않는다. 그 뒤 ImageCache를 열면 파일이 없다는 오류가 난다.
def test_failed_build_leaves_no_finished_files(tmp_path):
    tar_path, manifest, _ = make_dataset(tmp_path)
    out_dir = tmp_path / "cache"
    ghost = ManifestRow("pcb1/Data/Images/Normal/7777.JPG", "", "pcb1", "normal", "pool_normal", 0, "")
    with pytest.raises(FileNotFoundError, match="7777"):
        build_cache(tar_path, [*manifest, ghost], SIZE, out_dir, progress=False)
    assert list(out_dir.iterdir()) == []
    with pytest.raises(FileNotFoundError):
        ImageCache(out_dir, SIZE)


# 확인: 다시 만드는 도중 Ctrl-C(KeyboardInterrupt)가 나면 기존 캐시 파일이 바이트 단위로 그대로 남는다. 새
# 파일은 .part 이름으로만 쓰고 끝에서 바꾸기 때문이다.
def test_interrupted_rebuild_keeps_the_previous_cache_usable(tmp_path, monkeypatch):
    tar_path, manifest, files = make_dataset(tmp_path)
    out_dir = tmp_path / "cache"
    build_cache(tar_path, manifest, SIZE, out_dir, progress=False)
    before = {p.name: p.read_bytes() for p in out_dir.iterdir()}

    calls = []

    def broken(data, size):
        calls.append(size)
        raise KeyboardInterrupt

    monkeypatch.setattr(cache, "_decode_image", broken)
    with pytest.raises(KeyboardInterrupt):
        build_cache(tar_path, manifest, SIZE, out_dir, progress=False)
    assert calls
    assert {p.name: p.read_bytes() for p in out_dir.iterdir()} == before


# 확인: 같은 이미지가 두 번 나오는 매니페스트와 빈 매니페스트는 tar를 읽기 전에 ValueError다(행 번호가
# 겹치거나 빈 배열이 만들어지지 않게).
def test_build_rejects_bad_manifests(tmp_path):
    tar_path, manifest, _ = make_dataset(tmp_path)
    with pytest.raises(ValueError, match="more than once"):
        build_cache(tar_path, [*manifest, manifest[0]], SIZE, tmp_path / "cache", progress=False)
    with pytest.raises(ValueError, match="empty"):
        build_cache(tar_path, [], SIZE, tmp_path / "cache", progress=False)


# 확인: 진행 출력이 구성원 수(이미지 11 + 마스크 4 = 15)를 끝까지 센다. 수 분 걸리는 실제 캐시 생성에서 멈춤
# 여부를 로그로 볼 수 있다.
def test_progress_lines(tmp_path, capsys):
    tar_path, manifest, _ = make_dataset(tmp_path)
    build_cache(tar_path, manifest, SIZE, tmp_path / "cache", workers=1)
    out = capsys.readouterr().out
    assert f"[cache {SIZE}] 15/15 members" in out  # 11 images + 4 masks


# 확인: CLI는 한 번 만들고, 완성된 캐시가 있으면 build_cache를 부르지 않고 'already complete'라고만 한다.
# --force일 때만 다시 만든다(build_cache를 실패하는 가짜로 바꿔 호출 여부를 본다).
def test_main_builds_once_and_skips_a_complete_cache(tmp_path, capsys, monkeypatch):
    from defect_inspect.splits import write_manifest

    tar_path, manifest, _ = make_dataset(tmp_path)
    manifest_path = tmp_path / "visa.csv"
    write_manifest(manifest, manifest_path)
    out_dir = tmp_path / "cache"
    argv = [
        "--size",
        str(SIZE),
        "--tar",
        str(tar_path),
        "--manifest",
        str(manifest_path),
        "--out",
        str(out_dir),
    ]
    assert cache.main(argv) == 0
    assert cache.cache_is_complete(out_dir, SIZE, manifest)
    capsys.readouterr()

    def no_build(*args, **kwargs):
        raise AssertionError("a complete cache must not be rebuilt without --force")

    monkeypatch.setattr(cache, "build_cache", no_build)
    assert cache.main(argv) == 0
    assert "already complete" in capsys.readouterr().out
    with pytest.raises(AssertionError):
        cache.main([*argv, "--force"])


# 매니페스트를 CSV로 쓰고 cache.main에 넘길 인자 목록을 만든다.
def main_argv(tmp_path, manifest, *extra: str, size: int = SIZE, name: str = "visa.csv") -> list[str]:
    manifest_path = tmp_path / name
    write_manifest(manifest, manifest_path)
    tar, out = str(tmp_path / "visa.tar"), str(tmp_path / "cache")
    return ["--size", str(size), "--tar", tar, "--manifest", str(manifest_path), "--out", out, *extra]


# 장부(JSON Lines)의 줄들. 파일이 없으면 빈 목록.
def ledger_entries(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# 확인: 매니페스트에 봉인 테스트 행이 있으면 CLI가 장부에 stage cache, config cache-<크기>, 설명 한 줄을
# 남기고, 그 줄은 build_cache가 첫 테스트 이미지를 읽기 '전에' 쓰인다. 완성된 캐시를 건너뛸 때는 읽지 않으므로
# 기록하지 않고, --force나 다른 크기는 다시 읽으므로 줄이 하나씩 늘어난다.
# 왜: 봉인 테스트를 읽을 때마다 시각·커밋·단계를 남긴다는 규칙에서 캐시 생성은 예외로 허용하되 만들 때마다
# 장부에 남긴다(docs/experiments.md 봉인 테스트, 규칙 변경 1).
def test_main_records_the_test_set_read_in_the_ledger(tmp_path, capsys, monkeypatch, temporary_ledger):
    _, manifest, _ = make_dataset(tmp_path)
    assert {"test_normal", "test_defect"} <= {r.role for r in manifest}
    argv = main_argv(tmp_path, manifest)
    real_build = cache.build_cache
    recorded_before_build = []

    def watching_build(*args, **kwargs):
        recorded_before_build.append(len(ledger_entries(temporary_ledger)))
        return real_build(*args, **kwargs)

    monkeypatch.setattr(cache, "build_cache", watching_build)
    assert cache.main(argv) == 0
    expected = {
        "commit": "abc1234",
        "stage": "cache",
        "config": f"cache-{SIZE}",
        "note": "resize cache build: pixels copied into arrays, no scores or metrics",
    }
    entries = ledger_entries(temporary_ledger)
    assert len(entries) == 1
    assert {k: v for k, v in entries[0].items() if k != "time"} == expected
    assert recorded_before_build == [1]  # 첫 테스트 이미지를 읽기 전에 썼다
    assert str(temporary_ledger) in capsys.readouterr().out

    # 완성된 캐시는 건너뛴다: 아무것도 읽지 않고 아무것도 기록하지 않는다.
    assert cache.main(argv) == 0
    assert len(ledger_entries(temporary_ledger)) == 1 and recorded_before_build == [1]
    # --force는 테스트 이미지를 다시 읽으므로 둘째 줄이 생긴다.
    assert cache.main([*argv, "--force"]) == 0
    assert len(ledger_entries(temporary_ledger)) == 2 and recorded_before_build == [1, 2]
    # 다른 크기는 또 한 번의 읽기다.
    assert cache.main(main_argv(tmp_path, manifest, size=48)) == 0
    assert [e["config"] for e in ledger_entries(temporary_ledger)] == [f"cache-{SIZE}"] * 2 + ["cache-48"]


# 확인: 봉인 테스트 행이 없는 매니페스트로 만들면 장부 파일이 생기지 않는다(읽지 않은 것을 기록하지 않는다).
def test_main_records_nothing_without_test_rows(tmp_path, temporary_ledger):
    _, manifest, _ = make_dataset(tmp_path)
    open_rows = [r for r in manifest if not r.role.startswith("test_")]
    assert 0 < len(open_rows) < len(manifest)
    assert cache.main(main_argv(tmp_path, open_rows)) == 0
    assert cache.cache_is_complete(tmp_path / "cache", SIZE, open_rows)
    assert not temporary_ledger.exists()


# 확인: 테스트 정상만, 또는 테스트 결함만 있어도 봉인 테스트를 읽는 것이므로 한 줄을 남긴다.
@pytest.mark.parametrize("role", ["test_normal", "test_defect"])
def test_main_records_when_only_one_kind_of_test_row_is_present(tmp_path, temporary_ledger, role):
    _, manifest, _ = make_dataset(tmp_path)
    rows = [r for r in manifest if not r.role.startswith("test_") or r.role == role]
    assert cache.main(main_argv(tmp_path, rows)) == 0
    assert len(ledger_entries(temporary_ledger)) == 1


# 확인: 바꿀 파일이 사용 중이라 만들기를 시작할 수 없으면(PermissionError) 장부에 새 줄을 쓰지 않는다. 실제로
# 테스트 이미지를 읽지 않았는데 읽었다고 기록하지 않도록, 사용 중 검사를 장부 기록보다 먼저 한다.
def test_main_records_nothing_when_the_build_cannot_start(tmp_path, monkeypatch, temporary_ledger):
    _, manifest, _ = make_dataset(tmp_path)
    argv = main_argv(tmp_path, manifest)
    assert cache.main(argv) == 0
    monkeypatch.setattr(cache.os, "rename", in_use(f"visa_{SIZE}_images.npy"))
    with pytest.raises(PermissionError, match="in use"):
        cache.main([*argv, "--force"])
    assert len(ledger_entries(temporary_ledger)) == 1  # 두 번째에는 테스트 이미지를 읽지 않았다


# 확인: 다른 매니페스트(예: 한 범주만)로 다른 크기를 만들면 공유 마스크를 바꿔야 하는데, 그러면 이미 완성된
# 다른 크기 캐시를 못 쓰게 된다. CLI는 'ERROR'와 --force 안내를 출력하고 1을 돌려주며 아무 파일도 바꾸지 않고
# 장부도 쓰지 않는다. --force를 주면 만들고, 옛 크기는 열 때 거부되며, 남은 색인은 모두 자기 배열을 정확히
# 설명한다.
def test_main_refuses_to_rebuild_masks_that_other_sizes_use(tmp_path, capsys, temporary_ledger):
    _, manifest, files = make_dataset(tmp_path)
    out_dir = tmp_path / "cache"
    assert cache.main(main_argv(tmp_path, manifest)) == 0
    before = snapshot(out_dir)
    capsys.readouterr()

    subset = manifest[:5]  # 예: 한 범주의 매니페스트. 그 마스크가 전체 마스크를 대신하게 된다
    argv = main_argv(tmp_path, subset, size=48, name="subset.csv")
    assert cache.main(argv) == 1
    out = capsys.readouterr().out
    assert "ERROR" in out and "--force" in out and str(SIZE) in out
    assert snapshot(out_dir) == before
    assert len(ledger_entries(temporary_ledger)) == 1

    assert cache.main([*argv, "--force"]) == 0
    assert cache.cache_is_complete(out_dir, 48, subset)
    with pytest.raises(ValueError, match="different manifest"):
        ImageCache(out_dir, SIZE)
    check_indexes_describe_their_arrays(out_dir, manifest, files)


# 확인: 마스크를 쓰는 다른 크기 캐시가 없으면(같은 크기를 새 순서로 다시 만듦) 잃을 것이 없으므로 --force 없이
# 다시 만든다.
def test_main_rebuilds_masks_without_force_when_no_other_size_uses_them(tmp_path):
    _, manifest, files = make_dataset(tmp_path)
    out_dir = tmp_path / "cache"
    assert cache.main(main_argv(tmp_path, manifest)) == 0
    reordered = manifest[::-1]
    assert cache.main(main_argv(tmp_path, reordered, name="reordered.csv")) == 0  # 같은 크기: 잃을 것 없음
    assert cache.cache_is_complete(out_dir, SIZE, reordered)
    check_indexes_describe_their_arrays(out_dir, manifest, files)


# 확인: 마스크 색인, 마스크 배열, 이미지 색인 가운데 하나라도 없거나 마스크 색인의 행 순서가 다르면 완성된
# 캐시로 보지 않는다(CLI가 건너뛰지 않고 다시 만든다).
def test_cache_is_complete_needs_the_masks_and_their_index(tmp_path):
    tar_path, manifest, _ = make_dataset(tmp_path)
    out_dir = tmp_path / "cache"
    build_cache(tar_path, manifest, SIZE, out_dir, progress=False)
    assert cache.cache_is_complete(out_dir, SIZE, manifest)
    for name in ("visa_masks_256_index.csv", "visa_masks_256.npy", f"visa_{SIZE}_index.csv"):
        saved = (out_dir / name).read_bytes()
        (out_dir / name).unlink()
        assert not cache.cache_is_complete(out_dir, SIZE, manifest), name
        (out_dir / name).write_bytes(saved)
        assert cache.cache_is_complete(out_dir, SIZE, manifest)
    # 행 순서가 다른 마스크도 완성으로 치지 않는다.
    (out_dir / "visa_masks_256_index.csv").write_text(
        "".join(f"{r.image}\n" for r in manifest[::-1]), encoding="utf-8", newline="\n"
    )
    assert not cache.cache_is_complete(out_dir, SIZE, manifest)


# 확인: 이미지 배열의 크기·행 수·채널이 다르거나 마스크 배열 모양이 다르면 ImageCache가 'array shape'
# ValueError를 내고 완성으로도 보지 않는다. 다른 크기 캐시 파일과 섞이는 일을 막는다.
def test_image_cache_rejects_arrays_of_the_wrong_shape(tmp_path):
    tar_path, manifest, _ = make_dataset(tmp_path)
    out_dir = tmp_path / "cache"
    build_cache(tar_path, manifest, SIZE, out_dir, progress=False)
    n = len(manifest)
    images, masks = out_dir / f"visa_{SIZE}_images.npy", out_dir / "visa_masks_256.npy"
    good_images, good_masks = images.read_bytes(), masks.read_bytes()

    for shape in [(n, SIZE + 1, SIZE + 1, 3), (n - 1, SIZE, SIZE, 3), (n, SIZE, SIZE)]:
        np.save(images, np.zeros(shape, dtype=np.uint8))
        with pytest.raises(ValueError, match="image array shape"):
            ImageCache(out_dir, SIZE)
        assert not cache.cache_is_complete(out_dir, SIZE, manifest)
    images.write_bytes(good_images)

    for shape in [(n, SIZE, SIZE), (n + 1, MASK_SIZE, MASK_SIZE)]:
        np.save(masks, np.zeros(shape, dtype=np.uint8))
        with pytest.raises(ValueError, match="mask array shape"):
            ImageCache(out_dir, SIZE)
        assert not cache.cache_is_complete(out_dir, SIZE, manifest)
    masks.write_bytes(good_masks)
    ImageCache(out_dir, SIZE).close()


# 확인: tar 구성원 이름이 './'로 시작해도 매니페스트 경로와 맞춰 읽는다(tar를 만드는 방식에 따라 붙는 접두어).
def test_build_reads_member_names_with_a_dot_prefix(tmp_path):
    tar_path, manifest, files = make_dataset(tmp_path, prefix="./")
    with tarfile.open(tar_path) as tar:
        assert all(name.startswith("./") or name == "pcb1" for name in tar.getnames())
    out_dir = tmp_path / "cache"
    build_cache(tar_path, manifest, SIZE, out_dir, progress=False)
    assert cache.cache_is_complete(out_dir, SIZE, manifest)
    check_indexes_describe_their_arrays(out_dir, manifest, files)


# 지정한 파일에서만 PermissionError를 내는 os.rename. Windows가 메모리 맵으로 열린 파일을 바꾸지 못하는 상황을
# 흉내 낸다(캐시는 같은 이름으로 rename해 보아 사용 중인지 미리 확인한다).
def in_use(*names: str):
    """지정한 파일에서만 실패하는 `os.rename`. Windows가 메모리 맵으로 열린 파일에서 실패하는 것과 같다."""
    real_rename = os.rename

    def rename(src, dst):
        if Path(src).name in names:
            raise PermissionError(13, "The process cannot access the file", str(src))
        return real_rename(src, dst)

    return rename


# 호출되면 실패하는 디코더. '디코딩을 시작하기 전에 멈췄다'를 확인하는 데 쓴다.
def no_decoding(*args):
    raise AssertionError("the build must stop before it decodes anything")


# 확인: 바꿀 이미지 배열이나 색인이 사용 중이면 tar를 읽거나 디코딩하기 전에 'in use' PermissionError로
# 멈추고, 파일을 하나도 지우거나 .part를 남기지 않아 기존 캐시가 그대로 완성 상태다.
# 왜: tar 한 번 읽기는 수 분 걸리는데, 다 만든 뒤 마지막 교체에서 실패하면 그 시간이
# 낭비된다(_check_replaceable docstring).
@pytest.mark.parametrize("busy", [f"visa_{SIZE}_images.npy", f"visa_{SIZE}_index.csv"])
def test_build_stops_before_decoding_when_a_file_it_replaces_is_in_use(tmp_path, monkeypatch, busy):
    tar_path, manifest, _ = make_dataset(tmp_path)
    out_dir = tmp_path / "cache"
    build_cache(tar_path, manifest, SIZE, out_dir, progress=False)
    before = snapshot(out_dir)

    monkeypatch.setattr(cache.os, "rename", in_use(busy))
    monkeypatch.setattr(cache, "_decode_image", no_decoding)
    monkeypatch.setattr(cache, "_decode_mask", no_decoding)
    with pytest.raises(PermissionError, match="in use"):
        build_cache(tar_path, manifest, SIZE, out_dir, progress=False)
    assert snapshot(out_dir) == before  # 지운 것 없음, ".part" 파일 없음
    assert cache.cache_is_complete(out_dir, SIZE, manifest)


# 확인: 마스크 파일이 사용 중이어도 마스크를 재사용하는 빌드(같은 매니페스트, 다른 크기)는 진행되고, 마스크를
# 새로 만들어야 하는 빌드(다른 순서)만 디코딩 전에 멈춘다.
def test_masks_in_use_block_a_build_only_when_they_are_rebuilt(tmp_path, monkeypatch):
    tar_path, manifest, _ = make_dataset(tmp_path)
    out_dir = tmp_path / "cache"
    build_cache(tar_path, manifest, SIZE, out_dir, progress=False)
    before = snapshot(out_dir)
    monkeypatch.setattr(cache.os, "rename", in_use("visa_masks_256.npy"))

    # 같은 매니페스트, 다른 크기: 마스크를 재사용하므로 마스크가 열려 있어도 상관없다.
    build_cache(tar_path, manifest, 48, out_dir, progress=False)
    assert cache.cache_is_complete(out_dir, 48, manifest)
    before = snapshot(out_dir)

    monkeypatch.setattr(cache, "_decode_image", no_decoding)
    monkeypatch.setattr(cache, "_decode_mask", no_decoding)
    with pytest.raises(PermissionError, match="in use"):
        build_cache(tar_path, manifest[::-1], 64, out_dir, progress=False)  # 다른 순서: 새 마스크
    assert snapshot(out_dir) == before


# 확인(Windows에서만): 실제로 ImageCache를 열어 둔 채 다시 만들면 이미지·마스크 둘 다 사용 중이라 일찍
# 실패하고, 캐시는 그대로이며 열린 ImageCache도 계속 읽힌다. close() 뒤에는 다시 만들 수 있고 결과가 바이트
# 단위로 같다(결정적).
@pytest.mark.skipif(sys.platform != "win32", reason="only Windows refuses to replace a mapped file")
def test_rebuild_with_an_open_image_cache_fails_early_and_keeps_the_cache(tmp_path, monkeypatch):
    tar_path, manifest, files = make_dataset(tmp_path)
    out_dir = tmp_path / "cache"
    build_cache(tar_path, manifest, SIZE, out_dir, progress=False)
    before = snapshot(out_dir)
    real_decode = cache._decode_image

    store = ImageCache(out_dir, SIZE)
    monkeypatch.setattr(cache, "_decode_image", no_decoding)
    with pytest.raises(PermissionError, match="in use"):
        build_cache(tar_path, manifest, SIZE, out_dir, progress=False)
    with pytest.raises(PermissionError, match="in use"):  # 마스크도 메모리 맵으로 열려 있다
        build_cache(tar_path, manifest[::-1], 48, out_dir, progress=False)
    assert snapshot(out_dir) == before
    assert cache.cache_is_complete(out_dir, SIZE, manifest)
    assert np.array_equal(store.images(manifest[:1])[0], expected_image(files[manifest[0].image], SIZE))

    store.close()
    monkeypatch.setattr(cache, "_decode_image", real_decode)
    build_cache(tar_path, manifest, SIZE, out_dir, progress=False)
    assert snapshot(out_dir) == before  # 다시 만들기는 결정적이다


# 다 만든 .part 파일을 제자리로 옮기는 순서: 마스크 배열 → 마스크 색인 → 이미지 배열 → 이미지 색인. 배열을
# 먼저, 그 색인을 바로 뒤에 둔다.
PUBLISH_ORDER = [
    "visa_masks_256.npy",
    "visa_masks_256_index.csv",
    f"visa_{SIZE}_images.npy",
    f"visa_{SIZE}_index.csv",
]


# 확인: 옮기기(os.replace) 1~4번째 어디서 PermissionError나 Ctrl-C가 나도, 그 뒤로는 아무것도 옮기지 않고
# .part를 남기지 않으며, 남은 색인은 모두 자기 배열을 정확히 설명한다. 첫 교체가 거부되면 기존 캐시가 완성
# 상태로 남고, 이미지 배열 교체가 거부되면 옛 색인을 되돌려 놓는다(마스크는 이미 새 순서라 열 때 거부된다).
# 다음 빌드는 모든 것을 고친다.
# 왜: _publish docstring의 약속('무엇이 끊어도 자기를 설명하지 않는 배열 옆에 색인이 남지 않는다')을 모든 실패
# 지점에서 확인한다.
@pytest.mark.parametrize("error", [PermissionError, KeyboardInterrupt])
@pytest.mark.parametrize("fail_at", [1, 2, 3, 4])
def test_failed_publish_never_leaves_an_index_without_its_array(tmp_path, monkeypatch, fail_at, error):
    tar_path, manifest, files = make_dataset(tmp_path)
    out_dir = tmp_path / "cache"
    build_cache(tar_path, manifest, SIZE, out_dir, progress=False)
    before = snapshot(out_dir)
    reordered = manifest[::-1]  # 다른 행 순서: 마스크와 이미지 둘 다 다시 만들어지고 둘 다 달라진다
    real_replace = os.replace
    published: list[str] = []

    def failing_replace(src, dst):
        published.append(Path(dst).name)
        if len(published) == fail_at:
            raise error("simulated failure while publishing")
        return real_replace(src, dst)

    monkeypatch.setattr(cache.os, "replace", failing_replace)
    with pytest.raises(error):
        build_cache(tar_path, reordered, SIZE, out_dir, progress=False)
    monkeypatch.setattr(cache.os, "replace", real_replace)

    # 배열이 먼저, 색인은 자기 배열 바로 뒤에 옮기며, 실패한 뒤로는 아무것도 옮기지 않는다.
    assert published == PUBLISH_ORDER[:fail_at]
    assert not list(out_dir.glob("*.part"))
    check_indexes_describe_their_arrays(out_dir, manifest, files)
    assert not cache.cache_is_complete(out_dir, SIZE, reordered)
    if fail_at == 1 and error is PermissionError:
        # 맨 첫 교체가 거부됐다: 아무것도 바뀌지 않았고 이전 캐시가 여전히 완성 상태다.
        assert snapshot(out_dir) == before
        assert cache.cache_is_complete(out_dir, SIZE, manifest)
    if fail_at == 3 and error is PermissionError:
        # 이미지 배열을 바꾸지 못했다: 옛 색인이 옛 배열 옆으로 돌아왔다.
        assert (out_dir / f"visa_{SIZE}_index.csv").read_bytes() == before[f"visa_{SIZE}_index.csv"]
        with pytest.raises(ValueError, match="different manifest"):
            ImageCache(out_dir, SIZE)  # 마스크가 새 순서로 다시 만들어졌다: 틀리게 읽지 않고 거부한다

    # 다음 빌드가 모든 것을 고친다.
    build_cache(tar_path, reordered, SIZE, out_dir, progress=False)
    assert cache.cache_is_complete(out_dir, SIZE, reordered)
    check_indexes_describe_their_arrays(out_dir, manifest, files)


# 확인: 새 배열이 이미 옮겨진 직후에 Ctrl-C가 나면 옛 색인을 되돌려 놓지 않는다(되돌리면 새 배열 옆에 옛 행을
# 설명하는 색인이 생긴다). 그 배열은 색인 없이 남아 캐시는 미완성으로 읽힌다. 옛 색인을 되돌리는 것은 교체
# 자체가 실패한(OSError) 경우뿐이다.
@pytest.mark.parametrize("array_name", ["visa_masks_256.npy", f"visa_{SIZE}_images.npy"])
def test_interrupt_right_after_an_array_was_replaced_does_not_bring_the_old_index_back(
    tmp_path, monkeypatch, array_name
):
    # Ctrl-C는 os.replace가 새 배열을 이미 옮긴 뒤에 올 수 있다. 옛 색인은 교체 자체가 실패했을 때(OSError)만
    # 되돌려 놓는다. 여기서 되돌리면 그 색인은 엉뚱한 행을 설명하게 된다.
    tar_path, manifest, files = make_dataset(tmp_path)
    out_dir = tmp_path / "cache"
    build_cache(tar_path, manifest, SIZE, out_dir, progress=False)
    reordered = manifest[::-1]
    real_replace = os.replace

    def replaced_then_interrupted(src, dst):
        real_replace(src, dst)
        if Path(dst).name == array_name:
            raise KeyboardInterrupt

    monkeypatch.setattr(cache.os, "replace", replaced_then_interrupted)
    with pytest.raises(KeyboardInterrupt):
        build_cache(tar_path, reordered, SIZE, out_dir, progress=False)
    monkeypatch.setattr(cache.os, "replace", real_replace)

    index_name = array_name.replace("_images.npy", "_index.csv").replace("256.npy", "256_index.csv")
    assert (out_dir / array_name).exists() and not (out_dir / index_name).exists()
    assert not list(out_dir.glob("*.part"))
    check_indexes_describe_their_arrays(out_dir, manifest, files)
    assert not cache.cache_is_complete(out_dir, SIZE, manifest)
    assert not cache.cache_is_complete(out_dir, SIZE, reordered)


# 확인: 같은 매니페스트로 다시 만들다 이미지 배열과 색인 사이에서 끊기면 이미지 색인이 없어 캐시가
# '미완성'으로 읽히고 ImageCache는 열리지 않는다. 반쯤 믿을 수 있는 상태 대신 다시 만들어야 하는 상태로
# 남긴다.
def test_publish_of_the_same_manifest_survives_an_interrupt_between_array_and_index(tmp_path, monkeypatch):
    tar_path, manifest, files = make_dataset(tmp_path)
    out_dir = tmp_path / "cache"
    build_cache(tar_path, manifest, SIZE, out_dir, progress=False)
    real_replace = os.replace

    def interrupted(src, dst):
        if Path(dst).name == f"visa_{SIZE}_index.csv":
            raise KeyboardInterrupt
        return real_replace(src, dst)

    monkeypatch.setattr(cache.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        build_cache(tar_path, manifest, SIZE, out_dir, progress=False)
    monkeypatch.setattr(cache.os, "replace", real_replace)
    # 배열은 새것이고 그 색인은 없다: 캐시는 반쯤 믿을 상태가 아니라 미완성으로 읽힌다.
    assert sorted(p.name for p in out_dir.iterdir()) == [
        f"visa_{SIZE}_images.npy",
        "visa_masks_256.npy",
        "visa_masks_256_index.csv",
    ]
    assert not cache.cache_is_complete(out_dir, SIZE, manifest)
    with pytest.raises(FileNotFoundError):
        ImageCache(out_dir, SIZE)
    check_indexes_describe_their_arrays(out_dir, manifest, files)
