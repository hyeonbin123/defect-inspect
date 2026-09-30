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

from defect_inspect import cache, ledger, paths
from defect_inspect.cache import MASK_SIZE, ImageCache, build_cache, resize_image, resize_mask
from defect_inspect.splits import ManifestRow, write_manifest

SIZE = 32


@pytest.fixture(autouse=True)
def temporary_ledger(tmp_path, monkeypatch):
    """`cache.main` records sealed-test reads: no test may ever write to the real ledger."""
    path = tmp_path / "reports" / "test_ledger.jsonl"
    monkeypatch.setattr(paths, "TEST_LEDGER", path)
    monkeypatch.setattr(ledger, "git_commit", lambda root=None: "abc1234")
    return path


def jpeg_bytes(seed: int, width: int = 90, height: int = 60) -> bytes:
    rng = np.random.default_rng(seed)
    # Smooth content (upsampled noise) so that JPEG artefacts stay small.
    coarse = rng.integers(0, 256, size=(6, 9, 3), dtype=np.uint8)
    img = Image.fromarray(coarse).resize((width, height), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def mask_bytes(seed: int, width: int = 90, height: int = 60) -> bytes:
    rng = np.random.default_rng(seed)
    array = np.zeros((height, width), dtype=np.uint8)
    y, x = int(rng.integers(0, height - 8)), int(rng.integers(0, width - 8))
    array[y : y + 8, x : x + 8] = 1 + seed % 3  # class ids, not 255
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def make_dataset(tmp_path, n_normal: int = 7, n_defect: int = 4, prefix: str = ""):
    """A tiny tar whose member order differs from the manifest order, plus the files' bytes.

    `prefix` is put in front of every member name in the tar (not in the manifest).
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


def snapshot(out_dir) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in out_dir.iterdir()}


def check_indexes_describe_their_arrays(out_dir, manifest, files) -> None:
    """Every index file that exists sits next to an array with exactly those rows (no stale pairs)."""
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


def expected_image(data: bytes, size: int) -> np.ndarray:
    return resize_image(Image.open(io.BytesIO(data)), size)


def expected_mask(data: bytes) -> np.ndarray:
    return resize_mask(Image.open(io.BytesIO(data)))


def test_resize_image_is_square_rgb_bicubic():
    rng = np.random.default_rng(0)
    source = Image.fromarray(rng.integers(0, 256, size=(40, 70, 3), dtype=np.uint8))
    out = resize_image(source, 16)
    assert out.shape == (16, 16, 3) and out.dtype == np.uint8
    assert np.array_equal(out, np.asarray(source.resize((16, 16), Image.Resampling.BICUBIC)))
    # Shrinking averages neighbouring pixels (antialiasing) rather than picking single source pixels.
    stripes = np.zeros((64, 64, 3), dtype=np.uint8)
    stripes[:, ::2] = 255
    small = resize_image(Image.fromarray(stripes), 16)
    assert 100 < small.mean() < 155 and small.std() < 10


def test_resize_image_converts_other_modes_to_rgb():
    grey = Image.fromarray(np.full((20, 30), 77, dtype=np.uint8))
    out = resize_image(grey, 8)
    assert out.shape == (8, 8, 3) and (out == 77).all()
    rgba = Image.new("RGBA", (30, 20), (10, 20, 30, 0))
    assert (resize_image(rgba, 8) == np.array([10, 20, 30], dtype=np.uint8)).all()


def test_resize_mask_keeps_single_pixel_defects():
    height, width = 1100, 1500
    for y, x in [(0, 0), (1099, 1499), (0, 1499), (547, 733), (5, 6), (1098, 3)]:
        source = np.zeros((height, width), dtype=np.uint8)
        source[y, x] = 3  # a class id, not 255
        out = resize_mask(Image.fromarray(source))
        assert out.shape == (MASK_SIZE, MASK_SIZE) and out.dtype == np.uint8
        assert set(np.unique(out)) == {0, 1}
        assert out.sum() == 1
        # The box of a source pixel is the one that contains its centre.
        assert out[int((y + 0.5) * MASK_SIZE / height), int((x + 0.5) * MASK_SIZE / width)] == 1


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


def test_resize_mask_rejects_multichannel_masks():
    with pytest.raises(ValueError):
        resize_mask(Image.new("RGB", (8, 8)))


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
    # Rows really differ from each other (guards against every row getting the same image).
    assert len({images[i].tobytes() for i in range(len(manifest))}) == len(manifest)


def test_image_cache_returns_copies_in_row_order(tmp_path):
    tar_path, manifest, files = make_dataset(tmp_path)
    out_dir = tmp_path / "cache"
    build_cache(tar_path, manifest, SIZE, out_dir, workers=2, progress=False)
    store = ImageCache(out_dir, SIZE)
    assert store.size == SIZE and len(store) == len(manifest)

    rows = [manifest[9], manifest[0], manifest[10], manifest[3], manifest[9]]  # any order, repeats allowed
    images, masks = store.images(rows), store.masks(rows)
    assert images.shape == (5, SIZE, SIZE, 3) and images.dtype == np.uint8
    assert masks.shape == (5, MASK_SIZE, MASK_SIZE) and masks.dtype == np.uint8
    assert type(images) is np.ndarray and type(masks) is np.ndarray
    for i, row in enumerate(rows):
        assert np.array_equal(images[i], expected_image(files[row.image], SIZE))
        want = expected_mask(files[row.mask]) if row.mask else np.zeros((MASK_SIZE, MASK_SIZE), np.uint8)
        assert np.array_equal(masks[i], want)
    images[:] = 0  # in-memory copies: writable, and the cache itself is untouched
    assert np.array_equal(store.images(rows[:1])[0], expected_image(files[rows[0].image], SIZE))

    assert store.images([]).shape == (0, SIZE, SIZE, 3)
    assert store.masks([]).shape == (0, MASK_SIZE, MASK_SIZE)
    with pytest.raises(KeyError, match="not in the cache"):
        store.images(
            [ManifestRow("pcb1/Data/Images/Normal/9999.JPG", "", "pcb1", "normal", "pool_normal", 0, "")]
        )
    store.close()


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
    # The older size now disagrees with the mask order and says so instead of returning wrong masks.
    with pytest.raises(ValueError, match="different manifest"):
        ImageCache(out_dir, SIZE)


def test_failed_build_leaves_no_finished_files(tmp_path):
    tar_path, manifest, _ = make_dataset(tmp_path)
    out_dir = tmp_path / "cache"
    ghost = ManifestRow("pcb1/Data/Images/Normal/7777.JPG", "", "pcb1", "normal", "pool_normal", 0, "")
    with pytest.raises(FileNotFoundError, match="7777"):
        build_cache(tar_path, [*manifest, ghost], SIZE, out_dir, progress=False)
    assert list(out_dir.iterdir()) == []
    with pytest.raises(FileNotFoundError):
        ImageCache(out_dir, SIZE)


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


def test_build_rejects_bad_manifests(tmp_path):
    tar_path, manifest, _ = make_dataset(tmp_path)
    with pytest.raises(ValueError, match="more than once"):
        build_cache(tar_path, [*manifest, manifest[0]], SIZE, tmp_path / "cache", progress=False)
    with pytest.raises(ValueError, match="empty"):
        build_cache(tar_path, [], SIZE, tmp_path / "cache", progress=False)


def test_progress_lines(tmp_path, capsys):
    tar_path, manifest, _ = make_dataset(tmp_path)
    build_cache(tar_path, manifest, SIZE, tmp_path / "cache", workers=1)
    out = capsys.readouterr().out
    assert f"[cache {SIZE}] 15/15 members" in out  # 11 images + 4 masks


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


def main_argv(tmp_path, manifest, *extra: str, size: int = SIZE, name: str = "visa.csv") -> list[str]:
    manifest_path = tmp_path / name
    write_manifest(manifest, manifest_path)
    tar, out = str(tmp_path / "visa.tar"), str(tmp_path / "cache")
    return ["--size", str(size), "--tar", tar, "--manifest", str(manifest_path), "--out", out, *extra]


def ledger_entries(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


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
    assert recorded_before_build == [1]  # written before the first test image is read
    assert str(temporary_ledger) in capsys.readouterr().out

    # A complete cache is skipped: nothing is read, nothing is recorded.
    assert cache.main(argv) == 0
    assert len(ledger_entries(temporary_ledger)) == 1 and recorded_before_build == [1]
    # --force reads the test images again, so there is a second line.
    assert cache.main([*argv, "--force"]) == 0
    assert len(ledger_entries(temporary_ledger)) == 2 and recorded_before_build == [1, 2]
    # Another size is another read.
    assert cache.main(main_argv(tmp_path, manifest, size=48)) == 0
    assert [e["config"] for e in ledger_entries(temporary_ledger)] == [f"cache-{SIZE}"] * 2 + ["cache-48"]


def test_main_records_nothing_without_test_rows(tmp_path, temporary_ledger):
    _, manifest, _ = make_dataset(tmp_path)
    open_rows = [r for r in manifest if not r.role.startswith("test_")]
    assert 0 < len(open_rows) < len(manifest)
    assert cache.main(main_argv(tmp_path, open_rows)) == 0
    assert cache.cache_is_complete(tmp_path / "cache", SIZE, open_rows)
    assert not temporary_ledger.exists()


@pytest.mark.parametrize("role", ["test_normal", "test_defect"])
def test_main_records_when_only_one_kind_of_test_row_is_present(tmp_path, temporary_ledger, role):
    _, manifest, _ = make_dataset(tmp_path)
    rows = [r for r in manifest if not r.role.startswith("test_") or r.role == role]
    assert cache.main(main_argv(tmp_path, rows)) == 0
    assert len(ledger_entries(temporary_ledger)) == 1


def test_main_records_nothing_when_the_build_cannot_start(tmp_path, monkeypatch, temporary_ledger):
    _, manifest, _ = make_dataset(tmp_path)
    argv = main_argv(tmp_path, manifest)
    assert cache.main(argv) == 0
    monkeypatch.setattr(cache.os, "rename", in_use(f"visa_{SIZE}_images.npy"))
    with pytest.raises(PermissionError, match="in use"):
        cache.main([*argv, "--force"])
    assert len(ledger_entries(temporary_ledger)) == 1  # no test image was read the second time


def test_main_refuses_to_rebuild_masks_that_other_sizes_use(tmp_path, capsys, temporary_ledger):
    _, manifest, files = make_dataset(tmp_path)
    out_dir = tmp_path / "cache"
    assert cache.main(main_argv(tmp_path, manifest)) == 0
    before = snapshot(out_dir)
    capsys.readouterr()

    subset = manifest[:5]  # e.g. a manifest of one category: its masks would replace the full ones
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


def test_main_rebuilds_masks_without_force_when_no_other_size_uses_them(tmp_path):
    _, manifest, files = make_dataset(tmp_path)
    out_dir = tmp_path / "cache"
    assert cache.main(main_argv(tmp_path, manifest)) == 0
    reordered = manifest[::-1]
    assert cache.main(main_argv(tmp_path, reordered, name="reordered.csv")) == 0  # same size: nothing lost
    assert cache.cache_is_complete(out_dir, SIZE, reordered)
    check_indexes_describe_their_arrays(out_dir, manifest, files)


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
    # Masks in another row order do not count either.
    (out_dir / "visa_masks_256_index.csv").write_text(
        "".join(f"{r.image}\n" for r in manifest[::-1]), encoding="utf-8", newline="\n"
    )
    assert not cache.cache_is_complete(out_dir, SIZE, manifest)


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


def test_build_reads_member_names_with_a_dot_prefix(tmp_path):
    tar_path, manifest, files = make_dataset(tmp_path, prefix="./")
    with tarfile.open(tar_path) as tar:
        assert all(name.startswith("./") or name == "pcb1" for name in tar.getnames())
    out_dir = tmp_path / "cache"
    build_cache(tar_path, manifest, SIZE, out_dir, progress=False)
    assert cache.cache_is_complete(out_dir, SIZE, manifest)
    check_indexes_describe_their_arrays(out_dir, manifest, files)


def in_use(*names: str):
    """An `os.rename` that fails for the named files, the way Windows does for a mapped file."""
    real_rename = os.rename

    def rename(src, dst):
        if Path(src).name in names:
            raise PermissionError(13, "The process cannot access the file", str(src))
        return real_rename(src, dst)

    return rename


def no_decoding(*args):
    raise AssertionError("the build must stop before it decodes anything")


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
    assert snapshot(out_dir) == before  # nothing removed, no ".part" files
    assert cache.cache_is_complete(out_dir, SIZE, manifest)


def test_masks_in_use_block_a_build_only_when_they_are_rebuilt(tmp_path, monkeypatch):
    tar_path, manifest, _ = make_dataset(tmp_path)
    out_dir = tmp_path / "cache"
    build_cache(tar_path, manifest, SIZE, out_dir, progress=False)
    before = snapshot(out_dir)
    monkeypatch.setattr(cache.os, "rename", in_use("visa_masks_256.npy"))

    # Same manifest, other size: the masks are reused, so it does not matter that they are open.
    build_cache(tar_path, manifest, 48, out_dir, progress=False)
    assert cache.cache_is_complete(out_dir, 48, manifest)
    before = snapshot(out_dir)

    monkeypatch.setattr(cache, "_decode_image", no_decoding)
    monkeypatch.setattr(cache, "_decode_mask", no_decoding)
    with pytest.raises(PermissionError, match="in use"):
        build_cache(tar_path, manifest[::-1], 64, out_dir, progress=False)  # other order: new masks
    assert snapshot(out_dir) == before


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
    with pytest.raises(PermissionError, match="in use"):  # the masks are mapped too
        build_cache(tar_path, manifest[::-1], 48, out_dir, progress=False)
    assert snapshot(out_dir) == before
    assert cache.cache_is_complete(out_dir, SIZE, manifest)
    assert np.array_equal(store.images(manifest[:1])[0], expected_image(files[manifest[0].image], SIZE))

    store.close()
    monkeypatch.setattr(cache, "_decode_image", real_decode)
    build_cache(tar_path, manifest, SIZE, out_dir, progress=False)
    assert snapshot(out_dir) == before  # rebuilding is deterministic


PUBLISH_ORDER = [
    "visa_masks_256.npy",
    "visa_masks_256_index.csv",
    f"visa_{SIZE}_images.npy",
    f"visa_{SIZE}_index.csv",
]


@pytest.mark.parametrize("error", [PermissionError, KeyboardInterrupt])
@pytest.mark.parametrize("fail_at", [1, 2, 3, 4])
def test_failed_publish_never_leaves_an_index_without_its_array(tmp_path, monkeypatch, fail_at, error):
    tar_path, manifest, files = make_dataset(tmp_path)
    out_dir = tmp_path / "cache"
    build_cache(tar_path, manifest, SIZE, out_dir, progress=False)
    before = snapshot(out_dir)
    reordered = manifest[::-1]  # other row order: masks and images are both rebuilt and both differ
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

    # Arrays go first, each index right after its array, and nothing is published after a failure.
    assert published == PUBLISH_ORDER[:fail_at]
    assert not list(out_dir.glob("*.part"))
    check_indexes_describe_their_arrays(out_dir, manifest, files)
    assert not cache.cache_is_complete(out_dir, SIZE, reordered)
    if fail_at == 1 and error is PermissionError:
        # The very first replace was refused: nothing changed, the previous cache is still complete.
        assert snapshot(out_dir) == before
        assert cache.cache_is_complete(out_dir, SIZE, manifest)
    if fail_at == 3 and error is PermissionError:
        # The image array could not be replaced: its old index is back, next to the old array.
        assert (out_dir / f"visa_{SIZE}_index.csv").read_bytes() == before[f"visa_{SIZE}_index.csv"]
        with pytest.raises(ValueError, match="different manifest"):
            ImageCache(out_dir, SIZE)  # the masks were rebuilt in the new order: refused, not wrong

    # The next build repairs everything.
    build_cache(tar_path, reordered, SIZE, out_dir, progress=False)
    assert cache.cache_is_complete(out_dir, SIZE, reordered)
    check_indexes_describe_their_arrays(out_dir, manifest, files)


@pytest.mark.parametrize("array_name", ["visa_masks_256.npy", f"visa_{SIZE}_images.npy"])
def test_interrupt_right_after_an_array_was_replaced_does_not_bring_the_old_index_back(
    tmp_path, monkeypatch, array_name
):
    # Ctrl-C can land after os.replace has already moved the new array in. The old index is only
    # put back when the replace itself failed (OSError); here it would describe the wrong rows.
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
    # The array is new, its index is gone: the cache reads as unfinished instead of half-trusted.
    assert sorted(p.name for p in out_dir.iterdir()) == [
        f"visa_{SIZE}_images.npy",
        "visa_masks_256.npy",
        "visa_masks_256_index.csv",
    ]
    assert not cache.cache_is_complete(out_dir, SIZE, manifest)
    with pytest.raises(FileNotFoundError):
        ImageCache(out_dir, SIZE)
    check_indexes_describe_their_arrays(out_dir, manifest, files)
