import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

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

SIZE = 16


@pytest.fixture(autouse=True)
def temporary_ledger(tmp_path, monkeypatch):
    """`m2ad.main` records the read of test images: no test may ever write to the real ledger."""
    path = tmp_path / "reports" / "test_ledger.jsonl"
    monkeypatch.setattr(paths, "TEST_LEDGER", path)
    monkeypatch.setattr(ledger, "git_commit", lambda root=None: "abc1234")
    return path


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


def make_meta(train=("000", "003"), normal=("001",), lights=ILLUMINATIONS, views=(*VIEWS, "030")) -> dict:
    """Two kept categories and one that must be ignored; view 030 must be dropped as well.

    Per kept category the test split has the normal specimens plus three defective ones: `hole_1_000`
    (visible and detectable everywhere), `scratch_1_001` (not detectable in view 120, not visible in view
    240) and `dent_1_002` (not visible at all).
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
        meta["train"][category] = train_rows[::-1]  # the file order must not matter
        meta["test"][category] = test_rows[::-1]
    return meta


def write_jsons(tmp_path, meta, member=m2ad.META_MEMBER):
    path = tmp_path / "jsons.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(member, json.dumps(meta))
        z.writestr("meta_other.json", "{}")
    return path


def png_bytes(name: str, side: int = 40) -> bytes:
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, size=(5, 5, 3), dtype=np.uint8)
    img = Image.fromarray(coarse).resize((side, side), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def write_zips(zip_dir, rows, *, reverse=False, skip=()):
    """`<category>.zip` with every image of `rows` plus members that are not wanted."""
    zip_dir.mkdir(parents=True, exist_ok=True)
    for category in {r.category for r in rows}:
        names = [r.img_path for r in rows if r.category == category and r.img_path not in skip]
        names += [f"{category}/Good/000/A030_I01.png", f"{category}/GT/hole_1_000/A000_mask.png"]
        names.sort(key=lambda name: hashlib.sha256(name.encode()).hexdigest(), reverse=reverse)
        with zipfile.ZipFile(zip_dir / f"{category}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"{category}/", b"")
            for name in names:
                z.writestr(name, png_bytes(name))


def small_rows(tmp_path):
    meta = make_meta(lights=("01", "02"))
    # Only two illuminations here, so `read_meta` is not asked for completeness.
    return read_meta(write_jsons(tmp_path, meta))


# ----------------------------------------------------------------------------------------------- metadata


def test_read_meta_keeps_two_categories_and_three_views(tmp_path):
    rows = read_meta(write_jsons(tmp_path, make_meta()))
    assert {r.category for r in rows} == {"Motor", "Bird"}
    assert {r.view for r in rows} == set(VIEWS)
    assert {r.illumination for r in rows} == set(ILLUMINATIONS)
    # 2 categories x 3 views x 10 illuminations x (2 train + 1 normal + 3 defective specimens)
    assert len(rows) == 2 * 3 * 10 * 6
    assert all(isinstance(r, M2adRow) for r in rows)


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


def test_label_rule(tmp_path):
    rows = read_meta(write_jsons(tmp_path, make_meta()))
    by = {(r.category, r.specimen, r.view, r.illumination): r for r in rows}
    for category in ("Motor", "Bird"):
        for light in ILLUMINATIONS:
            # Normal specimens are label 0, in both splits.
            assert by[category, "000", "000", light].label == 0
            assert by[category, "001", "240", light].label == 0
            assert by[category, "001", "240", light].object_anomaly == 0
            # Visible and detectable defect.
            assert by[category, "hole_1_000", "120", light].label == 1
            assert by[category, "scratch_1_001", "000", light].label == 1
            # detectable False: excluded. Defect not visible (detectable ""): excluded.
            assert by[category, "scratch_1_001", "120", light].label == -1
            assert by[category, "scratch_1_001", "240", light].label == -1
            assert by[category, "dent_1_002", "000", light].label == -1
            assert by[category, "dent_1_002", "000", light].object_anomaly == 1
    assert {r.label for r in rows if r.split == "train"} == {0}


@pytest.mark.parametrize(
    ("object_anomaly", "image_anomaly", "detectable", "expected"),
    [(0, 0, "", 0), (1, 1, True, 1), (1, 1, False, -1), (1, 0, "", -1), (1, 1, "", -1), (1, 0, True, -1)],
)
def test_label_of(object_anomaly, image_anomaly, detectable, expected):
    assert m2ad.label_of(object_anomaly, image_anomaly, detectable) == expected


def test_label_of_needs_a_real_true():
    # A truthy value that is not the JSON `true` does not make a defect label.
    assert m2ad.label_of(1, 1, 1) == -1
    assert m2ad.label_of(1, 1, "True") == -1


def test_read_meta_finds_the_member_in_a_folder(tmp_path):
    path = write_jsons(tmp_path, make_meta(), member="jsons/" + m2ad.META_MEMBER)
    assert len(read_meta(path)) == 360


def test_read_meta_rejects_bad_metadata(tmp_path):
    meta = make_meta()
    kept = next(e for e in meta["test"]["Motor"] if e["view"] == "000")
    meta["test"]["Motor"].append(dict(kept))
    with pytest.raises(ValueError, match="more than once"):
        read_meta(write_jsons(tmp_path, meta))

    # The same entry again in a view that is not used is ignored.
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
    meta["test"]["Motor"].append(entry("Motor", "Again", "000", "000", "01"))  # a train specimen
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


# ------------------------------------------------------------------------------------------ specimen order


def sha_order(names):
    return sorted(set(names), key=lambda name: hashlib.sha256(name.encode("utf-8")).hexdigest())


def test_folds_follow_the_sha256_rank():
    names = [f"{i:03d}" for i in range(0, 60, 2)]
    folds = specimen_folds(names)
    ordered = sha_order(names)
    assert ordered != sorted(names)  # the hash order is not the name order
    assert folds == {name: rank % 5 for rank, name in enumerate(ordered)}
    assert sorted(np.bincount(list(folds.values())).tolist()) == [6, 6, 6, 6, 6]
    # The input order and repeated names (one per image) do not matter.
    assert specimen_folds(names[::-1] + names) == folds


def test_recal_specimens_are_the_first_in_the_same_order():
    names = [f"{i:03d}" for i in range(30)]
    ordered = sha_order(names)
    assert recal_specimens(names, 8) == ordered[:8]
    assert recal_specimens(names[::-1], 30) == ordered
    assert recal_specimens(names, 0) == []
    assert recal_specimens(names, 8) == recal_specimens(names, 30)[:8]  # nested
    # The first eight specimens cover every fold.
    folds = specimen_folds(names)
    assert [folds[s] for s in recal_specimens(names, 8)] == [0, 1, 2, 3, 4, 0, 1, 2]
    with pytest.raises(ValueError):
        recal_specimens(names, 31)
    with pytest.raises(ValueError):
        recal_specimens(names, -1)


def test_specimen_names_must_be_strings():
    with pytest.raises(TypeError):
        specimen_folds([1, 2, 3])


# --------------------------------------------------------------------------------------------------- cache


def expected_image(name: str) -> np.ndarray:
    with Image.open(io.BytesIO(png_bytes(name))) as img:
        return cache.resize_image(img, SIZE)


def test_constants_are_the_registered_ones():
    assert m2ad.CATEGORIES == ("Motor", "Bird") and m2ad.VIEWS == ("000", "120", "240")
    assert m2ad.ILLUMINATIONS == tuple(f"{i:02d}" for i in range(1, 11))
    assert m2ad.REFERENCE == "01" and m2ad.N_FOLDS == 5 and m2ad.RECAL_SIZES == (8, 30)


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
        # Each zip is read front to back (sequential on an HDD), which is not the order of the names,
        # and nothing but the wanted members is read, each of them once.
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
    assert len(decoded) == 2 * len(rows)  # only the wanted members are decoded
    for i in (0, 7, len(rows) - 1):
        assert np.array_equal(arrays[0][i], expected_image(rows[i].img_path))
    assert m2ad.cache_is_complete(tmp_path / "cache0", SIZE, rows)
    assert not m2ad.cache_is_complete(tmp_path / "cache0", SIZE, rows[::-1])
    assert not m2ad.cache_is_complete(tmp_path / "cache0", SIZE + 1, rows)


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


def test_build_cache_rejects_empty_and_repeated_rows(tmp_path):
    rows = small_rows(tmp_path)
    write_zips(tmp_path / "zips", rows)
    with pytest.raises(ValueError):
        build_cache(tmp_path / "zips", [], SIZE, tmp_path / "cache")
    with pytest.raises(ValueError):
        build_cache(tmp_path / "zips", [rows[0], rows[0]], SIZE, tmp_path / "cache")


def test_cache_names_a_missing_build(tmp_path):
    with pytest.raises(FileNotFoundError, match="defect_inspect.m2ad --size 24"):
        M2adCache(tmp_path, 24)


# ----------------------------------------------------------------------------------------------------- CLI


def ledger_entries(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_main_records_the_read_before_it_builds(tmp_path, capsys, monkeypatch, temporary_ledger):
    jsons = write_jsons(tmp_path, make_meta(lights=("01", "02")))
    rows = read_meta(jsons)
    write_zips(tmp_path / "zips", rows)
    argv = ["--size", str(SIZE), "--zips", str(tmp_path / "zips"), "--jsons", str(jsons)]
    argv += ["--out", str(tmp_path / "cache"), "--workers", "2"]
    real_build = m2ad.build_cache
    recorded_before_build = []

    def watching_build(*args, **kwargs):
        recorded_before_build.append(len(ledger_entries(temporary_ledger)))
        return real_build(*args, **kwargs)

    monkeypatch.setattr(m2ad, "build_cache", watching_build)
    assert m2ad.main(argv) == 0
    entries = ledger_entries(temporary_ledger)
    assert len(entries) == 1 and recorded_before_build == [1]
    assert (entries[0]["stage"], entries[0]["config"]) == ("cache", f"m2ad-cache-{SIZE}")
    assert entries[0]["commit"] == "abc1234"
    out = capsys.readouterr().out
    assert str(temporary_ledger) in out and f"({len(rows)}, {SIZE}, {SIZE}, 3)" in out
    assert m2ad.cache_is_complete(tmp_path / "cache", SIZE, rows)

    # A complete cache is skipped: nothing is read, nothing is recorded.
    assert m2ad.main(argv) == 0
    assert len(ledger_entries(temporary_ledger)) == 1 and recorded_before_build == [1]
    # --force reads the images again, so there is a second line.
    assert m2ad.main([*argv, "--force"]) == 0
    assert len(ledger_entries(temporary_ledger)) == 2 and recorded_before_build == [1, 2]


def test_main_records_nothing_when_the_zips_cannot_be_read(tmp_path, monkeypatch, temporary_ledger):
    jsons = write_jsons(tmp_path, make_meta(lights=("01", "02")))
    rows = read_meta(jsons)
    argv = ["--size", str(SIZE), "--jsons", str(jsons), "--out", str(tmp_path / "cache")]

    def no_decoding(data, size):
        raise AssertionError("nothing may be decoded")

    monkeypatch.setattr(cache, "_decode_image", no_decoding)
    # No zips at all: no test image can be read, so the ledger gets no line.
    with pytest.raises(FileNotFoundError):
        m2ad.main([*argv, "--zips", str(tmp_path / "nowhere")])
    assert not temporary_ledger.exists()
    # One wanted member is not in its zip: found in the central directory, before the ledger line.
    gone = [r.img_path for r in rows if r.category == "Bird"][3]
    write_zips(tmp_path / "zips", rows, skip={gone})
    with pytest.raises(FileNotFoundError, match="1 images are not in"):
        m2ad.main([*argv, "--zips", str(tmp_path / "zips")])
    assert not temporary_ledger.exists()
    assert not (tmp_path / "cache").exists() or not list((tmp_path / "cache").iterdir())
    # With the member back the same command builds the cache and records the read once.
    write_zips(tmp_path / "zips", rows)
    monkeypatch.setattr(cache, "_decode_image", lambda data, size: np.zeros((size, size, 3), dtype=np.uint8))
    assert m2ad.main([*argv, "--zips", str(tmp_path / "zips")]) == 0
    assert len(ledger_entries(temporary_ledger)) == 1


def test_count_table_lists_every_label(tmp_path):
    rows = read_meta(write_jsons(tmp_path, make_meta()))
    table = m2ad.count_table(rows).splitlines()
    # Motor test: 1 normal specimen (30 images); hole 30 + scratch 10 defects; scratch 20 + dent 30 excluded.
    assert table[1].split() == ["Motor", "train", "2", "60", "0", "0"]
    assert table[2].split() == ["Motor", "test", "4", "30", "40", "50"]
    assert table[4].split() == ["Bird", "test", "4", "30", "40", "50"]
    motor_000 = next(line for line in table if line.startswith("Motor 000"))
    assert motor_000.split()[2:] == ["1/2/1"] * 10
    motor_120 = next(line for line in table if line.startswith("Motor 120"))
    assert motor_120.split()[2:] == ["1/1/2"] * 10
