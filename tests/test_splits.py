import hashlib
import io
import tarfile
from collections import Counter, defaultdict

import pytest

from defect_inspect import paths, splits
from defect_inspect.splits import ManifestRow, SealedTestError
from defect_inspect.visa import CATEGORIES, VisaRow

# Synthetic split: (train normal, train anomaly, test normal, test anomaly) per category.
SIZES = {"pcb1": (23, 60, 11, 40), "candle": (31, 60, 9, 40), "capsules": (5, 27, 3, 4)}


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


@pytest.fixture(scope="module")
def manifest() -> list[ManifestRow]:
    rows, types = make_rows()
    return splits.build_manifest(rows, types)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_path_key_is_sha256_of_path_or_salted_path():
    path = "candle/Data/Images/Normal/0836.JPG"
    assert splits.path_key(path) == sha(path)
    assert splits.path_key(path, salt="2") == sha(f"2:{path}")
    assert splits.path_key(path, salt="0") == sha(f"0:{path}") != splits.path_key(path)
    assert splits.path_key(path, "") == splits.path_key(path, salt="") == sha(path)
    assert splits.path_key("a", "0") == sha("0:a")


@pytest.mark.parametrize("salt", [0, 1, None, False, 0.0, b"0"])
def test_path_key_rejects_salts_that_are_not_strings(salt):
    # An integer seed 0 is falsy: it must not silently give the unsalted key used for folds.
    with pytest.raises(TypeError, match="salt"):
        splits.path_key("candle/Data/Images/Anomaly/000.JPG", salt)
    with pytest.raises(TypeError):
        splits.path_key(None)


def test_label_subset_with_seed_zero_is_salted(manifest):
    pool = [r.image for r in manifest if r.category == "pcb1" and r.role == "label_pool"]
    got = [r.image for r in splits.label_subset(manifest, "pcb1", 40, 0)]
    assert got == sorted(pool, key=lambda image: sha(f"0:{image}")) != sorted(pool, key=sha)


def test_constants():
    assert splits.N_FOLDS == 5 and splits.N_DEV_DEFECTS == 20
    assert splits.K_VALUES == (5, 10, 20, 40) and splits.SEEDS == (0, 1, 2)
    assert splits.ROLES == ("pool_normal", "dev_defect", "label_pool", "test_normal", "test_defect")
    assert splits.SEALED_ROLES == ("test_normal", "test_defect")


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


def test_folds_follow_hash_rank_and_are_balanced(manifest):
    for category in SIZES:
        pool = [r for r in manifest if r.category == category and r.role == "pool_normal"]
        ranked = sorted(pool, key=lambda r: sha(r.image))
        assert [r.fold for r in ranked] == [i % 5 for i in range(len(ranked))]
        sizes = Counter(r.fold for r in pool)
        assert set(sizes) == {0, 1, 2, 3, 4}
        assert max(sizes.values()) - min(sizes.values()) <= 1
    assert all(r.fold == -1 for r in manifest if r.role != "pool_normal")


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


def test_build_is_independent_of_input_order(manifest):
    rows, types = make_rows()
    assert splits.build_manifest(rows[::-1], types) == manifest


def test_build_rejects_repeated_images():
    rows, types = make_rows()
    with pytest.raises(ValueError, match="more than once"):
        splits.build_manifest([*rows, rows[0]], types)


def test_manifest_round_trip_with_lf(manifest, tmp_path):
    path = tmp_path / "manifests" / "visa.csv"
    splits.write_manifest(manifest, path)
    raw = path.read_bytes()
    assert b"\r" not in raw
    assert raw.startswith(b"image,mask,category,label,role,fold,defect_types\n")
    assert raw.count(b"\n") == len(manifest) + 1
    assert splits.read_manifest(path) == manifest


def test_read_manifest_rejects_other_headers(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("image,mask\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="header"):
        splits.read_manifest(path)


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


def test_label_subset_rejects_k_beyond_the_pool(manifest):
    with pytest.raises(ValueError):
        splits.label_subset(manifest, "capsules", 10, 0)  # only 7 in this pool
    with pytest.raises(ValueError):
        splits.label_subset(manifest, "pcb1", -1, 0)


def test_pool_and_holdout_folds():
    assert splits.pool_folds("dev") == (1, 2, 3, 4)
    assert splits.pool_folds("test") == (0, 1, 2, 3, 4)
    assert splits.holdout_fold("dev") == 1
    assert splits.holdout_fold("test") == 0
    for fn in (splits.pool_folds, splits.holdout_fold):
        with pytest.raises(ValueError):
            fn("val")


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


def test_select_test_protocol_with_allow_test(manifest):
    eval_normal = splits.select(manifest, protocol="test", part="eval_normal", allow_test=True)
    eval_defect = splits.select(manifest, protocol="test", part="eval_defect", allow_test=True)
    assert eval_normal == [r for r in manifest if r.role == "test_normal"]
    assert eval_defect == [r for r in manifest if r.role == "test_defect"]
    # allow_test does not change what the dev protocol returns.
    assert splits.select(manifest, protocol="dev", part="eval_defect", allow_test=True) == [
        r for r in manifest if r.role == "dev_defect"
    ]


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


def test_select_rejects_unknown_names(manifest):
    with pytest.raises(ValueError):
        splits.select(manifest, protocol="val", part="pool_normal")
    with pytest.raises(ValueError):
        splits.select(manifest, protocol="dev", part="test_normal")
    # An unknown part is a ValueError even where the seal would otherwise apply.
    with pytest.raises(ValueError):
        splits.select(manifest, protocol="test", part="label_pool")


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


def test_main_writes_manifest_from_tar(manifest, tmp_path, capsys):
    rows, types = make_rows()
    tar_path = tmp_path / "visa.tar"
    make_tar(tar_path, rows, types)
    out = tmp_path / "out" / "visa.csv"
    assert splits.main(["--tar", str(tar_path), "--out", str(out), "--no-verify"]) == 0
    assert splits.read_manifest(out) == manifest
    printed = capsys.readouterr().out
    assert "pool_normal" in printed and "capsules" in printed and "total" in printed


def test_main_refuses_to_write_when_counts_are_not_the_official_ones(tmp_path, capsys):
    rows, types = make_rows()
    tar_path = tmp_path / "visa.tar"
    make_tar(tar_path, rows, types)
    out = tmp_path / "visa.csv"
    assert splits.main(["--tar", str(tar_path), "--out", str(out)]) == 1
    assert not out.exists()
    assert "ERROR" in capsys.readouterr().out


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
