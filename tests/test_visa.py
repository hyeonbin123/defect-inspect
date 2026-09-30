import io
import tarfile

import pytest

from defect_inspect import visa

SPLIT = (
    "object,split,label,image,mask\r\n"
    "candle,train,normal,candle/Data/Images/Normal/0001.JPG,\r\n"
    "candle,train,anomaly,candle/Data/Images/Anomaly/000.JPG,candle/Data/Masks/Anomaly/000.png\r\n"
    "pcb1,test,normal,pcb1/Data/Images/Normal/0002.JPG,\r\n"
    "pcb1,test,anomaly,pcb1/Data/Images/Anomaly/001.JPG,pcb1/Data/Masks/Anomaly/001.png\r\n"
)
CANDLE_ANNO = (
    "image,label,mask\r\n"
    "candle/Data/Images/Normal/0001.JPG,normal,\r\n"
    'candle/Data/Images/Anomaly/000.JPG,"weird candle wick, chunk of wax missing",'
    "candle/Data/Masks/Anomaly/000.png\r\n"
)
PCB1_ANNO = (
    "image,label,mask\n"
    "pcb1/Data/Images/Normal/0002.JPG,normal,\n"
    "pcb1/Data/Images/Anomaly/001.JPG,melt,pcb1/Data/Masks/Anomaly/001.png\n"
)


def make_tar(path, members: dict[str, str]) -> None:
    with tarfile.open(path, "w") as tar:
        for name, text in members.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


@pytest.fixture
def tar_path(tmp_path):
    path = tmp_path / "visa.tar"
    members = {
        "candle/image_anno.csv": CANDLE_ANNO,
        "candle/Data/Images/Normal/0001.JPG": "not an image, never decoded",
        "pcb1/image_anno.csv": PCB1_ANNO,
        "LICENSE-DATASET": "CC BY 4.0",
        "split_csv/1cls.csv": "object,split,label,image,mask\r\n",
        "split_csv/2cls_highshot.csv": SPLIT,
    }
    make_tar(path, members)
    return path


def test_categories():
    assert len(visa.CATEGORIES) == 12
    assert list(visa.CATEGORIES) == sorted(visa.CATEGORIES)


def test_read_split_csv(tar_path):
    rows = visa.read_split_csv(tar_path)
    assert rows == visa.read_split_csv(tar_path, "2cls_highshot")
    assert len(rows) == 4
    assert rows[0] == visa.VisaRow("candle", "train", "normal", "candle/Data/Images/Normal/0001.JPG", "")
    assert rows[3] == visa.VisaRow(
        "pcb1", "test", "anomaly", "pcb1/Data/Images/Anomaly/001.JPG", "pcb1/Data/Masks/Anomaly/001.png"
    )
    assert visa.read_split_csv(tar_path, "1cls") == []


def test_read_split_csv_missing_member(tar_path):
    with pytest.raises(KeyError):
        visa.read_split_csv(tar_path, "2cls_fewshot")


def test_read_defect_types_only_anomalies_sorted_and_stripped(tar_path):
    types = visa.read_defect_types(tar_path)
    assert types == {
        "candle/Data/Images/Anomaly/000.JPG": ("chunk of wax missing", "weird candle wick"),
        "pcb1/Data/Images/Anomaly/001.JPG": ("melt",),
    }


def test_bad_header_is_rejected(tmp_path):
    path = tmp_path / "bad.tar"
    make_tar(path, {"split_csv/2cls_highshot.csv": "object,split,label,path,mask\n"})
    with pytest.raises(ValueError, match="header"):
        visa.read_split_csv(path)


def test_bad_values_are_rejected(tmp_path):
    path = tmp_path / "bad.tar"
    text = "object,split,label,image,mask\ncandle,val,normal,candle/Data/Images/Normal/0001.JPG,\n"
    make_tar(path, {"split_csv/2cls_highshot.csv": text})
    with pytest.raises(ValueError, match="split/label"):
        visa.read_split_csv(path)


def split_tar(tmp_path, *lines: str):
    path = tmp_path / "rows.tar"
    make_tar(path, {"split_csv/2cls_highshot.csv": "\n".join(["object,split,label,image,mask", *lines, ""])})
    return path


def test_normal_row_without_the_trailing_comma_has_an_empty_mask(tmp_path):
    path = split_tar(tmp_path, "candle,train,normal,candle/Data/Images/Normal/0001.JPG")  # 4 columns
    rows = visa.read_split_csv(path)
    assert rows == [visa.VisaRow("candle", "train", "normal", "candle/Data/Images/Normal/0001.JPG", "")]
    assert rows[0].mask == "" and rows[0].mask is not None


@pytest.mark.parametrize(
    "line",
    [
        "candle,train,anomaly,candle/Data/Images/Anomaly/000.JPG,",  # anomaly without a mask
        "candle,train,anomaly,candle/Data/Images/Anomaly/000.JPG",  # same, 4 columns
        "candle,train,normal,candle/Data/Images/Normal/0001.JPG,candle/Data/Masks/Anomaly/000.png",
    ],
)
def test_mask_must_agree_with_the_label(tmp_path, line):
    with pytest.raises(ValueError, match="mask"):
        visa.read_split_csv(split_tar(tmp_path, line))


@pytest.mark.parametrize(
    "line",
    [
        "candle,train,defect,candle/Data/Images/Anomaly/000.JPG,",  # unknown label, no mask
        "candle,train,defect,candle/Data/Images/Anomaly/000.JPG,candle/Data/Masks/Anomaly/000.png",
        "candle,train,Normal,candle/Data/Images/Normal/0001.JPG,",
        "candle,train,,candle/Data/Images/Normal/0001.JPG,",
    ],
)
def test_unknown_labels_are_rejected(tmp_path, line):
    with pytest.raises(ValueError, match="split/label"):
        visa.read_split_csv(split_tar(tmp_path, line))


@pytest.mark.parametrize(
    "line",
    [
        "candle,train,normal",  # image column missing
        "candle,train,normal,candle/Data/Images/Normal/0001.JPG,,extra",  # one column too many
    ],
)
def test_rows_with_the_wrong_number_of_columns_are_rejected(tmp_path, line):
    with pytest.raises(ValueError, match="columns"):
        visa.read_split_csv(split_tar(tmp_path, line))


def test_defect_type_names_are_deduplicated(tmp_path):
    path = tmp_path / "types.tar"
    anno = (
        "image,label,mask\n"
        'c/Data/Images/Anomaly/1.JPG,"b, a ,b",c/Data/Masks/Anomaly/1.png\n'
        'c/Data/Images/Anomaly/2.JPG,"melt,melt",c/Data/Masks/Anomaly/2.png\n'
        'c/Data/Images/Anomaly/3.JPG,"z,,y, ",c/Data/Masks/Anomaly/3.png\n'
    )
    make_tar(path, {"c/image_anno.csv": anno})
    assert visa.read_defect_types(path) == {
        "c/Data/Images/Anomaly/1.JPG": ("a", "b"),
        "c/Data/Images/Anomaly/2.JPG": ("melt",),
        "c/Data/Images/Anomaly/3.JPG": ("y", "z"),
    }


def test_tar_is_scanned_once_per_process(tar_path, monkeypatch):
    visa.read_split_csv(tar_path)
    opened = []
    real_open = tarfile.open

    def counting_open(*args, **kwargs):
        opened.append(args)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(visa.tarfile, "open", counting_open)
    visa.read_split_csv(tar_path)
    visa.read_defect_types(tar_path)
    assert opened == []


def test_member_names_with_dot_prefix(tmp_path):
    path = tmp_path / "dot.tar"
    make_tar(path, {"./split_csv/2cls_highshot.csv": SPLIT, "./pcb1/image_anno.csv": PCB1_ANNO})
    assert len(visa.read_split_csv(path)) == 4
    assert visa.read_defect_types(path) == {"pcb1/Data/Images/Anomaly/001.JPG": ("melt",)}
