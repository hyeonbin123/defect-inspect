# [파일 역할] VisA tar 읽기 모듈 `defect_inspect.visa`를 검증하는 테스트 묶음이다. tar를 풀지 않고 공식 분할
# CSV(split_csv/*.csv)와 범주별 결함 유형 CSV(<범주>/image_anno.csv)를 읽는 함수가 행을 정확히 읽고, 이상한
# 행은 거절하며, tar를 프로세스당 한 번만 훑는지 확인한다.
# [왜 필요한가] 공식 분할의 test가 그대로 봉인 테스트가 되므로(docs/experiments.md 봉인 테스트 절), 행 하나를
# 잘못 읽어 다른 역할로 섞이면 결과 전체가 오염된다. 결함 유형은 2단계 H7(학습에서 보지 못한 결함 유형)의
# 기준이다. 실제 tar(1.93GB) 없이 작은 합성 tar로 규칙을 시험한다.
# [설계 메모] 합성 CSV는 실제 파일처럼 CRLF와 LF 줄 끝, 따옴표 안 쉼표로 이어진 유형 이름, 이미지가 아닌
# 멤버(라이선스 파일, 디코딩되지 않는 '이미지')를 섞어 둔다. torch 없이 표준 라이브러리만으로 돌아 CI에서
# 실행된다.

import io
import tarfile

import pytest

from defect_inspect import visa

# 합성 분할 CSV(CRLF): 학습 정상·결함, 테스트 정상·결함 한 줄씩. 정상 행은 마스크 칸이 비어 있다.
SPLIT = (
    "object,split,label,image,mask\r\n"
    "candle,train,normal,candle/Data/Images/Normal/0001.JPG,\r\n"
    "candle,train,anomaly,candle/Data/Images/Anomaly/000.JPG,candle/Data/Masks/Anomaly/000.png\r\n"
    "pcb1,test,normal,pcb1/Data/Images/Normal/0002.JPG,\r\n"
    "pcb1,test,anomaly,pcb1/Data/Images/Anomaly/001.JPG,pcb1/Data/Masks/Anomaly/001.png\r\n"
)
# 범주별 결함 유형 CSV. candle은 실제 파일처럼 유형 두 개를 따옴표 안에 쉼표로 이었고(CRLF), pcb1은 유형
# 하나에 LF 줄 끝이다.
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


# 멤버 이름 → 텍스트 표로 작은 tar를 만든다.
def make_tar(path, members: dict[str, str]) -> None:
    with tarfile.open(path, "w") as tar:
        for name, text in members.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


# CSV 멤버와 함께 이미지가 아닌 멤버(디코딩되지 않는 가짜 이미지, LICENSE-DATASET)를 섞은 합성 tar. 1cls
# 분할은 헤더뿐이라 빈 목록이 나와야 한다.
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


# 범주는 12개이고 알파벳 순이다. 이 순서가 매니페스트와 결과 표의 범주 순서 기준이다.
def test_categories():
    assert len(visa.CATEGORIES) == 12
    assert list(visa.CATEGORIES) == sorted(visa.CATEGORIES)


# 분할 CSV를 파일 순서대로 VisaRow로 읽는다(기본 이름은 2cls_highshot). 정상 행의 마스크는 빈 문자열이고,
# 헤더뿐인 분할은 빈 목록이다.
def test_read_split_csv(tar_path):
    rows = visa.read_split_csv(tar_path)
    assert rows == visa.read_split_csv(tar_path, "2cls_highshot")
    assert len(rows) == 4
    assert rows[0] == visa.VisaRow("candle", "train", "normal", "candle/Data/Images/Normal/0001.JPG", "")
    assert rows[3] == visa.VisaRow(
        "pcb1", "test", "anomaly", "pcb1/Data/Images/Anomaly/001.JPG", "pcb1/Data/Masks/Anomaly/001.png"
    )
    assert visa.read_split_csv(tar_path, "1cls") == []


# tar에 없는 분할을 달라고 하면 KeyError다.
def test_read_split_csv_missing_member(tar_path):
    with pytest.raises(KeyError):
        visa.read_split_csv(tar_path, "2cls_fewshot")


# 결함 유형 표는 이상 이미지만 담고, 쉼표로 나눈 유형 이름의 앞뒤 공백을 지운 뒤 정렬한 튜플이다(같은 이미지면
# 늘 같은 튜플).
def test_read_defect_types_only_anomalies_sorted_and_stripped(tar_path):
    types = visa.read_defect_types(tar_path)
    assert types == {
        "candle/Data/Images/Anomaly/000.JPG": ("chunk of wax missing", "weird candle wick"),
        "pcb1/Data/Images/Anomaly/001.JPG": ("melt",),
    }


# 열 이름이 기대와 다르면(image 대신 path) 엉뚱한 칸을 조용히 읽지 않고 ValueError로 멈춘다.
def test_bad_header_is_rejected(tmp_path):
    path = tmp_path / "bad.tar"
    make_tar(path, {"split_csv/2cls_highshot.csv": "object,split,label,path,mask\n"})
    with pytest.raises(ValueError, match="header"):
        visa.read_split_csv(path)


# split이 train/test가 아니면(val) 멈춘다. 분할 규칙이 다루지 않는 값이다.
def test_bad_values_are_rejected(tmp_path):
    path = tmp_path / "bad.tar"
    text = "object,split,label,image,mask\ncandle,val,normal,candle/Data/Images/Normal/0001.JPG,\n"
    make_tar(path, {"split_csv/2cls_highshot.csv": text})
    with pytest.raises(ValueError, match="split/label"):
        visa.read_split_csv(path)


# 헤더 + 주어진 행으로 분할 CSV 하나만 든 tar를 만든다(행 단위 검사용).
def split_tar(tmp_path, *lines: str):
    path = tmp_path / "rows.tar"
    make_tar(path, {"split_csv/2cls_highshot.csv": "\n".join(["object,split,label,image,mask", *lines, ""])})
    return path


# 끝 쉼표가 없어 마스크 칸 자체가 없는 정상 행(4열)은 None이 아니라 빈 마스크로 읽는다. csv.DictReader는 빠진
# 칸을 None으로 주므로 이를 따로 처리한다.
def test_normal_row_without_the_trailing_comma_has_an_empty_mask(tmp_path):
    path = split_tar(tmp_path, "candle,train,normal,candle/Data/Images/Normal/0001.JPG")  # 4열
    rows = visa.read_split_csv(path)
    assert rows == [visa.VisaRow("candle", "train", "normal", "candle/Data/Images/Normal/0001.JPG", "")]
    assert rows[0].mask == "" and rows[0].mask is not None


# 마스크와 라벨이 맞아야 한다: 이상 행인데 마스크가 없거나(빈 칸, 4열), 정상 행인데 마스크가 있으면 거절한다.
# 마스크 유무로 라벨을 한 번 더 교차 확인하는 것이다.
@pytest.mark.parametrize(
    "line",
    [
        "candle,train,anomaly,candle/Data/Images/Anomaly/000.JPG,",  # 마스크 없는 이상 행
        "candle,train,anomaly,candle/Data/Images/Anomaly/000.JPG",  # 같은 경우, 4열
        "candle,train,normal,candle/Data/Images/Normal/0001.JPG,candle/Data/Masks/Anomaly/000.png",
    ],
)
def test_mask_must_agree_with_the_label(tmp_path, line):
    with pytest.raises(ValueError, match="mask"):
        visa.read_split_csv(split_tar(tmp_path, line))


# 라벨은 정확히 normal 또는 anomaly여야 한다. 모르는 값(defect), 대소문자가 다른 값(Normal), 빈 값은 마스크
# 유무와 상관없이 거절한다.
@pytest.mark.parametrize(
    "line",
    [
        "candle,train,defect,candle/Data/Images/Anomaly/000.JPG,",  # 모르는 라벨, 마스크 없음
        "candle,train,defect,candle/Data/Images/Anomaly/000.JPG,candle/Data/Masks/Anomaly/000.png",
        "candle,train,Normal,candle/Data/Images/Normal/0001.JPG,",
        "candle,train,,candle/Data/Images/Normal/0001.JPG,",
    ],
)
def test_unknown_labels_are_rejected(tmp_path, line):
    with pytest.raises(ValueError, match="split/label"):
        visa.read_split_csv(split_tar(tmp_path, line))


# 열 수가 틀린 행(이미지 열 없음, 열 하나 더 많음)은 거절한다. 마스크 열만 빠진 경우(앞 테스트)와 구별한다.
@pytest.mark.parametrize(
    "line",
    [
        "candle,train,normal",  # 이미지 열 없음
        "candle,train,normal,candle/Data/Images/Normal/0001.JPG,,extra",  # 열이 하나 더 많음
    ],
)
def test_rows_with_the_wrong_number_of_columns_are_rejected(tmp_path, line):
    with pytest.raises(ValueError, match="columns"):
        visa.read_split_csv(split_tar(tmp_path, line))


# 유형 이름의 중복을 없애고, 빈 이름과 공백만 있는 이름은 버리며, 정렬한다.
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


# 한 번 훑은 뒤에는 분할 CSV와 결함 유형을 다시 읽어도 tar를 다시 열지 않는다(경로·크기·수정 시각을 키로 한
# 캐시). 실제 tar는 CSV가 맨 끝에 있어 한 번 훑는 데 헤더 약 1만 2천 개를 지나야 하고, 데이터 디스크가 HDD라
# 반복해 훑으면 느리다(visa._scan_csv_members docstring).
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


# tar 멤버 이름 앞에 './'가 붙어 있어도(tar를 만든 방식에 따라 생김) 같은 CSV로 찾는다.
def test_member_names_with_dot_prefix(tmp_path):
    path = tmp_path / "dot.tar"
    make_tar(path, {"./split_csv/2cls_highshot.csv": SPLIT, "./pcb1/image_anno.csv": PCB1_ANNO})
    assert len(visa.read_split_csv(path)) == 4
    assert visa.read_defect_types(path) == {"pcb1/Data/Images/Anomaly/001.JPG": ("melt",)}
