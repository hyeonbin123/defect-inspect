# [파일 역할] VisA tar(1.93GB)를 받는 이어받기·SHA-256 검증 다운로더(`defect_inspect.download`)를 검증하는
# 테스트 묶음이다.
# [왜 필요한가] 큰 파일을 받으므로 끊겨도 처음부터 다시 받지 않게 '.part' 파일에서 이어받는다.
# 그런데 이어받기는 잘못하면 서로 다른 내용이 이어 붙은 깨진 파일을 만든다. 그래서 서버가 Range를
# 무시하거나(200), 다른 위치부터 보내거나(206), 범위 밖이라고 답하는(416) 경우마다 part 파일을 이어
# 쓸지·버릴지가 맞게 갈리는지, 마지막에 해시가 틀리면 결과 파일을 남기지 않는지를 확인한다. 데이터 출처의
# 무결성은 docs/experiments.md 데이터 절의 sha256으로 고정돼 있다.
# [설계] 네트워크를 쓰지 않는다. download의 opener 인자에 FakeServer를 넘겨 서버 응답을 흉내 내고, 받은 Range
# 헤더를 기록해 몇 번 어떤 요청을 보냈는지 단언한다.

import hashlib
import io
import urllib.error

import pytest

from defect_inspect import download as dl

# .invalid 최상위 도메인은 예약돼 절대 해석되지 않으므로, 실수로 실제 요청이 나가도 외부로 연결되지 않는다.
# CONTENT는 0~255 바이트를 40번 되풀이한 10,240바이트로, 자른 위치가 내용으로 구별된다.
URL = "https://example.invalid/file.bin"
CONTENT = bytes(range(256)) * 40
SHA = hashlib.sha256(CONTENT).hexdigest()


# urlopen 응답 흉내: 본문을 BytesIO로 읽고 status와 headers만 붙인다. with 문(문맥 관리자)도 BytesIO가
# 지원한다.
class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, status: int, headers: dict[str, str]):
        super().__init__(body)
        self.status = status
        self.headers = headers


# urlopen 대신 넘기는 가짜 서버. 기본은 Range 요청에 206과 Content-Range로 답하고, 범위가 끝 이후면 416을
# 낸다. 옵션으로 Range를 무시하기(honour_range=False), 416에 전체 크기를 빼기(total_in_416=False), 206을
# 요청과 다른 위치부터 보내기(start_shift)를 흉내 낸다. ranges에 요청마다 받은 Range 헤더(없으면 None)가
# 쌓인다.
class FakeServer:
    """urlopen 대신 CONTENT를 내주는 가짜 서버. 무시하라는 옵션이 없으면 Range를 따른다."""

    def __init__(
        self,
        content: bytes = CONTENT,
        *,
        honour_range: bool = True,
        total_in_416: bool = True,
        start_shift: int = 0,
    ):
        self.content = content
        self.honour_range = honour_range
        self.total_in_416 = total_in_416  # False: 416 응답에 Content-Range를 싣지 않는다
        self.start_shift = start_shift  # 0이 아니면 206 응답이 요청과 다른 위치부터 시작한다
        self.ranges: list[str | None] = []

    def __call__(self, request):
        assert request.full_url == URL
        requested = request.get_header("Range")
        self.ranges.append(requested)
        total = len(self.content)
        if requested and self.honour_range:
            start = int(requested.removeprefix("bytes=").rstrip("-"))
            if start >= total:
                headers = {"Content-Range": f"bytes */{total}"} if self.total_in_416 else {}
                raise urllib.error.HTTPError(URL, 416, "Range Not Satisfiable", headers, io.BytesIO())
            start += self.start_shift
            body = self.content[start:]
            headers = {
                "Content-Length": str(len(body)),
                "Content-Range": f"bytes {start}-{total - 1}/{total}",
            }
            return FakeResponse(body, 206, headers)
        return FakeResponse(self.content, 200, {"Content-Length": str(total)})


# 확인: 조각 단위로 읽는 sha256_file이 hashlib에 한 번에 넣은 값과 같다. 7바이트처럼 나누어떨어지지 않는 조각
# 크기에서도 같은지 본다.
def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(CONTENT)
    assert dl.sha256_file(path) == SHA
    assert dl.sha256_file(path, chunk=7) == SHA


# 확인: 처음 받을 때는 Range 없이 한 번 요청하고, 없는 상위 폴더를 만들며, 해시가 맞으면 .part를 최종 이름으로
# 바꾸고 .part를 남기지 않는다.
def test_fresh_download_verifies_and_renames(tmp_path):
    dest = tmp_path / "sub" / "file.bin"
    server = FakeServer()
    assert dl.download(URL, dest, expected_sha256=SHA, opener=server) == dest
    assert dest.read_bytes() == CONTENT
    assert not dest.with_name("file.bin.part").exists()
    assert server.ranges == [None]


# 확인: 이미 있는 파일은 해시가 맞으면(대문자 해시도 소문자로 맞춰 비교) 또는 해시를 주지 않으면 요청을 하나도
# 보내지 않고 그대로 돌려준다. 재실행이 1.93GB를 다시 받지 않게 한다.
def test_existing_file_is_returned_without_a_request(tmp_path):
    dest = tmp_path / "file.bin"
    dest.write_bytes(CONTENT)
    server = FakeServer()
    assert dl.download(URL, dest, expected_sha256=SHA.upper(), opener=server) == dest
    assert dl.download(URL, dest, opener=server) == dest
    assert server.ranges == []


# 확인: 최종 파일이 있어도 해시가 다르면 새로 받아 바꾼다.
def test_existing_file_with_wrong_hash_is_replaced(tmp_path):
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"stale")
    dl.download(URL, dest, expected_sha256=SHA, opener=FakeServer())
    assert dest.read_bytes() == CONTENT


# 확인: .part가 3,000바이트 있으면 'bytes=3000-' Range로 나머지만 받아 이어 붙이고, 해시 검증 뒤 최종 파일로
# 바꾼다.
def test_resume_sends_range_and_appends(tmp_path):
    dest = tmp_path / "file.bin"
    part = tmp_path / "file.bin.part"
    part.write_bytes(CONTENT[:3000])
    server = FakeServer()
    dl.download(URL, dest, expected_sha256=SHA, opener=server)
    assert server.ranges == ["bytes=3000-"]
    assert dest.read_bytes() == CONTENT
    assert not part.exists()


# 확인: 서버가 Range를 무시하고 200으로 전체를 보내면 part 파일을 덮어쓴다(이어 붙이지 않는다). part가 원래
# 내용의 앞부분이 아니어도 결과가 맞는다.
def test_server_ignoring_range_restarts_from_zero(tmp_path):
    dest = tmp_path / "file.bin"
    part = tmp_path / "file.bin.part"
    part.write_bytes(b"x" * 3000)  # 원래 내용의 앞부분조차 아니다
    server = FakeServer(honour_range=False)
    dl.download(URL, dest, expected_sha256=SHA, opener=server)
    assert server.ranges == ["bytes=3000-"]
    assert dest.read_bytes() == CONTENT


# 확인: part가 이미 전부 받아진 상태면 416과 함께 Content-Range로 전체 크기를 받고, 그 크기가 part 크기와
# 같으므로 다시 받지 않고 해시만 검증한다.
def test_complete_part_file_is_verified_without_downloading_again(tmp_path):
    dest = tmp_path / "file.bin"
    part = tmp_path / "file.bin.part"
    part.write_bytes(CONTENT)
    server = FakeServer()
    dl.download(URL, dest, expected_sha256=SHA, opener=server)
    assert server.ranges == [f"bytes={len(CONTENT)}-"]
    assert dest.read_bytes() == CONTENT


# 확인: part가 실제 파일보다 길면(416의 전체 크기가 part 크기와 다름) part를 버리고 Range 없이 처음부터
# 받는다.
def test_oversized_part_file_is_discarded(tmp_path):
    dest = tmp_path / "file.bin"
    part = tmp_path / "file.bin.part"
    part.write_bytes(CONTENT + b"junk")
    server = FakeServer()
    dl.download(URL, dest, expected_sha256=SHA, opener=server)
    assert server.ranges == [f"bytes={len(CONTENT) + 4}-", None]
    assert dest.read_bytes() == CONTENT


# 확인: 416 응답에 Content-Range가 없으면 part가 완전한지 알 수 없으므로, 길이가 같든 길든 내용이 엉터리든
# part를 버리고 처음부터 받는다. 해시를 주지 않은 경우에도 그렇다.
# 왜: 해시가 없을 때 '길이만 맞으니 완전하다'고 믿으면 깨진 파일을 그대로 최종 파일로 둘 수 있다(download.py의
# 416 처리 주석).
@pytest.mark.parametrize("sha", [None, SHA], ids=["no-hash", "hash"])
@pytest.mark.parametrize("kind", ["oversized", "complete", "garbage"])
def test_416_without_a_total_discards_the_part_and_restarts(tmp_path, sha, kind):
    # Content-Range가 없으면 part 파일이 파일 전체라는 근거가 없다. 실제 파일보다 길 수도 있다.
    part_content = {"oversized": CONTENT + b"junk", "complete": CONTENT, "garbage": b"x" * 14000}[kind]
    dest = tmp_path / "file.bin"
    part = tmp_path / "file.bin.part"
    part.write_bytes(part_content)
    server = FakeServer(total_in_416=False)
    dl.download(URL, dest, expected_sha256=sha, opener=server)
    assert server.ranges == [f"bytes={len(part_content)}-", None]
    assert dest.read_bytes() == CONTENT
    assert not part.exists()


# 확인: 206 응답의 시작 위치가 요청한 오프셋과 다르면(앞으로 10, 뒤로 7) 이어 붙이지 않고 처음부터 다시
# 받는다. 겹치거나 빠진 바이트가 생긴 파일을 막는다.
@pytest.mark.parametrize("sha", [None, SHA], ids=["no-hash", "hash"])
@pytest.mark.parametrize("shift", [-10, 7])
def test_206_that_starts_elsewhere_restarts_from_zero(tmp_path, sha, shift):
    dest = tmp_path / "file.bin"
    part = tmp_path / "file.bin.part"
    part.write_bytes(CONTENT[:3000])
    server = FakeServer(start_shift=shift)
    dl.download(URL, dest, expected_sha256=sha, opener=server)
    assert server.ranges == ["bytes=3000-", None]
    assert dest.read_bytes() == CONTENT


# 확인: opener를 주지 않으면 urllib.request.urlopen을 timeout=60초로 부른다.
# 왜: 시간 제한이 없으면 멈춘 전송이 영원히 걸려 있게 된다(_urlopen_with_timeout 주석). monkeypatch로
# urlopen을 가짜로 바꿔 호출 인자만 본다.
def test_default_opener_has_a_timeout(tmp_path, monkeypatch):
    seen = []

    def fake_urlopen(request, *args, **kwargs):
        seen.append((request.full_url, args, kwargs))
        return FakeResponse(CONTENT, 200, {"Content-Length": str(len(CONTENT))})

    monkeypatch.setattr(dl.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "file.bin"
    assert dl.download(URL, dest, expected_sha256=SHA) == dest
    assert dest.read_bytes() == CONTENT
    assert seen == [(URL, (), {"timeout": 60})]
    assert dl.TIMEOUT_S == 60


# 확인: 본문을 읽다가 소켓 시간 초과(TimeoutError)가 나면 예외를 그대로 올리되, 그때까지 받은 4,096바이트는
# part에 남고 최종 파일은 생기지 않는다. 다음 호출은 'bytes=4096-'부터 이어받아 완성한다.
def test_stalled_connection_keeps_the_part_file_for_resuming(tmp_path):
    dest = tmp_path / "file.bin"
    part = tmp_path / "file.bin.part"

    class Stalling(FakeResponse):
        def read(self, size=-1):
            if self.tell() >= 4096:
                raise TimeoutError("timed out")  # 본문 중간에 소켓 시간 초과가 나면 이 예외가 올라온다
            return super().read(min(size, 4096))

    def stalling(request):
        return Stalling(CONTENT, 200, {"Content-Length": str(len(CONTENT))})

    with pytest.raises(TimeoutError):
        dl.download(URL, dest, expected_sha256=SHA, opener=stalling)
    assert not dest.exists()
    assert part.read_bytes() == CONTENT[:4096]
    server = FakeServer()
    dl.download(URL, dest, expected_sha256=SHA, opener=server)
    assert server.ranges == ["bytes=4096-"]
    assert dest.read_bytes() == CONTENT


# 확인: 받은 내용의 해시가 기대값과 다르면 'sha256 mismatch' ValueError를 내고 part와 최종 파일을 둘 다 남기지
# 않는다(틀린 파일이 다음 실행에서 이어받기의 출발점이 되지 않게).
def test_hash_mismatch_raises_and_removes_part(tmp_path):
    dest = tmp_path / "file.bin"
    with pytest.raises(ValueError, match="sha256 mismatch"):
        dl.download(URL, dest, expected_sha256="0" * 64, opener=FakeServer())
    assert not dest.exists()
    assert not dest.with_name("file.bin.part").exists()


# 확인: 내용이 틀린 part에 정상 나머지를 이어 붙이면 해시 검증에서 걸려 part가 지워지고, 다음 호출은 처음부터
# 받아 성공한다. 이어받기의 약점(앞부분이 틀려도 알 수 없음)을 해시가 막는 경로다.
def test_corrupt_part_file_fails_the_hash_check(tmp_path):
    dest = tmp_path / "file.bin"
    part = tmp_path / "file.bin.part"
    part.write_bytes(b"x" * 3000)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        dl.download(URL, dest, expected_sha256=SHA, opener=FakeServer())
    assert not dest.exists() and not part.exists()
    # 다음 호출은 빈 상태에서 새로 받아 성공한다.
    dl.download(URL, dest, expected_sha256=SHA, opener=FakeServer())
    assert dest.read_bytes() == CONTENT


# 확인: 서버가 알린 Content-Length보다 짧게 끊기면 'connection closed' OSError를 내고, 받은 4,000바이트는
# part에 남아 다음 호출이 이어받는다.
def test_truncated_body_keeps_the_part_file_for_resuming(tmp_path):
    dest = tmp_path / "file.bin"
    part = tmp_path / "file.bin.part"

    def cut_off(request):
        return FakeResponse(CONTENT[:4000], 200, {"Content-Length": str(len(CONTENT))})

    with pytest.raises(OSError, match="connection closed"):
        dl.download(URL, dest, expected_sha256=SHA, opener=cut_off)
    assert not dest.exists()
    assert part.read_bytes() == CONTENT[:4000]
    server = FakeServer()
    dl.download(URL, dest, expected_sha256=SHA, opener=server)
    assert server.ranges == ["bytes=4000-"]
    assert dest.read_bytes() == CONTENT


# 확인: VisA의 공식 S3 URL과 SHA-256이 docs/experiments.md 데이터 절에 등록한 값(2eb8690c…f362)과 같다. 값이
# 바뀌면 다른 데이터로 모든 측정을 하게 된다.
def test_visa_constants():
    assert dl.VISA_URL == "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
    assert dl.VISA_SHA256 == "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362"


# 확인: `python -m defect_inspect.download`(main)가 paths.VISA_TAR로 VisA URL과 해시를 넘기고, 크기(천 단위
# 쉼표)와 해시를 출력한다. download를 가짜로 바꿔 실제로 받지 않는다.
def test_main_fetches_visa_to_the_configured_path(tmp_path, monkeypatch, capsys):
    from defect_inspect import paths

    target = tmp_path / "raw" / "VisA_20220922.tar"
    monkeypatch.setattr(paths, "VISA_TAR", target)
    seen = {}

    def fake_download(url, dest, *, expected_sha256=None):
        seen.update(url=url, dest=dest, sha=expected_sha256)
        dest.parent.mkdir(parents=True)
        dest.write_bytes(CONTENT)
        return dest

    monkeypatch.setattr(dl, "download", fake_download)
    dl.main()
    assert seen == {"url": dl.VISA_URL, "dest": target, "sha": dl.VISA_SHA256}
    out = capsys.readouterr().out
    assert "10,240 bytes" in out and dl.VISA_SHA256 in out
