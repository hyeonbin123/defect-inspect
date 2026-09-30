import hashlib
import io
import urllib.error

import pytest

from defect_inspect import download as dl

URL = "https://example.invalid/file.bin"
CONTENT = bytes(range(256)) * 40  # 10,240 bytes
SHA = hashlib.sha256(CONTENT).hexdigest()


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, status: int, headers: dict[str, str]):
        super().__init__(body)
        self.status = status
        self.headers = headers


class FakeServer:
    """Stands in for urlopen: serves CONTENT, honouring Range unless told to ignore it."""

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
        self.total_in_416 = total_in_416  # False: a 416 answer carries no Content-Range
        self.start_shift = start_shift  # non-zero: a 206 answer starts somewhere else than asked
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


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(CONTENT)
    assert dl.sha256_file(path) == SHA
    assert dl.sha256_file(path, chunk=7) == SHA


def test_fresh_download_verifies_and_renames(tmp_path):
    dest = tmp_path / "sub" / "file.bin"
    server = FakeServer()
    assert dl.download(URL, dest, expected_sha256=SHA, opener=server) == dest
    assert dest.read_bytes() == CONTENT
    assert not dest.with_name("file.bin.part").exists()
    assert server.ranges == [None]


def test_existing_file_is_returned_without_a_request(tmp_path):
    dest = tmp_path / "file.bin"
    dest.write_bytes(CONTENT)
    server = FakeServer()
    assert dl.download(URL, dest, expected_sha256=SHA.upper(), opener=server) == dest
    assert dl.download(URL, dest, opener=server) == dest
    assert server.ranges == []


def test_existing_file_with_wrong_hash_is_replaced(tmp_path):
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"stale")
    dl.download(URL, dest, expected_sha256=SHA, opener=FakeServer())
    assert dest.read_bytes() == CONTENT


def test_resume_sends_range_and_appends(tmp_path):
    dest = tmp_path / "file.bin"
    part = tmp_path / "file.bin.part"
    part.write_bytes(CONTENT[:3000])
    server = FakeServer()
    dl.download(URL, dest, expected_sha256=SHA, opener=server)
    assert server.ranges == ["bytes=3000-"]
    assert dest.read_bytes() == CONTENT
    assert not part.exists()


def test_server_ignoring_range_restarts_from_zero(tmp_path):
    dest = tmp_path / "file.bin"
    part = tmp_path / "file.bin.part"
    part.write_bytes(b"x" * 3000)  # not even a prefix of the content
    server = FakeServer(honour_range=False)
    dl.download(URL, dest, expected_sha256=SHA, opener=server)
    assert server.ranges == ["bytes=3000-"]
    assert dest.read_bytes() == CONTENT


def test_complete_part_file_is_verified_without_downloading_again(tmp_path):
    dest = tmp_path / "file.bin"
    part = tmp_path / "file.bin.part"
    part.write_bytes(CONTENT)
    server = FakeServer()
    dl.download(URL, dest, expected_sha256=SHA, opener=server)
    assert server.ranges == [f"bytes={len(CONTENT)}-"]
    assert dest.read_bytes() == CONTENT


def test_oversized_part_file_is_discarded(tmp_path):
    dest = tmp_path / "file.bin"
    part = tmp_path / "file.bin.part"
    part.write_bytes(CONTENT + b"junk")
    server = FakeServer()
    dl.download(URL, dest, expected_sha256=SHA, opener=server)
    assert server.ranges == [f"bytes={len(CONTENT) + 4}-", None]
    assert dest.read_bytes() == CONTENT


@pytest.mark.parametrize("sha", [None, SHA], ids=["no-hash", "hash"])
@pytest.mark.parametrize("kind", ["oversized", "complete", "garbage"])
def test_416_without_a_total_discards_the_part_and_restarts(tmp_path, sha, kind):
    # Without Content-Range nothing says that the part file is the whole file: it may be longer.
    part_content = {"oversized": CONTENT + b"junk", "complete": CONTENT, "garbage": b"x" * 14000}[kind]
    dest = tmp_path / "file.bin"
    part = tmp_path / "file.bin.part"
    part.write_bytes(part_content)
    server = FakeServer(total_in_416=False)
    dl.download(URL, dest, expected_sha256=sha, opener=server)
    assert server.ranges == [f"bytes={len(part_content)}-", None]
    assert dest.read_bytes() == CONTENT
    assert not part.exists()


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


def test_stalled_connection_keeps_the_part_file_for_resuming(tmp_path):
    dest = tmp_path / "file.bin"
    part = tmp_path / "file.bin.part"

    class Stalling(FakeResponse):
        def read(self, size=-1):
            if self.tell() >= 4096:
                raise TimeoutError("timed out")  # what a socket timeout raises in the middle of a body
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


def test_hash_mismatch_raises_and_removes_part(tmp_path):
    dest = tmp_path / "file.bin"
    with pytest.raises(ValueError, match="sha256 mismatch"):
        dl.download(URL, dest, expected_sha256="0" * 64, opener=FakeServer())
    assert not dest.exists()
    assert not dest.with_name("file.bin.part").exists()


def test_corrupt_part_file_fails_the_hash_check(tmp_path):
    dest = tmp_path / "file.bin"
    part = tmp_path / "file.bin.part"
    part.write_bytes(b"x" * 3000)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        dl.download(URL, dest, expected_sha256=SHA, opener=FakeServer())
    assert not dest.exists() and not part.exists()
    # The next call starts clean and succeeds.
    dl.download(URL, dest, expected_sha256=SHA, opener=FakeServer())
    assert dest.read_bytes() == CONTENT


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


def test_visa_constants():
    assert dl.VISA_URL == "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
    assert dl.VISA_SHA256 == "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362"


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
