"""Resumable download with SHA-256 verification (standard library only)."""

import hashlib
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
VISA_SHA256 = "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362"
TIMEOUT_S = 60  # socket timeout of the default opener

_URLOPEN = urllib.request.urlopen  # the default value of `opener`, as it was at import time

_CONTENT_RANGE = re.compile(r"bytes\s+(\d+|\*)(?:-\d+)?/(\d+|\*)")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _header(headers, name: str) -> str | None:
    """Read one response header; tolerate responses that have no headers object."""
    if headers is None:
        return None
    value = headers.get(name)
    return None if value is None else str(value)


def _content_range(headers) -> tuple[int | None, int | None]:
    """(start, total) from a Content-Range header; None for anything missing or '*'."""
    match = _CONTENT_RANGE.match(_header(headers, "Content-Range") or "")
    if not match:
        return None, None
    start, total = (None if g == "*" else int(g) for g in match.groups())
    return start, total


def _status(response) -> int:
    status = getattr(response, "status", None)
    return int(status if status is not None else response.getcode())


def _stream(response, part: Path, mode: str, chunk: int = 1 << 20) -> None:
    """Copy the response body to the part file and check the length the server announced."""
    expected = _header(getattr(response, "headers", None), "Content-Length")
    written = 0
    with open(part, mode) as f:
        while block := response.read(chunk):
            f.write(block)
            written += len(block)
    if expected is not None and expected.isdigit() and written != int(expected):
        # Keep the part file: the next call resumes from it.
        raise OSError(f"connection closed after {written} of {expected} bytes: {part}")


def _fetch(url: str, part: Path, opener) -> None:
    """Bring the part file up to the full content, resuming when the server allows it."""
    offset = part.stat().st_size if part.exists() else 0
    if offset:
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"})
        try:
            response = opener(request)
        except urllib.error.HTTPError as err:
            if err.code != 416:
                raise
            # The range starts at or after the end. The part file is complete only when the server says
            # that the total size is exactly this offset; a different or unknown total (no
            # Content-Range) means the part file cannot be trusted: drop it and start again.
            _, total = _content_range(err.headers)
            err.close()
            if total == offset:
                return
            response = None
        if response is not None:
            with response:
                start, _ = _content_range(getattr(response, "headers", None))
                if _status(response) == 206 and start in (None, offset):
                    _stream(response, part, "ab")
                    return
                if _status(response) == 200:
                    # The server ignored the Range header and is sending everything again.
                    _stream(response, part, "wb")
                    return
        part.unlink()
    with opener(urllib.request.Request(url)) as response:
        _stream(response, part, "wb")


def _urlopen_with_timeout(request):
    # The timeout covers connecting and every blocking read, so a stalled transfer raises (keeping
    # the part file for the next call) instead of hanging forever.
    return urllib.request.urlopen(request, timeout=TIMEOUT_S)


def download(
    url: str, dest: Path, *, expected_sha256: str | None = None, opener=urllib.request.urlopen
) -> Path:
    """Download `url` to `dest` through `dest` + ".part"; resume a partial file; verify the hash.

    The default opener is `urllib.request.urlopen` with a timeout of `TIMEOUT_S` seconds.
    """
    if opener is _URLOPEN:
        opener = _urlopen_with_timeout
    dest = Path(dest)
    expected = expected_sha256.lower() if expected_sha256 else None
    if dest.exists() and (expected is None or sha256_file(dest) == expected):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    _fetch(url, part, opener)
    if expected is not None:
        actual = sha256_file(part)
        if actual != expected:
            part.unlink()
            raise ValueError(f"sha256 mismatch for {url}: expected {expected}, got {actual}")
    os.replace(part, dest)
    return dest


def main() -> None:
    from defect_inspect import paths

    path = download(VISA_URL, paths.VISA_TAR, expected_sha256=VISA_SHA256)
    # download() has just verified the hash, so it is not computed a second time here.
    print(f"{path}  {path.stat().st_size:,} bytes  sha256 {VISA_SHA256} (verified)")


if __name__ == "__main__":
    main()
