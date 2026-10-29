"""이어 받기가 되는 다운로드와 SHA-256 검증(표준 라이브러리만 씀)."""

# [파일 역할] VisA 원본 tar(1.93GB) 하나를 공식 S3 주소에서 받아 SHA-256으로 검증하는 0단계 다운로드
# 모듈이다(`python -m defect_inspect.download`). 같은 파일의 sha256_file은 다른 모듈들이 ONNX·모델 파일의
# 지문을 실행 기록에 남길 때도 쓴다.
# [설계 이유] 큰 파일을 받다가 끊겨도 처음부터 다시 받지 않도록 `.part` 파일에 이어 받고(HTTP Range), 다 받은
# 뒤 해시가 맞을 때만 최종 이름으로 바꾼다. 그래서 최종 경로에는 검증된 파일만 남는다. 표준 라이브러리만 써서,
# 기본 패키지만 설치한 환경에서도 돈다.
# [흐름] download(): https 주소인지 확인 → 이미 검증된 파일이 있으면 끝 → `.part` 크기만큼 Range 요청으로 이어
# 받기(_fetch) → 해시 검증 → os.replace로 이름 바꾸기.
# [연결] paths.VISA_TAR가 저장 위치다. sha256_file은 dinomaly_serving, run_dinomaly, run_perturb,
# runtime_compare가 가져다 쓴다. tests/test_download.py는 opener 인자에 가짜 서버를 넣어 이어 받기·416·206
# 같은 경우를 확인한다.

import hashlib
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# VisA 공식 S3 주소와 tar의 SHA-256(docs/experiments.md 공통 규칙의 `2eb8690c…f362`). 해시가 다르면 다른
# 파일이므로 분할·결과를 재현할 수 없어 받지 않은 것으로 본다.
VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
VISA_SHA256 = "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362"
# 기본 opener의 소켓 시간 제한(초). 연결과 각 읽기에 적용되어, 전송이 멈추면 무한히 기다리지 않고 예외로
# 끝난다.
TIMEOUT_S = 60

# import 시점의 urlopen(곧 `opener`의 기본값)을 따로 기억해 둔다. download()의 opener가 이 기본값과 같은
# 객체일 때만 시간 제한을 붙인 _urlopen_with_timeout으로 바꾸고, 테스트처럼 다른 opener를 넣으면 그대로 쓴다.
_URLOPEN = urllib.request.urlopen

# Content-Range 헤더(예: `bytes 100-199/1000`)에서 시작 위치와 전체 크기를 뽑는 정규식. 위치나 크기를 모를 때
# 서버가 보내는 '*'도 받아 들인다.
_CONTENT_RANGE = re.compile(r"bytes\s+(\d+|\*)(?:-\d+)?/(\d+|\*)")


# 1MB씩 읽어 수 GB 파일도 메모리에 다 올리지 않는다. 다른 모듈은 이 값을 실행 기록(run.json·리포트)에 남겨
# 어떤 모델 파일로 쟀는지 증명한다(예: 서비스 교체 점검의 ONNX sha256 `3bcf8e04…`).
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """파일을 조각씩 읽어 SHA-256 16진수 문자열을 구한다."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _header(headers, name: str) -> str | None:
    """응답 헤더 하나를 문자열로 읽는다. 헤더 객체가 없는 응답(테스트의 가짜 응답 등)이면 None."""
    if headers is None:
        return None
    value = headers.get(name)
    return None if value is None else str(value)


# _fetch가 이어 받기 응답이 정말 요청한 위치에서 시작하는지, 416 응답에서 서버가 말한 전체 크기가 무엇인지
# 확인하는 데 쓴다.
def _content_range(headers) -> tuple[int | None, int | None]:
    """Content-Range 헤더를 (시작 위치, 전체 크기)로 바꾼다. 없거나 형식이 다르거나 '*'인 칸은 None."""
    match = _CONTENT_RANGE.match(_header(headers, "Content-Range") or "")
    if not match:
        return None, None
    start, total = (None if g == "*" else int(g) for g in match.groups())
    return start, total


# HTTP 상태 코드. `status` 속성이 없는 옛 응답 객체면 getcode()로 읽는다.
def _status(response) -> int:
    status = getattr(response, "status", None)
    return int(status if status is not None else response.getcode())


# 1MB씩 쓴다. mode "ab"는 이어 쓰기, "wb"는 처음부터 쓰기다. 길이가 다르면(중간에 끊김) 예외를 낸다.
def _stream(response, part: Path, mode: str, chunk: int = 1 << 20) -> None:
    """응답 본문을 `.part` 파일에 복사하고, 서버가 알린 Content-Length와 받은 길이가 같은지 확인한다."""
    expected = _header(getattr(response, "headers", None), "Content-Length")
    written = 0
    with open(part, mode) as f:
        while block := response.read(chunk):
            f.write(block)
            written += len(block)
    if expected is not None and expected.isdigit() and written != int(expected):
        # 받은 `.part` 파일은 지우지 않는다: 다음 호출이 거기서부터 이어 받는다.
        raise OSError(f"connection closed after {written} of {expected} bytes: {part}")


# 서버 응답에 따라 경우를 나눈다.
# - `.part`가 없으면: 처음부터 받는다(맨 아래).
# - 있으면 `Range: bytes=<크기>-`로 나머지만 요청한다.
#   - 416(범위가 파일 끝 이후): 서버가 알린 전체 크기가 정확히 지금 크기면 이미 다 받은 것이라 끝낸다. 아니면
#     `.part`를 믿을 수 없어 지우고 처음부터 받는다.
#   - 206(부분 응답)이고 시작 위치가 요청한 위치면 이어 쓴다.
#   - 200(서버가 Range를 무시하고 전체를 보냄)이면 처음부터 덮어쓴다.
#   - 그 밖(다른 위치에서 시작하는 206 등)은 `.part`를 지우고 처음부터 받는다.
def _fetch(url: str, part: Path, opener) -> None:
    """`.part` 파일을 전체 내용까지 채운다. 서버가 허락하면 이어 받는다."""
    offset = part.stat().st_size if part.exists() else 0
    if offset:
        # S310 예외 사유(이 함수의 Request 두 곳 공통): url은 download()가 https인지 확인한 뒤에만 온다.
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"})  # noqa: S310
        try:
            response = opener(request)
        except urllib.error.HTTPError as err:
            if err.code != 416:
                raise
            # 범위가 파일 끝이나 그 뒤에서 시작한다. 서버가 말한 전체 크기가 정확히 이 위치일 때만 `.part`가
            # 완성된 것이다. 크기가 다르거나 모르면(Content-Range 없음) `.part`를 믿을 수 없으므로 버리고
            # 처음부터 받는다.
            _, total = _content_range(err.headers)
            err.close()
            if total == offset:
                return
            response = None
        # with로 응답을 닫는 것을 보장하고, 상태 코드와 Content-Range를 보고 이어 쓸지, 덮어쓸지, 버릴지
        # 정한다.
        if response is not None:
            with response:
                start, _ = _content_range(getattr(response, "headers", None))
                if _status(response) == 206 and start in (None, offset):
                    _stream(response, part, "ab")
                    return
                if _status(response) == 200:
                    # 서버가 Range 헤더를 무시하고 전체를 다시 보내고 있다.
                    _stream(response, part, "wb")
                    return
        # 위 어느 경우에도 해당하지 않으면 이어 받기를 포기하고 `.part`를 지운 뒤 아래에서 처음부터 받는다.
        part.unlink()
    with opener(urllib.request.Request(url)) as response:  # noqa: S310
        _stream(response, part, "wb")


# 기본 opener: urlopen에 TIMEOUT_S 시간 제한을 붙인 것. 함수 안에서 urllib.request.urlopen을 매번 찾아
# 부르므로 테스트가 monkeypatch로 바꿔 끼운 urlopen도 그대로 쓰인다.
def _urlopen_with_timeout(request):
    # 시간 제한은 연결과 모든 대기 읽기에 걸리므로, 멈춘 전송은 무한히 기다리지 않고 예외를 낸다(`.part`
    # 파일은 다음 호출을 위해 남는다). S310 예외 사유: request는 _fetch가 download()에서 https로 확인한
    # 주소로만 만든다.
    return urllib.request.urlopen(request, timeout=TIMEOUT_S)  # noqa: S310


# expected_sha256을 주면 검증까지 하고, 대소문자는 소문자로 맞춰 비교한다.
# - dest가 이미 있고 해시가 맞으면(또는 해시를 주지 않았으면) 요청 없이 바로 돌려준다. 해시가 틀린 dest는 다시
#   받는다.
# - 받은 `.part`의 해시가 틀리면 지우고 ValueError를 낸다. 손상된 파일로 이어 받기를 반복하지 않게 하려는
#   것이다.
# - 맞으면 os.replace로 한 번에 이름을 바꾼다. 이름 바꾸기는 원자적이라 중간에 멈춰도 dest에 덜 받은 파일이
#   남지 않는다.
# 호출 위치: 이 모듈의 main(VisA tar).
def download(
    url: str, dest: Path, *, expected_sha256: str | None = None, opener=urllib.request.urlopen
) -> Path:
    """`url`을 `dest` + ".part"를 거쳐 `dest`로 받는다. 덜 받은 파일은 이어 받고, 해시를 검증한다.

    기본 opener는 `TIMEOUT_S`초 시간 제한을 붙인 `urllib.request.urlopen`이다. https 주소만 받고, 다른
    scheme이면 요청 전에 `ValueError`를 낸다.
    """
    # https만 받는다: urlopen은 file:·ftp: 같은 다른 scheme도 열기 때문에, 잘못된 주소가 들어오면 로컬
    # 파일을 dest로 복사해 오거나 암호화되지 않은 연결로 받을 수 있다(해시를 주지 않으면 그대로 성공한다).
    # 이 저장소가 받는 주소(VISA_URL)는 https다. urlsplit은 scheme을 소문자로 돌려주므로 "HTTPS://"도 받는다.
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError(f"only https URLs are downloaded, got {url!r}")
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


# 명령행 입구: VisA tar를 paths.VISA_TAR(data/raw 아래)에 받고 검증한다. paths는 함수 안에서 import한다.
def main() -> None:
    from defect_inspect import paths

    path = download(VISA_URL, paths.VISA_TAR, expected_sha256=VISA_SHA256)
    # download()가 방금 해시를 검증했으므로 여기서 다시 계산하지 않는다.
    print(f"{path}  {path.stat().st_size:,} bytes  sha256 {VISA_SHA256} (verified)")


if __name__ == "__main__":
    main()
