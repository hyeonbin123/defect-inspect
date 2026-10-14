"""검사 서비스의 DAST 스캔(stackhawk.yml, zap/)에 넣을 HAR 시드, 곧 실제 multipart 요청을 만든다.

OpenAPI 명세로 요청을 만드는 스캐너는 올바른 이미지 업로드를 만들지 못한다. 그래서 이 시드가 없으면 스캔의
POST /inspect와 POST /calibrate가 모두 폼 검증(422)에서 멈추고 처리 코드에 닿지 않는다. HAR의 요청 본문은
텍스트(JSON 문자열)라서 이미지는 모든 바이트가 0x80 미만인 24비트 BMP로 만든다(PNG나 JPEG는 JSON 문자열을
왕복하면서 깨진다). 관리자 토큰은 여기에 넣지 않는다. 스캔이 모든 요청에 X-Admin-Token 헤더를 붙인다.

    uv run python hawk/make_seed_har.py                      # hawk/seed.har: demo 범주(오프라인 앱)
    uv run python hawk/make_seed_har.py --category pcb1 --out hawk/seed-pcb1.har   # 아티팩트 세트 하나
"""

# [파일 역할] 검사 서비스의 DAST(동적 보안 스캔)에 넣을 HAR 시드 파일을 만드는 스크립트다. 실제 이미지가 든
# multipart 요청 세 개(POST /inspect, POST /calibrate, DELETE /calibrate)를 HAR 형식 JSON으로 쓴다.
# [왜 필요한가] 2026-10-01 첫 HawkScan에서 실제로 업로드 요청이 전부 폼 검증(422)에서 멈췄다
# (docs/security.md). 진짜 업로드 요청을 시드로 주면 스캐너가 이것을 바꿔 가며 처리 코드까지 공격해 본다.
# [설계 이유] 서비스는 BMP도 받으므로 7비트 BMP로 충분하다.
# [연결] stackhawk.yml(HawkScan)과 zap/automation.yaml·zap/run_zap.sh(OWASP ZAP)가 이 파일이 만든
# hawk/seed.har(오프라인 앱의 demo 범주)나 hawk/seed-pcb1.har(아티팩트 서비스의 pcb1 범주)를 읽는다.

import argparse
import json
import struct
from pathlib import Path

# numpy는 시드를 받은 난수로 BMP 픽셀을 결정적으로 만드는 데만 쓴다. 같은 명령이면 같은 시드 파일이 나온다.
import numpy as np

# multipart 본문의 구분 문자열(본문에 나오지 않을 임의 문자열)과 기본 출력 경로(이 스크립트 옆 hawk/seed.har).
BOUNDARY = "hawkseedboundary7d3a"
OUT = Path(__file__).with_name("seed.har")


# 픽셀은 100~127의 회색 잡음이고, 헤더의 크기 값도 32px에서는 작아 0x80 미만 조건을 지킨다. 두 assert로 그
# 조건(행 패딩 없음, 7비트)이 깨지면 바로 멈춘다.
def ascii_bmp(seed: int, size: int = 32) -> bytes:
    """size×size 크기의 중간 회색 잡음 24비트 BMP 바이트를 만든다. 파일의 모든 바이트가 0x80 미만이다."""
    pixels = np.random.default_rng(seed).integers(100, 128, (size, size, 3), dtype=np.uint8)
    # BMP는 각 행을 4바이트 배수로 채워야 한다. 32px × 3바이트 = 96이라 패딩이 필요 없고, 패딩을 쓰지 않는
    # 간단한 구현을 assert로 보장한다.
    row = size * 3
    assert row % 4 == 0, "rows must need no padding"
    body = pixels[::-1].tobytes()  # BMP는 행을 아래에서 위 순서로 저장한다
    # 14바이트 파일 헤더("BM", 파일 크기, 픽셀 시작 위치 54)와 40바이트 BITMAPINFOHEADER(너비, 높이, 평면 1,
    # 24비트, 무압축, 본문 크기)를 리틀엔디언으로 묶는다.
    header = b"BM" + struct.pack("<IHHI", 54 + len(body), 0, 0, 54)
    info = struct.pack("<IiiHHIIiiII", 40, size, size, 1, 24, 0, len(body), 0, 0, 0, 0)
    data = header + info + body
    assert max(data) < 0x80, "the BMP must stay within 7-bit bytes"
    return data


# 폼 필드(이름, 값)와 파일(필드 이름, 파일 이름, 바이트)로 multipart/form-data 본문 문자열을 만든다. BMP
# 바이트가 7비트라 ascii로 그대로 문자열에 넣을 수 있다.
def multipart(fields: list[tuple[str, str]], files: list[tuple[str, str, bytes]]) -> str:
    parts = []
    for name, value in fields:
        parts.append(f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n')
    for name, filename, data in files:
        parts.append(
            f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: image/bmp\r\n\r\n" + data.decode("ascii") + "\r\n"
        )
    return "".join(parts) + f"--{BOUNDARY}--\r\n"


# HAR 1.2의 요청·응답 항목 하나를 만든다. 시드는 요청만 의미가 있으므로 응답은 형식을 채우는 빈 200 응답이고,
# 크기 값을 모르는 칸은 HAR 규약대로 -1이다. 본문이 있으면 multipart Content-Type 헤더와 postData를 넣는다.
def entry(method: str, url: str, body: str | None = None) -> dict:
    headers = []
    post = None
    if body is not None:
        mime = f"multipart/form-data; boundary={BOUNDARY}"
        headers.append({"name": "Content-Type", "value": mime})
        post = {"mimeType": mime, "text": body}
    request = {
        "method": method,
        "url": url,
        "httpVersion": "HTTP/1.1",
        "headers": headers,
        "queryString": [],
        "cookies": [],
        "headersSize": -1,
        "bodySize": -1 if body is None else len(body),
    }
    if post is not None:
        request["postData"] = post
    response = {
        "status": 200,
        "statusText": "OK",
        "httpVersion": "HTTP/1.1",
        "headers": [],
        "cookies": [],
        "content": {"size": 0, "mimeType": "application/json"},
        "redirectURL": "",
        "headersSize": -1,
        "bodySize": -1,
    }
    timings = {"send": 0, "wait": 0, "receive": 0}
    return {"request": request, "response": response, "cache": {}, "timings": timings}


# 시드 HAR 전체를 만든다.
# - POST /inspect: 이미지 한 장, preview=false, heatmap=true(히트맵 인코딩까지 처리 코드를 지나게 함).
# - POST /calibrate: 정상 사진 흉내 3장. 3장으로는 α = 5%의 보장 조건(19장 이상)을 채우지 못해 서비스가
#   400으로 거절하므로 allow_unguaranteed=true를 함께 보낸다.
# - DELETE /calibrate: 재보정을 되돌리는 경로.
# category는 서비스에 있는 범주여야 한다. 오프라인 앱은 demo, 아티팩트 서비스는 pcb1 같은 실제 범주다.
def build(host: str, category: str = "demo") -> dict:
    inspect = multipart([("category", category)], [("image", "part.bmp", ascii_bmp(0))])
    calibrate = multipart(
        [("category", category), ("alpha", "0.05"), ("allow_unguaranteed", "true")],
        [("images", f"normal-{i}.bmp", ascii_bmp(10 + i)) for i in range(3)],
    )
    entries = [
        entry("POST", f"{host}/inspect?preview=false&heatmap=true", inspect),
        entry("POST", f"{host}/calibrate", calibrate),
        entry("DELETE", f"{host}/calibrate?category={category}"),
    ]
    creator = {"name": "make_seed_har.py", "version": "1"}
    return {"log": {"version": "1.2", "creator": creator, "entries": entries}}


# 명령행 입구. --host(기본 127.0.0.1:8093, 스캔하는 로컬 서비스), --category, --out을 받아 HAR를 UTF-8·LF로
# 쓴다.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="http://127.0.0.1:8093")
    parser.add_argument("--category", default="demo", help="demo는 오프라인 앱, 그 밖의 값은 범주 폴더 이름")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    har = build(args.host.rstrip("/"), args.category)
    args.out.write_text(json.dumps(har, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
