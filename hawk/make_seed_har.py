"""Write hawk/seed.har: real multipart requests for the scan of the offline service (see stackhawk.yml).

The OpenAPI request builder cannot render a valid image upload, so without these seeds every POST /inspect
and POST /calibrate of a scan stops at form validation (422) and never reaches the handler. HAR bodies are
text, so the images are 24-bit BMPs whose bytes are all below 0x80 (a PNG or JPEG would not survive the
round trip through a JSON string). No token in here: the scan adds X-Admin-Token to every request.

    uv run python hawk/make_seed_har.py [--host http://127.0.0.1:8093]
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np

BOUNDARY = "hawkseedboundary7d3a"
OUT = Path(__file__).with_name("seed.har")


def ascii_bmp(seed: int, size: int = 32) -> bytes:
    """A size x size 24-bit BMP of mid grey noise; every byte of the file is below 0x80."""
    pixels = np.random.default_rng(seed).integers(100, 128, (size, size, 3), dtype=np.uint8)
    row = size * 3
    assert row % 4 == 0, "rows must need no padding"
    body = pixels[::-1].tobytes()  # BMP rows run bottom-up
    header = b"BM" + struct.pack("<IHHI", 54 + len(body), 0, 0, 54)
    info = struct.pack("<IiiHHIIiiII", 40, size, size, 1, 24, 0, len(body), 0, 0, 0, 0)
    data = header + info + body
    assert max(data) < 0x80, "the BMP must stay within 7-bit bytes"
    return data


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


def build(host: str) -> dict:
    inspect = multipart([("category", "demo")], [("image", "part.bmp", ascii_bmp(0))])
    calibrate = multipart(
        [("category", "demo"), ("alpha", "0.05"), ("allow_unguaranteed", "true")],
        [("images", f"normal-{i}.bmp", ascii_bmp(10 + i)) for i in range(3)],
    )
    entries = [
        entry("POST", f"{host}/inspect?preview=false&heatmap=true", inspect),
        entry("POST", f"{host}/calibrate", calibrate),
        entry("DELETE", f"{host}/calibrate?category=demo"),
    ]
    creator = {"name": "make_seed_har.py", "version": "1"}
    return {"log": {"version": "1.2", "creator": creator, "entries": entries}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="http://127.0.0.1:8093")
    args = parser.parse_args()
    OUT.write_text(json.dumps(build(args.host.rstrip("/")), indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
