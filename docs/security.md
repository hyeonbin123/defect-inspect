# 보안 점검 기록

검사 서비스(`src/defect_inspect/service.py`)를 동적 스캔(DAST)으로 점검한 기록이다. 스캔은 이 PC에서 띄운 로컬 서비스(`127.0.0.1:8093`)에만 한다.

## 설정
| 도구 | 설정 파일 | 비고 |
|---|---|---|
| HawkScan (StackHawk) | `stackhawk.yml` | 체험 기간이 2026-10-05에 끝난다 |
| OWASP ZAP 2.17.0 | `zap/automation.yaml`, `zap/run_zap.sh` | Apache-2.0. 공식 크로스플랫폼 패키지를 Java 21로 직접 실행(Docker 없음) |

두 설정은 같은 일을 한다.
- 서비스의 OpenAPI 명세(`/openapi.json`)에서 경로를 가져온다
- OpenAPI로는 이미지 업로드를 만들 수 없어 모든 POST가 폼 검증(422)에서 멈추므로, 실제 multipart 요청을 HAR 시드로 넣는다(`hawk/make_seed_har.py`). 합성 범주 `demo`로 도는 오프라인 서비스는 `hawk/seed.har`, 아티팩트로 띄운 서비스는 그 범주로 만든 시드(예: `hawk/seed-pcb1.har`)를 쓴다
- spider, passive scan, 기본 정책 active scan
- 모든 요청에 일회용 관리자 토큰(`X-Admin-Token`)을 붙여 `POST /calibrate`와 `DELETE /calibrate`까지 닿게 한다. 토큰은 실행할 때 환경 변수로 주고 파일에 적지 않는다

HawkScan은 ZAP 엔진 위에 만든 상용 도구지만 규칙 목록, 기본 설정, 버전이 ZAP 배포판과 같지 않다. 그래서 **두 도구의 결과는 일대일로 비교하지 않고, 같은 도구의 기록끼리만 비교한다.**

실행 (Git Bash, 스캐너 출력은 파일로만 보낸다):
```bash
ADMIN_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(24))")
# 오프라인 서비스 (모델 파일 없음, 범주 demo)
DEFECT_INSPECT_ADMIN_TOKEN=$ADMIN_TOKEN uv run uvicorn defect_inspect.service:create_offline_app --factory --port 8093
# 아티팩트로 띄운 서비스라면 DEFECT_INSPECT_ARTIFACTS=<아티팩트 폴더> ... service:create_app 이고, 시드는 그 범주의 것

# HawkScan (아티팩트 서비스는 SEED_HAR=hawk/seed-pcb1.har 를 더한다)
APP_ID=<application id> ADMIN_TOKEN=$ADMIN_TOKEN hawk scan --hawk-mem=4g stackhawk.yml > work/hawk.log 2>&1
# OWASP ZAP (ZAP_DIR: zap-2.17.0.jar 가 있는 폴더. 결과는 work/zap/<이름>/)
ZAP_DIR=<폴더> ADMIN_TOKEN=$ADMIN_TOKEN zap/run_zap.sh <이름> [시드 HAR]
```

## 기록

### 2026-10-01 HawkScan (서비스 모델 PatchCore, 오프라인 서비스)
- 1회차(scan `aa03b755`): OpenAPI만으로 스캔했다. `POST /inspect` 399회와 `POST /calibrate` 604회가 모두 폼 검증(422)에서 멈춰 처리 코드에 닿지 않았다. 그래서 HAR 시드를 만들었다
- 2회차(scan `b821d222`, HAR 시드 추가): 지적은 Anti-CSRF Tokens Check(Medium) 1건이다. 데모 페이지(`GET /`)의 폼에 anti-CSRF 토큰이 없다는 것이다. 오탐으로 표시했다. 폼은 `fetch()`로만 보내고, 서비스는 쿠키를 쓰지 않는다. 쓰기 경로(`/calibrate`)는 다른 출처의 페이지가 붙일 수 없는 `X-Admin-Token` 헤더가 필요하고(CORS 없음), 다른 출처의 쓰기 요청은 Sec-Fetch-Site/Origin으로 거절한다(`tests/test_service.py`)

### 2026-10-04 서비스 모델 교체 전후 (규칙, 스캔 전에 커밋)
2026-10-04 사용자가 서비스 모델을 6단계의 dms-280-car로 바꾸기로 정했다. 교체는 `service.py`가 복원 방식(Dinomaly) 아티팩트도 읽게 하는 것이다. 바뀌는 것은 범주마다 읽는 아티팩트(메모리 뱅크 → 모델이 직접 점수를 냄), `/categories`의 응답 필드, `/calibrate`가 점수를 내는 방법이다. 경로와 보호 장치(Host 검사, 다른 출처의 쓰기 요청 거절, 관리자 토큰, 본문 크기 제한, 보안 헤더)는 바꾸지 않는다. HawkScan 체험이 끝나는 시점이라 두 도구로 교체 전후를 스캔한다.

- 대상 (1) 교체 전: 이 규칙 커밋의 오프라인 서비스(합성 PatchCore 범주 `demo`, 시드 `hawk/seed.har`). 서비스 코드는 2026-10-01 스캔 때와 같다
- 대상 (2) 교체 후: 교체 커밋의 서비스를 `artifacts/dms-280-car`(12개 범주, onnxruntime CPU FP32)로 띄운 것, 시드 `hawk/seed-pcb1.har`
- 같은 날 (1) → 교체 커밋 → (2) 순서로, 대상마다 HawkScan과 ZAP을 한 번씩 돌린다. 둘 다 `127.0.0.1:8093`, 대상마다 새 일회용 토큰
- **통과 조건**: 도구마다 (2)에 (1)에 없던 High 또는 Medium 경보(규칙 이름 기준)가 없다. 새로 나온 것은 고치거나, 고치지 않는 사유를 이 문서와 README에 적는다. Low와 Informational은 적기만 한다
- ZAP (1)에서 HawkScan이 오탐으로 표시한 Anti-CSRF(Medium)가 다시 나오는지 적는다(판정 없음)
- ZAP active scan은 120분에서 끊는다. 끊기면 그 사실을 적는다
- 결과 요약은 `reports/dast/`에 JSON으로 남긴다(경보 이름, 위험도, 신뢰도, 건수, 경로. 토큰과 요청 본문은 넣지 않는다)
- 2026-10-05 이후 다시 스캔할 때는 ZAP을 쓰고, 이 날의 ZAP 기록과 비교한다

### 2026-10-04 결과 (도구가 바뀌었다: HawkScan → OWASP ZAP. 두 도구의 결과는 직접 비교하지 않는다)
규칙 커밋(`55c6149`) 뒤 같은 저녁에 (1) → 교체 커밋(`7d590d2`) → (2) 순서로 돌렸다. 요약 `reports/dast/2026-10-04.json`(경보, 서비스가 스캔 중에 낸 응답 코드 집계).

| 도구 | 대상 | High | Medium | Low | Informational | 스캔 중 요청 (처리 코드에 닿은 업로드) |
|---|---|---|---|---|---|---|
| HawkScan v6.4.0 | (1) 교체 전, 오프라인 (`9b683118`) | 0 | 1 (Anti-CSRF Tokens Check) | 0 | 0 | 3,186 (`POST /inspect` 200 13회, `POST /calibrate` 200 109회) |
| HawkScan v6.4.0 | (2) 교체 후, dms-280-car (`81b66cb7`) | 0 | 1 (Anti-CSRF Tokens Check) | 0 | 0 | 3,184 (13회, 109회) |
| OWASP ZAP 2.17.0 | (1) 교체 전, 오프라인 | 0 | 0 | 0 | 2 (Modern Web Application, User Agent Fuzzer) | 6,891 (517회, 1,320회) |
| OWASP ZAP 2.17.0 | (2) 교체 후, dms-280-car | 0 | 0 | 0 | 2 (같은 둘) | 6,894 (517회, 1,326회) |

- **판정: 통과.** 두 도구 모두 (2)에 (1)에 없던 High·Medium이 없다. 경보의 종류와 건수가 교체 전후에 같고, 네 스캔 모두 서비스가 5xx를 한 번도 내지 않았다(모델이 실제로 도는 (2)에서도 대기열이 넘친 503이 없었다)
- HawkScan의 Medium은 2026-10-01에 오탐으로 표시한 그 지적(데모 페이지 폼의 anti-CSRF 토큰)이고, 두 스캔 모두 오탐 상태로 남아 있다. 사유는 위 2026-10-01 기록과 같다
- **ZAP은 이 Anti-CSRF Medium을 다시 내지 않았다.** HawkScan의 "Anti-CSRF Tokens Check"와 같은 이름의 active 규칙은 ZAP 공식 배포판에 든 규칙 묶음(`ascanrules` release)에 없고, ZAP의 패시브 규칙 "Absence of Anti-CSRF Tokens"(10202)는 이 폼에 경보를 내지 않았다. 같은 폼을 보고 도구에 따라 결과가 다르다는 것이 "직접 비교하지 않는다"의 한 예다
- ZAP의 Informational 두 건은 데모 페이지가 링크 없이 스크립트로 그려지는 페이지라는 표시(Modern Web Application, 10109)와, User-Agent를 바꿔 보낸 `/calibrate` 요청의 응답이 원래 응답과 달랐다는 기록(User Agent Fuzzer, 10104)이다. `/calibrate`는 부를 때마다 임계값 상태가 바뀌어 응답 본문(`previous_threshold` 등)이 달라지는 경로라서, User-Agent에 따라 처리가 갈리는 것이 아니다. 둘 다 고칠 것이 아니다
- ZAP의 한계: DOM XSS 규칙은 이 PC에 Firefox가 없어 브라우저를 띄우지 못하고 건너뛰었고(두 대상 모두), Parameter Tamper 규칙은 multipart 매개변수를 바꾸다 ZAP 내부 오류(NullPointerException)로 16번 멈췄다(두 대상 모두). 그 밖의 기본 정책 규칙은 돌았다
- **2026-10-04 정정 (검토 지적)**: 바로 위 "그 밖의 기본 정책 규칙은 돌았다"는 틀렸다. ZAP 로그에 규칙마다 남는 완료·건너뜀 줄을 다시 세면, 두 대상 모두 active 규칙 53개 가운데 3개가 돌지 않았다. DOM XSS(위 사유), **Log4Shell**(CVE-2021-44228·45046. 응답이 아니라 외부 콜백으로 확인하는 규칙이라 OAST 콜백 서비스가 필요한데, 설정하지 않아 건너뛰었다), 스크립트 규칙(켠 스크립트가 없어 해당 없음)이다. 나머지 50개는 완료로 기록됐고, 그중 9개(XXE, 지수 엔터티 확장, SOAP 두 개, Padding Oracle, HTTP Only Site, HTTPS as HTTP, GET for POST, Persistent XSS)는 대상에 맞는 요청이 없어 한 건도 보내지 않았다. 서비스는 Python이라 Log4j를 쓰지 않지만, 이 스캔이 Log4Shell을 확인한 것은 아니다. 판정(교체 뒤 새 High·Medium 없음)은 바뀌지 않는다. 다음 ZAP 스캔을 이 기록과 비교할 때 이 세 규칙은 돌지 않은 것으로 본다
- 스캔 시간: HawkScan (1) 약 30초, (2) 약 3분. ZAP (1) 약 1분(active scan 28초), (2) 약 18분(active scan 17분 30초, 120분 한도 안). (2)는 업로드마다 모델이 CPU에서 돌아서 길다
- ZAP 첫 실행은 실행 스크립트가 시드 HAR 경로를 상대 경로로 넘겨 가져오기 단계에서 멈췄다(스캔 결과 없음). 스크립트를 고쳐(`86c209e`) 같은 서비스 프로세스에 다시 돌린 것이 위 (1)이다
