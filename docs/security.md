# 보안 점검 기록

검사 서비스(`src/defect_inspect/service.py`)를 동적 스캔(DAST)으로 점검한 기록이다. 스캔은 이 PC에서 띄운 로컬 서비스(`127.0.0.1:8093`)에만 한다.
서비스 이미지에 들어가는 의존성을 공개 취약점 데이터베이스와 대조한 기록은 DAST와 따로 맨 아래 [의존성 취약점 대조](#의존성-취약점-대조) 절에 적는다(2026-10-24부터).
compose 기본 설정으로 띄운 컨테이너의 자원 한도(업로드 임시 저장소, 메모리)를 실제로 채워 확인한 기록은 그 앞의 [운영 확인](#운영-확인) 절에 적는다(2026-11-05부터).

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

### 2026-10-15 모델 지문 대조를 넣은 뒤 재스캔 (규칙, 스캔 전에 커밋)
서비스가 시작할 때 meta.json에 적힌 모델 파일 SHA-256(`source.onnx_sha256`, 동적 INT8은 `source.onnx_int8_dynamic_sha256`)을 실제 모델 파일과 대조하고, 다르면 시작을 멈추게 했다(`inspector.check_model_sha256`, `service.load_inspectors`). 바뀐 것은 시작 경로뿐이고 요청 경로와 보호 장치(Host 검사, 다른 출처의 쓰기 요청 거절, 관리자 토큰, 본문 크기 제한, 보안 헤더)는 그대로다. 같은 커밋에서 CI에 의존성 취약점 대조(pip-audit)와 Docker 이미지 빌드·스모크 검사를 더했다.

- 도구는 OWASP ZAP 2.17.0(`zap/automation.yaml`, `zap/run_zap.sh`), 설정은 2026-10-04와 같다. HawkScan은 체험 기간이 끝나 쓰지 않는다
- 대상 두 개를 2026-10-04의 ZAP 기록과 같은 조건으로 스캔한다: (1) 오프라인 서비스(합성 범주 `demo`, 시드 `hawk/seed.har`), (2) `artifacts/dms-280-car`로 띄운 서비스(시드 `hawk/seed-pcb1.har`). 둘 다 `127.0.0.1:8093`, 대상마다 새 일회용 토큰. 서비스 코드는 이 규칙을 넣은 커밋의 것이다
- **통과 조건**: 대상마다 2026-10-04 ZAP 기록에 없던 High 또는 Medium 경보(규칙 이름 기준)가 없다. 새로 나온 것은 고치거나, 고치지 않는 사유를 이 문서에 적는다. Low와 Informational은 적기만 한다. 2026-10-04에 돌지 않은 active 규칙 세 개(DOM XSS, Log4Shell, 스크립트 규칙)는 이번에도 돌지 않을 것으로 보고, 돌았는지 다시 센다
- 서비스가 스캔 중에 5xx를 낸 횟수를 적는다. active scan은 120분에서 끊는다
- 요약은 `reports/dast/2026-10-15.json`(경보 이름, 규칙 번호, 위험도, 신뢰도, 건수, 경로 표본, 응답 코드 집계. 토큰과 요청 본문은 넣지 않는다)

### 2026-10-15 결과 (OWASP ZAP 2.17.0, 서비스 코드 `b17ce35`)
규칙 커밋(`b17ce35`, 위 절) 뒤에 (1) 오프라인 서비스 19:08~19:10, (2) dms-280-car 서비스 19:12~20:07 순서로 돌렸다. 대상마다 새 일회용 토큰, 둘 다 `127.0.0.1:8093`. 요약 `reports/dast/2026-10-15.json`. 그동안 다른 프로젝트 작업들이 CPU를 거의 100% 쓰고 있어 (2)의 active scan이 52분 58초 걸렸다(2026-10-04는 17분 30초, 120분 한도 안).

| 대상 | High | Medium | Low | Informational | 스캔 중 요청 (처리 코드에 닿은 업로드) | 5xx |
|---|---|---|---|---|---|---|
| (1) 오프라인 | 0 | 0 | 0 | 3 (Modern Web Application, User Agent Fuzzer, **Information Disclosure - Suspicious Comments**) | 6,895 (`POST /inspect` 200 517회, `POST /calibrate` 200 1,320회) | 0 |
| (2) dms-280-car | **1 (SQL Injection - SQLite (Time Based))** | 0 | 0 | 3 (같은 셋) | 6,890 (516회, 1,323회) | 0 |

- active 규칙: 두 대상 모두 53개 시작, 50개 완료, 3개 건너뜀(DOM XSS, Log4Shell, 스크립트 규칙. 2026-10-04와 같은 사유), 완료 중 9개는 보낸 요청 0건, Parameter Tamper의 ZAP 내부 NullPointerException 16회. 2026-10-04와 같다
- **새 High 1건: SQL Injection - SQLite (Time Based), `POST /calibrate`의 `images` 매개변수, 신뢰도 Medium. 오탐으로 판단해 고치지 않는다.** ZAP은 업로드 파일 이름(`normal-2.bmp`)을 `case randomblob(10000000) when not null then 1 else 1 end` 같은 SQLite 지연 구문으로 바꿨을 때 응답이 1,027 → 1,207 → 1,388ms로 길어졌다는 이유로 경보했다. 사유: (가) 서비스에는 데이터베이스도 SQL도 없고(`src/`에 sqlite·SQL 실행 코드 없음) 업로드 파일 이름을 읽지 않는다(파일 바이트만 디코딩한다). (나) 이 요청은 업로드마다 모델이 CPU에서 돌아 응답이 1초 안팎이고, 스캔 동안 CPU가 다른 작업으로 거의 100%라 수백 ms의 흔들림은 흔하다. 같은 규칙이 2026-10-04에는 40초 동안 270건을 보내 경보 0건이었고 이번에는 146초에 264건이었다. (다) 같은 서비스를 다시 띄워 그 `/calibrate` 요청을 원래 이름, randomblob(1e7), randomblob(1e8) 세 가지로 순서를 돌려 가며 15번씩 보낸 확인(`work/` 아래 로컬 스크립트, 결과는 요약 JSON의 `false_positive_check`)에서 응답 시간 중앙값이 948 / 970 / 976ms, 범위가 708~2,137ms로 겹쳤고 모두 200이었다. 구문 크기에 따라 체계적으로 늘지 않았다
- 새 Informational 1건: Information Disclosure - Suspicious Comments(`/static/demo.js`). 2026-10-13~14에 데모 페이지 스크립트에 한국어 설명 주석을 넣은 뒤라 생긴 것이다. 주석은 화면 동작과 보안 처리(서버 문자열을 `textContent`로만 넣는 이유 등)를 설명하고 비밀값은 없다. 고치지 않는다
- **판정**: 규칙의 통과 조건(2026-10-04 ZAP 기록에 없던 High·Medium 없음)을 글자 그대로 적용하면 (2)는 새 High 1건이 있어 통과가 아니다. 규칙대로 그 경보를 고치지 않는 사유를 위에 적었다(오탐). 그 밖에 새 High·Medium은 없고 5xx도 없다. 다음 재스캔은 CPU가 한가할 때 돌려, 이 시간 기반 규칙이 다시 경보하는지 본다

### 2026-10-18 dms-280-car 재스캔 (CPU가 한가할 때, 2026-10-15 규칙 그대로)
2026-10-15 결과 절에 적은 대로, SQLite 시간 기반 경보가 CPU 부하에서 나온 것인지 보려고 CPU가 한가할 때(시작 직전 2~4%) 대상 (2)만 같은 설정으로 다시 스캔했다. 서비스 코드는 `b17ce35`와 같다(그 뒤 커밋은 문서·리포트·실험 코드뿐). 요약 `reports/dast/2026-10-18.json`.
- 경보: High 0, Medium 0, Low 0, Informational 3(Information Disclosure - Suspicious Comments, Modern Web Application, User Agent Fuzzer). **SQLite 시간 기반 규칙은 30초 동안 257건을 보내 경보 0건**이었다(2026-10-15 부하 상태에서는 146초, 경보 1건). 2026-10-15의 High를 오탐으로 본 판단과 맞는다
- active 규칙 53개 시작, 50개 완료, 3개 건너뜀(DOM XSS, Log4Shell, 스크립트 규칙, 같은 사유). active scan 10분 26초. 요청 6,885건, 5xx 0
- **판정: 통과.** 2026-10-04 ZAP 기록에 없던 High·Medium이 없다

### 2026-10-30 요청·시작 경로를 바꾼 뒤 재스캔 (규칙, 스캔 전에 커밋)
2026-10-18 스캔 뒤 서비스 코드가 바뀌었다. 요청 경로: 진행 중인 업로드 요청(`POST /inspect`·`POST /calibrate`)이 작업자 + 대기열에 이르면 본문을 받기 전에 `503`과 `Retry-After: 1`로 거절하고 재보정은 한 번에 하나만 받으며, 업로드를 임시 저장소에 쓰다 공간이 모자라면 `503`으로 답한다(`4ee11da`). 이미 RGB인 그림은 사본 없이 디코딩하고, 모델의 이상 맵이 유한하지 않으면 판정 대신 `500`을 낸다(`6d4ff61`). 임계값 변경과 관리자 토큰 거절을 표준 출력에 감사 줄로 남기고 `/categories`에 모델 지문의 앞 12자를 더했다(`2d6b4f2`). 시작 경로: 모델 세션마다 회색 이미지로 예열하고(`6d4ff61`), 배포 설정의 기대 모델 지문(`DEFECT_INSPECT_EXPECTED_MODEL_SHA256`)이 있으면 그 목록에 없는 모델 파일로는 뜨지 않는다(`b577854`). 경로와 보호 장치(Host 검사, 다른 출처의 쓰기 요청 거절, 관리자 토큰, 본문 크기 제한, 보안 헤더)는 그대로다.

- 도구와 설정은 2026-10-18과 같다(OWASP ZAP 2.17.0, `zap/automation.yaml`, `zap/run_zap.sh`, 기본 정책 active scan, 120분 한도)
- 대상은 (1) 오프라인 서비스(합성 범주 `demo`) 하나이고, 이 규칙을 넣은 커밋의 서비스 코드로 스캔한다. 서비스는 `127.0.0.1:8096`에 띄운다(같은 PC에서 도는 다른 서비스와 포트가 겹치지 않게). 시드는 `hawk/make_seed_har.py --host http://127.0.0.1:8096`으로 만든 것이고, 같은 스크립트를 기본값으로 돌린 결과가 `hawk/seed.har`와 바이트 단위로 같으며 8096 시드는 그 파일에서 주소만 다르다(스캔 전에 확인함). 새 일회용 토큰
- (2) dms-280-car 대상은 이번에 스캔하지 않는다. 이번 재스캔은 실제 서빙 모델을 돌리지 않는 범위로 한정했다(이 대상은 업로드 요청 1,800여 건마다 모델이 CPU에서 돈다). 바뀐 요청 경로 가운데 입장 제한, 디코딩, 감사 기록은 모델과 상관없는 코드라 (1)에서도 똑같이 돈다. 복원 방식 검사기의 맵 유한성 검사는 (1)에서 돌지 않는다(오프라인 검사기는 PatchCore 흉내다). 시작 경로의 변경(예열, 기대 지문)은 DAST가 볼 수 없다(뜬 서비스의 응답만 본다). 이 둘은 테스트(`tests/test_service.py`, `tests/test_service_reconstruction.py`)와 CI의 Docker 스모크 검사가 맡는다
- **통과 조건**: 2026-10-04·2026-10-15의 (1) 기록에 없던 High 또는 Medium 경보(규칙 이름 기준)가 없다. 새로 나온 것은 고치거나, 고치지 않는 사유를 이 문서에 적는다. Low와 Informational은 적기만 한다. 2026-10-04부터 돌지 않은 active 규칙 세 개(DOM XSS, Log4Shell, 스크립트 규칙)는 이번에도 돌지 않을 것으로 보고, 돌았는지 다시 센다
- 서비스가 스캔 중에 낸 5xx를 상태 코드별로 센다. `503`은 이 서비스가 본문을 받기 전에 내는 의도된 거절이다(작업자·대기열이 찼거나 재보정이 이미 진행 중일 때. 서비스의 503은 모두 `Retry-After`를 붙인다). ZAP은 호스트마다 스레드 4개로 요청을 겹쳐 보내므로(`threadPerHost: 4`) 재보정 요청이 겹치면 503이 날 수 있다. 503은 오류로 세지 않고 따로 적으며, 판정에 쓰는 5xx 오류는 503이 아닌 5xx다(이전 기록의 "5xx 0"은 503도 없었다는 뜻이다)
- 요약은 `reports/dast/2026-10-30.json`(경보 이름, 규칙 번호, 위험도, 신뢰도, 건수, 경로 표본, 응답 코드 집계. 토큰과 요청 본문은 넣지 않는다)

### 2026-10-30 결과 (OWASP ZAP 2.17.0, 서비스 코드 `5cf1fc7`)
규칙 커밋(`5cf1fc7`, 위 절) 뒤 01:32~01:34(KST)에 (1) 오프라인 서비스를 `127.0.0.1:8096`, 새 일회용 토큰으로 스캔했다. 시작 직전 CPU 사용률은 36~62%였다(다른 작업이 돌고 있었다). 요약 `reports/dast/2026-10-30.json`.

| 대상 | High | Medium | Low | Informational | 스캔 중 요청 (처리 코드에 닿은 업로드) | 503 (의도된 거절) | 그 밖의 5xx |
|---|---|---|---|---|---|---|---|
| (1) 오프라인 | **1 (SQL Injection)** | 0 | 0 | 2 (Information Disclosure - Suspicious Comments, User Agent Fuzzer) | 6,860 (`POST /inspect` 200 518회, `POST /calibrate` 200 1,303회) | 753 (모두 `POST /calibrate`) | 0 |

- active 규칙: 53개 시작, 50개 완료, 3개 건너뜀(DOM XSS, Log4Shell, 스크립트 규칙. 2026-10-04와 같은 사유), 완료 중 9개는 보낸 요청 0건, Parameter Tamper의 ZAP 내부 NullPointerException 16회. 이전 기록과 같다. active scan은 30초
- 503 753회는 모두 `POST /calibrate`였다. 재보정을 한 번에 하나만 받게 한 뒤(`4ee11da`) ZAP이 겹쳐 보낸 재보정 요청이 본문을 받기 전에 거절된 것이다(이전 스캔에는 503이 없었다). 503이 아닌 5xx는 0이다. 감사 줄은 재보정 1,303줄·되돌리기 19줄로, 성공한 `POST /calibrate`·`DELETE /calibrate` 수와 같았고 모두 JSON 한 줄이었다(모든 요청에 관리자 토큰을 붙였으므로 거절 줄은 없다)
- **새 High 1건: SQL Injection(40018, 불리언 방식), `POST /calibrate`의 `allow_unguaranteed`·`alpha`·`category` 세 곳, 신뢰도 Medium. 오탐으로 판단해 고치지 않는다.** ZAP은 값을 참 조건(`AND 1=1` 따위)과 거짓 조건(`AND 1=2` 따위)으로 바꿔 보낸 두 응답이 달랐다는 이유로 경보했다. 사유: (가) 서비스에는 데이터베이스도 SQL도 없다(2026-10-15 판단과 같음). 세 값은 FastAPI 폼 검증(`alpha`는 0 초과 0.5 이하의 실수, `allow_unguaranteed`는 참거짓, `category`는 1~64자)을 거쳐 범주 이름 대조와 conformal 순위 계산에만 쓰인다. (나) `/calibrate`는 한 번에 하나만 받아, 다른 재보정이 진행 중이면 본문을 받기 전에 `503`을 낸다. ZAP은 호스트마다 스레드 4개로 요청을 겹쳐 보내므로, 두 조건 가운데 한쪽만 503을 받으면 응답이 달라진다. 이번 스캔의 503 753회가 모두 이 경로였다. (다) 새로 띄운 오프라인 서비스에 시드의 `/calibrate` 요청을 그 값만 ZAP이 보고한 조건으로 바꿔 한 번에 하나씩 5바퀴 보낸 확인(로컬 일회용 스크립트, 결과는 요약 JSON의 `false_positive_check`)에서 `allow_unguaranteed`·`alpha`는 두 조건 모두 같은 본문의 422, `category`는 세 값 모두 같은 404였다. 다른 재보정 하나를 본문 도중에 멈춰 둔 동안에는 모든 조건이 같은 503(`Retry-After: 1`)이었고, 풀어 주자 다시 422·404였다. 응답 차이는 조건이 아니라 보낸 시점에서 왔다
- 2026-10-04·10-15의 (1)에 있던 Informational 가운데 Modern Web Application(10109)은 이번에 나오지 않았다(원인은 따로 보지 않았다)
- **판정**: 규칙의 통과 조건(2026-10-04·10-15의 (1) 기록에 없던 High·Medium 없음)을 글자 그대로 적용하면 새 High 1건이 있어 통과가 아니다. 규칙대로 그 경보를 고치지 않는 사유를 위에 적었다(오탐). 그 밖에 새 High·Medium은 없고 503이 아닌 5xx도 없다. 같은 설정으로 다시 스캔하면 이 경보는 다시 날 수 있다(ZAP의 겹친 요청과 한 번에 하나인 재보정이 그대로다). 이전 기록과 비교하려고 스캔 설정은 바꾸지 않았다. dms-280-car 대상은 규칙대로 이번에 스캔하지 않았다

### 2026-11-05 본문 전 판단을 고친 뒤 재스캔: 오프라인 서비스와 dms-280-car (규칙, 스캔 전에 커밋)
2026-10-30 오프라인 스캔(서비스 코드 `5cf1fc7`) 뒤 요청 경로가 한 번 바뀌었다. 본문을 받기 전에 관리자 경로·업로드 자리·본문 한계를 판단하는 장치가, 서비스가 하위 경로(`root_path`)에 배치됐을 때도 라우터와 같은 앱 안 경로로 라우트를 고른다(`394483b`). `root_path`가 없으면 동작은 전과 같다. dms-280-car 대상은 2026-10-18(서비스 코드 `b17ce35`) 뒤로 스캔하지 않았고, 그 사이 요청 경로(`4ee11da`의 본문 전 503 입장 제한·한 번에 하나인 재보정·임시 저장소가 모자랄 때의 503, `6d4ff61`의 사본 없는 RGB 디코딩과 이상 맵 유한성 검사, `2d6b4f2`의 감사 줄과 `/categories` 모델 지문, 위 `394483b`), 시작 경로(`6d4ff61`의 예열, `b577854`의 기대 모델 지문), 서비스 이미지의 라이브러리(`d4e3380`의 Pillow 13.0.0·opentelemetry-api 1.45.1)가 바뀌었다. 복원 방식 검사기의 맵 유한성 검사는 이 대상에서만 돈다. 경로와 보호 장치(Host 검사, 다른 출처의 쓰기 요청 거절, 관리자 토큰, 본문 크기 제한, 보안 헤더)는 그대로다.

- 도구와 설정은 2026-10-30과 같다(OWASP ZAP 2.17.0, `zap/automation.yaml`, `zap/run_zap.sh`, 기본 정책 active scan, 호스트마다 스레드 4개, 120분 한도). 이전 기록과 비교하려고 스캔 설정은 바꾸지 않는다. 120분에서 끊기면 그 사실을 적는다
- 대상 두 개를 이 규칙을 넣은 커밋의 서비스 코드로 (1) → (2) 순서로 스캔한다. 둘 다 이 PC에서 uvicorn으로 `127.0.0.1:8096`에 `root_path` 없이 띄우고(이전 기록과 같은 조건), 대상마다 새 일회용 관리자 토큰을 쓴다. 하위 경로 배치에서의 판단은 테스트(`tests/test_service.py`)가 맡는다
  - (1) 오프라인 서비스(`create_offline_app`, 합성 범주 `demo`). 시드는 `hawk/make_seed_har.py --host http://127.0.0.1:8096`
  - (2) `DEFECT_INSPECT_ARTIFACTS=artifacts/dms-280-car`의 `create_app`(6단계 서빙 세트, 12개 범주, onnxruntime CPU FP32). 배포 권장대로 `DEFECT_INSPECT_EXPECTED_MODEL_SHA256`에 그 모델 파일의 지문(`3bcf8e04…`)을 준다. 기대 지문은 시작할 때만 대조하므로 요청 경로를 바꾸지 않는다. 시드는 `hawk/make_seed_har.py --host http://127.0.0.1:8096 --category pcb1`
  - 스캔 전에, 같은 스크립트를 기본 주소로 돌린 결과가 `hawk/seed.har`·`hawk/seed-pcb1.har`와 바이트 단위로 같고 8096 시드는 그 파일에서 주소만 다른지 확인한다
- **통과 조건** (대상마다): (가) 같은 대상의 2026-10-04 ZAP 기록에 없던 High 또는 Medium 경보(규칙 이름 기준)가 없다. 2026-10-04 기록은 두 대상 모두 High·Medium이 0이다. 2026-10-15 (2)의 SQLite 시간 기반 High와 2026-10-30 (1)의 불리언 SQL 주입 High는 오탐 사유와 함께 '통과 아님'으로 적은 것이라 비교 기준에 넣지 않는다 (나) 서비스가 스캔 중에 낸 503이 아닌 5xx가 0이다. 새로 나온 High·Medium은 고치거나, 고치지 않는 사유를 이 문서에 적는다. Low와 Informational은 적기만 한다
- 앞선 오탐과 같은 경보가 다시 나면 그때와 같은 방법으로 확인하고, 판정은 통과 조건을 글자 그대로 적용한다. 불리언 SQL 주입(40018)이 `POST /calibrate`에 나면, 새로 띄운 같은 대상 서비스에 시드의 재보정 요청을 ZAP이 보고한 참·거짓 조건으로 바꿔 한 번에 하나씩 순서를 돌려 가며 보내고, 다른 재보정 하나를 본문 도중에 붙잡아 둔 동안과 푼 뒤에 다시 보낸다(2026-10-30). SQLite 시간 기반 SQL 주입이 나면, 그 요청을 원래 값과 ZAP의 지연 구문들로 순서를 돌려 가며 여러 번 보내 응답 시간이 구문 크기에 따라 체계적으로 늘어나는지 본다(2026-10-15). 결과는 요약 JSON의 `false_positive_check`에 적는다
- 확인만 하는 것(판정에 쓰지 않음): 2026-10-04부터 돌지 않은 active 규칙 세 개(DOM XSS, Log4Shell, 스크립트 규칙)를 포함한 규칙별 시작·완료·건너뜀, 응답 상태 코드 집계와 503 수(본문을 받기 전의 의도된 거절, 모두 `Retry-After`), 감사 줄 수가 성공한 `POST /calibrate`·`DELETE /calibrate` 수와 같은지와 모두 JSON 한 줄인지, active scan 시간, 스캔 직전 CPU 사용률. 지연을 재는 일이 아니므로 CPU가 한가해지기를 최대 10분만 기다리고, 그래도 바쁘면 그 사실을 적고 스캔한다(바쁜 CPU는 시간 기반 규칙의 오탐을 만들 수 있다, 2026-10-15)
- 요약은 `reports/dast/2026-11-05.json`(대상마다 경보 이름, 규칙 번호, 위험도, 신뢰도, 건수, 경로 표본, 응답 코드 집계, 503 수와 503이 아닌 5xx 수, 감사 줄 수. 토큰과 요청 본문은 넣지 않는다)
- 결과는 이 절 아래 결과 절에 적고, README 보안 점검 줄에 날짜를 붙인 한 문장을 더한다
- **2026-11-07 보충 (스캔 전)**: 규칙을 커밋한 2026-11-05에는 스캔하지 못했고 2026-11-07에 스캔한다. 그 사이 서비스 코드는 바뀌지 않았다(규칙 커밋 `c16087e`와 이 보충 커밋의 서비스 코드는 `394483b` 그대로). 그래서 요약 파일은 위에 적은 `reports/dast/2026-11-05.json` 대신 스캔 날짜의 `reports/dast/2026-11-07.json`에 두고, 결과 절 제목도 스캔 날짜로 적는다. 도구·설정·대상·통과 조건·확인만 하는 것은 그대로다

### 2026-11-07 결과 (OWASP ZAP 2.17.0, 서비스 코드 `394483b`, 규칙 `c16087e`와 보충 `e405cf5`)
규칙 커밋과 보충 뒤, 서비스 코드가 `394483b`와 같은 커밋(`613ca0e`)의 깨끗한 작업 트리에서 (1) 오프라인 서비스를 15:27~15:31, (2) dms-280-car 서비스를 15:58~16:10(KST)에 각각 `127.0.0.1:8096`(`root_path` 없음), 새 일회용 토큰으로 스캔했다. 두 대상 모두 CPU가 한가해지기를 10분 기다렸지만 같은 PC의 다른 작업이 계속 돌아 규칙대로 그대로 스캔했다. 스캔 직전 CPU 사용률은 (1) 96·82·62·67%, (2) 24·21·44·17%였다. (2)의 첫 시도(15:43)는 ZAP이 자체 프록시 포트 8080을 열지 못해 시작하지 못했다(같은 PC의 다른 컨테이너가 `127.0.0.1:8080`을 쓰고 있었다). `zap/run_zap.sh`에 그 포트를 바꾸는 선택 변수 `ZAP_PORT`를 더한 뒤(`613ca0e`, 기본값 8080 그대로) 18080으로 다시 돌렸다. 이 포트는 ZAP이 자기 프록시를 듣는 포트일 뿐이고 이 계획은 그 프록시를 거치지 않으므로, 스캔 대상·정책·스레드 4개·120분 한도(`zap/automation.yaml`)는 2026-10-30·10-18과 같다. 요약 `reports/dast/2026-11-07.json`.

| 대상 | High | Medium | Low | Informational | 스캔 중 요청 (처리 코드에 닿은 업로드) | 503 (의도된 거절) | 그 밖의 5xx |
|---|---|---|---|---|---|---|---|
| (1) 오프라인 | **1 (SQL Injection)** | 0 | 0 | 2 (Modern Web Application, User Agent Fuzzer) | 6,881 (`POST /inspect` 200 517회, `POST /calibrate` 200 1,288회) | 738 (모두 `POST /calibrate`) | 0 |
| (2) dms-280-car | **1 (SQL Injection)** | 0 | 0 | 3 (Information Disclosure - Suspicious Comments, Modern Web Application, User Agent Fuzzer) | 6,880 (`POST /inspect` 200 519회, `POST /calibrate` 200 1,315회) | 795 (모두 `POST /calibrate`) | 0 |

- active 규칙: 두 대상 모두 53개 시작, 50개 완료, 3개 건너뜀(DOM XSS, Log4Shell, 스크립트 규칙. 2026-10-04와 같은 사유), 완료 중 9개는 보낸 요청 0건, Parameter Tamper의 ZAP 내부 NullPointerException 16회. 이전 기록과 같다. active scan은 (1) 55초(2026-10-30에는 30초. CPU가 바빴다), (2) 11분 18초(2026-10-18에는 10분 26초)
- 503은 모두 `POST /calibrate`였다((1) 738회, (2) 795회). 재보정을 한 번에 하나만 받으므로(`4ee11da`) ZAP이 겹쳐 보낸 재보정 요청이 본문을 받기 전에 거절된 것이다. 503이 아닌 5xx는 두 대상 모두 0이다. 감사 줄은 (1) 재보정 1,288줄·되돌리기 19줄, (2) 재보정 1,315줄·되돌리기 19줄로 각각 성공한 `POST /calibrate`·`DELETE /calibrate` 수와 같았고 모두 JSON 한 줄이었다(모든 요청에 관리자 토큰을 붙였으므로 거절 줄은 없다)
- **새 High 1건(두 대상 모두): SQL Injection(40018, 불리언 방식), `POST /calibrate`, 신뢰도 Medium. (1)은 `allow_unguaranteed`·`alpha`·`category` 세 곳(2026-10-30과 같음), (2)는 `category`와 파일 필드 `images` 두 곳. 2026-10-30과 같은 원인의 오탐으로 판단해 고치지 않는다.** 사유: (가) 서비스에는 데이터베이스도 SQL도 없다. 세 폼 값은 FastAPI 폼 검증(`alpha`는 0 초과 0.5 이하의 실수, `allow_unguaranteed`는 참거짓, `category`는 1~64자)을 거쳐 범주 이름 대조와 conformal 순위 계산에만 쓰이고, `images`는 이미지 디코더가 읽는 파일 내용이다. (나) `/calibrate`는 한 번에 하나만 받아, 다른 재보정이 진행 중이면 본문을 받기 전에 `503`을 낸다. ZAP은 호스트마다 스레드 4개로 요청을 겹쳐 보내므로, 두 조건 가운데 한쪽만 503을 받으면 응답이 달라진다. 이번 503도 모두 이 경로였다. (다) 규칙대로 새로 띄운 같은 대상 서비스에 시드의 재보정 요청을 ZAP이 보고한 참·거짓 조건으로 바꿔 한 번에 하나씩 순서를 돌려 가며 5바퀴 보낸 확인(로컬 일회용 스크립트, 결과는 요약 JSON의 `false_positive_check`): (1) `allow_unguaranteed`·`alpha`는 두 조건 모두 같은 본문의 422, `category`는 같은 404. (2) `category`는 같은 404, `images`(첫 파일의 내용을 조건 구문으로 바꿈)는 두 조건 모두 같은 본문의 400(이미지로 읽히지 않음). 두 대상 모두, 다른 재보정 하나를 본문 도중에 멈춰 둔 동안에는 모든 조건이 같은 503(`Retry-After: 1`)이었고, 풀어 주자 다시 같은 422·404·400이었다. 응답 차이는 조건이 아니라 보낸 시점에서 왔다
- Informational: (1)에서는 2026-10-30에 있던 Information Disclosure - Suspicious Comments(10027, `/static/demo.js`)가 이번에 나오지 않았고, 2026-10-30에 없던 Modern Web Application(10109)이 다시 나왔다(둘 다 passive 규칙이고 원인은 따로 보지 않았다). (2)는 2026-10-18과 같은 셋이다
- **판정**: 규칙의 통과 조건 (가)(같은 대상의 2026-10-04 기록에 없던 High·Medium 없음)를 글자 그대로 적용하면 두 대상 모두 새 High 1건이 있어 통과가 아니다. 규칙대로 그 경보를 고치지 않는 사유를 위에 적었다(오탐, 2026-10-30과 같은 경보·같은 원인). 그 밖에 새 High·Medium은 없고, 조건 (나)(503이 아닌 5xx 0)는 두 대상 모두 지켰다. `394483b`의 본문 전 판단 변경과 2026-10-18 뒤 dms-280-car 대상이 보지 못했던 요청 경로 변경(`4ee11da`·`6d4ff61`·`2d6b4f2`)은 새 경보를 만들지 않았다. 같은 설정으로 다시 스캔하면 40018은 다시 날 수 있다(ZAP의 겹친 요청과 한 번에 하나인 재보정이 그대로다). 이전 기록과 비교하려고 스캔 설정은 바꾸지 않았다

## 운영 확인

DAST와 의존성 대조 밖에서, compose 기본 설정으로 띄운 컨테이너가 자원 한도 안에서 도는지 실제로 채워 본 기록이다. 서비스 코드와 설정은 바꾸지 않고 확인만 한다.

### 2026-11-05 업로드 임시 저장소를 가장 많이 채웠을 때의 컨테이너 메모리 (규칙, 확인 전에 커밋)
compose.yaml 주석과 README는 `/tmp` tmpfs 768m가 서비스가 동시에 받아 둘 수 있는 업로드의 최대 합(검사 17개 × (파일 20MiB + multipart 여유 64KiB) + 재보정 1개 × 256MiB ≈ 597MiB)에 맞고 `mem_limit` 2g 안에도 든다고 계산으로만 적었다. tmpfs에 쓴 업로드는 컨테이너 메모리 한도에 포함되므로, 실제 서빙 세트로 띄운 컨테이너에서 그 상태를 만들어 메모리를 잰다. 그때까지 컨테이너의 자원 사용으로 잰 기록은 작은 시험 아티팩트로 띄운 컨테이너의 프로세스·스레드 수뿐이다(compose.yaml의 `pids_limit` 주석).

- 대상: 이 규칙을 넣은 커밋에서 `docker compose build --pull`로 다시 빌드한 이미지를 compose.yaml 기본값 그대로 띄운 컨테이너(작업자 2, 대기열 16, `/tmp` tmpfs 768m, `mem_limit` 2g, 읽기 전용 루트, `cap_drop: ALL`, `pids_limit` 256, uvicorn `--limit-concurrency 32`). 아티팩트는 6단계 서빙 세트 `artifacts/dms-280-car`(FP32)이고, `DEFECT_INSPECT_EXPECTED_MODEL_SHA256`에 그 모델 파일의 지문(`3bcf8e04…`)을 준다. 관리자 토큰은 새 일회용 값을 셸 변수로만 준다. 올리는 사진은 모두 무작위 잡음으로 만든 합성 PNG이고, 봉인 테스트 이미지는 읽지 않는다
- (0) 기동 확인(통과·실패만 적는다): `/healthz`가 `ok`와 12개 범주, `/categories`의 `model_sha256`이 모두 `3bcf8e04d771`, `docker inspect`로 healthy·RestartCount 0·읽기 전용 루트·CapDrop `ALL`·PidsLimit 256·`/tmp` tmpfs 768m·메모리 한도 2GiB. CI의 Docker 스모크 검사와 같은 합성 PNG 두 장(회색 64px, 그 위에 흰 칸을 넣은 것)으로 `POST /inspect`(pcb1)가 200이고 `is_defect`가 `score > threshold`와 같다. 따로, 기대 지문을 목록에 없는 값(0 64개)으로 주고 같은 설정으로 한 번 띄우면 0이 아닌 코드로 끝나고 로그에 기대 지문 거절 문구가 나온다
- (가) 가장 많이 받아 둔 상태: 본문을 끝의 multipart 닫는 경계만 빼고 보낸 채 연결을 쥐는 클라이언트로, 검사 17개(파일 하나 20,941,072바이트 PNG, 파일 한도 20MiB의 99.9%)와 재보정 1개(14,083,038바이트 PNG 19장, 본문이 한도 256MiB의 99.7%)를 동시에 붙잡는다. `/tmp` 사용량이 더 늘지 않으면 그 상태를 기록하고, 그동안 `/healthz`와 작은 검사(64KiB 미만 PNG)를 보낸다. 자리(작업자 + 대기열 18개)가 다 찼으므로 작은 검사는 본문을 받기 전에 503 + `Retry-After`여야 한다. 그다음 붙잡은 18개의 남은 바이트를 보내 모두 처리되게 한다
- (나) 임시 저장소가 찼을 때: 컨테이너 안에서 `/tmp`를 남은 공간이 없을 때까지 파일 하나로 채운 뒤(`docker exec`), 1MiB가 넘는 검사 업로드(2,020,901바이트 PNG) 3번과 1MiB 이하 검사(64KiB 미만 PNG) 3번을 보내고 `/healthz`를 본다. 끝나면 채운 파일을 지운다
- 기록: 단계마다 컨테이너 cgroup의 `memory.peak`·`memory.current`, `memory.stat`의 `anon`·`shmem`(tmpfs)·`file`, `/tmp` 사용량(`df`), `docker stats` 한 번, `docker inspect`의 OOMKilled·RestartCount, 응답 코드 집계와 걸린 시간. 단계 동안 컨테이너 안에서 0.5초 간격으로 `memory.current`를 읽어 최고값도 적는다. 지연을 재지 않으므로 CPU가 한가해지기를 기다리지 않고, 다른 프로그램이 CPU를 쓰고 있었으면 그 사실만 적는다
- **통과 조건**: (1) 모든 단계에서 OOMKilled false, RestartCount 0 (2) `/healthz`가 매번 200 (3) (가)의 작은 검사는 503 + `Retry-After`이고, 붙잡았던 18개는 모두 200(검사 17개, 19장 재보정 1개) (4) (나)의 1MiB가 넘는 업로드는 503 + `Retry-After`(400·500이 아님)이고 1MiB 이하 검사는 200 (5) 컨테이너 `memory.peak` < 2GiB(2,147,483,648바이트). 남은 여유를 MiB로 적는다
- 통과하지 못하면 수치와 원인을 결과에 적는다. 대기열·tmpfs·`mem_limit`을 바꾸는 일은 결과를 적은 뒤 따로 커밋한다(서비스나 compose를 바꾸면 다음 DAST 범위에 든다). 이 확인으로 기존 기록의 수치·판정을 고치지 않는다
- 결과는 이 절 아래 결과 절에 적고, compose.yaml의 tmpfs 주석과 README의 `/tmp` tmpfs 문장 뒤에 날짜를 붙인 실측 한 문장을 더한다. 확인에 쓴 클라이언트 스크립트는 저장소에 넣지 않는다(일회용)

### 2026-11-05 결과 (규칙 커밋 `17c5958`, 서비스 코드 `394483b`)
규칙 커밋(`17c5958`, 위 절)의 깨끗한 작업 트리에서 06:29~06:33(KST)에 확인했다. `docker compose build --pull`로 다시 빌드한 이미지(바탕 `python:3.11-slim@sha256:db20d8506e2f…`, uv `0.13.6@sha256:af22a52f5eaf…`)를 compose.yaml 기본값, 6단계 서빙 세트, 기대 지문으로 띄웠다. 확인하는 동안 다른 프로그램이 CPU를 쓰고 있었다(전체 사용률 54~82%). 지연은 재지 않았다. 규칙 변경 없음.

| 단계 | `/tmp` 사용 | `memory.current` | 그때까지의 `memory.peak` | 응답 |
|---|---|---|---|---|
| 띄운 직후 | 0 | 288.3MiB | 292.3MiB | (0) 기동 확인 모두 통과 |
| (가) 18개를 붙잡은 상태 | 594.7MiB | 895.8MiB(anon 290.6, shmem 594.7) | 895.8MiB | 작은 검사 3번 모두 503 + `Retry-After: 1`, `/healthz` 200 |
| (가) 붙잡은 18개를 처리한 뒤 | 0 | 322.9MiB | 990.1MiB | 검사 17개와 재보정(19장, 보장)이 모두 200 |
| (나) `/tmp`를 다 채운 상태 | 768.0MiB(100%) | 1,094.0MiB(anon 315.8, shmem 768.0) | 1,097.1MiB | 1MiB가 넘는 업로드 3번 모두 503 + `Retry-After: 1`, 1MiB 이하 검사 3번 200, `/healthz` 200 |
| (나) 채운 파일을 지운 뒤 | 0 | 327.9MiB | **1,106.6MiB** | — |

- (0) 기동 확인: `/healthz`가 `ok`와 12개 범주, `/categories` 12개 모두 `model_sha256` `3bcf8e04d771`, healthy·RestartCount 0·읽기 전용 루트·CapDrop `ALL`·PidsLimit 256·`/tmp` tmpfs `size=768m`·메모리 한도 2,147,483,648바이트(사용자 `inspect`, 기본 명령 그대로). 합성 PNG 두 장의 `POST /inspect`(pcb1)는 200이고 `is_defect`가 `score > threshold`와 같았다. 두 장 모두 불량으로 판정됐다: 회색 64px 그림은 pcb1의 정상 사진과 거리가 멀어 실제 서빙 모델에서는 점수가 임계값보다 높다(CI는 작은 시험 모델이라 회색이 양품이다). 기대 지문을 0 64개로 준 실행은 종료 코드 1로 끝났고, 로그에 "is not one of the expected model digests in DEFECT_INSPECT_EXPECTED_MODEL_SHA256"이 나왔다
- (가) 붙잡은 본문은 검사 20,941,310바이트 × 17과 재보정 267,580,183바이트이고, 보내는 데 12.7초 걸렸다. `/tmp` 사용량은 609,012KiB(594.7MiB, 계산한 최대 597MiB의 99.6%)에서 멈췄다. 그동안 프로세스·스레드 수는 56개였다(쉴 때 36~42개). 남은 바이트를 보낸 뒤 18개가 처리되는 14.2초 동안 `/healthz` 13번이 모두 200이었다
- 메모리 최고값은 붙잡은 상태(895.8MiB)보다 처리하는 동안(990.1MiB) 높았다. 처리가 시작되면 `/tmp`의 업로드를 읽어 디코딩하므로, 그 동안에는 아직 남은 업로드 파일과 읽어 들인 바이트·디코딩한 그림이 함께 있다. 0.5초 간격 표본의 최고값(960.9MiB, (나)에서는 1,097.4MiB)은 `memory.peak`보다 조금 낮았다(표본 사이의 순간을 놓친다)
- (나) `/tmp`를 다 채운 상태에서 1MiB가 넘는 업로드는 모두 503과 `Retry-After: 1`("temporary storage is full")을 받았고(400·500 없음), 그때마다 경고 로그 "the upload spool ran out of space"가 한 줄씩(3줄) 남았다. 1MiB 이하 검사는 임시 파일 없이 메모리에서 받아 200이었다. 마지막 `memory.peak`(1,106.6MiB)는 채운 파일을 지우기 전후의 확인용 `docker exec` 프로세스까지 포함한 값이다
- 모든 단계에서 OOMKilled false, RestartCount 0, swap 사용 0이었다. 확인 뒤 컨테이너를 내렸다(남은 컨테이너 없음)
- **판정: 통과.** 통과 조건 다섯 가지를 모두 지켰다. compose의 계산(동시에 받아 둘 수 있는 업로드 최대 597MiB가 tmpfs 768m에 든다)은 실측 594.7MiB로 맞았다. 컨테이너 메모리 최고값은 그 상태와 처리 동안 990.1MiB, `/tmp`를 768MiB까지 다 채운 경우까지 넣으면 1,106.6MiB로, 2GiB 한도까지 941.4MiB가 남았다. 설정은 바꾸지 않는다
- 한계: 한 번 확인한 값이다. 업로드한 사진은 무작위 잡음 PNG라 디코딩한 크기가 약 700만 화소(2,640 × 2,640)였다. 서비스는 2,500만 화소까지 받으므로, 그런 사진 두 장을 작업자 둘이 동시에 디코딩하면 최고값이 이보다 높을 수 있다(디코딩한 그림의 크기는 화소 수에 비례한다). 그 경우는 재지 않았다
- **2026-11-05 정정 (검토 지적)**: 위 규칙 절 첫 문단의 두 문장이 그때까지의 공개 기록과 맞지 않는다. 규칙 문장은 확인 전에 커밋한 그대로 두고 여기에 바로잡는다. 규칙의 대상·절차·통과 조건과 이 결과의 수치·판정은 바뀌지 않는다. (가) "그때까지 컨테이너의 자원 사용으로 잰 기록은 작은 시험 아티팩트로 띄운 컨테이너의 프로세스·스레드 수뿐이다"는 틀렸다. 2026-10-04 서비스 모델 교체 점검의 결과 절([experiments.md](experiments.md) '서비스 모델 교체 점검'의 Docker 줄, `7d590d2`)에 compose 기본 아티팩트(dms-280-car)로 띄운 컨테이너의 메모리 292MiB(한도 2GiB)가 이미 적혀 있었다. 이번에 띄운 직후 값(`memory.current` 288.3MiB, `memory.peak` 292.3MiB)과 비슷하다. (나) "compose.yaml 주석과 README는 … `mem_limit` 2g 안에도 든다고 계산으로만 적었다"에서 `mem_limit` 2g 안에 든다는 문장은 compose.yaml 주석에만 있었다. 그때 README는 `/tmp` tmpfs 768m가 동시에 받아 둘 수 있는 업로드의 최대 합에 맞는다는 계산만 적었다

## 의존성 취약점 대조

위의 DAST 기록과 따로, 서비스 이미지에 들어가는 의존성(기본 패키지 + `serve` 그룹, `uv.lock`에 고정된 버전)을 공개 취약점 데이터베이스와 대조한 기록이다. CI의 `audit` 작업(`.github/workflows/ci.yml`)이 push와 pull request마다 같은 집합을 pip-audit으로 대조하므로, 이 절에는 그 대조가 경보를 냈거나 손으로 다시 확인한 때를 적는다.

### 로컬 감사 절차
저장소 루트에서 Git Bash로 돌린다. 내보낸 목록은 커밋하지 않는 임시 파일에 둔다.
```bash
out=$(mktemp)
# (1) CI audit 작업과 같은 두 줄: 잠금에서 서비스 집합을 해시와 함께 뽑아, 설치하지 않고 그 버전들을 대조한다.
uv export --frozen --no-dev --group serve --no-emit-project --format requirements-txt -o "$out"
uvx pip-audit@2.10.1 -r "$out" --disable-pip --require-hashes
# (2) 같은 목록을 OSV 데이터베이스로 한 번 더 대조한다(pip-audit의 기본 대조 대상은 PyPI의 취약점 정보다).
uvx pip-audit@2.10.1 -r "$out" --disable-pip --require-hashes -s osv
# (3) 업로드 디코딩을 맡는 Pillow를 GitHub Advisory Database에서 직접 조회한다. 아무 줄도 나오지 않아야 한다.
#     affects의 패키지 이름은 표기를 구분하므로 두 표기를 모두 본다. <버전>은 uv.lock의 pillow 버전이다.
gh api -X GET advisories -f ecosystem=pip -f affects=pillow@<버전> --jq '.[].ghsa_id'
gh api -X GET advisories -f ecosystem=pip -f affects=Pillow@<버전> --jq '.[].ghsa_id'
# (4) 학습용 그룹의 torch·torchvision은 pytorch-cu128 인덱스의 CUDA 빌드(2.11.0+cu128 등)라 PyPI에 없는 판이어서
#     pip-audit이 건너뛴다. OSV API에 이름과 버전(+cu128을 뗀 것)으로 직접 묻는다. 결과의 vulns가 권고 목록이다
#     (권고가 없으면 {}).
curl -s -X POST https://api.osv.dev/v1/query -d '{"package":{"name":"torch","ecosystem":"PyPI"},"version":"<버전>"}'
curl -s -X POST https://api.osv.dev/v1/query -d '{"package":{"name":"torchvision","ecosystem":"PyPI"},"version":"<버전>"}'
```
- (2)와 (3)을 두는 이유: 2026-10-24에 Pillow 12.3.0을 대조했을 때 (1)은 35건을 냈지만 OSV와 GitHub Advisory Database에는 39건이 있었다. (1)에 없던 4건 가운데 하나가 critical인 GHSA-8g5g-8vmr-9782다. 데이터베이스마다 실린 권고가 달라서, CI가 녹색이어도 다른 데이터베이스에만 실린 권고가 있을 수 있다
- (3)의 affects 조회는 이름 표기에 따라 결과가 갈렸다(같은 날 `pillow@12.3.0` 11건, `Pillow@12.3.0` 28건, 겹치는 것 없음)
- (2)는 `--require-hashes`를 주어도 해시가 있는지만 보고 값이 맞는지는 확인하지 않는다는 경고를 낸다. 해시 검증은 (1)이 한다
- (4)를 두는 이유: 학습용 그룹(`train`·`dinomaly`)은 서비스 이미지에 들어가지 않아 CI가 대조하지 않는다. 이 그룹을 손으로 대조할 때 (1)·(2)의 `--group serve`를 `--group train --group dinomaly`로 바꾸면 torch·torchvision은 "Dependency not found on PyPI"로 빠지므로, 이 둘은 (4)로 채운다
- 2026-10-29 추가: CI audit 작업이 (1)에 이어 (2)의 OSV 대조도 돈다. (3)과 (4), 그리고 서비스 집합 밖의 그룹 대조는 여전히 손으로 한다. 잠금 전체를 한 번에 볼 때는 (1)·(2)의 `--no-dev --group serve` 대신 `--all-groups`를 준다
- 2026-11-05 추가: CI audit 작업이 잠금 전체도 대조한다(아래 2026-11-05 절). 그 단계는 환경 표지를 떼어 모든 Python 갈래를 대조하고, torch·torchvision의 로컬 판 표시(`+cu128`)를 떼어 PyPI의 같은 판으로 대조하므로 (4)가 메우던 빈자리까지 덮는다. 바로 위의 `--all-groups` 방식은 환경 표지를 남긴 채 대조하므로 pip-audit을 돌리는 파이썬에 맞지 않는 갈래(예: Python 3.12 이상에서 쓰는 numpy 2.5.3)를 말없이 건너뛴다. (3)은 여전히 손으로 한다

### 2026-10-24 Pillow 12.3.0 → 13.0.0, opentelemetry-api 1.45.0 → 1.45.1
- **계기**: 위 절차 (1)을 다시 돌리니 `Found 36 known vulnerabilities in 2 packages`가 나왔다. pillow 12.3.0에 35건(수정 판은 모두 13.0.0), fastapi가 끌어오는 opentelemetry-api 1.45.0에 1건(CVE-2026-114603 / GHSA-xcq7-p3pf-p88h, tracestate를 제한 없이 해석함, 수정 판 1.45.1)이다. 권고들은 2026-10-20~22에 올라왔고 마지막 push(2026-10-18)보다 뒤라 CI는 녹색으로 남아 있었지만, 다음 push부터 audit 작업이 실패할 상태였다. (2)와 (3)에서는 pillow가 39건이었다(GHSA-8g5g-8vmr-9782, GHSA-xvh5-g6gv-559v, GHSA-rfgq-q3w2-pmv2, GHSA-4558-3h8r-v8xp가 더 있음)
- **조치**: `pyproject.toml`의 하한을 `pillow>=13.0`으로 올리고 `uv lock --upgrade-package pillow --upgrade-package opentelemetry-api`로 잠금에서 두 패키지만 올렸다. 다른 패키지의 버전은 바뀌지 않았다. uv 0.13.1이 잠금 형식을 revision 3에서 5로 다시 쓰면서 imagecodecs 2026.8.16의 free-threaded(cp314t·cp315t)·WebAssembly 휠 항목 16개를 뺐는데, 이 프로젝트의 Python 3.11 환경에서는 쓰지 않는 휠이다. Docker 이미지를 만드는 uv 0.12.12가 새 잠금을 그대로 읽는 것(같은 내보내기 결과, `uv sync --frozen --no-dev --group serve --no-install-project --dry-run` 통과)을 확인해 Dockerfile은 바꾸지 않았다
- **조치 뒤 대조**: (1)과 (2) 모두 `No known vulnerabilities found`, (3)은 `pillow@13.0.0`·`Pillow@13.0.0`·`opentelemetry-api@1.45.1` 모두 0건
- **도달 분석**: 서비스가 업로드를 다루는 경로는 `service._decode`(`Image.open(..., formats=("PNG", "JPEG", "BMP", "TIFF", "WEBP"))` → 화소 수·비트 깊이·투명도 검사 → `convert("RGB")` → `resize(..., BICUBIC)`)와 응답 PNG 인코딩(`input_png`, `heatmap_png`)이다. 비인증 `POST /inspect`와 관리자 `POST /calibrate`가 이 경로를 쓴다. 조치 전 판(12.3.0)을 기준으로 나눴고, 수치는 이 PC(Windows 휠)에서 쟀다

| 권고 | 내용 | 12.3.0 서비스 경로 | 근거 |
|---|---|---|---|
| GHSA-24rc-8hj2-v98c (CVE-2026-96638) | BMP RLE 디코더의 메모리 증폭 | 닿음 | 1,084바이트 BMP(BI_RLE8, 폭 200,000·높이 1, 델타 명령 하나)가 화소 수 검사를 통과해 400 없이 디코딩되고, 디코딩 동안 파이썬 메모리 최고치가 약 98MB였다(13.0.0에서는 1MB 미만). 폭을 화소 수 한도(2,500만)까지 키우면 같은 구조로 수 GB가 되어 메모리 한도 2GB인 컨테이너가 죽을 수 있다 |
| GHSA-5xm2-6676-qcp9 (CVE-2026-112399) | APNG 배경 처분이 화소 수 검사 전에 전체 바탕 그림을 잡음 | 닿음 | 127바이트 APNG(머리 6,000 × 6,000, 1×1 프레임, 배경 처분)가 화소 수 검사에서 400이 되기 전에 `Image.open` 안에서 프로세스 확정 메모리를 약 137MB 늘렸다(13.0.0에서는 2MB 미만). 머리에 적은 크기에 비례해 커진다 |
| GHSA-c6hw-5grf-ff96 (CVE-2026-112404), GHSA-q45j-58j2-45rg (CVE-2026-112897), GHSA-462x-pgg6-55fp (CVE-2026-112395) | TIFF 디코더의 범위 밖 읽기, 초기화하지 않은 메모리를 화소로 돌려줌 | 닿음 | 압축 TIFF(JPEG·LZW·Deflate·Group 4)는 번들 libtiff 디코더로 디코딩된다. 악용 재현은 하지 않았다. `preview=true`의 `input_png`는 디코딩한 화소를 줄인 그림이라(업로드가 모델 입력 크기와 같으면 화소 그대로), 섞여 든 힙 내용이 응답으로 나갈 수 있는 경로다 |
| GHSA-g4qv-2jxx-769h (CVE-2026-96643) | PNG eXIf 청크의 EXIF 해석이 2차 시간 | 닿지 않음 | `getexif()`에서만 해석한다. 서비스는 EXIF를 읽지 않는다(회전 정보도 쓰지 않는다) |
| GHSA-8g5g-8vmr-9782 (CVE-2026-112403, critical) | JPEG 디코더의 힙 넘침 | 닿지 않음(조건부) | JPEG `draft()`(`thumbnail()`이 부른다)로 축소 디코딩을 요청할 때만 생긴다. 서비스는 draft와 thumbnail을 쓰지 않는다. 디코딩을 빠르게 하려고 draft를 넣는다면 13.0.0 이상에서만 넣는다 |
| GHSA-78g4-xqwv-q3g8 | 번들 libopenjp2의 힙 넘침 | 닿지 않음 | JPEG 2000은 받는 형식이 아니라 그 디코더가 열리지 않는다 |
| CVE-2026-114603 (opentelemetry-api) | tracestate를 제한 없이 해석 | 닿지 않음 | 트레이서를 설정하지 않는다(잠금에 opentelemetry-sdk가 없고 `src/`에서 opentelemetry를 쓰지 않는다). CI audit 실패를 없애려고 올렸다 |
| 나머지 pillow 권고 | 받지 않는 형식, 하지 않는 연산 | 닿지 않음 | 받지 않는 형식(GIF, PDF, EPS·Ghostscript, XBM, MSP, SPIDER, DCX, MIC, PSD, WMF·EMF), JPEG로 열린 MPO의 다중 프레임 이동(`seek`), 서비스가 하지 않는 저장(TIFF·JPEG·GIF 인코딩)과 그림 연산(quantize, merge, transform, 순위·형태 필터, 글꼴), 쓰지 않는 모듈(TarIO)과 `frombytes`의 P;2L·P;4L 원시 모드에 걸린다. 권고 요약과 서비스 코드를 대조해 나눴고 하나씩 재현하지는 않았다 |

- **회귀 테스트** (`tests/test_service.py`): `test_a_tiny_rle_bmp_does_not_blow_up_into_a_huge_allocation`은 위 BMP를 `_decode`에 넣고 디코딩 동안 늘어난 파이썬 메모리 최고치가 20MB보다 작은지 본다(12.3.0에서 97.8MB로 실패, 13.0.0에서 통과). `test_the_installed_pillow_has_the_decoder_security_fixes`는 Pillow 주 버전이 13 이상인지와 위 APNG가 400인지를 본다. APNG의 바탕 그림은 Pillow의 C 코드가 잡는 메모리라 tracemalloc으로 잴 수 없어서, 잠금이 낮은 버전으로 돌아가는 것을 버전 단언으로 막는다
- **측정 기록에 미치는 영향**: 없다. 봉인 test가 아닌 VisA 학습 풀 6,493장을 리사이즈 캐시와 같은 호출로 280·256px로 줄인 결과가 Pillow 12.3.0으로 만든 기존 캐시와 모두 바이트 단위로 같았다. 자세한 내용은 [experiments.md](experiments.md) 공통 규칙의 '전처리' 절에 적었다
- **학습용 torch (같은 날, 절차 (4))**: OSV에 물으니 torch 2.11.0에 GHSA-rrmf-rvhw-rf47(CVE-2025-3000, `torch.jit.script`의 메모리 손상, 심각도 low, 수정 판 2.13.0) 1건이 있고 torchvision 0.26.0에는 없었다. 판단: 그대로 둔다. torch는 학습·내보내기 그룹에만 있고 서비스 이미지에는 없다. 이 저장소의 코드는 `torch.jit.script`를 부르지 않는다. anomalib이 끌어오는 kornia가 가져올 때 자기 함수를 스크립팅하고, ONNX 내보내기(`dynamo=False`)가 TorchScript 기반 추적을 쓰지만, 다루는 것은 설치한 라이브러리와 이 저장소의 코드뿐이고 외부에서 받은 TorchScript를 불러오지 않는다. 수정 판은 `pytorch-cu128` 인덱스에 없다(이 인덱스의 마지막 판은 torch 2.11.0·torchvision 0.26.0이고, torch 2.13.0 이상은 cu130·cu132 인덱스에 있다). 올릴 때의 절차는 `pyproject.toml`의 인덱스 주석에 적었다

### 2026-10-29 학습·개발 그룹: aiohttp 3.14.3 → 3.14.5, certifi 2026.7.22 → 2026.10.23, filelock 4.0.7 → 4.2.4
- **계기**: 서비스 집합은 위 절차 (1)·(2) 모두 `No known vulnerabilities found`였다. 잠금 전체(`--all-groups`)를 같은 방식으로 대조하니 pip-audit이 PyPI에서 27건(4개 패키지), OSV에서 28건(5개 패키지)을 냈다. aiohttp 3.14.3에 8건(수정 판 3.14.4, `dinomaly` 그룹의 lightning → fsspec[http]가 끌어옴), certifi 2026.7.22에 PYSEC-2026-4320(수정 판 2026.10.19, `dev` 그룹의 httpx·httpcore), filelock 4.0.7에 14건(CVE-2026-112907 등, 수정 판 4.2.1이고 한 건은 4.0.9, `train`·`dinomaly` 그룹의 torch·huggingface-hub), setuptools 81.0.0에 2건, 그리고 OSV에만 torch 2.11.0+cu128의 PYSEC-2025-194(위 절의 CVE-2025-3000) 1건이다. certifi와 setuptools의 PYSEC 권고는 pip-audit 출력에 같은 ID가 두 줄씩 나와 건수에 두 번 들어간다
- **조치**: `uv lock --upgrade-package aiohttp --upgrade-package certifi --upgrade-package filelock`로 세 패키지만 올렸다. 다른 패키지의 판과 잠금 형식(revision 5)은 바뀌지 않았다. 서비스 집합(기본 + `serve`)의 내보내기 목록은 갱신 전과 같아 이미지 내용은 바뀌지 않는다. 세 패키지 모두 학습·채점 수치를 계산하는 경로의 라이브러리가 아니다(내려받기·잠금 파일·인증서 묶음)
- **조치 뒤 대조**: 서비스 집합 (1)·(2) 0건. 잠금 전체는 PyPI 3건·OSV 4건으로, 남은 것은 아래 setuptools와 torch(위 절의 판단 그대로)뿐이다. 같은 날 CI docker 작업이 잠시 설치하는 onnx 집합(`.github/ci/requirements-onnx.txt`)도 (1)·(2) 방식으로 대조해 0건이었다(CI는 이 집합을 대조하지 않는다)
- **남긴 것: setuptools 81.0.0** (`train`·`dinomaly` 그룹, torch가 끌어옴)

| 권고 | 내용 | 이 저장소 | 근거 |
|---|---|---|---|
| PYSEC-2026-3447 (CVE-2026-59890) | MANIFEST.in의 제외 규칙이 macOS APFS·HFS+에서 NFD 파일 이름으로 우회되어 소스 배포에 파일이 섞임, 수정 판 83.0.0 | 닿지 않음 | 이 저장소는 setuptools로 소스 배포를 만들지 않는다(빌드 백엔드는 hatchling) |
| GHSA-xmj3-9gf7-h52w | Windows에서 옛 방식 wheel 설치(`Wheel.install_as_egg`)가 wheel 안의 `namespace_packages.txt` 때문에 대상 폴더 밖에 `__init__.py`를 만듦, 수정 판 84.1.0 | 닿지 않음 | wheel은 uv가 설치하고 setuptools의 옛 설치 경로를 쓰지 않는다 |

- **setuptools를 올리지 못하는 이유**: torch 2.11.0이 `setuptools<82`를 요구해 `uv lock --upgrade-package setuptools`가 잠금을 바꾸지 않는다. torch는 학습·채점 수치에 닿는 라이브러리라 이 갱신에서 올리지 않는다(올릴 때의 절차는 `pyproject.toml`의 인덱스 주석). setuptools는 서비스 이미지에 들어가지 않는다

### 2026-11-05 CI 대조 범위를 잠금 전체로 넓힘
- **계기**: CI audit 작업과 같은 명령(위 절차 (1)·(2))이 실제로 대조한 패키지를 pip-audit의 JSON 출력으로 세었다. `uv.lock`의 패키지는 프로젝트 자신을 빼고 130개다(이름 124개. 그 가운데 numpy, scipy, contourpy, imagecodecs, networkx, tifffile 6개는 Python 3.12 이상에서 쓰는 둘째 판이 따로 있다). CI가 대조한 것은 그 가운데 22개였다. 서비스 집합(기본 + `serve`)만 내보내는 데다, pip-audit이 요구 줄의 환경 표지를 자기를 돌리는 파이썬(3.11)으로 판단해 서비스 집합의 둘째 판(numpy 2.5.3, scipy 1.18.1)을 말없이 건너뛰었다. 학습·개발·그림·OpenVINO 그룹은 손으로 하는 대조에만 들어 있었다
- **조치**: CI audit 작업에 잠금 전체를 대조하는 단계를 더했다(`.github/workflows/ci.yml`의 `Audit every locked package`). 모든 그룹을 해시 없이 내보내 환경 표지와 torch·torchvision의 로컬 판 표시(`+cu128`)를 떼고, 같은 이름의 둘째 판은 다른 목록으로 나눠, 목록마다 PyPI와 OSV에 `--no-deps --disable-pip --strict`로 대조한다. 내보낸 줄 수가 잠금의 패키지 수와 다르면 멈춘다. 무시하는 것은 이 문서에 받아들인 이유를 적은 세 권고(torch 2.11.0의 PYSEC-2025-194, setuptools 81.0.0의 PYSEC-2026-3447·GHSA-xmj3-9gf7-h52w)뿐이고, 셋 다 학습 그룹에만 있어 서비스 이미지에는 없다. 서비스 집합을 해시와 함께 대조하는 기존 두 줄은 그대로 두었다(무시하는 권고 없음)
- **대조 결과 (2026-11-05, 같은 단계를 로컬 Git Bash에서 그대로 돌림)**: 130개(124 + 6)를 모두 대조했고 건너뛴 패키지는 없었다. PyPI와 OSV 모두 나온 것은 받아들인 세 권고의 네 줄뿐이다(PYSEC-2026-3447은 두 줄로 나온다). 서비스 집합의 (1)·(2)는 0건, Pillow 13.0.0을 GitHub Advisory Database에 물은 (3)은 두 표기 모두 0건, OSV API에 물은 (4)는 torch 2.11.0에 GHSA-rrmf-rvhw-rf47(= PYSEC-2025-194) 1건, torchvision 0.26.0에 0건이다. 일부러 권고가 있는 판(pillow 12.3.0)으로 바꾼 목록, 빈 내보내기, 실패한 내보내기, 한 줄이 빠진 내보내기에서는 이 단계가 실패했다
- **OSV에만 있는 권고**: 없다. 로컬 판 표시를 떼면 PyPI도 torch 2.11.0의 PYSEC-2025-194를 내므로, 지금까지 OSV 쪽에서만 보이던(위 절차 (4)) 이 권고가 두 데이터베이스에 함께 나온다. 판단은 위 2026-10-24 절의 torch 단락과 2026-10-29 절의 setuptools 표 그대로다(받아들이고, 고친 판으로 올리지 않는다)
- **2026-11-07 다시 확인**: 같은 단계를 로컬 Git Bash에서 무시 목록 없이 다시 돌렸다. 130개(124 + 6)를 모두 대조했고, PyPI와 OSV 모두 나온 것은 받아들인 세 권고의 네 줄(setuptools 81.0.0 PYSEC-2026-3447 두 줄·GHSA-xmj3-9gf7-h52w, torch 2.11.0 PYSEC-2025-194)뿐이며 둘째 판 목록 6개는 0건이다. OSV에만 있는 권고는 없고 판단은 그대로다
