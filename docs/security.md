# 보안 점검 기록

검사 서비스(`src/defect_inspect/service.py`)를 동적 스캔(DAST)으로 점검한 기록이다. 스캔은 이 PC에서 띄운 로컬 서비스(`127.0.0.1:8093`)에만 한다.
서비스 이미지에 들어가는 의존성을 공개 취약점 데이터베이스와 대조한 기록은 DAST와 따로 맨 아래 [의존성 취약점 대조](#의존성-취약점-대조) 절에 적는다(2026-10-24부터).

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
