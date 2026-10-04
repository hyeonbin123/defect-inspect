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
