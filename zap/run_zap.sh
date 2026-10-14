#!/usr/bin/env bash
# [파일 역할] OWASP ZAP 2.17.0 릴리스(공식 크로스플랫폼 패키지, Java 17 이상 필요, Docker 없이 Java로 직접
# 실행)로 zap/automation.yaml을 돌려, 로컬에 띄운 검사 서비스를 DAST 스캔하는 실행 스크립트. 스캔 내용(OpenAPI
# 경로, HAR 시드의 실제 업로드 요청, spider, passive·active scan)은 zap/automation.yaml에 있고, 이 스크립트는
# 환경 변수를 맞추고 ZAP을 띄우기만 한다.
#
#   ZAP_DIR=<zap-2.17.0.jar가 있는 폴더> ADMIN_TOKEN=<토큰> zap/run_zap.sh <이름> [시드 HAR]
#
# [사용 조건] 서비스가 그 관리자 토큰으로 TARGET(기본 http://127.0.0.1:8093)에서 이미 돌고 있어야 한다.
# 오프라인 앱을 띄우는 방법은 stackhawk.yml을 본다. 기본 시드는 hawk/seed.har(범주 demo)이고, 아티팩트
# 세트를 스캔하려면 그 세트의 범주 하나에 맞춘 시드가 필요하다(hawk/make_seed_har.py --category).
# 보고서 <이름>.json과 <이름>.md, ZAP 로그, 새 ZAP 홈 폴더는 work/zap/<이름>/에 생긴다. 출력은 파일로만
# 보내고 파이프로는 절대 넘기지 않는다.
# [왜 필요한가] HawkScan 체험이 2026-10-05에 끝나 이후 재스캔은 ZAP으로 하고, 같은 도구의 기록끼리
# 비교한다(docs/security.md). 관리자 토큰은 파일에 적지 않고 실행할 때 환경 변수로만 받는다.
# [흐름] 인자·환경 변수 검사 → 출력 폴더(work/zap/<이름>/) 확인 → 대상 Host에만 관리자 토큰 헤더를 붙이도록
# ZAP 환경 변수 설정 → 경로를 절대 경로로 바꿔 ZAP 실행(로그는 파일로) → 종료 코드 전달.
# set -euo pipefail: 명령 실패, 정의되지 않은 변수, 파이프 중간 실패에서 바로 멈춘다.

set -euo pipefail

# 첫 인자(스캔 이름)는 필수, 둘째(시드 HAR)는 없으면 오프라인 서비스용 hawk/seed.har(범주 demo). root는
# 스크립트 위치에서 구한 저장소 루트라 어느 폴더에서 불러도 같다. ZAP_DIR과 ADMIN_TOKEN이 없으면 사용법을
# 알리고 멈춘다(:? 확장).
name=${1:?usage: zap/run_zap.sh <name> [seed HAR]}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
seed=${2:-$root/hawk/seed.har}
: "${ZAP_DIR:?set ZAP_DIR to the folder that holds zap-<version>.jar}"
: "${ADMIN_TOKEN:?set ADMIN_TOKEN to the token the service was started with}"

# 경로를 Java가 읽는 절대 경로로 바꾼다. Git Bash에서는 cygpath -m으로 "D:/..." 같은 Windows 형식이 된다.
# ZAP은 상대 경로를 작업 폴더가 아니라 계획 파일의 폴더(zap/) 기준으로 풀기 때문에 절대 경로가 필요하다. 처음
# 실행 때 시드 HAR 경로를 상대 경로로 넘겨 ZAP이 zap/ 폴더 기준으로 찾다가 가져오기 단계에서 멈춘 일이 있어
# 고친 것이다(커밋 86c209e, docs/security.md).
native() {
  local abs
  abs=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
  if command -v cygpath > /dev/null; then cygpath -m "$abs"; else printf '%s\n' "$abs"; fi
}

# 결과 폴더가 이미 있으면 이전 스캔의 보고서를 덮어쓰지 않도록 멈춘다. ZAP jar와 시드 HAR가 실제로 있는지도
# 시작 전에 확인한다.
out=$root/work/zap/$name
if [ -e "$out" ]; then
  echo "$out exists: pick another name" >&2
  exit 1
fi
jars=("$ZAP_DIR"/zap-*.jar)
[ -f "${jars[0]}" ] || { echo "no zap-*.jar in $ZAP_DIR" >&2; exit 1; }
[ -f "$seed" ] || { echo "no seed HAR at $seed" >&2; exit 1; }
mkdir -p "$out"

# 대상은 기본으로 로컬 서비스(127.0.0.1:8093)다. host는 TARGET에서 스킴(http://)을 떼고, 아래에서 포트·경로를
# 떼어 호스트 이름만 남긴다.
export TARGET=${TARGET:-http://127.0.0.1:8093}
host=${TARGET#*://}
# ZAP_AUTH_HEADER* 세 변수를 주면 ZAP이 그 호스트로 가는 모든 요청에 관리자 토큰 헤더를 붙인다. 그래서 스캔이
# POST·DELETE /calibrate까지 닿는다. replacer 작업은 ${...}를 풀지 않아 토큰을 계획 파일 변수로 넣을 수
# 없었다(zap/automation.yaml 머리 주석). SITE를 대상 호스트로 한정해 다른 곳으로 토큰이 나가지 않게 한다.
export ZAP_AUTH_HEADER=X-Admin-Token ZAP_AUTH_HEADER_VALUE=$ADMIN_TOKEN ZAP_AUTH_HEADER_SITE=${host%%[:/]*}
# automation.yaml이 읽는 변수: 시드 HAR, 보고서 폴더와 파일 이름. 모두 절대 경로로 넘긴다.
export SEED_HAR REPORT_DIR REPORT_NAME=$name
SEED_HAR=$(native "$seed")
REPORT_DIR=$(native "$out")

# ZAP 실행: -cmd(창 없이 명령행 모드), -notel(원격 측정 끔), -dir(스캔마다 새 ZAP 홈 폴더, 이전 설정이 섞이지
# 않게), 업데이트 확인 끔, -autorun(자동화 계획 실행). JVM 힙은 2GB로 둔다. 출력은 파이프로 넘기지 않고
# work/zap/<이름>/zap.log 파일로만 보낸다(머리 주석).
# || status=$?로 ZAP이 실패해도 set -e에 걸려 바로 끝나지 않게 하고, 종료 코드를 알린 뒤 그대로 돌려준다.
status=0
java -Xmx2g -jar "$(native "${jars[0]}")" -cmd -notel -dir "$(native "$out/home")" \
  -config start.checkForUpdates=false -autorun "$(native "$root/zap/automation.yaml")" \
  > "$out/zap.log" 2>&1 || status=$?
echo "ZAP exit status $status, reports and log in $out"
exit "$status"
