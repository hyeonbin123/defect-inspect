#!/usr/bin/env bash
# Run zap/automation.yaml with a ZAP release (cross-platform package, Java 17 or later) against a local service.
#
#   ZAP_DIR=<folder with zap-2.17.0.jar> ADMIN_TOKEN=<token> zap/run_zap.sh <name> [seed HAR]
#
# The service must already be running on TARGET (default http://127.0.0.1:8093) with that admin token; see
# stackhawk.yml for the offline app. The default seed is hawk/seed.har (category demo); an artifact set needs a
# seed for one of its categories (hawk/make_seed_har.py --category). The reports <name>.json and <name>.md,
# the ZAP log and a fresh ZAP home folder go to work/zap/<name>/. Output goes to files only, never to a pipe.
set -euo pipefail

name=${1:?usage: zap/run_zap.sh <name> [seed HAR]}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
seed=${2:-$root/hawk/seed.har}
: "${ZAP_DIR:?set ZAP_DIR to the folder that holds zap-<version>.jar}"
: "${ADMIN_TOKEN:?set ADMIN_TOKEN to the token the service was started with}"

native() { # the absolute path in a form Java understands (the Windows form under Git Bash). ZAP resolves a
  # relative path against the plan's folder (zap/), not the working directory.
  local abs
  abs=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
  if command -v cygpath > /dev/null; then cygpath -m "$abs"; else printf '%s\n' "$abs"; fi
}

out=$root/work/zap/$name
if [ -e "$out" ]; then
  echo "$out exists: pick another name" >&2
  exit 1
fi
jars=("$ZAP_DIR"/zap-*.jar)
[ -f "${jars[0]}" ] || { echo "no zap-*.jar in $ZAP_DIR" >&2; exit 1; }
[ -f "$seed" ] || { echo "no seed HAR at $seed" >&2; exit 1; }
mkdir -p "$out"

export TARGET=${TARGET:-http://127.0.0.1:8093}
host=${TARGET#*://}
# ZAP adds this header to every request for the target host (see zap/automation.yaml).
export ZAP_AUTH_HEADER=X-Admin-Token ZAP_AUTH_HEADER_VALUE=$ADMIN_TOKEN ZAP_AUTH_HEADER_SITE=${host%%[:/]*}
export SEED_HAR REPORT_DIR REPORT_NAME=$name
SEED_HAR=$(native "$seed")
REPORT_DIR=$(native "$out")

status=0
java -Xmx2g -jar "$(native "${jars[0]}")" -cmd -notel -dir "$(native "$out/home")" \
  -config start.checkForUpdates=false -autorun "$(native "$root/zap/automation.yaml")" \
  > "$out/zap.log" 2>&1 || status=$?
echo "ZAP exit status $status, reports and log in $out"
exit "$status"
