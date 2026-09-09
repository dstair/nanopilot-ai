#!/bin/bash
# Bring the sim stack up and REFUSE to hand it over until BetaFlight answers MSP.
#
# Why this exists: BetaFlight SITL comes up MSP-deaf often enough to matter. It
# still accepts TCP on 5761 and still prints "MSP configuration complete", but
# every request then times out, and a board in that state never arms -- so every
# episode of whatever you launched next dies with "BetaFlight lost arming". The
# cost asymmetry is the whole point: the check takes ~30s and the collection it
# guards takes ~25 minutes.
#
# Symptoms seen in start.sh's own log when the board came up deaf:
#     Features: MSP_FEATURE_CONFIG returned 0 bytes — skipping airmode check
#     Status: MSP_STATUS_EX returned 0 bytes
# ...but a board can ALSO pass that phase and go deaf before you use it, which
# is why the gate runs against the finished stack rather than trusting the log.
#
#   bash scripts/start_gated.sh --no-manage --airframe=80mm
#   GZ_WORLD=s_curve_heights_80mm GZ_HEADLESS=1 bash scripts/start_gated.sh --no-manage
#
# Exits 0 with the stack UP and running in the background once the gate passes;
# exit 1 if every attempt failed. Env: STACK_ATTEMPTS (default 5),
# STACK_SETTLE_S (default 25), STACK_LOG (default logs/start_gated.log).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

ATTEMPTS="${STACK_ATTEMPTS:-5}"
# The settle is not superstition: _wait_for_ready's IMU test is
# `any(|imu| > 1e-6)` and a motionless drone reads all-zero, so a collector
# started the instant STACK_READY appears can stall 40+s on imu=False.
SETTLE="${STACK_SETTLE_S:-25}"
LOG="${STACK_LOG:-$PROJECT_DIR/logs/start_gated.log}"
mkdir -p "$(dirname "$LOG")"

msp_ok() {
    uv run --env-file .env python - <<'PY'
import sys
sys.path.insert(0, "nanopilot")
from utils.betaflight import MSPClient
status = MSPClient().query_status(timeout=2.0)
# A deaf board returns (None, [], None, []): connected, answered nothing.
print(status)
sys.exit(0 if status[0] is not None else 1)
PY
}

# Wait for a genuinely clean slate. stop_all.sh returns before the machine is
# quiet: start.sh calls stop_all.sh from its own EXIT trap, so a dying start.sh
# lands a SIGKILL on BetaFlight and Gazebo some seconds AFTER we asked it to
# stop. Launch the next attempt into that window and the previous attempt's trap
# kills the stack you just built -- the log then reads "STACK_READY" immediately
# followed by "Killed: 9", which looks like a launch failure and is not one.
wait_quiet() {
    for _ in $(seq 1 30); do
        pgrep -f 'betaflight_SITL' >/dev/null 2>&1 && { sleep 1; continue; }
        pgrep -f 'gz sim -s' >/dev/null 2>&1 && { sleep 1; continue; }
        pgrep -f 'scripts/start\.sh' >/dev/null 2>&1 && { sleep 1; continue; }
        return 0
    done
    echo "  warning: processes still alive after stop_all"
    return 1
}

for attempt in $(seq 1 "$ATTEMPTS"); do
    echo "=== stack attempt $attempt/$ATTEMPTS ==="
    bash "$SCRIPT_DIR/stop_all.sh" >/dev/null 2>&1
    wait_quiet

    : > "$LOG"
    bash "$SCRIPT_DIR/start.sh" "$@" >"$LOG" 2>&1 &
    start_pid=$!

    ready=0
    for _ in $(seq 1 120); do
        grep -q STACK_READY "$LOG" 2>/dev/null && { ready=1; break; }
        kill -0 "$start_pid" 2>/dev/null || break
        sleep 1
    done
    if [ "$ready" -ne 1 ]; then
        echo "  start.sh never reached STACK_READY; see $LOG"
        tail -5 "$LOG" | sed 's/^/    /'
        continue
    fi

    sleep "$SETTLE"
    if msp_ok; then
        echo "=== stack UP and MSP healthy (attempt $attempt) ==="
        exit 0
    fi
    echo "  MSP gate FAILED — board is deaf, rebuilding"
done

echo "=== stack failed the MSP gate $ATTEMPTS times; not handing it over ==="
bash "$SCRIPT_DIR/stop_all.sh" >/dev/null 2>&1
exit 1
