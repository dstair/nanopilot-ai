#!/bin/bash
# One-command RL training: bring up Gazebo + BetaFlight SITL headless
# (no drone_manage), run nanopilot/rl_train.py, then tear down.
#
# Usage:
#     ./scripts/rl_train.sh [--airframe=65mm|80mm] [--world=<gz world>] \
#         [rl_train.py args...]
#
# Examples:
#     # Bounded smoke run (~1 min of flying)
#     ./scripts/rl_train.sh --total-steps=1000
#     # Overnight teacher training
#     ./scripts/rl_train.sh --total-steps=400000 --out=models/rl/ppo_gates_v1
#
# --airframe is forwarded to start.sh and rl_train.py. --world sets GZ_WORLD
# (default: slalom_gates_80mm). All other args are forwarded to rl_train.py.
# The sim runs in real time: budget ~70k steps/hour at the default 20Hz.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

AIRFRAME="80mm"
TRAIN_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --airframe=*)
            AIRFRAME="${arg#--airframe=}"
            ;;
        --world=*)
            export GZ_WORLD="${arg#--world=}"
            ;;
        *)
            TRAIN_ARGS+=("$arg")
            ;;
    esac
done

if [ -z "${GZ_WORLD:-}" ]; then
    export GZ_WORLD="slalom_gates_80mm"
fi

# Where checkpoints land, so a retry can resume instead of starting over.
OUT_DIR="models/rl/ppo_gates"
for arg in "${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}"; do
    case "$arg" in
        --out=*) OUT_DIR="${arg#--out=}" ;;
    esac
done

STACK_LOG="$LOG_DIR/start_no_manage.log"
STACK_PID=""

cleanup() {
    echo ""
    echo "Stopping sim stack..."
    [ -n "$STACK_PID" ] && kill -TERM "$STACK_PID" 2>/dev/null || true
    bash "$SCRIPT_DIR/stop_all.sh" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# BetaFlight SITL aborts at launch or drops its MSP socket at random on macOS
# ARM64, and can die mid-run. For a multi-hour unattended run that must not be
# fatal: rebuild the stack and resume from the last checkpoint instead.
MAX_ATTEMPTS="${RL_MAX_ATTEMPTS:-40}"
attempt=1
cd "$PROJECT_DIR"

while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    echo ""
    echo "=== Attempt $attempt/$MAX_ATTEMPTS ==="
    bash "$SCRIPT_DIR/stop_all.sh" >/dev/null 2>&1 || true
    sleep 2
    > "$STACK_LOG"

    echo "Bringing up sim stack (airframe=$AIRFRAME, world=$GZ_WORLD)..."
    GZ_HEADLESS=1 "$SCRIPT_DIR/start.sh" --no-manage "--airframe=$AIRFRAME" > "$STACK_LOG" 2>&1 &
    STACK_PID=$!

    TIMEOUT=90
    elapsed=0
    stack_ok=1
    while true; do
        if ! kill -0 "$STACK_PID" 2>/dev/null; then
            echo "start.sh exited before ready (BetaFlight flakiness?). Log tail:"
            tail -15 "$STACK_LOG"
            stack_ok=0
            break
        fi
        if grep -q STACK_READY "$STACK_LOG" 2>/dev/null; then
            echo "Stack ready after ${elapsed}s."
            break
        fi
        if [ "$elapsed" -ge "$TIMEOUT" ]; then
            echo "Timed out waiting for stack. Log tail:"
            tail -15 "$STACK_LOG"
            stack_ok=0
            break
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    if [ "$stack_ok" -eq 1 ]; then
        # Resume whenever a checkpoint exists. NOTE: on resume --total-steps
        # means that many ADDITIONAL steps (SB3 semantics), so an interrupted
        # run keeps going rather than restarting its budget.
        #
        # rl_train.py writes model.zip only when model.learn() RETURNS (clean
        # finish or KeyboardInterrupt). A run killed by a signal leaves progress
        # only in checkpoints/, so resuming on model.zip alone would silently
        # start from step 0 and discard the whole run. Promote the newest
        # checkpoint first if it is ahead of model.zip.
        if [ -d "$OUT_DIR/checkpoints" ]; then
            python3 "$SCRIPT_DIR/promote_checkpoint.py" --once \
                --out-dir="$OUT_DIR" || true
        fi

        RESUME_ARG=()
        if [ -f "$OUT_DIR/model.zip" ]; then
            echo "Resuming from $OUT_DIR"
            RESUME_ARG=("--resume=$OUT_DIR")
        elif compgen -G "$OUT_DIR/checkpoints/*.zip" >/dev/null 2>&1; then
            # Checkpoints exist but none could be promoted (missing vecnormalize
            # pkl, unreadable dir). Starting now would throw the run away, so
            # refuse and let a human decide. Retrying cannot help -- exit hard.
            echo "ERROR: $OUT_DIR/checkpoints/ holds checkpoints but there is no" >&2
            echo "  resumable $OUT_DIR/model.zip, and it could not be promoted." >&2
            echo "  Refusing to start from step 0 and discard that progress." >&2
            echo "  Inspect, then promote one by hand:" >&2
            echo "    python3 scripts/promote_checkpoint.py --once --out-dir=$OUT_DIR" >&2
            exit 2
        fi

        echo "Running rl_train.py..."
        # -u: SB3's progress tables must not sit in a block buffer.
        if uv run --env-file .env python -u nanopilot/rl_train.py \
                "--airframe=$AIRFRAME" "--world=$GZ_WORLD" \
                ${RESUME_ARG[@]+"${RESUME_ARG[@]}"} \
                ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}; then
            echo "Training finished successfully."
            exit 0
        fi
        echo "Trainer exited nonzero; will rebuild the stack and resume."
    fi

    kill -TERM "$STACK_PID" 2>/dev/null || true
    STACK_PID=""
    attempt=$((attempt + 1))
    sleep 5
done

echo "ERROR: gave up after $MAX_ATTEMPTS attempts."
exit 1
