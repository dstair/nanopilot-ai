#!/bin/bash
# Start the sim stack, fly a scripted collection pass, then train on the new tub.
#
# Usage:
#   ./scripts/collect_train.sh [--airframe=65mm|80mm] [--duration=30] [--model=models/scripted.pth] [--max-epochs=5]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

AIRFRAME="80mm"
DURATION="30"
WARMUP="8"
RATE_HZ="30"
MODEL_PATH="models/scripted_autonomous.pth"
MAX_EPOCHS=""

for arg in "$@"; do
    case "$arg" in
        --airframe=*) AIRFRAME="${arg#--airframe=}" ;;
        --duration=*) DURATION="${arg#--duration=}" ;;
        --warmup=*) WARMUP="${arg#--warmup=}" ;;
        --rate-hz=*) RATE_HZ="${arg#--rate-hz=}" ;;
        --model=*) MODEL_PATH="${arg#--model=}" ;;
        --max-epochs=*) MAX_EPOCHS="${arg#--max-epochs=}" ;;
        *)
            echo "error: unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

if [ "$AIRFRAME" != "65mm" ] && [ "$AIRFRAME" != "80mm" ]; then
    echo "error: --airframe must be 65mm or 80mm (got: $AIRFRAME)" >&2
    exit 1
fi

if [ -z "${GZ_WORLD:-}" ]; then
    GZ_WORLD="drone_course_${AIRFRAME}"
    if [ "$AIRFRAME" = "80mm" ]; then
        GZ_WORLD="baylands_80mm"
    fi
    export GZ_WORLD
fi

STACK_LOG="$LOG_DIR/collect_train_stack.log"
COLLECT_LOG="$LOG_DIR/collect_train_collect.log"
TRAIN_LOG="$LOG_DIR/collect_train_train.log"
> "$STACK_LOG"
> "$COLLECT_LOG"
> "$TRAIN_LOG"

echo "Bringing up sim stack (airframe=$AIRFRAME)..."
GZ_HEADLESS=1 "$SCRIPT_DIR/start.sh" --no-manage "--airframe=$AIRFRAME" > "$STACK_LOG" 2>&1 &
STACK_PID=$!

cleanup() {
    echo ""
    echo "Stopping sim stack..."
    kill -TERM "$STACK_PID" 2>/dev/null || true
    bash "$SCRIPT_DIR/stop_all.sh" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

TIMEOUT=90
echo "Waiting for stack readiness (timeout ${TIMEOUT}s)..."
elapsed=0
while true; do
    if ! kill -0 "$STACK_PID" 2>/dev/null; then
        echo "ERROR: start.sh exited before ready. Log tail:"
        tail -40 "$STACK_LOG"
        exit 1
    fi
    if grep -q STACK_READY "$STACK_LOG" 2>/dev/null; then
        echo "Stack ready after ${elapsed}s."
        break
    fi
    if [ "$elapsed" -ge "$TIMEOUT" ]; then
        echo "ERROR: timed out waiting for stack. Log tail:"
        tail -40 "$STACK_LOG"
        exit 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
done

echo "Collecting scripted flight data..."
cd "$PROJECT_DIR"
uv run --env-file .env python nanopilot/autonomous_collect.py \
    "--airframe=$AIRFRAME" \
    "--duration=$DURATION" \
    "--warmup=$WARMUP" \
    "--rate-hz=$RATE_HZ" | tee "$COLLECT_LOG"

TUB_PATH="$(grep '^TUB_PATH=' "$COLLECT_LOG" | tail -1 | cut -d= -f2-)"
if [ -z "$TUB_PATH" ] || [ ! -d "$TUB_PATH" ]; then
    echo "ERROR: collection did not produce a tub path. Log tail:"
    tail -40 "$COLLECT_LOG"
    exit 1
fi

echo "Training on $TUB_PATH..."
TRAIN_ARGS=(
    "--tubs=$TUB_PATH"
    "--model=$MODEL_PATH"
    "--myconfig=drone_config_${AIRFRAME}.py"
)
if [ -n "$MAX_EPOCHS" ]; then
    TRAIN_ARGS+=("--max-epochs=$MAX_EPOCHS")
fi

uv run --env-file .env python -u nanopilot/torch_train.py "${TRAIN_ARGS[@]}" | tee "$TRAIN_LOG"

echo "Collected tub: $TUB_PATH"
echo "Model: $MODEL_PATH"
