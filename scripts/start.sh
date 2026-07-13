#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Arg parsing ───────────────────────────────────────────────────
# --no-manage: bring up Gazebo + BetaFlight SITL but skip drone_manage.py.
#   Useful for test_thrust.py or any tool that needs the sim running without
#   drone_manage fighting over RC UDP port 9004.
# --airframe=80mm: which drone model + config + world to load.
#   Default is 80mm (Pavo Pico II).
SKIP_MANAGE=0
AIRFRAME="80mm"
USE_XBOX=0
MANAGE_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-manage)
            SKIP_MANAGE=1
            ;;
        --airframe=*)
            AIRFRAME="${arg#--airframe=}"
            ;;
        --xbox)
            USE_XBOX=1
            MANAGE_ARGS+=("$arg")
            ;;
        *)
            MANAGE_ARGS+=("$arg")
            ;;
    esac
done

if [ "$AIRFRAME" != "80mm" ]; then
    echo "error: only 80mm airframe is currently supported (got: $AIRFRAME)" >&2
    exit 1
fi

MODEL_NAME="betaloop_drone_cam_${AIRFRAME}"
DRONE_CONFIG="drone_config_${AIRFRAME}.py"

# ── Gazebo environment ────────────────────────────────────────────
export PATH="/opt/homebrew/opt/ruby/bin:/opt/homebrew/bin:$PATH"
export GZ_IP=127.0.0.1

# World to load — defaults to Baylands for the default 80mm profile, otherwise
# the matching drone_course. Override with GZ_WORLD env var.
DEFAULT_WORLD="drone_course_${AIRFRAME}"
if [ "$AIRFRAME" = "80mm" ]; then
    DEFAULT_WORLD="baylands_80mm"
fi
GZ_WORLD="${GZ_WORLD:-$DEFAULT_WORLD}"
GZ_WORLD="${GZ_WORLD%.sdf}"  # strip .sdf if user included it
export GZ_WORLD

# Resource paths: project worlds + aeroloop_gazebo models (if present)
AEROLOOP_DIR="${AEROLOOP_GAZEBO_DIR:-$HOME/dev/aeroloop_gazebo}"
export GZ_SIM_RESOURCE_PATH="$PROJECT_DIR/worlds"
if [ -d "$AEROLOOP_DIR/models" ]; then
    export GZ_SIM_RESOURCE_PATH="$GZ_SIM_RESOURCE_PATH:$AEROLOOP_DIR/models"
fi
if [ -d "$AEROLOOP_DIR/worlds" ]; then
    export GZ_SIM_RESOURCE_PATH="$GZ_SIM_RESOURCE_PATH:$AEROLOOP_DIR/worlds"
fi

# BetaflightPlugin shared library path (built from aeroloop_gazebo)
if [ -d "$AEROLOOP_DIR/plugins/build" ]; then
    export GZ_SIM_SYSTEM_PLUGIN_PATH="$AEROLOOP_DIR/plugins/build"
fi

# BetaFlight SITL binary (override with BETAFLIGHT_SITL_BIN env var)
BETAFLIGHT_BIN="${BETAFLIGHT_SITL_BIN:-$HOME/dev/betaflight/obj/main/betaflight_SITL.elf}"

# ── Cleanup on exit ──────────────────────────────────────────────
stop_xbox_bridge() {
    pkill -f "XboxBridge.app|Contents/MacOS/XboxBridge" 2>/dev/null || true
    rm -f /tmp/nanopilot_xbox.sock 2>/dev/null || true
}

cleanup() {
    echo ""
    echo "Stopping all processes..."
    stop_xbox_bridge
    # Kill all background children of this script
    jobs -p | xargs -r kill -9 2>/dev/null
    # Fall back to pkill for any stragglers
    "$SCRIPT_DIR/stop_all.sh"
    echo "Done."
}
trap cleanup EXIT INT TERM

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

EXISTING_STACK="$(pgrep -fl 'betaflight_SITL|gz sim|ruby.*gz|gz_camera_worker.py|drone_manage.py drive' || true)"
if [ -n "$EXISTING_STACK" ]; then
    echo "ERROR: existing Nanopilot sim processes are still running:" >&2
    echo "$EXISTING_STACK" >&2
    echo "Run: bash $SCRIPT_DIR/stop_all.sh" >&2
    exit 1
fi

# ── Launch BetaFlight SITL (before Gazebo to avoid FDM-during-init crash) ──
BF_LOG="$LOG_DIR/betaflight.log"

# BetaFlight SITL runs best with existing eeprom.bin (deletion causes SIGTRAP on first run)

echo "Starting BetaFlight SITL: $BETAFLIGHT_BIN (log: $BF_LOG)..."

# Reusable launcher — called again after a CLI eeprom save so the running BF
# loads fresh values from disk and comes up with a clean arming-flag state.
start_betaflight() {
    # BetaFlight SITL on macOS ARM64 sometimes crashes on first launch (SIGTRAP), but works on retry
    for attempt in 1 2; do
        > "$BF_LOG"
        # Removed gstdbuf to avoid DYLD_INSERT_LIBRARIES SIGTRAP on Apple Silicon
        "$BETAFLIGHT_BIN" > "$BF_LOG" 2>&1 &
        BF_PID=$!
        sleep 2
        if kill -0 "$BF_PID" 2>/dev/null; then
            echo "BetaFlight SITL running (PID $BF_PID)."
            return 0
        fi
        if [ $attempt -eq 1 ]; then
            echo "BetaFlight crashed on attempt 1, retrying..."
            sleep 1
        else
            echo "ERROR: BetaFlight SITL died on both attempts. Check $BF_LOG"
            exit 1
        fi
    done
}
start_betaflight

# ── Configure BetaFlight via MSP ─────────────────────────────────
# Sets ARM on AUX1, ANGLE on AUX2, changes failsafe to AUTO_LANDING if
# needed. Queries arming status for diagnostics. If CLI config saved new
# eeprom values, the script exits 42 → we kill and relaunch BF so the new
# values load from disk, then re-run (the 2nd pass finds values already
# match and proceeds to mode/failsafe setup).
for cfg_attempt in 1 2; do
# `set +e` so the intentional exit 42 from the config script doesn't trip
# `set -e` before we can check the return code.
set +e
SKIP_CLI_PROBE="$([ "$cfg_attempt" = "2" ] && echo 1 || echo 0)" \
AIRFRAME="$AIRFRAME" python3 -c "
import socket, time, struct, sys, os
SKIP_CLI = os.environ.get('SKIP_CLI_PROBE') == '1'
AIRFRAME = os.environ.get('AIRFRAME', '65mm')

def msp_send(sock, cmd, payload=b''):
    size = len(payload)
    checksum = size ^ cmd
    for b in payload:
        checksum ^= b
    frame = b'\$M<' + bytes([size, cmd]) + payload + bytes([checksum])
    sock.send(frame)
    time.sleep(0.1)
    # Read response — may need multiple recv() calls
    resp = b''
    try:
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            resp += chunk
            # Check if we have a complete MSP frame
            if len(resp) >= 4 and resp[:2] == b'\$M':
                if len(resp) >= 5:
                    pkt_size = resp[3]
                    expected = 6 + pkt_size  # header(3) + size(1) + cmd(1) + payload + checksum(1)
                    if len(resp) >= expected:
                        break
    except socket.timeout:
        pass
    if len(resp) >= 5 and resp[:3] == b'\$M>':
        return resp[5:5+resp[3]]
    if len(resp) >= 5 and resp[:3] == b'\$M!':
        print('  MSP cmd %d: ERROR response' % cmd)
        return b''
    if resp:
        print('  MSP cmd %d: unexpected response (%d bytes): %s' % (cmd, len(resp), resp[:20].hex()))
    return b''

def connect_msp():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for _ in range(30):
        try:
            sock.connect(('127.0.0.1', 5761))
            break
        except (ConnectionRefusedError, OSError):
            time.sleep(0.3)
    else:
        raise RuntimeError('Could not reconnect to BetaFlight MSP after reboot')
    sock.settimeout(1)
    time.sleep(0.5)
    try:
        sock.recv(4096)
    except socket.timeout:
        pass
    return sock

s = connect_msp()

# Disable AIRMODE feature if enabled. AIRMODE re-scales the motor mix so
# yaw/attitude commands retain authority at low throttle — which on our
# 53.7g airframe causes the drone to climb on yaw input even at CH3=1000.
# Feature changes require eeprom save + reboot to take effect.
FEATURE_AIRMODE = 1 << 22
feat = msp_send(s, 36)  # MSP_FEATURE_CONFIG
if len(feat) >= 4:
    current = struct.unpack('<I', feat[:4])[0]
    if current & FEATURE_AIRMODE:
        new_feat = current & ~FEATURE_AIRMODE
        msp_send(s, 37, struct.pack('<I', new_feat))  # MSP_SET_FEATURE_CONFIG
        msp_send(s, 250)  # MSP_EEPROM_WRITE
        print('  Features: 0x%08x -> 0x%08x (AIRMODE disabled), saved to eeprom' % (current, new_feat))
        # MSP_REBOOT (68) — BF restarts internally, socket dies
        try:
            frame = b'\$M<' + bytes([0, 68, 68])
            s.send(frame)
        except OSError:
            pass
        s.close()
        time.sleep(1.5)
        s = connect_msp()
        print('  Reconnected after reboot')
    else:
        print('  Features: 0x%08x (AIRMODE already disabled)' % current)
else:
    print('  Features: MSP_FEATURE_CONFIG returned %d bytes — skipping airmode check' % len(feat))

# Set mixer_type = LINEAR via CLI. LEGACY (the default) clips motors at PWM
# limits when yaw PID saturates — two motors slam to 2000, two to 1000, and
# total thrust doubles → drone rockets up on any yaw input. LINEAR scales
# throttle down instead of clipping, preserving motor headroom.
def cli_drain(sock, timeout=0.5, match=None):
    # Read from sock until timeout; stop early if match bytes appear.
    sock.settimeout(timeout)
    buf = b''
    try:
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
            if match is not None and match in buf:
                break
    except socket.timeout:
        pass
    sock.settimeout(1)
    return buf

# Lower the yaw PID gains. BF's stock PIDs are tuned for 5\" racing quads;
# on our small whoop airframes the yaw loop saturates on any non-zero stick
# deflection, producing a step-function motor asymmetry that lifts the
# drone ~20m on every turn (see --mode=yaw-airborne test). Reducing P/I/D/F
# restores proportional response.
#
# The 65mm Air65 has lower mass and inertia than the 80mm Pavo Pico II, so
# keep the 65mm yaw gains especially low.
if AIRFRAME == '65mm':
    YAW_PID = {'p_yaw': '5', 'i_yaw': '2', 'd_yaw': '0', 'f_yaw': '5'}
else:
    YAW_PID = {'p_yaw': '20', 'i_yaw': '10', 'd_yaw': '0', 'f_yaw': '20'}

# CLI phase. Probing via CLI latches BF's CLI arming-disable flag on the
# running SITL — MSP_REBOOT doesn't reliably re-init the MSP service on this
# SITL build, so the only way to clear the flag is to kill and restart.
#   pass 1 (SKIP_CLI=0): probe, apply if needed, save noreboot, exit 42
#   pass 2 (SKIP_CLI=1): skip CLI entirely, proceed to mode/failsafe setup
if not SKIP_CLI:
    # Send '#' only (no CR/LF) — the byte itself is the mode-switch trigger;
    # a trailing newline gets interpreted as an empty command and can race
    # with the prompt being drawn.
    s.sendall(b'#')
    banner = cli_drain(s, timeout=1.5, match=b'# ')
    print('  CLI banner: %r' % banner[-80:])

    s.sendall(b'get mixer_type\r\n')
    mixer_resp = cli_drain(s, timeout=1.5, match=b'# ')
    # Match the full assignment line — response also lists allowed values
    # (LEGACY, LINEAR, ...) so a bare b-LINEAR-in-resp is a false positive.
    mixer_ok = b'mixer_type = LINEAR' in mixer_resp
    print('  probe mixer_type: %r' % mixer_resp[:160])

    yaw_ok = True
    for k, v in YAW_PID.items():
        s.sendall(('get ' + k + '\r\n').encode())
        r = cli_drain(s, timeout=1.5, match=b'# ')
        print('  probe %s: %r' % (k, r[:160]))
        if ('= ' + v).encode() not in r:
            yaw_ok = False

    # Mode bindings:
    #   slot 0: ARM   on AUX1 (channel idx 0), range [1700, 2100]
    #   slot 1: ANGLE on AUX2 (channel idx 1), range [1700, 2100]
    # Format: 'aux <slot> <modeId> <auxChannel> <start> <end> <link> <linkId>'
    # modeIds: 0=ARM, 1=ANGLE. Saving via CLI persists across BF restarts —
    # MSP_SET_MODE_RANGE alone only writes RAM, which used to leave runs
    # silently in Acro because the post-restart MSP path didn't always take.
    AUX_BINDINGS = [
        ('0', '0 0 0 1700 2100 0 0'),  # ARM on AUX1
        ('1', '1 1 1 1700 2100 0 0'),  # ANGLE on AUX2
    ]
    aux_ok = True
    s.sendall(b'aux\r\n')
    aux_resp = cli_drain(s, timeout=1.5, match=b'# ')
    print('  probe aux: %r' % aux_resp[:240])
    for slot, expected in AUX_BINDINGS:
        needle = ('aux ' + slot + ' ' + expected).encode()
        if needle not in aux_resp:
            aux_ok = False

    if mixer_ok and yaw_ok and aux_ok:
        print('  CLI: values already at target, no save needed')
    else:
        if not mixer_ok:
            s.sendall(b'set mixer_type = LINEAR\r\n')
            cli_drain(s, timeout=1.5, match=b'# ')
        if not yaw_ok:
            for k, v in YAW_PID.items():
                s.sendall(('set ' + k + ' = ' + v + '\r\n').encode())
                cli_drain(s, timeout=1.5, match=b'# ')
        if not aux_ok:
            for slot, expected in AUX_BINDINGS:
                s.sendall(('aux ' + slot + ' ' + expected + '\r\n').encode())
                cli_drain(s, timeout=1.5, match=b'# ')
        # 'save noreboot' persists to eeprom.bin WITHOUT exiting the SITL
        # process. (Bare 'save' issues a reset that kills SITL on macOS —
        # the CLI reboot path is exit(), not a warm restart.)
        s.sendall(b'save noreboot\r\n')
        cli_drain(s, timeout=2.0, match=b'# ')
        print('  CLI: applied mixer_type=LINEAR + yaw PIDs %s + aux bindings, saved noreboot' % YAW_PID)

    s.close()
    print('  Exiting to restart BetaFlight (clears CLI arming flag)')
    sys.exit(42)

# MSP_SET_MODE_RANGE (35): redundant fallback — bindings are now saved to
# eeprom in the CLI block above. Left here so a wiped eeprom still gets
# usable bindings for the current session even if the CLI save path failed.
msp_send(s, 35, struct.pack('BBBBB', 0, 0, 0, 32, 48))  # ARM on AUX1
msp_send(s, 35, struct.pack('BBBBB', 1, 1, 1, 32, 48))  # ANGLE on AUX2
print('  Modes set (fallback): ARM on AUX1, ANGLE on AUX2')

# Change failsafe from GPS_RESCUE(2) to AUTO_LANDING(0) if needed
fs = msp_send(s, 75)  # MSP_FAILSAFE_CONFIG
if len(fs) >= 8:
    procedure = fs[7]
    if procedure == 2:  # GPS_RESCUE
        msp_send(s, 76, fs[:7] + bytes([0]))  # procedure=AUTO_LANDING
        print('  Failsafe: GPS_RESCUE -> AUTO_LANDING')
    else:
        print('  Failsafe: procedure=%d (OK)' % procedure)

# Query arming status
time.sleep(0.3)  # Let BetaFlight process config changes
status = msp_send(s, 150)  # MSP_STATUS_EX
if len(status) >= 20:
    # Parse arming disable flags (variable offset due to flight mode flags)
    flags_byte_count_raw = status[15]
    flags_byte_count = flags_byte_count_raw & 0x0F
    offset = 16 + flags_byte_count
    if len(status) >= offset + 5:
        arming_count = status[offset]
        arming_flags = struct.unpack('<I', status[offset+1:offset+5])[0]
        flag_names = ['NO_GYRO','FAILSAFE','RX_FAILSAFE','NOT_DISARMED',
            'BOXFAILSAFE','RUNAWAY','CRASH','THROTTLE','ANGLE','BOOT_GRACE',
            'NOPREARM','LOAD','CALIBRATING','CLI','CMS_MENU','BST','MSP',
            'PARALYZE','GPS','RESC','DSHOT_TELEM','REBOOT_REQ','DSHOT_BB',
            'ACC_CAL','MOTOR_PROTO','CRASHFLIP','ALTHOLD','POSHOLD','ARM_SWITCH']
        active = [flag_names[i] for i in range(min(len(flag_names), 29)) if arming_flags & (1 << i)]
        print('  Arming flags: 0x%08x %s' % (arming_flags, active if active else '(none)'))
    else:
        print('  Status: payload too short for arming flags (got %d bytes)' % len(status))
else:
    print('  Status: MSP_STATUS_EX returned %d bytes' % len(status))

print('BetaFlight MSP configuration complete.')
s.close()
" 2>&1
    cfg_rc=$?
    set -e
    if [ "$cfg_rc" = "42" ]; then
        if [ "$cfg_attempt" = "1" ]; then
            echo "Restarting BetaFlight SITL to load saved eeprom..."
            kill -9 "$BF_PID" 2>/dev/null || true
            wait "$BF_PID" 2>/dev/null || true
            sleep 1
            start_betaflight
            continue
        else
            echo "ERROR: BetaFlight CLI configuration still dirty after restart." >&2
            exit 1
        fi
    elif [ "$cfg_rc" != "0" ]; then
        echo "ERROR: BetaFlight MSP config exited $cfg_rc" >&2
        exit 1
    fi
    break
done

# ── Launch Gazebo standalone ─────────────────────────────────────
GZ_LOG="$LOG_DIR/gazebo.log"

echo "Starting Gazebo with world: $GZ_WORLD (log: $GZ_LOG)..."
> "$GZ_LOG"
# macOS requires separate server (-s) and GUI (-g) processes for gz sim
gz sim -s -r "${GZ_WORLD}.sdf" >> "$GZ_LOG" 2>&1 &
GZ_SERVER_PID=$!
GZ_PID=$GZ_SERVER_PID

# Wait for Gazebo readiness — use world service response (immune to stale topics)
TIMEOUT=60
echo "Waiting for Gazebo readiness (timeout ${TIMEOUT}s)..."
elapsed=0
while true; do
    if ! kill -0 "$GZ_PID" 2>/dev/null; then
        echo "ERROR: Gazebo process died. Check $GZ_LOG"
        exit 1
    fi
    if [ "$elapsed" -ge "$TIMEOUT" ]; then
        echo "ERROR: Timed out waiting for Gazebo. Check $GZ_LOG"
        exit 1
    fi
    # Check if Gazebo world service is responding (immune to stale topics).
    # gz service returns exit 0 even on timeout, so check output for "timed out".
    SVC_OUT=$(gz service -s /world/${GZ_WORLD}/gui/info --reqtype gz.msgs.Empty \
         --reptype gz.msgs.GUI --timeout 500 --req "" 2>&1 || true)
    if ! echo "$SVC_OUT" | grep -qi "timed out"; then
        # Also verify camera topic exists
        if gz topic -l 2>/dev/null | grep -q "${MODEL_NAME}.*camera/image"; then
            break
        fi
    fi
    sleep 1
    elapsed=$((elapsed + 1))
done
echo "Gazebo server ready (after ${elapsed}s)."

# Launch Gazebo GUI (optional — skip with GZ_HEADLESS=1)
if [ "${GZ_HEADLESS:-0}" != "1" ]; then
    GZ_GUI_LOG="$LOG_DIR/gazebo_gui.log"
    gz sim -g >> "$GZ_GUI_LOG" 2>&1 &
    GZ_GUI_PID=$!
    echo "Gazebo GUI launched (PID $GZ_GUI_PID)."
fi

# ── Run drone_manage.py in foreground (unless --no-manage) ───────
cd "$PROJECT_DIR"
if [ "$SKIP_MANAGE" = "1" ]; then
    echo "STACK_READY"  # sentinel for wrapper scripts (e.g. test_thrust.sh)
    echo "Sim stack up. Press Ctrl+C to stop."
    # Park the script on `wait` so INT/TERM interrupt promptly and the EXIT
    # trap fires. `tail -f /dev/null` is a cheap way to get a backgrounded
    # child to wait on.
    tail -f /dev/null &
    wait $!
else
    # Launch the Xbox controller bridge (.app bundle) if --xbox was passed.
    # GameController.framework is the only macOS path that sees an Xbox
    # controller (Apple's XboxGamepad dext blocks pygame/SDL/hidapi), and it
    # only works from a real .app bundle. The bridge sends controller frames at
    # 60Hz to /tmp/nanopilot_xbox.sock; XboxDroneController binds it.
    if [ "$USE_XBOX" = "1" ]; then
        XBOX_APP="$PROJECT_DIR/xbox_bridge/build/XboxBridge.app"
        XBOX_BIN="$XBOX_APP/Contents/MacOS/XboxBridge"
        if [ ! -d "$XBOX_APP" ]; then
            echo "ERROR: $XBOX_APP not found." >&2
            echo "Build it with: bash $PROJECT_DIR/xbox_bridge/build.sh" >&2
            exit 1
        fi
        if [ "$PROJECT_DIR/xbox_bridge/main.swift" -nt "$XBOX_BIN" ]; then
            echo "XboxBridge source is newer than the app; rebuilding..."
            bash "$PROJECT_DIR/xbox_bridge/build.sh"
        fi
        # Kill any prior instance so it re-enumerates the controller cleanly.
        stop_xbox_bridge
        sleep 0.3
        echo "Starting XboxBridge.app (controller → UDS @ /tmp/nanopilot_xbox.sock)..."
        open "$XBOX_APP"
    fi

    echo "Starting drone_manage.py (--myconfig=$DRONE_CONFIG)..."
    uv run --env-file .env python -W ignore::SyntaxWarning \
        nanopilot/drone_manage.py drive --myconfig="$DRONE_CONFIG" ${MANAGE_ARGS[@]+"${MANAGE_ARGS[@]}"}
fi
