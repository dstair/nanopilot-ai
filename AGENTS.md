# AGENT.md

This file provides guidance to coding agents working in this repository. Start here in a fresh session, then read the newest "Current Handoff" section at the bottom before editing or running long commands.

## Project Summary

Nanopilot adapts the DonkeyCar pipeline to fly a simulated quadrotor drone using a CNN trained from camera images. Workflow: manually fly → record data → train CNN → fly autonomously. Runs on BetaFlight SITL + Gazebo Harmonic, native Apple Silicon (ARM64).

**Semantic mapping** (identical key names to DonkeyCar, BetaFlight Angle mode):
- `steering [-1, 1]` = yaw rate
- `throttle [-1, 1]` = forward pitch (tilt angle)
- `roll [-1, 1]` = lateral roll (tilt angle)
- `altitude [-1, 1]` = motor throttle (direct power, no PID)

## Commands

```bash
# Install dependencies
uv sync

# Launch (manual drive, default airframe = 80mm Pavo Pico II)
./scripts/start.sh
# Launch the 65mm Air65 profile instead
./scripts/start.sh --airframe=65mm

# Launch (autopilot)
./scripts/start.sh --model=models/pilot.pth

# Run without start.sh (BetaFlight SITL + Gazebo must already be running)
uv run --env-file .env python -W ignore::SyntaxWarning nanopilot/drone_manage.py drive --myconfig=drone_config_65mm.py

# Bring up the sim stack only (no drone_manage) — useful for tools that talk to
# BetaFlight's RC port directly, like test_thrust.py
./scripts/start.sh --no-manage --airframe=65mm

# One-command thrust/hover test (starts stack, runs test, tears down)
./scripts/test_thrust.sh --airframe=65mm

# Scripted no-human data collection + training smoke/test run
./scripts/collect_train.sh --airframe=65mm --duration=30 --max-epochs=5 --model=models/scripted_autonomous.pth

# Train CNN (full/default run; can take a while)
uv run --env-file .env python nanopilot/torch_train.py --tubs=data/tub_NN_YY-MM-DD --model=models/pilot.pth

# Multiple tubs (comma-separated)
uv run --env-file .env python nanopilot/torch_train.py --tubs=data/tub_1_26-03-01,data/tub_2_26-03-01 --model=models/pilot.pth

# Bounded hello-world training run (known to finish quickly on Apple Silicon)
uv run --env-file .env python nanopilot/torch_train.py \
  --tubs=data/tub_3_26-05-09 \
  --model=models/hello_world_tub3.pth \
  --max-epochs=1 \
  --max-samples=400 \
  --batch-size=64 \
  --no-model-summary

# Evaluate one model on a held-out tub
uv run --env-file .env python nanopilot/evaluate.py \
  --tubs=data/tub_4_26-05-09 \
  --model=models/hello_world_tub3.pth

# Compare two models; if --tubs is omitted, evaluate.py uses its default benchmark tub
uv run --env-file .env python nanopilot/evaluate.py \
  --old-model=models/tub_3_26-05-09.pth \
  --new-model=models/hello_world_tub3.pth

# Stop everything
bash ./scripts/stop_all.sh

# Force kill
pkill -9 -f betaflight_SITL; pkill -9 -f "gz sim"; pkill -9 -f "ruby.*gz"

# Verify gz-python works
uv run --env-file .env python -c "import gz.transport13; print('OK')"
```

Web UI: http://127.0.0.1:8887
- Static files served from `~/dev/donkeycar-fork/donkeycar/parts/web_controller/templates/static/` (see `web.py:112`). Edit `main.js` there, not the copy in `nanopilot/templates/static/`.

## Fresh Session Checklist

1. Read the newest `## Current Handoff (...)` section near the bottom.
2. Check local changes before editing:
   ```bash
   git status --short --branch
   git diff --stat
   ```
3. Avoid full/default training unless explicitly requested. Use bounded flags (`--max-epochs`, `--max-samples`, `--batch-size`, `--no-model-summary`) for smoke runs.
4. Use `bash ./scripts/stop_all.sh` before and after sim tests if Gazebo/BetaFlight/XboxBridge might already be running.
5. Do not edit `nanopilot/config.py` for local tuning; edit `nanopilot/drone_config_65mm.py` or `nanopilot/drone_config_80mm.py`.

## Verification / Tests

There is no formal pytest suite. Use these targeted checks:

```bash
# Import/compile check for the main Python modules
uv run --env-file .env python -m py_compile \
  nanopilot/drone_gym.py nanopilot/drone_env.py nanopilot/gz_telemetry.py \
  nanopilot/tub_schema.py nanopilot/drone_manage.py \
  nanopilot/autonomous_collect.py nanopilot/dataset.py \
  nanopilot/torch_model.py nanopilot/torch_pilot.py \
  nanopilot/torch_train.py nanopilot/evaluate.py nanopilot/smoke_test.py

# Script CLI/import smoke checks
uv run --env-file .env python nanopilot/torch_train.py --help
uv run --env-file .env python nanopilot/evaluate.py --help
uv run --env-file .env python nanopilot/autonomous_collect.py --help

# Model load/inference smoke for a .pth checkpoint
uv run --env-file .env python - <<'PY'
import numpy as np
from nanopilot.torch_pilot import TorchPilot
pilot = TorchPilot(input_shape=(3, 240, 320), seq_len=3)
pilot.load('models/hello_world_tub3.pth')
print(tuple(round(float(v), 6) for v in pilot.run(np.zeros((240, 320, 3), dtype=np.uint8))))
PY

# Gazebo/BetaFlight hover/thrust integration test
./scripts/test_thrust.sh --airframe=65mm
```

For headless scripted collection:

```bash
GZ_HEADLESS=1 ./scripts/start.sh --no-manage --airframe=65mm
uv run --env-file .env python nanopilot/autonomous_collect.py \
  --airframe=65mm --duration=3 --warmup=1 --rate-hz=5 --ready-timeout=30
bash ./scripts/stop_all.sh
```

## Python Environment

- **Python 3.12 exactly** (`requires-python = "==3.12.*"`)
- Package manager: `uv` (not pip)
- `.env` file at project root is required (sets PYTHONPATH, DYLD_LIBRARY_PATH, GZ_IP)
- gz-python is NOT on PyPI — installed by `brew install gz-harmonic` into Homebrew site-packages
- No formal pytest suite or linting/formatting configuration; use the targeted checks above.

## Architecture

```
Web Browser (http://127.0.0.1:8887)
    ↓
LocalWebController (tornado) → user/steering, user/throttle, user/roll, user/altitude, user/mode
    ↓
DriveMode (selects user vs autopilot)
    ↓
DroneGymEnv (threaded DonkeyCar part)
  ├── update(): background thread with BetaFlight RC loop (50Hz UDP)
  │     → BetaFlight SITL (UDP 9004): RC channel packets (arm, roll, pitch, yaw, throttle)
  │     → Direct throttle control (no PID), Angle mode stabilization by BetaFlight
  ├── gz_camera_worker.py: separate subprocess for camera frames
  │     → subscribes to gz-transport topic
  │     → writes frames to POSIX shared memory (1-byte seq + RGB pixels)
  └── run_threaded(): reads shared memory, applies simulated delay, returns cam/image_array
    ↓ (autopilot mode)
TorchPilot (LinearModel CNN inference)
    ↓
TubWriter (records image + yaw/pitch/roll/altitude controls to data/)
```

### BetaFlight SITL Protocol

| Port | Direction | Purpose |
|------|-----------|---------|
| 9002 | SITL → Gazebo | Motor speeds [0-1] (Gazebo plugin) |
| 9003 | Gazebo → SITL | FDM state: IMU, position, orientation |
| **9004** | **Us → SITL** | **RC channels (16x uint16, 1000-2000 μs PWM)** |
| 5761 | TCP | Configurator (setup only) |

RC packet: `struct.pack('<d', timestamp)` + 16 × `struct.pack('<H', channel)` = 40 bytes

### Key design decisions

- **gz_camera_worker runs as a subprocess** (not thread) to avoid libprotobuf version conflicts between gz-python and TensorFlow/PyTorch
- **Shared memory IPC**: parent creates POSIX SharedMemory, worker writes frames with a sequence counter, parent polls counter in `run_threaded()` for zero-copy reads
- **Direct throttle**: no altitude PID — `altitude [-1,1]` maps to motor power, matching real BetaFlight Angle mode behavior for CNN transferability
- **Config system** (DonkeyCar pattern): `dk.load_config(config_path='config.py', myconfig='drone_config_65mm.py')` (or `_80mm`) — edit the airframe-specific file, never `config.py`

## Key Files

| File | Purpose |
|------|---------|
| `nanopilot/drone_manage.py` | Main entry point |
| `nanopilot/drone_gym.py` | DroneGymEnv: BetaFlight RC UDP + camera bridge |
| `nanopilot/drone_env.py` | Shared builder for constructing `DroneGymEnv` from config |
| `nanopilot/gz_telemetry.py` | Gazebo pose/IMU telemetry helpers |
| `nanopilot/drone_config_80mm.py` | Pavo Pico II O4 (80mm, 79g with selected battery) config — default airframe |
| `nanopilot/drone_config_65mm.py` | Air65 (65mm, ~31g AUW) config — alternate airframe |
| `nanopilot/config.py` | Base DonkeyCar config (**do not modify**) |
| `nanopilot/gz_camera_worker.py` | Subprocess: gz-transport camera → shared memory |
| `nanopilot/tub_schema.py` | Shared tub input/type schema and IMU key list |
| `nanopilot/dataset.py` | PyTorch tub dataset loader for image + IMU + previous controls |
| `nanopilot/torch_model.py` | Current PyTorch `LinearModel` architecture |
| `nanopilot/torch_pilot.py` | Inference wrapper for vehicle loop |
| `nanopilot/torch_train.py` | Training script |
| `nanopilot/evaluate.py` | Offline checkpoint evaluator and model comparison CLI |
| `nanopilot/autonomous_collect.py` | Headless scripted data collector |
| `nanopilot/smoke_test.py` | Lightweight local smoke checks |
| `scripts/start.sh` | One-command launcher. Flags: `--airframe=65mm\|80mm` (default 80mm), `--no-manage` (sim stack only). |
| `scripts/stop_all.sh` | Force-kill all processes |
| `scripts/stop-all.sh` | Compatibility wrapper around `scripts/stop_all.sh` |
| `scripts/collect_train.sh` | Starts sim stack, runs scripted collection, then trains |
| `scripts/test_thrust.sh` | Wrapper: `start.sh --no-manage --airframe=$X` + `test_thrust.py` + teardown |
| `nanopilot/test_thrust.py` | Ramps throttle 1000→2000 PWM and reports altitude at each step; used for tuning `motorConstant` and finding hover PWM |
| `data/` | Recorded DonkeyCar tub folders (`tub_N_YY-MM-DD`) |
| `models/` | Saved `.pth` checkpoints and baseline pilot models |
| `worlds/drone_course_65mm.sdf` | 65mm Air65 world (includes `betaloop_drone_cam_65mm`) |
| `worlds/drone_course_80mm.sdf` | 80mm Pavo Pico II world (includes `betaloop_drone_cam_80mm`) |

### External Files (outside this repo)

| File | Purpose |
|------|---------|
| `~/dev/aeroloop_gazebo/` | BetaFlight-Gazebo bridge plugin repo (gz branch) |
| `~/dev/aeroloop_gazebo/plugins/BetaflightPlugin.cc` | Bridge plugin source: UDP 9002/9003 between BetaFlight ↔ Gazebo |
| `~/dev/aeroloop_gazebo/plugins/build/libBetaflightPlugin.dylib` | Compiled plugin loaded by Gazebo at runtime |
| `~/dev/aeroloop_gazebo/models/betaloop_drone_cam_65mm/` | 65mm Air65 quadrotor model (4 rotors + LiftDrag + IMU + forward camera) |
| `~/dev/aeroloop_gazebo/models/betaloop_drone_cam_80mm/` | 80mm Pavo Pico II O4 quadrotor model |
| `~/dev/betaflight/` | BetaFlight firmware source (SITL target) |
| `~/dev/betaflight/obj/main/betaflight_SITL.elf` | Compiled BetaFlight SITL binary |
| `~/.gz/sim/8/server.config` | Gazebo default server plugins (Physics, UserCommands, SceneBroadcaster) |
| `~/dev/donkeycar-fork/donkeycar/parts/web_controller/templates/static/main.js` | Web UI JS served at runtime — edit this file, NOT `nanopilot/templates/static/main.js` |

## Camera Modes

Controlled by `DRONE_CAMERA_SOURCE` in your `nanopilot/drone_config_XXmm.py`:
- `"gz_transport"` (default): native macOS, Gazebo Harmonic, gz-python bindings
- `"rtsp"`: Docker legacy mode, Gazebo Classic, OpenCV VideoCapture

## Multi-modal Model (LinearModel)

- Current model in `nanopilot/torch_model.py` is an Attention-fused Residual CNN-GRU. It uses GELU activations for improved gradient flow.
- Inputs: image `(B, 3, H, W)` float32 `[0,1]`, IMU sequence `(B, seq_len, 6)`, previous controls `(B, 4)`.
- Output: `(B, 4)` for `[steering/yaw, throttle/pitch, roll, altitude]`.
- Offline training/evaluation use `TubDataset`, which returns `(image, imu, prev_ctrl, target)`.
- `TorchPilot` keeps an IMU history and feeds previous model outputs back as `prev_ctrl` during runtime inference.
- Training uses MPS (Apple Silicon GPU) automatically if available, then CUDA, then CPU

### Data and Model Artifacts

Current useful local data:
- `data/tub_3_26-05-09`: best training tub from recent work; `3368` rows/images; complete nonzero IMU.
- `data/tub_4_26-05-09`: small held-out smoke/eval tub; `161` rows/images; complete IMU.
- `data/tub_5_26-05-09` and `data/tub_6_26-05-09`: known empty or unusable.

Current useful local models:
- `models/hello_world_tub3.pth`: bounded hello-world checkpoint trained on 400 evenly spaced samples from `tub_3`; runtime load smoke passed.
- `models/tub_3_26-05-09.pth`: partial checkpoint from a longer run that was manually stopped.
- `models/pilot.pth`: older baseline.

`nanopilot/evaluate.py` can load current `control_feedback` checkpoints and legacy `legacy_imu_fc` / `legacy_imu_gru` formats for offline comparison.

## Important Config Parameters (`nanopilot/drone_config_{65,80}mm.py`)

- `DRONE_GZ_CAMERA_TOPIC`: must match world + model name — the default 80mm config points at `baylands_80mm`; the retained 65mm config points at `drone_course_65mm`
- `GZ_WORLD` env var in `scripts/start.sh`: defaults to `baylands_80mm` for the 80mm airframe and `drone_course_65mm` for 65mm; override manually only when selecting a specific world
- `BETAFLIGHT_RC_HOST`/`BETAFLIGHT_RC_PORT`: BetaFlight SITL RC endpoint (default 127.0.0.1:9004)
- `DRONE_HOVER_THROTTLE`: PWM midpoint for hover (default 1500)
- `DRONE_THROTTLE_RANGE`: altitude [-1,1] maps to ±this around hover (default 300)
- `DRONE_MAX_PITCH_ANGLE`: max pitch degrees for forward tilt (default 25.0)
- `DRONE_MAX_ROLL_ANGLE`: max roll degrees for lateral bank. Defaults to pitch if omitted; values above pitch increase roll stick PWM relative to pitch.
- `SIMULATED_DELAY_MS`: simulated camera delay in ms (0=off)
- `MEASURE_LOOP_DELAY`: log vehicle loop timing stats
- `IMAGE_W`/`IMAGE_H`: camera resolution for CNN pipeline (default 640×480)
- `DRIVE_LOOP_HZ`: vehicle loop frequency

## Flight Tuning: Throttle, PWM, Hover

### RC channels and PWM

BetaFlight SITL is fed 16-channel RC packets over UDP 9004 at 50Hz. Each channel is a `uint16` PWM value in microseconds, range `1000`–`2000` (1500 = centered stick). Defaults follow the BetaFlight AETR rxmap:

| Channel | Meaning (Angle mode) | How it's driven |
|---------|---------------------|-----------------|
| CH1 | Roll  | `1500 + roll × 500 × DRONE_INPUT_SENSITIVITY` — lateral tilt |
| CH2 | Pitch | `1500 + throttle × 500 × DRONE_INPUT_SENSITIVITY` — forward tilt |
| CH3 | **Motor throttle** | bipolar around hover: `clamp(HOVER_THROTTLE + altitude × THROTTLE_RANGE, 1000, 2000)` |
| CH4 | Yaw   | `1500 + steering × 500 × DRONE_INPUT_SENSITIVITY` |
| CH5 (AUX1) | Arm | 2000 armed, 1000 disarmed |
| CH6 (AUX2) | Angle mode | 2000 active |

CH3 is **bipolar**: `altitude=0` → hover PWM (drone holds altitude — sim thrust is deterministic), `altitude=+1` → `HOVER_THROTTLE + THROTTLE_RANGE` (climb), `altitude=-1` → `HOVER_THROTTLE - THROTTLE_RANGE` (descend). Arrow-key UI increments by `DRONE_THROTTLE_STEP_SIZE` per keydown and snaps altitude to 0 on keyup, giving an analog-stick feel. See `drone_gym.py:_map_controls_to_rc`.

### Hover PWM and the thrust-to-weight envelope

The hover PWM depends on: drone mass, `motorConstant` in `model.sdf`, `maxRpm`, and (secondarily) the LiftDrag plugin tuning on each rotor. At hover, total thrust must equal weight: `4 × motorConstant × ω² = m × g`.

For the current 65mm Air65-style model (total mass ≈ 0.031 kg, `maxRpm=2094`):

- `motorConstant = 7.0e-8` → hover at **PWM ≈ 1493** (50% throttle), TWR ≈ 4× — matches a real BetaFPV Air65 (65mm, ~31g AUW)
- The 80mm Pavo Pico II O4 profile uses `motorConstant = 1.9e-7`, total modeled mass 0.079kg with the selected battery, and calibrated hover PWM 1475.

If the drone launches at a PWM well below 1500, lower `motorConstant`. If it can't lift off at all, raise it. Since thrust ∝ motorConstant × ω² and ω scales linearly with PWM, motorConstant must scale with mass to preserve hover PWM when you change the airframe.

### Finding hover PWM: `test_thrust.py`

`./scripts/test_thrust.sh` is the fastest way to re-tune after any change to `model.sdf` or drone mass. It:
1. Brings up BetaFlight SITL + Gazebo (via `start.sh --no-manage`)
2. Arms BetaFlight with a proper **disarm → arm** sequence (BF sets a `NOT_DISARMED | ARM_SWITCH` flag on boot if it sees AUX1 HIGH before ever seeing it LOW — any test that skips the disarm phase will sit with motors at 0)
3. Ramps throttle 1000→2000 in 50 PWM steps (0.5s per step) and logs altitude at each
4. Tears everything down

Interpret the output like this: hover is the lowest PWM where altitude climbs above ~0.04m within 0.5s. Rows showing `0.010m` are "still on the ground" (the spawn altitude). A well-tuned drone should show hover near PWM 1500, with the altitude growing smoothly — not exponentially — for the next 2–3 steps.

### Why BetaFlight needs a disarm phase

BetaFlight's SITL boot state includes an "arm switch was high at boot" safety flag (`NOT_DISARMED | ARM_SWITCH` in MSP_STATUS_EX's arming disable flags). The flag clears only after BF observes AUX1 go LOW. `drone_gym.py` already does this in phase 1 of `_betaflight_loop`; any ad-hoc test or tool that sends RC packets to port 9004 must do the same or motors will stay at 0 regardless of the throttle channel.

### `DRONE_HOVER_THROTTLE` vs. `DRONE_THROTTLE_RANGE`

`DRONE_HOVER_THROTTLE` is the PWM that produces hover thrust (altitude=0). `DRONE_THROTTLE_RANGE` is the symmetric deflection around it. The result is clamped to `[1000, 2000]`. With defaults (`HOVER_THROTTLE=1500`, `THROTTLE_RANGE=300`):

- `altitude = -1.0` → PWM 1200 (descend)
- `altitude =  0.0` → PWM 1500 (hover)
- `altitude = +1.0` → PWM 1800 (climb)

The drone takes off on arm because CH3 starts at hover PWM as soon as the arm sequence (which holds CH3=1000 for 2s) completes. To land, hold Down to bring altitude toward -1. If you change the real hover PWM (e.g. by editing `motorConstant`), update `DRONE_HOVER_THROTTLE` so `altitude=0` still corresponds to hover — otherwise the CNN will learn a skewed altitude distribution.

### Manual control mappings

Keyboard/web controls:
- Up/down arrows control `altitude` (motor throttle around hover).
- Left/right arrows control `steering` (yaw).
- `j`/`l` control `roll`.

Xbox controls:
- Left stick X controls `steering` (yaw).
- Left stick Y controls `altitude`.
- Right stick X controls `roll`.
- Right stick Y controls `throttle` (forward pitch).
- RT is the explicit arm/deadman input.
- LT triggers one-shot pose reset (teleports drone back to spawn via `/world/<world>/set_pose`).
  B button cycles drive mode (user → local_angle → local).
- Reset implemented in `DroneGymEnv._do_reset()` using `gz.transport13.Node.request()`.

### Planned: vertical-velocity damper (not yet implemented)

Problem: "hover PWM" means thrust = weight → zero *acceleration*. It does not zero existing vertical *velocity*. In sim (minimal air drag) a drone that was climbing and has its stick released keeps coasting upward. Users expect "release = stop in mid-air," which needs an active damper.

Design — proportional altitude-hold in `drone_gym.py`:
1. Subscribe to `/world/drone_course/dynamic_pose/info` (Pose_V) in a gz-transport thread; keep latest position and compute vz by differencing consecutive poses (or subscribe to a velocity topic if one exists). `test_thrust.py` already has the subscribe-once-keep-latest pattern — lift it into `drone_gym.py` as `_PoseTracker`.
2. In `_map_controls_to_rc`, when `abs(altitude) < hold_deadband` (e.g. 0.05), bias CH3 by `-k × vz`. `k` in PWM-per-(m/s) — start with `k = 30` (i.e. a 1 m/s climb gets countered by -30 PWM, roughly -1.4 m/s² in current physics).
3. When the user gives altitude input (stick out of deadband), bypass the damper so the climb command dominates.
4. Add config knobs: `DRONE_ALTITUDE_HOLD_K = 30`, `DRONE_ALTITUDE_HOLD_DEADBAND = 0.05`, `DRONE_ALTITUDE_HOLD_ENABLED = True`.

Why proportional only: we specifically *don't* want integral because it would fight the user's altitude commands. We also don't want target-altitude-based hold (too complex, requires latching a target when stick releases).

Validation: add `--mode=damper` to `test_thrust.py`. Steps: (a) fly to ~2m with CH3=1600 for 2s, (b) cut to CH3=hover, sample altitude for 5s at 0.25s, (c) assert `|vz| < 0.1 m/s` within 1s and altitude doesn't drift more than 0.5m over the following 4s.

Risks: oscillation if `k` is too high. Start low and tune up. The Gazebo pose topic is ~30Hz so the damper loop won't be tight — expect smooth but not snappy settling.

### Planned: translational drag / lateral coast fix

Problem: the current Gazebo airframes have essentially no body translational drag. Both external model files set:

```xml
<velocity_decay>
  <linear>0.0</linear>
  <angular>0.0</angular>
</velocity_decay>
```

Files:
- `~/dev/aeroloop_gazebo/models/betaloop_drone_cam_65mm/model.sdf`
- `~/dev/aeroloop_gazebo/models/betaloop_drone_cam_80mm/model.sdf`

Observed symptom: after sideways roll input, returning to level stops lateral acceleration but does not bleed off existing XY velocity, producing a "skating rink" effect.

Validation now exists in `nanopilot/test_thrust.py --mode=lateral-coast`. It:
1. Climbs to altitude.
2. Applies roll for a configurable acceleration phase.
3. Centers roll and samples XY speed, XY drift, altitude drift, and speed half-life.

Example command:

```bash
./scripts/test_thrust.sh --airframe=80mm --mode=lateral-coast \
  --airborne-hover=1475 --airborne-climb=1500 --airborne-climb-s=1.5 \
  --lateral-roll-pwm=1700 --lateral-accel-s=1.5 --lateral-coast-s=6
```

Baseline result before drag, 80mm Pavo Pico II:
- Peak sampled XY speed after leveling: `3.200 m/s`
- Final XY speed after `6.3s`: `3.033 m/s`
- XY drift while leveled: `19.344 m`
- Speed half-life: `> 6.0s`

Implementation plan:
1. Start with SDF `velocity_decay` on `base_link`, because it is simple, reversible, and applies directly to the missing body velocity damping. Tune both 65mm and 80mm model files in `~/dev/aeroloop_gazebo/models/.../model.sdf`.
2. Sweep small linear decay values first, e.g. `0.05`, `0.10`, `0.20`, `0.35`. Do not change angular decay initially; attitude control and rotor dynamics are already sensitive.
3. After each value, run `--mode=lateral-coast` with the gentler command above. Target: obvious XY speed decay over 3-6s without making lateral control feel like it is flying through syrup. A reasonable first target is speed falling below 50% within roughly 2-4s for a ~3 m/s sideways entry.
4. Re-run `--mode=inflight-hover` or the known hover checks after choosing a drag value. Linear decay may change effective hover/climb behavior and autopilot training distribution.
5. If scalar `velocity_decay` is too blunt, replace it with a dedicated drag system/plugin that applies force `F = -k1*v - k2*|v|*v` in world XY/body axes, with separate horizontal and vertical coefficients. This is more realistic but should be second pass.

Important distinction: translational drag makes the simulated airframe physically bleed velocity. It is not the same as active position/velocity hold. Even with drag, a real Angle-mode quad will coast somewhat after stick release; it just should not preserve speed for many seconds.

## External Dependencies (not in pyproject.toml)

- BetaFlight SITL binary: `~/dev/betaflight/obj/main/betaflight_SITL.elf` (override with `BETAFLIGHT_SITL_BIN` env var)
  - Source: `~/dev/betaflight/` — build with `make TARGET=SITL` (needs dummy ARM SDK dir, see README)
- aeroloop_gazebo: BetaFlight-Gazebo bridge plugin — set `AEROLOOP_GAZEBO_DIR` env var
  - Source: `~/dev/aeroloop_gazebo/` (gz branch) — build: `cd plugins && mkdir build && cd build && cmake .. -DCMAKE_PREFIX_PATH="/opt/homebrew;/opt/homebrew/opt/qt@5" && make`
  - Plugin: `libBetaflightPlugin.dylib` — loaded via `GZ_SIM_SYSTEM_PLUGIN_PATH` (set by start.sh)
- Gazebo Harmonic: `brew install gz-harmonic` (ARM64 Homebrew only)
- ARM64 Ruby required for gz CLI wrapper: `/opt/homebrew/opt/ruby/bin/ruby`

## Gazebo Worlds & Drone Models

Two parallel airframes are available. `--airframe=65mm|80mm` on `start.sh` / `test_thrust.sh` selects between them (default 80mm). The default 80mm world is `baylands_80mm`; `drone_course_65mm` is retained for Air65.

The `worlds/*_80mm.sdf` files include `betaloop_drone_cam_80mm` and the retained `worlds/drone_course_65mm.sdf` includes `betaloop_drone_cam_65mm`. Each world must include all 5 required world plugins: `Physics`, `UserCommands`, `SceneBroadcaster`, `Sensors` (ogre2), `Imu`.

The 65mm model (`~/dev/aeroloop_gazebo/models/betaloop_drone_cam_65mm/model.sdf`) — BetaFPV Air65:
- 0.022kg base_link, ~0.031kg AUW, rotor positions ±0.023m (65mm wheelbase)
- `motorConstant = 7.0e-8` — hover ≈ PWM 1493, TWR ≈ 4×
- 31mm (1.2") props, 0802-class motors (vel_cmd_max=2094 rad/s)

The 80mm model (`~/dev/aeroloop_gazebo/models/betaloop_drone_cam_80mm/model.sdf`) — Pavo Pico II:
- 0.065kg base_link, 0.079kg total modeled mass with selected battery, rotor positions ±0.028284m (80mm diagonal wheelbase)
- `motorConstant = 1.9e-7` — calibrated hover PWM 1475
- LAVA 1102 14000KV motors and GF 45mm tri-blade prop approximation

Both have forward-facing camera (640×480, 30Hz, 80° FOV), IMU on base_link (1000Hz NED-rotated), BetaflightPlugin with rotor-to-joint mapping (BF QUADX motor order), and scaled Iris mesh as a visual placeholder.

Camera topics:
- 65mm: `/world/drone_course_65mm/model/betaloop_drone_cam_65mm/link/camera_link/sensor/camera/image`
- 80mm default: `/world/baylands_80mm/model/betaloop_drone_cam_80mm/link/camera_link/sensor/camera/image`

### BetaflightPlugin Rotor Mapping

**Critical gotcha:** BF SITL's `pwmCompleteMotorUpdate()` shuffles motors before the UDP send to Gazebo:

```
pkt[0] = BF motor 1   (Front-Right, CCW)
pkt[1] = BF motor 2   (Rear-Left,   CCW)
pkt[2] = BF motor 3   (Front-Left,  CW)
pkt[3] = BF motor 0   (Rear-Right,  CW)
```

BetaflightPlugin's `<rotor id="N">` indexes the **packet slot**, not BF's internal motor number. The model's rotor layout and the `<rotor>` blocks in `model.sdf` are intentionally ordered to match the post-remap packet, so the final `rotor id → joint` mapping is 1:1:

| `<rotor id>` | Receives | Position | Direction | Joint |
|--------------|----------|----------|-----------|-------|
| 0 | BF motor 1 | Front-Right | CCW | `rotor_0_joint` |
| 1 | BF motor 2 | Rear-Left   | CCW | `rotor_1_joint` |
| 2 | BF motor 3 | Front-Left  | CW  | `rotor_2_joint` |
| 3 | BF motor 0 | Rear-Right  | CW  | `rotor_3_joint` |

Do **not** re-map `<jointName>` using a "BF motor index → joint" table. That double-applies the remap: the drone inverts thrust at arm (measured: total thrust ≈ −2.3 N, motors wildly asymmetric) and flips upside-down immediately. The source of truth is the comment block at the top of the `<plugin name="...BetaflightPlugin">` element in `model.sdf`.

## Gotchas & Troubleshooting

### Gazebo on macOS requires separate server and GUI processes
`gz sim` cannot run server + GUI in one process on macOS ([gz-sim#44](https://github.com/gazebosim/gz-sim/issues/44)). `start.sh` launches `gz sim -s` (server) then `gz sim -g` (GUI) separately. Set `GZ_HEADLESS=1` to skip the GUI.

### Stale gz-transport topics
After killing Gazebo, camera topics can persist in gz-transport's multicast discovery cache for several minutes. The readiness check in `start.sh` matches the specific `betaloop_drone_cam_{65,80}mm` topic (derived from `--airframe`) to avoid false positives from stale topics.

### World-level plugins override server.config defaults
Adding ANY `<plugin>` to the world SDF causes Gazebo to skip loading `~/.gz/sim/8/server.config` plugins (Physics, UserCommands, SceneBroadcaster). This means each `worlds/drone_course_XXmm.sdf` must explicitly include ALL five plugins: Physics, UserCommands, SceneBroadcaster, Sensors, and Imu. Without Physics, nothing moves — no gravity, no forces, no joint actuation.

### aeroloop_gazebo CMake needs Qt5 path
The BetaflightPlugin links against gz-sim8 which transitively depends on gz-gui8 (Qt5). On macOS Homebrew, Qt5 is keg-only, so CMake needs: `cmake .. -DCMAKE_PREFIX_PATH="/opt/homebrew;/opt/homebrew/opt/qt@5"`

### BetaflightPlugin FDM bootstrap (deadlock fix)
The upstream BetaflightPlugin only sends FDM state after receiving motor commands, but BetaFlight SITL blocks on `recv(9003)` waiting for FDM before sending motors — a deadlock. Our fix (in `BetaflightPlugin.cc`) moves `SendState()` to run unconditionally every sim tick, before `ReceiveMotorCommand()`, so FDM flows immediately and BetaFlight unblocks. If you re-clone or update aeroloop_gazebo, this fix must be reapplied.

### BetaFlight SITL creates eeprom.bin in cwd
The SITL binary writes `eeprom.bin` (32KB BetaFlight config storage) in whatever directory it's launched from. `start.sh` runs it from the project root, so `eeprom.bin` appears there. It's gitignored. Delete it to reset BetaFlight configuration to defaults.

### Metal rendering crash (gz-sim #2877)
Gazebo camera sensors can crash with the Metal rendering backend. The world uses `ogre2` render engine explicitly. If crashes occur, try: `export GZ_SIM_RENDER_ENGINE=ogre2`

### Shared memory leak warning on shutdown
The `resource_tracker` may warn about leaked shared memory on shutdown — this is cosmetic, caused by a race between `stop_all.sh` killing processes and Python's resource tracker cleanup.

### Switching airframes
Use `--airframe=65mm|80mm` on `start.sh` and `test_thrust.sh` (default 80mm). By default, 80mm selects `baylands_80mm`; 65mm selects `drone_course_65mm`. Set `GZ_WORLD=<world_name>` to use another converted 80mm world such as `forest_80mm`, `landing_target_80mm`, `slalom_gates_80mm`, or `drone_course_80mm`.

## Current Status (2026-04-18)

Two airframes are maintained in parallel for A/B flight comparison: **65mm BetaFPV Air65** (~31g AUW, default) and **80mm Pavo Pico II O4** (79g with selected battery). Each has its own model dir (`betaloop_drone_cam_{65,80}mm`), world SDF (`drone_course_{65,80}mm.sdf`), and config (`drone_config_{65,80}mm.py`). The Pavo Pico II constants are based on BETAFPV's published 80mm wheelbase, LAVA 1102 14000KV motors, and GF 45mm props, with battery mass added to reach 79g AUW. Existing `eeprom.bin` retained across airframes (fresh eeprom path via start.sh is known-broken — see memory).

Known quirk: on Air65, yaw input causes significantly more climb than on larger profiles because the mixer ω² asymmetry produces more vertical acceleration on lower mass/inertia airframes. `DRONE_YAW_PWM_CAP` remains the main tuning knob.

Full stack working end-to-end: `start.sh` → Gazebo + BetaFlight SITL + drone_manage.py all launch, RC packets flow at 50Hz, camera frames arrive via shared memory, Web UI at :8887.

## Current Handoff (2026-05-09)

Goal: collect new-format drone data with no human intervention, then verify training on that new tub. The new model expects image + 6-axis IMU sequence inputs:
- `imu/acl_x`, `imu/acl_y`, `imu/acl_z`
- `imu/gyr_x`, `imu/gyr_y`, `imu/gyr_z`

Recent changes in progress:
- `nanopilot/drone_gym.py` now subscribes to Gazebo pose + IMU telemetry and can output raw 6-axis IMU fields alongside position, attitude, and velocity.
- `DRONE_RECORD_IMU = True` was added to both `drone_config_65mm.py` and `drone_config_80mm.py`.
- `nanopilot/drone_manage.py` records the new IMU fields into manual/autopilot tubs when `DRONE_RECORD_IMU` is enabled.
- `nanopilot/torch_pilot.py` was updated so `.pth` inference passes an IMU history tensor to `LinearModel`; the new model no longer works with image-only inference.
- `nanopilot/dataset.py` was updated for new-format tubs:
  - filters only `*.catalog` files, not `catalog_*.catalog_manifest`
  - skips non-sample catalog rows missing `cam/image_array`
  - resolves images from both tub root and DonkeyCar v2 `images/`
  - warns/counts records missing the six required IMU keys
- `nanopilot/gz_camera_worker.py` unregisters the attached shared memory from the child process resource tracker; parent owns unlinking.
- `nanopilot/autonomous_collect.py` is a headless scripted collector that directly writes a tub after confirming:
  - BetaFlight MSP is reachable on TCP 5761
  - camera frames are nonblank
  - RC/control loop has reached hover throttle after arming
  - Gazebo IMU telemetry is nonzero
- `scripts/collect_train.sh` starts `start.sh --no-manage`, runs `autonomous_collect.py`, then trains on the newly created tub.

Validated test run:
```bash
GZ_HEADLESS=1 ./scripts/start.sh --no-manage --airframe=65mm
uv run --env-file .env python nanopilot/autonomous_collect.py \
  --airframe=65mm --duration=3 --warmup=1 --rate-hz=5 --ready-timeout=30
uv run --env-file .env python nanopilot/torch_train.py \
  --tubs=data/tub_210_26-05-09 --model=/private/tmp/tub_210_test.pth --max-epochs=1
bash ./scripts/stop_all.sh
```

Results:
- `data/tub_210_26-05-09` was created by `autonomous_collect.py`
- 15 records were written
- catalog rows include nonzero `imu/acl_*` and `imu/gyr_*`
- one-epoch training on that tub completed successfully on MPS
- test model saved to `/private/tmp/tub_210_test.pth`

Known notes:
- `data/tub_209_26-05-09` was also created during testing and contains 50 valid records, but the first collector run exited nonzero due to a shared-memory unlink race that was then fixed in `DroneGymEnv.shutdown()`.
- `data/tub_208_26-05-09` was created during manual flight but has zero catalog records because recording stayed false.
- A cosmetic Python `resource_tracker` shared-memory warning may still appear on collector shutdown even when the process exits `0`.
- In manual web-controller runs, `AUTO_RECORD_ON_THROTTLE` may not record if only altitude/motor throttle changes and `user/throttle` stays zero; the scripted collector bypasses this by writing records directly.

Suggested next refactor:
1. Create a shared schema module, e.g. `nanopilot/tub_schema.py`, with `DRONE_TUB_INPUTS`, `DRONE_TUB_TYPES`, and `IMU_KEYS`; use it from `drone_manage.py`, `autonomous_collect.py`, and `dataset.py`.
2. Extract Gazebo telemetry helpers (`_SimTelemetry`, `_PoseTracker`, quaternion conversion) from `drone_gym.py` into a focused module, e.g. `nanopilot/gz_telemetry.py`.
3. Extract shared `build_drone_env(cfg, airframe=...)` construction so `drone_manage.py` and `autonomous_collect.py` do not duplicate the long `DroneGymEnv(...)` argument list.
4. Consider making autonomous flight profiles pluggable (`--profile=figure-eight`, `--profile=hover-sweep`, etc.) once the code is cleaner.

## Current Handoff (2026-05-09 evening)

Session goal was to continue the new-format drone data work, refactor it, validate real scripted collection, then start objective model evaluation.

Completed refactor work:
- Added shared tub schema module `nanopilot/tub_schema.py` with `DRONE_TUB_INPUTS`, `DRONE_TUB_TYPES`, `IMU_KEYS`, and `drone_tub_schema(...)`.
- Added `nanopilot/gz_telemetry.py` and moved Gazebo telemetry helpers out of `drone_gym.py`:
  - pose + IMU subscription
  - quaternion-to-Euler conversion
  - pose tracker for altitude-hold vertical velocity
- Added `nanopilot/drone_env.py` with shared `build_drone_env(...)`.
- Updated `drone_manage.py`, `autonomous_collect.py`, and `dataset.py` to use the shared schema and env construction.
- Refactor behavior was compile checked:
  ```bash
  uv run --env-file .env python -m py_compile \
    nanopilot/drone_gym.py nanopilot/drone_env.py nanopilot/gz_telemetry.py \
    nanopilot/tub_schema.py nanopilot/drone_manage.py \
    nanopilot/autonomous_collect.py nanopilot/dataset.py \
    nanopilot/torch_train.py nanopilot/torch_pilot.py
  ```
- `autonomous_collect.py --help` imports cleanly after refactor.
- One-epoch training smoke on `data/tub_210_26-05-09` completed and saved `/private/tmp/tub_210_refactor_smoke.pth`.

Real collection validation after refactor:
```bash
GZ_HEADLESS=1 ./scripts/start.sh --no-manage --airframe=65mm
uv run --env-file .env python nanopilot/autonomous_collect.py \
  --airframe=65mm --duration=3 --warmup=1 --rate-hz=5 --ready-timeout=30
bash ./scripts/stop_all.sh
```

Results:
- New tub created: `data/tub_211_26-05-09`
- `15` records written.
- Catalog check: `rows=15 samples=15 missing_imu=0 nonzero_imu=15`.
- First sample had nonzero IMU values, e.g. `imu/acl_z=-9.825064663923019`.
- Sim stack was stopped afterward. `pgrep -fl 'betaflight_SITL|gz sim|ruby.*gz'` returned nothing.

Model/evaluation work:
- User created `nanopilot/evaluate.py` from Gemini guidance. Initial file included pasted markdown after Python code and failed with:
  `SyntaxError: unterminated string literal`.
- Replaced it with a CLI evaluator that supports:
  - `--model` for single-model metrics.
  - `--old-model` + `--new-model` for comparison.
  - `--tubs`, `--image-h`, `--image-w`, `--seq-len`, `--batch-size`.
  - Metrics: per-axis MAE, RMSE, correlation, and output jitter.
- `evaluate.py` also detects old IMU-FC checkpoints (`imu_fc.*` keys) and loads them with a compatibility model. This matters because current `torch_model.py` now uses `imu_gru`, while available smoke checkpoints from earlier in the day use `imu_fc`.
- `uv run --env-file .env python -m py_compile nanopilot/evaluate.py` passed.

Available checkpoints in `/private/tmp`:
- `/private/tmp/tub_210_test.pth` - old IMU-FC checkpoint from earlier smoke.
- `/private/tmp/tub_210_refactor_smoke.pth` - old IMU-FC checkpoint after refactor smoke.
- `/private/tmp/tub_210_gru_smoke.pth` - one-epoch checkpoint trained with current GRU-based `torch_model.py`.

Evaluation runs/results:
```bash
uv run --env-file .env python nanopilot/evaluate.py \
  --tubs=data/tub_211_26-05-09 \
  --old-model=/private/tmp/tub_210_test.pth \
  --new-model=/private/tmp/tub_210_refactor_smoke.pth
```
- Old MAE: steering `0.0602`, throttle `0.1099`, altitude `0.0365`.
- New MAE: steering `0.0815`, throttle `0.0894`, altitude `0.0187`.
- On this tiny 15-sample held-out tub, the refactor smoke checkpoint improved throttle/altitude MAE but worsened steering MAE.

Then trained a current GRU smoke checkpoint:
```bash
uv run --env-file .env python nanopilot/torch_train.py \
  --tubs=data/tub_210_26-05-09 \
  --model=/private/tmp/tub_210_gru_smoke.pth \
  --max-epochs=1
```
- Completed on MPS.
- Best validation loss: `0.005988`.

Compared old IMU-FC vs current GRU smoke on held-out `tub_211`:
```bash
uv run --env-file .env python nanopilot/evaluate.py \
  --tubs=data/tub_211_26-05-09 \
  --old-model=/private/tmp/tub_210_refactor_smoke.pth \
  --new-model=/private/tmp/tub_210_gru_smoke.pth
```
- Old IMU-FC MAE: steering `0.0815`, throttle `0.0894`, altitude `0.0187`.
- New GRU MAE: steering `0.1470`, throttle `0.1105`, altitude `0.0684`.
- On this tiny held-out tub, the one-epoch GRU smoke checkpoint was worse on MAE across all three axes. Treat this as a smoke result only, not a real architecture verdict.

Current working tree at handoff:
```bash
git status --short --untracked-files=all
 M nanopilot/smoke_test.py
 M nanopilot/torch_model.py
 M nanopilot/torch_train.py
?? nanopilot/evaluate.py
```

Important caution:
- The modifications to `smoke_test.py`, `torch_model.py`, and `torch_train.py` appear to be user/new-session work. Do not revert them.
- `evaluate.py` is new and currently contains the CLI evaluator plus compatibility loader for old IMU-FC checkpoints.
- If continuing evaluation, use a larger, fixed benchmark tub; 15 records is enough only for smoke testing.

Recommended next steps:
1. Inspect the dirty changes in `torch_model.py`, `torch_train.py`, and `smoke_test.py` before editing; current `torch_model.py` uses an IMU GRU.
2. Decide whether `evaluate.py` should keep legacy IMU-FC checkpoint support or only support current GRU models.
3. Create a larger benchmark tub (or reserve one existing high-quality tub) and always evaluate new checkpoints against that same held-out data.
4. Add CLI/config support in `evaluate.py` if needed for non-320x240 configs, multiple tubs, CSV/JSON output, or aggregate score weighting by axis.

## Current Handoff (2026-05-09 late evening)

Continued evaluation/model workflow work:
- `nanopilot/evaluate.py` now defaults to the fixed held-out benchmark tub `data/tub_209_26-05-09` when `--tubs`/`--benchmark-tubs` are omitted. This tub has `50` samples, `missing_imu=0`, and `nonzero_imu=50`.
- The default benchmark can be overridden with `--tubs`, `--benchmark-tubs`, or `NANOPILOT_BENCHMARK_TUBS`.
- Evaluator now supports three checkpoint formats:
  - `legacy_imu_fc` (`imu_fc.*` keys)
  - `legacy_imu_gru` (`imu_gru.*` keys but no control-feedback branch)
  - current `control_feedback` checkpoints (`ctrl_fc.*` keys)
- Evaluator now accepts dataset batches as `(image, imu, prev_ctrl, target)`, computes an equal-weight MAE score by default, accepts `--weights steering,throttle,altitude`, and can write machine-readable results with `--json-output`.
- `torch_model.LinearModel.forward(...)` now accepts `prev_ctrl=None` and substitutes zero controls, preserving simple two-input smoke/debug calls while training/evaluation pass real previous controls.
- `torch_pilot.py` now passes previous model outputs back into the model as `prev_ctrl`, so runtime inference matches the current training graph. Its import also works both as `torch_pilot` from the DonkeyCar script path and as `nanopilot.torch_pilot`.

Validation performed:
```bash
uv run --env-file .env python -m py_compile \
  nanopilot/evaluate.py nanopilot/torch_model.py nanopilot/torch_pilot.py \
  nanopilot/torch_train.py nanopilot/smoke_test.py nanopilot/dataset.py

uv run --env-file .env python nanopilot/evaluate.py \
  --old-model=/private/tmp/tub_210_refactor_smoke.pth \
  --new-model=/private/tmp/tub_210_gru_smoke.pth \
  --json-output=/private/tmp/nanopilot_eval_compare.json

uv run --env-file .env python nanopilot/torch_train.py \
  --tubs=data/tub_210_26-05-09 \
  --model=/private/tmp/tub_210_ctrl_smoke.pth \
  --max-epochs=1

uv run --env-file .env python nanopilot/evaluate.py \
  --old-model=/private/tmp/tub_210_refactor_smoke.pth \
  --new-model=/private/tmp/tub_210_ctrl_smoke.pth \
  --json-output=/private/tmp/nanopilot_eval_ctrl_compare.json

uv run --env-file .env python nanopilot/smoke_test.py
```

Benchmark results on held-out `data/tub_209_26-05-09`:
- Old IMU-FC `/private/tmp/tub_210_refactor_smoke.pth`: score `0.1669`, MAE steering `0.2522`, throttle `0.1717`, altitude `0.0768`.
- Old GRU `/private/tmp/tub_210_gru_smoke.pth`: score `0.2011`, MAE steering `0.3178`, throttle `0.1590`, altitude `0.1264`.
- Current control-feedback `/private/tmp/tub_210_ctrl_smoke.pth`: score `0.1406`, MAE steering `0.2010`, throttle `0.1512`, altitude `0.0695`.
- Current control-feedback smoke improved equal-weight MAE score by `15.78%` vs the old IMU-FC smoke on this fixed benchmark. Treat as smoke-level evidence only; training tub still had only 15 samples.

Additional runtime check:
```bash
uv run --env-file .env python - <<'PY'
import numpy as np
from nanopilot.torch_pilot import TorchPilot
pilot = TorchPilot(input_shape=(3, 240, 320), seq_len=3)
pilot.load('/private/tmp/tub_210_ctrl_smoke.pth')
print(tuple(round(v, 6) for v in pilot.run(np.zeros((240, 320, 3), dtype=np.uint8))))
PY
```
- Loaded on MPS and returned `(0.068052, 0.110246, 0.078214)`.

Current working tree after this step:
```bash
 M AGENT.md
 M nanopilot/evaluate.py
 M nanopilot/torch_model.py
 M nanopilot/torch_pilot.py
```

Recommended next steps:
1. Collect a larger dedicated benchmark tub and update `DEFAULT_BENCHMARK_TUBS` (or use `NANOPILOT_BENCHMARK_TUBS`) so model comparisons are based on more than 50 held-out samples.
2. Train current control-feedback checkpoints on more than `data/tub_210_26-05-09`; the present comparison is useful for plumbing, not architecture selection.
3. Consider moving checkpoint compatibility classes out of `evaluate.py` if old checkpoints need to be loaded by runtime inference too.

## Current Handoff (2026-05-09 night)

Repository / branch state:
- `feature/PX4-OSX-native-build` was fast-forward merged into `main`, pushed to `origin/main`, then deleted both remotely and locally.
- Current branch is `main`.
- At signoff, `git status --short --branch` was clean: `## main...origin/main`.

Recent flight / controller work:
- Angle mode flies well manually.
- `DRONE_YAW_THROTTLE_FEEDFORWARD` sign-flip experiment was reverted to latest committed behavior. User found that the old behavior is better and will live with entering negative values locally if needed.
- Xbox controller bridge was validated with `xbox_bridge/smoke_test.py`:
  - frames arrived at ~60Hz
  - sticks, RT, A, and B were detected
  - `nonzero_input_seen=True`
- `scripts/stop_all.sh` already kills XboxBridge; it was hardened to match both `XboxBridge.app` and `Contents/MacOS/XboxBridge`, and removes `/tmp/nanopilot_xbox.sock`.
- Added `scripts/stop-all.sh` as a compatibility wrapper around `scripts/stop_all.sh`.
- `scripts/start.sh` now explicitly stops XboxBridge in its Ctrl+C/exit cleanup trap and before launching a fresh bridge for `--xbox`.
- Xbox arm debugging showed:
  - RT controls `user/arm`.
  - Without RT held, `ch5=1000` and motors stay at `0.0`.
  - With RT held, `ch5=2000`; however BetaFlight still would not re-arm if CH3 was already at hover/climb.
- `drone_gym.py` was updated so an explicit Xbox arm transition holds CH3 at `1000` for `1.0s` while CH5 is high, then resumes normal hover/climb throttle. Verified mapping:
  ```text
  initial explicit arm ch3/ch5 1000 2000
  after hold ch3/ch5 1590 2000
  disarm ch3/ch5 1490 1000
  ```
- `drone_gym.py` RC logs now include CH5 and CH6 PWM. `xbox_controller.py` logs RT arm state changes.

Data / model artifacts available now:
- `data/tub_3_26-05-09`: good training tub, `3368` catalog rows and `3368` images.
  - Validation during session showed: `missing_img=0`, `missing_imu=0`, `nonzero_imu=3368`.
  - Controls had signal: steering nonzero `2527`, throttle nonzero `3118`, altitude nonzero `151`.
- `data/tub_4_26-05-09`: small Xbox/control tub, `161` rows and `161` images.
- `data/tub_5_26-05-09` and `data/tub_6_26-05-09`: empty or unusable.
- Existing model files:
  - `models/pilot.pth` old baseline.
  - `models/tub_3_26-05-09.pth` partial training checkpoint, `9.9M`, created during a longer run that was manually stopped. It may be usable, but it was not a deliberate bounded "hello world" run.

Training status:
- A full/default `torch_train.py` run on `data/tub_3_26-05-09` was started:
  ```bash
  uv run --env-file .env python nanopilot/torch_train.py \
    --tubs=data/tub_3_26-05-09 \
    --model=models/tub_3_26-05-09.pth
  ```
- It wrote a checkpoint, then was stopped at user request before natural completion.
- Need next: a bounded "hello world" training run that finishes in <=5 minutes and creates an autopilot worth testing.

Recommended next-session plan:
1. Add or use a bounded training option that guarantees a short run. Existing `--max-epochs` helps, but the current full model can still take time on 3368 images. Prefer one of:
   - train `--max-epochs=1` on a smaller tub subset, if subset support is added; or
   - add CLI flags such as `--max-samples`, `--train-split`, `--no-model-summary`, and maybe `--batch-size`; or
   - create a temporary subset tub from `data/tub_3_26-05-09` with ~300-600 records.
2. Produce a test autopilot model, e.g. `models/hello_world_tub3.pth`.
3. Run a quick evaluator pass on held-out data if practical:
   ```bash
   uv run --env-file .env python nanopilot/evaluate.py \
     --tubs=data/tub_4_26-05-09 \
     --model=models/hello_world_tub3.pth
   ```
   `tub_4` is small and Xbox/control-biased, so treat it as a smoke check only.
4. Test autopilot manually with:
   ```bash
   ./scripts/start.sh --model=models/hello_world_tub3.pth --airframe=65mm
   ```
   Watch the first seconds carefully; be ready to stop with Ctrl+C / `scripts/stop_all.sh`.

## Current Handoff (2026-05-10 morning)

Bounded hello-world training run completed successfully.

Code change:
- `nanopilot/torch_train.py` now supports bounded smoke-training flags:
  - `--max-samples` uses a deterministic, evenly spaced subset of the loaded dataset.
  - `--batch-size` overrides config `BATCH_SIZE`.
  - `--train-split` overrides config `TRAIN_TEST_SPLIT`.
  - `--no-model-summary` skips verbose layer printing.
  - Train/val random split now uses a fixed seed (`42`) for repeatability.

Validation:
```bash
uv run --env-file .env python -m py_compile \
  nanopilot/torch_train.py nanopilot/evaluate.py nanopilot/dataset.py \
  nanopilot/torch_model.py nanopilot/torch_pilot.py

uv run --env-file .env python nanopilot/torch_train.py --help
```

Training command:
```bash
time uv run --env-file .env python nanopilot/torch_train.py \
  --tubs=data/tub_3_26-05-09 \
  --model=models/hello_world_tub3.pth \
  --max-epochs=1 \
  --max-samples=400 \
  --batch-size=64 \
  --no-model-summary
```

Training result:
- Device: `mps`
- Source tub samples: `3368`
- Deterministic training subset: `400`
- Train/val split: `320/80`
- Epoch time: `5.9s`
- Wall time from shell `time`: `7.937s`
- Best validation loss: `0.091767`
- Model written: `models/hello_world_tub3.pth` (`9.9M`)

Quick evaluation on small held-out tub:
```bash
uv run --env-file .env python nanopilot/evaluate.py \
  --tubs=data/tub_4_26-05-09 \
  --model=models/hello_world_tub3.pth \
  --json-output=/private/tmp/hello_world_tub3_eval.json
```

Evaluation result on `data/tub_4_26-05-09`:
- Samples: `161`
- Missing IMU: `0`
- Checkpoint kind: `control_feedback`
- Equal-weight MAE score: `0.2356`
- MAE: steering `0.0768`, throttle `0.2508`, altitude `0.3791`
- RMSE: steering `0.1004`, throttle `0.3017`, altitude `0.5868`
- Correlation: steering `-0.8059`, throttle `0.9632`, altitude `0.0961`
- Treat this only as a smoke check; tub 4 is small and Xbox/control-biased.

Runtime load smoke:
```bash
uv run --env-file .env python - <<'PY'
import numpy as np
from nanopilot.torch_pilot import TorchPilot
pilot = TorchPilot(input_shape=(3, 240, 320), seq_len=3)
pilot.load('models/hello_world_tub3.pth')
print(tuple(round(float(v), 6) for v in pilot.run(np.zeros((240, 320, 3), dtype=np.uint8))))
PY
```

Output:
```text
TorchPilot: loaded models/hello_world_tub3.pth on mps
(0.055345, 0.193857, -0.05971)
```

Autopilot test command:
```bash
./scripts/start.sh --model=models/hello_world_tub3.pth --airframe=65mm
```

Watch the first seconds carefully and be ready to stop with Ctrl+C or:
```bash
bash ./scripts/stop_all.sh
```

## Current Handoff (2026-05-10 full tub 3 training)

Full `data/tub_3_26-05-09` training run completed.

Command:
```bash
time uv run --env-file .env python nanopilot/torch_train.py \
  --tubs=data/tub_3_26-05-09 \
  --model=models/tub_3_26-05-09_full.pth \
  --max-epochs=10 \
  --batch-size=64 \
  --no-model-summary
```

Training result:
- Device: `mps`
- Samples: `3368`
- Train/val split: `2694/674`
- Stopped after epoch `8/10` via early stopping.
- Wall time: `5:03.97`
- Saved checkpoint: `models/tub_3_26-05-09_full.pth` (`9.9M`)
- Training log:
  - Epoch 1: train `0.038359`, val `0.030147`, saved
  - Epoch 2: train `0.014134`, val `0.005774`, saved
  - Epoch 3: train `0.002000`, val `0.000912`, saved
  - Epoch 4: train `0.001146`, val `0.002132`
  - Epoch 5: train `0.001115`, val `0.001230`
  - Epoch 6: train `0.000984`, val `0.000663`
  - Epoch 7: train `0.000874`, val `0.000725`
  - Epoch 8: train `0.000854`, val `0.000662`, early stop
- Note: `torch_train.py` only saves when validation improves by more than `MIN_DELTA` (`0.0005`), so the saved checkpoint is from epoch 3 even though epoch 6/8 had slightly lower absolute validation loss.

Quick evaluation on `data/tub_4_26-05-09`:
```bash
uv run --env-file .env python nanopilot/evaluate.py \
  --tubs=data/tub_4_26-05-09 \
  --model=models/tub_3_26-05-09_full.pth \
  --json-output=/private/tmp/tub3_full_eval_tub4.json
```

Evaluation result:
- Samples: `161`
- Missing IMU: `0`
- Checkpoint kind: `control_feedback`
- Equal-weight MAE score: `0.1712`
- MAE: steering `0.0552`, throttle `0.1231`, altitude `0.3353`
- RMSE: steering `0.0808`, throttle `0.2246`, altitude `0.5605`
- Correlation: steering `0.7389`, throttle `0.9514`, altitude `0.3720`
- This improves the tub 4 smoke score vs `models/hello_world_tub3.pth` (`0.2356`), but `tub_4` is still a small Xbox/control-biased smoke benchmark.

Runtime load smoke:
```bash
uv run --env-file .env python - <<'PY'
import numpy as np
from nanopilot.torch_pilot import TorchPilot
pilot = TorchPilot(input_shape=(3, 240, 320), seq_len=3)
pilot.load('models/tub_3_26-05-09_full.pth')
print(tuple(round(float(v), 6) for v in pilot.run(np.zeros((240, 320, 3), dtype=np.uint8))))
PY
```

Output:
```text
TorchPilot: loaded models/tub_3_26-05-09_full.pth on mps
(0.219729, 0.90419, -0.069066)
```

Autopilot test command:
```bash
./scripts/start.sh --model=models/tub_3_26-05-09_full.pth --airframe=65mm
```

## Current Handoff (2026-05-10 roll control)

Implemented four-axis controls end-to-end: yaw, pitch, roll, and altitude.

Control mapping:
- Keyboard/web: left/right arrows = yaw, `j`/`l` = roll, up/down arrows = altitude, throttle remains forward pitch.
- Browser Gamepad API: left stick X = yaw, left stick Y = altitude, right stick X = roll, right stick Y = pitch.
- Native Xbox bridge: UDP frame is now `"<fffffBB"`: `lX,lY,rX,rY,rT,buttons,connected`; Python receiver maps left stick X to yaw and right stick X to roll.
- BetaFlight RC mapping is AETR: CH1 roll, CH2 pitch, CH3 motor throttle, CH4 yaw.

Code areas changed:
- Tub schema now records `user/roll`; new training data is image + yaw/pitch/roll/altitude + mode + telemetry.
- `TubDataset`, `LinearModel`, `TorchPilot`, and `evaluate.py` now use four control outputs: `[yaw, pitch, roll, altitude]`.
- `drone_manage.py` wires `user/roll`, `pilot/roll`, and `DriveMode` through the vehicle loop.
- `drone_gym.py` maps roll to CH1 PWM and logs `roll_pwm`.
- `autonomous_collect.py` writes scripted roll into generated tubs.
- `xbox_bridge/main.swift` and `xbox_bridge/smoke_test.py` use the wider frame with left-stick X.

Validation performed:
```bash
uv run --env-file .env python -m py_compile \
  nanopilot/drone_gym.py nanopilot/drone_env.py nanopilot/drone_manage.py \
  nanopilot/xbox_controller.py nanopilot/tub_schema.py nanopilot/dataset.py \
  nanopilot/torch_model.py nanopilot/torch_pilot.py nanopilot/torch_train.py \
  nanopilot/evaluate.py nanopilot/autonomous_collect.py nanopilot/smoke_test.py \
  xbox_bridge/smoke_test.py

bash -n scripts/start.sh scripts/collect_train.sh scripts/test_thrust.sh
bash xbox_bridge/build.sh
uv run --env-file .env python nanopilot/smoke_test.py
```

Bounded four-output training/eval smoke:
```bash
uv run --env-file .env python nanopilot/torch_train.py \
  --tubs=data/tub_3_26-05-09 \
  --model=/private/tmp/roll4_smoke.pth \
  --max-epochs=1 --max-samples=64 --batch-size=16 --no-model-summary

uv run --env-file .env python nanopilot/evaluate.py \
  --tubs=data/tub_4_26-05-09 \
  --model=/private/tmp/roll4_smoke.pth
```

Results:
- Training smoke completed on MPS and saved `/private/tmp/roll4_smoke.pth`.
- Evaluation printed all four axes and produced equal-weight MAE score `0.1832`.
- TorchPilot load smoke returned four values.
- Focused RC check returned `{'ch1_roll': 1750, 'ch2_pitch': 1562, 'ch3_throttle': 1500, 'ch4_yaw': 1425}` for roll=1, pitch=0.25, yaw=-1 at 0.5 sensitivity and yaw cap 75.
- Headless full-stack smoke with `GZ_HEADLESS=1 GZ_WORLD=landing_target_65mm ./scripts/start.sh --airframe=65mm` reached Gazebo ready, started `drone_manage.py`, delivered nonblank camera frames, armed BetaFlight, and logged roll/pitch/yaw/throttle PWM.

Important compatibility note:
- Existing three-output checkpoints such as `models/hello_world_tub3.pth` and `models/tub_3_26-05-09_full.pth` are no longer runtime-compatible with the current four-output `LinearModel`.
- Old tubs without `user/roll` load with roll as `0.0`, which is acceptable for smoke/debugging but not enough to train meaningful roll behavior. Collect new roll-inclusive tubs before training a real autopilot.

Current workspace notes:
- `README.md` had user edits before this handoff; do not overwrite them casually.
- `nanopilot/drone_config_65mm.py` may have local/user tuning, including camera/world settings; inspect before editing.
- Sim processes were stopped after validation; `pgrep -fl 'gz sim|gzserver|gzclient|ArduCopter|ardupilot|drone_manage|sim_vehicle|mavproxy'` returned nothing.

## Current Handoff (2026-05-16 lateral drag investigation)

User reported a "skating rink" effect: after sideways motion, leveling the drone leaves it sliding for a long time, unlike Velocidrone.

Finding:
- The Gazebo airframe SDFs in `~/dev/aeroloop_gazebo/models/betaloop_drone_cam_65mm/model.sdf` and `~/dev/aeroloop_gazebo/models/betaloop_drone_cam_80mm/model.sdf` set base-link `<velocity_decay><linear>0.0</linear><angular>0.0</angular></velocity_decay>`.
- Current model has rotor/motor damping and yaw drag terms, but no meaningful body translational drag. So Angle mode levels the drone but does not remove existing XY velocity.

Added `nanopilot/test_thrust.py --mode=lateral-coast`:
- `make_channels(...)` now accepts `roll_pwm` for CH1.
- New `run_lateral_coast_test(...)` climbs, applies sideways roll, centers roll, then logs XY speed, XY drift, altitude drift, and speed half-life.
- New args: `--lateral-roll-pwm`, `--lateral-accel-s`, `--lateral-coast-s`.

Validation:
```bash
uv run --env-file .env python -m py_compile nanopilot/test_thrust.py

./scripts/test_thrust.sh --airframe=80mm --mode=lateral-coast \
  --airborne-hover=1475 --airborne-climb=1500 --airborne-climb-s=1.5 \
  --lateral-roll-pwm=1700 --lateral-accel-s=1.5 --lateral-coast-s=6
```

Result from gentler 80mm run:
- Peak sampled XY speed after leveling: `3.200 m/s`
- Final XY speed after `6.3s`: `3.033 m/s`
- XY drift while leveled: `19.344 m`
- Speed half-life: `> 6.0s`

Plan documented above under `### Planned: translational drag / lateral coast fix`. Start by tuning `base_link` linear `velocity_decay` in the external aeroloop model SDFs, then validate with `--mode=lateral-coast` and re-check hover behavior. Do not start with active horizontal velocity hold; first make the physics bleed velocity.

## Current Handoff (2026-05-16 translational drag selected)

Completed the lateral-drag tuning pass. Final selected coefficient is `0.25 1/s`, applied consistently to both external models:
- `~/dev/aeroloop_gazebo/models/betaloop_drone_cam_65mm/model.sdf`
- `~/dev/aeroloop_gazebo/models/betaloop_drone_cam_80mm/model.sdf`

Important implementation detail:
- The original SDF-only plan did not work under the current Bullet/Harmonic stack. Sweeping `base_link` `<velocity_decay><linear>` through `0.05`, `0.10`, `0.20`, and `0.35` produced no meaningful lateral speed decay.
- Added a narrow fallback to `~/dev/aeroloop_gazebo/plugins/BetaflightPlugin.cc`: it reads `<horizontalVelocityDecay>` from the model plugin block and applies horizontal drag as `F = -m * k * v_xy` on the canonical link. The base-link `<velocity_decay><linear>` is kept at the same value for documentation/SDF intent, but the measured effect comes from the plugin fallback.
- Angular decay remains `0.0`.

80mm lateral-coast command used for all sweep runs:
```bash
./scripts/test_thrust.sh --airframe=80mm --mode=lateral-coast \
  --airborne-hover=1475 --airborne-climb=1500 --airborne-climb-s=1.5 \
  --lateral-roll-pwm=1700 --lateral-accel-s=1.5 --lateral-coast-s=6
```

Baseline before drag:
- Peak sampled XY speed after leveling: `3.200 m/s`
- Final XY speed after `6.3s`: `3.033 m/s`
- XY drift while leveled: `19.344 m`
- Speed half-life: `> 6.0s`

SDF-only sweep, before plugin fallback:
- `0.05`: peak `3.239 m/s`, final `2.995 m/s`, drift `19.545 m`, half-life `>6s`
- `0.10`: peak `3.147 m/s`, final `3.105 m/s`, drift `19.436 m`, half-life `>6s`
- `0.20`: peak `3.195 m/s`, final `3.146 m/s`, drift `19.181 m`, half-life `>6s`
- `0.35`: peak `3.265 m/s`, final `3.042 m/s`, drift `19.398 m`, half-life `>6s`

Plugin-backed tuning results:
- `0.35`: peak `2.695 m/s`, final `0.378 m/s` after `6.7s`, drift `7.995 m`, half-life `2.67s`
- `0.20`: peak `2.667 m/s`, final `0.903 m/s` after `6.8s`, drift `11.348 m`, half-life `4.57s`
- Selected `0.25`: peak `2.582 m/s`, final `0.662 m/s` after `6.8s`, drift `9.719 m`, half-life `3.74s`

Hover/in-flight check after selecting `0.25`:
```bash
./scripts/test_thrust.sh --airframe=80mm --mode=inflight-hover \
  --airborne-climb=1500 --airborne-climb-s=1.5 \
  --hover-low=1450 --hover-high=1510 --hover-step=5
```
Result caveat: the sweep was not useful for hover tuning because all samples were climbing; closest-to-hover was reported at `PWM=1510` with `vz=+3.216 m/s`, and the script reported no zero crossing. The new drag force is horizontal-only, so this should not be caused by vertical drag, but hover tuning still needs a better bounded check/range.

Validation completed:
```bash
uv run --env-file .env python -m py_compile nanopilot/test_thrust.py
bash -n scripts/test_thrust.sh scripts/start.sh scripts/stop_all.sh
cmake --build /Users/Dan/dev/aeroloop_gazebo/plugins/build
gz sdf -k /Users/Dan/dev/aeroloop_gazebo/models/betaloop_drone_cam_65mm/model.sdf
gz sdf -k /Users/Dan/dev/aeroloop_gazebo/models/betaloop_drone_cam_80mm/model.sdf
```

## Current Handoff (2026-05-16 vertical hover investigation)

Investigated the "all tested hover PWMs climb" caveat from the lateral-drag pass.

Finding:
- The previous command used `--hover-low=1450 --hover-high=1510 --hover-step=5` after a stronger climb. Because `run_inflight_hover_sweep` sweeps high-to-low, the early high-PWM samples injected a lot of upward momentum, and later low-PWM samples still measured climb while the vehicle coasted upward.
- The script also allowed ground-contact samples into the hover summary. Once the drone hits the ground, `z` is clamped and apparent `vz` can look like zero, which can corrupt the closest-to-hover and interpolation results.
- Angle vs Acro did not materially change the hover result, so this does not look like a BetaFlight altitude-controller/PID issue. The test uses direct CH3 motor throttle; no BF altitude controller is active.

Code changed:
- `nanopilot/test_thrust.py`: `run_inflight_hover_sweep` now ignores samples below `0.25m` for summary/interpolation and only interpolates zero-vz from adjacent in-flight samples.
- `nanopilot/drone_config_80mm.py`: kept `DRONE_HOVER_THROTTLE = 1475`, updated `DRONE_ALTITUDE_HOLD_K = 45.0`.

Commands/results:
```bash
./scripts/test_thrust.sh --airframe=80mm --mode=inflight-hover \
  --airborne-climb=1500 --airborne-climb-s=1.0 \
  --hover-low=1150 --hover-high=1450 --hover-step=25
```
- `1450`: `vz=-0.094 m/s`, `alt=0.95m`
- `1425`: `vz=-0.201 m/s`, then too low
- This showed the real hover region was not above `1510`; the earlier caveat was a bad sweep setup.

Focused Angle sweep after fixing summary logic:
```bash
./scripts/test_thrust.sh --airframe=80mm --mode=inflight-hover \
  --airborne-climb=1500 --airborne-climb-s=1.0 \
  --hover-low=1435 --hover-high=1480 --hover-step=5
```
- `1480`: `vz=+1.143 m/s`
- `1475`: `vz=+1.058 m/s`
- `1470`: `vz=+0.691 m/s`
- `1465`: `vz=-0.024 m/s`
- `1460`: `vz=-1.081 m/s`
- Corrected summary: closest-to-hover `PWM=1465`; interpolated zero-vz `PWM=1465.2`.

Acro comparison:
```bash
./scripts/test_thrust.sh --airframe=80mm --mode=inflight-hover --acro \
  --airborne-climb=1500 --airborne-climb-s=1.0 \
  --hover-low=1435 --hover-high=1480 --hover-step=5
```
- Closest-to-hover `PWM=1465`; interpolated zero-vz `PWM=1465.3`.
- This matches Angle mode closely, so attitude mode is not the cause.

Runtime damper checks:
```bash
./scripts/test_thrust.sh --airframe=80mm --mode=damper-sim \
  --airborne-hover=1465 --airborne-climb=1500 --airborne-climb-s=1.0 \
  --damper-sample-s=8
```
- Failed: landed after several seconds; after-settle drift `-1.203m`, `vz_mean=-0.159m/s`.

```bash
./scripts/test_thrust.sh --airframe=80mm --mode=damper-sim \
  --airborne-hover=1470 --airborne-climb=1500 --airborne-climb-s=1.0 \
  --damper-sample-s=8
```
- Failed: landed after several seconds; after-settle drift `-1.323m`, `vz_mean=-0.174m/s`.

```bash
./scripts/test_thrust.sh --airframe=80mm --mode=damper-sim \
  --airborne-hover=1475 --airborne-climb=1500 --airborne-climb-s=1.0 \
  --damper-sample-s=8
```
- Nearly passed: total drift `+0.143m`; after-settle drift `-0.507m`, `vz_mean=-0.067m/s`.

```bash
./scripts/test_thrust.sh --airframe=80mm --mode=damper-sim \
  --airborne-hover=1475 --airborne-climb=1500 --airborne-climb-s=1.0 \
  --damper-k=45 --damper-sample-s=8
```
- Passed: total drift `+0.082m`; after-settle drift `-0.416m`, `alt_p2p=0.436m`, `vz_mean=-0.056m/s`, `ch3_mean=1477.2`.

Conclusion:
- Do not retune `motorConstant` from this caveat alone. The 80mm thrust model is close enough for the current controller.
- The real fix was to correct the hover test summary and raise the 80mm vertical damper gain from `30` to `45`.
- `1465` is the raw zero-vz estimate in a dynamic high-to-low sweep, but `1475` remains the better runtime hover bias with the current damper loop.

Validation:
```bash
uv run --env-file .env python -m py_compile \
  nanopilot/test_thrust.py nanopilot/drone_config_80mm.py \
  nanopilot/drone_env.py nanopilot/drone_gym.py
bash -n scripts/test_thrust.sh scripts/start.sh scripts/stop_all.sh
```

## Current Handoff (2026-05-16 angular drag)

Added configurable angular drag on top of the translational-drag work.

Implementation:
- External SDFs now set `base_link` `<velocity_decay><angular>0.50</angular>`.
- External SDF Betaflight plugin blocks now set `<angularVelocityDecay>0.50</angularVelocityDecay>`.
- `~/dev/aeroloop_gazebo/plugins/BetaflightPlugin.cc` reads `angularVelocityDecay` and applies first-order angular damping:
  - `torque = -k * I_world * omega_world`
  - This mirrors the earlier horizontal plugin-backed drag because Bullet/Harmonic did not visibly honor SDF `velocity_decay` for linear damping.
- `nanopilot/test_thrust.py --mode=attitude` now passes through `--yaw-pwm` and `--airborne-phase-s`; previously attitude mode silently used its defaults.

Selected angular value:
- `angularVelocityDecay = 0.50 1/s`
- Applied consistently to 65mm and 80mm.
- The value is conservative: it did not materially reduce commanded yaw authority, but it adds passive rotational damping and slightly reduces already-small pitch/roll shake.

Validation and measurements:
```bash
cmake --build /Users/Dan/dev/aeroloop_gazebo/plugins/build
uv run --env-file .env python -m py_compile nanopilot/test_thrust.py
gz sdf -k /Users/Dan/dev/aeroloop_gazebo/models/betaloop_drone_cam_65mm/model.sdf
gz sdf -k /Users/Dan/dev/aeroloop_gazebo/models/betaloop_drone_cam_80mm/model.sdf
```

Baseline before angular drag, attitude test effectively used `yaw_pwm=1530`:
```bash
./scripts/test_thrust.sh --airframe=80mm --mode=attitude \
  --airborne-hover=1475 --airborne-climb=1500 --airborne-climb-s=1.0 \
  --airborne-phase-s=4 --yaw-pwm=1540
```
- Due to missing argument plumbing, the run used `yaw=1530`, `sample_s=5.0`.
- Baseline yaw-on roll: `std=0.01°`, `p2p=0.03°`
- Baseline yaw-on pitch: `std=0.01°`, `p2p=0.06°`
- Baseline yaw rate: `-6.8°/s`

With angular `0.10`, normal yaw-cap test:
- `yaw=1530`: roll `std=0.00°`, pitch `std=0.01°`, yaw rate `-6.9°/s`
- `yaw=1600`: roll `std=0.01°`, pitch `std=0.01°`, yaw rate `-41.2°/s`
- Conclusion: stable, but too small to have much visible effect.

With selected angular `0.50`:
```bash
./scripts/test_thrust.sh --airframe=80mm --mode=attitude \
  --airborne-hover=1475 --airborne-climb=1500 --airborne-climb-s=1.0 \
  --airborne-phase-s=5 --yaw-pwm=1600
```
- Roll `std=0.00°`, `p2p=0.02°`
- Pitch `std=0.00°`, `p2p=0.02°`
- Yaw rate `-41.0°/s`

Normal yaw-cap check:
```bash
./scripts/test_thrust.sh --airframe=80mm --mode=attitude \
  --airborne-hover=1475 --airborne-climb=1500 --airborne-climb-s=1.0 \
  --airborne-phase-s=5 --yaw-pwm=1530
```
- Roll `std=0.00°`, `p2p=0.01°`
- Pitch `std=0.01°`, `p2p=0.04°`
- Yaw rate `-7.2°/s`

Regression checks with angular `0.50`:
```bash
./scripts/test_thrust.sh --airframe=80mm --mode=damper-sim \
  --airborne-hover=1475 --airborne-climb=1500 --airborne-climb-s=1.0 \
  --damper-k=45 --damper-sample-s=8
```
- Passed: total drift `+0.112m`; after-settle drift `-0.428m`; `alt_p2p=0.451m`; `vz_mean=-0.057m/s`.

```bash
./scripts/test_thrust.sh --airframe=80mm --mode=lateral-coast \
  --airborne-hover=1475 --airborne-climb=1500 --airborne-climb-s=1.5 \
  --lateral-roll-pwm=1700 --lateral-accel-s=1.5 --lateral-coast-s=6
```
- Peak `2.729 m/s`, final `0.736 m/s` after `6.6s`, drift `10.234m`, half-life `3.89s`.
- This is close to the selected translational-drag result and keeps the lateral fix intact.

## Current Handoff (2026-05-16 640x480 camera pipeline)

Kicked off the 640x480 resolution project.

Implementation:
- `nanopilot/drone_config_80mm.py`: `IMAGE_W = 640`, `IMAGE_H = 480`.
- `nanopilot/drone_config_65mm.py`: `IMAGE_W = 640`, `IMAGE_H = 480`.
- External Gazebo camera SDFs already render `640x480` at 30Hz for both 65mm and 80mm, so no external SDF edit was needed.
- `README.md` now documents that current `LinearModel` uses adaptive global pooling, so higher input resolution does not increase model parameter count, but it can increase render, JPEG decode, transform, memory, and tub-storage costs.

Validation:
```bash
uv run --env-file .env python -m py_compile \
  nanopilot/drone_config_80mm.py nanopilot/drone_config_65mm.py \
  nanopilot/drone_env.py nanopilot/drone_gym.py \
  nanopilot/gz_camera_worker.py nanopilot/torch_train.py \
  nanopilot/torch_model.py
```
- Passed.

Bounded training probe on old/local tub resized to 640:
```bash
uv run --env-file .env python nanopilot/torch_train.py \
  --tubs=data/tub_12_26-05-16 \
  --model=/private/tmp/nanopilot_640_probe.pth \
  --myconfig=drone_config_80mm.py \
  --max-epochs=1 --max-samples=60 --batch-size=16 --no-model-summary
```
- MPS, 60 samples, 48 train / 12 val, one epoch: `4.8s`.

Live headless collection check:
```bash
GZ_HEADLESS=1 ./scripts/start.sh --no-manage --airframe=80mm
GZ_WORLD=baylands_80mm uv run --env-file .env python nanopilot/autonomous_collect.py \
  --airframe=80mm --duration=3 --warmup=1 --rate-hz=10 --ready-timeout=30
bash ./scripts/stop_all.sh
```
- Required running outside sandbox because process-list/uv-cache access was restricted.
- Camera worker logged nonblank frames: `shape=(480, 640, 3)`, `min=0 max=218 mean=106.1`.
- Created `data/tub_14_26-05-16` with 30 images, tub size `596K`.
- Sample recorded JPEG size verified as `(640, 480)`.

Bounded training probe on newly recorded 640 tub:
```bash
uv run --env-file .env python nanopilot/torch_train.py \
  --tubs=data/tub_14_26-05-16 \
  --model=/private/tmp/nanopilot_tub14_640_probe.pth \
  --myconfig=drone_config_80mm.py \
  --max-epochs=1 --max-samples=28 --batch-size=8 --no-model-summary
```
- MPS, 28 samples, 22 train / 6 val, one epoch: `5.0s`.

Notes:
- First collector attempt subscribed to stale/default `/world/drone_course_80mm/...` while the running default world was `baylands_80mm`; setting `GZ_WORLD=baylands_80mm` fixed the topic.
- 720p recording/downsample remains a separate follow-up. The current model can resize before training, but Gazebo render cost and JPEG I/O/storage should be measured separately.

## Current Handoff (2026-05-17 horizontal damping)

Applied the README TODO to increase horizontal damping from `0.25` to `0.35` for both external Gazebo airframes:
- `~/dev/aeroloop_gazebo/models/betaloop_drone_cam_65mm/model.sdf`
- `~/dev/aeroloop_gazebo/models/betaloop_drone_cam_80mm/model.sdf`

Both files now set:
- `base_link` `<velocity_decay><linear>0.35</linear>`
- BetaflightPlugin `<horizontalVelocityDecay>0.35</horizontalVelocityDecay>`

Angular damping was left unchanged at `0.50`.

## Current Handoff (2026-05-17 pitch altitude compensation)

Investigated 80mm altitude loss during forward flight with pitch applied and neutral altitude.

Reproduction with no pitch feed-forward:
```bash
./scripts/test_thrust.sh --airframe=80mm --mode=damper-sim \
  --airborne-hover=1475 --airborne-climb=1500 --airborne-climb-s=1.0 \
  --damper-k=45 --damper-sample-s=8 \
  --damper-pitch-after=3 --damper-pitch-pwm=1700
```
- After pitch started, altitude dropped from about `1.11m` to ground contact around `0.50m`.
- Summary: after-settle drift `-0.546m`, `vz_mean=-0.074m/s`, failed the damper check.

Baseline with no pitch input still passed but was near the edge:
- after-settle drift `-0.423m`, `vz_mean=-0.058m/s`.

Fix:
- Added `DRONE_PITCH_THROTTLE_FEEDFORWARD`, passed from config through `drone_env.py` into `DroneGymEnv`.
- Runtime applies it only when altitude input is in the neutral/deadband region, proportional to `abs(throttle)` (pitch command), so it does not fight deliberate climb/descend commands.
- Added matching `test_thrust.py --damper-pitch-ff` and `--damper-pitch-cap` knobs for validation.
- Tuned 80mm to `DRONE_PITCH_THROTTLE_FEEDFORWARD = 8.0`.
- Left 65mm at `0.0` until separately validated.

Validation:
```bash
./scripts/test_thrust.sh --airframe=80mm --mode=damper-sim \
  --airborne-hover=1475 --airborne-climb=1500 --airborne-climb-s=1.0 \
  --damper-k=45 --damper-sample-s=8 \
  --damper-pitch-after=3 --damper-pitch-pwm=1700 \
  --damper-pitch-ff=8 --damper-pitch-cap=200
```
- Passed: after-settle drift `-0.106m`, `alt_p2p=0.130m`, `vz_mean=-0.013m/s`, `ch3_mean=1481.0`.
- Trial value `25` was too high and caused a strong climb, so keep `8.0` unless retuning for a different pitch sensitivity or airframe.

## Current Handoff (2026-05-17 web config display)

Surfaced read-only flight config values in the web UI BetaFlight panel:
- max pitch angle
- max roll angle
- hover PWM
- throttle range
- input sensitivity
- max yaw rate
- yaw PWM cap
- yaw throttle feed-forward
- pitch throttle feed-forward
- altitude hold K

Implementation:
- `nanopilot/drone_manage.py` stores a JSON-safe config snapshot on `LocalWebController` and sends it as `{"config": ...}` when a websocket client opens.
- `nanopilot/templates/static/main.js` renders the config in a compact grid under the RC values.
- The same `main.js` was synced to `~/dev/donkeycar-fork/donkeycar/parts/web_controller/templates/static/main.js` for the runtime static path noted above.

## Current Handoff (2026-05-17 quadratic horizontal drag)

Implemented additive quadratic horizontal drag in the external aeroloop plugin:
- `~/dev/aeroloop_gazebo/plugins/BetaflightPlugin.cc`
- New SDF param: `<horizontalQuadraticDrag>`
- Formula in `ApplyHorizontalVelocityDecay`: `F_xy = -m * (horizontalVelocityDecay + horizontalQuadraticDrag * |v_xy|) * v_xy`

Updated both external model SDFs:
- `~/dev/aeroloop_gazebo/models/betaloop_drone_cam_65mm/model.sdf`
- `~/dev/aeroloop_gazebo/models/betaloop_drone_cam_80mm/model.sdf`

Both now set:
- `<horizontalVelocityDecay>0.35</horizontalVelocityDecay>`
- `<horizontalQuadraticDrag>0.05</horizontalQuadraticDrag>`
- `<angularVelocityDecay>0.50</angularVelocityDecay>`

Rebuilt the plugin:
```bash
cd ~/dev/aeroloop_gazebo/plugins/build
make
```

Validation:
```bash
./scripts/test_thrust.sh --airframe=80mm --mode=lateral-coast \
  --airborne-hover=1475 --airborne-climb=1500 --airborne-climb-s=1.5 \
  --lateral-roll-pwm=1700 --lateral-accel-s=1.5 --lateral-coast-s=6
```
- Peak XY speed `2.466 m/s`, final `0.242 m/s` after `6.9s`, drift `6.297m`, half-life `2.23s`.
- Note: the lateral-coast maneuver still climbed significantly in Z, so future work may need roll/pitch tilt throttle compensation similar to the pitch-forward feed-forward.

Pitch-forward altitude compensation still passed after this change:
```bash
./scripts/test_thrust.sh --airframe=80mm --mode=damper-sim \
  --airborne-hover=1475 --airborne-climb=1500 --airborne-climb-s=1.0 \
  --damper-k=45 --damper-sample-s=8 \
  --damper-pitch-after=3 --damper-pitch-pwm=1700 \
  --damper-pitch-ff=8 --damper-pitch-cap=200
```
- Passed: after-settle drift `-0.097m`, `alt_p2p=0.124m`, `vz_mean=-0.012m/s`.
