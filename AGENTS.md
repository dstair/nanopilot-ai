# AGENTS.md

Concise runbook for coding agents working in this repo. Use `README.md` for
project overview, setup details, and public-facing docs. Keep this file focused
on agent workflow, sharp edges, and the current handoff.

## Fresh Session Checklist

1. Read the newest `## Current Handoff` section at the bottom.
2. Check local changes before editing:
   ```bash
   git status --short --branch
   git diff --stat
   ```
3. Avoid full/default training unless explicitly requested. Prefer bounded flags:
   `--max-epochs`, `--max-samples`, `--batch-size`, `--no-model-summary`.
4. Stop sim processes before and after integration tests:
   ```bash
   bash ./scripts/stop_all.sh
   ```
5. Do not edit `nanopilot/config.py` for local tuning. Edit
   `nanopilot/drone_config_65mm.py` or `nanopilot/drone_config_80mm.py`.
6. Web UI static files are served from the DonkeyCar fork, not this repo copy:
   `~/dev/donkeycar-fork/donkeycar/parts/web_controller/templates/static/main.js`.

## Project Essentials

Nanopilot adapts DonkeyCar to fly a simulated quadrotor using BetaFlight SITL,
Gazebo Harmonic, and a PyTorch CNN/GRU model trained from camera, IMU, and
previous-control data. It is developed for native Apple Silicon.

DonkeyCar key names map to drone controls as:

| Key | Meaning |
| --- | --- |
| `steering [-1, 1]` | yaw rate |
| `throttle [-1, 1]` | forward pitch / tilt |
| `roll [-1, 1]` | lateral roll / bank |
| `altitude [-1, 1]` | motor throttle around hover PWM |

The current model output is four-axis: `[steering, throttle, roll, altitude]`.
Older three-output checkpoints are not runtime-compatible with the current
`LinearModel`.

## Common Commands

```bash
# Install/update Python deps
uv sync

# Launch manual drive, default 80mm Pavo Pico II airframe
./scripts/start.sh

# Launch 65mm Air65 profile
./scripts/start.sh --airframe=65mm

# Launch with autopilot model
./scripts/start.sh --model=models/pilot.pth

# Start only Gazebo + BetaFlight, useful for test_thrust.py and collectors
./scripts/start.sh --no-manage --airframe=80mm

# Stop everything
bash ./scripts/stop_all.sh

# One-command hover/thrust integration test
./scripts/test_thrust.sh --airframe=80mm

# Scripted no-human collection + bounded training
./scripts/collect_train.sh --airframe=80mm --duration=30 --max-epochs=5 \
  --model=models/scripted_autonomous.pth

# Bounded smoke training run
uv run --env-file .env python nanopilot/torch_train.py \
  --tubs=data/tub_3_26-05-09 \
  --model=models/hello_world_tub3.pth \
  --max-epochs=1 --max-samples=400 --batch-size=64 --no-model-summary

# Evaluate a model
uv run --env-file .env python nanopilot/evaluate.py \
  --tubs=data/tub_4_26-05-09 \
  --model=models/hello_world_tub3.pth

# Verify gz-python bindings
uv run --env-file .env python -c "import gz.transport13; print('OK')"
```

Web UI: http://127.0.0.1:8887

## Verification

There is no formal pytest suite. Use targeted checks.

```bash
uv run --env-file .env python -m py_compile \
  nanopilot/drone_gym.py nanopilot/drone_env.py nanopilot/gz_telemetry.py \
  nanopilot/tub_schema.py nanopilot/drone_manage.py \
  nanopilot/autonomous_collect.py nanopilot/dataset.py \
  nanopilot/torch_model.py nanopilot/torch_pilot.py \
  nanopilot/torch_train.py nanopilot/evaluate.py nanopilot/smoke_test.py \
  nanopilot/test_thrust.py

uv run --env-file .env python nanopilot/torch_train.py --help
uv run --env-file .env python nanopilot/evaluate.py --help
uv run --env-file .env python nanopilot/autonomous_collect.py --help

uv run --env-file .env python - <<'PY'
import numpy as np
from nanopilot.torch_pilot import TorchPilot
pilot = TorchPilot(input_shape=(3, 480, 640), seq_len=3)
pilot.load('models/tub_5_26-05-17.pth')
print(tuple(round(float(v), 6) for v in pilot.run(np.zeros((480, 640, 3), dtype=np.uint8))))
PY
```

Headless collection smoke:

```bash
GZ_HEADLESS=1 ./scripts/start.sh --no-manage --airframe=80mm
GZ_WORLD=baylands_80mm uv run --env-file .env python nanopilot/autonomous_collect.py \
  --airframe=80mm --duration=3 --warmup=1 --rate-hz=10 --ready-timeout=30
bash ./scripts/stop_all.sh
```

## Environment

- Python must be `3.12.*`.
- Use `uv`, not `pip`.
- `.env` at repo root is required for `PYTHONPATH`, `DYLD_LIBRARY_PATH`, and
  `GZ_IP`.
- `gz-python` is not on PyPI. It comes from `brew install gz-harmonic` into
  Homebrew site-packages.
- Gazebo commands must run under ARM64 Homebrew/Ruby, not Rosetta.

## Architecture Notes

Pipeline:

```text
Web UI / Xbox -> LocalWebController -> DriveMode -> DroneGymEnv
  -> RC UDP 9004 -> BetaFlight SITL -> Gazebo bridge plugin
  -> Gazebo camera + IMU -> gz_camera_worker shared memory / telemetry
  -> TorchPilot / TubWriter
```

Key design constraints:

- `gz_camera_worker.py` is a subprocess to avoid protobuf conflicts.
- Camera frames are passed through POSIX shared memory.
- `altitude` is direct motor throttle around hover PWM, not an altitude PID.
- `TorchPilot` keeps IMU history and feeds previous model outputs as
  `prev_ctrl`.
- `TubDataset` expects image + 6-axis IMU + previous controls + four-axis target.

BetaFlight/Gazebo ports:

| Port | Direction | Purpose |
| --- | --- | --- |
| `9002` | SITL -> Gazebo | motor speeds |
| `9003` | Gazebo -> SITL | FDM/IMU/pose state |
| `9004` | Nanopilot -> SITL | RC channel packets |
| `5761` | TCP | MSP/configurator |

RC packet format: `struct.pack('<d', timestamp)` plus 16 little-endian
`uint16` channels.

## Key Files

| Path | Purpose |
| --- | --- |
| `nanopilot/drone_manage.py` | main DonkeyCar vehicle entrypoint |
| `nanopilot/drone_gym.py` | BetaFlight RC loop, camera bridge, reset/control logic |
| `nanopilot/drone_env.py` | shared `DroneGymEnv` construction |
| `nanopilot/gz_telemetry.py` | Gazebo pose/IMU helpers |
| `nanopilot/tub_schema.py` | tub input/type schema and IMU keys |
| `nanopilot/dataset.py` | PyTorch tub dataset loader |
| `nanopilot/torch_model.py` | current `LinearModel` |
| `nanopilot/torch_pilot.py` | runtime inference wrapper |
| `nanopilot/torch_train.py` | training CLI |
| `nanopilot/evaluate.py` | checkpoint evaluator/comparison CLI |
| `nanopilot/autonomous_collect.py` | headless scripted collector |
| `nanopilot/test_thrust.py` | hover, damper, attitude, lateral-coast test logic |
| `scripts/start.sh` | stack launcher |
| `scripts/stop_all.sh` | stack cleanup |
| `scripts/test_thrust.sh` | launches stack, runs `test_thrust.py`, tears down |
| `worlds/*_80mm.sdf` | 80mm worlds |
| `worlds/drone_course_65mm.sdf` | retained 65mm Air65 world |

External paths that matter:

| Path | Purpose |
| --- | --- |
| `~/dev/aeroloop_gazebo/plugins/BetaflightPlugin.cc` | Gazebo/BetaFlight bridge, drag implementation |
| `~/dev/aeroloop_gazebo/plugins/build/libBetaflightPlugin.dylib` | built plugin loaded by Gazebo |
| `~/dev/aeroloop_gazebo/models/betaloop_drone_cam_65mm/model.sdf` | 65mm airframe |
| `~/dev/aeroloop_gazebo/models/betaloop_drone_cam_80mm/model.sdf` | 80mm airframe |
| `~/dev/betaflight/obj/main/betaflight_SITL.elf` | BetaFlight SITL binary |
| `~/dev/donkeycar-fork/donkeycar/parts/web_controller/templates/static/main.js` | runtime Web UI JS |

## Flight/Sim Gotchas

- `start.sh` defaults to 80mm and `baylands_80mm`. Use
  `--airframe=65mm` for Air65. Use `GZ_WORLD=<world>` for alternate 80mm worlds.
- Camera topic must match both world and model name. Stale gz-transport topics
  can linger after Gazebo exits.
- On macOS, Gazebo server and GUI are separate processes. `GZ_HEADLESS=1`
  skips GUI.
- World SDFs with any world plugin must explicitly include all required plugins:
  Physics, UserCommands, SceneBroadcaster, Sensors, and Imu.
- BetaFlight requires a disarm phase before arm. Ad-hoc RC senders must send
  AUX1 low before high, or motors stay at zero.
- `eeprom.bin` is written in the process cwd and is gitignored. Delete it to
  reset BetaFlight config.
- Shared-memory `resource_tracker` warnings on shutdown are usually cosmetic.
- `scripts/stop_all.sh` also cleans up XboxBridge and `/tmp/nanopilot_xbox.sock`.

## Current Airframe/Control State

- 80mm Pavo Pico II is the default airframe, modeled at about 79g AUW.
- 65mm Air65 remains available for comparison.
- Current image resolution is `640x480` in both airframe configs.
- 80mm hover bias remains `DRONE_HOVER_THROTTLE = 1475`.
- 80mm vertical damper: `DRONE_ALTITUDE_HOLD_K = 45.0`.
- 80mm pitch throttle feed-forward: `DRONE_PITCH_THROTTLE_FEEDFORWARD = 8.0`.
- 65mm pitch feed-forward is still `0.0` unless separately validated.
- External model/plugin drag currently:
  - `<horizontalVelocityDecay>0.35</horizontalVelocityDecay>`
  - `<horizontalQuadraticDrag>0.05</horizontalQuadraticDrag>`
  - `<angularVelocityDecay>0.50</angularVelocityDecay>`

RC channels use AETR:

| Channel | Meaning |
| --- | --- |
| CH1 | roll |
| CH2 | pitch |
| CH3 | motor throttle around hover |
| CH4 | yaw |
| CH5/AUX1 | arm |
| CH6/AUX2 | Angle mode |

Keyboard/web controls:

- Up/down: altitude / motor throttle
- Left/right: yaw
- `j`/`l`: roll

Xbox controls:

- Left stick X: yaw
- Left stick Y: altitude
- Right stick X: roll
- Right stick Y: pitch
- RT: explicit arm/deadman
- LT: one-shot pose reset
- B: mode cycle

## Data And Models

Useful local data:

- `data/tub_3_26-05-09`: best older training tub, `3368` rows/images,
  complete nonzero IMU, but old data has roll as zero.
- `data/tub_4_26-05-09`: small held-out smoke/eval tub, `161` rows/images.
- `data/tub_14_26-05-16`: 640x480 scripted 80mm collection smoke, `30` images.

Useful local models:

- `models/tub_5_26-05-17.pth`: current four-output checkpoint that loads with
  `TorchPilot` at `640x480`, `seq_len=3`.
- `models/tub_5_26-05-17_seqlen7.pth`: four-output checkpoint trained with a
  longer sequence length.
- `models/hello_world_tub3.pth`: older bounded smoke checkpoint from `tub_3`;
  useful historically, but not compatible with current four-output runtime.
- `models/tub_3_26-05-09_full.pth`: full older 3-axis-era tub training result;
  no longer compatible after four-output roll change unless retrained/migrated.
- `models/pilot.pth`: older baseline.

## Current Handoff (2026-05-17)

Latest completed work: quadratic horizontal drag.

Implemented additive quadratic horizontal drag in the external aeroloop plugin:

- `~/dev/aeroloop_gazebo/plugins/BetaflightPlugin.cc`
- New SDF param: `<horizontalQuadraticDrag>`
- Formula: `F_xy = -m * (horizontalVelocityDecay + horizontalQuadraticDrag * |v_xy|) * v_xy`

Updated both external model SDFs:

- `~/dev/aeroloop_gazebo/models/betaloop_drone_cam_65mm/model.sdf`
- `~/dev/aeroloop_gazebo/models/betaloop_drone_cam_80mm/model.sdf`

Both now set:

- `<horizontalVelocityDecay>0.35</horizontalVelocityDecay>`
- `<horizontalQuadraticDrag>0.05</horizontalQuadraticDrag>`
- `<angularVelocityDecay>0.50</angularVelocityDecay>`

Plugin was rebuilt with:

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

Result: peak XY speed `2.466 m/s`, final `0.242 m/s` after `6.9s`, drift
`6.297m`, half-life `2.23s`.

Pitch-forward altitude compensation still passed after drag change:

```bash
./scripts/test_thrust.sh --airframe=80mm --mode=damper-sim \
  --airborne-hover=1475 --airborne-climb=1500 --airborne-climb-s=1.0 \
  --damper-k=45 --damper-sample-s=8 \
  --damper-pitch-after=3 --damper-pitch-pwm=1700 \
  --damper-pitch-ff=8 --damper-pitch-cap=200
```

Result: passed with after-settle drift `-0.097m`, `alt_p2p=0.124m`,
`vz_mean=-0.012m/s`.

Follow-ups:

1. Lateral-coast maneuver still climbs significantly in Z. Consider roll/pitch
   tilt throttle compensation similar to pitch-forward feed-forward.
2. Collect new roll-inclusive training tubs before training a serious autopilot;
   old tubs load roll as `0.0`.
3. If AGENTS.md needs historical handoff detail, recover it from git history
   rather than appending every session log forever.
