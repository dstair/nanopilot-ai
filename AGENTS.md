# AGENTS.md

Concise runbook for coding agents working in this repo. Use `README.md` for
project overview, setup details, and public-facing docs. Keep this file focused
on agent workflow, sharp edges, and the current handoff.

**Maintenance.** This file is loaded into context every session, so it has a
budget: keep it under ~150k characters. When adding a handoff, fold the
now-superseded one into `## Historical Record` — keep the decision and any
lesson that still binds, drop the blow-by-blow (git history has it). Before
committing, check `wc -c AGENTS.md`, and check you have not pasted a section
that already exists: this file once carried the same 197-line block **eight
times**, which was a third of its total size.

## Fresh Session Checklist

1. Read the newest `## Current Handoff` section at the bottom.
1b. **Check whether a long run is in progress** before touching the sim or
   starting anything that needs BetaFlight/Gazebo — any stack command will
   fight it for UDP 9004:
   ```bash
   pgrep -fl 'rl_train.sh|rl_collect|dagger_collect|eval_flight|torch_train'
   ```
   If one is alive, do NOT run `stop_all.sh`, `start.sh`, or any flight
   script until you have checked with the user.
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

| Key | Meaning under `DRONE_VELOCITY_CONTROL` (80mm) | Meaning in direct mode (65mm) |
| --- | --- | --- |
| `steering [-1, 1]` | yaw rate | yaw rate |
| `throttle [-1, 1]` | forward speed, fraction of `DRONE_MAX_FORWARD_SPEED` | forward pitch / tilt |
| `roll [-1, 1]` | rightward speed, fraction of `DRONE_MAX_LATERAL_SPEED` | lateral roll / bank |
| `altitude [-1, 1]` | LINEAR height trim, in metres of `DRONE_ALTITUDE_TRIM_RANGE` | motor throttle around hover PWM, via a 0.2-power curve |

The 80mm airframe sets `DRONE_VELOCITY_CONTROL = True`, so the env closes the
inner loop and the model regresses SETPOINTS, not stick deflections. 65mm does
not set it and is still direct-angle. `run_threaded(..., direct_thrust=X)`
bypasses the height hold with the old curve — every takeoff/landing ramp needs
it, because a ramp cannot be expressed as a height trim.

The current model output is four-axis: `[steering, throttle, roll, altitude]`.
Older three-output checkpoints are not runtime-compatible with the current
`LinearModel`.

`torch_train.py` now emits `<model>.ckpt` alongside `<model>.pth` on every run
(unless `--no-checkpoint`). The `.pth` stays a bare `state_dict` for
`TorchPilot`/`evaluate.py`, unchanged; the `.ckpt` holds full resumable
training state (model + optimizer + epoch + best-val) and is also a valid
source for `--init-weights`. Use `--resume=<model>.ckpt` to continue an
interrupted run, or `--init-weights=<path>.pth` to warm-start a fine-tune with
a fresh optimizer/epoch counter.

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

# One-command closed-loop autopilot eval: headless stack, TorchPilot flies
# N episodes, scored from Gazebo ground truth (survival/progress/gates), JSON out
./scripts/eval_flight.sh --model=models/bc_dagger1_heights.pth \
  --world=s_curve_heights_80mm --runs=5 --duration=60 --rate-hz=20
```

### The current pipeline: scripted expert -> BC -> DAgger

No RL run is needed. `GateExpert` is a pure function of privileged state, so it
needs no training, no checkpoint selection, and it labels any state on demand.

```bash
# 0. Bring the stack up, WAIT ~25s, then gate on MSP health before any long run
GZ_HEADLESS=1 GZ_WORLD=s_curve_heights_80mm ./scripts/start.sh \
  --no-manage --airframe=80mm
uv run --env-file .env python -c "
import sys; sys.path.insert(0,'nanopilot')
from utils.betaflight import MSPClient
print(MSPClient().query_status(timeout=2.0))"
# (None, [], None, []) means rebuild the stack NOW. Read the output; do not
# just run it. A dead board wastes the whole collection.

# 1. Sanity-check the expert on the course (no policy files needed)
uv run --env-file .env python scripts/gate_rollout.py --expert --episodes=5

# 2. Collect expert rollouts -> tub  (~100% yield, ~25 min for 40 episodes)
uv run --env-file .env python nanopilot/rl_collect.py \
  --expert --episodes=40 --min-gates=10

# 3. Train the student
uv run --env-file .env python nanopilot/torch_train.py \
  --tubs=data/<expert_tub> --model=models/bc_expert_vN.pth

# 4. DAgger: fly the STUDENT, label with the EXPERT. This is what makes the
#    student actually complete a course — an expert-only tub has zero
#    corrective data because the expert never fails.
uv run --env-file .env python nanopilot/dagger_collect.py \
  --expert --student=models/bc_expert_vN.pth --episodes=30

# 5. Retrain on BOTH tubs aggregated, then eval.
#    Judge a DAgger round by eval_flight ONLY, never by val loss.
uv run --env-file .env python nanopilot/torch_train.py \
  --tubs=data/<expert_tub>,data/<dagger_tub> --model=models/bc_dagger1_vN.pth
```

### Probes and diagnostics

```bash
# Altitude hold check: healthy = roughly constant z for 30s at neutral
uv run --env-file .env python scripts/probe_hover.py

# Altitude step response: how fast the hold tracks a MOVING target. probe_hover
# only ever tests a FIXED one, which says nothing about a varied-height course.
GZ_WORLD=s_curve_heights_80mm uv run --env-file .env python \
  scripts/probe_alt_step.py --amplitude=0.5

# Velocity cascade: signs, gains, cross-axis leakage, and a yaw-rate leg that
# FAILS on a wrong sign or a mis-scaled full-stick rate
uv run --env-file .env python scripts/probe_velocity.py

# Why is the policy missing gates? Per-crossing lateral/height/speed vs the
# pass window (needs a running stack)
uv run --env-file .env python scripts/gate_rollout.py --expert --episodes=6

# Scripted no-human collection + bounded training (direct-angle mode only)
./scripts/collect_train.sh --airframe=80mm --duration=30 --max-epochs=5 \
  --model=models/scripted_autonomous.pth

# Bounded smoke training run
uv run --env-file .env python nanopilot/torch_train.py \
  --tubs=data/tub_3_26-05-09 \
  --model=models/hello_world_tub3.pth \
  --max-epochs=1 --max-samples=400 --batch-size=64 --no-model-summary

# Fine-tune a good model on a new tub (lower LR, warm start; fresh optimizer/epoch)
uv run --env-file .env python nanopilot/torch_train.py \
  --tubs=data/<new_tub> --model=models/bc_v4.pth \
  --init-weights=models/bc_v3.pth --lr=0.0003 --max-epochs=20

# Resume an interrupted run (restores model + optimizer + epoch + best-val)
uv run --env-file .env python nanopilot/torch_train.py \
  --tubs=data/<tub> --model=models/bc_v4.pth --resume=models/bc_v4.ckpt

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
  nanopilot/test_thrust.py nanopilot/eval_flight.py \
  nanopilot/rl_env.py nanopilot/rl_train.py nanopilot/rl_collect.py \
  nanopilot/dagger_collect.py nanopilot/gate_expert.py nanopilot/line_expert.py \
  nanopilot/line_env.py \
  nanopilot/gz_reset_worker.py nanopilot/gz_camera_worker.py \
  scripts/gate_rollout.py scripts/ff_sweep.py scripts/detach_launch.py \
  scripts/promote_checkpoint.py scripts/alt_phase.py \
  scripts/probe_hover.py scripts/probe_velocity.py scripts/probe_alt_step.py

uv run --env-file .env python nanopilot/torch_train.py --help
uv run --env-file .env python nanopilot/evaluate.py --help
uv run --env-file .env python nanopilot/autonomous_collect.py --help

# TorchPilot load + forward. Note ALL THREE details: nanopilot/ on sys.path,
# the model built at 120x160 (NOT the 640x480 camera size), and run() taking
# velocity + yaw. Verified 2026-09-08; prints 4 floats.
uv run --env-file .env python - <<'PY'
import sys; sys.path.insert(0, 'nanopilot')
import numpy as np
from torch_pilot import TorchPilot
pilot = TorchPilot(input_shape=(3, 120, 160), seq_len=3)
pilot.load('models/bc_dagger1_heights.pth')
out = pilot.run(np.zeros((480, 640, 3), dtype=np.uint8), vel_x=1.0, yaw_deg=0.0)
print(tuple(round(float(v), 6) for v in out))
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
| `nanopilot/eval_flight.py` | closed-loop autopilot eval: survival/progress/gate metrics, JSON |
| `nanopilot/gate_expert.py` | scripted pure-pursuit gate expert over velocity setpoints |
| `nanopilot/line_expert.py` | scripted line-follow + landing expert; glide slope sized by camera FOV |
| `nanopilot/line_env.py` | `LineChaseEnv`: GateChaseEnv with landing termination, for the collectors |
| `nanopilot/rl_env.py` | `GateChaseEnv`: state-based Gymnasium env over a live DroneGymEnv |
| `nanopilot/rl_train.py` | PPO trainer for the state-based RL gate teacher |
| `nanopilot/rl_collect.py` | flies the expert (`--expert`) or an RL teacher, records tubs for BC distillation |
| `nanopilot/dagger_collect.py` | flies the STUDENT, labels each state with the expert — the DAgger collector |
| `nanopilot/gz_reset_worker.py` | persistent subprocess issuing gz set_pose teleports |
| `nanopilot/gz_camera_worker.py` | subprocess camera bridge; shared memory, avoids protobuf conflicts |
| `scripts/probe_hover.py` | takeoff + 30s neutral hold; altitude-hold check |
| `scripts/probe_alt_step.py` | altitude step response: overshoot, implied w_n/zeta |
| `scripts/probe_descent.py` | descent authority: ground effect, descent-rate lag, landing profile vs TRIM_RANGE |
| `scripts/probe_velocity.py` | velocity-cascade step response: signs, gains, cross-axis leakage |
| `scripts/gate_rollout.py` | flies the expert (`--expert`) or an RL policy; per-gate-crossing lateral/height/speed vs the pass window |
| `scripts/ff_sweep.py` | sweeps `DRONE_PITCH_THROTTLE_FEEDFORWARD` airborne, climb rate per pitch |
| `scripts/detach_launch.py` | `os.setsid()` launcher; long runs survive process-group kills |
| `scripts/promote_checkpoint.py` | promotes the newest RL checkpoint to `model.zip`/`vecnorm.pkl` |
| `scripts/alt_phase.py` | altitude period/phase analysis: damping vs position feedback vs lag |
| `scripts/start.sh` | stack launcher |
| `scripts/start_gated.sh` | stack launcher that REFUSES to hand over an MSP-deaf BetaFlight; rebuilds until healthy |
| `scripts/stop_all.sh` | stack cleanup |
| `scripts/test_thrust.sh` | launches stack, runs `test_thrust.py`, tears down |
| `scripts/eval_flight.sh` | launches stack, runs `eval_flight.py`, tears down |
| `scripts/rl_train.sh` | supervisor: launches stack, runs `rl_train.py`, rebuilds+resumes on failure |
| `worlds/s_curve_heights_80mm.sdf` | 10-gate varied-height S-curve; per-gate pass window. **Current benchmark** |
| `worlds/s_curve_gates_80mm.sdf` | 10-gate flat S-curve; same x/y/yaw, so height is the only difference |
| `worlds/slalom_gates_80mm.sdf` | retired 3-gate slalom; kept for historical comparison |
| `worlds/landing_target_80mm.sdf` | takeoff -> 20m guidance line -> landing pad at (20,0). **Next course**; see the 2026-09-09 plan |
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
- 65mm Air65 remains available for comparison, but has NOT been re-tested
  airborne since the pitch/roll PID retune or the yaw-scale correction.
- Camera is `640x480`; models train and infer at `TRAIN_IMAGE_H/W = 120x160`.
  **Build `TorchPilot`/`LinearModel` at 120x160, never at the camera size** —
  `AdaptiveAvgPool` accepts the wrong size silently and predicts badly.
- 80mm camera `horizontal_fov` is **2.44346 rad (140 deg)**, set in the model
  SDF outside this repo. Tubs recorded before that change are wrong optics.

80mm control values (all in `nanopilot/drone_config_80mm.py`; verified
2026-09-08 against the file):

| setting | value | note |
| --- | --- | --- |
| `DRONE_VELOCITY_CONTROL` | `True` | env closes the inner loop; model emits setpoints |
| `DRONE_HOVER_THROTTLE` | 1480 | re-measured after taming the BF PIDs |
| `DRONE_ALTITUDE_HOLD_K` | 90.0 | velocity term |
| `DRONE_ALTITUDE_HOLD_KP` | 30.0 | **position term — this is what makes the plant safe** |
| `DRONE_ALTITUDE_TRIM_RANGE` | **1.6** | metres of setpoint authority. Raised from 1.0 on 2026-09-09 so touchdown is in the action space; gate courses re-validated |
| `DRONE_ALTITUDE_HOLD_DEADBAND` | 0.1 | outside it the command is direct thrust |
| `DRONE_PITCH_THROTTLE_FEEDFORWARD` | 2.0 | swept airborne; higher values climb |
| `DRONE_YAW_THROTTLE_FEEDFORWARD` | 0.0 | correct — yaw-induced thrust is now ~0.3 PWM |
| `DRONE_MAX_PITCH_ANGLE` / `ROLL` | 20.0 / 10.0 | roll buys half the accel of pitch |
| `DRONE_MAX_YAW_RATE` | 120.0 | delivers 120 deg/s only since the 0.60 scale fix |
| `DRONE_MAX_FORWARD_SPEED` / `LATERAL` | 2.0 / 1.0 m/s | at full stick |
| `DRONE_VELOCITY_KP/KFF_FWD` | 0.6 / 0.35 | measured, do not guess |
| `DRONE_VELOCITY_KP/KFF_LAT` | 1.2 / 0.8 | lateral rise is authority-limited, not gain-limited |
| `TRAIN_ALTITUDE_PWM_SPACE` | `False` | **must go back ON if `DRONE_VELOCITY_CONTROL` goes off** |
| `TRAIN_LOSS_WEIGHTS` | `[1,1,1,1]` | uniform; do not re-tune without flight data |
| `EARLY_STOP_PATIENCE` / `SMOOTH_WINDOW` | 15 / 3 | **these two must move together** |

- 65mm pitch feed-forward is still `0.0`; it sets neither
  `DRONE_ALTITUDE_HOLD_KP` nor `DRONE_VELOCITY_CONTROL`, so it keeps the old
  pure-velocity damper and direct-angle sticks.
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

Keyboard/web controls (the axis meanings follow the table in Project
Essentials — on 80mm these are setpoints, not stick deflections):

- Up/down: altitude
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

**Current best model: `models/bc_dagger1_heights.pth`** — flies
`s_curve_heights_80mm` 10/10 gates, 5/5 episodes, at expert pace.

Current tubs (all 140-deg optics, velocity-setpoint interface, PD altitude
hold). **tub_22..25 predate `TRIM_RANGE = 1.6`** (altitude labels are in
units of it), so do not aggregate them with tub_34+:

| tub | rows | course | kind |
| --- | --- | --- | --- |
| `data/tub_22_26-09-07` | 22,202 | `s_curve_gates_80mm` (flat) | expert rollouts, 40/40 kept, all 10/10 |
| `data/tub_23_26-09-07` | 19,505 | `s_curve_gates_80mm` (flat) | DAgger round 1 |
| `data/tub_24_26-09-07` | 22,230 | `s_curve_heights_80mm` | expert rollouts, 40/40 kept, all 10/10 |
| `data/tub_25_26-09-08` | 18,563 | `s_curve_heights_80mm` | DAgger round 1 |
| `data/tub_34_26-09-10` | 14,940 | `landing_target_80mm` | expert landings, **40/40 kept** |
| `data/tub_35_26-09-10` | 10,473 | `landing_target_80mm` | DAgger round 1 (25 eps, all flyaway, all kept) |
| `data/tub_36_26-09-10` | 2,238 | `landing_target_80mm` | DAgger round 2 seed, 5 eps by `bc_dagger1_land` |

Current models:

| model | course | result |
| --- | --- | --- |
| **`models/bc_dagger1_heights.pth`** | heights | **10.0/10 gates, 5/5, crossing height err 0.254m** |
| `models/bc_dagger1_scurve.pth` | flat S-curve | 10.0/10 gates, 5/5 |
| `models/bc_expert_heights_v1.pth` | heights | 5.6/10 — expert-only, freezes/crashes off-manifold |
| `models/bc_expert_scurve_v1.pth` | flat S-curve | 3.4/10 — expert-only |

Everything else in `data/` and `models/` is superseded and most of it will not
even load:

- `models/rl/ppo_gates_v1` .. `v5`, `v4_best`, `reset_check` — the retired RL
  teachers. See the Historical Record.
- Anything trained before 2026-09-01 is `pre_velocity` (no `vel_fc.0.weight`)
  and `checkpoint_kind()` rejects it: `bc_rl_*`, `bc_v3/v4`, `tub_*`,
  `hello_world_tub3`, `pilot`.
- `tub_12` .. `tub_17` predate the velocity-setpoint interface; `tub_18` and
  `tub_19` are 80-degree optics; `tub_20`/`tub_21` are the abandoned 3-gate
  slalom layouts. `bc_expert_v1`, `bc_expert_fov140`, `bc_expert_turn_v1` and
  `bc_vel_dagger1` are slalom/old-plant students, kept only for comparison.

**Do not mix tubs across an interface, optics, plant or course change.** Each
of those silently changes what a label means.

## Historical Record (2026-08-07 -> 2026-09-06)

Condensed 2026-09-08. Every session below is superseded by the
velocity-setpoint interface and the scripted `GateExpert` (2026-09-06b) and by
the courses that followed. Full text of each original handoff is in git
history: `git log -p -- AGENTS.md`.

Kept here: the arc, so settled decisions are not re-litigated, and every lesson
that still binds.

### The arc

| date | what happened | why it no longer drives work |
| --- | --- | --- |
| 08-07 | Landing-target BC on `landing_target_80mm`. Max pitch/roll 45 -> 20 deg. | Plant and interface have both changed twice since; those tubs and models are dead. |
| 08-22 | RL teacher/student pipeline built: `rl_env.py`, `rl_train.py`, `rl_collect.py`, `gz_reset_worker.py`, `rl_train.sh`. Direct pixel RL rejected — sim is real-time-locked (~70k steps/hour at 20Hz) and pixel PPO needs millions of steps. | Pipeline still exists and `rl_collect.py --expert` is still used. The PPO teacher is not. |
| 08-23 | **Altitude flyaway blocker**, and its fix (below). | Fixed. |
| 08-23 eve | `DRONE_PITCH_THROTTLE_FEEDFORWARD` 8.0 -> 2.0. `ppo_gates_v3` (3-action) failed: 17/17 gate crossings missed on HEIGHT, because it had no altitude actuator. `ppo_gates_v4` (`--control-altitude --altitude-scale=0.09`) was the first teacher to pass gates. | Superseded by the scripted expert. |
| 08-24 | v4 finished 400k. **The 340k checkpoint was better than the 400k final** (10/10 vs 3/10 episodes). Student distillation failed: 5/5 crashes, all into the floor. Also the trainer fix that **separated best-weight SAVING from the early-stop test** (`MIN_DELTA_REL`) — still live, and referenced later as "the 2026-08-24 fix". | Teacher superseded; the trainer fix is current. |
| 08-31 | DAgger round 1 (no velocity input): survival 18.4 -> 36.5s but gates regressed 0.4 -> 0.2. Diagnosed a structural blocker — the teacher's obs included VELOCITY, the student's inputs did not, so the student was regressing a function of a variable it could not see. | Fixed 09-01. |
| 09-01 | Velocity input added to the student, unconditionally (see "Velocity input" below). Partial win: altitude variance tightened and the gate regression reversed, but crashes stayed 5/5. | Still current. |
| 09-05 | **Root cause of the crashes found**, and it was never an oscillation: a ~5 PWM (2.5% of full scale) DC thrust bias, integrated by a plant with no height regulation. `vz = B/K` predicted the observed sink to within 0.005 m/s in every episode. Invisible in val_loss; fatal in flight. | Fixed 09-05b. |
| 09-05b | **PD altitude hold added** (`DRONE_ALTITUDE_HOLD_KP = 30.0`, `DRONE_ALTITUDE_TRIM_RANGE = 1.0`). A steady bias now produces a bounded offset instead of an unbounded sink. Both existing models went stale in the same stroke. | Still current. |
| 09-06 | `ppo_gates_v5` on the PD plant: **zero height misses in 29 crossings** (v3 missed 17/17 on height), but much weaker laterally than v4 — 3/10 episodes 3/3 at 1.5x the steps. Takeoff capture height fixed via `set_altitude_setpoint`. | Teacher superseded 09-06b; the takeoff fix is still current. |

The through-line: nearly every fix from August onward was classical control
inserted between the network and the sticks (velocity damper, pitch
feed-forward, PWM-space labels, DC high-pass, the PD hold). 2026-09-06b
finished that job with a proper setpoint interface instead of continuing it one
axis at a time.

### The altitude flyaway, and its root cause (08-23)

At neutral stick the 80mm drone climbed 0.34m -> 69.8m in 30s, so every RL
episode died in ~2.5s and PPO had no gradient.

**It was not the hover bias.** BetaFlight's stock pitch/roll rate PIDs
(p=45-47, i=80-84, d=38-46, f=120-125) limit-cycle in SITL: at fixed CH3 with
the drone level, the front/rear motor pairs slammed bang-bang between ~1086 and
~1906 PWM (seen live via MSP_MOTOR, cmd 104). Attitude barely moves, but
thrust is proportional to w^2, so the symmetric split rectifies into phantom
collective thrust — measured +2 to +7.7 m/s^2, nearly independent of CH3. The
throttle stick and damper had almost no authority over net thrust. Same w^2
mechanism as the documented "yaw adds thrust" bug, but on the pitch axis and
spontaneous at hover. It also explains the BC-era altitude sag/climb, the
lateral-coast Z climb, and why hover "calibration" never reproduced.

Fixes, all still in the tree:

- `scripts/start.sh`: pitch/roll PIDs lowered to **p=20 i=10 d=0 f=20** (both
  axes, both airframes) via the CLI phase; `PID_GAINS` extends the old
  `YAW_PID` mechanism. The motor split collapses to ~30 PWM and thrust becomes
  a clean function of CH3 again.
- `scripts/start.sh` CLI phase hardened. Two silent-failure bugs meant a fresh
  eeprom could fly on stock PIDs + LEGACY mixer while the log claimed success:
  (1) after an in-process MSP_REBOOT this SITL build's CLI goes permanently
  deaf — '#' never returns a prompt and every get/set no-ops; (2) nothing
  verified the sets. Now airmode disable restarts via the exit-42 path, CLI
  entry retries until a prompt actually appears, and a post-apply `get p_pitch`
  probe aborts the launch if the sets did not stick.
- `DRONE_HOVER_THROTTLE` 1475 -> 1480, `DRONE_ALTITUDE_HOLD_K` 45 -> 90.

**Flight dynamics changed here (rate loops much softer, d=0), so no eval number
or model from before 08-23 is comparable to anything after it.**

### Velocity input (09-01, unconditional — no config flag)

`LinearModel.forward(img, imu, vel)` takes heading-frame velocity through a
small MLP branch. `tub_schema.heading_frame_velocity()` + `VELOCITY_SCALE` are
imported by BOTH `dataset.py` and `torch_pilot.py` so the transform cannot
drift; velocity is rotated into the drone-heading frame and scaled by 5,
matching `GateChaseEnv._get_obs()[5:8]`. `drone_manage.py` REFUSES to start if
`DRONE_RECORD_VELOCITY`/`ATTITUDE` are off.

Pre-velocity 4-output checkpoints are detected by `checkpoint_kind()` as
`pre_velocity` (no `vel_fc.0.weight`) and rejected with an explicit message.
Every checkpoint from before 09-01 is in that category.

Caveat owned deliberately: on real hardware vz needs a barometer or
optical-flow estimate. Fine for sim; a real-airframe port needs an estimator.

### Lessons that still bind (do not regress)

Plant and telemetry:

- **Never call gz services in-process from training/eval code.** Requests from
  a torch-loaded process reliably time out, and enough timed-out in-process
  requests wedge Gazebo's UserCommands service for ALL callers until the stack
  restarts. Use a subprocess (`gz_camera_worker.py`, `gz_reset_worker.py`).
- **gz `set_pose` does NOT zero velocity** — after a flyaway the drone carries
  its momentum through the teleport.
- **Velocities SATURATE at `MAX_VELOCITY`, never zero.** They were once zeroed
  above 20 m/s to hide teleport spikes; the altitude damper reads that same
  value, so past 20 m/s it saw `vz=0`, applied no correction, and the drone
  climbed unchecked. Teleports call `reset_history()` to drop the stale sample
  instead.
- **Neutral throttle does not lift the drone off the ground**, so a landed
  reset needs a real climb command. Takeoff must ramp and cut at LIFTOFF, not
  power to `start_alt`: throttle authority is ~+-100 PWM and the band between
  "won't leave the ground" (0.15) and "rockets to 9m" (0.6) sits inside it.
  Then arrest the climb and freeze the hold setpoint.
- **The altitude hold only regulates inside the command deadband**
  (`DRONE_ALTITUDE_HOLD_DEADBAND`, 0.1 for 80mm). Commanding altitude during a
  reset switches off the thing that arrests the drone; reset loops command 0.0.
- When resets misbehave, log pos AND vel together — position alone cannot
  distinguish "teleport failed" from "teleport worked, drone flew away". x/y
  pinned exactly on target while z runs away means the teleport is fine and
  something is driving it vertically.

Measurement:

- **The sim has NO vertical drag**, so open-loop climb *rate* at a PWM is
  meaningless (it just preserves entry momentum). Measure vertical
  *acceleration* (quadratic fit of z(t)), or close the loop.
- `test_thrust.py --mode=hover` sweeps from the ground and ground contact masks
  the true hover point. Measure hover airborne.
- A PI-on-vz "hover finder" reads ~40-60 PWM low while it is actively
  stabilizing (active PID activity adds rectified thrust even with tame gains).
  Trust ballistic holds entered at ~zero vz, cross-checked against static
  thrust math (km=1.9e-7, maxRpm=2094, mass 0.079kg -> hover ~= motor 1482).
- BetaFlight refuses CLI while armed, and a CLI socket needs ~0.3s after
  connect before sending '#' or it never answers.

Training and models:

- **`TorchPilot`/`LinearModel` must be built at `TRAIN_IMAGE_H/W` (120x160)**,
  not the camera's 640x480. `AdaptiveAvgPool` means a 640x480 model loads the
  same state_dict without error and silently returns badly biased predictions.
  If an offline check disagrees wildly with the reported val_loss, suspect the
  resolution before the model.
- **`TorchPilot`'s resize must stay antialiased.** It once used
  `F.interpolate(bilinear)` without antialias while `TubDataset` used PIL; at
  the 4x downscale that aliased hard (max pixel difference 0.355 of full range;
  0.0029 with antialias). Every flight eval before 2026-09-01 ran on images
  subtly different from training. Same silent failure class as the resolution
  gotcha and the `TRAIN_ALTITUDE_PWM_SPACE` flag.
- **Do not mix tubs across a control-mapping change.** tub_12 (45-deg pitch)
  and tub_13 (20-deg) carry ~2.25x different labels for the same physical
  maneuver; the combined model crashed 3/3 while each alone trained fine.
- Inference matches training to 1.7e-5 when checked correctly. **The first two
  frames of any such comparison MUST differ** — `TorchPilot`'s history starts
  zero-filled while the dataset has a full window. That is not a bug.
- Control-loop timing is not a suspect: measured `loop delay avg=50.0ms` at
  20Hz (min 45, max 55) with inference at 11.5ms, 23% of the period.

If the RL teacher is ever revived:

- **Pick the teacher by deterministic `gate_rollout.py`, never by final step
  count.** v4's 340k checkpoint scored 10/10 episodes; its own 400k final
  scored 3/10.
- **On `--resume`, `--total-steps` means that many ADDITIONAL steps** (SB3
  semantics). `rl_train.sh` passes the same value on every supervisor restart,
  so a run that restarts 8 times gets 8 fresh budgets — v5 reached 611k of a
  nominal 400k. **This is still UNFIXED**: have the script compute
  `total - already_done`.
- `rl_train.sh` is a supervisor (BetaFlight SITL randomly aborts at launch and
  randomly resets its MSP socket during the post-reboot config step); it
  retries up to `RL_MAX_ATTEMPTS` (40), rebuilding and resuming each time.
- Resume is safe now: before choosing a resume the script runs
  `promote_checkpoint.py --once`, so a run killed by a signal (progress only in
  `checkpoints/`) is promoted and resumed rather than silently restarted from
  step 0. If checkpoints exist but none can be promoted it refuses with exit 2.
  A clean final `model.zip` is never clobbered.
- Throughput is reset-dominated, not the flying time `rl_train.py` prints:
  ~5 env steps/sec early, approaching 20/sec as episodes lengthen. Budget
  10-20h for 400k, not 6h.
- `<out>/env_config.json` records rate/altitude-scale/etc so a teacher is flown
  exactly as trained; it does not exist until the run finishes.
- `--control-altitude` was added because the OLD plant had no height
  regulation. With the PD hold that reason is gone, and v5 saw 0/29 height
  misses — a 3-action policy is the natural configuration now.

Stack lifecycle:

- `stop_all.sh`'s wrapper kill is ANCHORED to a shell actually executing the
  script (`pgrep -f '^[^ ]*/?bash[^ ]* [^ ]*scripts/start\.sh'`). The
  unanchored pattern matched any process whose command line merely MENTIONED
  the path and killed a training command mid-launch. It also kills the
  `tail -f /dev/null` keepalive child FIRST, and skips any `start.sh` that is
  calling it (`start.sh` runs `stop_all.sh` from its own EXIT trap).
- BetaFlight SITL aborts at launch roughly half the time (SIGABRT or SIGTRAP,
  "pointer being freed was not allocated" in
  `stream_initAddress <- openSerialPort <- mspSerialAllocatePorts`). **Not
  random heap corruption — a thread race in the SITL build** (root-caused
  2026-10-13): `sitl.c`'s `tcpThread` spins `dyad_update()` with no lock while
  the main thread opens serial ports in `serial_tcp.c:tcpReconfigure`.
  `dyad_newStream()` creates a stream in state CLOSED; if `dyad_update()` runs
  before `dyad_listenEx()`, its `destroyClosedStreams()` frees the stream and
  `listenEx` then works on freed memory. Not eeprom, not config. The fix (one
  mutex around both, in `~/dev/betaflight`, then rebuild) is **not applied**;
  `start.sh` retries around it. Crash reports land in
  `~/Library/Logs/DiagnosticReports/`.
- A disarmed board can never take off, so `reset()` re-raises immediately when
  `env.bf_armed` is false rather than burning retries.

### Dead artifacts from this era

Models `models/rl/ppo_gates_v1` .. `v5` (and `smoke`, `reset_check`) are all
superseded: v1/v2 trained against the oscillating plant, v3 has no altitude
axis (its checkpoints will not load in a 4-action env), v4/v5 predate the
velocity-setpoint interface. Tubs 12-17 predate the interface change AND the
140-degree camera; every checkpoint before 2026-09-01 is `pre_velocity`.

### Still open from this era

- **65mm has never been re-tested airborne** on the tamed pitch/roll PID gains
  (08-23) or the corrected yaw scaling (09-06b). It does not set
  `DRONE_ALTITUDE_HOLD_KP` or `DRONE_VELOCITY_CONTROL`, so it is otherwise
  unaffected by the plant and interface changes.
- The `rl_train.sh` resume budget bug above.

## Historical Record (2026-09-06b -> 2026-09-10a)

Condensed 2026-09-21. These handoffs built the current interface, expert,
courses and landing task. Full text: `git log -p -- AGENTS.md`.

### The arc

| date | what happened |
| --- | --- |
| 09-06b | **Velocity-setpoint interface** (`DRONE_VELOCITY_CONTROL`) + scripted `GateExpert` replace the PPO teacher. Expert matched the best RL teacher (30/30 on slalom) with zero training. Yaw scale bug fixed. Camera widened 80 -> 140 deg. Student: 1.0/3 gates — target assignment ambiguity (gate 1 -> gate 3, skipping 2). |
| 09-06c | Slalom re-laid out to remove the ambiguity. Student then flew circles (1.0/3): zero corrective data in a flawless-expert tub. |
| 09-07 | 10-gate `s_curve_gates_80mm` built. Expert-only student 3.4/10 (best episode 9/10), plus a new FROZEN fixed point. Early-stopping noise fix. |
| 09-07b | **DAgger round 1 with `GateExpert` as labeller: 10/10, 5/5.** First vision student to complete a course. |
| 09-08 | `s_curve_heights_80mm` built; `GateExpert` height law fixed; `bc_dagger1_heights` 10/10, 5/5. |
| 09-08b | Altitude-hold bandwidth sweep: bench step response improved hugely, but the expert flew WORSE at every raised gain. Gains reverted. MSP launch gate added. |
| 09-09 | Landing plan (option C: `TRIM_RANGE` 1.0 -> 1.6). Stage 0 probes, Stage 1 `land` scorer, Stage 2 `LineExpert` (10/10), Stage 3 `TRIM_RANGE = 1.6` in config (gate courses did not regress). |
| 09-10a | Stage 4a: collectors learn the land task; **PoseTracker phantom-vz bug** found and fixed. Stage 4 result is the Current Handoff. |

### Interface and plant (09-06b)

- **Why the setpoint interface, not a v6 RL run:** every teacher/student
  result had flown well as a teacher and failed as a student; the 09-05 root
  cause (a 2.5% thrust bias, invisible in val_loss) was the INTERFACE, not the
  policy. ArduPilot was rejected as the wrong size of move (rewrites the bridge,
  env, telemetry and launch scripts; does not run on whoop boards).
- The old altitude curve `sign(a)*|a|**0.2` was near-vertical at zero
  (alt=-0.0001 was already -15.8 PWM). The trim axis is linear, no deadband.
- **Every takeoff/landing ramp passes `direct_thrust`** (`rl_env._takeoff`,
  `eval_flight.takeoff`, `probe_hover`, `ff_sweep`, the `-1.0` landing calls).
  `ff_sweep.py` forces `env.velocity_control = False`. `autonomous_collect.py`
  writes raw sticks and was NOT migrated — direct-mode only.
- **Cascade gains are measured, per axis** (roll max 10 deg vs pitch 20, so the
  same stick buys half the accel). Open-loop sweep: pitch 0.15/0.30/0.50/0.80
  -> 0.26/0.73/1.53/2.99 m/s; roll 0.30/0.60/1.00 -> 0.26/0.72/1.52 m/s. `KFF` is
  the inverse slope at cruise, `KP` trims the residual. `probe_velocity.py`:
  forward 1% error / 1.18s rise, lateral 5-6%, leakage <= 0.01 m/s. Lateral rise
  (~2.3s) is AUTHORITY-limited; the fix is `DRONE_MAX_ROLL_ANGLE` 20, which
  invalidates `KFF_LAT` and needs a re-sweep.
- **Yaw was 3.5x over-scaled.** `_YAW_DEG_PER_SEC_PER_PWM` 0.30 -> 0.60
  (response is strongly expo; calibrated at 120 deg/s). Every earlier yaw
  command, both RL teachers included, was ~3.5x too aggressive.
- **Measurement lesson: sample densely and UNWRAP.** An endpoint-only yaw probe
  read +119 deg for a true -240.8; the wrong sign in the expert made it fly the
  whole course BACKWARDS (heading-frame setpoints still tracked the x/y path),
  and that backward flight was the sole cause of the z swings chased before it.
  Do not reopen altitude on the strength of pre-09-06b traces.
- Verified clean plant: straight flight dz = 0.00 over 10s; yaw-induced thrust
  ~0.3 PWM, so `DRONE_YAW_THROTTLE_FEEDFORWARD = 0.0` is correct.
- `TRAIN_ALTITUDE_PWM_SPACE` OFF (the curve is no longer in the path; must go
  back on if velocity control goes off — train and inference read it
  independently). `eval_flight --alt-highpass/--alt-gain` refuse under velocity
  control; `--zero-altitude` now means "hold the takeoff height".

### Camera: 80 -> 140 deg (09-06b)

`~/dev/aeroloop_gazebo/models/betaloop_drone_cam_80mm/model.sdf`
`horizontal_fov` 1.39626 -> 2.44346; backup `model.sdf.fov80.bak`; evidence
renders in `docs/fov/`. At 80 deg the next slalom gate (57-67 deg relative) was
out of frame at the turn. Cost: pinhole projection, centre resolution
0.15 -> 0.49 deg/px, distant gates ~3.3x smaller. If that ever hurts, the
principled fix is Gazebo's `wideanglecamera` with an equidistant lens (not done:
risks the topic path the camera worker depends on). 65mm SDF unchanged.

Widening the field made the target visible and the target ASSIGNMENT harder:
the student saw two identical gates and chased the centred (wrong) one.
`eval_flight --trace` records x/y/yaw for exactly this kind of diagnosis.

### GateExpert (09-06b, 09-08)

- Pure pursuit along the gate CENTERLINE (carrot = own along-track position +
  `lookahead`, zero lateral offset); yaw toward the carrot keeps the gate in
  frame.
- **No cos(bearing) speed slowdown** — a car reflex; this vehicle is holonomic.
  With it, cruise collapsed to 0.375 m/s.
- **`height_gain` (Kz) is a STABILITY constraint**: it wraps a P loop around a
  hold that is already a position regulator, and the trim saturates into a
  relay. Kz=2.0 limit-cycled 1.1 <-> 3.1m.
- **Height law (09-08) is feed-forward + P:**
  `altitude = ((z_ref - setpoint) + Kz*(z_ref - z)) / TRIM_RANGE`, with `z_ref`
  piecewise-linear between gates at the drone's along-track position. The old
  pure-P law left 67% of `(gate_z - setpoint)` as permanent error at Kz=0.5 —
  invisible on a flat course. Bit-identical to the old law when flat.
  `height_lookahead` defaults to 0 (a 3m lead overshot every gate by 0.4-0.6m).
  Kz=0.75 by sweep (0.5: 97%, 0.75: 98%, 1.0: 96%, 1.5: 79% + crashes).
- A scripted expert is invariant to `TRIM_RANGE` (it divides by it, the env
  multiplies by it); only the clip point moves.

### Course design rules (09-06c, 09-07, 09-08)

- **Gate yaw = path TANGENT**, not the gate's own leg bearing — the drone exits
  aligned with the gate normal.
- **Constant curvature, not a zigzag.** In a bisector-aligned serpentine, gate
  n+2 sits dead centre after gate n while gate n+1 is off to the side — the
  gate-skip failure built into the track.
- **Zero-pad gate names** (`gate_01`..`gate_10`): `parse_world` sorts by name
  as a STRING.
- Unique gate colours did not help (the slalom had them); colour is not the
  signal.
- **Varied heights must be a smooth profile, not per-gate steps**: the PD hold
  is w_n = 0.46 rad/s, zeta = 0.42 (~5s rise, ~20s settle) against 2.5s between
  gates. Steps made the EXPERT miss 38-51%. Heights course uses
  `z = 1.75 + 0.55*sin(2*pi*s/18 + phi)`, climb and descent on both arcs.
- Height courses need **closed rectangular gates** (a visible vertical
  centre). Size the opening on the EXPERT's measured error: +-0.6m made the
  expert flaky, +-0.8m is used.
- `parse_world` derives a **per-gate pass window** from the SDF posts
  (`gate_z +- (post_height/2 - 0.2)`); flat courses reproduce the old 0.2-2.8m.
  `GateTracker` records every crossing (`gate_crossings`,
  `mean_crossing_height_err_m`) and discards crossings wider than
  `2*half_width` (a gate plane is infinite).

### Failure modes of a memoryless student (09-06b -> 09-08)

All off-manifold, all from flawless-expert tubs with zero corrective data:

| failure | seen | fix |
| --- | --- | --- |
| flies straight past, next gate out of frame | 80-deg camera | 140-deg camera |
| chases the centred (wrong) gate | 140-deg on slalom | course layout |
| orbits at 8-10m radius (limit cycle) | re-laid slalom | S-curve + DAgger |
| **freezes**: throttle ~0 is a STOP command under velocity control, a stopped frame begets the same output | S-curve, heights | DAgger (stopped states labelled "go") |

The frame stack / GRU (`LinearModel.forward` takes ONE image) is the
structural fix for memorylessness, but **no benchmark currently demands it** —
memoryless students fly both S-curves 10/10. Don't build it until one does.

### Training lessons (09-06b, 09-07, 09-07b)

- **Trainer save vs stop are separate tests.** Weights save on ANY raw
  improvement; patience uses `MIN_DELTA_REL` (1% of best, optional
  `MIN_DELTA_ABS`). The old absolute 0.0005 exceeded the entire loss at
  distillation scale and deployed epoch 8 over a 2x-better epoch 10.
- **Early stopping on noise.** Distillation val loss swings ~10x per epoch; one
  lucky early epoch stopped a run at 0.001186 that reaches 0.000052 (23x).
  Patience now runs on the median of the last `EARLY_STOP_SMOOTH_WINDOW` (3)
  val losses. The median LAGS, so it needs patience 15, not 5 — smoothing at
  stock patience was worse than none on both real runs. **Move them together.**
- **DAgger val loss measures the PREVIOUS policy's mistakes** (the split is
  chronological, so val is the tail of the DAgger tub). 200-350x train/val gaps
  on models that fly 10/10. **Judge DAgger rounds by `eval_flight` only.**
- `dagger_collect.py --expert` labels with `GateExpert.act_from_telemetry`
  (or `LineExpert` on the land task), forcing
  `{"control_altitude": True, "altitude_scale": 1.0}`.

### Altitude-hold bandwidth (09-08b) — do not re-run expecting a new answer

`probe_alt_step.py --gains` swept Kp:K from 30:90 to 320:294 at constant
damping (Kp ~ r^2, K ~ r — doubling both raises zeta too, not just w_n):
settle 24.4s -> 4.5s, zeta improves 0.36 -> 0.79, no stability limit anywhere.
Yet at Kp=120/K=180 the expert passed 72-85% with most episodes crashing
(Kp=30/K=90: 30/30).

**Root cause: the bench probe holds the drone STATIONARY, so it cannot see
the constraint.** In flight the velocity term dominates 73-76% of steps and
`alt_hold_bias` sat pinned at the +-200 clip 27-52% of the episode at Kp=120 —
a relay, not a regulator. Sizing rule: **`K * vz_p99 + Kp * err_max < 200`
with IN-FLIGHT vz** (heights course: vz median 0.40, p99 1.27 m/s). K=90
saturates only at 2.22 m/s, i.e. never on these courses. Bandwidth here is
clip-limited, not stability-limited: to raise it, raise the +-200 clip in
`drone_gym.py` first and measure saturation with `--trace-output`.
Steady hover offset is `bias/Kp` (~0.10m at Kp=30).

If gains are ever raised, a stiff hold drags the drone back below
`min_start_alt` after takeoff; the companion fix at the top of `_arrest_climb`:

    if getattr(self._env, "altitude_hold_kp", 0.0) > 0.0:
        self._env.set_altitude_setpoint(self.start_alt)

### Landing plan decisions (09-09) — do not re-litigate

- Vertical authority via **`TRIM_RANGE` 1.0 -> 1.6** (reachable floor -0.1m).
  Rejected: a 5th thrust output (invalidates everything), altitude as a
  climb-RATE (another interface change), scripted takeoff/landing (tests too
  little).
- **Scripted expert -> BC -> DAgger, not RL.** RL would only earn a place if
  the flare were to be discovered by terminal reward rather than specified.
- **Tubs and checkpoints do not cross `TRIM_RANGE` 1.0 -> 1.6**: the altitude
  label is in units of it. `tub_22..25`, `bc_dagger1_heights` and
  `bc_dagger1_scurve` are pre-1.6. `bc_dagger1_heights` still flies 10/10 at
  1.6 but commands 1.6x its intended offset (`z_ss = 1.273*z_ref -
  0.273*setpoint`, ~27% overshoot) — retrain before relying on it at 1.6 or on
  a course with larger excursions.

### Landing measurements (09-09 Stage 0)

- The hold behaves **identically at 0.6m and 1.5m**; **no ground effect**
  (offset +0.098m and throttle 1477.7 at every setpoint 1.5 -> 0.25m).
- **The altitude axis has no descent-rate command**: fastest descent is
  `alt = -1`, so descent speed is set by how far BELOW GROUND the target sits.
  With no vertical drag, a steady descent flies a lag above its target:
  `e = ((K/Kp)*r + B/Kp) / (1 + Kz)`, confirmed to 0.01m. Depth and rate are
  spent from the same trim range.
- Full-down landing from 1.5m: trim 1.6 -> touchdown 8.6s at -0.02 m/s;
  2.0 -> 4.0s at -0.18; 2.5 -> 2.7s at -0.37; 3.0 -> 2.1s at -0.56. Peak
  `alt_hold_bias` 49-90 pwm, 0% saturated. **On a landing the scarce resource
  is trim RANGE, not bias headroom** — the opposite of the gate courses.

### The `land` task in eval_flight (09-09 Stage 1)

- `detect_task` returns `land` for a pad with no gates; `--task=line` keeps the
  old horizontal-proximity fly-over test.
- Contact is `--touchdown-z` (0.06), not `--crash-z` (0.12, which a textbook
  landing passes on the way down).
- `classify_touchdown()` is a pure function: `completed` / `crashed`
  (`|vz| >= --settle-vz` 0.15) / `skidded` (`speed_xy >= --settle-speed` 0.5 —
  a belly-flop at cruise has tiny vz and must not score as a landing) /
  `landed_off_pad` (> `--pad-radius` 1.0).
- Impact speed is taken over the **0.5s BEFORE contact**
  (`TOUCHDOWN_VZ_WINDOW_S`): the filtered velocity at the contact sample is
  already decelerating.
- **If `TRIM_RANGE` ever goes past 1.6, raise `--settle-vz` with it** (trim 2.0
  touches at 0.18 m/s).

### LineExpert (09-09 Stage 2, 09-10a)

`nanopilot/line_expert.py`. A pure function of privileged state — no phase,
timer or latch — so DAgger can label a drone that drifted off the pad with "go
round". Pure pursuit along the line (carrot clamped at the pad); speed
`min(cruise, 0.6 * d_pad)`; height reference ramps from cruise to a
BELOW-GROUND `ground_target` (-0.1m) through the gate expert's FF+P law, Kz 0.75.

- The ramp **saturates at `land_radius` (0.5m)** instead of tapering: the hold
  settles ~0.10m above its target, so a tapering ref parks the drone at 0.2m.
- A target at exactly 0.0 is approached asymptotically and never reached.
- **The glide slope is sized by the CAMERA.** The forward 140-deg camera is
  +-64 deg vertically; a drone over the pad at height cannot see it. Level
  approach + vertical drop (`descend_radius=2`) landed fine but had the pad out
  of frame 16.7% of the episode with a near-binary altitude label;
  `descend_radius=8` gives 3.3% and 22.7% of the label on the ramp.
- `flare_alt` was explored (0.8 costs ~36% pad-out-of-frame); anything clever in
  the last metre is discarded by the -1 clip anyway, and a reference RATE LIMIT
  cannot work there for the same reason. Current default is the floor OFF.
- `eval_flight --expert` scores the expert with the same rules as a checkpoint;
  `--trim-range` overrides `TRIM_RANGE` in flight with a loud banner.

### BUG FIXED (09-10a): PoseTracker published a phantom vz = 0

`gz_telemetry.PoseTracker.update()` differenced consecutive samples whether or
not Gazebo had published a new pose, so polling faster than the pose topic gave
`vz = 0` and the `-K*vz` damper intermittently vanished. Harmless on gate
courses; on a landing (all damper) the collector — which calls `run_threaded`
twice per period — touched down at -1.02 m/s while the damper thought vz was
-0.017. Fixed: only difference on a changed pose. Gate courses improved
(expert crossing height err 0.36 -> 0.23m). **Same family as the MAX_VELOCITY
lesson: an estimator reporting a confident wrong number to a loop that cannot
tell.**

Also 09-10a: `_arrest_climb` finished on `|z - start_alt| < 0.15`, which the
drone satisfies on the way THROUGH (it settles 0.10m above), handing over a
ringing drone (+-0.12m all flight). Now gated on 30 consecutive samples of
`|vz| < 0.05` (1.5s; shorter is fooled by the ring's turning points).

Collector plumbing: `GateChaseEnv` has task hooks `_task_step`, `_task_info`,
`_task_summary`; `LineChaseEnv` (`line_env.py`) overrides them and passes the
pad as a one-entry gate list. Touchdown is classified by
`eval_flight.classify_touchdown` so the collector and scorer agree; `crash_z`
is disabled (-1.0). `rl_collect` keeps ONLY `completed` on the land task and
refuses to start if `TRIM_RANGE` cannot reach `ground_target`.

### Stack and tooling lessons (09-06c -> 09-10a)

- **`start.sh` now gates BetaFlight on MSP answering**
  (`betaflight_msp_alive`, 8 attempts). SITL aborts at launch ~50% of the time
  and "process alive" was not "MSP alive" — the root cause of every "rebuilt
  three times, never armed" session. `scripts/start_gated.sh` wraps the whole
  stack (launch, `STACK_READY`, 25s settle, MSP check, rebuild) and its
  `wait_quiet()` waits for the previous `start.sh` EXIT trap to finish killing
  things — a 2s sleep let the old trap kill the new stack (`Killed: 9`).
- **Until 2026-10-13 that retry loop could not survive a crash.** A crashed BF
  is already reaped, so the loop's `kill -9` / `wait` returned non-zero and
  `set -e` killed `start.sh` before the retry: any BF crash, on the first
  launch or the post-config restart, ended the launch. It only ever retried a
  live-but-MSP-deaf board. Fixed with `|| true` on both. Any cleanup after a
  background child under `set -e` needs the same guard.
- **IMU readiness race:** `imu_ready` needs non-zero IMU, and a motionless
  drone reads zero. Wait ~25s after `STACK_READY` before collectors or
  `eval_flight.py`; `eval_flight.sh` does not, so bring the stack up separately.
- **READ the MSP health output before launching** — a DAgger run was once
  started right after it printed `(None, [], None, [])`.
- **Never pipe a flight script into `tail`/`head`/`grep`**: the camera worker
  inherits stdout (pipe never EOFs, hangs silently) and its per-second log
  floods the output. Redirect stdout and stderr to separate files. Use
  `python -u` so a crash does not lose the buffer.
- `gate_rollout.py --world` OVERWRITES `GZ_WORLD` (default
  `slalom_gates_80mm`); `eval_flight.py` has NO `--world` (reads `GZ_WORLD`;
  only `eval_flight.sh` takes it). `eval_flight.py` writes JSON with
  `--json-output`.
- Run `stop_all.sh` on its own line; chained, it has killed the calling shell.
- **`direct_thrust=0.0` is a climb with no damper**, not "no command": it pins
  hover throttle and suppresses `alt_hold_bias`, so the drone coasts up forever.
  Arrest on the HOLD (`run_threaded(0,0,0,0)`). `probe_alt_step.py`'s takeoff
  still has this bug (costs ~30s per run).
- `probe_alt_step.py`: small steps report a bogus overshoot (the 0.10m residual
  is most of a 0.15m step) — read the large-step leg. `--settle-s` for low
  start altitudes. `--gains` sweeps KP:K pairs in one flight.
- `probe_descent.py`: a descent leg must ramp several time constants (>= 4s) or
  it measures a transient.

### Still open

- 65mm has never been re-tested airborne on the tamed pitch/roll PIDs or the
  corrected yaw scaling.
- `rl_train.sh` resume budget bug (see the earlier Historical Record).
- `bc_dagger1_scurve` was not re-flown at `TRIM_RANGE = 1.6` (heights, the
  harder test of the same mechanism, passed).

## Current Handoff (2026-09-10, THE LANDING TASK IS SOLVED)

**STATE: stopped, nothing running.** `models/bc_dagger1_land.pth` flies
`landing_target_80mm` **5/5, touching down 0.109m from the pad centre at
-0.027 m/s**, and 5/5 again in the jittered collector harness. The plan from
2026-09-09 is complete end to end: probes -> scorer -> expert -> TRIM_RANGE ->
collect -> distil -> DAgger.

### Result

| | pad offset | touchdown vz | eval_flight | jittered collector |
| --- | --- | --- | --- | --- |
| `LineExpert` (ceiling) | 0.045m | -0.032 m/s | 10/10 | 40/40 |
| `bc_expert_land_v1` | **0.53m** | -0.028 m/s | **5/5** | **0/25 (all flyaway)** |
| **`bc_dagger1_land`** | **0.109m** | -0.027 m/s | **5/5** | **5/5** |

Cross-track along the 20m line is 0.015m and every episode finishes in ~19.0s
against the expert's 18.6s.

**2026-10-13/15: `eval_flight` now takes off exactly like the collectors.**
Its old local `takeoff()` powered to `start_alt` (~0.9m overshoot, slow sink
back) and handed over on `|z - start_alt| < 0.15`, the on-the-way-through test
`rl_env` dropped on 09-10a. Both now call the module-level
`rl_env.takeoff()`; `--warmup` 6 -> 1s, `--climb-altitude` 0.15 -> 0.6,
`--takeoff-timeout` 10 -> 15s. Re-measured under it (5 runs, 20Hz):
`LineExpert` 5/5 at 0.044m / -0.032 m/s; `bc_dagger1_land` 5/5 at **0.120m**
(0.111-0.126) / -0.029 m/s. Numbers above are from the old takeoff.
On `s_curve_heights_80mm` (2 runs each): `GateExpert` 10/10 at 0.232m
crossing height error, `bc_dagger1_heights` 10/10 at 0.233m (was 0.254m).
`bc_dagger1_scurve` has NOT been re-flown under it.

`eval_flight --expert` now flies `GateExpert` on gate courses too (gate index
advances on ANY crossing of the current gate's plane, as in `GateChaseEnv`),
and `--record-dir=DIR` saves every camera frame the pilot is handed as
`DIR/epNN/NNNNN.jpg` (holds 20 fps). The README's "See it fly" videos were
made this way; the MP4s are hosted as GitHub attachments on issue #1, not in
the repo.

### The expert-only student landed 5/5 FIRST TRY -- and that was misleading

This is the first expert-only student in the project's history to complete a
course without DAgger (the gate students managed 1.0/3, 3.4/10 and 5.6/10).
The task is genuinely easier: a straight line with a painted guidance line,
one target, and no ordering ambiguity. `user/angle` in the expert tub has
std **0.055** with 0% of frames past 0.5 -- there is almost no yaw decision to
get wrong.

**But 5/5 under `eval_flight` hid a policy that fails 25/25 elsewhere.** It
touched down 0.53m from the pad against the expert's 0.045m, consistently
LONG (along-track 20.77m against 20.4m). Run in the collector, which jitters
the spawn, the same student overshot ~1.4m and flew away in every one of 25
episodes.

The mechanism, straight off a traced rollout: the student overshoots slightly,
**the pad passes behind it and out of the camera**, the view ahead then looks
exactly like open cruise, and a memoryless policy emits cruise commands --
altitude trim snapped from -1.00 to -0.001 and throttle from 0.38 back to
0.74 in a single second, at x=21.4 and z=0.35. It then climbed away.

That is the 2026-09-06b "the target is outside the camera" family again, and
the third distinct time it has appeared on this project. **A landing task
hides it better than a gate course does, because overshooting by half a metre
still scores as a success.**

### DAgger fixed it, and the coverage numbers say exactly why

`data/tub_35_26-09-10`, 25 episodes, 10,473 rows, every one a flyaway and
every one KEPT -- which is the point of the round.

| | tub_34 (expert) | tub_35 (DAgger) |
| --- | --- | --- |
| forward range | -0.11 .. +0.75 | **-0.75 .. +0.75** (reverse) |
| forward std | 0.359 | **0.650** |
| rows past the pad (x>20) | 25.0% | 36.4% |
| what the expert says there | thr **-0.069** (settling) | thr **-0.605** (come back) |
| yaw \|x\|>0.2 | 1.5% | **28.7%** |

The expert's own tub does contain rows past x=20 -- it lands slightly beyond
the pad centre -- but it is never more than a few centimetres out, so it
never demonstrates a RECOVERY. The DAgger tub does, at -0.605 throttle.

Retrained on both tubs aggregated (25,413 rows), best val **0.002185** against
**0.000111** for the expert-only run -- a 20x WORSE validation loss for the
model that flies 5/5 where the other flies 0/25. **The 2026-09-07b lesson
holds exactly: DAgger validation loss measures the previous policy's mistakes
on a distribution the improved policy no longer visits. Judge by eval_flight
only.**

### Label distribution: this task is altitude-dominated

The opposite of the gate courses, and worth knowing before designing the next
one:

| axis | mean | std | \|x\|>0.5 |
| --- | --- | --- | --- |
| yaw | +0.001 | **0.055** | 0.0% |
| forward speed | +0.496 | 0.359 | 66.7% |
| lateral speed | +0.002 | 0.113 | 0.9% |
| **height trim** | **-0.474** | **0.458** | **47.2%** |

Height-trim std is 0.458 here against 0.072 on the flat S-curve and 0.331 on
the varied-height one. The altitude label is 36% pinned at -1.0, 18% on the
ramp, 45% near zero.

**The plan expected a badly imbalanced label distribution needing loss
re-weighting. It does not need it** -- the descent is 8.8s of a 19s episode.
`TRAIN_LOSS_WEIGHTS` stays uniform.

### The glide slope was tuned on DATA quality, not on landing quality

All three settings land 100% once the plant bug below was fixed, so the choice
was made on how much of the episode has the pad out of frame:

| flare_alt | duration | pad out of frame | collector pad offset |
| --- | --- | --- | --- |
| **-0.1 (floor OFF, default)** | **17.9s** | **3.6%** | 0.058-0.067m |
| 0.35 | 19.1s | 9.9% | 0.022-0.027m |
| 0.8 | 21.1s | ~36% | 0.062-0.065m |

`descend_radius = 8` was chosen the same way in Stage 2 (3.3% out of frame
against 16.7% for a level approach). On this course the camera, not the
plant, sets the trajectory.

### Do this next

1. **DAgger round 2 is seeded and cheap.** `data/tub_36_26-09-10` already
   holds 5 episodes flown by `bc_dagger1_land` with expert labels. The
   remaining gap is 0.109m against the expert's 0.045m; whether that is worth
   closing depends on whether the pad gets smaller.
2. **The course cannot discriminate much further.** Every configuration lands.
   A harder benchmark would put the pad off the line's end (so approach
   heading matters), shrink it, or add a second decoy pad -- which would
   finally make the memory question from 2026-09-07b concrete again.
3. `bc_expert_land_v1` is kept only as the counterexample: it is the model
   that proves `eval_flight` alone can pass a policy the collector fails.
   **Evaluate a landing student in BOTH harnesses.**
4. Debris to delete when convenient: `data/tub_26/27/28_26-09-10` (empty, 8KB
   each, failed smokes) and `data/tub_29..33_26-09-10` (~70MB total, the flare
   sweep). `tub_34/35/36` are the real ones.
5. Still open from earlier: 65mm has never been re-tested airborne on the
   tamed pitch/roll PIDs or the corrected yaw scaling.
6. Optional: patch the BetaFlight SITL launch race (root cause in "Stack
   lifecycle" above) so launches stop needing retries at all.

