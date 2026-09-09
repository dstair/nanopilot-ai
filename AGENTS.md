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
| `DRONE_ALTITUDE_TRIM_RANGE` | 1.0 | metres of setpoint authority in-deadband; still un-swept |
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
hold — i.e. the only ones safe to train on today):

| tub | rows | course | kind |
| --- | --- | --- | --- |
| `data/tub_22_26-09-07` | 22,202 | `s_curve_gates_80mm` (flat) | expert rollouts, 40/40 kept, all 10/10 |
| `data/tub_23_26-09-07` | 19,505 | `s_curve_gates_80mm` (flat) | DAgger round 1 |
| `data/tub_24_26-09-07` | 22,230 | `s_curve_heights_80mm` | expert rollouts, 40/40 kept, all 10/10 |
| `data/tub_25_26-09-08` | 18,563 | `s_curve_heights_80mm` | DAgger round 1 |

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
- BetaFlight SITL intermittently aborts at launch (heap corruption in
  openSerialPort/stream_initAddress, macOS ARM64) — random per launch, not
  eeprom, not config. `start.sh` retries 5x with 2s pauses. Crash reports land
  in `~/Library/Logs/DiagnosticReports/`.
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

## Current Handoff (2026-09-06b, velocity-setpoint interface + scripted expert)

**STATE: stopped, nothing running.** The learned interface changed. Pitch/roll
are now SPEED SETPOINTS and altitude is a LINEAR height trim; the env closes the
inner loop. A scripted `GateExpert` replaces the PPO teacher. No RL run is
needed for the current pipeline.

### Why this instead of a v6 RL run

Every teacher/student result to date had the same shape: the teacher flew well
(v4 hit 30/30 gates deterministically) and the student never exceeded 1.0 gates.
Making v5 as good as v4 would have walked back into the same distillation wall.
The 2026-09-05 root cause was that a 5 PWM altitude bias (2.5% of full scale,
invisible in val_loss) was fatal because the plant integrated it -- i.e. the
problem was the INTERFACE the policy had to hit, not the policy. Nearly every
fix since August had been classical control inserted between the network and the
sticks (velocity damper, pitch feed-forward, PWM-space labels, DC high-pass, the
PD hold). This change finishes that job instead of continuing it one axis at a
time. ArduPilot was considered and rejected as the wrong size of move: it would
buy the same setpoint interface at the cost of rewriting the Gazebo bridge,
`drone_gym.py`, telemetry and the launch scripts, and it does not run on
whoop-class boards. The setpoint layer took a day to build in sim.

### The interface

`DRONE_VELOCITY_CONTROL = True` (80mm only; 65mm does not set it and is
unaffected). The DonkeyCar key names are unchanged so tubs, the web UI and the
training pipeline all still work, but the values mean:

| key | old meaning | new meaning |
| --- | --- | --- |
| steering | yaw rate | yaw rate (unchanged) |
| throttle | pitch angle | forward speed, fraction of `DRONE_MAX_FORWARD_SPEED` |
| roll | bank angle | rightward speed, fraction of `DRONE_MAX_LATERAL_SPEED` |
| altitude | thrust via a 0.2-power curve | LINEAR height trim in metres of `DRONE_ALTITUDE_TRIM_RANGE` |

The altitude change is the point: `sign(a)*|a|**0.2` was near-vertical at zero
(alt=-0.0001 was already -15.8 PWM), so tiny regression errors became large
thrust errors. The trim axis is linear with no deadband.

`run_threaded(..., direct_thrust=X)` is new and bypasses the hold entirely with
the old curve applied. Takeoff and landing ramps need real thrust and cannot
express it through a height trim, so **every takeoff/landing ramp now passes
`direct_thrust`** (`rl_env._takeoff`, `eval_flight.takeoff`, `probe_hover`,
`ff_sweep`, and all the `-1.0` landing calls). If you write a new ramp, use it.

Direct-angle-mode tools: `ff_sweep.py` sets the pitch stick directly to measure
tilt-vs-climb, so it now forces `env.velocity_control = False`.
`autonomous_collect.py` still writes raw stick values and is direct-mode-only;
it was NOT migrated.

### Measured cascade gains (do not guess these)

Gains are PER AXIS because `DRONE_MAX_ROLL_ANGLE` is 10 degrees against
`DRONE_MAX_PITCH_ANGLE` 20, so the same roll stick buys half the acceleration.
Open-loop sweep, 2026-09-06:

| pitch stick | fwd m/s | roll stick | right m/s |
| --- | --- | --- | --- |
| 0.15 | 0.26 | 0.30 | 0.26 |
| 0.30 | 0.73 | 0.60 | 0.72 |
| 0.50 | 1.53 | 1.00 | 1.52 |
| 0.80 | 2.99 | | |

`KFF` is the inverse slope at cruise (the stick that HOLDS the speed against
drag), `KP` trims the residual. `KP_FWD=0.6 KFF_FWD=0.35 KP_LAT=1.2 KFF_LAT=0.8`.
`probe_velocity.py` verdict after tuning: forward 1% steady-state error and
1.18s rise; lateral 5-6% error; cross-axis leakage <= 0.01 m/s; height drift
0.01m. Repeating the forward leg after a 250-degree yaw change gave 0% error and
-0.01 cross-axis, which is the proof the heading-frame rotation is right.

Lateral rise is ~2.3s and that is an AUTHORITY limit, not a gain to raise: the
roll stick is pinned at full deflection through the transient. Raising
`DRONE_MAX_ROLL_ANGLE` to 20 is the available fix if lateral response ever
matters; it would invalidate `KFF_LAT` and need a re-sweep.

### BUG FIXED: yaw was 3.5x over-scaled, and the sign was measured wrong

`_YAW_DEG_PER_SEC_PER_PWM` was 0.30, giving a 400 PWM cap. Measured with dense
UNWRAPPED sampling: 100 PWM -> 36.8 deg/s, 200 -> 120.9, 400 -> **423 deg/s**.
So `DRONE_MAX_YAW_RATE = 120` was actually delivering 423 deg/s, and the
response is strongly expo so no single constant is right everywhere. Now 0.60,
calibrated at the 120 deg/s operating point (cap ~200 PWM). **Every yaw command
in this project's history, including both RL teachers', was ~3.5x more
aggressive than the config claimed.**

**The measurement lesson is worth more than the fix.** The first probe sampled
yaw only at the START and END of a 2s leg and read "+0.2 -> +119.4 degrees" as
+119. The true motion was -240.8. Both fit the endpoints. That aliasing put the
wrong sign in the expert, which then drove yaw AWAY from the target until
bearing 180 became a stable equilibrium: **the drone flew the entire course
facing backwards**, commanding -0.7 reverse the whole way, while its x/y track
still looked correct because velocity setpoints are expressed in the heading
frame. `probe_velocity.py` now has a yaw-rate leg that unwraps, samples at
20Hz, and FAILS on both a wrong sign and a mis-scaled full-stick rate.

That backward flight was also the sole cause of the vertical oscillation chased
before it (z swinging 1.1-4.1m). With yaw fixed, z holds 1.34-1.76m. Do not
re-open the altitude investigation on the strength of the old traces.

### The scripted expert

`nanopilot/gate_expert.py`. Pure pursuit along the GATE CENTERLINE: the carrot
is the drone's own along-track position pushed forward by `lookahead`, at zero
lateral offset, so it converges onto the gate normal before arriving. Yaw turns
the nose toward the carrot, which also keeps the gate in the camera frame for
the student. Height is a proportional trim toward the gate height.

Two tuning findings, both non-obvious:

- **No cos(bearing) speed slowdown.** That is a car-like (nonholonomic) reflex
  and this vehicle is holonomic -- the setpoint is in the heading frame, so a
  misaligned nose still produces motion in the right world direction via the
  lateral axis. With it on, cruise collapsed to 0.375 m/s and an episode took
  849 steps (42s) to reach one gate. Off, episodes take ~265 steps (13s).
- **`height_gain` is a STABILITY constraint, not an accuracy one.** It wraps a
  second proportional loop around a hold that is already a position regulator,
  and the trim saturates at `|error| > TRIM_RANGE/Kz`, making it a relay. At
  Kz=2.0 z limit-cycled 1.65 -> 2.48 -> 1.38 -> 3.01 -> 1.11 -> 3.08. Kz=0.5
  gives zeta' = 0.67. Accuracy is not the tradeoff it looks like, because
  episodes anchor the hold setpoint at the gate height.

Run it with `gate_rollout.py --expert` (no policy files needed) and record tubs
with `rl_collect.py --expert`. Both force `control_altitude=True,
altitude_scale=1.0` so the height trim reaches the env unscaled.

### Acceptance result: the expert beats every RL teacher, with no training

`gate_rollout.py --expert --episodes=10` on `slalom_gates_80mm`:

    crossings=30 passes=30 (100%)
    |lateral offset|: mean 0.21m median 0.21m max 0.35m   (window 0.9m)
    height at crossing: mean 1.53m min 1.40 max 1.64      (gates at 1.50)
    approach speed: mean 1.54 m/s     misses: lateral 0, height 0

10/10 episodes 3/3, ~280 steps (14s) each. Compare the teachers:

| teacher | plant | 3/3 episodes | gates | height at crossing |
| --- | --- | --- | --- | --- |
| v4 @340k (best RL) | velocity damper | 10/10 | 30/30 | n/a |
| v5 @600k | PD hold | 3/10 | 19/30 | mean 1.73, range 1.22-2.62 |
| **scripted expert** | **PD hold** | **10/10** | **30/30** | **mean 1.53, range 1.40-1.64** |

It matches the best RL teacher ever produced, on the current plant, with far
tighter height control, in zero training time and with no checkpoint to pick.

### Verified along the way

- `probe_hover.py` after the change: z pinned, throttle 1477-1478, matching the
  2026-09-05b baseline. `direct_thrust` did not regress the hold.
- Straight flight causes NO climb: dz = 0.00 over 10s at both 1.5 and 2.0 m/s,
  and at 1.0 m/s lateral. The plant is clean.
- Yaw-induced thrust is now ~0.3 PWM (the 2026-08-23 PID retune evidently
  removed it). `DRONE_YAW_THROTTLE_FEEDFORWARD = 0.0` is correct; do not
  re-enable it on the strength of the old comments.

### Training-side consequence: TRAIN_ALTITUDE_PWM_SPACE is now OFF

That flag pre-applied `sign(a)*|a|**0.2` to the altitude labels because the axis
was thrust through that curve. Under velocity control the axis is a linear
height trim, the curve is not in the path, and pre-applying it would distort the
loss to fix a problem that no longer exists. **It must go back on if
`DRONE_VELOCITY_CONTROL` ever goes back off** -- train and inference read it from
config independently, so a mismatch is silent.

`eval_flight.py --alt-highpass` and `--alt-gain` now REFUSE to run under
`DRONE_VELOCITY_CONTROL`. Both operate in thrust/PWM space and were built to
diagnose the old integrating-thrust failure; against a linear height trim they
would silently distort a command that means something else. `--zero-altitude`
still works and now means "hold the takeoff height", which is a legitimate
ablation.

`TRAIN_LOSS_WEIGHTS` stays uniform. Altitude labels in an expert tub have
std 0.072 against roll 0.486, so the axis is still the smallest signal -- but it
is no longer FRAGILE, which is what mattered. A small altitude error is now
centimetres of height offset instead of an integrating thrust bias, so resist
re-tuning the weights unless flight data says otherwise.

### Distillation data

- `data/tub_18_26-09-06`: 558 rows, 2 episodes. Smoke artifact, safe to delete.
- `data/tub_19_26-09-06`: **10,717 rows, 40/40 episodes kept, 0 dropped**, all
  3/3 at `--min-gates=3`. 135MB. This is the training set.

**The yield is 100%**, against ~20% for the RL teacher, so distillation data is
now cheap: 40 episodes of clean 3/3 flight in ~25 minutes with no filtering.
Label distribution (throttle/roll are speed setpoints, altitude a height trim):

| axis | mean | std | range |
| --- | --- | --- | --- |
| user/angle (yaw) | -0.042 | 0.306 | -1.00 .. +0.99 |
| user/throttle (fwd speed) | +0.684 | 0.129 | -0.07 .. +0.75 |
| user/roll (lateral speed) | -0.068 | 0.497 | -1.00 .. +1.00 |
| user/altitude (height trim) | -0.028 | 0.053 | -0.19 .. +0.12 |

Altitude is still the narrowest axis, but it is no longer FRAGILE: a 0.05 error
is 5cm of height, not an integrating thrust bias. Note `user/throttle` is nearly
constant at +0.68 (the expert cruises); the visual signal lives in yaw and roll.

### BUG FIXED: the trainer deployed a worse model than it found

`torch_train.py` used one test for two different questions -- "are these the
best weights so far?" and "should we stop?" -- via
`val_loss < best_val_loss - MIN_DELTA`, where `MIN_DELTA` is an **absolute**
0.0005. That is sane at the ~0.01 losses of human-piloted tubs. It breaks
completely at expert-distillation scale, where the student reaches ~0.0003 and
the threshold is larger than the entire remaining loss, so no improvement can
ever clear it.

Observed on the first `tub_19` run: epochs 5, 7, 9 and 10 each beat the running
best and **none were saved**. The deployed model was epoch 8 (val 0.000654)
while epoch 10 reached 0.000317, and patience then expired on those same real
gains. Silent -- the log cheerfully reported "Best val_loss: 0.000654".

Now the two are separate: weights are saved on ANY true improvement, and
patience uses a scale-aware threshold (`MIN_DELTA_REL`, default 1% of the
current best, with optional `MIN_DELTA_ABS` floor) so it behaves identically at
0.01 or 0.0003. Replaying the observed loss sequence through the new logic keeps
0.000317 and still stops at epoch 13.

`models/bc_expert_v1_epoch8_stale.pth` is the pre-fix artifact, kept only for
comparison; delete it once `bc_expert_v1.pth` is validated.

The fix changed the outcome materially. Retrained on the same tub, the run went
33 epochs instead of stopping at 13, and best val went **0.000654 -> 0.000158**,
a 4.1x improvement that the old code discarded.

### The student: models/bc_expert_v1.pth

Trained on `data/tub_19_26-09-06` alone, best val 0.000158 at epoch 28 of 33.
Offline fidelity on that tub (in-sample, so optimistic, but the val loss agrees):

| axis | MAE | corr | previous best student (bc_rl_v4best_pwm) |
| --- | --- | --- | --- |
| yaw | 0.0060 | **0.9995** | 0.97 |
| pitch | 0.0051 | **0.9982** | 0.98 |
| roll | 0.0090 | **0.9997** | 0.99 |
| altitude | 0.0162 | **0.9365** | 0.86 |

Altitude MAE 0.0162 is ~1.6cm of height-target error at `TRIM_RANGE=1.0`.
Under the old thrust semantics the comparable error was 4.89 PWM of sustained
thrust, which integrated into the floor. Same axis, same fit quality, entirely
different consequence -- that is the whole point of the interface change.

### Closed-loop student result, and the NEXT structural blocker

`eval_flight.sh --model=models/bc_expert_v1.pth --world=slalom_gates_80mm
--runs=5 --duration=60 --rate-hz=20`:

| config | crash | survival | gates | alt_mean | alt_std |
| --- | --- | --- | --- | --- | --- |
| BC PWM space (old plant) | 5/5 | 18.4s | 0.4 | 0.93-1.63 | 0.57-0.81 |
| DAgger + velocity (old plant) | 5/5 | 37.4s | 0.4 | 1.10-1.29 | 0.57-0.63 |
| **expert-distilled, velocity mode** | **0/5** | **60.0s** | **1.0** | **1.40-1.42** | **0.11-0.12** |

**The altitude problem is over.** alt_std 0.11-0.12 against 0.57-0.81 for every
previous student, and 0.36-0.47 for the bare damper. Zero crashes, five for
five, where every prior configuration that commanded altitude crashed five for
five. That is the interface change doing exactly what it was for, and it should
not need revisiting.

**Gates are now limited by something else, and it is not the fit.** All five
episodes are near-identical: gate 1 passed at 3.0-4.0s, then nothing for the
remaining 57s while the drone flies on (87m of path, 29m max displacement) in
perfect control. Offline fidelity is 0.999/0.998/1.000/0.937, so the student
reproduces the expert almost exactly on the states it was shown.

**Root cause: the next gate is outside the camera at the moment of the turn.**
The camera is `horizontal_fov 1.39626` = 80 degrees, so +-40 degrees of view.
The slalom's geometry forces a larger turn than that:

| transition | world bearing | exit heading | relative bearing |
| --- | --- | --- | --- |
| gate 1 -> gate 2 | -38.7 deg | +28.6 deg | **-67.3 deg** |
| gate 2 -> gate 3 | +38.7 deg | -28.6 deg | **+67.3 deg** |

The drone exits each gate aligned with THAT gate's normal, which points it away
from the next one. The instrumented expert agrees: measured bearing to the next
target right after a crossing is +67.9 and -87.2 degrees. Both are far outside
+-40, so at the instant the expert commits to the turn, the student is looking
at empty space. It is being asked to regress a label computed from a gate it
cannot see -- the same class of problem as the velocity-observability finding of
2026-08-31, but about gate position.

Do NOT attack this with more data, more DAgger, or loss weights. The student
passes the gate it CAN see and fails the ones it cannot.

### FIX APPLIED: camera widened 80 -> 140 degrees, confirmed visually

`~/dev/aeroloop_gazebo/models/betaloop_drone_cam_80mm/model.sdf`:
`horizontal_fov` 1.39626 -> **2.44346** (80 -> 140 degrees). The old file is
backed up next to it as `model.sdf.fov80.bak`. Real FPV racing lenses are
150-170 degrees, so 80 was the artificial value, not 140.

Confirmed by rendering the actual view from the pose where the student went
blind -- position (5.21, 1.97, 1.57), yaw +17.7 deg, i.e. just after crossing
gate 1. Both images are in `docs/fov/`:

- `turn_view_fov80.png`: **only the distant gate 3 is visible, near centre.**
  Gate 2 (relative bearing 57.3 deg) is entirely out of frame. The student saw
  a far-off gate roughly straight ahead and no reason to turn -- which is
  exactly what it did: pass gate 1, fly on, never crash, never turn.
- `turn_view_fov140.png`: gate 2 is large and unmistakable on the right, with
  gate 3 still visible in the distance.

The 80 deg image is a geometrically exact reconstruction (the central +-40 deg
crop of the 140 deg frame; same optical centre and axis), not a re-render.

**Cost of the change**, which is real and worth remembering: the projection is
pinhole, so centre angular resolution degrades from 0.15 to 0.49 deg/px (focal
length 381 -> 116 px) and the frame edges stretch hard. Distant gates are ~3.3x
smaller in pixels than before. If that turns out to hurt, the principled fix is
Gazebo's `wideanglecamera` sensor with an equidistant lens, which spends pixels
uniformly across the field instead of piling them at the edges; it was not used
here because changing the sensor type risks the topic path the camera worker
depends on.

**This invalidates every tub recorded before it**, `tub_19` included -- the
optics differ, so the images are from a different sensor. `models/bc_expert_v1.pth`
is likewise 80-deg-only. The 65mm model SDF was NOT changed.

Re-collected at 140 deg: `data/tub_20_26-09-06`, **10,693 rows, 40/40 kept, 0
dropped**, again all 3/3. The expert is unaffected by the optics (it flies on
privileged state), so the yield stayed at 100%.

| tub | FOV | rows | use |
| --- | --- | --- | --- |
| `tub_18_26-09-06` | 80 deg | 558 | smoke artifact, delete |
| `tub_19_26-09-06` | 80 deg | 10,717 | superseded, wrong optics |
| **`tub_20_26-09-06`** | **140 deg** | **10,693** | **current training set** |

### The 140-degree student: models/bc_expert_fov140.pth

Trained on `data/tub_20_26-09-06` alone, 31 epochs, best val **0.000097**
(against 0.000158 for the 80-deg model). Offline fidelity improved on every
axis despite the 3.3x coarser centre resolution -- the wider field evidently
buys more than the lost pixels cost:

| axis | 80 deg (bc_expert_v1) | 140 deg (bc_expert_fov140) |
| --- | --- | --- |
| yaw | 0.9995 | **0.9997** |
| pitch | 0.9982 | 0.9980 |
| roll | 0.9997 | **0.9997** |
| altitude | 0.9365 | **0.9659** |
| best val loss | 0.000158 | **0.000097** |

### The wider camera did NOT fix gates. It changed the failure, not the count.

`eval_flight`, 5 x 60s, 20Hz, `bc_expert_fov140.pth`: crash 0/5, survival 60.0s,
**gates 1.0** -- identical headline to the 80-deg student. But the behaviour is
qualitatively different:

| | 80 deg | 140 deg |
| --- | --- | --- |
| gates | 1.0 | 1.0 |
| max displacement | 28.0-29.4m | **16.7-21.1m** |
| path length | 86.8-87.7m | **20.6-80.3m** (highly variable) |
| alt_std | 0.11-0.12 | **0.01-0.03** |
| which gate | gate_1 every time | gate_1 x4, **gate_3 once** |

The 80-deg student flew a consistent straight line past everything (87m of path,
29m out). The 140-deg student stays close (17-21m), flies erratic distances --
one episode covered only 20.6m in 60s, an average of 0.34 m/s, i.e. nearly
hovering -- and in one episode passed gate 3 instead of gate 1.

**Read that as indecision, not blindness.** The narrow camera made the student
fly confidently the wrong way; the wide one makes it dither. That is what you
would expect if the student can now SEE the next gate but cannot tell WHICH gate
to chase.

**Confirmed from the ground track** (`eval_flight --trace` now records x/y/yaw,
added for this). Two episodes, both 2/3:

    ep0  gate_1 @ 4.25s, gate_3 @ 15.9s     ep1  gate_1 @ 4.9s, gate_3 @ 16.7s
      t=5.6  (7.25, +2.32)                    t=5.6  (5.90, +2.67)
      t=8.4  (10.96, +1.75)   <-- gate 2      t=8.4  (9.70, +2.86)   <-- gate 2
      t=11.2 (14.44, +0.01)       is at        t=11.2 (12.97, +0.36)      is at
      t=14.0 (16.20, +2.01)       y = -2       t=14.0 (16.16, +1.44)      y = -2

The student flies **gate 1 -> gate 3, skipping gate 2 entirely**. It crosses
x=10 at y = +1.8 to +2.9 while gate 2 sits at y = -2; it never makes the dip.
After reaching gate 3 it mills around it for the remaining 30s, since nothing
lies beyond.

The mechanism is a direct consequence of the fix: after gate 1, gate 3 is nearly
straight ahead (17 deg off the nose) while gate 2 needs a 57 deg right turn. The
expert knows to chase gate 2 because it tracks `gate_idx`. The student sees two
near-identical gates in one frame with no ordering cue and flies to the centred
one. **Widening the field made the target visible and the target ASSIGNMENT
harder at the same time.**

Note this is NOT a fit problem: offline corr is 0.9997/0.9980/0.9997/0.9659 and
val loss 0.000097, the best of any model in this project. The student reproduces
the expert almost exactly on states it was shown. It cannot resolve which gate
is next from one frame, because that information is not in one frame.

### Do this next

1. DONE. `probe_velocity.py` re-run after the yaw fix: **all checks passed**.
   Full-stick yaw is now 120.2 deg/s against the configured 120 (was 423), and
   every rate is negative for a positive stick, confirming the sign. Forward
   tracks 1.00 m/s exactly; lateral 1.05 with the rise correctly reported as
   authority-limited (stick saturated 43% of the leg).
2. Distil: `rl_collect.py --expert --episodes=40`, then `torch_train.py`, then
   `eval_flight.sh --rate-hz=20`. The student now regresses speed setpoints, so
   a small error is a slightly wrong speed rather than an integrating thrust
   bias. **Existing tubs (15/16/17) and every existing checkpoint predate this
   interface and must not be mixed in.**
3. DONE. **Camera widened to 140 degrees** (see the section below).
4. **Resolve target assignment. This is now THE blocker.** "Which gate is next"
   is not recoverable from a single frame of a course with identical gates, so
   no amount of data or DAgger can fix it. In order of principle:
   a. **Short image history / recurrence.** `SEQUENCE_LENGTH=3` already exists
      and `TorchPilot` already keeps a 3-deep IMU history, but
      `LinearModel.forward(img, imu, vel)` takes ONE image. A frame stack would
      let the policy infer progress along the course. This is the honest fix and
      matches how the literature handles it.
   b. **A progress input.** Feed the policy a count of gates already passed
      (onboard-estimable: increment when a gate passes within a metre). Cheap,
      and it makes the label a function of the input again.
   c. **Visually distinct gates** (a colour per gate). A memoryless CNN can then
      learn a fixed priority order. Cheapest to try, but it only works for a
      fixed course and does not generalise -- treat it as an experiment that
      isolates the hypothesis, not as the fix.
   Recommended: (b) to confirm the diagnosis cheaply, then (a) as the real fix.
5. DAgger becomes nearly free: the expert is a function, so
   `dagger_collect.py` can label any visited state without loading a policy.
   Swap the teacher for `GateExpert` there. Worth doing AFTER the observability
   fix, not before -- DAgger cannot teach a policy to see around a corner.
6. Consider regressing gate centre and apparent size from the image and applying
   a fixed setpoint law, instead of end-to-end pixels-to-setpoints. Labels can be
   auto-generated by projecting gate corners through the camera.
7. Still open from earlier: 65mm has not been re-tested airborne on the tamed
   pitch/roll PID gains, and it now also inherits the corrected yaw scaling.
8. `DRONE_ALTITUDE_TRIM_RANGE = 1.0` is still un-swept.

## Current Handoff (2026-09-06c, course re-laid out to isolate the turn)

**STATE: stopped, nothing running.** Only `worlds/slalom_gates_80mm.sdf`
changed. No code, no config, no models.

### What changed and why

The 2026-09-06b blocker was TARGET ASSIGNMENT: after gate 1 the student saw two
identical gates in one frame, chased the more centred one (gate 3) and skipped
gate 2. The course now removes that ambiguity instead of asking the network to
resolve it, so the remaining question -- can a memoryless vision policy fly a
turn at all -- is testable on its own.

Gate order comes from `parse_world`, which sorts by MODEL NAME, so the layout is
expressed by editing poses in place; the names stay `gate_1/2/3`:

| gate | old pose | new pose | colour |
| --- | --- | --- | --- |
| gate_1 | (5, 2) yaw +28.6 | **unchanged** | yellow |
| gate_2 | (10, -2) yaw -28.6 | **(15, 2) yaw +28.6** (the old gate_3 spot) | cyan |
| gate_3 | (15, 2) yaw +28.6 | **(20.196, -1) yaw -30.0** | magenta |

The old middle gate is now the LAST gate, 6m from its predecessor at 30 degrees
off the +x track line, to the RIGHT (-y).

**The side is load-bearing, not arbitrary.** The drone exits a gate aligned with
that gate's normal, and gates 1 and 2 are yawed +28.6, so a LEFT (+y) offset
would have put the final gate ~1.4 degrees off the nose -- off the track on the
map, but no turn at all for the drone, and worse, more centred than gate_2 from
gate_1's exit, which would have recreated the exact skip. Measured geometry:

| transition | distance | world bearing | RELATIVE bearing |
| --- | --- | --- | --- |
| spawn -> gate_1 | 5.39m | +21.8 | +21.8 |
| gate_1 -> gate_2 | 10.00m | +0.0 | **-28.6** |
| gate_2 -> gate_3 | 6.00m | -30.0 | **-58.6** |

Decoy check from gate_1's exit: gate_2 sits at -28.6 deg and 10.0m, gate_3 at
-39.8 deg and 15.5m. The correct target is now both nearer AND more centred than
the distractor, which is the reverse of the old layout.

**Caveat worth deciding deliberately.** "30 degrees off the track" is 30 degrees
of MAP geometry, but the turn the drone actually flies is **58.6 degrees**,
because gate_2's normal points +28.6 while the leg to gate_3 runs at -30. If the
intent was a 30-degree turn as the DRONE experiences it, the one-value fix is to
set gate_2's yaw to 0 (`<pose>15 2 1.5 0 0 0</pose>`): gates 1 and 2 then lie on
one straight +x line, the exit heading is 0, and the final leg becomes exactly
-30. That was not done here because gate_2 is the old gate_3 and the instruction
was to leave it where it is.

### Verified: the scripted expert flies the new course

`gate_rollout.py --expert --episodes=5` on a fresh headless stack:

    crossings=15 passes=15 (100%)
    |lateral offset|: mean 0.14m median 0.06m max 0.36m   (window 0.9m)
    height at crossing: mean 1.50m min 1.35 max 1.75      (gates at 1.50)
    approach speed: mean 1.56 m/s    misses: lateral 0, height 0

5/5 episodes 3/3. The turn does cost accuracy in the expected place and only
there: the final gate's lateral offset is +0.28 to +0.36 in every episode
against 0.01-0.13 for the two straight gates. Still 2.5x inside the window.

### RESULT: re-distilled on the new layout. Gates did NOT improve; the
### failure mode changed from "flies straight past" to "flies in circles".

Full pipeline re-run on the new course (collect -> train -> eval):

- `data/tub_21_26-09-06`: **12,240 rows, 40/40 episodes kept, 0 dropped**, all
  3/3, 134MB. 100% yield again -- the expert is unaffected by the layout.
- `models/bc_expert_turn_v1.pth`: trained on that tub alone, best val
  **0.000087** at epoch 25, early-stopped at 30. That is the **best val loss in
  the project's history** (previous best 0.000097 for `bc_expert_fov140`).
- Offline fidelity on the tub (12,240 samples, uniform weights, score 0.0068):

| axis | MAE | corr |
| --- | --- | --- |
| yaw | 0.0051 | **0.9992** |
| pitch | 0.0029 | **0.9961** |
| roll | 0.0092 | **0.9995** |
| altitude | 0.0100 | 0.9239 |

`eval_flight.sh --runs=5 --duration=60 --rate-hz=20 --trace`:

| | old course (`bc_expert_fov140`) | new course (`bc_expert_turn_v1`) |
| --- | --- | --- |
| crash | 0/5 | **0/5** |
| survival | 60.0s | **60.0s** |
| gates | 1.0 | **1.0** (1, 2, 1, 1, 0) |
| alt_mean | -- | **1.53-1.55** (gates at 1.50) |
| alt_std | 0.01-0.03 | **0.02** |
| path length | 20.6-80.3m | **83.4-90.4m** |
| max displacement | 16.7-21.1m | 18.9-31.5m |

**Flight quality is not the problem and should not be re-investigated.**
Altitude is 1.53 +- 0.02 against a 1.50 target, zero crashes five for five,
cruise is the expert's 1.5 m/s (89m of path in 60s). The plant, the altitude
interface and the fit are all doing their jobs.

**The student flies large circles.** From the traces, every episode is the same
shape -- pass gate_1 (or miss it), then orbit at ~8-10m radius, repeatedly
looping back through the start. ep0 is representative:

    t= 4.3 ( 5.12, +1.89)  <- gate_1 passed
    t= 8.5 ( 8.65, +7.22)      turns LEFT, away from gate_2 at (15, +2)
    t=12.8 ( 6.85,+13.11)
    t=17.0 ( 0.95,+15.12)
    t=21.3 (-4.27,+11.74)
    t=25.5 (-4.59, +5.53)
    t=29.8 (-0.11, +1.01)  <- back at the start, a full loop
    t=38.3 (11.91, +0.48)      crosses x=12 at y=+0.5; gate_2 is at y=+2
    t=46.8 (23.07, +4.30)      loops again

ep1 passed **gate_3 at 18.9s and gate_2 at 55.4s -- out of order**, which is a
tracker artifact of orbiting through gate planes rather than flying a course.

This is a DIFFERENT failure from the 2026-09-06b one. The narrow-camera student
flew confidently straight past everything; the wide-camera student on the old
course dithered; this one turns continuously and never converges on a target.

**Interpretation, and what the data supports.** Offline corr is
0.9992/0.9961/0.9995 and val loss is the best ever recorded here, so the student
reproduces the expert almost exactly *on states the expert visits*. It has never
seen a state where the drone is mid-orbit with no gate in frame, because the
expert never produces one: 40/40 episodes are clean 3/3 runs with zero failures,
so the tub contains **no corrective data at all**. A small residual yaw error
takes the drone off the expert's manifold, and from there nothing in the
training set says "stop turning".

The new layout also made the labels *smoother*, which plausibly made this worse
rather than better:

| axis | tub_20 (old course) | tub_21 (new course) |
| --- | --- | --- |
| yaw std | 0.306 | **0.251** |
| yaw \|x\|>0.5 | 11.4% | **6.1%** |
| throttle std | 0.129 | **0.068** |
| throttle \|x\|>0.5 | 92.2% | **97.9%** |

The expert now cruises at a near-constant speed with less yaw activity, so there
is less visual-to-action signal to learn from, not more.

**Do not read this as the course change failing.** The change did what it was
specified to do -- the expert flies it 30/30, the correct gate is nearer and
more centred than the distractor, and gate assignment is no longer the binding
constraint. What it exposed is that target assignment was never the *only*
blocker: with it removed, the student still cannot hold a heading toward a gate
it has lost from frame.

One caveat to own about the layout: after gate_1 the two remaining gates sit at
-28.6 and -39.8 degrees, only **11 degrees apart** in bearing. That is tighter
angular separation than the old course had, even though the correct gate is the
nearer and more centred one. If a future experiment wants maximal separation,
move gate_3 further off-axis rather than further away.

### Infrastructure notes from this run (cost ~40 minutes)

- **`_wait_for_ready`'s IMU test can hang for 40+ seconds.** `imu_ready` is
  `any(|imu| > 1e-6)`, and a drone sitting motionless reads all-zero, so
  readiness only passes once something makes it twitch. Runs launched
  immediately after `STACK_READY` failed on `imu=False`; a ~25s settle before
  starting the collector fixed it.
- **`MSP query failed; assuming BF unreachable` is a real, repeatable stack
  failure, not the documented flakiness.** Three consecutive stack rebuilds all
  produced a BetaFlight that never armed, and every episode then died with
  `BetaFlight lost arming; rebuild the stack`. `MSPClient.query_status` is fine
  in isolation -- verified 15ms round-trips, `flags=268435460`
  (RX_FAILSAFE|ARM_SWITCH, expected before RC flows) -- so **check MSP health
  before spending 25 minutes on a collection**:

      uv run --env-file .env python -c "
      import sys; sys.path.insert(0,'nanopilot')
      from utils.betaflight import MSPClient
      print(MSPClient().query_status(timeout=2.0))"

  A `(None, [], None, [])` return means rebuild the stack now. Worth wiring
  into `rl_collect.py`/`start.sh` as a launch gate.
- A stack that passes that check ran 40 episodes with zero failures, so the
  retry-the-whole-stack loop is the right recovery -- it just needs the health
  check to decide *when* to retry instead of burning a full collection first.

### Do this next

1. **DAgger, with the scripted expert as the labeller.** This is now the
   indicated fix and it is nearly free: `GateExpert` is a pure function of
   privileged state, so `dagger_collect.py` can label any state the student
   visits without loading a policy. The specific gap is concrete and measurable
   -- the tub has zero off-manifold states because the expert never fails, and
   the student's failure is entirely off-manifold. Swap the teacher for
   `GateExpert` in `dagger_collect.py` and aggregate.
2. **Then the frame stack** (2026-09-06b item 4a). Still the honest fix for
   heading persistence: `LinearModel.forward(img, imu, vel)` takes ONE image,
   so the policy cannot tell "turning toward a gate" from "turning away". Gate
   it behind a config flag and record it in `arch_meta`.
3. A progress input (gates passed) remains the cheap confirmation, but note it
   would NOT explain circling on its own -- the drone loses the gates entirely,
   which is an observability failure about heading, not about ordering.
4. Superseded scope note: re-collecting and re-distilling against the new layout
   is DONE (this section). `data/tub_20_26-09-06` is stale on geometry.

### Superseded plan (kept for the record)

1. Re-collect and re-distil against the new layout. **`data/tub_20_26-09-06` is
   now stale on geometry** -- the optics match, but the labels are for the old
   zig-zag. `rl_collect.py --expert --episodes=40 --min-gates=3`, then
   `torch_train.py`, then `eval_flight.sh --rate-hz=20`.
2. Read the result as a turn test, not a gate count. gates 3.0 means the student
   can fly a 58.6-degree turn from a single frame. gates 2.0 with the drone
   flying past gate_3 means it cannot, and the 2026-09-06b fixes (frame stack /
   progress input) are still required -- but now for a reason that is about
   turning rather than about which gate is next.
3. Everything else in the 2026-09-06b "Do this next" list still stands.

## Current Handoff (2026-09-07, S-curve teaching course built and validated)

**STATE: stopped, nothing running.** New world `worlds/s_curve_gates_80mm.sdf`.
No code or config changed. `slalom_gates_80mm` is untouched and still valid for
comparison. Nothing has been collected or trained on the new course yet.

### Why a new course rather than DAgger first

The 2026-09-06c student circles: it passes gate_1 then orbits at 8-10m radius
without ever converging, while offline fidelity is 0.9992/0.9961/0.9995 and val
loss 0.000087 (best in project history). The failing skill is the gate-to-gate
TRANSITION, and the slalom tub contains 3 transitions x 40 episodes, only two of
which are distinct. DAgger was not chosen to lead because round 1 (2026-08-31)
regressed gates 0.6 -> 0.2 while doubling survival: adding recovery data on top
of an under-trained core skill trades task performance for staying alive. DAgger
remains the right SECOND step.

### Two geometry rules, both of which the slalom violated

1. **Gate yaw must be the path TANGENT, not the gate's own leg bearing.** The
   drone exits a gate aligned with that gate's normal. On the slalom, gate_1's
   normal points +28.6 deg while gate_2 lies at bearing 0, so the drone exits
   aimed 28.6 deg AWAY from its next target, at exactly the moment the student
   diverges.

2. **Constant curvature, NOT a zigzag.** This one is counter-intuitive and was
   checked numerically before building. In a symmetric serpentine with
   bisector-aligned gates, the bisector at a gate points along the mean course
   direction, which is exactly where gate n+2 sits:

       SERPENTINE (50 deg turns, 5m legs, bisector yaws)
         after g1:  g2 rel +25.0 deg @ 5.00m   g3 rel  +0.0 deg @ 9.06m
         after g3:  g4 rel -25.0 deg @ 5.00m   g5 rel  +0.0 deg @ 9.06m

   The distractor lands dead centre while the true next gate sits off to the
   side. That is the 2026-09-06b gate-skip failure built into the track. On a
   constant-curvature arc the next gate is nearer AND more centred at every
   transition.

### The course

`worlds/s_curve_gates_80mm.sdf`: left arc 160 deg, a 5.5m straight, right arc
160 deg. R=5.5m, one gate every 40 deg of arc, all at z=1.5.

| | slalom_gates_80mm | s_curve_gates_80mm |
| --- | --- | --- |
| gates | 3 | **10** |
| transitions per episode | 2 | **9** |
| worst transition bearing | 58.6 deg | **20.0 deg** |
| ambiguous transitions | 2 of 2 | **0 of 9** |
| turn directions | right only | **left, straight, right** |
| path / episode | ~14s | ~27s |
| max radius | 20.2m | 23.5m (`bounds_xy`=30) |

Every transition is +-20 deg except the straight (0 deg), and the sign reverses
once through the straight, so "turn the other way" and "straighten out" are both
represented. Verified with `parse_world`: next gate is more centred than every
other in-frame gate, or at least 2x nearer, at all 9 transitions.

**Gate names are ZERO-PADDED (`gate_01`..`gate_10`).** `parse_world` sorts by
model name as a STRING, so `gate_10` would otherwise sort between `gate_01` and
`gate_02` and silently scramble the course order. Any future course with 10+
gates must pad.

Note the arc curls back on itself, so up to 7 gates are in frame at once around
gate_03. The ambiguity check says the next gate is still the clearest at every
transition, but this is a busier visual scene than the slalom ever was.

Colours cycle yellow/cyan/magenta, so colour no longer identifies a unique gate.
That is not a loss: the 3-gate slalom DID give every gate a unique colour and the
student still chased the wrong one, so colour was never carrying the signal.

### Validated: the expert flies it 50/50

`gate_rollout.py --expert --episodes=5 --world=s_curve_gates_80mm`:

    crossings=50 passes=50 (100%)
    |lateral offset|: mean 0.11m median 0.12m max 0.38m   (window 0.9m)
    height at crossing: mean 1.56m min 1.18 max 1.89      (gates at 1.50)
    misses: lateral 0, height 0    near misses: 0

5/5 episodes 10/10, ~555 steps (27.8s) each. Lateral tracking is TIGHTER than on
the slalom (mean 0.11m vs 0.14-0.21m) despite 3.3x more gates and gates spaced
only 3.76m apart, so `GateExpert`'s 3.0m lookahead did not need retuning.

### Do this next

1. Collect and distil on the new course, same pipeline as 2026-09-06c:
   `rl_collect.py --expert --episodes=40 --min-gates=10 --world=s_curve_gates_80mm`,
   then `torch_train.py`, then
   `eval_flight.sh --world=s_curve_gates_80mm --runs=5 --duration=60 --rate-hz=20 --trace`.
   NOTE `--min-gates=10` now, not 3. Expect ~12k rows from 40 episodes at ~555
   steps each; budget ~25 min of collection.
2. Read the result as a transition test. The student now gets 9 transitions per
   episode instead of 2, all within 20 deg, with the correct gate unambiguous.
   If it still circles, coverage of the on-manifold skill was NOT the binding
   constraint and DAgger (or the frame stack) is required.
3. Then DAgger with `GateExpert` as the labeller, which is nearly free: the
   expert is a pure function of privileged state, so `dagger_collect.py` can
   label any visited state without loading a policy.
4. Use the MSP health gate before any long collection (see 2026-09-06c infra
   notes) -- it caught nothing this run, but it turns a 25-minute wasted
   collection into a 30-second retry.

### RESULT: distilled on the S-curve. Gates 1.0/3 -> 3.4/10, and the best
### episode went 2/3 -> 9/10. A NEW failure mode appeared: the student FREEZES.

- `data/tub_22_26-09-07`: **22,202 rows, 40/40 kept, 0 dropped**, every episode
  10/10. 304MB. 360 gate transitions against 80 in the slalom tub.
- `models/bc_expert_scurve_v1.pth`: best val **0.000052** at epoch 31,
  early-stopped at 46. Best in project history (slalom student: 0.000087).

`eval_flight.py --runs=5 --duration=60 --rate-hz=20 --trace`:

| | slalom student | S-curve student |
| --- | --- | --- |
| crash | 0/5 | **0/5** |
| survival | 60.0s | **60.0s** |
| gates | 1.0 / 3 | **3.4 / 10** |
| per-episode | 1,2,1,1,0 | **9,0,4,4,0** |
| best episode | 2/3 | **9/10** |
| alt_std | 0.02 | 0.03-0.04 |

**The headline fraction barely moved (33% -> 34%), but that reading is wrong.**
ep0 passed gates 01,02,04,05,06,07,08,09,10 -- it missed only gate_03 and flew
the ENTIRE remaining course in 30.7s against the expert's 27.8s, chaining 8
consecutive transitions at a correct 1.49 m/s cruise. No previous student ever
chained more than one. **The transition skill IS learnable from this data.**
What loses the other episodes is not the skill; it is getting captured by
off-manifold attractors.

### The new failure: a stable frozen fixed point

Two of five episodes stop dead and hover in place for the rest of the run:

    ep0: 9/10   frozen  0.3% of steps  cruise 1.49 m/s  first stop: never
    ep1: 0/10   frozen 94.4% of steps  cruise 1.17 m/s  first stop:  3.6s
    ep2: 4/10   frozen  6.1% of steps  cruise 1.49 m/s  first stop:  6.9s
    ep3: 4/10   frozen  5.1% of steps  cruise 1.49 m/s  first stop:  7.0s
    ep4: 0/10   frozen 88.2% of steps  cruise 1.37 m/s  first stop:  7.4s

ep1 travelled 4.2m total, parked 0.93m short of gate_01, and held position to
within 2cm from t=10s to t=60s. ep4 did the same at 4.16m from gate_01. ep2
froze 0.95m from gate_05 at the very end.

Under `DRONE_VELOCITY_CONTROL` the throttle axis is a SPEED SETPOINT, so a
throttle output near 0 does not coast -- it commands a full stop, and the PD
altitude hold parks the drone. That is self-reinforcing: stopped drone -> the
camera frame stops changing -> the memoryless policy emits the same near-zero
command -> stopped forever. A perfect closed-loop fixed point.

Note the expert makes this state maximally unlikely to be learned: `user/throttle`
in `tub_22` has mean +0.705, std 0.068, and **97.9% of frames above 0.5**. The
expert essentially never slows down, so "stopped" is as far off-manifold as a
state can be, and there is not one frame teaching recovery from it.

This is a different attractor from the 2026-09-06c circling (a limit cycle);
both are off-manifold traps for a memoryless policy, and both come from the same
root cause: 40/40 flawless expert episodes contain zero corrective data.

### Do this next: DAgger, and the case for it is now empirical

The course change did its job and should not be re-litigated: it proved the
transition skill is learnable (9/10 in one episode, 8 consecutive transitions)
and it removed gate assignment as the binding constraint. What remains is purely
recovery from off-manifold states, which is exactly and only what DAgger adds.

1. **DAgger with `GateExpert` as the labeller.** `dagger_collect.py` already
   flies the STUDENT and keeps crashed episodes; swap its teacher for
   `GateExpert` (a pure function of privileged state, so no policy to load and
   any visited state can be labelled). The frozen state is trivially labelable:
   the expert says "1.5 m/s toward gate N" at a dead stop. Aggregate with
   `tub_22` and retrain.
   Watch for the 2026-08-31 regression: round 1 traded gates for survival.
   Here survival is already 5/5, so there is nothing left to trade -- which is
   itself a reason this round should behave better than that one.
2. If freezing persists, the cheap mechanical guard is a throttle floor at
   inference (clamp the speed setpoint to >= ~0.3 when no gate is passed
   recently). That masks rather than fixes, so try DAgger first.
3. The frame stack (2026-09-06b item 4a) remains the structural fix for the
   memoryless-ness that makes both attractors possible.

### BUG: early stopping is decided by validation NOISE, not by convergence

The first training run on `tub_22` stopped at **epoch 7 with val 0.001186** and
deployed the epoch-2 weights while `train_loss` was still falling smoothly
(0.00144 -> 0.00040). Raising `EARLY_STOP_PATIENCE` 5 -> 15 in
`drone_config_80mm.py` and re-running the SAME data and code gave **0.000052 at
epoch 31**, a **23x improvement**, stopping at epoch 46.

Cause: validation loss on expert-distilled tubs swings ~10x epoch to epoch while
train loss falls smoothly, so whether patience expires depends on whether an
EARLY epoch happened to draw a lucky val value. On `tub_22` epoch 2 drew 0.001186
and nothing beat it for 5 epochs. On the smaller slalom tub the early epochs drew
UNLUCKY values (0.011, 0.029), patience kept resetting, and the identical code
ran 30 epochs to 0.000087. Same trainer, opposite outcome, from val noise alone.

This is the SECOND instance of the "trainer stops on real gains" failure class.
The 2026-08-24 fix separated best-weight SAVING from the STOPPING test; this one
is that the stopping test itself is dominated by noise. **FIXED (2026-09-07):** patience now runs on the MEDIAN of the last
`EARLY_STOP_SMOOTH_WINDOW` (default 3) validation losses, and epochs before the
window fills cannot seed `patience_ref`. Best-weight SAVING still uses the raw
value -- that is the 2026-08-24 fix and smoothing it would deploy weights from
an epoch that was never actually best.

**The median LAGS, so it needs MORE patience, not less.** Replaying the real
observed sequences:

| sequence | window=1 pat=5 (old) | window=3 pat=5 | window=3 pat=15 (now) |
| --- | --- | --- | --- |
| tub_22 S-curve, 46 ep | stop @36, 0.000053 | stop @26, **0.000068** | stop @42, **0.000052** |
| tub_21 slalom, 30 ep | stop @30, 0.000087 | stop @19, **0.000164** | stop @30, **0.000087** |
| lucky-epoch-2 prefix | stop @7, **0.001186** | stop @8, 0.000573 | stop @42, **0.000052** |

Smoothing at the STOCK patience is WORSE than no smoothing on both real runs.
The two settings must move together; `EARLY_STOP_SMOOTH_WINDOW = 3` and
`EARLY_STOP_PATIENCE = 15` are both set in `drone_config_80mm.py`.

### Infra: eval_flight.sh has the same IMU-readiness race as the collector

`eval_flight.sh` launches `eval_flight.py` immediately after its stack comes up,
so it hits the `imu=False` readiness stall documented in 2026-09-06c. Two of
three attempts died on it. Bring the stack up separately, wait ~25s, run the MSP
health check, then call `nanopilot/eval_flight.py` directly.

**`eval_flight.py` takes NO `--world` flag** -- it reads `GZ_WORLD` from the
environment (`eval_flight.py:422`). Only the shell wrapper accepts `--world`.
Passing it to the Python entrypoint is an argparse error (exit 2).

The MSP health gate earned its keep: it caught a dead board on one attempt and
rebuilt in ~30s instead of after a failed eval.

## Current Handoff (2026-09-07b, DAgger round 1: THE COURSE IS SOLVED)

**STATE: stopped, nothing running.** `models/bc_dagger1_scurve.pth` flies
`s_curve_gates_80mm` **10/10 gates, 5/5 episodes, at expert pace**. This is the
first vision student in the project's history to complete a full course.

### Result

`eval_flight.py --runs=5 --duration=60 --rate-hz=20`:

    success_rate 1.0   crash_rate 0.0   mean_gates 10.0/10   mean_survival 27.61s

| ep | result | gates | time | path | alt |
| --- | --- | --- | --- | --- | --- |
| 0 | completed | 10/10 | 28.31s | 41.8m | 1.56 +- 0.01 |
| 1-4 | completed | 10/10 | 27.4-27.6s | 40.3-40.5m | 1.54-1.55 +- 0.01 |

The course's geometric path is 40.6m and the scripted expert flies it in ~27.8s.
The student matches both: **40.3-41.8m of path (no wandering) and 27.4-28.3s**.
Height holds 1.54-1.56m against gates at 1.50m with std 0.01.

Progression across the whole arc of this work:

| student | course | gates |
| --- | --- | --- |
| `bc_expert_fov140` | slalom (3 gates) | 1.0 / 3 |
| `bc_expert_turn_v1` | slalom re-laid (3 gates) | 1.0 / 3 |
| `bc_expert_scurve_v1` | S-curve (10 gates) | 3.4 / 10 |
| **`bc_dagger1_scurve`** | **S-curve (10 gates)** | **10.0 / 10** |

### What was done

`nanopilot/dagger_collect.py` gained `--expert` (plus `--cruise-speed`,
`--lookahead`, `--height-gain`), mirroring `rl_collect.py`/`gate_rollout.py`:
`--policy` is now optional, the SB3 import is skipped, env config is forced to
`{"control_altitude": True, "altitude_scale": 1.0}` so the linear height trim
reaches the env unscaled, and labels come from
`GateExpert.act_from_telemetry(tel, gate_env._gate_idx)`.

`data/tub_23_26-09-07`: 27 episodes, **19,505 rows**. (Asked for 30; the run
died on episode 28 with `Could not reset + take off after 8 attempts`. 27 was
plenty -- the tub is complete and loads cleanly, verified through `TubDataset`.)

The coverage DAgger actually bought, which is the whole point:

| | tub_22 (expert rollouts) | tub_23 (DAgger) |
| --- | --- | --- |
| stopped (<0.15 m/s) | **0.4%** | **21.2%** |
| throttle range | +0.596 .. +0.750 | **-0.750 .. +0.750** (reverse) |
| throttle std | 0.029 | **0.459** |
| yaw abs>0.5 | 0.0% | **31.7%** |
| roll abs>0.5 | 41.3% | **67.2%** |

At stopped states the expert labels throttle **+0.452** ("go forward").
`tub_22` contains 12 stopped samples in total; `tub_23` has ~4,100. That is the
freeze fix, and it worked.

Trained on **both tubs aggregated** (41,707 samples), `--tubs=data/tub_22_26-09-07,data/tub_23_26-09-07`.

### LESSON: validation loss on DAgger data does not predict closed-loop success

This run's numbers looked alarming and were read as a likely failure. They were
not:

    ep1  train 0.012838  val 0.067819
    ep17 train 0.000495  val 0.063387   <- best
    ep22 train 0.000614  val 0.071640

Train fell 40x while val never went below 0.063 -- a **~200x train/val gap**,
against 1.2x for `tub_22` alone (train 0.000044, val 0.000052). The obvious
reading was "memorizes but cannot generalize; the recovery labels are not a
function of the image." **That reading was wrong**, and the closed-loop result
(10/10, 5/5) refutes it.

Why the signal misleads: the split is CHRONOLOGICAL, so the val set is the tail
of `tub_23` -- off-manifold states induced by the OLD student. The retrained
student does not make those errors, so it never visits those states. A high
error on a distribution the improved policy no longer induces costs nothing.
**DAgger validation loss is a moving target: it measures the previous policy's
mistakes.** Judge a DAgger round by `eval_flight`, never by val loss, and do not
compare it to a pure-expert run's loss.

### Do this next

1. The 10-gate S-curve is solved; it can no longer discriminate between models.
   Build a harder benchmark before tuning anything further -- tighter gate
   spacing, larger turn angles, varied gate heights, or a course whose ordering
   is genuinely ambiguous from one frame.
2. **The memory work is now unmotivated by evidence.** The frame stack and the
   visual GRU were justified by the target-assignment failure, and a memoryless
   policy just flew 10 gates with 9 transitions. Do not build them until a
   benchmark actually demands them.
3. DAgger round 2 is available but has nothing to fix on this course.
4. Still open from earlier: 65mm has not been re-tested airborne on the tamed
   pitch/roll PID gains and the corrected yaw scaling.

## Current Handoff (2026-09-08, varied gate heights: course built and SOLVED)

**STATE: stopped, nothing running.** `models/bc_dagger1_heights.pth` flies the
new varied-height course `s_curve_heights_80mm` **10/10 gates, 5/5 episodes**,
at expert pace, zero crashes. The altitude axis is now genuinely exercised:
expert labels have std **0.331** against 0.077 on the flat course.

### Result

`eval_flight.py --runs=5 --duration=60 --rate-hz=20`, all five episodes
`completed` in 27.1-28.1s (the scripted expert flies it in ~27.8s):

| student | course | gates | crossing height err |
| --- | --- | --- | --- |
| `bc_expert_scurve_v1` | flat S-curve | 3.4 / 10 | n/a (flat) |
| `bc_dagger1_scurve` | flat S-curve | 10.0 / 10 | n/a (flat) |
| `bc_expert_heights_v1` | **heights** | 5.6 / 10 | 0.271m |
| **`bc_dagger1_heights`** | **heights** | **10.0 / 10** | **0.254m (max 0.465)** |

The DAgger student tracks height BETTER than the expert it was distilled from
(0.254m vs 0.34m mean crossing error), which is the usual smoothing effect and
not a bug.

The expert-only student reproduced the pattern of 2026-09-07 exactly: it flew
well on-manifold (one episode 10/10 in 42.9s) and lost the others to the known
off-manifold traps -- one episode FROZE at (-5.71, 12.24) and held position to
within 2cm from t=28s to t=60s, one crashed. One DAgger round fixed it, again.

### The course: worlds/s_curve_heights_80mm.sdf

x/y/yaw are IDENTICAL to `s_curve_gates_80mm`, so a regression is attributable
to height and nothing else. Heights are 1.50 2.16 2.21 1.57 1.20 1.97 2.29 1.80
1.23 1.43 (range 1.20-2.29). Gate 01 sits at `start_alt` so there is no opening
transient.

**The profile is a HELIX, not per-gate steps, and that is forced by the plant.**
`scripts/probe_alt_step.py` (new) measures the PD altitude hold as a
second-order system with **w_n = 0.46 rad/s, zeta = 0.42** -- ~5s to first reach
a moved target, ~20s to settle. Gates are 3.76m apart, i.e. 2.5s at cruise, so a
per-gate step profile is a square wave five times faster than the loop can
follow. That was tried first (max step 0.90m) and the EXPERT managed only
49-62% of crossings, missing every one on height, with z swinging 0.2m to 3.9m.
Height is now `z(s) = 1.75 + 0.55*sin(2*pi*s/18 + phi)` along the 35.6m of
track: 18m wavelength, peak climb rate 0.29 m/s. Both arcs carry a climb AND a
descent, so the student cannot learn "turning left means climb".

**Gates are closed rectangles here** (1.9m wide x 1.9m tall; posts at y=+-1.0,
bars at z=+-1.0), unlike the goal-shaped 3m-tall gates of the flat courses. A
gate whose only horizontal member is a bar 1.5m above its centre has no visible
vertical centre, and height is the whole point of this course.

The opening was SIZED ON THE EXPERT'S MEASURED ERROR, not chosen for looks. At
a 1.4m-tall gate (window +-0.6m) the expert sat right on the edge and its pass
rate swung 88-100% run to run; +-0.8m leaves margin.

### The pass window is now PER GATE, derived from the SDF

An absolute 0.2-3.0m band would score a pass through a gate at z=2.29 for a
drone cruising at 1.5m, i.e. it would not measure height at all.
`eval_flight.parse_world` now measures each gate's tallest box (the posts) and
returns `z_min`/`z_max` = `gate_z +- (post_height/2 - 0.2)`; `GateTracker` and
`GateChaseEnv` both consume it and fall back to the old absolute band for a gate
with no measurable geometry. On the flat courses this yields **0.2-2.8m at a
gate of z=1.5**, which is what they always used, and gate ORDER is unchanged.

`GateTracker` also records EVERY crossing now (`gate_crossings` in the JSON,
`mean_crossing_height_err_m` in the aggregate). Read that, not gate count, to
judge height control: the count only says whether the drone was inside the
window. Crossings wider than `2 * half_width` laterally are discarded -- a
gate's local x=0 plane is infinite, so the drone crosses it whenever it moves
past that line anywhere in the world, and those samples would poison the
statistics. Pass counting was never affected by them.

### GateExpert: the height law changed, and it had to

Old law: `altitude = Kz * (gate_z - z) / TRIM_RANGE`. Under
`DRONE_VELOCITY_CONTROL` the env holds height at
`setpoint + altitude * TRIM_RANGE`, so a pure proportional command settles at
`(setpoint + Kz*gate_z)/(1 + Kz)` -- with Kz=0.5 that leaves **67% of
(gate_z - setpoint) as PERMANENT error**. On a flat course the setpoint IS the
gate height and the error is identically zero, which is why this never showed.
On a varied-height course it is fatal: a gate 0.9m above the setpoint would be
approached to within 0.6m and never reached.

The law is now a FEED-FORWARD plus a proportional term:

    z_ref     = the course height at the drone's own along-track position
                (piecewise-linear between gates; GateExpert._height_ref)
    altitude  = ((z_ref - setpoint) + Kz * (z_ref - z)) / TRIM_RANGE

The feed-forward asks for exactly the height wanted, with no steady-state error.
The setpoint is frozen at `start_alt` by every takeoff path, so it is a constant
per episode and the label stays a pure function of the gate -- which is what
keeps it learnable by a memoryless student. **Verified numerically identical to
the old law on a flat course** (max difference 0.00e+00 over the whole
s_curve_gates layout), so nothing about the solved flat course changed.

`height_lookahead` shifts `z_ref` forward to cancel loop lag and **defaults to
0**. Keep it small: in flight the hold tracks a 0.29 m/s ramp with almost no
lag, so a 3m lead put the drone 0.4-0.6m PAST every gate's height at the
crossing -- the lead error simply replaced the lag error.

`height_gain` default is now **0.75**, chosen by sweep on this course:

| Kz | crossings passed | error tail |
| --- | --- | --- |
| 0.5 | 97% | max 0.64m |
| **0.75** | **98% (49/50, 4 of 5 episodes 10/10)** | **max 0.83m** |
| 1.0 | 96% | max 1.19m |
| 1.5 | 79%, **crashed 2 of 3 episodes** | max 1.10m |

### Artifacts

| path | what |
| --- | --- |
| `worlds/s_curve_heights_80mm.sdf` | 10-gate varied-height S-curve |
| `data/tub_24_26-09-07` | 22,230 rows, expert rollouts, **40/40 kept, all 10/10**, 274MB |
| `data/tub_25_26-09-08` | 18,563 rows, DAgger round 1, 30/30 kept |
| `models/bc_expert_heights_v1.pth` | expert-only student, best val 0.000079, 5.6/10 gates |
| **`models/bc_dagger1_heights.pth`** | **DAgger student, 10/10 gates -- CURRENT BEST on this course** |
| `scripts/probe_alt_step.py` | altitude step response: overshoot, implied w_n/zeta |

Offline fidelity of `bc_expert_heights_v1` on tub_24 (in-sample):
yaw 0.9998, pitch 0.9965, roll 0.9998, **altitude 0.9991** (against 0.9239 on
the flat course, where the axis carried almost no signal).

DAgger coverage, which is what the round bought:

| | tub_24 (expert) | tub_25 (DAgger) |
| --- | --- | --- |
| throttle range | +0.57 .. +0.75 | **-0.75 .. +0.75** (reverse) |
| throttle std | 0.029 | **0.332** |
| teacher says "stop" (thr < 0.15) | **0.0%** | **8.8%** |
| yaw \|x\|>0.5 | 0.0% | **13.8%** |

As on 2026-09-07b, DAgger val loss is meaningless here: best val was 0.0277
against 0.000079 for the expert-only run, a ~350x gap, and the DAgger model
flies 10/10 while the other manages 5.6. The split is chronological, so val is
the tail of the DAgger tub -- off-manifold states induced by the OLD student,
which the retrained one never visits. **Judge DAgger rounds by eval_flight only.**

### The binding constraint is now the altitude BANDWIDTH, not the student

Worth knowing before designing a harder height course. With this plant you
cannot have both an expert that tracks reliably and a height spread much larger
than twice the pass window, because the hold's rise time (~5s) is twice the
gate-to-gate interval (2.5s). The levers, in order:

1. **Raise `DRONE_ALTITUDE_HOLD_KP` and `DRONE_ALTITUDE_HOLD_K` together.**
   w_n = sqrt(s * Kp) with a measured thrust slope s ~ 0.007 m/s^2 per PWM, so
   Kp=30 gives the 0.46 rad/s measured. Doubling both raises bandwidth 1.41x at
   constant damping. Watch the +-200 PWM clip on `alt_hold_bias`: at high K the
   velocity term alone saturates around 0.6 m/s of climb, and a saturated
   proportional loop is a relay.
2. Shorter gate spacing or a slower cruise, which buys the same thing by giving
   the loop more time per transition.
3. Do NOT just shrink the pass window. That was tried (+-0.6m) and it only made
   the EXPERT unreliable.

`DRONE_ALTITUDE_TRIM_RANGE = 1.0` is still un-swept, but it is no longer
un-exercised: with `start_alt = 1.5` it caps reachable heights at 0.5-2.5m, and
the course's 1.20-2.29 uses 79% of that authority.

### Do this next

1. The heights course is solved and can no longer discriminate. If a harder
   benchmark is wanted, raise the altitude-loop bandwidth first (above) and only
   then increase the height amplitude or shorten the wavelength -- the plant,
   not the student, is what limits the profile today.
2. Re-validate `bc_dagger1_scurve` on the flat course if you want belt and
   braces. It was NOT re-run this session; the reasoning that nothing changed
   for it is that the expert's law is bit-identical at constant gate height and
   the scoring window moved only from 0.2-3.0 to 0.2-2.8, against a student that
   flies at 1.54-1.56m.
3. Still open from earlier: 65mm has not been re-tested airborne on the tamed
   pitch/roll PID gains or the corrected yaw scaling.

### Infra notes

- The MSP health gate earned its keep twice and cost time once: a DAgger run was
  launched while the gate had just printed `(None, [], None, [])` and every
  episode died with `BetaFlight lost arming`. READ the gate's output before
  launching, not just run it.
- `eval_flight.py` writes its JSON with `--json-output`, not `--output`.
- `stop_all.sh` chained after other commands in one shell has killed the calling
  shell mid-sequence here. Run it on its own line.

## Current Handoff (2026-09-08b, altitude bandwidth swept: the plant got better and the FLIGHT got worse)

**STATE: stopped, nothing running.** `nanopilot/drone_config_80mm.py` and
`nanopilot/rl_env.py` are back at their 2026-09-08 values -- **no flight
behaviour changed this session**, and the revert is verified: the expert flies
`s_curve_heights_80mm` 30/30 crossings, 3/3 episodes 10/10, lateral mean 0.11m. What landed is tooling: an MSP launch gate in
`start.sh`, a new `scripts/start_gated.sh`, and a `--gains` sweep mode in
`scripts/probe_alt_step.py`.

### The headline, because it is counter-intuitive

The 2026-09-08 handoff named the altitude-loop bandwidth as "the binding
constraint... the plant, not the student" and said to raise
`DRONE_ALTITUDE_HOLD_KP`/`_K` first. That was measured, and the first half is
true: the step response improves enormously and there is no stability limit
anywhere in a 10x gain range. **But the expert then flies the course WORSE at
every outer gain tested, and crashes at most of them.** The gains were reverted.

**Do not re-run this sweep expecting a different answer.** The open question is
not "is the loop faster" (yes, dramatically) but "why does a faster inner loop
hurt the cascade", and there is a concrete hypothesis below.

### Measured: the PD altitude hold from Kp=30 to Kp=320

`scripts/probe_alt_step.py --amplitude=0.5 --hold-s=12 --gains=...`, one flight
per ladder (gains are read fresh every control cycle, so they can be swapped
mid-air -- that is what `--gains` exploits). Ladder built for constant damping:
`w_n = sqrt(s*Kp)`, `zeta = sqrt(s)*K/(2*sqrt(Kp))`, so Kp scales as r**2 and K
as r.

| Kp:K | w_n | zeta | settle | ss err | \|bias\| 0.5m step | \|bias\| 1m step |
| --- | --- | --- | --- | --- | --- | --- |
| **30:90** (current) | 0.47 | 0.36 | 24.4s | 0.10m | 12 | 33 |
| 60:127 | 0.56 | 0.57 | 12.4s | 0.05m | 26 | 63 |
| 90:156 | 0.66 | 0.65 | 9.4s | 0.03m | 42 | 93 |
| 120:180 | 0.73 | 0.69 | 8.0s | 0.02m | 57 | 123 |
| 180:220 | 0.88 | 0.73 | 6.2s | 0.02m | 84 | 183 |
| 240:255 | 0.99 | 0.76 | 5.4s | 0.01m | 117 | 243 (sat 1%) |
| 320:294 | 1.14 | 0.79 | 4.5s | 0.01m | 157 | 323 (sat 1%) |

Three things in that table are worth keeping:

- **There is no stability limit in this range.** zeta IMPROVES monotonically,
  0.36 -> 0.79. The 2026-09-08 lag argument predicted damping would degrade as
  bandwidth rose; it does the opposite. The loop is far more forgiving than
  assumed.
- **The real ceiling is actuator saturation, not stability.** `alt_hold_bias`
  clips at +-200 and `TRIM_RANGE = 1.0` lets a policy command a full 1.0m
  offset, so `Kp*TRIM_RANGE` plus the velocity term must fit inside 200 or the
  commanded authority becomes a relay. Kp=120 is the largest rung that keeps the
  WHOLE commanded range linear (123 pwm); Kp>=240 saturates outright. **Kp and
  TRIM_RANGE are coupled -- raising one means lowering the other.**
- **The old handoff's scaling rule was wrong.** "Doubling both raises bandwidth
  1.41x at constant damping" -- doubling both raises w_n by 1.41 AND zeta by
  1.41. Constant damping needs Kp*r**2 with K*r.

Also measured: the steady-state hover offset is pure `bias/Kp`. The hold settled
at 1.76 / 1.59 / 1.55 / 1.53m against a 1.50 setpoint at Kp=30/60/90/120 -- i.e.
there is a residual ~8 PWM thrust bias, and Kp is what bounds it.

### And yet: at Kp=120/K=180 the expert gets worse at every outer gain

`gate_rollout.py --expert --episodes=3 --world=s_curve_heights_80mm`, sweeping
`--height-gain` (Kz), on Kp=120/K=180:

| Kz | crossings passed | episodes (gates/10) |
| --- | --- | --- |
| 0.0 | 23/27 (85%) | 8, 6, 9 -- no crashes, one timeout |
| 0.25 | 18/25 (72%) | 5, 7, 6 -- **all crashed** |
| 0.5 | 18/22 (82%) | 5, 7, 6 -- **all crashed** |
| 0.75 (the tuned default) | 10/12 (83%) | 3, 4, 3 -- **2 crashed** |
| --- | --- | --- |
| **Kp=30/K=90, Kz=0.75 (reverted)** | **30/30 (100%)** | **10, 10, 10 -- no crashes** |

The failure is always the same shape and it is NOT overshoot: on the descending
half of the course the drone sits ABOVE the reference and will not come down.
At Kz=0.0, gate-by-gate dz went +0.01, +0.20, -0.41, +0.15, **+0.66, +1.15,
+1.29** -- tracking tightly until the descent, then falling behind it. At Kz=0.75
it lost the descent even earlier (dz=+0.95 at gate 3) and crashed.

**The hypothesis worth testing next, and the reason Kz cannot fix it.** The
expert commands `z_target = z_ref + Kz*(z_ref - z)`, and the env realises it as
`setpoint + altitude*TRIM_RANGE` with `altitude` CLIPPED to [-1, 1]. The
setpoint is frozen at `start_alt = 1.5` and `TRIM_RANGE = 1.0`, so the reachable
band is exactly 0.5-2.5m while the course spans 1.20-2.29. At gate_07 (z=2.29)
the feed-forward term alone is +0.79 of the available 1.0, so **any Kz
correction on the high gates is clipped away** -- which is why raising Kz makes
things worse rather than better, and why Kz=0 is the best of a bad set. Raising
Kp does not add authority here; it only makes the drone track a command that is
already saturated. The lever to try is `TRIM_RANGE` (still un-swept, flagged as
such since 2026-09-08), bounded by `Kp*TRIM_RANGE <~ 200`.

### FIXED: start.sh handed over MSP-deaf boards (this was the day's real bug)

BetaFlight SITL was measured aborting at launch on **~50% of attempts** today
(6 isolated launches: 3 died outright). `start_betaflight`'s health test was
`kill -0` after 2s -- process-alive, not MSP-alive -- so a board that cleared
that check and was still deaf got handed over, and every flight then died with
`BetaFlight lost arming`. Its log said so plainly and nothing acted on it:

    Features: MSP_FEATURE_CONFIG returned 0 bytes — skipping airmode check
    Status: MSP_STATUS_EX returned 0 bytes

`start_betaflight` now calls a new `betaflight_msp_alive` helper and retries (8
attempts) until MSP actually ANSWERS. Verified: a launch that survives startup
answers MSP every time (`status=772`), so "deaf board" and "SITL died at
startup" are the same failure -- **this is the root cause of the "three
consecutive rebuilds all produced a board that never armed" sessions.**

`scripts/start_gated.sh` wraps the whole stack in the same gate:

    GZ_HEADLESS=1 GZ_WORLD=s_curve_heights_80mm \
      bash scripts/start_gated.sh --no-manage --airframe=80mm

It stops, launches, waits for `STACK_READY`, settles 25s (the documented
`imu=False` readiness race), runs the MSP check, and rebuilds on failure; exits
0 with the stack up, 1 if every attempt failed. Since both fixes went in, every
stack has come up healthy on attempt 1.

**Its own bug is worth knowing about**, because it looked exactly like a launch
failure: `start.sh` runs `stop_all.sh` from its EXIT trap, so a dying `start.sh`
lands a SIGKILL some seconds AFTER you asked it to stop. The first version slept
2s between attempts and the previous attempt's trap then killed the stack the
next attempt had just built -- the log read `STACK_READY` immediately followed by
`Killed: 9`. `wait_quiet()` now polls until BetaFlight, `gz sim` and `start.sh`
are all actually gone.

### New: probe_alt_step.py --gains

Sweeps several `KP:K` pairs against ONE stack and ONE takeoff -- a stack cycle
costs ~40s and a takeoff ~25s against ~40s for a whole 3-leg gain pair. It also
now reports `|bias|max` and the saturated fraction of each leg, because a clean
step that spent its rise pinned at the +-200 clip is not evidence the gains are
usable.

### Tooling gotchas that cost real time today

- **Never pipe a nanopilot flight script into `tail`/`head`/`grep`.** Two
  separate ways it bites: (1) `gz_camera_worker` is a subprocess that INHERITS
  stdout, so when the parent dies the pipe never sees EOF and the command hangs
  forever with zero output; (2) the worker logs a line per second, so `2>&1 |
  tail -N` fills the window with camera spam and discards the actual results.
  Redirect stdout and stderr to separate FILES.
- **Use `python -u` for anything long-running.** Python block-buffers stdout to
  a file, so a crash loses the entire buffer -- a probe that died on a real
  traceback looked like it produced nothing at all.
- **`gate_rollout.py` takes `--world` and OVERWRITES `GZ_WORLD` from it**
  (default `slalom_gates_80mm`). `eval_flight.py` is the opposite: no `--world`
  flag, reads `GZ_WORLD`. Passing `GZ_WORLD=... gate_rollout.py` without
  `--world` silently flies the heights world against SLALOM gate coordinates.

### If the gains are ever raised, rl_env needs this one-line companion

At Kp=120 every takeoff burned the full 15s timeout with `Timed out arresting
climb after takeoff`. Cause: through the `direct_thrust` ramp the hold setpoint
is re-anchored to the CURRENT height every cycle, so at cutoff it sits at the
liftoff height, BELOW `min_start_alt` (0.8). That was harmless while the hold
was soft -- the drone coasted up through it on momentum -- but a stiff hold wins
and hauls it straight back down, so `z >= min_start_alt` never becomes true.
Fix, at the top of `_arrest_climb` (reverted with the gains, re-apply if needed):

    if getattr(self._env, "altitude_hold_kp", 0.0) > 0.0:
        self._env.set_altitude_setpoint(self.start_alt)

Verified: takeoff then succeeded first attempt with no retries.

### ROOT CAUSE FOUND: the bench probe cannot see the failure, because it
### measures a STATIONARY drone

Instrumented `GateExpert.act` to record `z_ref` and the PRE-clip altitude
command into `gate_rollout.py --trace-output`, then flew 2 episodes at each
gain. `alt_cmd` alone is post-clip and cannot tell "asked for 0.9" from "asked
for 1.8", which is why this needed new instrumentation.

Reconstructing the env's OWN PD term (`-Kp*(z - z_target) - K*vz`) from the
traces settles it:

| | env PD bias, \|max\| | saturated | \|z-z_ref\| mean / max |
| --- | --- | --- | --- |
| Kp=30/K=90 (good, 10/10 x2) | 98-150 pwm | **0.0%** | 0.23-0.36 / 0.63-0.96m |
| Kp=120/K=180 (crashed 8, 5) | **832-941 pwm** | **26.8% / 51.5%** | 0.32-0.54 / 1.69-2.12m |

At Kp=120 the altitude hold spends **a quarter to a half of every episode
pinned at the +-200 clip**. A saturated proportional loop is a relay, not a
regulator, so the higher gains destroy the very regulation they were meant to
buy. The expert's own command clipping (4.9-12.6% vs 0-1.6%) is a SYMPTOM, not
the cause: `alt_raw` only reaches 2.09 because the tracking error reached 2.1m
first.

**Why the sweep missed it, and this is the lesson.** `probe_alt_step.py` holds
the drone STATIONARY and steps a fixed setpoint, so `vz` stays near zero and the
velocity term never loads the clip -- it reported 0% saturation at every gain up
to Kp=240. In flight the velocity term dominates the position term **73-76% of
steps at BOTH gain settings**, so the clip is governed by `vz`, not by position
error. Measured `|vz|` on this course: median 0.40, p90 0.85, p99 1.27, max
1.58 m/s.

**The correct sizing rule is therefore `K * vz_p99 + Kp * err_max < 200`, using
IN-FLIGHT vz** -- not the bench rule `Kp*TRIM_RANGE < 200` that the step
response suggested. The velocity term alone saturates at `|vz| = 200/K`:

| K | vz that saturates | % of flight above it |
| --- | --- | --- |
| **90 (current)** | **2.22 m/s** | **0.0%** |
| 127 | 1.57 | 0.3% |
| 156 | 1.28 | 1.1% |
| 180 | 1.11 | 2.8% |
| 255 | 0.78 | 13.5% |

K=90 is already sized so that this course never saturates it. That is why it
works, and it was not luck -- it leaves the full +-200 for the position term.

**Consequence for any future bandwidth work:** raising `Kp` REQUIRES either
raising the `+-200` clip on `alt_hold_bias` (in `drone_gym.py`, and then the
`hover_throttle +- range` headroom becomes the limit) or accepting a lower `K`,
which costs damping. Bandwidth on this plant is clip-limited, not
stability-limited. **And never size these gains from `probe_alt_step.py`
alone** -- it cannot generate the vz that actually sets the constraint.

### Do this next

1. **Do NOT sweep TRIM_RANGE on the strength of the clipping argument.** It was
   investigated and rejected: the known-GOOD Kp=30 run clips the expert command
   as often as the bad ones (1 crossing in 30), and the Kp=120/Kz=0.0 run never
   clipped at all yet still failed. Per-gate headroom IS tight at the high gates
   (gate_07 has 0.21 of command left after the feed-forward, i.e. only 0.28m of
   correctable error at Kz=0.75), so it is worth knowing for COURSE DESIGN, but
   it is not what limits the current plant.
2. If more altitude bandwidth is ever genuinely needed, the first move is to
   raise the `+-200` clip in `drone_gym.py`, not `Kp` -- and re-measure
   saturation IN FLIGHT with `--trace-output`, not on the bench.
2. If a harder height course is still wanted after that, revisit Kp with the
   knowledge that stability is not the constraint -- and re-sweep Kz AND
   TRIM_RANGE together, since all three interact.
3. Everything from the 2026-09-08 list still stands: re-validate
   `bc_dagger1_scurve` on the flat course if belt and braces is wanted, and
   65mm has still never been re-tested airborne on the tamed pitch/roll PIDs or
   the corrected yaw scaling.
4. `models/bc_dagger1_heights.pth` was NOT re-evaluated this session. Nothing
   affecting it changed (the gains are reverted), so it should still be 10/10.

## Current Handoff (2026-09-09, PLAN: takeoff -> line-follow -> land on target)

**STATE: stopped, nothing running. Nothing implemented yet -- this section is
an agreed plan, not a result.** The previous handoff (2026-09-08b) still stands
in full; nothing in it is superseded.

**Decisions already made, do not re-litigate:**

1. **Vertical authority comes from raising `DRONE_ALTITUDE_TRIM_RANGE`
   (1.0 -> 1.6)**, NOT from a 5th model output, NOT from making `altitude` a
   climb-rate, and NOT from leaving takeoff/landing scripted. See "the four
   options" below for what was rejected and why.
2. **Scripted expert -> BC -> DAgger, not RL.** `rl_train.py`/`rl_env.py` stay
   retired. The record is decisive: the best RL teacher ever produced
   (`ppo_gates_v4`) reached 10/10 and the scripted `GateExpert` MATCHED it with
   zero training time and no checkpoint to pick, while every attempt to distil
   a student from an RL teacher failed. "Fly a line, then descend over a pad"
   is a pure function of privileged state, so it is an even better fit for a
   scripted expert than the gate courses were. RL would only earn its place if
   the landing FLARE were to be discovered rather than specified (terminal
   reward on touchdown velocity); that was considered and deliberately not
   chosen.

### Most of this task already exists

`worlds/landing_target_80mm.sdf` is already the described course. Measured from
the SDF:

| model | pose | size |
| --- | --- | --- |
| `start_platform` | (0, 0, 0.01) | 2 x 2 |
| `guidance_line` | (10, 0, 0.01) | 20 x 0.2 (runs x=0 -> x=20) |
| `landing_pad` | (20, 0, 0.01) | 0.5 x 0.5 box, 1.5m radius visual |

`eval_flight.py` also already carries the scaffolding from the 2026-08-07 BC
era: `parse_world` returns `(gates, pad_xy)`, `detect_task` returns `"line"`
when a pad exists and there are no gates, `episode_metrics` computes
`along_track_max_m` / `cross_track_mean_m` / `cross_track_max_m` /
`final_dist_to_pad_m`, and `--pad-radius` (default 1.0) exists.

**All of it predates the velocity-setpoint interface (2026-09-06b), so treat it
as unvalidated scaffolding, not as working code.**

### Blocker 1: the action interface cannot express a takeoff or a landing

Under `DRONE_VELOCITY_CONTROL`, `altitude` is a height TRIM about a setpoint
frozen at `start_alt`, so `z_target = setpoint + altitude * TRIM_RANGE` and the
reachable band is **0.5m .. 2.5m** at `start_alt=1.5, TRIM_RANGE=1.0`. The
ground is at 0. **At full negative stick the policy can still only ask for
0.5m; touchdown is not in its action space.** Takeoff and landing are today
scripted `direct_thrust` ramps that bypass the hold entirely, and
`direct_thrust` is not one of the policy's four outputs.

### Blocker 2: the scorer counts a landing as a crash

`eval_flight.py` ends an episode `"crashed"` at `pos[2] < crash_z` (default
0.12), and the `"line"` task already reports `"completed"` on HORIZONTAL
proximity to the pad alone (`fly_episode`, the `near_pad` branch) without ever
descending. So today a perfect landing scores as a crash and a fly-over scores
as a success. Nothing downstream can be judged until this is fixed.

### Blocker 3: there is no expert for this task

`GateExpert` is gate-specific -- pure pursuit along a gate centreline with
`gate_idx` tracking. A `LineExpert` is new, but small.

### The four options, and why C

| | approach | verdict |
| --- | --- | --- |
| A | 5th output = explicit thrust | REJECTED: breaks the 4-output model, invalidates every tub and checkpoint |
| B | make `altitude` a climb-RATE, integrating into the setpoint | REJECTED: a real interface change, same risk class as 2026-09-06b |
| **C** | **raise `TRIM_RANGE` until the band reaches ground** | **CHOSEN** |
| D | keep takeoff/landing scripted, learn only the line | REJECTED: cheapest but tests the least |

**`TRIM_RANGE = 1.6` puts the reachable floor at -0.1m** (ground reachable with
margin) and is affordable against the `+-200` clip on `alt_hold_bias`, using
the sizing rule established on 2026-09-08b (`K*vz + Kp*err < 200`, with
IN-FLIGHT vz, never bench vz):

    position term, worst case:  Kp * TRIM_RANGE = 30 * 1.6 = 48 pwm
    descent at 0.4 m/s:         K  * vz         = 90 * 0.4 = 36 pwm
    total                                        ~84 of 200

That is comfortable at the CURRENT gains. It would NOT be comfortable at the
Kp=120/K=180 gains rejected on 2026-09-08b (192 + 72 = 264, saturated) -- which
is another reason those gains stay reverted.

**Cost to own:** raising `TRIM_RANGE` rescales what the altitude axis MEANS, so
every existing tub and checkpoint goes stale for training on this axis. That is
acceptable because this course needs fresh data anyway, but it is exactly the
"do not mix tubs across an interface change" trap -- do not aggregate
pre-1.6 tubs with post-1.6 ones.

### Staged plan

**Stage 0 -- probes. This is the "learn how to change altitude well" payload.**
Nothing trains until these are measured:
- `probe_alt_step.py` near the ground (0.3-0.8m): does the hold behave the same
  low down as at 1.5m?
- **A new descent-rate probe.** Command increasing descent rates and find where
  the PD term saturates and where the drone stops tracking. This is the direct
  extension of the 2026-09-08b saturation finding and it is the number that
  sizes the landing. The descent is the one phase that deliberately runs `vz`
  up, and `vz` is what eats the clip budget.
- Confirm there is no ground effect in the sim (the plugin has no vertical
  drag, so probably none -- confirm rather than assume).

**Stage 1 -- make the task scorable.** Add a `"land"` task to `eval_flight.py`:
success = inside the pad radius AND `z < touchdown_z` AND `|vz| < settle_vz`,
scored distinctly from `"crashed"`. Report touchdown velocity and pad offset in
the metrics. Nothing downstream is meaningful until this exists.

**Stage 2 -- write `LineExpert` and prove it flies.** Pure pursuit along the
line to the pad, then a proportional descent once inside the pad radius. Same
shape as `GateExpert`, reachable through `act_from_telemetry`. **Gate: >=90%
clean landings over 10 episodes before collecting anything.** The expert is the
ceiling; a mediocre expert cannot be fixed downstream.

**Stage 3 -- set `TRIM_RANGE = 1.6` and RE-VALIDATE THE GATE COURSES.** This
changes the plant for every course, not just the new one. Re-run
`gate_rollout.py --expert --world=s_curve_heights_80mm` and confirm it is still
10/10 before going further. Non-negotiable checkpoint: if the gate courses
regress, stop and reconsider, because `bc_dagger1_heights` depends on that
plant.

**Stage 4 -- collect, distil, DAgger.** `rl_collect.py --expert` (~40
episodes), `torch_train.py`, then `eval_flight`. Expect the established
pattern: the expert-only student flies on-manifold and fails off it, and one
DAgger round fixes it. **Judge by `eval_flight` only, never by val loss** --
DAgger val loss measures the PREVIOUS policy's mistakes (2026-09-07b).

### STAGE 0 RESULT (2026-09-09): probes measured, and the plan's two stated
### risks are both FALSE. TRIM_RANGE = 1.6 is confirmed by measurement.

**STATE: stopped, nothing running.** No config, world, model or flight
behaviour changed. What landed: `scripts/probe_descent.py` (new), a `--settle-s`
flag on `probe_alt_step.py`, and the measurements below. All taken on
`landing_target_80mm` at the CURRENT gains (Kp=30, K=90, TRIM_RANGE=1.0),
sweeping trim range IN FLIGHT so the plant was never changed for any course.

#### 1. The hold behaves identically near the ground

`probe_alt_step.py --amplitude=0.25 --hold-s=20` at start-alt 1.5 and 0.6:

| | at 1.5m | at 0.6m |
| --- | --- | --- |
| t_cross, +0.15m step | 3.06 / 3.08s | 3.03 / 3.06s |
| steady-state error | **+0.10m** | **+0.10m** |
| \|bias\|max (0.6m step) | 18 pwm | 18 pwm |
| saturated | 0% | 0% |

Not merely similar -- indistinguishable. Nothing about the loop needs
re-tuning for the low-altitude phase of a landing.

**Probe gotcha found:** `analyse()` reports a bogus overshoot for SMALL steps,
because the hold's +0.10m residual offset IS 65% of a 0.15m step, so `final ==
peak` and the implied zeta=0.14 / w_n=0.57 are artifacts. Read the large-step
leg. The 0.60m leg reports -16% overshoot and +0.10m error, which is the truth.

#### 2. No ground effect, to the resolution of the measurement

Steady-state thrust demand at six setpoints from 1.50m down to 0.25m:

| setpoint | 1.50 | 1.00 | 0.70 | 0.50 | 0.35 | 0.25 |
| --- | --- | --- | --- | --- | --- | --- |
| z settled | 1.598 | 1.098 | 0.798 | 0.598 | 0.448 | 0.348 |
| offset | +0.098 | +0.098 | +0.098 | +0.098 | +0.098 | +0.098 |
| throttle pwm | 1477.7 | 1477.7 | 1477.7 | 1477.7 | 1477.7 | 1477.7 |

Offset spread **0.001m**, throttle spread **0.1 pwm**. As expected from a
plugin with no vertical drag and no ground-effect model, but now confirmed
rather than assumed. Both quantities were measured because either alone could
be read as drift.

That +0.098m offset is the plant's residual thrust bias divided by Kp: ~3 PWM
of excess lift at Kp=30. It is a constant of the plant, not an altitude effect.

#### 3. The altitude axis has NO descent-rate command, and that is the finding

The axis commands a target HEIGHT about a frozen setpoint, so the fastest
descent available is `alt = -1`, aiming at `setpoint - TRIM_RANGE`. Every
ramped `z_ref` is strictly slower. **Descent speed is therefore a pure function
of TRIM_RANGE**, and it is bounded by how far BELOW GROUND the target can be
placed.

Sustaining a descent also costs trim range, because the sim has no vertical
drag: constant-velocity descent needs zero net force, so the drone must fly a
steady LAG above its own target. Measured against the prediction, with the
expert's own height law driving it (Kz=0.75):

    e = ((K/Kp) * r + B/Kp) / (1 + Kz)      B/Kp = the 0.098m residual above

| commanded rate | achieved | predicted lag | **measured lag** |
| --- | --- | --- | --- |
| 0.10 m/s | 0.09 | 0.21m | **0.21m** |
| 0.20 m/s | 0.17 | 0.35m | **0.36m** |
| 0.30 m/s | 0.25 | 0.48m | **0.48m** |

Agreement to 0.01m. So the command budget is
`|alt| * TRIM = D + (K/Kp)*Kz*r/(1+Kz)` for a descent of depth D -- **depth and
rate are spent out of the same trim range**, which is the whole reason
`TRIM_RANGE` had to rise for this task.

#### 4. Landing profile: TRIM_RANGE = 1.6 lands in 8.6s at 0.02 m/s

`--mode=landing`, full-down command from a 1.5m cruise, trim swept in flight:

| trim | target | z<0.50 | z<0.20 | peak vz | touchdown | touchdown vz | \|bias\|max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **1.6** | **-0.10m** | 3.1s | 5.4s | -0.45 | **8.6s** | **-0.02 m/s** | **49** |
| 2.0 | -0.50m | 2.3s | 3.3s | -0.56 | 4.0s | -0.18 m/s | 62 |
| 2.5 | -1.00m | 1.7s | 2.4s | -0.71 | 2.7s | -0.37 m/s | 78 |
| 3.0 | -1.50m | 1.4s | 1.9s | -0.86 | 2.1s | -0.56 m/s | 90 |

Monotone and well-behaved: a deeper target lands faster and harder, because
the drone decelerates as it approaches the target and a target just below
ground is reached asymptotically. **1.6 gives a feather landing (0.02 m/s) in
8.6s**, which is affordable inside a 60s episode against a ~20m cruise. Nothing
here argues for going past it; if a future task wants a brisker touchdown, 2.0
buys 4.0s at a still-gentle 0.18 m/s.

#### Both risks the plan flagged are measurably false

- **"The descent eats the clip budget."** It does not, and not marginally: the
  +-200 clip on `alt_hold_bias` peaked at **49 pwm at trim 1.6**, and at 90 pwm
  even at trim 3.0. Saturation was **0% of every leg**, including the ramps.
  The 2026-09-08b sizing rule `K*vz + Kp*err < 200` is satisfied with room to
  spare because a descent this loop can command is slow: at K=90 the velocity
  term needs 2.22 m/s to saturate and the fastest measured descent was 0.86.
  **Do not spend time defending the clip on this task.** The scarce resource is
  trim RANGE, not bias headroom -- which is the opposite of the gate-course
  finding, and only because the gate courses fly fast horizontally.
- **"The hold might behave differently near the ground."** It does not (1).

#### The one number Stage 1 must respect

`eval_flight.py` calls `pos[2] < crash_z` (default **0.12**) a crash. At trim
1.6 the drone passes z<0.20 at 5.4s and touches at 8.6s. **A textbook landing
trips the crash test**, exactly as Blocker 2 predicted -- so the `"land"` task
must be added before any landing result means anything. Touchdown detection at
z < 0.06 worked cleanly in the probe and is a usable threshold; measured
touchdown vz gives `settle_vz` a sane scale (0.02 m/s at trim 1.6, so a
threshold of ~0.15 m/s separates a landing from a fall).

#### Tooling notes

- **`direct_thrust=0.0` is NOT "no command" -- it is a climb with no damper.**
  It pins the throttle at exactly hover AND suppresses `alt_hold_bias`
  entirely, and with no vertical drag the drone then coasts upward forever on
  its climb momentum. The first run of `probe_descent.py` reached **11m** in
  its "arrest the climb" phase for this reason. Arrest on the HOLD
  (`run_threaded(0,0,0,0)`), the way `rl_env._arrest_climb` does.
  **`scripts/probe_alt_step.py` still has this bug in its takeoff** -- harmless
  there only because the subsequent settle drags the drone back down, at the
  cost of ~30s per run.
- A descent leg must ramp for several time constants (1/w_n ~ 2.1s) or it
  measures a transient. `probe_descent.py`'s first ramp table reported "0%
  achieved" at 1.0 m/s purely because a 0.6m ramp at 1.0 m/s is over in 0.6
  seconds. The leg now refuses to score a ramp shorter than 4s.
- `probe_alt_step.py` gained `--settle-s`: reaching a LOW start-alt takes far
  longer than the default 12s, because the takeoff coasts well above it and the
  descent is unforced.

#### Stage 0 is complete. Next is Stage 1

Nothing measured here changes the plan. Proceed to Stage 1 (add the `"land"`
task to `eval_flight.py`), then Stage 2 (`LineExpert`), then Stage 3
(`TRIM_RANGE = 1.6` in config + **re-validate the gate courses**, which is
still non-negotiable: the trim range rescales what the altitude axis MEANS on
every course, so `bc_dagger1_heights` and the expert's height law both depend
on it).

### STAGE 1 RESULT (2026-09-09): eval_flight can now score a landing

**STATE: stopped, nothing running.** Only `nanopilot/eval_flight.py` changed.
No config, world or model changed. `bc_dagger1_heights` re-validated on the
gate course as a regression of the shared code (below), and it is unchanged.

#### The `land` task

`detect_task` now returns **`"land"`** for a world with a pad and no gates, so
`landing_target_80mm` scores as a landing rather than a fly-over. `"line"` is
kept and reachable with `--task=line` -- it completes on HORIZONTAL proximity
while still airborne, which is the right test for a fly-the-line ablation and
the wrong one for a task whose last five metres are vertical.

Ground contact is scored by `--touchdown-z` (0.06) instead of `--crash-z`,
which the land task does not use at all. That is Blocker 2: crash_z is 0.12
and a textbook landing passes 0.12 on the way down (Stage 0 measured z<0.20 at
5.4s, touchdown at 8.6s), so the old scorer called every successful landing a
crash.

`classify_touchdown()` is a PURE FUNCTION -- deliberately, so the scoring rules
can be tested without a stack -- and splits contact four ways:

| result | test | why it is its own outcome |
| --- | --- | --- |
| `completed` | gentle, stopped, on the pad | the only success |
| `crashed` | `\|vz\| >= --settle-vz` (0.15 m/s) | arrived as a fall |
| `skidded` | `speed_xy >= --settle-speed` (0.5 m/s) | belly-flop at cruise |
| `landed_off_pad` | offset > `--pad-radius` | a real landing, wrong place |

`success_rate` therefore keeps meaning what it means for the gates task, and
each failure gets a name that points at a different fix.

**`skidded` is an addition to the plan's stated criterion and it is
load-bearing.** The plan specified "inside the pad radius AND z < touchdown_z
AND \|vz\| < settle_vz". A drone that flies into the ground at 1.5 m/s
horizontally has a tiny VERTICAL speed at contact and would have scored as a
successful landing under that rule alone -- a realistic BC failure mode. The
0.5 m/s default is deliberately generous (a quarter of the 2.0 m/s cruise): its
job is to refuse to score a belly-flop as a success, not to grade landing style.

**Impact speed is measured over the 0.5s BEFORE contact** (`TOUCHDOWN_VZ_WINDOW_S`),
not at the contact sample. The pose-derived velocity is low-passed with
tau=0.1 and the ground stops the drone in one frame, so the sample at contact
is already decelerating and classifying on it alone scores every impact as
gentle. Verified against a synthetic hard landing whose contact sample reads
-0.01 m/s: the window catches the -1.10 and calls it `crashed`.

`--settle-vz = 0.15` sits between the two measured touchdown speeds from Stage
0 (0.02 m/s at TRIM_RANGE=1.6, 0.18 at 2.0). **If TRIM_RANGE is ever raised
past 1.6, raise this with it** or every landing scores as a crash.

New metrics per episode and in the aggregate: `touchdown_s`,
`touchdown_vz_ms`, `touchdown_speed_ms`, `touchdown_pad_offset_m`, plus
`touchdown_rate`, `skidded_rate`, `landed_off_pad_rate` and
`max_touchdown_vz_abs_ms`. An episode that never reaches the ground prints
**NEVER TOUCHED DOWN** with its final distance from the pad, so a fly-over can
never be confused with a landing at a glance.

#### Verified

- **Offline, no stack:** 8 classification cases (gentle-on-pad, pad edge, just
  outside, belly-flop, fall, fall-off-pad, the trim-2.0 touchdown speed, and
  the masked-impact case above) all classify as intended; `detect_task` returns
  `land` for `landing_target_80mm` and `gates` for all three gate worlds.
- **Live, real telemetry:** with `--touchdown-z` raised above cruise to force
  the branch, the touchdown path fires, computes vz/speed/pad offset from the
  real track, and aggregates. With it at 1.55 the drone held 1.641m and the run
  correctly reported NEVER TOUCHED DOWN.
- **Gates regression** (the shared track tuple gained a field, so this was not
  optional): `bc_dagger1_heights` on `s_curve_heights_80mm`, 3 runs,
  **3/3 completed, 10/10 gates, height err 0.254m mean / 0.467 max** -- matching
  the 2026-09-08 record (0.254 / 0.465) to the third decimal. `--trace` arrays
  are intact and aligned at 560 samples. This also closes the 2026-09-08 open
  item "`bc_dagger1_heights` was NOT re-evaluated".

#### Stage 1 is complete. Next is Stage 2 (`LineExpert`)

Gate before collecting anything: **>=90% clean landings over 10 episodes**, i.e.
`success_rate >= 0.9` on the land task. The expert is the ceiling.

### STAGE 2 RESULT (2026-09-09): LineExpert flies it 10/10, and the glide
### slope had to be sized by the CAMERA, not by the plant

**STATE: stopped, nothing running.** New `nanopilot/line_expert.py`;
`eval_flight.py` gains `--expert` and `--trim-range`. **No config changed** --
the trim range is overridden IN FLIGHT, so the plant is untouched for every
other course. That is still Stage 3's job.

#### Gate: PASSED, 10/10 clean landings (required >=90%)

`eval_flight.py --expert --trim-range=1.6 --runs=10 --duration=60 --rate-hz=20`:

    success_rate 1.0   crash_rate 0.0   skidded 0.0   landed_off_pad 0.0
    mean 18.60s   touchdown vz -0.032 m/s   0.045m from pad centre
    cross-track mean 0.014m

Ten for ten, and repeatable to the third decimal (pad offset 0.038-0.054m,
touchdown vz -0.029..-0.034). For scale, the pad is a 0.5m box inside a 1.5m
visual disc and `--pad-radius` is 1.0m, so the expert touches down 20x tighter
than the scoring window.

#### The law

Three parts, all pure functions of privileged state -- no phase variable, no
timer, no latch -- so `act()` is reproducible from state alone and DAgger can
label a drone that has drifted back off the pad with "climb and go round"
rather than with whatever phase it was last in.

  * **Horizontal:** pure pursuit along the line, carrot clamped at the pad.
    Same as `GateExpert`'s centerline pursuit, and it is what keeps the
    guidance line under the camera.
  * **Speed:** a proportional POSITION loop, `v = min(cruise, 0.6 * d_pad)`, so
    the drone eases in and stops over the pad rather than arriving at cruise
    and needing to be caught.
  * **Height:** a reference ramping from cruise to a target BELOW GROUND
    (-0.1m), through the same feed-forward-plus-proportional law the gate
    expert uses, at the same validated Kz = 0.75.

Two details are load-bearing and were arrived at by failing at them on paper
first:

  * **The height ramp SATURATES at `land_radius` (0.5m); it does not taper to
    the target.** The hold settles ~0.10m above whatever it is asked for (the
    plant's residual bias over Kp, Stage 0), so a reference that only reaches
    the ground target as d -> 0 leaves a drone hovering 0.3m off centre parked
    at z ~ 0.2m forever, scoring NEVER TOUCHED DOWN.
  * **`ground_target` is below ground on purpose.** Descent speed is set purely
    by how far below ground the target sits, because the axis commands a height
    and not a rate (Stage 0). Aiming at exactly 0.0 approaches asymptotically
    and never arrives.

#### The glide slope is sized by the CAMERA, and this is the reusable lesson

The first tuning (`descend_radius = 2.0`) flew level to the pad and then
descended vertically on the spot. It landed 10/10 at 0.075m. **It was still the
wrong expert**, because it generates bad DATA: the camera is forward-facing,
140 deg horizontal, which at 4:3 is **+-64 deg vertically**. A drone 0.3m from
the pad at 1.5m sees it at **78.7 deg depression -- out of frame** -- and the
pad only reappears below ~0.6m. That is exactly the 2026-09-06b failure ("the
target is outside the camera at the moment of the turn") pointed at the pad
instead of a gate, and it would have been discovered in Stage 4 as an
unexplained student failure.

Descending ALONG the approach fixes it. Measured, one episode each at trim 1.6:

| descend_radius | duration | pad offset | pad out of frame | alt label on the ramp |
| --- | --- | --- | --- | --- |
| 2m (level, then vertical) | 20.9s | 0.075m | **16.7%** | 6.1% |
| **8m (glide) -- DEFAULT** | **18.5s** | **0.048m** | **3.3%** | **22.7%** |
| 12m (glide) | 17.4s | 0.165m | 2.3% | 37.8% |

8m wins on every axis at once, so it is the default and the 10-episode gate
above was re-run at it. 12m descends so early that the drone arrives low and
still moving, which costs touchdown accuracy. The residual 3.3% out-of-frame is
the last few centimetres directly over the pad, by which point the drone is
committed.

**The altitude label shape is the second reason to prefer the glide.** At 2m
the label is nearly binary -- 53.7% at ~0 and 40.2% pinned at exactly -1.0,
with only 6.1% on the continuous ramp between -- so the student would have to
learn a switch triggered by a visual cue that is out of frame when it fires. At
8m, 22.7% of the episode is on the ramp. **This also revises the plan's
expectation that the landing would be a brief event needing loss re-weighting:
the descent is 8.8s of an 18.6s episode. `TRAIN_LOSS_WEIGHTS` should stay
uniform until flight data says otherwise.**

#### eval_flight gains --expert and --trim-range

`--expert` flies `LineExpert` and is scored **by exactly the same rules as a
checkpoint**. That is the point: the expert is the ceiling for anything
distilled from it, and a ceiling measured with a different scorer is not a
ceiling. It refuses to run on a non-`land` world.

`--trim-range` overrides `DRONE_ALTITUDE_TRIM_RANGE` in flight and prints a
loud banner. It exists so this stage could be validated at 1.6 without changing
the plant for the gate courses, and it warns explicitly when the range cannot
reach `ground_target` from the cruise height.

At the CURRENT config (1.0) the expert cannot land at all, by construction: the
command clips at -1 and the drone hovers at `start_alt - 1.0` = 0.5m, which the
land scorer reports as NEVER TOUCHED DOWN. That failure is self-diagnosing
rather than silent, which is deliberate.

#### Stage 2 is complete. Next is Stage 3

Set `DRONE_ALTITUDE_TRIM_RANGE = 1.6` in `drone_config_80mm.py` and **re-validate
the gate courses** -- non-negotiable, and now with a concrete expectation to
check against: the trim range rescales what the altitude axis MEANS, so
`GateExpert`'s commands on `s_curve_heights_80mm` all shrink by 1/1.6 while
meaning the same height, and `bc_dagger1_heights` (trained at 1.0) will be
commanding 1.6x the height offset it intends. **Expect the existing student to
regress and require retraining; that cost was accepted when option C was
chosen.** The gate to check is the EXPERT at 10/10, not the student.

### Two things expected to go wrong

- **The descent eats the clip budget.** It is the only phase that deliberately
  builds `vz`. If landings turn unreliable, check `alt_hold_bias` saturation in
  a `--trace-output` run BEFORE touching any gain -- that is the mistake made
  on 2026-09-08.
- **The label distribution is badly imbalanced**: a long cruise and a brief
  descent. The landing may need over-weighting or the student will smooth it
  away. Note `TRAIN_LOSS_WEIGHTS` is uniform `[1,1,1,1]` and the standing advice
  is not to re-tune it without flight data -- this task may be the first real
  reason to.
