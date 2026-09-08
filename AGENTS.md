# AGENTS.md

Concise runbook for coding agents working in this repo. Use `README.md` for
project overview, setup details, and public-facing docs. Keep this file focused
on agent workflow, sharp edges, and the current handoff.

## Fresh Session Checklist

1. Read the newest `## Current Handoff` section at the bottom.
1b. **Check whether an RL run is training right now** before touching the sim
   or starting anything that needs BetaFlight/Gazebo — a long run may be in
   progress and any stack command will fight it for UDP 9004:
   ```bash
   pgrep -fl rl_train.sh
   tail -3 logs/rl_train_ppo_gates_v4.log
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

| Key | Meaning |
| --- | --- |
| `steering [-1, 1]` | yaw rate |
| `throttle [-1, 1]` | forward pitch / tilt |
| `roll [-1, 1]` | lateral roll / bank |
| `altitude [-1, 1]` | motor throttle around hover PWM |

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
./scripts/eval_flight.sh --model=models/bc_v4.pth --world=landing_target_80mm --runs=3

# RL teacher: bounded PPO smoke run (headless stack + train + teardown)
./scripts/rl_train.sh --total-steps=1200 --n-steps=256 --batch-size=64 \
  --checkpoint-every=512 --out=models/rl/smoke

# RL teacher: long run. Self-supervising — rebuilds the stack and resumes from
# checkpoint if BetaFlight dies. The altitude flags are REQUIRED: without them
# the policy cannot hold gate height and will never pass a gate (see the
# newest handoff). Launch via detach_launch.py so a killed shell doesn't take
# the trainer down with it.
./scripts/rl_train.sh --total-steps=400000 --out=models/rl/ppo_gates_v4 \
  --control-altitude --altitude-scale=0.09

# Altitude hold check: healthy = roughly constant z for 30s at neutral
uv run --env-file .env python scripts/probe_hover.py

# Why is the policy missing gates? Per-crossing lateral/height/speed vs the
# pass window (needs a running stack)
uv run --env-file .env python scripts/gate_rollout.py \
  --policy=models/rl/ppo_gates_v4 --model=model.zip \
  --vecnorm=models/rl/ppo_gates_v4/vecnorm.pkl --episodes=6

# Distill a trained RL teacher into vision tubs for torch_train.py
# NOTE: --min-gates=3 currently yields ZERO episodes; use 2.
GZ_HEADLESS=1 GZ_WORLD=slalom_gates_80mm ./scripts/start.sh --no-manage --airframe=80mm
uv run --env-file .env python nanopilot/rl_collect.py \
  --policy=models/rl/ppo_gates_v4 --episodes=40 --min-gates=2

# Scripted no-human collection + bounded training
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
  nanopilot/gate_expert.py \
  scripts/gate_rollout.py scripts/ff_sweep.py scripts/detach_launch.py \
  scripts/probe_velocity.py

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
| `nanopilot/eval_flight.py` | closed-loop autopilot eval: survival/progress/gate metrics, JSON |
| `nanopilot/gate_expert.py` | scripted pure-pursuit gate expert over velocity setpoints |
| `nanopilot/rl_env.py` | `GateChaseEnv`: state-based Gymnasium env over a live DroneGymEnv |
| `nanopilot/rl_train.py` | PPO trainer for the state-based RL gate teacher |
| `nanopilot/rl_collect.py` | flies a trained RL teacher, records tubs for BC distillation |
| `nanopilot/gz_reset_worker.py` | persistent subprocess issuing gz set_pose teleports |
| `scripts/probe_hover.py` | takeoff + 30s neutral hold; altitude-hold check |
| `scripts/probe_velocity.py` | velocity-cascade step response: signs, gains, cross-axis leakage |
| `scripts/gate_rollout.py` | flies a trained RL policy; per-gate-crossing lateral/height/speed vs the pass window |
| `scripts/ff_sweep.py` | sweeps `DRONE_PITCH_THROTTLE_FEEDFORWARD` airborne, climb rate per pitch |
| `scripts/detach_launch.py` | `os.setsid()` launcher; long runs survive process-group kills |
| `scripts/start.sh` | stack launcher |
| `scripts/stop_all.sh` | stack cleanup |
| `scripts/test_thrust.sh` | launches stack, runs `test_thrust.py`, tears down |
| `scripts/eval_flight.sh` | launches stack, runs `eval_flight.py`, tears down |
| `scripts/rl_train.sh` | supervisor: launches stack, runs `rl_train.py`, rebuilds+resumes on failure |
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

## Current Handoff (2026-08-07)

Landing-target behavior-cloning iteration (prior handoff: quadratic drag work,
see git history).

Data and models:

- `data/tub_12_26-08-06`: 1715 frames, landing_target_80mm, recorded at the
  old 45-degree max pitch. Gentle demo (mean pitch cmd 0.148, 59% zero).
- `data/tub_13_26-08-07`: 2870 frames, landing_target_80mm, recorded at the
  new 20-degree max pitch, more roll corrections (15% of frames).
- `models/tub_12_26-08-06.pth`: trained fresh on tub_12, best val 0.0112.
- `models/tub_13_26-08-07_ft.pth`: fine-tuned from tub_12 model on tub_13
  (`--init-weights`, lr 3e-4), best val 0.0088.

eval_flight baselines on landing_target_80mm (pad at 20m):

- tub_12 model, 30s x3: 0 crashes, along-track mean 3.75m, cross-track 0.25m.
- tub_12 model, 90s x3: 0 crashes, along-track 11.0m (pace is linear,
  ~0.12 m/s — clones the slow demo), cross-track grows to 0.80m mean /
  1.6m max, altitude sags 2.40 -> 2.16m over the episode.
- tub_13 fine-tune (`tub_13_26-08-07_ft.pth`) 90s x3: all crashed ~17.5s.
  Commands roll ~0.8 from the first pilot frame at the eval start pose and
  flies sideways ~20m until attitude-crash. Offline evaluate.py on tub_13 is
  fine (pitch corr 0.95, roll MAE 0.07) — closed-loop distribution shift +
  forgetting of tub_12 behavior, not broken weights.
- combined fresh train (`tub_12_13_combined.pth`, both tubs) 90s x3: all
  crashed, 2 of 3 sinking to the floor within ~5s while commanding near-zero
  altitude. Val was noisy, best epoch was epoch 1 — likely label conflict:
  tub_12 was recorded at the 45-degree pitch mapping, tub_13 at 20 degrees,
  so similar frames carry ~2.25x different command labels for the same
  physical maneuver. Don't mix tubs across the mapping change without
  rescaling labels.
- Best flying model remains `models/tub_12_26-08-06.pth`.

Config/infra changes this session:

- `drone_config_80mm.py`: `DRONE_MAX_PITCH_ANGLE`/`DRONE_MAX_ROLL_ANGLE`
  45 -> 20 (keyboard pilots can't hold partial pitch; full deflection is now a
  comfortable cruise). `DRONE_PITCH_THROTTLE_FEEDFORWARD` 8 -> 7. Models
  trained on 45-degree-era tubs fly slower under this mapping — don't compare
  eval numbers across the change.
- `nanopilot/eval_flight.py request_reset`: re-issues the gz set_pose reset
  every 2s until the pose sticks; a single dropped 1s service call used to
  kill the whole eval.
- `scripts/start.sh start_betaflight`: retries raised 2 -> 5 with 2s pauses.
  BetaFlight SITL intermittently aborts at launch (heap corruption in
  openSerialPort/stream_initAddress, macOS ARM64). Random per launch; not
  eeprom, not config. Crash reports in `~/Library/Logs/DiagnosticReports/`.

Follow-ups:

1. Collect more 20-degree-era landing_target tubs (varied start views,
   deliberate centerline-recovery maneuvers, some altitude activity) and train
   fresh on those alone; don't mix with 45-degree-era tub_12 unless labels are
   rescaled by 45/20.
2. Consider evaluating a fresh (non-fine-tuned) train on tub_13 alone before
   collecting more.
3. Lateral-coast maneuver still climbs in Z; consider roll-tilt throttle
   compensation like the pitch feed-forward.
4. Verify `DRONE_PITCH_THROTTLE_FEEDFORWARD=7` doesn't over/under-compensate
   altitude at the 20-degree mapping (damper-sim mode of test_thrust.sh).

## Current Handoff (2026-08-22)

RL teacher-student pipeline built and smoke-tested (prior handoff:
landing-target BC iteration, above).

Plan: state-based PPO "teacher" learns gate flying from privileged Gazebo
ground truth (no camera — feasible at real-time sim speed), then
`rl_collect.py` flies the teacher to generate tubs and the existing BC
pipeline distills it into a deployable vision model. Direct pixel RL was
rejected: sim is real-time-locked (~70k steps/hour at 20Hz), pixel PPO needs
millions of steps.

New pieces:

- `nanopilot/rl_env.py`: `GateChaseEnv` — 20-dim state obs (gate-relative
  position/heading, velocity, attitude, gyro, prev action, next gate), gate
  progress/pass/miss/crash rewards, jittered spawns, calm-verified resets.
- `nanopilot/rl_train.py`: PPO/MlpPolicy + VecNormalize + checkpoints; out
  dir gets `model.zip` + `vecnorm.pkl`. `--resume=<out dir>` continues
  (NOTE: on resume `--total-steps` means ADDITIONAL steps, SB3 semantics).
- `nanopilot/rl_collect.py`: flies a trained policy, buffers rows per
  episode, drops crashed/flyaway episodes, writes standard tubs.
- `nanopilot/gz_reset_worker.py` + `scripts/rl_train.sh`.

Verified end-to-end (bounded smoke, 1280 steps + resume + collect): resets,
episodes, PPO updates, checkpoint/vecnorm saving, resume continuation, and
teacher tub writing all work. `models/rl/smoke/` is that junk-weight smoke
artifact — safe to delete, or reuse for pipeline tests.

Sharp edges found this session (all fixed, don't regress them):

- gz-transport service requests from a torch-loaded process reliably time
  out, and enough timed-out in-process requests wedge Gazebo's UserCommands
  service for ALL callers until the stack restarts. Pose resets therefore go
  through a `gz_reset_worker.py` subprocess (same pattern/reason as
  `gz_camera_worker.py`). Never call gz services in-process from
  training/eval code.
- gz `set_pose` does NOT zero velocity: after a fast flyaway the drone keeps
  its momentum through the teleport. `GateChaseEnv._teleport_to_spawn`
  re-pins the pose and requires a displacement-based calm window before
  starting the episode.
- `SimTelemetry`/`PoseTracker` velocities are finite differences and spike
  absurdly on teleports; both now clamp >20 m/s samples to zero. Before the
  clamp, the altitude damper would slam throttle on every reset and bounce
  the drone for many seconds.
- `init_pose_subscriber()` (altitude damper) used to read only the GZ_WORLD
  env var; when unset the damper silently subscribed to a dead topic and the
  drone climbed at 50+ m/s at hover PWM. It now takes the world from
  `DroneGymEnv.gz_world`; `rl_train.py`/`rl_collect.py` also export GZ_WORLD.
- At neutral controls the drone does not sit on the ground — hover bias +
  damper keep it wandering at 2-3m. RL episodes start from that hover;
  eval-style "takeoff to 1.5m" passes trivially.
- `DroneGymEnv.spawn_pose` (x, y, z, yaw) is now settable; `_do_reset`
  teleports there instead of hardcoded origin. The broken never-run
  `step()`/`reset()` methods were removed from `DroneGymEnv`; RL goes through
  `GateChaseEnv` wrapping `run_threaded`.

Next steps: superseded — see the 2026-08-23 handoff below. RL training is
blocked on an airframe altitude problem; do not start from this section.

## Current Handoff (2026-08-23)

**STATE: everything is stopped. No training run is in progress. RL training
is BLOCKED — see the BLOCKER section below and fix that first.**

The teacher/student pipeline itself works end to end (resets, episodes, PPO
updates, checkpointing, resume, tub writing all verified). What does not work
is the flight: on `slalom_gates_80mm` the drone climbs out of the world at
neutral stick, so every episode dies in ~2.5s and PPO has no gradient.

Two full runs were attempted:

- `models/rl/ppo_gates_v1` + `logs/rl_train_ppo_gates_v1_4axis.log`: 4-axis
  policy, reached 64.5k steps over 3h15m. `ep_rew_mean` −24.1 -> −23.7 and
  `ep_len_mean` 37 -> 38.9, i.e. FLAT. 1660/1660 episodes flyaway, 0/3 gates
  ever. Checkpoints to 60k are kept for reference only — the action space has
  since changed to 3, so they will NOT load in the current env.
- `models/rl/ppo_gates_v2`: 3-axis policy, only minutes of runtime before the
  blocker was identified. Junk, safe to delete.
- `models/rl/reset_check`, `models/rl/smoke`: bounded smoke artifacts, junk.

An earlier 400k attempt died at 30.7k steps from reset-worker starvation
(fixed, see below); its log is `logs/rl_train_ppo_gates_v1_failed_attempt1.log`.

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

1. Fix the altitude blocker (below). Start with
   `./scripts/test_thrust.sh --airframe=80mm`.
2. Re-verify with `uv run --env-file .env python scripts/probe_hover.py`
   against a running stack: a healthy airframe holds roughly constant z for
   30s at neutral. Today it climbs 0.34m -> 69.8m.
3. Only then restart the teacher:
   `./scripts/rl_train.sh --total-steps=400000 --out=models/rl/ppo_gates_v3`.
   Success looks like `Episode end:` shifting flyaway -> timeout -> completed
   and `ep_rew_mean` climbing off −24.
4. Then distill: `rl_collect.py --policy=models/rl/ppo_gates_v3 --episodes=20
   --min-gates=2`, `torch_train.py` on the new tub, `eval_flight.sh` on
   slalom_gates_80mm to compare student vs teacher.

### Throughput reality

The `~5.6h` rl_train.py prints counts flying time only and is wrong. Resets
cost ~3s and early episodes last ~2s, so wall-clock is reset-dominated:
measured ~5 env steps/sec early on (was 2.8 before this session's fixes).
Throughput rises as episodes lengthen, approaching 20/sec (the control rate)
if the policy learns to stay alive. Budget 10-20h for 400k, not 6h. Progress
is never lost: checkpoints land in `<out>/checkpoints/` every 20k steps and
`--resume=<out dir>` restores model + optimizer + VecNormalize.

NOTE: on `--resume`, `--total-steps` means that many ADDITIONAL steps (SB3
semantics), not a total budget.

### Bugs fixed this session (do not regress)

- **Velocity clamp blinded the altitude damper.** `gz_telemetry` used to zero
  any velocity above 20 m/s to hide teleport spikes. The damper reads that
  same value, so past 20 m/s it saw `vz=0`, applied no correction, and the
  drone climbed to 50-100m unchecked. Velocities now SATURATE at
  `MAX_VELOCITY` (never zero — a big reading must still produce a big
  correction), and teleports call `reset_history()` on `SimTelemetry` /
  `PoseTracker` to drop the stale sample instead.
- **The damper only runs inside the altitude deadband** (`abs(alt) <
  DRONE_ALTITUDE_HOLD_DEADBAND`, currently 0.1 for 80mm). Commanding any
  altitude during a reset switches off the very thing that arrests the drone.
  Reset loops must command altitude 0.0.
- **Reset worker starvation.** Spawning one `uv run` process per reset (~2s
  startup) meant a 1s re-request loop dropped 29% of requests behind a
  "previous worker still running" guard; the drone drifted unpinned and the
  400k run died at 30.7k steps. `gz_reset_worker.py --serve` is now a single
  persistent process fed pose lines on stdin; requests are ~free.
- **Reset "calm" criterion was stricter than physics.** The drone never goes
  still — at neutral it hovers on the damper and bobs between 2-4m. Requiring
  <0.3m drift over 0.5s made every reset run its full timeout; <1.0m (~2 m/s)
  gives 3s resets.
- **Neutral throttle does not lift the drone off the ground** (measured: 5s
  at hover PWM, zero movement), so a landed reset needs a real climb command
  to take off — see the ramped takeoff below.
- **Altitude is no longer a policy axis.** `GateChaseEnv` is 3-action
  (yaw/pitch/roll, obs 19 wide) and always commands altitude 0.0 so the
  vertical damper stays engaged; `control_altitude=True` / `--control-altitude`
  restores the 4th axis (obs 20, scaled by `--altitude-scale`). Reason: the
  axis is direct motor throttle, any nonzero command disengages the damper,
  and at 0.3 scale the policy had too little authority to arrest a climb it
  could still start. `rl_collect.py` records `last_applied_action` (the
  4-wide command actually flown), not the raw policy output.
- **Takeoff must stop at liftoff, not at start_alt.** Throttle authority is
  ~±100 PWM and the band between "won't leave the ground" (0.15) and "rockets
  to 9m" (0.6) sits inside it. `_takeoff` now ramps from 0.2 by +0.01/50ms and
  cuts the moment `vz > liftoff_vz`, then `_arrest_climb` holds neutral until
  vertical motion is small and accepts whatever height the damper settles on
  (`min_start_alt`). Powering all the way to `start_alt` made the drone cross
  it at ~8 m/s and coast to 9m with the damper saturated.
- BetaFlight sometimes fails to arm at stack start (`RX_FAILSAFE`,
  `NOT_DISARMED`, `ARM_SWITCH`, or MSP query failures). A disarmed board can
  never take off, so `reset()` now re-raises immediately when
  `env.bf_armed` is false instead of burning 8 retries before the supervisor
  rebuilds the stack.
- **`rl_train.sh` is now a supervisor.** BetaFlight SITL randomly aborts at
  launch AND randomly resets its MSP socket during the post-reboot config
  step (`start.sh` retries the launch but not that config step). The script
  retries up to `RL_MAX_ATTEMPTS` (40), rebuilding the stack and resuming
  from checkpoint each time, so an unattended run survives it.
- `<out>/env_config.json` records rate/altitude-scale/etc; `rl_collect.py`
  reads it so the teacher is flown exactly as trained.

### BLOCKER: the 80mm airframe does not hold altitude (2026-08-23)

**RESOLVED later the same day — see the "Current Handoff (2026-08-23,
altitude blocker resolved)" section at the bottom. Kept for the record.**

RL training on `slalom_gates_80mm` cannot work until this is resolved. Every
episode ends in vertical flyaway in ~2.5s no matter what the policy does, so
reward is flat (`ep_rew_mean` −24.1 -> −23.7 over 64k steps, 1660/1660
episodes flyaway, 0/3 gates ever).

Reproduce with `uv run --env-file .env python scripts/probe_hover.py` against
a running stack (ramps to liftoff, then holds neutral 30s logging pos/vel):

    t= 0.0s z= 0.34  vz=1.51  thr_pwm=1553
    t=10.7s z=26.38  vz=1.46  thr_pwm=1430
    t=29.9s z=69.83  vz=3.00  thr_pwm=1349

The drone climbs 0.34m -> 69.8m in 30s at neutral, while the damper holds
throttle 35-175 PWM BELOW the 1475 hover bias the whole time. Note also that
`DRONE_THROTTLE_RANGE=100` means the pilot band is only 1375-1575, so a pilot
cannot command descent at all — only the damper's extra bias gets below 1375.

This is NOT an RL bug and predates the RL work; it likely also explains the
altitude sag/climb noted in earlier BC handoffs and the lateral-coast Z climb.

Complicating it: `drone_config_80mm.py` says `DRONE_HOVER_THROTTLE = 1475` was
"Calibrated at 79g AUW with test_thrust damper-sim", so either that
calibration no longer holds (airframe mass/drag changed since — the model SDF
and plugin drag were both edited in earlier sessions) or the climb comes from
somewhere other than the hover bias. Note `DRONE_THROTTLE_SCALE = 0.2` also
shapes the stick curve non-linearly and is easy to overlook.

Do NOT jump to "hover throttle is simply too high" and start editing
`drone_config_80mm.py`. A throttle sweep (damper disabled, hover_throttle
varied 1475 -> 1200, teleport to 4m, measure drift) appeared to show the drone
FALLING at 1475 — the opposite of the neutral-hold result. That sweep is
itself suspect: every row reported z_end = 0.01, i.e. the drone was sitting on
the ground rather than hovering, so the teleport-to-4m step probably never
took and the numbers may mean nothing. Treat BOTH the sweep and any simple
"hover is too high" story as unverified. What IS solid and repeatable is the
neutral-hold climb above.

Candidate causes, none ruled out:

- `PoseTracker.get_vz()` / `SimTelemetry.velocity` reading the wrong entity on
  `dynamic_pose/info` (the subscriber matches any name containing "betaloop").
- BetaFlight airmode / throttle-mixer behaviour making thrust depend on
  history rather than the commanded CH3 PWM.
- The Gazebo plugin's thrust model interacting with tilt.

Suggested next step: `./scripts/test_thrust.sh --airframe=80mm` (its hover and
damper-sim modes exist for exactly this) to recalibrate
`DRONE_HOVER_THROTTLE`/`DRONE_THROTTLE_RANGE` before any further RL runs.
Raising `flyaway_z` is NOT a workaround: the drone would still leave the gates
far below it.

### Debugging notes

- gz service calls from a torch-loaded process time out; always use a
  subprocess (see `gz_camera_worker.py`, `gz_reset_worker.py`).
- When resets misbehave, log pos AND vel together — position alone cannot
  distinguish "teleport failed" from "teleport worked, drone flew away".
  A signature of x/y pinned exactly on target while z runs away means the
  teleport is fine and something is driving it vertically.

## Current Handoff (2026-08-23, altitude blocker resolved)

**The 80mm altitude flyaway is FIXED. RL training is unblocked.** Everything
is stopped; no training run is in progress.

### Root cause

Not the hover bias. BetaFlight's stock pitch/roll rate PIDs (p=45-47,
i=80-84, d=38-46, f=120-125) limit-cycle in SITL: at fixed CH3 with the drone
level, the front/rear motor pairs slammed bang-bang between ~1086 and ~1906
PWM (observed live via MSP_MOTOR, cmd 104). Attitude barely moves, but
thrust ∝ ω², so the symmetric split rectifies into phantom collective
thrust — measured +2 to +7.7 m/s² extra, nearly independent of CH3. The
throttle stick and damper had almost no authority over net thrust; average
thrust was set by the oscillation's duty cycle. Same ω² mechanism as the
documented "yaw adds thrust" bug, but on the pitch axis and spontaneous at
hover. It also explains the earlier BC-era altitude sag/climb, the
lateral-coast Z climb, and why hover "calibration" never reproduced.

### Fixes (all in the working tree)

- `scripts/start.sh`: pitch/roll PIDs lowered to p=20 i=10 d=0 f=20 (both
  axes, both airframes) via the CLI phase — `PID_GAINS` extends the old
  `YAW_PID` mechanism. With the tame gains the motor split collapses to
  ~30 PWM and thrust is a clean function of CH3 again.
- `scripts/start.sh` CLI phase hardened. Two silent-failure bugs meant a
  fresh eeprom could fly on stock PIDs + LEGACY mixer while the log claimed
  success: (1) after the airmode-disable in-process MSP_REBOOT, this SITL
  build's CLI goes permanently deaf — '#' never returns a prompt and every
  get/set no-ops; (2) nothing verified the sets. Now: airmode disable
  restarts via the exit-42 path instead of MSP_REBOOT (config loop is 3
  passes: airmode → CLI → clean MSP-only), CLI entry retries until a prompt
  actually appears, and a post-apply `get p_pitch` probe aborts the launch
  if the sets didn't stick.
- `nanopilot/drone_config_80mm.py`: `DRONE_HOVER_THROTTLE` 1475 → 1480
  (static hover re-measured at ≈1478 — the old value was essentially right;
  the oscillation was the climb). `DRONE_ALTITUDE_HOLD_K` 45 → 90: thrust
  slope near hover is only ~0.01 m/s² per PWM, so K=45 gave ~0.45/s of
  closed-loop damping, too soft against residual thrust scatter (~±0.5 m/s²
  from small I-term motor splits during activity).

### Verification

`probe_hover.py` on `slalom_gates_80mm`, from a WIPED `eeprom.bin` and fully
fresh stack: z 0.5 → 2.1m over 30s at neutral, vz +0.02-0.03 m/s, throttle
steady 1477-1479. Reproduced twice (also on a carried-over eeprom).
Before the fix: z 0.34 → 69.8m in 30s.

Measurement gotchas discovered (worth keeping):

- The sim has NO vertical drag: open-loop climb *rate* at a PWM is
  meaningless (it just preserves entry momentum). Measure vertical
  *acceleration* (quadratic fit of z(t)), or close the loop.
- `test_thrust.py --mode=hover` sweeps from the ground; ground contact
  masks the true hover point (that is how 1475 was "confirmed" wrong-ish).
  Measure hover airborne.
- Active PID activity adds rectified thrust even with tame gains, so a
  PI-on-vz "hover finder" reads ~40-60 PWM low while it is actively
  stabilizing. Trust ballistic holds entered at ~zero vz, cross-checked
  against static thrust math (km=1.9e-7, maxRpm=2094, mass 0.079kg →
  hover ≈ motor 1482).
- BetaFlight refuses CLI while armed, and a CLI socket needs ~0.3s after
  connect before sending '#', or it never answers.

### Follow-ups

1. Flight dynamics changed (rate loops much softer, d=0). Pre-existing
   models (tub_12 etc.) and old eval numbers are NOT comparable to the new
   tune. Re-validate `DRONE_PITCH_THROTTLE_FEEDFORWARD` and the damper-sim
   check (`test_thrust.sh` damper mode) before further BC work.
2. 65mm inherits the same pitch/roll gains from start.sh but was not
   re-tested airborne.
3. RL: resume from the morning handoff's step 3:
   `./scripts/rl_train.sh --total-steps=400000 --out=models/rl/ppo_gates_v3`.
   Old checkpoints (`ppo_gates_v1/v2`) trained against the oscillating plant
   are junk for the new dynamics.

NOTE: items 1 and 3 are DONE — see the newest handoff at the bottom. The
pitch feed-forward was retuned (8.0 -> 2.0) and `ppo_gates_v3` ran to 280k
and was superseded by `ppo_gates_v4`. Item 2 (65mm) is still open.

## Current Handoff (2026-08-23 evening, RL teacher learning to fly gates)

**STATE: `models/rl/ppo_gates_v4` is TRAINING RIGHT NOW** as a detached
process (see "Checking on the run"). It is the first teacher that actually
passes gates. Do not start a new RL run without checking whether this one is
still alive or has finished.

### What changed since the altitude-blocker handoff above

Three separate problems were fixed in sequence. Each one masked the next, so
the order matters if you are reading this to understand why the configs look
the way they do:

1. **BF pitch/roll PIDs limit-cycling** (fixed earlier today, see the section
   above): phantom collective thrust, drone left the world at neutral stick.
2. **Pitch throttle feed-forward too large**: `DRONE_PITCH_THROTTLE_FEEDFORWARD`
   8.0 -> 2.0. A velocity-only damper cancels a constant thrust bias only at a
   nonzero climb rate (`vz_steady = ff*pitch/K`), so a large ff made any
   sustained pitch climb. At ff=8 the drone gained ~4m per 60s of half-pitch
   flight. Swept 0/2/4/6/8 airborne with the damper engaged
   (`scripts/ff_sweep.py`); ff=2 minimizes |drift| and, unlike higher values,
   is not systematically upward (+2.0m/60s at half pitch, -0.9m at full).
   Also tested hover 1477 vs 1480 — 1480 is better, left alone.
3. **The RL policy had no altitude axis at all.** This was the real reason
   `ppo_gates_v3` could never pass gates (below).

### Why v3 failed, and how that was diagnosed

`models/rl/ppo_gates_v3` (3-action, no altitude axis) ran to 280k steps. It
looked healthy — `ep_rew_mean` climbed -71 -> -14, flyaways went to ~0,
nearly every episode ended `completed` — but the gate-pass rate plateaued at
~15% of episodes passing >=1 gate, **0% ever passing >=2**, and 0 completions
of 3/3 in 972 episodes.

`ep_rew_mean` was a misleading health metric here: "completed" only means the
drone crossed the plane of every gate, whether or not it went *through* the
opening (a miss still advances the gate index, costing -3 vs +10 for a pass).
So a policy that flies the course fast and misses every gate scores well.
**Track the `gates X/3` distribution, not reward.**

A diagnostic rollout (`scripts/gate_rollout.py`, now in the repo) settled it:
17 gate-plane crossings, 0 passes, and **all 17 missed on HEIGHT** — crossing
at 3.07-9.99m (mean 4.77m) when gates sit at z=1.5 with a 0.2-3.0m pass band.
Lateral aim was fine (median offset 1.55m against a 0.9m half-width). The
drone was flying over the gates and, with no altitude axis, had no actuator
to fix it: it starts near 1.4m and drifts above the 3.0m ceiling within ~25s,
after which every remaining gate is unreachable by construction.

The reward shaping was never the problem. Do not "fix" gate misses by raising
`pass_bonus` without first checking crossing geometry with `gate_rollout.py`.

### v4: what is running now

    ./scripts/rl_train.sh --total-steps=400000 --out=models/rl/ppo_gates_v4 \
        --control-altitude --altitude-scale=0.09

`--altitude-scale=0.09` is the load-bearing detail. `DRONE_THROTTLE_SCALE=0.2`
makes the throttle curve very steep near zero, so 0.09 keeps every policy
command inside the 0.1 damper deadband while still giving ~±62 PWM (~±1 m/s²)
of trim authority. The policy gets real altitude control AND the vertical
damper stays engaged as a safety net. The earlier abandoned attempt used
scale 0.3, where any meaningful command *disengages* the damper — an
all-or-nothing actuator, which is why it was judged unlearnable.

Verify the axis is live by grepping commanded altitude in the log: v4 shows
`alt=` values spread over ±0.09; v3 showed only `alt=0.00` (the 0.20-0.23
values in both logs are the scripted takeoff ramp, not the policy).

Results so far (v4 vs v3 at comparable points):

| metric | v3 @280k, 972 eps | v4 @~105k, 237 eps |
| --- | --- | --- |
| episodes passing >=1 gate | 15% | 75-80% |
| episodes passing >=2 gates | 0% | 17-20% |
| episodes passing 3/3 | 0 | 0 |
| `ep_rew_mean` | -14 | ~0 (briefly +2.8) |

At ~178k steps: gates 0/1/2/3 = 186/159/68/0 cumulative. 2-gate episodes end
`completed` after ~400 steps, i.e. the drone flies the whole course and misses
exactly one gate — a precision problem, not a capability one.

On the zero 3/3: the gate-count distribution matches near-independent per-gate
passing (p~0.32-0.34), which predicts 3/3 at only ~3-4%. There is no evidence
of a structural barrier at the third gate; it should appear on its own as p
rises (at p=0.45 it would be ~9%). Do not panic-tune because 3/3 is 0.

### Checking on the run

    pgrep -fl rl_train.sh                      # is it alive?
    tail -f logs/rl_train_ppo_gates_v4.log
    for g in 0 1 2 3; do \
      echo "gates $g/3: $(grep -c "gates $g/3" logs/rl_train_ppo_gates_v4.log)"; done

Windowed pass rate (the metric that matters):

    python3 - <<'PY'
    import re
    eps=[int(m.group(1)) for l in open('logs/rl_train_ppo_gates_v4.log',errors='replace')
         if (m:=re.search(r'Episode end: \w+, gates (\d)/3',l))]
    for lbl,w in (("last 100",eps[-100:]),("last 50",eps[-50:])):
        print(f"{lbl}: >=1 {sum(1 for g in w if g>=1)/len(w)*100:.0f}%  "
              f">=2 {sum(1 for g in w if g>=2)/len(w)*100:.0f}%  3/3={sum(1 for g in w if g==3)}")
    PY

### SHARP EDGE: a killed run silently restarts from scratch

`rl_train.py --resume` loads only `<out_dir>/model.zip`, and that file is
written **when `model.learn()` returns** (clean finish or KeyboardInterrupt).
A run killed by a signal leaves only `<out_dir>/checkpoints/`, and
`rl_train.sh` resumes only `if [ -f "$OUT_DIR/model.zip" ]` — so relaunching
after a hard kill **starts from step 0 with no warning and discards
everything**. This bit this session (the v4 background task was killed at
~178k; the process died by signal with no traceback).

Recovery, before relaunching:

    cp models/rl/ppo_gates_v4/checkpoints/ppo_gates_<N>_steps.zip \
       models/rl/ppo_gates_v4/model.zip
    cp models/rl/ppo_gates_v4/checkpoints/ppo_gates_vecnormalize_<N>_steps.pkl \
       models/rl/ppo_gates_v4/vecnorm.pkl

Then confirm the log says `Resuming from models/rl/ppo_gates_v4`. Remember
`--total-steps` on resume means ADDITIONAL steps (SB3 semantics), and you must
pass `--control-altitude --altitude-scale=0.09` again — they define the action
space the checkpoint expects, and `env_config.json` does not exist until the
run finishes.

Worth fixing properly: `rl_train.sh` should fall back to the newest checkpoint
in `checkpoints/`, or refuse to start, instead of quietly starting over.

**Launching so it survives:** macOS has no `setsid(1)`, and a plain background
launch dies with its process group. `scripts/detach_launch.py` forks, calls
`os.setsid()`, redirects stdio to the log and execs the supervisor:

    python3 scripts/detach_launch.py    # edit CMD/LOG at the top first

### Next steps

1. Let v4 finish (400k total; it resumed from 160k with 240k additional).
   Watch the windowed >=2 rate — that is what gates distillation.
2. Distill. NOTE `--min-gates=3` (the older handoff's suggestion) would
   currently yield ZERO episodes. Use 2:

       GZ_HEADLESS=1 GZ_WORLD=slalom_gates_80mm ./scripts/start.sh \
           --no-manage --airframe=80mm
       uv run --env-file .env python nanopilot/rl_collect.py \
           --policy=models/rl/ppo_gates_v4 --episodes=40 --min-gates=2

   At ~20% yield, 40 episodes gives ~8 clean tubs; scale episodes to taste.
3. `torch_train.py` on the new tub, then `eval_flight.sh` on
   `slalom_gates_80mm` to compare student vs teacher.
4. Still open from the earlier handoff: the 65mm airframe inherits the new
   pitch/roll PID gains from `start.sh` but has not been re-tested airborne.
5. Old artifacts: `ppo_gates_v1/v2` (oscillating-plant era) and
   `ppo_gates_v3` (no altitude axis) are both superseded. v3's checkpoints
   will NOT load in the 4-action env.

### New diagnostic scripts (added to the repo this session)

| Path | Purpose |
| --- | --- |
| `scripts/gate_rollout.py` | flies a trained policy, reports per-crossing lateral offset / height / speed vs the pass window; splits misses into lateral vs height causes |
| `scripts/ff_sweep.py` | sweeps `DRONE_PITCH_THROTTLE_FEEDFORWARD` airborne with the damper engaged, reporting climb rate per pitch |
| `scripts/detach_launch.py` | `os.setsid()` launcher so long runs survive process-group kills |

Both flight scripts need a running stack and take a few minutes each.

## Current Handoff (2026-08-24, teacher DONE, student distillation FAILS)

**STATE: everything is stopped. No run in progress.** The RL teacher is
finished and good. The BC student distilled from it does NOT fly. Offline the
student clones the teacher well; closed-loop it crashes within ~13s.

### The teacher is done and it works

`models/rl/ppo_gates_v4` finished 400,640 steps cleanly (log
`logs/rl_train_ppo_gates_v4.log`). Final training window: 100% of episodes
passed >=1 gate, ~96% >=2, ~69% 3/3, 149/150 `completed`.

**The 400k final model.zip is NOT the best checkpoint.** Performance peaked
around 320-340k and decayed after. Deterministic `gate_rollout.py`, 10
episodes each:

| teacher | 3/3 episodes | gates | misses |
| --- | --- | --- | --- |
| 340k checkpoint | 10/10 | 30/30 | 0 |
| 400k final | 3/10 | 23/30 | 7, ALL lateral |

Per-20k-step training rate confirms the arc: 3/3 was 69/81/78/61/56% at
300/320/340/360/380k. Use **`models/rl/ppo_gates_v4_best/`** (the 340k
checkpoint staged as model.zip + vecnorm.pkl + env_config.json, with
PROVENANCE.txt). `models/rl/ppo_gates_v4/` keeps the full run + checkpoints.

Lesson: pick the teacher by deterministic rollout, not by final step count.

### The student does not fly

- `data/tub_15_26-08-24`: 7181 frames, 24 kept / 16 dropped episodes,
  `rl_collect.py --min-gates=3` off the 340k teacher. Every kept episode is a
  clean 3/3 run. 95MB.
- `models/bc_rl_v4best.pth`: trained on that tub alone, best val (Huber)
  0.013433 at epoch 4, early-stopped at 9.

Offline the student is a GOOD clone (900 records, correct 120x160 resolution):

| axis | corr | MAE | pred std | label std |
| --- | --- | --- | --- | --- |
| yaw | 0.89 | 0.074 | 0.202 | 0.228 |
| pitch | 0.90 | 0.115 | 0.399 | 0.437 |
| roll | 0.97 | 0.144 | 0.613 | 0.686 |
| **altitude** | **0.25** | 0.014 | **0.0069** | **0.0200** |

Closed-loop `eval_flight.sh` on slalom_gates_80mm, 5 runs x 60s:

| rate | crash rate | mean survival | mean gates |
| --- | --- | --- | --- |
| 30Hz (eval default) | 5/5 | 9.65s | 0.4 |
| 20Hz (teacher's rate) | 5/5 | 13.38s | 0.6 |

### Diagnosis: the altitude axis is not learned, and it is what kills it

Every failed episode is a FLOOR crash, not a tilt crash: `alt_min` is
0.10-0.12 in all 10 episodes against `--crash-z 0.12`. Altitude oscillates
hugely (`alt_std` 0.68-0.91m, ranging 0.10 -> 2.3-3.9m) where the teacher held
z=1.1-1.9m through every gate.

Cause is visible in the offline table: teacher altitude labels have std 0.020
while the other axes are 0.23-0.69. With uniform `TRAIN_LOSS_WEIGHTS`, the
altitude axis contributes ~nothing to the Huber loss, so the model predicts a
near-constant -0.004 (pred std 0.0069, barely a third of the label std, corr
0.25). The student therefore loses the teacher's active vertical trimming and
develops a divergent vertical oscillation into the floor.

`torch_train.py` lines 240-245 already warn about this axis, but the comment
argues AGAINST up-weighting it. That reasoning holds for human-piloted tubs
where altitude really is ~0; it does NOT hold for RL-teacher tubs, where the
small altitude values are real, load-bearing trim.

### Both altitude fixes were tried. Axis is FIXED; flight still fails.

Eval rate note first: `eval_flight.py --rate-hz` defaults to **30** but the
teacher and tubs are **20**. Always pass `--rate-hz=20` for RL-distilled
students. All rows below are 20Hz, 5 runs x 60s, slalom_gates_80mm.

| config | crash | survival | gates | alt_mean | alt_std |
| --- | --- | --- | --- | --- | --- |
| uniform weights, model altitude | 5/5 | 13.4s | 0.6 | 1.48-2.47 | 0.68-0.91 |
| uniform weights, `--zero-altitude` | **0/5** | 60.0s | 0.4 | 3.06-3.46 | 0.36-0.47 |
| `TRAIN_LOSS_WEIGHTS=[1,1,1,25]` | 5/5 | 11.0s | 0.4 | **1.02-1.51** | 0.59-0.82 |

**Experiment 1 (`--zero-altitude`, new flag on eval_flight.py) confirmed the
diagnosis and killed the simple fix at the same time.** Zeroing axis 4 stops
every crash (5/5 -> 0/5, survives the full 60s), so the axis WAS the crash
cause. But the damper alone settles at 3.06-3.46m while gates sit at z=1.5
with a 0.2-3.0m pass band, so the drone cruises above the pass ceiling and
gates stay unreachable. The damper is a survival net, not a height controller.
Do not "fix" this by commanding altitude 0.0.

**Experiment 2 (up-weighting) worked as designed and was still not enough.**
`TRAIN_LOSS_WEIGHTS = [1.0, 1.0, 1.0, 25.0]` is now set in
`drone_config_80mm.py`. Offline on tub_15, altitude corr **0.25 -> 0.76** and
pred std **0.0069 -> 0.0151** (label 0.0200), and the other axes did NOT
suffer (yaw 0.89->0.90, pitch 0.90->0.91, roll 0.97->0.98) — so the warning in
`torch_train.py` about starving the visual axes does not apply at this weight.
Model: `models/bc_rl_v4best_w25.pth` (best val 0.012331; note the loss function
changed, so that number is NOT comparable to the uniform run's 0.013433).

In flight the axis now aims at the RIGHT HEIGHT — alt_mean 1.02-1.51m against
the teacher's 1.1-1.9m, versus 3.2m when zeroed — but altitude variance barely
moved (std 0.59-0.82) and every episode still oscillates down into the floor
(alt_min 0.10-0.12 vs `--crash-z 0.12`). Mean is right, control is not.

### Probe result: altitude was a REPRESENTATION bug, now fixed

The "smooth the altitude command" probe was run and **disproved**: the
student's command was already smoother than the teacher's (cmd_std
0.006-0.014 vs teacher 0.020, step-to-step |delta| ~0.002). It was not
oscillating -- it carried a small persistent NEGATIVE bias.

That bias mattered enormously because `DroneGymEnv._rc_channels` maps altitude
to throttle as `sign(a)*|a|**DRONE_THROTTLE_SCALE` with scale **0.2** — a curve
that is nearly vertical at zero:

    alt=-0.0001 -> -15.8 PWM      alt=-0.0028 -> -30.9 PWM  (teacher mean)
    alt=-0.0104 -> -40.1 PWM      alt=-0.0900 -> -61.8 PWM  (student mean)

So raw altitude labels look like a degenerate spike (std 0.020) while the
thrust they encode spans **std 39.8 PWM**. A 0.0076 raw-space error is ~1e-5 of
Huber loss — invisible even at weight 25 — but **9.3 PWM of sustained downward
thrust**, which is what flew the student into the floor. The axis was badly
CONDITIONED, not merely under-weighted.

Fix: `TRAIN_ALTITUDE_PWM_SPACE = True` in `drone_config_80mm.py` trains the
altitude target in PWM space (`TubDataset(altitude_pwm_scale=...)` pre-applies
the curve; `TorchPilot(altitude_pwm_scale=...)` applies the inverse at
inference; `torch_train.py` records it in `arch_meta`). Transformed labels have
std **0.397**, comparable to pitch 0.449 — so `TRAIN_LOSS_WEIGHTS` went back to
uniform `[1,1,1,1]`.

**The flag must match between training and inference or the altitude axis is
silently wrong, with no error** — same failure class as the 120x160 resolution
gotcha. `eval_flight.py` and `drone_manage.py` both read it from config.

Results (`models/bc_rl_v4best_pwm.pth`, best val 0.009384, trained 28 epochs vs
9-11 before — the loss finally has gradient on that axis):

| metric | alt x25 raw | PWM space |
| --- | --- | --- |
| altitude thrust error | 18.63 PWM | **4.89 PWM** |
| altitude corr | 0.76 | **0.86** |
| yaw / pitch / roll corr | 0.90 / 0.91 / 0.98 | **0.97 / 0.98 / 0.99** |
| mean survival | 11.0s | **18.4s** |
| crash rate | 5/5 | 5/5 |
| gates | 0.4 | 0.4 |

Note the other three axes improved too, because altitude no longer needs a
distorting 25x weight. Commanded altitude now matches the teacher's
distribution (cmd_mean -0.0047 vs teacher -0.0028, cmd_std ~0.017 vs 0.020).

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

The representation bug AND the axis-weight bug are both now ruled out. Offline
fidelity is excellent (0.97/0.98/0.99/0.86) and the commands match the
teacher's distribution, yet the drone still oscillates vertically (alt_std
0.57-0.81, unchanged from before the fix) and clips the floor at 0.12m. Note
the damper alone gives alt_std 0.36-0.47, so commanding the axis at all makes
the oscillation WORSE — this is a closed-loop interaction, not a fit problem.

That is genuine compounding error, and DAgger is the indicated fix.

## Current Handoff (2026-08-31, DAgger round 1 + the real structural blocker)

**STATE: everything stopped, nothing running.** DAgger round 1 is built, run,
trained, and evaluated. It doubled survival but regressed gate passing, and
the investigation turned up a STRUCTURAL reason the student cannot clone this
teacher. Read that section before doing another round.

### DAgger round 1

`nanopilot/dagger_collect.py` (new): flies the STUDENT so the visited states
are its own, and records the TEACHER's action at each state as the label. Cheap
here because the teacher is state-based and can label any state on demand.
Crashed episodes are KEPT by default (the pre-crash states are the corrective
data — dropping them defeats the purpose). `--beta` mixes in teacher-driven
steps if the student crashes too fast to gather useful states.

Verified the labels really are teacher corrections, not student echo: on
tub_16, label vs student output correlate only 0.41-0.65 with mean|diff| up to
0.72 on roll (label -0.17 vs student +0.55).

Data: `data/tub_16_26-08-24` (613 rows, smoke) + `data/tub_17_26-08-24`
(5643 rows, 30 episodes). Trained `models/bc_rl_dagger1.pth` on
tub_15+16+17 (13,437 frames), best val 0.035903. That val is NOT comparable to
the 0.009384 of the pure-teacher model: DAgger data is deliberately harder
off-distribution states.

| config | crash | survival | gates | alt_mean | alt_std |
| --- | --- | --- | --- | --- | --- |
| BC uniform | 5/5 | 13.4s | 0.6 | 1.48-2.47 | 0.68-0.91 |
| BC zero-alt | 0/5 | 60.0s | 0.4 | 3.06-3.46 | 0.36-0.47 |
| BC alt x25 | 5/5 | 11.0s | 0.4 | 1.02-1.51 | 0.59-0.82 |
| BC PWM space | 5/5 | 18.4s | 0.4 | 0.93-1.63 | 0.57-0.81 |
| **DAgger round 1** | 5/5 | **36.5s** | **0.2** | **1.22-1.43** | 0.52-0.98 |

Survival doubled (18.4 -> 36.5s, best of any config that commands altitude) and
altitude centering is the tightest yet (1.22-1.43 vs teacher 1.1-1.9). But
gates REGRESSED to 0.2, and all 5 episodes still end on the floor
(alt_min 0.12 every time). It learned to survive at the cost of gate-seeking.

### THE STRUCTURAL BLOCKER: the student cannot observe what the teacher acts on

`GateChaseEnv` observation `[5:8]` is **velocity** in the drone-heading frame.
The teacher's altitude control is therefore VELOCITY FEEDBACK. The student's
inputs are one camera frame plus a 3x6 IMU history (accel + gyro) — **no
velocity, and no image history**. Vertical velocity is not recoverable from
three accelerometer samples.

So the student is being asked to regress a function of a variable it cannot
see. That is why the vertical oscillation is immune to every data-side fix
tried so far: loss weights (x25), the PWM-space representation, and DAgger all
improved label fidelity without touching observability. Note the damper alone
gives alt_std 0.36-0.47 while every model-commanded config sits at 0.52-0.98.

**The data is already there.** `DRONE_TUB_INPUTS` has recorded
`vel/vel_x,y,z` in every tub including all three current ones (verified:
tub_17 row 0 has vel_z=0.837). `LinearModel.forward(img, imu)` simply never
consumes it, and `TorchPilot.run()` has no parameter for it.

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

1. **Feed velocity to the student.** Extend `LinearModel.forward` to take a
   velocity vector, `TubDataset` to return it (already in the tubs),
   `TorchPilot.run` to accept it, and the callers (`eval_flight.py`,
   `drone_manage.py`, `dagger_collect.py`) to pass `outputs[VEL]`. This is an
   architecture change: existing 4-output checkpoints will NOT load. Gate it
   behind a config flag the way `TRAIN_ALTITUDE_PWM_SPACE` is, and record it in
   `arch_meta`.
   Caveat worth deciding deliberately: on real hardware vz needs a barometer or
   optical-flow estimate. Fine for a sim course; a real-airframe port would
   need an estimator.
2. Only then consider DAgger round 2. Another round without fixing
   observability will keep trading gate accuracy for survival.
3. If velocity input is rejected, the alternative is to give the teacher the
   student's observation space (retrain the RL teacher WITHOUT velocity in its
   obs) so the target policy is actually representable by the student. That is
   a full teacher retrain (~4h) but it makes distillation well-posed.

### Also fixed this session

`stop_all.sh`'s new wrapper kill used `pgrep -f "scripts/start\.sh"`, which
matches any process whose command line merely MENTIONS that path. A training
command that grepped for stack processes matched it and was killed mid-launch
(no log, exit 1, ~30 min lost). The pattern is now ANCHORED to a shell actually
executing the script:
`pgrep -f '^[^ ]*/?bash[^ ]* [^ ]*scripts/start\.sh'`. Verified three ways:
a real wrapper is still reaped, a command merely mentioning the path is not,
and the exact failing command shape now runs to completion.

1. **DAgger is now the indicated fix**, and it is unusually cheap here: the
   teacher is state-based, so it can label ANY state the student visits.
   Fly the student, capture frames, query
   `models/rl/ppo_gates_v4_best` for the action at each visited privileged
   state, and append those corrected rows to the tub. That directly attacks
   the compounding error rather than the label distribution.
2. Cheaper thing worth trying first: smooth/rate-limit the altitude command at
   runtime, or feed the model its own previous altitude command. The failure
   is an oscillation, and the current model is memoryless on that axis.
3. Do NOT re-tune `TRAIN_LOSS_WEIGHTS` further. Altitude corr 0.76 with the
   right cruise height means the label fit is no longer the bottleneck.

### Fixed / added this session

- `scripts/gate_rollout.py` could not load any 4-action policy: it hardcoded
  `control_altitude=False`, so v4 (obs 20) died in `VecNormalize.load` with
  `spaces must have the same shape: (20,) != (19,)`. It now resolves
  `env_config.json` from `--policy` OR its parent (so pointing at
  `checkpoints/` works) and takes `--control-altitude` / `--altitude-scale`
  overrides. It prints what it resolved.
- `scripts/promote_checkpoint.py` (new): watches `<out>/checkpoints/` and
  atomically promotes the newest checkpoint pair to `model.zip`/`vecnorm.pkl`,
  capping the loss from a signal-killed run at one checkpoint interval. Exits
  when the supervisor pid dies so it can never clobber a clean final save.
  Used for the whole back half of the v4 run.

### Still open

- 65mm airframe still not re-tested airborne on the new pitch/roll PID gains.

### Resume footgun and wrapper leak: BOTH FIXED (2026-08-24)

`rl_train.sh` no longer silently restarts from step 0. Before choosing a
resume it runs `promote_checkpoint.py --once`, so a run killed by a signal
(progress only in `checkpoints/`) is promoted and resumed. If checkpoints
exist but none can be promoted, it now **refuses with exit 2** rather than
discarding them. A clean final `model.zip` is never clobbered, because
promotion skips any checkpoint older than `model.zip`.

Verified against four synthetic out-dirs, using the shipped lines extracted
from the script itself:

| case | behavior |
| --- | --- |
| fresh (no checkpoints) | starts at step 0 |
| killed (stale model.zip @20k, checkpoints @40k) | promotes 40k, resumes |
| unpromotable (zip, no vecnormalize pkl) | refuses, exit 2 |
| clean finish (model.zip newer than checkpoints) | keeps final, resumes |

`stop_all.sh` now also kills the `start.sh` wrapper, killing its
`tail -f /dev/null` keepalive child FIRST (otherwise the child is orphaned and
survives its parent anyway). It walks the ancestor pid chain and skips any
`start.sh` that is calling it — the real `start.sh` runs `stop_all.sh` from its
own EXIT trap (`start.sh:88`), so without that guard it would kill itself
mid-cleanup. Verified three ways: an orphaned wrapper is reaped, a wrapper that
invokes `stop_all.sh` itself survives, and a real `--no-manage` stack tears
down to zero wrappers/tails/gz/betaflight/camera-workers.

### Measurement gotcha (cost real time this session)

`TorchPilot`/`LinearModel` must be built at **TRAIN_IMAGE_H/W (120x160)**, not
the camera's 640x480. `AdaptiveAvgPool` means a 640x480 model loads the same
state_dict without error and silently returns badly biased predictions
(roll pred mean -0.92 vs label -0.20). `eval_flight.py` and `drone_manage.py`
both get this right via `getattr(cfg,'TRAIN_IMAGE_H',...)`; ad-hoc probes do
not. If an offline check disagrees wildly with the reported val_loss, suspect
the resolution before suspecting the model.


## Current Handoff (2026-09-01, velocity input added; partial win)

**STATE: stopped, nothing running.** The student now takes a velocity input.
It helped, but it did NOT fix the crashes. Also fixed a significant
pre-existing train/inference image bug found while verifying the change.

### Velocity input (no config flag — unconditional)

`LinearModel.forward(img, imu, vel)` now takes heading-frame velocity through a
small MLP branch. Wired end to end:

| file | change |
| --- | --- |
| `tub_schema.py` | `heading_frame_velocity()` + `VELOCITY_SCALE`, imported by BOTH dataset and pilot so the transform cannot drift |
| `torch_model.py` | `vel_dim=3`, `vel_fc` branch, `forward(img, imu, vel)` |
| `dataset.py` | returns `(image, imu, vel, target)`; velocity was already in every tub |
| `torch_pilot.py` | `run(..., vel_x, vel_y, vel_z, yaw_deg)`, rotates internally |
| `torch_train.py` | 4-tuple unpack, `vel_dim` in `arch_meta` |
| `evaluate.py` | 4-tuple unpack; legacy models still take `(img, imu)` |
| `eval_flight.py`, `dagger_collect.py` | pass `outputs[VEL]` + `att[2]` |
| `drone_manage.py` | `inputs += IMU_KEYS + VELOCITY_KEYS + ["imu/yaw"]`, and REFUSES to start if `DRONE_RECORD_VELOCITY`/`ATTITUDE` are off |

Velocity is rotated into the DRONE-HEADING frame and scaled by 5, matching
`GateChaseEnv._get_obs()[5:8]` exactly, so the student sees the axis in the
units the teacher acted on. Verified yaw-invariant.

Pre-velocity 4-output checkpoints are detected by `checkpoint_kind()` as
`pre_velocity` (no `vel_fc.0.weight`) and rejected with an explicit message
instead of a confusing `load_state_dict` error. `models/bc_rl_dagger1.pth` and
`models/bc_rl_v4best_pwm.pth` are both in that category now.

### Verified: inference matches training to 1.7e-5

Feeding the same tub records through `TubDataset`+`LinearModel` and through
`TorchPilot` sequentially agrees to **1.73e-05** max, once the IMU history has
filled. The first two frames of any such comparison MUST differ — TorchPilot's
history starts zero-filled while the dataset has a full window. Do not read
that as a bug (it cost time here).

### BUG FOUND AND FIXED: TorchPilot resize was not antialiased

`TubDataset` resizes with PIL (antialiased); `TorchPilot` used
`F.interpolate(bilinear)` WITHOUT antialias. At the 4x downscale from the
640x480 camera to 120x160 that aliases hard:

    max pixel difference   0.355  (35% of full range)  ->  0.0029 with antialias
    steady-state output    1.15e-03                    ->  1.73e-05  (66x)

**Every flight eval before 2026-09-01 ran on images subtly different from what
the model trained on.** Now `antialias=True`. Same failure class as the
120x160 resolution gotcha and the PWM-space flag: silent, no error.

### Flight result: real but partial

5 runs x 60s, 20Hz, slalom_gates_80mm:

| config | crash | survival | gates | alt_mean | alt_std |
| --- | --- | --- | --- | --- | --- |
| BC PWM space | 5/5 | 18.4s | 0.4 | 0.93-1.63 | 0.57-0.81 |
| DAgger (no vel) | 5/5 | 36.5s | 0.2 | 1.22-1.43 | 0.52-0.98 |
| **DAgger + velocity** | 5/5 | **37.4s** | **0.4** | **1.10-1.29** | **0.57-0.63** |

`models/bc_vel_dagger1.pth`, best val 0.038066 (not comparable across
architectures).

What velocity bought: altitude variance is now CONSISTENT (0.57-0.63, no more
0.98 outliers), mean height is the tightest yet, and the gate regression from
the no-velocity DAgger round reversed (0.2 -> 0.4). One episode flew 57.7s and
passed 2 gates.

What it did NOT buy: the crashes. Still 5/5, still every episode ending at
alt_min 0.12 (the crash threshold). The oscillation is damped and regularised,
not eliminated — alt_std 0.57-0.63 against 0.36-0.47 for the damper alone.

**So the observability hypothesis was PARTIALLY right.** Velocity was a genuine
missing input, and the evidence for that is the tighter, more consistent
altitude control. But it was not the cause of the floor crashes. Do not expect
another input to fix this on its own.

### Ruled out this session

- Control-loop timing: `loop delay: avg=50.0ms` (exactly 20Hz, min 45 max 55),
  inference 11.5ms = 23% of the period. The loop is not falling behind and
  latency is not pathological.

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

1. The remaining failure is a slow vertical oscillation that persists across
   every label-side and input-side fix. Instrument the actual z trajectory
   (period and phase vs the altitude command) rather than summary stats —
   `eval_flight.py` records `track` but only reports aggregates. If the command
   LAGS z by ~90 degrees, it is a phase/feedback problem, not a fit problem.
2. Compare directly against the teacher flying the same course with
   `gate_rollout.py`: the teacher holds this axis fine with the same actuator,
   so a side-by-side z trace localizes what the student does differently.
3. DAgger round 2 is now reasonable (round 1 aggregated cleanly and the
   velocity input makes the target representable), but expect incremental
   gains, not a fix.


## Current Handoff (2026-09-05, root cause FOUND: it was never an oscillation)

**STATE: stopped, nothing running.** The altitude failure is now fully
explained and quantitatively confirmed. It is NOT a vertical oscillation, not
compounding error, and not a fit problem. Do not spend more effort on label
weighting, representation, DAgger rounds, or extra inputs for this.

### Root cause: a ~5 PWM constant thrust bias, integrated by a plant with no height regulation

`DRONE_ALTITUDE_HOLD_K` is a pure VELOCITY damper (bias = -K*vz, K=90). It has
no position term, so the plant has no altitude regulation at all: any steady
thrust bias B settles at a constant sink rate `vz = B / K` and integrates
straight into the floor. Nothing arrests it.

The student emits exactly such a bias. Measured over the steady phase (first 5s
of takeoff transient excluded), with the prediction from `vz = B/K`:

| episode | mean cmd | predicted vz | actual vz |
| --- | --- | --- | --- |
| student ep0 | -9.3 PWM | -0.104 | **-0.100** |
| student ep1 | -4.6 PWM | -0.051 | **-0.053** |
| student ep2 | -4.9 PWM | -0.055 | **-0.048** |
| student ep3 | -4.8 PWM | -0.053 | **-0.057** |
| teacher (4 eps) | -0.8..+1.3 PWM | ~0 | +0.02..+0.04 |

The model predicts the observed sink to within 0.005 m/s in every episode, and
the predicted time-to-floor (17/30/34/29s) matches the observed episode lengths
(22/35/40/34s). A 5 PWM bias is **2.5% of full scale** and is invisible in
val_loss — but it is fatal, because the plant integrates it.

### Why "oscillation" was the wrong read

`alt_std` 0.57-0.62 looked like an oscillation in the summary stats. The actual
z trace (`eval_flight --trace`) shows a monotonic descent with the command
pushing DOWN the whole way: from t=17s to the crash, vz is negative on nearly
every sample and the command is negative too (-9 to -16 PWM). Per-vz bins make
it stark — while sinking at vz in [-0.3,-0.1] (1020 of the student's steps) the
student commands **-12.7 PWM** where the teacher commands -0.6.

Ruled out along the way, all with data:

- **Not position feedback / phase lag.** corr(cmd, -vz) is 0.60 for the student
  and 0.60 for the teacher — the student learned the right control LAW. Its lag
  is 0.20s vs the teacher's 0.06s, but that is ~1% of the 19s period, far too
  little to destabilize.
- **Not damper disengagement.** The student exceeds the 0.1 deadband on only
  0.73% of steps, and the last disengagement is 21-35s BEFORE each crash.
  (The teacher never exceeds it: max |cmd| = 0.090, exactly its action clamp.)
- **Not saturation.** Neither saturates (0.6-0.7% of steps).
- **Not control-loop timing.** 50.0ms period, inference 11.5ms (ruled out 09-01).

What IS different is command AMPLITUDE: the teacher modulates +-39 PWM (std
39.4) around ~0; the student sits in a narrow band (std 13.0) offset to -7.8.
Its effective damping gain is 7.8 PWM per m/s vs the teacher's 51.9. Offline
shrinkage on the tubs is only 0.68-0.74x, so closed-loop amplitude loss is
larger than the offline fit suggests.

### Confirmed by experiment: remove the DC, the crashes stop

`eval_flight.py --alt-highpass=TAU` high-passes the altitude command in PWM
space (DC station-keeping is the damper's job; the model supplies modulation).
5 runs x 60s, 20Hz:

| config | crash | survival | gates | alt_mean | alt_std | mean cmd |
| --- | --- | --- | --- | --- | --- | --- |
| student raw | 4/4 | 32.7s | 0.25 | 1.15-1.23 | 0.57-0.62 | -5.9 PWM |
| **student + highpass** | **0/5** | **60.0s** | 0.40 | 2.37-2.76 | **0.21-0.45** | **-0.2 PWM** |
| teacher | 0/4 | n/a | **3.00** | 1.57-1.72 | 0.13-0.17 | -1.9 PWM |

Mean vz went -0.05 -> +0.006..+0.021 and every episode survived the full 60s.
That closes the loop on the diagnosis.

But it is NOT a complete fix: with DC removed there is no station-keeping
either, so the drone drifts up to 2.4-2.8m (peaks 3.0-3.5m) against a 0.2-3.0m
gate window, and gates stay at 0.4. Same shape as the older `--zero-altitude`
result (3.1-3.5m), just less extreme. **DC authority is what holds height** —
the teacher's DC is small but not zero (-1.9 PWM) and it holds 1.57-1.72m.

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

The student must produce a DC term accurate to ~2 PWM (2% of full scale) to
hold height open-loop. That is an unreasonable ask of a regression model, and
it is the actual reason every data-side fix failed. Two ways forward, in order:

1. **Fix the plant, not the student (recommended).** Give the altitude hold a
   POSITION term — make it a PD hold about a height setpoint instead of a pure
   velocity damper, so a small thrust bias produces a small steady-state offset
   instead of an unbounded sink. The student''s command then becomes a height
   trim request rather than raw thrust, and a 5 PWM error stops being fatal.
   This also fixes the plant fragility for every future model. Note the teacher
   was trained against the velocity-damper plant, so it would need retraining
   or re-validation against the new dynamics.
2. **Or train the DC explicitly.** Add a loss term on the mean altitude command
   over a window (the current per-sample Huber is dominated by AC modulation and
   cannot see a 2% DC offset). Cheaper, but leaves the fragile plant in place.

Do NOT: re-tune TRAIN_LOSS_WEIGHTS, add more inputs, or run DAgger round 2 for
this failure. All three were tried and none address a DC calibration error.

### Added this session

| path | purpose |
| --- | --- |
| `eval_flight.py --trace` | per-step z / vz / altitude-command trace into the metrics JSON |
| `eval_flight.py --alt-highpass=TAU` | remove the command''s DC thrust bias (diagnostic + partial fix) |
| `eval_flight.py --alt-gain=G` | scale the command in THRUST space (raw * G**5); note the PWM curve is so compressive that G=4 saturates |
| `gate_rollout.py --trace-output` | same trace shape for the teacher, for side-by-side comparison |
| `scripts/alt_phase.py` | period + phase analysis: separates velocity damping from position feedback from lag |


## Current Handoff (2026-09-05b, PD altitude hold: plant FIXED, models now stale)

**STATE: stopped, nothing running.** The altitude hold now has a position term.
It fixes the plant fragility completely. The side effect is that BOTH the RL
teacher and the BC student are now trained against the wrong plant, so gate
performance regressed and they need retraining.

### What changed

`DroneGymEnv._rc_channels` altitude hold is now two modes, split by the
existing command deadband:

    |alt| >= deadband : explicit climb/descent. Direct thrust, and the held
                        height follows the drone (unchanged behaviour).
    |alt| <  deadband : HOLD. bias = -Kp*(z - z_target) - K*vz, about a FROZEN
                        setpoint, where z_target = setpoint + alt_scaled*TRIM.
                        The command trims the target HEIGHT; it no longer adds
                        thrust.

New config (`drone_config_80mm.py`):

    DRONE_ALTITUDE_HOLD_KP   = 30.0   # PWM per metre of height error
    DRONE_ALTITUDE_TRIM_RANGE = 1.0   # metres of setpoint authority in-deadband

`DRONE_ALTITUDE_HOLD_KP = 0` reproduces the old pure-velocity damper exactly,
and is the default when unset — so **65mm is unaffected** (it does not set it).
The setpoint is cleared on every teleport/reset alongside `reset_history()`.

Gain sizing: thrust slope near hover is ~0.01 m/s^2 per PWM, so w^2 = 0.01*Kp
and 2*zeta*w = 0.01*K = 0.9. Kp=30 gives w=0.55 rad/s, zeta=0.82 (well damped),
and turns a 5 PWM bias into a 0.17m offset instead of an unbounded sink.

### It works: the plant is no longer fragile

`probe_hover.py` (neutral stick, 30s): z pinned at **0.30-0.31m, vz 0.00**,
throttle steady 1477-1478. On the old plant the same probe drifted 0.5 -> 2.1m.

Student `bc_vel_dagger1`, 5 x 60s, 20Hz:

| plant | crash | survival | alt_std | mean vz |
| --- | --- | --- | --- | --- |
| velocity damper (old) | 4/4 | 32.7s | 0.57-0.62 | -0.05..-0.10 |
| PD hold (new) | **0/5** | **60.0s** | **0.04-0.09** | **-0.003..0.000** |

Altitude variance improved ~10x, and is now TIGHTER than the teacher ever
achieved on the old plant (0.13-0.17). Vertical drift is gone entirely. The
plant is doing the regulating, which is the point.

### The cost: both models are now trained against the wrong plant

Their altitude commands meant "thrust"; they now mean "height trim". Deterministic
`gate_rollout.py`:

| teacher | episodes 3/3 | gates | z mean | z range |
| --- | --- | --- | --- | --- |
| old plant | 10/10 (and 4/4) | 12/12 | 1.64 | 1.18-2.01 |
| PD plant | 2/6 | 14/18 | 1.35 | 1.00-1.99 |

The student holds rock-steady but at ~2.0-2.4m, and gates went 0.25 -> 0.20.
Two separate causes, do not conflate them:

1. **Stale semantics.** Expected and unavoidable; fixed by retraining.
2. **Takeoff capture height.** `eval_flight` climbs at `--climb-altitude=0.15`,
   which is ABOVE the 0.1 deadband, so the setpoint follows the drone until
   handover — and climb momentum carries it to ~2.4m before the hold freezes.
   The drone then holds 2.4m against gates at 1.5m, with only +-0.63m of trim
   authority to fix it. This is a takeoff-tuning bug, NOT a model failure, and
   it is worth fixing before drawing any conclusion from gate counts.

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

1. **Fix the takeoff capture height first** (cheap, and it confounds everything
   else): arrest the climb and let the hold freeze near `start_alt`, the way
   `rl_env._arrest_climb` already does for RL episodes. Verify the student holds
   ~1.5m rather than ~2.4m before judging gates.
2. **Retrain the teacher on the new plant** (~4h,
   `./scripts/rl_train.sh --total-steps=400000 --out=models/rl/ppo_gates_v5
   --control-altitude --altitude-scale=0.09`). Pick the checkpoint by
   deterministic `gate_rollout.py`, not final step count (see 2026-08-24).
   Consider whether `--altitude-scale` still wants to be 0.09: the axis is now a
   position trim inside the deadband, not thrust, so the old reasoning about
   keeping commands inside the deadband still applies but the authority it buys
   is different (0.09 -> ~0.39m of trim at TRIM_RANGE=1.0).
3. **Then re-collect and re-distill** (`rl_collect.py`, `torch_train.py`). The
   existing tubs (15/16/17) were recorded on the old plant and their altitude
   labels no longer mean what they meant; do not mix them with new data.
4. `DRONE_ALTITUDE_TRIM_RANGE=1.0` was chosen analytically, not tuned. On the
   new plant the teacher uses ~45 PWM mean command = ~0.45m of trim, against
   z excursions of ~0.2m on the old plant, so it may now be over-responsive.
   Worth a sweep once the teacher is retrained.

### Note for interpreting old results

Any eval or tub from before this change ran on a plant with NO height
regulation. Altitude numbers are not comparable across the change, the same way
the 2026-08-23 PID retune invalidated earlier flight comparisons.


## Current Handoff (2026-09-06, v5 teacher on the PD plant: altitude SOLVED, lateral is now the wall)

**STATE: stopped, nothing running.** `models/rl/ppo_gates_v5` trained to 611k
steps on the PD-hold plant and was stopped deliberately (see budget bug below).
`model.zip` holds the promoted 600k checkpoint.

### The takeoff capture fix worked, and it alone moved gates

`eval_flight._takeoff` now arrests the climb (waits for |vz|<0.15) and then
calls the new `DroneGymEnv.set_altitude_setpoint(start_alt)`, so episodes begin
at a repeatable height instead of wherever momentum left the setpoint.
`rl_env._arrest_climb` does the same. Same student model, no retraining:

| | before | after |
| --- | --- | --- |
| episode start z | 2.30-2.40 | **1.63-1.64** |
| alt_mean | 2.24-2.44 | **1.43-1.52** (gates at 1.50) |
| gates | 0.2 | **1.0** |

The old numbers were measuring a drone parked above the gates.

### BUDGET BUG: the run cannot terminate on its own

`rl_train.sh` restarted 8 times (BetaFlight SITL flakiness, expected). Every
resume passes `--total-steps=400000` again, and on resume that means 400k
ADDITIONAL steps (SB3 semantics, already documented). So each restart granted a
fresh 400k and the run reached 611k of a nominal 400k with no end in sight.
Resume/promotion all worked correctly -- no progress was ever lost -- but the
BUDGET is unbounded. **Fix before the next long run:** have `rl_train.sh`
compute remaining steps as `total - already_done` when resuming, or pass an
absolute target and let it subtract.

### The plant fix did exactly what it was supposed to

Deterministic `gate_rollout.py`, 10 episodes, v5 @600k:

    crossings=29 passes=19 (66%)
    misses by lateral=10, by height=0
    height at crossing: mean 1.73m, min 1.22, max 2.62 (band 0.2-3.0)
    |lateral offset|: mean 1.03m, median 0.52m, max 5.01m (window 0.9m)

**Zero height misses.** Altitude is no longer a failure mode for the teacher at
all, which is precisely what the PD hold was for. Compare v3 on the old plant,
which missed 17/17 crossings on HEIGHT.

### But the teacher is much weaker than v4, and it is all lateral

| teacher | plant | 3/3 episodes | gates |
| --- | --- | --- | --- |
| v4 @340k (best) | velocity damper | 10/10 | 30/30 |
| v4 @400k (final) | velocity damper | 3/10 | 23/30 |
| **v5 @600k** | **PD hold** | **3/10** | **19/30** |

v5 is still IMPROVING, not decayed (windowed 3/3 went 10% at 280k -> 17% in the
last 60 episodes), but it is far off v4 at 1.5x the steps. Its episodes are also
slower: mean 413 steps vs v4''s 307 at peak. 96% complete the course; they just
miss gates sideways.

### Leading hypothesis for the next run: DROP the altitude axis

With the PD hold, the altitude action is close to a no-op -- the plant holds
height whatever the policy does, and the rollout shows 0/29 height misses. A
4-action policy is therefore spending exploration on a dimension that barely
affects the outcome, which is a plausible cause of the slower lateral learning.

`--control-altitude` was originally added (v3 -> v4) *because* the old plant had
no height regulation and a 3-action policy flew over the gates. **That reason no
longer exists.** A 3-action policy (obs 19) on the PD plant is the natural
configuration and has a smaller search space.

This is a hypothesis, not a measured result. The alternative is simply that v5
needs more steps.

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

### Do this next (pick one)

1. **Retrain 3-action on the PD plant** (drop `--control-altitude`). Smaller
   action space, and the plant now owns altitude. Starts from scratch, ~6-10h.
2. **Resume v5** (`--resume` is automatic; it was still improving). Cheaper, but
   it is 1.5x past v4''s step count at a fifth of the 3/3 rate, so the return is
   uncertain. Remember the additive-steps semantics when setting the budget.

Either way: pick the final checkpoint by deterministic `gate_rollout.py`, never
by final step count.

### Also worth noting

- RL takeoff heights spread 1.44-1.91m (mean 1.72) against the 1.50 target --
  only 15% land within 0.15m. `eval_flight` hits 1.63-1.64 reliably, so the
  `rl_env` convergence loop (4s budget, 0.15m tolerance) gives up early. Harmless
  for robustness (spawns are jittered anyway) but it means the teacher trained
  from a looser start distribution than intended.
- `DRONE_ALTITUDE_TRIM_RANGE=1.0` is still un-swept.

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
