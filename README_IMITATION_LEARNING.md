# Imitation Learning in Nanopilot

How the vision autopilot in this repo is trained, why it is trained that way,
and the exact commands to reproduce or watch it. Written for someone arriving
cold — no prior context with the project assumed.

`AGENTS.md` is the working runbook and carries the full dated history. This
file is the explanation and the end-to-end path.

---

## 1. What the policy actually is

A quadrotor flies in Gazebo, controlled by BetaFlight SITL. A small PyTorch
network looks through the drone's forward camera and outputs four numbers,
20 times a second. There is no planner, no state estimator feeding it
positions, and no map. It flies from pixels.

**Inputs**

| input | shape | note |
| --- | --- | --- |
| camera image | 3 x 120 x 160 | downscaled from the 640x480 sensor; 140-degree lens |
| IMU history | 3 x 6 | last 3 frames of accel + gyro, through a GRU |
| velocity | 3 | own velocity, rotated into the drone's heading frame, scaled by 5 |

**Outputs** — four axes, each in `[-1, 1]`:

| output | meaning |
| --- | --- |
| steering | yaw rate |
| throttle | forward speed setpoint |
| roll | lateral speed setpoint |
| altitude | height trim, in metres of `DRONE_ALTITUDE_TRIM_RANGE` |

The network is a small residual CNN (BatchNorm + squeeze-excite, global
average pooling) for vision, a GRU over the IMU sequence, and a tiny MLP on
velocity; the three feature vectors are concatenated into a regression head.
See `nanopilot/torch_model.py`.

**Two design choices worth knowing up front**, because both were arrived at by
failing the other way first:

- **The outputs are SETPOINTS, not stick deflections.** The environment closes
  an inner loop: throttle means "fly forward at this speed", altitude means
  "hold this height offset". Earlier the axes were raw angle and thrust
  commands, and a 5 PWM altitude bias — 2.5% of full scale, invisible in
  validation loss — integrated the drone into the floor in 30 seconds, every
  episode. Under setpoints the same error is a few centimetres of height.
- **The previous control is deliberately NOT an input.** In imitation data
  consecutive controls are identical 90-99% of the time, so a policy given its
  own last action scores near-zero loss by copying it and ignoring the camera.
  At runtime that becomes a self-referential loop that never responds to
  vision.

---

## 2. The method: scripted expert -> behaviour cloning -> DAgger

Three stages. Each exists because the one before it is insufficient in a way
that was measured here, not assumed.

```
  privileged state                  camera pixels
  (exact pose from Gazebo)          (what the drone can really see)
         |                                   |
         v                                   v
   +-----------+   flies, records    +----------------+
   |  EXPERT   | ------------------> |    STUDENT     |
   | scripted  |      tubs           | CNN + GRU      |
   | function  | <------------------ | (deployable)   |
   +-----------+   labels the         +----------------+
                   student's own
                   mistakes (DAgger)
```

### Stage A — a scripted expert, not a learned teacher

The expert is an ordinary Python function of the drone's true position,
velocity and heading (`nanopilot/gate_expert.py`, `nanopilot/line_expert.py`).
It does pure pursuit toward the next target and a feed-forward height law. It
is not trained at all.

That sounds like a step backwards, and it is the single most important choice
in the project. What it buys:

- **Exact labels**, with no reward shaping and no checkpoint to pick. An RL
  teacher's quality varies run to run — the best one produced here scored
  10/10 episodes at 340k steps and 3/10 at its own 400k final.
- **Labels on demand for ANY state.** Because it is a pure function, it can be
  asked "what would you do here?" at a state the expert would never itself
  reach. That is what makes Stage C nearly free.
- **A failure that localises.** If the expert misses, either the control
  cascade is mistuned or the approach law is wrong — both directly measurable.

**Why not reinforcement learning?** It was tried, at length. Direct pixel RL is
infeasible here: the simulator is real-time-locked at roughly 70k steps/hour
and pixel PPO needs millions. State-based RL teachers were then trained (five
generations, `ppo_gates_v1`..`v5`) and distilled into vision students. The
teachers eventually flew well; **every distillation failed**. The scripted
expert matched the best RL teacher ever produced — 30/30 gates, 10/10 episodes
— with zero training time. `rl_train.py` and `rl_env.py` are still in the tree
but the PPO path is retired.

### Stage B — behaviour cloning

Fly the expert, record `(image, IMU, velocity) -> (four commands)` into a tub,
train the network by regression. Yield is ~100%: the expert does not crash, so
essentially every episode is usable.

**This alone does not produce a working policy, and the reason is structural.**
A tub of 40 flawless expert episodes contains no failures, so it contains no
demonstration of recovery. The student reproduces the expert almost exactly on
the states the expert visits — correlations of 0.999 and validation losses
around 1e-4 — and then a small residual error takes it somewhere the expert
never went, where nothing in the training set says what to do.

Observed on this project, four distinct times, all with excellent fits:

| failure | what it looked like |
| --- | --- |
| flies straight past | passed gate 1, then flew 87m in a straight line, in perfect control |
| circles | orbited at 8-10m radius indefinitely, never converging on a target |
| freezes | stopped dead 0.93m short of a gate and held position to within 2cm for 50s |
| flies away | overshot the landing pad, reverted to cruise commands, climbed out |

The freeze is the clearest illustration of why this is structural rather than a
tuning problem. Under speed setpoints, a throttle output near zero does not
coast — it commands a full stop. A stopped drone sees an unchanging camera
frame, so a memoryless policy emits the same near-zero command forever. It is a
stable closed-loop fixed point. The expert's own data cannot teach the way out,
because the expert is above 0.5 throttle in 98% of frames.

### Stage C — DAgger

Dataset Aggregation. Fly the **student** so the visited states are the ones it
actually reaches, but write the **expert's** action as the label at each of
them. Aggregate with the original tub and retrain.

This is cheap here precisely because the expert is a function: no teacher
checkpoint to load, and any state the student blunders into can be labelled by
one call. Crashed episodes are KEPT — those states are the entire point.

The coverage it adds is easy to see. Landing task, expert tub versus DAgger tub:

| | expert rollouts | DAgger round |
| --- | --- | --- |
| forward command range | -0.11 .. +0.75 | **-0.75 .. +0.75** (reverse) |
| forward command std | 0.359 | **0.650** |
| what the expert says past the pad | -0.069 (settling) | **-0.605** (come back) |
| yaw activity \|x\|>0.2 | 1.5% | **28.7%** |

One DAgger round has been sufficient on every course attempted so far.

---

## 3. The rule that trips everyone: ignore DAgger validation loss

**A DAgger model's validation loss is much worse than the model it replaces,
and this does not mean anything is wrong.**

| model | best val loss | closed-loop result |
| --- | --- | --- |
| `bc_expert_land_v1` (BC only) | **0.000111** | flies away 25 times out of 25 |
| `bc_dagger1_land` (BC + DAgger) | **0.002185** | lands 5/5 |

A 20x worse number for the model that actually works. The same pattern appeared
on the gate courses at 350x.

The reason: the train/val split is chronological, so the validation set is the
tail of the DAgger tub — states induced by the **old** student's mistakes. The
retrained student no longer makes those mistakes, so it never visits those
states. High error on a distribution the improved policy does not induce costs
nothing. **DAgger validation loss measures the previous policy's errors.**

Judge a DAgger round by `eval_flight` only. Never by validation loss, and never
by comparing it to a BC-only run's loss.

---

## 4. Running it end to end

All commands assume the repo root, Python 3.12, and `uv sync` already done.
`.env` at the repo root is required.

### 0. Bring the simulator up

```bash
GZ_HEADLESS=1 GZ_WORLD=s_curve_heights_80mm \
  bash scripts/start_gated.sh --no-manage --airframe=80mm
```

`start_gated.sh` launches BetaFlight + Gazebo, waits for readiness, settles
25s, and verifies the flight controller actually answers MSP — rebuilding the
stack if not. BetaFlight SITL aborts at launch perhaps half the time on macOS
ARM64, and a board that survives but is deaf produces flights that all die with
"BetaFlight lost arming". This gate turns a wasted 25-minute collection into a
30-second retry. Drop `GZ_HEADLESS=1` to get a window (see section 5).

### 1. Sanity-check the expert first

```bash
uv run --env-file .env python -u scripts/gate_rollout.py \
  --expert --world=s_curve_heights_80mm --episodes=5
```

The expert is the ceiling for everything distilled from it. If it is not near
100%, stop here — a mediocre expert cannot be repaired downstream.

### 2. Collect expert demonstrations

```bash
uv run --env-file .env python -u nanopilot/rl_collect.py \
  --expert --world=s_curve_heights_80mm --episodes=40 --min-gates=10
```

~25 minutes, ~22k rows, ~100% of episodes kept. Prints `TUB_PATH=...`.

### 3. Train the student

```bash
uv run --env-file .env python -u nanopilot/torch_train.py \
  --tubs=data/<expert_tub> --model=models/bc_expert_vN.pth
```

Writes `.pth` (bare weights, for inference) and `.ckpt` (full resumable state).
`--resume=models/bc_expert_vN.ckpt` continues an interrupted run.

### 4. Evaluate — closed loop, not offline

```bash
GZ_WORLD=s_curve_heights_80mm uv run --env-file .env python -u nanopilot/eval_flight.py \
  --model=models/bc_expert_vN.pth --runs=5 --duration=60 --rate-hz=20 --trace
```

Expect the student to fly well briefly and then hit one of the attractors in
section 2. That is the normal, expected outcome of BC alone.

### 5. DAgger round

```bash
uv run --env-file .env python -u nanopilot/dagger_collect.py \
  --expert --student=models/bc_expert_vN.pth \
  --world=s_curve_heights_80mm --episodes=25
```

Flies the student, labels with the expert, keeps failures.

### 6. Retrain on BOTH tubs, then re-evaluate

```bash
uv run --env-file .env python -u nanopilot/torch_train.py \
  --tubs=data/<expert_tub>,data/<dagger_tub> --model=models/bc_dagger1_vN.pth

GZ_WORLD=s_curve_heights_80mm uv run --env-file .env python -u nanopilot/eval_flight.py \
  --model=models/bc_dagger1_vN.pth --runs=5 --duration=60 --rate-hz=20
```

### 7. Stop

```bash
bash ./scripts/stop_all.sh
```

Run it on its own line — chained after other commands it has killed the calling
shell.

---

## 5. Watching it fly

Everything above runs headless. To see it, omit `GZ_HEADLESS=1`:

```bash
GZ_WORLD=landing_target_80mm bash scripts/start_gated.sh --no-manage --airframe=80mm
```

Then, against that same stack, the expert:

```bash
GZ_WORLD=landing_target_80mm uv run --env-file .env python -u nanopilot/eval_flight.py \
  --expert --runs=3 --duration=60 --rate-hz=20
```

and the trained student:

```bash
GZ_WORLD=landing_target_80mm uv run --env-file .env python -u nanopilot/eval_flight.py \
  --model=models/bc_dagger1_land.pth --runs=3 --duration=60 --rate-hz=20
```

Both run against one stack, so you can do them back to back. For the gate
course, swap the world and the model:

```bash
GZ_WORLD=s_curve_heights_80mm bash scripts/start_gated.sh --no-manage --airframe=80mm

uv run --env-file .env python -u scripts/gate_rollout.py --expert \
  --world=s_curve_heights_80mm --episodes=3

GZ_WORLD=s_curve_heights_80mm uv run --env-file .env python -u nanopilot/eval_flight.py \
  --model=models/bc_dagger1_heights.pth --runs=3 --duration=60 --rate-hz=20
```

**To watch a BC-only student fail**, which is more instructive than watching a
working one succeed: `--model=models/bc_expert_heights_v1.pth` on the heights
course freezes or crashes in most episodes.

Every published number was measured headless. The GUI is a separate process and
competes for the machine while the simulator runs in real time, so small timing
differences with the GUI open are expected and are not regressions.

---

## 6. Results

Each row is the same pipeline on a harder course. "Expert-only" is behaviour
cloning; "+ DAgger" adds one round.

| course | expert (ceiling) | expert-only student | + DAgger |
| --- | --- | --- | --- |
| 3-gate slalom | 3/3 | 1.0 / 3 | not attempted |
| 10-gate S-curve, flat | 10/10 | 3.4 / 10 | **10.0 / 10** |
| 10-gate S-curve, varied heights | 10/10 | 5.6 / 10 | **10.0 / 10** |
| line + landing pad | 0.045m from pad | 0.53m, and 0/25 off-distribution | **0.109m, 5/5** |

Current best models: `bc_dagger1_scurve.pth`, `bc_dagger1_heights.pth`,
`bc_dagger1_land.pth`.

---

## 7. Things that will waste your time if you do not know them

**Evaluate in more than one harness.** The landing student `bc_expert_land_v1`
scores 5/5 under `eval_flight` and fails 25 times out of 25 in the collector,
which jitters the spawn. A single harness passed a policy that does not work.

**Build the model at the TRAINING resolution, 120x160 — never the camera's
640x480.** Global average pooling means a wrongly-sized model loads the same
weights without any error and silently returns badly biased predictions.

**Do not mix tubs across a change of interface, optics, plant or course.** Each
silently changes what a label means. Two tubs recorded with different maximum
pitch angles carry labels 2.25x apart for the same manoeuvre; a model trained
on the union crashed 3/3 while each tub alone trained fine.

**Never pipe a flight script into `tail`, `head` or `grep`.** The camera runs in
a subprocess that inherits stdout, so the pipe never sees EOF and the command
hangs with no output. Redirect stdout and stderr to separate files, and use
`python -u` so a crash does not lose the buffer.

**`eval_flight.py` has no `--world`** — it reads `GZ_WORLD`. `gate_rollout.py`
is the opposite: it takes `--world` and overwrites `GZ_WORLD` from it. Passing
`GZ_WORLD=...` to `gate_rollout.py` without `--world` silently flies one world's
geometry against another world's coordinates.

**A vision policy cannot act on what is outside the frame.** This produced three
separate, initially baffling failures here: a gate 67 degrees off the nose at
the moment of the turn; two identical gates in frame with no cue as to which was
next; and a landing pad that passes behind the drone on an overshoot, after
which the view ahead looks like open cruise and the policy flies away. When a
student with a near-perfect fit behaves inexplicably, check what was visible
before suspecting the network.
