# Nanopilot: Multi-modal Attention-fused Autonomous Drone Flight

Major thanks and credit to the DonkeyCar project as this was originally forked from DonkeyCar. 

The idea here is that you can train and run a Multi-modal Attention-fused Network that can fly a drone in minutes.

1. Fly around manually; press the record button to keep training data.
2. Train a CNN using that data. The output is an autopilot that flies like you.
3. $un the same simulation using the 

This adapts DonkeyCar's implementation in a number of ways:
- use BetaFlight Software in the Loop. This means you might be able to swap in a real BetaFlight drone (Hardware in the Loop), and use the same flow as above to build an autopilot for it.
- use Gazebo for simulation. There are various adaptations for drone flight: drones move in 3D via pitch/yaw/roll, and throttle commands. This required a significant re-design of the controls.
- The CNN is now updated to take control input data, and IMU data (basically movement data). This gives the model some key inputs for flight; the model still sees the image, but now knows if the user punches the throttle and tilts the drone forward aggressively.

**The current autopilots are trained with no human pilot at all** — a scripted expert flies the course and the vision model is distilled from it (behaviour cloning + DAgger). See [README_IMITATION_LEARNING.md](README_IMITATION_LEARNING.md) for the method, the end-to-end commands, and how to watch it fly.

## Model updates

Why the name Multi-modal Attention-fused Residual CNN-GRU? It captures the vision part (Residual CNN), the temporal IMU part (GRU), and the integration mechanism (Multi-head Attention).
- IMU + GRU: This adds a temporal (Recurrent) dimension; it means the model has "memory." It doesn't just see a single frame; it understands the drone's inertia and momentum. Without this, a single blank frame result in the model giving nonsensical outputs; with it, the model can understand concepts like movement through space.
- GELU (Gaussian Error Linear Unit): This is a modern activation function (used in GPT and BERT). It indicates that the model is using state-of-the-art stochastic regularity rather than the simpler, piecewise-linear ReLU.
- Multi-head Attention: This transforms the model from a simple "concatenation" of inputs into a Transformer-hybrid.

## How It Works

The drone uses the exact same pipeline as DonkeyCar:

```
Web Controller  -->  DriveMode  -->  DroneGymEnv  -->  BetaFlight SITL + Gazebo
 (steering,          (user or         (RC channel        (flies the
  throttle,           autopilot)       packets via         drone)
  altitude)                            UDP 9004)

Camera frames  <--  DroneGymEnv  <--  Gazebo
 (640x480 RGB)      (gz-transport)    (renders scene)
```

Camera mode (set `DRONE_CAMERA_SOURCE` in your `nanopilot/drone_config_XXmm.py`):
- **`gz_transport`** (default) — native macOS: Gazebo Harmonic publishes images
  on a gz-transport topic. GPU-accelerated via Metal/OpenGL; this was key for performance.

**Semantic mapping (DonkeyCar key name → Drone flight axis):**

The four key names are fixed, but what the values MEAN depends on
`DRONE_VELOCITY_CONTROL` (on for the 80mm airframe since 2026-09-06):

| key | velocity mode (current) | direct mode (`DRONE_VELOCITY_CONTROL=False`) |
| --- | --- | --- |
| `steering [-1, 1]` | **yaw rate** (unchanged) | **yaw rate** |
| `throttle [-1, 1]` | **forward speed**, fraction of `DRONE_MAX_FORWARD_SPEED` | **forward pitch** angle |
| `roll [-1, 1]` | **rightward speed**, fraction of `DRONE_MAX_LATERAL_SPEED` | **lateral roll** angle |
| `altitude [-1, 1]` | **height trim**, linear, in metres of `DRONE_ALTITUDE_TRIM_RANGE` | **motor throttle** around hover, via a `\|a\|**0.2` curve |

In velocity mode the environment closes the inner loop (a P+feed-forward
cascade on measured velocity, plus a PD altitude hold), so the learned policy
emits setpoints rather than calibrated thrust and tilt. That change exists
because a 5 PWM altitude bias — 2.5% of full scale, invisible in validation
loss — was enough to fly a distilled model into the floor when the altitude
axis was raw thrust.

Takeoff and landing ramps still need raw thrust and pass it explicitly as
`run_threaded(..., direct_thrust=X)`, which bypasses the hold.

Since the key names are identical to the car version, the web controller,
CNN model (LinearModel), training pipeline, and data recording all work
unchanged across both modes.

## Prerequisites

- **Python 3.12** — gz-harmonic only supports Python 3.12, 3.13, or 3.14
- **uv** — Python package manager
- **macOS on Apple Silicon** — tested on M1, macOS Tahoe 26.3. Should work on any ARM64 Mac.

Side note: the setup was very intricate on macOS. I would recommend you use a coding agent to work through the various build failures you will likely hit. Gazebo was tricky to get up and running, largely because it is highly capable and requires many dependencies.

## Native macOS Setup (GPU-Accelerated)

Runs BetaFlight SITL and Gazebo Harmonic natively on Apple Silicon (ARM64). GPU rendering (not supported by Docker) was needed for acceptable performance. Docker setup for Gazebo was much easier, but rendered aroune 1 FPS on a 2020 Macbook Air, and used ~500% CPU.

### Step 1: Install dependencies (one-time)

Dependencies must come from the **ARM64 Homebrew** at `/opt/homebrew`. Do not
use Rosetta-based Homebrew (`/usr/local/bin/brew`).

```bash
# Gazebo Harmonic (ARM64) — installs gz-sim8, gz-transport13, etc.
brew tap osrf/simulation
brew install gz-harmonic

# Verify Gazebo installed correctly
gz sim --version   # should print: Gazebo Sim, version 8.x.x
```

### Step 2: Build BetaFlight SITL (one-time)

```bash
git clone https://github.com/betaflight/betaflight.git ~/dev/betaflight
cd ~/dev/betaflight
```

BetaFlight's Makefile checks for the ARM cross-compiler before loading the SITL
target config that disables it. Work around this by creating a dummy SDK directory:

```bash
# Hydrate the config submodule
make configs

# Create dummy ARM SDK dir to pass the toolchain check (SITL overrides this to use native clang)
mkdir -p tools/arm-gnu-toolchain-13.3.rel1-darwin-arm64-arm-none-eabi/bin

# Build the SITL target
make TARGET=SITL
```

The build compiles with native clang (not ARM cross-compiler) and produces a
native ARM64 executable. Verify:

```bash
file obj/main/betaflight_SITL.elf
# should print: Mach-O 64-bit executable arm64
```

The `-Ofast` deprecation warnings from clang are harmless.

### Step 3: Build the Gazebo-BetaFlight bridge plugin (one-time)

The [aeroloop_gazebo](https://github.com/aeroloop/aeroloop_gazebo) plugin
connects BetaFlight SITL to Gazebo by relaying motor commands and sensor data
over UDP. Follow its build instructions for ARM64 macOS. Set `AEROLOOP_GAZEBO_DIR`
to point to the install location.

### Step 4: Install Python dependencies

```bash
cd ~/dev/nanopilot-ai
uv sync
```

Create a `.env` file in the project root to tell `uv run` where to find the gz-python bindings:

```bash
# .env  (edit paths if your Homebrew prefix differs)
PYTHONPATH=/opt/homebrew/lib/python3.12/site-packages
DYLD_LIBRARY_PATH=/opt/homebrew/lib
GZ_IP=127.0.0.1   # suppresses noisy multicast warning
```

Verify the bindings are importable:

```bash
uv run --env-file .env python -c "import gz.transport13; print('OK')"
```

### Run Nanopilot

`scripts/start.sh` launches Gazebo and BetaFlight SITL in the background, waits
for readiness, then runs `nanopilot/drone_manage.py` in the foreground. Ctrl+C
stops everything.

```bash
./scripts/start.sh                                               # manual drive mode to collect training data
GZ_WORLD=forest_80mm ./scripts/start.sh --xbox                   # example of optional parameters
./scripts/start.sh --model=models/pilot.pth                      # launch with autopilot
```

Logs are written to `logs/gazebo.log` and `logs/betaflight.log`.

Then open http://127.0.0.1:8887

**What `scripts/start.sh` does:**
1. Launches Gazebo with the selected world
2. Launches BetaFlight SITL
3. Runs `nanopilot/drone_manage.py` which starts the Web UI and sends RC packets
4. On exit or Ctrl+C, calls `scripts/stop_all.sh` to kill all processes cleanly

**Environment variables** for customization:
- `GZ_WORLD` — Gazebo world name. Default is `baylands_80mm`
- `BETAFLIGHT_SITL_BIN` — path to BetaFlight SITL binary (default: `~/dev/betaflight/obj/main/betaflight_SITL.elf`)
- `AEROLOOP_GAZEBO_DIR` — path to aeroloop_gazebo install (default: `~/dev/aeroloop_gazebo`)

(Optional) Open the Gazebo GUI in a separate terminal while the sim is running:

```bash
GZ_IP=127.0.0.1 gz sim -g &
```

---

## Run the training pipeline

When you are ready to record training data, press "Start Recording" in the UI. Fly around to capture the outcome you are looking for. You have a few choices in the worlds/ folder. For example, if you want to train the drone to fly between trees, you can fly between trees in the baylands world. landing_target has a straight line you can fly along before landing. And slalom_gates provides gates for you to fly through.

Once you have collected your training data,, there will be a new "tub" in the `data/` folder. You can then use that data to train an autopilot. Note, this could take a while depending on your input data size. A "hello world" test on 300 images took 5 minutes on an M1 Mac. Here's the command to train. Point it at an existing tub. The --model specifies the name of the model that will be chreated.

```bash
# simpler command
uv run python nanopilot/torch_train.py --tubs=data/tub_[number]_[yy-mm-dd] --model=models/pilot.pth

# more complete command. PYTHONUNBUFFERED gives you more real time output to the logs.
PYTHONUNBUFFERED=1 uv run --env-file .env python -u nanopilot/torch_train.py --tubs=data/tub_10_26-05-25 --model=models/tub_10_26-05-25.pth --myconfig=drone_config_80mm.py --max-epochs=15 --batch-size=16 --no-model-summary
```

Once training completes, note the model output: models/pilot.pth. It is needed for next step.

If interested, you can optionally look at:
- number of epochs. Did early stopping kick in?
- val_loss and when minimum loss occurred at.
- number of samples in training and and validation datasets.

## Flying with the trained autopilot

To fly the autopilot model created in the previous step:

```bash
./scripts/start.sh --model=models/tub_10_26-05-25.pth
```

In the Web UI, switch to "local_angle" or "local" mode. You can always switch back.

### Automated testing of NanoPilot

eval_flight.py loads TorchPilot, resets to the a standard start pose, lets the model fly for N seconds, and scores from gz_telemetry pose data. Metrics that work with zero human judgment:
  - survival time until crash (z below threshold, or attitude beyond ±60°),
  - progress: distance traveled along the corridor axis (landing_target) or gates passed (gate positions are known from the SDF — count aperture plane-crossings),
  - cross-track error and altitude deviation along the way.

Run 3-5 runs; it emits JSON per run; the result is a scoreboard. 

The end to end flow is: collect (human, occasionally) → train → evaluate.py → eval_flight.py → compare JSON. Everything after "collect" runs unattended with GZ_HEADLESS=1.

The script `eval_flight.sh` can conduct an automated evaluation of a given model. Try training in a simple world like `landing_target_80mm` before progressing to the next level of difficulty, the gates (`slalom_gates_80mm`).

`./scripts/eval_flight.sh --model=models/[MODEL_NAME].pth --runs=3`

```bash
=== eval_flight: models/bc_v4.pth on landing_target_80mm (line) ===
  episodes: 3
  success_rate: 0.0
  crash_rate: 0.0
  mean_survival_s: 0.0
  mean_along_track_m: 0.0
  mean_cross_track_m: 0.0
JSON: /path/to/nanopilot-ai/nanopilot/../logs/eval_flight_20260713_140143.json
```

## Configuration

Two airframe configs are maintained side-by-side. Pick one with `--airframe=65mm|80mm` on `start.sh` (default `80mm`, the Pavo Pico II). Edit the matching `nanopilot/drone_config_XXmm.py` to adjust parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DRONE_CAMERA_SOURCE` | `"gz_transport"` | Camera mode: `"gz_transport"` (native macOS) or `"rtsp"` (Docker) |
| `DRONE_GZ_CAMERA_TOPIC` | (drone_course default) | gz-transport topic for camera images |
| `BETAFLIGHT_RC_HOST/PORT` | `127.0.0.1:9004` | BetaFlight SITL RC input endpoint |
| `DRONE_HOVER_THROTTLE` | 1500 | PWM midpoint for hover |
| `DRONE_THROTTLE_RANGE` | 300 | altitude [-1,1] maps to hover ± this PWM range |
| `DRONE_MAX_PITCH_ANGLE` | 25.0 | Max forward pitch angle (degrees) |
| `DRONE_MAX_ROLL_ANGLE` | pitch angle | Max lateral roll angle (degrees) |
| `DRONE_MAX_YAW_RATE` | 90.0 | Max yaw rate (deg/s) at full steering |
| `DRONE_VELOCITY_CONTROL` | False | Pitch/roll become speed setpoints, altitude a linear height trim |
| `DRONE_MAX_FORWARD_SPEED` | 2.0 | Forward speed (m/s) at `throttle` = 1 |
| `DRONE_MAX_LATERAL_SPEED` | 1.0 | Rightward speed (m/s) at `roll` = 1 |
| `SIMULATED_DELAY_MS` | 0 | Simulated camera delay in ms (0=off) |
| `MEASURE_LOOP_DELAY` | True | Log vehicle loop timing stats |
| `DRIVE_LOOP_HZ` | 30 | Vehicle loop frequency |

## Architecture

```
macOS Host (Apple Silicon, ARM64)
+----------------------------------------------+
|                                              |
| gz sim (Gazebo Harmonic, GPU-accelerated)    |
|   +-- drone model (camera sensor attached)   |
|   +-- gz-transport: publishes /world/.../image|
|   +-- aeroloop_gazebo: BF bridge plugin      |
|   |     UDP 9002 ← motor speeds from BF      |
|   |     UDP 9003 → FDM state to BF           |
|                                              |
| BetaFlight SITL (native ARM64 binary)        |
|   +-- Angle mode flight controller           |
|   +-- UDP 9004 ← RC channels from us         |
|                                              |
| drone_manage.py                              |
|   +-- DroneGymEnv                            |
|   |     +-- UDP RC packets → BF (port 9004)  |
|   |     +-- gz_camera_worker ← Gazebo images |
|   +-- LocalWebController (Web UI)            |
|   +-- DriveMode (user/autopilot switch)      |
|   +-- LinearModel (PyTorch CNN)              |
|   +-- TubWriter (records to data/)           |
+----------------------------------------------+
```

## Files

| File | Purpose |
|------|---------|
| `scripts/start.sh`        | Single-command launcher: starts Gazebo + BetaFlight SITL, runs drone_manage, cleans up on exit |
| `scripts/stop_all.sh`     | Kills all BetaFlight/Gazebo processes |
| `nanopilot/drone_manage.py` | Main entry point (replaces `manage.py` for drone use) |
| `nanopilot/drone_gym.py`    | DroneGymEnv part: BetaFlight RC UDP + gz-transport camera bridge |
| `nanopilot/drone_config_80mm.py` | Pavo Pico II 80mm, 79g AUW config (default airframe) |
| `nanopilot/drone_config_65mm.py` | BetaFPV Air65 config (alternate airframe) |
| `nanopilot/config.py`       | Base DonkeyCar config (shared, not modified) |
| `nanopilot/torch_model.py`  | CNN architecture (LinearModel, PyTorch) |
| `nanopilot/torch_pilot.py`  | Inference wrapper for vehicle loop |
| `nanopilot/torch_train.py`  | Training script |
| `worlds/drone_course_65mm.sdf` | 65mm Air65 world (colored walls + Air65 model) |
| `worlds/drone_course_80mm.sdf` | 80mm Pavo Pico II world (same course, 80mm model) |
| `worlds/baylands_80mm.sdf` | Default 80mm Baylands world |
| `worlds/landing_target_80mm.sdf` | most basic, for line following |

## Worlds for autopilot training

All these .sdf files are in the worlds/ folder. They are provided to allow the CNN to train on different scenarios, but feel free to make your own.

- baylands_80mm: default world, based on Gazebo Fuel Baylands
- forest_80mm
- landing_target_80mm
- slalom_gates_80mm
- drone_course_80mm
- drone_course_65mm: retained Air65 course world

GZ_WORLD=slalom_gates_80mm ./scripts/start.sh --xbox

## Troubleshooting

### BetaFlight SITL build

**`arm-none-eabi-gcc not in the PATH`**: The Makefile checks for the ARM toolchain before
the SITL target overrides it. Create the dummy SDK directory as shown in Step 2.

**Rebuild from clean**: If you need to rebuild:
```bash
cd ~/dev/betaflight
make clean TARGET=SITL
mkdir -p tools/arm-gnu-toolchain-13.3.rel1-darwin-arm64-arm-none-eabi/bin
make TARGET=SITL
```

### Gazebo / gz-transport

**`gz-python` not available via pip**: gz-python is installed by `brew install gz-harmonic`
into Homebrew's site-packages. It is NOT on PyPI. The `.env` file's `PYTHONPATH` makes it
importable.

**Camera topic not found / blank frames**: Run
`PATH=/opt/homebrew/opt/ruby/bin:$PATH gz topic -l | grep camera` while Gazebo
is running to confirm the exact topic name. Update `DRONE_GZ_CAMERA_TOPIC` in
`nanopilot/drone_config.py` if the model index differs (e.g., `_1` instead of `_0`).

**`gz sim` fails to load plugins / "incompatible architecture"**: The `gz` Ruby
wrapper must run under ARM64 Ruby. Ensure
`PATH=/opt/homebrew/opt/ruby/bin:/opt/homebrew/bin:$PATH` is set before running
any `gz` command. The x86_64 Ruby at `/usr/local/bin/ruby` will fail to dlopen
ARM64 Gazebo libraries.

### General notes

#### Resource utilization

Depending on your system, you might need to adjust a few parameters to get acceptable performance.

| Change | CPU Savings | Trade-off |
|--------|-------------|-----------|
| `DRIVE_LOOP_HZ = 10` | Less host-side work | Lower control rate |
| Lower camera `update_rate` in world SDF | Halves render load | Fewer training frames per second |

#### Worlds

The default world is **`baylands_80mm`**. The retained `drone_course_65mm`
world provides a 65mm Air65 course; the other included worlds are 80mm worlds.

To switch worlds, set `GZ_WORLD=<world>` when running `start.sh`. The drone
environment derives the camera topic from `GZ_WORLD` and the selected airframe.

### CNN Training size

LinearModel is size-agnostic. The CNN now uses adaptive global pooling after
the convolution stack, so model parameter count does not grow just because the
input image is larger. The main change needed is setting `IMAGE_W` and
`IMAGE_H` in the airframe config.

Higher resolutions can still slow training through JPEG decode, resize,
transforms, memory bandwidth, and larger tub storage. Gazebo rendering cost
also scales with the camera sensor pixel count.

Trade-offs by resolution:

| Resolution | Pixels | Training cost | Notes |
|------------|--------|---------------|-------|
| 160x120 | 19K | Baseline | DonkeyCar car default |
| 320x240 | 77K | Lower I/O | Previous Nanopilot default |
| 640x480 | 307K | Moderate I/O | Current Gazebo sensor native |
| 1280x720 | 922K | Heavy I/O/render | Useful for review/future data |

#### Cleaning Up Data

Several locations accumulate data over time:

| Location | Grows from | Typical size |
|----------|-----------|--------------|
| `data/` | DonkeyCar tub recordings (images + JSON per driving session) | ~1 MB per tub |
| `~/.gz/fuel/` | Gazebo Fuel model cache (downloaded meshes/textures) | ~400 MB |
| `~/.gz/auto_default.log` | Gazebo server log (appended each run) | grows over time |
| `~/dev/betaflight/obj/` | BetaFlight build artifacts | ~100 MB |


```bash
# Check sizes
du -sh data/ ~/.gz/fuel/ ~/.gz/auto_default.log ~/dev/betaflight/obj/

# Delete old tub recordings (keeps the data/ directory)
rm -rf data/tub_*

# Truncate Gazebo log
> ~/.gz/auto_default.log
```

**Safe to delete anytime:** `data/tub_*` (training data you've already used or don't need),
`~/.gz/fuel/` (re-downloads on next run), `~/.gz/auto_default.log`.

**Don't delete unless rebuilding:** `~/dev/betaflight/obj/` — takes a few minutes to rebuild.


## TODO:

X performance acceleration for training on M1 mac.
X Switch drone flying from 2D to 3D. CNN predicts 3 outputs [steering, throttle, altitude].
X Switch from PX4/MAVSDK to BetaFlight SITL for real-world tiny whoop transferability.
X Build aeroloop_gazebo bridge plugin for ARM64 macOS
X Create Gazebo drone model compatible with BetaFlight bridge (camera sensor attached)
X swap out quadcopter type - in my planned build, can't see the rotors.
X research improvements to CNN. multimodal with IMU and control inputs; cross encoding; GLU
X add/test input controller support.
X try a different world.
X add roll for better handling.
X add an 80mm size that matches the Pavo Pico II
X with battery, 79g.
X worlds - all 80 mm
X clean up the UI. Want to show pitch, roll, yaw, and throttle.
  X add status indicator, i.e. ARMED
X black background for nanopilot UI
X Address vertical climb issue.
X fix skating rink effect.
  X Test horizontalVelocityDecay of 0.35 
X Angular drag: adding angular drag (e.g., velocity_decay <angular>) to the model.sdf or the BetaflightPlugin would contribute to more realistic rotational dynamics
X Fix Yaw Speed.
X 640x480
X drone loses altitude slowly during forward flight, how to compensate??
X increase horizontal damping to 0.35.
X Surface more configs in the UI.
  X max throttle range.
  X max roll angle
  X max pitch angle
  X PWM adjustment factors
X Consider More Advanced Drag Models: Implemented additive quadratic horizontal drag in BetaflightPlugin as `F_xy = -m * (linear + quadratic * |v_xy|) * v_xy`.

X Refine model flight characteristics. At some point it still crashes in to the ground. If no road is visible it gets stuck flying in circles.

- try something simpler like just flying through gates? That seems like a better initial pass.

- Add RL component? RLAIF with its own UI? Gates are an interesting example. Or landing zone. Basically, get Drone to a target zone.

X UI - boxes are a bit small at 164 pixels. Double the size. Also put one on left hand side, one on right hand side for better aesthetics.
X UI - some components like input sensitivity just show --; remove them or fix them.

- do I have flipped axes again? I thought I fixed the 90 degree thing???

- Ask Gemini - the results are kind of bad, what to do? Switch to Ardupilot??


## LONG TERM:
- switch back to ArduPilot. Add LIDAR for ground detection, as well as optical flow for movement.
- maybe look at implementing a VLM model??
- Add randomization of worlds (wall locations, colors) for better CNN training
- research other tasks that would be interesting to implement (CNN to scan/build a 3D model of an object, for example)




