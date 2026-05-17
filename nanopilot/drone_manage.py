#!/usr/bin/env python3
"""
Scripts to fly a drone in BetaFlight SITL + Gazebo simulator.

This is a modified version of manage.py that uses DroneGymEnv instead of
DonkeyGymEnv. The DonkeyCar pipeline (web controller, CNN model, training,
recording) is reused unchanged.

Usage:
    drone_manage.py (drive) [--model=<model>] [--js] [--xbox] [--type=(linear|categorical)] [--camera=(single|stereo)] [--meta=<key:value> ...] [--myconfig=<filename>]
    drone_manage.py (train) [--tubs=tubs] (--model=<model>) [--type=(linear|inferred|tensorrt_linear|tflite_linear)]

Options:
    -h --help               Show this screen.
    --js                    Use physical joystick.
    --xbox                  Use Xbox controller (pygame, macOS-friendly).
    --meta=<key:value>      Key/Value strings describing a piece of meta data about this drive. Option may be used more than once.
    --myconfig=filename     Specify myconfig file to use.
                            [default: drone_config_65mm.py]
"""

import json
import os
from docopt import docopt

try:
    import cv2
except:
    pass

import donkeycar as dk
from donkeycar.parts.transform import TriggeredCallback, DelayedTrigger
from donkeycar.parts.tub_v2 import TubWriter
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.controller import (
    LocalWebController as _LocalWebController,
    WebFpv,
    JoystickController,
)
from donkeycar.parts.web_controller.web import latch_buttons
from donkeycar.parts.throttle_filter import ThrottleFilter
from donkeycar.parts.behavior import BehaviorPart
from donkeycar.parts.file_watcher import FileWatcher
from donkeycar.parts.launch import AiLaunch
from donkeycar.parts.kinematics import (
    NormalizeSteeringAngle,
    UnnormalizeSteeringAngle,
    TwoWheelSteeringThrottle,
)
from donkeycar.parts.kinematics import (
    Unicycle,
    InverseUnicycle,
    UnicycleUnnormalizeAngularVelocity,
)
from donkeycar.parts.kinematics import (
    Bicycle,
    InverseBicycle,
    BicycleUnnormalizeAngularVelocity,
)


class LocalWebController(_LocalWebController):
    """Override to use local templates and broadcast drone RC telemetry."""

    def __init__(self, *args, **kwargs):
        import os

        this_dir = os.path.dirname(os.path.realpath(__file__))
        local_static = os.path.join(this_dir, "templates", "static")

        super().__init__(*args, **kwargs)
        self.roll = 0.0

        import tornado.websocket
        from tornado.routing import URLSpec

        class _DroneWebSocketDriveAPI(tornado.websocket.WebSocketHandler):
            def check_origin(self, origin):
                return True

            def open(self):
                logger.info("New client connected")
                self.application.wsclients.append(self)

            def on_message(self, message):
                data = json.loads(message)
                self.application.angle = data.get("angle", self.application.angle)
                self.application.throttle = data.get("throttle", self.application.throttle)
                self.application.roll = data.get("roll", self.application.roll)
                self.application.altitude = data.get("altitude", self.application.altitude)
                if data.get("drive_mode") is not None:
                    self.application.mode = data["drive_mode"]
                    self.application.mode_latch = self.application.mode
                if data.get("recording") is not None:
                    self.application.recording = data["recording"]
                    self.application.recording_latch = self.application.recording
                if data.get("buttons") is not None:
                    latch_buttons(self.application.buttons, data["buttons"])

            def on_close(self):
                logger.info("Client disconnected")
                self.application.wsclients.remove(self)

        self.default_router.rules.insert(0, URLSpec(r"/wsDrive", _DroneWebSocketDriveAPI))

        # Donkeycar's parent registered /static/ at its own templates/static.
        # We want local files (e.g. our patched main.js) to take precedence
        # while donkeycar's bootstrap/jquery/etc. continue to be served from
        # the upstream path. Insert a higher-priority handler that only serves
        # files which actually exist locally.
        if os.path.isdir(local_static):
            from tornado.web import StaticFileHandler
            from tornado.routing import URLSpec

            local_dir = local_static
            upstream_dir = self.static_file_path

            class _OverlayStatic(StaticFileHandler):
                def initialize(self, path=None):
                    self._dirs = (local_dir, upstream_dir)
                    super().initialize(path=local_dir)

                async def get(self, path, include_body=True):
                    for d in self._dirs:
                        candidate = os.path.join(d, path)
                        if os.path.isfile(candidate):
                            self.root = d
                            break
                    return await super().get(path, include_body)

            spec = URLSpec(r"/static/(.*)", _OverlayStatic)
            # Insert before donkeycar's /static/ rule so we match first.
            self.default_router.rules.insert(0, spec)

        self._last_rc = {
            "roll": None,
            "pitch": None,
            "yaw": None,
            "throttle": None,
            "arm": None,
            "mode": None,
        }
        self._last_bf = {
            "armed": None,
            "arming_flags": None,
            "arming_disable_flags": None,
            "active_modes": None,
        }

    def run_threaded(
        self,
        img_arr=None,
        num_records=0,
        mode=None,
        recording=None,
        rc_roll=None,
        rc_pitch=None,
        rc_yaw=None,
        rc_throttle=None,
        rc_arm=None,
        rc_mode=None,
        bf_armed=None,
        bf_arming_flags=None,
        bf_arming_disable_flags=None,
        bf_active_modes=None,
    ):
        result = super().run_threaded(
            img_arr=img_arr,
            num_records=num_records,
            mode=mode,
            recording=recording,
        )

        # Broadcast PWMs only when they change to keep websocket traffic low.
        rc_changes = {}
        for key, val in (
            ("roll", rc_roll),
            ("pitch", rc_pitch),
            ("yaw", rc_yaw),
            ("throttle", rc_throttle),
            ("arm", rc_arm),
            ("mode", rc_mode),
        ):
            if val is None:
                continue
            ival = int(val)
            if self._last_rc[key] != ival:
                self._last_rc[key] = ival
                rc_changes[key] = ival

        if rc_changes and self.loop is not None:
            self.loop.add_callback(lambda: self.update_wsclients({"rc": rc_changes}))

        bf_values = {
            "armed": bool(bf_armed) if bf_armed is not None else None,
            "arming_flags": bf_arming_flags,
            "arming_disable_flags": bf_arming_disable_flags or "",
            "active_modes": bf_active_modes or "",
        }
        bf_changes = {}
        for key, val in bf_values.items():
            if self._last_bf[key] != val:
                self._last_bf[key] = val
                bf_changes[key] = val
        if bf_changes and self.loop is not None:
            self.loop.add_callback(lambda: self.update_wsclients({"bf": bf_changes}))

        angle, throttle, altitude, mode, recording, buttons = result
        return angle, throttle, self.roll, altitude, mode, recording, buttons

    def run(self, *args, **kwargs):
        return self.run_threaded(*args, **kwargs)


from donkeycar.parts.explode import ExplodeDict
from donkeycar.parts.transform import Lambda
from donkeycar.parts.pipe import Pipe
from donkeycar.utils import *
from drone_env import build_drone_env
from tub_schema import IMU_KEYS, drone_tub_schema

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def add_drone_sim(V, cfg):
    """
    Add the BetaFlight drone simulator as a DonkeyCar part.

    This replaces add_simulator() from manage.py. It creates a DroneGymEnv
    that connects to BetaFlight SITL via UDP RC packets and captures camera
    images via either gz-transport (native macOS) or RTSP (Docker).

    The memory key names (steering, throttle, cam/image_array) are kept
    identical to the car version so all other parts work unchanged.
    """
    gym = build_drone_env(cfg)

    inputs = ["steering", "throttle", "roll", "altitude", "user/arm", "user/reset"]
    outputs = [
        "cam/image_array",
        "rc/roll",
        "rc/pitch",
        "rc/yaw",
        "rc/throttle",
        "rc/arm",
        "rc/mode",
    ]

    if cfg.DRONE_RECORD_POSITION:
        outputs += ["pos/pos_x", "pos/pos_y", "pos/pos_z"]
    if cfg.DRONE_RECORD_ATTITUDE:
        outputs += ["imu/roll", "imu/pitch", "imu/yaw"]
    if cfg.DRONE_RECORD_VELOCITY:
        outputs += ["vel/vel_x", "vel/vel_y", "vel/vel_z"]
    if getattr(cfg, "DRONE_RECORD_IMU", False):
        outputs += IMU_KEYS
    outputs += [
        "bf/armed",
        "bf/arming_flags",
        "bf/arming_disable_flags",
        "bf/active_modes",
    ]

    V.add(gym, inputs=inputs, outputs=outputs, threaded=True)


def drive(
    cfg,
    model_path=None,
    use_joystick=False,
    use_xbox=False,
    model_type=None,
    camera_type="single",
    meta=[],
):
    """
    Construct a working drone vehicle from many parts.

    This mirrors manage.py's drive() function exactly, with add_simulator()
    replaced by add_drone_sim(). All other parts (web controller, model
    inference, DriveMode, TubWriter) are reused unchanged.
    """
    logger.info(f"PID: {os.getpid()}")

    if model_type is None:
        if cfg.TRAIN_LOCALIZER:
            model_type = "localizer"
        elif cfg.TRAIN_BEHAVIORS:
            model_type = "behavior"
        else:
            model_type = cfg.DEFAULT_MODEL_TYPE

    # Initialize vehicle
    V = dk.vehicle.Vehicle()

    # Initialize logging
    if cfg.HAVE_CONSOLE_LOGGING:
        logger.setLevel(logging.getLevelName(cfg.LOGGING_LEVEL))
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(cfg.LOGGING_FORMAT))
        logger.addHandler(ch)

    #
    # Add drone simulator (replaces add_simulator)
    #
    if getattr(cfg, "USE_DRONE_SIM", False):
        add_drone_sim(V, cfg)
    else:
        # Fall back to original car simulator if not in drone mode
        from donkeycar.parts.dgym import DonkeyGymEnv

        if cfg.DONKEY_GYM:
            gym = DonkeyGymEnv(
                cfg.DONKEY_SIM_PATH,
                host=cfg.SIM_HOST,
                env_name=cfg.DONKEY_GYM_ENV_NAME,
                conf=cfg.GYM_CONF,
                record_location=cfg.SIM_RECORD_LOCATION,
                record_gyroaccel=cfg.SIM_RECORD_GYROACCEL,
                record_velocity=cfg.SIM_RECORD_VELOCITY,
                record_lidar=cfg.SIM_RECORD_LIDAR,
                delay=cfg.SIM_ARTIFICIAL_LATENCY,
            )
            inputs = ["steering", "throttle"]
            outputs = ["cam/image_array"]
            V.add(gym, inputs=inputs, outputs=outputs, threaded=True)

    #
    # setup primary camera (MOCK when using drone sim)
    #
    add_camera(V, cfg, camera_type)

    if cfg.SHOW_FPS:
        from donkeycar.parts.fps import FrequencyLogger

        V.add(
            FrequencyLogger(cfg.FPS_DEBUG_INTERVAL),
            outputs=["fps/current", "fps/fps_list"],
        )

    #
    # add the user input controller(s)
    #
    has_input_controller = (
        hasattr(cfg, "CONTROLLER_TYPE") and cfg.CONTROLLER_TYPE != "mock"
    )
    ctr = add_user_controller(V, cfg, use_joystick, use_xbox=use_xbox)

    #
    # convert 'user/steering' to 'user/angle' for backward compatibility
    #
    V.add(Pipe(), inputs=["user/steering"], outputs=["user/angle"])

    #
    # explode the buttons input map
    #
    V.add(ExplodeDict(V.mem, "web/"), inputs=["web/buttons"])

    V.add(
        Lambda(lambda v: print(f"web/w1 clicked")),
        inputs=["web/w1"],
        run_condition="web/w1",
    )
    V.add(
        Lambda(lambda v: print(f"web/w2 clicked")),
        inputs=["web/w2"],
        run_condition="web/w2",
    )
    V.add(
        Lambda(lambda v: print(f"web/w3 clicked")),
        inputs=["web/w3"],
        run_condition="web/w3",
    )
    V.add(
        Lambda(lambda v: print(f"web/w4 clicked")),
        inputs=["web/w4"],
        run_condition="web/w4",
    )
    V.add(
        Lambda(lambda v: print(f"web/w5 clicked")),
        inputs=["web/w5"],
        run_condition="web/w5",
    )

    # throttle filter
    th_filter = ThrottleFilter()
    V.add(th_filter, inputs=["user/throttle"], outputs=["user/throttle"])

    #
    # maintain run conditions for user mode and autopilot mode parts
    #
    V.add(
        UserPilotCondition(show_pilot_image=getattr(cfg, "SHOW_PILOT_IMAGE", False)),
        inputs=["user/mode", "cam/image_array", "cam/image_array_trans"],
        outputs=["run_user", "run_pilot", "ui/image_array"],
    )

    rec_tracker_part = RecordTracker(cfg)
    V.add(rec_tracker_part, inputs=["tub/num_records"], outputs=["records/alert"])

    if cfg.AUTO_RECORD_ON_THROTTLE:

        def show_record_count_status():
            rec_tracker_part.last_num_rec_print = 0
            rec_tracker_part.force_alert = 1

        if (cfg.CONTROLLER_TYPE != "pigpio_rc") and (cfg.CONTROLLER_TYPE != "MM1"):
            if isinstance(ctr, JoystickController):
                ctr.set_button_down_trigger("circle", show_record_count_status)
        else:
            show_record_count_status()

    # IMU (from physical hardware, skip for drone sim since sim provides telemetry)
    if not getattr(cfg, "USE_DRONE_SIM", False):
        add_imu(V, cfg)

    # FPV preview
    if cfg.USE_FPV:
        V.add(WebFpv(), inputs=["cam/image_array"], threaded=True)

    #
    # load and configure model for inference
    #
    if model_path:
        if model_path.endswith(".pth"):
            from torch_pilot import TorchPilot

            kl = TorchPilot(
                input_shape=(cfg.IMAGE_DEPTH, cfg.IMAGE_H, cfg.IMAGE_W),
                seq_len=getattr(cfg, "SEQUENCE_LENGTH", 3),
            )
        else:
            kl = dk.utils.get_model_by_type(model_type, cfg)

        model_reload_cb = None
        if (
            ".h5" in model_path
            or ".trt" in model_path
            or ".tflite" in model_path
            or ".savedmodel" in model_path
            or ".pth" in model_path
        ):
            load_model(kl, model_path)

            def reload_model(filename):
                load_model(kl, filename)

            model_reload_cb = reload_model

        elif ".json" in model_path:
            load_model_json(kl, model_path)
            weights_path = model_path.replace(".json", ".weights")
            load_weights(kl, weights_path)

            def reload_weights(filename):
                weights_path = filename.replace(".json", ".weights")
                load_weights(kl, weights_path)

            model_reload_cb = reload_weights

        else:
            print("ERR>> Unknown extension type on model file!!")
            return

        V.add(FileWatcher(model_path, verbose=True), outputs=["modelfile/modified"])
        V.add(
            FileWatcher(model_path),
            outputs=["modelfile/dirty"],
            run_condition="run_pilot",
        )
        V.add(
            DelayedTrigger(100),
            inputs=["modelfile/dirty"],
            outputs=["modelfile/reload"],
            run_condition="run_pilot",
        )
        V.add(
            TriggeredCallback(model_path, model_reload_cb),
            inputs=["modelfile/reload"],
            run_condition="run_pilot",
        )

        # Model inputs
        if cfg.TRAIN_BEHAVIORS:
            bh = BehaviorPart(cfg.BEHAVIOR_LIST)
            V.add(
                bh,
                outputs=[
                    "behavior/state",
                    "behavior/label",
                    "behavior/one_hot_state_array",
                ],
            )
            try:
                ctr.set_button_down_trigger("L1", bh.increment_state)
            except:
                pass
            inputs = ["cam/image_array", "behavior/one_hot_state_array"]
        else:
            inputs = ["cam/image_array"]
            if (
                model_path.endswith(".pth")
                and getattr(cfg, "USE_DRONE_SIM", False)
                and getattr(cfg, "DRONE_RECORD_IMU", False)
            ):
                inputs += IMU_KEYS

        # Model outputs
        outputs = ["pilot/angle", "pilot/throttle", "pilot/roll", "pilot/altitude"]

        if cfg.TRAIN_LOCALIZER:
            outputs.append("pilot/loc")

        # Image transformations for inference
        if hasattr(cfg, "TRANSFORMATIONS") or hasattr(cfg, "POST_TRANSFORMATIONS"):
            from donkeycar.parts.image_transformations import ImageTransformations

            logger.info(f"Adding inference transformations")
            V.add(
                ImageTransformations(cfg, "TRANSFORMATIONS", "POST_TRANSFORMATIONS"),
                inputs=["cam/image_array"],
                outputs=["cam/image_array_trans"],
            )
            inputs = ["cam/image_array_trans"] + inputs[1:]

        V.add(kl, inputs=inputs, outputs=outputs, run_condition="run_pilot")

    #
    # Decide steering and throttle based on user or autopilot mode
    #
    V.add(
        DriveMode(cfg.AI_THROTTLE_MULT),
        inputs=[
            "user/mode",
            "user/angle",
            "user/throttle",
            "user/roll",
            "user/altitude",
            "pilot/angle",
            "pilot/throttle",
            "pilot/roll",
            "pilot/altitude",
        ],
        outputs=["steering", "throttle", "roll", "altitude"],
    )

    if (cfg.CONTROLLER_TYPE != "pigpio_rc") and (cfg.CONTROLLER_TYPE != "MM1"):
        if isinstance(ctr, JoystickController):
            aiLauncher = AiLaunch(
                cfg.AI_LAUNCH_DURATION,
                cfg.AI_LAUNCH_THROTTLE,
                cfg.AI_LAUNCH_KEEP_ENABLED,
            )
            V.add(
                aiLauncher,
                inputs=["user/mode", "pilot/throttle"],
                outputs=["pilot/throttle"],
            )
            ctr.set_button_down_trigger(
                cfg.AI_LAUNCH_ENABLE_BUTTON, aiLauncher.enable_ai_launch
            )

    # Recording control
    recording_control = ToggleRecording(
        cfg.AUTO_RECORD_ON_THROTTLE, cfg.RECORD_DURING_AI
    )
    V.add(recording_control, inputs=["user/mode", "recording"], outputs=["recording"])

    #
    # Drive train (MOCK for drone sim, so this is a no-op)
    #
    add_drivetrain(V, cfg)

    #
    # Tub data recording
    #
    inputs, types = drone_tub_schema()

    # Add drone telemetry to tub schema
    if getattr(cfg, "USE_DRONE_SIM", False):
        inputs, types = drone_tub_schema(
            record_position=cfg.DRONE_RECORD_POSITION,
            record_attitude=cfg.DRONE_RECORD_ATTITUDE,
            record_velocity=cfg.DRONE_RECORD_VELOCITY,
            record_imu=getattr(cfg, "DRONE_RECORD_IMU", False),
        )

    if cfg.HAVE_IMU or (cfg.CAMERA_TYPE == "D435" and cfg.REALSENSE_D435_IMU):
        inputs += IMU_KEYS
        types += ["float"] * len(IMU_KEYS)

    if cfg.RECORD_DURING_AI:
        inputs += ["pilot/angle", "pilot/throttle", "pilot/roll", "pilot/altitude"]
        types += ["float", "float", "float", "float"]

    tub_path = (
        TubHandler(path=cfg.DATA_PATH).create_tub_path()
        if cfg.AUTO_CREATE_NEW_TUB
        else cfg.DATA_PATH
    )
    meta += getattr(cfg, "METADATA", [])
    tub_writer = TubWriter(tub_path, inputs=inputs, types=types, metadata=meta)
    V.add(
        tub_writer,
        inputs=inputs,
        outputs=["tub/num_records"],
        run_condition="recording",
    )

    if cfg.PUB_CAMERA_IMAGES:
        from donkeycar.parts.network import TCPServeValue
        from donkeycar.parts.image import ImgArrToJpg

        pub = TCPServeValue("camera")
        V.add(ImgArrToJpg(), inputs=["cam/image_array"], outputs=["jpg/bin"])
        V.add(pub, inputs=["jpg/bin"])

    print("=" * 60)
    print("DRONE MODE (BetaFlight SITL)")
    print("=" * 60)
    if getattr(cfg, "USE_DRONE_SIM", False):
        camera_source = getattr(cfg, "DRONE_CAMERA_SOURCE", "gz_transport")
        if camera_source == "gz_transport":
            camera_info = f"gz-transport  {getattr(cfg, 'DRONE_GZ_CAMERA_TOPIC', '(default topic)')}"
        else:
            camera_info = (
                f"rtsp  {getattr(cfg, 'DRONE_RTSP_URL', 'rtsp://127.0.0.1:8554/live')}"
            )
        rc_host = getattr(cfg, "BETAFLIGHT_RC_HOST", "127.0.0.1")
        rc_port = getattr(cfg, "BETAFLIGHT_RC_PORT", 9004)
        hover = getattr(cfg, "DRONE_HOVER_THROTTLE", 1500)
        thr_range = getattr(cfg, "DRONE_THROTTLE_RANGE", 300)
        max_pitch = getattr(cfg, "DRONE_MAX_PITCH_ANGLE", 25.0)
        max_roll = getattr(cfg, "DRONE_MAX_ROLL_ANGLE", max_pitch)
        max_yaw = getattr(cfg, "DRONE_MAX_YAW_RATE", 90.0)
        yaw_pwm_cap = getattr(cfg, "DRONE_YAW_PWM_CAP", None)
        if yaw_pwm_cap is None:
            yaw_pwm_cap = int(max(0, min(500, round(float(max_yaw) / 0.30))))
        delay_ms = getattr(cfg, "SIMULATED_DELAY_MS", 0)
        angle_mode = getattr(cfg, "DRONE_ANGLE_MODE", True)
        altitude_hold = getattr(cfg, "DRONE_ALTITUDE_HOLD_ENABLED", False)
        print(f"  RC UDP:    {rc_host}:{rc_port}")
        print(f"  Camera:    {camera_info}")
        print(f"  Mode:      {'ANGLE' if angle_mode else 'ACRO'}")
        print(f"  Hover PWM: {hover} ± {thr_range}")
        print(f"  Alt hold:  {'on' if altitude_hold else 'off'}")
        print(f"  Max pitch: {max_pitch}°")
        print(f"  Max roll:  {max_roll}°")
        print(f"  Max yaw:   {max_yaw} deg/s (CH4 ±{yaw_pwm_cap}us)")
        if delay_ms > 0:
            print(f"  Sim delay: {delay_ms}ms")
    print(f"  Web UI:    http://localhost:{cfg.WEB_CONTROL_PORT}")
    print("=" * 60)

    if has_input_controller:
        if isinstance(ctr, JoystickController):
            ctr.set_tub(tub_writer.tub)
            ctr.print_controls()

    # Run the vehicle loop
    try:
        V.start(rate_hz=cfg.DRIVE_LOOP_HZ, max_loop_count=cfg.MAX_LOOPS)
    except BaseException as e:
        logger.error("Vehicle loop exited with %s: %s", type(e).__name__, e)
        import traceback

        traceback.print_exc()


# --- Helper classes reused from manage.py ---


class ToggleRecording:
    def __init__(self, auto_record_on_throttle, record_in_autopilot):
        self.auto_record_on_throttle = auto_record_on_throttle
        self.record_in_autopilot = record_in_autopilot
        self.recording_latch = None
        self.toggle_latch = False
        self.last_recording = None

    def set_recording(self, recording):
        self.recording_latch = recording

    def toggle_recording(self):
        self.toggle_latch = True

    def run(self, mode, recording):
        if recording != self.last_recording:
            logging.info(f"Recording Change = {recording}")

        if self.toggle_latch:
            if self.auto_record_on_throttle:
                logger.info("auto record on throttle is enabled; ignoring toggle.")
            else:
                recording = not self.last_recording
            self.toggle_latch = False

        if self.recording_latch is not None:
            recording = self.recording_latch
            self.recording_latch = None

        if recording and mode != "user" and not self.record_in_autopilot:
            logging.info("Ignoring recording in auto-pilot mode")
            recording = False

        if self.last_recording != recording:
            logging.info(f"Setting Recording = {recording}")

        self.last_recording = recording
        return recording


class DriveMode:
    def __init__(self, ai_throttle_mult=1.0):
        self.ai_throttle_mult = ai_throttle_mult

    def run(
        self,
        mode,
        user_steering,
        user_throttle,
        user_roll,
        user_altitude,
        pilot_steering,
        pilot_throttle,
        pilot_roll,
        pilot_altitude,
    ):
        if mode == "user":
            return (
                user_steering,
                user_throttle,
                user_roll if user_roll else 0.0,
                user_altitude if user_altitude else 0.0,
            )
        elif mode == "local_angle":
            return (
                pilot_steering if pilot_steering else 0.0,
                user_throttle,
                user_roll if user_roll else 0.0,
                user_altitude if user_altitude else 0.0,
            )
        return (
            pilot_steering if pilot_steering else 0.0,
            pilot_throttle * self.ai_throttle_mult if pilot_throttle else 0.0,
            pilot_roll if pilot_roll else 0.0,
            pilot_altitude if pilot_altitude else 0.0,
        )


class UserPilotCondition:
    def __init__(self, show_pilot_image=False):
        self.show_pilot_image = show_pilot_image

    def run(self, mode, user_image, pilot_image):
        if mode == "user":
            return True, False, user_image
        else:
            return False, True, pilot_image if self.show_pilot_image else user_image


class RecordTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_num_rec_print = 0
        self.dur_alert = 0
        self.force_alert = 0

    def run(self, num_records):
        if num_records is None:
            return 0

        if self.last_num_rec_print != num_records or self.force_alert:
            self.last_num_rec_print = num_records
            if num_records % 10 == 0:
                print("recorded", num_records, "records")
            if num_records % self.cfg.REC_COUNT_ALERT == 0 or self.force_alert:
                self.dur_alert = (
                    num_records
                    // self.cfg.REC_COUNT_ALERT
                    * self.cfg.REC_COUNT_ALERT_CYC
                )
                self.force_alert = 0

        if self.dur_alert > 0:
            self.dur_alert -= 1

        if self.dur_alert != 0:
            col = (0, 0, 0)
            for count, color in self.cfg.RECORD_ALERT_COLOR_ARR:
                if num_records >= count:
                    col = color
            return col

        return 0


def load_model(kl, model_path):
    start = time.time()
    print("loading model", model_path)
    kl.load(model_path)
    print("finished loading in %s sec." % (str(time.time() - start)))


def load_weights(kl, weights_path):
    start = time.time()
    try:
        print("loading model weights", weights_path)
        kl.model.load_weights(weights_path)
        print("finished loading in %s sec." % (str(time.time() - start)))
    except Exception as e:
        print(e)
        print("ERR>> problems loading weights", weights_path)


def load_model_json(kl, json_fnm):
    start = time.time()
    print("loading model json", json_fnm)
    from tensorflow.python import keras

    try:
        with open(json_fnm, "r") as handle:
            contents = handle.read()
            kl.model = keras.models.model_from_json(contents)
        print("finished loading json in %s sec." % (str(time.time() - start)))
    except Exception as e:
        print(e)
        print("ERR>> problems loading model json", json_fnm)


def get_camera(cfg):
    cam = None
    if cfg.CAMERA_TYPE == "PICAM":
        from donkeycar.parts.camera import PiCamera

        cam = PiCamera(
            image_w=cfg.IMAGE_W,
            image_h=cfg.IMAGE_H,
            image_d=cfg.IMAGE_DEPTH,
            vflip=cfg.CAMERA_VFLIP,
            hflip=cfg.CAMERA_HFLIP,
        )
    elif cfg.CAMERA_TYPE == "WEBCAM":
        from donkeycar.parts.camera import Webcam

        cam = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
    elif cfg.CAMERA_TYPE == "CVCAM":
        from donkeycar.parts.cv import CvCam

        cam = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
    elif cfg.CAMERA_TYPE == "MOCK":
        from donkeycar.parts.camera import MockCamera

        cam = MockCamera(
            image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH
        )
    elif cfg.CAMERA_TYPE == "IMAGE_LIST":
        from donkeycar.parts.camera import ImageListCamera

        cam = ImageListCamera(path_mask=cfg.PATH_MASK)
    else:
        raise Exception("Unknown camera type: %s" % cfg.CAMERA_TYPE)
    return cam


def add_camera(V, cfg, camera_type):
    if getattr(cfg, "USE_DRONE_SIM", False):
        # In drone sim mode, images come from DroneGymEnv (gz-transport or RTSP).
        # Add a MOCK camera as a fallback (it won't overwrite sim images
        # since DroneGymEnv writes to cam/image_array first).
        return

    inputs = []
    outputs = ["cam/image_array"]
    threaded = True
    cam = get_camera(cfg)
    if cam:
        V.add(cam, inputs=inputs, outputs=outputs, threaded=threaded)


def add_user_controller(V, cfg, use_joystick, use_xbox=False, input_image="ui/image_array"):
    ctr = LocalWebController(port=cfg.WEB_CONTROL_PORT, mode=cfg.WEB_INIT_MODE)
    V.add(
        ctr,
        inputs=[
            input_image,
            "tub/num_records",
            "user/mode",
            "recording",
            "rc/roll",
            "rc/pitch",
            "rc/yaw",
            "rc/throttle",
            "rc/arm",
            "rc/mode",
            "bf/armed",
            "bf/arming_flags",
            "bf/arming_disable_flags",
            "bf/active_modes",
        ],
        outputs=[
            "user/steering",
            "user/throttle",
            "user/roll",
            "user/altitude",
            "user/mode",
            "recording",
            "web/buttons",
        ],
        threaded=True,
    )

    if use_xbox:
        from xbox_controller import XboxDroneController

        xbox_ctr = XboxDroneController(
            deadzone=getattr(cfg, "XBOX_DEADZONE", 0.08),
            steering_scale=getattr(cfg, "XBOX_STEERING_SCALE", 1.0),
            throttle_scale=getattr(cfg, "XBOX_THROTTLE_SCALE", 1.0),
            roll_scale=getattr(cfg, "XBOX_ROLL_SCALE", 1.0),
            altitude_scale=getattr(cfg, "XBOX_ALTITUDE_SCALE", 1.0),
            arm_threshold=getattr(cfg, "XBOX_ARM_THRESHOLD", 0.5),
        )
        V.add(
            xbox_ctr,
            outputs=[
                "user/steering",
                "user/throttle",
                "user/roll",
                "user/altitude",
                "user/mode",
                "recording",
                "user/arm",
                "user/reset",
            ],
        )
        # Xbox part is the controller of record for trigger callbacks etc.
        ctr = xbox_ctr

    if use_joystick or cfg.USE_JOYSTICK_AS_DEFAULT:
        if cfg.CONTROLLER_TYPE == "pigpio_rc":
            from donkeycar.parts.controller import RCReceiver

            ctr = RCReceiver(cfg)
            V.add(
                ctr,
                inputs=["user/mode", "recording"],
                outputs=["user/steering", "user/throttle", "user/mode", "recording"],
                threaded=False,
            )
        elif cfg.CONTROLLER_TYPE == "custom":
            from my_joystick import MyJoystickController

            ctr = MyJoystickController(
                throttle_dir=cfg.JOYSTICK_THROTTLE_DIR,
                throttle_scale=cfg.JOYSTICK_MAX_THROTTLE,
                steering_scale=cfg.JOYSTICK_STEERING_SCALE,
                auto_record_on_throttle=cfg.AUTO_RECORD_ON_THROTTLE,
            )
            ctr.set_deadzone(cfg.JOYSTICK_DEADZONE)
            V.add(
                ctr,
                inputs=[input_image, "user/mode", "recording"],
                outputs=["user/steering", "user/throttle", "user/mode", "recording"],
                threaded=True,
            )
        elif cfg.CONTROLLER_TYPE == "mock":
            from donkeycar.parts.controller import MockController

            ctr = MockController(
                steering=cfg.MOCK_JOYSTICK_STEERING, throttle=cfg.MOCK_JOYSTICK_THROTTLE
            )
            V.add(
                ctr,
                inputs=[input_image, "user/mode", "recording"],
                outputs=["user/steering", "user/throttle", "user/mode", "recording"],
                threaded=True,
            )
        else:
            from donkeycar.parts.controller import get_js_controller

            ctr = get_js_controller(cfg)
            if cfg.USE_NETWORKED_JS:
                from donkeycar.parts.controller import JoyStickSub

                netwkJs = JoyStickSub(cfg.NETWORK_JS_SERVER_IP)
                V.add(netwkJs, threaded=True)
                ctr.js = netwkJs
            V.add(
                ctr,
                inputs=[input_image, "user/mode", "recording"],
                outputs=["user/steering", "user/throttle", "user/mode", "recording"],
                threaded=True,
            )
    return ctr


def add_imu(V, cfg):
    if cfg.HAVE_IMU:
        from donkeycar.parts.imu import IMU

        imu = IMU(
            sensor=cfg.IMU_SENSOR, addr=cfg.IMU_ADDRESS, dlp_setting=cfg.IMU_DLP_CONFIG
        )
        V.add(
            imu,
            outputs=IMU_KEYS,
            threaded=True,
        )


def add_drivetrain(V, cfg):
    if cfg.DRIVE_TRAIN_TYPE != "MOCK":
        from donkeycar.parts import actuator, pins
        from donkeycar.parts.actuator import TwoWheelSteeringThrottle

        if cfg.DRIVE_TRAIN_TYPE == "PWM_STEERING_THROTTLE":
            from donkeycar.parts.actuator import (
                PWMSteering,
                PWMThrottle,
                PulseController,
            )

            dt = cfg.PWM_STEERING_THROTTLE
            steering_controller = PulseController(
                pwm_pin=pins.pwm_pin_by_id(dt["PWM_STEERING_PIN"]),
                pwm_scale=dt["PWM_STEERING_SCALE"],
                pwm_inverted=dt["PWM_STEERING_INVERTED"],
            )
            steering = PWMSteering(
                controller=steering_controller,
                left_pulse=dt["STEERING_LEFT_PWM"],
                right_pulse=dt["STEERING_RIGHT_PWM"],
            )
            throttle_controller = PulseController(
                pwm_pin=pins.pwm_pin_by_id(dt["PWM_THROTTLE_PIN"]),
                pwm_scale=dt["PWM_THROTTLE_SCALE"],
                pwm_inverted=dt["PWM_THROTTLE_INVERTED"],
            )
            throttle = PWMThrottle(
                controller=throttle_controller,
                max_pulse=dt["THROTTLE_FORWARD_PWM"],
                zero_pulse=dt["THROTTLE_STOPPED_PWM"],
                min_pulse=dt["THROTTLE_REVERSE_PWM"],
            )
            V.add(steering, inputs=["steering"], threaded=True)
            V.add(throttle, inputs=["throttle"], threaded=True)


if __name__ == "__main__":
    args = docopt(__doc__)
    cfg = dk.load_config(
        config_path=os.path.join(os.path.dirname(__file__), "config.py"),
        myconfig=args["--myconfig"],
    )

    if args["drive"]:
        model_type = args["--type"]
        camera_type = args["--camera"]
        drive(
            cfg,
            model_path=args["--model"],
            use_joystick=args["--js"],
            use_xbox=args["--xbox"],
            model_type=model_type,
            camera_type=camera_type,
            meta=args["--meta"],
        )
    elif args["train"]:
        print("Use python train.py instead.\n")
