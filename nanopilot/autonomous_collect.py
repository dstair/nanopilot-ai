#!/usr/bin/env python3
"""
Headless scripted data collection for Nanopilot.

Requires BetaFlight SITL + Gazebo to already be running, for example via:
    ./scripts/start.sh --no-manage --airframe=65mm
"""

import argparse
import math
import os
import socket
import threading
import time

import donkeycar as dk
import numpy as np
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.tub_v2 import TubWriter

from drone_env import build_drone_env
from tub_schema import DRONE_TUB_INPUTS, DRONE_TUB_TYPES


def _flight_command(t):
    """Small bounded script that creates useful yaw/pitch/roll/altitude labels."""
    if t < 2.0:
        return 0.0, 0.0, 0.0, 0.0

    phase = t - 2.0
    steering = 0.55 * math.sin(phase * 0.55)
    throttle = 0.25 + 0.12 * math.sin(phase * 0.23)
    roll = 0.35 * math.sin(phase * 0.41)
    altitude = 0.25 * math.sin(phase * 0.35)

    # Every few seconds, briefly straighten out so the tub has recovery data.
    cycle = phase % 8.0
    if cycle > 6.5:
        steering *= 0.25
        throttle = 0.05
        roll *= 0.25
        altitude = -0.15

    return steering, throttle, roll, altitude


def _wait_for_tcp(host, port, timeout_s, label):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            time.sleep(0.25)
        finally:
            sock.close()
    raise RuntimeError(f"Timed out waiting for {label} at {host}:{port}")


def _wait_for_ready(env, timeout_s):
    """Wait until DroneGymEnv has armed BF, camera frames, and IMU telemetry."""
    deadline = time.time() + timeout_s
    last_status = None

    while time.time() < deadline:
        outputs = env.run_threaded(0.0, 0.0, 0.0, 0.0)
        image = outputs[0]
        camera_ready = bool(np.any(image))
        control_ready = env.last_throttle_pwm >= env.hover_throttle - env.throttle_range
        imu_ready = bool(np.any(np.abs(np.asarray(env.imu, dtype=np.float32)) > 1e-6))

        status = (
            f"camera={camera_ready} control={control_ready} imu={imu_ready} "
            f"rc_throttle={env.last_throttle_pwm}"
        )
        if status != last_status:
            print(f"Readiness: {status}")
            last_status = status

        if camera_ready and control_ready and imu_ready:
            return
        time.sleep(0.1)

    raise RuntimeError(f"Timed out waiting for collector readiness: {last_status}")


def collect(cfg, args):
    _wait_for_tcp(
        getattr(cfg, "BETAFLIGHT_RC_HOST", "127.0.0.1"),
        5761,
        args.betaflight_timeout,
        "BetaFlight MSP",
    )

    env = build_drone_env(
        cfg,
        airframe=args.airframe,
        record_position=True,
        record_attitude=True,
        record_velocity=True,
        record_imu=True,
    )
    update_thread = threading.Thread(target=env.update, daemon=True)
    update_thread.start()

    inputs = DRONE_TUB_INPUTS
    types = DRONE_TUB_TYPES
    tub_writer = None
    period = 1.0 / float(args.rate_hz)
    records = 0
    blank_frames = 0
    telemetry_count = 15

    try:
        print(f"Waiting for collector readiness (timeout {args.ready_timeout:.1f}s)...")
        _wait_for_ready(env, args.ready_timeout)

        print(f"Settling at hover: {args.warmup:.1f}s")
        settle_start = time.time()
        while time.time() - settle_start < args.warmup:
            env.run_threaded(0.0, 0.0, 0.0, 0.0)
            time.sleep(0.02)

        tub_path = TubHandler(path=cfg.DATA_PATH).create_tub_path()
        tub_writer = TubWriter(tub_path, inputs=inputs, types=types, metadata=[])
        print(f"TUB_PATH={tub_path}")

        print(f"Collecting: {args.duration:.1f}s at {args.rate_hz}Hz")
        collect_start = time.time()
        next_tick = collect_start
        while time.time() - collect_start < args.duration:
            t = time.time() - collect_start
            steering, throttle, roll, altitude = _flight_command(t)
            env_outputs = env.run_threaded(steering, throttle, roll, altitude)
            image = env_outputs[0]
            telemetry = env_outputs[7 : 7 + telemetry_count]
            if not np.any(image):
                blank_frames += 1
            tub_writer.run(
                image,
                steering,
                throttle,
                roll,
                altitude,
                "user",
                *telemetry,
            )
            records += 1
            if records % max(1, args.rate_hz * 5) == 0:
                print(f"recorded {records} records")
            next_tick += period
            time.sleep(max(0.0, next_tick - time.time()))
    finally:
        if tub_writer is not None:
            tub_writer.shutdown()
        env.run_threaded(0.0, 0.0, 0.0, -1.0)
        time.sleep(0.5)
        env.shutdown()
        update_thread.join(timeout=3.0)

    print(f"Records: {records}")
    if blank_frames:
        print(f"WARNING: {blank_frames} frames were blank")
    print(f"Done: {tub_path}")
    return tub_path


def main():
    parser = argparse.ArgumentParser(description="Collect scripted Nanopilot tub data")
    parser.add_argument("--airframe", choices=["65mm", "80mm"], default="65mm")
    parser.add_argument("--myconfig", default=None)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--warmup", type=float, default=8.0)
    parser.add_argument("--rate-hz", type=int, default=30)
    parser.add_argument("--ready-timeout", type=float, default=20.0)
    parser.add_argument("--betaflight-timeout", type=float, default=10.0)
    args = parser.parse_args()

    myconfig = args.myconfig or f"drone_config_{args.airframe}.py"
    cfg = dk.load_config(
        config_path=os.path.join(os.path.dirname(__file__), "config.py"),
        myconfig=myconfig,
    )
    collect(cfg, args)


if __name__ == "__main__":
    main()
