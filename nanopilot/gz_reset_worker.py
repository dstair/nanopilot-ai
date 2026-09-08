#!/usr/bin/env python3
"""
Pose reset helper.

Two modes:

    gz_reset_worker.py <world> <model> <x> <y> <z> <yaw>    one-shot
    gz_reset_worker.py <world> <model> --serve              read commands

Run as a subprocess (same pattern as gz_camera_worker): gz-transport service
requests made from a process that has torch/SB3 loaded time out reliably,
while a plain process succeeds instantly.

--serve reads "x y z yaw" lines from stdin and issues one set_pose_vector per
line, reusing a single Node. RL training resets hundreds of times, and paying
`uv run` startup (~2s) per reset both starves the reset loop and dominates
wall-clock; one long-lived worker makes a reset request effectively free.

Callers should not trust the exit code alone — verify the pose via telemetry
and re-request if it didn't stick.
"""

import math
import sys


def build_msg(model, x, y, z, yaw):
    from gz.msgs10.pose_v_pb2 import Pose_V

    msg = Pose_V()
    p = msg.pose.add()
    p.name = model
    p.position.x = x
    p.position.y = y
    p.position.z = z
    p.orientation.w = math.cos(yaw / 2.0)
    p.orientation.z = math.sin(yaw / 2.0)
    return msg


def send(node, world, model, x, y, z, yaw):
    from gz.msgs10.boolean_pb2 import Boolean

    msg = build_msg(model, x, y, z, yaw)
    result, reply = node.request(
        f"/world/{world}/set_pose_vector", msg, type(msg), Boolean, 2000
    )
    return bool(result) and reply is not None and reply.data


def main():
    world, model = sys.argv[1], sys.argv[2]

    from gz.transport13 import Node

    node = Node()

    if sys.argv[3] == "--serve":
        # One request per stdin line; stdout is a progress log for humans.
        for line in sys.stdin:
            parts = line.split()
            if len(parts) != 4:
                continue
            x, y, z, yaw = (float(v) for v in parts)
            ok = send(node, world, model, x, y, z, yaw)
            print(
                f"set_pose {'ok' if ok else 'FAILED'}: "
                f"({x:.2f}, {y:.2f}, {z:.2f}, yaw={yaw:.2f})",
                flush=True,
            )
        return

    x, y, z, yaw = (float(v) for v in sys.argv[3:7])
    ok = send(node, world, model, x, y, z, yaw)
    print(f"set_pose {'ok' if ok else 'FAILED'}: {model} -> ({x}, {y}, {z}, yaw={yaw})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
