"""
Smoke test for XboxBridge.app standalone.

Binds /tmp/nanopilot_xbox.sock, launches XboxBridge.app, prints a frame
every 0.5s for ~10s. Verifies that GameController.framework actually
delivers Xbox controller state to a real .app bundle on this Mac.

Usage:
    python xbox_bridge/smoke_test.py
"""
import os
import socket
import struct
import subprocess
import time

SOCK = "/tmp/nanopilot_xbox.sock"
APP = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                   "build", "XboxBridge.app")
FMT = "<fffffBB"
SIZE = struct.calcsize(FMT)


def main():
    try:
        os.unlink(SOCK)
    except FileNotFoundError:
        pass
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    s.bind(SOCK)
    s.setblocking(False)

    print(f"Bound {SOCK}; launching {APP}")
    # 'open' returns immediately after handing off to LaunchServices.
    subprocess.check_call(["open", APP])

    end = time.time() + 25.0
    last_print = 0.0
    last_frame = None
    rx_count = 0
    nonzero_seen = False
    print("===> WIGGLE STICKS / PULL TRIGGERS / PRESS A & B <===")
    while time.time() < end:
        try:
            data, _ = s.recvfrom(64)
            if len(data) >= SIZE:
                last_frame = struct.unpack(FMT, data[:SIZE])
                rx_count += 1
                lX, lY, rX, rY, rT, btns, conn = last_frame
                if abs(lX) + abs(lY) + abs(rX) + abs(rY) + rT > 0.01 or btns:
                    nonzero_seen = True
        except (BlockingIOError, OSError):
            pass
        now = time.time()
        if now - last_print > 0.5:
            last_print = now
            if last_frame is None:
                print("(no frames yet)")
            else:
                lX, lY, rX, rY, rT, btns, conn = last_frame
                a = "A" if btns & 1 else "-"
                b = "B" if btns & 2 else "-"
                marker = "  <-- INPUT!" if (abs(lX)+abs(lY)+abs(rX)+abs(rY)+rT > 0.01 or btns) else ""
                print(f"rx={rx_count:4d} conn={conn} "
                      f"lX={lX:+.2f} lY={lY:+.2f} rX={rX:+.2f} rY={rY:+.2f} "
                      f"rT={rT:.2f} {a}{b}{marker}")
    print(f"DONE. nonzero_input_seen={nonzero_seen}")
    print("done")
    s.close()
    os.unlink(SOCK)
    subprocess.call(["pkill", "-f", "XboxBridge.app"])


if __name__ == "__main__":
    main()
