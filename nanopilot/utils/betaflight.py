import logging
import socket
import struct
import time

logger = logging.getLogger(__name__)


class MSPClient:
    """Helper for BetaFlight MultiWii Serial Protocol over TCP."""

    ARMING_FLAG_NAMES = [
        "NO_GYRO", "FAILSAFE", "RX_FAILSAFE", "NOT_DISARMED",
        "BOXFAILSAFE", "RUNAWAY", "CRASH", "THROTTLE", "ANGLE",
        "BOOT_GRACE", "NOPREARM", "LOAD", "CALIBRATING", "CLI",
        "CMS_MENU", "BST", "MSP", "PARALYZE", "GPS", "RESC",
        "DSHOT_TELEM", "REBOOT_REQ", "DSHOT_BB", "ACC_CAL",
        "MOTOR_PROTO", "CRASHFLIP", "ALTHOLD", "POSHOLD", "ARM_SWITCH",
    ]

    MODE_FLAG_NAMES = [
        "ARM", "ANGLE", "HORIZON", "ANTIGRAVITY", "MAG", "HEADFREE",
        "HEADADJ", "CAMSTAB", "PASSTHRU", "BEEPER", "LEDLOW",
        "CALIB", "OSD", "TELEMETRY", "SERVO1", "SERVO2", "SERVO3",
        "BLACKBOX", "FAILSAFE_MODE", "AIRMODE",
    ]

    def __init__(self, host="127.0.0.1", port=5761):
        self.host = host
        self.port = port

    def query_status(self, timeout=0.5):
        """Return BF arming-disable flags and active modes.

        Returns (arming_flags, arming_names, mode_bits, mode_names). On failure
        returns (None, [], None, []).
        """
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((self.host, self.port))

            cmd = 150  # MSP_STATUS_EX
            frame = b"$M<" + bytes([0, cmd, 0 ^ cmd])
            sock.send(frame)

            resp = b""
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    chunk = sock.recv(4096)
                except socket.timeout:
                    break
                if not chunk:
                    break
                resp += chunk
                if len(resp) >= 6 and resp[:3] == b"$M>" and len(resp) >= 6 + resp[3]:
                    break

            if len(resp) < 6 or resp[:3] != b"$M>":
                return None, [], None, []

            payload = resp[5 : 5 + resp[3]]
            if len(payload) < 20:
                return None, [], None, []

            mode_bits = struct.unpack("<I", payload[6:10])[0]
            active_modes = [
                self.MODE_FLAG_NAMES[i]
                for i in range(min(len(self.MODE_FLAG_NAMES), 32))
                if mode_bits & (1 << i)
            ]

            flags_byte_count = payload[15] & 0x0F
            offset = 16 + flags_byte_count
            if len(payload) < offset + 5:
                return None, [], mode_bits, active_modes

            flags = struct.unpack("<I", payload[offset + 1 : offset + 5])[0]
            active_arming = [
                self.ARMING_FLAG_NAMES[i]
                for i in range(min(len(self.ARMING_FLAG_NAMES), 29))
                if flags & (1 << i)
            ]
            return flags, active_arming, mode_bits, active_modes
        except OSError as e:
            logger.warning("MSP status query failed: %s", e)
            return None, [], None, []
        finally:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass


def pack_rc_frame(channels):
    """Pack RC channels into BetaFlight SITL's timestamp + 16 uint16 frame."""
    packet = struct.pack("<d", time.time())
    for ch in channels[:16]:
        packet += struct.pack("<H", int(ch))
    for _ in range(16 - len(channels)):
        packet += struct.pack("<H", 1000)
    return packet
