#!/usr/bin/env python3
"""
Diagnostic script for Nanopilot BetaFlight SITL + Gazebo integration.

Run this WHILE the system is running (after ./scripts/start.sh or manually):
    python3 scripts/diagnose.py

It checks:
  1. BetaFlight SITL connectivity (MSP on port 5761)
  2. Mode ranges (ARM on AUX1, ANGLE on AUX2)
  3. Arming status and disable flags
  4. Motor output values
  5. RC channel values
  6. FDM receipt status (by checking motor output behavior)
"""

import socket
import struct
import sys
import time


def msp_send(sock, cmd, payload=b''):
    """Send MSP v1 command and return response payload."""
    size = len(payload)
    checksum = size ^ cmd
    for b in payload:
        checksum ^= b
    frame = b'$M<' + bytes([size, cmd]) + payload + bytes([checksum])
    sock.send(frame)
    time.sleep(0.15)

    resp = b''
    try:
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            resp += chunk
            if len(resp) >= 5 and resp[:3] == b'$M>':
                pkt_size = resp[3]
                expected = 6 + pkt_size
                if len(resp) >= expected:
                    break
            if len(resp) >= 5 and resp[:3] == b'$M!':
                break
    except socket.timeout:
        pass

    if len(resp) >= 5 and resp[:3] == b'$M>':
        return resp[5:5 + resp[3]]
    if len(resp) >= 5 and resp[:3] == b'$M!':
        return None  # Error response
    return b''


def check_mode_ranges(sock):
    """Query MSP_MODE_RANGES (34) to verify ARM and ANGLE are configured."""
    print("\n--- Mode Ranges (MSP 34) ---")
    data = msp_send(sock, 34)
    if not data:
        print("  ERROR: No response from MSP_MODE_RANGES")
        return

    # Each mode range is 4 bytes: boxId, auxChannel, rangeStart, rangeEnd
    count = len(data) // 4
    print(f"  {count} mode range slots:")

    MODE_NAMES = {0: "ARM", 1: "ANGLE", 2: "HORIZON", 3: "MAG", 5: "HEADFREE",
                  6: "HEADADJ", 26: "TURTLE", 27: "FAILSAFE", 28: "AIRMODE"}

    found_arm = False
    found_angle = False
    for i in range(count):
        offset = i * 4
        box_id, aux_ch, start_step, end_step = struct.unpack('BBBB', data[offset:offset + 4])
        if start_step == 0 and end_step == 0:
            continue  # Empty slot
        start_pwm = 900 + start_step * 25
        end_pwm = 900 + end_step * 25
        name = MODE_NAMES.get(box_id, f"MODE_{box_id}")
        print(f"  Slot {i}: {name} (boxId={box_id}) on AUX{aux_ch + 1} "
              f"range [{start_pwm}-{end_pwm}] PWM (steps {start_step}-{end_step})")
        if box_id == 0:
            found_arm = True
        if box_id == 1:
            found_angle = True

    if not found_arm:
        print("  WARNING: ARM mode not configured on any AUX channel!")
    if not found_angle:
        print("  WARNING: ANGLE mode not configured on any AUX channel!")


def check_arming_status(sock):
    """Query MSP_STATUS_EX (150) for arming flags."""
    print("\n--- Arming Status (MSP 150) ---")
    data = msp_send(sock, 150)
    if not data or len(data) < 20:
        print(f"  ERROR: MSP_STATUS_EX returned {len(data) if data else 0} bytes (need >=20)")
        return

    cycle_time = struct.unpack('<H', data[0:2])[0]
    i2c_errors = struct.unpack('<H', data[2:4])[0]
    sensors = struct.unpack('<H', data[4:6])[0]
    flight_mode = struct.unpack('<I', data[6:10])[0]
    config_profile = data[10]

    print(f"  Cycle time: {cycle_time} us")
    print(f"  I2C errors: {i2c_errors}")
    print(f"  Sensors: 0x{sensors:04x}")
    print(f"  Flight mode flags: 0x{flight_mode:08x}")
    print(f"  Config profile: {config_profile}")

    # Parse arming disable flags
    flag_names = ['NO_GYRO', 'FAILSAFE', 'RX_FAILSAFE', 'NOT_DISARMED',
                  'BOXFAILSAFE', 'RUNAWAY', 'CRASH', 'THROTTLE', 'ANGLE',
                  'BOOT_GRACE', 'NOPREARM', 'LOAD', 'CALIBRATING', 'CLI',
                  'CMS_MENU', 'BST', 'MSP', 'PARALYZE', 'GPS', 'RESC',
                  'DSHOT_TELEM', 'REBOOT_REQ', 'DSHOT_BB', 'ACC_CAL',
                  'MOTOR_PROTO', 'CRASHFLIP', 'ALTHOLD', 'POSHOLD', 'ARM_SWITCH']

    flags_byte_count_raw = data[15]
    flags_byte_count = flags_byte_count_raw & 0x0F
    offset = 16 + flags_byte_count
    if len(data) >= offset + 5:
        arming_count = data[offset]
        arming_flags = struct.unpack('<I', data[offset + 1:offset + 5])[0]
        active = [flag_names[i] for i in range(min(len(flag_names), 29))
                  if arming_flags & (1 << i)]
        armed = bool(flight_mode & (1 << 0))  # Bit 0 = armed
        print(f"  Armed: {armed}")
        print(f"  Arming disable flags: 0x{arming_flags:08x}")
        if active:
            print(f"  Active arming blocks: {', '.join(active)}")
        else:
            print("  No arming blocks (clear to arm)")
    else:
        print(f"  Status payload too short for arming flags ({len(data)} bytes)")


def check_motor_output(sock):
    """Query MSP_MOTOR (104) for current motor values."""
    print("\n--- Motor Output (MSP 104) ---")
    data = msp_send(sock, 104)
    if not data:
        print("  ERROR: No response from MSP_MOTOR")
        return

    motor_count = len(data) // 2
    motors = []
    for i in range(motor_count):
        val = struct.unpack('<H', data[i * 2:(i + 1) * 2])[0]
        motors.append(val)
        if i < 4:
            # Estimate what servo_packet value BF SITL would send
            servo_speed = val / 1000.0
            print(f"  Motor {i}: PWM={val}  (servo_packet: {servo_speed:.3f})")

    print(f"\n  NOTE: BetaFlight SITL sends motor_speed = PWM/1000.0 to Gazebo.")
    print(f"  The BetaflightPlugin expects [0.0, 1.0] but BF sends [{motors[0]/1000:.1f}, ...]")
    if any(m >= 1000 for m in motors[:4]):
        print("  WARNING: Motor values >= 1000 → servo_speed >= 1.0")
        print("  This maps to full RPM in BetaflightPlugin even when disarmed!")


def check_rc_channels(sock):
    """Query MSP_RC (105) for current RC channel values."""
    print("\n--- RC Channels (MSP 105) ---")
    data = msp_send(sock, 105)
    if not data:
        print("  ERROR: No response from MSP_RC")
        return

    ch_count = len(data) // 2
    ch_names = ['Roll', 'Pitch', 'Throttle', 'Yaw', 'AUX1(ARM)', 'AUX2(ANGLE)',
                'AUX3', 'AUX4']

    for i in range(min(ch_count, 8)):
        val = struct.unpack('<H', data[i * 2:(i + 1) * 2])[0]
        name = ch_names[i] if i < len(ch_names) else f'CH{i + 1}'
        print(f"  CH{i + 1} ({name}): {val}")

    # Check if RC is being received
    ch1 = struct.unpack('<H', data[0:2])[0]
    if ch1 == 0 or ch1 == 1500:
        print("\n  NOTE: If all channels show 0 or default values,")
        print("  BetaFlight is not receiving RC packets from drone_manage.py")


def check_motor_mapping():
    """Print the motor mapping analysis."""
    print("\n--- Motor Mapping Analysis ---")
    print("  BetaFlight SITL pwmCompleteMotorUpdate() remaps motors:")
    print("    pkt.motor_speed[0] = motorsPwm[1]  (BF Motor 1, Front-Right CCW)")
    print("    pkt.motor_speed[1] = motorsPwm[2]  (BF Motor 2, Rear-Left CCW)")
    print("    pkt.motor_speed[2] = motorsPwm[3]  (BF Motor 3, Front-Left CW)")
    print("    pkt.motor_speed[3] = motorsPwm[0]  (BF Motor 0, Rear-Right CW)")
    print()
    print("  BetaflightPlugin model.sdf maps (1:1 id→joint):")
    print("    rotor id=0 → rotor_0_joint (0.13,-0.22)  Front-Right CCW  gets pkt[0] = BF Motor 1 (FR CCW) ✓")
    print("    rotor id=1 → rotor_1_joint (-0.13,0.20)  Rear-Left CCW   gets pkt[1] = BF Motor 2 (RL CCW) ✓")
    print("    rotor id=2 → rotor_2_joint (0.13,0.22)   Front-Left CW   gets pkt[2] = BF Motor 3 (FL CW)  ✓")
    print("    rotor id=3 → rotor_3_joint (-0.13,-0.20) Rear-Right CW   gets pkt[3] = BF Motor 0 (RR CW)  ✓")
    print()
    print("  Result: Motor mapping is CORRECT after BF SITL remap.")


def main():
    print("=" * 60)
    print("Nanopilot BetaFlight SITL Diagnostic")
    print("=" * 60)

    # Connect to BetaFlight MSP
    print("\nConnecting to BetaFlight MSP on 127.0.0.1:5761...")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        s.connect(('127.0.0.1', 5761))
        time.sleep(0.5)
        # Drain initial data
        try:
            s.recv(4096)
        except socket.timeout:
            pass
        print("  Connected!")
    except (ConnectionRefusedError, socket.timeout) as e:
        print(f"  FAILED: {e}")
        print("  Is BetaFlight SITL running? Start it with: ./scripts/start.sh")
        sys.exit(1)

    check_mode_ranges(s)
    check_arming_status(s)
    check_motor_output(s)
    check_rc_channels(s)
    check_motor_mapping()

    s.close()

    # Check BetaFlight log for FDM/RC receipt
    print("\n--- BetaFlight Log Check ---")
    try:
        with open('logs/betaflight.log', 'r') as f:
            log = f.read()
        if '[SITL] new fdm' in log:
            print("  FDM received: YES (BetaFlight got FDM from Gazebo)")
        else:
            print("  FDM received: NO — BetaFlight never got FDM from Gazebo!")
            print("  This blocks motor output. Check:")
            print("    - Is GZ_SIM_SYSTEM_PLUGIN_PATH set correctly?")
            print("    - Does libBetaflightPlugin.dylib exist in the plugin build dir?")
            print("    - Run Gazebo with --verbose to check plugin loading")

        if '[SITL] new rc' in log:
            print("  RC received: YES (BetaFlight got RC packets from drone_manage)")
        else:
            print("  RC received: NO — BetaFlight never got RC packets!")
    except FileNotFoundError:
        print("  logs/betaflight.log not found")

    print("\n" + "=" * 60)
    print("Summary of issues to fix:")
    print("=" * 60)
    print("""
1. MOTOR VALUE RANGE: BF SITL sends motor_speed = PWM/1000 = [1.0, 2.0].
   Plugin expects [0.0, 1.0]. FIX APPLIED: BetaflightPlugin.cc now subtracts
   1.0 from each motor value before scaling by maxRpm.

2. MOTOR INDEX REMAP: BF SITL rotates motor indices in pwmCompleteMotorUpdate.
   After the remap, indices align correctly with model.sdf rotor positions.
   Status: CORRECT — no fix needed.

3. FDM FLOW: If BF log shows no "new fdm", the BetaflightPlugin isn't
   sending state to BF, which blocks motor output entirely.
""")


if __name__ == '__main__':
    main()
