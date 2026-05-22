import argparse
import struct
import time

import serial

from lidar_stl27l import LIDAR_BAUD, LIDAR_PORT, STL27LReader


FC_PORT = "/dev/ttyTHS1"
FC_BAUD = 115200
LOOP_HZ = 50

MSP_STATUS = 101
MSP_SET_RAW_RC = 200

DEFAULT_THRESHOLD_MM = 50
DEFAULT_NORMAL_THROTTLE = 1075
DEFAULT_SLOW_THROTTLE = 1000
DEFAULT_THROTTLE_STEP = 2

ARMING_FLAGS = {
    0: "NOGYRO",
    1: "FAILSAFE",
    2: "RXLOSS",
    3: "BADVIBECOUNT",
    4: "BOXFAILSAFE",
    5: "RUNAWAY_TAKEOFF",
    6: "CRASH_DETECTED",
    7: "THROTTLE",
    8: "ANGLE",
    9: "BOOTGRACE",
    10: "NOPREARM",
    11: "LOAD",
    12: "CALIB",
    13: "CLI",
    14: "CMS_MENU",
    15: "BST",
    16: "MSP",
    17: "PARALYZE",
    18: "GPS",
    19: "RESC",
    20: "RPMFILTER",
    21: "REBOOT_REQUIRED",
    22: "DSHOT_TELEM",
    23: "ACC_CALIB",
    24: "MOTOR_PROTOCOL",
    25: "ARM_SWITCH",
}


class MSP:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=1)
        self.ser.reset_input_buffer()

    def _send(self, cmd, data=bytearray()):
        size = len(data)
        checksum = size ^ cmd

        for b in data:
            checksum ^= b

        packet = bytearray(b"$M<")
        packet.append(size)
        packet.append(cmd)
        packet.extend(data)
        packet.append(checksum)
        self.ser.write(packet)

    def _read_response(self):
        timeout = time.time() + 1.0

        while time.time() < timeout:
            b = self.ser.read(1)

            if b != b"$":
                continue

            if self.ser.read(1) != b"M":
                continue

            direction = self.ser.read(1)

            if direction not in (b">", b"!"):
                continue

            size_b = self.ser.read(1)
            cmd_b = self.ser.read(1)

            if not size_b or not cmd_b:
                continue

            size = size_b[0]
            cmd = cmd_b[0]
            data = self.ser.read(size)
            self.ser.read(1)
            return cmd, data

        return None, None

    def get_status(self):
        self.ser.reset_input_buffer()
        time.sleep(0.05)
        self._send(MSP_STATUS)
        _cmd, data = self._read_response()

        if data and len(data) >= 11:
            return struct.unpack_from("<I", data, 6)[0]

        return None

    def send_rc(self, channels):
        data = bytearray()

        for ch in channels:
            data.extend(struct.pack("<H", int(ch)))

        self._send(MSP_SET_RAW_RC, data)

    def close(self):
        self.ser.close()


def decode_flags(flags):
    if flags is None:
        return []

    return [
        name for bit, name in ARMING_FLAGS.items()
        if flags & (1 << bit)
    ]


def disarm_rc():
    return [1500, 1500, 1000, 1500, 1000, 1000, 1000, 1000]


def arm_rc():
    return [1500, 1500, 1000, 1500, 1800, 1000, 1000, 1000]


def throttle_rc(throttle):
    return [1500, 1500, throttle, 1500, 1800, 1000, 1000, 1000]


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def step_toward(current, target, step):
    if current < target:
        return min(current + step, target)

    if current > target:
        return max(current - step, target)

    return current


def format_point(point):
    if not point:
        return "no valid point"

    return (
        f"{point['distance_mm']} mm at "
        f"{point['angle_deg']:.1f} deg conf={point['confidence']}"
    )


def print_status(msp, label):
    flags = msp.get_status()
    print(f"\n--- {label} ---")

    if flags is None:
        print("No MSP_STATUS response.")
        return

    active = decode_flags(flags)
    print(f"Raw flags: 0x{flags:08X}")

    if active:
        print(f"Blocking flags: {', '.join(active)}")
    else:
        print("Blocking flags: none")


def send_for_duration(msp, rc_fn, duration, label):
    print(f"{label} for {duration:.1f}s")
    end = time.time() + duration
    count = 0

    while time.time() < end:
        msp.send_rc(rc_fn())
        count += 1
        time.sleep(1.0 / LOOP_HZ)

    print(f"  sent {count} RC packets")


def confirm_or_exit():
    expected = "PROPS REMOVED"
    print(f'Type "{expected}" to arm and run this bench test.')
    answer = input("> ").strip()

    if answer != expected:
        raise SystemExit("Confirmation did not match. Aborting.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Low-throttle motor test with LiDAR-based smooth throttle reduction."
        )
    )
    parser.add_argument("--fc-port", default=FC_PORT)
    parser.add_argument("--fc-baud", type=int, default=FC_BAUD)
    parser.add_argument("--lidar-port", default=LIDAR_PORT)
    parser.add_argument("--lidar-baud", type=int, default=LIDAR_BAUD)
    parser.add_argument("--threshold-mm", type=int, default=DEFAULT_THRESHOLD_MM)
    parser.add_argument("--normal-throttle", type=int, default=DEFAULT_NORMAL_THROTTLE)
    parser.add_argument("--slow-throttle", type=int, default=DEFAULT_SLOW_THROTTLE)
    parser.add_argument("--throttle-step", type=int, default=DEFAULT_THROTTLE_STEP)
    parser.add_argument("--warmup-sec", type=float, default=8.0)
    parser.add_argument("--arm-sec", type=float, default=2.0)
    parser.add_argument("--run-sec", type=float, default=5.0)
    parser.add_argument("--disarm-sec", type=float, default=2.0)
    parser.add_argument(
        "--front-only",
        action="store_true",
        help="React only to obstacles in the +/-30 degree front sector.",
    )
    args = parser.parse_args()

    if args.slow_throttle > args.normal_throttle:
        raise SystemExit("--slow-throttle must be <= --normal-throttle")

    if args.slow_throttle < 1000 or args.normal_throttle > 1150:
        raise SystemExit("Keep this bench test between 1000 and 1150 throttle.")

    print("=== LiDAR LOW-THROTTLE MOTOR BENCH TEST ===")
    print("PROPELLERS MUST BE REMOVED.")
    print("Betaflight Configurator must be disconnected.")
    print("This changes only the RC throttle channel; it does not command motors directly.")
    print(f"FC:    {args.fc_port} @ {args.fc_baud}")
    print(f"LiDAR: {args.lidar_port} @ {args.lidar_baud}")
    print(f"Obstacle threshold: {args.threshold_mm} mm")
    print(f"Normal throttle:    {args.normal_throttle}")
    print(f"Slow throttle:      {args.slow_throttle}")
    print(f"Throttle step:      {args.throttle_step} per {1.0 / LOOP_HZ:.3f}s")
    print(f"Detection sector:   {'front +/-30 deg' if args.front_only else 'all angles'}")
    print("")

    confirm_or_exit()

    msp = MSP(args.fc_port, args.fc_baud)
    lidar = STL27LReader(args.lidar_port, args.lidar_baud, point_max_age_sec=0.25)
    lidar.start()

    current_throttle = args.normal_throttle
    last_print = 0

    try:
        send_for_duration(msp, disarm_rc, args.warmup_sec, "Warmup disarm RC")
        print_status(msp, "Status after warmup")

        send_for_duration(msp, arm_rc, args.arm_sec, "Arming")
        print_status(msp, "Status after arm")

        print(f"Running LiDAR throttle control for {args.run_sec:.1f}s")
        end = time.time() + args.run_sec

        while time.time() < end:
            nearest = lidar.get_nearest(front_only=args.front_only)

            if nearest and nearest["distance_mm"] <= args.threshold_mm:
                target_throttle = args.slow_throttle
                state = "TOO_CLOSE"
            else:
                target_throttle = args.normal_throttle
                state = "NORMAL"

            current_throttle = step_toward(
                current_throttle,
                target_throttle,
                max(1, args.throttle_step),
            )
            current_throttle = clamp(
                current_throttle,
                args.slow_throttle,
                args.normal_throttle,
            )

            msp.send_rc(throttle_rc(current_throttle))

            now = time.time()
            if now - last_print >= 0.25:
                last_print = now
                print(
                    f"state={state:<9} throttle={current_throttle} "
                    f"target={target_throttle} nearest={format_point(nearest)}"
                )

            time.sleep(1.0 / LOOP_HZ)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        print("Disarming now.")
        try:
            send_for_duration(msp, disarm_rc, args.disarm_sec, "Disarm")
            print_status(msp, "Final status")
        finally:
            lidar.close()
            msp.close()
            print("Done.")


if __name__ == "__main__":
    main()
