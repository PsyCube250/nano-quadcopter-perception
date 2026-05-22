import argparse
import struct
import time

import serial


FC_PORT = "/dev/ttyTHS1"
FC_BAUD = 115200
LOOP_HZ = 50

MSP_STATUS = 101
MSP_SET_RAW_RC = 200

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


def neutral_rc():
    return disarm_rc()


def arm_rc():
    return [1500, 1500, 1000, 1500, 1800, 1000, 1000, 1000]


def spin_rc(throttle):
    return [1500, 1500, throttle, 1500, 1800, 1000, 1000, 1000]


def send_loop(msp, channels_fn, duration, label):
    print(f"{label} for {duration:.1f}s")
    end = time.time() + duration
    count = 0

    while time.time() < end:
        msp.send_rc(channels_fn())
        count += 1
        time.sleep(1.0 / LOOP_HZ)

    print(f"  sent {count} RC packets")


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


def main():
    parser = argparse.ArgumentParser(
        description="Short low-throttle MSP motor bench test."
    )
    parser.add_argument("--fc-port", default=FC_PORT)
    parser.add_argument("--fc-baud", type=int, default=FC_BAUD)
    parser.add_argument("--warmup-sec", type=float, default=8.0)
    parser.add_argument("--arm-sec", type=float, default=2.0)
    parser.add_argument("--spin-sec", type=float, default=2.0)
    parser.add_argument("--throttle", type=int, default=1075)
    parser.add_argument("--disarm-sec", type=float, default=2.0)
    args = parser.parse_args()

    print("=== LOW THROTTLE MOTOR BENCH TEST ===")
    print("PROPELLERS MUST BE REMOVED.")
    print("Betaflight Configurator must be disconnected.")
    print(f"FC port: {args.fc_port} @ {args.fc_baud}")
    print(f"Throttle during spin: {args.throttle}")
    print("")

    msp = MSP(args.fc_port, args.fc_baud)

    try:
        send_loop(msp, neutral_rc, args.warmup_sec, "Warmup neutral/disarm RC")
        print_status(msp, "Status after warmup")

        send_loop(msp, arm_rc, args.arm_sec, "Arming")
        print_status(msp, "Status after arm")

        send_loop(
            msp,
            lambda: spin_rc(args.throttle),
            args.spin_sec,
            "Low throttle spin",
        )

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        print("Disarming now.")
        try:
            send_loop(msp, disarm_rc, args.disarm_sec, "Disarm")
            print_status(msp, "Final status")
        finally:
            msp.close()
            print("Done.")


if __name__ == "__main__":
    main()
