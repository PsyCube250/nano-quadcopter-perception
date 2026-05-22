#!/usr/bin/env python3
"""
Bench-test drone roll control from an Xbox controller through MSP RC.

Default mapping:
- Left stick X, joystick axis 0 -> RC roll
- RB, joystick button 7 -> deadman arm switch

Release RB or stop moving the controller for the timeout period to send
disarm/neutral RC.
"""

import argparse
import math
import select
import struct
import sys
import time

from motor_lidar_speed_control_test import (
    FC_BAUD,
    FC_PORT,
    LOOP_HZ,
    MSP,
    clamp,
    disarm_rc,
    print_status,
    send_for_duration,
)
from xbox_controller_detect import (
    JS_EVENT_AXIS,
    JS_EVENT_BUTTON,
    JS_EVENT_INIT,
    JSIOCGAXES,
    JSIOCGBUTTONS,
    axis_map,
    button_map,
    choose_device,
    ioctl_u8,
    joystick_name,
    normalize_axis,
)


RC_CENTER = 1500
RC_MIN = 1000
RC_MAX = 2000
ARM_AUX = 1800
DISARM_AUX = 1000

DEFAULT_ROLL_AXIS = 0
DEFAULT_DEADMAN_BUTTON = 7
DEFAULT_MAX_ROLL_OFFSET = 150
DEFAULT_CONTROLLER_TIMEOUT_SEC = 2.0


class XboxJoystick:
    def __init__(self, device):
        self.device = choose_device(device)
        self.fd = open(self.device, "rb", buffering=0)
        self.axes = axis_map(self.fd, ioctl_u8(self.fd, JSIOCGAXES))
        self.buttons = button_map(self.fd, ioctl_u8(self.fd, JSIOCGBUTTONS))
        self.axis_values = [0] * len(self.axes)
        self.button_values = [0] * len(self.buttons)
        self.name = joystick_name(self.fd)
        self.last_event_time = time.time()
        self.connected = True

    def close(self):
        self.fd.close()

    def poll(self):
        while True:
            readable, _, _ = select.select([self.fd], [], [], 0)
            if not readable:
                return

            data = self.fd.read(8)
            if len(data) != 8:
                self.connected = False
                return

            _event_time, value, event_type, number = struct.unpack("IhBB", data)
            event_type &= ~JS_EVENT_INIT

            if event_type == JS_EVENT_AXIS and number < len(self.axis_values):
                self.axis_values[number] = value
                self.last_event_time = time.time()
            elif event_type == JS_EVENT_BUTTON and number < len(self.button_values):
                self.button_values[number] = value
                self.last_event_time = time.time()

    def axis_value(self, axis):
        if axis < 0 or axis >= len(self.axis_values):
            return 0
        return self.axis_values[axis]

    def button_pressed(self, button):
        if button < 0 or button >= len(self.button_values):
            return False
        return bool(self.button_values[button])

    def has_recent_events(self, timeout_sec):
        if timeout_sec <= 0:
            return True
        return time.time() - self.last_event_time <= timeout_sec


def rc_channels(roll, pitch, throttle, yaw, arm):
    aux1 = ARM_AUX if arm else DISARM_AUX
    safe_throttle = throttle if arm else 1000
    safe_roll = roll if arm else RC_CENTER
    return [safe_roll, pitch, safe_throttle, yaw, aux1, 1000, 1000, 1000]


def scaled_axis(raw_value, deadzone):
    value = normalize_axis(raw_value)

    if abs(value) <= deadzone:
        return 0.0

    scaled = (abs(value) - deadzone) / (1.0 - deadzone)
    return math.copysign(clamp(scaled, 0.0, 1.0), value)


def roll_from_controller(raw_value, args):
    norm = scaled_axis(raw_value, args.deadzone)
    roll = RC_CENTER + args.roll_sign * norm * args.max_roll_offset
    return int(round(clamp(roll, RC_MIN, RC_MAX))), norm


def validate_args(args):
    if not 0.0 <= args.deadzone < 0.9:
        raise SystemExit("--deadzone must be between 0.0 and 0.9")

    if not 0 <= args.max_roll_offset <= 500:
        raise SystemExit("--max-roll-offset must be between 0 and 500")

    if not RC_MIN <= args.pitch <= RC_MAX:
        raise SystemExit("--pitch must be between 1000 and 2000")

    if not RC_MIN <= args.yaw <= RC_MAX:
        raise SystemExit("--yaw must be between 1000 and 2000")

    if not RC_MIN <= args.throttle <= RC_MAX:
        raise SystemExit("--throttle must be between 1000 and 2000")

    if args.throttle > 1150 and not args.allow_flight_throttle:
        raise SystemExit(
            "Refusing throttle above 1150 without --allow-flight-throttle. "
            "Use low throttle first with propellers removed."
        )

    if args.run_sec <= 0:
        raise SystemExit("--run-sec must be greater than zero")

    if args.warmup_sec < 0 or args.disarm_sec < 0:
        raise SystemExit("--warmup-sec and --disarm-sec cannot be negative")

    if args.controller_timeout_sec < 0:
        raise SystemExit("--controller-timeout-sec cannot be negative")

    if args.print_hz <= 0:
        raise SystemExit("--print-hz must be greater than zero")


def confirm_or_exit(args, joystick):
    print("=== XBOX ROLL CONTROL MSP BENCH TEST ===")
    print("PROPELLERS MUST BE REMOVED.")
    print("Betaflight Configurator must be disconnected.")
    print("This sends MSP RC roll commands, not direct motor commands.")
    print("")
    print(f"Controller:        {joystick.device} - {joystick.name}")
    print(f"Axes:              {', '.join(f'{i}:{name}' for i, name in enumerate(joystick.axes))}")
    print(f"Buttons:           {', '.join(f'{i}:{name}' for i, name in enumerate(joystick.buttons))}")
    print(f"Roll axis:         {args.roll_axis}")
    print(f"Deadman button:    {args.deadman_button} (hold to arm/send roll)")
    print(f"Max roll offset:   +/-{args.max_roll_offset}")
    print(f"Throttle:          {args.throttle}")
    print(f"Controller timeout:{args.controller_timeout_sec:.2f}s")
    print(f"Dry run:           {'yes' if args.dry_run else 'no'}")
    print("")

    if args.dry_run:
        print("Dry run mode: no flight-controller RC commands will be sent.")
        return

    expected = "PROPS REMOVED"
    print(f'Type "{expected}" to run this RC test.')
    answer = input("> ").strip()

    if answer != expected:
        raise SystemExit("Confirmation did not match. Aborting.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Use an Xbox controller left stick to send MSP RC roll commands."
    )
    parser.add_argument("--device", help="Joystick device, for example /dev/input/js0")
    parser.add_argument("--fc-port", default=FC_PORT)
    parser.add_argument("--fc-baud", type=int, default=FC_BAUD)
    parser.add_argument("--roll-axis", type=int, default=DEFAULT_ROLL_AXIS)
    parser.add_argument(
        "--roll-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="Use -1 if stick-right makes the drone roll the wrong way.",
    )
    parser.add_argument("--deadzone", type=float, default=0.08)
    parser.add_argument("--deadman-button", type=int, default=DEFAULT_DEADMAN_BUTTON)
    parser.add_argument("--max-roll-offset", type=int, default=DEFAULT_MAX_ROLL_OFFSET)
    parser.add_argument("--pitch", type=int, default=RC_CENTER)
    parser.add_argument("--yaw", type=int, default=RC_CENTER)
    parser.add_argument("--throttle", type=int, default=1000)
    parser.add_argument("--warmup-sec", type=float, default=3.0)
    parser.add_argument("--run-sec", type=float, default=20.0)
    parser.add_argument("--disarm-sec", type=float, default=2.0)
    parser.add_argument("--print-hz", type=float, default=5.0)
    parser.add_argument(
        "--controller-timeout-sec",
        type=float,
        default=DEFAULT_CONTROLLER_TIMEOUT_SEC,
        help="Disarm if no controller events arrive for this long. Use 0 to disable.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-flight-throttle", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    joystick = None
    msp = None

    try:
        joystick = XboxJoystick(args.device)
        confirm_or_exit(args, joystick)

        if not args.dry_run:
            msp = MSP(args.fc_port, args.fc_baud)
            send_for_duration(msp, disarm_rc, args.warmup_sec, "Warmup disarm RC")
            print_status(msp, "Status after warmup")

        print(f"Running Xbox roll control for {args.run_sec:.1f}s")
        print("Hold RB/deadman to arm. Left stick X controls roll. Release RB to disarm.")

        end = time.time() + args.run_sec
        last_print = 0.0
        print_period = 1.0 / args.print_hz
        loop_period = 1.0 / LOOP_HZ

        while time.time() < end:
            joystick.poll()

            raw_roll = joystick.axis_value(args.roll_axis)
            roll, roll_norm = roll_from_controller(raw_roll, args)
            deadman = joystick.button_pressed(args.deadman_button)
            recent = joystick.has_recent_events(args.controller_timeout_sec)
            arm = joystick.connected and deadman and recent

            channels = rc_channels(
                roll,
                args.pitch,
                args.throttle,
                args.yaw,
                arm=arm,
            )

            if msp is not None:
                msp.send_rc(channels)

            now = time.time()
            if now - last_print >= print_period:
                last_print = now
                if not joystick.connected:
                    state = "DISCONNECTED"
                elif not recent:
                    state = "TIMEOUT"
                elif deadman:
                    state = "ARMED"
                else:
                    state = "DISARMED"

                print(
                    f"state={state:<12} deadman={int(deadman)} "
                    f"axis_raw={raw_roll:6d} axis={roll_norm:+.2f} "
                    f"rc=roll:{channels[0]} pitch:{channels[1]} "
                    f"thr:{channels[2]} yaw:{channels[3]} aux1:{channels[4]}"
                )

            time.sleep(loop_period)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except OSError as exc:
        print(f"Controller or serial error: {exc}", file=sys.stderr)
        return 1
    finally:
        if msp is not None:
            print("Disarming now.")
            try:
                send_for_duration(msp, disarm_rc, args.disarm_sec, "Disarm")
                print_status(msp, "Final status")
            finally:
                msp.close()

        if joystick is not None:
            joystick.close()

        print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
