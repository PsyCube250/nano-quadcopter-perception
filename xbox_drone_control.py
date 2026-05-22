#!/usr/bin/env python3
"""
Control Betaflight RC channels from an Xbox controller through MSP.

Default mapping:
- Left stick X, axis 0 -> yaw
- Left stick Y, axis 1 -> throttle around hover throttle
- Right stick X, axis 2 -> roll
- Right stick Y, axis 3 -> pitch
- RB, button 7 -> arm/disarm toggle, only arms while throttle is low
- B, button 1 -> emergency disarm

This sends MSP_SET_RAW_RC packets to the flight controller. It does not command
motors directly.
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

DEFAULT_YAW_AXIS = 0
DEFAULT_THROTTLE_AXIS = 1
DEFAULT_ROLL_AXIS = 2
DEFAULT_PITCH_AXIS = 3
DEFAULT_DEADMAN_BUTTON = -1
DEFAULT_ARM_BUTTON = 7
DEFAULT_KILL_BUTTON = 1

DEFAULT_HOVER_THROTTLE = 1300
DEFAULT_THROTTLE_OFFSET = 300
DEFAULT_MAX_ANGLE_OFFSET = 180
DEFAULT_MAX_YAW_OFFSET = 160
DEFAULT_CONTROLLER_TIMEOUT_SEC = 0.0


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


def scaled_axis(raw_value, deadzone):
    value = normalize_axis(raw_value)

    if abs(value) <= deadzone:
        return 0.0

    scaled = (abs(value) - deadzone) / (1.0 - deadzone)
    return math.copysign(clamp(scaled, 0.0, 1.0), value)


def centered_rc(raw_value, deadzone, sign, max_offset):
    norm = scaled_axis(raw_value, deadzone)
    command_norm = sign * norm
    rc = RC_CENTER + command_norm * max_offset
    return int(round(clamp(rc, RC_MIN, RC_MAX))), command_norm


def throttle_rc(raw_value, args):
    norm = scaled_axis(raw_value, args.deadzone)
    command_norm = args.throttle_sign * norm
    throttle = args.hover_throttle + command_norm * args.throttle_offset
    throttle = int(round(clamp(throttle, args.min_throttle, args.max_throttle)))
    return throttle, command_norm


def rc_channels(roll, pitch, throttle, yaw, armed):
    aux1 = ARM_AUX if armed else DISARM_AUX

    if not armed:
        return disarm_rc()

    return [roll, pitch, throttle, yaw, aux1, 1000, 1000, 1000]


def validate_axis_button(args, joystick):
    for label, axis in (
        ("--yaw-axis", args.yaw_axis),
        ("--throttle-axis", args.throttle_axis),
        ("--roll-axis", args.roll_axis),
        ("--pitch-axis", args.pitch_axis),
    ):
        if axis < 0 or axis >= len(joystick.axes):
            raise SystemExit(f"{label} {axis} is not present on {joystick.device}")

    for label, button in (
        ("--arm-button", args.arm_button),
        ("--kill-button", args.kill_button),
    ):
        if button < 0 or button >= len(joystick.buttons):
            raise SystemExit(f"{label} {button} is not present on {joystick.device}")

    if args.deadman_button >= len(joystick.buttons):
        raise SystemExit(f"--deadman-button {args.deadman_button} is not present on {joystick.device}")


def validate_args(args):
    if not 0.0 <= args.deadzone < 0.9:
        raise SystemExit("--deadzone must be between 0.0 and 0.9")

    if not RC_MIN <= args.min_throttle <= RC_MAX:
        raise SystemExit("--min-throttle must be between 1000 and 2000")

    if not RC_MIN <= args.max_throttle <= RC_MAX:
        raise SystemExit("--max-throttle must be between 1000 and 2000")

    if args.min_throttle >= args.max_throttle:
        raise SystemExit("--min-throttle must be less than --max-throttle")

    if not args.min_throttle <= args.hover_throttle <= args.max_throttle:
        raise SystemExit("--hover-throttle must be between min and max throttle")

    if not 0 <= args.throttle_offset <= 600:
        raise SystemExit("--throttle-offset must be between 0 and 600")

    if args.hover_throttle - args.throttle_offset > args.arm_throttle_max:
        raise SystemExit(
            "With this hover throttle and throttle offset, full stick down cannot "
            "reach --arm-throttle-max. Increase --throttle-offset or lower "
            "--hover-throttle."
        )

    if not 0 <= args.max_roll_offset <= 500:
        raise SystemExit("--max-roll-offset must be between 0 and 500")

    if not 0 <= args.max_pitch_offset <= 500:
        raise SystemExit("--max-pitch-offset must be between 0 and 500")

    if not 0 <= args.max_yaw_offset <= 500:
        raise SystemExit("--max-yaw-offset must be between 0 and 500")

    if not RC_MIN <= args.arm_throttle_max <= RC_MAX:
        raise SystemExit("--arm-throttle-max must be between 1000 and 2000")

    if args.max_throttle > 1150 and not args.allow_flight_throttle and not args.dry_run:
        raise SystemExit(
            "Refusing flight-capable throttle without --allow-flight-throttle. "
            "Run dry first, then bench-test with propellers removed."
        )

    if args.warmup_sec < 0 or args.disarm_sec < 0:
        raise SystemExit("--warmup-sec and --disarm-sec cannot be negative")

    if args.run_sec < 0:
        raise SystemExit("--run-sec cannot be negative")

    if args.controller_timeout_sec < 0:
        raise SystemExit("--controller-timeout-sec cannot be negative")

    if args.print_hz <= 0:
        raise SystemExit("--print-hz must be greater than zero")


def confirm_or_exit(args, joystick):
    print("=== XBOX DRONE MSP RC CONTROL ===")
    print("PROPELLERS MUST BE REMOVED for bench testing.")
    print("Betaflight Configurator must be disconnected.")
    print("This sends RC channels through MSP_SET_RAW_RC.")
    print("")
    print(f"Controller:         {joystick.device} - {joystick.name}")
    print(f"Axes:               {', '.join(f'{i}:{name}' for i, name in enumerate(joystick.axes))}")
    print(f"Buttons:            {', '.join(f'{i}:{name}' for i, name in enumerate(joystick.buttons))}")
    print(f"Yaw axis:           {args.yaw_axis}")
    print(f"Throttle axis:      {args.throttle_axis} (center -> {args.hover_throttle})")
    print(f"Roll axis:          {args.roll_axis}")
    print(f"Pitch axis:         {args.pitch_axis}")
    deadman_label = "disabled" if args.deadman_button < 0 else f"{args.deadman_button} (hold)"
    print(f"Deadman button:     {deadman_label}")
    print(f"Arm button:         {args.arm_button} (press to toggle, arm requires low throttle)")
    print(f"Emergency disarm:   {args.kill_button}")
    print(f"Throttle range:     {args.min_throttle}..{args.max_throttle}")
    print(f"Hover throttle:     {args.hover_throttle}")
    print(f"Throttle offset:    +/-{args.throttle_offset}")
    print(f"Roll/Pitch offsets: +/-{args.max_roll_offset}/+/-{args.max_pitch_offset}")
    print(f"Yaw offset:         +/-{args.max_yaw_offset}")
    print(f"Controller timeout: {args.controller_timeout_sec:.2f}s")
    print(f"Dry run:            {'yes' if args.dry_run else 'no'}")
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
        description=(
            "Use an Xbox controller as an MSP RC transmitter. Left stick Y is "
            "centered around hover throttle."
        )
    )
    parser.add_argument("--device", help="Joystick device, for example /dev/input/js0")
    parser.add_argument("--fc-port", default=FC_PORT)
    parser.add_argument("--fc-baud", type=int, default=FC_BAUD)
    parser.add_argument("--yaw-axis", type=int, default=DEFAULT_YAW_AXIS)
    parser.add_argument("--throttle-axis", type=int, default=DEFAULT_THROTTLE_AXIS)
    parser.add_argument("--roll-axis", type=int, default=DEFAULT_ROLL_AXIS)
    parser.add_argument("--pitch-axis", type=int, default=DEFAULT_PITCH_AXIS)
    parser.add_argument(
        "--deadman-button",
        type=int,
        default=DEFAULT_DEADMAN_BUTTON,
        help="Optional hold-to-run button. Default -1 disables it.",
    )
    parser.add_argument("--arm-button", type=int, default=DEFAULT_ARM_BUTTON)
    parser.add_argument("--kill-button", type=int, default=DEFAULT_KILL_BUTTON)
    parser.add_argument("--deadzone", type=float, default=0.08)
    parser.add_argument("--hover-throttle", type=int, default=DEFAULT_HOVER_THROTTLE)
    parser.add_argument("--throttle-offset", type=int, default=DEFAULT_THROTTLE_OFFSET)
    parser.add_argument("--min-throttle", type=int, default=1000)
    parser.add_argument("--max-throttle", type=int, default=1600)
    parser.add_argument("--arm-throttle-max", type=int, default=1050)
    parser.add_argument("--max-roll-offset", type=int, default=DEFAULT_MAX_ANGLE_OFFSET)
    parser.add_argument("--max-pitch-offset", type=int, default=DEFAULT_MAX_ANGLE_OFFSET)
    parser.add_argument("--max-yaw-offset", type=int, default=DEFAULT_MAX_YAW_OFFSET)
    parser.add_argument(
        "--roll-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="Use -1 if right stick right rolls the wrong way.",
    )
    parser.add_argument(
        "--pitch-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=-1.0,
        help="Use 1 if right stick up pitches the wrong way.",
    )
    parser.add_argument(
        "--yaw-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="Use -1 if left stick right yaws the wrong way.",
    )
    parser.add_argument(
        "--throttle-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=-1.0,
        help="Default makes left stick up increase throttle.",
    )
    parser.add_argument("--warmup-sec", type=float, default=3.0)
    parser.add_argument(
        "--run-sec",
        type=float,
        default=0.0,
        help="Seconds to run. 0 means run until Ctrl+C.",
    )
    parser.add_argument("--disarm-sec", type=float, default=2.0)
    parser.add_argument("--print-hz", type=float, default=5.0)
    parser.add_argument(
        "--controller-timeout-sec",
        type=float,
        default=DEFAULT_CONTROLLER_TIMEOUT_SEC,
        help=(
            "Disarm if no controller events arrive for this long. Default 0 disables "
            "this because joystick devices may be quiet while a stick is held steady."
        ),
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
    armed = False
    previous_arm_button = False

    try:
        joystick = XboxJoystick(args.device)
        validate_axis_button(args, joystick)
        confirm_or_exit(args, joystick)

        if not args.dry_run:
            msp = MSP(args.fc_port, args.fc_baud)
            send_for_duration(msp, disarm_rc, args.warmup_sec, "Warmup disarm RC")
            print_status(msp, "Status after warmup")

        if args.run_sec > 0:
            print(f"Running Xbox drone control for {args.run_sec:.1f}s")
            end_time = time.time() + args.run_sec
        else:
            print("Running Xbox drone control until Ctrl+C")
            end_time = None

        print("Hold left stick down, then press RB/arm once to arm.")
        print("Press RB again or press B to disarm. Center left stick sends hover throttle after arming.")

        joystick.poll()
        previous_arm_button = joystick.button_pressed(args.arm_button)

        last_print = 0.0
        print_period = 1.0 / args.print_hz
        loop_period = 1.0 / LOOP_HZ

        while end_time is None or time.time() < end_time:
            joystick.poll()

            yaw, yaw_norm = centered_rc(
                joystick.axis_value(args.yaw_axis),
                args.deadzone,
                args.yaw_sign,
                args.max_yaw_offset,
            )
            throttle, throttle_norm = throttle_rc(
                joystick.axis_value(args.throttle_axis),
                args,
            )
            roll, roll_norm = centered_rc(
                joystick.axis_value(args.roll_axis),
                args.deadzone,
                args.roll_sign,
                args.max_roll_offset,
            )
            pitch, pitch_norm = centered_rc(
                joystick.axis_value(args.pitch_axis),
                args.deadzone,
                args.pitch_sign,
                args.max_pitch_offset,
            )

            deadman = args.deadman_button < 0 or joystick.button_pressed(args.deadman_button)
            arm_button = joystick.button_pressed(args.arm_button)
            arm_pressed = arm_button and not previous_arm_button
            previous_arm_button = arm_button
            kill_button = joystick.button_pressed(args.kill_button)
            recent = joystick.has_recent_events(args.controller_timeout_sec)
            link_ok = joystick.connected and recent

            if not link_ok or not deadman or kill_button:
                armed = False
            elif arm_pressed:
                if armed:
                    armed = False
                elif throttle <= args.arm_throttle_max:
                    armed = True

            channels = rc_channels(roll, pitch, throttle, yaw, armed=armed)

            if msp is not None:
                msp.send_rc(channels)

            now = time.time()
            if now - last_print >= print_period:
                last_print = now

                if not joystick.connected:
                    state = "DISCONNECTED"
                elif not recent:
                    state = "TIMEOUT"
                elif kill_button:
                    state = "KILL"
                elif not deadman:
                    state = "DEADMAN_OFF"
                elif not armed and arm_pressed and throttle > args.arm_throttle_max:
                    state = "THR_TOO_HIGH"
                elif armed:
                    state = "ARMED"
                else:
                    state = "READY"

                deadman_display = "-" if args.deadman_button < 0 else str(int(deadman))
                print(
                    f"state={state:<12} deadman={deadman_display} arm_btn={int(arm_button)} "
                    f"kill={int(kill_button)} "
                    f"axes yaw:{yaw_norm:+.2f} thr:{throttle_norm:+.2f} "
                    f"roll:{roll_norm:+.2f} pitch:{pitch_norm:+.2f} "
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
