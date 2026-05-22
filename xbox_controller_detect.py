#!/usr/bin/env python3
"""
Read an Xbox controller from Linux's joystick API and print live control events.

This uses /dev/input/js* directly, so it does not need pygame or evdev.
"""

import argparse
import array
import fcntl
import glob
import os
import select
import struct
import sys
import time


JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80

JSIOCGAXES = 0x80016A11
JSIOCGBUTTONS = 0x80016A12
JSIOCGNAME_BASE = 0x80006A13
JSIOCGAXMAP = 0x80406A32
JSIOCGBTNMAP = 0x80406A34

AXIS_NAMES = {
    0x00: "ABS_X / left_stick_x",
    0x01: "ABS_Y / left_stick_y",
    0x02: "ABS_Z",
    0x03: "ABS_RX / right_stick_x",
    0x04: "ABS_RY / right_stick_y",
    0x05: "ABS_RZ",
    0x06: "ABS_THROTTLE",
    0x07: "ABS_RUDDER",
    0x08: "ABS_WHEEL",
    0x09: "ABS_GAS",
    0x0A: "ABS_BRAKE",
    0x10: "ABS_HAT0X / dpad_x",
    0x11: "ABS_HAT0Y / dpad_y",
}

BUTTON_NAMES = {
    0x130: "BTN_SOUTH / A",
    0x131: "BTN_EAST / B",
    0x132: "BTN_C",
    0x133: "BTN_NORTH / Y",
    0x134: "BTN_WEST / X",
    0x135: "BTN_Z",
    0x136: "BTN_TL / LB",
    0x137: "BTN_TR / RB",
    0x138: "BTN_TL2",
    0x139: "BTN_TR2",
    0x13A: "BTN_SELECT / View",
    0x13B: "BTN_START / Menu",
    0x13C: "BTN_MODE / Xbox",
    0x13D: "BTN_THUMBL / Left Stick Click",
    0x13E: "BTN_THUMBR / Right Stick Click",
}


def ioctl_u8(fd, request):
    buf = array.array("B", [0])
    fcntl.ioctl(fd, request, buf, True)
    return buf[0]


def joystick_name(fd):
    buf = array.array("B", [0] * 128)
    try:
        fcntl.ioctl(fd, JSIOCGNAME_BASE + (len(buf) << 16), buf, True)
    except OSError:
        return "Unknown joystick"
    raw = buf.tobytes().split(b"\x00", 1)[0]
    return raw.decode("utf-8", errors="replace")


def axis_map(fd, count):
    buf = array.array("B", [0] * 64)
    try:
        fcntl.ioctl(fd, JSIOCGAXMAP, buf, True)
    except OSError:
        return [f"axis_{i}" for i in range(count)]
    return [AXIS_NAMES.get(code, f"axis_{i}_code_{code}") for i, code in enumerate(buf[:count])]


def button_map(fd, count):
    buf = array.array("H", [0] * 200)
    try:
        fcntl.ioctl(fd, JSIOCGBTNMAP, buf, True)
    except OSError:
        return [f"button_{i}" for i in range(count)]
    return [BUTTON_NAMES.get(code, f"button_{i}_code_{code}") for i, code in enumerate(buf[:count])]


def normalize_axis(value):
    if value < 0:
        return value / 32768.0
    return value / 32767.0


def choose_device(device):
    if device:
        return device

    devices = sorted(glob.glob("/dev/input/js*"))
    if not devices:
        raise FileNotFoundError("No /dev/input/js* joystick device found")
    return devices[0]


def print_device_list():
    devices = sorted(glob.glob("/dev/input/js*"))
    if not devices:
        print("No /dev/input/js* joystick devices found.")
        return 1

    for device in devices:
        try:
            with open(device, "rb", buffering=0) as fd:
                axes = ioctl_u8(fd, JSIOCGAXES)
                buttons = ioctl_u8(fd, JSIOCGBUTTONS)
                print(f"{device}: {joystick_name(fd)} ({axes} axes, {buttons} buttons)")
        except OSError as exc:
            print(f"{device}: could not open: {exc}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Print live Xbox controller button and axis events."
    )
    parser.add_argument(
        "--device",
        help="Joystick device to read, for example /dev/input/js0. Defaults to first js* device.",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.03,
        help="Stick deadzone for display filtering, from 0.0 to 1.0. Default: 0.03.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List joystick devices and exit.",
    )
    args = parser.parse_args()

    if args.list:
        return print_device_list()

    try:
        device = choose_device(args.device)
        fd = open(device, "rb", buffering=0)
    except OSError as exc:
        print(f"Could not open joystick device: {exc}", file=sys.stderr)
        print("Try: ls -l /dev/input/js*  or run with sudo if permissions block access.", file=sys.stderr)
        return 1

    with fd:
        axes_count = ioctl_u8(fd, JSIOCGAXES)
        buttons_count = ioctl_u8(fd, JSIOCGBUTTONS)
        axes = axis_map(fd, axes_count)
        buttons = button_map(fd, buttons_count)
        axis_values = {name: 0 for name in axes}
        button_values = {name: 0 for name in buttons}

        print(f"Device: {device}")
        print(f"Name: {joystick_name(fd)}")
        print(f"Axes: {', '.join(f'{i}:{name}' for i, name in enumerate(axes)) if axes else 'none'}")
        print(f"Buttons: {', '.join(f'{i}:{name}' for i, name in enumerate(buttons)) if buttons else 'none'}")
        print("Move a stick, trigger, or D-pad, or press buttons. Press Ctrl+C to stop.")

        while True:
            readable, _, _ = select.select([fd], [], [], 0.5)
            if not readable:
                continue

            data = fd.read(8)
            if len(data) != 8:
                time.sleep(0.01)
                continue

            event_time, value, event_type, number = struct.unpack("IhBB", data)
            is_initial = bool(event_type & JS_EVENT_INIT)
            event_type &= ~JS_EVENT_INIT
            prefix = "init" if is_initial else "event"

            if event_type == JS_EVENT_AXIS and number < len(axes):
                name = axes[number]
                axis_values[name] = value
                norm = normalize_axis(value)

                if abs(norm) < args.deadzone:
                    norm = 0.0

                print(f"{prefix:5} {event_time:10d} axis   {number:02d} {name:28s} raw={value:6d} norm={norm:+.2f}")

            elif event_type == JS_EVENT_BUTTON and number < len(buttons):
                name = buttons[number]
                button_values[name] = value
                state = "pressed" if value else "released"
                print(f"{prefix:5} {event_time:10d} button {number:02d} {name:28s} {state}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
