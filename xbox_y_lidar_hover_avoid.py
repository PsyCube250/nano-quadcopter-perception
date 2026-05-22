#!/usr/bin/env python3
"""
Wait for Xbox Y, then arm, boost throttle, and hover with front LiDAR avoidance.

Default controller mapping:
- Y, button 3 -> start sequence
- RB, button 7 -> toggle manual/LiDAR mode after boost
- B, button 1 -> emergency disarm/stop
- Left stick X, axis 0 -> manual yaw
- Left stick Y, axis 1 -> manual throttle around hover throttle
- Right stick X, axis 2 -> manual roll
- Right stick Y, axis 3 -> manual pitch

This sends MSP_SET_RAW_RC packets to the flight controller. It does not command
motors directly.
"""

import argparse
import math
import signal
import sys
import time

from lidar_stl27l import LIDAR_BAUD, LIDAR_PORT, STL27LReader
from motor_lidar_speed_control_test import (
    FC_BAUD,
    FC_PORT,
    LOOP_HZ,
    MSP,
    arm_rc,
    clamp,
    disarm_rc,
    print_status,
    send_for_duration,
    step_toward,
)
from xbox_drone_control import XboxJoystick, centered_rc, throttle_rc


RC_CENTER = 1500
RC_MIN = 1000
RC_MAX = 2000
ARM_AUX = 1800
DISARM_AUX = 1000

DEFAULT_START_BUTTON = 3
DEFAULT_MANUAL_BUTTON = 7
DEFAULT_KILL_BUTTON = 1
DEFAULT_YAW_AXIS = 0
DEFAULT_THROTTLE_AXIS = 1
DEFAULT_ROLL_AXIS = 2
DEFAULT_PITCH_AXIS = 3
DEFAULT_ARM_SEC = 1.0
DEFAULT_BOOST_SEC = 3.0
DEFAULT_BOOST_THROTTLE = 1250
DEFAULT_HOVER_THROTTLE = 1200
DEFAULT_THROTTLE_OFFSET = 300
DEFAULT_MAX_MANUAL_OFFSET = 180
DEFAULT_MAX_YAW_OFFSET = 160
DEFAULT_SET_DISTANCE_MM = 1000
DEFAULT_MAX_RANGE_MM = 6000
DEFAULT_MAX_RC_OFFSET = 120
DEFAULT_RC_STEP = 8


def stop_requested(_signum, _frame):
    raise KeyboardInterrupt


def normalize_signed_degrees(angle_deg):
    wrapped = (angle_deg + 180.0) % 360.0 - 180.0

    if wrapped == -180.0:
        return 180.0

    return wrapped


def rc_channels(roll, pitch, throttle, yaw=RC_CENTER, armed=True):
    aux1 = ARM_AUX if armed else DISARM_AUX

    if not armed:
        return disarm_rc()

    return [roll, pitch, throttle, yaw, aux1, 1000, 1000, 1000]


def front_bearing(point, angle_offset_deg):
    return normalize_signed_degrees(point["angle_deg"] + angle_offset_deg)


def nearest_front_obstacle(lidar_points, args):
    candidates = []

    for point in lidar_points:
        if point["distance_mm"] > args.max_range_mm:
            continue

        bearing = front_bearing(point, args.angle_offset_deg)

        if abs(bearing) <= args.front_half_angle_deg:
            candidates.append((point["distance_mm"], bearing, point))

    if not candidates:
        return None

    _distance, bearing, point = min(candidates, key=lambda item: item[0])
    return point, bearing


def avoidance_targets(obstacle_info, args):
    if obstacle_info is None:
        return RC_CENTER, RC_CENTER, "CLEAR", 0.0

    point, bearing = obstacle_info
    distance_mm = point["distance_mm"]
    error_mm = args.set_distance_mm - distance_mm

    if error_mm <= 0:
        return RC_CENTER, RC_CENTER, "CLEAR", 0.0

    control_band = max(1.0, args.set_distance_mm - args.deadband_mm)
    strength = clamp(error_mm / control_band, 0.0, 1.0)
    bearing_rad = math.radians(bearing)

    # With 0 deg as front, +bearing is right. Move opposite the obstacle vector.
    away_x = -math.sin(bearing_rad)
    away_y = -math.cos(bearing_rad)
    rc_scale = args.max_rc_offset * strength
    roll = RC_CENTER + args.rc_roll_sign * away_x * rc_scale
    pitch = RC_CENTER + args.rc_pitch_sign * away_y * rc_scale

    return int(round(roll)), int(round(pitch)), "AVOID", strength


def format_obstacle(obstacle_info):
    if obstacle_info is None:
        return "none"

    point, bearing = obstacle_info
    return (
        f"{point['distance_mm']} mm bearing={bearing:.1f} deg "
        f"raw_angle={point['angle_deg']:.1f} conf={point['confidence']}"
    )


def validate_button(label, button, joystick):
    if button < 0 or button >= len(joystick.buttons):
        raise SystemExit(f"{label} {button} is not present on {joystick.device}")


def validate_axis(label, axis, joystick):
    if axis < 0 or axis >= len(joystick.axes):
        raise SystemExit(f"{label} {axis} is not present on {joystick.device}")


def validate_args(args):
    if args.arm_sec < 0:
        raise SystemExit("--arm-sec cannot be negative")

    if args.boost_sec <= 0:
        raise SystemExit("--boost-sec must be greater than zero")

    if args.warmup_sec < 0 or args.disarm_sec < 0:
        raise SystemExit("--warmup-sec and --disarm-sec cannot be negative")

    if args.run_sec < 0:
        raise SystemExit("--run-sec cannot be negative")

    for label, value in (
        ("--boost-throttle", args.boost_throttle),
        ("--hover-throttle", args.hover_throttle),
        ("--yaw", args.yaw),
    ):
        if not RC_MIN <= value <= RC_MAX:
            raise SystemExit(f"{label} must be between 1000 and 2000")

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

    if not 0.0 <= args.deadzone < 0.9:
        raise SystemExit("--deadzone must be between 0.0 and 0.9")

    if not 0 <= args.max_roll_offset <= 500:
        raise SystemExit("--max-roll-offset must be between 0 and 500")

    if not 0 <= args.max_pitch_offset <= 500:
        raise SystemExit("--max-pitch-offset must be between 0 and 500")

    if not 0 <= args.max_yaw_offset <= 500:
        raise SystemExit("--max-yaw-offset must be between 0 and 500")

    if (
        max(args.boost_throttle, args.hover_throttle, args.max_throttle) > 1150
        and not args.allow_flight_throttle
        and not args.dry_run
    ):
        raise SystemExit(
            "Refusing throttle above 1150 without --allow-flight-throttle. "
            "Bench-test with propellers removed and verify low throttle first."
        )

    if args.set_distance_mm <= 0:
        raise SystemExit("--set-distance-mm must be greater than zero")

    if args.deadband_mm < 0:
        raise SystemExit("--deadband-mm cannot be negative")

    if args.deadband_mm >= args.set_distance_mm:
        raise SystemExit("--deadband-mm must be smaller than --set-distance-mm")

    if args.max_range_mm <= 0:
        raise SystemExit("--max-range-mm must be greater than zero")

    if not 0.0 < args.front_half_angle_deg <= 180.0:
        raise SystemExit("--front-half-angle-deg must be in the range 0..180")

    if not 0 <= args.max_rc_offset <= 400:
        raise SystemExit("--max-rc-offset must be between 0 and 400")

    if args.rc_step <= 0:
        raise SystemExit("--rc-step must be greater than zero")

    if args.print_hz <= 0:
        raise SystemExit("--print-hz must be greater than zero")


def confirm_or_exit(args, joystick):
    print("=== XBOX Y + LIDAR HOVER AVOID ===")
    print("PROPELLERS MUST BE REMOVED for bench testing.")
    print("Betaflight Configurator must be disconnected.")
    print("This sends MSP RC throttle, roll, and pitch commands.")
    print("")
    print(f"Controller:       {joystick.device} - {joystick.name}")
    print(f"Axes:             {', '.join(f'{i}:{name}' for i, name in enumerate(joystick.axes))}")
    print(f"Buttons:          {', '.join(f'{i}:{name}' for i, name in enumerate(joystick.buttons))}")
    print(f"Start button:     {args.start_button} (Y by default)")
    print(f"Manual toggle:    {args.manual_button} (RB by default)")
    print(f"Kill button:      {args.kill_button} (B by default)")
    print(f"Manual axes:      yaw={args.yaw_axis} throttle={args.throttle_axis} roll={args.roll_axis} pitch={args.pitch_axis}")
    print(f"FC:               {args.fc_port} @ {args.fc_baud}")
    print(f"LiDAR:            {args.lidar_port} @ {args.lidar_baud}")
    print(f"Boost:            {args.boost_throttle} for {args.boost_sec:.1f}s")
    print(f"Hover throttle:   {args.hover_throttle}")
    print(f"Front sector:     +/-{args.front_half_angle_deg:.1f} deg")
    print(f"Set distance:     {args.set_distance_mm:.0f} mm")
    print(f"Max RC offset:    +/-{args.max_rc_offset}")
    print(f"Dry run:          {'yes' if args.dry_run else 'no'}")
    print("")

    if args.dry_run:
        print("Dry run mode: no flight-controller RC commands will be sent.")
        return

    if args.no_confirm:
        print("No-confirm mode: skipping terminal confirmation.")
        return

    expected = "PROPS REMOVED"
    print(f'Type "{expected}" to arm and run this test.')
    answer = input("> ").strip()

    if answer != expected:
        raise SystemExit("Confirmation did not match. Aborting.")


def wait_for_start_button(joystick, msp, args):
    print("Waiting for Y/start button. Press B/kill or Ctrl+C to abort.")
    joystick.poll()
    previous_start = joystick.button_pressed(args.start_button)
    last_print = 0.0
    loop_period = 1.0 / LOOP_HZ

    while True:
        joystick.poll()
        start_button = joystick.button_pressed(args.start_button)
        start_pressed = start_button and not previous_start
        previous_start = start_button

        if not joystick.connected:
            raise RuntimeError("Controller disconnected while waiting for Y.")

        if joystick.button_pressed(args.kill_button):
            raise RuntimeError("Kill button pressed before start.")

        if msp is not None:
            msp.send_rc(disarm_rc())

        if start_pressed:
            print("Y/start pressed. Starting arm + boost sequence.")
            return

        now = time.time()
        if now - last_print >= 1.0:
            last_print = now
            print("state=WAIT_Y rc=disarmed")

        time.sleep(loop_period)


def arm_low_throttle(msp, joystick, args):
    if args.arm_sec <= 0:
        return

    print(f"Arming at low throttle for {args.arm_sec:.1f}s")
    end = time.time() + args.arm_sec
    loop_period = 1.0 / LOOP_HZ

    while time.time() < end:
        joystick.poll()

        if not joystick.connected:
            raise RuntimeError("Controller disconnected during arming.")

        if joystick.button_pressed(args.kill_button):
            raise RuntimeError("Kill button pressed during arming.")

        if msp is not None:
            msp.send_rc(arm_rc())

        time.sleep(loop_period)


def boost_throttle(msp, joystick, args):
    print(f"Boost throttle {args.boost_throttle} for {args.boost_sec:.1f}s")
    end = time.time() + args.boost_sec
    loop_period = 1.0 / LOOP_HZ
    last_print = 0.0

    while time.time() < end:
        joystick.poll()

        if not joystick.connected:
            raise RuntimeError("Controller disconnected during boost.")

        if joystick.button_pressed(args.kill_button):
            raise RuntimeError("Kill button pressed during boost.")

        channels = rc_channels(
            RC_CENTER,
            RC_CENTER,
            args.boost_throttle,
            yaw=args.yaw,
            armed=True,
        )

        if msp is not None:
            msp.send_rc(channels)

        now = time.time()
        if now - last_print >= 0.25:
            last_print = now
            print(
                f"state=BOOST rc=roll:{channels[0]} pitch:{channels[1]} "
                f"thr:{channels[2]} yaw:{channels[3]} aux1:{channels[4]}"
            )

        time.sleep(loop_period)


def hover_with_lidar(msp, joystick, lidar, args):
    if args.run_sec > 0:
        end_time = time.time() + args.run_sec
        print(f"Hover/manual control for {args.run_sec:.1f}s")
    else:
        end_time = None
        print("Hover/manual control until Ctrl+C or B/kill")

    current_roll = RC_CENTER
    current_pitch = RC_CENTER
    manual_mode = False
    previous_manual = joystick.button_pressed(args.manual_button)
    last_print = 0.0
    print_period = 1.0 / args.print_hz
    loop_period = 1.0 / LOOP_HZ

    while end_time is None or time.time() < end_time:
        joystick.poll()

        if not joystick.connected:
            raise RuntimeError("Controller disconnected during hover.")

        if joystick.button_pressed(args.kill_button):
            raise RuntimeError("Kill button pressed during hover.")

        manual_button = joystick.button_pressed(args.manual_button)
        manual_pressed = manual_button and not previous_manual
        previous_manual = manual_button

        if manual_pressed:
            manual_mode = not manual_mode
            print(f"mode={'MANUAL' if manual_mode else 'LIDAR'}")

        lidar_points = []
        obstacle = None
        strength = 0.0
        manual_norms = None

        if manual_mode:
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
            current_roll = roll
            current_pitch = pitch
            manual_norms = (yaw_norm, throttle_norm, roll_norm, pitch_norm)
            state = "MANUAL"
        else:
            yaw = args.yaw
            throttle = args.hover_throttle
            lidar_points = lidar.get_points()
            obstacle = nearest_front_obstacle(lidar_points, args)
            target_roll, target_pitch, state, strength = avoidance_targets(
                obstacle,
                args,
            )
            current_roll = step_toward(current_roll, target_roll, args.rc_step)
            current_pitch = step_toward(current_pitch, target_pitch, args.rc_step)
            current_roll = int(
                round(
                    clamp(
                        current_roll,
                        RC_CENTER - args.max_rc_offset,
                        RC_CENTER + args.max_rc_offset,
                    )
                )
            )
            current_pitch = int(
                round(
                    clamp(
                        current_pitch,
                        RC_CENTER - args.max_rc_offset,
                        RC_CENTER + args.max_rc_offset,
                    )
                )
            )

        channels = rc_channels(
            current_roll,
            current_pitch,
            throttle,
            yaw=yaw,
            armed=True,
        )

        if msp is not None:
            msp.send_rc(channels)

        now = time.time()
        if now - last_print >= print_period:
            last_print = now

            if manual_mode:
                yaw_norm, throttle_norm, roll_norm, pitch_norm = manual_norms
                detail = (
                    f"axes yaw:{yaw_norm:+.2f} thr:{throttle_norm:+.2f} "
                    f"roll:{roll_norm:+.2f} pitch:{pitch_norm:+.2f}"
                )
            else:
                detail = (
                    f"strength={strength:.2f} points={len(lidar_points):3d} "
                    f"obstacle={format_obstacle(obstacle)}"
                )

            print(
                f"state={state:<6} {detail} "
                f"rc=roll:{channels[0]} pitch:{channels[1]} "
                f"thr:{channels[2]} yaw:{channels[3]} aux1:{channels[4]}"
            )

        time.sleep(loop_period)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Wait for Xbox Y, boost throttle for 3 seconds, then hover while "
            "steering away from the nearest STL-27L LiDAR obstacle in the front 180 degrees."
        )
    )
    parser.add_argument("--device", help="Joystick device, for example /dev/input/js0")
    parser.add_argument("--fc-port", default=FC_PORT)
    parser.add_argument("--fc-baud", type=int, default=FC_BAUD)
    parser.add_argument("--lidar-port", default=LIDAR_PORT)
    parser.add_argument("--lidar-baud", type=int, default=LIDAR_BAUD)
    parser.add_argument("--start-button", type=int, default=DEFAULT_START_BUTTON)
    parser.add_argument("--manual-button", type=int, default=DEFAULT_MANUAL_BUTTON)
    parser.add_argument("--kill-button", type=int, default=DEFAULT_KILL_BUTTON)
    parser.add_argument("--yaw-axis", type=int, default=DEFAULT_YAW_AXIS)
    parser.add_argument("--throttle-axis", type=int, default=DEFAULT_THROTTLE_AXIS)
    parser.add_argument("--roll-axis", type=int, default=DEFAULT_ROLL_AXIS)
    parser.add_argument("--pitch-axis", type=int, default=DEFAULT_PITCH_AXIS)
    parser.add_argument("--warmup-sec", type=float, default=3.0)
    parser.add_argument("--arm-sec", type=float, default=DEFAULT_ARM_SEC)
    parser.add_argument("--boost-sec", type=float, default=DEFAULT_BOOST_SEC)
    parser.add_argument("--boost-throttle", type=int, default=DEFAULT_BOOST_THROTTLE)
    parser.add_argument("--hover-throttle", type=int, default=DEFAULT_HOVER_THROTTLE)
    parser.add_argument("--throttle-offset", type=int, default=DEFAULT_THROTTLE_OFFSET)
    parser.add_argument("--min-throttle", type=int, default=1000)
    parser.add_argument("--max-throttle", type=int, default=1600)
    parser.add_argument("--deadzone", type=float, default=0.08)
    parser.add_argument("--max-roll-offset", type=int, default=DEFAULT_MAX_MANUAL_OFFSET)
    parser.add_argument("--max-pitch-offset", type=int, default=DEFAULT_MAX_MANUAL_OFFSET)
    parser.add_argument("--max-yaw-offset", type=int, default=DEFAULT_MAX_YAW_OFFSET)
    parser.add_argument("--roll-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--pitch-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--yaw-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--throttle-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--yaw", type=int, default=RC_CENTER)
    parser.add_argument("--run-sec", type=float, default=0.0)
    parser.add_argument("--disarm-sec", type=float, default=2.0)
    parser.add_argument("--angle-offset-deg", type=float, default=180.0)
    parser.add_argument("--front-half-angle-deg", type=float, default=90.0)
    parser.add_argument("--set-distance-mm", type=float, default=DEFAULT_SET_DISTANCE_MM)
    parser.add_argument("--deadband-mm", type=float, default=50.0)
    parser.add_argument("--max-range-mm", type=float, default=DEFAULT_MAX_RANGE_MM)
    parser.add_argument("--max-rc-offset", type=int, default=DEFAULT_MAX_RC_OFFSET)
    parser.add_argument("--rc-step", type=int, default=DEFAULT_RC_STEP)
    parser.add_argument("--rc-roll-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--rc-pitch-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--print-hz", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-flight-throttle", action="store_true")
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help=(
            "Skip the terminal PROPS REMOVED prompt. Intended for supervised "
            "boot services that still wait for the Xbox Y button."
        ),
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)

    joystick = None
    lidar = None
    msp = None

    try:
        joystick = XboxJoystick(args.device)
        validate_button("--start-button", args.start_button, joystick)
        validate_button("--manual-button", args.manual_button, joystick)
        validate_button("--kill-button", args.kill_button, joystick)
        validate_axis("--yaw-axis", args.yaw_axis, joystick)
        validate_axis("--throttle-axis", args.throttle_axis, joystick)
        validate_axis("--roll-axis", args.roll_axis, joystick)
        validate_axis("--pitch-axis", args.pitch_axis, joystick)
        confirm_or_exit(args, joystick)

        lidar = STL27LReader(
            args.lidar_port,
            args.lidar_baud,
            point_max_age_sec=0.25,
        )
        lidar.start()

        if not args.dry_run:
            msp = MSP(args.fc_port, args.fc_baud)
            send_for_duration(msp, disarm_rc, args.warmup_sec, "Warmup disarm RC")
            print_status(msp, "Status after warmup")

        wait_for_start_button(joystick, msp, args)
        arm_low_throttle(msp, joystick, args)

        if msp is not None:
            print_status(msp, "Status after arm")

        boost_throttle(msp, joystick, args)
        hover_with_lidar(msp, joystick, lidar, args)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except (OSError, RuntimeError) as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        if msp is not None:
            print("Disarming now.")
            try:
                send_for_duration(msp, disarm_rc, args.disarm_sec, "Disarm")
                print_status(msp, "Final status")
            finally:
                msp.close()

        if lidar is not None:
            lidar.close()

        if joystick is not None:
            joystick.close()

        print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
