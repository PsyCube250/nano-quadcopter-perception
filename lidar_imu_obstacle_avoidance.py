import argparse
import math
import time
from dataclasses import dataclass

from lidar_3d_scan_mapper_mpu6050 import (
    MPUOrientationEstimator,
    lidar_point_to_vertical_local,
    open_mpu6050,
    parse_auto_int,
    rotate_map_orientation,
    rotate_orientation,
)
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


DEFAULT_SET_DISTANCE_MM = 1000
DEFAULT_MAX_RANGE_MM = 6000
DEFAULT_MIN_HEIGHT_MM = -500
DEFAULT_MAX_HEIGHT_MM = 1500
DEFAULT_MAX_RC_OFFSET = 120
DEFAULT_RC_STEP = 8
DEFAULT_THROTTLE = 1000
RC_CENTER = 1500
ARM_AUX = 1800
DISARM_AUX = 1000


@dataclass
class Obstacle:
    x_mm: float
    y_mm: float
    z_mm: float
    horizontal_mm: float
    distance_mm: int
    angle_deg: float
    confidence: int


def normalize_signed_degrees(angle_deg):
    wrapped = (angle_deg + 180.0) % 360.0 - 180.0

    if wrapped == -180.0:
        return 180.0

    return wrapped


def rc_channels(roll, pitch, throttle, yaw=RC_CENTER, arm=True):
    aux1 = ARM_AUX if arm else DISARM_AUX
    return [roll, pitch, throttle, yaw, aux1, 1000, 1000, 1000]


def point_to_obstacle(point, roll_deg, pitch_deg, yaw_deg, args):
    x_local, y_local, z_local = lidar_point_to_vertical_local(
        point,
        args.angle_offset_deg,
    )
    steering_yaw_deg = yaw_deg if args.use_yaw_frame else 0.0
    x_mm, y_mm, z_mm = rotate_orientation(
        x_local,
        y_local,
        z_local,
        roll_deg,
        pitch_deg,
        steering_yaw_deg,
    )
    x_mm, y_mm, z_mm = rotate_map_orientation(
        x_mm,
        y_mm,
        z_mm,
        args.map_roll_deg,
        args.map_pitch_deg,
        args.map_yaw_deg,
    )
    horizontal_mm = math.hypot(x_mm, y_mm)

    if horizontal_mm < args.min_horizontal_mm:
        return None

    if point["distance_mm"] > args.max_range_mm:
        return None

    if not args.min_height_mm <= z_mm <= args.max_height_mm:
        return None

    bearing_deg = math.degrees(math.atan2(x_mm, y_mm))
    if abs(normalize_signed_degrees(bearing_deg)) > args.sector_half_angle_deg:
        return None

    return Obstacle(
        x_mm=x_mm,
        y_mm=y_mm,
        z_mm=z_mm,
        horizontal_mm=horizontal_mm,
        distance_mm=point["distance_mm"],
        angle_deg=point["angle_deg"],
        confidence=point["confidence"],
    )


def nearest_obstacle(lidar_points, roll_deg, pitch_deg, yaw_deg, args):
    obstacles = []

    for point in lidar_points:
        obstacle = point_to_obstacle(point, roll_deg, pitch_deg, yaw_deg, args)

        if obstacle is not None:
            obstacles.append(obstacle)

    if not obstacles:
        return None

    return min(obstacles, key=lambda obstacle: obstacle.horizontal_mm)


def avoidance_targets(obstacle, args):
    if obstacle is None:
        return RC_CENTER, RC_CENTER, "CLEAR", 0.0

    error_mm = args.set_distance_mm - obstacle.horizontal_mm

    if error_mm <= args.deadband_mm:
        return RC_CENTER, RC_CENTER, "CLEAR", 0.0

    control_band_mm = max(1.0, args.set_distance_mm - args.deadband_mm)
    strength = clamp(error_mm / control_band_mm, 0.0, 1.0)
    away_x = -obstacle.x_mm / obstacle.horizontal_mm
    away_y = -obstacle.y_mm / obstacle.horizontal_mm
    rc_scale = args.max_rc_offset * strength
    roll = RC_CENTER + args.rc_roll_sign * away_x * rc_scale
    pitch = RC_CENTER + args.rc_pitch_sign * away_y * rc_scale

    return int(round(roll)), int(round(pitch)), "AVOID", strength


def format_obstacle(obstacle):
    if obstacle is None:
        return "none"

    bearing_deg = math.degrees(math.atan2(obstacle.x_mm, obstacle.y_mm))
    return (
        f"{obstacle.horizontal_mm:.0f} mm horizontal, "
        f"bearing={bearing_deg:.1f} deg, z={obstacle.z_mm:.0f} mm, "
        f"raw={obstacle.distance_mm} mm at {obstacle.angle_deg:.1f} deg, "
        f"conf={obstacle.confidence}"
    )


def confirm_or_exit(args):
    print("=== LiDAR + MPU-6050 OBSTACLE AVOIDANCE RC TEST ===")
    print("PROPELLERS MUST BE REMOVED for bench testing.")
    print("Betaflight Configurator must be disconnected.")
    print("This sends MSP RC roll/pitch commands to steer away from obstacles.")
    print("")
    print(f"FC:                 {args.fc_port} @ {args.fc_baud}")
    print(f"LiDAR:              {args.lidar_port} @ {args.lidar_baud}")
    print(f"Set distance:       {args.set_distance_mm} mm")
    print(f"Deadband:           {args.deadband_mm} mm")
    print(f"Height window:      {args.min_height_mm} to {args.max_height_mm} mm")
    print(f"Sector half-angle:  {args.sector_half_angle_deg} deg")
    print(f"Throttle:           {args.throttle}")
    print(f"Max RC offset:      +/-{args.max_rc_offset}")
    print(f"Dry run:            {'yes' if args.dry_run else 'no'}")
    print("")

    if args.dry_run:
        print("Dry run mode: no flight-controller RC commands will be sent.")
        return

    expected = "PROPS REMOVED"
    print(f'Type "{expected}" to arm and run this RC test.')
    answer = input("> ").strip()

    if answer != expected:
        raise SystemExit("Confirmation did not match. Aborting.")


def validate_args(args):
    if args.set_distance_mm <= 0:
        raise SystemExit("--set-distance-mm must be greater than zero")

    if args.deadband_mm < 0:
        raise SystemExit("--deadband-mm cannot be negative")

    if args.deadband_mm >= args.set_distance_mm:
        raise SystemExit("--deadband-mm must be smaller than --set-distance-mm")

    if args.max_range_mm <= 0:
        raise SystemExit("--max-range-mm must be greater than zero")

    if args.min_height_mm >= args.max_height_mm:
        raise SystemExit("--min-height-mm must be less than --max-height-mm")

    if not 0.0 < args.sector_half_angle_deg <= 180.0:
        raise SystemExit("--sector-half-angle-deg must be in the range 0..180")

    if args.min_horizontal_mm < 0:
        raise SystemExit("--min-horizontal-mm cannot be negative")

    if not 0 <= args.max_rc_offset <= 400:
        raise SystemExit("--max-rc-offset must be between 0 and 400")

    if args.rc_step <= 0:
        raise SystemExit("--rc-step must be greater than zero")

    if not 1000 <= args.throttle <= 2000:
        raise SystemExit("--throttle must be between 1000 and 2000")

    if not 1000 <= args.yaw <= 2000:
        raise SystemExit("--yaw must be between 1000 and 2000")

    if args.throttle > 1150 and not args.allow_flight_throttle:
        raise SystemExit(
            "Refusing throttle above 1150 without --allow-flight-throttle. "
            "Use only after the low-throttle bench behavior is verified."
        )

    if args.run_sec <= 0:
        raise SystemExit("--run-sec must be greater than zero")

    if args.calibration_sec < 0.5:
        raise SystemExit("--calibration-sec should be at least 0.5")

    if not 0.0 <= args.accel_alpha <= 1.0:
        raise SystemExit("--accel-alpha must be between 0.0 and 1.0")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Use STL-27L LiDAR plus MPU-6050 attitude to send MSP RC roll/pitch "
            "commands away from nearby obstacles until they are past a set distance."
        )
    )
    parser.add_argument("--fc-port", default=FC_PORT)
    parser.add_argument("--fc-baud", type=int, default=FC_BAUD)
    parser.add_argument("--lidar-port", default=LIDAR_PORT)
    parser.add_argument("--lidar-baud", type=int, default=LIDAR_BAUD)
    parser.add_argument("--i2c-bus", type=parse_auto_int, default=None)
    parser.add_argument("--imu-addr", type=parse_auto_int, default=None)
    parser.add_argument("--yaw-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--roll-axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--pitch-axis", choices=("x", "y", "z"), default="y")
    parser.add_argument("--roll-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--pitch-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--yaw-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--accel-alpha", type=float, default=0.98)
    parser.add_argument("--calibration-sec", type=float, default=3.0)
    parser.add_argument("--sample-hz", type=float, default=50.0)
    parser.add_argument("--angle-offset-deg", type=float, default=0.0)
    parser.add_argument("--map-roll-deg", type=float, default=0.0)
    parser.add_argument("--map-pitch-deg", type=float, default=0.0)
    parser.add_argument("--map-yaw-deg", type=float, default=0.0)
    parser.add_argument("--use-yaw-frame", action="store_true")
    parser.add_argument("--set-distance-mm", type=float, default=DEFAULT_SET_DISTANCE_MM)
    parser.add_argument("--deadband-mm", type=float, default=50.0)
    parser.add_argument("--max-range-mm", type=float, default=DEFAULT_MAX_RANGE_MM)
    parser.add_argument("--min-height-mm", type=float, default=DEFAULT_MIN_HEIGHT_MM)
    parser.add_argument("--max-height-mm", type=float, default=DEFAULT_MAX_HEIGHT_MM)
    parser.add_argument("--min-horizontal-mm", type=float, default=100.0)
    parser.add_argument("--sector-half-angle-deg", type=float, default=90.0)
    parser.add_argument("--max-rc-offset", type=int, default=DEFAULT_MAX_RC_OFFSET)
    parser.add_argument("--rc-step", type=int, default=DEFAULT_RC_STEP)
    parser.add_argument("--rc-roll-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--rc-pitch-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--throttle", type=int, default=DEFAULT_THROTTLE)
    parser.add_argument("--yaw", type=int, default=RC_CENTER)
    parser.add_argument("--warmup-sec", type=float, default=8.0)
    parser.add_argument("--arm-sec", type=float, default=2.0)
    parser.add_argument("--run-sec", type=float, default=10.0)
    parser.add_argument("--disarm-sec", type=float, default=2.0)
    parser.add_argument("--print-hz", type=float, default=4.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-flight-throttle", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    confirm_or_exit(args)

    msp = None
    lidar = None
    imu = None
    current_roll = RC_CENTER
    current_pitch = RC_CENTER

    try:
        imu = open_mpu6050(args.i2c_bus, args.imu_addr)
        imu.initialize()
        who = imu.who_am_i()
        print(f"MPU-6050 WHO_AM_I: 0x{who:02X}")

        if who not in (0x68, 0x70):
            print("Warning: WHO_AM_I is unusual for MPU-6050. Check address/wiring.")

        orientation = MPUOrientationEstimator(
            imu,
            roll_axis=args.roll_axis,
            pitch_axis=args.pitch_axis,
            yaw_axis=args.yaw_axis,
            roll_sign=args.roll_sign,
            pitch_sign=args.pitch_sign,
            yaw_sign=args.yaw_sign,
            accel_alpha=args.accel_alpha,
        )
        orientation.calibrate(args.calibration_sec)

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
            send_for_duration(msp, arm_rc, args.arm_sec, "Arming")
            print_status(msp, "Status after arm")

        print(f"Running obstacle avoidance for {args.run_sec:.1f}s")
        end_time = time.time() + args.run_sec
        last_print = 0.0
        print_period = 1.0 / max(0.1, args.print_hz)
        loop_period = 1.0 / max(1.0, min(args.sample_hz, LOOP_HZ))

        while time.time() < end_time:
            roll_deg, pitch_deg, yaw_deg = orientation.update()
            lidar_points = lidar.get_points()
            obstacle = nearest_obstacle(
                lidar_points,
                roll_deg,
                pitch_deg,
                yaw_deg,
                args,
            )
            target_roll, target_pitch, state, strength = avoidance_targets(
                obstacle,
                args,
            )
            current_roll = step_toward(
                current_roll,
                target_roll,
                args.rc_step,
            )
            current_pitch = step_toward(
                current_pitch,
                target_pitch,
                args.rc_step,
            )
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
                args.throttle,
                yaw=args.yaw,
                arm=True,
            )

            if msp is not None:
                msp.send_rc(channels)

            now = time.time()

            if now - last_print >= print_period:
                last_print = now
                print(
                    f"state={state:<5} strength={strength:.2f} "
                    f"rc=roll:{current_roll} pitch:{current_pitch} "
                    f"rpy=({roll_deg:.1f}, {pitch_deg:.1f}, {yaw_deg:.1f}) "
                    f"points={len(lidar_points):3d} obstacle={format_obstacle(obstacle)}"
                )

            time.sleep(loop_period)

    except KeyboardInterrupt:
        print("\nInterrupted.")

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

        if imu is not None:
            imu.close()

        print("Done.")


if __name__ == "__main__":
    main()
