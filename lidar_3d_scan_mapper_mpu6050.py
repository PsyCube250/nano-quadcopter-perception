import argparse
import math
import time
from pathlib import Path

from lidar_3d_scan_mapper import (
    color_for_point,
    save_csv,
    save_html_viewer,
    save_ply,
    voxel_key,
)
from lidar_stl27l import LIDAR_BAUD, LIDAR_PORT, STL27LReader
from mpu6050_i2c import MPU6050, MPU6050_ADDRS, detect_mpu6050


DEFAULT_OUTPUT_PREFIX = "lidar_3d_imu_scan"


class MPUOrientationEstimator:
    def __init__(
        self,
        imu,
        roll_axis="x",
        pitch_axis="y",
        yaw_axis="z",
        roll_sign=1.0,
        pitch_sign=1.0,
        yaw_sign=1.0,
        accel_alpha=0.98,
    ):
        self.imu = imu
        self.roll_axis = roll_axis
        self.pitch_axis = pitch_axis
        self.yaw_axis = yaw_axis
        self.roll_sign = roll_sign
        self.pitch_sign = pitch_sign
        self.yaw_sign = yaw_sign
        self.gyro_bias = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.roll_deg = 0.0
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0
        self.accel_alpha = accel_alpha
        self.last_time = None

    def _axis_rates(self, gx, gy, gz):
        rates = {"x": gx, "y": gy, "z": gz}
        return {
            "roll": (
                rates[self.roll_axis] - self.gyro_bias[self.roll_axis]
            ) * self.roll_sign,
            "pitch": (
                rates[self.pitch_axis] - self.gyro_bias[self.pitch_axis]
            ) * self.pitch_sign,
            "yaw": (
                rates[self.yaw_axis] - self.gyro_bias[self.yaw_axis]
            ) * self.yaw_sign,
        }

    @staticmethod
    def _accel_roll_pitch(ax, ay, az):
        roll = math.degrees(math.atan2(ay, az))
        pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
        return roll, pitch

    def calibrate(self, seconds=3.0, sample_hz=200.0):
        print(f"Calibrating MPU-6050 gyro for {seconds:.1f}s.")
        print("Keep the LiDAR + IMU completely still.")

        sums = {"x": 0.0, "y": 0.0, "z": 0.0}
        accel_roll_sum = 0.0
        accel_pitch_sum = 0.0
        count = 0
        end = time.time() + seconds
        sleep_sec = 1.0 / max(1.0, sample_hz)

        while time.time() < end:
            ax, ay, az, gx, gy, gz = self.imu.read_motion()
            accel_roll, accel_pitch = self._accel_roll_pitch(ax, ay, az)
            accel_roll_sum += accel_roll
            accel_pitch_sum += accel_pitch
            sums["x"] += gx
            sums["y"] += gy
            sums["z"] += gz
            count += 1
            time.sleep(sleep_sec)

        if count == 0:
            raise RuntimeError("No MPU-6050 samples were read during calibration.")

        self.gyro_bias = {
            axis: total / count
            for axis, total in sums.items()
        }
        self.roll_deg = accel_roll_sum / count
        self.pitch_deg = accel_pitch_sum / count
        self.yaw_deg = 0.0
        self.last_time = time.time()

        print(
            "Gyro bias dps: "
            f"x={self.gyro_bias['x']:.4f}, "
            f"y={self.gyro_bias['y']:.4f}, "
            f"z={self.gyro_bias['z']:.4f}"
        )
        print(
            "Initial accel attitude: "
            f"roll={self.roll_deg:.1f}, pitch={self.pitch_deg:.1f}, yaw=0.0 deg"
        )

    def update(self):
        now = time.time()
        ax, ay, az, gx, gy, gz = self.imu.read_motion()

        if self.last_time is None:
            self.last_time = now
            return self.roll_deg, self.pitch_deg, self.yaw_deg

        dt = max(0.0, min(0.1, now - self.last_time))
        self.last_time = now
        rates = self._axis_rates(gx, gy, gz)
        accel_roll, accel_pitch = self._accel_roll_pitch(ax, ay, az)

        gyro_roll = self.roll_deg + rates["roll"] * dt
        gyro_pitch = self.pitch_deg + rates["pitch"] * dt

        self.roll_deg = (
            self.accel_alpha * gyro_roll
            + (1.0 - self.accel_alpha) * accel_roll
        )
        self.pitch_deg = (
            self.accel_alpha * gyro_pitch
            + (1.0 - self.accel_alpha) * accel_pitch
        )
        self.yaw_deg += rates["yaw"] * dt
        return self.roll_deg, self.pitch_deg, self.yaw_deg


MPUYawEstimator = MPUOrientationEstimator


def lidar_point_to_vertical_local(point, angle_offset_deg):
    distance = point["distance_mm"]
    elevation = math.radians(point["angle_deg"] + angle_offset_deg)
    x_local = 0.0
    y_local = distance * math.cos(elevation)
    z_local = distance * math.sin(elevation)
    return x_local, y_local, z_local


def rotate_orientation(x_local, y_local, z_local, roll_deg, pitch_deg, yaw_deg):
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cr = math.cos(roll)
    sr = math.sin(roll)
    y1 = y_local * cr - z_local * sr
    z1 = y_local * sr + z_local * cr
    x1 = x_local

    cp = math.cos(pitch)
    sp = math.sin(pitch)
    x2 = x1 * cp + z1 * sp
    z2 = -x1 * sp + z1 * cp
    y2 = y1

    cy = math.cos(yaw)
    sy = math.sin(yaw)
    x3 = x2 * cy - y2 * sy
    y3 = x2 * sy + y2 * cy
    z3 = z2

    return x3, y3, z3


def rotate_map_orientation(x, y, z, roll_deg, pitch_deg, yaw_deg):
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cr = math.cos(roll)
    sr = math.sin(roll)
    y1 = y * cr - z * sr
    z1 = y * sr + z * cr
    x1 = x

    cp = math.cos(pitch)
    sp = math.sin(pitch)
    x2 = x1 * cp + z1 * sp
    z2 = -x1 * sp + z1 * cp
    y2 = y1

    cy = math.cos(yaw)
    sy = math.sin(yaw)
    x3 = x2 * cy - y2 * sy
    y3 = x2 * sy + y2 * cy
    z3 = z2

    return x3, y3, z3


def add_lidar_points(
    voxel_points,
    lidar_points,
    roll_deg,
    pitch_deg,
    yaw_deg,
    angle_offset_deg,
    map_roll_deg,
    map_pitch_deg,
    map_yaw_deg,
    voxel_mm,
    max_range_mm,
    max_points,
):
    added = 0

    for point in lidar_points:
        if point["distance_mm"] > max_range_mm:
            continue

        x_local, y_local, z_local = lidar_point_to_vertical_local(
            point,
            angle_offset_deg,
        )
        x, y, z = rotate_orientation(
            x_local,
            y_local,
            z_local,
            roll_deg,
            pitch_deg,
            yaw_deg,
        )
        x, y, z = rotate_map_orientation(
            x,
            y,
            z,
            map_roll_deg,
            map_pitch_deg,
            map_yaw_deg,
        )
        key = voxel_key(x, y, z, voxel_mm)

        if key in voxel_points:
            continue

        if len(voxel_points) >= max_points:
            break

        r, g, b = color_for_point(x, y, z, max_range_mm)
        voxel_points[key] = {
            "x": x,
            "y": y,
            "z": z,
            "r": r,
            "g": g,
            "b": b,
            "distance_mm": point["distance_mm"],
            "confidence": point["confidence"],
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
        }
        added += 1

    return added


def print_setup(args):
    print("=== 2D LiDAR + MPU-6050 Approximate 3D Mapper ===")
    print("This script does not touch the flight controller or motors.")
    print("")
    print("Physical setup:")
    print("- Mount the MPU-6050 rigidly to the LiDAR.")
    print("- Hold/mount the LiDAR so the 2D scan plane is vertical.")
    print("- Rotate the LiDAR + IMU smoothly around the room.")
    print("- Keep it still during gyro calibration at the start.")
    print("")
    print("Important limitation:")
    print("- MPU-6050 has no magnetometer, so yaw is gyro-integrated and will drift.")
    print("- Short, smooth scans work better than long scans.")
    print("")
    print(f"LiDAR:        {args.lidar_port} @ {args.lidar_baud}")
    if args.i2c_bus is None or args.imu_addr is None:
        print("MPU-6050:     auto-detect")
    else:
        print(f"MPU-6050:     bus {args.i2c_bus}, address 0x{args.imu_addr:02X}")
    print(
        "IMU axes:     "
        f"roll={args.roll_axis}*{args.roll_sign:g}, "
        f"pitch={args.pitch_axis}*{args.pitch_sign:g}, "
        f"yaw={args.yaw_axis}*{args.yaw_sign:g}"
    )
    print(f"Accel alpha:  {args.accel_alpha:.3f}")
    print(
        "Map rotate:   "
        f"roll={args.map_roll_deg:.1f}, "
        f"pitch={args.map_pitch_deg:.1f}, "
        f"yaw={args.map_yaw_deg:.1f} deg"
    )
    print(f"Duration:     {args.duration_sec:.1f}s")
    print(f"Voxel size:   {args.voxel_mm:.1f} mm")
    print(f"Max points:   {args.max_points}")
    print(f"Max range:    {args.max_range_mm} mm")
    print("")


def parse_auto_int(value):
    if isinstance(value, str) and value.lower() == "auto":
        return None

    return int(value, 0)


def open_mpu6050(i2c_bus, imu_addr):
    if i2c_bus is not None and imu_addr is not None:
        return MPU6050(i2c_bus, imu_addr)

    if i2c_bus is None:
        buses = None
    else:
        buses = [i2c_bus]

    if imu_addr is None:
        addresses = MPU6050_ADDRS
    else:
        addresses = (imu_addr,)

    found = detect_mpu6050(buses=buses, addresses=addresses)

    if not found:
        raise RuntimeError(
            "No MPU-6050 found at 0x68/0x69. Check SDA/SCL, 3.3V, GND, "
            "and run: python3 mpu6050_check.py"
        )

    bus, address, who = found[0]
    print(
        f"Auto-detected MPU-6050 on bus {bus}, "
        f"address 0x{address:02X}, WHO_AM_I=0x{who:02X}"
    )
    return MPU6050(bus, address)


def save_outputs(prefix, points, viewer_limit):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_prefix = Path(f"{prefix}_{timestamp}")
    ply_path = output_prefix.with_suffix(".ply")
    csv_path = output_prefix.with_suffix(".csv")
    html_path = output_prefix.with_suffix(".html")

    save_ply(ply_path, points)
    save_csv(csv_path, points)
    save_html_viewer(
        html_path,
        points,
        f"{output_prefix.name} ({len(points)} points)",
        viewer_limit,
    )

    print("")
    print(f"Saved {len(points)} sparse points.")
    print(f"PLY:  {ply_path}")
    print(f"CSV:  {csv_path}")
    print(f"HTML: {html_path}")
    print("")
    print("Open the HTML file in a browser to rotate the approximate 3D view.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build an approximate sparse 3D point cloud from STL-27L LiDAR "
            "and MPU-6050 gyro yaw."
        )
    )
    parser.add_argument("--lidar-port", default=LIDAR_PORT)
    parser.add_argument("--lidar-baud", type=int, default=LIDAR_BAUD)
    parser.add_argument("--i2c-bus", type=parse_auto_int, default=None)
    parser.add_argument(
        "--imu-addr",
        type=parse_auto_int,
        default=None,
    )
    parser.add_argument("--yaw-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--roll-axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--pitch-axis", choices=("x", "y", "z"), default="y")
    parser.add_argument("--roll-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--pitch-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--yaw-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--accel-alpha", type=float, default=0.98)
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--calibration-sec", type=float, default=3.0)
    parser.add_argument("--sample-hz", type=float, default=25.0)
    parser.add_argument("--angle-offset-deg", type=float, default=0.0)
    parser.add_argument("--map-roll-deg", type=float, default=0.0)
    parser.add_argument("--map-pitch-deg", type=float, default=0.0)
    parser.add_argument("--map-yaw-deg", type=float, default=0.0)
    parser.add_argument("--voxel-mm", type=float, default=50.0)
    parser.add_argument("--max-range-mm", type=int, default=6000)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--viewer-max-points", type=int, default=12000)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    args = parser.parse_args()

    if args.duration_sec <= 0:
        raise SystemExit("--duration-sec must be greater than zero")

    if args.calibration_sec < 0.5:
        raise SystemExit("--calibration-sec should be at least 0.5")

    if args.voxel_mm <= 0:
        raise SystemExit("--voxel-mm must be greater than zero")

    if args.max_points <= 0:
        raise SystemExit("--max-points must be greater than zero")

    if not 0.0 <= args.accel_alpha <= 1.0:
        raise SystemExit("--accel-alpha must be between 0.0 and 1.0")

    print_setup(args)

    imu = open_mpu6050(args.i2c_bus, args.imu_addr)
    lidar = STL27LReader(args.lidar_port, args.lidar_baud, point_max_age_sec=0.2)
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
    voxel_points = {}

    try:
        imu.initialize()
        who = imu.who_am_i()
        print(f"MPU-6050 WHO_AM_I: 0x{who:02X}")

        if who not in (0x68, 0x70):
            print("Warning: WHO_AM_I is unusual for MPU-6050. Check address/wiring.")

        orientation.calibrate(args.calibration_sec)
        lidar.start()

        print("")
        print("Start rotating the LiDAR + IMU now.")
        print("Press Ctrl-C to stop early and save what was captured.")
        print("")

        start_time = time.time()
        last_print = 0.0
        sleep_sec = 1.0 / max(1.0, args.sample_hz)

        while True:
            now = time.time()
            elapsed = now - start_time
            roll_deg, pitch_deg, yaw_deg = orientation.update()
            lidar_points = lidar.get_points()
            added = add_lidar_points(
                voxel_points,
                lidar_points,
                roll_deg,
                pitch_deg,
                yaw_deg,
                args.angle_offset_deg,
                args.map_roll_deg,
                args.map_pitch_deg,
                args.map_yaw_deg,
                args.voxel_mm,
                args.max_range_mm,
                args.max_points,
            )

            if now - last_print >= 1.0:
                last_print = now
                print(
                    f"time={elapsed:5.1f}s "
                    f"rpy=({roll_deg:6.1f}, {pitch_deg:6.1f}, {yaw_deg:7.1f}) deg "
                    f"live_points={len(lidar_points):3d} saved={len(voxel_points):5d} "
                    f"added={added:3d}"
                )

            if elapsed >= args.duration_sec or len(voxel_points) >= args.max_points:
                break

            time.sleep(sleep_sec)

    except KeyboardInterrupt:
        print("\nStopping early and saving captured points.")

    finally:
        lidar.close()
        imu.close()

    save_outputs(
        args.output_prefix,
        list(voxel_points.values()),
        args.viewer_max_points,
    )


if __name__ == "__main__":
    main()
