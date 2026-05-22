import argparse
import math
import time

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        f"Missing live-view dependency: {exc}\n"
        "Install OpenCV with: sudo apt install -y python3-opencv"
    )

from lidar_3d_scan_mapper_mpu6050 import (
    MPUOrientationEstimator,
    add_lidar_points,
    open_mpu6050,
    parse_auto_int,
    save_outputs,
)
from lidar_stl27l import LIDAR_BAUD, LIDAR_PORT, STL27LReader


WINDOW_NAME = "Live 3D LiDAR + MPU-6050 Map"
DEFAULT_OUTPUT_PREFIX = "lidar_3d_live_imu_scan"


def rotate_view(x, y, z, view_yaw_deg, view_pitch_deg):
    yaw = math.radians(view_yaw_deg)
    pitch = math.radians(view_pitch_deg)

    cy = math.cos(yaw)
    sy = math.sin(yaw)
    x1 = x * cy - y * sy
    y1 = x * sy + y * cy
    z1 = z

    cp = math.cos(pitch)
    sp = math.sin(pitch)
    x2 = x1
    y2 = y1 * cp - z1 * sp
    z2 = y1 * sp + z1 * cp

    return x2, y2, z2


def project_point(point, center, scale, max_range_mm, view_yaw_deg, view_pitch_deg):
    x, depth, z = rotate_view(
        point["x"],
        point["y"],
        point["z"],
        view_yaw_deg,
        view_pitch_deg,
    )

    camera_distance = max_range_mm * 4.0
    perspective = camera_distance / max(1.0, camera_distance - depth)
    px = int(center[0] + x * scale * perspective)
    py = int(center[1] - z * scale * perspective)
    return px, py, depth


def draw_axes(image, center, scale, max_range_mm, view_yaw_deg, view_pitch_deg):
    axis_len = max_range_mm * 0.35
    axes = (
        ((axis_len, 0, 0), (60, 80, 255), "X"),
        ((0, axis_len, 0), (80, 220, 80), "Y"),
        ((0, 0, axis_len), (255, 140, 80), "Z"),
    )

    origin = {"x": 0.0, "y": 0.0, "z": 0.0}
    ox, oy, _depth = project_point(
        origin,
        center,
        scale,
        max_range_mm,
        view_yaw_deg,
        view_pitch_deg,
    )

    for (x, y, z), color, label in axes:
        end_point = {"x": x, "y": y, "z": z}
        ex, ey, _depth = project_point(
            end_point,
            center,
            scale,
            max_range_mm,
            view_yaw_deg,
            view_pitch_deg,
        )
        cv2.line(image, (ox, oy), (ex, ey), color, 2)
        cv2.putText(
            image,
            label,
            (ex + 5, ey + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )


def visible_points(points, draw_max_points):
    if len(points) <= draw_max_points:
        return points

    step = max(1, len(points) // draw_max_points)
    return points[::step][:draw_max_points]


def render_cloud(
    voxel_points,
    roll_deg,
    pitch_deg,
    yaw_deg,
    live_lidar_count,
    added,
    args,
    view_yaw_deg,
    view_pitch_deg,
    zoom,
):
    image = np.zeros((args.window_size, args.window_size, 3), dtype=np.uint8)
    center = (args.window_size // 2, args.window_size // 2)
    scale = (args.window_size * 0.42 / args.max_range_mm) * zoom

    points = list(voxel_points.values())
    draw_points = visible_points(points, args.draw_max_points)
    projected = []

    for point in draw_points:
        px, py, depth = project_point(
            point,
            center,
            scale,
            args.max_range_mm,
            view_yaw_deg,
            view_pitch_deg,
        )
        projected.append((depth, px, py, point))

    projected.sort(key=lambda item: item[0])
    draw_axes(
        image,
        center,
        scale,
        args.max_range_mm,
        view_yaw_deg,
        view_pitch_deg,
    )

    for _depth, px, py, point in projected:
        if 0 <= px < args.window_size and 0 <= py < args.window_size:
            color = (point["b"], point["g"], point["r"])
            cv2.circle(image, (px, py), args.point_size, color, -1)

    lines = (
        "Live 3D LiDAR + MPU-6050 map",
        f"saved sparse points: {len(points)} / {args.max_points}",
        f"live lidar bins: {live_lidar_count} | added this frame: {added}",
        f"imu roll/pitch/yaw: {roll_deg:.1f}, {pitch_deg:.1f}, {yaw_deg:.1f} deg",
        f"voxel filter: {args.voxel_mm:.0f} mm | range: {args.max_range_mm} mm",
        "q: save+quit | arrows/a,d/w,s: view | +/-: zoom",
    )

    y = 24
    for line in lines:
        cv2.putText(
            image,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        y += 24

    return image


def handle_key(key, view_yaw_deg, view_pitch_deg, zoom):
    if key in (ord("a"), 81):
        view_yaw_deg -= 5
    elif key in (ord("d"), 83):
        view_yaw_deg += 5
    elif key in (ord("w"), 82):
        view_pitch_deg += 5
    elif key in (ord("s"), 84):
        view_pitch_deg -= 5
    elif key in (ord("+"), ord("=")):
        zoom = min(8.0, zoom * 1.1)
    elif key in (ord("-"), ord("_")):
        zoom = max(0.1, zoom / 1.1)
    elif key == ord("r"):
        view_yaw_deg = -35.0
        view_pitch_deg = -25.0
        zoom = 1.0

    view_pitch_deg = max(-85.0, min(85.0, view_pitch_deg))
    return view_yaw_deg, view_pitch_deg, zoom


def print_setup(args):
    print("=== Real-Time 3D LiDAR + MPU-6050 Mapper ===")
    print("This script does not touch the flight controller or motors.")
    print("")
    print("Setup:")
    print("- Mount the MPU-6050 rigidly to the LiDAR.")
    print("- Hold/mount the LiDAR scan plane vertical.")
    print("- Keep LiDAR + IMU still during calibration, then rotate together.")
    print("")
    print(f"LiDAR:      {args.lidar_port} @ {args.lidar_baud}")
    print("MPU-6050:   auto-detect unless --i2c-bus/--imu-addr are set")
    print(
        "IMU axes:   "
        f"roll={args.roll_axis}*{args.roll_sign:g}, "
        f"pitch={args.pitch_axis}*{args.pitch_sign:g}, "
        f"yaw={args.yaw_axis}*{args.yaw_sign:g}"
    )
    print(f"Duration:   {args.duration_sec:.1f}s")
    print(f"Voxel:      {args.voxel_mm:.1f} mm")
    print(f"Max points: {args.max_points}")
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="Show and save a sparse real-time 3D LiDAR + MPU-6050 map."
    )
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
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--calibration-sec", type=float, default=3.0)
    parser.add_argument("--sample-hz", type=float, default=25.0)
    parser.add_argument("--display-fps", type=float, default=20.0)
    parser.add_argument("--angle-offset-deg", type=float, default=0.0)
    parser.add_argument("--map-roll-deg", type=float, default=0.0)
    parser.add_argument("--map-pitch-deg", type=float, default=0.0)
    parser.add_argument("--map-yaw-deg", type=float, default=0.0)
    parser.add_argument("--voxel-mm", type=float, default=60.0)
    parser.add_argument("--max-range-mm", type=int, default=6000)
    parser.add_argument("--max-points", type=int, default=15000)
    parser.add_argument("--draw-max-points", type=int, default=8000)
    parser.add_argument("--viewer-max-points", type=int, default=12000)
    parser.add_argument("--window-size", type=int, default=900)
    parser.add_argument("--point-size", type=int, default=2)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    args = parser.parse_args()

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

    view_yaw_deg = -35.0
    view_pitch_deg = -25.0
    zoom = 1.0
    last_render = 0.0
    last_added = 0

    try:
        imu.initialize()
        who = imu.who_am_i()
        print(f"MPU-6050 WHO_AM_I: 0x{who:02X}")

        if who not in (0x68, 0x70):
            print("Warning: unusual WHO_AM_I. Check wiring if values look wrong.")

        orientation.calibrate(args.calibration_sec)
        lidar.start()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, args.window_size, args.window_size)

        print("")
        print("Start rotating LiDAR + IMU together now.")
        print("Press q in the window or Ctrl-C to save and quit.")
        print("")

        start_time = time.time()
        sample_sleep = 1.0 / max(1.0, args.sample_hz)
        render_interval = 1.0 / max(1.0, args.display_fps)

        while True:
            now = time.time()
            elapsed = now - start_time
            roll_deg, pitch_deg, yaw_deg = orientation.update()
            lidar_points = lidar.get_points()
            last_added = add_lidar_points(
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

            if now - last_render >= render_interval:
                last_render = now
                image = render_cloud(
                    voxel_points,
                    roll_deg,
                    pitch_deg,
                    yaw_deg,
                    len(lidar_points),
                    last_added,
                    args,
                    view_yaw_deg,
                    view_pitch_deg,
                    zoom,
                )
                cv2.imshow(WINDOW_NAME, image)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

                view_yaw_deg, view_pitch_deg, zoom = handle_key(
                    key,
                    view_yaw_deg,
                    view_pitch_deg,
                    zoom,
                )

            if elapsed >= args.duration_sec:
                break

            if len(voxel_points) >= args.max_points:
                print("Reached max point limit.")
                break

            time.sleep(sample_sleep)

    except KeyboardInterrupt:
        print("\nStopping early and saving captured points.")

    finally:
        lidar.close()
        imu.close()
        cv2.destroyAllWindows()

    save_outputs(
        args.output_prefix,
        list(voxel_points.values()),
        args.viewer_max_points,
    )


if __name__ == "__main__":
    main()
