import argparse
import math
import sys
import time

try:
    import cv2
    import numpy as np
except ImportError as exc:
    print(f"Missing live-view dependency: {exc}")
    print("Install OpenCV on Jetson with: sudo apt install -y python3-opencv")
    sys.exit(1)

from lidar_stl27l import LIDAR_BAUD, LIDAR_PORT, STL27LReader


WINDOW_NAME = "STL-27L Live Scan"


def normalize_signed_degrees(angle_deg):
    wrapped = (angle_deg + 180.0) % 360.0 - 180.0

    if wrapped == -180.0:
        return 180.0

    return wrapped


def oriented_angle(point, angle_offset_deg):
    return normalize_signed_degrees(point["angle_deg"] + angle_offset_deg)


def point_to_pixel(point, center, pixels_per_mm, angle_offset_deg):
    angle_rad = math.radians(oriented_angle(point, angle_offset_deg))
    radius_px = point["distance_mm"] * pixels_per_mm

    x = int(center[0] + math.sin(angle_rad) * radius_px)
    y = int(center[1] - math.cos(angle_rad) * radius_px)

    return x, y


def draw_grid(image, center, max_range_mm, pixels_per_mm, front_half_angle_deg):
    h, w = image.shape[:2]

    for distance_mm in range(1000, max_range_mm + 1, 1000):
        radius_px = int(distance_mm * pixels_per_mm)
        cv2.circle(image, center, radius_px, (55, 55, 55), 1)
        label_pos = (center[0] + 6, max(18, center[1] - radius_px - 4))
        cv2.putText(
            image,
            f"{distance_mm // 1000}m",
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )

    cv2.line(image, (center[0], 0), (center[0], h), (70, 70, 70), 1)
    cv2.line(image, (0, center[1]), (w, center[1]), (70, 70, 70), 1)

    for angle_deg in (-front_half_angle_deg, front_half_angle_deg):
        angle_rad = math.radians(angle_deg)
        end = (
            int(center[0] + math.sin(angle_rad) * max_range_mm * pixels_per_mm),
            int(center[1] - math.cos(angle_rad) * max_range_mm * pixels_per_mm),
        )
        cv2.line(image, center, end, (70, 95, 125), 1)

    cv2.circle(image, center, 5, (210, 210, 210), -1)
    cv2.putText(
        image,
        "FRONT",
        (center[0] - 30, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )


def draw_points(image, points, center, pixels_per_mm, max_range_mm, angle_offset_deg):
    visible = [
        point for point in points
        if point["distance_mm"] <= max_range_mm
    ]

    for point in visible:
        x, y = point_to_pixel(point, center, pixels_per_mm, angle_offset_deg)
        distance_ratio = min(1.0, point["distance_mm"] / max_range_mm)
        color = (
            int(40 + 160 * distance_ratio),
            int(220 - 120 * distance_ratio),
            int(255 - 220 * distance_ratio),
        )
        cv2.circle(image, (x, y), 3, color, -1)

    if visible:
        nearest = min(visible, key=lambda point: point["distance_mm"])
        x, y = point_to_pixel(nearest, center, pixels_per_mm, angle_offset_deg)
        cv2.circle(image, (x, y), 8, (0, 0, 255), 2)
        cv2.line(image, center, (x, y), (0, 0, 180), 1)
        cv2.putText(
            image,
            (
                f"{nearest['distance_mm']} mm @ raw {nearest['angle_deg']:.1f} deg "
                f"front {oriented_angle(nearest, angle_offset_deg):.1f} deg"
            ),
            (18, image.shape[0] - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return visible


def draw_status(
    image,
    points,
    visible,
    lidar_port,
    lidar_baud,
    angle_offset_deg,
    front_half_angle_deg,
):
    now = time.time()
    fresh_points = [
        point for point in points
        if now - point["time"] <= 0.25
    ]
    front_points = [
        point for point in visible
        if abs(oriented_angle(point, angle_offset_deg)) <= front_half_angle_deg
    ]

    status = (
        f"{lidar_port} @ {lidar_baud} | live bins: {len(points)} | "
        f"offset: {angle_offset_deg:.0f} deg"
    )
    cv2.putText(
        image,
        status,
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (215, 215, 215),
        1,
        cv2.LINE_AA,
    )

    if fresh_points:
        freshness = "data live"
        freshness_color = (80, 220, 80)
    else:
        freshness = "waiting for fresh data"
        freshness_color = (0, 180, 255)

    cv2.putText(
        image,
        freshness,
        (18, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        freshness_color,
        1,
        cv2.LINE_AA,
    )

    if front_points:
        front = min(front_points, key=lambda point: point["distance_mm"])
        text = (
            f"front +/-{front_half_angle_deg:.0f}: {front['distance_mm']} mm "
            f"raw={front['angle_deg']:.1f} front={oriented_angle(front, angle_offset_deg):.1f}"
        )
    else:
        text = f"front +/-{front_half_angle_deg:.0f}: clear/no points"

    cv2.putText(
        image,
        text,
        (18, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (215, 215, 215),
        1,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Draw a non-persistent real-time STL-27L LiDAR scan."
    )
    parser.add_argument("--lidar-port", default=LIDAR_PORT)
    parser.add_argument("--lidar-baud", type=int, default=LIDAR_BAUD)
    parser.add_argument("--range-mm", type=int, default=6000)
    parser.add_argument("--size", type=int, default=800)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--point-age-sec", type=float, default=0.25)
    parser.add_argument("--angle-offset-deg", type=float, default=180.0)
    parser.add_argument("--front-half-angle-deg", type=float, default=90.0)
    args = parser.parse_args()

    print("=== STL-27L Live Scan ===")
    print("This is a live scan view, not a saved map.")
    print("It does not connect to the flight controller.")
    print("Press q or Esc to quit.")
    print(f"LiDAR port: {args.lidar_port} @ {args.lidar_baud}")
    print(f"Display target: {args.fps} FPS")
    print(f"Live point age: {args.point_age_sec:.2f}s")
    print(f"Angle offset: {args.angle_offset_deg:.1f} deg")
    print(f"Front sector: +/-{args.front_half_angle_deg:.1f} deg")

    lidar = STL27LReader(
        args.lidar_port,
        args.lidar_baud,
        point_max_age_sec=args.point_age_sec,
    )
    lidar.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.size, args.size)

    center = (args.size // 2, args.size // 2)
    pixels_per_mm = (args.size * 0.46) / args.range_mm
    background = np.zeros((args.size, args.size, 3), dtype=np.uint8)
    draw_grid(
        background,
        center,
        args.range_mm,
        pixels_per_mm,
        args.front_half_angle_deg,
    )
    wait_ms = max(1, int(1000 / max(1, args.fps)))

    try:
        while True:
            image = background.copy()

            points = lidar.get_points()
            visible = draw_points(
                image,
                points,
                center,
                pixels_per_mm,
                args.range_mm,
                args.angle_offset_deg,
            )
            draw_status(
                image,
                points,
                visible,
                args.lidar_port,
                args.lidar_baud,
                args.angle_offset_deg,
                args.front_half_angle_deg,
            )

            cv2.imshow(WINDOW_NAME, image)

            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (27, ord("q")):
                break

    except KeyboardInterrupt:
        print("\nStopping.")

    finally:
        lidar.close()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
