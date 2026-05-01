import argparse
import math
import serial
import struct
import threading
import time

import matplotlib.pyplot as plt


FRAME_HEADER = 0x54
FRAME_VERLEN = 0x2C
POINTS_PER_FRAME = 12
FRAME_LEN = 47

scan_points = {}
lock = threading.Lock()


def parse_frame(frame, min_confidence):
    if len(frame) != FRAME_LEN:
        return []

    if frame[0] != FRAME_HEADER or frame[1] != FRAME_VERLEN:
        return []

    start_angle = struct.unpack_from("<H", frame, 4)[0] / 100.0
    end_angle = struct.unpack_from("<H", frame, 42)[0] / 100.0

    angle_diff = end_angle - start_angle
    if angle_diff < 0:
        angle_diff += 360.0

    points = []
    offset = 6

    for i in range(POINTS_PER_FRAME):
        distance_mm = struct.unpack_from("<H", frame, offset)[0]
        confidence = frame[offset + 2]
        offset += 3

        angle = start_angle + angle_diff * i / (POINTS_PER_FRAME - 1)
        angle = angle % 360.0

        if 30 <= distance_mm <= 25000 and confidence >= min_confidence:
            points.append((angle, distance_mm, confidence))

    return points


def read_lidar(port, baud, min_confidence):
    ser = serial.Serial(port, baud, timeout=1)
    buffer = bytearray()

    print("Reading LiDAR.")
    print("Close the plot window or press Ctrl+C to stop.")

    while True:
        buffer += ser.read(512)

        while len(buffer) >= FRAME_LEN:
            if buffer[0] != FRAME_HEADER:
                buffer.pop(0)
                continue

            if buffer[1] != FRAME_VERLEN:
                buffer.pop(0)
                continue

            frame = bytes(buffer[:FRAME_LEN])
            buffer = buffer[FRAME_LEN:]

            points = parse_frame(frame, min_confidence)

            with lock:
                for angle, distance_mm, confidence in points:
                    angle_bin = int(angle)
                    scan_points[angle_bin] = (
                        angle,
                        distance_mm,
                        confidence,
                        time.time(),
                    )


def main():
    parser = argparse.ArgumentParser(
        description="Display STL-27L LiDAR points in 2D."
    )
    parser.add_argument(
        "--port",
        required=True,
        help="Serial port, for example /dev/cu.usbserial-0001",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=921600,
        help="Serial baud rate",
    )
    parser.add_argument(
        "--range",
        type=float,
        default=1.5,
        help="Visible range in meters",
    )
    parser.add_argument(
        "--confidence",
        type=int,
        default=30,
        help="Minimum confidence value",
    )
    parser.add_argument(
        "--keep",
        type=float,
        default=1.0,
        help="Keep recent points for this many seconds",
    )
    args = parser.parse_args()

    thread = threading.Thread(
        target=read_lidar,
        args=(args.port, args.baud, args.confidence),
        daemon=True,
    )
    thread.start()

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    while plt.fignum_exists(fig.number):
        xs = []
        ys = []
        now = time.time()

        with lock:
            for angle, distance_mm, confidence, point_time in list(scan_points.values()):
                if now - point_time > args.keep:
                    continue

                radius_m = distance_mm / 1000.0
                radians = math.radians(angle)

                x = radius_m * math.cos(radians)
                y = radius_m * math.sin(radians)

                xs.append(x)
                ys.append(y)

        ax.clear()
        ax.scatter(xs, ys, s=14)
        ax.scatter([0], [0], s=40, marker="x")

        ax.set_title(f"STL-27L LiDAR 2D Scan | View +/- {args.range} m")
        ax.set_xlabel("X / meters")
        ax.set_ylabel("Y / meters")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-args.range, args.range)
        ax.set_ylim(-args.range, args.range)
        ax.grid(True)

        plt.pause(0.05)

    print("Viewer closed.")


if __name__ == "__main__":
    main()
