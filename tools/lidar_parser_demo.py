import argparse
import serial
import struct


FRAME_HEADER = 0x54
FRAME_VERLEN = 0x2C
POINTS_PER_FRAME = 12
FRAME_LEN = 47


def parse_frame(frame):
    if len(frame) != FRAME_LEN:
        return None

    if frame[0] != FRAME_HEADER or frame[1] != FRAME_VERLEN:
        return None

    speed = struct.unpack_from("<H", frame, 2)[0]
    start_angle = struct.unpack_from("<H", frame, 4)[0] / 100.0

    points = []
    offset = 6

    for _ in range(POINTS_PER_FRAME):
        distance_mm = struct.unpack_from("<H", frame, offset)[0]
        confidence = frame[offset + 2]
        points.append((distance_mm, confidence))
        offset += 3

    end_angle = struct.unpack_from("<H", frame, offset)[0] / 100.0
    timestamp = struct.unpack_from("<H", frame, offset + 2)[0]

    angle_diff = end_angle - start_angle
    if angle_diff < 0:
        angle_diff += 360.0

    parsed_points = []

    for i, (distance_mm, confidence) in enumerate(points):
        angle = start_angle + angle_diff * i / (POINTS_PER_FRAME - 1)
        angle = angle % 360.0

        parsed_points.append({
            "angle_deg": angle,
            "distance_mm": distance_mm,
            "confidence": confidence,
        })

    return {
        "speed": speed,
        "start_angle": start_angle,
        "end_angle": end_angle,
        "timestamp": timestamp,
        "points": parsed_points,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Parse STL-27L LiDAR frames."
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
    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)
    buffer = bytearray()

    print("Parsing LiDAR data.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            buffer += ser.read(256)

            while len(buffer) >= FRAME_LEN:
                if buffer[0] != FRAME_HEADER:
                    buffer.pop(0)
                    continue

                if buffer[1] != FRAME_VERLEN:
                    buffer.pop(0)
                    continue

                frame = bytes(buffer[:FRAME_LEN])
                buffer = buffer[FRAME_LEN:]

                result = parse_frame(frame)
                if result is None:
                    continue

                valid_points = [
                    point for point in result["points"]
                    if point["distance_mm"] > 0 and point["confidence"] > 0
                ]

                if not valid_points:
                    continue

                print(
                    f"speed={result['speed']} "
                    f"angle={result['start_angle']:.2f}->{result['end_angle']:.2f} "
                    f"points={len(valid_points)}"
                )

                for point in valid_points[:3]:
                    print(
                        f"  angle={point['angle_deg']:.2f} deg, "
                        f"distance={point['distance_mm']} mm, "
                        f"confidence={point['confidence']}"
                    )

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        ser.close()


if __name__ == "__main__":
    main()
