import argparse
import serial
import struct
import threading
import time


FC_PORT = "/dev/ttyTHS1"
FC_BAUD = 115200
LIDAR_PORT = "/dev/ttyUSB0"
LIDAR_BAUD = 921600
LOOP_HZ = 50

MSP_SET_RAW_RC = 200

FRAME_HEADER = 0x54
FRAME_VERLEN = 0x2C
FRAME_LEN = 47
POINTS_PER_FRAME = 12

MIN_DISTANCE_MM = 30
MAX_DISTANCE_MM = 25000
MIN_CONFIDENCE = 30
POINT_MAX_AGE_SEC = 1.0


def disarm_rc():
    return [1500, 1500, 1000, 1500, 1000, 1000, 1000, 1000]


class MSP:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=1)
        self.ser.reset_input_buffer()

    def send_rc(self, channels):
        data = bytearray()

        for ch in channels:
            data.extend(struct.pack("<H", int(ch)))

        size = len(data)
        checksum = size ^ MSP_SET_RAW_RC

        for b in data:
            checksum ^= b

        packet = bytearray(b"$M<")
        packet.append(size)
        packet.append(MSP_SET_RAW_RC)
        packet.extend(data)
        packet.append(checksum)

        self.ser.write(packet)

    def close(self):
        self.ser.close()


class STL27LReader:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=1)
        self.buffer = bytearray()
        self.points = [None] * 360
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def parse_frame(self, frame):
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

            if (
                MIN_DISTANCE_MM <= distance_mm <= MAX_DISTANCE_MM
                and confidence >= MIN_CONFIDENCE
            ):
                points.append(
                    {
                        "angle_deg": angle,
                        "distance_mm": distance_mm,
                        "confidence": confidence,
                        "time": time.time(),
                    }
                )

        return points

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.read_loop, daemon=True)
        self.thread.start()

    def read_loop(self):
        while self.running:
            data = self.ser.read(512)

            if not data:
                continue

            self.buffer += data

            while len(self.buffer) >= FRAME_LEN:
                if self.buffer[0] != FRAME_HEADER:
                    self.buffer.pop(0)
                    continue

                if self.buffer[1] != FRAME_VERLEN:
                    self.buffer.pop(0)
                    continue

                frame = bytes(self.buffer[:FRAME_LEN])
                self.buffer = self.buffer[FRAME_LEN:]

                parsed = self.parse_frame(frame)

                if parsed:
                    with self.lock:
                        for point in parsed:
                            angle_bin = int(round(point["angle_deg"])) % 360
                            self.points[angle_bin] = point

                        now = time.time()
                        for idx, point in enumerate(self.points):
                            if point and now - point["time"] > POINT_MAX_AGE_SEC:
                                self.points[idx] = None

    def get_nearest(self, front_only=False):
        with self.lock:
            now = time.time()

            for idx, point in enumerate(self.points):
                if point and now - point["time"] > POINT_MAX_AGE_SEC:
                    self.points[idx] = None

            points = [point for point in self.points if point]

        if not points:
            return None

        if front_only:
            filtered = []

            for point in points:
                angle = point["angle_deg"]

                if angle <= 30 or angle >= 330:
                    filtered.append(point)

            points = filtered

        if not points:
            return None

        return min(points, key=lambda point: point["distance_mm"])

    def close(self):
        self.running = False

        if self.thread:
            self.thread.join(timeout=1.0)

        self.ser.close()


def print_lidar_status(lidar):
    nearest = lidar.get_nearest(front_only=False)
    front = lidar.get_nearest(front_only=True)

    print("--- Status ---")

    if nearest:
        print(
            f"LiDAR nearest: {nearest['distance_mm']} mm "
            f"at {nearest['angle_deg']:.1f} deg "
            f"conf={nearest['confidence']}"
        )
    else:
        print("LiDAR nearest: no valid points")

    if front:
        print(
            f"LiDAR front +/-30 deg: {front['distance_mm']} mm "
            f"at {front['angle_deg']:.1f} deg "
            f"conf={front['confidence']}"
        )
    else:
        print("LiDAR front +/-30 deg: clear or no valid points")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fc-port", default=FC_PORT)
    parser.add_argument("--fc-baud", type=int, default=FC_BAUD)
    parser.add_argument("--lidar-port", default=LIDAR_PORT)
    parser.add_argument("--lidar-baud", type=int, default=LIDAR_BAUD)
    args = parser.parse_args()

    print("=== FC + LiDAR Monitor ===")
    print("SAFETY: REMOVE PROPELLERS.")
    print("This script does NOT arm.")
    print("It only sends disarm RC and reads LiDAR.")
    print(f"FC port:    {args.fc_port} @ {args.fc_baud}")
    print(f"LiDAR port: {args.lidar_port} @ {args.lidar_baud}")
    print("")

    msp = MSP(args.fc_port, args.fc_baud)
    lidar = STL27LReader(args.lidar_port, args.lidar_baud)
    lidar.start()

    last_print = 0

    try:
        while True:
            msp.send_rc(disarm_rc())

            now = time.time()

            if now - last_print >= 1.0:
                last_print = now
                print_lidar_status(lidar)

            time.sleep(1.0 / LOOP_HZ)

    except KeyboardInterrupt:
        print("\nStopping. Sending disarm RC...")

        for _ in range(20):
            msp.send_rc(disarm_rc())
            time.sleep(0.02)

    finally:
        lidar.close()
        msp.close()
        print("Stopped safely.")


if __name__ == "__main__":
    main()
