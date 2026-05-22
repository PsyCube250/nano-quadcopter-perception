import struct
import threading
import time

import serial
from serial import SerialException


LIDAR_PORT = "/dev/ttyUSB0"
LIDAR_BAUD = 921600

FRAME_HEADER = 0x54
FRAME_VERLEN = 0x2C
FRAME_LEN = 47
POINTS_PER_FRAME = 12

MIN_DISTANCE_MM = 30
MAX_DISTANCE_MM = 25000
MIN_CONFIDENCE = 30
POINT_MAX_AGE_SEC = 1.0


class STL27LReader:
    def __init__(
        self,
        port=LIDAR_PORT,
        baud=LIDAR_BAUD,
        point_max_age_sec=POINT_MAX_AGE_SEC,
    ):
        self.ser = serial.Serial(port, baud, timeout=1)
        self.buffer = bytearray()
        self.points = [None] * 360
        self.point_max_age_sec = point_max_age_sec
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
            try:
                data = self.ser.read(512)
            except SerialException as exc:
                print(f"LiDAR serial read stopped: {exc}")
                self.running = False
                break

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

                        self._drop_old_points()

    def _drop_old_points(self):
        now = time.time()
        for idx, point in enumerate(self.points):
            if point and now - point["time"] > self.point_max_age_sec:
                self.points[idx] = None

    def get_points(self):
        with self.lock:
            self._drop_old_points()
            return [point for point in self.points if point]

    def get_nearest(self, front_only=False):
        with self.lock:
            self._drop_old_points()
            points = [point for point in self.points if point]

        if not points:
            return None

        if front_only:
            points = [
                point for point in points
                if point["angle_deg"] <= 30 or point["angle_deg"] >= 330
            ]

        if not points:
            return None

        return min(points, key=lambda point: point["distance_mm"])

    def close(self):
        self.running = False

        if self.thread:
            self.thread.join(timeout=1.0)

        self.ser.close()
