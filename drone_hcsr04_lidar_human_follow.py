import argparse
import os
import site
import sys
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from lidar_3d_scan_mapper_mpu6050 import (
    MPUOrientationEstimator,
    open_mpu6050,
    parse_auto_int,
)
from lidar_imu_obstacle_avoidance import (
    RC_CENTER,
    avoidance_targets,
    format_obstacle,
    nearest_obstacle,
    rc_channels,
)
from lidar_stl27l import LIDAR_BAUD, LIDAR_PORT, STL27LReader
from motor_lidar_speed_control_test import (
    FC_BAUD,
    FC_PORT,
    MSP,
    arm_rc,
    clamp,
    disarm_rc,
    print_status,
    send_for_duration,
    step_toward,
)


DEFAULT_MODEL = "yolov8s.pt"
PERSON_CLASS_ID = 0
DEFAULT_TARGET_HEIGHT_MM = 1000
DEFAULT_PERSON_STOP_DISTANCE_MM = 1500
DEFAULT_TARGET_BOX_AREA_FRACTION = 0.18
DEFAULT_HOVER_THROTTLE = 1100
DEFAULT_MIN_THROTTLE = 1000
DEFAULT_MAX_THROTTLE = 1150
DEFAULT_ALTITUDE_KP = 0.08
DEFAULT_APPROACH_PITCH_OFFSET = 70
DEFAULT_YAW_CENTER_KP = 180.0
DEFAULT_SEARCH_YAW_OFFSET = 80
DEFAULT_MAX_RC_OFFSET = 120
DEFAULT_RC_STEP = 8


def bootstrap_cuda_library_path():
    candidates = []

    for package_dir in [site.getusersitepackages(), *site.getsitepackages()]:
        cuda_lib_dir = Path(package_dir) / "nvidia" / "cu12" / "lib"

        if cuda_lib_dir.exists():
            candidates.append(str(cuda_lib_dir))

    if not candidates:
        return

    current_paths = [
        path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":")
        if path
    ]
    missing_paths = [path for path in candidates if path not in current_paths]

    if not missing_paths or os.environ.get("JETSON_CUDA_LIBPATH_BOOTSTRAPPED") == "1":
        return

    os.environ["LD_LIBRARY_PATH"] = ":".join(missing_paths + current_paths)
    os.environ["JETSON_CUDA_LIBPATH_BOOTSTRAPPED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

@dataclass
class HumanTarget:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    frame_width: int
    frame_height: int

    @property
    def width(self):
        return max(1, self.x2 - self.x1)

    @property
    def height(self):
        return max(1, self.y2 - self.y1)

    @property
    def center_x(self):
        return (self.x1 + self.x2) * 0.5

    @property
    def center_error_x(self):
        return (
            (self.center_x - self.frame_width * 0.5)
            / max(1.0, self.frame_width * 0.5)
        )

    @property
    def area_fraction(self):
        return (
            (self.width * self.height)
            / max(1.0, self.frame_width * self.frame_height)
        )


class HCSR04HeightSensor:
    def __init__(
        self,
        trigger_pin,
        echo_pin,
        gpio_mode="BOARD",
        timeout_sec=0.03,
        settle_sec=0.0002,
    ):
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self.gpio_mode = gpio_mode.upper()
        self.timeout_sec = timeout_sec
        self.settle_sec = settle_sec
        self.gpio = None

    def start(self):
        try:
            import Jetson.GPIO as GPIO
        except Exception as exc:
            raise RuntimeError(
                "Jetson.GPIO is required for the HC-SR04 height sensor. "
                "Install it on the Jetson, or use --sim-height-mm for a no-GPIO dry run."
            ) from exc

        self.gpio = GPIO

        if self.gpio_mode == "BOARD":
            mode = GPIO.BOARD
        elif self.gpio_mode == "BCM":
            mode = GPIO.BCM
        else:
            raise RuntimeError("--gpio-mode must be BOARD or BCM")

        GPIO.setmode(mode)
        GPIO.setup(self.trigger_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.echo_pin, GPIO.IN)
        time.sleep(0.05)

    def _wait_for_level(self, level, deadline):
        while time.perf_counter() < deadline:
            if self.gpio.input(self.echo_pin) == level:
                return time.perf_counter()

        return None

    def read_once_mm(self):
        deadline = time.perf_counter() + self.timeout_sec
        self.gpio.output(self.trigger_pin, self.gpio.LOW)
        time.sleep(self.settle_sec)
        self.gpio.output(self.trigger_pin, self.gpio.HIGH)
        time.sleep(0.00001)
        self.gpio.output(self.trigger_pin, self.gpio.LOW)

        pulse_start = self._wait_for_level(self.gpio.HIGH, deadline)

        if pulse_start is None:
            return None

        pulse_end = self._wait_for_level(self.gpio.LOW, deadline)

        if pulse_end is None:
            return None

        pulse_sec = pulse_end - pulse_start
        return pulse_sec * 171500.0

    def read_height_mm(self, samples=3):
        readings = []

        for _ in range(max(1, samples)):
            value = self.read_once_mm()

            if value is not None and 20 <= value <= 4500:
                readings.append(value)

            time.sleep(0.02)

        if not readings:
            return None

        return statistics.median(readings)

    def close(self):
        if self.gpio is not None:
            self.gpio.cleanup([self.trigger_pin, self.echo_pin])


class SimulatedHeightSensor:
    def __init__(self, height_mm):
        self.height_mm = height_mm

    def start(self):
        pass

    def read_height_mm(self, samples=1):
        del samples
        return self.height_mm

    def close(self):
        pass


class HumanDetector:
    def __init__(self, args):
        from ultralytics import YOLO
        from yolo_human_picam import ArgusCamera, available_device

        self.args = args
        self.camera = ArgusCamera(args.sensor_id, args.width, args.height, args.fps)
        self.model = YOLO(args.model)
        self.device = available_device(args.device)
        self.last_frame_time = None

    def start(self):
        self.camera.start()
        self.last_frame_time = time.time()

    def detect(self):
        frame = self.camera.read()

        if frame is None:
            if time.time() - self.last_frame_time > self.args.no_frame_timeout_sec:
                raise RuntimeError("No camera frames received from CAM0.")

            return None

        self.last_frame_time = time.time()
        results = self.model.predict(
            frame,
            imgsz=self.args.imgsz,
            conf=self.args.conf,
            classes=[PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )

        if not results or results[0].boxes is None:
            return None

        height, width = frame.shape[:2]
        targets = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])

            if cls_id != PERSON_CLASS_ID:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            area = max(0, x2 - x1) * max(0, y2 - y1)

            if area < self.args.min_box_area:
                continue

            targets.append(HumanTarget(x1, y1, x2, y2, conf, width, height))

        if not targets:
            return None

        return max(
            targets,
            key=lambda target: (target.confidence, target.area_fraction),
        )

    def close(self):
        self.camera.close()


class AltitudeController:
    def __init__(self, args):
        self.args = args
        self.current_throttle = args.hover_throttle

    def target_throttle(self, height_mm):
        if height_mm is None:
            return self.args.hover_throttle

        error_mm = self.args.target_height_mm - height_mm
        throttle = self.args.hover_throttle + error_mm * self.args.altitude_kp
        return int(
            round(
                clamp(
                    throttle,
                    self.args.min_throttle,
                    self.args.max_throttle,
                )
            )
        )

    def update(self, height_mm):
        target = self.target_throttle(height_mm)
        self.current_throttle = step_toward(
            self.current_throttle,
            target,
            self.args.throttle_step,
        )
        self.current_throttle = int(
            round(
                clamp(
                    self.current_throttle,
                    self.args.min_throttle,
                    self.args.max_throttle,
                )
            )
        )
        return self.current_throttle


def yaw_delta_deg(current, baseline):
    return abs(current - baseline)


def make_height_sensor(args):
    if args.sim_height_mm is not None:
        return SimulatedHeightSensor(args.sim_height_mm)

    return HCSR04HeightSensor(
        args.hcsr04_trigger_pin,
        args.hcsr04_echo_pin,
        gpio_mode=args.gpio_mode,
        timeout_sec=args.hcsr04_timeout_sec,
    )


def choose_state(state, height_mm, target, obstacle, search_start_yaw, yaw_deg, args):
    if state == "CLIMB":
        if (
            height_mm is not None
            and abs(args.target_height_mm - height_mm) <= args.height_deadband_mm
        ):
            return "SEARCH", yaw_deg

        return state, search_start_yaw

    if state == "SEARCH":
        if target is not None:
            return "APPROACH", search_start_yaw

        if yaw_delta_deg(yaw_deg, search_start_yaw) >= args.search_turn_degrees:
            return "HOVER", search_start_yaw

        return state, search_start_yaw

    if state == "APPROACH":
        if target is None:
            return "SEARCH", yaw_deg

        too_close_lidar = (
            obstacle is not None
            and obstacle.horizontal_mm <= args.person_stop_distance_mm
        )
        too_close_camera = target.area_fraction >= args.target_box_area_fraction

        if too_close_lidar or too_close_camera:
            return "HOVER", search_start_yaw

        return state, search_start_yaw

    if state == "HOVER" and target is not None:
        too_close_lidar = (
            obstacle is not None
            and obstacle.horizontal_mm
            <= args.person_stop_distance_mm + args.person_distance_deadband_mm
        )
        too_close_camera = target.area_fraction >= args.target_box_area_fraction

        if not too_close_lidar and not too_close_camera:
            return "APPROACH", search_start_yaw

    return state, search_start_yaw


def command_for_state(state, target, obstacle, search_start_yaw, yaw_deg, args):
    del search_start_yaw
    roll = RC_CENTER
    pitch = RC_CENTER
    yaw = RC_CENTER

    avoid_roll, avoid_pitch, avoid_state, avoid_strength = avoidance_targets(
        obstacle,
        args,
    )

    if avoid_state == "AVOID":
        return avoid_roll, avoid_pitch, RC_CENTER, avoid_state, avoid_strength

    if state == "SEARCH":
        yaw = RC_CENTER + args.search_yaw_sign * args.search_yaw_offset
        return roll, pitch, yaw, "SEARCH", 0.0

    if state == "APPROACH" and target is not None:
        center_error = target.center_error_x
        yaw_offset = clamp(
            center_error * args.yaw_center_kp,
            -args.max_rc_offset,
            args.max_rc_offset,
        )
        yaw = int(round(RC_CENTER + args.yaw_sign_control * yaw_offset))

        if abs(center_error) <= args.approach_center_deadband:
            pitch = int(
                round(
                    RC_CENTER
                    + args.approach_pitch_sign * args.approach_pitch_offset
                )
            )

        return roll, pitch, yaw, "TRACK", 0.0

    return roll, pitch, yaw, "HOLD", 0.0


def format_height(height_mm):
    if height_mm is None:
        return "none"

    return f"{height_mm:.0f} mm"


def format_target(target):
    if target is None:
        return "none"

    return (
        f"conf={target.confidence:.2f} "
        f"center_error_x={target.center_error_x:.2f} "
        f"area={target.area_fraction:.3f}"
    )


def confirm_or_exit(args):
    print("=== HC-SR04 + LiDAR + YOLO Human Follow Test ===")
    print("This sends MSP RC commands for altitude hold, search yaw, and approach.")
    print("Use dry-run first. For any live test, remove propellers until RC directions are verified.")
    print("")
    print(f"FC:              {args.fc_port} @ {args.fc_baud}")
    print(f"LiDAR:           {args.lidar_port} @ {args.lidar_baud}")
    print(f"Camera:          CAM{args.sensor_id} {args.width}x{args.height} @ {args.fps}")
    print(f"YOLO model:      {args.model}")
    print(f"Target height:   {args.target_height_mm} mm")
    print(f"Person standoff: {args.person_stop_distance_mm} mm")
    print(f"Throttle range:  {args.min_throttle}..{args.max_throttle}")
    print(f"Dry run:         {'yes' if args.dry_run else 'no'}")
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
    if args.target_height_mm <= 0:
        raise SystemExit("--target-height-mm must be greater than zero")

    if args.height_deadband_mm < 0:
        raise SystemExit("--height-deadband-mm cannot be negative")

    if args.person_stop_distance_mm <= 0:
        raise SystemExit("--person-stop-distance-mm must be greater than zero")

    if not 0.0 < args.target_box_area_fraction < 1.0:
        raise SystemExit("--target-box-area-fraction must be between 0 and 1")

    if not (
        1000
        <= args.min_throttle
        <= args.hover_throttle
        <= args.max_throttle
        <= 2000
    ):
        raise SystemExit(
            "Throttle values must satisfy "
            "1000 <= min <= hover <= max <= 2000"
        )

    if args.max_throttle > 1150 and not args.allow_flight_throttle:
        raise SystemExit(
            "Refusing throttle above 1150 without --allow-flight-throttle. "
            "Verify RC directions with propellers removed first."
        )

    if args.max_rc_offset < 0 or args.max_rc_offset > 400:
        raise SystemExit("--max-rc-offset must be between 0 and 400")

    if args.rc_step <= 0 or args.throttle_step <= 0:
        raise SystemExit("--rc-step and --throttle-step must be greater than zero")

    if args.search_turn_degrees <= 0:
        raise SystemExit("--search-turn-degrees must be greater than zero")

    if args.run_sec <= 0:
        raise SystemExit("--run-sec must be greater than zero")

    if args.calibration_sec < 0.5:
        raise SystemExit("--calibration-sec should be at least 0.5")

    if args.hcsr04_trigger_pin <= 0 or args.hcsr04_echo_pin <= 0:
        raise SystemExit("HC-SR04 trigger and echo pins must be positive integers")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Separate experimental state-machine for HC-SR04 altitude hold, "
            "YOLO human search/follow, and LiDAR obstacle avoidance."
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
    parser.add_argument("--hcsr04-trigger-pin", type=int, default=7)
    parser.add_argument("--hcsr04-echo-pin", type=int, default=15)
    parser.add_argument("--gpio-mode", choices=("BOARD", "BCM"), default="BOARD")
    parser.add_argument("--hcsr04-timeout-sec", type=float, default=0.03)
    parser.add_argument("--height-samples", type=int, default=3)
    parser.add_argument("--sim-height-mm", type=float, default=None)
    parser.add_argument("--target-height-mm", type=float, default=DEFAULT_TARGET_HEIGHT_MM)
    parser.add_argument("--height-deadband-mm", type=float, default=100.0)
    parser.add_argument("--altitude-kp", type=float, default=DEFAULT_ALTITUDE_KP)
    parser.add_argument("--min-throttle", type=int, default=DEFAULT_MIN_THROTTLE)
    parser.add_argument("--hover-throttle", type=int, default=DEFAULT_HOVER_THROTTLE)
    parser.add_argument("--max-throttle", type=int, default=DEFAULT_MAX_THROTTLE)
    parser.add_argument("--throttle-step", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-box-area", type=int, default=1500)
    parser.add_argument("--no-frame-timeout-sec", type=float, default=5.0)
    parser.add_argument(
        "--person-stop-distance-mm",
        type=float,
        default=DEFAULT_PERSON_STOP_DISTANCE_MM,
    )
    parser.add_argument("--person-distance-deadband-mm", type=float, default=200.0)
    parser.add_argument(
        "--target-box-area-fraction",
        type=float,
        default=DEFAULT_TARGET_BOX_AREA_FRACTION,
    )
    parser.add_argument(
        "--approach-pitch-offset",
        type=int,
        default=DEFAULT_APPROACH_PITCH_OFFSET,
    )
    parser.add_argument("--approach-pitch-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--approach-center-deadband", type=float, default=0.18)
    parser.add_argument("--yaw-center-kp", type=float, default=DEFAULT_YAW_CENTER_KP)
    parser.add_argument("--yaw-sign-control", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--search-yaw-offset", type=int, default=DEFAULT_SEARCH_YAW_OFFSET)
    parser.add_argument("--search-yaw-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--search-turn-degrees", type=float, default=360.0)
    parser.add_argument("--angle-offset-deg", type=float, default=0.0)
    parser.add_argument("--map-roll-deg", type=float, default=0.0)
    parser.add_argument("--map-pitch-deg", type=float, default=0.0)
    parser.add_argument("--map-yaw-deg", type=float, default=0.0)
    parser.add_argument("--use-yaw-frame", action="store_true")
    parser.add_argument("--set-distance-mm", type=float, default=1000.0)
    parser.add_argument("--deadband-mm", type=float, default=50.0)
    parser.add_argument("--max-range-mm", type=float, default=6000.0)
    parser.add_argument("--min-height-mm", type=float, default=-500.0)
    parser.add_argument("--max-height-mm", type=float, default=1500.0)
    parser.add_argument("--min-horizontal-mm", type=float, default=100.0)
    parser.add_argument("--sector-half-angle-deg", type=float, default=90.0)
    parser.add_argument("--max-rc-offset", type=int, default=DEFAULT_MAX_RC_OFFSET)
    parser.add_argument("--rc-step", type=int, default=DEFAULT_RC_STEP)
    parser.add_argument("--rc-roll-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--rc-pitch-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--warmup-sec", type=float, default=8.0)
    parser.add_argument("--arm-sec", type=float, default=2.0)
    parser.add_argument("--run-sec", type=float, default=30.0)
    parser.add_argument("--disarm-sec", type=float, default=2.0)
    parser.add_argument("--loop-hz", type=float, default=15.0)
    parser.add_argument("--print-hz", type=float, default=4.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-flight-throttle", action="store_true")
    return parser


def main():
    bootstrap_cuda_library_path()
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    confirm_or_exit(args)

    msp = None
    lidar = None
    imu = None
    height_sensor = None
    detector = None
    state = "CLIMB"
    current_roll = RC_CENTER
    current_pitch = RC_CENTER
    current_yaw = RC_CENTER
    search_start_yaw = 0.0
    last_height_ok = time.time()

    try:
        height_sensor = make_height_sensor(args)
        height_sensor.start()

        detector = HumanDetector(args)
        detector.start()

        imu = open_mpu6050(args.i2c_bus, args.imu_addr)
        imu.initialize()
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

        altitude = AltitudeController(args)

        if not args.dry_run:
            msp = MSP(args.fc_port, args.fc_baud)
            send_for_duration(msp, disarm_rc, args.warmup_sec, "Warmup disarm RC")
            print_status(msp, "Status after warmup")
            send_for_duration(msp, arm_rc, args.arm_sec, "Arming")
            print_status(msp, "Status after arm")

        print(f"Running mission state machine for {args.run_sec:.1f}s")
        end_time = time.time() + args.run_sec
        print_period = 1.0 / max(0.1, args.print_hz)
        loop_period = 1.0 / max(1.0, args.loop_hz)
        last_print = 0.0

        while time.time() < end_time:
            roll_deg, pitch_deg, yaw_deg = orientation.update()
            height_mm = height_sensor.read_height_mm(args.height_samples)

            if height_mm is not None:
                last_height_ok = time.time()
            elif not args.dry_run and time.time() - last_height_ok > 1.0:
                raise RuntimeError(
                    "HC-SR04 height readings timed out for more than 1s."
                )

            lidar_points = lidar.get_points()
            obstacle = nearest_obstacle(
                lidar_points,
                roll_deg,
                pitch_deg,
                yaw_deg,
                args,
            )
            target = detector.detect()
            next_state, search_start_yaw = choose_state(
                state,
                height_mm,
                target,
                obstacle,
                search_start_yaw,
                yaw_deg,
                args,
            )

            if next_state != state and next_state == "SEARCH":
                search_start_yaw = yaw_deg

            state = next_state
            throttle = altitude.update(height_mm)
            (
                target_roll,
                target_pitch,
                target_yaw,
                command_state,
                command_strength,
            ) = command_for_state(
                state,
                target,
                obstacle,
                search_start_yaw,
                yaw_deg,
                args,
            )
            current_roll = int(
                round(step_toward(current_roll, target_roll, args.rc_step))
            )
            current_pitch = int(
                round(step_toward(current_pitch, target_pitch, args.rc_step))
            )
            current_yaw = int(
                round(step_toward(current_yaw, target_yaw, args.rc_step))
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
            current_yaw = int(
                round(
                    clamp(
                        current_yaw,
                        RC_CENTER - args.max_rc_offset,
                        RC_CENTER + args.max_rc_offset,
                    )
                )
            )
            channels = rc_channels(
                current_roll,
                current_pitch,
                throttle,
                yaw=current_yaw,
                arm=True,
            )

            if msp is not None:
                msp.send_rc(channels)

            now = time.time()

            if now - last_print >= print_period:
                last_print = now
                print(
                    f"state={state:<8} cmd={command_state:<6} "
                    f"rc=roll:{current_roll} pitch:{current_pitch} "
                    f"thr:{throttle} yaw:{current_yaw} "
                    f"h={format_height(height_mm)} "
                    f"rpy=({roll_deg:.1f},{pitch_deg:.1f},{yaw_deg:.1f}) "
                    f"target={format_target(target)} "
                    f"avoid={command_strength:.2f} obstacle={format_obstacle(obstacle)}"
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

        if detector is not None:
            detector.close()

        if imu is not None:
            imu.close()

        if height_sensor is not None:
            height_sensor.close()

        print("Done.")


if __name__ == "__main__":
    main()
