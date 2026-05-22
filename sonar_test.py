#!/usr/bin/env python3
"""
Standalone HC-SR04 sonar distance test for Jetson GPIO.

Default pins use physical BOARD numbering:
- trigger: pin 7
- echo: pin 15

The HC-SR04 echo pin is 5 V. Use a level shifter or resistor divider before
connecting echo to a Jetson GPIO input.
"""

import argparse
import os
import statistics
import sys
import time


DEFAULT_TRIGGER_PIN = 7
DEFAULT_ECHO_PIN = 15
DEFAULT_GPIO_MODE = "BOARD"
DEFAULT_TIMEOUT_SEC = 0.03
DEFAULT_RATE_HZ = 5.0
DEFAULT_SAMPLES = 3
MIN_VALID_MM = 20.0
MAX_VALID_MM = 4500.0


class HCSR04Sonar:
    def __init__(
        self,
        trigger_pin,
        echo_pin,
        gpio_mode=DEFAULT_GPIO_MODE,
        timeout_sec=DEFAULT_TIMEOUT_SEC,
        settle_sec=0.0002,
        temperature_c=20.0,
    ):
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self.gpio_mode = gpio_mode.upper()
        self.timeout_sec = timeout_sec
        self.settle_sec = settle_sec
        self.temperature_c = temperature_c
        self.gpio = None

    @property
    def sound_speed_mm_per_sec(self):
        return (331.3 + 0.606 * self.temperature_c) * 1000.0

    def start(self):
        GPIO = import_jetson_gpio()

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

        self.send_trigger_pulse()

        pulse_start = self._wait_for_level(self.gpio.HIGH, deadline)
        if pulse_start is None:
            return None

        pulse_end = self._wait_for_level(self.gpio.LOW, deadline)
        if pulse_end is None:
            return None

        pulse_sec = pulse_end - pulse_start
        return pulse_sec * self.sound_speed_mm_per_sec / 2.0

    def send_trigger_pulse(self):
        self.gpio.output(self.trigger_pin, self.gpio.LOW)
        time.sleep(self.settle_sec)
        self.gpio.output(self.trigger_pin, self.gpio.HIGH)
        time.sleep(0.00001)
        self.gpio.output(self.trigger_pin, self.gpio.LOW)

    def echo_level(self):
        return int(self.gpio.input(self.echo_pin))

    def capture_echo_edges(self):
        start = time.perf_counter()
        deadline = start + self.timeout_sec
        initial_level = self.echo_level()
        last_level = initial_level
        edges = []

        self.send_trigger_pulse()

        while time.perf_counter() < deadline:
            level = self.echo_level()

            if level != last_level:
                now = time.perf_counter()
                edges.append((now - start, level))
                last_level = level

        return initial_level, edges

    def read_median_mm(self, samples):
        readings = []

        for _ in range(max(1, samples)):
            value = self.read_once_mm()

            if value is not None and MIN_VALID_MM <= value <= MAX_VALID_MM:
                readings.append(value)

            time.sleep(0.02)

        if not readings:
            return None, []

        return statistics.median(readings), readings

    def close(self):
        if self.gpio is not None:
            self.gpio.cleanup([self.trigger_pin, self.echo_pin])


def read_device_tree_strings(path):
    try:
        with open(path, "rb") as fp:
            data = fp.read()
    except OSError:
        return []

    return [
        item.decode("utf-8", errors="replace")
        for item in data.rstrip(b"\x00").split(b"\x00")
        if item
    ]


def detect_jetson_model_override():
    compatible = read_device_tree_strings("/proc/device-tree/compatible")
    model = " ".join(read_device_tree_strings("/proc/device-tree/model"))
    joined = " ".join(compatible + [model]).lower()

    if "p3767-0003" in joined or "p3767-0004" in joined or "p3767-0005" in joined:
        return "JETSON_ORIN_NANO"

    if "p3767-0000" in joined or "p3767-0001" in joined:
        return "JETSON_ORIN_NX"

    if "jetson agx orin" in joined:
        return "JETSON_ORIN"

    return None


def clear_partial_jetson_gpio_import():
    for name in list(sys.modules):
        if name == "Jetson" or name.startswith("Jetson."):
            sys.modules.pop(name, None)


def import_jetson_gpio():
    try:
        import Jetson.GPIO as GPIO
        return GPIO
    except Exception as exc:
        if "Could not determine Jetson model" not in str(exc):
            raise RuntimeError(
                "Jetson.GPIO is required. Install it with: "
                "sudo apt install python3-jetson-gpio"
            ) from exc

        model_name = detect_jetson_model_override()

        if model_name is None:
            raise RuntimeError(
                "Jetson.GPIO is installed, but it could not determine this Jetson "
                "model. Try setting JETSON_MODEL_NAME manually."
            ) from exc

        os.environ["JETSON_MODEL_NAME"] = model_name
        clear_partial_jetson_gpio_import()

        try:
            import Jetson.GPIO as GPIO
            print(f"Using Jetson.GPIO model override: JETSON_MODEL_NAME={model_name}")
            return GPIO
        except Exception as retry_exc:
            raise RuntimeError(
                f"Jetson.GPIO model override {model_name} failed: {retry_exc}"
            ) from retry_exc


def build_parser():
    parser = argparse.ArgumentParser(
        description="Read distance from an HC-SR04 sonar using Jetson GPIO."
    )
    parser.add_argument("--trigger-pin", type=int, default=DEFAULT_TRIGGER_PIN)
    parser.add_argument("--echo-pin", type=int, default=DEFAULT_ECHO_PIN)
    parser.add_argument("--gpio-mode", choices=("BOARD", "BCM"), default=DEFAULT_GPIO_MODE)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--temperature-c", type=float, default=20.0)
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of readings to print. 0 means run until Ctrl+C.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Also print the valid raw sample values used for the median.",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print echo pin idle state and echo edges after each trigger pulse.",
    )
    return parser


def validate_args(args):
    if args.trigger_pin <= 0 or args.echo_pin <= 0:
        raise SystemExit("--trigger-pin and --echo-pin must be positive integers")

    if args.trigger_pin == args.echo_pin:
        raise SystemExit("--trigger-pin and --echo-pin must be different pins")

    if args.samples <= 0:
        raise SystemExit("--samples must be greater than zero")

    if args.rate_hz <= 0:
        raise SystemExit("--rate-hz must be greater than zero")

    if args.timeout_sec <= 0:
        raise SystemExit("--timeout-sec must be greater than zero")

    if args.count < 0:
        raise SystemExit("--count cannot be negative")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    sonar = HCSR04Sonar(
        trigger_pin=args.trigger_pin,
        echo_pin=args.echo_pin,
        gpio_mode=args.gpio_mode,
        timeout_sec=args.timeout_sec,
        temperature_c=args.temperature_c,
    )

    print("=== HC-SR04 SONAR TEST ===")
    print("This only reads GPIO. It does not touch the flight controller.")
    print("Use a level shifter or resistor divider on the 5 V echo line.")
    print(f"GPIO mode:   {args.gpio_mode}")
    print(f"Trigger pin: {args.trigger_pin}")
    print(f"Echo pin:    {args.echo_pin}")
    print(f"Samples:     {args.samples}")
    print(f"Rate:        {args.rate_hz:.1f} Hz")
    print("")

    period = 1.0 / args.rate_hz
    printed = 0

    try:
        sonar.start()

        if args.diagnose:
            run_diagnostics(sonar, args)
            return 0

        while args.count == 0 or printed < args.count:
            start = time.perf_counter()
            distance_mm, raw_readings = sonar.read_median_mm(args.samples)

            if distance_mm is None:
                print("distance: timeout / no valid echo")
            else:
                distance_cm = distance_mm / 10.0
                distance_in = distance_mm / 25.4
                message = (
                    f"distance: {distance_mm:7.1f} mm  "
                    f"{distance_cm:6.1f} cm  {distance_in:6.1f} in"
                )

                if args.raw:
                    raw = ", ".join(f"{value:.1f}" for value in raw_readings)
                    message += f"  raw=[{raw}]"

                print(message)

            printed += 1
            elapsed = time.perf_counter() - start
            time.sleep(max(0.0, period - elapsed))

    except KeyboardInterrupt:
        print("\nStopped.")
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        sonar.close()

    return 0


def run_diagnostics(sonar, args):
    total = args.count if args.count > 0 else 10
    period = 1.0 / args.rate_hz

    print("Diagnostic mode:")
    print("- idle_echo=0 is expected before a trigger")
    print("- a working sensor should show a rising edge then a falling edge")
    print("")

    for index in range(total):
        start = time.perf_counter()
        idle_echo = sonar.echo_level()
        initial_level, edges = sonar.capture_echo_edges()

        if edges:
            edge_text = ", ".join(
                f"{elapsed * 1_000_000:8.0f}us->{level}"
                for elapsed, level in edges
            )
        else:
            edge_text = "none"

        print(
            f"pulse {index + 1:02d}: idle_echo={idle_echo} "
            f"initial_after_trigger={initial_level} edges={edge_text}"
        )

        elapsed = time.perf_counter() - start
        time.sleep(max(0.0, period - elapsed))


if __name__ == "__main__":
    raise SystemExit(main())
