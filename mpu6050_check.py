import argparse
import time

from mpu6050_i2c import MPU6050, available_i2c_buses, detect_mpu6050


def main():
    parser = argparse.ArgumentParser(
        description="Find and live-check an MPU-6050 on Jetson I2C."
    )
    parser.add_argument("--bus", type=int)
    parser.add_argument("--addr", type=lambda value: int(value, 0))
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--hz", type=float, default=10.0)
    args = parser.parse_args()

    if args.bus is None or args.addr is None:
        print(f"Available I2C buses: {available_i2c_buses()}")
        found = detect_mpu6050()

        if not found:
            print("No MPU-6050 found at 0x68 or 0x69.")
            print("Check SDA/SCL wiring, 3.3V power, GND, and the I2C bus number.")
            return

        bus, addr, who = found[0]
        print(
            f"Using first MPU-6050 found: bus={bus}, "
            f"addr=0x{addr:02X}, WHO_AM_I=0x{who:02X}"
        )
    else:
        bus = args.bus
        addr = args.addr

    imu = MPU6050(bus, addr)

    try:
        imu.initialize()
        who = imu.who_am_i()
        print(f"WHO_AM_I: 0x{who:02X}")
        print("Move the LiDAR + IMU together. Gyro values should change.")
        print("ax ay az are in g. gx gy gz are deg/sec.")

        end = time.time() + args.seconds
        sleep_sec = 1.0 / max(1.0, args.hz)

        while time.time() < end:
            ax, ay, az, gx, gy, gz = imu.read_motion()
            print(
                f"accel=({ax: .3f}, {ay: .3f}, {az: .3f}) "
                f"gyro=({gx: .2f}, {gy: .2f}, {gz: .2f})"
            )
            time.sleep(sleep_sec)

    finally:
        imu.close()


if __name__ == "__main__":
    main()
