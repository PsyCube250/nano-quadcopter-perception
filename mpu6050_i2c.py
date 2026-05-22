import fcntl
import os
import time
from pathlib import Path


I2C_SLAVE = 0x0703

MPU6050_ADDRS = (0x68, 0x69)
PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B
WHO_AM_I = 0x75

ACCEL_SCALE = 16384.0
GYRO_SCALE = 131.0


def available_i2c_buses():
    buses = []

    for path in Path("/dev").glob("i2c-*"):
        try:
            buses.append(int(path.name.split("-", 1)[1]))
        except ValueError:
            continue

    return sorted(buses)


class LinuxI2CDevice:
    def __init__(self, bus_num, address):
        self.bus_num = bus_num
        self.address = address
        self.path = f"/dev/i2c-{bus_num}"
        self.fd = os.open(self.path, os.O_RDWR)
        fcntl.ioctl(self.fd, I2C_SLAVE, address)

    def write_byte_data(self, reg, value):
        os.write(self.fd, bytes([reg & 0xFF, value & 0xFF]))

    def read_byte_data(self, reg):
        os.write(self.fd, bytes([reg & 0xFF]))
        data = os.read(self.fd, 1)

        if len(data) != 1:
            raise OSError(f"Short I2C read from {self.path}")

        return data[0]

    def close(self):
        os.close(self.fd)


class MPU6050:
    def __init__(self, bus_num, address):
        self.bus_num = bus_num
        self.address = address
        self.dev = LinuxI2CDevice(bus_num, address)

    def initialize(self):
        self.dev.write_byte_data(PWR_MGMT_1, 0x00)
        time.sleep(0.1)
        self.dev.write_byte_data(SMPLRT_DIV, 0x07)
        self.dev.write_byte_data(CONFIG, 0x03)
        self.dev.write_byte_data(GYRO_CONFIG, 0x00)
        self.dev.write_byte_data(ACCEL_CONFIG, 0x00)

    def who_am_i(self):
        return self.dev.read_byte_data(WHO_AM_I)

    def read_word_signed(self, reg):
        high = self.dev.read_byte_data(reg)
        low = self.dev.read_byte_data(reg + 1)
        value = (high << 8) | low

        if value >= 0x8000:
            value -= 0x10000

        return value

    def read_motion(self):
        ax = self.read_word_signed(ACCEL_XOUT_H) / ACCEL_SCALE
        ay = self.read_word_signed(ACCEL_XOUT_H + 2) / ACCEL_SCALE
        az = self.read_word_signed(ACCEL_XOUT_H + 4) / ACCEL_SCALE
        gx = self.read_word_signed(ACCEL_XOUT_H + 8) / GYRO_SCALE
        gy = self.read_word_signed(ACCEL_XOUT_H + 10) / GYRO_SCALE
        gz = self.read_word_signed(ACCEL_XOUT_H + 12) / GYRO_SCALE
        return ax, ay, az, gx, gy, gz

    def close(self):
        self.dev.close()


def detect_mpu6050(buses=None, addresses=MPU6050_ADDRS):
    if buses is None:
        buses = available_i2c_buses()

    results = []

    for bus in buses:
        for address in addresses:
            imu = None

            try:
                imu = MPU6050(bus, address)
                who = imu.who_am_i()
            except OSError:
                continue
            finally:
                if imu:
                    imu.close()

            if who in (0x68, 0x70):
                results.append((bus, address, who))

    return results
