import argparse
import serial
import time


def main():
    parser = argparse.ArgumentParser(
        description="Read raw STL-27L LiDAR serial data."
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
        "--chunk",
        type=int,
        default=64,
        help="Number of bytes to read each time",
    )
    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)

    print("Reading raw LiDAR data.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            data = ser.read(args.chunk)
            if data:
                print(data.hex(" "))
            else:
                print("No data")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        ser.close()


if __name__ == "__main__":
    main()
