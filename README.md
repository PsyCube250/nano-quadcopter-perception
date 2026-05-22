# Nano Quadcopter Perception

Jetson FC + STL-27L LiDAR monitor and perception experiments for a Jetson Nano
or Jetson Orin Nano quadcopter setup.

The long-term goal is to combine camera-based YOLO object detection, 2D LiDAR
obstacle sensing, fixed-altitude target approach, and simple local obstacle
avoidance. The current code focuses on bench-safe Jetson hardware bring-up,
LiDAR validation, camera person detection, and flight-controller communication
tests.

This project keeps the first working monitor behavior:

- Opens the flight controller MSP UART on `/dev/ttyTHS1` at `115200`.
- Opens the STL-27L LiDAR on `/dev/ttyUSB0` at `921600`.
- Continuously sends disarm/neutral RC to the flight controller.
- Reads current STL-27L frames.
- Prints the nearest valid LiDAR obstacle and nearest front-sector obstacle.
- Does not arm.
- Does not spin motors.
- Does not create maps, use SLAM, store old scans persistently, plot a GUI, or run YOLO.

## Safety

Remove propellers before running this on the bench.

The monitor sends only disarm RC values:

```text
[1500, 1500, 1000, 1500, 1000, 1000, 1000, 1000]
```

## Files

- `drone_msp_lidar_monitor.py` is the simple one-file runnable monitor for Jetson testing.
- `msp_fc.py` contains the MSP disarm RC sender.
- `lidar_stl27l.py` contains the STL-27L reader/parser.
- `lidar_live_view.py` draws a non-persistent real-time LiDAR scan only.
- `motor_low_throttle_test.py` runs a short low-throttle motor bench test.
- `motor_lidar_speed_control_test.py` runs a low-throttle bench test that smoothly reduces throttle when LiDAR detects a close obstacle.
- `lidar_3d_scan_mapper.py` creates an approximate sparse 3D cloud from timed 2D LiDAR rotation.
- `lidar_3d_scan_mapper_mpu6050.py` creates an approximate sparse 3D cloud from 2D LiDAR plus MPU-6050 gyro yaw.
- `lidar_imu_obstacle_avoidance.py` sends MSP RC roll/pitch commands away from nearby LiDAR obstacles using MPU-6050 attitude compensation.
- `drone_hcsr04_lidar_human_follow.py` combines HC-SR04 height hold, YOLO human search/follow, and LiDAR obstacle avoidance in a separate experimental RC state machine.
- `xbox_controller_detect.py` reads the Bluetooth Xbox controller from `/dev/input/js*` and prints live button/axis events.
- `xbox_drone_control.py` maps the Xbox controller to MSP RC channels, with left stick center set to hover throttle.
- `sonar_test.py` reads an HC-SR04 sonar on Jetson GPIO and prints live distance.
- `lidar_3d_realtime_mpu6050.py` shows the sparse 3D LiDAR + MPU-6050 map while it is being built.
- `yolo_human_picam.py` detects people from the Jetson CAM0 CSI camera with YOLO.
- `mpu6050_check.py` finds and live-checks the MPU-6050 I2C connection.
- `requirements.txt` lists the Python dependency.

## Wiring

Flight controller:

- FC UART to Jetson UART
- Expected device: `/dev/ttyTHS1`
- Baud: `115200`

STL-27L LiDAR through USB-to-TTL:

- LiDAR TX wire -> USB-TTL RXD
- LiDAR PWM wire -> USB-TTL GND
- LiDAR GND wire -> USB-TTL GND
- LiDAR +5V wire -> USB-TTL +5V
- USB-TTL TXD is not used
- USB-TTL 3V3 is not used

Known color setup from the working test:

- Black -> RXD
- Red -> GND
- White -> GND
- Yellow -> +5V

If the LiDAR spins but there is no data, power is working but TX is probably not connected to RXD.

## Install

```bash
sudo apt update
sudo apt install -y python3-serial
```

Or with pip:

```bash
python3 -m pip install -r requirements.txt
```

## Run

From this directory:

```bash
sudo python3 drone_msp_lidar_monitor.py \
  --fc-port /dev/ttyTHS1 \
  --lidar-port /dev/ttyUSB0
```

With an explicit path:

```bash
sudo python3 /home/jetson/Documents/drone_msp_lidar_monitor.py \
  --fc-port /dev/ttyTHS1 \
  --lidar-port /dev/ttyUSB0
```

## Useful Port Checks

```bash
ls /dev/ttyUSB*
ls /dev/ttyTHS*
ls /dev/i2c-*
dmesg | tail -30
```

## Backup Archive

The initial workspace files were archived before adding LiDAR motor-control logic:

```text
beginning_files_backup_2026-05-05.tar.gz
```

## Check Only The LiDAR

First check that the USB serial device exists:

```bash
ls /dev/ttyUSB*
```

Then run the live scan viewer:

```bash
sudo python3 lidar_live_view.py --lidar-port /dev/ttyUSB0
```

The viewer draws only the current LiDAR points. It does not connect to the flight controller, send RC commands, create a persistent map, or stack old scans.

For a faster display refresh:

```bash
sudo python3 lidar_live_view.py --lidar-port /dev/ttyUSB0 --fps 60 --point-age-sec 0.25
```

The installed OpenCV package may expose a `cv2.cuda` module without a usable CUDA display path. The live viewer is written to stay light on the CPU instead: it keeps only the newest point per angle bin and pre-renders the static grid.

If the window opens but says `waiting for fresh data`, check the LiDAR TX-to-USB-RXD wiring and confirm the baud rate is `921600`.

## LiDAR Low-Throttle Motor Test

This test arms the flight controller and sends low RC throttle. It smoothly lowers throttle when the LiDAR nearest obstacle is at or below the threshold.

Run only with propellers removed:

```bash
sudo python3 motor_lidar_speed_control_test.py \
  --fc-port /dev/ttyTHS1 \
  --lidar-port /dev/ttyUSB0 \
  --threshold-mm 50 \
  --normal-throttle 1075 \
  --slow-throttle 1000 \
  --run-sec 5
```

The script requires typing `PROPS REMOVED` before it will arm. If Betaflight reports arming blocks such as `FAILSAFE` or `NOGYRO`, fix those before relying on the test.

## LiDAR + IMU Obstacle Avoidance RC Test

This test uses the `lidar_3d_scan_mapper_mpu6050.py` orientation helpers to
level LiDAR points, finds the nearest obstacle in a configurable height/sector
window, and sends roll/pitch RC offsets away from that obstacle until it is past
the set distance. Run it only after the low-throttle motor and LiDAR tests work.

Dry-run first, with no flight-controller RC commands:

```bash
sudo python3 lidar_imu_obstacle_avoidance.py \
  --lidar-port /dev/ttyUSB0 \
  --set-distance-mm 1000 \
  --dry-run
```

Bench RC test with propellers removed:

```bash
sudo python3 lidar_imu_obstacle_avoidance.py \
  --fc-port /dev/ttyTHS1 \
  --lidar-port /dev/ttyUSB0 \
  --set-distance-mm 1000 \
  --throttle 1000 \
  --run-sec 10
```

If roll or pitch moves toward the obstacle instead of away from it, stop and
reverse the matching sign with `--rc-roll-sign -1` or `--rc-pitch-sign -1`.

## Xbox Controller RC Test

First verify the Bluetooth Xbox controller mapping:

```bash
python3 xbox_controller_detect.py
```

Dry-run the full RC mapping without sending anything to the flight controller:

```bash
python3 xbox_drone_control.py --dry-run --run-sec 20
```

Default controls:

- Left stick X: yaw
- Left stick Y: throttle, centered at `--hover-throttle`
- Right stick X: roll
- Right stick Y: pitch
- RB: arm/disarm toggle, only arms while left stick is held down near minimum throttle
- B: emergency disarm

Bench-test with propellers removed:

```bash
python3 xbox_drone_control.py \
  --hover-throttle 1300 \
  --throttle-offset 300 \
  --allow-flight-throttle
```

If serial permissions require it, run the same command with `sudo`. Tune
`--hover-throttle` for the actual airframe; center stick sends that value only
after the arm latch is active. Press RB again or press B to immediately send
disarm RC.

## HC-SR04 Height + Human Follow Test

`drone_hcsr04_lidar_human_follow.py` is a separate experimental state machine:

- climb toward an HC-SR04 target height,
- hover and yaw-search up to 360 degrees,
- use the existing YOLO person detector from `yolo_human_picam.py`,
- approach the detected human while staying outside a LiDAR/person standoff,
- let LiDAR obstacle avoidance override approach commands.

The existing camera code detects a `person` class, not a face-only model. Add a
face detector later if the target must be an actual face.

The HC-SR04 echo pin is 5 V. Use a level shifter or resistor divider before
connecting echo to the Jetson GPIO pin.

Standalone sonar check:

```bash
python3 sonar_test.py --gpio-mode BOARD
```

Print five readings with the raw samples used for each median:

```bash
python3 sonar_test.py --count 5 --raw
```

Dry-run first. This still opens sensors, but sends no flight-controller RC:

```bash
sudo python3 drone_hcsr04_lidar_human_follow.py \
  --lidar-port /dev/ttyUSB0 \
  --hcsr04-trigger-pin 7 \
  --hcsr04-echo-pin 15 \
  --target-height-mm 1000 \
  --person-stop-distance-mm 1500 \
  --dry-run
```

For a no-GPIO command-line check:

```bash
python3 drone_hcsr04_lidar_human_follow.py --help
```

The default throttle range is capped at `1000..1150`, so it is intended for
bench direction checks. Any higher throttle requires `--allow-flight-throttle`
and should only be used after RC direction, altitude response, camera target
response, and LiDAR avoidance are verified with propellers removed.

## YOLO Human Detection From CAM0

The CAM0 CSI camera uses Jetson Argus/GStreamer, not `/dev/video0` on this setup.

Run:

```bash
python3 yolo_human_picam.py \
  --sensor-id 0 \
  --model yolov8s.pt \
  --conf 0.35
```

If you only want terminal output without a display window:

```bash
python3 yolo_human_picam.py --sensor-id 0 --model yolov8s.pt --no-display
```

The script filters YOLO to class `person` only. Torch CUDA currently may not be available if the installed Torch build does not match the Jetson driver, so the script defaults to CPU unless CUDA initializes cleanly.

## MPU-6050 3D LiDAR Scan

The MPU-6050 should be rigidly mounted to the LiDAR. This script assumes the LiDAR scan plane is vertical and uses MPU-6050 roll, pitch, and gyro yaw while you move the LiDAR around the room.

Install I2C tools if needed:

```bash
sudo apt install -y i2c-tools
```

Check the MPU-6050 connection:

```bash
python3 mpu6050_check.py
```

Run the IMU-based mapper:

```bash
sudo python3 lidar_3d_scan_mapper_mpu6050.py \
  --lidar-port /dev/ttyUSB0 \
  --duration-sec 30 \
  --voxel-mm 50 \
  --max-points 20000
```

The mapper auto-detects the MPU-6050 on I2C address `0x68` or `0x69`. Keep the LiDAR + IMU still during the first calibration seconds, then move smoothly. Roll and pitch are stabilized with the accelerometer; yaw is still gyro-integrated because the MPU-6050 has no magnetometer, so yaw will drift over time.

For a real-time 3D view while the map is being made:

```bash
sudo python3 lidar_3d_realtime_mpu6050.py \
  --lidar-port /dev/ttyUSB0 \
  --i2c-bus 7 \
  --imu-addr 0x68 \
  --duration-sec 60 \
  --voxel-mm 60 \
  --max-points 15000
```

Increase `--voxel-mm` to save fewer close-together points, for example `--voxel-mm 100`. The viewer also has a draw cap with `--draw-max-points`, but the saved storage cap is controlled by `--voxel-mm` and `--max-points`.

If the 3D cloud looks sideways or flipped by 90 degrees, add a map rotation correction. Try one of these:

```bash
--map-roll-deg 90
--map-roll-deg -90
--map-pitch-deg 90
--map-pitch-deg -90
```

If the LiDAR slice itself is rotated inside the scan plane, try:

```bash
--angle-offset-deg 90
```

If roll, pitch, or yaw responds on the wrong IMU axis, remap or flip the axes:

```bash
--roll-axis y --pitch-axis x
--roll-sign -1
--pitch-sign -1
--yaw-sign -1
```

## Expected Output

```text
=== FC + LiDAR Monitor ===
SAFETY: REMOVE PROPELLERS.
This script does NOT arm.
It only sends disarm RC and reads LiDAR.

--- Status ---
LiDAR nearest: 684 mm at 151.2 deg conf=82
LiDAR front +/-30 deg: 930 mm at 12.4 deg conf=76
```
