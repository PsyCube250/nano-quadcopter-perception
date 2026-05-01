cd ~/Desktop/nano-quadcopter-perception

cat > README.md <<'EOF'
# Nano Quadcopter Perception

An experimental perception stack for a quadcopter using a Jetson Orin Nano as the onboard analysis computer.

The long-term goal is to combine:

- Camera-based YOLO object detection
- 2D LiDAR obstacle sensing
- Fixed-altitude target approach
- Simple local obstacle avoidance

At this stage, the project focuses on hardware bring-up and LiDAR data validation. The obstacle avoidance algorithm is still under development.

## Project Goal

The drone is intended to fly at a fixed altitude while using onboard perception to:

1. Detect a target using a camera and YOLO.
2. Move toward the front of the target.
3. Use a 2D LiDAR to detect nearby obstacles.
4. Avoid the nearest obstacle while continuing the target approach task.

## Current Status

Completed:

- STL-27L LiDAR wiring verified.
- USB-to-TTL serial communication tested.
- Raw LiDAR packets received successfully.
- LiDAR frames parsed into angle, distance, and confidence values.
- Basic 2D scan visualization tested on macOS.

Not completed yet:

- Jetson deployment.
- YOLO integration.
- Flight controller integration.
- Real obstacle avoidance logic.
- Autonomous flight testing.

## Hardware

Current hardware used or planned:

- Jetson Orin Nano Developer Kit
- LDROBOT STL-27L 2D LiDAR
- USB-to-TTL serial adapter
- Camera for YOLO object detection
- Quadcopter frame
- Flight controller, to be decided or integrated later

## LiDAR Wiring

| LiDAR Signal | USB-to-TTL Adapter |
|---|---|
| TX | RXD |
| PWM | GND |
| GND | GND |
| P5V | +5V |

The adapter TXD pin is not required for basic LiDAR reading.

## Software Requirements

Install Python dependencies:

```bash
pip install pyserial matplotlib
