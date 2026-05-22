import argparse
import time

import cv2
import numpy as np

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
except Exception as exc:
    raise SystemExit(f"Python GStreamer bindings are missing: {exc}")

try:
    import torch
except Exception:
    torch = None

from ultralytics import YOLO


DEFAULT_MODEL = "yolov8s.pt"
PERSON_CLASS_ID = 0
WINDOW_NAME = "YOLO Human Detection - CAM0"


class ArgusCamera:
    def __init__(self, sensor_id=0, width=1920, height=1080, fps=60):
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.appsink = None

    def pipeline_string(self):
        return (
            f"nvarguscamerasrc sensor-id={self.sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
            f"format=NV12, framerate={self.fps}/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! "
            "appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
        )

    def start(self):
        Gst.init(None)
        self.pipeline = Gst.parse_launch(self.pipeline_string())
        self.appsink = self.pipeline.get_by_name("sink")

        if self.appsink is None:
            raise RuntimeError("Could not create GStreamer appsink.")

        result = self.pipeline.set_state(Gst.State.PLAYING)

        if result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Could not start nvarguscamerasrc pipeline.")

    def read(self, timeout_ms=1000):
        sample = self.appsink.emit("try-pull-sample", timeout_ms * Gst.MSECOND)

        if sample is None:
            return None

        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
        buffer = sample.get_buffer()
        success, info = buffer.map(Gst.MapFlags.READ)

        if not success:
            return None

        try:
            frame = np.frombuffer(info.data, dtype=np.uint8)
            frame = frame.reshape((height, width, 3))
            return frame.copy()
        finally:
            buffer.unmap(info)

    def close(self):
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)


def available_device(default_device):
    if default_device != "auto":
        return default_device

    if torch is not None and torch.cuda.is_available():
        return 0

    return "cpu"


def draw_detections(frame, result, min_box_area):
    humans = []

    if result.boxes is None:
        return humans

    for box in result.boxes:
        cls_id = int(box.cls[0])

        if cls_id != PERSON_CLASS_ID:
            continue

        conf = float(box.conf[0])
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        area = max(0, x2 - x1) * max(0, y2 - y1)

        if area < min_box_area:
            continue

        humans.append((x1, y1, x2, y2, conf))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 2)
        cv2.putText(
            frame,
            f"person {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 80),
            2,
            cv2.LINE_AA,
        )

    return humans


def main():
    parser = argparse.ArgumentParser(
        description="Detect humans from a Jetson CAM0 CSI camera using YOLO."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-box-area", type=int, default=1500)
    parser.add_argument("--print-every-sec", type=float, default=1.0)
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument("--no-frame-timeout-sec", type=float, default=5.0)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    device = available_device(args.device)

    print("=== YOLO Human Detection - Jetson CAM0 ===")
    print("Camera backend: nvarguscamerasrc / GStreamer")
    print(f"Sensor ID:      {args.sensor_id}")
    print(f"Camera size:    {args.width}x{args.height} @ {args.fps} FPS")
    print(f"Model:          {args.model}")
    print(f"YOLO imgsz:     {args.imgsz}")
    print(f"Confidence:     {args.conf}")
    print(f"Device:         {device}")
    print(f"No-frame timeout: {args.no_frame_timeout_sec:.1f}s")

    if torch is not None:
        print(f"Torch CUDA:     {torch.cuda.is_available()}")

    print("Press q or Esc to quit.")
    print("")

    camera = ArgusCamera(args.sensor_id, args.width, args.height, args.fps)
    model = YOLO(args.model)
    camera.start()

    last_print = 0.0
    frame_count = 0
    start_time = time.time()
    fps_start = start_time
    display_fps = 0.0
    last_frame_time = start_time

    try:
        while True:
            now = time.time()

            if args.duration_sec > 0 and now - start_time >= args.duration_sec:
                break

            frame = camera.read()

            if frame is None:
                if now - last_frame_time >= args.no_frame_timeout_sec:
                    raise RuntimeError(
                        "No camera frames received from nvarguscamerasrc. "
                        "Argus may not see the CAM0 camera."
                    )

                print("No camera frame received.")
                time.sleep(0.05)
                continue

            last_frame_time = time.time()
            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                classes=[PERSON_CLASS_ID],
                device=device,
                verbose=False,
            )
            annotated = frame.copy()
            humans = draw_detections(
                annotated,
                results[0],
                args.min_box_area,
            )

            frame_count += 1
            now = time.time()

            if now - fps_start >= 1.0:
                display_fps = frame_count / (now - fps_start)
                frame_count = 0
                fps_start = now

            cv2.putText(
                annotated,
                f"humans: {len(humans)}  fps: {display_fps:.1f}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if now - last_print >= args.print_every_sec:
                last_print = now
                if humans:
                    best = max(humans, key=lambda item: item[4])
                    print(
                        f"humans={len(humans)} best_conf={best[4]:.2f} "
                        f"box=({best[0]},{best[1]})-({best[2]},{best[3]})"
                    )
                else:
                    print("humans=0")

            if not args.no_display:
                cv2.imshow(WINDOW_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        print("\nStopping.")
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        print("Camera checks:")
        print("- Power off before reseating the CSI ribbon.")
        print("- Confirm the ribbon is in CAM0 and facing the correct direction.")
        print("- Reboot after connecting the CSI camera.")
        print("- Test with: nvgstcapture-1.0 --sensor-id=0")
        print("- If sensor-id 0 fails, try --sensor-id 1.")

    finally:
        camera.close()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
