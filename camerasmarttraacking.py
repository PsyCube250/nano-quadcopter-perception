import argparse
import os
import site
import sys
import time
import warnings
from pathlib import Path


def bootstrap_cuda_library_path():
    """Make pip-installed Jetson CUDA helper libs visible before torch imports."""

    candidates = []

    for package_dir in [site.getusersitepackages(), *site.getsitepackages()]:
        cuda_lib_dir = Path(package_dir) / "nvidia" / "cu12" / "lib"

        if cuda_lib_dir.exists():
            candidates.append(str(cuda_lib_dir))

    if not candidates:
        return

    current_paths = [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path]
    missing_paths = [path for path in candidates if path not in current_paths]

    if not missing_paths or os.environ.get("JETSON_CUDA_LIBPATH_BOOTSTRAPPED") == "1":
        return

    os.environ["LD_LIBRARY_PATH"] = ":".join(missing_paths + current_paths)
    os.environ["JETSON_CUDA_LIBPATH_BOOTSTRAPPED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


bootstrap_cuda_library_path()

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
except Exception:
    Gst = None

try:
    import torch
except Exception:
    torch = None


DEFAULT_MODEL = "yolov8n.pt"
PERSON_CLASS_ID = 0
WINDOW_NAME = "Appearance Locked Tracking"
DEFAULT_WIDTH = 1536
DEFAULT_HEIGHT = 864
DEFAULT_FPS = 90
DEFAULT_IMGSZ = [DEFAULT_HEIGHT, DEFAULT_WIDTH]
DEFAULT_TARGET_FPS = 60.0
DEFAULT_MAX_DETECT_INTERVAL = 8
DEFAULT_ZOOM_BOX_WIDTH = 720.0
ADAPTIVE_IMGSZ_STEPS = [
    [864, 1536],
    [768, 1344],
    [672, 1216],
    [640, 1152],
    [544, 960],
    [448, 800],
    [384, 672],
    [320, 576],
    [256, 448],
    [224, 384],
    [192, 352],
]


class ArgusCamera:
    """CSI camera reader for Jetson nvarguscamerasrc."""

    def __init__(self, sensor_id=0, width=1280, height=720, fps=30):
        if Gst is None:
            raise RuntimeError(
                "Python GStreamer bindings are missing. Install python3-gi and "
                "gir1.2-gstreamer-1.0, or run on the Jetson image that includes them."
            )

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
            "nvvidconv ! "
            f"video/x-raw, width={self.width}, height={self.height}, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! "
            "appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
        )

    def start(self):
        Gst.init(None)
        self.pipeline = Gst.parse_launch(self.pipeline_string())
        self.appsink = self.pipeline.get_by_name("sink")

        if self.appsink is None:
            raise RuntimeError("Could not create the GStreamer appsink.")

        result = self.pipeline.set_state(Gst.State.PLAYING)

        if result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Could not start the nvarguscamerasrc pipeline.")

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
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)


class OpenCVCamera:
    """Fallback for USB/V4L2 cameras."""

    def __init__(self, device=0, width=1280, height=720, fps=30):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.capture = None

    def start(self):
        self.capture = cv2.VideoCapture(self.device)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open OpenCV camera device {self.device}.")

    def read(self, timeout_ms=1000):
        del timeout_ms
        ok, frame = self.capture.read()
        return frame if ok else None

    def close(self):
        if self.capture is not None:
            self.capture.release()


def torch_cuda_status():
    if torch is None:
        return False, "PyTorch is not installed.", []

    messages = []

    try:
        torch_version = getattr(torch, "__version__", "unknown")
        torch_cuda = getattr(torch.version, "cuda", None)
        messages.append(f"PyTorch: {torch_version}")
        messages.append(f"PyTorch CUDA build: {torch_cuda or 'none'}")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            available = torch.cuda.is_available()

        warning_text = [str(item.message) for item in caught]

        if available:
            device_name = torch.cuda.get_device_name(0)
            messages.append(f"CUDA device 0: {device_name}")
            return True, "CUDA is available.", messages + warning_text

        return False, "torch.cuda.is_available() is false.", messages + warning_text
    except Exception as exc:
        return False, f"Could not initialize PyTorch CUDA: {exc}", messages


def choose_yolo_device(requested_device, model_path, require_gpu):
    suffix = Path(model_path).suffix.lower()

    if suffix == ".engine" and requested_device in {"auto", "cuda", "gpu", "0", "cuda:0"}:
        return None, True, "TensorRT engine selected; inference will run on the Jetson GPU."

    cuda_ok, cuda_reason, cuda_messages = torch_cuda_status()

    if requested_device == "auto":
        if cuda_ok:
            return 0, True, "Using CUDA device 0."

        if require_gpu:
            raise RuntimeError(format_cuda_error(cuda_reason, cuda_messages))

        return "cpu", False, format_cuda_error(cuda_reason, cuda_messages)

    if requested_device in {"cuda", "gpu", "0", "cuda:0"}:
        if cuda_ok:
            return 0, True, "Using CUDA device 0."

        raise RuntimeError(format_cuda_error(cuda_reason, cuda_messages))

    if requested_device == "cpu":
        return "cpu", False, "Using CPU because --device cpu was requested."

    return requested_device, requested_device != "cpu", f"Using YOLO device {requested_device}."


def format_cuda_error(reason, messages):
    detail = "\n".join(f"- {message}" for message in messages)
    return (
        "Jetson GPU inference is not available through PyTorch right now.\n"
        f"Reason: {reason}\n"
        f"{detail}\n"
        "Install a Jetson/L4T PyTorch wheel that matches this JetPack CUDA version, "
        "or use a TensorRT .engine model."
    )


def clamp_box(box, width, height):
    x1, y1, x2, y2 = [int(value) for value in box]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def get_hist(frame, box):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, width, height)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def color_backprojection(frame, hist):
    if hist is None:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backproj = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    return cv2.applyColorMap(backproj, cv2.COLORMAP_TURBO)


def compare_hist(h1, h2):
    if h1 is None or h2 is None:
        return 0.0

    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)


def box_area(box):
    x1, y1, x2, y2 = box[:4]
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_center(box):
    x1, y1, x2, y2 = box[:4]
    return (x1 + x2) // 2, (y1 + y2) // 2


class AdaptiveImageSize:
    def __init__(self, initial_imgsz, target_fps, enabled=True):
        self.target_fps = target_fps
        self.enabled = enabled and target_fps > 0
        self.steps = self._make_steps(initial_imgsz)
        self.index = 0
        self.last_change_time = 0.0

    def _make_steps(self, initial_imgsz):
        initial = normalize_imgsz(initial_imgsz)
        steps = [step for step in ADAPTIVE_IMGSZ_STEPS if step[0] <= initial[0] and step[1] <= initial[1]]

        if initial not in steps:
            steps.insert(0, initial)

        return steps

    @property
    def current(self):
        return self.steps[self.index]

    def maybe_reduce(self, measured_fps, now):
        if not self.enabled or self.index >= len(self.steps) - 1:
            return None

        if measured_fps <= 0 or now - self.last_change_time < 1.5:
            return None

        if measured_fps >= self.target_fps:
            return None

        self.index += 1
        self.last_change_time = now
        return self.current

    def at_minimum(self):
        return self.index >= len(self.steps) - 1


class AdaptiveDetectInterval:
    def __init__(self, initial_interval, target_fps, max_interval, enabled=True):
        self.current = max(1, initial_interval)
        self.target_fps = target_fps
        self.max_interval = max(1, max_interval)
        self.enabled = enabled and target_fps > 0
        self.last_change_time = 0.0

    def maybe_increase(self, measured_fps, now):
        if not self.enabled or self.current >= self.max_interval:
            return None

        if measured_fps <= 0 or now - self.last_change_time < 1.5:
            return None

        if measured_fps >= self.target_fps:
            return None

        self.current += 1
        self.last_change_time = now
        return self.current


def normalize_imgsz(imgsz):
    if isinstance(imgsz, int):
        return [imgsz, imgsz]

    if len(imgsz) == 1:
        return [imgsz[0], imgsz[0]]

    return [imgsz[0], imgsz[1]]


def detect_people(model, frame, args, yolo_device, use_half, imgsz):
    predict_kwargs = {
        "imgsz": imgsz,
        "conf": args.conf,
        "classes": [PERSON_CLASS_ID],
        "half": use_half,
        "verbose": False,
    }

    if yolo_device is not None:
        predict_kwargs["device"] = yolo_device

    if torch is not None:
        with torch.inference_mode():
            results = model.predict(frame, **predict_kwargs)
    else:
        results = model.predict(frame, **predict_kwargs)

    people = []

    if not results or results[0].boxes is None:
        return people

    height, width = frame.shape[:2]

    for box in results[0].boxes:
        cls_id = int(box.cls[0])

        if cls_id != PERSON_CLASS_ID:
            continue

        conf = float(box.conf[0])
        x1, y1, x2, y2 = clamp_box(box.xyxy[0].tolist(), width, height)
        area = box_area((x1, y1, x2, y2))

        if area < args.min_box_area:
            continue

        people.append((x1, y1, x2, y2, conf))

    return people


def pick_initial_target(frame, people):
    if not people:
        return None, None

    target = max(people, key=lambda item: (item[4], box_area(item)))
    target_box = target[:4]
    return target_box, get_hist(frame, target_box)


def match_existing_target(frame, people, target, target_hist, max_match_distance):
    if target is None:
        return None, None

    target_cx, target_cy = box_center(target)
    best_match = None
    best_hist = None
    best_score = -float("inf")

    for person in people:
        box = person[:4]
        conf = person[4]
        cx, cy = box_center(box)
        distance = abs(cx - target_cx) + abs(cy - target_cy)

        if distance > max_match_distance:
            continue

        hist = get_hist(frame, box)
        similarity = compare_hist(hist, target_hist)
        score = similarity * 100.0 + conf * 10.0 - distance * 0.5

        if score > best_score:
            best_score = score
            best_match = box
            best_hist = hist

    return best_match, best_hist


def smooth_target_box(smooth_box, target, alpha):
    x1, y1, x2, y2 = target

    if smooth_box is None:
        return [x1, y1, x2, y2]

    smooth_box[0] = int(alpha * smooth_box[0] + (1.0 - alpha) * x1)
    smooth_box[1] = int(alpha * smooth_box[1] + (1.0 - alpha) * y1)
    smooth_box[2] = int(alpha * smooth_box[2] + (1.0 - alpha) * x2)
    smooth_box[3] = int(alpha * smooth_box[3] + (1.0 - alpha) * y2)
    return smooth_box


def zoom_to_target(frame, box, zoom_smooth, args):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, width, height)
    center_x, center_y = box_center((x1, y1, x2, y2))
    box_width = max(1, x2 - x1)

    target_zoom = max(args.min_zoom, min(args.max_zoom, args.zoom_box_width / box_width))
    zoom_smooth = args.zoom_alpha * zoom_smooth + (1.0 - args.zoom_alpha) * target_zoom

    crop_width = max(1, int(width / zoom_smooth))
    crop_height = max(1, int(height / zoom_smooth))

    crop_x = center_x - crop_width // 2
    crop_y = center_y - crop_height // 2
    crop_x = max(0, min(crop_x, width - crop_width))
    crop_y = max(0, min(crop_y, height - crop_height))

    cropped = frame[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
    zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

    scale_x = width / crop_width
    scale_y = height / crop_height
    display_box = (
        int((x1 - crop_x) * scale_x),
        int((y1 - crop_y) * scale_y),
        int((x2 - crop_x) * scale_x),
        int((y2 - crop_y) * scale_y),
    )
    return zoomed, display_box, zoom_smooth


def draw_overlay(frame, display_box, target_locked, fps, color_locked=False):
    height, width = frame.shape[:2]

    if display_box is not None:
        x1, y1, x2, y2 = clamp_box(display_box, width, height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, box_center((x1, y1, x2, y2)), 5, (255, 0, 0), -1)

    cv2.circle(frame, (width // 2, height // 2), 5, (0, 0, 255), -1)
    status = "locked" if target_locked else "searching"
    color_status = "color" if color_locked else "no-color"
    cv2.putText(
        frame,
        f"{status}  {color_status}  fps: {fps:.1f}",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def make_camera(args):
    if args.camera == "argus":
        return ArgusCamera(args.sensor_id, args.width, args.height, args.fps)

    return OpenCVCamera(args.opencv_device, args.width, args.height, args.fps)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Jetson CSI camera person tracking with YOLO GPU support."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--camera", choices=("argus", "opencv"), default="argus")
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--opencv-device", type=int, default=0)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=DEFAULT_IMGSZ,
        metavar=("HEIGHT", "WIDTH"),
        help="YOLO input size. Default uses the 1536x864 landscape camera mode.",
    )
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default="cuda", help="cuda, auto, cpu, 0, or cuda:0")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--half", action="store_true", help="Use FP16 when running on CUDA.")
    parser.add_argument("--fp32", action="store_true", help="Disable FP16 on CUDA.")
    parser.add_argument("--detect-interval", type=int, default=1)
    parser.add_argument("--max-detect-interval", type=int, default=DEFAULT_MAX_DETECT_INTERVAL)
    parser.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS)
    parser.add_argument("--fixed-imgsz", action="store_true")
    parser.add_argument("--fixed-detect-interval", action="store_true")
    parser.add_argument("--adapt-warmup-sec", type=float, default=2.0)
    parser.add_argument("--min-box-area", type=int, default=1500)
    parser.add_argument("--max-match-distance", type=int, default=250)
    parser.add_argument("--smooth-alpha", type=float, default=0.3)
    parser.add_argument("--zoom-alpha", type=float, default=0.2)
    parser.add_argument("--min-zoom", type=float, default=1.2)
    parser.add_argument("--max-zoom", type=float, default=3.5)
    parser.add_argument("--zoom-box-width", type=float, default=DEFAULT_ZOOM_BOX_WIDTH)
    parser.add_argument("--no-zoom", action="store_true")
    parser.add_argument("--show-color-map", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument("--print-every-sec", type=float, default=1.0)
    parser.add_argument("--no-frame-timeout-sec", type=float, default=5.0)
    return parser.parse_args()


def main():
    args = parse_args()
    require_gpu = args.require_gpu or not args.allow_cpu
    args.imgsz = normalize_imgsz(args.imgsz)
    yolo_device, using_gpu, device_message = choose_yolo_device(
        args.device, args.model, require_gpu
    )
    use_half = using_gpu and (args.half or not args.fp32)

    print("=== Jetson Camera Smart Tracking ===")
    print("Camera backend:  ", "nvarguscamerasrc / GStreamer" if args.camera == "argus" else "OpenCV")
    print(f"Camera:          {args.width}x{args.height} @ {args.fps} FPS")
    print(f"Model:           {args.model}")
    print(f"YOLO imgsz/conf: {args.imgsz} / {args.conf}")
    print(f"Target FPS:      {args.target_fps:.1f}")
    print(f"Adaptive imgsz:  {not args.fixed_imgsz}")
    print(f"Detect every:    {args.detect_interval} frame(s)")
    print(f"Adaptive detect: {not args.fixed_detect_interval}")
    print(f"Zoom box width:  {args.zoom_box_width:.0f}px")
    print(f"YOLO device:     {yolo_device if yolo_device is not None else 'TensorRT engine'}")
    print(f"YOLO half:       {use_half}")
    print(device_message)
    print("Press q or Esc to quit.")
    print("")

    cv2.setUseOptimized(True)

    if using_gpu and torch is not None:
        torch.backends.cudnn.benchmark = True

    camera = make_camera(args)
    camera.start()
    model = YOLO(args.model)

    try:
        model.fuse()
    except Exception:
        pass

    adaptive_imgsz = AdaptiveImageSize(
        args.imgsz,
        args.target_fps,
        enabled=not args.fixed_imgsz,
    )
    adaptive_detect_interval = AdaptiveDetectInterval(
        args.detect_interval,
        args.target_fps,
        args.max_detect_interval,
        enabled=not args.fixed_detect_interval,
    )

    target = None
    target_hist = None
    smooth_box = None
    zoom_smooth = 1.0
    frame_id = 0
    frame_count = 0
    start_time = time.time()
    fps_start = start_time
    display_fps = 0.0
    last_print = 0.0
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
                        "No camera frames received. For the 15-pin CSI camera, "
                        "check the ribbon orientation, CAM0/CAM1 sensor-id, and nvargus-daemon."
                    )

                time.sleep(0.02)
                continue

            last_frame_time = time.time()
            frame_id += 1

            if target is None or frame_id % adaptive_detect_interval.current == 0:
                people = detect_people(
                    model,
                    frame,
                    args,
                    yolo_device,
                    use_half,
                    adaptive_imgsz.current,
                )

                if target is None:
                    target, target_hist = pick_initial_target(frame, people)
                else:
                    matched_target, matched_hist = match_existing_target(
                        frame,
                        people,
                        target,
                        target_hist,
                        args.max_match_distance,
                    )

                    if matched_target is not None:
                        target = matched_target
                        target_hist = matched_hist

            display_frame = frame
            display_box = None

            if target is not None and not args.no_display:
                smooth_box = smooth_target_box(smooth_box, target, args.smooth_alpha)

                if args.no_zoom:
                    display_box = smooth_box
                else:
                    display_frame, display_box, zoom_smooth = zoom_to_target(
                        frame,
                        smooth_box,
                        zoom_smooth,
                        args,
                    )

            frame_count += 1
            now = time.time()

            if now - fps_start >= 1.0:
                display_fps = frame_count / (now - fps_start)
                frame_count = 0
                fps_start = now

                if now - start_time >= args.adapt_warmup_sec:
                    next_imgsz = adaptive_imgsz.maybe_reduce(display_fps, now)

                    if next_imgsz is not None:
                        print(
                            f"adaptive-imgsz reduced to {next_imgsz} "
                            f"because fps={display_fps:.1f} target={args.target_fps:.1f}"
                        )
                    elif adaptive_imgsz.at_minimum():
                        next_interval = adaptive_detect_interval.maybe_increase(
                            display_fps,
                            now,
                        )

                        if next_interval is not None:
                            print(
                                f"adaptive-detect-interval increased to {next_interval} "
                                f"because fps={display_fps:.1f} target={args.target_fps:.1f}"
                            )

            if not args.no_display:
                draw_overlay(
                    display_frame,
                    display_box,
                    target is not None,
                    display_fps,
                    color_locked=target_hist is not None,
                )

            if now - last_print >= args.print_every_sec:
                last_print = now
                state = "locked" if target is not None else "searching"
                color_state = "yes" if target_hist is not None else "no"
                print(
                    f"state={state} fps={display_fps:.1f} "
                    f"imgsz={adaptive_imgsz.current} "
                    f"detect_interval={adaptive_detect_interval.current} "
                    f"color_hist={color_state} "
                    f"gpu={using_gpu}"
                )

            if not args.no_display:
                cv2.imshow(WINDOW_NAME, display_frame)

                if args.show_color_map:
                    backproj = color_backprojection(frame, target_hist)

                    if backproj is not None:
                        cv2.imshow("Target Color Map", backproj)

                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        camera.close()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
