import argparse
import html
import json
import math
import time
from pathlib import Path

from lidar_stl27l import LIDAR_BAUD, LIDAR_PORT, STL27LReader


DEFAULT_OUTPUT_PREFIX = "lidar_3d_scan"


def color_for_point(x, y, z, max_range_mm):
    distance = math.sqrt(x * x + y * y + z * z)
    t = max(0.0, min(1.0, distance / max_range_mm))
    height_t = max(0.0, min(1.0, (z + max_range_mm * 0.5) / max_range_mm))

    red = int(60 + 160 * t)
    green = int(80 + 140 * height_t)
    blue = int(220 - 120 * t)
    return red, green, blue


def vertical_yaw_point(point, yaw_deg, angle_offset_deg):
    # Assumes the LiDAR scan plane is vertical. Rotate the sensor around
    # the room at a smooth yaw rate while recording.
    distance = point["distance_mm"]
    elevation = math.radians(point["angle_deg"] + angle_offset_deg)
    yaw = math.radians(yaw_deg)

    horizontal = distance * math.cos(elevation)
    z = distance * math.sin(elevation)
    x = horizontal * math.sin(yaw)
    y = horizontal * math.cos(yaw)

    return x, y, z


def horizontal_pitch_point(point, pitch_deg, angle_offset_deg):
    # Assumes the LiDAR starts flat in a horizontal scan plane and you tilt
    # it smoothly through pitch_deg during capture.
    distance = point["distance_mm"]
    scan_angle = math.radians(point["angle_deg"] + angle_offset_deg)
    pitch = math.radians(pitch_deg)

    x0 = distance * math.sin(scan_angle)
    y0 = distance * math.cos(scan_angle)
    z0 = 0.0

    x = x0
    y = y0 * math.cos(pitch) - z0 * math.sin(pitch)
    z = y0 * math.sin(pitch) + z0 * math.cos(pitch)

    return x, y, z


def horizontal_only_point(point, yaw_deg, angle_offset_deg):
    # This is mostly for debugging. Rotating a flat 2D scan around yaw does
    # not create real height information, so z remains zero.
    distance = point["distance_mm"]
    scan_angle = math.radians(point["angle_deg"] + angle_offset_deg + yaw_deg)

    x = distance * math.sin(scan_angle)
    y = distance * math.cos(scan_angle)
    z = 0.0

    return x, y, z


def convert_point(point, mode, rotation_deg, angle_offset_deg):
    if mode == "vertical-yaw":
        return vertical_yaw_point(point, rotation_deg, angle_offset_deg)

    if mode == "horizontal-pitch":
        return horizontal_pitch_point(point, rotation_deg, angle_offset_deg)

    if mode == "horizontal-only":
        return horizontal_only_point(point, rotation_deg, angle_offset_deg)

    raise ValueError(f"Unsupported mode: {mode}")


def voxel_key(x, y, z, voxel_mm):
    return (
        int(round(x / voxel_mm)),
        int(round(y / voxel_mm)),
        int(round(z / voxel_mm)),
    )


def add_points_to_voxels(
    voxel_points,
    lidar_points,
    mode,
    rotation_deg,
    angle_offset_deg,
    voxel_mm,
    max_range_mm,
    max_points,
):
    added = 0

    for point in lidar_points:
        if point["distance_mm"] > max_range_mm:
            continue

        x, y, z = convert_point(point, mode, rotation_deg, angle_offset_deg)
        key = voxel_key(x, y, z, voxel_mm)

        if key in voxel_points:
            continue

        if len(voxel_points) >= max_points:
            break

        red, green, blue = color_for_point(x, y, z, max_range_mm)
        voxel_points[key] = {
            "x": x,
            "y": y,
            "z": z,
            "r": red,
            "g": green,
            "b": blue,
            "distance_mm": point["distance_mm"],
            "confidence": point["confidence"],
        }
        added += 1

    return added


def save_ply(path, points):
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for point in points:
            f.write(
                f"{point['x']:.1f} {point['y']:.1f} {point['z']:.1f} "
                f"{point['r']} {point['g']} {point['b']}\n"
            )


def save_csv(path, points):
    with path.open("w", encoding="utf-8") as f:
        f.write("x_mm,y_mm,z_mm,r,g,b,distance_mm,confidence\n")

        for point in points:
            f.write(
                f"{point['x']:.1f},{point['y']:.1f},{point['z']:.1f},"
                f"{point['r']},{point['g']},{point['b']},"
                f"{point['distance_mm']},{point['confidence']}\n"
            )


def viewer_points(points, limit):
    if len(points) <= limit:
        selected = points
    else:
        step = max(1, len(points) // limit)
        selected = points[::step][:limit]

    return [
        [
            round(point["x"], 1),
            round(point["y"], 1),
            round(point["z"], 1),
            point["r"],
            point["g"],
            point["b"],
        ]
        for point in selected
    ]


def save_html_viewer(path, points, title, viewer_limit):
    data = viewer_points(points, viewer_limit)
    point_json = json.dumps(data, separators=(",", ":"))
    escaped_title = html.escape(title)

    content = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{escaped_title}</title>
  <style>
    body {{
      margin: 0;
      background: #101214;
      color: #e8edf2;
      font-family: Arial, sans-serif;
      overflow: hidden;
    }}
    #hud {{
      position: fixed;
      top: 12px;
      left: 12px;
      background: rgba(0, 0, 0, 0.55);
      padding: 10px 12px;
      border-radius: 6px;
      font-size: 13px;
      line-height: 1.45;
      user-select: none;
    }}
    canvas {{
      display: block;
      width: 100vw;
      height: 100vh;
    }}
  </style>
</head>
<body>
<canvas id="view"></canvas>
<div id="hud">
  <strong>{escaped_title}</strong><br>
  Points shown: <span id="count"></span><br>
  Drag: rotate | Wheel: zoom | R: reset
</div>
<script>
const points = {point_json};
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
document.getElementById("count").textContent = points.length;

let rotZ = -0.6;
let rotX = 0.8;
let zoom = 1.0;
let dragging = false;
let lastX = 0;
let lastY = 0;

function resize() {{
  canvas.width = window.innerWidth * window.devicePixelRatio;
  canvas.height = window.innerHeight * window.devicePixelRatio;
}}

function bounds() {{
  if (!points.length) {{
    return {{cx: 0, cy: 0, cz: 0, radius: 1}};
  }}

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (const p of points) {{
    minX = Math.min(minX, p[0]);
    minY = Math.min(minY, p[1]);
    minZ = Math.min(minZ, p[2]);
    maxX = Math.max(maxX, p[0]);
    maxY = Math.max(maxY, p[1]);
    maxZ = Math.max(maxZ, p[2]);
  }}

  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const cz = (minZ + maxZ) / 2;
  const radius = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1) / 2;
  return {{cx, cy, cz, radius}};
}}

const cloud = bounds();

function project(rawX, rawY, rawZ) {{
  let x = rawX - cloud.cx;
  let y = rawY - cloud.cy;
  let z = rawZ - cloud.cz;

  const cz = Math.cos(rotZ);
  const sz = Math.sin(rotZ);
  const x1 = x * cz - y * sz;
  const y1 = x * sz + y * cz;
  const z1 = z;

  const cx = Math.cos(rotX);
  const sx = Math.sin(rotX);
  const x2 = x1;
  const y2 = y1 * cx - z1 * sx;
  const z2 = y1 * sx + z1 * cx;

  const baseScale = Math.min(canvas.width, canvas.height) * 0.42 / cloud.radius;
  const depth = cloud.radius * 4;
  const perspective = depth / (depth - z2);
  const scale = baseScale * zoom * perspective;

  return {{
    x: canvas.width / 2 + x2 * scale,
    y: canvas.height / 2 - y2 * scale,
    z: z2
  }};
}}

function drawAxis() {{
  const origin = project(cloud.cx, cloud.cy, cloud.cz);
  const len = cloud.radius * 0.35;
  const axes = [
    [cloud.cx + len, cloud.cy, cloud.cz, "#ff6666", "X"],
    [cloud.cx, cloud.cy + len, cloud.cz, "#66ff99", "Y"],
    [cloud.cx, cloud.cy, cloud.cz + len, "#66aaff", "Z"]
  ];

  ctx.lineWidth = 2 * window.devicePixelRatio;
  ctx.font = `${{14 * window.devicePixelRatio}}px Arial`;

  for (const axis of axes) {{
    const end = project(axis[0], axis[1], axis[2]);
    ctx.strokeStyle = axis[3];
    ctx.fillStyle = axis[3];
    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
    ctx.fillText(axis[4], end.x + 4, end.y + 4);
  }}
}}

function draw() {{
  ctx.fillStyle = "#101214";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  drawAxis();

  const projected = [];
  for (const p of points) {{
    const q = project(p[0], p[1], p[2]);
    projected.push([q.x, q.y, q.z, p[3], p[4], p[5]]);
  }}

  projected.sort((a, b) => a[2] - b[2]);

  const pointSize = Math.max(1.5, 2.5 * window.devicePixelRatio);
  for (const p of projected) {{
    ctx.fillStyle = `rgb(${{p[3]}},${{p[4]}},${{p[5]}})`;
    ctx.fillRect(p[0], p[1], pointSize, pointSize);
  }}

  requestAnimationFrame(draw);
}}

canvas.addEventListener("mousedown", (event) => {{
  dragging = true;
  lastX = event.clientX;
  lastY = event.clientY;
}});

window.addEventListener("mouseup", () => {{
  dragging = false;
}});

window.addEventListener("mousemove", (event) => {{
  if (!dragging) return;

  const dx = event.clientX - lastX;
  const dy = event.clientY - lastY;
  lastX = event.clientX;
  lastY = event.clientY;
  rotZ += dx * 0.007;
  rotX += dy * 0.007;
}});

canvas.addEventListener("wheel", (event) => {{
  event.preventDefault();
  zoom *= event.deltaY > 0 ? 0.9 : 1.1;
  zoom = Math.max(0.1, Math.min(8.0, zoom));
}}, {{passive: false}});

window.addEventListener("keydown", (event) => {{
  if (event.key.toLowerCase() === "r") {{
    rotZ = -0.6;
    rotX = 0.8;
    zoom = 1.0;
  }}
}});

window.addEventListener("resize", resize);
resize();
draw();
</script>
</body>
</html>
"""

    path.write_text(content, encoding="utf-8")


def print_mode_instructions(args):
    print("=== 2D LiDAR Approximate 3D Mapper ===")
    print("This script does not touch the flight controller or motors.")
    print("It saves a sparse 3D point cloud from the 2D LiDAR.")
    print("")

    if args.mode == "vertical-yaw":
        print("Physical setup:")
        print("- Hold or mount the LiDAR so its scan plane is vertical.")
        print("- Rotate it smoothly around the room during the capture.")
        print("- The script guesses yaw from elapsed time.")
    elif args.mode == "horizontal-pitch":
        print("Physical setup:")
        print("- Keep the LiDAR mostly flat and tilt it smoothly during capture.")
        print("- The script guesses pitch from elapsed time.")
    else:
        print("Physical setup:")
        print("- This debug mode keeps z=0, so it is not real 3D.")

    print("")
    print(f"LiDAR:        {args.lidar_port} @ {args.lidar_baud}")
    print(f"Mode:         {args.mode}")
    print(f"Duration:     {args.duration_sec:.1f}s")
    print(f"Rotation:     {args.start_deg:.1f} to {args.end_deg:.1f} deg")
    print(f"Voxel size:   {args.voxel_mm:.1f} mm")
    print(f"Max points:   {args.max_points}")
    print(f"Max range:    {args.max_range_mm} mm")
    print("")
    print("Press Ctrl-C to stop early and save what was captured.")
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="Build an approximate sparse 3D point cloud from a 2D LiDAR sweep."
    )
    parser.add_argument("--lidar-port", default=LIDAR_PORT)
    parser.add_argument("--lidar-baud", type=int, default=LIDAR_BAUD)
    parser.add_argument(
        "--mode",
        choices=("vertical-yaw", "horizontal-pitch", "horizontal-only"),
        default="vertical-yaw",
    )
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--start-deg", type=float, default=0.0)
    parser.add_argument("--end-deg", type=float, default=360.0)
    parser.add_argument("--angle-offset-deg", type=float, default=0.0)
    parser.add_argument("--sample-hz", type=float, default=20.0)
    parser.add_argument("--voxel-mm", type=float, default=50.0)
    parser.add_argument("--max-range-mm", type=int, default=6000)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--viewer-max-points", type=int, default=12000)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    args = parser.parse_args()

    if args.duration_sec <= 0:
        raise SystemExit("--duration-sec must be greater than zero")

    if args.voxel_mm <= 0:
        raise SystemExit("--voxel-mm must be greater than zero")

    if args.max_points <= 0:
        raise SystemExit("--max-points must be greater than zero")

    print_mode_instructions(args)

    lidar = STL27LReader(args.lidar_port, args.lidar_baud, point_max_age_sec=0.2)
    voxel_points = {}
    start_time = time.time()
    last_print = 0.0
    sleep_sec = 1.0 / max(1.0, args.sample_hz)

    try:
        lidar.start()

        while True:
            now = time.time()
            elapsed = now - start_time
            progress = min(1.0, elapsed / args.duration_sec)
            rotation_deg = args.start_deg + (args.end_deg - args.start_deg) * progress
            lidar_points = lidar.get_points()

            added = add_points_to_voxels(
                voxel_points,
                lidar_points,
                args.mode,
                rotation_deg,
                args.angle_offset_deg,
                args.voxel_mm,
                args.max_range_mm,
                args.max_points,
            )

            if now - last_print >= 1.0:
                last_print = now
                print(
                    f"time={elapsed:5.1f}s rotation={rotation_deg:7.1f} deg "
                    f"live_points={len(lidar_points):3d} saved={len(voxel_points):5d} "
                    f"added={added:3d}"
                )

            if progress >= 1.0 or len(voxel_points) >= args.max_points:
                break

            time.sleep(sleep_sec)

    except KeyboardInterrupt:
        print("\nStopping early and saving captured points.")

    finally:
        lidar.close()

    points = list(voxel_points.values())
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = Path(f"{args.output_prefix}_{timestamp}")
    ply_path = prefix.with_suffix(".ply")
    csv_path = prefix.with_suffix(".csv")
    html_path = prefix.with_suffix(".html")

    save_ply(ply_path, points)
    save_csv(csv_path, points)
    save_html_viewer(
        html_path,
        points,
        f"{prefix.name} ({len(points)} points)",
        args.viewer_max_points,
    )

    print("")
    print(f"Saved {len(points)} sparse points.")
    print(f"PLY:  {ply_path}")
    print(f"CSV:  {csv_path}")
    print(f"HTML: {html_path}")
    print("")
    print("Open the HTML file in a browser to rotate the approximate 3D view.")


if __name__ == "__main__":
    main()
