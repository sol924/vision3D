from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layout import open_local_mesh, open_local_pointcloud


GT_COLOR = "#1a8f3a"
PRED_COLOR = "#e2552f"
TEXT_DARK = "#1d252c"
TEXT_MUTED = "#66717c"
WHITE = "#ffffff"
SCENE_BG = np.array([255, 255, 255], dtype=np.int16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a static left-text/right-scene PNG report from sample_package.json."
    )
    parser.add_argument("sample_package_json", help="Path to a sample_package.json file.")
    parser.add_argument(
        "--output-png",
        default=None,
        help="Output PNG path. Defaults next to sample_package.json.",
    )
    parser.add_argument("--width", type=int, default=1800, help="Output image width.")
    parser.add_argument("--height", type=int, default=1100, help="Output image height.")
    parser.add_argument(
        "--scene-mode",
        choices=("auto", "mesh", "point"),
        default="mesh",
        help="Which generated scene artifact to render. Defaults to mesh for paper figures.",
    )
    parser.add_argument("--max-points", type=int, default=80000, help="Maximum rendered points.")
    parser.add_argument(
        "--renderer",
        choices=("auto", "open3d", "matplotlib"),
        default="auto",
        help="Static scene renderer. Auto tries Open3D, then Matplotlib.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_scene(package: dict, scene_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str, Path]:
    mode = scene_mode
    if mode == "auto":
        mode = "mesh"

    if mode == "mesh":
        path = resolve_path(package["mesh_outputs"]["mesh_ply"])
        points, colors, faces = open_local_mesh.load_ascii_mesh_ply(path)
        return points, colors, faces, "mesh", path

    path = resolve_path(package["point_outputs"]["overlay_ply"])
    points, colors = open_local_pointcloud.load_ascii_ply(path)
    return points, colors, None, "point", path


def bbox_corners(loc: list[float]) -> np.ndarray:
    center = np.asarray(loc[:3], dtype=np.float32)
    size = np.asarray(loc[3:6], dtype=np.float32)
    half = size / 2.0
    return np.array(
        [
            [center[0] - half[0], center[1] - half[1], center[2] - half[2]],
            [center[0] + half[0], center[1] - half[1], center[2] - half[2]],
            [center[0] + half[0], center[1] + half[1], center[2] - half[2]],
            [center[0] - half[0], center[1] + half[1], center[2] - half[2]],
            [center[0] - half[0], center[1] - half[1], center[2] + half[2]],
            [center[0] + half[0], center[1] - half[1], center[2] + half[2]],
            [center[0] + half[0], center[1] + half[1], center[2] + half[2]],
            [center[0] - half[0], center[1] + half[1], center[2] + half[2]],
        ],
        dtype=np.float32,
    )


def bbox_edges() -> list[tuple[int, int]]:
    return [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]


def collect_boxes(package: dict) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    if package.get("gt_bbox_loc"):
        boxes.append({"label": "GT", "loc": package["gt_bbox_loc"], "color": GT_COLOR, "width": 2.8})
    for item in package.get("model_pred_bbox_locs", []):
        if item.get("loc"):
            boxes.append(
                {
                    "label": f"Pred OBJ{int(item['object_id']):03d}",
                    "loc": item["loc"],
                    "color": PRED_COLOR,
                    "width": 2.4,
                }
            )
    return boxes


def target_limits(points: np.ndarray, boxes: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    scene_min = np.nanmin(points, axis=0)
    scene_max = np.nanmax(points, axis=0)
    if not boxes:
        return scene_min, scene_max

    corners = np.concatenate([bbox_corners(box["loc"]) for box in boxes], axis=0)
    target_min = np.min(corners, axis=0)
    target_max = np.max(corners, axis=0)
    target_center = (target_min + target_max) / 2.0
    target_span = np.maximum(target_max - target_min, 0.1)
    scene_span = np.maximum(scene_max - scene_min, 0.1)
    span = np.maximum(target_span * 3.2, scene_span * 0.35)
    span = np.minimum(span, scene_span)
    view_min = np.maximum(target_center - span / 2.0, scene_min)
    view_max = np.minimum(target_center + span / 2.0, scene_max)
    return view_min, view_max


def downsample(points: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= max_points:
        return points, colors
    idx = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[idx], colors[idx]


def draw_matplotlib_scene(
    points: np.ndarray,
    colors: np.ndarray,
    boxes: list[dict[str, Any]],
    size: tuple[int, int],
    elev: float,
    azim: float,
    max_points: int,
    faces: np.ndarray | None = None,
) -> Image.Image:
    width, height = size
    fig = plt.figure(figsize=(width / 160, height / 160), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    mins, maxs = target_limits(points, boxes)
    span = np.maximum(maxs - mins, 0.1)
    center = (mins + maxs) / 2.0
    max_span = float(np.max(span))
    limits_min = center - max_span / 2.0
    limits_max = center + max_span / 2.0

    rendered_mesh = False
    if faces is not None and len(faces) > 0:
        vertices_in_view = np.all((points >= limits_min) & (points <= limits_max), axis=1)
        face_mask = vertices_in_view[faces].any(axis=1)
        face_indices = np.flatnonzero(face_mask)
        max_faces = max(80000, min(240000, max_points * 3))
        if len(face_indices) > max_faces:
            face_indices = face_indices[np.linspace(0, len(face_indices) - 1, max_faces, dtype=np.int64)]
        selected_faces = faces[face_indices]
        if len(selected_faces) > 0:
            triangles = points[selected_faces]
            face_colors = np.clip(colors[selected_faces].mean(axis=1).astype(np.float32) / 255.0, 0, 1)
            mesh = Poly3DCollection(
                triangles,
                facecolors=face_colors,
                edgecolors="none",
                linewidths=0.0,
                alpha=1.0,
            )
            ax.add_collection3d(mesh)
            rendered_mesh = True

    if not rendered_mesh:
        points_small, colors_small = downsample(points, colors, max_points)
        colors_float = np.clip(colors_small.astype(np.float32) / 255.0, 0, 1)
        ax.scatter(
            points_small[:, 0],
            points_small[:, 1],
            points_small[:, 2],
            c=colors_float,
            s=0.16,
            linewidths=0,
            depthshade=False,
        )

    for box in boxes:
        corners = bbox_corners(box["loc"])
        for start, end in bbox_edges():
            ax.plot(
                [corners[start, 0], corners[end, 0]],
                [corners[start, 1], corners[end, 1]],
                [corners[start, 2], corners[end, 2]],
                color=box["color"],
                linewidth=box["width"],
                solid_capstyle="round",
            )

    ax.set_xlim(limits_min[0], limits_max[0])
    ax.set_ylim(limits_min[1], limits_max[1])
    ax.set_zlim(limits_min[2], limits_max[2])
    ax.set_box_aspect((1, 1, 0.72))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.margins(0)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB").resize(size, Image.Resampling.LANCZOS)


def score_scene(image: Image.Image) -> float:
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    diff = np.abs(arr - SCENE_BG).sum(axis=2)
    occupied = diff > 34
    coverage = occupied.mean()
    green = (arr[:, :, 1] > 105) & (arr[:, :, 0] < 90)
    orange = (arr[:, :, 0] > 170) & (arr[:, :, 1] < 130)
    target = green | orange
    target_score = min(float(target.sum()) / 900.0, 1.5)
    coverage_score = 1.0 - min(abs(float(coverage) - 0.34) / 0.34, 1.0)
    return target_score * 2.0 + coverage_score


def trim_scene_whitespace(image: Image.Image, padding: int = 34) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    diff = np.abs(arr - SCENE_BG).sum(axis=2)
    mask = diff > 34
    if not mask.any():
        return image
    ys, xs = np.where(mask)
    x0 = max(int(xs.min()) - padding, 0)
    x1 = min(int(xs.max()) + padding, image.width - 1)
    y0 = max(int(ys.min()) - padding, 0)
    y1 = min(int(ys.max()) + padding, image.height - 1)
    if x1 <= x0 or y1 <= y0:
        return image
    return image.crop((x0, y0, x1 + 1, y1 + 1))


def choose_camera(points: np.ndarray, colors: np.ndarray, boxes: list[dict[str, Any]]) -> dict[str, float]:
    candidates = [
        {"name": "front-right", "elev": 24.0, "azim": -54.0},
        {"name": "front-left", "elev": 24.0, "azim": 36.0},
        {"name": "high-front", "elev": 38.0, "azim": -42.0},
        {"name": "high-left", "elev": 42.0, "azim": 48.0},
        {"name": "back-right", "elev": 28.0, "azim": -132.0},
        {"name": "top-oblique", "elev": 58.0, "azim": -74.0},
    ]
    best = candidates[0] | {"score": -math.inf}
    for candidate in candidates:
        image = draw_matplotlib_scene(
            points,
            colors,
            boxes,
            size=(520, 380),
            elev=candidate["elev"],
            azim=candidate["azim"],
            max_points=22000,
        )
        score = score_scene(image)
        if score > best["score"]:
            best = candidate | {"score": score}
    return best


def try_render_open3d(
    points: np.ndarray,
    colors: np.ndarray,
    faces: np.ndarray | None,
    boxes: list[dict[str, Any]],
    size: tuple[int, int],
    camera: dict[str, float],
) -> Image.Image | None:
    try:
        import open3d as o3d
    except Exception:
        return None

    try:
        width, height = size
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        if faces is not None and len(faces) > 0:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(points.astype(np.float64))
            mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors.astype(np.float64) / 255.0, 0, 1))
            renderer.scene.add_geometry("scene_mesh", mesh, material)
        else:
            material.point_size = 2.0
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(np.clip(colors.astype(np.float64) / 255.0, 0, 1))
            renderer.scene.add_geometry("scene_points", pcd, material)

        line_material = o3d.visualization.rendering.MaterialRecord()
        line_material.shader = "unlitLine"
        line_material.line_width = 4.0
        for idx, box in enumerate(boxes):
            corners = bbox_corners(box["loc"]).astype(np.float64)
            lines = np.asarray(bbox_edges(), dtype=np.int32)
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(corners),
                lines=o3d.utility.Vector2iVector(lines),
            )
            rgb = tuple(int(box["color"].lstrip("#")[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
            line_set.colors = o3d.utility.Vector3dVector(np.tile(rgb, (len(lines), 1)))
            renderer.scene.add_geometry(f"bbox_{idx}", line_set, line_material)

        mins, maxs = target_limits(points, boxes)
        center = (mins + maxs) / 2.0
        radius = float(np.linalg.norm(maxs - mins))
        elev = math.radians(camera["elev"])
        azim = math.radians(camera["azim"])
        eye = center + radius * np.array(
            [math.cos(elev) * math.cos(azim), math.cos(elev) * math.sin(azim), math.sin(elev)]
        )
        renderer.setup_camera(42.0, center.tolist(), eye.tolist(), [0, 0, 1])
        image = renderer.render_to_image()
        renderer.scene.clear_geometry()
        return Image.fromarray(np.asarray(image)).convert("RGB")
    except Exception:
        return None


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.ImageFont, width: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=text_font) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def draw_section(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    label: str,
    body: str,
    accent: str,
) -> int:
    label_font = font(23, bold=True)
    body_font = font(25)
    draw.text((x, y), label, fill=accent, font=label_font)
    y += 38
    lines = wrap_text(draw, body, body_font, width)
    for line in lines[:8]:
        draw.text((x, y), line, fill=TEXT_DARK, font=body_font)
        y += 36
    if len(lines) > 8:
        draw.text((x, y), "...", fill=TEXT_MUTED, font=body_font)
        y += 36
    return y + 34


def measure_section_height(
    draw: ImageDraw.ImageDraw,
    width: int,
    body: str,
    max_lines: int = 8,
) -> int:
    body_font = font(25)
    line_count = min(len(wrap_text(draw, body, body_font, width)), max_lines)
    ellipsis = 1 if len(wrap_text(draw, body, body_font, width)) > max_lines else 0
    return 38 + (line_count + ellipsis) * 36 + 34


def measure_legend_height() -> int:
    return 62


def draw_bbox_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> int:
    legend_font = font(19)
    line_w = 44
    text_gap = 14
    row_gap = 28
    for idx, (color, label) in enumerate(((GT_COLOR, "GT bbox"), (PRED_COLOR, "Prediction bbox"))):
        row_y = y + idx * row_gap
        draw.line((x, row_y + 10, x + line_w, row_y + 10), fill=color, width=6)
        draw.text((x + line_w + text_gap, row_y), label, fill=TEXT_MUTED, font=legend_font)
    return y + measure_legend_height()


def render_report(
    package: dict,
    scene_image: Image.Image,
    output_png: Path,
    size: tuple[int, int],
    camera: dict[str, float],
    renderer_name: str,
    scene_kind: str,
) -> None:
    width, height = size
    canvas = Image.new("RGB", size, WHITE)
    draw = ImageDraw.Draw(canvas)

    left_x = 54
    left_w = 500
    text_w = 455
    right_x = 590
    right_y = 40
    right_w = width - right_x - 42
    right_h = height - 80

    gt_text = ", ".join(package.get("gt_answer", []))
    pred_text = package.get("model_prediction", "")
    sections = [
        ("Instruction", package.get("input_text", ""), "#2f5f87"),
        ("GT", gt_text, GT_COLOR),
        ("PREDICTION", pred_text, PRED_COLOR),
    ]
    total_text_h = (
        sum(measure_section_height(draw, text_w, body) for _label, body, _accent in sections)
        + measure_legend_height()
    )
    y = right_y + max((right_h - total_text_h) // 2, 0)
    for label, body, accent in sections:
        y = draw_section(draw, left_x, y, text_w, label, body, accent)
    draw_bbox_legend(draw, left_x, y - 4)

    scene = trim_scene_whitespace(scene_image).resize((right_w, right_h), Image.Resampling.LANCZOS)
    canvas.paste(scene, (right_x, right_y))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png)


def main() -> int:
    args = parse_args()
    package_json = resolve_path(args.sample_package_json)
    package = load_json(package_json)
    output_png = (
        resolve_path(args.output_png)
        if args.output_png
        else package_json.with_name("sample_report.png")
    )

    points, colors, faces, scene_kind, scene_path = load_scene(package, args.scene_mode)
    boxes = collect_boxes(package)
    camera = choose_camera(points, colors, boxes)
    scene_size = (args.width - 620, args.height - 80)
    renderer_name = "matplotlib"
    scene_image = None
    if args.renderer in ("auto", "open3d"):
        scene_image = try_render_open3d(points, colors, faces, boxes, scene_size, camera)
        if scene_image is not None:
            renderer_name = "open3d"
    if scene_image is None:
        if args.renderer == "open3d":
            raise RuntimeError("Open3D rendering was requested but failed.")
        scene_image = draw_matplotlib_scene(
            points,
            colors,
            boxes,
            size=scene_size,
            elev=camera["elev"],
            azim=camera["azim"],
            max_points=args.max_points,
            faces=faces,
        )

    render_report(
        package=package,
        scene_image=scene_image,
        output_png=output_png,
        size=(args.width, args.height),
        camera=camera,
        renderer_name=renderer_name,
        scene_kind=scene_kind,
    )
    summary = {
        "report_png": str(output_png.relative_to(REPO_ROOT) if output_png.is_relative_to(REPO_ROOT) else output_png),
        "scene_source": str(scene_path.relative_to(REPO_ROOT) if scene_path.is_relative_to(REPO_ROOT) else scene_path),
        "scene_kind": scene_kind,
        "renderer": renderer_name,
        "camera": camera,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
