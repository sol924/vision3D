from __future__ import annotations

import argparse
import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layout import open_local_mesh, open_local_pointcloud


GT_COLOR = "#1a8f3a"
PRED_COLOR = "#e2552f"
TEXT_DARK = "#000000"
WHITE = "#ffffff"
SCENE_BG = np.array([255, 255, 255], dtype=np.int16)


@dataclass(frozen=True)
class ReportStyle:
    text_size: int
    min_text_size: int
    line_step: int
    section_label_gap: int
    section_bottom_gap: int
    legend_row_gap: int
    legend_base_height: int
    left_x: int
    text_w: int
    right_x: int
    right_y: int
    right_margin: int
    bottom_margin: int
    legend_line_w: int
    legend_text_gap: int
    legend_line_width: int
    legend_line_y: int
    font_family: str


STYLE_PRESETS = {
    "compact": {
        "text_size": 25,
        "min_text_size": 20,
        "line_step": 36,
        "section_label_gap": 38,
        "section_bottom_gap": 34,
        "legend_row_gap": 36,
        "legend_base_height": 42,
        "left_x": 54,
        "text_w": 455,
        "right_x": 590,
        "right_y": 40,
        "right_margin": 42,
        "bottom_margin": 40,
        "legend_line_w": 44,
        "legend_text_gap": 14,
        "legend_line_width": 6,
        "legend_line_y": 14,
    },
    "a4-grid": {
        "text_size": 62,
        "min_text_size": 46,
        "line_step": 70,
        "section_label_gap": 68,
        "section_bottom_gap": 34,
        "legend_row_gap": 70,
        "legend_base_height": 78,
        "left_x": 60,
        "text_w": 780,
        "right_x": 880,
        "right_y": 40,
        "right_margin": 36,
        "bottom_margin": 40,
        "legend_line_w": 90,
        "legend_text_gap": 22,
        "legend_line_width": 10,
        "legend_line_y": 32,
    },
    "a4-2x6": {
        "text_size": 78,
        "min_text_size": 54,
        "line_step": 84,
        "section_label_gap": 70,
        "section_bottom_gap": 22,
        "legend_row_gap": 84,
        "legend_base_height": 92,
        "left_x": 70,
        "text_w": 1340,
        "right_x": 1500,
        "right_y": 40,
        "right_margin": 50,
        "bottom_margin": 40,
        "legend_line_w": 112,
        "legend_text_gap": 28,
        "legend_line_width": 12,
        "legend_line_y": 42,
    },
}

FONT_CANDIDATES = {
    "times": {
        False: [
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            "/System/Library/Fonts/Times.ttc",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        ],
        True: [
            "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
            "/System/Library/Fonts/Times.ttc",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        ],
    },
    "arial": {
        False: [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ],
        True: [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ],
    },
}


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
    parser.add_argument(
        "--text-preset",
        choices=("compact", "a4-grid", "a4-2x6"),
        default="compact",
        help="Left-text layout preset. Use a4-2x6 when a 2x6 grid will be placed on A4 paper.",
    )
    parser.add_argument(
        "--font-family",
        choices=("times", "arial"),
        default="times",
        help="Font family for the left text block and bbox legend.",
    )
    parser.add_argument(
        "--scene-scale",
        type=float,
        default=1.0,
        help=(
            "Internal scene-render scale relative to the right panel. "
            "The final report dimensions and text resolution are unchanged."
        ),
    )
    parser.add_argument(
        "--scene-colors",
        type=int,
        default=0,
        help=(
            "Maximum colors used in the scene panel after resizing. "
            "Use 0 to preserve full RGB color."
        ),
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
    gt_items = [item for item in package.get("gt_bbox_locs", []) if item.get("loc")]
    if gt_items:
        for item in gt_items:
            object_id = item.get("object_id")
            label = "GT" if object_id is None else f"GT OBJ{int(object_id):03d}"
            boxes.append({"label": label, "loc": item["loc"], "color": GT_COLOR, "width": 2.8})
    elif package.get("gt_bbox_loc"):
        boxes.append(
            {
                "label": f"GT OBJ{int(package.get('gt_object_id', 0)):03d}",
                "loc": package["gt_bbox_loc"],
                "color": GT_COLOR,
                "width": 2.8,
            }
        )
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


def collect_legend_items(package: dict) -> list[tuple[str, str]]:
    has_gt = bool(package.get("gt_bbox_loc")) or any(
        item.get("loc") for item in package.get("gt_bbox_locs", [])
    )
    has_pred = any(item.get("loc") for item in package.get("model_pred_bbox_locs", []))
    items: list[tuple[str, str]] = []
    if has_gt:
        items.append((GT_COLOR, "GT bbox"))
    if has_pred:
        items.append((PRED_COLOR, "Prediction bbox"))
    return items


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


def camera_basis(elev: float, azim: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elev_rad = math.radians(elev)
    azim_rad = math.radians(azim)
    view_dir = np.array(
        [
            math.cos(elev_rad) * math.cos(azim_rad),
            math.cos(elev_rad) * math.sin(azim_rad),
            math.sin(elev_rad),
        ],
        dtype=np.float32,
    )
    view_dir /= max(float(np.linalg.norm(view_dir)), 1e-6)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(world_up, view_dir)
    if float(np.linalg.norm(right)) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right /= max(float(np.linalg.norm(right)), 1e-6)
    up = np.cross(view_dir, right)
    up /= max(float(np.linalg.norm(up)), 1e-6)
    return right, up, view_dir


def project_vertices(
    vertices: np.ndarray,
    center: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    right, up, view_dir = (item.astype(np.float64) for item in basis)
    rel = vertices.astype(np.float64, copy=False) - center.astype(np.float64, copy=False)
    return np.stack(
        (
            np.einsum("...i,i->...", rel, right),
            np.einsum("...i,i->...", rel, up),
            np.einsum("...i,i->...", rel, view_dir),
        ),
        axis=-1,
    )


def expand_projected_limits(xy: np.ndarray, size: tuple[int, int]) -> tuple[float, float, float, float]:
    width, height = size
    x0, y0 = np.min(xy, axis=0)
    x1, y1 = np.max(xy, axis=0)
    x_span = max(float(x1 - x0), 1e-3)
    y_span = max(float(y1 - y0), 1e-3)
    target_aspect = width / height
    data_aspect = x_span / y_span
    if data_aspect > target_aspect:
        new_y_span = x_span / target_aspect
        pad = (new_y_span - y_span) / 2.0
        y0 -= pad
        y1 += pad
    else:
        new_x_span = y_span * target_aspect
        pad = (new_x_span - x_span) / 2.0
        x0 -= pad
        x1 += pad
    pad_x = (x1 - x0) * 0.045
    pad_y = (y1 - y0) * 0.045
    return float(x0 - pad_x), float(x1 + pad_x), float(y0 - pad_y), float(y1 + pad_y)


def draw_projected_mesh_scene(
    points: np.ndarray,
    colors: np.ndarray,
    faces: np.ndarray,
    boxes: list[dict[str, Any]],
    size: tuple[int, int],
    elev: float,
    azim: float,
    max_points: int,
) -> Image.Image | None:
    mins, maxs = target_limits(points, boxes)
    span = np.maximum(maxs - mins, 0.1)
    center = (mins + maxs) / 2.0
    max_span = float(np.max(span))
    limits_min = center - max_span / 2.0
    limits_max = center + max_span / 2.0

    vertices_in_view = np.all((points >= limits_min) & (points <= limits_max), axis=1)
    face_mask = vertices_in_view[faces].any(axis=1)
    face_indices = np.flatnonzero(face_mask)
    if len(face_indices) == 0:
        return None

    max_faces = max(120000, min(700000, max_points * 8))
    if len(face_indices) > max_faces:
        face_indices = face_indices[np.linspace(0, len(face_indices) - 1, max_faces, dtype=np.int64)]

    selected_faces = faces[face_indices]
    triangles = points[selected_faces]
    face_colors = np.clip(colors[selected_faces].mean(axis=1).astype(np.float32) / 255.0, 0, 1)
    basis = camera_basis(elev, azim)
    projected = project_vertices(triangles.reshape(-1, 3), center, basis).reshape(-1, 3, 3)

    projected_xy = projected[:, :, :2].reshape(-1, 2)
    if boxes:
        box_vertices = np.concatenate([bbox_corners(box["loc"]) for box in boxes], axis=0)
        box_projected = project_vertices(box_vertices, center, basis)[:, :2]
        projected_xy = np.concatenate((projected_xy, box_projected), axis=0)
    x0, x1, y0, y1 = expand_projected_limits(projected_xy, size)

    depth = projected[:, :, 2].mean(axis=1)
    order = np.argsort(depth)
    polygons = projected[order, :, :2]
    face_colors = face_colors[order]

    width, height = size
    fig, ax = plt.subplots(figsize=(width / 160, height / 160), dpi=160)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_position([0, 0, 1, 1])
    mesh = PolyCollection(
        polygons,
        facecolors=face_colors,
        edgecolors=face_colors,
        linewidths=0.12,
        antialiaseds=False,
    )
    ax.add_collection(mesh)

    for box in boxes:
        corners = bbox_corners(box["loc"])
        projected_corners = project_vertices(corners, center, basis)
        for start, end in bbox_edges():
            ax.plot(
                [projected_corners[start, 0], projected_corners[end, 0]],
                [projected_corners[start, 1], projected_corners[end, 1]],
                color=box["color"],
                linewidth=box["width"],
                solid_capstyle="round",
            )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=fig.get_facecolor(), pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB").resize(size, Image.Resampling.LANCZOS)


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
    if faces is not None and len(faces) > 0:
        projected_mesh = draw_projected_mesh_scene(
            points=points,
            colors=colors,
            faces=faces,
            boxes=boxes,
            size=size,
            elev=elev,
            azim=azim,
            max_points=max_points,
        )
        if projected_mesh is not None:
            return projected_mesh

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


def quantize_scene(image: Image.Image, colors: int) -> Image.Image:
    if colors <= 0:
        return image.convert("RGB")
    return image.convert("RGB").quantize(
        colors=colors,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    ).convert("RGB")


def choose_camera(
    points: np.ndarray,
    colors: np.ndarray,
    boxes: list[dict[str, Any]],
    faces: np.ndarray | None = None,
) -> dict[str, float]:
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
            faces=faces,
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


def build_style(text_preset: str, font_family: str) -> ReportStyle:
    spec = STYLE_PRESETS[text_preset].copy()
    spec["font_family"] = font_family
    return ReportStyle(**spec)


def resize_style(style: ReportStyle, text_size: int) -> ReportStyle:
    if text_size == style.text_size:
        return style
    scale = text_size / style.text_size
    return ReportStyle(
        text_size=text_size,
        min_text_size=style.min_text_size,
        line_step=max(text_size + 6, int(round(style.line_step * scale))),
        section_label_gap=max(text_size + 4, int(round(style.section_label_gap * scale))),
        section_bottom_gap=max(10, int(round(style.section_bottom_gap * scale))),
        legend_row_gap=max(text_size + 6, int(round(style.legend_row_gap * scale))),
        legend_base_height=max(text_size + 14, int(round(style.legend_base_height * scale))),
        left_x=style.left_x,
        text_w=style.text_w,
        right_x=style.right_x,
        right_y=style.right_y,
        right_margin=style.right_margin,
        bottom_margin=style.bottom_margin,
        legend_line_w=max(48, int(round(style.legend_line_w * scale))),
        legend_text_gap=max(12, int(round(style.legend_text_gap * scale))),
        legend_line_width=max(4, int(round(style.legend_line_width * scale))),
        legend_line_y=max(8, int(round(style.legend_line_y * scale))),
        font_family=style.font_family,
    )


def font(size: int, bold: bool = False, family: str = "times") -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = FONT_CANDIDATES[family][bold]
    if family != "times":
        candidates = candidates + FONT_CANDIDATES["times"][bold]
    candidates = candidates + FONT_CANDIDATES["arial"][bold]
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
    style: ReportStyle,
) -> int:
    label_font = font(style.text_size, bold=True, family=style.font_family)
    body_font = font(style.text_size, family=style.font_family)
    draw.text((x, y), label, fill=TEXT_DARK, font=label_font)
    y += style.section_label_gap
    lines = wrap_text(draw, body, body_font, width)
    for line in lines:
        draw.text((x, y), line, fill=TEXT_DARK, font=body_font)
        y += style.line_step
    return y + style.section_bottom_gap


def measure_section_height(
    draw: ImageDraw.ImageDraw,
    width: int,
    body: str,
    style: ReportStyle,
) -> int:
    body_font = font(style.text_size, family=style.font_family)
    lines = wrap_text(draw, body, body_font, width)
    return style.section_label_gap + len(lines) * style.line_step + style.section_bottom_gap


def measure_legend_height(style: ReportStyle, item_count: int = 2) -> int:
    if item_count <= 0:
        return 0
    return (item_count - 1) * style.legend_row_gap + style.legend_base_height


def draw_bbox_legend(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    items: list[tuple[str, str]],
    style: ReportStyle,
) -> int:
    legend_font = font(style.text_size, family=style.font_family)
    cursor_x = x
    for color, label in items:
        line_y = y + style.legend_line_y
        draw.line(
            (cursor_x, line_y, cursor_x + style.legend_line_w, line_y),
            fill=color,
            width=style.legend_line_width,
        )
        draw.text(
            (cursor_x + style.legend_line_w + style.legend_text_gap, y),
            label,
            fill=TEXT_DARK,
            font=legend_font,
        )
        label_w = draw.textlength(label, font=legend_font)
        cursor_x += int(style.legend_line_w + style.legend_text_gap + label_w + style.legend_line_w * 0.55)
    return y + measure_legend_height(style, len(items))


def text_block_height(
    draw: ImageDraw.ImageDraw,
    text_w: int,
    sections: list[tuple[str, str, str]],
    legend_items: list[tuple[str, str]],
    style: ReportStyle,
) -> int:
    return (
        sum(measure_section_height(draw, text_w, body, style) for _label, body, _accent in sections)
        + measure_legend_height(style, len(legend_items))
    )


def fit_text_style(
    draw: ImageDraw.ImageDraw,
    text_w: int,
    right_h: int,
    sections: list[tuple[str, str, str]],
    legend_items: list[tuple[str, str]],
    style: ReportStyle,
) -> ReportStyle:
    fitted = style
    while text_block_height(draw, text_w, sections, legend_items, fitted) > right_h and fitted.text_size > fitted.min_text_size:
        fitted = resize_style(style, max(fitted.min_text_size, fitted.text_size - 2))
    return fitted


def render_report(
    package: dict,
    scene_image: Image.Image,
    output_png: Path,
    size: tuple[int, int],
    camera: dict[str, float],
    renderer_name: str,
    scene_kind: str,
    style: ReportStyle,
    scene_colors: int = 0,
) -> None:
    width, height = size
    canvas = Image.new("RGB", size, WHITE)
    draw = ImageDraw.Draw(canvas)

    left_x = style.left_x
    text_w = style.text_w
    right_x = style.right_x
    right_y = style.right_y
    right_w = width - right_x - style.right_margin
    right_h = height - right_y - style.bottom_margin

    gt_text = ", ".join(package.get("gt_answer", []))
    pred_text = package.get("model_prediction", "")
    sections = [
        ("Instruction", package.get("input_text", ""), "#2f5f87"),
        ("GT", gt_text, GT_COLOR),
        ("PREDICTION", pred_text, PRED_COLOR),
    ]
    legend_items = collect_legend_items(package)
    style = fit_text_style(draw, text_w, right_h, sections, legend_items, style)
    total_text_h = text_block_height(draw, text_w, sections, legend_items, style)
    y = right_y + max((right_h - total_text_h) // 2, 0)
    for label, body, accent in sections:
        y = draw_section(draw, left_x, y, text_w, label, body, accent, style)
    draw_bbox_legend(draw, left_x, y - 4, legend_items, style)

    scene = trim_scene_whitespace(scene_image).resize(
        (right_w, right_h),
        Image.Resampling.LANCZOS,
    )
    scene = quantize_scene(scene, scene_colors)
    canvas.paste(scene, (right_x, right_y))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png, optimize=True, compress_level=9)


def main() -> int:
    args = parse_args()
    if not 0 < args.scene_scale <= 1:
        raise ValueError("--scene-scale must be greater than 0 and at most 1")
    if args.scene_colors < 0 or args.scene_colors > 256:
        raise ValueError("--scene-colors must be between 0 and 256")
    package_json = resolve_path(args.sample_package_json)
    package = load_json(package_json)
    output_png = (
        resolve_path(args.output_png)
        if args.output_png
        else package_json.with_name("sample_report.png")
    )

    points, colors, faces, scene_kind, scene_path = load_scene(package, args.scene_mode)
    boxes = collect_boxes(package)
    camera = choose_camera(points, colors, boxes, faces if scene_kind == "mesh" else None)
    style = build_style(args.text_preset, args.font_family)
    scene_panel_size = (
        args.width - style.right_x - style.right_margin,
        args.height - style.right_y - style.bottom_margin,
    )
    scene_render_size = (
        max(1, round(scene_panel_size[0] * args.scene_scale)),
        max(1, round(scene_panel_size[1] * args.scene_scale)),
    )
    renderer_name = "matplotlib"
    scene_image = None
    if args.renderer in ("auto", "open3d"):
        scene_image = try_render_open3d(
            points,
            colors,
            faces,
            boxes,
            scene_render_size,
            camera,
        )
        if scene_image is not None:
            renderer_name = "open3d"
    if scene_image is None:
        if args.renderer == "open3d":
            raise RuntimeError("Open3D rendering was requested but failed.")
        scene_image = draw_matplotlib_scene(
            points,
            colors,
            boxes,
            size=scene_render_size,
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
        style=style,
        scene_colors=args.scene_colors,
    )
    summary = {
        "report_png": str(output_png.relative_to(REPO_ROOT) if output_png.is_relative_to(REPO_ROOT) else output_png),
        "scene_source": str(scene_path.relative_to(REPO_ROOT) if scene_path.is_relative_to(REPO_ROOT) else scene_path),
        "scene_kind": scene_kind,
        "renderer": renderer_name,
        "text_preset": args.text_preset,
        "font_family": args.font_family,
        "scene_scale": args.scene_scale,
        "scene_render_size": list(scene_render_size),
        "scene_panel_size": list(scene_panel_size),
        "scene_colors": args.scene_colors,
        "camera": camera,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
