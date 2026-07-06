from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "High-quality mesh visualization for the visual tokens actually fed into the model. "
            "This follows the ScanNet official mesh-coloring style: full mesh vertices are recolored "
            "while faces are preserved."
        )
    )
    parser.add_argument("--annotation-root", required=True, help="Annotation root directory.")
    parser.add_argument(
        "--scene-root",
        required=True,
        help="Directory containing official ScanNet scene files.",
    )
    parser.add_argument(
        "--dataset-json",
        default="scanrefer_mask3d_val.json",
        help="Annotation json under annotation root.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index in dataset json.",
    )
    parser.add_argument(
        "--pred-attr-file",
        default="scannet_mask3d_val_attributes.pt",
        help="Predicted object attribute file under annotation root.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=100,
        help="Maximum number of object slots considered by the model.",
    )
    parser.add_argument(
        "--bbox-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to bbox sizes before selecting mesh vertices.",
    )
    parser.add_argument(
        "--color-mode",
        choices=("uniform_red", "palette"),
        default="palette",
        help="How to color the input-token objects.",
    )
    parser.add_argument(
        "--background-mode",
        choices=("dimmed", "original"),
        default="dimmed",
        help="Use dimmed scene colors or the original mesh colors for non-highlighted vertices.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/visualizations_mesh",
        help="Output directory for colored mesh and previews.",
    )
    return parser.parse_args()


def load_sample(annotation_root: Path, dataset_json: str, sample_index: int) -> dict:
    path = annotation_root / dataset_json
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data[sample_index]


def read_axis_align_matrix(meta_path: Path) -> np.ndarray:
    with meta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if "axisAlignment" not in line:
                continue
            values = [float(item) for item in line.strip().split("=")[1].split()]
            return np.asarray(values, dtype=np.float32).reshape(4, 4)
    return np.eye(4, dtype=np.float32)


def _parse_ply_header(path: Path) -> tuple[int, int, int]:
    vertex_count = None
    face_count = None
    header_end = None
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Invalid PLY file: {path}")
            if line.startswith(b"element vertex "):
                vertex_count = int(line.decode("ascii").strip().split()[-1])
            elif line.startswith(b"element face "):
                face_count = int(line.decode("ascii").strip().split()[-1])
            elif line.strip() == b"end_header":
                header_end = handle.tell()
                break
    if vertex_count is None or face_count is None or header_end is None:
        raise ValueError(f"Failed to parse PLY header: {path}")
    return vertex_count, face_count, header_end


def load_scene_mesh(scene_dir: Path, scene_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ply_path = scene_dir / f"{scene_id}_vh_clean_2.ply"
    meta_path = scene_dir / f"{scene_id}.txt"
    vertex_count, face_count, header_end = _parse_ply_header(ply_path)
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("alpha", "u1"),
        ]
    )
    face_dtype = np.dtype(
        [
            ("count", "u1"),
            ("v0", "<i4"),
            ("v1", "<i4"),
            ("v2", "<i4"),
        ]
    )

    with ply_path.open("rb") as handle:
        handle.seek(header_end)
        vertices = np.fromfile(handle, dtype=vertex_dtype, count=vertex_count)
        faces_raw = np.fromfile(handle, dtype=face_dtype, count=face_count)

    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(
        np.float32
    )
    colors = np.stack(
        [vertices["red"], vertices["green"], vertices["blue"]], axis=1
    ).astype(np.uint8)
    faces = np.stack([faces_raw["v0"], faces_raw["v1"], faces_raw["v2"]], axis=1).astype(
        np.int32
    )

    axis_align = read_axis_align_matrix(meta_path)
    points_h = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1
    )
    points = (points_h @ axis_align.T)[:, :3]
    return points, colors, faces


def load_pred_attributes(annotation_root: Path, pred_attr_file: str) -> dict:
    return torch.load(annotation_root / pred_attr_file, map_location="cpu")


def get_input_token_object_ids(pred_attrs: dict, scene_id: str, max_objects: int) -> list[int]:
    locs = pred_attrs[scene_id]["locs"]
    real_object_count = min(len(locs), max_objects)
    return list(range(real_object_count))


def bbox_to_mask(points: np.ndarray, loc: np.ndarray, scale: float) -> np.ndarray:
    center = loc[:3]
    size = loc[3:] * scale
    xyz_min = center - size / 2
    xyz_max = center + size / 2
    return np.all((points >= xyz_min) & (points <= xyz_max), axis=1)


def get_evenly_distributed_colors(count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    hsv = np.array([(i / max(count, 1), 1.0, 1.0) for i in range(count)], dtype=np.float32)
    colors = []
    for h, s, v in hsv:
        i = int(h * 6.0)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i = i % 6
        if i == 0:
            rgb = (v, t, p)
        elif i == 1:
            rgb = (q, v, p)
        elif i == 2:
            rgb = (p, v, t)
        elif i == 3:
            rgb = (p, q, v)
        elif i == 4:
            rgb = (t, p, v)
        else:
            rgb = (v, p, q)
        colors.append((np.array(rgb) * 255).astype(np.uint8))
    return np.stack(colors, axis=0)


def color_vertices_for_input_tokens(
    points: np.ndarray,
    base_colors: np.ndarray,
    locs: np.ndarray,
    token_object_ids: list[int],
    bbox_scale: float,
    color_mode: str,
    background_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int] | int]]:
    if background_mode == "original":
        output_colors = base_colors.copy()
    else:
        output_colors = np.clip(base_colors.astype(np.float32) * 0.45 + 20, 0, 255).astype(np.uint8)
    vertex_labels = np.zeros(len(points), dtype=np.int32)

    palette = (
        np.tile(np.array([[255, 80, 80]], dtype=np.uint8), (len(token_object_ids), 1))
        if color_mode == "uniform_red"
        else get_evenly_distributed_colors(len(token_object_ids))
    )

    volumes = []
    for object_id in token_object_ids:
        size = locs[object_id][3:]
        volumes.append(float(np.prod(size)))
    draw_order = [obj for _, obj in sorted(zip(volumes, token_object_ids), reverse=True)]

    object_point_counts: dict[str, int] = {}
    object_colors: dict[str, list[int]] = {}
    for object_id in draw_order:
        mask = bbox_to_mask(points, locs[object_id], bbox_scale)
        if not mask.any():
            object_point_counts[f"obj_{object_id:03d}"] = 0
            continue
        color = palette[token_object_ids.index(object_id)]
        output_colors[mask] = color
        vertex_labels[mask] = object_id + 1
        object_point_counts[f"obj_{object_id:03d}"] = int(mask.sum())
        object_colors[f"obj_{object_id:03d}"] = color.astype(int).tolist()

    metadata = {
        "input_token_object_count": len(token_object_ids),
        "object_point_counts": object_point_counts,
        "object_colors": object_colors,
    }
    return output_colors, vertex_labels, metadata


def write_colored_mesh(
    points: np.ndarray,
    colors: np.ndarray,
    faces: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False, dir=str(output_path.parent), suffix=".tmp"
    ) as handle:
        temp_path = Path(handle.name)
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")
    os.replace(temp_path, output_path)


def write_vertex_labels(vertex_labels: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False, dir=str(output_path.parent), suffix=".tmp"
    ) as handle:
        temp_path = Path(handle.name)
        np.savetxt(handle, vertex_labels, fmt="%d")
    os.replace(temp_path, output_path)


def save_preview(points: np.ndarray, colors: np.ndarray, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors_float = colors.astype(np.float32) / 255.0
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=220)
    axes[0].scatter(points[:, 0], points[:, 1], c=colors_float, s=0.08)
    axes[0].set_title("Top View (X-Y)")
    axes[0].set_aspect("equal")
    axes[0].axis("off")

    axes[1].scatter(points[:, 0], points[:, 2], c=colors_float, s=0.08)
    axes[1].set_title("Front View (X-Z)")
    axes[1].set_aspect("equal")
    axes[1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    annotation_root = Path(args.annotation_root)
    scene_root = Path(args.scene_root)
    output_dir = Path(args.output_dir)

    sample = load_sample(annotation_root, args.dataset_json, args.sample_index)
    scene_id = sample["scene_id"]
    scene_dir = scene_root / scene_id

    points, colors, faces = load_scene_mesh(scene_dir, scene_id)
    pred_attrs = load_pred_attributes(annotation_root, args.pred_attr_file)
    pred_locs = pred_attrs[scene_id]["locs"].cpu().numpy()

    token_object_ids = get_input_token_object_ids(pred_attrs, scene_id, args.max_objects)
    mesh_colors, vertex_labels, token_meta = color_vertices_for_input_tokens(
        points,
        colors,
        pred_locs,
        token_object_ids,
        args.bbox_scale,
        args.color_mode,
        args.background_mode,
    )

    output_stem = (
        f"{scene_id}_sample{args.sample_index:05d}_input_tokens_mesh_"
        f"{args.color_mode}_{args.background_mode}_{len(token_object_ids):03d}objs"
    )
    mesh_output = output_dir / f"{output_stem}.ply"
    preview_output = output_dir / f"{output_stem}.png"
    labels_output = output_dir / f"{output_stem}_vertex_labels.txt"
    meta_output = output_dir / f"{output_stem}.json"

    write_colored_mesh(points, mesh_colors, faces, mesh_output)
    write_vertex_labels(vertex_labels, labels_output)
    save_preview(
        points,
        mesh_colors,
        preview_output,
        title=(
            f"{scene_id} | sample={args.sample_index} | input token objects={len(token_object_ids)} | "
            f"visual tokens={len(token_object_ids) * 3}"
        ),
    )

    metadata = {
        "scene_id": scene_id,
        "sample_index": args.sample_index,
        "dataset_json": args.dataset_json,
        "scene_root": str(scene_root),
        "prompt": sample.get("prompt", ""),
        "input_token_object_ids": token_object_ids,
        "visual_token_groups": len(token_object_ids),
        "visual_token_count_current_setup": len(token_object_ids) * 3,
        "pred_attribute_file": args.pred_attr_file,
        "bbox_scale": args.bbox_scale,
        "color_mode": args.color_mode,
        "background_mode": args.background_mode,
        "mesh_output": str(mesh_output),
        "preview_output": str(preview_output),
        "vertex_labels_output": str(labels_output),
        **token_meta,
    }
    meta_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
