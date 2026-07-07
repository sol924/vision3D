from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


OBJ_PATTERN = re.compile(r"<OBJ(\d{3})>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize object-level regions for one annotation sample on a ScanNet scene."
    )
    parser.add_argument("--annotation-root", required=True, help="Annotation root directory.")
    parser.add_argument(
        "--scene-root",
        required=True,
        help="Directory containing downloaded ScanNet sample scenes.",
    )
    parser.add_argument(
        "--dataset-json",
        default="scanrefer_mask3d_val.json",
        help="Annotation json file under annotation root.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index inside dataset json.",
    )
    parser.add_argument(
        "--pred-attr-file",
        default="scannet_mask3d_val_attributes.pt",
        help="Predicted object attribute file under annotation root.",
    )
    parser.add_argument(
        "--gt-attr-file",
        default="scannet_val_attributes.pt",
        help="GT object attribute file under annotation root.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/visualizations",
        help="Output directory for ply/png/json artifacts.",
    )
    parser.add_argument(
        "--highlight-object-ids",
        nargs="*",
        type=int,
        default=None,
        help="Optional token object ids to highlight. Defaults to ids parsed from ref_captions.",
    )
    parser.add_argument(
        "--mode",
        choices=("token", "gt", "both"),
        default="token",
        help="What to color: token objects, GT object, or both. Default is token.",
    )
    parser.add_argument(
        "--bbox-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to bbox sizes when selecting points.",
    )
    parser.add_argument(
        "--max-preview-points",
        type=int,
        default=120000,
        help="Max number of points used in the PNG preview.",
    )
    return parser.parse_args()


def load_sample(annotation_root: Path, dataset_json: str, sample_index: int) -> dict:
    path = annotation_root / dataset_json
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data[sample_index]


def parse_ref_object_ids(sample: dict) -> list[int]:
    ref_ids: list[int] = []
    for text in sample.get("ref_captions", []):
        ref_ids.extend(int(match.group(1)) for match in OBJ_PATTERN.finditer(text))
    deduped: list[int] = []
    for object_id in ref_ids:
        if object_id not in deduped:
            deduped.append(object_id)
    return deduped


def read_axis_align_matrix(meta_path: Path) -> np.ndarray:
    with meta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if "axisAlignment" not in line:
                continue
            values = [float(item) for item in line.strip().split("=")[1].split()]
            return np.asarray(values, dtype=np.float32).reshape(4, 4)
    return np.eye(4, dtype=np.float32)


def _parse_ply_header(ply_path: Path) -> tuple[int, int]:
    vertex_count = None
    header_end = None
    with ply_path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Invalid PLY file: {ply_path}")
            if line.startswith(b"element vertex "):
                vertex_count = int(line.decode("ascii").strip().split()[-1])
            if line.strip() == b"end_header":
                header_end = handle.tell()
                break
    if vertex_count is None or header_end is None:
        raise ValueError(f"Failed to parse PLY header: {ply_path}")
    return vertex_count, header_end


def load_scene_points(scene_dir: Path, scene_id: str) -> tuple[np.ndarray, np.ndarray]:
    ply_path = scene_dir / f"{scene_id}_vh_clean_2.ply"
    meta_path = scene_dir / f"{scene_id}.txt"
    vertex_count, header_end = _parse_ply_header(ply_path)
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
    with ply_path.open("rb") as handle:
        handle.seek(header_end)
        vertex = np.fromfile(handle, dtype=vertex_dtype, count=vertex_count)
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)

    axis_align = read_axis_align_matrix(meta_path).astype(np.float64)
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    z = points[:, 2].astype(np.float64)
    points = np.column_stack(
        [
            x * axis_align[0, 0] + y * axis_align[0, 1] + z * axis_align[0, 2] + axis_align[0, 3],
            x * axis_align[1, 0] + y * axis_align[1, 1] + z * axis_align[1, 2] + axis_align[1, 3],
            x * axis_align[2, 0] + y * axis_align[2, 1] + z * axis_align[2, 2] + axis_align[2, 3],
        ]
    ).astype(np.float32)
    return points, colors


def load_attributes(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def load_gt_instance_mask(scene_dir: Path, scene_id: str, gt_object_id: int) -> np.ndarray | None:
    aggregation_path = scene_dir / f"{scene_id}.aggregation.json"
    segs_path = scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    if not aggregation_path.exists() or not segs_path.exists() or gt_object_id < 0:
        return None

    aggregation = json.loads(aggregation_path.read_text(encoding="utf-8"))
    segs = json.loads(segs_path.read_text(encoding="utf-8"))
    segment_indices = np.asarray(segs["segIndices"], dtype=np.int64)

    target_segments = None
    for group in aggregation.get("segGroups", []):
        if int(group.get("objectId", -1)) == gt_object_id:
            target_segments = set(int(item) for item in group.get("segments", []))
            break
    if target_segments is None:
        return None
    return np.isin(segment_indices, list(target_segments))


def bbox_to_mask(points: np.ndarray, loc: np.ndarray, scale: float) -> np.ndarray:
    center = loc[:3]
    size = loc[3:] * scale
    xyz_min = center - size / 2
    xyz_max = center + size / 2
    return np.all((points >= xyz_min) & (points <= xyz_max), axis=1)


def write_ply(points: np.ndarray, colors: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("element face 0\n")
        handle.write("property list uchar uint vertex_indices\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def save_preview(
    points: np.ndarray,
    colors: np.ndarray,
    output_path: Path,
    max_points: int,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
        colors = colors[indices]

    colors_float = colors.astype(np.float32) / 255.0
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180)
    axes[0].scatter(points[:, 0], points[:, 1], c=colors_float, s=0.2)
    axes[0].set_title("Top View (X-Y)")
    axes[0].set_aspect("equal")
    axes[0].axis("off")

    axes[1].scatter(points[:, 0], points[:, 2], c=colors_float, s=0.2)
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

    points, colors = load_scene_points(scene_dir, scene_id)
    pred_attrs = load_attributes(annotation_root / args.pred_attr_file)
    gt_attrs = load_attributes(annotation_root / args.gt_attr_file)

    pred_locs = pred_attrs[scene_id]["locs"].cpu().numpy()
    gt_locs = gt_attrs[scene_id]["locs"].cpu().numpy()

    gt_object_id = int(sample.get("obj_id", -1))
    token_object_ids = (
        args.highlight_object_ids
        if args.highlight_object_ids is not None and len(args.highlight_object_ids) > 0
        else parse_ref_object_ids(sample)
    )

    raw_output = output_dir / f"{scene_id}_raw_aligned.ply"
    overlay_name = f"{scene_id}_sample{args.sample_index:05d}_{args.mode}"
    if args.mode in ("token", "both"):
        overlay_name += (
            f"_token{'-'.join(f'{idx:03d}' for idx in token_object_ids) if token_object_ids else 'none'}"
        )
    if args.mode in ("gt", "both"):
        overlay_name += f"_gt{gt_object_id:03d}"
    overlay_output = output_dir / f"{overlay_name}.ply"
    preview_output = overlay_output.with_suffix(".png")
    meta_output = overlay_output.with_suffix(".json")

    write_ply(points, colors, raw_output)

    overlay_colors = np.clip(colors.astype(np.float32) * 0.65 + 20, 0, 255).astype(np.uint8)
    highlight_masks: dict[str, int] = {}

    palette = [
        np.array([255, 80, 80], dtype=np.uint8),
        np.array([255, 180, 60], dtype=np.uint8),
        np.array([255, 80, 220], dtype=np.uint8),
        np.array([255, 220, 90], dtype=np.uint8),
    ]

    if args.mode in ("token", "both"):
        for color_index, object_id in enumerate(token_object_ids):
            if object_id < 0 or object_id >= len(pred_locs):
                continue
            mask = bbox_to_mask(points, pred_locs[object_id], args.bbox_scale)
            overlay_colors[mask] = palette[color_index % len(palette)]
            highlight_masks[f"token_obj_{object_id:03d}"] = int(mask.sum())

    if args.mode in ("gt", "both") and 0 <= gt_object_id < len(gt_locs):
        gt_mask = load_gt_instance_mask(scene_dir, scene_id, gt_object_id)
        if gt_mask is None:
            gt_mask = bbox_to_mask(points, gt_locs[gt_object_id], args.bbox_scale)
        overlay_colors[gt_mask] = np.array([80, 255, 120], dtype=np.uint8)
        highlight_masks[f"gt_obj_{gt_object_id:03d}"] = int(gt_mask.sum())

    write_ply(points, overlay_colors, overlay_output)
    save_preview(
        points,
        overlay_colors,
        preview_output,
        args.max_preview_points,
        title=f"{scene_id} | sample={args.sample_index} | mode={args.mode} | token={token_object_ids} | gt={gt_object_id}",
    )

    metadata = {
        "scene_id": scene_id,
        "sample_index": args.sample_index,
        "mode": args.mode,
        "dataset_json": args.dataset_json,
        "scene_root": str(scene_root),
        "gt_object_id": gt_object_id,
        "token_object_ids_from_ref": token_object_ids,
        "prompt": sample.get("prompt", ""),
        "ref_captions": sample.get("ref_captions", []),
        "raw_output": str(raw_output),
        "overlay_output": str(overlay_output),
        "preview_output": str(preview_output),
        "point_count": int(len(points)),
        "highlight_point_counts": highlight_masks,
    }
    meta_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
