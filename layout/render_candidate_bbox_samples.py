from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Work around duplicate OpenMP runtime initialization on Windows/Conda.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import open_local_mesh as mesh_html
from point import visualize_input_tokens_mesh as mesh_viz


DEFAULT_SAMPLE_INDICES = [6002, 5987, 6008, 6816, 6890]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a shortlist of ScanRefer candidate samples as original-color meshes, "
            "interactive HTML viewers, and static bbox previews."
        )
    )
    parser.add_argument(
        "--annotation-root",
        default=None,
        help="Annotation root directory. If omitted, the script tries to infer it automatically.",
    )
    parser.add_argument(
        "--scene-root",
        default=str(REPO_ROOT / "sample_data" / "scannet"),
        help="Directory containing ScanNet scene folders.",
    )
    parser.add_argument(
        "--dataset-json",
        default="scanrefer_mask3d_val.json",
        help="Annotation json file under annotation root.",
    )
    parser.add_argument(
        "--gt-attr-file",
        default="scannet_val_attributes.pt",
        help="GT attribute file under annotation root.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "sample_bbox_candidates"),
        help="Output directory for generated artifacts.",
    )
    parser.add_argument(
        "--sample-indices",
        nargs="*",
        type=int,
        default=DEFAULT_SAMPLE_INDICES,
        help="Sample indices inside dataset json.",
    )
    parser.add_argument(
        "--preview-max-points",
        type=int,
        default=120000,
        help="Max points used for the static preview scatter plot.",
    )
    parser.add_argument(
        "--skip-html",
        action="store_true",
        help="Only write PLY/JSON/PNG artifacts, skip Plotly HTML export.",
    )
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Skip static bbox preview export.",
    )
    return parser.parse_args()


def load_dataset(annotation_root: Path, dataset_json: str) -> list[dict]:
    path = annotation_root / dataset_json
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_input_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path).resolve()


def build_metadata(
    sample: dict,
    sample_index: int,
    dataset_json: str,
    scene_root: Path,
) -> dict:
    return {
        "scene_id": sample["scene_id"],
        "sample_index": sample_index,
        "dataset_json": dataset_json,
        "scene_root": str(scene_root),
        "prompt": sample.get("prompt", ""),
        "ref_captions": sample.get("ref_captions", []),
        "obj_id": int(sample.get("obj_id", -1)),
    }


def render_html(
    mesh_output: Path,
    points: np.ndarray,
    colors: np.ndarray,
    faces: np.ndarray,
    metadata: dict,
    annotation_root: Path,
    gt_attr_file: str,
) -> Path:
    fig = mesh_html.build_figure(points, colors, faces)
    bbox_info = mesh_html.load_gt_bbox_from_metadata(metadata, annotation_root, gt_attr_file)
    if bbox_info is not None:
        loc, _, prompt_context_boxes = bbox_info
        mesh_html.add_bbox_trace(fig, loc, "rgb(0,255,0)")
        for context_box in prompt_context_boxes:
            mesh_html.add_bbox_trace(fig, context_box["loc"], "rgb(0,120,255)")

    output_html = mesh_output.with_suffix(".html")
    fig.write_html(str(output_html), include_plotlyjs="cdn", auto_open=False)
    return output_html


def draw_projected_bbox(
    ax,
    loc: np.ndarray,
    dims: tuple[int, int],
    color: str,
    linewidth: float,
) -> None:
    corners = mesh_html.bbox_corners_from_loc(loc)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for start, end in edges:
        ax.plot(
            [corners[start, dims[0]], corners[end, dims[0]]],
            [corners[start, dims[1]], corners[end, dims[1]]],
            color=color,
            linewidth=linewidth,
        )


def render_preview(
    points: np.ndarray,
    colors: np.ndarray,
    metadata: dict,
    annotation_root: Path,
    gt_attr_file: str,
    output_path: Path,
    max_points: int,
) -> Path | None:
    bbox_info = mesh_html.load_gt_bbox_from_metadata(metadata, annotation_root, gt_attr_file)
    if bbox_info is None:
        return None

    loc, _, prompt_context_boxes = bbox_info
    stride = max(1, len(points) // max_points)
    points_small = points[::stride]
    colors_small = colors[::stride].astype(np.float32) / 255.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=220)

    axes[0].scatter(points_small[:, 0], points_small[:, 1], c=colors_small, s=0.08)
    draw_projected_bbox(axes[0], loc, (0, 1), "#00ff00", 2.2)
    for context_box in prompt_context_boxes:
        draw_projected_bbox(axes[0], context_box["loc"], (0, 1), "#0078ff", 1.8)
    axes[0].set_title("Top View (X-Y)")
    axes[0].set_aspect("equal")
    axes[0].axis("off")

    axes[1].scatter(points_small[:, 0], points_small[:, 2], c=colors_small, s=0.08)
    draw_projected_bbox(axes[1], loc, (0, 2), "#00ff00", 2.2)
    for context_box in prompt_context_boxes:
        draw_projected_bbox(axes[1], context_box["loc"], (0, 2), "#0078ff", 1.8)
    axes[1].set_title("Front View (X-Z)")
    axes[1].set_aspect("equal")
    axes[1].axis("off")

    fig.suptitle(
        f"{metadata['scene_id']} | sample={metadata['sample_index']} | green=GT | blue=context\n"
        f"{metadata.get('prompt', '')}"
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    launch_cwd = Path.cwd()

    annotation_root = (
        resolve_input_path(args.annotation_root, launch_cwd)
        if args.annotation_root
        else mesh_html.find_annotation_root(None)
    )
    if annotation_root is None or not annotation_root.exists():
        raise FileNotFoundError(
            "Could not locate annotation root. Pass --annotation-root explicitly."
        )

    scene_root = resolve_input_path(args.scene_root, launch_cwd)
    output_dir = resolve_input_path(args.output_dir, launch_cwd)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(annotation_root, args.dataset_json)
    manifest: list[dict] = []

    for sample_index in args.sample_indices:
        sample = dataset[sample_index]
        metadata = build_metadata(sample, sample_index, args.dataset_json, scene_root)
        scene_id = metadata["scene_id"]

        points, colors, faces = mesh_viz.load_scene_mesh(scene_root / scene_id, scene_id)
        stem = f"{scene_id}_sample{sample_index:05d}_input_tokens_mesh_uniform_red_original_000objs"
        mesh_output = output_dir / f"{stem}.ply"
        meta_output = output_dir / f"{stem}.json"
        labels_output = output_dir / f"{stem}_vertex_labels.txt"

        mesh_viz.write_colored_mesh(points, colors, faces, mesh_output)
        mesh_viz.write_vertex_labels(np.zeros(len(points), dtype=np.int32), labels_output)
        meta_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        preview_output = None
        if not args.skip_preview:
            preview_output = render_preview(
                points=points,
                colors=colors,
                metadata=metadata,
                annotation_root=annotation_root,
                gt_attr_file=args.gt_attr_file,
                output_path=output_dir / f"{stem}_bbox_preview.png",
                max_points=args.preview_max_points,
            )

        html_output = None
        if not args.skip_html:
            html_output = render_html(
                mesh_output=mesh_output,
                points=points,
                colors=colors,
                faces=faces,
                metadata=metadata,
                annotation_root=annotation_root,
                gt_attr_file=args.gt_attr_file,
            )

        record = {
            "sample_index": sample_index,
            "scene_id": scene_id,
            "prompt": metadata.get("prompt", ""),
            "mesh_output": str(mesh_output),
            "metadata_output": str(meta_output),
            "vertex_labels_output": str(labels_output),
            "preview_output": str(preview_output) if preview_output else None,
            "html_output": str(html_output) if html_output else None,
        }
        manifest.append(record)
        print(json.dumps(record, ensure_ascii=False))

    manifest_path = output_dir / "candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
