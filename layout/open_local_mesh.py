from __future__ import annotations

import argparse
import json
import os
import re
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch


OBJ_PATTERN = re.compile(r"<OBJ\d{3}>")
GT_BBOX_COLOR = "rgb(34,139,34)"
CONTEXT_BBOX_COLOR = "rgb(0,120,255)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a local ASCII mesh PLY as an interactive Plotly Mesh3d viewer."
    )
    parser.add_argument(
        "input_ply",
        nargs="?",
        default="outputs/visualizations_mesh/scene0011_00_sample00000_input_tokens_mesh_uniform_red_100objs.ply",
        help="Path to an ASCII mesh PLY file.",
    )
    parser.add_argument(
        "--output-html",
        default=None,
        help="Output html path. Defaults next to the input PLY.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Only generate the html file, do not open a browser.",
    )
    parser.add_argument(
        "--annotation-root",
        default=None,
        help="Optional annotation root. If omitted, the script tries to infer it automatically.",
    )
    parser.add_argument(
        "--gt-attr-file",
        default="scannet_val_attributes.pt",
        help="GT attribute file under annotation root.",
    )
    parser.add_argument(
        "--no-bbox",
        action="store_true",
        help="Disable automatic GT bbox overlay.",
    )
    return parser.parse_args()


def load_ascii_mesh_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertex_count = None
    face_count = None
    with path.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Invalid ply file: {path}")
            if line.startswith("element vertex "):
                vertex_count = int(line.strip().split()[-1])
            elif line.startswith("element face "):
                face_count = int(line.strip().split()[-1])
            elif line.strip() == "end_header":
                break

        if vertex_count is None or face_count is None:
            raise ValueError(f"Missing mesh counts in ply file: {path}")

        try:
            vertices = np.loadtxt(handle, max_rows=vertex_count)
        except ValueError as exc:
            raise ValueError(
                f"Failed to read vertex section from {path}. The PLY file is likely incomplete or corrupted. "
                f"Please re-generate or re-download it. Original error: {exc}"
            ) from exc
        try:
            faces = np.loadtxt(handle, max_rows=face_count, dtype=np.int64)
        except ValueError as exc:
            raise ValueError(
                f"Failed to read face section from {path}. The PLY file is likely incomplete or corrupted. "
                f"Please re-generate or re-download it. Original error: {exc}"
            ) from exc

    points = vertices[:, :3].astype(np.float32)
    colors = vertices[:, 3:6].astype(np.uint8)
    faces = faces[:, 1:4].astype(np.int32)
    return points, colors, faces


def build_figure(points: np.ndarray, colors: np.ndarray, faces: np.ndarray) -> go.Figure:
    vertexcolor = [f"rgb({r},{g},{b})" for r, g, b in colors.tolist()]
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                vertexcolor=vertexcolor,
                flatshading=False,
                hoverinfo="skip",
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="Interactive Mesh Viewer",
        showlegend=False,
    )
    return fig


def load_metadata(input_ply: Path) -> dict | None:
    json_path = input_ply.with_suffix(".json")
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def find_annotation_root(explicit_root: str | None) -> Path | None:
    if explicit_root:
        path = Path(explicit_root)
        return path if path.exists() else None

    candidates = []
    for env_name in ("VISION3D_ANNO_ROOT", "FAST3D_ANNO_ROOT"):
        env_var = os.environ.get(env_name)
        if env_var:
            candidates.append(Path(env_var))

    search_bases = [Path.cwd(), Path(__file__).resolve()]
    for base in search_bases:
        for parent in [base] + list(base.parents):
            candidates.append(parent / "annotations")
            candidates.append(parent / "datasets" / "annotations")
            candidates.append(parent / "PointLLM_Reduction" / "datasets" / "annotations")

    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def load_gt_bbox_from_metadata(
    metadata: dict,
    annotation_root: Path,
    gt_attr_file: str,
) -> tuple[np.ndarray, int, list[dict]] | None:
    scene_id = metadata.get("scene_id")
    sample_index = metadata.get("sample_index")
    dataset_json = metadata.get("dataset_json")
    if scene_id is None or sample_index is None or dataset_json is None:
        return None

    dataset_path = annotation_root / dataset_json
    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    sample = dataset[int(sample_index)]
    gt_obj_id = int(sample.get("obj_id", -1))
    if gt_obj_id < 0:
        return None

    gt_attrs = torch.load(
        str(annotation_root / gt_attr_file), map_location="cpu", weights_only=False
    )
    all_locs = gt_attrs[scene_id]["locs"].cpu().numpy()
    loc = all_locs[gt_obj_id]

    target_label = None
    prompt_context_boxes: list[dict] = []
    repo_root = Path(__file__).resolve().parents[1]
    agg_candidates = []
    if metadata.get("scene_root"):
        agg_candidates.append(
            Path(str(metadata["scene_root"])) / scene_id / f"{scene_id}.aggregation.json"
        )
    agg_candidates.append(
        repo_root / "sample_data" / "scannet" / scene_id / f"{scene_id}.aggregation.json"
    )
    agg_path = next((path for path in agg_candidates if path.exists()), None)
    if agg_path is not None:
        aggregation = json.loads(agg_path.read_text(encoding="utf-8"))
        groups = [g for g in aggregation.get("segGroups", []) if int(g.get("objectId", -1)) < len(all_locs)]
        for group in aggregation.get("segGroups", []):
            if int(group.get("objectId", -1)) == gt_obj_id:
                target_label = group.get("label")
                break

        prompt_lower = sample.get("prompt", "").lower()
        label_to_object_ids: dict[str, list[int]] = {}
        for group in groups:
            label_to_object_ids.setdefault(group.get("label", "").lower(), []).append(
                int(group.get("objectId", -1))
            )

        chosen_labels: list[str] = []
        target_center = all_locs[gt_obj_id][:3]
        for label in sorted(label_to_object_ids.keys(), key=len, reverse=True):
            if not label or label == (target_label or "").lower():
                continue
            if label not in prompt_lower:
                continue
            if any(label in picked for picked in chosen_labels):
                continue
            object_ids = label_to_object_ids[label]
            nearest_obj = min(
                object_ids,
                key=lambda oid: float(np.linalg.norm(all_locs[oid][:3] - target_center)),
            )
            prompt_context_boxes.append(
                {
                    "obj_id": int(nearest_obj),
                    "label": label,
                    "loc": all_locs[nearest_obj],
                }
            )
            chosen_labels.append(label)

    return loc, gt_obj_id, prompt_context_boxes


def bbox_corners_from_loc(loc: np.ndarray) -> np.ndarray:
    center = loc[:3]
    size = loc[3:6]
    x, y, z = size / 2.0
    corners = np.array(
        [
            [center[0] - x, center[1] - y, center[2] - z],
            [center[0] + x, center[1] - y, center[2] - z],
            [center[0] + x, center[1] + y, center[2] - z],
            [center[0] - x, center[1] + y, center[2] - z],
            [center[0] - x, center[1] - y, center[2] + z],
            [center[0] + x, center[1] - y, center[2] + z],
            [center[0] + x, center[1] + y, center[2] + z],
            [center[0] - x, center[1] + y, center[2] + z],
        ],
        dtype=np.float32,
    )
    return corners


def add_bbox_trace(fig: go.Figure, loc: np.ndarray, color: str) -> None:
    corners = bbox_corners_from_loc(loc)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    xs, ys, zs = [], [], []
    for start, end in edges:
        xs.extend([corners[start, 0], corners[end, 0], None])
        ys.extend([corners[start, 1], corners[end, 1], None])
        zs.extend([corners[start, 2], corners[end, 2], None])
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=color, width=8),
            showlegend=False,
            hoverinfo="skip",
        )
    )


def main() -> int:
    args = parse_args()
    input_ply = Path(args.input_ply).resolve()
    output_html = (
        Path(args.output_html).resolve()
        if args.output_html is not None
        else input_ply.with_suffix(".html")
    )

    points, colors, faces = load_ascii_mesh_ply(input_ply)
    fig = build_figure(points, colors, faces)

    if not args.no_bbox:
        metadata = load_metadata(input_ply)
        annotation_root = find_annotation_root(args.annotation_root)
        if metadata is not None and annotation_root is not None:
            bbox_info = load_gt_bbox_from_metadata(metadata, annotation_root, args.gt_attr_file)
            if bbox_info is not None:
                loc, gt_obj_id, prompt_context_boxes = bbox_info
                add_bbox_trace(fig, loc, GT_BBOX_COLOR)
                for context_box in prompt_context_boxes:
                    add_bbox_trace(fig, context_box["loc"], CONTEXT_BBOX_COLOR)

    fig.write_html(str(output_html), include_plotlyjs="cdn", auto_open=False)

    print(f"Input PLY:   {input_ply}")
    print(f"Output HTML: {output_html}")
    print(f"Vertices:    {len(points)}")
    print(f"Faces:       {len(faces)}")

    if not args.no_open:
        webbrowser.open(output_html.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
