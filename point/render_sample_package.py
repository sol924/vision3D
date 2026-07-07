from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
OBJ_PATTERN = re.compile(r"<OBJ(\d{3})>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render one dataset sample into a complete visualization package: point cloud, "
            "mesh, HTML viewers, and metadata for input text, GT, and model prediction."
        )
    )
    parser.add_argument("--annotation-root", required=True, help="Annotation root directory.")
    parser.add_argument("--scene-root", required=True, help="Directory containing scene folders.")
    parser.add_argument("--predictions-json", required=True, help="Model prediction JSON file.")
    parser.add_argument(
        "--dataset-json",
        default="scanrefer_mask3d_val.json",
        help="Annotation JSON under annotation root.",
    )
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index to render.")
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
        default="outputs/sample_packages",
        help="Directory where the sample package folder is written.",
    )
    parser.add_argument(
        "--max-mesh-objects",
        type=int,
        default=30,
        help="Minimum number of predicted object slots shown in the mesh-token visualization.",
    )
    parser.add_argument("--bbox-scale", type=float, default=1.0, help="BBox scale for overlays.")
    parser.add_argument("--no-html", action="store_true", help="Skip interactive HTML viewers.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_object_ids(text: str) -> list[int]:
    ids: list[int] = []
    for match in OBJ_PATTERN.finditer(text or ""):
        object_id = int(match.group(1))
        if object_id not in ids:
            ids.append(object_id)
    return ids


def normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def select_prediction(predictions: list[dict], sample: dict, sample_index: int) -> dict:
    candidates: list[dict] = []
    if 0 <= sample_index < len(predictions):
        candidates.append(predictions[sample_index])
    candidates.extend(
        pred
        for pred in predictions
        if int(pred.get("qid", -1)) == sample_index and pred not in candidates
    )
    candidates.extend(
        pred
        for pred in predictions
        if pred.get("scene_id") == sample.get("scene_id")
        and normalize_text(pred.get("prompt", "")) == normalize_text(sample.get("prompt", ""))
        and pred not in candidates
    )

    for pred in candidates:
        if pred.get("scene_id") == sample.get("scene_id"):
            return pred
    raise ValueError(
        f"Could not find a prediction for sample_index={sample_index}, scene_id={sample.get('scene_id')}"
    )


def load_locs(annotation_root: Path, attr_file: str, scene_id: str) -> Any:
    attrs = torch.load(annotation_root / attr_file, map_location="cpu")
    return attrs[scene_id]["locs"].cpu().numpy()


def loc_to_list(locs: Any, object_id: int) -> list[float] | None:
    if object_id < 0 or object_id >= len(locs):
        return None
    return [float(item) for item in locs[object_id].tolist()]


def run_json_command(cmd: list[str]) -> dict:
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    start = result.stdout.find("{")
    if start < 0:
        raise RuntimeError(f"Command did not emit JSON: {' '.join(cmd)}\n{result.stdout}")
    return json.loads(result.stdout[start:])


def run_command(cmd: list[str]) -> str:
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def make_relative(path: str | Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    args = parse_args()
    annotation_root = Path(args.annotation_root).resolve()
    scene_root = Path(args.scene_root).resolve()
    predictions_json = Path(args.predictions_json).resolve()
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()

    dataset = load_json(annotation_root / args.dataset_json)
    predictions = load_json(predictions_json)
    sample = dataset[args.sample_index]
    prediction = select_prediction(predictions, sample, args.sample_index)
    scene_id = sample["scene_id"]

    gt_object_id = int(sample.get("obj_id", prediction.get("gt_id", -1)))
    gt_answer = sample.get("ref_captions", prediction.get("ref_captions", []))
    model_prediction = prediction.get("pred", "")
    model_pred_object_ids = extract_object_ids(model_prediction)
    if not model_pred_object_ids and prediction.get("pred_id") is not None:
        model_pred_object_ids = [int(prediction["pred_id"])]

    package_dir = output_root / f"{scene_id}_sample{args.sample_index:05d}"
    point_dir = package_dir / "point"
    mesh_dir = package_dir / "mesh"
    point_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    point_cmd = [
        sys.executable,
        str(REPO_ROOT / "point" / "visualize_sample_objects.py"),
        "--annotation-root",
        str(annotation_root),
        "--scene-root",
        str(scene_root),
        "--dataset-json",
        args.dataset_json,
        "--sample-index",
        str(args.sample_index),
        "--pred-attr-file",
        args.pred_attr_file,
        "--gt-attr-file",
        args.gt_attr_file,
        "--output-dir",
        str(point_dir),
        "--mode",
        "both",
        "--bbox-scale",
        str(args.bbox_scale),
    ]
    if model_pred_object_ids:
        point_cmd.extend(["--highlight-object-ids", *[str(item) for item in model_pred_object_ids]])
    point_meta = run_json_command(point_cmd)

    max_mesh_objects = max(args.max_mesh_objects, *(item + 1 for item in model_pred_object_ids), 1)
    mesh_meta = run_json_command(
        [
            sys.executable,
            str(REPO_ROOT / "point" / "visualize_input_tokens_mesh.py"),
            "--annotation-root",
            str(annotation_root),
            "--scene-root",
            str(scene_root),
            "--dataset-json",
            args.dataset_json,
            "--sample-index",
            str(args.sample_index),
            "--pred-attr-file",
            args.pred_attr_file,
            "--max-objects",
            str(max_mesh_objects),
            "--bbox-scale",
            str(args.bbox_scale),
            "--color-mode",
            "none",
            "--background-mode",
            "original",
            "--output-dir",
            str(mesh_dir),
        ]
    )

    html_outputs: dict[str, str] = {}
    if not args.no_html:
        point_html = package_dir / "point_scene.html"
        mesh_html = package_dir / "mesh_scene.html"
        run_command(
            [
                sys.executable,
                str(REPO_ROOT / "layout" / "open_local_pointcloud.py"),
                str(REPO_ROOT / point_meta["overlay_output"]),
                "--output-html",
                str(point_html),
                "--no-open",
            ]
        )
        run_command(
            [
                sys.executable,
                str(REPO_ROOT / "layout" / "open_local_mesh.py"),
                str(REPO_ROOT / mesh_meta["mesh_output"]),
                "--annotation-root",
                str(annotation_root),
                "--gt-attr-file",
                args.gt_attr_file,
                "--output-html",
                str(mesh_html),
                "--no-open",
            ]
        )
        html_outputs = {
            "point_html": make_relative(point_html),
            "mesh_html": make_relative(mesh_html),
        }

    gt_locs = load_locs(annotation_root, args.gt_attr_file, scene_id)
    pred_locs = load_locs(annotation_root, args.pred_attr_file, scene_id)
    package = {
        "scene_id": scene_id,
        "sample_index": args.sample_index,
        "qid": prediction.get("qid", args.sample_index),
        "dataset_json": args.dataset_json,
        "annotation_root": str(annotation_root),
        "scene_root": str(scene_root),
        "predictions_json": str(predictions_json),
        "input_text": sample.get("prompt", ""),
        "gt_answer": gt_answer,
        "gt_object_id": gt_object_id,
        "gt_bbox_loc": loc_to_list(gt_locs, gt_object_id),
        "model_prediction": model_prediction,
        "model_pred_object_ids": model_pred_object_ids,
        "model_pred_bbox_locs": [
            {"object_id": object_id, "loc": loc_to_list(pred_locs, object_id)}
            for object_id in model_pred_object_ids
        ],
        "validation": {
            "scene_matches_prediction": scene_id == prediction.get("scene_id"),
            "prompt_matches_prediction": normalize_text(sample.get("prompt", ""))
            == normalize_text(prediction.get("prompt", "")),
            "gt_id_matches_prediction": gt_object_id == int(prediction.get("gt_id", gt_object_id)),
        },
        "point_outputs": {
            "raw_ply": make_relative(point_meta["raw_output"]),
            "overlay_ply": make_relative(point_meta["overlay_output"]),
            "preview_png": make_relative(point_meta["preview_output"]),
            "metadata_json": make_relative(Path(point_meta["overlay_output"]).with_suffix(".json")),
        },
        "mesh_outputs": {
            "mesh_ply": make_relative(mesh_meta["mesh_output"]),
            "preview_png": make_relative(mesh_meta["preview_output"]),
            "vertex_labels": make_relative(mesh_meta["vertex_labels_output"]),
            "metadata_json": make_relative(Path(mesh_meta["mesh_output"]).with_suffix(".json")),
        },
        "html_outputs": html_outputs,
    }

    metadata_output = package_dir / "sample_package.json"
    metadata_output.write_text(json.dumps(package, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"package_metadata": make_relative(metadata_output), **package}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
