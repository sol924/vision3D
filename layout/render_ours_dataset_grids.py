from __future__ import annotations

import argparse
import json
import re
import shutil
import string
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from point.visualize_input_tokens_mesh import load_scene_mesh, write_colored_mesh


DEFAULT_OURS_ROOT = Path("/Volumes/T7 Shield/point_reduction_data/ours")
DEFAULT_ANNOTATION_ROOT = Path(
    "/Users/sol/Research/Training_Free_Token_Redcution/autodl/"
    "Training_Free_Token_Redcution/PointLLM_Reduction/datasets/annotations"
)
DEFAULT_SCENE_SOURCES = (
    Path("/Users/sol/Research/Training_Free_Token_Redcution/Fast3D/sample_data/scannet"),
    Path("/Users/sol/Research/PointLLM_Reduction/datasets/scannet_samples/scans"),
)
DATASET_JSON = {
    "scanrefer": "scanrefer_mask3d_val.json",
    "multi3dref": "multi3dref_mask3d_val.json",
    "scan2cap": "scan2cap_mask3d_val.json",
    "scanqa": "scanqa_val.json",
    "sqa3d": "sqa3d_val.json",
}
DEFAULT_DATASETS = ("scanrefer", "multi3dref", "scan2cap", "scanqa", "sqa3d")
SCANNET_BASE_URL = "https://kaldir.vc.in.tum.de/scannet/v2/scans"
SCENE_FILE_TEMPLATES = (
    "{scene}_vh_clean_2.ply",
    "{scene}.txt",
    "{scene}.aggregation.json",
    "{scene}_vh_clean_2.0.010000.segs.json",
)
OBJ_PATTERN = re.compile(r"<OBJ(\d{3})>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render correct-sample paper grids for each ours dataset prediction file."
    )
    parser.add_argument("--ours-root", default=str(DEFAULT_OURS_ROOT))
    parser.add_argument("--prediction-run", default="0625_best_02")
    parser.add_argument("--annotation-root", default=str(DEFAULT_ANNOTATION_ROOT))
    parser.add_argument(
        "--scene-source",
        action="append",
        default=[str(path) for path in DEFAULT_SCENE_SOURCES],
        help="Local ScanNet scene source directory. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=DEFAULT_DATASETS)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--width", type=int, default=3200)
    parser.add_argument("--height", type=int, default=1500)
    parser.add_argument("--renderer", choices=("auto", "open3d", "matplotlib"), default="matplotlib")
    parser.add_argument("--text-preset", choices=("compact", "a4-grid", "a4-2x6"), default="a4-2x6")
    parser.add_argument("--font-family", choices=("times", "arial"), default="times")
    parser.add_argument(
        "--correct-only",
        action="store_true",
        default=True,
        help="Select only samples that pass the dataset-specific correctness rule. Enabled by default.",
    )
    parser.add_argument(
        "--allow-incorrect",
        action="store_false",
        dest="correct_only",
        help="Disable correctness filtering and select the first available samples.",
    )
    parser.add_argument("--gt-attr-file", default="scannet_val_attributes.pt")
    parser.add_argument("--pred-attr-file", default="scannet_mask3d_val_attributes.pt")
    parser.add_argument("--no-download", action="store_true", help="Fail instead of downloading missing scenes.")
    parser.add_argument("--dry-run", action="store_true", help="Select samples and scenes without rendering.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def prediction_path(prediction_dir: Path, dataset: str) -> Path:
    matches = sorted(prediction_dir.glob(f"preds_epoch2_step*_{dataset}.json"))
    if not matches:
        matches = sorted(prediction_dir.glob(f"preds_*_{dataset}.json"))
    if not matches:
        raise FileNotFoundError(f"No prediction JSON found for {dataset} under {prediction_dir}")
    return matches[-1]


def eval_index(prediction: dict) -> int | None:
    match = re.search(r"_(\d+)$", str(prediction.get("eval_name_index", "")))
    return int(match.group(1)) if match else None


def build_scan2cap_lookup(dataset: list[dict]) -> dict[tuple[str, int, int], list[int]]:
    lookup: dict[tuple[str, int, int], list[int]] = {}
    for index, item in enumerate(dataset):
        key = (str(item.get("scene_id")), int(item.get("obj_id", -1)), int(item.get("pred_id", -1)))
        lookup.setdefault(key, []).append(index)
    return lookup


def normalize_answer_text(text: Any) -> str:
    normalized = str(text).lower().strip()
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    return " ".join(normalized.split())


def object_id_set_from_text(text: Any) -> set[int]:
    return {int(match.group(1)) for match in OBJ_PATTERN.finditer(str(text or ""))}


def is_prediction_correct(dataset_name: str, prediction: dict) -> bool:
    pred = str(prediction.get("pred", ""))
    refs = prediction.get("ref_captions", []) or []
    if dataset_name == "scanrefer":
        gt_ids: set[int] = set()
        for ref in refs:
            gt_ids.update(object_id_set_from_text(ref))
        return object_id_set_from_text(pred) == gt_ids
    if dataset_name == "multi3dref":
        gt_ids = {int(item) for item in refs if int(item) >= 0}
        pred_ids = object_id_set_from_text(pred)
        if not gt_ids:
            return normalize_answer_text(pred).startswith("no") and not pred_ids
        return pred_ids == gt_ids
    if dataset_name in ("scanqa", "sqa3d"):
        pred_norm = normalize_answer_text(pred)
        return any(
            ref_norm and (pred_norm == ref_norm or ref_norm in pred_norm)
            for ref_norm in (normalize_answer_text(ref) for ref in refs)
        )
    if dataset_name == "scan2cap":
        return int(prediction.get("pred_id", -1)) == int(prediction.get("gt_id", -2))
    return False


def gt_answer_for_score(dataset_name: str, sample: dict) -> str:
    answers = sample.get("ref_captions", [])
    if dataset_name == "multi3dref":
        if not answers:
            return "No."
        return ", ".join(f"<OBJ{int(item):03d}>" for item in answers)
    if not isinstance(answers, list):
        return str(answers)
    if dataset_name == "scan2cap" and answers:
        return str(answers[0])
    return ", ".join(str(item) for item in answers)


def rough_line_count(text: Any, chars_per_line: int = 42) -> int:
    normalized = " ".join(str(text).split())
    if not normalized:
        return 1
    words = normalized.split()
    lines = 1
    current = 0
    for word in words:
        word_len = len(word)
        if current == 0:
            current = word_len
        elif current + 1 + word_len <= chars_per_line:
            current += 1 + word_len
        else:
            lines += 1
            current = word_len
    return lines


def text_layout_score(dataset_name: str, sample: dict, prediction: dict) -> tuple[int, int, int]:
    texts = [
        sample.get("prompt", ""),
        gt_answer_for_score(dataset_name, sample),
        prediction.get("pred", ""),
    ]
    line_counts = [rough_line_count(text) for text in texts]
    char_count = sum(len(" ".join(str(text).split())) for text in texts)
    return max(line_counts), sum(line_counts), char_count


def resolve_annotation_index(
    dataset_name: str,
    prediction: dict,
    annotations: list[dict],
    scan2cap_lookup: dict[tuple[str, int, int], list[int]],
) -> int | None:
    index = eval_index(prediction)
    scene_id = prediction.get("scene_id")
    if index is not None and 0 <= index < len(annotations) and annotations[index].get("scene_id") == scene_id:
        return index
    if dataset_name == "scan2cap":
        key = (str(scene_id), int(prediction.get("gt_id", -1)), int(prediction.get("pred_id", -1)))
        candidates = scan2cap_lookup.get(key, [])
        return candidates[0] if candidates else None
    return None


def required_scene_files(scene_id: str) -> tuple[str, str]:
    return f"{scene_id}_vh_clean_2.ply", f"{scene_id}.txt"


def scene_is_usable(scene_dir: Path, scene_id: str) -> bool:
    ply_name, txt_name = required_scene_files(scene_id)
    return (scene_dir / ply_name).exists() and (scene_dir / txt_name).exists()


def find_local_scene(scene_sources: list[Path], scene_id: str) -> Path | None:
    for root in scene_sources:
        scene_dir = root / scene_id
        if scene_is_usable(scene_dir, scene_id):
            return scene_dir
    return None


def copy_scene(source_dir: Path, target_dir: Path, scene_id: str) -> list[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for template in SCENE_FILE_TEMPLATES:
        filename = template.format(scene=scene_id)
        source = source_dir / filename
        if not source.exists():
            continue
        target = target_dir / filename
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            shutil.copy2(source, target)
        copied.append(filename)
    return copied


def download_scene(scene_id: str, target_dir: Path) -> list[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    downloaded: list[str] = []
    for template in SCENE_FILE_TEMPLATES:
        filename = template.format(scene=scene_id)
        target = target_dir / filename
        if target.exists() and target.stat().st_size > 0:
            downloaded.append(filename)
            continue
        url = f"{SCANNET_BASE_URL}/{scene_id}/{filename}"
        print(f"download {url}")
        with session.get(url, stream=True, timeout=90, verify=False) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        downloaded.append(filename)
    return downloaded


def ensure_scene(
    scene_id: str,
    scene_root: Path,
    scene_sources: list[Path],
    allow_download: bool,
) -> dict[str, Any]:
    target_dir = scene_root / scene_id
    if scene_is_usable(target_dir, scene_id):
        return {"scene_id": scene_id, "scene_dir": str(target_dir), "source": "existing"}
    local_source = find_local_scene(scene_sources, scene_id)
    if local_source is not None:
        files = copy_scene(local_source, target_dir, scene_id)
        return {
            "scene_id": scene_id,
            "scene_dir": str(target_dir),
            "source": str(local_source),
            "files": files,
        }
    if not allow_download:
        raise FileNotFoundError(f"Scene {scene_id} is missing and --no-download was set.")
    files = download_scene(scene_id, target_dir)
    return {
        "scene_id": scene_id,
        "scene_dir": str(target_dir),
        "source": SCANNET_BASE_URL,
        "files": files,
    }


def scene_available_locally(scene_id: str, scene_root: Path, scene_sources: list[Path]) -> bool:
    return scene_is_usable(scene_root / scene_id, scene_id) or find_local_scene(scene_sources, scene_id) is not None


def select_records(
    dataset_name: str,
    annotations: list[dict],
    predictions: list[dict],
    scene_root: Path,
    scene_sources: list[Path],
    limit: int,
    correct_only: bool,
) -> list[dict]:
    scan2cap_lookup = build_scan2cap_lookup(annotations) if dataset_name == "scan2cap" else {}
    candidates: list[dict] = []
    for prediction_order, prediction in enumerate(predictions):
        is_correct = is_prediction_correct(dataset_name, prediction)
        if correct_only and not is_correct:
            continue
        scene_id = str(prediction.get("scene_id", ""))
        if not scene_id:
            continue
        annotation_index = resolve_annotation_index(dataset_name, prediction, annotations, scan2cap_lookup)
        if annotation_index is None:
            continue
        sample = annotations[annotation_index]
        is_local = scene_available_locally(scene_id, scene_root, scene_sources)
        record = {
            "dataset": dataset_name,
            "annotation_index": annotation_index,
            "prediction_order": prediction_order,
            "scene_id": scene_id,
            "eval_name_index": prediction.get("eval_name_index"),
            "prediction": prediction,
            "is_correct": is_correct,
            "is_scene_local": is_local,
            "text_score": text_layout_score(dataset_name, sample, prediction),
        }
        candidates.append(record)

    candidates.sort(key=lambda item: (not item["is_scene_local"], item["text_score"], item["prediction_order"]))
    selected: list[dict] = []
    selected_keys: set[int] = set()
    seen_scenes: set[str] = set()
    for record in candidates:
        if record["scene_id"] in seen_scenes:
            continue
        selected.append(record)
        selected_keys.add(record["prediction_order"])
        seen_scenes.add(record["scene_id"])
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for record in candidates:
            if record["prediction_order"] in selected_keys:
                continue
            selected.append(record)
            selected_keys.add(record["prediction_order"])
            if len(selected) >= limit:
                break
    if len(selected) < limit:
        label = "correct" if correct_only else "available"
        raise RuntimeError(f"{dataset_name}: only selected {len(selected)} {label} samples.")
    return selected


def extract_object_ids(text: str) -> list[int]:
    ids: list[int] = []
    for match in OBJ_PATTERN.finditer(text or ""):
        object_id = int(match.group(1))
        if object_id not in ids:
            ids.append(object_id)
    return ids


def loc_to_list(attrs: dict, scene_id: str, object_id: int) -> list[float] | None:
    if object_id < 0 or scene_id not in attrs:
        return None
    locs = attrs[scene_id]["locs"]
    if object_id >= len(locs):
        return None
    return [float(item) for item in locs[object_id].tolist()]


def format_obj(object_id: int) -> str:
    return f"<OBJ{object_id:03d}>"


def normalize_gt_answer(dataset_name: str, sample: dict) -> list[str]:
    answers = sample.get("ref_captions", [])
    if dataset_name == "multi3dref":
        if not answers:
            return ["No."]
        return [", ".join(format_obj(int(item)) for item in answers)]
    if not isinstance(answers, list):
        return [str(answers)]
    if dataset_name == "scan2cap" and answers:
        return [str(answers[0])]
    return [str(item) for item in answers] or [""]


def gt_object_ids(dataset_name: str, sample: dict, prediction: dict) -> list[int]:
    if dataset_name == "multi3dref":
        return [int(item) for item in sample.get("ref_captions", []) if int(item) >= 0]
    object_id = int(sample.get("obj_id", prediction.get("gt_id", -1)))
    return [object_id] if object_id >= 0 else []


def prediction_object_ids(dataset_name: str, prediction: dict) -> list[int]:
    ids = extract_object_ids(str(prediction.get("pred", "")))
    if ids:
        return ids
    if dataset_name == "scan2cap" and prediction.get("pred_id") is not None:
        pred_id = int(prediction["pred_id"])
        return [pred_id] if pred_id >= 0 else []
    return []


def build_scene_mesh(scene_id: str, scene_root: Path, scene_mesh_root: Path) -> Path:
    output_dir = scene_mesh_root / scene_id
    output_path = output_dir / f"{scene_id}_aligned_original_mesh.ply"
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    points, colors, faces = load_scene_mesh(scene_root / scene_id, scene_id)
    write_colored_mesh(points, colors, faces, output_path)
    return output_path


def build_package(
    record: dict,
    sample: dict,
    prediction: dict,
    mesh_path: Path,
    package_path: Path,
    annotation_root: Path,
    prediction_path_value: Path,
    dataset_json: str,
    gt_attrs: dict,
    pred_attrs: dict,
) -> dict:
    scene_id = str(sample["scene_id"])
    gt_ids = gt_object_ids(record["dataset"], sample, prediction)
    pred_ids = prediction_object_ids(record["dataset"], prediction)
    gt_locs = [
        {"object_id": object_id, "loc": loc_to_list(gt_attrs, scene_id, object_id)}
        for object_id in gt_ids
    ]
    pred_locs = [
        {"object_id": object_id, "loc": loc_to_list(pred_attrs, scene_id, object_id)}
        for object_id in pred_ids
    ]
    package = {
        "scene_id": scene_id,
        "sample_index": record["annotation_index"],
        "prediction_order": record["prediction_order"],
        "eval_name_index": record.get("eval_name_index"),
        "dataset_name": record["dataset"],
        "dataset_json": dataset_json,
        "annotation_root": str(annotation_root),
        "predictions_json": str(prediction_path_value),
        "input_text": sample.get("prompt", ""),
        "gt_answer": normalize_gt_answer(record["dataset"], sample),
        "gt_object_ids": gt_ids,
        "gt_bbox_locs": gt_locs,
        "gt_object_id": gt_ids[0] if gt_ids else -1,
        "gt_bbox_loc": gt_locs[0]["loc"] if len(gt_locs) == 1 else None,
        "model_prediction": str(prediction.get("pred", "")),
        "model_pred_object_ids": pred_ids,
        "model_pred_bbox_locs": pred_locs,
        "mesh_outputs": {
            "mesh_ply": str(mesh_path),
            "preview_png": "",
            "vertex_labels": "",
            "metadata_json": "",
        },
        "point_outputs": {},
    }
    write_json(package_path, package)
    return package


def render_report(
    package_path: Path,
    output_png: Path,
    width: int,
    height: int,
    renderer: str,
    text_preset: str,
    font_family: str,
) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "layout" / "render_sample_report.py"),
            str(package_path),
            "--output-png",
            str(output_png),
            "--scene-mode",
            "mesh",
            "--renderer",
            renderer,
            "--width",
            str(width),
            "--height",
            str(height),
            "--text-preset",
            text_preset,
            "--font-family",
            font_family,
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    start = result.stdout.find("{")
    return json.loads(result.stdout[start:]) if start >= 0 else {"report_png": str(output_png)}


def stitch_grid(tile_paths: list[Path], output_png: Path, cols: int, rows: int) -> None:
    if len(tile_paths) != cols * rows:
        raise ValueError(f"Expected {cols * rows} tiles, got {len(tile_paths)}")
    images = [Image.open(path).convert("RGB") for path in tile_paths]
    tile_w, tile_h = images[0].size
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    for index, image in enumerate(images):
        if image.size != (tile_w, tile_h):
            image = image.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        x = (index % cols) * tile_w
        y = (index // cols) * tile_h
        canvas.paste(image, (x, y))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png)
    for image in images:
        image.close()


def clear_previous_tiles(tile_dir: Path) -> int:
    if not tile_dir.exists():
        return 0
    removed = 0
    for path in tile_dir.glob("[0-9][0-9]_*.png"):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def clear_previous_grids(grid_dir: Path, dataset_name: str) -> int:
    if not grid_dir.exists():
        return 0
    removed = 0
    for path in grid_dir.glob(f"{dataset_name}_*_grid.png"):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def main() -> int:
    args = parse_args()
    ours_root = Path(args.ours_root)
    prediction_dir = ours_root / args.prediction_run
    annotation_root = Path(args.annotation_root)
    output_dir = Path(args.output_dir) if args.output_dir else ours_root / f"visualization_{args.prediction_run}"
    scene_root = output_dir / "scannet_samples"
    scene_sources = [Path(item) for item in args.scene_source]
    scene_mesh_root = output_dir / "scene_meshes"
    package_root = output_dir / "packages"
    tile_root = output_dir / "tiles"
    grid_root = output_dir / "grids"

    gt_attrs = torch.load(annotation_root / args.gt_attr_file, map_location="cpu", weights_only=False)
    pred_attrs = torch.load(annotation_root / args.pred_attr_file, map_location="cpu", weights_only=False)

    manifest: dict[str, Any] = {
        "prediction_run": args.prediction_run,
        "prediction_dir": str(prediction_dir),
        "annotation_root": str(annotation_root),
        "output_dir": str(output_dir),
        "correct_only": args.correct_only,
        "limit": args.limit,
        "cols": args.cols,
        "rows": args.rows,
        "tile_size": [args.width, args.height],
        "text_preset": args.text_preset,
        "font_family": args.font_family,
        "datasets": {},
    }

    for dataset_name in args.datasets:
        dataset_json = DATASET_JSON[dataset_name]
        annotations = load_json(annotation_root / dataset_json)
        pred_path = prediction_path(prediction_dir, dataset_name)
        predictions = load_json(pred_path)
        records = select_records(
            dataset_name,
            annotations,
            predictions,
            scene_root,
            scene_sources,
            args.limit,
            args.correct_only,
        )
        dataset_manifest: dict[str, Any] = {
            "dataset_json": dataset_json,
            "predictions_json": str(pred_path),
            "correct_only": args.correct_only,
            "records": [],
        }
        tile_paths: list[Path] = []
        if not args.dry_run:
            removed = clear_previous_tiles(tile_root / dataset_name)
            if removed:
                print(f"{dataset_name}: removed {removed} stale tile PNGs")
            removed_grids = clear_previous_grids(grid_root, dataset_name)
            if removed_grids:
                print(f"{dataset_name}: removed {removed_grids} stale grid PNGs")
        selection_label = "correct samples" if args.correct_only else "samples"
        print(f"{dataset_name}: selected {len(records)} {selection_label}")
        for ordinal, record in enumerate(records):
            sample = annotations[record["annotation_index"]]
            prediction = record["prediction"]
            if args.dry_run:
                scene_info = {
                    "scene_id": record["scene_id"],
                    "available_locally": scene_available_locally(record["scene_id"], scene_root, scene_sources),
                }
                package = {}
                tile_path = tile_root / dataset_name / f"{ordinal:02d}_{record['eval_name_index']}.png"
                report_info = {}
            else:
                scene_info = ensure_scene(
                    record["scene_id"],
                    scene_root,
                    scene_sources,
                    allow_download=not args.no_download,
                )
                mesh_path = build_scene_mesh(record["scene_id"], scene_root, scene_mesh_root)
                package_path = (
                    package_root
                    / dataset_name
                    / f"{ordinal:02d}_{record['eval_name_index']}__{record['scene_id']}"
                    / "sample_package.json"
                )
                package = build_package(
                    record,
                    sample,
                    prediction,
                    mesh_path,
                    package_path,
                    annotation_root,
                    pred_path,
                    dataset_json,
                    gt_attrs,
                    pred_attrs,
                )
                tile_path = tile_root / dataset_name / f"{ordinal:02d}_{record['eval_name_index']}__{record['scene_id']}.png"
                report_info = render_report(
                    package_path,
                    tile_path,
                    args.width,
                    args.height,
                    args.renderer,
                    args.text_preset,
                    args.font_family,
                )
            tile_paths.append(tile_path)
            dataset_manifest["records"].append(
                {
                    "ordinal": ordinal,
                    "dataset": dataset_name,
                    "scene_id": record["scene_id"],
                    "annotation_index": record["annotation_index"],
                    "prediction_order": record["prediction_order"],
                    "eval_name_index": record["eval_name_index"],
                    "is_correct": record["is_correct"],
                    "scene_info": scene_info,
                    "tile_png": str(tile_path),
                    "report_info": report_info,
                    "gt_answer": package.get("gt_answer") if package else None,
                    "model_prediction": prediction.get("pred"),
                }
            )
            print(f"  {ordinal:02d} {record['eval_name_index']} {record['scene_id']}")
        grid_path = grid_root / f"{dataset_name}_{args.limit}_grid.png"
        if not args.dry_run:
            stitch_grid(tile_paths, grid_path, args.cols, args.rows)
        dataset_manifest["grid_png"] = str(grid_path)
        manifest["datasets"][dataset_name] = dataset_manifest

    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"manifest": str(output_dir / "manifest.json"), "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
