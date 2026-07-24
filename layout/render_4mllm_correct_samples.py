from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "vision3d-matplotlib"),
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from point.visualize_input_tokens_mesh import load_scene_mesh, write_colored_mesh


DEFAULT_MODELS = ("qwen2_7b", "llava7b", "qwen3_8b", "llama3_8b")
DEFAULT_DATASETS = ("scanrefer", "multi3dref", "scan2cap", "scanqa", "sqa3d")
MODEL_LABELS = {
    "qwen2_7b": "Qwen2-7B-Instruct",
    "llava7b": "LLaVA / Vicuna-7B",
    "qwen3_8b": "Qwen3-8B-Instruct",
    "llama3_8b": "Llama-3-8B-Instruct",
}
DATASET_JSON = {
    "scanrefer": "scanrefer_mask3d_val.json",
    "multi3dref": "multi3dref_mask3d_val.json",
    "scan2cap": "scan2cap_mask3d_val.json",
    "scanqa": "scanqa_val.json",
    "sqa3d": "sqa3d_val.json",
}
OBJ_PATTERN = re.compile(r"<OBJ(\d{3})>")
GENERATION_SPECIAL_TOKENS = (
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<s>",
    "</s>",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select strictly correct samples for four MLLMs and render per-sample reports, "
            "2x5 dataset grids, and an HTML gallery."
        )
    )
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument(
        "--scene-source",
        action="append",
        required=True,
        help="Directory containing sceneXXXX_XX folders. Pass once per source root.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/90pct_4mllm_correct_samples",
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DEFAULT_DATASETS,
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--width", type=int, default=3200)
    parser.add_argument("--height", type=int, default=1500)
    parser.add_argument(
        "--renderer",
        choices=("auto", "open3d", "matplotlib"),
        default="matplotlib",
    )
    parser.add_argument(
        "--text-preset",
        choices=("compact", "a4-grid", "a4-2x6"),
        default="a4-2x6",
    )
    parser.add_argument("--font-family", choices=("times", "arial"), default="times")
    parser.add_argument("--max-points", type=int, default=80000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gt-attr-file", default="scannet_val_attributes.pt")
    parser.add_argument("--pred-attr-file", default="scannet_mask3d_val_attributes.pt")
    parser.add_argument(
        "--scan2cap-review-json",
        default=None,
        help="Review file. Defaults to <output-dir>/scan2cap_review.json.",
    )
    parser.add_argument(
        "--prepare-scan2cap-review",
        action="store_true",
        help="Write deterministic Scan2Cap review candidates and exit.",
    )
    parser.add_argument("--review-candidates-per-model", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").split())


def prediction_key(item: dict[str, Any]) -> tuple[str, str]:
    return str(item.get("scene_id", "")), normalize_text(item.get("prompt", ""))


def stable_candidate_id(model: str, dataset: str, item: dict[str, Any]) -> str:
    payload = json.dumps(
        [
            model,
            dataset,
            item.get("scene_id"),
            normalize_text(item.get("prompt")),
            item.get("gt_id"),
            item.get("pred_id"),
            item.get("pred"),
            item.get("ref_captions"),
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def clean_generated_text(text: Any) -> str:
    cleaned = str(text or "")
    for token in GENERATION_SPECIAL_TOKENS:
        cleaned = cleaned.replace(token, " ")
    cleaned = cleaned.replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace(" .", ".")
    return cleaned.strip()


def clean_answer(data: Any) -> str:
    # Kept in sync with Free3D/utils/helper.py and its LEO/ScanQA normalization.
    text = str(data or "").lower()
    text = re.sub(r"[ ]+$", "", text)
    text = re.sub(r"^[ ]+", "", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\.[ ]{2,}", ". ", text)
    text = re.sub(r"[^a-zA-Z0-9,'\s\-:]+", "", text)
    replacements = {
        r"\bletf\b": "left",
        r"\blet\b": "left",
        r"\btehre\b": "there",
        r"\brigth\b": "right",
        r"\brght\b": "right",
        r"\bbehine\b": "behind",
        r"\btv\b": "TV",
        r"\bchai\b": "chair",
        r"\bwasing\b": "washing",
        r"\bwaslked\b": "walked",
        r"\boclock\b": "o'clock",
        r"\bo'[ ]+clock\b": "o'clock",
        r"\b0\b": "zero",
        r"\bnone\b": "zero",
        r"\b1\b": "one",
        r"\b2\b": "two",
        r"\b3\b": "three",
        r"\b4\b": "four",
        r"\b5\b": "five",
        r"\b6\b": "six",
        r"\b7\b": "seven",
        r"\b8\b": "eight",
        r"\b9\b": "nine",
        r"\b10\b": "ten",
        r"\b11\b": "eleven",
        r"\b12\b": "twelve",
        r"\b13\b": "thirteen",
        r"\b14\b": "fourteen",
        r"\b15\b": "fifteen",
        r"\b16\b": "sixteen",
        r"\b17\b": "seventeen",
        r"\b18\b": "eighteen",
        r"\b19\b": "nineteen",
        r"\b20\b": "twenty",
        r"\b23\b": "twenty-three",
        r"\bbackwards\b": "backward",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\b([a-zA-Z]+)([0-9])\b", r"\g<1>", text)
    text = re.sub(r"\ba\b ([a-zA-Z]+)", r"\g<1>", text)
    text = re.sub(r"\ban\b ([a-zA-Z]+)", r"\g<1>", text)
    text = re.sub(r"\bthe\b ([a-zA-Z]+)", r"\g<1>", text)
    return text


def normalized_qa_answer(text: Any) -> str:
    generated = clean_generated_text(text)
    if len(generated) > 1 and generated.endswith("."):
        generated = generated[:-1]
    if generated:
        generated = generated[0].lower() + generated[1:]
    return clean_answer(generated)


def parse_object_ids(text: Any, object_count: int | None = None) -> list[int]:
    ids: list[int] = []
    for match in OBJ_PATTERN.finditer(str(text or "")):
        object_id = int(match.group(1))
        if object_count is None or 0 <= object_id < object_count:
            ids.append(object_id)
    return ids


def dedupe(items: list[Any]) -> list[Any]:
    result: list[Any] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def loc_array(attrs: dict[str, Any], scene_id: str, object_id: int) -> np.ndarray:
    return attrs[scene_id]["locs"][object_id].detach().cpu().numpy().astype(np.float64)


def loc_list(attrs: dict[str, Any], scene_id: str, object_id: int) -> list[float] | None:
    if scene_id not in attrs:
        return None
    locs = attrs[scene_id]["locs"]
    if object_id < 0 or object_id >= len(locs):
        return None
    return [float(value) for value in locs[object_id].detach().cpu().tolist()]


def axis_aligned_iou(loc_a: np.ndarray, loc_b: np.ndarray) -> float:
    a_size = np.maximum(loc_a[3:6], 0.0)
    b_size = np.maximum(loc_b[3:6], 0.0)
    a_min, a_max = loc_a[:3] - a_size / 2.0, loc_a[:3] + a_size / 2.0
    b_min, b_max = loc_b[:3] - b_size / 2.0, loc_b[:3] + b_size / 2.0
    intersection_size = np.maximum(0.0, np.minimum(a_max, b_max) - np.maximum(a_min, b_min))
    intersection = float(np.prod(intersection_size))
    union = float(np.prod(a_size) + np.prod(b_size) - intersection)
    return intersection / union if union > 0 else 0.0


def evaluate_prediction(
    dataset: str,
    item: dict[str, Any],
    gt_attrs: dict[str, Any],
    pred_attrs: dict[str, Any],
) -> dict[str, Any]:
    scene_id = str(item["scene_id"])
    if scene_id not in gt_attrs or scene_id not in pred_attrs:
        return {"correct": False, "reason": "missing_attributes"}

    gt_count = len(gt_attrs[scene_id]["locs"])
    pred_count = len(pred_attrs[scene_id]["locs"])

    if dataset == "scanrefer":
        ids = parse_object_ids(item.get("pred", ""), pred_count)
        gt_id = int(item.get("gt_id", -1))
        if not ids or not 0 <= gt_id < gt_count:
            return {"correct": False, "reason": "missing_valid_object_id"}
        pred_id = ids[0]
        iou = axis_aligned_iou(
            loc_array(pred_attrs, scene_id, pred_id),
            loc_array(gt_attrs, scene_id, gt_id),
        )
        return {
            "correct": iou >= 0.5,
            "metric": "Acc@0.50",
            "iou": iou,
            "threshold": 0.5,
            "gt_object_ids": [gt_id],
            "pred_object_ids": [pred_id],
        }

    if dataset == "multi3dref":
        pred_ids = parse_object_ids(item.get("pred", ""), pred_count)
        gt_ids = [
            int(value)
            for value in item.get("ref_captions", [])
            if 0 <= int(value) < gt_count
        ]
        if not gt_ids:
            correct = len(pred_ids) == 0
            return {
                "correct": correct,
                "metric": "F1@0.50",
                "f1": 1.0 if correct else 0.0,
                "threshold": 0.5,
                "gt_object_ids": [],
                "pred_object_ids": pred_ids,
                "matches": [],
            }
        if not pred_ids:
            return {
                "correct": False,
                "metric": "F1@0.50",
                "f1": 0.0,
                "threshold": 0.5,
                "gt_object_ids": gt_ids,
                "pred_object_ids": [],
                "matches": [],
            }
        matrix_size = max(len(pred_ids), len(gt_ids))
        iou_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float64)
        for pred_index, pred_id in enumerate(pred_ids):
            for gt_index, gt_id in enumerate(gt_ids):
                iou_matrix[pred_index, gt_index] = axis_aligned_iou(
                    loc_array(pred_attrs, scene_id, pred_id),
                    loc_array(gt_attrs, scene_id, gt_id),
                )
        rows, cols = linear_sum_assignment(-iou_matrix)
        matches: list[dict[str, Any]] = []
        true_positives = 0
        for row, col in zip(rows, cols):
            if row >= len(pred_ids) or col >= len(gt_ids):
                continue
            iou = float(iou_matrix[row, col])
            if iou >= 0.5:
                true_positives += 1
            matches.append(
                {
                    "pred_object_id": pred_ids[row],
                    "gt_object_id": gt_ids[col],
                    "iou": iou,
                }
            )
        denominator = len(pred_ids) + len(gt_ids)
        f1 = 2.0 * true_positives / denominator if denominator else 1.0
        return {
            "correct": math.isclose(f1, 1.0),
            "metric": "F1@0.50",
            "f1": f1,
            "threshold": 0.5,
            "gt_object_ids": gt_ids,
            "pred_object_ids": pred_ids,
            "matches": matches,
        }

    if dataset in ("scanqa", "sqa3d"):
        prediction = normalized_qa_answer(item.get("pred", ""))
        references = [clean_answer(value) for value in item.get("ref_captions", [])]
        correct = bool(prediction) and prediction in references
        gt_id = int(item.get("gt_id", -1))
        gt_ids = [gt_id] if 0 <= gt_id < gt_count else []
        return {
            "correct": correct,
            "metric": "EM1",
            "em1": 1 if correct else 0,
            "normalized_prediction": prediction,
            "normalized_references": references,
            "gt_object_ids": gt_ids,
            "pred_object_ids": [],
        }

    if dataset == "scan2cap":
        gt_id = int(item.get("gt_id", -1))
        pred_id = int(item.get("pred_id", -1))
        if not 0 <= gt_id < gt_count or not 0 <= pred_id < pred_count:
            return {"correct": False, "reason": "missing_valid_object_id"}
        iou = axis_aligned_iou(
            loc_array(pred_attrs, scene_id, pred_id),
            loc_array(gt_attrs, scene_id, gt_id),
        )
        return {
            "correct": iou >= 0.5,
            "spatial_pass": iou >= 0.5,
            "semantic_review_required": True,
            "metric": "IoU@0.50 + semantic review",
            "iou": iou,
            "threshold": 0.5,
            "gt_object_ids": [gt_id],
            "pred_object_ids": [pred_id],
        }

    raise ValueError(f"Unsupported dataset: {dataset}")


def rough_line_count(text: Any, chars_per_line: int = 42) -> int:
    normalized = normalize_text(text)
    if not normalized:
        return 1
    lines = 1
    current = 0
    for word in normalized.split():
        word_length = len(word)
        if current == 0:
            current = word_length
        elif current + 1 + word_length <= chars_per_line:
            current += 1 + word_length
        else:
            lines += 1
            current = word_length
    return lines


def text_layout_score(item: dict[str, Any]) -> tuple[int, int, int]:
    references = item.get("ref_captions", [])
    if not isinstance(references, list):
        references = [references]
    texts = [
        item.get("prompt", ""),
        " / ".join(str(value) for value in references),
        item.get("pred", ""),
    ]
    lines = [rough_line_count(text) for text in texts]
    return max(lines), sum(lines), sum(len(normalize_text(text)) for text in texts)


def find_scene_catalog(scene_sources: list[Path]) -> dict[str, dict[str, str]]:
    catalog: dict[str, dict[str, str]] = {}
    for source_root in scene_sources:
        if not source_root.is_dir():
            continue
        for scene_dir in sorted(source_root.glob("scene????_??")):
            scene_id = scene_dir.name
            ply_path = scene_dir / f"{scene_id}_vh_clean_2.ply"
            meta_path = scene_dir / f"{scene_id}.txt"
            if not ply_path.is_file() or not meta_path.is_file():
                continue
            if scene_id in catalog:
                continue
            aligned_mesh = (
                source_root.parent
                / "scene_meshes"
                / scene_id
                / f"{scene_id}_aligned_original_mesh.ply"
            )
            catalog[scene_id] = {
                "scene_dir": str(scene_dir),
                "scene_source": str(source_root),
                "aligned_mesh": str(aligned_mesh) if aligned_mesh.is_file() else "",
            }
    return catalog


def grouped_unique_rows(
    rows: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], int]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[prediction_key(row)].append(row)
    unique = {key: items[0] for key, items in groups.items() if len(items) == 1}
    duplicate_records = sum(len(items) for items in groups.values() if len(items) > 1)
    return unique, duplicate_records


def annotation_lookup(
    annotations: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], tuple[int, dict[str, Any]]], int]:
    groups: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, item in enumerate(annotations):
        groups[prediction_key(item)].append((index, item))
    unique = {key: items[0] for key, items in groups.items() if len(items) == 1}
    ambiguous_records = sum(len(items) for items in groups.values() if len(items) > 1)
    return unique, ambiguous_records


def build_candidates(
    model: str,
    dataset: str,
    predictions: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    scene_catalog: dict[str, dict[str, str]],
    gt_attrs: dict[str, Any],
    pred_attrs: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    prediction_rows, duplicate_prediction_records = grouped_unique_rows(predictions)
    annotation_rows, ambiguous_annotation_records = annotation_lookup(annotations)
    candidates: list[dict[str, Any]] = []
    missing_annotation = 0
    missing_scene = 0
    incorrect = 0

    prediction_order = {id(item): index for index, item in enumerate(predictions)}
    for key, item in prediction_rows.items():
        annotation_entry = annotation_rows.get(key)
        if annotation_entry is None:
            missing_annotation += 1
            continue
        annotation_index, _annotation = annotation_entry
        scene_id = str(item.get("scene_id", ""))
        if scene_id not in scene_catalog:
            missing_scene += 1
            continue
        evidence = evaluate_prediction(dataset, item, gt_attrs, pred_attrs)
        if not evidence.get("correct"):
            incorrect += 1
            continue
        candidate = {
            "candidate_id": stable_candidate_id(model, dataset, item),
            "model": model,
            "model_label": MODEL_LABELS.get(model, model),
            "dataset": dataset,
            "annotation_index": annotation_index,
            "prediction_order": prediction_order[id(item)],
            "scene_id": scene_id,
            "prompt": str(item.get("prompt", "")),
            "prediction": str(item.get("pred", "")),
            "references": item.get("ref_captions", []),
            "gt_id": int(item.get("gt_id", -1)),
            "pred_id": int(item.get("pred_id", -1)),
            "type_info": item.get("type_info"),
            "correctness": evidence,
            "text_score": list(text_layout_score(item)),
            "source_record": item,
        }
        candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            tuple(item["text_score"]),
            item["prediction_order"],
            item["candidate_id"],
        )
    )
    stats = {
        "prediction_records": len(predictions),
        "unique_prediction_records": len(prediction_rows),
        "excluded_duplicate_prediction_records": duplicate_prediction_records,
        "annotation_records": len(annotations),
        "excluded_ambiguous_annotation_records": ambiguous_annotation_records,
        "missing_annotation": missing_annotation,
        "missing_local_scene": missing_scene,
        "incorrect": incorrect,
        "strict_local_candidates": len(candidates),
        "strict_local_unique_scenes": len({item["scene_id"] for item in candidates}),
    }
    return candidates, stats


def diverse_order(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    seen_scenes: set[str] = set()
    for candidate in candidates:
        if candidate["scene_id"] in seen_scenes:
            deferred.append(candidate)
            continue
        result.append(candidate)
        seen_scenes.add(candidate["scene_id"])
    return result + deferred


def prepare_scan2cap_review(
    candidates_by_model: dict[str, list[dict[str, Any]]],
    review_path: Path,
    candidate_limit: int,
    force: bool,
) -> None:
    if review_path.exists() and not force:
        raise FileExistsError(
            f"{review_path} already exists. Pass --force to replace the review template."
        )
    models: dict[str, list[dict[str, Any]]] = {}
    for model, candidates in candidates_by_model.items():
        ordered = diverse_order(candidates)[:candidate_limit]
        models[model] = [
            {
                "candidate_id": candidate["candidate_id"],
                "scene_id": candidate["scene_id"],
                "annotation_index": candidate["annotation_index"],
                "prediction_order": candidate["prediction_order"],
                "prompt": candidate["prompt"],
                "prediction": candidate["prediction"],
                "references": candidate["references"],
                "bbox_iou": candidate["correctness"]["iou"],
                "approved": False,
                "matched_reference_index": None,
                "review_reason": "",
            }
            for candidate in ordered
        ]
    review = {
        "schema_version": 1,
        "criteria": {
            "spatial": "Mask3D candidate bbox IoU with GT bbox must be >= 0.5.",
            "semantic": (
                "Prediction must agree with at least one reference on object category and "
                "must not contradict a key color, shape, material, or spatial relation. "
                "Ambiguous cases are rejected."
            ),
            "selection": (
                "Review in listed order and approve the first ten passing candidates from "
                "ten different scenes for each model."
            ),
        },
        "models": models,
    }
    write_json(review_path, review)


def approved_scan2cap_candidates(
    candidates: list[dict[str, Any]],
    review_entries: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    selected: list[dict[str, Any]] = []
    seen_scenes: set[str] = set()
    for review in review_entries:
        if not review.get("approved"):
            continue
        candidate_id = str(review.get("candidate_id", ""))
        candidate = by_id.get(candidate_id)
        if candidate is None:
            raise ValueError(f"Approved Scan2Cap candidate is unavailable: {candidate_id}")
        scene_id = candidate["scene_id"]
        if scene_id in seen_scenes:
            continue
        reference_index = review.get("matched_reference_index")
        references = candidate["references"]
        if not isinstance(reference_index, int) or not 0 <= reference_index < len(references):
            raise ValueError(f"{candidate_id}: matched_reference_index is invalid")
        reason = normalize_text(review.get("review_reason", ""))
        if not reason:
            raise ValueError(f"{candidate_id}: review_reason is required")
        candidate = dict(candidate)
        candidate["semantic_review"] = {
            "approved": True,
            "matched_reference_index": reference_index,
            "matched_reference": references[reference_index],
            "review_reason": reason,
        }
        selected.append(candidate)
        seen_scenes.add(scene_id)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        raise RuntimeError(
            f"Scan2Cap review only approved {len(selected)} distinct-scene samples; need {limit}."
        )
    return selected


def select_candidates(
    candidates: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    ordered = diverse_order(candidates)
    selected: list[dict[str, Any]] = []
    seen_scenes: set[str] = set()
    for candidate in ordered:
        if candidate["scene_id"] in seen_scenes:
            continue
        selected.append(candidate)
        seen_scenes.add(candidate["scene_id"])
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        raise RuntimeError(
            f"Only {len(selected)} distinct-scene correct candidates are available; need {limit}."
        )
    return selected


def find_or_build_mesh(
    scene_id: str,
    scene_info: dict[str, str],
    output_mesh_root: Path,
) -> Path:
    cached = Path(scene_info.get("aligned_mesh", ""))
    if cached.is_file() and cached.stat().st_size > 0:
        return cached
    output_path = (
        output_mesh_root / scene_id / f"{scene_id}_aligned_original_mesh.ply"
    )
    if output_path.is_file() and output_path.stat().st_size > 0:
        return output_path
    points, colors, faces = load_scene_mesh(Path(scene_info["scene_dir"]), scene_id)
    write_colored_mesh(points, colors, faces, output_path)
    return output_path


def format_gt_answers(candidate: dict[str, Any]) -> list[str]:
    dataset = candidate["dataset"]
    references = candidate["references"]
    if dataset == "multi3dref":
        if not references:
            return ["No."]
        return [", ".join(f"GT instance {int(value):03d}" for value in references)]
    if dataset == "scan2cap":
        return [str(candidate["semantic_review"]["matched_reference"])]
    if not isinstance(references, list):
        return [str(references)]
    return dedupe([str(value) for value in references]) or [""]


def build_package(
    candidate: dict[str, Any],
    mesh_path: Path,
    annotation_root: Path,
    predictions_path: Path,
    gt_attrs: dict[str, Any],
    pred_attrs: dict[str, Any],
) -> dict[str, Any]:
    scene_id = candidate["scene_id"]
    correctness = candidate["correctness"]
    gt_ids = dedupe([int(value) for value in correctness.get("gt_object_ids", [])])
    pred_ids = dedupe([int(value) for value in correctness.get("pred_object_ids", [])])
    gt_boxes = [
        {"object_id": object_id, "loc": loc_list(gt_attrs, scene_id, object_id)}
        for object_id in gt_ids
    ]
    pred_boxes = [
        {"object_id": object_id, "loc": loc_list(pred_attrs, scene_id, object_id)}
        for object_id in pred_ids
    ]
    package: dict[str, Any] = {
        "scene_id": scene_id,
        "sample_index": candidate["annotation_index"],
        "prediction_order": candidate["prediction_order"],
        "candidate_id": candidate["candidate_id"],
        "model_id": candidate["model"],
        "model_label": candidate["model_label"],
        "dataset_name": candidate["dataset"],
        "dataset_json": DATASET_JSON[candidate["dataset"]],
        "annotation_root": str(annotation_root),
        "predictions_json": str(predictions_path),
        "input_text": candidate["prompt"],
        "gt_answer": format_gt_answers(candidate),
        "gt_object_ids": gt_ids,
        "gt_bbox_locs": gt_boxes,
        "gt_object_id": gt_ids[0] if len(gt_ids) == 1 else -1,
        "gt_bbox_loc": gt_boxes[0]["loc"] if len(gt_boxes) == 1 else None,
        "model_prediction": candidate["prediction"],
        "model_pred_object_ids": pred_ids,
        "model_pred_bbox_locs": pred_boxes,
        "correctness": correctness,
        "mesh_outputs": {
            "mesh_ply": str(mesh_path),
            "preview_png": "",
            "vertex_labels": "",
            "metadata_json": "",
        },
        "point_outputs": {},
    }
    if "semantic_review" in candidate:
        package["semantic_review"] = candidate["semantic_review"]
    return package


def render_tile(
    package_path: Path,
    output_png: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if output_png.is_file() and output_png.stat().st_size > 0 and not args.force:
        return {"report_png": str(output_png), "reused": True}
    environment = os.environ.copy()
    environment.setdefault(
        "MPLCONFIGDIR",
        str(Path(args.output_dir).resolve() / ".matplotlib-cache"),
    )
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
            args.renderer,
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--max-points",
            str(args.max_points),
            "--text-preset",
            args.text_preset,
            "--font-family",
            args.font_family,
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    start = result.stdout.find("{")
    return (
        json.loads(result.stdout[start:])
        if start >= 0
        else {"report_png": str(output_png), "reused": False}
    )


def stitch_grid(
    tile_paths: list[Path],
    output_png: Path,
    cols: int,
    rows: int,
) -> None:
    if len(tile_paths) != cols * rows:
        raise ValueError(f"Expected {cols * rows} tiles, got {len(tile_paths)}")
    images = [Image.open(path).convert("RGB") for path in tile_paths]
    try:
        tile_width, tile_height = images[0].size
        canvas = Image.new("RGB", (cols * tile_width, rows * tile_height), "white")
        for index, image in enumerate(images):
            if image.size != (tile_width, tile_height):
                image = image.resize(
                    (tile_width, tile_height),
                    Image.Resampling.LANCZOS,
                )
            canvas.paste(
                image,
                ((index % cols) * tile_width, (index // cols) * tile_height),
            )
        output_png.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_png)
        canvas.close()
    finally:
        for image in images:
            image.close()


def relative_href(path: Path, root: Path) -> str:
    return html.escape(str(path.resolve().relative_to(root.resolve())))


def correctness_label(record: dict[str, Any]) -> str:
    evidence = record["correctness"]
    metric = evidence.get("metric", "")
    if "iou" in evidence:
        return f"{metric}: {float(evidence['iou']):.3f}"
    if "f1" in evidence:
        return f"{metric}: {float(evidence['f1']):.3f}"
    if "em1" in evidence:
        return f"{metric}: {int(evidence['em1'])}"
    return str(metric)


def write_gallery(manifest: dict[str, Any], output_dir: Path) -> Path:
    navigation: list[str] = []
    sections: list[str] = []
    for model in manifest["models"]:
        model_id = model["model_id"]
        model_label = html.escape(model["model_label"])
        navigation.append(f'<a href="#{html.escape(model_id)}">{model_label}</a>')
        dataset_sections: list[str] = []
        for dataset in model["datasets"]:
            dataset_name = dataset["dataset"]
            grid_path = Path(dataset["grid_png"])
            cards: list[str] = []
            for record in dataset["records"]:
                tile_path = Path(record["tile_png"])
                package_path = Path(record["package_json"])
                cards.append(
                    f"""
                    <article class="card">
                      <a href="{relative_href(tile_path, output_dir)}">
                        <img loading="lazy" src="{relative_href(tile_path, output_dir)}"
                             alt="{html.escape(record['scene_id'])}">
                      </a>
                      <div class="card-copy">
                        <strong>{html.escape(record['scene_id'])}</strong>
                        <span>{html.escape(correctness_label(record))}</span>
                        <a href="{relative_href(package_path, output_dir)}">sample package</a>
                      </div>
                    </article>
                    """
                )
            dataset_sections.append(
                f"""
                <section class="dataset" id="{html.escape(model_id)}-{html.escape(dataset_name)}">
                  <h2>{html.escape(dataset_name)}</h2>
                  <a class="grid-link" href="{relative_href(grid_path, output_dir)}">
                    <img loading="lazy" src="{relative_href(grid_path, output_dir)}"
                         alt="{html.escape(model_label)} {html.escape(dataset_name)} grid">
                  </a>
                  <div class="cards">{''.join(cards)}</div>
                </section>
                """
            )
        sections.append(
            f"""
            <section class="model" id="{html.escape(model_id)}">
              <h1>{model_label}</h1>
              {''.join(dataset_sections)}
            </section>
            """
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>90% Pruned 4-MLLM Correct Samples</title>
  <style>
    :root {{ color-scheme: light; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: #151515; background: #f6f6f4; font-family: Arial, sans-serif; }}
    nav {{ position: sticky; top: 0; z-index: 5; display: flex; gap: 20px; padding: 14px 24px;
           background: rgba(255,255,255,.95); border-bottom: 1px solid #ddd; }}
    nav a {{ color: #245b82; text-decoration: none; font-weight: 700; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 28px; }}
    .model {{ margin-bottom: 80px; }}
    h1 {{ font-family: Georgia, serif; font-size: 38px; margin: 24px 0; }}
    h2 {{ font-size: 24px; margin: 42px 0 16px; text-transform: capitalize; }}
    .grid-link img {{ display: block; width: min(100%, 950px); height: auto; background: white; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
              gap: 18px; margin-top: 22px; }}
    .card {{ background: white; border: 1px solid #deded8; }}
    .card img {{ display: block; width: 100%; height: auto; }}
    .card-copy {{ display: grid; gap: 6px; padding: 12px; font-size: 13px; }}
    .card-copy span {{ color: #4b5a63; }}
    .card-copy a {{ color: #245b82; }}
  </style>
</head>
<body>
  <nav>{''.join(navigation)}</nav>
  <main>{''.join(sections)}</main>
</body>
</html>
"""
    output_path = output_dir / "index.html"
    output_path.write_text(page, encoding="utf-8")
    return output_path


def model_label(prediction_root: Path, model: str) -> str:
    manifest_path = prediction_root / model / "source_manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        label = normalize_text(manifest.get("model", ""))
        if label:
            return label
    return MODEL_LABELS.get(model, model)


def validate_args(args: argparse.Namespace) -> None:
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.cols <= 0 or args.rows <= 0:
        raise ValueError("--cols and --rows must be positive")
    if args.limit != args.cols * args.rows:
        raise ValueError("--limit must equal --cols * --rows")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")


def main() -> int:
    args = parse_args()
    validate_args(args)
    prediction_root = Path(args.prediction_root).resolve()
    annotation_root = Path(args.annotation_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    args.output_dir = str(output_dir)
    review_path = (
        Path(args.scan2cap_review_json).resolve()
        if args.scan2cap_review_json
        else output_dir / "scan2cap_review.json"
    )
    scene_sources = [Path(value).resolve() for value in args.scene_source]
    scene_catalog = find_scene_catalog(scene_sources)
    if not scene_catalog:
        raise RuntimeError("No usable local ScanNet scenes were found.")

    gt_attrs = torch.load(
        annotation_root / args.gt_attr_file,
        map_location="cpu",
        weights_only=False,
    )
    pred_attrs = torch.load(
        annotation_root / args.pred_attr_file,
        map_location="cpu",
        weights_only=False,
    )

    candidate_map: dict[str, dict[str, list[dict[str, Any]]]] = {}
    selection_stats: dict[str, dict[str, dict[str, int]]] = {}
    for model in args.models:
        MODEL_LABELS[model] = model_label(prediction_root, model)
        candidate_map[model] = {}
        selection_stats[model] = {}
        for dataset in args.datasets:
            prediction_path = prediction_root / model / f"{dataset}.json"
            annotation_path = annotation_root / DATASET_JSON[dataset]
            candidates, stats = build_candidates(
                model,
                dataset,
                load_json(prediction_path),
                load_json(annotation_path),
                scene_catalog,
                gt_attrs,
                pred_attrs,
            )
            candidate_map[model][dataset] = candidates
            selection_stats[model][dataset] = stats
            print(
                f"{model}/{dataset}: {len(candidates)} strict local candidates "
                f"across {stats['strict_local_unique_scenes']} scenes"
            )

    if args.prepare_scan2cap_review:
        if "scan2cap" not in args.datasets:
            raise ValueError("--prepare-scan2cap-review requires --datasets scan2cap")
        prepare_scan2cap_review(
            {
                model: candidate_map[model]["scan2cap"]
                for model in args.models
            },
            review_path,
            args.review_candidates_per_model,
            args.force,
        )
        print(json.dumps({"scan2cap_review": str(review_path)}, indent=2))
        return 0

    scan2cap_review = load_json(review_path) if "scan2cap" in args.datasets else None
    if "scan2cap" in args.datasets and scan2cap_review is None:
        raise FileNotFoundError(
            f"Scan2Cap semantic review is required: {review_path}. "
            "Run with --prepare-scan2cap-review first."
        )

    selected_map: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for model in args.models:
        selected_map[model] = {}
        for dataset in args.datasets:
            candidates = candidate_map[model][dataset]
            if dataset == "scan2cap":
                review_entries = scan2cap_review.get("models", {}).get(model, [])
                selected = approved_scan2cap_candidates(
                    candidates,
                    review_entries,
                    args.limit,
                )
            else:
                selected = select_candidates(candidates, args.limit)
            selected_map[model][dataset] = selected
            print(
                f"{model}/{dataset}: selected {len(selected)} samples from "
                f"{len({item['scene_id'] for item in selected})} scenes"
            )

    selection_audit = {
        "schema_version": 1,
        "prediction_root": str(prediction_root),
        "annotation_root": str(annotation_root),
        "scene_sources": [str(path) for path in scene_sources],
        "available_local_scenes": len(scene_catalog),
        "limit": args.limit,
        "cols": args.cols,
        "rows": args.rows,
        "selection_policy": "strict_correct_then_short_text_with_distinct_scenes",
        "stats": selection_stats,
        "selections": {
            model: {
                dataset: [
                    {
                        key: candidate[key]
                        for key in (
                            "candidate_id",
                            "scene_id",
                            "annotation_index",
                            "prediction_order",
                            "prompt",
                            "prediction",
                            "references",
                            "correctness",
                            "semantic_review",
                        )
                        if key in candidate
                    }
                    for candidate in selected_map[model][dataset]
                ]
                for dataset in args.datasets
            }
            for model in args.models
        },
    }
    write_json(output_dir / "selection_audit.json", selection_audit)
    if args.dry_run:
        print(
            json.dumps(
                {"selection_audit": str(output_dir / "selection_audit.json")},
                indent=2,
            )
        )
        return 0

    selected_scenes = sorted(
        {
            candidate["scene_id"]
            for model in args.models
            for dataset in args.datasets
            for candidate in selected_map[model][dataset]
        }
    )
    scene_meshes = {
        scene_id: find_or_build_mesh(
            scene_id,
            scene_catalog[scene_id],
            output_dir / "scene_meshes",
        )
        for scene_id in selected_scenes
    }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "prediction_root": str(prediction_root),
        "annotation_root": str(annotation_root),
        "output_dir": str(output_dir),
        "selection_audit": str(output_dir / "selection_audit.json"),
        "scan2cap_review": str(review_path) if "scan2cap" in args.datasets else None,
        "tile_size": [args.width, args.height],
        "grid": {"cols": args.cols, "rows": args.rows},
        "renderer": args.renderer,
        "text_preset": args.text_preset,
        "font_family": args.font_family,
        "models": [],
    }
    render_jobs: list[tuple[Path, Path, dict[str, Any]]] = []

    for model in args.models:
        model_manifest: dict[str, Any] = {
            "model_id": model,
            "model_label": MODEL_LABELS.get(model, model),
            "datasets": [],
        }
        for dataset in args.datasets:
            predictions_path = prediction_root / model / f"{dataset}.json"
            records: list[dict[str, Any]] = []
            for ordinal, candidate in enumerate(selected_map[model][dataset]):
                package_dir = (
                    output_dir
                    / "packages"
                    / model
                    / dataset
                    / (
                        f"{ordinal:02d}_{candidate['candidate_id']}"
                        f"__{candidate['scene_id']}"
                    )
                )
                package_path = package_dir / "sample_package.json"
                tile_path = (
                    output_dir
                    / "tiles"
                    / model
                    / dataset
                    / (
                        f"{ordinal:02d}_{candidate['candidate_id']}"
                        f"__{candidate['scene_id']}.png"
                    )
                )
                package = build_package(
                    candidate,
                    scene_meshes[candidate["scene_id"]],
                    annotation_root,
                    predictions_path,
                    gt_attrs,
                    pred_attrs,
                )
                write_json(package_path, package)
                record = {
                    "ordinal": ordinal,
                    "candidate_id": candidate["candidate_id"],
                    "scene_id": candidate["scene_id"],
                    "annotation_index": candidate["annotation_index"],
                    "prediction_order": candidate["prediction_order"],
                    "correctness": candidate["correctness"],
                    "semantic_review": candidate.get("semantic_review"),
                    "package_json": str(package_path),
                    "tile_png": str(tile_path),
                    "render_info": {},
                }
                records.append(record)
                render_jobs.append((package_path, tile_path, record))
            grid_path = (
                output_dir
                / "grids"
                / model
                / f"{dataset}_{args.limit}_grid.png"
            )
            model_manifest["datasets"].append(
                {
                    "dataset": dataset,
                    "records": records,
                    "grid_png": str(grid_path),
                }
            )
        manifest["models"].append(model_manifest)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(render_tile, package_path, tile_path, args): (
                tile_path,
                record,
            )
            for package_path, tile_path, record in render_jobs
        }
        completed = 0
        for future in as_completed(futures):
            tile_path, record = futures[future]
            record["render_info"] = future.result()
            completed += 1
            print(f"rendered {completed}/{len(render_jobs)}: {tile_path}")

    for model_manifest in manifest["models"]:
        for dataset_manifest in model_manifest["datasets"]:
            tile_paths = [
                Path(record["tile_png"])
                for record in dataset_manifest["records"]
            ]
            grid_path = Path(dataset_manifest["grid_png"])
            stitch_grid(tile_paths, grid_path, args.cols, args.rows)
            print(f"grid: {grid_path}")

    write_json(output_dir / "manifest.json", manifest)
    gallery_path = write_gallery(manifest, output_dir)
    summary = {
        "output_dir": str(output_dir),
        "manifest": str(output_dir / "manifest.json"),
        "gallery": str(gallery_path),
        "tiles": len(render_jobs),
        "grids": sum(
            len(model_manifest["datasets"])
            for model_manifest in manifest["models"]
        ),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
