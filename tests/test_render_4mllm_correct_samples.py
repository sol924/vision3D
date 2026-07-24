from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw
from pypdf import PdfReader


os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "vision3d-test-matplotlib"),
)

from layout.render_4mllm_correct_samples import (
    axis_aligned_iou,
    evaluate_prediction,
    grouped_unique_rows,
    write_grid_pdf,
    write_grid_preview,
)
from layout.render_sample_report import build_style, render_report


def attributes(locs: list[list[float]]) -> dict:
    return {
        "scene0000_00": {
            "locs": torch.tensor(locs, dtype=torch.float32),
        }
    }


class CorrectSampleSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gt_attrs = attributes(
            [
                [0, 0, 0, 2, 2, 2],
                [5, 0, 0, 2, 2, 2],
            ]
        )
        self.pred_attrs = attributes(
            [
                [5, 0, 0, 2, 2, 2],
                [0, 0, 0, 2, 2, 2],
                [20, 0, 0, 2, 2, 2],
            ]
        )

    def test_axis_aligned_iou(self) -> None:
        box = np.asarray([0, 0, 0, 2, 2, 2], dtype=np.float64)
        shifted = np.asarray([1, 0, 0, 2, 2, 2], dtype=np.float64)
        far = np.asarray([10, 0, 0, 2, 2, 2], dtype=np.float64)
        self.assertAlmostEqual(axis_aligned_iou(box, box), 1.0)
        self.assertAlmostEqual(axis_aligned_iou(box, shifted), 1.0 / 3.0)
        self.assertEqual(axis_aligned_iou(box, far), 0.0)

    def test_scanrefer_uses_bbox_iou_not_numeric_id(self) -> None:
        result = evaluate_prediction(
            "scanrefer",
            {
                "scene_id": "scene0000_00",
                "gt_id": 0,
                "pred": "<OBJ001>.",
            },
            self.gt_attrs,
            self.pred_attrs,
        )
        self.assertTrue(result["correct"])
        self.assertEqual(result["pred_object_ids"], [1])
        self.assertAlmostEqual(result["iou"], 1.0)

    def test_multi3dref_hungarian_full_match(self) -> None:
        result = evaluate_prediction(
            "multi3dref",
            {
                "scene_id": "scene0000_00",
                "ref_captions": [0, 1],
                "pred": "<OBJ000>, <OBJ001>.",
            },
            self.gt_attrs,
            self.pred_attrs,
        )
        self.assertTrue(result["correct"])
        self.assertAlmostEqual(result["f1"], 1.0)
        self.assertEqual(len(result["matches"]), 2)

    def test_multi3dref_rejects_partial_match(self) -> None:
        result = evaluate_prediction(
            "multi3dref",
            {
                "scene_id": "scene0000_00",
                "ref_captions": [0, 1],
                "pred": "<OBJ001>.",
            },
            self.gt_attrs,
            self.pred_attrs,
        )
        self.assertFalse(result["correct"])
        self.assertLess(result["f1"], 1.0)

    def test_qa_uses_official_cleaned_exact_match(self) -> None:
        exact = evaluate_prediction(
            "scanqa",
            {
                "scene_id": "scene0000_00",
                "gt_id": 0,
                "pred": "2.",
                "ref_captions": ["two"],
            },
            self.gt_attrs,
            self.pred_attrs,
        )
        expanded = evaluate_prediction(
            "scanqa",
            {
                "scene_id": "scene0000_00",
                "gt_id": 0,
                "pred": "There are two.",
                "ref_captions": ["two"],
            },
            self.gt_attrs,
            self.pred_attrs,
        )
        self.assertTrue(exact["correct"])
        self.assertFalse(expanded["correct"])

    def test_duplicate_scene_prompt_groups_are_excluded(self) -> None:
        rows = [
            {"scene_id": "scene0000_00", "prompt": "same", "pred": "a"},
            {"scene_id": "scene0000_00", "prompt": "same", "pred": "b"},
            {"scene_id": "scene0000_00", "prompt": "unique", "pred": "c"},
        ]
        unique, duplicate_records = grouped_unique_rows(rows)
        self.assertEqual(len(unique), 1)
        self.assertEqual(duplicate_records, 2)


class LightweightOutputTests(unittest.TestCase):
    def sample_package(self) -> dict:
        return {
            "input_text": "What color is the chair?",
            "gt_answer": ["brown"],
            "model_prediction": "Brown.",
            "gt_bbox_locs": [],
            "model_pred_bbox_locs": [],
        }

    def test_scene_compression_keeps_text_pixels_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            package = self.sample_package()
            style = build_style("compact", "times")
            full_scene = Image.new("RGB", (1168, 1020), "white")
            draw = ImageDraw.Draw(full_scene)
            draw.rectangle((180, 160, 960, 860), fill=(116, 92, 64))
            small_scene = full_scene.resize((584, 510), Image.Resampling.LANCZOS)
            full_path = root / "full.png"
            compact_path = root / "compact.png"
            render_report(
                package,
                full_scene,
                full_path,
                (1800, 1100),
                {},
                "matplotlib",
                "mesh",
                style,
                scene_colors=0,
            )
            render_report(
                package,
                small_scene,
                compact_path,
                (1800, 1100),
                {},
                "matplotlib",
                "mesh",
                style,
                scene_colors=64,
            )
            with Image.open(full_path) as full, Image.open(compact_path) as compact:
                left_box = (0, 0, style.right_x, 1100)
                difference = ImageChops.difference(
                    full.crop(left_box),
                    compact.crop(left_box),
                )
                self.assertIsNone(difference.getbbox())

    def test_grid_preview_obeys_hard_byte_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            tile_path = root / "tile.png"
            tile = Image.new("RGB", (320, 150), "white")
            draw = ImageDraw.Draw(tile)
            draw.text((10, 20), "Instruction GT PREDICTION", fill="black")
            draw.rectangle((170, 20, 310, 140), fill=(118, 91, 62))
            tile.save(tile_path)
            output_path = root / "grid.png"
            info = write_grid_preview(
                [tile_path] * 10,
                output_path,
                cols=2,
                rows=5,
                preferred_width=1000,
                max_bytes=100_000,
            )
            self.assertLessEqual(info["bytes"], 100_000)
            self.assertEqual(output_path.stat().st_size, info["bytes"])

    def test_grid_pdf_contains_vector_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            package_path = root / "sample_package.json"
            package_path.write_text(
                json.dumps(self.sample_package()),
                encoding="utf-8",
            )
            tile_path = root / "tile.png"
            tile = Image.new("RGB", (1800, 1100), "white")
            draw = ImageDraw.Draw(tile)
            draw.rectangle((590, 40, 1758, 1060), fill=(118, 91, 62))
            tile.save(tile_path)
            records = [
                {
                    "package_json": str(package_path),
                    "tile_png": str(tile_path),
                }
                for _ in range(10)
            ]
            output_path = root / "grid.pdf"
            info = write_grid_pdf(
                records,
                output_path,
                cols=2,
                rows=5,
                tile_size=(1800, 1100),
                text_preset="compact",
                font_family="times",
                scene_scale=0.5,
                scene_colors=64,
                page_width_inches=7.0,
                title="Test grid",
            )
            reader = PdfReader(output_path)
            self.assertEqual(len(reader.pages), 1)
            text = reader.pages[0].extract_text()
            self.assertIn("Instruction", text)
            self.assertIn("PREDICTION", text)
            self.assertTrue(info["vector_text"])
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
