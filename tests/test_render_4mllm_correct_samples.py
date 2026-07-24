from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
import torch


os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "vision3d-test-matplotlib"),
)

from layout.render_4mllm_correct_samples import (
    axis_aligned_iou,
    evaluate_prediction,
    grouped_unique_rows,
)


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


if __name__ == "__main__":
    unittest.main()
