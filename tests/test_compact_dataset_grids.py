from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw


os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "vision3d-test-matplotlib"),
)

from layout.compact_dataset_grids import compact_dataset_grids
from layout.render_sample_report import build_style


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CompactDatasetGridTests(unittest.TestCase):
    def build_fixture(self, root: Path) -> tuple[Path, Path]:
        tile_size = (700, 300)
        cols = 2
        rows = 1
        style = build_style("compact", "times")
        source_path = root / "scanrefer_2_grid.png"
        grid = Image.new(
            "RGB",
            (tile_size[0] * cols, tile_size[1] * rows),
            "white",
        )
        draw = ImageDraw.Draw(grid)
        for col in range(cols):
            tile_x = col * tile_size[0]
            draw.text(
                (tile_x + 24, 60),
                "Instruction\nGT\nPREDICTION",
                fill="black",
            )
            scene_box = (
                tile_x + style.right_x,
                style.right_y,
                tile_x + tile_size[0] - style.right_margin,
                tile_size[1] - style.bottom_margin,
            )
            draw.rectangle(scene_box, fill=(116 + col * 30, 92, 64))
            draw.line(
                (
                    scene_box[0],
                    scene_box[1],
                    scene_box[2],
                    scene_box[3],
                ),
                fill=(226, 85, 47),
                width=4,
            )
        grid.save(source_path)
        grid.close()

        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "cols": cols,
                    "rows": rows,
                    "tile_size": list(tile_size),
                    "text_preset": "compact",
                    "font_family": "times",
                    "datasets": {
                        "scanrefer": {
                            "grid_png": str(source_path),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return manifest_path, source_path

    def test_compacts_scene_without_changing_grid_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            manifest_path, source_path = self.build_fixture(root)
            source_hash = sha256_file(source_path)
            output_dir = root / "grids_compact"

            summary = compact_dataset_grids(
                manifest_path,
                output_dir=output_dir,
                scene_scale=0.25,
                scene_colors=32,
                final_colors=256,
                max_total_bytes=1_000_000,
            )

            output_path = output_dir / source_path.name
            self.assertEqual(sha256_file(source_path), source_hash)
            self.assertEqual(
                summary["total_png_bytes"],
                output_path.stat().st_size,
            )
            self.assertLessEqual(summary["total_png_bytes"], 1_000_000)
            self.assertTrue((output_dir / "compact_manifest.json").is_file())

            with Image.open(source_path) as source, Image.open(output_path) as compact:
                self.assertEqual(compact.size, source.size)
                self.assertEqual(compact.mode, "P")
                left_text_box = (0, 0, 590, 300)
                difference = ImageChops.difference(
                    source.convert("RGB").crop(left_text_box),
                    compact.convert("RGB").crop(left_text_box),
                )
                self.assertIsNone(difference.getbbox())

    def test_hard_limit_failure_publishes_no_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            manifest_path, _source_path = self.build_fixture(root)
            output_dir = root / "grids_compact"

            with self.assertRaisesRegex(RuntimeError, "exceeding"):
                compact_dataset_grids(
                    manifest_path,
                    output_dir=output_dir,
                    max_total_bytes=1,
                )

            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
