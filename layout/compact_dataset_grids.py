from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layout.render_sample_report import build_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create paper-friendly indexed PNG grids by reducing only the scene "
            "panels while preserving the original grid dimensions and text layout."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the render_ours_dataset_grids.py manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <manifest-dir>/grids_compact.",
    )
    parser.add_argument(
        "--scene-scale",
        type=float,
        default=0.25,
        help="Scene-panel downsampling factor before nearest-neighbor enlargement.",
    )
    parser.add_argument(
        "--scene-colors",
        type=int,
        default=32,
        help="Palette size used independently for each downsampled scene panel.",
    )
    parser.add_argument(
        "--final-colors",
        type=int,
        default=256,
        help="Palette size for each final indexed PNG.",
    )
    parser.add_argument(
        "--max-total-bytes",
        type=int,
        default=5_000_000,
        help="Hard byte limit for the sum of all output PNG files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset subset. Defaults to every dataset in the manifest.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace compact files with matching names in the output directory.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def validate_palette_size(value: int, label: str) -> None:
    if not 2 <= value <= 256:
        raise ValueError(f"{label} must be between 2 and 256, got {value}")


def scene_panel_box(
    tile_size: tuple[int, int],
    text_preset: str,
    font_family: str,
) -> tuple[int, int, int, int]:
    tile_width, tile_height = tile_size
    style = build_style(text_preset, font_family)
    box = (
        style.right_x,
        style.right_y,
        tile_width - style.right_margin,
        tile_height - style.bottom_margin,
    )
    left, top, right, bottom = box
    if left < 0 or top < 0 or right > tile_width or bottom > tile_height:
        raise ValueError(f"Scene panel {box} falls outside tile size {tile_size}")
    if right <= left or bottom <= top:
        raise ValueError(f"Scene panel {box} is empty for tile size {tile_size}")
    return box


def compact_grid(
    source_png: Path,
    output_png: Path,
    cols: int,
    rows: int,
    tile_size: tuple[int, int],
    scene_box: tuple[int, int, int, int],
    scene_scale: float,
    scene_colors: int,
    final_colors: int,
) -> dict[str, Any]:
    tile_width, tile_height = tile_size
    expected_size = (cols * tile_width, rows * tile_height)
    with Image.open(source_png) as source:
        grid = source.convert("RGB")
    if grid.size != expected_size:
        grid.close()
        raise ValueError(
            f"{source_png} has size {grid.size}, expected {expected_size}"
        )

    scene_left, scene_top, scene_right, scene_bottom = scene_box
    scene_width = scene_right - scene_left
    scene_height = scene_bottom - scene_top
    render_size = (
        max(1, round(scene_width * scene_scale)),
        max(1, round(scene_height * scene_scale)),
    )

    try:
        for row in range(rows):
            for col in range(cols):
                tile_x = col * tile_width
                tile_y = row * tile_height
                box = (
                    tile_x + scene_left,
                    tile_y + scene_top,
                    tile_x + scene_right,
                    tile_y + scene_bottom,
                )
                scene = grid.crop(box)
                reduced = scene.resize(render_size, Image.Resampling.LANCZOS)
                scene.close()
                indexed_scene = reduced.quantize(
                    colors=scene_colors,
                    method=Image.Quantize.MEDIANCUT,
                    dither=Image.Dither.NONE,
                )
                reduced.close()
                expanded = indexed_scene.convert("RGB").resize(
                    (scene_width, scene_height),
                    Image.Resampling.NEAREST,
                )
                indexed_scene.close()
                grid.paste(expanded, box[:2])
                expanded.close()

        indexed_grid = grid.quantize(
            colors=final_colors,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        try:
            indexed_grid.save(
                output_png,
                format="PNG",
                optimize=True,
                compress_level=9,
            )
        finally:
            indexed_grid.close()
    finally:
        grid.close()

    return {
        "width": expected_size[0],
        "height": expected_size[1],
        "mode": "P",
        "scene_panel": list(scene_box),
        "scene_render_size": list(render_size),
        "bytes": output_png.stat().st_size,
        "sha256": sha256_file(output_png),
    }


def compact_dataset_grids(
    manifest_path: Path,
    output_dir: Path | None = None,
    scene_scale: float = 0.25,
    scene_colors: int = 32,
    final_colors: int = 256,
    max_total_bytes: int = 5_000_000,
    datasets: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = load_json(manifest_path)
    output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else manifest_path.parent / "grids_compact"
    )
    if not 0 < scene_scale <= 1:
        raise ValueError("--scene-scale must be greater than 0 and at most 1")
    validate_palette_size(scene_colors, "--scene-colors")
    validate_palette_size(final_colors, "--final-colors")
    if max_total_bytes <= 0:
        raise ValueError("--max-total-bytes must be positive")

    manifest_datasets = manifest.get("datasets")
    if not isinstance(manifest_datasets, dict) or not manifest_datasets:
        raise ValueError(f"No datasets found in {manifest_path}")
    selected_datasets = datasets or list(manifest_datasets)
    unknown = [name for name in selected_datasets if name not in manifest_datasets]
    if unknown:
        raise ValueError(f"Datasets not found in manifest: {', '.join(unknown)}")

    cols = int(manifest["cols"])
    rows = int(manifest["rows"])
    tile_size = tuple(int(item) for item in manifest["tile_size"])
    if len(tile_size) != 2 or min(tile_size) <= 0:
        raise ValueError(f"Invalid tile_size in {manifest_path}: {tile_size}")
    text_preset = str(manifest["text_preset"])
    font_family = str(manifest.get("font_family", "times"))
    scene_box = scene_panel_box(tile_size, text_preset, font_family)

    sources: dict[str, Path] = {}
    output_names: set[str] = set()
    for dataset_name in selected_datasets:
        source_value = manifest_datasets[dataset_name].get("grid_png")
        if not source_value:
            raise ValueError(f"{dataset_name} has no grid_png in {manifest_path}")
        source = resolve_manifest_path(manifest_path, str(source_value))
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.name in output_names:
            raise ValueError(f"Duplicate output grid name: {source.name}")
        output_names.add(source.name)
        destination = output_dir / source.name
        if source == destination:
            raise ValueError("Compact output must not overwrite its source grid")
        sources[dataset_name] = source

    target_paths = [output_dir / source.name for source in sources.values()]
    target_paths.append(output_dir / "compact_manifest.json")
    if not overwrite:
        existing = [path for path in target_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Compact outputs already exist; pass --overwrite to replace them: "
                + ", ".join(str(path) for path in existing)
            )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".vision3d-grids-compact-",
        dir=output_dir.parent,
    ) as temporary_dir:
        staging_dir = Path(temporary_dir)
        dataset_results: dict[str, Any] = {}
        total_png_bytes = 0
        for dataset_name, source in sources.items():
            staged_png = staging_dir / source.name
            result = compact_grid(
                source,
                staged_png,
                cols,
                rows,
                tile_size,
                scene_box,
                scene_scale,
                scene_colors,
                final_colors,
            )
            total_png_bytes += int(result["bytes"])
            dataset_results[dataset_name] = {
                "source_grid": str(source),
                "source_sha256": sha256_file(source),
                "output_grid": str(output_dir / source.name),
                **result,
            }

        if total_png_bytes > max_total_bytes:
            raise RuntimeError(
                f"Compact PNG total is {total_png_bytes} bytes, exceeding "
                f"the {max_total_bytes}-byte limit"
            )

        compact_manifest = {
            "source_manifest": str(manifest_path),
            "output_dir": str(output_dir),
            "cols": cols,
            "rows": rows,
            "tile_size": list(tile_size),
            "text_preset": text_preset,
            "font_family": font_family,
            "scene_scale": scene_scale,
            "scene_colors": scene_colors,
            "final_colors": final_colors,
            "max_total_bytes": max_total_bytes,
            "total_png_bytes": total_png_bytes,
            "datasets": dataset_results,
        }
        staged_manifest = staging_dir / "compact_manifest.json"
        write_json(staged_manifest, compact_manifest)

        output_dir.mkdir(parents=True, exist_ok=True)
        for source in sources.values():
            os.replace(staging_dir / source.name, output_dir / source.name)
        os.replace(staged_manifest, output_dir / "compact_manifest.json")

    return compact_manifest


def main() -> int:
    args = parse_args()
    result = compact_dataset_grids(
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        scene_scale=args.scene_scale,
        scene_colors=args.scene_colors,
        final_colors=args.final_colors,
        max_total_bytes=args.max_total_bytes,
        datasets=args.datasets,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
