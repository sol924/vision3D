from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-render sample packages, paper-style PNG reports, and an HTML gallery."
    )
    parser.add_argument("--annotation-root", required=True, help="Annotation root directory.")
    parser.add_argument("--scene-root", required=True, help="Directory containing scene folders.")
    parser.add_argument("--predictions-json", required=True, help="Model prediction JSON file.")
    parser.add_argument(
        "--dataset-json",
        default="scanrefer_mask3d_val.json",
        help="Annotation JSON under annotation root.",
    )
    parser.add_argument(
        "--sample-indices",
        nargs="*",
        type=int,
        default=None,
        help="Specific sample indices to render. If omitted, valid samples are selected automatically.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="First dataset index for auto selection.")
    parser.add_argument("--limit", type=int, default=8, help="Number of samples for auto selection.")
    parser.add_argument("--output-dir", default="outputs/batch_gallery", help="Batch output directory.")
    parser.add_argument("--pred-attr-file", default="scannet_mask3d_val_attributes.pt")
    parser.add_argument("--gt-attr-file", default="scannet_val_attributes.pt")
    parser.add_argument("--max-mesh-objects", type=int, default=30)
    parser.add_argument("--bbox-scale", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1800, help="Report PNG width.")
    parser.add_argument("--height", type=int, default=1100, help="Report PNG height.")
    parser.add_argument("--max-points", type=int, default=80000, help="Maximum rendered points/faces hint.")
    parser.add_argument(
        "--renderer",
        choices=("auto", "open3d", "matplotlib"),
        default="auto",
        help="Renderer passed to render_report.sh.",
    )
    parser.add_argument(
        "--with-interactive-html",
        action="store_true",
        help="Also generate per-sample point/mesh interactive HTML viewers.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def extract_json(stdout: str) -> dict:
    start = stdout.find("{")
    if start < 0:
        raise RuntimeError(f"Command did not emit JSON:\n{stdout}")
    return json.loads(stdout[start:])


def run_json(cmd: list[str]) -> dict:
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    return extract_json(result.stdout)


def rel(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def select_sample_indices(
    dataset: list[dict],
    predictions: list[dict],
    scene_root: Path,
    explicit_indices: list[int] | None,
    start_index: int,
    limit: int,
) -> list[int]:
    if explicit_indices:
        return explicit_indices

    candidates: list[int] = []
    for index in range(max(start_index, 0), len(dataset)):
        sample = dataset[index]
        scene_id = sample.get("scene_id")
        if index >= len(predictions):
            continue
        if predictions[index].get("scene_id") != scene_id:
            continue
        scene_dir = scene_root / str(scene_id)
        if not (scene_dir / f"{scene_id}_vh_clean_2.ply").exists():
            continue
        candidates.append(index)

    selected: list[int] = []
    seen_scenes: set[str] = set()
    for index in candidates:
        scene_id = str(dataset[index].get("scene_id"))
        if scene_id in seen_scenes:
            continue
        selected.append(index)
        seen_scenes.add(scene_id)
        if len(selected) >= limit:
            break
    for index in candidates:
        if len(selected) >= limit:
            break
        if index not in selected:
            selected.append(index)
    return selected


def write_gallery(records: list[dict], output_dir: Path) -> Path:
    def gallery_href(path_str: str) -> str:
        path = resolve_path(path_str)
        try:
            return html.escape(str(path.relative_to(output_dir)))
        except ValueError:
            return html.escape(str(path))

    rows = []
    for record in records:
        report = gallery_href(record["report_png"])
        package = gallery_href(record["package_metadata"])
        scene_id = html.escape(record["scene_id"])
        prompt = html.escape(record["input_text"])
        gt = html.escape(", ".join(record.get("gt_answer", [])))
        pred = html.escape(record.get("model_prediction", ""))
        rows.append(
            f"""
            <article class="sample">
              <a href="{report}"><img src="{report}" alt="{scene_id} sample {record['sample_index']} report"></a>
              <div class="meta">
                <h2>{scene_id} / sample {record['sample_index']:05d}</h2>
                <p><strong>Instruction</strong> {prompt}</p>
                <p><strong>GT</strong> {gt}</p>
                <p><strong>Prediction</strong> {pred}</p>
                <p><a href="{package}">sample_package.json</a></p>
              </div>
            </article>
            """
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>vision3D Batch Gallery</title>
  <style>
    body {{ margin: 0; background: #fff; color: #1d252c; font-family: Arial, sans-serif; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 32px; }}
    h1 {{ font-size: 28px; margin: 0 0 24px; }}
    .sample {{ display: grid; grid-template-columns: minmax(520px, 1fr) 420px; gap: 28px; align-items: center; padding: 28px 0; border-top: 1px solid #e5e5e5; }}
    .sample:first-of-type {{ border-top: 0; }}
    img {{ width: 100%; height: auto; display: block; }}
    h2 {{ font-size: 18px; margin: 0 0 14px; }}
    p {{ font-size: 14px; line-height: 1.45; margin: 0 0 10px; }}
    strong {{ display: block; font-size: 12px; color: #66717c; text-transform: uppercase; margin-bottom: 2px; }}
    a {{ color: #255f8f; }}
    @media (max-width: 980px) {{ .sample {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <h1>vision3D Batch Gallery</h1>
    {''.join(rows)}
  </main>
</body>
</html>
"""
    gallery_path = output_dir / "index.html"
    gallery_path.write_text(html_text, encoding="utf-8")
    return gallery_path


def main() -> int:
    args = parse_args()
    annotation_root = resolve_path(args.annotation_root)
    scene_root = resolve_path(args.scene_root)
    predictions_json = resolve_path(args.predictions_json)
    output_dir = resolve_path(args.output_dir)
    package_root = output_dir / "packages"
    report_root = output_dir / "reports"
    package_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    dataset = load_json(annotation_root / args.dataset_json)
    predictions = load_json(predictions_json)
    sample_indices = select_sample_indices(
        dataset,
        predictions,
        scene_root,
        args.sample_indices,
        args.start_index,
        args.limit,
    )
    if not sample_indices:
        raise RuntimeError("No renderable sample indices found.")

    records: list[dict] = []
    for sample_index in sample_indices:
        package_cmd = [
            sys.executable,
            str(REPO_ROOT / "point" / "render_sample_package.py"),
            "--annotation-root",
            str(annotation_root),
            "--scene-root",
            str(scene_root),
            "--predictions-json",
            str(predictions_json),
            "--dataset-json",
            args.dataset_json,
            "--sample-index",
            str(sample_index),
            "--pred-attr-file",
            args.pred_attr_file,
            "--gt-attr-file",
            args.gt_attr_file,
            "--output-dir",
            str(package_root),
            "--max-mesh-objects",
            str(args.max_mesh_objects),
            "--bbox-scale",
            str(args.bbox_scale),
        ]
        if not args.with_interactive_html:
            package_cmd.append("--no-html")
        package_info = run_json(package_cmd)
        package_metadata = resolve_path(package_info["package_metadata"])
        report_png = report_root / f"{package_info['scene_id']}_sample{sample_index:05d}_report.png"
        report_info = run_json(
            [
                sys.executable,
                str(REPO_ROOT / "layout" / "render_sample_report.py"),
                str(package_metadata),
                "--output-png",
                str(report_png),
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
            ]
        )
        records.append(
            {
                "sample_index": sample_index,
                "scene_id": package_info["scene_id"],
                "input_text": package_info.get("input_text", ""),
                "gt_answer": package_info.get("gt_answer", []),
                "model_prediction": package_info.get("model_prediction", ""),
                "package_metadata": rel(package_metadata),
                "report_png": rel(resolve_path(report_info["report_png"])),
            }
        )
        print(json.dumps(records[-1], ensure_ascii=False))

    gallery_path = write_gallery(records, output_dir)
    summary = {
        "gallery_html": rel(gallery_path),
        "sample_count": len(records),
        "sample_indices": sample_indices,
        "reports": [record["report_png"] for record in records],
    }
    (output_dir / "batch_manifest.json").write_text(
        json.dumps(summary | {"records": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
