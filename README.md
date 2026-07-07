# vision3D

vision3D is a small toolkit for 3D point-cloud and dataset visualization. It organizes
ScanNet-style scenes, prompt metadata, ground-truth objects, and model answers into
renderable point-cloud or mesh artifacts, then opens them as static previews or
interactive HTML viewers.

## Directories

- `point/`: data loading and point/mesh artifact generation.
- `layout/`: static previews, bbox overlays, and interactive Plotly viewers.
- `run/`: short command wrappers for setup, rendering, and opening results.

## Quick Start

```bash
bash run/setup.sh
bash run/render_sample.sh --annotation-root /path/to/annotations --scene-root /path/to/scannet --sample-index 0
bash run/open_point.sh outputs/visualizations/scene0011_00_sample00000_token_token014.ply --no-open
```

For mesh-token visualization:

```bash
bash run/render_tokens_mesh.sh --annotation-root /path/to/annotations --scene-root /path/to/scannet --sample-index 0
bash run/open_mesh.sh outputs/visualizations_mesh/example.ply --no-open
```

To build a complete sample package with GT and model prediction metadata:

```bash
bash run/render_package.sh --annotation-root /path/to/annotations --scene-root /path/to/scannet --predictions-json /path/to/preds.json --sample-index 0
```

To render a paper-style static report with text on the left and mesh visualization on the right:

```bash
bash run/render_report.sh outputs/sample_packages/scene0011_00_sample00000/sample_package.json --scene-mode mesh
```
