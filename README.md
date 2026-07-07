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

To batch-render multiple samples and open a browsable gallery:

```bash
bash run/render_batch_gallery.sh --annotation-root /path/to/annotations --scene-root /path/to/scannet --predictions-json /path/to/preds.json --sample-indices 0 6002 6816 --output-dir outputs/batch_gallery
open outputs/batch_gallery/index.html
```

To render the five `ours/0625_best_02` datasets into paper-style 3x7 grids:

```bash
PYTHON=/Users/sol/Research/Training_Free_Token_Redcution/.venv/bin/python \
  bash run/render_ours_dataset_grids.sh
```

The default output directory is
`/Volumes/T7 Shield/point_reduction_data/ours/visualization_0625_best_02`.
This command uses the `a4-grid` text preset and Times-style font by default so each
dataset grid can be placed on an A4 paper page more clearly.
