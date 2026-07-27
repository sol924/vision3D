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

To render the five `ours/0625_best_02` datasets into paper-style 2x6 grids using
correctly predicted samples:

```bash
PYTHON=/Users/sol/Research/Training_Free_Token_Redcution/.venv/bin/python \
  bash run/render_ours_dataset_grids.sh
```

The default output directory is
`/Volumes/T7 Shield/point_reduction_data/ours/visualization_0625_best_02`.
This command uses the `a4-2x6` text preset and Times-style font by default so each
dataset grid can be placed on an A4 paper page more clearly. The default tile size is
`3200x1500`, producing `6400x9000` grids with complete text and one-line bbox legends.

To preserve those five grids while producing paper-friendly indexed PNG copies whose
combined size does not exceed 5,000,000 bytes:

```bash
/Users/sol/Research/Training_Free_Token_Redcution/.venv/bin/python \
  layout/compact_dataset_grids.py \
  --manifest "/Volumes/T7 Shield/point_reduction_data/ours/visualization_0625_best_02/manifest.json"
```

The compact copies are written to `grids_compact/`. The default settings keep the
original `6400x9000` canvas and text geometry, render each scene panel at 25% scale
with 32 colors, and store the final grid as a 256-color indexed PNG.

To render four MLLMs independently with ten strictly correct local samples per
dataset, first create and complete the Scan2Cap semantic-review file:

```bash
PYTHON=/path/to/python bash run/render_4mllm_correct_samples.sh \
  --prediction-root /path/to/90pct_4mllm_predictions \
  --annotation-root /path/to/annotations \
  --scene-source /path/to/scannet_samples \
  --prepare-scan2cap-review
```

Mark ten semantically correct, distinct-scene Scan2Cap candidates per model as
`approved` in `outputs/90pct_4mllm_correct_samples/scan2cap_review.json`, including
the matching reference index and a short review reason. Then render:

```bash
PYTHON=/path/to/python bash run/render_4mllm_correct_samples.sh \
  --prediction-root /path/to/90pct_4mllm_predictions \
  --annotation-root /path/to/annotations \
  --scene-source /path/to/scannet_samples
```

The command applies the strict Free3D correctness definitions: ScanRefer
`Acc@0.50`, Multi3DRef `F1@0.50=1`, cleaned exact-match QA/SQA answers, and
Scan2Cap bbox IoU at least 0.5 plus semantic review. It writes 200 individual
reports, twenty 2x5 grids, a selection audit, and an HTML gallery.

The four-model workflow defaults to paper-friendly lightweight output. The 3D
scene is rendered at 50% of the right-panel width and height while the text
canvas remains at full resolution. Each 2x5 grid has a selectable-vector-text
PDF under `pdf/` and a PNG preview under `grids/` that is guaranteed not to
exceed 1 MiB. Override these defaults with `--scene-scale`,
`--scene-colors`, `--grid-preview-width`, or
`--grid-preview-max-bytes`.
