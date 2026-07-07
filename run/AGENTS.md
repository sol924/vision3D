# run Agent Guide

## Purpose

`run/` contains short command wrappers for the most common vision3D workflows. Each
script resolves the repository root and then forwards all remaining arguments to the
matching Python module.

## Commands

- `bash run/setup.sh`: creates `.venv` and installs `requirements.txt`.
- `bash run/render_sample.sh ...`: calls `point/visualize_sample_objects.py` to render a
  single point-cloud sample with token/GT highlighting.
- `bash run/render_tokens_mesh.sh ...`: calls `point/visualize_input_tokens_mesh.py` to
  render the input-token object mesh.
- `bash run/render_package.sh ...`: validates a sample against prediction JSON and
  writes a complete point/mesh visualization package plus unified metadata.
- `bash run/open_point.sh <ply> ...`: calls `layout/open_local_pointcloud.py` to create
  or open an interactive point-cloud HTML file.
- `bash run/open_mesh.sh <ply> ...`: calls `layout/open_local_mesh.py` to create or open
  an interactive mesh HTML file, with bbox overlays when metadata is available.
- `bash run/render_candidates.sh ...`: calls `layout/render_candidate_bbox_samples.py`
  to batch-render selected dataset indices.
- `bash run/render_report.sh <sample_package.json> ...`: renders a static PNG report
  with text on the left and the scene visualization on the right.

## Typical Usage

```bash
bash run/render_sample.sh --annotation-root /path/to/annotations --scene-root /path/to/scannet --sample-index 0
bash run/render_tokens_mesh.sh --annotation-root /path/to/annotations --scene-root /path/to/scannet --sample-index 0
bash run/open_mesh.sh outputs/visualizations_mesh/example.ply --annotation-root /path/to/annotations --no-open
bash run/render_report.sh outputs/sample_packages/scene0011_00_sample00000/sample_package.json
```

Set `PYTHON=/path/to/python` to use a specific interpreter.
