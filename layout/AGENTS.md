# layout Agent Guide

## Purpose

`layout/` owns presentation of artifacts produced by `point/`. It converts PLY outputs
into static previews and interactive Plotly HTML viewers, and can overlay GT/context
bounding boxes when metadata and annotation files are available.

## Main Scripts

- `open_local_pointcloud.py`: opens an ASCII point-cloud PLY as a Plotly scatter viewer.
- `open_local_mesh.py`: opens an ASCII mesh PLY as a Plotly mesh viewer and optionally
  overlays GT/context bbox traces.
- `render_candidate_bbox_samples.py`: renders a small batch of candidate ScanRefer
  samples with original-color meshes, bbox previews, HTML viewers, and a manifest.

## Metadata Contract

Interactive bbox overlays work best when the PLY has a sibling JSON file containing:

- `scene_id`
- `sample_index`
- `dataset_json`
- `scene_root`
- `prompt`

The `point/` scripts and `render_candidate_bbox_samples.py` write this metadata.

## Path Rules

Pass `--annotation-root` explicitly when possible. If omitted, `open_local_mesh.py`
searches `VISION3D_ANNO_ROOT`, `FAST3D_ANNO_ROOT`, `annotations/`, and
`datasets/annotations/` from the current repository ancestry.
