# point Agent Guide

## Purpose

`point/` owns the data-facing part of vision3D. It reads dataset annotations, scene
geometry, prompt text, GT object ids, and predicted object attributes, then exports
visualization-ready artifacts.

## Main Scripts

- `visualize_sample_objects.py`: builds an aligned point-cloud PLY for one annotation
  sample, highlighting token objects, the GT object, or both. It also writes a PNG
  preview and JSON metadata containing prompt, GT id, token ids, and output paths.
- `visualize_input_tokens_mesh.py`: builds a ScanNet mesh with preserved faces. Use
  `--color-mode none --background-mode original` for paper reports so objects keep
  their original scene colors; use `palette` or `uniform_red` only for debugging token
  coverage. It writes mesh PLY, PNG preview, vertex labels, and JSON metadata.
- `render_sample_package.py`: validates one annotation sample against model predictions
  and writes a complete point/mesh package with input text, GT answer, model prediction,
  bbox metadata, previews, and optional HTML viewers.

## Inputs

- `--annotation-root`: directory containing annotation JSON files and attribute `.pt`
  files.
- `--scene-root`: directory containing scene subdirectories such as `scene0011_00/`.
- `--dataset-json`: annotation JSON, default `scanrefer_mask3d_val.json`.
- `--pred-attr-file`: predicted object attributes, default
  `scannet_mask3d_val_attributes.pt`.
- `--gt-attr-file`: GT object attributes, default `scannet_val_attributes.pt`.
- `--predictions-json`: model predictions. For ScanRefer-style outputs, each item should
  contain `scene_id`, `prompt`, `gt_id`, `pred`, and optionally `qid`/`pred_id`.

## Outputs

Use `outputs/visualizations/` for point-cloud overlays and
`outputs/visualizations_mesh/` for mesh artifacts unless a command passes a different
`--output-dir`. Paper-style report meshes should not recolor objects; bbox lines carry
the GT/prediction annotation.

Do not commit generated PLY, PNG, JSON, HTML, or vertex-label outputs unless the user
explicitly asks for them.
