# vision3D Agent Guide

## Project Goal

vision3D is for 3D point-cloud visualization and dataset visualization. The project
focuses on turning ScanNet-style scenes, text prompts, ground-truth object annotations,
and model answers into inspectable visual artifacts:

- colored point clouds and meshes,
- GT/context bounding boxes,
- prompt and object metadata JSON,
- PNG previews,
- interactive Plotly HTML viewers.

The first implementation is organized from the private reference repository:
`git@github.com:sol924/Training_Free_Token_Redcution.git`.

## Directory Structure

- `point/`: reads point-cloud scenes, prompts, GT labels, predicted object attributes,
  and model-token object regions; writes PLY/PNG/JSON artifacts.
- `layout/`: arranges generated point/mesh artifacts into static previews and
  interactive HTML viewers.
- `run/`: simple executable wrappers for common workflows.
- `outputs/`: generated visualizations. This directory is ignored by Git.
- `sample_data/`, `annotations/`, `datasets/`: optional local data roots. These are
  ignored by Git because ScanNet assets and model attributes can be large.

## Development Rules

- Keep scripts runnable from the repository root.
- Prefer explicit CLI arguments such as `--annotation-root`, `--scene-root`,
  `--dataset-json`, and `--output-dir` over hard-coded local paths.
- Keep generated files under `outputs/` unless the user explicitly asks for another
  path.
- Do not commit private datasets, checkpoints, `.pt/.pth` files, generated PLY/HTML/PNG
  outputs, virtual environments, or caches.
- After any code or documentation change, sync the repository to GitHub.

## GitHub Sync Requirement

This project remote is:

```bash
git@github.com:sol924/vision3D.git
```

Initial setup:

```bash
echo "# vision3D" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:sol924/vision3D.git
git push -u origin main
```

For future changes:

```bash
git status
git add README.md AGENTS.md point layout run requirements.txt .gitignore
git commit -m "<clear change summary>"
git push
```

If generated files are intentionally needed in Git, add them explicitly with `git add -f`
and explain why in the commit message.
