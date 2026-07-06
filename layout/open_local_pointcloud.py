from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


HIGHLIGHT_COLORS = np.array(
    [
        [255, 80, 80],
        [255, 180, 60],
        [255, 80, 220],
        [255, 220, 90],
        [80, 255, 120],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a local ASCII PLY point cloud in an interactive Plotly viewer."
    )
    parser.add_argument(
        "input_ply",
        nargs="?",
        default="outputs/visualizations/scene0011_00_sample00000_token_token014.ply",
        help="Path to the local ASCII PLY file.",
    )
    parser.add_argument(
        "--output-html",
        default=None,
        help="Optional output html path. Defaults next to the input ply.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=100000,
        help="Maximum number of displayed points. Highlight colors are preserved first.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.2,
        help="Marker size in the interactive viewer.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Only generate the html file, do not open a browser window.",
    )
    return parser.parse_args()


def load_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        vertex_count = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Invalid ply file: {path}")
            if line.startswith("element vertex "):
                vertex_count = int(line.strip().split()[-1])
            if line.strip() == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"Missing vertex count in ply file: {path}")
        data = np.loadtxt(handle, max_rows=vertex_count)
    points = data[:, :3].astype(np.float32)
    colors = data[:, 3:6].astype(np.uint8)
    return points, colors


def subsample_points(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= max_points:
        return points, colors

    highlight_mask = np.any(
        np.all(colors[:, None, :] == HIGHLIGHT_COLORS[None, :, :], axis=2), axis=1
    )
    highlight_idx = np.flatnonzero(highlight_mask)
    other_idx = np.flatnonzero(~highlight_mask)

    if len(highlight_idx) >= max_points:
        keep_idx = highlight_idx[:max_points]
    else:
        remain = max_points - len(highlight_idx)
        sampled_other = other_idx[np.linspace(0, len(other_idx) - 1, remain, dtype=np.int64)]
        keep_idx = np.concatenate([highlight_idx, sampled_other])

    keep_idx.sort()
    return points[keep_idx], colors[keep_idx]


def build_figure(points: np.ndarray, colors: np.ndarray, point_size: float) -> go.Figure:
    color_strings = [f"rgb({r},{g},{b})" for r, g, b in colors.tolist()]
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(size=point_size, color=color_strings, opacity=0.9),
                hoverinfo="skip",
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="Interactive Point Cloud Viewer",
        showlegend=False,
    )
    return fig


def main() -> int:
    args = parse_args()
    input_ply = Path(args.input_ply).resolve()
    if args.output_html is None:
        output_html = input_ply.with_suffix(".html")
    else:
        output_html = Path(args.output_html).resolve()

    points, colors = load_ascii_ply(input_ply)
    shown_points, shown_colors = subsample_points(points, colors, args.max_points)
    fig = build_figure(shown_points, shown_colors, args.point_size)
    fig.write_html(str(output_html), include_plotlyjs="cdn", auto_open=False)

    print(f"Input PLY:   {input_ply}")
    print(f"Output HTML: {output_html}")
    print(f"Shown points: {len(shown_points)} / {len(points)}")

    if not args.no_open:
        webbrowser.open(output_html.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
