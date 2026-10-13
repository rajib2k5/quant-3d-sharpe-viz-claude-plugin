"""Render the Sharpe skyline to a PNG with matplotlib (mplot3d), with no Blender and no display.

Usage: python scene_python.py out/screen_result.json [--out out/skyline.png] [--window 3y]

Same layout and visual encoding as the Blender and three.js scenes (skyline_common.py), with
the vertical axis in annualized Sharpe units so the luck plane sits exactly at the screen's
headline value. The top-N names get a bright outline; the top LABELLED_TOP carry ticker labels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import: no display is assumed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

from skyline_common import COLOR_STOPS, LUCK_PLANE_RGB, build_scene, load_result  # noqa: E402

FIGSIZE_IN = (16.0, 10.0)
DPI = 100
LABELLED_TOP = 10
CROWDED = 1000  # above this many towers: thinner bars, no per-bar edges, staggered labels
CROWDED_BAR = 0.55
BACKGROUND = "#0b0d14"
PANE = (0.06, 0.07, 0.11, 1.0)
TEXT = "#e8eaf0"
MUTED = "#9aa1b2"
TOP_EDGE = "#ffffff"
OTHER_EDGE = "#1b1f2c"


def build_figure(result: dict[str, Any], window: str | None = None) -> tuple[plt.Figure, Any, dict[str, Any]]:
    """Draw the skyline; returns the figure, its 3D axes and a summary of what was drawn."""
    scene = build_scene(result)
    meta = result["metadata"]
    window = window or scene["window"]
    if window not in scene["windows"]:
        raise ValueError(f"window {window!r} not in {scene['windows']}")
    plane_z = meta["luck_horizon_by_window"][window]["headline"]
    if plane_z is None:
        raise ValueError(f"no luck plane for window {window!r} (fewer than two names scored)")

    fig = plt.figure(figsize=FIGSIZE_IN, dpi=DPI, facecolor=BACKGROUND)
    ax = fig.add_axes((0.0, 0.0, 0.86, 1.0), projection="3d", facecolor=BACKGROUND)
    towers = scene["towers"]
    crowded = len(towers) > CROWDED
    size = scene["tower_size"] * (CROWDED_BAR if crowded else 1.0)
    heights = [t["sr_ann"][window] or 0.0 for t in towers]

    # mplot3d depth-sorts within a collection but not between collections, so each tower is
    # split at the luck plane: segments below it are drawn first, then the plane, then the
    # segments above it.
    ax.computed_zorder = False
    split = max(plane_z, 0.0)
    for top in (False, True):
        for above in (False, True):
            segments = []  # (tower, base z, height)
            for tower, h in zip(towers, heights):
                if tower["top"] is not top:
                    continue
                if above:
                    if h > split:
                        segments.append((tower, split, h - split))
                else:
                    segments.append((tower, 0.0, min(h, split)) if h >= 0 else (tower, h, -h))
            if segments:
                ax.bar3d(
                    [t["x"] - size / 2 for t, _, _ in segments], [t["y"] - size / 2 for t, _, _ in segments],
                    [z for _, z, _ in segments], size, size, [dz for _, _, dz in segments],
                    color=[t["color"] for t, _, _ in segments], shade=True, zorder=3 if above else 1,
                    edgecolor=TOP_EDGE if top else OTHER_EDGE,
                    linewidth=(0.8 if crowded else 1.1) if top else (0.0 if crowded else 0.3),
                )

    half_w = scene["extent"]["width"] / 2 + 1.5
    half_d = scene["extent"]["depth"] / 2 + 1.5
    xs, ys = np.meshgrid([-half_w, half_w], [-half_d, half_d])
    ax.plot_surface(xs, ys, np.full_like(xs, plane_z), color=LUCK_PLANE_RGB, alpha=0.28, shade=False, zorder=2)

    ranked = sorted((t for t in towers if t["rank"] is not None), key=lambda t: t["rank"])
    labelled = [t["ticker"] for t in ranked[:LABELLED_TOP]]
    z_span = max(max(heights), plane_z) - min(min(heights), 0.0)
    for i, tower in enumerate(ranked[:LABELLED_TOP]):
        top_z = max(tower["sr_ann"][window] or 0.0, 0.0)
        # Crowded scenes put the leaders next to each other: fan their labels out in height
        # and alternate them left and right of the tower so they don't print on top of each other.
        if crowded:
            top_z += 0.07 * z_span * (i % 5)
        ax.text(tower["x"], tower["y"], top_z + 0.12, tower["ticker"], color=TEXT, fontsize=8, fontweight="bold",
                ha=("right" if i % 2 else "left") if crowded else "center", va="bottom", zorder=5)
    for sector in scene["sectors"]:
        ax.text(sector["x"], sector["y"] - 0.6, 0.0, sector["name"], color=MUTED, fontsize=7, ha="center", va="top",
                zorder=4)

    z_values = heights + [plane_z, 0.0]
    ax.set_zlim(min(z_values) - 0.1, max(z_values) * 1.08 + 0.1)
    ax.set_xlim(-half_w, half_w)
    ax.set_ylim(-half_d, half_d)
    ax.set_box_aspect((2 * half_w, 2 * half_d, max(half_w, half_d) * 0.9))
    ax.view_init(elev=26, azim=-62)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zlabel("Annualized Sharpe", color=MUTED, labelpad=8)
    ax.tick_params(axis="z", colors=MUTED, labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(PANE)
        axis._axinfo["grid"]["color"] = (0.2, 0.22, 0.3, 0.5)

    top_n = scene["top_n"]
    legend = "\n".join([
        f"Sharpe Skyline: {meta['N']} names, {window} window",
        "Tower height = annualized Sharpe; colour = FF3 alpha t-statistic",
        f"White outline = top {top_n} by Sharpe; labels = top {LABELLED_TOP}",
        f"Red plane = {plane_z:.2f}: expected best-of-{meta['N']} Sharpe under a zero-alpha null.",
        "Rising above it is necessary, not sufficient; the skill test is DSR >= 0.95.",
        "Historical statistics, not investment advice.",
    ])
    fig.text(0.015, 0.975, legend, color=TEXT, fontsize=10, va="top", ha="left", linespacing=1.5,
             bbox={"boxstyle": "round,pad=0.6", "facecolor": (0.06, 0.075, 0.12, 0.9), "edgecolor": "#2a3042"})

    t_max = COLOR_STOPS[-1][0]
    cmap = LinearSegmentedColormap.from_list("t_stat", [(t / t_max, c) for t, c in COLOR_STOPS])
    cax = fig.add_axes((0.89, 0.25, 0.015, 0.5))
    bar = fig.colorbar(ScalarMappable(norm=Normalize(0.0, t_max), cmap=cmap), cax=cax)
    bar.set_label("FF3 alpha t-statistic", color=MUTED)
    bar.set_ticks([0, 2, 4], labels=["<= 0", "2", ">= 4"])
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    bar.outline.set_edgecolor("#2a3042")

    stats = {
        "window": window,
        "tower_count": len(towers),
        "bar_size": size,
        "plane_z": plane_z,
        "outlined": [t["ticker"] for t in towers if t["top"]],
        "labelled": labelled,
    }
    return fig, ax, stats


def render_png(result_path: Path, out: Path, window: str | None = None) -> dict[str, Any]:
    fig, _, stats = build_figure(load_result(result_path), window)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor=fig.get_facecolor(), metadata={"Software": None})
    plt.close(fig)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the Sharpe skyline to a PNG with matplotlib.")
    parser.add_argument("result", type=Path, help="screen_result.json from screen.py")
    parser.add_argument("--out", type=Path, help="default: skyline.png next to the result")
    parser.add_argument("--window", help="window label such as 1y, 3y or 5y (default: the screened window)")
    args = parser.parse_args(argv)
    out = args.out or args.result.parent / "skyline.png"
    try:
        stats = render_png(args.result, out, args.window)
    except ValueError as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"wrote {out} ({stats['tower_count']} towers, luck plane at {stats['plane_z']:.3f}, {stats['window']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
