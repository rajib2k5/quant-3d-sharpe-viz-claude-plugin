"""Draw the plugin icon: a dark rounded tile with a small skyline and the red luck plane.

Usage: python make_icon.py [--out-dir assets/] [--plugin-icon .claude-plugin/icon.png]

Writes icon-512.png and icon-128.png, plus a byte copy of icon-512.png at .claude-plugin/icon.png,
where the plugin directory looks for the icon. The towers use the same gray -> blue -> gold alpha
t-statistic ramp as the scenes (skyline_common.t_stat_color); the tallest, gold tower pierces
the translucent red plane. Output is deterministic: same bytes on every run.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import: no display is assumed

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402

from skyline_common import LUCK_PLANE_RGB, t_stat_color  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SIZES = (512, 128)
TILE = "#0b0d14"
EDGE = "#2c3245"
# (x, width, height, alpha t-statistic) in a 0-1 unit square, left to right.
TOWERS = [(0.17, 0.10, 0.26, 0.0), (0.30, 0.10, 0.40, 1.0), (0.43, 0.10, 0.33, 2.0),
          (0.56, 0.12, 0.70, 4.0), (0.71, 0.10, 0.45, 2.6)]
GROUND = 0.18
PLANE = 0.56
DEPTH = (0.035, 0.03)  # side-face offset that gives the towers a hint of depth


def _shade(rgb: tuple[float, float, float], factor: float) -> tuple[float, float, float]:
    return tuple(min(1.0, c * factor) for c in rgb)  # type: ignore[return-value]


def draw(size: int) -> plt.Figure:
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.04, 0.04), 0.92, 0.92, boxstyle="round,pad=0,rounding_size=0.18",
                                facecolor=TILE, edgecolor=EDGE, linewidth=max(1.0, size / 160)))
    dx, dy = DEPTH
    for x, w, h, t in TOWERS:
        color = t_stat_color(t)
        top = GROUND + h
        ax.add_patch(Polygon([(x + w, GROUND), (x + w + dx, GROUND + dy), (x + w + dx, top + dy), (x + w, top)],
                             closed=True, facecolor=_shade(color, 0.62), edgecolor="none", zorder=2))
        ax.add_patch(Polygon([(x, top), (x + dx, top + dy), (x + w + dx, top + dy), (x + w, top)],
                             closed=True, facecolor=_shade(color, 1.18), edgecolor="none", zorder=2))
        ax.add_patch(Rectangle((x, GROUND), w, h, facecolor=color, edgecolor="none", zorder=2))
    # The luck plane, drawn over the towers it hides and under the one that pierces it.
    ax.add_patch(Polygon([(0.10, PLANE), (0.20, PLANE + 0.07), (0.92, PLANE + 0.07), (0.82, PLANE)],
                         closed=True, facecolor=LUCK_PLANE_RGB, alpha=0.42, edgecolor="none", zorder=3))
    x, w, h, t = max(TOWERS, key=lambda tower: tower[2])
    top = GROUND + h
    ax.add_patch(Rectangle((x, PLANE + 0.07), w, top - PLANE - 0.07, facecolor=t_stat_color(t), edgecolor="none", zorder=4))
    ax.add_patch(Polygon([(x, top), (x + dx, top + dy), (x + w + dx, top + dy), (x + w, top)],
                         closed=True, facecolor=_shade(t_stat_color(t), 1.18), edgecolor="none", zorder=4))
    ax.add_patch(Polygon([(x + w, PLANE + 0.07), (x + w + dx, PLANE + 0.07 + dy), (x + w + dx, top + dy), (x + w, top)],
                         closed=True, facecolor=_shade(t_stat_color(t), 0.62), edgecolor="none", zorder=4))
    return fig


def write_icons(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for size in SIZES:
        path = out_dir / f"icon-{size}.png"
        fig = draw(size)
        fig.savefig(path, dpi=100, transparent=True, metadata={"Software": None})
        plt.close(fig)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the plugin icon PNGs.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "assets")
    parser.add_argument("--plugin-icon", type=Path, default=REPO_ROOT / ".claude-plugin" / "icon.png",
                        help="where to copy icon-512.png for the plugin directory")
    args = parser.parse_args(argv)
    written = write_icons(args.out_dir)
    args.plugin_icon.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(written[SIZES.index(512)], args.plugin_icon)
    for path in [*written, args.plugin_icon]:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
