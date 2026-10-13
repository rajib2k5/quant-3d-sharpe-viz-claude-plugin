"""Scene layout and visual encoding shared by the Blender, matplotlib and three.js skyline generators.

Encoding: one tower per name, grouped into one block per sector. Height is proportional to
the annualized Sharpe ratio (negative Sharpe extends below ground), scaled so the tallest
tower over all windows is MAX_HEIGHT units. Colour encodes the Fama-French 3 alpha
t-statistic on a gray -> blue -> gold ramp. The top-N names glow and carry a label. One
translucent red plane sits at the luck plane: the expected best-of-N Sharpe under a zero-alpha
null (see screen.py). Rising above it is necessary but not sufficient; the skill test is DSR >= 0.95.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

MAX_HEIGHT = 8.0
TOWER_SIZE = 1.0
TOWER_PITCH = 1.4
BLOCK_GAP = 2.4
COLOR_STOPS: list[tuple[float, tuple[float, float, float]]] = [
    (0.0, (0.55, 0.55, 0.58)),
    (2.0, (0.16, 0.42, 0.95)),
    (4.0, (1.00, 0.76, 0.15)),
]
LUCK_PLANE_RGB = (0.90, 0.12, 0.12)


def load_result(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def t_stat_color(t: float | None) -> tuple[float, float, float]:
    """sRGB colour for an alpha t-statistic: gray at t <= 0, blue at t = 2, gold at t >= 4."""
    if t is None or not math.isfinite(t) or t <= COLOR_STOPS[0][0]:
        return COLOR_STOPS[0][1]
    for (t0, c0), (t1, c1) in zip(COLOR_STOPS, COLOR_STOPS[1:]):
        if t <= t1:
            w = (t - t0) / (t1 - t0)
            return tuple(round(a + w * (b - a), 4) for a, b in zip(c0, c1))  # type: ignore[return-value]
    return COLOR_STOPS[-1][1]


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def _web_url(value: str | None) -> str | None:
    """Only absolute http(s) URLs from a user's universe CSV are used as links."""
    return value if value and value.lower().startswith(("http://", "https://")) else None


def build_scene(result: dict[str, Any]) -> dict[str, Any]:
    """Positions, heights and colours for every tower, plus the luck plane, in scene units."""
    meta = result["metadata"]
    labels = list(meta["windows"])
    rows = result["universe"]

    finite = [abs(v) for row in rows for v in row["sr_ann"].values() if v is not None]
    scale = MAX_HEIGHT / max(finite) if finite and max(finite) > 0 else 1.0

    by_sector: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda r: (r["sector"], r["rank"] is None, r["rank"] or 0, r["ticker"])):
        by_sector.setdefault(row["sector"], []).append(row)

    cols = max(1, math.ceil(math.sqrt(max(len(v) for v in by_sector.values()))))
    block_cols = max(1, math.ceil(math.sqrt(len(by_sector))))
    block_size = cols * TOWER_PITCH
    block_rows = math.ceil(len(by_sector) / block_cols)
    max_rows = max(math.ceil(len(v) / cols) for v in by_sector.values())
    block_depth = max_rows * TOWER_PITCH
    width = block_cols * block_size + (block_cols - 1) * BLOCK_GAP
    depth = block_rows * block_depth + (block_rows - 1) * BLOCK_GAP

    towers: list[dict[str, Any]] = []
    sectors: list[dict[str, Any]] = []
    for b, (sector, members) in enumerate(by_sector.items()):
        bx = (b % block_cols) * (block_size + BLOCK_GAP) - width / 2 + TOWER_PITCH / 2
        by = depth / 2 - (b // block_cols) * (block_depth + BLOCK_GAP) - TOWER_PITCH / 2
        front = by - max_rows * TOWER_PITCH  # every block's label on the same front line
        sectors.append({"name": sector, "x": round(bx + (cols - 1) * TOWER_PITCH / 2, 4), "y": round(front, 4)})
        for i, row in enumerate(members):
            towers.append({
                "ticker": row["ticker"],
                "sector": sector,
                "x": round(bx + (i % cols) * TOWER_PITCH, 4),
                "y": round(by - (i // cols) * TOWER_PITCH, 4),
                "height": {w: round((row["sr_ann"][w] or 0.0) * scale, 4) for w in labels},
                "sr_ann": {w: _r(row["sr_ann"][w]) for w in labels},
                "color": list(t_stat_color(row["t_alpha_ff3"])),
                "top": bool(row["in_top"]),
                "rank": row["rank"],
                "alpha_ff3_ann": _r(row["alpha_ff3_ann"]),
                "t_alpha_ff3": _r(row["t_alpha_ff3"], 2),
                "alpha_capm_ann": _r(row["alpha_capm_ann"]),
                "t_alpha_capm": _r(row["t_alpha_capm"], 2),
                "psr": _r(row["psr"]),
                "dsr": _r(row["dsr"]),
                "dsr_passes": bool(row["dsr_passes"]),
                "above_luck_plane": bool(row["above_luck_plane"]),
                "n_obs": row["n_obs"],
                "website": _web_url(row.get("website")),
            })

    horizons = meta["luck_horizon_by_window"]
    luck_ann = {w: _r(horizons[w]["headline"]) for w in labels}
    return {
        "windows": labels,
        "window": meta["window_label"],
        "N": meta["N"],
        "top_n": meta["top_n"],
        "top_dsr_passing": meta["top_dsr_passing"],
        "top_above_luck_plane": meta["top_above_luck_plane"],
        "scale": round(scale, 6),
        "luck_horizon_ann": luck_ann,
        "luck_height": {w: round((v or 0.0) * scale, 4) for w, v in luck_ann.items()},
        "extent": {"width": round(width, 4), "depth": round(depth, 4)},
        "tower_size": TOWER_SIZE,
        "luck_rgb": list(LUCK_PLANE_RGB),
        "color_stops": [[t, list(c)] for t, c in COLOR_STOPS],
        "sectors": sectors,
        "towers": towers,
    }
