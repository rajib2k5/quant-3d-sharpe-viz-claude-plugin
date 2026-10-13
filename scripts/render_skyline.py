"""Render the Sharpe skyline to a PNG with the best renderer available, no UI needed.

Usage: python render_skyline.py out/screen_result.json [--out out/skyline.png] [--blender PATH]
                                [--html [PATH]] [--open]

Tiers, in order:
1. The ``bpy`` module (``pip install bpy``) importable in this interpreter: the Blender scene
   script from scene_blender.py runs in-process as if launched with ``-- --render PATH``.
   Skipped when ``--blender`` is given.
2. A Blender binary: ``--blender PATH``, else $SHARPE_OBSERVATORY_BLENDER, else ``blender`` on
   PATH, run as ``blender -b --factory-startup --python SCRIPT --python-exit-code 1 -- --render PATH``.
3. matplotlib (scene_python.py), which needs no Blender at all.
A tier that is unavailable or fails is reported on stderr and the next one is tried; only a
missing matplotlib at tier 3 is fatal. The Blender tiers also leave the generated script next
to the PNG as skyline_bpy.py. ``--html`` also writes the interactive three.js page
(scene_threejs.py) and ``--open`` opens that page in the default browser (implies ``--html``).
"""

from __future__ import annotations

import argparse
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import scene_threejs
from scene_blender import render_script
from skyline_common import build_scene, load_result

ENV_VAR = "SHARPE_OBSERVATORY_BLENDER"
TIMEOUT_S = 600
TAIL_LINES = 12


class RenderError(RuntimeError):
    pass


def _bpy_available() -> bool:
    try:
        import bpy  # noqa: F401
    except ImportError:
        return False
    return True


def _resolve_binary(candidate: str, origin: str) -> str:
    found = shutil.which(candidate)
    if found is None:
        raise RenderError(f"{origin} {candidate!r} is not an executable Blender binary")
    return found


def find_blender(explicit: str | None) -> str | None:
    """The Blender binary to use: --blender, then $SHARPE_OBSERVATORY_BLENDER, then PATH."""
    if explicit:
        return _resolve_binary(explicit, "--blender")
    if os.environ.get(ENV_VAR):
        return _resolve_binary(os.environ[ENV_VAR], f"${ENV_VAR}")
    return shutil.which("blender")


def _tail(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return " | ".join(lines[-TAIL_LINES:]) or "(no output)"


def write_bpy_script(result_path: Path, script: Path) -> Path:
    script.write_text(render_script(build_scene(load_result(result_path))), encoding="utf-8")
    return script


def _check_png(out: Path, renderer: str) -> None:
    if not out.is_file() or out.stat().st_size == 0:
        raise RenderError(f"{renderer} finished but wrote no image to {out}")


def render_with_bpy(script: Path, out: Path) -> None:
    """Run the generated script in this interpreter with its ``-- --render`` arguments."""
    saved_argv = sys.argv
    sys.argv = [str(script), "--", "--render", str(out)]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except (Exception, SystemExit) as exc:
        raise RenderError(f"in-process bpy render failed: {type(exc).__name__}: {exc}") from exc
    finally:
        sys.argv = saved_argv
    _check_png(out, "in-process bpy render")


def render_with_binary(blender: str, script: Path, out: Path) -> None:
    command = [blender, "-b", "--factory-startup", "--python", str(script), "--python-exit-code", "1",
               "--", "--render", str(out)]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise RenderError(f"blender did not finish within {TIMEOUT_S} s") from exc
    if proc.returncode != 0:
        raise RenderError(f"blender exited with code {proc.returncode}: {_tail(proc.stderr or proc.stdout)}")
    _check_png(out, f"blender ({_tail(proc.stderr or proc.stdout)})")


def render_with_matplotlib(result_path: Path, out: Path) -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise RenderError("matplotlib is not installed; run pip install -r requirements.txt") from exc
    import scene_python

    scene_python.render_png(result_path, out)
    _check_png(out, "matplotlib")


def _skip(tier: str, reason: object) -> None:
    print(f"[render] {tier} skipped: {reason}", file=sys.stderr)


def render(result_path: Path, out: Path, explicit_blender: str | None = None) -> str:
    """Render ``result_path`` to ``out``; returns a description of the tier that produced it."""
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    script = out.parent / "skyline_bpy.py"

    if explicit_blender is None and _bpy_available():
        try:
            render_with_bpy(write_bpy_script(result_path, script), out)
            return "tier 1 (bpy module)"
        except RenderError as exc:
            _skip("tier 1 (bpy module)", exc)
    try:
        blender = find_blender(explicit_blender)
    except RenderError as exc:
        _skip("tier 2 (blender binary)", exc)
        blender = None
    if blender is not None:
        try:
            render_with_binary(blender, write_bpy_script(result_path, script), out)
            return f"tier 2 (blender binary {blender})"
        except RenderError as exc:
            _skip(f"tier 2 (blender binary {blender})", exc)
    render_with_matplotlib(result_path, out)
    return "tier 3 (matplotlib)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the Sharpe skyline to a PNG with the best available renderer.")
    parser.add_argument("result", type=Path, help="screen_result.json from screen.py")
    parser.add_argument("--out", type=Path, help="PNG path (default: skyline.png next to the result)")
    parser.add_argument("--blender", help="Blender binary to use (skips the bpy module tier)")
    parser.add_argument("--html", nargs="?", const="", metavar="PATH",
                        help="also write the interactive three.js page (default: the PNG path with .html)")
    parser.add_argument("--open", action="store_true", help="open the three.js page in the default browser (implies --html)")
    args = parser.parse_args(argv)
    if not args.result.is_file():
        parser.exit(1, f"error: {args.result} does not exist; run screen.py first\n")
    out = args.out or args.result.with_name("skyline.png")
    try:
        tier = render(args.result, out, args.blender)
    except (RenderError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"rendered with {tier}", file=sys.stderr)
    print(out.resolve())
    if args.html is not None or args.open:
        page = Path(args.html) if args.html else out.with_suffix(".html")
        print(scene_threejs.write_page(args.result, page, open_browser=args.open).resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
