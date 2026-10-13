---
name: sharpe-skyline
description: Render a Sharpe screen as a 3D skyline, by default as a PNG rendered headless with no installation (matplotlib, or Blender/bpy when available), optionally live in the user's Blender through a Blender MCP server, or as a single-file three.js web page that loads three.js from the jsdelivr CDN. Towers are stocks grouped by sector, height is annualized Sharpe, colour is alpha t-statistic, and a red plane marks the expected best-of-N Sharpe under a zero-alpha null. Use when the user asks to visualize, plot or render the Sharpe screen in 3D, wants a Blender scene of the results, or asks for a skyline or city view of risk-adjusted returns.
---

# Sharpe skyline

Turns `screen_result.json` from the `sharpe-screen` skill into a 3D scene. Run the screen first if `out/screen_result.json` does not exist:

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/screen.py" --out out/
```

## Default flow: PNG plus the interactive page, opened for the user

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/render_skyline.py" out/screen_result.json --out out/skyline.png --html
```

This renders `out/skyline.png` and also writes the interactive three.js page `out/skyline.html`. It prints both paths on stdout and which tier rendered the PNG on stderr. Then:

1. Show or send the PNG.
2. Open the interactive page for the user. If you have a built-in browser or preview pane (for example inside the Claude desktop app), open `out/skyline.html` there rather than shelling out to `xdg-open`. From a plain terminal session, add `--open` to the command above, which launches the default browser (it implies `--html`). If neither is possible, give the user the file path.

The PNG renderer is chosen automatically, best first:

1. **`--blender PATH`**, when given: that Blender binary, run headless.
2. **bpy module** in the current interpreter (Blender's Cycles engine, 64 samples, 1600x1000).
3. **`blender` on PATH**, run headless.
4. **matplotlib** (`scene_python.py`), which is in requirements.txt and always works; it renders in under a second.

A tier that is missing or fails is noted on stderr and the next one is used. The only hard failure is a missing matplotlib (`error: matplotlib is not installed; run pip install -r requirements.txt`). To skip the tier search and use matplotlib directly:

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/scene_python.py" out/screen_result.json --out out/skyline.png [--window 5y]
```

The matplotlib image uses the same encoding as the Blender scene, with the vertical axis in annualized Sharpe units: white outlines mark the top 20, ticker labels the top 10, and a legend box states how to read the luck plane.

### Optional quality upgrade: Blender

A Blender render (tier 1 or 2) has real lighting and glow. It is optional: `pip install bpy` works on x86_64 Linux, Windows and macOS (there is no ARM64 Linux wheel, so bpy is not in requirements.txt), or install a Blender binary (`sudo apt-get install -y blender`). Mention this only if the user asks for a nicer render. The Blender tiers also leave `out/skyline_bpy.py` next to the PNG, the same script scene_blender.py writes:

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/scene_blender.py" out/screen_result.json   # -> out/skyline_bpy.py
```

## Optional: interactive scene in the user's Blender (Blender MCP)

When the user has a Blender MCP server (for example an `execute_blender_code` tool) and wants to explore the scene interactively, read `out/skyline_bpy.py` and send its full text as the code to execute. If the tool rejects the size, regenerate in parts with `--chunk-bytes 30000` and send `out/skyline_bpy_part01.py`, `_part02.py`, ... in numeric order, one call each. The early parts only stash the data inside Blender; the last part builds the scene. Then use the server's viewport or screenshot tool to show the result.

The script only creates and replaces objects in its own `SharpeSkyline` collection (data names start with `Skyline_`), so re-running it updates the scene without touching the user's other objects. It does set the scene camera and, if needed, the world background colour. Only a `--render` run hides other objects from the render.

## The interactive page

The page can also be built on its own: `python "${CLAUDE_PLUGIN_ROOT}/scripts/scene_threejs.py" out/screen_result.json [--out PATH] [--open]`. It is one file that loads three.js 0.160.0 from cdn.jsdelivr.net, so it needs network access. Tell the user what it offers:

- **Navigate:** drag the city (floor or towers) to move across it like a map, drag the sky to orbit, right-drag to pan, scroll to zoom; arrow keys or WASD also move; R or the Reset view button restores the starting view; double-click any bar to fly to it and open its details. Movement stays within the scene's bounds. Window buttons (1y / 3y / 5y) switch the window; hover a tower for a quick summary.
- **Layout:** "By sector" (default) groups towers by sector, or by exchange in full-market mode. "By Sharpe" animates every tower into one grid ordered by the current window's Sharpe, tallest at the near corner, and hides the sector floors and labels. Window buttons re-sort it.
- **Ticker labels:** the top 20 are always labelled. Zooming in labels the nearest bars, up to 100 labels that fade with distance, so any bar can be identified by its ticker. The Labels button switches them off, and the selected bar keeps a highlighted label. The PNG labels only the top 10.
- **Click to inspect:** clicking a tower, or a row of the top-20 list on the right, opens a details panel and highlights the tower (a list click also flies the camera to it). The panel shows rank, Sharpe for every window, CAPM and FF3 alpha with t-statistics, PSR, DSR with a pass/fail chip against 0.95, whether the tower is above the luck plane, and the days of data. Click empty space or the x to close it.
- **External links (new tab):** Google Finance, SEC EDGAR company filings (sec.gov; share classes such as BRK.B are looked up as BRK-B, the form EDGAR uses) and the company website. The website link uses a `website` column from the user's `--universe` CSV when present (http/https only), otherwise a web search for the ticker's investor-relations page, labelled "Website (search)". These links leave the page; nothing is sent anywhere until the user clicks one.
- **Reference values in the scene:** a "Sharpe 0" level (negative towers go below it), the luck plane labelled with its value for the current window, and a value ruler at the near-left corner with the plane's value marked in red.
- **Legends:** a "how to read a bar" panel and a colour bar for the FF3 alpha t-statistic.

## Visual encoding

- **Tower** = one stock, grouped into one block per sector, with the sector name on the ground in front.
- **Height** = annualized Sharpe ratio, scaled so the tallest tower in any window is 8 units. A negative Sharpe goes below ground.
- **Colour** = Fama-French 3 alpha t-statistic: gray at t <= 0, blue at t = 2, gold at t >= 4.
- **White outline** = the top 20 by Sharpe in the selected window; **glow and label** = the top 10 (the Blender scene glows and labels all top 20).
- **Red translucent plane** = the luck plane: the expected best-of-N Sharpe under a zero-alpha null, in which every name has the universe's average Sharpe. Under that null the single best name still rises above it a little under half the time.
- **Animation** (Blender) / **window buttons** (three.js): heights and the luck plane move between the 1y, 3y and 5y windows.

## Reading the skyline

**Not financial advice:** always say that the skyline shows historical statistics for research and education, not a recommendation to buy, sell or hold anything.

Rising **through** the red plane is necessary but not sufficient: the skill test is a Deflated Sharpe Ratio of at least 0.95 (`dsr_passes` in the JSON, shown in the tooltip). Tall towers that stay **under** the plane, or poke through it with DSR below 0.95, cannot be distinguished from luck at this sample size; that reflects low statistical power, not proof of luck. A gold tower above the plane combines a high Sharpe with a significant alpha. Say this when you present the scene, and add that it describes the past, is not a forecast, and is not investment advice.
