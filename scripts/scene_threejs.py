"""Write a single-file three.js page of the Sharpe skyline (the interactive view).

Usage: python scene_threejs.py out/screen_result.json [--out out/skyline.html] [--open]

The page embeds its data but loads three.js 0.160.0 (and its OrbitControls / CSS2DRenderer
addons) from the cdn.jsdelivr.net CDN, so it needs no build step but does need network access.
Visual encoding is the same as the Blender and matplotlib scenes (skyline_common.py). On top
of that the page has a verdict strip, click-to-inspect details with external links and a
copy-JSON button, ticker search, a zero level, a labelled luck plane, a value ruler, sector
floor tints, a colour bar and a clickable top-N list.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
import webbrowser
from pathlib import Path
from typing import Any

from skyline_common import build_scene, load_result

THREE_VERSION = "0.160.0"
THREE_BASE = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}"
LABEL_TOP = 10
NAV_MARGIN = 2.0
LINK_TEMPLATES = {
    "google_finance": "https://www.google.com/finance?q={ticker}",
    "edgar": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&ticker={ticker}",
    "website_search": "https://www.google.com/search?q={ticker}+investor+relations",
}
SECTOR_TINTS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948",
                "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#86bcb6"]
TILE_MARGIN = 0.25
DETAIL_FIELDS = ["ticker", "sector", "rank", "sr_ann", "alpha_capm_ann", "t_alpha_capm", "alpha_ff3_ann",
                 "t_alpha_ff3", "psr", "dsr", "dsr_passes", "above_luck_plane", "n_obs", "website"]

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sharpe Skyline</title>
<style>
  :root {
    --bg: #0b0d14; --panel: rgba(17, 20, 31, 0.94); --text: #eceef3; --muted: #a3aabb; --line: #2c3245;
    --pass: #34c77b; --fail: #8f96a8; --luck: #c43a39; --accent: #6fd6ff;
    --font: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    --fs-title: 18px; --fs-section: 13px; --fs-body: 13px; --fs-caption: 11.5px; --fs-stat: 20px;
  }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text);
    font: var(--fs-body)/1.45 var(--font); overflow: hidden; }
  .stat b, .chip, dd, .sr, #toplist .rank { font-variant-numeric: tabular-nums; }
  #stage { position: fixed; inset: 0; }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px;
    box-sizing: border-box; }
  h1 { font-size: var(--fs-title); margin: 0 0 8px; letter-spacing: .01em; }
  h2 { font-size: var(--fs-section); margin: 0 0 6px; font-weight: 600; }
  .caption, .disclaimer { font-size: var(--fs-caption); color: var(--muted); }
  #legend { position: fixed; top: 16px; left: 16px; width: 330px; max-height: calc(100vh - 240px); overflow: auto; }
  #verdict { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-bottom: 8px; }
  .stat { border: 1px solid var(--line); border-radius: 8px; padding: 6px 8px; }
  .stat b { display: block; font-size: var(--fs-stat); line-height: 1.15; }
  .stat span { font-size: var(--fs-caption); color: var(--muted); }
  #verdict-line { margin: 0 0 8px; font-size: var(--fs-body); }
  #windows { margin: 6px 0 8px; display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
  button, input { font: inherit; color: var(--text); }
  #windows button, .toggle, #reset-view { background: transparent; border: 1px solid var(--line); border-radius: 6px; padding: 3px 10px; cursor: pointer; }
  #controls-hint { margin: 0 0 8px; }
  #reset-view { margin-left: 4px; font-size: var(--fs-caption); }
  #windows button[aria-pressed="true"] { background: #2d52e0; border-color: #2d52e0; }
  details summary { cursor: pointer; color: var(--text); font-weight: 600; font-size: var(--fs-section); margin: 4px 0; }
  details ul { margin: 4px 0; padding-left: 18px; color: var(--muted); }
  details li { margin: 3px 0; }
  details strong { color: var(--text); }
  #sectors { display: grid; grid-template-columns: 1fr 1fr; gap: 2px 10px; font-size: var(--fs-caption); color: var(--muted); }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
  #colorbar { position: fixed; left: 16px; bottom: 16px; display: flex; gap: 10px; align-items: stretch; }
  #colorbar .bar { width: 14px; height: 140px; border-radius: 3px; border: 1px solid var(--line); }
  #colorbar .ticks { display: flex; flex-direction: column; justify-content: space-between; color: var(--muted); font-size: var(--fs-caption); }
  #colorbar .title { writing-mode: vertical-rl; transform: rotate(180deg); color: var(--muted); font-size: var(--fs-caption); text-align: center; }
  #side { position: fixed; top: 16px; right: 16px; bottom: 16px; width: 310px; display: flex; flex-direction: column; gap: 10px;
    pointer-events: none; }
  #side > * { pointer-events: auto; }
  #search-box { display: flex; flex-direction: column; gap: 4px; }
  #search { background: #0f121c; border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px; }
  #search:focus, button:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
  #search-status { min-height: 1em; }
  #details[hidden] { display: none; }
  #details { position: relative; }
  #details h2 { font-size: var(--fs-title); margin: 0; }
  #details-close { position: absolute; top: 6px; right: 8px; background: transparent; border: 0; font-size: 20px; cursor: pointer; }
  #d-json { max-height: 160px; overflow: auto; font-size: var(--fs-caption); background: #0f121c; padding: 6px;
    border-radius: 6px; user-select: all; }
  #details dl { display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; margin: 8px 0; }
  #details dt { color: var(--muted); }
  #details dd { margin: 0; text-align: right; }
  #details nav { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
  #details nav a, #copy-json, #fly-to { color: var(--accent); border: 1px solid var(--line); border-radius: 6px; padding: 3px 8px;
    text-decoration: none; background: transparent; cursor: pointer; }
  .chip { display: inline-block; border-radius: 10px; padding: 0 8px; font-size: var(--fs-caption); font-weight: 600; color: #0b0d14; }
  .chip.pass { background: var(--pass); }
  .chip.fail { background: var(--fail); }
  .chip.above { background: var(--luck); color: #fff; }
  #toplist { flex: 1; min-height: 0; display: flex; flex-direction: column; }
  #toplist ol { list-style: none; margin: 0; padding: 0; overflow: auto; }
  #toplist button { width: 100%; display: grid; grid-template-columns: 26px 1fr auto auto; gap: 8px; align-items: center;
    background: transparent; border: 0; border-bottom: 1px solid var(--line); padding: 5px 2px; cursor: pointer; text-align: left; }
  #toplist button:hover, #toplist button[aria-current="true"] { background: rgba(111, 214, 255, 0.12); }
  #toplist .rank { color: var(--muted); }
  #tooltip { position: fixed; pointer-events: none; display: none; max-width: 240px; }
  #tooltip div { white-space: nowrap; }
  #toolbar { display: none; }
  .label { color: #fff; font-size: var(--fs-caption); font-weight: 600; text-shadow: 0 0 4px #000, 0 0 2px #000; }
  .sector { color: var(--muted); font-size: var(--fs-caption); text-transform: uppercase; letter-spacing: .04em; text-shadow: 0 0 3px #000; }
  .ref { font-size: var(--fs-caption); color: var(--muted); text-shadow: 0 0 3px #000; font-variant-numeric: tabular-nums; }
  .ref.luck { color: #fff; background: #b3302f; border-radius: 4px; padding: 1px 6px; font-weight: 600; }
  .ref.zero { color: #d5dbe7; border: 1px solid var(--line); border-radius: 4px; padding: 0 5px; background: rgba(11, 13, 20, 0.8); }
  @media (max-width: 760px) {
    #toolbar { display: flex; gap: 6px; position: fixed; top: 8px; left: 8px; right: 8px; z-index: 5; }
    .toggle { background: var(--panel); }
    .toggle[aria-pressed="true"] { background: #2d52e0; border-color: #2d52e0; }
    #legend { top: 46px; left: 8px; right: 8px; width: auto; max-height: 55vh; display: none; }
    #side { top: auto; left: 8px; right: 8px; bottom: 8px; width: auto; max-height: 55vh; }
    #toplist, #search-box { display: none; }
    body.show-legend #legend { display: block; }
    body.show-list #toplist, body.show-list #search-box { display: flex; }
    #colorbar { display: none; }
  }
</style>
</head>
<body>
<div id="stage"></div>
<div id="toolbar" role="toolbar" aria-label="Panels">
  <button type="button" class="toggle" data-panel="legend" aria-pressed="false">Summary</button>
  <button type="button" class="toggle" data-panel="list" aria-pressed="false">Top __TOP_N__ &amp; search</button>
</div>
<section id="legend" class="panel" aria-label="Summary">
  <h1>Sharpe Skyline</h1>
  <div id="verdict">
    <div class="stat"><b id="stat-n">__N__</b><span>screened</span></div>
    <div class="stat"><b id="stat-pass">__PASS__</b><span>pass DSR &ge; 0.95</span></div>
    <div class="stat"><b id="stat-plane">__PLANE__</b><span>luck plane Sharpe (__WINDOW__)</span></div>
  </div>
  <p id="verdict-line">__VERDICT__</p>
  <div id="windows" role="group" aria-label="Window"><span class="caption">Window:</span></div>
  <p id="controls-hint" class="caption">Drag floor to move · drag sky to orbit · scroll to zoom · double-click a bar to fly to it · R resets.
    <button id="reset-view" type="button">Reset view</button></p>
  <details id="how-to-read" open>
    <summary>How to read a bar</summary>
    <ul>
      <li><strong>Height</strong> = annualized Sharpe; below the "Sharpe 0" level means negative.</li>
      <li><strong>Colour</strong> = Fama-French 3 alpha t-statistic (colour bar, bottom left).</li>
      <li><strong>White outline</strong> = top __TOP_N__ by Sharpe; <strong>glow + label</strong> = top __LABEL_TOP__.</li>
      <li><strong>Red plane</strong> = expected best-of-N Sharpe under a zero-alpha null. Above it is necessary, not sufficient.</li>
      <li>The skill test is <strong>DSR &ge; 0.95</strong>. Below it, a record cannot be told apart from luck at this sample size.</li>
      <li>Click a tower, a list row or search a ticker for details. Esc closes them.</li>
    </ul>
  </details>
  <details id="sector-legend">
    <summary>Sectors (floor tint)</summary>
    <div id="sectors">
__SECTOR_LEGEND__
    </div>
  </details>
  <p class="disclaimer">Historical statistics, not investment advice.</p>
</section>
<section id="colorbar" class="panel" aria-label="Colour bar">
  <div class="title">FF3 alpha t-statistic</div>
  <div class="bar" id="colorbar-ramp"></div>
  <div class="ticks"><span>&ge; 4</span><span>2</span><span>&le; 0</span></div>
</section>
<div id="side">
  <section id="search-box" class="panel" aria-label="Find a ticker">
    <label for="search" class="caption">Find any of the __UNIVERSE__ names (Enter to fly there)</label>
    <input id="search" type="search" list="tickers" autocomplete="off" spellcheck="false" placeholder="Ticker, e.g. __EXAMPLE__">
    <datalist id="tickers">
__TICKER_OPTIONS__
    </datalist>
    <span id="search-status" class="caption" aria-live="polite"></span>
  </section>
  <aside id="details" class="panel" hidden aria-live="polite" aria-label="Selected stock">
    <button id="details-close" type="button" aria-label="Close details">&times;</button>
    <h2 id="d-ticker"></h2>
    <div id="d-sector" class="caption"></div>
    <dl id="d-fields"></dl>
    <div id="d-chips"></div>
    <nav aria-label="Links and actions">
      <a id="link-gf" target="_blank" rel="noopener">Google Finance</a>
      <a id="link-edgar" target="_blank" rel="noopener">SEC EDGAR</a>
      <a id="link-web" target="_blank" rel="noopener"></a>
      <button id="fly-to" type="button">Fly to</button>
      <button id="copy-json" type="button">Copy JSON</button>
    </nav>
    <pre id="d-json" hidden></pre>
  </aside>
  <section id="toplist" class="panel" aria-label="Top names">
    <h2>Top __TOP_N__ by Sharpe (__WINDOW__)</h2>
    <ol>
__TOP_LIST__
    </ol>
  </section>
</div>
<div id="tooltip" class="panel"></div>
<script id="nav-logic">
// Pure page rules, free of three.js and the DOM so the tests can run them in Node.
globalThis.SkylineLinks = {
  // SEC EDGAR spells share classes with a dash (BRK.B -> BRK-B); Google Finance accepts the dot.
  forTicker(template, ticker, dashClasses) {
    const symbol = dashClasses ? ticker.replaceAll(".", "-") : ticker;
    return template.replace("{ticker}", encodeURIComponent(symbol));
  },
};
globalThis.SkylineNav = {
  // Keep the orbit target inside the scene bounds (three.js world axes: x, y up, z).
  clampTarget(p, bounds) {
    const clamp = (v, [lo, hi]) => Math.min(Math.max(v, lo), hi);
    return { x: clamp(p.x, bounds.x), y: clamp(p.y, bounds.y), z: clamp(p.z, bounds.z) };
  },
  // Which drag a press starts: left on the floor pans like a map, left elsewhere orbits,
  // right always pans; anything else is left to OrbitControls.
  dragMode(button, hit) {
    if (button === 2) return "pan";
    if (button === 0) return hit === "floor" ? "pan" : "orbit";
    return null;
  },
  // Ground-plane step for arrow keys / WASD, relative to where the camera looks.
  keyPan(key, forward, step) {
    const dirs = { ArrowUp: [0, 1], w: [0, 1], ArrowDown: [0, -1], s: [0, -1],
      ArrowLeft: [-1, 0], a: [-1, 0], ArrowRight: [1, 0], d: [1, 0] };
    const dir = dirs[key.length === 1 ? key.toLowerCase() : key];
    if (!dir) return null;
    const len = Math.hypot(forward.x, forward.z) || 1;
    const fx = forward.x / len, fz = forward.z / len;
    // Screen-right on the ground is forward x up = (-fz, fx).
    return { x: (dir[1] * fx - dir[0] * fz) * step, z: (dir[1] * fz + dir[0] * fx) * step };
  },
};
</script>
<script id="skyline-data" type="application/json">__SKYLINE_JSON__</script>
<script type="importmap">
{ "imports": { "three": "__THREE_BASE__/build/three.module.js", "three/addons/": "__THREE_BASE__/examples/jsm/" } }
</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const DATA = JSON.parse(document.getElementById("skyline-data").textContent);
const stage = document.getElementById("stage");
const css = (rgb) => `rgb(${rgb.map((c) => Math.round(c * 255)).join(",")})`;
const fmt = (v, d = 2) => (v === null || v === undefined ? "n/a" : v.toFixed(d));
const pct = (v) => (v === null || v === undefined ? "n/a" : `${(100 * v).toFixed(1)}%`);
const BG = 0x0b0d14;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
stage.appendChild(renderer.domElement);
const labelRenderer = new CSS2DRenderer();
labelRenderer.domElement.style.position = "absolute";
labelRenderer.domElement.style.top = "0";
labelRenderer.domElement.style.pointerEvents = "none";
stage.appendChild(labelRenderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(BG);
const span = Math.max(DATA.extent.width, DATA.extent.depth, 8);
const halfW = DATA.extent.width / 2 + 1.5;
const halfD = DATA.extent.depth / 2 + 1.5;
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, span * 20);
const viewDir = new THREE.Vector3(0.45, 0.8, 1.15).normalize();
const sceneRadius = Math.hypot(halfW, halfD, 4);
const home = new THREE.Vector3(0, 3, 0);
function framedPosition() {
  const vfov = THREE.MathUtils.degToRad(camera.fov);
  const hfov = 2 * Math.atan(Math.tan(vfov / 2) * camera.aspect);
  const distance = sceneRadius / Math.sin(Math.min(vfov, hfov) / 2);
  return viewDir.clone().multiplyScalar(distance).add(home);
}
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(home);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.screenSpacePanning = false;
scene.fog = new THREE.Fog(BG, sceneRadius * 2.2, sceneRadius * 5.5);

// Two lights and no shadows keep the cost flat at 500+ towers.
scene.add(new THREE.HemisphereLight(0xcfd8ff, 0x10131c, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(span * 0.6, span * 1.2, span * 0.8);
scene.add(sun);

function label(text, className, x, y, z) {
  const div = document.createElement("div");
  div.className = className;
  div.textContent = text;
  const obj = new CSS2DObject(div);
  obj.position.set(x, y, z);
  scene.add(obj);
  return obj;
}

// Zero level and per-sector floor tints.
scene.add(new THREE.GridHelper(span * 1.6, 32, 0x2a3042, 0x161a26));
scene.add(new THREE.Mesh(
  new THREE.PlaneGeometry(halfW * 2, halfD * 2).rotateX(-Math.PI / 2),
  new THREE.MeshBasicMaterial({ color: 0x9aa1b2, transparent: true, opacity: 0.05, side: THREE.DoubleSide, depthWrite: false }),
));
// Invisible pick target: a press on it starts a map-style floor drag.
const floorHit = new THREE.Mesh(
  new THREE.PlaneGeometry(span * 1.6, span * 1.6).rotateX(-Math.PI / 2),
  new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false, colorWrite: false }),
);
scene.add(floorHit);
for (const tile of DATA.sector_tiles) {
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(tile.w, tile.d).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({ color: tile.color, transparent: true, opacity: 0.12, depthWrite: false }),
  );
  mesh.position.set(tile.cx, 0.005, -tile.cy);
  scene.add(mesh);
}
label(DATA.zero_label, "ref zero", -halfW, 0, halfD);

const box = new THREE.BoxGeometry(DATA.tower_size, 1, DATA.tower_size);
box.translate(0, 0.5, 0);
const edges = new THREE.EdgesGeometry(box);
const towers = [];
const byTicker = new Map();
for (const t of DATA.towers) {
  const color = new THREE.Color().setRGB(...t.color, THREE.SRGBColorSpace);
  const glow = t.rank !== null && t.rank <= DATA.label_top;
  const material = new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.1, emissive: color.clone() });
  const mesh = new THREE.Mesh(box, material);
  mesh.position.set(t.x, 0, -t.y);
  mesh.userData = { tower: t, base: glow ? 0.75 : 0 };
  material.emissiveIntensity = mesh.userData.base;
  if (t.top) {
    mesh.add(new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.7 })));
  }
  if (glow) mesh.userData.label = label(t.ticker, "label", t.x, 0, -t.y);
  scene.add(mesh);
  towers.push(mesh);
  byTicker.set(t.ticker.toUpperCase(), mesh);
}
for (const s of DATA.sectors) label(s.name, "sector", s.x, 0, -s.y);

// Luck plane, slightly desaturated so labels stay readable through it.
const luckColor = new THREE.Color().setRGB(...DATA.luck_rgb, THREE.SRGBColorSpace).lerp(new THREE.Color(0x777777), 0.25);
const luck = new THREE.Mesh(
  new THREE.PlaneGeometry(halfW * 2, halfD * 2).rotateX(-Math.PI / 2),
  new THREE.MeshBasicMaterial({ color: luckColor, transparent: true, opacity: 0.26, side: THREE.DoubleSide, depthWrite: false }),
);
scene.add(luck);
const luckLabel = label("", "ref luck", 0, 0, halfD);

// Value ruler at the near-left corner, in Sharpe units.
const rulerX = -halfW - 0.8;
const rulerZ = halfD;
const ticks = DATA.ruler_ticks;
const rulerPoints = [new THREE.Vector3(rulerX, ticks[0] * DATA.scale, rulerZ),
  new THREE.Vector3(rulerX, ticks[ticks.length - 1] * DATA.scale, rulerZ)];
for (const v of ticks) {
  const y = v * DATA.scale;
  rulerPoints.push(new THREE.Vector3(rulerX - 0.3, y, rulerZ), new THREE.Vector3(rulerX + 0.3, y, rulerZ));
  label(String(v), "ref", rulerX - 0.9, y, rulerZ);
}
scene.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(rulerPoints), new THREE.LineBasicMaterial({ color: 0x9aa1b2 })));
const planeTick = new THREE.Line(
  new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(rulerX - 0.5, 0, rulerZ), new THREE.Vector3(rulerX + 0.5, 0, rulerZ)]),
  new THREE.LineBasicMaterial({ color: 0xe6403f }),
);
scene.add(planeTick);
const planeTickLabel = label("", "ref luck", rulerX - 1.6, 0, rulerZ);

let current = DATA.window;
function applyWindow(w) {
  current = w;
  for (const mesh of towers) {
    const h = mesh.userData.tower.height[w];
    mesh.scale.y = Math.max(Math.abs(h), 0.001);
    mesh.position.y = h < 0 ? h : 0;
    if (mesh.userData.label) mesh.userData.label.position.set(mesh.position.x, Math.max(h, 0) + 0.45, mesh.position.z);
  }
  const planeY = DATA.luck_height[w];
  luck.position.y = planeY;
  luckLabel.position.y = planeY;
  luckLabel.element.textContent = DATA.luck_label[w];
  planeTick.position.y = planeY;
  planeTickLabel.position.y = planeY;
  planeTickLabel.element.textContent = fmt(DATA.luck_horizon_ann[w]);
  for (const b of document.querySelectorAll("#windows button")) b.setAttribute("aria-pressed", String(b.dataset.window === w));
}

const stops = DATA.color_stops;
const tMax = stops[stops.length - 1][0];
document.getElementById("colorbar-ramp").style.background =
  `linear-gradient(0deg, ${stops.map(([t, c]) => `${css(c)} ${(100 * t) / tMax}%`).join(", ")})`;
for (const w of DATA.windows) {
  const b = document.createElement("button");
  b.type = "button";
  b.dataset.window = w;
  b.textContent = w;
  b.addEventListener("click", () => applyWindow(w));
  document.getElementById("windows").appendChild(b);
}
applyWindow(current);
if (window.innerWidth < 900) document.getElementById("how-to-read").open = false;

// Small screens: panels are overlays behind toolbar toggles.
for (const toggle of document.querySelectorAll(".toggle")) {
  toggle.addEventListener("click", () => {
    const cls = `show-${toggle.dataset.panel}`;
    const on = !document.body.classList.contains(cls);
    document.body.classList.toggle(cls, on);
    toggle.setAttribute("aria-pressed", String(on));
  });
}

// Selection: details panel, highlight, camera move.
const details = document.getElementById("details");
const selectionCage = new THREE.Group();
for (const s of [1.08, 1.16]) {
  const cage = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x6fd6ff }));
  cage.scale.set(s, 1.0, s);
  selectionCage.add(cage);
}
let selected = null;
let hovered = null;

function setGlow(mesh) {
  const boost = mesh === selected ? 1.4 : mesh === hovered ? Math.max(mesh.userData.base, 0.45) : mesh.userData.base;
  mesh.material.emissiveIntensity = boost;
}
function field(name, value) {
  const dt = document.createElement("dt");
  dt.textContent = name;
  const dd = document.createElement("dd");
  dd.textContent = value;
  return [dt, dd];
}
function chip(text, cls) {
  const span = document.createElement("span");
  span.className = `chip ${cls}`;
  span.textContent = text;
  return span;
}
function rowJson(t) {
  return Object.fromEntries(DATA.detail_fields.map((k) => [k, t[k] ?? null]));
}

function select(mesh, moveCamera) {
  const previous = selected;
  selected = mesh;
  if (previous) { previous.remove(selectionCage); setGlow(previous); }
  setGlow(mesh);
  mesh.add(selectionCage);
  const t = mesh.userData.tower;
  document.getElementById("d-ticker").textContent = t.ticker;
  document.getElementById("d-sector").textContent = t.sector;
  document.getElementById("d-fields").replaceChildren(
    ...field("Rank", t.rank === null ? "not ranked" : `#${t.rank} (${DATA.window})`),
    ...DATA.windows.flatMap((w) => field(`Sharpe ${w}`, fmt(t.sr_ann[w]))),
    ...field("CAPM alpha", `${pct(t.alpha_capm_ann)} (t ${fmt(t.t_alpha_capm)})`),
    ...field("FF3 alpha", `${pct(t.alpha_ff3_ann)} (t ${fmt(t.t_alpha_ff3)})`),
    ...field("PSR", fmt(t.psr, 3)),
    ...field("DSR", fmt(t.dsr, 3)),
    ...field(`Days of data (${DATA.window})`, String(t.n_obs)),
  );
  document.getElementById("d-chips").replaceChildren(
    chip(t.dsr_passes ? "DSR >= 0.95: pass" : "DSR >= 0.95: fail", t.dsr_passes ? "pass" : "fail"), " ",
    chip(t.above_luck_plane ? "above luck plane" : "below luck plane", t.above_luck_plane ? "above" : "fail"),
  );
  document.getElementById("link-gf").href = SkylineLinks.forTicker(DATA.links.google_finance, t.ticker, false);
  document.getElementById("link-edgar").href = SkylineLinks.forTicker(DATA.links.edgar, t.ticker, true);
  const web = document.getElementById("link-web");
  web.href = t.website || SkylineLinks.forTicker(DATA.links.website_search, t.ticker, false);
  web.textContent = t.website ? "Company website" : "Website (search)";
  document.getElementById("copy-json").textContent = "Copy JSON";
  document.getElementById("d-json").hidden = true;
  details.hidden = false;
  for (const row of document.querySelectorAll("#toplist button")) {
    row.setAttribute("aria-current", String(row.dataset.ticker === t.ticker));
  }
  if (moveCamera) flyToTower(mesh);
}

function clearSelection() {
  if (!selected) return;
  const mesh = selected;
  selected = null;
  mesh.remove(selectionCage);
  setGlow(mesh);
  details.hidden = true;
  for (const row of document.querySelectorAll("#toplist button")) row.setAttribute("aria-current", "false");
}
document.getElementById("details-close").addEventListener("click", clearSelection);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    clearSelection();
    if (document.activeElement === searchInput) searchInput.blur();
    return;
  }
  if (document.activeElement === searchInput || event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.key === "r" || event.key === "R") { resetView(); return; }
  const forward = controls.target.clone().sub(camera.position);
  const step = SkylineNav.keyPan(event.key, { x: forward.x, z: forward.z }, span * 0.04);
  if (!step) return;
  event.preventDefault();
  flight = null;
  moveView(new THREE.Vector3(step.x, 0, step.z));
});
// The async Clipboard API is missing on insecure origins, so fall back to execCommand and,
// failing that, show the JSON for manual copying.
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const area = Object.assign(document.createElement("textarea"), { value: text });
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand("copy");
    area.remove();
    return ok;
  }
}
document.getElementById("copy-json").addEventListener("click", async (event) => {
  if (!selected) return;
  const text = JSON.stringify(rowJson(selected.userData.tower), null, 2);
  const shown = document.getElementById("d-json");
  if (await copyText(text)) {
    event.target.textContent = "Copied";
    shown.hidden = true;
  } else {
    shown.textContent = text;
    shown.hidden = false;
  }
});

let flight = null;
function flyTo(toTarget, toCamera, ms) {
  flight = { start: performance.now(), ms, fromTarget: controls.target.clone(), fromCamera: camera.position.clone(),
    toTarget, toCamera };
}
function flyToTower(mesh) {
  const h = Math.abs(mesh.userData.tower.height[current]);
  const target = new THREE.Vector3(mesh.position.x, mesh.position.y + h / 2, mesh.position.z);
  const offset = camera.position.clone().sub(controls.target).normalize().multiplyScalar(Math.max(10, h * 2.5));
  flyTo(target, target.clone().add(offset), 700);
}

function moveView(delta) {
  camera.position.add(delta);
  controls.target.add(delta);
  clampView();
}
function clampView() {
  const t = controls.target;
  const c = SkylineNav.clampTarget(t, DATA.nav_bounds);
  const correction = new THREE.Vector3(c.x - t.x, c.y - t.y, c.z - t.z);
  if (correction.lengthSq() === 0) return;
  t.add(correction);
  camera.position.add(correction);
}
function resetView() {
  flyTo(home.clone(), framedPosition(), 700);
}
document.getElementById("reset-view").addEventListener("click", resetView);
document.getElementById("fly-to").addEventListener("click", () => { if (selected) flyToTower(selected); });

for (const row of document.querySelectorAll("#toplist button")) {
  row.addEventListener("click", () => {
    const mesh = byTicker.get(row.dataset.ticker.toUpperCase());
    if (mesh) select(mesh, true);
  });
}

const searchInput = document.getElementById("search");
const searchStatus = document.getElementById("search-status");
searchInput.addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  const query = searchInput.value.trim().toUpperCase();
  if (!query) return;
  const mesh = byTicker.get(query)
    || towers.find((m) => m.userData.tower.ticker.toUpperCase().startsWith(query));
  if (!mesh) { searchStatus.textContent = `No ticker matches "${searchInput.value.trim()}".`; return; }
  searchStatus.textContent = "";
  searchInput.value = mesh.userData.tower.ticker;
  select(mesh, true);
});

const tooltip = document.getElementById("tooltip");
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
function towerAt(event) {
  aim(event);
  const hit = raycaster.intersectObjects(towers, false)[0];
  return hit ? hit.object : null;
}
renderer.domElement.addEventListener("pointermove", (event) => {
  const mesh = towerAt(event);
  if (mesh !== hovered) {
    const previous = hovered;
    hovered = mesh;
    if (previous) setGlow(previous);
    if (mesh) setGlow(mesh);
  }
  renderer.domElement.style.cursor = mesh ? "pointer" : "";
  if (!mesh) { tooltip.style.display = "none"; return; }
  const t = mesh.userData.tower;
  const lines = [
    `${t.ticker}${t.rank ? `  #${t.rank} (${DATA.window})` : ""}`,
    t.sector,
    `Sharpe (${current}): ${fmt(t.sr_ann[current])}`,
    `DSR: ${fmt(t.dsr, 3)} (${t.dsr_passes ? "passes" : "below"} 0.95)`,
    "Click for details",
  ];
  tooltip.replaceChildren(...lines.map((text, i) => {
    const div = document.createElement("div");
    div.textContent = text;
    if (i === 0) div.style.fontWeight = "600";
    return div;
  }));
  tooltip.style.display = "block";
  tooltip.style.left = `${Math.min(event.clientX + 14, window.innerWidth - tooltip.offsetWidth - 8)}px`;
  tooltip.style.top = `${Math.min(event.clientY + 14, window.innerHeight - tooltip.offsetHeight - 8)}px`;
});
// A click is a press and release without dragging. A left press on the floor starts a
// map-style drag that keeps the grabbed ground point under the cursor; any other drag is
// OrbitControls' (left orbits, right pans parallel to the ground, wheel zooms, touch gestures).
const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
let pressedAt = null;
let drag = null;
function aim(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
}
function hitKind(event) {
  aim(event);
  const hit = raycaster.intersectObjects([...towers, floorHit], false)[0];
  if (!hit) return null;
  return hit.object === floorHit ? "floor" : "tower";
}
function groundPoint(event) {
  aim(event);
  return raycaster.ray.intersectPlane(groundPlane, new THREE.Vector3());
}
// A single click opens the panel after a short pause, so the second click of a double-click
// still lands on the canvas rather than on a panel that just appeared under the cursor.
const DOUBLE_CLICK_MS = 250;
let clickTimer = null;
function handleClick(event) {
  const start = pressedAt;
  pressedAt = null;
  if (!start || Math.hypot(event.clientX - start[0], event.clientY - start[1]) > 5) return;
  const mesh = towerAt(event);
  clearTimeout(clickTimer);
  if (mesh) clickTimer = setTimeout(() => select(mesh, false), DOUBLE_CLICK_MS);
  else clearSelection();
}
stage.addEventListener("pointerdown", (event) => {
  pressedAt = [event.clientX, event.clientY];
  if (event.pointerType !== "mouse" || SkylineNav.dragMode(event.button, hitKind(event)) !== "pan" || event.button !== 0) return;
  const start = groundPoint(event);
  if (!start) return;
  event.stopPropagation();  // keep OrbitControls from also handling this drag
  flight = null;
  drag = { start };
  stage.setPointerCapture(event.pointerId);
  renderer.domElement.style.cursor = "grabbing";
}, { capture: true });
stage.addEventListener("pointermove", (event) => {
  if (!drag) return;
  const now = groundPoint(event);
  if (!now) return;
  const delta = drag.start.clone().sub(now);
  delta.y = 0;
  moveView(delta);
});
stage.addEventListener("pointerup", (event) => {
  if (!drag) return;
  drag = null;
  renderer.domElement.style.cursor = "";
  handleClick(event);
});
renderer.domElement.addEventListener("pointerup", handleClick);
renderer.domElement.addEventListener("dblclick", (event) => {
  const mesh = towerAt(event);
  clearTimeout(clickTimer);
  if (mesh) select(mesh, true);
});

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h);
  labelRenderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();
// Gentle intro: start higher and further out, then settle on the framed view over 1.5 s.
const framed = framedPosition();
camera.position.copy(framed.clone().sub(home).multiplyScalar(1.35).add(new THREE.Vector3(0, sceneRadius * 0.4, 0)).add(home));
flyTo(home.clone(), framed, 1500);
renderer.setAnimationLoop((now) => {
  if (flight) {
    const k = Math.min((now - flight.start) / flight.ms, 1);
    const e = k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2;
    controls.target.lerpVectors(flight.fromTarget, flight.toTarget, e);
    camera.position.lerpVectors(flight.fromCamera, flight.toCamera, e);
    if (k === 1) flight = null;
  }
  controls.update();
  clampView();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
});
</script>
</body>
</html>
"""


def _horizon_words(window: str) -> str:
    unit = {"y": "year", "d": "day"}.get(window[-1:], "")
    return f"{window[:-1]}-{unit}" if unit and window[:-1].isdigit() else window


def verdict_line(scene: dict[str, Any]) -> str:
    """One plain-English sentence summarising the screen for the selected window."""
    span = _horizon_words(scene["window"])
    n, k, above, top_n = scene["N"], scene["top_dsr_passing"], scene["top_above_luck_plane"], scene["top_n"]
    if k:
        passing = [t["ticker"] for t in sorted(scene["towers"], key=lambda t: t["rank"] or math.inf) if t["dsr_passes"]]
        names = ", ".join(passing[:3]) + (f" and {k - 3} more" if k > 3 else "")
        verb = "clears" if k == 1 else "clear"
        return (f"{names} {verb} the luck test (DSR ≥ 0.95): a {span} record unlikely to be just "
                f"the luckiest of {n} names.")
    if above:
        noun = "name rises" if above == 1 else "names rise"
        return (f"{above} of the top {top_n} {noun} above the luck plane, but none clears DSR ≥ 0.95: "
                f"no {span} record can yet be told apart from luck across {n} names.")
    return f"No name's {span} record beats what luck alone would hand the best of {n}."


def sector_tiles(scene: dict[str, Any]) -> list[dict[str, Any]]:
    """A faint floor rectangle per sector block, with its tint."""
    half = scene["tower_size"] / 2 + TILE_MARGIN
    tiles = []
    for i, sector in enumerate(scene["sectors"]):
        members = [t for t in scene["towers"] if t["sector"] == sector["name"]]
        x0, x1 = min(t["x"] for t in members) - half, max(t["x"] for t in members) + half
        y0, y1 = min(t["y"] for t in members) - half, max(t["y"] for t in members) + half
        tiles.append({"name": sector["name"], "color": SECTOR_TINTS[i % len(SECTOR_TINTS)],
                      "cx": round((x0 + x1) / 2, 4), "cy": round((y0 + y1) / 2, 4),
                      "w": round(x1 - x0, 4), "d": round(y1 - y0, 4)})
    return tiles


def nav_bounds(scene: dict[str, Any], low: int, high: int) -> dict[str, list[float]]:
    """Box the orbit target may move in, in three.js world axes (z = -scene y): the floor
    (scene extent plus the page's 1.5-unit border) plus NAV_MARGIN, and the ruler's Sharpe range."""
    half_w = scene["extent"]["width"] / 2 + 1.5 + NAV_MARGIN
    half_d = scene["extent"]["depth"] / 2 + 1.5 + NAV_MARGIN
    return {"x": [round(-half_w, 4), round(half_w, 4)],
            "y": [round(low * scene["scale"] - 1.0, 4), round(high * scene["scale"] + 1.0, 4)],
            "z": [round(-half_d, 4), round(half_d, 4)]}


def page_data(scene: dict[str, Any]) -> dict[str, Any]:
    """The scene plus what only the page needs: links, labels, ruler ticks, verdict and sector tints."""
    values = [v for t in scene["towers"] for v in t["sr_ann"].values() if v is not None]
    values += [v for v in scene["luck_horizon_ann"].values() if v is not None] + [0.0]
    low, high = math.floor(min(values)), math.ceil(max(values))
    plane = scene["luck_horizon_ann"][scene["window"]]
    return {
        **scene,
        "label_top": LABEL_TOP,
        "links": LINK_TEMPLATES,
        "detail_fields": DETAIL_FIELDS,
        "zero_label": "Sharpe 0",
        "luck_label": {w: f"Luck plane: Sharpe {v:.2f} ({w})" for w, v in scene["luck_horizon_ann"].items() if v is not None},
        "ruler_ticks": list(range(low, high + 1)),
        "nav_bounds": nav_bounds(scene, low, high),
        "verdict": {
            "n": scene["N"],
            "pass": f"{scene['top_dsr_passing']} of {scene['top_n']}",
            "plane": "n/a" if plane is None else f"{plane:.2f}",
            "line": verdict_line(scene),
        },
        "sector_tiles": sector_tiles(scene),
    }


def top_list_html(scene: dict[str, Any]) -> str:
    """Server-rendered rows of the top-N list (the page script makes them clickable)."""
    window = scene["window"]
    top = sorted((t for t in scene["towers"] if t["top"]), key=lambda t: t["rank"])
    rows = []
    for t in top:
        ticker = html.escape(t["ticker"])
        dsr = "n/a" if t["dsr"] is None else f"{t['dsr']:.2f}"
        rows.append(
            f'      <li><button type="button" data-ticker="{ticker}"><span class="rank">{t["rank"]}</span>'
            f'<span class="tk">{ticker}</span><span class="sr">{t["sr_ann"][window]:.2f}</span>'
            f'<span class="chip {"pass" if t["dsr_passes"] else "fail"}">DSR {dsr}</span></button></li>'
        )
    return "\n".join(rows)


def ticker_options_html(scene: dict[str, Any]) -> str:
    tickers = sorted(t["ticker"] for t in scene["towers"])
    return "\n".join(f'      <option value="{html.escape(t)}"></option>' for t in tickers)


def sector_legend_html(tiles: list[dict[str, Any]]) -> str:
    return "\n".join(
        f'      <span><i class="swatch" style="background:{tile["color"]}"></i>{html.escape(tile["name"])}</span>'
        for tile in tiles
    )


def render_page(scene: dict[str, Any]) -> str:
    data = page_data(scene)
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    leader = min((t for t in scene["towers"] if t["rank"] is not None), key=lambda t: t["rank"], default=None)
    replacements = {
        "__THREE_BASE__": THREE_BASE,
        "__LABEL_TOP__": str(LABEL_TOP),
        "__TOP_N__": str(scene["top_n"]),
        "__N__": str(scene["N"]),
        "__UNIVERSE__": str(len(scene["towers"])),
        "__PASS__": data["verdict"]["pass"],
        "__PLANE__": data["verdict"]["plane"],
        "__VERDICT__": html.escape(data["verdict"]["line"]),
        "__WINDOW__": html.escape(scene["window"]),
        "__EXAMPLE__": html.escape(leader["ticker"] if leader else ""),
        "__SECTOR_LEGEND__": sector_legend_html(data["sector_tiles"]),
        "__TICKER_OPTIONS__": ticker_options_html(scene),
        "__TOP_LIST__": top_list_html(scene),
    }
    page = PAGE
    for key, value in replacements.items():
        page = page.replace(key, value)
    return page.replace("__SKYLINE_JSON__", payload)


def write_page(result_path: Path, out: Path, open_browser: bool = False) -> Path:
    """Write the page for ``result_path`` to ``out``; optionally open it in the default browser."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_page(build_scene(load_result(result_path))), encoding="utf-8")
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a single-file three.js skyline page.")
    parser.add_argument("result", type=Path, help="screen_result.json from screen.py")
    parser.add_argument("--out", type=Path, help="default: skyline.html next to the result")
    parser.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = parser.parse_args(argv)
    out = write_page(args.result, args.out or args.result.parent / "skyline.html", args.open)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
