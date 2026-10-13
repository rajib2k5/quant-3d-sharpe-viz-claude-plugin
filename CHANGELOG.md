# Changelog

All notable changes to this project are recorded here. Versions follow [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-12

Initial release. Nothing was published before this version, so all development to date, including the "By Sharpe" layout toggle, is folded into 0.1.0.

### Screener and statistics
- Annualized Sharpe ratio over 1-, 3- and 5-year windows, with a 90% coverage rule.
- CAPM and Fama-French 3 Jensen's alpha with classic OLS t-statistics.
- Probabilistic Sharpe Ratio (Bailey & López de Prado 2012), using Pearson kurtosis.
- Luck plane: expected best-of-N Sharpe from the False Strategy Theorem, centred on the universe's mean Sharpe, with V[SR] = max(cross-sectional, analytic).
- Deflated Sharpe Ratio against the luck plane, with DSR >= 0.95 as the skill test.
- Data-quality rule: a covered name is left unscored, unranked and out of the luck plane when its raw daily returns in the window have zero variance ("flat price") or are more than 50% exactly zero ("stale quotes"), or when its Sharpe is not finite. Closes <= 0 are treated as missing.
- Exclusions are reported: `excluded_unscoreable` (ticker -> reason; this replaces the earlier development-only `excluded_nonfinite`) and `excluded_coverage` in the result JSON, plus an Exclusions section in `report.md`.
- Outputs: `screen_result.json`, plus `report.md` with a ranked table, the luck-plane callout, method notes and a not-investment-advice disclaimer.

### Data
- Offline by default, on a deterministic synthetic sample of 60 names.
- `--live` with the S&P 500 (default), fetched from the open datasets project with a Wikipedia fallback.
- `--live --index us-all`: every US-listed common stock from the Nasdaq Trader symbol directory. ETFs, test issues, `$` symbols, NextShares, and warrants, units, rights, notes, preferred and plain depositary shares are excluded; ADRs are kept. Towers are grouped by listing exchange.
- `--universe` with any CSV; an optional `website` column is passed through to the page.
- Prices come from Yahoo Finance through yfinance:
  - batched 50 per call, with a per-ticker second pass;
  - retries with backoff and an adaptive throttle (`--throttle` sets the floor);
  - two abort rules, and a resumable `--cache-dir`;
  - closes <= 0 are treated as missing.
- Risk-free rate from FRED (DGS3MO); Fama-French 3 daily factors from the Ken French library.

### Rendering
- `render_skyline.py` picks the best renderer available: the bpy module, then a Blender binary, then matplotlib. matplotlib always works and needs no installation.
- A self-contained Blender scene script, with window animation and chunked parts for Blender MCP sessions.
- An interactive three.js page:
  - verdict strip and click-to-inspect details, with Google Finance, SEC EDGAR and website links;
  - ticker search and copy-as-JSON;
  - map-style drag on the floor or towers, orbit kept above the horizon, zoom limits, fly-to and a reset view;
  - a labelled luck plane, zero level and value ruler, and a colour bar;
  - a "By sector" / "By Sharpe" layout toggle;
  - zoom-aware ticker labels: the top 20 always, plus up to 100 in view near the camera, with fade distances scaled to the scene and a Labels toggle; labels are decluttered in screen space (selected tower first, then rank, then nearest), so they never overlap at any zoom;
  - instanced rendering for 5,000+ names.

### Plugin
- Two skills: `sharpe-screen` and `sharpe-skyline`.
- A self-hosted marketplace (`quant-3d-sharpe-viz`).
- Privacy policy (PRIVACY.md), support guide (SUPPORT.md), and a generated icon (`scripts/make_icon.py`), also installed as the listing icon at `.claude-plugin/icon.png`.
- No environment variables are read: the Blender binary is found with `--blender PATH` or on PATH.
- The bundled sample prices ship as `data/sample/prices-1.csv` … `prices-4.csv` (each under 200 KiB); `load_directory` joins the parts and still accepts a single `prices.csv`.
