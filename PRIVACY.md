# Privacy policy

Applies to the Sharpe Observatory Claude Code plugin (`sharpe-observatory`), source at https://github.com/rajib2k5/quant-3d-sharpe-viz-claude-plugin. Last updated 2026-10-12.

## What the plugin collects

Nothing. The plugin has no telemetry, analytics, accounts, sign-in or API keys. It does not collect, store or transmit any personal data, usage data, prompts or results to the author or to any third party.

## Where your data stays

Everything the plugin produces stays in local files in a folder you choose (`out/` by default): the screen result JSON, the report, the PNG skyline, the Blender script and the interactive HTML page. Optional caches stay local too: `--cache-dir` price files, and the files written by `fetch_data.py --live --out-dir`. A universe CSV you pass with `--universe` is read locally and never uploaded.

## Network requests

**Offline by default.** With the bundled sample data, the scripts make no network requests.

With `--live`, the scripts download public market data directly from these services, with no intermediary. Each request carries only what the service needs to return public data, such as a ticker symbol or a data-series name. No user data is sent.

| Service | Hosts | What is requested | When |
|---|---|---|---|
| Open datasets project on GitHub | `raw.githubusercontent.com` | S&P 500 constituents list | `--live` with the default S&P 500 index and no `--universe` |
| Wikipedia | `en.wikipedia.org` | S&P 500 constituents table | Fallback only, if the GitHub list is unavailable |
| Nasdaq Trader | `www.nasdaqtrader.com` | US symbol directory files | `--live --index us-all` only |
| Yahoo Finance (through the open-source yfinance library) | `query1.finance.yahoo.com`, `query2.finance.yahoo.com`; for a session cookie, possibly `fc.yahoo.com`, `finance.yahoo.com`, `guce.yahoo.com`, `consent.yahoo.com` | Daily prices for each ticker screened | Every `--live` run |
| FRED, Federal Reserve Bank of St. Louis | `fred.stlouisfed.org` | 3-month Treasury bill rate (DGS3MO) | Every `--live` run |
| Kenneth R. French Data Library | `mba.tuck.dartmouth.edu` | Fama-French 3 daily factors | Every `--live` run |

yfinance keeps a small local cache of time zones and a session cookie in a `py-yfinance` folder under your user cache directory, for example `~/.cache/py-yfinance` on Linux. It stays on your machine.

## The interactive page

The three.js page (`skyline.html`) loads the three.js library from `cdn.jsdelivr.net` when you open it, as any web page loading a script from a CDN does. Its details panel has links to Google Finance (`www.google.com`), SEC EDGAR (`www.sec.gov`) and a company website: either the `website` URL from your own universe CSV, or a Google search. These links open in a new tab **only when you click them**. The page sends nothing anywhere by itself.

## Third-party services

The services above have their own terms and privacy policies, which govern their data. yfinance is an unofficial, community-maintained client for Yahoo Finance, not affiliated with Yahoo.

## Changes and contact

Changes to this policy are recorded in this file's history in the repository. For questions, see [SUPPORT.md](SUPPORT.md).
