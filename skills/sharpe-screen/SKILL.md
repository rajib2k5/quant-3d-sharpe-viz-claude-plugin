---
name: sharpe-screen
description: Screen a stock universe by Sharpe ratio and Jensen's alpha (CAPM and Fama-French 3) with a multiple-testing luck correction (Probabilistic Sharpe, expected maximum Sharpe, Deflated Sharpe). Use when the user asks which stocks have the best risk-adjusted returns, wants a Sharpe or alpha screen, asks whether a high Sharpe ratio could be luck, or mentions deflated Sharpe, PSR, DSR or the False Strategy Theorem. Runs offline on bundled sample data unless the user explicitly asks for live data.
---

# Sharpe screen with a luck correction

Screens every name in a universe by annualized Sharpe ratio over a trailing window, estimates CAPM and Fama-French 3 alphas with t-statistics, and computes the **luck plane**: the expected best-of-N Sharpe under a zero-alpha null in which every name has the universe's average Sharpe (False Strategy Theorem). Each name also gets a Probabilistic Sharpe Ratio (PSR, against zero) and a Deflated Sharpe Ratio (DSR, against the luck plane). The skill test is DSR >= 0.95.

## When to use

- "Screen these stocks by Sharpe" / "which names have the best risk-adjusted returns"
- "Is this Sharpe ratio real or luck?" / "deflate the Sharpe ratios"
- "Which stocks have significant alpha against the Fama-French factors?"
- As the first step before the `sharpe-skyline` visualization.

## How to run

The scripts need Python 3.10+ with the packages in `${CLAUDE_PLUGIN_ROOT}/requirements.txt` (`pip install -r` it if an import fails).

Offline, on the bundled synthetic sample (default; no network):

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/screen.py" --out out/
```

Options: `--window 252|756|1260` (1y, 3y, 5y trading days; default 756), `--top 20`, `--out DIR`, `--data-dir DIR` (a dataset in the sample format, for example one cached by `fetch_data.py --live`). The sample data can be regenerated deterministically with `python "${CLAUDE_PLUGIN_ROOT}/scripts/make_sample_data.py"`.

Outputs: `out/screen_result.json` (every name, plus metadata) and `out/report.md` (ranked top-N table, luck-horizon callout, method notes, disclaimer).

### Live data (`--live`) and network disclosure

Only when the user explicitly asks for real market data, and after telling them which hosts will be contacted:

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/screen.py" --live --save-universe out/sp500.csv --out out/
```

By default `--live` fetches the current S&P 500 constituents itself (about 500 names) and says so on stderr. A full run takes very roughly 8 to 10 minutes; a 10-name test took about 10 seconds. Tell the user before starting, and use `--limit-universe 10` for a quick check. `--save-universe PATH` writes the list used, so it can be reused. If the user has their own list, pass `--universe path/to/universe.csv` instead; the S&P 500 list is then not fetched.

`--live` contacts these sources and nothing else: **raw.githubusercontent.com** (the S&P 500 list from the open datasets/s-and-p-500-companies project; skipped with `--universe`), **en.wikipedia.org** (fallback only, if that list is unavailable, malformed or has fewer than 400 rows), **Yahoo Finance** (`query1`/`query2.finance.yahoo.com`, daily adjusted closes through the unofficial yfinance library, one ticker at a time, at least `--throttle` seconds apart (default 0.5) and slower automatically under backoff; yfinance may also touch `fc.yahoo.com`, `finance.yahoo.com` or Yahoo's consent pages for a session cookie), **fred.stlouisfed.org** (3-month Treasury bill rate, DGS3MO) and **mba.tuck.dartmouth.edu** (Ken French Fama-French 3 daily factors). A `--universe` CSV needs a ticker column (`ticker` or `Symbol`) and optionally a sector column (`sector` or `GICS Sector`) and a `website` column (an http/https URL per company, used by the skyline page's "Company website" link). If both S&P 500 sources fail, the run stops with `could not fetch the S&P 500 constituents`; offer `--universe` or the offline sample. Backoff is automatic: transient failures (timeouts, dropped connections, HTTP 429 or 500-504) are retried after 4 s and then 15 s, and the pause between tickers doubles (up to 8 s) after trouble and relaxes back to the `--throttle` floor after 20 clean fetches. Raise `--throttle` only if a run still hits the abort rules. Tickers that still fail are skipped and listed on stderr. The run stops with an error after 5 consecutive failures, or once at least 20 tickers have failed and more than half of those attempted failed; that usually means Yahoo is blocking or rate-limiting. Tell the user that yfinance is an unofficial API, do not try to get around a block, and offer the offline sample instead. If the error says `yfinance is not installed`, that is a local setup problem, not a block: run `pip install -r "${CLAUDE_PLUGIN_ROOT}/requirements.txt"` and retry.

## How to present the results

Read `out/report.md` and `out/screen_result.json`, then lead with three facts, in this order:

1. **N**, the number of names screened (`metadata.N`), and the window.
2. **How many of the top 20 have DSR >= 0.95** (`metadata.top_dsr_passing`). This is the skill test.
3. **The luck plane** (`metadata.luck_horizon_ann.headline`, annualized Sharpe) and how many of the top 20 rise above it (`metadata.top_above_luck_plane`), with the caveat that under the null the single best name exceeds the plane a little under half the time, so being above it is necessary, not sufficient.

Then show the top of the table. For names that rank high but have DSR below 0.95, say their record cannot be distinguished from luck at this sample size; that is low power, not proof of luck. The plane's variance is the larger of the cross-sectional and analytic estimates (`luck_horizon_ann.headline_variance` says which); mention the two variants only as a cross-check, and do not call the bar conservative beyond that choice.

After the table, offer the 3D skyline (the `sharpe-skyline` skill): a PNG plus an interactive page where the user can click any tower for its details and links to Google Finance, SEC EDGAR and the company website.

Rules:

- Never present the ranking as a recommendation, a buy list or a forecast. Say "historically had the highest Sharpe ratio", not "best stocks to buy".
- Always include the not-investment-advice line from the report.
- In sample mode, say the tickers are synthetic (sector prefix + number) and the data is simulated.
- Mention the main limitations when relevant: past Sharpe does not predict future Sharpe, current constituent lists carry survivorship bias, and classic OLS standard errors ignore heteroskedasticity and autocorrelation.
