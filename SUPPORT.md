# Support

Sharpe Observatory is maintained by Rajib Bahar on a best-effort basis.

## Where to ask

- **Bugs and feature requests:** open an issue at https://github.com/rajib2k5/quant-3d-sharpe-viz-claude-plugin/issues
- **Email:** rajib.bahar@gmail.com, for questions that don't fit a public issue.

## What to include in a bug report

- What you ran: the prompt you gave Claude, or the script command line with its flags (for example `screen.py --live --index us-all --cache-dir …`).
- Whether it was the offline sample or `--live`, and the window and index used.
- The full error text, plus the last lines of stderr. The `[fetch]`, `[clean]` and `[render]` lines are especially useful.
- Your environment: OS and CPU architecture, Python version, Claude Code version, and `pip show numpy pandas scipy yfinance matplotlib` versions.
- For rendering problems: which tier was used (the `rendered with tier …` line), and a screenshot if the page or PNG looks wrong.
- Never include passwords, API keys or account details. The plugin doesn't need any.

## Scope

- Supported: installing and running the plugin, the statistics as documented in the README's "The math" section, the data pipeline, and the renderers.
- Not provided: investment advice of any kind. The plugin is an educational statistics tool, and no ranking, Sharpe ratio, alpha or DSR it reports is a recommendation.
- Data availability from Yahoo Finance, FRED, the Ken French library, Nasdaq Trader or the S&P 500 list sources is outside this project's control. When a source changes or blocks requests, the plugin reports a clear error rather than working around it.
