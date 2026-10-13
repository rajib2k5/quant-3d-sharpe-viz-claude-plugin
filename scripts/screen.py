"""Screen a stock universe by Sharpe ratio and Jensen's alpha, with a multiple-testing luck horizon.

Usage: python screen.py [--live [--universe U.csv] [--save-universe U.csv]] [--data-dir DIR]
                        [--window 756] [--top 20] [--out out/]

Statistics are documented in metrics.py: annualized Sharpe (Sharpe 1994), Probabilistic
Sharpe Ratio (Bailey & López de Prado 2012), expected maximum Sharpe under the null and the
Deflated Sharpe Ratio (Bailey, Borwein, López de Prado & Zhu 2014; López de Prado 2020,
Machine Learning for Asset Managers, ch. 8 and App. B; Bailey & López de Prado 2014, "The
Deflated Sharpe Ratio"), CAPM and Fama-French 3 alphas (Jensen 1968; Fama & French 1993).

Skill test: a name passes when its DSR >= 0.95 (López de Prado's threshold). The DSR
benchmark, also drawn as the "luck plane", is the expected best-of-N per-period Sharpe under a
zero-alpha null in which every one of the N screened names has the universe's mean Sharpe:

    E[max SR_N] = mean_SR + sqrt(V) * ((1 - g) Z(1 - 1/N) + g Z(1 - 1/(N e)))

V is the larger of two estimates of the variance of a per-period SR: the cross-sectional
variance of the observed SRs, and the analytic estimation variance (1 + mean_SR^2 / 2) / n
with n the median observation count. Taking the larger makes V the more conservative of
these two estimates; it does not make the bar conservative in any wider sense. Both
single-estimate variants are reported alongside. Under the null the single best name still
exceeds E[max SR_N] a little under half the time, so rising above the plane is necessary
but not sufficient; DSR >= 0.95 is the test.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd

import metrics
from fetch_data import INDEXES, SAMPLE_DIR, THROTTLE_S, Dataset, fetch_live, live_universe, load_directory

DSR_THRESHOLD = 0.95
REPORT_WARNING = ("> **Not financial advice.** Historical statistics for research and education only; not a "
                  "recommendation to buy, sell or hold any security. Past Sharpe ratios and alphas do not predict future results.")


def raw_returns(dataset: Dataset) -> pd.DataFrame:
    """Daily total return on the price calendar (NaN where a name has no price)."""
    return dataset.prices.pct_change(fill_method=None).iloc[1:]


def excess_returns(dataset: Dataset) -> pd.DataFrame:
    """Daily total return minus daily risk-free, on the price calendar (NaN where a name has no price)."""
    returns = raw_returns(dataset)
    calendar = dataset.prices.index
    rf = dataset.rf_daily.reindex(dataset.rf_daily.index.union(calendar)).ffill().reindex(returns.index)
    return returns.sub(rf, axis=0)


STALE_ZERO_SHARE = 0.5


def unscoreable_reason(raw_window: np.ndarray, sr: float) -> str | None:
    """Data-quality rule (an implementation choice, not from the papers): a name that meets the
    coverage rule is still not scored in a window when its raw daily returns there have zero
    variance ("flat price"), when more than STALE_ZERO_SHARE of them are exactly zero ("stale
    quotes": halted or unrefreshed prices), or when its Sharpe is not finite. Excess returns of
    such a name are just minus the risk-free rate, whose tiny variance makes the Sharpe explode."""
    if raw_window.size and np.ptp(raw_window) == 0:
        return "flat price"
    if raw_window.size and np.mean(raw_window == 0) > STALE_ZERO_SHARE:
        return "stale quotes"
    if not math.isfinite(sr):
        return "non-finite Sharpe"
    return None


def window_labels(selected: int) -> dict[str, int]:
    labels = dict(metrics.WINDOWS)
    if selected not in labels.values():
        labels[f"{selected}d"] = selected
    return dict(sorted(labels.items(), key=lambda item: item[1]))


def luck_horizon(per_period_srs: list[float], n_obs: list[int]) -> dict[str, Any]:
    """Expected best-of-N Sharpe under the mean-centred null, per-period and annualized.

    ``headline`` uses V = max(cross-sectional variance, analytic variance); the two
    single-estimate variants share the same centre and differ only in V.
    """
    n_trials = len(per_period_srs)
    if n_trials < 2:
        return {"n_trials": n_trials, "headline": None, "cross_sectional": None, "analytic_null": None}
    center = float(np.mean(per_period_srs))
    median_n = int(median(n_obs))
    var_cross = float(np.var(per_period_srs, ddof=1))
    var_null = metrics.null_sharpe_variance(median_n, center)
    var_headline = max(var_cross, var_null)
    scale = math.sqrt(metrics.TRADING_DAYS)
    headline = metrics.expected_max_sharpe(n_trials, center, var_headline)
    cross = metrics.expected_max_sharpe(n_trials, center, var_cross)
    null = metrics.expected_max_sharpe(n_trials, center, var_null)
    return {
        "n_trials": n_trials,
        "headline": headline * scale,
        "headline_variance": "cross_sectional" if var_cross >= var_null else "analytic_null",
        "cross_sectional": cross * scale,
        "analytic_null": null * scale,
        "center_sr_ann": center * scale,
        "headline_per_period": headline,
        "cross_sectional_per_period": cross,
        "analytic_null_per_period": null,
        "center_sr_per_period": center,
        "var_sr_cross_sectional": var_cross,
        "var_sr_analytic_null": var_null,
        "median_n_obs": median_n,
    }


def _alpha_fields(prefix: str, result: metrics.OLSResult | None) -> dict[str, float | None]:
    if result is None:
        return {f"alpha_{prefix}_ann": None, f"t_alpha_{prefix}": None, f"beta_mkt_{prefix}": None}
    return {
        f"alpha_{prefix}_ann": result.alpha_ann,
        f"t_alpha_{prefix}": result.alpha_t,
        f"beta_mkt_{prefix}": float(result.coef[1]),
    }


def screen(dataset: Dataset, window: int, top: int) -> dict[str, Any]:
    excess = excess_returns(dataset)
    raw = raw_returns(dataset)
    labels = window_labels(window)
    selected_label = next(label for label, days in labels.items() if days == window)
    factors = dataset.factors.reindex(excess.index)
    sectors = dict(zip(dataset.universe["ticker"], dataset.universe["sector"]))
    websites = dict(zip(dataset.universe["ticker"], dataset.universe["website"])) if "website" in dataset.universe else {}

    horizons: dict[str, dict[str, Any]] = {}
    sr_ann: dict[str, dict[str, float]] = {t: {} for t in excess.columns}
    for label, days in labels.items():
        srs, ns, unscoreable = [], [], {}
        for ticker in excess.columns:
            observed = metrics.window_excess(excess[ticker], days)
            sr = math.nan if observed is None else metrics.sharpe_ratio(observed)
            # A covered name can still be unscoreable (flat or stale prices, non-finite SR): it
            # stays out of N and the plane's cross-section, and is left unranked.
            reason = None if observed is None else unscoreable_reason(raw[ticker].iloc[-days:].dropna().to_numpy(), sr)
            if reason is not None:
                unscoreable[ticker] = reason
                sr = math.nan
            sr_ann[ticker][label] = sr * math.sqrt(metrics.TRADING_DAYS) if math.isfinite(sr) else math.nan
            if math.isfinite(sr):
                srs.append(sr)
                ns.append(observed.size)
        horizons[label] = {**luck_horizon(srs, ns), "excluded_unscoreable": dict(sorted(unscoreable.items()))}

    headline = horizons[selected_label]
    uncovered = sorted(t for t in excess.columns if metrics.window_excess(excess[t], window) is None)
    if headline["n_trials"] < 2:
        raise RuntimeError(f"fewer than two tickers have {window} days of data at {metrics.MIN_COVERAGE:.0%} coverage")
    e_max = headline["headline_per_period"]
    lh_ann = headline["headline"]

    rows: list[dict[str, Any]] = []
    for ticker in excess.columns:
        series = excess[ticker]
        observed = metrics.window_excess(series, window)
        row: dict[str, Any] = {"ticker": ticker, "sector": sectors.get(ticker, "Unclassified"), "sr_ann": sr_ann[ticker]}
        if websites.get(ticker):
            row["website"] = websites[ticker]
        if observed is None or ticker in headline["excluded_unscoreable"]:
            row.update({"n_obs": int(series.iloc[-window:].notna().sum()), "sr_per_period": None, "skew": None,
                        "kurtosis_pearson": None, "psr": None, "dsr": None,
                        **_alpha_fields("capm", None), **_alpha_fields("ff3", None)})
        else:
            sr_pp = metrics.sharpe_ratio(observed)
            skew, kurt = metrics.sample_moments(observed)
            in_window = series.iloc[-window:]
            window_factors = factors.loc[in_window.index]
            row.update({
                "n_obs": int(observed.size),
                "sr_per_period": sr_pp,
                "skew": skew,
                "kurtosis_pearson": kurt,
                "psr": metrics.probabilistic_sharpe_ratio(sr_pp, 0.0, observed.size, skew, kurt),
                "dsr": metrics.deflated_sharpe_ratio(sr_pp, observed.size, skew, kurt, e_max),
                **_alpha_fields("capm", metrics.jensen_alpha(in_window, window_factors[["mkt_rf"]])),
                **_alpha_fields("ff3", metrics.jensen_alpha(in_window, window_factors[["mkt_rf", "smb", "hml"]])),
            })
        rows.append(row)

    ranked = sorted(
        (r for r in rows if math.isfinite(r["sr_ann"][selected_label])),
        key=lambda r: (-r["sr_ann"][selected_label], r["ticker"]),
    )
    unranked = sorted((r for r in rows if not math.isfinite(r["sr_ann"][selected_label])), key=lambda r: r["ticker"])
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
        row["sr_ann_window"] = row["sr_ann"][selected_label]
        row["above_luck_plane"] = row["sr_ann_window"] > lh_ann
        row["dsr_passes"] = row["dsr"] is not None and row["dsr"] >= DSR_THRESHOLD
        row["in_top"] = i <= top
    for row in unranked:
        row.update({"rank": None, "sr_ann_window": None, "above_luck_plane": False, "dsr_passes": False, "in_top": False})

    top_rows = ranked[:top]
    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_mode": dataset.mode,
            "data_start": str(excess.index.min().date()),
            "data_end": str(excess.index.max().date()),
            "window": window,
            "window_label": selected_label,
            "windows": labels,
            "min_coverage": metrics.MIN_COVERAGE,
            "N": headline["n_trials"],
            "excluded_unscoreable": headline["excluded_unscoreable"],
            "excluded_coverage": {"count": len(uncovered), "tickers": uncovered},
            "universe_size": len(rows),
            "top_n": len(top_rows),
            "rank_by": f"sr_ann[{selected_label}]",
            "dsr_threshold": DSR_THRESHOLD,
            "luck_horizon_ann": {
                "headline": headline["headline"],
                "headline_variance": headline["headline_variance"],
                "cross_sectional": headline["cross_sectional"],
                "analytic_null": headline["analytic_null"],
                "center_sr_ann": headline["center_sr_ann"],
                "definition": "expected best-of-N annualized Sharpe under a zero-alpha null: centre = cross-sectional "
                              "mean SR, V[SR] = max(cross-sectional variance, (1 + mean^2/2)/median_n)",
            },
            "luck_horizon_by_window": horizons,
            "top_dsr_passing": sum(r["dsr_passes"] for r in top_rows),
            "top_above_luck_plane": sum(r["above_luck_plane"] for r in top_rows),
        },
        "top": top_rows,
        "universe": ranked + unranked,
    }


def _nan_to_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _nan_to_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_nan_to_none(v) for v in value]
    return value


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


REASONS = ("flat price", "stale quotes", "non-finite Sharpe")
LISTED = 20


def _names(tickers: list[str]) -> str:
    shown = ", ".join(tickers[:LISTED])
    return shown + (f" and {len(tickers) - LISTED} more" if len(tickers) > LISTED else "")


def exclusion_lines(meta: dict[str, Any]) -> list[str]:
    """The report's account of every name left out of the screened window, and why."""
    coverage = meta["excluded_coverage"]
    unscoreable = meta["excluded_unscoreable"]
    lines = [f"## Exclusions ({meta['window_label']})", "",
             f"- Coverage: {coverage['count']} name{'s' if coverage['count'] != 1 else ''} with less than "
             f"{meta['min_coverage']:.0%} of the window's trading days"
             + (f": {_names(coverage['tickers'])}." if coverage["count"] else ".")]
    by_reason = {reason: sorted(t for t, r in unscoreable.items() if r == reason) for reason in REASONS}
    lines.append(f"- Unscoreable: {len(unscoreable)} name{'s' if len(unscoreable) != 1 else ''} "
                 "(data-quality rule, see the README): "
                 + ", ".join(f"{reason} {len(names)}" for reason, names in by_reason.items()) + ".")
    lines += [f"  - {reason}: {_names(names)}." for reason, names in by_reason.items() if names]
    return lines + [""]


def render_report(result: dict[str, Any]) -> str:
    meta = result["metadata"]
    lh = meta["luck_horizon_ann"]
    label = meta["window_label"]
    top = result["top"]
    lines = [
        "# Sharpe Observatory screen",
        "",
        REPORT_WARNING,
        "",
        f"**{meta['N']} names screened** over the trailing {meta['window']} trading days ({label}), "
        f"data {meta['data_start']} to {meta['data_end']} ({meta['data_mode']} data).",
        "",
        f"> **{meta['top_dsr_passing']} of {meta['top_n']}** top-ranked names have a Deflated Sharpe Ratio "
        f">= {meta['dsr_threshold']:.2f}, the evidence bar for a record that is more than the lucky best of "
        f"{meta['N']} names.",
        ">",
        f"> Luck plane: annualized Sharpe {_fmt(lh['headline'], '.2f')}, the expected best-of-{meta['N']} Sharpe under a "
        f"zero-alpha null in which every name has the universe's average Sharpe ({_fmt(lh['center_sr_ann'], '.2f')}). "
        f"{meta['top_above_luck_plane']} of {meta['top_n']} rise above it. Under that null the single best name exceeds "
        "the plane a little under half the time, so being above it is necessary, not sufficient. "
        f"(Variants: cross-sectional {_fmt(lh['cross_sectional'], '.2f')}, analytic {_fmt(lh['analytic_null'], '.2f')}; "
        f"the plane uses the larger, {lh['headline_variance'].replace('_', '-')}.)",
        "",
        f"| Rank | Ticker | Sector | Sharpe ({label}) | " + " | ".join(f"Sharpe {w}" for w in meta["windows"] if w != label)
        + f" | CAPM alpha (t) | FF3 alpha (t) | PSR | DSR | DSR >= {meta['dsr_threshold']:.2f} | Above plane |",
        "|---:|---|---|---:|" + "---:|" * (len(meta["windows"]) - 1) + "---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in top:
        others = " | ".join(_fmt(row["sr_ann"][w], ".2f") for w in meta["windows"] if w != label)
        lines.append(
            f"| {row['rank']} | {row['ticker']} | {row['sector']} | {row['sr_ann_window']:.2f} | {others} | "
            f"{_fmt(row['alpha_capm_ann'], '+.1%')} ({_fmt(row['t_alpha_capm'], '.2f')}) | "
            f"{_fmt(row['alpha_ff3_ann'], '+.1%')} ({_fmt(row['t_alpha_ff3'], '.2f')}) | "
            f"{_fmt(row['psr'], '.3f')} | {_fmt(row['dsr'], '.3f')} | {'yes' if row['dsr_passes'] else 'no'} | "
            f"{'yes' if row['above_luck_plane'] else 'no'} |"
        )
    lines += [
        "",
        *exclusion_lines(meta),
        "## How to read this",
        "",
        "- **Sharpe** is average daily return above the risk-free rate divided by its daily standard deviation, "
        "scaled by sqrt(252) to a yearly figure. A name needs at least "
        f"{meta['min_coverage']:.0%} of the window's trading days to be scored.",
        f"- **Luck plane.** Screening {meta['N']} names and picking the best is {meta['N']} tries at once, so the winner "
        "looks good partly by luck. The False Strategy Theorem (Bailey, Borwein, López de Prado & Zhu 2014) gives the "
        "Sharpe ratio the best of N tries is expected to reach if no name has an edge over the others. Here the null "
        "gives every name the universe's average Sharpe, and the spread of chance outcomes is the larger of two "
        "estimates: the observed spread of Sharpe ratios across the universe and the textbook sampling variance for "
        "this sample length. Taking the larger is the more conservative of those two estimates, nothing stronger.",
        "- **PSR** (Bailey & López de Prado 2012) is the probability that the true Sharpe ratio is above zero, "
        "allowing for sample length, skewness and fat tails. **DSR** is the same probability measured against the luck "
        f"plane instead of zero. A DSR of at least {meta['dsr_threshold']:.2f} is the skill test used here; below it, "
        "the record cannot be distinguished from luck at this sample size, which reflects low power, not proof of luck.",
        "- **Alpha** is the intercept of a regression of daily excess returns on the market factor (CAPM) or on the "
        "market, size and value factors (Fama-French 3), annualized (x252), with its classic OLS t-statistic.",
        "- Ranking is by Sharpe ratio over the selected window; it is a description of the past, not a forecast.",
        "",
        "## Not investment advice",
        "",
        "This report is an educational statistical screen of historical data. It is not investment advice, "
        "a recommendation, or an offer to buy or sell any security. Past risk-adjusted returns do not predict future "
        "returns. Current constituent lists suffer from survivorship bias, and classic OLS standard errors ignore "
        "heteroskedasticity and autocorrelation.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sharpe/alpha screen with a multiple-testing luck horizon.")
    parser.add_argument("--live", action="store_true",
                        help="download data from Yahoo Finance (yfinance), fred.stlouisfed.org and mba.tuck.dartmouth.edu")
    parser.add_argument("--universe", type=Path,
                        help="ticker/sector CSV; with --live it overrides the fetched S&P 500 list")
    parser.add_argument("--index", choices=INDEXES, default="sp500",
                        help="with --live and no --universe: sp500 (default) or us-all (every US-listed common stock)")
    parser.add_argument("--cache-dir", type=Path,
                        help="with --live, resumable per-ticker price cache: fetched tickers are saved and skipped next run")
    parser.add_argument("--save-universe", type=Path, help="with --live, write the universe used to this CSV")
    parser.add_argument("--limit-universe", type=int, help="with --live, screen only the first N names (debugging)")
    parser.add_argument("--throttle", type=float, default=THROTTLE_S,
                        help=f"with --live, minimum seconds between Yahoo requests (default {THROTTLE_S}); "
                             "backoff raises it automatically")
    parser.add_argument("--data-dir", type=Path, default=SAMPLE_DIR, help="offline dataset directory (default: bundled sample)")
    parser.add_argument("--window", type=int, default=756, help="trading days (252=1y, 756=3y, 1260=5y)")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args(argv)
    if args.window < 3 or args.top < 1 or (args.limit_universe is not None and args.limit_universe < 1):
        parser.error("--window must be >= 3, --top >= 1 and --limit-universe >= 1")
    if args.throttle < 0:
        parser.error("--throttle must be >= 0")

    if not args.live and (args.save_universe is not None or args.limit_universe is not None or args.cache_dir is not None):
        parser.error("--save-universe, --limit-universe and --cache-dir need --live")
    try:
        if args.live:
            universe = live_universe(args.universe, args.index)
            if args.save_universe is not None:
                args.save_universe.parent.mkdir(parents=True, exist_ok=True)
                universe.to_csv(args.save_universe, index=False)
                print(f"saved {len(universe)} names to {args.save_universe}", file=sys.stderr)
            if args.limit_universe is not None:
                universe = universe.head(args.limit_universe)
            dataset = fetch_live(universe, throttle_s=args.throttle, cache_dir=args.cache_dir)
        else:
            dataset = load_directory(args.data_dir, args.universe)
        result = _nan_to_none(screen(dataset, args.window, args.top))
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "screen_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.out / "report.md").write_text(render_report(result), encoding="utf-8")
    meta = result["metadata"]
    print(
        f"N={meta['N']} luck_plane_ann={_fmt(meta['luck_horizon_ann']['headline'], '.3f')} "
        f"top {meta['top_n']}: {meta['top_dsr_passing']} with DSR>={meta['dsr_threshold']} -> "
        f"{args.out}/screen_result.json, {args.out}/report.md"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
