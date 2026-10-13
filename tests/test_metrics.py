"""Acceptance tests. Expected values are literals worked out by hand (arithmetic in comments),
never recomputed by the code under test."""

from __future__ import annotations

import ast
import json
import math
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import fetch_data  # noqa: E402
import metrics  # noqa: E402
import render_skyline  # noqa: E402
import scene_python  # noqa: E402
import scene_threejs  # noqa: E402
import screen  # noqa: E402


# --- 1. Sharpe -------------------------------------------------------------------------

def alternating(a: float, b: float, n: int) -> np.ndarray:
    return np.array([a, b] * (n // 2))


def test_annualized_sharpe_matches_hand_value():
    # 252 days alternating +1.1% / -0.9%: mean = 0.001, every deviation is +-0.01, so
    # std(ddof=1) = 0.01 * sqrt(252/251). SR_daily = 0.1 * sqrt(251/252);
    # SR_ann = SR_daily * sqrt(252) = 0.1 * sqrt(251) = 1.584297951775486.
    excess = alternating(0.011, -0.009, 252)
    assert metrics.annualized_sharpe(excess) == pytest.approx(1.584297951775486, rel=1e-12)


def test_sharpe_uses_excess_over_risk_free():
    # Total returns +1.12% / -0.88% less a 0.02% daily risk-free rate give the series above.
    total = pd.Series(alternating(0.0112, -0.0088, 252))
    excess = total - 0.0002
    assert metrics.windowed_sharpe_ann(excess, 252) == pytest.approx(1.584297951775486, rel=1e-9)


def test_window_requires_ninety_percent_coverage():
    # ceil(0.9 * 252) = 227 observations needed in the last 252 rows.
    values = alternating(0.011, -0.009, 300).astype(float)
    short = values.copy()
    short[-252:-252 + 26] = np.nan  # 226 present
    enough = values.copy()
    enough[-252:-252 + 25] = np.nan  # 227 present
    assert math.isnan(metrics.windowed_sharpe_ann(pd.Series(short), 252))
    assert math.isfinite(metrics.windowed_sharpe_ann(pd.Series(enough), 252))
    assert math.isnan(metrics.windowed_sharpe_ann(pd.Series(values[:200]), 252))


# --- 2. PSR ----------------------------------------------------------------------------

def test_psr_is_one_half_when_sr_equals_benchmark():
    assert metrics.probabilistic_sharpe_ratio(0.05, 0.05, 252, 0.0, 3.0) == 0.5


def test_psr_hand_worked_case():
    # n = 253, SR = 0.1, SR* = 0, skew = -0.5, Pearson kurtosis = 5:
    #   variance term = 1 - (-0.5)(0.1) + ((5 - 1)/4)(0.1^2) = 1 + 0.05 + 0.01 = 1.06
    #   z = 0.1 * sqrt(252) / sqrt(1.06) = 1.5874507866 / 1.0295630141 = 1.5418685063
    #   PSR = Phi(1.5418685063) = 0.9384472247647292
    assert metrics.probabilistic_sharpe_ratio(0.1, 0.0, 253, -0.5, 5.0) == pytest.approx(0.9384472247647292, abs=1e-12)


def test_psr_kurtosis_convention_is_pearson():
    # Twenty returns, ten at 0.015 and ten at -0.005: a symmetric two-point distribution, so
    # skew = 0 and Pearson kurtosis = 1 exactly (excess kurtosis = -2).
    # SR = 0.005 / (0.01 * sqrt(20/19)) = 0.48733971724044817.
    # Pearson: variance term = 1 + ((1 - 1)/4) SR^2 = 1, z = SR * sqrt(19) = 2.1242645786,
    #          PSR = Phi(2.1242645786) = 0.9831759870626192.
    # Feeding excess kurtosis (-2) instead: variance term = 1 - 0.75 SR^2 = 0.8218, z = 2.3432,
    #          PSR = 0.9904399316901484 -- this test must reject that.
    returns = np.array([0.015, -0.005] * 10)
    skew, kurt = metrics.sample_moments(returns)
    assert skew == pytest.approx(0.0, abs=1e-12)
    assert kurt == pytest.approx(1.0, abs=1e-12)
    assert metrics.psr_from_returns(returns, 0.0) == pytest.approx(0.9831759870626192, abs=1e-12)


def test_kurtosis_of_normal_sample_is_near_three():
    rng = np.random.default_rng(11)
    _, kurt = metrics.sample_moments(rng.standard_normal(200_000))
    assert abs(kurt - 3.0) < 0.05


def test_dsr_is_psr_against_expected_max():
    # Same moments as the hand-worked PSR, benchmark 0.05: z = 0.05 * sqrt(252) / sqrt(1.06)
    # = 0.7709342531, DSR = Phi(0.7709342531) = 0.7796270485502477.
    assert metrics.deflated_sharpe_ratio(0.1, 253, -0.5, 5.0, 0.05) == pytest.approx(0.7796270485502477, abs=1e-12)


# --- 3. Expected maximum Sharpe (False Strategy Theorem) ---------------------------------

def test_expected_max_of_one_trial_is_the_mean():
    assert metrics.expected_max_sharpe(1, 0.07, 0.004) == 0.07


def test_expected_max_hand_value():
    # N = 100: Z(0.99) = 2.3263478740, Z(1 - 1/(100e)) = 2.6802104450,
    # (1 - 0.5772156649)(2.3263478740) + 0.5772156649(2.6802104450) = 2.5306028932.
    # With E[SR] = 0.1 and V[SR] = 0.04: 0.1 + 0.2 * 2.5306028932 = 0.6061205786.
    assert metrics.expected_max_sharpe(100, 0.0, 1.0) == pytest.approx(2.5306028932011424, abs=1e-9)
    assert metrics.expected_max_sharpe(100, 0.1, 0.04) == pytest.approx(0.6061205786402285, abs=1e-9)


def test_expected_max_matches_monte_carlo():
    # 4000 batches of N = 50 iid-normal null series of T = 252 days; the average of each
    # batch's maximum per-period SR is compared with the formula at E[SR] = 0, V[SR] = 1/T.
    rng = np.random.default_rng(12345)
    n_strategies, n_days, batches, chunk = 50, 252, 4000, 500
    maxima = []
    for _ in range(batches // chunk):
        x = rng.standard_normal((chunk, n_strategies, n_days))
        maxima.append((x.mean(axis=-1) / x.std(axis=-1, ddof=1)).max(axis=-1))
    simulated = float(np.concatenate(maxima).mean())
    analytic = metrics.expected_max_sharpe(n_strategies, 0.0, 1.0 / n_days)
    assert analytic == pytest.approx(simulated, rel=0.02)


# --- 4. Jensen's alpha -----------------------------------------------------------------

def test_ols_hand_worked_case():
    # x = [-2, -1, 0, 1, 2] (mean 0, Sxx = 10), y = [1, 3, 2, 5, 4]:
    #   beta = Sxy / Sxx = 8 / 10 = 0.8; alpha = mean(y) = 3
    #   residuals = [-0.4, 0.8, -1, 1.2, -0.6], RSS = 3.6, sigma^2 = 3.6 / (5 - 2) = 1.2
    #   se(alpha) = sqrt(1.2 / 5) = 0.4898979486, se(beta) = sqrt(1.2 / 10) = 0.3464101615
    #   t(alpha) = 3 / 0.4898979486 = 6.1237243570; alpha_ann = 3 * 252 = 756
    fit = metrics.ols_with_intercept([1, 3, 2, 5, 4], [-2, -1, 0, 1, 2])
    assert fit.alpha == pytest.approx(3.0, abs=1e-12)
    assert fit.coef[1] == pytest.approx(0.8, abs=1e-12)
    assert fit.alpha_se == pytest.approx(0.4898979485566356, abs=1e-12)
    assert fit.se[1] == pytest.approx(0.34641016151377546, abs=1e-12)
    assert fit.alpha_t == pytest.approx(6.123724356957945, abs=1e-9)
    assert fit.alpha_ann == pytest.approx(756.0, abs=1e-9)


def _simulated(alpha: float, seed: int) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = 5000
    factors = pd.DataFrame({
        "mkt_rf": rng.normal(0.0003, 0.01, n),
        "smb": rng.normal(0.0, 0.005, n),
        "hml": rng.normal(0.0, 0.005, n),
    })
    y = alpha + 1.2 * factors["mkt_rf"] + 0.4 * factors["smb"] - 0.3 * factors["hml"] + rng.normal(0.0, 0.01, n)
    return y, factors


def test_alpha_recovered_with_positive_t():
    true_alpha = 0.0004
    y, factors = _simulated(true_alpha, seed=7)
    fit = metrics.jensen_alpha(y, factors)
    assert abs(fit.alpha - true_alpha) < 2.576 * fit.alpha_se
    assert abs(fit.coef[1] - 1.2) < 2.576 * fit.se[1]
    assert abs(fit.coef[2] - 0.4) < 2.576 * fit.se[2]
    assert abs(fit.coef[3] + 0.3) < 2.576 * fit.se[3]
    assert fit.alpha_t > 2.0
    assert fit.n_obs == 5000


def test_zero_alpha_has_small_t():
    y, factors = _simulated(0.0, seed=8)
    capm = metrics.jensen_alpha(y, factors[["mkt_rf"]])
    ff3 = metrics.jensen_alpha(y, factors)
    assert abs(capm.alpha_t) < 2.0
    assert abs(ff3.alpha_t) < 2.0


# --- Data parsers (formats as served by FRED and the Ken French library) ----------------

def test_fred_parser_deannualizes_and_drops_missing():
    text = "observation_date,DGS3MO\n2026-10-07,4.22\n2026-10-08,.\n2026-10-09,4.20\n"
    rf = fetch_data.parse_fred_dgs3mo(text)
    assert list(rf.index.strftime("%Y-%m-%d")) == ["2026-10-07", "2026-10-09"]
    assert rf.iloc[0] == pytest.approx(0.00016746031746031745, abs=1e-15)  # 4.22 / 100 / 252


def test_ff3_parser_skips_text_and_converts_percent():
    text = (
        "This file was created using the CRSP database.\n\n"
        ",Mkt-RF,SMB,HML,RF\n"
        "19260701,    0.09,   -0.23,   -0.28,    0.01\n"
        "19260702,    0.45,   -0.34,   -0.03,    0.01\n\n"
        "Copyright 2026 Eugene F. Fama and Kenneth R. French\n"
    )
    ff = fetch_data.parse_ff3_daily_csv(text)
    assert list(ff.columns) == ["mkt_rf", "smb", "hml"]
    assert len(ff) == 2
    assert ff.iloc[0].tolist() == pytest.approx([0.0009, -0.0023, -0.0028], abs=1e-15)


def test_fetch_prices_aborts_after_five_consecutive_failures():
    requested: list[str] = []

    def always_fails(ticker: str) -> pd.Series:
        requested.append(ticker)
        raise ValueError("HTTP Error 404: Not Found")

    with pytest.raises(RuntimeError, match=r"^stopped after 5 consecutive ticker failures \(last: E\);"):
        fetch_data.fetch_prices(list("ABCDEFGH"), fetch_close=always_fails, throttle_s=0.0, sleep=lambda s: None)
    assert requested == list("ABCDE")


def test_fetch_prices_skips_isolated_failures():
    # Four failures, a success, four failures, a success: the streak never reaches five.
    good = {"E", "J"}

    def flaky(ticker: str) -> pd.Series:
        if ticker not in good:
            raise ValueError("no rows")
        return pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-02", "2026-01-05"]), name=ticker)

    prices, failures = fetch_data.fetch_prices(list("ABCDEFGHIJ"), fetch_close=flaky, throttle_s=0.0,
                                               sleep=lambda s: None)
    assert list(prices.columns) == ["E", "J"]
    assert sorted(failures) == list("ABCDFGHI")
    assert failures["A"] == "ValueError: no rows"


def test_fetch_prices_aborts_on_high_failure_rate():
    # Four failures then one success, repeated: never five in a row, but the 20th failure
    # arrives at ticker 24 (20/24 = 83% > 50%), so the fetch stops there.
    requested: list[str] = []
    tickers = [f"T{i:02d}" for i in range(50)]

    def four_fail_one_ok(ticker: str) -> pd.Series:
        requested.append(ticker)
        if int(ticker[1:]) % 5 != 4:
            raise ValueError("HTTP Error 404: Not Found")
        return pd.Series([1.0], index=pd.to_datetime(["2026-01-02"]), name=ticker)

    with pytest.raises(RuntimeError) as info:
        fetch_data.fetch_prices(tickers, fetch_close=four_fail_one_ok, throttle_s=0.0, sleep=lambda s: None)
    assert str(info.value) == ("stopped after 20 failures in 24 tickers (83%); "
                               "Yahoo Finance may be blocking or rate-limiting requests")
    assert len(requested) == 24


def _ok(ticker: str) -> pd.Series:
    return pd.Series([1.0], index=pd.to_datetime(["2026-01-02"]), name=ticker)


def test_transient_failure_retries_with_backoff():
    pauses: list[float] = []
    attempts: list[str] = []

    def fails_twice(ticker: str) -> pd.Series:
        attempts.append(ticker)
        if len(attempts) < 3:
            raise ConnectionError("Connection reset by peer")
        return _ok(ticker)

    prices, failures = fetch_data.fetch_prices(["AAA"], fetch_close=fails_twice, throttle_s=0.5, sleep=pauses.append)
    assert pauses == [4.0, 15.0]
    assert attempts == ["AAA", "AAA", "AAA"]
    assert failures == {} and list(prices.columns) == ["AAA"]

    pauses.clear()
    attempts.clear()

    def always_429(ticker: str) -> pd.Series:
        attempts.append(ticker)
        raise Exception("HTTP Error 429: Too Many Requests")

    prices, failures = fetch_data.fetch_prices(["BBB"], fetch_close=always_429, throttle_s=0.5, sleep=pauses.append)
    assert pauses == [4.0, 15.0]
    assert attempts == ["BBB", "BBB", "BBB"]
    assert failures == {"BBB": "Exception: HTTP Error 429: Too Many Requests"}


def test_transient_versus_permanent_classification():
    class YFRateLimitError(Exception):
        pass

    class ReadTimeout(Exception):
        pass

    transient = [ConnectionError("x"), TimeoutError("x"), YFRateLimitError("Too Many Requests. Rate limited."),
                 ReadTimeout("x"), Exception("HTTP Error 429"), Exception("HTTP Error 503"),
                 Exception("Operation timed out after 30001 milliseconds"), Exception("HTTP Error 500: Internal")]
    permanent = [Exception("HTTP Error 404: Not Found"), ValueError("Yahoo Finance returned no rows"),
                 Exception("$XYZ: possibly delisted; no price data found (period=max)"), KeyError("Close"),
                 ValueError("only 5000 rows")]
    assert [fetch_data.is_transient(e) for e in transient] == [True] * len(transient)
    assert [fetch_data.is_transient(e) for e in permanent] == [False] * len(permanent)

    pauses: list[float] = []
    attempts: list[str] = []

    def not_found(ticker: str) -> pd.Series:
        attempts.append(ticker)
        raise Exception("HTTP Error 404: Not Found")

    _, failures = fetch_data.fetch_prices(["CCC"], fetch_close=not_found, throttle_s=0.5, sleep=pauses.append)
    assert attempts == ["CCC"] and pauses == []
    assert failures == {"CCC": "Exception: HTTP Error 404: Not Found"}


def test_throttle_doubles_on_failure_up_to_cap(capsys: pytest.CaptureFixture[str]):
    # Tickers alternate fail, ok, fail, ok... from a 1 s base. Each failure doubles the pause
    # (1 -> 2 -> 4 -> 8) and it stays capped at 8 s; successes in between do not lower it.
    pauses: list[float] = []

    def alternate(ticker: str) -> pd.Series:
        if int(ticker[1:]) % 2 == 0:
            raise ValueError("HTTP Error 404: Not Found")
        return _ok(ticker)

    fetch_data.fetch_prices([f"T{i:02d}" for i in range(12)], fetch_close=alternate, throttle_s=1.0, sleep=pauses.append)
    assert pauses == [2.0, 2.0, 4.0, 4.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
    changes = [line for line in capsys.readouterr().err.splitlines() if "throttle" in line]
    assert changes == ["[fetch] throttle 1s -> 2s (T00 failed)", "[fetch] throttle 2s -> 4s (T02 failed)",
                       "[fetch] throttle 4s -> 8s (T04 failed)"]


def test_throttle_decays_after_twenty_clean_fetches():
    # Base 0.5 s; T00 and T01 fail (0.5 -> 1 -> 2), then 70 clean fetches. After every 20
    # clean fetches the pause halves (2 -> 1 -> 0.5) and never drops below the base.
    pauses: list[float] = []

    def first_two_fail(ticker: str) -> pd.Series:
        if ticker in {"T00", "T01"}:
            raise ValueError("HTTP Error 404: Not Found")
        return _ok(ticker)

    fetch_data.fetch_prices([f"T{i:02d}" for i in range(72)], fetch_close=first_two_fail, throttle_s=0.5,
                            sleep=pauses.append)
    assert pauses == [1.0] + [2.0] * 20 + [1.0] * 20 + [0.5] * 30

    # Base 3 s: the cap breaks the symmetry (3 -> 6 -> 8), so halving goes 8 -> 4 -> 2, which
    # must be floored at the 3 s base.
    pauses.clear()
    fetch_data.fetch_prices([f"T{i:02d}" for i in range(72)], fetch_close=first_two_fail, throttle_s=3.0,
                            sleep=pauses.append)
    assert pauses == [6.0] + [8.0] * 20 + [4.0] * 20 + [3.0] * 30


def test_retry_success_is_not_a_failure():
    # A-D fail, E fails once with a timeout then succeeds, F-I fail, J succeeds. If E's retry
    # counted as a failure, five failures in a row (A-E) would abort the fetch.
    pauses: list[float] = []
    calls: list[str] = []

    def fetch(ticker: str) -> pd.Series:
        calls.append(ticker)
        if ticker == "E" and calls.count("E") == 1:
            raise TimeoutError("read timed out")
        if ticker in {"E", "J"}:
            return _ok(ticker)
        raise ValueError("HTTP Error 404: Not Found")

    prices, failures = fetch_data.fetch_prices(list("ABCDEFGHIJ"), fetch_close=fetch, throttle_s=0.0, sleep=pauses.append)
    assert list(prices.columns) == ["E", "J"]
    assert sorted(failures) == list("ABCDFGHI")
    assert [p for p in pauses if p] == [4.0]


def test_missing_yfinance_is_reported_as_setup_problem():
    requested: list[str] = []

    def not_installed(ticker: str) -> pd.Series:
        requested.append(ticker)
        raise ModuleNotFoundError("No module named 'yfinance'", name="yfinance")

    with pytest.raises(RuntimeError) as info:
        fetch_data.fetch_prices(["AAA", "BBB"], fetch_close=not_installed, throttle_s=0.0)
    assert str(info.value) == "yfinance is not installed; run pip install -r requirements.txt"
    assert requested == ["AAA"]


def test_yahoo_adapter_requests_adjusted_daily_history(monkeypatch: pytest.MonkeyPatch):
    yfinance = pytest.importorskip("yfinance")
    calls: list[tuple[str, dict]] = []
    index = pd.DatetimeIndex(["2026-10-08 00:00", "2026-10-09 00:00"]).tz_localize("America/New_York")

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, **kwargs: object) -> pd.DataFrame:
            calls.append((self.symbol, kwargs))
            rows = [] if self.symbol == "EMPTY" else [10.0, 11.0]
            return pd.DataFrame({"Close": rows}, index=index[: len(rows)])

    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
    yfinance.config.debug.hide_exceptions = True
    try:
        close = fetch_data.fetch_yahoo_close("BRK.B")
        assert yfinance.config.debug.hide_exceptions is False
    finally:
        yfinance.config.debug.hide_exceptions = True
    symbol, kwargs = calls[0]
    assert symbol == "BRK-B"
    assert kwargs["interval"] == "1d" and kwargs["auto_adjust"] is True
    assert "raise_errors" not in kwargs
    assert close.name == "BRK.B"
    assert close.index.tz is None
    assert list(close.index.strftime("%Y-%m-%d")) == ["2026-10-08", "2026-10-09"]
    assert close.tolist() == [10.0, 11.0]
    with pytest.raises(ValueError, match="no rows"):
        fetch_data.fetch_yahoo_close("EMPTY")


# --- S&P 500 universe ------------------------------------------------------------------

DATASETS_CSV = (
    "Symbol,Security,GICS Sector,GICS Sub-Industry,Headquarters Location,Date added,CIK,Founded\n"
    ' MMM ,3M,Industrials,Industrial Conglomerates,"Saint Paul, Minnesota",1957-03-04,66740,1902\n'
    'ABT,Abbott Laboratories, Health Care ,Health Care Equipment,"North Chicago, Illinois",1957-03-04,1800,1888\n'
    'BRK.B,Berkshire Hathaway,Financials,Multi-Sector Holdings,"Omaha, Nebraska",2010-02-16,1067983,1839\n'
    'MMM,3M,Industrials,Industrial Conglomerates,"Saint Paul, Minnesota",1957-03-04,66740,1902\n'
)
EXPECTED_SMALL = [("ABT", "Health Care"), ("BRK.B", "Financials"), ("MMM", "Industrials")]

WIKIPEDIA_HTML = """<html><body>
<table><tr><th>Date</th><th>Added</th></tr><tr><td>2026-01-02</td><td>XYZ</td></tr></table>
<table class="wikitable" id="constituents">
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th></tr>
<tr><td>MMM</td><td>3M</td><td>Industrials</td><td>Industrial Conglomerates</td></tr>
<tr><td>ABT</td><td>Abbott Laboratories</td><td>Health Care</td><td>Health Care Equipment</td></tr>
<tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td><td>Multi-Sector Holdings</td></tr>
</table></body></html>"""


def _pairs(frame: pd.DataFrame) -> list[tuple[str, str]]:
    return list(zip(frame["ticker"], frame["sector"]))


def _big_universe(n: int, prefix: str) -> pd.DataFrame:
    return pd.DataFrame({"Symbol": [f"{prefix}{i:03d}" for i in range(n)], "GICS Sector": "Industrials"})


def test_sp500_datasets_csv_is_normalized(monkeypatch: pytest.MonkeyPatch):
    # Whitespace stripped, the repeated MMM row dropped, the dot in BRK.B kept, sorted by ticker.
    monkeypatch.setattr(fetch_data, "MIN_SP500_ROWS", 3)
    universe = fetch_data.normalize_sp500(fetch_data.parse_sp500_datasets_csv(DATASETS_CSV), "fixture")
    assert list(universe.columns) == ["ticker", "sector"]
    assert _pairs(universe) == EXPECTED_SMALL


def test_sp500_csv_with_renamed_columns(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(fetch_data, "MIN_SP500_ROWS", 3)
    renamed = "Ticker,Name,Sector\nMMM,3M,Industrials\nABT,Abbott,Health Care\nBRK.B,Berkshire,Financials\n"
    universe = fetch_data.normalize_sp500(fetch_data.parse_sp500_datasets_csv(renamed), "fixture")
    assert _pairs(universe) == EXPECTED_SMALL
    changed = "Code,Industry\nMMM,Industrials\n"
    with pytest.raises(ValueError, match=r"^fixture: unexpected schema \['Code', 'Industry'\]"):
        fetch_data.normalize_sp500(fetch_data.parse_sp500_datasets_csv(changed), "fixture")


def test_sp500_wikipedia_table_is_selected(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("lxml")
    monkeypatch.setattr(fetch_data, "MIN_SP500_ROWS", 3)
    table = fetch_data.parse_sp500_wikipedia_html(WIKIPEDIA_HTML)
    assert _pairs(fetch_data.normalize_sp500(table, "fixture")) == EXPECTED_SMALL
    with pytest.raises(ValueError, match="no table with Symbol and GICS Sector"):
        fetch_data.parse_sp500_wikipedia_html("<table><tr><th>Date</th></tr><tr><td>x</td></tr></table>")


def test_sp500_sanity_bound():
    with pytest.raises(ValueError, match=r"^src: only 399 constituents, fewer than 400$"):
        fetch_data.normalize_sp500(_big_universe(399, "A"), "src")
    assert len(fetch_data.normalize_sp500(_big_universe(400, "A"), "src")) == 400


def test_sp500_fallback_used_when_primary_fails():
    calls: list[str] = []

    def primary_down() -> pd.DataFrame:
        calls.append("primary")
        raise requests.ConnectionError("unreachable")

    def primary_short() -> pd.DataFrame:
        calls.append("primary")
        return _big_universe(399, "P")

    def fallback() -> pd.DataFrame:
        calls.append("fallback")
        return _big_universe(450, "F")

    for primary in (primary_down, primary_short):
        calls.clear()
        universe = fetch_data.fetch_sp500_universe([("primary", primary), ("fallback", fallback)])
        assert calls == ["primary", "fallback"]
        assert len(universe) == 450 and universe["ticker"].iloc[0] == "F000"

    calls.clear()
    universe = fetch_data.fetch_sp500_universe([("primary", lambda: _big_universe(500, "P")), ("fallback", fallback)])
    assert calls == [] and len(universe) == 500 and universe["ticker"].iloc[0] == "P000"

    with pytest.raises(RuntimeError, match=r"^could not fetch the S&P 500 constituents: primary: ConnectionError: unreachable; "
                                           r"fallback: ValueError: fallback: only 399 constituents"):
        fetch_data.fetch_sp500_universe([("primary", primary_down), ("fallback", lambda: _big_universe(399, "F"))])


def _sample_live_stub(universe: pd.DataFrame, throttle_s: float = 0.5, cache_dir: Path | None = None) -> fetch_data.Dataset:
    assert throttle_s >= 0
    sample = fetch_data.load_directory(fetch_data.SAMPLE_DIR)
    keep = list(universe["ticker"])
    return fetch_data.Dataset(universe=universe.reset_index(drop=True), prices=sample.prices[keep],
                              rf_daily=sample.rf_daily, factors=sample.factors, mode="live")


def test_live_without_universe_fetches_sp500(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    sample_universe = pd.read_csv(fetch_data.SAMPLE_DIR / "universe.csv")
    fetched: list[int] = []

    def stub_sp500() -> pd.DataFrame:
        fetched.append(1)
        return sample_universe

    monkeypatch.setattr(fetch_data, "fetch_sp500_universe", stub_sp500)
    monkeypatch.setattr(screen, "fetch_live", _sample_live_stub)
    saved = tmp_path / "universe.csv"
    assert screen.main(["--live", "--save-universe", str(saved), "--limit-universe", "12", "--out", str(tmp_path)]) == 0
    assert fetched == [1]
    assert "fetching S&P 500 constituents (raw.githubusercontent.com" in capsys.readouterr().err
    assert len(pd.read_csv(saved)) == 60
    result = json.loads((tmp_path / "screen_result.json").read_text())
    assert result["metadata"]["N"] == 12 and result["metadata"]["data_mode"] == "live"


def test_live_with_universe_does_not_fetch_sp500(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def must_not_run() -> pd.DataFrame:
        raise AssertionError("S&P 500 fetcher called despite --universe")

    monkeypatch.setattr(fetch_data, "fetch_sp500_universe", must_not_run)
    monkeypatch.setattr(screen, "fetch_live", _sample_live_stub)
    own = tmp_path / "mine.csv"
    own.write_text("ticker,sector\nTEC03,Information Technology\nFIN02,Financials\nIND04,Industrials\n")
    assert screen.main(["--live", "--universe", str(own), "--out", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "screen_result.json").read_text())["metadata"]["N"] == 3


# --- 5. End to end ---------------------------------------------------------------------

def run_script(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPTS / name), *args], capture_output=True, text=True, check=True)


@pytest.fixture(scope="module")
def screen_runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    first = tmp_path_factory.mktemp("run1")
    second = tmp_path_factory.mktemp("run2")
    run_script("screen.py", "--window", "756", "--top", "20", "--out", str(first))
    run_script("screen.py", "--window", "756", "--top", "20", "--out", str(second))
    return first, second


def test_screen_end_to_end(screen_runs: tuple[Path, Path]):
    out, _ = screen_runs
    result = json.loads((out / "screen_result.json").read_text())
    meta = result["metadata"]
    assert len(result["top"]) == 20
    assert [row["rank"] for row in result["top"]] == list(range(1, 21))
    assert meta["N"] == 60
    assert meta["luck_horizon_ann"]["headline"] > 0
    assert meta["luck_horizon_ann"]["cross_sectional"] > 0
    assert meta["luck_horizon_ann"]["analytic_null"] > 0
    first = result["top"][0]
    for key in ("ticker", "sector", "sr_ann", "alpha_capm_ann", "t_alpha_capm", "alpha_ff3_ann", "t_alpha_ff3",
                "psr", "dsr", "rank"):
        assert key in first
    assert set(first["sr_ann"]) == {"1y", "3y", "5y"}
    report = (out / "report.md").read_text()
    table_rows = [line for line in report.splitlines() if re.match(r"^\| \d+ \|", line)]
    assert len(table_rows) == 20
    assert "**1 of 20** top-ranked names have a Deflated Sharpe Ratio >= 0.95" in report
    assert "3 of 20 rise above it" in report
    assert "not investment advice" in report.lower()


def test_screen_with_fewer_than_two_names_fails_cleanly(tmp_path: Path):
    universe = tmp_path / "one.csv"
    universe.write_text("ticker,sector\nTEC03,Information Technology\n")
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "screen.py"), "--universe", str(universe), "--out", str(tmp_path / "out")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert proc.stderr == "error: fewer than two tickers have 756 days of data at 90% coverage\n"
    assert not (tmp_path / "out").exists()


def test_screen_with_malformed_universe_fails_cleanly(tmp_path: Path):
    universe = tmp_path / "bad.csv"
    universe.write_text("Code,Industry\nMMM,Industrials\n")
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "screen.py"), "--universe", str(universe), "--out", str(tmp_path / "out")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert proc.stderr.startswith("error: ")
    assert "Traceback" not in proc.stderr
    assert not (tmp_path / "out").exists()


def test_screen_is_deterministic(screen_runs: tuple[Path, Path]):
    first, second = (json.loads((d / "screen_result.json").read_text()) for d in screen_runs)
    del first["metadata"]["generated_at"], second["metadata"]["generated_at"]
    assert first == second
    assert (screen_runs[0] / "report.md").read_text() == (screen_runs[1] / "report.md").read_text()


def test_screen_respects_top(tmp_path: Path):
    run_script("screen.py", "--top", "7", "--out", str(tmp_path))
    assert len(json.loads((tmp_path / "screen_result.json").read_text())["top"]) == 7


# --- 5b. Pinned sample results (literals from one run of screen.py on data/sample, 3y window) ---

@pytest.fixture(scope="module")
def sample_result(screen_runs: tuple[Path, Path]) -> dict:
    return json.loads((screen_runs[0] / "screen_result.json").read_text())


def _row(result: dict, ticker: str) -> dict:
    return next(r for r in result["universe"] if r["ticker"] == ticker)


def test_pinned_ranking(sample_result: dict):
    assert [r["ticker"] for r in sample_result["top"][:5]] == ["TEC03", "FIN02", "IND04", "HLT05", "STP03"]


def test_pinned_psr_and_dsr(sample_result: dict):
    fin04 = _row(sample_result, "FIN04")
    assert fin04["rank"] == 15
    assert fin04["psr"] == pytest.approx(0.9216112215056622, abs=1e-12)
    assert fin04["dsr"] == pytest.approx(0.0031544401121948265, abs=1e-12)
    assert _row(sample_result, "TEC03")["dsr"] == pytest.approx(0.969046524812437, abs=1e-12)


def test_pinned_luck_plane_per_window(sample_result: dict):
    by_window = sample_result["metadata"]["luck_horizon_by_window"]
    assert by_window["1y"]["headline"] == pytest.approx(2.6945510488633913, abs=1e-12)
    assert by_window["3y"]["headline"] == pytest.approx(2.407983946474636, abs=1e-12)
    assert by_window["5y"]["headline"] == pytest.approx(2.173536589667446, abs=1e-12)
    # On the sample the cross-sectional variance is the larger one in every window.
    assert [by_window[w]["headline_variance"] for w in ("1y", "3y", "5y")] == ["cross_sectional"] * 3
    assert by_window["1y"]["analytic_null"] == pytest.approx(2.681675913568017, abs=1e-12)
    assert by_window["3y"]["analytic_null"] == pytest.approx(2.012589241462161, abs=1e-12)
    assert sample_result["metadata"]["luck_horizon_ann"]["headline"] == pytest.approx(2.407983946474636, abs=1e-12)


def test_luck_horizon_takes_larger_variance():
    # SRs [0, 0.001, 0.002], n = 252: centre 0.001; cross-sectional variance 1e-6; analytic
    # (1 + 0.001^2/2)/252 = 0.003968255952380952 is larger, so it sets the plane.
    # mix = (1 - g) Z(2/3) + g Z(1 - 1/(3e)) = 0.8528044961495741
    # headline = 0.001 + sqrt(0.003968255952380952) * 0.8528044961495741 = 0.054721647091585626
    # annualized x sqrt(252) = 0.8686792172170591; cross-sectional only: 0.0018528044961495741.
    horizon = screen.luck_horizon([0.0, 0.001, 0.002], [252, 252, 252])
    assert horizon["headline_variance"] == "analytic_null"
    assert horizon["headline_per_period"] == pytest.approx(0.054721647091585626, abs=1e-12)
    assert horizon["headline"] == pytest.approx(0.8686792172170591, abs=1e-12)
    assert horizon["cross_sectional_per_period"] == pytest.approx(0.0018528044961495741, abs=1e-12)


def test_pinned_sharpe_uses_risk_free(sample_result: dict):
    # TEC03, 3y window, excess over the sample's 4%/yr risk-free: 3.4941526243376835.
    # The same window computed on total returns (no risk-free subtraction) would be 3.6509080914372056.
    assert _row(sample_result, "TEC03")["sr_ann"]["3y"] == pytest.approx(3.4941526243376835, abs=1e-12)


def test_pinned_ff3_differs_from_capm(sample_result: dict):
    tec03 = _row(sample_result, "TEC03")
    assert tec03["alpha_capm_ann"] == pytest.approx(0.7661597912096079, abs=1e-12)
    assert tec03["alpha_ff3_ann"] == pytest.approx(0.7545390780091732, abs=1e-12)
    assert abs(tec03["alpha_capm_ann"] - tec03["alpha_ff3_ann"]) > 1e-3


def test_dsr_benchmark_is_above_zero(sample_result: dict):
    leader = sample_result["top"][0]
    assert leader["dsr"] < leader["psr"]


def test_pinned_counts(sample_result: dict):
    meta = sample_result["metadata"]
    assert meta["dsr_threshold"] == 0.95
    assert meta["top_dsr_passing"] == 1
    assert meta["top_above_luck_plane"] == 3
    assert [r["ticker"] for r in sample_result["top"] if r["dsr_passes"]] == ["TEC03"]
    assert [r["ticker"] for r in sample_result["top"] if r["above_luck_plane"]] == ["TEC03", "FIN02", "IND04"]
    # The three planted lucky names rank in the top 20 but none passes.
    lucky = {r["ticker"]: (r["rank"], r["dsr_passes"]) for r in sample_result["top"] if r["ticker"] in {"STP03", "UTL02", "CDS07"}}
    assert lucky == {"STP03": (5, False), "UTL02": (8, False), "CDS07": (20, False)}


def test_dsr_pass_is_counted():
    # 30 names over 300 days; WIN earns 1% a day on 1% daily volatility (per-period SR = 1),
    # far above any best-of-30 luck, so it alone must pass DSR >= 0.95.
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2024-01-01", periods=301, name="date")
    returns = pd.DataFrame(rng.normal(0.0, 0.01, (300, 30)), index=dates[1:], columns=[f"N{i:02d}" for i in range(30)])
    returns["N00"] = rng.normal(0.01, 0.01, 300)
    returns = returns.rename(columns={"N00": "WIN"})
    prices = pd.concat([pd.DataFrame(1.0, index=dates[:1], columns=returns.columns), (1 + returns).cumprod()])
    dataset = fetch_data.Dataset(
        universe=pd.DataFrame({"ticker": list(returns.columns), "sector": "Test"}),
        prices=prices,
        rf_daily=pd.Series(0.0, index=dates),
        factors=pd.DataFrame(rng.normal(0.0, 0.01, (300, 3)), index=dates[1:], columns=["mkt_rf", "smb", "hml"]),
        mode="test",
    )
    result = screen.screen(dataset, window=252, top=5)
    assert result["metadata"]["top_dsr_passing"] == 1
    assert [r["ticker"] for r in result["universe"] if r["dsr_passes"]] == ["WIN"]


# --- 6. Scene generators ---------------------------------------------------------------

def _embedded_bpy_data(source: str) -> dict:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "DATA":
            return json.loads(node.value.args[0].value)
    raise AssertionError("no DATA assignment in generated bpy script")


def _check_scene(data: dict, text: str, result: dict) -> None:
    meta = result["metadata"]
    leader = result["top"][0]
    assert leader["ticker"] in text
    towers = {t["ticker"]: t for t in data["towers"]}
    assert len(towers) == meta["universe_size"]
    assert towers[leader["ticker"]]["rank"] == 1
    assert sum(t["top"] for t in data["towers"]) == 20
    lh = meta["luck_horizon_ann"]["headline"]
    assert data["luck_horizon_ann"][meta["window_label"]] == pytest.approx(lh, abs=1e-4)
    assert data["luck_height"]["3y"] == pytest.approx(lh * data["scale"], abs=1e-3)
    assert max(abs(h) for t in data["towers"] for h in t["height"].values()) == pytest.approx(8.0, abs=1e-3)


def test_blender_scene(screen_runs: tuple[Path, Path], tmp_path: Path):
    src = screen_runs[0] / "screen_result.json"
    out = tmp_path / "skyline_bpy.py"
    run_script("scene_blender.py", str(src), "--out", str(out), "--chunk-bytes", "5000")
    text = out.read_text()
    assert text.strip()
    data = _embedded_bpy_data(text)
    _check_scene(data, text, json.loads(src.read_text()))
    assert re.search(r"""\.type\s*==\s*['"]BSDF_PRINCIPLED['"]""", text)
    assert not re.search(r"""nodes(\.get\(|\[)\s*['"]Principled""", text)
    assert "keyframe_insert" in text

    parts = sorted(tmp_path.glob("skyline_bpy_part*.py"))
    assert len(parts) >= 3
    fake_bpy = types.SimpleNamespace(app=types.SimpleNamespace(driver_namespace={}))
    sys.modules["bpy"] = fake_bpy  # type: ignore[assignment]
    try:
        for part in parts[:-1]:
            exec(compile(part.read_text(), str(part), "exec"), {})
    finally:
        del sys.modules["bpy"]
    assert json.loads(fake_bpy.app.driver_namespace["_sharpe_skyline_json"]) == data
    ast.parse(parts[-1].read_text())


def test_threejs_scene(screen_runs: tuple[Path, Path], tmp_path: Path):
    src = screen_runs[0] / "screen_result.json"
    out = tmp_path / "skyline.html"
    run_script("scene_threejs.py", str(src), "--out", str(out))
    html = out.read_text()
    match = re.search(r'<script id="skyline-data" type="application/json">(.*?)</script>', html, re.S)
    assert match
    _check_scene(json.loads(match.group(1)), html, json.loads(src.read_text()))
    assert "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js" in html
    assert "OrbitControls" in html


def _page_data(page: str) -> dict:
    match = re.search(r'<script id="skyline-data" type="application/json">(.*?)</script>', page, re.S)
    assert match
    return json.loads(match.group(1))


PINNED_TOP_20 = ["TEC03", "FIN02", "IND04", "HLT05", "STP03", "STP02", "STP06", "UTL02", "CDS05", "CDS06",
                 "IND07", "HLT06", "FIN03", "HLT08", "FIN04", "STP04", "TEC07", "TEC02", "UTL06", "CDS07"]


def test_threejs_page_interactivity(screen_runs: tuple[Path, Path], tmp_path: Path):
    out = tmp_path / "skyline.html"
    scene_threejs.write_page(screen_runs[0] / "screen_result.json", out)
    page = out.read_text()
    data = _page_data(page)
    assert data["links"] == {
        "google_finance": "https://www.google.com/finance?q={ticker}",
        "edgar": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&ticker={ticker}",
        "website_search": "https://www.google.com/search?q={ticker}+investor+relations",
    }
    assert data["luck_label"] == {"1y": "Luck plane: Sharpe 2.69 (1y)", "3y": "Luck plane: Sharpe 2.41 (3y)",
                                  "5y": "Luck plane: Sharpe 2.17 (5y)"}
    assert data["zero_label"] == "Sharpe 0"
    assert data["ruler_ticks"] == [-2, -1, 0, 1, 2, 3, 4]
    assert data["label_top"] == 10
    for marker in ('<aside id="details"', 'id="details-close"', 'id="link-gf"', 'id="link-edgar"', 'id="link-web"',
                   'rel="noopener"', '<section id="colorbar"', '<span>&ge; 4</span><span>2</span><span>&le; 0</span>'):
        assert marker in page
    assert re.findall(r'<button type="button" data-ticker="([^"]+)">', page) == PINNED_TOP_20
    assert re.search(r'data-ticker="TEC03"><span class="rank">1</span><span class="tk">TEC03</span>'
                     r'<span class="sr">3.49</span><span class="chip pass">DSR 0.97</span>', page)
    tec03 = next(t for t in data["towers"] if t["ticker"] == "TEC03")
    assert tec03["n_obs"] == 756 and tec03["above_luck_plane"] is True and tec03["website"] is None
    assert tec03["alpha_capm_ann"] == pytest.approx(0.7662, abs=1e-12) and tec03["t_alpha_capm"] == pytest.approx(6.46)


SAMPLE_TICKERS = sorted(f"{prefix}{i:02d}" for prefix, count in
                        [("TEC", 8), ("HLT", 8), ("FIN", 8), ("CDS", 8), ("IND", 7), ("ENE", 7), ("UTL", 7), ("STP", 7)]
                        for i in range(1, count + 1))


def test_page_executive_layer(screen_runs: tuple[Path, Path], tmp_path: Path):
    out = scene_threejs.write_page(screen_runs[0] / "screen_result.json", tmp_path / "skyline.html")
    page = out.read_text()
    assert _page_data(page)["verdict"] == {
        "n": 60, "pass": "1 of 20", "plane": "2.41",
        "line": "TEC03 clears the luck test (DSR \u2265 0.95): a 3-year record unlikely to be just the luckiest of 60 names.",
    }
    assert '<b id="stat-n">60</b>' in page
    assert '<b id="stat-pass">1 of 20</b>' in page
    assert '<b id="stat-plane">2.41</b><span>luck plane Sharpe (3y)</span>' in page
    assert ('<p id="verdict-line">TEC03 clears the luck test (DSR \u2265 0.95): a 3-year record unlikely to be just '
            'the luckiest of 60 names.</p>') in page
    assert '<details id="how-to-read" open>' in page and "if (window.innerWidth < 900) document.getElementById(\"how-to-read\").open = false;" in page


def test_verdict_line_cases():
    def scene(k: int, above: int, passing: list[str]) -> dict:
        towers = [{"ticker": t, "rank": i + 1, "dsr_passes": t in passing} for i, t in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"])]
        return {"window": "3y", "N": 495, "top_n": 20, "top_dsr_passing": k, "top_above_luck_plane": above, "towers": towers}

    assert scene_threejs.verdict_line(scene(0, 0, [])) == "No name's 3-year record beats what luck alone would hand the best of 495."
    assert scene_threejs.verdict_line(scene(0, 2, [])) == (
        "2 of the top 20 names rise above the luck plane, but none clears DSR \u2265 0.95: no 3-year record can yet "
        "be told apart from luck across 495 names.")
    assert scene_threejs.verdict_line(scene(4, 4, ["DDD", "AAA", "CCC", "BBB"])) == (
        "AAA, BBB, CCC and 1 more clear the luck test (DSR \u2265 0.95): a 3-year record unlikely to be just the "
        "luckiest of 495 names.")


def test_page_engineer_layer(screen_runs: tuple[Path, Path], tmp_path: Path):
    page = scene_threejs.write_page(screen_runs[0] / "screen_result.json", tmp_path / "skyline.html").read_text()
    data = _page_data(page)
    assert '<input id="search" type="search" list="tickers"' in page
    assert re.findall(r'<option value="([^"]+)"></option>', page) == SAMPLE_TICKERS
    assert re.search(r'event\.key === "Escape"\) \{\s*clearSelection\(\);', page)
    assert '<button id="copy-json" type="button">Copy JSON</button>' in page
    assert data["detail_fields"] == ["ticker", "sector", "rank", "sr_ann", "alpha_capm_ann", "t_alpha_capm", "alpha_ff3_ann",
                                     "t_alpha_ff3", "psr", "dsr", "dsr_passes", "above_luck_plane", "n_obs", "website"]
    assert 'data-panel="legend"' in page and 'data-panel="list"' in page
    assert "renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));" in page
    tiles = data["sector_tiles"]
    assert [t["name"] for t in tiles] == ["Consumer Discretionary", "Consumer Staples", "Energy", "Financials",
                                          "Health Care", "Industrials", "Information Technology", "Utilities"]
    assert tiles[0]["color"] == "#4e79a7" and len({t["color"] for t in tiles}) == 8
    assert page.count('<i class="swatch"') == 8


def _run_nav_logic(page: str, expression: str) -> object:
    """Run the page's own nav-logic script block in Node and return ``expression`` as JSON."""
    logic = re.search(r'<script id="nav-logic">(.*?)</script>', page, re.S)
    assert logic
    script = logic.group(1) + f"\nconst N = globalThis.SkylineNav;\nconsole.log(JSON.stringify({expression}));\n"
    proc = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def test_page_navigation_markup(screen_runs: tuple[Path, Path], tmp_path: Path):
    page = scene_threejs.write_page(screen_runs[0] / "screen_result.json", tmp_path / "skyline.html").read_text()
    assert ('<p id="controls-hint" class="caption">Drag the city to move \u00b7 drag sky to orbit \u00b7 scroll to zoom \u00b7 '
            'double-click a bar to fly to it \u00b7 R resets \u00b7 Layout: sector / by Sharpe \u00b7 zoom in to label nearby bars.') in page
    assert '<button id="reset-view" type="button">Reset view</button>' in page
    assert '<button id="fly-to" type="button">Fly to</button>' in page
    assert re.search(r'stage\.addEventListener\("dblclick", \(event\) => \{\s*const mesh = towerAt\(event\);\s*'
                     r'clearTimeout\(clickTimer\);\s*if \(mesh\) select\(mesh, true\);', page)
    assert "if (mesh) clickTimer = setTimeout(() => select(mesh, false), DOUBLE_CLICK_MS);" in page
    assert "controls.screenSpacePanning = false;" in page
    assert re.search(r"controls\.update\(\);\s*clampView\(\);", page)
    assert _page_data(page)["nav_bounds"] == {"x": [-12.2, 12.2], "y": [-5.5791, 10.1582], "z": [-12.2, 12.2]}


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js to run the page's navigation logic")
def test_page_navigation_logic(screen_runs: tuple[Path, Path], tmp_path: Path):
    page = scene_threejs.write_page(screen_runs[0] / "screen_result.json", tmp_path / "skyline.html").read_text()
    bounds = json.dumps(_page_data(page)["nav_bounds"])
    assert _run_nav_logic(page, f"[N.clampTarget({{x: 1, y: 2, z: -3}}, {bounds}), "
                                f"N.clampTarget({{x: 999, y: -999, z: -999}}, {bounds})]") == [
        {"x": 1, "y": 2, "z": -3}, {"x": 12.2, "y": -5.5791, "z": -12.2}]
    assert _run_nav_logic(page, '[N.dragMode(0, "floor"), N.dragMode(0, "tower"), N.dragMode(0, null), '
                                'N.dragMode(2, "tower"), N.dragMode(2, null), N.dragMode(1, "floor")]') == [
        "pan", "pan", "orbit", "pan", "pan", None]
    # Looking down -z: up/W move forward (-z), D moves right (+x), A left; S along -(0.6, 0.8) x 10.
    assert _run_nav_logic(page, '[N.keyPan("ArrowUp", {x: 0, z: -2}, 1), N.keyPan("W", {x: 0, z: -2}, 1), '
                                'N.keyPan("d", {x: 0, z: -2}, 1), N.keyPan("ArrowLeft", {x: 0, z: -2}, 1), '
                                'N.keyPan("s", {x: 3, z: 4}, 10), N.keyPan("q", {x: 0, z: -1}, 1)]') == [
        {"x": 0, "z": -1}, {"x": 0, "z": -1}, {"x": 1, "z": 0}, {"x": -1, "z": 0}, {"x": -6, "z": -8}, None]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js to run the page's link logic")
def test_page_link_ticker_mapping(screen_runs: tuple[Path, Path], tmp_path: Path):
    page = scene_threejs.write_page(screen_runs[0] / "screen_result.json", tmp_path / "skyline.html").read_text()
    links = json.dumps(_page_data(page)["links"])
    assert _run_nav_logic(page, f"(() => {{ const L = {links}; return ["
                                f"SkylineLinks.forTicker(L.edgar, 'BRK.B', true), "
                                f"SkylineLinks.forTicker(L.google_finance, 'BRK.B', false), "
                                f"SkylineLinks.forTicker(L.website_search, 'BRK.B', false), "
                                f"SkylineLinks.forTicker(L.edgar, 'AAPL', true)]; }})()") == [
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&ticker=BRK-B",
        "https://www.google.com/finance?q=BRK.B",
        "https://www.google.com/search?q=BRK.B+investor+relations",
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&ticker=AAPL",
    ]
    assert "SkylineLinks.forTicker(DATA.links.edgar, t.ticker, true)" in page
    assert "SkylineLinks.forTicker(DATA.links.google_finance, t.ticker, false)" in page


def test_website_column_passes_through(tmp_path: Path):
    universe = tmp_path / "universe.csv"
    universe.write_text(
        "ticker,sector,website\n"
        "TEC03,Information Technology,https://tec03.example.com/ir\n"
        "FIN02,Financials,\n"
        "IND04,Industrials,javascript:alert(1)\n"
    )
    out = tmp_path / "out"
    run_script("screen.py", "--universe", str(universe), "--out", str(out))
    rows = {r["ticker"]: r for r in json.loads((out / "screen_result.json").read_text())["universe"]}
    assert rows["TEC03"]["website"] == "https://tec03.example.com/ir"
    assert "website" not in rows["FIN02"]
    assert rows["IND04"]["website"] == "javascript:alert(1)"
    page = scene_threejs.render_page(scene_threejs.build_scene(json.loads((out / "screen_result.json").read_text())))
    towers = {t["ticker"]: t for t in _page_data(page)["towers"]}
    assert towers["TEC03"]["website"] == "https://tec03.example.com/ir"
    assert towers["FIN02"]["website"] is None and towers["IND04"]["website"] is None  # fallback search link is used
    assert "javascript:alert" not in page


def test_threejs_open_flag(screen_runs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    opened: list[str] = []
    monkeypatch.setattr(scene_threejs.webbrowser, "open", opened.append)
    src = str(screen_runs[0] / "screen_result.json")
    page = tmp_path / "a.html"
    assert scene_threejs.main([src, "--out", str(page)]) == 0
    assert opened == []
    assert scene_threejs.main([src, "--out", str(page), "--open"]) == 0
    assert opened == [page.resolve().as_uri()] and opened[0].startswith("file:///")


def test_render_skyline_html_and_open(screen_runs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                      no_bpy: None, no_blender: None, capsys: pytest.CaptureFixture[str]):
    opened: list[str] = []
    monkeypatch.setattr(scene_threejs.webbrowser, "open", opened.append)
    src = str(screen_runs[0] / "screen_result.json")
    png = tmp_path / "sky.png"
    assert render_skyline.main([src, "--out", str(png), "--html"]) == 0
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and len(png.read_bytes()) > 20_000
    page = png.with_suffix(".html")
    assert re.findall(r'<button type="button" data-ticker="([^"]+)">', page.read_text()) == PINNED_TOP_20
    assert capsys.readouterr().out.splitlines() == [str(png.resolve()), str(page.resolve())]
    assert opened == []
    custom = tmp_path / "pages" / "interactive.html"
    assert render_skyline.main([src, "--out", str(png), "--html", str(custom), "--open"]) == 0
    assert custom.is_file() and opened == [custom.resolve().as_uri()]
    assert render_skyline.main([src, "--out", str(tmp_path / "only.png")]) == 0
    assert not (tmp_path / "only.html").exists() and len(opened) == 1


# --- 7. Rendering to PNG ----------------------------------------------------------------

def test_matplotlib_skyline_contents(screen_runs: tuple[Path, Path], tmp_path: Path):
    src = screen_runs[0] / "screen_result.json"
    result = json.loads(src.read_text())
    fig, ax, stats = scene_python.build_figure(result)
    try:
        assert stats["window"] == "3y"
        assert stats["tower_count"] == result["metadata"]["universe_size"] == 60
        assert stats["plane_z"] == result["metadata"]["luck_horizon_by_window"]["3y"]["headline"]
        assert stats["plane_z"] == pytest.approx(2.407983946474636, abs=1e-12)
        top_rows = [r["ticker"] for r in result["top"]]
        assert stats["labelled"] == top_rows[:10]
        assert set(stats["labelled"]) <= set(top_rows)
        assert sorted(stats["outlined"]) == sorted(top_rows)
        assert fig.get_size_inches().tolist() == [16.0, 10.0] and fig.dpi == 100
        drawn = {t.get_text() for t in ax.texts}
        assert set(stats["labelled"]) <= drawn and "TEC03" in drawn
    finally:
        scene_python.plt.close(fig)
    _, _, five_year = scene_python.build_figure(result, "5y")
    assert five_year["plane_z"] == pytest.approx(2.173536589667446, abs=1e-12)

    out = tmp_path / "skyline.png"
    scene_python.main([str(src), "--out", str(out)])
    data = out.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 20_000
    width, height = int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    assert (width, height) == (1600, 1000)


FAKE_SCENE = """import os, sys
import bpy
args = sys.argv[sys.argv.index("--") + 1:]
assert args[0] == "--render", args
with open(args[1], "w") as fh:
    fh.write("in-process pid=%d stub=%s" % (os.getpid(), bpy.IS_STUB))
"""


def _fake_blender(directory: Path, name: str, exit_code: int = 0, write_png: bool = True) -> Path:
    """An executable standing in for Blender: logs its argv, optionally writes the PNG."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"open({str(directory / (name + '.argv'))!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
        f"if {write_png!r}:\n"
        "    open(sys.argv[sys.argv.index('--render') + 1], 'w').write('binary:' + sys.argv[0])\n"
        f"if {exit_code}:\n"
        "    sys.stderr.write('Error: Python script failed\\nTraceback line one\\nKeyError: boom\\n')\n"
        f"sys.exit({exit_code})\n"
    )
    path.chmod(0o755)
    return path


def _recorded_argv(binary: Path) -> list[str]:
    return json.loads(binary.with_name(binary.name + ".argv").read_text())


@pytest.fixture
def no_bpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "bpy", None)  # makes `import bpy` raise ImportError


@pytest.fixture
def no_blender(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    empty = tmp_path / "empty-path"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.delenv("SHARPE_OBSERVATORY_BLENDER", raising=False)


def test_render_tier1_bpy_module(screen_runs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                 capsys: pytest.CaptureFixture[str]):
    stub = types.ModuleType("bpy")
    stub.IS_STUB = True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bpy", stub)

    def fake_script(result: Path, script: Path) -> Path:
        script.write_text(FAKE_SCENE)
        return script

    monkeypatch.setattr(render_skyline, "write_bpy_script", fake_script)
    binary = _fake_blender(tmp_path / "bin", "blender")
    monkeypatch.setenv("PATH", str(binary.parent))
    png = tmp_path / "out" / "sky.png"
    argv_before = list(sys.argv)
    assert render_skyline.main([str(screen_runs[0] / "screen_result.json"), "--out", str(png)]) == 0
    assert png.read_text() == f"in-process pid={os.getpid()} stub=True"
    assert not binary.with_name("blender.argv").exists()
    assert sys.argv == argv_before
    captured = capsys.readouterr()
    assert captured.out.strip() == str(png.resolve())
    assert "rendered with tier 1 (bpy module)" in captured.err


def test_render_tier2_blender_binary(screen_runs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                     no_bpy: None):
    src = screen_runs[0] / "screen_result.json"
    on_path = _fake_blender(tmp_path / "path", "blender")
    from_env = _fake_blender(tmp_path / "env", "blender-env")
    explicit = _fake_blender(tmp_path / "explicit", "blender-flag")
    png = tmp_path / "sky.png"
    script = (tmp_path / "skyline_bpy.py").resolve()
    monkeypatch.setenv("PATH", str(on_path.parent))
    monkeypatch.delenv("SHARPE_OBSERVATORY_BLENDER", raising=False)

    assert render_skyline.render(src, png) == f"tier 2 (blender binary {on_path})"
    assert _recorded_argv(on_path) == ["-b", "--factory-startup", "--python", str(script),
                                       "--python-exit-code", "1", "--", "--render", str(png.resolve())]
    assert script.read_text().startswith("# Sharpe Observatory skyline: generated by scene_blender.py.")
    monkeypatch.setenv("SHARPE_OBSERVATORY_BLENDER", str(from_env))
    assert render_skyline.render(src, png) == f"tier 2 (blender binary {from_env})"
    assert render_skyline.render(src, png, str(explicit)) == f"tier 2 (blender binary {explicit})"
    assert png.read_text() == f"binary:{explicit}"


def test_render_tier3_matplotlib(screen_runs: tuple[Path, Path], tmp_path: Path, no_bpy: None, no_blender: None,
                                 capsys: pytest.CaptureFixture[str]):
    png = tmp_path / "sky.png"
    assert render_skyline.main([str(screen_runs[0] / "screen_result.json"), "--out", str(png)]) == 0
    data = png.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 20_000
    assert "rendered with tier 3 (matplotlib)" in capsys.readouterr().err


def test_render_falls_back_when_blender_fails(screen_runs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
                                               tmp_path: Path, no_bpy: None, capsys: pytest.CaptureFixture[str]):
    failing = _fake_blender(tmp_path / "bin", "blender", exit_code=3, write_png=False)
    monkeypatch.setenv("PATH", str(failing.parent))
    monkeypatch.delenv("SHARPE_OBSERVATORY_BLENDER", raising=False)
    png = tmp_path / "sky.png"
    assert render_skyline.render(screen_runs[0] / "screen_result.json", png) == "tier 3 (matplotlib)"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (f"[render] tier 2 (blender binary {failing}) skipped: blender exited with code 3: "
            "Error: Python script failed | Traceback line one | KeyError: boom") in capsys.readouterr().err


def test_render_without_matplotlib_is_the_only_hard_failure(screen_runs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
                                                             tmp_path: Path, no_bpy: None, no_blender: None,
                                                             capsys: pytest.CaptureFixture[str]):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(SystemExit) as info:
        render_skyline.main([str(screen_runs[0] / "screen_result.json"), "--out", str(tmp_path / "sky.png")])
    assert info.value.code == 1
    assert capsys.readouterr().err.endswith("error: matplotlib is not installed; run pip install -r requirements.txt\n")


@pytest.mark.skipif(shutil.which("blender") is None, reason="needs a real blender binary on PATH")
def test_real_blender_renders_sample_skyline(screen_runs: tuple[Path, Path], tmp_path: Path, no_bpy: None):
    png = tmp_path / "skyline.png"
    assert render_skyline.render(screen_runs[0] / "screen_result.json", png).startswith("tier 2")
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --- 8. Full US market universe and batched fetching -----------------------------------

NASDAQ_LISTED = (
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
    "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
    "QQQ|Invesco QQQ Trust|G|N|N|100|Y|N\n"
    "ZXZZT|NASDAQ TEST STOCK|G|Y|N|100|N|N\n"
    "ABC$A|ABC Corp Preferred A|G|N|N|100|N|N\n"
    "NXTS|Example NextShares Fund|G|N|N|100|N|Y\n"
    "AJAXU|Ajax Capital Acquisition Units|G|N|N|100|N|N\n"
    "AJAXW|Ajax Capital Acquisition Warrant|G|N|N|100|N|N\n"
    "AJAXR|Ajax Capital Acquisition Rights|G|N|N|100|N|N\n"
    "NNCO|NobleNotesCo Inc. - Common Stock|Q|N|N|100|N|N\n"
    " MSFT |Microsoft Corporation - Common Stock|Q|N|N|100|N|N\n"
    "File Creation Time: 1012202618:01|||||||\n"
)
OTHER_LISTED = (
    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
    "BRK.B|Berkshire Hathaway Inc. New Common Stock|N|BRK.B|N|40|N|BRK.B\n"
    "SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY\n"
    "XYZ|XYZ Corp Common Stock|A|XYZ|N|100|N|XYZ\n"
    "ARCX|Arca Listed Corp|P|ARCX|N|100|N|ARCX\n"
    "CBZ|Cboe Listed Corp|Z|CBZ|N|100|N|CBZ\n"
    "IEXS|IEX Listed Corp|V|IEXS|N|100|N|IEXS\n"
    "PRF$B|Pref Corp Preferred B|N|PRFpB|N|100|N|PRF-B\n"
    "TSTX|NYSE Test Stock|N|TSTX|N|100|Y|TSTX\n"
    "AAC.U|Ajax Capital Units|N|AAC.U|N|100|N|AAC=\n"
    "ABC.WS|ABC Corp Warrants|N|ABC.WS|N|100|N|ABC+\n"
    "XYZN|XYZ Corp 5.25% Notes due 2029|N|XYZN|N|100|N|XYZN\n"
    "PFDX|PFD Corp 6% Series A PREFERRED Stock|N|PFDX|N|100|N|PFDX\n"
    "BABA|Alibaba Group Holding Limited American Depositary Shares each representing eight Ordinary share|N|BABA|N|100|N|BABA\n"
    "BKXC|Bank X Depositary Shares each representing 1/40th interest in a share of Preferred Stock Series C|N|BKXC|N|100|N|BKXC\n"
    "DEPX|Example Co Depositary Shares each representing a 1/1000th interest|N|DEPX|N|100|N|DEPX\n"
    "ADRP|Example Bank American Depositary Shares each representing one Preferred Share|N|ADRP|N|100|N|ADRP\n"
    "TPX|Example Capital Trust Preferred Securities|N|TPX|N|100|N|TPX\n"
    "UTL|Unitil Corporation Common Stock|N|UTL|N|100|N|UTL\n"
    "File Creation Time: 1012202618:01||||||\n"
)


def test_symbol_directory_footer_is_skipped():
    frame = fetch_data._read_symbol_directory(NASDAQ_LISTED, "fixture", ["Symbol"])
    assert len(frame) == 10
    assert frame["Symbol"].tolist()[-1] == " MSFT "
    with pytest.raises(ValueError, match=r"^fixture: unexpected schema .*missing \['ETF'\]$"):
        fetch_data._read_symbol_directory("Symbol|Name\nAAA|A\n", "fixture", ["Symbol", "ETF"])


def test_nasdaq_listed_filters():
    # Kept: AAPL, MSFT (whitespace stripped), NNCO ("NobleNotesCo" has no whole-word "notes").
    # Dropped: QQQ (ETF), ZXZZT (test issue), ABC$A ("$": preferred), NXTS (NextShares),
    # AJAXU / AJAXW / AJAXR (units, warrant, rights by name), and the footer line.
    parsed = fetch_data.parse_nasdaq_listed(NASDAQ_LISTED)
    assert list(zip(parsed["ticker"], parsed["sector"])) == [("AAPL", "NASDAQ"), ("NNCO", "NASDAQ"), ("MSFT", "NASDAQ")]


def test_other_listed_filters_and_exchange_mapping():
    parsed = fetch_data.parse_other_listed(OTHER_LISTED)
    # Name filter drops units, warrants, notes, preferred (any case), trust preferred and plain
    # "Depositary Shares" (BKXC, DEPX). American depositary shares are kept (BABA) unless another
    # rule matches (ADRP: preferred). "Unitil" contains "unit" but not as a whole word.
    assert list(zip(parsed["ticker"], parsed["sector"])) == [
        ("BRK.B", "NYSE"), ("XYZ", "NYSE American"), ("ARCX", "NYSE Arca"), ("CBZ", "Cboe BZX"), ("IEXS", "Other (V)"),
        ("BABA", "NYSE"), ("UTL", "NYSE")]


def test_us_market_universe_sources_and_sanity_bound():
    def stocks(n: int, prefix: str) -> pd.DataFrame:
        return pd.DataFrame({"ticker": [f"{prefix}{i:04d}" for i in range(n)], "sector": "NYSE"})

    universe = fetch_data.fetch_us_market_universe([("a", lambda: stocks(1500, "A")), ("b", lambda: stocks(1500, "B"))])
    assert len(universe) == 3000 and universe["ticker"].iloc[0] == "A0000" and list(universe.columns) == ["ticker", "sector"]
    with pytest.raises(RuntimeError, match=r"^US market symbol directory: only 2999 stocks after filters, fewer than 3000$"):
        fetch_data.fetch_us_market_universe([("a", lambda: stocks(1500, "A")), ("b", lambda: stocks(1499, "B"))])

    def down() -> pd.DataFrame:
        raise requests.ConnectionError("unreachable")

    with pytest.raises(RuntimeError, match=r"^could not fetch the US market symbol directory: a: ConnectionError: "
                                           r"unreachable; b: ValueError: nasdaqtrader.com nasdaqlisted.txt: unexpected schema "
                                           r"\['Symbol', 'Name'\]; missing \['Test Issue', 'ETF', 'NextShares'\]$"):
        fetch_data.fetch_us_market_universe(
            [("a", down), ("b", lambda: fetch_data.parse_nasdaq_listed("Symbol|Name\nAAA|A\n"))])


def test_index_flag_selects_the_universe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[str] = []
    sample_universe = pd.read_csv(fetch_data.SAMPLE_DIR / "universe.csv")
    monkeypatch.setattr(fetch_data, "fetch_sp500_universe", lambda: (calls.append("sp500"), sample_universe)[1])
    monkeypatch.setattr(fetch_data, "fetch_us_market_universe", lambda: (calls.append("us-all"), sample_universe)[1])
    seen_cache: list[Path | None] = []

    def live_stub(universe: pd.DataFrame, throttle_s: float = 0.5, cache_dir: Path | None = None) -> fetch_data.Dataset:
        seen_cache.append(cache_dir)
        return _sample_live_stub(universe)

    monkeypatch.setattr(screen, "fetch_live", live_stub)
    assert screen.main(["--live", "--out", str(tmp_path / "a")]) == 0
    assert screen.main(["--live", "--index", "us-all", "--cache-dir", str(tmp_path / "cache"), "--out", str(tmp_path / "b")]) == 0
    assert calls == ["sp500", "us-all"]
    assert seen_cache == [None, tmp_path / "cache"]
    with pytest.raises(SystemExit):
        screen.main(["--cache-dir", str(tmp_path / "cache"), "--out", str(tmp_path / "c")])


def test_chunking_and_download_call(monkeypatch: pytest.MonkeyPatch):
    assert [len(c) for c in fetch_data.chunked([f"T{i}" for i in range(120)], 50)] == [50, 50, 20]
    yfinance = pytest.importorskip("yfinance")
    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(yfinance, "download", lambda tickers, **kwargs: (seen.append((tickers, kwargs)), pd.DataFrame())[1])
    fetch_data.yahoo_download(["AAPL", "BRK.B", "BF.B"])
    tickers, kwargs = seen[0]
    assert tickers == "AAPL BRK-B BF-B"
    assert kwargs["interval"] == "1d" and kwargs["auto_adjust"] is True and kwargs["group_by"] == "column"
    assert kwargs["progress"] is False


def _batch_frame(columns: dict[str, list[float]]) -> pd.DataFrame:
    index = pd.DatetimeIndex(["2026-10-08", "2026-10-09"]).tz_localize("America/New_York")
    data = {("Close", sym): values for sym, values in columns.items()}
    data.update({("Open", sym): values for sym, values in columns.items()})
    return pd.DataFrame(data, index=index)


def test_split_batch_routes_empty_tickers_to_second_pass():
    nan = float("nan")
    frame = _batch_frame({"AAPL": [1.0, 2.0], "BRK-B": [3.0, 4.0], "DEAD": [nan, nan]})
    found, missing = fetch_data.split_batch(frame, ["AAPL", "BRK.B", "DEAD", "GONE"])
    assert sorted(found) == ["AAPL", "BRK.B"]
    assert found["BRK.B"].tolist() == [3.0, 4.0] and found["BRK.B"].name == "BRK.B"
    assert list(found["AAPL"].index.strftime("%Y-%m-%d")) == ["2026-10-08", "2026-10-09"] and found["AAPL"].index.tz is None
    assert missing == ["DEAD", "GONE"]
    assert fetch_data.split_batch(pd.DataFrame(), ["X", "Y"]) == ({}, ["X", "Y"])
    flat = pd.DataFrame({"Close": [5.0, 6.0]}, index=pd.DatetimeIndex(["2026-10-08", "2026-10-09"]))
    found, missing = fetch_data.split_batch(flat, ["ONLY"])
    assert found["ONLY"].tolist() == [5.0, 6.0] and missing == []


def test_batched_fetch_resumes_from_cache_and_falls_back(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    cache = tmp_path / "cache"
    cache.mkdir()
    pd.Series([9.0, 9.5], index=pd.to_datetime(["2026-10-08", "2026-10-09"]), name="close").to_csv(
        cache / "CACHED.csv", index_label="date")
    requested: list[list[str]] = []
    second: list[str] = []
    pauses: list[float] = []
    nan = float("nan")

    def download(chunk: list[str]) -> pd.DataFrame:
        requested.append(list(chunk))
        if chunk[0] == "C":
            raise ConnectionError("chunk refused")
        return _batch_frame({t: ([nan, nan] if t == "B" else [1.0, 2.0]) for t in chunk})

    def per_ticker(ticker: str) -> pd.Series:
        second.append(ticker)
        return pd.Series([7.0], index=pd.to_datetime(["2026-10-09"]), name=ticker)

    tickers = ["A", "B", "CACHED", "C", "D", "E"]
    prices, failures = fetch_data.fetch_prices_batched(tickers, download=download, fetch_close=per_ticker,
                                                       throttle_s=1.0, sleep=pauses.append, cache_dir=cache, batch_size=2)
    assert requested == [["A", "B"], ["C", "D"], ["E"]]          # CACHED was skipped
    assert second == ["B", "C", "D"]                              # all-NaN B, and the whole failed chunk
    assert pauses == [1.0, 2.0, 1.0, 1.0]                         # chunk pauses (doubled after the failure), then 2 per-ticker
    assert list(prices.columns) == tickers and failures == {}
    assert prices["CACHED"].dropna().tolist() == [9.0, 9.5]
    assert sorted(p.stem for p in cache.glob("*.csv")) == ["A", "B", "C", "CACHED", "D", "E"]
    err = capsys.readouterr().err
    assert "[fetch] 1 of 6 tickers read from" in err and "[fetch] chunk 3/3: 3/6 tickers, 3 queued for retry" in err

    def must_not_download(chunk: list[str]) -> pd.DataFrame:
        raise AssertionError(f"downloaded {chunk} although every ticker is cached")

    resumed, _ = fetch_data.fetch_prices_batched(tickers, download=must_not_download, fetch_close=per_ticker, cache_dir=cache)
    assert list(resumed.columns) == tickers and resumed["B"].dropna().tolist() == [7.0]


def test_page_uses_instanced_towers(screen_runs: tuple[Path, Path], tmp_path: Path):
    page = scene_threejs.write_page(screen_runs[0] / "screen_result.json", tmp_path / "skyline.html").read_text()
    assert "const towerMesh = new THREE.InstancedMesh(box, new THREE.MeshStandardMaterial(" in page
    assert re.search(r"const hit = raycaster\.intersectObject\(towerMesh, false\)\[0\];\s*return hit \? towers\[hit\.instanceId\] : null;", page)
    assert "raycaster.intersectObjects([towerMesh, floorHit], false)" in page
    assert "towerMesh.setColorAt(handle.index, color);" in page and "towerMesh.instanceColor.needsUpdate = true;" in page
    assert "towerMesh.computeBoundingSphere();" in page
    assert "new THREE.Mesh(box" not in page


def test_matplotlib_scales_to_thousands():
    rows = [{"ticker": f"T{i}", "sector": ["NASDAQ", "NYSE", "NYSE American"][i % 3], "rank": i + 1,
             "in_top": i < 20, "sr_ann": {"3y": 2.0 - i / 1000}, "t_alpha_ff3": 1.0, "alpha_ff3_ann": 0.1,
             "alpha_capm_ann": 0.1, "t_alpha_capm": 1.0, "psr": 0.9, "dsr": 0.1, "dsr_passes": False,
             "above_luck_plane": False, "n_obs": 756} for i in range(1500)]
    result = {"metadata": {"windows": {"3y": 756}, "window_label": "3y", "N": 1500, "top_n": 20,
                           "top_dsr_passing": 0, "top_above_luck_plane": 0,
                           "luck_horizon_by_window": {"3y": {"headline": 2.5}}}, "universe": rows}
    fig, _, stats = scene_python.build_figure(result)
    scene_python.plt.close(fig)
    assert stats["tower_count"] == 1500 and stats["bar_size"] == 0.55 and len(stats["labelled"]) == 10


# --- 9. Market-scale data hygiene ------------------------------------------------------

def _hygiene_dir(tmp_path: Path) -> Path:
    """30 random names plus ZERO (one 0.0 close inside the 1y window) and FLAT (a constant price)."""
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2024-01-01", periods=301, name="date")
    returns = pd.DataFrame(rng.normal(0.0005, 0.01, (300, 31)), index=dates[1:],
                           columns=[f"N{i:02d}" for i in range(30)] + ["ZERO"])
    prices = pd.concat([pd.DataFrame(100.0, index=dates[:1], columns=returns.columns), 100 * (1 + returns).cumprod()])
    prices.loc[dates[200], "ZERO"] = 0.0
    prices["FLAT"] = 100.0
    dataset = fetch_data.Dataset(
        universe=pd.DataFrame({"ticker": list(prices.columns), "sector": "Test"}),
        prices=prices,
        rf_daily=pd.Series(0.0, index=dates),
        factors=pd.DataFrame(rng.normal(0.0, 0.01, (300, 3)), index=dates[1:], columns=["mkt_rf", "smb", "hml"]),
        mode="test",
    )
    out = tmp_path / "data"
    fetch_data.write_directory(dataset, out)
    return out


def test_zero_close_is_cleaned_and_name_still_screens(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    data_dir = _hygiene_dir(tmp_path)
    dataset = fetch_data.load_directory(data_dir)
    assert "[clean] 1 non-positive close set to missing (ZERO)" in capsys.readouterr().err
    assert dataset.prices["ZERO"].isna().sum() == 1 and (dataset.prices.fillna(1.0) > 0).all().all()
    result = screen.screen(dataset, window=252, top=5)
    zero = next(r for r in result["universe"] if r["ticker"] == "ZERO")
    assert zero["n_obs"] == 250 and zero["rank"] is not None and math.isfinite(zero["sr_ann"]["1y"])


def test_cache_read_cleans_zero_closes(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    cache = tmp_path / "cache"
    cache.mkdir()
    pd.Series([10.0, 0.0, 11.0], index=pd.to_datetime(["2026-10-07", "2026-10-08", "2026-10-09"]),
              name="close").to_csv(cache / "DEC.csv", index_label="date")
    prices, _ = fetch_data.fetch_prices_batched(["DEC"], download=lambda c: pd.DataFrame(), cache_dir=cache)
    assert prices["DEC"].tolist()[:1] == [10.0] and math.isnan(prices["DEC"].tolist()[1])
    assert "[clean] 1 non-positive close set to missing (DEC)" in capsys.readouterr().err


def test_nonfinite_sharpe_is_excluded_from_the_plane(tmp_path: Path):
    dataset = fetch_data.load_directory(_hygiene_dir(tmp_path))
    result = screen.screen(dataset, window=252, top=5)
    meta = result["metadata"]
    assert meta["excluded_unscoreable"] == {"FLAT": "flat price"}
    assert meta["N"] == 31 and meta["universe_size"] == 32
    assert math.isfinite(meta["luck_horizon_ann"]["headline"]) and math.isfinite(meta["luck_horizon_ann"]["cross_sectional"])
    flat = next(r for r in result["universe"] if r["ticker"] == "FLAT")
    assert flat["rank"] is None and flat["in_top"] is False


def test_summary_print_survives_missing_plane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture[str]):
    data_dir = _hygiene_dir(tmp_path)
    assert screen.main(["--data-dir", str(data_dir), "--window", "252", "--top", "5", "--out", str(tmp_path / "a")]) == 0
    assert re.search(r"^N=31 luck_plane_ann=\d+\.\d{3} top 5:", capsys.readouterr().out, re.M)
    real = screen.screen

    def plane_missing(*args: object, **kwargs: object) -> dict:
        result = real(*args, **kwargs)
        result["metadata"]["luck_horizon_ann"]["headline"] = None
        return result

    monkeypatch.setattr(screen, "screen", plane_missing)
    assert screen.main(["--data-dir", str(data_dir), "--window", "252", "--top", "5", "--out", str(tmp_path / "b")]) == 0
    assert "N=31 luck_plane_ann=n/a top 5:" in capsys.readouterr().out


# --- 10. "By Sharpe" layout ------------------------------------------------------------

def test_page_layout_toggle_markup(screen_runs: tuple[Path, Path], tmp_path: Path):
    page = scene_threejs.write_page(screen_runs[0] / "screen_result.json", tmp_path / "skyline.html").read_text()
    data = _page_data(page)
    assert '<button type="button" data-layout="sector" aria-pressed="true">By sector</button>' in page
    assert '<button type="button" data-layout="rank" aria-pressed="false">By Sharpe</button>' in page
    assert '<div id="layout-note" class="panel" hidden></div>' in page
    assert 'note.textContent = `sorted by Sharpe (${current})`;' in page
    assert 'navBounds = layout === "rank" ? DATA.nav_bounds_rank : DATA.nav_bounds;' in page
    assert "const c = SkylineNav.clampTarget(t, navBounds);" in page
    assert 'for (const item of sectorDecor) item.visible = layout === "sector";' in page
    assert "if (k === 1) { shuffle = null; towerMesh.computeBoundingSphere(); }" in page
    assert 'if (layout === "rank") { updateLayoutNote(); arrange(800); }' in page
    # 60 names: ceil(sqrt(60)) = 8 columns of pitch 1.4 -> half-width 8 * 1.4 / 2 + 1.5 + 2 = 9.1.
    assert data["rank_layout"] == {"cols": 8, "pitch": 1.4}
    assert data["nav_bounds_rank"] == {"x": [-9.1, 9.1], "y": [-5.5791, 10.1582], "z": [-9.1, 9.1]}
    assert data["floor_extent"] == {"width": 17.4, "depth": 17.4}


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js to run the page's layout logic")
def test_rank_layout_order_and_serpentine(screen_runs: tuple[Path, Path], tmp_path: Path):
    page = scene_threejs.write_page(screen_runs[0] / "screen_result.json", tmp_path / "skyline.html").read_text()
    # Sharpe: B 3 > D 2 > A 1 > E 0.5 > F 0.5 (tie, by ticker) > C NaN > G null (missing last, by
    # ticker). 2 columns, pitch 1: rows 0..3 (z = 1.5 - row), x = 0.5 - col; odd rows (1, 3) are
    # reversed, so G, alone in row 3, lands in column 1.
    cells = _run_nav_logic(page, 'N.rankLayout([{ticker: "A", sr: 1}, {ticker: "B", sr: 3}, {ticker: "C", sr: NaN}, '
                                 '{ticker: "D", sr: 2}, {ticker: "F", sr: 0.5}, {ticker: "E", sr: 0.5}, '
                                 '{ticker: "G", sr: null}], 2, 1)')
    assert [(c["ticker"], c["row"], c["col"], c["x"], c["z"]) for c in cells] == [
        ("B", 0, 0, 0.5, 1.5), ("D", 0, 1, -0.5, 1.5),
        ("A", 1, 1, -0.5, 0.5), ("E", 1, 0, 0.5, 0.5),
        ("F", 2, 0, 0.5, -0.5), ("C", 2, 1, -0.5, -0.5),
        ("G", 3, 1, -0.5, -1.5),
    ]


# --- 11. Publishing assets and docs ----------------------------------------------------

def test_icon_is_deterministic_and_sized(tmp_path: Path):
    import make_icon

    first = make_icon.write_icons(tmp_path / "a")
    second = make_icon.write_icons(tmp_path / "b")
    for a, b in zip(first, second):
        assert a.read_bytes() == b.read_bytes()
    for path, size in zip(first, (512, 128)):
        data = path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert (int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")) == (size, size)
        assert (REPO / "assets" / path.name).read_bytes() == data  # the committed icon is the generator's output


def test_publishing_docs_exist_and_links_resolve():
    for name in ("README.md", "PRIVACY.md", "SUPPORT.md", "CHANGELOG.md", "LICENSE", "docs/SUBMISSION.md"):
        assert (REPO / name).is_file(), name
    for doc in ("README.md", "PRIVACY.md", "SUPPORT.md", "CHANGELOG.md", "docs/SUBMISSION.md"):
        text = (REPO / doc).read_text()
        targets = re.findall(r"\]\(([^)\s]+)\)", text) + re.findall(r'src="([^"]+)"', text)
        local = [t for t in targets if not t.startswith(("http://", "https://", "#", "mailto:"))]
        for target in local:
            assert ((REPO / doc).parent / target).is_file(), f"{doc} links to missing {target}"
    readme = (REPO / "README.md").read_text()
    for link in ("PRIVACY.md", "SUPPORT.md", "CHANGELOG.md", "assets/icon-512.png"):
        assert link in readme
    manifest = json.loads((REPO / ".claude-plugin" / "plugin.json").read_text())
    assert manifest["repository"] == "https://github.com/rajib2k5/quant-3d-sharpe-viz-claude-plugin"


# --- 12. Zoom-aware ticker labels ------------------------------------------------------

def test_page_label_toggle_markup(screen_runs: tuple[Path, Path], tmp_path: Path):
    page = scene_threejs.write_page(screen_runs[0] / "screen_result.json", tmp_path / "skyline.html").read_text()
    assert '<button id="labels-toggle" type="button" aria-pressed="true" title="Ticker labels in the scene">Labels</button>' in page
    assert _page_data(page)["label_lod"] == {"always_top": 20, "budget": 100, "max_distance": 24.0, "full_distance": 14.0,
                                             "full_scale": 0.25, "max_scale": 0.45, "candidates": 400,
                                             "declutter": {"char_width": 7, "padding": 10, "height": 18, "cell": 64}}
    assert "maxDistance: labelFade.maxDistance, fullDistance: labelFade.fullDistance" in page
    assert "labelFade = SkylineNav.labelFade(framedDistance, DATA.label_lod);" in page
    assert re.search(r"clampView\(\);\s*updateLabels\(\);\s*renderer\.render", page)
    assert 'if (t.top) handle.userData.label = label(t.ticker, glow ? "label" : "label dim", t.x, 0, -t.y);' in page
    assert "own.visible = placed.has(handle.userData.tower.ticker);" in page
    assert "const placed = new Set(SkylineNav.declutter(candidates, { budget: lod.budget," in page
    assert "const candidates = selected ? [candidate(selected, true)] : [];" in page
    assert "camera.position, viewProj, { alwaysTop: lod.always_top, budget: lod.candidates," in page


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js to run the page's label logic")
def test_label_selection_logic(screen_runs: tuple[Path, Path], tmp_path: Path):
    page = scene_threejs.write_page(screen_runs[0] / "screen_result.json", tmp_path / "skyline.html").read_text()
    # Camera at the origin looking down -z: view = identity, projection = 90 degree FOV, aspect 1,
    # near 0.1, far 100 (column-major), so NDC = (x / -z, y / -z). Always the top 2 ranks; budget 4;
    # fade 5 -> 10 units. A (rank 1, far) and B (rank 2, behind the camera) are always labelled;
    # C (d 3) is near at full opacity; D (d 7) fades to (10 - 7) / (10 - 5) = 0.6; E (d 8) is cut
    # by the budget; F is behind the camera; G (d 12) is beyond maxDistance; H (d 5.83) is close but
    # below the frame (NDC y = -5 / 3), so it must not take D's place.
    proj = "[1,0,0,0, 0,1,0,0, 0,0,-1.002002002002002,-1, 0,0,-0.2002002002002002,0]"
    picks = _run_nav_logic(page, 'N.pickLabels(['
                                 '{ticker: "E", rank: null, x: 0, y: 0, z: -8}, {ticker: "A", rank: 1, x: 0, y: 0, z: -100}, '
                                 '{ticker: "G", rank: 7, x: 0, y: 0, z: -12}, {ticker: "D", rank: null, x: 0, y: 0, z: -7}, '
                                 '{ticker: "F", rank: null, x: 0, y: 0, z: 2}, {ticker: "C", rank: 3, x: 0, y: 0, z: -3}, '
                                 '{ticker: "H", rank: null, x: 0, y: -5, z: -3}, '
                                 '{ticker: "B", rank: 2, x: 0, y: 0, z: 5}], {x: 0, y: 0, z: 0}, ' + proj + ', '
                                 '{alwaysTop: 2, budget: 4, maxDistance: 10, fullDistance: 5})')
    assert picks == [{"ticker": "A", "opacity": 1}, {"ticker": "B", "opacity": 1},
                     {"ticker": "C", "opacity": 1}, {"ticker": "D", "opacity": 0.6}]


def test_flat_and_stale_prices_are_unscoreable(tmp_path: Path):
    # FLAT: every close equal. STALE: 60% of window returns exactly zero (halted / unrefreshed
    # quotes). PARTLY: 40% zero; HALF: exactly 50% zero (the rule is "more than 50%"), both scored.
    # SHORT: no prices for the first 100 days, so it fails the 90% coverage rule. The risk-free rate
    # varies over time, so FLAT's excess returns are not constant: only the raw-return test sees it.
    rng = np.random.default_rng(9)
    dates = pd.bdate_range("2024-01-01", periods=301, name="date")
    returns = pd.DataFrame(rng.normal(0.0005, 0.01, (300, 30)), index=dates[1:], columns=[f"N{i:02d}" for i in range(30)])
    stale = rng.normal(0.0005, 0.01, 300)
    stale[rng.permutation(300)[:180]] = 0.0
    partly = rng.normal(0.0005, 0.01, 300)
    partly[rng.permutation(300)[:120]] = 0.0
    half = rng.normal(0.0005, 0.01, 300)
    half[48 + rng.permutation(252)[:126]] = 0.0  # exactly 126 of the window's 252 returns
    returns["STALE"], returns["PARTLY"], returns["HALF"] = stale, partly, half
    returns["SHORT"] = rng.normal(0.0005, 0.01, 300)
    prices = pd.concat([pd.DataFrame(100.0, index=dates[:1], columns=returns.columns), 100 * (1 + returns).cumprod()])
    prices.loc[dates[:100], "SHORT"] = np.nan
    prices["FLAT"] = 50.0
    rf = pd.Series(0.0001 + 0.00005 * np.sin(np.arange(301) / 10.0), index=dates)
    dataset = fetch_data.Dataset(
        universe=pd.DataFrame({"ticker": list(prices.columns), "sector": "Test"}),
        prices=prices, rf_daily=rf,
        factors=pd.DataFrame(rng.normal(0.0, 0.01, (300, 3)), index=dates[1:], columns=["mkt_rf", "smb", "hml"]),
        mode="test",
    )
    result = screen.screen(dataset, window=252, top=5)
    meta = result["metadata"]
    assert meta["excluded_unscoreable"] == {"FLAT": "flat price", "STALE": "stale quotes"}
    assert meta["excluded_coverage"] == {"count": 1, "tickers": ["SHORT"]}
    assert meta["N"] == 32 and meta["universe_size"] == 35
    plane = meta["luck_horizon_ann"]["headline"]
    assert math.isfinite(plane) and 0 < plane < 5
    rows = {r["ticker"]: r for r in result["universe"]}
    assert rows["FLAT"]["rank"] is None and rows["STALE"]["rank"] is None and rows["STALE"]["dsr"] is None
    assert rows["PARTLY"]["rank"] is not None and rows["HALF"]["rank"] is not None and rows["SHORT"]["rank"] is None
    page = scene_threejs.render_page(scene_threejs.build_scene(screen._nan_to_none(result)))
    ticks = _page_data(page)["ruler_ticks"]
    assert ticks[0] >= -5 and ticks[-1] <= 5  # the ruler is not stretched by a degenerate Sharpe
    report = screen.render_report(screen._nan_to_none(result))
    assert ("## Exclusions (1y)\n\n"
            "- Coverage: 1 name with less than 90% of the window's trading days: SHORT.\n"
            "- Unscoreable: 2 names (data-quality rule, see the README): flat price 1, stale quotes 1, non-finite Sharpe 0.\n"
            "  - flat price: FLAT.\n"
            "  - stale quotes: STALE.\n") in report


# --- 13. Release-gate fixes --------------------------------------------------------------

@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js to run the page's zoom logic")
def test_zoom_limits_keep_the_scene_in_view(screen_runs: tuple[Path, Path], tmp_path: Path):
    page = scene_threejs.write_page(screen_runs[0] / "screen_result.json", tmp_path / "skyline.html").read_text()
    limits = _run_nav_logic(page, "N.distanceLimits(40, 13)")
    # framed 40, scene radius 13: max 3 x 40 = 120; fog from 40 + 13 = 53 to 53 + (120 - 53) / 0.35.
    assert limits["minDistance"] == 2 and limits["maxDistance"] == 120 and limits["fogNear"] == 53
    assert limits["fogFar"] == pytest.approx(244.42857142857142, abs=1e-9)
    assert (limits["maxDistance"] - limits["fogNear"]) / (limits["fogFar"] - limits["fogNear"]) == pytest.approx(0.35)
    diagonal = 2 * 13
    assert limits["far"] > limits["maxDistance"] + diagonal and limits["far"] >= limits["fogFar"]
    assert "const framedDistance = framedPosition().distanceTo(home);" in page
    assert "const limits = SkylineNav.distanceLimits(framedDistance, sceneRadius);" in page
    assert "controls.maxDistance = limits.maxDistance;" in page and "camera.far = limits.far;" in page
    assert re.search(r"camera\.updateProjectionMatrix\(\);\s*applyDistanceLimits\(\);", page)


def test_page_fly_to_and_panel_fixes(screen_runs: tuple[Path, Path], tmp_path: Path):
    page = scene_threejs.write_page(screen_runs[0] / "screen_result.json", tmp_path / "skyline.html").read_text()
    assert "const aim = SkylineNav.flyTarget(mesh.position, shuffle ? shuffle.to[mesh.index] : null, signed);" in page
    assert "#details { position: relative; flex: 0 1 auto; min-height: 0; max-height: calc(100vh - 32px); overflow-y: auto; }" in page
    assert '<p id="verdict-warning" class="caption">Not financial advice: historical statistics only.</p>' in page


def test_every_remote_host_is_disclosed():
    hosts = set()
    for path in sorted((REPO / "scripts").glob("*.py")):
        hosts.update(re.findall(r"https?://([A-Za-z0-9.-]+)", path.read_text()))
    assert {"raw.githubusercontent.com", "en.wikipedia.org", "www.nasdaqtrader.com", "fred.stlouisfed.org",
            "mba.tuck.dartmouth.edu", "cdn.jsdelivr.net", "www.google.com", "www.sec.gov"} <= hosts
    for doc in ("PRIVACY.md", "docs/SUBMISSION.md"):
        text = (REPO / doc).read_text()
        missing = sorted(h for h in hosts if h not in text)
        assert not missing, f"{doc} does not disclose {missing}"


def test_not_financial_advice_warnings(screen_runs: tuple[Path, Path]):
    readme = (REPO / "README.md").read_text()
    alert = readme.index("> [!WARNING]\n> **Not financial advice.**")
    assert alert < readme.index("![")
    assert readme.splitlines()[0].startswith("# ") and readme.splitlines()[2] == "> [!WARNING]"
    report = (screen_runs[0] / "report.md").read_text().splitlines()
    assert report[2] == ("> **Not financial advice.** Historical statistics for research and education only; not a "
                         "recommendation to buy, sell or hold any security. Past Sharpe ratios and alphas do not predict "
                         "future results.")
    for skill in ("skills/sharpe-screen/SKILL.md", "skills/sharpe-skyline/SKILL.md"):
        assert "**Not financial advice:**" in (REPO / skill).read_text()


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js to run the page's view logic")
def test_view_guards_and_label_fade(screen_runs: tuple[Path, Path], tmp_path: Path):
    page = scene_threejs.write_page(screen_runs[0] / "screen_result.json", tmp_path / "skyline.html").read_text()
    data = _page_data(page)
    # Orbit stays above the horizon: max polar = pi/2 - 0.08 = 1.490796; camera never below y = 0.5.
    assert data["polar"] == {"min": 0.05, "max": 1.490796, "camera_min_y": 0.5}
    assert "controls.minPolarAngle = DATA.polar.min;" in page and "controls.maxPolarAngle = DATA.polar.max;" in page
    assert "const above = SkylineNav.cameraAboveFloor(camera.position, DATA.polar.camera_min_y);" in page
    assert _run_nav_logic(page, "[N.cameraAboveFloor({x: 1, y: -3.4, z: 2}, 0.5), N.cameraAboveFloor({x: 1, y: 7, z: 2}, 0.5)]") == [
        {"x": 1, "y": 0.5, "z": 2}, {"x": 1, "y": 7, "z": 2}]
    # Label fade: absolute floors (14, 24) for a small scene; 0.25x / 0.45x the reset distance for a big one.
    lod = json.dumps(data["label_lod"])
    assert _run_nav_logic(page, f"[N.labelFade(40, {lod}), N.labelFade(400, {lod})]") == [
        {"fullDistance": 14, "maxDistance": 24}, {"fullDistance": 100, "maxDistance": 180}]
    # Fly-to aims at the resting place during a layout transition, at the current place otherwise;
    # the target height is the middle of the tower (negative towers hang below 0).
    assert _run_nav_logic(page, "[N.flyTarget({x: 1, y: 0, z: 2}, {x: 5, y: 0, z: 6}, 3), "
                                "N.flyTarget({x: 1, y: -2, z: 2}, null, -2)]") == [
        {"x": 5, "y": 1.5, "z": 6}, {"x": 1, "y": -1, "z": 2}]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js to run the page's declutter logic")
def test_label_declutter_priorities(screen_runs: tuple[Path, Path], tmp_path: Path):
    page = scene_threejs.write_page(screen_runs[0] / "screen_result.json", tmp_path / "skyline.html").read_text()
    # Boxes: width = len(ticker) * 7 + 10, height 15 (fixture value), centred on (x, y). Pairs (SEL, AAA) and
    # (BBB, CCC) overlap; DDD and EEE overlap nothing. SEL (selected) beats AAA (rank 1); BBB
    # (rank 2) beats CCC (nearer, but unranked); DDD and EEE are placed nearest-first.
    candidates = ('['
                  '{ticker: "EEE", x: 700, y: 100, selected: false, top: false, rank: null, distance: 9}, '
                  '{ticker: "CCC", x: 305, y: 102, selected: false, top: false, rank: null, distance: 5}, '
                  '{ticker: "AAA", x: 105, y: 100, selected: false, top: true, rank: 1, distance: 50}, '
                  '{ticker: "DDD", x: 500, y: 100, selected: false, top: false, rank: null, distance: 3}, '
                  '{ticker: "BBB", x: 300, y: 100, selected: false, top: true, rank: 2, distance: 60}, '
                  '{ticker: "SEL", x: 100, y: 100, selected: true, top: false, rank: 300, distance: 80}]')
    opts = "{charWidth: 7, padding: 10, height: 15, cell: 64, budget: %d}"
    assert _run_nav_logic(page, f"N.declutter({candidates}, {opts % 100})") == ["SEL", "BBB", "DDD", "EEE"]
    assert _run_nav_logic(page, f"N.declutter({candidates}, {opts % 3})") == ["SEL", "BBB", "DDD"]
    # A measured width overrides the estimate: WIDE (rank 3, 300 px of text) reaches EEE at x 700.
    wide = candidates[:-1] + ', {ticker: "WIDE", x: 560, y: 100, selected: false, top: true, rank: 3, distance: 70, width: 300}]'
    assert _run_nav_logic(page, f"N.declutter({wide}, {opts % 100})") == ["SEL", "BBB", "WIDE"]
