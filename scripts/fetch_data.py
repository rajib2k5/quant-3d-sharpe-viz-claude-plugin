"""Load the screening dataset: offline sample by default, live public sources behind --live.

Live sources (contacted only in live mode):
- Universe (unless a CSV is given): S&P 500 constituents from the open datasets project,
  https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv,
  falling back to the table at https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
- Prices: Yahoo Finance daily adjusted closes through the unofficial yfinance library
  (query1.finance.yahoo.com / query2.finance.yahoo.com)
- Risk-free: FRED 3-month Treasury bill, https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO
- Factors: Kenneth R. French Data Library, Fama-French 3 daily factors (Fama & French 1993),
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip

Every source is normalized to the on-disk format of data/sample/:
- universe.csv     ticker, sector
- prices.csv       date, <ticker>... (closes adjusted for splits and dividends)
- risk_free.csv    date, rf_daily (decimal per trading day)
- factors.csv      date, mkt_rf, smb, hml (decimal per trading day)
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = REPO_ROOT / "data" / "sample"

SP500_DATASETS_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_PRIMARY_LABEL = "raw.githubusercontent.com datasets/s-and-p-500-companies"
SP500_FALLBACK_LABEL = "en.wikipedia.org List of S&P 500 companies"
MIN_SP500_ROWS = 400
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
FF3_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
TIMEOUT_S = 30
THROTTLE_S = 0.5
MAX_THROTTLE_S = 8.0
RETRY_BACKOFF_S = (4.0, 15.0)
CLEAN_FETCHES_TO_DECAY = 20
MAX_CONSECUTIVE_FAILURES = 5
MIN_FAILURES_FOR_RATE_ABORT = 20
MAX_FAILURE_RATE = 0.5
BATCH_SIZE = 50
PROGRESS_EVERY = 10
USER_AGENT = "sharpe-observatory/0.1.0 (Claude Code plugin)"

TICKER_COLUMNS = ("ticker", "Ticker", "Symbol", "symbol")
SECTOR_COLUMNS = ("sector", "Sector", "GICS Sector", "gics_sector")
WEBSITE_COLUMNS = ("website", "Website")
FACTOR_COLUMNS = ["mkt_rf", "smb", "hml"]


@dataclass(frozen=True)
class Dataset:
    universe: pd.DataFrame
    prices: pd.DataFrame
    rf_daily: pd.Series
    factors: pd.DataFrame
    mode: str


def _universe_frame(raw: pd.DataFrame, source: str, require_sector: bool) -> pd.DataFrame:
    """Pick the ticker, sector and optional website columns, strip whitespace, drop repeated tickers (dots kept)."""
    columns = {str(c).strip(): c for c in raw.columns}
    ticker_col = next((columns[c] for c in TICKER_COLUMNS if c in columns), None)
    sector_col = next((columns[c] for c in SECTOR_COLUMNS if c in columns), None)
    if ticker_col is None or (require_sector and sector_col is None):
        raise ValueError(
            f"{source}: unexpected schema {list(columns)}; need a ticker column {TICKER_COLUMNS}"
            + (f" and a sector column {SECTOR_COLUMNS}" if require_sector else "")
        )
    universe = pd.DataFrame(
        {
            "ticker": raw[ticker_col].astype(str).str.strip().str.upper(),
            "sector": raw[sector_col].astype(str).str.strip() if sector_col is not None else "Unclassified",
        }
    )
    website_col = next((columns[c] for c in WEBSITE_COLUMNS if c in columns), None)
    if website_col is not None:
        universe["website"] = raw[website_col].fillna("").astype(str).str.strip()
    return universe[universe["ticker"] != ""].drop_duplicates("ticker").reset_index(drop=True)


def load_universe(path: Path) -> pd.DataFrame:
    """Read a ticker/sector CSV. Accepts common S&P 500 constituent layouts (Symbol, GICS Sector)."""
    return _universe_frame(pd.read_csv(path), str(path), require_sector=False)


def normalize_sp500(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    """Ticker/sector frame sorted by ticker; ValueError on a changed schema or under MIN_SP500_ROWS rows."""
    universe = _universe_frame(raw, source, require_sector=True).sort_values("ticker").reset_index(drop=True)
    if len(universe) < MIN_SP500_ROWS:
        raise ValueError(f"{source}: only {len(universe)} constituents, fewer than {MIN_SP500_ROWS}")
    return universe


def parse_sp500_datasets_csv(text: str) -> pd.DataFrame:
    """Raw constituents table from the datasets/s-and-p-500-companies CSV."""
    return pd.read_csv(io.StringIO(text))


def parse_sp500_wikipedia_html(text: str) -> pd.DataFrame:
    """The constituents table from the Wikipedia page: the first table with Symbol and GICS Sector."""
    for table in pd.read_html(io.StringIO(text)):
        names = {str(c).strip() for c in table.columns}
        if {"Symbol", "GICS Sector"} <= names:
            return table
    raise ValueError("Wikipedia page has no table with Symbol and GICS Sector columns")


def _get_text(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=TIMEOUT_S)
    response.raise_for_status()
    return response.text


def default_sp500_sources() -> list[tuple[str, Callable[[], pd.DataFrame]]]:
    session = _session()
    return [
        (SP500_PRIMARY_LABEL, lambda: parse_sp500_datasets_csv(_get_text(session, SP500_DATASETS_URL))),
        (SP500_FALLBACK_LABEL, lambda: parse_sp500_wikipedia_html(_get_text(session, SP500_WIKIPEDIA_URL))),
    ]


def fetch_sp500_universe(sources: list[tuple[str, Callable[[], pd.DataFrame]]] | None = None) -> pd.DataFrame:
    """Current S&P 500 constituents as (ticker, sector), trying each source in order.

    A source fails if it cannot be fetched or parsed, its schema changed, or it yields fewer
    than MIN_SP500_ROWS names; the next source is then tried. RuntimeError if all fail.
    """
    errors: list[str] = []
    for label, load in sources if sources is not None else default_sp500_sources():
        try:
            return normalize_sp500(load(), label)
        except (requests.RequestException, ValueError, ImportError) as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            print(f"[universe] {label} failed ({exc})", file=sys.stderr)
    raise RuntimeError("could not fetch the S&P 500 constituents: " + "; ".join(errors))


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
US_NASDAQ_LABEL = "nasdaqtrader.com nasdaqlisted.txt"
US_OTHER_LABEL = "nasdaqtrader.com otherlisted.txt"
OTHER_EXCHANGES = {"N": "NYSE", "A": "NYSE American", "P": "NYSE Arca", "Z": "Cboe BZX"}
MIN_US_ROWS = 3000
NON_COMMON_NAME = re.compile(r"\b(?:warrants?|units?|rights?|notes?|preferred|trust preferred)\b", re.IGNORECASE)
# Plain "Depositary Shares" listings are mostly preferred-share wrappers; American depositary
# shares (ADRs) are common-stock exposure and stay unless NON_COMMON_NAME also matches.
DEPOSITARY_NAME = re.compile(r"\bdepositary shares\b", re.IGNORECASE)
ADR_NAME = re.compile(r"\bamerican depositary\b", re.IGNORECASE)


def _read_symbol_directory(text: str, source: str, required: list[str]) -> pd.DataFrame:
    """Pipe-delimited Nasdaq Trader file, without its trailing "File Creation Time" line."""
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("File Creation Time")]
    frame = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str, keep_default_na=False)
    frame.columns = [c.strip() for c in frame.columns]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{source}: unexpected schema {list(frame.columns)}; missing {missing}")
    return frame


def _stocks_only(frame: pd.DataFrame, symbol_col: str) -> pd.DataFrame:
    """Drop test issues, ETFs, NextShares, symbols with "$" (preferred shares and units), and
    securities whose name marks them as non-common: NON_COMMON_NAME (warrants, units, rights,
    notes, preferred, trust preferred; whole words, any case), or "depositary shares" that are
    not American depositary shares."""
    keep = (frame["Test Issue"].str.strip() == "N") & (frame["ETF"].str.strip() == "N")
    if "NextShares" in frame.columns:
        keep &= frame["NextShares"].str.strip() != "Y"
    symbols = frame[symbol_col].str.strip()
    keep &= (symbols != "") & ~symbols.str.contains("$", regex=False)
    if "Security Name" in frame.columns:
        names = frame["Security Name"]
        plain_depositary = names.str.contains(DEPOSITARY_NAME) & ~names.str.contains(ADR_NAME)
        keep &= ~names.str.contains(NON_COMMON_NAME) & ~plain_depositary
    return frame[keep]


def parse_nasdaq_listed(text: str) -> pd.DataFrame:
    """nasdaqlisted.txt -> ticker, sector (= listing exchange: NASDAQ)."""
    frame = _stocks_only(
        _read_symbol_directory(text, US_NASDAQ_LABEL, ["Symbol", "Test Issue", "ETF", "NextShares"]), "Symbol")
    return pd.DataFrame({"ticker": frame["Symbol"].str.strip(), "sector": "NASDAQ"})


def parse_other_listed(text: str) -> pd.DataFrame:
    """otherlisted.txt -> ticker (ACT Symbol), sector (= listing exchange from its code)."""
    frame = _stocks_only(
        _read_symbol_directory(text, US_OTHER_LABEL, ["ACT Symbol", "Exchange", "Test Issue", "ETF"]), "ACT Symbol")
    codes = frame["Exchange"].str.strip()
    return pd.DataFrame({
        "ticker": frame["ACT Symbol"].str.strip(),
        "sector": [OTHER_EXCHANGES.get(code, f"Other ({code})") for code in codes],
    })


def default_us_market_sources() -> list[tuple[str, Callable[[], pd.DataFrame]]]:
    session = _session()
    return [
        (US_NASDAQ_LABEL, lambda: parse_nasdaq_listed(_get_text(session, NASDAQ_LISTED_URL))),
        (US_OTHER_LABEL, lambda: parse_other_listed(_get_text(session, OTHER_LISTED_URL))),
    ]


def fetch_us_market_universe(sources: list[tuple[str, Callable[[], pd.DataFrame]]] | None = None) -> pd.DataFrame:
    """Every US-listed common stock as (ticker, sector), where ``sector`` holds the listing
    exchange: no sector data exists at this scale. Both sources are needed (no fallback);
    RuntimeError names each one that failed, or reports fewer than MIN_US_ROWS stocks."""
    parts: list[pd.DataFrame] = []
    errors: list[str] = []
    for label, load in sources if sources is not None else default_us_market_sources():
        try:
            parts.append(load())
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError("could not fetch the US market symbol directory: " + "; ".join(errors))
    universe = _universe_frame(pd.concat(parts, ignore_index=True), "US market", require_sector=True)
    universe = universe.sort_values("ticker").reset_index(drop=True)
    if len(universe) < MIN_US_ROWS:
        raise RuntimeError(f"US market symbol directory: only {len(universe)} stocks after filters, fewer than {MIN_US_ROWS}")
    return universe


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Closes <= 0 are invalid market data (a zero close makes returns of -100% and then a
    division by zero), so they become missing. Reports the count and the tickers on stderr."""
    bad = prices <= 0
    count = int(bad.to_numpy().sum())
    if count:
        names = [str(t) for t in prices.columns[bad.any()]]
        shown = ", ".join(names[:10]) + (f" and {len(names) - 10} more" if len(names) > 10 else "")
        print(f"[clean] {count} non-positive close{'s' if count != 1 else ''} set to missing ({shown})", file=sys.stderr)
    return prices.mask(bad)


def load_directory(data_dir: Path, universe_path: Path | None = None) -> Dataset:
    """Load a dataset previously written in the sample format (no network)."""
    universe = load_universe(universe_path or data_dir / "universe.csv")
    prices = pd.read_csv(data_dir / "prices.csv", index_col="date", parse_dates=["date"]).sort_index()
    rf = pd.read_csv(data_dir / "risk_free.csv", index_col="date", parse_dates=["date"])["rf_daily"].sort_index()
    factors = pd.read_csv(data_dir / "factors.csv", index_col="date", parse_dates=["date"])[FACTOR_COLUMNS].sort_index()
    present = [t for t in universe["ticker"] if t in prices.columns]
    return Dataset(
        universe=universe[universe["ticker"].isin(present)].reset_index(drop=True),
        prices=clean_prices(prices[present]),
        rf_daily=rf,
        factors=factors,
        mode="sample" if data_dir.resolve() == SAMPLE_DIR.resolve() else "cached",
    )


def _session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def yahoo_symbol(ticker: str) -> str:
    """Yahoo spells share classes with a hyphen (BRK.B -> BRK-B)."""
    return ticker.upper().replace(".", "-")


def fetch_yahoo_close(ticker: str) -> pd.Series:
    """Daily split- and dividend-adjusted closes for one ticker from Yahoo Finance via yfinance."""
    import yfinance  # only live mode needs it; the offline sample must run without it

    # Surface errors as exceptions instead of log lines (replaces the deprecated raise_errors=True).
    yfinance.config.debug.hide_exceptions = False
    history = yfinance.Ticker(yahoo_symbol(ticker)).history(
        period="max", interval="1d", auto_adjust=True, actions=False, timeout=TIMEOUT_S
    )
    if history.empty:
        raise ValueError("Yahoo Finance returned no rows")
    close = history["Close"].rename(ticker)
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    return close


_TRANSIENT_STATUS = re.compile(r"\b(429|50[0-4])\b")
_TRANSIENT_TEXT = ("timed out", "timeout", "too many requests", "rate limit", "connection reset",
                   "connection aborted", "connection refused", "temporarily unavailable")
_TRANSIENT_TYPE_WORDS = ("timeout", "connection", "ratelimit")


def is_transient(exc: BaseException) -> bool:
    """Heuristic: is this failure worth retrying?

    Transient: TimeoutError/ConnectionError and their subclasses; exception types whose name
    contains Timeout, Connection or RateLimit (requests, curl_cffi and yfinance use these);
    messages naming HTTP 429 or 500-504, a timeout, a rate limit or a dropped connection.
    Everything else (HTTP 404, "possibly delisted", empty history, parse errors) is permanent.
    """
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__.lower()
    if any(word in name for word in _TRANSIENT_TYPE_WORDS):
        return True
    text = str(exc).lower()
    return bool(_TRANSIENT_STATUS.search(text)) or any(marker in text for marker in _TRANSIENT_TEXT)


def _fetch_with_retries(
    ticker: str, fetch_close: Callable[[str], pd.Series], sleep: Callable[[float], None]
) -> tuple[pd.Series | None, Exception | None, bool]:
    """(closes or None, final error or None, whether a retry was needed).

    A transient error is retried after each RETRY_BACKOFF_S pause in turn; a permanent one
    returns at once.
    """
    retried = False
    attempt = 0
    while True:
        try:
            return fetch_close(ticker), None, retried
        except ModuleNotFoundError as exc:
            raise RuntimeError(f"{exc.name} is not installed; run pip install -r requirements.txt") from exc
        except Exception as exc:  # yfinance surfaces HTTP, parsing and rate-limit problems as many types
            if attempt == len(RETRY_BACKOFF_S) or not is_transient(exc):
                return None, exc, retried
            pause = RETRY_BACKOFF_S[attempt]
            print(f"[fetch] {ticker}: {type(exc).__name__}: {exc}; retrying in {pause:g}s", file=sys.stderr)
            sleep(pause)
            retried = True
            attempt += 1


class AdaptiveThrottle:
    """Pause between requests: doubles (up to MAX_THROTTLE_S) after trouble and halves back
    toward the floor, never below it, after CLEAN_FETCHES_TO_DECAY clean requests in a row."""

    def __init__(self, floor_s: float) -> None:
        self.floor = floor_s
        self.current = floor_s
        self._clean = 0

    def _set(self, new: float, reason: str) -> None:
        if new != self.current:
            print(f"[fetch] throttle {self.current:g}s -> {new:g}s ({reason})", file=sys.stderr)
            self.current = new

    def trouble(self, reason: str) -> None:
        self._clean = 0
        self._set(min(self.current * 2, MAX_THROTTLE_S), reason)

    def clean(self) -> None:
        self._clean += 1
        if self._clean >= CLEAN_FETCHES_TO_DECAY and self.current > self.floor:
            self._clean = 0
            self._set(max(self.current / 2, self.floor), f"{CLEAN_FETCHES_TO_DECAY} clean fetches")


def fetch_prices(
    tickers: list[str],
    fetch_close: Callable[[str], pd.Series] = fetch_yahoo_close,
    throttle_s: float = THROTTLE_S,
    sleep: Callable[[float], None] = time.sleep,
    on_fetched: Callable[[str, pd.Series], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Closes for each ticker, one polite request at a time, with retries and an adaptive pause.

    Transient failures are retried (see is_transient and RETRY_BACKOFF_S). The pause between
    tickers is an AdaptiveThrottle starting at ``throttle_s``: a ticker that fails or needed a
    retry counts as trouble. A ticker that succeeds after a retry is a success. A ticker that
    still fails is skipped and recorded, and RuntimeError is raised instead of requesting the
    rest of the universe when the source looks blocked: after MAX_CONSECUTIVE_FAILURES failed
    tickers in a row, or once at least MIN_FAILURES_FOR_RATE_ABORT tickers have failed and
    they exceed MAX_FAILURE_RATE of those attempted. A missing package is reported as such,
    not as a blocked source. ``on_fetched`` sees every series as soon as it arrives.
    """
    closes: list[pd.Series] = []
    failures: dict[str, str] = {}
    streak = 0
    throttle = AdaptiveThrottle(throttle_s)
    for i, ticker in enumerate(tickers):
        if i:
            sleep(throttle.current)
        close, exc, retried = _fetch_with_retries(ticker, fetch_close, sleep)
        if close is not None:
            closes.append(close)
            streak = 0
            if on_fetched is not None:
                on_fetched(ticker, close)
        else:
            failures[ticker] = f"{type(exc).__name__}: {exc}"
            streak += 1
            print(f"[fetch] {ticker}: skipped ({failures[ticker]})", file=sys.stderr)
            if streak >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"stopped after {streak} consecutive ticker failures (last: {ticker}); "
                    "Yahoo Finance may be blocking or rate-limiting requests"
                ) from exc
            attempted = i + 1
            if len(failures) >= MIN_FAILURES_FOR_RATE_ABORT and len(failures) / attempted > MAX_FAILURE_RATE:
                raise RuntimeError(
                    f"stopped after {len(failures)} failures in {attempted} tickers ({len(failures) / attempted:.0%}); "
                    "Yahoo Finance may be blocking or rate-limiting requests"
                ) from exc
        if close is None or retried:
            throttle.trouble(f"{ticker} {'failed' if close is None else 'needed a retry'}")
        else:
            throttle.clean()
    prices = pd.concat(closes, axis=1).sort_index() if closes else pd.DataFrame()
    prices.index.name = "date"
    return prices, failures


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def yahoo_download(chunk: list[str]) -> pd.DataFrame:
    """One yf.download call for a chunk: daily adjusted history, space-joined Yahoo symbols.

    yfinance still requests each symbol separately but runs them in parallel threads.
    """
    import yfinance  # only live mode needs it; the offline sample must run without it

    return yfinance.download(" ".join(yahoo_symbol(t) for t in chunk), period="max", interval="1d",
                             auto_adjust=True, actions=False, group_by="column", threads=True,
                             progress=False, timeout=TIMEOUT_S)


def split_batch(frame: pd.DataFrame | None, chunk: list[str]) -> tuple[dict[str, pd.Series], list[str]]:
    """Closes per ticker from a yf.download frame, and the tickers that came back empty.

    yfinance returns (field, symbol) MultiIndex columns; a lone symbol may come back with flat
    columns. A symbol that is absent or all-NaN goes to the second, per-ticker pass.
    """
    found: dict[str, pd.Series] = {}
    if frame is None or frame.empty:
        return found, list(chunk)
    if isinstance(frame.columns, pd.MultiIndex):
        closes = frame["Close"] if "Close" in frame.columns.get_level_values(0) else pd.DataFrame(index=frame.index)
    elif "Close" in frame.columns and len(chunk) == 1:
        closes = frame[["Close"]].set_axis([yahoo_symbol(chunk[0])], axis=1)
    else:
        closes = pd.DataFrame(index=frame.index)
    for ticker in chunk:
        symbol = yahoo_symbol(ticker)
        series = closes[symbol].dropna() if symbol in closes else pd.Series(dtype=float)
        if series.empty:
            continue
        series = series.rename(ticker)
        series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
        found[ticker] = series
    return found, [t for t in chunk if t not in found]


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / f"{ticker.replace('/', '_')}.csv"


def read_cache(cache_dir: Path, tickers: list[str]) -> dict[str, pd.Series]:
    cached = {}
    for ticker in tickers:
        path = _cache_path(cache_dir, ticker)
        if path.is_file():
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
            cached[ticker] = frame.iloc[:, 0].rename(ticker)
    return cached


def write_cache(cache_dir: Path, ticker: str, close: pd.Series) -> None:
    close.rename("close").to_csv(_cache_path(cache_dir, ticker), index_label="date")


def fetch_prices_batched(
    tickers: list[str],
    download: Callable[[list[str]], pd.DataFrame] = yahoo_download,
    fetch_close: Callable[[str], pd.Series] = fetch_yahoo_close,
    throttle_s: float = THROTTLE_S,
    sleep: Callable[[float], None] = time.sleep,
    cache_dir: Path | None = None,
    batch_size: int = BATCH_SIZE,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Closes for many tickers: chunked multi-ticker downloads, then a per-ticker second pass.

    Tickers already in ``cache_dir`` are read from it; every new series is written there as
    soon as it arrives, so an interrupted run resumes where it stopped. The rest are fetched
    ``batch_size`` at a time with ``download``, with an AdaptiveThrottle pause between chunks
    (a chunk that raises counts as trouble). Tickers a chunk returns empty or all-NaN, and all
    tickers of a chunk that raised, get a second pass through fetch_prices with its retries,
    backoff and both abort rules. Progress is logged every PROGRESS_EVERY chunks.
    """
    closes: dict[str, pd.Series] = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        closes.update(read_cache(cache_dir, tickers))
        if closes:
            print(f"[fetch] {len(closes)} of {len(tickers)} tickers read from {cache_dir}", file=sys.stderr)

    def keep(ticker: str, close: pd.Series) -> None:
        closes[ticker] = close
        if cache_dir is not None:
            write_cache(cache_dir, ticker, close)

    remaining = [t for t in tickers if t not in closes]
    chunks = chunked(remaining, batch_size)
    second_pass: list[str] = []
    throttle = AdaptiveThrottle(throttle_s)
    started = clock()
    for i, chunk in enumerate(chunks):
        if i:
            sleep(throttle.current)
        try:
            found, missing = split_batch(download(chunk), chunk)
        except ModuleNotFoundError as exc:
            raise RuntimeError(f"{exc.name} is not installed; run pip install -r requirements.txt") from exc
        except Exception as exc:  # the whole chunk falls back to the per-ticker pass
            print(f"[fetch] chunk {i + 1}: {type(exc).__name__}: {exc}", file=sys.stderr)
            found, missing = {}, list(chunk)
            throttle.trouble(f"chunk {i + 1} failed")
        else:
            throttle.clean()
        for ticker, close in found.items():
            keep(ticker, close)
        second_pass.extend(missing)
        if (i + 1) % PROGRESS_EVERY == 0 or i + 1 == len(chunks):
            print(f"[fetch] chunk {i + 1}/{len(chunks)}: {len(closes)}/{len(tickers)} tickers, "
                  f"{len(second_pass)} queued for retry, {clock() - started:.0f}s elapsed", file=sys.stderr)

    failures: dict[str, str] = {}
    if second_pass:
        print(f"[fetch] second pass: {len(second_pass)} tickers one at a time", file=sys.stderr)
        _, failures = fetch_prices(second_pass, fetch_close, throttle_s, sleep, on_fetched=keep)
    ordered = [closes[t] for t in tickers if t in closes]
    prices = clean_prices(pd.concat(ordered, axis=1).sort_index()) if ordered else pd.DataFrame()
    prices.index.name = "date"
    return prices, failures


def parse_fred_dgs3mo(text: str) -> pd.Series:
    """FRED DGS3MO CSV (annualized percent) -> daily decimal rate: rate / 100 / 252."""
    frame = pd.read_csv(io.StringIO(text))
    date_col = frame.columns[0]
    rate = pd.to_numeric(frame["DGS3MO"], errors="coerce")
    series = pd.Series(rate.to_numpy() / 100.0 / 252.0, index=pd.to_datetime(frame[date_col]), name="rf_daily")
    series.index.name = "date"
    return series.dropna()


def fetch_risk_free(session: requests.Session) -> pd.Series:
    response = session.get(FRED_URL, timeout=TIMEOUT_S)
    response.raise_for_status()
    return parse_fred_dgs3mo(response.text)


_FF_ROW = re.compile(r"^\s*(\d{8})\s*,(.*)$")


def parse_ff3_daily_csv(text: str) -> pd.DataFrame:
    """Ken French FF3 daily CSV (percent) -> decimal mkt_rf, smb, hml.

    The file wraps the data in free-text header and copyright lines; data rows are the
    lines starting with an eight-digit YYYYMMDD date.
    """
    dates: list[str] = []
    rows: list[list[float]] = []
    for line in text.splitlines():
        match = _FF_ROW.match(line)
        if match:
            values = [float(v) for v in match.group(2).split(",")]
            dates.append(match.group(1))
            rows.append(values[:3])
    frame = pd.DataFrame(rows, index=pd.to_datetime(dates, format="%Y%m%d"), columns=FACTOR_COLUMNS) / 100.0
    frame.index.name = "date"
    return frame


def fetch_factors(session: requests.Session) -> pd.DataFrame:
    response = session.get(FF3_URL, timeout=TIMEOUT_S)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        text = archive.read(name).decode("latin-1")
    return parse_ff3_daily_csv(text)


def fetch_live(universe: pd.DataFrame, throttle_s: float = THROTTLE_S, cache_dir: Path | None = None) -> Dataset:
    """Download prices, risk-free and factors for the universe (network: Yahoo Finance, FRED, Dartmouth)."""
    prices, failures = fetch_prices_batched(list(universe["ticker"]), throttle_s=throttle_s, cache_dir=cache_dir)
    if prices.empty:
        raise RuntimeError(f"no prices downloaded; {len(failures)} tickers failed")
    session = _session()
    rf = fetch_risk_free(session)
    factors = fetch_factors(session)
    return Dataset(
        universe=universe[universe["ticker"].isin(prices.columns)].reset_index(drop=True),
        prices=prices,
        rf_daily=rf,
        factors=factors,
        mode="live",
    )


def write_directory(dataset: Dataset, out_dir: Path) -> None:
    """Persist a dataset in the sample format so later runs can use it offline (--data-dir)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.universe.to_csv(out_dir / "universe.csv", index=False)
    dataset.prices.to_csv(out_dir / "prices.csv", index_label="date")
    dataset.rf_daily.rename("rf_daily").to_csv(out_dir / "risk_free.csv", index_label="date")
    dataset.factors.to_csv(out_dir / "factors.csv", index_label="date")


INDEXES = ("sp500", "us-all")


def live_universe(universe_path: Path | None, index: str = "sp500") -> pd.DataFrame:
    """The user's CSV when given, else the chosen index fetched from the network: ``sp500``
    (S&P 500 constituents, grouped by GICS sector) or ``us-all`` (every US-listed common
    stock, grouped by listing exchange)."""
    if universe_path is not None:
        return load_universe(universe_path)
    if index == "us-all":
        print(f"fetching the US market symbol directory ({US_NASDAQ_LABEL} + {US_OTHER_LABEL})", file=sys.stderr)
        return fetch_us_market_universe()
    print(f"fetching S&P 500 constituents ({SP500_PRIMARY_LABEL}; fallback {SP500_FALLBACK_LABEL})", file=sys.stderr)
    return fetch_sp500_universe()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true",
                        help="download from Yahoo Finance (yfinance), fred.stlouisfed.org and mba.tuck.dartmouth.edu")
    parser.add_argument("--universe", type=Path, help="ticker/sector CSV overriding the fetched index")
    parser.add_argument("--index", choices=INDEXES, default="sp500",
                        help="universe fetched with --live and no --universe (us-all: every US-listed common stock)")
    parser.add_argument("--cache-dir", type=Path, help="resumable per-ticker price cache for --live")
    parser.add_argument("--out-dir", type=Path, default=Path("data/live"), help="where --live writes its CSVs")
    parser.add_argument("--throttle", type=float, default=THROTTLE_S,
                        help=f"minimum seconds between Yahoo requests (default {THROTTLE_S}); backoff raises it automatically")
    args = parser.parse_args(argv)
    if args.throttle < 0:
        parser.error("--throttle must be >= 0")

    if args.live:
        try:
            dataset = fetch_live(live_universe(args.universe, args.index), throttle_s=args.throttle,
                                 cache_dir=args.cache_dir)
        except RuntimeError as exc:
            parser.exit(1, f"error: live fetch failed: {exc}\n")
        write_directory(dataset, args.out_dir)
        destination = str(args.out_dir)
    else:
        dataset = load_directory(SAMPLE_DIR)
        destination = str(SAMPLE_DIR)
    print(
        f"{dataset.mode}: {len(dataset.universe)} tickers, {len(dataset.prices)} price rows "
        f"({dataset.prices.index.min().date()} to {dataset.prices.index.max().date()}) in {destination}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
