"""Generate the deterministic synthetic sample universe in data/sample/.

Excess returns follow a three-factor model (Fama & French 1993),

    r_excess = alpha + beta * mkt_rf + s * smb + h * hml + idio,

with three kinds of names:
- 52 ordinary names: alpha = 0, market-like betas, 19-35% annual idiosyncratic volatility.
- 5 genuine-alpha names with 19% idiosyncratic volatility: TEC03 earns 50% a year, IND04 45%,
  the other three 35%. The spread gives the 3y demo a range of outcomes.
- 3 lucky names: alpha = 0, low beta and very low volatility, and each one's idiosyncratic
  path is the best of LUCKY_DRAWS seeded draws over the last three years. Their Sharpe looks
  good purely through selection, which illustrates the multiple-testing problem; which names
  rise above the luck plane or pass the DSR test varies by seed.

Tickers are synthetic (sector prefix + number) and do not refer to real securities.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_data import SAMPLE_DIR

SEED = 20260101
N_PRICE_ROWS = 1301
END_DATE = "2025-12-31"
RF_ANNUAL = 0.04
TRADING_DAYS = 252
LUCK_WINDOW = 756
LUCKY_DRAWS = 40

SECTORS: list[tuple[str, str, int]] = [
    ("Information Technology", "TEC", 8),
    ("Health Care", "HLT", 8),
    ("Financials", "FIN", 8),
    ("Consumer Discretionary", "CDS", 8),
    ("Industrials", "IND", 7),
    ("Energy", "ENE", 7),
    ("Utilities", "UTL", 7),
    ("Consumer Staples", "STP", 7),
]
GENUINE_ALPHA_ANNUAL = {"TEC03": 0.50, "HLT05": 0.35, "FIN02": 0.35, "IND04": 0.45, "STP06": 0.35}
LUCKY = ["UTL02", "STP03", "CDS07"]
GENUINE_IDIO_VOL = 0.012
LUCKY_BETA = 0.3
LUCKY_IDIO_VOL = 0.005


def build(seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=END_DATE, periods=N_PRICE_ROWS, name="date")
    n_ret = N_PRICE_ROWS - 1

    factors = pd.DataFrame(
        {
            "mkt_rf": rng.normal(0.0003, 0.010, n_ret),
            "smb": rng.normal(0.0001, 0.005, n_ret),
            "hml": rng.normal(0.0001, 0.005, n_ret),
        },
        index=dates[1:],
    )
    rf_daily = RF_ANNUAL / TRADING_DAYS

    universe_rows: list[dict[str, str]] = []
    excess: dict[str, np.ndarray] = {}
    for sector, prefix, count in SECTORS:
        for i in range(1, count + 1):
            ticker = f"{prefix}{i:02d}"
            universe_rows.append({"ticker": ticker, "sector": sector})
            beta = rng.uniform(0.6, 1.4)
            s_load, h_load = rng.normal(0.0, 0.3, 2)
            idio_vol = rng.uniform(0.012, 0.022)
            alpha = 0.0
            idio = rng.normal(0.0, 1.0, n_ret)
            if ticker in GENUINE_ALPHA_ANNUAL:
                alpha = GENUINE_ALPHA_ANNUAL[ticker] / TRADING_DAYS
                idio_vol = GENUINE_IDIO_VOL
            elif ticker in LUCKY:
                beta, idio_vol = LUCKY_BETA, LUCKY_IDIO_VOL
                draws = rng.normal(0.0, 1.0, (LUCKY_DRAWS, n_ret))
                idio = draws[np.argmax(draws[:, -LUCK_WINDOW:].mean(axis=1))]
            excess[ticker] = (
                alpha
                + beta * factors["mkt_rf"].to_numpy()
                + s_load * factors["smb"].to_numpy()
                + h_load * factors["hml"].to_numpy()
                + idio_vol * idio
            )

    growth = pd.DataFrame(excess, index=dates[1:]) + rf_daily
    prices = 100.0 * (1.0 + growth).cumprod()
    prices = pd.concat([pd.DataFrame(100.0, index=dates[:1], columns=prices.columns), prices])
    prices.index.name = "date"
    risk_free = pd.DataFrame({"rf_daily": rf_daily}, index=dates)
    return {
        "universe": pd.DataFrame(universe_rows),
        "prices": prices.round(3),
        "risk_free": risk_free.round(10),
        "factors": factors.round(6),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the synthetic sample dataset.")
    parser.add_argument("--out-dir", type=Path, default=SAMPLE_DIR)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = build()
    frames["universe"].to_csv(args.out_dir / "universe.csv", index=False)
    for name in ("prices", "risk_free", "factors"):
        frames[name].to_csv(args.out_dir / f"{name}.csv", index_label="date")
    print(f"wrote {len(frames['universe'])} tickers x {len(frames['prices'])} rows to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
