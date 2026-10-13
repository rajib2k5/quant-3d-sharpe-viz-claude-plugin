"""Performance statistics for the Sharpe screen, with a multiple-testing luck correction.

References
----------
- Sharpe, W. F. (1994). "The Sharpe Ratio." Journal of Portfolio Management 21(1).
- Bailey, D. H. & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier."
  Journal of Risk 15(2). Source of the Probabilistic Sharpe Ratio (PSR).
- Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. (2014). "Pseudo-Mathematics
  and Financial Charlatanism." Notices of the AMS 61(5); López de Prado, M. (2020).
  Machine Learning for Asset Managers, ch. 8 and App. B (False Strategy Theorem with proof).
  Source of the expected maximum Sharpe ratio across N trials.
- Bailey, D. H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio." Journal of
  Portfolio Management 40(5); López de Prado, M. (2018). Advances in Financial Machine
  Learning, sections 14.7.2-14.7.3. Source of the Deflated Sharpe Ratio (DSR).
- Jensen, M. C. (1968). "The Performance of Mutual Funds in the Period 1945-1964."
  Journal of Finance 23(2); Fama, E. F. & French, K. R. (1993). "Common Risk Factors in
  the Returns on Stocks and Bonds." Journal of Financial Economics 33(1).

Conventions
-----------
Every Sharpe ratio passed into PSR, E[max SR] and DSR is PER-PERIOD (daily, not
annualized). Annualization (x sqrt(252)) is for display only. Kurtosis is always
Pearson kurtosis (normal = 3), never excess kurtosis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

TRADING_DAYS = 252
WINDOWS: dict[str, int] = {"1y": 252, "3y": 756, "5y": 1260}
MIN_COVERAGE = 0.90
EULER_MASCHERONI = 0.5772156649


def sharpe_ratio(excess: ArrayLike) -> float:
    """Per-period Sharpe ratio: mean(excess) / std(excess, ddof=1).

    ``excess`` is total return minus the per-period risk-free rate. Returns NaN for
    fewer than two observations or zero dispersion.
    """
    x = np.asarray(excess, dtype=float)
    if x.size < 2:
        return math.nan
    sd = x.std(ddof=1)
    if sd == 0.0:
        return math.nan
    return float(x.mean() / sd)


def annualized_sharpe(excess: ArrayLike, periods_per_year: int = TRADING_DAYS) -> float:
    """SR_ann = mean(excess_daily) / std(excess_daily, ddof=1) * sqrt(252) (Sharpe 1994)."""
    return sharpe_ratio(excess) * math.sqrt(periods_per_year)


def window_excess(excess: pd.Series, window: int, min_coverage: float = MIN_COVERAGE) -> NDArray[np.float64] | None:
    """Observed values in the last ``window`` rows of ``excess``.

    ``excess`` must be indexed on the full trading calendar, with NaN for missing days, so
    that the last ``window`` rows are the last ``window`` trading days. Returns None unless
    at least ``min_coverage`` of the window is present; a series shorter than the window
    counts its absent leading days as missing.
    """
    observed = excess.iloc[-window:].dropna().to_numpy(dtype=float)
    if observed.size < math.ceil(min_coverage * window):
        return None
    return observed


def windowed_sharpe_ann(excess: pd.Series, window: int, min_coverage: float = MIN_COVERAGE) -> float:
    """Annualized Sharpe over the trailing ``window`` trading days; NaN below coverage."""
    observed = window_excess(excess, window, min_coverage)
    if observed is None:
        return math.nan
    return annualized_sharpe(observed)


def sample_moments(x: ArrayLike) -> tuple[float, float]:
    """Sample skewness and PEARSON kurtosis (normal = 3) of ``x``.

    Uses the moment estimators of Bailey & López de Prado (2012): gamma3 = m3 / m2^1.5 and
    gamma4 = m4 / m2^2 with central moments m_k = mean((x - mean(x))^k). scipy's
    ``kurtosis`` returns excess kurtosis unless ``fisher=False``; PSR needs Pearson.
    """
    arr = np.asarray(x, dtype=float)
    skew = float(stats.skew(arr, bias=True))
    kurt = float(stats.kurtosis(arr, fisher=False, bias=True))
    return skew, kurt


def probabilistic_sharpe_ratio(sr: float, sr_star: float, n: int, skew: float, kurtosis: float) -> float:
    """Probabilistic Sharpe Ratio, Bailey & López de Prado (2012), eq. for PSR(SR*).

    PSR(SR*) = Phi( (SR - SR*) * sqrt(n - 1) / sqrt(1 - g3*SR + ((g4 - 1) / 4) * SR^2) )

    ``sr`` and ``sr_star`` are per-period (not annualized); ``n`` is the number of
    observations; ``skew`` is g3; ``kurtosis`` is g4, PEARSON (normal = 3). Returns NaN
    when the variance term is non-positive or any input is NaN.
    """
    if n < 2 or not all(math.isfinite(v) for v in (sr, sr_star, skew, kurtosis)):
        return math.nan
    variance_term = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr**2
    if variance_term <= 0.0:
        return math.nan
    z = (sr - sr_star) * math.sqrt(n - 1) / math.sqrt(variance_term)
    return float(stats.norm.cdf(z))


def psr_from_returns(excess: ArrayLike, sr_star: float) -> float:
    """PSR of an excess-return series against the per-period benchmark ``sr_star``."""
    x = np.asarray(excess, dtype=float)
    skew, kurt = sample_moments(x)
    return probabilistic_sharpe_ratio(sharpe_ratio(x), sr_star, x.size, skew, kurt)


def expected_max_sharpe(n_trials: int, mean_sr: float, var_sr: float) -> float:
    """Expected maximum Sharpe ratio across ``n_trials`` independent trials (False Strategy Theorem).

    E[max SR_N] = E[SR] + sqrt(V[SR]) * ((1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e)))

    with g the Euler-Mascheroni constant and Z the standard-normal quantile (Bailey,
    Borwein, López de Prado & Zhu 2014; López de Prado 2020, Machine Learning for Asset
    Managers, ch. 8 and App. B). Inputs are per-period.
    The expression is an extreme-value approximation that diverges at N = 1 because
    Z(0) = -inf; the maximum of a single trial is that trial, so N = 1 returns E[SR].
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return float(mean_sr)
    g = EULER_MASCHERONI
    quantile_mix = (1.0 - g) * stats.norm.ppf(1.0 - 1.0 / n_trials) + g * stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(mean_sr + math.sqrt(var_sr) * quantile_mix)


def null_sharpe_variance(n_obs: int, sr: float = 0.0) -> float:
    """Asymptotic variance of a per-period SR estimate for iid normal returns: (1 + SR^2/2) / n."""
    return (1.0 + sr**2 / 2.0) / n_obs


def deflated_sharpe_ratio(sr: float, n: int, skew: float, kurtosis: float, expected_max_sr: float) -> float:
    """Deflated Sharpe Ratio: PSR evaluated at SR* = E[max SR_N] (all per-period)."""
    return probabilistic_sharpe_ratio(sr, expected_max_sr, n, skew, kurtosis)


@dataclass(frozen=True)
class OLSResult:
    coef: NDArray[np.float64]
    se: NDArray[np.float64]
    n_obs: int

    @property
    def alpha(self) -> float:
        return float(self.coef[0])

    @property
    def alpha_se(self) -> float:
        return float(self.se[0])

    @property
    def alpha_t(self) -> float:
        return self.alpha / self.alpha_se

    @property
    def alpha_ann(self) -> float:
        return self.alpha * TRADING_DAYS


def ols_with_intercept(y: ArrayLike, x: ArrayLike) -> OLSResult:
    """OLS of ``y`` on a constant plus the columns of ``x`` with classic (homoskedastic) SEs.

    beta_hat = (X'X)^-1 X'y; sigma^2 = RSS / (n - k); se = sqrt(diag(sigma^2 (X'X)^-1)),
    where k counts the intercept. coef[0] is the intercept (Jensen's alpha when ``y`` is an
    excess return and ``x`` holds factor returns).
    """
    yv = np.asarray(y, dtype=float)
    xv = np.asarray(x, dtype=float)
    if xv.ndim == 1:
        xv = xv[:, None]
    design = np.column_stack([np.ones(yv.size), xv])
    n, k = design.shape
    if n <= k:
        raise ValueError(f"need more observations ({n}) than regressors ({k})")
    xtx_inv = np.linalg.inv(design.T @ design)
    coef = xtx_inv @ design.T @ yv
    resid = yv - design @ coef
    sigma2 = float(resid @ resid) / (n - k)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    return OLSResult(coef=coef, se=se, n_obs=n)


def jensen_alpha(excess: pd.Series, factors: pd.DataFrame) -> OLSResult | None:
    """Jensen's alpha of daily excess returns on factor returns (CAPM: mkt_rf; FF3: + smb, hml).

    Rows with any missing value are dropped. Returns None when too few rows remain.
    """
    joined = pd.concat([excess.rename("_y"), factors], axis=1, join="inner").dropna()
    if len(joined) <= factors.shape[1] + 1:
        return None
    return ols_with_intercept(joined["_y"].to_numpy(), joined[list(factors.columns)].to_numpy())
