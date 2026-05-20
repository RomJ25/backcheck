"""Statistical-integrity metrics for trading-strategy backtests.

These are the standard Bailey & Lopez de Prado results for deciding whether a
backtested Sharpe ratio reflects genuine skill or is an artifact of luck,
short samples, or searching over many strategies.

References:
  Bailey, D. & Lopez de Prado, M. (2012).
    "The Sharpe Ratio Efficient Frontier." Journal of Risk.
  Bailey, D. & Lopez de Prado, M. (2014).
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
     Overfitting and Non-Normality." Journal of Portfolio Management.

All Sharpe ratios in this module are PER-PERIOD (same frequency as the
returns array you pass in), not annualized. `sharpe_ratio` returns both so
callers can be explicit about which they mean.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# Euler-Mascheroni constant, used in the expected-maximum-Sharpe estimator.
EULER_MASCHERONI = 0.5772156649015329


def _clean(returns) -> np.ndarray:
    """Coerce input to a 1-D float array and validate it is usable."""
    arr = np.asarray(returns, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        raise ValueError("need at least 2 finite return observations")
    if arr.std(ddof=1) == 0:
        raise ValueError("returns have zero variance; Sharpe ratio is undefined")
    return arr


def sharpe_ratio(returns, periods_per_year: int = 252) -> tuple[float, float]:
    """Return (per_period_sharpe, annualized_sharpe).

    The per-period Sharpe is mean / std (sample std, ddof=1). The annualized
    figure multiplies by sqrt(periods_per_year) -- the conventional, if
    imperfect, scaling. Downstream integrity metrics use the per-period value.
    """
    arr = _clean(returns)
    sr_period = arr.mean() / arr.std(ddof=1)
    return float(sr_period), float(sr_period * np.sqrt(periods_per_year))


def probabilistic_sharpe_ratio(returns, benchmark_sr: float = 0.0) -> float:
    """Probability that the true per-period Sharpe ratio exceeds `benchmark_sr`.

    PSR adjusts the observed Sharpe for sample length, skewness, and kurtosis.
    A PSR of 0.95 means: given this track record, there is a 95% probability
    the strategy's true Sharpe is above the benchmark. `benchmark_sr` must be
    expressed per-period, same frequency as `returns` (0.0 is the common case).
    """
    arr = _clean(returns)
    n = arr.size
    sr = arr.mean() / arr.std(ddof=1)
    skew = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr, fisher=False))  # non-excess: normal == 3
    denom = np.sqrt(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("variance estimator is non-positive; cannot compute PSR")
    z = ((sr - benchmark_sr) * np.sqrt(n - 1)) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Expected maximum of `n_trials` independent Sharpe estimates.

    When you search over many strategy configurations and keep the best, the
    winning Sharpe is inflated purely by selection. This estimates how high a
    Sharpe you would expect from the *best of n_trials* even if no strategy had
    any real edge. `sr_variance` is the variance of the per-period Sharpe
    estimates across all the trials you ran.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    if sr_variance < 0:
        raise ValueError("sr_variance must be non-negative")
    sr_std = np.sqrt(sr_variance)
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sr_std * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(returns, n_trials: int, sr_variance: float) -> float:
    """Probabilistic Sharpe Ratio evaluated against the selection-bias benchmark.

    The Deflated Sharpe Ratio (DSR) is PSR with the benchmark set to the
    expected maximum Sharpe from `n_trials` (see `expected_max_sharpe`). A DSR
    below 0.95 means the backtested Sharpe is not convincingly better than what
    pure search-luck would have produced. This is the single most important
    metric here: it is the math that catches backtest overfitting.
    """
    sr_star = expected_max_sharpe(sr_variance, n_trials)
    return probabilistic_sharpe_ratio(returns, benchmark_sr=sr_star)


def minimum_track_record_length(
    returns, benchmark_sr: float = 0.0, confidence: float = 0.95
) -> float:
    """Minimum number of observations for PSR(`benchmark_sr`) to reach `confidence`.

    If this exceeds the length of your actual track record, your sample is too
    short to claim the Sharpe is real at the requested confidence level.
    Returns +inf when the observed Sharpe does not even exceed the benchmark.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    arr = _clean(returns)
    sr = arr.mean() / arr.std(ddof=1)
    if sr <= benchmark_sr:
        return float("inf")
    skew = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr, fisher=False))
    z = stats.norm.ppf(confidence)
    min_trl = 1.0 + (
        1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    ) * (z / (sr - benchmark_sr)) ** 2
    return float(min_trl)
