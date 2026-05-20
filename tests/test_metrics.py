"""Tests for backcheck's integrity metrics.

These check mathematical properties that must hold regardless of
implementation details: monotonicity, bounds, and known limiting cases.
"""

import numpy as np
import pytest

from backcheck import (
    audit,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


def _series(mean, std, n, seed=0):
    """A reproducible normal return series with a target mean and std."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, n)
    raw = (raw - raw.mean()) / raw.std(ddof=1)  # standardize, then rescale
    return raw * std + mean


# --- sharpe_ratio ---------------------------------------------------------

def test_sharpe_ratio_sign_and_annualization():
    r = _series(mean=0.001, std=0.01, n=500)
    sr_p, sr_a = sharpe_ratio(r, periods_per_year=252)
    assert sr_p > 0
    assert sr_a == pytest.approx(sr_p * np.sqrt(252))


def test_sharpe_ratio_rejects_degenerate_input():
    with pytest.raises(ValueError):
        sharpe_ratio([0.01])  # too few observations
    with pytest.raises(ValueError):
        sharpe_ratio([0.01, 0.01, 0.01])  # zero variance


# --- probabilistic_sharpe_ratio ------------------------------------------

def test_psr_is_a_probability():
    r = _series(mean=0.0008, std=0.01, n=750)
    psr = probabilistic_sharpe_ratio(r)
    assert 0.0 <= psr <= 1.0


def test_psr_rises_with_longer_track_record():
    """Same per-period Sharpe, more observations -> more confidence."""
    short = _series(mean=0.0005, std=0.01, n=120, seed=1)
    long = _series(mean=0.0005, std=0.01, n=2000, seed=1)
    assert probabilistic_sharpe_ratio(long) > probabilistic_sharpe_ratio(short)


def test_psr_falls_as_benchmark_rises():
    r = _series(mean=0.0008, std=0.01, n=750)
    low = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
    high = probabilistic_sharpe_ratio(r, benchmark_sr=0.05)
    assert high < low


# --- expected_max_sharpe / deflated_sharpe_ratio -------------------------

def test_expected_max_sharpe_zero_for_single_trial():
    assert expected_max_sharpe(sr_variance=0.01, n_trials=1) == 0.0


def test_expected_max_sharpe_grows_with_trials():
    few = expected_max_sharpe(sr_variance=0.01, n_trials=10)
    many = expected_max_sharpe(sr_variance=0.01, n_trials=10_000)
    assert many > few > 0.0


def test_dsr_never_exceeds_psr():
    """Deflation can only lower confidence: DSR <= PSR for n_trials >= 1."""
    r = _series(mean=0.0009, std=0.01, n=1000)
    psr = probabilistic_sharpe_ratio(r)
    dsr = deflated_sharpe_ratio(r, n_trials=500, sr_variance=0.02)
    assert dsr <= psr + 1e-9


# --- minimum_track_record_length -----------------------------------------

def test_min_trl_is_infinite_when_edge_is_nonpositive():
    r = _series(mean=-0.0002, std=0.01, n=400)
    assert minimum_track_record_length(r) == float("inf")


def test_min_trl_grows_with_confidence():
    r = _series(mean=0.0006, std=0.01, n=800)
    assert (
        minimum_track_record_length(r, confidence=0.99)
        > minimum_track_record_length(r, confidence=0.90)
    )


# --- audit (integration) -------------------------------------------------

def test_audit_flags_overfitting_when_many_trials():
    """A marginal strategy searched over thousands of trials should flag."""
    r = _series(mean=0.0004, std=0.01, n=600)
    report = audit(r, n_trials=5000, sr_variance=0.05)
    assert report.dsr is not None
    assert any("OVERFIT" in f for f in report.flags)


def test_audit_notes_missing_trial_count():
    r = _series(mean=0.0008, std=0.01, n=1000)
    report = audit(r)  # no n_trials
    assert report.dsr is None
    assert any("n_trials" in n for n in report.notes)


def test_audit_flags_tiny_sample():
    r = _series(mean=0.001, std=0.01, n=30)
    report = audit(r, periods_per_year=252)
    assert any("TINY SAMPLE" in f for f in report.flags)
