"""Assemble a plain-language integrity report for a backtest return series."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)

# Confidence threshold below which a Sharpe claim is treated as "not established".
SIGNIFICANCE = 0.95


@dataclass
class IntegrityReport:
    n_obs: int
    periods_per_year: int
    sharpe_period: float
    sharpe_annual: float
    psr: float
    min_trl: float
    n_trials: int | None
    dsr: float | None
    expected_max_sr: float | None
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """One-line bottom line, deliberately conservative."""
        if self.dsr is not None:
            if self.dsr < SIGNIFICANCE:
                return "LIKELY OVERFIT — the edge does not survive selection bias."
            return "Survives deflation — the edge is plausible (not proven)."
        if self.psr < SIGNIFICANCE:
            return "NOT ESTABLISHED — the Sharpe is not significant on this sample."
        return "Sharpe is significant in isolation — but trial count is unknown."


def audit(
    returns,
    periods_per_year: int = 252,
    n_trials: int | None = None,
    sr_variance: float | None = None,
    confidence: float = SIGNIFICANCE,
) -> IntegrityReport:
    """Run the full integrity audit on a single backtest return series.

    Parameters
    ----------
    returns : sequence of per-period returns (e.g. daily strategy returns).
    periods_per_year : observations per year, for annualizing the Sharpe only.
    n_trials : how many distinct strategy configurations you tried before
        keeping this one. Required to compute the Deflated Sharpe Ratio --
        without it, overfitting cannot be assessed.
    sr_variance : variance of the per-period Sharpe estimates across those
        trials. If omitted while n_trials is given, it is approximated from
        the single series (a weak fallback -- supplying the real value is
        strongly preferred).
    confidence : significance threshold for the Minimum Track Record Length.
    """
    arr = np.asarray(returns, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    n = arr.size

    sr_period, sr_annual = sharpe_ratio(arr, periods_per_year)
    psr = probabilistic_sharpe_ratio(arr)
    min_trl = minimum_track_record_length(arr, confidence=confidence)

    flags: list[str] = []
    notes: list[str] = []

    dsr: float | None = None
    exp_max: float | None = None
    if n_trials is not None:
        if sr_variance is None:
            # Fallback: estimate per-period SR sampling variance as (1/n).
            # This is the variance of a single SR estimate under the null,
            # NOT the cross-trial variance the DSR really wants. It is a
            # placeholder so the metric runs; supply the real value instead.
            sr_variance = 1.0 / n
            notes.append(
                "sr_variance was not supplied; used the 1/n null approximation. "
                "For a trustworthy DSR, pass the variance of the Sharpe ratios "
                "across all strategy configurations you actually tried."
            )
        exp_max = expected_max_sharpe(sr_variance, n_trials)
        dsr = deflated_sharpe_ratio(arr, n_trials, sr_variance)
    else:
        notes.append(
            "n_trials not provided -- the Deflated Sharpe Ratio was skipped. "
            "If you searched over many parameter sets, universes, or rules and "
            "kept the best, the raw Sharpe is inflated by selection and PSR "
            "alone will overstate your confidence."
        )

    # --- Flags: plain-language warnings, ordered most to least severe -------
    if dsr is not None and dsr < SIGNIFICANCE:
        flags.append(
            f"LIKELY OVERFIT: deflated Sharpe {dsr:.2f} < {SIGNIFICANCE:.2f}. "
            f"Against {n_trials} trials, this edge is not distinguishable from "
            f"the best you'd expect by chance."
        )
    if psr < SIGNIFICANCE:
        flags.append(
            f"SHARPE NOT SIGNIFICANT: PSR {psr:.2f} < {SIGNIFICANCE:.2f}. "
            f"Even ignoring selection bias, the sample does not establish a "
            f"positive Sharpe at {SIGNIFICANCE:.0%} confidence."
        )
    if np.isfinite(min_trl) and min_trl > n:
        flags.append(
            f"TRACK RECORD TOO SHORT: needs ~{min_trl:.0f} observations for "
            f"significance; you have {n}."
        )
    elif not np.isfinite(min_trl):
        flags.append(
            "NEGATIVE/ZERO EDGE: observed Sharpe does not exceed the benchmark; "
            "no track record length would make it significant."
        )
    if n < periods_per_year:
        flags.append(
            f"TINY SAMPLE: {n} observations is under one year "
            f"({periods_per_year}). Treat every number here as indicative only."
        )
    elif n < 2 * periods_per_year:
        notes.append(
            f"Sample is {n} observations (~{n / periods_per_year:.1f} years). "
            f"Short backtests are fragile; longer is materially better."
        )

    return IntegrityReport(
        n_obs=n,
        periods_per_year=periods_per_year,
        sharpe_period=sr_period,
        sharpe_annual=sr_annual,
        psr=psr,
        min_trl=min_trl,
        n_trials=n_trials,
        dsr=dsr,
        expected_max_sr=exp_max,
        flags=flags,
        notes=notes,
    )


def render_text(report: IntegrityReport) -> str:
    """Render an IntegrityReport as a human-readable text block."""
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("  backcheck — backtest integrity report")
    lines.append("=" * 64)
    lines.append(f"  observations        {report.n_obs}")
    lines.append(f"  Sharpe (per period) {report.sharpe_period:+.4f}")
    lines.append(f"  Sharpe (annualized) {report.sharpe_annual:+.3f}")
    lines.append(f"  PSR  (vs 0)         {report.psr:.4f}")
    if report.dsr is not None:
        lines.append(f"  trials searched     {report.n_trials}")
        lines.append(f"  expected max Sharpe {report.expected_max_sr:+.4f} (per period)")
        lines.append(f"  DSR  (deflated)     {report.dsr:.4f}")
    mt = "infinite" if not np.isfinite(report.min_trl) else f"{report.min_trl:.0f}"
    lines.append(f"  min track record    {mt} observations")
    lines.append("-" * 64)

    if report.flags:
        lines.append("  FLAGS")
        for f in report.flags:
            lines.append(f"   [!] {f}")
        lines.append("-" * 64)
    if report.notes:
        lines.append("  NOTES")
        for n in report.notes:
            lines.append(f"   -   {n}")
        lines.append("-" * 64)

    lines.append(f"  VERDICT: {report.verdict}")
    lines.append("=" * 64)
    return "\n".join(lines)
