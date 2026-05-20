"""backcheck — audit a trading-strategy backtest for the statistical sins
that make it lie to you.

Public API:
    sharpe_ratio                 per-period and annualized Sharpe
    probabilistic_sharpe_ratio   PSR — is the Sharpe significant at all?
    deflated_sharpe_ratio        DSR — does it survive selection bias?
    expected_max_sharpe          the selection-bias benchmark
    minimum_track_record_length  how much data significance would need
    audit                        full integrity report for a return series
    render_text                  format a report as readable text
"""

from .metrics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from .report import IntegrityReport, audit, render_text

__version__ = "0.1.0"

__all__ = [
    "sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "minimum_track_record_length",
    "audit",
    "render_text",
    "IntegrityReport",
    "__version__",
]
