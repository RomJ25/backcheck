"""Command-line interface for backcheck.

Reads a CSV of strategy returns (or prices) and prints an integrity report.

Examples:
    backcheck returns.csv
    backcheck returns.csv --column strategy_return
    backcheck prices.csv --prices --column close --trials 200
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from . import __version__
from .report import audit, render_text


def _load_series(path: str, column: str | None) -> np.ndarray:
    """Load one numeric column from a CSV as a float array.

    `column` may be a header name or a 0-based integer index. If omitted, the
    last column is used.
    """
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise SystemExit(f"error: {path} is empty")

    header = rows[0]
    has_header = any(not _looks_numeric(cell) for cell in header)
    data_rows = rows[1:] if has_header else rows

    idx: int
    if column is None:
        idx = len(header) - 1
    elif column.isdigit():
        idx = int(column)
    elif has_header and column in header:
        idx = header.index(column)
    else:
        raise SystemExit(
            f"error: column {column!r} not found. "
            f"available: {header if has_header else 'use a 0-based index'}"
        )

    values: list[float] = []
    for i, row in enumerate(data_rows, start=2):
        if idx >= len(row):
            continue
        cell = row[idx].strip()
        if cell == "":
            continue
        try:
            values.append(float(cell))
        except ValueError:
            raise SystemExit(f"error: non-numeric value {cell!r} on line {i}")

    if len(values) < 2:
        raise SystemExit("error: need at least 2 numeric observations")
    return np.asarray(values, dtype=float)


def _looks_numeric(cell: str) -> bool:
    try:
        float(cell.strip())
        return True
    except ValueError:
        return False


def _prices_to_returns(prices: np.ndarray) -> np.ndarray:
    """Convert a price series to simple per-period returns."""
    if np.any(prices[:-1] == 0):
        raise SystemExit("error: price series contains a zero; cannot form returns")
    return prices[1:] / prices[:-1] - 1.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="backcheck",
        description="Audit a backtest for the statistics that make it lie.",
    )
    p.add_argument("csv", help="path to a CSV of strategy returns (or prices)")
    p.add_argument(
        "--column", "-c", default=None,
        help="column to read: header name or 0-based index (default: last column)",
    )
    p.add_argument(
        "--prices", action="store_true",
        help="treat the column as a price series and convert to returns",
    )
    p.add_argument(
        "--periods-per-year", "-p", type=int, default=252,
        help="observations per year, for annualizing the Sharpe (default: 252)",
    )
    p.add_argument(
        "--trials", "-t", type=int, default=None,
        help="number of strategy configurations searched; enables the "
             "Deflated Sharpe Ratio. Omitting this hides overfitting.",
    )
    p.add_argument(
        "--sr-variance", type=float, default=None,
        help="variance of the per-period Sharpe ratios across all trials "
             "(used with --trials for an accurate Deflated Sharpe Ratio)",
    )
    p.add_argument(
        "--confidence", type=float, default=0.95,
        help="significance threshold for the minimum track record (default: 0.95)",
    )
    p.add_argument("--version", action="version", version=f"backcheck {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    series = _load_series(args.csv, args.column)
    returns = _prices_to_returns(series) if args.prices else series

    try:
        report = audit(
            returns,
            periods_per_year=args.periods_per_year,
            n_trials=args.trials,
            sr_variance=args.sr_variance,
            confidence=args.confidence,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")

    print(render_text(report))
    # Exit non-zero when the audit found a problem, so backcheck is usable
    # as a CI gate on a strategy repository.
    return 1 if report.flags else 0


if __name__ == "__main__":
    sys.exit(main())
