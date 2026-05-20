# backcheck

**Audit a trading-strategy backtest for the statistical sins that make it lie to you.**

Most backtests look better than the strategy really is. Three things cause it:

1. **Short samples** — a year of data is far too little to tell skill from luck.
2. **Selection bias** — if you tried 500 parameter sets and kept the best, the winning Sharpe ratio is inflated by the search itself, not by edge.
3. **Non-normal returns** — skew and fat tails make the naive Sharpe ratio overstate confidence.

`backcheck` takes a strategy's return series and reports whether its Sharpe ratio reflects genuine skill or is an artifact of those three. It implements the standard Bailey & López de Prado results — the Probabilistic Sharpe Ratio, the Deflated Sharpe Ratio, and the Minimum Track Record Length — behind a one-line CLI and a small Python API.

It does not run backtests. It checks the ones you already have.

## Why this exists

`backcheck` was built after a different project — a transparent quant stock screener — turned out to have a backtest inflated by survivorship bias and in-sample overfitting. The "edge" did not survive honest testing. This is the tool that would have caught it sooner. The lesson generalizes: in quantitative finance the default outcome of a backtest is a false positive, and almost no easy tooling makes the correction routine. `backcheck` makes it one command.

## Install

```bash
pip install -e .          # from a clone
# (a PyPI release will follow once the API settles)
```

Requires Python 3.9+, `numpy`, and `scipy`.

## Command line

```bash
# Audit a CSV of per-period (e.g. daily) strategy returns
backcheck returns.csv

# Pick a column by name, and tell backcheck how many strategies you searched
backcheck returns.csv --column strategy_return --trials 200

# Treat the column as a price series instead of returns
backcheck prices.csv --prices --column close
```

`backcheck` exits non-zero when the audit finds a problem, so it works as a CI
gate on a strategy repository.

A runnable example ships in [`examples/returns.csv`](examples/returns.csv) — a
marginal strategy that *looks* mildly profitable. Running:

```bash
backcheck examples/returns.csv --column strategy_return --trials 5000 --sr-variance 0.05
```

produces:

```
================================================================
  backcheck — backtest integrity report
================================================================
  observations        600
  Sharpe (per period) +0.0142
  Sharpe (annualized) +0.226
  PSR  (vs 0)         0.6360
  trials searched     5000
  expected max Sharpe +0.8246 (per period)
  DSR  (deflated)     0.0000
  min track record    13396 observations
----------------------------------------------------------------
  FLAGS
   [!] LIKELY OVERFIT: deflated Sharpe 0.00 < 0.95. Against 5000
       trials, this edge is not distinguishable from the best
       you'd expect by chance.
   [!] SHARPE NOT SIGNIFICANT: PSR 0.64 < 0.95. ...
   [!] TRACK RECORD TOO SHORT: needs ~13396 observations; you have 600.
----------------------------------------------------------------
  VERDICT: LIKELY OVERFIT — the edge does not survive selection bias.
================================================================
```

The strategy's annualized Sharpe of 0.23 looks unremarkable-but-positive. The
audit shows it is not significant even in isolation, and — once you account for
having searched 5000 configurations — indistinguishable from noise.

## Python API

```python
from backcheck import audit, render_text

returns = [...]  # per-period strategy returns

report = audit(
    returns,
    periods_per_year=252,
    n_trials=200,        # how many strategy configs you searched
    sr_variance=0.04,    # variance of the Sharpe ratios across those configs
)

print(render_text(report))
print(report.verdict)
print(report.dsr)        # deflated Sharpe ratio (0..1)
```

Individual metrics are available directly:

```python
from backcheck import (
    sharpe_ratio,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    minimum_track_record_length,
)
```

## The metrics

| Metric | Question it answers |
|---|---|
| **Probabilistic Sharpe Ratio (PSR)** | Given this much data and these return shapes, how confident can I be the true Sharpe is above zero? |
| **Deflated Sharpe Ratio (DSR)** | After accounting for how many strategies I searched, is the Sharpe still convincing? This is the metric that catches overfitting. |
| **Minimum Track Record Length** | How many observations would I actually need for significance? |

The honest interpretation: a **DSR below 0.95** means the backtested edge is not
distinguishable from the best result you would expect by chance from your
search. The most important input is therefore `--trials` / `n_trials` — the
number of strategy configurations you tried. Omit it and `backcheck` will say
so, because without it overfitting cannot be assessed.

## Status

`v0.1` — alpha. The metrics implemented (PSR, DSR, Minimum Track Record Length)
are closed-form and tested. Planned next:

- **Probability of Backtest Overfitting (PBO)** via combinatorially symmetric
  cross-validation — for when you have many candidate strategies' return series.
- Look-ahead- and survivorship-bias heuristics that read backtest metadata.
- A `pandas` convenience layer.

Issues and contributions are welcome.

## References

- Bailey, D. & López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier.* Journal of Risk.
- Bailey, D. & López de Prado, M. (2014). *The Deflated Sharpe Ratio.* Journal of Portfolio Management.
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.

## License

MIT — see [LICENSE](LICENSE).

---

*`backcheck` is a statistical tool, not financial advice. A passing report is
evidence a strategy is not obviously overfit — it is not a guarantee of future
returns.*
