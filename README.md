# hmm-regime-detection-trader
HMM-based regime detection trading system for stocks, crypto, futures, and options with backtesting and visualization

## Setup

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
```

Requires Python 3.10+.

## Monte Carlo simulation (`hmm_trader/monte_carlo.py`)

A backtest is one path out of many the same edge could have produced. The Monte
Carlo module generates thousands of alternative paths and reports the
distribution of final return, CAGR, max drawdown, Sharpe, P(loss) and risk of ruin.

**Resampling a backtest's per-period strategy returns**

```python
from hmm_trader import monte_carlo as mc

result = mc.run(strategy_returns, method="block_bootstrap", n_sims=5000, seed=1)
print(result.report())
result.table()["max_drawdown"][95]   # 95th-percentile max drawdown
```

| method            | what it does                                   | use when |
|-------------------|------------------------------------------------|----------|
| `shuffle`         | reorders the trades/returns                    | same total return, test drawdown sensitivity to ordering |
| `bootstrap`       | i.i.d. resampling with replacement             | independent trade returns |
| `block_bootstrap` | resamples contiguous blocks (`block_size`)     | daily returns with volatility clustering (default) |

**Stress-testing a regime strategy on HMM-generated markets**

Because an HMM is generative, you can sample fresh regime sequences and returns
from it, then trade them. The regime is estimated causally with a forward
filter (no look-ahead), so detection lag and misclassification are included:

```python
params = mc.GaussianHMMParams.from_hmmlearn(fitted_model)   # or pass transmat/means/stds
strat, market = mc.simulate_hmm_strategy(
    params,
    regime_exposure=[1.0, 0.0],   # position size per regime
    n_sims=5000, horizon=252,
    cost_per_turnover=0.0005,     # 5 bps per unit of position change
)
print(mc.summarize(strat, ruin_threshold=0.3).report())
```

Pass `use_true_states=True` for a perfect-detection upper bound.

Full demo: `python -m examples.monte_carlo_demo`

Caveats: resampling only recombines history it is given, and HMM paths are only
as realistic as the fitted model (Gaussian regimes understate fat tails).
