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

**On real prices** (`pip install -r requirements-examples.txt`): fits an HMM to
daily closes and runs all four analyses (resampled actual returns, and buy &
hold vs. the regime strategy on HMM-simulated paths). Ships with 2 years of SPY
closes in `examples/data/`:

```bash
python -m examples.spy_monte_carlo                       # bundled SPY sample
python -m examples.spy_monte_carlo prices.csv --states 3 --exposure 1,0.5,0
```

A CSV needs a `close` or `adj_close` column (plus an optional `date` column);
`--help` lists the other options. Two years with one crash is too little to fit
regimes reliably, so use 10+ years of history for real decisions.

**Walk-forward (out-of-sample) test.** By default the HMM is fitted on the same
data it trades, which flatters the strategy. `--walk-forward` fits only on data
before the test period, refits every `--refit-every` periods (default 63, about
quarterly) on an expanding window, and reports buy & hold vs. the strategy on
the unseen test period:

```bash
python -m examples.spy_monte_carlo prices.csv --walk-forward --test-start 2022-01-01
python -m examples.spy_monte_carlo --walk-forward --train-frac 0.6 --refit-every 0   # fit once
```

Each refit runs `--starts` random restarts, so long tests take a minute or two;
lower `--starts` to speed them up.

Caveats: resampling only recombines history it is given, and HMM paths are only
as realistic as the fitted model (Gaussian regimes understate fat tails).

## Prop firm challenge simulation (`hmm_trader/prop_firm.py`)

A prop firm evaluation is won or lost on the path, not the average return: the
account has to hit a profit target before it breaches a max drawdown or daily
loss limit. `simulate_challenge` replays simulated return paths through a set of
challenge rules and reports the pass rate, which rule failed the rest, and how
many days passing took.

```python
from hmm_trader import monte_carlo as mc, prop_firm as pf

rules = pf.ChallengeRules(
    profit_target=0.10,        # all limits are fractions of the initial balance
    max_drawdown=0.10,
    daily_loss_limit=0.05,     # from the start-of-day balance; None to disable
    drawdown_type="static",    # or "trailing" (real time) / "trailing_eod"
    trailing_cap=None,         # 0.0 = trailing floor stops at the starting balance
    min_trading_days=4,        # days with a nonzero return
    max_days=None,             # time limit, if the firm has one
    consistency=None,          # e.g. 0.5 = best day <= 50% of total profit
)
paths = mc.block_bootstrap_paths(backtest_daily_returns, n_sims=5000, horizon=252)
res = pf.simulate_challenge(paths, rules, leverage=1.0)
print(res.report())
res.pass_rate, res.rate(pf.FAILED_DAILY_LOSS), res.days_to_pass()[50]
```

`paths` can be any `(n_sims, n_days)` array of daily strategy returns, including
`mc.simulate_hmm_strategy` output. `leverage` scales every return, so sweeping it
shows how position size trades pass rate against blow-ups. With daily returns,
breaches are only seen on the daily close, which understates failures; pass
`(n_sims, n_days, steps_per_day)` intraday returns to catch intraday dips.

Demo comparing buy & hold and the regime filter across leverage under a static
and a trailing drawdown rule set: `python -m examples.prop_firm_demo`

## Micro futures backtest (`examples/micro_futures_backtest.py`)

Walk-forward test of the HMM regime filter on micro E-mini futures (MES, MNQ),
scored in dollars per contract and against a $50k futures prop evaluation
(default: $3,000 target, $2,000 end-of-day trailing drawdown that stops at the
starting balance; override with `--target`, `--max-dd`, `--dd-type`,
`--daily-loss`, ...).

```bash
python -m examples.micro_futures_backtest                       # MES + MNQ, 1/2/3/5 contracts
python -m examples.micro_futures_backtest --symbols MNQ --contracts 1,2 --overnight
```

- **Rolls** (`hmm_trader/futures.py`): continuous front-month series jump by the
  calendar spread at each quarterly roll (about +1% for index futures since
  2022), which an unadjusted backtest books as profit. The bundled data carries
  the second-month close, so roll-day P&L is measured from the contract actually
  rolled into. The roll calendar (3 sessions before the 3rd-Friday expiry)
  was checked against the cash index.
- **Sessions**: by default positions open at the session open and close at the
  settlement, flat through the daily break as futures prop firms require;
  `--overnight` holds between sessions. Costs: $0.62 commission plus 1 tick of
  slippage per side.
- **Intraday risk**: sessions are replayed as open -> high -> low -> close, the
  worst ordering for a long, so drawdown breaches are counted conservatively.

Bundled data: `examples/data/{mes,mnq}_daily.csv`, TradingView `CME_MINI:MES1!`
/ `MNQ1!` daily bars from 2019-05-06 to 2026-09-29 dated by trade date, with the
`2!` close as `next_close`.

## Opening range breakout on MES / MNQ (`examples/orb_backtest.py`)

`hmm_trader/orb.py` backtests a one-trade-a-day opening range breakout on 5-minute
bars: stop entries one tick beyond the first 5/15/30 minutes' range after 09:30
ET, stop at the other side, exit at 1R / 2R or flat at 15:55 ET, sized to a fixed
dollar risk (days too wide for one contract are skipped). Fills are conservative:
a tick of slippage on stop entries and exits, a bar touching both stop and target
counts as a stop, gaps fill at the open.

```bash
python -m examples.orb_backtest                        # MES + MNQ, $200 risk grid
python -m examples.orb_backtest --symbols MGC,MCL      # micro gold and crude
python -m examples.orb_backtest --symbols MNQ --risk 200,400 --split 2025-07-01
```

Session times per symbol (`SESSIONS` in the example): MES/MNQ range from 09:30 ET,
flat by 15:55; MGC from the 08:20 ET pit open, flat by the 13:30 settlement; MCL
from 09:00 ET, flat by 14:30.

The example picks the best grid point on data before `--split` and reports it on
the unseen data after, then scores it as a $50k prop evaluation next to a
zero-edge baseline (the same trades minus their average profit).

Bundled data: `examples/data/{mes,mnq,mgc,mcl}_5min_rth.csv.gz`, 5-minute bars of
the front contract from Massive.com, 2024-10-01 to 2026-09-28, times in
US/Eastern. MES/MNQ cover 09:30-16:00 and roll on the `hmm_trader.futures`
calendar; MGC/MCL cover 08:00-16:00 and use each day's highest-volume contract
(never rolling back), which puts gold on Feb/Apr/Jun/Aug/Dec and crude on the
next monthly contract about a week before expiry. The vendor data has a few
mid-session gaps; the backtest resumes at the next bar's open.

**Narrow-range filter** (`examples/orb_narrow_range.py`): `ORBParams(max_range_ratio=k,
range_lookback=N)` trades only when today's opening range is at most `k` times the
median of the previous `N` sessions (causal, with an `N`-session warm-up). The
example tests a 30-minute ORB with k = 1, N = 20 on all four symbols, with a
permutation test against random same-size day subsets, a split by period, a k x N
robustness grid and a prop evaluation against the zero-edge baseline:

```bash
python -m examples.orb_narrow_range
python -m examples.orb_narrow_range --k 0.8 --lookback 10
```
