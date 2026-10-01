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

## Mean reversion on MES (`examples/mes_mean_reversion.py`)

`hmm_trader/mean_reversion.py` has two rule sets, both fixed in advance:

- **Daily RSI(2)** (`rsi2_positions`, Connors & Alvarez 2008): buy at the close
  when RSI(2) < 10 and the close is above its 200-day average (`trend=None`
  drops the filter), sell at the first close above the 5-day average. Long
  only; holds overnight.
- **Intraday Bollinger fade** (`bollinger_fade`): fade a 5-minute close outside
  the 20-bar +/- 2 sd band at the next open, stop 2 sd away, exit on a close back
  through the average or at 15:55 ET; flat every night.

```bash
python -m examples.mes_mean_reversion                 # both parts
python -m examples.mes_mean_reversion --part a --horizon 250
python -m examples.mes_mean_reversion --part a --symbols MES,MNQ,M2K
python -m examples.mes_mean_reversion --part c --symbols MES,MNQ,M2K   # flat through breaks
python -m examples.mes_mean_reversion --part c --contracts 1,2,3,4,5,6,8 --horizon 250
python -m examples.mes_mean_reversion --part d --contracts 1,2,3,4,5,6,8 --horizon 250  # stops
python -m examples.mes_mean_reversion --part e --contracts 2,3,4,5,6 --horizon 250  # daily loss limit
```

Part C re-runs the RSI(2) signals for prop firms that require being flat through
the daily maintenance break: each held session is bought at its open and sold at
its close (a round trip every session), and, where 5-minute data exists, only
09:30-15:55 ET on the same days. Part D adds a stop k x ATR(10) below the first
entry (`flat_session_trades(..., stop_atr=k)`) for k = none / 1.5 / 2 / 3. Part E adds
an account daily loss limit of none / $1,000 / $750 / $500 (`session_stop`, in points
per contract): the position is sold when the session's loss from its open reaches
it, and resumes at the next open while the signal lasts. Part A runs on any symbol with bundled daily data (MES, MNQ and
`examples/data/m2k_daily.csv`, TradingView `M2K1!`/`M2K2!` 2019-05 to 2026-09;
M2K's continuous series rolls one session later than MES/MNQ, set per contract in
`ContractSpec.roll_sessions`). The example reports the RSI(2) trades against randomly timed trades of the same
lengths (so the long bias of a rising market does not count as skill), a
next-open entry variant, an entry x exit sensitivity grid, the fade in R by
period and before costs, and prop pass rates against a zero-edge baseline.

## RSI(2) pullback for TradingView (`pine/rsi2_pullback.pine`)

A Pine Script v6 strategy of the RSI(2) pullback from the mean-reversion study:
long when RSI(2) closes below 10 above the 200-day average, out on the first
close above the 5-day average. Its default mode is the prop version (buy at each
6 PM ET session open, sell at the daily close, so never in a position through
the daily break), 2 contracts, $0.62 commission and 1 tick of slippage per side.
"Hold overnight" mode keeps one position from the next open to the exit close.

To use it: open a daily `MES1!`, `MNQ1!`, `M2K1!`, `MYM1!`, `MGC1!` or `MCL1!` chart with back-adjustment on (`B-ADJ`),
add the script from the Pine Editor, and read the Strategy Tester. The size
follows the chart's symbol unless set: 2 MES, 1 MNQ, 3 M2K, 3 MYM, 1 MGC or 1 MCL, the
sizes tested for a $50k evaluation (the MYM, MNQ and M2K edges are weaker:
timing p = 0.06, 0.16 and 0.08 against 0.003 on MES; 3 MYM passed 32% of
evaluations against 18% with no edge). On gold the rule showed no edge (p = 0.43; 1 MGC
passed 7.6% of evaluations against 7.0% with no edge), and on crude (from
2022) prop mode lost money (p = 0.41) and passed almost no evaluations. Gold's
and crude's daily closes are their 1:30 and 2:30 PM ET settlements, so prop
mode on them sells then.
Its status panel lists the closing prices that would trigger a buy or an exit
at the next close (`mean_reversion.rsi2_levels` computes the same levels). It
plots the trend average, shades the sessions in a trade, shows a status panel
(RSI, trend filter, what to do at the next open, size, backtest evidence) and
sends alerts at the daily close, naming the symbol and size, when an alert is
created with "alert() function calls only". Alerts are per chart, so create one
on each of `MES1!`, `MNQ1!`, `M2K1!`, `MYM1!`, `MGC1!` and `MCL1!`.

`python -m examples.rsi2_trade_list` writes the Python backtest's trades to
`pine/rsi2_<symbol>_reference_trades.csv`, to compare with TradingView's List of
trades (2020-02 to 2026-09; MCL from 2022-04). Prop-mode totals per contract:

| Symbol | Trades | Prop P&L |
|---|---|---|
| MES | 60 | +$8,643 |
| MNQ | 55 | +$9,116 |
| M2K | 51 | +$2,515 |
| MYM | 60 | +$3,445 |
| MGC | 37 | +$748 |
| MCL | 20 | -$981 |

The strategy lists each session as its own trade, and small differences are
expected where TradingView's back-adjusted prices put RSI or an average right at
a threshold.

`pine/rsi2_combined_alerts.pine` is an indicator that watches MES and a second
index micro (MNQ or MYM, an input) from one chart and sends one alert per daily
close listing each buy, still-long and exit, with what to trade under a chosen
plan. `python -m examples.rsi2_combined` scores the plans in one $50k account
(2020-02 to 2026-09):

| Plan | Net $ | Worst day | Pass vs no edge |
|---|---|---|---|
| 2 MES only (default) | +17,287 | -1,450 | 39.0% vs 15.3% |
| 2 MES, else 1 MNQ | +15,677 | -1,450 | 36.5% vs 16.1% |
| 1 MES + 1 MNQ | +17,760 | -2,392 | 35.1% vs 14.1% |
| 2 MES, else 3 MYM | +14,172 | -1,825 | 35.5% vs 20.2% |
| 1 MES + 2 MYM | +15,533 | -1,991 | 35.2% vs 15.9% |

MES is in a trade together with MNQ on 143, and with MYM on 141, of its 206
sessions in a trade. The sessions with a signal on the second market alone lost
money ($1,609 per MNQ, $1,038 per MYM contract), so adding either stacks risk
without adding edge; the alert reports the second market as info only by
default.

`examples/data/mym_daily.csv` is TradingView `MYM1!`/`MYM2!` daily (micro Dow,
2019-05 to 2026-09), which rolls on the MES/MNQ schedule.

## Buy & hold as a prop evaluation (`examples/buy_and_hold.py`, `pine/buy_and_hold.pine`)

`python -m examples.buy_and_hold` compares buy & hold 1 MES (held, or flat
through every daily break) with RSI(2) on 2 MES (held overnight, or flat through
every break) from 2020-02-25, after costs:

| | Net $ | Max DD | Every start | Bootstrap | No edge |
|---|---|---|---|---|---|
| Buy & hold, 1 MES | +18,368 | 6,067 | 39.9% | 38.9% | 22.3% |
| Buy & hold, 1 MES, flat each break | +12,354 | 7,138 | 29.2% | 32.3% | 21.7% |
| RSI(2), 2 MES, held overnight | +19,339 | 3,092 | 57.3% | 46.5% | 14.1% |
| RSI(2), 2 MES, flat each break | +17,287 | 3,876 | 41.4% | 39.0% | 15.3% |

Buy & hold makes money because the S&P rose, but its drawdown is three times the
$2,000 limit: its pass rate by start year runs from 93% (2020, near the pandemic
low) to 15% (2024) and 19% (2022, when it lost $4,737 per contract). It also
prints net profit by calendar year.

`pine/buy_and_hold.pine` is the TradingView version (hold or prop mode, same
costs) with an evaluation check: from a start date it replays one $50k
evaluation on the chart ($3,000 target, $2,000 end-of-day trailing drawdown,
250 sessions) and marks the pass or failure. Its header lists the backtest's
outcomes for several start dates to compare against.

## Intraday momentum on MES / MNQ (`examples/intraday_momentum.py`)

`hmm_trader/intraday_momentum.py` has two published intraday rules on 5-minute
regular-hours bars, with their published settings:

- **Last half hour** (Gao, Han, Li & Zhou, "Market Intraday Momentum", JFE 2018):
  the return from the previous close to 10:00 ET sets the side, held 15:30 to
  16:00. A variant uses the return to 15:30 (Baltussen et al., JFE 2021).
- **Noise boundary + VWAP** (Zarattini, Aziz & Barbon, "Beat the Market", 2024):
  14-day noise bands around the open, checks every 30 minutes from 10:00,
  VWAP trailing exit, flat at 16:00, fixed size instead of volatility sizing.

```bash
python -m examples.intraday_momentum                 # MES (primary) and MNQ
```

The 5-minute files now carry `volume` (for VWAP) and `prev_close` (the
previous regular-hours close of the same contract), from Massive.com; every
price matched the bars already bundled. Results, 2024-10 to 2026-09, 1
contract after costs:

| | MES net | MES flip p | MNQ net | MNQ flip p |
|---|---|---|---|---|
| Last half hour, 10:00 signal | -1,736 | 0.47 | +817 | 0.24 |
| Last half hour, 15:30 signal | -4,451 | 0.96 | -5,309 | 0.94 |
| Noise boundary + VWAP | -2,698 | 0.66 | +6,567 | 0.10 |

On MES, the primary test, neither rule works: the first half hour had no
relation to the last (correlation -0.03). The noise-boundary rule on MNQ made
money in 7 of 8 quarters and in every neighbouring setting, and passed 44% of
bootstrap evaluations at 1 MNQ against 20% with no edge (52% of historical
start days), but it failed on MES and its flip p of 0.10 is not significant
over two years: a lead to watch, not a proven edge.

`pine/noise_boundary.pine` is the TradingView version for a 5-minute `MNQ1!`
chart: the same bands, checks, VWAP exit and 16:00 close, with a status panel
(position, bands, VWAP, the next check and its levels, today's P&L) and alerts.
`python -m examples.intraday_momentum --write-trades` writes the backtest's
trades to `pine/noise_boundary_<symbol>_reference_trades.csv` for comparison.

## Smart-money setups: sweeps, fair value gaps, order blocks (`examples/smc_backtest.py`)

`hmm_trader/smc.py` turns the discretionary "smart money" ideas into fixed
rules on 5-minute regular-hours bars:

- **Liquidity levels:** the previous session's high/low and today's swing points.
- **Sweep:** a bar trades through an untouched level and closes back inside.
- **Fair value gap:** a 3-bar gap; for a bullish gap, bar 3's low is above bar 1's high.
- **Order block (supply/demand zone):** the last opposite bar before the displacement bar.
- **Order flow:** proxied by the displacement bar's volume against normal volume at that time of day. There is no bid/ask data, so there is no true delta.

The full model is a sweep, then within 30 minutes a gap the other way on
>= 1.5x volume, then a limit order at the gap's edge. The order fills only if
price trades through it, the stop goes beyond the sweep, and the target is 2R.
The position is flat at 16:00, with $200 risk per trade. The other variants
drop or swap one piece. The side test enters each trade and its mirror image
at the next bar's open, so the shape of the fill bar does not favour either
side.

```bash
python -m examples.smc_backtest                     # MES and MNQ, 2024-10 .. 2026-09
```

| avg R after costs (trades) | MES | MNQ |
|---|---|---|
| Full model | +0.06 (117) | -0.05 (84) |
| No volume filter | -0.03 (393) | +0.10 (431) |
| FVG only (no sweep) | +0.02 (454) | +0.01 (392) |
| Order block entry | -0.02 (71) | -0.18 (55) |
| Sweep only (fade every sweep) | **-0.27 (843)** | **-0.14 (887)** |
| Full, 1R target | +0.17 (120) | -0.03 (84) |

- **No setup is consistently positive on both markets.** The full model is within noise on both.
- **Fading every sweep loses clearly.** Swept levels more often keep going than reverse.
- **The 1R version on MES** (side p 0.04) is one cell of 14 and failed on MNQ, so it is what chance alone could produce.

## Trend following on MES (`examples/mes_trend.py`)

`hmm_trader/trend.py` has three textbook rules on roll-adjusted daily closes,
all causal: `sma_cross_positions` (50/200, long/flat or long/short),
`donchian_positions` (55-day breakout, 20-day exit, Turtle-style) and
`tsmom_positions` (sign of the 12-month return). `signed_session_steps` turns
long/short positions into per-session P&L, held through the daily break or flat
through it, and `circular_shift_pvalue` is a timing test that shifts a rule's own
position series in time (same exposure and trade lengths, scrambled timing).

```bash
python -m examples.mes_trend                     # MES, 2020-05 .. 2026-09
python -m examples.mes_trend --symbol MNQ --contracts 1,2
python -m examples.mes_trend --symbol MGC --contracts 1,2,3   # gold
python -m examples.mes_trend --symbol MCL --contracts 1,2,3   # crude, 2022-07 on
```

`examples/data/mgc_daily.csv` is TradingView `MGC1!`/`MGC2!` daily, 2019-05 to
2026-09. Gold rolls to the next active month (Feb/Apr/Jun/Aug/Dec) on the
second-to-last session of Jan/Mar/May/Jul/Nov (`futures.gold_roll_mask`, chosen
per contract by `ContractSpec.roll_rule`). Its daily close is the 13:30 ET
settlement (`ContractSpec.daily_close_et`), so the flat-through-breaks version on
gold is also flat for the 13:30-16:10 ET hours a prop account could trade.

`examples/data/mcl_daily.csv` is TradingView `MCL1!`/`MCL2!` daily from MCL's
launch (2021-07) to 2026-09, so after the 252-session warm-up the test covers
2022-07 on. Crude rolls monthly, 2 sessions before WTI's last trading day
(3 business days before the 25th, 4 if the 25th is not a business day) and
1 session before it in February (`futures.crude_roll_mask`, with exchange
holidays from `futures.us_exchange_holidays`). Its daily close is the 14:30 ET
settlement, with the same caveat as gold.

The example compares each rule with buy & hold, reports the timing test, net by
year, a fast/slow sensitivity table and prop pass rates (flat through every
daily break) against the zero-edge baseline.
