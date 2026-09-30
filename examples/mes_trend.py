"""Trend following on MES daily bars: moving-average crossover, Donchian
breakout and time-series momentum.

Run from the repo root:  python -m examples.mes_trend

The rules and their settings are the textbook ones, fixed before looking at
this data (see ``hmm_trader.trend``); 1 contract, after costs (commission +
1 tick per side, a round trip for each roll carried):

- SMA 50/200 long/flat, and long/short
- Donchian 55-day breakout / 20-day exit, long/short (Turtle-style) and long only
- 12-month time-series momentum, long/short

S&P futures rose a lot in 2020-2026, so any rule that is mostly long looks
good. The timing test shifts each rule's own position series in time
(every circular shift of at least 60 sessions): same exposure and trade
lengths, scrambled timing. It asks whether *when* the rule is long, short or
flat added anything. By-year results show whether the rules did their job in
the 2020 crash and the 2022 bear market. A sensitivity table shows faster and
slower settings, and the prop evaluation uses the version that is flat
through every daily break (entered at each session open).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from examples.mes_mean_reversion import ACCOUNT, RULES, demean, max_drawdown
from examples.orb_backtest import block_bootstrap_index
from hmm_trader import futures as fu
from hmm_trader import prop_firm as pf
from hmm_trader import trend as tr

DATA_DIR = Path(__file__).parent / "data"
WARMUP = 252  # every rule has its inputs from here on

PRIMARY = {
    "SMA 50/200 long/flat": lambda c: tr.sma_cross_positions(c, 50, 200),
    "SMA 50/200 long/short": lambda c: tr.sma_cross_positions(c, 50, 200, allow_short=True),
    "Donchian 55/20 L/S": lambda c: tr.donchian_positions(c, 55, 20),
    "Donchian 55/20 long only": lambda c: tr.donchian_positions(c, 55, 20, allow_short=False),
    "TSMOM 12m L/S": lambda c: tr.tsmom_positions(c, 252),
}
SENSITIVITY = {
    "SMA long/flat": [(f"{f}/{s}", lambda c, f=f, s=s: tr.sma_cross_positions(c, f, s))
                      for f, s in ((20, 100), (50, 200), (100, 250))],
    "Donchian L/S": [(f"{e}/{x}", lambda c, e=e, x=x: tr.donchian_positions(c, e, x))
                     for e, x in ((20, 10), (55, 20), (100, 50))],
    "TSMOM L/S": [(f"{m}m", lambda c, m=m: tr.tsmom_positions(c, m * 21))
                  for m in (3, 6, 12)],
}


def sharpe(daily: np.ndarray) -> float:
    sd = daily.std(ddof=1)
    return float(daily.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbol", default="MES")
    ap.add_argument("--contracts", default="1,2,3",
                    type=lambda t: [int(x) for x in t.split(",") if int(x) > 0])
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--horizon", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    symbol = args.symbol.upper()
    path = DATA_DIR / f"{symbol.lower()}_daily.csv"
    if symbol not in fu.CONTRACTS or not path.exists():
        ap.error(f"no bundled daily data for {symbol}")
    rng = np.random.default_rng(args.seed)

    spec = fu.CONTRACTS[symbol]
    cost = 0.62 + spec.tick_value
    bars = fu.load_bars(path)
    roll = fu.roll_mask(bars.dates, spec)
    close = bars.close[0] * np.r_[1.0, np.cumprod(1 + fu.adjusted_returns(bars, roll))]
    dates = bars.dates[1:]
    sl = slice(WARMUP, None)
    years = np.array([d[:4] for d in dates[sl]])
    ones = np.ones(dates.size, dtype=int)
    gross = tr.signed_session_steps(bars, roll, ones, spec.multiplier, 0.0).sum(axis=1)
    bh = tr.signed_session_steps(bars, roll, ones, spec.multiplier, cost).sum(axis=1)[sl]

    def run(pos, overnight=True):
        steps = tr.signed_session_steps(bars, roll, pos, spec.multiplier, cost, overnight)
        return steps, steps[sl].sum(axis=1)

    print(f"Trend following, 1 {symbol}, {dates[WARMUP]} .. {dates[-1]} "
          f"({dates.size - WARMUP} sessions), held through the daily break, after costs\n")
    header = (f"  {'':<24}{'net $':>9}{'max DD':>8}{'Sharpe':>8}{'long':>6}{'short':>6}"
              f"{'trades':>7}{'shifted $':>11}{'p':>7}")
    print(header)
    print(f"  {'buy & hold':<24}{bh.sum():>+9,.0f}{max_drawdown(bh):>8,.0f}{sharpe(bh):>8.2f}"
          f"{1:>6.0%}{0:>6.0%}{1:>7d}")
    positions = {}
    for name, rule in PRIMARY.items():
        pos = rule(close)
        positions[name] = pos
        _, daily = run(pos)
        observed = float(np.dot(pos[sl], gross[sl]))
        shifted, p = tr.circular_shift_pvalue(gross, pos, WARMUP, observed)
        n_trades = len([s for s in tr.position_spans(pos) if s[0] >= WARMUP])
        print(f"  {name:<24}{daily.sum():>+9,.0f}{max_drawdown(daily):>8,.0f}"
              f"{sharpe(daily):>8.2f}{np.mean(pos[sl] > 0):>6.0%}{np.mean(pos[sl] < 0):>6.0%}"
              f"{n_trades:>7d}{shifted:>+11,.0f}{p:>7.3f}")
    print("  (shifted $ / p: the same position series shifted in time, before costs; "
          "p = share of shifts earning at least as much)")

    print("\n  Net $ by year:")
    uy = np.unique(years)
    print(f"  {'':<24}" + "".join(f"{y:>9}" for y in uy))
    print(f"  {'buy & hold':<24}" + "".join(f"{bh[years == y].sum():>+9,.0f}" for y in uy))
    for name, pos in positions.items():
        _, daily = run(pos)
        print(f"  {name:<24}" + "".join(f"{daily[years == y].sum():>+9,.0f}" for y in uy))

    print("\n  Sensitivity (net $, timing p):")
    for family, variants in SENSITIVITY.items():
        cells = []
        for label, rule in variants:
            pos = rule(close)
            _, daily = run(pos)
            observed = float(np.dot(pos[sl], gross[sl]))
            _, p = tr.circular_shift_pvalue(gross, pos, WARMUP, observed)
            cells.append(f"{label}: {daily.sum():+,.0f} (p {p:.2f})")
        print(f"  {family:<16}" + "   ".join(cells))

    print(f"\n  Prop evaluation, flat through every daily break (bought at each session open, "
          f"sold at the daily close, {spec.daily_close_et} ET), {args.horizon} sessions max,\n"
          f"  bootstrap pass rate vs zero-edge baseline "
          f"($50k, $3k target, $2k EOD trailing drawdown):")
    if spec.daily_close_et < "16:00":
        print(f"  Note: {symbol}'s daily close is its {spec.daily_close_et} ET settlement, so this "
              f"version is also flat from then until the break, hours a prop account could "
              f"trade.")
    print(f"  {'':<24}" + "".join(f"{str(n) + ' ' + symbol:>18}" for n in args.contracts)
          + f"{'net $ (1 contract)':>21}")
    idx = block_bootstrap_index(dates.size - WARMUP, args.sims, args.horizon, 10, rng)
    for name, pos in positions.items():
        steps, daily = run(pos, overnight=False)
        cells = []
        for n in args.contracts:
            s = steps[sl] * n
            rates = [pf.simulate_challenge(x[idx] / ACCOUNT, RULES, compounding=False).pass_rate
                     for x in (s, demean(s, pos[sl] != 0))]
            cells.append(f"{rates[0]:>7.1%} vs {rates[1]:>5.1%}")
        print(f"  {name:<24}" + "".join(f"{c:>18}" for c in cells) + f"{daily.sum():>+21,.0f}")


if __name__ == "__main__":
    main()
