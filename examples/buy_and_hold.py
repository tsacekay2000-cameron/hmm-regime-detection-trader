"""Buy & hold vs the RSI(2) pullback on MES, as prop evaluations.

Run from the repo root:  python -m examples.buy_and_hold

Buy & hold made the most money in 2020-2026 because the S&P 500 roughly
doubled, but an evaluation asks for +$3,000 before a $2,000 trailing
drawdown. This compares, from the first RSI(2) trade (2020-02-25) on:

- buy & hold 1 MES, held (rolled each quarter) or flat through every daily
  break (bought at each 6 PM ET open, sold at the daily close);
- RSI(2) on 2 MES, held overnight (bought at the next open after the signal,
  as ``pine/rsi2_pullback.pine`` does) or flat through every daily break.

Each is scored by replaying an evaluation from every historical start day,
by a block bootstrap, and against the same trades with the edge removed.
Pass rates by start year show how much buy & hold depends on when you start.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from examples.mes_mean_reversion import ACCOUNT, RULES, demean, max_drawdown
from examples.orb_backtest import block_bootstrap_index
from hmm_trader import futures as fu
from hmm_trader import mean_reversion as mr
from hmm_trader import prop_firm as pf
from hmm_trader import trend as tr

DATA_DIR = Path(__file__).parent / "data"


def strategies(symbol: str = "MES") -> tuple[np.ndarray, int, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Session dates, first active session and {name: (steps, active)} per strategy."""
    spec = fu.CONTRACTS[symbol]
    cost = 0.62 + spec.tick_value  # commission + 1 tick of slippage per side
    bars = fu.load_bars(DATA_DIR / f"{symbol.lower()}_daily.csv")
    roll = fu.roll_mask(bars.dates, spec)
    adj_close = bars.close[0] * np.r_[1.0, np.cumprod(1 + fu.adjusted_returns(bars, roll))]
    held = mr.rsi2_positions(adj_close).astype(int)
    ones = np.ones_like(held)

    rsi_hold = tr.signed_session_steps(bars, roll, held, spec.multiplier, cost, overnight=True)
    first = (held == 1) & (np.r_[0, held[:-1]] == 0)
    rsi_hold[first, 0] = -cost  # bought at the next open: entry cost, no overnight gap
    rsi_flat = tr.signed_session_steps(bars, roll, held, spec.multiplier, cost, overnight=False)
    bh_hold = tr.signed_session_steps(bars, roll, ones, spec.multiplier, cost, overnight=True)
    bh_flat = tr.signed_session_steps(bars, roll, ones, spec.multiplier, cost, overnight=False)

    start = int(np.flatnonzero(held)[0])
    bh_hold[start, 0] = -cost  # bought at the first session's open
    everyday, in_trade = ones == 1, held == 1
    return bars.dates[1:], start, {
        f"Buy & hold, 1 {symbol}": (bh_hold, everyday),
        f"Buy & hold, 1 {symbol}, flat each break": (bh_flat, everyday),
        f"RSI(2), 2 {symbol}, held overnight": (2 * rsi_hold, in_trade),
        f"RSI(2), 2 {symbol}, flat each break": (2 * rsi_flat, in_trade),
    }


def every_start(steps: np.ndarray, horizon: int) -> pf.ChallengeResult:
    """One evaluation from each historical start day (windows of ``horizon`` sessions)."""
    frac = steps / ACCOUNT
    win = np.lib.stride_tricks.sliding_window_view(frac, horizon, axis=0)
    return pf.simulate_challenge(win.transpose(0, 2, 1), RULES, compounding=False)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbol", default="MES")
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--horizon", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    dates, start, strats = strategies(args.symbol.upper())
    n = dates.size - start
    idx = block_bootstrap_index(n, args.sims, args.horizon, 10, np.random.default_rng(args.seed))
    start_years = np.array([d[:4] for d in dates[start:start + n - args.horizon + 1]])
    years = np.array([d[:4] for d in dates[start:]])

    print(f"Buy & hold vs RSI(2), {dates[start]} .. {dates[-1]}, after costs")
    print(f"Evaluation: $50k, +$3,000 target, $2,000 end-of-day trailing drawdown, "
          f"{args.horizon} sessions\n")
    print(f"  {'':<34}{'net $':>9}{'worst day':>11}{'max DD':>8}{'every start':>13}"
          f"{'bootstrap':>11}{'no edge':>9}{'failed':>8}{'days':>6}")
    by_start, by_year = {}, {}
    for name, (steps, active) in strats.items():
        s, a = steps[start:], active[start:]
        day = s.sum(axis=1)
        hist = every_start(s, args.horizon)
        boot = pf.simulate_challenge(s[idx] / ACCOUNT, RULES, compounding=False)
        null = pf.simulate_challenge(demean(s, a)[idx] / ACCOUNT, RULES, compounding=False)
        print(f"  {name:<34}{day.sum():>+9,.0f}{day.min():>+11,.0f}{max_drawdown(day):>8,.0f}"
              f"{hist.pass_rate:>13.1%}{boot.pass_rate:>11.1%}{null.pass_rate:>9.1%}"
              f"{boot.rate(pf.FAILED_DRAWDOWN):>8.0%}{boot.days_to_pass()[50]:>6.0f}")
        passed = hist.outcomes == pf.PASSED
        by_start[name] = {y: passed[start_years == y].mean() for y in np.unique(start_years)}
        by_year[name] = {y: day[years == y].sum() for y in np.unique(years)}
    print("  (every start / bootstrap / no edge: pass rates; failed: bootstrap share that hit the "
          "drawdown; days: median sessions to pass)")

    uy = list(next(iter(by_start.values())))
    print("\n  Pass rate by the year the evaluation started (every start):")
    print(f"  {'':<34}" + "".join(f"{y:>8}" for y in uy))
    for name, row in by_start.items():
        print(f"  {name:<34}" + "".join(f"{row[y]:>8.0%}" for y in uy))

    uy = list(next(iter(by_year.values())))
    print("\n  Net $ by calendar year:")
    print(f"  {'':<34}" + "".join(f"{y:>8}" for y in uy))
    for name, row in by_year.items():
        print(f"  {name:<34}" + "".join(f"{row[y]:>+8,.0f}" for y in uy))


if __name__ == "__main__":
    main()
