"""Intraday mean reversion on MES and MNQ: VWAP reversion, gaps, RSI(2) on 15-minute bars.

Run from the repo root:  python -m examples.intraday_mean_reversion [--symbols MES,MNQ]

Rules fixed before running (see ``hmm_trader.intraday_mr``); MES is the
primary test and MNQ the check; 1 contract, $0.62 + 1 tick per side:

- VWAP reversion: fade a close beyond VWAP +/- 2 volume-weighted standard
  deviations (10:00-15:00), exit back at VWAP, beyond 3 SD, or when the day
  breaks its noise-boundary bands (a trend day); entries only while unbroken.
- Gaps: fade small gaps (< 0.3 x the 14-day average range) at the open toward
  the previous close, stop 0.5 x range; follow large gaps (>= 0.6 x range)
  with the stop at the previous close; flat at 16:00.
- RSI(2) on 15-minute bars: long under 10 above the 200-bar average, short
  over 90 below it, out on a close back across the 5-bar average; flat at 16:00.

Each section shows neighbouring settings. The flip test keeps each rule's
timing but makes every day's side a coin flip; for gaps, each trade is
compared with its mirror image. Prop rows compare bootstrap pass rates with
the same days' P&L with the average removed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np

from examples.mes_mean_reversion import ACCOUNT, RULES, demean
from examples.orb_backtest import block_bootstrap_index
from hmm_trader import futures as fu
from hmm_trader import intraday_momentum as im
from hmm_trader import intraday_mr as mr
from hmm_trader import prop_firm as pf
from hmm_trader import smc

DATA_DIR = Path(__file__).parent / "data"
SPLIT = "2025-10-01"


def position_rules(b: im.DayBars) -> dict[str, np.ndarray]:
    return {
        "VWAP rev 2 SD + range filter": mr.vwap_reversion_positions(b),
        "  ... no range filter": mr.vwap_reversion_positions(b, range_filter=False),
        "  ... 1.5 SD": mr.vwap_reversion_positions(b, k=1.5),
        "  ... 2.5 SD": mr.vwap_reversion_positions(b, k=2.5),
        "RSI(2) 15-min long/short": mr.rsi2_intraday_positions(b),
        "  ... long only": mr.rsi2_intraday_positions(b, allow_short=False),
        "  ... 5/95": mr.rsi2_intraday_positions(b, entry=5),
        "  ... 5-min bars": mr.rsi2_intraday_positions(b, minutes=5),
    }


def prop_rates(steps: np.ndarray, active: np.ndarray, idx: np.ndarray) -> str:
    rates = [pf.simulate_challenge(x[idx] / ACCOUNT, RULES, compounding=False).pass_rate
             for x in (steps, demean(steps, active))]
    return f"{rates[0]:>6.1%} vs {rates[1]:>5.1%}"


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbols", default="MES,MNQ")
    ap.add_argument("--contracts", default="1,3",
                    type=lambda t: [int(x) for x in t.split(",") if int(x) > 0])
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--horizon", type=int, default=250)
    ap.add_argument("--flips", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--write-trades", action="store_true",
                    help="write pine/{vwap_reversion,rsi2_15m,gap_fade}_<symbol>_reference_trades.csv")
    args = ap.parse_args(argv)

    for symbol in args.symbols.upper().split(","):
        spec = fu.CONTRACTS[symbol]
        b = im.load_day_bars(DATA_DIR / f"{symbol.lower()}_5min_rth.csv.gz")
        rng = np.random.default_rng(args.seed)
        first = b.date < SPLIT
        idx = block_bootstrap_index(b.date.size, args.sims, args.horizon, 10, rng)
        if args.write_trades:
            rows = im.trade_list(b, mr.vwap_reversion_positions(b), spec.multiplier, spec.tick)
            out = Path(__file__).parents[1] / "pine" / f"vwap_reversion_{symbol.lower()}_reference_trades.csv"
            with open(out, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["date", "side", "entry_time_et", "entry_fill", "exit_time_et",
                            "exit_fill", "pnl_1_contract"])
                w.writerows(rows)
            print(f"{symbol}: {len(rows)} VWAP-reversion trades, {sum(r[6] for r in rows):+,.0f} "
                  f"-> {out.relative_to(out.parents[1])}")
            rows = im.trade_list(b, mr.rsi2_intraday_positions(b), spec.multiplier, spec.tick)
            out = Path(__file__).parents[1] / "pine" / f"rsi2_15m_{symbol.lower()}_reference_trades.csv"
            with open(out, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["date", "side", "entry_time_et", "entry_fill", "exit_time_et",
                            "exit_fill", "pnl_1_contract"])
                w.writerows(rows)
            print(f"{symbol}: {len(rows)} RSI(2) 15-minute trades, {sum(r[6] for r in rows):+,.0f} "
                  f"-> {out.relative_to(out.parents[1])}")
            rows = mr.gap_trade_rows(b, mr.gap_trades(b, spec.multiplier, spec.tick, "fade"))
            out = Path(__file__).parents[1] / "pine" / f"gap_fade_{symbol.lower()}_reference_trades.csv"
            with open(out, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["date", "side", "entry_time_et", "entry_fill", "exit_time_et",
                            "exit_fill", "exit_reason", "pnl_1_contract"])
                w.writerows(rows)
            print(f"{symbol}: {len(rows)} gap-fade trades, {sum(r[7] for r in rows):+,.0f} "
                  f"-> {out.relative_to(out.parents[1])}")
        print(f"\n== {symbol}: {b.date[0]} .. {b.date[-1]}, {b.date.size} full sessions, "
              f"1 contract, after costs ==")
        print(f"  {'':<30}{'days':>6}{'trades':>8}{'net $':>9}{'gross $':>9}{'year 1':>9}"
              f"{'year 2':>9}{'flip p':>8}" + "".join(f"{f'prop {n} ' + symbol:>18}" for n in args.contracts))
        for name, pos in position_rules(b).items():
            net = im.simulate_positions(b, pos, spec.multiplier, spec.tick)
            gross = im.simulate_positions(b, pos, spec.multiplier, 0.0, 0.0)
            on = net.pnl != 0
            props = ""
            if not name.startswith(" "):
                props = "".join(f"{prop_rates(net.steps * n, on, idx):>18}" for n in args.contracts)
            print(f"  {name:<30}{on.sum():>6}{net.trades.sum():>8}{net.pnl.sum():>+9,.0f}"
                  f"{gross.pnl.sum():>+9,.0f}{net.pnl[first].sum():>+9,.0f}{net.pnl[~first].sum():>+9,.0f}"
                  f"{im.flip_test(gross.pnl, rng, args.flips):>8.2f}{props}")

        print(f"\n  {'gaps':<30}{'trades':>7}{'win':>6}{'avg R':>8}{'± se':>7}{'net $':>9}"
              f"{'year 1':>8}{'year 2':>8}{'side p':>8}" + "".join(f"{f'prop {n} ' + symbol:>18}" for n in args.contracts))
        for mode, label in (("fade", "fade small gaps"), ("go", "follow large gaps")):
            tr = mr.gap_trades(b, spec.multiplier, spec.tick, mode)
            r = np.array([t.r for t in tr])
            pnl = np.array([t.pnl for t in tr])
            y1 = np.array([b.date[t.day] < SPLIT for t in tr], dtype=bool)
            yr = lambda m: f"{r[m].mean():>+8.3f}" if m.any() else f"{'-':>8}"
            active = np.zeros(b.date.size, dtype=bool)
            active[[t.day for t in tr]] = True
            props = "".join(
                f"{prop_rates(smc.daily_steps(b, mr.gap_trades(b, spec.multiplier, spec.tick, mode, contracts=n), spec.multiplier), active, idx):>18}"
                for n in args.contracts)
            print(f"  {label:<30}{len(tr):>7}{np.mean(pnl > 0):>6.0%}{r.mean():>+8.3f}"
                  f"{r.std(ddof=1) / np.sqrt(r.size):>7.3f}{pnl.sum():>+9,.0f}{yr(y1)}{yr(~y1)}"
                  f"{smc.mirror_pvalue(tr, rng, 5000):>8.2f}{props}")
        print(f"  (year 1: before {SPLIT}; prop: bootstrap pass rate vs zero-edge baseline, $50k, "
              f"$3k target, $2k EOD trailing drawdown, {args.horizon} sessions)")


if __name__ == "__main__":
    main()
