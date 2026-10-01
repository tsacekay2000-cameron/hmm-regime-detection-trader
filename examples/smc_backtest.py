"""Liquidity sweeps, fair value gaps, order blocks and a volume proxy for order flow.

Run from the repo root:  python -m examples.smc_backtest [--symbols MES,MNQ]

"Smart money" setups made mechanical (see ``hmm_trader.smc``) on 5-minute
regular-hours bars, 2024-10 to 2026-09. The full model, fixed before
running: a sweep of the previous session's high/low or a swing point, then
within 30 minutes a fair value gap in the other direction whose displacement
bar had >= 1.5x normal volume, then a limit order at the gap's edge (filled
only if price trades through it), stop beyond the sweep, 2R target, flat at
16:00, $200 risk per trade. The other rows drop or swap one piece to show
which part matters.

The side test enters the same trade, and its mirror image, at the next bar's
open with the same stop and target distances; p is the share of random
long/short choices doing at least as well. With 7 variants on 2 symbols,
one p near 0.05 is what chance alone gives. The prop evaluation compares
bootstrap pass rates with the same days' P&L with the average removed.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

from examples.mes_mean_reversion import ACCOUNT, RULES, demean
from examples.orb_backtest import block_bootstrap_index
from hmm_trader import futures as fu
from hmm_trader import intraday_momentum as im
from hmm_trader import prop_firm as pf
from hmm_trader import smc

DATA_DIR = Path(__file__).parent / "data"
SPLIT = "2025-10-01"


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbols", default="MES,MNQ")
    ap.add_argument("--risk", default="200,400",
                    type=lambda t: [float(x) for x in t.split(",") if float(x) > 0])
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--horizon", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    for symbol in args.symbols.upper().split(","):
        spec = fu.CONTRACTS[symbol]
        b = im.load_day_bars(DATA_DIR / f"{symbol.lower()}_5min_rth.csv.gz")
        rng = np.random.default_rng(args.seed)
        print(f"\n== {symbol}: {b.date[0]} .. {b.date[-1]}, {b.date.size} full sessions, "
              f"$200 risk per trade, after costs ==")
        print(f"  {'':<24}{'trades':>7}{'win':>6}{'avg R':>8}{'± se':>7}{'net $':>9}"
              f"{'year 1':>8}{'year 2':>8}{'side p':>8}   exits (target/stop/16:00)")
        idx = block_bootstrap_index(b.date.size, args.sims, args.horizon, 10, rng)
        prop_rows = []
        for name, p in smc.variants().items():
            tr = smc.backtest(b, spec.multiplier, spec.tick, p)
            r = np.array([t.r for t in tr])
            pnl = np.array([t.pnl for t in tr])
            y1 = np.array([b.date[t.day] < SPLIT for t in tr])
            ex = [sum(t.reason == k for t in tr) for k in ("target", "stop", "close")]
            print(f"  {name:<24}{len(tr):>7}{np.mean(pnl > 0):>6.0%}{r.mean():>+8.3f}"
                  f"{r.std(ddof=1) / np.sqrt(r.size):>7.3f}{pnl.sum():>+9,.0f}{r[y1].mean():>+8.3f}"
                  f"{r[~y1].mean():>+8.3f}{smc.mirror_pvalue(tr, rng, 5000):>8.2f}   "
                  f"{ex[0]}/{ex[1]}/{ex[2]}")
            cells = []
            for risk in args.risk:
                t2 = smc.backtest(b, spec.multiplier, spec.tick, replace(p, risk_dollars=risk))
                steps = smc.daily_steps(b, t2, spec.multiplier)
                active = np.zeros(b.date.size, dtype=bool)
                active[[t.day for t in t2]] = True
                rates = [pf.simulate_challenge(x[idx] / ACCOUNT, RULES, compounding=False).pass_rate
                         for x in (steps, demean(steps, active))]
                cells.append(f"{rates[0]:>6.1%} vs {rates[1]:>5.1%}")
            prop_rows.append((name, cells))
        print(f"  (avg R after costs; year 1 is before {SPLIT}; side p: random long/short "
              f"choices from the next open doing at least as well)")
        print(f"\n  Prop evaluation, {args.horizon} sessions max, bootstrap pass rate vs zero-edge "
              f"baseline ($50k, $3k target, $2k EOD trailing drawdown):")
        print(f"  {'':<24}" + "".join(f"{'$' + format(r, ',.0f') + ' risk':>18}" for r in args.risk))
        for name, cells in prop_rows:
            print(f"  {name:<24}" + "".join(f"{c:>18}" for c in cells))


if __name__ == "__main__":
    main()
