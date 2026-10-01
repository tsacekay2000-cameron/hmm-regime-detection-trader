"""Two published intraday momentum rules on MES and MNQ, scored as prop evaluations.

Run from the repo root:  python -m examples.intraday_momentum [--symbols MES,MNQ]

Rules, fixed before running (see ``hmm_trader.intraday_momentum``):

- A. Last half hour (Gao, Han, Li & Zhou 2018): the return from the previous
  close to 10:00 ET sets the side, held 15:30 -> 16:00. A' is the Baltussen
  et al. (2021) signal, previous close -> 15:30.
- B. Noise boundary + VWAP (Zarattini, Aziz & Barbon 2024): 14-day noise
  bands around the open, checks every 30 minutes from 10:00 to 15:30, VWAP
  trailing exit, flat at 16:00.

MES is the primary test (the papers use SPY); MNQ is the out-of-sample
check. 1 contract, $0.62 commission + 1 tick of slippage per side, full
sessions only (early closes dropped). The flip test keeps each rule's
timing and trade lengths but makes every side a coin flip; p is the share of
flips earning at least as much before costs. A small sensitivity table shows
neighbouring settings, and the prop evaluation compares bootstrap pass rates
with the same days' P&L with the average profit removed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from examples.mes_mean_reversion import ACCOUNT, RULES, demean, max_drawdown
from examples.orb_backtest import block_bootstrap_index
from hmm_trader import futures as fu
from hmm_trader import intraday_momentum as im
from hmm_trader import prop_firm as pf

DATA_DIR = Path(__file__).parent / "data"
SPLIT = "2025-10-01"  # first year / second year


def rules(b: im.DayBars) -> dict[str, np.ndarray]:
    return {
        "A  last 30 min, 10:00 signal": im.last_half_hour_positions(b, "10:00")[0],
        "A' last 30 min, 15:30 signal": im.last_half_hour_positions(b, "15:30")[0],
        "B  noise boundary + VWAP": im.noise_boundary_positions(b),
    }


def sensitivity(b: im.DayBars) -> dict[str, list[tuple[str, np.ndarray]]]:
    return {
        "A signal ends at": [(t, im.last_half_hour_positions(b, t)[0])
                             for t in ("10:00", "10:30", "12:00", "14:00")],
        "B lookback (days)": [(str(n), im.noise_boundary_positions(b, lookback=n))
                              for n in (10, 14, 20)],
        "B checks every": [(f"{m}m", im.noise_boundary_positions(b, every=m))
                           for m in (15, 30, 60)],
    }


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbols", default="MES,MNQ")
    ap.add_argument("--contracts", default="1,3,6",
                    type=lambda t: [int(x) for x in t.split(",") if int(x) > 0])
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--horizon", type=int, default=250)
    ap.add_argument("--flips", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    for symbol in args.symbols.upper().split(","):
        spec = fu.CONTRACTS[symbol]
        b = im.load_day_bars(DATA_DIR / f"{symbol.lower()}_5min_rth.csv.gz")
        rng = np.random.default_rng(args.seed)
        first = b.date < SPLIT

        def run(pos):
            net = im.simulate_positions(b, pos, spec.multiplier, spec.tick)
            gross = im.simulate_positions(b, pos, spec.multiplier, 0.0, 0.0)
            return net, gross

        print(f"\n== {symbol}: {b.date[0]} .. {b.date[-1]}, {b.date.size} full sessions, "
              f"1 contract, after costs ==")
        i30 = im.bar_index("15:30")
        last = b.close[:, -1] / b.open[:, i30] - 1
        for end in ("10:00", "15:30"):
            _, sig = im.last_half_hour_positions(b, end)
            ok = np.isfinite(sig)
            print(f"  previous close -> {end} vs last half hour: correlation "
                  f"{np.corrcoef(sig[ok], last[ok])[0, 1]:+.3f}, same sign "
                  f"{np.mean(np.sign(sig[ok]) == np.sign(last[ok])):.0%} of {ok.sum()} days")

        print(f"\n  {'':<30}{'days':>6}{'trades':>8}{'net $':>9}{'gross $':>9}{'avg/day':>9}"
              f"{'win':>6}{'worst':>8}{'year 1':>9}{'year 2':>9}{'flip p':>8}")
        results = {}
        for name, pos in rules(b).items():
            net, gross = run(pos)
            on = net.pnl != 0
            p = im.flip_test(gross.pnl, rng, args.flips)
            results[name] = (net, on)
            print(f"  {name:<30}{on.sum():>6}{net.trades.sum():>8}{net.pnl.sum():>+9,.0f}"
                  f"{gross.pnl.sum():>+9,.0f}{net.pnl[on].mean():>+9.1f}{np.mean(net.pnl[on] > 0):>6.0%}"
                  f"{net.pnl.min():>+8,.0f}{net.pnl[first].sum():>+9,.0f}{net.pnl[~first].sum():>+9,.0f}"
                  f"{p:>8.3f}")
        always = np.zeros(b.open.shape)
        always[:, i30:] = 1
        net, _ = run(always)
        print(f"  {'(always long, last 30 min)':<30}{b.date.size:>6}{b.date.size:>8}{net.pnl.sum():>+9,.0f}")
        print(f"  (year 1: before {SPLIT}; flip p: share of random long/short choices with the "
              f"same timing earning at least as much, before costs)")

        print("\n  Sensitivity (net $, flip p):")
        for family, variants in sensitivity(b).items():
            cells = []
            for label, pos in variants:
                net, gross = run(pos)
                cells.append(f"{label}: {net.pnl.sum():+,.0f} (p {im.flip_test(gross.pnl, rng, 5000):.2f})")
            print(f"  {family:<20}" + "   ".join(cells))

        print(f"\n  Prop evaluation, {args.horizon} sessions max, bootstrap pass rate vs zero-edge "
              f"baseline ($50k, $3k target, $2k EOD trailing drawdown):")
        print(f"  {'':<30}" + "".join(f"{str(n) + ' ' + symbol:>18}" for n in args.contracts)
              + f"{'max DD (1)':>12}")
        idx = block_bootstrap_index(b.date.size, args.sims, args.horizon, 10, rng)
        for name, (net, on) in results.items():
            cells = []
            for n in args.contracts:
                s = net.steps * n
                rates = [pf.simulate_challenge(x[idx] / ACCOUNT, RULES, compounding=False).pass_rate
                         for x in (s, demean(s, on))]
                cells.append(f"{rates[0]:>7.1%} vs {rates[1]:>5.1%}")
            print(f"  {name:<30}" + "".join(f"{c:>18}" for c in cells)
                  + f"{max_drawdown(net.pnl):>12,.0f}")


if __name__ == "__main__":
    main()
