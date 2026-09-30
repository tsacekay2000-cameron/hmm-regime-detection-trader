"""RSI(2) pullback on MES plus MNQ or MYM: which plan to trade in one prop account.

Run from the repo root:  python -m examples.rsi2_combined [--second MNQ,MYM]

Both symbols use the same rules (RSI(2) < 10 above the 200-day average, exit
on a close above the 5-day average) in the prop version, flat through every
daily break. The plans match ``pine/rsi2_combined_alerts.pine``: 2 MES only,
2 MES and the second market (1 MNQ or 3 MYM) on sessions when MES has no
signal, or both together (1 MES + 1 MNQ, or 1 MES + 2 MYM). Each is scored on
the $50k evaluation against its zero-edge baseline.
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


def session_steps(symbol: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dates, RSI(2) long flags and flat-through-breaks steps for 1 contract, every session."""
    spec = fu.CONTRACTS[symbol]
    bars = fu.load_bars(DATA_DIR / f"{symbol.lower()}_daily.csv")
    roll = fu.roll_mask(bars.dates, spec)
    adj_close = bars.close[0] * np.r_[1.0, np.cumprod(1 + fu.adjusted_returns(bars, roll))]
    held = mr.rsi2_positions(adj_close).astype(int)
    steps = tr.signed_session_steps(bars, roll, np.ones_like(held), spec.multiplier,
                                    0.62 + spec.tick_value, overnight=False)
    return bars.dates[1:], held, steps


# second-market contracts: alone (MES flat), together with 1 MES, and alongside 2 MES
SIZES = {"MNQ": (1, 1, 1), "MYM": (3, 2, 3)}


def plans(mes: np.ndarray, other: np.ndarray,
          symbol: str = "MNQ") -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Contracts of (MES, second market) per session for each plan."""
    alone, together, full = SIZES[symbol]
    return {
        "2 MES only": (2 * mes, 0 * other),
        f"2 MES, else {alone} {symbol}": (2 * mes, alone * other * (1 - mes)),
        f"1 MES + {together} {symbol}": (mes, together * other),
        f"2 MES + {full} {symbol}": (2 * mes, full * other),
    }


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--second", default="MNQ,MYM")
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--horizon", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    dates, h_mes, s_mes = session_steps("MES")
    for symbol in args.second.upper().split(","):
        dates_b, h_b, s_b = session_steps(symbol)
        if not np.array_equal(dates, dates_b):
            raise ValueError(f"MES and {symbol} data must cover the same sessions")
        start = int(np.flatnonzero(h_mes | h_b)[0])
        both = int((h_mes & h_b)[start:].sum())
        alone = (h_b == 1) & (h_mes == 0)
        idx = block_bootstrap_index(dates.size - start, args.sims, args.horizon, 10,
                                    np.random.default_rng(args.seed))

        print(f"RSI(2) on MES and {symbol}, flat through every daily break, "
              f"{dates[start]} .. {dates[-1]}")
        print(f"  sessions in a trade: MES {h_mes[start:].sum()}, {symbol} {h_b[start:].sum()}, "
              f"both {both}; {symbol} alone {alone[start:].sum()} "
              f"({(s_b.sum(axis=1) * alone)[start:].sum():+,.0f} per {symbol} contract)\n")
        print(f"  {'plan':<20}{'net $':>9}{'worst day':>11}{'max DD':>9}{'pass':>8}{'no edge':>9}")
        for name, (a, b) in plans(h_mes, h_b, symbol).items():
            steps = (a[:, None] * s_mes + b[:, None] * s_b)[start:]
            active = (a + b)[start:] > 0
            day = steps.sum(axis=1)
            rates = [pf.simulate_challenge(x[idx] / ACCOUNT, RULES, compounding=False).pass_rate
                     for x in (steps, demean(steps, active))]
            print(f"  {name:<20}{day.sum():>+9,.0f}{day.min():>+11,.0f}{max_drawdown(day):>9,.0f}"
                  f"{rates[0]:>8.1%}{rates[1]:>9.1%}")
        print()
    print(f"  Pass rates: $50k account, $3k target, $2k end-of-day trailing drawdown, "
          f"{args.horizon} sessions, bootstrap of 10-session blocks.")


if __name__ == "__main__":
    main()
