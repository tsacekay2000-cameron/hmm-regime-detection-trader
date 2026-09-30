"""RSI(2) pullback trades on MES, MNQ, M2K, MYM, MGC and MCL as reference lists for the Pine Script port.

Run from the repo root:  python -m examples.rsi2_trade_list [--symbols MES,MNQ,M2K,MYM,MGC,MCL]

Writes ``pine/rsi2_<symbol>_reference_trades.csv``: one row per trade with the
signal close, the first and last session held, and the flat-through-breaks
(prop) P&L for 1 contract after costs (commission $0.62 + 1 tick per side,
a round trip every session), which is what ``pine/rsi2_pullback.pine``
simulates in its default mode. Compare it with TradingView's List of trades
for ``MES1!`` / ``MNQ1!`` / ``M2K1!`` / ``MYM1!`` / ``MGC1!`` / ``MCL1!`` on a daily chart with back-adjustment on; the Pine strategy
counts each session as its own trade, so group its rows by entry date.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from hmm_trader import futures as fu
from hmm_trader import mean_reversion as mr

ROOT = Path(__file__).parents[1]


def reference_trades(symbol: str = "MES") -> list[tuple[str, str, str, int, float]]:
    """(signal close, first session, last session, sessions, prop P&L for 1 contract)."""
    spec = fu.CONTRACTS[symbol]
    bars = fu.load_bars(ROOT / "examples" / "data" / f"{symbol.lower()}_daily.csv")
    roll = fu.roll_mask(bars.dates, spec)
    adj_close = bars.close[0] * np.r_[1.0, np.cumprod(1 + fu.adjusted_returns(bars, roll))]
    held = mr.rsi2_positions(adj_close)  # sessions 1..n-1
    b = bars[1:]
    cost = 0.62 + spec.tick_value
    session_pnl = (b.close - b.open) * spec.multiplier - 2 * cost
    return [(str(bars.dates[s]), str(b.dates[s]), str(b.dates[e - 1]), e - s,
             round(float(session_pnl[s:e].sum()), 2)) for s, e in mr.trade_spans(held)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbols", default="MES,MNQ,M2K,MYM,MGC,MCL")
    args = ap.parse_args(argv)
    for symbol in args.symbols.upper().split(","):
        rows = reference_trades(symbol)
        out = ROOT / "pine" / f"rsi2_{symbol.lower()}_reference_trades.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["signal_close", "first_session", "last_session", "sessions",
                        "prop_pnl_1_contract"])
            w.writerows(rows)
        total = sum(r[4] for r in rows)
        print(f"{symbol}: {len(rows)} trades, {sum(r[3] for r in rows)} sessions held, "
              f"prop P&L {total:+,.0f} per contract -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
