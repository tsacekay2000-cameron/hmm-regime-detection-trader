"""Trend-following signals for daily futures bars, and signed P&L steps.

All signals are causal: the position for session t+1 is decided from closes
up to t, and each function returns positions (+1 long, -1 short, 0 flat) for
sessions 1..n-1, like ``mean_reversion.rsi2_positions``. Closes should be
roll-adjusted.

- ``sma_cross_positions``: long when the fast average is above the slow one;
  flat (or short with ``allow_short``) otherwise.
- ``donchian_positions`` (Turtle-style): enter long on a close above the
  highest close of the previous ``entry`` sessions, exit on a close below
  the lowest close of the previous ``exit`` sessions; shorts mirror it.
- ``tsmom_positions`` (time-series momentum): sign of the return over the
  previous ``lookback`` sessions.

``signed_session_steps`` turns positions into per-session dollar P&L for
one contract, either held through the daily break or flat through it
(entered at each session open), as 4 intraday steps in the worst order for
the position: open -> high -> low -> close for a long, open -> low -> high
-> close for a short.
"""

from __future__ import annotations

import numpy as np

from hmm_trader.mean_reversion import sma


def sma_cross_positions(close: np.ndarray, fast: int = 50, slow: int = 200,
                        allow_short: bool = False) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    f, s = sma(close, fast), sma(close, slow)
    ok = np.isfinite(f) & np.isfinite(s)
    pos = np.where(ok & (f > s), 1, np.where(ok & (f < s) & allow_short, -1, 0))
    return pos[:-1].astype(int)


def donchian_positions(close: np.ndarray, entry: int = 55, exit: int = 20,
                       allow_short: bool = True) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    pos = np.zeros(close.size, dtype=int)  # position after the close of day t
    for t in range(close.size):
        prev = pos[t - 1] if t else 0
        cur = prev
        if t >= exit:
            lo_x, hi_x = close[t - exit:t].min(), close[t - exit:t].max()
            if prev > 0 and close[t] < lo_x:
                cur = 0
            elif prev < 0 and close[t] > hi_x:
                cur = 0
        if cur == 0 and t >= entry:
            hi_e, lo_e = close[t - entry:t].max(), close[t - entry:t].min()
            if close[t] > hi_e:
                cur = 1
            elif close[t] < lo_e and allow_short:
                cur = -1
        pos[t] = cur
    return pos[:-1]


def tsmom_positions(close: np.ndarray, lookback: int = 252,
                    allow_short: bool = True) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    pos = np.zeros(close.size, dtype=int)
    ret = close[lookback:] / close[:-lookback] - 1
    pos[lookback:] = np.where(ret > 0, 1, -1 if allow_short else 0)
    return pos[:-1]


def signed_session_steps(bars, roll: np.ndarray, pos: np.ndarray, multiplier: float,
                         cost_per_side: float, overnight: bool = True) -> np.ndarray:
    """Dollar P&L per session for ``pos`` contracts (sessions 1..n-1), shape (n-1, 4).

    Held overnight, P&L runs close to close (from the second-month close on
    roll days) and costs are charged per unit of position change plus a round
    trip for a position carried through a roll. Flat through the break, each
    held session is entered at its open and closed at its close, a round trip
    per session. Costs are charged on step 0.
    """
    pos = np.asarray(pos, dtype=float)
    b = slice(1, None)
    prev_close = np.where(roll[1:], bars.next_close[:-1], bars.close[:-1])
    o, h, lo, c = bars.open[b], bars.high[b], bars.low[b], bars.close[b]
    gap = (o - prev_close) if overnight else np.zeros_like(o)
    long_path = np.column_stack([gap, h - o, lo - h, c - lo])
    short_path = np.column_stack([gap, lo - o, h - lo, c - h])
    steps = np.where((pos > 0)[:, None], long_path, short_path) * pos[:, None] * multiplier
    if overnight:
        prev = np.r_[0.0, pos[:-1]]
        carried = (prev == pos) & (pos != 0) & roll[1:]
        sides = np.abs(pos - prev) + 2 * np.abs(pos) * carried
    else:
        sides = 2 * np.abs(pos)
    steps[:, 0] -= sides * cost_per_side
    return steps


def position_spans(pos: np.ndarray) -> list[tuple[int, int, int]]:
    """``(start, end, side)`` for runs of constant nonzero position (end exclusive)."""
    out, start = [], None
    for t in range(pos.size + 1):
        cur = pos[t] if t < pos.size else 0
        prev = pos[t - 1] if t else 0
        if cur != prev:
            if prev != 0:
                out.append((start, t, int(np.sign(prev))))
            start = t if cur != 0 else None
    return out


def circular_shift_pvalue(pnl_by_session: np.ndarray, pos: np.ndarray, active_from: int,
                          observed: float, min_shift: int = 60) -> tuple[float, float]:
    """Timing test: P&L of the same position series shifted in time.

    ``pnl_by_session`` is the gross P&L of +1 contract each session. Every
    circular shift of ``pos`` over the active window by at least
    ``min_shift`` sessions keeps exposure and trade lengths but breaks the
    timing. Returns (mean shifted P&L, share of shifts >= observed).
    """
    p = pos[active_from:].astype(float)
    g = pnl_by_session[active_from:]
    shifts = np.arange(min_shift, p.size - min_shift)
    sims = np.array([np.dot(np.roll(p, k), g) for k in shifts])
    return float(sims.mean()), float(np.mean(sims >= observed))
