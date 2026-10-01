"""Mechanical "smart money" setups on 5-minute RTH bars: liquidity sweeps,
fair value gaps, order blocks (supply/demand zones) and a volume proxy for
order flow.

These ideas are usually traded by eye; here each one is a fixed rule so it
can be tested:

- Liquidity levels: the previous session's high and low (skipped on roll
  days, when that session was another contract) and today's swing highs and
  lows (a high above the 2 bars on each side, known 2 bars later).
- Sweep: a bar trades beyond an untouched level and closes back inside it
  (below a low and closes above it, for a bullish sweep). The level is then
  used up.
- Fair value gap (bullish): ``low[k] > high[k-2]``; the gap is
  ``(high[k-2], low[k])`` and bar ``k-1`` is the displacement bar.
- Order block / demand zone: the last down bar before the displacement bar.
- Order flow proxy: the displacement bar's volume divided by the average
  volume at that time of day over the previous 20 sessions (there is no
  bid/ask data, so no true delta).

Setup (long; short mirrors it): a sweep, then within ``fvg_within`` bars a
bullish FVG (optionally with relative volume >= ``min_rvol``); a limit buy at
the top of the gap (or at the order block's high) waits ``fill_within``
bars. The stop is one tick below the lowest low since the sweep, the target
``target_r`` times the risk; flat at the 16:00 close. Size is a fixed dollar
risk. Entries from ``first_entry`` to ``last_entry``; one position at a
time, at most one long and one short per day.

Fills are conservative: the limit fills at its price only when a bar trades
through it by a tick (a touch is not enough); on the fill bar only the stop is checked; afterwards a bar
touching both stop and target counts as a stop; stops and the close pay one
tick of slippage, and every side pays commission.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from hmm_trader.intraday_momentum import BARS_PER_DAY, DayBars, bar_index


@dataclass(frozen=True)
class SMCParams:
    entry: str = "fvg"            # "fvg": top of the gap; "ob": order block high; "sweep": sweep bar close
    need_sweep: bool = True
    min_rvol: Optional[float] = 1.5
    target_r: float = 2.0
    fvg_within: int = 6           # bars after the sweep for the gap to form
    fill_within: int = 12         # bars after the gap for the limit to fill
    swing: int = 2                # bars on each side of a swing point
    rvol_days: int = 20
    first_entry: str = "09:45"
    last_entry: str = "15:00"
    risk_dollars: float = 200.0
    max_contracts: int = 20

    @property
    def label(self) -> str:
        parts = ["sweep" if self.need_sweep else "no sweep",
                 {"fvg": "FVG", "ob": "order block", "sweep": "enter on sweep"}[self.entry]]
        if self.min_rvol and self.entry != "sweep":
            parts.append(f"vol >= {self.min_rvol:g}x")
        return " + ".join(parts) + f", {self.target_r:g}R"


@dataclass
class Trade:
    day: int
    side: int          # +1 long, -1 short
    fill_bar: int
    entry: float
    stop: float
    target: float
    contracts: int
    exit_bar: int
    exit_price: float
    reason: str        # "target" / "stop" / "close"
    pnl: float         # dollars after costs
    r: float           # pnl / risk dollars
    r_cmp: float       # R judged from the bar after the fill (for the side test)
    mirror_r: float    # the same, taken the other way


def relative_volume(b: DayBars, days: int = 20) -> np.ndarray:
    """Volume over the mean volume at the same time of day in the previous ``days`` sessions."""
    out = np.full(b.volume.shape, np.nan)
    for d in range(days, b.volume.shape[0]):
        avg = b.volume[d - days:d].mean(axis=0)
        out[d] = b.volume[d] / np.where(avg > 0, avg, np.nan)
    return out


def _prev_day_levels(b: DayBars) -> tuple[np.ndarray, np.ndarray]:
    """Previous session high/low, NaN on the first day and on roll days."""
    hi = np.r_[np.nan, b.high[:-1].max(axis=1)]
    lo = np.r_[np.nan, b.low[:-1].min(axis=1)]
    last = np.r_[np.nan, b.close[:-1, -1]]
    same_contract = np.isclose(last, b.prev_close)
    return np.where(same_contract, hi, np.nan), np.where(same_contract, lo, np.nan)


def _manage(b: DayBars, d: int, side: int, fill: int, entry: float, stop: float,
            target: float, tick: float, from_next: bool = False) -> tuple[int, float, str]:
    """Exit bar, exit price and reason for a position filled during bar ``fill``.

    ``from_next`` ignores the fill bar's range (used for the side test, where
    the fill bar's path was shaped by the entry and would favour one side).
    """
    hi, lo = b.high[d], b.low[d]
    for i in range(fill + 1 if from_next else fill, BARS_PER_DAY):
        hit_stop = lo[i] <= stop if side > 0 else hi[i] >= stop
        if hit_stop:
            gap_through = (b.open[d, i] < stop) if side > 0 else (b.open[d, i] > stop)
            px = (b.open[d, i] if gap_through and i > fill else stop) - side * tick
            return i, px, "stop"
        if i > fill and (hi[i] >= target if side > 0 else lo[i] <= target):
            gap_through = (b.open[d, i] > target) if side > 0 else (b.open[d, i] < target)
            return i, (b.open[d, i] if gap_through else target), "target"
    return BARS_PER_DAY - 1, b.close[d, -1] - side * tick, "close"


def _setups(b: DayBars, d: int, side: int, p: SMCParams, rvol: np.ndarray,
            pd_level: float, tick: float = 0.0) -> list[tuple[int, float, float]]:
    """Candidate (fill bar, entry, stop) for one side on day ``d``, in time order."""
    o, h, l, c = b.open[d], b.high[d], b.low[d], b.close[d]
    if side < 0:  # mirror prices so the long logic serves both sides
        o, h, l, c = -o, -l, -h, -c
        pd_level = -pd_level if np.isfinite(pd_level) else pd_level
    first, last = bar_index(p.first_entry), bar_index(p.last_entry)
    levels = [pd_level] if np.isfinite(pd_level) else []  # untouched lows below price
    out = []
    i = 0
    while i < BARS_PER_DAY:
        # swing low confirmed at bar i (formed at j = i - swing)
        j = i - p.swing
        if j >= p.swing and all(l[j] < l[j - k] and l[j] < l[j + k] for k in range(1, p.swing + 1)):
            levels.append(l[j])
        swept = None
        if p.need_sweep:
            for lv in levels:
                if l[i] < lv < c[i]:
                    swept = lv
            levels = [lv for lv in levels if l[i] >= lv]  # any level traded through is used up
        else:
            swept = np.nan
        if swept is None:
            i += 1
            continue
        if p.entry == "sweep":
            if first <= i + 1 <= last and i + 1 < BARS_PER_DAY:
                out.append((i + 1, o[i + 1], l[i]))  # buy the next bar's open
            i += 1
            continue
        # look for a bullish FVG after the sweep (or anywhere, without a sweep)
        start = i + 2 if p.need_sweep else i
        stop_floor = l[i] if p.need_sweep else None
        found = False
        for k in range(max(start, 2), min(i + p.fvg_within + 2, BARS_PER_DAY) if p.need_sweep else min(i + 1, BARS_PER_DAY)):
            if not (l[k] > h[k - 2] and c[k - 1] > o[k - 1]):
                continue
            if p.min_rvol and not (rvol[d, k - 1] >= p.min_rvol):
                continue
            if p.entry == "ob":
                ob = [m for m in range(k - 2, (i if p.need_sweep else max(k - 6, 0)) - 1, -1) if c[m] < o[m]]
                if not ob:
                    continue
                level = h[ob[0]]
            else:
                level = l[k]
            low_since = min(l[(i if p.need_sweep else k - 2):k + 1])
            stop = low_since if stop_floor is None else min(stop_floor, low_since)
            for f in range(k + 1, min(k + 1 + p.fill_within, BARS_PER_DAY)):
                if f > last:
                    break
                if l[f] < stop:
                    break  # the stop went first: the setup failed before filling
                if f >= first and l[f] <= level - tick:  # traded through, not just touched
                    out.append((f, level, stop))
                    found = True
                    break
            if found:
                break
        i = (k + 1) if found else i + 1
    if side < 0:
        out = [(f, -e, -s) for f, e, s in out]
    return out


def backtest(b: DayBars, multiplier: float, tick: float, p: SMCParams = SMCParams(),
             commission: float = 0.62) -> list[Trade]:
    rvol = relative_volume(b, p.rvol_days)
    pdh, pdl = _prev_day_levels(b)
    trades = []
    for d in range(b.open.shape[0]):
        if p.min_rvol and d < p.rvol_days and p.entry != "sweep":
            continue
        cands = [(f, +1, e, s) for f, e, s in _setups(b, d, +1, p, rvol, pdl[d], tick)]
        cands += [(f, -1, e, s) for f, e, s in _setups(b, d, -1, p, rvol, pdh[d], tick)]
        cands.sort()
        busy_until, used = -1, set()
        for f, side, entry, stop in cands:
            if f <= busy_until or side in used:
                continue
            if p.entry == "sweep":
                entry += side * tick  # a market order pays a tick of slippage
            risk_pts = (entry - stop) * side + tick  # stop pays a tick of slippage
            if risk_pts <= tick:
                continue
            n = int(min(p.max_contracts, p.risk_dollars // (risk_pts * multiplier)))
            if n < 1:
                continue
            target = entry + side * p.target_r * (risk_pts - tick)
            xb, xp, why = _manage(b, d, side, f, entry, stop, target, tick)
            pnl = side * (xp - entry) * multiplier * n - 2 * n * commission
            # side test: this side and the other way, entered at the next bar's
            # open with the same stop and target distances, so the fill bar's
            # shape (price came into a limit order) favours neither side
            risk_dollars = risk_pts * multiplier * n
            cmp = []
            nxt = min(f + 1, BARS_PER_DAY - 1)
            e2 = b.open[d, nxt] if f + 1 < BARS_PER_DAY else b.close[d, -1]
            for s_ in (side, -side):
                st = e2 - s_ * (risk_pts - tick)
                tg = e2 + s_ * p.target_r * (risk_pts - tick)
                _, px, _ = _manage(b, d, s_, nxt, e2, st, tg, tick)
                cmp.append((s_ * (px - e2) * multiplier * n - 2 * n * commission) / risk_dollars)
            trades.append(Trade(d, side, f, entry, stop, target, n, xb, xp, why, pnl,
                                pnl / risk_dollars, cmp[0], cmp[1]))
            busy_until, _ = xb, used.add(side)
    return trades


def daily_steps(b: DayBars, trades: list[Trade], multiplier: float) -> np.ndarray:
    """(n_days, 3) dollars: to peak, to trough, to final, from each day's trades in order.

    The trough is each trade's worst price while open (capped at its stop),
    the peak its best (capped at its target).
    """
    out = np.zeros((b.open.shape[0], 3))
    by_day: dict[int, list[Trade]] = {}
    for t in trades:
        by_day.setdefault(t.day, []).append(t)
    for d, day_trades in by_day.items():
        eq = peak = trough = 0.0
        for t in day_trades:
            seg = slice(t.fill_bar, t.exit_bar + 1)
            size = multiplier * t.contracts
            worst = b.low[d, seg].min() if t.side > 0 else b.high[d, seg].max()
            best = b.high[d, seg].max() if t.side > 0 else b.low[d, seg].min()
            adverse = max(t.side * (worst - t.entry), t.side * (t.stop - t.entry)) * size
            favour = min(t.side * (best - t.entry), t.side * (t.target - t.entry)) * size
            trough = min(trough, eq + min(adverse, 0.0))
            peak = max(peak, eq + max(favour, 0.0))
            eq += t.pnl
            peak, trough = max(peak, eq), min(trough, eq)
        out[d] = (peak, trough - peak, eq - trough)
    return out


def variants() -> dict[str, SMCParams]:
    """The full model first, then versions that drop or swap one piece."""
    base = SMCParams()
    return {
        "full": base,
        "no volume filter": replace(base, min_rvol=None),
        "FVG only (no sweep)": replace(base, need_sweep=False),
        "order block entry": replace(base, entry="ob"),
        "sweep only": replace(base, entry="sweep", min_rvol=None),
        "full, 1R target": replace(base, target_r=1.0),
        "full, 3R target": replace(base, target_r=3.0),
    }


def mirror_pvalue(trades: list[Trade], rng: np.random.Generator, n: int = 20000) -> float:
    """Share of random long/short choices (each trade's own or mirrored R) doing at least as well."""
    if not trades:
        return float("nan")
    own = np.array([t.r_cmp for t in trades])
    mir = np.array([t.mirror_r for t in trades])
    pick = rng.random((n, own.size)) < 0.5
    return float(np.mean(np.where(pick, own, mir).sum(axis=1) >= own.sum()))
