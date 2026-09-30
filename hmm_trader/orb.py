"""Opening range breakout (ORB) backtest on intraday futures bars.

Rules, one trade per day at most:

- The opening range is the high and low of the first ``range_minutes`` of the
  regular session (09:30 ET for CME equity index futures).
- A buy stop sits ``entry_ticks`` above the range and a sell stop the same
  distance below it; the first one touched is the trade. If a single bar
  touches both, the day is skipped (the order is unknowable from bars).
- The protective stop is the opposite entry level, so the risk is the range
  width plus two offsets. Size is ``floor(risk_dollars / risk per contract)``,
  capped at ``max_contracts``; a day whose range is too wide for one contract
  is skipped.
- Exit at ``target_r`` times the risk, at the stop, or at the close of the
  last bar starting before ``flatten`` (default 15:55 ET), whichever is first.
- Optional narrow-range filter: with ``max_range_ratio`` set, trade only when
  today's range is at most that multiple of the median range of the previous
  ``range_lookback`` sessions (causal; the first ``range_lookback`` sessions
  are warm-up and not traded).

Fills are conservative: stop entries and stop/flatten exits pay
``slippage_ticks``, a bar that touches both stop and target counts as a stop,
and a bar that opens beyond a level fills at its open. Targets are limit
orders with no slippage.

Each traded day is also summarized as three intraday steps for the prop firm
simulator: best open profit, worst open loss, final P&L, in that order (peak
before trough is the worst case for a trailing drawdown).
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from hmm_trader.futures import ContractSpec


@dataclass(frozen=True)
class ORBParams:
    range_minutes: int = 15
    target_r: Optional[float] = 2.0  # None: hold until the flatten time
    risk_dollars: float = 200.0
    max_contracts: int = 50
    entry_ticks: int = 1
    slippage_ticks: float = 1.0
    commission_per_side: float = 0.62
    session_open: int = 9 * 60 + 30  # minutes after midnight, exchange-local ET
    flatten: int = 15 * 60 + 55
    bar_minutes: int = 5
    direction: str = "both"  # "both", "long" or "short"
    max_range_ratio: Optional[float] = None  # narrow-range filter, off by default
    range_lookback: int = 20

    def __post_init__(self) -> None:
        if self.range_minutes <= 0 or self.range_minutes % self.bar_minutes:
            raise ValueError("range_minutes must be a positive multiple of bar_minutes")
        if self.target_r is not None and self.target_r <= 0:
            raise ValueError("target_r must be positive or None")
        if self.risk_dollars <= 0 or self.max_contracts < 1:
            raise ValueError("risk_dollars and max_contracts must be positive")
        if self.direction not in ("both", "long", "short"):
            raise ValueError("direction must be 'both', 'long' or 'short'")
        if self.max_range_ratio is not None and self.max_range_ratio <= 0:
            raise ValueError("max_range_ratio must be positive or None")
        if self.range_lookback < 1:
            raise ValueError("range_lookback must be >= 1")

    def label(self) -> str:
        target = f"{self.target_r:g}R" if self.target_r is not None else "EOD"
        side = "" if self.direction == "both" else f" {self.direction}"
        nr = ("" if self.max_range_ratio is None
              else f" NR<={self.max_range_ratio:g}x{self.range_lookback}d")
        return f"ORB {self.range_minutes}m / {target}{side}{nr}"


@dataclass
class IntradayBars:
    """Bars sorted by time. ``minute`` is the bar start in minutes after midnight ET."""

    date: np.ndarray    # "YYYY-MM-DD"
    minute: np.ndarray  # int
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray

    def days(self):
        """Yield ``(date, slice)`` per trading date."""
        starts = np.flatnonzero(np.r_[True, self.date[1:] != self.date[:-1]])
        ends = np.r_[starts[1:], self.date.size]
        for s, e in zip(starts, ends):
            yield self.date[s], slice(s, e)


def load_intraday(path: Path | str) -> IntradayBars:
    """Read ``datetime,open,high,low,close[,...]`` with ``datetime`` as
    ``YYYY-MM-DD HH:MM`` in exchange-local Eastern time (``.gz`` allowed)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        table = np.genfromtxt(fh, delimiter=",", names=True, dtype=None, encoding="utf-8")
    stamp = np.asarray(table["datetime"], dtype=str)
    date = np.array([s[:10] for s in stamp])
    minute = np.array([int(s[11:13]) * 60 + int(s[14:16]) for s in stamp])
    order = np.lexsort((minute, date))
    if np.any((date[order][1:] == date[order][:-1]) & (np.diff(minute[order]) == 0)):
        raise ValueError(f"duplicate bars in {path}")
    cols = [np.asarray(table[c], dtype=float)[order] for c in ("open", "high", "low", "close")]
    return IntradayBars(date[order], minute[order], *cols)


@dataclass
class ORBTrades:
    """One row per session with a complete opening range."""

    date: np.ndarray
    side: np.ndarray        # +1 long, -1 short, 0 no trade
    contracts: np.ndarray
    entry: np.ndarray
    exit: np.ndarray
    risk_points: np.ndarray
    range_points: np.ndarray  # opening range width, high - low
    pnl: np.ndarray         # dollars after costs
    reason: np.ndarray      # target / stop / flatten / none / ambiguous / too_wide /
                            # filtered / warmup
    steps: np.ndarray       # (n_days, 3) dollars: to peak, to trough, to final
    multiplier: float       # dollars per point

    @property
    def traded(self) -> np.ndarray:
        return self.side != 0

    @property
    def r_multiple(self) -> np.ndarray:
        """P&L in units of the dollars risked, for traded days."""
        t = self.traded
        return self.pnl[t] / (self.risk_points[t] * self.contracts[t] * self.multiplier)


def _bar_path(side, entry, stop, target, o, h, lo, c, slip, tick):
    """Walk bars after the entry fill; return (exit price, reason, mtm extremes).

    ``o, h, lo, c`` include the entry bar first. MTM values are in points per
    contract (positive = profit), before costs.
    """
    best = worst = 0.0
    for i in range(o.size):
        fav = h[i] if side > 0 else lo[i]
        adv = lo[i] if side > 0 else h[i]
        hit_stop = (adv <= stop) if side > 0 else (adv >= stop)
        hit_target = target is not None and ((fav >= target) if side > 0 else (fav <= target))
        if hit_stop:
            # gapped through: fill at the open; otherwise at the stop, plus slippage
            beyond = (o[i] < stop) if side > 0 else (o[i] > stop)
            px = (o[i] if beyond and i > 0 else stop) - side * slip * tick
            cap = target if target is not None else fav
            best = max(best, side * ((min(fav, cap) if side > 0 else max(fav, cap)) - entry))
            worst = min(worst, side * (px - entry))
            return px, "stop", best, worst
        if hit_target:
            beyond = (o[i] > target) if side > 0 else (o[i] < target)
            px = o[i] if beyond and i > 0 else target
            best = max(best, side * (px - entry))
            worst = min(worst, side * (adv - entry))
            return px, "target", best, worst
        best = max(best, side * (fav - entry))
        worst = min(worst, side * (adv - entry))
    px = c[-1] - side * slip * tick
    worst = min(worst, side * (px - entry))
    return px, "flatten", best, worst


def backtest(bars: IntradayBars, spec: ContractSpec, p: ORBParams) -> ORBTrades:
    tick = spec.tick
    n_range = p.range_minutes // p.bar_minutes
    rows, widths = [], []
    for date, sl in bars.days():
        minute = bars.minute[sl]
        o, h, lo, c = bars.open[sl], bars.high[sl], bars.low[sl], bars.close[sl]
        in_range = (minute >= p.session_open) & (minute < p.session_open + p.range_minutes)
        expected = p.session_open + p.bar_minutes * np.arange(n_range)
        if not np.array_equal(minute[in_range], expected):
            continue  # missing opening bars: late open, holiday or data gap
        or_high, or_low = h[in_range].max(), lo[in_range].min()
        width = or_high - or_low
        history = widths[-p.range_lookback:]
        widths.append(width)
        tradable = np.flatnonzero((minute >= p.session_open + p.range_minutes)
                                  & (minute + p.bar_minutes <= p.flatten))
        buy_at = or_high + p.entry_ticks * tick
        sell_at = or_low - p.entry_ticks * tick
        risk_pts = buy_at - sell_at + p.slippage_ticks * tick * 2
        row = dict(date=date, side=0, contracts=0, entry=np.nan, exit=np.nan,
                   risk_points=risk_pts, range_points=width, pnl=0.0, reason="none",
                   steps=(0.0, 0.0, 0.0))
        if p.max_range_ratio is not None:
            if len(history) < p.range_lookback:
                row["reason"] = "warmup"
            elif width > p.max_range_ratio * np.median(history):
                row["reason"] = "filtered"
            if row["reason"] != "none":
                rows.append(row)
                continue
        for k, i in enumerate(tradable):
            up = h[i] >= buy_at and p.direction != "short"
            down = lo[i] <= sell_at and p.direction != "long"
            if not (up or down):
                continue
            if up and down:
                row["reason"] = "ambiguous"
                break
            side = 1 if up else -1
            level, stop = (buy_at, sell_at) if up else (sell_at, buy_at)
            gap = (o[i] > level) if up else (o[i] < level)
            entry = (o[i] if gap else level) + side * p.slippage_ticks * tick
            risk = abs(entry - stop) + p.slippage_ticks * tick
            n = min(int(p.risk_dollars // (risk * spec.multiplier)), p.max_contracts)
            if n < 1:
                row["reason"] = "too_wide"
                break
            target = None if p.target_r is None else entry + side * p.target_r * risk
            rest = tradable[k:]
            px, why, best, worst = _bar_path(side, entry, stop, target, o[rest], h[rest],
                                             lo[rest], c[rest], p.slippage_ticks, tick)
            dollars = n * spec.multiplier
            cost = 2 * p.commission_per_side * n
            pnl = side * (px - entry) * dollars - cost
            peak = best * dollars - p.commission_per_side * n
            trough = min(worst * dollars - p.commission_per_side * n, pnl)
            peak = max(peak, pnl)
            row.update(side=side, contracts=n, entry=entry, exit=px, risk_points=risk,
                       pnl=pnl, reason=why, steps=(peak, trough - peak, pnl - trough))
            break
        rows.append(row)
    if not rows:
        raise ValueError("no session had a complete opening range")
    return ORBTrades(
        date=np.array([r["date"] for r in rows]),
        side=np.array([r["side"] for r in rows], dtype=int),
        contracts=np.array([r["contracts"] for r in rows], dtype=int),
        entry=np.array([r["entry"] for r in rows], dtype=float),
        exit=np.array([r["exit"] for r in rows], dtype=float),
        risk_points=np.array([r["risk_points"] for r in rows], dtype=float),
        range_points=np.array([r["range_points"] for r in rows], dtype=float),
        pnl=np.array([r["pnl"] for r in rows], dtype=float),
        reason=np.array([r["reason"] for r in rows]),
        steps=np.array([r["steps"] for r in rows], dtype=float),
        multiplier=spec.multiplier,
    )


def summary(t: ORBTrades) -> dict[str, float]:
    """Headline stats: trades, win rate, avg R, net $, max DD $, profit factor, Sharpe."""
    traded = t.traded
    pnl = t.pnl[traded]
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    equity = np.r_[0.0, np.cumsum(t.pnl)]
    daily = t.pnl
    sd = daily.std(ddof=1) if daily.size > 1 else 0.0
    return {
        "sessions": float(t.date.size),
        "trades": float(traded.sum()),
        "win_rate": float(np.mean(pnl > 0)) if pnl.size else float("nan"),
        "avg_r": float(t.r_multiple.mean()) if pnl.size else float("nan"),
        "net": float(pnl.sum()),
        "max_dd": float(np.max(np.maximum.accumulate(equity) - equity)),
        "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
        "sharpe": float(daily.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
    }
