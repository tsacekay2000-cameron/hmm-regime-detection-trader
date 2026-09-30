"""Mean-reversion strategies for futures: a daily RSI(2) pullback and an
intraday Bollinger-band fade.

``rsi2_positions`` (daily, long only, after Connors & Alvarez 2008): buy at
the close when RSI(2) is below ``entry`` and the close is above its
``trend``-day average; exit at the first close above the ``exit_sma``-day
average. Indicators use roll-adjusted closes. The result says, for each
session, whether a long was held into it, so it plugs into the daily P&L
tools in ``hmm_trader.futures``.

``bollinger_fade`` (intraday, both directions): when a bar closes outside
``sma(lookback) +/- k * sd(lookback)`` of the session's closes so far, fade
it at the next bar's open. The stop is ``k * sd`` from the entry (the band
distance at the signal); exit when a bar closes back through the moving
average, at the stop, or at the close of the last bar before ``flatten``.
Several trades per day are allowed, one position at a time. Fills follow
``hmm_trader.orb``: slippage on entries, stops and market exits; a stop is
checked before the mean-reversion exit on the same bar; a gap through the
stop fills at the open. Each day is also summarized as peak / trough /
final P&L steps for the prop firm simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from hmm_trader.futures import ContractSpec
from hmm_trader.orb import IntradayBars


# ---------------------------------------------------------------------------
# Daily RSI(2)
# ---------------------------------------------------------------------------

def rsi(close: np.ndarray, n: int = 2) -> np.ndarray:
    """Wilder's RSI; NaN until ``n`` changes are available."""
    close = np.asarray(close, dtype=float)
    out = np.full(close.size, np.nan)
    if close.size <= n:
        return out
    diff = np.diff(close)
    gain, loss = np.maximum(diff, 0.0), np.maximum(-diff, 0.0)
    avg_g, avg_l = gain[:n].mean(), loss[:n].mean()
    for i in range(n, close.size):
        if i > n:
            avg_g = (avg_g * (n - 1) + gain[i - 1]) / n
            avg_l = (avg_l * (n - 1) + loss[i - 1]) / n
        out[i] = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    return out


def sma(x: np.ndarray, n: int) -> np.ndarray:
    """Trailing simple moving average; NaN for the first ``n - 1`` values."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.size, np.nan)
    if x.size >= n:
        c = np.cumsum(np.r_[0.0, x])
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def rsi2_positions(
    close: np.ndarray,
    entry: float = 10.0,
    trend: Optional[int] = 200,
    exit_sma: int = 5,
    rsi_n: int = 2,
) -> np.ndarray:
    """Long (True) held into sessions 1..n-1, decided at each previous close.

    ``close`` should be roll-adjusted (e.g. cumulated ``adjusted_returns``).
    ``trend=None`` drops the trend filter.
    """
    close = np.asarray(close, dtype=float)
    r, exit_ma = rsi(close, rsi_n), sma(close, exit_sma)
    long_ma = np.full(close.size, -np.inf) if trend is None else sma(close, trend)
    held = np.zeros(close.size, dtype=bool)  # position after the close of day t
    for t in range(close.size):
        prev = held[t - 1] if t else False
        if prev:
            held[t] = not (close[t] > exit_ma[t])
        elif np.isfinite(r[t]) and not np.isnan(long_ma[t]):
            held[t] = r[t] < entry and close[t] > long_ma[t]
    return held[:-1]


def trade_spans(held: np.ndarray) -> list[tuple[int, int]]:
    """``(start, end)`` index ranges (end exclusive) of consecutive True runs."""
    edges = np.diff(np.r_[0, held.astype(int), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def true_range_atr(bars, roll: np.ndarray, n: int = 10) -> np.ndarray:
    """Wilder ATR on daily bars; the previous close is the second-month close
    on roll days, so the contract switch does not count as range."""
    prev = np.r_[np.nan, np.where(roll[1:], bars.next_close[:-1], bars.close[:-1])]
    tr = np.fmax(bars.high, prev) - np.fmin(bars.low, prev)
    tr[0] = bars.high[0] - bars.low[0]
    out = np.full(tr.size, np.nan)
    if tr.size >= n:
        out[n - 1] = tr[:n].mean()
        for i in range(n, tr.size):
            out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


@dataclass
class SessionTrades:
    """Positions entered at each session open and closed by its close."""

    steps: np.ndarray       # (n-1, 4) dollars per contract: costs, to high, to low, to close
    pnl: np.ndarray         # per trade, dollars per contract after costs
    stopped: np.ndarray     # per trade: stop hit
    in_market: np.ndarray   # (n-1,) sessions actually held
    limit_hit: np.ndarray   # (n-1,) sessions closed early by the daily loss limit


def flat_session_trades(
    bars,
    roll: np.ndarray,
    held: np.ndarray,
    multiplier: float,
    cost_per_side: float,
    stop_atr: Optional[float] = None,
    atr_n: int = 10,
    session_stop: Optional[float] = None,
) -> SessionTrades:
    """Hold each ``held`` session from its open to its close (flat through the
    daily break), with an optional stop ``stop_atr`` x ATR below the first
    entry, kept fixed for the trade (shifted by the calendar spread on roll
    days). A stop fills at the stop, or at the open if the session opens
    below it, and ends the trade until the next signal. A trade whose signal
    comes before ``atr_n`` sessions of data has no stop. Rows and ``held``
    cover sessions 1..n-1, as in ``rsi2_positions``.

    ``session_stop`` is a daily loss limit in points per contract: when a
    session falls that far below its open (the entry), the position is sold
    there and the day is over, but the trade resumes at the next session's
    open while the signal lasts. Whichever of the two levels is higher is hit
    first on the way down.

    Each session is replayed open -> high -> low -> close, the worst order for
    a long; a stopped session ends at the fill.
    """
    atr = true_range_atr(bars, roll, atr_n) if stop_atr is not None else None
    steps = np.zeros((held.size, 4))
    in_market = np.zeros(held.size, dtype=bool)
    limit_hit = np.zeros(held.size, dtype=bool)
    pnl, stopped = [], []
    for s, e in trade_spans(held):
        stop, total, hit = None, 0.0, False
        for t in range(s, e):
            b = t + 1  # bar index of session t
            o, h, lo, c = bars.open[b], bars.high[b], bars.low[b], bars.close[b]
            if stop is not None and roll[b]:
                stop += bars.next_close[b - 1] - bars.close[b - 1]
            if t == s and atr is not None and np.isfinite(atr[b - 1]):
                stop = o - stop_atr * atr[b - 1]  # ATR known at the signal close
            in_market[t] = True
            row = [-2 * cost_per_side, (h - o) * multiplier, 0.0, 0.0]
            day_limit = o - session_stop if session_stop is not None else -np.inf
            trade_stop = stop if stop is not None else -np.inf
            if lo <= max(day_limit, trade_stop):
                if trade_stop >= day_limit:
                    fill = min(o, trade_stop)
                    hit = True
                else:
                    fill = day_limit
                    limit_hit[t] = True
                row[2] = (fill - h) * multiplier
            else:
                row[2], row[3] = (lo - h) * multiplier, (c - lo) * multiplier
            steps[t] = row
            total += sum(row)
            if hit:
                break
        pnl.append(total)
        stopped.append(hit)
    return SessionTrades(steps, np.array(pnl), np.array(stopped, dtype=bool), in_market,
                         limit_hit)


# ---------------------------------------------------------------------------
# Intraday Bollinger fade
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FadeParams:
    lookback: int = 20
    k: float = 2.0
    risk_dollars: float = 200.0
    max_contracts: int = 50
    slippage_ticks: float = 1.0
    commission_per_side: float = 0.62
    session_open: int = 9 * 60 + 30
    flatten: int = 15 * 60 + 55
    bar_minutes: int = 5

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be >= 2")
        if self.k <= 0 or self.risk_dollars <= 0 or self.max_contracts < 1:
            raise ValueError("k, risk_dollars and max_contracts must be positive")

    def label(self) -> str:
        return f"fade {self.lookback} bars / {self.k:g} sd"


@dataclass
class FadeTrades:
    """Trade list plus one row per session for daily P&L and prop steps."""

    date: np.ndarray        # per trade
    side: np.ndarray        # +1 / -1
    contracts: np.ndarray
    entry: np.ndarray
    exit: np.ndarray
    risk_points: np.ndarray  # stop distance incl. exit slippage
    pnl: np.ndarray         # dollars after costs
    reason: np.ndarray      # mean / stop / flatten
    multiplier: float
    day_date: np.ndarray    # per session
    day_pnl: np.ndarray
    day_steps: np.ndarray   # (n_days, 3): to peak, to trough, to final

    @property
    def r_multiple(self) -> np.ndarray:
        return self.pnl / (self.risk_points * self.contracts * self.multiplier)


def bollinger_fade(bars: IntradayBars, spec: ContractSpec, p: FadeParams) -> FadeTrades:
    tick, slip = spec.tick, p.slippage_ticks * spec.tick
    trades: list[dict] = []
    day_date, day_pnl, day_steps = [], [], []
    for date, sl in bars.days():
        minute = bars.minute[sl]
        keep = (minute >= p.session_open) & (minute + p.bar_minutes <= p.flatten)
        o, h, lo, c = (getattr(bars, f)[sl][keep] for f in ("open", "high", "low", "close"))
        n = c.size
        realized, path = 0.0, [0.0]
        pos = None  # dict(side, n, entry, stop)
        pending = None
        for i in range(n):
            if pending is not None:  # fill the signal from the previous close
                side, dist = pending
                pending = None
                entry = o[i] + side * slip
                risk = dist + slip
                size = min(int(p.risk_dollars // (risk * spec.multiplier)), p.max_contracts)
                if size >= 1:
                    pos = dict(side=side, n=size, entry=entry, stop=entry - side * dist,
                               risk=risk, first=True)
                    realized -= p.commission_per_side * size
                    path.append(realized)
            if pos is not None:
                side, size, entry, stop = pos["side"], pos["n"], pos["entry"], pos["stop"]
                dollars = size * spec.multiplier
                fav = h[i] if side > 0 else lo[i]
                adv = lo[i] if side > 0 else h[i]
                window = c[max(0, i - p.lookback + 1):i + 1]
                mean = window.mean()
                exit_px, why = None, None
                if (adv <= stop) if side > 0 else (adv >= stop):
                    gapped = (o[i] < stop) if side > 0 else (o[i] > stop)
                    exit_px = (o[i] if gapped and not pos["first"] else stop) - side * slip
                    why = "stop"
                elif (c[i] >= mean) if side > 0 else (c[i] <= mean):
                    exit_px, why = c[i] - side * slip, "mean"
                elif i == n - 1:
                    exit_px, why = c[i] - side * slip, "flatten"
                # open P&L at the bar's best, then worst point (peak before trough)
                path.append(realized + side * (fav - entry) * dollars)
                if exit_px is not None:
                    closed = side * (exit_px - entry) * dollars
                    worst = closed if why == "stop" else min(side * (adv - entry) * dollars,
                                                            closed)
                    path.append(realized + worst)
                    pnl = closed - 2 * p.commission_per_side * size
                    realized += closed - p.commission_per_side * size
                    path.append(realized)
                    trades.append(dict(date=date, side=side, contracts=size, entry=entry,
                                       exit=exit_px, risk_points=pos["risk"], pnl=pnl,
                                       reason=why))
                    pos = None
                else:
                    path.append(realized + side * (adv - entry) * dollars)
                    path.append(realized + side * (c[i] - entry) * dollars)
                    pos["first"] = False
                    continue  # no new signal while in a position
            # signal on this bar's close, filled at the next bar's open
            if pos is None and i + 1 < n and i + 1 >= p.lookback:
                window = c[i - p.lookback + 1:i + 1]
                mu, sd = window.mean(), window.std(ddof=1)
                if sd > 0:
                    if c[i] < mu - p.k * sd:
                        pending = (1, p.k * sd)
                    elif c[i] > mu + p.k * sd:
                        pending = (-1, p.k * sd)
        peak, trough = max(path), min(path)
        day_date.append(date)
        day_pnl.append(realized)
        day_steps.append((peak, trough - peak, realized - trough))
    arr = lambda key, dtype=float: np.array([t[key] for t in trades], dtype=dtype)  # noqa: E731
    return FadeTrades(
        date=np.array([t["date"] for t in trades]), side=arr("side", int),
        contracts=arr("contracts", int), entry=arr("entry"), exit=arr("exit"),
        risk_points=arr("risk_points"), pnl=arr("pnl"),
        reason=np.array([t["reason"] for t in trades]), multiplier=spec.multiplier,
        day_date=np.array(day_date), day_pnl=np.array(day_pnl),
        day_steps=np.array(day_steps, dtype=float),
    )
