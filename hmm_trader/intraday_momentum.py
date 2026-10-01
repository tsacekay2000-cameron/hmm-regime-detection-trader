"""Intraday momentum rules on 5-minute RTH bars of index futures.

Both rules come from published studies on SPY and are used with their
published settings, adapted only to 5-minute bars and futures costs:

- ``last_half_hour``: market intraday momentum (Gao, Han, Li & Zhou,
  "Market Intraday Momentum", JFE 2018). The return from the previous
  close to 10:00 ET predicts the last half hour: trade 15:30 -> 16:00 in its
  direction. ``signal_end=15:30`` gives the variant of Baltussen, Da,
  Lammers & Martens (JFE 2021), who use the return up to 15:30.
- ``noise_boundary``: "Beat the Market" (Zarattini, Aziz & Barbon, 2024).
  Around each day's open, a noise band at each time of day is the average
  absolute move from the open at that time over the previous 14 days,
  measured from max/min(open, previous close). At every :00 and :30 from
  10:00 to 15:30, go long above the band and short below it; a long exits
  when the close falls below max(upper band, VWAP), a short when it rises
  above min(lower band, VWAP). Everything is closed at 16:00. The paper sizes
  by volatility; here the size is a fixed number of contracts.

Decisions use the close of the bar ending at a checkpoint and fill at the
next bar's open plus slippage, so nothing uses information before it exists.
Each day is reduced to (to peak, to trough, to final) dollar steps for the
prop simulator, with the trough checked from bar highs and lows.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BAR = 5                  # minutes per bar
OPEN_MIN = 9 * 60 + 30   # 09:30 ET, minute of the first bar
BARS_PER_DAY = 78        # 09:30 .. 15:55 bar starts


def bar_index(hhmm: str) -> int:
    """Index of the bar that STARTS at ``hhmm`` (``"15:30"`` -> 72)."""
    h, m = map(int, hhmm.split(":"))
    return (h * 60 + m - OPEN_MIN) // BAR


@dataclass
class DayBars:
    """Full regular sessions only, as (n_days, 78) arrays."""

    date: np.ndarray        # (n_days,) "YYYY-MM-DD"
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    prev_close: np.ndarray  # (n_days,) previous RTH close of the same contract (NaN if unknown)


def load_day_bars(path: Path | str) -> DayBars:
    """Read ``datetime,open,high,low,close,contract,volume,prev_close`` 5-minute RTH bars.

    Days without all 78 bars (early closes, gaps) are dropped.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        t = np.genfromtxt(fh, delimiter=",", names=True, dtype=None, encoding="utf-8")
    stamp = np.asarray(t["datetime"], dtype=str)
    date = np.array([s[:10] for s in stamp])
    minute = np.array([int(s[11:13]) * 60 + int(s[14:16]) for s in stamp])
    idx = (minute - OPEN_MIN) // BAR
    days, inv = np.unique(date, return_inverse=True)
    keep = np.bincount(inv, minlength=days.size) == BARS_PER_DAY
    cols = {}
    for name in ("open", "high", "low", "close", "volume"):
        a = np.full((days.size, BARS_PER_DAY), np.nan)
        a[inv, idx] = np.asarray(t[name], dtype=float)
        cols[name] = a[keep]
    prev = np.full(days.size, np.nan)
    prev[inv] = np.asarray(t["prev_close"], dtype=float)
    return DayBars(days[keep], **cols, prev_close=prev[keep])


@dataclass
class DayTrades:
    date: np.ndarray
    position: np.ndarray   # (n_days, 78) contracts held during each bar (+1/-1/0)
    pnl: np.ndarray        # (n_days,) dollars after costs, 1 contract
    steps: np.ndarray      # (n_days, 3) to peak, to trough, to final
    trades: np.ndarray     # (n_days,) round trips per day


def simulate_positions(b: DayBars, pos: np.ndarray, multiplier: float, tick: float,
                       commission: float = 0.62) -> DayTrades:
    """Dollar P&L of holding ``pos[d, i]`` contracts through bar ``i``.

    A position change at bar ``i`` fills at that bar's open with one tick of
    slippage against the trade; everything is closed at the last bar's close
    (one tick of slippage). Commission is per contract per side.
    """
    n = pos.shape[0]
    pnl = np.zeros(n)
    steps = np.zeros((n, 3))
    trades = np.zeros(n, dtype=int)
    for d in range(n):
        p = pos[d]
        cash, held, eq_path = 0.0, 0.0, [0.0]
        peak = trough = 0.0
        for i in range(BARS_PER_DAY):
            target = p[i]
            if target != held:
                q = target - held
                fill = b.open[d, i] + np.sign(q) * tick
                cash -= q * fill * multiplier + abs(q) * commission
                trades[d] += int(target != 0)  # each new position is a trade
                held = target
            if held:
                worst = b.low[d, i] if held > 0 else b.high[d, i]
                best = b.high[d, i] if held > 0 else b.low[d, i]
                trough = min(trough, cash + held * worst * multiplier)
                peak = max(peak, cash + held * best * multiplier)
        if held:
            fill = b.close[d, -1] - np.sign(held) * tick
            cash += held * fill * multiplier - abs(held) * commission
        final = cash
        peak, trough = max(peak, final), min(trough, final)
        pnl[d] = final
        steps[d] = (peak, trough - peak, final - trough)
    return DayTrades(b.date, pos, pnl, steps, trades)


def last_half_hour_positions(b: DayBars, signal_end: str = "10:00",
                             trade_start: str = "15:30") -> tuple[np.ndarray, np.ndarray]:
    """Positions for the Gao et al. rule and the signal return per day.

    The signal is the return from the previous close to the close of the bar
    ending at ``signal_end``; the trade holds its sign from ``trade_start`` to
    the close. Days without a previous close are flat.
    """
    end_bar = bar_index(signal_end) - 1          # bar ending at signal_end
    sig = b.close[:, end_bar] / b.prev_close - 1
    pos = np.zeros(b.open.shape)
    side = np.where(np.isfinite(sig), np.sign(sig), 0)
    pos[:, bar_index(trade_start):] = side[:, None]
    return pos, sig


def vwap(b: DayBars) -> np.ndarray:
    """Running VWAP at each bar's close, from the typical price (high + low + close) / 3."""
    tp = (b.high + b.low + b.close) / 3
    v = np.where(b.volume > 0, b.volume, 0.0)
    cum_v = np.cumsum(v, axis=1)
    out = np.cumsum(tp * v, axis=1) / np.where(cum_v > 0, cum_v, np.nan)
    return np.where(np.isfinite(out), out, b.close)


def noise_bands(b: DayBars, lookback: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Upper and lower noise bands at each bar's close (NaN during the warm-up)."""
    move = np.abs(b.close / b.open[:, :1] - 1)
    sigma = np.full(move.shape, np.nan)
    for d in range(lookback, move.shape[0]):
        sigma[d] = move[d - lookback:d].mean(axis=0)
    ref_hi = np.fmax(b.open[:, 0], b.prev_close)[:, None]
    ref_lo = np.fmin(b.open[:, 0], b.prev_close)[:, None]
    return ref_hi * (1 + sigma), ref_lo * (1 - sigma)


def noise_boundary_positions(b: DayBars, lookback: int = 14, first: str = "10:00",
                             last: str = "15:30", every: int = 30) -> np.ndarray:
    """Positions for the Zarattini et al. rule: checks at every ``every`` minutes."""
    upper, lower = noise_bands(b, lookback)
    vw = vwap(b)
    pos = np.zeros(b.open.shape)
    checks = range(bar_index(first), bar_index(last) + 1, every // BAR)
    for d in range(b.open.shape[0]):
        held = 0
        for k in checks:
            i = k - 1                       # bar ending at the checkpoint
            c, ub, lb = b.close[d, i], upper[d, i], lower[d, i]
            if not np.isfinite(ub):
                break
            if held > 0 and c < max(ub, vw[d, i]):
                held = 0
            elif held < 0 and c > min(lb, vw[d, i]):
                held = 0
            if held == 0:
                held = 1 if c > ub else -1 if c < lb else 0
            pos[d, k:] = held
    return pos


def flip_test(pnl_gross: np.ndarray, rng: np.random.Generator, n: int = 20000) -> float:
    """Share of random sign flips of each day's gross P&L that do at least as well.

    Keeps when and how long the rule trades but makes each direction a coin
    flip, so it asks whether the rule's choice of long or short added anything.
    """
    traded = pnl_gross != 0
    x = pnl_gross[traded]
    signs = rng.choice((-1.0, 1.0), size=(n, x.size))
    return float(np.mean((signs * x).sum(axis=1) >= x.sum()))
