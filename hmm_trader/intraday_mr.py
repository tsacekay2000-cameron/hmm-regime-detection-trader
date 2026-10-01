"""Intraday mean-reversion rules (and gap continuation) on 5-minute RTH bars.

All rules are fixed before testing; decisions use bar closes and fill at the
next bar's open unless an order has a price (gap trades' stops and targets).

- ``vwap_reversion_positions``: fade stretches from the session VWAP. Bands
  are VWAP +/- k volume-weighted standard deviations of the typical price,
  both from 09:30. From the bar ending 10:00 to the one ending 15:00, go long
  on a close below the lower band, short above the upper; exit on a close back
  at VWAP or beyond the ``stop_k`` band, and (with ``range_filter``) when the
  day breaks its noise-boundary bands (see ``intraday_momentum``), the sign
  of a trend day. With the filter, entries also need the bands unbroken so far.
- ``gap_trades``: today's open against the previous close of the same
  contract, in units of the 14-day average RTH range. Small gaps are faded at
  the open toward the previous close (target), stop ``fade_stop`` x range
  away; large gaps are followed with the stop at the previous close. Flat at
  16:00, a fixed number of contracts, intrabar fills as in ``smc``.
- ``rsi2_intraday_positions``: Connors' RSI(2) on 15-minute (or 5-minute) bars:
  long under ``entry`` above the 200-bar average, out above the 5-bar average;
  shorts mirror it. Flat at 16:00.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from hmm_trader import intraday_momentum as im
from hmm_trader.mean_reversion import rsi, sma
from hmm_trader.smc import Trade, _manage

N = im.BARS_PER_DAY


def vwap_bands(b: im.DayBars) -> tuple[np.ndarray, np.ndarray]:
    """Session VWAP and volume-weighted standard deviation at each bar's close."""
    tp = (b.high + b.low + b.close) / 3
    v = np.where(b.volume > 0, b.volume, 0.0)
    cv = np.cumsum(v, axis=1)
    safe = np.where(cv > 0, cv, np.nan)
    vw = np.cumsum(v * tp, axis=1) / safe
    var = np.cumsum(v * tp * tp, axis=1) / safe - vw * vw
    vw = np.where(np.isfinite(vw), vw, b.close)
    sd = np.sqrt(np.clip(np.nan_to_num(var), 0.0, None))
    return vw, sd


def band_broken(b: im.DayBars, lookback: int = 14) -> np.ndarray:
    """True from the first close outside the noise-boundary bands (NaN days: True)."""
    upper, lower = im.noise_bands(b, lookback)
    out = (b.close > upper) | (b.close < lower) | ~np.isfinite(upper)
    return np.maximum.accumulate(out, axis=1)


def vwap_reversion_positions(b: im.DayBars, k: float = 2.0, stop_k: float = 3.0,
                             range_filter: bool = True, first: str = "10:00",
                             last: str = "15:00") -> np.ndarray:
    vw, sd = vwap_bands(b)
    broken = band_broken(b) if range_filter else np.zeros(b.close.shape, dtype=bool)
    pos = np.zeros(b.close.shape)
    i0, i1 = im.bar_index(first) - 1, im.bar_index(last) - 1   # bars ending at first / last
    for d in range(b.close.shape[0]):
        held = 0
        for i in range(i0, N - 1):
            c, m, s = b.close[d, i], vw[d, i], sd[d, i]
            if held > 0 and (c >= m or c < m - stop_k * s or broken[d, i]):
                held = 0
            elif held < 0 and (c <= m or c > m + stop_k * s or broken[d, i]):
                held = 0
            if held == 0 and i <= i1 and s > 0 and not broken[d, i]:
                held = 1 if c < m - k * s else -1 if c > m + k * s else 0
            pos[d, i + 1] = held
    return pos


def daily_range(b: im.DayBars, n: int = 14) -> np.ndarray:
    """Average RTH high-low range of the previous ``n`` sessions (NaN before)."""
    rng = b.high.max(axis=1) - b.low.min(axis=1)
    out = np.full(rng.size, np.nan)
    for d in range(n, rng.size):
        out[d] = rng[d - n:d].mean()
    return out


def gap_trades(b: im.DayBars, multiplier: float, tick: float, mode: str = "fade",
               small: float = 0.3, large: float = 0.6, fade_stop: float = 0.5,
               contracts: int = 1, commission: float = 0.62) -> list[Trade]:
    """Gap fade (small gaps) or gap-and-go (large gaps), one trade at the 09:30 open."""
    adr = daily_range(b)
    trades = []
    for d in range(b.close.shape[0]):
        pc, o = b.prev_close[d], b.open[d, 0]
        if not (np.isfinite(pc) and np.isfinite(adr[d])) or o == pc:
            continue
        gap = (o - pc) / adr[d]
        if mode == "fade" and abs(gap) < small:
            side = -int(np.sign(gap))
            target_dist = abs(o - pc)
            stop_dist = fade_stop * adr[d]
        elif mode == "go" and abs(gap) >= large:
            side = int(np.sign(gap))
            stop_dist = abs(o - pc)
            target_dist = np.inf
        else:
            continue
        entry = o + side * tick                       # market order at the open
        if np.isfinite(target_dist):                  # fade: the target is the previous close
            target_dist = side * (pc - entry)
            if target_dist <= 0:
                continue
        risk_pts = stop_dist + tick
        n = contracts
        stop = entry - side * stop_dist
        target = entry + side * target_dist if np.isfinite(target_dist) else entry + side * 1e9
        xb, xp, why = _manage(b, d, side, 0, entry, stop, target, tick)
        pnl = side * (xp - entry) * multiplier * n - 2 * n * commission
        risk = risk_pts * multiplier * n
        # mirror: the other side with the same distances, same bars
        m_stop = entry + side * stop_dist
        m_target = entry - side * target_dist if np.isfinite(target_dist) else entry - side * 1e9
        _, mp, _ = _manage(b, d, -side, 0, entry, m_stop, m_target, tick)
        m_pnl = -side * (mp - entry) * multiplier * n - 2 * n * commission
        trades.append(Trade(d, side, 0, entry, stop, target, n, xb, xp, why, pnl,
                            pnl / risk, pnl / risk, m_pnl / risk))
    return trades


def rsi2_intraday_positions(b: im.DayBars, minutes: int = 15, entry: float = 10.0,
                            trend: int = 200, exit_sma: int = 5, allow_short: bool = True,
                            last: str = "15:30") -> np.ndarray:
    """Positions (5-minute grid) for RSI(2) on ``minutes`` bars, flat every night."""
    k = minutes // im.BAR
    per_day = N // k
    days = b.close.shape[0]
    close = b.close[:, k - 1::k][:, :per_day].reshape(-1)       # close of each bigger bar
    r, fast, slow = rsi(close, 2), sma(close, exit_sma), sma(close, trend)
    r, fast, slow = (x.reshape(days, per_day) for x in (r, fast, slow))
    c = close.reshape(days, per_day)
    last_j = (im.bar_index(last) // k) - 1                       # bar ending at `last`
    pos = np.zeros(b.close.shape)
    for d in range(days):
        held = 0
        for j in range(per_day - 1):
            if held > 0 and c[d, j] > fast[d, j]:
                held = 0
            elif held < 0 and c[d, j] < fast[d, j]:
                held = 0
            if held == 0 and j <= last_j and np.isfinite(r[d, j]) and np.isfinite(slow[d, j]):
                if r[d, j] < entry and c[d, j] > slow[d, j]:
                    held = 1
                elif allow_short and r[d, j] > 100 - entry and c[d, j] < slow[d, j]:
                    held = -1
            pos[d, (j + 1) * k:(j + 2) * k] = held
    return pos
