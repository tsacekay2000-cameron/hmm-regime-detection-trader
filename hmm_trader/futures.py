"""Futures helpers: contract specs, quarterly rolls and per-contract dollar P&L.

Continuous front-month series (e.g. TradingView ``MES1!``) jump on the day
they switch to the next contract. The jump is the calendar spread, not a
return: at 4-5% interest rates an equity index contract rolls into one ~1%
more expensive every quarter, so an unadjusted series overstates a long's
return by about that much. Given the second-month close as well, the roll
can be handled exactly: a position rolled at the previous close was bought
at the second-month price, so the P&L on the roll day is measured from it.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    multiplier: float  # dollars per point
    tick: float        # minimum price increment, in points

    @property
    def tick_value(self) -> float:
        return self.multiplier * self.tick


CONTRACTS = {
    "MES": ContractSpec("MES", multiplier=5.0, tick=0.25),   # Micro E-mini S&P 500
    "MNQ": ContractSpec("MNQ", multiplier=2.0, tick=0.25),   # Micro E-mini Nasdaq-100
    "M2K": ContractSpec("M2K", multiplier=5.0, tick=0.10),   # Micro E-mini Russell 2000
    "MYM": ContractSpec("MYM", multiplier=0.5, tick=1.0),    # Micro E-mini Dow
    "MGC": ContractSpec("MGC", multiplier=10.0, tick=0.10),  # Micro Gold, $10/oz
    "MCL": ContractSpec("MCL", multiplier=100.0, tick=0.01),  # Micro WTI Crude, $100/bbl
}


@dataclass
class FuturesBars:
    """Daily bars of the front contract plus the second contract's close."""

    dates: np.ndarray       # trade dates, "YYYY-MM-DD"
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    next_close: np.ndarray  # second-month contract close

    def __len__(self) -> int:
        return self.dates.size

    def __getitem__(self, idx) -> "FuturesBars":
        return FuturesBars(*(getattr(self, f)[idx] for f in
                             ("dates", "open", "high", "low", "close", "next_close")))


def load_bars(path: Path | str) -> FuturesBars:
    """Read a CSV with columns date, open, high, low, close, next_close."""
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    fields = ("date", "open", "high", "low", "close", "next_close")
    missing = set(fields) - set(table.dtype.names)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    dates = np.asarray(table["date"], dtype=str)
    if np.any(dates[1:] <= dates[:-1]):
        raise ValueError("dates must be strictly increasing")
    return FuturesBars(dates, *(np.asarray(table[f], dtype=float) for f in fields[1:]))


def quarterly_expiries(years: Sequence[int]) -> list[dt.date]:
    """Equity index expiries: 3rd Friday of Mar/Jun/Sep/Dec.

    When that Friday is Juneteenth (June 19, an exchange holiday) the contract
    expires the Thursday before.
    """
    out = []
    for y in years:
        for m in (3, 6, 9, 12):
            first = dt.date(y, m, 1)
            friday = first + dt.timedelta(days=(4 - first.weekday()) % 7 + 14)
            if (friday.month, friday.day) == (6, 19):
                friday -= dt.timedelta(days=1)
            out.append(friday)
    return out


def quarterly_roll_mask(dates: Sequence[str], sessions_before_expiry: int = 3) -> np.ndarray:
    """True on the first session traded in the new contract.

    The default matches TradingView's continuous ``1!`` series for CME equity
    index futures, which switches ``sessions_before_expiry`` sessions before
    the expiry session (checked against the cash index from 2019 to 2026).
    If the data ends before an expiry, the sessions still to come are assumed
    to be weekdays other than Juneteenth.
    """
    dates = np.asarray(dates, dtype=str)
    mask = np.zeros(dates.size, dtype=bool)
    if dates.size == 0:
        return mask
    last = np.datetime64(dates[-1])
    years = range(int(dates[0][:4]), int(dates[-1][:4]) + 1)
    for expiry in quarterly_expiries(years):
        exp = np.datetime64(expiry)
        n_upto = int(np.searchsorted(dates, str(exp), side="right"))
        # sessions after the data ends, up to and including the expiry
        holidays = [f"{expiry.year}-06-19"]
        pending = int(np.busday_count(last + 1, exp + 1, holidays=holidays)) if exp > last else 0
        idx = n_upto - 1 - sessions_before_expiry + pending
        if 1 <= idx < dates.size:
            mask[idx] = True
    return mask


def point_changes(bars: FuturesBars, roll: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Close-to-close points for holding the front contract, and the overnight gap.

    Returns ``(close_change, gap)`` for sessions 1..n-1: ``close_change[t]`` is
    the P&L in points of one contract held from close ``t-1`` to close ``t``,
    and ``gap[t]`` the part of it from close ``t-1`` to open ``t``. On a roll
    session both are measured from the previous second-month close.
    """
    prev = np.where(roll[1:], bars.next_close[:-1], bars.close[:-1])
    return bars.close[1:] - prev, bars.open[1:] - prev


def adjusted_returns(bars: FuturesBars, roll: np.ndarray) -> np.ndarray:
    """Roll-adjusted close-to-close simple returns (length n-1)."""
    prev = np.where(roll[1:], bars.next_close[:-1], bars.close[:-1])
    return bars.close[1:] / prev - 1.0


def intraday_long_steps(bars: FuturesBars, roll: np.ndarray, hold_overnight: bool) -> np.ndarray:
    """Per-contract point P&L of a long as intraday steps, shape ``(n-1, 4)``.

    Steps are prev close -> open (the overnight gap, zero unless
    ``hold_overnight``), open -> high -> low -> close. Putting the high before
    the low is the worst case for a long under a trailing drawdown (the peak
    lifts the floor before the dip), so breach checks err on the strict side.
    Shorts are the negation, with the ordering then favourable instead.
    """
    b = bars[1:]
    _, gap = point_changes(bars, roll)
    if not hold_overnight:
        gap = np.zeros_like(gap)
    return np.column_stack([gap, b.high - b.open, b.low - b.high, b.close - b.low])
