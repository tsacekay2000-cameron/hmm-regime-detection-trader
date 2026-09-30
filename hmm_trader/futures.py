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
    roll_sessions: int = 3  # TradingView "1!" switches this many sessions before expiry
    roll_rule: str = "quarterly"  # "quarterly" (equity index), "gold" or "crude"
    daily_close_et: str = "16:00"  # time of TradingView's daily close (settlement), ET

    @property
    def tick_value(self) -> float:
        return self.multiplier * self.tick


CONTRACTS = {
    "MES": ContractSpec("MES", multiplier=5.0, tick=0.25),   # Micro E-mini S&P 500
    "MNQ": ContractSpec("MNQ", multiplier=2.0, tick=0.25),   # Micro E-mini Nasdaq-100
    "M2K": ContractSpec("M2K", multiplier=5.0, tick=0.10, roll_sessions=2),  # Micro Russell
    "MYM": ContractSpec("MYM", multiplier=0.5, tick=1.0),    # Micro E-mini Dow
    "MGC": ContractSpec("MGC", multiplier=10.0, tick=0.10, roll_rule="gold",
                        daily_close_et="13:30"),  # Micro Gold
    "MCL": ContractSpec("MCL", multiplier=100.0, tick=0.01, roll_rule="crude",
                        daily_close_et="14:30"),  # Micro WTI Crude, $100/bbl
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

    The default matches TradingView's continuous ``1!`` series for MES/MNQ,
    which switches ``sessions_before_expiry`` sessions before the expiry
    session; M2K switches one session later (``ContractSpec.roll_sessions``).
    Both checked against the cash indexes from 2019 to 2026.
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


def gold_roll_mask(dates: Sequence[str]) -> np.ndarray:
    """True on the first session of the new contract for TradingView's MGC1!.

    The series moves to the next active month (Feb/Apr/Jun/Aug/Dec, skipping
    Oct) on the second-to-last session of Jan, Mar, May, Jul and Nov, the day
    before first notice. Matched against individual contracts for 2024-2026;
    for 2019-2026 the adjustment removes the step in the futures-spot basis at
    these dates (+0.69% raw, -0.13% adjusted, -0.04% on random days). A month
    the data ends in is skipped.
    """
    dates = np.asarray(dates, dtype=str)
    mask = np.zeros(dates.size, dtype=bool)
    months = np.array([d[:7] for d in dates])
    for m in np.unique(months):
        idx = np.flatnonzero(months == m)
        if int(m[5:]) in (1, 3, 5, 7, 11) and idx.size >= 2 and idx[-1] + 1 < dates.size:
            mask[idx[-2]] = True
    return mask


def us_exchange_holidays(years: Sequence[int]) -> list[str]:
    """CME holidays for US energy and index futures (no settlement those days).

    New Year's, MLK, Presidents', Good Friday, Memorial, Juneteenth (from
    2022), Independence, Labor, Thanksgiving and Christmas, moved to the
    nearest weekday when they fall on a weekend (New Year's on a Saturday is
    not moved back into December). Listed rather than read from the data,
    whose series can carry bars dated on these days.
    """
    def nth(y, m, weekday, n):  # n-th weekday of the month, n=-1 for the last
        if n > 0:
            first = dt.date(y, m, 1)
            return first + dt.timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
        last = dt.date(y, m + 1, 1) - dt.timedelta(days=1)
        return last - dt.timedelta(days=(last.weekday() - weekday) % 7)

    def observed(d):
        return d + dt.timedelta(days={5: -1, 6: 1}.get(d.weekday(), 0))

    def easter(y):  # anonymous Gregorian algorithm (Meeus/Butcher)
        a, b, c = y % 19, y // 100, y % 100
        d, e = b // 4, b % 4
        g = (b - (b + 8) // 25 + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        l_ = (32 + 2 * e + 2 * (c // 4) - h - c % 4) % 7
        m = (a + 11 * h + 22 * l_) // 451
        n = h + l_ - 7 * m + 114
        return dt.date(y, n // 31, n % 31 + 1)

    out = []
    for y in years:
        days = [nth(y, 1, 0, 3), nth(y, 2, 0, 3), easter(y) - dt.timedelta(days=2),
                nth(y, 5, 0, -1), observed(dt.date(y, 7, 4)), nth(y, 9, 0, 1),
                nth(y, 11, 3, 4), observed(dt.date(y, 12, 25))]
        if dt.date(y, 1, 1).weekday() != 5:
            days.append(observed(dt.date(y, 1, 1)))
        if y >= 2022:
            days.append(observed(dt.date(y, 6, 19)))
        out += [d.isoformat() for d in days]
    return sorted(out)


def crude_roll_mask(dates: Sequence[str]) -> np.ndarray:
    """True on the first session of the new contract for TradingView's MCL1!.

    WTI's last trading day is 3 business days before the 25th of the month
    before delivery (4 if the 25th is not a business day). TradingView moves
    MCL1! to the next month 2 business days before that (1 in February),
    which matched every roll from 2024-10 to 2026-09 against individual
    contracts. Sessions are counted from the data: the series dates the
    session after an exchange holiday with the holiday's date (e.g. a
    2025-01-20 bar and no 2025-01-21), so the count is right, but whether the
    25th is a business day comes from ``us_exchange_holidays``. A month the
    data ends before the 25th of is skipped.
    """
    dates = np.asarray(dates, dtype=str)
    mask = np.zeros(dates.size, dtype=bool)
    if dates.size == 0:
        return mask
    years = range(int(dates[0][:4]), int(dates[-1][:4]) + 1)
    holidays = us_exchange_holidays(years)
    for m in np.unique([d[:7] for d in dates]):
        t25 = f"{m}-25"
        before = int(np.searchsorted(dates, t25))  # sessions dated before the 25th
        if before >= dates.size:
            continue
        ltd = before - (3 if np.is_busday(t25, holidays=holidays) else 4)
        roll = ltd - (1 if m.endswith("-02") else 2)
        if roll >= 1:
            mask[roll] = True
    return mask


def roll_mask(dates: Sequence[str], spec: "ContractSpec") -> np.ndarray:
    """Roll days of TradingView's continuous ``1!`` series for ``spec``."""
    if spec.roll_rule == "gold":
        return gold_roll_mask(dates)
    if spec.roll_rule == "crude":
        return crude_roll_mask(dates)
    if spec.roll_rule == "quarterly":
        return quarterly_roll_mask(dates, spec.roll_sessions)
    raise ValueError(f"unknown roll rule {spec.roll_rule!r}")


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
