"""Prop firm challenge simulation: pass probability under evaluation rules.

A prop firm evaluation is a barrier problem: the account must reach a profit
target before it breaches a max drawdown or a daily loss limit. Average return
matters less than how the path gets there, so this module replays many
simulated return paths through a set of challenge rules and reports how often
each one passes, which rule fails it, and how long passing takes.

Feed it any ``(n_sims, n_days)`` array of per-day simple returns, e.g. from
``monte_carlo.block_bootstrap_paths`` or ``monte_carlo.simulate_hmm_strategy``.
Pass ``(n_sims, n_days, steps_per_day)`` for intraday returns so the daily loss
limit and a real-time trailing drawdown see intraday dips; with daily returns,
breaches are only detected on the daily close, which understates failures.

All limits are fractions of the initial balance, as most firms quote them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

DRAWDOWN_TYPES = ("static", "trailing", "trailing_eod")

PASSED = "passed"
FAILED_DRAWDOWN = "max_drawdown"
FAILED_DAILY_LOSS = "daily_loss"
TIMED_OUT = "timed_out"
OUTCOMES = (PASSED, FAILED_DRAWDOWN, FAILED_DAILY_LOSS, TIMED_OUT)


@dataclass(frozen=True)
class ChallengeRules:
    """Evaluation rules, all as fractions of the initial balance.

    ``drawdown_type``:
      - ``"static"``        floor fixed at ``1 - max_drawdown``
      - ``"trailing"``      floor trails the highest equity seen, step by step
      - ``"trailing_eod"``  floor trails the highest end-of-day balance

    ``trailing_cap`` is the highest the trailing floor may rise, as a return on
    the initial balance: ``0.0`` stops trailing once the floor reaches the
    starting balance, as many futures firms do. ``None`` trails forever.

    ``daily_loss_limit`` is a loss of that fraction of the initial balance,
    measured from the start-of-day balance.
    ``min_trading_days`` counts only days with a nonzero return (days the
    strategy sat flat do not count). ``max_days`` ends the challenge as timed
    out; ``None`` means no time limit beyond the simulated horizon.

    ``consistency`` caps the best single day's profit as a share of the total
    profit, e.g. ``0.5`` means no day may exceed half the profit. It does not
    fail the account; like at most firms, you keep trading until it holds.
    """

    profit_target: float = 0.10
    max_drawdown: float = 0.10
    daily_loss_limit: Optional[float] = 0.05
    drawdown_type: str = "static"
    trailing_cap: Optional[float] = None
    min_trading_days: int = 0
    max_days: Optional[int] = None
    consistency: Optional[float] = None

    def __post_init__(self) -> None:
        if self.profit_target <= 0:
            raise ValueError("profit_target must be positive")
        if not 0 < self.max_drawdown < 1:
            raise ValueError("max_drawdown must be between 0 and 1")
        if self.daily_loss_limit is not None and not 0 < self.daily_loss_limit < 1:
            raise ValueError("daily_loss_limit must be between 0 and 1")
        if self.drawdown_type not in DRAWDOWN_TYPES:
            raise ValueError(f"drawdown_type must be one of {DRAWDOWN_TYPES}")
        if self.min_trading_days < 0:
            raise ValueError("min_trading_days must be >= 0")
        if self.max_days is not None and self.max_days < 1:
            raise ValueError("max_days must be >= 1")
        if self.consistency is not None and not 0 < self.consistency <= 1:
            raise ValueError("consistency must be in (0, 1]")

    def describe(self) -> str:
        parts = [
            f"target +{self.profit_target:.1%}",
            f"max DD {self.max_drawdown:.1%} ({self.drawdown_type}"
            + (f", capped at {self.trailing_cap:+.1%}" if self.trailing_cap is not None
               and self.drawdown_type != "static" else "")
            + ")",
        ]
        if self.daily_loss_limit is not None:
            parts.append(f"daily loss {self.daily_loss_limit:.1%}")
        if self.min_trading_days:
            parts.append(f"min days {self.min_trading_days}")
        if self.max_days is not None:
            parts.append(f"max days {self.max_days}")
        if self.consistency is not None:
            parts.append(f"best day <= {self.consistency:.0%} of profit")
        return " | ".join(parts)


@dataclass
class ChallengeResult:
    """Per-path outcome of a simulated challenge.

    ``days`` is the day the outcome was decided (1-based); for timed-out paths
    it is the number of days simulated.
    """

    outcomes: np.ndarray
    days: np.ndarray
    final_equity: np.ndarray
    rules: ChallengeRules
    percentiles: tuple[int, ...] = field(default=(25, 50, 75))

    @property
    def n_sims(self) -> int:
        return self.outcomes.size

    def rate(self, outcome: str) -> float:
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        return float(np.mean(self.outcomes == outcome))

    @property
    def pass_rate(self) -> float:
        return self.rate(PASSED)

    def days_to_pass(self) -> dict[int, float]:
        """Percentiles of days needed, among paths that passed (NaN if none did)."""
        passed = self.days[self.outcomes == PASSED]
        if passed.size == 0:
            return {p: float("nan") for p in self.percentiles}
        return dict(zip(self.percentiles, np.percentile(passed, self.percentiles)))

    def report(self) -> str:
        labels = {
            PASSED: "passed",
            FAILED_DRAWDOWN: "failed: max drawdown",
            FAILED_DAILY_LOSS: "failed: daily loss",
            TIMED_OUT: "timed out",
        }
        lines = [f"Prop challenge: {self.n_sims} simulations", f"  {self.rules.describe()}"]
        lines += [f"{labels[o]:<22}{self.rate(o):>7.1%}" for o in OUTCOMES]
        days = self.days_to_pass()
        pct = " / ".join(f"p{p}" for p in days)
        vals = " / ".join("-" if np.isnan(v) else f"{v:.0f}" for v in days.values())
        lines.append(f"days to pass ({pct}): {vals}")
        return "\n".join(lines)


def simulate_challenge(
    paths: np.ndarray,
    rules: ChallengeRules,
    leverage: float = 1.0,
    compounding: bool = True,
) -> ChallengeResult:
    """Run each return path through the challenge rules.

    ``paths`` is ``(n_sims, n_days)`` of per-day simple returns, or
    ``(n_sims, n_days, steps_per_day)`` for intraday returns. ``leverage``
    scales every return, so sweeping it shows how position size trades off
    pass rate against blow-ups.

    With ``compounding=False`` each value is P&L as a fraction of the initial
    balance and simply adds up, which is right for a fixed number of futures
    contracts (e.g. $150 on a $50,000 account is ``0.003``).

    Breaches are checked after every step, using the equity at that step (a
    step that gaps through the floor still fails). The profit target, minimum
    trading days and consistency rule are checked on each day's close, after
    which the path stops trading.
    """
    paths = np.asarray(paths, dtype=float)
    if paths.ndim == 2:
        paths = paths[:, :, None]
    if paths.ndim != 3 or paths.size == 0:
        raise ValueError("paths must be (n_sims, n_days) or (n_sims, n_days, steps_per_day)")
    if not np.all(np.isfinite(paths)):
        raise ValueError("paths must be finite")
    if leverage <= 0:
        raise ValueError("leverage must be positive")
    paths = paths * leverage
    if compounding:
        paths = np.maximum(paths, -1.0)

    n_sims, n_days, _ = paths.shape
    if rules.max_days is not None:
        n_days = min(n_days, rules.max_days)

    equity = np.ones(n_sims)
    peak = np.ones(n_sims)
    active = np.ones(n_sims, dtype=bool)
    outcomes = np.full(n_sims, TIMED_OUT, dtype=object)
    days = np.full(n_sims, n_days)
    traded_days = np.zeros(n_sims, dtype=int)
    best_day = np.full(n_sims, -np.inf)
    floor_cap = np.inf if rules.trailing_cap is None else 1.0 + rules.trailing_cap

    def floor() -> np.ndarray | float:
        if rules.drawdown_type == "static":
            return 1.0 - rules.max_drawdown
        return np.minimum(peak - rules.max_drawdown, floor_cap)

    def finish(mask: np.ndarray, outcome: str, day: int) -> None:
        outcomes[mask] = outcome
        days[mask] = day
        active[mask] = False

    for d in range(n_days):
        day_start = equity.copy()
        traded = np.zeros(n_sims, dtype=bool)
        for r in paths[:, d].T:
            step = equity * r if compounding else r
            equity = np.where(active, equity + step, equity)
            traded |= active & (r != 0)
            if rules.drawdown_type == "trailing":
                peak = np.where(active, np.maximum(peak, equity), peak)
            finish(active & (equity <= floor()), FAILED_DRAWDOWN, d + 1)
            if rules.daily_loss_limit is not None:
                hit = active & (day_start - equity >= rules.daily_loss_limit)
                finish(hit, FAILED_DAILY_LOSS, d + 1)

        traded_days += traded
        best_day = np.where(active, np.maximum(best_day, equity - day_start), best_day)
        if rules.drawdown_type == "trailing_eod":
            peak = np.where(active, np.maximum(peak, equity), peak)
        profit = equity - 1.0
        passed = active & (profit >= rules.profit_target)
        passed &= traded_days >= rules.min_trading_days
        if rules.consistency is not None:
            passed &= best_day <= rules.consistency * profit
        finish(passed, PASSED, d + 1)
        if not active.any():
            break

    return ChallengeResult(outcomes.astype(str), days, equity, rules)
