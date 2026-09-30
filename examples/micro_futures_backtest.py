"""Walk-forward HMM regime backtest on micro futures, scored as a prop firm evaluation.

Run from the repo root (bundled MES and MNQ daily data, 2019-05 to 2026-09):

    python -m examples.micro_futures_backtest
    python -m examples.micro_futures_backtest --symbols MNQ --contracts 1,2,4 --overnight
    python -m examples.micro_futures_backtest --target 3000 --max-dd 2500 --dd-type trailing

Strategy: long N contracts when the HMM's filtered probability of the calm
regime is at least 0.5, flat otherwise. The HMM is fitted on roll-adjusted
close-to-close returns, only on data before each test block (walk-forward,
refit quarterly on an expanding window), and each session's position uses
information up to the previous close. The benchmark is long N contracts in
every session.

By default positions are opened at the session open and closed at the
settlement close, so nothing is held through the daily maintenance break, as
futures prop firms require. ``--overnight`` holds between sessions instead
(and pays roll costs), for firms or accounts that allow it.

Each session is replayed as prev close -> open -> high -> low -> close, the
worst ordering for a long under a trailing drawdown, so intraday breaches
are counted conservatively. Prop results are shown two ways: a challenge
started on every historical session of the test period (overlapping windows,
so not independent), and a block bootstrap of test-period sessions.

Data: TradingView ``CME_MINI:MES1!``/``MNQ1!`` daily bars (front contract)
with the ``2!`` close, dated by trade date; see ``hmm_trader.futures``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from examples.spy_monte_carlo import walk_forward_positions
from hmm_trader import futures as fu
from hmm_trader import prop_firm as pf

DATA_DIR = Path(__file__).parent / "data"
COMMISSION_PER_SIDE = 0.62  # typical micro round turn is ~$1.24 all-in


def load(symbol: str) -> fu.FuturesBars:
    return fu.load_bars(DATA_DIR / f"{symbol.lower()}_daily.csv")


def session_steps(
    bars: fu.FuturesBars,
    in_market: np.ndarray,
    spec: fu.ContractSpec,
    overnight: bool,
    cost_per_side: float,
) -> np.ndarray:
    """Dollar P&L of one contract per session as 4 intraday steps, shape (n-1, 4).

    ``in_market`` covers sessions 1..n-1. Costs are charged on step 0 (before
    the session's price moves), so they count against breach checks.
    """
    roll = fu.quarterly_roll_mask(bars.dates, spec.roll_sessions)
    steps = fu.intraday_long_steps(bars, roll, overnight) * spec.multiplier
    held = in_market.astype(float)
    steps *= held[:, None]
    if overnight:
        prev = np.r_[0.0, held[:-1]]
        sides = np.abs(held - prev) + 2 * held * prev * roll[1:]
    else:
        sides = 2 * held
    steps[:, 0] -= sides * cost_per_side
    return steps


def regime_in_market(bars: fu.FuturesBars, train: int, args) -> np.ndarray:
    """Walk-forward long/flat signal for sessions ``train+1 .. n-1``."""
    rets = fu.adjusted_returns(bars, fu.quarterly_roll_mask(bars.dates))
    exposure = np.r_[1.0, np.zeros(args.states - 1)]
    p_calm, _ = walk_forward_positions(rets, args.states, exposure, train,
                                       args.refit_every, args.starts)
    return p_calm >= 0.5


def history_stats(daily: np.ndarray, in_market: np.ndarray, overnight: bool) -> str:
    equity = np.r_[0.0, np.cumsum(daily)]
    max_dd = np.max(np.maximum.accumulate(equity) - equity)
    sharpe = daily.mean() / daily.std(ddof=1) * np.sqrt(252) if daily.std() > 0 else 0.0
    if overnight:  # one round turn per stretch in the market (rolls aside)
        trades = int(np.sum(in_market & ~np.r_[False, in_market[:-1]]))
    else:  # one round turn per session in the market
        trades = int(in_market.sum())
    return (f"{equity[-1]:>+11,.0f}{max_dd:>10,.0f}{daily.min():>10,.0f}"
            f"{sharpe:>8.2f}{in_market.mean():>8.0%}{trades:>8d}")


def block_bootstrap_index(n: int, n_sims: int, horizon: int, block: int,
                          rng: np.random.Generator) -> np.ndarray:
    n_blocks = -(-horizon // block)
    starts = rng.integers(0, n, size=(n_sims, n_blocks, 1))
    return ((starts + np.arange(block)) % n).reshape(n_sims, -1)[:, :horizon]


def challenge_row(label: str, res: pf.ChallengeResult) -> str:
    median = res.days_to_pass()[50]
    return (f"  {label:<22}{res.pass_rate:>7.1%}{res.rate(pf.FAILED_DRAWDOWN):>8.1%}"
            f"{res.rate(pf.FAILED_DAILY_LOSS):>8.1%}{res.rate(pf.TIMED_OUT):>9.1%}"
            f"{'-' if np.isnan(median) else f'{median:.0f}':>8}")


def build_rules(args) -> pf.ChallengeRules:
    a = args.account
    return pf.ChallengeRules(
        profit_target=args.target / a,
        max_drawdown=args.max_dd / a,
        daily_loss_limit=args.daily_loss / a if args.daily_loss else None,
        drawdown_type=args.dd_type,
        trailing_cap=None if args.no_trail_cap else 0.0,
        min_trading_days=args.min_days,
        consistency=args.consistency,
    )


def run_symbol(symbol: str, args, rules: pf.ChallengeRules) -> None:
    spec = fu.CONTRACTS[symbol]
    bars = load(symbol)
    dates = bars.dates[1:]
    train = int(np.searchsorted(dates, args.test_start))
    if not 0 < train < dates.size - args.horizon:
        raise SystemExit(f"--test-start must leave training data and at least "
                         f"{args.horizon} test sessions")
    cost = COMMISSION_PER_SIDE + args.slippage_ticks * spec.tick_value

    regime = regime_in_market(bars, train, args)
    always = np.ones(dates.size - train, dtype=bool)
    strategies = {}
    for name, sig in (("always long", always), ("regime filter", regime)):
        full = np.zeros(dates.size, dtype=bool)
        full[train:] = sig
        strategies[name] = (session_steps(bars, full, spec, args.overnight, cost)[train:], sig)

    mode = "held overnight" if args.overnight else "flat between sessions"
    print(f"\n######## {symbol} (${spec.multiplier:g}/pt, cost ${cost:.2f}/side, {mode}) ########")
    print(f"train {bars.dates[0]} .. {dates[train - 1]}, "
          f"test {dates[train]} .. {dates[-1]} ({dates.size - train} sessions), "
          f"refit every {args.refit_every or 'never'}")

    print(f"\nHistorical test period, 1 contract, after costs:")
    print(f"  {'':<16}{'net P&L $':>11}{'max DD $':>10}{'worst day':>10}"
          f"{'Sharpe':>8}{'in mkt':>8}{'trades':>8}")
    for name, (steps, sig) in strategies.items():
        print(f"  {name:<16}" + history_stats(steps.sum(axis=1), sig, args.overnight))

    header = (f"  {'':<22}{'pass':>7}{'max DD':>8}{'daily':>8}{'timeout':>9}{'days':>8}")
    rng = np.random.default_rng(args.seed)
    n_test = dates.size - train
    boot_idx = block_bootstrap_index(n_test, args.sims, args.horizon, args.block, rng)
    for title, make in (
        (f"Challenge started on every test session ({n_test - args.horizon + 1} "
         f"overlapping windows)",
         lambda s: np.lib.stride_tricks.sliding_window_view(s, args.horizon, axis=0)
         .transpose(0, 2, 1)),
        (f"Block bootstrap of test sessions ({args.sims} paths, {args.block}-day blocks)",
         lambda s: s[boot_idx]),
    ):
        print(f"\n{title}, {args.horizon} sessions max:")
        print(header)
        for name, (steps, _) in strategies.items():
            paths = make(steps) / args.account
            for n in args.contracts:
                res = pf.simulate_challenge(paths, rules, leverage=n, compounding=False)
                print(challenge_row(f"{name} x{n}", res))


def parse_contracts(text: str) -> list[int]:
    vals = [int(x) for x in text.split(",")]
    if any(v < 1 for v in vals):
        raise argparse.ArgumentTypeError("contract counts must be >= 1")
    return vals


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbols", default="MES,MNQ", help="comma-separated, from bundled data")
    ap.add_argument("--contracts", type=parse_contracts, default=[1, 2, 3, 5],
                    help="contract counts to test, e.g. 1,2,3,5")
    ap.add_argument("--overnight", action="store_true",
                    help="hold between sessions instead of flat at each close")
    ap.add_argument("--slippage-ticks", type=float, default=1.0, help="per side")
    ap.add_argument("--test-start", default="2022-01-01")
    ap.add_argument("--refit-every", type=int, default=63)
    ap.add_argument("--states", type=int, default=2)
    ap.add_argument("--starts", type=int, default=10, help="random restarts per HMM fit")
    rules = ap.add_argument_group("prop firm rules (dollars)")
    rules.add_argument("--account", type=float, default=50_000)
    rules.add_argument("--target", type=float, default=3_000)
    rules.add_argument("--max-dd", type=float, default=2_000)
    rules.add_argument("--dd-type", choices=pf.DRAWDOWN_TYPES, default="trailing_eod")
    rules.add_argument("--no-trail-cap", action="store_true",
                       help="keep trailing above the starting balance")
    rules.add_argument("--daily-loss", type=float, default=0, help="0 = none")
    rules.add_argument("--min-days", type=int, default=0)
    rules.add_argument("--consistency", type=float, help="e.g. 0.5 = best day <= 50%%")
    mcg = ap.add_argument_group("simulation")
    mcg.add_argument("--horizon", type=int, default=90, help="sessions allowed to pass")
    mcg.add_argument("--sims", type=int, default=5000)
    mcg.add_argument("--block", type=int, default=10, help="bootstrap block, sessions")
    mcg.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    for s in symbols:
        if s not in fu.CONTRACTS or not (DATA_DIR / f"{s.lower()}_daily.csv").exists():
            ap.error(f"no bundled data for {s}")
    prop = build_rules(args)
    print(f"Prop rules on a ${args.account:,.0f} account: {prop.describe()}")
    for s in symbols:
        run_symbol(s, args, prop)


if __name__ == "__main__":
    main()
