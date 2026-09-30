"""Opening range breakout on MES / MNQ 5-minute bars, scored as a prop firm evaluation.

Run from the repo root (bundled data: regular-session 5-min bars of the front
contract, 2024-10-01 to 2026-09-29):

    python -m examples.orb_backtest
    python -m examples.orb_backtest --symbols MNQ --risk 100,200 --split 2025-07-01

1. Runs a grid of opening-range lengths (5/15/30 min) and exits (1R, 2R, hold
   to 15:55 ET) with a fixed dollar risk per trade (``--grid-risk``), and
   reports each on the whole sample and on both sides of ``--split``. Days
   whose range is too wide to risk one contract are skipped, so the risk
   level also acts as a filter on range width.
2. Picks the grid point with the best Sharpe *before* the split only, and
   reports how it did after the split, which it never saw. With 9 variants
   tried, a good in-sample result alone proves little.
3. Scores that choice as a prop evaluation (default: $50k account, $3,000
   target, $2,000 end-of-day trailing drawdown that stops at the starting
   balance) for several risk-per-trade levels, next to a zero-edge baseline:
   the same trades with their average profit subtracted, so identical
   volatility and shape but no edge. Beating the baseline is what matters;
   the pass rate alone mostly reflects the rules. Rows marked "after" use
   only sessions the configuration was not chosen on.

See ``hmm_trader/orb.py`` for the exact trade rules and fill assumptions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from hmm_trader import futures as fu
from hmm_trader import orb
from hmm_trader import prop_firm as pf

DATA_DIR = Path(__file__).parent / "data"
GRID = [(r, t) for r in (5, 15, 30) for t in (1.0, 2.0, None)]


def load(symbol: str) -> orb.IntradayBars:
    return orb.load_intraday(DATA_DIR / f"{symbol.lower()}_5min_rth.csv.gz")


def stats_row(label: str, t: orb.ORBTrades, split: str) -> str:
    s = orb.summary(t)
    pre, post = t.date < split, t.date >= split
    return (f"  {label:<18}{s['trades']:>7.0f}{s['win_rate']:>7.0%}{s['avg_r']:>+8.3f}"
            f"{s['net']:>+10,.0f}{s['max_dd']:>9,.0f}{s['profit_factor']:>6.2f}"
            f"{s['sharpe']:>+7.2f}{t.pnl[pre].sum():>+10,.0f}{t.pnl[post].sum():>+10,.0f}")


def sharpe(pnl: np.ndarray) -> float:
    sd = pnl.std(ddof=1) if pnl.size > 1 else 0.0
    return float(pnl.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0


def zero_edge(steps: np.ndarray, traded: np.ndarray) -> np.ndarray:
    """Remove the average profit per trade from the final step of traded days."""
    out = steps.copy()
    out[traded, 2] -= steps[traded].sum(axis=1).mean()
    return out


def block_bootstrap_index(n: int, n_sims: int, horizon: int, block: int,
                          rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=(n_sims, -(-horizon // block), 1))
    return ((starts + np.arange(block)) % n).reshape(n_sims, -1)[:, :horizon]


def challenge_row(label: str, res: pf.ChallengeResult) -> str:
    median = res.days_to_pass()[50]
    return (f"  {label:<34}{res.pass_rate:>7.1%}{res.rate(pf.FAILED_DRAWDOWN):>8.1%}"
            f"{res.rate(pf.FAILED_DAILY_LOSS):>8.1%}{res.rate(pf.TIMED_OUT):>9.1%}"
            f"{'-' if np.isnan(median) else f'{median:.0f}':>7}")


def run_symbol(symbol: str, args, rules: pf.ChallengeRules) -> None:
    spec = fu.CONTRACTS[symbol]
    bars = load(symbol)
    base = dict(risk_dollars=args.grid_risk, slippage_ticks=args.slippage_ticks,
                commission_per_side=args.commission)
    print(f"\n######## {symbol}: {bars.date[0]} .. {bars.date[-1]}, "
          f"{np.unique(bars.date).size} sessions, ${args.grid_risk:,.0f} risk per trade ########")
    print(f"  {'':<18}{'trades':>7}{'win':>7}{'avg R':>8}{'net $':>10}{'max DD':>9}"
          f"{'PF':>6}{'Sharpe':>7}{'< ' + args.split:>10}{'>= ' + args.split:>10}")
    results = {}
    for rng_min, target in GRID:
        p = orb.ORBParams(range_minutes=rng_min, target_r=target, **base)
        t = orb.backtest(bars, spec, p)
        results[(rng_min, target)] = t
        print(stats_row(p.label(), t, args.split))

    pre_sharpe = {k: sharpe(t.pnl[t.date < args.split]) for k, t in results.items()}
    best = max(pre_sharpe, key=pre_sharpe.get)
    t = results[best]
    post = t.date >= args.split
    label = orb.ORBParams(range_minutes=best[0], target_r=best[1]).label()
    r_post = t.pnl[post & t.traded] / (t.risk_points[post & t.traded]
                                        * t.contracts[post & t.traded] * spec.multiplier)
    se = r_post.std(ddof=1) / np.sqrt(r_post.size) if r_post.size > 1 else float("nan")
    print(f"\n  Chosen on data before {args.split}: {label} (Sharpe {pre_sharpe[best]:+.2f} there)."
          f"\n  After {args.split} (unseen): {r_post.size} trades, avg R {r_post.mean():+.3f}"
          f" +/- {se:.3f} (1 s.e.), net ${t.pnl[post].sum():+,.0f}, "
          f"Sharpe {sharpe(t.pnl[post]):+.2f}")

    rng = np.random.default_rng(args.seed)
    print(f"\n  Prop evaluation, {label}, {args.horizon} sessions max. 'all' includes the "
          f"data the\n  config was chosen on; 'after' resamples only sessions from {args.split}:")
    print(f"  {'':<34}{'pass':>7}{'max DD':>8}{'daily':>8}{'timeout':>9}{'days':>7}")
    for risk in args.risk:
        p = orb.ORBParams(range_minutes=best[0], target_r=best[1],
                          **{**base, "risk_dollars": risk})
        tr = orb.backtest(bars, spec, p)
        n = tr.date.size
        if n < args.horizon:
            print(f"  (only {n} sessions; lower --horizon)")
            return
        after = np.flatnonzero(tr.date >= args.split)
        idx = block_bootstrap_index(n, args.sims, args.horizon, args.block, rng)
        idx_after = after[block_bootstrap_index(after.size, args.sims, args.horizon,
                                                args.block, rng)]
        frac = tr.steps / args.account
        windows = np.lib.stride_tricks.sliding_window_view(frac, args.horizon, axis=0)
        null = zero_edge(tr.steps, tr.traded) / args.account
        rows = (
            ("every start, all", windows.transpose(0, 2, 1)),
            ("bootstrap, all", frac[idx]),
            ("bootstrap, after", frac[idx_after]),
            ("zero-edge bootstrap, all", null[idx]),
        )
        for name, paths in rows:
            res = pf.simulate_challenge(paths, rules, compounding=False)
            print(challenge_row(f"${risk:,.0f} risk, {name}", res))


def parse_floats(text: str) -> list[float]:
    vals = [float(x) for x in text.split(",")]
    if any(v <= 0 for v in vals):
        raise argparse.ArgumentTypeError("values must be positive")
    return vals


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbols", default="MES,MNQ")
    ap.add_argument("--risk", type=parse_floats, default=[100.0, 200.0, 400.0],
                    help="dollars risked per trade in the prop evaluation")
    ap.add_argument("--grid-risk", type=float, default=200.0,
                    help="dollars risked per trade in the grid (days too wide to trade "
                         "one contract at this risk are skipped)")
    ap.add_argument("--split", default="2025-10-01",
                    help="choose the grid point before this date, test after it")
    ap.add_argument("--slippage-ticks", type=float, default=1.0)
    ap.add_argument("--commission", type=float, default=0.62, help="per contract per side")
    rules = ap.add_argument_group("prop firm rules (dollars)")
    rules.add_argument("--account", type=float, default=50_000)
    rules.add_argument("--target", type=float, default=3_000)
    rules.add_argument("--max-dd", type=float, default=2_000)
    rules.add_argument("--dd-type", choices=pf.DRAWDOWN_TYPES, default="trailing_eod")
    rules.add_argument("--daily-loss", type=float, default=0, help="0 = none")
    sim = ap.add_argument_group("simulation")
    sim.add_argument("--horizon", type=int, default=90, help="sessions allowed to pass")
    sim.add_argument("--sims", type=int, default=5000)
    sim.add_argument("--block", type=int, default=10)
    sim.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    for s in symbols:
        if s not in fu.CONTRACTS or not (DATA_DIR / f"{s.lower()}_5min_rth.csv.gz").exists():
            ap.error(f"no bundled 5-min data for {s}")
    a = args.account
    prop = pf.ChallengeRules(
        profit_target=args.target / a, max_drawdown=args.max_dd / a,
        daily_loss_limit=args.daily_loss / a if args.daily_loss else None,
        drawdown_type=args.dd_type, trailing_cap=0.0,
    )
    print(f"Prop rules on a ${a:,.0f} account: {prop.describe()}")
    for s in symbols:
        run_symbol(s, args, prop)


if __name__ == "__main__":
    main()
