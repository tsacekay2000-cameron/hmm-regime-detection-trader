"""Does an opening range breakout work better on narrow-range days?

Run from the repo root:  python -m examples.orb_narrow_range

The idea came from the MNQ results of ``examples.orb_backtest``: at $200 risk
the 30-minute ORB only fit on narrow-range days, and those trades looked good.
That observation was made on this same data, so re-testing it on MNQ alone
would be circular. The rules below are fixed before looking at any result:

- Filter: trade only when today's opening range is at most ``k`` times the
  median range of the previous ``N`` sessions (k = 1.0, N = 20). This is a
  relative, causal rule, not the accidental dollar cap from before.
- Primary test: MNQ, 30-minute range, 2R exit. 1R and end-of-day exits are
  shown too. MES, MGC and MCL are the out-of-sample checks: nobody looked at
  their narrow-range results before this was written.
- Risk is $1,000 per trade so the size cap almost never skips a day; results
  are in R (P&L per dollar risked), so the risk level does not matter.

Evidence reported per symbol and exit:

1. average R on narrow days, wide days and all days;
2. a permutation p-value: how often a random subset of the same number of
   traded days does at least as well as the narrow days (one-sided);
3. MNQ by year (before / after 2025-10-01);
4. a robustness grid over k and N for the primary exit, since a real effect
   should not live at one parameter value only;
5. a prop evaluation of the filtered MNQ strategy against the zero-edge
   baseline.

With 4 symbols x 3 exits there are 12 tests: a single p near 0.05 is
expected by chance (Bonferroni threshold 0.05 / 12 = 0.004).
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np

from examples.orb_backtest import (DATA_DIR, SESSIONS, block_bootstrap_index, challenge_row,
                                   load, zero_edge)
from hmm_trader import futures as fu
from hmm_trader import orb
from hmm_trader import prop_firm as pf

EXITS = (("1R", 1.0), ("2R", 2.0), ("EOD", None))


def params(symbol: str, target, risk=1000.0, **kw) -> orb.ORBParams:
    session_open, flatten = SESSIONS[symbol]
    return orb.ORBParams(range_minutes=30, target_r=target, risk_dollars=risk,
                         session_open=session_open, flatten=flatten, **kw)


def narrow_mask(width: np.ndarray, k: float, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """(eligible, narrow) per day: eligible once ``lookback`` prior days exist."""
    eligible = np.arange(width.size) >= lookback
    narrow = np.zeros(width.size, dtype=bool)
    for i in np.flatnonzero(eligible):
        narrow[i] = width[i] <= k * np.median(width[i - lookback:i])
    return eligible, narrow


def r_by_day(t: orb.ORBTrades) -> np.ndarray:
    """R multiple per day (NaN when no trade)."""
    r = np.full(t.date.size, np.nan)
    r[t.traded] = t.r_multiple
    return r


def perm_p(r: np.ndarray, pick: np.ndarray, rng: np.random.Generator, n_perm: int) -> float:
    """One-sided p: share of random same-size subsets with mean >= the picked mean."""
    pool = r[~np.isnan(r)]
    chosen = r[pick & ~np.isnan(r)]
    if chosen.size == 0 or chosen.size >= pool.size:
        return float("nan")
    idx = np.argsort(rng.random((n_perm, pool.size)), axis=1)[:, :chosen.size]
    return float(np.mean(pool[idx].mean(axis=1) >= chosen.mean()))


def fmt(x: np.ndarray) -> str:
    x = x[~np.isnan(x)]
    if x.size < 2:
        return f"{'-':>22}"
    return f"{x.mean():>+7.3f} ±{x.std(ddof=1) / np.sqrt(x.size):.3f} ({x.size:3d})"


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--split", default="2025-10-01")
    ap.add_argument("--perms", type=int, default=5000)
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--horizon", type=int, default=90)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    k, n = args.k, args.lookback

    print(f"Narrow day: 30-min range <= {k:g} x median of the previous {n} sessions.")
    print("avg R ± 1 s.e. (trades); p = share of random same-size subsets doing as well\n")
    print(f"  {'':<10}{'all eligible days':>22}{'narrow days':>24}{'wide days':>24}{'p':>7}")
    runs = {}
    for symbol in ("MNQ", "MES", "MGC", "MCL"):
        bars, spec = load(symbol), fu.CONTRACTS[symbol]
        for name, target in EXITS:
            t = orb.backtest(bars, spec, params(symbol, target))
            eligible, narrow = narrow_mask(t.range_points, k, n)
            r = r_by_day(t)
            runs[(symbol, name)] = (t, r, eligible, narrow)
            p = perm_p(np.where(eligible, r, np.nan), narrow, rng, args.perms)
            tag = " <- primary" if (symbol, name) == ("MNQ", "2R") else ""
            print(f"  {symbol + ' ' + name:<10}{fmt(r[eligible]):>24}{fmt(r[narrow]):>24}"
                  f"{fmt(r[eligible & ~narrow]):>24}{p:>7.3f}{tag}")
        skipped = int((t.reason == "too_wide").sum())
        if skipped:
            print(f"  ({symbol}: {skipped} days too wide even at $1,000 risk)")

    print(f"\nMNQ narrow days by period (before / from {args.split}):")
    for name, _ in EXITS:
        t, r, eligible, narrow = runs[("MNQ", name)]
        pre, post = t.date < args.split, t.date >= args.split
        print(f"  {name:<5}{fmt(r[narrow & pre]):>24}{fmt(r[narrow & post]):>24}")

    print("\nRobustness, 2R exit: narrow-day avg R for k (columns) x lookback (rows)")
    ks, lookbacks = (0.6, 0.8, 1.0, 1.2, 1.5), (10, 20, 40)
    for symbol in ("MNQ", "MES"):
        t, r, _, _ = runs[(symbol, "2R")]
        print(f"  {symbol:<6}" + "".join(f"{'k=' + format(x, 'g'):>16}" for x in ks))
        for lb in lookbacks:
            cells = []
            for kk in ks:
                _, nm = narrow_mask(t.range_points, kk, lb)
                x = r[nm & ~np.isnan(r)]
                cells.append(f"{x.mean():+.3f} ({x.size:3d})" if x.size else "-")
            print(f"  N={lb:<4}" + "".join(f"{c:>16}" for c in cells))

    rules = pf.ChallengeRules(profit_target=0.06, max_drawdown=0.04, daily_loss_limit=None,
                              drawdown_type="trailing_eod", trailing_cap=0.0)
    print(f"\nProp evaluation, MNQ 30m / 2R narrow-range filter, $50k: {rules.describe()}")
    print(f"  {'':<34}{'pass':>7}{'max DD':>8}{'daily':>8}{'timeout':>9}{'days':>7}")
    bars, spec = load("MNQ"), fu.CONTRACTS["MNQ"]
    for risk in (200.0, 400.0):
        t = orb.backtest(bars, spec, params("MNQ", 2.0, risk=risk, max_range_ratio=k,
                                            range_lookback=n))
        idx = block_bootstrap_index(t.date.size, args.sims, args.horizon, 10, rng)
        for label, steps in (("bootstrap", t.steps),
                             ("zero-edge bootstrap", zero_edge(t.steps, t.traded))):
            res = pf.simulate_challenge(steps[idx] / 50_000, rules, compounding=False)
            print(challenge_row(f"${risk:,.0f} risk, {label}", res))


if __name__ == "__main__":
    main()
