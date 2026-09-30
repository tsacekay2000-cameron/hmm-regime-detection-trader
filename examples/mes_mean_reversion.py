"""Mean reversion on MES: a daily RSI(2) pullback and an intraday Bollinger fade.

Run from the repo root:  python -m examples.mes_mean_reversion

Both rule sets are fixed in advance (see ``hmm_trader.mean_reversion``):

A. Daily RSI(2), 1 contract, long only (Connors & Alvarez, 2008): buy at the
   close when RSI(2) < 10 and the close is above its 200-day average, sell at
   the first close above the 5-day average. Published in 2008, so the whole
   2019-2026 sample is out-of-sample for these rules. It holds overnight, so
   it only fits prop accounts that allow that. S&P futures rose a lot in this
   period, so any long strategy looks good: a permutation test compares the
   trades with randomly timed trades of the same lengths, taken on days the
   200-day filter allowed, to see whether the RSI(2) timing adds anything.

C. (``--part c``) RSI(2) for prop firms that require being flat through the
   daily maintenance break: the same signals, but each held session is
   bought at its open (18:00 ET the evening before) and sold at its close,
   paying a round trip every session (V1). V2 holds only 09:30-15:55 ET on
   the same days, from the 5-minute data (MES/MNQ, 2024-10 onward only).

B. Intraday Bollinger fade on 5-minute bars, $200 risk per trade: fade a
   close outside the 20-bar +/- 2 sd band at the next open, stop 2 sd away,
   exit on a close back through the average or at 15:55 ET. Flat every
   night. Reported in R with a split by period, a k x lookback sensitivity
   grid, and a gross (no costs) comparison.

Both are scored as a $50k prop evaluation ($3,000 target, $2,000 end-of-day
trailing drawdown that stops at the starting balance) next to a zero-edge
baseline: the same daily P&L paths with their average profit removed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from examples.micro_futures_backtest import session_steps
from examples.orb_backtest import block_bootstrap_index, challenge_row
from hmm_trader import futures as fu
from hmm_trader import mean_reversion as mr
from hmm_trader import orb
from hmm_trader import prop_firm as pf

DATA_DIR = Path(__file__).parent / "data"
SPEC = fu.CONTRACTS["MES"]
COST_PER_SIDE = 0.62 + SPEC.tick_value  # commission + 1 tick slippage
RULES = pf.ChallengeRules(profit_target=0.06, max_drawdown=0.04, daily_loss_limit=None,
                          drawdown_type="trailing_eod", trailing_cap=0.0)
ACCOUNT = 50_000


def max_drawdown(daily: np.ndarray) -> float:
    equity = np.r_[0.0, np.cumsum(daily)]
    return float(np.max(np.maximum.accumulate(equity) - equity))


def demean(steps: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Zero-edge copy: subtract the mean P&L of active days from their last step."""
    out = steps.copy()
    out[active, -1] -= steps[active].sum(axis=1).mean()
    return out


def prop_rows(label: str, steps: np.ndarray, active: np.ndarray, args,
              rng: np.random.Generator, history: bool = True) -> None:
    frac, null = steps / ACCOUNT, demean(steps, active) / ACCOUNT
    idx = block_bootstrap_index(steps.shape[0], args.sims, args.horizon, 10, rng)
    rows = []
    if history:
        win = np.lib.stride_tricks.sliding_window_view(frac, args.horizon, axis=0)
        rows.append(("every start", win.transpose(0, 2, 1)))
    rows += [("bootstrap", frac[idx]), ("zero-edge bootstrap", null[idx])]
    for name, paths in rows:
        res = pf.simulate_challenge(paths, RULES, compounding=False)
        print(challenge_row(f"{label}, {name}", res))


def trade_pnl(steps: np.ndarray, spans) -> np.ndarray:
    """P&L per trade. Includes the session after the trade, where
    ``session_steps`` charges the exit side of a position held overnight."""
    return np.array([steps[s:e + 1].sum() for s, e in spans])


def part_a(args, rng: np.random.Generator, symbol: str = "MES") -> None:
    spec = fu.CONTRACTS[symbol]
    cost = 0.62 + spec.tick_value  # commission + 1 tick slippage per side
    bars = fu.load_bars(DATA_DIR / f"{symbol.lower()}_daily.csv")
    roll = fu.quarterly_roll_mask(bars.dates, spec.roll_sessions)
    rets = fu.adjusted_returns(bars, roll)
    adj_close = bars.close[0] * np.r_[1.0, np.cumprod(1 + rets)]
    held = mr.rsi2_positions(adj_close)                       # sessions 1..n-1
    dates = bars.dates[1:]
    start = int(np.argmax(np.isfinite(mr.sma(adj_close, 200))))  # first session with a filter
    sl = slice(start, None)

    steps = session_steps(bars, held, spec, overnight=True, cost_per_side=cost)
    always = np.ones_like(held)
    bh = session_steps(bars, always, spec, overnight=True, cost_per_side=cost)
    gross = session_steps(bars, always, spec, overnight=True, cost_per_side=0.0).sum(axis=1)
    daily, bh_daily = steps.sum(axis=1)[sl], bh.sum(axis=1)[sl]
    spans = [(s, e) for s, e in mr.trade_spans(held) if s >= start]
    pnl = trade_pnl(steps, spans)
    hold = np.array([e - s for s, e in spans])
    notional = np.array([adj_close[s] for s, _ in spans]) * spec.multiplier

    print(f"== A. Daily RSI(2) < 10, close > 200-day avg, exit close > 5-day avg; "
          f"1 {symbol}, {dates[start]} .. {dates[-1]} ==")
    print(f"  {len(spans)} trades, win {np.mean(pnl > 0):.0%}, avg ${pnl.mean():+.0f} per trade, "
          f"({(pnl / notional).mean():+.2%} of notional), avg hold {hold.mean():.1f} sessions, "
          f"in market {held[sl].mean():.0%} of sessions")
    print(f"  net ${daily.sum():+,.0f}, max DD ${max_drawdown(daily):,.0f}   |   buy & hold: "
          f"net ${bh_daily.sum():+,.0f}, max DD ${max_drawdown(bh_daily):,.0f}")
    years = np.array([dates[s][:4] for s, _ in spans])
    print("  by year (trades, net $):  " + "  ".join(
        f"{y}: {np.sum(years == y)}, {pnl[years == y].sum():+,.0f}" for y in np.unique(years)))

    # Timing test: same trade lengths, random starts on days the 200-day filter allowed.
    trend_ok = adj_close > mr.sma(adj_close, 200)
    eligible = np.flatnonzero(trend_ok[:-1])            # decided at close t -> session t
    eligible = eligible[eligible >= start]
    cum = np.r_[0.0, np.cumsum(gross)]
    observed = np.mean([cum[e] - cum[s] for s, e in spans])
    sims = np.empty(args.perms)
    for j in range(args.perms):
        starts = rng.choice(eligible, size=len(spans))
        ends = np.minimum(starts + hold, gross.size)
        sims[j] = np.mean(cum[ends] - cum[starts])
    print(f"  timing test (before costs): avg ${observed:+.0f} per trade vs "
          f"${sims.mean():+.0f} for random entries of the same lengths; "
          f"p = {np.mean(sims >= observed):.3f}")

    # Buying at the signal close assumes an order at the settlement; buying at
    # the next open instead drops the first overnight gap from each trade.
    _, gap = fu.point_changes(bars, roll)
    next_open = pnl - np.array([gap[s] for s, _ in spans]) * spec.multiplier
    print(f"  entering at the next open instead: avg ${next_open.mean():+.0f} per trade, "
          f"win {np.mean(next_open > 0):.0%}, net ${next_open.sum():+,.0f}")

    print("\n  Sensitivity: trades, avg $ per trade (after costs) for RSI(2) entry level "
          "(columns) x exit average (rows)")
    entries = (5, 10, 20, 30)
    print("  " + " " * 10 + "".join(f"{'RSI<' + str(e):>16}" for e in entries))
    for exit_n in (3, 5, 10):
        cells = []
        for level in entries:
            h = mr.rsi2_positions(adj_close, entry=level, exit_sma=exit_n)
            s = session_steps(bars, h, spec, overnight=True, cost_per_side=cost)
            tp = trade_pnl(s, [(a, b) for a, b in mr.trade_spans(h) if a >= start])
            cells.append(f"{tp.size:3d}, {tp.mean():+5.0f}" if tp.size else "-")
        print(f"  exit>{exit_n}d  " + "".join(f"{c:>16}" for c in cells))
    h = mr.rsi2_positions(adj_close, trend=None)
    s = session_steps(bars, h, spec, overnight=True, cost_per_side=cost)
    tp = trade_pnl(s, [(a, b) for a, b in mr.trade_spans(h) if a >= start])
    print(f"  without the 200-day filter: {tp.size} trades, avg ${tp.mean():+.0f}, "
          f"win {np.mean(tp > 0):.0%}, net ${tp.sum():+,.0f}")

    print(f"\n  Prop evaluation (holds overnight), {args.horizon} sessions max:")
    print(f"  {'':<34}{'pass':>7}{'max DD':>8}{'daily':>8}{'timeout':>9}{'days':>7}")
    for n in (1, 2, 3):
        prop_rows(f"{n} {symbol}", steps[sl] * n, held[sl], args, rng)


def rth_open_close(symbol: str) -> dict[str, tuple[float, float]]:
    """Trade date -> (09:30 open, 15:55 bar close) from the bundled 5-min data."""
    path = DATA_DIR / f"{symbol.lower()}_5min_rth.csv.gz"
    if not path.exists():
        return {}
    bars = orb.load_intraday(path)
    out = {}
    for date, sl in bars.days():
        minute = bars.minute[sl]
        if minute[0] == 9 * 60 + 30 and minute[-1] == 15 * 60 + 55:
            out[str(date)] = (bars.open[sl][0], bars.close[sl][-1])
    return out


def part_c(args, rng: np.random.Generator, symbol: str) -> None:
    spec = fu.CONTRACTS[symbol]
    cost = 0.62 + spec.tick_value
    bars = fu.load_bars(DATA_DIR / f"{symbol.lower()}_daily.csv")
    roll = fu.quarterly_roll_mask(bars.dates, spec.roll_sessions)
    adj_close = bars.close[0] * np.r_[1.0, np.cumprod(1 + fu.adjusted_returns(bars, roll))]
    held = mr.rsi2_positions(adj_close)
    dates = bars.dates[1:]
    start = int(np.argmax(np.isfinite(mr.sma(adj_close, 200))))
    sl = slice(start, None)
    spans = [(s, e) for s, e in mr.trade_spans(held) if s >= start]

    over = session_steps(bars, held, spec, overnight=True, cost_per_side=cost)
    flat = session_steps(bars, held, spec, overnight=False, cost_per_side=cost)
    p_over = trade_pnl(over, spans)
    p_flat = trade_pnl(flat, spans)
    days = int(held[sl].sum())
    print(f"== C. RSI(2) flat through every daily break, 1 {symbol}, "
          f"{dates[start]} .. {dates[-1]} ==")
    print(f"  Same signals: {len(spans)} trades, {days} sessions in the market.")
    print(f"  held overnight (part A):       avg ${p_over.mean():+.0f} per trade, win "
          f"{np.mean(p_over > 0):.0%}, net ${p_over.sum():+,.0f}")
    print(f"  V1 session open -> close each day: avg ${p_flat.mean():+.0f} per trade, win "
          f"{np.mean(p_flat > 0):.0%}, net ${p_flat.sum():+,.0f}, max DD "
          f"${max_drawdown(flat[sl].sum(axis=1)):,.0f} (a round trip every session)")

    gross = session_steps(bars, np.ones_like(held), spec, overnight=False,
                          cost_per_side=0.0).sum(axis=1)
    cum = np.r_[0.0, np.cumsum(gross)]
    hold = np.array([e - s for s, e in spans])
    trend_ok = adj_close > mr.sma(adj_close, 200)
    eligible = np.flatnonzero(trend_ok[:-1])
    eligible = eligible[eligible >= start]
    observed = np.mean([cum[e] - cum[s] for s, e in spans])
    sims = np.array([np.mean(cum[np.minimum(st + hold, gross.size)] - cum[st])
                     for st in (rng.choice(eligible, size=len(spans))
                                for _ in range(args.perms))])
    print(f"  V1 timing test (before costs): avg ${observed:+.0f} per trade vs "
          f"${sims.mean():+.0f} for random entries of the same lengths; "
          f"p = {np.mean(sims >= observed):.3f}")

    rth = rth_open_close(symbol)
    idx = [t for t in np.flatnonzero(held) if t >= start and str(dates[t]) in rth]
    if idx:
        o = np.array([rth[str(dates[t])][0] for t in idx])
        c = np.array([rth[str(dates[t])][1] for t in idx])
        rth_pnl = (c - o) * spec.multiplier - 2 * cost
        sess_pnl = flat[idx].sum(axis=1)
        se = rth_pnl.std(ddof=1) / np.sqrt(rth_pnl.size) if rth_pnl.size > 1 else np.nan
        print(f"  V2 09:30 -> 15:55 ET only, {dates[idx[0]]} .. {dates[idx[-1]]} "
              f"({len(idx)} held sessions with 5-min data): avg ${rth_pnl.mean():+.0f} ± "
              f"{se:.0f} per session vs ${sess_pnl.mean():+.0f} for the full session "
              f"(V1) on the same days")

    print(f"\n  Prop evaluation, V1 (flat through every break), {args.horizon} sessions max:")
    print(f"  {'':<34}{'pass':>7}{'max DD':>8}{'daily':>8}{'timeout':>9}{'days':>7}")
    for n in (1, 2, 3):
        prop_rows(f"{n} {symbol}", flat[sl] * n, held[sl], args, rng)


def part_b(args, rng: np.random.Generator) -> None:
    bars = orb.load_intraday(DATA_DIR / "mes_5min_rth.csv.gz")
    p = mr.FadeParams()
    t = mr.bollinger_fade(bars, SPEC, p)
    r = t.r_multiple
    se = lambda x: x.std(ddof=1) / np.sqrt(x.size)  # noqa: E731
    days = np.unique(t.day_date).size
    print(f"\n== B. Intraday {p.label()}, $200 risk, flat by 15:55 ET; "
          f"{t.day_date[0]} .. {t.day_date[-1]} ({days} sessions) ==")
    reasons = ", ".join(f"{k} {np.mean(t.reason == k):.0%}" for k in ("mean", "stop", "flatten"))
    print(f"  {r.size} trades ({r.size / days:.1f}/day), win {np.mean(t.pnl > 0):.0%}, "
          f"avg R {r.mean():+.3f} ± {se(r):.3f}, net ${t.pnl.sum():+,.0f}, "
          f"max DD ${max_drawdown(t.day_pnl):,.0f}; exits: {reasons}")
    for name, m in (("before " + args.split, t.date < args.split),
                    ("from " + args.split, t.date >= args.split)):
        print(f"  {name:<18} {m.sum():4d} trades, avg R {r[m].mean():+.3f} ± {se(r[m]):.3f}, "
              f"net ${t.pnl[m].sum():+,.0f}")
    g = mr.bollinger_fade(bars, SPEC, mr.FadeParams(slippage_ticks=0, commission_per_side=0))
    print(f"  before costs: avg R {g.r_multiple.mean():+.3f} "
          f"(costs take {g.r_multiple.mean() - r.mean():.3f} R per trade)")

    print("\n  Sensitivity: avg R (trades) for k (columns) x lookback (rows)")
    ks = (1.5, 2.0, 2.5, 3.0)
    print("  " + " " * 8 + "".join(f"{'k=' + format(k, 'g'):>18}" for k in ks))
    for lb in (10, 20, 40):
        cells = []
        for k in ks:
            x = mr.bollinger_fade(bars, SPEC, mr.FadeParams(lookback=lb, k=k)).r_multiple
            cells.append(f"{x.mean():+.3f} ({x.size:4d})" if x.size else "- (   0)")
        print(f"  N={lb:<6}" + "".join(f"{c:>18}" for c in cells))

    print(f"\n  Prop evaluation (flat every night), {args.horizon} sessions max:")
    print(f"  {'':<34}{'pass':>7}{'max DD':>8}{'daily':>8}{'timeout':>9}{'days':>7}")
    for risk in (200.0, 400.0):
        tr = mr.bollinger_fade(bars, SPEC, mr.FadeParams(risk_dollars=risk))
        prop_rows(f"${risk:,.0f} risk", tr.day_steps, tr.day_pnl != 0, args, rng)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=("a", "b", "c", "both", "all"), default="both",
                    help="both = A and B; all = A, B and C")
    ap.add_argument("--symbols", default="MES",
                    help="part A symbols with bundled daily data, e.g. MES,MNQ,M2K")
    ap.add_argument("--split", default="2025-10-01", help="period split for part B")
    ap.add_argument("--perms", type=int, default=5000)
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--horizon", type=int, default=90)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    print(f"Prop rules on a ${ACCOUNT:,} account: {RULES.describe()}\n")
    symbols = [x.strip().upper() for x in args.symbols.split(",")]
    for symbol in symbols:
        if symbol not in fu.CONTRACTS or not (DATA_DIR / f"{symbol.lower()}_daily.csv").exists():
            ap.error(f"no bundled daily data for {symbol}")
    if args.part in ("a", "both", "all"):
        for symbol in symbols:
            part_a(args, rng, symbol)
            print()
    if args.part in ("c", "all"):
        for symbol in symbols:
            part_c(args, rng, symbol)
            print()
    if args.part in ("b", "both", "all"):
        part_b(args, rng)


if __name__ == "__main__":
    main()
