"""Fit a Gaussian HMM to real daily prices and run Monte Carlo on a regime strategy.

Run from the repo root (defaults to the bundled SPY sample):

    python -m examples.spy_monte_carlo
    python -m examples.spy_monte_carlo my_prices.csv --states 3 --exposure 1,0.5,0
    python -m examples.spy_monte_carlo my_prices.csv --walk-forward --test-start 2022-01-01

Input: a CSV with a ``close`` (or ``adj_close``) column and optional ``date``
column, or a plain text file with one close per line (``#`` comments allowed),
oldest first.

Sections A-B resample the actual buy & hold and strategy returns; C-D trade
fresh market paths sampled from the fitted HMM. By default the HMM is fitted on
the same data the historical strategy trades, so A-B are in-sample.

With ``--walk-forward`` the HMM is fitted only on data before the test period
and refitted every ``--refit-every`` periods on an expanding window, so every
position uses information available at the time. A-B then cover only the
out-of-sample test period, and C-D use the model fitted on the training data.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from hmm_trader import monte_carlo as mc

DEFAULT_DATA = Path(__file__).parent / "data" / "spy_daily_close.txt"


def load_prices(path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (closes, dates); dates is None when the file has no date column."""
    if path.suffix.lower() == ".csv":
        table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        cols = {name.lower(): name for name in table.dtype.names}
        dates = np.asarray(table[cols["date"]], dtype=str) if "date" in cols else None
        for key in ("adj_close", "adjclose", "close", "c"):
            if key in cols:
                return np.asarray(table[cols[key]], dtype=float), dates
        raise ValueError(f"no close column in {path}; found {table.dtype.names}")
    return np.loadtxt(path, comments="#", dtype=float), None


def fit_hmm(returns: np.ndarray, n_states: int, n_starts: int) -> mc.GaussianHMMParams:
    """Fit with several random starts, keep the best, order states calm -> volatile."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:
        raise SystemExit("this example needs hmmlearn: pip install hmmlearn") from exc
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)  # EM wobble near convergence

    X = returns.reshape(-1, 1)
    best, best_score = None, -np.inf
    for seed in range(n_starts):
        # A start can collapse a regime onto a few points (zero variance -> NaN); skip it.
        try:
            model = GaussianHMM(n_states, "full", n_iter=500, random_state=seed).fit(X)
            score = model.score(X)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(score) and score > best_score:
            best, best_score = model, score
    if best is None:
        raise SystemExit(f"all {n_starts} HMM fits failed; try fewer --states or more --starts")
    p = mc.GaussianHMMParams.from_hmmlearn(best)
    order = np.argsort(p.stds)
    return mc.GaussianHMMParams(
        p.transmat[np.ix_(order, order)], p.means[order], p.stds[order], p.startprob[order]
    )


def regime_positions(
    returns: np.ndarray, params: mc.GaussianHMMParams, exposure: np.ndarray
) -> np.ndarray:
    """Position for each period, set from the filtered regime at the previous period."""
    probs = mc.forward_filter(returns[None], params)[0]
    return np.r_[0.0, (probs @ exposure)[:-1]]


def walk_forward_positions(
    returns: np.ndarray,
    n_states: int,
    exposure: np.ndarray,
    train: int,
    refit_every: int,
    n_starts: int,
) -> tuple[np.ndarray, list[mc.GaussianHMMParams]]:
    """Out-of-sample positions for ``returns[train:]`` on an expanding window.

    Each block of ``refit_every`` periods (0 = one block) uses a model fitted
    only on returns before the block, and each position within it uses the
    filter up to the previous period, so nothing sees data from its own day
    or later. Returns the positions and the fitted params, one per block.
    """
    if not 0 < train < returns.size:
        raise ValueError("train must leave at least one test period")
    step = refit_every if refit_every > 0 else returns.size
    pos = np.empty(returns.size - train)
    fits = []
    for start in range(train, returns.size, step):
        end = min(start + step, returns.size)
        params = fit_hmm(returns[:start], n_states, n_starts)
        fits.append(params)
        pos[start - train:end - train] = regime_positions(returns[:end], params, exposure)[start:end]
    return pos, fits


def strategy_returns(returns: np.ndarray, pos: np.ndarray, cost: float) -> np.ndarray:
    return pos * returns - cost * np.abs(np.diff(pos, prepend=0.0))


def print_regimes(params: mc.GaussianHMMParams, exposure: np.ndarray, ppy: int, title: str) -> None:
    pi = mc.stationary_distribution(params.transmat)
    print(f"{title} (calm -> volatile):")
    for k in range(params.n_states):
        stay = 1 / (1 - params.transmat[k, k]) if params.transmat[k, k] < 1 else np.inf
        print(f"  regime {k}: mean {params.means[k] * ppy:+7.1%}/yr  "
              f"vol {params.stds[k] * np.sqrt(ppy):6.1%}/yr  time share {pi[k]:4.0%}  "
              f"avg stay {stay:6.1f} periods  exposure {exposure[k]:g}")


def print_comparison(rows: dict[str, np.ndarray], ppy: int, exposures: dict[str, float]) -> None:
    print(f"  {'':<16}{'return':>9}{'CAGR':>8}{'max DD':>8}{'Sharpe':>8}{'exposure':>10}")
    for name, r in rows.items():
        total = np.prod(1 + r) - 1
        cagr = (1 + total) ** (ppy / r.size) - 1
        sharpe = r.mean() / r.std(ddof=1) * np.sqrt(ppy) if r.std() > 0 else 0.0
        print(f"  {name:<16}{total:>+9.1%}{cagr:>+8.1%}{mc.max_drawdowns(r[None])[0]:>8.1%}"
              f"{sharpe:>8.2f}{exposures[name]:>10.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prices", nargs="?", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--states", type=int, default=2, help="number of HMM regimes")
    ap.add_argument("--exposure", help="position per regime, calm to volatile "
                    "(default: 1 in the calmest regime, 0 elsewhere)")
    ap.add_argument("--cost", type=float, default=0.0005, help="cost per unit turnover")
    ap.add_argument("--horizon", type=int, default=252, help="simulated periods")
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--ruin", type=float, default=0.2, help="drawdown counted as ruin")
    ap.add_argument("--periods-per-year", type=int, default=252)
    ap.add_argument("--starts", type=int, default=20, help="random restarts for the HMM fit")
    ap.add_argument("--seed", type=int, default=1)
    wf = ap.add_argument_group("walk-forward (out-of-sample) test")
    wf.add_argument("--walk-forward", action="store_true",
                    help="fit only on past data and trade the unseen test period")
    wf.add_argument("--train-frac", type=float, default=0.5,
                    help="share of returns used for the first fit (default 0.5)")
    wf.add_argument("--test-start", help="first test date, YYYY-MM-DD (needs a CSV date "
                    "column; overrides --train-frac)")
    wf.add_argument("--refit-every", type=int, default=63,
                    help="refit on all data so far every N periods; 0 = fit once (default 63)")
    args = ap.parse_args()

    close, dates = load_prices(args.prices)
    rets = close[1:] / close[:-1] - 1
    ret_dates = dates[1:] if dates is not None else None
    ppy = args.periods_per_year
    if args.exposure:
        exposure = np.array([float(x) for x in args.exposure.split(",")])
    else:
        exposure = np.r_[1.0, np.zeros(args.states - 1)]
    if exposure.size != args.states:
        ap.error(f"--exposure needs {args.states} values")

    print(f"{args.prices.name}: {close.size} closes, {rets.size} returns"
          + (f" ({dates[0]} .. {dates[-1]})" if dates is not None else ""))

    if args.walk_forward:
        if args.test_start:
            if ret_dates is None:
                ap.error("--test-start needs a CSV with a date column")
            train = int(np.searchsorted(ret_dates, args.test_start))
        else:
            train = int(round(rets.size * args.train_frac))
        if not 0 < train < rets.size:
            ap.error("the training period must leave at least one test period")

        pos, fits = walk_forward_positions(
            rets, args.states, exposure, train, args.refit_every, args.starts)
        params = fits[0]
        test = rets[train:]
        mkt, strat = test, strategy_returns(test, pos, args.cost)
        span = f" ({ret_dates[train]} .. {ret_dates[-1]})" if ret_dates is not None else ""
        print(f"walk-forward: train on first {train} returns, test on {test.size}{span}, "
              f"{len(fits)} fit(s), refit every {args.refit_every or 'never'}\n")
        print_regimes(params, exposure, ppy, f"Initial {args.states}-regime fit, training data only")
        print("\nOut-of-sample test period:")
        print_comparison({"buy & hold": mkt, "regime strategy": strat}, ppy,
                         {"buy & hold": 1.0, "regime strategy": pos.mean()})
        label = "out-of-sample"
    else:
        params = fit_hmm(rets, args.states, args.starts)
        pos = regime_positions(rets, params, exposure)
        mkt, strat = rets, strategy_returns(rets, pos, args.cost)
        print(f"buy & hold actual: {close[-1] / close[0] - 1:+.1%}, "
              f"max DD {mc.max_drawdowns(rets[None])[0]:.1%}\n")
        print_regimes(params, exposure, ppy, f"Fitted {args.states}-regime HMM")
        print(f"\nregime strategy actual (in-sample fit): {np.prod(1 + strat) - 1:+.1%}, "
              f"max DD {mc.max_drawdowns(strat[None])[0]:.1%}, avg exposure {pos.mean():.0%}")
        label = "actual"

    common = dict(n_sims=args.sims, horizon=args.horizon, seed=args.seed,
                  periods_per_year=ppy, ruin_threshold=args.ruin)
    print(f"\n==== A. Block bootstrap of {label} returns (buy & hold) ====")
    print(mc.run(mkt, "block_bootstrap", **common).report())
    print(f"\n==== B. Block bootstrap of the regime strategy's {label} returns ====")
    print(mc.run(strat, "block_bootstrap", **common).report())

    sim_strat, sim_mkt = mc.simulate_hmm_strategy(
        params, exposure, n_sims=args.sims, horizon=args.horizon,
        cost_per_turnover=args.cost, rng=np.random.default_rng(args.seed + 1),
    )
    print("\n==== C. Buy & hold on HMM-simulated paths ====")
    print(mc.summarize(sim_mkt, ppy, args.ruin).report())
    print("\n==== D. Regime strategy on the same simulated paths ====")
    print(mc.summarize(sim_strat, ppy, args.ruin).report())


if __name__ == "__main__":
    main()
