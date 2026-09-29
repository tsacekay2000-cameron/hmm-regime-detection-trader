"""Fit a Gaussian HMM to real daily prices and run Monte Carlo on a regime strategy.

Run from the repo root (defaults to the bundled SPY sample):

    python -m examples.spy_monte_carlo
    python -m examples.spy_monte_carlo my_prices.csv --states 3 --exposure 1,0.5,0

Input: a CSV with a ``close`` (or ``adj_close``) column, or a plain text file
with one close per line (``#`` comments allowed), oldest first.

Sections A-B resample the actual buy & hold and strategy returns; C-D trade
fresh market paths sampled from the fitted HMM. The HMM is fitted on the same
data the historical strategy trades, so A-B and the "actual" line are in-sample.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from hmm_trader import monte_carlo as mc

DEFAULT_DATA = Path(__file__).parent / "data" / "spy_daily_close.txt"


def load_closes(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".csv":
        table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        cols = {name.lower(): name for name in table.dtype.names}
        for key in ("adj_close", "adjclose", "close", "c"):
            if key in cols:
                return np.asarray(table[cols[key]], dtype=float)
        raise ValueError(f"no close column in {path}; found {table.dtype.names}")
    return np.loadtxt(path, comments="#", dtype=float)


def fit_hmm(returns: np.ndarray, n_states: int, n_starts: int) -> mc.GaussianHMMParams:
    """Fit with several random starts, keep the best, order states calm -> volatile."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:
        raise SystemExit("this example needs hmmlearn: pip install hmmlearn") from exc
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)  # EM wobble near convergence

    X = returns.reshape(-1, 1)
    best = max(
        (GaussianHMM(n_states, "full", n_iter=500, random_state=s).fit(X) for s in range(n_starts)),
        key=lambda m: m.score(X),
    )
    p = mc.GaussianHMMParams.from_hmmlearn(best)
    order = np.argsort(p.stds)
    return mc.GaussianHMMParams(
        p.transmat[np.ix_(order, order)], p.means[order], p.stds[order], p.startprob[order]
    )


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
    args = ap.parse_args()

    close = load_closes(args.prices)
    rets = close[1:] / close[:-1] - 1
    ppy = args.periods_per_year
    if args.exposure:
        exposure = np.array([float(x) for x in args.exposure.split(",")])
    else:
        exposure = np.r_[1.0, np.zeros(args.states - 1)]
    if exposure.size != args.states:
        ap.error(f"--exposure needs {args.states} values")

    params = fit_hmm(rets, args.states, args.starts)
    pi = mc.stationary_distribution(params.transmat)

    print(f"{args.prices.name}: {close.size} closes, {rets.size} returns")
    print(f"buy & hold actual: {close[-1] / close[0] - 1:+.1%}, "
          f"max DD {mc.max_drawdowns(rets[None])[0]:.1%}\n")
    print(f"Fitted {args.states}-regime HMM (calm -> volatile):")
    for k in range(args.states):
        stay = 1 / (1 - params.transmat[k, k]) if params.transmat[k, k] < 1 else np.inf
        print(f"  regime {k}: mean {params.means[k] * ppy:+7.1%}/yr  "
              f"vol {params.stds[k] * np.sqrt(ppy):6.1%}/yr  time share {pi[k]:4.0%}  "
              f"avg stay {stay:6.1f} periods  exposure {exposure[k]:g}")

    # Historical strategy: position for day t uses the filtered regime at t-1.
    probs = mc.forward_filter(rets[None], params)[0]
    pos = np.r_[0.0, (probs @ exposure)[:-1]]
    strat = pos * rets - args.cost * np.abs(np.diff(pos, prepend=0.0))
    print(f"\nregime strategy actual (in-sample fit): {np.prod(1 + strat) - 1:+.1%}, "
          f"max DD {mc.max_drawdowns(strat[None])[0]:.1%}, avg exposure {pos.mean():.0%}")

    common = dict(n_sims=args.sims, horizon=args.horizon, seed=args.seed,
                  periods_per_year=ppy, ruin_threshold=args.ruin)
    print("\n==== A. Block bootstrap of actual returns (buy & hold) ====")
    print(mc.run(rets, "block_bootstrap", **common).report())
    print("\n==== B. Block bootstrap of the regime strategy's actual returns ====")
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
