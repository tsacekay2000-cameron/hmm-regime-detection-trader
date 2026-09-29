"""Monte Carlo demo: bootstrap a backtest, then stress a regime strategy on HMM paths.

Run from the repo root:  python -m examples.monte_carlo_demo
"""

import numpy as np

from hmm_trader import monte_carlo as mc

# Illustrative 2-regime model of daily returns (bull: calm, bear: volatile).
# Replace with GaussianHMMParams.from_hmmlearn(fitted_model) for a real fit.
PARAMS = mc.GaussianHMMParams(
    transmat=[[0.98, 0.02], [0.05, 0.95]],
    means=[0.0008, -0.0015],
    stds=[0.008, 0.025],
)


def main() -> None:
    rng = np.random.default_rng(42)

    # 1) Resample an existing backtest's daily strategy returns.
    #    Here a synthetic stand-in; pass your real backtest returns instead.
    backtest, _ = mc.simulate_hmm(PARAMS, n_sims=1, horizon=756, rng=rng)
    print("== Block bootstrap of a 3-year backtest ==")
    print(mc.run(backtest[0], method="block_bootstrap", n_sims=5000, seed=1).report())

    # 2) Run a long-in-bull / flat-in-bear strategy on fresh HMM market paths,
    #    detecting the regime causally with the forward filter.
    strat, market = mc.simulate_hmm_strategy(
        PARAMS, regime_exposure=[1.0, 0.0], n_sims=5000, horizon=252,
        cost_per_turnover=0.0005, rng=rng,
    )
    print("\n== Buy & hold on HMM paths (1 year) ==")
    print(mc.summarize(market, ruin_threshold=0.3).report())
    print("\n== Regime strategy on the same paths (filtered detection, 5 bps costs) ==")
    print(mc.summarize(strat, ruin_threshold=0.3).report())


if __name__ == "__main__":
    main()
