"""Prop firm demo: pass rate vs. position size, buy & hold vs. a regime filter.

Run from the repo root:  python -m examples.prop_firm_demo
"""

import numpy as np

from hmm_trader import monte_carlo as mc
from hmm_trader import prop_firm as pf

# Same illustrative 2-regime model as monte_carlo_demo (bull: calm, bear: volatile).
PARAMS = mc.GaussianHMMParams(
    transmat=[[0.98, 0.02], [0.05, 0.95]],
    means=[0.0008, -0.0015],
    stds=[0.008, 0.025],
)

# Two common rule styles; check your firm's exact terms.
RULE_SETS = {
    "static DD (target 10%, max DD 10%, daily 5%)": pf.ChallengeRules(
        profit_target=0.10, max_drawdown=0.10, daily_loss_limit=0.05, min_trading_days=4,
    ),
    "trailing EOD DD (target 6%, trailing 4% capped at start, no daily)": pf.ChallengeRules(
        profit_target=0.06, max_drawdown=0.04, daily_loss_limit=None,
        drawdown_type="trailing_eod", trailing_cap=0.0,
    ),
}
LEVERAGES = (0.5, 1.0, 2.0, 3.0, 5.0)
HORIZON = 252  # trading days available to pass (no firm time limit)


def main() -> None:
    rng = np.random.default_rng(7)
    strat, market = mc.simulate_hmm_strategy(
        PARAMS, regime_exposure=[1.0, 0.0], n_sims=5000, horizon=HORIZON,
        cost_per_turnover=0.0005, rng=rng,
    )
    header = f"{'strategy':<14}{'leverage':>9}{'pass':>8}{'max DD':>8}{'daily':>8}"
    header += f"{'timeout':>9}{'median days':>13}"
    for name, rules in RULE_SETS.items():
        print(f"\n== {name} ==")
        print(header)
        print("-" * len(header))
        for label, paths in (("buy & hold", market), ("regime filter", strat)):
            for lev in LEVERAGES:
                res = pf.simulate_challenge(paths, rules, leverage=lev)
                median = res.days_to_pass()[50]
                print(
                    f"{label:<14}{lev:>8.1f}x"
                    f"{res.pass_rate:>8.1%}{res.rate(pf.FAILED_DRAWDOWN):>8.1%}"
                    f"{res.rate(pf.FAILED_DAILY_LOSS):>8.1%}{res.rate(pf.TIMED_OUT):>9.1%}"
                    f"{'-' if np.isnan(median) else f'{median:.0f}':>13}"
                )
    print("\nDaily returns only: intraday dips below the limits are not seen, so real")
    print("failure rates are higher. Pass intraday paths (n_sims, days, steps) for that.")


if __name__ == "__main__":
    main()
