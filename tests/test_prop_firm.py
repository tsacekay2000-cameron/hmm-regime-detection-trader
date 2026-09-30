import numpy as np
import pytest

from hmm_trader import monte_carlo as mc
from hmm_trader import prop_firm as pf


def run(paths, **rules):
    return pf.simulate_challenge(np.asarray(paths, dtype=float), pf.ChallengeRules(**rules))


def test_passes_on_target_and_stops_trading():
    res = run([[0.05, 0.06, -0.5]], daily_loss_limit=None)
    assert res.outcomes[0] == pf.PASSED
    assert res.days[0] == 2
    # the -50% day after passing is never traded
    assert res.final_equity[0] == pytest.approx(1.05 * 1.06)


def test_static_drawdown_breach():
    res = run([[-0.05, -0.06, 0.5]], daily_loss_limit=None)
    assert res.outcomes[0] == pf.FAILED_DRAWDOWN
    assert res.days[0] == 2


def test_daily_loss_limit_on_daily_and_intraday_returns():
    assert run([[-0.06]], daily_loss_limit=0.05).outcomes[0] == pf.FAILED_DAILY_LOSS
    # intraday dip breaches even though the day closes well above the limit
    res = run([[[-0.03, -0.03, 0.05]]], daily_loss_limit=0.05)
    assert res.outcomes[0] == pf.FAILED_DAILY_LOSS
    assert run([[[-0.03, 0.05, -0.03]]], daily_loss_limit=0.05).outcomes[0] == pf.TIMED_OUT
    # measured from the start-of-day balance: still above the initial balance, but
    # down 1.08 * 6% = 6.5% of the initial balance on the day
    assert run([[0.08, -0.06]], daily_loss_limit=0.05).outcomes[0] == pf.FAILED_DAILY_LOSS
    # the limit is an amount of the initial balance: -4.9% of 1.08 is 5.3% of it
    assert run([[0.08, -0.049]], daily_loss_limit=0.05).outcomes[0] == pf.FAILED_DAILY_LOSS
    assert run([[0.08, -0.045]], daily_loss_limit=0.05).outcomes[0] == pf.TIMED_OUT


def test_trailing_drawdown_follows_peak():
    path = [[0.08, -0.10]]  # peak 1.08 -> trailing floor 0.98; equity ends at 0.972
    assert run(path, daily_loss_limit=None).outcomes[0] == pf.TIMED_OUT
    res = run(path, daily_loss_limit=None, drawdown_type="trailing")
    assert res.outcomes[0] == pf.FAILED_DRAWDOWN


def test_trailing_intraday_vs_end_of_day():
    path = [[[0.05, -0.08]]]  # intraday peak 1.05, closes at 0.966
    rules = dict(max_drawdown=0.05, daily_loss_limit=None)
    assert run(path, drawdown_type="trailing", **rules).outcomes[0] == pf.FAILED_DRAWDOWN
    assert run(path, drawdown_type="trailing_eod", **rules).outcomes[0] == pf.TIMED_OUT


def test_trailing_cap_stops_floor_at_starting_balance():
    path = [[0.20, -0.12]]  # peak 1.2 -> uncapped floor 1.1; equity ends at 1.056
    rules = dict(profit_target=0.5, daily_loss_limit=None, drawdown_type="trailing_eod")
    assert run(path, **rules).outcomes[0] == pf.FAILED_DRAWDOWN
    assert run(path, trailing_cap=0.0, **rules).outcomes[0] == pf.TIMED_OUT


def test_min_trading_days_ignores_flat_days():
    res = run([[0.11, 0.0, 0.0, 0.001, 0.001]], min_trading_days=3)
    assert res.outcomes[0] == pf.PASSED
    assert res.days[0] == 5


def test_consistency_rule_delays_pass():
    # day 1 alone hits the target but is 100% of the profit; by day 3 the best
    # day (+10%) is under half of the total profit (+21.3%)
    res = run([[0.10, 0.05, 0.05]], consistency=0.5)
    assert res.outcomes[0] == pf.PASSED
    assert res.days[0] == 3
    assert run([[0.10, 0.05, 0.05]]).days[0] == 1


def test_max_days_times_out():
    res = run([[0.01] * 20], max_days=5)
    assert res.outcomes[0] == pf.TIMED_OUT
    assert res.days[0] == 5
    assert res.final_equity[0] == pytest.approx(1.01**5)


def test_leverage_scales_returns():
    path = [[0.04, 0.04]]
    assert run(path).outcomes[0] == pf.TIMED_OUT
    res = pf.simulate_challenge(np.array(path), pf.ChallengeRules(), leverage=2.0)
    assert res.outcomes[0] == pf.PASSED


def test_driftless_pass_rate_matches_gamblers_ruin():
    # zero-drift log returns: P(hit +T before -D) = D / (D + T) in log space
    rng = np.random.default_rng(0)
    paths = np.expm1(rng.normal(0.0, 0.004, size=(2000, 4000)))
    res = run(paths, profit_target=0.10, max_drawdown=0.10, daily_loss_limit=None)
    up, down = np.log(1.10), -np.log(0.90)
    assert res.rate(pf.TIMED_OUT) < 0.01
    assert res.pass_rate == pytest.approx(down / (up + down), abs=0.04)


def test_report_and_rates_sum_to_one(returns_paths):
    res = pf.simulate_challenge(returns_paths, pf.ChallengeRules(max_days=60))
    assert sum(res.rate(o) for o in pf.OUTCOMES) == pytest.approx(1.0)
    text = res.report()
    assert "Prop challenge: 500 simulations" in text
    assert "max DD 10.0% (static)" in text
    with pytest.raises(ValueError):
        res.rate("nope")


def test_days_to_pass_nan_when_nothing_passes():
    res = run([[0.0, 0.0]])
    assert all(np.isnan(v) for v in res.days_to_pass().values())
    assert "days to pass (p25 / p50 / p75): - / - / -" in res.report()


def test_regime_strategy_paths_run():
    params = mc.GaussianHMMParams(
        transmat=[[0.98, 0.02], [0.05, 0.95]],
        means=[0.0008, -0.0015],
        stds=[0.008, 0.025],
    )
    strat, market = mc.simulate_hmm_strategy(params, [1.0, 0.0], n_sims=300, horizon=120,
                                             rng=np.random.default_rng(1))
    rules = pf.ChallengeRules(drawdown_type="trailing_eod", trailing_cap=0.0)
    for paths in (strat, market):
        res = pf.simulate_challenge(paths, rules, leverage=2.0)
        assert res.n_sims == 300
        assert np.all((res.days >= 1) & (res.days <= 120))


@pytest.mark.parametrize("kwargs", [
    dict(profit_target=0),
    dict(max_drawdown=1.0),
    dict(daily_loss_limit=0),
    dict(drawdown_type="weekly"),
    dict(min_trading_days=-1),
    dict(max_days=0),
    dict(consistency=1.5),
])
def test_rules_validation(kwargs):
    with pytest.raises(ValueError):
        pf.ChallengeRules(**kwargs)


def test_simulate_validation():
    with pytest.raises(ValueError):
        pf.simulate_challenge(np.zeros(5), pf.ChallengeRules())
    with pytest.raises(ValueError):
        pf.simulate_challenge(np.array([[np.nan]]), pf.ChallengeRules())
    with pytest.raises(ValueError):
        pf.simulate_challenge(np.zeros((2, 2)), pf.ChallengeRules(), leverage=0)


@pytest.fixture
def returns_paths():
    return np.random.default_rng(3).normal(0.001, 0.01, size=(500, 100))


def test_non_compounding_adds_pnl_fractions():
    # +5% then +5% of the initial balance = +10%: passes without compounding
    res = pf.simulate_challenge(np.array([[0.05, 0.05]]), pf.ChallengeRules(),
                                compounding=False)
    assert res.outcomes[0] == pf.PASSED
    assert res.final_equity[0] == pytest.approx(1.10)
    # losses add up too: -6% then -4.5% breaches the 10% static floor exactly past it
    res = pf.simulate_challenge(np.array([[-0.06, -0.045]]),
                                pf.ChallengeRules(daily_loss_limit=None), compounding=False)
    assert res.outcomes[0] == pf.FAILED_DRAWDOWN
    # leverage scales the P&L linearly
    res = pf.simulate_challenge(np.array([[0.03, 0.03]]), pf.ChallengeRules(),
                                leverage=2.0, compounding=False)
    assert res.final_equity[0] == pytest.approx(1.12)  # +6% a day, passes on day 2
    assert res.days[0] == 2
