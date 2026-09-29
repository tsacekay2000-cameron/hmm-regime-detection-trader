import numpy as np
import pytest

from hmm_trader import monte_carlo as mc


@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    return rng.normal(0.0005, 0.01, size=500)


@pytest.fixture
def params():
    return mc.GaussianHMMParams(
        transmat=[[0.98, 0.02], [0.05, 0.95]],
        means=[0.0008, -0.0015],
        stds=[0.008, 0.025],
    )


def test_shuffle_preserves_total_return(returns):
    paths = mc.shuffle_paths(returns, n_sims=50, rng=np.random.default_rng(1))
    assert paths.shape == (50, returns.size)
    expected = np.prod(1 + returns)
    np.testing.assert_allclose(np.prod(1 + paths, axis=1), expected)
    # ordering actually changes, so drawdowns differ
    assert np.unique(np.round(mc.max_drawdowns(paths), 10)).size > 1


def test_bootstrap_draws_from_input(returns):
    paths = mc.bootstrap_paths(returns, n_sims=20, horizon=100, rng=np.random.default_rng(2))
    assert paths.shape == (20, 100)
    assert np.isin(paths, returns).all()


def test_block_bootstrap_keeps_contiguous_blocks():
    data = np.arange(1, 101) / 1000.0
    paths = mc.block_bootstrap_paths(data, n_sims=10, horizon=55, block_size=5,
                                     rng=np.random.default_rng(3))
    assert paths.shape == (10, 55)
    idx = np.rint(paths * 1000).astype(int) - 1
    for row in idx:
        for start in range(0, 55, 5):
            block = row[start:start + 5]
            np.testing.assert_array_equal(np.diff(block) % 100, 1)


def test_max_drawdown_known_path():
    # up 10%, down 50%, up 20% -> peak 1.1, trough 0.55 -> 50% drawdown
    dd = mc.max_drawdowns(np.array([[0.1, -0.5, 0.2]]))
    np.testing.assert_allclose(dd, [0.5])
    assert mc.max_drawdowns(np.array([[0.01, 0.02]]))[0] == 0.0
    # loss on the very first period counts against the starting equity
    np.testing.assert_allclose(mc.max_drawdowns(np.array([[-0.2, 0.1]])), [0.2])


def test_summarize_metrics(returns):
    res = mc.run(returns, method="block_bootstrap", n_sims=300, seed=4)
    assert res.n_sims == 300
    table = res.table()
    assert set(table) == {"final_return", "cagr", "max_drawdown", "sharpe"}
    assert table["max_drawdown"][5] <= table["max_drawdown"][95]
    assert 0.0 <= res.prob_ruin <= 1.0
    assert 0.0 <= res.prob_loss <= 1.0
    assert "Monte Carlo: 300 simulations" in res.report()


def test_run_is_reproducible_and_validates(returns):
    a = mc.run(returns, method="bootstrap", n_sims=50, seed=7)
    b = mc.run(returns, method="bootstrap", n_sims=50, seed=7)
    np.testing.assert_array_equal(a.final_returns, b.final_returns)
    with pytest.raises(ValueError):
        mc.run(returns, method="nope")
    with pytest.raises(ValueError):
        mc.run([])
    with pytest.raises(ValueError):
        mc.run([0.1, -1.5])


def test_prob_ruin_threshold():
    paths = np.array([[-0.6], [-0.1], [0.2]])
    res = mc.summarize(paths, periods_per_year=1, ruin_threshold=0.5)
    assert res.prob_ruin == pytest.approx(1 / 3)
    assert res.prob_loss == pytest.approx(2 / 3)


def test_hmm_params_validation():
    with pytest.raises(ValueError):
        mc.GaussianHMMParams(transmat=[[0.5, 0.4], [0.1, 0.9]], means=[0, 0], stds=[1, 1])
    with pytest.raises(ValueError):
        mc.GaussianHMMParams(transmat=[[1.0]], means=[0, 0], stds=[1, 1])
    with pytest.raises(ValueError):
        mc.GaussianHMMParams(transmat=[[0.9, 0.1], [0.1, 0.9]], means=[0, 0], stds=[1, 0])


def test_stationary_distribution(params):
    pi = mc.stationary_distribution(params.transmat)
    np.testing.assert_allclose(pi @ params.transmat, pi)
    np.testing.assert_allclose(pi, [5 / 7, 2 / 7])


def test_simulate_hmm_matches_regime_statistics(params):
    rets, states = mc.simulate_hmm(params, n_sims=200, horizon=500, rng=np.random.default_rng(5))
    assert rets.shape == states.shape == (200, 500)
    # time spent in each regime approaches the stationary distribution
    np.testing.assert_allclose(np.mean(states == 0), 5 / 7, atol=0.03)
    # conditional return moments match each regime
    for k in range(params.n_states):
        np.testing.assert_allclose(rets[states == k].std(), params.stds[k], rtol=0.05)
    # empirical transition probabilities
    stay = np.mean(states[:, 1:][states[:, :-1] == 0] == 0)
    assert stay == pytest.approx(0.98, abs=0.005)


def test_forward_filter_detects_regimes(params):
    rets, states = mc.simulate_hmm(params, n_sims=50, horizon=400, rng=np.random.default_rng(6))
    probs = mc.forward_filter(rets, params)
    assert probs.shape == (50, 400, 2)
    np.testing.assert_allclose(probs.sum(axis=2), 1.0)
    accuracy = np.mean(probs.argmax(axis=2) == states)
    assert accuracy > 0.8


def test_hmm_strategy_lags_and_costs(params):
    rng = np.random.default_rng(8)
    strat, market = mc.simulate_hmm_strategy(
        params, regime_exposure=[1.0, 0.0], n_sims=100, horizon=300,
        use_true_states=True, rng=rng,
    )
    assert strat.shape == market.shape == (100, 300)
    # flat on the first period: no information yet
    np.testing.assert_array_equal(strat[:, 0], 0.0)

    # always-long with no costs reproduces the market after the first period
    s2, m2 = mc.simulate_hmm_strategy(params, [1.0, 1.0], n_sims=20, horizon=50,
                                      rng=np.random.default_rng(9))
    np.testing.assert_allclose(s2[:, 1:], m2[:, 1:])

    # costs only ever reduce returns
    s3, _ = mc.simulate_hmm_strategy(params, [1.0, 0.0], n_sims=20, horizon=50,
                                     cost_per_turnover=0.001, rng=np.random.default_rng(10))
    s4, _ = mc.simulate_hmm_strategy(params, [1.0, 0.0], n_sims=20, horizon=50,
                                     rng=np.random.default_rng(10))
    assert np.all(s3 <= s4 + 1e-15)
    assert np.any(s3 < s4)

    with pytest.raises(ValueError):
        mc.simulate_hmm_strategy(params, [1.0], n_sims=2, horizon=5)


def test_regime_strategy_reduces_drawdown(params):
    strat, market = mc.simulate_hmm_strategy(params, [1.0, 0.0], n_sims=500, horizon=252,
                                             rng=np.random.default_rng(11))
    assert np.median(mc.max_drawdowns(strat)) < np.median(mc.max_drawdowns(market))
