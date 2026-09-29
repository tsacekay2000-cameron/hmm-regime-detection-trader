import numpy as np
import pytest

pytest.importorskip("hmmlearn")

from examples import spy_monte_carlo as ex  # noqa: E402


@pytest.fixture(scope="module")
def returns():
    close, _ = ex.load_prices(ex.DEFAULT_DATA)
    return close[1:] / close[:-1] - 1


def test_walk_forward_has_no_look_ahead(returns):
    exposure = np.array([1.0, 0.0])
    train, refit = 250, 100
    pos, fits = ex.walk_forward_positions(returns, 2, exposure, train, refit, n_starts=2)
    assert pos.shape == (returns.size - train,)
    assert len(fits) == 3  # blocks start at 250, 350, 450
    assert np.all((pos >= 0) & (pos <= 1))

    # Scramble everything from day k on: positions up to and including day k
    # may only depend on returns before k, so they must not change.
    k = 380
    scrambled = returns.copy()
    scrambled[k:] = np.random.default_rng(0).permutation(scrambled[k:]) * 3
    pos2, _ = ex.walk_forward_positions(scrambled, 2, exposure, train, refit, n_starts=2)
    np.testing.assert_array_equal(pos[: k - train + 1], pos2[: k - train + 1])
    assert not np.array_equal(pos, pos2)


def test_walk_forward_single_fit_matches_train_only_model(returns):
    exposure = np.array([1.0, 0.0])
    train = 300
    pos, fits = ex.walk_forward_positions(returns, 2, exposure, train, 0, n_starts=2)
    assert len(fits) == 1
    expected = ex.regime_positions(returns, fits[0], exposure)[train:]
    np.testing.assert_allclose(pos, expected)


def test_walk_forward_validates_train(returns):
    with pytest.raises(ValueError):
        ex.walk_forward_positions(returns, 2, np.array([1.0, 0.0]), returns.size, 10, 1)


def test_strategy_returns_charges_turnover():
    r = np.array([0.01, 0.02, -0.01])
    pos = np.array([1.0, 0.0, 1.0])
    np.testing.assert_allclose(ex.strategy_returns(r, pos, 0.001),
                               [0.01 - 0.001, 0.0 - 0.001, -0.01 - 0.001])
