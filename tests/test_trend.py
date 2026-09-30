import numpy as np
import pytest

from hmm_trader import futures as fu
from hmm_trader import trend as tr


def test_sma_cross_long_flat_and_short():
    close = np.r_[np.arange(10.0, 0, -1), np.arange(1.0, 12)]  # down then up
    lf = tr.sma_cross_positions(close, 2, 4)
    ls = tr.sma_cross_positions(close, 2, 4, allow_short=True)
    assert lf.size == close.size - 1
    assert set(np.unique(lf)) <= {0, 1} and (lf[-3:] == 1).all()
    assert (ls[3:8] == -1).all() and (ls[-3:] == 1).all()
    assert (lf[:3] == 0).all()  # no slow average yet


def test_donchian_entry_exit_and_causality():
    close = np.r_[np.full(5, 10.0), 11, 12, 13, 12.5, 12, 11, 10, 9, 8, 7]
    pos = tr.donchian_positions(close, entry=5, exit=2)
    # close 5 (11) breaks the prior 5-day high -> long into session 6
    assert pos[5] == 1
    # close 9 (12) is below the prior 2 closes (13, 12.5) -> flat
    assert pos[8] == 1 and pos[9] == 0
    # close 12 (9) is below the prior 5 closes -> short
    assert pos[12] == -1
    later = close.copy()
    later[-1] = 100.0
    np.testing.assert_array_equal(tr.donchian_positions(later, 5, 2)[:-1], pos[:-1])


def test_tsmom_sign():
    close = np.r_[np.linspace(100, 120, 10), np.linspace(119, 90, 10)]
    pos = tr.tsmom_positions(close, lookback=5)
    assert (pos[:5] == 0).all()
    assert pos[8] == 1 and pos[-1] == -1
    assert set(np.unique(tr.tsmom_positions(close, 5, allow_short=False))) <= {0, 1}


def daily(rows, next_close=None):
    a = np.array(rows, dtype=float)
    nc = a[:, 3] if next_close is None else np.asarray(next_close, dtype=float)
    return fu.FuturesBars(np.array([f"d{i}" for i in range(len(a))]),
                          a[:, 0], a[:, 1], a[:, 2], a[:, 3], nc)


def test_signed_session_steps_short_order_and_costs():
    b = daily([(100, 100, 100, 100), (101, 104, 99, 102), (102, 103, 100, 101)])
    roll = np.zeros(3, dtype=bool)
    pos = np.array([-1, -1])
    over = tr.signed_session_steps(b, roll, pos, multiplier=5, cost_per_side=1)
    # short from close 100: gap -1, then open->low (+2), low->high (-5), high->close (+2)
    np.testing.assert_allclose(over[0], [-1 * 5 - 1, 2 * 5, -5 * 5, 2 * 5])
    assert over.sum() == pytest.approx((100 - 101) * 5 - 1)  # one entry side, no exit yet
    flat = tr.signed_session_steps(b, roll, pos, 5, 1, overnight=False)
    np.testing.assert_allclose(flat.sum(axis=1), [(101 - 102) * 5 - 2, (102 - 101) * 5 - 2])


def test_signed_steps_roll_cost_only_when_carried():
    b = daily([(100, 100, 100, 100), (101, 101, 101, 101), (102, 102, 102, 102)],
              next_close=[101, 102, 103])
    roll = np.array([False, False, True])
    carried = tr.signed_session_steps(b, roll, np.array([1, 1]), 1, 1)
    assert carried[1, 0] == pytest.approx((102 - 102) - 2)  # from the 2nd-month close, 2 sides
    fresh = tr.signed_session_steps(b, roll, np.array([0, 1]), 1, 1)
    assert fresh[1, 0] == pytest.approx(0 - 1)  # entry side only


def test_position_spans_and_shift_test():
    pos = np.array([0, 1, 1, -1, 0, 1])
    assert tr.position_spans(pos) == [(1, 3, 1), (3, 4, -1), (5, 6, 1)]
    rng = np.random.default_rng(0)
    g = rng.normal(0, 1, 400)
    perfect = np.sign(g).astype(int)
    _, p = tr.circular_shift_pvalue(g, perfect, 0, float(np.dot(perfect, g)), min_shift=10)
    assert p < 0.01


def test_example_runs(capsys):
    from examples import mes_trend as ex
    ex.main(["--sims", "100", "--horizon", "60", "--contracts", "1"])
    out = capsys.readouterr().out
    assert "TSMOM 12m L/S" in out and "Sensitivity" in out and "Net $ by year" in out
