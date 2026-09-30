import numpy as np

from examples import orb_backtest as ex


def test_zero_edge_removes_average_profit_only():
    steps = np.array([[10.0, -5.0, 3.0], [0.0, 0.0, 0.0], [2.0, -8.0, 0.0]])
    traded = np.array([True, False, True])
    null = ex.zero_edge(steps, traded)
    assert abs(null[traded].sum(axis=1).mean()) < 1e-12
    np.testing.assert_array_equal(null[:, :2], steps[:, :2])
    np.testing.assert_array_equal(null[~traded], steps[~traded])


def test_example_runs_end_to_end(capsys):
    ex.main(["--symbols", "MES", "--risk", "200", "--sims", "200", "--horizon", "60"])
    out = capsys.readouterr().out
    assert "ORB 15m / 2R" in out
    assert "Chosen on data before 2025-10-01" in out
    assert "$200 risk, zero-edge bootstrap, all" in out
