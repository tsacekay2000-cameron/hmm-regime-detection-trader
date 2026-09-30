import numpy as np
import pytest

pytest.importorskip("hmmlearn")

from examples import micro_futures_backtest as ex  # noqa: E402
from hmm_trader import futures as fu  # noqa: E402


def test_session_steps_costs_and_overnight():
    b = fu.FuturesBars(
        np.array(["2024-01-02", "2024-01-03", "2024-01-04"]),
        open=np.array([100.0, 101.0, 103.0]), high=np.array([100.0, 102.0, 104.0]),
        low=np.array([100.0, 100.0, 102.0]), close=np.array([100.0, 101.5, 103.5]),
        next_close=np.array([100.0, 101.5, 103.5]),
    )
    spec = fu.CONTRACTS["MES"]
    in_mkt = np.array([True, True])
    flat = ex.session_steps(b, in_mkt, spec, overnight=False, cost_per_side=1.0)
    # open->close per session, round turn every session
    np.testing.assert_allclose(flat.sum(axis=1), [0.5 * 5 - 2, 0.5 * 5 - 2])
    over = ex.session_steps(b, in_mkt, spec, overnight=True, cost_per_side=1.0)
    # close->close, one entry side only
    np.testing.assert_allclose(over.sum(axis=1), [1.5 * 5 - 1, 2.0 * 5])
    out = ex.session_steps(b, np.array([False, False]), spec, overnight=True, cost_per_side=1.0)
    assert not out.any()


def test_example_runs_end_to_end(capsys):
    ex.main(["--symbols", "MES", "--contracts", "1,2", "--refit-every", "0",
             "--starts", "2", "--sims", "200", "--test-start", "2025-01-01"])
    out = capsys.readouterr().out
    assert "MES ($5/pt" in out
    assert "regime filter x2" in out
    assert "always long x1" in out
