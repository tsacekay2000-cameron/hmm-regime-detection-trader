import numpy as np

from examples import orb_narrow_range as ex


def test_narrow_mask_uses_prior_days_only():
    width = np.array([2.0, 2.0, 4.0, 1.0, 3.0])
    eligible, narrow = ex.narrow_mask(width, 1.0, 2)
    assert list(eligible) == [False, False, True, True, True]
    # day 3: 4 > median(2, 2); day 4: 1 <= median(2, 4); day 5: 3 > median(4, 1) = 2.5
    assert list(narrow) == [False, False, False, True, False]


def test_perm_p_extremes():
    rng = np.random.default_rng(0)
    r = np.array([1.0, 1.0, -1.0, -1.0, np.nan, -1.0])
    best = np.array([True, True, False, False, False, False])
    worst = np.array([False, False, True, True, False, False])
    assert ex.perm_p(r, best, rng, 2000) < 0.2
    assert ex.perm_p(r, worst, rng, 2000) == 1.0


def test_example_runs(capsys):
    ex.main(["--perms", "50", "--sims", "100", "--horizon", "60"])
    out = capsys.readouterr().out
    assert "<- primary" in out and "Robustness" in out
