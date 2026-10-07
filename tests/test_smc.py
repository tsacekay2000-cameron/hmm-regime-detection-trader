from dataclasses import replace

import numpy as np
import pytest

from hmm_trader import intraday_momentum as im
from hmm_trader import smc

N = im.BARS_PER_DAY


def flat_days(n=2, px=100.0):
    a = np.full((n, N), px)
    return {k: a.copy() for k in ("open", "high", "low", "close")}


def make(cols, volume=1.0):
    n = cols["open"].shape[0]
    prev = np.r_[np.nan, cols["close"][:-1, -1]]
    return im.DayBars(np.array([f"2025-01-0{i + 1}" for i in range(n)]), cols["open"], cols["high"],
                      cols["low"], cols["close"], np.full(cols["open"].shape, volume), prev)


def bar(cols, d, i, o, h, l, c):
    for k, v in zip(("open", "high", "low", "close"), (o, h, l, c)):
        cols[k][d, i] = v


def long_setup():
    """Day 0 low 99; day 1 sweeps it at 09:55, gaps up, retraces into the gap, then rallies."""
    c = flat_days()
    c["low"][0] -= 1.0                                      # previous session low = 99
    bar(c, 1, 5, 100.0, 100.0, 98.5, 99.5)                  # sweep: below 99, close back above
    bar(c, 1, 6, 99.5, 99.75, 99.25, 99.5)                  # bar k-2, high 99.75
    bar(c, 1, 7, 99.5, 101.5, 99.5, 101.25)                 # displacement
    bar(c, 1, 8, 101.25, 101.5, 100.5, 101.0)               # bar k, low 100.5 > 99.75: gap
    for i in range(9, N):
        bar(c, 1, i, 101.0, 101.25, 100.75, 101.0)
    bar(c, 1, 10, 101.0, 101.0, 100.0, 100.25)              # trades through 100.5 -> fill
    for i in range(11, N):
        bar(c, 1, i, 101.0, 106.0, 100.75, 105.0)           # rally through the 2R target
    return c


def test_sweep_fvg_fill_and_target():
    b = make(long_setup())
    p = smc.SMCParams(min_rvol=None, first_entry="09:30")
    tr = smc.backtest(b, multiplier=5.0, tick=0.25, p=p)
    assert len(tr) == 1
    t = tr[0]
    assert (t.day, t.side, t.fill_bar, t.entry) == (1, 1, 10, 100.5)
    assert t.stop == 98.5                                    # lowest low since the sweep
    assert t.target == pytest.approx(100.5 + 2 * (100.5 - 98.5))
    assert t.reason == "target" and t.exit_price == pytest.approx(t.target)
    risk = (100.5 - 98.5 + 0.25) * 5 * t.contracts
    assert t.contracts == int(200 // ((100.5 - 98.5 + 0.25) * 5))
    assert t.r == pytest.approx((4.0 * 5 * t.contracts - 2 * t.contracts * 0.62) / risk)


def test_touch_is_not_a_fill_and_volume_filter():
    c = long_setup()
    bar(c, 1, 10, 101.0, 101.0, 100.5, 100.75)               # touches 100.5 but does not trade through
    for i in range(11, N):
        bar(c, 1, i, 101.0, 106.0, 100.75, 105.0)
    b = make(c)
    assert smc.backtest(b, 5.0, 0.25, smc.SMCParams(min_rvol=None, first_entry="09:30")) == []
    # with the volume filter and no volume history, nothing trades
    assert smc.backtest(make(long_setup()), 5.0, 0.25, smc.SMCParams(first_entry="09:30")) == []


def test_stop_first_on_ambiguous_bar_and_mirror_symmetry():
    c = long_setup()
    for i in range(11, N):
        bar(c, 1, i, 101.0, 106.0, 98.0, 100.0)              # every bar touches both stop and target
    tr = smc.backtest(make(c), 5.0, 0.25, smc.SMCParams(min_rvol=None, first_entry="09:30"))
    assert tr[0].reason == "stop" and tr[0].exit_price == pytest.approx(98.5 - 0.25)
    # the mirrored price path produces the mirrored short trade
    m = {k: -v for k, v in long_setup().items()}
    m["high"], m["low"] = -long_setup()["low"], -long_setup()["high"]
    for k in m:
        m[k] = m[k] + 300.0
    tr_s = smc.backtest(make(m), 5.0, 0.25, smc.SMCParams(min_rvol=None, first_entry="09:30"))
    assert len(tr_s) == 1 and tr_s[0].side == -1 and tr_s[0].reason == "target"
    assert tr_s[0].r == pytest.approx(smc.backtest(make(long_setup()), 5.0, 0.25,
                                      smc.SMCParams(min_rvol=None, first_entry="09:30"))[0].r)


def test_relative_volume_and_steps_and_pvalue():
    c = flat_days(n=22)
    b = make(c)
    b.volume[21, 7] = 3.0
    rv = smc.relative_volume(b, days=20)
    assert np.isnan(rv[5, 0]) and rv[21, 7] == pytest.approx(3.0) and rv[21, 6] == pytest.approx(1.0)
    b2 = make(long_setup())
    tr = smc.backtest(b2, 5.0, 0.25, smc.SMCParams(min_rvol=None, first_entry="09:30"))
    steps = smc.daily_steps(b2, tr, 5.0)
    assert steps[1].sum() == pytest.approx(tr[0].pnl) and steps[0].sum() == 0
    assert set(smc.variants()) >= {"full", "sweep only", "order block entry", "FVG only (no sweep)"}
    rng = np.random.default_rng(0)
    assert np.isnan(smc.mirror_pvalue([], rng))


def test_example_runs(capsys):
    from examples import smc_backtest as ex
    ex.main(["--symbols", "MES", "--sims", "100", "--horizon", "60", "--risk", "200"])
    out = capsys.readouterr().out
    assert "sweep only" in out and "Prop evaluation" in out


def test_reference_trades_and_pine_defaults():
    import csv
    from pathlib import Path
    from hmm_trader import futures as fu
    root = Path(__file__).parents[1]
    b = im.load_day_bars(root / "examples" / "data" / "mes_5min_rth.csv.gz")
    spec = fu.CONTRACTS["MES"]
    tr = smc.backtest(b, spec.multiplier, spec.tick, smc.variants()["full, 1R target"])
    with open(root / "pine" / "smc_1r_mes_reference_trades.csv") as fh:
        saved = list(csv.DictReader(fh))
    assert len(saved) == len(tr) == 120
    assert sum(float(r["pnl"]) for r in saved) == pytest.approx(sum(t.pnl for t in tr), abs=0.05)
    b_nq = im.load_day_bars(root / "examples" / "data" / "mnq_5min_rth.csv.gz")
    nq = smc.backtest(b_nq, 2.0, 0.25, smc.variants()["full, 1R target"])
    with open(root / "pine" / "smc_1r_mnq_reference_trades.csv") as fh:
        saved_nq = list(csv.DictReader(fh))
    assert len(saved_nq) == len(nq) == 84
    assert sum(float(r["pnl"]) for r in saved_nq) == pytest.approx(sum(t.pnl for t in nq), abs=0.05)
    b_ym = im.load_day_bars(root / "examples" / "data" / "mym_5min_rth.csv.gz")
    ym = smc.backtest(b_ym, 0.5, 1.0, smc.variants()["full, 1R target"])
    with open(root / "pine" / "smc_1r_mym_reference_trades.csv") as fh:
        saved_ym = list(csv.DictReader(fh))
    assert len(saved_ym) == len(ym) == 130
    assert sum(float(r["pnl"]) for r in saved_ym) == pytest.approx(sum(t.pnl for t in ym), abs=0.05)
    src = (root / "pine" / "smc_sweep_fvg.pine").read_text()
    for snippet in ("//@version=6", 'input.float(200, "Risk per trade ($)"', 'input.float(1.0, "Target (R multiple)"',
                    'input.float(1.5, "Min. relative volume', 'input.int(6, "Gap must form within',
                    'input.int(12, "Limit order valid for', "RVOL_DAYS = 20", "commission_value = 0.62",
                    "slippage = 1", "margin_long = 10"):
        assert snippet in src
