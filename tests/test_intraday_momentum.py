from pathlib import Path

import numpy as np
import pytest

from hmm_trader import intraday_momentum as im

N = im.BARS_PER_DAY


def days(closes, prev_close=None, volume=1.0):
    """DayBars from per-day close paths (open = previous bar's close, flat highs/lows)."""
    c = np.asarray(closes, dtype=float)
    o = np.concatenate([c[:, :1], c[:, :-1]], axis=1)
    hi, lo = np.maximum(o, c), np.minimum(o, c)
    prev = np.r_[np.nan, c[:-1, -1]] if prev_close is None else np.asarray(prev_close, float)
    return im.DayBars(np.array([f"d{i}" for i in range(len(c))]), o, hi, lo, c,
                      np.full(c.shape, volume), prev)


def test_bar_index():
    assert im.bar_index("09:30") == 0 and im.bar_index("10:00") == 6
    assert im.bar_index("15:30") == 72 and im.bar_index("15:55") == N - 1


def test_last_half_hour_side_and_window():
    up = np.linspace(100, 110, N)
    b = days([up, up], prev_close=[np.nan, 99.0])
    pos, sig = im.last_half_hour_positions(b)
    assert np.isnan(sig[0]) and (pos[0] == 0).all()             # no previous close
    assert sig[1] == pytest.approx(up[5] / 99 - 1)               # close of the bar ending 10:00
    assert (pos[1, :72] == 0).all() and (pos[1, 72:] == 1).all()
    down = days([up[::-1], up[::-1]], prev_close=[np.nan, 120.0])
    assert (im.last_half_hour_positions(down)[0][1, 72:] == -1).all()


def test_simulate_positions_costs_and_steps():
    path = np.r_[np.full(72, 100.0), np.linspace(101, 106, 6)]
    b = days([path])
    pos = np.zeros((1, N))
    pos[0, 72:] = 1
    t = im.simulate_positions(b, pos, multiplier=5, tick=0.25, commission=0.62)
    # bought at the 15:30 open (100) + 1 tick, sold at the close (106) - 1 tick
    assert t.pnl[0] == pytest.approx((105.75 - 100.25) * 5 - 2 * 0.62)
    assert t.trades[0] == 1
    assert sum(t.steps[0]) == pytest.approx(t.pnl[0])


def test_noise_boundary_long_on_trend_and_no_lookahead():
    rng = np.random.default_rng(0)
    flat = [100 + np.cumsum(rng.normal(0, 0.05, N)) for _ in range(14)]
    trend = np.linspace(100, 104, N)
    b = days(flat + [trend])
    pos = im.noise_boundary_positions(b)
    assert (pos[:14] == 0).all()                                # warm-up
    assert pos[14, im.bar_index("10:00")] == 1 and pos[14, -1] == 1
    later = days(flat + [np.r_[trend[:40], np.full(N - 40, 90.0)]])
    np.testing.assert_array_equal(im.noise_boundary_positions(later)[14, :41], pos[14, :41])


def test_flip_test_and_vwap():
    rng = np.random.default_rng(1)
    assert im.flip_test(np.abs(rng.normal(0, 1, 200)), rng, 2000) < 0.01
    b = days([np.linspace(100, 101, N)])
    vw = im.vwap(b)
    assert vw[0, 0] == pytest.approx((b.high[0, 0] + b.low[0, 0] + b.close[0, 0]) / 3)
    assert b.low[0].min() <= vw[0, -1] <= b.high[0].max()


def test_bundled_data_and_example(capsys):
    root = Path(__file__).parents[1] / "examples" / "data"
    b = im.load_day_bars(root / "mes_5min_rth.csv.gz")
    assert b.date.size == 489 and (b.volume > 0).mean() > 0.99
    assert np.isfinite(b.prev_close[1:]).all()
    from examples import intraday_momentum as ex
    ex.main(["--symbols", "MES", "--sims", "100", "--horizon", "60", "--flips", "200",
             "--contracts", "1"])
    out = capsys.readouterr().out
    assert "noise boundary" in out and "Sensitivity" in out and "Prop evaluation" in out


def test_trade_list_matches_simulator_and_reference_csv():
    import csv
    from hmm_trader import futures as fu
    root = Path(__file__).parents[1]
    b = im.load_day_bars(root / "examples" / "data" / "mnq_5min_rth.csv.gz")
    spec = fu.CONTRACTS["MNQ"]
    pos = im.noise_boundary_positions(b)
    rows = im.trade_list(b, pos, spec.multiplier, spec.tick)
    net = im.simulate_positions(b, pos, spec.multiplier, spec.tick)
    assert len(rows) == net.trades.sum() == 380
    assert sum(r[6] for r in rows) == pytest.approx(net.pnl.sum(), abs=0.01)
    with open(root / "pine" / "noise_boundary_mnq_reference_trades.csv") as fh:
        saved = list(csv.DictReader(fh))
    assert [(r["date"], r["side"], r["entry_time_et"]) for r in saved] == [r[:3] for r in rows]
    src = (root / "pine" / "noise_boundary.pine").read_text()
    for snippet in ("//@version=6", 'input.int(14, "Noise lookback (sessions)"',
                    'input.int(30, "Check every (minutes)"', "commission_value = 0.62",
                    "slippage = 1", "margin_long = 10", "math.max(upper, vw)", "math.min(lower, vw)"):
        assert snippet in src
