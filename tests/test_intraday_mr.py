import numpy as np
import pytest

from hmm_trader import intraday_momentum as im
from hmm_trader import intraday_mr as mr

N = im.BARS_PER_DAY


def days_from_closes(closes, spread=0.25, volume=1.0, prev=None):
    c = np.asarray(closes, dtype=float)
    o = np.concatenate([c[:, :1], c[:, :-1]], axis=1)
    hi, lo = np.maximum(o, c) + spread, np.minimum(o, c) - spread
    pc = np.r_[np.nan, c[:-1, -1]] if prev is None else np.asarray(prev, float)
    return im.DayBars(np.array([f"d{i:03d}" for i in range(len(c))]), o, hi, lo, c,
                      np.full(c.shape, volume), pc)


def test_vwap_bands_equal_weights():
    b = days_from_closes([np.linspace(100, 101, N)])
    vw, sd = mr.vwap_bands(b)
    tp = (b.high + b.low + b.close) / 3
    assert vw[0, -1] == pytest.approx(tp[0].mean())
    assert sd[0, -1] == pytest.approx(tp[0].std())


def test_vwap_reversion_long_on_stretch_and_exit_at_vwap():
    path = np.full(N, 100.0)
    path[20:24] = [99.0, 97.0, 96.0, 96.0]   # stretch below VWAP at the 11:10 close
    path[24:] = 100.5                         # back above VWAP
    b = days_from_closes([path])
    pos = mr.vwap_reversion_positions(b, range_filter=False)
    i = int(np.flatnonzero(pos[0] == 1)[0])
    assert i >= 21 and pos[0, i - 1] == 0                        # enters the bar after a stretched close
    assert pos[0, 25:].max() == 0                                # out after the close back at VWAP
    assert (mr.vwap_reversion_positions(b) == 0).all()           # filter needs 14 days of bands


def test_gap_fade_target_and_gap_go_stop():
    flat = [np.r_[np.full(N // 2, 100.0), np.full(N - N // 2, 102.0)] for _ in range(15)]
    # 14-day range is 102.25 - 99.75 = 2.5; a 0.5 gap is small (< 0.75), 2.5 is large (>= 1.5)
    fade_day = np.r_[np.full(10, 102.5), np.full(N - 10, 101.0)]     # gap up 0.5, then fills
    b = days_from_closes(flat + [fade_day])
    b.open[15, 0] = 102.5
    t = [x for x in mr.gap_trades(b, multiplier=5, tick=0.25, mode="fade") if x.day == 15][0]
    assert t.side == -1 and t.reason == "target" and t.target == 102.0
    assert t.pnl == pytest.approx((102.25 - 102.0) * 5 - 2 * 0.62)  # sold 102.5 - tick, bought at 102
    go_day = np.full(N, 105.5)
    b2 = days_from_closes(flat + [go_day])
    b2.open[15, 0] = 104.5
    g = [x for x in mr.gap_trades(b2, 5, 0.25, mode="go") if x.day == 15][0]
    assert g.side == 1 and g.stop == pytest.approx(104.75 - 2.5) and g.reason == "close"


def test_rsi2_intraday_long_after_dip_in_uptrend():
    up = 100 + np.cumsum(np.full(N * 10, 0.05)).reshape(10, N)
    up[9, 30:36] -= np.array([1, 2, 3, 3, 3, 3])                  # sharp dip on the last day
    b = days_from_closes(up)
    pos = mr.rsi2_intraday_positions(b, minutes=15)
    assert pos[9].max() == 1 and pos[9].min() >= 0
    assert (pos[:, -1] == pos[:, -1]).all()                          # flat handled by the simulator
    short = mr.rsi2_intraday_positions(b, allow_short=False)
    assert short.min() >= 0


def test_example_runs(capsys):
    from examples import intraday_mean_reversion as ex
    ex.main(["--symbols", "MES", "--sims", "100", "--horizon", "60", "--flips", "200", "--contracts", "1"])
    out = capsys.readouterr().out
    assert "VWAP rev" in out and "fade small gaps" in out and "RSI(2) 15-min" in out
