import gzip

import numpy as np
import pytest

from hmm_trader import futures as fu
from hmm_trader import orb

MES = fu.CONTRACTS["MES"]  # $5/pt, tick 0.25 -> $1.25


def day(bars, date="2025-01-02", start=9 * 60 + 30):
    """Build IntradayBars from (o, h, l, c) tuples of consecutive 5-min bars."""
    arr = np.array(bars, dtype=float)
    n = len(arr)
    return orb.IntradayBars(np.array([date] * n), start + 5 * np.arange(n),
                            arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])


def concat(*days):
    return orb.IntradayBars(*(np.concatenate([getattr(d, f) for d in days])
                              for f in ("date", "minute", "open", "high", "low", "close")))


# 5-minute opening range 100-102; no slippage/commission unless a test adds them
P = dict(range_minutes=5, slippage_ticks=0, commission_per_side=0, risk_dollars=100)


def run(bars, **kw):
    return orb.backtest(bars, MES, orb.ORBParams(**{**P, **kw}))


def test_long_breakout_hits_target():
    # buy stop 102.25, sell stop 99.75 -> risk 2.5 pts = $12.50/contract -> 8 contracts
    t = run(day([(101, 102, 100, 101), (101, 102.5, 101, 102.4), (102.4, 107.5, 102, 107)]),
            target_r=2.0)
    assert t.side[0] == 1 and t.reason[0] == "target"
    assert t.entry[0] == 102.25 and t.contracts[0] == 8
    assert t.exit[0] == pytest.approx(102.25 + 5.0)
    assert t.pnl[0] == pytest.approx(5.0 * 8 * 5)
    assert t.r_multiple[0] == pytest.approx(2.0)


def test_short_breakout_stopped_out():
    t = run(day([(101, 102, 100, 101), (100.5, 100.6, 99.5, 99.6), (99.6, 102.5, 99, 102)]),
            target_r=2.0)
    assert t.side[0] == -1 and t.reason[0] == "stop"
    assert t.exit[0] == 102.25
    assert t.r_multiple[0] == pytest.approx(-1.0)


def test_stop_wins_when_bar_touches_both():
    t = run(day([(101, 102, 100, 101), (101, 102.5, 101, 102.4), (103, 108, 99, 104)]),
            target_r=2.0)
    assert t.reason[0] == "stop"


def test_gap_through_stop_fills_at_open():
    t = run(day([(101, 102, 100, 101), (101, 102.5, 101, 102.4), (98, 99, 97, 98)]))
    assert t.reason[0] == "stop" and t.exit[0] == 98


def test_flatten_at_last_bar_close_and_slippage_costs():
    bars = day([(101, 102, 100, 101), (101, 102.5, 101, 102.4), (102.4, 103, 102, 102.9)])
    t = run(bars, target_r=None, flatten=9 * 60 + 45)
    assert t.reason[0] == "flatten" and t.exit[0] == 102.9
    t2 = run(bars, target_r=None, flatten=9 * 60 + 45, slippage_ticks=1, commission_per_side=1)
    # entry 1 tick worse, exit 1 tick worse, $1/side commission per contract
    n = t2.contracts[0]
    assert t2.entry[0] == 102.5 and t2.exit[0] == 102.65
    assert t2.pnl[0] == pytest.approx((102.65 - 102.5) * 5 * n - 2 * n)


def test_ambiguous_bar_and_too_wide_skip():
    t = run(day([(101, 102, 100, 101), (101, 103, 99, 101)]))
    assert t.side[0] == 0 and t.reason[0] == "ambiguous"
    t = run(day([(101, 102, 100, 101), (101, 102.5, 101, 102.4)]), risk_dollars=10)
    assert t.side[0] == 0 and t.reason[0] == "too_wide"


def test_direction_filter_and_no_breakout():
    bars = day([(101, 102, 100, 101), (100.5, 100.6, 99.5, 99.6)])
    assert run(bars, direction="long").reason[0] == "none"
    assert run(bars, direction="short").side[0] == -1


def test_incomplete_opening_range_skips_day():
    good = day([(101, 102, 100, 101), (101, 101.5, 100.5, 101)], date="2025-01-02")
    late = day([(101, 102, 100, 101)], date="2025-01-03", start=9 * 60 + 35)
    t = run(concat(good, late))
    assert list(t.date) == ["2025-01-02"]


def test_steps_sum_to_pnl_and_order_peak_then_trough():
    bars = day([(101, 102, 100, 101), (101, 102.5, 101, 102.4), (102.4, 104, 101, 103),
                (103, 103.5, 102.5, 103)])
    t = run(bars, target_r=None, commission_per_side=0.5)
    np.testing.assert_allclose(t.steps.sum(axis=1), t.pnl)
    peak, down, _ = t.steps[0]
    n = t.contracts[0]
    assert peak == pytest.approx((104 - 102.25) * 5 * n - 0.5 * n)
    assert peak + down == pytest.approx((101 - 102.25) * 5 * n - 0.5 * n)


def test_summary_and_params_validation():
    t = run(day([(101, 102, 100, 101), (101, 102.5, 101, 102.4), (102.4, 107.5, 102, 107)]))
    s = orb.summary(t)
    assert s["trades"] == 1 and s["win_rate"] == 1.0 and s["net"] > 0
    with pytest.raises(ValueError):
        orb.ORBParams(range_minutes=7)
    with pytest.raises(ValueError):
        orb.ORBParams(target_r=0)
    with pytest.raises(ValueError):
        orb.ORBParams(direction="up")
    assert orb.ORBParams(range_minutes=30, target_r=None).label() == "ORB 30m / EOD"


def test_load_intraday_gz_and_sorting(tmp_path):
    p = tmp_path / "x.csv.gz"
    with gzip.open(p, "wt") as fh:
        fh.write("datetime,open,high,low,close,contract\n"
                 "2025-01-02 09:35,2,3,1,2,MESH5\n2025-01-02 09:30,1,2,0.5,1.5,MESH5\n")
    b = orb.load_intraday(p)
    assert list(b.minute) == [570, 575]
    assert list(b.open) == [1, 2]
    with gzip.open(p, "wt") as fh:
        fh.write("datetime,open,high,low,close\n2025-01-02 09:30,1,1,1,1\n"
                 "2025-01-02 09:30,1,1,1,1\n")
    with pytest.raises(ValueError, match="duplicate"):
        orb.load_intraday(p)


def test_bundled_data_is_regular_session_front_month():
    from pathlib import Path
    b = orb.load_intraday(Path(__file__).parents[1] / "examples" / "data" / "mes_5min_rth.csv.gz")
    assert b.minute.min() == 9 * 60 + 30 and b.minute.max() == 15 * 60 + 55
    assert np.all(b.high >= np.maximum(b.open, b.close))
    assert np.all(b.low <= np.minimum(b.open, b.close))
