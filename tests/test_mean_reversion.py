import numpy as np
import pytest

from hmm_trader import futures as fu
from hmm_trader import mean_reversion as mr
from hmm_trader import orb

MES = fu.CONTRACTS["MES"]  # $5/pt, tick 0.25


def test_rsi_extremes_and_known_value():
    up = np.arange(10.0)
    assert np.isnan(mr.rsi(up, 2)[:2]).all()
    assert mr.rsi(up, 2)[-1] == 100.0
    assert mr.rsi(-up, 2)[-1] == 0.0
    # changes +1, -1 -> avg gain 0.5, avg loss 0.5 -> RSI 50
    assert mr.rsi(np.array([1.0, 2.0, 1.0]), 2)[2] == pytest.approx(50.0)


def test_sma():
    np.testing.assert_allclose(mr.sma(np.arange(5.0), 3), [np.nan, np.nan, 1, 2, 3])


def test_rsi2_positions_entry_exit_and_causality():
    # long uptrend (above the 3-day average), then a two-day dip, then a rally
    close = np.r_[np.arange(100.0, 110.0), 108.0, 106.0, 107.0, 112.0, 113.0]
    held = mr.rsi2_positions(close, entry=10, trend=3, exit_sma=2)
    assert held.size == close.size - 1
    # held[t] is the position during session t+1, decided at close t
    dip = 11  # close 106: RSI(2) ~ 0 but 106 < 3-day avg -> no entry
    assert not held[dip]
    # changing a later close never changes an earlier decision
    later = close.copy()
    later[-1] = 50.0
    np.testing.assert_array_equal(mr.rsi2_positions(later, 10, 3, 2)[:-1], held[:-1])


def test_rsi2_positions_enters_and_exits():
    # Wilder smoothing remembers the uptrend: a shallow dip does not reach RSI < 10
    close = np.r_[np.linspace(100, 130, 30), 126.0, 121.0, 127.0, 128.0]
    assert mr.rsi(close, 2)[31] < 10 < mr.rsi(close, 2)[30]
    held = mr.rsi2_positions(close, entry=10, trend=30, exit_sma=2)
    spans = mr.trade_spans(held)
    assert len(spans) == 1
    s, e = spans[0]
    assert s == 31  # bought at close 31 (121, above the 30-day avg), held into session 32
    assert e == 32  # close 32 (127) is above the 2-day average -> out after one session
    # the same dip below the trend filter is not bought
    assert not mr.rsi2_positions(close, entry=10, trend=10, exit_sma=2).any()
    # ... unless the filter is off
    assert mr.trade_spans(mr.rsi2_positions(close, entry=10, trend=None, exit_sma=2)) == [(31, 32)]


def test_trade_spans():
    assert mr.trade_spans(np.array([0, 1, 1, 0, 1], dtype=bool)) == [(1, 3), (4, 5)]
    assert mr.trade_spans(np.zeros(3, dtype=bool)) == []


def day(bars, date="2025-01-02", start=9 * 60 + 30):
    arr = np.array(bars, dtype=float)
    n = len(arr)
    return orb.IntradayBars(np.array([date] * n), start + 5 * np.arange(n),
                            arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])


FLAT = [(100, 100.5, 99.5, 100 + 0.25 * ((-1) ** i)) for i in range(6)]
P = dict(lookback=5, k=1.0, slippage_ticks=0, commission_per_side=0, risk_dollars=100)


def test_fade_long_exits_at_mean():
    # a drop closes below the band, next bar fills at the open, then a close
    # back above the 5-bar average exits
    bars = FLAT + [(100, 100, 98, 98), (98, 99, 97.9, 98.5), (98.5, 100.5, 98.4, 100.2)]
    t = mr.bollinger_fade(day(bars), MES, mr.FadeParams(**P))
    assert list(t.side) == [1]
    assert t.entry[0] == 98 and t.reason[0] == "mean" and t.exit[0] == 100.2
    assert t.pnl[0] == pytest.approx((100.2 - 98) * 5 * t.contracts[0])
    np.testing.assert_allclose(t.day_steps.sum(axis=1), t.day_pnl)


def test_fade_short_stopped_and_flatten():
    bars = FLAT + [(100, 102, 100, 102), (102, 110, 101.5, 109)]
    t = mr.bollinger_fade(day(bars), MES, mr.FadeParams(**P))
    assert list(t.side) == [-1] and t.reason[0] == "stop"
    assert t.r_multiple[0] == pytest.approx(-1.0)
    # no exit signal before the last tradable bar -> flatten at its close
    bars = FLAT + [(100, 100, 98, 98), (98, 98.2, 97.5, 97.8)]
    t = mr.bollinger_fade(day(bars), MES, mr.FadeParams(**{**P, "k": 1.0}))
    assert t.reason[-1] in ("flatten", "stop")
    assert t.day_pnl[0] == pytest.approx(t.pnl.sum())


def test_fade_costs_and_validation():
    bars = FLAT + [(100, 100, 98, 98), (98, 99, 97.9, 98.5), (98.5, 100.5, 98.4, 100.2)]
    net = mr.bollinger_fade(day(bars), MES, mr.FadeParams(**{**P, "slippage_ticks": 1,
                                                              "commission_per_side": 1}))
    n = net.contracts[0]
    assert net.entry[0] == 98.25 and net.exit[0] == 99.95
    assert net.pnl[0] == pytest.approx((99.95 - 98.25) * 5 * n - 2 * n)
    with pytest.raises(ValueError):
        mr.FadeParams(lookback=1)
    with pytest.raises(ValueError):
        mr.FadeParams(k=0)


def test_example_runs(capsys):
    from examples import mes_mean_reversion as ex
    ex.main(["--perms", "50", "--sims", "100", "--horizon", "60"])
    out = capsys.readouterr().out
    assert "A. Daily RSI(2)" in out and "timing test" in out
    assert "B. Intraday fade 20 bars / 2 sd" in out and "Sensitivity" in out


def test_example_part_a_other_symbols(capsys):
    from examples import mes_mean_reversion as ex
    ex.main(["--part", "a", "--symbols", "MNQ,M2K", "--perms", "20", "--sims", "50",
             "--horizon", "60"])
    out = capsys.readouterr().out
    assert "1 MNQ," in out and "1 M2K," in out
    with pytest.raises(SystemExit):
        ex.main(["--part", "a", "--symbols", "XYZ"])
