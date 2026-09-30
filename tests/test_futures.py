import numpy as np
import pytest

from hmm_trader import futures as fu


def bars(dates, close, next_close, open_=None, high=None, low=None):
    close = np.asarray(close, dtype=float)
    open_ = close if open_ is None else np.asarray(open_, dtype=float)
    return fu.FuturesBars(
        np.asarray(dates), open_,
        np.maximum(open_, close) if high is None else np.asarray(high, dtype=float),
        np.minimum(open_, close) if low is None else np.asarray(low, dtype=float),
        close, np.asarray(next_close, dtype=float),
    )


def test_quarterly_expiries_third_friday_and_juneteenth():
    exp = fu.quarterly_expiries([2024, 2026])
    assert [d.isoformat() for d in exp[:4]] == [
        "2024-03-15", "2024-06-21", "2024-09-20", "2024-12-20"]
    assert exp[5].isoformat() == "2026-06-18"  # June 19 is a holiday -> Thursday


def test_roll_mask_three_sessions_before_expiry():
    # March 2024 expiry Fri 15th -> roll on Tue 12th
    dates = ["2024-03-08", "2024-03-11", "2024-03-12", "2024-03-13", "2024-03-14",
             "2024-03-15", "2024-03-18"]
    assert list(np.flatnonzero(fu.quarterly_roll_mask(dates))) == [2]
    # holiday inside the window (Juneteenth 2024 was a Wednesday) -> Monday
    dates = ["2024-06-13", "2024-06-14", "2024-06-17", "2024-06-18", "2024-06-20",
             "2024-06-21"]
    assert list(np.flatnonzero(fu.quarterly_roll_mask(dates))) == [2]
    # data ending in the roll week still flags the roll
    assert list(np.flatnonzero(fu.quarterly_roll_mask(dates[:4]))) == [2]
    assert not fu.quarterly_roll_mask(dates[:2]).any()


def test_bundled_data_rolls_every_quarter():
    b = fu.load_bars(fu.Path(__file__).parents[1] / "examples" / "data" / "mes_daily.csv")
    mask = fu.quarterly_roll_mask(b.dates)
    assert mask.sum() == 30  # 2019-06 .. 2026-09
    assert "2026-06-15" in b.dates[mask]
    # roll sessions confirmed against the S&P 500 cash index
    for d in ("2019-06-18", "2024-06-17", "2025-03-18", "2026-09-15"):
        assert d in b.dates[mask]
    # since rates rose, each roll moves into a more expensive contract (contango)
    idx = np.flatnonzero(mask)
    idx = idx[b.dates[idx] > "2022-09"]
    spread = b.next_close[idx - 1] / b.close[idx - 1] - 1
    assert np.all((spread > 0) & (spread < 0.02))


def test_point_changes_and_returns_use_second_month_on_roll():
    b = bars(["d1", "d2", "d3"], close=[100, 102, 103], next_close=[101, 103, 104],
             open_=[100, 101.5, 102])
    roll = np.array([False, True, False])
    change, gap = fu.point_changes(b, roll)
    np.testing.assert_allclose(change, [102 - 101, 103 - 102])
    np.testing.assert_allclose(gap, [101.5 - 101, 102 - 102])
    np.testing.assert_allclose(fu.adjusted_returns(b, roll), [102 / 101 - 1, 103 / 102 - 1])


def test_intraday_steps_sum_to_session_pnl():
    b = bars(["d1", "d2"], close=[100, 104], next_close=[100, 104],
             open_=[100, 101], high=[100, 106], low=[100, 99])
    roll = np.zeros(2, dtype=bool)
    over = fu.intraday_long_steps(b, roll, hold_overnight=True)
    np.testing.assert_allclose(over, [[1, 5, -7, 5]])
    assert over.sum() == pytest.approx(4)  # close to close
    flat = fu.intraday_long_steps(b, roll, hold_overnight=False)
    np.testing.assert_allclose(flat, [[0, 5, -7, 5]])  # open to close only


def test_contract_specs():
    assert fu.CONTRACTS["MES"].tick_value == pytest.approx(1.25)
    assert fu.CONTRACTS["MNQ"].tick_value == pytest.approx(0.50)


def test_load_bars_validates(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("date,open,high,low,close\n2024-01-02,1,1,1,1\n")
    with pytest.raises(ValueError, match="next_close"):
        fu.load_bars(p)
    p.write_text("date,open,high,low,close,next_close\n2024-01-03,1,1,1,1,1\n"
                 "2024-01-02,1,1,1,1,1\n")
    with pytest.raises(ValueError, match="increasing"):
        fu.load_bars(p)


def test_m2k_rolls_one_session_later():
    assert fu.CONTRACTS["M2K"].roll_sessions == 2
    assert fu.CONTRACTS["MES"].roll_sessions == 3
    b = fu.load_bars(fu.Path(__file__).parents[1] / "examples" / "data" / "m2k_daily.csv")
    mask = fu.quarterly_roll_mask(b.dates, fu.CONTRACTS["M2K"].roll_sessions)
    assert mask.sum() == 30
    # Wednesday of expiry week; Tuesday when Juneteenth falls in it (checked vs RUT)
    for d in ("2024-12-18", "2025-06-17", "2026-06-16", "2026-09-16"):
        assert d in b.dates[mask]


def test_gold_roll_mask_and_dispatch():
    b = fu.load_bars(fu.Path(__file__).parents[1] / "examples" / "data" / "mgc_daily.csv")
    spec = fu.CONTRACTS["MGC"]
    mask = fu.roll_mask(b.dates, spec)
    # matched against individual MGC contracts (Massive, 2024-2026)
    for d in ("2024-11-27", "2025-01-30", "2025-03-28", "2025-05-29", "2025-07-30",
              "2025-11-26", "2026-01-29", "2026-05-28", "2026-07-30"):
        assert d in b.dates[mask]
    assert {d[5:7] for d in b.dates[mask]} == {"01", "03", "05", "07", "11"}
    # second-to-last session of the month; an unfinished last month is skipped
    dates = ["2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31", "2025-02-03",
             "2025-03-27", "2025-03-28"]
    assert list(np.flatnonzero(fu.gold_roll_mask(dates))) == [2]
    np.testing.assert_array_equal(fu.roll_mask(b.dates, fu.CONTRACTS["MES"]),
                                  fu.quarterly_roll_mask(b.dates, 3))
    with pytest.raises(ValueError):
        fu.roll_mask(b.dates, fu.ContractSpec("X", 1.0, 1.0, roll_rule="weekly"))


def test_crude_roll_mask_and_holidays():
    b = fu.load_bars(fu.Path(__file__).parents[1] / "examples" / "data" / "mcl_daily.csv")
    mask = fu.roll_mask(b.dates, fu.CONTRACTS["MCL"])
    # matched against individual MCL contracts (Massive, 2024-10 .. 2026-09)
    observed = ["2024-10-18", "2024-11-18", "2024-12-17", "2025-01-16", "2025-02-19",
                "2025-03-18", "2025-04-17", "2025-05-16", "2025-06-17", "2025-07-18",
                "2025-08-18", "2025-09-18", "2025-10-17", "2025-11-18", "2025-12-17",
                "2026-01-15", "2026-02-19", "2026-03-18", "2026-04-17", "2026-05-15",
                "2026-06-17", "2026-07-17", "2026-08-18", "2026-09-18"]
    assert [d for d in b.dates[mask] if d >= "2024-10"] == observed
    np.testing.assert_array_equal(mask, fu.crude_roll_mask(b.dates))
    h = fu.us_exchange_holidays([2025, 2026])
    for d in ("2025-01-20", "2025-04-18", "2025-05-26", "2025-06-19", "2025-11-27",
              "2026-04-03", "2026-05-25", "2026-07-03", "2026-12-25"):
        assert d in h
    # New Year's on a Saturday (2022) is not moved back into December
    assert "2021-12-31" not in fu.us_exchange_holidays([2021, 2022])
