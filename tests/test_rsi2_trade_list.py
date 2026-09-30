import csv
from pathlib import Path

import numpy as np
import pytest

from examples import rsi2_trade_list as ex

PINE = Path(__file__).parents[1] / "pine"


# symbol, trades, flat-through-breaks total per contract reported in the study
@pytest.mark.parametrize("symbol,n,total", [("MES", 60, 8643), ("MNQ", 55, 9116), ("M2K", 51, 2515), ("MYM", 60, 3445), ("MGC", 37, 748), ("MCL", 20, -981)])
def test_reference_csv_is_current(symbol, n, total):
    rows = ex.reference_trades(symbol)
    with open(PINE / f"rsi2_{symbol.lower()}_reference_trades.csv") as f:
        saved = list(csv.DictReader(f))
    assert len(saved) == len(rows) == n
    for got, want in zip(rows, saved):
        assert got[:3] == (want["signal_close"], want["first_session"], want["last_session"])
        assert got[3] == int(want["sessions"])
        assert got[4] == pytest.approx(float(want["prop_pnl_1_contract"]))
    assert sum(r[4] for r in rows) == pytest.approx(total, abs=1)


def test_pine_script_matches_the_tested_rules():
    src = (PINE / "rsi2_pullback.pine").read_text()
    assert "//@version=6" in src
    for snippet in ('input.int(2, "RSI length"', 'input.float(10, "Buy when RSI is below"',
                    'input.int(200, "Trend average (days)"', 'input.int(5, "Exit average (days)"',
                    "commission_value = 0.62", "slippage = 1", "margin_long = 10",
                    'root == "MNQ" or root == "MGC" or root == "MCL" ? 1 : root == "M2K" or root == "MYM" ? 3 : 2', "qty = qty"):
        assert snippet in src


def test_combined_plans_and_example(capsys):
    from examples import rsi2_combined as cb
    mes, mnq = np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])
    p = cb.plans(mes, mnq)
    np.testing.assert_array_equal(p["2 MES, else 1 MNQ"][1], [0, 0, 1, 0])
    np.testing.assert_array_equal(p["1 MES + 1 MNQ"][0], mes)
    q = cb.plans(mes, mnq, "MYM")
    np.testing.assert_array_equal(q["2 MES, else 3 MYM"][1], [0, 0, 3, 0])
    np.testing.assert_array_equal(q["1 MES + 2 MYM"][1], [2, 0, 2, 0])
    cb.main(["--sims", "100", "--horizon", "60"])
    out = capsys.readouterr().out
    assert "both 143" in out and "both 141" in out and "+17,287" in out
    assert "MYM alone 76" in out


def test_combined_pine_alert_matches_the_rules():
    src = (PINE / "rsi2_combined_alerts.pine").read_text()
    for snippet in ("//@version=6", 'input.int(2, "RSI length"', 'input.float(10, "Buy when RSI is below"',
                    'input.int(200, "Trend average (days)"', 'input.int(5, "Exit average (days)"',
                    "backadjustment = backadjustment.on", "PLAN_MES", "alert(msg, alert.freq_once_per_bar_close)",
                    'options = ["MNQ", "MYM"]', 'qtyB   = second == "MNQ" ? 1 : plan == PLAN_BOTH ? 2 : 3'):
        assert snippet in src
