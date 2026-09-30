import csv
from pathlib import Path

import pytest

from examples import rsi2_trade_list as ex

PINE = Path(__file__).parents[1] / "pine"


# symbol, trades, flat-through-breaks total per contract reported in the study
@pytest.mark.parametrize("symbol,n,total", [("MES", 60, 8643), ("MNQ", 55, 9116)])
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
                    "commission_value = 0.62", "slippage = 1",
                    'root == "MNQ" ? 1 : root == "M2K" ? 3 : 2', "qty = qty"):
        assert snippet in src
