import csv
from pathlib import Path

import pytest

from examples import rsi2_trade_list as ex

CSV = Path(__file__).parents[1] / "pine" / "rsi2_mes_reference_trades.csv"


def test_reference_csv_is_current():
    rows = ex.reference_trades()
    with open(CSV) as f:
        saved = list(csv.DictReader(f))
    assert len(saved) == len(rows) == 60
    for got, want in zip(rows, saved):
        assert got[:3] == (want["signal_close"], want["first_session"], want["last_session"])
        assert got[3] == int(want["sessions"])
        assert got[4] == pytest.approx(float(want["prop_pnl_1_contract"]))
    # the flat-through-breaks total reported in the study: +$8,643 per contract
    assert sum(r[4] for r in rows) == pytest.approx(8643, abs=1)


def test_pine_script_matches_the_tested_rules():
    src = (Path(__file__).parents[1] / "pine" / "rsi2_pullback.pine").read_text()
    assert "//@version=6" in src
    for snippet in ('input.int(2, "RSI length"', 'input.float(10, "Buy when RSI is below"',
                    'input.int(200, "Trend average (days)"', 'input.int(5, "Exit average (days)"',
                    "commission_value = 0.62", "slippage = 1", "default_qty_value = 2"):
        assert snippet in src
