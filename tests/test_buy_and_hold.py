from pathlib import Path

import numpy as np

from examples import buy_and_hold as ex


def test_strategies_and_known_outcomes():
    dates, start, strats = ex.strategies()
    assert dates[start] == "2020-02-25"
    hold, active = strats["Buy & hold, 1 MES"]
    assert active.all()
    assert round(float(hold[start:].sum())) == 18368
    # the evaluation from the first day fails in the March 2020 crash, on session 10
    res = ex.every_start(hold[start:], 250)
    assert res.outcomes[0] == "max_drawdown" and int(res.days[0]) == 10
    rsi_hold = strats["RSI(2), 2 MES, held overnight"][0]
    assert round(float(rsi_hold[start:].sum())) == 19339


def test_example_runs(capsys):
    ex.main(["--sims", "100", "--horizon", "60"])
    out = capsys.readouterr().out
    assert "Pass rate by the year" in out and "Net $ by calendar year" in out
    assert "+18,368" in out and "-4,737" in out


def test_pine_script_defaults():
    src = (Path(__file__).parents[1] / "pine" / "buy_and_hold.pine").read_text()
    for snippet in ("//@version=6", "commission_value = 0.62", "slippage = 1", "margin_long = 10",
                    'input.float(3000, "Profit target ($)"', 'input.float(2000, "Trailing drawdown ($)"',
                    'input.int(250, "Sessions allowed"', 'timestamp("2020-02-25T00:00:00")'):
        assert snippet in src
