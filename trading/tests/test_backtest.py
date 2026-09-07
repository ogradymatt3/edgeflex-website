import numpy as np
import pandas as pd

from algo import backtest, data
from algo.strategy import Params, indicators


def test_no_lookahead_in_indicators():
    df = data.synthetic(n=2000)
    d = indicators(df, Params())
    i = 500
    # breakout level at bar i must use only bars < i
    assert d["hh"].iloc[i] == df["high"].iloc[i - 48 : i].max()


def test_engine_runs_and_conserves_value():
    df = data.synthetic(n=3000)
    r = backtest.run(df, Params(), backtest.Costs(), 100.0, 60)
    assert len(r.equity) == len(df)
    assert r.equity.iloc[0] == 100.0
    assert (r.equity > 0).all()
    assert r.stats["trades"] > 0


def test_fees_hurt():
    df = data.synthetic(n=3000)
    free = backtest.run(df, Params(), backtest.Costs(0, 0, 0), 100.0, 60).stats["final_equity"]
    paid = backtest.run(df, Params(), backtest.Costs(0.01, 0.01, 0.001), 100.0, 60).stats["final_equity"]
    assert paid < free


def test_flat_market_does_nothing_crazy():
    idx = pd.date_range("2024-01-01", periods=1500, freq="60min", tz="UTC")
    px = 100 + 0.01 * np.sin(np.arange(1500) / 10)
    df = pd.DataFrame({"open": px, "high": px + 0.02, "low": px - 0.02, "close": px, "volume": 1.0}, index=idx)
    r = backtest.run(df, Params(), backtest.Costs(), 100.0, 60)
    assert r.equity.iloc[-1] > 95
