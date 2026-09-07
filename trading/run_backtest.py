#!/usr/bin/env python3
"""Backtest + walk-forward on real (or synthetic) data.

  python run_backtest.py                       # BTC/USD 1h, last 730 days from Coinbase
  python run_backtest.py --pair ETH/USD --days 365
  python run_backtest.py --synthetic           # engine check with no network
"""
import argparse
import itertools

import pandas as pd

from algo import backtest, data
from algo.strategy import Params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="BTC/USD")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--cash", type=float, default=100.0)
    ap.add_argument("--maker", type=float, default=0.0025)
    ap.add_argument("--taker", type=float, default=0.0040)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-refresh", action="store_true", help="use cached CSV only")
    a = ap.parse_args()

    df = data.synthetic() if a.synthetic else data.load(a.pair, a.interval, "coinbase", a.days, not a.no_refresh)
    costs = backtest.Costs(a.maker, a.taker)
    print(f"\n{len(df)} bars  {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}\n")

    res = backtest.run(df, Params(), costs, a.cash, a.interval)
    bh = backtest.buy_and_hold(df, a.cash, a.interval, costs)
    print("STRATEGY (default params)")
    for k, v in res.stats.items():
        print(f"  {k:14} {v}")
    print(f"\nBUY & HOLD     final={bh['final_equity']}  sharpe={bh['sharpe']}  maxdd={bh['max_drawdown']}")

    grid = [Params(e, x, atr_stop_mult=s) for e, x, s in itertools.product([24, 48, 96], [12, 24, 48], [2.0, 3.0, 4.0])
            if x < e]
    print("\nWALK-FORWARD (params chosen in-sample, scored out-of-sample)")
    wf = backtest.walk_forward(df, grid, costs, a.cash, a.interval)
    with pd.option_context("display.width", 140):
        print(wf.to_string(index=False))
    print(f"\n  mean IS sharpe {wf.is_sharpe.mean():.2f}   mean OOS sharpe {wf.oos_sharpe.mean():.2f}")
    print("  If OOS is far below IS, the parameters are fit to noise. Do not trade that.\n")


if __name__ == "__main__":
    main()
