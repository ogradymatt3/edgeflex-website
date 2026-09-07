#!/usr/bin/env python3
"""Paper (default) or live trading loop on Kraken.

  python run_live.py                      # paper, BTC/USD 1h, $100
  python run_live.py --once               # one evaluation, then exit
  python run_live.py --live               # REAL orders; needs KRAKEN_API_KEY / KRAKEN_API_SECRET
"""
import argparse

from algo import live
from algo.strategy import Params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="BTC/USD")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--cash", type=float, default=100.0)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    if a.live:
        confirm = input("Live mode places REAL orders. Type 'I understand' to continue: ")
        if confirm.strip() != "I understand":
            return
    live.run(a.pair, a.interval, a.cash, Params(), a.live, a.once)


if __name__ == "__main__":
    main()
