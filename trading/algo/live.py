"""Live loop: every bar, refresh data, compute the signal, reconcile the position.

Default is paper. The loop is deliberately simple and stateless per bar: it recomputes
the desired weight from scratch and trades the difference, so a restart is safe.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import data
from .broker import KrakenBroker, PaperBroker
from .risk import Limits, RiskManager
from .strategy import Params, indicators, target_weight

STATE = Path(__file__).resolve().parent.parent / "data" / "live_state.json"


def desired_weight(df, p: Params, interval_min: int, in_position: bool, stop: float | None):
    """Same rules as the backtester, evaluated on the last CLOSED bar."""
    d = indicators(df, p)
    row = d.iloc[-2]  # -1 is the still-forming bar
    bars_per_year = 365 * 24 * 60 / interval_min
    if not np.isfinite(row["hh"]) or not np.isfinite(row["atr"]):
        return 0.0, stop, "warmup"
    new_stop = row["close"] - p.atr_stop_mult * row["atr"]
    if in_position:
        stop = max(stop or -np.inf, new_stop)
        if row["close"] < row["ll"] or row["close"] < row["ema"] or row["low"] <= stop:
            return 0.0, None, "exit"
        return None, stop, "hold"  # None = leave position as is
    if row["close"] > row["hh"] and row["close"] > row["ema"]:
        return target_weight(row, p, bars_per_year), new_stop, "enter"
    return 0.0, None, "flat"


def run(pair: str, interval_min: int, cash: float, p: Params, live: bool = False, once: bool = False,
        limits: Limits = Limits()):
    broker = KrakenBroker(data.KRAKEN_PAIRS.get(pair, pair)) if live else PaperBroker(cash)
    state = json.loads(STATE.read_text()) if STATE.exists() else {"stop": None, "units": 0.0}
    risk = RiskManager(limits, cash)
    print(f"[{'LIVE' if live else 'PAPER'}] {pair} {interval_min}m  params={p}")
    while True:
        df = data.load(pair, interval_min, source="kraken")
        price = float(df["close"].iloc[-1])
        units = state["units"] if live else broker.units
        eq = cash if live else broker.equity(price)
        ok, why = risk.check(eq, datetime.now(timezone.utc))
        w, stop, action = desired_weight(df, p, interval_min, units > 0, state["stop"])
        if not ok:
            action, w = f"blocked: {why}", 0.0 if risk.halted else None
        if w is not None:
            target_units = risk.cap_weight(w) * eq / price
            delta = target_units - units
            if abs(delta) * price >= 5.0:
                side = "buy" if delta > 0 else "sell"
                fill = broker.order(side, abs(delta), price, post_only=True)
                print(f"  {side} {abs(delta):.6f} @ {price:.2f} -> {fill}")
                state["units"] = units + delta
        state["stop"] = stop
        STATE.parent.mkdir(exist_ok=True)
        STATE.write_text(json.dumps(state))
        print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M}  px={price:.2f}  eq={eq:.2f}  action={action}  stop={stop}")
        if once:
            break
        time.sleep(interval_min * 60 - time.time() % (interval_min * 60) + 5)  # wake just after bar close
