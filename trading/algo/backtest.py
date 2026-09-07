"""Bar-by-bar backtester with fees, slippage, vol-targeted sizing and a trailing stop.

Orders decided on bar t's close are filled at bar t+1's open (no lookahead). Entries are
modeled as maker (limit) fills, exits as taker (stop hit, market out). This is pessimistic
on exits and roughly right for a patient entry style.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .strategy import Params, indicators, target_weight


@dataclass
class Costs:
    maker_fee: float = 0.0025   # Kraken Pro base tier maker; Coinbase Advanced base is higher
    taker_fee: float = 0.0040
    slippage: float = 0.0005    # half-spread + impact, per side


@dataclass
class Result:
    equity: pd.Series
    trades: pd.DataFrame
    stats: dict = field(default_factory=dict)


def run(df: pd.DataFrame, p: Params = Params(), costs: Costs = Costs(), start_cash: float = 100.0,
        interval_min: int = 60, min_order_usd: float = 5.0) -> Result:
    bars_per_year = 365 * 24 * 60 / interval_min
    d = indicators(df, p)
    o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    hh, ll, ema, atr_ = d["hh"].values, d["ll"].values, d["ema"].values, d["atr"].values

    cash, units, stop = start_cash, 0.0, np.nan
    entry_px, entry_i = np.nan, -1
    pending = 0.0  # target weight to execute at next open (NaN = no order)
    pending_set = False
    equity = np.empty(len(d))
    trades = []

    for i in range(len(d)):
        # 1) execute the order decided on the previous bar at this bar's open
        if pending_set:
            pending_set = False
            eq = cash + units * o[i]
            target_units = pending * eq / o[i]
            delta = target_units - units
            if abs(delta) * o[i] >= min_order_usd:
                if delta > 0:
                    px = o[i] * (1 + costs.slippage)
                    fee = costs.maker_fee
                    entry_px, entry_i = px, i
                else:
                    px = o[i] * (1 - costs.slippage)
                    fee = costs.taker_fee
                notional = delta * px
                cash -= notional + abs(notional) * fee
                units += delta
                if units * o[i] < min_order_usd:  # fully out
                    trades.append({"entry_time": d.index[entry_i], "exit_time": d.index[i], "entry": entry_px,
                                   "exit": px, "ret": px / entry_px - 1, "bars": i - entry_i, "reason": "signal"})
                    units, stop = 0.0, np.nan

        # 2) intrabar stop check (taker exit at the stop price, slippage applied)
        if units > 0 and np.isfinite(stop) and l[i] <= stop:
            px = stop * (1 - costs.slippage)
            cash += units * px * (1 - costs.taker_fee)
            trades.append({"entry_time": d.index[entry_i], "exit_time": d.index[i], "entry": entry_px,
                           "exit": px, "ret": px / entry_px - 1, "bars": i - entry_i, "reason": "stop"})
            units, stop = 0.0, np.nan

        equity[i] = cash + units * c[i]

        # 3) decide the order for the next bar using only information through this close
        if not np.isfinite(hh[i]) or not np.isfinite(ema[i]) or not np.isfinite(atr_[i]):
            continue
        if units > 0:
            stop = max(stop, c[i] - p.atr_stop_mult * atr_[i]) if np.isfinite(stop) else c[i] - p.atr_stop_mult * atr_[i]
            if c[i] < ll[i] or c[i] < ema[i]:
                pending, pending_set = 0.0, True
        else:
            if c[i] > hh[i] and c[i] > ema[i]:
                w = target_weight(d.iloc[i], p, bars_per_year)
                if w > 0:
                    pending, pending_set = w, True
                    stop = c[i] - p.atr_stop_mult * atr_[i]

    eq = pd.Series(equity, index=d.index, name="equity")
    tr = pd.DataFrame(trades)
    return Result(eq, tr, stats(eq, tr, bars_per_year, start_cash))


def stats(eq: pd.Series, trades: pd.DataFrame, bars_per_year: float, start_cash: float) -> dict:
    rets = eq.pct_change().dropna()
    years = len(eq) / bars_per_year
    total = eq.iloc[-1] / start_cash - 1
    cagr = (eq.iloc[-1] / start_cash) ** (1 / years) - 1 if years > 0 else np.nan
    vol = rets.std() * np.sqrt(bars_per_year)
    sharpe = rets.mean() / rets.std() * np.sqrt(bars_per_year) if rets.std() > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min()
    win = (trades["ret"] > 0).mean() if len(trades) else np.nan
    return {
        "final_equity": round(float(eq.iloc[-1]), 2),
        "total_return": round(float(total), 4),
        "cagr": round(float(cagr), 4),
        "ann_vol": round(float(vol), 4),
        "sharpe": round(float(sharpe), 2),
        "max_drawdown": round(float(dd), 4),
        "trades": int(len(trades)),
        "win_rate": round(float(win), 3) if len(trades) else None,
        "avg_trade": round(float(trades["ret"].mean()), 4) if len(trades) else None,
        "years": round(years, 2),
    }


def buy_and_hold(df: pd.DataFrame, start_cash: float, interval_min: int, costs: Costs = Costs()) -> dict:
    eq = start_cash * (1 - costs.taker_fee) * df["close"] / df["close"].iloc[0]
    return stats(eq, pd.DataFrame(), 365 * 24 * 60 / interval_min, start_cash)


def walk_forward(df: pd.DataFrame, grid: list[Params], costs: Costs, start_cash: float, interval_min: int,
                 n_folds: int = 4) -> pd.DataFrame:
    """Pick params on each in-sample fold, score on the following out-of-sample fold.

    If OOS Sharpe collapses relative to IS, the edge is fit to noise. That is the whole point of this.
    """
    n = len(df)
    edges = np.linspace(0, n, n_folds + 2, dtype=int)
    rows = []
    for k in range(n_folds):
        ins, oos = df.iloc[edges[k] : edges[k + 1]], df.iloc[edges[k + 1] : edges[k + 2]]
        best = max(grid, key=lambda p: run(ins, p, costs, start_cash, interval_min).stats["sharpe"])
        is_s = run(ins, best, costs, start_cash, interval_min).stats
        oos_s = run(oos, best, costs, start_cash, interval_min).stats
        rows.append({"fold": k, "entry": best.entry_lookback, "exit": best.exit_lookback, "stop": best.atr_stop_mult,
                     "is_sharpe": is_s["sharpe"], "oos_sharpe": oos_s["sharpe"], "oos_return": oos_s["total_return"],
                     "oos_maxdd": oos_s["max_drawdown"], "oos_trades": oos_s["trades"]})
    return pd.DataFrame(rows)
