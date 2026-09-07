"""Trend-following with a volatility filter. Long-only (spot exchange, no shorting).

Why this and not something fancier: breakout / trend strategies on liquid crypto at the
1h-4h horizon are the one family with a documented positive expectancy, they trade rarely
(so fees matter less), and they have few parameters (so they overfit less).

Signal:
  enter  when close breaks above the N-bar Donchian high AND close > slow EMA (regime filter)
  exit   on ATR trailing stop, or close below the M-bar Donchian low (M < N: exits are faster)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Params:
    entry_lookback: int = 48       # bars for breakout high
    exit_lookback: int = 24        # bars for breakout low
    ema_regime: int = 200          # only long when above this
    atr_len: int = 24
    atr_stop_mult: float = 3.0     # trailing stop distance in ATRs
    vol_target_annual: float = 0.40  # size positions so the portfolio runs ~40% annual vol
    max_leverage: float = 1.0      # spot: cannot exceed 1x


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def indicators(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    out = df.copy()
    out["ema"] = out["close"].ewm(span=p.ema_regime, adjust=False).mean()
    out["atr"] = atr(out, p.atr_len)
    # shift(1): today's breakout compares against PRIOR bars, never the current bar (no lookahead)
    out["hh"] = out["high"].rolling(p.entry_lookback).max().shift(1)
    out["ll"] = out["low"].rolling(p.exit_lookback).min().shift(1)
    out["ret"] = out["close"].pct_change()
    return out


def target_weight(row: pd.Series, p: Params, bars_per_year: float) -> float:
    """Fraction of equity to hold so realized vol lands near the target. Capped at 1x (spot)."""
    if not np.isfinite(row["atr"]) or row["atr"] <= 0:
        return 0.0
    bar_vol = row["atr"] / row["close"]
    ann_vol = bar_vol * np.sqrt(bars_per_year)
    return float(min(p.vol_target_annual / max(ann_vol, 1e-9), p.max_leverage))
