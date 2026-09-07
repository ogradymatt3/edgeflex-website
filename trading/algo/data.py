"""OHLCV data: fetch from Kraken / Coinbase public endpoints, cache to CSV, or synthesize."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "data"
COLS = ["time", "open", "high", "low", "close", "volume"]

KRAKEN_PAIRS = {"BTC/USD": "XBTUSD", "ETH/USD": "ETHUSD", "SOL/USD": "SOLUSD"}


def _to_df(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLS).astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.drop_duplicates("time").sort_values("time").set_index("time")


def fetch_kraken(pair: str = "BTC/USD", interval_min: int = 60) -> pd.DataFrame:
    """Kraken returns the last ~720 candles per call. Merges with the local cache so history grows."""
    import requests

    r = requests.get(
        "https://api.kraken.com/0/public/OHLC",
        params={"pair": KRAKEN_PAIRS.get(pair, pair.replace("/", "")), "interval": interval_min},
        timeout=20,
    )
    r.raise_for_status()
    body = r.json()
    if body.get("error"):
        raise RuntimeError(body["error"])
    key = next(k for k in body["result"] if k != "last")
    rows = [[c[0], c[1], c[2], c[3], c[4], c[6]] for c in body["result"][key]]
    return _to_df(rows)


def fetch_coinbase(pair: str = "BTC/USD", interval_min: int = 60, days: int = 365) -> pd.DataFrame:
    """Coinbase Exchange public candles, 300 per call, paged backwards. Good for multi-year history."""
    import requests

    product = pair.replace("/", "-")
    gran = interval_min * 60
    end = int(time.time())
    start_limit = end - days * 86400
    frames = []
    while end > start_limit:
        start = max(end - 300 * gran, start_limit)
        r = requests.get(
            f"https://api.exchange.coinbase.com/products/{product}/candles",
            params={"granularity": gran, "start": start, "end": end},
            timeout=20,
        )
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        # coinbase: [time, low, high, open, close, volume]
        frames.append(_to_df([[c[0], c[3], c[2], c[1], c[4], c[5]] for c in rows]))
        end = start
        time.sleep(0.25)  # public rate limit is ~10 req/s; be polite
    if not frames:
        raise RuntimeError("no candles returned")
    return pd.concat(frames).sort_index()
    

def load(pair: str = "BTC/USD", interval_min: int = 60, source: str = "coinbase", days: int = 365,
         refresh: bool = True) -> pd.DataFrame:
    """Load candles, merging fresh data into the CSV cache so history accumulates across runs."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{pair.replace('/', '')}_{interval_min}m.csv"
    cached = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else None
    if refresh:
        fresh = fetch_coinbase(pair, interval_min, days) if source == "coinbase" else fetch_kraken(pair, interval_min)
        df = pd.concat([cached, fresh]) if cached is not None else fresh
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_csv(path)
        return df
    if cached is None:
        raise FileNotFoundError(f"no cache at {path}; run with refresh=True")
    return cached


def synthetic(n: int = 24 * 365 * 2, seed: int = 7, interval_min: int = 60) -> pd.DataFrame:
    """Regime-switching random walk with realistic crypto-ish hourly vol. For engine tests only.

    Trend regimes give a trend-follower something to catch; the flat regime is pure noise so
    the test also shows what a strategy does when there is no edge.
    """
    rng = np.random.default_rng(seed)
    hourly_vol = 0.012
    drift = np.zeros(n)
    i = 0
    while i < n:
        length = int(rng.integers(24 * 10, 24 * 90))
        regime = rng.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
        drift[i : i + length] = regime * 0.0006
        i += length
    rets = drift + hourly_vol * rng.standard_normal(n) * (1 + 0.5 * (rng.random(n) < 0.05))
    close = 30_000 * np.exp(np.cumsum(rets))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    wick = np.abs(rng.standard_normal(n)) * hourly_vol * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    vol = rng.lognormal(3, 0.5, n)
    idx = pd.date_range("2024-01-01", periods=n, freq=f"{interval_min}min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
