# trading

A small, honest crypto trading system. Long-only trend following on spot, vol-targeted sizing,
ATR trailing stop, real fee and slippage modeling, walk-forward validation, paper trading by default.

## Quick start (on your machine, exchange APIs are needed)

```bash
cd trading
pip install -r requirements.txt

# 1. backtest on 2 years of real BTC/USD hourly candles from Coinbase (public, no key)
python run_backtest.py
python run_backtest.py --pair ETH/USD --days 365

# 2. paper trade it live against Kraken prices (no key, no risk)
python run_live.py --once      # one evaluation
python run_live.py             # loop forever, wakes after each hourly close

# 3. real money (only after weeks of paper)
export KRAKEN_API_KEY=... KRAKEN_API_SECRET=...
python run_live.py --live
```

## How to read the backtest

- **STRATEGY vs BUY & HOLD**: if the strategy does not beat holding on risk-adjusted terms
  (Sharpe, max drawdown), it is not worth running.
- **WALK-FORWARD**: parameters are picked on one slice and scored on the next, unseen slice.
  If the out-of-sample Sharpe is far below in-sample, the "edge" is curve-fit. Do not trade it.
- **Fees**: defaults are Kraken Pro base tier (0.25% maker / 0.40% taker). Set `--maker/--taker`
  to your actual tier. On a $100 account this is the number that decides everything.

## Layout

| file | what |
|---|---|
| `algo/data.py` | Coinbase / Kraken public candles, CSV cache, synthetic generator |
| `algo/strategy.py` | Donchian breakout + EMA regime filter + ATR stop + vol targeting |
| `algo/backtest.py` | bar-by-bar engine, next-open fills, fee/slippage, stats, walk-forward |
| `algo/risk.py` | daily loss limit, drawdown kill switch, position cap |
| `algo/broker.py` | PaperBroker and Kraken signed REST (post-only limit orders) |
| `algo/live.py` | the trading loop |

## What this will not do

Turn $100 into $100k. Expect something in the range of buy-and-hold returns with a fraction of
the drawdown in trending years, and small losses in chop. The account scales with capital and
discipline, not with parameters.
