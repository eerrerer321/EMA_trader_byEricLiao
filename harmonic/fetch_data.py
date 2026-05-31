"""下載 Binance ETH/BTC 4h OHLCV 供諧波回測使用，存到 data/。

  python fetch_data.py                       # ETH/USDT
  python fetch_data.py --symbol BTC/USDT
"""
from __future__ import annotations

import argparse
import os

import ccxt
import pandas as pd


def fetch(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    ex = ccxt.binance({"enableRateLimit": True})
    ex.load_markets()
    since, endms, rows = ex.parse8601(start), ex.parse8601(end), []
    while since < endms:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += [r for r in batch if r[0] < endms]
        since = batch[-1][0] + 1
        if len(batch) < 1000:
            break
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")


def main():
    ap = argparse.ArgumentParser(description="下載 OHLCV 供諧波回測")
    ap.add_argument("--symbol", default="ETH/USDT")
    ap.add_argument("--timeframe", default="4h", help="4h / 1h / 30m …")
    ap.add_argument("--start", default="2017-08-01T00:00:00Z")
    ap.add_argument("--end", default="2026-05-20T00:00:00Z")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = fetch(args.symbol, args.timeframe, args.start, args.end)
    out = args.out or os.path.join(
        os.path.dirname(__file__), "data",
        f"{args.symbol.split('/')[0].lower()}_{args.timeframe}.csv"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out)
    print(f"{args.symbol} {args.timeframe}: {len(df)} 根 {df.index.min().date()}~{df.index.max().date()} -> {out}")


if __name__ == "__main__":
    main()
