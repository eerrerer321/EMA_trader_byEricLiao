"""
Run a single backtest using an Optuna best_params.json file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest_eth_strategy_4h import (
    DEFAULT_FEE_RATE,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_SLIPPAGE_RATE,
    fetch_ohlcv_from_bybit,
    load_ohlcv_csv,
    print_summary,
    run_backtest,
)
from eth_strategy_4h_autotrading import (
    DEFAULT_QTY_PERCENT,
    SYMBOL,
    TARGET_POSITION_LEVERAGE,
    TIMEFRAME,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest Optuna best parameters")
    parser.add_argument("--best-params", required=True)
    parser.add_argument("--csv", help="Load OHLCV from CSV instead of Bybit")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--timeframe", default=TIMEFRAME)
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--qty-percent", type=float, default=DEFAULT_QTY_PERCENT)
    parser.add_argument("--target-leverage", type=float, default=TARGET_POSITION_LEVERAGE)
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE)
    parser.add_argument("--warmup-months", type=int, default=12)
    parser.add_argument("--out-dir", default="best_params_single_backtest")
    parser.add_argument("--side-mode", choices=["best", "both", "long_only", "short_only"], default="best")
    parser.add_argument("--exit-on-reverse-signal", choices=["best", "true", "false"], default="best")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = json.loads(Path(args.best_params).read_text(encoding="utf-8"))
    params = payload["strategy_params"]
    controls = payload.get("controls", {})
    side_mode = (
        controls.get("side_mode", "both")
        if args.side_mode == "best"
        else args.side_mode
    )
    exit_on_reverse_signal = controls.get("exit_on_reverse_signal", True)
    if args.exit_on_reverse_signal != "best":
        exit_on_reverse_signal = args.exit_on_reverse_signal == "true"

    if args.csv:
        raw_df = load_ohlcv_csv(args.csv)
    else:
        data_start = pd.Timestamp(args.start) - pd.DateOffset(months=args.warmup_months)
        raw_df = fetch_ohlcv_from_bybit(
            args.symbol,
            args.timeframe,
            data_start.strftime("%Y-%m-%d"),
            args.end,
        )

    summary, trades, equity_curve = run_backtest(
        raw_df=raw_df,
        initial_capital=args.initial_capital,
        qty_percent=args.qty_percent,
        target_leverage=args.target_leverage,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        params=params,
        trade_start=args.start,
        trade_end=args.end,
        exit_on_reverse_signal=exit_on_reverse_signal,
        allow_long=side_mode != "short_only",
        allow_short=side_mode != "long_only",
    )

    print_summary(summary)
    print("\n=== Best Params Controls ===")
    print(
        json.dumps(
            {
                "side_mode": side_mode,
                "exit_on_reverse_signal": exit_on_reverse_signal,
                "source_controls": controls,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_dir / "trades.csv", index=False)
    equity_curve.to_csv(out_dir / "equity_curve.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nSaved results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
