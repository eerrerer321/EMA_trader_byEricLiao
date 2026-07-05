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
    ensure_symbol_matches_params,
    fetch_ohlcv_from_bybit,
    inject_btc_funding_3d,
    live_circuit_breaker,
    live_funding_boost,
    live_vol_target,
    load_ohlcv_csv,
    print_summary,
    run_backtest,
)
from strategy_core import (
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
    parser.add_argument(
        "--live-overlays",
        action="store_true",
        help="套用實盤風控 overlay（回撤熔斷＋波動度目標倉位）驗證實盤等效配置；"
        "預設關閉以維持與既有快取結果可比。費率過濾/加碼需另搭配 --funding-csv；"
        "不含 CVD 背離降風險（實盤獨有，所有回測器皆無）",
    )
    parser.add_argument(
        "--funding-csv",
        default=None,
        help="BTC 幣本位費率歷史 CSV（timestamp,funding_rate）。提供時注入 btc_funding_3d，"
        "啟用實盤的費率「多頭擁擠」過濾與深負費率加碼",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    ensure_symbol_matches_params(args.symbol)
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
    if args.funding_csv:
        raw_df = inject_btc_funding_3d(raw_df, args.funding_csv)

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
        circuit_breaker=live_circuit_breaker() if args.live_overlays else None,
        vol_target=live_vol_target() if args.live_overlays else None,
        funding_boost=live_funding_boost() if args.funding_csv else None,
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
