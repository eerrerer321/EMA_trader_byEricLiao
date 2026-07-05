"""
Run rolling-window backtests for the ETH 4h strategy.

Default setup:
- Window length: 6 months
- Step: 2 months
- Warmup: 12 months before the first trading window
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    run_backtest,
)
from strategy_core import (
    DEFAULT_QTY_PERCENT,
    STRATEGY_PARAMS,
    SYMBOL,
    TARGET_POSITION_LEVERAGE,
    TIMEFRAME,
)


def build_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    window_months: int,
    step_months: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor + pd.DateOffset(months=window_months) <= end:
        window_end = cursor + pd.DateOffset(months=window_months)
        windows.append((cursor, window_end))
        cursor = cursor + pd.DateOffset(months=step_months)
    return windows


def run_rolling_backtest(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_symbol_matches_params(args.symbol)  # 放函式入口：deploy_gate 走此路徑一併受防護
    trade_start = pd.Timestamp(args.start)
    trade_end = pd.Timestamp(args.end)
    data_start = trade_start - pd.DateOffset(months=args.warmup_months)

    if args.csv:
        raw_df = load_ohlcv_csv(args.csv)
    else:
        raw_df = fetch_ohlcv_from_bybit(
            args.symbol,
            args.timeframe,
            data_start.strftime("%Y-%m-%d"),
            trade_end.strftime("%Y-%m-%d"),
        )
    if args.funding_csv:
        raw_df = inject_btc_funding_3d(raw_df, args.funding_csv)

    windows = build_windows(
        trade_start,
        trade_end,
        args.window_months,
        args.step_months,
    )
    if not windows:
        raise ValueError("No rolling windows generated; check start/end/window settings")

    params = STRATEGY_PARAMS.copy()
    if args.adx_threshold is not None:
        params["adx_threshold"] = args.adx_threshold
    if args.params_json:
        payload = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
        params.update(payload.get("strategy_params", payload))
        if args.side_mode == "best":
            args.side_mode = payload.get("controls", {}).get("side_mode", "both")
        if args.exit_on_reverse_signal == "best":
            args.exit_on_reverse_signal = str(
                payload.get("controls", {}).get("exit_on_reverse_signal", True)
            ).lower()

    out_dir = Path(args.out_dir)
    detail_dir = out_dir / "window_details"
    detail_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for idx, (window_start, window_end) in enumerate(windows, start=1):
        summary, trades, equity_curve = run_backtest(
            raw_df=raw_df,
            initial_capital=args.initial_capital,
            qty_percent=args.qty_percent,
            target_leverage=args.target_leverage,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            params=params,
            trade_start=window_start,
            trade_end=window_end,
            exit_on_reverse_signal=args.exit_on_reverse_signal != "false",
            allow_long=args.side_mode != "short_only",
            allow_short=args.side_mode != "long_only",
            circuit_breaker=live_circuit_breaker() if args.live_overlays else None,
            vol_target=live_vol_target() if args.live_overlays else None,
            funding_boost=live_funding_boost() if args.funding_csv else None,
        )

        window_name = f"{idx:02d}_{window_start.date()}_{window_end.date()}"
        trades.to_csv(detail_dir / f"{window_name}_trades.csv", index=False)
        equity_curve.to_csv(detail_dir / f"{window_name}_equity.csv", index=False)

        rows.append(
            {
                "window": idx,
                "start": window_start.date().isoformat(),
                "end": window_end.date().isoformat(),
                **summary,
            }
        )

    results = pd.DataFrame(rows)
    aggregate = {
        "windows": int(len(results)),
        "profitable_windows": int((results["total_return_pct"] > 0).sum()),
        "profitable_window_rate_pct": float(
            (results["total_return_pct"] > 0).mean() * 100
        ),
        "median_return_pct": float(results["total_return_pct"].median()),
        "mean_return_pct": float(results["total_return_pct"].mean()),
        "worst_return_pct": float(results["total_return_pct"].min()),
        "best_return_pct": float(results["total_return_pct"].max()),
        "median_max_drawdown_pct": float(results["max_drawdown_pct"].median()),
        "worst_max_drawdown_pct": float(results["max_drawdown_pct"].min()),
        "median_trades": float(results["trades"].median()),
        "total_trades": int(results["trades"].sum()),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "rolling_summary.csv", index=False)
    (out_dir / "rolling_aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    if args.save_data:
        raw_df.to_csv(out_dir / "ohlcv_with_warmup.csv")
    return results, aggregate


def print_report(results: pd.DataFrame, aggregate: dict[str, Any]) -> None:
    print("\n=== Rolling Window Aggregate ===")
    print(f"Windows                 : {aggregate['windows']}")
    print(
        f"Profitable windows      : {aggregate['profitable_windows']} "
        f"({aggregate['profitable_window_rate_pct']:.2f}%)"
    )
    print(f"Median return           : {aggregate['median_return_pct']:.2f}%")
    print(f"Mean return             : {aggregate['mean_return_pct']:.2f}%")
    print(f"Best / worst return     : {aggregate['best_return_pct']:.2f}% / {aggregate['worst_return_pct']:.2f}%")
    print(f"Median max drawdown     : {aggregate['median_max_drawdown_pct']:.2f}%")
    print(f"Worst max drawdown      : {aggregate['worst_max_drawdown_pct']:.2f}%")
    print(f"Total trades            : {aggregate['total_trades']}")

    cols = [
        "window",
        "start",
        "end",
        "total_return_pct",
        "max_drawdown_pct",
        "trades",
        "win_rate_pct",
        "profit_factor",
    ]
    print("\n=== Rolling Window Summary ===")
    print(results[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== Worst 5 Windows ===")
    worst = results.nsmallest(5, "total_return_pct")
    print(worst[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run rolling-window backtests")
    parser.add_argument("--csv", help="Load OHLCV from CSV instead of Bybit")
    parser.add_argument("--start", required=True, help="Trade-window start date")
    parser.add_argument("--end", required=True, help="Trade-window end date")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--timeframe", default=TIMEFRAME)
    parser.add_argument("--window-months", type=int, default=6)
    parser.add_argument("--step-months", type=int, default=2)
    parser.add_argument("--warmup-months", type=int, default=12)
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--qty-percent", type=float, default=DEFAULT_QTY_PERCENT)
    parser.add_argument("--target-leverage", type=float, default=TARGET_POSITION_LEVERAGE)
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE)
    parser.add_argument("--out-dir", default="rolling_backtest_results")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--no-reverse-exit", action="store_true")
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--short-only", action="store_true")
    parser.add_argument("--side-mode", choices=["best", "both", "long_only", "short_only"], default="both")
    parser.add_argument("--exit-on-reverse-signal", choices=["best", "true", "false"], default="true")
    parser.add_argument("--adx-threshold", type=float)
    parser.add_argument("--params-json", help="Load strategy_params and optional controls from Optuna best_params.json")
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
    if args.long_only and args.short_only:
        raise ValueError("--long-only and --short-only cannot both be enabled")
    if args.long_only:
        args.side_mode = "long_only"
    if args.short_only:
        args.side_mode = "short_only"
    if args.no_reverse_exit:
        args.exit_on_reverse_signal = "false"
    results, aggregate = run_rolling_backtest(args)
    print_report(results, aggregate)
    print(f"\nSaved results to: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
