"""
Backtest tool for eth_strategy_4h_autotrading.py.

Assumptions:
- Signals are evaluated on a completed 4h candle.
- Entries/exits from signals are filled at the next candle open.
- Fixed stops are simulated intrabar with OHLC high/low.
- Trailing stops activate/update conservatively and are not allowed to both
  activate and exit on the same 4h candle.
- PnL is futures-style mark-to-market; margin reservation is not modeled.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ccxt
import pandas as pd

from eth_strategy_4h_autotrading import (
    ALLOW_SAME_BAR_REVERSAL,
    DEFAULT_QTY_PERCENT,
    STRATEGY_PARAMS,
    SYMBOL,
    TARGET_POSITION_LEVERAGE,
    TIMEFRAME,
    calculate_indicators,
)


DEFAULT_INITIAL_CAPITAL = 1000.0
DEFAULT_FEE_RATE = 0.0006
DEFAULT_SLIPPAGE_RATE = 0.0002


@dataclass
class Position:
    side: str
    qty: float
    entry_price: float
    entry_time: pd.Timestamp
    entry_fee: float
    peak: float | None = None
    trough: float | None = None
    trail_active: bool = False
    trail_stop: float | None = None
    bars_held: int = 0


def parse_timestamp(value: str | None) -> int | None:
    if not value:
        return None
    return int(pd.Timestamp(value, tz="UTC").timestamp() * 1000)


def fetch_ohlcv_from_bybit(
    symbol: str,
    timeframe: str,
    start: str,
    end: str | None,
    limit: int = 1000,
) -> pd.DataFrame:
    exchange = ccxt.bybit(
        {
            "enableRateLimit": True,
            "options": {
                "defaultType": "linear",
                "adjustForTimeDifference": True,
            },
        }
    )
    exchange.load_markets()

    since = parse_timestamp(start)
    until = parse_timestamp(end)
    if since is None:
        raise ValueError("--start is required when downloading data")

    rows: list[list[float]] = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not batch:
            break

        for row in batch:
            ts = int(row[0])
            if until is not None and ts >= until:
                break
            rows.append(row)

        last_ts = int(batch[-1][0])
        next_since = last_ts + 1
        if next_since <= since:
            break
        since = next_since
        if until is not None and since >= until:
            break
        # Bybit can return 999 candles even when more data is available, so only
        # use this short-batch break when no explicit end date was requested.
        if until is None and len(batch) < limit:
            break

    if not rows:
        raise RuntimeError("No OHLCV data fetched")

    df = pd.DataFrame(
        rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(
        None
    )
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return df.set_index("timestamp")


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    return df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]


def long_signal(bar: pd.Series, params: dict[str, float]) -> bool:
    adx_threshold = params.get("long_adx_threshold", params["adx_threshold"])
    return bool(
        bar["close"] > bar["ema90"]
        and bar["low"] > bar["ema90"]
        and bar["close"] > bar["ema200"]
        and bar["adx"] > adx_threshold
        and bar["rsi"] <= 70
    )


def short_signal(bar: pd.Series, params: dict[str, float]) -> bool:
    adx_threshold = params.get("short_adx_threshold", params["adx_threshold"])
    return bool(
        bar["close"] < bar["ema90"]
        and bar["high"] < bar["ema90"]
        and bar["close"] < bar["ema200"]
        and bar["adx"] > adx_threshold
        and bar["rsi"] >= 30
    )


def fill_entry_price(side: str, raw_price: float, slippage_rate: float) -> float:
    if side == "long":
        return raw_price * (1 + slippage_rate)
    return raw_price * (1 - slippage_rate)


def fill_exit_price(side: str, raw_price: float, slippage_rate: float) -> float:
    if side == "long":
        return raw_price * (1 - slippage_rate)
    return raw_price * (1 + slippage_rate)


def stop_fill_price(side: str, stop_price: float, bar_open: float) -> float:
    if side == "long" and bar_open < stop_price:
        return bar_open
    if side == "short" and bar_open > stop_price:
        return bar_open
    return stop_price


def calc_unrealized(position: Position | None, close_price: float) -> float:
    if position is None:
        return 0.0
    if position.side == "long":
        return (close_price - position.entry_price) * position.qty
    return (position.entry_price - close_price) * position.qty


def close_position(
    position: Position,
    exit_time: pd.Timestamp,
    raw_exit_price: float,
    reason: str,
    fee_rate: float,
    slippage_rate: float,
) -> dict[str, Any]:
    exit_price = fill_exit_price(position.side, raw_exit_price, slippage_rate)
    exit_fee = position.qty * exit_price * fee_rate

    if position.side == "long":
        gross_pnl = (exit_price - position.entry_price) * position.qty
    else:
        gross_pnl = (position.entry_price - exit_price) * position.qty

    net_pnl = gross_pnl - position.entry_fee - exit_fee
    notional = position.qty * position.entry_price
    return_pct = net_pnl / notional if notional else 0.0

    return {
        "side": position.side,
        "entry_time": position.entry_time,
        "exit_time": exit_time,
        "entry_price": position.entry_price,
        "exit_price": exit_price,
        "qty": position.qty,
        "entry_fee": position.entry_fee,
        "exit_fee": exit_fee,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "return_pct": return_pct,
        "exit_reason": reason,
        "bars_held": position.bars_held,
    }


def circuit_breaker_scale(drawdown: float, cb: dict[str, float] | None) -> float:
    """Entry size scale for the drawdown circuit breaker (same semantics as the
    live engine's _drawdown_state).

    cb=None disables it (always 1.0). ``drawdown`` is the equity drop from its
    running high-water mark. Rationale: Grossman-Zhou (1993) / CPPI tiered de-
    risking, with drawdown-multiple thresholds (~1.5x warn, ~2.0x halt) used as
    strategy-decay tripwires in walk-forward practice (Pardo 2008, Kestner 2003).
    """
    if not cb:
        return 1.0
    baseline = cb.get("baseline_drawdown", 0.18)
    warn = baseline * cb.get("warn_mult", 1.5)
    halt = baseline * cb.get("halt_mult", 2.0)
    if drawdown >= halt:
        return 0.0
    if drawdown >= warn:
        return cb.get("warn_size_scale", 0.5)
    return 1.0


def open_position(
    side: str,
    entry_time: pd.Timestamp,
    raw_entry_price: float,
    equity: float,
    qty_percent: float,
    target_leverage: float,
    fee_rate: float,
    slippage_rate: float,
) -> Position:
    entry_price = fill_entry_price(side, raw_entry_price, slippage_rate)
    target_notional = equity * qty_percent / 100 * target_leverage
    qty = target_notional / entry_price
    entry_fee = qty * entry_price * fee_rate
    if side == "long":
        return Position(
            side=side,
            qty=qty,
            entry_price=entry_price,
            entry_time=entry_time,
            entry_fee=entry_fee,
            peak=entry_price,
        )
    return Position(
        side=side,
        qty=qty,
        entry_price=entry_price,
        entry_time=entry_time,
        entry_fee=entry_fee,
        trough=entry_price,
    )


def fixed_stop_price(position: Position, params: dict[str, float]) -> float:
    if position.side == "long":
        return position.entry_price * (1 - params["long_fixed_stop_loss_percent"])
    return position.entry_price * (1 + params["short_fixed_stop_loss_percent"])


def update_trailing_stop(position: Position, bar: pd.Series, params: dict[str, float]) -> None:
    if position.side == "long":
        position.peak = max(position.peak or position.entry_price, float(bar["high"]))
        activate_price = position.entry_price * (
            1 + params["long_trailing_activate_profit_percent"]
        )
        if not position.trail_active and float(bar["high"]) >= activate_price:
            position.trail_active = True
            position.trail_stop = position.entry_price * (
                1 + params["long_trailing_min_profit_percent"]
            )
        if position.trail_active:
            new_stop = position.peak * (1 - params["long_trailing_pullback_percent"])
            min_profit = position.entry_price * (
                1 + params["long_trailing_min_profit_percent"]
            )
            position.trail_stop = max(position.trail_stop or 0, new_stop, min_profit)
        return

    position.trough = min(position.trough or position.entry_price, float(bar["low"]))
    activate_price = position.entry_price * (
        1 - params["short_trailing_activate_profit_percent"]
    )
    if not position.trail_active and float(bar["low"]) <= activate_price:
        position.trail_active = True
        position.trail_stop = position.entry_price * (
            1 - params["short_trailing_min_profit_percent"]
        )
    if position.trail_active:
        new_stop = position.trough * (1 + params["short_trailing_pullback_percent"])
        min_profit = position.entry_price * (
            1 - params["short_trailing_min_profit_percent"]
        )
        position.trail_stop = min(position.trail_stop or math.inf, new_stop, min_profit)


def check_stop_exit(position: Position, bar: pd.Series, params: dict[str, float]) -> tuple[str, float] | None:
    bar_open = float(bar["open"])
    bar_high = float(bar["high"])
    bar_low = float(bar["low"])

    fixed_stop = fixed_stop_price(position, params)
    if position.side == "long":
        if bar_low <= fixed_stop:
            return "fixed_stop", stop_fill_price("long", fixed_stop, bar_open)
        if position.trail_active and position.trail_stop and bar_low <= position.trail_stop:
            return "trailing_stop", stop_fill_price("long", position.trail_stop, bar_open)
        return None

    if bar_high >= fixed_stop:
        return "fixed_stop", stop_fill_price("short", fixed_stop, bar_open)
    if position.trail_active and position.trail_stop and bar_high >= position.trail_stop:
        return "trailing_stop", stop_fill_price("short", position.trail_stop, bar_open)
    return None


def max_drawdown(equity_curve: pd.DataFrame) -> float:
    peaks = equity_curve["equity"].cummax()
    drawdowns = equity_curve["equity"] / peaks - 1
    return float(drawdowns.min()) if len(drawdowns) else 0.0


def summarize(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    initial_capital: float,
) -> dict[str, Any]:
    final_equity = float(equity_curve["equity"].iloc[-1])
    total_return = final_equity / initial_capital - 1

    days = max(
        (equity_curve["timestamp"].iloc[-1] - equity_curve["timestamp"].iloc[0]).days,
        1,
    )
    # A non-positive final equity (account wiped out) makes the fractional power
    # undefined; report it as a total loss instead of crashing with a complex/NaN.
    if final_equity > 0 and initial_capital > 0:
        cagr = (final_equity / initial_capital) ** (365 / days) - 1
    else:
        cagr = -1.0
    mdd = max_drawdown(equity_curve)

    if trades.empty:
        return {
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "total_return_pct": total_return * 100,
            "cagr_pct": cagr * 100,
            "max_drawdown_pct": mdd * 100,
            "trades": 0,
        }

    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] <= 0]
    gross_profit = float(wins["net_pnl"].sum())
    gross_loss = abs(float(losses["net_pnl"].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss else math.inf

    returns = equity_curve["equity"].pct_change().dropna()
    sharpe = 0.0
    if len(returns) > 1 and returns.std() != 0:
        sharpe = float(returns.mean() / returns.std() * math.sqrt(365 * 24 / 4))

    return {
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "max_drawdown_pct": mdd * 100,
        "trades": int(len(trades)),
        "win_rate_pct": len(wins) / len(trades) * 100,
        "profit_factor": profit_factor,
        "avg_trade_pnl": float(trades["net_pnl"].mean()),
        "best_trade": float(trades["net_pnl"].max()),
        "worst_trade": float(trades["net_pnl"].min()),
        "sharpe_4h": sharpe,
    }


def run_backtest(
    raw_df: pd.DataFrame,
    initial_capital: float,
    qty_percent: float,
    target_leverage: float,
    fee_rate: float,
    slippage_rate: float,
    params: dict[str, float],
    trade_start: str | pd.Timestamp | None = None,
    trade_end: str | pd.Timestamp | None = None,
    exit_on_reverse_signal: bool = True,
    allow_long: bool = True,
    allow_short: bool = True,
    allow_same_bar_reversal: bool = ALLOW_SAME_BAR_REVERSAL,
    circuit_breaker: dict[str, float] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    required_indicator_columns = {"ema90", "ema200", "adx", "rsi"}
    if required_indicator_columns.issubset(raw_df.columns):
        df = raw_df
    else:
        df = calculate_indicators(raw_df.copy())
    if len(df) < 3:
        raise ValueError("Not enough indicator-ready candles for backtest")

    start_i = 1
    end_i = len(df)
    if trade_start is not None:
        start_ts = pd.Timestamp(trade_start)
        start_i = max(1, int(df.index.searchsorted(start_ts, side="left")))
    if trade_end is not None:
        end_ts = pd.Timestamp(trade_end)
        end_i = int(df.index.searchsorted(end_ts, side="left"))
    if end_i - start_i < 2:
        raise ValueError("Not enough candles inside requested trade window")

    equity = float(initial_capital)
    equity_peak = float(initial_capital)  # high-water mark for circuit-breaker drawdown
    position: Position | None = None
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    for i in range(start_i, end_i):
        signal_bar = df.iloc[i - 1]
        bar = df.iloc[i]
        timestamp = df.index[i]

        signal_long = long_signal(signal_bar, params)
        signal_short = short_signal(signal_bar, params)

        reversed_this_bar = False
        if position is not None and exit_on_reverse_signal:
            should_reverse = (
                position.side == "long"
                and signal_short
                or position.side == "short"
                and signal_long
            )
            if should_reverse:
                trade = close_position(
                    position,
                    timestamp,
                    float(bar["open"]),
                    "reverse_signal",
                    fee_rate,
                    slippage_rate,
                )
                equity += trade["net_pnl"]
                trades.append(trade)
                position = None
                reversed_this_bar = True

        # The live engine closes on a reverse signal and waits for the next bar
        # before opening the opposite side (ALLOW_SAME_BAR_REVERSAL=False), so by
        # default the backtest must not flip on the same bar either.
        if position is None and (allow_same_bar_reversal or not reversed_this_bar):
            # Drawdown circuit breaker: scale (or halt) new-entry size by the
            # equity drop from its high-water mark. Flat here, so equity == mark
            # equity and the drawdown is measured correctly.
            cb_drawdown = (
                (equity_peak - equity) / equity_peak if equity_peak > 0 else 0.0
            )
            entry_qty_percent = qty_percent * circuit_breaker_scale(
                cb_drawdown, circuit_breaker
            )
            if entry_qty_percent > 0 and signal_long and allow_long:
                position = open_position(
                    "long",
                    timestamp,
                    float(bar["open"]),
                    equity,
                    entry_qty_percent,
                    target_leverage,
                    fee_rate,
                    slippage_rate,
                )
            elif entry_qty_percent > 0 and signal_short and allow_short:
                position = open_position(
                    "short",
                    timestamp,
                    float(bar["open"]),
                    equity,
                    entry_qty_percent,
                    target_leverage,
                    fee_rate,
                    slippage_rate,
                )

        if position is not None:
            stop_exit = check_stop_exit(position, bar, params)
            if stop_exit is not None:
                reason, stop_price = stop_exit
                trade = close_position(
                    position,
                    timestamp,
                    stop_price,
                    reason,
                    fee_rate,
                    slippage_rate,
                )
                equity += trade["net_pnl"]
                trades.append(trade)
                position = None
            else:
                update_trailing_stop(position, bar, params)
                position.bars_held += 1

        mark_equity = equity + calc_unrealized(position, float(bar["close"]))
        equity_peak = max(equity_peak, mark_equity)
        equity_rows.append(
            {
                "timestamp": timestamp,
                "equity": mark_equity,
                "cash_equity": equity,
                "position_side": position.side if position else "flat",
                "position_qty": position.qty if position else 0.0,
            }
        )

    if position is not None:
        last_ts = df.index[end_i - 1]
        last_close = float(df.iloc[end_i - 1]["close"])
        trade = close_position(
            position,
            last_ts,
            last_close,
            "end_of_backtest",
            fee_rate,
            slippage_rate,
        )
        equity += trade["net_pnl"]
        trades.append(trade)
        equity_rows.append(
            {
                "timestamp": last_ts,
                "equity": equity,
                "cash_equity": equity,
                "position_side": "flat",
                "position_qty": 0.0,
            }
        )

    equity_curve = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trades)
    summary = summarize(trades_df, equity_curve, initial_capital)
    return summary, trades_df, equity_curve


def print_summary(summary: dict[str, Any]) -> None:
    print("\n=== Backtest Summary ===")
    print(f"Initial capital : {summary['initial_capital']:.2f} USDT")
    print(f"Final equity    : {summary['final_equity']:.2f} USDT")
    print(f"Total return    : {summary['total_return_pct']:.2f}%")
    print(f"CAGR            : {summary['cagr_pct']:.2f}%")
    print(f"Max drawdown    : {summary['max_drawdown_pct']:.2f}%")
    print(f"Trades          : {summary.get('trades', 0)}")
    if summary.get("trades", 0):
        print(f"Win rate        : {summary['win_rate_pct']:.2f}%")
        print(f"Profit factor   : {summary['profit_factor']:.2f}")
        print(f"Avg trade PnL   : {summary['avg_trade_pnl']:.2f} USDT")
        print(f"Best trade      : {summary['best_trade']:.2f} USDT")
        print(f"Worst trade     : {summary['worst_trade']:.2f} USDT")
        print(f"4h Sharpe       : {summary['sharpe_4h']:.2f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest the ETH 4h strategy")
    parser.add_argument("--csv", help="Load OHLCV from CSV instead of Bybit")
    parser.add_argument("--start", default="2020-01-01", help="UTC start date")
    parser.add_argument("--end", default=None, help="UTC end date")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--timeframe", default=TIMEFRAME)
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--qty-percent", type=float, default=DEFAULT_QTY_PERCENT)
    parser.add_argument("--target-leverage", type=float, default=TARGET_POSITION_LEVERAGE)
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE)
    parser.add_argument("--out-dir", default="backtest_results")
    parser.add_argument("--save-data", action="store_true", help="Save downloaded OHLCV")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.csv:
        raw_df = load_ohlcv_csv(args.csv)
    else:
        raw_df = fetch_ohlcv_from_bybit(
            args.symbol,
            args.timeframe,
            args.start,
            args.end,
        )

    summary, trades, equity_curve = run_backtest(
        raw_df=raw_df,
        initial_capital=args.initial_capital,
        qty_percent=args.qty_percent,
        target_leverage=args.target_leverage,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        params=STRATEGY_PARAMS.copy(),
    )

    print_summary(summary)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_dir / "trades.csv", index=False)
    equity_curve.to_csv(out_dir / "equity_curve.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    if args.save_data:
        raw_df.to_csv(out_dir / "ohlcv.csv")

    print(f"\nSaved results to: {out_dir.resolve()}")


if __name__ == "__main__":
    # Keep stdout readable on Windows terminals.
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
