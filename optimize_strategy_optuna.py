"""
Optuna optimizer for the ETH 4h strategy.

The objective is intentionally rolling-window based to reduce overfitting:

Quality Score =
    robust_calmar * robust_profit_factor
    * consistency_bonus
    * trade_count_penalty
    * drawdown_penalty

Where:
- robust_calmar blends median and 25th-percentile window Calmar Ratio.
- robust_profit_factor blends median and 25th-percentile Profit Factor.
- consistency_bonus rewards a higher profitable-window rate.
- trade_count_penalty penalizes parameter sets with too few trades.
- drawdown_penalty penalizes very deep worst-window drawdown.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from backtest_eth_strategy_4h import (
    DEFAULT_FEE_RATE,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_SLIPPAGE_RATE,
    fetch_ohlcv_from_bybit,
    load_ohlcv_csv,
    run_backtest,
)
from eth_strategy_4h_autotrading import (
    DEFAULT_QTY_PERCENT,
    STRATEGY_PARAMS,
    SYMBOL,
    TARGET_POSITION_LEVERAGE,
    TIMEFRAME,
    calculate_indicators,
)
from rolling_window_backtest import build_windows


def build_best_payload(
    trial: optuna.trial.FrozenTrial,
    optimization_config: dict[str, Any],
) -> dict[str, Any]:
    if "metrics" in trial.user_attrs:
        metrics = trial.user_attrs["metrics"]
    else:
        metrics = {
            key: value
            for key, value in trial.user_attrs.items()
            if key not in ["strategy_params", "controls"]
        }
    return {
        "quality_score": trial.value,
        "strategy_params": trial.user_attrs["strategy_params"],
        "controls": trial.user_attrs["controls"],
        "metrics": metrics,
        "optimization_config": optimization_config,
        "base_params_reference": STRATEGY_PARAMS,
    }


def build_optuna_storage(storage_url: str | None) -> str | optuna.storages.BaseStorage | None:
    if not storage_url:
        return None
    if storage_url.startswith("sqlite:///"):
        return optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs={"connect_args": {"timeout": 120}},
        )
    if storage_url.startswith("journal://"):
        journal_path = storage_url.removeprefix("journal://")
        return optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(journal_path)
        )
    return storage_url


def suggest_params(
    trial: optuna.Trial,
    search_scope: str = "all",
    force_side_mode: str | None = None,
    force_exit_on_reverse_signal: bool | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    params = dict(STRATEGY_PARAMS)

    optimize_long = search_scope in {"all", "long"}
    optimize_short = search_scope in {"all", "short"}

    if search_scope == "long":
        long_adx = trial.suggest_int("long_adx_threshold", 18, 45)
        params["adx_threshold"] = long_adx
        params["long_adx_threshold"] = long_adx
    elif search_scope == "short":
        short_adx = trial.suggest_int("short_adx_threshold", 18, 45)
        params["adx_threshold"] = short_adx
        params["short_adx_threshold"] = short_adx
    else:
        params["adx_threshold"] = trial.suggest_int("adx_threshold", 18, 45)

    if optimize_long:
        long_activate = trial.suggest_float(
            "long_trailing_activate_profit_percent", 0.004, 0.04, log=True
        )
        long_min_ratio = trial.suggest_float("long_min_profit_to_activate", 0.35, 1.0)
        params.update(
            {
                "long_fixed_stop_loss_percent": trial.suggest_float(
                    "long_fixed_stop_loss_percent", 0.006, 0.045, log=True
                ),
                "long_trailing_activate_profit_percent": long_activate,
                "long_trailing_pullback_percent": trial.suggest_float(
                    "long_trailing_pullback_percent", 0.006, 0.09, log=True
                ),
                "long_trailing_min_profit_percent": long_activate * long_min_ratio,
            }
        )

    if optimize_short:
        short_activate = trial.suggest_float(
            "short_trailing_activate_profit_percent", 0.004, 0.05, log=True
        )
        short_min_ratio = trial.suggest_float("short_min_profit_to_activate", 0.35, 1.0)
        params.update(
            {
                "short_fixed_stop_loss_percent": trial.suggest_float(
                    "short_fixed_stop_loss_percent", 0.006, 0.045, log=True
                ),
                "short_trailing_activate_profit_percent": short_activate,
                "short_trailing_pullback_percent": trial.suggest_float(
                    "short_trailing_pullback_percent", 0.006, 0.09, log=True
                ),
                "short_trailing_min_profit_percent": short_activate * short_min_ratio,
            }
        )

    controls = {
        "side_mode": force_side_mode
        if force_side_mode is not None
        else trial.suggest_categorical(
            "side_mode", ["both", "long_only", "short_only"]
        ),
        "exit_on_reverse_signal": force_exit_on_reverse_signal
        if force_exit_on_reverse_signal is not None
        else trial.suggest_categorical("exit_on_reverse_signal", [True, False]),
    }
    return params, controls


def score_rolling_results(
    results: pd.DataFrame,
    min_median_trades: float,
    max_worst_drawdown_pct: float,
) -> tuple[float, dict[str, float]]:
    returns = results["total_return_pct"].astype(float) / 100
    drawdowns = results["max_drawdown_pct"].astype(float).abs() / 100
    drawdowns = drawdowns.replace(0, np.nan)
    calmars = (returns / drawdowns).replace([np.inf, -np.inf], np.nan).fillna(0)

    profit_factors = (
        results["profit_factor"]
        .replace([np.inf, -np.inf], 5.0)
        .fillna(0.0)
        .astype(float)
        .clip(lower=0.0, upper=5.0)
    )

    median_calmar = float(calmars.median())
    p25_calmar = float(calmars.quantile(0.25))
    robust_calmar = 0.70 * median_calmar + 0.30 * p25_calmar

    median_pf = float(profit_factors.median())
    p25_pf = float(profit_factors.quantile(0.25))
    robust_pf = 0.70 * median_pf + 0.30 * p25_pf

    profitable_rate = float((results["total_return_pct"] > 0).mean())
    consistency_bonus = 0.50 + profitable_rate

    median_trades = float(results["trades"].median())
    trade_count_penalty = min(1.0, median_trades / min_median_trades)

    worst_drawdown_pct = abs(float(results["max_drawdown_pct"].min()))
    excess_dd = max(0.0, worst_drawdown_pct - max_worst_drawdown_pct)
    drawdown_penalty = math.exp(-excess_dd / max(max_worst_drawdown_pct, 1e-9))

    quality_score = (
        robust_calmar
        * robust_pf
        * consistency_bonus
        * trade_count_penalty
        * drawdown_penalty
    )

    metrics = {
        "quality_score": quality_score,
        "robust_calmar": robust_calmar,
        "median_calmar": median_calmar,
        "p25_calmar": p25_calmar,
        "robust_profit_factor": robust_pf,
        "median_profit_factor": median_pf,
        "p25_profit_factor": p25_pf,
        "profitable_window_rate_pct": profitable_rate * 100,
        "median_return_pct": float(results["total_return_pct"].median()),
        "mean_return_pct": float(results["total_return_pct"].mean()),
        "best_return_pct": float(results["total_return_pct"].max()),
        "worst_return_pct": float(results["total_return_pct"].min()),
        "median_max_drawdown_pct": float(results["max_drawdown_pct"].median()),
        "worst_max_drawdown_pct": float(results["max_drawdown_pct"].min()),
        "median_trades": median_trades,
        "total_trades": float(results["trades"].sum()),
    }
    return quality_score, metrics


def run_candidate(
    raw_df: pd.DataFrame,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    params: dict[str, float],
    controls: dict[str, Any],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    allow_long = controls["side_mode"] != "short_only"
    allow_short = controls["side_mode"] != "long_only"

    for idx, (window_start, window_end) in enumerate(windows, start=1):
        summary, _trades, _equity_curve = run_backtest(
            raw_df=raw_df,
            initial_capital=args.initial_capital,
            qty_percent=args.qty_percent,
            target_leverage=args.target_leverage,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            params=params,
            trade_start=window_start,
            trade_end=window_end,
            exit_on_reverse_signal=controls["exit_on_reverse_signal"],
            allow_long=allow_long,
            allow_short=allow_short,
        )
        rows.append(
            {
                "window": idx,
                "start": window_start.date().isoformat(),
                "end": window_end.date().isoformat(),
                **summary,
            }
        )
    return pd.DataFrame(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize strategy parameters with Optuna")
    parser.add_argument("--csv", help="Load OHLCV from CSV instead of Bybit")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
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
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-median-trades", type=float, default=25)
    parser.add_argument("--max-worst-drawdown-pct", type=float, default=35)
    parser.add_argument(
        "--search-scope",
        choices=["all", "long", "short"],
        default="all",
        help="Optimize all params, only long-side params, or only short-side params.",
    )
    parser.add_argument(
        "--force-side-mode",
        choices=["both", "long_only", "short_only"],
        help="Force one side mode instead of letting Optuna choose.",
    )
    parser.add_argument(
        "--force-exit-on-reverse-signal",
        choices=["true", "false"],
        help="Force reverse-signal exits instead of letting Optuna choose.",
    )
    parser.add_argument(
        "--storage",
        help=(
            "Optional Optuna RDB storage, e.g. "
            "sqlite:///optuna_results/study.db. Enables resumable trials."
        ),
    )
    parser.add_argument("--study-name", default="eth_strategy_quality")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing study and run until --n-trials completed trials.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Write best checkpoint and trials snapshot every N completed trials.",
    )
    parser.add_argument("--out-dir", default="optuna_results")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    force_exit_on_reverse_signal = None
    if args.force_exit_on_reverse_signal is not None:
        force_exit_on_reverse_signal = args.force_exit_on_reverse_signal == "true"

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
    raw_df = calculate_indicators(raw_df.copy())

    windows = build_windows(
        trade_start,
        trade_end,
        args.window_months,
        args.step_months,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    storage = build_optuna_storage(args.storage)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=storage,
        study_name=args.study_name if args.storage else None,
        load_if_exists=args.resume,
    )

    optimization_config = {
        "search_scope": args.search_scope,
        "force_side_mode": args.force_side_mode,
        "force_exit_on_reverse_signal": force_exit_on_reverse_signal,
        "storage": args.storage,
        "study_name": args.study_name if args.storage else None,
        "resume": args.resume,
    }

    def write_checkpoint() -> None:
        try:
            best_payload = build_best_payload(study.best_trial, optimization_config)
        except ValueError:
            return
        (out_dir / "best_params_checkpoint.json").write_text(
            json.dumps(best_payload, indent=2), encoding="utf-8"
        )
        study.trials_dataframe(attrs=("number", "value", "params", "user_attrs")).to_csv(
            out_dir / "trials_checkpoint.csv", index=False
        )

    def checkpoint_callback(
        current_study: optuna.Study,
        _trial: optuna.trial.FrozenTrial,
    ) -> None:
        completed = sum(
            trial.state == optuna.trial.TrialState.COMPLETE
            for trial in current_study.trials
        )
        if completed and completed % args.checkpoint_every == 0:
            write_checkpoint()

    def objective(trial: optuna.Trial) -> float:
        params, controls = suggest_params(
            trial,
            search_scope=args.search_scope,
            force_side_mode=args.force_side_mode,
            force_exit_on_reverse_signal=force_exit_on_reverse_signal,
        )
        results = run_candidate(raw_df, windows, params, controls, args)
        score, metrics = score_rolling_results(
            results,
            min_median_trades=args.min_median_trades,
            max_worst_drawdown_pct=args.max_worst_drawdown_pct,
        )
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("strategy_params", params)
        trial.set_user_attr("controls", controls)
        return score

    completed_trials = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    trials_to_run = (
        max(0, args.n_trials - completed_trials) if args.resume else args.n_trials
    )
    if trials_to_run:
        study.optimize(
            objective,
            n_trials=trials_to_run,
            n_jobs=args.n_jobs,
            callbacks=[checkpoint_callback],
        )
    write_checkpoint()

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    trials_df.to_csv(out_dir / "trials.csv", index=False)

    best = study.best_trial
    best_params = best.user_attrs["strategy_params"]
    best_controls = best.user_attrs["controls"]
    best_results = run_candidate(raw_df, windows, best_params, best_controls, args)
    best_results.to_csv(out_dir / "best_rolling_summary.csv", index=False)

    best_payload = build_best_payload(best, optimization_config)
    (out_dir / "best_params.json").write_text(
        json.dumps(best_payload, indent=2), encoding="utf-8"
    )

    print("\n=== Optuna Best Trial ===")
    print(f"Quality Score: {best.value:.6f}")
    print(json.dumps(best_payload, indent=2, ensure_ascii=False))
    print(f"\nSaved results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
