"""
部署閘門：新 Optuna 參數上實盤前的最後一關。

重用 rolling_window_backtest 的滾動視窗驗證，強制套用實盤等效風控
（--live-overlays，即回撤熔斷＋波動度目標倉位），並可選加 --funding-csv
啟用費率過濾/加碼，讓「驗證看到的風險」與「實盤實際承受的風險」一致
（2026-07-02 completion-audit 議會裁定：Optuna 優化階段刻意不套 overlay，
但部署前必須用套了 overlay 的等效設定重新檢驗，否則新參數在實盤的
風險/報酬會系統性偏離驗證結果）。

用法：
    python deploy_gate.py --params-json optuna_results_xxx/best_params.json \\
        --csv rolling_backtest_results_cache/eth_4h.csv --start 2021-10-01 --end 2026-06-10 \\
        --funding-csv rolling_backtest_results_cache/funding_btc_inverse.csv

門檻不過會以非 0 結束碼結束（可接 CI/腳本判斷），並把逐項判定寫入
<out_dir>/gate_verdict.json。門檻皆可用 CLI 覆寫；預設值是「明顯劣化」
的保守下限，不是理想目標（歷史滾動獲利率實測都在 70~85%）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from backtest_eth_strategy_4h import live_circuit_breaker, live_vol_target
from rolling_window_backtest import build_arg_parser as _rolling_arg_parser
from rolling_window_backtest import print_report, run_rolling_backtest


def build_arg_parser():
    parser = _rolling_arg_parser()
    parser.description = "部署閘門：套 rolling_window_backtest 做實盤等效滾動驗證，並依風險門檻判定 PASS/FAIL"
    parser.set_defaults(out_dir="deploy_gate_results")
    parser.add_argument(
        "--min-profitable-window-rate", type=float, default=60.0,
        help="滾動視窗獲利率下限 %%（預設 60，歷史實測皆 70~85%%，此為劣化警戒線非目標）",
    )
    parser.add_argument(
        "--drawdown-floor-pct", type=float, default=-30.0,
        help="最差視窗回撤下限（負值，如 -30 代表不得比 -30%% 更差）。"
        "應設在標的熔斷 halt 級（baseline×2.0）之內，讓閘門先於熔斷示警",
    )
    parser.add_argument(
        "--min-median-return-pct", type=float, default=0.0,
        help="視窗報酬中位數下限 %%（預設 0，僅擋淨虧損參數組）",
    )
    parser.add_argument(
        "--min-total-trades", type=int, default=20,
        help="全視窗交易數下限（擋樣本數過少、靠 1-2 筆幸運單撐出漂亮指標的參數組）",
    )
    parser.add_argument(
        "--min-median-profit-factor", type=float, default=1.0,
        help="視窗獲利因子(PF)中位數下限（預設 1.0=打平；擋高勝率但靠少數大單撐、"
        "獲利邊際薄弱的參數組——這正是「穩定」與「碰運氣」的分野）",
    )
    parser.add_argument(
        "--min-median-sharpe", type=float, default=0.0,
        help="視窗 4h Sharpe 中位數下限（預設 0.0=僅擋負夏普；獲利穩定性的直接量測）",
    )
    parser.add_argument(
        "--require-funding-csv", action="store_true",
        help="未提供 --funding-csv 時直接判定失敗（要求費率過濾/加碼也一併驗證）",
    )
    return parser


def evaluate_gate(results, aggregate: dict, args) -> list[dict]:
    """回傳逐項門檻判定；純函式方便單元測試，不依賴 CLI/檔案 I/O。

    profit_factor/sharpe_4h 取自逐視窗 results（而非 aggregate，rolling_window_backtest
    的 aggregate 目前不含這兩項中位數）——PF/Sharpe 直接量測「獲利穩定」而非只看勝率，
    擋掉勝率達標但靠少數大單撐、邊際獲利薄弱的參數組（DeepSeek 審議建議）。
    """
    checks = [
        ("滾動視窗獲利率(%)", aggregate["profitable_window_rate_pct"], args.min_profitable_window_rate),
        ("最差視窗回撤(%)", aggregate["worst_max_drawdown_pct"], args.drawdown_floor_pct),
        ("視窗報酬中位數(%)", aggregate["median_return_pct"], args.min_median_return_pct),
        ("總交易數", aggregate["total_trades"], args.min_total_trades),
        ("視窗PF中位數", float(results["profit_factor"].replace([float("inf")], float("nan")).median()),
         args.min_median_profit_factor),
        ("視窗Sharpe中位數", float(results["sharpe_4h"].median()), args.min_median_sharpe),
    ]
    return [
        {"check": name, "value": value, "threshold": threshold, "pass": value >= threshold}
        for name, value, threshold in checks
    ]


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.params_json:
        raise SystemExit("部署閘門需要 --params-json 指向欲驗證的 Optuna best_params.json")
    if not args.live_overlays:
        print("ℹ️ 部署閘門強制套用實盤等效風控（回撤熔斷＋波動度目標倉位），已自動開啟 --live-overlays。")
        args.live_overlays = True
    if args.require_funding_csv and not args.funding_csv:
        raise SystemExit("--require-funding-csv 已開啟但未提供 --funding-csv")
    if not args.funding_csv:
        print("⚠️ 未提供 --funding-csv：費率「多頭擁擠」過濾與深負費率加碼不會被驗證，"
              "閘門結果不完全等於實盤風險（見 --require-funding-csv）。")

    # 實際生效的 overlay 設定（而非只記錄旗標本身）：CIRCUIT_BREAKER_ENABLED /
    # VOL_TARGET_ENABLED 若在 strategy_core.py 被關閉，live_*() 回 None，此時即使
    # --live-overlays 是 True，該項風控其實沒真的套上——verdict 必須反映真相，
    # 否則使用者看 JSON 會誤以為驗證涵蓋了實際上沒生效的風控（DeepSeek 審議建議）。
    cb_active = live_circuit_breaker() is not None
    vt_active = live_vol_target() is not None

    results, aggregate = run_rolling_backtest(args)
    print_report(results, aggregate)

    verdict_rows = evaluate_gate(results, aggregate, args)
    passed = all(row["pass"] for row in verdict_rows)

    print("\n=== 部署閘門判定 ===")
    for row in verdict_rows:
        mark = "✅" if row["pass"] else "❌"
        print(f"{mark} {row['check']}: {row['value']:.2f}（門檻 >= {row['threshold']:.2f}）")
    print(f"   （回撤熔斷{'已生效' if cb_active else '未生效(strategy_core 已關閉)'}、"
          f"波動度目標{'已生效' if vt_active else '未生效(strategy_core 已關閉)'}）")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate_verdict.json").write_text(
        json.dumps(
            {"pass": passed, "checks": verdict_rows, "params_json": args.params_json,
             "circuit_breaker_active": cb_active, "vol_target_active": vt_active,
             "funding_csv": args.funding_csv},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if passed:
        print(f"\n✅ 部署閘門通過：{args.params_json} 可上實盤。")
    else:
        print(f"\n❌ 部署閘門未通過：{args.params_json} 暫不建議上實盤，請檢視未達標項目。")
        sys.exit(1)


if __name__ == "__main__":
    main()
