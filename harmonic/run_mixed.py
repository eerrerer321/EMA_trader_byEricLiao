"""
混合策略回測：EMA 趨勢 + 諧波（順大勢）共用同一筆資金
=====================================================

把主專案的 EMA 趨勢策略（趨勢跟隨）與本資料夾的諧波策略（順大勢回調反轉）
放在**同一筆資金**上分時複用：

  - 趨勢訊號優先；趨勢空倉時（佔 ~85% 時間），諧波用閒置資金進場。
  - 兩者互斥持倉（一次一個倉位），共享同一個權益池、一起複利。

為什麼有效（實測）：諧波加上「順大勢」過濾後變成正期望強策略（PF≈1.70、
MDD 僅 −12%），且與趨勢**弱負相關**。兩者疊加 → 報酬大增、回撤不增，
Sharpe/Calmar 同步上升（真正的 1+1>2）。

用法:
  python run_mixed.py --csv data/eth_4h.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harmonic_strategy import detect_opportunities, backtest as harmonic_backtest, FEE, SLIP, QTY_PCT, LEV  # noqa: E402
from strategy_core import STRATEGY_PARAMS, calculate_indicators  # noqa: E402
from backtest_eth_strategy_4h import (  # noqa: E402
    run_backtest, load_ohlcv_csv, long_signal, short_signal,
    check_stop_exit, update_trailing_stop, open_position as bt_open,
    close_position as bt_close, calc_unrealized,
    vol_target_scale, circuit_breaker_scale, live_vol_target, live_circuit_breaker,
)


def mixed_backtest(df, opps, initial=1000.0, params=None,
                   circuit_breaker=None, vol_target=None, vol_target_harm=None):
    """趨勢優先、諧波填補空倉的單一資金池回測。回傳 equity DataFrame。

    circuit_breaker / vol_target 語意與主回測 run_backtest 相同（None=停用）：
    - circuit_breaker：以混合權益池的 high-water mark 回撤分級縮減/暫停新開倉，
      兩條腿共用（同一筆資金的風控本來就該看同一個權益池）。
    - vol_target：趨勢腿的波動度目標倉位係數（用 signal bar 的 ret_vol）。
    - vol_target_harm：諧波腿的波動度目標（與趨勢分開控制，方便對照實驗）。
    """
    params = params or dict(STRATEGY_PARAMS)
    opp_by_j = {}
    for op in opps:
        opp_by_j.setdefault(op["j"], op)
    n = len(df)
    equity = initial
    equity_peak = float(initial)  # 混合權益池 high-water mark（熔斷用）
    pos = ptype = hp = None
    eq_rows = []
    closes = df["close"].values
    for i in range(1, n):
        sb = df.iloc[i - 1]; bar = df.iloc[i]; ts = df.index[i]
        sl_long, sl_short = long_signal(sb, params), short_signal(sb, params)
        reversed_bar = False
        # 1) 出場：趨勢反向 / 諧波單根 SL-TP（順序對齊 run_backtest）
        if ptype == "trend" and pos.side == "long" and sl_short:
            tr = bt_close(pos, ts, float(bar["open"]), "reverse", FEE, SLIP)
            equity += tr["net_pnl"]; pos = ptype = None; reversed_bar = True
        elif ptype == "harm":
            hi, lo, bull = float(bar["high"]), float(bar["low"]), hp["bull"]
            ex = None
            if bull:
                ex = hp["sl"] if lo <= hp["sl"] else (hp["tp"] if hi >= hp["tp"] else None)
            else:
                ex = hp["sl"] if hi >= hp["sl"] else (hp["tp"] if lo <= hp["tp"] else None)
            if ex is not None:
                fill = ex * (1 - SLIP) if bull else ex * (1 + SLIP)
                gross = (fill - hp["entry"]) * hp["qty"] if bull else (hp["entry"] - fill) * hp["qty"]
                equity += gross - hp["qty"] * fill * FEE
                hp = pos = ptype = None
        # 2) 進場（空倉時）：趨勢優先，否則諧波（倉位 = 基準 × 熔斷係數 × 波動度係數）
        if pos is None and not reversed_bar:
            cb_dd = (equity_peak - equity) / equity_peak if equity_peak > 0 else 0.0
            cb_s = circuit_breaker_scale(cb_dd, circuit_breaker)
            if sl_long:
                scale = cb_s * vol_target_scale(
                    sb.get("ret_vol") if vol_target else None, vol_target)
                if scale > 0:
                    pos = bt_open("long", ts, float(bar["open"]), equity,
                                  QTY_PCT * scale, LEV, FEE, SLIP); ptype = "trend"
            elif i in opp_by_j:
                scale = cb_s * vol_target_scale(
                    sb.get("ret_vol") if vol_target_harm else None, vol_target_harm)
                if scale > 0:
                    op = opp_by_j[i]; entry = op["entry"]
                    qty = equity * QTY_PCT * scale / 100 * LEV / entry
                    equity -= qty * entry * FEE
                    hp = {"entry": entry, "sl": op["sl"], "tp": op["tp"], "bull": op["bull"], "qty": qty}
                    ptype = "harm"; pos = "H"
                    # 諧波進場那根同樣判 SL（對稱於下方趨勢腿「含進場那根」；
                    # 成交必先於 SL 掃穿，見 harmonic_strategy.simulate_trade 說明）
                    if (hp["bull"] and float(bar["low"]) <= hp["sl"]) or \
                            ((not hp["bull"]) and float(bar["high"]) >= hp["sl"]):
                        fill = hp["sl"] * (1 - SLIP) if hp["bull"] else hp["sl"] * (1 + SLIP)
                        gross = (fill - hp["entry"]) * hp["qty"] if hp["bull"] else (hp["entry"] - fill) * hp["qty"]
                        equity += gross - hp["qty"] * fill * FEE
                        hp = pos = ptype = None
        # 3) 趨勢同根停損檢查（含進場那根）
        if ptype == "trend":
            st = check_stop_exit(pos, bar, params)
            if st:
                tr = bt_close(pos, ts, st[1], st[0], FEE, SLIP); equity += tr["net_pnl"]; pos = ptype = None
            else:
                update_trailing_stop(pos, bar, params); pos.bars_held += 1
        # 標記權益（含未實現）
        if ptype == "trend":
            mk = equity + calc_unrealized(pos, float(closes[i]))
        elif ptype == "harm":
            mk = equity + ((closes[i] - hp["entry"]) if hp["bull"] else (hp["entry"] - closes[i])) * hp["qty"]
        else:
            mk = equity
        equity_peak = max(equity_peak, mk)
        eq_rows.append({"timestamp": ts, "equity": mk})
    return pd.DataFrame(eq_rows)


def stats(eq, initial, label):
    ret = (eq.iloc[-1] / initial - 1) * 100
    mdd = (eq / eq.cummax() - 1).min() * 100
    days = max((eq.index[-1] - eq.index[0]).days, 1)
    cagr = ((eq.iloc[-1] / initial) ** (365 / days) - 1) * 100
    mr = eq.resample("ME").last().pct_change().dropna()
    sh = mr.mean() / mr.std() * np.sqrt(12) if mr.std() > 0 else 0.0
    print(f"  {label:16s} 報酬 {ret:7.0f}%  CAGR {cagr:5.1f}%  MDD {mdd:6.1f}%  "
          f"Sharpe {sh:5.2f}  Calmar {cagr/abs(mdd) if mdd else 0:.2f}")
    return mr


def main():
    ap = argparse.ArgumentParser(description="趨勢 + 諧波 混合回測（同一筆資金）")
    ap.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "data", "eth_4h.csv"))
    ap.add_argument("--start", default="2017-10-01")
    ap.add_argument("--end", default="2026-05-19")
    args = ap.parse_args()

    df = calculate_indicators(load_ohlcv_csv(args.csv).copy())
    df = df[(df.index >= pd.Timestamp(args.start)) & (df.index <= pd.Timestamp(args.end))]
    idx = df.index
    S, E = pd.Timestamp(args.start), pd.Timestamp(args.end)

    s, _, eqc = run_backtest(df, 1000, QTY_PCT, LEV, FEE, SLIP, dict(STRATEGY_PARAMS),
                             trade_start=S, trade_end=E, allow_long=True, allow_short=False)
    trend = eqc.set_index("timestamp")["equity"].reindex(idx).ffill().fillna(1000)

    _, he = harmonic_backtest(df, 1000, trend_filter=True)
    harm = he.set_index("timestamp")["equity"].reindex(idx).ffill().fillna(1000)

    opps = detect_opportunities(df, trend_filter=True)
    mixed = mixed_backtest(df, opps, 1000).set_index("timestamp")["equity"].reindex(idx).ffill().fillna(1000)

    print(f"\n=== 同一筆資金 1000 USDT | {idx[0].date()}~{idx[-1].date()} ===")
    mt = stats(trend, 1000, "純趨勢")
    mh = stats(harm, 1000, "純諧波(順勢)")
    stats(mixed, 1000, "混合")
    print(f"\n  趨勢 vs 諧波 月報酬相關性: {mt.corr(mh):+.2f}  (負/低 → 分散效果好)")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
