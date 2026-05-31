"""
🔱+📈 混合策略自動交易（實盤）— 4h 趨勢 + 1h 諧波，同一資金池
================================================================

把主專案的 EMA 趨勢策略（4h）與本資料夾的順大勢諧波策略（1h）放在**同一個
Bybit 帳戶、同一筆資金**上分時複用，對齊回測 run_mixed.py 的協調規則：

  - **互斥持倉**：一次只有一個倉位（趨勢 或 諧波）。
  - **趨勢優先**：趨勢訊號出現時，若諧波只是「掛單未成交」→ 取消諧波單讓趨勢進；
    若諧波已成交持倉 → 趨勢讓步等諧波結束。
  - 趨勢空倉（~85% 時間）→ 諧波用閒置資金在 1h 上找順大勢回調機會。

趨勢出場邏輯直接重用回測的 Position / update_trailing_stop / check_stop_exit，
與主專案一致（固定停損 + 移動停損 + 反向訊號）。諧波為 PRZ 限價單帶交易所 SL/TP。

⚠️ 安全：
  - 預設 DRY_RUN=True（環境變數 HARMONIC_DRY_RUN=0 才實際下單）。請先觀察數天。
  - **未經實單測試**；首次實單用最小金額，於 Bybit App 確認下單/SL/TP。
  - 與回測必有差異（成交價、滑價、樞軸確認延遲、4h/1h 邊界對齊）。
"""
from __future__ import annotations

import json
import os
import sys
import time

import ccxt
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trade_logger import log_event  # noqa: E402
from eth_strategy_4h_autotrading import (  # noqa: E402
    calculate_indicators, fetch_bybit_klines, get_latest_completed_bar, STRATEGY_PARAMS,
)
from backtest_eth_strategy_4h import (  # noqa: E402
    Position, update_trailing_stop, check_stop_exit, long_signal, short_signal,
)
from harmonic_strategy import find_pivots, match, TP_RATIO, SL_BUFFER, QTY_PCT, LEV  # noqa: E402

load_dotenv()
API_KEY, API_SECRET = os.getenv("BYBIT_API_KEY"), os.getenv("BYBIT_API_SECRET")
SYMBOL = os.getenv("HARMONIC_SYMBOL", "ETH/USDT")
TREND_TF = "4h"
HARM_TF = os.getenv("HARMONIC_TIMEFRAME", "1h")
HARM_PIVOT_N = int(os.getenv("HARMONIC_PIVOT_N", "3"))
HARM_WAIT = 60
DRY_RUN = os.getenv("HARMONIC_DRY_RUN", "1") != "0"
STOP_TRIGGER_BY = "MarkPrice"
POSITION_IDX = 0
P = dict(STRATEGY_PARAMS)
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mixed_state.json")
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "mixed_trades.csv")


class MixedLiveTrader:
    def __init__(self):
        self.ex = ccxt.bybit({
            "apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True,
            "options": {"defaultType": "linear", "adjustForTimeDifference": True,
                        "recvWindow": 120000, "unified": True}})
        try:
            self.ex.load_time_difference()
        except Exception:
            pass
        self.ex.load_markets()
        self.symbol = SYMBOL
        self.bybit_symbol = self.ex.market(self.symbol).get("id", SYMBOL.replace("/", ""))
        self.active = "none"        # none / trend / harm_pending / harm_pos
        self.trend = None           # backtest Position
        self.harm = None            # dict
        self.last_4h = None
        self.last_1h = None
        self.load_state()
        print(f"✅ 混合交易初始化 | {self.symbol} | 趨勢 {TREND_TF} + 諧波 {HARM_TF}(N={HARM_PIVOT_N}) "
              f"| {'🟡 DRY-RUN' if DRY_RUN else '🔴 實單'} | 目前狀態: {self.active}")

    # ---------- 基礎 ----------
    def _fmt_amt(self, a):
        try:
            return float(self.ex.amount_to_precision(self.symbol, a))
        except Exception:
            return round(a, 3)

    def _fmt_px(self, p):
        try:
            return self.ex.price_to_precision(self.symbol, p)
        except Exception:
            return f"{p:.2f}"

    def _free(self):
        try:
            return float(self.ex.fetch_balance().get("free", {}).get("USDT", 0) or 0)
        except Exception:
            return 0.0

    def _calc_qty(self, price):
        notional = self._free() * QTY_PCT / 100 * LEV
        qty = self._fmt_amt(notional / price)
        mn = self.ex.market(self.symbol)["limits"]["amount"]["min"] or 0.001
        return qty if qty >= mn else 0.0

    def _exch_pos_size(self):
        try:
            if hasattr(self.ex, "private_get_v5_position_list"):
                for pos in self.ex.private_get_v5_position_list(
                        {"category": "linear", "symbol": self.bybit_symbol}).get("result", {}).get("list", []):
                    sz = float(pos.get("size", 0) or 0)
                    if sz > 0:
                        return sz if pos.get("side") == "Buy" else -sz
        except Exception as e:
            print(f"查持倉失敗: {e}")
        return 0.0

    def _market(self, side, qty, sl=None, tp=None):
        params = {"category": "linear", "positionIdx": POSITION_IDX, "reduceOnly": False}
        if sl:
            params.update({"stopLoss": self._fmt_px(sl), "slTriggerBy": STOP_TRIGGER_BY,
                           "slOrderType": "Market", "tpslMode": "Full"})
        if tp:
            params.update({"takeProfit": self._fmt_px(tp), "tpTriggerBy": STOP_TRIGGER_BY, "tpOrderType": "Market"})
        if DRY_RUN:
            return {"id": "DRY"}
        return self.ex.create_order(self.symbol, "market", side, qty, None, params)

    def _limit(self, side, qty, price, sl, tp):
        params = {"category": "linear", "positionIdx": POSITION_IDX, "reduceOnly": False, "timeInForce": "GTC",
                  "stopLoss": self._fmt_px(sl), "slTriggerBy": STOP_TRIGGER_BY, "slOrderType": "Market",
                  "takeProfit": self._fmt_px(tp), "tpTriggerBy": STOP_TRIGGER_BY, "tpOrderType": "Market",
                  "tpslMode": "Full"}
        if DRY_RUN:
            return {"id": "DRY"}
        return self.ex.create_order(self.symbol, "limit", side, qty, price, params)

    def _close_market(self, side):
        if DRY_RUN:
            return
        self.ex.create_order(self.symbol, "market", side, abs(self.trend.qty) if self.trend else 0,
                             None, {"category": "linear", "positionIdx": POSITION_IDX, "reduceOnly": True})

    def _cancel(self, oid):
        if DRY_RUN or not oid or oid == "DRY":
            return
        try:
            self.ex.cancel_order(oid, self.symbol, {"category": "linear"})
        except Exception as e:
            print(f"撤單失敗(可能已成交): {e}")

    # ---------- 趨勢（4h，重用回測邏輯） ----------
    def _open_trend(self, bar):
        entry = float(bar["close"])
        qty = self._calc_qty(entry)
        if qty <= 0:
            print("資金不足，趨勢進場略過。")
            return
        sl = entry * (1 - P["long_fixed_stop_loss_percent"])
        print(f"🟢 [趨勢] 市價做多 {qty} @ ~{entry:.2f} | 交易所固定SL {sl:.2f}")
        order = self._market("buy", qty, sl=sl)
        if order is None:
            return
        self.trend = Position(side="long", qty=qty, entry_price=entry, entry_time=bar.name, entry_fee=0.0, peak=entry)
        self.active = "trend"
        log_event(LOG_FILE, strategy="trend", timeframe=TREND_TF, action="entry", side="long",
                  price=round(entry, 2), qty=qty, sl=round(sl, 2), dry=DRY_RUN, order_id=order.get("id", ""))
        self.save_state()

    def _manage_trend(self, bar):
        if short_signal(bar, P):
            px = float(bar["close"]); pnl = (px - self.trend.entry_price) * self.trend.qty
            print("🔴 [趨勢] 反向訊號 → 平多")
            self._close_market("sell")
            log_event(LOG_FILE, strategy="trend", timeframe=TREND_TF, action="exit", side="sell",
                      price=round(px, 2), qty=self.trend.qty, reason="reverse", pnl=round(pnl, 2), dry=DRY_RUN)
            self.trend = None; self.active = "none"; self.save_state(); return
        st = check_stop_exit(self.trend, bar, P)
        if st:
            pnl = (st[1] - self.trend.entry_price) * self.trend.qty
            print(f"🔴 [趨勢] {st[0]} 觸發 @ {st[1]:.2f} → 平多")
            self._close_market("sell")
            log_event(LOG_FILE, strategy="trend", timeframe=TREND_TF, action="exit", side="sell",
                      price=round(st[1], 2), qty=self.trend.qty, reason=st[0], pnl=round(pnl, 2), dry=DRY_RUN)
            self.trend = None; self.active = "none"; self.save_state(); return
        update_trailing_stop(self.trend, bar, P)
        self.trend.bars_held += 1
        if self.trend.trail_active and self.trend.trail_stop:
            print(f"   [趨勢] 移動停損更新 → {self.trend.trail_stop:.2f}（請同步交易所端）")

    # ---------- 諧波（1h） ----------
    def _detect_harm(self, df):
        alt = find_pivots(df, HARM_PIVOT_N)
        if len(alt) < 4:
            return None
        X, A, B, C = alt[-4], alt[-3], alt[-2], alt[-1]
        if not (X[3] != A[3] and A[3] != B[3] and B[3] != C[3]):
            return None
        bull = X[3] == "L"
        cands = match(X[2], A[2], B[2], C[2], bull)
        if not cands:
            return None
        close, ema = float(df["close"].iloc[-1]), float(df["ema200"].iloc[-1])
        if not ((close > ema) if bull else (close < ema)):
            return None
        name, lo, hi = cands[0]
        if bull:
            entry, sl, tp = hi, lo * (1 - SL_BUFFER), hi + TP_RATIO * (A[2] - hi)
            if not (close > hi and sl < entry < tp):
                return None
        else:
            entry, sl, tp = lo, hi * (1 + SL_BUFFER), lo - TP_RATIO * (lo - A[2])
            if not (close < lo and tp < entry < sl):
                return None
        return {"entry": entry, "sl": sl, "tp": tp, "bull": bull, "pattern": name,
                "x_price": X[2], "c_time": str(C[1])}

    def _place_harm(self, sig):
        entry = float(self._fmt_px(sig["entry"]))
        qty = self._calc_qty(entry)
        if qty <= 0:
            return
        side = "buy" if sig["bull"] else "sell"
        print(f"🎯 [諧波] {sig['pattern']} {'多' if sig['bull'] else '空'} 掛限價 {side} {qty} @ {entry} "
              f"| SL {sig['sl']:.2f} TP {sig['tp']:.2f}")
        order = self._limit(side, qty, entry, sig["sl"], sig["tp"])
        self.harm = {**sig, "order_id": order.get("id"), "qty": qty}
        self.active = "harm_pending"
        log_event(LOG_FILE, strategy="harmonic", timeframe=HARM_TF, action="place", side=side,
                  pattern=sig["pattern"], price=entry, qty=qty, sl=round(sig["sl"], 2), tp=round(sig["tp"], 2),
                  dry=DRY_RUN, order_id=order.get("id", ""))
        self.save_state()

    def _cancel_harm(self, reason):
        if not self.harm:
            return
        print(f"🗑️ [諧波] 撤掛單（{reason}）")
        log_event(LOG_FILE, strategy="harmonic", timeframe=HARM_TF, action="cancel",
                  pattern=self.harm.get("pattern", ""), reason=reason, dry=DRY_RUN,
                  order_id=self.harm.get("order_id", ""))
        self._cancel(self.harm.get("order_id"))
        self.harm = None
        self.active = "none"
        self.save_state()

    def _manage_harm_pending(self, df):
        if self._exch_pos_size() != 0 or (DRY_RUN and self._dry_harm_filled(df)):
            print("✅ [諧波] 限價單成交 → 持倉（SL/TP 由交易所端執行）")
            log_event(LOG_FILE, strategy="harmonic", timeframe=HARM_TF, action="fill",
                      side="buy" if self.harm["bull"] else "sell", pattern=self.harm["pattern"],
                      price=self.harm["entry"], qty=self.harm["qty"], sl=round(self.harm["sl"], 2),
                      tp=round(self.harm["tp"], 2), dry=DRY_RUN)
            self.active = "harm_pos"; self.save_state(); return
        close = float(df["close"].iloc[-1]); bull = self.harm["bull"]
        if (bull and close < self.harm["x_price"]) or (not bull and close > self.harm["x_price"]):
            self._cancel_harm("價格突破 X，型態失效")

    def _dry_harm_filled(self, df):
        """DRY-RUN 模擬：最新 1h K 線是否觸及限價。"""
        bull = self.harm["bull"]; hi, lo = float(df["high"].iloc[-1]), float(df["low"].iloc[-1])
        return (lo <= self.harm["entry"]) if bull else (hi >= self.harm["entry"])

    def _monitor_harm(self):
        if self._exch_pos_size() == 0 and not DRY_RUN:
            print("✅ [諧波] 持倉已由交易所 SL/TP 平倉 → 回到等待")
            log_event(LOG_FILE, strategy="harmonic", timeframe=HARM_TF, action="exit",
                      pattern=self.harm.get("pattern", "") if self.harm else "",
                      reason="exchange_sl_tp", dry=DRY_RUN)
            self.harm = None; self.active = "none"; self.save_state()

    # ---------- 協調 ----------
    def on_4h(self, bar):
        if self.active == "trend":
            self._manage_trend(bar)
        elif self.active in ("none", "harm_pending"):
            if long_signal(bar, P):
                if self.active == "harm_pending":
                    self._cancel_harm("趨勢優先")
                self._open_trend(bar)
        # harm_pos: 趨勢讓步等諧波結束

    def on_1h(self, bar, df):
        if self.active == "trend":
            return
        if self.active == "harm_pending":
            self._manage_harm_pending(df)
        elif self.active == "harm_pos":
            self._monitor_harm()
        elif self.active == "none":
            sig = self._detect_harm(df)
            if sig:
                self._place_harm(sig)

    # ---------- 狀態 ----------
    def save_state(self):
        tp = None
        if self.trend:
            tp = {"side": self.trend.side, "qty": self.trend.qty, "entry_price": self.trend.entry_price,
                  "entry_time": str(self.trend.entry_time), "peak": self.trend.peak,
                  "trail_active": self.trend.trail_active, "trail_stop": self.trend.trail_stop,
                  "bars_held": self.trend.bars_held}
        st = {"active": self.active, "trend": tp, "harm": self.harm,
              "last_4h": str(self.last_4h) if self.last_4h is not None else None,
              "last_1h": str(self.last_1h) if self.last_1h is not None else None}
        try:
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(st, f, indent=2, default=str); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, STATE_FILE)
        except Exception as e:
            print(f"存檔失敗: {e}")

    def load_state(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            s = json.load(open(STATE_FILE, encoding="utf-8"))
            self.active = s.get("active", "none"); self.harm = s.get("harm")
            self.last_4h = pd.to_datetime(s["last_4h"]) if s.get("last_4h") else None
            self.last_1h = pd.to_datetime(s["last_1h"]) if s.get("last_1h") else None
            tp = s.get("trend")
            if tp:
                self.trend = Position(side=tp["side"], qty=tp["qty"], entry_price=tp["entry_price"],
                                      entry_time=pd.to_datetime(tp["entry_time"]), entry_fee=0.0,
                                      peak=tp.get("peak"), trail_active=tp.get("trail_active", False),
                                      trail_stop=tp.get("trail_stop"), bars_held=tp.get("bars_held", 0))
        except Exception as e:
            print(f"載入狀態失敗: {e}")

    # ---------- 主迴圈 ----------
    def run(self):
        print(f"--- 混合實盤啟動（互斥、趨勢優先）---  狀態: {self.active}")
        last = 0
        while True:
            try:
                if time.time() - last >= 60:
                    df4 = calculate_indicators(fetch_bybit_klines(SYMBOL, TREND_TF, limit=300))
                    b4 = get_latest_completed_bar(df4, TREND_TF)
                    if b4 is not None and (self.last_4h is None or b4.name > self.last_4h):
                        print(f"\n🔔 新 4h K 線 {b4.name} | 價 {b4['close']:.2f} EMA200 {b4['ema200']:.2f} | 狀態 {self.active}")
                        self.last_4h = b4.name; self.on_4h(b4); self.save_state()

                    df1 = calculate_indicators(fetch_bybit_klines(SYMBOL, HARM_TF, limit=500))
                    b1 = get_latest_completed_bar(df1, HARM_TF)
                    if b1 is not None and (self.last_1h is None or b1.name > self.last_1h):
                        self.last_1h = b1.name; self.on_1h(b1, df1); self.save_state()
                    last = time.time()
                time.sleep(1)
            except Exception as e:
                print(f"\n❌ 主迴圈錯誤: {e}")
                time.sleep(60)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    print("🔱+📈 混合自動交易 | 4h 趨勢 + 1h 諧波 | 同一資金池、互斥、趨勢優先")
    print(f"   {SYMBOL} | {'DRY-RUN(觀察)' if DRY_RUN else '實單'}")
    if not DRY_RUN:
        try:
            input("⚠️ 實單模式，按 Enter 確認風險...")
        except EOFError:
            time.sleep(2)
    MixedLiveTrader().run()
