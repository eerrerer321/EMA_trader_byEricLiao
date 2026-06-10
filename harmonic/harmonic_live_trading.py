"""
🔱 諧波型態自動交易（實盤）— Bybit
==================================

把 harmonic_strategy.py 驗證過的「不重繪 + 順大勢」諧波策略接上 Bybit 實盤。
設計刻意對齊主專案 eth_strategy_4h_autotrading.py 的模式（連線、sizing、
交易所端停損），但多了諧波特有的「**限價掛單生命週期管理**」：

  趨勢 = 市價進場；諧波 = 在 PRZ 掛限價單，等價格回來觸及才成交。

流程（一次只維護一個活躍交易，互斥、降低風險）：
  1. 每根完成的 4h K 線 → 不重繪偵測最新確認的 X-A-B-C → 算 PRZ。
  2. 順大勢 + 型態有效 + 現價尚未穿過 PRZ → 掛**限價單(帶 SL+TP)**在 PRZ。
  3. 掛單未成交：型態失效(超時 / 價格突破 X)就撤單。
  4. 成交後：SL/TP 已掛在交易所端，自動觸發；本地僅監控。

⚠️ 安全：
  - 預設 DRY_RUN=True（環境變數 HARMONIC_DRY_RUN=0 才真下單）。先觀察訊號數天。
  - 本程式未經實盤實單測試，首次實單請用**最小金額**驗證掛單/SL/TP 是否如預期。
  - 與回測會有差異（限價成交、滑價、樞軸確認延遲）。
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

# 靜音 requests 對 urllib3/chardet 版本的無害相容性警告（不影響運作）
warnings.filterwarnings("ignore", message=".*doesn't match a supported version.*")

import ccxt
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trade_logger import log_event  # noqa: E402
from eth_strategy_4h_autotrading import (  # noqa: E402  複用已驗證的指標與 K 線抓取
    calculate_indicators, fetch_bybit_klines, get_latest_completed_bar, timeframe_to_timedelta,
    MIN_PLAUSIBLE_EQUITY_USDT,
)
from harmonic_strategy import (  # noqa: E402  複用偵測核心
    find_pivots, match, PIVOT_N, WAIT, TP_RATIO, SL_BUFFER, QTY_PCT, LEV,
)

load_dotenv()

# --- 配置 ---
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
SYMBOL = (os.getenv("HARMONIC_SYMBOL") or os.getenv("TRADE_SYMBOL") or "ETH/USDT").upper().replace(" ", "")
TIMEFRAME = os.getenv("HARMONIC_TIMEFRAME", "1h")     # 推薦 1h：機會是 4h 的 3.7 倍且 edge 守住
PIVOT_N_LIVE = int(os.getenv("HARMONIC_PIVOT_N", "3"))  # 樞軸大小（1h≈3, 30m≈8）
DRY_RUN = os.getenv("HARMONIC_DRY_RUN", "1") != "0"   # 預設只觀察、不下單
# DRY-RUN 模擬資金：帳戶餘額不足時改用此金額計算倉位，否則所有訊號會因 qty=0 被吞掉
DRY_RUN_SIM_CAPITAL = float(os.getenv("HARMONIC_DRY_CAPITAL", "1000"))
STOP_TRIGGER_BY = "MarkPrice"
POSITION_IDX_ONE_WAY = 0
TRADE_SLEEP_SECONDS = 60
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "harmonic_state.json")
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "harmonic_trades.csv")


class HarmonicTrader:
    def __init__(self):
        self.exchange = ccxt.bybit({
            "apiKey": BYBIT_API_KEY, "secret": BYBIT_API_SECRET, "sandbox": False,
            "enableRateLimit": True,
            "options": {"defaultType": "linear", "adjustForTimeDifference": True,
                        "recvWindow": 120000, "unified": True},
        })
        try:
            self.exchange.load_time_difference()
        except Exception:
            pass
        self.exchange.load_markets()
        self.symbol = SYMBOL
        self.bybit_symbol = self.exchange.market(self.symbol).get("id", SYMBOL.replace("/", ""))

        # 活躍交易狀態（一次一個）：pending 限價單 或 已成交持倉
        self.pending = None         # {order_id, entry, sl, tp, bull, pattern, x_price, placed_kline, deadline_kline}
        self.position = None        # {entry, sl, tp, bull, pattern, qty}
        self.last_signal_key = None  # 已處理過的 C 樞軸時間戳，避免重複掛同型態
        self.last_processed_kline = None
        self.load_state()
        self._sync_with_exchange()
        print(f"✅ 諧波交易初始化 | {self.symbol} | {'🟡 DRY-RUN 觀察模式' if DRY_RUN else '🔴 實單模式'}")

    # ---------- 基礎設施（對齊主專案） ----------
    def _format_amount(self, amount):
        try:
            return float(self.exchange.amount_to_precision(self.symbol, amount))
        except Exception:
            return round(amount, 3)

    def _format_price(self, price):
        try:
            return self.exchange.price_to_precision(self.symbol, price)
        except Exception:
            return f"{price:.2f}"

    def _free_balance(self):
        try:
            return float(self.exchange.fetch_balance().get("free", {}).get("USDT", 0) or 0)
        except Exception as e:
            print(f"獲取餘額失敗: {e}")
            return 0.0

    def _fetch_raw_positions(self):
        if not hasattr(self.exchange, "private_get_v5_position_list"):
            return []
        resp = self.exchange.private_get_v5_position_list({"category": "linear", "symbol": self.bybit_symbol})
        return resp.get("result", {}).get("list", [])

    def _exchange_position_size(self):
        if DRY_RUN:
            return 0.0
        try:
            if not hasattr(self.exchange, "private_get_v5_position_list"):
                return None
            for pos in self._fetch_raw_positions():
                size = float(pos.get("size", 0) or 0)
                if size > 0:
                    return size if pos.get("side") == "Buy" else -size
            return 0.0
        except Exception as e:
            print(f"查持倉失敗: {e}")
            return None

    def _calc_qty(self, entry_price):
        free = self._free_balance()
        if DRY_RUN and free < MIN_PLAUSIBLE_EQUITY_USDT:
            print(f"💡 DRY-RUN：可用餘額 {free:.2f} USDT 不足，以模擬資金 {DRY_RUN_SIM_CAPITAL:.0f} USDT 計算倉位。")
            free = DRY_RUN_SIM_CAPITAL
        notional = free * QTY_PCT / 100 * LEV
        qty = self._format_amount(notional / entry_price)
        market = self.exchange.market(self.symbol)
        min_amt = market["limits"]["amount"]["min"] if "amount" in market["limits"] else 0.001
        return qty if qty >= min_amt else 0.0

    def _deadline_from(self, kline_time):
        if kline_time is None:
            return None
        return kline_time + timeframe_to_timedelta(TIMEFRAME) * WAIT

    # ---------- 偵測（不重繪 + 順大勢） ----------
    def _detect_signal(self, df):
        """回傳最新確認 X-A-B-C 的 PRZ 掛單參數，或 None。"""
        alt = find_pivots(df, PIVOT_N_LIVE)
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
        if not ((close > ema) if bull else (close < ema)):   # 順大勢過濾
            return None
        name, prz_lo, prz_hi = cands[0]
        if bull:
            entry, sl, tp = prz_hi, prz_lo * (1 - SL_BUFFER), prz_hi + TP_RATIO * (A[2] - prz_hi)
            if not (close > prz_hi and sl < entry < tp):     # 現價需仍在 PRZ 上方（等回落）
                return None
        else:
            entry, sl, tp = prz_lo, prz_hi * (1 + SL_BUFFER), prz_lo - TP_RATIO * (prz_lo - A[2])
            if not (close < prz_lo and tp < entry < sl):
                return None
        # dedup key 必須用 C 樞軸的「時間戳」：C[1] 是滾動視窗內的位置索引，
        # 每根新 K 線都會位移，拿來去重會誤判（同型態重複掛單／不同型態被誤擋）
        return {"entry": entry, "sl": sl, "tp": tp, "bull": bull, "pattern": name,
                "x_price": X[2], "c_time": str(df.index[C[1]])}

    # ---------- 下單（限價 + 交易所端 SL/TP） ----------
    def _place_limit_with_sltp(self, sig):
        entry = float(self._format_price(sig["entry"]))
        qty = self._calc_qty(entry)
        if qty <= 0:
            print(f"🚨 偵測到 {sig['pattern']} 訊號，但資金不足最小下單量，略過（訊號不視為已處理）。")
            log_event(LOG_FILE, strategy="harmonic", timeframe=TIMEFRAME, action="skip",
                      side="buy" if sig["bull"] else "sell", pattern=sig["pattern"],
                      price=entry, reason="insufficient_funds", dry=DRY_RUN)
            return False
        side = "buy" if sig["bull"] else "sell"
        params = {
            "category": "linear", "positionIdx": POSITION_IDX_ONE_WAY,
            "timeInForce": "GTC", "reduceOnly": False,
            "stopLoss": self._format_price(sig["sl"]), "slTriggerBy": STOP_TRIGGER_BY, "slOrderType": "Market",
            "takeProfit": self._format_price(sig["tp"]), "tpTriggerBy": STOP_TRIGGER_BY, "tpOrderType": "Market",
            "tpslMode": "Full",
        }
        print(f"\n🎯 諧波 {sig['pattern']} {'多' if sig['bull'] else '空'} | 掛限價 {side} {qty} @ {entry} "
              f"| SL {params['stopLoss']} TP {params['takeProfit']}")
        if DRY_RUN:
            print("   (DRY-RUN：不實際下單)")
            self.pending = {"order_id": "DRY", "entry": entry, "sl": sig["sl"], "tp": sig["tp"],
                            "bull": sig["bull"], "pattern": sig["pattern"], "x_price": sig["x_price"],
                            "deadline": self._deadline_from(self.last_processed_kline), "qty": qty}
            self.save_state()
            log_event(LOG_FILE, strategy="harmonic", timeframe=TIMEFRAME, action="place", side=side,
                      pattern=sig["pattern"], price=entry, qty=qty, sl=round(sig["sl"], 2),
                      tp=round(sig["tp"], 2), dry=True, note="dry-run")
            return True
        try:
            order = self.exchange.create_order(self.symbol, "limit", side, qty, entry, params)
            self.pending = {"order_id": order.get("id"), "entry": entry, "sl": sig["sl"], "tp": sig["tp"],
                            "bull": sig["bull"], "pattern": sig["pattern"], "x_price": sig["x_price"],
                            "deadline": self._deadline_from(self.last_processed_kline), "qty": qty}
            self.save_state()
            print(f"   ✅ 已掛單 id={order.get('id')}")
            log_event(LOG_FILE, strategy="harmonic", timeframe=TIMEFRAME, action="place", side=side,
                      pattern=sig["pattern"], price=entry, qty=qty, sl=round(sig["sl"], 2),
                      tp=round(sig["tp"], 2), dry=False, order_id=order.get("id", ""))
            return True
        except Exception as e:
            print(f"   ❌ 掛單失敗: {e}")
            return False

    def _cancel_pending(self, reason):
        if not self.pending:
            return True
        print(f"🗑️ 撤銷掛單 ({reason}) id={self.pending.get('order_id')}")
        if not DRY_RUN:
            try:
                self.exchange.cancel_order(self.pending["order_id"], self.symbol, {"category": "linear"})
            except Exception as e:
                print(f"   撤單失敗(可能已成交/已撤): {e}")
                pos_size = self._exchange_position_size()
                if pos_size is not None and pos_size != 0:
                    print("⚠️ 撤單失敗且偵測到持倉，改列為諧波持倉，避免重複開倉。")
                    self.position = {**self.pending, "qty": abs(pos_size)}
                    self.pending = None
                    self.save_state()
                    return False
                print("⚠️ 撤單結果不明，保留掛單狀態，避免誤開新倉。")
                self.save_state()
                return False
        log_event(LOG_FILE, strategy="harmonic", timeframe=TIMEFRAME, action="cancel",
                  pattern=self.pending.get("pattern", ""), reason=reason, dry=DRY_RUN,
                  order_id=self.pending.get("order_id", ""))
        self.pending = None
        self.save_state()
        return True

    def _check_pending(self, df):
        """掛單未成交時：偵測是否已成交 / 型態失效需撤單。"""
        pos_size = self._exchange_position_size()
        if pos_size is None:
            print("⚠️ 無法確認掛單是否成交，保留掛單狀態。")
            return
        dry_filled = DRY_RUN and self._dry_pending_filled(df)
        if pos_size != 0 or dry_filled:   # 已成交 → 轉持倉（SL/TP 已掛交易所端）
            qty = self.pending.get("qty", abs(pos_size)) if DRY_RUN else abs(pos_size)
            self.position = {**self.pending, "qty": qty}
            log_event(LOG_FILE, strategy="harmonic", timeframe=TIMEFRAME, action="fill",
                      side="buy" if self.pending["bull"] else "sell", pattern=self.pending["pattern"],
                      price=self.pending["entry"], qty=qty, sl=round(self.pending["sl"], 2),
                      tp=round(self.pending["tp"], 2), dry=DRY_RUN)
            self.pending = None
            self.save_state()
            print(f"✅ 限價單成交，轉持倉。SL/TP 由交易所端自動執行。")
            return
        # 失效判斷：價格突破 X（型態作廢）或超過等待期限
        bull = self.pending["bull"]; close = float(df["close"].iloc[-1])
        if (bull and close < self.pending["x_price"]) or (not bull and close > self.pending["x_price"]):
            self._cancel_pending("價格突破 X，型態失效")
        elif self.pending.get("deadline") and self.last_processed_kline and self.last_processed_kline >= self.pending["deadline"]:
            self._cancel_pending("超過等待期限未成交")

    def _dry_pending_filled(self, df):
        bull = self.pending["bull"]
        hi, lo = float(df["high"].iloc[-1]), float(df["low"].iloc[-1])
        return (lo <= self.pending["entry"]) if bull else (hi >= self.pending["entry"])

    def _monitor_dry_position(self, df):
        if self.position.get("sl") is None or self.position.get("tp") is None:
            print("⚠️ DRY-RUN 持倉缺少 SL/TP，保留狀態等待人工確認。")
            return
        hi, lo, bull = float(df["high"].iloc[-1]), float(df["low"].iloc[-1]), self.position["bull"]
        if bull:
            hit = "SL" if lo <= self.position["sl"] else ("TP" if hi >= self.position["tp"] else None)
        else:
            hit = "SL" if hi >= self.position["sl"] else ("TP" if lo <= self.position["tp"] else None)
        if not hit:
            return
        px = self.position["sl"] if hit == "SL" else self.position["tp"]
        pnl = ((px - self.position["entry"]) if bull else (self.position["entry"] - px)) * self.position["qty"]
        log_event(LOG_FILE, strategy="harmonic", timeframe=TIMEFRAME, action="exit",
                  side="sell" if bull else "buy", pattern=self.position["pattern"], price=round(px, 2),
                  qty=self.position["qty"], reason=hit, pnl=round(pnl, 2), dry=True)
        print(f"✅ (DRY) {hit} 觸及 @ {px:.2f}，回到等待型態。")
        self.position = None
        self.last_signal_key = None
        self.save_state()

    # ---------- 狀態 ----------
    def save_state(self):
        state = {"pending": self.pending, "position": self.position,
                 "last_signal_key": self.last_signal_key,
                 "last_processed_kline": self.last_processed_kline.isoformat() if self.last_processed_kline else None}
        try:
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp, STATE_FILE)
        except Exception as e:
            print(f"存檔失敗: {e}")

    def load_state(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            s = json.load(open(STATE_FILE, encoding="utf-8"))
            self.pending = s.get("pending"); self.position = s.get("position")
            self.last_signal_key = s.get("last_signal_key")
            lk = s.get("last_processed_kline")
            self.last_processed_kline = pd.to_datetime(lk) if lk else None
            if self.pending and self.pending.get("deadline"):
                self.pending["deadline"] = pd.to_datetime(self.pending["deadline"])
        except Exception as e:
            print(f"載入狀態失敗: {e}")

    def _sync_with_exchange(self):
        """以交易所實際持倉校正本地狀態（重啟/手動干預後）。"""
        if DRY_RUN:
            return
        pos_size = self._exchange_position_size()
        if pos_size is None:
            print("⚠️ 無法確認交易所持倉，保留本地狀態。")
            return
        if pos_size == 0:
            if self.position:
                print("交易所已無持倉，清除本地持倉狀態。")
                self.position = None; self.save_state()
        else:
            if not self.position:
                print(f"⚠️ 偵測到未記錄的持倉 {pos_size}，SL/TP 請自行於 Bybit 確認。")
                self.position = {"order_id": "", "entry": None, "sl": None, "tp": None,
                                 "bull": pos_size > 0, "pattern": "external", "qty": abs(pos_size)}
                self.save_state()

    # ---------- 主流程 ----------
    def process(self, bar, df):
        # 1) 有持倉：交易所端 SL/TP 自動執行，僅監控
        if self.position is not None:
            if DRY_RUN:
                self._monitor_dry_position(df)
                return
            pos_size = self._exchange_position_size()
            if pos_size is None:
                print("⚠️ 無法確認持倉狀態，保留本地狀態。")
                return
            if pos_size == 0:
                print("✅ 持倉已由交易所端 SL/TP 平倉，回到等待型態。")
                log_event(LOG_FILE, strategy="harmonic", timeframe=TIMEFRAME, action="exit",
                          pattern=self.position.get("pattern", ""), reason="exchange_sl_tp", dry=DRY_RUN)
                self.position = None; self.last_signal_key = None; self.save_state()
            return
        # 2) 有掛單：管理生命週期
        if self.pending is not None:
            self._check_pending(df)
            return
        # 3) 空閒：偵測新型態 → 掛限價單
        sig = self._detect_signal(df)
        if not sig or sig["c_time"] == self.last_signal_key:
            return
        # 只有真的掛出單才消耗 dedup key，否則訊號會被無聲吞掉
        if self._place_limit_with_sltp(sig):
            self.last_signal_key = sig["c_time"]
        self.save_state()

    def run(self):
        print("--- 諧波實盤交易啟動 ---")
        last_check = 0
        while True:
            try:
                now = time.time()
                if not DRY_RUN and now - last_check >= TRADE_SLEEP_SECONDS:
                    self._sync_with_exchange()
                if now - last_check >= TRADE_SLEEP_SECONDS:
                    # limit=1000：500 根視窗的 EMA200 暖機殘差會讓順大勢過濾與回測不一致
                    df = calculate_indicators(fetch_bybit_klines(SYMBOL, TIMEFRAME, limit=1000))
                    bar = get_latest_completed_bar(df, TIMEFRAME)
                    if bar is not None:
                        if self.last_processed_kline is None or bar.name > self.last_processed_kline:
                            print(f"\n🔔 新 4h K 線 {bar.name} | 價格 {bar['close']:.2f} | EMA200 {bar['ema200']:.2f}")
                            self.last_processed_kline = bar.name
                            self.process(bar, df)
                            self.save_state()
                        else:
                            # 同根 K 線內也要管理掛單（價格可能觸及/失效）
                            if self.pending is not None:
                                self._check_pending(df)
                    last_check = now
                time.sleep(1)
            except Exception as e:
                print(f"\n❌ 主迴圈錯誤: {e}")
                time.sleep(TRADE_SLEEP_SECONDS)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    print("🔱 諧波型態自動交易")
    print(f"   標的 {SYMBOL} | 模式 {'DRY-RUN(觀察)' if DRY_RUN else '實單'}")
    if not DRY_RUN:
        try:
            input("⚠️ 實單模式，按 Enter 確認風險後開始...")
        except EOFError:
            time.sleep(2)
    HarmonicTrader().run()
