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
import warnings

# 靜音 requests 對 urllib3/chardet 版本的無害相容性警告（不影響運作）
warnings.filterwarnings("ignore", message=".*doesn't match a supported version.*")

import ccxt
import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trade_logger import log_event, write_active, ensure_log_header  # noqa: E402
from strategy_core import (  # noqa: E402
    calculate_indicators, fetch_bybit_klines, get_latest_completed_bar, timeframe_to_timedelta,
    STRATEGY_PARAMS, MIN_PLAUSIBLE_EQUITY_USDT, MAX_PLAUSIBLE_DRAWDOWN, FUNDING_BOOST, apply_funding_boost,
    fetch_btc_funding_3d,
)
from backtest_eth_strategy_4h import (  # noqa: E402
    Position, update_trailing_stop, check_stop_exit, long_signal, short_signal,
    vol_target_scale, live_vol_target, circuit_breaker_scale, live_circuit_breaker,
)
from harmonic_strategy import find_pivots, match, TP_RATIO, SL_BUFFER, QTY_PCT, LEV  # noqa: E402

load_dotenv()
API_KEY, API_SECRET = os.getenv("BYBIT_API_KEY"), os.getenv("BYBIT_API_SECRET")
SYMBOL = (os.getenv("HARMONIC_SYMBOL") or os.getenv("TRADE_SYMBOL") or "ETH/USDT").upper().replace(" ", "")
TREND_TF = "4h"
HARM_TF = os.getenv("HARMONIC_TIMEFRAME", "1h")
HARM_PIVOT_N = int(os.getenv("HARMONIC_PIVOT_N", "3"))
HARM_WAIT = 60
DRY_RUN = os.getenv("HARMONIC_DRY_RUN", "1") != "0"
# DRY-RUN 模擬資金：帳戶餘額不足時改用此金額計算倉位，否則所有訊號會因 qty=0 被吞掉
DRY_RUN_SIM_CAPITAL = float(os.getenv("HARMONIC_DRY_CAPITAL", "1000"))
STOP_TRIGGER_BY = "MarkPrice"
POSITION_IDX = 0
STOP_LOSS_SYNC_RETRIES = 3
STOP_LOSS_SYNC_RETRY_DELAY_SECONDS = 2
P = dict(STRATEGY_PARAMS)
# 趨勢腿波動度目標倉位（與主程式同一組設定）。混合回測實證（Binance 2017-2026 /
# Bybit 2021-2026）：只加趨勢腿報酬 291%→530% / 99%→182%、Calmar 0.51→0.69 /
# 0.80→1.10，滾動視窗獲利率不變；諧波腿加了反而讓滾動獲利率 78%→74%，故諧波維持固定倉位。
VOL_TARGET = live_vol_target()
CIRCUIT_BREAKER = live_circuit_breaker()  # B9：回撤熔斷設定（None=停用）；以權益高水位回撤分級降/暫停新開倉

# === 深負費率「抄底腿」（使用者 2026-06-12 核准半倉版） ===
# 進場：3 日均 BTC 幣本位費率「由上往下穿越」-0.01%/8h（深負=空方深度擁擠=軋空燃料；
#       同一輪深負期間只進一次，需先回到閾值上方重新武裝）。
# 出場：30 天（180 根 4h）時間出場。刻意無停損——回測四變體實證 -10% 災難停損
#       全面有害（會精準砍在擠壓前最低點）。
# 倉位：半倉 10%×3 × 波動度係數（不吃費率加碼）。三腿協調回測（ETH Bybit 21-26）：
#       風險包絡與現行完全相同（MDD/worst/wMDD 一字不差），滾動獲利率 73.1→76.9%；
#       長歷史（Binance 18-26）350%→473%、獲利率 74.4→83.7%、wMDD 不變。
# 風險：設計上在恐慌中接刀，2021-11~2022-06 瀑布段該腿 PF 僅 0.40，靠其後 PF 10+ 補回。
DIP_ENABLED = os.getenv("HARMONIC_DIP_ENABLED", "1") != "0"
DIP_QTY_PCT = QTY_PCT / 2          # 半倉
DIP_FUNDING_THRESHOLD = -0.0001    # 與費率假說 deep_neg 分桶邊界一致（非擬合值）

# === CVD 背離降風險微調（使用者 2026-06-15 觀察＋核准） ===
# 規律：價格漲但 CVD（taker 主動買−賣）沒跟上＝空虛上漲/追多脆弱 → 易跌。
# 動作：趨勢腿開倉時，若近 42 根(4h) 正規化 CVD 動能(Σdelta/Σvol) < 0，倉位 ×0.5（半倉）。
#       只作用在趨勢腿（與回測一致），諧波/抄底腿不受影響。CVD 與費率正交（相關 −0.21）。
# 實證（完整 5 關：前瞻→OOS→正交→獨立增量→混合增量全過）：三腿混合 ETH MDD −18.7→−14.6、
#   Calmar 1.57→1.82、滾動 80.8→84.6%、Sharpe 持平；BTC 全面改善。是降風險 overlay（換報酬得更穩/更小回撤）。
# CVD 取自 Binance 期貨 K 線 taker 量（Bybit K 線不提供），失敗 fail-open（不調整，full size）。
CVD_TILT_ENABLED = os.getenv("HARMONIC_CVD_TILT", "1") != "0"
CVD_TILT_FACTOR = 0.5
CVD_TILT_LOOKBACK = 42             # 4h 根數（與回測一致）
_BINANCE_FAPI = "https://fapi.binance.com/fapi/v1/klines"
_LAST_GOOD_CVD = None              # (epoch_seconds, momentum)，fail-open 沿用
DIP_HOLD_BARS = 180                # 30 天的 4h 根數
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mixed_state.json")


# active 狀態機內部值 → 人類可讀標籤（僅顯示用；state 檔與比較邏輯維持原值）
ACTIVE_LABELS = {
    "none": "空倉等待訊號",
    "trend": "趨勢持倉中",
    "harm_pending": "諧波掛單等待成交",
    "harm_pos": "諧波持倉中",
    "dip_pos": "抄底持倉中(30天時間出場)",
    "external_pos": "外部持倉(暫停新訊號)",
}


def _active_label(active):
    return ACTIVE_LABELS.get(active, str(active))


def _tw(ts):
    """K 線時間戳（UTC）轉台灣時間字串，僅供顯示；內部狀態/比較一律維持 UTC。"""
    try:
        return str(pd.Timestamp(ts) + pd.Timedelta(hours=8))
    except Exception:
        return str(ts)
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "mixed_trades.csv")
ACTIVE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "mixed_active.csv")


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
        self.active = "none"        # none / trend / harm_pending / harm_pos / dip_pos / external_pos
        self.trend = None           # backtest Position
        self.harm = None            # dict
        self.dip = None             # dict {entry, qty, entry_time}（深負費率抄底腿）
        self.dip_armed = True       # fresh-crossing 武裝狀態（費率回到閾值上方才重新武裝）
        self.last_signal_key = None  # 已處理過的諧波 C 樞軸，避免對同型態重複掛單
        self.last_4h = None
        self.last_1h = None
        self.equity_peak = 0.0      # B9：權益高水位（HWM），回撤熔斷用；由 load_state 還原
        self._last_good_equity = None  # B9：(epoch, equity) 最近一次成功讀到的總權益，瞬斷時沿用
        self.load_state()
        if CIRCUIT_BREAKER and self.equity_peak <= 0:  # B9：首次啟動/無持久化 → 以當前總權益播種
            eq0 = self._total_equity()
            if eq0 >= MIN_PLAUSIBLE_EQUITY_USDT:
                self.equity_peak = eq0
                print(f"⚠️ [熔斷] 無歷史權益高水位記錄，以當前總權益 {eq0:.2f} USDT 播種（歷史回撤未知）。")
            else:
                print("⚠️ [熔斷] 啟動時無法取得合理總權益，高水位暫未播種（下次同步補上）。")
        if not self._check_position_mode():
            raise SystemExit("持倉模式不符，已中止啟動。")
        self._ensure_leverage_capacity()  # 啟動先驗一次（失敗只警告；每次開倉前還會再擋）
        self._sync_with_exchange("啟動同步")
        ensure_log_header(LOG_FILE)  # 啟動即建立交易事件日誌（空表頭，待第一筆交易填入）
        self._write_active()  # 啟動即寫一次當前活躍快照（即使無持倉也立刻產生檔案）
        print(f"✅ 混合交易初始化 | {self.symbol} | 趨勢 {TREND_TF} + 諧波 {HARM_TF}(N={HARM_PIVOT_N}) "
              f"| {'🟡 DRY-RUN' if DRY_RUN else '🔴 實單'} | 目前狀態: {_active_label(self.active)}")
        print(f"   📒 日誌：logs/mixed_trades.csv（交易事件流）｜ logs/mixed_active.csv（當前活躍快照）")

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

    def _total_equity(self):
        """總權益（供回撤熔斷 HWM/drawdown 用，非 sizing）。Bybit unified 取 totalEquity，
        備援 ccxt 統一 total["USDT"]。DRY_RUN 用模擬資金。讀不到合理值回 0.0（呼叫端 fail-closed）。"""
        if DRY_RUN:
            return DRY_RUN_SIM_CAPITAL
        eq = 0.0
        try:
            bal = self.ex.fetch_balance({"accountType": "UNIFIED"})
            try:
                eq = float(bal.get("info", {}).get("result", {}).get("list", [{}])[0].get("totalEquity", 0) or 0)
            except Exception:
                eq = 0.0
            if eq < MIN_PLAUSIBLE_EQUITY_USDT:  # 備援：ccxt 統一欄位
                eq = float(bal.get("total", {}).get("USDT", 0) or 0)
        except Exception as e:
            print(f"查詢總權益失敗: {e}")
            eq = 0.0
        if eq >= MIN_PLAUSIBLE_EQUITY_USDT:
            self._last_good_equity = (time.time(), eq)
            return eq
        # 讀取失敗/壞讀：沿用近期 last-good（≤15 分），避免瞬斷誤觸 fail-closed（與 _btc_funding_3d 同模式）
        if self._last_good_equity is not None and (time.time() - self._last_good_equity[0]) <= 900:
            return self._last_good_equity[1]
        return 0.0

    def _drawdown_scale(self):
        """回撤熔斷係數：讀總權益→更新高水位→算回撤→回 circuit_breaker_scale。
        讀不到合理權益 → 回 0.0（fail-closed，擋下新進場，不在盲區照常開倉）。"""
        if not CIRCUIT_BREAKER:
            return 1.0
        eq = self._total_equity()
        if eq < MIN_PLAUSIBLE_EQUITY_USDT:
            print("🚨 [熔斷] 無法取得合理總權益，本次新進場暫停（fail-closed）。")
            return 0.0
        if eq > self.equity_peak:   # 只用合理讀數升高 HWM，瞬態低讀不毒化
            self.equity_peak = eq
        dd = (self.equity_peak - eq) / self.equity_peak if self.equity_peak > 0 else 0.0
        if dd > MAX_PLAUSIBLE_DRAWDOWN:  # 異常深回撤視為壞讀
            print(f"🚨 [熔斷] 回撤 {dd:.1%} 超過合理上限，視為壞讀，本次新進場暫停（fail-closed）。")
            return 0.0
        cb = circuit_breaker_scale(dd, CIRCUIT_BREAKER)
        if cb < 1.0:
            print(f"🛡️ [熔斷] 權益回撤 {dd:.1%}（高水位 {self.equity_peak:.0f}）→ 新倉係數 x{cb:.2f}")
        return cb

    def _refresh_circuit_breaker(self):
        """每小時更新權益高水位；若達熔斷暫停級且有未成交諧波掛單，撤單（避免熔斷期間成交）。"""
        if DRY_RUN or not CIRCUIT_BREAKER:
            return
        eq = self._total_equity()
        if eq < MIN_PLAUSIBLE_EQUITY_USDT:
            return
        if eq > self.equity_peak:
            self.equity_peak = eq
            self.save_state()  # B9：HWM 升高即持久化，避免崩潰丟失高水位
        dd = (self.equity_peak - eq) / self.equity_peak if self.equity_peak > 0 else 0.0
        if dd <= MAX_PLAUSIBLE_DRAWDOWN and circuit_breaker_scale(dd, CIRCUIT_BREAKER) <= 0.0 \
                and self.active == "harm_pending":
            print(f"🛡️ [熔斷] 回撤 {dd:.1%} 達暫停級，撤銷未成交諧波掛單。")
            self._cancel_harm("熔斷暫停")

    def _btc_funding_3d(self):
        """BTC 幣本位 3 日平滑資金費率，委派主程式共用實作（含重試 + last-good 快取）。

        單一事實來源：費率讀取的重試/快取/旁路語意三入口一致；用公開行情連線
        （與認證交易連線分離，不爭用 rate budget）。
        """
        return fetch_btc_funding_3d()

    def _cvd_tilt(self):
        """趨勢腿 CVD 背離降風險係數：近 CVD_TILT_LOOKBACK 根正規化 CVD 動能 (Σdelta/Σvol) < 0
        ＝空虛上漲 → 回 CVD_TILT_FACTOR(0.5)；否則 1.0。CVD 取 Binance 期貨 taker 量
        （Bybit K 線不提供）。失敗沿用近期 last-good（12h 內），再不行回 1.0（fail-open 不調整）。
        只供趨勢腿使用，與回測 cvd_mix_eval 定義一致。"""
        global _LAST_GOOD_CVD
        if not CVD_TILT_ENABLED:
            return 1.0
        try:
            bsym = self.symbol.replace("/", "")  # ETH/USDT -> ETHUSDT
            k = requests.get(_BINANCE_FAPI, params={"symbol": bsym, "interval": TREND_TF,
                             "limit": CVD_TILT_LOOKBACK + 5}, timeout=15).json()
            if isinstance(k, list) and len(k) >= CVD_TILT_LOOKBACK + 1:
                kk = k[:-1][-CVD_TILT_LOOKBACK:]   # 丟最後一根未收完的，取已收完的 N 根
                vol = sum(float(x[5]) for x in kk)
                delta = sum(2 * float(x[9]) - float(x[5]) for x in kk)  # x[9]=taker buy base
                if vol > 0:
                    mom = delta / vol
                    _LAST_GOOD_CVD = (time.time(), mom)
                    return CVD_TILT_FACTOR if mom < 0 else 1.0
            print("⚠️ CVD K 線筆數不足，本次趨勢開倉跳過 CVD 微調。")
        except Exception as e:
            print(f"⚠️ 讀取 CVD 失敗（{e}），本次趨勢開倉跳過 CVD 微調。")
        if _LAST_GOOD_CVD is not None and (time.time() - _LAST_GOOD_CVD[0]) / 3600 <= 12:
            age_h = (time.time() - _LAST_GOOD_CVD[0]) / 3600
            print(f"⚠️ CVD 取得失敗，沿用 {age_h:.1f}h 前的快取值。")
            return CVD_TILT_FACTOR if _LAST_GOOD_CVD[1] < 0 else 1.0
        return 1.0

    def _check_position_mode(self):
        """實單啟動前確認帳戶為單向（one-way）持倉。

        本程式所有下單寫死 positionIdx=0，hedge mode 下會被 Bybit 拒單；
        與其在第一筆訊號時才爆炸，不如啟動就擋下（對齊主程式的模式偵測）。
        """
        if DRY_RUN:
            return True
        try:
            if hasattr(self.ex, "fetch_position_mode"):
                mode = self.ex.fetch_position_mode(self.symbol, {"category": "linear"})
                if mode.get("hedged"):
                    print("🚨 Bybit 帳戶為對沖(hedge)持倉模式，本程式僅支援單向(one-way)：請先在 Bybit 切換後再啟動。")
                    return False
        except Exception as e:
            print(f"⚠️ 無法確認持倉模式（{e}），請自行確認 Bybit 為單向(one-way)模式。")
        return True

    def _exchange_leverage(self):
        """讀取交易所目前槓桿（對齊主程式 _get_exchange_leverage）；讀不到回 None。"""
        try:
            if hasattr(self.ex, "fetch_leverage"):
                info = self.ex.fetch_leverage(self.symbol, {"category": "linear"})
                vals = [float(v) for v in (info.get("longLeverage"), info.get("shortLeverage"))
                        if v not in (None, "", 0, "0")]
                if vals:
                    return min(vals)
        except Exception:
            pass
        try:
            for pos in self.ex.private_get_v5_position_list(
                    {"category": "linear", "symbol": self.bybit_symbol}).get("result", {}).get("list", []):
                lv = pos.get("leverage")
                if lv not in (None, "", "0", 0):
                    return float(lv)
        except Exception:
            pass
        return None

    def _ensure_leverage_capacity(self):
        """實單開倉前確認交易所槓桿 >= 程式名目槓桿 LEV，不足則取消本次開倉。

        vol targeting 係數上限 2.0 時名目曝險最高 = QTY_PCT×LEV×2 ≈ 1.2x 權益，
        交易所槓桿不足會導致保證金不足拒單或部分成交（對齊主程式的容量檢查）。
        """
        if DRY_RUN:
            return True
        actual = self._exchange_leverage()
        if actual is None:
            print(f"⚠️ 無法讀取交易所槓桿，仍會下單；請自行確認 Bybit 槓桿 >= {LEV:.0f}x。")
            return True
        if actual >= LEV:
            return True
        print(f"🚨 Bybit 槓桿 {actual}x < 程式名目槓桿 {LEV:.0f}x，本次開倉取消。請調高交易所槓桿或降低 LEV。")
        return False

    def _calc_qty(self, price, recent_vol=None, btc_funding_3d=None, qty_pct=QTY_PCT, cvd_tilt=1.0):
        if not self._ensure_leverage_capacity():
            return 0.0
        free = self._free()
        if DRY_RUN and free < MIN_PLAUSIBLE_EQUITY_USDT:
            print(f"💡 DRY-RUN：可用餘額 {free:.2f} USDT 不足，以模擬資金 {DRY_RUN_SIM_CAPITAL:.0f} USDT 計算倉位。")
            free = DRY_RUN_SIM_CAPITAL
        scale = vol_target_scale(recent_vol, VOL_TARGET)
        if abs(scale - 1.0) > 1e-9:
            print(f"📊 波動度部位調整：近期波動 {float(recent_vol):.4f} → 倉位係數 x{scale:.2f}")
        # 深負費率加碼：與回測/主程式共用 apply_funding_boost（單一事實來源）
        boosted = apply_funding_boost(scale, btc_funding_3d, FUNDING_BOOST)
        if abs(boosted - scale) > 1e-9:
            print(f"🚀 [趨勢] 深負費率加碼：BTC 3日均費率 {float(btc_funding_3d)*100:+.4f}%/8h → 倉位係數 x{scale:.2f}→x{boosted:.2f}")
            scale = boosted
        # CVD 背離降風險微調（只趨勢腿傳 <1，諧波/抄底腿恆 1.0 不受影響）
        if abs(cvd_tilt - 1.0) > 1e-9:
            print(f"🛡️ [趨勢] CVD 背離降風險：空虛上漲(價漲但缺主動買盤) → 倉位係數 x{scale:.2f}→x{scale*cvd_tilt:.2f}")
            scale *= cvd_tilt
        # B9：回撤熔斷（portfolio 風控，最終係數；halt→0→qty 0 跳過進場）。趨勢/抄底/諧波三腿皆經此。
        scale *= self._drawdown_scale()
        notional = free * qty_pct / 100 * LEV * scale
        qty = self._fmt_amt(notional / price)
        mn = self.ex.market(self.symbol)["limits"]["amount"]["min"] or 0.001
        return qty if qty >= mn else 0.0

    def _exch_pos_size(self):
        if DRY_RUN:
            return 0.0
        try:
            if not hasattr(self.ex, "private_get_v5_position_list"):
                return None
            for pos in self.ex.private_get_v5_position_list(
                    {"category": "linear", "symbol": self.bybit_symbol}).get("result", {}).get("list", []):
                sz = float(pos.get("size", 0) or 0)
                if sz > 0:
                    return sz if pos.get("side") == "Buy" else -sz
            return 0.0
        except Exception as e:
            print(f"查持倉失敗: {e}")
            return None

    def _exch_pos_entry(self, expected_side=None):
        """讀取交易所該標的目前持倉的成交均價（avgPrice）。單一部位設計下用於校正進場價。"""
        if DRY_RUN:
            return None
        try:
            if not hasattr(self.ex, "private_get_v5_position_list"):
                return None
            for pos in self.ex.private_get_v5_position_list(
                    {"category": "linear", "symbol": self.bybit_symbol}).get("result", {}).get("list", []):
                if float(pos.get("size", 0) or 0) <= 0:
                    continue
                if expected_side and pos.get("side") != expected_side:
                    continue
                avg = pos.get("avgPrice") or pos.get("entryPrice")
                return float(avg) if avg not in (None, "", "0", 0) else None
            return None
        except Exception as e:
            print(f"查詢持倉均價失敗: {e}")
            return None

    def _confirmed_entry(self, order, side, fallback_entry):
        """市價單送出後，以交易所實際成交均價校正進場價：優先讀部位 avgPrice，
        備援 fetch_order().average；3 次輪詢仍無則回退訊號收盤價並告警。"""
        if DRY_RUN or not order or order.get("id") == "DRY":
            return fallback_entry
        order_id = order.get("id")
        expected_side = "Buy" if side == "buy" else "Sell"
        for attempt in range(1, 4):
            avg = self._exch_pos_entry(expected_side)
            if avg and avg > 0:
                return avg
            if order_id:
                try:
                    fresh = self.ex.fetch_order(order_id, self.symbol, {"category": "linear"})
                    avg = fresh.get("average") or fresh.get("info", {}).get("avgPrice")
                    if avg and float(avg) > 0:
                        return float(avg)
                except Exception as e:
                    print(f"查詢成交均價失敗({attempt}/3): {e}")
            if attempt < 3:
                time.sleep(1)
        print(f"⚠️ 無法確認成交均價，暫用訊號收盤價 {fallback_entry:.2f}（移動停損基準可能略偏）。")
        return fallback_entry

    def _set_exchange_stop_loss(self, stop_price, reason):
        if DRY_RUN:
            return True
        if not stop_price or stop_price <= 0:
            return False
        if not hasattr(self.ex, "private_post_v5_position_trading_stop"):
            print("⚠️ 目前 ccxt 版本不支援 Bybit trading-stop，無法同步交易所端停損。")
            return False

        params = {
            "category": "linear",
            "symbol": self.bybit_symbol,
            "tpslMode": "Full",
            "positionIdx": POSITION_IDX,
            "stopLoss": self._fmt_px(stop_price),
            "slTriggerBy": STOP_TRIGGER_BY,
            "slOrderType": "Market",
        }
        for attempt in range(1, STOP_LOSS_SYNC_RETRIES + 1):
            try:
                resp = self.ex.private_post_v5_position_trading_stop(params)
                if resp.get("retCode") in (0, "0"):
                    print(f"🛡️ [趨勢] 已同步交易所端停損 ({reason}) @ {params['stopLoss']}")
                    return True
                print(f"⚠️ 同步交易所端停損失敗 ({attempt}/{STOP_LOSS_SYNC_RETRIES}): {resp}")
            except Exception as e:
                print(f"⚠️ 同步交易所端停損失敗 ({attempt}/{STOP_LOSS_SYNC_RETRIES}): {e}")
            if attempt < STOP_LOSS_SYNC_RETRIES:
                time.sleep(STOP_LOSS_SYNC_RETRY_DELAY_SECONDS)
        print("🚨 交易所端停損同步連續失敗，請人工確認 Bybit 保護停損。")
        return False

    def _sync_with_exchange(self, reason="同步"):
        if DRY_RUN:
            return
        recon = self._reconcile_untracked_orders(reason)  # True=已確認清乾淨 / None=讀取失敗 / False=撤單失敗已暫停
        if recon is False:  # B5：撤孤兒單失敗已暫停，停在此別讓下方清掉 external_pos
            return
        pos_size = self._exch_pos_size()
        if pos_size is None:
            print(f"⚠️ [{reason}] 無法確認交易所持倉，保留本地狀態並暫停同步決策。")
            return
        if pos_size == 0:
            if self.active == "external_pos":
                if recon is not True:  # 開單讀取失敗,無法確認孤兒單已清 → 維持暫停,別誤解除
                    return
                print(f"✅ [{reason}] 交易所已無外部持倉，回到等待。")
                self.active = "none"
                self.save_state()
            elif self.active == "harm_pos":
                print(f"✅ [{reason}] 諧波持倉已不在交易所，清除本地狀態。")
                self.harm = None
                self.active = "none"
                self.save_state()
            elif self.active == "trend" and self.trend is not None:
                print(f"✅ [{reason}] 趨勢持倉已不在交易所，清除本地狀態。")
                self.trend = None
                self.active = "none"
                self.save_state()
            elif self.active == "dip_pos" and self.dip is not None:
                print(f"✅ [{reason}] 抄底持倉已不在交易所（可能人工平倉），清除本地狀態。")
                self.dip = None
                self.active = "none"
                self.save_state()
            return

        if self.active == "harm_pending" and self.harm is not None:
            print(f"✅ [{reason}] 偵測到諧波掛單已成交，轉為持倉狀態。")
            log_event(LOG_FILE, strategy="harmonic", timeframe=HARM_TF, action="fill",
                      side="buy" if self.harm["bull"] else "sell", pattern=self.harm["pattern"],
                      price=self.harm["entry"], qty=abs(pos_size), sl=round(self.harm["sl"], 2),
                      tp=round(self.harm["tp"], 2), dry=False)
            self.harm["qty"] = abs(pos_size)
            self.active = "harm_pos"
            self.save_state()
            return

        if self.active == "trend" and self.trend is not None:
            self.trend.qty = abs(pos_size)
            stop = self.trend.trail_stop if self.trend.trail_active and self.trend.trail_stop else (
                self.trend.entry_price * (1 - P["long_fixed_stop_loss_percent"])
            )
            self._set_exchange_stop_loss(stop, reason)
            return

        if self.active == "dip_pos" and self.dip is not None:
            self.dip["qty"] = abs(pos_size)  # 抄底腿設計上無交易所端停損，僅校正數量
            return

        if self.active in ("none", "harm_pending"):
            print(f"⚠️ [{reason}] 偵測到交易所已有未記錄持倉 {pos_size}，暫停新訊號直到人工確認或持倉歸零。")
            self.active = "external_pos"
            self.save_state()

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
        """市價平倉。回傳 (是否成功, 實際成交均價或 None)；均價讀不到時呼叫端
        退回理論價記錄 PnL。DRY-RUN 恆 (True, None)。"""
        if DRY_RUN:
            return True, None
        pos_size = self._exch_pos_size()
        if pos_size not in (None, 0):
            qty = abs(pos_size)
        elif self.trend:
            qty = abs(self.trend.qty)
        elif self.dip:
            qty = abs(self.dip["qty"])
        else:
            qty = 0
        if qty <= 0:
            print("⚠️ 平倉前查無交易所持倉，保留本地狀態等待下次同步。")
            return False, None
        try:
            order = self.ex.create_order(self.symbol, "market", side, qty, None,
                                         {"category": "linear", "positionIdx": POSITION_IDX, "reduceOnly": True})
            time.sleep(2)
            final_pos = self._exch_pos_size()
        except Exception as e:
            print(f"❌ 平倉失敗: {e}")
            return False, None
        if final_pos is None:
            print("⚠️ 平倉後無法確認交易所持倉，保留本地狀態等待下次同步。")
            return False, None
        if final_pos != 0:
            print("⚠️ 平倉後仍偵測到交易所持倉，請人工確認。")
            return False, None
        # 成敗判定已完成才讀成交均價：均價查詢的任何未來失誤都不得把「已平倉」誤報為失敗
        return True, self._exit_fill_price(order)

    def _exit_fill_price(self, order):
        """讀取平倉市價單的實際成交均價；讀不到回 None（呼叫端退回理論價記錄）。
        出場時部位已歸零，無法像進場那樣讀部位 avgPrice，只能查訂單本身。"""
        oid = (order or {}).get("id")
        if not oid:
            return None
        try:
            fresh = self.ex.fetch_order(oid, self.symbol, {"category": "linear"})
            avg = fresh.get("average") or fresh.get("info", {}).get("avgPrice")
            return float(avg) if avg and float(avg) > 0 else None
        except Exception as e:
            print(f"⚠️ 查詢平倉成交均價失敗，PnL 記錄改用理論價: {e}")
            return None

    def _cancel(self, oid):
        if DRY_RUN or not oid or oid == "DRY":
            return True
        for attempt in range(2):  # B19：網路瞬斷重試一次,避免卡在 harm_pending
            try:
                self.ex.cancel_order(oid, self.symbol, {"category": "linear"})
                return True
            except Exception as e:
                if attempt == 0:
                    time.sleep(1)
                    continue
                print(f"撤單失敗(可能已成交): {e}")
        return False

    def _reconcile_untracked_orders(self, reason):
        """帳戶為本系統專用(owner 確認不手動下單)：撤掉任何「非當前追蹤」的掛單,
        清掉崩潰/重啟殘留的孤兒限價單。對帳只看部位無法偵測這些單,故另外列舉掛單。
        回傳 None=讀取失敗略過;False=撤單失敗已暫停;True=無孤兒或已清乾淨。"""
        if DRY_RUN:
            return True
        try:
            orders = self.ex.fetch_open_orders(self.symbol, None, None, {"category": "linear"})
        except Exception as e:
            print(f"⚠️ [{reason}] 讀取掛單失敗，跳過孤兒單對帳: {e}")
            return None
        tracked = str((self.harm or {}).get("order_id") or "")
        ok = True
        for o in orders or []:
            oid = str(o.get("id") or "")
            info = o.get("info", {}) or {}
            reduce_only = str(o.get("reduceOnly", info.get("reduceOnly", False))).lower() == "true"
            # 只撤「純限價進場單」：跳過保護性/條件單。注意 Bybit V5 普通單的 stopOrderType 常為
            # "" 或 "UNKNOWN"（truthy 但非條件單），故須比對「已知條件型別」而非單純 truthy，
            # 否則會誤判每張單都是保護單 → B5 完全不撤孤兒單（codex 後審抓到）。
            stop_type = str(info.get("stopOrderType", "")).strip().lower()
            close_on_trigger = str(info.get("closeOnTrigger", "")).lower() == "true"
            conditional = (stop_type not in ("", "unknown", "none")
                           or bool(o.get("triggerPrice") or o.get("stopPrice") or info.get("triggerPrice"))
                           or close_on_trigger
                           or str(o.get("type", "")).lower() not in ("", "limit"))
            if not oid or reduce_only or conditional or (tracked and oid == tracked):
                continue
            print(f"⚠️ [{reason}] 偵測到非本系統追蹤的掛單 {oid}，撤單(帳戶為 bot 專用)。")
            if self._cancel(oid):
                log_event(LOG_FILE, strategy="reconcile", timeframe="", action="cancel",
                          reason="orphan_order", dry=False, order_id=oid)
            else:
                print("🚨 孤兒掛單撤單失敗，暫停新訊號並等待人工確認。")
                self.active = "external_pos"
                self.save_state()
                ok = False
        return ok

    # ---------- 趨勢（4h，重用回測邏輯） ----------
    def _open_trend(self, bar):
        entry = float(bar["close"])
        # 與回測一致：用訊號 bar 的 ret_vol / btc_funding_3d 做倉位調整，並套 CVD 背離降風險（諧波/抄底腿不適用）
        qty = self._calc_qty(entry, recent_vol=bar.get("ret_vol"),
                             btc_funding_3d=bar.get("btc_funding_3d"),
                             cvd_tilt=self._cvd_tilt())
        if qty <= 0:
            print("資金不足，趨勢進場略過。")
            return
        sl = entry * (1 - P["long_fixed_stop_loss_percent"])
        print(f"🟢 [趨勢] 市價做多 {qty} @ ~{entry:.2f} | 交易所固定SL {sl:.2f}")
        order = self._market("buy", qty, sl=sl)
        if order is None:
            return
        # B3：以交易所實際成交均價校正 entry/qty/SL（市價單可能偏離 bar.close）
        entry = self._confirmed_entry(order, "buy", entry)
        filled = self._exch_pos_size()
        if not DRY_RUN and not filled:  # 實單未確認到持倉 → 不建本地部位(避免幻影),暫停待人工確認
            print("🚨 [趨勢] 市價單後未確認到持倉，暫停新訊號待人工確認。")
            self.active = "external_pos"; self.save_state(); return
        if filled:
            qty = abs(filled)
        sl = entry * (1 - P["long_fixed_stop_loss_percent"])
        self._set_exchange_stop_loss(sl, "趨勢成交均價校正")
        self.trend = Position(side="long", qty=qty, entry_price=entry, entry_time=bar.name, entry_fee=0.0, peak=entry)
        self.active = "trend"
        log_event(LOG_FILE, strategy="trend", timeframe=TREND_TF, action="entry", side="long",
                  price=round(entry, 2), qty=qty, sl=round(sl, 2), dry=DRY_RUN, order_id=order.get("id", ""))
        self.save_state()

    def _manage_trend(self, bar):
        if short_signal(bar, P):
            print("🔴 [趨勢] 反向訊號 → 平多")
            ok, fill = self._close_market("sell")
            if not ok:
                return
            px = fill or float(bar["close"])  # 實際成交均價優先，讀不到退回訊號收盤價
            pnl = (px - self.trend.entry_price) * self.trend.qty
            log_event(LOG_FILE, strategy="trend", timeframe=TREND_TF, action="exit", side="sell",
                      price=round(px, 2), qty=self.trend.qty, reason="reverse", pnl=round(pnl, 2), dry=DRY_RUN)
            self.trend = None; self.active = "none"; self.save_state(); return
        st = check_stop_exit(self.trend, bar, P)
        if st:
            print(f"🔴 [趨勢] {st[0]} 觸發 @ {st[1]:.2f} → 平多")
            ok, fill = self._close_market("sell")
            if not ok:
                return
            px = fill or st[1]  # 實際成交均價優先，讀不到退回理論停損價
            pnl = (px - self.trend.entry_price) * self.trend.qty
            log_event(LOG_FILE, strategy="trend", timeframe=TREND_TF, action="exit", side="sell",
                      price=round(px, 2), qty=self.trend.qty, reason=st[0], pnl=round(pnl, 2), dry=DRY_RUN)
            self.trend = None; self.active = "none"; self.save_state(); return
        old_trail = self.trend.trail_stop
        update_trailing_stop(self.trend, bar, P)
        self.trend.bars_held += 1
        if self.trend.trail_active and self.trend.trail_stop:
            print(f"   [趨勢] 移動停損更新 → {self.trend.trail_stop:.2f}")
            if old_trail != self.trend.trail_stop:
                self._set_exchange_stop_loss(self.trend.trail_stop, "趨勢移動停損更新")

    # ---------- 抄底腿（4h，深負費率事件） ----------
    def _open_dip(self, bar):
        entry = float(bar["close"])
        # 半倉 + 波動度係數；刻意不吃費率加碼（回測驗證的 sizing 不含 boost）
        qty = self._calc_qty(entry, recent_vol=bar.get("ret_vol"), qty_pct=DIP_QTY_PCT)
        if qty <= 0:
            print("🚨 [抄底] 觸發深負費率事件，但資金不足最小下單量，本輪作廢。")
            log_event(LOG_FILE, strategy="dip", timeframe=TREND_TF, action="skip",
                      side="buy", reason="insufficient_funds", dry=DRY_RUN)
            return
        f3 = bar.get("btc_funding_3d")
        print(f"🟢 [抄底] BTC 3日均費率 {f3*100:+.4f}%/8h 下穿 {DIP_FUNDING_THRESHOLD*100:.2f}% → "
              f"市價做多 {qty} @ ~{entry:.2f}（{DIP_HOLD_BARS} 根/30天時間出場，無停損）")
        order = self._market("buy", qty)  # 設計上無 SL：災難停損實測會精準砍在擠壓前低點
        if order is None:
            return
        # B3：以交易所實際成交均價校正 entry/qty（抄底腿設計上無 SL，不重算停損）
        entry = self._confirmed_entry(order, "buy", entry)
        filled = self._exch_pos_size()
        if not DRY_RUN and not filled:  # 實單未確認到持倉 → 不建本地部位(避免幻影),暫停待人工確認
            print("🚨 [抄底] 市價單後未確認到持倉，暫停新訊號待人工確認。")
            self.active = "external_pos"; self.save_state(); return
        if filled:
            qty = abs(filled)
        self.dip = {"entry": entry, "qty": qty, "entry_time": str(bar.name)}
        self.active = "dip_pos"
        log_event(LOG_FILE, strategy="dip", timeframe=TREND_TF, action="entry", side="long",
                  price=round(entry, 2), qty=qty, dry=DRY_RUN, order_id=order.get("id", ""),
                  note=f"funding {f3*100:+.4f}%/8h")
        self.save_state()

    def _manage_dip(self, bar):
        held = (bar.name - pd.Timestamp(self.dip["entry_time"])) / timeframe_to_timedelta(TREND_TF)
        if held < DIP_HOLD_BARS:
            return
        px = float(bar["close"])
        pnl = (px - self.dip["entry"]) * self.dip["qty"]
        print(f"🔴 [抄底] 持有滿 {DIP_HOLD_BARS} 根（30天）→ 市價平多 @ ~{px:.2f}（約{pnl:+.2f} USDT）")
        ok, fill = self._close_market("sell")
        if not ok:
            return
        if fill:  # 實際成交均價優先，讀不到退回訊號收盤價
            px = fill
            pnl = (px - self.dip["entry"]) * self.dip["qty"]
        log_event(LOG_FILE, strategy="dip", timeframe=TREND_TF, action="exit", side="sell",
                  price=round(px, 2), qty=self.dip["qty"], reason="time", pnl=round(pnl, 2), dry=DRY_RUN)
        self.dip = None
        self.active = "none"
        self.save_state()

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
        # dedup key 必須用 C 樞軸的「時間戳」：C[1] 是滾動視窗內的位置索引，
        # 每根新 K 線都會位移，拿來去重會誤判（同型態重複掛單／不同型態被誤擋）
        return {"entry": entry, "sl": sl, "tp": tp, "bull": bull, "pattern": name,
                "x_price": X[2], "c_time": str(df.index[C[1]])}

    def _place_harm(self, sig, placed_time=None):
        entry = float(self._fmt_px(sig["entry"]))
        qty = self._calc_qty(entry)
        if qty <= 0:
            print(f"🚨 [諧波] 偵測到 {sig['pattern']} 訊號，但資金不足最小下單量，略過（訊號不視為已處理）。")
            log_event(LOG_FILE, strategy="harmonic", timeframe=HARM_TF, action="skip",
                      side="buy" if sig["bull"] else "sell", pattern=sig["pattern"],
                      price=entry, reason="insufficient_funds", dry=DRY_RUN)
            return False
        side = "buy" if sig["bull"] else "sell"
        print(f"🎯 [諧波] {sig['pattern']} {'多' if sig['bull'] else '空'} 掛限價 {side} {qty} @ {entry} "
              f"| SL {sig['sl']:.2f} TP {sig['tp']:.2f}")
        order = self._limit(side, qty, entry, sig["sl"], sig["tp"])
        self.harm = {**sig, "order_id": order.get("id"), "qty": qty, "placed": str(placed_time) if placed_time is not None else ""}
        self.active = "harm_pending"
        log_event(LOG_FILE, strategy="harmonic", timeframe=HARM_TF, action="place", side=side,
                  pattern=sig["pattern"], price=entry, qty=qty, sl=round(sig["sl"], 2), tp=round(sig["tp"], 2),
                  dry=DRY_RUN, order_id=order.get("id", ""))
        self.save_state()
        return True

    def _cancel_harm(self, reason):
        if not self.harm:
            return True
        print(f"🗑️ [諧波] 撤掛單（{reason}）")
        cancelled = self._cancel(self.harm.get("order_id"))
        if not cancelled:
            pos_size = self._exch_pos_size()
            if pos_size is not None and pos_size != 0:
                print("⚠️ 撤單失敗且偵測到持倉，改列為諧波持倉，避免重複開倉。")
                log_event(LOG_FILE, strategy="harmonic", timeframe=HARM_TF, action="fill",
                          side="buy" if self.harm["bull"] else "sell", pattern=self.harm["pattern"],
                          price=self.harm["entry"], qty=abs(pos_size), sl=round(self.harm["sl"], 2),
                          tp=round(self.harm["tp"], 2), dry=False)
                self.harm["qty"] = abs(pos_size)
                self.active = "harm_pos"
                self.save_state()
                return False
            print("⚠️ 撤單結果不明，保留掛單狀態，避免誤開新倉。")
            self.save_state()
            return False
        log_event(LOG_FILE, strategy="harmonic", timeframe=HARM_TF, action="cancel",
                  pattern=self.harm.get("pattern", ""), reason=reason, dry=DRY_RUN,
                  order_id=self.harm.get("order_id", ""))
        self.harm = None
        self.active = "none"
        self.save_state()
        return True

    def _manage_harm_pending(self, df):
        pos_size = self._exch_pos_size()
        if pos_size is None:
            print("⚠️ 無法確認諧波掛單是否成交，保留掛單狀態。")
            return
        if pos_size != 0 or (DRY_RUN and self._dry_harm_filled(df)):
            print("✅ [諧波] 限價單成交 → 持倉（SL/TP 由交易所端執行）")
            log_event(LOG_FILE, strategy="harmonic", timeframe=HARM_TF, action="fill",
                      side="buy" if self.harm["bull"] else "sell", pattern=self.harm["pattern"],
                      price=self.harm["entry"], qty=self.harm["qty"], sl=round(self.harm["sl"], 2),
                      tp=round(self.harm["tp"], 2), dry=DRY_RUN)
            self.active = "harm_pos"; self.save_state(); return
        close = float(df["close"].iloc[-1]); bull = self.harm["bull"]
        if (bull and close < self.harm["x_price"]) or (not bull and close > self.harm["x_price"]):
            self._cancel_harm("價格突破 X，型態失效")
        elif self.harm.get("placed") and (
            df.index[-1] - pd.Timestamp(self.harm["placed"])
        ) > timeframe_to_timedelta(HARM_TF) * HARM_WAIT:
            self._cancel_harm(f"超過 {HARM_WAIT} 根未成交")

    def _dry_harm_filled(self, df):
        """DRY-RUN 模擬：最新 1h K 線是否觸及限價。"""
        bull = self.harm["bull"]; hi, lo = float(df["high"].iloc[-1]), float(df["low"].iloc[-1])
        return (lo <= self.harm["entry"]) if bull else (hi >= self.harm["entry"])

    def _monitor_harm(self, df):
        if not DRY_RUN:
            pos_size = self._exch_pos_size()
            if pos_size is None:
                print("⚠️ 無法確認諧波持倉狀態，保留本地狀態。")
                return
            if pos_size == 0:   # 實單：交易所端 SL/TP 已平倉
                print("✅ [諧波] 持倉已由交易所 SL/TP 平倉 → 回到等待")
                log_event(LOG_FILE, strategy="harmonic", timeframe=HARM_TF, action="exit",
                          pattern=self.harm.get("pattern", "") if self.harm else "",
                          reason="exchange_sl_tp", dry=False)
                self.harm = None; self.active = "none"; self.save_state()
            return
        # DRY-RUN：用最新 K 線模擬交易所端 SL/TP 觸發（否則持倉會卡住不出場）
        hi, lo, bull = float(df["high"].iloc[-1]), float(df["low"].iloc[-1]), self.harm["bull"]
        if bull:
            hit = "SL" if lo <= self.harm["sl"] else ("TP" if hi >= self.harm["tp"] else None)
        else:
            hit = "SL" if hi >= self.harm["sl"] else ("TP" if lo <= self.harm["tp"] else None)
        if hit:
            px = self.harm["sl"] if hit == "SL" else self.harm["tp"]
            pnl = ((px - self.harm["entry"]) if bull else (self.harm["entry"] - px)) * self.harm["qty"]
            print(f"✅ [諧波] (DRY) {hit} 觸及 @ {px:.2f} → 平倉")
            log_event(LOG_FILE, strategy="harmonic", timeframe=HARM_TF, action="exit",
                      side="sell" if bull else "buy", pattern=self.harm["pattern"], price=round(px, 2),
                      qty=self.harm["qty"], reason=hit, pnl=round(pnl, 2), dry=True)
            self.harm = None; self.active = "none"; self.save_state()

    # ---------- 協調 ----------
    def on_4h(self, bar):
        # 抄底腿武裝管理：費率回到閾值上方 → 重新武裝；下穿且武裝中 → 觸發一次
        f3 = bar.get("btc_funding_3d")
        has_f = f3 is not None and not pd.isna(f3)
        if has_f and f3 >= DIP_FUNDING_THRESHOLD:
            self.dip_armed = True
        dip_trigger = DIP_ENABLED and self.dip_armed and has_f and f3 < DIP_FUNDING_THRESHOLD
        if dip_trigger:
            self.dip_armed = False  # 同一輪深負期間只觸發一次；倉位被佔用即作廢（與回測一致）

        if self.active == "trend":
            self._manage_trend(bar)
        elif self.active == "dip_pos":
            self._manage_dip(bar)
        elif self.active in ("none", "harm_pending"):
            if long_signal(bar, P):
                if self.active == "harm_pending":
                    if not self._cancel_harm("趨勢優先"):
                        return
                self._open_trend(bar)
            elif dip_trigger:
                if self.active == "harm_pending":
                    if not self._cancel_harm("深負費率抄底進場"):
                        return
                self._open_dip(bar)
        # harm_pos: 趨勢/抄底讓步等諧波結束

    def on_1h(self, bar, df):
        if self.active in ("trend", "dip_pos"):
            return  # 趨勢/抄底持倉中，諧波讓步（互斥）
        if self.active == "harm_pending":
            self._manage_harm_pending(df)
        elif self.active == "harm_pos":
            self._monitor_harm(df)
        elif self.active == "none":
            # 只有 _detect_harm 吃「已收完 K 線」切片：形成中 K 線的 close/high/low 會持續變動，
            # 拿來當樞軸確認 bar 或趨勢過濾（close vs ema200）會 repaint、且與回測
            # detect_opportunities（只看已收完 bar）分歧。掛限價單本就等回踩，晚 0-59 秒
            # 偵測不影響成交。_manage_harm_pending / _monitor_harm / _dry_harm_filled
            # 維持吃完整 df——那些刻意用盤中極值做撤單/成交/SLTP 模擬（議會 2026-07-02 核准）。
            sig = self._detect_harm(df[df.index <= bar.name])
            if sig and sig["c_time"] != self.last_signal_key:
                # 只有真的掛出單才消耗 dedup key，否則訊號會被無聲吞掉
                if self._place_harm(sig, bar.name):
                    self.last_signal_key = sig["c_time"]

    # ---------- 狀態 ----------
    def save_state(self):
        tp = None
        if self.trend:
            tp = {"side": self.trend.side, "qty": self.trend.qty, "entry_price": self.trend.entry_price,
                  "entry_time": str(self.trend.entry_time), "peak": self.trend.peak, "trough": self.trend.trough,
                  "trail_active": self.trend.trail_active, "trail_stop": self.trend.trail_stop,
                  "bars_held": self.trend.bars_held}
        st = {"active": self.active, "trend": tp, "harm": self.harm,
              "dip": self.dip, "dip_armed": self.dip_armed,
              "last_signal_key": self.last_signal_key,
              "equity_peak": self.equity_peak,
              "last_4h": str(self.last_4h) if self.last_4h is not None else None,
              "last_1h": str(self.last_1h) if self.last_1h is not None else None}
        try:
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(st, f, indent=2, default=str); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, STATE_FILE)
        except Exception as e:
            print(f"存檔失敗: {e}")
        self._write_active()

    def _write_active(self):
        """更新『當前活躍』快照：只列尚未停利/停損的掛單與持倉（互斥下 0~1 筆）。"""
        items = []
        if self.active == "trend" and self.trend is not None:
            if self.trend.trail_active and self.trend.trail_stop:
                sl, note = self.trend.trail_stop, "移動停損保護中"
            else:
                sl, note = self.trend.entry_price * (1 - P["long_fixed_stop_loss_percent"]), "固定停損"
            items.append({"status": "趨勢持倉中", "strategy": "trend", "side": "long", "pattern": "",
                          "entry": round(self.trend.entry_price, 2), "stop_loss": round(sl, 2),
                          "take_profit": "", "qty": self.trend.qty, "since": _tw(self.trend.entry_time), "note": note})
        elif self.active == "harm_pending" and self.harm is not None:
            items.append({"status": "諧波掛單等待成交", "strategy": "harmonic",
                          "side": "buy" if self.harm["bull"] else "sell", "pattern": self.harm["pattern"],
                          "entry": round(self.harm["entry"], 2), "stop_loss": round(self.harm["sl"], 2),
                          "take_profit": round(self.harm["tp"], 2), "qty": self.harm["qty"],
                          "since": _tw(self.harm["placed"]) if self.harm.get("placed") else "",
                          "note": "等價格觸及 PRZ"})
        elif self.active == "harm_pos" and self.harm is not None:
            items.append({"status": "諧波持倉中", "strategy": "harmonic",
                          "side": "buy" if self.harm["bull"] else "sell", "pattern": self.harm["pattern"],
                          "entry": round(self.harm["entry"], 2), "stop_loss": round(self.harm["sl"], 2),
                          "take_profit": round(self.harm["tp"], 2), "qty": self.harm["qty"],
                          "since": _tw(self.harm["placed"]) if self.harm.get("placed") else "",
                          "note": "等 SL/TP 觸發"})
        elif self.active == "dip_pos" and self.dip is not None:
            items.append({"status": "抄底持倉中", "strategy": "dip", "side": "long", "pattern": "",
                          "entry": round(self.dip["entry"], 2), "stop_loss": "無(設計如此)",
                          "take_profit": "", "qty": self.dip["qty"],
                          "since": _tw(self.dip["entry_time"]),
                          "note": f"深負費率事件，{DIP_HOLD_BARS}根(30天)時間出場"})
        elif self.active == "external_pos":
            items.append({"status": "交易所已有未記錄持倉", "strategy": "external",
                          "side": "", "pattern": "", "entry": "", "stop_loss": "",
                          "take_profit": "", "qty": "", "since": "", "note": "暫停新訊號，請人工確認"})
        write_active(ACTIVE_FILE, items)

    def load_state(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            s = json.load(open(STATE_FILE, encoding="utf-8"))
            self.active = s.get("active", "none"); self.harm = s.get("harm")
            self.dip = s.get("dip"); self.dip_armed = s.get("dip_armed", True)
            self.last_signal_key = s.get("last_signal_key")
            self.equity_peak = float(s.get("equity_peak", 0.0) or 0.0)
            self.last_4h = pd.to_datetime(s["last_4h"]) if s.get("last_4h") else None
            self.last_1h = pd.to_datetime(s["last_1h"]) if s.get("last_1h") else None
            tp = s.get("trend")
            if tp:
                self.trend = Position(side=tp["side"], qty=tp["qty"], entry_price=tp["entry_price"],
                                      entry_time=pd.to_datetime(tp["entry_time"]), entry_fee=0.0,
                                      peak=tp.get("peak"), trough=tp.get("trough"),
                                      trail_active=tp.get("trail_active", False),
                                      trail_stop=tp.get("trail_stop"), bars_held=tp.get("bars_held", 0))
            # 壞狀態防護：active 指向的持倉/掛單 payload 缺失（檔案毀損或手動編輯）時
            # 重設為 none，避免管理函式每根 K 線 AttributeError 崩潰循環且永不自癒。
            # 實單下若交易所仍有持倉，啟動後第一次每小時同步會偵測到未記錄持倉並
            # 轉 external_pos 暫停新訊號（fail-safe），不會憑空重複開倉。
            payload_by_active = {"trend": self.trend, "harm_pending": self.harm,
                                 "harm_pos": self.harm, "dip_pos": self.dip}
            if self.active in payload_by_active and payload_by_active[self.active] is None:
                print(f"⚠️ 狀態檔 active={self.active} 但對應資料缺失，重設為空倉等待（等待同步對帳）。")
                self.active = "none"
        except Exception as e:
            print(f"載入狀態失敗: {e}")

    # ---------- 主迴圈 ----------
    def run(self):
        print(f"--- 混合實盤啟動（互斥、趨勢優先）---  狀態: {_active_label(self.active)}")
        last = 0
        last_sync = 0
        fetch_fails = 0  # observability：連續行情抓取失敗計數（偵測 API 中斷），不參與任何交易決策
        while True:
            try:
                if not DRY_RUN and time.time() - last_sync >= 3600:
                    self._sync_with_exchange("每小時同步")
                    self._refresh_circuit_breaker()  # B9：更新權益高水位 + halt 時撤未成交諧波掛單
                    last_sync = time.time()
                if time.time() - last >= 60:
                    # limit=1000：300/500 根視窗的 EMA200 暖機殘差會產生回測沒有的幽靈訊號
                    raw4 = fetch_bybit_klines(SYMBOL, TREND_TF, limit=1000)
                    df4 = calculate_indicators(raw4)
                    b4 = get_latest_completed_bar(df4, TREND_TF)
                    if b4 is not None and (self.last_4h is None or b4.name > self.last_4h):
                        b4["btc_funding_3d"] = self._btc_funding_3d()  # None 時過濾自動旁路
                        f_msg = (f" | BTC費率(3d) {b4['btc_funding_3d']*100:+.4f}%"
                                 if b4["btc_funding_3d"] is not None else "")
                        _cl = _tw(b4.name + timeframe_to_timedelta(TREND_TF))  # 收盤時間（台灣），避免開盤標籤誤會
                        print(f"\n🔔 新 4h K 線 開 {_tw(b4.name)} → 收 {_cl} (台灣) | 價 {b4['close']:.2f} EMA200 {b4['ema200']:.2f}{f_msg} | 狀態 {_active_label(self.active)}")
                        self.last_4h = b4.name; self.on_4h(b4); self.save_state()

                    raw1 = fetch_bybit_klines(SYMBOL, HARM_TF, limit=1000)
                    df1 = calculate_indicators(raw1)
                    b1 = get_latest_completed_bar(df1, HARM_TF)
                    if b1 is not None and (self.last_1h is None or b1.name > self.last_1h):
                        self.last_1h = b1.name; self.on_1h(b1, df1); self.save_state()
                    # observability-only：偵測行情持續抓取失敗（API 中斷）；純告警，不影響任何交易決策
                    _empty_tfs = [tf for tf, raw in ((TREND_TF, raw4), (HARM_TF, raw1)) if raw.empty]
                    if _empty_tfs:
                        fetch_fails += 1
                        if fetch_fails in (10, 60, 180) or (fetch_fails > 180 and fetch_fails % 180 == 0):
                            print(f"🚨 行情抓取連續失敗約 {fetch_fails} 分鐘 | 失敗腿={','.join(_empty_tfs)} | {SYMBOL}，請檢查網路/Bybit。")
                    elif fetch_fails > 0:
                        print(f"✅ 行情恢復（中斷約 {fetch_fails} 分鐘）。")
                        fetch_fails = 0
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
