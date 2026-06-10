"""
🏆 ETH 4小時自動交易策略 - 實時交易版本 (移動停損優化版)
🎯 參數來源: 2020-2025年優化結果 (穩定性評分48.38, 獲利5371.25 USDT)
📊 策略特點: 95.45%季度獲利率, 54.74%平均勝率, 14.06%最大回撤
🔧 停損機制: Bybit交易所端保護停損 + 本地固定/移動停損輪詢
"""

import pandas as pd
from datetime import datetime, timedelta
import os
import sys
import time
import ccxt
from dotenv import load_dotenv
import json
import math

# 載入 .env 檔案中的環境變數
load_dotenv()


def _configure_utf8_output():
    """強制 stdout/stderr 使用 UTF-8 輸出。

    Windows 的 zh-TW 主控台預設為 cp950，無法編碼本程式大量使用的 emoji，
    會讓 print 直接拋出 UnicodeEncodeError 而中斷交易迴圈。以 errors="replace"
    確保任何輸出都不會讓程式崩潰。
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_utf8_output()

# --- 🔧 策略配置參數 (集中管理) ---
# 交易所設定
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

# 交易標的選擇：在 .env 設定 TRADE_SYMBOL=ETH/USDT 或 BTC/USDT。
# 每個標的使用各自獨立優化的最佳參數與熔斷 baseline（見下方對應字典）。
VALID_TRADE_SYMBOLS = {"ETH/USDT", "BTC/USDT"}
TRADE_SYMBOL = os.getenv("TRADE_SYMBOL", "ETH/USDT").upper().replace(" ", "")
if TRADE_SYMBOL not in VALID_TRADE_SYMBOLS:
    raise ValueError(
        f"TRADE_SYMBOL 必須是 {sorted(VALID_TRADE_SYMBOLS)}，目前是 {TRADE_SYMBOL!r}"
    )
SYMBOL = TRADE_SYMBOL
TIMEFRAME = "4h"

# 資金管理設定
DEFAULT_QTY_PERCENT = 30  # 每次交易使用帳戶可用餘額的百分比
TARGET_POSITION_LEVERAGE = 3  # 程式用來計算名目倉位，不會強制改交易所槓桿
AUTO_SET_EXCHANGE_LEVERAGE = False  # True 時會嘗試把交易所槓桿調到 TARGET_POSITION_LEVERAGE
REQUIRE_EXCHANGE_LEVERAGE_CAPACITY = True  # 交易所槓桿低於程式目標時，禁止新開倉
POSITION_IDX_ONE_WAY = 0  # Bybit one-way mode；若是 hedge mode，會從持倉資料自動讀取

# 風控設定
STOP_TRIGGER_BY = "MarkPrice"  # Bybit 支援 LastPrice / MarkPrice / IndexPrice
EXIT_ON_REVERSE_SIGNAL = True
ALLOW_SAME_BAR_REVERSAL = False
TRADE_SIDE_MODE = os.getenv("TRADE_SIDE_MODE", "long_only").lower()  # "long_only", "short_only", "both"
VALID_TRADE_SIDE_MODES = {"long_only", "short_only", "both"}
ENTRY_PRICE_SYNC_TOLERANCE_PCT = 0.0001  # 0.01%; exchange avgPrice drift above this triggers local sync
LOCAL_FIXED_STOP_FALLBACK_BUFFER_PCT = 0.002  # 0.2%; let exchange MarkPrice SL trigger first when price is near stop
STOP_LOSS_SYNC_RETRIES = 3
STOP_LOSS_SYNC_RETRY_DELAY_SECONDS = 2
KLINE_COMPLETION_BUFFER_SECONDS = 10
# Bybit/ccxt can transiently return a partial balance snapshot (near-zero equity)
# even on a funded account. Equity reads below this floor are treated as bad reads
# so they can't poison peak_capital / max_drawdown statistics.
MIN_PLAUSIBLE_EQUITY_USDT = 1.0
# A stop-protected strategy cannot realistically draw down this deep in one sample;
# a persisted/observed drawdown at/above this is treated as corrupted data.
MAX_PLAUSIBLE_DRAWDOWN = 0.95

# === 回撤熔斷 (drawdown circuit breaker) ===
# 目的：當實盤回撤超過策略「正常」回撤分布時，分級降低／暫停【開新倉】曝險。
#       既有倉位的固定／移動停損不受影響，平倉與停損永遠暢通。
#
# 科學依據（設計原理 + 來源）：
#   1) 分級降曝險而非一刀切 —— Grossman & Zhou (1993)
#      "Optimal Investment Strategies for Controlling Drawdowns" 證明：在最大回撤
#      約束下，最優曝險應隨「距 high-water mark 的回撤」遞減；CPPI (Black & Jones,
#      1987) 同理，曝險與緩衝墊 (equity − floor) 成正比。故採「警戒減碼→熔斷暫停」。
#   2) 門檻用「drawdown multiple」(實盤回撤 / 回測正常回撤) —— 系統交易實務界
#      (Robert Pardo, *The Evaluation and Optimization of Trading Strategies*, 2008,
#      walk-forward 偏離容忍；Lars Kestner, *Quantitative Trading Strategies*, 2003)
#      普遍以 ~1.5x 視為警戒、~2.0x 視為策略可能已失效 (regime change)。
#   3) 機率解讀 —— SPC/Shewhart 管制圖：將權益視為製程，回撤超出歷史分布尾端
#      (約 ±3σ ≈ 99.7%) 即「失控訊號」；Magdon-Ismail & Atiya (2004) 給定 Sharpe
#      與時間的期望最大回撤閉式解，可校準 baseline。
#
# baseline 由【滾動回測的 worst_max_drawdown_pct】推導，不是隨意拍板，且依標的不同。
# ETH：真實 Bybit 資料 (2021-2026) long_only 全期 MDD ≈ 17%、滾動最差視窗 ≈ 15% → 取保守上界 0.18。
# BTC：全期 MDD ≈ 21% → 取保守上界 0.22。換參數／換標的後請依新的滾動回測最差回撤同步更新。
CIRCUIT_BREAKER_ENABLED = True
SYMBOL_CIRCUIT_BREAKER_BASELINE = {
    "ETH/USDT": 0.18,
    "BTC/USDT": 0.22,
}
CIRCUIT_BREAKER_BASELINE_DRAWDOWN = SYMBOL_CIRCUIT_BREAKER_BASELINE.get(SYMBOL, 0.18)  # high-water-mark 基準
CIRCUIT_BREAKER_WARN_MULT = 1.5           # 回撤 ≥ 1.5x baseline → 警戒減碼
CIRCUIT_BREAKER_HALT_MULT = 2.0           # 回撤 ≥ 2.0x baseline → 熔斷暫停開新倉
CIRCUIT_BREAKER_WARN_SIZE_SCALE = 0.5     # 警戒級的開倉資金比例係數 (離散化 CPPI)

# === 波動度目標部位 (volatility targeting) ===
# 邏輯：波動低於目標時加碼、高於目標時減碼，讓「每筆交易的風險貢獻」維持穩定。
# 效果：平穩趨勢(策略最賺)放大部位、崩盤震盪(最易受傷)自動縮手 → 報酬放大但回撤不放大，
#       提升 Calmar/Sharpe(現代 CTA/risk-parity 的核心技術)。實測 Calmar 0.33→0.51。
# 安全：cap 上限 2.0 代表基準 20% 最多加到 40% 曝險(=1.2x 實質槓桿)，仍在「停損失效可恢復」區。
VOL_TARGET_ENABLED = True
VOL_TARGET_LOOKBACK = 180   # 4h K 線根數，約 30 天的實現波動 (rolling std of 4h returns)
VOL_SCALE_MIN = 0.3         # 波動極高時最多減碼到 30%
VOL_SCALE_MAX = 2.0         # 波動極低時最多加碼到 200%
SYMBOL_VOL_TARGET = {       # 各標的「典型波動」基準 = 歷史 rolling(180) 4h報酬std 中位數
    "ETH/USDT": 0.016,
    "BTC/USDT": 0.012,
}
VOL_TARGET_DEFAULT = 0.016

# 各標的最佳參數。
# ETH/USDT: 2026-05-19 Optuna；2017-2021 純樣本外 12 月視窗 76% 獲利、中位年化 ~20%。
SYMBOL_STRATEGY_PARAMS = {
    "ETH/USDT": {
        "adx_threshold": 30,
        "long_adx_threshold": 30,
        "short_adx_threshold": 45,
        "long_fixed_stop_loss_percent": 0.032381484813749646,
        "long_trailing_activate_profit_percent": 0.02845767104168034,
        "long_trailing_pullback_percent": 0.07857282056523368,
        "long_trailing_min_profit_percent": 0.02531577320019027,
        "short_fixed_stop_loss_percent": 0.014180455668711924,
        "short_trailing_activate_profit_percent": 0.004073022320643781,
        "short_trailing_pullback_percent": 0.08106115862332733,
        "short_trailing_min_profit_percent": 0.0014520289476228749,
    },
}
# BTC/USDT 刻意「沿用」ETH 的 long_only 參數，而非為 BTC 單獨優化。
# 三組獨立驗證證明 BTC 單獨優化都會過擬合到 2021-26 的單一下跌 regime：
#   - 讓 Optuna 自由選方向 → 選 short_only：樣本內 PF 1.07，但 2017-21 樣本外 PF 0.98（虧）。
#   - 強制 long_only 單獨優化 → 樣本內 PF 1.87、91% 視窗獲利，但 2017-21 樣本外暴跌到 PF 0.66。
#   - 直接套用 ETH 參數 → BTC 樣本內 PF 1.18、樣本外 2017-21 PF 1.17、71% 視窗獲利（穩健）。
# 趨勢跟隨捕捉的是加密貨幣普遍的趨勢動能，通用參數比逐標的精調更能泛化。
SYMBOL_STRATEGY_PARAMS["BTC/USDT"] = dict(SYMBOL_STRATEGY_PARAMS["ETH/USDT"])

# 預設 live 方向為 long_only（滾動視窗評分最高，且空單在加密貨幣長期虧損）。
STRATEGY_PARAMS = dict(SYMBOL_STRATEGY_PARAMS[SYMBOL])

# 系統設定
TRADE_SLEEP_SECONDS = 60  # 每隔多久檢查一次新K線 (60秒檢查一次)
# 抓取 K 線數量：EMA 是遞迴指標，視窗太短會殘留初始化偏差。
# 實測 (2025-01~2026-06, 3154 根 4h)：300 根時 EMA200 與全歷史值中位差 0.40%、最大 2.33%，
# 造成 5 個回測不存在的幽靈進場訊號；1000 根時偏差 <0.002%、訊號 100% 一致。
FETCH_KLINE_LIMIT = 1000

# 定義保存狀態的檔案路徑（用腳本所在目錄，避免 cron/systemd 等不同 cwd 啟動時寫到錯地方）
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategy_state.json")


# --- 1. 數據載入 (從 Bybit API 獲取數據) ---
_PUBLIC_EXCHANGE = None


def _get_public_exchange():
    """延遲建立並快取一個公開行情用的 Bybit 連線。

    舊版每次（每 60 秒）都重建連線並呼叫 load_markets()，既慢又容易觸發限流。
    這裡改為重用單一實例，並啟用 enableRateLimit 保護。
    """
    global _PUBLIC_EXCHANGE
    if _PUBLIC_EXCHANGE is not None:
        return _PUBLIC_EXCHANGE

    exchange = ccxt.bybit(
        {
            "enableRateLimit": True,  # 啟用速率限制，避免被交易所 ban IP
            "options": {
                "defaultType": "linear",  # Bybit USDT 線性合約
                "adjustForTimeDifference": True,  # 自動調整時間差
                "recvWindow": 120000,  # 增加接收窗口時間到2分鐘
            },
        }
    )

    # 同步時間（僅在建立連線時做一次，之後由 adjustForTimeDifference 維持）
    try:
        exchange.load_time_difference()
    except Exception:
        pass

    # 載入市場資訊，ccxt 需要知道市場資訊才能正確處理交易對
    exchange.load_markets()
    _PUBLIC_EXCHANGE = exchange
    return exchange


def fetch_bybit_klines(symbol, timeframe, limit=FETCH_KLINE_LIMIT):
    """
    從 Bybit 獲取指定交易對和時間週期的 K 線數據。
    """
    try:
        exchange = _get_public_exchange()
    except Exception as e:
        print(f"初始化行情連線失敗: {e}")
        return pd.DataFrame()

    try:
        # 獲取 K 線數據
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df.columns = [col.lower() for col in df.columns]  # 統一列名為小寫
        return df
    except ccxt.NetworkError as e:
        print(f"網路錯誤: {e}")
        return pd.DataFrame()
    except ccxt.ExchangeError as e:
        print(f"交易所錯誤: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"獲取 K 線數據時發生未知錯誤: {e}")
        return pd.DataFrame()


# --- 2. 指標計算 ---
def calculate_ema(series, period):
    """計算指數移動平均線"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_adx(high, low, close, period=14):
    """計算ADX指標 - 使用標準的Wilder平滑法"""
    # 計算True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # 計算方向移動
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    # 當+DM > -DM時，-DM = 0；當-DM > +DM時，+DM = 0
    plus_dm[(plus_dm <= minus_dm)] = 0
    minus_dm[(minus_dm <= plus_dm)] = 0

    # 使用Wilder平滑法 (alpha = 1/period)
    alpha = 1.0 / period

    # 計算平滑的TR和DM (使用指數移動平均)
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    # 計算DI
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)

    # 計算DX和ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx, plus_di, minus_di


def calculate_rsi(close, period=14):
    """計算RSI指標 - 使用 Wilder's RSI (EMA 平滑版)"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 使用 Wilder's 方法 (alpha = 1/period)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(close, fast=12, slow=26, signal=9):
    """計算MACD指標"""
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_indicators(df):
    """計算所有技術指標"""
    df = df.copy()

    # EMA指標
    df["ema90"] = calculate_ema(df["close"], 90)
    df["ema200"] = calculate_ema(df["close"], 200)

    # ADX指標
    adx, plus_di, minus_di = calculate_adx(df["high"], df["low"], df["close"], 14)
    df["adx"] = adx
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # RSI指標
    df["rsi"] = calculate_rsi(df["close"], 14)

    # MACD指標
    macd_line, signal_line, histogram = calculate_macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_histogram"] = histogram

    # 近期實現波動 (供波動度目標部位 vol targeting 用)。
    # 用 min_periods=2 並把 ret_vol 排除在 dropna 判斷之外，避免 180 根暖機 NaN
    # 縮短回測樣本、改變既有回測起點與結果（核心指標的 dropna 行為維持不變）。
    df["ret_vol"] = (
        df["close"].pct_change().rolling(VOL_TARGET_LOOKBACK, min_periods=2).std()
    )
    core_cols = [c for c in df.columns if c != "ret_vol"]
    return df.dropna(subset=core_cols)


def timeframe_to_timedelta(timeframe):
    """將 ccxt timeframe 轉成 timedelta，用來確認 K 線是否已收完。"""
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    raise ValueError(f"不支援的 timeframe: {timeframe}")


def get_latest_completed_bar(df, timeframe):
    """回傳真正已完成的最新 K 線，避免依賴 iloc[-2] 的交易所回傳假設。"""
    if df.empty:
        return None
    now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)
    cutoff = now_utc - timedelta(seconds=KLINE_COMPLETION_BUFFER_SECONDS)
    duration = timeframe_to_timedelta(timeframe)
    close_times = pd.DatetimeIndex(df.index) + duration
    completed = df[close_times <= cutoff]
    if completed.empty:
        return None
    return completed.iloc[-1]


# --- 3. 交易邏輯實現 ---
class TradingStrategy:
    def __init__(self, custom_params=None):
        """
        初始化交易策略
        custom_params: 可選的自定義參數字典，會覆蓋預設參數
        """
        # 使用預設參數，並允許自定義覆蓋
        params = STRATEGY_PARAMS.copy()
        if custom_params:
            params.update(custom_params)

        # 資金管理設定
        self.default_qty_percent = DEFAULT_QTY_PERCENT
        self.target_position_leverage = TARGET_POSITION_LEVERAGE
        self.auto_set_exchange_leverage = AUTO_SET_EXCHANGE_LEVERAGE
        self.require_exchange_leverage_capacity = REQUIRE_EXCHANGE_LEVERAGE_CAPACITY
        self.stop_trigger_by = STOP_TRIGGER_BY
        self.exit_on_reverse_signal = EXIT_ON_REVERSE_SIGNAL
        self.allow_same_bar_reversal = ALLOW_SAME_BAR_REVERSAL
        self.trade_side_mode = TRADE_SIDE_MODE.lower()
        if self.trade_side_mode not in VALID_TRADE_SIDE_MODES:
            raise ValueError(
                f"TRADE_SIDE_MODE 必須是 {sorted(VALID_TRADE_SIDE_MODES)}，目前是 {TRADE_SIDE_MODE!r}"
            )
        self.allow_long_entries = self.trade_side_mode in {"long_only", "both"}
        self.allow_short_entries = self.trade_side_mode in {"short_only", "both"}

        # 策略參數先初始化，避免讀取交易所既有持倉時觸發未定義屬性。
        self.adx_threshold = params["adx_threshold"]
        self.long_adx_threshold = params.get("long_adx_threshold", self.adx_threshold)
        self.short_adx_threshold = params.get("short_adx_threshold", self.adx_threshold)
        self.long_fixed_stop_loss_percent = params["long_fixed_stop_loss_percent"]
        self.long_trailing_activate_profit_percent = params[
            "long_trailing_activate_profit_percent"
        ]
        self.long_trailing_pullback_percent = params["long_trailing_pullback_percent"]
        self.long_trailing_min_profit_percent = params[
            "long_trailing_min_profit_percent"
        ]
        self.short_fixed_stop_loss_percent = params["short_fixed_stop_loss_percent"]
        self.short_trailing_activate_profit_percent = params[
            "short_trailing_activate_profit_percent"
        ]
        self.short_trailing_pullback_percent = params["short_trailing_pullback_percent"]
        self.short_trailing_min_profit_percent = params[
            "short_trailing_min_profit_percent"
        ]

        self.current_capital = 0
        self.total_equity = 0
        self.position_size = 0
        self.entry_price = 0
        self.long_entry_price = None
        self.long_peak = None
        self.long_trail_stop_price = None
        self.is_long_trail_active = False
        self.short_entry_price = None
        self.short_trough = None
        self.short_trail_stop_price = None
        self.is_short_trail_active = False
        self.peak_capital = 0
        self.max_drawdown = 0.0
        self.last_processed_kline_timestamp = None
        self.trade_log = []  # 實時交易日誌記錄

        # ccxt 交易所實例 - 統一帳戶合約交易
        self.exchange = ccxt.bybit(
            {
                "apiKey": BYBIT_API_KEY,
                "secret": BYBIT_API_SECRET,
                "sandbox": False,  # 實際交易模式
                "options": {
                    "defaultType": "linear",  # 統一帳戶線性合約
                    "adjustForTimeDifference": True,
                    "recvWindow": 120000,  # 增加接收窗口時間到2分鐘
                    "unified": True,  # 啟用統一帳戶模式
                },
                "enableRateLimit": True,  # 啟用速率限制，避免被交易所 ban IP
            }
        )

        # 同步時間
        try:
            self.exchange.load_time_difference()
        except:
            pass
        self.exchange.load_markets()
        self.symbol = SYMBOL
        self.bybit_symbol = self.exchange.market(self.symbol).get(
            "id", SYMBOL.replace("/", "")
        )

        # 程式槓桿只控制倉位大小，交易所槓桿只控制保證金需求。
        print(
            f"⚠️ 槓桿設定: 程式目標名目槓桿 {self.target_position_leverage}x，"
            "請將 Bybit 合約槓桿設為大於或等於此值。"
        )
        print("   若交易所槓桿更高，訂單較容易通過保證金檢查；但 isolated 模式爆倉距離也會更近。")

        # 嘗試從檔案加載狀態
        if not self.load_state():
            print("未找到或無法加載狀態檔案，初始化策略狀態...")
            self._refresh_account_balances()  # 從 Bybit 獲取當前可用資金與總權益
            actual_position = self._get_current_position_size()  # 從 Bybit 獲取當前持倉
            self.position_size = actual_position
            actual_avg_price = (
                self._get_position_avg_price("long")
                if actual_position > 0
                else self._get_position_avg_price("short")
                if actual_position < 0
                else None
            )
            if actual_position > 0 and actual_avg_price:
                self.entry_price = actual_avg_price
                self.long_entry_price = actual_avg_price
            elif actual_position < 0 and actual_avg_price:
                self.entry_price = actual_avg_price
                self.short_entry_price = actual_avg_price
            else:
                self.entry_price = 0
                self.long_entry_price = None
                self.short_entry_price = None
            self.peak_capital = self.total_equity or self.current_capital
            self.max_drawdown = 0.0
            self.save_state()
        else:
            print("策略狀態已從檔案加載。")
            # 🔧 重要修正：加載後必須重新同步實際持倉和進場價格
            self._refresh_account_balances()
            actual_position = self._get_current_position_size()

            # � 重要修正：加載後必須重新同步實際持倉和進場價格
            if actual_position != 0:
                actual_avg_price = (
                    self._get_position_avg_price("long")
                    if actual_position > 0
                    else self._get_position_avg_price("short")
                )

                if actual_avg_price and actual_avg_price > 0:
                    old_price = self.long_entry_price if actual_position > 0 else self.short_entry_price

                    # 只有在價格不同時才顯示更新訊息
                    if abs(actual_avg_price - (old_price or 0)) > 0.01:
                        print(f"🔧 進場價格同步: ${old_price} → ${actual_avg_price:.2f}")

                    self.entry_price = actual_avg_price
                    self.position_size = actual_position
                    if actual_position > 0:
                        self.long_entry_price = actual_avg_price
                        # 重置空單相關狀態
                        self.short_entry_price = None
                        self.short_trough = None
                        self.short_trail_stop_price = None
                        self.is_short_trail_active = False
                    else:
                        self.short_entry_price = actual_avg_price
                        # 重置多單相關狀態
                        self.long_entry_price = None
                        self.long_peak = None
                        self.long_trail_stop_price = None
                        self.is_long_trail_active = False

                    self.save_state()  # 保存更正後的狀態
            else:
                # 無持倉時重置所有進場相關狀態
                self.position_size = 0
                self.entry_price = 0
                self.long_entry_price = None
                self.short_entry_price = None
                self.long_peak = None
                self.long_trail_stop_price = None
                self.is_long_trail_active = False
                self.short_trough = None
                self.short_trail_stop_price = None
                self.is_short_trail_active = False
                self.save_state()

            print(
                f"📊 帳戶狀態：未使用資金: {self.current_capital:.2f} USDT, 持倉量: {self.position_size:.3f} {SYMBOL.split('/')[0]}"
            )

        if self.position_size != 0:
            self._sync_exchange_protective_stop("初始化同步")

        print(
            f"✅ 策略初始化完成 | 未使用資金: {self.current_capital:.2f} USDT | "
            f"持倉: {self.position_size:.3f} {SYMBOL.split('/')[0]} | "
            f"目標名目槓桿: {self.target_position_leverage}x | "
            f"方向模式: {self.trade_side_mode}"
        )

    def _get_free_balance(self, currency="USDT"):
        """獲取 Bybit 帳戶的可用資金"""
        try:
            balance = self.exchange.fetch_balance()
            return balance.get("free", {}).get(currency, 0)
        except Exception as e:
            print(f"獲取帳戶餘額失敗: {e}")
            return 0

    def _refresh_account_balances(self, currency="USDT"):
        """同步可用資金與總權益；下單只使用 free balance，回撤統計使用 total equity。"""
        try:
            balance = self.exchange.fetch_balance()
            free_balance = float(balance.get("free", {}).get(currency, 0) or 0)
            total_equity = float(
                balance.get("total", {}).get(currency, free_balance) or free_balance
            )
            self.current_capital = free_balance
            self.total_equity = total_equity
            return free_balance, total_equity
        except Exception as e:
            print(f"獲取帳戶餘額失敗: {e}")
            return self.current_capital, self.total_equity

    def _fetch_raw_positions(self):
        """直接從 Bybit V5 讀取目前 symbol 的持倉資料。"""
        if not hasattr(self.exchange, "private_get_v5_position_list"):
            return []
        params = {"category": "linear", "symbol": self.bybit_symbol}
        response = self.exchange.private_get_v5_position_list(params)
        return response.get("result", {}).get("list", [])

    def _get_position_idx(self, position_side=None):
        """
        取得 Bybit positionIdx。
        one-way mode 是 0；hedge mode 多單是 1、空單是 2。
        """
        try:
            positions = self._fetch_raw_positions()
            for pos in positions:
                size = float(pos.get("size", 0) or 0)
                side = pos.get("side", "")
                if size <= 0:
                    continue
                if position_side == "long" and side == "Buy":
                    return int(pos.get("positionIdx", POSITION_IDX_ONE_WAY) or POSITION_IDX_ONE_WAY)
                if position_side == "short" and side == "Sell":
                    return int(pos.get("positionIdx", POSITION_IDX_ONE_WAY) or POSITION_IDX_ONE_WAY)
                if position_side is None:
                    return int(pos.get("positionIdx", POSITION_IDX_ONE_WAY) or POSITION_IDX_ONE_WAY)
        except Exception as e:
            print(f"⚠️ 讀取 positionIdx 失敗，改用 one-way mode: {e}")
        return POSITION_IDX_ONE_WAY

    def _get_order_position_idx(self, side):
        """新開倉時依帳戶 position mode 選擇 positionIdx。"""
        try:
            if hasattr(self.exchange, "fetch_position_mode"):
                mode = self.exchange.fetch_position_mode(
                    self.symbol, {"category": "linear"}
                )
                if mode.get("hedged"):
                    return 1 if side.lower() == "buy" else 2
        except Exception:
            pass
        return POSITION_IDX_ONE_WAY

    def _format_amount(self, amount):
        """依交易所精度格式化數量，避免 round() 造成超額下單。"""
        try:
            return float(self.exchange.amount_to_precision(self.symbol, amount))
        except Exception:
            return math.floor(amount * 100) / 100

    def _format_price(self, price):
        """依交易所精度格式化價格。"""
        try:
            return self.exchange.price_to_precision(self.symbol, price)
        except Exception:
            return f"{price:.2f}"

    def _get_exchange_leverage(self):
        """讀取交易所目前設定槓桿；讀不到時回傳 None，不阻擋策略。"""
        try:
            if hasattr(self.exchange, "fetch_leverage"):
                leverage_info = self.exchange.fetch_leverage(
                    self.symbol, {"category": "linear"}
                )
                values = [
                    leverage_info.get("longLeverage"),
                    leverage_info.get("shortLeverage"),
                ]
                leverages = [float(v) for v in values if v not in [None, "", 0, "0"]]
                if leverages:
                    return min(leverages)
        except Exception:
            pass

        try:
            leverages = []
            for pos in self._fetch_raw_positions():
                leverage = pos.get("leverage")
                if leverage not in [None, "", "0", 0]:
                    leverages.append(float(leverage))
            if leverages:
                return min(leverages)
        except Exception:
            pass
        return None

    def _ensure_exchange_leverage_capacity(self):
        """
        確認交易所槓桿足以支援程式目標名目槓桿。
        交易所槓桿 >= 程式目標槓桿時，訂單保證金需求通常會低於設定的資金使用比例。
        """
        target = float(self.target_position_leverage)
        actual = self._get_exchange_leverage()

        if actual is None:
            print("⚠️ 無法讀取交易所槓桿，會繼續下單；請自行確認 Bybit 槓桿 >= 程式目標槓桿。")
            return True

        if actual >= target:
            return True

        if self.auto_set_exchange_leverage:
            try:
                response = self.exchange.private_post_v5_position_set_leverage(
                    {
                        "category": "linear",
                        "symbol": self.bybit_symbol,
                        "buyLeverage": str(target),
                        "sellLeverage": str(target),
                    }
                )
                if response.get("retCode") == 0:
                    print(f"✅ 已將 Bybit 槓桿調整為 {target}x")
                    return True
                print(f"⚠️ 自動調整 Bybit 槓桿失敗: {response}")
            except Exception as e:
                print(f"⚠️ 自動調整 Bybit 槓桿失敗: {e}")

        message = (
            f"⚠️ Bybit 目前槓桿 {actual}x 低於程式目標名目槓桿 {target}x。"
            "請手動調高交易所槓桿，或降低 TARGET_POSITION_LEVERAGE。"
        )
        if self.require_exchange_leverage_capacity:
            print(message + " 本次新開倉取消。")
            return False
        print(message + " 仍會嘗試下單。")
        return True

    def _calculate_trade_quantity(self, current_close, size_scale=1.0):
        """根據資金比例與目標名目槓桿計算下單數量。

        size_scale 供回撤熔斷分級減碼使用 (1.0=正常, 0.5=警戒減半, 0=熔斷暫停)。
        """
        self.current_capital, _ = self._refresh_account_balances()
        effective_qty_percent = self.default_qty_percent * size_scale
        margin_budget_usd = self.current_capital * effective_qty_percent / 100
        target_notional_usd = margin_budget_usd * self.target_position_leverage
        trade_qty_unrounded = target_notional_usd / current_close
        trade_qty = self._format_amount(trade_qty_unrounded)

        market = self.exchange.market(self.symbol)
        min_amount = (
            market["limits"]["amount"]["min"] if "amount" in market["limits"] else 0.001
        )

        if trade_qty < min_amount:
            print(
                f"❌ 計算出的交易數量 {trade_qty:.6f} 小於最小交易量 {min_amount:.6f}，跳過交易。"
            )
            return 0

        print(
            f"📐 倉位計算: 可用資金 {self.current_capital:.2f} USDT x "
            f"{effective_qty_percent:.1f}% x 目標名目槓桿 {self.target_position_leverage}x "
            f"= 目標名目 {target_notional_usd:.2f} USDT，數量 {trade_qty:.4f} ETH"
        )
        return trade_qty

    def _drawdown_state(self):
        """回傳 (size_scale, level, drawdown)，依當前回撤分級控制【開新倉】曝險。

        level 0=正常, 1=警戒減碼, 2=熔斷暫停。即時依 total_equity 對 high-water
        mark 的回撤計算，回撤縮回 baseline 以下即自動恢復，無需額外持久化狀態。
        資料不足或讀數異常時回傳正常 (1.0, 0)，避免誤殺。
        """
        if not CIRCUIT_BREAKER_ENABLED:
            return 1.0, 0, 0.0
        peak = self.peak_capital or 0.0
        equity = self.total_equity
        # 異常／不足資料時不熔斷（保守：寧可不誤殺，也不在壞讀數上停單）
        if peak <= 0 or equity is None or equity < MIN_PLAUSIBLE_EQUITY_USDT:
            return 1.0, 0, 0.0
        peak = max(peak, equity)  # 若創新高但 peak 尚未更新，回撤視為 0
        drawdown = (peak - equity) / peak
        warn = CIRCUIT_BREAKER_BASELINE_DRAWDOWN * CIRCUIT_BREAKER_WARN_MULT
        halt = CIRCUIT_BREAKER_BASELINE_DRAWDOWN * CIRCUIT_BREAKER_HALT_MULT
        if drawdown >= halt:
            return 0.0, 2, drawdown
        if drawdown >= warn:
            return CIRCUIT_BREAKER_WARN_SIZE_SCALE, 1, drawdown
        return 1.0, 0, drawdown

    def _prepare_entry_quantity(self, current_close, recent_vol=None):
        """新開倉前統一檢查槓桿、回撤熔斷、波動度目標，並以最新 free balance 計算下單數量。"""
        if not self._ensure_exchange_leverage_capacity():
            return 0

        size_scale, level, drawdown = self._drawdown_state()
        warn_th = CIRCUIT_BREAKER_BASELINE_DRAWDOWN * CIRCUIT_BREAKER_WARN_MULT
        halt_th = CIRCUIT_BREAKER_BASELINE_DRAWDOWN * CIRCUIT_BREAKER_HALT_MULT
        if level >= 2:
            print(
                f"🟥 回撤熔斷啟動：當前回撤 {drawdown*100:.1f}% ≥ 熔斷門檻 {halt_th*100:.1f}% "
                f"(={CIRCUIT_BREAKER_HALT_MULT}x baseline)，暫停開新倉。"
                "（既有倉位的固定／移動停損照常運作）"
            )
            return 0
        if level == 1:
            print(
                f"🟧 回撤警戒：當前回撤 {drawdown*100:.1f}% ≥ 警戒門檻 {warn_th*100:.1f}% "
                f"(={CIRCUIT_BREAKER_WARN_MULT}x baseline)，本次開倉資金比例降為 {size_scale*100:.0f}%。"
            )

        # 波動度目標部位：與熔斷係數相乘（兩者都是對「基準曝險」的乘性調整）。
        vol_scale = self._vol_scale(recent_vol)
        if abs(vol_scale - 1.0) > 1e-9:
            arrow = "加碼" if vol_scale > 1 else "減碼"
            print(
                f"📊 波動度部位調整：近期波動 {float(recent_vol):.4f} → 倉位係數 x{vol_scale:.2f}（{arrow}）"
            )
        size_scale *= vol_scale
        return self._calculate_trade_quantity(current_close, size_scale=size_scale)

    def _vol_scale(self, recent_vol):
        """波動度目標部位係數 = 目標波動 / 近期波動，clip 在 [MIN, MAX]。

        近期波動低於目標 → 係數>1（加碼）；高於目標 → 係數<1（減碼）。
        未啟用或資料不足時回 1.0（不調整）。
        """
        if not VOL_TARGET_ENABLED or recent_vol is None:
            return 1.0
        try:
            v = float(recent_vol)
        except (TypeError, ValueError):
            return 1.0
        if pd.isna(v) or v <= 0:
            return 1.0
        target = SYMBOL_VOL_TARGET.get(SYMBOL, VOL_TARGET_DEFAULT)
        return float(min(max(target / v, VOL_SCALE_MIN), VOL_SCALE_MAX))

    def _fixed_stop_price_for_side(self, position_side):
        if position_side == "long" and self.long_entry_price:
            return self.long_entry_price * (1 - self.long_fixed_stop_loss_percent)
        if position_side == "short" and self.short_entry_price:
            return self.short_entry_price * (1 + self.short_fixed_stop_loss_percent)
        return None

    def _current_protective_stop_price(self, position_side):
        """取得目前應該掛在交易所端的保護停損價。"""
        if position_side == "long":
            if self.is_long_trail_active and self.long_trail_stop_price:
                return self.long_trail_stop_price
            return self._fixed_stop_price_for_side("long")
        if position_side == "short":
            if self.is_short_trail_active and self.short_trail_stop_price:
                return self.short_trail_stop_price
            return self._fixed_stop_price_for_side("short")
        return None

    def _handle_local_fixed_stop_fallback(
        self, position_side, current_low, current_high, current_close, current_time
    ):
        """本地 1m 固定停損兜底，避免交易所端 SL 設定失敗時裸倉。

        用 1m K 線的 low/high 判斷，對齊回測中 `bar_low <= fixed_stop` /
        `bar_high >= fixed_stop` 的觸發語意。保留 buffer 讓交易所端 MarkPrice
        SL 有機會先觸發。
        """
        fixed_stop = self._fixed_stop_price_for_side(position_side)
        if not fixed_stop:
            return False

        # The exchange protective SL triggers on MarkPrice. Keep this as a true
        # fallback by requiring price to move clearly beyond the fixed stop.
        long_fallback_stop = fixed_stop * (1 - LOCAL_FIXED_STOP_FALLBACK_BUFFER_PCT)
        short_fallback_stop = fixed_stop * (1 + LOCAL_FIXED_STOP_FALLBACK_BUFFER_PCT)
        triggered = (
            (position_side == "long" and current_low <= long_fallback_stop)
            or (position_side == "short" and current_high >= short_fallback_stop)
        )
        if not triggered:
            return False

        print(f"\n\n🚨 === 本地 1m 固定停損兜底觸發 ===")
        print(f"時間: {current_time}")
        print(f"方向: {position_side.upper()}")
        trigger_extreme = current_low if position_side == "long" else current_high
        print(f"觸發極值: ${trigger_extreme:.2f} (close ${current_close:.2f})")
        print(f"固定停損價: ${fixed_stop:.2f}")
        fallback_stop = long_fallback_stop if position_side == "long" else short_fallback_stop
        print(f"兜底觸發價: ${fallback_stop:.2f}")
        close_success = self._close_position(current_close)
        if close_success:
            print("✅ 本地固定停損兜底平倉完成")
        else:
            print("❌ 本地固定停損兜底平倉失敗，請立即人工檢查")
        return True

    def _set_exchange_stop_loss(self, position_side, stop_price, reason="protective stop"):
        """
        在 Bybit 交易所端設定整倉保護停損。
        本地輪詢仍保留，但交易所端 SL 是斷線或 API 失敗時的真正保險絲。
        """
        if not stop_price or stop_price <= 0:
            return False
        if not hasattr(self.exchange, "private_post_v5_position_trading_stop"):
            print("⚠️ 目前 ccxt 版本不支援 Bybit trading-stop 原始 API，無法設定交易所端停損。")
            return False

        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            ticker_info = ticker.get("info") if isinstance(ticker.get("info"), dict) else {}
            market_price = self._coerce_positive_float(
                ticker.get("mark")
                or ticker_info.get("markPrice")
                or ticker.get("last")
                or ticker.get("close")
            )
        except Exception as e:
            market_price = None
            print(f"⚠️ 停損合理性檢查無法取得市價，仍會嘗試設定停損: {e}")

        if market_price:
            if position_side == "long" and stop_price >= market_price:
                print(
                    f"⚠️ 多單停損 ${stop_price:.2f} >= 現價 ${market_price:.2f}，"
                    "跳過交易所端停損設定，請檢查 entry_price / trail state。"
                )
                return False
            if position_side == "short" and stop_price <= market_price:
                print(
                    f"⚠️ 空單停損 ${stop_price:.2f} <= 現價 ${market_price:.2f}，"
                    "跳過交易所端停損設定，請檢查 entry_price / trail state。"
                )
                return False

        params = {
            "category": "linear",
            "symbol": self.bybit_symbol,
            "tpslMode": "Full",
            "positionIdx": self._get_position_idx(position_side),
            "stopLoss": self._format_price(stop_price),
            "slTriggerBy": self.stop_trigger_by,
            "slOrderType": "Market",
        }
        for attempt in range(1, STOP_LOSS_SYNC_RETRIES + 1):
            try:
                response = self.exchange.private_post_v5_position_trading_stop(params)
                if response.get("retCode") == 0:
                    print(
                        f"🛡️ 已更新交易所端停損 ({reason}): "
                        f"{position_side.upper()} @ {params['stopLoss']} ({self.stop_trigger_by})"
                    )
                    return True
                print(
                    f"⚠️ 設定交易所端停損失敗 "
                    f"({attempt}/{STOP_LOSS_SYNC_RETRIES}): {response}"
                )
            except Exception as e:
                print(
                    f"⚠️ 設定交易所端停損失敗 "
                    f"({attempt}/{STOP_LOSS_SYNC_RETRIES}): {e}"
                )

            if attempt < STOP_LOSS_SYNC_RETRIES:
                time.sleep(STOP_LOSS_SYNC_RETRY_DELAY_SECONDS)

        print(
            "🚨 交易所端停損連續設定失敗，請立即人工確認 Bybit 保護停損。"
            "本地 1m 固定停損兜底仍會運作，但程式斷線時將失去交易所端保護。"
        )
        return False

    def _sync_exchange_protective_stop(self, reason="sync protective stop"):
        """依目前本地狀態，把交易所端停損校正到應有位置。"""
        if self.position_size > 0:
            stop_price = self._current_protective_stop_price("long")
            return self._set_exchange_stop_loss("long", stop_price, reason)
        if self.position_size < 0:
            stop_price = self._current_protective_stop_price("short")
            return self._set_exchange_stop_loss("short", stop_price, reason)
        return False

    def _coerce_positive_float(self, value):
        try:
            if value in [None, "", "N/A"]:
                return None
            parsed = float(value)
            return parsed if parsed > 0 else None
        except (TypeError, ValueError):
            return None

    def _order_execution_price(self, order, fallback_price):
        """Best-effort execution price extraction; falls back to caller's market snapshot."""
        if not order:
            return fallback_price
        for field in ("average", "price", "avgPrice", "lastPrice", "executedPrice"):
            price = self._coerce_positive_float(order.get(field))
            if price:
                return price
        info = order.get("info") if isinstance(order.get("info"), dict) else {}
        for field in ("avgPrice", "price", "lastPrice", "executedPrice"):
            price = self._coerce_positive_float(info.get(field))
            if price:
                return price
        return fallback_price

    def _entry_price_changed(self, old_price, new_price):
        if not old_price or old_price <= 0:
            return True
        return abs(new_price - old_price) / old_price > ENTRY_PRICE_SYNC_TOLERANCE_PCT

    def _refresh_trailing_state_after_entry_sync(self, position_side, avg_price, mark_price):
        current_price = self._coerce_positive_float(mark_price) or avg_price
        if position_side == "long":
            self.long_peak = max(self.long_peak or current_price, current_price)
            profit_percent = (current_price - avg_price) / avg_price
            if self.is_long_trail_active or profit_percent > self.long_trailing_activate_profit_percent:
                self.is_long_trail_active = True
                min_profit = avg_price * (1 + self.long_trailing_min_profit_percent)
                pullback_stop = self.long_peak * (1 - self.long_trailing_pullback_percent)
                self.long_trail_stop_price = max(
                    self.long_trail_stop_price or 0,
                    min_profit,
                    pullback_stop,
                )
            return

        self.short_trough = min(self.short_trough or current_price, current_price)
        profit_percent = (avg_price - current_price) / avg_price
        if self.is_short_trail_active or profit_percent > self.short_trailing_activate_profit_percent:
            self.is_short_trail_active = True
            min_profit = avg_price * (1 - self.short_trailing_min_profit_percent)
            pullback_stop = self.short_trough * (1 + self.short_trailing_pullback_percent)
            self.short_trail_stop_price = min(
                self.short_trail_stop_price or float("inf"),
                min_profit,
                pullback_stop,
            )

    def _sync_entry_price_from_exchange(self, position_side, avg_price, mark_price=None):
        local_price = (
            self.long_entry_price if position_side == "long" else self.short_entry_price
        )
        if not self._entry_price_changed(local_price, avg_price):
            return False

        print(
            f"🔧 進場均價同步 ({position_side.upper()}): "
            f"${local_price or 0:.2f} → ${avg_price:.2f}"
        )
        print("⚠️ 進場價已同步，移動停損狀態可能立即變更；若剛手動加倉，請確認這符合你的倉位意圖。")
        self.entry_price = avg_price
        if position_side == "long":
            self.long_entry_price = avg_price
            self.short_entry_price = None
            self.short_trough = None
            self.short_trail_stop_price = None
            self.is_short_trail_active = False
        else:
            self.short_entry_price = avg_price
            self.long_entry_price = None
            self.long_peak = None
            self.long_trail_stop_price = None
            self.is_long_trail_active = False

        self._refresh_trailing_state_after_entry_sync(position_side, avg_price, mark_price)
        self.save_state()
        self._sync_exchange_protective_stop("進場均價同步後校正停損")
        return True

    def _get_current_position_size(self, sync_entry_price=True):
        """獲取 Bybit 統一帳戶目前持倉量，可選擇同步交易所 avgPrice 到本地停損基準。"""
        try:
            positions = self._fetch_raw_positions()
            for pos in positions:
                size = float(pos.get("size", 0) or 0)
                if size <= 0:
                    continue

                side = pos.get("side", "")
                avg_price = self._coerce_positive_float(pos.get("avgPrice"))
                mark_price = pos.get("markPrice")
                if side == "Buy":
                    self.position_size = size
                    if sync_entry_price and avg_price:
                        self._sync_entry_price_from_exchange("long", avg_price, mark_price)
                    return size
                if side == "Sell":
                    self.position_size = -size
                    if sync_entry_price and avg_price:
                        self._sync_entry_price_from_exchange("short", avg_price, mark_price)
                    return -size

            self.position_size = 0
            print("📊 無持倉")
            return 0

        except Exception as e:
            print(f"獲取持倉失敗: {e}")
            print("⚠️ 持倉查詢失敗時保留本地持倉狀態，避免誤判空倉。")
            return self.position_size

    def _get_position_avg_price(self, position_side=None):
        """獲取當前持倉的平均進場價格"""
        try:
            # 使用原始API直接獲取持倉
            positions = self._fetch_raw_positions()
            if positions:

                    for pos in positions:
                        size = float(pos.get("size", 0))
                        side = pos.get("side", "")
                        avg_price = pos.get("avgPrice", "N/A")

                        if size > 0:
                            if position_side == "long" and side != "Buy":
                                continue
                            if position_side == "short" and side != "Sell":
                                continue
                            if avg_price != "N/A" and avg_price != "" and avg_price != "0":
                                try:
                                    return float(avg_price)
                                except:
                                    pass
            return None
        except Exception as e:
            print(f"❌ 獲取持倉平均價格失敗: {e}")
            return None

    def _place_order(
        self,
        side,
        trade_qty,
        price_type="market",
        reduce_only=False,
        position_idx=None,
        stop_loss_price=None,
    ):
        """下單到 Bybit 統一帳戶。

        stop_loss_price 提供時，會在進場單上「直接夾帶」交易所端整倉固定停損
        (MarkPrice 即時觸發)，成交當下即生效、零裸倉窗口，不需事後再設一次。
        """
        try:
            # 確保數量是非零的
            if trade_qty <= 0:
                print(f"嘗試下單數量為 {trade_qty}，訂單取消。")
                return None

            # 嘗試不同的下單參數組合
            print(
                f"🔄 嘗試下單: {side} {trade_qty} {self.symbol}"
                f"{' reduce-only' if reduce_only else ''}"
                f"{f' SL@{stop_loss_price:.2f}' if stop_loss_price and not reduce_only else ''}"
            )

            if position_idx is None:
                position_idx = self._get_order_position_idx(side)

            # 進場單夾帶交易所端固定停損（平倉/減倉單不帶）。
            sl_params = {}
            if stop_loss_price and stop_loss_price > 0 and not reduce_only:
                sl_params = {
                    "stopLoss": self._format_price(stop_loss_price),
                    "slTriggerBy": self.stop_trigger_by,
                    "slOrderType": "Market",
                    "tpslMode": "Full",
                }

            order_params_common = {
                "category": "linear",
                "positionIdx": position_idx,
                "reduceOnly": reduce_only,
                **sl_params,
            }

            # 方法1: 強制使用線性合約參數
            try:
                order = self.exchange.create_order(
                    symbol=self.symbol,
                    type=price_type,
                    side=side,
                    amount=trade_qty,
                    price=None,
                    params=order_params_common,
                )
            except Exception as e1:
                print(f"方法1失敗: {e1}")

                # 方法2: 使用原始API確保合約交易
                try:
                    if hasattr(self.exchange, "private_post_v5_order_create"):
                        order_params = {
                            "category": "linear",  # 強制線性合約
                            "symbol": self.bybit_symbol,
                            "side": side.capitalize(),
                            "orderType": "Market",
                            "qty": str(trade_qty),
                            "positionIdx": position_idx,
                            "reduceOnly": reduce_only,
                            **sl_params,
                        }
                        response = self.exchange.private_post_v5_order_create(
                            order_params
                        )
                        if response.get("retCode") == 0:
                            # 🔧 修正：嘗試獲取成交價格
                            filled_price = None
                            result = response.get("result", {})
                            
                            # 嘗試多個可能的價格字段
                            price_fields = ["avgPrice", "price", "executedPrice", "lastPrice"]
                            for field in price_fields:
                                if field in result and result[field] not in [None, "", "0", 0]:
                                    try:
                                        filled_price = float(result[field])
                                        break
                                    except:
                                        continue
                            
                            # 如果還是沒有價格，嘗試獲取當前市價
                            if filled_price is None:
                                try:
                                    ticker = self.exchange.fetch_ticker(self.symbol)
                                    filled_price = ticker['last']
                                except:
                                    filled_price = "市價"

                            order = {
                                "id": result.get("orderId", "unknown"),
                                "side": side,
                                "amount": trade_qty,
                                "filled": 0,
                                "price": filled_price,
                                "symbol": self.symbol,
                                "type": price_type,
                                "status": "submitted",
                            }
                        else:
                            raise Exception(f"API錯誤: {response}")
                    else:
                        raise Exception("無可用的下單方法")
                except Exception as e2:
                    print(f"方法2失敗: {e2}")
                    raise e2
            print(
                f"下單已送出: {order.get('side', side)} {order.get('amount', trade_qty)} "
                f"{order.get('symbol', self.symbol)} @ {order.get('price', 'N/A')} "
                f"(類型: {order.get('type', price_type)})"
            )
            self.trade_log.append(
                {
                    "time": datetime.now().isoformat(),
                    "type": f"ORDER_{side.upper()}",  # isoformat 讓 datetime 可 JSON 序列化
                    "price": order.get("price", "N/A"),
                    "qty": trade_qty,
                    "status": order.get("status", "submitted"),
                    "order_id": order.get("id", "unknown"),
                    "reduce_only": reduce_only,
                }
            )
            return order
        except ccxt.InsufficientFunds as e:
            print(f"資金不足，無法下單 {side} {trade_qty} {self.symbol}")
            print(f"詳細錯誤: {e}")
            self.trade_log.append(
                {
                    "time": datetime.now().isoformat(),
                    "type": "ERROR_INSUFFICIENT_FUNDS",
                    "details": str(e),
                }
            )
        except ccxt.InvalidOrder as e:
            print(f"無效訂單: {e}")
            self.trade_log.append(
                {
                    "time": datetime.now().isoformat(),
                    "type": "ERROR_INVALID_ORDER",
                    "details": str(e),
                }
            )
        except Exception as e:
            print(f"下單失敗: {e}")
            self.trade_log.append(
                {
                    "time": datetime.now().isoformat(),
                    "type": "ERROR_ORDER_FAILED",
                    "details": str(e),
                }
            )
        return None

    def _close_position(self, current_close):
        """平倉當前持有的所有倉位"""
        print(f"\n🔄 開始平倉程序...")

        # 添加重試機制確保狀態同步
        max_retries = 3
        actual_position = 0

        for attempt in range(max_retries):
            print(f"📊 嘗試 {attempt+1}/{max_retries}: 查詢當前持倉...")
            actual_position = self._get_current_position_size()

            if actual_position == 0:
                if attempt < max_retries - 1:
                    print(f"⏳ 查詢顯示無持倉，等待2秒後重試...")
                    time.sleep(2)
                    continue
                else:
                    print("📊 多次查詢確認無實際持倉")
                    # 重置內部狀態
                    print("🔧 重置所有內部交易狀態...")
                    self.position_size = 0
                    self.entry_price = 0
                    self.long_entry_price = None
                    self.long_peak = None
                    self.long_trail_stop_price = None
                    self.is_long_trail_active = False
                    self.short_entry_price = None
                    self.short_trough = None
                    self.short_trail_stop_price = None
                    self.is_short_trail_active = False
                    self.save_state()
                    return False
            else:
                print(f"✅ 確認持倉: {actual_position:.5f} ETH")
                break

        if actual_position == 0:
            print("❌ 多次查詢後仍顯示無持倉，可能存在同步問題")
            return False

        # 使用實際持倉數量進行平倉
        abs_pos_size = abs(actual_position)
        order = None

        print(f"🔄 準備平倉: 實際持倉 {actual_position:.5f} ETH")

        if actual_position > 0:  # 平多單
            print(
                f"📉 平多單: {actual_position:.5f} {self.symbol} @ ${current_close:.2f}"
            )
            order = self._place_order(
                "sell",
                abs_pos_size,
                "market",
                reduce_only=True,
                position_idx=self._get_position_idx("long"),
            )
        elif actual_position < 0:  # 平空單
            print(f"📈 平空單: {abs_pos_size:.5f} {self.symbol} @ ${current_close:.2f}")
            order = self._place_order(
                "buy",
                abs_pos_size,
                "market",
                reduce_only=True,
                position_idx=self._get_position_idx("short"),
            )

        if order:
            print(f"✅ 平倉訂單已提交: {order.get('id', 'N/A')}")

            # 等待並確認平倉結果
            print(f"⏳ 等待3秒後確認平倉結果...")
            time.sleep(3)

            # 確認平倉是否成功
            confirmation_retries = 3
            final_position = None

            for confirm_attempt in range(confirmation_retries):
                print(
                    f"🔍 確認嘗試 {confirm_attempt+1}/{confirmation_retries}: 查詢平倉後持倉..."
                )
                final_position = self._get_current_position_size(sync_entry_price=False)

                if final_position == 0:
                    print(f"✅ 平倉成功確認：持倉已清零")
                    break
                else:
                    print(f"⚠️ 平倉可能未完成，剩餘持倉: {final_position:.5f}")
                    if confirm_attempt < confirmation_retries - 1:
                        time.sleep(2)

            if final_position != 0:
                print(f"❌ 平倉確認失敗，剩餘持倉: {final_position:.5f}")
                self.trade_log.append(
                    {
                        "time": datetime.now().isoformat(),
                        "type": "ERROR_CLOSE_FAILED",
                        "remaining_position": final_position,
                        "order_id": order.get("id", "N/A"),
                    }
                )
                return False

            # 計算盈虧。優先使用訂單回傳成交價，取不到才退回觸發時的市場快照。
            exit_price = self._order_execution_price(order, current_close)
            entry_price_for_calc = (
                self.long_entry_price if actual_position > 0 else self.short_entry_price
            )
            if entry_price_for_calc is not None and entry_price_for_calc > 0:
                if actual_position > 0:
                    profit_loss = (
                        exit_price - entry_price_for_calc
                    ) * actual_position
                else:
                    profit_loss = (entry_price_for_calc - exit_price) * abs(
                        actual_position
                    )
                print(f"💰 平倉盈虧: ${profit_loss:.2f} USDT (估算成交價 ${exit_price:.2f})")
            else:
                profit_loss = 0
                print("⚠️ 無進場價格記錄，無法計算精確盈虧")

            # 更新狀態
            self._refresh_account_balances()  # 平倉後再次更新可用資金與總權益
            self.position_size = 0
            self.entry_price = 0
            self.long_entry_price = None
            self.long_peak = None
            self.long_trail_stop_price = None
            self.is_long_trail_active = False
            self.short_entry_price = None
            self.short_trough = None
            self.short_trail_stop_price = None
            self.is_short_trail_active = False

            self.trade_log.append(
                {
                    "time": datetime.now().isoformat(),
                    "type": "EXIT_REAL",
                    "price": exit_price,
                    "trigger_price": current_close,
                    "profit_loss": profit_loss,
                    "current_position_size": self.position_size,
                    "current_capital": self.current_capital,
                    "order_id": order.get("id", "N/A"),
                }
            )
            self.save_state()  # 平倉後保存狀態
            print(f"✅ 平倉完成，狀態已重置")
            return True
        else:
            print(f"❌ 平倉失敗，訂單未成功")
            return False

    # --- 新增：保存策略狀態到 JSON 檔案 ---
    def save_state(self):
        state = {
            "position_size": self.position_size,
            "entry_price": self.entry_price,
            "long_entry_price": self.long_entry_price,
            "long_peak": self.long_peak,
            "long_trail_stop_price": self.long_trail_stop_price,
            "is_long_trail_active": self.is_long_trail_active,
            "short_entry_price": self.short_entry_price,
            "short_trough": self.short_trough,
            "short_trail_stop_price": self.short_trail_stop_price,
            "is_short_trail_active": self.is_short_trail_active,
            "peak_capital": self.peak_capital,
            "max_drawdown": self.max_drawdown,
            "current_capital": self.current_capital,  # free balance alias, kept for backward compatibility
            "free_balance": self.current_capital,
            "total_equity": self.total_equity,
            "last_processed_kline_timestamp": (
                self.last_processed_kline_timestamp.isoformat()
                if self.last_processed_kline_timestamp is not None
                else None
            ),
            # trade_log 不建議保存所有歷史，只保存關鍵交易狀態
        }
        try:
            tmp_path = STATE_FILE + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, STATE_FILE)
            print(f"策略狀態已保存到 {STATE_FILE}")
            return True
        except Exception as e:
            print(f"保存策略狀態失敗: {e}")
            return False

    # --- 新增：從 JSON 檔案加載策略狀態 ---
    def load_state(self):
        if not os.path.exists(STATE_FILE):
            return False
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)

            self.position_size = state.get("position_size", 0)
            self.entry_price = state.get("entry_price", 0)
            self.long_entry_price = state.get("long_entry_price")
            self.long_peak = state.get("long_peak")
            self.long_trail_stop_price = state.get("long_trail_stop_price")
            self.is_long_trail_active = state.get("is_long_trail_active", False)
            self.short_entry_price = state.get("short_entry_price")
            self.short_trough = state.get("short_trough")
            self.short_trail_stop_price = state.get("short_trail_stop_price")
            self.is_short_trail_active = state.get("is_short_trail_active", False)
            # Read persisted balances without eagerly hitting the exchange:
            # state.get(key, self._get_free_balance()) evaluates the default on
            # every call, so it used to make an API request even when the key
            # existed — and made offline restarts depend on connectivity.
            self.peak_capital = state.get("peak_capital")
            if self.peak_capital is None:
                self.peak_capital = self._get_free_balance()  # 如果檔案中沒有，則初始化
            self.max_drawdown = state.get("max_drawdown", 0.0)
            # Self-heal a corrupted drawdown (e.g. an old bad balance read that
            # ratcheted it to ~100%). It is a monitoring stat, not a trading input,
            # so resetting it is safe and avoids reporting a fake near-total loss.
            if not isinstance(self.max_drawdown, (int, float)) or not (
                0.0 <= self.max_drawdown < MAX_PLAUSIBLE_DRAWDOWN
            ):
                print(
                    f"⚠️ 偵測到異常的 max_drawdown={self.max_drawdown}，重置為 0.0（疑似舊版壞讀數污染）。"
                )
                self.max_drawdown = 0.0
            self.current_capital = state.get("free_balance")
            if self.current_capital is None:
                self.current_capital = state.get("current_capital")
            if self.current_capital is None:
                self.current_capital = self._get_free_balance()
            self.total_equity = state.get("total_equity", self.current_capital)
            last_ts = state.get("last_processed_kline_timestamp")
            self.last_processed_kline_timestamp = (
                pd.to_datetime(last_ts) if last_ts else None
            )

            return True
        except Exception as e:
            print(f"加載策略狀態失敗: {e}")
            return False


        # --- 新增：與交易所同步校正 JSON 狀態（可定期呼叫） ---
    def sync_state_with_exchange(self, reason="scheduled hourly check"):
            """從交易所讀取實際持倉，並校正本地 JSON 狀態。
            - 會同步：持倉方向/數量、進場價位、資金餘額
            - 若已無持倉，會清空本地進場相關欄位
            - 僅在偵測到變更時才保存與打印，以減少噪音
            """
            try:
                changed = False

                # 同步資金
                try:
                    self._refresh_account_balances()
                except Exception:
                    pass

                # 同步持倉與進場價
                actual_position = self._get_current_position_size()
                if actual_position == 0:
                    # 若實際無持倉，但本地仍有記錄，則重置
                    if (
                        self.position_size != 0
                        or self.long_entry_price is not None
                        or self.short_entry_price is not None
                    ):
                        self.position_size = 0
                        self.entry_price = 0
                        # 清空多單狀態
                        self.long_entry_price = None
                        self.long_peak = None
                        self.long_trail_stop_price = None
                        self.is_long_trail_active = False
                        # 清空空單狀態
                        self.short_entry_price = None
                        self.short_trough = None
                        self.short_trail_stop_price = None
                        self.is_short_trail_active = False
                        changed = True
                elif actual_position > 0:
                    # 多單持倉
                    avg = (
                        self._get_position_avg_price("long")
                        or self.long_entry_price
                        or self.entry_price
                        or 0
                    )
                    if (
                        self.position_size != actual_position
                        or not self.long_entry_price
                        or abs((self.long_entry_price or 0) - avg) > 1e-9
                        or self.short_entry_price is not None
                    ):
                        self.position_size = actual_position
                        self.entry_price = avg
                        self.long_entry_price = avg
                        # 清空空單狀態避免殘留
                        self.short_entry_price = None
                        self.short_trough = None
                        self.short_trail_stop_price = None
                        self.is_short_trail_active = False
                        changed = True
                else:
                    # 空單持倉
                    avg = (
                        self._get_position_avg_price("short")
                        or self.short_entry_price
                        or self.entry_price
                        or 0
                    )
                    if (
                        self.position_size != actual_position
                        or not self.short_entry_price
                        or abs((self.short_entry_price or 0) - avg) > 1e-9
                        or self.long_entry_price is not None
                    ):
                        self.position_size = actual_position
                        self.entry_price = avg
                        self.short_entry_price = avg
                        # 清空多單狀態避免殘留
                        self.long_entry_price = None
                        self.long_peak = None
                        self.long_trail_stop_price = None
                        self.is_long_trail_active = False
                        changed = True

                if changed:
                    self.save_state()
                    try:
                        # 僅在變更時輸出一行簡訊息，避免干擾
                        side = "LONG" if self.position_size > 0 else ("SHORT" if self.position_size < 0 else "FLAT")
                        entry = self.long_entry_price if self.position_size > 0 else (self.short_entry_price if self.position_size < 0 else 0)
                        print(f"🛠️ 已校正JSON狀態（{reason}）| 狀態: {side}, 持倉: {self.position_size:.5f}, 進場價: {entry}")
                    except Exception:
                        pass
                if self.position_size != 0:
                    self._sync_exchange_protective_stop(reason)
                return True
            except Exception as e:
                print(f"⚠️ 校正JSON狀態失敗: {e}")
                return False

    def _compute_entry_signals(self, current_bar):
        """集中計算原始多空訊號與 side-mode 過濾後的新開倉訊號。"""
        current_close = current_bar["close"]
        current_high = current_bar["high"]
        current_low = current_bar["low"]
        current_adx = current_bar["adx"]
        current_rsi = current_bar["rsi"]

        raw_long = (
            current_close > current_bar["ema90"]
            and current_low > current_bar["ema90"]
            and current_close > current_bar["ema200"]
            and current_adx > self.long_adx_threshold
            and current_rsi <= 70
        )
        raw_short = (
            current_close < current_bar["ema90"]
            and current_high < current_bar["ema90"]
            and current_close < current_bar["ema200"]
            and current_adx > self.short_adx_threshold
            and current_rsi >= 30
        )
        return {
            "raw_long": bool(raw_long),
            "raw_short": bool(raw_short),
            "long": bool(raw_long and self.allow_long_entries),
            "short": bool(raw_short and self.allow_short_entries),
        }

    def process_bar(self, current_bar):
        current_time = current_bar.name
        current_close = current_bar["close"]
        current_high = current_bar["high"]
        current_low = current_bar["low"]
        current_adx = current_bar["adx"]
        recent_vol = current_bar.get("ret_vol") if hasattr(current_bar, "get") else None

        # 檢查關鍵數據是否為None
        if current_close is None or current_high is None or current_low is None:
            print(
                f"❌ 價格數據不完整: close={current_close}, high={current_high}, low={current_low}"
            )
            return

        if current_adx is None:
            print(f"❌ ADX數據不完整: {current_adx}")
            return

        if (
            pd.isna(current_bar["ema90"])
            or pd.isna(current_bar["ema200"])
            or pd.isna(current_adx)
        ):
            print(f"數據不足以計算指標在 {current_time}，跳過。")
            return

        # === 📊 關鍵指標報告 ===
        # 將 UTC K 線時間轉換為台北時間 (UTC+8)
        local_time = current_time + timedelta(hours=8)

        print(
            f"\n\n� 新K線: {local_time.strftime('%Y-%m-%d %H:%M:%S')} | 價格: ${current_close:.2f}"
        )

        # 技術指標（一行顯示）
        print(
            f"📈 技術指標 | EMA90: ${current_bar['ema90']:.2f} | EMA200: ${current_bar['ema200']:.2f} | ADX: {current_adx:.2f} | RSI: {current_bar['rsi']:.2f}"
        )

        signals = self._compute_entry_signals(current_bar)
        raw_long_entry_condition = signals["raw_long"]
        raw_short_entry_condition = signals["raw_short"]
        long_entry_condition = signals["long"]
        short_entry_condition = signals["short"]

        print(
            f"🎯 進場信號 | 模式: {self.trade_side_mode} | "
            f"多單: {'✅' if long_entry_condition else '❌'} | "
            f"空單: {'✅' if short_entry_condition else '❌'}"
        )

        # --- 🔧 修正：先更新當前資金和持倉狀態，再顯示持倉資訊 ---
        self._refresh_account_balances()
        self.position_size = self._get_current_position_size()

        # 簡化持倉和盈虧分析（使用更新後的持倉資訊）
        if self.position_size != 0:
            if self.position_size > 0:  # 多單
                entry_price = self.long_entry_price
                current_profit_usd = (
                    (current_close - entry_price) * self.position_size
                    if entry_price
                    else 0
                )
                current_profit_percent = (
                    ((current_close - entry_price) / entry_price * 100)
                    if entry_price
                    else 0
                )

                print(
                    f"\n📋 當前持倉: 多單 {self.position_size} ETH | 進場: ${entry_price:.2f} | 盈虧: {current_profit_percent:+.2f}%"
                )

                # 停損設置（簡化）
                if entry_price:
                    fixed_stop = entry_price * (1 - self.long_fixed_stop_loss_percent)
                    trail_stop = self.long_trail_stop_price
                    trail_status = "已激活" if self.is_long_trail_active else "未激活"

                    print(
                        f"🛡️ 停損設置: 固定 ${fixed_stop:.2f} | 移動停損: {trail_status}"
                    )

            else:  # 空單
                entry_price = self.short_entry_price
                abs_position = abs(self.position_size)
                current_profit_usd = (
                    (entry_price - current_close) * abs_position if entry_price else 0
                )
                current_profit_percent = (
                    ((entry_price - current_close) / entry_price * 100)
                    if entry_price
                    else 0
                )

                print(
                    f"\n📋 當前持倉: 空單 {abs_position} ETH | 進場: ${entry_price:.2f} | 盈虧: {current_profit_percent:+.2f}%"
                )

                # 停損設置（簡化）
                if entry_price:
                    fixed_stop = entry_price * (1 + self.short_fixed_stop_loss_percent)
                    trail_stop = self.short_trail_stop_price
                    trail_status = "已激活" if self.is_short_trail_active else "未激活"

                    print(
                        f"🛡️ 停損設置: 固定 ${fixed_stop:.2f} | 移動停損: {trail_status}"
                    )
        else:
            print(f"\n📋 當前持倉: 無持倉")

        # 帳戶狀態（簡化）
        print(
            f"💰 帳戶狀態: 未使用資金 {self.current_capital:.2f} USDT"
        )

        if self.position_size == 0:
            trade_qty = self._prepare_entry_quantity(current_close, recent_vol)
        else:
            trade_qty = 0

        # --- 處理多單邏輯 ---
        if self.position_size == 0:
            if long_entry_condition and float(trade_qty) > 0:
                print(f"{current_time} - 觸發多單進場條件。")
                entry_stop_price = current_close * (1 - self.long_fixed_stop_loss_percent)
                order = self._place_order(
                    "buy", trade_qty, "market", stop_loss_price=entry_stop_price
                )
                if order:
                    # 🔧 修正：下單後等待並查詢實際持倉來獲取真實進場價
                    time.sleep(2)

                    # 重新查詢持倉以獲取實際數量和平均價格
                    actual_position = self._get_current_position_size()
                    if actual_position > 0:
                        self.position_size = actual_position
                        # 從持倉資訊中獲取實際進場價格
                        actual_entry_price = self._get_position_avg_price("long")
                        if actual_entry_price and actual_entry_price > 0:
                            self.entry_price = actual_entry_price
                            self.long_entry_price = self.entry_price
                        else:
                            # 如果無法獲取實際價格，使用當前收盤價
                            self.entry_price = current_close
                            self.long_entry_price = self.entry_price
                    else:
                        # 如果查詢不到持倉，使用訂單資訊
                        self.position_size = trade_qty
                        fallback_price = order.get("price", current_close)
                        self.entry_price = (
                            fallback_price
                            if isinstance(fallback_price, (int, float))
                            else current_close
                        )
                        self.long_entry_price = self.entry_price

                    self.long_peak = max(self.long_peak or current_high, current_high)
                    if not self.is_long_trail_active:
                        self.long_trail_stop_price = None
                    self._sync_exchange_protective_stop("多單進場後保護停損")
                    print(
                        f"多單已進場，數量: {self.position_size:.3f} @ {self.entry_price:.2f}"
                    )
                    self.save_state()

        elif self.position_size > 0:
            # 🔧 修正：確保有進場價格才能執行停損邏輯
            if self.long_entry_price is None or self.long_entry_price <= 0:
                print(f"⚠️ 警告：檢測到多單持倉但無進場價格記錄，無法執行停損！")
                print(f"   建議手動檢查持倉或重啟程式以重新同步狀態")
                return

            # 確保long_peak不為None (移動停損需要)
            if self.long_peak is None:
                self.long_peak = current_high

            else:
                self.long_peak = max(self.long_peak, current_high)

            # 計算當前盈虧百分比
            current_profit_percent = (
                current_close - self.long_entry_price
            ) / self.long_entry_price

            # 計算固定停損價格
            long_fixed_stop_loss_price = self.long_entry_price * (
                1 - self.long_fixed_stop_loss_percent
            )

            # 固定停損觸發條件 - 用 4H K 線最低點判斷（對齊回測 bar_low <= fixed_stop）
            long_fixed_stop_loss_triggered = current_low <= long_fixed_stop_loss_price

            # 添加詳細的固定停損檢查日誌
            print(f"\n📊 多單固定停損檢查:")
            print(f"   固定停損價格: {long_fixed_stop_loss_price:.2f}")
            print(f"   K線最低: {current_low:.2f} | 收盤: {current_close:.2f}")

            # 顯示移動停損詳細信息
            if self.is_long_trail_active:
                peak_str = f"${self.long_peak:.2f}" if self.long_peak else "N/A"
                trail_price_str = (
                    f"${self.long_trail_stop_price:.2f}"
                    if self.long_trail_stop_price
                    else "N/A"
                )
                print(f"   追蹤峰值: {peak_str}，保護停損價格: {trail_price_str}")
            else:
                print(f"   移動停損狀態: 未激活")

            if long_fixed_stop_loss_triggered:
                print(
                    f"🚨 固定止損觸發: K線最低${current_low:.2f} <= 固定止損價${long_fixed_stop_loss_price:.2f}"
                )

                print(f"\n🚨 === 多單固定停損平倉觸發 ===")
                print(f"時間: {current_time}")
                print(f"觸發原因: FIXED_STOP")
                print(f"觸發極值: ${current_low:.2f} (close ${current_close:.2f})")
                print(f"持倉量: {self.position_size}")
                print(f"進場價: ${self.long_entry_price:.2f}")

                close_success = self._close_position(current_close)
                if not close_success:
                    print(f"❌ 多單固定停損平倉失敗，請檢查")
                else:
                    print(f"✅ 多單固定停損平倉完成")
                return

            if self.exit_on_reverse_signal and raw_short_entry_condition:
                print(f"🔄 多單遇到反向空方訊號，先平倉退出。")
                close_success = self._close_position(current_close)
                if not close_success:
                    print("❌ 反向訊號平多失敗，請檢查")
                    return
                print("✅ 反向訊號平多完成")
                if not self.allow_same_bar_reversal:
                    return
                self._refresh_account_balances()
                self.position_size = self._get_current_position_size()
                if self.position_size != 0:
                    print("⚠️ 平多後仍偵測到持倉，取消同根反手。")
                    return
                trade_qty = self._prepare_entry_quantity(current_close, recent_vol)

        # --- 處理空單邏輯 ---
        if self.position_size == 0:
            if short_entry_condition and float(trade_qty) > 0:
                print(f"{current_time} - 觸發空單進場條件。")
                entry_stop_price = current_close * (1 + self.short_fixed_stop_loss_percent)
                order = self._place_order(
                    "sell", trade_qty, "market", stop_loss_price=entry_stop_price
                )
                if order:
                    # 🔧 修正：下單後等待並查詢實際持倉來獲取真實進場價
                    print("⏳ 等待2秒後查詢實際持倉資訊...")
                    time.sleep(2)

                    # 重新查詢持倉以獲取實際數量和平均價格
                    actual_position = self._get_current_position_size()
                    if actual_position < 0:
                        self.position_size = actual_position
                        # 從持倉資訊中獲取實際進場價格
                        actual_entry_price = self._get_position_avg_price("short")
                        if actual_entry_price and actual_entry_price > 0:
                            self.entry_price = actual_entry_price
                            self.short_entry_price = self.entry_price
                            print(f"✅ 獲取實際進場價格: ${self.entry_price:.2f}")
                        else:
                            # 如果無法獲取實際價格，使用當前收盤價
                            self.entry_price = current_close
                            self.short_entry_price = self.entry_price
                            print(f"⚠️ 無法獲取實際進場價格，使用當前收盤價: ${current_close:.2f}")
                    else:
                        # 如果查詢不到持倉，使用訂單資訊
                        self.position_size = -trade_qty
                        fallback_price = order.get("price", current_close)
                        self.entry_price = (
                            fallback_price
                            if isinstance(fallback_price, (int, float))
                            else current_close
                        )
                        self.short_entry_price = self.entry_price
                        print(f"⚠️ 查詢持倉失敗，使用訂單資訊: ${self.entry_price:.2f}")

                    self.short_trough = min(self.short_trough or current_low, current_low)
                    if not self.is_short_trail_active:
                        self.short_trail_stop_price = None
                    self._sync_exchange_protective_stop("空單進場後保護停損")
                    print(
                        f"空單已進場，數量: {abs(self.position_size):.3f} @ {self.entry_price:.2f}"
                    )
                    self.save_state()

        elif self.position_size < 0:
            # 🔧 修正：確保有進場價格才能執行停損邏輯
            if self.short_entry_price is None or self.short_entry_price <= 0:
                print(f"⚠️ 警告：檢測到空單持倉但無進場價格記錄，無法執行停損！")
                print(f"   建議手動檢查持倉或重啟程式以重新同步狀態")
                return

            # 確保short_trough不為None (移動停損需要)
            if self.short_trough is None:
                self.short_trough = current_low

            else:
                self.short_trough = min(self.short_trough, current_low)

            # 計算當前盈虧百分比用於調試
            current_profit_percent = (
                self.short_entry_price - current_close
            ) / self.short_entry_price
            print(
                f"\n📊 空單狀態: 進場價${self.short_entry_price:.2f}, 當前價${current_close:.2f}, 盈虧{current_profit_percent*100:.2f}%"
            )

            # 計算固定停損價格
            short_fixed_stop_loss_price = self.short_entry_price * (
                1 + self.short_fixed_stop_loss_percent
            )

            # 固定停損觸發條件 - 用 4H K 線最高點判斷（對齊回測 bar_high >= fixed_stop）
            short_fixed_stop_loss_triggered = (
                current_high >= short_fixed_stop_loss_price
            )

            # 添加詳細的固定停損檢查日誌
            print(f"📊 空單固定停損檢查:")
            print(f"   固定停損價格: {short_fixed_stop_loss_price:.2f}")
            print(f"   K線最高: {current_high:.2f} | 收盤: {current_close:.2f}")

            # 顯示移動停損詳細信息
            if self.is_short_trail_active:
                trough_str = f"${self.short_trough:.2f}" if self.short_trough else "N/A"
                trail_price_str = (
                    f"${self.short_trail_stop_price:.2f}"
                    if self.short_trail_stop_price
                    else "N/A"
                )
                print(f"   追蹤谷值: {trough_str}，保護停損價格: {trail_price_str}")
            else:
                print(f"   移動停損狀態: 未激活")

            if short_fixed_stop_loss_triggered:
                print(
                    f"🚨 固定止損觸發: K線最高${current_high:.2f} >= 固定止損價${short_fixed_stop_loss_price:.2f}"
                )

                print(f"\n🚨 === 空單固定停損平倉觸發 ===")
                print(f"時間: {current_time}")
                print(f"觸發原因: FIXED_STOP")
                print(f"觸發極值: ${current_high:.2f} (close ${current_close:.2f})")
                print(f"持倉量: {self.position_size}")
                print(f"進場價: ${self.short_entry_price:.2f}")

                close_success = self._close_position(current_close)
                if not close_success:
                    print(f"❌ 空單固定停損平倉失敗，請檢查")
                else:
                    print(f"✅ 空單固定停損平倉完成")
                return

            if self.exit_on_reverse_signal and raw_long_entry_condition:
                print(f"🔄 空單遇到反向多方訊號，先平倉退出。")
                close_success = self._close_position(current_close)
                if not close_success:
                    print("❌ 反向訊號平空失敗，請檢查")
                    return
                print("✅ 反向訊號平空完成")
                if not self.allow_same_bar_reversal:
                    return
                self._refresh_account_balances()
                self.position_size = self._get_current_position_size()
                if self.position_size != 0:
                    print("⚠️ 平空後仍偵測到持倉，取消同根反手。")
                    return
                trade_qty = self._prepare_entry_quantity(current_close, recent_vol)

        # --- 更新資金和回撤計算 ---
        try:
            _free_balance, total_equity = self._refresh_account_balances()
            # Only feed plausible equity samples into the peak/drawdown stats. A
            # partial balance snapshot (near-zero equity) would otherwise push
            # max_drawdown to ~100% permanently, since it only ever ratchets up.
            if total_equity >= MIN_PLAUSIBLE_EQUITY_USDT:
                self.peak_capital = max(self.peak_capital, total_equity)
                if self.peak_capital > 0:
                    current_drawdown = (
                        self.peak_capital - total_equity
                    ) / self.peak_capital
                    if current_drawdown < MAX_PLAUSIBLE_DRAWDOWN:
                        self.max_drawdown = max(self.max_drawdown, current_drawdown)
                    else:
                        print(
                            f"⚠️ 略過異常回撤樣本 ({current_drawdown*100:.2f}%)："
                            f"權益讀數 {total_equity:.4f} 可能為不完整快照。"
                        )
            else:
                print(
                    f"⚠️ 略過異常權益讀數 {total_equity:.4f} USDT，不更新回撤統計。"
                )
        except Exception as e:
            print(f"更新實時資金和回撤失敗: {e}")

        self.save_state()  # 每處理完一根K線都保存一次狀態，確保最新狀態被記錄

    def check_trailing_stop_only(self):
        """
        每分鐘檢查本地固定停損兜底與移動停損，不處理進場邏輯
        靜默執行，只在重要事件時打印日誌
        """
        if self.position_size == 0:
            return  # 無持倉時不需要檢查

        try:
            # 獲取當前價格（使用1分鐘K線的最新數據）
            df_1m = fetch_bybit_klines(SYMBOL, "1m", limit=2)
            if df_1m.empty or len(df_1m) < 1:
                # 靜默跳過，不打印錯誤信息
                return

            current_bar_1m = df_1m.iloc[-1]  # 最新的1分鐘K線
            current_close = current_bar_1m["close"]
            current_high = current_bar_1m["high"]
            current_low = current_bar_1m["low"]
            current_time = current_bar_1m.name

            # 檢查關鍵數據是否為None
            if current_close is None or current_high is None or current_low is None:
                # 靜默跳過，不打印錯誤信息
                return

            # 靜默執行，不打印常規檢查信息

            # --- 處理多單移動停損 ---
            if self.position_size > 0:
                if self.long_entry_price is None or self.long_entry_price <= 0:
                    return  # 靜默跳過
                if self._handle_local_fixed_stop_fallback(
                    "long", current_low, current_high, current_close, current_time
                ):
                    return

                # 步驟 1：先用「上一輪結束時」的 trail stop 檢查觸發，對齊回測
                # check_stop_exit → update_trailing_stop 的順序（避免同一根 K 線啟用 trail 後立刻出場）
                if (
                    self.is_long_trail_active
                    and self.long_trail_stop_price is not None
                    and current_low <= self.long_trail_stop_price
                ):
                    print(f"\n\n🚨 === 多單移動停損觸發 ===")
                    print(f"時間: {current_time}")
                    print(f"觸發極值: ${current_low:.2f} (close ${current_close:.2f})")
                    print(f"移動停損價: ${self.long_trail_stop_price:.2f}")
                    print(f"持倉量: {self.position_size}")

                    close_success = self._close_position(current_close)
                    if close_success:
                        print(f"✅ 多單移動停損平倉完成")
                    else:
                        print(f"❌ 多單移動停損平倉失敗")
                    return  # 觸發後不再更新 trailing state，對齊回測

                # 步驟 2：沒觸發才更新峰值
                if self.long_peak is None:
                    self.long_peak = current_high
                else:
                    old_peak = self.long_peak
                    self.long_peak = max(self.long_peak, current_high)
                    # 只在峰值有顯著更新時才打印（避免頻繁打印）
                    if (
                        self.long_peak > old_peak
                        and (self.long_peak - old_peak) / old_peak > 0.005
                    ):  # 0.5%以上的變化才打印
                        print(
                            f"\n\n📈 多單峰值更新: ${old_peak:.2f} → ${self.long_peak:.2f}"
                        )

                # 步驟 3：沒觸發才檢查是否激活 trail（用 high 對齊回測 bar["high"] >= activate_price）
                if (
                    not self.is_long_trail_active
                    and current_high
                    >= self.long_entry_price
                    * (1 + self.long_trailing_activate_profit_percent)
                ):
                    self.long_trail_stop_price = self.long_entry_price * (
                        1 + self.long_trailing_min_profit_percent
                    )
                    self.is_long_trail_active = True
                    print(
                        f"\n\n✅ 多單移動停損激活 | 初始止損價: ${self.long_trail_stop_price:.2f}"
                    )
                    self._set_exchange_stop_loss(
                        "long", self.long_trail_stop_price, "多單移動停損激活"
                    )
                    self.save_state()

                # 步驟 4：沒觸發才更新 trail stop（下一輪才用到，避免本輪即啟即出）
                if self.is_long_trail_active and self.long_peak is not None:
                    # 計算基於峰值回撤的停損價格
                    new_trail_stop = self.long_peak * (
                        1 - self.long_trailing_pullback_percent
                    )

                    # 🔧 重要修正：確保移動停損價格不低於最小獲利保護
                    min_profit_protection = self.long_entry_price * (
                        1 + self.long_trailing_min_profit_percent
                    )

                    # 移動停損價格取較高者（峰值回撤 vs 最小獲利保護）
                    new_trail_stop = max(new_trail_stop, min_profit_protection)

                    old_trail_stop = self.long_trail_stop_price
                    self.long_trail_stop_price = max(
                        (
                            self.long_trail_stop_price
                            if self.long_trail_stop_price is not None
                            else 0
                        ),
                        new_trail_stop,
                    )
                    # 只在停損價格有顯著更新時才打印
                    if (
                        old_trail_stop
                        and self.long_trail_stop_price > old_trail_stop
                        and (self.long_trail_stop_price - old_trail_stop)
                        / old_trail_stop
                        > 0.003
                    ):  # 0.3%以上的變化才打印
                        print(
                            f"\n\n📊 多單移動停損更新: ${old_trail_stop:.2f} → ${self.long_trail_stop_price:.2f}"
                        )
                        self._set_exchange_stop_loss(
                            "long", self.long_trail_stop_price, "多單移動停損更新"
                        )
                        self.save_state()

            # --- 處理空單移動停損 ---
            elif self.position_size < 0:
                if self.short_entry_price is None or self.short_entry_price <= 0:
                    return  # 靜默跳過
                if self._handle_local_fixed_stop_fallback(
                    "short", current_low, current_high, current_close, current_time
                ):
                    return

                # 步驟 1：先用「上一輪結束時」的 trail stop 檢查觸發，對齊回測
                # check_stop_exit → update_trailing_stop 的順序（避免同一根 K 線啟用 trail 後立刻出場）
                if (
                    self.is_short_trail_active
                    and self.short_trail_stop_price is not None
                    and current_high >= self.short_trail_stop_price
                ):
                    print(f"\n\n🚨 === 空單移動停損觸發 ===")
                    print(f"時間: {current_time}")
                    print(f"觸發極值: ${current_high:.2f} (close ${current_close:.2f})")
                    print(f"移動停損價: ${self.short_trail_stop_price:.2f}")
                    print(f"持倉量: {self.position_size}")

                    close_success = self._close_position(current_close)
                    if close_success:
                        print(f"✅ 空單移動停損平倉完成")
                    else:
                        print(f"❌ 空單移動停損平倉失敗")
                    return  # 觸發後不再更新 trailing state，對齊回測

                # 步驟 2：沒觸發才更新谷值
                if self.short_trough is None:
                    self.short_trough = current_low
                else:
                    old_trough = self.short_trough
                    self.short_trough = min(self.short_trough, current_low)
                    # 只在谷值有顯著更新時才打印（避免頻繁打印）
                    if (
                        self.short_trough < old_trough
                        and (old_trough - self.short_trough) / old_trough > 0.005
                    ):  # 0.5%以上的變化才打印
                        print(
                            f"\n\n📉 空單谷值更新: ${old_trough:.2f} → ${self.short_trough:.2f}"
                        )

                # 步驟 3：沒觸發才檢查是否激活 trail（用 low 對齊回測 bar["low"] <= activate_price）
                if (
                    not self.is_short_trail_active
                    and current_low
                    <= self.short_entry_price
                    * (1 - self.short_trailing_activate_profit_percent)
                ):
                    self.short_trail_stop_price = self.short_entry_price * (
                        1 - self.short_trailing_min_profit_percent
                    )
                    self.is_short_trail_active = True
                    print(
                        f"\n\n✅ 空單移動停損激活 | 初始止損價: ${self.short_trail_stop_price:.2f}"
                    )
                    self._set_exchange_stop_loss(
                        "short", self.short_trail_stop_price, "空單移動停損激活"
                    )
                    self.save_state()

                # 步驟 4：沒觸發才更新 trail stop（下一輪才用到，避免本輪即啟即出）
                if self.is_short_trail_active and self.short_trough is not None:
                    # 計算基於谷值回撤的停損價格
                    new_trail_stop = self.short_trough * (
                        1 + self.short_trailing_pullback_percent
                    )

                    # 🔧 重要修正：確保移動停損價格不高於最小獲利保護
                    min_profit_protection = self.short_entry_price * (
                        1 - self.short_trailing_min_profit_percent
                    )

                    # 移動停損價格取較低者（谷值回撤 vs 最小獲利保護）
                    new_trail_stop = min(new_trail_stop, min_profit_protection)

                    old_trail_stop = self.short_trail_stop_price
                    self.short_trail_stop_price = min(
                        (
                            self.short_trail_stop_price
                            if self.short_trail_stop_price is not None
                            else float("inf")
                        ),
                        new_trail_stop,
                    )
                    # 只在停損價格有顯著更新時才打印
                    if (
                        old_trail_stop
                        and self.short_trail_stop_price < old_trail_stop
                        and (old_trail_stop - self.short_trail_stop_price)
                        / old_trail_stop
                        > 0.003
                    ):  # 0.3%以上的變化才打印
                        print(
                            f"\n\n📊 空單移動停損更新: ${old_trail_stop:.2f} → ${self.short_trail_stop_price:.2f}"
                        )
                        self._set_exchange_stop_loss(
                            "short", self.short_trail_stop_price, "空單移動停損更新"
                        )
                        self.save_state()

        except Exception as e:
            print(f"❌ 移動停損檢查發生錯誤: {e}")


# --- 輔助函數：動態狀態顯示 ---
def calculate_next_kline_time(last_kline_timestamp):
    """計算下次K線時間（4小時週期）"""
    if last_kline_timestamp is None:
        return "未知"

    # 確保 last_kline_timestamp 是 datetime 對象
    if isinstance(last_kline_timestamp, str):
        try:
            last_kline_timestamp = pd.to_datetime(last_kline_timestamp)
        except:
            return "時間格式錯誤"

    # 移除時區資訊，統一使用本地時間
    if hasattr(last_kline_timestamp, "tz") and last_kline_timestamp.tz is not None:
        last_kline_timestamp = last_kline_timestamp.tz_localize(None)

    # 找到當前時間對應的4小時週期
    now = datetime.now()

    # 4小時週期的開始時間點：00:00, 04:00, 08:00, 12:00, 16:00, 20:00
    current_hour = now.hour

    # 計算下一個4小時週期的開始時間
    next_cycle_hours = [0, 4, 8, 12, 16, 20]

    next_kline = None
    for cycle_hour in next_cycle_hours:
        if cycle_hour > current_hour:
            next_kline = now.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
            break

    # 如果沒有找到（當前時間超過20:00），下次週期是明天的00:00
    if next_kline is None:
        next_kline = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    return next_kline


def format_time_remaining(next_kline_time):
    """計算並格式化到下次K線的剩餘時間"""
    if next_kline_time == "未知" or next_kline_time == "時間格式錯誤":
        return next_kline_time

    now = datetime.now()

    # 如果 next_kline_time 有時區資訊，移除它進行比較
    if hasattr(next_kline_time, "tz") and next_kline_time.tz is not None:
        next_kline_time = next_kline_time.tz_localize(None)

    remaining = next_kline_time - now
    total_seconds = remaining.total_seconds()

    if total_seconds <= 0:
        return "應該有新K線了"

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    if hours > 0:
        return f"{hours}時{minutes}分{seconds}秒"
    elif minutes > 0:
        return f"{minutes}分{seconds}秒"
    else:
        return f"{seconds}秒"


def get_spinner_char(counter):
    """獲取旋轉動畫字符"""
    spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    return spinner_chars[counter % len(spinner_chars)]


# --- 主運行邏輯 (實時交易) ---
def run_live_trading():
    """實時交易主函數"""
    # 初始化策略實例 (使用預設最佳參數)
    strategy = TradingStrategy()

    last_kline_timestamp = strategy.last_processed_kline_timestamp
    spinner_counter = 0

    print("\n--- 開始實時交易 ---")
    last_check_time = 0  # 記錄上次檢查K線的時間
    last_trailing_stop_check_time = 0  # 記錄上次檢查移動停損的時間
    TRAILING_STOP_CHECK_SECONDS = 60  # 每60秒檢查一次移動停損

    # 每小時校正一次 JSON 狀態（避免手動干預造成狀態偏移）
    last_state_sync_time = time.time()
    STATE_SYNC_INTERVAL_SECONDS = 3600


    while True:
        try:
            current_time = time.time()

            # 每分鐘檢查一次移動停損
            if (
                current_time - last_trailing_stop_check_time
                >= TRAILING_STOP_CHECK_SECONDS
            ):
                # 只有在有持倉時才檢查移動停損
                if strategy.position_size != 0:
                    # 靜默執行移動停損檢查，不打印額外日誌
                    strategy.check_trailing_stop_only()
                last_trailing_stop_check_time = current_time

            # 每小時與交易所同步一次狀態，校正JSON（進場價/方向/數量）
            if current_time - last_state_sync_time >= STATE_SYNC_INTERVAL_SECONDS:
                try:
                    strategy.sync_state_with_exchange(reason="每小時校正")
                finally:
                    last_state_sync_time = current_time

            # 每60秒檢查一次K線數據
            if current_time - last_check_time >= TRADE_SLEEP_SECONDS:
                # 獲取最新 K 線數據
                df_klines = fetch_bybit_klines(
                    SYMBOL, TIMEFRAME, limit=FETCH_KLINE_LIMIT
                )

                if df_klines.empty:
                    print("\n\n❌ 未獲取到 K 線數據，等待下一週期...")
                    last_check_time = current_time
                    continue

                df_processed = calculate_indicators(df_klines.copy())

                # 確保 df_processed 至少有一根已完成 K 線
                if len(df_processed) < 1:
                    print(
                        f"\n\n⚠️ 數據不足，至少需要1根完整K線。當前僅有 {len(df_processed)} 根。"
                    )
                    last_check_time = current_time
                    continue

                current_bar = get_latest_completed_bar(df_processed, TIMEFRAME)
                if current_bar is None:
                    print("\n\n⚠️ 尚未確認有已收完的 4H K 線，等待下一輪檢查。")
                    last_check_time = current_time
                    continue

                # 如果是第一次運行或有新的K線形成
                if (
                    last_kline_timestamp is None
                    or current_bar.name > last_kline_timestamp
                ):
                    # 先換行，避免覆蓋動態狀態行
                    # 將 UTC 時間轉換為台北時間顯示
                    kline_taipei_time = current_bar.name + timedelta(hours=8)
                    print(f"\n\n🔔 檢測到新 4小時 K 線: {kline_taipei_time}")
                    print(f"⏰ 開始技術分析和交易判斷...")

                    # 將最新完成的 K 線傳入策略進行處理
                    strategy.process_bar(current_bar)
                    strategy.last_processed_kline_timestamp = current_bar.name
                    strategy.save_state()
                    last_kline_timestamp = current_bar.name

                last_check_time = current_time

            # 靜默等待，不顯示任何狀態更新
            spinner_counter += 1

            time.sleep(1)  # 每秒更新一次顯示

        except Exception as e:
            print(f"\n\n❌ 主循環發生錯誤: {e}")
            time.sleep(TRADE_SLEEP_SECONDS * 2)  # 錯誤時等待更久，避免頻繁報錯


# --- 主程式入口 ---
if __name__ == "__main__":
    print("🏆 ETH 4小時自動交易策略啟動 (移動停損優化版)")
    print("📅 版本更新日期: 2025/8/24")
    print("🔧 新功能: RSI計算改為Wilder's方法(與TradingView一致)")
    print("⚠️ 實時交易模式 | 請確認風險")

    # 處理非互動模式
    try:
        input("請確認您已理解風險並準備好，按 Enter 鍵繼續...")
    except EOFError:
        print("檢測到非互動模式，自動確認繼續...")
        time.sleep(2)

    run_live_trading()
