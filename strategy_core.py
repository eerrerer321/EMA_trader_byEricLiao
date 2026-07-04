"""
strategy_core — 策略共用核心(指標 / 進場訊號 / 資金費率 / K線抓取 / 配置常數)。

原本位於 eth_strategy_4h_autotrading.py(實時交易引擎),為了讓多腿版 run_mixed_live.py
與 backtest_eth_strategy_4h.py 能在淘汰該引擎後獨立運作,將「live 與回測共用的單一事實
來源」逐字抽出至此。內容與原檔一致,未更動任何邏輯。
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

# === 深負費率加碼 (funding boost) ===
# BTC 幣本位 3 日均資金費率 < -0.01%/8h（空方深度擁擠 = 軋空燃料充足）時，
# 新開多單倉位 × 1.5；與波動度係數相乘後 cap 在 VOL_SCALE_MAX，不放大已驗證
# 的曝險包絡。閾值 -0.01% 取自費率假說檢驗的 deep_neg 分桶邊界（該情境 BTC
# 30 天前瞻 +17.5% vs 無條件 +5.1%），非回測擬合值；寬鬆版（<0 就加碼）實測
# 反而傷 Sharpe/MDD，已否決。實證（含過熱過濾的基線 → 加上 boost）：
#   ETH Bybit 21-26: 182.8%→188.7%，ETH Binance 18-26: 281%→298%，
#   BTC 18-26: 511%→637%；三資料集 MDD 完全不變、PF/Sharpe 持平或升。
FUNDING_BOOST_ENABLED = True
FUNDING_BOOST = (
    {"threshold": -0.0001, "factor": 1.5, "cap": VOL_SCALE_MAX}
    if FUNDING_BOOST_ENABLED
    else None
)

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
        # 多頭擁擠過濾：BTC 幣本位 3 日平滑資金費率 > 0.01%/8h（Bybit 標準費率
        # 上緣，非擬合值）時不開新多單。實證：該情境的多單 PF 僅 0.81（虧錢），
        # 過濾後 ETH Bybit 21-26 PF 1.58→1.72、滾動獲利率 69→73%；BTC 18-26
        # 報酬 330%→511%、MDD -32→-27。訊號bar缺 btc_funding_3d 欄位時自動旁路。
        "max_btc_funding_3d": 0.0001,
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

    # 空表防護：行情抓取瞬斷時 fetch_bybit_klines 會回傳空表（已自行印出網路錯誤）。
    # 此處原樣返回，避免在 df["close"] 上拋出隱晦的 KeyError('close')；下游
    # get_latest_completed_bar(空表) 會回 None，呼叫端的 if bar is not None 自動跳過本輪。
    if df.empty or "close" not in df.columns:
        return df

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


# --- 2.5 進場訊號（live 與回測共用的單一事實來源） ---
def long_signal(bar, params):
    """趨勢多單進場訊號。bar 需含 ema90/ema200/adx/rsi 欄位。

    含選配的「多頭擁擠」過濾：BTC 幣本位 3 日平滑資金費率高於
    params["max_btc_funding_3d"] 時不開新多單（實證該情境多單 PF 僅 0.81）。
    params 無此鍵或 bar 無 btc_funding_3d 欄位時自動旁路。
    """
    adx_threshold = params.get("long_adx_threshold", params["adx_threshold"])
    base = bool(
        bar["close"] > bar["ema90"]
        and bar["low"] > bar["ema90"]
        and bar["close"] > bar["ema200"]
        and bar["adx"] > adx_threshold
        and bar["rsi"] <= 70
    )
    if not base:
        return False
    max_funding = params.get("max_btc_funding_3d")
    if max_funding is not None:
        funding = bar.get("btc_funding_3d")
        if funding is not None and not pd.isna(funding) and funding > max_funding:
            return False
    return True


def short_signal(bar, params):
    """趨勢空單進場訊號。bar 需含 ema90/ema200/adx/rsi 欄位。"""
    adx_threshold = params.get("short_adx_threshold", params["adx_threshold"])
    return bool(
        bar["close"] < bar["ema90"]
        and bar["high"] < bar["ema90"]
        and bar["close"] < bar["ema200"]
        and bar["adx"] > adx_threshold
        and bar["rsi"] >= 30
    )


def apply_funding_boost(base_scale, btc_funding_3d, boost):
    """深負費率加碼：費率低於 boost["threshold"] 時 base_scale × factor，cap 限制。

    live 與回測共用（單一事實來源）。boost=None 或費率缺失/非數值時原樣返回。
    """
    if not boost or btc_funding_3d is None:
        return base_scale
    try:
        f = float(btc_funding_3d)
    except (TypeError, ValueError):
        return base_scale
    if pd.isna(f) or f >= boost.get("threshold", -0.0001):
        return base_scale
    return float(min(base_scale * boost.get("factor", 1.5), boost.get("cap", VOL_SCALE_MAX)))


# 最近一次成功讀到的 BTC 3 日均費率 (epoch_seconds, value)，供讀取失敗時沿用。
_LAST_GOOD_BTC_FUNDING = None
# 3 日平滑費率數小時內幾乎不動，沿用 last-good 的最大容許陳舊時間。
BTC_FUNDING_MAX_STALE_HOURS = 12


def fetch_btc_funding_3d(retries=3, backoff_seconds=2.0):
    """BTC 幣本位 3 日平滑資金費率（最近 9 筆 8h 均值），供 long_signal 擁擠過濾。

    公開端點、無需 API key。實測 4h K 線收線時多 bot 同時打 API，funding 端點
    偶發瞬時失敗（時段相依，受控探針打不出來），故：
      1. 瞬時失敗重試（retries 次，間隔 backoff_seconds）；
      2. 全部重試仍失敗 → 沿用近期 last-good 值（3 日均費率數小時內幾乎不動），
         避免讀取失敗時「靜默停用過濾」這個 fail-open 風控缺口；
      3. 連 last-good 都沒有（或過舊）才回 None，由呼叫端旁路。
    """
    global _LAST_GOOD_BTC_FUNDING
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            exchange = _get_public_exchange()
            hist = exchange.fetch_funding_rate_history("BTC/USD:BTC", limit=9)
            rates = [float(r["fundingRate"]) for r in hist if r.get("fundingRate") is not None]
            if len(rates) >= 3:
                val = sum(rates) / len(rates)
                _LAST_GOOD_BTC_FUNDING = (time.time(), val)
                return val
            last_err = "資料筆數不足"
        except Exception as e:
            last_err = e
        if attempt < retries:
            time.sleep(backoff_seconds)
    # 全部重試失敗 → 沿用近期 last-good（風控不靜默停用）
    if _LAST_GOOD_BTC_FUNDING is not None:
        ts, val = _LAST_GOOD_BTC_FUNDING
        age_h = (time.time() - ts) / 3600
        if age_h <= BTC_FUNDING_MAX_STALE_HOURS:
            print(f"⚠️ BTC 費率讀取失敗（{last_err}），沿用 {age_h:.1f}h 前的值 {val*100:+.4f}%/8h。")
            return val
    print(f"⚠️ 讀取 BTC 資金費率失敗（{last_err}）且無可用近期值，本根 K 線跳過擁擠過濾。")
    return None


