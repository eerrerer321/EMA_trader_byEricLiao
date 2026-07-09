# 🏆 ETH 4小時自動交易策略

> **基於技術分析的以太坊自動交易機器人 - 實時交易版本**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Bybit](https://img.shields.io/badge/Exchange-Bybit-orange.svg)](https://www.bybit.com)

## 📊 策略績效表現

### 🎯 優化期間表現 (2020-2025年)
- **穩定性評分**: 48.38/60 (優秀等級)
- **總獲利**: 5,371.25 USDT (537.13%)
- **年化收益率**: 102.3%
- **平均勝率**: 54.74%
- **最大回撤**: 14.06%
- **獲利季度**: 21/22 (95.45%)

### 📈 驗證期間表現 (2017-2019年)
- **穩定性評分**: 28.38/60 (一般等級)
- **總獲利**: 1,262.19 USDT (126.22%)
- **年化收益率**: 42.1%
- **平均勝率**: 50.0%
- **最大回撤**: 37.1%

### 🔬 真實 Bybit 永續資料回測 (2026 更新)

> 以下使用**實盤交易標的**（Bybit ETH/USDT 永續）真實 4 小時 K 線重新驗證。
> 條件：每筆 **20% 資金 × 3 倍名目槓桿**（名目曝險約 60%）、**複利**、手續費 0.06% + 滑價 0.02%。
> 報酬以「最終權益 ÷ 初始權益」計算，貼近實盤真實表現（較保守，不含早期牛市）。

**ETH `long_only` 核心策略 (2021-10 ~ 2026-05，無風控 overlay)**

| 總報酬 | 年化(CAGR) | 最大回撤 | 交易數 | 勝率 | Profit Factor | Sharpe |
|---|---|---|---|---|---|---|
| 89.2% | 14.8% | −17.3% | 145 | 55.9% | 1.47 | 0.91 |

**ETH `long_only` 實盤完整配置 (2021-10 ~ 2026-06)：核心策略 + 波動度目標倉位 + BTC 費率過濾/加碼**

| 總報酬 | 年化(CAGR) | 最大回撤 | 交易數 | 勝率 | Profit Factor | Sharpe | 滾動視窗獲利率 |
|---|---|---|---|---|---|---|---|
| 188.7% | ~25% | −20.0% | 119 | 57.1% | 1.73 | 1.21 | 73.1% |

> 三個 overlay 各自經 2-3 個獨立資料集 + 滾動視窗驗證後逐一加入（2026-06），
> 居功最大者為波動度目標倉位與 BTC 資金費率「多頭擁擠」過濾，詳見對應 commit 訊息。
> 進階：`harmonic/run_mixed_live.py` 混合實盤（趨勢+諧波+費率抄底三腿）。
> ⚠️ 2026-07 諧波回測誠實化（進場根停損，cfda183）後重驗：Bybit 場地上諧波腿報酬
> 幾乎打平但回撤更深（趨勢+抄底雙腿 Calmar 1.49 vs 三腿 1.25），抄底腿則穩定加分。
> 腿級開關與證據見 [harmonic/README.md](harmonic/README.md)。

**純樣本外驗證 (2017-2021，參數從未在此期間優化；Binance ETH)**
- 12 個月滾動視窗：**76% 視窗獲利、中位年化 ~20%**，與優化期間一致 → 策略**非過擬合**
- 6 個月短視窗較敏感（45% 獲利）：趨勢策略需要足夠持有時間讓厚尾獲利展開

**跨標的 (同一套參數套用 BTC)**
- BTC 也是正期望（Profit Factor > 1.15、樣本外 71% 視窗獲利），證明參數捕捉的是普遍趨勢動能
- 但 ETH 各項指標皆優於 BTC → **ETH 是最適合此策略的標的**
- 為 BTC 單獨優化反而會過擬合，故 BTC 直接沿用 ETH 的穩健參數

## 核心功能特點

### 技術指標
- **EMA90/200**: 雙重指數移動平均線趨勢判斷
- **ADX**: 平均趨向指數 (多單30 / 空單45) 確保強趨勢交易
- **RSI**: 相對強弱指數輔助判斷
- **MACD**: 動量指標計算 (不用於進場判斷)

### 風險控制機制
- **固定停損**: 多單3.2381% / 空單1.4180%，進場後同步到 Bybit 交易所端保護單
- **移動停損**: 智能追蹤止盈保護獲利，觸發或更新時同步調整交易所端停損
- **資金管理**: 每筆交易使用30%可用資金 × 3倍名目槓桿（名目曝險約90%），複利滾入（實盤預設值；上方績效回測以較保守的 20% 配置／名目約 60% 驗證）
- **波動度目標倉位**: 倉位 × (目標波動 ÷ 近期波動)，限制 0.3~2.0 倍——平穩趨勢自動加碼、劇烈震盪自動縮手（實測報酬接近翻倍、回撤不變）
- **BTC 費率「多頭擁擠」過濾**: BTC 幣本位 3 日均資金費率 > 0.01%/8h 時不開新多單（該情境歷史多單 PF 僅 0.81；過濾後 PF 1.58→1.72、滾動獲利率 69→73%）
- **BTC 深負費率加碼**: 費率 < −0.01%/8h（空方深度擁擠）時新多單 ×1.5、總係數 cap 2.0（三資料集報酬提升、MDD 完全不變）
- **回撤熔斷**: 依當前回撤分級降低曝險（警戒減半 → 熔斷暫停開新倉），只擋開新倉、不影響既有倉位停損；正常運作時零干擾，僅在異常深回撤時啟動（詳見下方說明）
- **方向控制**: 可用 `TRADE_SIDE_MODE=long_only|short_only|both` 控制只做多、只做空或雙向（回測顯示加密貨幣只做多最穩健）
- **多標的支援**: 可用 `TRADE_SYMBOL=ETH/USDT|BTC/USDT` 切換交易標的，自動套用各自參數與熔斷 baseline
- **實時監控**: 每60秒檢查新K線，每分鐘檢查移動停損
- **狀態校正**: 每小時與交易所同步一次 JSON 狀態（持倉方向/數量、進場價、資金餘額）

### 自動化特性
- **24/7運行**: 持續監控市場機會
- **狀態保存**: 自動保存交易狀態，支援重啟恢復
- **錯誤處理**: 完善的異常處理和重試機制
- **日誌記錄**: 詳細的交易日誌和績效追蹤

## 快速開始

### 系統需求
- Python 3.8+
- Bybit 統一交易帳戶
- 穩定的網路連接

### API 設置
1. 註冊 [Bybit 帳戶](https://www.bybit.com/invite?ref=JY5GXR)
2. 開啟合約交易功能
3. 升級為「統一交易帳戶」
4. 申請 API 金鑰和密鑰

### 安裝步驟

1. **克隆專案**
```bash
git clone <repository-url>
cd "EMA_trader_byEricLiao - 開放版本"
```

2. **安裝依賴**
```bash
pip install -r requirements.txt
```

3. **環境配置**
創建 `.env` 文件並設置 API 資訊：
```env
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
TRADE_SIDE_MODE=long_only
TRADE_SYMBOL=ETH/USDT
```

- `TRADE_SYMBOL`：選擇交易標的，可設 `ETH/USDT` 或 `BTC/USDT`（預設 `ETH/USDT`）。
  程式會自動套用該標的各自獨立優化的最佳參數與回撤熔斷 baseline，無需手動改 code。
- `TRADE_SIDE_MODE`：`long_only`（預設）/ `short_only` / `both`。回測顯示加密貨幣只做多最穩健。

4. **槓桿設置**
- 登入 Bybit 網站 → 合約交易
- 選擇與 `TRADE_SYMBOL` 相同的交易對（ETH/USDT 或 BTC/USDT）
- 將 Bybit 合約槓桿設為 **大於或等於程式中的 `TARGET_POSITION_LEVERAGE`**
- 程式槓桿只用來計算名目倉位大小；交易所槓桿較高時較容易通過保證金檢查
- 若使用 isolated margin，交易所槓桿越高，爆倉距離通常越近，請保守設定

5. **資金劃轉**
將投資資金劃轉到統一交易帳戶

### 運行策略

實盤主程式為混合實盤 `harmonic/run_mixed_live.py`（含本策略趨勢腿 + 諧波腿 + 費率抄底腿，互斥持倉、趨勢優先）：

```bash
python harmonic/run_mixed_live.py
```

預設 DRY-RUN（觀察模式）；環境變數 `HARMONIC_DRY_RUN=0` 才實際下單，實單啟動需按 Enter 確認風險，保持 CMD 視窗開啟以維持自動交易。詳見 [harmonic/README.md](harmonic/README.md)。

## 策略參數

### 最佳參數組合 (Optuna 2026-05-19 更新)

| 參數類型 | 參數名稱 | 數值 | 說明 |
|---------|---------|------|------|
| 趨勢判斷 | 多單 ADX 閾值 | 30 | 多單強趨勢判斷標準 |
| 趨勢判斷 | 空單 ADX 閾值 | 45 | 空單強趨勢判斷標準 |
| 多頭交易 | 固定停損 | 3.2381% | 多頭固定停損點 |
| 多頭交易 | 移動停損激活 | 2.8458% | 開始追蹤止盈的獲利點 |
| 多頭交易 | 移動停損回撤 | 7.8573% | 允許的最大回撤幅度 |
| 多頭交易 | 最小保護獲利 | 2.5316% | 保證的最小獲利 |
| 空頭交易 | 固定停損 | 1.4180% | 空頭固定停損點 |
| 空頭交易 | 移動停損激活 | 0.4073% | 開始追蹤止盈的獲利點 |
| 空頭交易 | 移動停損回撤 | 8.1061% | 允許的最大回撤幅度 |
| 空頭交易 | 最小保護獲利 | 0.1452% | 保證的最小獲利 |
| 資金管理 | 可用資金比例 | 20% | 每次交易使用的可用 USDT 比例 |
| 資金管理 | 目標名目槓桿 | 3x | 程式計算倉位大小用；交易所槓桿需大於或等於此值 |

> 上表為 ETH 參數；倉位以「當前權益 × 20% × 3 倍」計算，**複利**滾入。

### 多標的參數

| 標的 | 參數來源 | 熔斷 baseline |
|------|---------|--------------|
| `ETH/USDT` | 上表（Optuna 2026-05-19） | 18% |
| `BTC/USDT` | 沿用 ETH 參數（單獨優化會過擬合到近期下跌，樣本外驗證沿用 ETH 參數 PF 1.17 較穩健） | 22% |

在 `.env` 設定 `TRADE_SYMBOL=ETH/USDT` 或 `BTC/USDT` 即可切換，程式自動套用對應參數與熔斷 baseline。

### 回撤熔斷參數

| 等級 | 觸發（以標的 baseline 為基準） | 動作 |
|------|------------------------------|------|
| 正常 | 回撤 < 1.5× baseline | 100% 曝險 |
| 警戒 | 回撤 ≥ 1.5× baseline | 開倉資金減半 |
| 熔斷 | 回撤 ≥ 2.0× baseline | 暫停開新倉（既有倉位停損照常運作） |

設計依據：Grossman-Zhou (1993) / CPPI「回撤越深、曝險越低」原則，搭配系統交易實務的 drawdown-multiple 門檻（~1.5× 警戒、~2.0× 視為失效）。可在 `strategy_core.py` 頂部用 `CIRCUIT_BREAKER_ENABLED` 開關、`SYMBOL_CIRCUIT_BREAKER_BASELINE` 調整。回撤縮回後自動恢復正常曝險。

## 專案結構

```
EMA_trader_byEricLiao/
├── strategy_core.py                # 策略共用核心（指標/進場訊號/K線抓取/費率/配置常數）
├── backtest_eth_strategy_4h.py     # 回測引擎（Position/停損出場/風控 overlay）
├── run_best_params_backtest.py     # 最佳參數回測重現
├── rolling_window_backtest.py      # 滾動視窗驗證
├── optimize_strategy_optuna.py     # Optuna 參數優化
├── harmonic/                       # 混合實盤與諧波策略
│   └── run_mixed_live.py           # 實盤主程式（趨勢+諧波+抄底，預設 DRY-RUN）
├── requirements.txt                # Python 依賴套件
├── strategy_state.json             # 策略狀態保存文件（舊單腿引擎遺留）
├── .env                            # API 配置文件 (需自行創建)
└── README.md                       # 專案說明文件
```

## 交易邏輯

### 多頭進場條件
- 收盤價 > EMA90
- 最低價 > EMA90  
- 收盤價 > EMA200
- ADX > 30 (強趨勢確認)
- RSI ≤ 70（避免過熱追多）
- BTC 幣本位 3 日均資金費率 ≤ 0.01%/8h（多頭擁擠過濾；費率讀取失敗時自動旁路）

### 空頭進場條件
- 收盤價 < EMA90
- 最高價 < EMA90
- 收盤價 < EMA200  
- ADX > 45 (強趨勢確認)
- RSI ≥ 30（避免超賣追空）

### 出場條件
- 固定停損觸發
- 移動停損觸發
- 反向信號出現

### 實盤保護
- 平倉單使用 `reduceOnly`，降低誤開反向倉風險
- 進場後會用 Bybit `trading-stop` 設定整倉 stop loss
- 本地每分鐘移動停損仍保留，但交易所端停損是斷線時的保險絲
- 回撤熔斷只影響「開新倉」的曝險，平倉與停損永遠暢通
- 啟動時強制 UTF-8 輸出，避免 Windows cp950 主控台輸出 emoji 時崩潰中斷交易

## 回測工具

專案提供獨立回測腳本，不會觸發實盤下單：

```bash
python backtest_eth_strategy_4h.py --start 2020-01-01 --end 2025-01-01 --initial-capital 1000
```

### 回測輸出
- `backtest_results/summary.json`: 總報酬、CAGR、最大回撤、勝率、profit factor、Sharpe 等摘要
- `backtest_results/trades.csv`: 每筆交易進出場、盈虧、出場原因
- `backtest_results/equity_curve.csv`: 權益曲線

### 重要假設
- 訊號使用已完成的 4 小時 K 線計算
- 進出場訊號以下一根 K 線開盤價成交，避免偷看未來
- 固定停損用 4 小時 K 線 high/low 模擬盤中觸發
- 移動停損採保守模擬，同一根 K 線不允許「剛激活又立刻停利」
- 反向訊號平倉後不在同一根 K 線反手開倉（與實盤一致；可用 `allow_same_bar_reversal` 調整）
- 手續費和滑價可用 `--fee-rate`、`--slippage-rate` 調整

### 使用 CSV 回測
CSV 需包含 `timestamp,open,high,low,close,volume` 欄位：

```bash
python backtest_eth_strategy_4h.py --csv data/eth_4h.csv --initial-capital 1000
```

## Optuna 多參數優化

Optuna 優化腳本會用半年滑窗評估每組參數，而不是只看單一全期間報酬，降低過度擬合風險：

```bash
python optimize_strategy_optuna.py --csv rolling_backtest_results_5y_6m_step2m/ohlcv_with_warmup.csv --start 2021-05-19 --end 2026-05-19 --n-trials 300 --n-jobs 4 --out-dir optuna_results_5y_quality
```

> ⚠️ 建議加上 `--force-side-mode long_only --search-scope long`。實測讓 Optuna 自由選方向時，
> 容易過擬合到近期下跌而選 `short_only`（樣本內漂亮但樣本外會虧）；加密貨幣趨勢策略以只做多最穩健。
> 同理，逐標的單獨精調反而脆弱，務必用樣本外（不同期間／不同標的）驗證後再採用新參數。

### Quality Score
核心目標函數為：

```text
Quality Score = robust_calmar * robust_profit_factor * consistency_bonus * trade_count_penalty * drawdown_penalty
```

- `robust_calmar`: median Calmar 與 25 分位 Calmar 的加權值
- `robust_profit_factor`: median Profit Factor 與 25 分位 Profit Factor 的加權值
- `consistency_bonus`: 盈利滑窗比例越高越好
- `trade_count_penalty`: 交易太少會被懲罰，避免樣本不足的假高分
- `drawdown_penalty`: 最差滑窗回撤過深會被懲罰

### 輸出
- `best_params.json`: 最佳參數、控制項與品質指標
- `best_rolling_summary.csv`: 最佳參數在每個半年滑窗的績效
- `trials.csv`: 所有 trial 的參數與分數

## 風險提醒
## ⚠️ 風險提醒

### 🚨 重要注意事項
- **高風險投資**: 加密貨幣交易具有高度風險
- **資金管理**: 建議僅投入可承受損失的資金
- **監控必要**: 定期檢查策略運行狀況
- **參數調整**: 市場環境變化時可能需要調整參數

### 💡 使用建議
- **小額測試**: 建議先用小資金測試策略效果
- **風險分散**: 不要將全部資金投入單一策略
- **持續學習**: 了解技術分析和風險管理原理
- **備份重要**: 定期備份 `.env` 和狀態文件

## 📞 技術支援

如有問題或建議，請通過以下方式聯繫：
- 📧 Email: eerrerer321@gmail.com
- 💬 Telegram: https://t.me/eerrerer321

## 📄 免責聲明

本軟體僅供教育和研究目的使用。使用者需自行承擔所有交易風險，開發者不對任何投資損失負責。請在充分了解風險的情況下使用本策略。

---

**⭐ 如果這個專案對您有幫助，請給我一個 Star！**
