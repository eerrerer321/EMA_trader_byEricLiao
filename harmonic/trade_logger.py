"""
簡單的交易事件 CSV 紀錄器 — 給諧波 / 混合實盤用。

每個下單相關事件寫一行，方便事後用 pandas / Excel 分析來調整參數：
  - 各諧波型態的命中率、各策略(趨勢/諧波)的貢獻、進出場時間分布…
  - DRY-RUN 期間的紀錄(dry=True)也會寫入，跑幾天就能先檢視訊號品質。

用法:
    from trade_logger import log_event
    log_event(LOG_FILE, strategy="harmonic", action="place", side="buy",
              pattern="Gartley", price=1611.7, qty=0.1, sl=1559.5, tp=1700.0, dry=True)
"""
from __future__ import annotations

import csv
import os
from datetime import datetime

FIELDS = ["time", "strategy", "timeframe", "action", "side", "pattern",
          "price", "qty", "sl", "tp", "reason", "pnl", "dry", "order_id", "note"]

# 「當前活躍」快照欄位（只列尚未停利/停損的掛單與持倉）
ACTIVE_FIELDS = ["update_time", "status", "strategy", "side", "pattern",
                 "entry", "stop_loss", "take_profit", "qty", "since", "note"]


def log_event(path: str, **kw) -> None:
    """Append 一筆事件到 CSV（首次自動寫表頭）。失敗只警告、不中斷交易。"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        row = {f: kw.get(f, "") for f in FIELDS}
        if not row["time"]:
            row["time"] = datetime.now().isoformat(timespec="seconds")
        new_file = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
            w.writerow(row)
    except Exception as e:
        print(f"⚠️ 寫交易日誌失敗: {e}")


def ensure_log_header(path: str) -> None:
    """確保事件日誌檔存在（先建立只有表頭的空檔），讓使用者啟動即看得到檔案。"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    except Exception as e:
        print(f"⚠️ 建立交易日誌表頭失敗: {e}")


def write_active(path: str, items) -> None:
    """覆寫『當前活躍（尚未停利/停損）』快照 CSV。

    items 為 dict 列表（互斥設計下通常 0~1 筆）。每次狀態變動就覆寫，
    所以這個檔永遠只反映「此刻」掛了什麼單、持有什麼倉，方便一眼查看。
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        now = datetime.now().isoformat(timespec="seconds")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ACTIVE_FIELDS)
            w.writeheader()
            if items:
                for it in items:
                    row = {k: it.get(k, "") for k in ACTIVE_FIELDS}
                    row["update_time"] = now
                    w.writerow(row)
            else:
                w.writerow({"update_time": now, "status": "無持倉、無掛單（等待訊號）"})
    except Exception as e:
        print(f"⚠️ 寫活躍快照失敗: {e}")
