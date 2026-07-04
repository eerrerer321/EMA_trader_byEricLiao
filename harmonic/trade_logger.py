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

import requests

FIELDS = ["time", "strategy", "timeframe", "action", "side", "pattern",
          "price", "qty", "sl", "tp", "reason", "pnl", "dry", "order_id", "note"]

# 「當前活躍」快照欄位（只列尚未停利/停損的掛單與持倉）
ACTIVE_FIELDS = ["update_time", "status", "strategy", "side", "pattern",
                 "entry", "stop_loss", "take_profit", "qty", "since", "note"]

_NOTIFY_ACTIONS = {"place", "entry", "fill", "exit", "cancel", "skip"}


def _telegram_target() -> tuple[str | None, str | None]:
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("TELEGRAM_HOME_CHANNEL") or os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    return (token or None, chat_id or None)


def _format_telegram_event(**kw) -> str:
    status = "DRY-RUN" if str(kw.get("dry", "")).lower() in {"1", "true", "yes"} else "LIVE"
    ts = kw.get("time") or datetime.now().isoformat(timespec="seconds")
    strategy = kw.get("strategy", "")
    tf = kw.get("timeframe", "")
    action = kw.get("action", "")
    side = kw.get("side", "")
    pattern = kw.get("pattern", "")
    price = kw.get("price", "")
    qty = kw.get("qty", "")
    sl = kw.get("sl", "")
    tp = kw.get("tp", "")
    reason = kw.get("reason", "")
    pnl = kw.get("pnl", "")
    note = kw.get("note", "")
    lines = [
        f"{status}｜{ts}",
        f"{strategy} {tf} {action} {side} {pattern}".strip(),
    ]
    details = []
    if price not in ("", None):
        details.append(f"價 {price}")
    if qty not in ("", None):
        details.append(f"量 {qty}")
    if sl not in ("", None):
        details.append(f"SL {sl}")
    if tp not in ("", None):
        details.append(f"TP {tp}")
    if reason not in ("", None):
        details.append(f"原因 {reason}")
    if pnl not in ("", None):
        details.append(f"PnL {pnl}")
    if note not in ("", None):
        details.append(f"備註 {note}")
    if details:
        lines.append("｜".join(details))
    return "\n".join(lines)


def maybe_send_telegram_event(**kw) -> None:
    if str(kw.get("action", "")).lower() not in _NOTIFY_ACTIONS:
        return
    token, chat_id = _telegram_target()
    if not token or not chat_id:
        return
    text = _format_telegram_event(**kw)
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as e:
        # requests 例外訊息可能內嵌請求 URL（含 bot token），印出前遮蔽避免洩漏到日誌
        print(f"⚠️ Telegram 推播失敗: {str(e).replace(token, '***')}")


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
        maybe_send_telegram_event(**row)
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
