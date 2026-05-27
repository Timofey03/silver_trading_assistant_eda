"""
argentum/backend/db.py — SQLite storage для positions + closed_trades.

Atomic writes, concurrent-safe. Заменяет positions.json.

Schema:
  positions (id, ticker, figi, opened_at, entry_price, lots, lot_size_g,
             peak_price, source)
  closed_trades (id, original_id, ticker, figi, opened_at, closed_at,
                 entry_price, exit_price, lots, lot_size_g, realized_pnl_pct,
                 exit_reason)

Также легко мигрируется из существующего positions.json.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_FILE = Path(__file__).resolve().parent / "data" / "argentum.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    id            TEXT PRIMARY KEY,
    ticker        TEXT NOT NULL,
    figi          TEXT NOT NULL,
    opened_at     TEXT NOT NULL,
    entry_price   REAL NOT NULL,
    lots          INTEGER NOT NULL DEFAULT 1,
    lot_size_g    INTEGER NOT NULL DEFAULT 100,
    peak_price    REAL NOT NULL,
    source        TEXT DEFAULT 'user'
);

CREATE TABLE IF NOT EXISTS closed_trades (
    id                TEXT PRIMARY KEY,
    original_id       TEXT NOT NULL,
    ticker            TEXT NOT NULL,
    figi              TEXT NOT NULL,
    opened_at         TEXT NOT NULL,
    closed_at         TEXT NOT NULL,
    entry_price       REAL NOT NULL,
    exit_price        REAL NOT NULL,
    lots              INTEGER NOT NULL,
    lot_size_g        INTEGER NOT NULL,
    realized_pnl_pct  REAL NOT NULL,
    exit_reason       TEXT DEFAULT 'user_close',
    days_held         INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_closed_at ON closed_trades(closed_at DESC);
"""


def _conn() -> sqlite3.Connection:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_FILE), check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # atomic writes
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with _conn() as c:
        c.executescript(SCHEMA)


def migrate_from_json(json_path: Path = None) -> int:
    """Импорт существующего positions.json в SQLite."""
    if json_path is None:
        json_path = Path(__file__).resolve().parent / "data" / "positions.json"
    if not json_path.exists():
        return 0
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if not data:
        return 0
    init_db()
    with _conn() as c:
        # Skip duplicates
        existing_ids = {r[0] for r in c.execute("SELECT id FROM positions")}
        n_imported = 0
        for p in data:
            if p["id"] in existing_ids:
                continue
            c.execute(
                "INSERT INTO positions (id, ticker, figi, opened_at, entry_price,"
                " lots, lot_size_g, peak_price, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    p["id"], p["ticker"], p["figi"], p["opened_at"],
                    float(p["entry_price"]), int(p["lots"]),
                    int(p.get("lot_size_g", p.get("lot_size_oz", 100))),
                    float(p.get("peak_price", p["entry_price"])),
                    p.get("source", "user"),
                ),
            )
            n_imported += 1
        c.commit()
    # Backup the JSON
    backup = json_path.with_suffix(".json.imported")
    if not backup.exists():
        json_path.rename(backup)
    return n_imported


# ─── Position CRUD ──────────────────────────────────────────────────────────

def list_positions() -> list[dict]:
    init_db()
    with _conn() as c:
        rows = c.execute("SELECT * FROM positions ORDER BY opened_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_position(position_id: str) -> Optional[dict]:
    init_db()
    with _conn() as c:
        r = c.execute("SELECT * FROM positions WHERE id = ?", (position_id,)).fetchone()
    return dict(r) if r else None


def insert_position(pos: dict) -> None:
    init_db()
    with _conn() as c:
        c.execute(
            "INSERT INTO positions (id, ticker, figi, opened_at, entry_price,"
            " lots, lot_size_g, peak_price, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                pos["id"], pos["ticker"], pos["figi"], pos["opened_at"],
                float(pos["entry_price"]), int(pos["lots"]),
                int(pos.get("lot_size_g", 100)),
                float(pos.get("peak_price", pos["entry_price"])),
                pos.get("source", "user"),
            ),
        )
        c.commit()


def update_peak(position_id: str, peak: float) -> None:
    init_db()
    with _conn() as c:
        c.execute("UPDATE positions SET peak_price = ? WHERE id = ?",
                  (float(peak), position_id))
        c.commit()


def delete_position(position_id: str) -> None:
    init_db()
    with _conn() as c:
        c.execute("DELETE FROM positions WHERE id = ?", (position_id,))
        c.commit()


# ─── Closed trades ──────────────────────────────────────────────────────────

def insert_closed_trade(trade: dict) -> None:
    """Записать закрытый trade в историю."""
    init_db()
    with _conn() as c:
        c.execute(
            "INSERT INTO closed_trades (id, original_id, ticker, figi, opened_at,"
            " closed_at, entry_price, exit_price, lots, lot_size_g,"
            " realized_pnl_pct, exit_reason, days_held)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trade.get("id"),
                trade.get("original_id"),
                trade.get("ticker"),
                trade.get("figi"),
                trade.get("opened_at"),
                trade.get("closed_at"),
                float(trade.get("entry_price", 0)),
                float(trade.get("exit_price", 0)),
                int(trade.get("lots", 1)),
                int(trade.get("lot_size_g", 100)),
                float(trade.get("realized_pnl_pct", 0)),
                trade.get("exit_reason", "user_close"),
                int(trade.get("days_held", 0)),
            ),
        )
        c.commit()


def list_closed_trades(limit: int = 50) -> list[dict]:
    init_db()
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM closed_trades ORDER BY closed_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    n = migrate_from_json()
    print(f"Migrated {n} positions from positions.json")
    print(f"Now {len(list_positions())} positions in SQLite")
