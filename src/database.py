"""
SQLite persistence layer for the bot.

Thin wrapper over the stdlib `sqlite3` module. No ORM is used because the
data model is simple and we want to keep dependencies minimal.

Tables:
  - trades              → all positions (open and closed)
  - balance_history     → balance snapshot every time it changes
  - decisions_log       → every TradeDecision emitted (including NO_TRADE)
  - analyses_log        → every LLM MarketAnalysis for auditing / reporting

Conventions:
- All dates are stored as ISO 8601 UTC strings (not numeric timestamps:
  they are readable from DB Browser and do not break with timezone changes).
- Amounts in EUR (paper trading); converted to USD where applicable.
- No method raises an exception to the caller — they log a warning and continue.
  The bot must be able to keep operating even if the DB fails occasionally.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from src.models import (
    CloseReason,
    FailureCategory,
    KnowledgeBaseEntry,
    MarketAnalysis,
    PerformanceSnapshot,
    Position,
    PostMortem,
    TradeDecision,
    TradeSide,
    TradeStatus,
)


# =====================================================
# Schema
# =====================================================

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id            TEXT PRIMARY KEY,
    market_question     TEXT NOT NULL,
    market_slug         TEXT NOT NULL DEFAULT '',
    token_id            TEXT NOT NULL,
    side                TEXT NOT NULL,
    entry_price         REAL NOT NULL,
    size_eur            REAL NOT NULL,
    size_usd            REAL NOT NULL,
    tokens_quantity     REAL NOT NULL,
    entry_timestamp     TEXT NOT NULL,
    stop_loss_price     REAL NOT NULL,
    take_profit_price   REAL NOT NULL,
    status              TEXT NOT NULL,
    exit_price          REAL,
    exit_timestamp      TEXT,
    close_reason        TEXT,
    pnl_eur             REAL,
    pnl_pct             REAL,
    entry_reason        TEXT,
    exit_reason_text    TEXT,
    confidence          INTEGER
);

CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_entry  ON trades(entry_timestamp);

CREATE TABLE IF NOT EXISTS balance_history (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               TEXT NOT NULL,
    balance_eur             REAL NOT NULL,
    peak_balance            REAL NOT NULL,
    drawdown_pct            REAL NOT NULL,
    open_positions          INTEGER NOT NULL,
    event                   TEXT,
    consolidated_profit_eur REAL NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_balance_timestamp ON balance_history(timestamp);

CREATE TABLE IF NOT EXISTS decisions_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    market_id       TEXT NOT NULL,
    market_question TEXT NOT NULL,
    action          TEXT NOT NULL,
    side            TEXT,
    size_eur        REAL,
    confidence      INTEGER,
    edge            REAL,
    skip_reasons    TEXT,
    rationale       TEXT
);

CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions_log(timestamp);

CREATE TABLE IF NOT EXISTS analyses_log (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp                   TEXT NOT NULL,
    market_id                   TEXT NOT NULL,
    market_question             TEXT NOT NULL,
    current_yes_price           REAL,
    consensus_probability_yes   REAL,
    edge                        REAL,
    confidence                  INTEGER,
    sentiment_score             REAL,
    impact_score                REAL,
    recommendation              TEXT,
    timeframe                   TEXT,
    contradictory_sources       INTEGER,
    summary                     TEXT,
    num_articles_analyzed       INTEGER,
    llm_model                   TEXT,
    llm_input_tokens            INTEGER,
    llm_output_tokens           INTEGER
);

CREATE INDEX IF NOT EXISTS idx_analyses_market ON analyses_log(market_id);
CREATE INDEX IF NOT EXISTS idx_analyses_ts     ON analyses_log(timestamp);

CREATE TABLE IF NOT EXISTS post_mortems (
    id               TEXT PRIMARY KEY,
    trade_id         TEXT NOT NULL,
    failure_category TEXT NOT NULL,
    root_cause       TEXT,
    lesson           TEXT,
    market_slug      TEXT,
    predicted_prob   REAL,
    actual_outcome   INTEGER,
    pnl_pct          REAL,
    time_held_hours  REAL,
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pm_trade_id  ON post_mortems(trade_id);
CREATE INDEX IF NOT EXISTS idx_pm_created   ON post_mortems(created_at);

CREATE TABLE IF NOT EXISTS knowledge_base (
    id               TEXT PRIMARY KEY,
    market_pattern   TEXT NOT NULL,
    lesson           TEXT NOT NULL,
    failure_category TEXT NOT NULL,
    confidence       REAL NOT NULL DEFAULT 0.4,
    times_confirmed  INTEGER NOT NULL DEFAULT 0,
    category         TEXT NOT NULL DEFAULT 'general',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kb_confidence ON knowledge_base(confidence DESC);

CREATE TABLE IF NOT EXISTS performance_snapshots (
    snapshot_date  TEXT PRIMARY KEY,
    win_rate       REAL,
    sharpe_ratio   REAL,
    max_drawdown   REAL,
    profit_factor  REAL,
    brier_score    REAL,
    total_trades   INTEGER,
    open_positions INTEGER
);
"""


# =====================================================
# Helpers
# =====================================================


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat()


def _iso_to_datetime(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


# =====================================================
# Database
# =====================================================


class Database:
    """Wrapper around sqlite3 with bot-specific methods."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = logger.bind(module="database")

        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()
        self._log.debug("Database initialized at {}", self.db_path)

    def _init_schema(self) -> None:
        try:
            self._conn.executescript(SCHEMA_SQL)
            # Migrate existing DBs that don't have market_slug column yet
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(trades)").fetchall()}
            if "market_slug" not in cols:
                self._conn.execute("ALTER TABLE trades ADD COLUMN market_slug TEXT NOT NULL DEFAULT ''")
                self._log.info("Migrated trades table: added market_slug column")
            # Migrate existing DBs that don't have consolidated_profit_eur column yet
            bal_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(balance_history)").fetchall()}
            if "consolidated_profit_eur" not in bal_cols:
                self._conn.execute(
                    "ALTER TABLE balance_history ADD COLUMN consolidated_profit_eur REAL NOT NULL DEFAULT 0.0"
                )
                self._log.info("Migrated balance_history: added consolidated_profit_eur column")
            # Migrate existing DBs that don't have knowledge_base.category column yet
            kb_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(knowledge_base)").fetchall()}
            if "category" not in kb_cols:
                self._conn.execute(
                    "ALTER TABLE knowledge_base ADD COLUMN category TEXT NOT NULL DEFAULT 'general'"
                )
                self._log.info("Migrated knowledge_base: added category column")
        except sqlite3.Error as exc:
            self._log.error("Error initializing schema: {}", exc)
            raise

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    # =====================================================
    # Trades
    # =====================================================

    def insert_trade(self, position: Position) -> bool:
        try:
            self._conn.execute(
                """
                INSERT INTO trades (
                    trade_id, market_question, market_slug, token_id, side,
                    entry_price, size_eur, size_usd, tokens_quantity,
                    entry_timestamp,
                    stop_loss_price, take_profit_price,
                    status, exit_price, exit_timestamp, close_reason,
                    pnl_eur, pnl_pct,
                    entry_reason, exit_reason_text, confidence
                ) VALUES (?,?,?,?,?, ?,?,?,?, ?, ?,?, ?,?,?,?, ?,?, ?,?,?)
                """,
                (
                    position.trade_id,
                    position.market_question,
                    position.market_slug,
                    position.token_id,
                    position.side.value,
                    position.entry_price,
                    position.size_eur,
                    position.size_usd,
                    position.tokens_quantity,
                    _datetime_to_iso(position.entry_timestamp),
                    position.stop_loss_price,
                    position.take_profit_price,
                    position.status.value,
                    position.exit_price,
                    _datetime_to_iso(position.exit_timestamp),
                    position.close_reason.value if position.close_reason else None,
                    position.pnl_eur,
                    position.pnl_pct,
                    position.entry_reason,
                    position.exit_reason_text,
                    position.confidence,
                ),
            )
            return True
        except sqlite3.Error as exc:
            self._log.error("insert_trade failed: {}", exc)
            return False

    def update_trade_close(self, position: Position) -> bool:
        try:
            self._conn.execute(
                """
                UPDATE trades SET
                    status = ?, exit_price = ?, exit_timestamp = ?,
                    close_reason = ?, pnl_eur = ?, pnl_pct = ?,
                    exit_reason_text = ?
                WHERE trade_id = ?
                """,
                (
                    position.status.value,
                    position.exit_price,
                    _datetime_to_iso(position.exit_timestamp),
                    position.close_reason.value if position.close_reason else None,
                    position.pnl_eur,
                    position.pnl_pct,
                    position.exit_reason_text,
                    position.trade_id,
                ),
            )
            return True
        except sqlite3.Error as exc:
            self._log.error("update_trade_close failed: {}", exc)
            return False

    def get_open_positions(self) -> list[Position]:
        try:
            cur = self._conn.execute(
                "SELECT * FROM trades WHERE status = 'OPEN'"
            )
            return [self._row_to_position(row) for row in cur.fetchall()]
        except sqlite3.Error as exc:
            self._log.error("get_open_positions failed: {}", exc)
            return []

    def get_all_trades(self) -> list[Position]:
        try:
            cur = self._conn.execute(
                "SELECT * FROM trades ORDER BY entry_timestamp DESC"
            )
            return [self._row_to_position(row) for row in cur.fetchall()]
        except sqlite3.Error as exc:
            self._log.error("get_all_trades failed: {}", exc)
            return []

    def get_trades_in_range(
        self, start: datetime, end: datetime
    ) -> list[Position]:
        try:
            cur = self._conn.execute(
                """
                SELECT * FROM trades
                WHERE entry_timestamp >= ? AND entry_timestamp <= ?
                ORDER BY entry_timestamp ASC
                """,
                (_datetime_to_iso(start), _datetime_to_iso(end)),
            )
            return [self._row_to_position(row) for row in cur.fetchall()]
        except sqlite3.Error as exc:
            self._log.error("get_trades_in_range failed: {}", exc)
            return []

    @staticmethod
    def _row_to_position(row: sqlite3.Row) -> Position:
        from src.models import _now_utc

        entry_ts = _iso_to_datetime(row["entry_timestamp"]) or _now_utc()
        exit_ts = _iso_to_datetime(row["exit_timestamp"])
        close_reason = (
            CloseReason(row["close_reason"]) if row["close_reason"] else None
        )

        return Position(
            trade_id=row["trade_id"],
            market_question=row["market_question"],
            market_slug=row["market_slug"] if "market_slug" in row.keys() else "",
            token_id=row["token_id"],
            side=TradeSide(row["side"]),
            entry_price=row["entry_price"],
            size_eur=row["size_eur"],
            size_usd=row["size_usd"],
            tokens_quantity=row["tokens_quantity"],
            entry_timestamp=entry_ts,
            stop_loss_price=row["stop_loss_price"],
            take_profit_price=row["take_profit_price"],
            status=TradeStatus(row["status"]),
            exit_price=row["exit_price"],
            exit_timestamp=exit_ts,
            close_reason=close_reason,
            pnl_eur=row["pnl_eur"],
            pnl_pct=row["pnl_pct"],
            entry_reason=row["entry_reason"] or "",
            exit_reason_text=row["exit_reason_text"] or "",
            confidence=row["confidence"] or 0,
        )

    # =====================================================
    # Balance history
    # =====================================================

    def log_balance(
        self,
        balance_eur: float,
        peak_balance: float,
        drawdown_pct: float,
        open_positions: int,
        event: str = "DAILY_SNAPSHOT",
        consolidated_profit_eur: float = 0.0,
    ) -> bool:
        try:
            self._conn.execute(
                """
                INSERT INTO balance_history
                (timestamp, balance_eur, peak_balance, drawdown_pct,
                 open_positions, event, consolidated_profit_eur)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (_now_iso(), balance_eur, peak_balance, drawdown_pct,
                 open_positions, event, consolidated_profit_eur),
            )
            return True
        except sqlite3.Error as exc:
            self._log.error("log_balance failed: {}", exc)
            return False

    def get_balance_history(
        self, since: Optional[datetime] = None
    ) -> list[dict[str, Any]]:
        try:
            if since:
                cur = self._conn.execute(
                    "SELECT * FROM balance_history WHERE timestamp >= ? "
                    "ORDER BY timestamp ASC",
                    (_datetime_to_iso(since),),
                )
            else:
                cur = self._conn.execute(
                    "SELECT * FROM balance_history ORDER BY timestamp ASC"
                )
            return [dict(row) for row in cur.fetchall()]
        except sqlite3.Error as exc:
            self._log.error("get_balance_history failed: {}", exc)
            return []

    def get_consolidated_profit(self) -> float:
        """Returns total EUR swept to consolidated profit across all time."""
        try:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(consolidated_profit_eur), 0.0) "
                "FROM balance_history WHERE event = 'profit_sweep'"
            ).fetchone()
            return float(row[0]) if row else 0.0
        except sqlite3.Error as exc:
            self._log.error("get_consolidated_profit failed: {}", exc)
            return 0.0

    # =====================================================
    # Decisions & analyses logs
    # =====================================================

    def log_decision(self, decision: TradeDecision) -> bool:
        try:
            self._conn.execute(
                """
                INSERT INTO decisions_log
                (timestamp, market_id, market_question, action, side,
                 size_eur, confidence, edge, skip_reasons, rationale)
                VALUES (?,?,?,?,?, ?,?,?,?,?)
                """,
                (
                    _datetime_to_iso(decision.decided_at),
                    decision.market_id,
                    decision.market_question,
                    decision.action.value,
                    decision.side.value if decision.side else None,
                    decision.size_eur,
                    decision.confidence,
                    decision.edge,
                    json.dumps([r.value for r in decision.skip_reasons]),
                    decision.rationale,
                ),
            )
            return True
        except sqlite3.Error as exc:
            self._log.error("log_decision failed: {}", exc)
            return False

    def log_analysis(self, analysis: MarketAnalysis) -> bool:
        try:
            self._conn.execute(
                """
                INSERT INTO analyses_log (
                    timestamp, market_id, market_question,
                    current_yes_price, consensus_probability_yes,
                    edge, confidence, sentiment_score, impact_score,
                    recommendation, timeframe, contradictory_sources,
                    summary, num_articles_analyzed,
                    llm_model, llm_input_tokens, llm_output_tokens
                ) VALUES (?,?,?, ?,?, ?,?,?,?, ?,?,?, ?,?, ?,?,?)
                """,
                (
                    _datetime_to_iso(analysis.analyzed_at),
                    analysis.market_id,
                    analysis.market_question,
                    analysis.current_yes_price,
                    analysis.consensus_probability_yes,
                    analysis.edge,
                    analysis.confidence,
                    analysis.sentiment_score,
                    analysis.impact_score,
                    analysis.recommendation.value,
                    analysis.timeframe.value,
                    int(analysis.contradictory_sources),
                    analysis.summary,
                    analysis.num_articles_analyzed,
                    analysis.llm_model,
                    analysis.llm_input_tokens,
                    analysis.llm_output_tokens,
                ),
            )
            return True
        except sqlite3.Error as exc:
            self._log.error("log_analysis failed: {}", exc)
            return False

    # =====================================================
    # Compound layer: post-mortems
    # =====================================================

    def save_post_mortem(self, pm: PostMortem) -> None:
        try:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO post_mortems
                (id, trade_id, failure_category, root_cause, lesson,
                 market_slug, predicted_prob, actual_outcome,
                 pnl_pct, time_held_hours, created_at)
                VALUES (?,?,?,?,?, ?,?,?, ?,?,?)
                """,
                (
                    str(pm.trade_id) + "-pm",
                    pm.trade_id,
                    pm.failure_category.value,
                    pm.root_cause,
                    pm.lesson,
                    pm.market_slug,
                    pm.predicted_prob,
                    int(pm.actual_outcome) if pm.actual_outcome is not None else None,
                    pm.pnl_pct,
                    pm.time_held_hours,
                    _datetime_to_iso(pm.created_at),
                ),
            )
        except sqlite3.Error as exc:
            self._log.error("save_post_mortem failed: {}", exc)

    def get_post_mortems_today(self) -> list[dict[str, Any]]:
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            cur = self._conn.execute(
                "SELECT * FROM post_mortems WHERE created_at >= ? ORDER BY created_at DESC",
                (today,),
            )
            return [dict(r) for r in cur.fetchall()]
        except sqlite3.Error as exc:
            self._log.error("get_post_mortems_today failed: {}", exc)
            return []

    # =====================================================
    # Compound layer: knowledge base
    # =====================================================

    def save_knowledge_entry(self, entry: KnowledgeBaseEntry) -> None:
        try:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO knowledge_base
                (id, market_pattern, lesson, failure_category,
                 confidence, times_confirmed, category, created_at, updated_at)
                VALUES (?,?,?,?, ?,?,?,?,?)
                """,
                (
                    entry.id,
                    entry.market_pattern,
                    entry.lesson,
                    entry.failure_category.value,
                    entry.confidence,
                    entry.times_confirmed,
                    entry.category,
                    _datetime_to_iso(entry.created_at),
                    _datetime_to_iso(entry.updated_at),
                ),
            )
        except sqlite3.Error as exc:
            self._log.error("save_knowledge_entry failed: {}", exc)

    def update_knowledge_entry_confidence(
        self, id: str, times_confirmed: int, confidence: float
    ) -> None:
        try:
            now = _now_iso()
            self._conn.execute(
                """
                UPDATE knowledge_base
                SET times_confirmed=?, confidence=?, updated_at=?
                WHERE id=?
                """,
                (times_confirmed, confidence, now, id),
            )
        except sqlite3.Error as exc:
            self._log.error("update_knowledge_entry_confidence failed: {}", exc)

    def get_knowledge_base(self, limit: int = 50) -> list[KnowledgeBaseEntry]:
        try:
            cur = self._conn.execute(
                """
                SELECT * FROM knowledge_base
                ORDER BY confidence DESC, times_confirmed DESC
                LIMIT ?
                """,
                (limit,),
            )
            results = []
            for row in cur.fetchall():
                r = dict(row)
                results.append(KnowledgeBaseEntry(
                    id=r["id"],
                    market_pattern=r["market_pattern"],
                    lesson=r["lesson"],
                    failure_category=FailureCategory(r["failure_category"]),
                    confidence=r["confidence"],
                    times_confirmed=r["times_confirmed"],
                    category=r.get("category", "general"),
                    created_at=_iso_to_datetime(r["created_at"]) or datetime.now(timezone.utc),
                    updated_at=_iso_to_datetime(r["updated_at"]) or datetime.now(timezone.utc),
                ))
            return results
        except sqlite3.Error as exc:
            self._log.error("get_knowledge_base failed: {}", exc)
            return []

    def delete_knowledge_entries(self, ids: list[str]) -> None:
        if not ids:
            return
        try:
            placeholders = ",".join("?" * len(ids))
            self._conn.execute(
                f"DELETE FROM knowledge_base WHERE id IN ({placeholders})", ids
            )
        except sqlite3.Error as exc:
            self._log.error("delete_knowledge_entries failed: {}", exc)

    def get_post_mortems_by_pattern(self, pattern: str) -> list[dict[str, Any]]:
        """Fetch post-mortems whose market_slug contains keywords from the pattern.

        Uses the first meaningful words (length > 3) from the pattern as fuzzy
        LIKE matches against market_slug. Returns an empty list if no meaningful
        words exist or on DB error.
        """
        words = [w for w in pattern.lower().replace("-", " ").split() if len(w) > 3]
        if not words:
            return []
        try:
            clauses = " OR ".join("market_slug LIKE ?" for _ in words)
            params = [f"%{w}%" for w in words]
            cur = self._conn.execute(
                f"SELECT * FROM post_mortems WHERE {clauses}", params
            )
            return [dict(r) for r in cur.fetchall()]
        except sqlite3.Error as exc:
            self._log.error("get_post_mortems_by_pattern failed: {}", exc)
            return []

    # =====================================================
    # Compound layer: performance snapshots
    # =====================================================

    def save_performance_snapshot(self, snap: PerformanceSnapshot) -> None:
        try:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO performance_snapshots
                (snapshot_date, win_rate, sharpe_ratio, max_drawdown,
                 profit_factor, brier_score, total_trades, open_positions)
                VALUES (?,?,?,?, ?,?,?,?)
                """,
                (
                    snap.snapshot_date.isoformat(),
                    snap.win_rate,
                    snap.sharpe_ratio,
                    snap.max_drawdown,
                    snap.profit_factor,
                    snap.brier_score,
                    snap.total_trades,
                    snap.open_positions,
                ),
            )
        except sqlite3.Error as exc:
            self._log.error("save_performance_snapshot failed: {}", exc)

    def get_performance_history(self, days: int = 30) -> list[PerformanceSnapshot]:
        try:
            from datetime import timedelta, date
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
            cur = self._conn.execute(
                "SELECT * FROM performance_snapshots WHERE snapshot_date >= ? ORDER BY snapshot_date ASC",
                (cutoff,),
            )
            results = []
            for row in cur.fetchall():
                r = dict(row)
                results.append(PerformanceSnapshot(
                    snapshot_date=date.fromisoformat(r["snapshot_date"]),
                    win_rate=r["win_rate"] or 0.0,
                    sharpe_ratio=r["sharpe_ratio"] or 0.0,
                    max_drawdown=r["max_drawdown"] or 0.0,
                    profit_factor=r["profit_factor"] or 0.0,
                    brier_score=r["brier_score"] or 0.0,
                    total_trades=r["total_trades"] or 0,
                    open_positions=r["open_positions"] or 0,
                ))
            return results
        except sqlite3.Error as exc:
            self._log.error("get_performance_history failed: {}", exc)
            return []

    def get_latest_performance_snapshot(self) -> Optional[PerformanceSnapshot]:
        snaps = self.get_performance_history(days=365)
        return snaps[-1] if snaps else None

    def get_closed_trades(self, limit: int = 200) -> list[dict[str, Any]]:
        try:
            cur = self._conn.execute(
                """
                SELECT * FROM trades
                WHERE status = 'CLOSED'
                ORDER BY exit_timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]
        except sqlite3.Error as exc:
            self._log.error("get_closed_trades failed: {}", exc)
            return []

    def get_closed_trades_in_window(self, days: int) -> list[dict[str, Any]]:
        try:
            from datetime import timedelta
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            cur = self._conn.execute(
                """
                SELECT * FROM trades
                WHERE status = 'CLOSED' AND exit_timestamp >= ?
                ORDER BY exit_timestamp ASC
                """,
                (cutoff,),
            )
            return [dict(r) for r in cur.fetchall()]
        except sqlite3.Error as exc:
            self._log.error("get_closed_trades_in_window failed: {}", exc)
            return []

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
