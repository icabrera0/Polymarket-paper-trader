"""
Persistencia SQLite del bot.

Capa fina sobre el módulo `sqlite3` de la stdlib. NO usamos ORM porque el
modelo de datos es simple y queremos mantener mínimas las dependencias.

Tablas:
  - trades              → todas las posiciones (abiertas y cerradas)
  - balance_history     → snapshot del balance cada vez que cambia
  - decisions_log       → cada TradeDecision emitida (incluyendo NO_TRADE)
  - analyses_log        → cada MarketAnalysis del LLM para auditoría / reporte

Convenciones:
- Todas las fechas se guardan como ISO 8601 UTC string (no timestamps numéricos:
  son legibles desde DB Browser, no se rompen con cambios de zona horaria).
- Importes en EUR (paper trading); convertimos a USD donde aplique.
- Ningún método tira excepción al caller — loguean warning y continúan.
  El bot debe poder seguir operando aunque la DB falle puntualmente.
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
    MarketAnalysis,
    Position,
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
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    balance_eur     REAL NOT NULL,
    peak_balance    REAL NOT NULL,
    drawdown_pct    REAL NOT NULL,
    open_positions  INTEGER NOT NULL,
    event           TEXT
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
    """Wrapper sobre sqlite3 con métodos específicos del bot."""

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
        self._log.debug("Base de datos inicializada en {}", self.db_path)

    def _init_schema(self) -> None:
        try:
            self._conn.executescript(SCHEMA_SQL)
            # Migrate existing DBs that don't have market_slug column yet
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(trades)").fetchall()}
            if "market_slug" not in cols:
                self._conn.execute("ALTER TABLE trades ADD COLUMN market_slug TEXT NOT NULL DEFAULT ''")
                self._log.info("Migrated trades table: added market_slug column")
        except sqlite3.Error as exc:
            self._log.error("Error inicializando schema: {}", exc)
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
            self._log.error("insert_trade falló: {}", exc)
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
            self._log.error("update_trade_close falló: {}", exc)
            return False

    def get_open_positions(self) -> list[Position]:
        try:
            cur = self._conn.execute(
                "SELECT * FROM trades WHERE status = 'OPEN'"
            )
            return [self._row_to_position(row) for row in cur.fetchall()]
        except sqlite3.Error as exc:
            self._log.error("get_open_positions falló: {}", exc)
            return []

    def get_all_trades(self) -> list[Position]:
        try:
            cur = self._conn.execute(
                "SELECT * FROM trades ORDER BY entry_timestamp DESC"
            )
            return [self._row_to_position(row) for row in cur.fetchall()]
        except sqlite3.Error as exc:
            self._log.error("get_all_trades falló: {}", exc)
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
            self._log.error("get_trades_in_range falló: {}", exc)
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
    ) -> bool:
        try:
            self._conn.execute(
                """
                INSERT INTO balance_history
                (timestamp, balance_eur, peak_balance, drawdown_pct,
                 open_positions, event)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (_now_iso(), balance_eur, peak_balance, drawdown_pct,
                 open_positions, event),
            )
            return True
        except sqlite3.Error as exc:
            self._log.error("log_balance falló: {}", exc)
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
            self._log.error("get_balance_history falló: {}", exc)
            return []

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
            self._log.error("log_decision falló: {}", exc)
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
            self._log.error("log_analysis falló: {}", exc)
            return False

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
