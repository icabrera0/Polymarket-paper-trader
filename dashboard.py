"""
Polymarket Paper Trading Bot Dashboard — v4.
Run: streamlit run dashboard.py
"""

from __future__ import annotations

import html as html_mod
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT   = Path(__file__).resolve().parent
_OVERRIDES_PATH = PROJECT_ROOT / "data" / "overrides.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config  # noqa: E402
from src.database import Database  # noqa: E402


def _read_override(key: str, default):
    try:
        if _OVERRIDES_PATH.exists():
            return json.loads(_OVERRIDES_PATH.read_text()).get(key, default)
    except Exception:
        pass
    return default


def _write_override(key: str, value) -> None:
    try:
        data: dict = {}
        if _OVERRIDES_PATH.exists():
            data = json.loads(_OVERRIDES_PATH.read_text())
        data[key] = value
        _OVERRIDES_PATH.parent.mkdir(exist_ok=True)
        _OVERRIDES_PATH.write_text(json.dumps(data, indent=2))
    except Exception:
        pass

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PolyBot Terminal",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto+Mono:wght@400;500;600&display=swap');

/* ── Fix ghost / greyed-out duplicate caused by time.sleep() before rerun ── */
[data-stale="true"] { display: none !important; }
.stAppRunningIndicator { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }

/* ── Variables ── */
:root {
    --bg:           #09090F;
    --surface:      #0D0D1A;
    --card:         #161622;
    --card2:        #1C1C2E;
    --border:       #262640;
    --border2:      #3A3A60;
    --primary:      #6366F1;
    --primary-dim:  rgba(99,102,241,0.12);
    --primary-glow: rgba(99,102,241,0.3);
    --success:      #10B981;
    --success-dim:  rgba(16,185,129,0.12);
    --danger:       #F43F5E;
    --danger-dim:   rgba(244,63,94,0.12);
    --warn:         #F59E0B;
    --text:         #F0F0FC;
    --text2:        #A0A0C0;
    --muted:        #6060A0;
    --muted2:       #40405A;
}

/* ── Base ── */
.stApp {
    background: var(--bg) !important;
    font-family: 'Inter', sans-serif;
}
* { box-sizing: border-box; }

/* ── Headings ── */
h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    color: var(--text) !important;
    font-weight: 600 !important;
    letter-spacing: -0.3px;
}
h1 { font-size: 1.4rem !important; }
h3 { font-size: 1.0rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover { border-color: var(--border2); }
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--primary);
    opacity: 0.8;
}
[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Roboto Mono', monospace !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'Roboto Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600;
    letter-spacing: 0.2px;
    border: 1px solid var(--primary);
    color: var(--primary) !important;
    background: transparent !important;
    border-radius: 6px;
    padding: 0.45rem 1.2rem;
    transition: all 0.2s ease;
    cursor: pointer;
}
.stButton > button:hover {
    background: var(--primary) !important;
    color: white !important;
    box-shadow: 0 0 16px var(--primary-glow);
}
.stButton > button[kind="primary"] {
    background: var(--primary) !important;
    color: white !important;
    border-color: var(--primary);
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.88;
    box-shadow: 0 0 16px var(--primary-glow);
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border) !important;
    gap: 0;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500;
    letter-spacing: 0.2px;
    color: var(--muted) !important;
    padding: 0.6rem 1.1rem !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s;
}
[data-testid="stTabs"] [role="tab"]:hover { color: var(--text2) !important; }
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom-color: var(--primary) !important;
    background: transparent !important;
}

/* ── Divider ── */
hr { border: none; border-top: 1px solid var(--border) !important; margin: 1.2rem 0; }

/* ── DataFrames / Tables ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
}
iframe { background: var(--card) !important; }

/* ── Sliders ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: var(--primary) !important;
}

/* ── Code block ── */
.stCodeBlock, [data-testid="stCode"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px;
    font-family: 'Roboto Mono', monospace !important;
}

/* ── Info / warning banners ── */
[data-testid="stAlert"] {
    border-radius: 6px;
    border-left: 3px solid var(--primary) !important;
    background: var(--primary-dim) !important;
    font-family: 'Inter', sans-serif;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] [data-baseweb="select"] {
    background: var(--card) !important;
    border-color: var(--border) !important;
}

/* ── Number input ── */
[data-testid="stNumberInput"] input {
    background: var(--card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: 'Roboto Mono', monospace !important;
}

/* ── Radio ── */
[data-testid="stRadio"] label { font-family: 'Inter', sans-serif; font-size: 0.88rem; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted2); }

/* ── Custom components ── */

/* Card */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.3rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.card:hover {
    border-color: var(--border2);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* Status dot */
.dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
}
.dot-live { background: var(--success); box-shadow: 0 0 5px var(--success); animation: pulse 2s infinite; }
.dot-off  { background: var(--muted); }
@keyframes pulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.3; }
}

/* Badge pill */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 9px;
    border-radius: 20px;
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-open  { background: var(--success-dim); color: var(--success); border: 1px solid rgba(16,185,129,0.3); }
.badge-empty { background: rgba(96,96,160,0.1); color: var(--muted); border: 1px solid rgba(96,96,160,0.2); }
.badge-warn  { background: rgba(245,158,11,0.1); color: var(--warn); border: 1px solid rgba(245,158,11,0.25); }

/* Position card — left accent border */
.pos-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.3rem;
    margin-bottom: 0.9rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.pos-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.25); }
.pos-card.side-yes { border-left-color: var(--primary); }
.pos-card.side-no  { border-left-color: var(--danger); }

.pos-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.92rem;
    font-weight: 500;
    color: var(--text);
    line-height: 1.45;
    margin-bottom: 0.4rem;
}
.pos-meta {
    font-family: 'Roboto Mono', monospace;
    font-size: 0.71rem;
    color: var(--muted);
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
    margin-top: 0.3rem;
}
.pos-meta span { white-space: nowrap; }
.pos-pnl {
    font-family: 'Roboto Mono', monospace;
    font-weight: 600;
    font-size: 1.2rem;
    text-align: right;
}
.pos-pnl.pos { color: var(--success); }
.pos-pnl.neg { color: var(--danger); }
.pos-pnl.neu { color: var(--muted); }

.tag-yes {
    background: var(--primary-dim);
    color: var(--primary);
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.4px;
    padding: 1px 7px;
    border-radius: 4px;
    border: 1px solid rgba(99,102,241,0.35);
}
.tag-no {
    background: var(--danger-dim);
    color: var(--danger);
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.4px;
    padding: 1px 7px;
    border-radius: 4px;
    border: 1px solid rgba(244,63,94,0.35);
}

/* Progress bar (SL → TP) */
.progress-wrap { margin: 0.8rem 0 0.2rem; }
.progress-track {
    position: relative;
    height: 4px;
    background: rgba(255,255,255,0.05);
    border-radius: 2px;
}
.progress-fill {
    position: absolute;
    top: 0; height: 100%;
    border-radius: 2px;
    transition: width 0.5s ease;
}
.pip {
    position: absolute;
    top: 50%;
    transform: translate(-50%, -50%);
    border-radius: 50%;
}
.progress-labels {
    display: flex;
    justify-content: space-between;
    font-family: 'Roboto Mono', monospace;
    font-size: 0.66rem;
    color: var(--muted);
    margin-top: 4px;
}

/* Section header */
.section-header {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--muted);
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.9rem;
}

/* Gain/loss inline */
.gain { color: var(--success); font-weight: 600; }
.loss { color: var(--danger); font-weight: 600; }
.mono { font-family: 'Roboto Mono', monospace; }

/* Sidebar section label */
.sb-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.63rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--muted2);
    padding: 0.7rem 0 0.25rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource
def get_config():
    return load_config()


@st.cache_data(ttl=10)
def fetch_live_prices(_config, token_ids: tuple[str, ...]) -> dict[str, float]:
    """Fetch current midpoint prices from the CLOB API for all open position tokens.
    Cached 10 s. Returns {token_id: current_price}."""
    if not token_ids:
        return {}
    try:
        from src.clob_client import ClobApiClient
        return ClobApiClient().fetch_midpoints(list(token_ids))
    except Exception:
        return {}


def get_db(config) -> Database:
    return Database(config.database.path)


def fmt_eur(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}€{v:.2f}"


def fmt_pct(v: float) -> str:
    return f"{'+'if v>0 else ''}{v:.2%}"


def pnl_color_class(v: float) -> str:
    if v > 0: return "pos"
    if v < 0: return "neg"
    return "neu"


def _chart_style(fig: go.Figure, height: int = 280) -> None:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6060A0", family="Roboto Mono", size=11),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", showline=False, title=""),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", showline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        height=height,
        margin=dict(l=0, r=8, t=28, b=0),
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(config) -> tuple[int, int]:
    with st.sidebar:
        # Logo / brand
        st.markdown("""
        <div style="padding: 1.4rem 0.8rem 0.8rem; border-bottom: 1px solid #262640;">
            <div style="font-family:'Inter',sans-serif; font-size:1.1rem; font-weight:700;
                        color:#6366F1; letter-spacing:1px;">
                POLYBOT
            </div>
            <div style="font-family:'Roboto Mono',monospace; font-size:0.62rem;
                        color:#40405A; margin-top:3px; letter-spacing:0.8px;">
                PAPER TRADING TERMINAL
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-label">Auto-refresh</div>', unsafe_allow_html=True)
        refresh = st.slider("Seconds", 5, 120, 10, step=5,
                            key="refresh_slider", label_visibility="collapsed")

        st.markdown('<div class="sb-label">Position Limit</div>', unsafe_allow_html=True)
        if "max_pos_slider" not in st.session_state:
            st.session_state["max_pos_slider"] = _read_override(
                "max_simultaneous_positions",
                config.risk.max_simultaneous_positions,
            )
        max_pos = st.slider(
            "Max simultaneous positions",
            min_value=1, max_value=10,
            step=1,
            key="max_pos_slider",
            label_visibility="collapsed",
        )
        if max_pos != config.risk.max_simultaneous_positions:
            config.risk.max_simultaneous_positions = max_pos
            _write_override("max_simultaneous_positions", max_pos)
            st.success(f"Limit updated to {max_pos}")
        else:
            config.risk.max_simultaneous_positions = max_pos

        st.markdown('<div class="sb-label">LLM Workers</div>', unsafe_allow_html=True)
        if "llm_workers_slider" not in st.session_state:
            st.session_state["llm_workers_slider"] = _read_override(
                "llm_parallelism",
                getattr(config.llm, "llm_parallelism", 1),
            )
        llm_workers = st.slider(
            "LLM Workers",
            min_value=1, max_value=4,
            step=1,
            key="llm_workers_slider",
            label_visibility="collapsed",
        )
        _write_override("llm_parallelism", llm_workers)
        if llm_workers > 1:
            st.markdown(
                '<div style="font-family:\'Inter\',sans-serif;font-size:0.70rem;'
                'color:#F59E0B;padding:2px 0 4px 0;">'
                f'⚠ Requires OLLAMA_NUM_PARALLEL={llm_workers}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sb-label">News Sources</div>', unsafe_allow_html=True)
        sources = []
        if config.news.gdelt.enabled:     sources.append(("GDELT", True))
        if config.news.newsapi.enabled:   sources.append(("NewsAPI", True))
        if config.news.telegram.enabled:  sources.append(("Telegram", True))
        if not sources:                   sources.append(("None active", False))

        for name, active in sources:
            dot_cls = "dot-live" if active else "dot-off"
            color = "#10B981" if active else "#6060A0"
            st.markdown(
                f'<div style="font-family:\'Inter\',sans-serif; font-size:0.82rem; '
                f'color:{color}; padding:2px 0;">'
                f'<span class="dot {dot_cls}"></span>{name}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sb-label">Risk Config</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Roboto Mono',monospace; font-size:0.72rem;
                    color:#A0A0C0; background:#161622;
                    border:1px solid #262640;
                    border-radius:8px; padding:0.75rem 0.85rem; line-height:2;">
            <span style="color:#6060A0">SL&nbsp;&nbsp;</span>
            <span style="color:#F43F5E">{config.risk.stop_loss_pct:.0%}</span>
            &nbsp;&nbsp;
            <span style="color:#6060A0">TP&nbsp;&nbsp;</span>
            <span style="color:#10B981">{config.risk.take_profit_pct:.0%}</span><br>
            <span style="color:#6060A0">DD max&nbsp;</span>
            <span style="color:#F59E0B">{config.risk.max_drawdown_pct:.0%}</span><br>
            <span style="color:#6060A0">pos size</span>
            <span style="color:#6366F1">{config.risk.max_position_size_pct:.0%}</span>
            <span style="color:#40405A"> of balance</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-label">System</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-family:\'Roboto Mono\',monospace; font-size:0.68rem; '
            f'color:#40405A; line-height:1.9;">'
            f'v{config.app.version}<br>'
            f'{config.llm.provider} / {config.llm.model.split(":")[0]}</div>',
            unsafe_allow_html=True,
        )

    return refresh, max_pos


# ── Overview Tab ──────────────────────────────────────────────────────────────

def render_overview(db: Database, config) -> None:
    history = db.get_balance_history()
    open_pos = db.get_open_positions()
    all_trades = db.get_all_trades()
    closed = [t for t in all_trades if t.status.value == "CLOSED"]

    bal      = float(history[-1]["balance_eur"]) if history else config.paper_trading.initial_balance_eur
    peak     = float(history[-1]["peak_balance"]) if history else bal
    drawdown = (peak - bal) / peak if peak > 0 else 0.0
    initial  = config.paper_trading.initial_balance_eur
    pnl      = bal - initial
    pnl_pct  = pnl / initial if initial > 0 else 0.0
    winners  = [t for t in closed if (t.pnl_eur or 0) > 0]
    win_rate = len(winners) / len(closed) if closed else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Balance", f"€{bal:.2f}",
              delta=f"{'+' if pnl >= 0 else ''}€{pnl:.2f}",
              delta_color="normal" if pnl >= 0 else "inverse")
    c2.metric("Total P&L", f"{'+'if pnl>=0 else ''}€{pnl:.2f}",
              delta=fmt_pct(pnl_pct),
              delta_color="normal" if pnl >= 0 else "inverse")
    c3.metric("Drawdown", f"{drawdown:.2%}",
              delta=f"max {config.risk.max_drawdown_pct:.0%}",
              delta_color="off")
    c4.metric("Positions", f"{len(open_pos)} / {config.risk.max_simultaneous_positions}",
              delta="open")
    c5.metric("Win Rate", f"{win_rate:.1%}",
              delta=f"{len(closed)} closed",
              delta_color="off")

    st.divider()

    if len(history) < 2:
        st.info("No history yet. Start the bot: `python main.py`")
        return

    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="section-header">Balance curve</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["balance_eur"],
            mode="lines", name="Balance",
            line=dict(color="#6366F1", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(99,102,241,0.06)",
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["peak_balance"],
            mode="lines", name="Peak",
            line=dict(color="#6060A0", width=1, dash="dot"),
        ))
        fig.add_hline(y=initial, line_dash="dash", line_color="#2A2A42",
                      annotation_text="Initial",
                      annotation_font_color="#6060A0", annotation_font_size=10)
        _chart_style(fig, height=270)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Drawdown</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["timestamp"], y=df["drawdown_pct"] * 100,
            mode="lines",
            line=dict(color="#F43F5E", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(244,63,94,0.06)",
        ))
        fig2.add_hline(
            y=config.risk.max_drawdown_pct * 100,
            line_dash="dash", line_color="#F59E0B",
            annotation_text=f"Limit {config.risk.max_drawdown_pct:.0%}",
            annotation_font_color="#F59E0B", annotation_font_size=10,
        )
        _chart_style(fig2, height=270)
        st.plotly_chart(fig2, use_container_width=True)

    if closed:
        st.markdown('<div class="section-header">P&L per trade (last 20)</div>', unsafe_allow_html=True)
        pnls   = [(t.pnl_eur or 0) for t in closed[-20:]]
        colors = ["#10B981" if p >= 0 else "#F43F5E" for p in pnls]
        fig3   = go.Figure(go.Bar(
            x=list(range(1, len(pnls)+1)), y=pnls,
            marker_color=colors,
            marker_line_width=0,
            text=[f"€{p:+.2f}" for p in pnls],
            textposition="outside",
            textfont=dict(size=8, color="#6060A0"),
        ))
        _chart_style(fig3, height=190)
        st.plotly_chart(fig3, use_container_width=True)


# ── Position progress bar ─────────────────────────────────────────────────────

def _progress_html(entry: float, sl: float, tp: float, current: float,
                   is_live: bool = False) -> str:
    span = tp - sl
    if span <= 0:
        return ""
    p_entry   = max(0.0, min(1.0, (entry   - sl) / span)) * 100
    p_current = max(0.0, min(1.0, (current - sl) / span)) * 100
    is_up     = current >= entry
    fc        = "#10B981" if is_up else "#F43F5E"
    left      = min(p_entry, p_current)
    width     = abs(p_current - p_entry)
    price_color = "#10B981" if is_live else "#40405A"
    price_dot   = "&#9679;" if is_live else "&#9711;"
    price_note  = "" if is_live else " (entry)"
    return (
        '<div class="progress-wrap">'
        '<div class="progress-track">'
        f'<div class="progress-fill" style="left:{left:.1f}%;width:{width:.1f}%;background:{fc}55;"></div>'
        f'<div class="pip" style="left:2px;width:7px;height:7px;background:#F43F5E;"></div>'
        f'<div class="pip" style="left:{p_entry:.1f}%;width:3px;height:12px;border-radius:2px;background:#6060A0;"></div>'
        f'<div class="pip" style="left:{p_current:.1f}%;width:4px;height:14px;border-radius:2px;background:{fc};"></div>'
        f'<div class="pip" style="right:2px;left:auto;transform:translate(0,-50%);width:7px;height:7px;background:#10B981;"></div>'
        '</div>'
        f'<div class="progress-labels"><span style="color:#F43F5E">SL {sl:.4f}</span><span>entry {entry:.4f}</span><span style="color:#10B981">TP {tp:.4f}</span></div>'
        f'<div style="text-align:center;margin-top:3px;font-family:\'Roboto Mono\',monospace;font-size:0.72rem;font-weight:500;color:{price_color};">{price_dot} Current: {current:.4f}{price_note}</div>'
        '</div>'
    )


# ── Positions Tab ─────────────────────────────────────────────────────────────

def render_positions(db: Database, config) -> None:
    open_pos = db.get_open_positions()

    if not open_pos:
        badge = '<span class="badge badge-empty">0 open</span>'
    else:
        badge = f'<span class="badge badge-open">{len(open_pos)} / {config.risk.max_simultaneous_positions} open</span>'

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:1rem;">'
        f'<span style="font-family:\'Inter\',sans-serif;font-size:0.9rem;'
        f'font-weight:600;color:#F0F0FC;">Open Positions</span>{badge}</div>',
        unsafe_allow_html=True,
    )

    if not open_pos:
        st.info("No open positions. The bot will open one when it finds a valid edge.")
        return

    eur_rate = config.paper_trading.eur_to_usd_rate

    token_ids   = tuple(p.token_id for p in open_pos)
    live_prices = fetch_live_prices(config, token_ids)
    prices_live = bool(live_prices)

    total_unrealized = 0.0

    for pos in open_pos:
        side_val   = pos.side.value if pos.side else "—"
        is_yes     = side_val == "BUY_YES"
        side_tag   = '<span class="tag-yes">YES</span>' if is_yes else '<span class="tag-no">NO</span>'
        card_class = "pos-card side-yes" if is_yes else "pos-card side-no"

        age_h = 0.0
        if pos.entry_timestamp:
            age_h = (datetime.now(timezone.utc) - pos.entry_timestamp).total_seconds() / 3600

        current_price = live_prices.get(pos.token_id, pos.entry_price)
        is_live       = pos.token_id in live_prices
        est_pnl_eur   = pos.current_pnl_eur(current_price)
        est_pnl_pct   = pos.current_pnl_pct(current_price)
        total_unrealized += est_pnl_eur

        pnl_cls = pnl_color_class(est_pnl_eur)
        sl_eur  = (pos.stop_loss_price   - pos.entry_price) * pos.tokens_quantity / eur_rate
        tp_eur  = (pos.take_profit_price - pos.entry_price) * pos.tokens_quantity / eur_rate

        poly_url = (
            f"https://polymarket.com/event/{pos.market_slug}"
            if pos.market_slug else None
        )
        progress = _progress_html(pos.entry_price, pos.stop_loss_price,
                                   pos.take_profit_price, current_price, is_live)

        q     = pos.market_question or ""
        title = html_mod.escape(q[:75] + ("…" if len(q) > 75 else ""))

        price_label = (
            f'<span style="font-family:\'Roboto Mono\',monospace;font-size:0.72rem;">'
            f'<span style="color:#10B981;font-size:0.55rem;">&#9679;</span> '
            f'{current_price:.4f}</span>'
            if is_live else
            f'<span style="font-family:\'Roboto Mono\',monospace;font-size:0.72rem;color:#40405A;">'
            f'&#9679; {current_price:.4f} (entry)</span>'
        )

        card = (
            f'<div class="{card_class}">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;">'
            f'<div style="flex:1;">'
            f'<div class="pos-title">{title}</div>'
            f'<div class="pos-meta">'
            f'<span>{side_tag}</span>'
            f'<span>Size <b style="color:#A0A0C0">€{pos.size_eur:.2f}</b></span>'
            f'<span>Conf <b style="color:#A0A0C0">{pos.confidence}</b>/100</span>'
            f'<span>Open <b style="color:#A0A0C0">{age_h:.1f}h</b></span>'
            f'<span>Now {price_label}</span>'
            f'</div></div>'
            f'<div style="text-align:right;flex-shrink:0;">'
            f'<div class="pos-pnl {pnl_cls}">€{est_pnl_eur:+.2f}</div>'
            f'<div style="font-family:\'Roboto Mono\',monospace;font-size:0.72rem;color:#6060A0;">{est_pnl_pct:+.2%}</div>'
            f'</div></div>'
            f'{progress}'
            f'<div style="display:flex;gap:1.5rem;margin-top:0.7rem;font-family:\'Roboto Mono\',monospace;font-size:0.72rem;color:#6060A0;">'
            f'<span>SL <b style="color:#F43F5E">€{sl_eur:+.2f}</b></span>'
            f'<span>TP <b style="color:#10B981">€{tp_eur:+.2f}</b></span>'
            + (f'<span><a href="{poly_url}" target="_blank" style="color:#6366F1;text-decoration:none;font-family:\'Inter\',sans-serif;font-size:0.75rem;">View on Polymarket →</a></span>' if poly_url else '<span style="color:#2A2A42;font-size:0.72rem;">No link (old position)</span>')
            + '</div></div>'
        )
        st.markdown(card, unsafe_allow_html=True)

    st.divider()
    total_invested = sum(p.size_eur for p in open_pos)
    bal_hist    = db.get_balance_history()
    current_bal = float(bal_hist[-1]["balance_eur"]) if bal_hist else 0.0
    free_bal    = max(0.0, current_bal - total_invested)

    c1, c2, c3 = st.columns(3)
    c1.metric("Capital in positions", f"€{total_invested:.2f}")
    c2.metric(
        "Unrealized P&L",
        f"€{total_unrealized:+.2f}",
        delta="live" if prices_live else "entry price (API unavailable)",
        delta_color="normal" if total_unrealized >= 0 else "inverse",
    )
    c3.metric("Free balance", f"€{free_bal:.2f}")


# ── History Tab ───────────────────────────────────────────────────────────────

def render_history(db: Database) -> None:
    all_trades = db.get_all_trades()
    closed     = [t for t in all_trades if t.status.value == "CLOSED"]

    st.markdown(
        f'<div style="font-family:\'Inter\',sans-serif;font-size:0.9rem;'
        f'font-weight:600;color:#F0F0FC;margin-bottom:1rem;">'
        f'Trade History <span style="color:#6060A0;font-size:0.72rem;font-weight:400;">'
        f'({len(closed)} closed)</span></div>',
        unsafe_allow_html=True,
    )

    if not closed:
        st.info("No closed trades yet.")
        return

    c1, c2 = st.columns(2)
    filter_side   = c1.selectbox("Side", ["All", "BUY_YES", "BUY_NO"])
    filter_result = c2.selectbox("Result", ["All", "Winners", "Losers"])

    filtered = closed
    if filter_side != "All":
        filtered = [t for t in filtered if t.side and t.side.value == filter_side]
    if filter_result == "Winners":
        filtered = [t for t in filtered if (t.pnl_eur or 0) > 0]
    elif filter_result == "Losers":
        filtered = [t for t in filtered if (t.pnl_eur or 0) <= 0]

    rows = [{
        "Market":  t.market_question[:48],
        "Side":    t.side.value if t.side else "",
        "Entry":   f"{t.entry_price:.4f}",
        "Exit":    f"{t.exit_price:.4f}" if t.exit_price else "—",
        "Size":    f"€{t.size_eur:.2f}",
        "P&L €":   f"{'+'if (t.pnl_eur or 0)>=0 else ''}€{(t.pnl_eur or 0):.2f}",
        "P&L %":   fmt_pct(t.pnl_pct or 0),
        "Reason":  t.close_reason.value if t.close_reason else "—",
        "Conf":    t.confidence,
    } for t in filtered]

    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=440)


# ── LLM Analysis Tab ──────────────────────────────────────────────────────────

def render_analyses(db: Database) -> None:
    st.markdown(
        '<div style="font-family:\'Inter\',sans-serif;font-size:0.9rem;'
        'font-weight:600;color:#F0F0FC;margin-bottom:1rem;">'
        'LLM Analyses <span style="color:#6060A0;font-size:0.72rem;font-weight:400;">(last 50)</span></div>',
        unsafe_allow_html=True,
    )

    try:
        cur  = db._conn.execute("SELECT * FROM analyses_log ORDER BY timestamp DESC LIMIT 50")
        rows = [dict(r) for r in cur.fetchall()]
    except Exception:
        rows = []

    if not rows:
        st.info("No analyses recorded yet.")
        return

    data = [{
        "Time":       r["timestamp"][:19],
        "Market":     (r["market_question"] or "")[:42],
        "YES price":  f"{r['current_yes_price']:.3f}",
        "Prob":       f"{r['consensus_probability_yes']:.3f}",
        "Edge":       fmt_pct(r["edge"] or 0),
        "Conf":       r["confidence"],
        "Rec":        r["recommendation"],
        "Articles":   r["num_articles_analyzed"],
        "Tokens":     f"{(r['llm_input_tokens'] or 0)+(r['llm_output_tokens'] or 0):,}",
    } for r in rows]
    st.dataframe(pd.DataFrame(data), use_container_width=True, height=360)

    from collections import Counter
    recs = [r["recommendation"] for r in rows if r.get("recommendation")]
    if recs:
        st.markdown('<div class="section-header">Recommendation distribution</div>',
                    unsafe_allow_html=True)
        counts     = Counter(recs)
        colors_map = {
            "BUY_YES":           "#6366F1",
            "BUY_NO":            "#F43F5E",
            "WAIT":              "#F59E0B",
            "INSUFFICIENT_DATA": "#6060A0",
        }
        fig = go.Figure(go.Bar(
            x=list(counts.keys()), y=list(counts.values()),
            marker_color=[colors_map.get(k, "#A0A0C0") for k in counts.keys()],
            marker_line_width=0,
            text=list(counts.values()), textposition="outside",
            textfont=dict(color="#6060A0", size=11),
        ))
        _chart_style(fig, height=190)
        st.plotly_chart(fig, use_container_width=True)


# ── Balance Manager Tab ───────────────────────────────────────────────────────

def render_balance_manager(db: Database, config) -> None:
    st.markdown(
        '<div style="font-family:\'Inter\',sans-serif;font-size:0.9rem;'
        'font-weight:600;color:#F0F0FC;margin-bottom:1rem;">Virtual Balance</div>',
        unsafe_allow_html=True,
    )

    history  = db.get_balance_history()
    current  = float(history[-1]["balance_eur"]) if history else config.paper_trading.initial_balance_eur
    peak     = float(history[-1]["peak_balance"]) if history else current
    open_pos = db.get_open_positions()

    c1, c2, c3 = st.columns(3)
    c1.metric("Current balance",    f"€{current:.2f}")
    c2.metric("All-time peak",      f"€{peak:.2f}")
    c3.metric("Initial (config)",   f"€{config.paper_trading.initial_balance_eur:.2f}")

    st.divider()

    action = st.radio(
        "Action",
        ["Reset to initial", "Reset to custom", "Add funds", "Withdraw funds"],
        horizontal=True,
    )

    def _log(new_bal: float, event: str) -> None:
        new_peak = max(new_bal, peak)
        dd       = max(0.0, (new_peak - new_bal) / new_peak) if new_peak > 0 else 0.0
        db.log_balance(new_bal, new_peak, dd, len(open_pos), event)

    if action == "Reset to initial":
        target = config.paper_trading.initial_balance_eur
        st.warning(f"Will reset balance to €{target:.2f}. Open positions are NOT closed.")
        if st.button("Confirm reset", type="primary"):
            _log(target, "MANUAL_RESET")
            st.success(f"Balance reset to €{target:.2f}")
            st.rerun()

    elif action == "Reset to custom":
        new_val = st.number_input("New balance (€)", min_value=1.0, max_value=50000.0,
                                  value=current, step=10.0)
        if st.button("Confirm custom reset", type="primary"):
            _log(new_val, "MANUAL_RESET")
            st.success(f"Balance reset to €{new_val:.2f}")
            st.rerun()

    elif action == "Add funds":
        amount = st.number_input("Amount to add (€)", min_value=1.0, max_value=50000.0,
                                 value=50.0, step=10.0)
        st.info(f"Resulting balance: €{current + amount:.2f}")
        if st.button("Confirm deposit", type="primary"):
            _log(current + amount, "MANUAL_ADD")
            st.success(f"Added €{amount:.2f}")
            st.rerun()

    else:
        max_ret = max(1.0, current - 1.0)
        amount  = st.number_input("Amount to withdraw (€)", min_value=1.0,
                                  max_value=max_ret, value=min(10.0, max_ret), step=5.0)
        st.info(f"Resulting balance: €{current - amount:.2f}")
        if st.button("Confirm withdrawal", type="primary"):
            _log(current - amount, "MANUAL_SUBTRACT")
            st.success(f"Withdrew €{amount:.2f}")
            st.rerun()

    st.divider()
    st.markdown('<div class="section-header">Manual adjustments log</div>',
                unsafe_allow_html=True)
    manual = [h for h in history if (h.get("event") or "").startswith("MANUAL")]
    if manual:
        st.dataframe(pd.DataFrame([{
            "Timestamp": h["timestamp"][:19],
            "Event":     h["event"],
            "Balance":   f"€{h['balance_eur']:.2f}",
        } for h in reversed(manual)]), use_container_width=True)
    else:
        st.caption("No manual adjustments yet.")


# ── Backtest Tab ──────────────────────────────────────────────────────────────

def render_backtest(config) -> None:
    st.markdown(
        '<div style="font-family:\'Inter\',sans-serif;font-size:0.9rem;'
        'font-weight:600;color:#F0F0FC;margin-bottom:0.3rem;">Backtesting</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Simulate the bot on resolved markets. "
        "_current_ mode: today's news (fast, look-ahead bias). "
        "_replay_ mode: historical GDELT news (slower)."
    )

    col1, col2, col3 = st.columns(3)
    mode      = col1.selectbox("Mode", ["current", "replay"])
    n_markets = col2.slider("Markets", 5, 100, 20)
    balance   = col3.number_input(
        "Initial balance (€)", min_value=10.0, max_value=10000.0,
        value=float(config.paper_trading.initial_balance_eur), step=10.0,
    )

    col_a, col_b = st.columns(2)
    export_xl = col_a.checkbox("Export Excel on finish")
    col_b.markdown(
        f'<div style="font-family:\'Roboto Mono\',monospace;font-size:0.75rem;'
        f'color:#6060A0;padding-top:0.6rem;">'
        f'LLM: {config.llm.provider} / {config.llm.model.split(":")[0]}</div>',
        unsafe_allow_html=True,
    )

    if st.button("Run Backtest", type="primary"):
        with st.spinner(f"Analyzing {n_markets} markets… this may take a few minutes."):
            try:
                from src.backtester import Backtester
                result = Backtester(
                    config=config, mode=mode,
                    max_markets=n_markets, initial_balance=balance,
                ).run()
                st.session_state["last_backtest"] = result
            except Exception as exc:
                st.error(f"Error: {exc}")
                return

    if "last_backtest" not in st.session_state:
        return

    result = st.session_state["last_backtest"]
    st.divider()
    st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Markets",   result.markets_analyzed)
    c2.metric("Trades",    result.trades_executed)
    c3.metric("Win Rate",  f"{result.win_rate:.1%}")
    c4.metric("P&L",
              f"{'+'if result.total_pnl_eur>=0 else ''}€{result.total_pnl_eur:.2f}",
              delta_color="normal" if result.total_pnl_eur >= 0 else "inverse")
    c5.metric("Max DD",    f"{result.max_drawdown_pct:.2%}")

    c6, c7, c8 = st.columns(3)
    c6.metric("Final Balance", f"€{result.final_balance:.2f}")
    retorno = (result.final_balance - result.initial_balance) / result.initial_balance
    c7.metric("Total Return",  fmt_pct(retorno),
              delta_color="normal" if retorno >= 0 else "inverse")
    c8.metric("Sharpe (approx)", f"{result.sharpe_ratio:.2f}")

    executed = [t for t in result.trades
                if str(t.decision) in ("OPEN_TRADE", "DecisionAction.OPEN_TRADE")]
    if not executed:
        return

    st.markdown(f'<div class="section-header">Executed trades ({len(executed)})</div>',
                unsafe_allow_html=True)
    rows = [{
        "Market":   t.market_question[:48],
        "YES won":  "Yes" if t.resolved_yes else "No",
        "Side":     t.side.value if hasattr(t.side, "value") else str(t.side),
        "P&L €":    f"{'+'if t.pnl_eur>=0 else ''}€{t.pnl_eur:.2f}",
        "P&L %":    fmt_pct(t.pnl_pct),
        "Conf":     t.confidence,
        "Edge":     fmt_pct(t.edge),
        "Articles": t.num_articles,
        "Low-info": "Yes" if t.is_low_info else "",
        "LLM rec":  t.llm_recommendation,
    } for t in executed]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    running = result.initial_balance
    curve   = []
    for t in executed:
        running += t.pnl_eur
        curve.append(running)

    st.markdown('<div class="section-header">Simulated balance curve</div>',
                unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(curve)+1)), y=curve,
        mode="lines+markers",
        line=dict(color="#6366F1", width=2),
        marker=dict(
            color=["#10B981" if v >= result.initial_balance else "#F43F5E" for v in curve],
            size=5,
        ),
    ))
    fig.add_hline(y=result.initial_balance, line_dash="dash", line_color="#2A2A42")
    _chart_style(fig, height=240)
    fig.update_layout(xaxis_title="Trade #", yaxis_title="Balance (€)")
    st.plotly_chart(fig, use_container_width=True)

    if export_xl and executed:
        try:
            from scripts.run_backtest import _export_excel
            _export_excel(result, config)
            st.success("Excel exported to reports/")
        except Exception as e:
            st.warning(f"Excel not exported: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    config         = get_config()
    refresh, _     = render_sidebar(config)

    now_str = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding:0.2rem 0 1.2rem;border-bottom:1px solid #262640;
                margin-bottom:1.2rem;">
      <div>
        <div style="font-family:'Inter',sans-serif;font-size:1.5rem;font-weight:700;
                    color:#6366F1;letter-spacing:0.5px;">
          POLYBOT
        </div>
        <div style="font-family:'Inter',sans-serif;font-size:0.75rem;
                    color:#40405A;letter-spacing:0.5px;margin-top:2px;">
          Paper Trading Terminal · Polymarket
        </div>
      </div>
      <div style="text-align:right;">
        <div style="font-family:'Roboto Mono',monospace;font-size:0.8rem;
                    color:#6366F1;letter-spacing:1px;">
          <span class="dot dot-live"></span>{now_str}
        </div>
        <div style="font-family:'Roboto Mono',monospace;font-size:0.65rem;
                    color:#40405A;margin-top:2px;">
          refresh every {refresh}s
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    db = get_db(config)

    if not db.get_balance_history():
        st.info("No data yet. Start the bot: `python main.py`")

    tabs = st.tabs([
        "Overview",
        "Positions",
        "History",
        "LLM Analysis",
        "Balance",
        "Backtest",
    ])

    with tabs[0]: render_overview(db, config)
    with tabs[1]: render_positions(db, config)
    with tabs[2]: render_history(db)
    with tabs[3]: render_analyses(db)
    with tabs[4]: render_balance_manager(db, config)
    with tabs[5]: render_backtest(config)

    db.close()

    time.sleep(refresh)
    st.rerun()


if __name__ == "__main__":
    main()
