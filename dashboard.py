"""
Polymarket Paper Trading Bot Dashboard
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

PROJECT_ROOT    = Path(__file__).resolve().parent
_OVERRIDES_PATH = PROJECT_ROOT / "data" / "overrides.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config  # noqa: E402
from src.database import Database          # noqa: E402


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


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Polymarket Portfolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

/* ── Ghost/stale state ── */
[data-stale="true"]            { display: none !important; }
.stAppRunningIndicator         { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }

/* ══════════════════════════════════════════════════════
   DESIGN TOKENS
   ══════════════════════════════════════════════════════ */
:root {
    /* Backgrounds */
    --bg:            #F2F4F7;
    --card:          #FFFFFF;
    --card-hover:    #FAFBFC;
    --border:        #E4E8EF;
    --border2:       #CDD3DC;

    /* Sidebar */
    --sb-bg:         #141B2D;
    --sb-surface:    #1C253D;
    --sb-border:     #252F47;
    --sb-text:       #C8D0E0;
    --sb-muted:      #5A6580;
    --sb-accent:     #4F7CF6;

    /* Accent — electric blue */
    --blue:          #2563EB;
    --blue-mid:      rgba(37,99,235,0.15);
    --blue-dim:      rgba(37,99,235,0.07);
    --blue-bright:   #3B82F6;

    /* Semantic */
    --green:         #059669;
    --green-dim:     rgba(5,150,105,0.08);
    --green-mid:     rgba(5,150,105,0.18);
    --red:           #E11D48;
    --red-dim:       rgba(225,29,72,0.08);
    --amber:         #D97706;
    --amber-dim:     rgba(217,119,6,0.08);

    /* Text */
    --text:          #0F1623;
    --text2:         #4A5568;
    --muted:         #8A96A8;
    --muted2:        #C5CDD8;

    /* Shadows */
    --shadow-sm:     0 1px 3px rgba(15,22,35,0.06), 0 1px 2px rgba(15,22,35,0.04);
    --shadow-md:     0 4px 12px rgba(15,22,35,0.08), 0 2px 4px rgba(15,22,35,0.04);
    --shadow-lg:     0 8px 24px rgba(15,22,35,0.10), 0 2px 8px rgba(15,22,35,0.06);
    --radius:        12px;
    --radius-sm:     8px;
}

/* ══════════════════════════════════════════════════════
   BASE — force light theme on main area
   ══════════════════════════════════════════════════════ */
.stApp, .stApp > * {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}
section.main .block-container {
    background: var(--bg) !important;
    padding-top: 1.5rem !important;
    max-width: 1280px;
}

/* Force all native text in main area to dark */
.main p, .main span, .main label, .main div {
    color: var(--text) !important;
}
.main h1, .main h2, .main h3, .main h4 {
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
}

/* ══════════════════════════════════════════════════════
   SIDEBAR
   ══════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: var(--sb-bg) !important;
    border-right: 1px solid var(--sb-border) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: var(--sb-text) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--sb-text) !important;
}
/* Slider thumb */
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    background: var(--sb-accent) !important;
}

/* ══════════════════════════════════════════════════════
   NATIVE STREAMLIT METRICS → hidden, replaced by custom HTML
   ══════════════════════════════════════════════════════ */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.1rem 1.2rem !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.68rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
}

/* ══════════════════════════════════════════════════════
   BUTTONS
   ══════════════════════════════════════════════════════ */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    border: 1.5px solid var(--blue) !important;
    color: var(--blue) !important;
    background: transparent !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.45rem 1.3rem !important;
    transition: all 0.18s ease !important;
    cursor: pointer;
    letter-spacing: 0.1px;
}
.stButton > button:hover {
    background: var(--blue-dim) !important;
    color: var(--blue-bright) !important;
}
.stButton > button[kind="primary"] {
    background: var(--blue) !important;
    color: #FFFFFF !important;
    border-color: var(--blue) !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--blue-bright) !important;
    box-shadow: 0 4px 12px var(--blue-mid) !important;
}

/* ══════════════════════════════════════════════════════
   TABS
   ══════════════════════════════════════════════════════ */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 4px !important;
    gap: 2px !important;
    margin-bottom: 1.2rem !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.80rem !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    padding: 0.5rem 1.1rem !important;
    border-radius: var(--radius-sm) !important;
    border: none !important;
    transition: all 0.18s ease !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
    color: var(--text2) !important;
    background: var(--bg) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: var(--blue) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ══════════════════════════════════════════════════════
   INPUTS & FORMS
   ══════════════════════════════════════════════════════ */
.main [data-baseweb="input"] input,
.main [data-baseweb="textarea"] textarea,
.main [data-testid="stNumberInput"] input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.main [data-testid="stSelectbox"] [data-baseweb="select"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

/* ══════════════════════════════════════════════════════
   MISC NATIVE
   ══════════════════════════════════════════════════════ */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.25rem 0 !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stAlert"] {
    background: var(--blue-dim) !important;
    border-radius: var(--radius-sm) !important;
    border-left: 3px solid var(--blue) !important;
    color: var(--text) !important;
}
[data-testid="stAlert"] p { color: var(--text) !important; }
.main [data-testid="stRadio"] label { color: var(--text2) !important; }
.main [data-testid="stCaption"],
.main [data-testid="stCaptionContainer"],
.main small { color: var(--muted) !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

/* ══════════════════════════════════════════════════════
   CUSTOM COMPONENTS
   ══════════════════════════════════════════════════════ */

/* ── Portfolio hero ── */
.portfolio-hero {
    background: linear-gradient(130deg, #FFFFFF 0%, #EEF4FF 100%);
    border: 1px solid #D6E4FF;
    border-radius: 16px;
    padding: 1.8rem 2rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: var(--shadow-md);
    position: relative;
    overflow: hidden;
}
.portfolio-hero::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(37,99,235,0.07) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    color: #7A90B4;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    margin-bottom: 0.4rem;
}
.hero-balance {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3rem;
    font-weight: 600;
    color: #0F1623;
    letter-spacing: -1.5px;
    line-height: 1;
    margin-bottom: 0.45rem;
}
.hero-pnl {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.92rem;
    font-weight: 500;
}
.hero-pnl.pos { color: #059669; }
.hero-pnl.neg { color: #E11D48; }
.hero-initial {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    color: #8A96A8;
    margin-top: 0.3rem;
}

/* ── Stat cards row ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.2rem;
}
.stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.1rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.18s;
}
.stat-card:hover { box-shadow: var(--shadow-md); }
.stat-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.67rem;
    font-weight: 500;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.35rem;
}
.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.1;
}
.stat-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.70rem;
    color: var(--muted);
    margin-top: 0.25rem;
}
.stat-sub.pos { color: var(--green); }
.stat-sub.neg { color: var(--red); }

/* ── Section header ── */
.sh {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.7px;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* ── Live dot ── */
.dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    vertical-align: middle;
    margin-right: 5px;
}
.dot-live { background: var(--green); box-shadow: 0 0 5px var(--green); animation: pulse 2s infinite; }
.dot-off  { background: var(--muted); }
@keyframes pulse {
    0%,100% { opacity: 1; box-shadow: 0 0 5px var(--green); }
    50%      { opacity: 0.4; box-shadow: 0 0 2px var(--green); }
}

/* ── Holding card (position row) ── */
.holding {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.18s, border-color 0.18s;
    position: relative;
    overflow: hidden;
}
.holding::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: var(--border);
}
.holding.side-yes::before { background: var(--blue); }
.holding.side-no::before  { background: var(--red); }
.holding:hover {
    box-shadow: var(--shadow-md);
    border-color: var(--border2);
}
.holding-name {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.3rem;
    line-height: 1.4;
}
.holding-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    color: var(--muted);
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
}
.holding-meta b { color: var(--text2); font-weight: 500; }
.holding-pnl {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 1.1rem;
    text-align: right;
    line-height: 1.1;
}
.holding-pnl.pos { color: var(--green); }
.holding-pnl.neg { color: var(--red); }
.holding-pnl.neu { color: var(--muted); }
.holding-pct {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    text-align: right;
}
.holding-pct.pos { color: var(--green); }
.holding-pct.neg { color: var(--red); }

/* ── Badge pill ── */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 9px;
    border-radius: 20px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-yes  { background: var(--blue-dim); color: var(--blue); border: 1px solid rgba(37,99,235,0.2); }
.badge-no   { background: var(--red-dim);  color: var(--red);  border: 1px solid rgba(225,29,72,0.2); }
.badge-open { background: var(--green-dim); color: var(--green); border: 1px solid rgba(5,150,105,0.2); }
.badge-empty{ background: var(--bg); color: var(--muted); border: 1px solid var(--border); }

/* ── Price track ── */
.price-track-wrap { margin: 0.7rem 0 0.3rem; }
.price-track {
    position: relative;
    height: 3px;
    background: var(--bg);
    border-radius: 3px;
    border: 1px solid var(--border);
}
.price-fill {
    position: absolute;
    top: 0; height: 100%;
    border-radius: 3px;
}
.price-pip {
    position: absolute;
    top: 50%;
    transform: translate(-50%, -50%);
    border-radius: 50%;
}
.price-labels {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.60rem;
    color: var(--muted);
    margin-top: 4px;
}

/* ── Sidebar components ── */
.sb-logo-wrap {
    padding: 1.4rem 1rem 1rem;
    border-bottom: 1px solid var(--sb-border);
}
.sb-portfolio-wrap {
    padding: 1rem 1rem 0.8rem;
    border-bottom: 1px solid var(--sb-border);
    background: var(--sb-surface);
}
.sb-section {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.58rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--sb-muted) !important;
    padding: 0.8rem 1rem 0.25rem;
}
.sb-ctrl {
    padding: 0 1rem 0.3rem;
}
.sb-source {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    padding: 3px 1rem;
}
.sb-kv {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.70rem;
    padding: 0 1rem 0.2rem;
}

/* ── Page header ── */
.page-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.4rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}
.page-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text);
}
.page-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 1px;
}
.page-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    text-align: right;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource
def get_config():
    return load_config()


@st.cache_data(ttl=10)
def fetch_live_prices(_config, token_ids: tuple[str, ...]) -> dict[str, float]:
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
    return f"{'+'if v>0 else ''}€{v:.2f}"


def fmt_pct(v: float) -> str:
    return f"{'+'if v>0 else ''}{v:.2%}"


def pnl_cls(v: float) -> str:
    return "pos" if v > 0 else ("neg" if v < 0 else "neu")


def _stat_card(label: str, value: str, sub: str = "", sub_cls: str = "neutral") -> str:
    sub_html = (
        f'<div class="stat-sub {sub_cls}">{sub}</div>' if sub else ""
    )
    return (
        f'<div class="stat-card">'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>'
        f'{sub_html}'
        f'</div>'
    )


def _sh(title: str, extra: str = "") -> str:
    return (
        f'<div class="sh">'
        f'<span>{title}</span>'
        f'<span style="font-weight:400;font-size:0.65rem;">{extra}</span>'
        f'</div>'
    )


def _chart_style(fig: go.Figure, height: int = 280) -> None:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8A96A8", family="DM Sans", size=11),
        xaxis=dict(
            gridcolor="rgba(15,22,35,0.05)",
            showline=False, title="",
            tickcolor="#C5CDD8", tickfont=dict(family="JetBrains Mono", size=10),
        ),
        yaxis=dict(
            gridcolor="rgba(15,22,35,0.05)",
            showline=False,
            tickfont=dict(family="JetBrains Mono", size=10),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, family="DM Sans"),
        ),
        height=height,
        margin=dict(l=4, r=4, t=28, b=4),
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(config, history: list, open_pos: list) -> tuple[int, int]:
    with st.sidebar:
        # Brand
        st.markdown("""
        <div class="sb-logo-wrap">
            <div style="display:flex;align-items:center;gap:0.5rem;">
                <div style="width:30px;height:30px;background:#2563EB;border-radius:8px;
                            display:flex;align-items:center;justify-content:center;
                            font-size:0.9rem;">📈</div>
                <div>
                    <div style="font-family:'DM Sans',sans-serif;font-size:0.95rem;
                                font-weight:700;color:#E2E8F0;">PolyBot</div>
                    <div style="font-family:'DM Sans',sans-serif;font-size:0.60rem;
                                color:#5A6580;margin-top:1px;letter-spacing:0.5px;">
                        Paper Trading
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Portfolio summary
        bal = float(history[-1]["balance_eur"]) if history else config.paper_trading.initial_balance_eur
        initial = config.paper_trading.initial_balance_eur
        pnl = bal - initial
        pnl_sign = "+" if pnl >= 0 else ""
        pnl_color = "#059669" if pnl >= 0 else "#E11D48"

        st.markdown(f"""
        <div class="sb-portfolio-wrap">
            <div style="font-family:'DM Sans',sans-serif;font-size:0.62rem;
                        font-weight:500;color:#5A6580;text-transform:uppercase;
                        letter-spacing:0.6px;margin-bottom:0.4rem;">Portfolio value</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.5rem;
                        font-weight:600;color:#E2E8F0;letter-spacing:-0.5px;
                        line-height:1;">€{bal:,.2f}</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.78rem;
                        font-weight:500;color:{pnl_color};margin-top:0.3rem;">
                {pnl_sign}€{pnl:.2f} all time
            </div>
            <div style="margin-top:0.6rem;display:flex;gap:1rem;">
                <div style="font-family:'DM Sans',sans-serif;font-size:0.70rem;color:#5A6580;">
                    <span style="color:#8A96A8">Open</span>&nbsp;
                    <b style="color:#C8D0E0">{len(open_pos)}/{config.risk.max_simultaneous_positions}</b>
                </div>
                <div style="font-family:'DM Sans',sans-serif;font-size:0.70rem;color:#5A6580;">
                    <span style="color:#8A96A8">Initial</span>&nbsp;
                    <b style="color:#C8D0E0">€{initial:.0f}</b>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Controls
        st.markdown('<div class="sb-section">Settings</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-ctrl">', unsafe_allow_html=True)

        refresh = st.slider("↻ Refresh (s)", 5, 120, 10, step=5, key="refresh_slider")

        if "max_pos_slider" not in st.session_state:
            st.session_state["max_pos_slider"] = _read_override(
                "max_simultaneous_positions", config.risk.max_simultaneous_positions
            )
        max_pos = st.slider(
            "⚖ Max positions", min_value=1, max_value=15, step=1,
            key="max_pos_slider",
        )
        if max_pos != config.risk.max_simultaneous_positions:
            config.risk.max_simultaneous_positions = max_pos
            _write_override("max_simultaneous_positions", max_pos)
        else:
            config.risk.max_simultaneous_positions = max_pos

        if "llm_workers_slider" not in st.session_state:
            st.session_state["llm_workers_slider"] = _read_override(
                "llm_parallelism", getattr(config.llm, "llm_parallelism", 1)
            )
        llm_workers = st.slider(
            "⚡ LLM workers", min_value=1, max_value=4, step=1,
            key="llm_workers_slider",
        )
        _write_override("llm_parallelism", llm_workers)
        if llm_workers > 1:
            st.caption(f"⚠ Needs OLLAMA_NUM_PARALLEL={llm_workers}")

        st.markdown('</div>', unsafe_allow_html=True)

        # News sources
        st.markdown('<div class="sb-section">News Sources</div>', unsafe_allow_html=True)
        for name, enabled in [
            ("GDELT", config.news.gdelt.enabled),
            ("NewsAPI", config.news.newsapi.enabled),
            ("Telegram", config.news.telegram.enabled),
        ]:
            color = "#059669" if enabled else "#5A6580"
            dot = "dot-live" if enabled else "dot-off"
            st.markdown(
                f'<div class="sb-source">'
                f'<span class="dot {dot}" style="background:{color};box-shadow:{"0 0 4px "+color if enabled else "none"};"></span>'
                f'<span style="color:{color};font-family:\'DM Sans\',sans-serif;font-size:0.78rem;">{name}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Risk params
        st.markdown('<div class="sb-section">Risk Parameters</div>', unsafe_allow_html=True)
        for label, val, color in [
            ("Stop-loss",  f"{config.risk.stop_loss_pct:.0%}",   "#E11D48"),
            ("Take-profit",f"{config.risk.take_profit_pct:.0%}", "#059669"),
            ("Max DD",     f"{config.risk.max_drawdown_pct:.0%}","#D97706"),
            ("Max size",   f"{config.risk.max_position_size_pct:.0%}","#4F7CF6"),
        ]:
            st.markdown(
                f'<div class="sb-kv">'
                f'<span style="color:#5A6580;font-family:\'DM Sans\',sans-serif;font-size:0.72rem;">{label}</span>'
                f'<span style="color:{color};font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;font-weight:600;">{val}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # System
        st.markdown('<div class="sb-section">System</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;'
            f'color:#5A6580;line-height:1.9;padding:0 1rem 0.5rem;">'
            f'v{config.app.version}<br>'
            f'{config.llm.provider} / {config.llm.model.split(":")[0]}</div>',
            unsafe_allow_html=True,
        )

        # ── Kill Switch ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Emergency Controls")

        _kill_active = bool(_read_override("kill_switch_active", False))

        if _kill_active:
            st.sidebar.error("🛑 KILL SWITCH ACTIVE — Bot halted, closing positions")
            if st.sidebar.button("Resume Trading", type="secondary"):
                _write_override("kill_switch_active", False)
                st.sidebar.success("Kill switch deactivated. Bot will resume on next cycle.")
                st.rerun()
        else:
            if st.sidebar.button("⚡ Emergency Stop — Close All Positions", type="primary"):
                _write_override("kill_switch_active", True)
                st.sidebar.error("Kill switch activated. Bot will close all positions on next cycle.")
                st.rerun()

    return refresh, max_pos


# ── Overview Tab ──────────────────────────────────────────────────────────────

def render_overview(db: Database, config) -> None:
    history    = db.get_balance_history()
    open_pos   = db.get_open_positions()
    all_trades = db.get_all_trades()
    closed     = [t for t in all_trades if t.status.value == "CLOSED"]

    bal      = float(history[-1]["balance_eur"]) if history else config.paper_trading.initial_balance_eur
    peak     = float(history[-1]["peak_balance"]) if history else bal
    drawdown = (peak - bal) / peak if peak > 0 else 0.0
    initial  = config.paper_trading.initial_balance_eur
    pnl      = bal - initial
    pnl_pct  = pnl / initial if initial > 0 else 0.0
    winners  = [t for t in closed if (t.pnl_eur or 0) > 0]
    win_rate = len(winners) / len(closed) if closed else 0.0

    # ── Portfolio hero ────────────────────────────────────────────────────────
    pnl_sign = "+" if pnl >= 0 else ""
    hero_cls = "pos" if pnl >= 0 else "neg"
    pnl_arrow = "↑" if pnl >= 0 else "↓"
    st.markdown(f"""
    <div class="portfolio-hero">
        <div class="hero-label">Total Portfolio Value</div>
        <div class="hero-balance">€{bal:,.2f}</div>
        <div class="hero-pnl {hero_cls}">
            {pnl_arrow} {pnl_sign}€{pnl:.2f} &nbsp;({fmt_pct(pnl_pct)}) since inception
        </div>
        <div class="hero-initial">Started at €{initial:.2f} · Peak €{peak:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick stats ───────────────────────────────────────────────────────────
    dd_sub_cls = "neg" if drawdown > config.risk.max_drawdown_pct * 0.7 else "neutral"
    st.markdown(
        '<div class="stat-row">'
        + _stat_card("Win Rate", f"{win_rate:.1%}", f"{len(winners)}/{len(closed)} trades")
        + _stat_card("Drawdown", f"{drawdown:.2%}", f"limit {config.risk.max_drawdown_pct:.0%}", dd_sub_cls)
        + _stat_card("Positions", f"{len(open_pos)} / {config.risk.max_simultaneous_positions}", "open now")
        + _stat_card("Closed", str(len(closed)), "all time")
        + '</div>',
        unsafe_allow_html=True,
    )

    if len(history) < 2:
        st.info("No history yet — start the bot: `python main.py`")
        return

    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── Balance chart (full width) ────────────────────────────────────────────
    st.markdown(_sh("Portfolio Performance"), unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["balance_eur"],
        mode="lines", name="Balance",
        line=dict(color="#2563EB", width=2.5, shape="spline", smoothing=0.3),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.06)",
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["peak_balance"],
        mode="lines", name="Peak",
        line=dict(color="#CBD5E1", width=1, dash="dot"),
    ))
    fig.add_hline(y=initial, line_dash="dash", line_color="#E4E8EF",
                  annotation_text="Start", annotation_font_color="#8A96A8",
                  annotation_font_size=10)
    _chart_style(fig, height=260)
    st.plotly_chart(fig, use_container_width=True)

    # ── Drawdown + P&L bars ───────────────────────────────────────────────────
    col_l, col_r = st.columns([2, 3])

    with col_l:
        st.markdown(_sh("Drawdown"), unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["timestamp"], y=df["drawdown_pct"] * 100,
            mode="lines",
            line=dict(color="#E11D48", width=1.5, shape="spline", smoothing=0.3),
            fill="tozeroy",
            fillcolor="rgba(225,29,72,0.06)",
        ))
        fig2.add_hline(
            y=config.risk.max_drawdown_pct * 100,
            line_dash="dash", line_color="#FCD34D",
            annotation_text=f"Limit {config.risk.max_drawdown_pct:.0%}",
            annotation_font_color="#D97706", annotation_font_size=10,
        )
        _chart_style(fig2, height=200)
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        if closed:
            st.markdown(_sh("P&L per Trade", "last 20"), unsafe_allow_html=True)
            pnls   = [(t.pnl_eur or 0) for t in closed[-20:]]
            colors = ["#059669" if p >= 0 else "#E11D48" for p in pnls]
            fig3   = go.Figure(go.Bar(
                x=list(range(1, len(pnls)+1)), y=pnls,
                marker_color=colors,
                marker_line_width=0,
                marker_cornerradius=3,
                text=[f"€{p:+.2f}" for p in pnls],
                textposition="outside",
                textfont=dict(size=8, color="#8A96A8"),
            ))
            _chart_style(fig3, height=200)
            st.plotly_chart(fig3, use_container_width=True)


# ── Price track ───────────────────────────────────────────────────────────────

def _price_track(entry: float, sl: float, tp: float, current: float,
                 is_live: bool = False) -> str:
    span = tp - sl
    if span <= 0:
        return ""
    p_entry   = max(0.0, min(1.0, (entry   - sl) / span)) * 100
    p_current = max(0.0, min(1.0, (current - sl) / span)) * 100
    is_up     = current >= entry
    fc        = "#059669" if is_up else "#E11D48"
    left      = min(p_entry, p_current)
    width     = abs(p_current - p_entry)
    cur_color = "#2563EB" if is_live else "#8A96A8"
    cur_label = f"◉ {current:.4f}" if is_live else f"○ {current:.4f} (entry)"
    return (
        '<div class="price-track-wrap">'
        '<div class="price-track">'
        f'<div class="price-fill" style="left:{left:.1f}%;width:{width:.1f}%;background:{fc}44;"></div>'
        f'<div class="price-pip" style="left:2%;width:8px;height:8px;background:#E11D48;border-radius:2px;"></div>'
        f'<div class="price-pip" style="left:{p_entry:.1f}%;width:2px;height:10px;background:#8A96A8;border-radius:1px;"></div>'
        f'<div class="price-pip" style="left:{p_current:.1f}%;width:3px;height:12px;background:{fc};border-radius:1px;"></div>'
        f'<div class="price-pip" style="right:2%;left:auto;transform:translate(0,-50%);width:8px;height:8px;background:#059669;border-radius:2px;"></div>'
        '</div>'
        f'<div class="price-labels">'
        f'<span style="color:#E11D48">SL {sl:.4f}</span>'
        f'<span style="color:{cur_color};font-weight:500;">{cur_label}</span>'
        f'<span style="color:#059669">TP {tp:.4f}</span>'
        f'</div>'
        '</div>'
    )


# ── Positions Tab ─────────────────────────────────────────────────────────────

def render_positions(db: Database, config) -> None:
    open_pos = db.get_open_positions()

    count_badge = (
        f'<span class="badge badge-open">{len(open_pos)} open</span>'
        if open_pos else
        f'<span class="badge badge-empty">0 open</span>'
    )
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:1rem;">'
        f'<span style="font-family:\'DM Sans\',sans-serif;font-size:1.05rem;'
        f'font-weight:700;color:var(--text);">Positions</span>'
        f'{count_badge}</div>',
        unsafe_allow_html=True,
    )

    if not open_pos:
        st.info("No open positions. The bot will open one when it finds a valid edge.")
        return

    eur_rate    = config.paper_trading.eur_to_usd_rate
    token_ids   = tuple(p.token_id for p in open_pos)
    live_prices = fetch_live_prices(config, token_ids)
    prices_live = bool(live_prices)
    total_unrealized = 0.0

    for pos in open_pos:
        side_val = pos.side.value if pos.side else "—"
        is_yes   = side_val == "BUY_YES"
        side_badge = '<span class="badge badge-yes">YES</span>' if is_yes else '<span class="badge badge-no">NO</span>'
        card_cls   = "holding side-yes" if is_yes else "holding side-no"

        age_h = 0.0
        if pos.entry_timestamp:
            age_h = (datetime.now(timezone.utc) - pos.entry_timestamp).total_seconds() / 3600

        current_price    = live_prices.get(pos.token_id, pos.entry_price)
        is_live          = pos.token_id in live_prices
        est_pnl_eur      = pos.current_pnl_eur(current_price)
        est_pnl_pct      = pos.current_pnl_pct(current_price)
        total_unrealized += est_pnl_eur

        pc  = pnl_cls(est_pnl_eur)
        sl_eur = (pos.stop_loss_price   - pos.entry_price) * pos.tokens_quantity / eur_rate
        tp_eur = (pos.take_profit_price - pos.entry_price) * pos.tokens_quantity / eur_rate

        poly_url = (
            f"https://polymarket.com/event/{pos.market_slug}"
            if pos.market_slug else None
        )
        track = _price_track(pos.entry_price, pos.stop_loss_price,
                              pos.take_profit_price, current_price, is_live)

        q     = pos.market_question or ""
        title = html_mod.escape(q[:80] + ("…" if len(q) > 80 else ""))

        link_html = (
            f'<a href="{poly_url}" target="_blank" '
            f'style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;'
            f'font-weight:500;color:#2563EB;text-decoration:none;">View market ↗</a>'
            if poly_url else ""
        )

        card = (
            f'<div class="{card_cls}">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;">'
            # Left side
            f'<div style="flex:1;min-width:0;">'
            f'<div style="display:flex;align-items:center;gap:0.45rem;margin-bottom:0.35rem;">'
            f'{side_badge}'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.58rem;'
            f'color:var(--muted);">{pos.token_id[:10]}…</span>'
            f'</div>'
            f'<div class="holding-name">{title}</div>'
            f'<div class="holding-meta">'
            f'<span>Size <b>€{pos.size_eur:.2f}</b></span>'
            f'<span>Conf <b>{pos.confidence}%</b></span>'
            f'<span>Age <b>{age_h:.1f}h</b></span>'
            f'<span>SL <b style="color:#E11D48">€{sl_eur:+.2f}</b></span>'
            f'<span>TP <b style="color:#059669">€{tp_eur:+.2f}</b></span>'
            f'</div>'
            f'</div>'
            # Right side — P&L
            f'<div style="flex-shrink:0;text-align:right;">'
            f'<div class="holding-pnl {pc}">€{est_pnl_eur:+.2f}</div>'
            f'<div class="holding-pct {pc}">{est_pnl_pct:+.2%}</div>'
            f'</div>'
            f'</div>'
            f'{track}'
            f'<div style="margin-top:0.5rem;display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.60rem;color:var(--muted);">'
            f'entry {pos.entry_price:.4f}'
            f'</span>'
            f'{link_html}'
            f'</div>'
            f'</div>'
        )
        st.markdown(card, unsafe_allow_html=True)

    # ── Summary footer ────────────────────────────────────────────────────────
    st.divider()
    total_invested = sum(p.size_eur for p in open_pos)
    bal_hist    = db.get_balance_history()
    current_bal = float(bal_hist[-1]["balance_eur"]) if bal_hist else 0.0
    free_bal    = max(0.0, current_bal - total_invested)
    pc          = pnl_cls(total_unrealized)

    st.markdown(
        '<div class="stat-row">'
        + _stat_card("Invested", f"€{total_invested:.2f}", f"{len(open_pos)} positions")
        + _stat_card(
            "Unrealized P&L",
            f"€{total_unrealized:+.2f}",
            "live prices" if prices_live else "entry price",
            pc if prices_live else "neutral",
          )
        + _stat_card("Free Cash", f"€{free_bal:.2f}", "available")
        + _stat_card("Positions", f"{len(open_pos)} / {config.risk.max_simultaneous_positions}", "used")
        + '</div>',
        unsafe_allow_html=True,
    )


# ── History Tab ───────────────────────────────────────────────────────────────

def render_history(db: Database) -> None:
    all_trades = db.get_all_trades()
    closed     = [t for t in all_trades if t.status.value == "CLOSED"]

    st.markdown(
        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:1.05rem;'
        f'font-weight:700;color:var(--text);margin-bottom:1rem;">'
        f'Trade History '
        f'<span style="font-size:0.78rem;font-weight:400;color:var(--muted);">'
        f'({len(closed)} closed trades)</span></div>',
        unsafe_allow_html=True,
    )

    if not closed:
        st.info("No closed trades yet.")
        return

    c1, c2 = st.columns(2)
    filter_side   = c1.selectbox("Side",   ["All", "BUY_YES", "BUY_NO"])
    filter_result = c2.selectbox("Result", ["All", "Winners",  "Losers"])

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


# ── Panel Vote helpers (Team 4) ───────────────────────────────────────────────

_PANEL_REC_COLORS = {
    "BUY_YES":           ("var(--blue)",  "#E8F0FE"),
    "BUY_NO":            ("var(--red)",   "#FDE8EC"),
    "WAIT":              ("#D97706",      "#FEF3C7"),
    "INSUFFICIENT_DATA": ("var(--muted)", "#F3F4F6"),
}


def _parse_panel_prefix(summary: str) -> list[tuple[str, str, str]] | None:
    """Parses the [PANEL: Q=BUY_YES/85, D=WAIT/30, A=WAIT/45] prefix.

    Returns a list of (agent_label, recommendation, confidence) tuples, or None
    if the prefix is not present or cannot be parsed.
    """
    import re

    if not summary or not summary.startswith("[PANEL:"):
        return None
    bracket_end = summary.find("]")
    if bracket_end < 0:
        return None
    prefix_content = summary[7:bracket_end].strip()
    agent_map = {"Q": "Quant", "D": "Domain Expert", "A": "Adversarial"}
    votes: list[tuple[str, str, str]] = []
    for token in prefix_content.split(","):
        token = token.strip()
        m = re.match(r"([QDA])=([A-Z_]+)/(\d+)", token)
        if m:
            key, rec, conf = m.group(1), m.group(2), m.group(3)
            votes.append((agent_map.get(key, key), rec, conf))
    return votes if votes else None


def _render_panel_vote(row: dict | None) -> None:
    """Renders the Panel Vote expandable for the most recent analysis."""
    if row is None:
        return
    summary = row.get("summary", "") or ""
    votes = _parse_panel_prefix(summary)
    if votes is None:
        return
    final_rec = row.get("recommendation", "WAIT")
    market_q = (row.get("market_question") or "")[:60]
    with st.expander("Panel Vote — most recent analysis", expanded=False):
        st.markdown(
            f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.82rem;'
            f'color:var(--muted);margin-bottom:0.75rem;">'
            f'{html_mod.escape(market_q)}</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(len(votes) + 1)
        for idx, (agent_label, rec, conf) in enumerate(votes):
            color, bg = _PANEL_REC_COLORS.get(rec, ("var(--muted)", "#F3F4F6"))
            cols[idx].markdown(
                f'<div style="background:{bg};border-radius:8px;padding:10px 12px;text-align:center;">'
                f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;font-weight:600;'
                f'color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">'
                f'{html_mod.escape(agent_label)}</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.82rem;'
                f'font-weight:600;color:{color};">{html_mod.escape(rec)}</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;'
                f'color:var(--muted);">{conf}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        final_color, final_bg = _PANEL_REC_COLORS.get(final_rec, ("var(--muted)", "#F3F4F6"))
        cols[-1].markdown(
            f'<div style="background:{final_bg};border:2px solid {final_color};border-radius:8px;'
            f'padding:10px 12px;text-align:center;">'
            f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;font-weight:600;'
            f'color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">'
            f'Final Call</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.82rem;'
            f'font-weight:700;color:{final_color};">{html_mod.escape(final_rec)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── LLM Analysis Tab ──────────────────────────────────────────────────────────

def render_analyses(db: Database) -> None:
    st.markdown(
        '<div style="font-family:\'DM Sans\',sans-serif;font-size:1.05rem;'
        'font-weight:700;color:var(--text);margin-bottom:1rem;">'
        'LLM Analyses '
        '<span style="font-size:0.78rem;font-weight:400;color:var(--muted);">(last 50)</span></div>',
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
        "Time":      r["timestamp"][:19],
        "Market":    (r["market_question"] or "")[:42],
        "YES price": f"{r['current_yes_price']:.3f}",
        "Prob":      f"{r['consensus_probability_yes']:.3f}",
        "Edge":      fmt_pct(r["edge"] or 0),
        "Conf":      r["confidence"],
        "Rec":       r["recommendation"],
        "Articles":  r["num_articles_analyzed"],
        "Tokens":    f"{(r['llm_input_tokens'] or 0)+(r['llm_output_tokens'] or 0):,}",
    } for r in rows]
    st.dataframe(pd.DataFrame(data), use_container_width=True, height=360)

    # Panel vote breakdown for most recent analysis
    _render_panel_vote(rows[0] if rows else None)

    from collections import Counter
    recs = [r["recommendation"] for r in rows if r.get("recommendation")]
    if recs:
        st.markdown(_sh("Recommendation Distribution"), unsafe_allow_html=True)
        counts = Counter(recs)
        colors_map = {
            "BUY_YES":           "#2563EB",
            "BUY_NO":            "#E11D48",
            "WAIT":              "#D97706",
            "INSUFFICIENT_DATA": "#C5CDD8",
        }
        fig = go.Figure(go.Bar(
            x=list(counts.keys()), y=list(counts.values()),
            marker_color=[colors_map.get(k, "#8A96A8") for k in counts.keys()],
            marker_line_width=0,
            marker_cornerradius=4,
            text=list(counts.values()), textposition="outside",
            textfont=dict(color="#8A96A8", size=11),
        ))
        _chart_style(fig, height=200)
        st.plotly_chart(fig, use_container_width=True)


# ── Learnings Tab ─────────────────────────────────────────────────────────────

_REPORT_FILE       = PROJECT_ROOT / "data" / "llm_report.md"
_OUTCOMES_FILE     = PROJECT_ROOT / "data" / "llm_outcomes.jsonl"
_SOCIAL_STATS_FILE = PROJECT_ROOT / "data" / "social_stats.json"
_FILTER_STATS_PATH = PROJECT_ROOT / "data" / "filter_stats.json"


def render_learnings(db: Database) -> None:
    st.markdown(
        '<div style="font-family:\'DM Sans\',sans-serif;font-size:1.05rem;'
        'font-weight:700;color:var(--text);margin-bottom:1rem;">'
        'Performance &amp; Learnings</div>',
        unsafe_allow_html=True,
    )

    # ── Performance metrics ───────────────────────────────────────────────────
    st.markdown(_sh("Latest Performance Snapshot"), unsafe_allow_html=True)
    try:
        snap = db.get_latest_performance_snapshot()
    except Exception:
        snap = None

    if snap:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Win Rate", f"{snap.win_rate:.1%}")
        c2.metric("Sharpe", f"{snap.sharpe_ratio:.2f}")
        c3.metric("Profit Factor", f"{snap.profit_factor:.2f}")
        c4.metric("Max Drawdown", f"{snap.max_drawdown:.1%}")
        c5.metric("Trades (90d)", snap.total_trades)
    else:
        st.info("No performance snapshot yet — nightly consolidation runs at 23:55.")

    # ── Performance history chart ─────────────────────────────────────────────
    try:
        history = db.get_performance_history(days=90)
    except Exception:
        history = []

    if len(history) >= 2:
        st.markdown(_sh("Win Rate History (90d)"), unsafe_allow_html=True)
        dates = [str(s.snapshot_date) for s in history]
        win_rates = [s.win_rate * 100 for s in history]
        fig = go.Figure(go.Scatter(
            x=dates, y=win_rates,
            mode="lines+markers",
            line=dict(color="#2563EB", width=2),
            marker=dict(size=6, color="#2563EB"),
            fill="tozeroy",
            fillcolor="rgba(37,99,235,0.08)",
        ))
        fig.add_hline(y=50, line_dash="dash", line_color="#CBD5E1",
                      annotation_text="50% breakeven", annotation_position="right")
        _chart_style(fig, height=180)
        fig.update_layout(yaxis_title="Win rate %", yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

    # ── Knowledge base ────────────────────────────────────────────────────────
    st.markdown(_sh("Knowledge Base", "(lessons from past trades)"), unsafe_allow_html=True)
    try:
        entries = db.get_knowledge_base(limit=100)
    except Exception:
        entries = []

    if entries:
        kb_data = [{
            "Pattern":    e.market_pattern,
            "Lesson":     e.lesson,
            "Category":   e.failure_category.value,
            "Confidence": f"{e.confidence:.0%}",
            "Confirmed":  e.times_confirmed,
            "Updated":    str(e.updated_at)[:10],
        } for e in entries]
        st.dataframe(pd.DataFrame(kb_data), use_container_width=True, height=300)

        # Category breakdown bar
        from collections import Counter
        cats = Counter(e.failure_category.value for e in entries)
        color_map = {
            "BAD_PREDICTION":  "#E11D48",
            "BAD_TIMING":      "#D97706",
            "BAD_EXECUTION":   "#7C3AED",
            "EXTERNAL_SHOCK":  "#0891B2",
            "NOT_A_LOSS":      "#16A34A",
        }
        fig2 = go.Figure(go.Bar(
            x=list(cats.keys()), y=list(cats.values()),
            marker_color=[color_map.get(k, "#8A96A8") for k in cats.keys()],
            marker_line_width=0, marker_cornerradius=4,
            text=list(cats.values()), textposition="outside",
            textfont=dict(color="#8A96A8", size=11),
        ))
        _chart_style(fig2, height=180)
        st.plotly_chart(fig2, use_container_width=True)

        # ── KB Health ─────────────────────────────────────────────────────────
        st.markdown(_sh("Knowledge Base Health"), unsafe_allow_html=True)
        h1, h2, h3, h4 = st.columns(4)
        avg_conf = sum(e.confidence for e in entries) / len(entries)
        at_risk  = sum(1 for e in entries if e.confidence < 0.2)
        high_conf = sum(1 for e in entries if e.confidence >= 0.6)
        h1.metric("Total Entries", len(entries))
        h2.metric("Avg Confidence", f"{avg_conf:.0%}")
        h3.metric("High Confidence (≥60%)", high_conf)
        h4.metric("At Risk (<20%)", at_risk, delta=f"-{at_risk}" if at_risk else None,
                  delta_color="inverse")

        # Topic category breakdown (the new compound.py `category` field)
        topic_cats = Counter(getattr(e, "category", "general") for e in entries)
        if len(topic_cats) > 1 or list(topic_cats.keys()) != ["general"]:
            topic_color_map = {
                "politics":      "#2563EB",
                "economics":     "#16A34A",
                "crypto":        "#D97706",
                "sports":        "#7C3AED",
                "legal":         "#0891B2",
                "science":       "#0D9488",
                "entertainment": "#E11D48",
                "general":       "#8A96A8",
            }
            fig3 = go.Figure(go.Bar(
                x=list(topic_cats.keys()), y=list(topic_cats.values()),
                marker_color=[topic_color_map.get(k, "#8A96A8") for k in topic_cats.keys()],
                marker_line_width=0, marker_cornerradius=4,
                text=list(topic_cats.values()), textposition="outside",
                textfont=dict(color="#8A96A8", size=11),
            ))
            _chart_style(fig3, height=160)
            fig3.update_layout(xaxis_title="Topic domain", yaxis_title="Entries")
            st.plotly_chart(fig3, use_container_width=True)

        # Confidence distribution histogram
        confs = [e.confidence for e in entries]
        fig4 = go.Figure(go.Histogram(
            x=confs, nbinsx=10,
            marker_color="#2563EB", marker_line_width=0,
            opacity=0.8,
        ))
        fig4.add_vline(x=0.1, line_dash="dash", line_color="#E11D48",
                       annotation_text="cull threshold", annotation_position="top right",
                       annotation_font_color="#E11D48")
        _chart_style(fig4, height=150)
        fig4.update_layout(xaxis_title="Confidence", yaxis_title="Count",
                           xaxis=dict(tickformat=".0%", range=[0, 1]))
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No lessons yet — lessons are generated automatically after each trade closes.")

    # ── Recent post-mortems ───────────────────────────────────────────────────
    st.markdown(_sh("Recent Post-Mortems"), unsafe_allow_html=True)
    if _OUTCOMES_FILE.exists():
        try:
            lines = _OUTCOMES_FILE.read_text(encoding="utf-8").splitlines()[-20:]
            if lines:
                records = []
                for raw in reversed(lines):
                    try:
                        r = json.loads(raw)
                        records.append({
                            "Date":     r.get("ts", "")[:10],
                            "Market":   r.get("market_slug", "")[:35],
                            "Side":     r.get("side", ""),
                            "P&L":      fmt_pct(float(r.get("pnl_pct", 0))),
                            "Category": r.get("failure_category", ""),
                            "Lesson":   r.get("lesson", ""),
                        })
                    except (json.JSONDecodeError, ValueError):
                        pass
                if records:
                    st.dataframe(pd.DataFrame(records), use_container_width=True, height=250)
            else:
                st.info("No outcomes recorded yet.")
        except OSError:
            st.info("Outcomes file not readable.")
    else:
        st.info("No outcomes recorded yet — outcomes are appended after each trade closes.")

    # ── Report file link ──────────────────────────────────────────────────────
    st.markdown(_sh("Human-Readable Report"), unsafe_allow_html=True)
    if _REPORT_FILE.exists():
        report_text = _REPORT_FILE.read_text(encoding="utf-8")
        with st.expander("View llm_report.md", expanded=False):
            st.markdown(report_text)
        st.caption(f"File: `{_REPORT_FILE}` — updated nightly and after each trade closes.")
    else:
        st.info("`data/llm_report.md` will appear here after the first trade closes or nightly consolidation runs.")


# ── Balance Manager Tab ───────────────────────────────────────────────────────

def render_balance_manager(db: Database, config) -> None:
    st.markdown(
        '<div style="font-family:\'DM Sans\',sans-serif;font-size:1.05rem;'
        'font-weight:700;color:var(--text);margin-bottom:1rem;">Virtual Balance</div>',
        unsafe_allow_html=True,
    )

    history             = db.get_balance_history()
    current             = float(history[-1]["balance_eur"]) if history else config.paper_trading.initial_balance_eur
    peak                = float(history[-1]["peak_balance"]) if history else current
    open_pos            = db.get_open_positions()
    consolidated_profit = db.get_consolidated_profit()

    st.markdown(
        '<div class="stat-row">'
        + _stat_card("Current Balance",  f"€{current:.2f}")
        + _stat_card("All-time Peak",    f"€{peak:.2f}")
        + _stat_card("Initial (config)", f"€{config.paper_trading.initial_balance_eur:.2f}")
        + _stat_card("Positions", f"{len(open_pos)}", "open")
        + '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="stat-row" style="grid-template-columns: repeat(2, 1fr);">'
        + _stat_card(
            "Consolidated Profit",
            f"€{consolidated_profit:.2f}",
            "swept from trading balance",
            "pos" if consolidated_profit > 0 else "neutral",
        )
        + _stat_card(
            "Total Value",
            f"€{current + consolidated_profit:.2f}",
            "balance + profit",
            "pos" if consolidated_profit > 0 else "neutral",
        )
        + '</div>',
        unsafe_allow_html=True,
    )

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
            _log(target, "MANUAL_RESET"); st.success(f"Reset to €{target:.2f}"); st.rerun()

    elif action == "Reset to custom":
        new_val = st.number_input("New balance (€)", min_value=1.0, max_value=50000.0,
                                  value=current, step=10.0)
        if st.button("Confirm custom reset", type="primary"):
            _log(new_val, "MANUAL_RESET"); st.success(f"Reset to €{new_val:.2f}"); st.rerun()

    elif action == "Add funds":
        amount = st.number_input("Amount to add (€)", min_value=1.0, max_value=50000.0,
                                 value=50.0, step=10.0)
        st.info(f"Resulting balance: €{current + amount:.2f}")
        if st.button("Confirm deposit", type="primary"):
            _log(current + amount, "MANUAL_ADD"); st.success(f"Added €{amount:.2f}"); st.rerun()

    else:
        max_ret = max(1.0, current - 1.0)
        amount  = st.number_input("Amount to withdraw (€)", min_value=1.0,
                                  max_value=max_ret, value=min(10.0, max_ret), step=5.0)
        st.info(f"Resulting balance: €{current - amount:.2f}")
        if st.button("Confirm withdrawal", type="primary"):
            _log(current - amount, "MANUAL_SUBTRACT"); st.success(f"Withdrew €{amount:.2f}"); st.rerun()

    st.divider()
    st.markdown(_sh("Manual Adjustments"), unsafe_allow_html=True)
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
        '<div style="font-family:\'DM Sans\',sans-serif;font-size:1.05rem;'
        'font-weight:700;color:var(--text);margin-bottom:0.3rem;">Backtesting</div>',
        unsafe_allow_html=True,
    )
    st.caption(
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
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;'
        f'color:var(--muted);padding-top:0.6rem;">'
        f'{config.llm.provider} / {config.llm.model.split(":")[0]}</div>',
        unsafe_allow_html=True,
    )

    if st.button("Run Backtest", type="primary"):
        with st.spinner(f"Analysing {n_markets} markets…"):
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
    st.markdown(_sh("Results"), unsafe_allow_html=True)

    retorno = (result.final_balance - result.initial_balance) / result.initial_balance
    st.markdown(
        '<div class="stat-row">'
        + _stat_card("Markets",    str(result.markets_analyzed))
        + _stat_card("Trades",     str(result.trades_executed))
        + _stat_card("Win Rate",   f"{result.win_rate:.1%}")
        + _stat_card("Total P&L",  fmt_eur(result.total_pnl_eur), "",
                     pnl_cls(result.total_pnl_eur))
        + '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="stat-row">'
        + _stat_card("Final Balance", f"€{result.final_balance:.2f}")
        + _stat_card("Return",        fmt_pct(retorno), "", pnl_cls(retorno))
        + _stat_card("Sharpe",        f"{result.sharpe_ratio:.2f}")
        + _stat_card("Max DD",        f"{result.max_drawdown_pct:.2%}")
        + '</div>',
        unsafe_allow_html=True,
    )

    executed = [t for t in result.trades
                if str(t.decision) in ("OPEN_TRADE", "DecisionAction.OPEN_TRADE")]
    if not executed:
        return

    st.markdown(_sh(f"Executed Trades", f"{len(executed)} trades"), unsafe_allow_html=True)
    rows = [{
        "Market":   t.market_question[:48],
        "YES won":  "Yes" if t.resolved_yes else "No",
        "Side":     t.side.value if hasattr(t.side, "value") else str(t.side),
        "P&L €":    fmt_eur(t.pnl_eur),
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

    st.markdown(_sh("Simulated Balance Curve"), unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(curve)+1)), y=curve,
        mode="lines+markers",
        line=dict(color="#2563EB", width=2, shape="spline", smoothing=0.3),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.06)",
        marker=dict(
            color=["#059669" if v >= result.initial_balance else "#E11D48" for v in curve],
            size=5,
        ),
    ))
    fig.add_hline(y=result.initial_balance, line_dash="dash", line_color="#CBD5E1")
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
    config  = get_config()
    db      = get_db(config)
    history = db.get_balance_history()
    open_pos = db.get_open_positions()

    refresh, _ = render_sidebar(config, history, open_pos)

    # ── Page header ──────────────────────────────────────────────────────────
    now_str = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div class="page-header">
      <div>
        <div class="page-title">Portfolio Dashboard</div>
        <div class="page-sub">Polymarket paper trading · {config.app.name}</div>
      </div>
      <div class="page-time">
        <span class="dot dot-live"></span>{now_str}
        <div style="font-size:0.60rem;color:var(--muted);margin-top:2px;">↻ {refresh}s</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not history:
        st.info("No data yet — start the bot: `python main.py`")

    tabs = st.tabs([
        "Overview",
        "Positions",
        "History",
        "Analyses",
        "Balance",
        "Backtest",
        "Learnings",
        "Social Signals",
        "Scanner",
    ])

    with tabs[0]: render_overview(db, config)
    with tabs[1]: render_positions(db, config)
    with tabs[2]: render_history(db)
    with tabs[3]: render_analyses(db)
    with tabs[4]: render_balance_manager(db, config)
    with tabs[5]: render_backtest(config)
    with tabs[6]: render_learnings(db)
    with tabs[7]: render_social_signals(config)
    with tabs[8]: render_scanner_stats()

    db.close()


# ── Social Signals Tab (Team 3) ───────────────────────────────────────────────

def render_social_signals(config) -> None:
    """Social Signals tab — shows counts from the last SocialIngestor cycle."""
    st.markdown(
        '<div style="font-family:\'DM Sans\',sans-serif;font-size:1.05rem;'
        'font-weight:700;color:var(--text);margin-bottom:1rem;">'
        'Social Signals</div>',
        unsafe_allow_html=True,
    )
    social_cfg = getattr(config, "social", None)
    if social_cfg is None or not social_cfg.enabled:
        st.info("Social ingestor is disabled. Set `social.enabled: true` in `config/settings.yaml`.")
        return
    stats: dict = {}
    if _SOCIAL_STATS_FILE.exists():
        try:
            stats = json.loads(_SOCIAL_STATS_FILE.read_text(encoding="utf-8"))
        except Exception:
            stats = {}
    telegram_count = stats.get("telegram", 0)
    reddit_count   = stats.get("reddit", 0)
    rss_count      = stats.get("rss", 0)
    total_count    = telegram_count + reddit_count + rss_count
    last_updated   = stats.get("updated_at", "—")
    if last_updated and last_updated != "—":
        try:
            last_updated = last_updated[:19].replace("T", " ")
        except Exception:
            pass
    st.markdown(_sh("Source Status", f"last cycle: {last_updated}"), unsafe_allow_html=True)
    sources = [
        ("Telegram", getattr(social_cfg.telegram, "enabled", False),
         f"{len(getattr(social_cfg.telegram, 'channels', []))} channels"),
        ("Reddit",   getattr(social_cfg.reddit, "enabled", False),
         f"{len(getattr(social_cfg.reddit, 'subreddits', []))} subreddits"),
        ("RSS",      getattr(social_cfg.rss, "enabled", False),
         f"{len(getattr(social_cfg.rss, 'feeds', []))} feeds"),
    ]
    cols = st.columns(3)
    for col, (name, enabled, detail) in zip(cols, sources):
        color = "#059669" if enabled else "#5A6580"
        with col:
            st.markdown(
                f'<div class="stat-card">'
                f'<span class="stat-label">{name} {"✓" if enabled else "—"}</span>'
                f'<div class="stat-sub">{detail}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(_sh("Articles Fetched — Last Cycle"), unsafe_allow_html=True)
    st.markdown(
        '<div class="stat-row">'
        + _stat_card("Telegram", str(telegram_count), "messages")
        + _stat_card("Reddit",   str(reddit_count),   "posts")
        + _stat_card("RSS",      str(rss_count),       "items")
        + _stat_card("Total",    str(total_count),     "social articles")
        + '</div>',
        unsafe_allow_html=True,
    )
    if not stats:
        st.info("No social stats yet — bot writes `data/social_stats.json` after each cycle.")


# ── Scanner Stats Tab (Team 6) ────────────────────────────────────────────────

def render_scanner_stats() -> None:
    """Market Scanner tab — shows pre-analysis filter stats from filter_stats.json."""
    st.markdown(
        '<div style="font-family:\'DM Sans\',sans-serif;font-size:1.05rem;'
        'font-weight:700;color:var(--text);margin-bottom:1rem;">Market Scanner</div>',
        unsafe_allow_html=True,
    )
    if not _FILTER_STATS_PATH.exists():
        st.info(
            "No filter stats yet — start the bot and wait for the first scan cycle. "
            "Stats are written to `data/filter_stats.json`."
        )
        return
    try:
        raw = json.loads(_FILTER_STATS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Could not read filter_stats.json: {exc}")
        return
    scanned  = raw.get("markets_scanned",  0)
    passed   = raw.get("markets_passed",   0)
    rejected = raw.get("markets_rejected", 0)
    ts       = (raw.get("timestamp", "") or "")[:19].replace("T", " ")
    pass_rate = (passed / scanned * 100) if scanned > 0 else 0.0
    st.markdown(_sh("Last Scan Cycle", ts), unsafe_allow_html=True)
    st.markdown(
        '<div class="stat-row">'
        + _stat_card("Scanned",  str(scanned),          "raw markets")
        + _stat_card("Passed",   str(passed),            f"{pass_rate:.0f}% pass rate")
        + _stat_card("Rejected", str(rejected),          "by pre-filters")
        + _stat_card("Pass Rate", f"{pass_rate:.0f}%",  f"{passed}/{scanned}")
        + '</div>',
        unsafe_allow_html=True,
    )
    rejections: dict = raw.get("rejections", {})
    st.markdown(_sh("Filter Breakdown"), unsafe_allow_html=True)
    if not rejections:
        st.success("All scanned markets passed the filters in the last cycle.")
    else:
        label_map = {
            "low_volume":        "Low Volume",
            "closing_soon":      "Closing Soon (<12h)",
            "wide_spread":       "Wide Spread",
            "excluded_category": "Excluded Category",
        }
        with st.expander("Rejection Counts", expanded=True):
            col1, col2 = st.columns(2)
            for k, count in rejections.items():
                label = label_map.get(k, k)
                col1.markdown(f"**{label}**")
                col2.markdown(str(count))


if __name__ == "__main__":
    main()
