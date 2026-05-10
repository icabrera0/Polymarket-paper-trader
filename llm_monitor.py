"""
LLM Analysis Monitor — Rich TUI for observing qwen panel decisions in real time.

Run in a separate terminal while the bot is running:
    python llm_monitor.py

Reads logs/llm_trace.jsonl (written by the bot) every 2 seconds.
Press Ctrl+C or q to exit.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

TRACE_FILE = Path("logs/llm_trace.jsonl")
POLL_INTERVAL = 2.0   # seconds between file reads
MAX_POST_MORTEMS = 5  # rows to show in the stream table


def _read_new_lines(path: Path, offset: int) -> tuple[list[str], int]:
    """Returns new lines since `offset` and the new file offset."""
    if not path.exists():
        return [], offset
    try:
        with path.open("r", encoding="utf-8") as f:
            f.seek(offset)
            new_content = f.read()
            new_offset = f.tell()
        lines = [l for l in new_content.splitlines() if l.strip()]
        return lines, new_offset
    except OSError:
        return [], offset


def _parse_events(lines: list[str]) -> list[dict]:
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return events


class MonitorState:
    """Holds the most recent state for each panel section.

    Uses per-market buffers so parallel analysis (llm_parallelism > 1) doesn't
    overwrite agent results mid-flight.  Display switches to a market only when
    its SYNTHESIS_RESULT arrives; until then the last completed market is shown.
    """

    def __init__(self) -> None:
        self.market_question: str = "—"
        self.yes_price: float = 0.0
        self.no_price: float = 0.0
        self.num_articles: int = 0
        self.kb_lessons: list[str] = []
        self.agents: dict[str, dict] = {
            "Quant": {},
            "Domain": {},
            "Adversarial": {},
        }
        self.synthesis: dict = {}
        self.post_mortems: list[dict] = []
        self.last_updated: str = "—"

        # Per-market buffers — keyed by market_id
        self._buffers: dict[str, dict] = {}
        # market_id currently shown in the UI
        self._display_id: str = ""

    def apply(self, event: dict) -> None:
        kind = event.get("event", "")
        ts = event.get("ts", "")[:19].replace("T", " ")
        mid = event.get("market_id", "")

        if kind == "PANEL_START":
            self._buffers[mid] = {
                "question": event.get("market_question", "—"),
                "yes_price": float(event.get("yes_price", 0)),
                "no_price": float(event.get("no_price", 0)),
                "num_articles": int(event.get("num_articles", 0)),
                "kb_lessons": event.get("kb_lessons", []),
                "agents": {"Quant": {}, "Domain": {}, "Adversarial": {}},
                "synthesis": {},
                "ts": ts,
                "done": False,
            }
            # First market ever — show it while waiting for agents
            if not self._display_id:
                self._display_id = mid
                self._sync_display()

        elif kind == "AGENT_RESULT":
            if mid in self._buffers:
                name = event.get("agent", "Unknown")
                if name in self._buffers[mid]["agents"]:
                    self._buffers[mid]["agents"][name] = event
                self.last_updated = ts
                if mid == self._display_id:
                    self._sync_display()

        elif kind == "SYNTHESIS_RESULT":
            if mid in self._buffers:
                self._buffers[mid]["synthesis"] = event
                self._buffers[mid]["done"] = True
                self._display_id = mid
                self.last_updated = ts
                self._sync_display()
                # Prune old completed buffers
                for k in [k for k, v in self._buffers.items() if v["done"] and k != mid]:
                    del self._buffers[k]

        elif kind == "POST_MORTEM":
            self.post_mortems.insert(0, event)
            self.post_mortems = self.post_mortems[:MAX_POST_MORTEMS]

    def _sync_display(self) -> None:
        buf = self._buffers.get(self._display_id)
        if not buf:
            return
        self.market_question = buf["question"]
        self.yes_price = buf["yes_price"]
        self.no_price = buf["no_price"]
        self.num_articles = buf["num_articles"]
        self.kb_lessons = buf["kb_lessons"]
        self.agents = buf["agents"]
        self.synthesis = buf["synthesis"]
        if buf["ts"]:
            self.last_updated = buf["ts"]


def _agent_panel(name: str, data: dict) -> Panel:
    if not data:
        content = Text("waiting...", style="dim")
    else:
        rec = data.get("recommendation", "?")
        conf = data.get("confidence", 0)
        prob = data.get("probability", 0.0)
        edge = data.get("edge", 0.0)
        tok_in = data.get("input_tokens", 0)
        tok_out = data.get("output_tokens", 0)
        just = data.get("justification_excerpt", "")[:80]

        rec_style = "green" if "BUY" in rec else "yellow" if rec == "WAIT" else "dim"
        content = (
            Text(f"Rec:  ", style="bold") + Text(rec, style=rec_style) + Text("\n") +
            Text(f"Conf: {conf}\n") +
            Text(f"P(YES): {prob:.4f}\n") +
            Text(f"Edge:   {edge:+.4f}\n") +
            Text(f"Tokens: {tok_in:,} in / {tok_out:,} out\n") +
            Text(f"\n{just}", style="dim")
        )
    colors = {"Quant": "cyan", "Domain": "green", "Adversarial": "red"}
    color = colors.get(name, "white")
    return Panel(content, title=f"[bold {color}]{name}[/]", border_style=color, padding=(0, 1))


def _build_layout(state: MonitorState) -> Table:
    root = Table.grid(padding=0)
    root.add_column()

    # Header
    header = Panel(
        Text(
            f"Market: {state.market_question[:90]}\n"
            f"YES={state.yes_price:.3f}  NO={state.no_price:.3f}  "
            f"| {state.num_articles} articles  "
            f"| {len(state.kb_lessons)} KB lessons injected",
            style="bold white",
        ),
        title="[bold blue]LLM Analysis Monitor[/bold blue]",
        subtitle=f"[dim]updated {state.last_updated} UTC  |  ctrl+c to quit[/]",
        border_style="blue",
    )
    root.add_row(header)

    # Agent panels side by side
    agent_panels = Columns(
        [_agent_panel(name, state.agents[name]) for name in ("Quant", "Domain", "Adversarial")],
        equal=True,
        expand=True,
    )
    root.add_row(agent_panels)

    # Synthesis
    if state.synthesis:
        rec = state.synthesis.get("final_recommendation", "—")
        conf = state.synthesis.get("final_confidence", 0)
        prob = state.synthesis.get("final_probability", 0.0)
        rules = state.synthesis.get("rules_triggered", "")[:120]
        toks = state.synthesis.get("total_tokens", 0)
        rec_style = "bold green" if "BUY" in rec else "bold yellow"
        synth_content = (
            Text(f"Final: ", style="bold") + Text(rec, style=rec_style) +
            Text(f"  |  conf={conf}  |  p={prob:.4f}  |  {toks:,} total tokens\n") +
            Text(rules, style="dim")
        )
    else:
        synth_content = Text("waiting for synthesis...", style="dim")
    root.add_row(Panel(synth_content, title="[bold]Synthesis[/]", border_style="magenta", padding=(0, 1)))

    # KB Injections
    if state.kb_lessons:
        kb_lines = "\n".join(f"  {l[:100]}" for l in state.kb_lessons[:5])
        kb_content = Text(kb_lines, style="italic dim")
    else:
        kb_content = Text("none injected", style="dim")
    root.add_row(Panel(kb_content, title="[bold]KB Injections[/]", border_style="yellow", padding=(0, 1)))

    # Post-mortem stream
    pm_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    pm_table.add_column("Trade", style="dim", max_width=10)
    pm_table.add_column("Side", max_width=8)
    pm_table.add_column("P&L", max_width=8)
    pm_table.add_column("Category", max_width=16)
    pm_table.add_column("Lesson", max_width=60)
    for pm in state.post_mortems:
        pnl = float(pm.get("pnl_pct", 0))
        pnl_style = "green" if pnl >= 0 else "red"
        pm_table.add_row(
            pm.get("trade_id", "")[:8],
            pm.get("side", ""),
            Text(f"{pnl:+.1%}", style=pnl_style),
            pm.get("failure_category", ""),
            pm.get("lesson", "")[:60],
        )
    root.add_row(Panel(pm_table, title="[bold]Post-Mortem Stream[/]", border_style="cyan", padding=(0, 0)))

    return root


def main() -> None:
    console = Console()
    state = MonitorState()
    offset = 0

    with Live(console=console, refresh_per_second=1, screen=False) as live:
        while True:
            new_lines, offset = _read_new_lines(TRACE_FILE, offset)
            if new_lines:
                for event in _parse_events(new_lines):
                    state.apply(event)
            elif not TRACE_FILE.exists():
                state.market_question = "Waiting for bot to start (logs/llm_trace.jsonl not found)..."

            live.update(_build_layout(state))
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
