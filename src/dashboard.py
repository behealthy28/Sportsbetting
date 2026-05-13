#!/usr/bin/env python3
"""
EdgeFinder — Live Bloomberg-style Terminal Dashboard

Layout:
  +------------------ HEADER (UTC clock) -------------------+
  | STATUS         |          MARKET SCANNER                |
  |  freshness •   |   Match · Sport · Mkt% · Model% · Edge |
  |  alerts        |   sorted by |edge| desc                |
  +----------------+----------------------------------------+
  | PERFORMANCE    |          TRADE LOG + P&L               |
  |  totals + sport|   last bets · running totals footer    |
  +------------------ FOOTER (ticker, LIVE/DRY) ------------+

Launch:
  python main.py dashboard          # one-shot
  python main.py                    # then type `dashboard` in REPL
"""
from __future__ import annotations

import os
import time
import sys
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# --- Color palette (Bloomberg terminal aesthetic) ---
ACCENT = "bright_green"
BRAND = "bright_blue"
DIM = "bright_black"
WARN = "yellow"
LOSS = "red"
WIN = "bright_green"
MUTED = "dim white"
AWAY = "bright_magenta"

# Sport short codes for column display
SPORT_TAG = {
    "football": "FB",
    "soccer":   "FB",
    "tennis":   "TEN",
    "ufc":      "UFC",
    "mma":      "MMA",
    "boxing":   "BOX",
    "cricket":  "CRI",
    "darts":    "DRT",
    "badminton": "BAD",
    "table tennis": "TT",
}

SPORT_LABELS = ["football", "tennis", "ufc", "boxing", "cricket", "darts", "badminton"]

# League substring → sport (used when fixtures don't expose a sport field)
LEAGUE_SPORT_HINTS = [
    ("ufc", "ufc"),
    ("mma", "ufc"),
    ("boxing", "boxing"),
    ("tennis", "tennis"),
    ("atp", "tennis"),
    ("wta", "tennis"),
    ("cricket", "cricket"),
    ("darts", "darts"),
    ("badminton", "badminton"),
]


def _infer_sport(fixture: dict) -> str:
    league = (fixture.get("league", "") or "").lower()
    for hint, sport in LEAGUE_SPORT_HINTS:
        if hint in league:
            return sport
    return "football"


class DashboardState:
    """Tracks live state across scan cycles."""

    def __init__(self):
        self.cycle = 0
        self.scanning = False
        self.scan_status = "Initializing..."
        self.fixtures: list = []
        self.rows: list = []          # rendered scanner rows (precomputed)
        self.predictions: dict = {}   # (e1, e2, date) -> {model_p, market_p, edge, side, sport}
        self.predict_cursor = 0       # rotating index across fixtures
        self.alerts: list = []        # last alerts from watchlist
        self.last_alert_cycle = -999
        self.last_error: Optional[str] = None
        # Per-source freshness: source -> (timestamp, status)
        self.sources: dict = {
            "Fixtures": (0.0, "idle"),
            "Polymarket": (0.0, "idle"),
            "Kalshi": (0.0, "idle"),
            "Pinnacle": (0.0, "idle"),
            "Watchlist": (0.0, "idle"),
        }

    def mark_source(self, name: str, status: str = "ok"):
        self.sources[name] = (time.time(), status)


state = DashboardState()


# ─── Data refresh ──────────────────────────────────────────────────────────────

def _fetch_fixtures(days_ahead: int):
    """Pull upcoming fixtures and update state.fixtures + scanner rows scaffold."""
    state.scan_status = "Fetching fixtures..."
    try:
        from src.data.scrapers.fixtures import get_todays_fixtures
        fixtures = get_todays_fixtures(days_ahead=days_ahead)
        state.fixtures = fixtures[:20]  # cap to keep predictions tractable
        state.mark_source("Fixtures", "ok" if fixtures else "empty")
    except Exception as e:
        state.last_error = f"fixtures: {e}"
        state.mark_source("Fixtures", "err")


def _predict_one(fixture: dict) -> Optional[dict]:
    """Predict + fetch odds for a single fixture. Returns row dict or None."""
    from src.predictor import SPORT_HANDLERS
    from src.data.market import get_market_odds
    from src.market.edge import calculate_edge, best_bet

    sport = _infer_sport(fixture)
    handler = SPORT_HANDLERS.get(sport)
    if handler is None:
        return None

    home = fixture.get("home") or ""
    away = fixture.get("away") or ""
    if not home or not away:
        return None

    date = (fixture.get("date") or "")[:10] or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Model prediction (call handler directly to skip terminal output)
    try:
        result = handler.predict(home, away, date, {
            "competition": fixture.get("league", ""),
            "is_neutral": False,
            "venue": fixture.get("venue", ""),
        })
    except Exception:
        return None

    probs = result.probabilities or {}
    market = result.market_probs
    if market:
        state.mark_source("Polymarket", "ok")
    edges = result.edges or {}
    if not edges and market:
        edges = calculate_edge(probs, market)

    # Pick the most relevant outcome to display
    side_key = None
    side_label = ""
    if edges:
        side_key, _ = best_bet(edges)
    if side_key is None:
        side_key = max(probs, key=lambda k: probs[k]) if probs else None
    if side_key is None:
        return None

    # Map outcome key → display side
    if side_key in ("home_win", "p1_win", "f1_win"):
        side_label = "HOME"
    elif side_key in ("away_win", "p2_win", "f2_win"):
        side_label = "AWAY"
    elif side_key == "draw":
        side_label = "DRAW"
    else:
        side_label = side_key.upper()[:5]

    model_p = probs.get(side_key)
    market_p = market.get(side_key) if market else None
    edge_info = edges.get(side_key) if edges else None
    edge_pct = edge_info.get("edge_pct") if edge_info else None

    return {
        "match": f"{home} v {away}",
        "sport": SPORT_TAG.get(sport, sport[:3].upper()),
        "date": date[5:],  # MM-DD
        "model_p": model_p,
        "market_p": market_p,
        "edge_pct": edge_pct,
        "side": side_label,
        "league": fixture.get("league", ""),
    }


def _refresh_predictions(batch_size: int = 4):
    """Rotate through fixtures and refresh model predictions for a small batch."""
    if not state.fixtures:
        return
    state.scan_status = f"Predicting batch of {batch_size}..."
    n = len(state.fixtures)
    for i in range(batch_size):
        if not state.fixtures:
            break
        idx = (state.predict_cursor + i) % n
        f = state.fixtures[idx]
        key = (f.get("home", ""), f.get("away", ""), (f.get("date") or "")[:10])
        row = _predict_one(f)
        if row is not None:
            state.predictions[key] = row
    state.predict_cursor = (state.predict_cursor + batch_size) % n


def _refresh_alerts():
    state.scan_status = "Checking watchlist..."
    try:
        from src.alerts import check_watchlist_movements
        new = check_watchlist_movements(threshold_pp=5.0) or []
        if new:
            state.alerts = (new + state.alerts)[:10]
        state.mark_source("Watchlist", "ok")
    except Exception as e:
        state.last_error = f"watchlist: {e}"
        state.mark_source("Watchlist", "err")


def run_scan_cycle(days_ahead: int, fixture_refresh_every: int = 5):
    """One full scan: optionally refresh fixtures + always refresh a batch of predictions."""
    state.cycle += 1
    state.scanning = True

    if state.cycle == 1 or state.cycle % fixture_refresh_every == 1:
        _fetch_fixtures(days_ahead)

    _refresh_predictions(batch_size=4)

    if state.cycle - state.last_alert_cycle >= 4:
        _refresh_alerts()
        state.last_alert_cycle = state.cycle

    state.scan_status = "Idle — waiting"
    state.scanning = False


# ─── Renderers ────────────────────────────────────────────────────────────────

def _mode_badge() -> str:
    try:
        from src.execution import is_configured
        live = is_configured()
    except Exception:
        live = False
    return f"[{WIN}]LIVE[/{WIN}]" if live else f"[{WARN}]DRY[/{WARN}]"


def make_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=2),
    )
    layout["left"].split_column(
        Layout(name="status", ratio=1),
        Layout(name="performance", ratio=1),
    )
    layout["right"].split_column(
        Layout(name="scanner", ratio=2),
        Layout(name="trades", ratio=3),
    )
    return layout


def render_header() -> Panel:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="center", ratio=2)
    grid.add_column(justify="right", ratio=1)
    grid.add_row(
        Text(" EDGEFINDER TERMINAL", style=f"bold {BRAND}"),
        Text("MODEL vs MARKET — LIVE EDGE SCANNER", style=DIM),
        Text(f"{now} ", style=MUTED),
    )
    return Panel(grid, style=BRAND, box=box.HEAVY)


def render_status() -> Panel:
    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    table.add_column("label", style=MUTED, width=14)
    table.add_column("value")

    if state.scanning:
        dot = f"[{WARN}]◌[/{WARN}]"
        status_text = f"{dot} SCANNING"
    elif state.cycle > 0:
        dot = f"[{ACCENT}]●[/{ACCENT}]"
        status_text = f"{dot} ACTIVE"
    else:
        dot = f"[{WARN}]○[/{WARN}]"
        status_text = f"{dot} STARTING"

    table.add_row("Pipeline", status_text)
    table.add_row("Cycle", f"#{state.cycle}" if state.cycle else "—")
    table.add_row("Activity", f"[{DIM}]{state.scan_status[:24]}[/{DIM}]")
    table.add_row("Mode", _mode_badge())
    table.add_row("", "")

    # Per-source freshness dots
    now = time.time()
    for src_name, (ts, status) in state.sources.items():
        if status == "idle" or ts == 0:
            dot = f"[{DIM}]○[/{DIM}]"
            age = "—"
        elif status == "err":
            dot = f"[{LOSS}]●[/{LOSS}]"
            age = f"{int(now - ts)}s err"
        else:
            age_s = int(now - ts)
            if age_s < 120:
                dot = f"[{ACCENT}]●[/{ACCENT}]"
            elif age_s < 600:
                dot = f"[{WARN}]●[/{WARN}]"
            else:
                dot = f"[{LOSS}]●[/{LOSS}]"
            age = f"{age_s}s ago" if age_s < 60 else f"{age_s // 60}m ago"
        table.add_row(f"{dot} {src_name}", f"[{DIM}]{age}[/{DIM}]")

    # Recent alerts
    table.add_row("", "")
    if state.alerts:
        table.add_row(f"[bold]ALERTS[/bold]", f"[{WARN}]{len(state.alerts)} recent[/{WARN}]")
        for a in state.alerts[:3]:
            line = f"{a.get('entity1','')[:8]} v {a.get('entity2','')[:8]} {a.get('outcome','')[:4]} {a.get('delta_pp',0):+.1f}pp"
            colour = ACCENT if a.get("delta_pp", 0) >= 0 else AWAY
            table.add_row(f"[{DIM}]·[/{DIM}]", f"[{colour}]{line}[/{colour}]")
    else:
        table.add_row(f"[bold]ALERTS[/bold]", f"[{DIM}]none[/{DIM}]")

    return Panel(table, title=f"[bold]PIPELINE STATUS[/bold]", border_style=BRAND, box=box.ROUNDED)


def render_performance() -> Panel:
    try:
        from src import trades as trade_log
        all_trades = trade_log.load()
        s = trade_log.stats(all_trades)
        by_sport = trade_log.by_sport(all_trades)
        brier = trade_log.brier_score(all_trades)
    except Exception as e:
        state.last_error = f"perf: {e}"
        return Panel(Text(f"  err: {e}", style=LOSS), title=f"[bold]PERFORMANCE[/bold]", border_style=BRAND, box=box.ROUNDED)

    inner = Table.grid(expand=True)
    inner.add_column()

    # Top totals
    totals = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    totals.add_column("k", style=MUTED, width=12)
    totals.add_column("v")
    pnl_style = ACCENT if s["total_pnl"] >= 0 else LOSS
    roi_style = ACCENT if s["roi"] >= 0 else LOSS
    wr_style = ACCENT if s["win_rate"] >= 55 else (WARN if s["win_rate"] >= 45 else LOSS)
    totals.add_row("Bets", f"[{ACCENT}]{s['total']}[/{ACCENT}]  ({s['pending']} pending)")
    totals.add_row("Settled", f"{s['settled']}  ·  {s['wins']} W")
    totals.add_row("Win Rate", f"[{wr_style}]{s['win_rate']:.1f}%[/{wr_style}]")
    totals.add_row("P&L", f"[{pnl_style}]{'+' if s['total_pnl'] >= 0 else ''}{s['total_pnl']:.2f}[/{pnl_style}]")
    totals.add_row("ROI", f"[{roi_style}]{'+' if s['roi'] >= 0 else ''}{s['roi']:.1f}%[/{roi_style}]")

    if brier.get("yours") is not None:
        if brier.get("market") is not None:
            beating = brier.get("beating")
            tag = f"[{ACCENT}]BEATING[/{ACCENT}]" if beating else f"[{LOSS}]TRAILING[/{LOSS}]"
            totals.add_row("Brier", f"{brier['yours']} / {brier['market']} {tag}")
        else:
            totals.add_row("Brier", f"{brier['yours']}")

    inner.add_row(totals)
    inner.add_row(Text(" BY SPORT", style=f"bold {DIM}"))

    # Per-sport breakdown
    sport_tbl = Table(show_header=True, box=box.SIMPLE_HEAD, expand=True, padding=(0, 1))
    sport_tbl.add_column("Sport", style=MUTED, width=8)
    sport_tbl.add_column("N", justify="right", width=3)
    sport_tbl.add_column("WR%", justify="right", width=5)
    sport_tbl.add_column("ROI%", justify="right", width=6)
    sport_tbl.add_column("P&L", justify="right", width=7)

    if not by_sport:
        sport_tbl.add_row(f"[{DIM}]no bets yet[/{DIM}]", "", "", "", "")
    else:
        for sp, st in sorted(by_sport.items(), key=lambda kv: -(kv[1]["count"])):
            wr = st["win_rate"]
            roi = st["roi"]
            pnl = st["pnl"]
            wr_s = "—" if wr is None else f"{wr:.0f}"
            roi_s = "—" if roi is None else f"{roi:+.0f}"
            pnl_s = f"{pnl:+.1f}" if pnl else "0.0"
            wr_style = ACCENT if (wr or 0) >= 55 else (WARN if (wr or 0) >= 45 else DIM)
            pnl_style = ACCENT if pnl > 0 else (LOSS if pnl < 0 else DIM)
            sport_tbl.add_row(
                sp[:8],
                str(st["count"]),
                f"[{wr_style}]{wr_s}[/{wr_style}]",
                f"[{wr_style}]{roi_s}[/{wr_style}]",
                f"[{pnl_style}]{pnl_s}[/{pnl_style}]",
            )

    inner.add_row(sport_tbl)
    return Panel(inner, title=f"[bold]PERFORMANCE[/bold]", border_style=BRAND, box=box.ROUNDED)


def render_scanner() -> Panel:
    table = Table(show_header=True, box=box.SIMPLE_HEAD, expand=True, padding=(0, 1))
    table.add_column("Match", max_width=30)
    table.add_column("Spt", width=3, style=MUTED)
    table.add_column("Date", width=5, style=MUTED)
    table.add_column("Side", justify="center", width=5)
    table.add_column("Mkt%", justify="right", width=5)
    table.add_column("Model%", justify="right", width=6)
    table.add_column("Edge", justify="right", width=6)

    rows = list(state.predictions.values())
    # Sort by |edge| desc; rows with no edge sink to the bottom
    rows.sort(key=lambda r: abs(r.get("edge_pct") or 0), reverse=True)

    if not rows:
        if state.fixtures:
            table.add_row(f"[{DIM}]Computing edges… ({len(state.fixtures)} fixtures queued)[/{DIM}]", "", "", "", "", "", "")
        else:
            table.add_row(f"[{DIM}]Waiting for first fixtures fetch…[/{DIM}]", "", "", "", "", "", "")
        return Panel(table, title=f"[bold]MARKET SCANNER[/bold] · Model vs Market", border_style=BRAND, box=box.ROUNDED)

    for r in rows[:12]:
        edge = r.get("edge_pct")
        if edge is None:
            edge_str = f"[{DIM}]—[/{DIM}]"
        elif edge >= 5:
            edge_str = f"[{ACCENT}]+{edge:.1f}%[/{ACCENT}]"
        elif edge >= 2:
            edge_str = f"[{WARN}]+{edge:.1f}%[/{WARN}]"
        elif edge >= 0:
            edge_str = f"[{DIM}]+{edge:.1f}%[/{DIM}]"
        else:
            edge_str = f"[{LOSS}]{edge:.1f}%[/{LOSS}]"

        side = r.get("side", "")
        side_style = AWAY if side == "AWAY" else (ACCENT if side == "HOME" else MUTED)

        mp = r.get("market_p")
        modp = r.get("model_p")
        mkt_str = f"{mp * 100:.0f}" if mp is not None else f"[{DIM}]—[/{DIM}]"
        mod_str = f"{modp * 100:.0f}" if modp is not None else f"[{DIM}]—[/{DIM}]"

        table.add_row(
            r["match"][:30],
            r["sport"],
            r["date"],
            f"[{side_style}]{side}[/{side_style}]",
            mkt_str,
            f"[{ACCENT}]{mod_str}[/{ACCENT}]",
            edge_str,
        )

    return Panel(table, title=f"[bold]MARKET SCANNER[/bold] · Model vs Market", border_style=BRAND, box=box.ROUNDED)


def render_trades() -> Panel:
    try:
        from src import trades as trade_log
        all_trades = trade_log.load()
        s = trade_log.stats(all_trades)
    except Exception as e:
        return Panel(Text(f"  err: {e}", style=LOSS), title=f"[bold]TRADE LOG[/bold]", border_style=BRAND, box=box.ROUNDED)

    table = Table(show_header=True, box=box.SIMPLE_HEAD, expand=True, padding=(0, 1))
    table.add_column("Time", width=10, style=MUTED)
    table.add_column("Match", max_width=22)
    table.add_column("Spt", width=3, style=MUTED)
    table.add_column("Bet", max_width=12)
    table.add_column("Stake", justify="right", width=5)
    table.add_column("Edge%", justify="right", width=6)
    table.add_column("Mode", justify="center", width=4)
    table.add_column("Status", justify="center", width=7)
    table.add_column("P&L", justify="right", width=7)

    if not all_trades:
        table.add_row(f"[{DIM}]No trades yet — log via 'predict' → confirm[/{DIM}]", "", "", "", "", "", "", "", "")
    else:
        # newest first
        for t in list(reversed(all_trades))[:10]:
            sport = (t.get("sport", "") or "").lower()
            sport_tag = SPORT_TAG.get(sport, sport[:3].upper() if sport else "—")
            ts = (t.get("timestamp", "") or "")[5:16].replace("T", " ")
            match = f"{t.get('entity1','')} v {t.get('entity2','')}"
            stake = t.get("stake") or 0
            edge_pct = t.get("edge_pct")
            edge_str = "—" if edge_pct is None else f"{edge_pct:+.1f}"
            status = t.get("status", "pending")
            is_dry = t.get("is_dry_run", False)
            mode_str = f"[{WARN}]SIM[/{WARN}]" if is_dry else f"[{DIM}]LIVE[/{DIM}]"
            if status == "won":
                status_str = f"[{ACCENT}]✓ WON[/{ACCENT}]"
            elif status == "lost":
                status_str = f"[{LOSS}]✗ LOST[/{LOSS}]"
            else:
                status_str = f"[{WARN}]PEND[/{WARN}]"
            pnl = t.get("pnl")
            if pnl is None:
                pnl_str = f"[{DIM}]—[/{DIM}]"
            elif pnl >= 0:
                pnl_str = f"[{ACCENT}]+{pnl:.1f}[/{ACCENT}]" if not is_dry else f"[{WARN}]+{pnl:.1f}[/{WARN}]"
            else:
                pnl_str = f"[{LOSS}]{pnl:.1f}[/{LOSS}]"

            table.add_row(
                ts,
                match[:22],
                sport_tag,
                (t.get("bet_label", "") or "")[:12],
                f"{stake:.0f}",
                edge_str,
                mode_str,
                status_str,
                pnl_str,
            )

    dry_count = sum(1 for t in all_trades if t.get("is_dry_run"))
    live_count = s["total"] - dry_count
    title = (
        f"[bold]TRADE LOG + P&L[/bold] · "
        f"{live_count} real · [{WARN}]{dry_count} sim[/{WARN}] · "
        f"{s['win_rate']:.0f}% WR · ROI {s['roi']:+.1f}% · P&L {s['total_pnl']:+.1f}"
    )
    return Panel(table, title=title, border_style=BRAND, box=box.ROUNDED)


def render_footer() -> Panel:
    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=3)
    grid.add_column(justify="center", ratio=2)
    grid.add_column(justify="right", ratio=2)

    if state.alerts:
        a = state.alerts[0]
        delta = a.get("delta_pp", 0)
        colour = ACCENT if delta >= 0 else AWAY
        ticker = f"[{colour}]>[/{colour}] [{MUTED}]ALERT[/{MUTED}] {a.get('entity1','')} v {a.get('entity2','')} {a.get('outcome','')} {delta:+.1f}pp"
    elif state.last_error:
        ticker = f"[{LOSS}]>[/{LOSS}] [{MUTED}]{state.last_error[:80]}[/{MUTED}]"
    else:
        ticker = f"[{DIM}]> waiting for alerts…[/{DIM}]"

    grid.add_row(
        ticker,
        f"[{DIM}]Ctrl+C to exit[/{DIM}]",
        f"{_mode_badge()}  [{DIM}]·[/{DIM}]  cycle [{ACCENT}]{state.cycle}[/{ACCENT}]  ",
    )
    return Panel(grid, style=BRAND, box=box.HEAVY)


# ─── Main loop ────────────────────────────────────────────────────────────────

def _render_all(layout: Layout):
    layout["header"].update(render_header())
    layout["status"].update(render_status())
    layout["performance"].update(render_performance())
    layout["scanner"].update(render_scanner())
    layout["trades"].update(render_trades())
    layout["footer"].update(render_footer())


def run_dashboard(scan_interval: float = 60.0, days_ahead: int = 2):
    """Launch the live dashboard. Refreshes a batch of predictions every scan_interval seconds."""
    layout = make_layout()
    _render_all(layout)

    try:
        with Live(layout, console=console, refresh_per_second=2, screen=True):
            last_scan = 0.0
            while True:
                now = time.time()
                if now - last_scan >= scan_interval:
                    try:
                        run_scan_cycle(days_ahead=days_ahead)
                    except Exception as e:
                        state.last_error = str(e)
                        state.scanning = False
                    last_scan = now
                _render_all(layout)
                time.sleep(0.5)
    except KeyboardInterrupt:
        console.print(
            f"\n[{ACCENT}]Dashboard stopped. {state.cycle} cycles · "
            f"{len(state.predictions)} fixtures tracked.[/{ACCENT}]\n"
        )


if __name__ == "__main__":
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    run_dashboard(scan_interval=interval, days_ahead=days)
