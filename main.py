#!/usr/bin/env python3
"""
EdgeFinder — Sports Betting Predictor (interactive terminal)

Run with no arguments to enter the interactive REPL:
  python main.py

Or pass a one-shot query directly:
  python main.py "Portugal vs Spain Nations League"
  python main.py "Djokovic vs Alcaraz Wimbledon"
  python main.py trades
  python main.py show sports
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.padding import Padding

console = Console()


def _banner():
    console.print()
    console.print(Panel(
        Text.assemble(
            ("  EdgeFinder", "bold bright_blue"),
            ("  ·  Sports Betting Predictor\n\n", "dim white"),
            ("  Commands:\n", "bold dim"),
            ("    <match query>  ", "white"), ("— predict (e.g. Portugal vs Spain)\n", "dim"),
            ("    data <name>    ", "white"), ("— show raw data for a team/player before betting\n", "dim"),
            ("    today          ", "white"), ("— list today's & upcoming fixtures\n", "dim"),
            ("    dashboard      ", "white"), ("— live Bloomberg-style terminal (4-panel)\n", "dim"),
            ("    autotrade      ", "white"), ("— one-shot: scan fixtures and auto-log dry-run bets with strong edge\n", "dim"),
            ("    dashboard auto ", "white"), ("— dashboard + continuous auto-trading (dry-run)\n", "dim"),
            ("    ask <query>    ", "white"), ("— predict + LLM verdict (BET / SKIP / MARGINAL)\n", "dim"),
            ("    place <id>     ", "white"), ("— submit a logged real bet to Polymarket\n", "dim"),
            ("    sim <id>       ", "white"), ("— dry-run: simulate Polymarket execution (no real money)\n", "dim"),
            ("    trades         ", "white"), ("— view your bet history & P&L\n", "dim"),
            ("    settle <id>    ", "white"), ("— mark a trade won/lost (e.g. settle a3f9)\n", "dim"),
            ("    sports         ", "white"), ("— list all supported sports\n", "dim"),
            ("    exit           ", "white"), ("— quit\n", "dim"),
        ),
        border_style="bright_blue",
        padding=(0, 1),
    ))


def _run_prediction(query: str):
    """Run prediction and offer to log as trade. Returns the result or None."""
    from src.predictor import predict_query
    result = predict_query(query)
    if result is None:
        return None

    from src.display import terminal
    log_data = terminal.prompt_log_trade(result)
    if log_data:
        bet_outcome, bet_label, stake, odds, is_dry_run = log_data
        from src import trades as trade_log
        trade = trade_log.log(result, bet_outcome, bet_label, stake, odds, is_dry_run=is_dry_run)
        if is_dry_run:
            console.print(
                Padding(
                    Text(f"  [SIM] Dry run logged  [id: {trade['id']}]  — type 'sim {trade['id']}' to simulate execution, 'settle {trade['id']}' when result known.",
                         style="dim yellow"),
                    (0, 1),
                )
            )
        else:
            console.print(
                Padding(
                    Text(f"  Trade logged  [id: {trade['id']}]  — type 'place {trade['id']}' to execute, 'settle {trade['id']}' when result known.",
                         style="dim green"),
                    (0, 1),
                )
            )

    return result


def _show_trades():
    from src import trades as trade_log
    from src.display import terminal
    all_trades = trade_log.load()
    s = trade_log.stats(all_trades)
    terminal.render_trades(all_trades, s)


def _inspect(name: str):
    """Show all raw data for a team/player/fighter."""
    from src.inspect import (
        inspect_football_team, inspect_fighter,
        inspect_tennis_player, inspect_player,
    )
    from src.parser import parse

    # Use the parser to figure out what kind of entity this is
    result = parse(name)
    sport = result.get("sport", "football")

    if sport == "ufc":
        inspect_fighter(name)
    elif sport == "tennis":
        inspect_tennis_player(name)
    elif sport == "football":
        # Check if it looks like a player name vs a team name
        # Heuristic: if it has more than one word and looks like a person, try player first
        words = name.strip().split()
        looks_like_player = (
            len(words) >= 2
            and not any(c.isdigit() for c in name)
            and not any(kw in name.lower() for kw in ("fc", "united", "city", "athletic", "real", "club"))
        )
        if looks_like_player:
            inspect_player(name)
        else:
            inspect_football_team(name)
    else:
        inspect_football_team(name)


def _show_fixtures(days_ahead: int = 3):
    """Display upcoming fixtures across all major sports."""
    from src.data.scrapers.fixtures import get_todays_fixtures
    from rich.table import Table
    from rich import box
    from datetime import datetime, timezone

    console.print("\n[dim]Fetching upcoming fixtures...[/dim]")
    fixtures = get_todays_fixtures(days_ahead=days_ahead)

    if not fixtures:
        console.print("[yellow]  No fixtures found.[/yellow]")
        return

    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold dim",
        padding=(0, 1),
        show_edge=False,
    )
    table.add_column("Date", style="dim", width=12)
    table.add_column("League", style="dim", width=22)
    table.add_column("Home", style="white", width=22)
    table.add_column("Away", style="white", width=22)
    table.add_column("Status", style="dim", width=14)

    for f in fixtures[:60]:
        # Parse date
        raw_date = f.get("date", "")
        try:
            dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            date_str = dt.strftime("%a %d %b")
        except Exception:
            date_str = raw_date[:10]

        status = f.get("status", "")
        home_score = f.get("home_score", "")
        away_score = f.get("away_score", "")

        if status in ("Final", "In Progress", "Halftime") and home_score != "":
            score_str = f"{home_score} - {away_score}"
            if status == "Final":
                status_display = "[dim]FT[/dim]"
            elif status == "Halftime":
                status_display = "[yellow]HT[/yellow]"
            else:
                clock = f.get("clock", "")
                status_display = f"[bright_green]{clock}[/bright_green]"
        else:
            score_str = "vs"
            status_display = "[dim]Upcoming[/dim]"

        table.add_row(
            date_str,
            f.get("league", ""),
            f.get("home", ""),
            f.get("away", ""),
            status_display,
        )

    console.print(
        Panel(
            table,
            title="[bold white]  Upcoming Fixtures[/bold white]",
            border_style="bright_blue",
            padding=(0, 1),
        )
    )
    console.print(f"[dim]  {len(fixtures)} fixtures found  ·  type a match name to predict it[/dim]\n")


def _run_autotrade():
    """One-shot autotrade scan: fetch fixtures, predict, auto-log qualifying dry-run bets."""
    from src.data.scrapers.fixtures import get_todays_fixtures
    from src.autotrader import scan_and_trade, MIN_EDGE_PCT, MIN_KELLY_PCT, MAX_DAILY_TRADES
    from rich.table import Table
    from rich import box as rich_box

    console.print(
        f"\n[dim]  Scanning fixtures for edges ≥ {MIN_EDGE_PCT}% · Kelly ≥ {MIN_KELLY_PCT}% · max {MAX_DAILY_TRADES}/day …[/dim]"
    )
    fixtures = get_todays_fixtures(days_ahead=3)
    if not fixtures:
        console.print("[yellow]  No fixtures found.[/yellow]\n")
        return

    console.print(f"[dim]  {len(fixtures)} fixtures loaded — running predictions…[/dim]")
    new_trades = scan_and_trade(fixtures, dry_run=True)

    if not new_trades:
        console.print(
            Panel(
                Text("  No qualifying edges found in current fixtures.\n"
                     "  Try again later or lower AUTOTRADE_MIN_EDGE in .env.", style="dim"),
                title="[dim]Autotrade — no bets[/dim]",
                border_style="dim",
            )
        )
        return

    table = Table(box=rich_box.SIMPLE, show_header=True, header_style="bold dim", padding=(0, 1))
    table.add_column("ID",    style="dim",       width=8)
    table.add_column("Match",                    min_width=22)
    table.add_column("Bet",                      min_width=12)
    table.add_column("Edge%", justify="right",   width=6)
    table.add_column("Kelly%", justify="right",  width=7)
    table.add_column("Stake",  justify="right",  width=5)

    for t in new_trades:
        edge = t.get("edge_pct")
        edge_str = f"+{edge:.1f}" if edge else "—"
        edge_style = "bright_green" if (edge or 0) >= 5 else "yellow"
        table.add_row(
            t["id"],
            f"{t['entity1']} v {t['entity2']}",
            t.get("bet_label", "—"),
            f"[{edge_style}]{edge_str}[/{edge_style}]",
            str(t.get("kelly_stake_pct") or "—"),
            str(t.get("stake") or "—"),
        )

    console.print(
        Panel(
            table,
            title=f"[bold bright_green]  Autotrade — {len(new_trades)} dry-run bet{'s' if len(new_trades) != 1 else ''} logged[/bold bright_green]",
            border_style="bright_green",
            padding=(0, 1),
        )
    )
    console.print(
        f"[dim]  Run 'trades' to review · 'sim <id>' to simulate execution · 'settle <id>' when results are in[/dim]\n"
    )


def _settle(args: str):
    from src import trades as trade_log
    from src.display import terminal
    from rich.prompt import Confirm

    parts = args.strip().split()
    if not parts:
        console.print("  Usage: settle <trade-id>", style="dim red")
        return

    trade_id = parts[0]
    trades = trade_log.load()
    match = next((t for t in trades if t["id"] == trade_id), None)

    if not match:
        console.print(f"  Trade '{trade_id}' not found.", style="red")
        return

    console.print(
        f"\n  [dim]{match['entity1']} vs {match['entity2']}  ·  Bet: [bold]{match['bet_label']}[/bold]  ·  Stake: {match['stake']}[/dim]"
    )
    won = Confirm.ask("  Did this bet win?")
    t = trade_log.settle(trade_id, won)
    pnl = t["pnl"]
    sign = "+" if pnl >= 0 else ""
    style = "bright_green" if pnl >= 0 else "red"
    console.print(
        Padding(
            Text(f"  Settled {'WIN' if won else 'LOSS'}  ·  P&L: {sign}{pnl:.1f}", style=style),
            (0, 1),
        )
    )


def _dispatch(line: str) -> bool:
    """Handle one REPL line. Returns False to exit."""
    cmd = line.strip()
    if not cmd:
        return True

    low = cmd.lower()

    if low in ("exit", "quit", "q"):
        return False

    if low in ("trades", "t", "history"):
        _show_trades()
        return True

    if low.startswith("settle ") or low.startswith("s "):
        args = cmd.split(" ", 1)[1]
        _settle(args)
        return True

    if low in ("sports", "show sports", "list sports"):
        from src.display import terminal
        terminal.render_sports_list()
        return True

    if low in ("today", "fixtures", "schedule", "games"):
        _show_fixtures(days_ahead=3)
        return True

    if low in ("dashboard", "live", "monitor", "terminal"):
        from src.dashboard import run_dashboard
        run_dashboard(scan_interval=60.0, days_ahead=2)
        return True

    if low in ("dashboard auto", "dashboard autotrade", "live auto"):
        from src.dashboard import run_dashboard
        run_dashboard(scan_interval=60.0, days_ahead=2, autotrade=True)
        return True

    if low in ("autotrade", "autobet", "auto"):
        _run_autotrade()
        return True

    if low.startswith("ask "):
        query = cmd.split(" ", 1)[1].strip()
        if query:
            from src.verdict import ask_about
            ask_about(query)
        else:
            console.print("  Usage: ask <match query>", style="dim red")
        return True

    if low.startswith("place "):
        trade_id = cmd.split(" ", 1)[1].strip()
        if trade_id:
            from src.bet_cli import place_logged_trade
            place_logged_trade(trade_id)
        else:
            console.print("  Usage: place <trade-id>", style="dim red")
        return True

    if low.startswith("sim "):
        trade_id = cmd.split(" ", 1)[1].strip()
        if trade_id:
            from src.bet_cli import simulate_logged_trade
            simulate_logged_trade(trade_id)
        else:
            console.print("  Usage: sim <trade-id>", style="dim red")
        return True

    if low.startswith("today "):
        # e.g. "today 7" for 7 days ahead
        try:
            days = int(cmd.split()[1])
        except (IndexError, ValueError):
            days = 3
        _show_fixtures(days_ahead=days)
        return True

    if low.startswith("data ") or low.startswith("inspect "):
        name = cmd.split(" ", 1)[1].strip()
        if name:
            _inspect(name)
        else:
            console.print("  Usage: data <team or player name>", style="dim red")
        return True

    if low in ("help", "h", "?"):
        _banner()
        return True

    # Treat anything else as a match prediction query
    _run_prediction(cmd)
    return True


def _repl():
    _banner()
    while True:
        try:
            line = console.input("\n[bright_blue]>[/bright_blue] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]  Goodbye.[/dim]\n")
            break
        if not _dispatch(line):
            console.print("[dim]  Goodbye.[/dim]\n")
            break


def main():
    if len(sys.argv) >= 2:
        # One-shot mode — single command then exit
        query = " ".join(sys.argv[1:])
        low = query.strip().lower()
        if low in ("trades", "t", "history"):
            _show_trades()
        elif low.startswith("settle "):
            _settle(query.split(" ", 1)[1])
        elif low in ("sports", "show sports"):
            from src.display import terminal
            terminal.render_sports_list()
        elif low in ("today", "fixtures", "schedule"):
            _show_fixtures(days_ahead=3)
        elif low in ("dashboard", "live", "monitor", "terminal"):
            from src.dashboard import run_dashboard
            run_dashboard(scan_interval=60.0, days_ahead=2)
        elif low in ("dashboard auto", "dashboard autotrade", "live auto"):
            from src.dashboard import run_dashboard
            run_dashboard(scan_interval=60.0, days_ahead=2, autotrade=True)
        elif low in ("autotrade", "autobet", "auto"):
            _run_autotrade()
        elif low.startswith("ask "):
            from src.verdict import ask_about
            ask_about(query.split(" ", 1)[1].strip())
        elif low.startswith("place "):
            from src.bet_cli import place_logged_trade
            place_logged_trade(query.split(" ", 1)[1].strip())
        elif low.startswith("sim "):
            from src.bet_cli import simulate_logged_trade
            simulate_logged_trade(query.split(" ", 1)[1].strip())
        elif low.startswith("data ") or low.startswith("inspect "):
            name = query.split(" ", 1)[1].strip()
            _inspect(name)
        else:
            from src.predictor import predict_query
            predict_query(query)
    else:
        _repl()


if __name__ == "__main__":
    main()
