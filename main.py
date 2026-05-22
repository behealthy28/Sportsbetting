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
            ("    edges          ", "white"), ("— biggest upcoming events from Polymarket (60 days), top 20 by edge\n", "dim"),
            ("    dashboard      ", "white"), ("— live Bloomberg-style terminal (4-panel)\n", "dim"),
            ("    autotrade      ", "white"), ("— one-shot: scan fixtures and auto-log dry-run bets with strong edge\n", "dim"),
            ("    dashboard auto ", "white"), ("— dashboard + continuous auto-trading (dry-run)\n", "dim"),
            ("    ask <query>    ", "white"), ("— predict + LLM verdict (BET / SKIP / MARGINAL)\n", "dim"),
            ("    place <id>     ", "white"), ("— submit a logged real bet to Polymarket\n", "dim"),
            ("    sim <id>       ", "white"), ("— dry-run: simulate Polymarket execution (no real money)\n", "dim"),
            ("    trades         ", "white"), ("— view your bet history & P&L\n", "dim"),
            ("    settle <id>    ", "white"), ("— mark a trade won/lost (e.g. settle a3f9)\n", "dim"),
            ("    train          ", "white"), ("— train ML models + fit Dixon-Coles on real match history\n", "dim"),
            ("    backtest       ", "white"), ("— show accuracy / Brier score / calibration from held-out test set\n", "dim"),
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


def _run_train(sports: list = None):
    """Train ML models and fit Dixon-Coles on real historical data."""
    from src.training.train import main as train_main
    console.print(
        Panel(
            Text.assemble(
                ("  Training ML ensembles and Dixon-Coles model on real match history.\n\n", "dim"),
                ("  Football: scraping Understat (EPL/La Liga/Bundesliga/Serie A/Ligue 1, 2014–present)\n", "dim"),
                ("  Tennis  : downloading Jeff Sackmann ATP CSVs (2015–present)\n\n", "dim"),
                ("  First run may take 10–20 minutes due to Understat rate limits.\n", "yellow"),
                ("  Results cached — reruns are much faster.", "dim"),
            ),
            title="[bold white]  Train Models[/bold white]",
            border_style="bright_blue",
            padding=(0, 1),
        )
    )
    console.print()
    train_main(sports)


def _run_backtest(sport: str = None):
    """Display backtest accuracy, Brier score, and calibration charts."""
    from src.backtest import run_backtest, run_all_backtests
    if sport:
        run_backtest(sport.lower(), console)
    else:
        run_all_backtests(console)


_NON_SPORTS = [
    "election", "crypto", "bitcoin", "ethereum", "price", "president",
    "senate", "congress", "fed rate", "interest rate", "gdp", "war",
    "oscar", "emmy", "grammy", "nobel", "inflation", "stock", "trump",
    "biden", "harris", "elon", "spacex", "nasa",
]

_SPORT_TAGS = {
    "sports", "soccer", "football", "basketball", "tennis", "mma", "ufc",
    "boxing", "cricket", "nfl", "nba", "mlb", "nhl", "golf", "rugby",
}


def _fetch_polymarket_sports(days_ahead: int = 60) -> list:
    """
    Fetch upcoming sports markets from Polymarket gamma API.

    The gamma API returns markets sorted by endDateIso ascending but includes
    expired markets — active/closed filters are unreliable. We paginate from
    an estimated starting offset (based on empirical density ~20 mkt/day) and
    collect markets whose endDate falls between now and cutoff.
    """
    import requests
    from datetime import datetime, timezone, timedelta

    now = datetime.now(timezone.utc)
    now_str = now.strftime("%Y-%m-%d")
    cutoff = now + timedelta(days=days_ahead)
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Referer": "https://polymarket.com/",
        "Origin": "https://polymarket.com",
    }

    def _gamma_page(offset: int, limit: int = 100) -> list:
        try:
            resp = requests.get(
                "https://gamma-api.polymarket.com/markets",
                params={"limit": limit, "offset": offset,
                        "order": "endDateIso", "ascending": "true"},
                headers=headers, timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data if isinstance(data, list) else data.get("data", data.get("markets", []))
        except Exception:
            pass
        return []

    def _ed(m: dict) -> str:
        return (m.get("endDateIso") or m.get("endDate") or "")[:10]

    # --- Binary-search for the starting offset where endDate >= today ---
    lo, hi = 0, 8000
    for _ in range(14):          # log2(8000) ≈ 13 iterations
        mid = (lo + hi) // 2
        page = _gamma_page(mid, limit=1)
        if not page:
            hi = mid
            continue
        if _ed(page[0]) < now_str:
            lo = mid + 1
        else:
            hi = mid
    start_offset = max(0, lo - 50)   # step back a bit to avoid missing edge

    # --- Collect markets from start_offset until endDate > cutoff ---
    raw = []
    offset = start_offset
    consecutive_past = 0
    consecutive_future = 0

    while offset < start_offset + 3000:   # hard cap: 3000 markets past start
        batch = _gamma_page(offset, limit=100)
        if not batch:
            break

        for m in batch:
            ed = _ed(m)
            if not ed or ed < now_str:
                consecutive_past += 1
                continue
            if ed > cutoff_str:
                consecutive_future += 1
                continue
            raw.append(m)
            consecutive_past = 0
            consecutive_future = 0

        # Stop if the last market in this batch is well beyond cutoff
        last = _ed(batch[-1])
        if last and last > cutoff_str:
            break

        offset += 100

    if not raw:
        return []

    # Deduplicate
    seen_q: set = set()
    deduped = []
    for m in raw:
        q = (m.get("question") or "").strip()
        if q and q not in seen_q:
            seen_q.add(q)
            deduped.append(m)
    raw = deduped

    out = []
    for m in raw:
        q = (m.get("question") or "").lower()

        if any(kw in q for kw in _NON_SPORTS):
            continue

        # Primary: " vs " is the strongest matchup signal on Polymarket
        has_vs = " vs " in q or " vs." in q

        # Secondary: specific sport/league names that appear without "vs"
        # (e.g. "Will Arsenal win the Champions League?")
        has_sport_name = any(kw in q for kw in [
            "nfl", "nba", "mlb", "nhl", "ncaab", "ncaaf", "ncaa",
            "ufc", " mma ", "boxing", "heavyweight", "lightweight",
            "premier league", "champions league", "europa league",
            "la liga", "serie a", "bundesliga", "ligue 1",
            "world cup", "copa america", "euro 2026", "euro 2025",
            "wimbledon", "us open", "french open", "australian open",
            "grand slam", " atp ", " wta ", "roland garros",
            "formula 1", " f1 ", "grand prix", "motogp",
            "cricket", " ipl ", "test match", "odi",
            "rugby", "six nations", "super rugby",
            "super bowl", "world series", "stanley cup", "nba finals",
        ])

        if not has_vs and not has_sport_name:
            continue

        # Parse outcome prices
        probs = {}
        tokens = m.get("tokens") or []
        if tokens:
            for tok in tokens:
                outcome = (tok.get("outcome") or "").lower()
                price = float(tok.get("price") or 0)
                if outcome:
                    probs[outcome] = price
        else:
            try:
                outcomes = m.get("outcomes") or "[]"
                prices = m.get("outcomePrices") or "[]"
                if isinstance(outcomes, str):
                    outcomes = _json.loads(outcomes)
                if isinstance(prices, str):
                    prices = _json.loads(prices)
                for o, p in zip(outcomes, prices):
                    probs[str(o).lower()] = float(p)
            except Exception:
                pass

        volume = float(m.get("volume_num") or m.get("volumeNum") or m.get("volume") or 0)
        end_date = (
            m.get("end_date_iso") or m.get("endDateIso") or m.get("endDate") or ""
        )
        out.append({
            "question": m.get("question", ""),
            "end_date": end_date[:10] if end_date else "",
            "probs": probs,
            "volume": volume,
        })

    out.sort(key=lambda x: x["volume"], reverse=True)
    return out


def _parse_matchup(question: str):
    """Extract (entity1, entity2) from a Polymarket question. Returns (None, None) if ambiguous."""
    import re
    # Match "X vs Y" or "X v Y" with flexible surrounding context
    m = re.search(
        r'([A-Za-z][A-Za-z\s\'\-\.]{1,28?}?)\s+vs?\.?\s+([A-Za-z][A-Za-z\s\'\-\.]{1,28?}?)'
        r'(?=\s*[\?\-:\|,]|\s+(?:to\s|in\s|at\s|for\s|game|match|fight|bout|final|who|which|2024|2025|2026)|$)',
        question, re.IGNORECASE,
    )
    if not m:
        return None, None
    e1 = re.sub(r'(?i)^(will\s+|who\s+wins\s+|does\s+|can\s+)', '', m.group(1)).strip()
    e2 = re.sub(r'(?i)\s+(win|beat|wins?|to\s+win).*$', '', m.group(2)).strip()
    if e1 and e2 and len(e1) > 2 and len(e2) > 2:
        return e1.strip(), e2.strip()
    return None, None


def _infer_sport_from_question(question: str, tags: list) -> str:
    q = question.lower()
    t = " ".join(tags).lower()
    combined = q + " " + t
    if any(k in combined for k in ("ufc", "mma", "fight", "knockout", "submission", "bout")):
        return "ufc"
    if any(k in combined for k in ("tennis", "atp", "wta", "grand slam", "wimbledon", "open")):
        return "tennis"
    if any(k in combined for k in ("boxing", "heavyweight", "title bout", "round")):
        return "boxing"
    if any(k in combined for k in ("cricket", "ipl", "test match", "odi")):
        return "cricket"
    return "football"


def _show_edges(days_ahead: int = 60, top_n: int = 20):
    """Pull biggest upcoming events from Polymarket, run pipeline, rank by edge."""
    from src.predictor import SPORT_HANDLERS
    from src.display.terminal import _outcome_label
    from src.market.edge import calculate_edge, best_bet
    from rich.table import Table
    from rich import box

    console.print("\n[dim]  Pulling sports markets from Polymarket (next 60 days, sorted by volume)...[/dim]")
    markets = _fetch_polymarket_sports(days_ahead=days_ahead)

    if not markets:
        console.print(
            "[yellow]  Could not reach Polymarket API — check your internet connection.[/yellow]\n"
        )
        return

    console.print(f"[dim]  {len(markets)} sports markets found — running EdgeFinder pipeline...[/dim]\n")

    rows = []
    no_data = []
    checked = 0

    for mkt in markets:
        question = mkt["question"]
        e1, e2 = _parse_matchup(question)
        if not e1 or not e2:
            no_data.append({"question": question, "reason": "cannot parse matchup"})
            continue

        sport = _infer_sport_from_question(question, [])
        handler = SPORT_HANDLERS.get(sport)
        if handler is None:
            no_data.append({"question": question, "reason": "no handler"})
            continue

        checked += 1
        console.print(
            f"[dim]  [{checked}] {e1} v {e2}[/dim]" + " " * 20,
            end="\r",
        )

        try:
            result = handler.predict(e1, e2, mkt["end_date"], {
                "competition": question,
                "is_neutral": False,
            })
        except Exception:
            no_data.append({"question": question, "reason": "no data"})
            continue

        # Inject Polymarket probs directly (already fetched)
        market_probs = mkt["probs"]
        if not market_probs:
            no_data.append({"question": question, "reason": "no market prices"})
            continue

        # Build model probs in the same key space as market probs
        model_probs = {}
        for k in market_probs:
            if e1.lower() in k or "yes" in k:
                model_probs[k] = result.probabilities.get("home_win", result.probabilities.get("win", 0))
            elif e2.lower() in k or "no" in k:
                model_probs[k] = result.probabilities.get("away_win", result.probabilities.get("lose", 0))
            elif "draw" in k or "tie" in k:
                model_probs[k] = result.probabilities.get("draw", 0)

        if not model_probs:
            # Fallback: map by position
            keys = list(market_probs.keys())
            probs_list = list(result.probabilities.values())
            for i, k in enumerate(keys):
                if i < len(probs_list):
                    model_probs[k] = probs_list[i]

        edges = calculate_edge(model_probs, market_probs)
        best_key, best_info = best_bet(edges)

        if not best_info:
            edge_pct = None
            kelly = None
            model_p = market_p = None
            bet_label = "—"
        else:
            edge_pct = best_info["edge_pct"]
            model_p = best_info["model_prob"]
            market_p = best_info["market_prob"]
            # Quarter-Kelly
            if market_p and market_p > 0:
                dec = 1 / market_p
                b = dec - 1
                q = 1 - model_p
                raw_k = (model_p * b - q) / b
                kelly = max(0.0, raw_k / 4) * 100
            else:
                kelly = None
            bet_label = best_key.title() if best_key else "—"

        rows.append({
            "question": question,
            "e1": result.entity1,
            "e2": result.entity2,
            "sport": sport,
            "date": mkt["end_date"],
            "volume": mkt["volume"],
            "edge_pct": edge_pct,
            "kelly": kelly,
            "model_p": model_p,
            "market_p": market_p,
            "bet_label": bet_label,
        })

    console.print(" " * 80, end="\r")

    # Sort by edge descending, show top N
    rows.sort(key=lambda x: x["edge_pct"] if x["edge_pct"] is not None else -999, reverse=True)
    rows = rows[:top_n]

    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold dim",
        padding=(0, 1),
        show_edge=False,
    )
    table.add_column("Date",      style="dim",       width=10)
    table.add_column("Sport",     style="dim",       width=9)
    table.add_column("Match",                        min_width=28)
    table.add_column("Best Bet",                     min_width=14)
    table.add_column("Model%",    justify="right",   width=7)
    table.add_column("Mkt%",      justify="right",   width=6)
    table.add_column("Edge",      justify="right",   width=8)
    table.add_column("Kelly%",    justify="right",   width=7)
    table.add_column("Signal",                       width=10)

    for r in rows:
        edge_pct = r["edge_pct"]
        if edge_pct is not None:
            if edge_pct >= 5:
                edge_str = f"[bright_green]+{edge_pct:.1f}%[/bright_green]"
                signal   = "[bright_green]STRONG ✦[/bright_green]"
            elif edge_pct >= 2:
                edge_str = f"[yellow]+{edge_pct:.1f}%[/yellow]"
                signal   = "[yellow]MODERATE[/yellow]"
            elif edge_pct > 0:
                edge_str = f"[dim]+{edge_pct:.1f}%[/dim]"
                signal   = "[dim]WEAK[/dim]"
            else:
                edge_str = f"[red]{edge_pct:.1f}%[/red]"
                signal   = "[red]NEGATIVE[/red]"
        else:
            edge_str = "[dim]—[/dim]"
            signal   = "[dim]—[/dim]"

        table.add_row(
            r["date"],
            r["sport"].upper()[:8],
            f"{r['e1']} v {r['e2']}",
            r["bet_label"],
            f"{r['model_p']*100:.1f}%" if r["model_p"] else "—",
            f"{r['market_p']*100:.1f}%" if r["market_p"] else "—",
            edge_str,
            f"{r['kelly']:.1f}%" if r["kelly"] else "—",
            signal,
        )

    console.print(
        Panel(
            table,
            title=f"[bold white]  Top {top_n} Edges — Biggest Polymarket Sports (next {days_ahead} days)[/bold white]",
            border_style="bright_green",
            padding=(0, 1),
        )
    )

    no_data_count = len(no_data)
    if no_data_count:
        console.print(
            f"[dim]  {no_data_count} markets skipped (no matchup data / unsupported sport) · "
            f"type a match name to run any prediction manually[/dim]\n"
        )
    else:
        console.print(f"[dim]  type a match name to run the full prediction[/dim]\n")


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

    if low in ("edges", "scan", "value", "upcoming edges", "find edges"):
        _show_edges()
        return True

    if low.startswith("edges ") or low.startswith("scan "):
        try:
            days = int(cmd.split()[1])
        except (IndexError, ValueError):
            days = 5
        _show_edges(days_ahead=days)
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

    if low.startswith("train"):
        parts = cmd.split()
        sports = parts[1:] if len(parts) > 1 else None
        _run_train(sports)
        return True

    if low.startswith("backtest"):
        parts = cmd.split()
        sport = parts[1] if len(parts) > 1 else None
        _run_backtest(sport)
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
        elif low in ("edges", "scan", "value", "upcoming edges", "find edges"):
            _show_edges()
        elif low.startswith("edges ") or low.startswith("scan "):
            try:
                days = int(low.split()[1])
            except (IndexError, ValueError):
                days = 5
            _show_edges(days_ahead=days)
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
        elif low.startswith("train"):
            parts = query.split()
            sports = parts[1:] if len(parts) > 1 else None
            _run_train(sports)
        elif low.startswith("backtest"):
            parts = query.split()
            sport = parts[1] if len(parts) > 1 else None
            _run_backtest(sport)
        else:
            from src.predictor import predict_query
            predict_query(query)
    else:
        _repl()


if __name__ == "__main__":
    main()
