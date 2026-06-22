"""Rich terminal display for prediction results."""
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich import box
from rich.rule import Rule
from rich.padding import Padding

console = Console()

OUTCOME_LABELS = {
    "home_win": "Home Win",
    "draw": "Draw",
    "away_win": "Away Win",
    "p1_win": "Win",
    "p2_win": "Win",
    "f1_win": "Win",
    "f2_win": "Win",
}

EDGE_COLORS = {
    "STRONG": "bold green",
    "MODERATE": "yellow",
    "WEAK": "dim white",
    "NEGATIVE": "red",
}


def _prob_bar(prob: float, width: int = 20) -> str:
    if prob is None:
        return "─" * width
    filled = int(round(prob * width))
    empty = width - filled
    return "█" * filled + "░" * empty


def _outcome_label(key: str, entity1: str, entity2: str, player: str = "") -> str:
    if key in ("home_win", "p1_win", "f1_win"):
        return entity1
    if key in ("away_win", "p2_win", "f2_win"):
        return entity2
    if key == "draw":
        return "Draw"
    # Prop-specific friendly names
    _PROP_LABELS = {
        "over": "Over",
        "under": "Under",
        "yes": "Yes (BTTS)",
        "no": "No (BTTS)",
        "scores": f"{player} Scores" if player else "Scores",
        "no_goal": "No Goal",
        "assists": f"{player} Assists" if player else "Assists",
        "no_assist": "No Assist",
        "first_scorer": f"{player} First" if player else "First Scorer",
        "not_first": "Not First",
        "top_scorer": f"{player} Top Scorer" if player else "Top Scorer",
        "not_top": "Not Top Scorer",
        "distance": "Goes Distance",
        "finish": "Stopped Early",
        "tiebreak": "Tiebreak",
        "no_tiebreak": "No Tiebreak",
        "p1_set1": entity1,
        "p2_set1": entity2,
        "ht_home_win": f"HT: {entity1}",
        "ht_draw": "HT: Draw",
        "ht_away_win": f"HT: {entity2}",
        "f1_ko": f"{entity1} KO/TKO",
        "f1_sub": f"{entity1} Submission",
        "f1_dec": f"{entity1} Decision",
        "f2_ko": f"{entity2} KO/TKO",
        "f2_sub": f"{entity2} Submission",
        "f2_dec": f"{entity2} Decision",
        "f1_ko_tko": f"{entity1} KO/TKO",
        "f2_ko_tko": f"{entity2} KO/TKO",
        "f1_dec_w": f"{entity1} Decision",
        "f2_dec_w": f"{entity2} Decision",
        "left_foot_goal": f"Left Foot Goal",
        "right_foot_goal": f"Right Foot Goal",
        "head_goal": "Header Goal",
        "other": "Other / No",
    }
    if key in _PROP_LABELS:
        return _PROP_LABELS[key]
    # e.g. "left_foot_goal", "2_1" correct score
    return key.replace("_", " ").replace(" goal", " Goal").title()


def render(result) -> None:
    """Render a PredictionResult to the terminal."""
    console.print()

    # Header
    sport_emoji = {
        "Football": "⚽", "Tennis": "🎾", "Ufc/Mma": "🥊", "Ufc": "🥊",
        "Boxing": "🥊", "Cricket": "🏏", "Darts": "🎯",
        "Badminton": "🏸", "Table Tennis": "🏓",
    }.get(result.sport.title(), "🏆")

    is_prop = getattr(result, "bet_type", "match_result") != "match_result"
    prop_desc = getattr(result, "prop_description", "")
    prop_player = getattr(result, "prop_player", "")

    if is_prop and prop_desc:
        title = f"{sport_emoji}  {prop_desc}"
        subtitle_parts = [f"{result.entity1} vs {result.entity2}"]
    else:
        title = f"{sport_emoji}  {result.entity1}  vs  {result.entity2}"
        subtitle_parts = []

    if result.competition:
        subtitle_parts.append(result.competition)
    subtitle_parts.append(result.date)
    if result.venue and result.venue not in ("Unknown", ""):
        subtitle_parts.append(result.venue)
    subtitle = "  •  ".join(subtitle_parts)

    border = "magenta" if is_prop else "bright_blue"
    console.print(Panel(
        Text(subtitle, justify="center", style="dim"),
        title=f"[bold white]{title}[/bold white]",
        border_style=border,
        padding=(0, 2),
    ))

    # Probability table
    prob_table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold dim",
        padding=(0, 1),
        expand=True,
    )
    prob_table.add_column("Outcome", style="white", min_width=16)
    prob_table.add_column("Probability", min_width=28)
    prob_table.add_column("Model %", justify="right", min_width=8)
    prob_table.add_column("Market %", justify="right", min_width=8)
    prob_table.add_column("Edge", justify="right", min_width=9)

    market_probs = result.market_probs or {}
    edges = result.edges or {}

    for key, prob in result.probabilities.items():
        if prob is None:
            continue

        label = _outcome_label(key, result.entity1, result.entity2,
                               getattr(result, "prop_player", ""))
        bar = _prob_bar(prob)
        model_pct = f"{prob * 100:.1f}%"

        mkt_p = market_probs.get(key)
        mkt_str = f"{mkt_p * 100:.1f}%" if mkt_p is not None else "N/A"

        edge_info = edges.get(key, {})
        edge_val = edge_info.get("edge_pct", None)
        if edge_val is not None:
            sign = "+" if edge_val > 0 else ""
            edge_str = f"{sign}{edge_val:.1f}%"
            edge_color = EDGE_COLORS.get(edge_info.get("rating", "WEAK"), "white")
        else:
            edge_str = "—"
            edge_color = "dim"

        # Highlight best bet row
        is_best = key == result.best_bet
        row_style = "bold" if is_best else ""

        prob_table.add_row(
            Text(("★ " if is_best else "  ") + label, style=f"{row_style} {'bright_yellow' if is_best else 'white'}"),
            Text(bar, style="bright_blue" if prob > 0.5 else "white"),
            Text(model_pct, style=f"{row_style} bright_white"),
            Text(mkt_str, style="dim"),
            Text(edge_str, style=f"{row_style} {edge_color}"),
        )

    console.print(Padding(prob_table, (0, 1)))

    # Key factors
    if result.key_factors or result.news_flags:
        factors_text = Text()
        for f in result.key_factors[:4]:
            factors_text.append(f"  • {f}\n", style="dim white")
        for f in result.news_flags[:3]:
            factors_text.append(f"  ⚠ {f}\n", style="yellow")

        console.print(Panel(
            factors_text,
            title="[dim]Key Factors[/dim]",
            border_style="dim",
            padding=(0, 1),
        ))

    # Recommendation box
    if result.best_bet and result.best_edge_pct > 0:
        best_label = _outcome_label(result.best_bet, result.entity1, result.entity2)
        edge_color = "bright_green" if result.best_edge_pct >= 5 else "yellow"
        edge_rating = "STRONG VALUE" if result.best_edge_pct >= 5 else "MODERATE VALUE" if result.best_edge_pct >= 2 else "WEAK VALUE"

        rec_text = Text()
        rec_text.append(f"  Best Bet: {best_label}\n", style="bold bright_white")
        rec_text.append(f"  Edge over market: ", style="dim")
        rec_text.append(f"+{result.best_edge_pct:.1f}%  [{edge_rating}]\n", style=f"bold {edge_color}")
        rec_text.append(f"  Kelly Stake: ", style="dim")
        rec_text.append(f"{result.kelly_stake_pct:.1f}% of bankroll\n", style="bold white")
        rec_text.append(f"  Model Confidence: ", style="dim")
        rec_text.append(f"{result.confidence:.0f}/100\n", style="bold white")

        console.print(Panel(
            rec_text,
            title="[bold bright_green]★  RECOMMENDED BET[/bold bright_green]",
            border_style=edge_color,
            padding=(0, 1),
        ))
    else:
        console.print(Panel(
            Text("  No clear edge detected vs market odds — pass on this one.", style="dim"),
            title="[dim]Recommendation[/dim]",
            border_style="dim",
        ))

    # Model breakdown
    breakdown_parts = []
    for model_name, model_probs in (result.model_breakdown or {}).items():
        if model_probs:
            key = next((k for k in ["home_win", "p1_win", "f1_win"] if k in model_probs), None)
            if key:
                breakdown_parts.append(f"{model_name}: {model_probs[key]*100:.1f}%")

    if breakdown_parts:
        console.print(
            Padding(
                Text("  Models: " + "  |  ".join(breakdown_parts), style="dim"),
                (0, 1)
            )
        )

    # Monte Carlo simulation
    sim = getattr(result, "simulation", None)
    if sim:
        _render_simulation(sim, result.entity1, result.entity2)

    # Data sources
    sources_str = " · ".join(result.data_sources) if result.data_sources else "Seeded data"
    console.print(Padding(Text(f"  Data: {sources_str}", style="dim"), (0, 1)))
    console.print()


def _render_simulation(sim: dict, entity1: str, entity2: str) -> None:
    """Render the Monte Carlo breakdown: scorelines, total & first-half goals."""
    n = sim.get("n_sims", 0)

    # Most frequent exact scorelines
    score_table = Table(
        box=box.SIMPLE, show_header=True, header_style="bold dim",
        padding=(0, 1), expand=True,
    )
    score_table.add_column("Scoreline", style="white", min_width=10)
    score_table.add_column("Frequency", min_width=20)
    score_table.add_column("Times", justify="right", min_width=7)
    score_table.add_column("%", justify="right", min_width=6)

    scorelines = sim.get("scorelines", [])
    top_pct = scorelines[0]["pct"] if scorelines else 1
    for s in scorelines:
        bar_w = int(round((s["pct"] / top_pct) * 18)) if top_pct else 0
        bar = "█" * bar_w + "░" * (18 - bar_w)
        is_top = s is scorelines[0]
        score_table.add_row(
            Text(("★ " if is_top else "  ") + s["score"],
                 style="bright_yellow" if is_top else "white"),
            Text(bar, style="bright_blue"),
            Text(f"{s['count']:,}", style="bright_white"),
            Text(f"{s['pct']}%", style="dim"),
        )

    # Total goals + first-half goals side by side
    def _goal_table(title: str, rows: list) -> Table:
        t = Table(box=box.SIMPLE, show_header=True, header_style="bold dim",
                  padding=(0, 1), title=title, title_style="dim")
        t.add_column("Goals", style="white", justify="center", min_width=6)
        t.add_column("Bar", min_width=14)
        t.add_column("%", justify="right", min_width=6)
        peak = max((r[2] for r in rows), default=1) or 1
        for label, count, pct in rows:
            bw = int(round((pct / peak) * 12))
            t.add_row(label, Text("█" * bw + "░" * (12 - bw), style="cyan"),
                      Text(f"{pct}%", style="dim"))
        return t

    total_tbl = _goal_table("Total goals", sim.get("total_goals", []))
    fh_tbl = _goal_table("First-half goals", sim.get("first_half_goals", []))

    ou = sim.get("over_under", {})
    summary = Text()
    summary.append(f"  Avg goals: ", style="dim")
    summary.append(f"{sim.get('avg_total_goals', 0)}", style="bright_white")
    summary.append(f"  ·  1st-half avg: ", style="dim")
    summary.append(f"{sim.get('avg_first_half_goals', 0)}", style="bright_white")
    summary.append(f"  ·  Over 2.5: ", style="dim")
    summary.append(f"{ou.get(2.5, 0)}%", style="bright_white")
    summary.append(f"  ·  BTTS: ", style="dim")
    summary.append(f"{sim.get('btts_pct', 0)}%", style="bright_white")

    body = Text.assemble(
        ("  Most likely scorelines  ", "dim"),
        (f"({entity1} home – away {entity2})\n", "dim"),
    )
    console.print(Panel(
        body,
        title=f"[bold white]🎲  Monte Carlo Simulation[/bold white]  [dim]· {n:,} runs[/dim]",
        border_style="bright_blue",
        padding=(0, 1),
    ))
    console.print(Padding(score_table, (0, 1)))
    console.print(Padding(Columns([total_tbl, fh_tbl], expand=True, equal=True), (0, 1)))
    console.print(Padding(summary, (0, 1)))


def render_sports_list():
    """Show all supported sports and their data sources."""
    console.print()
    table = Table(title="Supported Sports", box=box.ROUNDED, border_style="bright_blue")
    table.add_column("Sport", style="bold white")
    table.add_column("Accuracy Target", style="bright_green")
    table.add_column("Data Source", style="dim")
    table.add_column("Why Predictable", style="dim white")

    sports = [
        ("⚽ Football/Soccer", "~63%", "FBRef, Understat, ESPN", "Dixon-Coles xG model"),
        ("🎾 Tennis", "~68%", "Jeff Sackmann ATP/WTA CSVs", "Surface ELO, serve stats"),
        ("🥊 UFC/MMA", "~63%", "UFCStats.com", "Strike/grapple differentials"),
        ("🥊 Boxing", "~62%", "BoxRec, Tapology", "ELO + style matchups"),
        ("🏏 Cricket", "~66%", "CricSheet ball-by-ball", "Batting/bowling averages"),
        ("🎯 Darts", "~67%", "PDC rankings", "Most consistent sport"),
        ("🏸 Badminton", "~64%", "BWF world rankings", "1v1 ELO very predictive"),
        ("🏓 Table Tennis", "~64%", "ITTF rankings", "Consistent, high data"),
    ]

    for row in sports:
        table.add_row(*row)

    console.print(table)
    console.print(Padding(
        Text("Usage: python main.py \"Portugal vs Spain Nations League\"", style="dim"),
        (1, 2)
    ))
    console.print()


def render_error(message: str):
    console.print(Panel(
        Text(f"  {message}", style="red"),
        title="[red]Error[/red]",
        border_style="red",
    ))
    console.print(Padding(
        Text('Try: "Portugal vs Spain"  or  "show sports"  or  "trades"', style="dim"),
        (0, 2)
    ))


def render_trades(trades: list, stats: dict) -> None:
    """Render the full trade ledger with summary stats."""
    console.print()

    if not trades:
        console.print(Panel(
            Text("  No trades logged yet.\n  After a prediction, choose to log it as a trade.", style="dim"),
            title="[bold]Trade History[/bold]",
            border_style="dim",
        ))
        return

    table = Table(
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold dim",
        padding=(0, 1),
        title="[bold white]Trade History[/bold white]",
    )
    table.add_column("ID",       style="dim",          min_width=8)
    table.add_column("Date",     style="dim",          min_width=10)
    table.add_column("Match",                          min_width=24)
    table.add_column("Bet",                            min_width=12)
    table.add_column("Edge",     justify="right",      min_width=7)
    table.add_column("Stake",    justify="right",      min_width=6)
    table.add_column("Odds",     justify="right",      min_width=5)
    table.add_column("Status",   justify="center",     min_width=9)
    table.add_column("P&L",      justify="right",      min_width=8)

    for t in reversed(trades):
        status = t["status"]
        status_text, status_style = {
            "pending": ("pending", "dim yellow"),
            "won":     ("✓ won",   "bright_green"),
            "lost":    ("✗ lost",  "red"),
        }.get(status, (status, "white"))

        pnl = t.get("pnl")
        if pnl is None:
            pnl_str = "—"
            pnl_style = "dim"
        elif pnl >= 0:
            pnl_str = f"+{pnl:.1f}"
            pnl_style = "bright_green"
        else:
            pnl_str = f"{pnl:.1f}"
            pnl_style = "red"

        edge = t.get("edge_pct")
        edge_str = f"+{edge:.1f}%" if edge and edge > 0 else (f"{edge:.1f}%" if edge is not None else "—")
        edge_style = "green" if edge and edge >= 5 else ("yellow" if edge and edge > 0 else "red" if edge and edge < 0 else "dim")

        match_str = f"{t['entity1']} v {t['entity2']}"
        date_str = t["date"][:10] if t["date"] else t["timestamp"][:10]

        table.add_row(
            t["id"],
            date_str,
            match_str,
            t["bet_label"],
            Text(edge_str, style=edge_style),
            str(t["stake"]),
            str(t["odds"]) if t.get("odds") else "—",
            Text(status_text, style=status_style),
            Text(pnl_str, style=pnl_style),
        )

    console.print(table)

    # Summary bar
    pnl_color = "bright_green" if stats["total_pnl"] >= 0 else "red"
    pnl_sign = "+" if stats["total_pnl"] >= 0 else ""
    roi_sign = "+" if stats["roi"] >= 0 else ""

    summary = Text()
    summary.append(f"  {stats['total']} bets  ", style="dim white")
    summary.append(f"({stats['pending']} pending)  ", style="dim yellow")
    if stats["settled"]:
        summary.append(f"Win rate: ", style="dim")
        summary.append(f"{stats['win_rate']:.0f}%  ", style="bold white")
        summary.append(f"P&L: ", style="dim")
        summary.append(f"{pnl_sign}{stats['total_pnl']:.1f}  ", style=f"bold {pnl_color}")
        summary.append(f"ROI: ", style="dim")
        summary.append(f"{roi_sign}{stats['roi']:.1f}%", style=f"bold {pnl_color}")

    console.print(Padding(summary, (0, 1)))
    console.print()


def prompt_log_trade(result) -> tuple:
    """
    Ask the user whether to log a trade. Returns (outcome_key, bet_label, stake, odds)
    or None if they skip.
    """
    from rich.prompt import Prompt, Confirm
    console.print()

    if not Confirm.ask("[dim]  Log this as a trade?[/dim]", default=False):
        return None

    # Build outcome choices
    outcomes = []
    for key, prob in result.probabilities.items():
        if prob is None:
            continue
        label = _outcome_label(key, result.entity1, result.entity2,
                               getattr(result, "prop_player", ""))
        outcomes.append((key, label, prob))

    console.print("  Which outcome are you betting on?")
    for i, (key, label, prob) in enumerate(outcomes, 1):
        star = " ★" if key == result.best_bet else ""
        console.print(f"    [{i}] {label}  ({prob*100:.1f}%){star}", style="dim white")

    while True:
        choice = Prompt.ask("  Choice", default="1")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(outcomes):
                break
        except ValueError:
            pass
        console.print("  Invalid choice.", style="red")

    bet_outcome, bet_label, _ = outcomes[idx]

    stake_str = Prompt.ask("  Stake (units)", default="10")
    try:
        stake = float(stake_str)
    except ValueError:
        stake = 10.0

    odds_str = Prompt.ask("  Decimal odds (optional, Enter to skip)", default="")
    try:
        odds = float(odds_str) if odds_str else None
    except ValueError:
        odds = None

    return bet_outcome, bet_label, stake, odds
