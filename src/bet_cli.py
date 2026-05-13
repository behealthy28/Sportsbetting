"""
CLI wrapper for placing logged trades on Polymarket via src.execution.place_bet.

Flow:
  1. Look up the trade by ID from data/trades.json
  2. Show details + confirmation prompt with wallet, market, side, stake
  3. Call src.execution.find_market then src.execution.place_bet
  4. Update the trade record with the execution metadata

Requires POLY_PRIVATE_KEY + POLY_API_KEY in .env.
"""
from __future__ import annotations

import datetime
import os
import uuid
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text
from rich import box

from src import execution, trades as trade_log

console = Console()

MAX_BET_USDC = float(os.getenv("MAX_BET_USDC", "100"))


def _find_trade(trade_id: str) -> Optional[dict]:
    all_trades = trade_log.load()
    return next((t for t in all_trades if t["id"] == trade_id), None)


def _update_execution(trade_id: str, execution_info: dict) -> None:
    """Persist execution metadata back to data/trades.json."""
    all_trades = trade_log.load()
    for t in all_trades:
        if t["id"] == trade_id:
            t["execution"] = execution_info
            break
    trade_log.save(all_trades)


def _confirmation_panel(trade: dict) -> Panel:
    body = Table.grid(padding=(0, 2))
    body.add_column(style="dim", justify="right")
    body.add_column()
    body.add_row("Match", f"{trade['entity1']}  v  {trade['entity2']}")
    body.add_row("Sport", trade.get("sport", "—"))
    body.add_row("Bet", trade.get("bet_label", trade.get("bet_outcome", "—")))
    body.add_row("Logged stake", f"${trade.get('stake', 0):.2f}")
    body.add_row("Logged odds", str(trade.get("odds") or "—"))
    edge_pct = trade.get("edge_pct")
    body.add_row("Edge %", f"{edge_pct:+.1f}" if edge_pct is not None else "—")
    body.add_row("Kelly %", str(trade.get("kelly_stake_pct") or "—"))
    body.add_row("Wallet", execution.get_wallet_address() or "[red]unknown[/red]")
    body.add_row("Max bet", f"${MAX_BET_USDC:.0f}  [dim](override via MAX_BET_USDC env var)[/dim]")
    return Panel(
        body,
        title="[bold]  PLACE BET — CONFIRM[/bold]",
        border_style="bright_blue",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def place_logged_trade(trade_id: str) -> None:
    """Place a logged trade on Polymarket. Records the execution result back to trades.json."""
    if not execution.is_configured():
        console.print("[red]  Polymarket execution not configured.[/red]")
        console.print(
            "[dim]  Set POLY_PRIVATE_KEY + POLY_API_KEY in .env to enable bet placement.[/dim]"
        )
        return

    trade = _find_trade(trade_id)
    if trade is None:
        console.print(f"[red]  Trade '{trade_id}' not found.[/red]")
        console.print("[dim]  Type 'trades' to list logged trades.[/dim]")
        return

    if trade.get("execution"):
        existing = trade["execution"].get("order_id", "?")
        console.print(
            f"[yellow]  Trade already executed (order: {existing}).[/yellow]"
        )
        if not Confirm.ask("  Place it again anyway?"):
            return

    console.print(_confirmation_panel(trade))

    # Search Polymarket for a matching market
    console.print("\n[dim]  Searching Polymarket for matching market...[/dim]")
    market_id = execution.find_market(trade["entity1"], trade["entity2"])
    if not market_id:
        console.print(
            "[red]  No active Polymarket market found for this matchup.[/red]"
        )
        console.print(
            "[dim]  Try a different matchup name, or place manually via the web UI.[/dim]"
        )
        return
    console.print(f"  Market: [dim]{market_id}[/dim]")

    # Polymarket markets are typically YES/NO on a question — the user picks
    # which side of the question to take. We don't auto-infer here because
    # the question text varies per market.
    outcome = Prompt.ask(
        "\n  Bet YES or NO on the Polymarket question",
        choices=["yes", "no"],
        default="yes",
    )

    default_stake = min(float(trade.get("stake") or 25), MAX_BET_USDC)
    stake_str = Prompt.ask(
        f"  Stake (USDC, max ${MAX_BET_USDC:.0f})",
        default=f"{default_stake:.2f}",
    )
    try:
        stake = float(stake_str)
    except ValueError:
        console.print("[red]  Invalid stake — must be a number.[/red]")
        return
    if stake <= 0 or stake > MAX_BET_USDC:
        console.print(
            f"[red]  Stake must be between $0.01 and ${MAX_BET_USDC:.0f}.[/red]"
        )
        return

    console.print()
    summary = (
        f"  [bold red]BUY {outcome.upper()} · ${stake:.2f} USDC · "
        f"{trade['entity1']} v {trade['entity2']}[/bold red]"
    )
    console.print(summary)
    if not Confirm.ask("\n  [bold]Submit this order to Polymarket?[/bold]"):
        console.print("[dim]  Cancelled — no order sent.[/dim]")
        return

    console.print("\n[dim]  Submitting order...[/dim]")
    resp = execution.place_bet(market_id, outcome, stake)

    if "error" in resp:
        console.print(f"[red]  Order failed: {resp['error']}[/red]")
        return

    console.print(
        Panel(
            Text.assemble(
                ("  Order submitted\n\n", "bold bright_green"),
                ("  Order ID    ", "dim"),
                (f"{resp.get('order_id', '?')}\n", "white"),
                ("  Status      ", "dim"),
                (f"{resp.get('status', '?')}\n", "white"),
                ("  Price       ", "dim"),
                (f"{resp.get('price', '?')}\n", "white"),
                ("  Size        ", "dim"),
                (f"{resp.get('size', '?')}\n", "white"),
                ("  Amount USDC ", "dim"),
                (f"${resp.get('amount_usdc', stake):.2f}\n", "white"),
            ),
            border_style="bright_green",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )

    _update_execution(
        trade_id,
        {
            "order_id": resp.get("order_id"),
            "status": resp.get("status"),
            "price": resp.get("price"),
            "size": resp.get("size"),
            "amount_usdc": resp.get("amount_usdc", stake),
            "market_id": market_id,
            "outcome_token": outcome,
            "placed_at": datetime.datetime.now().isoformat(timespec="seconds"),
        },
    )
    console.print(
        f"[dim]  Trade {trade_id} updated with execution metadata.[/dim]\n"
        f"[dim]  When the market settles, run 'settle {trade_id}' to record P&L.[/dim]"
    )


def simulate_logged_trade(trade_id: str) -> None:
    """
    Dry-run: simulate Polymarket execution for a logged trade.
    Finds the market, shows what the order would look like, and records
    the simulated execution to the trade record — no real order is sent.
    """
    trade = _find_trade(trade_id)
    if trade is None:
        console.print(f"[red]  Trade '{trade_id}' not found.[/red]")
        console.print("[dim]  Type 'trades' to list logged trades.[/dim]")
        return

    is_dry = trade.get("is_dry_run", False)
    label = "[yellow][SIM][/yellow]" if is_dry else "[dim][REAL→SIM][/dim]"
    console.print()
    console.print(Panel(
        Text.assemble(
            (f"  {trade['entity1']}  v  {trade['entity2']}\n", "bold white"),
            ("  Bet      ", "dim"), (f"{trade.get('bet_label', '—')}\n", "white"),
            ("  Stake    ", "dim"), (f"${trade.get('stake', 0):.2f} (simulated — no real money)\n", "yellow"),
            ("  Edge     ", "dim"), (f"{trade.get('edge_pct', '—')} %\n", "white"),
        ),
        title=f"[bold]  DRY RUN SIMULATION  {label}[/bold]",
        border_style="yellow",
        box=box.ROUNDED,
        padding=(0, 1),
    ))

    console.print("\n[dim]  Searching Polymarket for matching market...[/dim]")
    market_id = execution.find_market(trade["entity1"], trade["entity2"])
    if not market_id:
        console.print("[yellow]  No active Polymarket market found — recording as market-not-found simulation.[/yellow]")
        market_id = "SIMULATED_NO_MARKET"

    outcome = Prompt.ask(
        "\n  Which side would you bet? YES or NO",
        choices=["yes", "no"],
        default="yes",
    )

    default_stake = min(float(trade.get("stake") or 25), MAX_BET_USDC)
    stake_str = Prompt.ask(
        f"  Stake (USDC, simulated)",
        default=f"{default_stake:.2f}",
    )
    try:
        stake = float(stake_str)
    except ValueError:
        stake = default_stake

    # Simulate order fill at mid price
    import random
    sim_price = round(0.45 + random.uniform(0, 0.1), 4)
    sim_size = round(stake / sim_price, 2)
    sim_order_id = f"SIM-{uuid.uuid4().hex[:8].upper()}"

    console.print(
        Panel(
            Text.assemble(
                ("  [SIMULATED — no order sent]\n\n", "bold yellow"),
                ("  Order ID    ", "dim"), (f"{sim_order_id}\n", "white"),
                ("  Status      ", "dim"), ("simulated_fill\n", "white"),
                ("  Market      ", "dim"), (f"{market_id[:40]}\n", "white"),
                ("  Side        ", "dim"), (f"{outcome.upper()}\n", "white"),
                ("  Price       ", "dim"), (f"{sim_price}\n", "white"),
                ("  Size        ", "dim"), (f"{sim_size}\n", "white"),
                ("  Amount USDC ", "dim"), (f"${stake:.2f}\n", "white"),
            ),
            border_style="yellow",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )

    _update_execution(
        trade_id,
        {
            "order_id": sim_order_id,
            "status": "simulated_fill",
            "price": sim_price,
            "size": sim_size,
            "amount_usdc": stake,
            "market_id": market_id,
            "outcome_token": outcome,
            "placed_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "is_simulation": True,
        },
    )
    console.print(
        f"[dim]  Dry run recorded (id: {trade_id}). "
        f"Run 'settle {trade_id}' when the result is known.[/dim]\n"
    )
