"""
Autotrader — scans upcoming fixtures and auto-logs dry-run trades for strong edges.

No API keys required. If ANTHROPIC_API_KEY is set and AUTOTRADE_LLM_FILTER=true,
Claude's verdict is used as a secondary filter (only BET/MARGINAL verdicts pass).

Configurable via .env:
  AUTOTRADE_MIN_EDGE=5.0        minimum edge % to qualify (default: 5.0)
  AUTOTRADE_MIN_KELLY=1.0       minimum Kelly % to qualify (default: 1.0)
  AUTOTRADE_MIN_CONFIDENCE=55   minimum model confidence (default: 55)
  AUTOTRADE_MAX_DAILY=10        max auto-trades logged per day (default: 10)
  AUTOTRADE_LLM_FILTER=false    use Claude verdict as secondary filter (default: false)
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

MIN_EDGE_PCT = float(os.getenv("AUTOTRADE_MIN_EDGE", "5.0"))
MIN_KELLY_PCT = float(os.getenv("AUTOTRADE_MIN_KELLY", "1.0"))
MIN_CONFIDENCE = float(os.getenv("AUTOTRADE_MIN_CONFIDENCE", "55"))
MAX_DAILY_TRADES = int(os.getenv("AUTOTRADE_MAX_DAILY", "10"))
USE_LLM_FILTER = os.getenv("AUTOTRADE_LLM_FILTER", "false").lower() == "true"


def _already_logged(entity1: str, entity2: str, date: str, trades: list) -> bool:
    e1, e2, d = entity1.lower(), entity2.lower(), date[:10]
    for t in trades:
        if (
            t.get("entity1", "").lower() == e1
            and t.get("entity2", "").lower() == e2
            and (t.get("date") or "")[:10] == d
        ):
            return True
    return False


def _today_auto_count(trades: list) -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return sum(
        1 for t in trades
        if t.get("auto_traded") and (t.get("timestamp") or "")[:10] == today
    )


def _qualifies(result) -> bool:
    """Rule-based filter — no LLM required."""
    if result.best_edge_pct < MIN_EDGE_PCT:
        return False
    if result.kelly_stake_pct < MIN_KELLY_PCT:
        return False
    if result.confidence < MIN_CONFIDENCE:
        return False
    if not result.market_probs:
        return False
    return True


def _llm_approves(result) -> bool:
    """Optional LLM secondary filter. Returns True if verdict is BET or MARGINAL."""
    try:
        from src.verdict import get_verdict, is_configured
        if not is_configured():
            return True  # no key → don't block
        verdict = get_verdict(result)
        if verdict is None:
            return True
        return verdict.decision in ("BET", "MARGINAL")
    except Exception:
        return True


def scan_and_trade(fixtures: list, dry_run: bool = True) -> list:
    """
    Predict fixtures, filter by thresholds, auto-log qualifying trades.
    Returns list of newly logged trade dicts.
    Skips fixtures already logged (deduplicates by entity1+entity2+date).
    """
    from src.predictor import SPORT_HANDLERS
    from src import trades as trade_log
    from src.display.terminal import _outcome_label

    try:
        from src.dashboard import _infer_sport
    except ImportError:
        def _infer_sport(f):
            return "football"

    existing = trade_log.load()
    if _today_auto_count(existing) >= MAX_DAILY_TRADES:
        return []

    newly_logged: list = []

    for fixture in fixtures:
        if _today_auto_count(existing) + len(newly_logged) >= MAX_DAILY_TRADES:
            break

        home = fixture.get("home", "")
        away = fixture.get("away", "")
        date = (fixture.get("date") or "")[:10]
        if not home or not away or not date:
            continue

        if _already_logged(home, away, date, existing + newly_logged):
            continue

        sport = _infer_sport(fixture)
        handler = SPORT_HANDLERS.get(sport)
        if handler is None:
            continue

        try:
            result = handler.predict(home, away, date, {
                "competition": fixture.get("league", ""),
                "is_neutral": False,
                "venue": fixture.get("venue", ""),
            })
        except Exception:
            continue

        if not _qualifies(result):
            continue

        if USE_LLM_FILTER and not _llm_approves(result):
            continue

        best_key = result.best_bet
        if not best_key:
            continue

        bet_label = _outcome_label(best_key, result.entity1, result.entity2)
        trade = trade_log.log(
            result,
            best_key,
            bet_label,
            stake=round(result.kelly_stake_pct, 1),
            odds=None,
            is_dry_run=dry_run,
        )

        # Stamp auto_traded after log (log() doesn't know about this field)
        all_trades = trade_log.load()
        for t in all_trades:
            if t["id"] == trade["id"]:
                t["auto_traded"] = True
                break
        trade_log.save(all_trades)
        trade["auto_traded"] = True

        newly_logged.append(trade)

    return newly_logged


def summary_line() -> str:
    """One-line summary of today's auto-trade activity for status panels."""
    from src import trades as trade_log
    trades = trade_log.load()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    auto_today = [
        t for t in trades
        if t.get("auto_traded") and (t.get("timestamp") or "")[:10] == today
    ]
    if not auto_today:
        return "0 auto-trades today"
    sports = {}
    for t in auto_today:
        sp = t.get("sport", "?")
        sports[sp] = sports.get(sp, 0) + 1
    sp_str = "  ".join(f"{v}×{k[:3].upper()}" for k, v in sports.items())
    return f"{len(auto_today)} auto  {sp_str}"
