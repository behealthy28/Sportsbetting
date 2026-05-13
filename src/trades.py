"""Trade log — persist bets to data/trades.json and compute P&L."""
import json
import os
import uuid
from datetime import datetime
from typing import Optional

_TRADES_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "trades.json")


def _path() -> str:
    os.makedirs(os.path.dirname(_TRADES_FILE), exist_ok=True)
    return os.path.abspath(_TRADES_FILE)


def load() -> list:
    p = _path()
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return json.load(f)


def save(trades: list) -> None:
    with open(_path(), "w") as f:
        json.dump(trades, f, indent=2)


def log(
    result,
    bet_outcome: str,
    bet_label: str,
    stake: float,
    odds: Optional[float],
    is_dry_run: bool = False,
) -> dict:
    """Add a new pending trade from a PredictionResult."""
    edge_info = (result.edges or {}).get(bet_outcome, {})
    model_prob = (result.probabilities or {}).get(bet_outcome)
    market_prob = (result.market_probs or {}).get(bet_outcome) if result.market_probs else None

    trade = {
        "id": uuid.uuid4().hex[:8],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sport": result.sport,
        "entity1": result.entity1,
        "entity2": result.entity2,
        "competition": result.competition or "",
        "date": result.date,
        "bet_outcome": bet_outcome,
        "bet_label": bet_label,
        "model_prob": round(model_prob * 100, 1) if model_prob is not None else None,
        "market_prob": round(market_prob * 100, 1) if market_prob is not None else None,
        "edge_pct": edge_info.get("edge_pct"),
        "kelly_stake_pct": result.kelly_stake_pct,
        "stake": stake,
        "odds": odds,
        "status": "pending",
        "pnl": None,
        "is_dry_run": is_dry_run,
    }

    trades = load()
    trades.append(trade)
    save(trades)
    return trade


def settle(trade_id: str, won: bool) -> Optional[dict]:
    """Mark a trade as won or lost and compute P&L."""
    trades = load()
    for t in trades:
        if t["id"] == trade_id:
            t["status"] = "won" if won else "lost"
            stake = t["stake"]
            odds = t.get("odds")
            if won:
                t["pnl"] = round(stake * (odds - 1), 2) if odds else stake
            else:
                t["pnl"] = -stake
            save(trades)
            return t
    return None


def stats(trades: list) -> dict:
    settled = [t for t in trades if t["status"] in ("won", "lost")]
    won = [t for t in settled if t["status"] == "won"]
    total_stake = sum(t["stake"] for t in settled if t["stake"])
    total_pnl = sum(t["pnl"] for t in settled if t["pnl"] is not None)
    return {
        "total": len(trades),
        "pending": len([t for t in trades if t["status"] == "pending"]),
        "settled": len(settled),
        "wins": len(won),
        "win_rate": round(len(won) / len(settled) * 100, 1) if settled else 0,
        "total_stake": round(total_stake, 2),
        "total_pnl": round(total_pnl, 2),
        "roi": round(total_pnl / total_stake * 100, 1) if total_stake else 0,
    }


def brier_score(trades: list) -> dict:
    """Calibration score — lower is better. Also computes market Brier for comparison."""
    settled = [
        t for t in trades
        if t["status"] in ("won", "lost") and t.get("model_prob") is not None
    ]
    if not settled:
        return {"yours": None, "market": None, "beating": None, "n": 0}

    yours_sq = []
    market_sq = []
    for t in settled:
        outcome = 1.0 if t["status"] == "won" else 0.0
        model_p = t["model_prob"] / 100.0
        yours_sq.append((model_p - outcome) ** 2)
        if t.get("market_prob") is not None:
            market_sq.append((t["market_prob"] / 100.0 - outcome) ** 2)

    yours = round(sum(yours_sq) / len(yours_sq), 4)
    market = round(sum(market_sq) / len(market_sq), 4) if market_sq else None
    return {
        "yours": yours,
        "market": market,
        "beating": (yours < market) if market is not None else None,
        "n": len(settled),
    }


def by_sport(trades: list) -> dict:
    """Break down P&L, win rate, ROI, and avg edge by sport."""
    sports: dict = {}
    for t in trades:
        sp = t.get("sport", "unknown")
        if sp not in sports:
            sports[sp] = {"count": 0, "wins": 0, "settled": 0, "pnl": 0.0, "stake": 0.0, "edge_sum": 0.0, "edge_n": 0}
        s = sports[sp]
        s["count"] += 1
        if t["status"] in ("won", "lost"):
            s["settled"] += 1
            s["stake"] += t.get("stake", 0) or 0
            s["pnl"] += t.get("pnl", 0) or 0
            if t["status"] == "won":
                s["wins"] += 1
        if t.get("edge_pct") is not None:
            s["edge_sum"] += t["edge_pct"]
            s["edge_n"] += 1

    result = {}
    for sp, s in sports.items():
        result[sp] = {
            "count": s["count"],
            "settled": s["settled"],
            "win_rate": round(s["wins"] / s["settled"] * 100, 1) if s["settled"] else None,
            "pnl": round(s["pnl"], 2),
            "roi": round(s["pnl"] / s["stake"] * 100, 1) if s["stake"] else None,
            "avg_edge": round(s["edge_sum"] / s["edge_n"], 1) if s["edge_n"] else None,
        }
    return result


def best_worst(trades: list, n: int = 3) -> dict:
    """Top-n wins and worst-n losses by P&L."""
    settled = [t for t in trades if t["status"] in ("won", "lost") and t.get("pnl") is not None]
    sorted_all = sorted(settled, key=lambda t: t["pnl"], reverse=True)
    def _fmt(t):
        return {
            "id": t["id"],
            "entity1": t["entity1"],
            "entity2": t["entity2"],
            "sport": t.get("sport", ""),
            "bet_label": t.get("bet_label", ""),
            "stake": t.get("stake"),
            "odds": t.get("odds"),
            "pnl": t["pnl"],
            "status": t["status"],
            "timestamp": t.get("timestamp", ""),
        }
    return {
        "best": [_fmt(t) for t in sorted_all[:n]],
        "worst": [_fmt(t) for t in sorted_all[-n:] if t["pnl"] < 0],
    }
