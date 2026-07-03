"""FastAPI web server — serves the dashboard and prediction API."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import traceback

from dotenv import load_dotenv
load_dotenv()

from src import parser as query_parser
from src.sports.football import FootballPredictor
from src.sports.tennis import TennisPredictor
from src.sports.ufc import UFCPredictor
from src.sports.boxing import BoxingPredictor
from src.sports.cricket import CricketPredictor
from src.sports.darts import DartsPredictor, BadmintonPredictor

app = FastAPI(title="Sports Betting Predictor", version="1.0.0")

SPORT_HANDLERS = {
    "football": FootballPredictor(),
    "soccer": FootballPredictor(),
    "tennis": TennisPredictor(),
    "ufc": UFCPredictor(),
    "mma": UFCPredictor(),
    "boxing": BoxingPredictor(),
    "cricket": CricketPredictor(),
    "darts": DartsPredictor(),
    "badminton": BadmintonPredictor(),
    "table tennis": BadmintonPredictor(),
}

SUPPORTED_SPORTS = [
    {"sport": "Football / Soccer", "emoji": "⚽", "accuracy": "~63%", "source": "FBRef, Understat, ESPN", "tip": "Dixon-Coles xG model"},
    {"sport": "Tennis", "emoji": "🎾", "accuracy": "~68%", "source": "Jeff Sackmann ATP/WTA", "tip": "Surface ELO + serve stats"},
    {"sport": "UFC / MMA", "emoji": "🥊", "accuracy": "~63%", "source": "UFCStats.com", "tip": "Strike/grapple differentials"},
    {"sport": "Boxing", "emoji": "🥊", "accuracy": "~62%", "source": "BoxRec, Tapology", "tip": "ELO + style matchups"},
    {"sport": "Cricket", "emoji": "🏏", "accuracy": "~66%", "source": "CricSheet ball-by-ball", "tip": "Batting/bowling averages"},
    {"sport": "Darts", "emoji": "🎯", "accuracy": "~67%", "source": "PDC rankings", "tip": "Most consistent sport"},
    {"sport": "Badminton", "emoji": "🏸", "accuracy": "~64%", "source": "BWF world rankings", "tip": "1v1 ELO very predictive"},
    {"sport": "Table Tennis", "emoji": "🏓", "accuracy": "~64%", "source": "ITTF rankings", "tip": "Consistent, high data"},
]

EXAMPLE_QUERIES = [
    "Portugal vs Spain Nations League",
    "Over 2.5 goals Arsenal vs Liverpool",
    "BTTS Barcelona vs Real Madrid",
    "Will Haaland score vs Real Madrid",
    "Will Neymar score with his left foot vs Brazil",
    "Djokovic vs Alcaraz — first set Wimbledon",
    "Djokovic vs Alcaraz — tiebreak Wimbledon",
    "Jon Jones vs Stipe Miocic — wins by KO",
    "Jon Jones vs Stipe Miocic — goes the distance",
    "Tyson Fury vs Oleksandr Usyk method of victory boxing",
    "India vs Australia over 350 runs T20 World Cup",
    "Rohit Sharma top scorer vs England",
]


class PredictRequest(BaseModel):
    query: str


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/sports")
async def get_sports():
    return {"sports": SUPPORTED_SPORTS, "examples": EXAMPLE_QUERIES}


@app.post("/api/predict")
async def predict(req: PredictRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    parsed = query_parser.parse(query)

    if "error" in parsed:
        raise HTTPException(status_code=422, detail=parsed["error"])

    if "command" in parsed:
        return {"command": parsed["command"], "sports": SUPPORTED_SPORTS}

    entity1 = parsed["entity1"]
    entity2 = parsed["entity2"]
    sport = parsed["sport"]
    date = parsed["date"]
    context = {
        "competition": parsed.get("competition", ""),
        "is_neutral": parsed.get("is_neutral", False),
        "venue": parsed.get("venue", ""),
    }

    handler = SPORT_HANDLERS.get(sport)
    if not handler:
        raise HTTPException(status_code=422, detail=f"Sport '{sport}' not supported")

    bet_type = parsed.get("bet_type", "match_result")
    prop_params = parsed.get("prop_params", {})
    prop_player = parsed.get("prop_player", "")

    try:
        from src.predictor import _predict_prop
        if bet_type == "match_result":
            result = handler.predict(entity1, entity2, date, context)
        else:
            result = _predict_prop(
                sport, entity1, entity2, date, context,
                bet_type, prop_params, prop_player, handler,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return _serialize(result, sport)


def _serialize(result, sport: str) -> dict:
    """Convert PredictionResult to JSON-serializable dict."""
    outcome_labels = _outcome_labels(result.entity1, result.entity2, result.probabilities)

    outcomes = []
    for key, prob in result.probabilities.items():
        if prob is None:
            continue
        edge_info = (result.edges or {}).get(key, {})
        market_prob = (result.market_probs or {}).get(key)
        outcomes.append({
            "key": key,
            "label": outcome_labels.get(key, key),
            "model_prob": round(prob * 100, 1),
            "market_prob": round(market_prob * 100, 1) if market_prob is not None else None,
            "edge_pct": edge_info.get("edge_pct"),
            "edge_rating": edge_info.get("rating"),
            "is_best_bet": key == result.best_bet,
        })

    return {
        "sport": result.sport,
        "entity1": result.entity1,
        "entity2": result.entity2,
        "date": result.date,
        "competition": result.competition,
        "venue": result.venue,
        "outcomes": outcomes,
        "best_bet": result.best_bet,
        "best_bet_label": outcome_labels.get(result.best_bet) if result.best_bet else None,
        "best_edge_pct": result.best_edge_pct,
        "kelly_stake_pct": result.kelly_stake_pct,
        "confidence": result.confidence,
        "key_factors": result.key_factors,
        "news_flags": result.news_flags,
        "data_sources": result.data_sources,
        "has_market": result.market_probs is not None,
        "bet_type": getattr(result, "bet_type", "match_result"),
        "prop_description": getattr(result, "prop_description", ""),
        "prop_player": getattr(result, "prop_player", ""),
        "model_breakdown": {
            k: {kk: round(vv * 100, 1) for kk, vv in v.items()} if v else {}
            for k, v in (result.model_breakdown or {}).items()
        },
    }


def _outcome_labels(e1: str, e2: str, probs: dict) -> dict:
    _PROP = {
        "over": "Over", "under": "Under",
        "yes": "Yes (BTTS)", "no": "No (BTTS)",
        "scores": "Scores", "no_goal": "No Goal",
        "assists": "Assists", "no_assist": "No Assist",
        "first_scorer": "First Scorer", "not_first": "Not First",
        "top_scorer": "Top Scorer", "not_top": "Not Top Scorer",
        "distance": "Goes Distance", "finish": "Stopped Early",
        "tiebreak": "Tiebreak", "no_tiebreak": "No Tiebreak",
        "p1_set1": e1, "p2_set1": e2,
        "ht_home_win": f"HT: {e1}", "ht_draw": "HT: Draw", "ht_away_win": f"HT: {e2}",
        "f1_ko": f"{e1} KO/TKO", "f1_sub": f"{e1} Sub", "f1_dec": f"{e1} Decision",
        "f2_ko": f"{e2} KO/TKO", "f2_sub": f"{e2} Sub", "f2_dec": f"{e2} Decision",
        "left_foot_goal": "Left Foot Goal", "right_foot_goal": "Right Foot Goal",
        "head_goal": "Header Goal", "other": "Other / No",
    }
    labels = {}
    for key in probs:
        if key in ("home_win", "p1_win", "f1_win"):
            labels[key] = e1
        elif key in ("away_win", "p2_win", "f2_win"):
            labels[key] = e2
        elif key == "draw":
            labels[key] = "Draw"
        elif key in _PROP:
            labels[key] = _PROP[key]
        else:
            labels[key] = key.replace("_", " ").title()
    return labels


# ── Trade log (web) ──────────────────────────────────────────────────────────

class TradeLogRequest(BaseModel):
    query: str
    outcome_key: str
    bet_label: str
    stake: float
    odds: float


class SettleRequest(BaseModel):
    won: bool


@app.post("/api/trade/log")
async def trade_log_endpoint(req: TradeLogRequest):
    from src import trades as trade_log
    parsed = query_parser.parse(req.query)
    e1 = parsed.get("entity1", req.query)
    e2 = parsed.get("entity2", "")
    sport = parsed.get("sport", "football")

    class _FakeResult:
        entity1 = e1
        entity2 = e2
        competition = ""
        date = ""
        probabilities = {req.outcome_key: None}
        market_probs = None
        edges = {}
        kelly_stake_pct = None
        sport = sport

    trade = trade_log.log(_FakeResult(), req.outcome_key, req.bet_label, req.stake, req.odds)
    return trade


@app.post("/api/trade/settle/{trade_id}")
async def trade_settle_endpoint(trade_id: str, req: SettleRequest):
    from src import trades as trade_log
    t = trade_log.settle(trade_id, req.won)
    if not t:
        raise HTTPException(status_code=404, detail=f"Trade '{trade_id}' not found")
    return t


# ── Markets ──────────────────────────────────────────────────────────────────

@app.get("/api/markets/today")
async def get_markets_today(days: int = 2, force: int = 0):
    try:
        from src.data.scrapers.fixtures import get_todays_fixtures
        from src.data import cache as _cache
        if force:
            # Bust the fixture cache so fresh data is fetched
            for d in range(1, 8):
                _cache.invalidate("fixtures_today", {"days": d, "sports": "None"})
        fixtures = get_todays_fixtures(days_ahead=max(1, min(days, 7)))
        sources = list({f.get("source", "unknown") for f in fixtures})
        return {"fixtures": fixtures[:200], "count": len(fixtures), "sources": sources}
    except Exception as e:
        import traceback
        return {"fixtures": [], "count": 0, "error": str(e), "detail": traceback.format_exc()}


@app.get("/api/debug/fixtures")
async def debug_fixtures():
    """Returns raw diagnostic info from each fixture source."""
    import requests
    from datetime import datetime
    from src.data.scrapers.fixtures import ESPN_HEADERS, SPORTSDB_HEADERS
    results = {}
    today = datetime.utcnow().strftime("%Y%m%d")
    today_iso = datetime.utcnow().strftime("%Y-%m-%d")

    # Test ESPN
    try:
        r = requests.get(
            f"https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard",
            params={"dates": today, "limit": 5},
            headers=ESPN_HEADERS, timeout=10
        )
        results["espn"] = {"status": r.status_code, "events": len(r.json().get("events", [])) if r.status_code == 200 else 0}
    except Exception as e:
        results["espn"] = {"error": str(e)}

    # Test TheSportsDB
    try:
        r = requests.get(
            "https://www.thesportsdb.com/api/v1/json/3/eventsday.php",
            params={"d": today_iso, "s": "Soccer"},
            headers=SPORTSDB_HEADERS, timeout=10
        )
        data = r.json() if r.status_code == 200 else {}
        results["sportsdb"] = {"status": r.status_code, "events": len(data.get("events") or [])}
    except Exception as e:
        results["sportsdb"] = {"error": str(e)}

    return results


# ── Watchlist ─────────────────────────────────────────────────────────────────

class WatchlistAddRequest(BaseModel):
    entity1: str
    entity2: str
    sport: str = "football"


@app.get("/api/watchlist")
async def get_watchlist():
    from src.alerts import load_watchlist
    return {"items": load_watchlist()}


@app.post("/api/watchlist/add")
async def watchlist_add(req: WatchlistAddRequest):
    from src.alerts import add_to_watchlist
    item = add_to_watchlist(req.entity1, req.entity2, req.sport)
    return {"ok": True, "item": item}


@app.delete("/api/watchlist/remove")
async def watchlist_remove(entity1: str, entity2: str):
    from src.alerts import remove_from_watchlist
    removed = remove_from_watchlist(entity1, entity2)
    return {"ok": removed}


# ── Alerts ────────────────────────────────────────────────────────────────────

@app.get("/api/alerts")
async def get_alerts(threshold: float = 5.0):
    try:
        from src.alerts import check_watchlist_movements
        alerts = check_watchlist_movements(threshold_pp=threshold)
        return {"alerts": alerts, "count": len(alerts)}
    except Exception as e:
        return {"alerts": [], "count": 0, "error": str(e)}


# ── Portfolio ─────────────────────────────────────────────────────────────────

@app.get("/api/portfolio")
async def get_portfolio():
    from src import trades as trade_log
    trades = trade_log.load()
    return {
        "trades": trades,
        "stats": trade_log.stats(trades),
        "brier": trade_log.brier_score(trades),
        "by_sport": trade_log.by_sport(trades),
        **trade_log.best_worst(trades, n=3),
    }


# ── Execution ─────────────────────────────────────────────────────────────────

class ExecutionRequest(BaseModel):
    entity1: str
    entity2: str
    outcome_key: str
    stake_usdc: float


@app.get("/api/execution/status")
async def execution_status():
    from src.execution import is_configured, get_wallet_address
    configured = is_configured()
    return {
        "configured": configured,
        "wallet_address": get_wallet_address() if configured else None,
    }


@app.post("/api/execution/place")
async def execution_place(req: ExecutionRequest):
    from src.execution import is_configured, find_market, place_bet
    if not is_configured():
        raise HTTPException(
            status_code=403,
            detail="Wallet not configured. Set POLY_PRIVATE_KEY and POLY_API_KEY in .env",
        )
    if req.stake_usdc <= 0:
        raise HTTPException(status_code=400, detail="stake_usdc must be > 0")

    market_id = find_market(req.entity1, req.entity2)
    if not market_id:
        raise HTTPException(
            status_code=404,
            detail=f"No open Polymarket market found for '{req.entity1} vs {req.entity2}'",
        )

    result = place_bet(market_id, req.outcome_key, req.stake_usdc)
    if "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])
    return result


# Mount static files last so API routes take priority
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
