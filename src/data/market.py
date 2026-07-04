"""Fetch market odds from Polymarket and Kalshi (no API key required)."""
import requests
import re
import time as _time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from src.data import cache

POLYMARKET_API = "https://clob.polymarket.com/markets"
KALSHI_API = "https://trading-api.kalshi.com/trade-api/v2/events"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SportsBettingPredictor/1.0)",
    "Accept": "application/json",
}


def _search_polymarket(query: str) -> Optional[dict]:
    """Search Polymarket for a sports market matching query."""
    cached = cache.get("polymarket", {"q": query})
    if cached:
        return cached

    try:
        params = {"limit": 20, "active": "true"}
        resp = requests.get(POLYMARKET_API, params=params, headers=HEADERS, timeout=(3, 6))
        if resp.status_code != 200:
            return None

        data = resp.json()
        markets = data.get("data", []) if isinstance(data, dict) else data

        query_words = set(query.lower().split())
        best = None
        best_score = 0

        for mkt in markets:
            q = mkt.get("question", "").lower()
            overlap = len(query_words & set(q.split()))
            if overlap > best_score:
                best_score = overlap
                best = mkt

        if best and best_score >= 2:
            tokens = best.get("tokens", [])
            probs = {}
            for tok in tokens:
                outcome = tok.get("outcome", "").lower()
                price = float(tok.get("price", 0))
                probs[outcome] = price

            result = {"source": "polymarket", "market": best.get("question"), "probs": probs}
            cache.set("polymarket", {"q": query}, result, ttl_seconds=1800)
            return result
    except Exception:
        pass
    return None


def _search_kalshi(query: str) -> Optional[dict]:
    """Search Kalshi for a sports event."""
    cached = cache.get("kalshi", {"q": query})
    if cached:
        return cached

    try:
        params = {"limit": 20, "status": "open"}
        resp = requests.get(KALSHI_API, params=params, headers=HEADERS, timeout=(3, 6))
        if resp.status_code != 200:
            return None

        events = resp.json().get("events", [])
        query_words = set(query.lower().split())

        for event in events:
            title = event.get("title", "").lower()
            overlap = len(query_words & set(title.split()))
            if overlap >= 2:
                markets = event.get("markets", [])
                probs = {}
                for mkt in markets:
                    yes_price = mkt.get("yes_ask", 0) / 100
                    no_price = 1 - yes_price
                    subtitle = mkt.get("subtitle", mkt.get("title", ""))
                    probs[subtitle.lower()] = yes_price

                result = {"source": "kalshi", "market": event.get("title"), "probs": probs}
                cache.set("kalshi", {"q": query}, result, ttl_seconds=1800)
                return result
    except Exception:
        pass
    return None


def get_prop_market_odds(team1: str, team2: str, bet_type: str,
                         prop_params: dict, player: str = "") -> Optional[dict]:
    """
    Search Polymarket/Kalshi for a specific prop market.
    Returns {outcome_key: prob} or None.
    """
    # Build a focused search query
    stat = prop_params.get("stat", "")
    threshold = prop_params.get("threshold", "")
    direction = prop_params.get("direction", "")
    foot = prop_params.get("foot", "")

    if bet_type == "over_under":
        query = f"{team1} {team2} over {threshold} {stat}s"
    elif bet_type == "btts":
        query = f"{team1} {team2} both teams score"
    elif bet_type in ("player_scorer", "player_foot", "player_header", "player_first_scorer"):
        query = f"{player or team1} score goal"
    elif bet_type == "method_victory":
        query = f"{team1} {team2} {prop_params.get('method', 'ko')} victory"
    elif bet_type == "goes_distance":
        query = f"{team1} {team2} goes distance"
    elif bet_type == "tennis_first_set":
        query = f"{team1} {team2} first set"
    elif bet_type == "tennis_tiebreak":
        query = f"{team1} {team2} tiebreak"
    else:
        query = f"{team1} {team2} {bet_type}"

    pm = _search_polymarket(query)
    if pm and pm.get("probs"):
        return _normalize_prop_probs(pm["probs"], bet_type, prop_params, team1, team2)

    ka = _search_kalshi(query)
    if ka and ka.get("probs"):
        return _normalize_prop_probs(ka["probs"], bet_type, prop_params, team1, team2)

    return None


def _normalize_prop_probs(probs: dict, bet_type: str, prop_params: dict,
                           t1: str, t2: str) -> Optional[dict]:
    """Map raw market probs to the correct outcome keys for a prop type."""
    if bet_type == "over_under":
        threshold = prop_params.get("threshold", 2.5)
        over_key = f"over_{threshold}"
        under_key = f"under_{threshold}"
        for k, v in probs.items():
            if "over" in k or "yes" in k or "more" in k:
                rest = 1 - v
                return {"over": round(v, 4), "under": round(rest, 4)}
        return None

    if bet_type == "btts":
        for k, v in probs.items():
            if "yes" in k or "both" in k:
                return {"yes": round(v, 4), "no": round(1 - v, 4)}

    if bet_type in ("player_scorer", "player_foot", "player_header"):
        for k, v in probs.items():
            if "yes" in k or "score" in k or "goal" in k:
                return {"scores": round(v, 4), "no_goal": round(1 - v, 4)}

    if bet_type == "goes_distance":
        for k, v in probs.items():
            if "yes" in k or "distance" in k or "full" in k:
                return {"distance": round(v, 4), "finish": round(1 - v, 4)}

    return None


def get_market_odds(team1: str, team2: str, sport: str = "") -> dict:
    """Fetch implied probabilities from prediction markets.

    Returns {home_win, draw, away_win} (vig removed), or None if no market.

    Pinnacle, Polymarket and Kalshi are queried concurrently with tight
    timeouts, and the result — including a 'no market found' miss — is cached, so
    a prediction never blocks for more than a few seconds and repeats are instant.
    Most non-headline / non-football fixtures simply have no market, which is
    exactly the case that used to cost ~60s of sequential timeouts.
    """
    ck = {"t1": team1.lower(), "t2": team2.lower(), "s": (sport or "").lower()}
    cached = cache.get("market_odds", ck)
    if cached is not None:
        return cached.get("v")  # {"v": <result-or-None>}

    query = f"{team1} {team2}".strip()

    def _pinnacle():
        from src.data.scrapers.odds_extra import get_pinnacle_odds
        p = get_pinnacle_odds(team1, team2, sport or "football")
        if not p:
            return None
        return _normalize_football_probs(
            {
                team1.lower().split()[0]: p.get("home_win", 0),
                "draw": p.get("draw"),
                team2.lower().split()[0]: p.get("away_win", 0),
            },
            team1, team2,
        ) or p

    def _poly():
        pm = _search_polymarket(query)
        return _normalize_football_probs(pm["probs"], team1, team2) if pm else None

    def _kal():
        ka = _search_kalshi(query)
        return _normalize_football_probs(ka["probs"], team1, team2) if ka else None

    # Run all three at once with a hard wall-clock deadline. A source that blows
    # past the deadline is abandoned (its thread finishes in the background and
    # populates its own cache), so a prediction never waits more than ~8s.
    result = None
    _DEADLINE = 8.0
    pool = ThreadPoolExecutor(max_workers=3)
    futures = {"pinnacle": pool.submit(_pinnacle),
               "poly": pool.submit(_poly),
               "kalshi": pool.submit(_kal)}
    deadline = _time.monotonic() + _DEADLINE
    found = {}
    for name, fut in futures.items():
        remaining = max(0.0, deadline - _time.monotonic())
        try:
            found[name] = fut.result(timeout=remaining)
        except Exception:
            found[name] = None
    pool.shutdown(wait=False)
    result = found.get("pinnacle") or found.get("poly") or found.get("kalshi")

    # Cache hit and miss alike (short ttl) so we never re-pay the lookup soon.
    cache.set("market_odds", ck, {"v": result}, ttl_seconds=600)
    return result


def _normalize_football_probs(probs: dict, team1: str, team2: str) -> dict:
    """Map raw market probs to home/draw/away structure and remove vig."""
    t1 = team1.lower().split()[0]
    t2 = team2.lower().split()[0]

    home_p = draw_p = away_p = None
    for k, v in probs.items():
        if t1 in k:
            home_p = v
        elif t2 in k:
            away_p = v
        elif "draw" in k or "tie" in k:
            draw_p = v

    if home_p is None and away_p is None:
        vals = list(probs.values())
        if len(vals) >= 2:
            home_p, away_p = vals[0], vals[1]
            draw_p = vals[2] if len(vals) > 2 else None

    if home_p is None:
        return None

    if draw_p is None:
        total = home_p + away_p
        home_p /= total
        away_p /= total
        return {"home_win": round(home_p, 4), "draw": None, "away_win": round(away_p, 4)}

    total = home_p + (draw_p or 0) + away_p
    return {
        "home_win": round(home_p / total, 4),
        "draw": round(draw_p / total, 4),
        "away_win": round(away_p / total, 4),
    }
