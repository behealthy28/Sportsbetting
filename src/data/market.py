"""Fetch market odds from Polymarket and Kalshi (no API key required)."""
import requests
from typing import Optional
from src.data import cache

POLYMARKET_API = "https://clob.polymarket.com/markets"
KALSHI_API = "https://trading-api.kalshi.com/trade-api/v2/events"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SportsBettingPredictor/1.0)",
    "Accept": "application/json",
}


def _entity_in(name: str, text: str) -> bool:
    """
    True if any significant word of name (>3 chars) appears in text,
    or if the full name appears as a substring.
    More robust than first-word-only matching.
    """
    name_l = name.lower().strip()
    text_l = text.lower()
    if name_l in text_l:
        return True
    return any(w in text_l for w in name_l.split() if len(w) > 3)


def _search_polymarket(query: str) -> Optional[dict]:
    """
    Search Polymarket for a sports market matching query.
    Returns {source, market (question text), probs {outcome: price}}.
    """
    cached = cache.get("polymarket", {"q": query})
    if cached:
        return cached

    try:
        params = {"limit": 20, "active": "true"}
        resp = requests.get(POLYMARKET_API, params=params, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return None

        data = resp.json()
        markets = data.get("data", []) if isinstance(data, dict) else data

        query_words = set(w for w in query.lower().split() if len(w) > 3)
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

            result = {
                "source": "polymarket",
                "market": best.get("question", ""),
                "probs": probs,
            }
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
        resp = requests.get(KALSHI_API, params=params, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return None

        events = resp.json().get("events", [])
        query_words = set(w for w in query.lower().split() if len(w) > 3)

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

                result = {
                    "source": "kalshi",
                    "market": event.get("title", ""),
                    "probs": probs,
                }
                cache.set("kalshi", {"q": query}, result, ttl_seconds=1800)
                return result
    except Exception:
        pass
    return None


def _normalize_football_probs(
    probs: dict, team1: str, team2: str, question: str = ""
) -> Optional[dict]:
    """
    Map raw market token prices to {home_win, draw, away_win} and remove vig.

    Handles two market structures:
    - Named-outcome (e.g. "Manchester City", "Draw", "Arsenal"): matched by entity name
    - Binary YES/NO (Polymarket's most common format): mapped using the question text
      to determine which entity YES corresponds to.

    Returns None when the mapping is ambiguous rather than silently returning
    wrong probabilities.
    """
    home_p = draw_p = away_p = None

    # Pass 1: named-outcome matching using robust multi-word search
    for k, v in probs.items():
        k_s = str(k).lower()
        if _entity_in(team1, k_s):
            home_p = float(v)
        elif _entity_in(team2, k_s):
            away_p = float(v)
        elif k_s in ("draw", "tie", "x"):
            draw_p = float(v)

    # Pass 2: binary YES/NO — use question text to resolve direction
    if home_p is None and away_p is None:
        yes_p = next(
            (float(v) for k, v in probs.items() if str(k).lower() == "yes"), None
        )
        no_p = next(
            (float(v) for k, v in probs.items() if str(k).lower() == "no"), None
        )
        if yes_p is not None and no_p is not None:
            # "Will [team1] win?" → YES = team1
            if question and _entity_in(team1, question):
                home_p, away_p = yes_p, no_p
            # "Will [team2] win?" → YES = team2
            elif question and _entity_in(team2, question):
                home_p, away_p = no_p, yes_p
            else:
                # Can't determine direction safely — don't guess
                return None

    # Pass 3: positional last-resort for markets with non-standard token names
    if home_p is None and away_p is None:
        vals = [float(v) for v in probs.values() if isinstance(v, (int, float))]
        if len(vals) == 3:
            home_p, draw_p, away_p = vals[0], vals[1], vals[2]
        elif len(vals) == 2:
            home_p, away_p = vals[0], vals[1]

    if home_p is None or away_p is None:
        return None

    # Remove vig by normalizing
    total = home_p + (draw_p or 0.0) + away_p
    if total <= 0:
        return None

    if draw_p is None:
        return {
            "home_win": round(home_p / total, 4),
            "draw": None,
            "away_win": round(away_p / total, 4),
        }
    return {
        "home_win": round(home_p / total, 4),
        "draw": round(draw_p / total, 4),
        "away_win": round(away_p / total, 4),
    }


def get_market_odds(team1: str, team2: str, sport: str = "") -> Optional[dict]:
    """
    Fetch implied probabilities from prediction markets.
    Returns {home_win, draw, away_win} (normalized, vig removed), or None.
    Tries Pinnacle (sharpest market) → Polymarket → Kalshi.
    """
    # Pinnacle has the sharpest lines and lowest vig
    try:
        from src.data.scrapers.odds_extra import get_pinnacle_odds
        pinnacle = get_pinnacle_odds(team1, team2, sport or "football")
        if pinnacle:
            normalized = _normalize_football_probs(
                {
                    team1.lower().split()[0]: pinnacle.get("home_win", 0),
                    "draw": pinnacle.get("draw"),
                    team2.lower().split()[0]: pinnacle.get("away_win", 0),
                },
                team1,
                team2,
            )
            if normalized:
                return normalized
    except Exception:
        pass

    query = f"{team1} {team2}".strip()

    pm = _search_polymarket(query)
    if pm and pm.get("probs"):
        result = _normalize_football_probs(
            pm["probs"], team1, team2, question=pm.get("market", "")
        )
        if result:
            return result

    ka = _search_kalshi(query)
    if ka and ka.get("probs"):
        result = _normalize_football_probs(
            ka["probs"], team1, team2, question=ka.get("market", "")
        )
        if result:
            return result

    return None


def get_prop_market_odds(
    team1: str,
    team2: str,
    bet_type: str,
    prop_params: dict,
    player: str = "",
) -> Optional[dict]:
    """
    Search Polymarket/Kalshi for a specific prop market.
    Returns {outcome_key: prob} or None.
    """
    stat = prop_params.get("stat", "")
    threshold = prop_params.get("threshold", "")

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


def _normalize_prop_probs(
    probs: dict, bet_type: str, prop_params: dict, t1: str, t2: str
) -> Optional[dict]:
    """Map raw market probs to the correct outcome keys for a prop type."""
    if bet_type == "over_under":
        for k, v in probs.items():
            if "over" in k or "yes" in k or "more" in k:
                return {"over": round(float(v), 4), "under": round(1 - float(v), 4)}
        return None

    if bet_type == "btts":
        for k, v in probs.items():
            if "yes" in k or "both" in k:
                return {"yes": round(float(v), 4), "no": round(1 - float(v), 4)}

    if bet_type in ("player_scorer", "player_foot", "player_header"):
        for k, v in probs.items():
            if "yes" in k or "score" in k or "goal" in k:
                return {"scores": round(float(v), 4), "no_goal": round(1 - float(v), 4)}

    if bet_type == "goes_distance":
        for k, v in probs.items():
            if "yes" in k or "distance" in k or "full" in k:
                return {"distance": round(float(v), 4), "finish": round(1 - float(v), 4)}

    return None
