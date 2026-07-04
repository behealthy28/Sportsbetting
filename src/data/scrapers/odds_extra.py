"""
Additional odds sources beyond Polymarket/Kalshi.

Pinnacle — most efficient sportsbook market, public guest API (no key needed).
OddsPortal — aggregates multiple bookmakers, HTML scraping.
"""
import requests
import re
from bs4 import BeautifulSoup
from src.data import cache

HEADERS_JSON = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept": "application/json",
    "X-API-Key": "",
}
HEADERS_HTML = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# ── Pinnacle public guest API ─────────────────────────────────────────────────
# No auth required for reading live odds
PINNACLE_BASE = "https://guest.api.arcadia.pinnacle.com/0.1"

PINNACLE_SPORTS = {
    "football": 29,   # soccer
    "soccer":   29,
    "tennis":   33,
    "ufc":      23,   # MMA
    "mma":      23,
    "boxing":   10,
    "cricket":  72,
}


def _pinnacle_get(path: str) -> dict | list:
    try:
        resp = requests.get(
            f"{PINNACLE_BASE}{path}",
            headers={**HEADERS_JSON, "Accept": "application/json"},
            timeout=12,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def _pinnacle_leagues(sport_id: int) -> list:
    """Fetch all active leagues for a sport from Pinnacle."""
    cached = cache.get("pinnacle_leagues", {"sport": sport_id})
    if cached:
        return cached
    data = _pinnacle_get(f"/sports/{sport_id}/leagues?brandId=0")
    leagues = data if isinstance(data, list) else []
    cache.set("pinnacle_leagues", {"sport": sport_id}, leagues, ttl_seconds=3600 * 6)
    return leagues


def _pinnacle_events(league_id: int) -> list:
    """Fetch events for a league (cached 5 min — this is called per league in a
    scan loop, so an uncached version means dozens of network calls per predict)."""
    cached = cache.get("pinnacle_events", {"lid": league_id})
    if cached is not None:
        return cached
    data = _pinnacle_get(f"/leagues/{league_id}/matchups")
    out = data if isinstance(data, list) else []
    cache.set("pinnacle_events", {"lid": league_id}, out, ttl_seconds=300)
    return out


def _pinnacle_odds(matchup_id: int) -> dict:
    data = _pinnacle_get(f"/matchups/{matchup_id}/markets/straight")
    return data if isinstance(data, dict) else {}


def get_pinnacle_odds(team1: str, team2: str, sport: str = "football") -> dict | None:
    """
    Search Pinnacle for a match and return vig-removed probabilities.
    Returns {home_win, draw, away_win} or {p1_win, p2_win} or None.
    """
    cache_key = {"t1": team1.lower(), "t2": team2.lower(), "sport": sport}
    cached = cache.get("pinnacle_odds", cache_key)
    if cached is not None:
        return cached or None

    sport_id = PINNACLE_SPORTS.get(sport, 29)
    leagues = _pinnacle_leagues(sport_id)
    if not leagues:
        cache.set("pinnacle_odds", cache_key, {}, ttl_seconds=300)
        return None

    t1_words = set(team1.lower().split())
    t2_words = set(team2.lower().split())

    for league in leagues[:8]:  # scan only the top active leagues — bounded work
        league_id = league.get("id")
        if not league_id:
            continue

        events = _pinnacle_events(league_id)
        for event in events:
            participants = event.get("participants", [])
            if len(participants) < 2:
                continue

            names = [p.get("name", "").lower() for p in participants]
            name_str = " ".join(names)

            # Check if both teams appear
            t1_match = any(w in name_str for w in t1_words if len(w) > 3)
            t2_match = any(w in name_str for w in t2_words if len(w) > 3)

            if t1_match and t2_match:
                matchup_id = event.get("id")
                odds_data = _pinnacle_odds(matchup_id)
                result = _parse_pinnacle_odds(odds_data, team1, team2, sport)
                if result:
                    cache.set("pinnacle_odds", cache_key, result, ttl_seconds=1800)
                    return result

    cache.set("pinnacle_odds", cache_key, {}, ttl_seconds=300)
    return None


def _parse_pinnacle_odds(data: dict, team1: str, team2: str, sport: str) -> dict | None:
    """Parse Pinnacle moneyline/1X2 markets into devigged probabilities."""
    markets = data.get("markets", [])
    if not markets:
        # Try direct list format
        markets = data if isinstance(data, list) else []

    for market in markets:
        market_type = market.get("type", "")
        if market_type not in ("moneyline", "1x2", "match_winner", "MONEYLINE"):
            continue

        prices = market.get("prices", [])
        if not prices:
            continue

        home_price = away_price = draw_price = None
        for p in prices:
            designation = p.get("designation", p.get("side", "")).lower()
            price = p.get("price")
            if price is None:
                continue
            # Convert American odds to probability
            if isinstance(price, (int, float)):
                if price > 0:
                    prob = 100 / (price + 100)
                else:
                    prob = abs(price) / (abs(price) + 100)
            else:
                continue

            if designation in ("home", "1", "team1"):
                home_price = prob
            elif designation in ("away", "2", "team2"):
                away_price = prob
            elif designation in ("draw", "x", "tie"):
                draw_price = prob

        if home_price and away_price:
            # Remove vig (overround)
            total = home_price + away_price + (draw_price or 0)
            result = {
                "home_win": round(home_price / total, 4),
                "away_win": round(away_price / total, 4),
            }
            if draw_price:
                result["draw"] = round(draw_price / total, 4)
            result["source"] = "Pinnacle"
            return result

    return None


# ── OddsPortal HTML scraping (backup) ────────────────────────────────────────

ODDSPORTAL_BASE = "https://www.oddsportal.com"
SPORT_PATHS = {
    "football": "soccer", "soccer": "soccer",
    "tennis": "tennis", "ufc": "mma", "boxing": "boxing",
    "cricket": "cricket",
}


def get_oddsportal_odds(team1: str, team2: str, sport: str = "football") -> dict | None:
    """Scrape OddsPortal for the best available odds on a match."""
    cache_key = {"t1": team1.lower(), "t2": team2.lower(), "sport": sport}
    cached = cache.get("oddsportal", cache_key)
    if cached is not None:
        return cached or None

    sport_path = SPORT_PATHS.get(sport, "soccer")
    search_url = f"{ODDSPORTAL_BASE}/search/results/{team1.replace(' ', '+')}/"

    try:
        resp = requests.get(search_url, headers=HEADERS_HTML, timeout=12)
        if resp.status_code != 200:
            cache.set("oddsportal", cache_key, {}, ttl_seconds=300)
            return None

        soup = BeautifulSoup(resp.text, "lxml")
        t1 = team1.lower()
        t2 = team2.lower()

        # Find match link
        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = link.get_text(separator=" ").lower()
            if (sport_path in href
                    and any(w in text for w in t1.split() if len(w) > 3)
                    and any(w in text for w in t2.split() if len(w) > 3)):
                match_url = ODDSPORTAL_BASE + href if href.startswith("/") else href
                result = _scrape_oddsportal_match(match_url)
                if result:
                    cache.set("oddsportal", cache_key, result, ttl_seconds=1800)
                    return result

    except Exception:
        pass

    cache.set("oddsportal", cache_key, {}, ttl_seconds=300)
    return None


def _scrape_oddsportal_match(url: str) -> dict | None:
    """Scrape odds from an OddsPortal match page."""
    try:
        resp = requests.get(url, headers=HEADERS_HTML, timeout=12)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "lxml")

        # OddsPortal embeds odds in JSON within script tags
        for script in soup.find_all("script"):
            text = script.string or ""
            if "odds" in text and "homeOdds" in text:
                home_m = re.search(r'"homeOdds":\s*([\d.]+)', text)
                draw_m = re.search(r'"drawOdds":\s*([\d.]+)', text)
                away_m = re.search(r'"awayOdds":\s*([\d.]+)', text)
                if home_m and away_m:
                    home_dec = float(home_m.group(1))
                    away_dec = float(away_m.group(1))
                    draw_dec = float(draw_m.group(1)) if draw_m else None

                    if home_dec <= 1 or away_dec <= 1:
                        continue

                    home_p = 1 / home_dec
                    away_p = 1 / away_dec
                    draw_p = 1 / draw_dec if draw_dec and draw_dec > 1 else None

                    total = home_p + away_p + (draw_p or 0)
                    result = {
                        "home_win": round(home_p / total, 4),
                        "away_win": round(away_p / total, 4),
                        "source": "OddsPortal",
                    }
                    if draw_p:
                        result["draw"] = round(draw_p / total, 4)
                    return result
    except Exception:
        pass
    return None
