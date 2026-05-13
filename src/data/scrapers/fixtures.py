"""
Live fixtures and upcoming matches.
Primary: ESPN unofficial API (works great from residential IPs).
Fallback: TheSportsDB free API (no key needed, more permissive).
"""
import requests
from datetime import datetime, timedelta
from src.data import cache

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

# Full browser headers — ESPN requires these from non-browser clients
ESPN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.espn.com/soccer/scoreboard/",
    "Origin": "https://www.espn.com",
    "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}

SPORTSDB_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

# All major football leagues available on ESPN
FOOTBALL_LEAGUES = [
    ("soccer", "eng.1",    "Premier League 🏴󠁧󠁢󠁥󠁮󠁧󠁿"),
    ("soccer", "esp.1",    "La Liga 🇪🇸"),
    ("soccer", "ger.1",    "Bundesliga 🇩🇪"),
    ("soccer", "ita.1",    "Serie A 🇮🇹"),
    ("soccer", "fra.1",    "Ligue 1 🇫🇷"),
    ("soccer", "uefa.champions", "Champions League 🏆"),
    ("soccer", "uefa.europa",    "Europa League 🏆"),
    ("soccer", "uefa.nations",   "Nations League 🌍"),
    ("soccer", "fifa.world",     "World Cup 🌍"),
    ("soccer", "fifa.worldq.uefa", "WC Qualifiers Europe 🌍"),
    ("soccer", "usa.1",    "MLS 🇺🇸"),
    ("soccer", "ned.1",    "Eredivisie 🇳🇱"),
    ("soccer", "por.1",    "Primeira Liga 🇵🇹"),
    ("soccer", "tur.1",    "Süper Lig 🇹🇷"),
    ("soccer", "mex.1",    "Liga MX 🇲🇽"),
    ("soccer", "bra.1",    "Brasileirão 🇧🇷"),
    ("soccer", "arg.1",    "Liga Profesional 🇦🇷"),
    ("soccer", "sco.1",    "Scottish Prem 🏴󠁧󠁢󠁳󠁣󠁴󠁿"),
    ("soccer", "col.1",    "Liga BetPlay 🇨🇴"),
    ("soccer", "jpn.1",    "J-League 🇯🇵"),
    ("soccer", "sau.1",    "Saudi Pro League 🇸🇦"),
]

OTHER_SPORTS = [
    ("tennis", "atp",       "ATP Tennis 🎾"),
    ("tennis", "wta",       "WTA Tennis 🎾"),
    ("mma",    "ufc",       "UFC 🥊"),
    ("boxing", "boxing",    "Boxing 🥊"),
    ("cricket","icc.world", "Cricket 🏏"),
]

# TheSportsDB sport labels mapping
SPORTSDB_SPORTS = [
    ("Soccer",  "Football ⚽"),
    ("Tennis",  "Tennis 🎾"),
    ("MMA",     "MMA/UFC 🥊"),
    ("Boxing",  "Boxing 🥊"),
    ("Cricket", "Cricket 🏏"),
]


def _get_espn(url: str, params: dict = None) -> dict:
    try:
        resp = requests.get(url, params=params, headers=ESPN_HEADERS, timeout=12)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def _get_sportsdb(url: str, params: dict = None) -> dict:
    try:
        resp = requests.get(url, params=params, headers=SPORTSDB_HEADERS, timeout=12)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def _parse_espn_events(data: dict, league_label: str) -> list:
    events = []
    for event in data.get("events", []):
        comps = event.get("competitions", [{}])
        comp = comps[0] if comps else {}
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

        status_obj = event.get("status", {})
        status = status_obj.get("type", {}).get("description", "Scheduled")
        clock = status_obj.get("displayClock", "")
        period = status_obj.get("period", 0)

        events.append({
            "league": league_label,
            "name": event.get("name", ""),
            "date": event.get("date", ""),
            "home": home.get("team", {}).get("displayName", ""),
            "away": away.get("team", {}).get("displayName", ""),
            "home_score": home.get("score", ""),
            "away_score": away.get("score", ""),
            "status": status,
            "clock": clock,
            "period": period,
            "venue": comp.get("venue", {}).get("fullName", ""),
            "source": "espn",
        })
    return events


def _parse_sportsdb_events(data: dict, league_label: str) -> list:
    events = []
    for event in (data.get("events") or []):
        home = event.get("strHomeTeam", "")
        away = event.get("strAwayTeam", "")
        if not home or not away:
            continue

        date_str = event.get("dateEvent", "")
        time_str = event.get("strTime", "00:00:00") or "00:00:00"
        # Build ISO date
        iso_date = f"{date_str}T{time_str}Z" if date_str else ""

        home_score = event.get("intHomeScore") or ""
        away_score = event.get("intAwayScore") or ""
        status = event.get("strStatus", "") or "Scheduled"
        if status in ("", "NS"):
            status = "Scheduled"
        elif status == "FT":
            status = "Final"
        elif status == "HT":
            status = "Halftime"

        events.append({
            "league": event.get("strLeague", league_label),
            "name": f"{home} vs {away}",
            "date": iso_date,
            "home": home,
            "away": away,
            "home_score": str(home_score) if home_score != "" else "",
            "away_score": str(away_score) if away_score != "" else "",
            "status": status,
            "clock": "",
            "period": 0,
            "venue": event.get("strVenue", ""),
            "source": "sportsdb",
        })
    return events


def _fetch_espn_football(days_ahead: int) -> list:
    """Fetch football/soccer fixtures from ESPN (reliable coverage)."""
    all_events = []
    today = datetime.utcnow()
    for sport, league_id, label in FOOTBALL_LEAGUES:
        for day_offset in range(days_ahead + 1):
            date_str = (today + timedelta(days=day_offset)).strftime("%Y%m%d")
            url = f"{ESPN_BASE}/{sport}/{league_id}/scoreboard"
            data = _get_espn(url, params={"dates": date_str, "limit": 30})
            all_events.extend(_parse_espn_events(data, label))
    return all_events


def _fetch_espn(days_ahead: int) -> list:
    """Fetch all sports from ESPN. Used as last-resort fallback only."""
    all_events = []
    today = datetime.utcnow()
    leagues = FOOTBALL_LEAGUES + OTHER_SPORTS

    for sport, league_id, label in leagues:
        for day_offset in range(days_ahead + 1):
            date_str = (today + timedelta(days=day_offset)).strftime("%Y%m%d")
            url = f"{ESPN_BASE}/{sport}/{league_id}/scoreboard"
            data = _get_espn(url, params={"dates": date_str, "limit": 30})
            all_events.extend(_parse_espn_events(data, label))

    return all_events


def _fetch_sportsdb(days_ahead: int) -> list:
    """TheSportsDB free API — no key needed, permissive with server requests."""
    all_events = []
    today = datetime.utcnow()

    for sport_label, display_label in SPORTSDB_SPORTS:
        for day_offset in range(days_ahead + 1):
            date_str = (today + timedelta(days=day_offset)).strftime("%Y-%m-%d")
            url = "https://www.thesportsdb.com/api/v1/json/3/eventsday.php"
            data = _get_sportsdb(url, params={"d": date_str, "s": sport_label})
            all_events.extend(_parse_sportsdb_events(data, display_label))

    return all_events


def get_todays_fixtures(sports: list = None, days_ahead: int = 1) -> list:
    """
    Fetch today's + upcoming fixtures across all major sports.
    Football from ESPN (reliable). Other sports always from TheSportsDB —
    ESPN silently returns empty for tennis/UFC/boxing/cricket so we don't
    rely on it as a fallback for those.
    Returns list of event dicts sorted by date.
    """
    cache_key = {"days": days_ahead, "sports": str(sports)}
    cached = cache.get("fixtures_today", cache_key)
    if cached:
        return cached

    # Football from ESPN (reliable, good coverage)
    all_events = _fetch_espn_football(days_ahead)

    # Other sports always from TheSportsDB — ESPN silently fails for these
    all_events.extend(_fetch_sportsdb(days_ahead))

    # If both returned empty, try ESPN's other endpoints as last resort
    if not all_events:
        all_events = _fetch_espn(days_ahead)

    # Sort by date
    all_events.sort(key=lambda e: e.get("date", ""))

    # Deduplicate
    seen = set()
    unique = []
    for e in all_events:
        key = (e["home"].lower(), e["away"].lower(), e.get("date", "")[:10])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    if unique:
        cache.set("fixtures_today", cache_key, unique, ttl_seconds=900)  # 15-min cache
    return unique


def search_fixture(team1: str, team2: str, days_ahead: int = 7) -> dict | None:
    """Find a specific upcoming fixture between two teams."""
    all_fixtures = get_todays_fixtures(days_ahead=days_ahead)
    t1 = team1.lower()
    t2 = team2.lower()

    for f in all_fixtures:
        h = f["home"].lower()
        a = f["away"].lower()
        if (t1 in h or any(w in h for w in t1.split())
                or t1 in a or any(w in a for w in t1.split())):
            if (t2 in h or any(w in h for w in t2.split())
                    or t2 in a or any(w in a for w in t2.split())):
                return f
    return None


def get_live_scores() -> list:
    """Return only in-progress matches right now."""
    all_events = get_todays_fixtures(days_ahead=0)
    return [e for e in all_events if e["status"] in ("In Progress", "Halftime", "Final")]
