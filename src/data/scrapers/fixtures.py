"""
Live fixtures and upcoming matches.
Primary: ESPN unofficial API (works great from residential IPs).
Fallback: TheSportsDB free API (no key needed, more permissive).
"""
import time
import requests
from concurrent.futures import ThreadPoolExecutor
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
    # Second tiers & cups (more matches, more days with fixtures)
    ("soccer", "eng.2",    "Championship 🏴󠁧󠁢󠁥󠁮󠁧󠁿"),
    ("soccer", "eng.fa",   "FA Cup 🏴󠁧󠁢󠁥󠁮󠁧󠁿"),
    ("soccer", "eng.league_cup", "EFL Cup 🏴󠁧󠁢󠁥󠁮󠁧󠁿"),
    ("soccer", "esp.2",    "La Liga 2 🇪🇸"),
    ("soccer", "ita.2",    "Serie B 🇮🇹"),
    ("soccer", "ger.2",    "2. Bundesliga 🇩🇪"),
    ("soccer", "fra.2",    "Ligue 2 🇫🇷"),
    ("soccer", "uefa.champions_qual", "UCL Qualifying 🏆"),
    ("soccer", "uefa.europa.conf",    "Conference League 🏆"),
    ("soccer", "fifa.cwc",            "Club World Cup 🌍"),
    ("soccer", "fifa.worldq.conmebol","WC Qualifiers S.America 🌍"),
    ("soccer", "fifa.worldq.concacaf","WC Qualifiers N.America 🌍"),
    ("soccer", "fifa.worldq.afc",     "WC Qualifiers Asia 🌍"),
    ("soccer", "fifa.worldq.caf",     "WC Qualifiers Africa 🌍"),
    # More national top flights across every confederation
    ("soccer", "bel.1",    "Belgian Pro League 🇧🇪"),
    ("soccer", "swi.1",    "Swiss Super League 🇨🇭"),
    ("soccer", "aut.1",    "Austrian Bundesliga 🇦🇹"),
    ("soccer", "gre.1",    "Super League Greece 🇬🇷"),
    ("soccer", "rus.1",    "Russian Premier League 🇷🇺"),
    ("soccer", "ukr.1",    "Ukrainian Premier League 🇺🇦"),
    ("soccer", "den.1",    "Danish Superliga 🇩🇰"),
    ("soccer", "nor.1",    "Eliteserien 🇳🇴"),
    ("soccer", "swe.1",    "Allsvenskan 🇸🇪"),
    ("soccer", "pol.1",    "Ekstraklasa 🇵🇱"),
    ("soccer", "cze.1",    "Czech First League 🇨🇿"),
    ("soccer", "rou.1",    "Liga I 🇷🇴"),
    ("soccer", "aus.1",    "A-League 🇦🇺"),
    ("soccer", "chn.1",    "Chinese Super League 🇨🇳"),
    ("soccer", "kor.1",    "K League 1 🇰🇷"),
    ("soccer", "uae.1",    "UAE Pro League 🇦🇪"),
    ("soccer", "chi.1",    "Chile Primera 🇨🇱"),
    ("soccer", "uru.1",    "Uruguay Primera 🇺🇾"),
    ("soccer", "ecu.1",    "Ecuador Serie A 🇪🇨"),
    ("soccer", "per.1",    "Peru Liga 1 🇵🇪"),
    ("soccer", "par.1",    "Paraguay Primera 🇵🇾"),
    ("soccer", "conmebol.libertadores", "Copa Libertadores 🌎"),
    ("soccer", "conmebol.sudamericana", "Copa Sudamericana 🌎"),
    ("soccer", "concacaf.champions",    "CONCACAF Champions 🌎"),
    ("soccer", "caf.champions",         "CAF Champions League 🌍"),
    ("soccer", "afc.champions",         "AFC Champions League 🌏"),
    ("soccer", "usa.nwsl",              "NWSL 🇺🇸"),
    ("soccer", "eng.w.1",               "WSL (Women) 🏴󠁧󠁢󠁥󠁮󠁧󠁿"),
    ("soccer", "fifa.wwc",              "Women's World Cup 🌍"),
]

OTHER_SPORTS = [
    # Tennis
    ("tennis", "atp",       "ATP Tennis 🎾"),
    ("tennis", "wta",       "WTA Tennis 🎾"),
    # Combat
    ("mma",    "ufc",       "UFC / MMA 🥊"),
    # Basketball
    ("basketball", "nba",                     "NBA 🏀"),
    ("basketball", "wnba",                    "WNBA 🏀"),
    ("basketball", "nba-summer-las-vegas",    "NBA Summer League 🏀"),
    ("basketball", "mens-college-basketball", "NCAA Basketball 🏀"),
    ("basketball", "fiba.world",              "FIBA Basketball 🌍🏀"),
    ("basketball", "fiba.olympics",           "Olympic Basketball 🏀"),
    ("basketball", "euroleague",              "EuroLeague 🏀"),
    ("basketball", "nbl",                     "NBL Australia 🇦🇺🏀"),
    # Baseball
    ("baseball", "mlb",            "MLB ⚾"),
    ("baseball", "college-baseball", "NCAA Baseball ⚾"),
    ("baseball", "npb",            "NPB Japan 🇯🇵⚾"),
    # Ice hockey
    ("hockey", "nhl",   "NHL 🏒"),
    ("hockey", "mens-college-hockey", "NCAA Hockey 🏒"),
    # American football
    ("football", "nfl",              "NFL 🏈"),
    ("football", "college-football", "NCAA Football 🏈"),
    ("football", "cfl",              "CFL 🇨🇦🏈"),
    ("football", "xfl",              "UFL 🏈"),
    # Rugby (ESPN league IDs)
    ("rugby", "164205",  "Six Nations 🏉"),
    ("rugby", "289234",  "Rugby World Cup 🏉"),
    ("rugby", "270559",  "United Rugby Championship 🏉"),
    ("rugby", "267979",  "Premiership Rugby 🏉"),
    # Cricket (ESPN coverage patchy — SportsDB fills the gaps below)
    ("cricket", "8048", "Cricket 🏏"),
]

# TheSportsDB sport labels mapping — covers sports ESPN doesn't (darts, table
# tennis, badminton, snooker, rugby, handball, motorsport) and acts as a
# fallback for the rest. Always merged with ESPN, then de-duplicated.
SPORTSDB_SPORTS = [
    ("Soccer",            "Football ⚽"),
    ("Basketball",        "Basketball 🏀"),
    ("Ice Hockey",        "Ice Hockey 🏒"),
    ("Baseball",          "Baseball ⚾"),
    ("American Football", "American Football 🏈"),
    ("Tennis",            "Tennis 🎾"),
    ("MMA",               "MMA/UFC 🥊"),
    ("Boxing",            "Boxing 🥊"),
    ("Cricket",           "Cricket 🏏"),
    ("Rugby",             "Rugby 🏉"),
    ("Darts",             "Darts 🎯"),
    ("Snooker",           "Snooker 🎱"),
    ("Table Tennis",      "Table Tennis 🏓"),
    ("Badminton",         "Badminton 🏸"),
    ("Handball",          "Handball 🤾"),
    ("Volleyball",        "Volleyball 🏐"),
]

# ESPN sport slug -> internal predict-sport key (routes to a prediction handler).
# Note ESPN uses "soccer" for football/soccer and "football" for American football.
_ESPN_SPORT_TO_KEY = {
    "soccer": "football",
    "tennis": "tennis",
    "mma": "ufc",
    "basketball": "basketball",
    "baseball": "baseball",
    "hockey": "hockey",
    "football": "americanfootball",
    "cricket": "cricket",
}

# TheSportsDB API sport label -> internal predict-sport key.
_SPORTSDB_SPORT_TO_KEY = {
    "Soccer": "football",
    "Basketball": "basketball",
    "Ice Hockey": "hockey",
    "Baseball": "baseball",
    "American Football": "americanfootball",
    "Tennis": "tennis",
    "MMA": "ufc",
    "Boxing": "boxing",
    "Cricket": "cricket",
    "Rugby": "rugby",
    "Motorsport": "motorsport",
    "Darts": "darts",
    "Snooker": "snooker",
    "Table Tennis": "table tennis",
    "Badminton": "badminton",
    "Handball": "handball",
    "Volleyball": "volleyball",
}


def _get_espn(url: str, params: dict = None) -> dict:
    try:
        resp = requests.get(url, params=params, headers=ESPN_HEADERS, timeout=12)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def _get_sportsdb(url: str, params: dict = None) -> dict:
    # The free shared key is rate-limited; a per-(sport,date) cache means a full
    # window only costs a handful of live calls, and 429s get retried with
    # backoff instead of silently dropping a whole sport.
    ck = {"u": url, "p": params or {}}
    cached = cache.get("sportsdb_raw", ck)
    if cached is not None:
        return cached

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=SPORTSDB_HEADERS, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                cache.set("sportsdb_raw", ck, data, ttl_seconds=3600)  # 1h
                return data
            if resp.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
        except Exception:
            time.sleep(0.5)
    return {}


def _winpct_from_competitor(competitor: dict):
    """Extract season win% from an ESPN competitor's overall record ('50-32').

    Returns a float 0..1, or None when no usable record is present.
    """
    records = competitor.get("records") or []
    summary = ""
    for rec in records:
        if rec.get("type") in ("total", "overall") or rec.get("name") in ("overall", "All Splits"):
            summary = rec.get("summary", "")
            break
    if not summary and records:
        summary = records[0].get("summary", "")
    if not summary:
        return None
    parts = summary.split("-")
    try:
        nums = [int(p) for p in parts if p.strip().isdigit()]
    except ValueError:
        return None
    if len(nums) < 2:
        return None
    wins, losses = nums[0], nums[1]
    ties = nums[2] if len(nums) > 2 else 0
    total = wins + losses + ties
    if total <= 0:
        return None
    return wins / total


def _parse_espn_events(data: dict, league_label: str, sport_key: str = "") -> list:
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
            "sport": sport_key,
            "name": event.get("name", ""),
            "date": event.get("date", ""),
            "home": home.get("team", {}).get("displayName", ""),
            "away": away.get("team", {}).get("displayName", ""),
            "home_score": home.get("score", ""),
            "away_score": away.get("score", ""),
            "home_winpct": _winpct_from_competitor(home),
            "away_winpct": _winpct_from_competitor(away),
            "status": status,
            "clock": clock,
            "period": period,
            "venue": comp.get("venue", {}).get("fullName", ""),
            "source": "espn",
        })
    return events


def _parse_sportsdb_events(data: dict, league_label: str, sport_key: str = "") -> list:
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
            "sport": sport_key,
            "name": f"{home} vs {away}",
            "date": iso_date,
            "home": home,
            "away": away,
            "home_score": str(home_score) if home_score != "" else "",
            "away_score": str(away_score) if away_score != "" else "",
            "home_winpct": None,
            "away_winpct": None,
            "status": status,
            "clock": "",
            "period": 0,
            "venue": event.get("strVenue", ""),
            "source": "sportsdb",
        })
    return events


# Max concurrent HTTP requests across all fixture sources. ESPN/SportsDB are
# fine with this from a single client; keeps a ~50-endpoint sweep to a few seconds.
_FETCH_WORKERS = 24


def _fetch_espn(days_ahead: int) -> list:
    today = datetime.utcnow()
    leagues = FOOTBALL_LEAGUES + OTHER_SPORTS

    def _one(sport, league_id, label, day_offset):
        date_str = (today + timedelta(days=day_offset)).strftime("%Y%m%d")
        url = f"{ESPN_BASE}/{sport}/{league_id}/scoreboard"
        data = _get_espn(url, params={"dates": date_str, "limit": 30})
        return _parse_espn_events(data, label, _ESPN_SPORT_TO_KEY.get(sport, sport))

    tasks = [
        (sport, lid, label, d)
        for sport, lid, label in leagues
        for d in range(days_ahead + 1)
    ]
    all_events = []
    with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
        for events in pool.map(lambda t: _one(*t), tasks):
            all_events.extend(events)
    return all_events


def _fetch_sportsdb(days_ahead: int) -> list:
    """TheSportsDB free API — no key needed, permissive with server requests."""
    today = datetime.utcnow()

    def _one(sport_label, display_label, day_offset):
        date_str = (today + timedelta(days=day_offset)).strftime("%Y-%m-%d")
        url = "https://www.thesportsdb.com/api/v1/json/3/eventsday.php"
        data = _get_sportsdb(url, params={"d": date_str, "s": sport_label})
        return _parse_sportsdb_events(data, display_label, _SPORTSDB_SPORT_TO_KEY.get(sport_label, ""))

    tasks = [
        (sport_label, display_label, d)
        for sport_label, display_label in SPORTSDB_SPORTS
        for d in range(days_ahead + 1)
    ]
    all_events = []
    # Low concurrency on purpose — the free shared key 429s under bursty load.
    with ThreadPoolExecutor(max_workers=3) as pool:
        for events in pool.map(lambda t: _one(*t), tasks):
            all_events.extend(events)
    return all_events


def get_todays_fixtures(sports: list = None, days_ahead: int = 1) -> list:
    """
    Fetch today's + upcoming fixtures across all major sports.
    Returns list of event dicts sorted by date.
    """
    cache_key = {"days": days_ahead, "sports": str(sports)}
    cached = cache.get("fixtures_today", cache_key)
    if cached:
        return cached

    # Fetch both sources concurrently: ESPN (rich match/live data) + TheSportsDB
    # (covers darts, table tennis, badminton, snooker, rugby, motorsport, etc.).
    # De-duplication below removes overlaps.
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_espn = pool.submit(_fetch_espn, days_ahead)
        f_sdb = pool.submit(_fetch_sportsdb, days_ahead)
        all_events = f_espn.result()
        sdb_events = f_sdb.result()

    # Prefer ESPN entries (they carry live scores/clock); append SportsDB after so
    # the dedup keeps the richer ESPN record when a match exists in both.
    all_events.extend(sdb_events)

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
