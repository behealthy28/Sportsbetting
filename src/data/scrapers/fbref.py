"""
Football data — FBRef, Understat, ESPN, ClubElo, EloRatings.
Covers ANY team in the world with live data, not just seeded ones.
"""
import requests
import json
import re
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from src.data import cache

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"

# Known national teams that need form data sourced from ESPN / FIFA results
# ELOs now come from elo_db.py; these averages are used as scoring priors
INTL_SCORING_PRIORS = {
    "argentina":   {"avg_goals": 2.1, "avg_conceded": 0.6},
    "france":      {"avg_goals": 2.2, "avg_conceded": 0.9},
    "brazil":      {"avg_goals": 2.0, "avg_conceded": 0.7},
    "england":     {"avg_goals": 1.8, "avg_conceded": 0.8},
    "spain":       {"avg_goals": 1.9, "avg_conceded": 0.7},
    "portugal":    {"avg_goals": 2.1, "avg_conceded": 0.8},
    "netherlands": {"avg_goals": 1.9, "avg_conceded": 1.0},
    "germany":     {"avg_goals": 1.7, "avg_conceded": 1.1},
    "italy":       {"avg_goals": 1.5, "avg_conceded": 0.8},
    "croatia":     {"avg_goals": 1.6, "avg_conceded": 0.9},
    "belgium":     {"avg_goals": 1.8, "avg_conceded": 0.9},
    "morocco":     {"avg_goals": 1.4, "avg_conceded": 0.7},
    "denmark":     {"avg_goals": 1.7, "avg_conceded": 0.9},
    "switzerland": {"avg_goals": 1.6, "avg_conceded": 0.9},
    "uruguay":     {"avg_goals": 1.5, "avg_conceded": 0.8},
    "colombia":    {"avg_goals": 1.6, "avg_conceded": 0.9},
    "mexico":      {"avg_goals": 1.5, "avg_conceded": 1.0},
    "usa":         {"avg_goals": 1.4, "avg_conceded": 1.0},
    "wales":       {"avg_goals": 1.3, "avg_conceded": 1.0},
    "scotland":    {"avg_goals": 1.4, "avg_conceded": 1.1},
    "turkey":      {"avg_goals": 1.5, "avg_conceded": 1.1},
    "poland":      {"avg_goals": 1.4, "avg_conceded": 1.0},
    "japan":       {"avg_goals": 1.6, "avg_conceded": 0.9},
    "south korea": {"avg_goals": 1.5, "avg_conceded": 1.0},
    "senegal":     {"avg_goals": 1.3, "avg_conceded": 0.8},
    "nigeria":     {"avg_goals": 1.3, "avg_conceded": 1.0},
    "egypt":       {"avg_goals": 1.3, "avg_conceded": 0.9},
    "australia":   {"avg_goals": 1.3, "avg_conceded": 1.1},
    "canada":      {"avg_goals": 1.3, "avg_conceded": 1.0},
    "ukraine":     {"avg_goals": 1.5, "avg_conceded": 1.0},
    "austria":     {"avg_goals": 1.6, "avg_conceded": 0.9},
    "sweden":      {"avg_goals": 1.5, "avg_conceded": 0.9},
    "norway":      {"avg_goals": 1.5, "avg_conceded": 1.0},
    "republic of ireland": {"avg_goals": 1.2, "avg_conceded": 1.1},
    "czech republic": {"avg_goals": 1.4, "avg_conceded": 1.0},
    "serbia":      {"avg_goals": 1.5, "avg_conceded": 0.9},
    "romania":     {"avg_goals": 1.3, "avg_conceded": 1.1},
    "hungary":     {"avg_goals": 1.3, "avg_conceded": 1.1},
    "albania":     {"avg_goals": 1.2, "avg_conceded": 1.1},
    "georgia":     {"avg_goals": 1.2, "avg_conceded": 1.2},
    "ivory coast": {"avg_goals": 1.4, "avg_conceded": 0.9},
    "ghana":       {"avg_goals": 1.3, "avg_conceded": 1.1},
    "cameroon":    {"avg_goals": 1.2, "avg_conceded": 1.0},
    "saudi arabia": {"avg_goals": 1.3, "avg_conceded": 1.0},
    "iran":        {"avg_goals": 1.3, "avg_conceded": 0.9},
    "ecuador":     {"avg_goals": 1.3, "avg_conceded": 0.9},
    "chile":       {"avg_goals": 1.4, "avg_conceded": 1.0},
    "peru":        {"avg_goals": 1.2, "avg_conceded": 0.9},
    "venezuela":   {"avg_goals": 1.2, "avg_conceded": 1.1},
    "bolivia":     {"avg_goals": 1.0, "avg_conceded": 1.4},
    "costa rica":  {"avg_goals": 1.1, "avg_conceded": 1.0},
    "panama":      {"avg_goals": 1.0, "avg_conceded": 1.1},
    "iceland":     {"avg_goals": 1.3, "avg_conceded": 1.0},
    "north macedonia": {"avg_goals": 1.1, "avg_conceded": 1.2},
    "slovenia":    {"avg_goals": 1.2, "avg_conceded": 0.9},
    "slovakia":    {"avg_goals": 1.2, "avg_conceded": 1.0},
    "finland":     {"avg_goals": 1.2, "avg_conceded": 1.1},
    "greece":      {"avg_goals": 1.1, "avg_conceded": 1.0},
    "bosnia":      {"avg_goals": 1.3, "avg_conceded": 1.1},
}

# ESPN league IDs for live data lookups
ESPN_LEAGUES = [
    "eng.1", "esp.1", "ger.1", "ita.1", "fra.1",
    "por.1", "ned.1", "tur.1", "sco.1", "usa.1",
    "bra.1", "arg.1", "mex.1", "sau.1", "jpn.1",
    "uefa.champions", "uefa.europa", "uefa.nations",
    "fifa.worldq.uefa", "fifa.world",
]


def get_team_data(team_name: str) -> dict:
    """
    Get football team stats for ANY team.
    Pipeline:
      1. Cache hit
      2. Understat (live xG — top 5 European leagues)
      3. ESPN stats API (any league worldwide)
      4. International scoring priors + EloRatings/ClubElo for ELO
      5. Generic default
    """
    name_lower = team_name.lower().strip()
    cached = cache.get("fbref_team", {"team": name_lower})
    if cached:
        return cached

    # Try Understat first (most accurate xG — club teams in top 5 leagues)
    understat_data = get_understat_team(team_name)
    if understat_data:
        # Enrich with real ELO from ClubElo
        from src.data.scrapers.elo_db import get_elo
        understat_data["elo"] = get_elo(team_name, is_national=False)
        understat_data["data_sources"] = ["Understat (live xG)", "ClubElo"]
        cache.set("fbref_team", {"team": name_lower}, understat_data, ttl_seconds=3600 * 6)
        return understat_data

    # Try ESPN for any worldwide team
    espn_data = _get_espn_team_data(team_name)
    if espn_data:
        cache.set("fbref_team", {"team": name_lower}, espn_data, ttl_seconds=3600 * 6)
        return espn_data

    # International team — use scoring priors + live ELO
    priors = INTL_SCORING_PRIORS.get(name_lower)
    if priors:
        from src.data.scrapers.elo_db import get_national_elo
        elo = get_national_elo(team_name) or 1800
        result = {
            "team": team_name,
            "elo": elo,
            "avg_goals": priors["avg_goals"],
            "avg_conceded": priors["avg_conceded"],
            "form": 0.60,  # will be overridden by ESPN if available
            "is_national": True,
            "data_sources": ["FIFA/EloRatings (national ELO)", "Historical scoring priors"],
        }
        result.update(_get_espn_intl_form(team_name))
        cache.set("fbref_team", {"team": name_lower}, result, ttl_seconds=3600 * 6)
        return result

    # Unknown team — try ELO databases and use scoring average.
    # Cache this fallback too: without it, a team that misses Understat/ESPN/priors
    # re-runs the full (slow) scrape pipeline on every prediction.
    from src.data.scrapers.elo_db import get_elo
    elo = get_elo(team_name)
    result = {
        "team": team_name,
        "elo": elo,
        "avg_goals": _elo_to_goals(elo),
        "avg_conceded": _elo_to_goals(3400 - elo),  # mirror
        "form": _elo_to_form(elo),
        "is_national": False,
        "data_sources": ["ClubElo (ELO)", "ELO-derived scoring estimate"],
    }
    cache.set("fbref_team", {"team": name_lower}, result, ttl_seconds=3600 * 6)
    return result


def _elo_to_goals(elo: float) -> float:
    """Estimate avg goals from ELO rating (rough linear mapping)."""
    # 2000 ELO ≈ 2.2 goals, 1500 ELO ≈ 1.0 goals
    return round(max(0.6, min(2.8, (elo - 1500) / 500 * 1.2 + 1.0)), 2)


def _elo_to_form(elo: float) -> float:
    return round(max(0.35, min(0.85, (elo - 1500) / 600 * 0.5 + 0.5)), 3)


def _get_espn_team_data(team_name: str) -> dict:
    """Search ESPN across all leagues to find team stats.

    The per-league team-list lookups are independent, so they run concurrently
    (a cold 21-league sequential scan was a major slice of prediction latency).
    """
    name_lower = team_name.lower()

    def _find_in_league(league: str):
        try:
            resp = requests.get(f"{ESPN_BASE}/{league}/teams", headers=HEADERS, timeout=(3, 6))
            if resp.status_code != 200:
                return None
            data = resp.json()
            sports = data.get("sports", [{}])
            leagues_data = sports[0].get("leagues", [{}]) if sports else [{}]
            teams = leagues_data[0].get("teams", []) if leagues_data else []
            for team_entry in teams:
                t = team_entry.get("team", {})
                display = t.get("displayName", "").lower()
                short = t.get("shortDisplayName", "").lower()
                abbr = t.get("abbreviation", "").lower()
                if (name_lower in display or display in name_lower
                        or name_lower in short or abbr == name_lower[:3]):
                    return (league, t.get("id"))
        except Exception:
            return None
        return None

    with ThreadPoolExecutor(max_workers=10) as pool:
        matches = [m for m in pool.map(_find_in_league, ESPN_LEAGUES) if m]

    for league, team_id in matches:
        stats = _get_espn_team_stats(league, team_id, team_name)
        if stats:
            from src.data.scrapers.elo_db import get_elo
            stats["elo"] = get_elo(team_name)
            return stats
    return {}


def _get_espn_team_stats(league: str, team_id: str, team_name: str) -> dict:
    """Fetch statistics for a team from ESPN."""
    try:
        url = f"{ESPN_BASE}/{league}/teams/{team_id}/statistics"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return {}

        data = resp.json()
        splits = data.get("splits", {}).get("categories", [])

        goals_for = goals_against = None
        for cat in splits:
            for stat in cat.get("stats", []):
                name = stat.get("name", "").lower()
                val = stat.get("value")
                if val is None:
                    continue
                if "goalsper" in name or name in ("goalpergame",):
                    if goals_for is None:
                        goals_for = float(val)
                elif "goalsconceded" in name or "goalsallowed" in name or "goalsagainst" in name:
                    if goals_against is None:
                        goals_against = float(val)

        if goals_for is None:
            return {}

        # Get recent form from scoreboard
        form = _get_espn_team_form(league, team_id)

        return {
            "team": team_name,
            "avg_goals": goals_for,
            "avg_conceded": goals_against or 1.2,
            "form": form,
            "is_national": False,
            "data_sources": [f"ESPN ({league})"],
        }
    except Exception:
        return {}


def _get_espn_team_form(league: str, team_id: str) -> float:
    """Estimate form from last 5 ESPN results."""
    try:
        url = f"{ESPN_BASE}/{league}/teams/{team_id}/schedule"
        resp = requests.get(url, headers=HEADERS, timeout=8)
        if resp.status_code != 200:
            return 0.5

        events = resp.json().get("events", [])
        results = []
        for ev in reversed(events):
            comps = ev.get("competitions", [{}])
            comp = comps[0] if comps else {}
            competitors = comp.get("competitors", [])
            for c in competitors:
                if str(c.get("id")) == str(team_id):
                    outcome = c.get("winner")
                    score = c.get("score", "0")
                    opp_scores = [x.get("score", "0") for x in competitors if str(x.get("id")) != str(team_id)]
                    if outcome is True:
                        results.append("w")
                    elif outcome is False:
                        results.append("l")
                    else:
                        results.append("d")
            if len(results) >= 10:
                break

        if not results:
            return 0.5
        wins = results.count("w")
        draws = results.count("d")
        return round((wins + 0.5 * draws) / len(results), 3)
    except Exception:
        return 0.5


def _get_espn_intl_form(team_name: str) -> dict:
    """Fetch recent form for a national team from ESPN."""
    try:
        # Try FIFA World Cup qualifiers and Nations League feeds
        for league in ["fifa.worldq.uefa", "fifa.worldq.conmebol", "uefa.nations",
                       "fifa.worldq.concacaf", "fifa.worldq.caf", "fifa.world"]:
            url = f"{ESPN_BASE}/{league}/teams"
            resp = requests.get(url, headers=HEADERS, timeout=8)
            if resp.status_code != 200:
                continue

            data = resp.json()
            name_lower = team_name.lower()
            sports = data.get("sports", [{}])
            leagues_data = sports[0].get("leagues", [{}]) if sports else [{}]
            teams_list = leagues_data[0].get("teams", []) if leagues_data else []

            for te in teams_list:
                t = te.get("team", {})
                display = t.get("displayName", "").lower()
                if name_lower in display or display in name_lower:
                    team_id = t.get("id")
                    form = _get_espn_team_form(league, team_id)
                    if form != 0.5:
                        return {"form": form, "data_sources": [f"ESPN ({league})"]}
    except Exception:
        pass
    return {}


def get_understat_team(team_name: str) -> dict:
    """Scrape xG data from Understat (top 5 European leagues)."""
    cached = cache.get("understat", {"team": team_name})
    if cached:
        return cached

    from datetime import datetime
    year = datetime.now().year
    url = f"https://understat.com/team/{team_name.replace(' ', '_')}/{year}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=(3, 7))
        if resp.status_code != 200:
            return {}

        soup = BeautifulSoup(resp.text, "lxml")
        scripts = soup.find_all("script")

        dates_data = []
        for script in scripts:
            text = script.string or ""
            if "datesData" in text:
                m = re.search(r"datesData\s*=\s*JSON\.parse\('(.+?)'\)", text)
                if m:
                    try:
                        raw = m.group(1).encode("utf-8").decode("unicode_escape")
                        dates_data = json.loads(raw)
                        break
                    except Exception:
                        pass

        if not dates_data:
            return {}

        recent = dates_data[-10:]
        xg_for = [float(m.get("xG", 0)) for m in recent]
        xg_against = [float(m.get("xGA", 0)) for m in recent]
        goals = [int(m.get("scored", 0)) for m in recent]
        conceded = [int(m.get("missed", 0)) for m in recent]
        results = [m.get("result", "") for m in recent]
        wins = results.count("w")
        draws = results.count("d")
        total = len(results)

        result = {
            "team": team_name,
            "avg_xg":     round(sum(xg_for) / len(xg_for), 3) if xg_for else 1.4,
            "avg_xga":    round(sum(xg_against) / len(xg_against), 3) if xg_against else 1.4,
            "avg_goals":  round(sum(goals) / len(goals), 3) if goals else 1.4,
            "avg_conceded": round(sum(conceded) / len(conceded), 3) if conceded else 1.4,
            "form":       round((wins + 0.5 * draws) / total, 4) if total else 0.5,
            "matches_analyzed": total,
            "is_national": False,
            "data_sources": ["Understat (live xG, last 10 games)"],
        }
        cache.set("understat", {"team": team_name}, result, ttl_seconds=3600 * 6)
        return result

    except Exception:
        return {}


def get_h2h(team1: str, team2: str) -> dict:
    """Get head-to-head football records."""
    cached = cache.get("h2h_football", {"t1": team1.lower(), "t2": team2.lower()})
    if cached:
        return cached

    H2H_SEEDS = {
        frozenset(["portugal", "spain"]):       {"home_wins": 10, "draws": 7,  "away_wins": 14, "total": 31},
        frozenset(["england", "france"]):        {"home_wins": 17, "draws": 7,  "away_wins": 9,  "total": 33},
        frozenset(["brazil", "argentina"]):      {"home_wins": 36, "draws": 25, "away_wins": 42, "total": 103},
        frozenset(["germany", "france"]):        {"home_wins": 20, "draws": 11, "away_wins": 20, "total": 51},
        frozenset(["spain", "germany"]):         {"home_wins": 11, "draws": 6,  "away_wins": 9,  "total": 26},
        frozenset(["england", "germany"]):       {"home_wins": 13, "draws": 5,  "away_wins": 9,  "total": 27},
        frozenset(["brazil", "france"]):         {"home_wins": 6,  "draws": 3,  "away_wins": 6,  "total": 15},
        frozenset(["argentina", "france"]):      {"home_wins": 6,  "draws": 2,  "away_wins": 3,  "total": 11},
        frozenset(["england", "argentina"]):     {"home_wins": 5,  "draws": 2,  "away_wins": 7,  "total": 14},
        frozenset(["brazil", "germany"]):        {"home_wins": 12, "draws": 5,  "away_wins": 10, "total": 27},
        frozenset(["spain", "italy"]):           {"home_wins": 15, "draws": 14, "away_wins": 9,  "total": 38},
        frozenset(["netherlands", "germany"]):   {"home_wins": 11, "draws": 8,  "away_wins": 19, "total": 38},
        frozenset(["england", "italy"]):         {"home_wins": 8,  "draws": 7,  "away_wins": 7,  "total": 22},
        frozenset(["spain", "france"]):          {"home_wins": 16, "draws": 10, "away_wins": 14, "total": 40},
        frozenset(["wales", "england"]):         {"home_wins": 14, "draws": 21, "away_wins": 66, "total": 101},
        frozenset(["scotland", "england"]):      {"home_wins": 41, "draws": 24, "away_wins": 48, "total": 113},
        frozenset(["barcelona", "real madrid"]): {"home_wins": 100,"draws": 52, "away_wins": 95, "total": 247},
    }

    key = frozenset([team1.lower(), team2.lower()])
    if key in H2H_SEEDS:
        data = dict(H2H_SEEDS[key])
        t1_is_first = team1.lower() < team2.lower()
        h2h = {
            "team1_wins": data["home_wins"] if t1_is_first else data["away_wins"],
            "draws": data["draws"],
            "team2_wins": data["away_wins"] if t1_is_first else data["home_wins"],
            "total": data["total"],
        }
        h2h["team1_win_rate"] = round(h2h["team1_wins"] / h2h["total"], 4)
        cache.set("h2h_football", {"t1": team1.lower(), "t2": team2.lower()}, h2h, ttl_seconds=86400)
        return h2h

    return {"team1_wins": 5, "draws": 3, "team2_wins": 5, "total": 13, "team1_win_rate": 0.385}
