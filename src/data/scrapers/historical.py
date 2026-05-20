"""
Historical match data scrapers for ML training and backtesting.
Sources: Understat (football xG), Jeff Sackmann CSVs (tennis).
All free, no API keys required.
"""
import io
import json
import re
import requests
from datetime import datetime
from pathlib import Path
from src.data import cache

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Understat supports EPL, La_liga, Bundesliga, Serie_A, Ligue_1
UNDERSTAT_LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"]


def fetch_understat_league_season(league: str, season: int) -> list:
    """
    Fetch completed matches for one league-season from Understat.
    Returns list of dicts with xG, goals, teams, date.
    season=2023 → 2023/24 campaign.
    """
    cached = cache.get("understat_hist", {"league": league, "season": season})
    if cached is not None:
        return cached

    url = f"https://understat.com/league/{league}/{season}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=25)
        if resp.status_code != 200:
            return []

        # Understat embeds match data as JSON.parse() call in page JS
        m = re.search(r"datesData\s*=\s*JSON\.parse\('(.+?)'\)", resp.text)
        if not m:
            return []

        raw = m.group(1).encode().decode("unicode_escape")
        matches_data = json.loads(raw)

        results = []
        for match in matches_data:
            if not match.get("isResult"):
                continue
            try:
                results.append({
                    "home_team": match["h"]["title"],
                    "away_team": match["a"]["title"],
                    "home_goals": int(match["goals"]["h"]),
                    "away_goals": int(match["goals"]["a"]),
                    "home_xg": float(match["xG"]["h"]),
                    "away_xg": float(match["xG"]["a"]),
                    "date": match.get("datetime", ""),
                    "league": league,
                    "season": season,
                })
            except (KeyError, ValueError, TypeError):
                continue

        cache.set("understat_hist", {"league": league, "season": season}, results, ttl_seconds=86400 * 30)
        return results

    except Exception:
        return []


def fetch_football_history(
    seasons: list = None,
    leagues: list = None,
    verbose: bool = False,
) -> list:
    """
    Fetch multi-season football match history from Understat.
    Default: EPL/La Liga/Bundesliga/Serie A/Ligue 1, 2014–present.
    Returns flat list sorted by date ascending.
    """
    if seasons is None:
        current_year = datetime.now().year
        seasons = list(range(2014, current_year))
    if leagues is None:
        leagues = UNDERSTAT_LEAGUES

    all_matches = []
    for league in leagues:
        for season in seasons:
            matches = fetch_understat_league_season(league, season)
            if matches and verbose:
                print(f"  {league} {season}/{season+1}: {len(matches)} matches")
            all_matches.extend(matches)

    def _parse_date(m):
        try:
            return datetime.strptime(m["date"][:10], "%Y-%m-%d")
        except Exception:
            return datetime(2000, 1, 1)

    all_matches.sort(key=_parse_date)
    return all_matches


def fetch_tennis_history(
    years: list = None,
    tour: str = "atp",
    verbose: bool = False,
) -> list:
    """
    Fetch tennis match history from Jeff Sackmann's public GitHub CSVs.
    Returns list of match dicts, sorted by tourney_date ascending.
    """
    from src.data.scrapers.sackmann import _get_matches

    if years is None:
        current_year = datetime.now().year
        years = list(range(2015, current_year + 1))

    all_matches = []
    for year in years:
        df = _get_matches(year, tour)
        if df.empty:
            continue

        count = 0
        for _, r in df.iterrows():
            try:
                w_svpt = float(r.get("w_svpt") or 1) or 1
                l_svpt = float(r.get("l_svpt") or 1) or 1
                all_matches.append({
                    "winner": str(r.get("winner_name", "")),
                    "loser": str(r.get("loser_name", "")),
                    "surface": str(r.get("surface", "Hard")),
                    "tourney_name": str(r.get("tourney_name", "")),
                    "tourney_date": str(r.get("tourney_date", "")),
                    "tourney_level": str(r.get("tourney_level", "A")),
                    "winner_rank": float(r.get("winner_rank") or 200),
                    "loser_rank": float(r.get("loser_rank") or 200),
                    "winner_age": float(r.get("winner_age") or 25),
                    "loser_age": float(r.get("loser_age") or 25),
                    # serve stats for rolling computation
                    "w_ace": float(r.get("w_ace") or 0),
                    "l_ace": float(r.get("l_ace") or 0),
                    "w_1stIn": float(r.get("w_1stIn") or 0),
                    "l_1stIn": float(r.get("l_1stIn") or 0),
                    "w_svpt": w_svpt,
                    "l_svpt": l_svpt,
                    "w_bpFaced": float(r.get("w_bpFaced") or 0),
                    "l_bpFaced": float(r.get("l_bpFaced") or 0),
                    "w_bpSaved": float(r.get("w_bpSaved") or 0),
                    "l_bpSaved": float(r.get("l_bpSaved") or 0),
                    "year": year,
                    "tour": tour,
                })
                count += 1
            except Exception:
                continue

        if verbose and count:
            print(f"  {tour.upper()} {year}: {count} matches")

    def _parse_tdate(m):
        try:
            return datetime.strptime(str(m["tourney_date"])[:8], "%Y%m%d")
        except Exception:
            return datetime(2000, 1, 1)

    all_matches.sort(key=_parse_tdate)
    return all_matches
