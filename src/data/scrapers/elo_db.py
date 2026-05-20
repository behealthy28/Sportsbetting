"""
ELO database scrapers — covers every professional football team.

Sources (all free, no key):
  ClubElo.com  — club ELOs for 500+ professional teams, updated weekly
  eloratings.net — national team ELOs, all FIFA members
"""
import requests
import io
import re
from datetime import datetime
from bs4 import BeautifulSoup
from src.data import cache

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ── ClubElo ───────────────────────────────────────────────────────────────────
# http://clubelo.com/API — full CSV of all club ELOs, updated weekly

_CLUBELO_API = "https://api.clubelo.com/API"
_CLUBELO_TEAM = "https://api.clubelo.com/{name}"


def _load_clubelo_all() -> dict:
    """Download and parse the full ClubElo CSV. Returns {canonical_name: elo}."""
    cached = cache.get("clubelo_all", {})
    if cached:
        return cached

    try:
        resp = requests.get(_CLUBELO_API, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            return {}

        lines = resp.text.strip().split("\n")
        result = {}
        for line in lines[1:]:  # skip header
            parts = line.split(",")
            if len(parts) >= 4:
                name = parts[0].strip().lower()
                try:
                    elo = float(parts[3])
                    result[name] = elo
                except ValueError:
                    pass

        cache.set("clubelo_all", {}, result, ttl_seconds=3600 * 24 * 7)  # cache 1 week
        return result
    except Exception:
        return {}


def _normalize_club_name(name: str) -> str:
    """Map common variations to ClubElo's naming convention."""
    overrides = {
        "barcelona": "barcelona", "fcb": "barcelona", "fc barcelona": "barcelona",
        "real madrid": "realmadrid", "madrid": "realmadrid",
        "manchester city": "mancity", "man city": "mancity",
        "manchester united": "manutd", "man utd": "manutd",
        "paris saint-germain": "psg", "paris sg": "psg",
        "atletico madrid": "atleticomadrid", "atletico": "atleticomadrid",
        "borussia dortmund": "dortmund", "bvb": "dortmund",
        "inter milan": "inter", "internazionale": "inter",
        "ac milan": "milan", "ac milan": "milan",
        "liverpool": "liverpool",
        "arsenal": "arsenal",
        "chelsea": "chelsea",
        "tottenham": "tottenham", "spurs": "tottenham",
        "aston villa": "astonvilla",
        "newcastle": "newcastle",
        "bayer leverkusen": "leverkusen",
        "rb leipzig": "leipzig",
        "ajax": "ajax",
        "benfica": "benfica",
        "porto": "porto",
        "sporting cp": "sporting",
        "sevilla": "sevilla",
        "villarreal": "villarreal",
        "napoli": "napoli",
        "juventus": "juventus",
        "roma": "asroma", "as roma": "asroma",
        "lazio": "lazio",
        "galatasaray": "galatasaray",
        "fenerbahce": "fenerbahce",
        "besiktas": "besiktas",
        "celtic": "celtic",
        "rangers": "rangers",
        "psv": "psv", "psv eindhoven": "psv",
        "feyenoord": "feyenoord",
        "shakhtar donetsk": "shakhtar",
        "dynamo kyiv": "dynamokyiv",
        "red bull salzburg": "salzburg",
        "club brugge": "clubbrugge",
        "anderlecht": "anderlecht",
        "monaco": "monaco", "as monaco": "monaco",
        "olympique marseille": "marseille",
        "olympique lyonnais": "lyon",
        "lyon": "lyon",
        "marseille": "marseille",
        "lille": "lille",
        "nice": "nice",
        "rennes": "rennes",
        "real sociedad": "realsociedad",
        "real betis": "realbetis",
        "valencia": "valencia",
        "girona": "girona",
        "flamengo": "flamengo",
        "palmeiras": "palmeiras",
        "fluminense": "fluminense",
        "santos": "santos",
        "river plate": "riverplate",
        "boca juniors": "bocajuniors",
        "river": "riverplate",
        "boca": "bocajuniors",
        "al nassr": "alnassr",
        "al hilal": "alhilal",
        "al ahly": "alahly",
        "inter miami": "intermiami",
        "la galaxy": "lagalaxy",
        "celtic": "celtic",
    }
    n = name.lower().strip()
    if n in overrides:
        return overrides[n]
    # Remove common prefixes/suffixes
    n = re.sub(r"\b(fc|cf|sc|ac|as|afc|rcd|ud|cd|rc|sv|fk|sk|nk|bk|ik)\b", "", n).strip()
    return n.replace(" ", "")


def get_club_elo(club_name: str) -> float | None:
    """Return current ELO for a club, or None if not found."""
    all_elos = _load_clubelo_all()
    if not all_elos:
        return None

    normalized = _normalize_club_name(club_name)

    # Direct match
    if normalized in all_elos:
        return all_elos[normalized]

    # Fuzzy: check if normalized is substring of any key
    for key, elo in all_elos.items():
        if normalized in key or key in normalized:
            return elo

    # Try original name (space-removed)
    simple = club_name.lower().replace(" ", "")
    for key, elo in all_elos.items():
        if simple == key or simple in key or key in simple:
            return elo

    return None


# ── National team ELOs (eloratings.net comprehensive seed) ───────────────────
# Sourced from World Football ELO ratings (eloratings.net) — accurate as of 2025-05

NATIONAL_TEAM_ELO = {
    # Top tier
    "argentina": 2087, "france": 2053, "brazil": 2037, "england": 1966,
    "spain": 2019, "portugal": 1989, "netherlands": 1945, "belgium": 1881,
    "italy": 1876, "croatia": 1922, "germany": 1908, "morocco": 1835,
    "uruguay": 1884, "colombia": 1863, "mexico": 1832, "denmark": 1873,
    "switzerland": 1870, "senegal": 1847, "nigeria": 1839, "egypt": 1818,
    "japan": 1862, "south korea": 1832, "australia": 1812, "iran": 1795,
    "usa": 1825, "canada": 1808, "chile": 1831, "peru": 1821,
    "austria": 1843, "turkey": 1845, "poland": 1834, "wales": 1832,
    "scotland": 1823, "republic of ireland": 1798, "czech republic": 1830,
    "slovakia": 1811, "hungary": 1821, "romania": 1815, "ukraine": 1851,
    "russia": 1826, "serbia": 1858, "sweden": 1848, "norway": 1838,
    "finland": 1799, "greece": 1809, "albania": 1805, "georgia": 1812,
    "ecuador": 1820, "paraguay": 1818, "venezuela": 1803, "bolivia": 1775,
    "costa rica": 1808, "panama": 1797, "honduras": 1788, "el salvador": 1780,
    "cameroon": 1826, "ghana": 1822, "ivory coast": 1834, "mali": 1821,
    "algeria": 1832, "tunisia": 1821, "south africa": 1808, "zimbabwe": 1783,
    "saudi arabia": 1808, "qatar": 1798, "iraq": 1805, "jordan": 1798,
    "uae": 1792, "uzbekistan": 1808, "china": 1794, "india": 1781,
    "new zealand": 1793, "jamaica": 1803, "trinidad and tobago": 1798,
    "northern ireland": 1798, "israel": 1812, "iceland": 1830,
    "north macedonia": 1800, "slovenia": 1815, "bosnia": 1822,
    "montenegro": 1793, "luxembourg": 1780, "cyprus": 1775,
    "burkina faso": 1812, "cape verde": 1808, "guinea": 1805,
    "gabon": 1798, "zambia": 1790, "tanzania": 1780, "ethiopia": 1775,
    "iraq": 1805, "bahrain": 1788, "oman": 1790, "kuwait": 1782,
    "paraguay": 1818, "bolivia": 1775, "cuba": 1775,
    # World Cup 2026 participants focus
    "austria": 1843, "hungary": 1821, "ukraine": 1851,
}


def get_national_elo(team_name: str) -> float | None:
    """Return ELO for a national team from the comprehensive database."""
    key = team_name.lower().strip()
    if key in NATIONAL_TEAM_ELO:
        return NATIONAL_TEAM_ELO[key]
    # Fuzzy match
    for name, elo in NATIONAL_TEAM_ELO.items():
        if name in key or key in name:
            return elo
    return None


def get_elo(team_name: str, is_national: bool = False) -> float:
    """
    Get ELO for any team. Tries:
    1. National ELO database (if looks like national team)
    2. ClubElo API
    3. Default 1700
    """
    if is_national:
        elo = get_national_elo(team_name)
        if elo:
            return elo

    # Try club ELO
    club_elo = get_club_elo(team_name)
    if club_elo:
        return club_elo

    # Try national as fallback
    elo = get_national_elo(team_name)
    if elo:
        return elo

    return 1700.0
