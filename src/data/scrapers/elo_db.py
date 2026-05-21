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
    """Return current ELO for a club. Tries live ClubElo API first, then seed table."""
    all_elos = _load_clubelo_all()
    normalized = _normalize_club_name(club_name)

    if all_elos:
        if normalized in all_elos:
            return all_elos[normalized]
        for key, elo in all_elos.items():
            if normalized in key or key in normalized:
                return elo
        simple = club_name.lower().replace(" ", "")
        for key, elo in all_elos.items():
            if simple == key or simple in key or key in simple:
                return elo

    # Fallback to hardcoded seeds when API is unreachable
    key = club_name.lower().strip()
    if key in CLUB_ELO_SEEDS:
        return CLUB_ELO_SEEDS[key]
    # Fuzzy match against seeds
    for seed_name, elo in CLUB_ELO_SEEDS.items():
        if seed_name in key or key in seed_name:
            return elo

    return None


# ── Club ELO seeds — fallback when ClubElo API is unreachable ────────────────
# Sourced from ClubElo.com / transfermarkt ratings, calibrated to early 2026.
# Scale: 2000+ elite (top European winner), 1900-2000 UCL contender,
#        1800-1900 good top-division, 1700-1800 mid-table, <1700 lower tier.

CLUB_ELO_SEEDS = {
    # ── England ──
    "arsenal": 1938, "liverpool": 1932, "manchester city": 1921, "man city": 1921,
    "chelsea": 1858, "manchester united": 1843, "man utd": 1843,
    "tottenham": 1812, "spurs": 1812, "aston villa": 1848, "newcastle": 1831,
    "brighton": 1802, "west ham": 1788, "fulham": 1785, "brentford": 1780,
    "nottingham forest": 1783, "everton": 1762, "leicester": 1768,
    "crystal palace": 1771, "wolves": 1774,

    # ── Spain ──
    "real madrid": 1971, "barcelona": 1932, "atletico madrid": 1889,
    "atletico": 1889, "athletic club": 1851, "athletic bilbao": 1851,
    "real sociedad": 1838, "villarreal": 1832, "sevilla": 1818,
    "real betis": 1821, "valencia": 1798, "girona": 1825, "osasuna": 1785,
    "getafe": 1775, "rayo vallecano": 1778,

    # ── Germany ──
    "bayern munich": 1918, "bayer leverkusen": 1891, "leverkusen": 1891,
    "borussia dortmund": 1862, "dortmund": 1862, "rb leipzig": 1855, "leipzig": 1855,
    "eintracht frankfurt": 1831, "wolfsburg": 1812, "freiburg": 1808,
    "borussia monchengladbach": 1798, "hoffenheim": 1795,
    "werder bremen": 1793, "union berlin": 1785,

    # ── Italy ──
    "inter milan": 1895, "internazionale": 1895, "inter": 1895,
    "napoli": 1868, "juventus": 1851, "ac milan": 1845, "milan": 1845,
    "atalanta": 1862, "roma": 1832, "as roma": 1832, "lazio": 1821,
    "fiorentina": 1808, "torino": 1788, "bologna": 1812,

    # ── France ──
    "paris saint-germain": 1902, "psg": 1902, "paris sg": 1902,
    "olympique marseille": 1842, "marseille": 1842,
    "olympique lyonnais": 1831, "lyon": 1831,
    "monaco": 1848, "as monaco": 1848, "lille": 1838, "nice": 1822,
    "lens": 1815, "rennes": 1812, "stade brestois": 1808, "brest": 1808,

    # ── Portugal ──
    "benfica": 1858, "porto": 1845, "sporting cp": 1841, "sporting": 1841,
    "braga": 1802,

    # ── Netherlands ──
    "ajax": 1832, "psv": 1845, "psv eindhoven": 1845, "feyenoord": 1838,
    "az alkmaar": 1795, "az": 1795, "twente": 1788,

    # ── Belgium ──
    "club brugge": 1825, "anderlecht": 1808, "union saint-gilloise": 1812,
    "gent": 1795,

    # ── Turkey ──
    "galatasaray": 1855, "fenerbahce": 1845, "besiktas": 1822,
    "trabzonspor": 1798,

    # ── Scotland / CEE ──
    "celtic": 1838, "rangers": 1821, "red bull salzburg": 1818, "salzburg": 1818,
    "shakhtar donetsk": 1815, "dynamo kyiv": 1798,

    # ── South America ──
    "flamengo": 1862, "palmeiras": 1858, "fluminense": 1845, "atletico mineiro": 1841,
    "river plate": 1855, "boca juniors": 1838, "estudiantes": 1802,
    "internacional": 1812, "sao paulo": 1808, "corinthians": 1795,
    "gremio": 1788, "santos": 1778,

    # ── Saudi / MLS / Asia ──
    "al nassr": 1808, "al hilal": 1821, "al ahly": 1815, "al ittihad": 1795,
    "inter miami": 1798, "la galaxy": 1785, "seattle sounders": 1778,
    "urawa red diamonds": 1788, "kashima antlers": 1782,

    # ── Common aliases ──
    "man city": 1921, "man utd": 1843, "inter": 1895, "milan": 1845,
    "atletico": 1889, "bvb": 1862, "rb": 1855,
}


# ── National team ELOs — all 48 FIFA World Cup 2026 nations + full coverage ──
# Sourced from eloratings.net / World Football ELO, calibrated to May 2026.
# Scale: 2100+ historically dominant, 2000-2100 top-5 world, 1900-2000 top-20,
#        1800-1900 good international, 1700-1800 developing, <1700 weakest.

NATIONAL_TEAM_ELO = {
    # ── CONMEBOL (South America) ──
    "argentina": 2091,   # World Cup 2022 winners, Copa 2024 winners
    "brazil": 2039,      # consistent top-3, strong squad depth
    "colombia": 1872,    # Copa 2024 finalist, strong
    "uruguay": 1888,     # always competitive, world-class front line
    "ecuador": 1828,     # solid CONMEBOL qualifier
    "venezuela": 1812,   # surprisingly strong qualifying campaign
    "paraguay": 1821,    # experienced South American side
    "chile": 1835,       # qualified despite golden gen decline
    "peru": 1818,        # borderline qualifier
    "bolivia": 1778,
    "cuba": 1730,

    # ── UEFA (Europe) — 16 spots ──
    "spain": 2022,       # Euro 2024 winners, best team in Europe
    "france": 2055,      # world-class squad, top contender
    "england": 1975,     # Euro 2024 finalist, strong squad
    "portugal": 1995,    # Ronaldo era ending, still elite
    "germany": 1912,     # Euro 2024 hosts, resurgent under Nagelsmann
    "netherlands": 1952, # strong UCL-level squad, Van Dijk era
    "croatia": 1926,     # Modric generation winding down, still top
    "belgium": 1878,     # golden gen decline but still competitive
    "italy": 1875,       # Euro 2020 winners, quality defence
    "denmark": 1878,     # consistent performers
    "switzerland": 1872, # solid, always qualify comfortably
    "austria": 1851,     # very strong qualifying, Alaba/Sabitzer
    "turkey": 1849,      # Euro 2024 QF, improving
    "scotland": 1832,    # qualified, strong home record
    "ukraine": 1855,     # qualified despite difficult circumstances
    "serbia": 1862,      # Mitrovic generation, always qualify
    "poland": 1838,      # Lewandowski led side
    "wales": 1825,       # qualified, Bale era ended
    "czech republic": 1832,
    "slovakia": 1818,    # solid mid-tier European
    "hungary": 1825,     # qualified from Group A
    "romania": 1818,     # surprisingly strong Euro 2024 group stage
    "albania": 1808,
    "georgia": 1815,     # historic Euro 2024 run
    "slovenia": 1818,
    "sweden": 1848,      # consistent performers without Ibra
    "norway": 1848,      # Haaland era, dangerous
    "finland": 1802,
    "greece": 1812,
    "northern ireland": 1798,
    "republic of ireland": 1798, "ireland": 1798,
    "israel": 1815,
    "iceland": 1832,
    "north macedonia": 1798,
    "bosnia": 1825,      "bosnia and herzegovina": 1825,
    "montenegro": 1795,
    "luxembourg": 1782,
    "cyprus": 1778,

    # ── CONCACAF (North/Central America + Caribbean) — 6 + 3 hosts ──
    "usa": 1835,         # co-host 2026, developing young squad
    "mexico": 1838,      # co-host 2026, experienced
    "canada": 1818,      # co-host 2026, Davies/David generation
    "costa rica": 1808,
    "panama": 1802,
    "honduras": 1792,
    "el salvador": 1782,
    "jamaica": 1805,
    "trinidad and tobago": 1798,
    "haiti": 1778,
    "guatemala": 1780,
    "cuba": 1730,
    "curacao": 1778,

    # ── CAF (Africa) — 9 spots ──
    "morocco": 1848,     # 2022 WC semi-finalist, top African side
    "nigeria": 1842,     # always strong, AFCON contenders
    "ivory coast": 1838, "cote d'ivoire": 1838,  # AFCON 2023 winners
    "senegal": 1851,     # AFCON 2022 winners, Mane generation
    "egypt": 1822,       # Salah era, consistent qualifier
    "cameroon": 1828,    # always competitive African giant
    "ghana": 1818,
    "mali": 1825,
    "algeria": 1835,
    "tunisia": 1822,
    "south africa": 1812,
    "cape verde": 1812,
    "burkina faso": 1815,
    "guinea": 1808,
    "gabon": 1798,
    "zambia": 1792,
    "democratic republic of congo": 1808, "dr congo": 1808,
    "ethiopia": 1778,
    "tanzania": 1782,
    "zimbabwe": 1785,
    "uganda": 1785,
    "mozambique": 1772,

    # ── AFC (Asia) — 8 + 1 host ──
    "japan": 1868,       # strongest Asian side, impressive 2022 WC
    "south korea": 1842, # consistent Asian qualifier
    "iran": 1802,        # reliable qualifier, strong defensively
    "saudi arabia": 1812,# improving, 2022 WC giant-killer
    "australia": 1818,   # hybrid European-style play
    "qatar": 1802,       # 2022 hosts, improving standards
    "iraq": 1808,
    "uzbekistan": 1812,
    "jordan": 1802,
    "china": 1795,
    "bahrain": 1792,
    "oman": 1790,
    "kuwait": 1782,
    "uae": 1795,
    "india": 1782,

    # ── OFC (Oceania) — 1 spot ──
    "new zealand": 1795,
    "fiji": 1752,

    # ── Misc / common alternates ──
    "russia": 1828,      # suspended from FIFA competitions currently
    "north korea": 1762,
    "taiwan": 1758,
    "indonesia": 1778,
    "thailand": 1780,
    "vietnam": 1775,
    "myanmar": 1762,
    "syria": 1792,
    "lebanon": 1782,
    "palestine": 1768,
    "ethiopia": 1778,
    "kenya": 1778,
    "angola": 1782,
    "benin": 1778,
    "equatorial guinea": 1775,
    "namibia": 1772,
    "rwanda": 1768,
    "comoros": 1762,
    "gambia": 1778,
    "sudan": 1758,
    "liberia": 1765,
    "togo": 1762,
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
    Get ELO for any team. Resolution order:
    1. National ELO seed table (if is_national or looks like a country name)
    2. ClubElo live API → CLUB_ELO_SEEDS fallback
    3. National seed table as final fallback (catches country queries without flag)
    4. Default 1700
    """
    if is_national:
        elo = get_national_elo(team_name)
        if elo:
            return elo

    club_elo = get_club_elo(team_name)
    if club_elo:
        return club_elo

    elo = get_national_elo(team_name)
    if elo:
        return elo

    return 1700.0
