"""Natural language parser for sports match queries."""
import re
import json
from datetime import datetime, timedelta
from pathlib import Path

TEAMS_FILE = Path(__file__).parent.parent / "data" / "mappings" / "teams.json"
PLAYERS_FILE = Path(__file__).parent.parent / "data" / "mappings" / "players.json"

DATE_KEYWORDS = {
    "today": 0, "tonight": 0, "now": 0,
    "tomorrow": 1, "tmrw": 1,
    "monday": None, "tuesday": None, "wednesday": None,
    "thursday": None, "friday": None, "saturday": None, "sunday": None,
}
WEEKDAY_MAP = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
               "friday": 4, "saturday": 5, "sunday": 6}

SPORT_INDICATORS = {
    "table tennis": ["table tennis", "ittf", "ping pong", "fan zhendong", "ma long"],
    "darts": ["darts", "pdc", "bdo", "premier league darts", "world darts",
              "van gerwen", "mvg", "littler", "humphries"],
    "badminton": ["badminton", "bwf", "all england", "thomas cup", "axelsen",
                  "carolina marin", "shi yuqi"],
    "cricket": ["cricket", "test match", " odi ", " t20 ", "t20i", "ipl", "ashes",
                "bcci", "batting", "bowling", "wicket", "t20 world cup", "cricket world cup"],
    "tennis": ["atp", "wta", "wimbledon", "us open", "french open", "australian open",
               "roland garros", "grand slam", "masters 1000", "tennis",
               "djokovic", "alcaraz", "sinner", "swiatek", "sabalenka", "nadal"],
    "ufc": ["ufc", " mma ", "fight night", "bellator", "one fc", "ufc ppv",
            "makhachev", "adesanya", "volkanovski", "mcgregor"],
    "boxing": ["boxing", "wbc", "wba", "ibf", "wbo", "heavyweight boxing",
               "boxing title", "world boxing", "fury", "usyk", "canelo", "joshua"],
    "football": ["football", "soccer", "fc ", " fc", " united", "city fc",
                 "premier league", "la liga", "bundesliga", "serie a", "ligue 1",
                 "champions league", "world cup", "euros", "euro ", "copa america",
                 "nations league", "afcon", "friendly"],
}

NEUTRAL_KEYWORDS = ["neutral", "at wembley", "at munich", "at doha", "world cup",
                    "euro final", "at stadium", "hosted by",
                    "champions league final", "ucl final", "europa league final",
                    "fa cup final", "copa final", "cup final", "grand final",
                    "at lisbon", "at istanbul", "at paris", "at berlin", "at madrid",
                    "at rome", "at london", "at milan"]
COMPETITION_PATTERNS = [
    r"(world cup|euro \d{4}|euros?|copa america|afcon|nations league|"
    r"champions league|premier league|la liga|bundesliga|serie a|ligue 1|"
    r"wimbledon|us open|french open|australian open|roland garros|"
    r"grand slam|masters|ufc \d+|ipl|ashes|t20 world cup|cricket world cup|"
    r"pdc world|uk open|premier league darts)",
]


def _load_all_known_names() -> dict:
    """Load all known team/player names for entity resolution."""
    names = {}
    try:
        teams = json.loads(TEAMS_FILE.read_text())
        for sport, team_map in teams.items():
            for alias, info in team_map.items():
                names[alias.lower()] = {"sport": sport, "canonical": info.get("name", alias)}
    except Exception:
        pass

    try:
        players = json.loads(PLAYERS_FILE.read_text())
        for sport, player_map in players.items():
            for alias, info in player_map.items():
                names[alias.lower()] = {"sport": sport, "canonical": info.get("name", alias)}
    except Exception:
        pass

    return names


def _parse_date(text: str) -> str:
    """Extract date from text. Returns ISO date string."""
    text_lower = text.lower()
    today = datetime.now().date()

    for kw, delta in DATE_KEYWORDS.items():
        if kw in text_lower:
            if delta is not None:
                return str(today + timedelta(days=delta))
            # Weekday
            if kw in WEEKDAY_MAP:
                target_wd = WEEKDAY_MAP[kw]
                days_ahead = (target_wd - today.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                return str(today + timedelta(days=days_ahead))

    # Try to match date patterns: "May 15", "15 May", "2026-05-15"
    date_patterns = [
        r"(\d{4}-\d{2}-\d{2})",
        r"(\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}?)",
        r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}(?:st|nd|rd|th)?(?:\s+\d{4})?)",
    ]
    for pattern in date_patterns:
        m = re.search(pattern, text_lower, re.IGNORECASE)
        if m:
            try:
                from dateutil import parser as dateparser
                dt = dateparser.parse(m.group(1), default=datetime(today.year, today.month, today.day))
                return str(dt.date())
            except Exception:
                pass

    return str(today + timedelta(days=1))  # default: tomorrow


def _extract_entities(text: str, known_names: dict) -> tuple:
    """Extract two competing entities (teams or players) from text."""
    text_lower = text.lower()

    # Split on "vs", "versus", "v ", " - "
    vs_patterns = [r"\bvs\.?\b", r"\bversus\b", r"\bv\b(?=\s)", r" - "]
    entities = None
    for pat in vs_patterns:
        parts = re.split(pat, text_lower, flags=re.IGNORECASE)
        if len(parts) == 2:
            entities = [p.strip() for p in parts]
            break

    if not entities or len(entities) < 2:
        return None, None

    e1_raw, e2_raw = entities

    def resolve(raw: str) -> str:
        # Remove date/competition noise
        clean = re.sub(
            r"\b(tomorrow|today|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
            r"next week|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
            r"\d{4}|\d{1,2}(?:st|nd|rd|th)?)\b",
            "", raw, flags=re.IGNORECASE
        ).strip()
        clean = re.sub(r"\s+", " ", clean).strip()

        # Check known names
        for alias, info in known_names.items():
            if alias in clean:
                return info["canonical"]

        # Return cleaned title-case
        return clean.title() if clean else raw.title()

    return resolve(e1_raw), resolve(e2_raw)


def _detect_sport(text: str, e1: str = "", e2: str = "") -> str:
    """Detect sport from text and entity names."""
    full_text = (text + " " + e1 + " " + e2).lower()

    # Check entity names against known sport mappings — only trust for
    # unambiguous sport-specific entities (not "football" which shares names)
    known = _load_all_known_names()
    for entity in [e1.lower(), e2.lower()]:
        for alias, info in known.items():
            if (alias == entity or alias in entity.split() or entity == alias):
                sport = info.get("sport", "")
                if sport in ["tennis", "ufc", "boxing"]:
                    return sport

    # Keyword-based detection — ordered most specific to least specific
    detection_order = ["table tennis", "darts", "badminton", "cricket", "tennis", "ufc", "boxing", "football"]
    for sport_name in detection_order:
        keywords = SPORT_INDICATORS.get(sport_name, [])
        if any(kw in full_text for kw in keywords):
            return sport_name

    # If both entities are known football teams, return football
    for entity in [e1.lower(), e2.lower()]:
        for alias, info in known.items():
            if alias == entity and info.get("sport") == "football":
                return "football"

    return "football"  # default


def _extract_competition(text: str) -> str:
    """Extract competition/tournament name from text."""
    for pattern in COMPETITION_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).title()
    return ""


# ── Prop detection ────────────────────────────────────────────────────────────
# Ordered from most specific to least specific to avoid false matches.

_PROP_PATTERNS = [
    # ─ Exact scoreline e.g. "2-1" or "2:1 correct score"
    (r"\bcorrect\s+score\s+(\d)[:\-](\d)\b",                     "correct_score",    None),
    (r"\b(\d)[:\-](\d)\s+correct\s+score\b",                     "correct_score",    None),
    # ─ Foot / header scorer
    (r"\bscores?\s+with\s+(?:his|her|their)?\s*(left|right)\s+foot\b",  "player_foot", None),
    (r"\b(left|right)\s+foot\s+(?:goal|scorer?)\b",                     "player_foot", None),
    (r"\b(header|heads?)\s+(?:goal|scorer?)\b",                         "player_header", None),
    (r"\bscores?\s+(?:a\s+)?(header|with\s+his\s+head)\b",              "player_header", None),
    # ─ First scorer
    (r"\bfirst\s+(?:goal)?scorer?\b",                            "player_first_scorer", None),
    (r"\bscores?\s+first\b",                                     "player_first_scorer", None),
    # ─ Anytime scorer / assists
    (r"\bto\s+score\b",                                          "player_scorer",    None),
    (r"\banytime\s+scorer\b",                                    "player_scorer",    None),
    (r"\bwill\s+\w+(?:\s+\w+)?\s+score\b",                      "player_scorer",    None),
    (r"\bscorer?\s+anytime\b",                                   "player_scorer",    None),
    (r"\bto\s+(?:get\s+an?\s+)?assist\b",                       "player_assists",   None),
    (r"\banytime\s+assist\b",                                    "player_assists",   None),
    # ─ BTTS
    (r"\bbtts\b",                                                "btts",             None),
    (r"\bboth\s+teams?\s+to\s+score\b",                         "btts",             None),
    (r"\bboth\s+(?:sides?|teams?)\s+score\b",                   "btts",             None),
    # ─ Half-time result
    (r"\bhalf.?time\s+(?:result|winner|score)\b",               "half_time",        None),
    (r"\bht\s+(?:result|winner)\b",                             "half_time",        None),
    # ─ Over / Under — goals / runs / rounds / sets / points
    (r"\bover\s+(\d+\.?\d*)\s+(goals?|runs?|rounds?|sets?|points?|corners?|cards?|assists?)\b",
                                                                 "over_under",       "over"),
    (r"\bunder\s+(\d+\.?\d*)\s+(goals?|runs?|rounds?|sets?|points?|corners?|cards?|assists?)\b",
                                                                 "over_under",       "under"),
    (r"\b(\d+\.?\d*)\s+(goals?|runs?|rounds?|sets?|points?)\s*(over|under|o/u|ou)\b",
                                                                 "over_under",       None),
    (r"\b(o/u|ou)\s+(\d+\.?\d*)\s*(goals?|runs?|rounds?|sets?)?\b",
                                                                 "over_under",       None),
    # ─ UFC / Boxing method of victory
    (r"\b(?:wins?\s+by|victory\s+by|finish\s+by)\s+(ko|knockout|tko|submission|sub|decision|dec|points)\b",
                                                                 "method_victory",   None),
    (r"\b(ko|knockout|tko|submission|stoppage)\s+(?:win|victory|finish)\b",
                                                                 "method_victory",   None),
    (r"\bmethod\s+of\s+victory\b",                              "method_victory",   None),
    # ─ Goes distance
    (r"\bgoes?\s+(?:the\s+)?distance\b",                        "goes_distance",    None),
    (r"\bfull\s+\d+\s+rounds?\b",                               "goes_distance",    None),
    (r"\bno\s+(?:ko|finish|stoppage)\b",                        "goes_distance",    None),
    # ─ Tennis first set
    (r"\bfirst\s+set\b",                                        "tennis_first_set", None),
    (r"\bwins?\s+(?:the\s+)?first\s+set\b",                    "tennis_first_set", None),
    # ─ Tiebreak
    (r"\b(?:a\s+)?tie.?break\b",                                "tennis_tiebreak",  None),
    # ─ Sets over/under (already caught by generic over/under)
    # ─ Cricket top scorer
    (r"\btop\s+(bat(?:sman|ter)?|scorer|run[- ]scorer)\b",      "cricket_top_bat",  None),
    (r"\bhighest\s+(?:individual\s+)?scorer\b",                 "cricket_top_bat",  None),
    # ─ Wickets over/under
    (r"\bover\s+(\d+\.?\d*)\s+wickets?\b",                      "cricket_wickets_ou","over"),
    (r"\bunder\s+(\d+\.?\d*)\s+wickets?\b",                     "cricket_wickets_ou","under"),
    # ─ Century
    (r"\bscores?\s+(?:a\s+)?centur(?:y|ies)\b",                "cricket_century",  None),
    (r"\bcentur(?:y|ies)\b",                                    "cricket_century",  None),
]


def _detect_prop(query: str) -> tuple:
    """
    Returns (bet_type, prop_params) where:
      bet_type    — string key or "match_result"
      prop_params — dict with threshold, direction, stat, foot, etc.
    """
    low = query.lower()

    for pattern, bet_type, direction_hint in _PROP_PATTERNS:
        m = re.search(pattern, low, re.IGNORECASE)
        if not m:
            continue

        params: dict = {}
        if direction_hint:
            params["direction"] = direction_hint

        if bet_type == "over_under":
            # Extract number and stat
            num_match = re.search(r"(\d+\.?\d*)", low)
            stat_match = re.search(
                r"(goals?|runs?|rounds?|sets?|points?|corners?|cards?|assists?|wickets?)",
                low, re.IGNORECASE
            )
            if num_match:
                params["threshold"] = float(num_match.group(1))
            params["stat"] = stat_match.group(1).rstrip("s").lower() if stat_match else "goal"
            if not direction_hint:
                params["direction"] = "over" if "over" in low else "under"

        elif bet_type == "correct_score":
            nums = re.findall(r"\d", low[:30])
            if len(nums) >= 2:
                params["home_goals"] = int(nums[0])
                params["away_goals"] = int(nums[1])

        elif bet_type == "player_foot":
            foot_match = re.search(r"\b(left|right)\b", low)
            params["foot"] = foot_match.group(1) if foot_match else "right"

        elif bet_type == "method_victory":
            method_match = re.search(
                r"\b(ko|knockout|tko|submission|sub|decision|dec|points)\b", low
            )
            if method_match:
                raw = method_match.group(1).lower()
                params["method"] = "ko_tko" if raw in ("ko", "tko", "knockout") \
                    else "submission" if raw in ("sub", "submission") \
                    else "decision"

        elif bet_type == "cricket_wickets_ou":
            bet_type = "over_under"
            params["stat"] = "wicket"
            num_match = re.search(r"(\d+\.?\d*)", low)
            if num_match:
                params["threshold"] = float(num_match.group(1))

        return bet_type, params

    return "match_result", {}


def _extract_player(query: str, entity1: str, entity2: str) -> str:
    """
    For player props, extract the player name from the query.
    Searches the raw query first so verb-phrases like "will X score" are caught
    before entity stripping can destroy them.
    """
    low = query.lower()

    # Try raw query first — catches "will Neymar score", "Haaland anytime scorer"
    patterns = [
        r"\bwill\s+([a-z][a-z\s\-'\.]{2,25}?)\s+(?:score|assist|get|bag)",
        r"\b([a-z][a-z\s\-'\.]{2,25}?)\s+(?:to\s+score|anytime\s+scorer|first\s+(?:goal)?scorer)",
        r"\b([a-z][a-z\s\-'\.]{2,25}?)\s+(?:scores?\s+(?:with|a\s+header|first)|header\s+goal)",
        r"\b([a-z][a-z\s\-'\.]{2,25}?)\s+(?:scores?|assists?)\b",
    ]
    stopwords = {"the", "a", "an", "for", "with", "his", "her", "their",
                 "in", "at", "vs", "versus", "will", "both", "all"}
    for pat in patterns:
        for m in re.finditer(pat, low, re.IGNORECASE):
            name = m.group(1).strip().rstrip("'s").strip()
            words = name.split()
            # Filter out junk matches
            if (len(name) >= 4
                    and not all(w in stopwords for w in words)
                    and not re.match(r"^\d", name)
                    and name.lower() not in (entity1.lower(), entity2.lower())):
                return name.title()

    return ""


def parse(query: str) -> dict:
    """
    Parse a natural language sports query.
    Returns: {entity1, entity2, date, sport, competition, is_neutral, bet_type, prop_params,
               prop_player, raw_query}
    """
    known = _load_all_known_names()

    # ── Special commands ──
    ql = query.strip().lower()
    if ql in ("show sports", "list sports", "sports"):
        return {"command": "show_sports"}
    if ql in ("help", "h", "?"):
        return {"command": "help"}

    # ── Prop-only queries without vs (e.g. "will Neymar score?", "over 2.5 goals Barca vs Madrid")
    # Detect prop type BEFORE entity extraction so we can strip prop noise
    bet_type, prop_params = _detect_prop(query)

    entity1, entity2 = _extract_entities(query, known)

    if not entity1 or not entity2:
        return {"error": f"Could not identify two competitors in: '{query}'"}

    date = _parse_date(query)
    sport = _detect_sport(query, entity1, entity2)
    competition = _extract_competition(query)
    is_neutral = any(kw in query.lower() for kw in NEUTRAL_KEYWORDS)

    prop_player = ""
    if bet_type not in ("match_result",):
        prop_player = _extract_player(query, entity1, entity2)

    return {
        "entity1": entity1,
        "entity2": entity2,
        "date": date,
        "sport": sport,
        "competition": competition,
        "is_neutral": is_neutral,
        "venue": "",
        "bet_type": bet_type,
        "prop_params": prop_params,
        "prop_player": prop_player,
        "raw_query": query,
    }
