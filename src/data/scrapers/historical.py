"""
Comprehensive sports data downloader — all free, no API keys required.

Football (primary — instant, no rate limits):
  openfootball/football.json  (GitHub)
    14 seasons × EPL/La Liga/Bundesliga/Serie A/Ligue 1 + more = ~25k matches
  StatsBomb open-data           (GitHub)
    2,400+ matches with shot-level xG extracted automatically

Tennis (primary — instant):
  Jeff Sackmann GitHub CSVs — complete ATP & WTA 2000-present

UFC:
  UFCStats.com  — scraped; seeded data covers top fighters as fallback

Optional (requires Kaggle credentials in ~/.kaggle/kaggle.json):
  mdabbert/ultimate-ufc-dataset  — 3,700 fights with per-fight fighter stats
  hugomathien/soccer             — European Soccer DB with historical betting odds

Data is stored under data/raw/ for instant reruns.
"""
import io
import json
import re
import time
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from src.data import cache

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}

RAW_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── openfootball/football.json config ─────────────────────────────────────────
OPENFB_BASE = "https://raw.githubusercontent.com/openfootball/football.json/master"

OPENFB_LEAGUES = {
    "en.1":  "EPL",
    "es.1":  "La Liga",
    "de.1":  "Bundesliga",
    "it.1":  "Serie A",
    "fr.1":  "Ligue 1",
    "pt.1":  "Primeira Liga",
    "nl.1":  "Eredivisie",
    "be.1":  "Belgian Pro League",
    "sc.1":  "Scottish Premiership",
    "en.2":  "Championship",
    "de.2":  "2. Bundesliga",
    "es.2":  "La Liga 2",
}

def _openfb_seasons(min_year: int = 2010, max_year: int = None) -> list:
    """Return season strings in openfootball format, e.g. '2022-23'."""
    if max_year is None:
        max_year = datetime.now().year
    return [f"{y}-{str(y+1)[-2:]}" for y in range(min_year, max_year + 1)]


def fetch_openfootball_season(league_code: str, season: str) -> list:
    """
    Download one league-season JSON from openfootball.
    Returns list of {home_team, away_team, home_goals, away_goals, date, league, season}.
    """
    cache_key = f"openfb_{league_code}_{season}"
    local = RAW_DIR / f"{cache_key}.json"
    if local.exists():
        try:
            return json.loads(local.read_text())
        except Exception:
            pass

    url = f"{OPENFB_BASE}/{season}/{league_code}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []
        data = resp.json()
    except Exception:
        return []

    results = []
    league_name = OPENFB_LEAGUES.get(league_code, league_code)
    for m in data.get("matches", []):
        score = m.get("score", {})
        ft = score.get("ft")
        if ft is None or len(ft) < 2:
            continue
        try:
            results.append({
                "home_team":  m["team1"].rstrip(" FC").rstrip(" CF").strip(),
                "away_team":  m["team2"].rstrip(" FC").rstrip(" CF").strip(),
                "home_goals": int(ft[0]),
                "away_goals": int(ft[1]),
                "date":       m.get("date", ""),
                "league":     league_name,
                "season":     season,
                "home_xg":    None,
                "away_xg":    None,
            })
        except (KeyError, ValueError, TypeError):
            continue

    local.write_text(json.dumps(results))
    return results


def fetch_statsbomb_xg_index() -> dict:
    """
    Build a lookup {(home_norm, away_norm, date): (home_xg, away_xg)}
    from all StatsBomb full-season match files.
    Only downloads event files for competitions that have a full season (≥300 matches).
    """
    local = RAW_DIR / "statsbomb_xg_index.json"
    if local.exists():
        try:
            raw = json.loads(local.read_text())
            return {tuple(k.split("|")): tuple(v) for k, v in raw.items()}
        except Exception:
            pass

    comps_url = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json"
    try:
        comps = requests.get(comps_url, headers=HEADERS, timeout=15).json()
    except Exception:
        return {}

    # Only process men's competitions with full seasons (≥300 matches) to limit downloads
    target_comps = [
        (c["competition_id"], c["season_id"], c["competition_name"])
        for c in comps
        if c.get("competition_gender") == "male"
        and c["competition_name"] in (
            "Premier League", "La Liga", "1. Bundesliga", "Serie A", "Ligue 1",
        )
    ]

    xg_index = {}

    for cid, sid, cname in target_comps:
        matches_url = (
            f"https://raw.githubusercontent.com/statsbomb/open-data/master"
            f"/data/matches/{cid}/{sid}.json"
        )
        try:
            r = requests.get(matches_url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                continue
            matches = r.json()
            if len(matches) < 300:
                continue  # skip partial seasons

            print(f"  StatsBomb xG: {cname} ({len(matches)} matches)…", flush=True)
            for m in matches:
                mid = m["match_id"]
                hteam = m["home_team"]["home_team_name"]
                ateam = m["away_team"]["away_team_name"]
                mdate = m["match_date"]

                events_url = (
                    f"https://raw.githubusercontent.com/statsbomb/open-data/master"
                    f"/data/events/{mid}.json"
                )
                try:
                    er = requests.get(events_url, headers=HEADERS, timeout=20)
                    if er.status_code != 200:
                        continue
                    events = er.json()
                    shots = [e for e in events if e.get("type", {}).get("name") == "Shot"]
                    home_xg = sum(
                        e.get("shot", {}).get("statsbomb_xg", 0)
                        for e in shots
                        if e.get("team", {}).get("name") == hteam
                    )
                    away_xg = sum(
                        e.get("shot", {}).get("statsbomb_xg", 0)
                        for e in shots
                        if e.get("team", {}).get("name") == ateam
                    )
                    key = f"{_norm(hteam)}|{_norm(ateam)}|{mdate}"
                    xg_index[key] = (round(home_xg, 3), round(away_xg, 3))
                    time.sleep(0.05)  # polite rate limit
                except Exception:
                    continue

        except Exception:
            continue

    # Persist
    serialisable = {k: list(v) for k, v in xg_index.items()}
    local.write_text(json.dumps(serialisable))
    return xg_index


_UMLAUT_MAP = {
    "ü": "u", "ö": "o", "ä": "a", "ß": "ss", "é": "e", "è": "e",
    "ê": "e", "ë": "e", "ñ": "n", "ú": "u", "ó": "o", "á": "a",
    "í": "i", "î": "i", "ï": "i", "ç": "c", "ø": "o", "å": "a",
    "æ": "ae", "œ": "oe", "ã": "a", "õ": "o", "â": "a", "ô": "o",
}
_CLUB_PREFIXES = (
    "1. fc ", "bsc ", "afc ", "sfc ", "ssc ",
    "fc ", "sc ", "sv ", "fk ", "vfl ", "vfb ", "ss ",
)

# Post-normalisation aliases — map openfootball norms to StatsBomb norms (or vice versa).
# Both directions are stored so lookup works regardless of which dataset is the query side.
_NORM_ALIASES: dict[str, str] = {
    # German cities: umlaut-transliterated ≠ English spelling
    "bayernmunchen":             "bayernmunich",
    "borussiamonchengladbach":   "borussiamnchengladbach",
    "kln":                       "cologne",
    "1koln":                     "cologne",
    "1fckoln":                   "cologne",
    "fckoln":                    "cologne",
    "nrnberg":                   "nuremberg",
    "1fcnurnberg":               "1fcnuremberg",
    "mainz05":                   "fsvmainz05",
    "fsvmainz":                  "fsvmainz05",
    "1fsmainz05":                "fsvmainz05",
    # French
    "olympiquedemarseille":      "marseille",
    "olympiquelyon":             "lyon",
    "parisstgermain":            "parisstgermain",
    "psg":                       "parisstgermain",
    "stbretagne":                "stade rennais",
    # Spanish
    "atleticodemadrid":          "atleticomadrid",
    "deportivoalavs":            "alavs",
    "alavs":                     "deportivaalaves",
    "rceltavigo":                "celtavigo",
    "realbetisbalompi":          "realbetis",
    "realvalladolid":            "valladolid",
    "girona":                    "gironaf",
    # Italian
    "hellasveronafc":            "hellasveronafc",
    "usdinesec":                 "udinese",
    "cagliaricalcio":            "cagliari",
    "empolif":                   "empoli",
    "bresciaf":                  "brescia",
    "leccef":                    "lecce",
    # Reverse aliases (StatsBomb → openfootball)
    "bayernmunich":              "bayernmunchen",
    "borussiamnchengladbach":    "borussiamonchengladbach",
    "cologne":                   "1fckoln",
    "marseille":                 "olympiquedemarseille",
    "lyon":                      "olympiquelyon",
    "atleticomadrid":            "atleticodemadrid",
}


def _norm(name: str) -> str:
    """Normalise team name — transliterates umlauts, strips club prefixes, applies aliases."""
    n = name.lower()
    for src, dst in _UMLAUT_MAP.items():
        n = n.replace(src, dst)
    for p in _CLUB_PREFIXES:
        if n.startswith(p):
            n = n[len(p):]
            break
    n = re.sub(r"[^a-z0-9]", "", n)
    return _NORM_ALIASES.get(n, n)


def _norm_variants(name: str) -> list[str]:
    """Return all normalised variants of a team name (primary + alias chain)."""
    primary = _norm(name)
    variants = [primary]
    alias = _NORM_ALIASES.get(primary)
    if alias and alias != primary:
        variants.append(alias)
    # Also check if primary appears as a value in the alias dict (reverse lookup)
    for k, v in _NORM_ALIASES.items():
        if v == primary and k != primary and k not in variants:
            variants.append(k)
    return variants


def _merge_xg(matches: list, xg_index: dict) -> list:
    """Attach StatsBomb xG values to openfootball matches where available.

    Tries all name variants (primary normalisation + aliases) to maximise match rate.
    """
    if not xg_index:
        return matches

    # Build a flat dict for fast lookup: (h_norm, a_norm, date) → (hxg, axg)
    flat: dict[tuple, tuple] = {}
    for k, v in xg_index.items():
        if isinstance(k, str):
            parts = k.split("|")
            if len(parts) == 3:
                flat[tuple(parts)] = v
        else:
            flat[k] = v

    enriched = []
    for m in matches:
        if m.get("home_xg") is None:
            date = m["date"]
            hit = None
            for h_var in _norm_variants(m["home_team"]):
                for a_var in _norm_variants(m["away_team"]):
                    hit = flat.get((h_var, a_var, date))
                    if hit:
                        break
                if hit:
                    break
            if hit:
                m = dict(m)
                m["home_xg"], m["away_xg"] = hit
        enriched.append(m)
    return enriched


def fetch_football_history(
    min_season_year: int = 2010,
    leagues: list = None,
    include_xg: bool = True,
    verbose: bool = False,
) -> list:
    """
    Download complete football match history from openfootball GitHub + optional StatsBomb xG.

    Parameters
    ----------
    min_season_year : first season to include (default 2010 → 15 seasons)
    leagues         : subset of OPENFB_LEAGUES keys; None = all majors
    include_xg      : if True, augment with StatsBomb shot xG where available
    verbose         : print progress

    Returns flat list sorted by date, ~25k+ matches.
    """
    if leagues is None:
        leagues = ["en.1", "es.1", "de.1", "it.1", "fr.1",
                   "pt.1", "nl.1", "be.1", "en.2", "de.2"]

    seasons = _openfb_seasons(min_year=min_season_year)

    all_matches = []
    for code in leagues:
        league_name = OPENFB_LEAGUES.get(code, code)
        for season in seasons:
            ms = fetch_openfootball_season(code, season)
            if ms:
                all_matches.extend(ms)
                if verbose:
                    print(f"  {league_name} {season}: {len(ms)} matches")

    # Sort by date
    def _dt(m):
        try:
            return datetime.strptime(m["date"], "%Y-%m-%d")
        except Exception:
            return datetime(2000, 1, 1)

    all_matches.sort(key=_dt)

    if include_xg and all_matches:
        if verbose:
            print("  Building StatsBomb xG index (downloads event files)…")
        xg_index = fetch_statsbomb_xg_index()
        if xg_index:
            all_matches = _merge_xg(all_matches, xg_index)
            enriched = sum(1 for m in all_matches if m.get("home_xg") is not None)
            if verbose:
                print(f"  xG enriched: {enriched}/{len(all_matches)} matches")

    if verbose:
        print(f"  Total: {len(all_matches)} football matches loaded.")

    return all_matches


# ── Understat (legacy / live xG supplement) ───────────────────────────────────
# Kept for live team-level xG fetching; historical training uses openfootball above.
UNDERSTAT_LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"]


def fetch_understat_league_season(league: str, season: int) -> list:
    """Fetch completed matches with xG from Understat (rate-limited; for live use)."""
    cached = cache.get("understat_hist", {"league": league, "season": season})
    if cached is not None:
        return cached

    url = f"https://understat.com/league/{league}/{season}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=25)
        if resp.status_code != 200:
            return []
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
                    "home_team":  match["h"]["title"],
                    "away_team":  match["a"]["title"],
                    "home_goals": int(match["goals"]["h"]),
                    "away_goals": int(match["goals"]["a"]),
                    "home_xg":    float(match["xG"]["h"]),
                    "away_xg":    float(match["xG"]["a"]),
                    "date":       match.get("datetime", "")[:10],
                    "league":     league,
                    "season":     season,
                })
            except (KeyError, ValueError, TypeError):
                continue
        cache.set("understat_hist", {"league": league, "season": season}, results, ttl_seconds=86400 * 30)
        return results
    except Exception:
        return []


# ── Tennis: Jeff Sackmann ─────────────────────────────────────────────────────

def fetch_tennis_history(
    years: list = None,
    tours: list = None,
    verbose: bool = False,
) -> list:
    """
    Fetch tennis history from Jeff Sackmann's public GitHub CSVs.

    Parameters
    ----------
    years : years to fetch; default 2010-present
    tours : list of tours to fetch — any of "atp", "wta", "atp_chall", "wta_itf"
            default = ["atp", "wta"]  (main tour ATP + WTA, ~200k+ matches total)
    verbose : print progress

    Returns flat list sorted by tourney_date.
    """
    from src.data.scrapers.sackmann import _get_matches

    if years is None:
        current_year = datetime.now().year
        years = list(range(2010, current_year + 1))
    if tours is None:
        tours = ["atp", "wta"]

    all_matches = []
    for tour in tours:
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
                        "winner":        str(r.get("winner_name", "")),
                        "loser":         str(r.get("loser_name", "")),
                        "surface":       str(r.get("surface", "Hard")),
                        "tourney_name":  str(r.get("tourney_name", "")),
                        "tourney_date":  str(r.get("tourney_date", "")),
                        "tourney_level": str(r.get("tourney_level", "A")),
                        "winner_rank":   float(r.get("winner_rank") or 200),
                        "loser_rank":    float(r.get("loser_rank") or 200),
                        "winner_age":    float(r.get("winner_age") or 25),
                        "loser_age":     float(r.get("loser_age") or 25),
                        "w_ace":     float(r.get("w_ace") or 0),
                        "l_ace":     float(r.get("l_ace") or 0),
                        "w_1stIn":   float(r.get("w_1stIn") or 0),
                        "l_1stIn":   float(r.get("l_1stIn") or 0),
                        "w_svpt":    w_svpt,
                        "l_svpt":    l_svpt,
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

    def _sort_key(m):
        try:
            return datetime.strptime(str(m["tourney_date"])[:8], "%Y%m%d")
        except Exception:
            return datetime(2000, 1, 1)

    all_matches.sort(key=_sort_key)
    return all_matches


# ── Kaggle integration (optional) ─────────────────────────────────────────────

def _kaggle_available() -> bool:
    """Return True if kaggle package is installed and credentials are configured."""
    import os
    from pathlib import Path as P
    creds = P.home() / ".kaggle" / "kaggle.json"
    env_ok = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    if not (creds.exists() or env_ok):
        return False
    try:
        import importlib
        spec = importlib.util.find_spec("kaggle")
        return spec is not None
    except Exception:
        return False


def download_kaggle_dataset(dataset_id: str, output_dir: Path = None) -> Path | None:
    """
    Download a Kaggle dataset to output_dir.
    Requires kaggle package + credentials.
    Returns path to the downloaded directory, or None on failure.
    """
    if not _kaggle_available():
        return None
    if output_dir is None:
        output_dir = RAW_DIR / "kaggle" / dataset_id.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download",
         "-d", dataset_id, "--unzip", "-p", str(output_dir)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return output_dir
    print(f"[kaggle] Download failed for {dataset_id}: {result.stderr[:200]}")
    return None


def fetch_kaggle_ufc(verbose: bool = False) -> list:
    """
    Download the Ultimate UFC Dataset from Kaggle (mdabbert/ultimate-ufc-dataset).
    Returns list of fight dicts with per-fight fighter stats.
    Requires kaggle credentials; returns [] if unavailable.
    """
    local = RAW_DIR / "kaggle" / "mdabbert_ultimate-ufc-dataset" / "ufc-master.csv"
    if not local.exists():
        if verbose:
            print("  Downloading Kaggle UFC dataset (mdabbert/ultimate-ufc-dataset)…")
        path = download_kaggle_dataset("mdabbert/ultimate-ufc-dataset")
        if path is None:
            return []
        # Find the CSV
        csvs = list(path.rglob("*.csv"))
        if not csvs:
            return []
        local = csvs[0]

    try:
        df = pd.read_csv(local)
        fights = []
        for _, row in df.iterrows():
            try:
                fights.append({
                    "r_fighter": str(row.get("R_fighter", "")),
                    "b_fighter": str(row.get("B_fighter", "")),
                    "winner":    str(row.get("Winner", "")),
                    "date":      str(row.get("date", "")),
                    "weight_class": str(row.get("weight_class", "")),
                    "r_slpm":   float(row.get("R_avg_SIG_STR_pct", 0.45)),
                    "b_slpm":   float(row.get("B_avg_SIG_STR_pct", 0.45)),
                    "r_td_avg": float(row.get("R_avg_TD_pct", 0.4)),
                    "b_td_avg": float(row.get("B_avg_TD_pct", 0.4)),
                    "r_win_rate": float(row.get("R_wins", 0)) / max(float(row.get("R_wins", 0)) + float(row.get("R_losses", 1)), 1),
                    "b_win_rate": float(row.get("B_wins", 0)) / max(float(row.get("B_wins", 0)) + float(row.get("B_losses", 1)), 1),
                })
            except Exception:
                continue
        if verbose:
            print(f"  Kaggle UFC: {len(fights)} fights loaded.")
        return fights
    except Exception as e:
        if verbose:
            print(f"  UFC Kaggle parse error: {e}")
        return []


def fetch_kaggle_soccer_db(verbose: bool = False) -> list:
    """
    Download the European Soccer Database from Kaggle (hugomathien/soccer).
    Returns list of match dicts with odds data.
    Requires kaggle credentials; returns [] if unavailable.
    """
    import sqlite3
    local_db = RAW_DIR / "kaggle" / "hugomathien_soccer" / "database.sqlite"
    if not local_db.exists():
        if verbose:
            print("  Downloading Kaggle European Soccer DB (hugomathien/soccer)…")
        path = download_kaggle_dataset("hugomathien/soccer")
        if path is None:
            return []
        dbs = list(path.rglob("*.sqlite"))
        if not dbs:
            return []
        local_db = dbs[0]

    try:
        conn = sqlite3.connect(str(local_db))
        df = pd.read_sql_query("""
            SELECT m.date, ht.team_long_name AS home_team, at.team_long_name AS away_team,
                   m.home_team_goal, m.away_team_goal,
                   m.B365H, m.B365D, m.B365A
            FROM Match m
            JOIN Team ht ON m.home_team_api_id = ht.team_api_id
            JOIN Team at ON m.away_team_api_id = at.team_api_id
            WHERE m.home_team_goal IS NOT NULL
        """, conn)
        conn.close()

        results = []
        for _, r in df.iterrows():
            try:
                results.append({
                    "home_team":  str(r["home_team"]),
                    "away_team":  str(r["away_team"]),
                    "home_goals": int(r["home_team_goal"]),
                    "away_goals": int(r["away_team_goal"]),
                    "date":       str(r["date"])[:10],
                    "league":     "European Soccer DB",
                    "season":     str(r["date"])[:4],
                    "home_xg":    None,
                    "away_xg":    None,
                    # Betting odds for calibration
                    "b365h":      float(r["B365H"]) if pd.notna(r.get("B365H")) else None,
                    "b365d":      float(r["B365D"]) if pd.notna(r.get("B365D")) else None,
                    "b365a":      float(r["B365A"]) if pd.notna(r.get("B365A")) else None,
                })
            except Exception:
                continue

        if verbose:
            print(f"  Kaggle Soccer DB: {len(results)} matches loaded.")
        return results
    except Exception as e:
        if verbose:
            print(f"  Kaggle Soccer DB parse error: {e}")
        return []


def get_data_summary() -> dict:
    """Return a summary of locally cached data."""
    football = list(RAW_DIR.glob("openfb_*.json"))
    has_xg_index = (RAW_DIR / "statsbomb_xg_index.json").exists()
    kaggle_dir = RAW_DIR / "kaggle"
    has_ufc_kaggle = any(kaggle_dir.rglob("ufc-master.csv")) if kaggle_dir.exists() else False
    has_soccer_kaggle = any(kaggle_dir.rglob("database.sqlite")) if kaggle_dir.exists() else False

    xg_count = 0
    if has_xg_index:
        try:
            xg_count = len(json.loads((RAW_DIR / "statsbomb_xg_index.json").read_text()))
        except Exception:
            pass

    return {
        "football_files": len(football),
        "statsbomb_xg_matches": xg_count,
        "kaggle_ufc": has_ufc_kaggle,
        "kaggle_soccer_db": has_soccer_kaggle,
        "kaggle_available": _kaggle_available(),
    }
