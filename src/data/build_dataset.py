"""Build a REAL historical match + player dataset from open-source data.

Source: openfootball (https://github.com/openfootball), pulled via
raw.githubusercontent.com. These are real, community-maintained results — no
fabricated/synthetic data. Anything unreachable is simply skipped, so the
dataset reflects exactly what was actually downloaded.

Outputs (under data/historical/):
  matches.json       — every real match: date, competition, teams, score, result
  team_last50.json   — per team, their most recent <=50 matches + rolling stats
  player_goals.json  — real goal-scorer tallies (from World Cup/Euro goal feeds)
  manifest.json      — what was fetched, counts, and provenance

Run:  python -m src.data.build_dataset
"""
import json
import os
import time
from collections import defaultdict

import requests

RAW = "https://raw.githubusercontent.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (SportsBettingPredictor dataset builder)"}

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "historical",
)

# Competition weight mirrors the model's COMPETITION_WEIGHTS (importance).
LEAGUES = ["en.1", "es.1", "de.1", "it.1", "fr.1", "en.2", "es.2", "pt.1", "nl.1"]
LEAGUE_SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2011, 2025)]  # 2011-12 .. 2024-25

WORLDCUP_YEARS = ["2010", "2014", "2018", "2022"]
EURO_YEARS = ["2020", "2024"]


def _get(url: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _sources():
    """Yield (url, competition, is_international)."""
    for y in WORLDCUP_YEARS:
        yield f"{RAW}/openfootball/worldcup.json/master/{y}/worldcup.json", "world cup", True
    for y in EURO_YEARS:
        yield f"{RAW}/openfootball/euro.json/master/{y}/euro.json", "euro", True
    for season in LEAGUE_SEASONS:
        for lg in LEAGUES:
            comp = {"en.1": "premier league", "es.1": "la liga", "de.1": "bundesliga",
                    "it.1": "serie a", "fr.1": "ligue 1"}.get(lg, "league")
            yield f"{RAW}/openfootball/football.json/master/{season}/{lg}.json", comp, False


def _clean(team: str) -> str:
    """Normalise team names to match the rest of the app (lower, drop FC/CF...)."""
    t = team.strip()
    for suffix in (" FC", " CF", " AFC", " SC", " BK", " FK"):
        if t.endswith(suffix):
            t = t[: -len(suffix)]
    return t.strip()


def _result(hs: int, as_: int) -> str:
    return "home_win" if hs > as_ else ("away_win" if as_ > hs else "draw")


def build():
    os.makedirs(OUT_DIR, exist_ok=True)
    matches = []
    player_goals = defaultdict(lambda: {"goals": 0, "matches_scored_in": 0, "teams": set()})
    fetched, missed = [], []

    for url, comp, is_intl in _sources():
        data = _get(url)
        if not data or "matches" not in data:
            missed.append(url.replace(RAW, ""))
            continue
        n_before = len(matches)
        for m in data["matches"]:
            score = (m.get("score") or {}).get("ft")
            if not score or len(score) != 2:
                continue
            try:
                hs, as_ = int(score[0]), int(score[1])
            except (TypeError, ValueError):
                continue
            home, away = _clean(m.get("team1", "")), _clean(m.get("team2", ""))
            if not home or not away:
                continue
            matches.append({
                "date": m.get("date", ""),
                "competition": comp,
                "international": is_intl,
                "home": home, "away": away,
                "hs": hs, "as": as_,
                "result": _result(hs, as_),
            })
            # Real player goals (present in international tournament feeds).
            for side, team in (("goals1", home), ("goals2", away)):
                scorers = m.get(side) or []
                seen = set()
                for g in scorers:
                    nm = (g.get("name") or "").strip()
                    if not nm:
                        continue
                    player_goals[nm]["goals"] += 1
                    player_goals[nm]["teams"].add(team)
                    seen.add(nm)
                for nm in seen:
                    player_goals[nm]["matches_scored_in"] += 1
        fetched.append({"src": url.replace(RAW, ""), "matches": len(matches) - n_before})
        time.sleep(0.05)

    matches.sort(key=lambda x: x["date"])
    _write("matches.json", matches)

    # Per-team last 50 + rolling stats.
    by_team = defaultdict(list)
    for m in matches:
        by_team[m["home"]].append(m)
        by_team[m["away"]].append(m)
    team_last50 = {}
    for team, ms in by_team.items():
        recent = sorted(ms, key=lambda x: x["date"])[-50:]
        team_last50[team] = {
            "n": len(recent),
            "stats": _team_stats(team, recent),
            "matches": recent,
        }
    _write("team_last50.json", team_last50)

    pg = {k: {"goals": v["goals"], "matches_scored_in": v["matches_scored_in"],
              "teams": sorted(v["teams"])}
          for k, v in sorted(player_goals.items(), key=lambda kv: -kv[1]["goals"])}
    _write("player_goals.json", pg)

    manifest = {
        "source": "openfootball via raw.githubusercontent.com (real data)",
        "total_matches": len(matches),
        "total_teams": len(team_last50),
        "total_scorers": len(pg),
        "files_fetched": [f for f in fetched if f["matches"] > 0],
        "files_missed": len(missed),
        "date_range": [matches[0]["date"], matches[-1]["date"]] if matches else [],
    }
    _write("manifest.json", manifest)
    print(f"Built: {len(matches)} matches | {len(team_last50)} teams | {len(pg)} scorers")
    print(f"Date range: {manifest['date_range']}")
    return manifest


def _team_stats(team: str, recent: list) -> dict:
    if not recent:
        return {}
    gf = ga = wins = draws = 0
    for m in recent:
        if m["home"] == team:
            gf += m["hs"]; ga += m["as"]
            wins += m["result"] == "home_win"; draws += m["result"] == "draw"
        else:
            gf += m["as"]; ga += m["hs"]
            wins += m["result"] == "away_win"; draws += m["result"] == "draw"
    n = len(recent)
    return {
        "games": n,
        "avg_goals": round(gf / n, 3),
        "avg_conceded": round(ga / n, 3),
        "win_rate": round(wins / n, 3),
        "draw_rate": round(draws / n, 3),
        "points_per_game": round((wins * 3 + draws) / n, 3),
    }


def _write(name: str, obj):
    with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    build()
