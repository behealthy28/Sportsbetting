"""Build a REAL international-only dataset for the current World Cup teams.

Source: martj42/international_results (https://github.com/martj42/international_results),
the canonical open dataset of every men's international since 1872, pulled via
raw.githubusercontent.com. Real data only — no clubs, no fabrication.

Scope (per user spec):
  - INTERNATIONAL teams only (no club football).
  - Restricted to teams in the CURRENT World Cup (auto-detected from the file's
    2026 'FIFA World Cup' fixtures).
  - Each team's most recent <=N_PER_TEAM played internationals.

Outputs (under data/historical/):
  matches.json       — real played internationals (date, comp, teams, score,
                       result, neutral flag) involving current WC teams
  team_last50.json   — per WC team, last <=N_PER_TEAM matches + rolling stats
  player_goals.json  — real goal-scorer tallies (recent, WC teams)
  wc_teams.json      — the detected current-WC participant list
  manifest.json      — provenance + counts

Run:  python -m src.data.build_dataset
"""
import csv
import io
import json
import os
from collections import defaultdict

import requests

RAW = "https://raw.githubusercontent.com/martj42/international_results/master"
HEADERS = {"User-Agent": "Mozilla/5.0 (SportsBettingPredictor dataset builder)"}

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "historical",
)

N_PER_TEAM = 50          # most recent N internationals per team
RECENCY_FROM = "2018-01-01"   # don't go further back than this

# Tournament -> competition label used by the model's COMPETITION_WEIGHTS.
def _competition(tournament: str) -> str:
    t = tournament.lower()
    if "world cup" in t and "qual" not in t:
        return "world cup"
    if "world cup qual" in t:
        return "world cup qualifier"
    if "euro" in t and "qual" not in t:
        return "euro"
    if "copa am" in t or "copa amé" in t:
        return "copa america"
    if "nations league" in t:
        return "nations league"
    if "african cup" in t and "qual" not in t:
        return "afcon"
    if "asian cup" in t and "qual" not in t:
        return "asian cup"
    if "friendly" in t:
        return "friendly"
    return "international"


def _get_csv(name: str) -> list:
    r = requests.get(f"{RAW}/{name}", headers=HEADERS, timeout=30)
    r.raise_for_status()
    return list(csv.DictReader(io.StringIO(r.text)))


def _result(hs: int, as_: int) -> str:
    return "home_win" if hs > as_ else ("away_win" if as_ > hs else "draw")


def _detect_wc_teams(rows: list) -> set:
    """Teams appearing in the current (latest-year) FIFA World Cup fixtures."""
    wc_years = sorted({r["date"][:4] for r in rows
                       if r["tournament"] == "FIFA World Cup"})
    if not wc_years:
        return set()
    latest = wc_years[-1]
    teams = set()
    for r in rows:
        if r["tournament"] == "FIFA World Cup" and r["date"].startswith(latest):
            teams.add(r["home_team"])
            teams.add(r["away_team"])
    return teams


def build():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = _get_csv("results.csv")
    wc_teams = _detect_wc_teams(rows)

    # Played matches involving a current WC team, within the recency window.
    played = []
    for r in rows:
        if r["date"] < RECENCY_FROM:
            continue
        if r["home_team"] not in wc_teams and r["away_team"] not in wc_teams:
            continue
        try:
            hs, as_ = int(r["home_score"]), int(r["away_score"])
        except (ValueError, KeyError):
            continue  # NA = upcoming/unplayed
        played.append({
            "date": r["date"],
            "competition": _competition(r["tournament"]),
            "international": True,
            "neutral": r.get("neutral", "").upper() == "TRUE",
            "home": r["home_team"], "away": r["away_team"],
            "hs": hs, "as": as_,
            "result": _result(hs, as_),
        })
    played.sort(key=lambda x: x["date"])

    # Keep only each WC team's most recent N matches (union -> match set).
    per_team = defaultdict(list)
    for m in played:
        per_team[m["home"]].append(m)
        per_team[m["away"]].append(m)
    keep_ids = set()
    for team in wc_teams:
        for m in per_team.get(team, [])[-N_PER_TEAM:]:
            keep_ids.add(id(m))
    matches = [m for m in played if id(m) in keep_ids]
    _write("matches.json", matches)

    # Per-team last-N + rolling stats (WC teams only).
    team_last50 = {}
    for team in sorted(wc_teams):
        recent = per_team.get(team, [])[-N_PER_TEAM:]
        if recent:
            team_last50[team] = {"n": len(recent),
                                 "stats": _team_stats(team, recent),
                                 "matches": recent}
    _write("team_last50.json", team_last50)

    # Real goal scorers (recent, WC teams).
    player_goals = defaultdict(lambda: {"goals": 0, "teams": set()})
    try:
        for g in _get_csv("goalscorers.csv"):
            if g["date"] < RECENCY_FROM or g.get("own_goal", "").upper() == "TRUE":
                continue
            team = g.get("team", "")
            if team not in wc_teams:
                continue
            nm = (g.get("scorer") or "").strip()
            if nm:
                player_goals[nm]["goals"] += 1
                player_goals[nm]["teams"].add(team)
    except Exception:
        pass
    pg = {k: {"goals": v["goals"], "teams": sorted(v["teams"])}
          for k, v in sorted(player_goals.items(), key=lambda kv: -kv[1]["goals"])}
    _write("player_goals.json", pg)
    _write("wc_teams.json", sorted(wc_teams))

    manifest = {
        "source": "martj42/international_results via raw.githubusercontent (real data)",
        "scope": "international only · current World Cup teams · no clubs",
        "wc_teams": len(wc_teams),
        "n_per_team": N_PER_TEAM,
        "total_matches": len(matches),
        "total_scorers": len(pg),
        "date_range": [matches[0]["date"], matches[-1]["date"]] if matches else [],
    }
    _write("manifest.json", manifest)
    print(f"WC teams detected: {len(wc_teams)}")
    print(f"Built: {len(matches)} real international matches | {len(pg)} scorers")
    print(f"Date range: {manifest['date_range']}")
    return manifest


def _team_stats(team: str, recent: list) -> dict:
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
