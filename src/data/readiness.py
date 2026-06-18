"""Squad readiness / recovery factor.

Estimates how match-ready a team's squad is as a 0-1 readiness score, then maps
that to a squad-strength multiplier that feeds the prediction models (via the
``home_key_players`` / ``away_key_players`` context slots).

------------------------------------------------------------------------------
DATA ETHICS — read before extending this module
------------------------------------------------------------------------------
This module deliberately does NOT scrape, harvest, infer, or store any
individual's private biometric data from WHOOP, Oura, Garmin, Apple Health,
Fitbit, Polar, Coros, or any other health-tracking service — even when a
misconfigured "public" leaderboard makes a named person's recovery/strain
score technically reachable. Tracking identifiable athletes' health metrics
without consent (and especially to bet against them) is a privacy violation,
not a data source. It is also bad modelling: a self-reported, unverifiable
score leaked for a handful of starters is noise, not edge.

Readiness here is derived only from non-invasive, defensible signals:
  1. Days of rest since a team's last fixture (fixture congestion).
  2. Publicly reported, team-level injury/availability news (optional; the
     app already pulls Google News RSS elsewhere).
  3. An OPTIONAL manual override the user supplies for what-if analysis — an
     explicit estimate the user types in, clearly labelled as such, never
     presented as harvested biometric truth.
"""
import json
import os
from typing import Optional

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "mappings",
)
_OVERRIDES_PATH = os.path.join(_DATA_DIR, "readiness.json")
_LINEUPS_PATH = os.path.join(_DATA_DIR, "lineups.json")

# Severity of a publicly reported availability concern, by headline language.
_OUT_KEYWORDS = ("ruled out", "out of", "withdrawn", "withdraws", "miss the",
                 "will miss", "sidelined", "suspended", "ban", "surgery", "torn")
_DOUBT_KEYWORDS = ("doubt", "doubtful", "fitness test", "race against time",
                   "assessed", "knock", "limped", "scan")
_INJURY_KEYWORDS = ("injury", "injured", "hamstring", "knee", "ankle", "muscle",
                    "calf", "groin", "fracture", "strain", "illness", "ill")

# Multiplier band. A fully fresh / no-concern squad is neutral (1.0); a
# depleted or fixture-congested squad is penalised toward _MULT_MIN. Readiness
# never *inflates* a team above baseline, so an unknown team stays at 1.0.
_MULT_MIN = 0.85
_MULT_MAX = 1.00


def _load_overrides() -> dict:
    try:
        with open(_OVERRIDES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k.lower(): v for k, v in data.items() if not k.startswith("_")}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _load_lineups() -> dict:
    try:
        with open(_LINEUPS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k.lower(): v for k, v in data.items()
                if not k.startswith("_") and isinstance(v, list)}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _name_in_headline(player: str, headline: str) -> bool:
    """Match a player by full name or surname within a headline."""
    h = headline.lower()
    full = player.lower()
    if full in h:
        return True
    surname = full.split()[-1] if full.split() else full
    return len(surname) >= 4 and surname in h


def _severity(headline: str) -> float:
    """How much a publicly reported concern dents availability (0 = none)."""
    h = headline.lower()
    if not any(k in h for k in _INJURY_KEYWORDS + _OUT_KEYWORDS + _DOUBT_KEYWORDS):
        return 0.0
    if any(k in h for k in _OUT_KEYWORDS):
        return 1.0
    if any(k in h for k in _DOUBT_KEYWORDS):
        return 0.5
    return 0.3  # generic injury mention


def get_squad_availability(team: str, news_sentiment: Optional[dict] = None,
                           manual_out: Optional[list] = None) -> dict:
    """Importance-weighted squad availability from PUBLIC injury/availability news.

    Scans the team's public roster (lineups.json) against publicly reported
    news headlines for each key player, weighting any concern by that player's
    importance. Uses NO private biometric/tracker data.

    Args:
        team: team name.
        news_sentiment: dict from ``news.get_sentiment`` (uses its headlines).
        manual_out: optional list of player names the user marks as out.

    Returns:
        {"availability" (0-1), "flagged": [{player, importance, severity, note}]}.
    """
    roster = _load_lineups().get(team.lower(), [])
    if not roster:
        return {"availability": 1.0, "flagged": []}

    headlines = (news_sentiment or {}).get("headlines", []) or \
        (news_sentiment or {}).get("flags", [])
    manual_out = {m.lower() for m in (manual_out or [])}

    total_importance = sum(p.get("importance", 0.5) for p in roster) or 1.0
    lost = 0.0
    flagged = []
    for p in roster:
        name = p.get("name", "")
        imp = p.get("importance", 0.5)
        sev = 0.0
        note = ""
        if name.lower() in manual_out:
            sev, note = 1.0, "marked out (manual)"
        else:
            for hl in headlines:
                if _name_in_headline(name, hl):
                    s = _severity(hl)
                    if s > sev:
                        sev, note = s, hl[:80]
        if sev > 0:
            lost += imp * sev
            flagged.append({"player": name, "importance": imp,
                            "severity": sev, "note": note})

    availability = max(0.0, 1.0 - lost / total_importance)
    return {"availability": round(availability, 4), "flagged": flagged}


def _rest_component(days_rest: int) -> float:
    """Map days of rest to a 0-1 freshness sub-score.

    ~6-8 days is the ideal international turnaround. Short rest = fatigue;
    a very long layoff carries mild match-rust.
    """
    if days_rest is None:
        return 0.85
    if days_rest >= 13:
        return 0.92          # long layoff — slight rust
    if days_rest >= 6:
        return 1.00          # ideal
    if days_rest >= 4:
        return 0.92
    if days_rest >= 3:
        return 0.85
    return 0.78              # congested fixture / minimal recovery


def readiness_to_multiplier(score: float) -> float:
    """Map a 0-1 readiness score to a squad-strength multiplier."""
    score = max(0.0, min(1.0, score))
    return round(_MULT_MIN + (_MULT_MAX - _MULT_MIN) * score, 4)


def get_readiness(
    team: str,
    days_rest: int = 7,
    manual: Optional[float] = None,
    news_sentiment: Optional[dict] = None,
    manual_out: Optional[list] = None,
) -> dict:
    """Return a readiness assessment for a team.

    Args:
        team: team name (case-insensitive).
        days_rest: days since the team's last fixture.
        manual: optional user-supplied 0-1 readiness estimate (what-if knob).
                Overrides any value in readiness.json.
        news_sentiment: optional dict from ``src.data.news.get_sentiment``;
                only its flag *count* nudges readiness (kept small so we do not
                double-count the separate news adjustment applied downstream).

    Returns:
        {"score", "multiplier", "sources", "notes"}.
    """
    overrides = _load_overrides()
    entry = overrides.get(team.lower(), {}) if isinstance(overrides.get(team.lower()), dict) else {}

    sources = ["Days-rest / congestion model"]
    notes = []

    rest_score = _rest_component(days_rest)

    # Manual override: explicit user estimate (arg wins over file).
    if manual is None:
        manual = entry.get("readiness")
    if manual is not None:
        manual = max(0.0, min(1.0, float(manual)))
        score = 0.65 * manual + 0.35 * rest_score
        sources.append("Manual readiness estimate (user-provided)")
        notes.append(f"manual readiness estimate {manual:.2f}")
        if entry.get("note"):
            notes.append(str(entry["note"]))
    else:
        score = rest_score

    # Importance-weighted squad availability from PUBLIC injury news + roster.
    avail = get_squad_availability(team, news_sentiment, manual_out)
    if avail["flagged"]:
        # Availability directly scales the score: losing key players hurts most.
        score *= avail["availability"]
        sources.append("Public injury/availability news (player-weighted)")
        for fp in avail["flagged"]:
            tag = {1.0: "OUT", 0.5: "doubt", 0.3: "injury"}.get(fp["severity"], "concern")
            notes.append(f"{fp['player']} {tag} (imp {fp['importance']:.2f})")

    score = max(0.0, min(1.0, score))
    return {
        "score": round(score, 4),
        "multiplier": readiness_to_multiplier(score),
        "availability": avail["availability"],
        "flagged": avail["flagged"],
        "sources": sources,
        "notes": notes,
    }
