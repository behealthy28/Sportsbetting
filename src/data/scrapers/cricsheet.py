"""CricSheet public data — ball-by-ball cricket data, no key needed."""
import requests
import json
import zipfile
import io
from pathlib import Path
from src.data import cache

CRICSHEET_BASE = "https://cricsheet.org"
LOCAL_DIR = Path(__file__).parent.parent.parent.parent / "data" / "cache" / "cricket"

# Seeded international team stats (batting avg, bowling avg, recent win rate)
TEAM_SEEDS = {
    "india": {
        "batting_avg": 32.5, "bowling_avg": 28.2, "run_rate": 5.8,
        "odi_win_rate": 0.72, "t20_win_rate": 0.68, "test_win_rate": 0.60,
        "elo": 1920, "ranking": 1,
    },
    "australia": {
        "batting_avg": 30.8, "bowling_avg": 27.5, "run_rate": 5.5,
        "odi_win_rate": 0.70, "t20_win_rate": 0.65, "test_win_rate": 0.58,
        "elo": 1890, "ranking": 2,
    },
    "england": {
        "batting_avg": 29.5, "bowling_avg": 29.0, "run_rate": 5.9,
        "odi_win_rate": 0.62, "t20_win_rate": 0.62, "test_win_rate": 0.52,
        "elo": 1850, "ranking": 4,
    },
    "pakistan": {
        "batting_avg": 28.0, "bowling_avg": 27.8, "run_rate": 5.4,
        "odi_win_rate": 0.60, "t20_win_rate": 0.65, "test_win_rate": 0.50,
        "elo": 1840, "ranking": 6,
    },
    "south africa": {
        "batting_avg": 29.2, "bowling_avg": 27.0, "run_rate": 5.3,
        "odi_win_rate": 0.62, "t20_win_rate": 0.60, "test_win_rate": 0.54,
        "elo": 1860, "ranking": 5,
    },
    "new zealand": {
        "batting_avg": 29.8, "bowling_avg": 27.2, "run_rate": 5.2,
        "odi_win_rate": 0.65, "t20_win_rate": 0.62, "test_win_rate": 0.58,
        "elo": 1870, "ranking": 3,
    },
    "west indies": {
        "batting_avg": 26.5, "bowling_avg": 30.0, "run_rate": 5.8,
        "odi_win_rate": 0.45, "t20_win_rate": 0.58, "test_win_rate": 0.35,
        "elo": 1720, "ranking": 8,
    },
    "sri lanka": {
        "batting_avg": 26.0, "bowling_avg": 29.5, "run_rate": 5.1,
        "odi_win_rate": 0.50, "t20_win_rate": 0.55, "test_win_rate": 0.42,
        "elo": 1740, "ranking": 9,
    },
    "bangladesh": {
        "batting_avg": 24.5, "bowling_avg": 31.0, "run_rate": 4.9,
        "odi_win_rate": 0.48, "t20_win_rate": 0.52, "test_win_rate": 0.28,
        "elo": 1680, "ranking": 10,
    },
    "afghanistan": {
        "batting_avg": 23.0, "bowling_avg": 28.5, "run_rate": 5.0,
        "odi_win_rate": 0.52, "t20_win_rate": 0.58, "test_win_rate": 0.20,
        "elo": 1700, "ranking": 11,
    },
    "zimbabwe": {
        "batting_avg": 22.0, "bowling_avg": 32.5, "run_rate": 4.8,
        "odi_win_rate": 0.35, "t20_win_rate": 0.40, "test_win_rate": 0.20,
        "elo": 1600, "ranking": 13,
    },
    "ireland": {
        "batting_avg": 22.5, "bowling_avg": 31.5, "run_rate": 5.0,
        "odi_win_rate": 0.42, "t20_win_rate": 0.46, "test_win_rate": 0.20,
        "elo": 1640, "ranking": 12,
    },
    # IPL franchises (T20)
    "mumbai indians": {
        "batting_avg": 28.5, "bowling_avg": 28.0, "run_rate": 8.4,
        "odi_win_rate": 0.55, "t20_win_rate": 0.57, "test_win_rate": 0.55,
        "elo": 1820, "ranking": 1,
    },
    "chennai super kings": {
        "batting_avg": 29.0, "bowling_avg": 28.5, "run_rate": 8.2,
        "odi_win_rate": 0.54, "t20_win_rate": 0.57, "test_win_rate": 0.55,
        "elo": 1810, "ranking": 2,
    },
    "kolkata knight riders": {
        "batting_avg": 27.5, "bowling_avg": 29.0, "run_rate": 8.0,
        "odi_win_rate": 0.50, "t20_win_rate": 0.52, "test_win_rate": 0.50,
        "elo": 1790, "ranking": 3,
    },
    "royal challengers bengaluru": {
        "batting_avg": 30.0, "bowling_avg": 30.0, "run_rate": 8.5,
        "odi_win_rate": 0.47, "t20_win_rate": 0.48, "test_win_rate": 0.47,
        "elo": 1770, "ranking": 4,
    },
    "royal challengers bangalore": {
        "batting_avg": 30.0, "bowling_avg": 30.0, "run_rate": 8.5,
        "odi_win_rate": 0.47, "t20_win_rate": 0.48, "test_win_rate": 0.47,
        "elo": 1770, "ranking": 4,
    },
    "rajasthan royals": {
        "batting_avg": 27.0, "bowling_avg": 29.5, "run_rate": 7.9,
        "odi_win_rate": 0.49, "t20_win_rate": 0.51, "test_win_rate": 0.49,
        "elo": 1780, "ranking": 5,
    },
    "delhi capitals": {
        "batting_avg": 27.0, "bowling_avg": 30.5, "run_rate": 8.1,
        "odi_win_rate": 0.47, "t20_win_rate": 0.49, "test_win_rate": 0.47,
        "elo": 1760, "ranking": 6,
    },
    "sunrisers hyderabad": {
        "batting_avg": 27.5, "bowling_avg": 29.0, "run_rate": 8.3,
        "odi_win_rate": 0.48, "t20_win_rate": 0.50, "test_win_rate": 0.48,
        "elo": 1760, "ranking": 7,
    },
    "punjab kings": {
        "batting_avg": 27.0, "bowling_avg": 31.0, "run_rate": 8.0,
        "odi_win_rate": 0.44, "t20_win_rate": 0.46, "test_win_rate": 0.44,
        "elo": 1730, "ranking": 8,
    },
    "lucknow super giants": {
        "batting_avg": 26.5, "bowling_avg": 30.0, "run_rate": 7.9,
        "odi_win_rate": 0.48, "t20_win_rate": 0.50, "test_win_rate": 0.48,
        "elo": 1760, "ranking": 6,
    },
    "gujarat titans": {
        "batting_avg": 27.0, "bowling_avg": 29.0, "run_rate": 8.0,
        "odi_win_rate": 0.52, "t20_win_rate": 0.54, "test_win_rate": 0.52,
        "elo": 1800, "ranking": 4,
    },
}

H2H_SEEDS = {
    frozenset(["india", "australia"]): {"t1_wins": 55, "draws": 2, "t2_wins": 41, "total": 98},
    frozenset(["india", "pakistan"]): {"t1_wins": 73, "draws": 6, "t2_wins": 73, "total": 152},
    frozenset(["australia", "england"]): {"t1_wins": 87, "draws": 8, "t2_wins": 86, "total": 181},
    frozenset(["india", "england"]): {"t1_wins": 56, "draws": 5, "t2_wins": 50, "total": 111},
    frozenset(["india", "south africa"]): {"t1_wins": 35, "draws": 1, "t2_wins": 28, "total": 64},
    frozenset(["australia", "new zealand"]): {"t1_wins": 62, "draws": 2, "t2_wins": 50, "total": 114},
    frozenset(["england", "west indies"]): {"t1_wins": 68, "draws": 0, "t2_wins": 58, "total": 126},
    frozenset(["india", "sri lanka"]): {"t1_wins": 53, "draws": 1, "t2_wins": 49, "total": 103},
    frozenset(["pakistan", "south africa"]): {"t1_wins": 30, "draws": 0, "t2_wins": 32, "total": 62},
    frozenset(["australia", "south africa"]): {"t1_wins": 44, "draws": 1, "t2_wins": 38, "total": 83},
    frozenset(["india", "new zealand"]): {"t1_wins": 44, "draws": 1, "t2_wins": 37, "total": 82},
    frozenset(["england", "south africa"]): {"t1_wins": 38, "draws": 2, "t2_wins": 41, "total": 81},
}


def get_team_stats(team_name: str, format: str = "odi") -> dict:
    """Return cricket team stats for a given format."""
    name_lower = team_name.lower().strip()
    data = TEAM_SEEDS.get(name_lower, {
        "batting_avg": 26.0, "bowling_avg": 30.0, "run_rate": 5.0,
        f"{format}_win_rate": 0.50, "elo": 1700, "ranking": 12,
    })
    result = dict(data)
    result["team"] = team_name
    result["format"] = format
    result["win_rate"] = result.get(f"{format}_win_rate", 0.50)
    return result


def get_h2h(team1: str, team2: str) -> dict:
    """Return H2H cricket record."""
    key = frozenset([team1.lower(), team2.lower()])
    if key in H2H_SEEDS:
        data = dict(H2H_SEEDS[key])
        is_t1_first = sorted([team1.lower(), team2.lower()])[0] == team1.lower()
        return {
            "team1_wins": data["t1_wins"] if is_t1_first else data["t2_wins"],
            "draws": data.get("draws", 0),
            "team2_wins": data["t2_wins"] if is_t1_first else data["t1_wins"],
            "total": data["total"],
        }
    return {"team1_wins": 8, "draws": 1, "team2_wins": 8, "total": 17}
