"""Jeff Sackmann tennis data — public GitHub CSVs, no key needed."""
import io
import requests
import pandas as pd
from pathlib import Path
from src.data import cache

BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
WTA_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master"
LOCAL_DIR = Path(__file__).parent.parent.parent.parent / "data" / "cache" / "tennis"

SURFACES = {"Hard": "hard", "Clay": "clay", "Grass": "grass", "Carpet": "carpet"}


def _fetch_csv(url: str) -> pd.DataFrame:
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            return pd.read_csv(io.StringIO(resp.text), low_memory=False)
    except Exception:
        pass
    return pd.DataFrame()


def _get_matches(year: int, tour: str = "atp") -> pd.DataFrame:
    cached = cache.get("sackmann_csv", {"year": year, "tour": tour})
    if cached:
        return pd.DataFrame(cached)

    # Map tour names to their GitHub CSV URLs
    if tour == "atp":
        url = f"{BASE_URL}/atp_matches_{year}.csv"
    elif tour == "wta":
        url = f"{WTA_BASE}/wta_matches_{year}.csv"
    elif tour == "atp_chall":
        url = f"{BASE_URL}/atp_matches_qual_chall_{year}.csv"
    elif tour == "wta_itf":
        url = f"{WTA_BASE}/wta_matches_qual_itf_{year}.csv"
    else:
        url = f"{BASE_URL}/{tour}_matches_{year}.csv"

    df = _fetch_csv(url)

    if not df.empty:
        cache.set("sackmann_csv", {"year": year, "tour": tour}, df.to_dict("records"), ttl_seconds=86400 * 7)
    return df


def get_player_stats(player_name: str, months: int = 12, tour: str = "atp") -> dict:
    """Return rolling stats for a player over the last N months."""
    cached = cache.get("player_stats", {"name": player_name, "months": months, "tour": tour})
    if cached:
        return cached

    from datetime import datetime, timedelta
    import numpy as np

    cutoff = datetime.now() - timedelta(days=months * 30)
    current_year = datetime.now().year
    years = [current_year, current_year - 1]

    all_matches = []
    for year in years:
        df = _get_matches(year, tour)
        if df.empty:
            continue

        name_lower = player_name.lower()
        mask_w = df["winner_name"].str.lower().str.contains(name_lower, na=False)
        mask_l = df["loser_name"].str.lower().str.contains(name_lower, na=False)
        relevant = df[mask_w | mask_l].copy()

        if "tourney_date" in relevant.columns:
            relevant["date"] = pd.to_datetime(relevant["tourney_date"], format="%Y%m%d", errors="coerce")
            relevant = relevant[relevant["date"] >= cutoff]

        all_matches.append(relevant)

    if not all_matches:
        return _empty_stats(player_name)

    matches = pd.concat(all_matches, ignore_index=True)
    if matches.empty:
        return _empty_stats(player_name)

    name_lower = player_name.lower()

    stats = {}
    for surface_raw, surface_key in SURFACES.items():
        surf_df = matches[matches.get("surface", pd.Series(dtype=str)) == surface_raw] if "surface" in matches.columns else matches

        wins = surf_df[surf_df["winner_name"].str.lower().str.contains(name_lower, na=False)]
        losses = surf_df[surf_df["loser_name"].str.lower().str.contains(name_lower, na=False)]
        total = len(wins) + len(losses)

        if total == 0:
            stats[surface_key] = {"win_rate": 0.5, "matches": 0}
            continue

        win_rate = len(wins) / total

        # Serve stats from winner columns
        def safe_mean(series):
            return float(series.dropna().mean()) if len(series.dropna()) > 0 else None

        w_1st_pct = safe_mean(wins.get("w_1stIn", pd.Series(dtype=float)) / wins.get("w_svpt", pd.Series(dtype=float)).replace(0, float("nan")))
        w_ace_rate = safe_mean(wins.get("w_ace", pd.Series(dtype=float)) / wins.get("w_svpt", pd.Series(dtype=float)).replace(0, float("nan")))

        stats[surface_key] = {
            "win_rate": round(win_rate, 4),
            "matches": total,
            "first_serve_pct": round(w_1st_pct, 4) if w_1st_pct else 0.62,
            "ace_rate": round(w_ace_rate, 4) if w_ace_rate else 0.05,
        }

    wins_all = matches[matches["winner_name"].str.lower().str.contains(name_lower, na=False)]
    losses_all = matches[matches["loser_name"].str.lower().str.contains(name_lower, na=False)]
    total_all = len(wins_all) + len(losses_all)

    result = {
        "player": player_name,
        "overall_win_rate": round(len(wins_all) / total_all, 4) if total_all > 0 else 0.5,
        "total_matches": total_all,
        "surface_stats": stats,
        "recent_rank": _get_rank(matches, name_lower),
    }
    cache.set("player_stats", {"name": player_name, "months": months, "tour": tour}, result, ttl_seconds=3600 * 12)
    return result


def get_h2h(player1: str, player2: str, tour: str = "atp") -> dict:
    """Return head-to-head record between two players."""
    cached = cache.get("h2h_tennis", {"p1": player1, "p2": player2, "tour": tour})
    if cached:
        return cached

    from datetime import datetime
    current_year = datetime.now().year
    years = range(current_year - 5, current_year + 1)

    p1_wins = p2_wins = 0
    surface_records = {}

    for year in years:
        df = _get_matches(year, tour)
        if df.empty:
            continue

        p1_low, p2_low = player1.lower(), player2.lower()
        mask = (
            (df["winner_name"].str.lower().str.contains(p1_low, na=False) & df["loser_name"].str.lower().str.contains(p2_low, na=False)) |
            (df["winner_name"].str.lower().str.contains(p2_low, na=False) & df["loser_name"].str.lower().str.contains(p1_low, na=False))
        )
        h2h_df = df[mask]

        for _, row in h2h_df.iterrows():
            winner = str(row.get("winner_name", "")).lower()
            surf = str(row.get("surface", "Hard"))
            if p1_low in winner:
                p1_wins += 1
                surface_records[surf] = surface_records.get(surf, [0, 0])
                surface_records[surf][0] += 1
            else:
                p2_wins += 1
                surface_records[surf] = surface_records.get(surf, [0, 0])
                surface_records[surf][1] += 1

    total = p1_wins + p2_wins
    result = {
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "total": total,
        "p1_win_rate": round(p1_wins / total, 4) if total > 0 else 0.5,
        "surface_records": surface_records,
    }
    cache.set("h2h_tennis", {"p1": player1, "p2": player2, "tour": tour}, result, ttl_seconds=3600 * 12)
    return result


def _get_rank(matches: pd.DataFrame, name_lower: str) -> int:
    for col in ["winner_rank", "loser_rank"]:
        sub = matches[matches.get(col.replace("rank", "name"), pd.Series(dtype=str)).str.lower().str.contains(name_lower, na=False)]
        if not sub.empty and col in sub.columns:
            rank = sub[col].dropna()
            if len(rank) > 0:
                return int(rank.iloc[-1])
    return 999


def _empty_stats(player_name: str) -> dict:
    return {
        "player": player_name,
        "overall_win_rate": 0.5,
        "total_matches": 0,
        "surface_stats": {s: {"win_rate": 0.5, "matches": 0} for s in ["hard", "clay", "grass", "carpet"]},
        "recent_rank": 999,
    }
