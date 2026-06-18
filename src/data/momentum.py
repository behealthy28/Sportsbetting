"""Momentum features: winning/losing streaks and bounce-back behaviour.

Two ideas, computed from a team's recent REAL results:
  - current streak: consecutive wins (+) or losses (-), signed.
  - bounce-back: how a team responds after dropping points — its win rate in
    the games that immediately FOLLOW a draw or loss. A high value means the
    team reacts well to setbacks; a low value means setbacks tend to compound.

A single ``streak_features`` function is used by BOTH training (fed the
leak-free rolling history) and inference (fed recent real results from
team_last50.json), so the feature semantics are identical on both sides.
"""
import json
import os

_HIST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "historical", "team_last50.json",
)
_CACHE = None


def streak_features(games: list) -> dict:
    """Compute momentum features from a chronological list of results.

    Args:
        games: ordered oldest->newest, each a dict with "w" in {1.0, 0.5, 0.0}
               (win / draw / loss from this team's perspective).
    """
    if not games:
        return {"streak": 0, "streak_norm": 0.0, "bounceback": 0.5,
                "response_to_loss": 0.5}

    last = games[-1]["w"]
    streak = 0
    if last == 1.0:
        for g in reversed(games):
            if g["w"] == 1.0:
                streak += 1
            else:
                break
    elif last == 0.0:
        for g in reversed(games):
            if g["w"] == 0.0:
                streak -= 1
            else:
                break

    after_drop, after_loss = [], []
    for i in range(1, len(games)):
        prev = games[i - 1]["w"]
        won = 1.0 if games[i]["w"] == 1.0 else 0.0
        if prev < 1.0:               # came off a draw or loss
            after_drop.append(won)
        if prev == 0.0:              # came off a loss specifically
            after_loss.append(won)

    return {
        "streak": streak,
        "streak_norm": max(-1.0, min(1.0, streak / 5.0)),
        "bounceback": round(sum(after_drop) / len(after_drop), 3) if after_drop else 0.5,
        "response_to_loss": round(sum(after_loss) / len(after_loss), 3) if after_loss else 0.5,
    }


def _load() -> dict:
    global _CACHE
    if _CACHE is None:
        try:
            with open(_HIST_PATH, encoding="utf-8") as f:
                _CACHE = {k.lower(): v for k, v in json.load(f).items()}
        except (FileNotFoundError, json.JSONDecodeError):
            _CACHE = {}
    return _CACHE


def get_momentum(team: str) -> dict:
    """Momentum features for a team from its recent real results (inference)."""
    entry = _load().get(team.lower())
    if not entry:
        return streak_features([])
    games = []
    for m in entry.get("matches", []):
        if m["home"].lower() == team.lower():
            w = 1.0 if m["result"] == "home_win" else (0.5 if m["result"] == "draw" else 0.0)
        else:
            w = 1.0 if m["result"] == "away_win" else (0.5 if m["result"] == "draw" else 0.0)
        games.append({"w": w})
    return streak_features(games)
