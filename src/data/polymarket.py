"""Polymarket sports market fetcher — shared between edges command and dashboard."""
from __future__ import annotations

import json as _json
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://polymarket.com/",
    "Origin": "https://polymarket.com",
}

_NON_SPORTS = [
    "election", "crypto", "bitcoin", "ethereum", "price", "president",
    "senate", "congress", "fed rate", "interest rate", "gdp", "war",
    "oscar", "emmy", "grammy", "nobel", "inflation", "stock", "trump",
    "biden", "harris", "elon", "spacex", "nasa",
]

_SPORT_KEYWORDS = [
    "nfl", "nba", "mlb", "nhl", "ncaab", "ncaaf", "ncaa",
    "ufc", " mma ", "boxing", "heavyweight", "lightweight",
    "premier league", "champions league", "europa league",
    "la liga", "serie a", "bundesliga", "ligue 1",
    "world cup", "copa america", "euro 2026", "euro 2025",
    "wimbledon", "us open", "french open", "australian open",
    "grand slam", " atp ", " wta ", "roland garros",
    "formula 1", " f1 ", "grand prix", "motogp",
    "cricket", " ipl ", "test match", "odi",
    "rugby", "six nations", "super rugby",
    "super bowl", "world series", "stanley cup", "nba finals",
]


def _gamma_page(offset: int, limit: int = 100) -> list:
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"limit": limit, "offset": offset,
                    "order": "endDateIso", "ascending": "true"},
            headers=_HEADERS, timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data if isinstance(data, list) else data.get("data", data.get("markets", []))
    except Exception:
        pass
    return []


def _ed(m: dict) -> str:
    return (m.get("endDateIso") or m.get("endDate") or "")[:10]


def fetch_sports_markets(days_ahead: int = 60) -> list:
    """
    Fetch upcoming sports markets from Polymarket gamma API.

    Binary-searches for the offset where endDateIso >= today, then pages forward
    collecting markets within the date window. Returns list of dicts:
      {question, end_date, probs {outcome: price}, volume}
    sorted by volume descending.
    """
    now = datetime.now(timezone.utc)
    now_str = now.strftime("%Y-%m-%d")
    cutoff_str = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # Binary-search for the starting offset where endDate >= today
    lo, hi = 0, 8000
    for _ in range(14):
        mid = (lo + hi) // 2
        page = _gamma_page(mid, limit=1)
        if not page:
            hi = mid
            continue
        if _ed(page[0]) < now_str:
            lo = mid + 1
        else:
            hi = mid
    start_offset = max(0, lo - 50)

    # Collect markets from start_offset until endDate > cutoff
    raw = []
    offset = start_offset
    consecutive_future = 0

    while offset < start_offset + 3000:
        batch = _gamma_page(offset, limit=100)
        if not batch:
            break

        for m in batch:
            ed = _ed(m)
            if not ed or ed < now_str:
                continue
            if ed > cutoff_str:
                consecutive_future += 1
                continue
            raw.append(m)
            consecutive_future = 0

        last = _ed(batch[-1])
        if last and last > cutoff_str:
            break
        offset += 100

    if not raw:
        return []

    # Deduplicate by question
    seen_q: set = set()
    deduped = []
    for m in raw:
        q = (m.get("question") or "").strip()
        if q and q not in seen_q:
            seen_q.add(q)
            deduped.append(m)

    out = []
    for m in deduped:
        q = (m.get("question") or "").lower()

        if any(kw in q for kw in _NON_SPORTS):
            continue

        has_vs = " vs " in q or " vs." in q
        has_sport_name = any(kw in q for kw in _SPORT_KEYWORDS)
        if not has_vs and not has_sport_name:
            continue

        probs: dict = {}
        tokens = m.get("tokens") or []
        if tokens:
            for tok in tokens:
                outcome = (tok.get("outcome") or "").lower()
                price = float(tok.get("price") or 0)
                if outcome:
                    probs[outcome] = price
        else:
            try:
                outcomes = m.get("outcomes") or "[]"
                prices = m.get("outcomePrices") or "[]"
                if isinstance(outcomes, str):
                    outcomes = _json.loads(outcomes)
                if isinstance(prices, str):
                    prices = _json.loads(prices)
                for o, p in zip(outcomes, prices):
                    probs[str(o).lower()] = float(p)
            except Exception:
                pass

        volume = float(m.get("volume_num") or m.get("volumeNum") or m.get("volume") or 0)
        end_date = (m.get("end_date_iso") or m.get("endDateIso") or m.get("endDate") or "")
        out.append({
            "question": m.get("question", ""),
            "end_date": end_date[:10] if end_date else "",
            "probs": probs,
            "volume": volume,
        })

    out.sort(key=lambda x: x["volume"], reverse=True)
    return out


def parse_matchup(question: str) -> tuple[Optional[str], Optional[str]]:
    """Extract (entity1, entity2) from a Polymarket question."""
    import re
    q = question.strip()

    sep = re.search(r'\s+vs?\.?\s+', q, re.IGNORECASE)
    if not sep:
        return None, None

    left = q[:sep.start()].strip()
    right = q[sep.end():].strip()

    if ":" in left:
        left = left.split(":")[-1].strip()
    right = re.split(r'\s*[:\|]\s*|\s+-\s+|\s+\d{4}-\d{2}-\d{2}', right)[0].strip()

    _PREFIX = (
        r'(?i)^(t20(\s+series)?|t10|odi(\s+series)?|test(\s+series|\s+match)?|'
        r'ipl|the\s+hundred|friendly|international\s+friendly|'
        r'premier\s+league|la\s+liga|serie\s+a|bundesliga|ligue\s+1|'
        r'champions\s+league|europa\s+league|ucl|uel|epl|mls)\s+'
    )
    prev = None
    while prev != left:
        prev = left
        left = re.sub(_PREFIX, '', left).strip()

    left = re.sub(r'(?i)^(will|who\s+wins|does|can)\s+', '', left).strip()
    right = re.sub(r'(?i)\s+(to\s+win|win|beat|wins?)\b.*$', '', right).strip()

    if 2 < len(left) < 40 and 2 < len(right) < 40:
        return left, right
    return None, None


def infer_sport(question: str, tags: list = None) -> str:
    """Infer sport from Polymarket question text and optional tags."""
    q = question.lower()
    t = " ".join(str(x) for x in (tags or [])).lower()
    c = q + " " + t

    if any(k in c for k in (
        "ufc", "mma", " bellator", "knockout", "submission", "octagon",
        "wins by", "fight night", "fight result",
    )):
        return "ufc"
    if any(k in c for k in (
        "tennis", " atp", " wta", "grand slam", "wimbledon", "us open",
        "french open", "australian open", "roland garros", "first set", "tiebreak",
    )):
        return "tennis"
    if any(k in c for k in (
        "boxing", "title bout", "weigh-in", "heavyweight title",
        "by decision", "by tko", "by ko",
    )):
        return "boxing"
    if any(k in c for k in (
        "cricket", " ipl", "test match", " odi", "t20", "the hundred",
        "most sixes", "most runs", "most wickets",
    )):
        return "cricket"
    return "football"
