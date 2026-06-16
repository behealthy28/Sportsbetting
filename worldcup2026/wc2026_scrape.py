"""
wc2026_scrape.py — keep the model's result set current.

Pulls finished 2026 World Cup group-stage results from Wikipedia (static HTML,
unlike Flashscore/Sofascore which are JS-walled), parses final scores, dedups
against wc2026_played.PLAYED, and prints any NEW finished fixtures as ready-to-
paste tuples. It NEVER fabricates: only rows with a real "X–Y" full-time score
are emitted, and in-progress / not-yet-played fixtures are skipped.

Usage:
    python wc2026_scrape.py            # show new finished results vs PLAYED
    python wc2026_scrape.py --all      # show every finished result it found
    python wc2026_scrape.py --write    # append new rows into wc2026_played.py

Requires:  pip install requests
"""
import sys, os, re, html, datetime, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

GROUPS = list("ABCDEFGHIJKL")
API = "https://en.wikipedia.org/w/api.php"

# Map Wikipedia's team naming to the names used inside the model (wc2026_*).
ALIAS = {
    "South Korea": "Korea Republic", "Korea Republic": "Korea Republic",
    "United States": "USA", "Czech Republic": "Czechia", "Czechia": "Czechia",
    "Bosnia and Herzegovina": "Bosnia", "Ivory Coast": "Ivory Coast",
    "Curaçao": "Curacao", "Curacao": "Curacao", "Cape Verde": "Cape Verde",
    "IR Iran": "Iran", "Türkiye": "Turkey", "Turkey": "Turkey",
}


def _norm(name):
    name = html.unescape(name).strip()
    return ALIAS.get(name, name)


def fetch_group_results(letter, session):
    """Return list of (home, hg, away, ag) finished results for one group."""
    title = f"2026 FIFA World Cup Group {letter}"
    try:
        r = session.get(API, params={
            "action": "parse", "page": title, "prop": "wikitext",
            "format": "json", "formatversion": "2",
        }, timeout=20)
        r.raise_for_status()
        wt = r.json()["parse"]["wikitext"]
    except Exception as e:
        print(f"  [group {letter}] skipped: {e}", file=sys.stderr)
        return []

    out = []
    # {{Football box ... |team1=X |score=2–1 |team2=Y ...}}  (en-dash score)
    for m in re.finditer(r"\{\{Football box(.*?)\}\}", wt, re.S):
        block = m.group(1)
        t1 = re.search(r"team1\s*=\s*([^\|\n}]+)", block)
        t2 = re.search(r"team2\s*=\s*([^\|\n}]+)", block)
        sc = re.search(r"score\s*=\s*(\d+)\s*[–\-]\s*(\d+)", block)
        if not (t1 and t2 and sc):     # no final score -> not finished, skip
            continue
        home = _norm(re.sub(r"\{\{.*?\}\}", "", t1.group(1)))
        away = _norm(re.sub(r"\{\{.*?\}\}", "", t2.group(1)))
        out.append((home, int(sc.group(1)), away, int(sc.group(2))))
    return out


def scrape_all():
    try:
        import requests
    except ImportError:
        sys.exit("Need requests:  pip install requests")
    s = requests.Session()
    s.headers["User-Agent"] = "wc2026-model/1.0 (personal prediction project)"
    results = []
    for g in GROUPS:
        results.extend(fetch_group_results(g, s))
    return results


def _key(t):
    """Orientation-independent fixture key."""
    return tuple(sorted([t[0], t[2]]))


def main():
    import wc2026_played as played
    found = scrape_all()
    known = {_key(t) for t in played.PLAYED}

    if "--all" in sys.argv:
        print(f"\n{len(found)} finished results on Wikipedia:")
        for h, hg, a, ag in found:
            print(f'    ("{h}", {hg}, "{a}", {ag}),')
        return

    new = [t for t in found if _key(t) not in known]
    if not new:
        print(f"\nUp to date — {len(found)} finished results found, all already "
              f"in wc2026_played.PLAYED ({len(played.PLAYED)} rows). Nothing new.")
        return

    print(f"\n{len(new)} NEW finished result(s) not yet in PLAYED:")
    rows = "".join(f'    ("{h}", {hg}, "{a}", {ag}),\n' for h, hg, a, ag in new)
    print(rows)

    if "--write" in sys.argv:
        path = os.path.join(os.path.dirname(__file__), "wc2026_played.py")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        stamp = datetime.date.today().isoformat()
        src = src.replace("\nPLAYED = [\n",
                          f"\nPLAYED = [\n    # ---- auto-added {stamp} ----\n{rows}", 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        print(f"  -> appended {len(new)} row(s) to wc2026_played.py. "
              f"Re-run wc2026_recalc.py to recondition the odds.")
    else:
        print("  (re-run with --write to append these into wc2026_played.py)")


if __name__ == "__main__":
    main()
