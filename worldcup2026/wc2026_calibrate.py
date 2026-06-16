"""
Calibrate WC2026 ratings from REAL data, combining TWO models + Phase-2 style.

Inputs: hicruben_results.json (913 internationals 2023-11..2026-06).

Produces wc2026_ratings.json with, per WC team:
  1. dc:   attack/defense from the repo's Dixon-Coles MLE   (scoreline shape)
  2. elo:  Hicruben's prior-anchored Elo, REPLICATED exactly (calibrate.mjs):
           seed priors -> form nudge (K by importance, 18mo half-life,
           goal-diff multiplier, host +75/2) -> 70% calibrated + 30% prior.
           Extended with 9 priors for draw teams he didn't seed (flagged).
  3. form: real last-25 goals-for/against + points-per-game
  4. style (PHASE 2, real-data proxies):
           - overall goal margin
           - margin vs STRONG opponents (prior >= 1850) => big-game delta
           - tempo (gf+ga) => open vs cagey style
  5. h2h:  real head-to-head per WC pairing
"""
import json, os, sys, math, datetime, re
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
# src/ may be a parent dir when this suite sits in worldcup2026/; add the nearest
# ancestor containing it so `from src...` resolves wherever this file lives.
_d = ROOT
for _ in range(4):
    if os.path.isdir(os.path.join(_d, "src")):
        sys.path.insert(0, _d); break
    _d = os.path.dirname(_d)
from src.models.dixon_coles import DixonColesModel

RESULTS = os.path.join(ROOT, "hicruben_results.json")
OUT = os.path.join(ROOT, "wc2026_ratings.json")
HALF_LIFE_DAYS = 550.0
FORM_WINDOW = 25
STRONG_PRIOR = 1850          # opponent prior >= this => "strong" (big-game split)

NAME_MAP = {  # WC group name -> dataset name
    "Korea Republic": "South Korea", "Bosnia": "Bosnia & Herzegovina",
    "Czechia": "Czech Republic", "Turkey": "Türkiye",
    "Cape Verde": "Cape Verde Islands", "DR Congo": "Congo DR",
    "Curacao": "Curaçao",
}
WC_TEAMS = [
    "Korea Republic","Mexico","Czechia","South Africa","Switzerland","Bosnia","Canada","Qatar",
    "Brazil","Morocco","Scotland","Haiti","Turkey","USA","Paraguay","Australia","Germany",
    "Ivory Coast","Ecuador","Curacao","Netherlands","Japan","Sweden","Tunisia","Belgium","Egypt",
    "Iran","New Zealand","Spain","Uruguay","Saudi Arabia","Cape Verde","France","Senegal","Norway",
    "Iraq","Argentina","Austria","Algeria","Jordan","Portugal","Colombia","Uzbekistan","DR Congo",
    "England","Croatia","Ghana","Panama",
]

# Hicruben SEED priors (calibrate.mjs), keyed by DATASET name. Last 9 are our extensions
# for draw teams he never seeded (clearly his-method, our-prior).
SEED = {
    "Argentina":2085,"France":2065,"Spain":2055,"Brazil":2045,"England":2000,"Portugal":1980,
    "Netherlands":1965,"Germany":1945,"Belgium":1925,"Colombia":1890,"Uruguay":1875,"Croatia":1870,
    "Morocco":1840,"Switzerland":1825,"USA":1830,"Mexico":1825,"Japan":1810,"Senegal":1795,
    "Ecuador":1760,"Australia":1735,"South Korea":1730,"Iran":1720,"Canada":1700,"Ghana":1665,
    "Tunisia":1655,"Ivory Coast":1655,"Saudi Arabia":1640,"Qatar":1630,"Egypt":1620,"Algeria":1615,
    "Scotland":1610,"Paraguay":1595,"Czech Republic":1570,"Bosnia & Herzegovina":1545,
    "South Africa":1520,"New Zealand":1495,"Panama":1480,"Jordan":1420,"Haiti":1380,
    # --- extensions (not in Hicruben's seed) ---
    "Türkiye":1800,"Norway":1770,"Austria":1760,"Sweden":1755,"Congo DR":1660,
    "Uzbekistan":1635,"Cape Verde Islands":1610,"Iraq":1600,"Curaçao":1520,
}
EXTENDED = {"Türkiye","Norway","Austria","Sweden","Congo DR","Uzbekistan","Cape Verde Islands","Iraq","Curaçao"}
HOST_DS = {"USA", "Mexico", "Canada"}
HOME_ADV = 75


# ---------- competition weights ----------
def dc_comp_w(league):
    l = (league or "").lower()
    if "friendl" in l: return 0.55
    if "nations league" in l: return 0.90
    if "world cup" in l: return 1.0
    if "qualif" in l: return 0.92
    if any(k in l for k in ("euro","copa","africa","asian","gold cup")): return 0.95
    return 0.85

def hic_baseK(league):  # exact from calibrate.mjs
    n = (league or "").lower()
    if re.search(r"world cup(?!.*qual)", n): return 55
    if re.search(r"world cup.*qual|qualification", n): return 40
    if re.search(r"copa america|euro championship\b|asian cup|africa cup|gold cup", n): return 50
    if re.search(r"nations league|nations cup", n): return 32
    if re.search(r"friendl", n): return 18
    return 28


def main():
    data = json.load(open(RESULTS, encoding="utf-8"))
    matches = [m for m in data["matches"] if m["hg"] is not None and m["ag"] is not None]
    matches.sort(key=lambda x: x["ts"])
    max_ts = matches[-1]["ts"]
    now_sec = max_ts

    # ===== 1. repo Dixon-Coles MLE =====
    fit_rows = [{
        "home_team": m["homeName"], "away_team": m["awayName"],
        "home_goals": m["hg"], "away_goals": m["ag"],
        "weight": (0.5 ** ((max_ts - m["ts"]) / 86400.0 / HALF_LIFE_DAYS)) * dc_comp_w(m["leagueName"]),
    } for m in matches]
    print(f"Fitting repo Dixon-Coles on {len(fit_rows)} matches...")
    dc = DixonColesModel(); dc.fit(fit_rows)
    print(f"  rho={dc.rho:.4f}")

    # ===== 2. Hicruben Elo (exact replication of calibrate.mjs), keyed by name =====
    R = {}
    def getR(name):
        if name not in R:
            R[name] = SEED.get(name, 1500)
        return R[name]
    gMult = lambda gd: (1 if abs(gd) <= 1 else (1.5 if abs(gd) == 2 else (11 + abs(gd)) / 8))
    exp_sc = lambda a, b, hb: 1 / (1 + 10 ** ((b - (a + hb)) / 400))
    for m in matches:
        ra, rb = getR(m["homeName"]), getR(m["awayName"])
        hb = HOME_ADV / 2 if m["homeName"] in HOST_DS else 0
        e = exp_sc(ra, rb, hb)
        sc = 1.0 if m["hg"] > m["ag"] else (0.0 if m["hg"] < m["ag"] else 0.5)
        recency = 0.5 ** (((now_sec - m["ts"]) / (30.44 * 86400)) / 18)
        k = hic_baseK(m["leagueName"]) * recency * gMult(m["hg"] - m["ag"])
        delta = k * (sc - e)
        R[m["homeName"]] = ra + delta
        R[m["awayName"]] = rb - delta
    hic_elo = {}
    for name in SEED:  # 70% calibrated + 30% prior
        hic_elo[name] = round(0.7 * R.get(name, SEED[name]) + 0.3 * SEED[name])

    # ===== 3+4. form + style (per dataset name) =====
    hist = defaultdict(list)
    for m in matches:
        hist[m["homeName"]].append((m["hg"], m["ag"], m["awayName"]))
        hist[m["awayName"]].append((m["ag"], m["hg"], m["homeName"]))

    def form_and_style(ds):
        rows = hist.get(ds, [])
        if not rows: return None, None
        last = rows[-FORM_WINDOW:]
        n = len(last)
        gf = sum(r[0] for r in last) / n
        ga = sum(r[1] for r in last) / n
        ppg = sum(3 if r[0] > r[1] else (1 if r[0] == r[1] else 0) for r in last) / n
        last6 = rows[-6:]
        ppg6 = sum(3 if r[0] > r[1] else (1 if r[0] == r[1] else 0) for r in last6) / len(last6)
        form = {"n": n, "gf": round(gf, 3), "ga": round(ga, 3), "ppg": round(ppg, 3),
                "ppg6": round(ppg6, 3)}
        # style: margin vs strong opponents over FULL history
        strong = [(f, a) for (f, a, opp) in rows if SEED.get(opp, 1500) >= STRONG_PRIOR]
        overall_margin = sum(f - a for (f, a, _) in rows) / len(rows)
        if strong:
            vs_strong_margin = sum(f - a for f, a in strong) / len(strong)
        else:
            vs_strong_margin = overall_margin
        style = {
            "tempo": round(gf + ga, 3),                              # open vs cagey
            "overall_margin": round(overall_margin, 3),
            "vs_strong_margin": round(vs_strong_margin, 3),
            "vs_strong_n": len(strong),
            "big_game_delta": round(vs_strong_margin - overall_margin, 3),  # raises/drops vs elite
            "mom": round(ppg6 - ppg, 3),                             # recent momentum vs baseline
        }
        return form, style

    # ===== 5. H2H =====
    ds_of = lambda t: NAME_MAP.get(t, t)
    wc_ds = {ds_of(t) for t in WC_TEAMS}
    h2h = defaultdict(lambda: {"a": 0, "b": 0, "d": 0, "agf": 0, "bgf": 0, "n": 0})
    for m in matches:
        h, a = m["homeName"], m["awayName"]
        if h in wc_ds and a in wc_ds:
            first = sorted([h, a])[0]
            fg, sg = (m["hg"], m["ag"]) if h == first else (m["ag"], m["hg"])
            rec = h2h["|".join(sorted([h, a]))]
            rec["agf"] += fg; rec["bgf"] += sg; rec["n"] += 1
            rec["a" if fg > sg else ("b" if fg < sg else "d")] += 1

    out = {
        "generatedAt": datetime.datetime.now(datetime.UTC).isoformat(),
        "source": "Hicruben results.json (913 internationals)",
        "models": "repo Dixon-Coles MLE (attack/def) + Hicruben prior-anchored Elo",
        "rho": dc.rho, "strong_prior": STRONG_PRIOR,
        "dc": {}, "elo": {}, "elo_extended": sorted(EXTENDED),
        "form": {}, "style": {}, "h2h": dict(h2h), "name_map": NAME_MAP,
    }
    for t in WC_TEAMS:
        ds = ds_of(t)
        if ds in dc.attack:
            out["dc"][t] = {"attack": round(dc.attack[ds], 4), "defense": round(dc.defense[ds], 4)}
        out["elo"][t] = hic_elo.get(ds)
        f, s = form_and_style(ds)
        out["form"][t] = f; out["style"][t] = s

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Wrote {OUT}")

    rank = sorted(((t, out["elo"][t]) for t in WC_TEAMS), key=lambda x: -x[1])
    print("\nHicruben-Elo ranking (top 16):")
    for t, e in rank[:16]:
        flag = " *" if ds_of(t) in EXTENDED else ""
        print(f"  {t:16s} {e}{flag}")


if __name__ == "__main__":
    main()
