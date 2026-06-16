"""
WC2026 per-match SCORELINES — blended model + 9-agent match-condition panel.

Pipeline per match:
  1. base lambdas = 45% repo Dixon-Coles fit + 55% Hicruben prior-anchored Elo
  2. 9-AGENT PANEL (wc2026_agents.run_panel) nudges each side's lambda:
       form, momentum, H2H, tactician/big-game, tempo, host/crowd,
       travel/rest, climate(hook), injury/news
  3. SCORELINE = most-likely score CONSISTENT WITH the predicted result
       (fixes the "everything is 1-1" artifact of naive argmax). Penalties are
       shown only when a DRAW is the predicted outcome in a knockout.

Usage:
  python wc2026_scorelines.py            # core run (agent 9 off)
  python wc2026_scorelines.py --news     # also fetch live squad news (agent 9 LIVE)
  python wc2026_scorelines.py --breakdown  # print full 9-agent breakdown per KO match
"""
import math, json, os, sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
# The shared Dixon-Coles lib lives in the repo's src/ package, which is a parent
# dir when this suite sits in worldcup2026/. Add the nearest ancestor containing
# src/ so `from src...` resolves wherever this file lives.
_d = ROOT
for _ in range(4):
    if os.path.isdir(os.path.join(_d, "src")):
        sys.path.insert(0, _d); break
    _d = os.path.dirname(_d)
from src.models.dixon_coles import _score_probability, _outcome_probs
import wc2026_agents as panel
import wc2026_players as players
import wc2026_tactics as tactics
import wc2026_matchsim as matchsim

R = json.load(open(os.path.join(ROOT, "wc2026_ratings.json"), encoding="utf-8"))
DC, ELO = R["dc"], R["elo"]
RHO = R["rho"]
W_DC = 0.45
HA_DC = 0.20
HB_ELO = 75
W_PLAYER = 0.30          # player/flank lane layer weight (analyst assumptions)
W_TACTICS = 1.0          # weight on additive tactical xG deltas
USE_LINEUPS = True       # blend the player/tactics layer by default
USE_EVENTSIM = True      # scorelines from the game-state event sim, not static Poisson
MAXG = 8
HOSTS = {"USA", "Mexico", "Canada"}

GROUPS = {
    "A": ["Korea Republic","Mexico","Czechia","South Africa"],
    "B": ["Switzerland","Bosnia","Canada","Qatar"],
    "C": ["Brazil","Morocco","Scotland","Haiti"],
    "D": ["Turkey","USA","Paraguay","Australia"],
    "E": ["Germany","Ivory Coast","Ecuador","Curacao"],
    "F": ["Netherlands","Japan","Sweden","Tunisia"],
    "G": ["Belgium","Egypt","Iran","New Zealand"],
    "H": ["Spain","Uruguay","Saudi Arabia","Cape Verde"],
    "I": ["France","Senegal","Norway","Iraq"],
    "J": ["Argentina","Austria","Algeria","Jordan"],
    "K": ["Portugal","Colombia","Uzbekistan","DR Congo"],
    "L": ["England","Croatia","Ghana","Panama"],
}
KO_ROUNDS = [
    ("Round of 16", [("France","Germany"),("Netherlands","Mexico"),("Brazil","Senegal"),
        ("England","Sweden"),("Spain","Croatia"),("Belgium","Turkey"),
        ("Argentina","USA"),("Portugal","Switzerland")]),
    ("Quarter-final", [("France","Netherlands"),("Spain","Belgium"),
        ("Brazil","England"),("Argentina","Portugal")]),
    ("Semi-final", [("France","Spain"),("Argentina","Brazil")]),
    ("3rd place", [("Brazil","Spain")]),
    ("FINAL", [("Argentina","France")]),
]


def _elo_lambda(rs, ro, hb):
    return max(0.3, min(3.5, 1.35 + (rs - ro + hb) / 400.0))


def base_lambdas(th, ta, neutral):
    dh, da = DC[th], DC[ta]
    eh, ea = ELO[th], ELO[ta]
    ha_dc = hb_h = hb_a = 0.0
    if not neutral:
        if th in HOSTS: ha_dc += HA_DC; hb_h += HB_ELO; hb_a -= HB_ELO / 2
        if ta in HOSTS: ha_dc -= HA_DC; hb_a += HB_ELO; hb_h -= HB_ELO / 2
    lam_dc_h = math.exp(dh["attack"] + da["defense"] + ha_dc)
    lam_dc_a = math.exp(da["attack"] + dh["defense"])
    lam_el_h = _elo_lambda(eh, ea, hb_h)
    lam_el_a = _elo_lambda(ea, eh, hb_a)
    return (W_DC * lam_dc_h + (1 - W_DC) * lam_el_h,
            W_DC * lam_dc_a + (1 - W_DC) * lam_el_a)


def _gamestate_profiles(info):
    """Build matchsim game-state profiles for both sides from the player layer."""
    ph, pa = info["ph"], info["pa"]
    th_thr = matchsim._transition_threat(ph["style"], ph["units"]["at"])
    ta_thr = matchsim._transition_threat(pa["style"], pa["units"]["at"])
    gh = matchsim.gamestate_profile(ph["style"], (ph["units"]["df"], ph["units"]["gk"]), th_thr)
    ga = matchsim.gamestate_profile(pa["style"], (pa["units"]["df"], pa["units"]["gk"]), ta_thr)
    return gh, ga


def match_lambdas(th, ta, neutral=True, knockout=False, news_scores=None, use_lineups=USE_LINEUPS):
    """Single source of truth for a match's expected goals:
    base blend -> 9-agent panel -> tactical interaction delta -> player/flank
    lane layer (W_PLAYER). `info` carries the full nuance bundle (lane duels,
    tactics reasons, game-state profiles) for the narrative + event sim."""
    lam_h, lam_a = base_lambdas(th, ta, neutral)
    # build player profiles up front so the panel's hydration agent can see the
    # coaches (and so we don't rebuild profiles inside player_xg).
    ph0 = pa0 = None
    coaches = None
    if use_lineups:
        ph0, pa0 = players.profile(th, ELO[th]), players.profile(ta, ELO[ta])
        coaches = (ph0["coach"], pa0["coach"])
    ctx = {"neutral": neutral, "knockout": knockout, "news_scores": news_scores,
           "coaches": coaches}
    mh, ma, breakdown = panel.run_panel(th, ta, R, ctx)
    lam_h *= mh; lam_a *= ma
    info = None
    if use_lineups:
        # player / flank lane layer
        plh, pla, pinfo = players.player_xg(th, ta, ELO[th], ELO[ta], ph=ph0, pa=pa0)
        # mechanistic tactical interaction (style vs style)
        tac = tactics.matchup(pinfo["ph"]["style"], pinfo["pa"]["style"], th, ta)
        dh, da = tac["delta"]
        plh = (plh + W_TACTICS * dh) * tac["cagey"]
        pla = (pla + W_TACTICS * da) * tac["cagey"]
        lam_h = (1 - W_PLAYER) * lam_h + W_PLAYER * plh
        lam_a = (1 - W_PLAYER) * lam_a + W_PLAYER * pla
        gh_prof, ga_prof = _gamestate_profiles(pinfo)
        info = {**pinfo, "tactics": tac, "gh_prof": gh_prof, "ga_prof": ga_prof}
    return max(0.12, lam_h), max(0.12, lam_a), (mh, ma), breakdown, info


import numpy as _np
_RNG = _np.random.default_rng(2026)


# --- probability calibration (senior-DS pass, see wc2026_eval.py) -----------
# On the live games the FULL stack is mildly OVER-CONFIDENT in favourites
# (predicts ~61% favourite-win vs the better-calibrated DC+Elo core's ~56%).
# We shrink the final 1X2 toward that core: wdl* = (1-s)*full + s*core.
# s is set by PRINCIPLE (mild de-confidence + ensemble variance reduction),
# NOT tuned to the n=10 sample — leave-one-out showed the argmin-s is unstable
# (0.73-1.0), i.e. tuning it would overfit. 0.35 keeps the nuance while
# trimming the over-sharpening. Set USE_CALIB=False to recover the raw stack.
USE_CALIB = True
CALIB_SHRINK = 0.35


def predict(th, ta, neutral=True, knockout=False, news_scores=None, use_lineups=USE_LINEUPS,
            eventsim=USE_EVENTSIM, n_sim=4000):
    lam_h, lam_a, (mh, ma), breakdown, info = match_lambdas(
        th, ta, neutral, knockout, news_scores, use_lineups)

    if eventsim and info:
        # game-state event sim -> scoreline distribution (captures park-the-bus,
        # chasing, comebacks) instead of a static Poisson grid.
        dist = matchsim.expected_distribution(
            lam_h, lam_a, info["gh_prof"], info["ga_prof"], _RNG, n=n_sim)
        wdl = {"home_win": dist["home_win"], "draw": dist["draw"], "away_win": dist["away_win"]}
        if USE_CALIB:
            bh0, ba0 = base_lambdas(th, ta, neutral)
            core = _outcome_probs(bh0, ba0, RHO)
            s = CALIB_SHRINK
            wdl = {k: (1 - s) * wdl[k] + s * core[k] for k in wdl}
        scores = dist["scores"]
        outcome = max(wdl, key=wdl.get)
        if outcome == "home_win":
            cand = {s: p for s, p in scores.items() if s[0] > s[1]}
        elif outcome == "away_win":
            cand = {s: p for s, p in scores.items() if s[0] < s[1]}
        else:
            cand = {s: p for s, p in scores.items() if s[0] == s[1]}
        score = max(cand, key=cand.get) if cand else (0, 0)
        p_score = cand.get(score, 0.0)
        info["_lam"] = (lam_h, lam_a)
        return {"score": score, "p_score": p_score, "outcome": outcome,
                "xg": (round(lam_h, 2), round(lam_a, 2)), "wdl": wdl,
                "mult": (round(mh, 3), round(ma, 3)), "breakdown": breakdown,
                "lineup": info, "dist": dist}

    grid = [((h, a), _score_probability(h, a, lam_h, lam_a, RHO))
            for h in range(MAXG + 1) for a in range(MAXG + 1)]
    tot = sum(p for _, p in grid)
    grid = [(s, p / tot) for s, p in grid]
    wdl = _outcome_probs(lam_h, lam_a, RHO)

    outcome = max(wdl, key=wdl.get)
    if outcome == "home_win":
        cand = [(s, p) for s, p in grid if s[0] > s[1]]
    elif outcome == "away_win":
        cand = [(s, p) for s, p in grid if s[0] < s[1]]
    else:
        cand = [(s, p) for s, p in grid if s[0] == s[1]]
    score = max(cand, key=lambda x: x[1])[0]
    p_score = max(cand, key=lambda x: x[1])[1]

    return {"score": score, "p_score": p_score, "outcome": outcome,
            "xg": (round(lam_h, 2), round(lam_a, 2)), "wdl": wdl,
            "mult": (round(mh, 3), round(ma, 3)), "breakdown": breakdown, "lineup": info}


def fixtures(teams):
    return [(teams[i], teams[j]) for i in range(4) for j in range(i + 1, 4)]


def winner_tag(th, ta, r):
    o = r["outcome"]
    return th if o == "home_win" else (ta if o == "away_win" else "draw")


def tier(c):
    """Result-confidence tier from probability of the predicted outcome (%)."""
    if c >= 65: return "STRONG "
    if c >= 50: return "CLEAR  "
    if c >= 40: return "LEAN   "
    return "TOSS-UP"


def fmt(th, ta, r, knockout=False):
    sh, sa = r["score"]
    conf = max(r["wdl"].values()) * 100            # result confidence
    psc = r["p_score"] * 100                        # scoreline confidence
    call = "DRAW->pens" if (knockout and r["outcome"] == "draw") else winner_tag(th, ta, r)
    return (f"{th:>15} {sh}-{sa} {ta:<15} | {call:<16} "
            f"| conf {conf:4.1f}% [{tier(conf)}] | scoreline {psc:4.1f}%")


def fetch_news(teams):
    print("Fetching live squad news (Google News RSS, agent 9)...", flush=True)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.data import news
    scores = {}
    def one(t):
        try:
            return t, news.get_sentiment(t).get("score", 0.0)
        except Exception:
            return t, 0.0
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(one, t): t for t in teams}
        for f in as_completed(futs, timeout=90):
            try:
                t, s = f.result(timeout=10); scores[t] = s
            except Exception:
                pass
    flagged = {t: s for t, s in scores.items() if abs(s) > 0.05}
    print(f"  got {len(scores)}/{len(teams)} teams; {len(flagged)} with notable news\n")
    return scores


def main():
    args = sys.argv[1:]
    do_news = "--news" in args
    do_bd = "--breakdown" in args
    use_lineups = "--no-lineups" not in args
    all_teams = [t for ts in GROUPS.values() for t in ts]
    news_scores = fetch_news(all_teams) if do_news else None

    print("=" * 104)
    print("FIFA WORLD CUP 2026 — SCORELINES  (45% DixonColes + 55% Elo -> 9-agent panel"
          + (" -> Phase-3 lineups" if use_lineups else "") + ")")
    print("Scoreline = most-likely score consistent with predicted result; PENS only for true draws.")
    print(f"Agent 9 (injury/news): {'LIVE' if do_news else 'OFF'} | "
          f"Phase-3 player/lineup layer: {'ON (assumed XIs)' if use_lineups else 'OFF'}")
    print("=" * 104)

    print("\n########## GROUP STAGE (72) ##########")
    for g, teams in GROUPS.items():
        print(f"\n--- Group {g} ---")
        for th, ta in fixtures(teams):
            neutral = not (th in HOSTS or ta in HOSTS)
            r = predict(th, ta, neutral, knockout=False, news_scores=news_scores, use_lineups=use_lineups)
            print(fmt(th, ta, r) + ("" if neutral else " [host]"))

    print("\n########## KNOCKOUT — CHALK PATH ##########")
    for rnd, ms in KO_ROUNDS:
        print(f"\n--- {rnd} ---")
        for th, ta in ms:
            r = predict(th, ta, neutral=True, knockout=True, news_scores=news_scores, use_lineups=use_lineups)
            print(fmt(th, ta, r, knockout=True))
            if use_lineups and r["lineup"]:
                import wc2026_narrative as narrative
                for line in narrative.analysis(th, ta, r):
                    print(line)
                if do_bd:
                    for line in narrative.flow(th, ta, r["lineup"], _RNG):
                        print(line)
            if do_bd:
                for name, amh, ama, note in r["breakdown"]:
                    if abs(amh - 1) > 0.005 or abs(ama - 1) > 0.005 or "LIVE" in note:
                        print(f"        {name:14s} x{amh:.3f}/{ama:.3f}  {note}")

    print("\nAgent panel: 1-6 LIVE | 7 PRIOR | 8 HOOK | 9 " + ("LIVE" if do_news else "OFF")
          + " || Phase-3 lineups: " + ("ON (assumed XIs, analyst judgment)" if use_lineups else "OFF"))


if __name__ == "__main__":
    main()
