"""
WC2026 Monte Carlo — title odds from the BLENDED ratings + 9-agent panel +
flank/player lanes + tactical engine, run through the GAME-STATE EVENT SIM.

Each simulation plays all 72 group games (18-phase event sim, so leads get
protected and trailing sides chase), builds standings + the 8 best third-placed
teams, seeds an Elo bracket, and plays the knockout — counting how often each
team wins.

This module is VECTORISED: instead of looping one tournament at a time in
Python, it advances all N simulations together with numpy (group fixtures, group
standings, best-third selection, Elo seeding, and every knockout round are array
ops). That makes N=20,000 event-sim tournaments run in seconds. A scalar
reference implementation (simulate_scalar) is kept for cross-validation.

Caveat (unchanged): the knockout uses a SEEDED bracket (top teams spread into
opposite halves), which favours the favourites slightly vs a real fixed-slot
draw. Title ORDER is robust.
"""
import sys, os, random
import numpy as np
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import wc2026_scorelines as S
import wc2026_players as players
import wc2026_matchsim as matchsim

R, HOSTS, ELO = S.R, S.HOSTS, S.ELO
rng = np.random.default_rng(42)
N = 20000

TEAMS = [t for ts in S.GROUPS.values() for t in ts]   # 48 teams, group order
IDX = {t: i for i, t in enumerate(TEAMS)}
NT = len(TEAMS)


def bracket_order(n):
    order = [1, 2]
    while len(order) < n:
        m = len(order) * 2
        order = [x for s in order for x in (s, m + 1 - s)]
    return order


# ======================================================================
# SCALAR REFERENCE PATH (kept for validation; not used by simulate)
# ======================================================================
_lam_cache = {}
def setup(th, ta, neutral, knockout):
    key = (th, ta, neutral, knockout)
    if key not in _lam_cache:
        lh, la, _, _, info = S.match_lambdas(th, ta, neutral, knockout, news_scores=None,
                                             use_lineups=S.USE_LINEUPS)
        if info:
            ghp, gap = info["gh_prof"], info["ga_prof"]
        else:
            ghp = matchsim.gamestate_profile("balanced", (75, 75), 0.5); gap = ghp
        _lam_cache[key] = (lh, la, ghp, gap)
    return _lam_cache[key]


def sim_group_scalar(teams):
    pts = {t: 0 for t in teams}; gd = {t: 0 for t in teams}; gf = {t: 0 for t in teams}
    for i in range(4):
        for j in range(i + 1, 4):
            th, ta = teams[i], teams[j]
            neutral = not (th in HOSTS or ta in HOSTS)
            lh, la, ghp, gap = setup(th, ta, neutral, False)
            gh, ga = matchsim.simulate_scoreline(lh, la, ghp, gap, rng)
            gf[th] += gh; gf[ta] += ga; gd[th] += gh - ga; gd[ta] += ga - gh
            if gh > ga: pts[th] += 3
            elif gh < ga: pts[ta] += 3
            else: pts[th] += 1; pts[ta] += 1
    return sorted(teams, key=lambda t: (pts[t], gd[t], gf[t], random.random()), reverse=True), pts, gd, gf


def ko_winner_scalar(a, b):
    lh, la, ghp, gap = setup(a, b, True, True)
    gh, ga = matchsim.simulate_scoreline(lh, la, ghp, gap, rng)
    if gh > ga: return a
    if ga > gh: return b
    return a if matchsim.penalties(ELO[a], ELO[b], rng) == "home" else b


def simulate_scalar(n=2000):
    champ = {}; finalist = {}; semi = {}; advance = {}
    rn = ["R16", "QF", "SF", "F", "CHAMP"]
    for _ in range(n):
        winners, runners, thirds = [], [], []
        for g, teams in S.GROUPS.items():
            ranked, pts, gd, gf = sim_group_scalar(teams)
            winners.append(ranked[0]); runners.append(ranked[1])
            t = ranked[2]; thirds.append((t, pts[t], gd[t], gf[t]))
        best = [x[0] for x in sorted(thirds, key=lambda z: (z[1], z[2], z[3], random.random()), reverse=True)[:8]]
        quals = winners + runners + best
        seeded = sorted(quals, key=lambda t: ELO[t], reverse=True)
        slots = [seeded[s - 1] for s in bracket_order(32)]
        reached = {t: "R32" for t in quals}
        rd = 0
        while len(slots) > 1:
            nxt = []
            for i in range(0, len(slots), 2):
                w = ko_winner_scalar(slots[i], slots[i + 1]); nxt.append(w); reached[w] = rn[rd]
            slots = nxt; rd += 1
        champ[slots[0]] = champ.get(slots[0], 0) + 1
        for t in quals: advance[t] = advance.get(t, 0) + 1
        for t, r in reached.items():
            if r in ("F", "CHAMP"): finalist[t] = finalist.get(t, 0) + 1
            if r in ("SF", "F", "CHAMP"): semi[t] = semi.get(t, 0) + 1
    return {"n": n, "champ": champ, "finalist": finalist, "semi": semi, "advance": advance}


# ======================================================================
# VECTORISED PATH
# ======================================================================
_TABLES = None
def build_tables():
    """Precompute per-team game-state profiles + Elo, the KO lambda matrices for
    all ordered pairs, and the per-fixture group lambdas. One-time cost."""
    global _TABLES
    if _TABLES is not None:
        return _TABLES
    park = np.zeros(NT); chase = np.zeros(NT); sol = np.zeros(NT); trans = np.zeros(NT)
    elo = np.zeros(NT)
    for t, i in IDX.items():
        p = players.profile(t, ELO[t]); u = p["units"]
        thr = matchsim._transition_threat(p["style"], u["at"])
        gp = matchsim.gamestate_profile(p["style"], (u["df"], u["gk"]), thr)
        park[i], chase[i], sol[i], trans[i] = gp["park"], gp["chase"], gp["solidity"], gp["transition"]
        elo[i] = ELO[t]
    PROF = {"park": park, "chase": chase, "solidity": sol, "transition": trans}

    LAMH = np.zeros((NT, NT)); LAMA = np.zeros((NT, NT))
    for a in range(NT):
        for b in range(NT):
            if a == b:
                continue
            lh, la, _, _, _ = S.match_lambdas(TEAMS[a], TEAMS[b], neutral=True,
                                              knockout=True, use_lineups=S.USE_LINEUPS)
            LAMH[a, b] = lh; LAMA[a, b] = la

    groups = []
    for g, teams in S.GROUPS.items():
        gi = np.array([IDX[t] for t in teams])
        fixt = []
        for ii in range(4):
            for jj in range(ii + 1, 4):
                th, ta = teams[ii], teams[jj]
                neutral = not (th in HOSTS or ta in HOSTS)
                lh, la, _, _, _ = S.match_lambdas(th, ta, neutral=neutral,
                                                  knockout=False, use_lineups=S.USE_LINEUPS)
                fixt.append((lh, la, ii, jj))
        groups.append((gi, fixt))
    _TABLES = dict(PROF=PROF, LAMH=LAMH, LAMA=LAMA, elo=elo, groups=groups)
    return _TABLES


def _gather(idx):
    P = _TABLES["PROF"]
    return {k: P[k][idx] for k in P}


def _play_round(slots):
    """slots [n, k] global team indices (k even). Returns winners [n, k//2]."""
    n, k = slots.shape
    A = slots[:, 0::2].reshape(-1)
    B = slots[:, 1::2].reshape(-1)
    lh = _TABLES["LAMH"][A, B]; la = _TABLES["LAMA"][A, B]
    gh, ga = matchsim.simulate_scoreline_vec(lh, la, _gather(A), _gather(B), rng, len(A))
    pen_home = matchsim.penalties_vec(_TABLES["elo"][A], _TABLES["elo"][B], rng, len(A))
    win = np.where(gh > ga, A, np.where(ga > gh, B, np.where(pen_home, A, B)))
    return win.reshape(n, k // 2)


def _rank_key(pts, gd, gf, rng_):
    # pts (max 9) >> gd (shifted +50) >> gf >> random tiebreak
    return pts * 1e7 + (gd + 50.0) * 1e4 + gf * 10.0 + rng_.random(pts.shape)


def played_lookup(games):
    """games: list of (home, home_goals, away, away_goals). Returns a dict keyed
    by BOTH orientations -> oriented (gh, ga) so fixtures match either way."""
    d = {}
    for a, sa, b, sb in games:
        d[(a, b)] = (sa, sb)
        d[(b, a)] = (sb, sa)
    return d


def simulate(n=N, played=None):
    """Monte Carlo title odds. If `played` (a played_lookup dict) is given, those
    group fixtures are PINNED to their real result instead of simulated, so the
    odds are conditioned on what has actually happened."""
    T = build_tables(); PROF = T["PROF"]
    played = played or {}
    winners, runners, thirds, tkeys = [], [], [], []
    for gi, fixt in T["groups"]:
        pts = np.zeros((n, 4)); gd = np.zeros((n, 4)); gf = np.zeros((n, 4))
        for lh, la, hi, ai in fixt:
            key = (TEAMS[int(gi[hi])], TEAMS[int(gi[ai])])
            if key in played:
                gv, av = played[key]
                gh = np.full(n, gv, dtype=np.int64)
                ga = np.full(n, av, dtype=np.int64)
            else:
                ph = {kk: PROF[kk][gi[hi]] for kk in PROF}
                pa = {kk: PROF[kk][gi[ai]] for kk in PROF}
                gh, ga = matchsim.simulate_scoreline_vec(lh, la, ph, pa, rng, n)
            gf[:, hi] += gh; gf[:, ai] += ga
            gd[:, hi] += gh - ga; gd[:, ai] += ga - gh
            hw = gh > ga; aw = ga > gh; dr = gh == ga
            pts[:, hi] += np.where(hw, 3, np.where(dr, 1, 0))
            pts[:, ai] += np.where(aw, 3, np.where(dr, 1, 0))
        order = np.argsort(-_rank_key(pts, gd, gf, rng), axis=1)        # [n,4] local
        rank_global = gi[order]                                         # [n,4] global
        winners.append(rank_global[:, 0]); runners.append(rank_global[:, 1])
        third_local = order[:, 2]
        thirds.append(rank_global[:, 2])
        tp = np.take_along_axis(pts, third_local[:, None], 1)[:, 0]
        tgd = np.take_along_axis(gd, third_local[:, None], 1)[:, 0]
        tgf = np.take_along_axis(gf, third_local[:, None], 1)[:, 0]
        tkeys.append(tp * 1e7 + (tgd + 50.0) * 1e4 + tgf * 10.0)
    winners = np.stack(winners, 1); runners = np.stack(runners, 1)      # [n,12]
    thirds = np.stack(thirds, 1)                                        # [n,12]
    tkey = np.stack(tkeys, 1) + rng.random((n, 12))
    best8 = np.take_along_axis(thirds, np.argsort(-tkey, axis=1)[:, :8], 1)  # [n,8]
    quals = np.concatenate([winners, runners, best8], axis=1)          # [n,32]

    # Elo-seeded bracket
    seeded = np.take_along_axis(quals, np.argsort(-T["elo"][quals], axis=1), 1)
    slots = seeded[:, np.array(bracket_order(32)) - 1]                 # [n,32]

    advance = np.bincount(quals.reshape(-1), minlength=NT)
    rounds = []
    cur = slots
    for _ in range(5):
        cur = _play_round(cur); rounds.append(cur)
    r16 = np.bincount(rounds[0].reshape(-1), minlength=NT)             # reached R16 (16)
    qf = np.bincount(rounds[1].reshape(-1), minlength=NT)              # reached QF (8)
    semi = np.bincount(rounds[2].reshape(-1), minlength=NT)            # reached SF (4)
    finalist = np.bincount(rounds[3].reshape(-1), minlength=NT)        # finalists (2)
    champ = np.bincount(rounds[4].reshape(-1), minlength=NT)           # winner (1)

    tod = lambda arr: {TEAMS[i]: int(arr[i]) for i in range(NT) if arr[i] > 0}
    return {"n": n, "champ": tod(champ), "finalist": tod(finalist),
            "semi": tod(semi), "qf": tod(qf), "r16": tod(r16), "advance": tod(advance)}


def main():
    use_scalar = "--scalar" in sys.argv
    n = N if not use_scalar else 2000
    res = simulate_scalar(n) if use_scalar else simulate(N)
    champ, finalist, semi, advance = res["champ"], res["finalist"], res["semi"], res["advance"]
    nn = res["n"]
    rows = sorted(TEAMS, key=lambda t: champ.get(t, 0), reverse=True)
    mode = "SCALAR ref" if use_scalar else "vectorised"
    print(f"WC2026 Monte Carlo — {nn:,} game-state event-sim tournaments ({mode})\n")
    print(f"{'Team':16s}{'WIN%':>7}{'FINAL%':>8}{'SEMI%':>7}{'ADV(R32)%':>11}{'Elo':>7}")
    for t in rows[:16]:
        print(f"{t:16s}{champ.get(t,0)/nn*100:6.1f}%{finalist.get(t,0)/nn*100:7.1f}%"
              f"{semi.get(t,0)/nn*100:6.1f}%{advance.get(t,0)/nn*100:10.1f}%{ELO[t]:7d}")


if __name__ == "__main__":
    main()
