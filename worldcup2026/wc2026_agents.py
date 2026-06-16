"""
10-AGENT MATCH-CONDITION PANEL (the quantified MiroFish-style swarm).

Each agent evaluates ONE match condition and returns multiplicative nudges
(mult_home, mult_away) on the expected-goals (lambda) of each side, plus a
short note. run_panel() combines them (product, then clamped) and returns the
full per-agent breakdown so the effect is transparent.

STATUS of each agent (honesty matters):
  1 Form Scout .......... LIVE  (last-25 goals for/against)
  2 Momentum Tracker .... LIVE  (last-6 vs last-25 points trend)
  3 H2H Psychologist .... LIVE  (historical head-to-head, >=3 meetings)
  4 Tactician ........... LIVE  (big-game level vs elite + attack/def edge)
  5 Tempo Analyst ....... LIVE  (open vs cagey style interaction)
  6 Host & Crowd Scout .. LIVE  (host nation + diaspora support)
  7 Travel & Rest ....... PRIOR (no fixture calendar -> host-travel edge only)
  8 Climate & Altitude .. HOOK  (no venue assignments -> neutral; pluggable)
  9 Injury & Availability LIVE* (Google News RSS via repo news.py; off unless enabled)
 10 Hydration Breaks ... PRIOR (WC2026 cooling-break rule; compresses mismatches)

Agents 1-6 are data-grounded. 7 and 10 are small documented priors. 8 is a neutral
hook (activating it without venue data would be fabrication). 9 is live only when
news fetching is enabled (pass news_scores), else neutral.
"""
import math

GF_BASE = 1.35
CLAMP_LO, CLAMP_HI = 0.65, 1.45     # max total per-side adjustment

# per-agent weights
W_FORM = 0.12
W_MOM = 0.06
W_H2H = 0.05
W_BIG = 0.08
W_TEMPO = 0.10
TEMPO_BASE = 2.70
W_NEWS = 0.6

HOSTS = {"USA", "Mexico", "Canada"}


def _n(w, x, lo=-2.0, hi=2.0):
    return math.exp(w * max(lo, min(hi, x)))


def agent_form(th, ta, R, ctx):
    fh, fa = R["form"].get(th), R["form"].get(ta)
    if not fh or not fa:
        return 1.0, 1.0, "no data"
    mh = ((fh["gf"] / GF_BASE) * (fa["ga"] / GF_BASE)) ** W_FORM
    ma = ((fa["gf"] / GF_BASE) * (fh["ga"] / GF_BASE)) ** W_FORM
    return mh, ma, f"gf25 {fh['gf']:.1f}/{fa['gf']:.1f} ga25 {fh['ga']:.1f}/{fa['ga']:.1f}"


def agent_momentum(th, ta, R, ctx):
    sh, sa = R["style"].get(th), R["style"].get(ta)
    if not sh or not sa:
        return 1.0, 1.0, "no data"
    mh, ma = _n(W_MOM, sh["mom"], -1.5, 1.5), _n(W_MOM, sa["mom"], -1.5, 1.5)
    return mh, ma, f"mom {sh['mom']:+.2f}/{sa['mom']:+.2f}"


def agent_h2h(th, ta, R, ctx):
    ds = lambda t: R["name_map"].get(t, t)
    rec = R["h2h"].get("|".join(sorted([ds(th), ds(ta)])))
    if not rec or rec["n"] < 3:
        return 1.0, 1.0, "<3 meetings"
    first = sorted([ds(th), ds(ta)])[0]
    gd = (rec["agf"] - rec["bgf"]) / rec["n"]
    gd_h = gd if ds(th) == first else -gd
    return _n(W_H2H, gd_h), _n(W_H2H, -gd_h), f"h2h gd/g {gd_h:+.2f} (n={rec['n']})"


def agent_tactician(th, ta, R, ctx):
    """Big-game adaptation: does a side raise/drop its level vs elite? Fires in
    knockouts or against strong opponents (where tactics/coaching matter most)."""
    sh, sa = R["style"].get(th), R["style"].get(ta)
    strong = R["strong_prior"]
    ko = ctx.get("knockout", False)
    mh = ma = 1.0
    note = []
    if sh and sh["vs_strong_n"] >= 3 and (ko or R["elo"].get(ta, 1500) >= strong):
        mh = _n(W_BIG, sh["big_game_delta"]); note.append(f"H {sh['big_game_delta']:+.2f}")
    if sa and sa["vs_strong_n"] >= 3 and (ko or R["elo"].get(th, 1500) >= strong):
        ma = _n(W_BIG, sa["big_game_delta"]); note.append(f"A {sa['big_game_delta']:+.2f}")
    return mh, ma, ("bigGame " + " ".join(note)) if note else "inactive (not vs elite)"


def agent_tempo(th, ta, R, ctx):
    sh, sa = R["style"].get(th), R["style"].get(ta)
    if not sh or not sa:
        return 1.0, 1.0, "no data"
    pair = (sh["tempo"] + sa["tempo"]) / 2.0
    m = (pair / TEMPO_BASE) ** W_TEMPO
    return m, m, f"tempo {pair:.2f} ({'open' if m>1 else 'cagey'})"


def agent_host(th, ta, R, ctx):
    """Host nation + diaspora. Mexico carries large support across US venues even
    when nominally neutral; USA/Canada get a crowd edge at host venues."""
    mh = ma = 1.0
    if th == "Mexico":
        mh *= math.exp(0.05)
    if ta == "Mexico":
        ma *= math.exp(0.05)
    if not ctx.get("neutral", True):
        if th in HOSTS: mh *= math.exp(0.03)
        if ta in HOSTS: ma *= math.exp(0.03)
    note = "host/diaspora" if (mh != 1.0 or ma != 1.0) else "neutral venue"
    return mh, ma, note


def agent_travel(th, ta, R, ctx):
    """PRIOR: without the fixture calendar we can't model congestion/jet-lag.
    Hosts travel less within North America -> tiny edge. Everyone else neutral."""
    mh = math.exp(0.02) if th in HOSTS else 1.0
    ma = math.exp(0.02) if ta in HOSTS else 1.0
    return mh, ma, "host-travel edge" if (mh != 1.0 or ma != 1.0) else "neutral (no schedule)"


def agent_climate(th, ta, R, ctx):
    """HOOK: heat/altitude depends on the assigned venue (Mexico City 2240m,
    US-south heat). No venue map -> neutral. Pluggable when venues are known."""
    return 1.0, 1.0, "neutral (no venue map)"


def agent_injury(th, ta, R, ctx):
    """LIVE only when news_scores supplied (Google News RSS via repo news.py).
    Negative squad news -> lower lambda for that side."""
    ns = ctx.get("news_scores")
    if not ns:
        return 1.0, 1.0, "off (no news feed)"
    sh, sa = ns.get(th, 0.0), ns.get(ta, 0.0)
    return _n(W_NEWS, sh, -0.5, 0.3), _n(W_NEWS, sa, -0.5, 0.3), f"news {sh:+.2f}/{sa:+.2f}"


def agent_hydration(th, ta, R, ctx):
    """WC2026 HYDRATION / COOLING BREAKS (hot US/Mexico venues). A break is a
    RESET available to BOTH sides, so it is NOT an underdog handout. The pre-match
    edge it confers goes to the better-COACHED side (more value extracted from the
    extra in-game reset windows) — bidirectional and TINY, because we have no
    in-play or temperature data to fit it (pure prior). The in-game effects
    (momentum interruption + the leader firming up its block at the 67' break)
    live in the event sim, wc2026_matchsim."""
    coaches = ctx.get("coaches")
    if not coaches:
        return 1.0, 1.0, "off (no coach data)"
    ch, ca = coaches
    f = max(-0.035, min(0.035, (ch - ca) / 100.0 * 0.18))
    mh, ma = math.exp(f), math.exp(-f)
    note = (f"cooling-break resets favour the better coach ({ch:.0f} v {ca:.0f})"
            if abs(f) > 0.003 else f"coaches level ({ch:.0f} v {ca:.0f}) -> minimal")
    return mh, ma, note


AGENTS = [
    ("1 Form Scout", agent_form),
    ("2 Momentum", agent_momentum),
    ("3 H2H Psych", agent_h2h),
    ("4 Tactician", agent_tactician),
    ("5 Tempo", agent_tempo),
    ("6 Host/Crowd", agent_host),
    ("7 Travel/Rest", agent_travel),
    ("8 Climate/Alt", agent_climate),
    ("9 Injury/News", agent_injury),
    ("10 Hydration", agent_hydration),
]


def run_panel(th, ta, R, ctx):
    mult_h = mult_a = 1.0
    breakdown = []
    for name, fn in AGENTS:
        mh, ma, note = fn(th, ta, R, ctx)
        mult_h *= mh; mult_a *= ma
        breakdown.append((name, mh, ma, note))
    mult_h = max(CLAMP_LO, min(CLAMP_HI, mult_h))
    mult_a = max(CLAMP_LO, min(CLAMP_HI, mult_a))
    return mult_h, mult_a, breakdown
