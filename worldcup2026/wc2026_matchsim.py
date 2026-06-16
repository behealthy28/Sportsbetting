"""
WC2026 GAME-STATE EVENT SIMULATOR — replaces the static two-Poisson draw.

A static Poisson draw CANNOT express how a match actually unfolds: a side that
goes 1-0 up and parks the bus, a favourite that chases an equaliser and gets hit
on the counter, a late comeback at the death. This module plays each match as a
sequence of PHASES. At every phase each side's scoring rate is its base lambda
(for that phase) modulated by the live GAME STATE:

  * a LEADER shuts up shop  -> own rate down (LEAD_DAMP), opponent's conversion
    down (SOLIDITY_DAMP x its defensive solidity): the "park the bus" effect.
  * a TRAILING side pushes   -> own rate up (CHASE_BOOST) but concedes more on
    the break (COUNTER_BOOST x the leader's transition threat).
  * effects grow with the MARGIN (LEAD_STEP) and how LATE it is (LATE_EXP).

MATCH CLOCK (why late goals happen): goals are NOT uniform across the match.
Scoring rate rises through the game and a real share of goals fall in 90'+
STOPPAGE time — long in modern football (officiating directive + VAR + subs) and
longer still at WC2026, where TWO mandatory hydration/cooling breaks (~22' and
~67') add ~5-6 minutes by themselves. We model 18 regulation phases on a gentle
late ramp plus one stoppage phase, with weights NORMALISED so the expected total
still equals the input lambda — the TIMING changes, not the total (no goal
inflation). Stoppage is played at full late-game state, so late equalisers and
winners occur at a realistic rate.

The CALIBRATION constants are tuned by wc2026_calibrate_gamestate.py against
real base rates (first-goal->win, lead-at-half->win, comeback/draw rates). Scalar
and vectorised paths share them, so a re-tune applies everywhere.

EPISTEMIC NOTE: game-state behaviour is analyst-tuned, not fitted to labelled
in-play data; it is calibrated so the *aggregate* lead/comeback statistics match
public base rates. While the game is level the phase totals sum to the input
lambda, so this does not silently inflate/deflate scoring.
"""
import numpy as np

# ---- CALIBRATION BLOCK (tuned in wc2026_calibrate_gamestate.py) -------------
# Re-tuned 2026-06-14 for the late-skewed + stoppage clock: first-goal->win 0.70,
# lead-at-half->win 0.76, draw 0.25 (targets 0.70/0.80/0.26). Half-lead-win lands
# a few points under the domestic 0.80 ON PURPOSE -- with realistic late/stoppage
# goals, half-time leads are genuinely less safe; forcing 0.80 wrecks the draw
# rate. ~0.76 is defensible for cagey, lower-scoring tournament football.
LEAD_DAMP = 0.20        # leader reduces its OWN attacking rate
SOLIDITY_DAMP = 0.22    # leader reduces the OPPONENT's conversion (x solidity)
CHASE_BOOST = 0.72      # trailing side lifts its OWN attacking rate
COUNTER_BOOST = 0.70    # leader's counter threat vs a committed trailer
LATE_EXP = 1.30         # how steeply game-state effects ramp toward full time
LEAD_STEP = 0.25        # extra management per goal of margin beyond the first
# ----------------------------------------------------------------------------

# ---- MATCH CLOCK: late-skewed timing + stoppage time -----------------------
REG_PHASES = 18                 # regulation, ~5 min each (0'-90')
PHASES = REG_PHASES + 1         # + one stoppage phase (90'+)
HALF = REG_PHASES // 2          # ~half-time index (~45')
LATE_SKEW = 0.30                # regulation ramp: late phases score a bit more than early
STOPPAGE_W = 1.25               # stoppage-phase weight (2 mandatory breaks + modern
                                #   officiating -> ~6% of goals fall in 90'+, realistic)
_RAMP = [1.0 + LATE_SKEW * ((k / (REG_PHASES - 1)) - 0.5) for k in range(REG_PHASES)]
PHASE_W = np.array(_RAMP + [STOPPAGE_W])              # per-phase scoring weight
W_SUM = float(PHASE_W.sum())                          # normaliser (keeps total = lambda)
PHASE_T = np.array([(k + 0.5) / REG_PHASES for k in range(REG_PHASES)] + [1.0])  # game-state clock
PHASE_MIN0 = [5 * k for k in range(REG_PHASES)] + [90]   # minute label start per phase
PHASE_LEN = [5] * REG_PHASES + [6]                       # minutes per phase (stoppage ~6')
# ----------------------------------------------------------------------------

# style -> (park: how hard a leader manages the game, chase: how hard a trailing
# side commits forward). 0..1.
STYLE_GAMESTATE = {
    "compact":    (0.95, 0.45),
    "counter":    (0.85, 0.55),
    "transition": (0.75, 0.65),
    "balanced":   (0.60, 0.60),
    "wing":       (0.55, 0.70),
    "possession": (0.45, 0.75),
    "press":      (0.35, 0.80),
}


def gamestate_profile(style, def_units, transition_threat):
    """Derive game-state behaviour params for one team.
    def_units: (df, gk) 0-100. transition_threat: 0..1 (counter danger)."""
    park, chase = STYLE_GAMESTATE.get(style, (0.6, 0.6))
    df, gk = def_units
    solidity = (0.6 * df + 0.4 * gk) / 100.0
    return {"park": park, "chase": chase, "solidity": solidity,
            "transition": transition_threat}


def _transition_threat(style, attack_unit):
    base = {"counter": .9, "transition": .82, "press": .7, "wing": .55,
            "balanced": .5, "possession": .3, "compact": .55}.get(style, .5)
    return 0.5 * base + 0.5 * (attack_unit / 100.0)


# ---- scalar path (single match, narrative) ---------------------------------
def _phase_rates(rh0, ra0, diff, t, gh_prof, ga_prof):
    """Adjust a phase's BASE rates (rh0/ra0) for the live game state.
    diff = home_goals - away_goals (home perspective); t in [0,1] elapsed."""
    rh, ra = rh0, ra0
    if diff == 0:
        return rh, ra
    m = min(abs(diff), 3)
    late = t ** LATE_EXP
    lead = (m - 1) * LEAD_STEP + 1.0
    if diff > 0:                         # HOME leads
        leader, trailer = gh_prof, ga_prof
        rh *= 1 - LEAD_DAMP * leader["park"] * late * lead
        rh *= 1 + COUNTER_BOOST * leader["transition"] * trailer["chase"] * late
        ra *= 1 - SOLIDITY_DAMP * leader["solidity"] * late
        ra *= 1 + CHASE_BOOST * trailer["chase"] * late
    else:                                # AWAY leads (mirror)
        leader, trailer = ga_prof, gh_prof
        ra *= 1 - LEAD_DAMP * leader["park"] * late * lead
        ra *= 1 + COUNTER_BOOST * leader["transition"] * trailer["chase"] * late
        rh *= 1 - SOLIDITY_DAMP * leader["solidity"] * late
        rh *= 1 + CHASE_BOOST * trailer["chase"] * late
    return max(0.001, rh), max(0.001, ra)


def simulate_scoreline(lam_h, lam_a, gh_prof, ga_prof, rng):
    """Fast scalar path: play the match, return (goals_h, goals_a)."""
    gh = ga = 0
    for k in range(PHASES):
        w = PHASE_W[k] / W_SUM
        rh, ra = _phase_rates(lam_h * w, lam_a * w, gh - ga, PHASE_T[k], gh_prof, ga_prof)
        gh += int(rng.poisson(rh))
        ga += int(rng.poisson(ra))
    return gh, ga


def simulate_timeline(lam_h, lam_a, gh_prof, ga_prof, rng, th="Home", ta="Away"):
    """Single-match path: returns final score + event timeline for narrative
    (minutes include 90'+ stoppage goals)."""
    gh = ga = 0
    events = []
    for k in range(PHASES):
        w = PHASE_W[k] / W_SUM
        rh, ra = _phase_rates(lam_h * w, lam_a * w, gh - ga, PHASE_T[k], gh_prof, ga_prof)
        nh, na = int(rng.poisson(rh)), int(rng.poisson(ra))
        m0, ln = PHASE_MIN0[k], PHASE_LEN[k]
        for _ in range(nh):
            gh += 1
            events.append((m0 + int(rng.integers(1, ln + 1)), th, gh, ga))
        for _ in range(na):
            ga += 1
            events.append((m0 + int(rng.integers(1, ln + 1)), ta, gh, ga))
    events.sort(key=lambda e: e[0])
    return gh, ga, events


def penalties(elo_h, elo_a, rng):
    """Knockout shootout: tilt by Elo. Returns 'home' or 'away'."""
    p = 1.0 / (1.0 + 10 ** (-(elo_h - elo_a) / 400.0))
    return "home" if rng.random() < p else "away"


# ---- vectorised path (Monte Carlo + fast single-match distribution) --------
def _arr(x, n):
    if np.isscalar(x):
        return np.full(n, float(x))
    return np.asarray(x, dtype=float)


def _profile_arrays(prof, n):
    return {k: _arr(prof[k], n) for k in ("park", "chase", "solidity", "transition")}


def phase_rates_vec(rh0, ra0, diff, t, H, A):
    """Vectorised per-phase rates. rh0/ra0 = this phase's base rates [N]; diff [N]
    int; H/A = dicts of length-N profile arrays."""
    late = t ** LATE_EXP
    mm = np.minimum(np.abs(diff), 3)
    lead = np.where(mm >= 1, (mm - 1) * LEAD_STEP + 1.0, 1.0)
    hl_rh = rh0 * (1 - LEAD_DAMP * H["park"] * late * lead) \
                * (1 + COUNTER_BOOST * H["transition"] * A["chase"] * late)
    hl_ra = ra0 * (1 - SOLIDITY_DAMP * H["solidity"] * late) \
                * (1 + CHASE_BOOST * A["chase"] * late)
    al_ra = ra0 * (1 - LEAD_DAMP * A["park"] * late * lead) \
                * (1 + COUNTER_BOOST * A["transition"] * H["chase"] * late)
    al_rh = rh0 * (1 - SOLIDITY_DAMP * A["solidity"] * late) \
                * (1 + CHASE_BOOST * H["chase"] * late)
    rh = np.where(diff > 0, hl_rh, np.where(diff < 0, al_rh, rh0))
    ra = np.where(diff > 0, hl_ra, np.where(diff < 0, al_ra, ra0))
    return np.maximum(0.001, rh), np.maximum(0.001, ra)


def simulate_scoreline_vec(lam_h, lam_a, gh_prof, ga_prof, rng, n):
    """Vectorised event sim for n parallel matches. lam_*/profiles may be scalars
    (same matchup n times) or length-n arrays (one per sim). Returns (gh[n], ga[n])."""
    lamh, lama = _arr(lam_h, n), _arr(lam_a, n)
    H, A = _profile_arrays(gh_prof, n), _profile_arrays(ga_prof, n)
    gh = np.zeros(n, dtype=np.int64)
    ga = np.zeros(n, dtype=np.int64)
    for k in range(PHASES):
        w = PHASE_W[k] / W_SUM
        rh, ra = phase_rates_vec(lamh * w, lama * w, gh - ga, PHASE_T[k], H, A)
        gh += rng.poisson(rh)
        ga += rng.poisson(ra)
    return gh, ga


def penalties_vec(elo_h, elo_a, rng, n):
    eh, ea = _arr(elo_h, n), _arr(elo_a, n)
    p = 1.0 / (1.0 + 10 ** (-(eh - ea) / 400.0))
    return rng.random(n) < p


def expected_distribution(lam_h, lam_a, gh_prof, ga_prof, rng, n=4000):
    """Monte-Carlo a single match n times (vectorised) -> outcome probs + scores."""
    gh, ga = simulate_scoreline_vec(lam_h, lam_a, gh_prof, ga_prof, rng, n)
    hw = float(np.mean(gh > ga)); aw = float(np.mean(ga > gh)); dw = 1.0 - hw - aw
    scores = {}
    pairs, counts = np.unique(np.stack([gh, ga], axis=1), axis=0, return_counts=True)
    for (a, b), c in zip(pairs, counts):
        scores[(int(a), int(b))] = c / n
    return {"home_win": hw, "draw": dw, "away_win": aw, "scores": scores, "n": n}


def measure(lam_h, lam_a, gh_prof, ga_prof, rng, n):
    """Instrumented vectorised sim for CALIBRATION. Returns raw counts:
    games, draws, total_goals, first-goal games + first-goal-scorer wins,
    half-lead games + half-leader wins."""
    lamh, lama = _arr(lam_h, n), _arr(lam_a, n)
    H, A = _profile_arrays(gh_prof, n), _profile_arrays(ga_prof, n)
    gh = np.zeros(n, dtype=np.int64); ga = np.zeros(n, dtype=np.int64)
    first = np.zeros(n, dtype=np.int8)
    half_diff = np.zeros(n, dtype=np.int64)
    for k in range(PHASES):
        w = PHASE_W[k] / W_SUM
        rh, ra = phase_rates_vec(lamh * w, lama * w, gh - ga, PHASE_T[k], H, A)
        nh, na = rng.poisson(rh), rng.poisson(ra)
        none = first == 0
        only_h = none & (nh > 0) & (na == 0)
        only_a = none & (na > 0) & (nh == 0)
        both = none & (nh > 0) & (na > 0)
        coin = rng.random(n) < 0.5
        first = np.where(only_h, 1, first)
        first = np.where(only_a, 2, first)
        first = np.where(both, np.where(coin, 1, 2), first)
        gh += nh; ga += na
        if k == HALF - 1:
            half_diff = gh - ga
    winner = np.where(gh > ga, 1, np.where(ga > gh, 2, 0))
    has_first = first != 0
    first_win = int(np.sum(has_first & (winner == first)))
    half_lead = half_diff != 0
    half_leader = np.where(half_diff > 0, 1, 2)
    half_win = int(np.sum(half_lead & (winner == half_leader)))
    return {"games": n, "draws": int(np.sum(gh == ga)),
            "goals": int(np.sum(gh + ga)),
            "n_first": int(np.sum(has_first)), "first_win": first_win,
            "n_half": int(np.sum(half_lead)), "half_win": half_win}


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    rng = np.random.default_rng(7)
    gh = gamestate_profile("compact", (84, 84), _transition_threat("compact", 79))
    ga = gamestate_profile("press", (78, 83), _transition_threat("press", 81))
    d = expected_distribution(1.3, 1.5, gh, ga, rng, n=20000)
    print("compact(1.3) vs press(1.5):", {k: round(v, 3) for k, v in d.items() if k != "scores"})
    m = measure(1.4, 1.4, gh, ga, rng, 40000)
    print(f"equal teams: first-goal-win {m['first_win']/m['n_first']:.3f}  "
          f"half-lead-win {m['half_win']/m['n_half']:.3f}  "
          f"draw {m['draws']/m['games']:.3f}  goals {m['goals']/m['games']:.2f}")
    print(f"stoppage share of weight: {STOPPAGE_W/W_SUM*100:.1f}%  "
          f"(last 20': {PHASE_W[15:].sum()/W_SUM*100:.0f}%)")
