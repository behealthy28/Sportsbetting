"""
Calibrate the game-state event sim (wc2026_matchsim) against real-football base
rates. We sample a representative set of WC2026 matchups (so the lambda spread
matches the real field: group mismatches + even knockout ties), play each many
times with the instrumented sim, and aggregate:

  * P(win | scored first)        real-world aggregate ~ 0.70
  * P(win | leading at half)     real-world aggregate ~ 0.80
  * draw rate (90 min)           internationals ~ 0.24-0.28
  * goals/game                   must stay ~ the model's input total (no drift)

Targets are public aggregates across professional football (the first goal
scorer wins ~70% of matches; a side leading at half-time wins ~80%). They are
aggregates over a realistic mix of mismatches and even games, which is why we
sample the actual WC field rather than only equal teams.

Usage:
  python wc2026_calibrate_gamestate.py            # measure current constants
  python wc2026_calibrate_gamestate.py --tune     # grid-search + recommend
"""
import sys, os, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wc2026_scorelines as S
import wc2026_matchsim as ms
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

TARGETS = {"first_win": 0.70, "half_win": 0.80, "draw": 0.26}
ALL = [t for ts in S.GROUPS.values() for t in ts]


def sample_matchups(k=120, seed=1):
    """k representative matchups with their lambdas + game-state profiles.
    Mix: half random pairs (group-like, includes mismatches), half top-24 pairs
    (knockout-like, more even)."""
    rng = np.random.default_rng(seed)
    top = sorted(ALL, key=lambda t: S.ELO[t], reverse=True)[:24]
    out = []
    for i in range(k):
        pool = ALL if i % 2 == 0 else top
        a, b = rng.choice(pool, size=2, replace=False)
        lam_h, lam_a, _, _, info = S.match_lambdas(str(a), str(b), neutral=True,
                                                    knockout=False, use_lineups=True)
        if info:
            out.append((lam_h, lam_a, info["gh_prof"], info["ga_prof"]))
    return out


def evaluate(matchups, rng, n=4000):
    """Aggregate base-rate metrics across the matchup sample."""
    acc = {"games": 0, "draws": 0, "goals": 0, "n_first": 0, "first_win": 0,
           "n_half": 0, "half_win": 0}
    for lam_h, lam_a, gh, ga in matchups:
        m = ms.measure(lam_h, lam_a, gh, ga, rng, n)
        for kk in acc:
            acc[kk] += m[kk]
    return {
        "first_win": acc["first_win"] / max(1, acc["n_first"]),
        "half_win": acc["half_win"] / max(1, acc["n_half"]),
        "draw": acc["draws"] / acc["games"],
        "goals": acc["goals"] / acc["games"],
    }


def show(label, met):
    print(f"  {label:22s} first-goal-win {met['first_win']:.3f}  "
          f"half-lead-win {met['half_win']:.3f}  draw {met['draw']:.3f}  "
          f"goals {met['goals']:.2f}")


def loss(met):
    return (abs(met["first_win"] - TARGETS["first_win"]) * 2.0
            + abs(met["half_win"] - TARGETS["half_win"]) * 2.0
            + abs(met["draw"] - TARGETS["draw"]) * 1.0)


def main():
    tune = "--tune" in sys.argv
    print("Sampling representative WC2026 matchups (lambda spread = real field)...")
    matchups = sample_matchups(k=120)
    print(f"  {len(matchups)} matchups\n")
    rng = np.random.default_rng(20260613)

    base = evaluate(matchups, rng)
    print("CURRENT constants:")
    print(f"   LEAD_DAMP={ms.LEAD_DAMP} SOLIDITY_DAMP={ms.SOLIDITY_DAMP} "
          f"CHASE_BOOST={ms.CHASE_BOOST} COUNTER_BOOST={ms.COUNTER_BOOST}")
    show("current", base)
    print(f"  targets: first {TARGETS['first_win']} half {TARGETS['half_win']} "
          f"draw {TARGETS['draw']}  | loss {loss(base):.4f}\n")

    if not tune:
        return

    # coarse grid around the levers that move lead-stickiness vs comebacks
    print("Tuning (grid search)...")
    grid = {
        "SOLIDITY_DAMP": [0.22, 0.30, 0.40],
        "CHASE_BOOST":   [0.62, 0.72, 0.80],
        "LEAD_DAMP":     [0.20, 0.27],
        "COUNTER_BOOST": [0.55, 0.70],
    }
    keys = list(grid)
    best = (loss(base), {k: getattr(ms, k) for k in keys}, base)
    orig = {k: getattr(ms, k) for k in keys}
    for combo in itertools.product(*[grid[k] for k in keys]):
        for k, v in zip(keys, combo):
            setattr(ms, k, v)
        met = evaluate(matchups, np.random.default_rng(20260613), n=3000)
        L = loss(met)
        if L < best[0]:
            best = (L, dict(zip(keys, combo)), met)
    for k, v in orig.items():       # restore
        setattr(ms, k, v)

    print("\nBEST candidate:")
    for k, v in best[1].items():
        print(f"   {k} = {v}")
    show("best", best[2])
    print(f"  loss {best[0]:.4f}  (current loss {loss(base):.4f})")
    print("\nIf this improves on current, set these in wc2026_matchsim.py CALIBRATION block.")


if __name__ == "__main__":
    main()
