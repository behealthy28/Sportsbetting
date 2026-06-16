"""
wc2026_eval.py — rigorous evaluation + calibration harness for the WC2026 model.

Applies senior-data-scientist discipline to a TINY (n=10) sample:
  * proper probabilistic metrics (multiclass Brier, log-loss)
  * a baseline to beat (naive 'pick favourite', and the plain DC+Elo core)
  * BOOTSTRAP confidence intervals on metric differences (is the gap real?)
  * an over-confidence diagnostic (predicted vs actual favourite win-rate)
  * two PRINCIPLED, non-data-dredged fixes, each justified by theory:
       (A) shrinkage ensemble  : blend full stack toward the better-calibrated core
       (B) goal over-dispersion: Gamma-mixed Poisson (fatter tails -> more draws,
                                  more blowouts, softer favourites)
  * leave-one-out stability check (guards against overfitting the 10 games)

Run:  python wc2026_eval.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import wc2026_scorelines as S
import wc2026_played as P
import wc2026_matchsim as ms

RNG = np.random.default_rng(20260616)
N_FULL = 20000

def oh(o):  return {"H": (1,0,0), "D": (0,1,0), "A": (0,0,1)}[o]
def brier(p, o):  t = oh(o); return sum((a-b)**2 for a, b in zip(p, t))
def logloss(p, o):
    t = oh(o); eps = 1e-12
    return -sum(b*np.log(max(a, eps)) for a, b in zip(p, t))
def result(hg, ag): return "H" if hg > ag else ("A" if ag > hg else "D")
def argmax_pick(p): return "HDA"[int(np.argmax(p))]

# ---------------------------------------------------------------- gather
games = []
for (h, hg, a, ag) in P.PLAYED:
    o = result(hg, ag)
    bh, ba = S.base_lambdas(h, a, neutral=True)
    w0 = S._outcome_probs(bh, ba, S.RHO)
    base = (w0["home_win"], w0["draw"], w0["away_win"])
    lam_h, lam_a, (mh, ma), bd, info = S.match_lambdas(h, a, neutral=True)
    d = ms.expected_distribution(lam_h, lam_a, info["gh_prof"], info["ga_prof"], RNG, n=N_FULL)
    full = (d["home_win"], d["draw"], d["away_win"])
    fav_home = S.ELO[h] >= S.ELO[a]
    games.append(dict(h=h, a=a, o=o, base=base, full=full, lam=(lam_h, lam_a),
                      prof=(info["gh_prof"], info["ga_prof"]), fav_home=fav_home))
n = len(games)

def fav_winprob(p, g):    # prob the model assigns to the FAVOURITE winning
    return p[0] if g["fav_home"] else p[2]
def fav_won(g):
    return (g["o"] == "H") if g["fav_home"] else (g["o"] == "A")

def report_block(name, probs):
    b = np.array([brier(p, g["o"]) for p, g in zip(probs, games)])
    l = np.array([logloss(p, g["o"]) for p, g in zip(probs, games)])
    hits = sum(argmax_pick(p) == g["o"] for p, g in zip(probs, games))
    draws_called = sum(argmax_pick(p) == "D" for p in probs)
    fav_pred = np.mean([fav_winprob(p, g) for p, g in zip(probs, games)])
    return dict(name=name, brier=b.mean(), logloss=l.mean(), hits=hits,
                draws=draws_called, fav_pred=fav_pred, b_arr=b)

base_probs = [g["base"] for g in games]
full_probs = [g["full"] for g in games]
fav_probs  = [(1,0,0) if g["fav_home"] else (0,0,1) for g in games]

R_base = report_block("base core (DC+Elo)", base_probs)
R_full = report_block("FULL stack", full_probs)
R_fav  = report_block("naive favourite", fav_probs)
fav_actual = np.mean([fav_won(g) for g in games])

print("="*78)
print(f"  EVALUATION  (n={n} played games)   metrics: Brier & log-loss, lower=better")
print("="*78)
print(f"{'model':26} {'Brier':>7} {'logloss':>8} {'hits':>6} {'draws#':>7} {'favWinPred':>11}")
for R in (R_fav, R_base, R_full):
    print(f"{R['name']:26} {R['brier']:7.3f} {R['logloss']:8.3f} "
          f"{R['hits']:>4}/{n} {R['draws']:>6} {R['fav_pred']*100:9.0f}%")
print(f"{'ACTUAL favourite win-rate':26} {'':7} {'':8} {'':6} {'':7} {fav_actual*100:9.0f}%")

# ---------------------------------------------------------------- bootstrap: is full WORSE than base for real?
diff = R_full["b_arr"] - R_base["b_arr"]          # >0 means full worse
boot = np.array([np.mean(diff[RNG.integers(0, n, n)]) for _ in range(20000)])
lo, hi = np.percentile(boot, [2.5, 97.5])
print("\n" + "-"*78)
print("BOOTSTRAP — Brier(full) - Brier(base):")
print(f"   point estimate {diff.mean():+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   "
      f"P(full worse)={np.mean(boot>0)*100:.0f}%")
print("   => CI straddles 0" if lo < 0 < hi else "   => CI excludes 0",
      "- the base-vs-full gap is NOT statistically distinguishable from noise"
      if lo < 0 < hi else "- difference is significant")

# ---------------------------------------------------------------- (A) shrinkage ensemble toward the core
print("\n" + "-"*78)
print("(A) SHRINKAGE ENSEMBLE   wdl* = (1-s)*full + s*base   (s set by PRINCIPLE, not argmin)")
def blend(s):
    return [tuple((1-s)*np.array(f) + s*np.array(b)) for f, b in zip(full_probs, base_probs)]
for s in (0.0, 0.25, 0.5, 0.75, 1.0):
    R = report_block(f"s={s}", blend(s))
    star = "  <- equal-weight default" if abs(s-0.5) < 1e-9 else ""
    print(f"   s={s:<4} Brier {R['brier']:.3f}  logloss {R['logloss']:.3f}  "
          f"draws#={R['draws']}  favWinPred={R['fav_pred']*100:.0f}%{star}")
# LOO stability of the *argmin* s (diagnostic only — to check overfitting risk)
grid = np.linspace(0, 1, 41)
loo_best = []
for i in range(n):
    idx = [j for j in range(n) if j != i]
    errs = [np.mean([brier(blend(s)[j], games[j]["o"]) for j in idx]) for s in grid]
    loo_best.append(grid[int(np.argmin(errs))])
print(f"   LOO argmin-s range: [{min(loo_best):.2f}, {max(loo_best):.2f}] "
      f"median {np.median(loo_best):.2f}  (wide => tuning s would overfit; "
      f"use principled s=0.5)")

# ---------------------------------------------------------------- (B) goal over-dispersion (Gamma-mixed Poisson)
print("\n" + "-"*78)
print("(B) OVER-DISPERSION   lambda *= Gamma(k,1/k) per side  (k from football prior, NOT fit)")
def dispersed(g, k, G=24):
    lam_h, lam_a = g["lam"]; gh, ga = g["prof"]; acc = np.zeros(3)
    for _ in range(G):
        mh, ma = RNG.gamma(k, 1/k), RNG.gamma(k, 1/k)
        d = ms.expected_distribution(lam_h*mh, lam_a*ma, gh, ga, RNG, n=2500)
        acc += [d["home_win"], d["draw"], d["away_win"]]
    return tuple(acc/G)
for k in (12, 8, 6):
    dp = [dispersed(g, k) for g in games]
    R = report_block(f"k={k}", dp)
    print(f"   k={k:<3}(CV={1/np.sqrt(k):.2f})  Brier {R['brier']:.3f}  logloss {R['logloss']:.3f}  "
          f"draws#={R['draws']}  favWinPred={R['fav_pred']*100:.0f}%")

print("\n" + "-"*78)
print("Calibration target: favWinPred should ~= ACTUAL favourite win-rate "
      f"({fav_actual*100:.0f}%). Full stack predicts {R_full['fav_pred']*100:.0f}% "
      "=> over-confident.")
