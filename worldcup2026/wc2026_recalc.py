"""
Recalculate WC2026 odds conditioned on results played so far (wc2026_played.py).

Runs the vectorised Monte Carlo twice with COMMON RANDOM NUMBERS (same seed):
once pre-tournament (baseline) and once with the played fixtures PINNED. Prints
title-odds shifts and, for the groups that have started, advancement shifts.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wc2026_montecarlo as M
import wc2026_played as P
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

N = M.N
SEED = 42


def run(played):
    M.rng = np.random.default_rng(SEED)     # common random numbers for fair deltas
    return M.simulate(N, played)


def pct(d, t):
    return d.get(t, 0) / N * 100


def main():
    lookup = M.played_lookup(P.PLAYED)
    played_teams = {a for a, _, _, _ in P.PLAYED} | {b for _, _, b, _ in P.PLAYED}
    print(f"Conditioning on {len(P.PLAYED)} played games "
          f"({len(played_teams)} teams active). {N:,} sims each.\n")
    print("Results pinned:")
    for a, sa, b, sb in P.PLAYED:
        print(f"   {a} {sa}-{sb} {b}")
    print()

    base = run(None)
    cond = run(lookup)

    # ---- title odds ----
    rows = sorted(M.TEAMS, key=lambda t: cond["champ"].get(t, 0), reverse=True)
    print("TITLE ODDS  (pre-tournament -> now)")
    print(f"{'Team':16s}{'WIN%':>8}{'was':>8}{'Δ':>8}{'SEMI%':>8}{'Δ':>7}")
    for t in rows[:14]:
        w, w0 = pct(cond["champ"], t), pct(base["champ"], t)
        s, s0 = pct(cond["semi"], t), pct(base["semi"], t)
        print(f"{t:16s}{w:7.1f}%{w0:7.1f}%{w-w0:+7.1f}{s:7.1f}%{s-s0:+6.1f}")

    # ---- advancement for groups that have started ----
    started = [g for g, teams in M.S.GROUPS.items() if any(t in played_teams for t in teams)]
    print(f"\nADVANCEMENT (reach knockout)  — groups underway: {', '.join(started)}")
    print(f"{'Team':16s}{'ADV%':>8}{'was':>8}{'Δ':>8}")
    for g in started:
        print(f"  Group {g}:")
        for t in M.S.GROUPS[g]:
            a, a0 = pct(cond["advance"], t), pct(base["advance"], t)
            flag = "  <-- played" if t in played_teams else ""
            print(f"    {t:14s}{a:7.1f}%{a0:7.1f}%{a-a0:+7.1f}{flag}")


if __name__ == "__main__":
    main()
