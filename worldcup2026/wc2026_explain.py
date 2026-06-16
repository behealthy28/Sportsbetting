"""
Full per-match DECOMPOSITION — shows every layer that produces the prediction,
in the order they stack, so a scoreline is never a black box:

  1. DATA MODEL ........ 45% Dixon-Coles MLE + 55% Elo  (the statistical/ML base)
  2. 10-AGENT PANEL .... multiplicative match-condition nudges (incl. hydration)
  3. TACTICAL ENGINE ... style-vs-style mechanisms (additive xG)
  4. PLAYER/FLANK LANES  positional player-vs-player duels (the on-pitch matchups)
  4b. FORMATION + CHEM   shape (midfield numbers, wing-backs, low block) + cohesion
  5. BLEND ............. final expected goals (lambda)
  6. GAME-STATE ........ how each side behaves leading/trailing
  7. EVENT SIM ......... 18-phase sim -> outcome split + scoreline distribution,
                         with the SINGLE most-likely outcome and scoreline stated.

Mirrors wc2026_scorelines.match_lambdas exactly (same weights/constants).
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wc2026_scorelines as S
import wc2026_players as players
import wc2026_tactics as tactics
import wc2026_matchsim as matchsim
panel = S.panel
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def explain(th, ta, neutral=True, knockout=False, n=60000):
    print("=" * 70)
    print(f"  {th}  vs  {ta}   ({'neutral' if neutral else 'host'}"
          f"{' / knockout' if knockout else ' / group'})")
    print("=" * 70)

    # 1. DATA MODEL ------------------------------------------------------
    bh, ba = S.base_lambdas(th, ta, neutral)
    print(f"\n1. DATA MODEL (45% Dixon-Coles + 55% Elo)   base xG: "
          f"{th} {bh:.2f} - {ba:.2f} {ta}")
    print(f"   Elo {S.ELO[th]} vs {S.ELO[ta]}")

    # 2. 10-AGENT PANEL --------------------------------------------------
    ph0, pa0 = players.profile(th, S.ELO[th]), players.profile(ta, S.ELO[ta])
    ctx = {"neutral": neutral, "knockout": knockout, "news_scores": None,
           "coaches": (ph0["coach"], pa0["coach"])}
    mh, ma, bd = panel.run_panel(th, ta, S.R, ctx)
    print(f"\n2. {len(bd)}-AGENT PANEL (xG multipliers)   net: x{mh:.3f} / x{ma:.3f}")
    for name, amh, ama, note in bd:
        tag = "" if (abs(amh - 1) > .003 or abs(ama - 1) > .003) else "  (neutral)"
        print(f"     {name:14s} x{amh:.3f}/{ama:.3f}  {note}{tag}")
    ph_, pa_ = bh * mh, ba * ma
    print(f"   => after panel: {ph_:.2f} - {pa_:.2f}")

    # 3 & 4. PLAYER LANES + TACTICS -------------------------------------
    plh, pla, info = players.player_xg(th, ta, S.ELO[th], S.ELO[ta], ph=ph0, pa=pa0)
    P, A = info["ph"], info["pa"]
    d = info["duels"]
    tac = tactics.matchup(P["style"], A["style"], th, ta)
    dh, da = tac["delta"]

    print(f"\n3. TACTICAL ENGINE   style: {P['style']} vs {A['style']}"
          f"   delta xG: {dh:+.2f}/{da:+.2f}   cagey x{tac['cagey']:.2f}")
    for r in tac["reasons_h"] + tac["reasons_a"]:
        print(f"     - {r}")
    if tac["tempo_note"]:
        print(f"     - {tac['tempo_note']}")

    print(f"\n4. PLAYER / FLANK LANES (units {th}: gk{P['units']['gk']:.0f} "
          f"df{P['units']['df']:.0f} md{P['units']['md']:.0f} at{P['units']['at']:.0f}"
          f" | {ta}: gk{A['units']['gk']:.0f} df{A['units']['df']:.0f} "
          f"md{A['units']['md']:.0f} at{A['units']['at']:.0f})")
    print(f"   channels {th:>12}: L {d['channels_h']['left']:+.2f}  "
          f"R {d['channels_h']['right']:+.2f}  C {d['channels_h']['central']:+.2f}")
    print(f"   channels {ta:>12}: L {d['channels_a']['left']:+.2f}  "
          f"R {d['channels_a']['right']:+.2f}  C {d['channels_a']['central']:+.2f}")
    print(f"   {d['midfield_note']}")
    for r in d["reasons_h"] + d["reasons_a"]:
        print(f"     - {r}")
    sh = d.get("shape", {})
    if sh:
        print(f"\n4b. FORMATION + CHEMISTRY   {th} {sh['form'][0]} (chem {sh['chem'][0]:.0f})"
              f"  vs  {ta} {sh['form'][1]} (chem {sh['chem'][1]:.0f})"
              f"   shape dxg {sh['delta'][0]:+.2f}/{sh['delta'][1]:+.2f}  mid {sh['mid']:+.2f}")
        for r in sh["reasons"]:
            print(f"     - {r}")
    plh2 = (plh + S.W_TACTICS * dh) * tac["cagey"]
    pla2 = (pla + S.W_TACTICS * da) * tac["cagey"]
    print(f"   => player-layer xG (incl. shape+chem): {plh2:.2f} - {pla2:.2f}")

    # 5. BLEND -----------------------------------------------------------
    lh = (1 - S.W_PLAYER) * ph_ + S.W_PLAYER * plh2
    la = (1 - S.W_PLAYER) * pa_ + S.W_PLAYER * pla2
    lh, la = max(0.12, lh), max(0.12, la)
    print(f"\n5. BLEND (70% data+panel / 30% player-layer)   FINAL xG: "
          f"{th} {lh:.2f} - {la:.2f} {ta}")

    # 6. GAME-STATE ------------------------------------------------------
    ghp, gap = S._gamestate_profiles(info)
    print(f"\n6. GAME-STATE   {th}: park {ghp['park']:.2f} chase {ghp['chase']:.2f} "
          f"solidity {ghp['solidity']:.2f} | {ta}: park {gap['park']:.2f} "
          f"chase {gap['chase']:.2f} solidity {gap['solidity']:.2f}")

    # 7. EVENT SIM -------------------------------------------------------
    rng = np.random.default_rng(2026)
    dist = matchsim.expected_distribution(lh, la, ghp, gap, rng, n=n)
    wdl = {th: dist["home_win"], "Draw": dist["draw"], ta: dist["away_win"]}
    best_out = max(wdl, key=wdl.get)
    sc = dist["scores"]
    best_sc = max(sc, key=sc.get)
    print(f"\n7. EVENT SIM ({n:,} runs)")
    print(f"   outcome split: {th} {wdl[th]*100:.0f}%  /  Draw {wdl['Draw']*100:.0f}%"
          f"  /  {ta} {wdl[ta]*100:.0f}%")
    # marginal goal distributions
    for who, axis in ((th, 0), (ta, 1)):
        marg = {}
        for (a, b), p in sc.items():
            k = (a, b)[axis]; marg[k] = marg.get(k, 0) + p
        mean = sum(k * p for k, p in marg.items())
        dd = " ".join(f"{k}:{marg.get(k,0)*100:.0f}%" for k in range(0, 5))
        p3 = sum(p for k, p in marg.items() if k >= 3)
        print(f"   {who:>12} goals: mean {mean:.2f}  [{dd}  3+:{p3*100:.0f}%]")
    top = sorted(sc.items(), key=lambda x: -x[1])[:5]
    print(f"   top scorelines: " + ", ".join(f"{a}-{b} {p*100:.0f}%" for (a, b), p in top))
    print(f"\n   >>> SINGLE MOST-LIKELY OUTCOME : {best_out} "
          f"({wdl[best_out]*100:.0f}%)")
    print(f"   >>> SINGLE MOST-LIKELY SCORELINE: {th} {best_sc[0]}-{best_sc[1]} {ta} "
          f"({sc[best_sc]*100:.0f}%)")
    print()


if __name__ == "__main__":
    explain("Germany", "Curacao")
    explain("Netherlands", "Japan")
    explain("Ivory Coast", "Ecuador")
