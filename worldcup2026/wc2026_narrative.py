"""
WC2026 TACTICAL NARRATIVE — turns the model's structured nuance into a deep,
readable match read: danger lanes, the plays each side can make, the tactical
mechanisms that decide it, the midfield battle, how each side will manage the
game state, and a representative flow of the match.

Pure presentation over the bundle produced by wc2026_scorelines.predict():
  r["lineup"] -> {ph, pa, duels, tactics, gh_prof, ga_prof}
  r["dist"]   -> event-sim outcome + scoreline distribution
"""


def _top_lanes(channels, team):
    items = sorted(channels.items(), key=lambda x: -x[1])
    best = items[0]
    label = {"left": "down the left", "right": "down the right", "central": "through the middle"}
    return f"{team} most dangerous {label[best[0]]} ({best[1]:+.2f} xG lane)"


def _gamestate_line(team, prof, opp):
    park, chase, sol = prof["park"], prof["chase"], prof["solidity"]
    if park >= 0.8:
        lead = f"{team} will game-manage a lead hard (park the bus; solidity {sol*100:.0f})"
    elif park <= 0.45:
        lead = f"{team} keeps playing even when ahead (won't sit back; vulnerable to nothing-to-lose {opp})"
    else:
        lead = f"{team} manages a lead moderately"
    if chase >= 0.75:
        trail = f"chases hard when behind (commits numbers, exposed on the counter)"
    elif chase <= 0.5:
        trail = f"stays compact even when behind (low comeback ceiling)"
    else:
        trail = f"pushes when behind without over-committing"
    return f"  {lead}; {trail}."


def analysis(th, ta, r):
    """Return a list of printable lines: the deep tactical read."""
    info = r.get("lineup")
    if not info:
        return ["  (no player/tactics layer for this match)"]
    d = info["duels"]
    tac = info["tactics"]
    out = []

    # --- the call + game flow ---
    sh, sa = r["score"]
    dist = r.get("dist")
    out.append(f"  Expected: {th} {sh}-{sa} {ta}  (xG {r['xg'][0]}-{r['xg'][1]}, "
               f"styles: {tac['styles'][0]} vs {tac['styles'][1]})")
    if dist:
        out.append(f"  Outcome split: {th} {dist['home_win']*100:.0f}% / draw "
                   f"{dist['draw']*100:.0f}% / {ta} {dist['away_win']*100:.0f}%")

    # --- danger lanes ---
    out.append("  DANGER LANES:")
    out.append("    " + _top_lanes(d["channels_h"], th))
    out.append("    " + _top_lanes(d["channels_a"], ta))
    out.append("    " + d["midfield_note"])

    # --- player duels / the plays ---
    plays = d["reasons_h"] + d["reasons_a"]
    if plays:
        out.append("  KEY DUELS & PLAYS:")
        for p in plays:
            out.append(f"    - {p}")

    # --- formation / chemistry shape ---
    sh = d.get("shape")
    if sh and sh.get("reasons"):
        out.append("  SHAPE & CHEMISTRY:")
        out.append(f"    {th} {sh['form'][0]} vs {ta} {sh['form'][1]}")
        for r in sh["reasons"]:
            out.append(f"    - {r}")

    # --- tactical mechanisms (why an approach works / fails) ---
    mech = tac["reasons_h"] + tac["reasons_a"]
    if mech or tac["tempo_note"]:
        out.append("  TACTICAL READ:")
        for m in mech:
            out.append(f"    - {m}")
        if tac["tempo_note"]:
            out.append(f"    - game shape: {tac['tempo_note']}")

    # --- game-state behaviour ---
    out.append("  GAME-STATE:")
    out.append(_gamestate_line(th, info["gh_prof"], ta))
    out.append(_gamestate_line(ta, info["ga_prof"], th))

    return out


def flow(th, ta, info, rng):
    """A representative minute-by-minute story of the match (sampled once)."""
    import wc2026_matchsim as ms
    lam_h, lam_a = info["_lam"]
    gh, ga, events = ms.simulate_timeline(
        lam_h, lam_a, info["gh_prof"], info["ga_prof"], rng, th, ta)
    if not events:
        return [f"  REPRESENTATIVE FLOW: a tight, low-event game ({th} 0-0 {ta})."]
    lines = ["  REPRESENTATIVE FLOW:"]
    for minute, scorer, h, a in events:
        lines.append(f"    {minute}'  {scorer} scores -> {h}-{a}")
    return lines
