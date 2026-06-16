"""
WC2026 FORMATION + SQUAD-CHEMISTRY layer (first-class shape inputs).

FORMATIONS are now explicit (4-3-3, 3-5-2, 5-3-2, ...) and change the match
mechanically, not just cosmetically:
  * MIDFIELD NUMBERS — an extra central midfielder (3 v 2) buys control.
  * WING-BACKS (back-3 shapes) — extra wide attacking thrust, but space in
    behind them for the opponent to counter.
  * LOW BLOCK (back-5) — compresses the opponent's chances, dampens own attack.
  * TWO STRIKERS — more central/box presence.

SQUAD CHEMISTRY is a 0-100 cohesion rating (settled XI / time under the coach /
caps together). A more cohesive side sustains pressure and stays organised — a
small edge that talent alone doesn't capture (a galaxy of stars thrown together
underperforms a drilled unit).

EPISTEMIC NOTE: formations and chemistry here are analyst judgment (likely
shape + subjective cohesion), not data. Magnitudes are deliberately secondary
to talent and tactics.
"""

# shape attributes per formation
FORMATIONS = {
    "4-3-3":   dict(back=4, mids=3, wide=2, strikers=1, wingbacks=False, lowblock=False),
    "4-2-3-1": dict(back=4, mids=3, wide=2, strikers=1, wingbacks=False, lowblock=False),
    "4-4-2":   dict(back=4, mids=2, wide=2, strikers=2, wingbacks=False, lowblock=False),
    "4-5-1":   dict(back=4, mids=3, wide=2, strikers=1, wingbacks=False, lowblock=False),
    "3-4-3":   dict(back=3, mids=2, wide=2, strikers=1, wingbacks=True,  lowblock=False),
    "3-5-2":   dict(back=3, mids=3, wide=2, strikers=2, wingbacks=True,  lowblock=False),
    "5-3-2":   dict(back=5, mids=3, wide=0, strikers=2, wingbacks=False, lowblock=True),
    "5-4-1":   dict(back=5, mids=4, wide=1, strikers=1, wingbacks=False, lowblock=True),
}

STYLE_FORMATION = {
    "possession": "4-3-3", "counter": "4-2-3-1", "transition": "4-2-3-1",
    "press": "4-3-3", "wing": "3-4-3", "compact": "5-3-2", "balanced": "4-2-3-1",
}

# per-team formation overrides (likely first-choice shapes, early 2026)
FORM_OVERRIDE = {
    "Argentina": "4-3-3", "Spain": "4-3-3", "France": "4-2-3-1", "England": "4-2-3-1",
    "Brazil": "4-3-3", "Portugal": "4-3-3", "Germany": "4-2-3-1", "Netherlands": "4-3-3",
    "Belgium": "4-2-3-1", "Croatia": "4-3-3", "Uruguay": "4-3-3", "Morocco": "4-3-3",
    "Japan": "3-4-3", "Mexico": "4-3-3", "USA": "4-3-3", "Switzerland": "4-2-3-1",
    "Senegal": "4-3-3", "Colombia": "4-2-3-1", "Scotland": "3-5-2", "Norway": "4-3-3",
    "Sweden": "4-4-2", "Ecuador": "4-3-3", "Austria": "4-2-3-1", "Ivory Coast": "4-3-3",
    "Korea Republic": "4-2-3-1", "Canada": "4-3-3", "Algeria": "4-3-3", "Ghana": "4-3-3",
    "Paraguay": "4-4-2", "Turkey": "4-2-3-1", "Egypt": "4-2-3-1",
}

# squad chemistry 0-100 (settled & drilled high; talented-but-rotating lower)
CHEM = {
    "Argentina": 90, "Morocco": 88, "Croatia": 87, "Japan": 85, "Uruguay": 84,
    "Spain": 84, "Scotland": 84, "Mexico": 82, "USA": 82, "Ecuador": 82,
    "Senegal": 82, "Norway": 80, "Portugal": 80, "Netherlands": 79, "Croatia ": 87,
    "France": 77, "Germany": 76, "Turkey": 76, "England": 75, "Brazil": 74,
    "Ghana": 74, "Algeria": 74, "Ivory Coast": 72, "Belgium": 68,
}
CHEM_DEFAULT = 76


def team_shape(team, style):
    """Return (formation_string, chemistry) for a team."""
    form = FORM_OVERRIDE.get(team, STYLE_FORMATION.get(style, "4-2-3-1"))
    return form, CHEM.get(team, CHEM_DEFAULT)


def attrs(form):
    return FORMATIONS.get(form, FORMATIONS["4-2-3-1"])


def effects(form_h, form_a, nh, na):
    """Mechanical shape effects. Returns (dxg_home, dxg_away, mid_shift, reasons).
    mid_shift > 0 favours home (extra territory/control)."""
    H, A = attrs(form_h), attrs(form_a)
    dh = da = mid = 0.0
    reasons = []

    # 1. midfield numbers
    mn = H["mids"] - A["mids"]
    if mn != 0:
        mid += 0.035 * mn
        w = nh if mn > 0 else na
        reasons.append(f"{w} win the midfield numbers ({form_h} v {form_a}) "
                       f"-> extra man, more control")

    # 2. wing-backs: own wide thrust up, but space in behind for the opponent
    if H["wingbacks"]:
        dh += 0.05; da += 0.05
        reasons.append(f"{nh} wing-backs ({form_h}) push high -> wide threat, "
                       f"but {na} can attack the space behind them")
    if A["wingbacks"]:
        da += 0.05; dh += 0.05
        reasons.append(f"{na} wing-backs ({form_a}) push high -> wide threat, "
                       f"but {nh} can attack the space behind them")

    # 3. low block (back-five): compress opponent, slightly dampen own attack
    if H["lowblock"]:
        da -= 0.10; dh -= 0.04
        reasons.append(f"{nh} back-five low block ({form_h}) compresses {na}'s chances")
    if A["lowblock"]:
        dh -= 0.10; da -= 0.04
        reasons.append(f"{na} back-five low block ({form_a}) compresses {nh}'s chances")

    # 4. two strikers: extra central/box presence
    if H["strikers"] >= 2:
        dh += 0.04; reasons.append(f"{nh} play two strikers ({form_h}) -> more box presence")
    if A["strikers"] >= 2:
        da += 0.04; reasons.append(f"{na} play two strikers ({form_a}) -> more box presence")

    dh = max(-0.18, min(0.18, dh)); da = max(-0.18, min(0.18, da))
    return dh, da, mid, reasons


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    for fh, fa in [("3-5-2", "4-3-3"), ("5-3-2", "4-2-3-1"), ("4-4-2", "4-3-3")]:
        dh, da, mid, rs = effects(fh, fa, "HOME", "AWAY")
        print(f"\n{fh} vs {fa}: dxg {dh:+.2f}/{da:+.2f} mid {mid:+.2f}")
        for r in rs:
            print("   -", r)
