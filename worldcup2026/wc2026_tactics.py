"""
WC2026 TACTICAL INTERACTION ENGINE (mechanistic, replaces the 5-rule _style_mod).

EPISTEMIC NOTE: like the rest of the Phase-3 layer this encodes ANALYST
JUDGMENT, not fitted data. The point is to make the *mechanisms* explicit and
tunable instead of hiding them in a handful of additive constants.

A team's style is a VECTOR of playing-shape attributes in [0,1]:

  line_height  0 deep block ............ 1 very high line
  press        0 passive/contain ....... 1 aggressive gegenpress
  directness   0 patient build-up ...... 1 direct / long / vertical
  width        0 central overloads ..... 1 wing / crossing focused
  transition   0 slow, gets men back ... 1 lightning counter
  compactness  0 stretched ............. 1 tight low block

interaction(att, deff) returns (delta_xg, reasons[]) — the additive change to
the ATTACKING side's expected goals from how its shape meets the opponent's
shape, plus human-readable mechanism strings. Symmetric calls (A->B and B->A)
give each side its own tactical xG delta. Magnitudes are deliberately small
(~ +-0.35 xG total) and clamped; personnel-specific effects live in
wc2026_players.py and stack on top of this.
"""

DIMS = ("line_height", "press", "directness", "width", "transition", "compactness")

# one-word styles (used across wc2026_squads.py) -> shape vectors
STYLE_VECTORS = {
    "possession": dict(line_height=.62, press=.55, directness=.20, width=.45, transition=.30, compactness=.40),
    "counter":    dict(line_height=.35, press=.35, directness=.70, width=.55, transition=.90, compactness=.70),
    "transition": dict(line_height=.50, press=.55, directness=.55, width=.50, transition=.82, compactness=.50),
    "press":      dict(line_height=.80, press=.90, directness=.45, width=.50, transition=.70, compactness=.35),
    "wing":       dict(line_height=.55, press=.50, directness=.45, width=.85, transition=.55, compactness=.45),
    "compact":    dict(line_height=.30, press=.30, directness=.55, width=.40, transition=.55, compactness=.85),
    "balanced":   dict(line_height=.50, press=.50, directness=.50, width=.50, transition=.50, compactness=.50),
}


def vec(style):
    return dict(STYLE_VECTORS.get(style, STYLE_VECTORS["balanced"]))


def _c(x):
    return max(0.0, min(1.0, x))


def interaction(att, deff, att_name="", def_name=""):
    """Tactical xG delta for the ATTACKING side (att vector) vs a defending
    shape (deff vector). Returns (delta, reasons[])."""
    d = 0.0
    reasons = []

    # 1. SPACE IN BEHIND — a fast, vertical side punishes a high defensive line.
    space = (deff["line_height"] - 0.5) * (att["transition"] * 0.6 + att["directness"] * 0.4 - 0.45)
    if space > 0.015:
        g = round(0.9 * space, 3)
        d += g
        reasons.append(f"+{g:.2f} space in behind {def_name or 'opp'}'s high line "
                       f"({att_name or 'they'} break fast)")

    # 2. LOW-BLOCK FRUSTRATION — patient possession without verticality stalls
    #    against a compact, deep block.
    block = (deff["compactness"] - 0.5) * (0.5 - att["directness"]) * (0.5 - deff["press"] * 0.3)
    if block > 0.015:
        g = round(-0.85 * block, 3)
        d += g
        reasons.append(f"{g:.2f} {def_name or 'opp'} sits in a low block; "
                       f"{att_name or 'they'} struggle to penetrate centrally")

    # 3. PRESS vs PLAY-OUT — a heavy press forces turnovers against a side that
    #    insists on building short; a direct side bypasses it (and exposes the
    #    presser's high line, handled in #1 for the other direction).
    press_gain = (att["press"] - 0.5) * (0.5 - deff["directness"])
    if press_gain > 0.015:
        g = round(0.8 * press_gain, 3)
        d += g
        reasons.append(f"+{g:.2f} {att_name or 'their'} press forces turnovers "
                       f"as {def_name or 'opp'} plays out from the back")

    # 4. WIDTH vs NARROW BLOCK — crossing threat when a wide side meets a team
    #    that defends narrow/compact.
    cross = (att["width"] - 0.55) * (deff["compactness"] - 0.45)
    if cross > 0.01:
        g = round(0.55 * cross, 3)
        d += g
        reasons.append(f"+{g:.2f} {att_name or 'they'} attack the channels/"
                       f"byline against a narrow block")

    return max(-0.30, min(0.30, round(d, 3))), reasons


def cagey_factor(a, b):
    """Both sides cautious/low-tempo -> fewer total chances (game killer).
    Returns a small symmetric multiplier on both lambdas."""
    openness = (a["line_height"] + b["line_height"] + a["press"] + b["press"]) / 4.0
    if openness < 0.42:
        return 0.92, "both sides cagey / deep -> low-event game"
    if openness > 0.70:
        return 1.06, "two open, high teams -> end-to-end game"
    return 1.0, ""


def matchup(style_h, style_a, name_h="", name_a=""):
    """Full tactical read for a fixture. Returns dict with each side's tactical
    xG delta, the cagey/open multiplier, and all mechanism reasons."""
    vh, va = vec(style_h), vec(style_a)
    dh, rh = interaction(vh, va, name_h, name_a)
    da, ra = interaction(va, vh, name_a, name_h)
    cf, cnote = cagey_factor(vh, va)
    return {
        "vec": (vh, va),
        "delta": (dh, da),          # additive xG deltas (home, away)
        "cagey": cf,                # multiplicative, both sides
        "reasons_h": rh, "reasons_a": ra,
        "tempo_note": cnote,
        "styles": (style_h, style_a),
    }


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    for sh, sa in [("wing", "counter"), ("possession", "compact"),
                   ("press", "possession"), ("counter", "counter")]:
        m = matchup(sh, sa, sh.upper(), sa.upper())
        print(f"\n{sh} vs {sa}  delta={m['delta']}  cagey x{m['cagey']}")
        for r in m["reasons_h"] + m["reasons_a"]:
            print("   ", r)
        if m["tempo_note"]:
            print("    ~", m["tempo_note"])
