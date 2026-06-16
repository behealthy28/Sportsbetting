"""
WC2026 PLAYER / FLANK LANE MODEL — positional, player-vs-player duels.

WHY: wc2026_squads.py rates a team with ONE number per unit (DEF=84), so it
cannot express that Brazil's threat is Vinicius (LW, elite pace) attacking the
SPACE Hakimi (RB) vacates by bombing forward. This module resolves each team
into positional SLOTS, each with a 0-100 rating and attribute TAGS, and scores
the match LANE BY LANE: left channel, right channel, central, plus midfield
control. Tag interactions create the nuance (pace vs a high full-back, an aerial
striker vs a small CB pair, a press-resistant midfield vs a gegenpress).

EPISTEMIC NOTE: assumed first-choice players and subjective ratings/tags as of
early 2026 — analyst judgment, not data. ~18 teams carry hand-authored flank
detail; the rest are decomposed from their unit ratings (wc2026_squads) with
flat tags and flagged detailed=False.

SLOTS: GK LB LCB RCB RB DM CM AM LW RW ST
TAGS (attacking): pace dribble crossing aerial poacher creator press_resistant
TAGS (defending): high_fb(overlapping/aggressive FB) slow recovery_pace aerial
                  ball_playing destroyer
TAGS (gk): sweeper shot_stopper
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wc2026_squads as squads
import wc2026_formations as formations

# ---- hand-authored flank detail for marquee teams --------------------------
# (slot -> (rating, "tags")). Only slots that differ from the unit baseline or
# carry meaningful tags need listing; the rest fall back to the unit number.
FLANK = {
    "Brazil": dict(
        GK=(86, "shot_stopper sweeper"),
        LB=(80, ""), LCB=(83, "ball_playing"), RCB=(82, "recovery_pace"), RB=(78, "high_fb"),
        DM=(80, "destroyer"), CM=(82, "press_resistant"), AM=(84, "creator"),
        LW=(92, "pace dribble"),                 # Vinicius Jr
        RW=(86, "pace creator"),                 # Raphinha / Rodrygo
        ST=(82, "poacher"),
        names=dict(LW="Vinicius", RW="Raphinha", ST="Endrick/G.Jesus",
                   AM="Paqueta", DM="B.Guimaraes", LB="Wendell", RB="Vanderson",
                   LCB="Marquinhos", RCB="Militao", GK="Alisson")),
    "Morocco": dict(
        GK=(84, "shot_stopper"),
        LB=(78, ""), LCB=(83, "aerial"), RCB=(82, ""), RB=(86, "high_fb pace"),  # Hakimi
        DM=(82, "destroyer"), CM=(80, "press_resistant"), AM=(80, "creator"),
        LW=(80, "dribble"), RW=(79, "pace"), ST=(80, "aerial poacher"),
        names=dict(RB="Hakimi", LCB="Saiss", RCB="Aguerd", DM="Amrabat",
                   AM="Ounahi", LW="Diaz", RW="Ziyech", ST="En-Nesyri",
                   GK="Bounou", LB="Mazraoui")),
    "Argentina": dict(
        GK=(88, "shot_stopper"),
        LB=(80, "high_fb"), LCB=(83, "aerial"), RCB=(82, "recovery_pace"), RB=(80, ""),
        DM=(83, "destroyer press_resistant"), CM=(86, "press_resistant creator"), AM=(88, "creator"),
        LW=(82, "dribble"), RW=(80, "pace"), ST=(85, "poacher"),
        names=dict(AM="Messi", CM="Mac Allister/E.Fernandez", DM="De Paul",
                   ST="J.Alvarez", GK="E.Martinez", LCB="Romero", RCB="Otamendi")),
    "France": dict(
        GK=(87, "shot_stopper sweeper"),
        LB=(84, "high_fb pace"), LCB=(88, "recovery_pace aerial"), RCB=(85, "recovery_pace"), RB=(80, ""),
        DM=(85, "destroyer"), CM=(84, "press_resistant"), AM=(82, "creator"),
        LW=(94, "pace dribble"),                  # Mbappe
        RW=(86, "dribble creator"),               # Dembele
        ST=(83, "poacher aerial"),
        names=dict(LW="Mbappe", RW="Dembele", ST="Thuram", DM="Tchouameni",
                   CM="Camavinga", LCB="Saliba", RCB="Upamecano", LB="T.Hernandez",
                   GK="Maignan")),
    "Spain": dict(
        GK=(80, "sweeper"),
        LB=(83, "high_fb"), LCB=(84, "ball_playing"), RCB=(83, "ball_playing recovery_pace"), RB=(80, "high_fb"),
        DM=(90, "destroyer press_resistant"), CM=(91, "press_resistant creator"), AM=(88, "creator"),
        LW=(85, "pace dribble"),                  # N. Williams
        RW=(91, "dribble creator"),               # Lamine Yamal
        ST=(82, "poacher"),
        names=dict(DM="Rodri", CM="Pedri", AM="Fabian/Olmo", RW="Yamal",
                   LW="N.Williams", ST="Oyarzabal", GK="Simon", LCB="Cubarsi",
                   RCB="Le Normand", LB="Cucurella")),
    "England": dict(
        GK=(80, "shot_stopper"),
        LB=(80, "high_fb"), LCB=(83, "aerial"), RCB=(82, "ball_playing"), RB=(79, "high_fb"),
        DM=(85, "destroyer"), CM=(86, "press_resistant creator"), AM=(86, "creator"),
        LW=(82, "dribble"), RW=(88, "dribble creator"), ST=(86, "poacher aerial"),
        names=dict(ST="Kane", RW="Saka", AM="Bellingham", CM="Foden", DM="Rice",
                   LCB="Guehi", RCB="Stones", LB="Shaw", GK="Pickford")),
    "Portugal": dict(
        GK=(82, "shot_stopper"),
        LB=(80, "high_fb"), LCB=(81, "aerial"), RCB=(80, "ball_playing"), RB=(80, "high_fb pace"),
        DM=(83, "destroyer"), CM=(86, "press_resistant creator"), AM=(85, "creator"),
        LW=(88, "pace dribble"), RW=(82, "dribble"), ST=(82, "poacher aerial"),
        names=dict(LW="R.Leao", AM="B.Fernandes", CM="Vitinha/B.Silva", ST="Ronaldo/Ramos",
                   RB="Cancelo", LCB="Dias", GK="D.Costa")),
    "Germany": dict(
        GK=(83, "sweeper"),
        LB=(80, "high_fb"), LCB=(78, "aerial"), RCB=(78, "ball_playing"), RB=(80, "high_fb"),
        DM=(86, "destroyer press_resistant"), CM=(88, "press_resistant creator"), AM=(87, "creator dribble"),
        LW=(86, "dribble creator"), RW=(84, "dribble"), ST=(80, "poacher aerial"),
        names=dict(CM="Wirtz", AM="Musiala", DM="Kimmich", ST="Fullkrug/Havertz",
                   LCB="Tah", RCB="Rudiger", GK="ter Stegen")),
    "Netherlands": dict(
        GK=(82, "sweeper"),
        LB=(80, "high_fb"), LCB=(88, "aerial ball_playing recovery_pace"), RCB=(82, ""), RB=(84, "high_fb pace"),
        DM=(80, "destroyer"), CM=(81, "press_resistant"), AM=(80, "creator"),
        LW=(82, "pace"), RW=(80, "dribble"), ST=(80, "aerial poacher"),
        names=dict(LCB="van Dijk", RB="Dumfries", LW="Gakpo", AM="Simons",
                   CM="Gravenberch/Reijnders", GK="Verbruggen")),
    "Belgium": dict(
        GK=(84, "shot_stopper"),
        LB=(74, ""), LCB=(74, ""), RCB=(74, "aerial"), RB=(76, "high_fb"),
        DM=(78, "destroyer"), CM=(86, "creator press_resistant"), AM=(86, "creator"),
        LW=(84, "pace dribble"), RW=(80, "dribble"), ST=(82, "poacher aerial"),
        names=dict(AM="De Bruyne", LW="Doku", ST="Lukaku", CM="Onana",
                   GK="Courtois")),
    "Croatia": dict(
        DM=(82, "press_resistant"), CM=(87, "press_resistant creator"), AM=(83, "creator"),
        LW=(74, "dribble"), RW=(73, ""), ST=(74, "poacher"),
        names=dict(CM="Modric/Kovacic", AM="Sucic", ST="Budimir/Kramaric",
                   LCB="Gvardiol", GK="Livakovic")),
    "Uruguay": dict(
        DM=(80, "destroyer"), CM=(84, "press_resistant"), AM=(80, "creator"),
        LW=(78, "pace"), RW=(78, "dribble"), ST=(83, "pace poacher"),
        names=dict(CM="Valverde", ST="Nunez", LCB="Araujo", RCB="Gimenez",
                   GK="Rochet")),
    "Colombia": dict(
        AM=(85, "creator"), LW=(86, "pace dribble"), RW=(78, "dribble"), ST=(80, "poacher"),
        names=dict(LW="L.Diaz", AM="J.Rodriguez", ST="Cordoba", CM="Lerma/Rios")),
    "Japan": dict(
        DM=(80, "press_resistant"), CM=(84, "press_resistant creator"), AM=(82, "creator"),
        LW=(85, "pace dribble"), RW=(82, "creator"), ST=(78, "poacher"),
        names=dict(LW="Mitoma", RW="Kubo", AM="Kamada", DM="Endo", ST="Ueda")),
    "USA": dict(
        LW=(84, "pace dribble"), AM=(83, "creator"), DM=(80, "destroyer"),
        names=dict(LW="Pulisic", AM="Reyna", DM="Adams", CM="McKennie",
                   ST="Balogun", GK="Turner")),
    "Senegal": dict(
        GK=(82, "shot_stopper"),
        LCB=(82, "aerial recovery_pace"), DM=(80, "destroyer"),
        RW=(82, "pace dribble"), ST=(81, "pace poacher"),
        names=dict(RW="I.Sarr", ST="Jackson", LCB="Koulibaly", GK="E.Mendy",
                   DM="P.Gueye")),
    "Switzerland": dict(
        DM=(80, "destroyer"), CM=(80, "press_resistant"),
        ST=(76, "pace aerial"),
        names=dict(CM="Xhaka", ST="Embolo", GK="Sommer", LCB="Akanji")),
    "Mexico": dict(
        ST=(76, "aerial poacher"), AM=(76, "creator"),
        names=dict(ST="R.Jimenez", AM="Lozano", DM="E.Alvarez", GK="Malagon")),
    # ---- expanded set (knockout-capable / star-carrying) -------------------
    "Norway": dict(style="transition", coach=76,
        GK=(80, "shot_stopper"), LB=(78, "high_fb"), LCB=(80, "aerial"),
        RCB=(79, "recovery_pace"), RB=(80, "high_fb pace"),
        DM=(80, "destroyer"), CM=(82, "press_resistant"), AM=(88, "creator"),
        LW=(82, "pace dribble"), RW=(78, "pace"), ST=(92, "poacher aerial pace"),
        names=dict(ST="Haaland", AM="Odegaard", LW="Nusa", CM="Berge",
                   RB="Ryerson", LCB="Ajer", GK="Nyland")),
    "Sweden": dict(style="counter", coach=75,
        GK=(80, "shot_stopper"), LB=(78, ""), LCB=(82, "aerial"),
        RCB=(80, "ball_playing"), RB=(78, "high_fb"),
        DM=(78, "destroyer"), CM=(80, "press_resistant"), AM=(83, "creator dribble"),
        LW=(80, "pace"), RW=(82, "dribble creator"), ST=(86, "pace poacher"),
        names=dict(ST="Isak/Gyokeres", AM="Kulusevski", LCB="Lindelof",
                   CM="Bergvall", GK="Olsen")),
    "Egypt": dict(style="counter", coach=74,
        GK=(82, "shot_stopper"), LB=(76, ""), LCB=(80, "aerial"),
        RCB=(78, ""), RB=(78, "high_fb"),
        DM=(78, "destroyer"), CM=(79, "press_resistant"), AM=(80, "creator"),
        LW=(78, "pace"), RW=(89, "pace dribble creator"), ST=(80, "poacher"),
        names=dict(RW="Salah", GK="El Shenawy", LCB="Hegazi", DM="Elneny",
                   ST="Marmoush", LW="Trezeguet")),
    "Turkey": dict(style="possession", coach=78,
        GK=(80, ""), LB=(78, "high_fb"), LCB=(80, "aerial"),
        RCB=(80, "ball_playing"), RB=(80, "high_fb"),
        DM=(84, "destroyer press_resistant"), CM=(82, "press_resistant creator"),
        AM=(85, "creator dribble"),
        LW=(82, "pace dribble"), RW=(80, "dribble"), ST=(79, "poacher"),
        names=dict(AM="Guler", DM="Calhanoglu", LW="Yildiz", GK="Cakir",
                   LCB="Demiral", RB="Muldur", ST="Akturkoglu")),
    "Ecuador": dict(style="press", coach=78,
        GK=(79, "shot_stopper"), LB=(80, "high_fb pace"), LCB=(83, "recovery_pace aerial"),
        RCB=(82, "recovery_pace"), RB=(79, ""),
        DM=(86, "destroyer press_resistant"), CM=(80, "press_resistant"), AM=(80, "creator"),
        LW=(78, "pace"), RW=(78, "dribble"), ST=(80, "poacher pace"),
        names=dict(DM="Caicedo", ST="E.Valencia", LCB="Pacho", RCB="Hincapie",
                   LB="Estupinan", AM="Paez", GK="Galindez")),
    "Austria": dict(style="press", coach=79,
        GK=(80, ""), LB=(78, "high_fb"), LCB=(82, "ball_playing"),
        RCB=(80, "aerial"), RB=(80, "high_fb"),
        DM=(82, "destroyer"), CM=(84, "press_resistant creator"), AM=(80, "creator"),
        LW=(78, "pace"), RW=(78, "dribble"), ST=(79, "aerial poacher"),
        names=dict(CM="Sabitzer", ST="Arnautovic", LCB="Alaba", DM="Seiwald",
                   RB="Laimer", AM="Baumgartner", GK="Pentz")),
    "Ivory Coast": dict(style="transition", coach=79,
        GK=(80, ""), LB=(78, "high_fb"), LCB=(82, "aerial recovery_pace"),
        RCB=(80, ""), RB=(80, "high_fb pace"),
        DM=(82, "destroyer"), CM=(83, "press_resistant"), AM=(80, "creator"),
        LW=(82, "pace dribble"), RW=(83, "pace dribble"), ST=(81, "aerial poacher"),
        names=dict(CM="Kessie", RW="Pepe", LW="Adingra", ST="Haller",
                   RB="Aurier", LCB="Ndicka", DM="Fofana", GK="Yahia")),
    "Korea Republic": dict(style="transition", coach=77,
        GK=(78, ""), LB=(78, "high_fb"), LCB=(84, "aerial recovery_pace"),
        RCB=(79, ""), RB=(78, "high_fb"),
        DM=(79, "destroyer"), CM=(80, "press_resistant"), AM=(84, "creator dribble"),
        LW=(87, "pace dribble creator"), RW=(78, "pace"), ST=(80, "pace poacher"),
        names=dict(LW="Son", AM="Lee Kang-in", LCB="Kim Min-jae",
                   ST="Hwang Hee-chan", CM="Hwang In-beom", GK="Kim Seung-gyu")),
    "Canada": dict(style="counter", coach=76,
        GK=(78, ""), LB=(84, "high_fb pace crossing"), LCB=(80, "aerial"),
        RCB=(79, "recovery_pace"), RB=(78, "high_fb"),
        DM=(79, "destroyer"), CM=(81, "press_resistant"), AM=(80, "creator"),
        LW=(80, "pace"), RW=(80, "pace dribble"), ST=(83, "pace poacher"),
        names=dict(LB="A.Davies", ST="J.David", RW="Buchanan", CM="Eustaquio",
                   GK="St-Clair", LCB="Vitoria")),
    "Scotland": dict(style="compact", coach=78,
        GK=(79, "shot_stopper"), LB=(83, "high_fb crossing"), LCB=(80, "aerial"),
        RCB=(79, ""), RB=(78, "high_fb"),
        DM=(80, "destroyer"), CM=(81, "press_resistant"), AM=(84, "creator aerial poacher"),
        LW=(78, "dribble"), RW=(79, "creator"), ST=(78, "aerial poacher"),
        names=dict(LB="Robertson", AM="McTominay", RW="McGinn", CM="Gilmour",
                   GK="Gunn", ST="Adams/Dykes")),
    "Algeria": dict(style="possession", coach=76,
        GK=(79, ""), LB=(78, "high_fb"), LCB=(80, "aerial"),
        RCB=(79, "ball_playing"), RB=(80, "high_fb pace"),
        DM=(80, "destroyer"), CM=(83, "press_resistant creator"), AM=(82, "creator"),
        LW=(80, "pace dribble"), RW=(85, "dribble creator"), ST=(82, "pace poacher"),
        names=dict(RW="Mahrez", CM="Bennacer", ST="Amoura", RB="Atal",
                   AM="Gouiri", LCB="Mandi", GK="Mandrea")),
    "Ghana": dict(style="transition", coach=75,
        GK=(78, ""), LB=(78, "high_fb"), LCB=(80, "aerial"),
        RCB=(79, "recovery_pace"), RB=(79, "high_fb pace"),
        DM=(83, "destroyer press_resistant"), CM=(80, "press_resistant"),
        AM=(85, "creator dribble pace"),
        LW=(81, "pace dribble"), RW=(80, "pace"), ST=(80, "pace poacher"),
        names=dict(AM="Kudus", DM="Partey", LW="Sulemana", ST="Semenyo",
                   RW="J.Ayew", LCB="Djiku", GK="Ati-Zigi")),
    "Paraguay": dict(style="counter", coach=74,
        GK=(79, "shot_stopper"), LB=(76, ""), LCB=(80, "aerial"),
        RCB=(79, ""), RB=(78, "high_fb"),
        DM=(80, "destroyer"), CM=(79, "press_resistant"), AM=(80, "creator dribble"),
        LW=(79, "dribble"), RW=(81, "pace"), ST=(79, "poacher aerial"),
        names=dict(RW="Almiron", AM="Enciso", ST="Sanabria", LCB="Balbuena",
                   DM="Cubas", GK="Coronel")),
}

_SLOTS = ("GK", "LB", "LCB", "RCB", "RB", "DM", "CM", "AM", "LW", "RW", "ST")


def _base_from_units(s):
    """Decompose a team's 4 unit ratings into flat positional slots."""
    gk, df, md, at = s["gk"], s["df"], s["md"], s["at"]
    return {
        "GK": gk, "LB": df - 2, "LCB": df, "RCB": df, "RB": df - 2,
        "DM": md - 1, "CM": md, "AM": (md + at) / 2,
        "LW": at, "RW": at, "ST": at,
    }


def profile(team, elo):
    """Return {slot: {'r': rating, 'tags': set, 'name': str}}, plus team meta.
    For teams with a FLANK entry the 4 unit ratings are DERIVED from the authored
    slots (so units stay consistent with the flank detail), and style/coach come
    from the FLANK entry if given, else from wc2026_squads."""
    s = squads.get_squad(team, elo)
    base = _base_from_units(s)
    fl = FLANK.get(team, {})
    names = fl.get("names", {})
    slots = {}
    for k in _SLOTS:
        if k in fl:
            r, tags = fl[k]
        else:
            r, tags = base[k], ""
        slots[k] = {"r": float(r), "tags": set(tags.split()) if tags else set(),
                    "name": names.get(k, "")}
    if fl:
        df = (slots["LB"]["r"] + slots["LCB"]["r"] + slots["RCB"]["r"] + slots["RB"]["r"]) / 4
        md = (slots["DM"]["r"] + slots["CM"]["r"] + slots["AM"]["r"]) / 3
        at = (slots["LW"]["r"] + slots["RW"]["r"] + slots["ST"]["r"]) / 3
        units = dict(gk=slots["GK"]["r"], df=df, md=md, at=at)
        style = fl.get("style", s["style"])
        coach = fl.get("coach", s["coach"])
        detailed = True
    else:
        units = dict(gk=s["gk"], df=s["df"], md=s["md"], at=s["at"])
        style, coach, detailed = s["style"], s["coach"], s.get("detailed", False)
    form, chem = formations.team_shape(team, style)
    return {"team": team, "slots": slots, "coach": coach, "style": style,
            "detailed": detailed, "units": units, "form": form, "chem": chem}


# ---- lane duels ------------------------------------------------------------
def _has(slot, tag):
    return tag in slot["tags"]


def _channel(att_wide, att_fb, att_support, def_fb, def_cb, def_cover, side, an, dn):
    """Score one wide channel. att_* and def_* are slot dicts. Returns
    (threat_xg, reasons[]). att_wide is the winger, att_fb the overlapping
    full-back, att_support the AM; def_fb the opposing full-back, def_cb the
    near centre-back, def_cover the screening midfielder."""
    reasons = []
    # raw quality of the attacking trio's wide overload vs the defensive trio
    atk = 0.55 * att_wide["r"] + 0.25 * att_fb["r"] + 0.20 * att_support["r"]
    dfn = 0.50 * def_fb["r"] + 0.35 * def_cb["r"] + 0.15 * def_cover["r"]
    threat = (atk - dfn) / 60.0          # ~ xG units per channel
    # --- tag nuance ---
    # winger pace into the space a high/attacking full-back vacates
    if _has(att_wide, "pace") and _has(def_fb, "high_fb"):
        threat += 0.10
        reasons.append(f"{an} {att_wide['name'] or side+' wing'} runs in behind "
                       f"{dn} {def_fb['name'] or 'FB'} who pushes high ({side} channel)")
    # winger dribble vs a slow full-back with no cover pace
    if _has(att_wide, "dribble") and not _has(def_fb, "recovery_pace") and att_wide["r"] - def_fb["r"] >= 6:
        threat += 0.06
        reasons.append(f"{an} {att_wide['name'] or side+' wing'} beats {dn} "
                       f"{def_fb['name'] or 'FB'} 1v1")
    # overlapping full-back adds a second wave if unopposed by a winger tracking
    if _has(att_fb, "high_fb") and att_fb["r"] >= 80:
        threat += 0.03
    return max(-0.15, min(0.45, threat)), reasons


def lane_duels(ph, pa):
    """Compute the full lane-by-lane read for home (ph) vs away (pa) profiles.
    Returns dict: per-channel home/away threats, midfield control, central
    threats, aerial/set-piece notes, and reasons."""
    H, A = ph["slots"], pa["slots"]
    out = {"reasons_h": [], "reasons_a": []}

    # HOME attacking LEFT (LW+LB) vs AWAY right defence (RB+RCB)
    lh, r = _channel(H["LW"], H["LB"], H["AM"], A["RB"], A["RCB"], A["DM"],
                     "left", ph["team"], pa["team"])
    out["reasons_h"] += r
    # HOME attacking RIGHT (RW+RB) vs AWAY left defence (LB+LCB)
    rh, r = _channel(H["RW"], H["RB"], H["AM"], A["LB"], A["LCB"], A["DM"],
                     "right", ph["team"], pa["team"])
    out["reasons_h"] += r
    # AWAY attacking left/right vs HOME defence
    la, r = _channel(A["LW"], A["LB"], A["AM"], H["RB"], H["RCB"], H["DM"],
                     "left", pa["team"], ph["team"])
    out["reasons_a"] += r
    ra, r = _channel(A["RW"], A["RB"], A["AM"], H["LB"], H["LCB"], H["DM"],
                     "right", pa["team"], ph["team"])
    out["reasons_a"] += r

    # CENTRAL: striker + AM vs CB pair + GK
    def central(att_st, att_am, def_lcb, def_rcb, def_gk, an, dn):
        rs = []
        atk = 0.6 * att_st["r"] + 0.4 * att_am["r"]
        dfn = 0.45 * def_lcb["r"] + 0.45 * def_rcb["r"] + 0.10 * def_gk["r"]
        t = (atk - dfn) / 70.0
        if _has(att_st, "aerial") and not (_has(def_lcb, "aerial") or _has(def_rcb, "aerial")):
            t += 0.05
            rs.append(f"{an} {att_st['name'] or 'ST'} wins aerial duels vs {dn}'s CB pair")
        if _has(def_gk, "shot_stopper"):
            t -= 0.04
        return max(-0.15, min(0.4, t)), rs
    ch, r = central(H["ST"], H["AM"], A["LCB"], A["RCB"], A["GK"], ph["team"], pa["team"])
    out["reasons_h"] += r
    ca, r = central(A["ST"], A["AM"], H["LCB"], H["RCB"], H["GK"], pa["team"], ph["team"])
    out["reasons_a"] += r

    # MIDFIELD CONTROL: territory & ability to sustain pressure / spring counters
    mid_h = (H["DM"]["r"] + H["CM"]["r"] + H["AM"]["r"]) / 3.0
    mid_a = (A["DM"]["r"] + A["CM"]["r"] + A["AM"]["r"]) / 3.0
    midedge = (mid_h - mid_a) / 100.0
    if abs(midedge) > 0.03:
        w = ph["team"] if midedge > 0 else pa["team"]
        out["midfield_note"] = f"{w} control midfield (+{abs(midedge)*100:.0f} unit)"
    else:
        out["midfield_note"] = "midfield evenly matched"

    out["channels_h"] = {"left": round(lh, 3), "right": round(rh, 3), "central": round(ch, 3)}
    out["channels_a"] = {"left": round(la, 3), "right": round(ra, 3), "central": round(ca, 3)}
    out["midfield_edge"] = midedge
    return out


def player_xg(th, ta, elo_h, elo_a, ph=None, pa=None):
    """Player/flank expected-goals estimate + structured info, replacing
    wc2026_squads.player_lambdas with lane resolution. Venue handled elsewhere.
    Optional ph/pa accept prebuilt profiles to avoid rebuilding them."""
    if ph is None:
        ph = profile(th, elo_h)
    if pa is None:
        pa = profile(ta, elo_a)
    d = lane_duels(ph, pa)
    base = 1.25
    lam_h = base + sum(d["channels_h"].values()) + 0.6 * d["midfield_edge"]
    lam_a = base + sum(d["channels_a"].values()) - 0.6 * d["midfield_edge"]

    # FORMATION shape effects (midfield numbers, wing-backs, low block, 2 strikers)
    dh_f, da_f, mid_f, freasons = formations.effects(ph["form"], pa["form"], th, ta)
    lam_h += dh_f + 0.6 * mid_f
    lam_a += da_f - 0.6 * mid_f

    # SQUAD CHEMISTRY (cohesion edge — a drilled unit beats thrown-together talent)
    coh = (ph["chem"] - pa["chem"]) / 100.0
    lam_h += 0.18 * coh
    lam_a -= 0.18 * coh
    chem_reasons = []
    if abs(coh) > 0.03:
        more, less = (th, ta) if coh > 0 else (ta, th)
        chem_reasons.append(f"{more} the more settled/cohesive side "
                            f"(chemistry {ph['chem']:.0f} v {pa['chem']:.0f})")

    lam_h = max(0.25, min(3.6, lam_h))
    lam_a = max(0.25, min(3.6, lam_a))
    d["shape"] = {"form": (ph["form"], pa["form"]), "chem": (ph["chem"], pa["chem"]),
                  "delta": (round(dh_f, 3), round(da_f, 3)), "mid": round(mid_f, 3),
                  "reasons": freasons + chem_reasons}
    return lam_h, lam_a, {"ph": ph, "pa": pa, "duels": d}


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    R = __import__("json").load(open(os.path.join(os.path.dirname(__file__), "wc2026_ratings.json"), encoding="utf-8"))
    ELO = R["elo"]
    for th, ta in [("Brazil", "Morocco"), ("France", "Spain")]:
        lh, la, info = player_xg(th, ta, ELO[th], ELO[ta])
        print(f"\n=== {th} vs {ta}  player-xG {lh:.2f}-{la:.2f} ===")
        print("  channels", th, info["duels"]["channels_h"])
        print("  channels", ta, info["duels"]["channels_a"])
        print(" ", info["duels"]["midfield_note"])
        for r in info["duels"]["reasons_h"] + info["duels"]["reasons_a"]:
            print("   -", r)
