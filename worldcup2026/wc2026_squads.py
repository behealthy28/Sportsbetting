"""
PHASE 3 — player / lineup / coach-tactics layer (ANALYST ASSUMPTIONS).

IMPORTANT EPISTEMIC NOTE: everything here is my best-judgment assumption as of
early 2026 — assumed first-choice XIs, subjective 0-100 unit ratings, coach and
style calls. It is NOT data. Confirmed XIs only exist ~1h pre-kickoff. For ~18
teams these reflect real squad knowledge; the rest are anchored to overall Elo
with a flat profile (flagged detailed=False).

Model: each team is rated by positional UNIT (GK / DEF / MID / ATT) plus COACH
and a play STYLE. Per match we compute a player-model expected-goals estimate
from unit matchups + midfield control + style clash, then scorelines.py blends
it at W_PLAYER with the data model.
"""

# 0-100 unit ratings. detailed=True => personnel-specific judgment.
DETAILED = {
    "Argentina": dict(gk=87, df=82, md=85, at=84, coach=88, style="transition",
        key=dict(GK="E. Martínez", DF="Romero/Otamendi", MD="Mac Allister/E.Fernández/De Paul",
                 AT="J. Álvarez/Messi")),
    "Spain": dict(gk=80, df=84, md=92, at=88, coach=85, style="possession",
        key=dict(GK="Simón", DF="Le Normand/Cubarsí/Cucurella", MD="Rodri/Pedri/Fabián",
                 AT="Yamal/N. Williams/Oyarzabal")),
    "France": dict(gk=87, df=86, md=83, at=90, coach=80, style="counter",
        key=dict(GK="Maignan", DF="Saliba/Upamecano/T. Hernández", MD="Tchouaméni/Camavinga",
                 AT="Mbappé/Dembélé/Thuram")),
    "England": dict(gk=80, df=83, md=86, at=86, coach=79, style="possession",
        key=dict(GK="Pickford", DF="Stones/Guéhi/Shaw", MD="Rice/Bellingham/Foden",
                 AT="Kane/Saka")),
    "Brazil": dict(gk=86, df=82, md=80, at=88, coach=80, style="wing",
        key=dict(GK="Alisson", DF="Marquinhos/Militão", MD="B. Guimarães/Paquetá",
                 AT="Vinícius/Rodrygo/Raphinha")),
    "Portugal": dict(gk=82, df=80, md=85, at=85, coach=78, style="possession",
        key=dict(GK="Costa", DF="Dias/Cancelo", MD="B. Fernandes/Vitinha/B. Silva",
                 AT="Leão/R. Leão/Ronaldo")),
    "Germany": dict(gk=83, df=78, md=87, at=81, coach=81, style="press",
        key=dict(GK="ter Stegen", DF="Tah/Rüdiger", MD="Kimmich/Wirtz/Musiala",
                 AT="Havertz/Füllkrug")),
    "Netherlands": dict(gk=82, df=85, md=80, at=81, coach=77, style="balanced",
        key=dict(GK="Verbruggen", DF="van Dijk/Aké/Dumfries", MD="Gravenberch/Reijnders",
                 AT="Gakpo/Simons")),
    "Belgium": dict(gk=84, df=74, md=80, at=80, coach=73, style="transition",
        key=dict(GK="Courtois", DF="Theate/Castagne", MD="De Bruyne/Onana",
                 AT="Lukaku/Doku")),
    "Croatia": dict(gk=80, df=76, md=85, at=72, coach=80, style="possession",
        key=dict(GK="Livaković", DF="Gvardiol/Šutalo", MD="Modrić/Kovačić/Sučić",
                 AT="Kramarić/Budimir")),
    "Uruguay": dict(gk=78, df=80, md=81, at=81, coach=83, style="press",
        key=dict(GK="Rochet", DF="Araújo/Giménez", MD="Valverde/Ugarte/Bentancur",
                 AT="Núñez/Pellistri")),
    "Colombia": dict(gk=76, df=77, md=82, at=80, coach=77, style="wing",
        key=dict(GK="Vargas", DF="Lucumí/Mojica", MD="J. Rodríguez/Lerma/Ríos",
                 AT="L. Díaz/Córdoba")),
    "Morocco": dict(gk=83, df=84, md=80, at=79, coach=81, style="counter",
        key=dict(GK="Bounou", DF="Hakimi/Saïss/Mazraoui", MD="Amrabat/Ounahi/Amallah",
                 AT="En-Nesyri/Ziyech")),
    "Japan": dict(gk=76, df=77, md=83, at=79, coach=79, style="press",
        key=dict(GK="Suzuki", DF="Itakura/Tomiyasu", MD="Endo/Kubo/Kamada",
                 AT="Mitoma/Ueda")),
    "Mexico": dict(gk=79, df=75, md=76, at=75, coach=73, style="possession",
        key=dict(GK="Ochoa/Malagón", DF="Montes/Gallardo", MD="Edson Álvarez/Chávez",
                 AT="Raúl Jiménez/Lozano")),
    "USA": dict(gk=78, df=75, md=81, at=76, coach=78, style="transition",
        key=dict(GK="Turner", DF="Robinson/Richards", MD="Pulisic/McKennie/Adams",
                 AT="Weah/Balogun")),
    "Switzerland": dict(gk=81, df=78, md=78, at=74, coach=72, style="compact",
        key=dict(GK="Sommer", DF="Akanji/Rodríguez", MD="Xhaka/Freuler",
                 AT="Embolo/Ndoye")),
    "Senegal": dict(gk=81, df=81, md=78, at=81, coach=75, style="counter",
        key=dict(GK="É. Mendy", DF="Koulibaly/Diallo", MD="I. Gueye/P. Gueye",
                 AT="Sarr/Jackson")),
}

# style clash: additive goals for the ATTACKING side given (att_style vs def_style)
def _style_mod(att_style, def_style):
    m = 0.0
    if att_style in ("counter", "transition") and def_style in ("possession", "press"):
        m += 0.10          # space in behind a high line
    if att_style == "possession" and def_style == "compact":
        m -= 0.08          # hard to break a low block
    if def_style == "press" and att_style in ("possession", "balanced"):
        m -= 0.05          # pressed into mistakes
    return m


def get_squad(team, elo):
    if team in DETAILED:
        s = dict(DETAILED[team]); s["detailed"] = True
        return s
    r = round(55 + (elo - 1500) * (88 - 55) / (2075 - 1500))   # anchor to Elo
    r = max(48, min(90, r))
    return dict(gk=r, df=r, md=r, at=r, coach=r, style="balanced",
                key=dict(GK="", DF="", MD="", AT=""), detailed=False)


def _off(s):  return 0.50 * s["at"] + 0.35 * s["md"] + 0.15 * s["coach"]
def _sol(s):  return 0.45 * s["df"] + 0.25 * s["md"] + 0.20 * s["gk"] + 0.10 * s["coach"]


def player_lambdas(th, ta, elo_h, elo_a):
    """Player-model expected goals (neutral talent; venue handled by data model)."""
    sh, sa = get_squad(th, elo_h), get_squad(ta, elo_a)
    lam_h = 1.35 + (_off(sh) - _sol(sa)) / 38.0
    lam_a = 1.35 + (_off(sa) - _sol(sh)) / 38.0
    # midfield control -> territory: stronger midfield boosts own attack, suppresses opp
    midctrl = (sh["md"] - sa["md"]) / 100.0
    lam_h += midctrl * 0.6
    lam_a -= midctrl * 0.6
    # style clash
    lam_h += _style_mod(sh["style"], sa["style"])
    lam_a += _style_mod(sa["style"], sh["style"])
    lam_h = max(0.25, min(3.6, lam_h))
    lam_a = max(0.25, min(3.6, lam_a))
    info = {
        "h": sh, "a": sa,
        "units": {"GK": (sh["gk"], sa["gk"]), "DEF": (sh["df"], sa["df"]),
                  "MID": (sh["md"], sa["md"]), "ATT": (sh["at"], sa["at"]),
                  "COACH": (sh["coach"], sa["coach"])},
        "midfield_edge": th if sh["md"] > sa["md"] else ta,
        "style": (sh["style"], sa["style"]),
    }
    return lam_h, lam_a, info


def duel_lines(th, ta, info):
    """Key positional matchups for marquee output."""
    sh, sa = info["h"], info["a"]
    out = []
    if sh.get("detailed") or sa.get("detailed"):
        out.append(f"      ATT {th} [{sh['key'].get('AT','?')}] vs DEF {ta} [{sa['key'].get('DF','?')}]")
        out.append(f"      MID {th} [{sh['key'].get('MD','?')}]  vs  [{sa['key'].get('MD','?')}] {ta}"
                   f"  -> midfield edge: {info['midfield_edge']}")
        out.append(f"      ATT {ta} [{sa['key'].get('AT','?')}] vs DEF {th} [{sh['key'].get('DF','?')}]")
    u = info["units"]
    out.append(f"      units GK {u['GK'][0]}/{u['GK'][1]}  DEF {u['DEF'][0]}/{u['DEF'][1]}  "
               f"MID {u['MID'][0]}/{u['MID'][1]}  ATT {u['ATT'][0]}/{u['ATT'][1]}  "
               f"COACH {u['COACH'][0]}/{u['COACH'][1]}  | style {info['style'][0]} vs {info['style'][1]}")
    return out
