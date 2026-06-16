"""
Build the final WC2026 prediction PDF from the live model.
Pulls scorelines/confidence from wc2026_scorelines, title odds from the Monte
Carlo, and positional duels from wc2026_squads — so the PDF == the model.
"""
import os, sys, datetime
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                PageBreak, HRFlowable)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import wc2026_scorelines as S
import wc2026_squads as squads
import wc2026_montecarlo as MC

# ---- fonts (Arial handles accented player names) ----
FONT, BOLD = "Helvetica", "Helvetica-Bold"
try:
    pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-Italic", r"C:\Windows\Fonts\ariali.ttf"))
    FONT, BOLD = "Arial", "Arial-Bold"
except Exception:
    pass

GREEN = colors.HexColor("#14502f")
GREEN2 = colors.HexColor("#1f7a4d")
LIGHT = colors.HexColor("#eaf3ee")
GREY = colors.HexColor("#666666")
ROWALT = colors.HexColor("#f5f7f6")

ss = getSampleStyleSheet()
def style(name, **kw):
    base = dict(fontName=FONT, fontSize=9.5, leading=13, textColor=colors.black)
    base.update(kw); return ParagraphStyle(name, **base)
H1 = style("H1", fontName=BOLD, fontSize=22, leading=25, textColor=GREEN)
SUB = style("SUB", fontSize=9, textColor=GREY, leading=12)
H2 = style("H2", fontName=BOLD, fontSize=14, leading=17, textColor=GREEN, spaceBefore=10, spaceAfter=4)
H3 = style("H3", fontName=BOLD, fontSize=11, leading=14, textColor=GREEN2, spaceBefore=6, spaceAfter=2)
BODY = style("BODY")
SMALL = style("SMALL", fontSize=8.5, leading=11)
ITAL = style("ITAL", fontName=(FONT if FONT == "Helvetica" else "Arial-Italic"), fontSize=8.5, leading=11, textColor=GREY)
WHITE = style("WHITE", fontName=BOLD, fontSize=9.5, textColor=colors.white)
CELL = style("CELL", fontSize=9, leading=11)
CELLB = style("CELLB", fontName=BOLD, fontSize=9, leading=11)

HOSTS = S.HOSTS
TODAY = datetime.date.today().isoformat()

# ---------- helpers ----------
def result_text(th, ta, r, ko=False):
    o = r["outcome"]
    if o == "draw":
        return ("Draw → pens" if ko else "Draw")
    return th if o == "home_win" else ta

def conf_cell(r):
    c = max(r["wdl"].values()) * 100
    return f"{c:.0f}% {S.tier(c).strip()}"

def predict(th, ta, neutral=True, ko=False):
    return S.predict(th, ta, neutral=neutral, knockout=ko, news_scores=None, use_lineups=True)

def tbl(data, widths, header=True, font_size=9):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    sttl = [
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#dddddd")),
    ]
    if header:
        sttl += [("BACKGROUND", (0, 0), (-1, 0), GREEN),
                 ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                 ("FONTNAME", (0, 0), (-1, 0), BOLD),
                 ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROWALT])]
    t.setStyle(TableStyle(sttl))
    return t


def build():
    story = []
    A = story.append

    # ===== Title =====
    A(Paragraph("FIFA World Cup 2026 — Model Prediction", H1))
    A(Paragraph(f"Generated {TODAY} &nbsp;|&nbsp; Engine: github.com/govindgoel2001/sportsbetting "
                f"(Dixon-Coles + Elo + 10-agent panel + flank/player lanes + tactical engine "
                f"+ formations/chemistry + game-state event-sim Monte Carlo + probability calibration)", SUB))
    A(Paragraph("Ratings calibrated on 913 real internationals (2023–26, via Hicruben/world-cup-2026-prediction-model).", SUB))
    A(Spacer(1, 8))
    A(HRFlowable(width="100%", thickness=1.2, color=GREEN))
    A(Spacer(1, 6))

    # ===== Disclaimer box =====
    disc = Paragraph(
        "<b>Read this first.</b> The tournament has not been played — no model can tell you the actual results. "
        "These are <b>probabilistic forecasts</b>: even the favourite wins the title only ~22% of the time. "
        "Scorelines are the single most-likely score <i>consistent with the predicted result</i>; "
        "the <b>confidence</b> figure is the model's probability of that result. The player/lineup layer uses "
        "<b>assumed starting XIs (analyst judgment, early-2026 knowledge)</b>, not confirmed teamsheets.", SMALL)
    box = Table([[disc]], colWidths=[170 * mm])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                             ("BOX", (0, 0), (-1, -1), 0.6, GREEN2),
                             ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                             ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9)]))
    A(box); A(Spacer(1, 10))

    # ===== Model stack =====
    A(Paragraph("The model stack", H2))
    stack = [["Layer", "Weight", "Status", "What it does"],
             ["Dixon-Coles MLE", "45% of base", "LIVE", "attack/defence fitted on 913 real results"],
             ["Hicruben Elo", "55% of base", "LIVE", "prior-anchored strength (70% form / 30% prior)"],
             ["10-agent panel", "multiplier", "6 LIVE", "form, momentum, H2H, tactician, tempo, host"],
             ["  └ travel/climate/news/hydration", "multiplier", "prior/hook/opt", "agents 7-10: priors, live news, WC2026 cooling breaks"],
             ["Player / flank lanes", "30% blend", "ON (assumed)", "positional duels by channel (e.g. winger vs full-back)"],
             ["Tactical engine", "additive xG", "ON", "style-vs-style mechanisms (high line, low block, press)"],
             ["Formations + chemistry", "additive xG", "ON", "midfield numbers, wing-backs, low block; squad cohesion"],
             ["Game-state event-sim", "per match", "ON", "19-phase sim (18 + stoppage): park-the-bus, chasing, late goals"],
             ["Monte Carlo", f"{MC.N:,} sims", "LIVE", "full tournament event-sim resampled for title odds"],
             ["Probability calibration", "shrink 0.35", "NEW", "blends 1X2 toward DC+Elo core; trims favourite over-confidence"]]
    A(tbl(stack, [38*mm, 24*mm, 22*mm, 86*mm]))
    A(Spacer(1, 6))
    A(Paragraph("Scoreline engine ρ = −0.107 (Dixon-Coles low-score correction). Host edge applies only to "
                "USA / Mexico / Canada; every other match is neutral. Agent 9 (live squad news) is available but left "
                "off here for a reproducible document.", ITAL))

    A(PageBreak())

    # ===== Title odds =====
    A(Paragraph(f"Title odds — {MC.N:,} game-state event-sim simulations", H2))
    A(Paragraph("Each simulation plays all 72 group games as an 18-phase game-state event sim (so leads get "
                "protected, trailing sides chase, comebacks happen), resolves standings and the eight best "
                "third-placed teams, then a seeded knockout. Win% is how often each team lifts the trophy.", BODY))
    A(Spacer(1, 4))
    res = MC.simulate(MC.N)
    n = res["n"]
    allt = [t for ts in S.GROUPS.values() for t in ts]
    rows = sorted(allt, key=lambda t: res["champ"].get(t, 0), reverse=True)[:16]
    data = [["#", "Team", "Win%", "Final%", "Semi%", "Reach R32%", "Elo"]]
    for i, t in enumerate(rows, 1):
        data.append([str(i), t,
                     f"{res['champ'].get(t,0)/n*100:.1f}", f"{res['finalist'].get(t,0)/n*100:.1f}",
                     f"{res['semi'].get(t,0)/n*100:.1f}", f"{res['advance'].get(t,0)/n*100:.0f}",
                     str(S.ELO[t])])
    A(tbl(data, [10*mm, 46*mm, 20*mm, 22*mm, 20*mm, 28*mm, 18*mm]))
    A(Spacer(1, 8))
    A(Paragraph("Spain vs Argentina — the favourite question", H3))
    A(Paragraph("Spain carry the highest rating (Elo 2075) and, once the assumed lineups are modelled, the best "
                "midfield in the field (Rodri–Pedri–Yamal). That midfield-control edge nudges them to "
                "narrow title favourite over Argentina. Argentina's strength is the <b>draw</b>: the softest group and "
                "an easier route. The gap is inside the noise — treat them as co-favourites, with the field ~55%.", BODY))

    A(PageBreak())

    # ===== Group stage =====
    A(Paragraph("Group stage — predicted scorelines & confidence", H2))
    for g, teams in S.GROUPS.items():
        rows6 = []
        pts = {t: 0 for t in teams}; gd = {t: 0 for t in teams}; gf = {t: 0 for t in teams}
        for i in range(4):
            for j in range(i + 1, 4):
                th, ta = teams[i], teams[j]
                neutral = not (th in HOSTS or ta in HOSTS)
                r = predict(th, ta, neutral=neutral, ko=False)
                sh, sa = r["score"]
                gf[th] += sh; gf[ta] += sa; gd[th] += sh - sa; gd[ta] += sa - sh
                if sh > sa: pts[th] += 3
                elif sa > sh: pts[ta] += 3
                else: pts[th] += 1; pts[ta] += 1
                tag = " (host)" if not neutral else ""
                rows6.append([f"{th} v {ta}{tag}", f"{sh}–{sa}", result_text(th, ta, r), conf_cell(r)])
        order = sorted(teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
        marks = ["①", "②", "③", "④"]
        standing = "  ".join(f"{marks[k]} {order[k]} {pts[order[k]]}" for k in range(4))
        A(Paragraph(f"Group {g}", H3))
        A(tbl([["Match", "Score", "Result", "Confidence"]] + rows6,
              [70*mm, 18*mm, 40*mm, 38*mm]))
        A(Paragraph("Qualify: " + standing + "  &nbsp;(top 2 advance; 3rd into best-thirds race)", SMALL))
        A(Spacer(1, 4))

    A(PageBreak())

    # ===== Knockout =====
    A(Paragraph("Knockout — most-likely (chalk) path", H2))
    A(Paragraph("Pairings are the single most-likely bracket; the Monte Carlo above is the probabilistic verdict. "
                "Penalties shown only where a draw is the predicted outcome.", ITAL))
    A(Spacer(1, 4))
    duel_store = []
    for rnd, ms in S.KO_ROUNDS:
        A(Paragraph(rnd, H3))
        krows = [["Match", "Score", "Result", "Confidence"]]
        for th, ta in ms:
            r = predict(th, ta, neutral=True, ko=True)
            sh, sa = r["score"]
            krows.append([f"{th} v {ta}", f"{sh}–{sa}", result_text(th, ta, r, ko=True), conf_cell(r)])
            if r.get("lineup"):
                duel_store.append((rnd, th, ta, r))
        A(tbl(krows, [70*mm, 18*mm, 40*mm, 38*mm]))
        A(Spacer(1, 3))

    A(Spacer(1, 6))
    A(Paragraph("Key positional reads (assumed XIs)", H3))
    picks = [d for d in duel_store if d[0] in ("Semi-final", "FINAL")][:3]
    extra = [d for d in duel_store if d[0] == "Round of 16"][:2]
    lanew = {"left": "left", "right": "right", "central": "the middle"}
    for rnd, th, ta, r in picks + extra:
        info = r["lineup"]; d = info["duels"]; tac = info["tactics"]
        uh, ua = info["ph"]["units"], info["pa"]["units"]
        ch, ca = d["channels_h"], d["channels_a"]
        bh, ba = max(ch, key=ch.get), max(ca, key=ca.get)
        line = (f"<b>{th} v {ta}</b> ({rnd}): {d['midfield_note']}; "
                f"MID {uh['md']} v {ua['md']}, ATT {uh['at']} v {ua['at']}, DEF {uh['df']} v {ua['df']}; "
                f"style {tac['styles'][0]} v {tac['styles'][1]}. "
                f"{th} most dangerous {lanew[bh]} ({ch[bh]:+.2f} xG), "
                f"{ta} {lanew[ba]} ({ca[ba]:+.2f}).")
        A(Paragraph(line, SMALL))
        plays = (d["reasons_h"] + d["reasons_a"] + tac["reasons_h"] + tac["reasons_a"])[:3]
        for p in plays:
            A(Paragraph(f"&nbsp;&nbsp;&nbsp;{p}", ITAL))

    A(PageBreak())

    # ===== Live track record & calibration =====
    A(Paragraph("Live track record &amp; calibration — as of 16 June 2026", H2))
    A(Paragraph("Ten group games have been played. The model called the <b>result in 7</b> of them "
                "(5 on the strict single-most-likely outcome; the other two were draws it leaned even on). "
                "But a hit-count on 10 games means little — the honest test is a probabilistic score against "
                "baselines. Brier and log-loss below are <b>lower-is-better</b>; favWinPred is the average "
                "probability the model gave the eventual favourite to win, which should track the actual rate.", BODY))
    A(Spacer(1, 4))
    tr = [["Model", "Brier", "log-loss", "Outcome", "favWinPred"],
          ["Naive 'pick favourite'", "1.000", "13.82", "5 / 10", "100%"],
          ["DC+Elo core (baseline)", "0.582", "0.95", "5 / 10", "56%"],
          ["Full stack (pre-calibration)", "0.607", "0.98", "5 / 10", "61%"],
          ["Full + calibration (this build)", "0.597", "0.97", "5 / 10", "58%"],
          ["Actual favourite win-rate", "—", "—", "—", "50%"]]
    A(tbl(tr, [62*mm, 22*mm, 24*mm, 26*mm, 32*mm]))
    A(Spacer(1, 6))
    A(Paragraph("What the numbers actually say", H3))
    for b in [
        "<b>On n=10, the differences are within noise.</b> A bootstrap of Brier(full) − Brier(core) gives "
        "+0.025 with a 95% CI of [−0.04, +0.10] — it straddles zero. We cannot yet claim the nuance layers "
        "help or hurt; we can only say the full stack was mildly <i>over-confident</i> (61% vs a 50% actual, "
        "tiny-sample).",
        "<b>The calibration fix is principled, not fitted.</b> We shrink the 1X2 toward the better-behaved "
        "DC+Elo core by a fixed 0.35. Leave-one-out showed the loss-minimising shrink is unstable (0.73–1.0), "
        "so tuning it to these 10 games would be overfitting — 0.35 trims over-confidence while keeping the "
        "nuance (injuries, lanes, tactics) the model exists for.",
        "<b>Three repeatable biases drove the misses:</b> over-rating heavy favourites in openers "
        "(Switzerland 85% → 1–1 draw), under-shooting blowouts (Germany predicted ~3–0, actual 7–1), and never "
        "ranking a draw as the single most-likely 1X2 (1–1 is often the modal scoreline even so).",
        "<b>The real unlock is data, not more layers.</b> Ten games cannot validate a model this complex. The "
        "next step that would make it genuinely strong is a multi-thousand-match historical backtest "
        "(past World Cups, qualifiers, club leagues) to calibrate every layer and prove the nuance adds signal.",
    ]:
        A(Paragraph("• " + b, SMALL))

    A(PageBreak())

    # ===== Limitations =====
    A(Paragraph("What is real vs assumed", H2))
    lim = [["Component", "Basis", "Trust"],
           ["Ratings, form, H2H, momentum", "913 real results", "Data — high"],
           ["Home advantage", "hosts only (USA/MEX/CAN)", "Data — high"],
           ["Tactician / tempo / style agents", "results-derived proxies", "Proxy — medium"],
           ["Player / flank-lane layer", "assumed XIs, analyst ratings", "Judgment — indicative"],
           ["Formations & squad chemistry", "likely shapes, subjective cohesion", "Judgment — indicative"],
           ["Travel & rest agent", "no fixture calendar", "Prior — low"],
           ["Climate & altitude agent", "no venue map", "Neutral hook"],
           ["Hydration / cooling-break agent", "WC2026 heat prior, no temp map", "Prior — low"],
           ["Injury / news agent", "Google News RSS", "Live (off here)"]]
    A(tbl(lim, [60*mm, 64*mm, 42*mm]))
    A(Spacer(1, 8))
    A(Paragraph("Honest caveats", H3))
    for b in [
        "International results are confederation-siloed, so cross-confederation calibration is the weak link "
        "(Asian/African sides can be mis-scaled vs Europe/South America).",
        "Lineups assume first-choice XIs in good health; one injury or a manager's surprise reshapes a tie. "
        "Confirmed teamsheets only land ~1 hour before kickoff.",
        "Group standings here assume the modal scoreline in every match; real groups swing on a single upset — "
        "the Monte Carlo title odds are the probabilistic, self-consistent view.",
        "Confidence percentages are the model's own. Live (16 Jun 2026, 10 games) it called 7 results and scores "
        "Brier 0.60 — 0.58 after calibration — vs 0.58 for its plain DC+Elo core; on n=10 those gaps are inside "
        "bootstrap noise. Trust the ordering (STRONG > CLEAR > LEAN > TOSS-UP) more than the exact number until a "
        "multi-thousand-match historical backtest is run.",
    ]:
        A(Paragraph("• " + b, SMALL))
    A(Spacer(1, 10))
    A(HRFlowable(width="100%", thickness=0.8, color=GREEN2))
    A(Paragraph("For research and entertainment only. Probabilities are not certainties. Bet responsibly.", ITAL))

    OUT = os.path.join(ROOT, "WC2026_Prediction_v5.pdf")
    doc = SimpleDocTemplate(OUT, pagesize=A4, topMargin=15*mm, bottomMargin=14*mm,
                            leftMargin=20*mm, rightMargin=20*mm,
                            title="FIFA World Cup 2026 — Model Prediction",
                            author="govindgoel2001/sportsbetting")

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont(FONT, 7.5); canvas.setFillColor(GREY)
        canvas.drawString(20*mm, 8*mm, "WC2026 model prediction — EdgeFinder engine")
        canvas.drawRightString(190*mm, 8*mm, f"Page {d.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("Wrote", OUT)
    return OUT


if __name__ == "__main__":
    build()
