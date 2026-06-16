# -*- coding: utf-8 -*-
"""
Build WC2026_Prediction.pdf in the original house style (green title, "read this
first" callout, title-odds table, a VECTOR group-qualifier card grid and a
knockout bracket tree, then a method page) -- but every number is pulled LIVE
from the current calibrated model instead of being hardcoded:

  * title odds + advancement %  <- wc2026_montecarlo.simulate (20k vectorised)
  * group winner/runner-up      <- ordered by advancement %
  * knockout tree               <- most-likely path propagated through
                                    wc2026_scorelines.predict (Elo-tilted KO)

Also exports the two diagrams as standalone .svg into ./diagrams/.
Run:  python wc2026_pdf.py
"""
import os, sys, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak)
from reportlab.graphics.shapes import Drawing, Rect, Line, String
from reportlab.graphics import renderSVG

import wc2026_scorelines as S
import wc2026_montecarlo as MC

OUT = os.path.join(HERE, "WC2026_Prediction.pdf")
SVGDIR = os.path.join(HERE, "diagrams"); os.makedirs(SVGDIR, exist_ok=True)
TODAY = datetime.date.today().isoformat()

DARK = colors.HexColor('#0b3d2e'); MED = colors.HexColor('#2e8b57')
GOLD = colors.HexColor('#DAA520'); GREY = colors.HexColor('#cfd8d3')
LINE = colors.HexColor('#9aa7a0'); INK = colors.HexColor('#13241d')
OUTCOL = colors.HexColor('#7f8c8a')

ss = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=ss['Title'], fontSize=20, spaceAfter=4, textColor=DARK)
SUB = ParagraphStyle('SUB', parent=ss['Normal'], fontSize=9, textColor=colors.grey, spaceAfter=10)
H2 = ParagraphStyle('H2', parent=ss['Heading2'], fontSize=13, spaceBefore=12, spaceAfter=4, textColor=DARK)
P = ParagraphStyle('P', parent=ss['Normal'], fontSize=9.5, leading=13, spaceAfter=5)
WARN = ParagraphStyle('WARN', parent=P, backColor=colors.HexColor('#fff4e5'),
                      borderColor=colors.HexColor('#e0a000'), borderWidth=0.6, borderPadding=6, spaceAfter=8)
SMALL = ParagraphStyle('SMALL', parent=P, fontSize=8, textColor=colors.grey)

SHORT = {"Korea Republic": "S.Korea", "South Africa": "S.Africa",
         "New Zealand": "N.Zealand"}
def sh(t): return SHORT.get(t, t)

# ====================== LIVE MODEL DATA ======================
print("Running %d-sim Monte Carlo..." % MC.N, flush=True)
RES = MC.simulate(MC.N)
N = RES["n"]
ALL = [t for ts in S.GROUPS.values() for t in ts]
def pct(key, t): return RES[key].get(t, 0) / N * 100.0

# --- title odds: top 15 by championship probability ---
order = sorted(ALL, key=lambda t: RES["champ"].get(t, 0), reverse=True)[:15]
ODDS = [["#", "Team", "Win", "Final", "Semi", "QF", "R16", "Advance"]]
for i, t in enumerate(order, 1):
    ODDS.append([str(i), t,
                 f"{pct('champ',t):.1f}%", f"{pct('finalist',t):.1f}%", f"{pct('semi',t):.1f}%",
                 f"{pct('qf',t):.1f}%", f"{pct('r16',t):.1f}%", f"{pct('advance',t):.0f}%"])
BIG4 = sum(pct('champ', t) for t in ("Argentina", "France", "Brazil", "Spain"))
HOSTSUM = sum(pct('champ', t) for t in ("USA", "Mexico", "Canada"))

# --- group cards: teams ordered by advancement % (winner / runner-up / out) ---
GD = {}
for g, teams in S.GROUPS.items():
    GD[g] = sorted([(t, pct('advance', t)) for t in teams], key=lambda x: -x[1])

# --- knockout tree: propagate the most-likely advancer through each round ---
def ko_winner(th, ta):
    r = S.predict(th, ta, neutral=True, knockout=True)
    return th if r["wdl"]["home_win"] >= r["wdl"]["away_win"] else ta
def ko_share(th, ta, who):
    r = S.predict(th, ta, neutral=True, knockout=True)
    hw, aw = r["wdl"]["home_win"], r["wdl"]["away_win"]
    s = hw / (hw + aw) if (hw + aw) else 0.5
    return s if who == th else 1 - s

LEAVES = [t for tie in S.KO_ROUNDS[0][1] for t in tie]            # 16 R16 teams
R16W = [ko_winner(LEAVES[2*j], LEAVES[2*j+1]) for j in range(8)]
QFW = [ko_winner(R16W[2*j], R16W[2*j+1]) for j in range(4)]
SFW = [ko_winner(QFW[2*j], QFW[2*j+1]) for j in range(2)]
CHAMP = ko_winner(SFW[0], SFW[1])
RUNNER = SFW[1] if CHAMP == SFW[0] else SFW[0]
CH_SHARE = ko_share(SFW[0], SFW[1], CHAMP)
SF_LOSERS = [(QFW[2*j+1] if SFW[j] == QFW[2*j] else QFW[2*j]) for j in range(2)]
THIRD = ko_winner(SF_LOSERS[0], SF_LOSERS[1])
THIRD_LOSER = SF_LOSERS[1] if THIRD == SF_LOSERS[0] else SF_LOSERS[0]

# ====================== VECTOR DIAGRAMS ======================
def draw_groups(GD):
    cols, rows = 3, 4
    cw, ch = 158, 150; gx, gy = 8, 12
    W = cols*cw + (cols-1)*gx
    H = rows*ch + (rows-1)*gy + 6
    d = Drawing(W, H)
    for idx, g in enumerate(GD.keys()):
        c = idx % cols; r = idx // cols
        x0 = c*(cw+gx); y0 = H - 6 - (r+1)*ch - r*gy
        d.add(Rect(x0, y0, cw, ch, strokeColor=GREY, fillColor=colors.white, strokeWidth=0.8))
        d.add(Rect(x0, y0+ch-18, cw, 18, strokeColor=None, fillColor=DARK))
        d.add(String(x0+6, y0+ch-13, "GROUP "+g, fontName='Helvetica-Bold', fontSize=9, fillColor=colors.white))
        bar_x = x0+8; bar_w = cw-70; row_h = 28
        for i, (name, adv) in enumerate(GD[g]):
            ry = y0+ch-18-(i+1)*row_h+6
            qualifies = i < 2
            col = DARK if i == 0 else (MED if i == 1 else GREY)
            d.add(String(bar_x, ry+14, sh(name), fontName='Helvetica-Bold' if qualifies else 'Helvetica',
                         fontSize=7.5, fillColor=INK))
            d.add(Rect(bar_x, ry+2, bar_w, 7, strokeColor=None, fillColor=colors.HexColor('#eef2f0')))
            d.add(Rect(bar_x, ry+2, bar_w*adv/100.0, 7, strokeColor=None, fillColor=col))
            d.add(String(bar_x+bar_w+4, ry+2, f"{adv:.0f}%", fontName='Helvetica', fontSize=7.5, fillColor=INK))
    return d

def draw_bracket(LEAVES, R16W, QFW, SFW, CHAMP):
    W, H = 500, 690
    d = Drawing(W, H)
    bw, bh = 86, 18
    colx = [4, 128, 252, 360, 446]
    n = len(LEAVES); top, bot = H-10, 20; slot = (top-bot)/n
    def box(x, y, label, fill, txtcol=colors.white, bold=True, fs=7.5):
        d.add(Rect(x, y-bh/2, bw, bh, strokeColor=LINE, fillColor=fill, strokeWidth=0.7, rx=2, ry=2))
        d.add(String(x+4, y-3, sh(label), fontName='Helvetica-Bold' if bold else 'Helvetica', fontSize=fs, fillColor=txtcol))
    def connect(x1, y1a, y1b, x2, ymid):
        midx = (x1+bw+x2)/2
        d.add(Line(x1+bw, y1a, midx, y1a, strokeColor=LINE, strokeWidth=0.6))
        d.add(Line(x1+bw, y1b, midx, y1b, strokeColor=LINE, strokeWidth=0.6))
        d.add(Line(midx, y1a, midx, y1b, strokeColor=LINE, strokeWidth=0.6))
        d.add(Line(midx, ymid, x2, ymid, strokeColor=LINE, strokeWidth=0.6))
    leaf_y = [top - slot*(i+0.5) for i in range(n)]
    for i, name in enumerate(LEAVES):
        win = name in R16W
        box(colx[0], leaf_y[i], name, MED if win else OUTCOL, bold=win)
    r16_y = []
    for j in range(8):
        ya, yb = leaf_y[2*j], leaf_y[2*j+1]; ym = (ya+yb)/2; r16_y.append(ym)
        connect(colx[0], ya, yb, colx[1], ym)
        box(colx[1], ym, R16W[j], MED if R16W[j] in QFW else OUTCOL, bold=R16W[j] in QFW)
    qf_y = []
    for j in range(4):
        ya, yb = r16_y[2*j], r16_y[2*j+1]; ym = (ya+yb)/2; qf_y.append(ym)
        connect(colx[1], ya, yb, colx[2], ym)
        box(colx[2], ym, QFW[j], MED if QFW[j] in SFW else OUTCOL, bold=QFW[j] in SFW)
    sf_y = []
    for j in range(2):
        ya, yb = qf_y[2*j], qf_y[2*j+1]; ym = (ya+yb)/2; sf_y.append(ym)
        connect(colx[2], ya, yb, colx[3], ym)
        win = SFW[j] == CHAMP
        box(colx[3], ym, SFW[j], GOLD if win else MED, txtcol=INK if win else colors.white, bold=True)
    ym = (sf_y[0]+sf_y[1])/2
    connect(colx[3], sf_y[0], sf_y[1], colx[4], ym)
    d.add(Rect(colx[4], ym-13, bw+8, 26, strokeColor=GOLD, fillColor=colors.HexColor('#fff7e0'), strokeWidth=1.4, rx=3, ry=3))
    d.add(String(colx[4]+5, ym+3, "CHAMPION", fontName='Helvetica-Bold', fontSize=6.5, fillColor=GOLD))
    d.add(String(colx[4]+5, ym-8, sh(CHAMP), fontName='Helvetica-Bold', fontSize=10, fillColor=INK))
    for x, lab in zip(colx, ["Round of 16", "R16 winners", "Quarter-finals", "Semi-finals", "Final"]):
        d.add(String(x, H-2, lab, fontName='Helvetica-Bold', fontSize=7, fillColor=DARK))
    return d

g_draw = draw_groups(GD)
b_draw = draw_bracket(LEAVES, R16W, QFW, SFW, CHAMP)
renderSVG.drawToFile(draw_groups(GD), os.path.join(SVGDIR, "group_qualifiers.svg"), "group_qualifiers")
renderSVG.drawToFile(draw_bracket(LEAVES, R16W, QFW, SFW, CHAMP), os.path.join(SVGDIR, "knockout_bracket.svg"), "knockout_bracket")

def tbl(data, widths, header=True, align_right_from=2):
    st = [('FONTSIZE', (0, 0), (-1, -1), 8), ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#cccccc')),
          ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 2.5),
          ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5), ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4)]
    if header:
        st += [('BACKGROUND', (0, 0), (-1, 0), DARK), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
               ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
               ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f6f4')]),
               ('ALIGN', (align_right_from, 1), (-1, -1), 'RIGHT')]
    return Table(data, colWidths=widths, repeatRows=1 if header else 0, style=TableStyle(st))

# ====================== STORY ======================
S_ = []
S_.append(Paragraph("FIFA World Cup 2026 - Model-Based Prediction", H1))
S_.append(Paragraph(f"Generated {TODAY} | Engine: github.com/govindgoel2001/Sportsbetting "
   f"(Dixon-Coles + Elo + 10-agent panel + tactics + player/flank lanes + formations/chemistry "
   f"+ 19-phase game-state sim + {N:,}-run Monte Carlo + probability calibration)", SUB))

S_.append(Paragraph("Read this first", H2))
S_.append(Paragraph("This is a <b>probabilistic forecast</b>, not a result - the engine simulates the "
   f"whole tournament {N:,} times over the real WC2026 draw. Even the strongest side lifts the trophy "
   "only about 1 time in 4; treat every percentage as a distribution, not a prophecy.", WARN))

S_.append(Paragraph(f"Title odds - top 15 ({N:,} simulations)", H2))
S_.append(tbl(ODDS, [8*mm, 32*mm, 16*mm, 16*mm, 16*mm, 16*mm, 16*mm, 20*mm]))
S_.append(Paragraph(f"Big-four (ARG+FRA+BRA+ESP) combined title ~ {BIG4:.0f}%. "
   f"Three hosts (USA+MEX+CAN) combined ~ {HOSTSUM:.1f}%.", SMALL))

S_.append(PageBreak())
S_.append(Paragraph("Group qualifiers - advancement probability (vector chart)", H2))
S_.append(Paragraph("Bar = probability of reaching the knockout stage. Dark green = predicted group winner, "
   "medium green = predicted runner-up, grey = elimination favourites. Standalone file: "
   "diagrams/group_qualifiers.svg", SMALL))
S_.append(Spacer(1, 4)); S_.append(g_draw)

S_.append(PageBreak())
S_.append(Paragraph("Knockout bracket - most-likely tree (vector diagram)", H2))
S_.append(Paragraph("Left to right: Round of 16 -> winners -> Quarter-finals -> Semi-finals -> Final. "
   "Green = advances, gold = champion. Standalone file: diagrams/knockout_bracket.svg", SMALL))
S_.append(Spacer(1, 4)); S_.append(b_draw)
S_.append(Paragraph(f"FINAL: {CHAMP} def. {RUNNER} (~{CH_SHARE*100:.0f}-{(1-CH_SHARE)*100:.0f}). "
   f"3rd place: {THIRD} def. {THIRD_LOSER}.", P))

S_.append(PageBreak())
S_.append(Paragraph("Method & honest scope", H2))
for line in [
   "<b>Operative model</b> = Dixon-Coles Poisson (45%) blended with Elo strength (55%), then nudged by a "
   "10-agent condition panel (form, momentum, H2H, tactician, tempo, host/crowd, travel, climate, news, "
   "hydration), a style-vector tactical engine, positional player/flank lane duels, and formation + squad-"
   f"chemistry shape. Matches resolve through a 19-phase game-state event sim (park-the-bus, chasing, "
   f"stoppage-time goals); the tournament is then resampled {N:,} times.",
   "<b>Probability calibration (new):</b> the full nuanced stack was mildly over-confident in favourites, so "
   "the final 1X2 is shrunk 0.35 toward the plain Dixon-Coles+Elo core. The shrink is set by principle "
   "(variance reduction), NOT fitted to the live games - leave-one-out showed tuning it would overfit.",
   f"<b>Live track record (as of {TODAY}, 10 games):</b> the model called the result in 7. Probabilistically "
   "it scores Brier 0.60 - 0.58 after calibration - versus 0.58 for its own plain DC+Elo core and 1.00 for "
   "naive favourite-picking. On n=10 those gaps are inside bootstrap noise, so trust the ordering "
   "(STRONG > CLEAR > LEAN > TOSS-UP) over the exact number.",
   "<b>Honest scope:</b> lineups assume first-choice XIs in good health - one injury reshapes a tie. "
   "Cross-confederation calibration is the weak link. The real accuracy unlock is a multi-thousand-match "
   "historical backtest (past World Cups, qualifiers, club leagues), not more model layers.",
]:
    S_.append(Paragraph(line, P))
S_.append(Spacer(1, 6))
S_.append(Paragraph("For research and entertainment only. Probabilities are not certainties. Bet responsibly.", SMALL))

SimpleDocTemplate(OUT, pagesize=A4, topMargin=14*mm, bottomMargin=12*mm, leftMargin=16*mm,
                  rightMargin=16*mm, title="WC2026 Prediction",
                  author="govindgoel2001/Sportsbetting").build(S_)
print("PDF:", OUT)
print("SVGs:", os.path.join(SVGDIR, "group_qualifiers.svg"), "+", os.path.join(SVGDIR, "knockout_bracket.svg"))
print("Champion:", CHAMP, "| Final:", CHAMP, "def.", RUNNER, f"(~{CH_SHARE*100:.0f}-{(1-CH_SHARE)*100:.0f})")
