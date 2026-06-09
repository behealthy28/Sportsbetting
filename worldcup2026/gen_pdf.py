# -*- coding: utf-8 -*-
"""
Build WC2026_Prediction.pdf with reportlab, including VECTOR diagrams:
  - Group qualifiers chart (advancement %) - drawn as scalable vector graphics
  - Knockout bracket tree (R16 -> Final) - drawn as scalable vector graphics
Also exports each diagram as a standalone .svg into ./diagrams/.
No external binaries required.  Run:  python gen_pdf.py
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak)
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Group, Polygon
from reportlab.graphics import renderSVG

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "WC2026_Prediction.pdf")
SVGDIR = os.path.join(HERE, "diagrams"); os.makedirs(SVGDIR, exist_ok=True)

DARK = colors.HexColor('#0b3d2e'); MED = colors.HexColor('#2e8b57')
GOLD = colors.HexColor('#DAA520'); GREY = colors.HexColor('#cfd8d3')
LINE = colors.HexColor('#9aa7a0'); INK = colors.HexColor('#13241d')

ss = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=ss['Title'], fontSize=20, spaceAfter=4, textColor=DARK)
SUB = ParagraphStyle('SUB', parent=ss['Normal'], fontSize=9, textColor=colors.grey, spaceAfter=10)
H2 = ParagraphStyle('H2', parent=ss['Heading2'], fontSize=13, spaceBefore=12, spaceAfter=4, textColor=DARK)
P  = ParagraphStyle('P', parent=ss['Normal'], fontSize=9.5, leading=13, spaceAfter=5)
WARN = ParagraphStyle('WARN', parent=P, backColor=colors.HexColor('#fff4e5'), borderColor=colors.HexColor('#e0a000'), borderWidth=0.6, borderPadding=6, spaceAfter=8)
SMALL = ParagraphStyle('SMALL', parent=P, fontSize=8, textColor=colors.grey)

# ---------------- data ----------------
GROUPS = {
 "A":[("S.Korea",27.3,71.7),("Mexico",26.9,70.2),("Czechia",25.3,68.4),("S.Africa",20.4,62.9)],
 "B":[("Switzerland",38.0,80.0),("Bosnia",23.0,67.4),("Canada",21.5,65.1),("Qatar",17.5,59.7)],
 "C":[("Brazil",74.6,97.1),("Morocco",11.6,64.6),("Scotland",9.0,58.7),("Haiti",4.8,39.7)],
 "D":[("Turkey",29.9,73.5),("USA",26.1,69.8),("Paraguay",22.4,65.0),("Australia",21.6,64.2)],
 "E":[("Germany",46.4,87.0),("Ivory Coast",22.7,68.8),("Ecuador",19.9,66.4),("Curacao",11.0,48.1)],
 "F":[("Netherlands",49.5,88.4),("Japan",21.2,67.4),("Sweden",16.9,61.5),("Tunisia",12.4,52.4)],
 "G":[("Belgium",41.0,83.1),("Egypt",22.8,68.2),("Iran",18.8,61.4),("New Zealand",17.4,58.9)],
 "H":[("Spain",66.9,95.6),("Uruguay",18.7,73.5),("Saudi Arabia",7.3,47.6),("Cape Verde",7.1,46.4)],
 "I":[("France",74.8,97.4),("Senegal",10.2,60.3),("Norway",9.5,59.0),("Iraq",5.5,43.6)],
 "J":[("Argentina",82.0,98.6),("Austria",8.0,60.3),("Algeria",6.4,54.0),("Jordan",3.7,42.0)],
 "K":[("Portugal",62.9,94.1),("Colombia",18.8,70.9),("Uzbekistan",9.4,50.7),("DR Congo",8.8,49.6)],
 "L":[("England",50.9,90.9),("Croatia",32.2,82.9),("Ghana",9.5,49.7),("Panama",7.4,43.3)],
}
# knockout bracket leaves (R16), ordered for a clean top->bottom tree
LEAVES = ["France","Germany","Netherlands","Mexico","Spain","Croatia","Belgium","Turkey",
          "Brazil","Senegal","England","Sweden","Argentina","USA","Portugal","Switzerland"]
R16W = ["France","Netherlands","Spain","Belgium","Brazil","England","Argentina","Portugal"]
QFW  = ["France","Spain","Brazil","Argentina"]
SFW  = ["France","Argentina"]
CHAMP= "Argentina"

# ---------------- vector diagram: group qualifiers ----------------
def draw_groups():
    cols, rows = 3, 4
    cw, ch = 158, 150            # card size (pt)
    gx, gy = 8, 12
    W = cols*cw + (cols-1)*gx
    H = rows*ch + (rows-1)*gy + 6
    d = Drawing(W, H)
    keys = list(GROUPS.keys())
    for idx, g in enumerate(keys):
        c = idx % cols; r = idx // cols
        x0 = c*(cw+gx); y0 = H - 6 - (r+1)*ch - r*gy
        d.add(Rect(x0, y0, cw, ch, strokeColor=GREY, fillColor=colors.white, strokeWidth=0.8))
        d.add(Rect(x0, y0+ch-18, cw, 18, strokeColor=None, fillColor=DARK))
        d.add(String(x0+6, y0+ch-13, "GROUP "+g, fontName='Helvetica-Bold', fontSize=9, fillColor=colors.white))
        teams = GROUPS[g]
        bar_x = x0+8; bar_w = cw-70; row_h = 28
        for i,(name,win,adv) in enumerate(teams):
            ry = y0+ch-18-(i+1)*row_h+6
            qualifies = i < 2
            col = DARK if i==0 else (MED if i==1 else GREY)
            d.add(String(bar_x, ry+14, name, fontName='Helvetica-Bold' if qualifies else 'Helvetica',
                         fontSize=7.5, fillColor=INK))
            d.add(Rect(bar_x, ry+2, bar_w, 7, strokeColor=None, fillColor=colors.HexColor('#eef2f0')))
            d.add(Rect(bar_x, ry+2, bar_w*adv/100.0, 7, strokeColor=None, fillColor=col))
            d.add(String(bar_x+bar_w+4, ry+2, f"{adv:.0f}%", fontName='Helvetica', fontSize=7.5, fillColor=INK))
    return d

# ---------------- vector diagram: knockout bracket ----------------
def draw_bracket():
    W, H = 500, 690
    d = Drawing(W, H)
    bw, bh = 86, 18
    colx = [4, 128, 252, 360, 446]   # x of each column
    n = len(LEAVES)
    top, bot = H-10, 20
    slot = (top-bot)/n
    def box(x, y, label, fill, txtcol=colors.white, bold=True, fs=7.5):
        d.add(Rect(x, y-bh/2, bw, bh, strokeColor=LINE, fillColor=fill, strokeWidth=0.7, rx=2, ry=2))
        d.add(String(x+4, y-3, label, fontName='Helvetica-Bold' if bold else 'Helvetica', fontSize=fs, fillColor=txtcol))
    def connect(x1, y1a, y1b, x2, ymid):
        midx = (x1+bw+x2)/2
        d.add(Line(x1+bw, y1a, midx, y1a, strokeColor=LINE, strokeWidth=0.6))
        d.add(Line(x1+bw, y1b, midx, y1b, strokeColor=LINE, strokeWidth=0.6))
        d.add(Line(midx, y1a, midx, y1b, strokeColor=LINE, strokeWidth=0.6))
        d.add(Line(midx, ymid, x2, ymid, strokeColor=LINE, strokeWidth=0.6))
    # column ys
    leaf_y = [top - slot*(i+0.5) for i in range(n)]
    # col0 leaves
    for i,name in enumerate(LEAVES):
        win = name in R16W
        box(colx[0], leaf_y[i], name, MED if win else colors.HexColor('#7f8c8a'), bold=win)
    # col1 R16 winners (pair midpoints)
    r16_y=[]
    for j in range(8):
        ya, yb = leaf_y[2*j], leaf_y[2*j+1]; ym=(ya+yb)/2; r16_y.append(ym)
        connect(colx[0], ya, yb, colx[1], ym)
        win = R16W[j] in QFW
        box(colx[1], ym, R16W[j], MED if win else colors.HexColor('#7f8c8a'), bold=win)
    # col2 QF winners
    qf_y=[]
    for j in range(4):
        ya,yb=r16_y[2*j],r16_y[2*j+1]; ym=(ya+yb)/2; qf_y.append(ym)
        connect(colx[1], ya, yb, colx[2], ym)
        win = QFW[j] in SFW
        box(colx[2], ym, QFW[j], MED if win else colors.HexColor('#7f8c8a'), bold=win)
    # col3 SF winners
    sf_y=[]
    for j in range(2):
        ya,yb=qf_y[2*j],qf_y[2*j+1]; ym=(ya+yb)/2; sf_y.append(ym)
        connect(colx[2], ya, yb, colx[3], ym)
        win = SFW[j]==CHAMP
        box(colx[3], ym, SFW[j], GOLD if win else MED, txtcol=INK if win else colors.white, bold=True)
    # champion
    ym=(sf_y[0]+sf_y[1])/2
    connect(colx[3], sf_y[0], sf_y[1], colx[4], ym)
    d.add(Rect(colx[4], ym-13, bw+8, 26, strokeColor=GOLD, fillColor=colors.HexColor('#fff7e0'), strokeWidth=1.4, rx=3, ry=3))
    d.add(String(colx[4]+5, ym+3, "CHAMPION", fontName='Helvetica-Bold', fontSize=6.5, fillColor=GOLD))
    d.add(String(colx[4]+5, ym-8, CHAMP, fontName='Helvetica-Bold', fontSize=10, fillColor=INK))
    # column headers
    for x,lab in zip(colx,["Round of 16","R16 winners","Quarter-finals","Semi-finals","Final"]):
        d.add(String(x, H-2, lab, fontName='Helvetica-Bold', fontSize=7, fillColor=DARK))
    return d

# export standalone SVGs
def export_svg(drawing, name):
    renderSVG.drawToFile(drawing, os.path.join(SVGDIR, name), name)

g_draw = draw_groups(); b_draw = draw_bracket()
export_svg(draw_groups(), "group_qualifiers.svg")
export_svg(draw_bracket(), "knockout_bracket.svg")

def tbl(data, widths, header=True, align_right_from=1):
    st = [('FONTSIZE',(0,0),(-1,-1),8),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#cccccc')),
          ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),2.5),
          ('BOTTOMPADDING',(0,0),(-1,-1),2.5),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)]
    if header:
        st += [('BACKGROUND',(0,0),(-1,0),DARK),('TEXTCOLOR',(0,0),(-1,0),colors.white),
               ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
               ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f6f4')]),
               ('ALIGN',(align_right_from,1),(-1,-1),'RIGHT')]
    return Table(data, colWidths=widths, repeatRows=1 if header else 0, style=TableStyle(st))

S=[]
S.append(Paragraph("FIFA World Cup 2026 - Model-Based Prediction", H1))
S.append(Paragraph("Generated 2026-06-09 | Engine: github.com/govindgoel2001/sportsbetting "
   "(Dixon-Coles + Elo + 20,000-run Monte Carlo) | Swarm overlay: github.com/666ghj/MiroFish "
   "(method only) | Tournament starts 11 June 2026", SUB))
S.append(Paragraph("Read this first", H2))
S.append(Paragraph("The World Cup has NOT been played yet. This is a <b>probabilistic forecast</b> - "
   "the engine simulated the tournament 20,000 times over the real WC2026 draw. A 24% favourite still "
   "loses about 3 times in 4.", WARN))

S.append(Paragraph("Title odds - top 15 (20,000 simulations)", H2))
odds=[["#","Team","Win","Final","Semi","QF","R16","Advance"],
 ["1","Argentina","23.8%","34.8%","49.8%","64.3%","76.4%","98.6%"],
 ["2","France","17.8%","29.5%","44.0%","59.2%","78.3%","97.4%"],
 ["3","Brazil","13.0%","21.5%","36.9%","53.5%","71.5%","97.1%"],
 ["4","Spain","10.6%","19.2%","31.9%","43.9%","63.1%","95.6%"],
 ["5","Portugal","7.1%","13.7%","24.4%","44.4%","64.2%","94.1%"],
 ["6","England","5.4%","11.0%","21.9%","39.0%","60.4%","90.9%"],
 ["7","Netherlands","3.9%","9.0%","17.6%","33.2%","50.8%","88.4%"],
 ["8","Croatia","2.4%","6.2%","13.6%","26.3%","47.6%","82.9%"],
 ["9","Germany","1.9%","5.0%","11.6%","22.6%","52.5%","87.0%"],
 ["10","Uruguay","1.0%","3.0%","7.5%","15.4%","29.2%","73.5%"],
 ["11","Belgium","1.0%","3.2%","8.2%","20.9%","45.8%","83.1%"],
 ["12","Colombia","0.9%","2.6%","6.8%","15.6%","33.1%","70.9%"],
 ["13","Switzerland","0.8%","2.6%","7.0%","20.2%","45.2%","80.0%"],
 ["14","Japan","0.7%","2.2%","6.2%","14.8%","29.1%","67.4%"],
 ["15","Turkey","0.6%","1.9%","5.4%","14.1%","36.5%","73.5%"]]
S.append(tbl(odds,[8*mm,32*mm,16*mm,16*mm,16*mm,16*mm,16*mm,20*mm]))
S.append(Paragraph("Big-four (ARG+FRA+BRA+ESP) combined title ~ 65%. Three hosts combined ~ 1.2%.", SMALL))

S.append(PageBreak())
S.append(Paragraph("Group qualifiers - advancement probability (vector chart)", H2))
S.append(Paragraph("Bar = probability of reaching the knockout stage. Dark green = predicted group winner, "
   "medium green = predicted runner-up, grey = elimination favourites. Standalone file: diagrams/group_qualifiers.svg", SMALL))
S.append(Spacer(1,4)); S.append(g_draw)

S.append(PageBreak())
S.append(Paragraph("Knockout bracket - most-likely tree (vector diagram)", H2))
S.append(Paragraph("Left to right: Round of 16 -> winners -> Quarter-finals -> Semi-finals -> Final. Green = "
   "advances, gold = champion. Standalone file: diagrams/knockout_bracket.svg", SMALL))
S.append(Spacer(1,4)); S.append(b_draw)
S.append(Paragraph("FINAL: Argentina def. France (~52-48). 3rd place: Brazil def. Spain.", P))

S.append(PageBreak())
S.append(Paragraph("Method & honest scope", H2))
for line in [
 "<b>Operative model</b> = Dixon-Coles Poisson (0.50) + Elo (0.30), renormalised to 0.625/0.375 "
 "(exactly src/sports/football.py). Elo->goals via supremacy=(EloA-EloB)/130 on a 2.6-goal WC baseline; "
 "scorelines from the repo's Dixon-Coles grid (rho=-0.13); knockouts to penalties as a near coin-flip.",
 "<b>What did NOT run:</b> XGBoost/Random-Forest fell back (no trained .pkl for nations); news sentiment, "
 "agent-debate, H2H, injuries and chemistry were inactive (need live feeds + API keys). The Elo ratings "
 "carry the signal.",
 "<b>MiroFish overlay</b> is qualitative only - the real stack (paid Qwen + Zep keys, camel-oasis, long "
 "LLM-agent runs) was not deployed; a 6-persona analyst panel adjusted the baseline by judgment, not new computation.",
 "<b>Betting layer:</b> true arbitrage (same outcome priced differently across Polymarket/Kalshi by more than "
 "fees) found NONE on the WC outright. Model 'edge' is NOT risk-free; for a sharp market assume most positive "
 "edges are model error. Sizing = quarter-Kelly. Jurisdiction (New Delhi): verify legality before funding."]:
    S.append(Paragraph(line, P))
S.append(Spacer(1,6))
S.append(Paragraph("For research and entertainment only. Probabilities are not certainties. Bet responsibly.", SMALL))

SimpleDocTemplate(OUT, pagesize=A4, topMargin=14*mm, bottomMargin=12*mm, leftMargin=16*mm,
                  rightMargin=16*mm, title="WC2026 Prediction").build(S)
print("PDF:", OUT)
print("SVGs:", os.path.join(SVGDIR,"group_qualifiers.svg"), "+", os.path.join(SVGDIR,"knockout_bracket.svg"))
