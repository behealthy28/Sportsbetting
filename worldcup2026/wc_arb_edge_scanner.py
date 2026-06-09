"""
WC2026 Arbitrage + Edge scanner — Polymarket x Kalshi.
Built on the sportsbetting repo's own market math:
  src/market/odds.py   -> remove_vig, prob/odds conversions
  src/market/kelly.py  -> quarter-Kelly stake
  src/market/edge.py   -> edge = model_prob - devigged_market_prob

TWO distinct money-makers (do not confuse them):
  (A) CROSS-VENUE ARBITRAGE  -> risk-free IF the SAME outcome is priced differently
      on Polymarket vs Kalshi by more than total fees. Needs NO model. Rare, capital-
      and-fee constrained, and the window closes fast.
  (B) MODEL EDGE / VALUE      -> you think your model is better-calibrated than the
      market. NOT risk-free. Only profitable long-run IF the model truly beats a very
      sharp market. For a heavily-traded WC outright, assume the market is sharp and
      treat 'edge' with heavy skepticism.

Run from anywhere:  python worldcup2026/wc_arb_edge_scanner.py   (scan only, never auto-places)
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
from src.market.odds import remove_vig, prob_to_decimal
from src.market.kelly import kelly_fraction, expected_value

# ---- Model probabilities P(win the World Cup), from wc2026_sim.py 20k Monte Carlo ----
MODEL_TITLE_PROB = {
    "argentina":0.238,"france":0.178,"brazil":0.130,"spain":0.106,"portugal":0.071,
    "england":0.054,"netherlands":0.039,"croatia":0.024,"germany":0.019,"uruguay":0.010,
    "belgium":0.010,"colombia":0.009,"switzerland":0.008,"japan":0.007,"turkey":0.006,
}
FEE_BUFFER = 0.035   # min YES-price gap across venues needed to clear fees and arb
BANKROLL = 1000.0

def _norm(s): return (s or "").lower().replace("the ","").strip()

def fetch_polymarket_wc():
    out={}
    try:
        r=requests.get("https://gamma-api.polymarket.com/markets",
            params={"limit":500,"closed":"false","order":"volume","ascending":"false"},
            headers={"User-Agent":"Mozilla/5.0"},timeout=20)
        data=r.json() if r.status_code==200 else []
        if isinstance(data,dict): data=data.get("data",data.get("markets",[]))
        import json as J
        for m in data:
            q=_norm(m.get("question",""))
            if "world cup" not in q or "win" not in q: continue
            prices=m.get("outcomePrices")
            if isinstance(prices,str): prices=J.loads(prices)
            if not prices: continue
            for team in MODEL_TITLE_PROB:
                if team in q: out[team]=float(prices[0]); break
    except Exception as e:
        print("  [polymarket fetch failed]",e)
    return out

def fetch_kalshi_wc():
    out={}
    try:
        r=requests.get("https://api.elections.kalshi.com/trade-api/v2/markets",
            params={"limit":1000,"status":"open","series_ticker":"KXWORLDCUP"},timeout=20)
        data=r.json().get("markets",[]) if r.status_code==200 else []
        for m in data:
            title=_norm(m.get("title","")+" "+m.get("subtitle","")+" "+m.get("yes_sub_title",""))
            yes=m.get("yes_ask") or m.get("last_price")
            if yes is None: continue
            for team in MODEL_TITLE_PROB:
                if team in title: out[team]=float(yes)/100.0; break
    except Exception as e:
        print("  [kalshi fetch failed]",e)
    return out

def scan():
    poly=fetch_polymarket_wc(); kal=fetch_kalshi_wc()
    print(f"\nPolymarket teams: {len(poly)} | Kalshi teams: {len(kal)}\n")
    print("="*78)
    print("(A) CROSS-VENUE ARBITRAGE  (buy YES on cheap venue, buy NO on dear venue)")
    print("="*78)
    arb=False
    for team in MODEL_TITLE_PROB:
        if team in poly and team in kal:
            spread=abs(poly[team]-kal[team])
            if spread>FEE_BUFFER:
                arb=True
                cheap="Polymarket" if poly[team]<kal[team] else "Kalshi"
                dear="Kalshi" if cheap=="Polymarket" else "Polymarket"
                print(f"  {team.title():12s} ARB ~{(spread-FEE_BUFFER)*100:.1f}%/$1 net | "
                      f"buy YES@{min(poly[team],kal[team]):.3f} on {cheap}, buy NO on {dear}")
    if not arb: print("  none above fee buffer (normal - these markets are efficient).")
    print("\n"+"="*78)
    print("(B) MODEL EDGE / VALUE  (NOT risk-free - needs model > market)")
    print("="*78)
    print(f"  {'Team':12s} {'Model':>7s} {'Poly':>7s} {'Kalshi':>7s} {'BestMkt':>8s} {'Edge':>7s} {'qKelly':>8s} {'EV':>7s}")
    rows=[]
    for team,mp in MODEL_TITLE_PROB.items():
        venues={k:v for k,v in (("poly",poly.get(team)),("kalshi",kal.get(team))) if v}
        if not venues: continue
        best=min(venues,key=venues.get); mkt=venues[best]
        edge=mp-mkt; dec=prob_to_decimal(mkt)
        rows.append((edge,team,mp,poly.get(team),kal.get(team),mkt,kelly_fraction(mp,dec,0.25),expected_value(mp,dec)))
    for edge,team,mp,pp,kp,mkt,kpct,ev in sorted(rows,reverse=True):
        flag="  <= +EV" if edge>0 else ""
        print(f"  {team.title():12s} {mp*100:6.1f}% {('-' if pp is None else f'{pp*100:.1f}%'):>7s} "
              f"{('-' if kp is None else f'{kp*100:.1f}%'):>7s} {mkt*100:6.1f}% {edge*100:+6.1f}% "
              f"{kpct:6.1f}% ${BANKROLL*kpct/100:6.0f}{flag}")
    print("\nNOTES:")
    print(" - 'Edge' is model vs a SHARP market - most positive edges are model error.")
    print(" - Quarter-Kelly applied. Cap any single bet. Placement is intentionally NOT automated.")

if __name__=="__main__":
    scan()
