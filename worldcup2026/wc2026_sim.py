"""
World Cup 2026 full-tournament simulator.
Uses the sportsbetting repo's OWN models:
  - src/models/dixon_coles.py  (_score_probability, rho=-0.13)  -> scorelines
  - src/models/elo.py          (win_probability, _estimate_draw_prob)
  - src/data/scrapers/elo_db.py NATIONAL_TEAM_ELO (WC2026-tuned ratings)
Ensemble blend = repo's football.py weights: Dixon-Coles 0.50 / ELO 0.30  (ML absent for nations)

Run from anywhere:  python worldcup2026/wc2026_sim.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math, random, json
from collections import defaultdict
from src.models.dixon_coles import _score_probability
from src.models.elo import win_probability, _estimate_draw_prob
from src.data.scrapers.elo_db import NATIONAL_TEAM_ELO as ELO

random.seed(2026)
MU = 2.6            # WC neutral-venue avg total goals
ELO_PER_GOAL = 130 # supremacy mapping
RHO = -0.13
MAXG = 10
DC_W, ELO_W = 0.50, 0.30   # repo football.py blend; normalized below
WSUM = DC_W + ELO_W

GROUPS = {
 "A": ["mexico","south africa","south korea","czech republic"],
 "B": ["canada","bosnia","qatar","switzerland"],
 "C": ["brazil","morocco","haiti","scotland"],
 "D": ["usa","paraguay","australia","turkey"],
 "E": ["germany","curacao","ivory coast","ecuador"],
 "F": ["netherlands","japan","sweden","tunisia"],
 "G": ["belgium","egypt","iran","new zealand"],
 "H": ["spain","cape verde","saudi arabia","uruguay"],
 "I": ["france","senegal","iraq","norway"],
 "J": ["argentina","algeria","austria","jordan"],
 "K": ["portugal","dr congo","uzbekistan","colombia"],
 "L": ["england","croatia","ghana","panama"],
}
DISPLAY = {t:t.title() for g in GROUPS.values() for t in g}
DISPLAY.update({"usa":"USA","uae":"UAE","dr congo":"DR Congo"})

def elo(t):
    e = ELO.get(t)
    if e is None: raise SystemExit(f"missing elo: {t}")
    return e

def lambdas(a,b):
    sup = (elo(a)-elo(b))/ELO_PER_GOAL
    la = max(0.2, min(5.0, MU/2 + sup/2))
    lb = max(0.2, min(5.0, MU/2 - sup/2))
    return la, lb

def match_model(a,b):
    """Return blended (W,D,L) for a vs b at neutral venue + reweighted scoreline matrix."""
    la,lb = lambdas(a,b)
    M = [[_score_probability(h,k,la,lb,RHO) for k in range(MAXG+1)] for h in range(MAXG+1)]
    s = sum(sum(r) for r in M); M=[[c/s for c in r] for r in M]
    dcW=sum(M[h][k] for h in range(MAXG+1) for k in range(MAXG+1) if h>k)
    dcD=sum(M[h][h] for h in range(MAXG+1))
    dcL=1-dcW-dcD
    pa=win_probability(elo(a),elo(b)); pb=1-pa
    d=_estimate_draw_prob(elo(a),elo(b))
    eW,eD,eL = pa*(1-d), d, pb*(1-d)
    W=(DC_W*dcW+ELO_W*eW)/WSUM
    D=(DC_W*dcD+ELO_W*eD)/WSUM
    L=(DC_W*dcL+ELO_W*eL)/WSUM
    return W,D,L,M,(dcW,dcD,dcL)

CACHE={}
def model(a,b):
    if (a,b) not in CACHE: CACHE[(a,b)]=match_model(a,b)
    return CACHE[(a,b)]

def sample_score(a,b):
    W,D,L,M,(dcW,dcD,dcL)=model(a,b)
    fw=W/max(dcW,1e-9); fd=D/max(dcD,1e-9); fl=L/max(dcL,1e-9)
    cells=[]; probs=[]
    for h in range(MAXG+1):
        for k in range(MAXG+1):
            p=M[h][k]*(fw if h>k else fd if h==k else fl)
            cells.append((h,k)); probs.append(p)
    tot=sum(probs); r=random.random()*tot; c=0
    for (h,k),p in zip(cells,probs):
        c+=p
        if r<=c: return h,k
    return cells[-1]

def knockout_winner(a,b):
    W,D,L,_,_=model(a,b)
    r=random.random()
    if r<W: return a
    if r<W+L: return b
    pa=0.5+max(-0.15,min(0.15,(elo(a)-elo(b))/4000))
    return a if random.random()<pa else b

R32 = [
 ("M73",("RU","A"),("RU","B")),
 ("M74",("W","E"),("3",set("ABCDF"))),
 ("M75",("W","F"),("RU","C")),
 ("M76",("W","C"),("RU","F")),
 ("M77",("W","I"),("3",set("CDFGH"))),
 ("M78",("RU","E"),("RU","I")),
 ("M79",("W","A"),("3",set("CEFHI"))),
 ("M80",("W","L"),("3",set("EHIJK"))),
 ("M81",("W","D"),("3",set("BEFIJ"))),
 ("M82",("W","G"),("3",set("AEHIJ"))),
 ("M83",("RU","K"),("RU","L")),
 ("M84",("W","H"),("RU","J")),
 ("M85",("W","B"),("3",set("EFGIJ"))),
 ("M86",("W","J"),("RU","H")),
 ("M87",("W","K"),("3",set("DEIJL"))),
 ("M88",("RU","D"),("RU","G")),
]
R16=[("M89","M74","M77"),("M90","M73","M75"),("M91","M76","M78"),("M92","M79","M80"),
     ("M93","M83","M84"),("M94","M81","M82"),("M95","M86","M88"),("M96","M85","M87")]
QF=[("M97","M89","M90"),("M98","M93","M94"),("M99","M91","M92"),("M100","M95","M96")]
SF=[("M101","M97","M98"),("M102","M99","M100")]

def assign_thirds(third_teams_by_group):
    """Perfect bipartite matching: 8 third-place groups -> 8 '3' slots (allowed-group sets)."""
    slots=[]
    for name,a,b in R32:
        for spec in (a,b):
            if spec[0]=="3": slots.append((name,spec[1]))
    groups=list(third_teams_by_group)
    order=sorted(range(len(slots)), key=lambda i: sum(1 for g in groups if g in slots[i][1]))
    res={}
    def bt(k, used):
        if k==len(order): return True
        name,allowed=slots[order[k]]
        for g in groups:
            if g not in used and g in allowed:
                res[name]=third_teams_by_group[g]; used.add(g)
                if bt(k+1,used): return True
                used.discard(g); del res[name]
        return False
    bt(0,set())
    return res

def group_table(teams, scores):
    st={t:{"pts":0,"gf":0,"ga":0} for t in teams}
    for (a,b),(ga,gb) in scores.items():
        st[a]["gf"]+=ga; st[a]["ga"]+=gb; st[b]["gf"]+=gb; st[b]["ga"]+=ga
        if ga>gb: st[a]["pts"]+=3
        elif gb>ga: st[b]["pts"]+=3
        else: st[a]["pts"]+=1; st[b]["pts"]+=1
    def key(t): return (st[t]["pts"], st[t]["gf"]-st[t]["ga"], st[t]["gf"], random.random())
    order=sorted(teams,key=key,reverse=True)
    return order, st

def simulate_once():
    winners={}; runners={}; thirds_pool=[]
    for g,teams in GROUPS.items():
        scores={}
        for i in range(len(teams)):
            for j in range(i+1,len(teams)):
                a,b=teams[i],teams[j]
                scores[(a,b)]=sample_score(a,b)
        order,st=group_table(teams,scores)
        winners[g]=order[0]; runners[g]=order[1]
        t=order[2]; thirds_pool.append((st[t]["pts"],st[t]["gf"]-st[t]["ga"],st[t]["gf"],random.random(),g,t))
    thirds_pool.sort(reverse=True)
    best8=thirds_pool[:8]
    third_by_group={g:team for *_,g,team in best8}
    third_slot=assign_thirds(third_by_group)
    def resolve(spec):
        kind,arg=spec
        if kind=="W": return winners[arg]
        if kind=="RU": return runners[arg]
        return None
    res={}
    for name,a,b in R32:
        ta = third_slot[name] if a[0]=="3" else resolve(a)
        tb = third_slot[name] if b[0]=="3" else resolve(b)
        res[name]=knockout_winner(ta,tb)
    for rnd in (R16,QF,SF):
        for name,x,y in rnd:
            res[name]=knockout_winner(res[x],res[y])
    champ=knockout_winner(res["M101"],res["M102"])
    finalists={res["M101"],res["M102"]}
    semis={res["M97"],res["M98"],res["M99"],res["M100"]}
    qfs=set(res[m] for m in ["M89","M90","M91","M92","M93","M94","M95","M96"])
    r16=set(res[m] for m in [r[0] for r in R32])
    adv={winners[g] for g in GROUPS}|{runners[g] for g in GROUPS}|set(third_by_group.values())
    return champ,finalists,semis,qfs,r16,adv,winners,runners

if __name__=="__main__":
    N=int(os.environ.get("WC_SIMS","20000"))
    stat=defaultdict(lambda: defaultdict(int))
    for _ in range(N):
        champ,fin,semi,qf,r16,adv,W,R=simulate_once()
        for t in adv: stat[t]["adv"]+=1
        for t in r16: stat[t]["r16"]+=1
        for t in qf: stat[t]["qf"]+=1
        for t in semi: stat[t]["sf"]+=1
        for t in fin: stat[t]["final"]+=1
        stat[champ]["title"]+=1
        for g in GROUPS: stat[W[g]]["gw"]+=1
    rows=[]
    for t in [x for g in GROUPS.values() for x in g]:
        s=stat[t]
        rows.append((t, s["adv"]/N, s["r16"]/N, s["qf"]/N, s["sf"]/N, s["final"]/N, s["title"]/N))
    rows.sort(key=lambda r:-r[6])
    print("DEEPRUN", json.dumps({"n":N,"rows":[[DISPLAY[r[0]]]+[round(x*100,1) for x in r[1:]] for r in rows]}))
