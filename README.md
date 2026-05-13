# EdgeFinder — Sports Betting Predictor

Multi-model sports prediction tool that generates true probabilities and calculates your edge over Polymarket and Kalshi.

**Zero API keys required.** All data from public sources.

---

## Sports supported

| Sport | Model | Key data source |
|-------|-------|-----------------|
| ⚽ Football / Soccer | Dixon-Coles (xG) + ELO | Understat (xG), ESPN, ClubElo |
| 🎾 Tennis | Surface-adjusted ELO + H2H | Jeff Sackmann ATP/WTA dataset |
| 🥊 UFC / MMA | Striking/grappling stats + ELO | UFCStats.com (official) |
| 🥊 Boxing | ELO + style heuristics | Seeded ELO + BoxRec |
| 🏏 Cricket | Dixon-Coles + format stats | CricSheet ball-by-ball data |
| 🎯 Darts | ELO (PDC rankings seed) | PDC world rankings |
| 🏸 Badminton | ELO (BWF rankings seed) | BWF world rankings |
| 🏓 Table Tennis | ELO (ITTF rankings seed) | ITTF rankings |

> No backtesting has been run on this codebase. The model outputs probabilities based on real data and established methods (Dixon-Coles, ELO, quarter-Kelly), but predicted accuracy figures have not been validated. Treat all edge numbers as model estimates, not guaranteed results.

---

## How it works

1. **Data** — scrapes public sources (no signup): FBRef, Understat, ESPN unofficial API, Jeff Sackmann's GitHub CSV archives, UFCStats.com, CricSheet.org, Google News RSS
2. **Models** — Dixon-Coles Poisson (football/cricket) + surface-adjusted ELO (tennis) + striking/grappling stats model (UFC) + Random Forest / XGBoost ensemble
3. **Market comparison** — fetches live odds from Polymarket and Kalshi (public APIs, no auth), removes vig, calculates edge per outcome
4. **Stake sizing** — quarter-Kelly criterion for safe bankroll management

---

## Quick start

```bash
git clone https://github.com/govindgoel2001/sportsbetting
cd sportsbetting
pip install -r requirements.txt
```

### Web app (local)

```bash
python app.py
# Open http://localhost:8000
```

### Deploy to Vercel (share a public URL)

```bash
npm i -g vercel   # one-time install
vercel            # follow prompts — done in ~30 seconds
```

> **Free tier note:** Vercel Hobby has a 10-second function timeout. Most predictions finish in 3–8s but heavy football queries can hit the limit. Upgrade to Vercel Pro (60s timeout) or use Railway/Render if you hit it regularly.

### CLI

```bash
python main.py "Portugal vs Spain Nations League"
python main.py "Djokovic vs Alcaraz Wimbledon tomorrow"
python main.py "Jon Jones vs Stipe Miocic UFC"
python main.py "India vs Australia T20 World Cup"
python main.py show sports
```

---

## Example output

```
⚽  Portugal  vs  Spain  —  Nations League  •  2026-05-12

  Outcome       Probability          Model    Market    Edge
  ─────────────────────────────────────────────────────────
  ★ Portugal    ████████░░░░░░░░░░   41.1%    32.2%    +8.9%  STRONG
    Draw        ██████░░░░░░░░░░░░   28.6%    25.4%    +3.2%
    Spain       ██████░░░░░░░░░░░░   30.3%    42.4%    -12.1%

  Best Bet: Portugal Win
  Edge: +8.9%  [STRONG VALUE]
  Kelly Stake: 5.2% of bankroll
```

---

## Edge calculation

```
edge = model_probability − devigged_market_probability

Thresholds:
  STRONG   > 5%   — high confidence value bet
  MODERATE 2–5%   — moderate value
  WEAK     0–2%   — minimal edge, skip
  NEGATIVE < 0%   — market knows more, avoid
```

---

## Model design rationale

Four reasons the model structure is reasonable, though no accuracy claims are made:

1. **Dixon-Coles xG** — expected goals regress to true team quality faster than actual scorelines; using xG instead of goals is mathematically sounder
2. **Surface-adjusted ELO** (tennis) — a player's hard-court ELO and clay-court ELO differ substantially; flat ELO ignores this
3. **Quarter-Kelly sizing** — full Kelly is theoretically optimal but requires a perfectly calibrated model; quarter-Kelly is more conservative and robust to calibration errors
4. **Live ELO from ClubElo/eloratings.net** — ratings are fetched live rather than static seeds, so recent form is reflected

---

## No API keys needed

All data sources are public:

| Source | What it provides |
|--------|-----------------|
| [FBRef.com](https://fbref.com) | Football team stats, xG, form |
| [Understat.com](https://understat.com) | Expected goals per match |
| [ESPN unofficial API](https://site.api.espn.com) | Fixtures, standings, scores |
| [Jeff Sackmann's GitHub](https://github.com/JeffSackmann/tennis_atp) | Complete ATP/WTA match history |
| [UFCStats.com](http://ufcstats.com) | Official UFC fighter statistics |
| [CricSheet.org](https://cricsheet.org) | Ball-by-ball cricket data |
| [Polymarket CLOB API](https://clob.polymarket.com) | Prediction market odds |
| [Kalshi API](https://trading-api.kalshi.com) | Prediction market odds |
| Google News RSS | Team/player news for sentiment |

---

## Optional: API keys for higher rate limits

Create a `.env` file (copy from `.env.example`):

```env
# api-sports.io — free 100 req/day, unlocks deeper injury data
API_FOOTBALL_KEY=your_key_here

# the-odds-api.com — free 500 req/month, unlocks sportsbook lines
ODDS_API_KEY=your_key_here
```

The tool works fully without these.

---

## Disclaimer

For educational and research purposes. Past prediction accuracy does not guarantee future results. Bet responsibly.
