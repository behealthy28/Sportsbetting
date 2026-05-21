# EdgeFinder — Sports Betting Predictor

Multi-model sports prediction engine that calculates true win probabilities and finds your edge over Polymarket and Kalshi. Combines Dixon-Coles, surface-adjusted ELO, physics-based fight simulation, ML ensemble (RF + XGBoost + LightGBM), and a 9-agent AI debate layer.

**Zero API keys required** for predictions. All data from public sources.

---

## Sports supported

| Sport | Model | Key data source |
|-------|-------|-----------------|
| ⚽ Football / Soccer | Dixon-Coles (xG) + ELO + ML | Understat (xG), ESPN, ClubElo |
| 🎾 Tennis | Surface ELO + real H2H + ML | Jeff Sackmann ATP/WTA dataset |
| 🥊 UFC / MMA | Striking/grappling stats + ELO + ML | UFCStats.com (official) |
| 🥊 Boxing | Style-matchup sim + ELO + ML | BOXER_PROFILES (52 fighters) |
| 🏏 Cricket | ELO + batting/bowling model + ML | CricSheet ball-by-ball data |
| 🎯 Darts | ELO (PDC rankings seed) | PDC world rankings |
| 🏸 Badminton | ELO (BWF rankings seed) | BWF world rankings |
| 🏓 Table Tennis | ELO (ITTF rankings seed) | ITTF rankings |

---

## Quick start

```bash
git clone https://github.com/govindgoel2001/Sportsbetting
cd Sportsbetting
pip install -r requirements.txt
python main.py train          # train ML models (~2 min)
```

### Web app
```bash
python app.py
# Open http://localhost:8000
```

### Deploy to Vercel
```bash
npm i -g vercel
vercel
```

### CLI
```bash
python main.py "Portugal vs Spain Nations League"
python main.py "Djokovic vs Alcaraz Wimbledon"
python main.py "Khamzat Chimaev vs Sean Strickland UFC"
python main.py "Canelo Alvarez vs David Benavidez"
python main.py "India vs Australia T20 World Cup"
python main.py dashboard       # live Bloomberg-style terminal
python main.py ask "Arsenal vs Liverpool Premier League"
```

---

## Example output

```
🥊  Khamzat Chimaev  vs  Sean Strickland  —  UFC  •  2026-05-21

  Outcome          Probability      Model    Market    Edge
  ──────────────────────────────────────────────────────────
  ★ Chimaev        ████████████░░   56.0%    52.0%    +4.0%  MODERATE
    Strickland     █████████░░░░░   44.0%    48.0%    -4.0%

  Best Bet: Chimaev Win
  Edge: +4.0%  |  Kelly Stake: 2.8% of bankroll
  Confidence: 61%

  Key factors:
  • ELO: Chimaev 1800 vs Strickland 1770
  • Grappling dominance: 6.34 TD/15min vs 1.30
  • Chimaev undefeated (13-0), 77% finish rate
```

---

## Architecture

### Prediction pipeline

```
Raw data (scrapers)
    ↓
ELO ratings (ClubElo live / seeded)
    ↓
Statistical model (Dixon-Coles / striking-grappling / surface form)
    ↓
ML ensemble (Random Forest 25% + XGBoost 37.5% + LightGBM 37.5%)
    ↓
News sentiment adjustment (Google News RSS)
    ↓
9-agent AI debate (optional, requires ANTHROPIC_API_KEY)
    ↓
Market odds (Polymarket / Kalshi)
    ↓
Edge calculation + Kelly stake sizing
```

### ML ensemble

Three calibrated models soft-voted with weights optimised per sport:
- **Random Forest** — handles non-linear feature interactions, resistant to overfitting on small datasets
- **XGBoost** — gradient boosting, strong on tabular sports data
- **LightGBM** — fast leaf-wise boosting, captures subtle statistical edges

Training data: Jeff Sackmann ATP/WTA archives (100k+ matches), physics-based fight simulation for UFC/boxing (30k simulated bouts with real fighter profiles), CricSheet ball-by-ball data.

### 9-agent debate layer

When `ANTHROPIC_API_KEY` is set, nine specialist Claude Haiku agents each analyse the match independently and produce a probability + confidence + reasoning. Their estimates are confidence-weighted and blended with the ML output at 72%/28%.

| Agent | Weight | Signal domain |
|---|---|---|
| StatisticsAnalyst | 22% | Form, H2H, ELO, averages |
| TacticsScout | 18% | Style matchups, physical attributes |
| InjurySpecialist | 15% | Key absences, positional gaps, depth |
| NewsIntelligence | 13% | Motivation, morale, managerial changes |
| Contrarian | 10% | Stress-tests the favourite, upset patterns |
| MarketIntelligence | 10% | Sharp money, line movement, public bias |
| PsychologyAnalyst | 5% | Rivalry, desperation, complacency |
| WeatherAnalyst | 4% | Wind, rain, altitude (outdoor sports) |
| HomeGroundExpert | 3% | Travel fatigue, crowd, time zones |

The debate layer falls back gracefully when the API key is absent — predictions are unaffected.

### ELO coverage

Real ELO ratings seeded for:
- **UFC**: all 43 top fighters across 12 weight classes (Jones 1920, Makhachev 1870, McGregor 1680)
- **Boxing**: 52 fighters across 17 weight divisions (Usyk 1950, Canelo 1930, Inoue 1920)
- **Tennis**: 90+ ATP/WTA players (Djokovic 2280, Sinner 2270, Swiatek 2250, Sabalenka 2240)
- **Cricket**: 14 national teams (India 1920, Australia 1900)
- **Darts**: 40+ PDC players (Humphries 1930, Littler 1920, MVG 1910)
- **Badminton/TT**: 40+ BWF/ITTF players

---

## Edge calculation

```
edge = model_probability − devigged_market_probability

Thresholds:
  STRONG   > 5%   — high confidence, bet
  MODERATE 2–5%   — moderate value
  WEAK     0–2%   — skip
  NEGATIVE < 0%   — market knows more, avoid
```

Kelly stake sizing uses quarter-Kelly for conservative bankroll management.

---

## Data sources (all free, no signup)

| Source | What it provides |
|--------|-----------------|
| [FBRef.com](https://fbref.com) | Football team stats, xG, form |
| [Understat.com](https://understat.com) | Expected goals per match |
| [ESPN unofficial API](https://site.api.espn.com) | Fixtures, standings |
| [Jeff Sackmann's GitHub](https://github.com/JeffSackmann/tennis_atp) | Full ATP/WTA history |
| [UFCStats.com](https://www.ufcstats.com) | Official UFC fighter stats |
| [ClubElo](https://clubelo.com) | Live football club ELO ratings |
| [CricSheet.org](https://cricsheet.org) | Ball-by-ball cricket data |
| [Polymarket CLOB API](https://clob.polymarket.com) | Prediction market odds |
| [Kalshi API](https://trading-api.kalshi.com) | Prediction market odds |
| Google News RSS | News sentiment for injuries/form |

---

## Optional: unlock more features

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Key | Effect |
|-----|--------|
| `ANTHROPIC_API_KEY` | Enables 9-agent debate layer + `ask` LLM verdict command |
| `API_FOOTBALL_KEY` | Deeper injury/lineup data (api-sports.io, free 100 req/day) |
| `ODDS_API_KEY` | Sportsbook lines (the-odds-api.com, free 500 req/month) |
| `POLY_PRIVATE_KEY` + others | Direct Polymarket execution via `place <id>` command |

---

## Project structure

```
src/
  sports/          # per-sport prediction handlers
  models/
    dixon_coles.py # MLE-fitted Poisson model
    elo.py         # surface-adjusted ELO
    ml_ensemble.py # RF + XGBoost + LightGBM
    agent_debate.py# 9-agent Claude debate layer
    calibrator.py  # probability blending + calibration
  data/scrapers/   # FBRef, Understat, UFCStats, Sackmann, etc.
  market/          # edge calculation, Kelly, odds conversion
  training/        # train.py — trains all sport models
data/
  mappings/        # ELO seeds for all sports (players.json, teams.json)
  models/          # trained .pkl files (gitignored, rebuilt via train)
```

---

## Disclaimer

For educational and research purposes only. No accuracy guarantees. Bet responsibly. Past model performance does not predict future results.
