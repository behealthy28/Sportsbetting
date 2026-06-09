# World Cup 2026 — Full Prediction Package

This folder is a self-contained World Cup 2026 forecast built **on top of this repo's own
prediction engine** (EdgeFinder). If you were handed this repo cold, this README is your full
context: what was done, how, what to trust, and how to reproduce it.

> **Reality check.** The tournament starts 11 June 2026. Nobody can tell you the *actual* result
> of each match — these are **model probabilities**, not prophecy. A 24% favourite still loses
> ~3 times in 4.

---

## TL;DR results (20,000-run Monte Carlo)

- **Predicted champion: Argentina** (23.8%), beating **France** (17.8%) in the final.
- Title contenders: Argentina 23.8 · France 17.8 · Brazil 13.0 · Spain 10.6 · Portugal 7.1 · England 5.4 (%).
- Big-four combined ≈ 65%. The three hosts (USA/Mexico/Canada) combined ≈ 1.2%.
- Full tables, group-by-group qualification odds, and the bracket are in
  **`WC2026_Prediction.pdf`** (5 pages, with vector diagrams) and `diagrams/*.svg`.

---

## What's in here

| File | What it is |
|------|-----------|
| `wc2026_sim.py` | The tournament simulator — uses this repo's Dixon-Coles + Elo models |
| `wc_arb_edge_scanner.py` | Polymarket × Kalshi arbitrage + model-edge scanner |
| `gen_pdf.py` | Builds the PDF report **and** the vector diagrams |
| `WC2026_Prediction.pdf` | The report (title odds, groups, bracket, method, betting layer) |
| `diagrams/group_qualifiers.svg` | Scalable group-advancement chart |
| `diagrams/knockout_bracket.svg` | Scalable R16→Final bracket tree |

---

## How the model works (A→Z)

The operative model is **exactly the blend in `src/sports/football.py`**:
**Dixon-Coles Poisson (0.50) + Elo (0.30)**, renormalised to 0.625 / 0.375.

1. **Elo → expected goals.** supremacy `= (Elo_A − Elo_B) / 130`; with a neutral-venue WC baseline of
   2.6 total goals, `λ_A = 1.3 + sup/2`, `λ_B = 1.3 − sup/2`.
2. **Dixon-Coles scorelines** via `src/models/dixon_coles._score_probability` (ρ = −0.13), integrated to Win/Draw/Loss.
3. **Elo W/D/L** via `src/models/elo.win_probability` + `_estimate_draw_prob`.
4. **Blend** the two, then **Monte Carlo** the whole tournament 20,000×.
5. **Knockouts**: top-2 per group + 8 best third-placed teams, slotted into the **official R32 bracket**
   (third-place allocation solved by exact bipartite matching). Draws → penalties (near coin-flip, small Elo tilt).

Ratings come from `src/data/scrapers/elo_db.py` (`NATIONAL_TEAM_ELO`, tuned for the WC2026 field).
The real group draw and bracket structure were taken from the official 5 Dec 2025 draw + FIFA bracket.

### Honest scope — what did NOT run
- **XGBoost / Random Forest** fell back: the repo only loads them if trained `.pkl` files exist; none ship for
  national teams, so weight redistributes to Dixon-Coles + Elo (this is the repo's own behaviour).
- **News sentiment, agent-debate, H2H, injuries, "chemistry"** were **inactive** — they need live feeds / API keys.
- The **Elo ratings carry the predictive signal.** Don't read more sophistication into it than that.

### MiroFish swarm overlay (qualitative only)
The report includes a 6-persona analyst "swarm" overlay inspired by
[github.com/666ghj/MiroFish](https://github.com/666ghj/MiroFish). The **real MiroFish stack was NOT
deployed** (it needs paid LLM + Zep keys, the `camel-oasis` framework, and long multi-round agent runs).
The overlay is analyst judgment, not new computation.

---

## Reproduce it

```bash
pip install -r requirements.txt          # repo deps
pip install reportlab                     # for the PDF + SVG diagrams

python worldcup2026/wc2026_sim.py         # ~80s for 20,000 sims -> probabilities
python worldcup2026/gen_pdf.py            # builds the PDF + diagrams/*.svg
python worldcup2026/wc_arb_edge_scanner.py  # live Polymarket/Kalshi scan (no keys needed to read)
```
Tune via env: `WC_SIMS=50000 python worldcup2026/wc2026_sim.py`.

---

## Betting layer — read before risking money

- **True arbitrage** = the *same* outcome priced differently across Polymarket vs Kalshi by more than fees.
  Risk-free, needs no model. The scanner found **none** on the WC outright — that market is extremely sharp.
- **Model edge / value** = betting because you believe the model beats the market. **NOT risk-free.** For a
  heavily-traded event, assume most positive "edges" are *model error*. Sizing uses **quarter-Kelly**; negative edge → $0.
- **Jurisdiction:** Kalshi is CFTC-regulated (effectively US-resident only); Polymarket uses USDC/Polygon and
  blocks US persons. Outside those, **verify local legality and tax before funding anything.**
- Realistic alpha is *not* the outright but thin, slow-repricing per-match markets (e.g. right after injury news).

---

*For research and entertainment only. Probabilities are not certainties. Bet responsibly.*
