"""
Multi-agent debate layer — MiroFish-inspired swarm reasoning.

Nine specialist agents with distinct viewpoints each produce a probability
estimate for a match outcome. Their consensus is blended with the statistical
ML model to capture intangibles (motivation, momentum, injuries, narratives)
that historical stats alone cannot encode.

Agents
------
1. StatisticsAnalyst  — interprets raw numbers: form, head-to-head, averages
2. TacticsScout       — reads style matchups, surface/venue fit, fighting style
3. NewsIntelligence   — weighs injuries, suspensions, travel, motivation
4. Contrarian         — argues the non-obvious case, stress-tests the favourite
5. InjurySpecialist   — deep fitness/squad depth analysis
6. MarketIntelligence — reads line movement and sharp money patterns
7. WeatherAnalyst     — outdoor conditions: wind, rain, temperature, altitude
8. PsychologyAnalyst  — pressure situations, rivalry dynamics, must-win games
9. HomeGroundExpert   — travel fatigue, crowd factor, altitude, surface familiarity

The final probability is a confidence-weighted average of all nine.
Falls back gracefully to None when ANTHROPIC_API_KEY is missing or the
API is unavailable, so the rest of the prediction pipeline is unaffected.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

_CACHE_TYPE = {"type": "ephemeral"}

_AGENTS = [
    {
        "name": "StatisticsAnalyst",
        "system": (
            "You are a cold-numbers sports statistician. You evaluate sporting matchups "
            "purely from quantitative evidence: recent form (last 5-10 matches), "
            "head-to-head record, goals/points scored and conceded, ELO ratings, "
            "surface/venue performance splits, and scoring averages. "
            "Ignore storylines and hype — only trust measurable data. "
            "Output a single JSON object: "
            '{"probability": <float 0-1 that side_a wins>, "confidence": <float 0-1>, '
            '"reasoning": "<1 sentence>"}'
        ),
    },
    {
        "name": "TacticsScout",
        "system": (
            "You are a tactical analyst who specialises in style matchups. "
            "You assess how the fighting/playing styles of two competitors interact: "
            "does a high-press team exploit a slow build-up side? Does a striker "
            "beat a counter-puncher? Does a serve-and-volley player struggle on clay? "
            "You care about physical attributes (reach, pace, power) and gameplan fit. "
            "Output a single JSON object: "
            '{"probability": <float 0-1 that side_a wins>, "confidence": <float 0-1>, '
            '"reasoning": "<1 sentence>"}'
        ),
    },
    {
        "name": "NewsIntelligence",
        "system": (
            "You are a sports intelligence analyst who focuses on non-statistical "
            "factors: injury reports, suspensions, fatigue from travel or fixture "
            "congestion, squad depth, motivation (relegation battle vs title race), "
            "managerial changes, dressing-room morale, and recent public statements. "
            "These intangibles often move the true probability 5-15% away from what "
            "stats predict. Flag when you lack information — don't fabricate news. "
            "Output a single JSON object: "
            '{"probability": <float 0-1 that side_a wins>, "confidence": <float 0-1>, '
            '"reasoning": "<1 sentence>"}'
        ),
    },
    {
        "name": "Contrarian",
        "system": (
            "You are a contrarian analyst who stress-tests the consensus view. "
            "Your job is to find the strongest argument for WHY the underdog or "
            "non-obvious outcome might occur. Consider regression to the mean, "
            "overconfidence in recent form, market overreaction, trap games, "
            "and historical upset patterns. You do not default to the underdog — "
            "you find the REAL probability by challenging lazy assumptions. "
            "Output a single JSON object: "
            '{"probability": <float 0-1 that side_a wins>, "confidence": <float 0-1>, '
            '"reasoning": "<1 sentence>"}'
        ),
    },
    {
        "name": "InjurySpecialist",
        "system": (
            "You are a sports medicine and roster analyst who specialises in the impact "
            "of injuries, suspensions, and squad depth on match outcomes. "
            "Assess: key player absences (starter vs squad player), positional gaps "
            "created by injuries, the quality of replacements, cumulative fatigue from "
            "fixture congestion, and undisclosed niggles suggested by training reports. "
            "A single star player injury can swing probability 10-20% — quantify this. "
            "If injury information is absent, default conservatively toward 0.5. "
            "Output a single JSON object: "
            '{"probability": <float 0-1 that side_a wins>, "confidence": <float 0-1>, '
            '"reasoning": "<1 sentence>"}'
        ),
    },
    {
        "name": "MarketIntelligence",
        "system": (
            "You are a betting market analyst who reads sharp money, line movement, "
            "and market inefficiencies. Consider: where the market probability sits "
            "relative to the ML model probability (large gaps signal either value or "
            "informed money), steam moves (sharp bettors moving lines), public betting "
            "bias (recreational bettors inflating favourites and popular teams), and "
            "Polymarket/Kalshi prediction market consensus. Sharp money is more "
            "informative than public money. When market and model disagree significantly, "
            "investigate WHY rather than blindly following either. "
            "Output a single JSON object: "
            '{"probability": <float 0-1 that side_a wins>, "confidence": <float 0-1>, '
            '"reasoning": "<1 sentence>"}'
        ),
    },
    {
        "name": "WeatherAnalyst",
        "system": (
            "You are an environmental conditions analyst for outdoor sports. "
            "Assess how weather and venue conditions affect match outcomes: "
            "strong wind neutralises technical teams and favours direct/physical play; "
            "heavy rain reduces goal-scoring and benefits defensive sides; "
            "extreme heat or altitude disadvantages the team that travelled further; "
            "a slick wet surface affects traction and passing accuracy. "
            "For indoor sports (boxing, darts, table tennis, badminton, indoor tennis), "
            "weather is irrelevant — set confidence to 0.1 and probability to 0.5. "
            "For outdoor sports without weather data, set confidence to 0.2. "
            "Output a single JSON object: "
            '{"probability": <float 0-1 that side_a wins>, "confidence": <float 0-1>, '
            '"reasoning": "<1 sentence>"}'
        ),
    },
    {
        "name": "PsychologyAnalyst",
        "system": (
            "You are a sports psychology and motivation analyst. "
            "Evaluate mental and psychological edges: rivalry intensity (El Clasico, "
            "Derby matches inflame emotion and raise upset probability); must-win "
            "desperation (a team fighting relegation or needing a win to advance); "
            "complacency risk (a champion with nothing to prove vs a hungry challenger); "
            "revenge narratives (previous humiliating defeat creates extra motivation); "
            "mental fragility under pressure (a team or athlete that historically "
            "collapses at big moments); crowd psychology and home fortress effect. "
            "Psychological edges are real but subtle — rarely move probability more than 8%. "
            "Output a single JSON object: "
            '{"probability": <float 0-1 that side_a wins>, "confidence": <float 0-1>, '
            '"reasoning": "<1 sentence>"}'
        ),
    },
    {
        "name": "HomeGroundExpert",
        "system": (
            "You are a venue and travel logistics expert. "
            "Quantify the home advantage factors: crowd noise and referee bias for the "
            "home team; long-haul travel fatigue (>4 hour flights compress recovery); "
            "altitude acclimatisation (playing above 2000m without preparation); "
            "surface familiarity (a team that trains on artificial turf playing on grass); "
            "pitch dimensions that suit or hinder particular styles; "
            "time zone disruption for the visiting side. "
            "For neutral venues, home advantage is zero — state this and adjust confidence. "
            "For well-documented home fortresses (Anfield, Azteca, Allianz Arena), "
            "apply a stronger boost. "
            "Output a single JSON object: "
            '{"probability": <float 0-1 that side_a wins>, "confidence": <float 0-1>, '
            '"reasoning": "<1 sentence>"}'
        ),
    },
]

# Weights for each agent in the final ensemble (must sum to 1.0)
_AGENT_WEIGHTS = {
    "StatisticsAnalyst": 0.22,
    "TacticsScout":      0.18,
    "NewsIntelligence":  0.13,
    "Contrarian":        0.10,
    "InjurySpecialist":  0.15,
    "MarketIntelligence":0.10,
    "WeatherAnalyst":    0.04,
    "PsychologyAnalyst": 0.05,
    "HomeGroundExpert":  0.03,
}


@dataclass
class AgentEstimate:
    agent: str
    probability: float
    confidence: float
    reasoning: str


@dataclass
class DebateResult:
    consensus_probability: float          # final blended probability
    agent_estimates: list[AgentEstimate]
    debate_confidence: float              # avg confidence across agents
    summary: str                          # one-line summary of key driver


def _parse_agent_response(text: str) -> dict:
    """Extract JSON from agent response, tolerating markdown code fences."""
    text = text.strip()
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


def _build_context_prompt(
    side_a: str,
    side_b: str,
    sport: str,
    context: dict,
) -> str:
    """Format match context into a rich prompt for each agent."""
    lines = [
        f"Sport: {sport}",
        f"Match: {side_a} (side_a) vs {side_b} (side_b)",
    ]

    if context.get("date"):
        lines.append(f"Date: {context['date']}")
    if context.get("competition"):
        lines.append(f"Competition: {context['competition']}")
    if context.get("surface"):
        lines.append(f"Surface/Venue: {context['surface']}")

    # Statistical context
    stat_keys = [
        ("ml_probability_a", "ML model P(side_a wins)"),
        ("market_prob_a", "Market/Polymarket P(side_a wins)"),
        ("elo_a", "ELO side_a"), ("elo_b", "ELO side_b"),
        ("form_a", "Recent form side_a (0-1)"), ("form_b", "Recent form side_b"),
        ("avg_goals_a", "Avg goals/pts scored side_a"), ("avg_goals_b", "Avg goals/pts scored side_b"),
        ("avg_conceded_a", "Avg conceded side_a"), ("avg_conceded_b", "Avg conceded side_b"),
        ("h2h_wins_a", "H2H wins side_a in last 10"), ("h2h_wins_b", "H2H wins side_b in last 10"),
        ("home_advantage", "Home side (neutral if no home team)"),
        ("venue", "Venue/Stadium"),
        ("altitude_m", "Venue altitude (metres)"),
        ("travel_km_a", "Travel distance side_a (km)"),
        ("travel_km_b", "Travel distance side_b (km)"),
    ]
    stats = {label: context[key] for key, label in stat_keys if key in context}
    if stats:
        lines.append("\nStatistics:")
        for label, val in stats.items():
            lines.append(f"  {label}: {val}")

    # Weather
    weather = context.get("weather", {})
    if weather:
        lines.append("\nWeather conditions:")
        for k, v in weather.items():
            lines.append(f"  {k}: {v}")

    # Injuries / squad news
    injuries = context.get("injuries", [])
    if injuries:
        lines.append(f"\nInjury/suspension reports: {'; '.join(injuries)}")

    # News flags
    news = context.get("news_flags", [])
    if news:
        lines.append(f"\nNews/flags: {'; '.join(news)}")

    # Key factors from ML model
    factors = context.get("key_factors", [])
    if factors:
        lines.append(f"\nML key factors: {'; '.join(str(f) for f in factors[:5])}")

    lines.append(
        "\nGive your best probability that SIDE_A wins. "
        "Be decisive — output a JSON with probability, confidence, reasoning."
    )
    return "\n".join(lines)


def run_debate(
    side_a: str,
    side_b: str,
    sport: str,
    context: dict,
    timeout: int = 25,
) -> Optional[DebateResult]:
    """
    Run all nine agents sequentially and return the weighted consensus.
    Returns None if Anthropic API is unavailable.
    """
    try:
        import anthropic
    except ImportError:
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None

    user_prompt = _build_context_prompt(side_a, side_b, sport, context)
    estimates = []

    for agent in _AGENTS:
        try:
            response = client.messages.create(
                model="claude-opus-4-8",
                max_tokens=256,
                system=[
                    {"type": "text", "text": agent["system"], "cache_control": _CACHE_TYPE}
                ],
                messages=[{"role": "user", "content": user_prompt}],
                timeout=timeout,
            )
            raw = response.content[0].text if response.content else ""
            parsed = _parse_agent_response(raw)

            prob = float(parsed.get("probability", 0.5))
            conf = float(parsed.get("confidence", 0.5))
            reasoning = str(parsed.get("reasoning", ""))

            prob = max(0.05, min(0.95, prob))
            conf = max(0.0, min(1.0, conf))

            estimates.append(AgentEstimate(
                agent=agent["name"],
                probability=prob,
                confidence=conf,
                reasoning=reasoning,
            ))
        except Exception:
            # Skip failed agents — remaining agents still produce a consensus
            continue

    if not estimates:
        return None

    # Confidence-weighted average (weight = base_weight * agent_confidence)
    total_w = 0.0
    weighted_p = 0.0
    for est in estimates:
        base_w = _AGENT_WEIGHTS.get(est.agent, 0.25)
        w = base_w * (0.5 + 0.5 * est.confidence)
        weighted_p += w * est.probability
        total_w += w

    consensus_p = weighted_p / total_w if total_w > 0 else 0.5
    avg_conf = sum(e.confidence for e in estimates) / len(estimates)

    # Summary: highest-confidence agent's reasoning
    best = max(estimates, key=lambda e: e.confidence)
    summary = f"[{best.agent}] {best.reasoning}"

    return DebateResult(
        consensus_probability=round(consensus_p, 4),
        agent_estimates=estimates,
        debate_confidence=round(avg_conf, 4),
        summary=summary,
    )


def blend_with_ml(
    ml_probability: float,
    debate_result: Optional[DebateResult],
    ml_weight: float = 0.65,
) -> float:
    """
    Blend ML model probability with debate consensus.

    Weight allocation (adjustable):
      ml_weight (default 0.65): statistical ML models — large data, no reasoning
      debate_weight (0.35): multi-agent consensus — context, intangibles, reasoning

    The debate weight scales with debate_confidence so a low-confidence debate
    panel has less influence than a high-confidence one.
    """
    if debate_result is None:
        return ml_probability

    debate_weight = (1.0 - ml_weight) * debate_result.debate_confidence
    effective_ml_weight = 1.0 - debate_weight

    blended = effective_ml_weight * ml_probability + debate_weight * debate_result.consensus_probability
    return round(max(0.05, min(0.95, blended)), 4)
