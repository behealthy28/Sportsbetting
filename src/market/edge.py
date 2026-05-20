"""Edge calculation: model probability vs market implied probability."""
from src.market.odds import remove_vig


EDGE_THRESHOLDS = {
    "STRONG": 0.05,     # >5% edge → strong value bet
    "MODERATE": 0.02,   # 2-5% edge → moderate value
    "WEAK": 0.00,       # 0-2% → weak / no bet
    "NEGATIVE": -999,   # negative edge → avoid
}


def calculate_edge(model_probs: dict, market_probs: dict) -> dict:
    """
    Calculate edge for each outcome.
    edge = model_prob - devigged_market_prob
    Returns dict with edge values and ratings.
    """
    if not market_probs:
        return {}

    devigged = remove_vig({k: v for k, v in market_probs.items() if v is not None})

    edges = {}
    for outcome, model_p in model_probs.items():
        if model_p is None:
            continue
        market_p = devigged.get(outcome)
        if market_p is None:
            continue

        edge_val = model_p - market_p
        edges[outcome] = {
            "model_prob": round(model_p, 4),
            "market_prob": round(market_p, 4),
            "edge": round(edge_val, 4),
            "edge_pct": round(edge_val * 100, 2),
            "rating": _rate_edge(edge_val),
        }

    return edges


def _rate_edge(edge: float) -> str:
    if edge >= EDGE_THRESHOLDS["STRONG"]:
        return "STRONG"
    elif edge >= EDGE_THRESHOLDS["MODERATE"]:
        return "MODERATE"
    elif edge >= EDGE_THRESHOLDS["WEAK"]:
        return "WEAK"
    return "NEGATIVE"


def best_bet(edges: dict) -> tuple:
    """Return (outcome, edge_info) for the highest positive edge bet."""
    if not edges:
        return None, None

    positive = {k: v for k, v in edges.items() if v["edge"] > 0}
    if not positive:
        return None, None

    best = max(positive.items(), key=lambda x: x[1]["edge"])
    return best


# Minimum model probability to surface a bet recommendation
MIN_BET_CONFIDENCE = 0.65


def high_confidence_best_bet(edges: dict, model_probs: dict,
                              min_confidence: float = MIN_BET_CONFIDENCE) -> tuple:
    """
    Like best_bet() but only returns the bet if the model probability
    for that outcome is >= min_confidence AND edge is positive.
    This targets the user's 65% win-rate goal by filtering to situations
    where the model is sufficiently certain.
    """
    if not edges or not model_probs:
        return None, None

    positive = {
        k: v for k, v in edges.items()
        if v["edge"] > 0 and model_probs.get(k, 0) >= min_confidence
    }
    if not positive:
        return None, None

    best = max(positive.items(), key=lambda x: x[1]["edge"])
    return best
