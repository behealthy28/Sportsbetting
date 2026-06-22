"""Abstract base class for sport-specific prediction handlers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PredictionResult:
    sport: str
    entity1: str                    # home team / player 1 / subject of prop
    entity2: str                    # away team / player 2 (or "" for solo props)
    date: str
    probabilities: dict             # {outcome: float}
    market_probs: Optional[dict]    # from Polymarket/Kalshi
    edges: dict                     # {outcome: {edge, rating, ...}}
    best_bet: Optional[str]
    best_edge_pct: float
    kelly_stake_pct: float
    confidence: float               # 0-100
    key_factors: list               # list of strings
    news_flags: list                # negative news items
    data_sources: list              # which scrapers were used
    model_breakdown: dict           # individual model contributions
    venue: str = "Unknown"
    competition: str = ""
    is_neutral: bool = False
    simulation: Optional[dict] = None   # Monte Carlo scoreline/goals breakdown
    # Prop betting fields
    bet_type: str = "match_result"  # match_result | over_under | btts | player_scorer |
                                    # player_foot | player_assists | method_victory |
                                    # goes_distance | round_ou | tennis_first_set |
                                    # tennis_tiebreak | tennis_sets | cricket_runs_ou
    prop_description: str = ""      # human-readable e.g. "Over 2.5 goals"
    prop_player: str = ""           # player name for player props


class AbstractSport(ABC):
    """Base class for all sport handlers."""

    @property
    @abstractmethod
    def sport_name(self) -> str:
        pass

    @property
    @abstractmethod
    def sport_keywords(self) -> list:
        """Keywords that identify this sport in natural language."""
        pass

    @abstractmethod
    def predict(
        self,
        entity1: str,
        entity2: str,
        date: str,
        context: dict,
    ) -> PredictionResult:
        """Run the full prediction pipeline."""
        pass

    def matches_sport(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.sport_keywords)
