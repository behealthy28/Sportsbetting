"""
Played WC2026 results — update this list as games finish. Consumed by
wc2026_recalc.py, which PINS these fixtures and re-simulates the rest to
condition the title/advancement odds on what has actually happened.

Format: (home, home_goals, away, away_goals). Orientation does not matter —
wc2026_montecarlo.played_lookup stores both directions.

As of 2026-06-15: matchday 1 done for Groups A-F (10 games). Ivory Coast vs
Ecuador (Group E) still in progress -> NOT pinned, still simulated.
"""
PLAYED = [
    # ---- Group A ----
    ("Mexico", 2, "South Africa", 0),     # model call, user-confirmed
    ("Korea Republic", 2, "Czechia", 1),
    # ---- Group B ----
    ("Canada", 1, "Bosnia", 1),
    ("Switzerland", 1, "Qatar", 1),
    # ---- Group C ----
    ("Brazil", 1, "Morocco", 1),
    ("Scotland", 1, "Haiti", 0),
    # ---- Group D ----
    ("USA", 4, "Paraguay", 1),
    ("Australia", 2, "Turkey", 0),        # upset
    # ---- Group E ----
    ("Germany", 7, "Curacao", 1),         # Germany ran riot (Havertz x2, Musiala...)
    # ---- Group F ----
    ("Netherlands", 2, "Japan", 2),       # Japan came back twice; Kamada 88' equaliser
]
