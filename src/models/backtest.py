"""Backtesting & calibration harness.

Measures whether the model's probabilities are actually *accurate* against
historical results — the legitimate way to "maintain accuracy", as opposed to
bolting on unverifiable private data.

Metrics:
  - Brier score (multiclass): mean squared error of predicted prob vectors.
    Lower is better; 0 = perfect.
  - Log loss: penalises confident wrong calls harder. Lower is better.
  - Reliability bins: of the times the model said "~p", how often did it
    actually happen? A well-calibrated model has hit-rate ~= predicted prob.

Input format (JSON list), each record:
    {"probs": {"home_win": .., "draw": .., "away_win": ..}, "outcome": "home_win"}
Outcome must be one of the keys in ``probs``.
"""
import json
import math
from typing import Optional


def brier_score(records: list) -> float:
    """Multiclass Brier score across records."""
    if not records:
        return float("nan")
    total = 0.0
    for r in records:
        probs, outcome = r["probs"], r["outcome"]
        for k, p in probs.items():
            y = 1.0 if k == outcome else 0.0
            total += (p - y) ** 2
    return round(total / len(records), 5)


def log_loss(records: list, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood of the realised outcomes."""
    if not records:
        return float("nan")
    total = 0.0
    for r in records:
        p = max(eps, min(1.0, r["probs"].get(r["outcome"], eps)))
        total += -math.log(p)
    return round(total / len(records), 5)


def reliability_bins(records: list, n_bins: int = 10) -> list:
    """Group every (prob, hit) pair into bins; compare predicted vs observed.

    Returns a list of {bin, count, avg_pred, observed} for non-empty bins.
    """
    bins = [{"lo": i / n_bins, "hi": (i + 1) / n_bins, "preds": [], "hits": []}
            for i in range(n_bins)]
    for r in records:
        for k, p in r["probs"].items():
            idx = min(n_bins - 1, int(p * n_bins))
            bins[idx]["preds"].append(p)
            bins[idx]["hits"].append(1.0 if k == r["outcome"] else 0.0)

    out = []
    for b in bins:
        if not b["preds"]:
            continue
        out.append({
            "bin": f"{b['lo']:.1f}-{b['hi']:.1f}",
            "count": len(b["preds"]),
            "avg_pred": round(sum(b["preds"]) / len(b["preds"]), 3),
            "observed": round(sum(b["hits"]) / len(b["hits"]), 3),
        })
    return out


def expected_calibration_error(records: list, n_bins: int = 10) -> float:
    """Weighted mean gap between predicted prob and observed frequency."""
    bins = reliability_bins(records, n_bins)
    total = sum(b["count"] for b in bins) or 1
    ece = sum(b["count"] * abs(b["avg_pred"] - b["observed"]) for b in bins)
    return round(ece / total, 5)


def evaluate(records: list, n_bins: int = 10) -> dict:
    """Full report for a set of historical prediction records."""
    return {
        "n": len(records),
        "brier": brier_score(records),
        "log_loss": log_loss(records),
        "ece": expected_calibration_error(records, n_bins),
        "reliability": reliability_bins(records, n_bins),
    }


def load_records(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [r for r in data if "probs" in r and "outcome" in r]


def render(report: dict) -> None:
    """Pretty-print a report (uses rich if available, else plain)."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        c = Console()
        c.print(f"\n[bold]Backtest[/bold]  ·  n={report['n']}  "
                f"Brier={report['brier']}  LogLoss={report['log_loss']}  "
                f"ECE={report['ece']}")
        t = Table(box=box.SIMPLE, header_style="bold dim")
        for col in ("Prob bin", "Count", "Predicted", "Observed", "Gap"):
            t.add_column(col)
        for b in report["reliability"]:
            gap = b["observed"] - b["avg_pred"]
            t.add_row(b["bin"], str(b["count"]), f"{b['avg_pred']:.3f}",
                      f"{b['observed']:.3f}", f"{gap:+.3f}")
        c.print(t)
    except ImportError:
        print(f"Backtest n={report['n']} Brier={report['brier']} "
              f"LogLoss={report['log_loss']} ECE={report['ece']}")
        for b in report["reliability"]:
            print(f"  {b['bin']}: n={b['count']} pred={b['avg_pred']} obs={b['observed']}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("usage: python -m src.models.backtest <records.json>")
        sys.exit(1)
    render(evaluate(load_records(path)))
