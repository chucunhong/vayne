"""
Utility and group-fairness metrics (Sect. 3.2).

  BA   = (TPR + TNR) / 2
  Acc  = accuracy
  SP   = P(Ŷ=+ | ā) − P(Ŷ=+ | a)                    (signed, Eq. 5)
  EOD  = Σ_{y∈{+,−}} |P(Ŷ=+ | a, y) − P(Ŷ=+ | ā, y)|  (Eq. 6)
  PP   = |P(Y=+ | Ŷ=+, a) − P(Y=+ | Ŷ=+, ā)|         (Eq. 7; NaN if undefined)

Group a is the protected group (a_i = 0). Decisions are taken per scoring
run, Ŷ⁽ʳ⁾ = 1[s⁽ʳ⁾ ≥ t] (Eq. 4); every metric is computed on each run and
reported as the mean and standard deviation over the R runs.
"""

import numpy as np

from src.config import DECISION_THRESHOLD

METRICS = ["BA", "Acc", "SP", "EOD", "PP"]


def _rate(x: np.ndarray) -> float:
    return float(x.mean()) if len(x) else np.nan


def classification_metrics(score: np.ndarray, y: np.ndarray, a: np.ndarray,
                           threshold: float = DECISION_THRESHOLD) -> dict[str, float]:
    pred = (score >= threshold).astype(int)
    tpr = _rate(pred[y == 1])
    tnr = _rate(1 - pred[y == 0])
    out = {"BA": (tpr + tnr) / 2, "Acc": float((pred == y).mean())}

    prot, non = a == 0, a == 1
    out["SP"] = _rate(pred[non]) - _rate(pred[prot])
    out["EOD"] = sum(abs(_rate(pred[prot & (y == v)]) - _rate(pred[non & (y == v)]))
                     for v in (1, 0))
    out["PP"] = abs(_rate(y[prot & (pred == 1)]) - _rate(y[non & (pred == 1)]))
    return out


def metrics_over_runs(mat: np.ndarray, y: np.ndarray, a: np.ndarray,
                      threshold: float = DECISION_THRESHOLD) -> dict[str, float]:
    """
    mat: (n, R) score matrix, one column per scoring run.
    Returns {M: mean over runs, M_sd: SD over runs} for every metric; a run in
    which PP is undefined is left out of the PP mean.
    """
    runs = [classification_metrics(mat[:, r], y, a, threshold) for r in range(mat.shape[1])]
    out = {}
    for k in METRICS:
        vals = np.array([r[k] for r in runs], dtype=float)
        vals = vals[~np.isnan(vals)]
        out[k] = float(vals.mean()) if len(vals) else np.nan
        out[f"{k}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
    return out
