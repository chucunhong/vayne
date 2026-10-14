"""
Utility and group-fairness metrics (Sect. 3.2) and the bootstrap used for
every reported standard deviation (Sect. 4.1, "Repeated scoring and
statistical reporting").

  BA   = (TPR + TNR) / 2
  Acc  = accuracy
  SP   = P(Ŷ=+ | ā) − P(Ŷ=+ | a)                    (signed, Eq. 5)
  EOD  = Σ_{y∈{+,−}} |P(Ŷ=+ | a, y) − P(Ŷ=+ | ā, y)|  (Eq. 6)
  PP   = |P(Y=+ | Ŷ=+, a) − P(Y=+ | Ŷ=+, ā)|         (Eq. 7; NaN if undefined)

Group a is the protected group (a_i = 0).
"""

import numpy as np

from src.config import DECISION_THRESHOLD, N_BOOTSTRAP, SEED

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


# ── Bootstrap ───────────────────────────────────────────────────────────────

def stratified_rows(rng: np.random.Generator, y: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Resample applicants with replacement within each (A, Y) cell."""
    cell = a * 2 + y
    rows = []
    for c in np.unique(cell):
        members = np.flatnonzero(cell == c)
        rows.append(rng.choice(members, size=len(members), replace=True))
    return np.concatenate(rows)


def resampled_mean(rng: np.random.Generator, mat: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """s̄ for the resampled applicants, drawing their R stored scores with replacement."""
    R = mat.shape[1]
    if R == 1:
        return mat[rows, 0]
    cols = rng.integers(0, R, size=(len(rows), R))
    return np.nanmean(mat[rows[:, None], cols], axis=1)


def bootstrap(stat_fn, mats: dict[str, np.ndarray], y: np.ndarray, a: np.ndarray,
              n_boot: int = N_BOOTSTRAP, seed: int = SEED,
              resample_repeats: bool = True) -> list[dict[str, float]]:
    """
    Generic stratified bootstrap.

    stat_fn(means: dict[name → s̄ vector], y, a) → dict[str, float]
    mats   : name → (n, R) score matrix over the same applicants
    Returns one stat dict per bootstrap replicate.
    """
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        rows = stratified_rows(rng, y, a)
        if resample_repeats:
            means = {k: resampled_mean(rng, m, rows) for k, m in mats.items()}
        else:
            means = {k: np.nanmean(m[rows], axis=1) for k, m in mats.items()}
        reps.append(stat_fn(means, y[rows], a[rows]))
    return reps


def sd_from_reps(reps: list[dict[str, float]]) -> dict[str, float]:
    keys = reps[0].keys()
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in reps], dtype=float)
        vals = vals[~np.isnan(vals)]
        out[k] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
    return out


def metrics_with_sd(mat: np.ndarray, y: np.ndarray, a: np.ndarray,
                    n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> dict[str, float]:
    """Point estimates on s̄ plus bootstrap SDs, returned as {M: value, M_sd: sd}."""
    point = classification_metrics(np.nanmean(mat, axis=1), y, a)
    reps = bootstrap(lambda m, yy, aa: classification_metrics(m["s"], yy, aa),
                     {"s": mat}, y, a, n_boot, seed)
    sd = sd_from_reps(reps)
    out = {}
    for k in METRICS:
        out[k] = point[k]
        out[f"{k}_sd"] = sd[k]
    return out
