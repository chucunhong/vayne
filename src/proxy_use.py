"""
Score-difference statistics computed on the development split:

  NF        = E_z |s̄⁽¹⁾(z) − s̄⁽²⁾(z)|                  noise floor (Eq. 11)
  DU_A      = E_z |s̄(z⁽¹⁾) − s̄(z⁽⁰⁾)|                  paired swap (Eq. 9)
  DU_A^abl  = E_(a,z) |s̄(z⁽ᵃ⁾) − s̄(z⁽ᵘ⁾)|               ablation variant (Eq. 10)
  PU_j      = E_z |s̄(z) − s̄(z₋ⱼ)|                     proxy use (Eq. 14)

All inputs are (n, R) score matrices over the same applicants; SDs come from
the stratified applicant-and-repeat bootstrap.
"""

import numpy as np

from src.config import N_BOOTSTRAP, SEED
from src.fairness_metrics import resampled_mean, stratified_rows


def _mad(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.mean(np.abs(u - v)))


def sensitivity(base_1: np.ndarray, base_2: np.ndarray, z1: np.ndarray, z0: np.ndarray,
                zu: np.ndarray, y: np.ndarray, a: np.ndarray,
                n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> dict[str, float]:
    """
    base_1, base_2 : two independent sets of R queries of the deployment profile
    z1, z0         : audit profiles with A inserted as the non-protected / protected value
    zu             : audit profile with A inserted as UNKNOWN
    """
    def _stats(m1, m2, s1, s0, su, aa):
        own = np.where(aa == 1, s1, s0)
        return {"NF": _mad(m1, m2), "DU_A": _mad(s1, s0), "DU_A_abl": _mad(own, su)}

    mean = lambda m: np.nanmean(m, axis=1)
    point = _stats(mean(base_1), mean(base_2), mean(z1), mean(z0), mean(zu), a)

    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        rows = stratified_rows(rng, y, a)
        reps.append(_stats(*(resampled_mean(rng, m, rows)
                             for m in (base_1, base_2, z1, z0, zu)), a[rows]))
    out = dict(point)
    for k in point:
        out[f"{k}_sd"] = float(np.std([r[k] for r in reps], ddof=1)) if n_boot > 1 else np.nan
    return out


def proxy_use(base: np.ndarray, ablated: dict[str, np.ndarray], y: np.ndarray,
              a: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    """
    PU_j for every ablated feature.
    Returns {"pu": {feat: v}, "pu_sd": {feat: sd}, "boot": (n_boot, d) array}.
    """
    feats = list(ablated)
    s = np.nanmean(base, axis=1)
    pu = {f: _mad(s, np.nanmean(ablated[f], axis=1)) for f in feats}

    rng = np.random.default_rng(seed)
    boot = np.zeros((n_boot, len(feats)))
    for b in range(n_boot):
        rows = stratified_rows(rng, y, a)
        sb = resampled_mean(rng, base, rows)
        for j, f in enumerate(feats):
            boot[b, j] = _mad(sb, resampled_mean(rng, ablated[f], rows))
    sd = {f: float(boot[:, j].std(ddof=1)) if n_boot > 1 else np.nan
          for j, f in enumerate(feats)}
    return {"pu": pu, "pu_sd": sd, "boot": boot}
