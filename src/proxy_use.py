"""
Ablation-based sensitivities computed on the development split:

  DU_A = E_x |s̄(x) − s̄(x₋A)|     direct sensitivity (Eq. 9): the record with
                                  its stated A value vs. A replaced by UNKNOWN
  PU_j = E_z |s̄(z) − s̄(z₋ⱼ)|     proxy use (Eq. 14): the deployment profile
                                  vs. X_j replaced by UNKNOWN

All inputs are (n, R) score matrices over the same applicants; s̄ is the
average over the R repeated queries (Eq. 3).
"""

import numpy as np


def _mad(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.mean(np.abs(u - v)))


def _mean(m: np.ndarray) -> np.ndarray:
    return np.nanmean(m, axis=1)


def direct_sensitivity(stated: np.ndarray, unknown: np.ndarray) -> float:
    return _mad(_mean(stated), _mean(unknown))


def proxy_use(base: np.ndarray, ablated: dict[str, np.ndarray]) -> dict[str, float]:
    s = _mean(base)
    return {f: _mad(s, _mean(m)) for f, m in ablated.items()}
