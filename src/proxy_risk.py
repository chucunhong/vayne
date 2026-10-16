"""
Proxy risk score PRS_j = PC_j × PU_j (Eq. 15), adaptive graph thresholds,
and the feature roles shared by the bar plot, the scatter plot, and the
proxy path graph.

Thresholds (Sect. 4.1, "Proxy graph thresholds"):
  τ₁ = median PC,  τ₂ = median PU,  τ₃ = median non-zero pairwise MI

Roles (one rule for every figure and table):
  high_risk     PC > τ₁ and PU > τ₂   (capacity and use edge; proxy risk zone)
  capacity_only PC > τ₁, PU ≤ τ₂
  use_only      PC ≤ τ₁, PU > τ₂
  low           neither
"""

import numpy as np

# Values within EPS of a threshold count as on it, not above it (a median
# threshold equals some feature's value up to floating-point noise).
EPS = 1e-9

ROLE_TEXT = {
    "high_risk":     "Complete path",
    "capacity_only": "Capacity only",
    "use_only":      "Use only",
    "low":           "Neither",
}


def proxy_risk(pc: dict[str, float], pu: dict[str, float]) -> dict[str, float]:
    return {f: pc[f] * pu[f] for f in pu if f in pc}


def thresholds(pc: dict[str, float], pu: dict[str, float],
               pair_mi: dict | None = None) -> tuple[float, float, float]:
    tau1 = float(np.median(list(pc.values())))
    tau2 = float(np.median(list(pu.values())))
    nz = [v for v in (pair_mi or {}).values() if v > 0]
    tau3 = float(np.median(nz)) if nz else np.inf
    return tau1, tau2, tau3


def role(pc: float, pu: float, tau1: float, tau2: float) -> str:
    cap, use = pc > tau1 + EPS, pu > tau2 + EPS
    if cap and use:
        return "high_risk"
    if cap:
        return "capacity_only"
    if use:
        return "use_only"
    return "low"


def mask_set(prs: dict[str, float], theta: float | None) -> tuple[list[str], float]:
    """
    F_mask = {j : PRS_j > θ_PRS}, ordered by PRS. The paper sets θ_PRS per
    dataset from the RQ2 diagnosis (src/config.py); without one, θ falls back
    to the 75th percentile of PRS.
    """
    if theta is None:
        theta = float(np.percentile(list(prs.values()), 75))
    feats = sorted((f for f, v in prs.items() if v > theta), key=lambda f: -prs[f])
    return feats, theta
