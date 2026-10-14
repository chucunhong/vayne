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
    "high_risk":     "High PC and PU",
    "capacity_only": "High PC, low PU",
    "use_only":      "Low PC, high PU",
    "low":           "Low PC and PU",
}


def proxy_risk(pc: dict[str, float], pu: dict[str, float]) -> dict[str, float]:
    return {f: pc[f] * pu[f] for f in pu if f in pc}


def proxy_risk_sd(pc_boot: np.ndarray, pu_boot: np.ndarray, feats: list[str]) -> dict[str, float]:
    """SD of PC_b × PU_b over paired bootstrap replicates (columns aligned with feats)."""
    n = min(len(pc_boot), len(pu_boot))
    if n < 2:
        return {f: np.nan for f in feats}
    prod = pc_boot[:n] * pu_boot[:n]
    return {f: float(prod[:, j].std(ddof=1)) for j, f in enumerate(feats)}


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
    """F_mask = {j : PRS_j > θ_PRS}, ordered by PRS. θ defaults to the 75th percentile."""
    if theta is None:
        theta = float(np.percentile(list(prs.values()), 75))
    feats = sorted((f for f, v in prs.items() if v > theta), key=lambda f: -prs[f])
    return feats, theta
