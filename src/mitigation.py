"""
Mitigation strategies (Sect. 3.6) and the prompt-level baselines (Sect. 4.1).

  Masking   — values of F_mask replaced by UNKNOWN (Eq. 16)
  Filtering — fields of F_filt ⊆ F_mask omitted from the profile; F_filt is
              selected on the development split as the set with the lowest
              EOD whose BA is at most δ below the unmitigated scorer (ties →
              smallest set)
  FPO       — one fixed fairness instruction prepended to the system message
  Fair-ICL  — the k demonstrations balanced over sex × label

Each strategy is an Intervention that the serializer applies to every query.
"""

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np

from src.config import FILTER_BA_TOLERANCE, FPO_INSTRUCTION, MAX_FILTER_CANDIDATES

logger = logging.getLogger(__name__)

STRATEGIES = ["Baseline", "Masking", "Filtering", "FPO", "Fair-ICL"]


@dataclass
class Intervention:
    name: str
    mask_cols: list[str] = field(default_factory=list)
    omit_cols: list[str] = field(default_factory=list)
    system_prefix: str = ""
    demos: list | None = None          # None → the configuration's demonstrations


def masking(f_mask: list[str]) -> Intervention:
    return Intervention("Masking", mask_cols=list(f_mask))


def filtering(f_filt: list[str]) -> Intervention:
    return Intervention("Filtering", omit_cols=list(f_filt))


def fpo() -> Intervention:
    return Intervention("FPO", system_prefix=FPO_INSTRUCTION)


def fair_icl(balanced_demos: list) -> Intervention:
    return Intervention("Fair-ICL", demos=balanced_demos)


def filter_candidates(f_mask: list[str]) -> list[tuple[str, ...]]:
    """Non-empty subsets of F_mask; above MAX_FILTER_CANDIDATES features, nested top-PRS prefixes."""
    if len(f_mask) <= MAX_FILTER_CANDIDATES:
        return [c for r in range(1, len(f_mask) + 1)
                for c in itertools.combinations(f_mask, r)]
    logger.warning("|F_mask| = %d > %d; searching nested prefixes only",
                   len(f_mask), MAX_FILTER_CANDIDATES)
    return [tuple(f_mask[:r]) for r in range(1, len(f_mask) + 1)]


def select_filter_set(dev_metrics: dict[tuple[str, ...], dict], base_ba: float,
                      delta: float = FILTER_BA_TOLERANCE) -> tuple[tuple[str, ...], list[dict]]:
    """
    dev_metrics: candidate → metrics on the development split.
    Returns (F_filt, audit log of every candidate). An empty F_filt means no
    candidate met the BA tolerance, so filtering equals the baseline.
    """
    log = []
    for cand, m in dev_metrics.items():
        ok = m["BA"] >= base_ba - delta
        log.append({"features": list(cand), "BA": m["BA"], "EOD": m["EOD"], "feasible": ok})
    feasible = [(m["EOD"], len(c), c) for c, m in dev_metrics.items()
                if m["BA"] >= base_ba - delta and not np.isnan(m["EOD"])]
    if not feasible:
        logger.warning("No filtering candidate within BA tolerance δ=%.3f", delta)
        return (), log
    return min(feasible)[2], log
