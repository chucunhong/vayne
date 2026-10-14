"""
Repeated scoring: each prompt is queried R times and the analysis uses the
average s̄ (Eq. 3). Scores are kept as (n, R) matrices so that the bootstrap
can resample repeats within applicants.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from src.config import MAX_WORKERS, MAX_OUTPUT_TOKENS
from src.llm_client import query_p_yes

logger = logging.getLogger(__name__)


class Scorer:
    """Fixed model / temperature / output budget; counts queries and parse failures."""

    def __init__(self, model: str, temperature: float, max_tokens: int = MAX_OUTPUT_TOKENS,
                 max_workers: int = MAX_WORKERS):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.n_queries = 0
        self.n_failed = 0

    def score(self, prompts: list[tuple[str, str]], repeats: int,
              rep_offset: int = 0, desc: str = "") -> np.ndarray:
        """
        Query every (system, user) prompt `repeats` times.
        Returns an (n, repeats) float matrix with NaN where parsing failed.
        """
        n = len(prompts)
        out = np.full((n, repeats), np.nan)
        tasks = [(i, r) for i in range(n) for r in range(repeats)]

        def _run(i, r):
            system, user = prompts[i]
            return query_p_yes(system, user, self.model, self.temperature,
                               rep=rep_offset + r, max_tokens=self.max_tokens)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_run, i, r): (i, r) for i, r in tasks}
            for fut in tqdm(as_completed(futures), total=len(tasks), desc=desc, leave=False):
                i, r = futures[fut]
                try:
                    p = fut.result()
                except Exception as exc:
                    logger.warning("Query failed (%s, row %d, rep %d): %s", desc, i, r, exc)
                    p = None
                if p is not None:
                    out[i, r] = p

        self.n_queries += len(tasks)
        self.n_failed += int(np.isnan(out).sum())
        return out

    @property
    def fallback_rate(self) -> float:
        return self.n_failed / self.n_queries if self.n_queries else 0.0


def fill_baseline(mat: np.ndarray) -> np.ndarray:
    """
    Unmodified profiles: failed repeats are dropped from the average; an
    applicant whose every repeat failed gets 0.5.
    """
    out = mat.copy()
    all_nan = np.isnan(out).all(axis=1)
    if all_nan.any():
        logger.warning("%d applicants had no parseable baseline reply; using 0.5",
                       int(all_nan.sum()))
        out[all_nan] = 0.5
    return out


def fill_modified(mat: np.ndarray, base_mean: np.ndarray) -> np.ndarray:
    """
    Ablated / swapped profiles: a failed query falls back to the applicant's
    baseline score s̄(z), which biases DU and PU toward zero (Sect. 4.1).
    """
    out = mat.copy()
    rows, cols = np.where(np.isnan(out))
    out[rows, cols] = base_mean[rows]
    return out


def mean_score(mat: np.ndarray) -> np.ndarray:
    return np.nanmean(mat, axis=1)
