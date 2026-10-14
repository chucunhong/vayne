"""
Proxy capacity PC_j = I(X_j; A), in nats, computed on the training split
(Sect. 3.4.1).

  categorical X_j : plug-in estimate from the empirical joint table (Eq. 12)
  continuous  X_j : nearest-neighbour estimator of Ross (2014) for a
                    continuous–discrete pair, K = 3 (Eq. 13); negative
                    estimates are truncated at zero

Integer features with at most MAX_CATEGORICAL_LEVELS distinct values are
treated as categorical, since their ties cannot be resolved by the
nearest-neighbour construction.

Pairwise I(X_j; X_k) among non-protected features gives the auxiliary
association links of the proxy path graph and the threshold τ₃.
"""

import itertools
import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mutual_info_score

from src.config import KNN_NEIGHBORS, MAX_CATEGORICAL_LEVELS, N_BOOTSTRAP, SEED
from src.fairness_metrics import stratified_rows

logger = logging.getLogger(__name__)


def feature_kind(series: pd.Series) -> str:
    """'categorical' or 'continuous'."""
    num = pd.to_numeric(series, errors="coerce")
    known = series.astype(str) != "?"
    if num[known].isna().mean() > 0.05:
        return "categorical"
    vals = num.dropna()
    is_int = np.allclose(vals, np.round(vals))
    if is_int and vals.nunique() <= MAX_CATEGORICAL_LEVELS:
        return "categorical"
    return "continuous"


def _codes(series: pd.Series) -> np.ndarray:
    return pd.factorize(series.astype(str))[0]


def _mi_cat_cat(x: np.ndarray, z: np.ndarray) -> float:
    return float(mutual_info_score(x, z))


def _mi_cont_cat(x: np.ndarray, z: np.ndarray, seed: int) -> float:
    ok = ~np.isnan(x)
    mi = mutual_info_classif(x[ok].reshape(-1, 1), z[ok], discrete_features=False,
                             n_neighbors=KNN_NEIGHBORS, random_state=seed)[0]
    return max(0.0, float(mi))


def _mi_cont_cont(x: np.ndarray, z: np.ndarray, seed: int) -> float:
    ok = ~np.isnan(x) & ~np.isnan(z)
    mi = mutual_info_regression(x[ok].reshape(-1, 1), z[ok], discrete_features=False,
                                n_neighbors=KNN_NEIGHBORS, random_state=seed)[0]
    return max(0.0, float(mi))


def _encode(df: pd.DataFrame, cols: list[str]) -> tuple[dict, dict]:
    kinds, arrays = {}, {}
    for c in cols:
        kinds[c] = feature_kind(df[c])
        arrays[c] = (_codes(df[c]) if kinds[c] == "categorical"
                     else pd.to_numeric(df[c], errors="coerce").to_numpy(float))
    return kinds, arrays


def _pc_all(arrays: dict, kinds: dict, a: np.ndarray, seed: int) -> dict[str, float]:
    return {c: (_mi_cat_cat(x, a) if kinds[c] == "categorical" else _mi_cont_cat(x, a, seed))
            for c, x in arrays.items()}


def compute_proxy_capacity(df: pd.DataFrame, cols: list[str], a: np.ndarray,
                           y: np.ndarray | None = None, n_boot: int = N_BOOTSTRAP,
                           seed: int = SEED, n_jobs: int = -1) -> dict:
    """
    PC_j for every feature with bootstrap SDs over resampled training records.

    Returns {"pc": {feat: value}, "pc_sd": {feat: sd}, "kinds": {feat: kind},
             "boot": (n_boot, d) array aligned with `cols`}.

    Note: resampling with replacement duplicates records, which the
    nearest-neighbour estimator sees as zero-distance neighbours; bootstrap
    replicates of continuous features are therefore biased upward and the SD
    should be read as approximate.
    """
    kinds, arrays = _encode(df, cols)
    point = _pc_all(arrays, kinds, a, seed)

    boot = np.zeros((0, len(cols)))
    sd = {c: np.nan for c in cols}
    if n_boot > 0:
        strat_y = y if y is not None else np.zeros_like(a)
        rng = np.random.default_rng(seed)
        row_sets = [stratified_rows(rng, strat_y, a) for _ in range(n_boot)]

        def _one(rows, b):
            sub = {c: x[rows] for c, x in arrays.items()}
            vals = _pc_all(sub, kinds, a[rows], seed + b + 1)
            return [vals[c] for c in cols]

        boot = np.array(Parallel(n_jobs=n_jobs)(
            delayed(_one)(rows, b) for b, rows in enumerate(row_sets)))
        sd = {c: float(boot[:, j].std(ddof=1)) for j, c in enumerate(cols)}

    logger.info("Proxy capacity: top-3 %s",
                sorted(point.items(), key=lambda kv: -kv[1])[:3])
    return {"pc": point, "pc_sd": sd, "kinds": kinds, "boot": boot}


def pairwise_feature_mi(df: pd.DataFrame, cols: list[str], seed: int = SEED,
                        n_jobs: int = -1) -> dict[tuple[str, str], float]:
    """I(X_j; X_k) for every unordered pair of non-protected features."""
    kinds, arrays = _encode(df, cols)

    def _pair(c1, c2):
        k1, k2 = kinds[c1], kinds[c2]
        x1, x2 = arrays[c1], arrays[c2]
        if k1 == "categorical" and k2 == "categorical":
            return _mi_cat_cat(x1, x2)
        if k1 == "continuous" and k2 == "categorical":
            return _mi_cont_cat(x1, x2, seed)
        if k1 == "categorical" and k2 == "continuous":
            return _mi_cont_cat(x2, x1, seed)
        return _mi_cont_cont(x1, x2, seed)

    pairs = list(itertools.combinations(cols, 2))
    vals = Parallel(n_jobs=n_jobs)(delayed(_pair)(c1, c2) for c1, c2 in pairs)
    return dict(zip(pairs, vals))
