"""
Data loading, train/dev/test splits, audit subsampling, and demonstrations.

Splits are stratified by (A, Y) with a fixed seed. Demonstrations are drawn
only from the training split; every record within a configuration is scored
with the same demonstration set.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, SEED, SPLIT_FRACTIONS

logger = logging.getLogger(__name__)


@dataclass
class Split:
    """One data split with labels y (1 = positive class) and group a (0 = protected)."""
    df: pd.DataFrame
    y:  np.ndarray
    a:  np.ndarray

    def __len__(self) -> int:
        return len(self.df)

    def subset(self, idx: np.ndarray) -> "Split":
        return Split(self.df.iloc[idx].reset_index(drop=True), self.y[idx], self.a[idx])


@dataclass
class DatasetSplits:
    config: dict
    train:  Split
    dev:    Split
    test:   Split
    full_sizes: tuple[int, int, int]   # split sizes before audit subsampling

    @property
    def feature_cols(self) -> list[str]:
        return feature_cols(self.train.df, self.config)


def load_dataset(config: dict) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / config["file"], skipinitialspace=True, low_memory=False)
    df = df.drop(columns=[c for c in config.get("exclude_cols", []) if c in df.columns])
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    # A record without a recorded protected attribute cannot be assigned to a group.
    sa = config["sensitive_col"]
    known = df[sa].astype(str).isin([str(v) for v in config["protected_display"]])
    if (~known).any():
        logger.info("%s: dropping %d records with missing %s",
                    config["name"], int((~known).sum()), sa)
        df = df[known]
    return df.reset_index(drop=True)


def feature_cols(df: pd.DataFrame, config: dict) -> list[str]:
    """Non-protected features: the deployment profile z = x_{-A}."""
    skip = {config["sensitive_col"], config["target_col"]}
    return [c for c in df.columns if c not in skip]


def labels(df: pd.DataFrame, config: dict) -> np.ndarray:
    pos = str(config["positive_label"])
    return (df[config["target_col"]].astype(str) == pos).astype(int).to_numpy()


def groups(df: pd.DataFrame, config: dict) -> np.ndarray:
    """a_i = 0 for the protected group, 1 otherwise."""
    prot = str(config["protected_value"])
    return (df[config["sensitive_col"]].astype(str) != prot).astype(int).to_numpy()


def make_splits(config: dict, seed: int = SEED, audit: bool = True) -> DatasetSplits:
    """
    Stratified 60/20/20 split. With audit=True, dev and test are subsampled to
    config['audit_n'] records (stratified), as done for Credit Card.
    """
    df = load_dataset(config)
    y, a = labels(df, config), groups(df, config)
    strata = a * 2 + y
    idx = np.arange(len(df))

    f_tr, f_dev, f_te = SPLIT_FRACTIONS
    idx_tr, idx_rest = train_test_split(idx, train_size=f_tr, stratify=strata,
                                        random_state=seed)
    idx_dev, idx_te = train_test_split(idx_rest, test_size=f_te / (f_dev + f_te),
                                       stratify=strata[idx_rest], random_state=seed)

    whole = Split(df, y, a)
    train, dev, test = whole.subset(idx_tr), whole.subset(idx_dev), whole.subset(idx_te)
    sizes = (len(train), len(dev), len(test))

    if audit and config.get("audit_n"):
        n_dev, n_te = config["audit_n"]
        dev  = _stratified_subsample(dev,  n_dev, seed)
        test = _stratified_subsample(test, n_te,  seed)

    logger.info("%s: train/dev/test = %d/%d/%d (audited dev/test = %d/%d)",
                config["name"], *sizes, len(dev), len(test))
    return DatasetSplits(config, train, dev, test, sizes)


def _stratified_subsample(split: Split, n: int, seed: int) -> Split:
    if n >= len(split):
        return split
    idx, _ = train_test_split(np.arange(len(split)), train_size=n,
                              stratify=split.a * 2 + split.y, random_state=seed)
    return split.subset(np.sort(idx))


# ── Demonstrations ──────────────────────────────────────────────────────────

def sample_demos(train: Split, k: int, seed: int = SEED) -> list[tuple[pd.Series, int]]:
    """k demonstrations uniformly from the training split; with replacement if k > |train|."""
    if k <= 0:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train), size=k, replace=k > len(train))
    return [(train.df.iloc[i], int(train.y[i])) for i in idx]


def sample_balanced_demos(train: Split, per_cell: int,
                          seed: int = SEED) -> list[tuple[pd.Series, int]]:
    """Fair-ICL: per_cell demonstrations in each sex × label cell, shuffled."""
    rng = np.random.default_rng(seed)
    idx = []
    for a_val in (0, 1):
        for y_val in (0, 1):
            cell = np.flatnonzero((train.a == a_val) & (train.y == y_val))
            if len(cell) == 0:
                raise ValueError(f"Empty training cell a={a_val}, y={y_val}")
            idx.extend(rng.choice(cell, size=per_cell, replace=per_cell > len(cell)))
    idx = rng.permutation(idx)
    return [(train.df.iloc[i], int(train.y[i])) for i in idx]
