"""
Traditional credit-scoring baselines (Sect. 4.1): Decision Tree, Naive Bayes,
Multilayer Perceptron, k-Nearest Neighbors with scikit-learn defaults,
trained on the training split and evaluated on the same test records as the
LLM configurations. Categorical features are one-hot encoded and numerical
features standardized on the training split; the protected attribute is not
an input, matching the deployment profile.
"""

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import N_BOOTSTRAP, SEED
from src.data_loader import DatasetSplits
from src.fairness_metrics import metrics_with_sd
from src.proxy_capacity import feature_kind

MODELS = {
    "DT":  lambda: DecisionTreeClassifier(random_state=SEED),
    "NB":  lambda: GaussianNB(),
    "MLP": lambda: MLPClassifier(random_state=SEED),
    "kNN": lambda: KNeighborsClassifier(),
}


def _design(splits: DatasetSplits):
    cols = splits.feature_cols
    kinds = {c: feature_kind(splits.train.df[c]) for c in cols}
    cat = [c for c in cols if kinds[c] == "categorical"]
    num = [c for c in cols if kinds[c] == "continuous"]

    def frame(df: pd.DataFrame) -> pd.DataFrame:
        out = df[cols].copy()
        out[cat] = out[cat].astype(str)
        for c in num:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ])
    return pre, frame


def evaluate_traditional(splits: DatasetSplits, n_boot: int = N_BOOTSTRAP) -> list[dict]:
    """One row per baseline: point metrics on the test split with bootstrap SDs."""
    pre, frame = _design(splits)
    X_tr, X_te = frame(splits.train.df), frame(splits.test.df)
    rows = []
    for name, make in MODELS.items():
        clf = make_pipeline(clone(pre), make())
        clf.fit(X_tr, splits.train.y)
        pred = clf.predict(X_te).astype(float)
        m = metrics_with_sd(pred[:, None], splits.test.y, splits.test.a, n_boot)
        rows.append({"Model": name, **m})
    return rows
