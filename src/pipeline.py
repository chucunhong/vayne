"""
VAYNE pipeline (Algorithm 1) for one dataset under one audit configuration.

  1. Split D into train/dev/test; draw the k demonstrations from train
  2. Test split: repeated scoring of the deployment profile → baseline metrics
  3. Dev split: repeated scoring of the deployment profile → s̄(z)
  4. Dev split: A inserted with the stated value and with UNKNOWN → DU_A
  5. Train split: proxy capacity PC_j and pairwise feature MI
  6. Dev split: single-feature ablation → proxy use PU_j
  7. PRS_j, thresholds τ₁ τ₂ τ₃, proxy path graph
  8. F_mask from θ_PRS; F_filt selected on dev
  9. Test split: Masking, Filtering, FPO, Fair-ICL
 10. Traditional baselines on the same test records

Every score matrix is returned so that statistics can be recomputed offline.
"""

import logging
import time

import numpy as np

from src.config import (
    AUDIT_ENCODING, AUDIT_K, AUDIT_MODEL, AUDIT_TEMPERATURE, DATASET_CONFIGS,
    FAIR_ICL_PER_CELL, FILTER_BA_TOLERANCE, MAX_OUTPUT_TOKENS, MITIGATE_DEMOS,
    N_REPEATS, SEED,
)
from src.data_loader import Split, make_splits, sample_balanced_demos, sample_demos
from src.fairness_metrics import classification_metrics, metrics_over_runs
from src.llm_client import MOCK
from src.mitigation import (
    Intervention, fair_icl, filter_candidates, filtering, fpo, masking, select_filter_set,
)
from src.proxy_capacity import compute_proxy_capacity, pairwise_feature_mi
from src.proxy_graph import build_proxy_graph, get_proxy_paths, graph_to_dict
from src.proxy_risk import mask_set, proxy_risk, role, thresholds
from src.proxy_use import direct_sensitivity, proxy_use
from src.scoring import Scorer, fill_baseline, fill_modified, mean_score
from src.serializer import UNKNOWN, serialize
from src.traditional import evaluate_traditional

logger = logging.getLogger(__name__)


class _Prompter:
    """Builds the prompts of one configuration for a split."""

    def __init__(self, config: dict, encoding: str, demos: list):
        self.config, self.encoding, self.demos = config, encoding, demos

    def __call__(self, split: Split, iv: Intervention | None = None,
                 mask_cols=(), insert_protected=None) -> list[tuple[str, str]]:
        """insert_protected: None, one value for every record, or one value per record."""
        iv = iv or Intervention("Baseline")
        if insert_protected is None or isinstance(insert_protected, str):
            insert_protected = [insert_protected] * len(split)
        demos = iv.demos if iv.demos is not None else self.demos
        mask = list(mask_cols) + iv.mask_cols
        demo_mask = iv.mask_cols if MITIGATE_DEMOS else ()
        demo_omit = iv.omit_cols if MITIGATE_DEMOS else ()
        return [serialize(row, self.config, self.encoding, demos,
                          mask_cols=mask, omit_cols=iv.omit_cols,
                          insert_protected=ins,
                          system_prefix=iv.system_prefix,
                          demo_mask_cols=demo_mask, demo_omit_cols=demo_omit)
                for (_, row), ins in zip(split.df.iterrows(), insert_protected)]


def run_audit(
    dataset: str,
    model: str = AUDIT_MODEL,
    encoding: str = AUDIT_ENCODING,
    k: int = AUDIT_K,
    temperature: float = AUDIT_TEMPERATURE,
    repeats: int = N_REPEATS,
    theta_prs: float | None = None,
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> dict:
    config = DATASET_CONFIGS[dataset]
    t0 = time.time()
    splits = make_splits(config)
    train, dev, test = splits.train, splits.dev, splits.test
    feats = splits.feature_cols
    R = repeats

    demos = sample_demos(train, k)
    prompts = _Prompter(config, encoding, demos)
    scorer = Scorer(model, temperature, max_tokens)

    n_q = len(test) * R * 5 + len(dev) * R * (3 + len(feats))
    logger.info("%s: ≈%d queries before filtering selection (R=%d, %d features)",
                dataset, n_q, R, len(feats))

    # ── 2. Baseline on the test split ───────────────────────────────────────
    test_mats = {"Baseline": fill_baseline(scorer.score(prompts(test), R, desc="test baseline"))}
    s_test = mean_score(test_mats["Baseline"])

    # ── 3. Baseline on the dev split ─────────────────────────────────────────
    base_dev = fill_baseline(scorer.score(prompts(dev), R, desc="dev baseline"))
    s_dev = mean_score(base_dev)

    # ── 4. Direct sensitivity: stated A vs. A = UNKNOWN ─────────────────────
    disp = {str(k): v for k, v in config["protected_display"].items()}
    stated_vals = dev.df[config["sensitive_col"]].astype(str).map(disp).tolist()
    stated = fill_modified(scorer.score(prompts(dev, insert_protected=stated_vals), R,
                                        desc="A stated"), s_dev)
    unknown = fill_modified(scorer.score(prompts(dev, insert_protected=UNKNOWN), R,
                                         desc="A=UNKNOWN"), s_dev)
    du_a = direct_sensitivity(stated, unknown)
    logger.info("DU_A=%.4f", du_a)

    # ── 5. Proxy capacity on the training split ──────────────────────────────
    pc_res = compute_proxy_capacity(train.df, feats, train.a)
    pair_mi = pairwise_feature_mi(train.df, feats)

    # ── 6. Proxy use on the dev split ────────────────────────────────────────
    ablated = {f: fill_modified(scorer.score(prompts(dev, mask_cols=[f]), R,
                                             desc=f"ablate {f}"), s_dev)
               for f in feats}
    pu = proxy_use(base_dev, ablated)

    # ── 7. Proxy risk, thresholds, graph ────────────────────────────────────
    pc = pc_res["pc"]
    prs = proxy_risk(pc, pu)
    tau1, tau2, tau3 = thresholds(pc, pu, pair_mi)
    G = build_proxy_graph(config["sensitive_col"], pc, pu, pair_mi, tau1, tau2, tau3)

    proxy_table = sorted(
        ({"feature": f, "kind": pc_res["kinds"][f], "PC": pc[f], "PU": pu[f],
          "PRS": prs[f], "role": role(pc[f], pu[f], tau1, tau2)} for f in feats),
        key=lambda r: -r["PRS"])

    # ── 8. Mitigation plan ───────────────────────────────────────────────────
    theta = theta_prs if theta_prs is not None else config.get("theta_prs")
    f_mask, theta = mask_set(prs, theta)
    base_dev_m = classification_metrics(s_dev, dev.y, dev.a)
    dev_filter = {}
    for cand in filter_candidates(f_mask):
        mat = fill_modified(scorer.score(prompts(dev, filtering(cand)), R,
                                         desc=f"filter {'+'.join(cand)}"), s_dev)
        dev_filter[cand] = classification_metrics(mean_score(mat), dev.y, dev.a)
    f_filt, filter_log = select_filter_set(dev_filter, base_dev_m["BA"], FILTER_BA_TOLERANCE)
    logger.info("F_mask=%s (θ=%.4f)  F_filt=%s", f_mask, theta, list(f_filt))

    # ── 9. Interventions on the test split ──────────────────────────────────
    balanced = sample_balanced_demos(train, FAIR_ICL_PER_CELL)
    for iv in (masking(f_mask), filtering(f_filt), fpo(), fair_icl(balanced)):
        if not (iv.mask_cols or iv.omit_cols or iv.system_prefix or iv.demos is not None):
            test_mats[iv.name] = test_mats["Baseline"]   # empty feature set
            continue
        test_mats[iv.name] = fill_modified(
            scorer.score(prompts(test, iv), R, desc=f"test {iv.name}"), s_test)

    mitigation = [{"Strategy": name, **metrics_over_runs(mat, test.y, test.a)}
                  for name, mat in test_mats.items()]

    # ── 10. Traditional baselines ───────────────────────────────────────────
    traditional = evaluate_traditional(splits, R)

    return {
        "meta": {
            "dataset": dataset, "model": model, "encoding": encoding, "k": k,
            "temperature": temperature, "repeats": R,
            "max_tokens": max_tokens, "seed": SEED, "mock": MOCK,
            "split_sizes": dict(zip(("train", "dev", "test"), splits.full_sizes)),
            "audited": {"dev": len(dev), "test": len(test)},
            "n_queries": scorer.n_queries, "fallback_rate": scorer.fallback_rate,
            "runtime_s": round(time.time() - t0, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "sensitivity": {"DU_A": du_a, "tau2": tau2, "ratio": du_a / tau2 if tau2 else np.nan},
        "thresholds": {"tau1": tau1, "tau2": tau2, "tau3": tau3},
        "proxy_table": proxy_table,
        "pair_mi": [{"f1": a, "f2": b, "mi": v} for (a, b), v in pair_mi.items()],
        "graph": graph_to_dict(G),
        "proxy_paths": get_proxy_paths(G, config["sensitive_col"]),
        "mitigation_plan": {"theta_prs": theta, "F_mask": f_mask, "F_filt": list(f_filt),
                            "filter_search": filter_log, "dev_baseline": base_dev_m,
                            "delta": FILTER_BA_TOLERANCE},
        "mitigation": mitigation,
        "traditional": traditional,
        "_arrays": {
            "dev_y": dev.y, "dev_a": dev.a, "test_y": test.y, "test_a": test.a,
            "dev_base": base_dev, "dev_A_stated": stated, "dev_A_unknown": unknown,
            **{f"dev_ablate::{f}": m for f, m in ablated.items()},
            **{f"test::{name}": m for name, m in test_mats.items()},
        },
    }
