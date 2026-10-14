"""
VAYNE pipeline (Algorithm 1) for one dataset under one audit configuration.

  1. Split D into train/dev/test; draw the k demonstrations from train
  2. Test split: repeated scoring of the deployment profile → baseline metrics
  3. Dev split: 2R queries of the deployment profile → s̄ and noise floor NF
  4. Dev split: paired swap z⁽¹⁾/z⁽⁰⁾ and UNKNOWN z⁽ᵘ⁾ → DU_A, DU_A^abl
  5. Train split: proxy capacity PC_j and pairwise feature MI
  6. Dev split: single-feature ablation → proxy use PU_j
  7. PRS_j, thresholds τ₁ τ₂ τ₃, proxy path graph
  8. F_mask from θ_PRS; F_filt selected on dev
  9. Test split: Masking, Filtering, FPO, Fair-ICL
 10. Utility qualification against the traditional baselines

Every score matrix is returned so that statistics can be recomputed offline.
"""

import logging
import time

import numpy as np

from src.config import (
    AUDIT_ENCODING, AUDIT_K, AUDIT_MODEL, AUDIT_TEMPERATURE, DATASET_CONFIGS,
    FAIR_ICL_PER_CELL, FILTER_BA_TOLERANCE, MAX_OUTPUT_TOKENS, MITIGATE_DEMOS,
    N_BOOTSTRAP, N_REPEATS, SEED, UTILITY_TOLERANCE,
)
from src.data_loader import Split, make_splits, sample_balanced_demos, sample_demos
from src.fairness_metrics import classification_metrics, metrics_with_sd
from src.llm_client import MOCK
from src.mitigation import (
    Intervention, fair_icl, filter_candidates, filtering, fpo, masking, select_filter_set,
)
from src.proxy_capacity import compute_proxy_capacity, pairwise_feature_mi
from src.proxy_graph import build_proxy_graph, get_proxy_paths, graph_to_dict
from src.proxy_risk import mask_set, proxy_risk, proxy_risk_sd, role, thresholds
from src.proxy_use import proxy_use, sensitivity
from src.scoring import Scorer, fill_baseline, fill_modified, mean_score
from src.serializer import UNKNOWN, serialize
from src.traditional import evaluate_traditional

logger = logging.getLogger(__name__)


def utility_status(ba: float, ba_sd: float, best_traditional_ba: float,
                   tol: float = UTILITY_TOLERANCE) -> str:
    """'qualified', 'caveat', or 'exploratory' (Sect. 4.1, utility criterion)."""
    above_chance = ba - 0.5 > 1.96 * ba_sd
    if not above_chance:
        return "exploratory"
    return "qualified" if ba >= best_traditional_ba - tol else "caveat"


class _Prompter:
    """Builds the prompts of one configuration for a split."""

    def __init__(self, config: dict, encoding: str, demos: list):
        self.config, self.encoding, self.demos = config, encoding, demos

    def __call__(self, split: Split, iv: Intervention | None = None,
                 mask_cols=(), insert_protected=None) -> list[tuple[str, str]]:
        iv = iv or Intervention("Baseline")
        demos = iv.demos if iv.demos is not None else self.demos
        mask = list(mask_cols) + iv.mask_cols
        demo_mask = iv.mask_cols if MITIGATE_DEMOS else ()
        demo_omit = iv.omit_cols if MITIGATE_DEMOS else ()
        return [serialize(row, self.config, self.encoding, demos,
                          mask_cols=mask, omit_cols=iv.omit_cols,
                          insert_protected=insert_protected,
                          system_prefix=iv.system_prefix,
                          demo_mask_cols=demo_mask, demo_omit_cols=demo_omit)
                for _, row in split.df.iterrows()]


def run_audit(
    dataset: str,
    model: str = AUDIT_MODEL,
    encoding: str = AUDIT_ENCODING,
    k: int = AUDIT_K,
    temperature: float = AUDIT_TEMPERATURE,
    repeats: int = N_REPEATS,
    n_boot: int = N_BOOTSTRAP,
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

    n_q = (len(test) * R * 5 + len(dev) * R * (2 + 3 + len(feats)))
    logger.info("%s: ≈%d queries before filtering selection (R=%d, %d features)",
                dataset, n_q, R, len(feats))

    # ── 2. Baseline on the test split ───────────────────────────────────────
    test_mats = {"Baseline": fill_baseline(scorer.score(prompts(test), R, desc="test baseline"))}
    s_test = mean_score(test_mats["Baseline"])

    # ── 3. Dev baseline (2R queries) and noise floor ─────────────────────────
    dev_all = scorer.score(prompts(dev), 2 * R, desc="dev baseline")
    base1, base2 = fill_baseline(dev_all[:, :R]), fill_baseline(dev_all[:, R:])
    s_dev = mean_score(base1)

    # ── 4. Controlled sensitivity: paired swap and ablation variant ─────────
    disp = config["protected_display"]
    prot_key = next(v for v in disp if str(v) == str(config["protected_value"]))
    non_key = next(v for v in disp if v != prot_key)
    z1 = fill_modified(scorer.score(prompts(dev, insert_protected=disp[non_key]), R,
                                    desc="swap a=1"), s_dev)
    z0 = fill_modified(scorer.score(prompts(dev, insert_protected=disp[prot_key]), R,
                                    desc="swap a=0"), s_dev)
    zu = fill_modified(scorer.score(prompts(dev, insert_protected=UNKNOWN), R,
                                    desc="A=UNKNOWN"), s_dev)
    sens = sensitivity(base1, base2, z1, z0, zu, dev.y, dev.a, n_boot)
    logger.info("NF=%.4f  DU_A=%.4f  DU_A_abl=%.4f", sens["NF"], sens["DU_A"], sens["DU_A_abl"])

    # ── 5. Proxy capacity on the training split ──────────────────────────────
    pc_res = compute_proxy_capacity(train.df, feats, train.a, train.y, n_boot)
    pair_mi = pairwise_feature_mi(train.df, feats)

    # ── 6. Proxy use on the dev split ────────────────────────────────────────
    ablated = {f: fill_modified(scorer.score(prompts(dev, mask_cols=[f]), R,
                                             desc=f"ablate {f}"), s_dev)
               for f in feats}
    pu_res = proxy_use(base1, ablated, dev.y, dev.a, n_boot)

    # ── 7. Proxy risk, thresholds, graph ────────────────────────────────────
    pc, pu = pc_res["pc"], pu_res["pu"]
    prs = proxy_risk(pc, pu)
    prs_sd = proxy_risk_sd(pc_res["boot"], pu_res["boot"], feats)
    tau1, tau2, tau3 = thresholds(pc, pu, pair_mi)
    G = build_proxy_graph(config["sensitive_col"], pc, pu, pair_mi, tau1, tau2, tau3)

    proxy_table = sorted(
        ({"feature": f, "kind": pc_res["kinds"][f],
          "PC": pc[f], "PC_sd": pc_res["pc_sd"][f],
          "PU": pu[f], "PU_sd": pu_res["pu_sd"][f],
          "PRS": prs[f], "PRS_sd": prs_sd[f],
          "role": role(pc[f], pu[f], tau1, tau2)} for f in feats),
        key=lambda r: -r["PRS"])

    # ── 8. Mitigation plan ───────────────────────────────────────────────────
    theta = theta_prs if theta_prs is not None else config.get("theta_prs")
    f_mask, theta = mask_set(prs, theta)
    base_dev = classification_metrics(s_dev, dev.y, dev.a)
    dev_filter = {}
    for cand in filter_candidates(f_mask):
        mat = fill_modified(scorer.score(prompts(dev, filtering(cand)), R,
                                         desc=f"filter {'+'.join(cand)}"), s_dev)
        dev_filter[cand] = classification_metrics(mean_score(mat), dev.y, dev.a)
    f_filt, filter_log = select_filter_set(dev_filter, base_dev["BA"], FILTER_BA_TOLERANCE)
    logger.info("F_mask=%s (θ=%.4f)  F_filt=%s", f_mask, theta, list(f_filt))

    # ── 9. Interventions on the test split ──────────────────────────────────
    balanced = sample_balanced_demos(train, FAIR_ICL_PER_CELL)
    for iv in (masking(f_mask), filtering(f_filt), fpo(), fair_icl(balanced)):
        if not (iv.mask_cols or iv.omit_cols or iv.system_prefix or iv.demos is not None):
            test_mats[iv.name] = test_mats["Baseline"]   # empty feature set
            continue
        test_mats[iv.name] = fill_modified(
            scorer.score(prompts(test, iv), R, desc=f"test {iv.name}"), s_test)

    mitigation = [{"Strategy": name, **metrics_with_sd(mat, test.y, test.a, n_boot)}
                  for name, mat in test_mats.items()]

    # ── 10. Utility qualification ───────────────────────────────────────────
    traditional = evaluate_traditional(splits, n_boot)
    best_trad = max(traditional, key=lambda r: r["BA"])
    base = mitigation[0]
    status = utility_status(base["BA"], base["BA_sd"], best_trad["BA"])

    return {
        "meta": {
            "dataset": dataset, "model": model, "encoding": encoding, "k": k,
            "temperature": temperature, "repeats": R, "n_boot": n_boot,
            "max_tokens": max_tokens, "seed": SEED, "mock": MOCK,
            "split_sizes": dict(zip(("train", "dev", "test"), splits.full_sizes)),
            "audited": {"dev": len(dev), "test": len(test)},
            "n_queries": scorer.n_queries, "fallback_rate": scorer.fallback_rate,
            "runtime_s": round(time.time() - t0, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "sensitivity": sens,
        "thresholds": {"tau1": tau1, "tau2": tau2, "tau3": tau3},
        "proxy_table": proxy_table,
        "pair_mi": [{"f1": a, "f2": b, "mi": v} for (a, b), v in pair_mi.items()],
        "graph": graph_to_dict(G),
        "proxy_paths": get_proxy_paths(G, config["sensitive_col"]),
        "mitigation_plan": {"theta_prs": theta, "F_mask": f_mask, "F_filt": list(f_filt),
                            "filter_search": filter_log, "dev_baseline": base_dev,
                            "delta": FILTER_BA_TOLERANCE},
        "mitigation": mitigation,
        "traditional": traditional,
        "utility": {"status": status, "best_traditional": best_trad["Model"],
                    "best_traditional_BA": best_trad["BA"]},
        "_arrays": {
            "dev_y": dev.y, "dev_a": dev.a, "test_y": test.y, "test_a": test.a,
            "dev_base1": base1, "dev_base2": base2, "dev_z1": z1, "dev_z0": z0, "dev_zu": zu,
            **{f"dev_ablate::{f}": m for f, m in ablated.items()},
            **{f"test::{name}": m for name, m in test_mats.items()},
        },
    }
