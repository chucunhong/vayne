"""
Save and print the output of src.pipeline.run_audit.

Per dataset, in the results directory:
  {ds}.json                 — everything except raw scores
  {ds}_scores.npz           — every (n, R) score matrix, labels, and groups
  {ds}_proxy_risk.csv       — feature × PC, PU, PRS (with SDs) × role
  {ds}_mitigation.csv       — strategy × SP, EOD, PP, Acc, BA (with SDs)
  {ds}_sensitivity.csv      — NF, DU_A, DU_A^abl (with SDs)
  {ds}_traditional.csv      — DT, NB, MLP, kNN on the same test records
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.config import RESULTS_DIR

logger = logging.getLogger(__name__)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_results(result: dict, results_dir: Path = RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    ds = result["meta"]["dataset"]

    np.savez_compressed(results_dir / f"{ds}_scores.npz", **result["_arrays"])
    summary = {k: v for k, v in result.items() if k != "_arrays"}
    (results_dir / f"{ds}.json").write_text(json.dumps(_jsonable(summary), indent=2))

    pd.DataFrame(result["proxy_table"]).to_csv(results_dir / f"{ds}_proxy_risk.csv", index=False)
    pd.DataFrame(result["mitigation"]).to_csv(results_dir / f"{ds}_mitigation.csv", index=False)
    pd.DataFrame(result["traditional"]).to_csv(results_dir / f"{ds}_traditional.csv", index=False)
    pd.DataFrame([result["sensitivity"]]).to_csv(results_dir / f"{ds}_sensitivity.csv", index=False)
    logger.info("Results saved → %s/%s.*", results_dir, ds)


def load_result(results_dir: Path, dataset: str) -> dict:
    res = json.loads((results_dir / f"{dataset}.json").read_text())
    npz = results_dir / f"{dataset}_scores.npz"
    if npz.exists():
        with np.load(npz) as z:
            res["_arrays"] = {k: z[k] for k in z.files}
    return res


def _fmt(v, sd=None, nd=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    s = f"{v:.{nd}f}"
    return f"{s} ± {sd:.{nd}f}" if sd is not None and not np.isnan(sd) else s


def print_results(result: dict) -> None:
    m = result["meta"]
    print(f"\n{'=' * 78}\n{m['dataset']}  |  {m['model']}  {m['encoding']}  k={m['k']}  "
          f"T={m['temperature']}  R={m['repeats']}{'  [MOCK]' if m['mock'] else ''}")
    print(f"queries={m['n_queries']}  fallback={m['fallback_rate']:.2%}  "
          f"utility={result['utility']['status']}")
    s = result["sensitivity"]
    print("NF=%s  DU_A=%s  DU_A_abl=%s" % (_fmt(s["NF"], s["NF_sd"]),
                                          _fmt(s["DU_A"], s["DU_A_sd"]),
                                          _fmt(s["DU_A_abl"], s["DU_A_abl_sd"])))
    t = result["thresholds"]
    print(f"τ1={t['tau1']:.4f}  τ2={t['tau2']:.4f}  τ3={t['tau3']:.4f}")
    print(tabulate([[r["feature"], _fmt(r["PC"], r["PC_sd"], 3), _fmt(r["PU"], r["PU_sd"], 3),
                     _fmt(r["PRS"], r["PRS_sd"]), r["role"]] for r in result["proxy_table"][:10]],
                   headers=["feature", "PC", "PU", "PRS", "role"], tablefmt="simple"))
    plan = result["mitigation_plan"]
    print(f"F_mask={plan['F_mask']} (θ={plan['theta_prs']:.4f})  F_filt={plan['F_filt']}")
    print(tabulate([[r["Strategy"]] + [_fmt(r[k], r[f"{k}_sd"]) for k in ("SP", "EOD", "PP", "Acc", "BA")]
                    for r in result["mitigation"] + [{"Strategy": x["Model"], **x}
                                                     for x in result["traditional"]]],
                   headers=["Strategy", "SP", "EOD", "PP", "Acc", "BA"], tablefmt="simple"))
