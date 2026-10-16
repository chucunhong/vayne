"""
VAYNE — full audit (RQ2 diagnosis + RQ3 mitigation) under the audit
configuration of the paper: Claude-3-Haiku, Template encoding, k = 128,
temperature 0.4, R repeated queries per prompt.

Usage:
  python main.py                                  # all three datasets
  python main.py --datasets german_credit --repeats 5
  VAYNE_MOCK=1 python main.py --results-dir /tmp/vayne_smoke   # offline smoke test

Figures and LaTeX tables are produced afterwards from the saved files:
  python -m scripts.make_figures && python -m scripts.make_tables
"""

import argparse
import logging
import sys
from pathlib import Path

from src.config import (
    AUDIT_ENCODING, AUDIT_K, AUDIT_MODEL, AUDIT_TEMPERATURE, DATASET_CONFIGS,
    DATASET_ORDER, MAX_OUTPUT_TOKENS, N_REPEATS, RESULTS_DIR,
)
from src.llm_client import MOCK
from src.pipeline import run_audit
from src.reporting import print_results, save_results

logging.basicConfig(level=logging.INFO, stream=sys.stdout, datefmt="%H:%M:%S",
                    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")
logger = logging.getLogger("vayne.main")


def parse_args():
    p = argparse.ArgumentParser(description="VAYNE: proxy fairness audit of an LLM credit scorer")
    p.add_argument("--datasets", nargs="+", default=DATASET_ORDER, choices=list(DATASET_CONFIGS))
    p.add_argument("--model", default=AUDIT_MODEL)
    p.add_argument("--encoding", default=AUDIT_ENCODING, choices=["template", "markdown"])
    p.add_argument("--k", type=int, default=AUDIT_K)
    p.add_argument("--temperature", type=float, default=AUDIT_TEMPERATURE)
    p.add_argument("--repeats", type=int, default=N_REPEATS)
    p.add_argument("--max-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    p.add_argument("--theta-prs", type=float, default=None,
                   help="θ_PRS for every dataset (default: the per-dataset value in src/config.py)")
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    if MOCK and results_dir.resolve() == RESULTS_DIR.resolve():
        sys.exit("VAYNE_MOCK=1 must not write to the real results directory; pass --results-dir")

    for ds in args.datasets:
        logger.info("─" * 60)
        logger.info("Audit: %s  (%s, %s, k=%d, T=%.2f, R=%d)", ds, args.model, args.encoding,
                    args.k, args.temperature, args.repeats)
        result = run_audit(ds, args.model, args.encoding, args.k, args.temperature,
                           args.repeats, args.theta_prs, args.max_tokens)
        save_results(result, results_dir)
        print_results(result)


if __name__ == "__main__":
    main()
