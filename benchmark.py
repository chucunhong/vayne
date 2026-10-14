"""
Few-shot credit-classification benchmark (RQ1) and traditional baselines.

Every configuration (model × k × encoding × temperature) is scored on the
test split (and optionally the development split, used for selecting the
audit configuration) with the same demonstration set for every record, a
single query per prompt by default, and bootstrap SDs over applicants.

Results are merged into results/benchmark/{model}_{dataset}.csv, keyed by
(Split, Encoding, K, Temp), so sweeps can be run in several invocations.

Usage:
  # k sweep, all three models, T = 0.0 (Tables fullca/fullcc/fullgc)
  python benchmark.py
  # temperature sweep of the audit model at k = 128 (Table tempsweep);
  # the dev split is used to choose the audit temperature
  python benchmark.py --models anthropic/claude-3-haiku --k 128 \
      --temps 0 0.2 0.4 0.6 0.8 1.0 --splits dev test
  # traditional baselines only (no API calls)
  python benchmark.py --traditional
"""

import argparse
import logging
import re
import sys

import pandas as pd

from src.config import (
    BENCHMARK_DIR, BENCHMARK_K_VALUES, BENCHMARK_MODELS, BENCHMARK_TEMPERATURE,
    DATASET_ORDER, DATASET_CONFIGS, ENCODINGS, MAX_OUTPUT_TOKENS, N_BOOTSTRAP,
)
from src.data_loader import make_splits, sample_demos
from src.fairness_metrics import metrics_with_sd
from src.llm_client import MOCK
from src.scoring import Scorer, fill_baseline
from src.serializer import serialize
from src.traditional import evaluate_traditional

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger("benchmark")

KEY = ["Split", "Encoding", "K", "Temp"]


def parse_args():
    p = argparse.ArgumentParser(description="VAYNE few-shot benchmark")
    p.add_argument("--datasets", nargs="+", default=DATASET_ORDER, choices=list(DATASET_CONFIGS))
    p.add_argument("--models", nargs="+", default=BENCHMARK_MODELS)
    p.add_argument("--k", nargs="+", type=int, default=BENCHMARK_K_VALUES)
    p.add_argument("--encodings", nargs="+", default=ENCODINGS, choices=ENCODINGS)
    p.add_argument("--temps", nargs="+", type=float, default=[BENCHMARK_TEMPERATURE])
    p.add_argument("--splits", nargs="+", default=["test"], choices=["dev", "test"])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    p.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    p.add_argument("--traditional", action="store_true",
                   help="Only evaluate DT/NB/MLP/kNN (no API calls)")
    p.add_argument("--out", default=str(BENCHMARK_DIR))
    return p.parse_args()


def _merge_save(rows: list[dict], path) -> None:
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        if not set(KEY) <= set(old.columns):
            sys.exit(f"{path} is in the pre-split legacy format; move it aside before "
                     "writing new benchmark results to this directory")
        merged = pd.concat([old, new]).drop_duplicates(subset=KEY, keep="last")
    else:
        merged = new
    merged.sort_values(KEY).to_csv(path, index=False, float_format="%.6f")
    logger.info("Saved → %s (%d rows)", path, len(merged))


def main():
    args = parse_args()
    from pathlib import Path
    out = Path(args.out)
    if MOCK and not args.traditional and out.resolve() == BENCHMARK_DIR.resolve():
        sys.exit("VAYNE_MOCK=1 must not write to the real benchmark directory; pass --out")
    out.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        config = DATASET_CONFIGS[ds]
        splits = make_splits(config)

        trad = pd.DataFrame(evaluate_traditional(splits, args.n_boot))
        trad.insert(0, "Dataset", ds)
        trad.to_csv(out / f"traditional_{ds}.csv", index=False, float_format="%.6f")
        logger.info("%s traditional baselines:\n%s", ds,
                    trad[["Model", "BA", "Acc", "SP", "EOD", "PP"]].round(4).to_string(index=False))
        if args.traditional:
            continue

        for model in args.models:
            slug = re.sub(r"[^\w\-]", "_", model.split("/")[-1])
            rows = []
            for temp in args.temps:
                scorer = Scorer(model, temp, args.max_tokens)
                for k in args.k:
                    demos = sample_demos(splits.train, k)
                    for enc in args.encodings:
                        for split_name in args.splits:
                            split = getattr(splits, split_name)
                            prompts = [serialize(r, config, enc, demos)
                                       for _, r in split.df.iterrows()]
                            n_fail0 = scorer.n_failed
                            mat = scorer.score(prompts, args.repeats,
                                               desc=f"{slug} {ds} {split_name} {enc} k={k} T={temp}")
                            fails = scorer.n_failed - n_fail0
                            m = metrics_with_sd(fill_baseline(mat), split.y, split.a, args.n_boot)
                            rows.append({"Dataset": ds, "Model": slug, "Split": split_name,
                                         "Encoding": enc, "K": k, "Temp": temp,
                                         "N": len(split), "Repeats": args.repeats,
                                         "FallbackRate": fails / mat.size, **m})
                            logger.info("%s %s %s k=%d T=%.1f  BA=%.4f Acc=%.4f",
                                        slug, split_name, enc, k, temp, m["BA"], m["Acc"])
            _merge_save(rows, out / f"{slug}_{ds}.csv")


if __name__ == "__main__":
    main()
