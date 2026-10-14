"""
Regenerate every paper figure from saved results — no LLM calls.

Inputs (whichever exist):
  results/benchmark/{model}_{ds}.csv, traditional_{ds}.csv   (benchmark.py)
  results/vayne/{ds}.json                                      (main.py)
Output: results/figures/

Usage:
  python -m scripts.make_figures
  python -m scripts.make_figures --vayne-dir /tmp/vayne_smoke --out /tmp/figs
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import (
    AUDIT_K, AUDIT_MODEL, BENCHMARK_DIR, BENCHMARK_TEMPERATURE, DATASET_ORDER,
    FIGURES_DIR, RESULTS_DIR,
)
from src import plots

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("make_figures")


def load_benchmark(bench_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    llm = [pd.read_csv(p) for p in bench_dir.glob("*.csv")
           if not p.name.startswith("traditional_") and "Split" in pd.read_csv(p, nrows=0).columns]
    trad = [pd.read_csv(p) for p in bench_dir.glob("traditional_*.csv")]
    return (pd.concat(llm) if llm else pd.DataFrame(),
            pd.concat(trad) if trad else pd.DataFrame())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench-dir", default=str(BENCHMARK_DIR))
    p.add_argument("--vayne-dir", default=str(RESULTS_DIR))
    p.add_argument("--out", default=str(FIGURES_DIR))
    args = p.parse_args()
    out = Path(args.out)

    plots.plot_pc_pu_concept(out / "fig_proxy_use_and_proxy_capacity.png")

    bench, trad = load_benchmark(Path(args.bench_dir))
    if bench.empty:
        logger.warning("No benchmark CSVs in the new format — skipping RQ1 figures")
    else:
        test = bench[bench["Split"] == "test"]
        ksweep = test[test["Temp"].round(3) == round(BENCHMARK_TEMPERATURE, 3)]
        ds = [d for d in DATASET_ORDER if d in set(ksweep["Dataset"])]
        if ds:
            plots.plot_accuracy_vs_k(ksweep, trad, out / "fig_accuracy_vs_k.pdf", ds)
        audit = AUDIT_MODEL.split("/")[-1]
        tsweep = test[(test["Model"] == audit) & (test["K"] == AUDIT_K)]
        ds = [d for d in DATASET_ORDER if d in set(tsweep["Dataset"])]
        if ds and tsweep["Temp"].nunique() > 1:
            plots.plot_accuracy_vs_temperature(tsweep, out / "fig_accuracy_vs_temperature.pdf", ds)
        else:
            logger.warning("No temperature sweep for %s, k=%d — skipping", audit, AUDIT_K)

    for ds in DATASET_ORDER:
        path = Path(args.vayne_dir) / f"{ds}.json"
        if not path.exists() or "proxy_table" not in (res := json.loads(path.read_text())):
            logger.warning("No VAYNE result in the new format for %s — skipping", ds)
            continue
        plots.plot_proxy_use_bar(res, out / f"fig_{ds}_proxy_bar_plot.png")
        plots.plot_pc_pu_scatter(res, out / f"fig_{ds}_pc_pu_scatter.png")
        plots.plot_graph(res, out / f"fig_{ds}_proxy_graph.png")


if __name__ == "__main__":
    main()
