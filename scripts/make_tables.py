"""
Write the paper's result tables as LaTeX row blocks from saved results, so
every number in paper.tex can be pasted from a file the code produced.

Outputs in results/tables/:
  tab_full_{ds}.tex     — benchmark rows (Tables fullca / fullcc / fullgc)
  tab_tempsweep.tex     — temperature sweep (Table tempsweep)
  tab_sensitivity.tex   — DU_A against the median PU (Table sensitivity)
  tab_proxyrisk.tex     — top-5 proxies (Table proxyrisk)
  tab_mitigation.tex    — mitigation (Table mitigation)
  numbers.tex           — \\newcommand macros for in-text values (DU_A, τ, θ_PRS, ...)

Highlighting follows the paper: \\best / \\second per column (max for BA and
Acc, closest to zero for SP, EOD, PP), \\underline for the best within an LLM.

Usage:
  python -m scripts.make_tables
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    AUDIT_K, AUDIT_MODEL, BENCHMARK_DIR, BENCHMARK_TEMPERATURE, DATASET_CONFIGS,
    DATASET_ORDER, RESULTS_DIR, TABLES_DIR,
)
from src.mitigation import STRATEGIES
from src.plots import MODEL_STYLE

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("make_tables")

COLS = ["BA", "Acc", "SP", "EOD", "PP"]
HIGHER = {"BA", "Acc"}
ENC = {"template": "T", "markdown": "M"}


def _cell(v, sd, nd=4):
    if v is None or pd.isna(v):
        return "n/a"
    s = f"{v:.{nd}f}"
    return s + (f"\\sd{{{sd:.{nd}f}}}" if sd is not None and not pd.isna(sd) else "")


def _rank_key(col, v):
    if pd.isna(v):
        return np.inf
    return -v if col in HIGHER else abs(v)


def _marks(df: pd.DataFrame, group_col: str | None = None) -> dict:
    """(row index, col) → set of marks {'best','second','under'}."""
    marks = {}
    for col in COLS:
        keys = df[col].map(lambda v: _rank_key(col, v))
        uniq = sorted(set(k for k in keys if np.isfinite(k)))
        for i, k in keys.items():
            if uniq and k == uniq[0]:
                marks.setdefault((i, col), set()).add("best")
            elif len(uniq) > 1 and k == uniq[1]:
                marks.setdefault((i, col), set()).add("second")
        if group_col:
            for _, g in df[df[group_col] != ""].groupby(group_col):
                gk = g[col].map(lambda v: _rank_key(col, v))
                if np.isfinite(gk.min()):
                    for i in gk[gk == gk.min()].index:
                        marks.setdefault((i, col), set()).add("under")
    return marks


MITIGATION_COLS = ["SP", "EOD", "PP", "Acc", "BA"]


def _fmt_row(r, i, marks, cols=COLS):
    out = []
    for c in cols:
        s = _cell(r[c], r.get(f"{c}_sd"))
        m = marks.get((i, c), set())
        if "under" in m:
            s = f"\\underline{{{s}}}"
        if "best" in m:
            s = f"\\best{{{s}}}"
        elif "second" in m:
            s = f"\\second{{{s}}}"
        out.append(s)
    return out


def full_tables(bench: pd.DataFrame, trad: pd.DataFrame, out: Path):
    for ds in DATASET_ORDER:
        b = bench[(bench["Dataset"] == ds) & (bench["Split"] == "test")
                  & (bench["Temp"].round(3) == round(BENCHMARK_TEMPERATURE, 3))]
        t = trad[trad["Dataset"] == ds]
        if b.empty and t.empty:
            continue
        rows = [dict(r, Group="", Label=r["Model"], P="--", K="--") for _, r in t.iterrows()]
        for model, (label, _, _) in MODEL_STYLE.items():
            m = b[b["Model"] == model].sort_values(["K", "Encoding"], ascending=[True, False])
            rows += [dict(r, Group=label, Label=label, P=ENC[r["Encoding"]], K=int(r["K"]))
                     for _, r in m.iterrows()]
        df = pd.DataFrame(rows).reset_index(drop=True)
        marks = _marks(df, "Group")
        lines, prev = [], "__"
        for i, r in df.iterrows():
            if r["Group"] != prev:
                if prev != "__":
                    lines.append("\\midrule")
                if r["Group"]:
                    n = int((df["Group"] == r["Group"]).sum())
                    lines.append(f"\\multirow{{{n}}}{{1.6cm}}{{\\centering\\arraybackslash"
                                 f"\\exhyphenpenalty=0 {r['Group']}}}")
                prev = r["Group"]
            first = "" if r["Group"] else r["Label"]
            lines.append(f"{first} & {r['P']} & {r['K']} & " + " & ".join(_fmt_row(r, i, marks))
                         + " \\\\")
        (out / f"tab_full_{ds}.tex").write_text("\n".join(lines) + "\n")
        logger.info("tab_full_%s.tex", ds)


def tempsweep_table(bench: pd.DataFrame, out: Path):
    audit = AUDIT_MODEL.split("/")[-1]
    b = bench[(bench["Model"] == audit) & (bench["K"] == AUDIT_K) & (bench["Split"] == "test")]
    if b["Temp"].nunique() < 2:
        return
    lines = []
    for ds in DATASET_ORDER:
        d = b[b["Dataset"] == ds].sort_values(["Encoding", "Temp"], ascending=[False, True])
        if d.empty:
            continue
        d = d.reset_index(drop=True)
        marks = _marks(d)
        title = DATASET_CONFIGS[ds]["title"].replace(" ", "\\\\")
        lines.append(f"\\multirow{{{len(d)}}}{{*}}{{\\makecell[l]{{{title}}}}}")
        for i, r in d.iterrows():
            lines.append(f"    & {ENC[r['Encoding']]} & {r['Temp']:.1f} & "
                         + " & ".join(_fmt_row(r, i, marks)) + " \\\\")
        lines.append("\\midrule")
    (out / "tab_tempsweep.tex").write_text("\n".join(lines[:-1]) + "\n")
    logger.info("tab_tempsweep.tex")


def _esc(s: str) -> str:
    return s.replace("_", "\\_")


def vayne_tables(results: dict, out: Path):
    from src.proxy_risk import ROLE_TEXT
    proxy, mitig, sens, macros = [], [], [], []
    for ds, res in results.items():
        title = DATASET_CONFIGS[ds]["title"].replace(" ", "\\\\")
        top = res["proxy_table"][:5]
        proxy.append(f"\\multirow{{5}}{{*}}{{\\makecell[l]{{{title}}}}}")
        for r in top:
            proxy.append(f"  & {_esc(r['feature'])} & {_cell(r['PC'], None, 3)} & "
                         f"{_cell(r['PU'], None, 3)} & {_cell(r['PRS'], None)} & "
                         f"{ROLE_TEXT[r['role']]} \\\\")
        proxy.append("\\midrule")

        df = pd.DataFrame(res["mitigation"])
        df["SP"] = df["SP"].abs()
        df = df.set_index("Strategy").reindex(STRATEGIES).reset_index()
        marks = _marks(df)
        mitig.append(f"\\multirow{{{len(df)}}}{{*}}{{\\makecell[l]{{{title}}}}}")
        for i, r in df.iterrows():
            mitig.append(f"   & {r['Strategy']} & " + " & ".join(_fmt_row(r, i, marks, MITIGATION_COLS)) + " \\\\")
        mitig.append("\\midrule")

        tag = "".join(w[0] for w in ds.split("_")).upper()   # CA, CC, GC
        s, t, plan, meta = res["sensitivity"], res["thresholds"], res["mitigation_plan"], res["meta"]
        sens.append(f"{DATASET_CONFIGS[ds]['title']} & {s['DU_A']:.3f} & {s['tau2']:.3f} & "
                    f"{s['ratio']:.2f} \\\\")
        macros.append(f"\\newcommand{{\\DUA{tag}}}{{{s['DU_A']:.3f}}}")
        macros += [f"\\newcommand{{\\tauone{tag}}}{{{t['tau1']:.3f}}}",
                   f"\\newcommand{{\\tautwo{tag}}}{{{t['tau2']:.3f}}}",
                   f"\\newcommand{{\\tauthree{tag}}}{{{t['tau3']:.3f}}}",
                   f"\\newcommand{{\\thetaPRS{tag}}}{{{plan['theta_prs']:.3f}}}",
                   f"\\newcommand{{\\Fmask{tag}}}{{{_esc(', '.join(plan['F_mask']))}}}",
                   f"\\newcommand{{\\Ffilt{tag}}}{{{_esc(', '.join(plan['F_filt']))}}}",
                   f"\\newcommand{{\\fallback{tag}}}{{{100 * meta['fallback_rate']:.2f}\\%}}",
                   f"% {ds}: {meta['model']} {meta['encoding']} k={meta['k']} "
                   f"T={meta['temperature']} R={meta['repeats']} "
                   f"queries={meta['n_queries']} mock={meta['mock']}"]
    (out / "tab_sensitivity.tex").write_text("\n".join(sens) + "\n")
    (out / "tab_proxyrisk.tex").write_text("\n".join(proxy[:-1]) + "\n")
    (out / "tab_mitigation.tex").write_text("\n".join(mitig[:-1]) + "\n")
    (out / "numbers.tex").write_text("\n".join(macros) + "\n")
    logger.info("tab_sensitivity.tex, tab_proxyrisk.tex, tab_mitigation.tex, numbers.tex")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench-dir", default=str(BENCHMARK_DIR))
    p.add_argument("--vayne-dir", default=str(RESULTS_DIR))
    p.add_argument("--out", default=str(TABLES_DIR))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from scripts.make_figures import load_benchmark
    bench, trad = load_benchmark(Path(args.bench_dir))
    if not bench.empty or not trad.empty:
        if bench.empty:
            bench = pd.DataFrame(columns=["Dataset", "Split", "Temp", "Model", "K", "Encoding"])
        full_tables(bench, trad, out)
        tempsweep_table(bench, out)

    results = {}
    for ds in DATASET_ORDER:
        path = Path(args.vayne_dir) / f"{ds}.json"
        if path.exists() and "proxy_table" in (res := json.loads(path.read_text())):
            results[ds] = res
    if results:
        vayne_tables(results, out)


if __name__ == "__main__":
    main()
