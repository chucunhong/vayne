"""
Transcribe the numbers reported in paper.tex into results/paper/ and draw
the figures those numbers determine.

Outputs:
  benchmark_{ds}.csv        Tables fullca / fullcc / fullgc (LLM + traditional rows)
  tempsweep.csv             Table tempsweep
  proxy_risk_top5.csv       Table proxyrisk
  mitigation.csv            Table mitigation
  text_values.json          in-text values (NF, DU_A, τ, θ_PRS, F_mask, F_filt)
  figures/fig_accuracy_vs_k.pdf, figures/fig_accuracy_vs_temperature.pdf,
  figures/fig_proxy_use_and_proxy_capacity.png,
  figures/fig_{ds}_proxy_bar_plot.png / _pc_pu_scatter.png / _proxy_graph.png
    (PC/PU of every feature from results/vayne/{ds}.json, the run behind
     Table proxyrisk; pairwise feature MI computed on the training split)

Usage:
  python -m scripts.paper_results [--paper paper.tex] [--out results/paper]
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src import plots
from src.config import BASE_DIR

DATASET_OF = {"tab:fullca": "credit_approval", "tab:fullcc": "credit_card",
              "tab:fullgc": "german_credit"}
TITLE_TO_DS = {"CreditApproval": "credit_approval", "CreditCard": "credit_card",
               "GermanCredit": "german_credit"}
MODEL_KEY = {"claude-3-haiku": "claude-3-haiku", "gpt-4o-mini": "gpt-4o-mini",
             "gpt-40-mini": "gpt-4o-mini", "deepseek-v4-flash": "deepseek-v4-flash-20260423"}
ENC = {"T": "template", "M": "markdown"}
NUM = r"-?\d+(?:\.\d+)?"


# ── LaTeX helpers ───────────────────────────────────────────────────────────

def _table_body(tex: str, label: str) -> list[tuple[int, str]]:
    """(line number, text) of the rows between \\midrule and \\bottomrule/\\botrule."""
    lines = tex.splitlines()
    start = next(i for i, l in enumerate(lines) if f"\\label{{{label}}}" in l)
    out, inside = [], False
    for i in range(start, len(lines)):
        l = lines[i]
        if "\\midrule" in l and not inside:
            inside = True
            continue
        if inside and ("\\bottomrule" in l or "\\botrule" in l):
            break
        if inside:
            out.append((i + 1, l))
    return out


def _cell(text: str) -> dict:
    """Parse one cell: value, sd, highlight."""
    sd_m = re.search(r"\\sd\{(?:\\ph\{)?(" + NUM + r")", text)
    head = text[:text.index("\\sd")] if "\\sd" in text else text
    val_m = re.search(NUM, head.replace("\\second", "").replace("\\best", ""))
    return {
        "value": float(val_m.group()) if val_m else None,
        "sd": float(sd_m.group(1)) if sd_m else None,
        "highlight": "best" if "\\best" in text else "second" if "\\second" in text else "",
        "underline": "\\underline" in text,
    }


def _split_cells(row: str) -> list[str]:
    row = row.split("\\\\")[0]
    return [c.strip() for c in row.split("&")]


def _flatten(prefix_cols: dict, metrics: list[str], cells: list[str], line: int) -> dict:
    rec = dict(prefix_cols)
    for m, c in zip(metrics, cells):
        p = _cell(c)
        rec[m] = p["value"]
        rec[f"{m}_sd"] = p["sd"]
        rec[f"{m}_highlight"] = p["highlight"] + ("+underline" if p["underline"] else "")
    rec["paper_line"] = line
    return rec


def _model_key(label: str) -> str:
    return MODEL_KEY[label.strip().lower()]


# ── Tables ──────────────────────────────────────────────────────────────────

def benchmark_tables(tex: str) -> dict[str, pd.DataFrame]:
    metrics = ["BA", "Acc", "SP", "EOD", "PP"]
    out = {}
    for label, ds in DATASET_OF.items():
        rows, model, model_label = [], None, None
        for ln, l in _table_body(tex, label):
            if "\\multirow" in l:
                model_label = re.search(r"exhyphenpenalty=0\s+([^}]+)\}", l).group(1).strip()
                model = _model_key(model_label)
                continue
            if "&" not in l:
                continue
            cells = _split_cells(l)
            if cells[0] in ("DT", "NB", "MLP", "kNN"):
                rows.append(_flatten({"Dataset": ds, "Model": cells[0], "ModelLabelInPaper": cells[0],
                                      "Encoding": "", "K": None, "Temp": None, "Split": "test"},
                                     metrics, cells[3:8], ln))
            else:
                rows.append(_flatten({"Dataset": ds, "Model": model, "ModelLabelInPaper": model_label,
                                      "Encoding": ENC[cells[1]], "K": int(cells[2]), "Temp": 0.0,
                                      "Split": "test"}, metrics, cells[3:8], ln))
        out[ds] = pd.DataFrame(rows)
    return out


def tempsweep_table(tex: str) -> pd.DataFrame:
    metrics = ["BA", "Acc", "SP", "EOD", "PP"]
    rows, ds = [], None
    for ln, l in _table_body(tex, "tab:tempsweep"):
        if "\\multirow" in l:
            title = re.search(r"\\makecell\[l\]\{([^}]*)\}", l).group(1)
            ds = TITLE_TO_DS[title.replace("\\\\", "").replace(" ", "")]
            continue
        if "&" not in l:
            continue
        cells = _split_cells(l)
        rows.append(_flatten({"Dataset": ds, "Model": "claude-3-haiku", "Encoding": ENC[cells[1]],
                              "K": 128, "Temp": float(cells[2]), "Split": "test"},
                             metrics, cells[3:8], ln))
    return pd.DataFrame(rows)


def _dataset_blocks(tex: str, label: str):
    ds = None
    for ln, l in _table_body(tex, label):
        if "\\multirow" in l:
            title = re.search(r"\\makecell\[l\]\{([^}]*)\}", l).group(1)
            ds = TITLE_TO_DS[title.replace("\\\\", "").replace(" ", "")]
            continue
        if "&" in l:
            yield ds, ln, _split_cells(l)


def proxy_table(tex: str) -> pd.DataFrame:
    rows = []
    for ds, ln, cells in _dataset_blocks(tex, "tabproxyrisk"):
        rec = _flatten({"Dataset": ds, "feature": cells[1].replace("\\_", "_")},
                       ["PC", "PU", "PRS"], cells[2:5], ln)
        rec["role_in_paper"] = cells[5]
        rows.append(rec)
    return pd.DataFrame(rows)


def mitigation_table(tex: str) -> pd.DataFrame:
    return pd.DataFrame([_flatten({"Dataset": ds, "Strategy": cells[1]},
                                  ["SP", "EOD", "PP", "Acc", "BA"], cells[2:7], ln)
                         for ds, ln, cells in _dataset_blocks(tex, "tabmitigation")])


# ── In-text values ──────────────────────────────────────────────────────────

def _find_line(tex: str, needle: str) -> tuple[int, str]:
    for i, l in enumerate(tex.splitlines(), 1):
        if needle in l:
            return i, l
    raise ValueError(f"'{needle}' not found in paper.tex")


def _numbers(segment: str) -> list[float]:
    """Plain decimal numbers in a text segment (an optional \\ph{} wrapper is ignored)."""
    return [float(v) for v in re.findall(r"(?<![\w.])(\d+\.\d+)", segment)]


def _feats(s: str) -> list[str]:
    s = s.replace("\\_", "_").replace("\\ph{", "").replace("}", "")
    return [f.strip() for f in re.split(r",|\band\b", s) if f.strip()]


def text_values(tex: str) -> dict:
    order = ["german_credit", "credit_card", "credit_approval"]   # order used in the text
    ln, l = _find_line(tex, "The paired-swap sensitivity is small")
    du = _numbers(l[l.index("$\\mathrm{DU}_A$ is"):l.index("against a noise floor")])
    nf = _numbers(l[l.index("against a noise floor"):l.index("ablation variant gives")])
    abl = _numbers(l[l.index("ablation variant gives"):])[:3]

    ln_tau, l_tau = _find_line(tex, "Roles are given relative to the median thresholds")
    taus = [float(v) for v in re.findall(r"\\tau_[12]=(" + NUM + r")", l_tau)]

    ln_t3, l_t3 = _find_line(tex, "is the median non-zero pairwise mutual information")
    tau3 = _numbers(l_t3[l_t3.index("which is"):])

    ln_m, l_m = _find_line(tex, "With $\\theta_{\\mathrm{PRS}}$ set to")
    theta = _numbers(l_m[l_m.index("set to"):l_m.index("$\\mathcal{F}_{\\mathrm{mask}}$")])
    mask_txt = l_m[l_m.index("contains") + len("contains"):l_m.index("Selecting")]
    fmask = [_feats(x) for x in re.split(r" on (?:German Credit|Credit Card|Credit Approval)[,.]?\s*", mask_txt)
             if x.strip().rstrip(".")]
    ffilt = [_feats(x) for x in re.findall(r"\\\{(.*?)\\\}", l_m)]

    out = {}
    for i, ds in enumerate(order):
        out[ds] = {"DU_A": du[i], "NF": nf[i], "DU_A_abl": abl[i],
                   "tau1": taus[2 * i], "tau2": taus[2 * i + 1], "tau3": tau3[i],
                   "theta_prs": theta[i], "F_mask": fmask[i], "F_filt": ffilt[i]}
    out["_source_lines"] = {"DU_A/NF/DU_A_abl": ln, "tau1/tau2": ln_tau, "tau3": ln_t3,
                            "theta_prs/F_mask/F_filt": ln_m}
    return out


# ── Proxy figures ───────────────────────────────────────────────────────────

def proxy_figures(vayne_dir: Path, fig_dir: Path, tau3_paper: dict) -> None:
    from src.config import DATASET_CONFIGS
    from src.data_loader import make_splits
    from src.proxy_capacity import pairwise_feature_mi
    from src.proxy_graph import build_proxy_graph, graph_to_dict
    from src.proxy_risk import role

    for ds in ["german_credit", "credit_card", "credit_approval"]:
        old = json.loads((vayne_dir / f"{ds}.json").read_text())
        pc, pu = old["proxy_capacity"], old["proxy_use"]
        tau1, tau2 = old["tau_capacity"], old["tau_use"]
        tau3 = tau3_paper[ds]["tau3"]
        splits = make_splits(DATASET_CONFIGS[ds])
        pair_mi = pairwise_feature_mi(splits.train.df, [f for f in pu if f in splits.feature_cols])
        G = build_proxy_graph(DATASET_CONFIGS[ds]["sensitive_col"], pc, pu, pair_mi,
                              tau1, tau2, tau3)
        res = {
            "meta": {"dataset": ds},
            "thresholds": {"tau1": tau1, "tau2": tau2, "tau3": tau3},
            "proxy_table": sorted(({"feature": f, "PC": pc[f], "PU": pu[f], "PU_sd": None,
                                    "PRS": pc[f] * pu[f], "role": role(pc[f], pu[f], tau1, tau2)}
                                   for f in pu), key=lambda r: -r["PRS"]),
            "graph": graph_to_dict(G),
        }
        plots.plot_proxy_use_bar(res, fig_dir / f"fig_{ds}_proxy_bar_plot.png")
        plots.plot_pc_pu_scatter(res, fig_dir / f"fig_{ds}_pc_pu_scatter.png")
        plots.plot_graph(res, fig_dir / f"fig_{ds}_proxy_graph.png")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paper", default=str(BASE_DIR / "paper.tex"))
    p.add_argument("--out", default=str(BASE_DIR / "results" / "paper"))
    args = p.parse_args()
    tex = Path(args.paper).read_text()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bench = benchmark_tables(tex)
    for ds, df in bench.items():
        df.to_csv(out / f"benchmark_{ds}.csv", index=False)
    temp = tempsweep_table(tex)
    temp.to_csv(out / "tempsweep.csv", index=False)
    proxy_table(tex).to_csv(out / "proxy_risk_top5.csv", index=False)
    mitigation_table(tex).to_csv(out / "mitigation.csv", index=False)
    tv = text_values(tex)
    (out / "text_values.json").write_text(json.dumps(tv, indent=2))

    all_bench = pd.concat(bench.values())
    llm = all_bench[all_bench["Encoding"] != ""]
    trad = all_bench[all_bench["Encoding"] == ""]
    datasets = ["credit_approval", "credit_card", "german_credit"]
    plots.plot_accuracy_vs_k(llm, trad, out / "figures" / "fig_accuracy_vs_k.pdf", datasets)
    plots.plot_accuracy_vs_temperature(temp, out / "figures" / "fig_accuracy_vs_temperature.pdf",
                                       datasets)
    plots.plot_pc_pu_concept(out / "figures" / "fig_proxy_use_and_proxy_capacity.png")
    proxy_figures(BASE_DIR / "results" / "vayne", out / "figures", tv)
    for f in sorted(out.rglob("*.*")):
        print(f.relative_to(out))


if __name__ == "__main__":
    main()
