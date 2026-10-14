# VAYNE

**Explaining and Mitigating Proxy Fairness in LLM-based Tabular Decisions**

VAYNE is a black-box framework that audits Large Language Models (LLMs) used as tabular classifiers (credit approval, loan scoring, etc.). It does two things existing fairness audits don't:

1. **Utility-qualified fairness** — it first checks whether an LLM is actually a competent few-shot tabular classifier before trusting any fairness conclusion drawn from it. A model that can't do the task doesn't produce a meaningful fairness signal.
2. **Proxy-aware explanation** — instead of only reporting *that* a fairness gap exists (via Statistical Parity, Equalized Odds, etc.), it explains *why*: whether the LLM is directly sensitive to the protected attribute, or whether unfairness is transmitted indirectly through non-protected features that act as demographic proxies (e.g. zipcode standing in for race, occupation standing in for gender).

This repository contains the experiment code behind the VAYNE paper.

![VAYNE framework overview](assets/fig_overview.png)
*LLM scoring → utility-qualified baseline fairness → Direct Unfairness via protected-attribute masking → Proxy Risk analysis (Proxy Capacity × Proxy Use → Proxy Path Graph) → mitigation.*

## How it works

An applicant record is `x = (a, x_1, ..., x_d)`. The deployed scorer sees only the deployment profile `z = x_{-A}` and returns `s(z) = p_yes`. Every prompt is queried `R` times and the analysis uses the average `s̄(z)`; decisions are `Ŷ = 1[s̄ ≥ 0.5]`.

| Quantity | Definition | Split |
|---|---|---|
| **SP / EOD / PP** | signed positive-rate gap; sum of TPR and FPR gaps; precision gap (n/a if undefined) | test |
| **NF** (noise floor) | `E_z abs(s̄⁽¹⁾(z) − s̄⁽²⁾(z))`, two independent averages of R queries | dev |
| **DU_A** (paired swap) | `E_z abs(s̄(z⁽¹⁾) − s̄(z⁽⁰⁾))`, an A field inserted and swapped | dev |
| **DU_A^abl** | `E abs(s̄(z⁽ᵃ⁾) − s̄(z⁽ᵘ⁾))`, own value vs. UNKNOWN | dev |
| **PC_j** | `I(X_j; A)` in nats: plug-in (categorical) or Ross k-NN, K=3 (continuous) | train |
| **PU_j** | `E_z abs(s̄(z) − s̄(z_{-j}))`, value of X_j replaced by UNKNOWN | dev |
| **PRS_j** | `PC_j × PU_j` | — |

Every value is reported with a bootstrap SD (B resamples, stratified by A and Y, resampling both applicants and their R stored scores).

The **Proxy Path Graph** has a capacity edge `A → X_j` when `PC_j > τ₁` and a use edge `X_j → D` when `PU_j > τ₂`; τ₁ and τ₂ are the median PC and PU. Undirected association links join features with `I(X_j; X_k) > τ₃` (median non-zero pairwise MI) and are descriptive only. Feature roles, used identically in every figure and table: **high risk** (both edges), **capacity only**, **use only**, **low**.

### Mitigation

| Strategy | What it does |
|---|---|
| Masking | values of `F_mask = {j : PRS_j > θ_PRS}` replaced by UNKNOWN |
| Filtering | fields of `F_filt ⊆ F_mask` omitted; `F_filt` is the subset with the lowest dev-split EOD whose BA is at most δ below the unmitigated scorer |
| FPO | one fixed fairness instruction prepended to the system message |
| Fair-ICL | the k = 128 demonstrations balanced over sex × label (32 per cell) |

θ_PRS is set per dataset in `src/config.py` (`theta_prs`); when unset it defaults to the 75th percentile of PRS.

## Protocol

- Splits: stratified 60/20/20 train/dev/test, seed 42. Credit Card audits 1,000 dev and 1,000 test records. Credit Approval drops the 12 records with a missing sex value.
- Demonstrations: k records from the training split (with replacement if k exceeds it), the same set for every record of a configuration.
- Audit configuration: Claude-3-Haiku, Template encoding, k = 128, temperature 0.4, 128 output tokens, R = 5.
- A reply without a valid `{"p_yes": x}` object is retried up to three times; a modified profile that still fails falls back to the applicant's baseline score.
- Every completed query is cached in `cache/queries.sqlite`, so an interrupted run resumes without re-issuing queries.

## Repository structure

```
main.py              VAYNE audit (RQ2 diagnosis + RQ3 mitigation) per dataset
benchmark.py         few-shot benchmark (RQ1): models × k × encoding × temperature,
                     plus the DT/NB/MLP/kNN baselines
scripts/
  make_figures.py    every paper figure from saved results (no LLM calls)
  make_tables.py     LaTeX rows for every result table + numbers.tex macros
src/
  config.py          protocol constants, datasets, audit configuration
  data_loader.py     loading, stratified splits, demonstrations
  serializer.py      Template / Markdown prompts, masking, omission, A insertion
  llm_client.py      OpenRouter client, JSON parsing, retries, query cache, mock mode
  scoring.py         repeated scoring into (n, R) matrices, fallbacks
  fairness_metrics.py  BA, Acc, SP, EOD, PP and the bootstrap
  proxy_capacity.py  PC_j and pairwise feature MI
  proxy_use.py       NF, DU_A, DU_A^abl, PU_j
  proxy_risk.py      PRS, thresholds, roles, F_mask
  proxy_graph.py     graph construction and rendering
  mitigation.py      strategies and filtering selection
  traditional.py     DT, NB, MLP, kNN baselines
  pipeline.py        Algorithm 1
  reporting.py       JSON / CSV / npz output
  plots.py           figure functions
```

## Setup

```bash
./setup.sh                      # creates .venv, installs requirements.txt
source .venv/bin/activate
cp .env.example .env            # then set OPENROUTER_API_KEY=... inside
```

## Usage

```bash
# RQ1: k sweep for three models at T = 0.0 (Tables fullca/fullcc/fullgc)
python benchmark.py
# RQ1: temperature sweep of the audit model at k = 128, dev split for selection
python benchmark.py --models anthropic/claude-3-haiku --k 128 \
    --temps 0 0.2 0.4 0.6 0.8 1.0 --splits dev test
# RQ2 + RQ3: the audit
python main.py
# figures and LaTeX tables from the saved results
python -m scripts.make_figures
python -m scripts.make_tables
```

Offline smoke test with a deterministic fake scorer (never writes to `results/`):

```bash
VAYNE_MOCK=1 python main.py --datasets credit_approval --repeats 2 --n-boot 20 --results-dir /tmp/vayne_smoke
```

⚠️ **Cost:** per dataset the audit issues about `R × (5·n_test + (5 + d)·n_dev)` queries plus `R · n_dev` per filtering candidate (≈ 300k queries in total at R = 5 with k = 128 demonstrations per prompt). Check the logged estimate before a full run.

## Output

`results/vayne/{ds}.json` (everything except raw scores), `{ds}_scores.npz` (every score matrix), `{ds}_proxy_risk.csv`, `{ds}_mitigation.csv`, `{ds}_sensitivity.csv`, `{ds}_traditional.csv`; `results/benchmark/{model}_{ds}.csv` and `traditional_{ds}.csv`; `results/figures/` and `results/tables/`.

`results/benchmark_ori/`, the `results/benchmark/*.csv` files without a `Split` column, and the older files in `results/vayne/` come from the previous protocol (no splits, single query, ±0.03 jitter in the benchmark) and are not used by the scripts above.

<!-- ## Citation

A formal citation will be added once the paper is published. In the meantime, cite the repository directly:

```bibtex
@misc{vayne2026,
  title  = {VAYNE: Explaining and Mitigating Proxy Fairness in LLM-based Tabular Decisions},
  author = {Giang Thi Thu, Huyen and Doan Viet, Thang and Nguyen Thanh, Trung and
            Ta Quoc, Tuan and Le Quy, Tai and Nguyen Long, Giang and Ban, Ha-Bang},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```
-->

