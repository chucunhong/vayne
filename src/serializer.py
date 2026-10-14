"""
Serialize an applicant profile into a (system, user) prompt pair.

Two encodings carry the same attribute-value information (Sect. 4.1,
"Prompt serialization"):
  template — one natural-language sentence
  markdown — a two-column Markdown table

The protected attribute and the target are never serialized, except when the
controlled audit inserts an A field (`insert_protected`). Column names become
labels by replacing hyphens/underscores with spaces and applying title case.
"""

import pandas as pd

from src.config import SYSTEM_PROMPT

UNKNOWN = "UNKNOWN"


def col_label(col: str) -> str:
    return col.replace("-", " ").replace("_", " ").title()


def _profile_pairs(
    row: pd.Series,
    config: dict,
    mask_cols=(),
    omit_cols=(),
    insert_protected: str | None = None,
) -> list[tuple[str, object]]:
    """
    Attribute-value pairs of the profile, in column order.

    mask_cols        : values replaced by UNKNOWN (ablation, masking)
    omit_cols        : fields left out entirely (filtering)
    insert_protected : if given, an A field is kept at its column position with
                       this value (a display value or UNKNOWN)
    """
    sa, target = config["sensitive_col"], config["target_col"]
    mask, omit = set(mask_cols), set(omit_cols)
    pairs = []
    for col, val in row.items():
        if col == target or col in omit:
            continue
        if col == sa:
            if insert_protected is not None:
                pairs.append((config["protected_label"], insert_protected))
            continue
        pairs.append((col_label(col), UNKNOWN if col in mask else val))
    return pairs


def _sentence(pairs: list[tuple[str, object]]) -> str:
    clauses = [f"{k} is {v}" for k, v in pairs]
    if len(clauses) > 1:
        clauses = clauses[:-1] + [f"and {clauses[-1]}"]
    return "The applicant's " + ", ".join(clauses) + "."


def _demo_block(demos, config: dict, encoding: str, mask_cols=(), omit_cols=()) -> str:
    lines = [f"The following {len(demos)} labeled examples show the expected output format:\n"]
    for row, label in demos:
        p_yes = 0.9 if label == 1 else 0.1
        pairs = _profile_pairs(row, config, mask_cols, omit_cols)
        if encoding == "template":
            lines.append(f"  {_sentence(pairs)} → {{\"p_yes\": {p_yes}}}")
        else:
            attrs = " | ".join(f"{k}: {v}" for k, v in pairs)
            lines.append(f"  `[{attrs}]` → `{{\"p_yes\": {p_yes}}}`")
    lines.append("\nNow evaluate the following applicant:\n")
    return "\n".join(lines)


def serialize(
    row: pd.Series,
    config: dict,
    encoding: str = "template",
    demos: list[tuple[pd.Series, int]] | None = None,
    mask_cols=(),
    omit_cols=(),
    insert_protected: str | None = None,
    system_prefix: str = "",
    demo_mask_cols=(),
    demo_omit_cols=(),
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt)."""
    system = f"{system_prefix}\n\n{SYSTEM_PROMPT}" if system_prefix else SYSTEM_PROMPT
    pairs = _profile_pairs(row, config, mask_cols, omit_cols, insert_protected)
    prefix = _demo_block(demos, config, encoding, demo_mask_cols, demo_omit_cols) if demos else ""
    note = config.get("currency_note")

    if encoding == "template":
        lines = [prefix + "Applicant Information: " + _sentence(pairs), "",
                 f"Task: {config['task_desc']}"]
        if note:
            lines.append(f"Note: {note}")
        lines += ["", "Return ONLY a JSON object:",
                  '{"p_yes": <float between 0.0 and 1.0>}']
    elif encoding == "markdown":
        lines = [prefix + "## Applicant Profile", "",
                 "| Attribute | Value |", "|:----------|:------|"]
        lines += [f"| {k} | {v} |" for k, v in pairs]
        lines += ["", f"**Task:** {config['task_desc']}"]
        if note:
            lines.append(f"Note: {note}")
        lines += ["", "Return ONLY a JSON object:",
                  '`{"p_yes": <float between 0.0 and 1.0>}`']
    else:
        raise ValueError(f"Unknown encoding '{encoding}'")

    return system, "\n".join(lines)
