"""Agreement metrics for blind CVS-Act surgeon annotations."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import pandas as pd

from .annotation_store import ACTION_FIELDS


AGREEMENT_COLUMNS = ["granularity", "label_field", "n_clips", "fleiss_kappa"]
DISAGREEMENT_COLUMNS = ["clip_id", "granularity", "n_annotators", "labels", "disagreement_rate"]


def fleiss_kappa(labels_by_item: list[list[str]]) -> float:
    """Compute Fleiss' kappa for categorical labels."""
    complete = [labels for labels in labels_by_item if labels and len(set(labels)) >= 1]
    if not complete:
        return float("nan")
    n_raters = len(complete[0])
    if n_raters < 2 or any(len(labels) != n_raters for labels in complete):
        return float("nan")

    categories = sorted({label for labels in complete for label in labels})
    n_items = len(complete)
    p_j = {
        cat: sum(Counter(labels).get(cat, 0) for labels in complete) / (n_items * n_raters)
        for cat in categories
    }
    p_bar_e = sum(p * p for p in p_j.values())
    p_i = []
    for labels in complete:
        counts = Counter(labels)
        p_i.append(
            (sum(count * count for count in counts.values()) - n_raters)
            / (n_raters * (n_raters - 1))
        )
    p_bar = sum(p_i) / n_items
    if p_bar_e == 1:
        return 1.0 if p_bar == 1 else float("nan")
    return (p_bar - p_bar_e) / (1 - p_bar_e)


def compute_agreement(
    annotations,
    annotator_ids: Iterable[str] | None = None,
    label_field: str = "action_code",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-granularity agreement and high-disagreement clips."""
    df = annotations.copy() if isinstance(annotations, pd.DataFrame) else pd.DataFrame(annotations)
    if df.empty:
        return pd.DataFrame(columns=AGREEMENT_COLUMNS), pd.DataFrame(columns=DISAGREEMENT_COLUMNS)
    if label_field not in ACTION_FIELDS:
        raise ValueError(f"Unsupported label_field={label_field!r}; expected one of {ACTION_FIELDS}")
    if annotator_ids is not None:
        annotator_ids = list(annotator_ids)
        df = df[df["annotator_id"].isin(annotator_ids)]

    rows = []
    disagreements = []
    for granularity, sub in df.dropna(subset=[label_field]).groupby("granularity"):
        by_clip = []
        for clip_id, clip_rows in sub.groupby("clip_id"):
            labels = clip_rows.sort_values("annotator_id")[label_field].astype(str).tolist()
            n_labels = len(labels)
            if annotator_ids is not None and n_labels != len(annotator_ids):
                continue
            if n_labels < 2:
                continue
            by_clip.append(labels)
            counts = Counter(labels)
            majority_count = counts.most_common(1)[0][1]
            disagreements.append(
                {
                    "clip_id": clip_id,
                    "granularity": granularity,
                    "n_annotators": n_labels,
                    "labels": dict(counts),
                    "disagreement_rate": 1 - majority_count / n_labels,
                }
            )
        rows.append(
            {
                "granularity": granularity,
                "label_field": label_field,
                "n_clips": len(by_clip),
                "fleiss_kappa": fleiss_kappa(by_clip),
            }
        )
    agreement = (
        pd.DataFrame(rows, columns=AGREEMENT_COLUMNS)
        .sort_values("granularity")
        .reset_index(drop=True)
    )
    disagreement_df = (
        pd.DataFrame(disagreements, columns=DISAGREEMENT_COLUMNS)
        .sort_values(["disagreement_rate", "clip_id"], ascending=[False, True])
        .reset_index(drop=True)
        if disagreements
        else pd.DataFrame(columns=DISAGREEMENT_COLUMNS)
    )
    return agreement, disagreement_df


def print_agreement(annotations, annotator_ids: Iterable[str] | None = None, label_field: str = "action_code") -> None:
    agreement, disagreements = compute_agreement(annotations, annotator_ids, label_field)
    print("Agreement")
    print(agreement.to_string(index=False) if not agreement.empty else "(no complete labels)")
    print("\nHighest-disagreement clips")
    cols = ["clip_id", "granularity", "disagreement_rate", "labels"]
    print(disagreements[cols].head(20).to_string(index=False) if not disagreements.empty else "(none)")
