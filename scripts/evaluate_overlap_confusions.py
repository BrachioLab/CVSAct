#!/usr/bin/env python3
"""Build GT-anchored max-overlap confusion matrices for current v3.1."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from cvs_act.action_segment_eval import (
    GRANULARITIES,
    actor_label,
    build_actor_video_maps,
    labels_match,
    max_overlap_labels,
)


PIPELINE_DIRECTORY = "qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v3_1"
METHOD = "structured_prediction_cvs_context"
GT_RELATIVE_PATH = Path(
    "qwen3.5_audit_v11_seg_extract_spec_eval/audit_v11_simple_actions.json"
)
NO_OVERLAP = "<no_overlap>"


def _multiclass_macro_f1(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> float:
    """Return unweighted per-class F1, matching the paper-era helper."""
    evaluated_labels = labels or sorted(set(y_true) | set(y_pred))
    scores = []
    for label in evaluated_labels:
        tp = sum(true == label and pred == label for true, pred in zip(y_true, y_pred))
        fp = sum(true != label and pred == label for true, pred in zip(y_true, y_pred))
        fn = sum(true == label and pred != label for true, pred in zip(y_true, y_pred))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return mean(scores) if scores else 0.0


def _read_json(path: Path):
    return json.loads(path.read_text())


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _display_label(label: Any) -> str:
    if isinstance(label, tuple):
        return " | ".join(str(value) for value in label)
    return str(label)


def _prediction_records(results_path: Path) -> list[dict]:
    records = []
    for item in _read_json(results_path)["results"]:
        if item.get("status") != "ok" or item.get("method") != METHOD:
            continue
        prediction = dict(item["extracted_actions"])
        prediction.update(
            example_id=item["example_id"],
            video_id=item["video_id"],
            criterion=item.get("criterion"),
        )
        records.append(prediction)
    return records


def _matrix_rows(
    counts: Counter,
    gt_labels: list[str],
    pred_labels: list[str],
    normalized: bool,
) -> list[dict]:
    rows = []
    for gt_label in gt_labels:
        total = sum(counts[(gt_label, pred_label)] for pred_label in pred_labels)
        row = {"gt_label": gt_label}
        for pred_label in pred_labels:
            value = counts[(gt_label, pred_label)]
            row[pred_label] = value / total if normalized and total else value
        rows.append(row)
    return rows


def _plot_matrix(
    counts_path: Path,
    normalized_path: Path,
    output_path: Path,
    title: str,
    *,
    all_overlaps: bool = False,
) -> None:
    counts = pd.read_csv(counts_path, index_col="gt_label")
    normalized = pd.read_csv(normalized_path, index_col="gt_label")
    width = min(30, max(8, 0.65 * len(normalized.columns)))
    height = min(34, max(5, 0.48 * len(normalized.index)))
    figure, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(normalized.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(
        range(len(normalized.columns)),
        normalized.columns,
        rotation=60,
        ha="right",
        fontsize=8,
    )
    axis.set_yticks(range(len(normalized.index)), normalized.index, fontsize=8)
    axis.set_xlabel(
        "Predicted overlapping label"
        if all_overlaps
        else "Predicted max-overlap label"
    )
    axis.set_ylabel("GT label")
    axis.set_title(title)
    if normalized.size <= 200:
        for row in range(len(normalized.index)):
            for column in range(len(normalized.columns)):
                count = int(counts.iat[row, column])
                if not count:
                    continue
                rate = normalized.iat[row, column]
                color = "white" if rate >= 0.55 else "black"
                axis.text(
                    column,
                    row,
                    f"{count}\n{rate:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.8)
    colorbar.set_label("Fraction within GT row")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/mnt/md0/weiqiuy/surgent/notebooks/artifacts"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks/archive/results/current/confusion_matrices"),
    )
    args = parser.parse_args()

    gt_records = _read_json(args.artifact_root / GT_RELATIVE_PATH)
    pred_records = _prediction_records(
        args.artifact_root / PIPELINE_DIRECTORY / "results.json"
    )
    if {row["example_id"] for row in gt_records} != {
        row["example_id"] for row in pred_records
    }:
        raise ValueError("v3.1 prediction and GT example sets differ")

    anchor_rows = []
    pair_rows = []
    all_overlap_pair_rows = []
    summary_rows = []
    for actor in ("left", "right", "camera"):
        pred_by_video = build_actor_video_maps(pred_records, actor)
        gt_by_video = build_actor_video_maps(gt_records, actor)
        for granularity in GRANULARITIES:
            level = "raw" if granularity == "exact" else granularity
            counts: Counter = Counter()
            match_counts: Counter = Counter()
            all_overlap_counts: Counter = Counter()
            all_overlap_match_counts: Counter = Counter()
            all_overlap_frames: Counter = Counter()
            level_anchor_rows = []
            for video_id, gt_segments in sorted(gt_by_video.items()):
                candidates = [
                    (
                        segment["start_frame"],
                        segment["end_frame"],
                        actor_label(segment, actor, granularity),
                    )
                    for segment in pred_by_video.get(video_id, [])
                ]
                for segment in gt_segments:
                    anchor = (
                        segment["start_frame"],
                        segment["end_frame"],
                        actor_label(segment, actor, granularity),
                    )
                    winning_labels, winning_overlap_frames = max_overlap_labels(
                        anchor, candidates
                    )
                    matching_winners = [
                        label
                        for label in winning_labels
                        if labels_match(anchor[2], label, actor, granularity)
                    ]
                    # Prefer a matching label in a duration tie, exactly as F1@max
                    # does; use text order only to make remaining ties reproducible.
                    winner_pool = matching_winners or winning_labels
                    predicted_label = (
                        min(winner_pool, key=_display_label) if winner_pool else None
                    )
                    gt_text = _display_label(anchor[2])
                    pred_text = (
                        _display_label(predicted_label)
                        if predicted_label is not None
                        else NO_OVERLAP
                    )
                    evaluator_match = predicted_label is not None and labels_match(
                        anchor[2], predicted_label, actor, granularity
                    )
                    counts[(gt_text, pred_text)] += 1
                    match_counts[(gt_text, pred_text)] += int(evaluator_match)

                    overlapping_candidates = []
                    for candidate_start, candidate_end, candidate_label in candidates:
                        pair_overlap_frames = max(
                            0,
                            min(anchor[1], candidate_end)
                            - max(anchor[0], candidate_start),
                        )
                        if pair_overlap_frames:
                            overlapping_candidates.append(
                                (candidate_label, pair_overlap_frames)
                            )
                    if overlapping_candidates:
                        for candidate_label, pair_overlap_frames in overlapping_candidates:
                            candidate_text = _display_label(candidate_label)
                            pair = (gt_text, candidate_text)
                            all_overlap_counts[pair] += 1
                            all_overlap_frames[pair] += pair_overlap_frames
                            all_overlap_match_counts[pair] += int(
                                labels_match(
                                    anchor[2], candidate_label, actor, granularity
                                )
                            )
                    else:
                        all_overlap_counts[(gt_text, NO_OVERLAP)] += 1
                    level_anchor_rows.append(
                        {
                            "pipeline": "v3.1",
                            "actor": actor,
                            "level": level,
                            "video_id": video_id,
                            "start_frame": anchor[0],
                            "end_frame": anchor[1],
                            "gt_label": gt_text,
                            "pred_label": pred_text,
                            "max_overlap_frames": winning_overlap_frames,
                            "n_tied_winners": len(winning_labels),
                            "evaluator_match": evaluator_match,
                        }
                    )

            gt_labels = sorted({gt for gt, _ in counts})
            pred_labels = sorted(
                {pred for _, pred in counts},
                key=lambda label: (label == NO_OVERLAP, label),
            )
            fieldnames = ["gt_label", *pred_labels]
            stem = f"{actor}_{level}"
            _write_csv(
                args.output_dir / f"{stem}_counts.csv",
                _matrix_rows(counts, gt_labels, pred_labels, normalized=False),
                fieldnames,
            )
            _write_csv(
                args.output_dir / f"{stem}_row_normalized.csv",
                _matrix_rows(counts, gt_labels, pred_labels, normalized=True),
                fieldnames,
            )
            _plot_matrix(
                args.output_dir / f"{stem}_counts.csv",
                args.output_dir / f"{stem}_row_normalized.csv",
                args.output_dir / f"{stem}_row_normalized.png",
                f"v3.1 {actor.title()} {level.title()} Confusion Matrix",
            )
            all_overlap_pred_labels = sorted(
                {pred for _, pred in all_overlap_counts},
                key=lambda label: (label == NO_OVERLAP, label),
            )
            all_overlap_fieldnames = ["gt_label", *all_overlap_pred_labels]
            _write_csv(
                args.output_dir / f"{stem}_all_overlaps_counts.csv",
                _matrix_rows(
                    all_overlap_counts,
                    gt_labels,
                    all_overlap_pred_labels,
                    normalized=False,
                ),
                all_overlap_fieldnames,
            )
            _write_csv(
                args.output_dir / f"{stem}_all_overlaps_row_normalized.csv",
                _matrix_rows(
                    all_overlap_counts,
                    gt_labels,
                    all_overlap_pred_labels,
                    normalized=True,
                ),
                all_overlap_fieldnames,
            )
            _plot_matrix(
                args.output_dir / f"{stem}_all_overlaps_counts.csv",
                args.output_dir / f"{stem}_all_overlaps_row_normalized.csv",
                args.output_dir / f"{stem}_all_overlaps_row_normalized.png",
                f"v3.1 {actor.title()} {level.title()} All-Overlap Confusion Matrix",
                all_overlaps=True,
            )
            for (gt_label, pred_label), count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            ):
                gt_total = sum(
                    pair_count
                    for (candidate_gt, _), pair_count in counts.items()
                    if candidate_gt == gt_label
                )
                pair_rows.append(
                    {
                        "pipeline": "v3.1",
                        "actor": actor,
                        "level": level,
                        "gt_label": gt_label,
                        "pred_label": pred_label,
                        "count": count,
                        "gt_row_rate": count / gt_total,
                        "evaluator_match_count": match_counts[(gt_label, pred_label)],
                    }
                )
            for (gt_label, pred_label), count in sorted(
                all_overlap_counts.items(), key=lambda item: (-item[1], item[0])
            ):
                gt_total = sum(
                    pair_count
                    for (candidate_gt, _), pair_count in all_overlap_counts.items()
                    if candidate_gt == gt_label
                )
                all_overlap_pair_rows.append(
                    {
                        "pipeline": "v3.1",
                        "actor": actor,
                        "level": level,
                        "gt_label": gt_label,
                        "pred_label": pred_label,
                        "overlap_pair_count": count,
                        "gt_row_pair_rate": count / gt_total,
                        "overlap_frames_total": all_overlap_frames[
                            (gt_label, pred_label)
                        ],
                        "evaluator_match_pair_count": all_overlap_match_counts[
                            (gt_label, pred_label)
                        ],
                    }
                )
            anchor_rows.extend(level_anchor_rows)
            n_anchors = len(level_anchor_rows)
            true_labels = [row["gt_label"] for row in level_anchor_rows]
            predicted_labels = [row["pred_label"] for row in level_anchor_rows]
            majority_label, majority_count = Counter(true_labels).most_common(1)[0]
            summary_rows.append(
                {
                    "pipeline": "v3.1",
                    "actor": actor,
                    "level": level,
                    "n_gt_anchors": n_anchors,
                    "n_gt_classes": len(gt_labels),
                    "n_pred_classes_including_no_overlap": len(pred_labels),
                    "majority_gt_label": majority_label,
                    "majority_gt_count": majority_count,
                    "majority_gt_share": majority_count / n_anchors,
                    "no_overlap_count": sum(
                        row["pred_label"] == NO_OVERLAP for row in level_anchor_rows
                    ),
                    "all_overlap_pair_count_including_misses": sum(
                        all_overlap_counts.values()
                    ),
                    "all_overlap_matching_pair_count": sum(
                        all_overlap_match_counts.values()
                    ),
                    "class_macro_f1_gt_anchored_max": _multiclass_macro_f1(
                        true_labels, predicted_labels
                    ),
                    "majority_class_macro_f1": _multiclass_macro_f1(
                        true_labels,
                        [majority_label] * n_anchors,
                        gt_labels,
                    ),
                    "exact_string_match_rate_micro": sum(
                        true == predicted
                        for true, predicted in zip(true_labels, predicted_labels)
                    )
                    / n_anchors,
                    "evaluator_match_count": sum(
                        bool(row["evaluator_match"]) for row in level_anchor_rows
                    ),
                    "evaluator_match_rate_micro": (
                        sum(bool(row["evaluator_match"]) for row in level_anchor_rows)
                        / n_anchors
                        if n_anchors
                        else 0.0
                    ),
                }
            )

    _write_csv(args.output_dir / "confusion_summary.csv", summary_rows)
    _write_csv(args.output_dir / "confusion_pairs_long.csv", pair_rows)
    _write_csv(
        args.output_dir / "confusion_all_overlaps_pairs_long.csv",
        all_overlap_pair_rows,
    )
    _write_csv(args.output_dir / "confusion_anchors.csv", anchor_rows)


if __name__ == "__main__":
    main()
