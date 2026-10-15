#!/usr/bin/env python3
"""Build GT-anchored max-overlap confusion matrices for current v3.1."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

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
    summary_rows = []
    for actor in ("left", "right", "camera"):
        pred_by_video = build_actor_video_maps(pred_records, actor)
        gt_by_video = build_actor_video_maps(gt_records, actor)
        for granularity in GRANULARITIES:
            level = "raw" if granularity == "exact" else granularity
            counts: Counter = Counter()
            match_counts: Counter = Counter()
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
                    winning_labels, overlap_frames = max_overlap_labels(anchor, candidates)
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
                            "max_overlap_frames": overlap_frames,
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
            anchor_rows.extend(level_anchor_rows)
            n_anchors = len(level_anchor_rows)
            summary_rows.append(
                {
                    "pipeline": "v3.1",
                    "actor": actor,
                    "level": level,
                    "n_gt_anchors": n_anchors,
                    "n_gt_classes": len(gt_labels),
                    "n_pred_classes_including_no_overlap": len(pred_labels),
                    "no_overlap_count": sum(
                        row["pred_label"] == NO_OVERLAP for row in level_anchor_rows
                    ),
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
    _write_csv(args.output_dir / "confusion_anchors.csv", anchor_rows)


if __name__ == "__main__":
    main()
