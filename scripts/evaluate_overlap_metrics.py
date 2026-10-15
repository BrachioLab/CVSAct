#!/usr/bin/env python3
"""Recompute interval-anchored overlap metrics for archived pipeline versions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

from cvs_act.action_segment_eval import evaluate_method_overlap_predictions


PIPELINES = {
    "base": (
        "qwen3.5_audit_v11_seg_extract_spec_eval",
        "structured_prediction",
    ),
    "cvs_v1": (
        "qwen3.5_audit_v11_seg_extract_spec_eval_with_cvs_context",
        "structured_prediction_cvs_context",
    ),
    "cvs_v2": (
        "qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v2__baseline",
        "structured_prediction_cvs_context",
    ),
    "cvs_v3": (
        "qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v3",
        "structured_prediction_cvs_context",
    ),
    "v3.1": (
        "qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v3_1",
        "structured_prediction_cvs_context",
    ),
    "v3.2": (
        "qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v3_2",
        "structured_prediction_cvs_context",
    ),
    "v4": (
        "qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v4",
        "structured_prediction_cvs_context",
    ),
}

# This snapshot is byte-identical across the base, CVS-v1, CVS-v2, CVS-v3,
# and CVS-v4 evaluation directories and is the GT conversion archived with the
# original experiments. The separately regenerated shared GT file differs in a
# small number of right/camera rows, so it is intentionally not mixed in here.
GT_RELATIVE_PATH = Path(
    "qwen3.5_audit_v11_seg_extract_spec_eval/audit_v11_simple_actions.json"
)
SELECTED_COMPONENTS = {("left", "exact"), ("right", "medium"), ("camera", "coarse")}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text())


def _prediction_records(results_path: Path, method: str) -> list[dict]:
    records = []
    for item in _read_json(results_path)["results"]:
        if item.get("status") != "ok" or item.get("method") != method:
            continue
        prediction = dict(item["extracted_actions"])
        prediction.update(
            example_id=item["example_id"],
            video_id=item["video_id"],
            criterion=item.get("criterion"),
        )
        records.append(prediction)
    return records


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/mnt/md0/weiqiuy/surgent/notebooks/artifacts"),
        help="Directory containing the preserved raw pipeline result directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks/archive/results/current"),
    )
    args = parser.parse_args()

    gt_path = args.artifact_root / GT_RELATIVE_PATH
    gt_records = _read_json(gt_path)
    gt_example_ids = {record["example_id"] for record in gt_records}
    if len(gt_example_ids) != len(gt_records):
        raise ValueError("Ground truth contains duplicate example_id values")
    detailed = {}
    provenance = {"ground_truth": {"path": str(gt_path), "sha256": _sha256(gt_path)}}
    component_rows = []
    breakdown_rows = []
    summary_rows = []
    selected_rows = []

    for pipeline, (directory, method) in PIPELINES.items():
        results_path = args.artifact_root / directory / "results.json"
        pred_records = _prediction_records(results_path, method)
        pred_example_ids = {record["example_id"] for record in pred_records}
        if len(pred_example_ids) != len(pred_records):
            raise ValueError(f"{pipeline} contains duplicate structured predictions")
        if pred_example_ids != gt_example_ids:
            missing = sorted(gt_example_ids - pred_example_ids)
            extra = sorted(pred_example_ids - gt_example_ids)
            raise ValueError(f"{pipeline} example mismatch; missing={missing}, extra={extra}")
        rows = evaluate_method_overlap_predictions(pred_records, gt_records)
        detailed[pipeline] = rows
        provenance[pipeline] = {
            "path": str(results_path),
            "sha256": _sha256(results_path),
            "method": method,
            "n_prediction_records": len(pred_records),
        }

        for row in rows:
            component_rows.append(
                {
                    "pipeline": pipeline,
                    "method": method,
                    "actor": row["actor"],
                    "granularity": row["granularity"],
                    "overlap_mode": row["overlap_mode"],
                    "f1_mean": row["f1_mean"],
                    "pred_anchor_accuracy_mean": row["pred_anchor_accuracy_mean"],
                    "gt_anchor_accuracy_mean": row["gt_anchor_accuracy_mean"],
                    "n_videos": row["n_videos"],
                }
            )

        for mode in ("any", "max"):
            mode_rows = [row for row in rows if row["overlap_mode"] == mode]
            summary_rows.append(
                {
                    "pipeline": pipeline,
                    "method": method,
                    "overlap_mode": mode,
                    "f1_mean": mean(row["f1_mean"] for row in mode_rows),
                    "pred_anchor_accuracy_mean": mean(
                        row["pred_anchor_accuracy_mean"] for row in mode_rows
                    ),
                    "gt_anchor_accuracy_mean": mean(
                        row["gt_anchor_accuracy_mean"] for row in mode_rows
                    ),
                    "actor_granularity_rows": len(mode_rows),
                }
            )
            chosen = [
                row
                for row in mode_rows
                if (row["actor"], row["granularity"]) in SELECTED_COMPONENTS
            ]
            selected_rows.append(
                {
                    "pipeline": pipeline,
                    "method": method,
                    "overlap_mode": mode,
                    "f1_mean": mean(row["f1_mean"] for row in chosen),
                    "pred_anchor_accuracy_mean": mean(
                        row["pred_anchor_accuracy_mean"] for row in chosen
                    ),
                    "gt_anchor_accuracy_mean": mean(
                        row["gt_anchor_accuracy_mean"] for row in chosen
                    ),
                    "components": "left/exact+right/medium+camera/coarse",
                }
            )
            component_lookup = {
                (row["actor"], row["granularity"]): row["f1_mean"]
                for row in mode_rows
            }
            breakdown_rows.append(
                {
                    "pipeline": pipeline,
                    "overlap_mode": mode,
                    **{
                        f"{actor}_{'raw' if granularity == 'exact' else granularity}": component_lookup[
                            (actor, granularity)
                        ]
                        for actor in ("left", "right", "camera")
                        for granularity in ("exact", "medium", "coarse")
                    },
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "overlap_metrics_summary.csv", summary_rows)
    _write_csv(args.output_dir / "overlap_metrics_selected_components.csv", selected_rows)
    _write_csv(args.output_dir / "overlap_metrics_by_actor_granularity.csv", component_rows)
    _write_csv(args.output_dir / "overlap_metrics_f1_breakdown.csv", breakdown_rows)
    (args.output_dir / "overlap_metric_results.json").write_text(
        json.dumps(
            {
                "metric_policy": {
                    "interval_convention": "half-open [start_frame, end_frame)",
                    "any": "anchor correct if any positive-overlap opposing interval has a matching label",
                    "max": "anchor correct if a matching label ties for greatest unioned accumulated overlap duration",
                    "pred_anchor": "precision-like: correct predicted anchors / predicted anchors",
                    "gt_anchor": "recall-like: correct GT anchors / GT anchors",
                    "f1": "harmonic mean per video, then macro mean over GT-positive videos",
                    "matching": "not one-to-one; each anchor is judged independently",
                },
                "provenance": provenance,
                "results": detailed,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
