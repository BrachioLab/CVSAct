#!/usr/bin/env python3
"""Onset-aligned pointwise CVS-Act action-recommendation evaluation.

This is a post-hoc evaluator over existing prediction JSONL files. It does not
run model inference and intentionally does not modify the older recommendation
evaluators so prior results remain reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT_DIR / "scripts" / "eval"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from cvs_act.action_segment_eval import GRANULARITIES, actor_label, labels_match  # noqa: E402
from evaluate_audit_v11_simple_recommendation import (  # noqa: E402
    choose_start_frame,
    gt_actor_row,
    gt_records_by_video,
    pred_actor_row,
    prediction_frame_number,
    write_json,
)
from evaluate_cvs_act_v1_combined import (  # noqa: E402
    collect_prediction_groups,
    load_audit_v11_action_records,
)


ACTORS = ["left", "right", "camera"]
MODEL_SHORT = {
    "gpt-5.4-mini": "gpt",
    "claude-haiku-4-5-20251001": "claude",
    "gemini-2.5-flash": "gemini",
}
VARIANT_LABELS = {
    "cot_fixedk3_norecdescs_fmeta": "Baseline default",
    "cot_fixedk3_no_cvs_norecdescs_fmeta": "Baseline no-cvs",
    "cot_fixedk3_no_cvs_no_guideline_norecdescs_fmeta": "Baseline no-cvs-no-guideline",
    "cot_fixedk3_no_cvs_no_desc_norecdescs_fmeta": "Baseline no-cvs-no-desc",
    "pref-cvs_arec_steps5_fixedk3_norecdescs_fmeta": "SurGent default",
}

# Explicit absent/present definitions.
LEFT_ABSENT = {"(not set)", "", None}
LEFT_PRESENT_PREFIXES = ("KEEP_", "RETRACT_")

RIGHT_ABSENT_VALUES = {"(not set)", "", None}
RIGHT_FIELDS = ["tool_type", "action_code", "target_structure", "target_context_1", "target_context_2"]

CAMERA_ABSENT = {"CAMERA_NO_CHANGE", "(not set)", "", None}
CAMERA_PRESENT = {"CAMERA_ZOOM_IN", "CAMERA_ZOOM_OUT", "CAMERA_REPOSITION"}
CAMERA_EXCLUDE_ON_GT = {"CAMERA_UNCERTAIN"}
CAMERA_PRED_UNCERTAIN = {"CAMERA_UNCERTAIN"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Prediction roots to scan.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-5.4-mini", "claude-haiku-4-5-20251001", "gemini-2.5-flash"],
    )
    parser.add_argument("--taxonomy", nargs="+", default=["cvs_act_current_simple_v1"])
    parser.add_argument("--common-only", action="store_true")
    parser.add_argument(
        "--audit-actions-dir",
        type=Path,
        default=ROOT_DIR / "data/processed/CVS_Challenge_SAGES_v1/cvs_act_annotations/v1/audit_v11",
    )
    parser.add_argument(
        "--metrics-base",
        type=Path,
        default=ROOT_DIR / "metrics_cvs_act_v1_onset_pointwise",
    )
    parser.add_argument("--include-partial", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def method_variant(method: str) -> str:
    return method.split("/")[0]


def method_model(method: str) -> str:
    parts = method.split("/")
    return parts[1] if len(parts) > 1 else ""


def method_taxonomy(method: str) -> str:
    parts = method.split("/")
    return parts[2] if len(parts) > 2 else ""


def method_label(method: str) -> str:
    variant = method_variant(method)
    model = method_model(method)
    return f"{VARIANT_LABELS.get(variant, variant)} {MODEL_SHORT.get(model, model)}"


def model_short(model: str) -> str:
    return MODEL_SHORT.get(model, model)


def is_empty_right_tuple(row: dict) -> bool:
    return all(str(row.get(field, "(not set)") or "(not set)") in RIGHT_ABSENT_VALUES for field in RIGHT_FIELDS)


def presence_call(actor: str, row: dict, *, is_gt: bool) -> tuple[bool | None, bool, str]:
    """Return (present, excluded, raw_value)."""
    if actor == "left":
        code = row.get("retraction_direction_code", "(not set)")
        if code in LEFT_ABSENT:
            return False, False, str(code)
        if str(code).startswith(LEFT_PRESENT_PREFIXES):
            return True, False, str(code)
        return None, False, str(code)
    if actor == "right":
        return (not is_empty_right_tuple(row)), False, str(tuple(row.get(field, "(not set)") for field in RIGHT_FIELDS))
    if actor == "camera":
        code = row.get("action_code", "(not set)")
        if is_gt and code in CAMERA_EXCLUDE_ON_GT:
            return None, True, str(code)
        if (not is_gt) and code in CAMERA_PRED_UNCERTAIN:
            return False, False, str(code)
        if code in CAMERA_PRESENT:
            return True, False, str(code)
        if code in CAMERA_ABSENT:
            return False, False, str(code)
        return None, False, str(code)
    raise ValueError(actor)


def label_value(row: dict, actor: str, granularity: str) -> Any:
    return actor_label(row, actor, granularity)


def label_text(value: Any) -> str:
    if isinstance(value, tuple):
        return repr(tuple(str(item) for item in value))
    return str(value)


def active_rows_at(record: dict, actor: str, frame_num: int) -> list[dict]:
    rows = []
    clip_end = int(record["frame_range"][1])
    for row in record.get(actor, []) or []:
        start = int(row.get("start_frame", -1))
        end = int(row.get("end_frame", -1))
        if start <= frame_num < end or (frame_num == end and end == clip_end):
            rows.append(row)
    return rows


def actor_onset_points(record: dict, actor: str) -> list[dict]:
    points = []
    seen: set[tuple[int, str]] = set()
    for row_index, row in enumerate(record.get(actor, []) or []):
        if row.get("start_frame") is None:
            continue
        start = int(row["start_frame"])
        active = active_rows_at(record, actor, start)
        key = (start, json.dumps(active, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        points.append(
            {
                "actor": actor,
                "point_frame": start,
                "source_row_index": row_index,
                "active_rows": active,
            }
        )
    return points


def label_signature(rows: Sequence[dict], actor: str) -> tuple[str, ...]:
    """Stable actor-label signature used to merge connected equal segments."""
    labels = []
    for row in rows:
        labels.append(tuple(label_text(label_value(row, actor, granularity)) for granularity in GRANULARITIES))
    return tuple(sorted(repr(label) for label in labels))


def filled_actor_segments(record: dict, actor: str) -> list[dict]:
    """Return a clip-covering actor timeline with no-action gaps filled.

    Labeled action rows define intervals. Any interval with no active row is
    represented by the actor's default absent row from gt_actor_row. Adjacent
    intervals with the same actor-label signature are merged.
    """
    clip_start, clip_end = [int(value) for value in record["frame_range"]]
    boundaries = {clip_start, clip_end}
    for row in record.get(actor, []) or []:
        if row.get("start_frame") is not None:
            boundaries.add(max(clip_start, min(clip_end, int(row["start_frame"]))))
        if row.get("end_frame") is not None:
            boundaries.add(max(clip_start, min(clip_end, int(row["end_frame"]))))
    ordered = sorted(boundaries)

    segments = []
    for start, end in zip(ordered, ordered[1:]):
        if start >= end:
            continue
        rows = active_rows_at(record, actor, start)
        if not rows:
            rows = [gt_actor_row(record, actor, start)]
        signature = label_signature(rows, actor)
        if segments and segments[-1]["end_frame"] == start and segments[-1]["signature"] == signature:
            segments[-1]["end_frame"] = end
            continue
        segments.append(
            {
                "actor": actor,
                "start_frame": start,
                "end_frame": end,
                "active_rows": rows,
                "signature": signature,
            }
        )
    return segments


def actor_complete_onset_points(record: dict) -> list[dict]:
    """Evaluate every actor at the union of filled actor-segment starts."""
    segments_by_actor = {actor: filled_actor_segments(record, actor) for actor in ACTORS}
    starts = sorted({segment["start_frame"] for segments in segments_by_actor.values() for segment in segments})
    points = []
    for start in starts:
        for actor in ACTORS:
            segment_rows = []
            for segment in segments_by_actor[actor]:
                if segment["start_frame"] <= start < segment["end_frame"]:
                    segment_rows = list(segment["active_rows"])
                    break
            if not segment_rows:
                segment_rows = [gt_actor_row(record, actor, start)]
            points.append(
                {
                    "actor": actor,
                    "point_frame": start,
                    "source_row_index": None,
                    "active_rows": segment_rows,
                }
            )
    return points


def fine_clip_start_points(record: dict) -> list[dict]:
    start = int(record["frame_range"][0])
    return [
        {
            "actor": actor,
            "point_frame": start,
            "source_row_index": None,
            "active_rows": active_rows_at(record, actor, start),
        }
        for actor in ACTORS
    ]


def collect_eval_points(records: Sequence[dict]) -> list[dict]:
    points = []
    for record in records:
        clip_level = record.get("clip_level", "coarse")
        if clip_level == "coarse":
            point_source = "coarse_actor_onset"
            for point in actor_complete_onset_points(record):
                points.append({**point, "record": record, "point_source": point_source})
        elif clip_level == "fine":
            for point in fine_clip_start_points(record):
                points.append({**point, "record": record, "point_source": "fine_clip_start_current"})
            for point in actor_complete_onset_points(record):
                points.append({**point, "record": record, "point_source": "fine_actor_onset"})
    return points


def resolve_gt_rows(point: dict, actor: str, frame_num: int) -> list[dict]:
    rows = point.get("active_rows") or []
    if rows:
        return list(rows)
    return [gt_actor_row(point["record"], actor, frame_num)]


def choose_resolved_gt_label(gt_labels: list[Any], pred_label: Any, actor: str, granularity: str) -> Any:
    for label in gt_labels:
        if labels_match(pred_label, label, actor, granularity):
            return label
    return sorted(gt_labels, key=label_text)[0] if gt_labels else "(not set)"


def match_any(pred_row: dict, gt_rows: Sequence[dict], actor: str, granularity: str) -> tuple[bool, Any, list[Any]]:
    pred_label = label_value(pred_row, actor, granularity)
    gt_labels = [label_value(row, actor, granularity) for row in gt_rows]
    return any(labels_match(pred_label, gt_label, actor, granularity) for gt_label in gt_labels), pred_label, gt_labels


def present_label_rows(gt_rows: Sequence[dict], actor: str) -> list[dict]:
    rows = []
    for row in gt_rows:
        present, excluded, _raw = presence_call(actor, row, is_gt=True)
        if excluded:
            continue
        if present:
            rows.append(row)
    return rows


def evaluate_points(frames_by_method: dict, eval_points: Sequence[dict]) -> list[dict]:
    rows: list[dict] = []
    by_video_points: dict[str, list[dict]] = defaultdict(list)
    for point in eval_points:
        by_video_points[str(point["record"]["video_id"])].append(point)

    for method, by_video in sorted(frames_by_method.items()):
        variant = method_variant(method)
        model = method_model(method)
        taxonomy = method_taxonomy(method)
        for video_id, points in sorted(by_video_points.items()):
            pred_frames = by_video.get(video_id)
            if not pred_frames:
                continue
            pred_frames = sorted(pred_frames, key=prediction_frame_number)
            available = [prediction_frame_number(frame) for frame in pred_frames]
            by_frame = {prediction_frame_number(frame): frame for frame in pred_frames}
            for point in points:
                actor = point["actor"]
                record = point["record"]
                true_start = int(point["point_frame"])
                eval_frame = choose_start_frame(true_start, available)
                if eval_frame is None:
                    continue
                pred_row = pred_actor_row(by_frame[eval_frame], actor)
                gt_rows = resolve_gt_rows(point, actor, true_start)
                present_gt_rows = present_label_rows(gt_rows, actor)

                gt_present_values = [presence_call(actor, row, is_gt=True) for row in gt_rows]
                gt_excluded = actor == "camera" and any(excluded for _present, excluded, _raw in gt_present_values)
                gt_unknown = any((present is None and not excluded) for present, excluded, _raw in gt_present_values)
                gt_present = any(bool(present) for present, excluded, _raw in gt_present_values if not excluded)
                pred_present, pred_excluded, pred_presence_raw = presence_call(actor, pred_row, is_gt=False)
                pred_unknown = pred_present is None

                base = {
                    "method": method,
                    "method_label": method_label(method),
                    "variant": variant,
                    "variant_label": VARIANT_LABELS.get(variant, variant),
                    "model": model,
                    "model_short": model_short(model),
                    "taxonomy": taxonomy,
                    "video_id": video_id,
                    "example_id": record.get("example_id"),
                    "criterion": record.get("criterion"),
                    "clip_level": record.get("clip_level", "coarse"),
                    "point_source": point["point_source"],
                    "actor": actor,
                    "true_onset_frame": true_start,
                    "eval_keyframe": eval_frame,
                    "n_active_rows": len(gt_rows),
                    "n_present_gt_rows": len(present_gt_rows),
                    "gt_presence_raw": " || ".join(raw for _present, _excluded, raw in gt_present_values),
                    "pred_presence_raw": pred_presence_raw,
                    "gt_present": gt_present,
                    "pred_present": bool(pred_present) if pred_present is not None else None,
                    "gt_presence_excluded_uncertain": gt_excluded,
                    "gt_presence_unknown": gt_unknown,
                    "pred_presence_unknown": pred_unknown,
                    "lookahead_status": "unverified",
                    "lookahead_flag": "unverified",
                }
                for granularity in GRANULARITIES:
                    label_gt_rows = present_gt_rows
                    if not label_gt_rows:
                        gt_labels = [label_value(row, actor, granularity) for row in gt_rows]
                        pred_label = label_value(pred_row, actor, granularity)
                        label_match = None
                        resolved_gt = choose_resolved_gt_label(gt_labels, pred_label, actor, granularity) if gt_labels else "(not set)"
                    else:
                        label_match, pred_label, gt_labels = match_any(pred_row, label_gt_rows, actor, granularity)
                        resolved_gt = choose_resolved_gt_label(gt_labels, pred_label, actor, granularity)
                    base[f"gt_label_{granularity}"] = " || ".join(label_text(label) for label in gt_labels)
                    base[f"gt_label_resolved_{granularity}"] = label_text(resolved_gt)
                    base[f"pred_label_{granularity}"] = label_text(pred_label)
                    base[f"label_match_{granularity}"] = label_match
                rows.append(base)
    return rows


def binary_f1(y_true: Sequence[bool], y_pred: Sequence[bool]) -> float | None:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else None


def balanced_accuracy(y_true: Sequence[bool], y_pred: Sequence[bool]) -> float | None:
    pos = [p for t, p in zip(y_true, y_pred) if t]
    neg = [p for t, p in zip(y_true, y_pred) if not t]
    vals = []
    if pos:
        vals.append(sum(pos) / len(pos))
    if neg:
        vals.append(sum(not p for p in neg) / len(neg))
    return mean(vals) if vals else None


def inactive_fpr(y_true: Sequence[bool], y_pred: Sequence[bool]) -> float | None:
    neg = [p for t, p in zip(y_true, y_pred) if not t]
    if not neg:
        return None
    return sum(neg) / len(neg)


def multiclass_macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] | None = None) -> float | None:
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    scores = []
    for label in labels:
        tp = sum((t == label) and (p == label) for t, p in zip(y_true, y_pred))
        fp = sum((t != label) and (p == label) for t, p in zip(y_true, y_pred))
        fn = sum((t == label) and (p != label) for t, p in zip(y_true, y_pred))
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return mean(scores) if scores else None


def summarize_presence(rows: list[dict]) -> list[dict]:
    out = []
    keys = ["method", "method_label", "variant_label", "model_short", "clip_level", "point_source", "actor"]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row["gt_presence_excluded_uncertain"] or row["gt_presence_unknown"] or row["pred_presence_unknown"]:
            continue
        groups[tuple(row[k] for k in keys)].append(row)
    for key, group in sorted(groups.items()):
        y_true = [bool(row["gt_present"]) for row in group]
        y_pred = [bool(row["pred_present"]) for row in group]
        null_pred = [True] * len(y_true)
        row = dict(zip(keys, key))
        row.update(
            {
                "n": len(group),
                "n_gt_present": sum(y_true),
                "n_gt_absent": len(y_true) - sum(y_true),
                "balanced_accuracy": balanced_accuracy(y_true, y_pred),
                "present_f1": binary_f1(y_true, y_pred),
                "inactive_false_positive_rate": inactive_fpr(y_true, y_pred),
                "always_present_balanced_accuracy": balanced_accuracy(y_true, null_pred),
                "always_present_f1": binary_f1(y_true, null_pred),
                "always_present_inactive_false_positive_rate": inactive_fpr(y_true, null_pred),
            }
        )
        out.append(row)
    out.extend(macro_actor_rows(out, ["balanced_accuracy", "present_f1", "inactive_false_positive_rate"]))
    return out


def summarize_labels(rows: list[dict]) -> list[dict]:
    out = []
    keys = ["method", "method_label", "variant_label", "model_short", "clip_level", "point_source", "actor", "granularity"]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if not row["gt_present"] or row["gt_presence_excluded_uncertain"] or row["gt_presence_unknown"]:
            continue
        for granularity in GRANULARITIES:
            item = dict(row)
            item["granularity"] = granularity
            item["gt_label"] = str(row[f"gt_label_resolved_{granularity}"])
            item["pred_label"] = str(row[f"pred_label_{granularity}"])
            item["label_match"] = bool(row[f"label_match_{granularity}"])
            groups[tuple(item[k] for k in keys)].append(item)
    for key, group in sorted(groups.items()):
        y_true = [row["gt_label"] for row in group]
        y_pred = [row["pred_label"] for row in group]
        labels = sorted(set(y_true) | set(y_pred))
        gt_labels = sorted(set(y_true))
        majority_label = Counter(y_true).most_common(1)[0][0] if y_true else ""
        majority_pred = [majority_label] * len(y_true)
        row = dict(zip(keys, key))
        row.update(
            {
                "n": len(group),
                "n_labels": len(set(y_true)),
                "label_macro_f1": multiclass_macro_f1(y_true, y_pred, labels),
                "majority_label": majority_label,
                "majority_label_macro_f1": multiclass_macro_f1(y_true, majority_pred, gt_labels),
                "exact_match_rate": sum(row["label_match"] for row in group) / len(group) if group else None,
            }
        )
        out.append(row)
    out.extend(macro_actor_rows(out, ["label_macro_f1", "majority_label_macro_f1", "exact_match_rate"], include_granularity=True))
    return out


def macro_actor_rows(rows: list[dict], metrics: Sequence[str], include_granularity: bool = False) -> list[dict]:
    out = []
    base_keys = ["method", "method_label", "variant_label", "model_short", "clip_level", "point_source"]
    if include_granularity:
        base_keys.append("granularity")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("actor") in ACTORS:
            groups[tuple(row[k] for k in base_keys)].append(row)
    for key, group in sorted(groups.items()):
        item = dict(zip(base_keys, key))
        item["actor"] = "macro_actor"
        item["n"] = sum(int(row.get("n") or 0) for row in group)
        for metric in metrics:
            vals = [row.get(metric) for row in group if row.get(metric) is not None and not math.isnan(float(row.get(metric)))]
            item[metric] = mean(vals) if vals else None
        out.append(item)
    return out


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def inventory(records: Sequence[dict], frames_by_method: dict, eval_rows: Sequence[dict]) -> dict:
    gt_counts = defaultdict(Counter)
    for record in records:
        for actor in ACTORS:
            for row in record.get(actor, []) or []:
                present, excluded, raw = presence_call(actor, row, is_gt=True)
                bucket = "excluded_uncertain" if excluded else ("unknown" if present is None else ("present" if present else "absent"))
                gt_counts[(record.get("clip_level", "coarse"), actor)][bucket] += 1
                gt_counts[(record.get("clip_level", "coarse"), actor)][f"value:{raw}"] += 1

    pred_counts = defaultdict(Counter)
    for row in eval_rows:
        key = (row["method_label"], row["clip_level"], row["point_source"], row["actor"])
        if row["pred_presence_unknown"]:
            pred_counts[key]["unknown"] += 1
        elif row["pred_present"]:
            pred_counts[key]["present"] += 1
        else:
            pred_counts[key]["absent"] += 1
        pred_counts[key][f"value:{row['pred_presence_raw']}"] += 1

    return {
        "n_methods": len(frames_by_method),
        "videos_by_method": {method: sorted(by_video) for method, by_video in sorted(frames_by_method.items())},
        "gt_presence_counts": {"/".join(key): dict(counts) for key, counts in sorted(gt_counts.items())},
        "prediction_presence_counts": {"/".join(key): dict(counts) for key, counts in sorted(pred_counts.items())},
        "absence_schema": {
            "left_absent": sorted(str(x) for x in LEFT_ABSENT if x is not None),
            "right_absent": sorted(str(x) for x in RIGHT_ABSENT_VALUES if x is not None),
            "camera_absent": sorted(str(x) for x in CAMERA_ABSENT if x is not None),
            "camera_present": sorted(CAMERA_PRESENT),
            "camera_gt_excluded": sorted(CAMERA_EXCLUDE_ON_GT),
        },
        "prediction_format_can_express_absence": {
            "left": "yes: missing/omitted left slot is parsed as '(not set)', and explicit '(not set)' is representable",
            "right": "yes: missing/omitted right slot is parsed as an empty '(not set)' tuple, and explicit empty tuple is representable",
            "camera": "yes: CAMERA_NO_CHANGE is an allowed camera action; CAMERA_UNCERTAIN is treated as absent for prediction presence",
        },
        "horizon_verification": "unverified_from_existing_traces",
    }


def write_report(path: Path, inv: dict, presence: list[dict], labels: list[dict]) -> None:
    def top_rows(rows: list[dict], metric: str, n: int = 12) -> list[dict]:
        usable = [row for row in rows if row.get(metric) is not None and row.get("actor") != "macro_actor"]
        return sorted(usable, key=lambda row: float(row[metric]), reverse=True)[:n]

    def md_table(rows: list[dict], cols: Sequence[str]) -> str:
        if not rows:
            return "(empty)"
        widths = {col: max(len(col), *(len(fv(row.get(col))) for row in rows)) for col in cols}
        lines = ["| " + " | ".join(col.ljust(widths[col]) for col in cols) + " |"]
        lines.append("| " + " | ".join("-" * widths[col] for col in cols) + " |")
        for row in rows:
            lines.append("| " + " | ".join(fv(row.get(col)).ljust(widths[col]) for col in cols) + " |")
        return "\n".join(lines)

    lines = [
        "# CVS-Act v1 Onset-Aligned Pointwise Action Recommendation Eval",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Methods**: {inv['n_methods']}",
        f"**Horizon verification**: {inv['horizon_verification']}",
        "",
        "## Inventory and Warnings",
        "",
        "Prediction format can express absent values, but the existing prompts/runs may still be behaviorally biased toward non-empty actor slots. Presence metrics are therefore descriptive and should be interpreted against the always-present null.",
        "",
        "Information horizon could not be verified from existing trace metadata in this script. Any onset-aligned gain remains potentially confounded if a method used frames after the evaluated onset.",
        "",
        "## Headline Questions",
        "",
        "1. Does any method beat the always-present null on presence? Check `presence_summary.csv`; left presence is expected to be near-ceiling/uninformative.",
        "2. Does SurGent exceed CoT on right-hand conditional label macro-F1 across granularities/models? Check `conditional_label_summary.csv` with actor=`right`.",
        "",
        "## Top Presence Balanced Accuracy Rows",
        "",
        md_table(
            top_rows(presence, "balanced_accuracy"),
            ["method_label", "clip_level", "point_source", "actor", "n", "balanced_accuracy", "always_present_balanced_accuracy", "inactive_false_positive_rate"],
        ),
        "",
        "## Top Conditional Label Macro-F1 Rows",
        "",
        md_table(
            top_rows(labels, "label_macro_f1"),
            ["method_label", "clip_level", "point_source", "actor", "granularity", "n", "label_macro_f1", "majority_label_macro_f1", "majority_label"],
        ),
    ]
    path.write_text("\n".join(lines) + "\n")


def fv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    args = parse_args()
    metrics_dir = args.metrics_base if args.metrics_base.is_absolute() else ROOT_DIR / args.metrics_base
    metrics_dir.mkdir(parents=True, exist_ok=True)

    records = load_audit_v11_action_records(args.audit_actions_dir, "both")
    gt_by_video = gt_records_by_video(records)
    frames_by_method, paths_by_method, skipped = collect_prediction_groups(
        [path if path.is_absolute() else ROOT_DIR / path for path in args.output_dirs],
        gt_by_video,
        args.include_partial,
        None,
        set(args.models) if args.models else None,
        set(args.taxonomy) if args.taxonomy else None,
        args.verbose,
    )
    if args.common_only and frames_by_method:
        common = set.intersection(*(set(by_video) for by_video in frames_by_method.values()))
        frames_by_method = {
            method: {video_id: frames for video_id, frames in by_video.items() if video_id in common}
            for method, by_video in frames_by_method.items()
        }
        paths_by_method = {
            method: {video_id: path for video_id, path in by_video.items() if video_id in common}
            for method, by_video in paths_by_method.items()
        }
        print(f"Common videos across {len(frames_by_method)} group(s): {len(common)}")

    eval_points = collect_eval_points(records)
    eval_rows = evaluate_points(frames_by_method, eval_points)
    presence = summarize_presence(eval_rows)
    labels = summarize_labels(eval_rows)
    inv = inventory(records, frames_by_method, eval_rows)
    inv["skipped"] = skipped
    inv["paths_by_method"] = paths_by_method
    inv["n_eval_points_defined"] = len(eval_points)
    inv["n_eval_rows"] = len(eval_rows)

    write_csv(metrics_dir / "onset_pointwise_details.csv", eval_rows)
    write_csv(metrics_dir / "presence_summary.csv", presence)
    write_csv(metrics_dir / "conditional_label_summary.csv", labels)
    write_json(metrics_dir / "inventory.json", inv)
    write_report(metrics_dir / "onset_pointwise_report.md", inv, presence, labels)

    print("Absent/present mapping:")
    print(json.dumps(inv["absence_schema"], indent=2))
    print()
    print(f"Methods: {inv['n_methods']}")
    print(f"Defined evaluation points: {inv['n_eval_points_defined']}")
    print(f"Evaluated rows: {inv['n_eval_rows']}")
    print(f"Horizon verification: {inv['horizon_verification']}")
    print(f"Wrote {metrics_dir / 'onset_pointwise_details.csv'}")
    print(f"Wrote {metrics_dir / 'presence_summary.csv'}")
    print(f"Wrote {metrics_dir / 'conditional_label_summary.csv'}")
    print(f"Wrote {metrics_dir / 'inventory.json'}")
    print(f"Wrote {metrics_dir / 'onset_pointwise_report.md'}")


if __name__ == "__main__":
    main()
