"""Evaluation helpers for CVS-Act recommendation outputs."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_FINAL_SCORE_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("left", "exact"),
    ("right", "medium"),
    ("camera", "coarse"),
)


def bool_value(value: Any) -> bool:
    """Interpret CSV/JSON boolean-like values consistently across notebooks."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", "", "nan", "none"}:
        return False
    return bool(value)


def multiclass_macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Macro-F1 over the union of true and predicted labels."""
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return math.nan
    scores = []
    for label in labels:
        tp = sum((true == label) and (pred == label) for true, pred in zip(y_true, y_pred))
        fp = sum((true != label) and (pred == label) for true, pred in zip(y_true, y_pred))
        fn = sum((true == label) and (pred != label) for true, pred in zip(y_true, y_pred))
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return mean(scores)


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def final_score_from_component_scores(component_scores: Mapping[str, Any]) -> float:
    """Average component scores only when every requested component is finite."""
    values = list(component_scores.values())
    if not values or any(not is_finite_number(value) for value in values):
        return math.nan
    return mean(float(value) for value in values)


def score_onset_pointwise_final_components(
    rows: Sequence[Mapping[str, Any]],
    *,
    components: Sequence[tuple[str, str]] = DEFAULT_FINAL_SCORE_COMPONENTS,
    group_keys: Sequence[str] = ("method", "method_label", "variant_label", "model_short"),
    clip_level: str = "coarse",
    point_source: str = "coarse_actor_onset",
    video_ids: Iterable[str] | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compute the CVS-Act final score used in recent analysis notebooks.

    The score is ``mean(left/exact, right/medium, camera/coarse)``. Each
    component is the average clip-level conditional multiclass macro-F1 over
    GT-present, non-unknown/non-uncertain onset rows for that actor and
    granularity.
    """
    video_filter = {str(video_id) for video_id in video_ids} if video_ids is not None else None
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if video_filter is not None and str(row.get("video_id")) not in video_filter:
            continue
        grouped[tuple(row.get(key) for key in group_keys)].append(row)

    out: list[dict[str, Any]] = []
    def sort_key(item: tuple[tuple[Any, ...], list[Mapping[str, Any]]]) -> tuple[str, ...]:
        return tuple("" if value is None else str(value) for value in item[0])

    for key, group_rows in sorted(grouped.items(), key=sort_key):
        item = dict(zip(group_keys, key))
        if extra_fields:
            item.update(extra_fields)
        component_scores: dict[str, float] = {}
        component_ns: dict[str, int] = {}
        component_clip_ns: dict[str, int] = {}
        for actor, granularity in components:
            usable = [
                row
                for row in group_rows
                if row.get("clip_level") == clip_level
                and row.get("point_source") == point_source
                and row.get("actor") == actor
                and bool_value(row.get("gt_present"))
                and not bool_value(row.get("gt_presence_excluded_uncertain"))
                and not bool_value(row.get("gt_presence_unknown"))
            ]
            gt_col = f"gt_label_resolved_{granularity}"
            pred_col = f"pred_label_{granularity}"
            pairs_by_clip: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for row in usable:
                if row.get(gt_col) is None or row.get(pred_col) is None:
                    continue
                pairs_by_clip[str(row.get("video_id"))].append(
                    (str(row.get(gt_col)), str(row.get(pred_col)))
                )
            key_name = f"{actor}_{granularity}"
            clip_scores = [
                multiclass_macro_f1([true for true, _ in pairs], [pred for _, pred in pairs])
                for pairs in pairs_by_clip.values()
                if pairs
            ]
            component_scores[key_name] = mean(clip_scores) if clip_scores else math.nan
            component_ns[f"{key_name}_n"] = sum(len(pairs) for pairs in pairs_by_clip.values())
            component_clip_ns[f"{key_name}_clips_n"] = len(clip_scores)

        finite_components = [
            value for value in component_scores.values() if is_finite_number(value)
        ]
        item.update(component_scores)
        item.update(component_ns)
        item.update(component_clip_ns)
        item.update(
            {
                "final_score": final_score_from_component_scores(component_scores),
                "n_components": len(finite_components),
                "clip_level": clip_level,
                "point_source": point_source,
                "component_policy": "+".join(f"{actor}/{granularity}" for actor, granularity in components),
            }
        )
        out.append(item)
    return out
