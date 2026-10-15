import math

from cvs_act.action_segment_eval import interval_anchored_f1


def _left(start, end, code):
    return {
        "start_frame": start,
        "end_frame": end,
        "retraction_direction_code": code,
    }


def test_any_overlap_reports_both_anchor_directions():
    gt = [_left(0, 100, "KEEP_RETRACT_LATERAL")]
    pred = [
        _left(0, 10, "KEEP_RETRACT_LATERAL"),
        _left(10, 20, "KEEP_RETRACT_MEDIAL"),
    ]

    metric = interval_anchored_f1(pred, gt, "left", "exact", "any")

    assert metric["pred_anchor_accuracy"] == 0.5
    assert metric["gt_anchor_accuracy"] == 1.0
    assert math.isclose(metric["f1"], 2 / 3)


def test_touching_boundaries_are_not_positive_overlap():
    gt = [_left(10, 20, "KEEP_RETRACT_LATERAL")]
    pred = [_left(0, 10, "KEEP_RETRACT_LATERAL")]

    metric = interval_anchored_f1(pred, gt, "left", "exact", "any")

    assert metric["f1"] == 0.0


def test_max_overlap_accumulates_disconnected_same_label_intervals():
    gt = [_left(0, 100, "KEEP_RETRACT_LATERAL")]
    pred = [
        _left(0, 25, "KEEP_RETRACT_LATERAL"),
        _left(25, 65, "KEEP_RETRACT_MEDIAL"),
        _left(65, 90, "KEEP_RETRACT_LATERAL"),
    ]

    metric = interval_anchored_f1(pred, gt, "left", "exact", "max")

    # GT anchor: lateral wins 50 accumulated frames over medial's 40.
    assert metric["gt_anchor_accuracy"] == 1.0
    # Pred anchors: two lateral intervals match; the medial interval does not.
    assert math.isclose(metric["pred_anchor_accuracy"], 2 / 3)
    assert math.isclose(metric["f1"], 0.8)


def test_max_overlap_does_not_double_count_overlapping_same_label_intervals():
    gt = [_left(0, 100, "KEEP_RETRACT_LATERAL")]
    pred = [
        _left(0, 30, "KEEP_RETRACT_LATERAL"),
        _left(10, 40, "KEEP_RETRACT_LATERAL"),
        _left(40, 85, "KEEP_RETRACT_MEDIAL"),
    ]

    metric = interval_anchored_f1(pred, gt, "left", "exact", "max")

    # Lateral union is 40 frames, not 60, so medial's 45 frames wins.
    assert metric["gt_anchor_accuracy"] == 0.0
