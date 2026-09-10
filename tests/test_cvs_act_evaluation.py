import math

from cvs_act.evaluation import score_onset_pointwise_final_components


def _row(
    method,
    actor,
    exact,
    medium,
    coarse,
    pred_exact,
    pred_medium,
    pred_coarse,
    *,
    present=True,
    video_id="clip1",
):
    return {
        "video_id": video_id,
        "method": method,
        "method_label": method,
        "variant_label": "variant",
        "model_short": "model",
        "clip_level": "coarse",
        "point_source": "coarse_actor_onset",
        "actor": actor,
        "gt_present": present,
        "gt_presence_excluded_uncertain": False,
        "gt_presence_unknown": False,
        "gt_label_resolved_exact": exact,
        "gt_label_resolved_medium": medium,
        "gt_label_resolved_coarse": coarse,
        "pred_label_exact": pred_exact,
        "pred_label_medium": pred_medium,
        "pred_label_coarse": pred_coarse,
    }


def test_score_onset_pointwise_final_components_uses_notebook_policy():
    rows = [
        _row("m1", "left", "L1", "LM1", "LC1", "L1", "LM1", "LC1"),
        _row("m1", "left", "L2", "LM2", "LC2", "wrong", "LM2", "LC2"),
        _row("m1", "right", "R1", "RM1", "RC1", "R1", "RM1", "RC1"),
        _row("m1", "right", "R2", "RM2", "RC2", "R2", "wrong", "RC2"),
        _row("m1", "camera", "C1", "CM1", "CC1", "C1", "CM1", "CC1"),
        _row("m1", "camera", "C2", "CM2", "CC2", "C2", "CM2", "wrong"),
        _row("m1", "left", "absent", "absent", "absent", "wrong", "wrong", "wrong", present=False),
    ]

    [score] = score_onset_pointwise_final_components(rows)

    assert score["left_exact"] == 1 / 3
    assert score["right_medium"] == 1 / 3
    assert score["camera_coarse"] == 1 / 3
    assert score["left_exact_n"] == 2
    assert score["right_medium_n"] == 2
    assert score["camera_coarse_n"] == 2
    assert score["left_exact_clips_n"] == 1
    assert score["right_medium_clips_n"] == 1
    assert score["camera_coarse_clips_n"] == 1
    assert math.isclose(score["final_score"], 1 / 3)
    assert score["component_policy"] == "left/exact+right/medium+camera/coarse"


def test_score_onset_pointwise_final_components_averages_over_clips():
    rows = [
        _row("m1", "left", "A", "A", "A", "A", "A", "A", video_id="clip1"),
        _row("m1", "left", "B", "B", "B", "wrong", "wrong", "wrong", video_id="clip2"),
    ]

    [score] = score_onset_pointwise_final_components(
        rows,
        components=(("left", "exact"),),
    )

    assert math.isclose(score["left_exact"], 0.5)
    assert score["left_exact_n"] == 2
    assert score["left_exact_clips_n"] == 2
    assert math.isclose(score["final_score"], 0.5)
