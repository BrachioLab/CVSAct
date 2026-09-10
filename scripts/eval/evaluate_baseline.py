#!/usr/bin/env python3
"""Evaluate a baseline JSONL prediction file against an annotation JSON file.

Computes:
  1. Frame-level and video-level CVS mAP (average precision per criterion)
  2. Fuzzy-match precision/recall on auto-generated sentences (whole sentence)
  3. Binary precision/recall/F1 with a fuzzy threshold on core-field sentences
  4. Per-field hierarchical accuracy (tool_type, action_code, target_structure, intention)

Results are auto-saved to metrics/<pred_dir_name>/<pred_filename>.json.

Usage:
    python scripts/eval/evaluate_baseline.py \
        --pred outputs/baseline/VIDEO__direct__model.jsonl \
        --gt data/processed/CVS_Challenge_SAGES_v1/cvs_act_annotations/v1/train/audit_v9/VIDEO.json \
        [--threshold 0.70] [--no-normalize] [--verbose] [--json-out results.json]
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Resolve paths relative to repo root
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# ---------------------------------------------------------------------------
# Load taxonomy v9
# ---------------------------------------------------------------------------
TAXONOMY_PATH = os.path.join(
    ROOT_DIR,
    "data/processed/CVS_Challenge_SAGES_v1/cvs_act_annotations/v1/taxonomy_v9.json",
)
with open(TAXONOMY_PATH) as f:
    TAXONOMY = json.load(f)

ACTOR_TEXT = TAXONOMY["nl_mappings"]["actor_text"]
TOOL_TEXT = TAXONOMY["nl_mappings"]["tool_text"]
ACTION_TEXT = TAXONOMY["nl_mappings"]["action_text"]
TARGET_TEXT = TAXONOMY["nl_mappings"]["target_text"]
RETRACT_DIRECTION = TAXONOMY["nl_mappings"]["retract_direction"]

# ---------------------------------------------------------------------------
# Load normalization map
# ---------------------------------------------------------------------------
NORM_MAP_PATH = os.path.join(
    ROOT_DIR,
    "data/processed/CVS_Challenge_SAGES_v1/cvs_act_annotations/v1/normalization_map_v9.json",
)
NORM_MAP = {}
if os.path.isfile(NORM_MAP_PATH):
    with open(NORM_MAP_PATH) as f:
        NORM_MAP = json.load(f).get("normalization_map", {})

NORM_FIELDS = ["actor_role", "tool_type", "action_code", "target_structure", "intention"]


def _canonicalize_action_dict(action):
    out = dict(action) if isinstance(action, dict) else {}
    actor_role = out.get("actor_role")
    action_code = out.get("action_code")
    if action_code == "ICG_SWITCH" and actor_role in (None, "", "(not set)", "unknown"):
        out["actor_role"] = "other"
    return out

# ---------------------------------------------------------------------------
# Sentence generation (mirrors notebook cell 7)
# ---------------------------------------------------------------------------

def generate_sentence(actor_role, tool_type, action_code, target_structure,
                      intention, extra_info,
                      target_context_1="(not set)", target_context_2="(not set)"):
    actor = ACTOR_TEXT.get(actor_role, actor_role)
    tool = TOOL_TEXT.get(
        tool_type, tool_type.lower() if tool_type and tool_type != "(not set)" else ""
    )
    actor_part = f"{actor} ({tool})" if tool else actor

    target = TARGET_TEXT.get(
        target_structure, target_structure if target_structure != "(not set)" else ""
    )

    if action_code in RETRACT_DIRECTION:
        direction = RETRACT_DIRECTION[action_code]
        action_target = f"retracts {target} {direction}" if target else f"retracts {direction}"
    else:
        verb = ACTION_TEXT.get(
            action_code,
            action_code.lower().replace("_", " ")
            if action_code and action_code != "(not set)"
            else "?",
        )
        action_target = f"{verb} {target}" if target else verb

    s = f"{actor_part} {action_target}"
    ctx_parts = []
    if target_context_1 and target_context_1 != "(not set)":
        ctx_parts.append(target_context_1)
    if target_context_2 and target_context_2 != "(not set)":
        ctx_parts.append(target_context_2)
    if ctx_parts:
        s += f' ({", ".join(ctx_parts)})'
    if intention and intention not in ("(not set)", ""):
        s += f", to {intention}"
    if extra_info and extra_info not in ("(not set)", ""):
        s += f" ({extra_info})"
    return s + "."


# ---------------------------------------------------------------------------
# Normalization / sentence helpers
# ---------------------------------------------------------------------------

def normalize_action(action, norm_map):
    out = _canonicalize_action_dict(action)
    for field in NORM_FIELDS:
        if field in out and field in norm_map:
            out[field] = norm_map[field].get(out[field], out[field])
    return out


def _sentence_from_action(a):
    return generate_sentence(
        actor_role=a.get("actor_role", "unknown"),
        tool_type=a.get("tool_type", "(not set)"),
        action_code=a.get("action_code", "(not set)"),
        target_structure=a.get("target_structure", "(not set)"),
        intention=a.get("intention", "(not set)"),
        extra_info=a.get("extra_info", ""),
        target_context_1=a.get("target_context_1", "(not set)"),
        target_context_2=a.get("target_context_2", "(not set)"),
    )


def _core_sentence(a):
    """Sentence from core fields only (no target_context, no extra_info)."""
    return generate_sentence(
        actor_role=a.get("actor_role", "unknown"),
        tool_type=a.get("tool_type", "(not set)"),
        action_code=a.get("action_code", "(not set)"),
        target_structure=a.get("target_structure", "(not set)"),
        intention=a.get("intention", "(not set)"),
        extra_info="",
        target_context_1="(not set)",
        target_context_2="(not set)",
    )


def _sentences(actions):
    return [_sentence_from_action(a) for a in actions]


def _core_sentences(actions):
    return [_core_sentence(a) for a in actions]


# ---------------------------------------------------------------------------
# Fuzzy matching (uses fuzzywuzzy)
# ---------------------------------------------------------------------------
try:
    from fuzzywuzzy import fuzz as fwfuzz
except ImportError:
    sys.exit("ERROR: fuzzywuzzy is required. Install with: pip install fuzzywuzzy")


def compute_fuzzy_precision_recall(pred_sentences, gt_sentences):
    """Soft fuzzy precision/recall (average best-match ratios)."""
    if not pred_sentences and not gt_sentences:
        return 1.0, 1.0, [], []
    if not pred_sentences:
        return float("nan"), 0.0, [], [0.0] * len(gt_sentences)
    if not gt_sentences:
        return 0.0, float("nan"), [0.0] * len(pred_sentences), []

    prec_scores = []
    for pred in pred_sentences:
        best = max(fwfuzz.ratio(pred, gt) for gt in gt_sentences) / 100.0
        prec_scores.append(best)

    recall_scores = []
    for gt in gt_sentences:
        best = max(fwfuzz.ratio(gt, pred) for pred in pred_sentences) / 100.0
        recall_scores.append(best)

    return float(np.mean(prec_scores)), float(np.mean(recall_scores)), prec_scores, recall_scores


def compute_binary_precision_recall(pred_sents, gt_sents, threshold=0.70):
    """Binary thresholded precision/recall on fuzzy-matched sentences."""
    if not pred_sents and not gt_sents:
        return 1.0, 1.0, 0, 0, 0, 0

    tp_rec, fn = 0, 0
    for gs in gt_sents:
        if pred_sents:
            best = max(fwfuzz.ratio(gs, ps) for ps in pred_sents) / 100.0
        else:
            best = 0.0
        if best >= threshold:
            tp_rec += 1
        else:
            fn += 1

    tp_prec, fp = 0, 0
    for ps in pred_sents:
        if gt_sents:
            best = max(fwfuzz.ratio(ps, gs) for gs in gt_sents) / 100.0
        else:
            best = 0.0
        if best >= threshold:
            tp_prec += 1
        else:
            fp += 1

    prec = tp_prec / (tp_prec + fp) if (tp_prec + fp) > 0 else 0.0
    rec = tp_rec / (tp_rec + fn) if (tp_rec + fn) > 0 else 0.0
    return prec, rec, tp_prec, fp, tp_rec, fn


# ---------------------------------------------------------------------------
# Hierarchical per-field accuracy
# ---------------------------------------------------------------------------
HIER_FIELDS = ["tool_type", "action_code", "target_structure", "intention"]

ACTOR_GROUPS = {
    "Left": ["left_instrument"],
    "Right": ["right_instrument"],
    "Camera": ["camera"],
    "Other": ["other"],
}


def _best_match_per_gt(pred_actions, gt_actions, norm_map=None):
    if norm_map:
        pred_use = [normalize_action(a, norm_map) for a in pred_actions]
        gt_use = [normalize_action(a, norm_map) for a in gt_actions]
    else:
        pred_use = list(pred_actions)
        gt_use = list(gt_actions)

    pred_sents = _core_sentences(pred_use) if pred_use else []
    gt_sents = _core_sentences(gt_use)

    pairs = []
    for gi, (gt_a, gs) in enumerate(zip(gt_use, gt_sents)):
        if not pred_sents:
            pairs.append((gt_a, None))
            continue
        best_score, best_pi = -1, -1
        for pi, ps in enumerate(pred_sents):
            s = fwfuzz.ratio(gs, ps) / 100.0
            if s > best_score:
                best_score = s
                best_pi = pi
        pairs.append((gt_a, pred_use[best_pi]))
    return pairs


def _best_match_per_pred(pred_actions, gt_actions, norm_map=None):
    if norm_map:
        pred_use = [normalize_action(a, norm_map) for a in pred_actions]
        gt_use = [normalize_action(a, norm_map) for a in gt_actions]
    else:
        pred_use = list(pred_actions)
        gt_use = list(gt_actions)

    pred_sents = _core_sentences(pred_use)
    gt_sents = _core_sentences(gt_use) if gt_use else []

    pairs = []
    for pi, (pred_a, ps) in enumerate(zip(pred_use, pred_sents)):
        if not gt_sents:
            pairs.append((pred_a, None))
            continue
        best_score, best_gi = -1, -1
        for gi, gs in enumerate(gt_sents):
            s = fwfuzz.ratio(ps, gs) / 100.0
            if s > best_score:
                best_score = s
                best_gi = gi
        pairs.append((pred_a, gt_use[best_gi]))
    return pairs


def compute_hierarchical_accuracy(results, norm_map=None, actor_filter=None):
    """Per-field, per-class recall and precision on all GT/pred actions.

    Returns field_stats[field] -> {recall: {micro, macro, class_acc}, precision: ...}
    """
    recall_matches = {f: defaultdict(list) for f in HIER_FIELDS}
    prec_matches = {f: defaultdict(list) for f in HIER_FIELDS}
    n_gt_total = 0
    n_pred_total = 0

    for r in results:
        if actor_filter is not None:
            gt_acts = [a for a in r["gt_actions"] if a.get("actor_role") in actor_filter]
            pred_acts = [a for a in r["pred_actions"] if a.get("actor_role") in actor_filter]
        else:
            gt_acts = r["gt_actions"]
            pred_acts = r["pred_actions"]

        n_gt_total += len(gt_acts)
        n_pred_total += len(pred_acts)

        gt_pairs = _best_match_per_gt(pred_acts, gt_acts, norm_map=norm_map)
        for gt_a, best_pred in gt_pairs:
            for field in HIER_FIELDS:
                gt_val = gt_a.get(field, "(not set)")
                if best_pred is None:
                    recall_matches[field][gt_val].append(0)
                else:
                    pred_val = best_pred.get(field, "(not set)")
                    recall_matches[field][gt_val].append(1 if gt_val == pred_val else 0)

        pred_pairs = _best_match_per_pred(pred_acts, gt_acts, norm_map=norm_map)
        for pred_a, best_gt in pred_pairs:
            for field in HIER_FIELDS:
                pred_val = pred_a.get(field, "(not set)")
                if best_gt is None:
                    prec_matches[field][pred_val].append(0)
                else:
                    gt_val = best_gt.get(field, "(not set)")
                    prec_matches[field][pred_val].append(1 if pred_val == gt_val else 0)

    def _aggregate(field_matches):
        class_acc = {}
        total_correct = 0
        total_all = 0
        for cls_val, matches in sorted(field_matches.items()):
            correct = sum(matches)
            total = len(matches)
            class_acc[cls_val] = {
                "correct": correct,
                "total": total,
                "acc": correct / total if total > 0 else 0.0,
            }
            total_correct += correct
            total_all += total
        micro = total_correct / total_all if total_all > 0 else 0.0
        per_class = [v["acc"] for v in class_acc.values()]
        macro = float(np.mean(per_class)) if per_class else 0.0
        return {"micro": micro, "macro": macro, "class_acc": class_acc}

    field_stats = {}
    for field in HIER_FIELDS:
        field_stats[field] = {
            "recall": _aggregate(recall_matches[field]),
            "precision": _aggregate(prec_matches[field]),
        }

    return field_stats, n_gt_total, n_pred_total


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def parse_frame_number(frame_id_str):
    """Extract integer frame number from e.g. 'frame_000150' -> 150."""
    m = re.search(r"(\d+)", frame_id_str)
    return int(m.group(1)) if m else -1


def load_predictions(pred_path, action_sources=None):
    """Load JSONL predictions -> list of normalized dicts (one per frame).

    Handles both baseline.v1 (frame_id, actions_ranked) and
    surgent sequential (current_frame_id, scene_cvs_output, action_rec_output,
    predicted_taxonomy_action).

    For surgent sequential outputs:
      - frame_id: prefers labeled_source_frame_id (pattern-aligned) for reliable
        matching against CVS labels and annotations, falls back to current_frame_id.
      - actions_ranked: merges scene_cvs_output.actions_ranked and
        action_rec_output.actions_ranked (actual tool outputs), falling back to
        predicted_taxonomy_action if neither is present.
      - pred: uses scene_cvs_output.pred (simple float dict) when available,
        since cvs_predictions stores the same scores in nested format.

    Args:
        action_sources: Optional set/list of source keys to include for surgent
            action merging. Valid values: "scene_cvs", "action_rec",
            "instrument_rec" (left/right/camera_rec_output).
            If None, all sources are included (default behavior).
            Only affects surgent files that lack pre-built actions_ranked.
    """
    if action_sources is not None:
        action_sources = set(action_sources)

    preds = []
    with open(pred_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARNING: skipping bad JSON at {pred_path} line {line_num}: {e}")
                continue

            is_surgent = "current_frame_id" in d or "scene_cvs_output" in d

            # Normalize frame_id
            if "frame_id" not in d:
                # For surgent: prefer labeled_source_frame_id (pattern-aligned
                # keyframe number) for matching against annotation frame ranges
                # and CVS label frame IDs.
                lsf = d.get("labeled_source_frame_id")
                if lsf is not None:
                    d["frame_id"] = f"frame_{int(lsf):06d}"
                elif "current_frame_id" in d:
                    d["frame_id"] = d["current_frame_id"]

            # Normalize pred (simple {c1: float, c2: float, c3: float} dict)
            if "pred" not in d and is_surgent:
                sco = d.get("scene_cvs_output")
                if isinstance(sco, dict):
                    sco_pred = sco.get("pred")
                    if isinstance(sco_pred, dict):
                        d["pred"] = sco_pred

            # Normalize actions_ranked
            # For surgent files, always rebuild from sources when action_sources
            # is specified (to allow filtering), even if actions_ranked exists.
            rebuild = "actions_ranked" not in d or (is_surgent and action_sources is not None)
            if rebuild:
                merged_actions = []
                seen_keys = set()

                def _add_action(a):
                    if not isinstance(a, dict) or not a:
                        return
                    a = _canonicalize_action_dict(a)
                    dedup_key = (
                        a.get("actor_role", ""),
                        a.get("tool_type", ""),
                        a.get("action_code", ""),
                        a.get("target_structure", ""),
                        a.get("intention", ""),
                    )
                    if dedup_key not in seen_keys:
                        seen_keys.add(dedup_key)
                        merged_actions.append(a)

                # Sources with actions_ranked (list of actions)
                source_map = {
                    "scene_cvs_output": "scene_cvs",
                    "action_rec_output": "action_rec",
                }
                for source_key, source_tag in source_map.items():
                    if action_sources is not None and source_tag not in action_sources:
                        continue
                    source = d.get(source_key)
                    if not isinstance(source, dict):
                        continue
                    ar = source.get("actions_ranked", [])
                    if not isinstance(ar, list):
                        continue
                    for a in ar:
                        _add_action(a)

                # Sources with single action dict (instrument-specific recs)
                if action_sources is None or "instrument_rec" in action_sources:
                    for source_key in ("left_rec_output", "right_rec_output", "camera_rec_output"):
                        source = d.get(source_key)
                        if not isinstance(source, dict):
                            continue
                        _add_action(source.get("action"))

                # Fall back to predicted_taxonomy_action if no tool outputs
                if not merged_actions:
                    pta = d.get("predicted_taxonomy_action")
                    if isinstance(pta, dict):
                        merged_actions = [_canonicalize_action_dict(pta)]
                    elif isinstance(pta, list):
                        merged_actions = [_canonicalize_action_dict(a) for a in pta if isinstance(a, dict)]

                d["actions_ranked"] = merged_actions
            elif isinstance(d.get("actions_ranked"), list):
                d["actions_ranked"] = [
                    _canonicalize_action_dict(a) for a in d["actions_ranked"] if isinstance(a, dict)
                ]

            preds.append(d)
    return preds


def load_annotations(gt_path):
    """Load annotation JSON -> list of annotation records."""
    with open(gt_path) as f:
        return json.load(f)


def collect_pred_actions_in_range(pred_frames, start_frame, end_frame):
    """Collect predicted actions from the prediction frame closest to start_frame.

    The annotation range [start_frame, end_frame] describes an action at the
    transition point (start_frame).  We evaluate only the prediction at (or
    nearest to) that start frame, not the union of all frames in the range.
    """
    # Find the prediction frame closest to start_frame within [start_frame, end_frame]
    best_pf = None
    best_dist = float("inf")
    for pf in pred_frames:
        fnum = parse_frame_number(pf["frame_id"])
        if fnum < start_frame or fnum > end_frame:
            continue
        dist = abs(fnum - start_frame)
        if dist < best_dist:
            best_dist = dist
            best_pf = pf
    if best_pf is None:
        return []
    return list(best_pf.get("actions_ranked", []))


# ---------------------------------------------------------------------------
# CVS label loading and mAP computation
# ---------------------------------------------------------------------------
CRITERIA = ["c1", "c2", "c3"]
RATERS = ["rater1", "rater2", "rater3"]

# Default labels directories – try train then test
LABELS_DIRS = [
    os.path.join(ROOT_DIR, "data/raw/CVS_Challenge_SAGES_v1/train/labels"),
    os.path.join(ROOT_DIR, "data/raw/CVS_Challenge_SAGES_v1/test/labels"),
]
LABELS_DIR = LABELS_DIRS[0]  # legacy single-dir default


def extract_pred_cvs(frame_dict):
    """Extract (frame_number, {c1, c2, c3}) from a prediction frame dict.

    Handles both baseline.v1 (pred.c1) and surgent sequential (cvs_predictions.c1.score).
    """
    # frame id
    fid = frame_dict.get("frame_id") or frame_dict.get("current_frame_id")
    fnum = parse_frame_number(fid) if fid else -1

    # baseline.v1: pred: {c1: 0.5, c2: 0.3, c3: 0.8}
    pred = frame_dict.get("pred")
    if pred and isinstance(pred, dict) and "c1" in pred:
        scores = {}
        for c in CRITERIA:
            v = pred.get(c)
            if v is None:
                return fnum, None
            scores[c] = float(v)
        return fnum, scores

    # surgent sequential: cvs_predictions: {c1: {score: 0.5}, ...}
    cvs = frame_dict.get("cvs_predictions")
    if cvs and isinstance(cvs, dict):
        scores = {}
        for c in CRITERIA:
            entry = cvs.get(c, {})
            if isinstance(entry, dict) and "score" in entry:
                scores[c] = float(entry["score"])
            else:
                scores[c] = 0.0
        return fnum, scores

    return fnum, None


def _resolve_labels_dir(video_id, labels_dir=None):
    """Find the labels directory that contains this video's data."""
    if labels_dir:
        return labels_dir
    for ld in LABELS_DIRS:
        if os.path.isdir(os.path.join(ld, video_id)):
            return ld
    return LABELS_DIR  # fallback


def load_frame_labels(video_id, labels_dir=None):
    """Load frame.csv for a video. Returns {frame_number: {c1: avg, c2: avg, c3: avg}}."""
    ld = _resolve_labels_dir(video_id, labels_dir)
    frame_csv = os.path.join(ld, video_id, "frame.csv")
    if not os.path.isfile(frame_csv):
        return None
    labels = {}
    with open(frame_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fnum = int(row["frame_id"])
            scores = {}
            for c in CRITERIA:
                vals = [int(row[f"{c}_{r}"]) for r in RATERS]
                scores[c] = sum(vals) / len(vals)
            labels[fnum] = scores
    return labels


def load_video_labels(video_id, labels_dir=None):
    """Load video.csv for a video. Returns {c1: avg, c2: avg, c3: avg}."""
    ld = _resolve_labels_dir(video_id, labels_dir)
    video_csv = os.path.join(ld, video_id, "video.csv")
    if not os.path.isfile(video_csv):
        return None
    with open(video_csv) as f:
        reader = csv.DictReader(f)
        row = next(reader)
    scores = {}
    for c in CRITERIA:
        vals = [int(row[f"{c}_{r}"]) for r in RATERS]
        scores[c] = sum(vals) / len(vals)
    return scores


def compute_ap(pred_scores, gt_labels):
    """Compute Average Precision for a single criterion.

    pred_scores: list of float (predicted confidence)
    gt_labels: list of float (ground truth, averaged across raters)

    We binarize GT at threshold 0.5 (majority vote), then compute AP.
    """
    if not pred_scores:
        return float("nan")
    gt_binary = [1 if g >= 0.5 else 0 for g in gt_labels]
    if sum(gt_binary) == 0 or sum(gt_binary) == len(gt_binary):
        # All same class — AP is not well-defined; return accuracy
        correct = sum(1 for p, g in zip(pred_scores, gt_binary)
                      if (p >= 0.5) == (g == 1))
        return correct / len(gt_binary)

    # Sort by predicted score descending
    pairs = sorted(zip(pred_scores, gt_binary), key=lambda x: -x[0])
    tp = 0
    precisions = []
    for i, (score, label) in enumerate(pairs):
        if label == 1:
            tp += 1
            precisions.append(tp / (i + 1))
    return sum(precisions) / sum(gt_binary) if precisions else 0.0


def evaluate_cvs(pred_frames, video_id, labels_dir=None, verbose=False):
    """Compute frame-level and video-level CVS metrics for a single video.

    Returns dict with frame_mAP, video_mAP, per-criterion APs, etc. or None if no labels.
    """
    _print = print if verbose else (lambda *a, **k: None)

    frame_labels = load_frame_labels(video_id, labels_dir)
    video_labels = load_video_labels(video_id, labels_dir)

    if frame_labels is None:
        return None

    # Build aligned frame-level predictions and labels
    frame_preds = {}  # fnum -> {c1, c2, c3}
    for pf in pred_frames:
        fnum, scores = extract_pred_cvs(pf)
        if scores is not None and fnum >= 0:
            frame_preds[fnum] = scores

    # Intersect: frames present in both predictions and labels
    common_frames = sorted(set(frame_preds.keys()) & set(frame_labels.keys()))

    result = {"n_frames_pred": len(frame_preds), "n_frames_gt": len(frame_labels),
              "n_frames_common": len(common_frames)}

    if not common_frames:
        result["frame_mAP"] = float("nan")
        result["frame_ap"] = {c: float("nan") for c in CRITERIA}
        result["video_mAP"] = float("nan")
        result["video_ap"] = {c: float("nan") for c in CRITERIA}
        return result

    # Frame-level AP per criterion
    frame_aps = {}
    for c in CRITERIA:
        preds_c = [frame_preds[f][c] for f in common_frames]
        gts_c = [frame_labels[f][c] for f in common_frames]
        frame_aps[c] = compute_ap(preds_c, gts_c)
    frame_map = float(np.mean([v for v in frame_aps.values() if not np.isnan(v)]))

    result["frame_ap"] = frame_aps
    result["frame_mAP"] = frame_map

    _print(f"  Frame-level CVS:  mAP={frame_map:.3f}  "
           f"c1={frame_aps['c1']:.3f}  c2={frame_aps['c2']:.3f}  c3={frame_aps['c3']:.3f}  "
           f"(n={len(common_frames)} frames)")

    # Frame-level binary P/R/F1 per criterion (threshold at 0.5)
    frame_binary = {}
    for c in CRITERIA:
        preds_c = [frame_preds[f][c] for f in common_frames]
        gts_c = [frame_labels[f][c] for f in common_frames]
        pred_bin = [1 if p > 0.5 else 0 for p in preds_c]
        gt_bin = [1 if g >= 0.5 else 0 for g in gts_c]
        tp = sum(p == 1 and g == 1 for p, g in zip(pred_bin, gt_bin))
        fp = sum(p == 1 and g == 0 for p, g in zip(pred_bin, gt_bin))
        fn = sum(p == 0 and g == 1 for p, g in zip(pred_bin, gt_bin))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        frame_binary[c] = {"prec": prec, "rec": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    # Micro-average across criteria
    total_tp = sum(frame_binary[c]["tp"] for c in CRITERIA)
    total_fp = sum(frame_binary[c]["fp"] for c in CRITERIA)
    total_fn = sum(frame_binary[c]["fn"] for c in CRITERIA)
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    result["frame_binary"] = frame_binary
    result["frame_binary_prec"] = micro_prec
    result["frame_binary_rec"] = micro_rec
    result["frame_binary_f1"] = micro_f1

    _print(f"  Frame-level CVS binary (>0.5):  P={micro_prec:.3f}  R={micro_rec:.3f}  F1={micro_f1:.3f}  "
           f"c1=[P={frame_binary['c1']['prec']:.3f} R={frame_binary['c1']['rec']:.3f}]  "
           f"c2=[P={frame_binary['c2']['prec']:.3f} R={frame_binary['c2']['rec']:.3f}]  "
           f"c3=[P={frame_binary['c3']['prec']:.3f} R={frame_binary['c3']['rec']:.3f}]")

    # Video-level: aggregate frame predictions (max across frames)
    video_pred = {}
    for c in CRITERIA:
        video_pred[c] = max(frame_preds[f][c] for f in common_frames)

    video_aps = {}
    if video_labels:
        for c in CRITERIA:
            # Single video: just check binary correctness
            gt_bin = 1 if video_labels[c] >= 0.5 else 0
            pred_bin = 1 if video_pred[c] > 0.5 else 0
            video_aps[c] = 1.0 if pred_bin == gt_bin else 0.0
        video_map = float(np.mean(list(video_aps.values())))
        _print(f"  Video-level CVS:  mAP={video_map:.3f}  "
               f"c1={video_aps['c1']:.3f}  c2={video_aps['c2']:.3f}  c3={video_aps['c3']:.3f}  "
               f"(pred={video_pred}  gt={video_labels})")
    else:
        video_map = float("nan")

    result["video_pred"] = video_pred
    result["video_gt"] = video_labels
    result["video_ap"] = video_aps
    result["video_mAP"] = video_map

    return result


# ---------------------------------------------------------------------------
# Build evaluation records (matching notebook's all_results structure)
# ---------------------------------------------------------------------------

def build_eval_records(pred_frames, annotations):
    """Build list of result dicts for evaluation.

    When multiple criteria have overlapping annotation regions (same granularity
    and frame range), they are grouped together. Each group stores multiple GT
    action sets (one per criterion) so the evaluation can match predictions
    against each GT set separately and take the best (max F1).

    Each record has:
      - gt_action_sets: list of {"criterion": str, "gt_actions": list} dicts
      - gt_actions: GT set from the best-matching criterion (set during eval)
      - pred_actions: predicted actions in the frame range
    """
    from collections import OrderedDict

    # Collect raw entries keyed by (granularity, start_frame, end_frame)
    grouped = OrderedDict()

    for ann in annotations:
        criterion = ann["criterion"].lower()
        video_id = ann["video_id"]
        coarse = ann["coarse"]
        gt_actions = [_canonicalize_action_dict(a) for a in coarse["annotation"]["actions_ranked"]]
        key = ("coarse", coarse["start_frame"], coarse["end_frame"])
        grouped.setdefault(key, {"video_id": video_id, "gt_sets": []})
        grouped[key]["gt_sets"].append({
            "criterion": criterion,
            "gt_actions": gt_actions,
        })

        for fine_region in ann.get("fine", []):
            gt_fine = [_canonicalize_action_dict(a) for a in fine_region["annotation"]["actions_ranked"]]
            fkey = ("fine", fine_region["start_frame"], fine_region["end_frame"])
            grouped.setdefault(fkey, {"video_id": video_id, "gt_sets": []})
            grouped[fkey]["gt_sets"].append({
                "criterion": criterion,
                "gt_actions": gt_fine,
            })

    records = []
    for (granularity, start_frame, end_frame), info in grouped.items():
        pred_actions = collect_pred_actions_in_range(
            pred_frames, start_frame, end_frame
        )
        criteria_list = [gs["criterion"] for gs in info["gt_sets"]]
        records.append({
            "video_id": info["video_id"],
            "criterion": ",".join(sorted(set(criteria_list))),
            "granularity": granularity,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "gt_action_sets": info["gt_sets"],
            "gt_actions": info["gt_sets"][0]["gt_actions"],  # default; overwritten during eval
            "pred_actions": pred_actions,
        })

    return records


# ---------------------------------------------------------------------------
# Evaluation + printing
# ---------------------------------------------------------------------------

def evaluate(records, threshold, use_norm, verbose=False):
    """Run all evaluations. Print results only if verbose. Returns a summary dict."""
    norm_map = NORM_MAP if use_norm else None
    all_output = {}

    def _print(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # ── 1. Per-clip fuzzy + binary metrics ──
    _print("=" * 100)
    _print("PER-CLIP EVALUATION")
    _print("=" * 100)

    clip_rows = []
    for r in records:
        pred_acts = r["pred_actions"]
        gt_action_sets = r.get("gt_action_sets", [{"criterion": r["criterion"], "gt_actions": r["gt_actions"]}])

        # Evaluate against each GT set, pick the one with best strict binary F1
        best_result = None
        best_f1 = -1.0
        best_gt_set = gt_action_sets[0]

        for gt_set in gt_action_sets:
            gt_acts = gt_set["gt_actions"]

            gt_core = _core_sentences(gt_acts)
            pred_core = _core_sentences(pred_acts)
            b_prec, b_rec, tp_p, fp_val, tp_r, fn_val = compute_binary_precision_recall(
                pred_core, gt_core, threshold=threshold
            )
            b_f1 = 2 * b_prec * b_rec / (b_prec + b_rec) if (b_prec + b_rec) > 0 else 0.0

            if b_f1 > best_f1:
                best_f1 = b_f1
                best_gt_set = gt_set
                best_result = {
                    "b_prec": b_prec, "b_rec": b_rec, "b_f1": b_f1,
                    "tp_p": tp_p, "fp": fp_val, "tp_r": tp_r, "fn": fn_val,
                }

        # Use the best GT set for all metrics
        gt_acts = best_gt_set["gt_actions"]
        r["gt_actions"] = gt_acts  # update record for downstream (hierarchical, actor)

        gt_sents = _sentences(gt_acts)
        pred_sents = _sentences(pred_acts)
        fuzzy_prec, fuzzy_rec, _, _ = compute_fuzzy_precision_recall(pred_sents, gt_sents)

        b_prec = best_result["b_prec"]
        b_rec = best_result["b_rec"]
        b_f1 = best_result["b_f1"]
        tp_p = best_result["tp_p"]
        fp = best_result["fp"]
        tp_r = best_result["tp_r"]
        fn = best_result["fn"]

        # With-context binary (uses full sentences including target_context_1/2)
        gt_full = _sentences(gt_acts)
        pred_full = _sentences(pred_acts)
        ctx_b_prec, ctx_b_rec, ctx_tp_p, ctx_fp, ctx_tp_r, ctx_fn = compute_binary_precision_recall(
            pred_full, gt_full, threshold=threshold
        )
        ctx_b_f1 = 2 * ctx_b_prec * ctx_b_rec / (ctx_b_prec + ctx_b_rec) if (ctx_b_prec + ctx_b_rec) > 0 else 0.0

        # Normalized binary (if requested)
        nb_prec = nb_rec = nb_f1 = None
        ntp_p = nfp = ntp_r = nfn = 0
        if norm_map:
            gt_norm = [normalize_action(a, norm_map) for a in gt_acts]
            pred_norm = [normalize_action(a, norm_map) for a in pred_acts]
            gt_norm_core = _core_sentences(gt_norm)
            pred_norm_core = _core_sentences(pred_norm)
            nb_prec, nb_rec, ntp_p, nfp, ntp_r, nfn = compute_binary_precision_recall(
                pred_norm_core, gt_norm_core, threshold=threshold
            )
            nb_f1 = 2 * nb_prec * nb_rec / (nb_prec + nb_rec) if (nb_prec + nb_rec) > 0 else 0.0

        n_gt_sets = len(gt_action_sets)
        best_crit = best_gt_set["criterion"].upper()
        crit_label = r["criterion"].upper()
        gran_label = r["granularity"]
        if n_gt_sets > 1:
            _print(
                f"  {r['video_id'][:12]}.. {crit_label} {gran_label:<6s}  "
                f"fuzzy P={fuzzy_prec:.3f} R={fuzzy_rec:.3f}  "
                f"binary P={b_prec:.3f} R={b_rec:.3f} F1={b_f1:.3f}  "
                f"(gt={len(gt_acts)} pred={len(pred_acts)}) "
                f"[best of {n_gt_sets} GT sets: {best_crit}]"
            )
        else:
            _print(
                f"  {r['video_id'][:12]}.. {crit_label} {gran_label:<6s}  "
                f"fuzzy P={fuzzy_prec:.3f} R={fuzzy_rec:.3f}  "
                f"binary P={b_prec:.3f} R={b_rec:.3f} F1={b_f1:.3f}  "
                f"(gt={len(gt_acts)} pred={len(pred_acts)})"
            )

        clip_rows.append({
            "video_id": r["video_id"],
            "criterion": r["criterion"],
            "granularity": r["granularity"],
            "start_frame": r["start_frame"],
            "end_frame": r["end_frame"],
            "n_gt": len(gt_acts),
            "n_pred": len(pred_acts),
            "fuzzy_prec": fuzzy_prec,
            "fuzzy_rec": fuzzy_rec,
            "binary_prec": b_prec,
            "binary_rec": b_rec,
            "binary_f1": b_f1,
            "tp_prec": tp_p, "fp": fp, "tp_rec": tp_r, "fn": fn,
            "ctx_tp_prec": ctx_tp_p, "ctx_fp": ctx_fp, "ctx_tp_rec": ctx_tp_r, "ctx_fn": ctx_fn,
            "norm_binary_prec": nb_prec,
            "norm_binary_rec": nb_rec,
            "norm_binary_f1": nb_f1,
            "norm_tp_prec": ntp_p, "norm_fp": nfp, "norm_tp_rec": ntp_r, "norm_fn": nfn,
        })

    all_output["per_clip"] = clip_rows

    # ── 2. Aggregate binary metrics (micro-averaged) ──
    _print()
    _print("=" * 100)
    _print(f"AGGREGATE BINARY METRICS  (threshold={threshold})")
    _print("=" * 100)

    def _micro_stats(rows, prefix=""):
        tp_p = sum(r[f"{prefix}tp_prec"] for r in rows)
        fp_total = sum(r[f"{prefix}fp"] for r in rows)
        tp_r = sum(r[f"{prefix}tp_rec"] for r in rows)
        fn_total = sum(r[f"{prefix}fn"] for r in rows)
        micro_prec = tp_p / (tp_p + fp_total) if (tp_p + fp_total) > 0 else 0.0
        micro_rec = tp_r / (tp_r + fn_total) if (tp_r + fn_total) > 0 else 0.0
        micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)
                     if (micro_prec + micro_rec) > 0 else 0.0)
        return {
            "prec": micro_prec, "rec": micro_rec, "f1": micro_f1,
            "tp_prec": tp_p, "fp": fp_total, "tp_rec": tp_r, "fn": fn_total,
            "n": len(rows),
        }

    def _print_summary_table(rows, label, prefix=""):
        s = _micro_stats(rows, prefix)
        _print(f"  {label:<18s}  P={s['prec']:.3f}  R={s['rec']:.3f}  F1={s['f1']:.3f}  (n={s['n']})")
        return s

    agg = {}

    # Overall
    _print("\n  STRICT:")
    agg["overall_strict"] = _print_summary_table(clip_rows, "Overall")

    coarse_rows = [r for r in clip_rows if r["granularity"] == "coarse"]
    fine_rows = [r for r in clip_rows if r["granularity"] == "fine"]
    if coarse_rows:
        agg["coarse_strict"] = _print_summary_table(coarse_rows, "Coarse only")
    if fine_rows:
        agg["fine_strict"] = _print_summary_table(fine_rows, "Fine only")

    criteria = sorted(set(r["criterion"] for r in clip_rows))
    for crit in criteria:
        crit_rows = [r for r in clip_rows if r["criterion"] == crit]
        agg[f"{crit}_strict"] = _print_summary_table(crit_rows, f"{crit.upper()} only")

    _print("\n  WITH CONTEXT (target_context_1/2 included):")
    agg["overall_ctx"] = _print_summary_table(clip_rows, "Overall", prefix="ctx_")
    if coarse_rows:
        agg["coarse_ctx"] = _print_summary_table(coarse_rows, "Coarse only", prefix="ctx_")
    if fine_rows:
        agg["fine_ctx"] = _print_summary_table(fine_rows, "Fine only", prefix="ctx_")

    if norm_map:
        _print("\n  NORMALIZED:")
        agg["overall_norm"] = _print_summary_table(clip_rows, "Overall", prefix="norm_")
        if coarse_rows:
            agg["coarse_norm"] = _print_summary_table(coarse_rows, "Coarse only", prefix="norm_")
        if fine_rows:
            agg["fine_norm"] = _print_summary_table(fine_rows, "Fine only", prefix="norm_")
        for crit in criteria:
            crit_rows = [r for r in clip_rows if r["criterion"] == crit]
            agg[f"{crit}_norm"] = _print_summary_table(crit_rows, f"{crit.upper()} only", prefix="norm_")

    all_output["aggregate"] = agg

    # ── 3. Actor-role breakdown ──
    _print()
    _print("=" * 100)
    _print("ACTOR-ROLE BREAKDOWN  (binary, strict)")
    _print("=" * 100)

    actor_agg = {}
    for actor_label, actor_roles in ACTOR_GROUPS.items():
        actor_rows_strict = []
        actor_rows_norm = []
        for r in records:
            gt_acts = [a for a in r["gt_actions"] if a.get("actor_role") in actor_roles]
            pred_acts = [a for a in r["pred_actions"] if a.get("actor_role") in actor_roles]
            # Always include clip even when both GT and pred are empty for this actor,
            # so that n is consistent across methods (important for --common-only).
            # Strict
            gt_core = _core_sentences(gt_acts)
            pred_core = _core_sentences(pred_acts)
            b_prec, b_rec, tp_p, fp_val, tp_r, fn_val = compute_binary_precision_recall(
                pred_core, gt_core, threshold=threshold
            )
            actor_rows_strict.append({
                "tp_prec": tp_p, "fp": fp_val, "tp_rec": tp_r, "fn": fn_val,
            })
            # Normalized
            if norm_map:
                gt_norm = [normalize_action(a, norm_map) for a in gt_acts]
                pred_norm = [normalize_action(a, norm_map) for a in pred_acts]
                gt_norm_core = _core_sentences(gt_norm)
                pred_norm_core = _core_sentences(pred_norm)
                nb_prec, nb_rec, ntp_p, nfp, ntp_r, nfn = compute_binary_precision_recall(
                    pred_norm_core, gt_norm_core, threshold=threshold
                )
                actor_rows_norm.append({
                    "tp_prec": ntp_p, "fp": nfp, "tp_rec": ntp_r, "fn": nfn,
                })

        actor_entry = {}
        if actor_rows_strict:
            strict_stats = _micro_stats(actor_rows_strict)
            _print(f"  {actor_label:<18s}  P={strict_stats['prec']:.3f}  R={strict_stats['rec']:.3f}  F1={strict_stats['f1']:.3f}  (n={strict_stats['n']})")
            actor_entry["strict"] = strict_stats
        if actor_rows_norm:
            actor_entry["norm"] = _micro_stats(actor_rows_norm)
        if actor_entry:
            actor_agg[actor_label] = actor_entry

    all_output["actor_breakdown"] = actor_agg

    # ── 4. Hierarchical per-field accuracy ──
    _print()
    _print("=" * 100)
    _print("HIERARCHICAL PER-FIELD ACCURACY  (best-match, no threshold)")
    _print("=" * 100)

    hier_output = {}

    for variant_label, nmap in [("STRICT", None), ("NORMALIZED", norm_map)]:
        if nmap is None and variant_label == "NORMALIZED":
            if not norm_map:
                continue
        if variant_label == "STRICT":
            nmap_use = None
        else:
            nmap_use = nmap

        _print(f"\n  {variant_label}:")

        for actor_scope_label in ["Overall"] + list(ACTOR_GROUPS.keys()):
            if actor_scope_label == "Overall":
                af = None
            else:
                af = ACTOR_GROUPS[actor_scope_label]

            fs, n_gt, n_pred = compute_hierarchical_accuracy(records, norm_map=nmap_use, actor_filter=af)

            if n_gt == 0 and n_pred == 0:
                continue

            _print(f"\n    [{actor_scope_label}]  (GT actions: {n_gt}, Pred actions: {n_pred})")

            for metric_key, metric_label in [("recall", "Recall"), ("precision", "Precision")]:
                _print(f"\n      {metric_label}:")
                _print(f"        {'Field':<22s} {'Micro':>7s} {'Macro':>7s}")
                _print(f"        {'-'*20} {'-'*7} {'-'*7}")
                for field in HIER_FIELDS:
                    ss = fs[field][metric_key]
                    _print(f"        {field:<22s} {ss['micro']:7.3f} {ss['macro']:7.3f}")

                    # Per-class breakdown
                    for cls_val, cls_stats in sorted(
                        ss["class_acc"].items(), key=lambda x: -x[1]["total"]
                    ):
                        _print(
                            f"          {cls_val:<28s} {cls_stats['acc']:5.2f}  "
                            f"({cls_stats['correct']}/{cls_stats['total']})"
                        )

            key = f"{variant_label.lower()}_{actor_scope_label.lower()}"
            hier_output[key] = {
                "n_gt": n_gt,
                "n_pred": n_pred,
                "fields": {
                    field: {
                        mk: {
                            "micro": fs[field][mk]["micro"],
                            "macro": fs[field][mk]["macro"],
                            "class_acc": fs[field][mk]["class_acc"],
                        }
                        for mk in ["recall", "precision"]
                    }
                    for field in HIER_FIELDS
                },
            }

    all_output["hierarchical"] = hier_output

    return all_output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _to_serializable(obj):
    """Make numpy/nan values JSON-serializable."""
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline predictions against ground-truth annotations."
    )
    parser.add_argument("--pred", required=True, help="Path to baseline JSONL prediction file")
    parser.add_argument("--gt", required=False, default=None,
                        help="Path to annotation JSON file (for action eval; optional)")
    parser.add_argument(
        "--threshold", type=float, default=0.70,
        help="Fuzzy-match threshold for binary precision/recall (default: 0.70)",
    )
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="Skip normalized metrics even if normalization map is available",
    )
    parser.add_argument(
        "--labels-dir", default=None,
        help="Path to labels directory (default: data/raw/.../train/labels)",
    )
    parser.add_argument(
        "--json-out", default=None,
        help="Path to write JSON results (overrides auto-save path)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Only evaluate the first N prediction frames.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed evaluation output to stdout",
    )
    args = parser.parse_args()

    if args.no_normalize:
        global NORM_MAP
        NORM_MAP = {}

    verbose = args.verbose
    labels_dir = args.labels_dir

    # Determine output path: explicit --json-out, or auto-save to metrics/<pred_dir_name>/<pred_filename>.json
    if args.json_out:
        json_out = args.json_out
    else:
        pred_dir_name = os.path.basename(os.path.dirname(os.path.abspath(args.pred)))
        pred_filename = os.path.splitext(os.path.basename(args.pred))[0] + ".json"
        json_out = os.path.join(ROOT_DIR, "metrics", pred_dir_name, pred_filename)

    if verbose:
        print(f"Predictions: {args.pred}")
        print(f"Annotations: {args.gt}")
        print(f"Threshold:   {args.threshold}")
        print(f"Normalize:   {bool(NORM_MAP)}")
        print()

    pred_frames = load_predictions(args.pred)
    if args.max_frames is not None and args.max_frames > 0:
        pred_frames = pred_frames[:args.max_frames]
    pred_video = pred_frames[0].get("video_id", "unknown") if pred_frames else "unknown"

    output = {}

    # ── CVS mAP evaluation (frame-level + video-level) ──
    cvs_result = evaluate_cvs(pred_frames, pred_video, labels_dir=labels_dir, verbose=verbose)
    if cvs_result:
        output["cvs"] = cvs_result
    elif verbose:
        print(f"  No CVS labels found for video {pred_video}")

    # ── Action evaluation (if --gt provided) ──
    if args.gt:
        annotations = load_annotations(args.gt)
        gt_videos = set(a["video_id"] for a in annotations)

        if pred_video not in gt_videos:
            print(f"WARNING: Prediction video_id '{pred_video}' not found in annotation video_ids {gt_videos}")
            print("         Results may not be meaningful if the videos don't match.")

        if verbose:
            print(f"Loaded {len(pred_frames)} prediction frames, {len(annotations)} annotation records")
            print()

        records = build_eval_records(pred_frames, annotations)
        if verbose:
            print(f"Built {len(records)} evaluation clips (coarse + fine)\n")

        action_output = evaluate(records, threshold=args.threshold, use_norm=bool(NORM_MAP), verbose=verbose)
        output["actions"] = action_output

    # Write JSON
    os.makedirs(os.path.dirname(json_out), exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(output, f, indent=2, default=_to_serializable)
    print(f"Saved metrics to {json_out}")


if __name__ == "__main__":
    main()
