#!/usr/bin/env python3
"""Run evaluation on all prediction files and aggregate results.

Scans output directories for JSONL prediction files, runs per-file evaluation
(CVS mAP + action metrics if annotations exist), saves per-file metrics, then
prints an aggregated summary table across all methods, grouped by model.

Usage:
    python scripts/eval/evaluate_all.py \
        --actions-dir data/processed/CVS_Challenge_SAGES_v1/cvs_act_annotations/v1/train/audit_v9 \
        [--output-dirs baseline direct cot surgent_sequential_agent] \
        [--labels-dir data/raw/CVS_Challenge_SAGES_v1/train/labels] \
        [--models gpt-5-mini gemini-2.5-flash] \
        [--threshold 0.70] [--verbose]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Import from evaluate_baseline (same directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from evaluate_baseline import (
    ROOT_DIR,
    CRITERIA,
    load_predictions,
    load_annotations,
    build_eval_records,
    evaluate,
    evaluate_cvs,
    _to_serializable,
    NORM_MAP,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_video_id_from_pred(pred_frames):
    """Get video_id from first prediction frame."""
    if not pred_frames:
        return None
    return pred_frames[0].get("video_id")


def extract_model_from_pred(pred_frames, pred_path=None):
    """Get model name from prediction data or filename.

    Tries (in order):
      1. "model" field in the first frame
      2. Parse from filename patterns:
         - baseline (new): {vid}__{preset}__{model}__{taxonomy}.jsonl
         - baseline (old): {vid}__{preset}__{model}.jsonl
         - surgent:        {vid}__multi__C1-C2-C3__{model}__{taxonomy}.jsonl
      3. Falls back to "unknown"
    """
    if pred_frames:
        model = pred_frames[0].get("model")
        if model:
            return str(model)
    # Try parsing from filename
    if pred_path:
        basename = os.path.splitext(os.path.basename(pred_path))[0]
        parts = basename.split("__")
        # surgent pattern: {vid}__multi__C1-C2-C3__{model}__{taxonomy}
        # parts: [vid, multi, C1-C2-C3, model, taxonomy_v9]
        if len(parts) >= 5 and parts[1] == "multi":
            return parts[3]
        # baseline (new): {vid}__{preset}__{model}__{taxonomy}
        # baseline (old): {vid}__{preset}__{model}
        # Detect taxonomy suffix: last part starts with "taxonomy_"
        if len(parts) >= 4 and parts[-1].startswith("taxonomy_"):
            return parts[-2]
        if len(parts) >= 3:
            return parts[-1]
    return "unknown"


def extract_taxonomy_from_pred(pred_frames, pred_path=None):
    """Get taxonomy version from prediction data or filename.

    Tries (in order):
      1. "taxonomy_version" field in the first frame (baseline stores this)
      2. Nested "predicted_taxonomy_action.taxonomy_version" (SurGent stores this)
      3. Parse from filename: last "__taxonomy_vN" segment
      4. Falls back to "unknown"
    """
    if pred_frames:
        tv = pred_frames[0].get("taxonomy_version")
        if tv:
            return str(tv)
        pred_action = pred_frames[0].get("predicted_taxonomy_action")
        if isinstance(pred_action, dict):
            tv = pred_action.get("taxonomy_version")
            if tv:
                return str(tv)
    if pred_path:
        basename = os.path.splitext(os.path.basename(pred_path))[0]
        parts = basename.split("__")
        # Look for a part like "taxonomy_v8" or "taxonomy_v9"
        for part in reversed(parts):
            if part.startswith("taxonomy_"):
                return part.replace("taxonomy_", "")
    return "unknown"


def find_annotation_file(video_id, actions_dir):
    """Find annotation JSON for a video_id in the actions directory."""
    if not actions_dir:
        return None
    candidate = os.path.join(actions_dir, f"{video_id}.json")
    return candidate if os.path.isfile(candidate) else None


def collect_pred_files(output_dir):
    """Collect all JSONL prediction files from an output directory."""
    pattern = os.path.join(output_dir, "*.jsonl")
    return sorted(glob.glob(pattern))


def collect_pred_files_recursive(output_dir):
    """Collect JSONL files from output_dir and all nested subdirectories.

    Returns list of (method_label, pred_path) tuples.
    - Files in output_dir itself get method_label = basename(output_dir)
    - Files in output_dir/a/b/... get method_label = basename(output_dir)/a/b/...
    """
    dir_name = os.path.basename(output_dir)
    results = []
    for root, _dirs, files in os.walk(output_dir):
        jsonl_files = sorted(f for f in files if f.endswith(".jsonl"))
        if not jsonl_files:
            continue
        rel = os.path.relpath(root, output_dir)
        if rel == ".":
            label = dir_name
        else:
            label = f"{dir_name}/{rel}"
        for fname in jsonl_files:
            results.append((label, os.path.join(root, fname)))
    return results


# ---------------------------------------------------------------------------
# Run evaluation on a single file
# ---------------------------------------------------------------------------

def evaluate_single_file(pred_path, actions_dir, labels_dir, threshold, verbose,
                         method_label=None, max_frames=None, action_sources=None):
    """Evaluate a single prediction file. Returns (output_dict, metrics_path) or (None, None).

    Args:
        method_label: e.g. "direct" or "surgent_sequential_agent/direct".
                      Used for the metrics save path. If None, derived from pred_path parent dir.
        max_frames: If set, only evaluate the first N prediction frames.
        action_sources: For surgent files, only include actions from these sources.
    """
    pred_frames = load_predictions(pred_path, action_sources=action_sources)
    if not pred_frames:
        return None, None

    if max_frames is not None and max_frames > 0:
        pred_frames = pred_frames[:max_frames]

    video_id = extract_video_id_from_pred(pred_frames)
    if not video_id:
        return None, None

    model = extract_model_from_pred(pred_frames, pred_path)
    taxonomy = extract_taxonomy_from_pred(pred_frames, pred_path)

    output = {"_model": model, "_video_id": video_id, "_taxonomy": taxonomy}

    # CVS mAP
    cvs_result = evaluate_cvs(pred_frames, video_id, labels_dir=labels_dir, verbose=verbose)
    if cvs_result:
        output["cvs"] = cvs_result

    # Action evaluation
    gt_path = find_annotation_file(video_id, actions_dir)
    if gt_path:
        annotations = load_annotations(gt_path)
        records = build_eval_records(pred_frames, annotations)
        action_output = evaluate(records, threshold=threshold, use_norm=bool(NORM_MAP), verbose=verbose)
        output["actions"] = action_output

    if "cvs" not in output and "actions" not in output:
        return None, None

    # Save metrics — use method_label for nested dirs (e.g. surgent_sequential_agent/direct)
    if method_label:
        metrics_subdir = method_label
    else:
        metrics_subdir = os.path.basename(os.path.dirname(os.path.abspath(pred_path)))
    pred_filename = os.path.splitext(os.path.basename(pred_path))[0] + ".json"
    metrics_path = os.path.join(ROOT_DIR, "metrics", metrics_subdir, pred_filename)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2, default=_to_serializable)

    return output, metrics_path


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(all_metrics):
    """Aggregate per-file metrics into summary statistics.

    all_metrics: list of (method_name, output_dict) tuples.
    Groups by (method, model) when model info is available.
    Returns {display_key: summary_dict}.
    """
    # Group by method/model/taxonomy
    by_method = {}
    for method, output in all_metrics:
        model = output.get("_model", "unknown")
        taxonomy = output.get("_taxonomy", "unknown")
        if taxonomy != "unknown":
            key = f"{method}/{model}/{taxonomy}"
        elif model != "unknown":
            key = f"{method}/{model}"
        else:
            key = method
        by_method.setdefault(key, []).append(output)

    summaries = {}
    for method, outputs in sorted(by_method.items()):
        summary = {"n_files": len(outputs)}

        # CVS metrics
        frame_maps = []
        video_maps = []
        frame_aps = {c: [] for c in CRITERIA}
        video_aps = {c: [] for c in CRITERIA}
        for o in outputs:
            cvs = o.get("cvs")
            if not cvs:
                continue
            fm = cvs.get("frame_mAP")
            if fm is not None and not (isinstance(fm, float) and np.isnan(fm)):
                frame_maps.append(fm)
            vm = cvs.get("video_mAP")
            if vm is not None and not (isinstance(vm, float) and np.isnan(vm)):
                video_maps.append(vm)
            for c in CRITERIA:
                fa = cvs.get("frame_ap", {}).get(c)
                if fa is not None and not (isinstance(fa, float) and np.isnan(fa)):
                    frame_aps[c].append(fa)
                va = cvs.get("video_ap", {}).get(c)
                if va is not None and not (isinstance(va, float) and np.isnan(va)):
                    video_aps[c].append(va)

        if frame_maps:
            summary["frame_mAP"] = float(np.mean(frame_maps))
            summary["frame_ap"] = {c: float(np.mean(frame_aps[c])) if frame_aps[c] else None
                                    for c in CRITERIA}
            summary["n_cvs_videos"] = len(frame_maps)
        if video_maps:
            summary["video_mAP"] = float(np.mean(video_maps))
            summary["video_ap"] = {c: float(np.mean(video_aps[c])) if video_aps[c] else None
                                    for c in CRITERIA}

        # Frame-level binary CVS P/R/F1 (micro across all videos and criteria)
        cvs_tp = cvs_fp = cvs_fn = 0
        cvs_per_c = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CRITERIA}
        n_cvs_binary = 0
        for o in outputs:
            cvs = o.get("cvs")
            if not cvs:
                continue
            fb = cvs.get("frame_binary")
            if not fb:
                continue
            n_cvs_binary += 1
            for c in CRITERIA:
                cb = fb.get(c, {})
                cvs_tp += cb.get("tp", 0)
                cvs_fp += cb.get("fp", 0)
                cvs_fn += cb.get("fn", 0)
                cvs_per_c[c]["tp"] += cb.get("tp", 0)
                cvs_per_c[c]["fp"] += cb.get("fp", 0)
                cvs_per_c[c]["fn"] += cb.get("fn", 0)

        if n_cvs_binary > 0:
            bp = cvs_tp / (cvs_tp + cvs_fp) if (cvs_tp + cvs_fp) > 0 else 0.0
            br = cvs_tp / (cvs_tp + cvs_fn) if (cvs_tp + cvs_fn) > 0 else 0.0
            bf1 = 2 * bp * br / (bp + br) if (bp + br) > 0 else 0.0
            summary["frame_binary_prec"] = bp
            summary["frame_binary_rec"] = br
            summary["frame_binary_f1"] = bf1
            per_c_binary = {}
            for c in CRITERIA:
                t = cvs_per_c[c]
                cp = t["tp"] / (t["tp"] + t["fp"]) if (t["tp"] + t["fp"]) > 0 else 0.0
                cr = t["tp"] / (t["tp"] + t["fn"]) if (t["tp"] + t["fn"]) > 0 else 0.0
                cf1 = 2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0.0
                per_c_binary[c] = {"prec": cp, "rec": cr, "f1": cf1}
            summary["frame_binary_per_c"] = per_c_binary

        # Action metrics: aggregate binary P/R/F1 across all files (micro)
        # Collect for overall, coarse, fine, ctx, and normalized variants
        def _aggregate_action_variant(outputs, agg_key):
            tp_p = fp_v = tp_r = fn_v = 0
            n_files = 0
            for o in outputs:
                actions = o.get("actions")
                if not actions:
                    continue
                agg = actions.get("aggregate", {})
                entry = agg.get(agg_key)
                if entry:
                    tp_p += entry.get("tp_prec", 0)
                    fp_v += entry.get("fp", 0)
                    tp_r += entry.get("tp_rec", 0)
                    fn_v += entry.get("fn", 0)
                    n_files += 1
            if n_files == 0:
                return None
            prec = tp_p / (tp_p + fp_v) if (tp_p + fp_v) > 0 else 0.0
            rec = tp_r / (tp_r + fn_v) if (tp_r + fn_v) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            return {"prec": prec, "rec": rec, "f1": f1, "n": n_files,
                    "n_pred": tp_p + fp_v, "n_gt": tp_r + fn_v}

        overall_strict = _aggregate_action_variant(outputs, "overall_strict")
        if overall_strict:
            summary["action_prec"] = overall_strict["prec"]
            summary["action_rec"] = overall_strict["rec"]
            summary["action_f1"] = overall_strict["f1"]
            summary["n_action_videos"] = overall_strict["n"]
            summary["n_pred"] = overall_strict["n_pred"]
            summary["n_gt"] = overall_strict["n_gt"]

        # Coarse/fine strict
        for gran in ("coarse", "fine"):
            result = _aggregate_action_variant(outputs, f"{gran}_strict")
            if result:
                summary[f"{gran}_action_prec"] = result["prec"]
                summary[f"{gran}_action_rec"] = result["rec"]
                summary[f"{gran}_action_f1"] = result["f1"]
                summary[f"n_{gran}_action_videos"] = result["n"]

        # With-context variant (target_context_1/2 included)
        ctx_result = _aggregate_action_variant(outputs, "overall_ctx")
        if ctx_result:
            summary["ctx_action_prec"] = ctx_result["prec"]
            summary["ctx_action_rec"] = ctx_result["rec"]
            summary["ctx_action_f1"] = ctx_result["f1"]
        for gran in ("coarse", "fine"):
            result = _aggregate_action_variant(outputs, f"{gran}_ctx")
            if result:
                summary[f"{gran}_ctx_action_prec"] = result["prec"]
                summary[f"{gran}_ctx_action_rec"] = result["rec"]
                summary[f"{gran}_ctx_action_f1"] = result["f1"]

        # Normalized action metrics
        norm_result = _aggregate_action_variant(outputs, "overall_norm")
        if norm_result:
            summary["norm_action_prec"] = norm_result["prec"]
            summary["norm_action_rec"] = norm_result["rec"]
            summary["norm_action_f1"] = norm_result["f1"]

        # Per-actor action metrics (micro across all files)
        actor_labels = ["Left", "Right", "Camera"]
        actor_summary = {}
        for actor_label in actor_labels:
            for variant in ["strict", "norm"]:
                tp_p = fp_v = tp_r = fn_v = 0
                n_files_with_actor = 0
                for o in outputs:
                    actions = o.get("actions")
                    if not actions:
                        continue
                    ab = actions.get("actor_breakdown", {})
                    actor_data = ab.get(actor_label, {})
                    # Handle both new format (strict/norm sub-dicts) and
                    # legacy format (flat dict = strict only)
                    if variant in actor_data:
                        vd = actor_data[variant]
                    elif variant == "strict" and "prec" in actor_data:
                        vd = actor_data  # legacy flat format
                    else:
                        continue
                    tp_p += vd.get("tp_prec", 0)
                    fp_v += vd.get("fp", 0)
                    tp_r += vd.get("tp_rec", 0)
                    fn_v += vd.get("fn", 0)
                    n_files_with_actor += 1
                if n_files_with_actor > 0:
                    aprec = tp_p / (tp_p + fp_v) if (tp_p + fp_v) > 0 else 0.0
                    arec = tp_r / (tp_r + fn_v) if (tp_r + fn_v) > 0 else 0.0
                    af1 = 2 * aprec * arec / (aprec + arec) if (aprec + arec) > 0 else 0.0
                    key = f"actor_{actor_label.lower()}_{variant}"
                    actor_summary[key] = {
                        "prec": aprec, "rec": arec, "f1": af1,
                        "n": n_files_with_actor,
                    }
        if actor_summary:
            summary["actor"] = actor_summary

        summaries[method] = summary

    return summaries


def print_summary_table(summaries):
    """Print a formatted comparison table across methods."""
    if not summaries:
        print("No results to display.")
        return

    # Dynamic method column width: pad all names to the longest one
    W = max((len(m) for m in summaries), default=6) + 2  # +2 for breathing room
    has_total = any("n_total" in s for s in summaries.values())

    def _n_header():
        """Return n column header(s)."""
        if has_total:
            return f" {'n':>4s} {'N':>4s}"
        return f" {'n':>4s}"

    def _n_sep():
        """Return n column separator(s)."""
        if has_total:
            return f" {'-'*4} {'-'*4}"
        return f" {'-'*4}"

    def _n_val(s, n_key="n_cvs_videos"):
        """Return formatted n value(s)."""
        if has_total:
            return f" {s.get(n_key, 0):4d} {s.get('n_total', s.get(n_key, 0)):4d}"
        return f" {s.get(n_key, 0):4d}"

    print()
    print("=" * 140)
    print("AGGREGATED RESULTS")
    print("=" * 140)

    # CVS table
    has_cvs = any("frame_mAP" in s for s in summaries.values())
    if has_cvs:
        print(f"\n  CVS Scores (averaged across videos)")
        print(f"  {'Method':<{W}s} {'Frame mAP':>10s} {'F.c1':>6s} {'F.c2':>6s} {'F.c3':>6s}"
              f"  {'Video mAP':>10s} {'V.c1':>6s} {'V.c2':>6s} {'V.c3':>6s}{_n_header()}")
        print(f"  {'-'*(W-2)}" + f" {'-'*10}" + f" {'-'*6}" * 3
              + f"  {'-'*10}" + f" {'-'*6}" * 3 + _n_sep())
        for method, s in sorted(summaries.items()):
            if "frame_mAP" not in s:
                continue
            fa = s.get("frame_ap", {})
            va = s.get("video_ap", {})
            vm = s.get("video_mAP")
            row = f"  {method:<{W}s} {s['frame_mAP']:10.3f}"
            for c in CRITERIA:
                v = fa.get(c)
                row += f" {v:6.3f}" if v is not None else f" {'N/A':>6s}"
            row += f"  {vm:10.3f}" if vm is not None else f"  {'N/A':>10s}"
            for c in CRITERIA:
                v = va.get(c)
                row += f" {v:6.3f}" if v is not None else f" {'N/A':>6s}"
            row += _n_val(s, "n_cvs_videos")
            print(row)

    # CVS Binary P/R/F1 table
    has_cvs_binary = any("frame_binary_f1" in s for s in summaries.values())
    if has_cvs_binary:
        print(f"\n  CVS Frame-level Binary P/R/F1 (pred >0.5 = positive, micro-averaged)")
        print(f"  {'Method':<{W}s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}"
              f"  {'c1.P':>5s} {'c1.R':>5s} {'c1.F1':>5s}"
              f"  {'c2.P':>5s} {'c2.R':>5s} {'c2.F1':>5s}"
              f"  {'c3.P':>5s} {'c3.R':>5s} {'c3.F1':>5s}"
              f"{_n_header()}")
        print(f"  {'-'*(W-2)}" + f" {'-'*6}" * 3
              + f"  {'-'*5}" * 3 + f"  {'-'*5}" * 3 + f"  {'-'*5}" * 3 + _n_sep())
        for method, s in sorted(summaries.items()):
            if "frame_binary_f1" not in s:
                continue
            pc = s.get("frame_binary_per_c", {})
            row = (f"  {method:<{W}s} {s['frame_binary_prec']:6.3f} {s['frame_binary_rec']:6.3f} "
                   f"{s['frame_binary_f1']:6.3f}")
            for c in CRITERIA:
                cb = pc.get(c, {})
                row += f"  {cb.get('prec', 0):5.3f} {cb.get('rec', 0):5.3f} {cb.get('f1', 0):5.3f}"
            row += _n_val(s, "n_cvs_videos")
            print(row)

    # Action table
    has_actions = any("action_f1" in s for s in summaries.values())
    if has_actions:
        has_norm = any("norm_action_f1" in s for s in summaries.values())
        print(f"\n  Action Metrics (micro-averaged binary, strict)")
        header = f"  {'Method':<{W}s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s}"
        if has_norm:
            header += f"  {'N.Prec':>8s} {'N.Rec':>8s} {'N.F1':>8s}"
        header += f" {'#pred':>6s} {'#gt':>6s}"
        header += _n_header()
        print(header)
        sep = f"  {'-'*(W-2)}" + f" {'-'*8}" * 3
        if has_norm:
            sep += f"  {'-'*8}" * 3
        sep += f" {'-'*6} {'-'*6}"
        sep += _n_sep()
        print(sep)
        for method, s in sorted(summaries.items()):
            if "action_f1" not in s:
                continue
            row = (f"  {method:<{W}s} {s['action_prec']:8.3f} {s['action_rec']:8.3f} "
                   f"{s['action_f1']:8.3f}")
            if has_norm:
                if "norm_action_f1" in s:
                    row += (f"  {s['norm_action_prec']:8.3f} {s['norm_action_rec']:8.3f} "
                            f"{s['norm_action_f1']:8.3f}")
                else:
                    row += f"  {'N/A':>8s} {'N/A':>8s} {'N/A':>8s}"
            row += f" {s.get('n_pred', 0):6d} {s.get('n_gt', 0):6d}"
            row += _n_val(s, "n_action_videos")
            print(row)

    # Action by granularity (coarse / fine)
    has_coarse = any("coarse_action_f1" in s for s in summaries.values())
    has_fine = any("fine_action_f1" in s for s in summaries.values())
    if has_coarse or has_fine:
        for gran, label in [("coarse", "Coarse"), ("fine", "Fine")]:
            has_gran = any(f"{gran}_action_f1" in s for s in summaries.values())
            if not has_gran:
                continue
            print(f"\n  Action Metrics — {label} only (micro-averaged binary, strict)")
            print(f"  {'Method':<{W}s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s}{_n_header()}")
            print(f"  {'-'*(W-2)}" + f" {'-'*8}" * 3 + _n_sep())
            for method, s in sorted(summaries.items()):
                if f"{gran}_action_f1" not in s:
                    continue
                row = (f"  {method:<{W}s} {s[f'{gran}_action_prec']:8.3f} "
                       f"{s[f'{gran}_action_rec']:8.3f} {s[f'{gran}_action_f1']:8.3f}")
                row += _n_val(s, f"n_{gran}_action_videos")
                print(row)

    # Action with context (target_context_1/2 included in sentence)
    has_ctx = any("ctx_action_f1" in s for s in summaries.values())
    if has_ctx:
        has_ctx_coarse = any("coarse_ctx_action_f1" in s for s in summaries.values())
        has_ctx_fine = any("fine_ctx_action_f1" in s for s in summaries.values())
        print(f"\n  Action Metrics — With Context (target_context_1/2 included)")
        header = f"  {'Method':<{W}s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s}"
        if has_ctx_coarse:
            header += f"  {'C.Prec':>8s} {'C.Rec':>8s} {'C.F1':>8s}"
        if has_ctx_fine:
            header += f"  {'F.Prec':>8s} {'F.Rec':>8s} {'F.F1':>8s}"
        header += _n_header()
        print(header)
        sep = f"  {'-'*(W-2)}" + f" {'-'*8}" * 3
        if has_ctx_coarse:
            sep += f"  {'-'*8}" * 3
        if has_ctx_fine:
            sep += f"  {'-'*8}" * 3
        sep += _n_sep()
        print(sep)
        for method, s in sorted(summaries.items()):
            if "ctx_action_f1" not in s:
                continue
            row = (f"  {method:<{W}s} {s['ctx_action_prec']:8.3f} "
                   f"{s['ctx_action_rec']:8.3f} {s['ctx_action_f1']:8.3f}")
            if has_ctx_coarse:
                if "coarse_ctx_action_f1" in s:
                    row += (f"  {s['coarse_ctx_action_prec']:8.3f} "
                            f"{s['coarse_ctx_action_rec']:8.3f} {s['coarse_ctx_action_f1']:8.3f}")
                else:
                    row += f"  {'N/A':>8s} {'N/A':>8s} {'N/A':>8s}"
            if has_ctx_fine:
                if "fine_ctx_action_f1" in s:
                    row += (f"  {s['fine_ctx_action_prec']:8.3f} "
                            f"{s['fine_ctx_action_rec']:8.3f} {s['fine_ctx_action_f1']:8.3f}")
                else:
                    row += f"  {'N/A':>8s} {'N/A':>8s} {'N/A':>8s}"
            row += _n_val(s, "n_action_videos")
            print(row)

    # Per-actor breakdown tables
    actor_labels = ["Left", "Right", "Camera"]
    has_actor = any("actor" in s for s in summaries.values())
    if has_actor:
        has_actor_norm = any(
            f"actor_{al.lower()}_norm" in s.get("actor", {})
            for s in summaries.values()
            for al in actor_labels
        )
        for actor_label in actor_labels:
            # Check if any method has data for this actor
            any_data = any(
                f"actor_{actor_label.lower()}_strict" in s.get("actor", {})
                for s in summaries.values()
            )
            if not any_data:
                continue

            print(f"\n  Action by Actor: {actor_label}")
            header = f"  {'Method':<{W}s} {'S.Prec':>8s} {'S.Rec':>8s} {'S.F1':>8s}"
            if has_actor_norm:
                header += f"  {'N.Prec':>8s} {'N.Rec':>8s} {'N.F1':>8s}"
            header += _n_header()
            print(header)
            sep = f"  {'-'*(W-2)}" + f" {'-'*8}" * 3
            if has_actor_norm:
                sep += f"  {'-'*8}" * 3
            sep += _n_sep()
            print(sep)
            for method, s in sorted(summaries.items()):
                actor_data = s.get("actor", {})
                strict_key = f"actor_{actor_label.lower()}_strict"
                norm_key = f"actor_{actor_label.lower()}_norm"
                if strict_key not in actor_data:
                    continue
                sd = actor_data[strict_key]
                row = (f"  {method:<{W}s} {sd['prec']:8.3f} {sd['rec']:8.3f} "
                       f"{sd['f1']:8.3f}")
                if has_actor_norm:
                    if norm_key in actor_data:
                        nd = actor_data[norm_key]
                        row += f"  {nd['prec']:8.3f} {nd['rec']:8.3f} {nd['f1']:8.3f}"
                    else:
                        row += f"  {'N/A':>8s} {'N/A':>8s} {'N/A':>8s}"
                row += _n_val(s, "n_action_videos")
                print(row)

    print()


# ---------------------------------------------------------------------------
# Also aggregate from existing metrics/ files
# ---------------------------------------------------------------------------

def _extract_model_from_metrics_filename(fname):
    """Extract model from metrics filename pattern.

    Handles:
      - {vid}__{preset}__{model}__{taxonomy}.json  (new)
      - {vid}__{preset}__{model}.json              (old)
      - {vid}__multi__C1-C2-C3__{model}__{taxonomy}.json  (surgent)
    """
    base = os.path.splitext(fname)[0]
    parts = base.split("__")
    if len(parts) >= 5 and parts[1] == "multi":
        return parts[3]
    if len(parts) >= 4 and parts[-1].startswith("taxonomy_"):
        return parts[-2]
    if len(parts) >= 3:
        return parts[-1]
    return None


def _extract_taxonomy_from_metrics_filename(fname):
    """Extract taxonomy version from metrics filename (e.g. 'taxonomy_v9' -> 'v9')."""
    base = os.path.splitext(fname)[0]
    parts = base.split("__")
    for part in reversed(parts):
        if part.startswith("taxonomy_"):
            return part.replace("taxonomy_", "")
    return None


def load_existing_metrics(metrics_dirs, model_filter=None, video_filter=None,
                          taxonomy_filter=None):
    """Load already-computed metrics JSON files from metrics/ subdirectories.

    Handles both flat dirs (metrics/direct/) and nested dirs
    (metrics/surgent_sequential_agent/direct/).

    Args:
        metrics_dirs: list of subdirectory names under metrics/
        model_filter: if set, only include files matching these model names
        video_filter: if set, only include files matching these video IDs
        taxonomy_filter: if set, only include files matching these taxonomy versions

    Returns list of (method_label, output_dict) tuples.
    """
    results = []

    def _load_dir(dir_path, method_label):
        if not os.path.isdir(dir_path):
            return
        for fname in sorted(os.listdir(dir_path)):
            fpath = os.path.join(dir_path, fname)
            if os.path.isdir(fpath):
                # Recurse one level for nested preset dirs
                _load_dir(fpath, f"{method_label}/{fname}")
                continue
            if not fname.endswith(".json"):
                continue
            with open(fpath) as f:
                output = json.load(f)
            # Ensure _model is set (backfill from filename if missing)
            if "_model" not in output:
                model = _extract_model_from_metrics_filename(fname)
                if model:
                    output["_model"] = model
            # Ensure _taxonomy is set (backfill from filename if missing)
            if "_taxonomy" not in output:
                tax = _extract_taxonomy_from_metrics_filename(fname)
                if tax:
                    output["_taxonomy"] = tax
            # Apply taxonomy filter
            if taxonomy_filter:
                file_tax = output.get("_taxonomy", "unknown")
                if file_tax not in taxonomy_filter:
                    continue
            # Apply model filter
            if model_filter:
                file_model = output.get("_model", "unknown")
                if file_model not in model_filter:
                    continue
            # Apply video filter (common-only)
            if video_filter is not None:
                file_vid = output.get("_video_id")
                if not file_vid:
                    base = os.path.splitext(fname)[0]
                    parts = base.split("__")
                    if parts:
                        file_vid = parts[0]
                if file_vid not in video_filter:
                    continue
            results.append((method_label, output))

    for mdir in metrics_dirs:
        full_path = os.path.join(ROOT_DIR, "metrics", mdir)
        _load_dir(full_path, mdir)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIRS = ["baseline", "direct", "cot", "subrubric", "auto", "direct_causal", "cot_causal", "subrubric_causal", "auto_causal", "surgent_sequential_agent"]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all prediction files and print aggregated results."
    )
    parser.add_argument(
        "--actions-dir", required=True,
        help="Directory containing annotation JSON files (e.g. .../audit_v8). "
             "Only videos with annotations here will have action metrics evaluated.",
    )
    parser.add_argument(
        "--output-dirs", nargs="+", default=DEFAULT_OUTPUT_DIRS,
        help=f"Output subdirectory names under outputs/ (default: {DEFAULT_OUTPUT_DIRS})",
    )
    parser.add_argument(
        "--labels-dir", default=None,
        help="Path to CVS labels directory (default: data/raw/.../train/labels)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Filter to only these model names (e.g. gpt-5-mini gemini-2.5-flash). "
             "Default: all models.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.70,
        help="Fuzzy-match threshold for binary P/R (default: 0.70)",
    )
    parser.add_argument(
        "--common-only", action="store_true",
        help="Only evaluate videos that have predictions in ALL output dirs.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Only evaluate the first N prediction frames per file.",
    )
    parser.add_argument(
        "--taxonomy", nargs="+", default=None,
        help="Filter to only these taxonomy versions (e.g. v9). "
             "Default: all taxonomy versions.",
    )
    parser.add_argument(
        "--action-sources", nargs="+", default=None,
        help="For surgent files, only include actions from these sources. "
             "Valid: scene_cvs, action_rec, instrument_rec. "
             "Default: all sources merged.",
    )
    parser.add_argument(
        "--exclude", nargs="+", default=None,
        help="Exclude surgent prediction files whose path contains any of these substrings "
             "(e.g. --exclude _aa to skip analyze-actions runs). "
             "Only applies to surgent dirs; baselines are always included.",
    )
    parser.add_argument(
        "--include", nargs="+", default=None,
        help="Only include surgent prediction files whose path contains at least one "
             "of these substrings (e.g. --include _ra to keep only _ra runs). "
             "Only applies to surgent dirs; baselines are always included.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-file evaluation details",
    )
    args = parser.parse_args()

    # Resolve actions dir to get the set of annotated video IDs
    actions_dir = os.path.abspath(args.actions_dir)
    annotated_video_ids = set()
    if os.path.isdir(actions_dir):
        for fname in os.listdir(actions_dir):
            if fname.endswith(".json"):
                annotated_video_ids.add(os.path.splitext(fname)[0])
    model_filter = set(args.models) if args.models else None
    taxonomy_filter = set(args.taxonomy) if args.taxonomy else None
    action_sources = args.action_sources  # None means all
    print(f"Actions dir: {actions_dir}  ({len(annotated_video_ids)} annotated videos)")
    if action_sources:
        print(f"Action sources filter: {action_sources}")
    if model_filter:
        print(f"Model filter: {sorted(model_filter)}")
    if taxonomy_filter:
        print(f"Taxonomy filter: {sorted(taxonomy_filter)}")

    # Build full list of (method_label, pred_path) across all output dirs
    all_pred_entries = []  # list of (method_label, pred_path)
    for dir_name in args.output_dirs:
        output_dir = os.path.join(ROOT_DIR, "outputs", dir_name)
        if not os.path.isdir(output_dir):
            continue
        all_pred_entries.extend(collect_pred_files_recursive(output_dir))

    # Apply --include / --exclude filters (only to surgent dirs; baselines always kept)
    def _is_surgent(label):
        return "surgent" in label

    if args.include:
        before = len(all_pred_entries)
        all_pred_entries = [
            (label, path) for label, path in all_pred_entries
            if not _is_surgent(label) or any(pat in path for pat in args.include)
        ]
        print(f"Include filter {args.include} (surgent only): {before} -> {len(all_pred_entries)} files")

    if args.exclude:
        before = len(all_pred_entries)
        all_pred_entries = [
            (label, path) for label, path in all_pred_entries
            if not _is_surgent(label) or not any(pat in path for pat in args.exclude)
        ]
        print(f"Exclude filter {args.exclude} (surgent only): {before} -> {len(all_pred_entries)} files")

    # Pre-scan: find video IDs per (method, model, taxonomy) group (for --common-only)
    # This ensures every row in the final table has the same set of videos.
    common_video_ids = None
    if args.common_only:
        per_group_vids = {}
        for method_label, pf in all_pred_entries:
            pred_frames = load_predictions(pf)
            vid = extract_video_id_from_pred(pred_frames)
            if not (vid and vid in annotated_video_ids):
                continue
            m = extract_model_from_pred(pred_frames, pf)
            t = extract_taxonomy_from_pred(pred_frames, pf)
            if model_filter and m not in model_filter:
                continue
            if taxonomy_filter and t not in taxonomy_filter:
                continue
            group_key = f"{method_label}/{m}/{t}"
            per_group_vids.setdefault(group_key, set()).add(vid)
        if per_group_vids:
            common_video_ids = set.intersection(*per_group_vids.values())
        else:
            common_video_ids = set()
        print(f"Common videos across {sorted(per_group_vids.keys())}: "
              f"{sorted(common_video_ids)} ({len(common_video_ids)})")

    # Phase 1: Run evaluation on all prediction files and save metrics
    all_metrics = []  # list of (method_label, output_dict)
    n_evaluated = 0

    # Group entries by method_label for display
    from collections import OrderedDict
    entries_by_method = OrderedDict()
    for method_label, pf in all_pred_entries:
        entries_by_method.setdefault(method_label, []).append(pf)

    for method_label, pred_files in entries_by_method.items():
        # Filter to only annotated videos (and optionally by model / common set)
        filtered = []
        for pf in pred_files:
            pred_frames = load_predictions(pf)
            vid = extract_video_id_from_pred(pred_frames)
            if not (vid and vid in annotated_video_ids):
                continue
            if model_filter:
                m = extract_model_from_pred(pred_frames, pf)
                if m not in model_filter:
                    continue
            if taxonomy_filter:
                t = extract_taxonomy_from_pred(pred_frames, pf)
                if t not in taxonomy_filter:
                    continue
            if common_video_ids is not None and vid not in common_video_ids:
                continue
            filtered.append((pf, pred_frames, vid))

        if not filtered:
            continue

        print(f"\n  {method_label}/: {len(filtered)} files (of {len(pred_files)} total) match filters")

        for pred_path, pred_frames, video_id in filtered:
            pred_basename = os.path.basename(pred_path)
            if args.verbose:
                print(f"\n    Evaluating {pred_basename} ...")

            output, metrics_path = evaluate_single_file(
                pred_path, actions_dir, args.labels_dir, args.threshold, args.verbose,
                method_label=method_label, max_frames=args.max_frames,
                action_sources=action_sources,
            )
            if output:
                all_metrics.append((method_label, output))
                n_evaluated += 1
                if not args.verbose:
                    print(f"    {pred_basename} -> {os.path.relpath(metrics_path, ROOT_DIR)}")

    print(f"\nEvaluated {n_evaluated} prediction files.")

    # Phase 2: Always use freshly-computed Phase 1 results to avoid mixing
    # with stale metrics from previous runs on disk.
    all_existing = all_metrics
    print(f"Using {len(all_existing)} freshly-computed metrics")

    # Phase 3: Aggregate and print
    summaries = aggregate_metrics(all_existing)

    # Inject n_total per group when --common-only so tables can show total vs common
    total_per_group = None
    if args.common_only and per_group_vids:
        total_per_group = {k: len(v) for k, v in per_group_vids.items()}
        for key, summary in summaries.items():
            if key in total_per_group:
                summary["n_total"] = total_per_group[key]

    print_summary_table(summaries)

    # Save aggregated summary
    agg_path = os.path.join(ROOT_DIR, "metrics", "aggregated_summary.json")
    os.makedirs(os.path.dirname(agg_path), exist_ok=True)
    with open(agg_path, "w") as f:
        json.dump(summaries, f, indent=2, default=_to_serializable)
    print(f"Aggregated summary saved to {agg_path}")


if __name__ == "__main__":
    main()
