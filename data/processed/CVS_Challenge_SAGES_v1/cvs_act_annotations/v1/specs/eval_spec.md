# Multi-Granularity F1@k Evaluation Specification

## Overview

Evaluate predicted action segments against ground truth at three label granularity levels (exact, medium, coarse) crossed with three temporal overlap thresholds (IoU@10, IoU@25, IoU@50). Compute per-video F1, then average across videos.

---

## 1. Label Granularity Mappings

### 1.1 Left Retraction

**Exact** — use the full code as-is:
```
KEEP_RETRACT_LATERAL
KEEP_RETRACT_MEDIAL
KEEP_RETRACT_UPWARD
RETRACT_LATERAL
RETRACT_LATERAL_TO_MEDIAL
RETRACT_LATERAL_TO_UPWARD
RETRACT_MEDIAL
RETRACT_MEDIAL_TO_LATERAL
RETRACT_UPWARD_TO_LATERAL
```

**Medium** — collapse to final direction (the direction the retraction ends at):
```
KEEP_RETRACT_LATERAL      → lateral
RETRACT_LATERAL            → lateral
RETRACT_MEDIAL_TO_LATERAL  → lateral
RETRACT_UPWARD_TO_LATERAL  → lateral

KEEP_RETRACT_MEDIAL        → medial
RETRACT_MEDIAL             → medial
RETRACT_LATERAL_TO_MEDIAL  → medial

KEEP_RETRACT_UPWARD        → upward
RETRACT_LATERAL_TO_UPWARD  → upward
```

**Coarse** — collapse to maintain vs change:
```
KEEP_RETRACT_LATERAL       → maintain
KEEP_RETRACT_MEDIAL        → maintain
KEEP_RETRACT_UPWARD        → maintain

RETRACT_LATERAL            → change
RETRACT_LATERAL_TO_MEDIAL  → change
RETRACT_LATERAL_TO_UPWARD  → change
RETRACT_MEDIAL             → change
RETRACT_MEDIAL_TO_LATERAL  → change
RETRACT_UPWARD_TO_LATERAL  → change
```

### 1.2 Camera

**Exact** — use the full code as-is:
```
CAMERA_ZOOM_IN
CAMERA_ZOOM_OUT
CAMERA_REPOSITION
CAMERA_UNCERTAIN
CAMERA_NO_CHANGE
```

**Medium** — collapse zoom directions and merge uncertain into no_change:
```
CAMERA_ZOOM_IN    → zoom
CAMERA_ZOOM_OUT   → zoom

CAMERA_REPOSITION → reposition

CAMERA_UNCERTAIN  → no_change
CAMERA_NO_CHANGE  → no_change
```

**Coarse** — binary change vs no_change:
```
CAMERA_ZOOM_IN    → change
CAMERA_ZOOM_OUT   → change
CAMERA_REPOSITION → change

CAMERA_UNCERTAIN  → no_change
CAMERA_NO_CHANGE  → no_change
```

### 1.3 Right Tool

**Exact** — full tuple: `(tool_type, action_code, target_structure, target_context)`

Example: `(Hook, DISSECT, CysticDuct, "between presumed cystic duct and presumed cystic artery")`

**Medium** — drop target_context, keep triplet: `(tool_type, action_code, target_structure)`

Example: `(Hook, DISSECT, CysticDuct)`

**Coarse** — drop tool_type and target_context, keep: `(action_code, target_structure)`

Example: `(DISSECT, CysticDuct)`

Full coarse collapse — all exact tuples that share the same (action_code, target_structure) are treated as the same label regardless of tool:
```
(Hook, DISSECT, CysticArtery, *)           → (DISSECT, CysticArtery)
(Maryland, DISSECT, CysticArtery, *)        → (DISSECT, CysticArtery)
(Hook, DISSECT, CysticDuct, *)              → (DISSECT, CysticDuct)
(Maryland, DISSECT, CysticDuct, *)          → (DISSECT, CysticDuct)
(Hook, DISSECT, CysticPlate, *)             → (DISSECT, CysticPlate)
(Hook, DISSECT, HepatocysticTriangle, *)    → (DISSECT, HepatocysticTriangle)
(Maryland, DISSECT, HepatocysticTriangle, *)→ (DISSECT, HepatocysticTriangle)
(Scissors, DISSECT, HepatocysticTriangle, *)→ (DISSECT, HepatocysticTriangle)
(Hook, COAGULATE_HEMOSTASIS, CysticArtery, *)           → (COAGULATE_HEMOSTASIS, CysticArtery)
(Hook, IRRIGATOR_COUNTERTRACTION_ASSIST, CysticArtery, *)→ (IRRIGATOR_COUNTERTRACTION_ASSIST, CysticArtery)
(Hook, IRRIGATOR_COUNTERTRACTION_ASSIST, CysticPlate, *)→ (IRRIGATOR_COUNTERTRACTION_ASSIST, CysticPlate)
(Irrigator, IRRIGATOR_ASPIRATE, HepatocysticTriangle, *)→ (IRRIGATOR_ASPIRATE, HepatocysticTriangle)
(Irrigator, IRRIGATOR_COUNTERTRACTION_ASSIST, GallbladderNeck_Infundibulum, *) → (IRRIGATOR_COUNTERTRACTION_ASSIST, GallbladderNeck_Infundibulum)
(Hook, RETRACT_DOWNWARD, HepatocysticTriangle, *)       → (RETRACT_DOWNWARD, HepatocysticTriangle)
(Hook, TOOL_WITHDRAW_UNBLOCKS_VIEW, CysticPlate, *)     → (TOOL_WITHDRAW_UNBLOCKS_VIEW, CysticPlate)
(Maryland, TOOL_WITHDRAW_UNBLOCKS_VIEW, CysticPlate, *) → (TOOL_WITHDRAW_UNBLOCKS_VIEW, CysticPlate)
(Hook, TOOL_WITHDRAW_UNBLOCKS_VIEW, HepatocysticTriangle, *)    → (TOOL_WITHDRAW_UNBLOCKS_VIEW, HepatocysticTriangle)
(Maryland, TOOL_WITHDRAW_UNBLOCKS_VIEW, HepatocysticTriangle, *)→ (TOOL_WITHDRAW_UNBLOCKS_VIEW, HepatocysticTriangle)
(Maryland, sweeping, HepatocysticTriangle, *)            → (sweeping, HepatocysticTriangle)
(clipper, CLIP, CysticDuct, *)              → (CLIP, CysticDuct)
(clipper, CLIP, HepatocysticTriangle, *)    → (CLIP, HepatocysticTriangle)
```

### 1.4 ICG_SWITCH

Only one code, no granularity levels. Evaluate with temporal IoU only (always exact match on label).

---

## 2. Segment IoU Computation

For predicted segment (s_p, e_p) and ground truth segment (s_g, e_g):

```python
def compute_iou(s_p, e_p, s_g, e_g):
    intersection = max(0, min(e_p, e_g) - max(s_p, s_g))
    union = max(e_p, e_g) - min(s_p, s_g)
    if union == 0:
        return 0.0
    return intersection / union
```

---

## 3. F1@k Computation

For each action type (left, camera, right) and each granularity level (exact, medium, coarse):

```python
def f1_at_k(pred_segments, gt_segments, k, granularity_map):
    """
    pred_segments: list of (start_frame, end_frame, label)
    gt_segments:   list of (start_frame, end_frame, label)
    k:             IoU threshold (0.1, 0.25, or 0.5)
    granularity_map: function that maps a label to its collapsed version
    """
    # Step 1: Apply granularity mapping to both pred and gt labels
    pred_mapped = [(s, e, granularity_map(l)) for s, e, l in pred_segments]
    gt_mapped   = [(s, e, granularity_map(l)) for s, e, l in gt_segments]

    tp, fp = 0, 0
    matched_gt = set()

    # Step 2: Greedy matching — for each prediction, find best IoU ground truth
    # Sort predictions by start frame for deterministic ordering
    for s_p, e_p, l_p in sorted(pred_mapped, key=lambda x: x[0]):
        best_iou = 0.0
        best_idx = None
        for i, (s_g, e_g, l_g) in enumerate(gt_mapped):
            if i in matched_gt:
                continue
            if l_p != l_g:
                continue
            iou = compute_iou(s_p, e_p, s_g, e_g)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou >= k and best_idx is not None:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn = len(gt_mapped) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1, precision, recall, tp, fp, fn
```

### Per-video, then average:
```python
def evaluate(all_videos, k_values, granularity_levels):
    """
    all_videos: dict of video_id -> (pred_segments, gt_segments)
    k_values: [0.1, 0.25, 0.5]
    granularity_levels: dict of level_name -> granularity_map function
    """
    results = {}
    for level_name, g_map in granularity_levels.items():
        for k in k_values:
            f1_scores = []
            for vid, (preds, gts) in all_videos.items():
                f1, _, _, _, _, _ = f1_at_k(preds, gts, k, g_map)
                f1_scores.append(f1)
            results[(level_name, k)] = sum(f1_scores) / len(f1_scores)
    return results
```

---

## 4. Output Tables

### 4.1 Per action type — one table each for Left, Camera, Right:

```
Left Retraction F1
             | IoU@10 | IoU@25 | IoU@50
  Exact      |  ...   |  ...   |  ...
  Medium     |  ...   |  ...   |  ...
  Coarse     |  ...   |  ...   |  ...

Camera F1
             | IoU@10 | IoU@25 | IoU@50
  Exact      |  ...   |  ...   |  ...
  Medium     |  ...   |  ...   |  ...
  Coarse     |  ...   |  ...   |  ...

Right Tool F1
             | IoU@10 | IoU@25 | IoU@50
  Exact      |  ...   |  ...   |  ...
  Medium     |  ...   |  ...   |  ...
  Coarse     |  ...   |  ...   |  ...
```

### 4.2 Also report supplementary metrics per action type:

- **Edit Score** (Levenshtein distance on the sequence of mapped action labels, normalized): captures segment ordering independent of boundary precision
- **Frame-level accuracy**: percentage of frames with correct mapped label (for reference, but less informative than F1@k)

### 4.3 Aggregate table (optional):

Define a segment as having a combined label = (left_label, camera_label, right_label). Evaluate the combined label at each granularity level — a TP requires ALL three components to match at that level AND IoU ≥ k.

```
Combined F1 (all three action components must match)
             | IoU@10 | IoU@25 | IoU@50
  Exact      |  ...   |  ...   |  ...
  Medium     |  ...   |  ...   |  ...
  Coarse     |  ...   |  ...   |  ...
```

---

## 5. Notes

- Each action type (left, camera, right) has its own set of segments — they may have different segment boundaries from each other. Evaluate each independently.
- If a video has no ground truth segments for an action type, skip that video for that action type's average.
- If a video has no predictions for an action type, F1 = 0 for that video (FN = number of gt segments).
- For right tool: if target_context is "(not set)" or missing, treat it as a distinct value at exact level. At medium and coarse levels it is dropped anyway.
- Report mean ± std across videos if variance is informative.
