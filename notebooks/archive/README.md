# CVS-Act synthetic-data pipeline audit

## Bottom line

The current generator is **v3.1 structured LLM extraction**: Qwen3.6 views a transition clip as a temporally ordered frame bundle, receives per-anchor CVS reviewer scores as context, segments the clip, and then converts the segmentation into the controlled left/right/camera/other taxonomy. A second morphology questionnaire can correct one known right-tool confusion, but only with the narrow rule `Hook -> Maryland` when both stages support it.

This was the best defensible version at the decision point, not a finished high-accuracy labeler. On the fixed 10-video validation split, its selected score was `0.428`; on the held-out 20-video test split it fell to `0.321` (`left exact=0.461`, `right medium=0.308`, `camera coarse=0.194`). Synthetic GT ranked six downstream systems similarly to human GT (Pearson `0.947`, Spearman `0.886`), but this is only six points and Pearson's video-bootstrap 95% interval is wide (`0.270–0.968`). Ranking correlation is not label accuracy.

**Recommendation: do not publish the generated labels as validated ground truth yet.** A private/preview release explicitly marked as unverified synthetic annotations is defensible. A public ground-truth-style release needs the expert audit and release gates in [What remains](#what-remains-before-a-final-public-release).

The original headline numbers below are regenerated from archived artifacts by [`pipeline_metrics.ipynb`](pipeline_metrics.ipynb); the interval-anchored metrics are loaded and audited by [`overlap_metrics.ipynb`](overlap_metrics.ipynb). “Measured” means present in a result artifact; “inferred” identifies a conclusion not directly tested.

## Current v3.1 pipeline

1. Build one clip around each CVS transition for C1, C2, or C3.
2. Sample anchors every 150 frames (5 seconds at 30 fps), four evenly spaced in-between frames per anchor interval, and the clip endpoint.
3. Label anchor images with `C1=k/3, C2=k/3, C3=k/3`, where `k` is the number of the three CVS raters calling the criterion satisfied. In-between images have frame numbers only.
4. Send the ordered images, the system prompt, CVS context, and the structured segmentation prompt to `Qwen/Qwen3.6-35B-A3B-FP8` with temperature 0 and thinking disabled.
5. Make a second text-only LLM call that maps the segmentation JSON to the fixed taxonomy. Invalid values are dropped, timestamps are clipped to the clip range, and adjacent identical segments are merged.
6. Optionally run the right-tool morphology checker over the full clip. v3.1 changes a predicted `Hook` to `Maryland` only when that checker predicts `Maryland`; every other checker prediction is ignored.
7. Export only the `structured_prediction_cvs_context` records, split the 1,312 training records by video into 1,119 train / 193 validation records, and keep 481 test records.

The implementation lives in [`src/cvs_act/action_segment_eval.py`](../../src/cvs_act/action_segment_eval.py), and the exact generation notebooks are under [`current_pipeline/`](current_pipeline/).

The generation notebooks connect to an internal OpenAI-compatible endpoint and read its credential from the `OPENAI_API_KEY` environment variable; no API key is stored in the notebooks.

### Important implementation discrepancy

The evaluated 90-clip run genuinely differs from v3: it applied 39 overrides across four method variants, including 10 of 90 structured predictions, and raised structured aggregate F1 from `0.323` to `0.330`. The later full-export notebooks also contain and executed the v3.1 post-processing wrapper, but it changed no exported labels. The full train export logged **0 applied overrides for 1,312 structured examples** because none had a matching rubric-cache entry. The full test export logged **0 applied overrides for 481 examples**; only 90 had any rubric entry, and none met the exact condition `base clip tool = Hook` plus `rubric tool = Maryland`. Therefore, the full-export labels equal their immediately preceding base-v3 full-run predictions apart from v3.1 provenance/metadata; this does **not** mean that the separately evaluated 90-clip v3.1 artifact equals its v3 source. Checker coverage must be completed and the full exports regenerated before claiming that the released dataset received the v3.1 correction.

## Why this version was chosen

### Action-generation lineage (same 90 development clips)

| Version | Structured spec F1 | What changed | Decision | Notebook |
|---|---:|---|---|---|
| Base | 0.195 | No per-frame CVS context | Rejected; weakest structured result. | [`seg_extract_spec_eval.ipynb`](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval.ipynb) |
| CVS context v1 | 0.302 | Added criterion/context information | Clear gain; retained direction. | [`with_cvs_context.ipynb`](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_with_cvs_context.ipynb) |
| CVS context v2 | 0.326 | Refined context/rules | Small further gain. | [`cvsctx_v2.ipynb`](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v2.ipynb) |
| CVS context v3 | 0.323 | Added stable-vs-transition left rules and detailed right-action rules | Similar aggregate, but became the selected prompt family based on the contemporaneous evaluation and qualitative error analysis. | [`cvsctx_v3.ipynb`](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v3.ipynb) |
| **v3.1** | **0.330** | Narrow clip-level Hook-to-Maryland correction | Chosen: measurable right-actor improvement with a conservative edit. | [`cvsctx_v3_1.ipynb`](current_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v3_1.ipynb) |
| v3.2 | 0.329 | Ran the morphology checker per predicted interval; 70 rewrites vs 39 for v3.1 | Not adopted: no measured aggregate gain despite more calls and edits. | [`cvsctx_v3_2.ipynb`](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v3_2.ipynb) |
| v4 | 0.340 | Added explicit camera start/end state questions | Not adopted: right structured F1 regressed from 0.211 (v3) / 0.231 (v3.1) to 0.168, camera only moved from 0.168 to 0.171, and selection was still on the same development set. | [`cvsctx_v4.ipynb`](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v4.ipynb) |
| v3.1-left/right + v4-camera hybrid | not measured | Spliced actor outputs | Not adopted because the notebook only generated the hybrid; it did not establish an independent quantitative improvement. | [`hybrid_v3_1_left_right_v4_camera.ipynb`](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_hybrid_v3_1_left_right_v4_camera.ipynb) |

### Comparability audit

These are end-to-end pipeline comparisons, not all single-variable prompt ablations. The notebooks use the same 90 records, model ID, temperature 0, disabled thinking, taxonomy, four output strategies, and evaluator unless stated below. Temperature 0 also does not guarantee bitwise deterministic model output. The notebooks import shared prompt/message helpers rather than snapshotting their source revision, so historical reproducibility additionally depends on the repository version used when each cache was generated.

| Comparison | What stayed controlled | Changes besides the shorthand version name | Valid interpretation |
|---|---|---|---|
| Base → CVS context v1 | Same 90 clips, images, model settings, taxonomy, output strategies, and evaluator | Adds the CVS definitions/clip-context prefix **and** changes anchor-frame labels from frame number only to frame number plus `C1/C2/C3 = k/3`; cache/method identifiers also change | Measures the combined benefit of CVS text plus reviewer-score frame labels. It does **not** isolate CVS definitions from score labels. |
| CVS v1 → v2 | Same CVS segmentation wrapper and scored frame labels | Adds a left-only extraction rule distinguishing brief direction changes from stable multidirectional exposure | Primarily a left extraction-rule ablation, not an additional CVS-context ablation. |
| CVS v2 → v3 | Same segmentation prompt, CVS wrapper, score labels, and v2 left rule | Adds ten detailed right-tool/action/target extraction rules | Measures the effect of the expanded right extraction instructions within the same general pipeline. |
| v3 → v3.1 | Reuses the cached v3 predictions | Applies only the deterministic clip-level `Hook → Maryland` disagreement rule | Closest to a controlled causal comparison; ten of 90 structured rows changed. |
| v3.1 → v3.2 | Both start from the same v3 source family and use the morphology rubric | Changes from one clip-level decision to per-predicted-interval checking/recombination, increasing calls and rewrites | Compares two checker/post-processing strategies, not two generation prompts. |
| v3 → v4 | Same CVS wrapper, frame labels, and v3 left/right extraction rules | Expands the segmentation camera questions and structured schema with start/end wide, close, and centered view states, then reruns joint generation | Camera-focused intervention, but the joint response can change non-camera actors too; do not attribute the aggregate difference solely to camera accuracy. |
| v3.1 + v4 hybrid | Splices already generated actor outputs | Uses v3.1 left/right and v4 camera rather than rerunning a unified prompt | Construction only; no independent metric was produced. |

Sources: the archived `spec_f1_overall_aggregate.csv` and actor tables in [`results/current/`](results/current/); v3.1/v3.2 output cells and override logs; [`pipeline_metrics.ipynb`](pipeline_metrics.ipynb). All action-lineage values above reuse the same 90 human-labeled clips, so they are development comparisons, not independent test estimates.

### Interval-anchored F1@any and F1@max

The historical evaluator is prediction-first, greedy, one-to-one segment matching. Despite occasionally being described as F1@0.1/0.2/0.5, the actual code and saved tables use IoU thresholds **0.10, 0.25, and 0.50**. The new metrics below do not replace those stricter detection metrics; they answer whether the label is plausible somewhere within each interval without requiring a particular IoU.

- **Prediction-anchored accuracy** (precision-like): for every predicted interval, judge it against overlapping GT intervals.
- **GT-anchored accuracy** (recall-like): for every GT interval, judge it against overlapping predicted intervals.
- **F1**: harmonic mean of those two directional rates per video, followed by the same macro-average over GT-positive videos used by the historical evaluator.
- **F1@any**: an anchor is correct when at least one opposing interval has positive temporal overlap and a matching label.
- **F1@max**: clip opposing intervals to the anchor, group them by label, union overlaps within each label, and sum disconnected pieces. The anchor is correct when its label is among the labels with the greatest accumulated overlap duration. A tie that includes the correct label counts as correct.

Intervals use half-open `[start_frame, end_frame)` boundaries, so merely touching endpoints do not overlap. Matching is deliberately **not one-to-one**: one interval may make several opposite-direction anchors correct. This makes both metrics diagnostic and lenient. In particular, F1@any can reward fragmented predictions, while F1@max ignores how much of the anchor is covered once the correct label has the largest share. Always report both directional rates and retain IoU F1 for publication-quality evaluation.

On the same 90 development clips, using each prompt family's structured LLM extraction output:

| Pipeline | F1@any | Pred-anchored @any | GT-anchored @any | F1@max | Pred-anchored @max | GT-anchored @max |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0.260 | 0.265 | 0.286 | 0.232 | 0.255 | 0.249 |
| CVS context v1 | 0.400 | 0.405 | 0.447 | 0.373 | 0.389 | 0.411 |
| CVS context v2 | 0.406 | 0.410 | 0.444 | 0.384 | 0.398 | 0.410 |
| CVS context v3 | 0.417 | 0.418 | 0.466 | 0.387 | 0.404 | 0.416 |
| **v3.1** | **0.426** | **0.426** | **0.477** | **0.394** | **0.410** | **0.427** |
| v3.2 | 0.426 | 0.426 | 0.479 | 0.395 | 0.410 | 0.429 |
| v4 | **0.453** | 0.440 | **0.517** | **0.412** | **0.425** | **0.449** |

That broad average contains all ten actor/granularity rows (left, right, and camera at exact/medium/coarse plus other at exact). Under the release-score components used elsewhere—left exact, right medium, and camera coarse—the comparison is:

| Pipeline | F1@any | F1@max |
|---|---:|---:|
| Base | 0.234 | 0.196 |
| CVS context v1 | 0.384 | 0.347 |
| CVS context v2 | 0.408 | 0.376 |
| CVS context v3 | 0.425 | 0.373 |
| **v3.1** | **0.448** | **0.393** |
| v3.2 | **0.450** | **0.395** |
| v4 | 0.411 | 0.361 |

For current v3.1 on those three components, F1@any has prediction-/GT-anchored rates `0.449/0.525`; F1@max has `0.419/0.443`. v3.2 is numerically only `0.002` above v3.1 on both new F1 summaries, so these metrics do not establish a meaningful v3.2 improvement. v4's broad average is highest, but its release-component score is lower because its right-medium result regresses; this is another reason not to select a pipeline on one global average.

Exact component rows are in [`overlap_metrics_by_actor_granularity.csv`](results/current/overlap_metrics_by_actor_granularity.csv), summaries are in [`overlap_metrics_summary.csv`](results/current/overlap_metrics_summary.csv), and per-video counts/rates plus SHA-256 provenance are in [`overlap_metric_results.json`](results/current/overlap_metric_results.json). Recompute them with [`scripts/evaluate_overlap_metrics.py`](../../scripts/evaluate_overlap_metrics.py). The comparison uses the GT conversion archived inside the original base experiment; that file is byte-identical across the base, CVS-v1, CVS-v2, CVS-v3, and CVS-v4 experiment directories. A separately regenerated shared GT artifact differs in five right-action records and three camera records, so it is intentionally not mixed into this comparison. The raw prediction files remain in the preserved legacy artifact tree and are identified by path and hash in the JSON; they are not duplicated into this Git archive.

### Prompt/output-format ablations

Each version notebook evaluated the same four output strategies. Structured segmentation followed by an LLM taxonomy-extraction call was consistently strongest.

| Prompt family | Description | Hint questions | Structured LLM extraction | Structured deterministic extraction | Notebook |
|---|---:|---:|---:|---:|---|
| No CVS context | 0.165 | 0.156 | **0.195** | 0.158 | [`seg_extract_spec_eval.ipynb`](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval.ipynb) |
| CVS context v1 | 0.185 | 0.262 | **0.302** | 0.271 | [`with_cvs_context.ipynb`](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_with_cvs_context.ipynb) |
| CVS context v2 | 0.215 | 0.269 | **0.326** | 0.275 | [`cvsctx_v2.ipynb`](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v2.ipynb) |
| CVS context v3 | 0.199 | 0.282 | **0.323** | 0.276 | [`cvsctx_v3.ipynb`](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v3.ipynb) |
| CVS context v4 | 0.186 | 0.299 | **0.340** | 0.263 | [`cvsctx_v4.ipynb`](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v4.ipynb) |

### Right-tool morphology ablations

The separate right-tool experiment measured 88 evaluable clips. All variants and their prompt construction are in [`qwen3.5_audit_v11_right_tool_rubric_clip_eval_1fps.ipynb`](experimental_action_pipeline/qwen3.5_audit_v11_right_tool_rubric_clip_eval_1fps.ipynb).

| Right-tool method | Clip-tool accuracy | Decision |
|---|---:|---|
| Original v3 structured prediction | **0.591** | Retained as the primary prediction. |
| All morphology questions in one JSON | 0.398 | Rejected as a replacement. |
| All questions in one JSON + 10 shape examples | **0.568** | Best checker, but still below original v3. Used only for the narrow v3.1 disagreement rule. |
| Separate per-question calls with examples | 0.409 | Rejected; worse and more expensive. |
| Separate per-question calls without examples | 0.386 | Rejected; weakest checker. |

The checker was therefore not trusted as a wholesale replacement. The narrow Hook-to-Maryland rule nevertheless raised structured right-actor F1 from `0.211` to `0.231` and aggregate F1 from `0.323` to `0.330`. This selective correction is the main empirical reason for v3.1.

## Earlier `lvlm_cvs_reasoning` work

This older repository studied **CVS achievement classification**, not timestamped action-label generation. It is historical motivation for temporal/CVS context, but its mAP values are not v3.1 action-label accuracy.

On five development videos / 90 frames with GPT-4.1-mini:

| Input/protocol | C1 AP | C2 AP | C3 AP | Mean AP | Notebook |
|---|---:|---:|---:|---:|---|
| Single frame, direct, no organs | 0.258 | 0.431 | 0.222 | 0.304 | [`visualize_results_initial_ablations.ipynb`](lvlm_cvs_reasoning/visualize_results_initial_ablations.ipynb) |
| Sequential, direct, no organs | 0.368 | 0.484 | 0.334 | **0.395** | [`visualize_results_initial_ablations.ipynb`](lvlm_cvs_reasoning/visualize_results_initial_ablations.ipynb) |
| Single frame, direct, organ overlay | 0.249 | 0.450 | 0.339 | 0.346 | [`visualize_results_initial_ablations.ipynb`](lvlm_cvs_reasoning/visualize_results_initial_ablations.ipynb) |
| Sequential, direct, organ overlay | 0.233 | 0.452 | 0.261 | 0.316 | [`visualize_results_initial_ablations.ipynb`](lvlm_cvs_reasoning/visualize_results_initial_ablations.ipynb) |
| Single frame, observe-then-decide, no organs | 0.311 | 0.425 | 0.222 | 0.319 | [`visualize_results_initial_ablations.ipynb`](lvlm_cvs_reasoning/visualize_results_initial_ablations.ipynb) |
| Sequential, observe-then-decide, no organs | 0.366 | **0.528** | 0.291 | **0.395** | [`visualize_results_initial_ablations.ipynb`](lvlm_cvs_reasoning/visualize_results_initial_ablations.ipynb) |

Comparability caveat: these rows use the same model family and 90-frame/five-video development sample, but they are end-to-end configurations. “Sequential” changes the frame bundle, “observe-then-decide” changes the reasoning protocol, and “organ overlay” changes the visual input; combinations therefore alter more than one token-level detail relative to the single-frame direct baseline. The archived metric files do not pin a complete source/environment revision, so these results support exploratory directional conclusions rather than a strict isolated causal claim.

Measured conclusion: sequential context improved direct mean AP from `0.304` to `0.395`; adding the particular organ overlay to the sequential direct run reduced it to `0.316`. Observe-then-decide and sequential direct were essentially tied in mean AP (`0.395210` vs `0.395228`) even though the former improved C2.

The old qualitative note selected sequential observe-then-decide without organs because it emphasized recall. That recall claim is **not reproduced by the archived `metrics.json` files**, which contain AP rather than precision/recall/F1, and the experiment had only five development videos. Treat “organ metadata causes premature anchoring” and “observation-first is necessary” as hypotheses inferred from exploratory behavior, not confirmed findings.

The five-way organ-injection sweep used the same 90 frames and is preserved in [`visualize_results_organ_ablations.ipynb`](lvlm_cvs_reasoning/visualize_results_organ_ablations.ipynb):

| Organ representation | Mean AP | Notebook |
|---|---:|---|
| Segmentation contour overlay | **0.381** | [`visualize_results_organ_ablations.ipynb`](lvlm_cvs_reasoning/visualize_results_organ_ablations.ipynb) |
| Segmentation filled overlay | 0.351 | [`visualize_results_organ_ablations.ipynb`](lvlm_cvs_reasoning/visualize_results_organ_ablations.ipynb) |
| Bounding-box text/top-k | 0.319 | [`visualize_results_organ_ablations.ipynb`](lvlm_cvs_reasoning/visualize_results_organ_ablations.ipynb) |
| Centroid text/top-k | 0.305 | [`visualize_results_organ_ablations.ipynb`](lvlm_cvs_reasoning/visualize_results_organ_ablations.ipynb) |
| Statistical text/top-k | 0.298 | [`visualize_results_organ_ablations.ipynb`](lvlm_cvs_reasoning/visualize_results_organ_ablations.ipynb) |

Supporting historical notebooks: [`decisions.ipynb`](lvlm_cvs_reasoning/decisions.ipynb) records the contemporaneous qualitative choice; [`metadata_clean_pipeline.ipynb`](lvlm_cvs_reasoning/metadata_clean_pipeline.ipynb) contains the metadata-generation workflow; [`metadata_eval_all_criteria_by_version_train_dev.ipynb`](lvlm_cvs_reasoning/metadata_eval_all_criteria_by_version_train_dev.ipynb) evaluates metadata variants; and [`eval_all_models_summary_new.ipynb`](lvlm_cvs_reasoning/eval_all_models_summary_new.ipynb) collects the wider model summaries.

Archived notebooks are in [`lvlm_cvs_reasoning/`](lvlm_cvs_reasoning/), and their result files are in [`results/lvlm/`](results/lvlm/).

## Exact prompts

The bracketed values below are record-specific substitutions, not omitted instructions. The controlled vocabularies shown are the exact current HF taxonomy values. Images are interleaved after the user text with labels such as `Frame 002100, C1=1/3, C2=2/3, C3=0/3`.

### What “final v3.1 prompt” means

v3.1 does not introduce a new generation prompt. The selected prompt stack is the **v3 multimodal segmentation prompt plus the v3 structured LLM extraction prompt** reproduced below. v3.1 then applies a deterministic post-processing rule: if the v3 clip-level right tool is exactly `Hook` and the separate morphology checker predicts `Maryland`, rewrite the right-hand `Hook` labels to `Maryland`. Thus the complete selected pipeline has two generation calls, an optional checker call, and a deterministic correction; the correction itself is not prompt text.

The structured LLM extraction prompt is the second, text-only call. It does not see images or human GT. It receives the entire first-call segmentation response, the clip identifiers/range, complete allowed vocabularies, normalization rules, a strict output schema, and the additional v3 left/right disambiguation rules. Its purpose is to map descriptive segmentation JSON into the final controlled left/right/camera/other taxonomy.

### Prompt change history

| Version | Segmentation prompt/input change | Extraction/checking change | Notebook |
|---|---|---|---|
| Base | Default action segmentation prompt; frame-number labels only | Shared constrained extraction prompt | [`seg_extract_spec_eval.ipynb`](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval.ipynb) |
| CVS v1 | Prepends CVS definitions, target criterion/range, and context-use instruction; anchor labels also gain `C1/C2/C3 = k/3` reviewer scores | Otherwise uses the shared extraction prompt | [`with_cvs_context.ipynb`](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_with_cvs_context.ipynb) |
| CVS v2 | Same CVS segmentation wrapper and scored labels as v1 | Appends the left stable-versus-transition rule | [`cvsctx_v2.ipynb`](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v2.ipynb) |
| CVS v3 | Same segmentation prompt as v2 | Retains the left rule and appends ten right-tool/action/target rules | [`cvsctx_v3.ipynb`](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v3.ipynb) |
| v3.1 | No prompt change from v3 | Adds the clip-level morphology checker and deterministic `Hook → Maryland` rule | [`cvsctx_v3_1.ipynb`](current_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v3_1.ipynb) |
| v3.2 | No base-generation prompt change from v3 | Runs the morphology questionnaire per predicted right interval and recombines intervals | [`cvsctx_v3_2.ipynb`](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v3_2.ipynb) |
| v4 | Expands camera instructions and structured schema with start/end wide, close, and centered states | Retains the v3 left/right extraction rules | [`cvsctx_v4.ipynb`](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v4.ipynb) |

### Segmentation call: system message

```text
You are a surgical video understanding assistant. You will receive laparoscopic video frames in chronological order. Each image is preceded by a text label containing its exact frame number.
```

### Segmentation call: complete structured v3 user message

```text
CVS criteria definitions:
- C1: Two and only two tubular structures are visible entering the gallbladder.
- C2: The hepatocystic triangle is cleared of fat and fibrous tissue.
- C3: The lower third of the gallbladder is detached from the liver bed.

Per-frame CVS satisfaction labels:
Anchor frames (every 5 seconds, i.e. at the stride-150 frame numbers) are annotated by 3 reviewers and shown with labels like "Frame 002100, C1=1/3, C2=2/3, C3=0/3". The value k/3 is the number of reviewers (of 3) who say that criterion is satisfied at that frame: 3/3 means all three reviewers agree it is satisfied, 0/3 means none, 1/3 means exactly one of three. The interleaved finer-grained frames (between anchors) carry only their frame number with no score because they were not separately rated.

This clip primarily concerns criterion [C1|C2|C3]. Clip frame range: [START] to [END].

Use the CVS context only as clip-level context. Still base segment boundaries and action timing on the provided frames and visible frame labels.

The required anchor frames are sampled every 5 seconds from one continuous video segment. Optional in-between frames may also be included.

Task: segment the video into multiple frame-number ranges where each range contains one continuous different surgical action or no action.

Actions you should be reporting:
1. Did the grasper's retraction direction of the gallbladder neck change? If so, where did it start and where did it end? Examples of acceptable answers: (here left and right can be replaced with left/right/upward)
    - Started in the [left], and dragged to the [right] in the end.
    - Started not retracted, and then it became retracted to the [left].
    - Started retracted to the [left], and it was kept that way until the end.
    - Started retracted to the [left] a bit, and then it was retracted more that way.
    - If you are not sure, you can always report unsure instead of forcing an answer.
    - Try to say which direction the gallbladder is retracted to if you talk about the grasper retracting the gallbladder neck, or explicitly say unclear direction if you don't know.
2. Is there a tool actively operating in the area?
    - If a tool is just hanging there without interacting with any tissue, then you can ignore it.
    - Include: tool type, action, target.
    - Action: Operating includes dissecting, irrigating, aspirating, coagulating, retracting.
    - Tool type: The tool can be Maryland grasper, hook, scissors, irrigator, or clipper. Withdrawing is an action, not a tool type: use it only when the tool was previously blocking a region and is now removed from there.
    - Target: Target can be hepatocystic triangle if it is general in that area and the tissues are not clearly delineated, or be more specific as cystic duct, cystic artery, cystic plate, which describe the proxy area. Even if those ducts or regions are not perfectly delineated, if you see they are somewhat delineated and the tool is clearly operating onto those areas, then prefer the more specific regions. For withdrawal, if the tool started blocking only a smaller region (e.g. cystic plate) and then is withdrawn, then you should say it's withdrawn from the "cystic plate" instead of the more general "hepatocystic triangle".
    - Target context: Also describe where the target is relative to visible anatomy when it is clear, for example between cystic artery and cystic plate, near the cystic duct, close to the gallbladder neck, near the base of the hepatocystic triangle, behind the two tubular structures, or whether the view is moving from posterior to anterior / anterior to posterior. If no such context is visible, say not set or omit it.
    - If you are not sure, always report unsure instead of forcing an answer. In prose/table outputs, you can report unsure for some subset of (action, tool type and target) tuples and still have answers for others.
3. Is there any significant camera movements?
    - Did the camera zoom in, zoom out, reposition, moving in an unclear way, or not moving?
    - If you are not sure, you can always report unsure instead of forcing an answer.
4. Is the smaller segment blurry or obscured such that it's hard to judge?

Use only the provided frames and their text frame labels. Do not infer exact boundaries between sampled frames more precisely than the visible frame numbers.
For start_frame and end_frame, copy the actual frame numbers from the visible text labels, e.g. Frame 000150 means 150. Do not use the frame's zero-based or one-based position/index in the provided sequence.

Segment continuity requirements:
- Cover the full provided frame range with no gaps.
- The first segment must start at the first provided frame label.
- The final segment must end at the last provided frame label.
- Each segment's end_frame must equal the next segment's start_frame, e.g. 000150-000180, then 000180-000300. Do not skip intervals like 000150-000180 followed by 000210-000300.

If something changes, like the retraction direction of the gallbladder between two frames, you should also report that.

For each segment, report:
- start_frame: actual frame number from a provided text label, not a sequence index
- end_frame: actual frame number from a provided text label, not a sequence index
- label: action or no_action
- detailed action_description (3-5 sentences)

Right-tool ontology for active_tools:
- tool_type must be exactly one of: ["Hook", "Irrigator", "Maryland", "Scissors", "clipper"]
- action_code must be exactly one of: ["CLIP", "COAGULATE_HEMOSTASIS", "COUNTERTRACTION_ASSIST", "DISSECT", "IRRIGATOR_ASPIRATE", "RETRACT_DOWNWARD", "SWEEPING", "TOOL_WITHDRAW_UNBLOCKS_VIEW"]
- target_structure must be exactly one of: ["CysticArtery", "CysticDuct", "CysticPlate", "GallbladderNeck_Infundibulum", "HepatocysticTriangle"]
- target_context_1 and target_context_2 must each be exactly one of: ["between presumed cystic duct and presumed cystic artery", "between cystic artery and cystic plate", "near the base of the hepatocystic triangle", "close to the gallbladder neck", "near the cystic duct", "near the cystic plate", "(not set)", "on the outside of cystic duct"]
- Use "(not set)" for target_context_1 or target_context_2 when no listed context is visible.
- Only include an active_tools row when tool_type, action_code, and target_structure can all be selected from the listed options. If the active tool is uncertain, leave active_tools empty and explain the uncertainty in action_description.
- Use action_code "COUNTERTRACTION_ASSIST" when a tool, including an irrigator, provides temporary countertraction/pushing to assist exposure or regrasping.
- Use action_code "TOOL_WITHDRAW_UNBLOCKS_VIEW" when a tool is withdrawn because it was blocking the camera view or key anatomy.

Return STRICT JSON only. Do not wrap the JSON in Markdown fences.
Use schema:
{"segments": [{"start_frame": 150, "end_frame": 180, "label": "action | no_action | unsure", "action_description": "...", "grasper_retraction": {"changed": "yes | no | unsure", "start_direction": "left | right | upward | downward | not_retracted | unclear | unsure", "end_direction": "left | right | upward | downward | not_retracted | unclear | unsure", "description": "..."}, "active_tools": [{"tool_type": "Hook", "action_code": "DISSECT", "target_structure": "CysticArtery", "target_context_1": "(not set)", "target_context_2": "(not set)", "evidence": "short visible evidence"}], "camera_movement": "zoom_in | zoom_out | reposition | unclear_movement | not_moving | unsure", "visibility": "clear | blurry | obscured | blurry_and_obscured | unsure"}]}
```

### Extraction call: complete user-message template

The extraction call is text-only. `[PREDICTION_JSON]` is the entire first-call response.

System message:

```text
Extract constrained timestamped surgical actions. Return valid JSON only.
```

Complete user-message template:

```text
Convert the model's action segmentation output into simple timestamped actions using ONLY the allowed audit_v11 options.

Clip:
- example_id: [EXAMPLE_ID]
- frame range: [START]-[END]

Allowed left retraction_direction_code options:
["KEEP_RETRACT_LATERAL", "KEEP_RETRACT_MEDIAL", "KEEP_RETRACT_UPWARD", "RETRACT_LATERAL", "RETRACT_MEDIAL", "RETRACT_UPWARD", "RETRACT_LATERAL_TO_MEDIAL", "RETRACT_LATERAL_TO_UPWARD", "RETRACT_MEDIAL_TO_LATERAL", "RETRACT_MEDIAL_TO_UPWARD", "RETRACT_UPWARD_TO_LATERAL", "RETRACT_UPWARD_TO_MEDIAL"]

Allowed right options:
- tool_type must be exactly one of:
["Hook", "Irrigator", "Maryland", "Scissors", "clipper"]
- action_code must be exactly one of:
["CLIP", "COAGULATE_HEMOSTASIS", "COUNTERTRACTION_ASSIST", "DISSECT", "IRRIGATOR_ASPIRATE", "RETRACT_DOWNWARD", "SWEEPING", "TOOL_WITHDRAW_UNBLOCKS_VIEW"]
- target_structure must be exactly one of:
["CysticArtery", "CysticDuct", "CysticPlate", "GallbladderNeck_Infundibulum", "HepatocysticTriangle"]
- target_context_1 and target_context_2 must each be exactly one of:
["between presumed cystic duct and presumed cystic artery", "between cystic artery and cystic plate", "near the base of the hepatocystic triangle", "close to the gallbladder neck", "near the cystic duct", "near the cystic plate", "(not set)", "on the outside of cystic duct"]

Allowed camera action_code options:
["CAMERA_NO_CHANGE", "CAMERA_REPOSITION", "CAMERA_UNCERTAIN", "CAMERA_ZOOM_IN", "CAMERA_ZOOM_OUT"]

Allowed other action_code options:
["ICG_SWITCH"]

Rules:
- Use the prediction text only. Do not inspect GT labels.
- Preserve timestamps from the prediction.
- For left, output only retraction_direction_code.
- For right, output tool_type, action_code, target_structure, target_context_1, target_context_2.
- For camera, output only allowed camera action_code.
- Use other for actions such as ICG.
- If an actor is absent, use an empty list.
- If no allowed option matches, omit the row instead of inventing a new option.
- Return STRICT JSON only. No markdown.

Schema:
{"method": "structured_prediction_cvs_context", "left": [{"start_frame": 150, "end_frame": 300, "retraction_direction_code": "KEEP_RETRACT_LATERAL", "confidence": 0.0, "evidence": "short phrase"}], "right": [{"start_frame": 150, "end_frame": 300, "tool_type": "Hook", "action_code": "DISSECT", "target_structure": "HepatocysticTriangle", "target_context_1": "(not set)", "target_context_2": "(not set)", "confidence": 0.0, "evidence": "short phrase"}], "camera": [{"start_frame": 150, "end_frame": 300, "action_code": "CAMERA_ZOOM_OUT", "confidence": 0.0, "evidence": "short phrase"}], "other": [{"start_frame": 150, "end_frame": 300, "action_code": "ICG_SWITCH", "confidence": 0.0, "evidence": "short phrase"}]}

Prediction text from method `structured_prediction_cvs_context`:
[PREDICTION_JSON]

Additional left-only rule for this v3 run:
- Use a left change code such as RETRACT_LATERAL_TO_MEDIAL or RETRACT_MEDIAL_TO_LATERAL only for the short segment where the retraction direction visibly transitions from one direction to another.
- Do not use a change code for a long interval that is merely described as holding the gallbladder upward and laterally, medially and upward, or otherwise stably exposed in multiple directions.
- If the segment is mostly stable rather than an actual transition, prefer the matching KEEP_RETRACT_* or RETRACT_* endpoint direction code instead of a *_TO_* code.

Additional right-tool rules for this v3 run:
- First decide whether the right-hand tool is doing a meaningful action. If a tool is merely present without a meaningful interaction, do not force an action label.
- For right-hand tool actions, choose the most appropriate action, target, and target context using the following rules.

1. Dissection: choose the main target first.
- For dissection actions, the tool is usually operating in the hepatocystic triangle region.
- More specifically, it may be operating around the cystic artery, cystic duct, or cystic plate.
- If the hepatocystic triangle is not cleared at all and the tool is dissecting only in the general hepatocystic triangle region, report the target as HepatocysticTriangle. In that case, you do not need a more specific target.
- Even if the triangle is not completely cleared, if you can tell that the tool is operating mostly near the presumed cystic duct, presumed cystic artery, or cystic plate, then prefer the more specific target as the main target.

2. Dissection: add target context when the operating side is visible.
- If the tool is operating mostly on one side of the target, specify that side in the target context.
- For example, the hook may be dissecting near the cystic artery:
  - between presumed cystic duct and presumed cystic artery, or
  - between cystic artery and cystic plate.
- If the tool is operating on both sides, put one side in target_context_1 and the other in target_context_2. The order does not matter.

3. Use hepatocystic triangle with precise context when appropriate.
- If the two tubular structures are clear, but the tool is operating in the general region between them rather than trying to skeletonize either the cystic duct or the cystic artery, choose HepatocysticTriangle as the target and use between presumed cystic duct and presumed cystic artery as the context.
- If the tool is pushing down (retracting) or dissecting near the base of the hepatocystic triangle, that should also be included in the context when visible.

4. Countertraction assist.
- Sometimes an irrigator or a hook is helping the grasper that is retracting the gallbladder neck change direction.
- In that situation, you can label the action as COUNTERTRACTION_ASSIST.

5. Irrigator aspirating.
- If an irrigator is present and blood disappears from the scene, label the action as IRRIGATOR_ASPIRATE.

6. Tool withdrawn and view unblocked.
- If a tool is present at first, does not perform any meaningful action, and is then simply removed from the scene, you can label the action as TOOL_WITHDRAW_UNBLOCKS_VIEW.
- If the tool was blocking only part of the scene, then the target should be only the specific region that was blocked and then unblocked.
- For example, if it was blocking only the cystic plate, while the rest of the field was already visible, then the target should be CysticPlate, not the whole hepatocystic triangle.

7. Coagulating to stop bleeding.
- Sometimes an electrocautery tool may be used to coagulate blood and stop bleeding.
- In that case, label the action as COAGULATE_HEMOSTASIS.

8. Clipping.
- Sometimes a clipper may be trying to place a clip on the cystic duct or the cystic artery.
- In that case, label the action as CLIP and choose the appropriate target.

9. Sweeping.
- If the tool only sweeps across tissue, without actually dissecting or cutting, then the action may be sweeping.
- This can include sweeping blood or lightly moving across tissue without true dissection.

10. Retracting.
- Sometimes the tool is retracting rather than dissecting.
- This means it is gently pushing or holding tissue to improve visualization, without actually cutting or dissecting it.
```

### Additional right-tool checker

System message:

```text
You answer surgical instrument questionnaires from images. Return strict JSON only.
```

Complete questionnaire structure used by the winning checker (`prompt2_all_in_one_json_with_shapes`):

```text
You are a surgical instrument identification assistant.
You are viewing one or more frames from the same laparoscopic cholecystectomy clip.
The first image is the target frame. Additional images, when shown, are neighboring frames from about 1 second before or after.
Your task is to answer the exact questionnaire below about the actively working RIGHT-side instrument.

Rules:
- Answer every question with exactly one of: Yes, No, Uncertain.
- For the 10 instrument questions, judge the visible instrument morphology only.
- For the 3 frame-check questions, judge image visibility or instrument presence only.
- Use all provided frames together when they help resolve visibility or ambiguity.
- If the active working instrument is not clearly present, use Uncertain for morphology questions when needed.
- If you cannot clearly see evidence for either the Yes features or the No features, answer Uncertain.
- Do not skip any question.

You will also be shown 10 reference example images for the morphology questions. Use them as visual guidance for the shape concepts, then answer the questionnaire on the target frame and any provided neighboring frames.

Questions:
- instrument_1: Does the distal tube have hole-like openings? Yes: visible small round holes on a metal tube/shaft. No: no holes, non-metal texture, or blood artifacts. Uncertain: blurry/obscured/no operating tool.
- instrument_2: Does each visible tip have an inside cutting edge/blade? Yes: visible sharp inside blade. No: visibly blunt. Uncertain: buried/blurry/obscured/no operating tool.
- instrument_3: Is each metal tip tapered? Yes: width smoothly decreases. No: width is mostly constant. Uncertain: blurry/obscured/no operating tool.
- instrument_4: Does the distal end consist of a tube only, without a separate tip? Yes: just a tube. No: distinct tip(s). Uncertain: blurry/obscured/no operating tool.
- instrument_5: Does the tip split into two visible parts? Yes: two jaws/tips. No: one continuous tip. Uncertain: blurry/obscured/no operating tool.
- instrument_6: Is the tip wire-like (very thin and rod-like)? Yes: thin metal wire. No: flat, broad, or visibly thick. Uncertain: blurry/obscured/no operating tool.
- instrument_7: Is the tip hollow rather than solid? Yes: opening/lumen visible. No: visibly solid. Uncertain: blurry/obscured/no operating tool.
- instrument_8: Is a hinge/pivot visible where the tip attaches to the shaft? Yes: pivot/pin/joint visible. No: no joint visible. Uncertain: blurry/obscured/no operating tool.
- instrument_9: Do visible tips have repeated tooth-like serrations? Yes: saw/comb pattern. No: smooth/flat edge. Uncertain: blurry/obscured/no operating tool.
- instrument_10: Do two tips originate from a common pivot and open like a fan? Yes: meet at one joint and spread. No: stay parallel without convergence. Uncertain: base blurry/obscured/no operating tool.
- frame_check_1: Is the active working instrument (not the gallbladder retractor) present? Answer Yes, No, or Uncertain from visibility.
- frame_check_2: Is its tip obscured by tissue? Answer Yes, No, or Uncertain if the tip cannot be located.
- frame_check_3: Is its tip blurry? Answer Yes, No, or Uncertain if the tip cannot be located.

Return STRICT JSON only:
{"answers": {"instrument_1": "Yes|No|Uncertain", "instrument_2": "Yes|No|Uncertain", "instrument_3": "Yes|No|Uncertain", "instrument_4": "Yes|No|Uncertain", "instrument_5": "Yes|No|Uncertain", "instrument_6": "Yes|No|Uncertain", "instrument_7": "Yes|No|Uncertain", "instrument_8": "Yes|No|Uncertain", "instrument_9": "Yes|No|Uncertain", "instrument_10": "Yes|No|Uncertain", "frame_check_1": "Yes|No|Uncertain", "frame_check_2": "Yes|No|Uncertain", "frame_check_3": "Yes|No|Uncertain"}, "brief_reasoning": "1-3 short sentences"}
```

The actual multimodal user content then interleaves, for each of the ten morphology questions, a text block repeating its question/Yes/No definitions and its reference image, followed by all 1-fps clip images. Exact original wording and attachment construction are preserved in [`qwen3.5_audit_v11_right_tool_rubric_clip_eval_1fps.ipynb`](experimental_action_pipeline/qwen3.5_audit_v11_right_tool_rubric_clip_eval_1fps.ipynb) and `src/cvs_act/annotate/right_tool.py`.

### Verbatim additional-checker user text

The following is copied from the saved winning-method prompt record, without rewriting. The multimodal message appends ten reference-description/image pairs and then the 1-fps clip frames after this text.

```text
You are a surgical instrument identification assistant.
You are viewing one or more frames from the same laparoscopic cholecystectomy clip.
The first image is the target frame. Additional images, when shown, are neighboring frames from about 1 second before or after.
Your task is to answer the exact questionnaire below about the actively working RIGHT-side instrument.

Rules:
- Answer every question with exactly one of: Yes, No, Uncertain.
- For the 10 instrument questions, judge the visible instrument morphology only.
- For the 3 frame-check questions, judge image visibility or instrument presence only.
- Use all provided frames together when they help resolve visibility or ambiguity.
- If the active working instrument is not clearly present, use Uncertain for morphology questions when needed.
- If you cannot clearly see evidence for either the Yes features or the No features, answer Uncertain.
- Do not skip any question.

You will also be shown 10 reference example images for the morphology questions. Use them as visual guidance for the shape concepts, then answer the questionnaire on the target frame and any provided neighboring frames.

Questions:
- instrument_1: Does the distal tube have hole-like openings?
  Yes definition: Yes if there are one or more visible holes on the tube/shaft. Note that this should only be holes on a metal tube with metal color instead of other textured tubes. And the holes should be small round holes on the tube instead of wide rectangle openings without closure at the end of the tool.
  No definition: No if the tube/shaft surface does not have holes. If the tool is not metal, or if there is blood that looks like round holes, then those are not holes.
  Uncertain definition: Uncertain if the image is blurry or if the end of the tube/shaft is obscured by tissue or other tools. If there is no tool operating, then it should also be uncertain.
  Allowed answers: Yes, No, Uncertain
- instrument_2: Does the tip (each tip, if multiple) have a cutting edge (blade)?
  Yes definition: Yes if sharp cutting edges or blades are visible for the metal tips of the tool. We only consider blades between two metal tips, on the inside where two metal tips are adjacent to each other, instead of blades outside. Do not choose yes if the tips are buried in tissue and cannot be confirmed.
  No definition: No if the tips of the tool are blunt. Do not choose no if the tips are buried in tissue and cannot be confirmed.
  Uncertain definition: Uncertain if the image is blurry or if the tips of the tool are obscured by tissue or other tools. If there is no tool operating, then it should also be uncertain.
  Allowed answers: Yes, No, Uncertain
- instrument_3: Is the metal tip of the tool (each tip, if multiple) tapered?
  Yes definition: Yes if the width decreases smoothly toward the tip (tapered).
  No definition: No if the width is mostly constant or only slightly changes (not tapered).
  Uncertain definition: Uncertain if the image is blurry or if the tips of the tool are obscured by tissue or other tools. If there is no tool operating, then it should also be uncertain.
  Allowed answers: Yes, No, Uncertain
- instrument_4: Does the distal end consist of a tube only (without any separate tip structure)?
  Yes definition: Yes if the end is just a tube.
  No definition: No if there is a distinct tip or multiple tips attached.
  Uncertain definition: Uncertain if the distal end is blurry or obscured by tissue or other tools, or if there is no clearly operating tool.
  Allowed answers: Yes, No, Uncertain
- instrument_5: Does the tip split into two visible parts?
  Yes definition: Yes if two separate tips or jaws are visible.
  No definition: No if only a single continuous tip is visible.
  Uncertain definition: Uncertain if the tip is blurry or obscured by tissue or other tools, or if there is no clearly operating tool.
  Allowed answers: Yes, No, Uncertain
- instrument_6: Does the tip appear wire-like (very thin and rod-like, much thinner than typical jaws or blades)?
  Yes definition: Yes if the tip resembles a thin metal wire.
  No definition: No if the tip is flat, broad, or has noticeable thickness.
  Uncertain definition: Uncertain if the tip is blurry or obscured by tissue or other tools, or if there is no clearly operating tool.
  Allowed answers: Yes, No, Uncertain
- instrument_7: Does the tip appear hollow (having an opening or lumen) rather than solid?
  Yes definition: Yes if an opening or inner cavity is visible.
  No definition: No if the tip appears solid with no opening.
  Uncertain definition: Uncertain if the tip is blurry or obscured by tissue or other tools, or if there is no clearly operating tool.
  Allowed answers: Yes, No, Uncertain
- instrument_8: Is a hinge (pivot joint) visible where the tip is attached to the shaft?
  Yes definition: Yes if a pivot, pin, or joint structure is visible between the tip and where it connects to the shaft.
  No definition: No if no such joint is visible.
  Uncertain definition: Uncertain if the hinge region is blurry or obscured by tissue or other tools, or if there is no clearly operating tool.
  Allowed answers: Yes, No, Uncertain
- instrument_9: Do any visible tips have serrations (small, repeated tooth-like ridges along the edge)?
  Yes definition: Yes if a saw or comb-like pattern is visible.
  No definition: No if the edge is smooth or flat.
  Uncertain definition: Uncertain if the edge is blurry or obscured by tissue or other tools, or if there is no clearly operating tool.
  Allowed answers: Yes, No, Uncertain
- instrument_10: Do the two tips originate from a common pivot point at the base (opening like a fan)?
  Yes definition: Yes if both tips meet at a single joint and spread apart from that point.
  No definition: No if the tips remain parallel and do not converge to a single pivot point.
  Uncertain definition: Uncertain if the base of the two tips is blurry or obscured by tissue or other tools, or if there is no clearly operating tool.
  Allowed answers: Yes, No, Uncertain
- frame_check_1: Is the active working instrument (the one performing dissection, cutting, aspiration, or similar, not the instrument retracting the gallbladder) present in the frame?
  Yes definition: Yes if the active working instrument is visibly present in the frame.
  No definition: No if the active working instrument is clearly absent from the frame.
  Uncertain definition: Uncertain if visibility is too poor or the frame does not provide enough evidence to decide whether the active working instrument is present.
  Allowed answers: Yes, No, Uncertain
- frame_check_2: Is the tip of the actively working instrument obscured by tissue?
  Yes definition: Yes if the tip of the active working instrument is covered or blocked by tissue.
  No definition: No if the tip of the active working instrument is not obscured by tissue.
  Uncertain definition: Uncertain if the instrument tip cannot be located clearly enough to judge whether tissue is obscuring it.
  Allowed answers: Yes, No, Uncertain
- frame_check_3: Is the tip of the actively working instrument blurry?
  Yes definition: Yes if the tip of the active working instrument appears blurry or out of focus.
  No definition: No if the tip of the active working instrument appears sharp enough to inspect.
  Uncertain definition: Uncertain if the instrument tip cannot be located clearly enough to judge whether it is blurry.
  Allowed answers: Yes, No, Uncertain

Return STRICT JSON only with this schema:
{
  "answers": {
    "instrument_1": "Yes|No|Uncertain",
    "instrument_2": "Yes|No|Uncertain",
    "instrument_3": "Yes|No|Uncertain",
    "instrument_4": "Yes|No|Uncertain",
    "instrument_5": "Yes|No|Uncertain",
    "instrument_6": "Yes|No|Uncertain",
    "instrument_7": "Yes|No|Uncertain",
    "instrument_8": "Yes|No|Uncertain",
    "instrument_9": "Yes|No|Uncertain",
    "instrument_10": "Yes|No|Uncertain",
    "frame_check_1": "Yes|No|Uncertain",
    "frame_check_2": "Yes|No|Uncertain",
    "frame_check_3": "Yes|No|Uncertain"
  },
  "brief_reasoning": "1-3 short sentences"
}
```


## Limitations and untried choices

- **Prompt complexity:** the segmentation prompt mixes four actors, CVS definitions, rater scores, continuity, a taxonomy, uncertainty handling, and many exception rules. The extraction prompt repeats much of this. Instruction interference and silent omission are plausible; no prompt-comprehension or rule-ablation study was run.
- **Selection bias:** most prompt/version choices were made on the same 90 clips. v4's apparently higher aggregate is therefore not clean evidence of superiority.
- **Weak actors:** held-out right-medium and camera-coarse scores are `0.308` and `0.194`; these are incompatible with a “high-accuracy ground truth” claim.
- **Sparse/imbalanced taxonomy:** rare tools, actions, contexts, camera transitions, and `other` have unstable or sometimes misleading aggregate contributions.
- **Non-independent checker:** the generator and morphology checker use the same model family and overlapping images, so agreement is not independent evidence.
- **Boundary ceiling:** frames are sampled, so exact transition boundaries between samples cannot be observed. The dataset should state this temporal resolution.
- **CVS leakage/use:** CVS rater scores are inputs. That is intentional context, but downstream claims must not imply that actions were inferred from video alone.
- **No expert reference yet:** the 30-video surgeon workflow exists but is not complete/adjudicated.
- **No cross-model label ensemble, calibrated abstention, active-learning loop, or human-in-the-loop correction was evaluated.**
- **No independent-dataset/domain-shift validation** was performed.
- **No measurement of inter-surgeon agreement or the achievable human ceiling** exists for this taxonomy.
- **Not tried:** shorter actor-specific prompts; separate left/right/camera models; constrained decoding instead of a second extraction LLM; explicit negative examples; checker coverage over all train/test clips; class-balanced sampling; confidence calibration; temporal video models; multiple random seeds/models; or blinded expert comparison of v3.1, v4, and a simpler baseline.

## What remains before a final public release

“High enough” must be a preregistered release criterion, not a subjective impression after looking at results. Recommended minimum gate:

1. Freeze the taxonomy, prompt, model revision, sampling code, and split before the final audit.
2. Fix the v3.1 checker coverage bug and regenerate every record, or rename the dataset honestly as base v3 and remove the claimed fix.
3. Obtain two independent surgeon annotations plus adjudication on a class-balanced, video-disjoint sample large enough for narrow intervals (at least 400 clips; increase until every release-critical class has at least 50 positive instances).
4. Measure segment detection and component correctness separately. Report precision, recall, F1, calibration/abstention coverage, per-class scores, video-bootstrap 95% confidence intervals, and inter-annotator agreement.
5. Require all of the following on the untouched expert test set:
   - overall adjudicated component accuracy lower 95% bound at least `0.90`;
   - left exact, right medium, and camera coarse F1 each at least `0.80`, with lower 95% bound at least `0.75`;
   - safety-critical right action/target precision and recall each at least `0.85`;
   - no common class below `0.70` F1; rare classes either meet the same bar or are explicitly marked unvalidated/removed;
   - performance no more than 5 percentage points below the measured non-adjudicating surgeon agreement ceiling;
   - downstream system-ranking correlation measured on at least 20 meaningfully different systems/configurations, with Spearman lower 95% bound at least `0.80`.
6. If those gates fail, publish only a silver-label/weak-supervision dataset with confidence, provenance, known-error fields, and a verified expert subset—never under an unqualified ground-truth claim.
7. Add dataset-card documentation for source licenses/consent, intended uses, prohibited clinical use, sampling, model/version hashes, known errors, class distributions, and a transparent correction/versioning policy.

The numeric thresholds above are **proposed**, not measured community standards. They are intentionally demanding because the requested standard is that external users should not reasonably characterize the labels as low-quality. The human-agreement ceiling may justify revising a threshold, but only prospectively and with the raw agreement evidence published.

## Archive inventory

- [`current_pipeline/`](current_pipeline/): the v3.1 derivation, full train/test generation, and final correlation notebooks.
- [`experimental_action_pipeline/`](experimental_action_pipeline/): base, CVS-context v1/v2/v3, v3.2, v4, hybrid, right-tool checker, and selection/evaluation notebooks.
- [`lvlm_cvs_reasoning/`](lvlm_cvs_reasoning/): selected earlier CVS-classification decision, metadata, ablation, and rubric notebooks.
- [`results/`](results/): compact source artifacts consumed by the reproducibility notebook.
- [`pipeline_metrics.ipynb`](pipeline_metrics.ipynb): single entry point that loads the archived results and produces every headline quantitative table.
- [`overlap_metrics.ipynb`](overlap_metrics.ipynb): loads the F1@any/F1@max summaries, directional rates, exact actor/granularity rows, and raw-artifact provenance.

The copied historical notebooks retain their original absolute paths and are evidence snapshots; some will need path updates before rerunning. The metrics notebook is the portable, read-only audit entry point.
