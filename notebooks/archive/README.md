# CVS-Act synthetic-label pipeline: quick handoff

This page is the starting point for a student who is new to the project. It explains what the current pipeline predicts, what the evaluation says is working, what is failing, and what should happen next.

**CVS** means the *Critical View of Safety*, a safety view surgeons establish during laparoscopic gallbladder removal. The source clips are centered around changes in the three CVS criteria, but this pipeline's output is more detailed: timestamped grasper, working-tool, and camera actions that may explain how the view changed.

## Current pipeline: v3.1

The selected pipeline is **v3.1 structured extraction**. It converts a short laparoscopic video clip around a CVS transition into timestamped action segments for three actors:

![CVS-Act v3.1 component diagram emphasizing the three main modules: the multimodal first call, text-only second call, and right-tool morphology branch](pipeline_overview_core_components.svg)

The diagram shows the actual information representations between components: sampled images and reviewer scores enter the first call; its free-text temporal segmentation becomes synchronized left/right/camera lanes; the second call maps those descriptions to controlled JSON; deterministic code then validates, optionally corrects the right-tool identity, and writes the dataset splits.

- **Left:** how the left grasper retracts the gallbladder.
- **Right:** what the active right-hand tool is doing and to which anatomy.
- **Camera:** whether the camera zooms, repositions, or stays unchanged.

### How one clip is processed

1. **Construct the clip.** Start with one clip around a transition in CVS criterion C1, C2, or C3.
2. **Sample ordered frames.** Use anchor frames every 150 frames, or about 5 seconds at 30 fps; add four evenly spaced frames between anchors and include the endpoint.
3. **Attach CVS context.** Each anchor is labeled with `C1=k/3, C2=k/3, C3=k/3`, where `k` is the number of three reviewers who judged that criterion satisfied. In-between frames contain only their frame number.
4. **Segment with the multimodal model.** `Qwen/Qwen3.6-35B-A3B-FP8`, at temperature 0 with thinking disabled, receives the ordered images, CVS context, and segmentation prompt. It describes continuous left, right, camera, and visibility behavior over timestamped ranges.
5. **Extract the controlled taxonomy.** A second, text-only LLM call receives the complete segmentation response and maps it into the fixed left/right/camera/other JSON schema. It does not see images or human GT.
6. **Validate and normalize.** Invalid labels are dropped, timestamps are clipped to the clip range, and adjacent identical segments are merged.
7. **Apply the optional v3.1 right-tool correction.** A separate morphology questionnaire examines the clip. The deterministic rule changes `Hook` to `Maryland` only when the base prediction is exactly `Hook` and the checker predicts `Maryland`; no other disagreement is changed.
8. **Export by video.** Only structured-extraction records are retained. The 1,312 training records are split by video into 1,119 train and 193 validation records; 481 test records are kept separately.

The [full v3.1 prompt stack](README_COMPREHENSIVE.md#exact-prompts) contains the complete first-call segmentation prompt, second-call structured extraction prompt, and additional morphology-checker prompt. Importantly, v3.1 does not change the base v3 generation prompts; it adds the checker and narrow deterministic post-processing rule.

> [!IMPORTANT]
> **The evaluated v3.1 run and the full exports did not receive the same effective correction.** On the 90-clip development artifact, the wrapper changed 10 of 90 structured predictions. In the full train and test exports it changed zero labels because checker coverage or the exact `Hook`-to-`Maryland` condition was missing. The wrapper is present in the export code, but the exported labels equal their immediately preceding base-v3 predictions apart from v3.1 provenance metadata.

## Bottom line on current quality

> [!WARNING]
> **Current release status: do not publish these labels as trusted ground truth.**
>
> The apparently high left-medium overlap score is mostly caused by repeatedly predicting the common `lateral` class, not by recognizing all left-action classes. Right-medium contains some useful class signal, but it is weak; camera and the remaining granularities are also weak. A preview release clearly marked as **unverified synthetic annotation** would be more defensible.

The [fair evaluation protocol](EVALUATION_PROTOCOL.md) is a **potential future direction for replacing the current evaluation**, whose overlap averages, conditional support, and system-ranking correlation can hide class collapse or degenerate tasks. It has not yet been implemented or validated. The next study should freeze that protocol and an independent expert test set before changing the prompt again.

The results below come from the same **90 development clips** used while developing the pipeline. They are useful for diagnosis, but they are not an independent final test.

## Terms to know

- A **segment** is a frame interval such as frames 300–450 with one action label.
- **GT** means the human-authored ground-truth interval and label.
- A **prediction** is the synthetic interval and label generated by the pipeline.
- An **actor** is left grasper, right working tool, or camera.
- **Raw** is also called **exact** in the code. It retains the most label detail.

### What raw, medium, and coarse mean

| Actor | Raw / exact | Medium | Coarse |
|---|---|---|---|
| Left | Full retraction code, e.g. `RETRACT_MEDIAL_TO_LATERAL` | Final direction: lateral, medial, or upward | Whether retraction is maintained or changed |
| Right | Tool + action + target + anatomical context | Tool + action + target | Action + target |
| Camera | Exact code such as zoom-in, zoom-out, reposition | Zoom, reposition, or no-change | Change or no-change |

Important: left-medium and left-coarse answer different questions. Medium asks **which direction**; coarse asks **maintain or change**. Coarse is therefore not guaranteed to be easier.

## Current status by actor and granularity

Two different questions must be kept separate:

- **F1@max** asks whether the dominant overlapping label is correct and then averages by video. It is useful for temporal overlap, but it does **not** weight action classes equally.
- **Class-macro F1** computes F1 separately for every label and averages the labels. This is closer to the metric used in the paper and reveals majority-class collapse. Here it is computed from the pooled, GT-anchored F1@max assignments, so it is still a diagnostic classification score rather than a complete segment-detection metric.

| Actor | Level | F1@max | Class-macro F1 | Majority baseline | Interpretation |
|---|---|---:|---:|---:|---|
| Left | Raw | 0.539 | 0.091 | 0.071 | Very weak; mostly `KEEP_RETRACT_LATERAL` |
| Left | Medium | **0.859** | **0.287** | **0.305** | Misleading overlap score; worse than always predicting `lateral` |
| Left | Coarse | 0.577 | **0.228** | **0.346** | Worse than always predicting `maintain` |
| Right | Raw | 0.094 | 0.033 | 0.004 | Very weak exact tool/action/target/context prediction |
| Right | Medium | 0.368 | **0.138** | 0.021 | Weak, but contains real class signal above baseline |
| Right | Coarse | 0.419 | 0.115 | 0.039 | Weak; target confusion remains |
| Camera | Raw | 0.146 | 0.133 | 0.108 | Very weak |
| Camera | Medium | 0.168 | **0.153** | **0.222** | Worse than the majority baseline |
| Camera | Coarse | 0.271 | **0.272** | **0.456** | Worse than always predicting `change` |

The majority baseline always predicts the most frequent GT class at that actor/granularity. The class-macro values are pooled across 83 left, 123 right, or 68 camera GT intervals from the 90 development clips. `<no_overlap>` is retained as an incorrect predicted outcome. Right-raw is exact-string scoring here; the historical raw evaluator instead allows partial unordered context matching. Neither table is an independent held-out test.

### What happens as interval matching becomes stricter

The historical metric uses one-to-one segment matching at intersection-over-union (IoU) thresholds 0.10, 0.25, and 0.50. A larger threshold requires predicted boundaries to align more closely with GT. These are sometimes called “different interval sizes,” but the interval length does not change; the required overlap fraction does. `@any` and `@max` are separate, non-one-to-one diagnostics and are not additional IoU thresholds.

| Actor | Level | F1@IoU=.10 | F1@IoU=.25 | F1@IoU=.50 | F1@any | F1@max |
|---|---|---:|---:|---:|---:|---:|
| Left | Raw | 0.533 | 0.523 | 0.518 | 0.565 | 0.539 |
| Left | Medium | 0.764 | 0.644 | 0.611 | 0.867 | 0.859 |
| Left | Coarse | 0.574 | 0.564 | 0.559 | 0.598 | 0.577 |
| Right | Raw | 0.056 | 0.056 | 0.056 | 0.096 | 0.094 |
| Right | Medium | 0.345 | 0.312 | 0.254 | 0.411 | 0.368 |
| Right | Coarse | 0.386 | 0.342 | 0.284 | 0.462 | 0.419 |
| Camera | Raw | 0.162 | 0.119 | 0.068 | 0.178 | 0.146 |
| Camera | Medium | 0.190 | 0.150 | 0.088 | 0.214 | 0.168 |
| Camera | Coarse | 0.321 | 0.256 | 0.158 | 0.367 | 0.271 |

The decline from IoU 0.10 to 0.50 exposes boundary errors, especially for camera and right-medium/coarse. Right-raw stays at 0.056 because label identity, rather than boundary strictness, is already the dominant failure. Left-medium remains high under every overlap metric because the dominant `lateral` class is common; its low class-macro F1 above is the necessary counter-check. Do not rank `@max` against an IoU column as if one were simply stricter: `@max` uses dominant accumulated label overlap without one-to-one assignment, while IoU F1 greedily matches whole intervals one-to-one.

## Three examples that explain the scores

### 1. Why left-medium can be correct while left-raw and left-coarse are wrong

```text
GT:         RETRACT_MEDIAL_TO_LATERAL
Prediction: KEEP_RETRACT_LATERAL

Raw:    wrong — changing direction is not the same as maintaining it
Medium: correct — both finish in the lateral direction
Coarse: wrong — GT is change, prediction is maintain
```

This pattern explains the high left-medium **overlap** score and much lower left-coarse score. It does not establish useful left-medium classification: 79 of 83 left intervals receive the medium label `lateral`, and class-macro F1 is below the constant-`lateral` baseline.

### 2. Typical right-hand target confusion

```text
GT:         Hook | DISSECT | CysticArtery
Prediction: Hook | DISSECT | CysticDuct
```

The tool and action are correct, but the anatomical target is wrong. This exact artery-to-duct confusion occurs eight times in the medium confusion matrix. Another eight intervals change `CysticDuct` to `HepatocysticTriangle`.

### 3. Typical camera failure

```text
GT:         CAMERA_ZOOM_OUT
Prediction: <no_overlap>
```

This is a timing/detection miss, not merely the wrong camera class. At camera-coarse level, the 57 GT `change` intervals break down as:

- 15 predicted `change`;
- 20 predicted `no_change`;
- 22 with no overlapping camera prediction.

## Major confusion patterns

### Left

The model defaults heavily to `KEEP_RETRACT_LATERAL`. It recognizes the eventual lateral direction better than it recognizes that a direction change occurred.

[View left raw confusion plot](results/current/confusion_matrices/left_raw_row_normalized.png)

[View left raw all-overlap plot](results/current/confusion_matrices/left_raw_all_overlaps_row_normalized.png)

### Right

The model often recognizes “Hook dissection” but chooses the wrong nearby anatomy. Fine-grained anatomical context makes raw accuracy especially low. Removing context helps, but target confusion remains.

[View right medium confusion plot](results/current/confusion_matrices/right_medium_row_normalized.png)

[View right medium all-overlap plot](results/current/confusion_matrices/right_medium_all_overlaps_row_normalized.png)

### Camera

The problem is both classification and detection: many camera changes are called no-change, and many have no predicted interval at all.

[View camera coarse confusion plot](results/current/confusion_matrices/camera_coarse_row_normalized.png)

[View camera coarse all-overlap plot](results/current/confusion_matrices/camera_coarse_all_overlaps_row_normalized.png)

The original confusion plots assign one max-overlap label to each GT interval. The new **all-overlap** plots count every GT–prediction interval pair with positive overlap; a fragmented prediction can therefore contribute multiple counts to one GT row. If nothing overlaps, `<no_overlap>` is counted once. Row normalization in this view is over overlap-pair counts, not over unique GT intervals, so use it to reveal secondary labels and fragmentation rather than as an accuracy score.

The all-overlap view does not reverse the diagnosis. For left-medium GT, medial intervals overlap 12 `lateral` predictions but only 2 `medial` predictions, while upward intervals overlap 7 `lateral` predictions and only 1 `upward` prediction. The 123 right GT intervals produce 263 overlap pairs, exposing substantial segmentation/label fragmentation. For camera-coarse GT `change`, there are 38 overlaps with predicted `change`, 39 with predicted `no_change`, and 22 intervals with no overlap; one GT interval can appear in both overlap columns.

## What worked in previous experiments—and what did not

These are development-set comparisons, not independent confirmations. Several versions changed more than one prompt/input component, so only v3→v3.1 is close to a controlled post-processing comparison.

| Experiment | Result | Verdict | Evidence |
|---|---|---|---|
| Structured LLM taxonomy extraction | It was the best of the four output strategies in every tested prompt family. | **Worked consistently; retain it.** | [Base](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval.ipynb), [CVS v1](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_with_cvs_context.ipynb), [v2](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v2.ipynb), [v3](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v3.ipynb), [v4](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v4.ipynb) |
| Add CVS context and reviewer scores to frames | Structured score rose from 0.195 to 0.302. Text context and scored frame labels changed together. | **Worked as a bundle; individual contribution is unknown.** | [Base](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval.ipynb), [CVS v1](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_with_cvs_context.ipynb) |
| v2 left stable-versus-transition rules | Aggregate structured score rose slightly, 0.302→0.326. Current confusions nevertheless show severe maintain/change collapse. | **Small aggregate gain; did not solve the left problem.** | [v2](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v2.ipynb) |
| v3 detailed right extraction rules | Aggregate score changed 0.326→0.323; right overlap diagnostics improved, but exact and medium class-macro F1 remain low. | **Mixed; useful direction, not a solution.** | [v3](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v3.ipynb) |
| v3.1 clip-level Hook→Maryland checker | On the 90-clip artifact, changed 10 structured predictions and raised right F1 0.211→0.231 and aggregate F1 0.323→0.330. | **Worked narrowly on development data.** Full train/test exports applied zero corrections because checker coverage/conditions were insufficient. | [v3.1](current_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v3_1.ipynb) |
| v3.2 interval-level checker | Made more calls and 70 rewrites, but aggregate score was 0.329 versus 0.330 for v3.1. | **Did not justify its extra complexity.** | [v3.2](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v3_2.ipynb) |
| v4 explicit camera start/end questions | Aggregate score reached 0.340, but camera changed only 0.168→0.171 and right regressed to 0.168. | **Did not demonstrate a reliable camera gain; joint prompting affected other actors.** | [v4](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v4.ipynb) |
| v3.1 left/right plus v4 camera | Outputs were spliced, but no independent metric was produced. | **Unmeasured, not evidence of improvement.** | [Hybrid](experimental_action_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_hybrid_v3_1_left_right_v4_camera.ipynb) |

Older CVS-classification experiments also found that sequential frame context improved mean AP from 0.304 to 0.395, while the tested sequential organ overlay reduced it to 0.316. Observe-then-decide tied sequential direct prompting in mean AP rather than improving it. These five-video experiments support retaining temporal context, but they do not measure action-generation accuracy: [initial ablations](lvlm_cvs_reasoning/visualize_results_initial_ablations.ipynb), [organ ablations](lvlm_cvs_reasoning/visualize_results_organ_ablations.ipynb), and [all-model summary](lvlm_cvs_reasoning/eval_all_models_summary_new.ipynb).

## Best next step for each actor

### Left

Treat **direction** and **state change** as separate predictions. First infer start direction and end direction, then derive maintain/change instead of asking one label to encode both. Add balanced examples and hard negatives for medial, upward, and direction-change classes; reject or flag `KEEP_*` when the inferred start and end states differ. Evaluate on a class-balanced held-out set, because another prompt tuned on the same 90 clips will not establish improvement.

### Right

Keep the structured extraction and conservative morphology correction, but decompose the task into tool, action, target, and context fields before composing the final label. Focus annotation/prompt examples on the observed artery↔duct and duct↔hepatocystic-triangle confusions. Test anatomy crops, tracking/segmentation cues, and targeted hard-negative pairs. Expand checker coverage before regeneration, then accept a correction only when it improves held-out per-class F1 rather than aggregate accuracy alone.

### Camera

Use a separate camera pass instead of expanding the joint left/right/camera prompt again. Estimate start/end zoom and viewpoint states and derive the motion; add explicit global-motion/zoom cues so tissue or instrument motion is not mistaken for camera motion. Measure camera-event coverage and false positives separately from label correctness. The next experiment should compare this isolated camera pass with v3.1 on frozen held-out clips; the existing v4 and unscored hybrid do not answer that question.

### Evaluation shared by all three

Do not choose one convenient aggregate and call it “accuracy.” First decide whether the intended dataset claim is event detection, boundary localization, action classification, or all three; then evaluate each claim separately on the same frozen expert annotations.

| Question | Meaningful primary measurement |
|---|---|
| Did the system find an event? | Label-blind one-to-one event precision/recall/F1, including negative clips and extra predictions |
| Are its boundaries useful? | F1 at IoU 0.10/0.25/0.50 plus onset and offset absolute error in seconds and fraction within a preregistered tolerance |
| Is the action label correct once an event is matched? | Fixed-taxonomy per-class precision/recall/F1 and unweighted class-macro F1, always beside class support and a majority baseline |
| Is the entire segment prediction correct? | Per-class one-to-one segment F1 or AP at the IoU thresholds, macro-averaged across release-critical classes |
| Can uncertainty reduce errors? | Risk versus retained coverage for confidence-based abstention |
| Is performance close to what humans can achieve? | Independent surgeon–surgeon agreement using the identical matching/scoring protocol |

For left, score start direction, end direction, and maintain/change separately before the combined raw code. For right, score tool, action, target, and context separately as well as their exact joint label. For camera, do not use coarse `change/no_change` as the headline label metric. In the held-out correlation calculation, every evaluated camera-coarse GT row is `change`—66/66 for human GT and 25/25 for synthetic GT—with no `no_change` reference rows. The score therefore measures whether a system emits `change` at selected positive onsets; it cannot measure specificity or false positives on stable camera periods. Evaluate event presence on positive **and negative** windows, then evaluate zoom/reposition direction and boundaries on matched events.

Downstream system-ranking correlation is only a secondary utility check. The current `0.947` headline uses six systems and is largely camera-driven: camera-coarse Pearson correlation is `0.963`, while removing camera reduces final-score Pearson correlation to `0.451`. Human and synthetic GT also evaluate different numbers of actor events. Correlation therefore cannot validate annotation fidelity; future reporting should use at least 20 diverse systems, identical event support, actor-specific and leave-one-actor-out results, and confidence intervals.

## What students should do next

1. **Do not call the current exports validated ground truth.** The evaluated v3.1 90-clip artifact received morphology corrections, but the full train/test exports received zero such corrections because checker coverage was missing.
2. **Freeze a meaningful evaluation protocol before more prompt tuning:** separate detection, timing, and classification, and score every prediction against the same expert reference universe.
3. **Focus model/prompt work on the actor-specific experiments above:** left start/end state decomposition, right anatomy-aware field prediction, and a separate camera-change pass.
4. **Run a blinded, video-held-out expert audit** with enough examples per common and rare class plus representative negative windows. Do not tune prompts on that final audit set.
5. **Report uncertainty and human agreement:** video-bootstrap confidence intervals, per-class support, precision/recall, and surgeon–surgeon agreement, not only one averaged F1.
6. **Regenerate the full exports** after the pipeline, ontology, and checker are frozen, then rerun integrity checks before any Hugging Face upload.

Do not retain the earlier blanket target of `0.80 F1` for left-raw/right-medium/camera-coarse without redesigning the evaluation first: those three scores mix different tasks, and camera-coarse is not sufficiently informative. Final numeric release gates should be preregistered per task and class after measuring surgeon agreement; no critical class should be hidden by an aggregate.

## Where to look next

- [Fair evaluation protocol](EVALUATION_PROTOCOL.md): potential future replacement for the current evaluation; it is a proposed frozen-test, one-to-one matching, actor-specific metric, human-agreement, and release-gate procedure, not a completed result.
- [Full v3.1 prompt stack](README_COMPREHENSIVE.md#exact-prompts): complete segmentation prompt, structured LLM extraction prompt, and additional morphology-checker prompt.
- [Comprehensive technical report](README_COMPREHENSIVE.md): full pipeline history, exact prompts, comparability audit, every result table, limitations, and release gates.
- [Overlap-metrics notebook](overlap_metrics.ipynb): F1@any/F1@max tables and directional rates.
- [Confusion-matrix notebook](confusion_matrices.ipynb): interactive plots for all nine actor/granularity combinations.
- [All static confusion plots and CSVs](results/current/confusion_matrices/): counts, row-normalized matrices, and interval assignments.
- [Current v3.1 generation notebook](current_pipeline/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v3_1.ipynb): evaluated pipeline implementation.
- [Human-versus-synthetic system scores](results/current/recomputed_six_system_human_synthetic_final_scores_test20.csv): source for the component-level and leave-one-actor-out correlation diagnosis.
- [Recompute overlap metrics](../../scripts/evaluate_overlap_metrics.py) and [recompute confusion matrices](../../scripts/evaluate_overlap_confusions.py).

## One-sentence handoff

**v3.1's high left-medium overlap score is majority-class collapse, right-medium has weak but real signal, and no actor is yet accurate enough across classes and time to publish as trusted ground truth.**
