# CVS-Act fair evaluation protocol

This document specifies how to evaluate synthetic timestamped action labels fairly. It is a proposed protocol for the next evaluation, not a claim that the existing results already satisfy it.

The central rule is: **evaluate event detection, temporal boundaries, and action classification separately before reporting any combined score.** A system can find the right time but use the wrong label, or predict the right dominant label while missing rare actions. One aggregate F1 cannot distinguish these failures.

For the current pipeline summary, see the [concise handoff](README.md).

## 1. Decide what the dataset promises

Before collecting the final test annotations, state which claims the public dataset will make:

1. **Event presence:** whether an actor performs a meaningful action.
2. **Temporal localization:** when the action starts and ends.
3. **Action identity:** which tool, motion, direction, and anatomical target are involved.
4. **Uncertainty:** which labels should be withheld or flagged for review.

Only evaluate and advertise a claim if the annotation protocol and test set actually contain positive and negative evidence for it. For example, a `change/no_change` score containing only `change` GT rows cannot measure specificity and must not be called camera classification accuracy.

## 2. Construct an independent expert test set

- Freeze the ontology, prompt, model revision, sampling code, matching policy, and test split before the final audit.
- Use videos that were not used to develop prompts or correction rules.
- Have two surgeons annotate independently, followed by adjudication. Preserve both original annotations so human–human agreement can be measured.
- Annotate one fixed video timeline or one fixed set of candidate windows. Do not let human GT and synthetic GT create different evaluation points.
- Include two complementary samples:
  - a **natural-prevalence set** for real-world performance;
  - a **class-balanced diagnostic set** for measuring rare-class capability.
- Include explicit negative/stable windows for each actor. Without negatives, false-positive rate and specificity cannot be measured.
- Continue sampling until every release-critical class has enough support for useful confidence intervals. A starting requirement is at least 50 positive events per critical class, increased when the video-bootstrap interval remains wide.

Report the number of videos, events, negative windows, duration, and class support for every result.

## 3. Match temporal events without using their labels

For each actor and video, construct a bipartite graph between predicted and GT intervals.

1. Compute temporal intersection-over-union (IoU) for every GT–prediction pair.
2. For a selected IoU threshold, remove edges below the threshold.
3. Find a one-to-one assignment that first maximizes the number of matched pairs and then maximizes total IoU.
4. Never allow one GT event or prediction to be used more than once.
5. Perform matching **without action labels**. Labels are evaluated after matching.

Run this at IoU thresholds `0.10`, `0.25`, and `0.50`. These show sensitivity to boundary strictness; they do not represent different interval lengths.

Because the current generator observes anchors approximately every five seconds, also report a boundary-tolerance analysis grounded in that observable resolution. The tolerance must be chosen before examining test results and compared with surgeon boundary disagreement.

## 4. Report separate metric groups

### A. Event detection

Ignore the action label during matching and report:

- event precision;
- event recall;
- event F1;
- missed GT event count;
- unmatched prediction count;
- false positives per video and per minute of negative footage.

This answers whether the system found an event at all.

### B. Boundary quality

For label-blind matched pairs, report:

- F1 at IoU `0.10`, `0.25`, and `0.50`;
- median and 90th-percentile onset absolute error in seconds;
- median and 90th-percentile offset absolute error in seconds;
- fraction of onsets and offsets within the preregistered tolerance.

Do not compute boundary error only for label-correct events; doing so hides localization performance on classification mistakes.

### C. Conditional action classification

Using the same label-blind matched pairs, report:

- precision, recall, and F1 for every fixed taxonomy class;
- unweighted class-macro F1;
- exact-match rate;
- majority-class and random baselines;
- class support and a confusion matrix.

This isolates label understanding from detection. It must be labeled **conditional classification**, because unmatched predictions and missed GT events are not part of this number.

### D. End-to-end labeled segmentation

Require both sufficient IoU and a correct label. For every release-critical class, compute one-to-one segment precision, recall, and F1 at each IoU threshold, then macro-average across the fixed class set.

This is the most appropriate primary publication metric because it penalizes wrong labels, misses, extra predictions, and poor boundaries. Report the conditional metrics above alongside it so failures remain interpretable.

### E. Uncertainty and abstention

If the pipeline produces a meaningful confidence score, report:

- retained coverage;
- error rate at that coverage;
- risk–coverage curves;
- class-specific abstention rates.

A dataset that labels only 70% of events accurately is different from one that labels 100% unreliably. Do not treat the current placeholder confidence values as calibrated probabilities.

## 5. Use actor-meaningful targets

### Left grasper

Evaluate separately:

- start direction;
- end direction;
- maintain versus change;
- exact combined transition code.

This prevents a dominant endpoint such as `lateral` from hiding failure to recognize the transition itself.

### Right working tool

Evaluate separately:

- tool;
- action;
- anatomical target;
- each context field;
- exact joint label.

The joint label remains important, but field-level scores explain whether an error comes from morphology, action recognition, or anatomy. Report safety-critical target precision and recall explicitly.

### Camera

Use two stages:

1. Evaluate camera-event presence on both positive and stable/negative windows, including false positives per minute.
2. For matched true events, evaluate zoom versus reposition, motion direction such as zoom-in versus zoom-out, and temporal boundaries.

Do not use the current conditional camera-coarse score as the headline metric. In the held-out six-system correlation calculation, all evaluated camera-coarse GT rows are `change`: 66/66 for human GT and 25/25 for synthetic GT. It therefore contains no reference examples with which to measure `no_change` specificity.

## 6. Handle class imbalance explicitly

- Predeclare the full evaluated class list.
- Use class-macro rather than micro or prevalence-weighted F1 as the primary classification summary.
- Never silently remove a difficult class because it is rare or receives no predictions.
- Show class support and per-class confidence intervals.
- Always show a constant-majority baseline.
- Report natural-prevalence and balanced-diagnostic results separately rather than mixing their sampling weights.
- Treat classes with inadequate support as unvalidated; collect more data, merge the ontology prospectively, or exclude them from the public claim.

## 7. Use confusion matrices only for the question they answer

The archive's max-overlap and all-overlap matrices are useful diagnostics, but neither is a standalone accuracy metric:

- max-overlap selects one dominant predicted label per GT interval and can hide secondary fragments;
- all-overlap counts every overlapping interval pair and can count one GT interval several times.

For the final evaluation, add a **one-to-one matched confusion matrix**:

- rows are GT classes plus a `<spurious_prediction>` row;
- columns are predicted classes plus a `<missed_gt>` column;
- matched pairs contribute one cell;
- unmatched GT events contribute to `<missed_gt>`;
- unmatched predictions contribute to `<spurious_prediction>`.

Keep max-overlap and all-overlap plots as supplementary error-analysis views.

## 8. Quantify uncertainty at the video level

Bootstrap complete videos, not individual intervals, because events from the same surgery are correlated. For every headline metric report:

- point estimate;
- 95% video-bootstrap confidence interval;
- number of videos;
- number of events and negative windows;
- per-class support.

Use paired video bootstrap when comparing two pipeline versions.

## 9. Compare with the human ceiling

Run the identical matcher and metrics between the two independent surgeon annotations before adjudication. This establishes:

- achievable boundary agreement;
- ambiguous or unstable classes;
- whether the ontology itself is reproducible;
- a defensible performance target for automation.

Numeric release gates should be chosen after measuring this ceiling. A proposed starting gate is that the pipeline is within five percentage points of non-adjudicating surgeon agreement, every preregistered critical macro metric has a lower 95% confidence bound of at least `0.75`, and no common critical class is below `0.70` F1. These are proposed safeguards, not community standards.

## 10. Recommended release dashboard

For every actor, publish:

1. label-blind event F1@IoU `0.25`;
2. end-to-end class-macro segment F1@IoU `0.25`;
3. end-to-end class-macro segment F1@IoU `0.50`;
4. conditional class-macro F1 on matched events;
5. median onset and offset error;
6. false positives per minute;
7. worst common-class F1;
8. surgeon–surgeon results using the same metrics;
9. video-bootstrap 95% intervals.

The full dashboard, rather than one scalar, determines readiness. Each actor and every safety-critical class must pass its own preregistered gate.

If a single summary is required for a table, compute each actor's macro-average of its release-critical per-class segment F1@IoU `0.25`, then average the three actor scores equally. Never allow this mean to override a failing actor or critical class.

## 11. Role of downstream system correlation

System-ranking correlation is a secondary utility experiment, not annotation validation. A future correlation study should use:

- the identical videos and event support under human and synthetic GT;
- at least 20 meaningfully different systems/configurations;
- actor-specific correlations;
- leave-one-actor-out correlations;
- Pearson and Spearman estimates with video-bootstrap intervals.

The current six-system correlation should not be used as a publication gate: its aggregate Pearson correlation is `0.947`, but camera-coarse alone is `0.963`, and removing camera lowers the aggregate correlation to `0.451`.

## 12. Implementation checklist

- [ ] Freeze intended claims, taxonomy, split, and matching rules.
- [ ] Collect positive and negative expert annotations on identical fixed timelines.
- [ ] Preserve two independent annotations and an adjudicated reference.
- [ ] Implement label-blind one-to-one matching.
- [ ] Implement detection, boundary, conditional-label, and end-to-end metrics separately.
- [ ] Add matched confusion matrices with missed and spurious categories.
- [ ] Report class support, majority baselines, and video-bootstrap intervals.
- [ ] Establish actor- and class-specific gates from human agreement.
- [ ] Keep downstream correlation secondary.
- [ ] Publish the evaluator, frozen test manifest, and metric definitions with the dataset.
