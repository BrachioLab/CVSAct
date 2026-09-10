# Recomputed human-vs-synthetic GT correlation

This notebook recomputes onset-pointwise evaluator detail rows from raw GT JSONL and prediction files before making the scatter plot.

- Test videos in seeded split: `20`
- Human GT records on test split: `60`
- v3.1 synthetic GT records on test split: `60`
- Human detail videos after recompute: `20`
- Synthetic detail videos after recompute: `20`
- Pearson r: `0.947250 ± 0.189873`
- Spearman r: `0.885714 ± 0.220545`
- Bootstrap examples: `60`
- Bootstrap resamples: `1000`

## Outputs
- human details: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_human_synthetic_correlation_recomputed_details/recomputed_human_gt_onset_pointwise_details_test20.csv`
- synthetic details: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_human_synthetic_correlation_recomputed_details/recomputed_v31_synthetic_gt_onset_pointwise_details_test20.csv`
- final scores: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_human_synthetic_correlation_recomputed_details/recomputed_six_system_human_synthetic_final_scores_test20.csv`
- scatter points: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_human_synthetic_correlation_recomputed_details/recomputed_six_system_human_synthetic_final_score_points_test20.csv`
- correlation CSV: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_human_synthetic_correlation_recomputed_details/recomputed_six_system_human_synthetic_final_score_correlation_test20.csv`
- bootstrap correlation CSV: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_human_synthetic_correlation_recomputed_details/recomputed_six_system_human_synthetic_correlation_example_bootstrap_test20.csv`
- clip scores for bootstrap CSV: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_human_synthetic_correlation_recomputed_details/recomputed_clip_final_scores_for_correlation_bootstrap_test20.csv`
- correlation TeX: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_human_synthetic_correlation_recomputed_details/recomputed_test20_six_system_human_synthetic_correlation.tex`
- scatter PNG: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_human_synthetic_correlation_recomputed_details/recomputed_test20_six_system_human_synthetic_final_score_scatter.png`
- scatter PDF: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_human_synthetic_correlation_recomputed_details/recomputed_test20_six_system_human_synthetic_final_score_scatter.pdf`
- scatter wide PNG: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_human_synthetic_correlation_recomputed_details/recomputed_test20_six_system_human_synthetic_final_score_scatter_wide.png`
- scatter wide PDF: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_human_synthetic_correlation_recomputed_details/recomputed_test20_six_system_human_synthetic_final_score_scatter_wide.pdf`
