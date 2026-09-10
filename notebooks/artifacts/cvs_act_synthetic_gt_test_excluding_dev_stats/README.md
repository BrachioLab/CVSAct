# CVS-Act synthetic-GT scores excluding dev10

- Split seed: `20260610`
- Dev videos excluded from score tables: `10`
- Held-out human-GT test videos requested: `20`
- Held-out human-GT test videos scored in small table: `20`-`20`
- Larger plain-baseline videos before dev exclusion: `65`
- Larger plain-baseline videos after dev exclusion: `55`
- Pooled larger-minus-dev unique videos scored: `55`

## Outputs
- `small_scores_csv`: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_synthetic_gt_test_excluding_dev_stats/synthetic_gt_baseline_vs_surgent_test20_excluding_dev10_final_scores.csv`
- `small_inventory_csv`: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_synthetic_gt_test_excluding_dev_stats/synthetic_gt_baseline_vs_surgent_test20_excluding_dev10_inventory.csv`
- `pooled_scores_csv`: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_synthetic_gt_test_excluding_dev_stats/synthetic_gt_baseline_only_larger_plus_test20_excluding_dev10_final_scores.csv`
- `pooled_inventory_csv`: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_synthetic_gt_test_excluding_dev_stats/synthetic_gt_baseline_only_larger_plus_test20_excluding_dev10_inventory.csv`
- `majority_csv`: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_synthetic_gt_test_excluding_dev_stats/synthetic_gt_majority_baseline_excluding_dev10_final_scores.csv`
- `clip_stats_csv`: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_synthetic_gt_test_excluding_dev_stats/synthetic_gt_split_clip_counts_excluding_dev10.csv`
- `small_scores_tex`: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_synthetic_gt_test_excluding_dev_stats/synthetic_gt_baseline_vs_surgent_test20_excluding_dev10_final_scores.tex`
- `pooled_scores_tex`: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_synthetic_gt_test_excluding_dev_stats/synthetic_gt_baseline_only_larger_plus_test20_excluding_dev10_final_scores.tex`
- `clip_stats_tex`: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_synthetic_gt_test_excluding_dev_stats/synthetic_gt_split_clip_counts_excluding_dev10.tex`
- `small_scores_png`: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_synthetic_gt_test_excluding_dev_stats/synthetic_gt_baseline_vs_surgent_test20_excluding_dev10_final_scores.png`
- `small_scores_pdf`: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_synthetic_gt_test_excluding_dev_stats/synthetic_gt_baseline_vs_surgent_test20_excluding_dev10_final_scores.pdf`
- `pooled_scores_png`: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_synthetic_gt_test_excluding_dev_stats/synthetic_gt_baseline_only_larger_plus_test20_excluding_dev10_final_scores.png`
- `pooled_scores_pdf`: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_synthetic_gt_test_excluding_dev_stats/synthetic_gt_baseline_only_larger_plus_test20_excluding_dev10_final_scores.pdf`
- `pooled_scores_csv_alias`: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_synthetic_gt_test_excluding_dev_stats/synthetic_gt_baseline_only_larger_excluding_dev10_final_scores.csv`
- `pooled_inventory_csv_alias`: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_synthetic_gt_test_excluding_dev_stats/synthetic_gt_baseline_only_larger_excluding_dev10_inventory.csv`
- `pooled_scores_tex_alias`: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_synthetic_gt_test_excluding_dev_stats/synthetic_gt_baseline_only_larger_excluding_dev10_final_scores.tex`
- `pooled_scores_png_alias`: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_synthetic_gt_test_excluding_dev_stats/synthetic_gt_baseline_only_larger_excluding_dev10_final_scores.png`
- `pooled_scores_pdf_alias`: `/mnt/md0/weiqiuy/surgent/notebooks/artifacts/cvs_act_synthetic_gt_test_excluding_dev_stats/synthetic_gt_baseline_only_larger_excluding_dev10_final_scores.pdf`

## Naming note

The `larger_plus_test20` pooled filenames are retained for compatibility with earlier SCP commands, but their contents now use the corrected de-duplicated pool: the 65-video larger run after removing the 10 development videos (`N=55`). The held-out 20 videos are not appended a second time. Equivalent clearer aliases are also available with `larger_excluding_dev10` in the filename.
