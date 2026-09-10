# CVS-Act

CVS-Act contains the annotation workflows, synthetic-label pipeline, dataset export, model/agent generation scripts, and evaluation code for CVS-grounded surgical action recommendation.

## Layout

- `src/cvs_act/`: core CVS-Act dataset, annotation, conversion, and evaluation utilities.
- `src/surgent/`: SurGent agent implementation used as one evaluated system.
- `scripts/`: command-line entry points.
- `notebooks/`: annotation, synthetic generation, export, and final analysis notebooks.
- `data/processed/CVS_Challenge_SAGES_v1/`: processed labels, splits, synthetic labels, and surgeon-validation assets.
- `hf_repos/cvs-act/`: local Hugging Face dataset export.
- `outputs/`: only the final prediction JSONL files used by the human-vs-synthetic correlation analysis.

## Setup

With `venv` and `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

With conda:

```bash
conda create -n cvsact python=3.11
conda activate cvsact
pip install -e ".[dev]"
```

Basic checks:

```bash
PYTHONPATH=src pytest tests/test_cvs_act_evaluation.py -q
python scripts/export_hf_cvs_act.py --validate
```

Open JupyterLab from the repo root:

```bash
jupyter lab --no-browser --port=8999
```

## Human Annotation

The final trained-annotator labels are `audit_v11`.

Labeling notebook:

- `notebooks/audit_timestamped_tool_interface.ipynb`

End-result / conversion notebook:

- `notebooks/audit_v11_simple_action_gt_conversion.ipynb`

Final artifacts:

- `data/processed/CVS_Challenge_SAGES_v1/cvs_act_annotations/v1/audit_v11/`
- `notebooks/artifacts/audit_v11_simple_action_gt/`

Human split:

- `notebooks/artifacts/qwen3.5_audit_v11_current_best_prompt_summary/validation_test_video_split_seed20260610.csv`
- dev/validation: 10 videos
- test: 20 videos

## Expert / Surgeon Annotation

Expert annotation uses a separate 30-video surgeon-validation sample.

Sampling notebook:

- `notebooks/surgeon_validation/stratified_sample_30.ipynb`

Expert labeling notebook:

- `notebooks/surgeon_validation/annotate_audit_v11_video_menu.ipynb`

Final artifacts:

- `data/processed/CVS_Challenge_SAGES_v1/cvs_act_surgeon_annotations/`

Key files:

- `sample_30.csv`
- `selected_video_clips.csv`
- `clip_manifest.csv`
- `annotations/cvs_act_action_annotations.jsonl`
- `aws_upload_assets.zip` is a local-only upload bundle and is ignored unless Git LFS is configured.

## Synthetic Labels

The final synthetic labels use the v3.1 CVS-context pipeline.

Notebook:

- `notebooks/qwen3.5_audit_v11_seg_extract_spec_eval/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v3_1.ipynb`

Final artifacts:

- `data/processed/CVS_Challenge_SAGES_v1/cvs_act_synthetic_data/audit_v11_cvsctx_v3_1/`

## Hugging Face Export

Script:

- `scripts/export_hf_cvs_act.py`

Run:

```bash
python scripts/export_hf_cvs_act.py --validate
```

Final dataset:

- `hf_repos/cvs-act/`

## System Generation

The final correlation analysis uses only baseline default and SurGent default predictions for the 20-video human test split.

Baseline generation:

```bash
bash scripts/run/run_cot_cvs_act_v1_eval.sh --all --models "gpt-5.4-mini gemini-2.5-flash claude-haiku-4-5-20251001"
```

SurGent generation:

```bash
bash scripts/run/run_surgent_cvs_act_v1_eval.sh --all --models "gpt-5.4-mini gemini-2.5-flash claude-haiku-4-5-20251001"
```

Core runners:

- `scripts/run/baseline.py`
- `scripts/run/run_surgent_sequential.py`

Copied prediction artifacts:

- `outputs/cot_audit_v11_simple/cot_fixedk3_norecdescs_fmeta/`
- `outputs/surgent_sequential_agent_audit_v11_simple/cot/pref-cvs_arec_steps5_fixedk3_norecdescs_fmeta/`

Only `__action_taxonomy.jsonl` files for the three final models and 20 test videos are included.

## Evaluation

Primary scripts:

- `scripts/eval/evaluate_cvs_act_v1_combined.py`
- `scripts/eval/evaluate_cvs_act_v1_onset_pointwise.py`

Final human-vs-synthetic correlation notebook:

- `notebooks/qwen3.5_audit_v11_seg_extract_spec_eval/cvs_act_human_synthetic_correlation_recomputed_details.ipynb`

Final artifacts:

- `notebooks/artifacts/cvs_act_human_synthetic_correlation_recomputed_details/`

Final 20-video test correlation:

- Pearson: `0.947`
- Spearman: `0.886`
