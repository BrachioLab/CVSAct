#!/usr/bin/env bash
set -euo pipefail

# Public CVS-Act v1 CoT runner.
# Uses the HF-packaged trained-annotator split to determine the evaluation set,
# while still reading frames locally.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

FRAMES_DIR="data/processed/CVS_Challenge_SAGES_v1/frames/test"
HF_LABELS_JSONL="hf_repos/cvs-act/sages_trained_annotator/test.jsonl"
SYNTHETIC_LABELS_JSONL="hf_repos/cvs-act/sages_synthetic/test.jsonl"
AUDIT_V11_DIR="data/processed/CVS_Challenge_SAGES_v1/cvs_act_annotations/v1/audit_v11"
TAXONOMY_JSON="hf_repos/cvs-act/taxonomy/action_taxonomy.json"
TAXONOMY_TAG="$(basename "${TAXONOMY_JSON}" .json)"
OUT_ROOT="outputs/cot_audit_v11_simple"
LOG_DIR="outputs/logs"

PRESET="cot"
MODELS="gpt-4.1-mini gemini-2.5-flash claude-haiku-4-5-20251001 Qwen/Qwen3.6-35B-A3B-FP8"
MAX_PARALLEL_VIDEOS=3
FIXED_K=3
EXTRA_FLAGS="--no-rec-descs --include-field-meta"
ACTION_REC_RULES="default"
ACTION_PROMPT_VARIANT="default"

MODE="all"
VIDEO_SET="trained"
NUM_SHARDS=1
SHARD_INDEX=0
DRYRUN=""
OVERWRITE=""
SEED=42
TEST_COUNT=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test)                 MODE="test"; shift ;;
    --all)                  MODE="all"; shift ;;
    --rest)                 MODE="rest"; shift ;;
    --synthetic-only-exclude-audit) VIDEO_SET="synthetic_no_audit"; shift ;;
    --num-shards)           NUM_SHARDS="$2"; shift 2 ;;
    --shard-index)          SHARD_INDEX="$2"; shift 2 ;;
    --dryrun)               DRYRUN="1"; shift ;;
    --overwrite)            OVERWRITE="1"; shift ;;
    --seed)                 SEED="$2"; shift 2 ;;
    --models)               MODELS="$2"; shift 2 ;;
    --max-parallel-videos)  MAX_PARALLEL_VIDEOS="$2"; shift 2 ;;
    --action-rec-rules)     ACTION_REC_RULES="$2"; shift 2 ;;
    --action-prompt-variant) ACTION_PROMPT_VARIANT="$2"; shift 2 ;;
    --no-cvs)               ACTION_PROMPT_VARIANT="no_cvs"; shift ;;
    --no-cvs-no-guideline)  ACTION_PROMPT_VARIANT="no_cvs_no_guideline"; shift ;;
    --no-cvs-no-desc)       ACTION_PROMPT_VARIANT="no_cvs_no_desc"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

sanitize_tag() {
  echo "$1" | sed 's/[^a-zA-Z0-9._-]/_/g'
}

mapfile -t ALL_VIDEOS < <("$PYTHON_BIN" - "$VIDEO_SET" "$HF_LABELS_JSONL" "$SYNTHETIC_LABELS_JSONL" "$AUDIT_V11_DIR" <<'PY'
import json
import sys
from pathlib import Path

video_set = sys.argv[1]
trained_labels_path = Path(sys.argv[2])
synthetic_labels_path = Path(sys.argv[3])
audit_dir = Path(sys.argv[4])

labels_path = synthetic_labels_path if video_set == "synthetic_no_audit" else trained_labels_path
seen = set()
videos = []
with labels_path.open() as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        vid = str(row.get("video_id") or "").strip()
        if vid and vid not in seen:
            seen.add(vid)
            videos.append(vid)

if video_set == "synthetic_no_audit":
    audit_videos = {p.stem for p in audit_dir.glob("*.json")}
    videos = [vid for vid in videos if vid not in audit_videos]

for vid in sorted(videos):
    print(vid)
PY
)

FILTERED_VIDEOS=()
for vid in "${ALL_VIDEOS[@]}"; do
  if [ -d "${FRAMES_DIR}/${vid}" ]; then
    FILTERED_VIDEOS+=("$vid")
  else
    echo "WARN: frames not found for ${vid}, skipping"
  fi
done
ALL_VIDEOS=("${FILTERED_VIDEOS[@]}")

_test_videos() {
  printf '%s\n' "${ALL_VIDEOS[@]}" | sort | "$PYTHON_BIN" -c "
import sys, random
vids = [l.strip() for l in sys.stdin if l.strip()]
random.seed(${SEED})
sample = random.sample(vids, min(${TEST_COUNT}, len(vids)))
for v in sample:
    print(v)
"
}

if [ "$MODE" = "test" ]; then
  VIDEOS=()
  while IFS= read -r v; do
    VIDEOS+=("$v")
  done < <(_test_videos)
  echo "TEST MODE: selected ${#VIDEOS[@]} / ${#ALL_VIDEOS[@]} videos (seed=${SEED})"
elif [ "$MODE" = "rest" ]; then
  TEST_SET=()
  while IFS= read -r v; do
    TEST_SET+=("$v")
  done < <(_test_videos)
  declare -A _SKIP
  for v in "${TEST_SET[@]}"; do _SKIP["$v"]=1; done
  VIDEOS=()
  for v in "${ALL_VIDEOS[@]}"; do
    if [ -z "${_SKIP[$v]+x}" ]; then
      VIDEOS+=("$v")
    fi
  done
  echo "REST MODE: selected ${#VIDEOS[@]} / ${#ALL_VIDEOS[@]} videos (skipped ${#TEST_SET[@]} test videos, seed=${SEED})"
else
  VIDEOS=("${ALL_VIDEOS[@]}")
fi

if [ "$NUM_SHARDS" -lt 1 ]; then
  echo "--num-shards must be >= 1"; exit 1
fi
if [ "$SHARD_INDEX" -lt 0 ] || [ "$SHARD_INDEX" -ge "$NUM_SHARDS" ]; then
  echo "--shard-index must be in [0, num-shards)"; exit 1
fi
if [ "$NUM_SHARDS" -gt 1 ]; then
  mapfile -t VIDEOS < <(printf '%s\n' "${VIDEOS[@]}" | "$PYTHON_BIN" -c "
import random
import sys

num_shards = int(sys.argv[1])
shard_index = int(sys.argv[2])
seed = int(sys.argv[3])
videos = [line.strip() for line in sys.stdin if line.strip()]
rng = random.Random(seed)
rng.shuffle(videos)
for idx, video in enumerate(videos):
    if idx % num_shards == shard_index:
        print(video)
" "$NUM_SHARDS" "$SHARD_INDEX" "$SEED")
fi

is_valid_jsonl() {
  local fpath="$1"
  local expected="${2:-0}"
  [ -f "$fpath" ] || return 1
  [ -s "$fpath" ] || return 1
  "$PYTHON_BIN" -c "
import json, sys
expected = int(sys.argv[1])
with open('$fpath') as f:
    lines = f.readlines()
if not lines:
    sys.exit(1)
valid = 0
for line in lines:
    line = line.strip()
    if not line:
        continue
    try:
        json.loads(line)
        valid += 1
    except json.JSONDecodeError:
        sys.exit(1)
if expected > 0 and valid < expected:
    sys.exit(1)
sys.exit(0)
" "$expected" 2>/dev/null
}

FRAME_STEP=5
PRESET_TAG="${PRESET}_fixedk${FIXED_K}"
ACTION_REC_RULES_ARG=""
ACTION_PROMPT_VARIANT_ARG=""
if [ "$ACTION_PROMPT_VARIANT" != "default" ]; then
  case "$ACTION_PROMPT_VARIANT" in
    no_cvs|no_cvs_no_guideline|no_cvs_no_desc) ;;
    *) echo "Unknown action prompt variant: ${ACTION_PROMPT_VARIANT}"; exit 1 ;;
  esac
  PRESET_TAG="${PRESET_TAG}_${ACTION_PROMPT_VARIANT}"
  ACTION_PROMPT_VARIANT_ARG="--action-prompt-variant ${ACTION_PROMPT_VARIANT}"
fi
PRESET_TAG="${PRESET_TAG}_norecdescs_fmeta"
if [ "$ACTION_REC_RULES" != "default" ]; then
  ACTION_REC_RULES_SAFE="$(echo "$ACTION_REC_RULES" | sed 's/_/-/g; s/[^a-zA-Z0-9.-]/_/g')"
  PRESET_TAG="${PRESET_TAG}_arecrules-${ACTION_REC_RULES_SAFE}"
  ACTION_REC_RULES_ARG="--action-rec-rules ${ACTION_REC_RULES}"
fi

echo "=== CoT CVS-Act v1 ==="
echo "Mode:            ${MODE}"
echo "Video set:       ${VIDEO_SET}"
echo "Shard:           ${SHARD_INDEX}/${NUM_SHARDS} (seed=${SEED})"
echo "Models:          ${MODELS}"
echo "Videos:          ${#VIDEOS[@]}"
echo "Preset:          ${PRESET}"
echo "HF labels:       ${HF_LABELS_JSONL}"
if [ "$VIDEO_SET" = "synthetic_no_audit" ]; then
  echo "Synthetic labels:${SYNTHETIC_LABELS_JSONL}"
  echo "Audit excluded:  ${AUDIT_V11_DIR}"
fi
echo "Taxonomy:        ${TAXONOMY_TAG}"
echo "Output root:     ${OUT_ROOT}"
echo "Fixed-K:         ${FIXED_K}"
echo "Action-rec rules:${ACTION_REC_RULES}"
echo "Action prompt:   ${ACTION_PROMPT_VARIANT}"
echo ""

COUNT_FILE=$(mktemp)
: > "$COUNT_FILE"

run_video() {
  local vid="$1"
  local my_total=0 my_launched=0 my_skipped=0
  local PIDS=()

  echo "[video] START ${vid}"

  local n_frames
  n_frames=$(ls "${FRAMES_DIR}/${vid}"/ 2>/dev/null | wc -l)
  local expected_lines=$(( n_frames / FRAME_STEP ))

  for MODEL in ${MODELS}; do
    local MODEL_TAG
    MODEL_TAG=$(sanitize_tag "$MODEL")

    local BL_OUT="${OUT_ROOT}/${PRESET_TAG}/${vid}__${PRESET_TAG}__${MODEL_TAG}__${TAXONOMY_TAG}.jsonl"
    local BL_LOG_DIR="${LOG_DIR}/cot_cvs_act_v1_${PRESET_TAG}"
    local BL_LOG="${BL_LOG_DIR}/${vid}__${PRESET}__${MODEL_TAG}__${TAXONOMY_TAG}.log"
    local BL_CMD="cd ${REPO_ROOT} && ${PYTHON_BIN} scripts/run/baseline.py --image-dir ${FRAMES_DIR}/${vid} --model ${MODEL} --preset ${PRESET} --action-mode recommend --taxonomy-json ${TAXONOMY_JSON} --fixed-k ${FIXED_K} ${ACTION_REC_RULES_ARG} ${ACTION_PROMPT_VARIANT_ARG} ${EXTRA_FLAGS} --out-jsonl ${BL_OUT}"

    my_total=$((my_total + 1))
    if [ -z "$DRYRUN" ] && [ -z "$OVERWRITE" ] && is_valid_jsonl "$BL_OUT" "$expected_lines"; then
      echo "  SKIP [${MODEL}] ${BL_OUT}"
      my_skipped=$((my_skipped + 1))
    else
      my_launched=$((my_launched + 1))
      mkdir -p "$BL_LOG_DIR"
      if [ -n "$DRYRUN" ]; then
        echo "  [dryrun] [${MODEL}] ${BL_CMD}"
      else
        echo "  RUN  [${MODEL}] -> ${BL_LOG}"
        nohup bash -c "$BL_CMD" > "$BL_LOG" 2>&1 &
        PIDS+=($!)
      fi
    fi
  done

  if [ ${#PIDS[@]} -gt 0 ]; then
    echo "  [${vid}] Waiting for ${#PIDS[@]} jobs..."
    wait "${PIDS[@]}"
  fi
  echo "[video] DONE ${vid} (total=${my_total} launched=${my_launched} skipped=${my_skipped})"
  echo "${my_total} ${my_launched} ${my_skipped}" >> "$COUNT_FILE"
}

VIDEO_PIDS=()
for vid in "${VIDEOS[@]}"; do
  while [ ${#VIDEO_PIDS[@]} -ge ${MAX_PARALLEL_VIDEOS} ]; do
    wait -n 2>/dev/null || true
    NEW_PIDS=()
    for pid in "${VIDEO_PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        NEW_PIDS+=("$pid")
      fi
    done
    VIDEO_PIDS=("${NEW_PIDS[@]}")
  done

  run_video "$vid" &
  VIDEO_PIDS+=($!)
done

if [ ${#VIDEO_PIDS[@]} -gt 0 ]; then
  echo ""
  echo "Waiting for final ${#VIDEO_PIDS[@]} video workers..."
  wait "${VIDEO_PIDS[@]}"
fi

TOTAL=0; LAUNCHED=0; SKIPPED=0
while read -r t l s; do
  TOTAL=$((TOTAL + t))
  LAUNCHED=$((LAUNCHED + l))
  SKIPPED=$((SKIPPED + s))
done < "$COUNT_FILE"
rm -f "$COUNT_FILE"

echo ""
echo "========================================="
echo "Total=${TOTAL}  Launched=${LAUNCHED}  Skipped=${SKIPPED}"
echo "Outputs: ${OUT_ROOT}/"
echo "Logs:    ${LOG_DIR}/"
