#!/usr/bin/env bash
set -euo pipefail

# Public CVS-Act v1 SurGent runner.
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
OUT_ROOT="outputs/surgent_sequential_agent_audit_v11_simple"
LOG_DIR="outputs/logs"

MODELS="gpt-4.1-mini gemini-2.5-flash claude-haiku-4-5-20251001 Qwen/Qwen3.6-35B-A3B-FP8"
PRESET="cot"
PREFERRED_TOOLS="scene_cvs_analyzer,action_rec"
FINALIZE_TOOLS=""
MAX_STEPS="${SURGENT_MAX_STEPS:-5}"
MAX_PARALLEL_VIDEOS=3
FIXED_K=3
EXTRA_FLAGS="--no-rec-descs --include-field-meta"
CVS_CONTEXT_MODE="latest"
ACTION_REC_RULES="default"

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
    --preferred-tools)      PREFERRED_TOOLS="$2"; shift 2 ;;
    --finalize-tools)       FINALIZE_TOOLS="$2"; shift 2 ;;
    --scene-preset)         PRESET="$2"; shift 2 ;;
    --cvs-context-mode)     CVS_CONTEXT_MODE="$2"; shift 2 ;;
    --action-rec-rules)     ACTION_REC_RULES="$2"; shift 2 ;;
    --max-steps)            MAX_STEPS="$2"; shift 2 ;;
    --max-parallel-videos)  MAX_PARALLEL_VIDEOS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

sanitize_tag() {
  echo "$1" | sed 's/[^a-zA-Z0-9._-]/_/g'
}

preferred_tag() {
  local preferred_tools="$1"
  if [ -z "$preferred_tools" ]; then
    echo "pref-none"
  else
    echo "pref-$(echo "$preferred_tools" | sed 's/scene_cvs_analyzer/cvs/g; s/scene_cvs_act_analyzer/cvsact/g; s/scene_snapper/snap/g; s/clip_analyzer/clip/g; s/action_rec/arec/g; s/zoom/zoom/g; s/,/_/g')"
  fi
}

finalize_tag() {
  local finalize_tools="$1"
  if [ -z "$finalize_tools" ]; then
    echo ""
  else
    echo "_final-$(echo "$finalize_tools" | sed 's/scene_cvs_analyzer/cvs/g; s/scene_cvs_act_analyzer/cvsact/g; s/scene_snapper/snap/g; s/clip_analyzer/clip/g; s/action_rec/arec/g; s/zoom/zoom/g; s/,/-/g')"
  fi
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
PREF_TAG="$(preferred_tag "$PREFERRED_TOOLS")"
PREF_TAG_DIR="${PREF_TAG}_steps${MAX_STEPS}_fixedk${FIXED_K}_norecdescs_fmeta"
case "$CVS_CONTEXT_MODE" in
  latest) ;;
  current) PREF_TAG_DIR="${PREF_TAG_DIR}_cvsctx-current" ;;
  full) PREF_TAG_DIR="${PREF_TAG_DIR}_cvsctx-full" ;;
  video_summary) PREF_TAG_DIR="${PREF_TAG_DIR}_cvsctx-vidsummary" ;;
  *) echo "Unknown --cvs-context-mode: ${CVS_CONTEXT_MODE}"; exit 1 ;;
esac
PREF_TAG_DIR="${PREF_TAG_DIR}$(finalize_tag "$FINALIZE_TOOLS")"
ACTION_REC_RULES_ARG=""
if [ "$ACTION_REC_RULES" != "default" ]; then
  ACTION_REC_RULES_SAFE="$(echo "$ACTION_REC_RULES" | sed 's/_/-/g; s/[^a-zA-Z0-9.-]/_/g')"
  PREF_TAG_DIR="${PREF_TAG_DIR}_arecrules-${ACTION_REC_RULES_SAFE}"
  ACTION_REC_RULES_ARG="--action-rec-rules ${ACTION_REC_RULES}"
fi

echo "=== SurGent CVS-Act v1 ==="
echo "Mode:            ${MODE}"
echo "Video set:       ${VIDEO_SET}"
echo "Shard:           ${SHARD_INDEX}/${NUM_SHARDS} (seed=${SEED})"
echo "Models:          ${MODELS}"
echo "Videos:          ${#VIDEOS[@]}"
echo "Preset:          ${PRESET}"
echo "Preferred tools: ${PREFERRED_TOOLS}"
echo "Finalize tools:  ${FINALIZE_TOOLS:-none}"
echo "HF labels:       ${HF_LABELS_JSONL}"
if [ "$VIDEO_SET" = "synthetic_no_audit" ]; then
  echo "Synthetic labels:${SYNTHETIC_LABELS_JSONL}"
  echo "Audit excluded:  ${AUDIT_V11_DIR}"
fi
echo "Taxonomy:        ${TAXONOMY_TAG}"
echo "Output root:     ${OUT_ROOT}"
echo "Steps:           ${MAX_STEPS}"
echo "Fixed-K:         ${FIXED_K}"
echo "CVS context:     ${CVS_CONTEXT_MODE}"
echo "Action-rec rules:${ACTION_REC_RULES}"
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

    local SG_OUT="${OUT_ROOT}/${PRESET}/${PREF_TAG_DIR}/${vid}__multi__C1-C2-C3__${MODEL_TAG}__${TAXONOMY_TAG}.jsonl"
    local SG_LOG_DIR="${LOG_DIR}/surgent_cvs_act_v1_${PRESET}_${PREF_TAG_DIR}"
    local SG_LOG="${SG_LOG_DIR}/${vid}__surgent_seq__${MODEL_TAG}__${TAXONOMY_TAG}.log"
    local FINALIZE_ARG=""
    if [ -n "$FINALIZE_TOOLS" ]; then
      FINALIZE_ARG="--finalize-tools ${FINALIZE_TOOLS}"
    fi
    local SG_CMD="cd ${REPO_ROOT} && SURGENT_MAX_STEPS=${MAX_STEPS} ${PYTHON_BIN} scripts/run/run_surgent_sequential.py --image-dir ${FRAMES_DIR}/${vid} --model ${MODEL} --scene-preset ${PRESET} --preferred-tools ${PREFERRED_TOOLS} ${FINALIZE_ARG} --taxonomy-json ${TAXONOMY_JSON} --fixed-k ${FIXED_K} --cvs-context-mode ${CVS_CONTEXT_MODE} ${ACTION_REC_RULES_ARG} ${EXTRA_FLAGS} --out-jsonl ${SG_OUT}"

    my_total=$((my_total + 1))
    if [ -z "$DRYRUN" ] && [ -z "$OVERWRITE" ] && is_valid_jsonl "$SG_OUT" "$expected_lines"; then
      echo "  SKIP [${MODEL}] ${SG_OUT}"
      my_skipped=$((my_skipped + 1))
    else
      my_launched=$((my_launched + 1))
      mkdir -p "$SG_LOG_DIR"
      if [ -n "$DRYRUN" ]; then
        echo "  [dryrun] [${MODEL}] ${SG_CMD}"
      else
        echo "  RUN  [${MODEL}] -> ${SG_LOG}"
        nohup bash -c "$SG_CMD" > "$SG_LOG" 2>&1 &
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
  echo "Waiting for all videos to finish..."
  wait "${VIDEO_PIDS[@]}"
fi

TOTAL=0
LAUNCHED=0
SKIPPED=0
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
