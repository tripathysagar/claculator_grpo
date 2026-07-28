#!/usr/bin/env bash
set -euo pipefail

SESSION="${COLAB_SESSION:-calc-negative10-strict5-a100-20260724}"
RUN_TAG="${RUN_TAG:-a100negative10strict5-20260724}"
SFT_REPO="${SFT_REPO:-tripathysagar/qwen3-0.6b-calc-sft200-negative10-a100-20260724}"
GRPO_REPO="${GRPO_REPO:-tripathysagar/qwen3-0.6b-calc-grpo600-negative10-strict5-a100-20260724}"
DATASET_DIR="data/calculator_qwen3_negative10"
DATASET_MANIFEST="data/calculator_qwen3_negative10_manifest.json"
DATASET_ARCHIVE="/tmp/calculator_qwen3_negative10-20260724.tar.gz"
LOCAL_ROOT="outputs/evaluations/qwen3-calc-negative10-strict5-a100-20260724"
LOCAL_NOTEBOOK_ROOT="outputs/notebooks/qwen3-calc-negative10-strict5-a100-20260724"
TMP_NOTEBOOK_ROOT="/tmp/qwen3-calc-negative10-strict5-a100-20260724"
REMOTE_SFT_ROOT="/content/experiments/qwen3-calc-sft200-seed42-${RUN_TAG}"
REMOTE_GRPO_RUN="/content/experiments/qwen3-calc-grpo600-full-vllm-colocate-seed42-${RUN_TAG}"
REMOTE_EVAL_ROOT="/content/experiments/qwen3-calc-negative10-strict5-eval"

stop_session() {
  colab stop -s "$SESSION" >/dev/null 2>&1 || true
}
trap stop_session EXIT

inject_secrets() {
  colab upload -s "$SESSION" ".env" "/content/.env"
  colab exec -s "$SESSION" \
    -f ".cursor/skills/colab-cli/scripts/load-env.py" \
    --timeout 60
}

set_sft_controls() {
  python_code=$(cat <<PY
import os
os.environ.update({
    "RUN_MODE": "full",
    "RUN_TAG": "$RUN_TAG",
    "HUB_REPO_ID": "$SFT_REPO",
    "TRACKIO_PROJECT": "calc-rlvr-sft-negative10",
    "TRACKIO_SPACE_ID": "tripathysagar/calc-rlvr-sft",
    "TRACKIO_BUCKET_ID": "tripathysagar/calc-rlvr-sft-bucket",
})
print("SFT_CONTROLS_READY")
PY
)
  printf '%s\n' "$python_code" | colab exec -s "$SESSION" --timeout 60
}

set_grpo_controls() {
  local sft_revision="$1"
  python_code=$(cat <<PY
import os
os.environ.update({
    "RUN_MODE": "full",
    "RUN_TAG": "$RUN_TAG",
    "USE_VLLM": "1",
    "ALLOW_MIXED_DATASET": "1",
    "SFT_ADAPTER_ID": "$SFT_REPO",
    "SFT_ADAPTER_REVISION": "$sft_revision",
    "GRPO_HUB_REPO_ID": "$GRPO_REPO",
    "GRPO_TRACKIO_PROJECT": "calc-rlvr-grpo-negative10",
    "GRPO_TRACKIO_SPACE_ID": "tripathysagar/calc-rlvr-grpo",
    "GRPO_TRACKIO_BUCKET_ID": "tripathysagar/calc-rlvr-grpo-bucket",
    "NUM_GENERATIONS": "4",
    "PER_DEVICE_TRAIN_BATCH_SIZE": "8",
    "GRADIENT_ACCUMULATION_STEPS": "4",
    "STEPS_PER_GENERATION": "4",
    "VLLM_GPU_MEMORY_UTILIZATION": "0.20",
    "STRICT_INVALID_REWARD": "1",
    "STRICT_VALID_CORRECT_REWARD": "5.0",
    "LEGAL_PROGRESS_BONUS_MAX": "0.0",
    "LEGAL_PROGRESS_BONUS_PER_CALL": "0.0",
    "EXECUTION_BINARY_REWARD": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
})
print("GRPO_CONTROLS_READY")
PY
)
  printf '%s\n' "$python_code" | colab exec -s "$SESSION" --timeout 60
}

archive_remote_directory() {
  local remote_directory="$1"
  local remote_archive="$2"
  python_code=$(cat <<PY
import shutil
from pathlib import Path
source = Path("$remote_directory")
assert source.is_dir(), source
archive = shutil.make_archive("$remote_archive", "zip", root_dir=source)
print("ARCHIVE_READY", archive)
PY
)
  printf '%s\n' "$python_code" | colab exec -s "$SESSION" --timeout 300
}

if [[ ! -d "$DATASET_DIR" || ! -f "$DATASET_MANIFEST" ]]; then
  echo "Missing generated negative10 dataset or manifest" >&2
  exit 1
fi
if [[ ! -f ".env" ]]; then
  echo "Missing ignored project .env required for Hub uploads" >&2
  exit 1
fi

mkdir -p "$LOCAL_ROOT" "$LOCAL_NOTEBOOK_ROOT" "$TMP_NOTEBOOK_ROOT"
cp "notebooks/sft_qwen3_calculator.ipynb" "$TMP_NOTEBOOK_ROOT/sft_qwen3_calculator.ipynb"
cp "notebooks/grpo_qwen3_calculator.ipynb" "$TMP_NOTEBOOK_ROOT/grpo_qwen3_calculator.ipynb"
tar -czf "$DATASET_ARCHIVE" -C "data" "calculator_qwen3_negative10"

colab version
colab help new
colab sessions
colab new -s "$SESSION" --gpu A100
colab status -s "$SESSION"

printf '%s\n' \
  "from pathlib import Path; Path('/content/data').mkdir(parents=True, exist_ok=True); Path('/content/experiments').mkdir(parents=True, exist_ok=True); print('REMOTE_DIRS_READY')" |
  colab exec -s "$SESSION" --timeout 60
colab upload -s "$SESSION" "$DATASET_ARCHIVE" "/content/calculator_qwen3_negative10.tar.gz"
colab upload -s "$SESSION" "$DATASET_MANIFEST" "/content/data/calculator_qwen3_negative10_manifest.json"
colab upload -s "$SESSION" "data/calculator_qwen3_sft200_ids.txt" "/content/data/calculator_qwen3_sft200_ids.txt"
colab upload -s "$SESSION" "data/calculator_qwen3_grpo600_ids.txt" "/content/data/calculator_qwen3_grpo600_ids.txt"
colab upload -s "$SESSION" "scripts/calculator_semantics.py" "/content/calculator_semantics.py"
colab upload -s "$SESSION" "scripts/evaluate_sft_qwen3_calculator.py" "/content/evaluate_sft_qwen3_calculator.py"
colab upload -s "$SESSION" "notebooks/sft_qwen3_calculator.ipynb" "/content/sft_qwen3_calculator.ipynb"
colab upload -s "$SESSION" "notebooks/grpo_qwen3_calculator.ipynb" "/content/grpo_qwen3_calculator.ipynb"

python_code=$(cat <<'PY'
import shutil
import tarfile
from pathlib import Path

data = Path("/content/data")
with tarfile.open("/content/calculator_qwen3_negative10.tar.gz") as archive:
    archive.extractall(data, filter="data")
source = data / "calculator_qwen3_negative10"
target = data / "calculator_qwen3"
if target.exists():
    shutil.rmtree(target)
source.rename(target)
assert (target / "dataset_dict.json").is_file()
print("NEGATIVE10_DATASET_READY")
PY
)
printf '%s\n' "$python_code" | colab exec -s "$SESSION" --timeout 180

inject_secrets
set_sft_controls
colab exec -s "$SESSION" \
  -f "$TMP_NOTEBOOK_ROOT/sft_qwen3_calculator.ipynb" \
  --timeout 7200 | tee "$LOCAL_ROOT/sft-execution.log"
if [[ -f "$TMP_NOTEBOOK_ROOT/sft_qwen3_calculator_output.ipynb" ]]; then
  mv "$TMP_NOTEBOOK_ROOT/sft_qwen3_calculator_output.ipynb" \
    "$LOCAL_NOTEBOOK_ROOT/sft_qwen3_calculator_output.ipynb"
fi
colab ls -s "$SESSION" "$REMOTE_SFT_ROOT/full"

colab restart-kernel -s "$SESSION"
inject_secrets
printf '%s\n' "exec(open('/content/upload_sft_adapter.py').read())" |
  colab exec -s "$SESSION" --timeout 900 | tee "$LOCAL_ROOT/sft-upload.log"
colab download -s "$SESSION" \
  "$REMOTE_SFT_ROOT/full/hub_revision.json" \
  "$LOCAL_ROOT/sft_hub_revision.json"
SFT_REVISION="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$LOCAL_ROOT/sft_hub_revision.json")"
[[ "$SFT_REVISION" =~ ^[0-9a-f]{40}$ ]]
archive_remote_directory "$REMOTE_SFT_ROOT" "/content/negative10-sft-artifacts"
colab download -s "$SESSION" \
  "/content/negative10-sft-artifacts.zip" \
  "$LOCAL_ROOT/negative10-sft-artifacts.zip"

python_code=$(cat <<'PY'
import importlib.metadata
import subprocess
import sys

wheel = (
    "https://github.com/vllm-project/vllm/releases/download/v0.23.0/"
    "vllm-0.23.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
)
try:
    installed_vllm = importlib.metadata.version("vllm")
except importlib.metadata.PackageNotFoundError:
    installed_vllm = ""
installed_torch = importlib.metadata.version("torch")
if not (
    installed_vllm.startswith("0.23.0+cu129") and "+cu129" in installed_torch
):
    subprocess.run(
        [
            "uv", "pip", "install", "--system", "--reinstall", wheel,
            "--extra-index-url", "https://download.pytorch.org/whl/cu129",
            "--index-strategy", "unsafe-best-match",
        ],
        check=True,
    )
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q", "--upgrade",
        "trl==1.8.0", "transformers==5.14.1", "peft==0.19.1",
        "datasets==5.0.0", "accelerate==1.14.0", "trackio==0.31.5",
        "torchao==0.17.0", "jmespath==1.0.1",
    ],
    check=True,
)
print("GRPO_STACK_INSTALLED")
PY
)
printf '%s\n' "$python_code" |
  colab exec -s "$SESSION" --timeout 1800 | tee "$LOCAL_ROOT/grpo-install.log"
colab restart-kernel -s "$SESSION"
inject_secrets
set_grpo_controls "$SFT_REVISION"
colab exec -s "$SESSION" \
  -f "$TMP_NOTEBOOK_ROOT/grpo_qwen3_calculator.ipynb" \
  --timeout 7200 | tee "$LOCAL_ROOT/grpo-execution.log"
if [[ -f "$TMP_NOTEBOOK_ROOT/grpo_qwen3_calculator_output.ipynb" ]]; then
  mv "$TMP_NOTEBOOK_ROOT/grpo_qwen3_calculator_output.ipynb" \
    "$LOCAL_NOTEBOOK_ROOT/grpo_qwen3_calculator_output.ipynb"
fi
colab ls -s "$SESSION" "$REMOTE_GRPO_RUN"

colab restart-kernel -s "$SESSION"
inject_secrets
printf '%s\n' "exec(open('/content/upload_grpo_adapter.py').read())" |
  colab exec -s "$SESSION" --timeout 900 | tee "$LOCAL_ROOT/grpo-upload.log"
colab download -s "$SESSION" \
  "$REMOTE_GRPO_RUN/hub_revision.json" \
  "$LOCAL_ROOT/grpo_hub_revision.json"
GRPO_REVISION="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$LOCAL_ROOT/grpo_hub_revision.json")"
[[ "$GRPO_REVISION" =~ ^[0-9a-f]{40}$ ]]
archive_remote_directory "$REMOTE_GRPO_RUN" "/content/negative10-grpo-artifacts"
colab download -s "$SESSION" \
  "/content/negative10-grpo-artifacts.zip" \
  "$LOCAL_ROOT/negative10-grpo-artifacts.zip"

for policy in sft-negative10 grpo-negative10-strict5; do
  if [[ "$policy" == "sft-negative10" ]]; then
    repo="$SFT_REPO"
    revision="$SFT_REVISION"
  else
    repo="$GRPO_REPO"
    revision="$GRPO_REVISION"
  fi

  colab restart-kernel -s "$SESSION"
  inject_secrets
  python_code=$(cat <<PY
import gc
import runpy
import sys
import torch

sys.path.insert(0, "/content")
for split in ("eval", "ood_length", "ood_negative", "ood_float"):
    sys.argv = [
        "evaluate_sft_qwen3_calculator.py",
        "--adapter", "$repo",
        "--adapter-revision", "$revision",
        "--tokenizer", "$repo",
        "--tokenizer-revision", "$revision",
        "--policy-name", "$policy",
        "--greedy",
        "--dataset", "/content/data/calculator_qwen3",
        "--split", split,
        "--output-dir", "$REMOTE_EVAL_ROOT/$policy/" + split,
        "--seeds", "42",
        "--batch-size", "16",
        "--max-new-tokens", "128",
    ]
    namespace = runpy.run_path(
        "/content/evaluate_sft_qwen3_calculator.py", run_name="__main__"
    )
    del namespace
    gc.collect()
    torch.cuda.empty_cache()
print("POLICY_EVAL_COMPLETE", "$policy")
PY
)
  printf '%s\n' "$python_code" |
    colab exec -s "$SESSION" --timeout 7200 | tee "$LOCAL_ROOT/${policy}-eval.log"
  colab ls -s "$SESSION" "$REMOTE_EVAL_ROOT/$policy"
  archive_remote_directory \
    "$REMOTE_EVAL_ROOT/$policy" \
    "/content/${policy}-evaluation"
  colab download -s "$SESSION" \
    "/content/${policy}-evaluation.zip" \
    "$LOCAL_ROOT/${policy}-evaluation.zip"
done

colab log -s "$SESSION" -o "$LOCAL_ROOT/colab-session.ipynb"
colab ls -s "$SESSION" "/content/experiments"
colab stop -s "$SESSION"
trap - EXIT
colab sessions
echo "NEGATIVE10_ABLATION_COMPLETE $LOCAL_ROOT"
