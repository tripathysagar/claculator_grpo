#!/usr/bin/env bash
set -euo pipefail

SESSION="${COLAB_SESSION:-calc-float-mix-sweep-a100-20260724}"
RESUME_FROM_GRPO40="${RESUME_FROM_GRPO40:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"
LOCAL_ROOT="outputs/evaluations/qwen3-calc-float-mix-sweep-a100-20260724"
LOCAL_NOTEBOOK_ROOT="outputs/notebooks/qwen3-calc-float-mix-sweep-a100-20260724"
TMP_NOTEBOOK_ROOT="/tmp/qwen3-calc-float-mix-sweep-a100-20260724"
REMOTE_EVAL_ROOT="/content/experiments/qwen3-calc-float-mix-sweep-eval"
ARMS=(20 30 40)

declare -A SFT_REVISIONS
declare -A GRPO_REVISIONS

if [[ "$RESUME_FROM_GRPO40" == "1" ]]; then
  for arm in "${ARMS[@]}"; do
    SFT_REVISIONS[$arm]="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$LOCAL_ROOT/float${arm}/sft_hub_revision.json")"
    [[ "${SFT_REVISIONS[$arm]}" =~ ^[0-9a-f]{40}$ ]]
  done
  for arm in 20 30; do
    GRPO_REVISIONS[$arm]="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$LOCAL_ROOT/float${arm}/grpo_hub_revision.json")"
    [[ "${GRPO_REVISIONS[$arm]}" =~ ^[0-9a-f]{40}$ ]]
  done
fi
if [[ "$EVAL_ONLY" == "1" ]]; then
  for arm in "${ARMS[@]}"; do
    SFT_REVISIONS[$arm]="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$LOCAL_ROOT/float${arm}/sft_hub_revision.json")"
    GRPO_REVISIONS[$arm]="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$LOCAL_ROOT/float${arm}/grpo_hub_revision.json")"
    [[ "${SFT_REVISIONS[$arm]}" =~ ^[0-9a-f]{40}$ ]]
    [[ "${GRPO_REVISIONS[$arm]}" =~ ^[0-9a-f]{40}$ ]]
  done
fi

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

activate_dataset() {
  local arm="$1"
  python_code=$(cat <<PY
import os
import shutil
from pathlib import Path

target = Path("/content/data/calculator_qwen3")
if target.is_symlink() or target.is_file():
    target.unlink()
elif target.is_dir():
    shutil.rmtree(target)
source = Path("/content/data/calculator_qwen3_float$arm")
assert (source / "dataset_dict.json").is_file(), source
os.symlink(source, target, target_is_directory=True)
print("DATASET_ACTIVE", "$arm")
PY
)
  printf '%s\n' "$python_code" | colab exec -s "$SESSION" --timeout 60
}

set_sft_controls() {
  local arm="$1"
  python_code=$(cat <<PY
import os
os.environ.update({
    "RUN_MODE": "full",
    "RUN_TAG": "a100float${arm}strict5-20260724",
    "HUB_REPO_ID": "tripathysagar/qwen3-0.6b-calc-sft200-float${arm}-a100-20260724",
    "TRACKIO_PROJECT": "calc-rlvr-sft-float${arm}",
    "TRACKIO_SPACE_ID": "tripathysagar/calc-rlvr-sft",
    "TRACKIO_BUCKET_ID": "tripathysagar/calc-rlvr-sft-bucket",
})
print("SFT_CONTROLS_READY", "$arm")
PY
)
  printf '%s\n' "$python_code" | colab exec -s "$SESSION" --timeout 60
}

set_grpo_controls() {
  local arm="$1"
  local revision="$2"
  python_code=$(cat <<PY
import os
os.environ.update({
    "RUN_MODE": "full",
    "RUN_TAG": "a100float${arm}strict5-20260724",
    "USE_VLLM": "1",
    "ALLOW_MIXED_DATASET": "1",
    "SFT_ADAPTER_ID": "tripathysagar/qwen3-0.6b-calc-sft200-float${arm}-a100-20260724",
    "SFT_ADAPTER_REVISION": "$revision",
    "GRPO_HUB_REPO_ID": "tripathysagar/qwen3-0.6b-calc-grpo600-float${arm}-strict5-a100-20260724",
    "GRPO_TRACKIO_PROJECT": "calc-rlvr-grpo-float${arm}",
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
print("GRPO_CONTROLS_READY", "$arm")
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
print("ARCHIVE_READY", shutil.make_archive("$remote_archive", "zip", root_dir=source))
PY
)
  printf '%s\n' "$python_code" | colab exec -s "$SESSION" --timeout 600
}

remove_remote_artifacts() {
  local remote_directory="$1"
  local remote_archive="$2"
  python_code=$(cat <<PY
import shutil
from pathlib import Path
directory = Path("$remote_directory")
archive = Path("$remote_archive")
if directory.exists():
    shutil.rmtree(directory)
if archive.exists():
    archive.unlink()
print("REMOTE_ARTIFACTS_RELEASED")
PY
)
  printf '%s\n' "$python_code" | colab exec -s "$SESSION" --timeout 300
}

if [[ ! -f ".env" ]]; then
  echo "Missing ignored project .env" >&2
  exit 1
fi
for arm in "${ARMS[@]}"; do
  if [[ ! -d "data/calculator_qwen3_float${arm}" ]]; then
    echo "Missing data/calculator_qwen3_float${arm}" >&2
    exit 1
  fi
done

mkdir -p "$LOCAL_ROOT" "$LOCAL_NOTEBOOK_ROOT" "$TMP_NOTEBOOK_ROOT"
for arm in "${ARMS[@]}"; do
  mkdir -p "$LOCAL_ROOT/float${arm}" "$LOCAL_NOTEBOOK_ROOT/float${arm}" "$TMP_NOTEBOOK_ROOT/float${arm}"
  cp "notebooks/sft_qwen3_calculator.ipynb" "$TMP_NOTEBOOK_ROOT/float${arm}/sft_qwen3_calculator.ipynb"
  cp "notebooks/grpo_qwen3_calculator.ipynb" "$TMP_NOTEBOOK_ROOT/float${arm}/grpo_qwen3_calculator.ipynb"
  tar -czf "/tmp/calculator_qwen3_float${arm}-20260724.tar.gz" \
    -C "data" "calculator_qwen3_float${arm}"
done

colab version
colab sessions
colab new -s "$SESSION" --gpu A100
colab status -s "$SESSION"
printf '%s\n' \
  "from pathlib import Path; Path('/content/data').mkdir(parents=True, exist_ok=True); Path('/content/experiments').mkdir(parents=True, exist_ok=True); print('REMOTE_DIRS_READY')" |
  colab exec -s "$SESSION" --timeout 60

for arm in "${ARMS[@]}"; do
  colab upload -s "$SESSION" \
    "/tmp/calculator_qwen3_float${arm}-20260724.tar.gz" \
    "/content/calculator_qwen3_float${arm}.tar.gz"
done
colab upload -s "$SESSION" \
  "data/calculator_qwen3_float_mix_manifest.json" \
  "/content/data/calculator_qwen3_float_mix_manifest.json"
colab upload -s "$SESSION" "data/calculator_qwen3_sft200_ids.txt" "/content/data/calculator_qwen3_sft200_ids.txt"
colab upload -s "$SESSION" "data/calculator_qwen3_grpo600_ids.txt" "/content/data/calculator_qwen3_grpo600_ids.txt"
colab upload -s "$SESSION" "scripts/calculator_semantics.py" "/content/calculator_semantics.py"
colab upload -s "$SESSION" "scripts/evaluate_sft_qwen3_calculator.py" "/content/evaluate_sft_qwen3_calculator.py"
colab upload -s "$SESSION" "notebooks/sft_qwen3_calculator.ipynb" "/content/sft_qwen3_calculator.ipynb"
colab upload -s "$SESSION" "notebooks/grpo_qwen3_calculator.ipynb" "/content/grpo_qwen3_calculator.ipynb"

python_code=$(cat <<'PY'
import tarfile
from pathlib import Path

data = Path("/content/data")
for arm in (20, 30, 40):
    with tarfile.open(f"/content/calculator_qwen3_float{arm}.tar.gz") as archive:
        archive.extractall(data, filter="data")
    assert (data / f"calculator_qwen3_float{arm}" / "dataset_dict.json").is_file()
print("FLOAT_SWEEP_DATASETS_READY")
PY
)
printf '%s\n' "$python_code" | colab exec -s "$SESSION" --timeout 300

if [[ "$RESUME_FROM_GRPO40" != "1" && "$EVAL_ONLY" != "1" ]]; then
  for arm in "${ARMS[@]}"; do
    colab restart-kernel -s "$SESSION"
    inject_secrets
    activate_dataset "$arm"
    set_sft_controls "$arm"
    colab exec -s "$SESSION" \
      -f "$TMP_NOTEBOOK_ROOT/float${arm}/sft_qwen3_calculator.ipynb" \
      --timeout 7200 | tee "$LOCAL_ROOT/float${arm}/sft-execution.log"
    if [[ -f "$TMP_NOTEBOOK_ROOT/float${arm}/sft_qwen3_calculator_output.ipynb" ]]; then
      mv "$TMP_NOTEBOOK_ROOT/float${arm}/sft_qwen3_calculator_output.ipynb" \
        "$LOCAL_NOTEBOOK_ROOT/float${arm}/sft_qwen3_calculator_output.ipynb"
    fi

    remote_sft_root="/content/experiments/qwen3-calc-sft200-seed42-a100float${arm}strict5-20260724"
    colab ls -s "$SESSION" "$remote_sft_root/full"
    colab restart-kernel -s "$SESSION"
    inject_secrets
    printf '%s\n' "exec(open('/content/upload_sft_adapter.py').read())" |
      colab exec -s "$SESSION" --timeout 900 | tee "$LOCAL_ROOT/float${arm}/sft-upload.log"
    colab download -s "$SESSION" \
      "$remote_sft_root/full/hub_revision.json" \
      "$LOCAL_ROOT/float${arm}/sft_hub_revision.json"
    SFT_REVISIONS[$arm]="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$LOCAL_ROOT/float${arm}/sft_hub_revision.json")"
    [[ "${SFT_REVISIONS[$arm]}" =~ ^[0-9a-f]{40}$ ]]
    archive_remote_directory "$remote_sft_root" "/content/float${arm}-sft-artifacts"
    colab download -s "$SESSION" \
      "/content/float${arm}-sft-artifacts.zip" \
      "$LOCAL_ROOT/float${arm}/sft-artifacts.zip"
    remove_remote_artifacts "$remote_sft_root" "/content/float${arm}-sft-artifacts.zip"
    echo "SFT_ARM_COMPLETE $arm ${SFT_REVISIONS[$arm]}"
  done
fi

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

for arm in "${ARMS[@]}"; do
  if [[ "$EVAL_ONLY" == "1" ]]; then
    continue
  fi
  if [[ "$RESUME_FROM_GRPO40" == "1" && "$arm" != "40" ]]; then
    continue
  fi
  colab restart-kernel -s "$SESSION"
  inject_secrets
  activate_dataset "$arm"
  set_grpo_controls "$arm" "${SFT_REVISIONS[$arm]}"
  colab exec -s "$SESSION" \
    -f "$TMP_NOTEBOOK_ROOT/float${arm}/grpo_qwen3_calculator.ipynb" \
    --timeout 7200 | tee "$LOCAL_ROOT/float${arm}/grpo-execution.log"
  if [[ -f "$TMP_NOTEBOOK_ROOT/float${arm}/grpo_qwen3_calculator_output.ipynb" ]]; then
    mv "$TMP_NOTEBOOK_ROOT/float${arm}/grpo_qwen3_calculator_output.ipynb" \
      "$LOCAL_NOTEBOOK_ROOT/float${arm}/grpo_qwen3_calculator_output.ipynb"
  fi

  remote_grpo_run="/content/experiments/qwen3-calc-grpo600-full-vllm-colocate-seed42-a100float${arm}strict5-20260724"
  colab ls -s "$SESSION" "$remote_grpo_run"
  colab restart-kernel -s "$SESSION"
  inject_secrets
  printf '%s\n' "exec(open('/content/upload_grpo_adapter.py').read())" |
    colab exec -s "$SESSION" --timeout 900 | tee "$LOCAL_ROOT/float${arm}/grpo-upload.log"
  colab download -s "$SESSION" \
    "$remote_grpo_run/hub_revision.json" \
    "$LOCAL_ROOT/float${arm}/grpo_hub_revision.json"
  GRPO_REVISIONS[$arm]="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$LOCAL_ROOT/float${arm}/grpo_hub_revision.json")"
  [[ "${GRPO_REVISIONS[$arm]}" =~ ^[0-9a-f]{40}$ ]]
  archive_remote_directory "$remote_grpo_run" "/content/float${arm}-grpo-artifacts"
  colab download -s "$SESSION" \
    "/content/float${arm}-grpo-artifacts.zip" \
    "$LOCAL_ROOT/float${arm}/grpo-artifacts.zip"
  remove_remote_artifacts "$remote_grpo_run" "/content/float${arm}-grpo-artifacts.zip"
  echo "GRPO_ARM_COMPLETE $arm ${GRPO_REVISIONS[$arm]}"
done

for arm in "${ARMS[@]}"; do
  for stage in sft grpo; do
    if [[ "$stage" == "sft" ]]; then
      policy="sft-float${arm}"
      repo="tripathysagar/qwen3-0.6b-calc-sft200-float${arm}-a100-20260724"
      revision="${SFT_REVISIONS[$arm]}"
    else
      policy="grpo-float${arm}-strict5"
      repo="tripathysagar/qwen3-0.6b-calc-grpo600-float${arm}-strict5-a100-20260724"
      revision="${GRPO_REVISIONS[$arm]}"
    fi
    if [[ "$EVAL_ONLY" == "1" && -f "$LOCAL_ROOT/${policy}-evaluation.zip" ]]; then
      echo "POLICY_EVAL_ALREADY_COMPLETE $policy"
      continue
    fi

    colab restart-kernel -s "$SESSION"
    inject_secrets
    activate_dataset 20
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
    archive_remote_directory "$REMOTE_EVAL_ROOT/$policy" "/content/${policy}-evaluation"
    colab download -s "$SESSION" \
      "/content/${policy}-evaluation.zip" \
      "$LOCAL_ROOT/${policy}-evaluation.zip"
    remove_remote_artifacts \
      "$REMOTE_EVAL_ROOT/$policy" \
      "/content/${policy}-evaluation.zip"
  done
done

colab log -s "$SESSION" -o "$LOCAL_ROOT/colab-session.ipynb"
colab ls -s "$SESSION" "/content/experiments"
colab stop -s "$SESSION"
trap - EXIT
colab sessions
echo "FLOAT_MIX_SWEEP_COMPLETE $LOCAL_ROOT"
