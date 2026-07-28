#!/usr/bin/env bash
set -euo pipefail

SESSION="${COLAB_SESSION:-calc-ood-a100-20260724}"
POLICY_FILTER="${POLICY_FILTER:-}"
REMOTE_ROOT="/content/experiments/qwen3-calc-ood-a100-20260724"
LOCAL_ROOT="outputs/evaluations/qwen3-calc-ood-a100-20260724"
DATA_ARCHIVE="/tmp/calculator_qwen3-20260724.tar.gz"

POLICIES=(
  "sft|tripathysagar/qwen3-0.6b-calc-sft200-a100-rerun-20260724|57ebda77e10cac33866472b38432bf2cf4ac3b3a"
  "terminal|tripathysagar/qwen3-0.6b-calc-grpo600-terminal-a100-rerun-20260724|881af2a6f78b8f99444cad3f0397d223fe42e348"
  "progress01|tripathysagar/qwen3-0.6b-calc-grpo600-progress01-a100-rerun-20260724|dfff17d7b25b2b7017218705492a421c0a3110d5"
  "progress05|tripathysagar/qwen3-0.6b-calc-grpo600-progress05-a100-rerun-20260724|6e356b4483b2d15b1dc3781e4b2e8bc6cbd96c3f"
  "strict2|tripathysagar/qwen3-0.6b-calc-grpo600-strict2-a100-rerun-20260724|711d20dad63801faf2bc8d3bdfaea4afdb259a7c"
  "strict5|tripathysagar/qwen3-0.6b-calc-grpo600-strict5-a100-rerun-20260724|462b3331214ffbb14d85718ba6198e502d38e2ec"
  "strict10|tripathysagar/qwen3-0.6b-calc-grpo600-strict10-a100-rerun-20260724|fcc053edc05ab903fe2c52e665f29bc79957bb39"
  "binary|tripathysagar/qwen3-0.6b-calc-grpo600-binary-a100-rerun-20260724|430adbca6bbe3beb31e5d8ef79ad34f2f8264aa8"
)
SPLITS=(ood_length ood_negative ood_float)

stop_session() {
  colab stop -s "$SESSION" >/dev/null 2>&1 || true
}
trap stop_session EXIT

if [[ ! -f "$DATA_ARCHIVE" ]]; then
  echo "Missing dataset archive: $DATA_ARCHIVE" >&2
  exit 1
fi

colab sessions
colab new -s "$SESSION" --gpu A100
colab status -s "$SESSION"

echo "from pathlib import Path; Path('/content/data').mkdir(parents=True, exist_ok=True); Path('$REMOTE_ROOT').mkdir(parents=True, exist_ok=True)" |
  colab exec -s "$SESSION" --timeout 60
colab upload -s "$SESSION" "$DATA_ARCHIVE" "/content/calculator_qwen3-20260724.tar.gz"
echo "import tarfile; tarfile.open('/content/calculator_qwen3-20260724.tar.gz').extractall('/content/data', filter='data'); print('DATASET_READY')" |
  colab exec -s "$SESSION" --timeout 120
colab upload -s "$SESSION" \
  "scripts/evaluate_sft_qwen3_calculator.py" \
  "/content/evaluate_sft_qwen3_calculator.py"
colab upload -s "$SESSION" \
  "scripts/calculator_semantics.py" \
  "/content/calculator_semantics.py"

colab install -s "$SESSION" \
  "transformers==5.14.1" \
  "peft==0.19.1" \
  "datasets==5.0.0" \
  "accelerate==1.14.0" \
  "torchao==0.17.0"
colab restart-kernel -s "$SESSION"

for policy in "${POLICIES[@]}"; do
  IFS="|" read -r policy_name repo_id revision <<<"$policy"
  if [[ -n "$POLICY_FILTER" && " $POLICY_FILTER " != *" $policy_name "* ]]; then
    continue
  fi

  # A fresh kernel prevents one merged adapter from influencing the next.
  colab restart-kernel -s "$SESSION"
  colab upload -s "$SESSION" ".env" "/content/.env"
  colab exec -s "$SESSION" \
    -f ".cursor/skills/colab-cli/scripts/load-env.py" \
    --timeout 60

  python_code=$(cat <<PY
import gc
import runpy
import sys
import torch

sys.path.insert(0, "/content")
policy_name = ${policy_name@Q}
repo_id = ${repo_id@Q}
revision = ${revision@Q}
for split in ("ood_length", "ood_negative", "ood_float"):
    output_dir = f"$REMOTE_ROOT/{policy_name}/{split}"
    sys.argv = [
        "evaluate_sft_qwen3_calculator.py",
        "--adapter", repo_id,
        "--adapter-revision", revision,
        "--tokenizer", repo_id,
        "--tokenizer-revision", revision,
        "--policy-name", policy_name,
        "--greedy",
        "--split", split,
        "--output-dir", output_dir,
        "--seeds", "42",
        "--batch-size", "16",
        "--max-new-tokens", "128",
    ]
    namespace = runpy.run_path(
        "/content/evaluate_sft_qwen3_calculator.py",
        run_name="__main__",
    )
    del namespace
    gc.collect()
    torch.cuda.empty_cache()
print("POLICY_OOD_COMPLETE", policy_name)
PY
)
  printf '%s\n' "$python_code" |
    colab exec -s "$SESSION" --timeout 3600

  for split in "${SPLITS[@]}"; do
    remote_dir="$REMOTE_ROOT/$policy_name/$split"
    local_dir="$LOCAL_ROOT/$policy_name/$split"
    mkdir -p "$local_dir/42"
    colab ls -s "$SESSION" "$remote_dir"
    for file in config.json runtime.json aggregate_metrics.json; do
      colab download -s "$SESSION" \
        "$remote_dir/$file" \
        "$local_dir/$file"
    done
    for file in metrics.json predictions.jsonl; do
      colab download -s "$SESSION" \
        "$remote_dir/42/$file" \
        "$local_dir/42/$file"
    done
  done
done

colab log -s "$SESSION" \
  -o "$LOCAL_ROOT/colab-session.ipynb"
colab stop -s "$SESSION"
trap - EXIT
colab sessions
echo "OOD_SWEEP_COMPLETE $LOCAL_ROOT"
