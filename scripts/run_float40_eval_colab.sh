#!/usr/bin/env bash
set -euo pipefail

SESSION="${COLAB_SESSION:-calc-float40-eval-a100-20260724}"
LOCAL_ROOT="outputs/evaluations/qwen3-calc-float-mix-sweep-a100-20260724"
REMOTE_ROOT="/content/experiments/qwen3-calc-float40-eval"
DATA_ARCHIVE="/tmp/calculator_qwen3_float20-20260724.tar.gz"
SFT_REVISION="$(python -c 'import json; print(json.load(open("outputs/evaluations/qwen3-calc-float-mix-sweep-a100-20260724/float40/sft_hub_revision.json"))["revision"])')"
GRPO_REVISION="$(python -c 'import json; print(json.load(open("outputs/evaluations/qwen3-calc-float-mix-sweep-a100-20260724/float40/grpo_hub_revision.json"))["revision"])')"

stop_session() {
  colab stop -s "$SESSION" >/dev/null 2>&1 || true
}
trap stop_session EXIT

[[ "$SFT_REVISION" =~ ^[0-9a-f]{40}$ ]]
[[ "$GRPO_REVISION" =~ ^[0-9a-f]{40}$ ]]
[[ -f "$DATA_ARCHIVE" && -f ".env" ]]

colab sessions
colab new -s "$SESSION" --gpu A100
colab status -s "$SESSION"
printf '%s\n' \
  "from pathlib import Path; Path('/content/data').mkdir(parents=True, exist_ok=True); Path('$REMOTE_ROOT').mkdir(parents=True, exist_ok=True)" |
  colab exec -s "$SESSION" --timeout 60
colab upload -s "$SESSION" "$DATA_ARCHIVE" "/content/calculator_qwen3_float20.tar.gz"
colab upload -s "$SESSION" "scripts/calculator_semantics.py" "/content/calculator_semantics.py"
colab upload -s "$SESSION" "scripts/evaluate_sft_qwen3_calculator.py" "/content/evaluate_sft_qwen3_calculator.py"
python_code=$(cat <<'PY'
import tarfile
from pathlib import Path

with tarfile.open("/content/calculator_qwen3_float20.tar.gz") as archive:
    archive.extractall("/content/data", filter="data")
source = Path("/content/data/calculator_qwen3_float20")
target = Path("/content/data/calculator_qwen3")
source.rename(target)
assert (target / "dataset_dict.json").is_file()
print("EVAL_DATASET_READY")
PY
)
printf '%s\n' "$python_code" | colab exec -s "$SESSION" --timeout 180
colab install -s "$SESSION" \
  "transformers==5.14.1" \
  "peft==0.19.1" \
  "datasets==5.0.0" \
  "accelerate==1.14.0" \
  "torchao==0.17.0"
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
policies = (
    (
        "sft-float40",
        "tripathysagar/qwen3-0.6b-calc-sft200-float40-a100-20260724",
        "$SFT_REVISION",
    ),
    (
        "grpo-float40-strict5",
        "tripathysagar/qwen3-0.6b-calc-grpo600-float40-strict5-a100-20260724",
        "$GRPO_REVISION",
    ),
)
for policy, repo, revision in policies:
    for split in ("eval", "ood_length", "ood_negative", "ood_float"):
        sys.argv = [
            "evaluate_sft_qwen3_calculator.py",
            "--adapter", repo,
            "--adapter-revision", revision,
            "--tokenizer", repo,
            "--tokenizer-revision", revision,
            "--policy-name", policy,
            "--greedy",
            "--dataset", "/content/data/calculator_qwen3",
            "--split", split,
            "--output-dir", "$REMOTE_ROOT/" + policy + "/" + split,
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
    print("POLICY_EVAL_COMPLETE", policy)
print("FLOAT40_EVAL_COMPLETE")
PY
)
printf '%s\n' "$python_code" |
  colab exec -s "$SESSION" --timeout 7200 | tee "$LOCAL_ROOT/float40-eval.log"

python_code=$(cat <<PY
import shutil
from pathlib import Path
root = Path("$REMOTE_ROOT")
assert root.is_dir()
print("ARCHIVE_READY", shutil.make_archive("/content/float40-evaluation", "zip", root_dir=root))
PY
)
printf '%s\n' "$python_code" | colab exec -s "$SESSION" --timeout 300
colab download -s "$SESSION" \
  "/content/float40-evaluation.zip" \
  "$LOCAL_ROOT/float40-evaluation.zip"
colab log -s "$SESSION" -o "$LOCAL_ROOT/colab-session-float40-eval.ipynb"
colab stop -s "$SESSION"
trap - EXIT
colab sessions
echo "FLOAT40_EVAL_RECOVERY_COMPLETE"
