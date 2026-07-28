
I want to generate a dataset of 1000 problems for a calculator agent. The problems should be in the following format:

```md
[system]
You are a calculator agent. You cannot do arithmetic yourself — you must
call the `calculator` tool for every operation. Respect standard operator
precedence (multiplication before addition/subtraction).

[user]
Solve: 2*3 + 3*8

[assistant]
<think>
Precedence means both multiplications resolve before the addition.
I'll start with 2*3.
</think>
<tool_call>{"name": "calculator", "arguments": {"op": "*", "a": 2, "b": 3}}</tool_call>

[tool]
{"result": 6}

[assistant]
<think>
Now the second term, 3*8.
</think>
<tool_call>{"name": "calculator", "arguments": {"op": "*", "a": 3, "b": 8}}</tool_call>

[tool]
{"result": 24}

[assistant]
<think>
Both terms resolved: 6 and 24. Combine with addition.
</think>
<tool_call>{"name": "calculator", "arguments": {"op": "+", "a": 6, "b": 24}}</tool_call>

[tool]
{"result": 30}

[assistant]
<think>
Final result matches the combined terms.
</think>
The answer is 30.
```


TODO

- [x] Expression generator: no forced parens, `naive_left_to_right_answer != final_answer` filter kept, dedup kept
- [x] Think-text templates (small, mechanical, keyed to step type) + tool-call generator, assembled into the full multi-turn trajectory per the example format
- [x] Verify the exact message-dict schema against Qwen3's real chat template output, then serialize all 1,000 records as that schema
- [x] Add `tier`, `num_terms`, and `max_factors_in_a_term` metadata so dataset splits do not require parsing expression strings
- [x] Add deterministic, tier-stratified training, evaluation, and test splits
- [x] Refactor datagen to produce a native Hugging Face `DatasetDict`, persist it
  with `save_to_disk`, retain opt-in legacy JSONL export, and support
  optional Hub publishing
- [x] Generate and validate the on-disk HF dataset: `800` train, `100` eval,
  and `100` test rows
- [x] Add three isolated 100-row OOD splits for expression length, negative
  operands, and floating-point operands
- [x] Verify deterministic OOD generation, Qwen3 chat-template compatibility,
  and execution-grounded semantic replay for every OOD reference trajectory
- [x] Migrate active notebooks and scripts to `load_from_disk`, make JSONL
  export opt-in, and remove obsolete data JSONLs

## Implementation

- Generator: `scripts/generate_calculator_dataset.py`
- Hugging Face dataset: `data/calculator_qwen3/` (1,300-row `DatasetDict`)
- In-distribution splits: `train` (800), `eval` (100), `test` (100)
- OOD splits: `ood_length` (100), `ood_negative` (100), `ood_float` (100)
- Default tokenizer used for validation: `Qwen/Qwen3-0.6B`

Regenerate the dataset with:

```bash
python scripts/generate_calculator_dataset.py
```

Each dataset row contains `id`, `expression`, structured `messages`, the
calculator's JSON-schema `tools` definition, and validation `metadata`. The
generator's primary output is a Hugging Face `DatasetDict`, loadable with
`datasets.load_from_disk("data/calculator_qwen3")`. Legacy JSONL export is
disabled by default; use `--export-jsonl` only when an external consumer
requires it, or `--hub-repo owner/name` to publish the dataset.
Difficulty tiers use operation count: 2 is `easy`, 3–4 is `medium`, and 5 is
`hard`; the structural fields remain available for custom split policies.
The default 80/10/10 splits are deterministic, disjoint, and stratified by
difficulty tier. Each row's metadata includes its `split` assignment.

### OOD split contract

OOD splits are generated separately from the 1,000 in-distribution rows and
are excluded from SFT and GRPO training:

| Split | Shift held out from training | Constraint | Rows |
|---|---|---|---:|
| `ood_length` | Longer reasoning chains | 6–8 operations; positive integer operands | 100 |
| `ood_negative` | Signed operands | 2–5 operations; at least one source operand in `[-20, -2]` | 100 |
| `ood_float` | Non-integer arithmetic | 2–5 operations; positive half-integer operands and at least one fractional value | 100 |

All OOD expressions remain precedence-sensitive, contain both multiplication
and addition/subtraction, differ under naive left-to-right evaluation, and keep
all reference intermediate values within `±100,000`. OOD rows include
`metadata.ood_type` and `metadata.source_operands`. The float split uses JSON
Schema `number` operands; the integer splits retain `integer`.

Keep these splits isolated in reports. A combined OOD score would hide whether
failure comes from sequence length, signs, or numeric representation.


---

## Evaluate Qwen3-0.6B on the dataset

- [x] Create `notebooks/evaluate_qwen3_calculator.ipynb`
- [x] Load `data/calculator_qwen3/` and select the held-out HF `test` split
- [x] Apply the real `Qwen/Qwen3-0.6B` tokenizer and tool-aware chat template
- [x] Implement the calculator tool loop and trajectory-level metrics
- [x] Add GPU-efficient dynamic batching:
  - left-pad decoder-only prompts
  - batch up to 16 active trajectories per generation
  - bucket prompts by length to reduce padding waste
  - re-batch unfinished trajectories after each tool round
  - disable thinking mode and cap turns at 128 tokens for the fast benchmark
- [x] Run the optimized evaluation on a Colab T4
- [x] Download `metrics.json`, `predictions.jsonl`, and the executed notebook
- [x] Record overall and per-tier evaluation results
- [x] Fix observed evaluator issues:
  - parse all observed final-answer formats
  - separate tool-call validity from completion parsing
  - validate alternative legal operation orders semantically
  - stop generation at the first tool call
  - bucket prompts by token count
  - rename cumulative per-row completion timing
  - record model/runtime/decoding reproducibility metadata
  - checkpoint completed examples with configuration-safe resume

### Strict one-call benchmark results (100 held-out examples)

- Protocol: one tool call → execute → append result → generate again
- Evaluation wall time: 41.0 seconds (2.44 examples/second)
- Used the calculator tool: 100%
- Schema-valid tool use: 100%
- Parseable completion: 100%
- Semantically valid partial trace: 17%
- Semantically complete trace: 0%
- Correct tool-call count: 0%
- Final-answer accuracy: 1%
- Task success / exact reference trace: 0%
- Final-answer accuracy by tier: easy 0%, medium 0%, hard 2.78%

### Non-thinking result sanity checks

- Tier sizes are uneven: easy `11`, medium `53`, and hard `36`.
- The hard-tier result is exactly `1/36 = 2.78%`; it is one lucky match, not
  evidence that harder examples are easier for the model.
- The only answer match is `calculator_0884`:
  - expression: `14 - 18*7 + 18*5 + 18`
  - reference reduction: `14 - 126 + 90 + 18 = -4`
  - model call: `14 - 18 = -4`, followed by the final response `-4`
- The model therefore guessed the correct final number through an
  invalid-precedence operation. The scorer correctly records
  `answer_correct=true`, `semantic_trace_valid=false`,
  `semantic_trace_complete=false`, and `task_success=false`.
- Report the hard-tier result as `1/36 (2.78%)` and treat it as noise.

Artifacts:

- `outputs/notebooks/evaluate_qwen3_calculator_output.ipynb`
- `outputs/evaluations/metrics.json`
- `outputs/evaluations/predictions.jsonl`

---

## Evaluate Qwen3-0.6B in thinking mode

- [x] Create `notebooks/evaluate_qwen3_calculator_thinking.ipynb` without replacing the
  deterministic non-thinking benchmark
- [x] Enable Qwen3 thinking mode and use its recommended decoding configuration:
  - `enable_thinking=True`
  - `do_sample=True`
  - temperature `0.6`, top-p `0.95`, and top-k `20`
  - maximum `512` new tokens per generation
- [x] Preserve the strict protocol: one tool call → execute → append result →
  generate again
- [x] Guard against unbounded thinking:
  - stop generation at the first complete `</tool_call>`
  - detect generations that exhaust the token budget without a complete call
  - record `thinking_budget_exhausted` instead of retrying in non-thinking mode
  - retain the six-round trajectory limit
- [x] Record per-turn generated-token counts plus aggregate thinking-budget
  exhaustion rate
- [x] Disable checkpoint resume for sampled runs unless RNG state is restored
- [x] Run three fixed seeds
- [x] Aggregate mean and per-seed answer accuracy, semantic task success,
  tool-call validity, token usage, and exhaustion rate
- [x] Run the thinking benchmark on a Colab L4 and save the executed notebook
- [ ] Recover the remaining artifacts after Colab reclaimed the runtime:
  - seed `20260721` predictions
  - per-seed metrics JSON files
  - aggregate metrics JSON

### Thinking benchmark results (mean of 3 seeds, 100 held-out examples each)

- Final-answer accuracy: 26.33% (seed range: 21%–29%)
- Used the calculator tool: 56.33%
- Schema-valid tool use: 55.67%
- Parseable completion: 27.67%
- Semantically valid partial trace: 22.33%
- Semantically complete trace: 1.67%
- Correct tool-call count: 3.67%
- Task success: 1% (seed range: 0%–2%)
- Thinking-budget exhaustion: 56.67%
- Mean generated tokens per example: 860.6
- P95 generated tokens per turn: 512 (the configured limit)

### Quality finding: thinking versus non-thinking

- Thinking substantially improves raw answer accuracy from 1% to 26.33%.
- It does **not** make Qwen3-0.6B a reliable calculator agent: policy-aware task
  success only improves from 0% to 1%.
- Non-thinking mode always emits schema-valid tool calls, but usually chooses an
  invalid operation order or fails to continue the full reduction.
- Thinking mode often computes the correct answer internally instead of following
  the required calculator-tool trajectory; tool use drops from 100% to 56.33%.
- More than half of thinking trajectories exhaust the 512-token generation
  budget, making thinking roughly ten times slower while remaining unreliable.
- Conclusion: use SFT to teach precedence-aware, one-call-per-turn tool behavior.
  Treat semantic task success as the primary metric and raw answer accuracy as a
  secondary solver-quality metric.

Artifacts:

- `outputs/notebooks/evaluate_qwen3_calculator_thinking_output.ipynb` (all three seeds and aggregate
  printed metrics)
- `outputs/evaluations/thinking/20260719/predictions.jsonl`
- `outputs/evaluations/thinking/20260720/predictions.jsonl`

---

## SFT Qwen3-0.6B on calculator tool+thinking trajectories (Colab)

Goal: teach the *format* of precedence-aware think → one tool call per turn,
using a **small SFT slice** of train. Leave most train prompts unseen by SFT
so later **GRPO** can optimize on non-memorized examples.

Data is TRL-ready (`messages` + `tools`). Split policy for post-training:

| Slice | Size | Use |
|---|---:|---|
| SFT subset | **200** from train | supervised fine-tune only |
| GRPO pool | remaining **~600** train | reserved — do not SFT on these |
| eval / test | 100 / 100 | untouched during SFT (no trainer eval) |

Never SFT on eval, test, or the GRPO pool.

### Consistency locks (do these in order)

1. **Freeze the seed** — `SEED = 42` for SFT-200/GRPO-600 carve, `SFTConfig.seed`,
   and `data_seed`. After the id lists are committed, do not change the seed.
2. **Pin the LoRA weights** — after the successful full SFT run, treat that
   adapter as immutable. Record Hub `repo_id` + **commit revision** (preferred)
   so GRPO always loads the same weights, not “latest on Drive.”

### Artifact policy: Hub primary, local disk for speed, Drive optional

| Path | Role |
|---|---|
| `/content/...` (local Colab disk) | Fast train + epoch checkpoints during the run |
| **Hugging Face Hub** | Primary durable store for final LoRA + `sft_run_config.json` (faster/more reliable than Drive FUSE for small adapters) |
| Google Drive | Optional backup only; do not rely on mid-run saves to Drive |

Do **not** `push_to_hub` every epoch — save locally, push **once** at end of the
full run. Overfit adapters stay local (or a throwaway Hub tag); only pin the
**full** run revision for GRPO.

### Experiment tracking: Trackio

Use Trackio through the native Transformers integration. This records training
health only; because SFT intentionally has no eval dataset, there will be no
`eval_loss` or task-quality metrics.

- [x] Install/upgrade Trackio together with TRL, Transformers, PEFT, Datasets,
  Accelerate, and TorchAO so pip resolves one compatible stack
- [x] Log to one Trackio project, with separate immutable run names:
  - `qwen3-calc-sft200-overfit-seed42`
  - `qwen3-calc-sft200-full-seed42`
- [x] Set `report_to="trackio"`, `project="calc-rlvr-sft"`, `run_name=...`
- [x] Validate injected `HF_TOKEN` without printing it:
  - identity: `tripathysagar`
  - private model creation/upload: passed
  - private Bucket creation/read/write: passed
- [x] Explicitly create and pass private Bucket
  `tripathysagar/calc-rlvr-sft-bucket` (auto-derived Bucket previously returned
  `404 xet-write-token`)
- [x] Confirm trainer/Trackio callback captures: `loss`, `learning_rate`, `grad_norm`,
  `epoch`, `mean_token_accuracy`, `entropy`, and final runtime/throughput
- [x] Record the Trackio project/Space URL and run name in
  `sft_run_config.json`
- [x] Inspect full run: all logged numeric metrics finite; loss decreased
  `0.968647 → 0.017859` over 75 optimizer steps
- [x] Fix Trackio persistence and recover both completed histories:
  - overfit: 9 Trainer history records uploaded
  - full: 16 Trainer history records uploaded
  - verified three Trackio inbox objects in the private Bucket (including probe)
- [ ] Optional dashboard: private Gradio Space creation returned `402 Payment
  Required`; Hugging Face PRO is required. Keep metrics private in the Bucket,
  or explicitly accept a public static Space.
- [x] Preserve `trainer_state.json` and `sft_run_config.json`; Trackio is
  observability, not the authoritative checkpoint/config store

### Stage A — carve the 200 and set up plumbing

- [x] Create `notebooks/sft_qwen3_calculator.ipynb` for Colab (T4 or L4)
- [x] Upgrade `trl`, `transformers`, `peft`, `datasets`, `accelerate`,
  `trackio`, and Colab's stale `torchao` together; record resolved versions in
  `sft_run_config.json`
  - resolved: TRL `1.8.0`, Transformers `5.14.1`, PEFT `0.19.1`, Datasets
    `5.0.0`, Accelerate `1.14.0`, Trackio `0.31.5`, TorchAO `0.17.0`
- [x] Notebook carves tier-stratified 200 (`SEED=42`) from the HF `train` split
  and writes:
  - `data/calculator_qwen3_sft200_ids.txt`
  - `data/calculator_qwen3_grpo600_ids.txt`
- [x] Run on Colab: upload `data/calculator_qwen3/`, execute split cell, and
  commit the two ID artifacts locally (seed now frozen)
- [x] Update notebook: `output_dir` on **local** `/content/...` (not Drive);
  read `HF_TOKEN`, `RUN_MODE`, and optional Hub/Trackio overrides from the
  environment injected by the Colab CLI `.env` loader
- [x] Before each CLI run: upload `.env` to `/content/.env`, execute
  `.cursor/skills/colab-cli/scripts/load-env.py`, and confirm the loader deletes
  the remote plaintext file (never print secret names or values)
- [x] Bundle lightweight artifacts under `/content`; retrieve them with
  `colab download` rather than browser-only `google.colab.files.download`
- [x] Load with `datasets`; keep `messages` + `tools` only
- [x] Smoke-check: `apply_chat_template(..., tools=…)` renders think + tool calls
- [x] Measure max tokenized length on the **200 only**; assert `<= 1024`
- [x] Wire `SFTTrainer` with finalized configs (**no `eval_dataset`**);
  use FP16 on T4 (compute capability 7.5), BF16 only on Ampere+

### Stage B — tiny overfit on a handful from the 200

- [x] Colab T4: `RUN_MODE = "overfit"` (`OVERFIT_ROWS = 16`, 20 epochs)
  - 40 optimizer steps; loss `0.910898 → 0.010669`
- [x] Spot-check generation: four valid one-call turns (`2*6`, `14*2`,
  `12+28`, `40+3`) followed by final answer `43`
- [x] Gate passed: precedence, tool schema, continuation, and final answer correct
- [x] Do **not** pin/publish overfit weights as the GRPO base

### Stage C — SFT on the 200 only

- [x] Colab T4: `RUN_MODE = "full"` on all **200** SFT rows (`SEED=42`)
- [x] Log train metrics only — 75 steps / 3 epochs; **no eval loop**
- [x] Save adapter + `sft_run_config.json` to local `/content/.../full/`
- [x] Push final adapter once from a clean post-Trackio kernel:
  - repo: `tripathysagar/qwen3-0.6b-calc-sft200`
  - pinned revision: `86db06c14d91acc734e89a45bb5ca3ec4e1ee8f3`
  - local manifest: `outputs/notebooks/qwen3-calc-sft200-full-hub_revision.json`
- [ ] Optional: copy the same folder to Drive as backup
- [x] Downstream rule: GRPO loads `PeftModel.from_pretrained(HUB_REPO_ID,
  revision=<pinned>)` only

### Post-SFT smoke finding

- Full-run checkpoint did **not** pass the one-example semantic gate despite low
  token loss: after `2*6=12` and `14*2=28`, it incorrectly called `12+3=15`
  and answered `15`, omitting `28`.
- Do not infer agent correctness from SFT loss. Keep the adapter as the fixed
  GRPO initialization, but treat semantic tool-loop success as unresolved.

### Post-SFT held-out eval result

- [x] Loaded base revision `c1899de289a04d12100db370d81485cdf75e47ca`
  and adapter revision `86db06c14d91acc734e89a45bb5ca3ec4e1ee8f3`.
- [x] Safely merged and unloaded LoRA before inference; evaluated only the
  100-row `eval` split. The test split remains untouched.
- [x] Controlled A100 comparison held hardware, FP16 inference stack, batch
  size, tokenizer/template revision, eval rows, parser, and scorer fixed.
  Rendered initial prompt hash matched across policies:
  `a30c96107481d46b93caa0429a5c557bf848c1b5a107cf4c54a0f586c47153de`.
- [x] Primary greedy result:
  - semantic `task_success`: base **3%** → SFT **82%** (**+79 points**)
  - answer accuracy: base **34%** → SFT **85%**
  - valid tool use: base **53%** → SFT **100%**
  - thinking-budget exhaustion: base **52%** → SFT **0%**
- [x] Three-seed sampled robustness result (`20260719`, `20260720`, `20260721`):
  - semantic `task_success`: base **1.33%** → SFT **81.67%**
    (**+80.34 points**)
  - answer accuracy: base **25.33%** → SFT **83.67%**
  - valid tool use: base **53.33%** → SFT **99.67%**
  - thinking-budget exhaustion: base **62%** → SFT **0%**
- Verdict: SFT produces a large, hardware-controlled behavioral improvement and
  removes the base model's dominant failure mode of unbounded thinking.
- Comparison summary and complete prediction archives:
  `outputs/evaluations/qwen3-controlled-*`.

### Thinking-budget calibration (before GRPO/production)

- [x] On one fixed GPU and inference stack, run the merged SFT policy with
  greedy decoding at per-turn caps **128, 256, and 512** using the same eval
  rows and rendered-prompt hash
- [x] Record semantic `task_success`, valid tool use, budget exhaustion,
  generated tokens per turn/example, latency, and tier breakdown for each cap
- [x] Choose the smallest cap whose task success is within **1–2 points** of the
  best result, exhaustion is below **1%**, and tool validity remains stable
- [x] Use deterministic greedy decoding for this calibration; do **not** run a
  sampling-seed sweep
- [x] Keep early stopping at `</tool_call>`/EOS and retain the six-call/six-round
  safety limits
- Selected cap: **128 tokens per assistant turn**. All three caps produced
  byte-identical predictions: **82%** semantic task success, **100%** valid tool
  use, **0%** exhaustion, mean **47.831** and P95 **59** generated tokens per
  turn. Summary: `outputs/evaluations/qwen3-sft-a100-greedy-budget-comparison.json`.
- [x] For GRPO, reward correctness and complete semantic traces first; apply a
  small token-length penalty only to successful trajectories and penalize budget
  exhaustion separately. Final reward weights and failure IDs are recorded in
  `outputs/evaluations/qwen3-sft-a100-greedy-failure-analysis.json`.

### SFT handoff cleanup

- [x] Set `scripts/evaluate_sft_qwen3_calculator.py` default inference cap to the selected
  **128 tokens per assistant turn**
- [x] Recover and validate the authoritative full-run artifacts under
  `outputs/notebooks/qwen3-calc-sft200-full-artifacts/`:
  - `sft_run_config.json` from pinned Hub revision
    `86db06c14d91acc734e89a45bb5ca3ec4e1ee8f3`
  - original `trainer_state.json` (75 steps / 3 epochs) from the preserved full
    artifact ZIP
  - `training_args.bin` from the pinned Hub revision
  - `full_trackio_history.jsonl` (16 records) from the private Bucket
- [x] Classify all 18 residual greedy failures:
  - all are hard-tier examples
  - 8 omitted operations, 6 precedence violations, 3 wrong operands/operators,
    and 1 duplicated operation
  - 3 produced the correct answer through an invalid trace; therefore standalone
    answer correctness receives zero GRPO reward
- [x] Freeze the first GRPO reward as the discrete table below. Do not mix in
  partial-progress, schema, call-count, or length shaping until the unshaped
  baseline has been measured; standalone answer correctness is never rewarded

### Post-SFT handoff status

- [x] Load the SFT policy from its pinned Hub revision for GRPO.
- [x] Train GRPO on the reserved 600 rows without eval/test contamination.
- [x] Keep the test split untouched through SFT, GRPO, and checkpoint
  selection.

### Finalized configs (from TRL SFT docs + measured lengths)

Measured on full train with Qwen3 chat template + `tools=`: max **674** tokens
(P50 574). Default `max_length=1024` is enough — no truncation risk.

Doc defaults we keep or override:

| Knob | TRL default | Our choice | Why |
|---|---|---|---|
| `learning_rate` | `2e-5` (full FT) | **`1e-4`** | TRL LoRA tip ≈1e-4 |
| `assistant_only_loss` | `False` | **`True`** | only learn assistant think/tool/answer |
| `packing` | `False` | **`False`** | keep tool-turn boundaries clean |
| `max_length` | `1024` | **`1024`** | covers max 674 with headroom |
| `loss_type` | `chunked_nll` | **`chunked_nll`** | lower peak memory |
| `gradient_checkpointing` | `True` | **`True`** | Colab VRAM |
| `bf16` | `True` if no fp16 | **L4: bf16 / T4: fp16** | T4 has no bf16 |
| `logging_steps` | `10` | **`5`** | 200-row run is short |
| `do_eval` | `False` | **`False`** | no eval during SFT |
| `shuffle_dataset` | `False` | **`True`** | better mixing on 200 rows |
| `report_to` | `"none"` | **`"trackio"`** | persist training curves |

```python
import os

from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

SEED = 42  # also used to carve SFT-200 vs GRPO-600

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# Colab: bf16 on L4; fp16 on T4. Train to local /content (fast); Hub push once at end.
HUB_REPO_ID = "YOUR_USER/qwen3-0.6b-calc-sft200"  # set before full run
TRACKIO_SPACE_ID = "YOUR_USER/calc-rlvr-sft"
TRACKIO_BUCKET_ID = "YOUR_USER/calc-rlvr-sft-bucket"
RUN_MODE = os.environ.get("RUN_MODE", "overfit")
sft_config = SFTConfig(
    output_dir="/content/sft_qwen3_calc_200",  # local disk during train
    seed=SEED,
    data_seed=SEED,
    # data / loss
    max_length=1024,
    packing=False,
    assistant_only_loss=True,
    loss_type="chunked_nll",
    shuffle_dataset=True,
    # train schedule (200 rows → ~25 opts/epoch → ~75 total)
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # effective batch = 8
    learning_rate=1e-4,
    warmup_steps=4,  # recompute as ~5% of optimizer steps for overfit/full mode
    lr_scheduler_type="cosine",
    weight_decay=0.0,
    max_grad_norm=1.0,
    # precision / memory
    bf16=True,   # T4: bf16=False, fp16=True
    gradient_checkpointing=True,
    # logging / checkpoint (no eval)
    do_eval=False,
    eval_strategy="no",
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="trackio",
    project="calc-rlvr-sft",
    run_name=f"qwen3-calc-sft200-{RUN_MODE}-seed{SEED}",
    trackio_space_id=TRACKIO_SPACE_ID,
    trackio_bucket_id=TRACKIO_BUCKET_ID,
    trackio_static_space_id=False,
    hub_private_repo=True,
    model_init_kwargs={"dtype": "bfloat16"},  # T4: "float16"
)

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    args=sft_config,
    train_dataset=sft200_dataset,  # messages + tools only
    peft_config=peft_config,
)
trainer.train()
# After full run only:
# trainer.model.push_to_hub(HUB_REPO_ID)
# tokenizer.push_to_hub(HUB_REPO_ID)
# then record the returned commit revision and freeze it for GRPO
```

Overfit sanity (Stage B): same config but `num_train_epochs=20` (or until
loss flats), dataset = 8–16 rows from SFT-200. Do not Hub-pin overfit.

### Design note

SFT on all 800 would teach the trajectories but risk the policy memorizing
prompts you later want GRPO to explore. SFT-200 teaches the tool+think
schema; GRPO-600 stays fresh for reward-driven search. Seed freezes the
data split; Hub revision freezes the LoRA weights.


---

## GRPO Training

### Reward Design

#### 1. Completed: terminal-only reward design and results

| Trajectory | Final Answer | Reward |
|---|---|---|
| Valid | Correct | **+2.0** |
| Invalid | Correct (coincidental) | **0** |
| Valid | Incorrect | **0** |
| Invalid | Incorrect | **-1.0** |
| — | No answer stated (budget/rounds exhausted) | **-0.1** |

Results with this reward:

- Selected gen-4 adapter: greedy eval task success `0.95`, then one-time
  held-out test task success `0.98`.
- Batch-16 gen-4 candidate: greedy eval task success `0.96`.
- Tool syntax and parsing were `1.00`, with no budget exhaustion.
- The generation-count ablation found no benefit from more rollouts at seed
  `42`: gen-4 `0.96`, gen-8 `0.95`, and gen-16 `0.93` eval task success.

#### 2. Completed experiment: bounded legal-progress shaping (rejected)

This completed run used a normalized shaping term:

`progress_bonus = 0.1 * legal_call_count / expected_call_count`

| Event | Added Reward | Guard |
|---|---:|---|
| Each newly accepted grounded call | `+0.1 / expected_call_count` | total trajectory bonus capped at `+0.1` |
| Duplicate, illegal, or ungrounded call | `0` | never farm repeated calls |
| Parseable final answer | terminal reward from design 1 plus earned bonus | complete trace still determines validity |
| No parseable final answer | no progress bonus; terminal reward remains `-0.1` | abstention cannot become profitable |

- [x] Implement the shaping term behind an explicit config flag; keep the
  terminal-only scorer as the default and preserve reward unit tests.
- [x] Add tests for normalized reward across easy/medium/hard call counts,
  incomplete legal prefixes, duplicate calls, illegal calls, and no-answer
  trajectories.
- [x] Run a seed-42 exact-config smoke using gen `4`, batch `16`, and the same
  SFT initialization; reject it if reward farming, over-calling, no-answer
  exploitation, or zero-variance regression appears.
- [x] If the smoke passes, run the same 600 prompt exposures and 75 optimizer
  updates as the terminal-only gen-4 baseline.
- [x] Evaluate greedily on the identical 100-row eval split and compare task
  success, hard-tier success, invalid-trace rate, correct-but-invalid rate,
  call-count errors, KL, generated tokens, and wall time.
- [x] Keep the held-out test split untouched and do not replace the selected
  adapter unless the shaped candidate clearly passes the eval gate.

The smoke passed its safety gates: no over-calling, no no-answer exploitation,
nonzero reward variance, and a mean progress bonus of `0.0806`. The full run
used seed `42`, gen `4`, batch `16`, gradient accumulation `2`,
`steps_per_generation=2`, and colocated vLLM memory utilization `0.25`. It
completed 600 prompt exposures and 75 optimizer updates in `455.6` seconds.
The candidate is pinned at
`tripathysagar/qwen3-0.6b-calc-grpo600-progress01-b16@cd65704e2e94ea1f09baf0a6bd2a78426a0a86b4`.

Controlled comparison against the exact-config terminal-only batch-16 gen-4
baseline:

| Metric | Terminal only | Progress `0.1` | Delta |
|---|---:|---:|---:|
| Eval task success | `0.96` | `0.91` | `-0.05` |
| Hard-tier task success | `0.9167` | `0.8056` | `-0.1111` |
| Invalid-trace rate | `0.04` | `0.09` | `+0.05` |
| Correct-but-invalid rate | `0.00` | `0.01` | `+0.01` |
| Call-count error rate | `0.03` | `0.04` | `+0.01` |
| Mean training KL | `0.00928` | `0.00755` | `-0.00173` |
| Training generated tokens | `1,210,291` | `1,215,397` | `+5,106` |
| Training wall time | `433.4 s` | `455.6 s` | `+22.2 s` |
| Eval generated tokens | `22,369` | `22,384` | `+15` |
| Eval wall time | `120.1 s` | `118.5 s` | `-1.6 s` |

Tool syntax, parsing, and budget exhaustion remained perfect, but semantic
trace validity regressed, especially on hard examples. Reject this shaping
candidate, leave the held-out test untouched, and retain the selected
terminal-only policy.

#### 3. Completed experiment: `+0.5` per legal call (rejected)

Test the more aggressive count-based shaping requested after experiment 2:

`progress_bonus = 0.5 * legal_call_count`

| Event | Added Reward | Guard |
|---|---:|---|
| Each newly accepted grounded call | `+0.5` | cumulative bonus is `0.5 * legal_call_count` |
| Duplicate, illegal, or ungrounded call | `0` | repeated calls do not increase `legal_call_count` |
| Parseable final answer | terminal reward from design 1 plus earned bonus | correct valid reward is `2 + progress_bonus` |
| No parseable final answer | no progress bonus; terminal reward remains `-0.1` | abstention cannot collect progress |

Unlike experiment 2, this bonus is not normalized by task length. A five-call
trajectory can earn `+2.5`, including when its final answer is wrong. Treat
reward farming, over-calling, and invalid-trace rate as explicit rejection
gates.

- [x] Implement the count-based mode behind a separate config variable and
  preserve terminal-only and normalized-reward behavior.
- [x] Run the exact-config seed-42 smoke with gen `4`, batch `16`, and the
  pinned SFT initialization.
- [x] If the smoke passes, train for 600 prompt exposures and 75 optimizer
  updates.
- [x] Evaluate greedily on the identical 100-row eval split; do not touch the
  held-out test split.
- [x] Compare with the terminal-only batch-16 baseline and keep the existing
  selected policy unless the candidate passes the eval gate.

The smoke had zero over-calling, zero no-answer trajectories, nonzero reward
variance, and mean progress bonus `1.4531`. The full run completed all 600
prompt exposures and 75 optimizer updates in `442.5` seconds. It used mean GPU
utilization `82.3%` and had only `746 MiB` free at peak memory. The candidate
is pinned at
`tripathysagar/qwen3-0.6b-calc-grpo600-progress05-b16@b2829eb96e9c23b7276f7435f2fe8df12b1f636d`.

Controlled comparison against the exact-config terminal-only batch-16 gen-4
baseline:

| Metric | Terminal only | Per-call `0.5` | Delta |
|---|---:|---:|---:|
| Eval task success | `0.96` | `0.95` | `-0.01` |
| Hard-tier task success | `0.9167` | `0.8889` | `-0.0278` |
| Invalid-trace rate | `0.04` | `0.05` | `+0.01` |
| Correct-but-invalid rate | `0.00` | `0.00` | `0.00` |
| Call-count error rate | `0.03` | `0.03` | `0.00` |
| Mean training KL | `0.00928` | `0.01141` | `+0.00213` |
| Training generated tokens | `1,210,291` | `1,210,992` | `+701` |
| Training wall time | `433.4 s` | `442.5 s` | `+9.1 s` |
| Eval generated tokens | `22,369` | `22,193` | `-176` |
| Eval wall time | `120.1 s` | `121.6 s` | `+1.4 s` |

Failure examples from the 100-row eval are below. Eval itself does not award
rewards; the last column is the reward each generated trajectory would receive
from the experiment-3 training scorer:

`shaped_reward = -1.0 + 0.5 * legal_call_count`

| ID | Expression | Failure | Legal / Total Calls | Shaped Reward |
|---|---|---|---:|---:|
| `calculator_0600` | `4 + 10 - 10 + 17*9` | reached the correct result `157`, then reused consumed `17` and returned `174` | `4 / 5` | `+1.0` |
| `calculator_0683` | `12 - 9*19 + 17*15*17` | mixed the two multiplication branches by subtracting `4335` instead of `171` | `3 / 5` | `+0.5` |
| `calculator_0778` | `7 + 16*13*18 + 13*6` | attached `6` to `3744` instead of evaluating the separate `13*6` branch | `2 / 5` | `0.0` |
| `calculator_0098` | `18 + 6*5 + 6 + 11 + 5` | reused the already consumed second `6`, producing one extra call and answer `76` instead of `70` | `3 / 6` | `+0.5` |
| `calculator_0188` | `12*10 + 11*4*14*4` | reached `2584`, then reused consumed `4` and returned `10336` | `4 / 6` | `+1.0` |

Generated action/final-answer strings, with the repetitive `<think>` text
omitted but every generated tool call preserved:

- `calculator_0600`:
  `*(17,9)->153 | +(4,10)->14 | -(14,10)->4 | +(4,153)->157 | +(157,17)->174 | "The answer is 174."`
- `calculator_0683`:
  `*(9,19)->171 | *(17,15)->255 | *(255,17)->4335 | -(12,4335)->-4323 | +(-4323,4335)->12 | "The answer is 12."`
- `calculator_0778`:
  `*(16,13)->208 | *(208,18)->3744 | *(3744,6)->22464 | +(7,22464)->22471 | +(22471,22464)->44935 | "The answer is 44935."`
- `calculator_0098`:
  `*(6,5)->30 | +(18,30)->48 | +(48,6)->54 | +(54,6)->60 | +(60,11)->71 | +(71,5)->76 | "The answer is 76."`
- `calculator_0188`:
  `*(12,10)->120 | *(11,4)->44 | *(44,14)->616 | *(616,4)->2464 | +(2464,120)->2584 | *(2584,4)->10336 | "The answer is 10336."`

All five had parseable answers and valid tool syntax, but all final answers and
semantic traces were wrong. Four of the five failures therefore received a
positive shaped reward despite failing. Under group-relative optimization this
can reinforce late-stage invalid trajectories when a rollout group contains
no correct completion.

`calculator_0188` also exposes a checker limitation: its fifth call,
`2464 + 120`, is mathematically equivalent to the expected `120 + 2464`, but
the current reducer rejects swapped operands even for commutative operations.
The following extra `2584 * 4` call still makes the trajectory invalid, so this
does not change its eval outcome; it does undercount legal progress by one.

Tool syntax, parsing, call-count accuracy, and budget behavior did not improve.
The candidate lost one overall eval success and one hard-tier success relative
to terminal-only reward. Reject it, leave the held-out test untouched, and
retain the selected terminal-only policy.

#### 4. Completed experiment: strict invalid-trajectory penalties (not selected)

Keep every invalid trajectory negative while retaining a small amount of
within-failure progress signal:

`progress = legal_call_count / expected_call_count`

| Scenario | Detection | Reward |
|---|---|---:|
| Valid trace, correct answer | fully reduced, exact call count, correct stated answer | `+2.0` |
| Valid trace, wrong stated answer | fully reduced and exact, but answer mismatch | `-0.5` |
| No parseable answer | no supported final-answer form | `-1.2` |
| Legal but incomplete | no illegal call, reduction unfinished | `-1.0 + 0.2 * progress` |
| Reused operand | operand occurrence was already consumed | `-1.0 + 0.2 * progress - 0.5` |
| Branch mismatch | operands do not form one legal unresolved operation | `-1.0 + 0.2 * progress - 0.5` |
| Precedence violation | addition/subtraction attempted while multiplication remains | `-1.0 + 0.2 * progress - 0.5` |
| Call after completion | expression was already fully reduced | `-1.0 + 0.2 * progress - 0.75` |
| Over-call | actual calls exceed expected calls | additional `-0.25` per extra call |
| Correct but invalid | correct stated answer with any invalid trace | use the applicable invalid reward; never forgive invalidity |

Apply an upper bound of `-0.5` to every invalid reward. Previously earned
progress never cancels an invalidity penalty, and no invalid trajectory can
receive zero or positive reward. This preserves ordering among failed rollouts
without allowing a late invalid trajectory to look successful.

Expected rewards for the five experiment-3 failures after accepting swapped
operands for commutative operations:

| ID | Classification | Proposed Reward |
|---|---|---:|
| `calculator_0600` | post-completion call plus one over-call | `-1.80` |
| `calculator_0683` | branch mismatch at progress `3/5` | `-1.38` |
| `calculator_0778` | branch mismatch at progress `2/5` | `-1.42` |
| `calculator_0098` | reused operand plus one over-call at progress `3/5` | `-1.63` |
| `calculator_0188` | post-completion call plus one over-call | `-1.80` |

- [x] Fix semantic reduction for commutative operations: accept `(a, b)` and
  `(b, a)` for addition and multiplication, while keeping subtraction ordered.
- [x] Extend the shared scorer with `first_illegal_call_index`,
  `first_illegal_reason`, `post_completion_call_count`, and `overcall_count`.
- [x] Distinguish `reused_operand`, `branch_mismatch`,
  `precedence_violation`, `post_completion_call`, and legal-but-incomplete
  traces without relying on exact reference-trace matching.
- [x] Implement this reward behind a new explicit mode; preserve terminal-only,
  normalized `0.1`, and per-call `0.5` modes for reproducibility.
- [x] Add unit tests for every matrix row, commutative operand reversal,
  repeated literals, multiple branches, extra calls after completion, and the
  five recorded failure trajectories.
- [x] Replay all 100 eval trajectories through the revised scorer and verify
  that valid outcomes do not change except for genuinely commutative traces.
- [x] Run a seed-42 gen-4/batch-16 smoke; reject on positive invalid rewards,
  no-answer exploitation, reward farming, or zero reward variance.
- [x] Only if the smoke passes, train 600 prompt exposures and 75 optimizer
  updates, then evaluate on the same 100-row eval split. Keep held-out test
  untouched until the candidate beats the terminal-only eval gate.

The first full attempt at vLLM memory utilization `0.25` failed from CUDA
fragmentation while requesting `9.28 GiB` with `9.19 GiB` free. A clean retry
used expandable CUDA segments and vLLM utilization `0.24`; all learning
hyperparameters remained identical. It completed 600 prompt exposures and 75
optimizer updates in `432.9` seconds with mean GPU utilization `82.3%` and
`9,736 MiB` free at peak memory. The adapter is pinned at
`tripathysagar/qwen3-0.6b-calc-grpo600-strict-b16@0cb74b7059cf87c48f7e137c2323b1d9c56df75b`.

The training safety objective worked: nonnegative-invalid rate was `0.00`
throughout training. Mean no-answer rate was `0.00375`, maximum was `0.0625`,
and greedy eval had no no-answer or budget-exhaustion cases.

Controlled eval comparison with the terminal-only batch-16 gen-4 baseline:

| Metric | Terminal only | Strict invalid | Delta |
|---|---:|---:|---:|
| Eval task success | `0.96` | `0.96` | `0.00` |
| Medium-tier task success | `0.9811` | `1.00` | `+0.0189` |
| Hard-tier task success | `0.9167` | `0.8889` | `-0.0278` |
| Invalid-trace rate | `0.04` | `0.04` | `0.00` |
| Correct-but-invalid rate | `0.00` | `0.00` | `0.00` |
| Call-count error rate | `0.03` | `0.01` | `-0.02` |
| Mean training KL | `0.00928` | `0.00751` | `-0.00177` |
| Training generated tokens | `1,210,291` | `1,210,472` | `+181` |
| Training wall time | `433.4 s` | `432.9 s` | `-0.5 s` |
| Eval generated tokens | `22,369` | `22,174` | `-195` |
| Eval wall time | `120.1 s` | `120.8 s` | `+0.7 s` |

The strict policy fixed `calculator_0188` and `calculator_0600`, while
`calculator_0098` and `calculator_0683` remained failures. It introduced two
hard-tier failures, `calculator_0431` and `calculator_0778`. Overall success
tied terminal-only, call-count accuracy improved, but hard-tier success
regressed. The candidate therefore does not pass the replacement gate: leave
held-out test untouched and retain the selected terminal-only policy.

#### 5. Completed experiment: strict penalties with a `+10` success spike (rejected)

Use the complete experiment-4 reward unchanged except for one row:

| Scenario | Detection | Experiment 4 | Experiment 5 |
|---|---|---:|---:|
| Valid trace, correct answer | fully reduced, exact call count, correct stated answer | `+2.0` | `+10.0` |
| Valid trace, wrong stated answer | fully reduced and exact, but answer mismatch | `-0.5` | `-0.5` |
| No parseable answer | no supported final-answer form | `-1.2` | `-1.2` |
| Legal but incomplete | no illegal call, reduction unfinished | `-1.0 + 0.2 * progress` | unchanged |
| Reused operand | operand occurrence was already consumed | `-1.0 + 0.2 * progress - 0.5` | unchanged |
| Branch mismatch | operands do not form one legal unresolved operation | `-1.0 + 0.2 * progress - 0.5` | unchanged |
| Precedence violation | addition/subtraction attempted while multiplication remains | `-1.0 + 0.2 * progress - 0.5` | unchanged |
| Call after completion | expression was already fully reduced | `-1.0 + 0.2 * progress - 0.75` | unchanged |
| Over-call | actual calls exceed expected calls | additional `-0.25` per extra call | unchanged |
| Correct but invalid | correct answer with any invalid trace | applicable invalid reward | unchanged |

This tests whether widening the valid-success margin improves hard examples.
Because GRPO normalizes rewards within each rollout group, changing `+2` to
`+10` may have little effect when a group contains only two outcome levels:
the normalized advantages can remain nearly identical. Groups containing
multiple invalid-reward levels can still change, so this must be measured
rather than assumed.

- [x] Add a configurable valid-correct reward value with default `2.0`; set it
  to `10.0` only for this experiment.
- [x] Preserve every experiment-4 invalid reward and diagnostic unchanged.
- [ ] Add tests proving that only valid-correct reward changes between strict
  `+2` and strict `+10` modes.
- [ ] Replay the existing 100-row eval trajectories through both scorers and
  compare raw and group-normalized rewards.
- [ ] Run the exact seed-42 gen-4/batch-16 smoke and inspect reward variance,
  normalized advantages, KL, clipping, invalid rate, and no-answer rate.
- [x] At user request, skip the preflight tests, replay, and smoke; run 600
  prompt exposures and 75 optimizer updates directly, then evaluate greedily
  on the identical 100-row eval split.
- [x] Keep held-out test untouched and retain terminal-only unless strict
  `+10` clearly beats the `0.96` eval gate without hard-tier regression.

The direct run used seed `42`, gen `4`, batch `16`, gradient accumulation `2`,
`steps_per_generation=2`, expandable CUDA segments, and colocated vLLM memory
utilization `0.24`. It completed in `447.5` seconds with mean GPU utilization
`81.0%` and `9,916 MiB` free at peak memory. The adapter is pinned at
`tripathysagar/qwen3-0.6b-calc-grpo600-strict10-b16@ba579a014ad44333b1b094330a945a06c7d84237`.

Complete 100-row greedy eval comparison:

| Metric | Strict `+2` | Strict `+10` | Delta |
|---|---:|---:|---:|
| Eval task success | `0.96` | `0.94` | `-0.02` |
| Medium-tier task success | `1.00` | `0.9811` | `-0.0189` |
| Hard-tier task success | `0.8889` | `0.8611` | `-0.0278` |
| Invalid-trace rate | `0.04` | `0.06` | `+0.02` |
| Correct-but-invalid rate | `0.00` | `0.00` | `0.00` |
| Call-count error rate | `0.01` | `0.05` | `+0.04` |
| Mean training KL | `0.00751` | `0.00747` | `-0.00004` |
| Training generated tokens | `1,210,472` | `1,212,141` | `+1,669` |
| Training wall time | `432.9 s` | `447.5 s` | `+14.6 s` |
| Eval generated tokens | `22,174` | `22,217` | `+43` |
| Eval wall time | `120.8 s` | `122.3 s` | `+1.5 s` |

The `+10` spike fixed `calculator_0683`, but `calculator_0098`,
`calculator_0431`, and `calculator_0778` remained failures. It introduced
`calculator_0426`, `calculator_0566`, and `calculator_0600`. Nonnegative
invalid reward remained `0.00`, but quality and call-count accuracy regressed.
This supports the expectation that group normalization largely removes the
benefit of a raw success-magnitude spike. Reject this candidate, leave held-out
test untouched, and retain the selected terminal-only policy.

#### 6. Completed experiment: execution-grounded binary reward (rejected)

Replace reference-trace matching with actual calculator execution and
occurrence-aware expression reduction. Let `N` be the number of binary
operators in the parsed expression. A successful trajectory must make exactly
`N` calculator calls because every accepted call consumes exactly one
unresolved operator.

For each emitted calculator call:

1. Execute the requested operation using the emitted arguments; never trust a
   model-stated intermediate result.
2. Match both operands to unresolved original operand occurrences or results
   produced by earlier accepted calls.
3. Verify that the call performs one currently legal operation in the parsed
   expression tree, then replace that operation with the actual tool result.
4. Preserve operand order for subtraction. Allow swapped operands for
   commutative addition and multiplication.
5. Reject reused operands, unrelated operands, skipped branches, calls after
   full reduction, and any call that cannot reduce the current expression
   state.

Do not require the model's call sequence to exactly match one reference trace.
Independent branches may be evaluated in any legal order.

| Scenario | Detection | Proposed reward |
|---|---|---:|
| Complete valid solution | exactly `N` calls; every call is an accepted reduction; expression reduces to the expected value; stated answer matches | `+2.0` |
| Wrong call count | fewer or more than `N` calls | `0.0` |
| Illegal reduction | reused, ungrounded, unrelated, or precedence-invalid operands/operator | `0.0` |
| Incomplete reduction | emitted calls are legal but unresolved operators remain | `0.0` |
| Wrong reduced value | final tool-derived root value differs from the expected answer | `0.0` |
| Wrong or missing stated answer | tool reduction is valid but final response is absent or mismatched | `0.0` |
| Correct answer with invalid trace | stated answer matches but any execution-grounded trace condition fails | `0.0` |
| Call after completion | another tool call occurs after the expression reaches one root value | `0.0` |

The total reward range is `0.0` to `2.0`. Final-answer correctness earns
nothing unless actual tool execution proves a complete legal reduction.

- [x] Implement the execution-grounded scorer behind a new reward mode without
  changing the selected terminal-only default.
- [x] Use the same scorer in GRPO and evaluation so reward and reported task
  success cannot disagree.
- [x] Unit-test alternate legal branch orders and swapped operands for `+` and
  `*`.
- [x] Unit-test repeated literal occurrences, skipped operations, reused
  operands, `add(expected_answer, 0)` shortcuts, under-calls, and post-completion
  over-calls.
- [x] Replay all recorded eval trajectories through the new scorer and inspect
  every label change before training.
- [x] Instrument grouped smoke and full-training rollouts and measure reward
  variance and
  zero-standard-deviation frequency.
- [x] Run the same seed-42/gen-4/batch-16 experiment and compare against the
  selected terminal-only policy on the identical 100-row greedy eval.
- [x] Reject the candidate if it does not exceed `0.96` task success without
  hard-tier, invalid-trace, or call-count regression.

The 10-step smoke showed that zero-variance groups were mostly already-solved:
mean all-success rate `0.5625`, all-failure rate `0.025`, mixed rate `0.4125`,
and hard all-failure rate `0.05`. This passed the smoke gate because failed
hard prompts rarely lost all learning signal.

The full run used seed `42`, gen `4`, batch `16`, gradient accumulation `2`,
`steps_per_generation=2`, expandable CUDA segments, and colocated vLLM memory
utilization `0.24`. It completed in `450.9` seconds with mean GPU utilization
`80.4%` and `9,316 MiB` free at peak memory. The adapter is pinned at
`tripathysagar/qwen3-0.6b-calc-grpo600-binary-b16@ac23397573b214d2a6c8d4c0bb8167201e889d37`.

Across full training, mean all-success group rate was `0.6183`, all-failure
rate `0.0283`, mixed rate `0.3533`, and hard all-failure rate `0.0603`.
Although `64.7%` of groups had binary zero variance, most were all-success
groups; only a small minority were all-failure groups.

Controlled comparison with the terminal-only batch-16 gen-4 baseline:

| Metric | Terminal only | Binary reward | Delta |
|---|---:|---:|---:|
| Eval task success | `0.96` | `0.93` | `-0.03` |
| Medium-tier task success | `0.9811` | `0.9811` | `0.00` |
| Hard-tier task success | `0.9167` | `0.8333` | `-0.0834` |
| Invalid-trace rate | `0.04` | `0.06` | `+0.02` |
| Correct-but-invalid rate | `0.00` | `0.00` | `0.00` |
| Call-count error rate | `0.03` | `0.05` | `+0.02` |
| Mean training KL | `0.00928` | `0.00619` | `-0.00309` |
| Training generated tokens | `1,210,291` | `1,223,282` | `+12,991` |
| Training wall time | `433.4 s` | `450.9 s` | `+17.5 s` |
| Eval generated tokens | `22,369` | `23,395` | `+1,026` |
| Eval wall time | `120.1 s` | `128.8 s` | `+8.7 s` |

The binary policy fixed `calculator_0188` and `calculator_0683`, while
`calculator_0098` and `calculator_0600` remained failures. It introduced five
new failures: `calculator_0406`, `calculator_0421`, `calculator_0431`,
`calculator_0800`, and `calculator_0809`. Reward/eval label replay produced
zero disagreements, so the scorer worked as specified; the binary learning
signal itself did not improve policy quality. Reject the candidate, leave
held-out test untouched, and retain the selected terminal-only policy.

#### 7. Completed experiment: strict penalties with a `+5` success spike (rejected)

Repeat experiment 5 with only the valid-correct reward changed from `+10.0` to
`+5.0`. All invalid rewards, diagnostics, data, seed, optimizer settings, and
evaluation settings remain unchanged. At user request, run the full training
directly without a smoke.

The run used seed `42`, gen `4`, batch `16`, gradient accumulation `2`,
`steps_per_generation=2`, expandable CUDA segments, and colocated vLLM memory
utilization `0.24`. It completed in `444.7` seconds with mean GPU utilization
`81.3%` and `9,796 MiB` free at peak memory. The adapter is pinned at
`tripathysagar/qwen3-0.6b-calc-grpo600-strict5-b16@841db07675ed40bc46e5bc22187e88e479deab3b`.

Complete 100-row greedy eval comparison:

| Metric | Strict `+2` | Strict `+5` | Strict `+10` |
|---|---:|---:|---:|
| Eval task success | `0.96` | `0.90` | `0.94` |
| Medium-tier task success | `1.00` | `0.9623` | `0.9811` |
| Hard-tier task success | `0.8889` | `0.7778` | `0.8611` |
| Invalid-trace rate | `0.04` | `0.10` | `0.06` |
| Correct-but-invalid rate | `0.00` | `0.01` | `0.00` |
| Call-count error rate | `0.01` | `0.07` | `0.05` |
| Mean training KL | `0.00751` | `0.00769` | `0.00747` |
| Training generated tokens | `1,210,472` | `1,210,871` | `1,212,141` |
| Training wall time | `432.9 s` | `444.7 s` | `447.5 s` |
| Eval generated tokens | `22,174` | `22,366` | `22,217` |
| Eval wall time | `120.8 s` | `119.9 s` | `122.3 s` |

The `+5` policy fixed `calculator_0098` relative to strict `+2`, but retained
`calculator_0431`, `calculator_0683`, and `calculator_0778` and introduced
seven failures: `calculator_0188`, `calculator_0393`, `calculator_0422`,
`calculator_0470`, `calculator_0600`, `calculator_0609`, and
`calculator_0979`. Nonnegative-invalid reward remained `0.00`, so the reward
safety invariant held, but task success, hard-tier success, trace validity,
and call-count accuracy all regressed.

The success-magnitude response is non-monotonic (`+2`: `0.96`, `+5`: `0.90`,
`+10`: `0.94`) and provides no evidence that increasing the raw valid-correct
reward helps under group-normalized GRPO. Reject strict `+5`, leave held-out
test untouched, and retain the selected terminal-only policy.

### Shared validity rules and selected terminal-only precedence

**A trajectory is valid if all three hold:**

1. **Every operand is grounded, not just correct-looking.** Each call's operands must be either (a) original constants from the expression that haven't been consumed yet, or (b) the result of a tool call the model has *already made and observed* earlier in this same trajectory — never a number the model simply states without having derived it. This is what closes the "state `add(8,20)` before ever calling `mul(4,5)`" loophole from a few turns back.

2. **Every required operation is consumed exactly once — no more, no less.** The set of needed operations comes directly from the expression's structure (same one you use to generate `expected_call_count`). No omissions, no duplicate re-calls of an already-satisfied operation. This closes the repeat-farming loophole.

3. **The full expression ends fully reduced.** The last call's result must be the correct reduction of the *entire* expression — not a partial reduction that stops early or wanders off into an irrelevant subset of the required operations.

If all three hold → valid trajectory. If any one fails → invalid trajectory, regardless of what the final stated answer happens to be.

Reward precedence is explicit:

1. If there is no parseable final answer, return `-0.1` regardless of the
   partial trace.
2. Otherwise validate the complete trace and answer, then apply the four
   remaining rows.

Do not add partial-progress, format, call-count, or length shaping in the first
run. Log those as diagnostics only. Track the no-answer rate because `-0.1` is
better than `-1.0`; abort if the policy learns to exploit abstention rather
than improving task success.

The eval harness has the right reduction-state algorithm in
`validate_semantic_trace`, but it does not contain a separate
`is_legal_step`, and its first returned boolean only means at least one call
was present. Before GRPO, extract one shared checker used by evaluation and
reward:

```python
result = score_trajectory(
    expression=expression,
    calls=ordered_calls,
    stated_answer=parsed_final_answer,
    expected_call_count=expected_call_count,
)
# all_calls_legal, fully_reduced, answer_correct, valid_trajectory
```

`valid_trajectory` requires all calls legal, exactly the expected number of
calls, and full reduction to the expression's true value. Never use exact
reference-trace matching.

### Policy initialization: continue the SFT LoRA

GRPO starts from the exact SFT policy already evaluated at 82% greedy task
success:

- base model: `Qwen/Qwen3-0.6B` at revision
  `c1899de289a04d12100db370d81485cdf75e47ca`
- SFT adapter: `tripathysagar/qwen3-0.6b-calc-sft200` at revision
  `86db06c14d91acc734e89a45bb5ca3ec4e1ee8f3`

Load the base model, attach the pinned SFT adapter with
`is_trainable=True`, and give that `PeftModel` directly to `GRPOTrainer`.
Do **not** call `merge_and_unload()` and do **not** pass a new `peft_config`.
This continues updating the same rank-16 LoRA matrices while the base weights
remain frozen. The final GRPO adapter contains the combined SFT+GRPO delta
relative to the original base model.

Use a new output directory and Hub repository for GRPO. Never overwrite the
pinned SFT adapter.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_ID = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SFT_ADAPTER_ID = "tripathysagar/qwen3-0.6b-calc-sft200"
SFT_ADAPTER_REVISION = "86db06c14d91acc734e89a45bb5ca3ec4e1ee8f3"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    revision=BASE_REVISION,
    dtype="float16",  # L4: bfloat16
)
policy = PeftModel.from_pretrained(
    base_model,
    SFT_ADAPTER_ID,
    revision=SFT_ADAPTER_REVISION,
    is_trainable=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    SFT_ADAPTER_ID,
    revision=SFT_ADAPTER_REVISION,
)
```

Set `beta=0.02` for the first run. With a pretrained `PeftModel`, TRL copies
the initial SFT adapter into a frozen `ref` adapter and trains the `default`
adapter. This makes the KL reference the SFT policy, not the raw base model.
Before training, assert that:

1. only `default` LoRA parameters require gradients;
2. `ref` exists, is frozen, and initially equals `default`; and
3. no base-model parameter requires gradients.

### GRPO-600 dataset

Build the training dataset by filtering the HF `train` split in
`data/calculator_qwen3/` with `data/calculator_qwen3_grpo600_ids.txt`. Assert
exactly 600 unique rows,
zero overlap with the SFT-200 IDs, and matching input SHA256 values from the
SFT run artifacts.

Each row contains:

- `prompt`: only the original system and user messages (`messages[:2]`);
- `expression`, `final_answer`, `expected_call_count`, `tier`, and `id`;
- the calculator tool schema, but no reference assistant/tool trajectory.

The reference trajectory is not a label and must not appear in the prompt or
reward. Any precedence-respecting reduction order is acceptable. Keep the
100-row eval split for checkpoint selection and leave test untouched until the
final report.

### Stateful calculator rollout

Use TRL 1.8.0's `environment_factory` so every sampled completion receives a
fresh calculator environment. Its `reset()` stores the expression, answer,
expected operation count, and untouched reduction state. Expose one
`calculator(op, a, b)` method with the same schema and arithmetic behavior as
the eval harness, recording every call in order.

The trainer handles the multi-turn loop: generate, execute a tool call, append
its result, and continue until a final answer or the cap. Keep generation
aligned with evaluation:

- thinking enabled;
- sampling enabled for GRPO (`temperature=0.8`, `top_p=0.95`);
- at most 6 tool-calling turns;
- at most 1024 generated tokens across the complete rollout;
- the same Qwen chat template and calculator schema used by SFT/eval.

Before training, compare 20 fixed-seed environment rollouts with the existing
eval loop. Tool parsing, observations, answer parsing, budget termination, and
semantic-validity results must agree.

### Initial GRPO configuration

These are smoke-test defaults, not claims of an optimal final run:

| Knob | Initial choice | Reason |
|---|---:|---|
| `num_generations` | `4` | within-prompt contrast that fits T4/L4 smoke |
| `per_device_train_batch_size` | `4` | divisible by 4 generations on one GPU |
| `gradient_accumulation_steps` | `2` | effective completion batch of 8 |
| `learning_rate` | `1e-5` | conservative continuation of a trained adapter |
| `num_train_epochs` | `1` | first full pass over GRPO-600 |
| `beta` | `0.02` | KL anchor to the frozen SFT adapter |
| `loss_type` | `"dapo"` | avoids original GRPO response-length bias |
| `scale_rewards` | `"batch"` | avoids per-question reward-std difficulty bias |
| `temperature` / `top_p` | `0.8` / `0.95` | controlled exploration |
| `max_completion_length` | `1024` | total multi-turn rollout budget |
| `max_tool_calling_iterations` | `6` | matches evaluation |
| `mask_truncated_completions` | `False` | exhausted rollouts retain their penalty |
| precision | T4 `fp16`; L4 `bf16` | match supported hardware |
| `gradient_checkpointing` | `True` | reduce Colab memory use |
| generation backend | Transformers on T4 | vLLM 0.23/FlashInfer fails on SM75; benchmark colocated vLLM only as an explicit L4/A100 opt-in |

Run gates:

1. **Reward unit tests:** hand-construct every reward row plus omitted,
   duplicate, precedence-violating, wrong-operand, and alternative-valid-order
   traces.
2. **Overfit smoke:** 8 prompts, 4 generations each, 10-20 optimizer steps.
   Verify reward variance, changing LoRA weights, frozen base/reference
   weights, and finite loss/KL.
3. **GRPO-600 run:** save locally and upload once at the end. Record versions,
   revisions, hashes, config, trainer state, all five outcome rates, per-tier
   task success, under/exact/over-call rates, legal-progress mean,
   correct-answer-but-invalid-trace rate, truncation rate, reward standard
   deviation/zero-variance fraction, entropy, and KL.
4. **Checkpoint selection:** use greedy task success on the 100-row eval split
   with the existing 128-token-per-turn/6-call evaluator. Do not select using
   trainer loss or mean reward alone.
5. **Final artifact:** push the adapter and tokenizer to a new private Hub
   repo, pin its commit, then evaluate the held-out test split once.

### GRPO implementation and remaining tasks

- [x] Extract the semantic reducer, stateful environment, answer parser, and
  reward scorer into `scripts/calculator_semantics.py`; update the GRPO notebook and
  eval script to import it.
- [x] Add notebook reward unit tests covering every terminal outcome and
  omitted/duplicate/precedence/wrong-operand/wrong-operator traces.
- [x] Build the 600-row prompt-only dataset with overlap/hash assertions.
- [x] Implement the stateful calculator environment.
- [x] Run the planned fixed-seed 20-row parity test between the GRPO
  environment and eval harness. All tool observations, parsed answers, complete
  traces, rewards, and omitted-final-call rejection checks passed; preserve
  `outputs/evaluations/grpo-environment-parity.json`.
- [x] Create `notebooks/grpo_qwen3_calculator.ipynb` using the pinned
  trainable SFT adapter; run its smoke gate before marking runtime tasks done.
- [x] Run the 8-prompt/10-step T4 smoke: reward std max `1.3887`, KL max
  `0.00569`, no-answer rate `0.0-0.125`; trainable adapter changed, frozen
  reference remained identical, and all artifact checks passed.
- [x] Run the 8-prompt/10-step A100 colocated-vLLM smoke: 33.9 seconds,
  nonzero reward contrast, trainable adapter changed, frozen reference remained
  identical, and all artifact checks passed.
- [x] Run all 600 GRPO rows for one epoch on A100 with colocated vLLM: 300
  optimizer steps in 912 seconds; final-batch task success `0.875`, reward std
  `1.0607`, KL `0.00891`, no truncation, and all finite/hash gates passed.
  Upload the adapter privately and pin Hub revision
  `0c92c02bb79e1e186cb61ce2bbf50c239a0afae1`.
- [x] Benchmark a 10-step A100 smoke with batch size `16`, four generations,
  and GPU telemetry: 4.52 completions/s versus 2.363 at batch size `4`;
  mean utilization `67.2%`, p95 `100%`, peak memory `39,314/40,960 MiB`,
  leaving only `1,646 MiB`. Treat batch `16` as too tight for an unmonitored
  full run with variable completion lengths.
- [x] Run a separate full GRPO-600 batch-`16` performance candidate with vLLM
  memory reservation reduced to `0.25`: 75 optimizer steps in 433.4 seconds
  versus 300 steps in 912.0 seconds for the batch-`4` run (`2.10x` wall-time
  speedup for the same 600 prompts/four generations). Mean GPU utilization was
  `81.2%`, peak memory `38,648/40,960 MiB`, and minimum free memory `2,312 MiB`.
  The final batch reached task success `0.9375`, reward std `0.6278`, KL
  `0.01130`, and no truncation; adapter/reference hash gates passed. Upload as
  a separate private candidate and pin revision
  `13ae8ff6e3c74f3cc62a4f17ce388f37c97e7825`; do not replace the selected
  98%-test adapter without a controlled eval comparison.

### Single-seed generation-count ablation

Goal: compare `num_generations` `4`, `8`, and `16` at seed `42` without
confounding the result with fewer unique prompts per optimizer update, while
keeping the A100 close to the proven batch-`16` utilization envelope.

Common controls:

- Same pinned SFT initialization, GRPO-600 prompt order, reward, one epoch,
  learning rate, KL coefficient, decoding parameters, and seed `42`.
- Keep `per_device_train_batch_size=16`, `steps_per_generation=2`, and
  colocated-vLLM memory utilization `0.25`. This holds each vLLM generation
  batch at 32 completions (`max_num_seqs=32`) and uses the measured
  `38,648/40,960 MiB` batch-`16` envelope.
- Hold eight unique prompts and 75 optimizer updates per arm by setting
  gradient accumulation to `2`, `4`, and `8` for generation counts `4`, `8`,
  and `16`, respectively.
- Preserve every adapter/checkpoint and GPU telemetry before releasing the
  runtime. Require a short exact-config smoke before each full arm.
- Compare both final quality and efficiency: greedy task success overall/by
  tier on the same 100-row eval split, reward contrast/zero-std fraction, KL,
  entropy, truncation, generated tokens, wall time, completions/s, mean/p95 GPU
  utilization, peak memory, and eval gain per generated token/minute.
- The arms intentionally match prompt exposure and optimizer updates, not
  rollout compute: `8` and `16` generations consume `2x` and `4x` as many
  completions as `4`. Report convergence against wall time/generated tokens so
  extra generations are not credited for extra compute without qualification.
- Do not touch the held-out test split or replace the selected adapter from
  this ablation; use eval only for the comparison.

- [x] `4` generations: batch `16`, gradient accumulation `2`,
  `steps_per_generation=2`; full candidate pinned at
  `13ae8ff6e3c74f3cc62a4f17ce388f37c97e7825`.
- [x] `8` generations: exact-config smoke passed, then full batch `16`,
  gradient accumulation `4`, `steps_per_generation=2`; 75 updates in `820.5`
  seconds, mean GPU utilization `83.1%`, peak memory `38,404 MiB`. Pin private
  candidate revision `3348e4da1765e7a544fd7540bb8d4d1695f9d872`.
- [x] `16` generations: exact-config smoke passed, then full batch `16`,
  gradient accumulation `8`, `steps_per_generation=2`; 75 updates in `1526.2`
  seconds, mean GPU utilization `86.6%`, peak memory `40,052 MiB` with only
  `908 MiB` free. Pin private candidate revision
  `b446308cb6397645d68ae792b6a012c39c7e63b1`. The first completed runtime was
  reclaimed before adapter recovery, so preserve the deterministic retry
  bundle immediately.
- [x] Evaluate all three pinned candidates greedily on the identical 100-row
  eval split. Task success was gen-4 `0.96`, gen-8 `0.95`, and gen-16 `0.93`;
  hard-tier success was `0.9167`, `0.8889`, and `0.8333`. Gen-4 dominates:
  gen-8 used `1.89x` wall time/`2x` rollouts and gen-16 used `3.52x` wall
  time/`4x` rollouts without a quality gain. Preserve
  `outputs/evaluations/qwen3-calc-grpo-generation-ablation-comparison.json`
  and the per-arm eval artifacts.

- [x] Evaluate the pinned final GRPO adapter greedily on all 100 eval rows:
  task success `0.95` versus the controlled SFT baseline `0.82`; easy `1.00`,
  medium `0.9811`, hard `0.8889`, with no parse errors or budget exhaustion.
- [x] Select the pinned final adapter as the final-test candidate: it is the
  only preserved GRPO checkpoint and passed the eval gate at `0.95` task
  success. Intermediate-checkpoint comparison would require a new run that
  preserves those adapters.
- [x] Preserve eval artifacts:
  `outputs/evaluations/qwen3-calc-grpo600-eval-greedy-a100-*`.
- [x] Analyze the five failed GRPO eval rows. All five used syntactically valid
  tools but failed semantic grounding/reduction and gave wrong answers; none
  was a correct-answer-through-invalid-trace, parser, or budget failure.
  Preserve `outputs/evaluations/qwen3-calc-grpo600-eval-greedy-a100-failure-analysis.json`.
- [x] Evaluate the selected pinned GRPO adapter on the held-out test split
  exactly once. Greedy A100 task success was `0.98`: easy `1.00`, medium
  `1.00`, hard `0.9444`; no parse errors or budget exhaustion. Preserve
  `outputs/evaluations/qwen3-calc-grpo600-test-greedy-a100-*`.
- [x] Write the final base-vs-SFT-vs-GRPO comparison report from pinned
  revisions and preserved artifacts: `GRPO_FINAL_REPORT.md`.
- [ ] Optional: rerun GRPO only if intermediate-checkpoint comparison is
  required; preserve each candidate adapter before releasing the runtime.
- [ ] Optional: run additional GRPO seeds or sampled evals to measure training
  and decoding variance.

---

## A100 rerun and OOD evaluation (2026-07-24)

- [x] Rerun full SFT-200 training on A100 and evaluate greedily on the 100-row
  in-distribution eval split.
  - SFT adapter:
    `tripathysagar/qwen3-0.6b-calc-sft200-a100-rerun-20260724@57ebda77e10cac33866472b38432bf2cf4ac3b3a`
  - Eval task success: `0.80`; answer accuracy: `0.82`; valid tool use:
    `0.99`; budget exhaustion: `0.00`.
- [x] Rerun all seven GRPO reward experiments from that pinned SFT adapter on
  A100 with seed `42`, four generations, batch size `16`, and greedy 100-row
  eval.
  - Terminal-only: `0.91` eval task success.
  - Normalized progress `0.1`: `0.92`.
  - Per-call progress `0.5`: `0.91`.
  - Strict invalid penalties, success `+2`: `0.93`.
  - Strict invalid penalties, success `+5`: `0.89`.
  - Strict invalid penalties, success `+10`: `0.93`.
  - Execution-grounded binary reward: `0.92`.
  - The first strict `+5` attempt failed at step `25` from CUDA fragmentation.
    A clean retry with
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` completed all `75`
    updates.
- [x] Preserve each GRPO LoRA in a separate private Hub repository with an
  immutable revision, plus local revision manifests, training bundles, executed
  notebooks, and session histories under `outputs/`.
- [x] Evaluate the rerun SFT policy and all seven GRPO policies on each isolated
  100-row OOD split using A100, greedy decoding, seed `42`, batch size `16`,
  and `128` new tokens per assistant turn.
  - The evaluator derives the tool budget from the split: `8` calls for
    `ood_length` (6–8 operations), and the existing `6`-call safety cap for
    `ood_negative` and `ood_float`.
  - Task success by policy, ordered as length / negative / float:
    - SFT: `0.18 / 0.14 / 0.66`.
    - Terminal-only: `0.28 / 0.19 / 0.81`.
    - Normalized progress `0.1`: `0.23 / 0.19 / 0.84`.
    - Per-call progress `0.5`: `0.25 / 0.19 / 0.87`.
    - Strict `+2`: `0.26 / 0.20 / 0.87`.
    - Strict `+5`: `0.30 / 0.24 / 0.87`.
    - Strict `+10`: `0.31 / 0.22 / 0.90`.
    - Binary: `0.21 / 0.20 / 0.86`.
  - Best length and float policy: strict `+10` at `0.31` and `0.90`.
  - Best negative policy: strict `+5` at `0.24`.
  - Float transfer is strong, but longer trajectories and signed operands
    remain major generalization failures even after removing the artificial
    six-call ceiling from the length split.
- [x] Validate all `24` OOD evaluations and `2,400` predictions: exactly `100`
  unique IDs per run, finite metrics, matching split/configuration metadata,
  and no budget-exhaustion cases.
- [x] Preserve complete OOD artifacts and the comparison summary under
  `outputs/evaluations/qwen3-calc-ood-a100-20260724/`; preserve the reusable
  Colab runner as `scripts/run_ood_eval_sweep_colab.sh`.
- [x] Stop both A100 OOD sessions and verify that no Colab runtime remains.
- [ ] Classify OOD failures by omitted operations, precedence errors,
  signed-operand grounding, and decimal-representation errors before changing
  the reward again.

---

## Negative-operand 10% data-mix ablation

Goal: test whether direct negative-operand exposure improves the weakest OOD
shift while holding model, total prompt exposure, optimizer settings, and GRPO
reward fixed.

- [x] Generate `80` fresh negative-operand expressions disjoint from every
  existing train, eval, test, and OOD expression. Replace 80 removed train slots
  while preserving their IDs so the frozen 200/600 partition remains auditable;
  output IDs remain globally unique. Keep the original 100-row `ood_negative`
  evaluation split untouched.
- [x] Build `data/calculator_qwen3_negative10/` with:
  - frozen SFT partition: `180` ID rows + `20` fresh negative rows.
  - frozen GRPO partition: `540` ID rows + `60` different fresh negative rows.
  - unchanged `eval`, `test`, `ood_length`, `ood_negative`, and `ood_float`
    splits.
- [x] Preserve tier proportions where possible; assert exact `10%` negative
  composition in both stages, no SFT/GRPO overlap, no evaluation contamination,
  and deterministic hashes/fingerprints.
- [x] Start from the pinned Qwen3-0.6B base model and train a fresh SFT LoRA on
  the mixed 200-row SFT partition.
- [x] Continue that exact new SFT LoRA with GRPO on the mixed 600-row GRPO
  partition.
- [x] Hold the GRPO reward fixed to the strict-invalid `+5` configuration. This
  is the empirically best existing Negative OOD reward arm (`0.24` task
  success), so the controlled baseline is the 0%-negative strict `+5` policy,
  not terminal-only.
- [x] Keep the proven A100 controls: seed `42`, four generations, an effective
  batch of 8 prompts / 32 completions per optimizer update, and generation
  batch size `32`. Use microbatch `8`, gradient accumulation `4`, and
  `steps_per_generation=4` after microbatch `16` repeatedly exceeded 40 GB on
  the mixed trajectories. Use colocated vLLM memory utilization `0.20` and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- [x] Run base → SFT → strict-`+5` GRPO recoverably on A100. A transient
  websocket cleanup race and two invalid GRPO OOM attempts required fresh
  allocations; no failed adapter was published. The successful SFT and final
  GRPO/evaluation runs have clean kernel restarts, secure `.env` reinjection,
  immutable Hub revisions, immediate artifact recovery, and verified cleanup.
- [x] Evaluate both the new SFT and GRPO policies greedily on the unchanged
  100-row `eval`, `ood_length`, `ood_negative`, and `ood_float` splits. Leave
  `test` untouched.
- [x] Compare against the existing 0%-negative SFT and strict-`+5` baselines,
  reporting Negative OOD gain together with any ID, length, or float regression.
- [x] Preserve the mixed dataset manifest, ID lists, training bundles, executed
  notebooks/session history, Hub revision manifests, metrics, predictions, and
  comparison summary under `outputs/`.

### Negative-10% ablation results

- Dataset validation: exactly `20/200` negative SFT rows and `60/600` negative
  GRPO rows. Every evaluated split has the same selected-record SHA-256 as its
  0%-negative baseline; every prediction file contains 100 unique IDs; `test`
  was never evaluated.
- SFT adapter:
  `tripathysagar/qwen3-0.6b-calc-sft200-negative10-a100-20260724@d471cd159f9fc6f0ce27f916f670f8897c9786e7`.
  Task success changed: eval `0.80 → 0.83`, length `0.18 → 0.15`, negative
  `0.14 → 0.42`, float `0.66 → 0.69`.
- Strict-`+5` GRPO adapter:
  `tripathysagar/qwen3-0.6b-calc-grpo600-negative10-strict5-a100-20260724@e3d65a3d2df983705c4355721aff522744284044`.
  Task success changed: eval `0.89 → 0.94`, length `0.30 → 0.32`, negative
  `0.24 → 0.54`, float `0.87 → 0.93`.
- Verdict: 10% direct negative-operand exposure produced a large, targeted
  Negative OOD gain: **+28 points after SFT** and **+30 points after strict-`+5`
  GRPO**. GRPO also removed the SFT length regression and improved all measured
  non-test splits.
- Verified artifacts:
  `outputs/evaluations/qwen3-calc-negative10-strict5-a100-20260724/` and
  `outputs/notebooks/qwen3-calc-negative10-strict5-a100-20260724/`.
  All Colab sessions are stopped.

---

## Floating-point 20% / 30% / 40% data-mix ablation

Goal: measure the float-data dose response while holding total SFT/GRPO
exposure, model, reward, optimizer, seed, and evaluation records fixed. These
are float-only treatments; do not combine them with the completed negative-10%
mixture.

- [x] Generate one deterministic pool of `320` fresh positive half-integer
  trajectories: `80` reserved for SFT and `240` different rows reserved for
  GRPO. Make every expression disjoint from existing train, eval, test, and OOD
  rows. Keep the original 100-row `ood_float` split untouched.
- [x] Use nested, tier-stratified float subsets so the 20% rows are contained in
  the 30% arm and the 30% rows are contained in the 40% arm. Use corresponding
  nested ID replacements to isolate mixture amount from row-selection noise.
- [x] Build three deterministic datasets with frozen stage sizes:
  - `data/calculator_qwen3_float20/`: SFT `160` ID + `40` float; GRPO `480` ID
    + `120` float.
  - `data/calculator_qwen3_float30/`: SFT `140` ID + `60` float; GRPO `420` ID
    + `180` float.
  - `data/calculator_qwen3_float40/`: SFT `120` ID + `80` float; GRPO `360` ID
    + `240` float.
  - Copy unchanged `eval`, `test`, `ood_length`, `ood_negative`, and
    `ood_float` splits into every arm.
- [x] Assert the exact float percentage in both stages of every arm, no
  SFT/GRPO expression overlap, no evaluation contamination, globally unique
  output IDs, nested-subset invariants, and deterministic manifests/hashes.
- [x] For each of 20%, 30%, and 40%, start independently from pinned
  `Qwen/Qwen3-0.6B`, train a fresh 200-row SFT LoRA, then continue that arm's
  exact SFT LoRA with GRPO on its 600-row partition. Never initialize a larger
  mixture from a smaller mixture's adapter.
- [x] Hold GRPO fixed to the strict-invalid `+5` reward selected for the
  negative ablation.
- [x] Use the memory-safe A100 GRPO geometry: seed `42`, four generations,
  microbatch `8`, gradient accumulation `4`, `steps_per_generation=4`,
  generation batch `32`, vLLM utilization `0.20`, and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- [x] Create one resumable A100 driver that runs all SFT arms followed by all
  GRPO arms, minimizing package-stack transitions. Restart and reinject the
  environment between stages, pin each Hub revision, and download each arm's
  artifacts immediately so reclaimed runtimes can resume at the next unfinished
  stage.
- [x] Evaluate all six pinned policies greedily on the unchanged 100-row
  `eval`, `ood_length`, `ood_negative`, and `ood_float` splits with seed `42`;
  leave `test` untouched. Recover each policy's evaluation before continuing.
- [x] Compare the 0%, 20%, 30%, and 40% float mixtures separately for SFT and
  strict-`+5` GRPO. Report Float OOD gain, marginal gain per additional 10
  points of float data, and every eval/length/negative regression.
- [x] Preserve dataset manifests, replacement mappings, training bundles,
  executed notebooks, immutable Hub revisions, per-arm metrics/predictions, the
  combined dose-response comparison JSON, and verified Colab cleanup under
  `outputs/`.

### Floating-point mixture sweep results

- All 24 evaluations used the same selected-record SHA-256 values as their
  0%-float baselines, produced 100 unique predictions, and finite metrics.
  `test` remained untouched.
- SFT task success by mixture `(eval, length, negative, float)`:
  - 0% baseline: `(0.80, 0.18, 0.14, 0.66)`.
  - 20%: `(0.78, 0.09, 0.18, 0.63)`.
  - 30%: `(0.81, 0.19, 0.11, 0.89)`.
  - 40%: `(0.82, 0.16, 0.17, 0.84)`.
- Strict-`+5` GRPO task success by mixture `(eval, length, negative, float)`:
  - 0% baseline: `(0.89, 0.30, 0.24, 0.87)`.
  - 20%: `(0.97, 0.36, 0.23, 0.92)`.
  - 30%: `(0.95, 0.37, 0.13, 0.92)`.
  - 40%: `(0.91, 0.38, 0.23, 0.90)`.
- Verdict: **20% is the best GRPO mixture**. It ties the best Float OOD result
  (`0.92`) while giving the highest eval result (`0.97`) and avoiding the
  30%-arm Negative OOD collapse. More float data is not monotonically better.
  For SFT alone, 30% gives the highest Float OOD (`0.89`), but that advantage
  does not survive GRPO.
- Immutable revisions:
  - SFT-20 `df910e368095d582fce93b8f8cc28ba22fa37c44`; GRPO-20
    `a5d21a6e7124aae032b888a75add58a23e914925`.
  - SFT-30 `089f5e98a3890752c57e83ac53e68f79bd2fa0b9`; GRPO-30
    `bb0a1a4fd2dfa4df1a47005ebab0fdf01120831b`.
  - SFT-40 `1eb65dfe24ccf852c64addd1a857411a86f18d8b`; GRPO-40
    `89aa2e24b7105d6d2b474bb34fe4ee6d3638c269`.
- Verified artifacts:
  `outputs/evaluations/qwen3-calc-float-mix-sweep-a100-20260724/` and
  `outputs/notebooks/qwen3-calc-float-mix-sweep-a100-20260724/`.
  All Colab sessions are stopped.

---

## Combined generalization mixture with 10% length data

Goal: test the evidence-informed combined mixture without exceeding the float
ratio that preserved cross-domain performance:

- ID: `60%`
- negative operands: `10%`
- floating-point operands: `20%`
- long expressions with 6–8 operations: `10%`

- [x] Generate fresh, evaluation-disjoint domain pools and build one
  deterministic dataset with:
  - SFT: `120` ID + `20` negative + `40` float + `20` length = `200`.
  - GRPO: `360` ID + `60` negative + `120` float + `60` length = `600`.
  - Different domain examples for SFT and GRPO.
  - Unchanged `eval`, `test`, `ood_length`, `ood_negative`, and `ood_float`
    splits.
- [x] Preserve the frozen SFT/GRPO ID partition through explicit precomputed-ID
  loading. Do not recompute the partition by tier after adding 6–8-operation
  rows, because all length rows are classified as hard.
- [x] Assert exact domain percentages, globally unique output IDs, no
  SFT/GRPO expression overlap, no evaluation contamination, and deterministic
  manifests/hashes.
- [x] Keep integer operands in ID, negative, and length records while using one
  numeric calculator schema capable of executing every row in this mixed GRPO
  arm without coercing fractional values.
- [x] Make SFT maximum sequence length configurable. Measure all mixed
  trajectories first and use the smallest non-truncating value, up to `2048`.
- [x] Make GRPO tool iterations configurable and use `8` so every
  six-to-eight-operation training trajectory can finish.
- [x] Train a fresh SFT LoRA from pinned `Qwen/Qwen3-0.6B`; then train
  strict-invalid `+5` GRPO from that exact immutable SFT revision.
- [x] Use seed `42`, four generations, effective generation batch `32`, and the
  established memory-safe A100 microbatch/accumulation geometry. Reduce the
  microbatch further only if the longer trajectories cause measured OOM.
- [x] Evaluate SFT and GRPO greedily on the unchanged 100-row `eval`,
  `ood_length`, `ood_negative`, and `ood_float` splits. Leave `test` untouched.
- [x] Compare against the 0%-OOD baseline and the best isolated treatments:
  negative-10% and float-20%. Report task success for every split and identify
  cross-domain interference rather than selecting only on average accuracy.
- [x] If 10% length improves Length OOD without material regression, run the
  planned second arm with `50%` ID, `10%` negative, `20%` float, and `20%`
  length to locate the length-data threshold.
- [x] Preserve dataset provenance, executed notebooks, immutable Hub revisions,
  metrics, predictions, comparison JSON, and verified Colab cleanup under
  `outputs/`.

### Combined 60/10/20/10 results

- The deterministic mixed dataset contains the exact stage counts above, keeps
  the original frozen 200/600 IDs, and leaves all five evaluation/test splits
  byte-equivalent by canonical SHA-256. SFT/GRPO replacement pools are disjoint.
- The longest selected SFT trajectory is `941` tokens, so `1024` is the smallest
  supported configured context that avoids truncation. GRPO used eight tool
  rounds.
- SFT task success `(eval, length, negative, float)`:
  `(0.77, 0.22, 0.41, 0.74)`.
- Strict-`+5` GRPO task success `(eval, length, negative, float)`:
  `(0.91, 0.33, 0.53, 0.89)`.
- Versus the 0%-OOD GRPO baseline `(0.89, 0.30, 0.24, 0.87)`, the combined arm
  changes task success by `(+0.02, +0.03, +0.29, +0.02)`: every measured split
  improves, but Length OOD moves only three points.
- Versus negative-10% GRPO `(0.94, 0.32, 0.54, 0.93)`, the combined arm changes
  `(-0.03, +0.01, -0.01, -0.04)`. Versus float-20% GRPO
  `(0.97, 0.36, 0.23, 0.92)`, it changes
  `(-0.06, -0.03, +0.30, -0.03)`. This is measurable cross-domain
  interference: the combined policy is balanced and better than the ID-only
  policy, but does not preserve each isolated treatment's peak score.
- Immutable revisions:
  - SFT `1abd82f606fdad24be363823aacaf123b56a14d0`.
  - GRPO `c66e7108f22addd6cfd05a2c2116bb5ec9f47d30`.
- Verified artifacts:
  `outputs/evaluations/qwen3-calc-combined60102010-a100-20260725/` and
  `outputs/notebooks/qwen3-calc-combined60102010-a100-20260725/`.
  The A100 session is stopped.

### Combined 50/10/20/20 results

- This is a nested treatment: all negative, float, and original 10%-length
  examples are identical to the 60/10/20/10 arm. Only another `20` SFT and `60`
  GRPO hard ID rows were replaced by fresh 6–8-operation expressions.
- SFT task success `(eval, length, negative, float)`:
  `(0.80, 0.28, 0.43, 0.74)`.
- Strict-`+5` GRPO task success `(eval, length, negative, float)`:
  `(0.93, 0.44, 0.57, 0.92)`.
- Increasing length data from 10% to 20% changes GRPO task success by
  `(+0.02, +0.11, +0.04, +0.03)`. The Length OOD gain is substantial and no
  measured split regresses.
- Versus the 0%-OOD GRPO baseline, the 20%-length arm changes task success by
  `(+0.04, +0.14, +0.33, +0.05)`. It also nearly matches negative-10% on eval
  and Float OOD while exceeding it on Length and Negative OOD.
- Verdict: the `50/10/20/20` mixture is the best balanced policy tested. The
  10%→20% response does not yet show a length-data saturation threshold.
- Immutable revisions:
  - SFT `fd0ee9d88f0025f68845ea8314a5e03c2d78be63`.
  - GRPO `69e333d3bbcc132798eefbeff3a6fc9da9f16963`.
- Verified artifacts:
  `outputs/evaluations/qwen3-calc-combined50102020-a100-20260725/` and
  `outputs/notebooks/qwen3-calc-combined50102020-a100-20260725/`.
  The A100 session is stopped.

