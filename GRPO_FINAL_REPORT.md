# Calculator policy final report

## Decision

Use `tripathysagar/qwen3-0.6b-calc-grpo600` at revision
`0c92c02bb79e1e186cb61ce2bbf50c239a0afae1` as the final calculator
policy. It passed the 100-row eval selection gate at 95% task success and,
when evaluated exactly once on the untouched 100-row test split, reached 98%.

## Pinned policies

- Base: `Qwen/Qwen3-0.6B` at
  `c1899de289a04d12100db370d81485cdf75e47ca`.
- SFT: `tripathysagar/qwen3-0.6b-calc-sft200` at
  `86db06c14d91acc734e89a45bb5ca3ec4e1ee8f3`.
- GRPO: `tripathysagar/qwen3-0.6b-calc-grpo600` at
  `0c92c02bb79e1e186cb61ce2bbf50c239a0afae1`.

## Controlled eval comparison

The primary comparison uses the same 100 eval prompts, A100 hardware, greedy
decoding, tool loop, six-call limit, and 128-token per-turn cap.

- Base task success: 3%; answer accuracy: 34%; valid tool use: 53%; budget
  exhaustion: 52%.
- SFT task success: 82%; answer accuracy: 85%; valid tool use: 100%; budget
  exhaustion: 0%.
- GRPO task success: 95%; answer accuracy: 95%; valid tool use: 100%; budget
  exhaustion: 0%.

SFT improves task success by 79 percentage points over base. GRPO adds another
13 points over SFT and reaches a 92-point improvement over base. The result is
not merely better arithmetic accuracy: both trained policies eliminate the
base model's tool-format and budget failures, while GRPO further improves
semantic trajectory correctness.

GRPO eval task success by tier was 100% easy, 98.11% medium, and 88.89% hard.
All five misses used syntactically valid tool calls but made semantic
grounding/reduction errors and ended with wrong answers. There were no
correct-answer-through-invalid-trace cases, parser failures, or budget
exhaustions. Detailed classifications are preserved in
`outputs/evaluations/qwen3-calc-grpo600-eval-greedy-a100-failure-analysis.json`.

## Held-out test

The selected GRPO revision was evaluated once on all 100 untouched test rows
using the same greedy 128-token, six-call protocol on an A100:

- Task success and answer accuracy: 98%.
- Valid tool use and parseable completion: 100%.
- Exact call count and semantic trace completion: 98%.
- Easy and medium task success: 100%.
- Hard task success: 94.44% (34 of 36).
- Budget exhaustion: 0%.

The two test failures were hard examples. `calculator_0209` omitted an
addition and stopped one call early. `calculator_0728` recomputed a consumed
product, then used the stale partial result. Both had valid tool syntax but
invalid semantic trajectories and incorrect answers.

## Reproducibility and limitations

- GRPO was trained for one epoch over the reserved 600 rows on an A100 using
  colocated vLLM. The final adapter is the only retained GRPO checkpoint.
- The 20-row fixed-seed environment/evaluator parity gate passed, including
  tool observations, final-answer parsing, valid complete traces, and rejection
  of omitted final calls. Its artifact is
  `outputs/evaluations/grpo-environment-parity.json`.
- Eval selected the checkpoint; test was not used for training, tuning, or
  checkpoint choice.
- Base and SFT were not rerun on the test split. Their figures above are the
  controlled eval baselines; the 98% test result is reported only for the
  preselected GRPO policy.
- One GRPO training seed and one greedy final-test decoding were run.
  Additional seeds or sampled decoding would estimate variance but are not
  required to support the selected deterministic deployment result.

## Artifact index

- Controlled base/SFT comparison:
  `outputs/evaluations/qwen3-controlled-a100-comparison.json`
- GRPO eval:
  `outputs/evaluations/qwen3-calc-grpo600-eval-greedy-a100-*`
- GRPO held-out test:
  `outputs/evaluations/qwen3-calc-grpo600-test-greedy-a100-*`
- GRPO Hub revision:
  `outputs/notebooks/qwen3-calc-grpo600-full-vllm-a100-hub_revision.json`
