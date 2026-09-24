**Agenda: quantize + characterize `qwen3-0.6b-calc-grpo600`, ship as a public artifact.**

**Goal:** answer "how much of this GRPO result survives running on hardware someone actually owns, and at what speed/accuracy tradeoff" — turn a finished training result into a deployable, demo-able artifact instead of a closed experiment.

**Day 1** — Export the model to GGUF, quantize at multiple bit-widths (Q8, Q4_K_M, Q3, optionally Q2) via llama.cpp. Get every variant loading and generating on the MSI (CUDA path — not the Mac, keep to one pipeline).

**Day 2** — Re-run the existing held-out eval set (reused as-is, not regenerated) against each quantized variant. Log task_success and tokens/sec per bit-width. Delegate this export→quantize→eval→log loop to a coding agent as one closed, verifiable task, with the eval script and eval set pinned explicitly.

**Day 3** — Find the breakpoint: which bit-width is where accuracy starts degrading, and specifically whether it hits your known hard-tier failure cases first. This interpretation step is yours, not the agent's.

**Day 4** — Package and ship:
- Push the model (best-tradeoff quantized variant, or a couple of variants) to Hugging Face with a proper model card
- Small HF Space with a live demo people can poke at
- Short write-up of the bit-width/accuracy/speed curve and the reward-design pattern behind it
- Post it yourself + a handful of direct outreach messages to people at target companies — the build is bait, distribution is what gets callbacks

That's the full loop: finished GRPO result → quantized → benchmarked on your own hardware → public, usable artifact → outreach.

---

## Pre-req Setup (completed 2026-09-24)

### Hardware — MSI / WSL2

| Component | Detail |
|---|---|
| CPU | AMD Ryzen 7 3750H (4c/8t physical; WSL2 sees 3c/6t) |
| GPU | Radeon RX 5500M (4GB) + Vega 10 iGPU |
| RAM | 8GB total (WSL gets ~7.8GB) |
| Disk | 915GB free |
| OS | Ubuntu 24.04 LTS on WSL2 |
| Inference path | **CPU-only** — RX 5500M is RDNA1, no ROCm support in WSL2 |

### Connectivity

- SSH via `ssh wsl` (host `sagar.local`, port 2222, key `~/.ssh/id_ed25519`)
- HF token set in WSL `~/.bashrc` (`HF_TOKEN`), authenticated as `tripathysagar`

### Software installed (all userspace, no sudo)

| Tool | Version | Location |
|---|---|---|
| Python | 3.13.7 | `~/miniconda3/bin/python` |
| PyTorch | 2.8.0+cu128 | pip (CUDA not used, CPU fallback) |
| transformers | 5.9.0 | pip |
| peft | latest | pip |
| accelerate | latest | pip |
| safetensors | 0.7.0 | pip |
| huggingface-hub | 1.15.0 | pip |
| gguf | 0.19.0 | pip |
| cmake | 4.4.3 | pip |
| g++ | 15.2.0 (conda-forge) | `~/miniconda3/bin/g++` |
| make | 4.4.1 | `~/miniconda3/bin/make` |
| llama.cpp | 0.5.0-dev (commit 013b31c, shallow clone) | `~/git/llama.cpp/build/bin/` |

### Models on disk (`~/models/`)

| Path | What | Size |
|---|---|---|
| `Qwen3-0.6B/` | Base model (HF safetensors) | 1.5GB |
| `qwen3-0.6b-calc-grpo600/` | LoRA adapter from HF | 50MB |
| `qwen3-0.6b-calc-grpo600-merged/` | LoRA merged into base (fp16 safetensors) | 1.2GB |
| `qwen3-0.6b-calc-grpo600.f16.gguf` | GGUF fp16 (conversion source) | 1.2GB |
| `qwen3-0.6b-calc-grpo600.Q8_0.gguf` | Quantized Q8_0 | 610MB |
| `qwen3-0.6b-calc-grpo600.Q4_K_M.gguf` | Quantized Q4_K_M | 379MB |
| `qwen3-0.6b-calc-grpo600.Q3_K_M.gguf` | Quantized Q3_K_M | 332MB |
| `qwen3-0.6b-calc-grpo600.Q2_K.gguf` | Quantized Q2_K | 283MB |

### Smoke test result

- **Model**: Q4_K_M
- **Prompt**: `Calculate 15 + 27`
- **Output**: `[Start thinking] 27 + 15 [End thinking] The answer is 42.` ✅
- **Speed**: Prompt 20.0 t/s, Generation 21.3 t/s (CPU, 4 threads)

### Notes

- Agenda said "CUDA path" — updated to CPU-only since MSI has AMD GPU (RX 5500M RDNA1, no ROCm in WSL2)
- No sudo access on WSL; all tooling installed via miniconda/pip in userspace
- WSL default RAM was 5.8GB, bumped to 7.8GB after restart; consider adding `~/.wslconfig` with `memory=6GB` if OOM hits during eval
- For 0.6B model, CPU inference is adequate (~21 t/s) — GPU acceleration not critical at this scale

---

## Day 1 Results (completed 2026-09-24)

All 4 quantized variants exported, loaded, and generated on MSI via llama.cpp CPU path.

### Simple arithmetic: `Calculate 15 + 27` (answer: 42)

| Quant | Output | Correct | Prompt t/s | Gen t/s |
|---|---|---|---|---|
| Q8_0 | `[Start thinking] The answer is 42. [End thinking] The answer is 42.` | ✅ | 46.3 | 24.4 |
| Q4_K_M | `[Start thinking] The answer is 42. [End thinking] The answer is 42.` | ✅ | 65.9 | 28.8 |
| Q3_K_M | `[Start thinking] 15 + 27 is 42. [End thinking] The answer is 42.` | ✅ | 49.5 | 25.2 |
| Q2_K | `15 + 27 is 15 + 27` (degenerate loop) | ❌ | 54.1 | 27.0 |

### Compound arithmetic: `Calculate (3.5 * 8) - (12 / 4)` (answer: 25)

| Quant | Output | Correct |
|---|---|---|
| Q8_0 | 24 | ❌ (off by 1) |
| Q4_K_M | -25 (computed 28 and 3 correctly, subtracted wrong order) | ❌ |
| Q3_K_M | 3.5 | ❌ |
| Q2_K | gibberish (counting 2,3,4,5,6) | ❌ |

### Notes on Day 1 smoke tests

These are raw-prompt completions (no tool-use protocol, no calculator tool, no multi-turn loop). The compound-arithmetic failures — including Q8 — reflect the model doing mental math without its tool, not quantization damage. **Day 2's systematic eval (with the full tool-use harness) is the real benchmark**: Q8 scores 94% task_success there. Don't read the Day 1 anecdote as contradicting that — it's a different protocol.

- **Q2_K is broken** — degenerate output even on trivial prompts, unusable
- Thinking-tag structure (`[Start thinking]...[End thinking]`) preserved in Q8/Q4/Q3, lost in Q2
- Speed roughly comparable across quants (~25 t/s gen), Q4_K_M slightly fastest (smaller model, more cache-friendly)

---

## Day 2 Results (completed 2026-09-24)

Full held-out eval (100 examples, greedy decoding) against all 4 quantized GGUF variants via `llama-server` + multi-turn tool-use harness. Script: `scripts/eval_gguf_quants.py`. Results: `~/experiments/gguf-quant-eval/` on WSL.

### Aggregate task_success by quant

| Quant | Size | task_success | answer_correct | valid_tool_use | t/s | wall time |
|---|---|---|---|---|---|---|
| **Q8_0** | 639 MB | **0.940** | 0.940 | 1.000 | 15.5 | 23 min |
| **Q4_K_M** | 397 MB | **0.720** | 0.790 | 1.000 | 18.2 | 19 min |
| Q3_K_M | 347 MB | 0.040 | 0.040 | 0.200 | 17.0 | 9 min* |
| Q2_K | 296 MB | 0.000 | 0.000 | 0.000 | 21.6 | 20 min |

\* Q3's 9-min wall time is not a script bug. 76/100 examples hit `missing_final_answer` on turn 1 (model emits `<tool>` instead of `<tool_call>`, harness can't parse it, example ends). Those examples average 49 tokens / 2.7s vs. Q8's 214 tokens / 13.8s per example. The 16 examples that Q3 *does* parse run full multi-turn loops at normal speed (~15s each). Total tokens generated: Q3 = 9,506 vs Q8 = 21,364 — genuinely half the work.

### task_success by tier

| Quant | Easy (n=11) | Medium (n=53) | Hard (n=36) |
|---|---|---|---|
| Q8_0 | 1.000 | 0.962 | 0.889 |
| Q4_K_M | 1.000 | 0.755 | 0.583 |
| Q3_K_M | 0.000 | 0.019 | 0.083 |
| Q2_K | 0.000 | 0.000 | 0.000 |

### Error breakdown

- **Q8_0**: zero errors — every example parseable, valid tool use, clean traces
- **Q4_K_M**: zero errors — all parseable, but wrong answers/traces on 28% of examples
- **Q3_K_M**: 76% `missing_final_answer`, 3% `wrong_schema`, 2% `invalid_operator`, 2% `max_tool_calls_exceeded`, 1% `thinking_budget_exhausted` — model can barely produce structured output
- **Q2_K**: 100% `thinking_budget_exhausted` — model loops in thinking tokens, never produces tool calls or answers

### Key findings

1. **Breakpoint is between Q4_K_M and Q3_K_M** — accuracy cliff from 72% → 4%
2. **Q8_0 retains nearly full capability** (94% vs fp16 baseline) — negligible quantization loss
3. **Q4_K_M is deployable** — 72% task_success, 100% valid tool use, 38% smaller than Q8
4. **The degradation hits tool-use structure first** — Q3 can't even format valid tool calls (only 20% valid_tool_use), not just wrong answers
5. **Hard tier degrades fastest** — Q4_K_M drops from 100% (easy) → 58% (hard), matching the known failure pattern from GRPO training
6. **Speed inversely correlates with quality** — smaller models generate faster but produce garbage (Q2_K fastest at 21.6 t/s but 0% accuracy)

---

## Day 3 — Breakpoint Analysis (completed 2026-09-24)

Analysis script: `scripts/analyze_quant_breakpoint.py`

### Where the cliff is

**Between Q4_K_M and Q3_K_M.** Not gradual — a wall.

| Quant | Easy | Medium | Hard |
|---|---|---|---|
| Q8_0 | 1.000 | 0.962 | 0.889 |
| Q4_K_M | 1.000 | 0.755 | 0.583 |
| Q3_K_M | **0.000** | 0.019 | 0.083 |
| Q2_K | 0.000 | 0.000 | 0.000 |

Q3 can't solve even easy 2-op problems. Not a precision issue — structural collapse.

### What breaks at Q3: the tool-call format itself

Q3 on every easy example — model *thinks* correctly but can't format the tool call:

```
<think>
The next precedence operation is 9*16.
</think>
<tool>
{"name": "calculator", "arguments": {"op": "*", 9, 16}}
```

Three fatal problems: (1) `<tool>` instead of `<tool_call>`, (2) JSON values without keys, (3) inconsistent brackets. Reasoning survives quantization but exact token sequences for structured output don't. 76/100 errors = `missing_final_answer`.

### What breaks at Q4_K_M: only hard-tier, only 5-op expressions

24 examples pass Q8 but fail Q4. Distribution:
- **0 easy** failures
- **12 medium** (ops=3-4)
- **12 hard** — **all ops=5, every single one**

### How Q4 fails: two modes

**Mode A — Premature stop (14/24):** Fewer tool calls than needed, partial answer. E.g. `7 - 14*3*17` → 2/3 calls, echoes first term.

**Mode B — Wrong execution order (10/24):** Right call count but wrong order (precedence violations). E.g. `10 - 8 - 6*15` → expected -88, got 92.

### Q2_K: total collapse

100% `thinking_budget_exhausted` — model loops in `<think>` tokens, never exits thinking mode. Quantization destroyed the think→act transition.

### Does it hit hard-tier failure cases first?

**Yes, exactly.** Degradation mirrors the GRPO training difficulty curve:
1. Easy survives to Q4 (100%) — patterns GRPO nailed hardest
2. Medium starts failing at Q4 (75.5%) — multi-term precedence gets fragile
3. Hard fails most at Q4 (58.3%) — **every failure is a 5-op expression**
4. At Q3, even easy collapses — structural output format destroyed

The quantization breakpoint directly mirrors the GRPO reward signal: learned behaviors requiring the most training signal (tool-call formatting, multi-step chains) degrade first under bit-width reduction.