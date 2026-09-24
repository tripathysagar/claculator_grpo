"""Day 3: Breakpoint analysis — where and how quantization kills accuracy."""

import json
from collections import Counter
from pathlib import Path
from statistics import mean

BASE = Path.home() / "experiments/gguf-quant-eval"
QUANTS = ["Q8_0", "Q4_K_M", "Q3_K_M", "Q2_K"]


def load_preds(quant):
    path = BASE / quant / "predictions.jsonl"
    return {
        r["id"]: r
        for line in path.read_text().splitlines()
        if line.strip()
        for r in [json.loads(line)]
    }


def main():
    data = {q: load_preds(q) for q in QUANTS}
    ids = sorted(data["Q8_0"].keys())

    # ── 1. Progression table: which examples survive each quant ──
    print("=" * 70)
    print("1. SURVIVAL PROGRESSION (task_success per example across quants)")
    print("=" * 70)
    for tier in ("easy", "medium", "hard"):
        tier_ids = [eid for eid in ids if data["Q8_0"][eid]["tier"] == tier]
        print(f"\n  {tier.upper()} (n={len(tier_ids)}):")
        for q in QUANTS:
            rate = mean(float(data[q][eid]["task_success"]) for eid in tier_ids)
            print(f"    {q:<10} task_success={rate:.3f}")

    # ── 2. Q8 pass → Q4 fail: what breaks first ──
    print("\n" + "=" * 70)
    print("2. EXAMPLES THAT PASS Q8 BUT FAIL Q4_K_M")
    print("=" * 70)
    q8_pass_q4_fail = []
    for eid in ids:
        q8 = data["Q8_0"][eid]
        q4 = data["Q4_K_M"][eid]
        if q8["task_success"] and not q4["task_success"]:
            q8_pass_q4_fail.append(q4)

    print(f"Count: {len(q8_pass_q4_fail)}")
    print(f"By tier: {dict(Counter(e['tier'] for e in q8_pass_q4_fail))}")
    print(f"By op count: {dict(sorted(Counter(e['expected_tool_call_count'] for e in q8_pass_q4_fail).items()))}")

    # Failure modes at Q4
    print("\nFailure modes:")
    for e in sorted(q8_pass_q4_fail, key=lambda x: x["expected_tool_call_count"]):
        ans_ok = "correct" if e["answer_correct"] else "wrong"
        trace_ok = "valid" if e["semantic_trace_valid"] else "invalid"
        tc = f"{e['tool_call_count']}/{e['expected_tool_call_count']}"
        print(
            f"  [{e['tier']:>6}] ops={e['expected_tool_call_count']}  "
            f"{e['expression']:<30}  expected={e['final_answer']:<8}  "
            f"got={str(e['predicted_answer']):<8}  "
            f"ans={ans_ok:<7}  trace={trace_ok:<7}  calls={tc}"
        )

    # ── 3. Q3 failure modes ──
    print("\n" + "=" * 70)
    print("3. Q3_K_M FAILURE MODES")
    print("=" * 70)
    q3_errors = Counter(data["Q3_K_M"][eid].get("error") or "none" for eid in ids)
    for err, count in q3_errors.most_common():
        print(f"  {err:<30} {count:>3}")

    # Check what Q3 does with easy examples
    print("\nQ3 on easy examples:")
    for eid in ids:
        r = data["Q3_K_M"][eid]
        if r["tier"] == "easy":
            turns = r.get("generated_turns", [])
            first = turns[0][:120] if turns else "NO OUTPUT"
            print(f"  {r['expression']:<25}  error={r.get('error','none'):<25}  output={first}")

    # ── 4. Q2 failure mode ──
    print("\n" + "=" * 70)
    print("4. Q2_K FAILURE MODE")
    print("=" * 70)
    q2_errors = Counter(data["Q2_K"][eid].get("error") or "none" for eid in ids)
    for err, count in q2_errors.most_common():
        print(f"  {err:<30} {count:>3}")

    # ── 5. Answer-only analysis (ignoring trace validity) ──
    print("\n" + "=" * 70)
    print("5. ANSWER-ONLY ACCURACY (ignoring tool trace quality)")
    print("=" * 70)
    for q in QUANTS:
        by_tier = {}
        for tier in ("easy", "medium", "hard"):
            tier_ids = [eid for eid in ids if data[q][eid]["tier"] == tier]
            by_tier[tier] = mean(float(data[q][eid]["answer_correct"]) for eid in tier_ids)
        overall = mean(float(data[q][eid]["answer_correct"]) for eid in ids)
        print(
            f"  {q:<10}  overall={overall:.3f}  "
            f"easy={by_tier['easy']:.3f}  "
            f"medium={by_tier['medium']:.3f}  "
            f"hard={by_tier['hard']:.3f}"
        )

    # ── 6. Hard-tier deep dive: does Q4 fail on known-hard patterns? ──
    print("\n" + "=" * 70)
    print("6. HARD-TIER DEEP DIVE: Q4_K_M failures")
    print("=" * 70)
    hard_ids = [eid for eid in ids if data["Q8_0"][eid]["tier"] == "hard"]
    q4_hard_fail = [
        data["Q4_K_M"][eid]
        for eid in hard_ids
        if not data["Q4_K_M"][eid]["task_success"]
    ]
    print(f"Hard examples failed at Q4: {len(q4_hard_fail)}/{len(hard_ids)}")
    print(f"Op count distribution of failures: {dict(sorted(Counter(e['expected_tool_call_count'] for e in q4_hard_fail).items()))}")

    # Check if failures cluster by expression complexity
    for e in sorted(q4_hard_fail, key=lambda x: -x["expected_tool_call_count"]):
        q8 = data["Q8_0"][e["id"]]
        print(
            f"  ops={e['expected_tool_call_count']}  {e['expression']:<35}  "
            f"Q8={'PASS' if q8['task_success'] else 'FAIL'}  "
            f"Q4_pred={e['predicted_answer']}  expected={e['final_answer']}  "
            f"calls={e['tool_call_count']}/{e['expected_tool_call_count']}"
        )


if __name__ == "__main__":
    main()
