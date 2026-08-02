"""
Evaluation and reporting — now with detailed CoT trace output.
"""

import json
import os
from collections import defaultdict

import numpy as np

from utils import answers_match

def evaluate_results(results: list[dict], output_dir: str) -> dict:
    """
    Compute all required metrics and save reports including CoT traces.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Accuracy per step ─────────────────────────────────────
    step_correct = defaultdict(list)

    for r in results:
        gold = r["gold_answer"]
        for step_data in r["steps"]:
            step_idx = step_data["step"]
            ans = step_data["uncertainty"]["majority_answer"] or step_data["answer"]
            is_correct = answers_match(ans, gold)
            step_correct[step_idx].append(is_correct)

    accuracy_per_step = {}
    for step_idx in sorted(step_correct.keys()):
        correct_list = step_correct[step_idx]
        acc = sum(correct_list) / len(correct_list) if correct_list else 0.0
        accuracy_per_step[step_idx] = {
            "accuracy": round(acc * 100, 2),
            "num_examples": len(correct_list),
            "num_correct": sum(correct_list),
        }

    # ── 2. Average number of calls ───────────────────────────────
    total_calls_list = [r["total_calls"] for r in results]
    num_steps_list = [r["num_steps"] for r in results]
    avg_calls = np.mean(total_calls_list)
    avg_steps = np.mean(num_steps_list)
    early_stop_rate = sum(1 for r in results if r["stopped_early"]) / len(results)

    # ── 3. Final accuracy ────────────────────────────────────────
    final_correct = []
    for r in results:
        gold = r["gold_answer"]
        is_correct = answers_match(r["final_answer"], gold)
        final_correct.append(is_correct)
    final_accuracy = sum(final_correct) / len(final_correct) if final_correct else 0.0

    # ── 4. False-certainty cases ─────────────────────────────────
    false_certainty = []
    for r in results:
        gold = r["gold_answer"]
        last_step = r["steps"][-1]
        agreement = last_step["uncertainty"]["agreement"]
        final_ans = r["final_answer"]
        is_correct = answers_match(final_ans, gold)

        if agreement >= 0.8 and not is_correct:
            false_certainty.append({
                "question": r["question"],
                "final_answer": final_ans,
                "gold_answer": gold,
                "agreement": agreement,
                "num_steps": r["num_steps"],
                "chain_of_thought": r.get("chain_of_thought", []),
            })

    # ── 5. Uncertainty drops but still wrong ─────────────────────
    uncertainty_drop_wrong = []
    for r in results:
        gold = r["gold_answer"]
        steps = r["steps"]
        if len(steps) < 2:
            continue

        first_agreement = steps[0]["uncertainty"]["agreement"]
        last_agreement = steps[-1]["uncertainty"]["agreement"]
        final_ans = r["final_answer"]
        is_correct = answers_match(final_ans, gold)

        if last_agreement > first_agreement and not is_correct:
            uncertainty_drop_wrong.append({
                "question": r["question"],
                "final_answer": final_ans,
                "gold_answer": gold,
                "agreement_first_step": first_agreement,
                "agreement_last_step": last_agreement,
                "num_steps": r["num_steps"],
                "chain_of_thought": r.get("chain_of_thought", []),
            })

    # ── Build summary ────────────────────────────────────────────
    summary = {
        "num_examples": len(results),
        "final_accuracy_pct": round(final_accuracy * 100, 2),
        "accuracy_per_step": accuracy_per_step,
        "avg_total_calls": round(avg_calls, 2),
        "avg_num_steps": round(avg_steps, 2),
        "early_stop_rate_pct": round(early_stop_rate * 100, 2),
        "false_certainty_count": len(false_certainty),
        "uncertainty_drop_but_wrong_count": len(uncertainty_drop_wrong),
    }

    # ── Save outputs ─────────────────────────────────────────────

    # 1. Summary (same as before)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 2. Full results with CoT
    with open(os.path.join(output_dir, "full_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # 3. Error analysis files with CoT
    with open(os.path.join(output_dir, "false_certainty_cases.json"), "w") as f:
        json.dump(false_certainty, f, indent=2)

    with open(os.path.join(output_dir, "uncertainty_drop_wrong.json"), "w") as f:
        json.dump(uncertainty_drop_wrong, f, indent=2)

    # ── 6. NEW: Detailed CoT trace file (human-readable) ────────
    _save_cot_report(results, output_dir)

    # ── Print report ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ITERATIVE AUDIT LOOP — EVALUATION REPORT")
    print("=" * 65)
    print(f"  Examples evaluated:        {len(results)}")
    print(f"  Final accuracy:            {summary['final_accuracy_pct']}%")
    print(f"  Avg total LLM calls:       {summary['avg_total_calls']}")
    print(f"  Avg audit steps:           {summary['avg_num_steps']}")
    print(f"  Early stop rate:           {summary['early_stop_rate_pct']}%")
    print("-" * 65)
    print("  Accuracy per step:")
    for step_idx, info in accuracy_per_step.items():
        role = "Proposer " if step_idx == 0 else f"Auditor {'B' if step_idx % 2 == 1 else 'A'}"
        print(f"    Step {step_idx} ({role}): "
              f"{info['accuracy']}%  ({info['num_correct']}/{info['num_examples']})")
    print("-" * 65)
    print(f"  False-certainty cases:     {len(false_certainty)}")
    print(f"  Uncertainty ↓ but wrong:   {len(uncertainty_drop_wrong)}")
    print("=" * 65)
    print(f"\n  📄 CoT traces saved to:    {output_dir}/cot_traces.md")
    print(f"  📄 Full results saved to:  {output_dir}/full_results.json")
    print()

    return summary

def _save_cot_report(results: list[dict], output_dir: str):
    """
    Save a human-readable Markdown file with full CoT traces for every
    example at every step.
    """
    filepath = os.path.join(output_dir, "cot_traces.md")

    with open(filepath, "w") as f:
        f.write("# Chain-of-Thought Traces — Iterative Audit Loop\n\n")
        f.write("---\n\n")

        for r in results:
            idx = r.get("example_idx", "?")
            gold = r["gold_answer"]
            final = r["final_answer"]
            correct = "✅" if answers_match(final, gold) else "❌"
            stopped = "Yes" if r["stopped_early"] else "No"

            f.write(f"## Example {idx}\n\n")
            f.write(f"**Question:** {r['question']}\n\n")
            f.write(f"**Gold Answer:** {gold}\n\n")
            f.write(f"**Final Answer:** {final} {correct}\n\n")
            f.write(f"**Steps Used:** {r['num_steps']} | "
                    f"**Early Stop:** {stopped} | "
                    f"**Total Calls:** {r['total_calls']}\n\n")

            # ── CoT for each step ────────────────────────────────
            cot_list = r.get("chain_of_thought", [])
            for cot in cot_list:
                step = cot["step"]
                role = cot["role"]
                answer = cot["extracted_answer"]

                f.write(f"### Step {step} — {role.replace('_', ' ').title()}\n\n")

                # Main CoT response
                f.write(f"**Extracted Answer:** `{answer}`\n\n")
                f.write(f"<details>\n")
                f.write(f"<summary>📝 Full Chain of Thought (click to expand)</summary>\n\n")
                f.write(f"```\n{cot['chain_of_thought']}\n```\n\n")
                f.write(f"</details>\n\n")

                # Uncertainty samples
                samples = cot.get("uncertainty_samples", [])
                if samples:
                    sample_answers = [s["extracted_answer"] for s in samples]
                    f.write(f"**Uncertainty Samples:** {sample_answers}\n\n")

                    f.write(f"<details>\n")
                    f.write(f"<summary>🎲 Uncertainty Sample CoTs "
                            f"({len(samples)} samples, click to expand)</summary>\n\n")
                    for s in samples:
                        f.write(f"**Sample {s['sample_idx']}** → `{s['extracted_answer']}`\n\n")
                        f.write(f"```\n{s['chain_of_thought']}\n```\n\n")
                    f.write(f"</details>\n\n")

            f.write("---\n\n")