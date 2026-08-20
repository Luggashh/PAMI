"""
Scoring and reporting for the iterative audit loop.

Everything is written into the output directory (default `results/`):

    solutions.jsonl                   one compact record per solved problem
    per_example.csv                   flat table for spreadsheets / plotting
    summary.json                      all aggregate metrics
    summary.md                        human-readable report
    baseline_vs_audit_comparison.md   side-by-side per-example comparison
"""

import csv
import json
import os

from config import AGREEMENT_THRESHOLD, MAX_STEPS
from utils import answers_match


def evaluate_results(results: list[dict], output_dir: str) -> dict:
    """
    Score every solved problem, write all report files, print a summary.

    Returns the summary dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not results:
        print("⚠️  No results to evaluate.")
        return {}

    results = sorted(results, key=lambda r: r.get("example_idx", 0))
    rows = [_score_example(r) for r in results]

    summary = _aggregate(rows)

    _save_solutions(rows, output_dir)
    _save_per_example_csv(rows, output_dir)
    _save_summary_json(summary, output_dir)
    _save_summary_md(summary, output_dir)
    _save_baseline_vs_audit_comparison(results, output_dir)

    _print_summary(summary, output_dir)
    return summary


# ── Per-example scoring ──────────────────────────────────────────


def _score_example(result: dict) -> dict:
    """Flatten one audit-loop result into a scored record."""
    gold = result["gold_answer"]
    steps = result.get("steps", [])

    step_answers = [s["answer"] for s in steps]
    step_correct = [answers_match(a, gold) for a in step_answers]
    step_agreement = [s["uncertainty"]["agreement"] for s in steps]

    final_answer = result.get("final_answer")
    final_correct = answers_match(final_answer, gold)

    baseline_answer = step_answers[0] if step_answers else None
    baseline_correct = step_correct[0] if step_correct else False

    first_agreement = step_agreement[0] if step_agreement else 0.0
    last_agreement = step_agreement[-1] if step_agreement else 0.0
    last_agreement_count = steps[-1]["uncertainty"]["agreement_count"] if steps else 0

    return {
        "example_idx": result.get("example_idx"),
        "question": result["question"],
        "gold_answer": gold,
        "baseline_answer": baseline_answer,
        "baseline_correct": baseline_correct,
        "final_answer": final_answer,
        "final_correct": final_correct,
        "num_steps": result.get("num_steps", len(steps)),
        "total_calls": result.get("total_calls", 0),
        "stopped_early": result.get("stopped_early", False),
        "step_answers": step_answers,
        "step_correct": step_correct,
        "step_agreement": step_agreement,
        "first_agreement": first_agreement,
        "final_agreement": last_agreement,
        "final_agreement_count": last_agreement_count,
        # Confident but wrong: the loop met the stopping threshold on an
        # incorrect answer.
        "false_certainty": last_agreement_count >= AGREEMENT_THRESHOLD and not final_correct,
        # Uncertainty fell over the loop, yet the answer is still wrong.
        "uncertainty_dropped_but_wrong": last_agreement > first_agreement and not final_correct,
        "audit_fixed": (not baseline_correct) and final_correct,
        "audit_broke": baseline_correct and (not final_correct),
    }


# ── Aggregation ──────────────────────────────────────────────────


def _pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def _aggregate(rows: list[dict]) -> dict:
    n = len(rows)

    accuracy_by_step = []
    for step_idx in range(MAX_STEPS):
        reached = [r for r in rows if len(r["step_correct"]) > step_idx]
        correct = sum(1 for r in reached if r["step_correct"][step_idx])
        accuracy_by_step.append(
            {
                "step": step_idx,
                "examples_reaching_step": len(reached),
                "correct": correct,
                "accuracy_pct": _pct(correct, len(reached)),
            }
        )

    mean_agreement_by_step = []
    for step_idx in range(MAX_STEPS):
        vals = [r["step_agreement"][step_idx] for r in rows if len(r["step_agreement"]) > step_idx]
        mean_agreement_by_step.append(
            {
                "step": step_idx,
                "n": len(vals),
                "mean_agreement": round(sum(vals) / len(vals), 4) if vals else 0.0,
            }
        )

    false_certainty = [r for r in rows if r["false_certainty"]]
    dropped_but_wrong = [r for r in rows if r["uncertainty_dropped_but_wrong"]]

    return {
        "num_examples": n,
        "baseline_accuracy_pct": _pct(sum(1 for r in rows if r["baseline_correct"]), n),
        "audit_loop_accuracy_pct": _pct(sum(1 for r in rows if r["final_correct"]), n),
        "baseline_correct": sum(1 for r in rows if r["baseline_correct"]),
        "final_correct": sum(1 for r in rows if r["final_correct"]),
        "accuracy_by_step": accuracy_by_step,
        "mean_agreement_by_step": mean_agreement_by_step,
        "avg_calls_per_example": round(sum(r["total_calls"] for r in rows) / n, 2),
        "total_calls": sum(r["total_calls"] for r in rows),
        "avg_steps_per_example": round(sum(r["num_steps"] for r in rows) / n, 2),
        "early_stop_pct": _pct(sum(1 for r in rows if r["stopped_early"]), n),
        "audit_fixed": sum(1 for r in rows if r["audit_fixed"]),
        "audit_broke": sum(1 for r in rows if r["audit_broke"]),
        "false_certainty_count": len(false_certainty),
        "false_certainty_pct": _pct(len(false_certainty), n),
        "false_certainty_examples": [r["example_idx"] for r in false_certainty],
        "uncertainty_dropped_but_wrong_count": len(dropped_but_wrong),
        "uncertainty_dropped_but_wrong_pct": _pct(len(dropped_but_wrong), n),
        "uncertainty_dropped_but_wrong_examples": [r["example_idx"] for r in dropped_but_wrong],
    }


# ── Writers ──────────────────────────────────────────────────────


def _save_solutions(rows: list[dict], output_dir: str) -> None:
    """One compact JSON record per solved problem."""
    path = os.path.join(output_dir, "solutions.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _save_per_example_csv(rows: list[dict], output_dir: str) -> None:
    path = os.path.join(output_dir, "per_example.csv")
    fields = [
        "example_idx",
        "gold_answer",
        "baseline_answer",
        "baseline_correct",
        "final_answer",
        "final_correct",
        "num_steps",
        "total_calls",
        "stopped_early",
        "first_agreement",
        "final_agreement",
        "false_certainty",
        "uncertainty_dropped_but_wrong",
        "audit_fixed",
        "audit_broke",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_summary_json(summary: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _save_summary_md(summary: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Iterative Audit Loop — Results\n\n")
        f.write(f"Examples evaluated: **{summary['num_examples']}**\n\n")

        f.write("## Headline\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| Baseline accuracy (proposer only) | {summary['baseline_accuracy_pct']}% |\n")
        f.write(f"| Audit-loop accuracy (final answer) | {summary['audit_loop_accuracy_pct']}% |\n")
        f.write(f"| Fixed by auditing (wrong → right) | {summary['audit_fixed']} |\n")
        f.write(f"| Broken by auditing (right → wrong) | {summary['audit_broke']} |\n")
        f.write(f"| Avg. LLM calls per example | {summary['avg_calls_per_example']} |\n")
        f.write(f"| Avg. steps per example | {summary['avg_steps_per_example']} |\n")
        f.write(f"| Stopped early (high agreement) | {summary['early_stop_pct']}% |\n")
        f.write(f"| Total LLM calls | {summary['total_calls']} |\n\n")

        f.write("## Accuracy per step\n\n")
        f.write("| Step | Examples reaching step | Correct | Accuracy |\n|---|---|---|---|\n")
        for s in summary["accuracy_by_step"]:
            f.write(
                f"| {s['step']} | {s['examples_reaching_step']} | "
                f"{s['correct']} | {s['accuracy_pct']}% |\n"
            )
        f.write("\n")

        f.write("## Mean majority-vote agreement per step\n\n")
        f.write("| Step | n | Mean agreement |\n|---|---|---|\n")
        for s in summary["mean_agreement_by_step"]:
            f.write(f"| {s['step']} | {s['n']} | {s['mean_agreement']} |\n")
        f.write("\n")

        f.write("## Failure modes\n\n")
        f.write(
            f"- **False certainty** (agreement ≥ {AGREEMENT_THRESHOLD}/5 but answer wrong): "
            f"{summary['false_certainty_count']} ({summary['false_certainty_pct']}%)\n"
        )
        f.write(
            f"- **Uncertainty dropped but still wrong**: "
            f"{summary['uncertainty_dropped_but_wrong_count']} "
            f"({summary['uncertainty_dropped_but_wrong_pct']}%)\n\n"
        )
        f.write(
            f"False-certainty example indices: "
            f"{summary['false_certainty_examples'][:50]}"
            f"{' …' if len(summary['false_certainty_examples']) > 50 else ''}\n\n"
        )
        f.write(
            f"Uncertainty-dropped-but-wrong example indices: "
            f"{summary['uncertainty_dropped_but_wrong_examples'][:50]}"
            f"{' …' if len(summary['uncertainty_dropped_but_wrong_examples']) > 50 else ''}\n"
        )


def _save_baseline_vs_audit_comparison(results: list[dict], output_dir: str) -> None:
    """Saves a clear comparison between the baseline proposer and the final audit loop."""
    filepath = os.path.join(output_dir, "baseline_vs_audit_comparison.md")
    baseline_correct = 0
    audit_correct = 0

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# Baseline vs Audit Loop Comparison\n\n")

        for r in results:
            gold = r["gold_answer"]
            baseline_ans = r["steps"][0]["answer"] if r.get("steps") else None
            final_ans = r["final_answer"]

            is_baseline_correct = answers_match(baseline_ans, gold)
            is_audit_correct = answers_match(final_ans, gold)

            if is_baseline_correct:
                baseline_correct += 1
            if is_audit_correct:
                audit_correct += 1

            f.write(f"## Example {r.get('example_idx', '?')}\n")
            f.write(f"**Question:** {r['question']}\n\n")
            f.write(f"- **Gold Answer:** {gold}\n")
            f.write(f"- **Baseline (No Audit):** {baseline_ans} ({'✅' if is_baseline_correct else '❌'})\n")
            f.write(
                f"- **Final (Audit Loop):** {final_ans} "
                f"({'✅' if is_audit_correct else '❌'}) | Steps: {r['num_steps']}\n\n"
            )
            f.write("---\n")

        total = len(results)
        f.write("\n# Summary\n")
        f.write(f"- Baseline Accuracy: {_pct(baseline_correct, total)}%\n")
        f.write(f"- Audit Loop Accuracy: {_pct(audit_correct, total)}%\n")


def _print_summary(summary: dict, output_dir: str) -> None:
    print("\n" + "=" * 62)
    print("  RESULTS")
    print("=" * 62)
    print(f"  Examples:                 {summary['num_examples']}")
    print(f"  Baseline accuracy:        {summary['baseline_accuracy_pct']}%")
    print(f"  Audit-loop accuracy:      {summary['audit_loop_accuracy_pct']}%")
    print(f"  Fixed by auditing:        {summary['audit_fixed']}")
    print(f"  Broken by auditing:       {summary['audit_broke']}")
    print(f"  Avg. calls per example:   {summary['avg_calls_per_example']}")
    print(f"  Avg. steps per example:   {summary['avg_steps_per_example']}")
    print(f"  Early stops:              {summary['early_stop_pct']}%")
    print(f"  False certainty:          {summary['false_certainty_count']} "
          f"({summary['false_certainty_pct']}%)")
    print(f"  Uncertainty down, wrong:  {summary['uncertainty_dropped_but_wrong_count']} "
          f"({summary['uncertainty_dropped_but_wrong_pct']}%)")
    print("-" * 62)
    print("  Accuracy per step:")
    for s in summary["accuracy_by_step"]:
        print(f"    step {s['step']}: {s['accuracy_pct']:6.2f}%  "
              f"(n={s['examples_reaching_step']})")
    print("=" * 62)
    print(f"\n📁 All files written to: {os.path.abspath(output_dir)}")
