def _save_baseline_vs_audit_comparison(results: list[dict], output_dir: str):
    """Saves a clear comparison between the baseline proposer and the final audit loop."""
    import os
    from utils import answers_match
    
    filepath = os.path.join(output_dir, "baseline_vs_audit_comparison.md")
    baseline_correct = 0
    audit_correct = 0
    
    with open(filepath, "w") as f:
        f.write("# Baseline vs Audit Loop Comparison\n\n")
        
        for r in results:
            gold = r["gold_answer"]
            baseline_ans = r["steps"][0]["answer"] 
            final_ans = r["final_answer"]
            
            is_baseline_correct = answers_match(baseline_ans, gold)
            is_audit_correct = answers_match(final_ans, gold)
            
            if is_baseline_correct: baseline_correct += 1
            if is_audit_correct: audit_correct += 1
                
            f.write(f"## Example {r.get('example_idx', '?')}\n")
            f.write(f"**Question:** {r['question']}\n\n")
            f.write(f"- **Gold Answer:** {gold}\n")
            f.write(f"- **Baseline (No Audit):** {baseline_ans} ({'✅' if is_baseline_correct else '❌'})\n")
            f.write(f"- **Final (Audit Loop):** {final_ans} ({'✅' if is_audit_correct else '❌'}) | Steps: {r['num_steps']}\n\n")
            f.write("---\n")
            
        f.write(f"\n# Summary\n")
        f.write(f"- Baseline Accuracy: {round((baseline_correct/len(results))*100, 2)}%\n")
        f.write(f"- Audit Loop Accuracy: {round((audit_correct/len(results))*100, 2)}%\n")