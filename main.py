#!/usr/bin/env python3
"""
Main entry point for the Iterative Audit Loop pipeline.

Usage:
    python main.py --num_examples 50 --output_dir results/

This runs the full pipeline:
    1. Load GSM8K test examples
    2. For each example, run the iterative audit loop (Model A → B → A → B)
    3. At each step, sample 5 times for majority-vote uncertainty
    4. Evaluate and report results
"""

import argparse
import json
import sys
import time

from tqdm import tqdm

from config import NUM_EXAMPLES, OUTPUT_DIR, GSM8K_SPLIT
from data_loader import load_gsm8k
from ollama_client import check_ollama_ready
from audit_loop import run_audit_loop
from evaluation import evaluate_results

def main():
    parser = argparse.ArgumentParser(
        description="Iterative Audit Loops with Language Models (GSM8K)"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=NUM_EXAMPLES,
        help=f"Number of GSM8K examples to evaluate (default: {NUM_EXAMPLES})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Directory to save results (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=GSM8K_SPLIT,
        help=f"GSM8K split to use (default: {GSM8K_SPLIT})",
    )
    args = parser.parse_args()

    # ── Preflight checks ─────────────────────────────────────────
    print("🔍 Checking Ollama availability...")
    if not check_ollama_ready():
        print("❌ Ollama is not ready. Please start it and pull the model.")
        sys.exit(1)
    print("✅ Ollama is ready.\n")

    # ── Load data ─────────────────────────────────────────────────
    print(f"📚 Loading GSM8K ({args.split}) — {args.num_examples} examples...")
    examples = load_gsm8k(split=args.split, num_examples=args.num_examples)
    print(f"   Loaded {len(examples)} examples.\n")

    # ── Run audit loops ──────────────────────────────────────────
    print("🔄 Running iterative audit loops...\n")
    results = []
    start_time = time.time()

    for i, example in enumerate(tqdm(examples, desc="Processing")):
        result = run_audit_loop(question=example["question"])
        result["gold_answer"] = example["answer"]
        result["example_idx"] = i
        results.append(result)

        # Progress update every 10 examples
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(examples) - i - 1) / rate
            tqdm.write(
                f"   [{i+1}/{len(examples)}] "
                f"Elapsed: {elapsed:.0f}s | "
                f"ETA: {remaining:.0f}s | "
                f"Rate: {rate:.2f} ex/s"
            )

    total_time = time.time() - start_time
    print(f"\n⏱  Total time: {total_time:.1f}s "
          f"({total_time/len(examples):.1f}s per example)\n")

    # ── Evaluate and report ──────────────────────────────────────
    summary = evaluate_results(results, output_dir=args.output_dir)

    return summary

if __name__ == "__main__":
    main()