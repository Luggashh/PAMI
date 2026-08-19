#!/usr/bin/env python3
import argparse
import json
import sys
import time
import subprocess
import os

from tqdm import tqdm
from config import NUM_EXAMPLES, OUTPUT_DIR, GSM8K_SPLIT, MODEL_NAME
from data_loader import load_gsm8k
from ollama_client import check_ollama_ready
from audit_loop import run_audit_loop
from evaluation import evaluate_results

def setup_ollama():
    """Starts Ollama in the background and ensures the model is pulled."""
    print("🚀 Starting Ollama server...")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3) 
    
    print(f"📥 Verifying model '{MODEL_NAME}' is pulled...")
    subprocess.run(["ollama", "pull", MODEL_NAME], check=True)

def main():
    parser = argparse.ArgumentParser(description="Iterative Audit Loops (GSM8K)")
    parser.add_argument("--num_examples", type=int, default=NUM_EXAMPLES)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--split", type=str, default=GSM8K_SPLIT)
    args = parser.parse_args()

    setup_ollama()

    print("🔍 Checking Ollama availability...")
    if not check_ollama_ready():
        print("❌ Ollama is not ready.")
        sys.exit(1)
    print("✅ Ollama is ready.\n")

    print(f"📚 Loading GSM8K ({args.split}) — {args.num_examples} examples...")
    examples = load_gsm8k(split=args.split, num_examples=args.num_examples)

    print("🔄 Running iterative audit loops...\n")
    results = []
    start_time = time.time()

    for i, example in enumerate(tqdm(examples, desc="Processing")):
        result = run_audit_loop(question=example["question"])
        result["gold_answer"] = example["answer"]
        result["example_idx"] = i
        results.append(result)

    total_time = time.time() - start_time
    print(f"\n⏱ Total time: {total_time:.1f}s\n")

    summary = evaluate_results(results, output_dir=args.output_dir)
    return summary

if __name__ == "__main__":
    main()