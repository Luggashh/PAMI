#!/usr/bin/env python3
"""
Main entry point for the Iterative Audit Loop pipeline.

Everything needed to run the experiment happens here:

    python main.py

That installs Ollama into the user's home directory if it is not already
available (no root privileges required), launches one Ollama server per GPU,
pulls llama3.2:3b, solves the GSM8K problems in parallel across the GPUs, and
writes all results into `results/`.

The run is checkpointed after every problem, so it can be interrupted and
restarted without losing work.
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# Progress output contains non-ASCII characters; make sure redirecting stdout
# to a log file cannot crash the run on a server with a non-UTF-8 locale.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from config import (
    GSM8K_SPLIT,
    NUM_EXAMPLES,
    NUM_PARALLEL_PER_SERVER,
    OUTPUT_DIR,
)
from audit_loop import run_audit_loop
from data_loader import load_gsm8k
from evaluation import evaluate_results
from ollama_client import check_ollama_ready, configure_endpoints
from ollama_setup import shutdown_servers, start_servers

CHECKPOINT_NAME = "all_results.jsonl"

_write_lock = threading.Lock()


# ── Checkpointing ────────────────────────────────────────────────


def load_checkpoint(path: str) -> dict[int, dict]:
    """Read previously completed examples, keyed by example_idx."""
    if not os.path.exists(path):
        return {}

    done: dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                done[record["example_idx"]] = record
            except (json.JSONDecodeError, KeyError):
                # A partially flushed final line from an interrupted run.
                print(f"   ⚠️  Skipping malformed checkpoint line {line_no}.")
    return done


def append_checkpoint(path: str, record: dict) -> None:
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


# ── Worker ───────────────────────────────────────────────────────


def solve_example(idx: int, example: dict) -> dict:
    result = run_audit_loop(question=example["question"])
    result["gold_answer"] = example["answer"]
    result["example_idx"] = idx
    return result


# ── Main ─────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterative Audit Loops with Language Models (GSM8K)"
    )
    parser.add_argument("--num_examples", type=int, default=NUM_EXAMPLES,
                        help=f"Number of GSM8K examples to evaluate (default: {NUM_EXAMPLES})")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help=f"Directory to save results (default: {OUTPUT_DIR})")
    parser.add_argument("--split", type=str, default=GSM8K_SPLIT,
                        help=f"GSM8K split to use (default: {GSM8K_SPLIT})")
    parser.add_argument("--workers", type=int, default=0,
                        help="Concurrent examples (default: servers × parallel-per-server)")
    parser.add_argument("--no_resume", action="store_true",
                        help="Ignore the existing checkpoint and start from scratch")
    parser.add_argument("--skip_setup", action="store_true",
                        help="Do not install/launch Ollama; use servers that are already running")
    return parser.parse_args()


def main() -> dict:
    args = parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    checkpoint_path = os.path.join(output_dir, CHECKPOINT_NAME)

    # ── 1. Bring up Ollama across the GPUs ──────────────────────────
    if args.skip_setup:
        print("⏭  Skipping Ollama setup (--skip_setup).\n")
        base_urls = None
    else:
        base_urls = start_servers(log_dir=log_dir)
        configure_endpoints(base_urls)

    # ── 2. Preflight check ──────────────────────────────────────────
    print("🔍 Checking Ollama availability...")
    if not check_ollama_ready():
        print("❌ Ollama is not ready. See the logs in "
              f"{log_dir} for details.")
        shutdown_servers()
        sys.exit(1)
    print("✅ Ollama is ready.\n")

    n_servers = len(base_urls) if base_urls else 1
    workers = args.workers or max(1, n_servers * NUM_PARALLEL_PER_SERVER)

    # ── 3. Load data ────────────────────────────────────────────────
    print(f"📚 Loading GSM8K ({args.split}) — {args.num_examples} examples...")
    examples = load_gsm8k(split=args.split, num_examples=args.num_examples)
    print(f"   Loaded {len(examples)} examples.\n")

    # ── 4. Resume from checkpoint ───────────────────────────────────
    if args.no_resume and os.path.exists(checkpoint_path):
        os.replace(checkpoint_path, checkpoint_path + ".bak")
        print(f"♻️  --no_resume: previous checkpoint moved to {checkpoint_path}.bak\n")

    done = load_checkpoint(checkpoint_path)
    done = {idx: rec for idx, rec in done.items() if idx < len(examples)}
    if done:
        print(f"♻️  Resuming: {len(done)} example(s) already solved.\n")

    pending = [(i, ex) for i, ex in enumerate(examples) if i not in done]
    results = list(done.values())

    # ── 5. Run audit loops in parallel across the GPUs ──────────────
    print(f"🔄 Running iterative audit loops "
          f"({len(pending)} to go, {workers} concurrent workers over "
          f"{n_servers} server(s))...\n")

    start_time = time.time()
    failures: list[tuple[int, str]] = []

    if pending:
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = {executor.submit(solve_example, i, ex): i for i, ex in pending}
        try:
            with tqdm(total=len(pending), desc="Solving", unit="problem") as bar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001 — one bad problem must not kill the run
                        failures.append((idx, str(exc)))
                        tqdm.write(f"   ❌ Example {idx} failed: {exc}")
                    else:
                        append_checkpoint(checkpoint_path, result)
                        results.append(result)
                    bar.update(1)
                    bar.set_postfix(done=len(results), failed=len(failures))
        except KeyboardInterrupt:
            print("\n\n⛔ Interrupted — cancelling remaining work and scoring "
                  "what has been solved so far...")
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=True)

    elapsed = time.time() - start_time
    if results:
        print(f"\n⏱  Wall-clock: {elapsed:.1f}s "
              f"({elapsed / max(len(pending), 1):.1f}s per newly solved problem)")
    if failures:
        print(f"⚠️  {len(failures)} example(s) failed and were not checkpointed; "
              f"re-run `python main.py` to retry just those.")

    # ── 6. Evaluate and report ──────────────────────────────────────
    summary = evaluate_results(results, output_dir=output_dir)
    print(f"   Full chain-of-thought traces: {checkpoint_path}")

    shutdown_servers()
    return summary


if __name__ == "__main__":
    main()
