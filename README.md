# Project 5: Iterative Audit Loops with Language Models

**Goal:** Study whether iterative language-model auditing reduces uncertainty and improves correctness.

## Architecture

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Model A     │─────▶│  Model B     │─────▶│  Model A     │
│  (Proposer)  │      │  (Auditor)   │      │  (Re-Auditor)│
│  llama3.2:3b │      │  llama3.2:3b │      │  llama3.2:3b │
└─────────────┘      └─────────────┘      └─────────────┘
     Step 0              Step 1               Step 2 ...
```

The loop continues until:
- **4 total answers** have been produced, OR
- **Numerical-answer agreement is high** (majority vote ≥ 4/5 across uncertainty samples)

## Running it

```bash
pip install -r requirements.txt
python main.py
```

That is the whole procedure. `main.py` does everything itself:

1. Finds an `ollama` binary, or installs one under `~/.local/ollama` — **no sudo
   required** (the official Linux tarball is unpacked into your home directory
   rather than `/usr/local`).
2. Launches **one Ollama server per GPU**, each pinned with `CUDA_VISIBLE_DEVICES`
   and bound to its own port (11434, 11435, …). Models live in `~/.ollama/models`.
3. Pulls `llama3.2:3b` and warms up every server.
4. Solves 800 GSM8K problems concurrently across the GPUs.
5. Writes all reports into `results/` and shuts the servers down.

### Why one server per GPU

`llama3.2:3b` is ~2 GB, so it fits many times over on a 40 GB card. Sharding one
copy across four GPUs would only add communication overhead. Replicating it —
one server per GPU, each serving `OLLAMA_NUM_PARALLEL` concurrent requests —
gives near-linear throughput instead. On a 4×A100 node that is 16 problems in
flight at once.

### Options

```bash
python main.py --num_examples 800     # how many GSM8K problems (default 800)
python main.py --workers 32           # concurrent problems (default: servers × 4)
python main.py --output_dir results   # where reports go
python main.py --split test           # GSM8K split
python main.py --no_resume            # ignore the checkpoint, start over
python main.py --skip_setup           # use Ollama servers that are already running
```

Environment overrides: `PAMI_NUM_SERVERS`, `PAMI_PARALLEL_PER_SERVER`, `PAMI_BASE_PORT`.

### Resuming

Every solved problem is appended to `results/all_results.jsonl` and fsynced
immediately. If the run is interrupted (Ctrl-C, SSH drop, node eviction), just
run `python main.py` again — it skips what is already done and picks up the
rest. Ctrl-C also scores whatever finished so far rather than discarding it.

Problems whose LLM calls fail permanently are reported but *not* checkpointed,
so a re-run retries exactly those.

## Output (`results/`)

| File | Contents |
|---|---|
| `all_results.jsonl` | Full chain-of-thought traces — every step, every uncertainty sample. One JSON object per problem. Also the resume checkpoint. |
| `solutions.jsonl` | One compact scored record per solved problem |
| `per_example.csv` | Flat table for plotting / spreadsheets |
| `summary.json` | All aggregate metrics |
| `summary.md` | Human-readable report |
| `baseline_vs_audit_comparison.md` | Per-example proposer vs. audit-loop comparison |
| `logs/` | stdout/stderr of each Ollama server |

## Roadmap Coverage

1. ✅ Model A proposes → Model B audits → Model A re-audits. Stop after 4 answers or high agreement.
2. ✅ At each step, sample the auditor 5 times. Compute majority-vote agreement as uncertainty proxy.
3. ✅ Report accuracy per step, average number of calls, false-certainty cases, and uncertainty-drop-but-wrong examples.

## Requirements

- Python 3.10+
- Linux x86-64 with NVIDIA GPUs (falls back to CPU, slowly)
- ~10 GB free disk in `$HOME` for the Ollama tarball + model
- No root access needed

## References

- Farquhar et al., *Detecting Hallucinations in LLMs Using Semantic Entropy* (Nature, 2024)
- Manakul et al., *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection* (EMNLP, 2023)
- Cohen et al., *LM vs LM: Detecting Factual Errors via Cross Examination* (2023)
- Kamoi et al., *When Can LLMs Actually Correct Their Own Mistakes?* (2024)
