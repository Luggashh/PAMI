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

## Requirements

- Python 3.10+
- Ollama installed and running with `llama3.2:3b` pulled
- ~4 GB RAM for the model

## Setup

```bash
# 1. Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull the model
ollama pull llama3.2:3b

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Run the pipeline
python main.py --num_examples 50 --output_dir results/
```

## Roadmap Coverage

1. ✅ Model A proposes → Model B audits → Model A re-audits. Stop after 4 answers or high agreement.
2. ✅ At each step, sample the auditor 5 times. Compute majority-vote agreement as uncertainty proxy.
3. ✅ Report accuracy per step, average number of calls, false-certainty cases, and uncertainty-drop-but-wrong examples.

## References

- Farquhar et al., *Detecting Hallucinations in LLMs Using Semantic Entropy* (Nature, 2024)
- Manakul et al., *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection* (EMNLP, 2023)
- Cohen et al., *LM vs LM: Detecting Factual Errors via Cross Examination* (2023)
- Kamoi et al., *When Can LLMs Actually Correct Their Own Mistakes?* (2024)
```