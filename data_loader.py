"""
Load GSM8K dataset from HuggingFace and extract ground-truth answers.
"""

from datasets import load_dataset
import re

def load_gsm8k(split: str = "test", num_examples: int | None = None) -> list[dict]:
    """
    Load GSM8K examples.

    Each example has:
        - question: the math word problem
        - answer: the ground-truth final numerical answer
        - solution: the full chain-of-thought solution

    Returns:
        List of dicts with keys: question, answer, solution
    """
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    examples = []
    for item in dataset:
        question = item["question"]
        full_answer = item["answer"]

        # GSM8K format: "#### <number>" at the end
        numerical_answer = extract_gsm8k_answer(full_answer)

        examples.append({
            "question": question,
            "answer": numerical_answer,
            "solution": full_answer,
        })

        if num_examples and len(examples) >= num_examples:
            break

    return examples

def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the final numerical answer from GSM8K '#### <number>' format."""
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return answer_text.strip()