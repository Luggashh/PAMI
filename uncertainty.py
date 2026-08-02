"""
Uncertainty estimation via majority-vote agreement.

Inspired by:
- SelfCheckGPT (Manakul et al., 2023): using self-consistency as a proxy
  for hallucination / uncertainty.
- Semantic Entropy (Farquhar et al., 2024): clustering semantically
  equivalent answers to measure uncertainty.

Here we implement a simpler but effective proxy: sample the current auditor
K times, extract numerical answers, and compute majority-vote agreement.
"""

from collections import Counter
from ollama_client import generate
from utils import extract_numerical_answer
from config import UNCERTAINTY_SAMPLES, TEMPERATURE

def compute_uncertainty(
    prompt: str,
    system_prompt: str,
    k: int = UNCERTAINTY_SAMPLES,
) -> dict:
    """
    Sample the model k times and compute majority-vote agreement.

    Args:
        prompt: The prompt to send to the model.
        system_prompt: System prompt for the model role.
        k: Number of samples.

    Returns:
        Dict with:
            - samples: list of raw responses
            - answers: list of extracted numerical answers
            - majority_answer: the most common answer
            - agreement: fraction of samples agreeing with majority
            - agreement_count: raw count of majority votes
    """
    samples = []
    answers = []

    for i in range(k):
        response = generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=TEMPERATURE,
            seed=None,  # Stochastic sampling
        )
        samples.append(response)
        ans = extract_numerical_answer(response)
        answers.append(ans)

    # Filter out None answers for voting
    valid_answers = [a for a in answers if a is not None]

    if not valid_answers:
        return {
            "samples": samples,
            "answers": answers,
            "majority_answer": None,
            "agreement": 0.0,
            "agreement_count": 0,
        }

    counter = Counter(valid_answers)
    majority_answer, majority_count = counter.most_common(1)[0]

    return {
        "samples": samples,
        "answers": answers,
        "majority_answer": majority_answer,
        "agreement": majority_count / k,
        "agreement_count": majority_count,
    }