"""
Utility functions for answer extraction and comparison.
"""

import re

def extract_numerical_answer(text: str) -> str | None:
    """
    Extract the final numerical answer from model output.

    Handles formats like:
        - "The answer is 42"
        - "#### 42"
        - "**42**"
        - Just a number at the end
        - Negative numbers and decimals
    """
    if not text:
        return None

    # Try "#### <number>" format first (GSM8K style)
    match = re.search(r"####\s*([-\d,\.]+)", text)
    if match:
        return _clean_number(match.group(1))

    # Try "the answer is <number>" pattern
    match = re.search(
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*([-\d,\.]+)",
        text,
        re.IGNORECASE,
    )
    if match:
        return _clean_number(match.group(1))

    # Try "= <number>" at end of text
    match = re.search(r"=\s*([-\d,\.]+)\s*$", text)
    if match:
        return _clean_number(match.group(1))

    # Try bold number pattern **<number>**
    match = re.search(r"\*\*([-\d,\.]+)\*\*", text)
    if match:
        return _clean_number(match.group(1))

    # Last resort: find the last number in the text
    numbers = re.findall(r"[-]?\d[\d,]*\.?\d*", text)
    if numbers:
        return _clean_number(numbers[-1])

    return None

def _clean_number(s: str) -> str:
    """Remove commas and trailing dots from number strings."""
    s = s.replace(",", "").strip(".")
    # Normalize: remove trailing zeros after decimal
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return s

def answers_match(predicted: str | None, gold: str) -> bool:
    """Check if predicted answer matches the gold answer numerically."""
    if predicted is None:
        return False
    try:
        return abs(float(predicted) - float(gold)) < 1e-5
    except (ValueError, TypeError):
        return predicted.strip() == gold.strip()