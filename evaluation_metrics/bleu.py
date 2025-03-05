"""
Calcualte BLEU Score 
"""

import math



def extract_n_grams(sentence: str, n: int) -> list:
    """
    Extract n-grams from a sentence
    """
    words = sentence.split()
    n_grams = []
    for i in range(len(words) - n + 1):
        n_grams.append(" ".join(words[i:i+n]))
    return n_grams

def calculate_brevity_penalty(candidate_length: int, reference_length: int) -> float:
    """
    Calculate Brevity Penalty
    """
    if candidate_length > reference_length:
        return 1
    else:
        reference_length += 1e-6
        candidate_length += 1e-6
        return math.exp(1 - reference_length / candidate_length)


if __name__ == "__main__":
    sentence = "the cat is on the mat"
    one_grams = extract_n_grams(sentence, 1)
    two_grams = extract_n_grams(sentence, 2)
    print(f"One-grams: {one_grams}")
    print(f"Two-grams: {two_grams}")