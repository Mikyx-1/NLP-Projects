"""
Author: Le Hoang Viet
Creation date: 06 March 2025

BLEU Score Calculation

This implementation calculates the BLEU (Bilingual Evaluation Understudy) score, which evaluates the quality of machine-generated text compared to one or more reference texts.

Algorithm overview:
1. N-gram Extraction: Extracts contiguous sequences of words of length n from candidate and reference sentences.
2. Precision Calculation: For each n-gram size, calculates the maximum precision across all references — i.e., the proportion of candidate n-grams that appear in the reference(s).
3. Brevity Penalty: Penalises short candidate sentences by comparing the candidate length to the shortest reference.
4. BLEU Score: Combines the precisions for different n-grams and applies the brevity penalty.

Input:
- candidate: The machine-generated sentence.
- references: A list of human-written reference sentences.

Output:
A dictionary with:
- "bleu": Final BLEU score.
- "precisions": N-gram precisions.
- "brevity_penalty": Brevity penalty value.
- "candidate_length": Length of the candidate sentence.
- "reference_lengths": Lengths of the reference sentences.
"""

import math
from collections import Counter
class BLEUScoreCalculator:

    @staticmethod
    def _extract_n_grams(sentence: str, n: int) -> list:
        """
        Extract n-grams from a sentence
        """
        words = sentence.split()
        return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]

    @staticmethod
    def _calculate_brevity_penalty(candidate_length: int, reference_lengths: list) -> float:
        """
        Calculate Brevity Penalty using the shortest reference length
        """
        shortest_length = min(reference_lengths)
        if candidate_length >= shortest_length:
            return 1
        else:
            return math.exp(1 - shortest_length / candidate_length)

    @staticmethod
    def _calculate_n_gram_precision(candidate: str, references: list, n: int) -> float:
        """
        Calculate n-gram precision by taking the maximum match across all references
        """
        candidate_n_grams = BLEUScoreCalculator._extract_n_grams(candidate, n)
        candidate_counts = Counter(candidate_n_grams)

        total_count = len(candidate_n_grams)
        if total_count == 0:
            return 0

        max_match_count = 0
        for reference in references:
            reference_n_grams = BLEUScoreCalculator._extract_n_grams(reference, n)
            reference_counts = Counter(reference_n_grams)
            match_count = sum(min(candidate_counts[ng], reference_counts.get(ng, 0)) for ng in candidate_counts)
            max_match_count = max(max_match_count, match_count)

        return max_match_count / total_count

    @staticmethod
    def compute(candidate: str, references: list) -> dict:
        """
        Calculate BLEU score and return a detailed result
        """
        precisions = []

        max_n_grams = min(len(ref.split()) for ref in references)
        for n in range(1, max_n_grams + 1):
            precision = BLEUScoreCalculator._calculate_n_gram_precision(candidate, references, n)
            precisions.append(precision)

        # Calculate BLEU score as the product of all precisions
        bleu_score = math.exp(sum(math.log(p) if p > 0 else float('-inf') for p in precisions) / max_n_grams)

        candidate_length = len(candidate.split())
        reference_lengths = [len(ref.split()) for ref in references]
        brevity_penalty = BLEUScoreCalculator._calculate_brevity_penalty(candidate_length, reference_lengths)

        return {
            "bleu": brevity_penalty * bleu_score,
            "precisions": precisions,
            "brevity_penalty": brevity_penalty,
            "candidate_length": candidate_length,
            "reference_lengths": reference_lengths
        }


if __name__ == "__main__":
    # Usage
    candidate = "I have thirty six years"
    references = ["I am thirty six years old", "I've been thirty six for a while", "I am thirty six"]
    result = BLEUScoreCalculator.compute(candidate, references)
    print(result)
