import math
from collections import Counter


class BLEUScoreCalculator:
    def __init__(self, max_n_grams: int = 4):
        self.max_n_grams = max_n_grams

    def _extract_n_grams(self, sentence: str, n: int) -> list:
        """
        Extract n-grams from a sentence
        """
        words = sentence.split()
        return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]

    def _calculate_brevity_penalty(self, candidate_length: int, reference_length: int) -> float:
        """
        Calculate Brevity Penalty
        """
        if candidate_length >= reference_length:
            return 1
        else:
            return math.exp(1 - reference_length / candidate_length)

    def _calculate_n_gram_precision(self, candidate: str, reference: str, n: int) -> float:
        """
        Calculate n-gram precision
        """
        candidate_n_grams = self._extract_n_grams(candidate, n)
        reference_n_grams = self._extract_n_grams(reference, n)

        candidate_counts = Counter(candidate_n_grams)
        reference_counts = Counter(reference_n_grams)

        match_count = sum(min(candidate_counts[ng], reference_counts.get(ng, 0)) for ng in candidate_counts)
        total_count = len(candidate_n_grams)

        if total_count == 0:
            return 0

        return match_count / total_count

    def compute(self, candidate: str, reference: str) -> dict:
        """
        Calculate BLEU score and return a detailed result
        """
        precisions = []
        for n in range(1, self.max_n_grams + 1):
            precision = self._calculate_n_gram_precision(candidate, reference, n)
            precisions.append(precision)

        # Calculate BLEU score even if some precisions are 0
        log_precisions = [math.log(p) if p > 0 else float('-inf') for p in precisions]
        avg_log_precision = sum(log_precisions) / self.max_n_grams

        # BLEU score calculation
        bleu_score = math.exp(avg_log_precision) if all(p != float('-inf') for p in log_precisions) else 0

        candidate_length = len(candidate.split())
        reference_length = len(reference.split())
        brevity_penalty = self._calculate_brevity_penalty(candidate_length, reference_length)

        return {
            "bleu": brevity_penalty * bleu_score,
            "precisions": precisions,
            "brevity_penalty": brevity_penalty,
            "candidate_length": candidate_length,
            "reference_length": reference_length
        }


if __name__ == "__main__":
    candidate = "I have thirty six years"
    reference = "I am thirty six"

    bleu_calculator = BLEUScoreCalculator(4)
    result = bleu_calculator.compute(candidate, reference)

    print(result)
