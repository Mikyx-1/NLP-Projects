"""
Author: Le Hoang Viet
Creation date: 08 March 2025

ROUGE Score Calculation

This implementation calculates the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) score,
which evaluates the quality of machine-generated text compared to one or more reference texts.

Algorithm overview:
1. N-gram Extraction: Extracts contiguous sequences of words of length n from candidate and reference sentences.
2. Precision, Recall, and F1 Calculation: Measures the overlap between candidate and reference n-grams.
    - Precision: Proportion of candidate n-grams that appear in the reference(s).
    - Recall: Proportion of reference n-grams that appear in the candidate.
    - F1 Score: Harmonic mean of precision and recall.
3. Longest Common Subsequence (LCS): Measures the longest sequence of words common between the candidate and reference,
    without requiring consecutive matching.
4. Score Aggregation: For each metric, calculates low, mid (average), and high values across multiple references.

Input:
- candidate: The machine-generated sentence.
- references: A list of human-written reference sentences.

Output:
A dictionary with:
- "ROUGE-1": Unigram-based precision, recall, and F1 score.
- "ROUGE-2": Bigram-based precision, recall, and F1 score.
- "ROUGE-L": LCS-based precision, recall, and F1 score.
"""
class RougeScoreCalculator:
    @staticmethod
    def extract_n_grams(sentence: str, n: int) -> list[str]:
        """
        Extract n-grams from a sentence
        """
        words = sentence.split()
        return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]

    @staticmethod
    def calculate_precision_recall(candidate_grams: list[str], ref_grams: list[str]) -> tuple[float, float, float]:
        if not candidate_grams or not ref_grams:
            return 0.0, 0.0, 0.0

        match_count = sum(1 for gram in candidate_grams if gram in ref_grams)
        precision = match_count / len(candidate_grams)
        recall = match_count / len(ref_grams)
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) != 0 else 0.0
        return precision, recall, f1_score

    @staticmethod
    def lcs(s1: list[str], s2: list[str]) -> int:
        m, n = len(s1), len(s2)
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if s2[i - 1] == s1[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[n][m]

    @staticmethod
    def aggregate_scores(scores: list[float]) -> dict[str, float]:
        return {
            "low": min(scores),
            "mid": sum(scores) / len(scores),
            "high": max(scores)
        } if scores else {"low": 0.0, "mid": 0.0, "high": 0.0}

    @staticmethod
    def compute(candidate: str, references: list[str]) -> dict[str, dict[str, dict[str, float]]]:
        result = {f"ROUGE-{n}": {"precision": None, "recall": None, "f1_score": None} for n in (1, 2)}
        result["ROUGE-L"] = {"precision": None, "recall": None, "f1_score": None}

        for n in (1, 2):
            candidate_grams = RougeScoreCalculator.extract_n_grams(candidate, n)
            precisions, recalls, f1_scores = [], [], []

            for ref in references:
                ref_grams = RougeScoreCalculator.extract_n_grams(ref, n)
                prec, rec, f1 = RougeScoreCalculator.calculate_precision_recall(candidate_grams, ref_grams)
                precisions.append(prec)
                recalls.append(rec)
                f1_scores.append(f1)

            result[f"ROUGE-{n}"]["precision"] = RougeScoreCalculator.aggregate_scores(precisions)
            result[f"ROUGE-{n}"]["recall"] = RougeScoreCalculator.aggregate_scores(recalls)
            result[f"ROUGE-{n}"]["f1_score"] = RougeScoreCalculator.aggregate_scores(f1_scores)

        candidate_words = candidate.split()
        precisions, recalls, f1_scores = [], [], []

        for ref in references:
            ref_words = ref.split()
            lcs_length = RougeScoreCalculator.lcs(candidate_words, ref_words)
            precision = lcs_length / len(candidate_words)
            recall = lcs_length / len(ref_words)
            f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) != 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)

        result["ROUGE-L"]["precision"] = RougeScoreCalculator.aggregate_scores(precisions)
        result["ROUGE-L"]["recall"] = RougeScoreCalculator.aggregate_scores(recalls)
        result["ROUGE-L"]["f1_score"] = RougeScoreCalculator.aggregate_scores(f1_scores)

        return result

if __name__ == "__main__":
    candidate = "I really loved reading the Hunger Games"
    references = ["I loved reading the Hunger Games"]

    result = RougeScoreCalculator.compute(candidate, references)
    print(f"Result: {result}")
