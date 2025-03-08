"""
Author: Le Hoang Viet
Creation date: 08 March 2025

Word Error Rate (WER) Calculation

This implementation calculates the Word Error Rate (WER), which measures the difference between a hypothesis (machine-generated text) and a reference (ground-truth text). It’s commonly used in speech recognition and OCR systems.

Algorithm overview:
1. Edit Distance Calculation: Computes the minimum number of operations (insertions, deletions, substitutions) required to transform one string into another.
2. Character-Level and Word-Level Comparison: Calculates the edit distance both on the character and word levels.
3. Word Error Rate: Normalizes the word-level edit distance by the number of words in the reference text.

Input:
- text1: The reference (ground-truth) sentence.
- text2: The hypothesis (machine-generated) sentence.

Output:
A dictionary with:
- "char_level_edit_dist": Edit distance calculated at the character level.
- "word_level_edit_dist": Edit distance calculated at the word level.
- "wer": Final Word Error Rate score.
"""

from typing import Union
import re

class WER:
    """
    A class to calculate Word Error Rate (WER) and edit distances.
    """
    
    @staticmethod
    def edit_distance(text1: Union[str, list[str]], text2: Union[str, list[str]]) -> int:
        """
        Calculates the edit distance between two strings or lists of strings.

        Args:
            text1 (Union[str, list[str]]): The first text or list of words.
            text2 (Union[str, list[str]]): The second text or list of words.

        Returns:
            int: The minimum number of operations required to transform text1 into text2.
        """
        m, n = len(text1), len(text2)
        # Create a DP table with (m+1) rows and (n+1) columns
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Base cases: transforming an empty string
        for i in range(m + 1):
            dp[i][0] = i  # Deleting all characters from word1
        for j in range(n + 1):
            dp[0][j] = j  # Inserting all characters into word1

        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    # Characters match: no operation needed
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # Choose the minimum of insert, delete, or substitute
                    dp[i][j] = min(dp[i - 1][j] + 1,     # Deletion
                                   dp[i][j - 1] + 1,     # Insertion
                                   dp[i - 1][j - 1] + 1) # Substitution

        # The bottom-right cell contains the edit distance
        return dp[m][n]

    @staticmethod
    def compute(text1: str, text2: str) -> dict:
        """
        Computes the character-level and word-level edit distances and the Word Error Rate.

        Args:
            text1 (str): The reference (ground-truth) sentence.
            text2 (str): The hypothesis (machine-generated) sentence.

        Returns:
            dict: A dictionary containing character-level edit distance, word-level edit distance, and WER.
        """
        # Normalize spaces and split words
        words1 = re.split(r'\s+', text1.strip())
        words2 = re.split(r'\s+', text2.strip())

        char_level_edit_dist = WER.edit_distance(text1, text2)
        word_level_edit_dist = WER.edit_distance(words1, words2)

        # Calculate WER as word-level edit distance normalized by number of words in reference text
        wer = word_level_edit_dist / len(words1) if words1 else float('inf')

        result = {"char_level_edit_dist": char_level_edit_dist,
                  "word_level_edit_dist": word_level_edit_dist,
                  "wer": wer}
        return result

if __name__ == "__main__":
    s1 = "Hello My name is Viet"
    s2 = "My name Viet"

    result: dict = WER.compute(s1, s2)
    print(f"Char_level_edit_dist: {result['char_level_edit_dist']}, Word_level_edit_dist: {result['word_level_edit_dist']}, WER: {result['wer']:.2f}")
