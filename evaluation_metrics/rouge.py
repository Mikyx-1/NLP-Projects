

class RougeScoreCalculator:
    @staticmethod
    def extract_n_grams(sentence: str, n: int) -> list:
        """
        Extract n-grams from a sentence
        """
        words = sentence.split()
        return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
    

    @staticmethod
    def calculate_precision_recall(candidate_grams: list[str], ref_grams: list[str]):

        candidate_length = len(candidate_grams)
        ref_length = len(ref_grams)

        if candidate_length == 0 or ref_length == 0:
            return 0, 0

        match_count = 0
        for candidate_gram in candidate_grams:
            if candidate_gram in ref_grams:
                match_count += 1
        
        precision = match_count / candidate_length
        recall = match_count / ref_length
        f1_score = 2*(precision*recall)/(precision+recall)
        return precision, recall, f1_score

    @staticmethod
    def get_lcs(sentence: str):
        pass

    @staticmethod
    def compute(candidate: str, references: list[str]):
        
        result = {"ROUGE-1": {"precision": None, 
                              "recall": None, 
                              "f1_score": None}, 
                  "ROUGE-2": {"precision": None, 
                              "recall": None, 
                              "f1_score": None}}

        for ith_gram in range(1, 3):
            candidate_grams = RougeScoreCalculator.extract_n_grams(candidate, ith_gram)
            ref_grams = [RougeScoreCalculator.extract_n_grams(ref, ith_gram) for ref in references]
            precisions, recalls, f1_scores = [], [], []
            for ref_gram in ref_grams:
                prec, rec, f1_score = RougeScoreCalculator.calculate_precision_recall(candidate_grams, ref_gram)
                precisions.append(prec)
                recalls.append(rec)
                f1_scores.append(f1_score)

            
            result[f"ROUGE-{ith_gram}"]["precision"] = {"low": min(precisions), "mid": sum(precisions)/len(precisions), "high": max(precisions)}
            result[f"ROUGE-{ith_gram}"]["recall"] = {"low": min(recalls), "mid": sum(recalls)/len(recalls), "high": max(recalls)}
            result[f"ROUGE-{ith_gram}"]["f1_score"] = {"low": min(f1_scores), "mid": sum(f1_scores)/len(f1_scores), "high": max(f1_scores)}


        return result



if __name__ == "__main__":
    # Usage
    candidate = "I really loved reading the Hunger Games"
    references = ["I loved reading the Hunger Games"]

    result = RougeScoreCalculator.compute(candidate, references)

    print(f"Result: {result}")