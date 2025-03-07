from ReviewDataset import ReviewDataset
from LLMService import LLMService
from ScoreService import ScoreService
import numpy as np
import pandas as pd

class Worker:    
    def __init__(self, dataset: ReviewDataset, scorer: LLMService):
            self.dataset = dataset
            self.scorer = scorer

    def run(self):
        results = []

        for review in self.dataset.reviews:
            true_score = review.true_score

            predicted_score = self.scorer.predict_score(review.text)
            direct_distance = ScoreService.calculate_distance(predicted_score, true_score)

            aspect_scores = self.scorer.predict_aspect_scores(review.text)
            avg_aspect_score = np.mean(list(aspect_scores.values()))
            aspect_distance = ScoreService.calculate_distance(avg_aspect_score, true_score)

            results.append({
                "true_score": true_score,
                "direct_prediction": predicted_score,
                "direct_distance": direct_distance,
                "aspect_prediction": avg_aspect_score,
                "aspect_distance": aspect_distance,
                "aspect_scores": aspect_scores
            })

        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        
        # Compute average errors
        direct_avg_error = df_results["direct_distance"].mean()
        aspect_avg_error = df_results["aspect_distance"].mean()

        print(f"Average error for direct prediction: {direct_avg_error:.2f}")
        print(f"Average error for aspect-based prediction: {aspect_avg_error:.2f}")

        if aspect_avg_error < direct_avg_error:
            print("Aspect-based averaging improves predictions!")
        else:
            print("Aspect-based averaging does not improve predictions.")

        return df_results