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
            direct_cubic_distance = ScoreService.calculate_cubic_distance(predicted_score, true_score)

            aspect_scores = self.scorer.predict_aspect_scores(review.text)
            avg_aspect_score = np.mean(list(aspect_scores.values()))
            aspect_distance = ScoreService.calculate_distance(avg_aspect_score, true_score)
            aspect_cubic_distance = ScoreService.calculate_cubic_distance(avg_aspect_score, true_score)

            results.append({
                "true_score": true_score,
                "direct_prediction": predicted_score,
                "direct_distance": direct_distance,
                "direct_cubic_distance": direct_cubic_distance,
                "aspect_prediction": avg_aspect_score,
                "aspect_distance": aspect_distance,
                "aspect_cubic_distance": aspect_cubic_distance,
                "aspect_scores": aspect_scores
            })

        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        
        # Compute average errors
        direct_avg_error = df_results["direct_distance"].mean()
        direct_cubic_avg_error = df_results["direct_cubic_distance"].mean()
        aspect_avg_error = df_results["aspect_distance"].mean()
        aspect_cubic_avg_error = df_results["aspect_cubic_distance"].mean()

        print(f"Average ABS error for direct prediction: {direct_avg_error:.2f}")
        print(f"Average ABS error for aspect-based prediction: {aspect_avg_error:.2f}")

        print(f"Average CUBIC error for direct prediction: {direct_cubic_avg_error:.2f}")
        print(f"Average CUBIC error for aspect-based prediction: {aspect_cubic_avg_error:.2f}")

        if aspect_avg_error < direct_avg_error:
            print("Aspect-based ABS averaging improves predictions!")
        else:
            print("Aspect-based ABS averaging does not improve predictions.")

        if aspect_cubic_avg_error < direct_cubic_avg_error:
            print("Aspect-based CUBIC averaging improves predictions!")
        else:
            print("Aspect-based CUBIC averaging does not improve predictions.")

        return df_results