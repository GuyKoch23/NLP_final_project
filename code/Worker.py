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

        for iter, review in enumerate(self.dataset.reviews):
            print(f"Iteration number {iter}")
            true_score = review.true_score

            predicted_score, predicted_score_notes = self.scorer.predict_score(
                review.text
            )
            direct_distance = ScoreService.calculate_distance(
                predicted_score, true_score
            )
            direct_square_distance = ScoreService.calculate_square_distance(
                predicted_score, true_score
            )

            aspect_scores, aspect_scores_notes = self.scorer.predict_aspect_scores(
                review.text
            )
            # Filter out zero values before calculating the mean
            non_zero_scores = [score for score in aspect_scores.values() if score != 0]
            avg_aspect_score = (
                np.mean(non_zero_scores) if non_zero_scores else 3.0
            )  # Default to 3.0 if all scores are zero
            aspect_distance = ScoreService.calculate_distance(
                avg_aspect_score, true_score
            )
            aspect_square_distance = ScoreService.calculate_square_distance(
                avg_aspect_score, true_score
            )

            results.append(
                {
                    "true_score": true_score,
                    "direct_prediction": predicted_score,
                    "direct_distance": direct_distance,
                    "direct_square_distance": direct_square_distance,
                    "aspect_prediction": avg_aspect_score,
                    "aspect_distance": aspect_distance,
                    "aspect_square_distance": aspect_square_distance,
                    "aspect_scores": aspect_scores,
                    "predicted_score_notes": predicted_score_notes,
                    "aspect_scores_notes": aspect_scores_notes,
                }
            )

        # Convert results to DataFrame
        df_results = pd.DataFrame(results)

        # Compute average errors
        direct_avg_error = df_results["direct_distance"].mean()
        direct_square_avg_error = df_results["direct_square_distance"].mean()
        aspect_avg_error = df_results["aspect_distance"].mean()
        aspect_square_avg_error = df_results["aspect_square_distance"].mean()

        print(f"Average ABS error for direct prediction: {direct_avg_error:.2f}")
        print(f"Average ABS error for aspect-based prediction: {aspect_avg_error:.2f}")

        print(
            f"Average square error for direct prediction: {direct_square_avg_error:.2f}"
        )
        print(
            f"Average square error for aspect-based prediction: {aspect_square_avg_error:.2f}"
        )

        if aspect_avg_error < direct_avg_error:
            print("Aspect-based ABS averaging improves predictions!")
        else:
            print("Aspect-based ABS averaging does not improve predictions.")

        if aspect_square_avg_error < direct_square_avg_error:
            print("Aspect-based square averaging improves predictions!")
        else:
            print("Aspect-based square averaging does not improve predictions.")

        return df_results
