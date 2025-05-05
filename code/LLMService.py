from typing import Dict, Tuple
from Aspects import Aspects
import ollama
import re


class LLMService:
    def __init__(self, model_name: str = "llama3.2:1b"):
        """Initialize the LLM service with Ollama.

        Args:
            model_name: The name of the Ollama model to use (default: "llama3.2:1b")
        """
        self.model_name = model_name
        self.Aspects = Aspects()

        # Define the common detailed scoring criteria to use in both prompts
        self.SCORING_CRITERIA = """
        ## Detailed Scoring Criteria:
        - 1: Very negative - Extreme dissatisfaction, severe problems mentioned, angry tone, warnings to avoid
        - 2: Somewhat negative - Clear dissatisfaction, multiple issues mentioned, disappointment expressed
        - 3: Neutral - Mixed feedback with both positives and negatives, or generally unemotional description
        - 4: Somewhat positive - Clear satisfaction, minor issues might be mentioned but overall positive experience
        - 5: Extremely positive - Complete enthusiasm, exceptional experience described, strong recommendation
        """

        # Define scoring prompt template with the common criteria
        self.SCORE_PROMPT_TEMPLATE = f"""
        You are an AI trained to analyze reviews and assign scores on a scale from 1 to 5.
        
        {self.SCORING_CRITERIA}
        
        ## Review:
        "{{review}}"
        
        ## Task:
        Based on the detailed criteria, analyze the review and provide a numerical score from the scale above.
        You MUST provide a score between 1 and 5. Just return the number without any explanation.
        """

        # Define aspect scoring prompt template with the same criteria
        self.ASPECT_PROMPT_TEMPLATE = f"""
        You are an AI trained to analyze reviews and assign scores based on specific aspects using a scale from 1 to 5.
        
        ## Aspect: {{aspect}}
        
        {self.SCORING_CRITERIA}
        
        ## Review:
        "{{review}}"
        
        ## Task:
        Focusing ONLY on the {{aspect}} aspect, analyze the review and provide a numerical score from the scale above.
        You MUST provide a score between 1 and 5. Just return the number without any explanation.
        """

    def predict_score(self, review_text: str) -> Tuple[float, bool]:
        """Predicts a score for the given review using Ollama LLaMA model."""
        prompt = self.SCORE_PROMPT_TEMPLATE.format(review=review_text)
        err = False
        response = ollama.chat(
            model=self.model_name, messages=[{"role": "user", "content": prompt}]
        )

        # Extract numerical score from response
        content = response["message"]["content"].strip()
        # Try to find a number in the response (for scores 1-5 with optional decimal)
        match = re.search(r"([1-5](\.\d+)?)", content)
        if match:
            score = float(match.group(1))
        else:
            # Fallback if no valid score is found
            print(f"No valid score was found in response: '{content}'")
            err = True
            score = 3

        return max(1.0, min(5.0, score)), err  # Clamp scores between 1 and 5

    def predict_aspect_scores(self, review_text: str) -> Tuple[dict, dict]:
        """Predicts aspect-based scores and returns a dictionary of results."""
        aspect_scores = {}
        errs = {key: False for key in self.Aspects.getAspects()}
        for aspect in self.Aspects.getAspects():
            prompt = self.ASPECT_PROMPT_TEMPLATE.format(
                review=review_text, aspect=aspect
            )

            response = ollama.chat(
                model=self.model_name, messages=[{"role": "user", "content": prompt}]
            )

            # Extract numerical score from response
            content = response["message"]["content"].strip()
            match = re.search(r"([1-5](\.\d+)?)", content)
            if match:
                aspect_scores[aspect] = float(match.group(1))
            else:
                # Fallback if no valid score is found
                print(
                    f"No valid score found for aspect '{aspect}' in response: '{content}'"
                )
                errs[aspect] = True
                aspect_scores[aspect] = 0

            # Ensure scores are within range
            if aspect_scores[aspect] != 0:
                aspect_scores[aspect] = max(1.0, min(5.0, aspect_scores[aspect]))

        return aspect_scores, errs
