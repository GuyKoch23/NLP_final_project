from typing import Dict
from Aspects import Aspects
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class LLMService:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        self.Aspects = Aspects()

    def predict_score(self, review_text: str) -> float:
        """Predicts a score for the given review using BERT."""
        inputs = self.tokenizer(review_text, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        with torch.no_grad():
            output = self.model(**inputs)
            score = output.logits.squeeze().item()  # Assuming regression output
        
        return max(1.0, min(5.0, score))  # Clamp scores between 1 and 10

    def predict_aspect_scores(self, review_text: str) -> dict:
        """Predicts aspect-based scores and returns a dictionary of results."""
        aspect_scores = {}
        for aspect in self.Aspects.getAspects():
            aspect_prompt = f"Rate this review based on {aspect}: {review_text}"
            aspect_scores[aspect] = self.predict_score(aspect_prompt)
        return aspect_scores