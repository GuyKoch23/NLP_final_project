import pandas as pd
from typing import List
from Review import Review

class ReviewDataset:
    def __init__(self, file_path: str):
        self.reviews = self.load_reviews(file_path)

    def load_reviews(self, file_path: str) -> List[Review]:
        df = pd.read_csv(file_path)
        return [Review(row['Review'], row['Rating']) for _, row in df.iterrows()]