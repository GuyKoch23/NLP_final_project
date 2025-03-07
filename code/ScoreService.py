class ScoreService:
    def calculate_distance(predicted: float, true: float) -> float:
        return abs(predicted - true)