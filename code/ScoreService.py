
class ScoreService:
    def calculate_distance(predicted: float, true: float) -> float:
        return abs(predicted - true)
    
    def calculate_square_distance(predicted: float, true: float) -> float:
        return pow((predicted - true),2)