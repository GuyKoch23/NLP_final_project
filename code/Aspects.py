class Aspects:
    def __init__(
        self,
        aspects=[
            "Room Quality",
            "Service Quality",
            "Food and Dining",
            "Facilities and Amenities",
            "Cleanliness",
        ],
    ):
        self.aspects = aspects

    def getAspects(self):
        return self.aspects
