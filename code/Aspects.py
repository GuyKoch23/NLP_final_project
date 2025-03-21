class Aspects:
    def __init__(self, aspects = 
                 [
                    "Room Quality",
                    "Service Quality",
                    "Location", 
                    "Value for Money",
                    "Facilities and Amenities",
                    "Food and Dining",
                    "Cleanliness",
                    "Comfort",
                    "Wi-Fi and Technology",
                    "Atmosphere"
                 ]
                ):
        self.aspects = aspects

    def getAspects(self):
        return self.aspects