class User:

    def __init__(self,name):

        self.name = name
        self.user_id = id(self)
        self.reviews = []

    #assuming one review per movie.
    #{movie name:...,movie_id:...,moviereview:{rating etc....}...,} 
    def add_review(self,review):

        self.reviews.append(review)

    def __str__(self):
        return f"{self.name} @ {self.user_id}"