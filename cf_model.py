import torch
import torch.nn as nn
import torch.optim as optim

class CollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim):
        super(CollaborativeFiltering, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, 1)
    
    def forward(self, user_ids, movie_ids):
        user_embeds = self.user_embedding(user_ids)
        movie_embeds = self.movie_embedding(movie_ids)
        concat_embeds = torch.cat([user_embeds, movie_embeds], dim=1)
        output = self.fc(concat_embeds)
        return output.squeeze()



import torch
import torch.nn as nn
import torch.optim as optim

class ContentFiltering(nn.Module):
    def __init__(self, num_users, num_features, embedding_dim):
        super(ContentFiltering, self).__init__()
        # User embeddings based on their preference profiles
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        # A linear layer to transform item features into the same embedding space as users
        self.feature_transform = nn.Linear(num_features, embedding_dim)
        # Final layer to predict the rating
        self.fc = nn.Linear(embedding_dim * 2, 1)
    
    def forward(self, user_ids, movie_features):
        # Embed the users
        user_embeds = self.user_embedding(user_ids)
        # Transform movie features
        movie_embeds = self.feature_transform(movie_features)
        # Concatenate user and movie embeddings
        concat_embeds = torch.cat([user_embeds, movie_embeds], dim=1)
        # Predict the rating
        output = self.fc(concat_embeds)
        return output.squeeze()

# Example usage
num_users = 1000  # Example number of users
num_features = 20  # Example number of movie features (genre, director, etc.)
embedding_dim = 50  # Embedding dimensionality

model = ContentFiltering(num_users, num_features, embedding_dim)