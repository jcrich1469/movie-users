import torch
import torch.nn as nn
import torch.optim as optim

class CollaborativeFilter(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim):
        super(CollaborativeFilter, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, 1)
    
    def forward(self, user_ids, movie_ids):
        user_embeds = self.user_embedding(user_ids)
        movie_embeds = self.movie_embedding(movie_ids)
        concat_embeds = torch.cat([user_embeds, movie_embeds], dim=1)
        output = self.fc(concat_embeds)
        return output.squeeze()

#original dorupout = 0.5 = 50%
class MLPCollaborativeFilter(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim, hidden_dims=[64, 32],dropout_rate=0.5):
        super(MLPCollaborativeFilter, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # Define a list to hold all the layers
        layers = []
        input_dim = embedding_dim * 2  # because we concatenate user and movie embeddings
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # Batch normalization layer
            # Apply dropout after each linear layer (except for the last linear layer)
            layers.append(nn.Dropout(p=dropout_rate))
            layers.append(nn.ReLU())  # Adding non-linearity
            input_dim = hidden_dim  # Next layer's input is current layer's output
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        # Combine all layers into a Sequential model
        self.layers = nn.Sequential(*layers)
    
    def forward(self, user_ids, movie_ids):
        user_embeds = self.user_embedding(user_ids)
        movie_embeds = self.movie_embedding(movie_ids)
        concat_embeds = torch.cat([user_embeds, movie_embeds], dim=1)
        output = self.layers(concat_embeds)
        return output.squeeze()



#Only genres for now. deadline, 12th March
#JC WEBTOKENS - rust backend, js

class ContentFilter(nn.Module):
    def __init__(self, num_users, num_features, embedding_dim):
        super(ContentFilter, self).__init__()
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


class MLPContentFilter(nn.Module):
    def __init__(self, num_users, num_features, embedding_dim, hidden_dims=[100, 50], dropout_rate=0.5):
        super(MLPContentFilter, self).__init__()
        # User embeddings based on their preference profiles
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        # Initial linear layer to transform item features into the same embedding space as users
        self.initial_feature_transform = nn.Linear(num_features, embedding_dim)
        self.initial_bn = nn.BatchNorm1d(embedding_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Define hidden layers for the MLP with accompanying BatchNorm layers
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        layer_sizes = [embedding_dim] + hidden_dims
        for i in range(len(layer_sizes) - 1):
            self.hidden_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.bn_layers.append(nn.BatchNorm1d(layer_sizes[i+1]))

        # Final layer to predict the rating
        self.fc_final = nn.Linear(hidden_dims[-1] * 2, 1)

    def forward(self, user_ids, movie_features):
        # Embed the users
        user_embeds = self.user_embedding(user_ids)
        
        # Transform movie features through the initial layer with ReLU and BatchNorm
        movie_embeds = F.relu(self.initial_bn(self.initial_feature_transform(movie_features)))

        # Pass movie features through hidden layers
        for layer, bn_layer in zip(self.hidden_layers, self.bn_layers):
            movie_embeds = layer(movie_embeds)
            movie_embeds = bn_layer(movie_embeds)
            movie_embeds = F.relu(movie_embeds)
            movie_embeds = self.dropout(movie_embeds)

        # Concatenate user and movie embeddings
        concat_embeds = torch.cat([user_embeds, movie_embeds], dim=1)

        # Predict the rating
        output = self.fc_final(concat_embeds)
        return output.squeeze()