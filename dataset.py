import torch
from torch.utils.data import Dataset

class CFDataset(Dataset):
    
    def __init__(self, data,encoder):

        user_ids = [did[2] for did in data]
        movie_ids = [did[1] for did in data]
        ratings = [int(did[4].split('/')[0]) for did in data]


        self.dataset = [[torch.tensor(encoder.encode(user_ids[i],encoder.user_to_idx),dtype=torch.int64),torch.tensor(encoder.encode(movie_ids[i],encoder.movie_to_idx),dtype=torch.int64),torch.tensor(ratings[i],dtype=torch.float)] for i in range(len(data))]
    
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        user_id, movie_id, rating = self.dataset[idx]
        return user_id, movie_id, rating

    
class Encoder:
    def __init__(self, user_ids, movie_ids):
        self.user_to_idx = self.build_vocab(user_ids, {})
        self.movie_to_idx = self.build_vocab(movie_ids, {})

        self.idx_to_user = {v:k for k,v in self.user_to_idx.items()}
        self.idx_to_movie ={v:k for k,v in self.movie_to_idx.items()}

    def build_vocab(self, items, dict):
        for item_id in items:
            if item_id not in dict.keys():
                dict[item_id] = len(dict)
        return dict

    def encode(self, item_id,vocab):
        return vocab.get(item_id)

    def decode(self, idx, vocab):
        return vocab.get(idx)


#split according to tokens.......????
def split_data(data, split = 0.8):
    
    
    split_idx = int(split * len(data))  # 80% for training, 20% for testing
    
    # Split the data, train and test
    return data[:split_idx], data[split_idx:]

# # Example usage:
# # Assuming you have user-item interaction data in tensors (user_ids, movie_ids, ratings)
# user_ids = [1, 2, 1, 3, ...]  # Example user IDs
# movie_ids = [101, 102, 105, 103, ...]  # Example movie IDs
# ratings = [4.0, 3.5, 5.0, 2.0, ...]  # Example ratings

# # Create an instance of the dataset
# dataset = CollaborativeFilteringDataset(user_ids, movie_ids, ratings)

# # Access individual data samples
# for i in range(len(dataset)):
#     sample = dataset[i]
#     print(f"User ID: {sample['user_id']}, Movie ID: {sample['movie_id']}, Rating: {sample['rating']}")
