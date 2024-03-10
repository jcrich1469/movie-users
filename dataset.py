import torch
from torch.utils.data import Dataset

class ColFDataset(Dataset):
    
    def __init__(self,data,encoder):

        user_ids = [did[2] for did in data]
        movie_ids = [did[1] for did in data]
        #may differ depending..... for 
        # ratings = [int(did[4].split('/')[0]) for did in data]
        ratings = [did[4] for did in data]


        self.dataset = [[torch.tensor(encoder.encode(user_ids[i],encoder.user_to_idx),dtype=torch.int64),torch.tensor(encoder.encode(movie_ids[i],encoder.movie_to_idx),dtype=torch.int64),torch.tensor(ratings[i],dtype=torch.float)] for i in range(len(data))]
    
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        user_id, movie_id, rating = self.dataset[idx]
        return user_id, movie_id, rating


# compound data
class ConFDataset(Dataset):
    
    def __init__(self, data, encoder):
        self.user_data = [torch.tensor(encoder.encode(did[0], 'users'), dtype=torch.int64) for did in data]
        # Assuming did[1] contains the genre information as a comma-separated string
        #encoder.one_hot_encode(genre_data, 'genres')
        self.genre_data = [encoder.one_hot_encode(did[1],'genres') for did in data]  # No need for torch.tensor if encoder.one_hot_encode returns a tensor
        self.country_data = [encoder.one_hot_encode(did[2], 'countries') for did in data]
        self.ratings_data = [torch.tensor(did[3], dtype=torch.float) for did in data]
        self.length = len(data)
    
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        movie_features = torch.cat((self.genre_data[idx], self.country_data[idx]), dim=0)

        return self.user_data[idx],movie_features,self.ratings_data[idx]




class Encoder:
    def __init__(self, **kwargs):
        self.vocab_to_idx = {}
        self.idx_to_vocab = {}

        for category_name, items in kwargs.items():
            # Flatten the items if they are lists (for genres, for example)
            # Otherwise, use the items as they are (for user names and other categories)
            flattened_items = []
            for item in items:
                if isinstance(item, list):
                    flattened_items.extend(item)
                else:
                    flattened_items.append(item)
            self.vocab_to_idx[category_name] = self.build_vocab(flattened_items)
            self.idx_to_vocab[category_name] = {v: k for k, v in self.vocab_to_idx[category_name].items()}
    
    
    def build_vocab(self, items):
        vocab = {}
        for item in items:
            if item not in vocab:
                vocab[item] = len(vocab)
        return vocab

    def flatten_lists(self, items):
        # Split items by comma and return unique items
        split_items = set()
        for item in items:
            for subitem in item.split(','):
                split_items.add(subitem.strip())
        return list(split_items)

    def encode(self, item, category):
        # Handle both single items and lists for encoding
        if isinstance(item, list):
            # Return a list of indices for list items
            return [self.vocab_to_idx[category].get(i, None) for i in item]
        else:
            # Return a single index for a single item
            return self.vocab_to_idx[category].get(item, None)

    def one_hot_encode(self, items, category):
        # Ensure 'items' is a list even if a single string is passed
        items = [items] if isinstance(items, str) else items
        
        vocab_size = len(self.vocab_to_idx[category])
        one_hot_vector = torch.zeros(vocab_size,dtype=torch.float)
        for item in items:
            idx = self.encode(item, category)
            if idx is not None:
                one_hot_vector[idx] = 1
        return one_hot_vector

    def one_hot_decode(self, one_hot_vector, category):
        items = []
        for idx, value in enumerate(one_hot_vector):
            if value == 1:
                decoded_item = self.decode(idx, category)
                items.append(decoded_item)
        return items

    def decode(self, idx, category):
        return self.idx_to_vocab[category].get(idx, None)




#split according to tokens.......????
def split_data(data, split = 0.8):
    
    
    split_idx = int(split * len(data))  # 80% for training, 20% for testing
    
    # Split the data, train and test
    return data[:split_idx], data[split_idx:]

