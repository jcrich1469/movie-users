import torch
from torch.utils.data import Dataset

class ColFDataset(Dataset):
    
    def __init__(self,data,encoder):
 
        self.user_ids = [torch.tensor(encoder.encode(did[0],'users'),dtype=torch.int64) for did in data]
        self.movie_ids = [torch.tensor(encoder.encode(did[1],'movies'),dtype=torch.int64) for did in data]
        self.ratings = [torch.tensor(did[-1],dtype=torch.float) for did in data]
        self.length = len(self.ratings)
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]


# compound data
class ConFDataset(Dataset):
    
    def __init__(self, data, encoder):
        self.user_data = [torch.tensor(encoder.encode(did[0], 'users'), dtype=torch.int64) for did in data]

        self.genre_data = [encoder.one_hot_encode(did[1],'genres') for did in data]  # No need for torch.tensor if encoder.one_hot_encode returns a tensor
        self.country_data = [encoder.one_hot_encode(did[2], 'countries') for did in data]
        self.ratings_data = [torch.tensor(did[-1],dtype=torch.float) for did in data]
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
        # Start indexing from 1, reserving 0 for "unknown"
        vocab = {"<UNKNOWN>": 0}
        for item in items:
            if item not in vocab:
                vocab[item] = len(vocab)
        return vocab

    def update_vocab(self,new_items,category_name):
        """
        Update the vocabulary for a specific category with new items.
        This method adds any new items not already present in the vocabulary.
        
        :param category_name: The category for which to update the vocabulary.
        :param new_items: A list of new items to be added to the vocabulary.
        """
        # Flatten the items if they are lists
        flattened_items = []
        for item in new_items:
            if isinstance(item, list):
                flattened_items.extend(item)
            else:
                flattened_items.append(item)
    
        # Check if the category exists, create if not
        if category_name not in self.vocab_to_idx:
            self.vocab_to_idx[category_name] = {}
            self.idx_to_vocab[category_name] = {}
    
        current_vocab_size = len(self.vocab_to_idx[category_name])
        for item in flattened_items:
            if item not in self.vocab_to_idx[category_name]:
                self.vocab_to_idx[category_name][item] = current_vocab_size
                self.idx_to_vocab[category_name][current_vocab_size] = item
                current_vocab_size += 1

    def flatten_lists(self, items):
        # Split items by comma and return unique items
        split_items = set()
        for item in items:
            for subitem in item.split(','):
                split_items.add(subitem.strip())
        return list(split_items)

    def encode(self, item, category):
        # Handle both single items and lists for encoding
        # if isinstance(item, list):
        #     # Return a list of indices for list items
        #     return [self.vocab_to_idx[category].get(i, None) for i in item]
        # else:
        #     # Return a single index for a single item
        #     return self.vocab_to_idx[category].get(item, None)
        # Handle both single items and lists for encoding
        if isinstance(item, list):
            # Return a list of indices for list items, defaulting to 0 for unknown items
            return [self.vocab_to_idx[category].get(i, self.vocab_to_idx[category].get("<UNKNOWN>")) for i in item]
        else:
            # Return a single index for a single item, defaulting to 0 for unknown
            return self.vocab_to_idx[category].get(item, self.vocab_to_idx[category].get("<UNKNOWN>"))

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
