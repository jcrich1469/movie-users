from fastapi import FastAPI
# Import your model and any other dependencies
app = FastAPI()

import torch
# Load your model here (e.g., from a file)
from torch.utils.data import DataLoader
from dataset import Encoder, ColFDataset,ConFDataset
import torch
from torch.nn.functional import cosine_similarity
from cf_model import MLPCollaborativeFilter,MLPContentFilter  # Import your model class
import pickle


# Replace 'yourfile.pkl' with the path to your actual pickle file
pickle_file_path = 'col_encoder.pkl'

# Open the file in binary read mode
with open(pickle_file_path, 'rb') as file:
    # Load the object from the file
    col_encoder = pickle.load(file)

# Replace 'yourfile.pkl' with the path to your actual pickle file
pickle_file_path = 'con_encoder.pkl'

# Open the file in binary read mode
with open(pickle_file_path, 'rb') as file:
    # Load the object from the file
    con_encoder = pickle.load(file)
    




FEATURES = 700
#col features
num_users = len(col_encoder.vocab_to_idx['users'])+1
num_movies = len(col_encoder.vocab_to_idx['movies'])+1
#con features
num_mv_features = len(con_encoder.one_hot_encode('Action','genres')) + len(con_encoder.one_hot_encode('Bulgaria','countries'))
con_num_users = len(con_encoder.vocab_to_idx['users'])




col_model = MLPCollaborativeFilter(num_users, num_movies, embedding_dim=FEATURES)

# Load the entire checkpoint
checkpoint = torch.load('./final_col_model_checkpoint.pth')

# Extract the model's state dictionary from the checkpoint
model_state_dict = checkpoint['model_state_dict']

# Now load the state dictionary into your model
col_model.load_state_dict(model_state_dict)

con_model = MLPContentFilter(len(con_encoder.vocab_to_idx['users']),num_mv_features, embedding_dim=FEATURES)
# Load the entire checkpoint
checkpoint = torch.load('./final_con_model_checkpoint.pth')
# Extract the model's state dictionary from the checkpoint
model_state_dict = checkpoint['model_state_dict']
# Now load the state dictionary into your model
con_model.load_state_dict(model_state_dict)


def find_similar_users_avghybrid(target_user, genre, country, top_n=10):
    """
    Find top_n most similar users to the target_user_id based on their user embeddings.

    Args:
    - model: The trained collaborative filtering model with user embeddings.
    - target_user_id: The ID of the user for whom to find similar users.
    - top_n: Number of similar users to retrieve.

    Returns:
    - top_similar_users: Indices of the top_n similar users.
    """
    # Assuming 'user_embeddings' is retrieved from your model
    user_embeddings = col_model.user_embedding.weight.data

    # Ensure target_user_id is valid
    target_user_enc = col_encoder.encode(target_user,'users')
    print(target_user_enc)
    if target_user_enc >= len(user_embeddings):
        raise ValueError("Target user ID is out of range.")

    target_user_embedding = user_embeddings[target_user_enc].unsqueeze(0)
    cf_similarities = cosine_similarity(target_user_embedding, user_embeddings)



    from dataset import ConFDataset

    data = [(target_user,genre,country,0.0)]

    ds = ConFDataset(data,con_encoder)

    dl = torch.utils.data.DataLoader(ds, batch_size=1)

    con_model.eval() # swtich off batch normalisation
    with torch.no_grad():  # Disable gradient computation
        for user, movie_features, ratings in dl:
            cbf_scores = con_model(user,movie_features)  # Generate predictions


    # Step 3: Average CF and CBF similarities/scores
    combined_scores = (cf_similarities.squeeze() + cbf_scores) / 2

    print('combined_scores :',combined_scores)
    # Step 4: Find top similar users, excluding the target user
    _, indices = torch.topk(combined_scores, top_n + 1)  # Get indices of top scores
    top_similar_users = [con_encoder.decode(idx.item(),'users') for idx in indices if idx != target_user_enc][:top_n]  # Exclude target user

    return top_similar_users





@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/matchuser")
async def make_matches(data: dict):
    # Process your input data and make a prediction with your model
    n,g,c = preprocess_data(data)
    
    result = find_similar_users_avghybrid(n,g,c)

    return result

def preprocess_data(ddict):
    
    name = ddict['name']
    genre = ddict['genre']
    country = ddict['country']
    return name,genre,country
    
    


