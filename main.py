from fastapi import FastAPI
# Import your model and any other dependencies

app = FastAPI()

import torch
# Load your model here (e.g., from a file)
from torch.utils.data import DataLoader

folder_dir = 'holocenemodels'
collab_model = torch.load_model(folder_dir+'best_colfilter_checkpoint.pth')
content_model = torch.load_model(folder_dir'+best_contentfiltermodel_checkpoint.pth')
from dataset import Encoder, ColFDataset,ConFDataset

import pickle

with open('col_encoder.pkl', 'rb') as file:
    col_encoder = pickle.load(file)

with open('con_encoder.pkl', 'rb') as file:
    con_encoder = pickle.load(file)

#test the encoder:

print(encoder.vocab_to_idx['users'][:10])

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def make_prediction(data: dict):
    # Process your input data and make a prediction with your model
    coldl,condl = preprocess_data(data)
    # prediction = model.predict(processed_data)
    result = run_prediction(coldl,condl)
    return {"prediction": "your_prediction_result"}

def preprocess_data(ddict):
    
    #check and update vocabs

    if set(ddict['user']) - encoder.vocab_to_idx['user'].keys():

            encoder.update_vocab(ddict['user'],'users')

    #for movie in ddict['movies']:

    if set(movie['movie']) - encoder.vocab_to_idx['movie'].keys()

            encoder.update_vocab(ddict['movie'],'movies')

    for feature in ddict['features']
        
        #traverse features.

        if set(feature[0] - encoder.vocab_to_idx['genres'].keys()
            
            encoder.update_vocab(feature[0].split(','),'genres')
            
        if set(feature[1] - encoder.vocab_to_idx['countries'].keys()
            
            encoder.update_vocab(feature[1],'countries')


    colds = ColFDataset([tuple([ddict['user'],ddict['movie'],0])],encoder)
            
    conds = ConFDataset([tuple([ddict['user'],ddict['movie'],0])],encoder)
    
    coldsdl = DataLoader(batch_size=1)    
    condsdl = DataLoader(batch_size=1)    


    return coldsdl,condsdl


def run_prediction(coldl,condl):
    """
    Makes predictions using both collaborative and content filter models,
    and returns their average score.

    :param collab_model: The collaborative filtering model
    :param content_model: The content filtering model
    :param user_ids: Tensor of user IDs
    :param movie_ids: Tensor of movie IDs (for collaborative model)
    :param movie_features: Tensor of movie features (for content model)
    :return: Average score from both models
    """
    # Ensure model is in evaluation mode
    collab_model.eval()
    content_model.eval()

    with torch.no_grad():

        for batch in coldl:

            collab_pred = collab_model(user_ids, movie_ids)
    
    
    with torch.no_grad():  # No need to track gradients

        for batch in condl:
            content_pred = content_model(user_ids, movie_features)
    
        # Calculate the average of the predictions
    avg_pred = (collab_pred + content_pred) / 2

    return avg_pred

