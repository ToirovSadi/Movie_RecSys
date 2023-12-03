import torch
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np

# metrics:
#   - loss: mse
#   - accuracy: satisfactory score, predict rating and if it's >3 it's good else bad (binary classification)

def calc_loss(model, user_id, movie_id, movie_feat, rating, device='cpu'):
    user_id = torch.tensor([user_id]).to(device)
    movie_id = torch.tensor([movie_id]).to(device)
    movie_feat = torch.tensor(movie_feat).unsqueeze(0).to(device)
    rating = torch.tensor([rating]).unsqueeze(0).to(device)
    model.eval()
    model.to(device)
    with torch.no_grad():
        pred_rating = model(user_id, movie_id, movie_feat)
        loss = F.mse_loss(pred_rating, rating)
    pred_rating = pred_rating.cpu().item()
    loss = loss.cpu().item()
    
    return loss, pred_rating 
    

def evaluate(model, val_dataset, already_watched, movies):
    loss = []
    accuracy = []
    
    predictions = dict()
    # predict for each user in val_dataset a list of movies
    for user_id in tqdm(val_dataset['user_id'].unique(), desc='Predicting'):
        aw = already_watched[already_watched['user_id'] == user_id]['already_watched'].to_list()[0]
        aw = aw.split(' ')
        aw = [int(x) for x in aw]
        
        pred = predict_movie(model, user_id, aw, movies, top_k=20, return_rating=True)
        
        # movies in val dataset
        watch = list(val_dataset[val_dataset['user_id'] == user_id]['item_id'].unique())
        watch = [x for x in watch if x not in aw] # exclude movies already watched
        
        # convert id of movies to their names
        watch = [movies[movies['item_id'] == idx]['movie_title'].to_list()[0] for idx in watch]
        predictions[user_id] = {
            'pred': pred,
            'watch': watch,
        }
    
    for i in tqdm(val_dataset.index, desc='Calculating Metrics'):
        user_id = val_dataset['user_id'][i]
        movie_id = val_dataset['item_id'][i]
        movie_feat = val_dataset.iloc[i, 7:].to_list()
        rating = val_dataset['rating'][i]
        
        # calculate loss (how it reconstructs the rating)
        _loss, _pred_rating = calc_loss(model, user_id, movie_id, movie_feat, rating)
        loss.append(_loss)
        accuracy.append(int((rating >= 2.5) == (_pred_rating >= 2.5)))
        
        p = predictions[user_id]
        pred, watch = p['pred'], p['watch']
        pred = [x[0] for x in pred] # take only name's
    
    loss = np.mean(loss)
    accuracy = np.mean(accuracy)
    
    return loss, accuracy


# predict movie
# model - 
# user_id - for a particular user to make prediction
# already_watched - already_watched movies (only id of movies given)
# top_k - top movies to return
# device - device to perform operations
# return_rating - return rating with movie names or not
def predict_movie(model, user_id, already_watched, movies, top_k=None, device='cpu', return_rating=False):  
    # exclude already watched movies
    movie_feat = movies.iloc[:, 4:].to_numpy(dtype=np.int64)
    movie_feat = [
        (movies['item_id'][idx], movie_feat[i])
        for i, idx in enumerate(movies.index)
        if movies['item_id'][idx] not in already_watched
    ]
    
    movie_feat_tensor = torch.stack([torch.tensor(x[1]) for x in movie_feat], dim=0)
    
    user_id = torch.tensor([user_id] * len(movie_feat))
    items_id = torch.tensor([x[0] for x in movie_feat])

    # now make predictions
    model.to(device)
    model.eval()
    with torch.no_grad():
        user_id = user_id.to(device)
        items_id = items_id.to(device)
        movie_feat_tensor = movie_feat_tensor.to(device)
        pred_rating = model(user_id, items_id, movie_feat_tensor)
        pred_rating = pred_rating.squeeze(1).cpu().numpy()
    
    # retrieve name of movies
    pred_movies = [(pred_rating[i], movie_feat[i][0]) for i in range(len(pred_rating))]
    
    pred_movies = sorted(pred_movies, key=lambda x: x[0], reverse=True)
    if top_k:
        pred_movies = pred_movies[:top_k]
    
    output = []
    for rating, idx in pred_movies:
        movie_name = movies[movies['item_id'] == idx]['movie_title'].to_list()[0]
        # print(movies[movies['item_id'] == idx]['movie_title'].to_list()[0])
        if return_rating:
            output.append((movie_name, rating))
        else:
            output.append(movie_name)
    
    return output