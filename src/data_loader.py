import pandas as pd
import numpy as np

datapath = 'D:/Elovvo_Pathways/Movie_recommender/data/ml-100k'
def load_data(datapath=datapath):
    ratings = pd.read_csv(f'{datapath}/u.data', sep='\t', names=['user_id','item_id','rating','timestamp'])
    movies = pd.read_csv(f'{datapath}/u.item', sep='|', encoding='latin-1', names=[
        'item_id','title','release_date','video_release_date','IMDb_URL','unknown',
        'Action','Adventure','Animation','Children','Comedy','Crime','Documentary',
        'Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi',
        'Thriller','War','Western'
    ])
    users = pd.read_csv(f'{datapath}/u.user', sep="|", names=["user_id", "age", "gender", "occupation", "zip_code"])
    return ratings , movies ,users

