from src.data_loader import load_data
import pandas as pd
import numpy as np




def preprocess():


    ratings , movies,users = load_data()
    #print(ratings.shape)
    #print(movies.shape)
    #print(users.shape)


    movies.drop(columns=['video_release_date','IMDb_URL'] , axis=1, inplace=True)
    num_ratings_per_user = ratings.groupby('user_id')['rating'].count()
    num_ratings_per_movie = ratings.groupby('item_id')['rating'].count()

    movies = pd.merge(movies,num_ratings_per_movie,on='item_id')
    movies_ratings_merged = pd.merge(ratings[['user_id','item_id','rating']],movies[['title','item_id','rating']],on='item_id',how='inner')
    users_percentile = 75
    threshold_users = np.percentile(num_ratings_per_user,users_percentile)
    movies_ratings_merged = movies_ratings_merged[movies_ratings_merged['rating_y']>=threshold_users]

    users = pd.merge(users,num_ratings_per_user,on='user_id')
    data_merged = pd.merge(movies_ratings_merged[['user_id','item_id','rating_x','title','rating_y']],users[['user_id','rating']],on='user_id',how='inner')
    movies_percentile = 75
    threshold_movies = np.percentile(num_ratings_per_movie,movies_percentile)

    data_merged = data_merged[data_merged['rating']>=threshold_movies][['user_id','item_id','rating_x','title']]
    data_merged = data_merged.rename(columns={'rating_x':'rating'})
    return data_merged


#data = preprocess()

#print(data)






