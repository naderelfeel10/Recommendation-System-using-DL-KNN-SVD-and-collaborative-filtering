import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.preprocess import preprocess
from src.data_loader import load_data

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import svds
import pandas as pd

class SVD_MF:
    def __init__(self):
        self.ratings , self.movies , self.users = load_data()

        self.user_item_matrix = self.ratings.pivot(index='user_id',columns='item_id',values='rating')
        self.user_item_matrix.fillna(0,inplace=True)

        self.R = self.user_item_matrix.values
        self.user_ids = self.user_item_matrix.index
        self.movie_ids = self.user_item_matrix.columns

    def fit(self,k=50):
        self.U, self.sigma, self.Vt = svds(self.R, k=k)
        self.sigma = np.diag(self.sigma)
        self.predicted_ratings = np.dot((np.dot(self.U,self.sigma)),self.Vt)
        self.R_pred_df = pd.DataFrame(self.predicted_ratings, index=self.user_ids, columns=self.movie_ids)


    def recommend_movies(self,user_id, n=10):


        user_ratings = self.user_item_matrix.loc[user_id]
        preds = self.R_pred_df.loc[user_id]


        already_rated = user_ratings[user_ratings > 0].index
        preds = preds.drop(already_rated)

        top_movies = preds.sort_values(ascending=False).head(n).index

        return self.movies[self.movies['item_id'].isin(top_movies)][['item_id', 'title']]




#svd = SVD_MF()
#svd.fit()
#print(svd.recommend_movies(5))






