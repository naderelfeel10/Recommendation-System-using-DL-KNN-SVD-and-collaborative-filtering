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



def create_X(df):
    M = df['user_id'].nunique()
    N = df['item_id'].nunique()

    user_mapper = dict(zip(np.unique(df["user_id"]), list(range(M))))
    movie_mapper = dict(zip(np.unique(df["item_id"]), list(range(N))))
    

    user_inv_mapper = dict(zip(list(range(M)), np.unique(df["user_id"])))
    movie_inv_mapper = dict(zip(list(range(N)), np.unique(df["item_id"])))

    user_index = [user_mapper[i] for i in df['user_id']]
    item_index = [movie_mapper[i] for i in df['item_id']]

    X = csr_matrix((df["rating"], (user_index, item_index)), shape=(M, N))

    return  X,user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper


def match_movie_name(movie_name):
    _ , movies, _ = load_data()
    all_movie_titles = movies['title'].tolist()
    nearest_title = process.extractOne(movie_name,all_movie_titles)
    return nearest_title[0]


class ItemBasedCF:

    def __init__(self):
        data = preprocess()
        _ , self.movies, _ = load_data()

        self.train_df, self.test_df = train_test_split(data, test_size=0.2, random_state=42)
    
        self.train_matrix = self.train_df.pivot(index = 'user_id' , columns='item_id' , values = 'rating')
        self.train_matrix.fillna(0,inplace=True)
    
        self.test_matrix = self.test_df.pivot(index = 'user_id' , columns='item_id' , values = 'rating')
        self.test_matrix.fillna(0,inplace=True)
    
        self.X_train = csr_matrix(self.train_matrix.values)
        self.X_test = csr_matrix(self.test_matrix.values)
    
        _,self.user_mapper, self.movie_mapper, self.user_inv_mapper, self.movie_inv_mapper = create_X(data)
        self.model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model_knn.fit(self.train_matrix.T)        




    def find_similar_movies(self,movie_id, k=5):

        X = self.X_train
        X = X.T
        neighbour_ids = []

        movie_ind = self.movie_mapper[movie_id]
        movie_vec = X[movie_ind]

        if isinstance(movie_vec, (np.ndarray)):
            movie_vec = movie_vec.reshape(1,-1)

        # use k+1 since kNN output includes the movieId of interest
        kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric='cosine')
        kNN.fit(X)
        neighbour = kNN.kneighbors(movie_vec, return_distance=False)
        for i in range(0,k):
            n = neighbour.item(i)
            neighbour_ids.append(self.movie_inv_mapper[n])
        neighbour_ids.pop(0)
        return neighbour_ids
    


    def recommend_movies(self,user_id,n_recommendations=10):

        rated_movies = self.train_matrix.loc[user_id]
        rated_movies = rated_movies[rated_movies>=4].index
        if len(rated_movies) > 0 :
            movie_id = np.random.choice(rated_movies)

        #title = match_movie_name(movie_name)
        #movie_id = self.data[self.data['title']==title]['item_id'].values[0]

        similar_ids = self.find_similar_movies(movie_id, n_recommendations)
        similar_movies = self.movies[self.movies['item_id'].isin(similar_ids)][['item_id','title']]
        
        return similar_movies

    

    def mean_precision_at_k(self,k):
        precisions = []
        for user_id in self.test_matrix.index:
            rated_movies = self.train_matrix.loc[user_id]
            rated_movies = rated_movies[rated_movies>=4].index
            if len(rated_movies) > 0 :
                movie_id = np.random.choice(rated_movies)

            recommended_movies = self.find_similar_movies(movie_id,k)

            liked = self.test_matrix.loc[user_id][self.test_matrix.loc[user_id] >= 4].index
            correct = len(set(recommended_movies) & set(liked))
            precision = correct / k
            precisions.append(precision)
        return precisions    
    



#item_based = ItemBasedCF()
#precisions = item_based.mean_precision_at_k(10)
#mean_precision = np.mean(precisions)
#print("mean precisions : ",mean_precision)

#print(item_based.recommend_movies(474))
#print(item_based.find_similar_movies(1,10))


















