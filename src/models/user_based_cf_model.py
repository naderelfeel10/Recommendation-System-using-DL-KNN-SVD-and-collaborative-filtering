import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.preprocess import preprocess
from src.data_loader import load_data

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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


def get_similar_users(user_id, user_mapper,user_inv_mapper,user_similarity,k):
    user_indx = user_mapper[user_id]
    sim_row = list(enumerate(user_similarity[user_indx]))
    sim_row = sorted(sim_row, key=lambda x: x[1], reverse=True)

    top_k = sim_row[1:k+1]
    similar_user_indices = [i[0] for i in top_k] 
    return similar_user_indices



class UserBasedCF():

    def __init__(self):
        data = preprocess()

        self.train_df, self.test_df = train_test_split(data, test_size=0.2, random_state=42)
    
        self.train_matrix = self.train_df.pivot(index = 'user_id' , columns='item_id' , values = 'rating')
        self.train_matrix.fillna(0,inplace=True)
    
        self.test_matrix = self.test_df.pivot(index = 'user_id' , columns='item_id' , values = 'rating')
        self.test_matrix.fillna(0,inplace=True)
    
    
        self.X_train = csr_matrix(self.train_matrix.values)
        self.X_test = csr_matrix(self.test_matrix.values)
    
        _,self.user_mapper, self.movie_mapper, self.user_inv_mapper, self.movie_inv_mapper = create_X(data)
    
        self.user_similarity = cosine_similarity(self.X_train)


    def recommend_movies(self,user_id,k,n_recommendations):
            user_index = self.user_mapper[user_id]

            top_k_similar_users = get_similar_users(user_id,self.user_mapper,self.user_inv_mapper,self.user_similarity,k)  #top k similar user ids
            
            top_k_sim_scores = self.user_similarity[user_index][top_k_similar_users]
            
            X_k = self.X_train[top_k_similar_users].toarray()
            weighted_sum = top_k_sim_scores @ X_k
            normalization = np.array([np.abs(top_k_sim_scores).sum()] * self.X_train.shape[1] )
            predicted_ratings = weighted_sum / normalization
        
            user_ratings = self.X_train[user_index].toarray().flatten()
            predicted_ratings[user_ratings > 0] = 0
        
            top_movies_idx = np.argsort(predicted_ratings)[::-1][:n_recommendations]
            recommend_movies_ids = [self.movie_inv_mapper[i] for i in top_movies_idx]
            return recommend_movies_ids
    

    def mean_precision_at_k(self,k=50,n_recommendations=10):
        precisions = []
        for user_id in self.test_matrix.index:
            recommended_movies = self.recommend_movies(user_id,k,n_recommendations)

            relevant_items = self.test_matrix.loc[user_id]
            relevant_items = relevant_items[relevant_items > 0].index.tolist()
            correct = len(set(recommended_movies) & set(relevant_items))

            precision = correct / n_recommendations
            #print("precision : ",precision)
            precisions.append(precision)

        return precisions
    


#user_based_recommendation = UserBasedCF()
#precisions = user_based_recommendation.mean_precision_at_k(k=50,n_recommendations=10)
#mean_precision = sum(precisions)/len(precisions)
#print("mean Test precisions : ",mean_precision)    
    



















