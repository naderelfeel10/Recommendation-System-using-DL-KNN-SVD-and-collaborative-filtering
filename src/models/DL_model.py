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

from tensorflow.keras import layers , models , metrics , optimizers , regularizers
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint
import joblib
from tensorflow.keras.models import load_model



class DL_MatrixFactorization:
    def __init__(self,embedding_size=100,load_from_checkpoint=False):
        self.ratings ,self.movies , self.users = load_data()
        self.n_users = 0
        self.n_movies = 0
        self.embedding_size = embedding_size
        self.model = None

        self.checkpoint_dir = os.path.join(os.path.dirname(__file__), '../../MF_checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(self.checkpoint_dir, 'mf_best_model.keras')
        user_enc_path = os.path.join(self.checkpoint_dir, 'user_enc.pkl')
        movie_enc_path = os.path.join(self.checkpoint_dir, 'movie_enc.pkl')


        if load_from_checkpoint and os.path.exists(checkpoint_path):
            print("Loading model from checkpoint...")
            self.model = load_model(checkpoint_path)
            self.user_enc = joblib.load(user_enc_path)
            self.movie_enc = joblib.load(movie_enc_path)
            self.n_users = len(self.user_enc.classes_)
            self.n_movies = len(self.movie_enc.classes_)



    def MF_preprocess(self):
            
            self.user_enc = LabelEncoder()
            self.ratings['user'] = self.user_enc.fit_transform(self.ratings.user_id.values)
            self.n_users = self.ratings['user_id'].nunique()

            self.movie_enc = LabelEncoder()
            self.ratings['movie'] = self.movie_enc.fit_transform(self.ratings.item_id.values)
            self.n_movies = self.ratings['item_id'].nunique()

            X = self.ratings[['user', 'movie']].values
            y = self.ratings['rating'].values
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=42)
            #n_users, n_movies


    def model_structure(self):
            
            #embedding_size = 100

            user = layers.Input(shape=(1,) , name='user_id')
            user_emb = layers.Embedding(self.n_users,self.embedding_size,embeddings_regularizer=regularizers.l2(1e-6) , name='user_embedding_LUT')(user)
            user_emb = layers.Reshape((self.embedding_size,))(user_emb)
            
            movie = layers.Input(shape=(1,) , name='movie_id')
            movie_emb = layers.Embedding(self.n_movies,self.embedding_size,embeddings_regularizer=regularizers.l2(1e-6),name='movie_embedding_LUT')(movie)
            movie_emb = layers.Reshape((self.embedding_size,))(movie_emb)
            
            x = layers.Dot(axes=1 , name='sim_measure')([user_emb,movie_emb]) 
            
            #x = layers.Dense(64,activation='relu')(x)
            #x = layers.Dense(1,activation='linear',name='predicted_rating')(x)
            
            
            self.model = models.Model(inputs=[user,movie],outputs=x)
            
            self.model.compile(
                loss = 'mse',
                optimizer = optimizers.AdamW(learning_rate=0.001),
                metrics = [metrics.RootMeanSquaredError()]
            )
            
            self.model.summary()
            plot_model(self.model , show_shapes=True , show_layer_names=True)

    def fit(self):
        self.MF_preprocess()
        self.model_structure()
        early_stopping = EarlyStopping(
            monitor='val_root_mean_squared_error',  
            patience=3, 
            restore_best_weights=True, 
            verbose=1
        )

        checkpoint_path = os.path.join(self.checkpoint_dir, 'mf_best_model.keras')
        checkpoint_cb = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_root_mean_squared_error',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )

        self.model.compile(loss='mse',  metrics=[metrics.RootMeanSquaredError()],
                      optimizer=optimizers.Adam(learning_rate=0.001))
        history = self.model.fit(
            x=[self.X_train[:, 0], self.X_train[:, 1]],
            y=self.y_train,

            batch_size=64,
            epochs=20,
            verbose=1,
            validation_data=([self.X_test[:, 0], self.X_test[:, 1]], self.y_test),
            callbacks=[early_stopping,checkpoint_cb]

        )


    def recommend_movies(self,user_id, n=10):

        user_idx = self.user_enc.transform([user_id])[0]


        movie_indices = np.arange(self.n_movies)

        preds = self.model.predict(
            [np.full_like(movie_indices, user_idx), movie_indices],
            verbose=0
        )


        top_indices = preds.flatten().argsort()[-n:][::-1]


        recommended_movie_ids = self.movie_enc.inverse_transform(top_indices)

        recommendations = self.movies[self.movies['item_id'].isin(recommended_movie_ids)][['item_id', 'title']]
        recommendations = recommendations.set_index('item_id').loc[recommended_movie_ids].reset_index()

        return recommendations
    



mf = DL_MatrixFactorization(100)
mf.fit()
print(mf.recommend_movies(5))