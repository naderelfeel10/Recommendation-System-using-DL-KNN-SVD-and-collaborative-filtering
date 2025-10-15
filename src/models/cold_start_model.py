import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.preprocess import preprocess
from src.data_loader import load_data

# Cold start and most simple method : 
def weighted_rating(x, M, C):
    v = x['count']
    R = x['mean']
    return (v/(v+M) * R) + (M/(M+v) * C)


def top_k_movies(k=10):
    data = preprocess()
    ratings , movies , users = load_data()
    #print(data)
    movie_stats = data.groupby('item_id')['rating'].agg(['count', 'mean']).reset_index()
    C = movie_stats['mean'].mean()
    M = movie_stats['count'].quantile(0.90)

    movie_stats['score'] = weighted_rating(movie_stats,M,C)
    movie_stats = movie_stats.merge(movies[['item_id','title']] , on='item_id')
    top_movies = movie_stats.sort_values('score', ascending=False).head(10)

    return top_movies

#print(top_k_movies())







