import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from fuzzywuzzy import process
import tensorflow.keras as keras
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.cold_start_model import top_k_movies
from src.models.item_based_cf_model import ItemBasedCF
from src.models.user_based_cf_model import UserBasedCF  # <-- موديل user-based CF
from src.models.DL_model import DL_MatrixFactorization  # عدل المسار حسب مشروعك

from src.data_loader import load_data

# Load data
ratings, movies, users = load_data()
item_based_model = ItemBasedCF()
user_based_model = UserBasedCF()

model_path = r"D:\Elovvo_Pathways\Movie_recommender\MF_checkpoints\mf_best_model.keras"
mf_model = keras.models.load_model(model_path, compile=False)

mf = DL_MatrixFactorization(embedding_size=100, load_from_checkpoint=True)  # load_from_checkpoint=True لتحميل النموذج المحفوظ

def show_user_recommendations(user_id):
    st.write(f"### 🎯 Recommended Movies for User {user_id}")
    
    # استخدم recommend_movies من كائن MF
    recommended_movies = mf.recommend_movies(user_id, n=10)
    
    cols = st.columns(3)
    for idx, row in recommended_movies.iterrows():
        col = cols[idx % 3]
        with col.container():
            st.markdown(f"### {row['title']}")
            if st.button(f"See Similar", key=f"sim_{user_id}_{row['item_id']}"):
                st.session_state.selected_movie = row['item_id']
            st.markdown("---")
            

# Precompute movie stats
movie_stats = ratings.groupby('item_id')['rating'].agg(['mean', 'count']).reset_index()
movies_with_stats = movies.merge(movie_stats, on='item_id', how='left')

st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎬 Movie Recommender")

# Session state
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None
if 'selected_user' not in st.session_state:
    st.session_state.selected_user = None

# --- Search Bar ---
def match_movie_name(movie_name):
    all_movie_titles = movies['title'].tolist()
    nearest_title = process.extractOne(movie_name, all_movie_titles)
    return nearest_title[0]

search_input = st.text_input("🔍 Search for a movie by name:")

if search_input:
    matched_title = match_movie_name(search_input)
    st.write(f"Closest match: **{matched_title}**")
    item_id = movies.loc[movies['title']==matched_title, 'item_id'].values[0]
    st.session_state.selected_movie = item_id

# --- Select User ---
user_id_input = st.selectbox("👤 Select User ID:", users['user_id'].tolist())
if user_id_input:
    st.session_state.selected_user = user_id_input


# --- Show Top Movies ---
def show_top_movies():
    top_movies = top_k_movies(k=10)
    if not top_movies.empty:
        st.write("### 🎬 Top Movies")
        cols = st.columns(3)
        for idx, row in top_movies.iterrows():
            col = cols[idx % 3]
            with col.container():
                st.markdown(f"### {row['title']}")
                st.markdown(f"⭐ **Rating:** {row['mean']:.2f} | 👥 **Votes:** {int(row['count'])}")
                if st.button(f"See Similar", key=f"similar_{row['item_id']}"):
                    st.session_state.selected_movie = row['item_id']
                st.markdown("---")
    else:
        st.warning("No Recommendations for now")

# --- Cold Start Button ---
st.write("### ⚡ Cold Start")
if st.button("Show Top-Rated Movies (Cold Start)"):
    st.session_state.selected_movie = None
    st.session_state.selected_user = None
    show_top_movies()


# --- Show Similar Movies ---
def show_similar_movies(item_id):
    movie_title = movies.loc[movies['item_id']==item_id, 'title'].values[0]
    st.write(f"### 🎥 Movies Similar to {movie_title}")
    
    similar_ids = item_based_model.find_similar_movies(item_id, k=6)
    similar_movies = movies_with_stats[movies_with_stats['item_id'].isin(similar_ids)]
    
    cols = st.columns(3)
    for idx, row in similar_movies.iterrows():
        col = cols[idx % 3]
        with col.container():
            st.markdown(f"### {row['title']}")
            st.markdown(f"⭐ **Rating:** {row['mean']:.2f} | 👥 **Votes:** {int(row['count'])}")
            if st.button(f"See Similar", key=f"similar_{row['item_id']}"):
                st.session_state.selected_movie = row['item_id']
            st.markdown("---")

# --- Show User Recommendations ---
def show_user_recommendations(user_id):
    st.write(f"### 🎯 Recommended Movies for User {user_id}")
    recommended_ids = user_based_model.recommend_movies(user_id, k=10, n_recommendations=10)
    recommended_movies = movies_with_stats[movies_with_stats['item_id'].isin(recommended_ids)]
    
    cols = st.columns(3)
    for idx, row in recommended_movies.iterrows():
        col = cols[idx % 3]
        with col.container():
            st.markdown(f"### {row['title']}")
            st.markdown(f"⭐ **Rating:** {row['mean']:.2f} | 👥 **Votes:** {int(row['count'])}")
            if st.button(f"See Similar", key=f"similar_{row['item_id']}"):
                st.session_state.selected_movie = row['item_id']
            st.markdown("---")



# --- Main Logic ---
if st.session_state.selected_movie is not None:
    show_similar_movies(st.session_state.selected_movie)
    if st.button("⬅️ Back to Top Movies"):
        st.session_state.selected_movie = None
elif st.session_state.selected_user is not None:
    show_user_recommendations(st.session_state.selected_user)
else:
    show_top_movies()
