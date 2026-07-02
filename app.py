import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load saved files
tfidf = joblib.load('tfidf_vectorizer.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')
anime = joblib.load('anime_data.pkl')
indices = joblib.load('indices.pkl')

# Recommendation function
def recommend_anime(title, top_n=10):
    try:
        idx = indices[title]
    except:
        return ["Anime Not Found"]
    
    # Compute Similarity Scores
    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]))

    # Top N Recommendations
    sim_scores = sim_scores[1:top_n+1]

    anime_indices = [i[0] for i in sim_scores]
    return anime['name'].iloc[anime_indices].values

# UI
st.title("Anime Recommendation System")

anime_name = st.text_input("Enter an Anime Name:")
if st.button("Recommend"):
    results = recommend_anime(anime_name)

    st.subheader("Recommended Animes:")
    for i, anime_title in enumerate(results, 1):
        st.write(f"{i}. {anime_title}")
else:
    st.warning("Please enter an anime name to get recommendations.")