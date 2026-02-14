import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Movie Recommendation System")

@st.cache_resource
def load_model():

    movies = pd.read_csv("movies_metadata.csv", low_memory=False)
    movies = movies[['title', 'overview', 'genres']]
    movies.dropna(inplace=True)

    def convert(text):
        try:
            L = []
            for i in ast.literal_eval(text):
                L.append(i['name'])
            return L
        except:
            return []

    movies['genres'] = movies['genres'].apply(convert)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    movies['tags'] = movies['overview'] + movies['genres']
    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

    new_df = movies[['title','tags']].head(10000)
    new_df.reset_index(drop=True, inplace=True)

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()

    return new_df, vectors

new_df, vectors = load_model()

def recommend(movie):
    if movie not in new_df['title'].values:
        return ["Movie not found!"]

    movie_index = new_df[new_df['title'] == movie].index[0]
    movie_vector = vectors[movie_index].reshape(1, -1)

    similarity_scores = cosine_similarity(movie_vector, vectors)[0]

    similar_movies = sorted(
        list(enumerate(similarity_scores)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    return [new_df.iloc[i[0]].title for i in similar_movies]

movie_name = st.text_input("Enter movie name")

if st.button("Recommend"):
    recommendations = recommend(movie_name)
    for movie in recommendations:
        st.write(movie)
