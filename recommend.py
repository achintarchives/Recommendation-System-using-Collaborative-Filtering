import streamlit as st

st.title("🎬 CineMatch Movie Recommender")

user_id = st.selectbox(
    "Select User",
    user_movie_matrix.index.tolist()
)

if st.button("Get Recommendations"):
    recs = recommend_movies(user_id, n=5)

    st.subheader("Top Recommendations")
    for movie_id, score in recs:
        title = movies[movies['movieId'] == movie_id]['title'].values
        title = title[0] if len(title) > 0 else "Unknown"
        st.write(f"{title} — Predicted Rating: {score:.2f}")
