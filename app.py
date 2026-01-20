import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from ai.model import (
    recommend_movies,
    hybrid_recommend_movies,
    user_movie_matrix,
    movies
)

import streamlit as st
import sys
import os

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide"
)

import base64
import streamlit as st
def get_base64_bg_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
BG_IMAGE_PATH = r"D:\Untitled design_20260117_190625_0000.jpg"
bg_image_base64 = get_base64_bg_image(BG_IMAGE_PATH)

st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            url("data:image/jpg;base64,{bg_image_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #e5e7eb;
        font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont;
    }}

    section[data-testid="stSidebar"] {{
        background-color: #12141b;
        border-right: 1px solid #1f2230;
    }}

    h1, h2, h3 {{
        font-weight: 800 !important;
        letter-spacing: -0.03em;
        color: #ffffff;
    }}

    p, span, label {{
        color: #cbd0dd;
        font-size: 0.95rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)





st.title(" CineMatch Movie Recommender")
st.caption("Top Recommended System and Hybrid Recommendation System")

user_id = st.selectbox(
    "Select a User",
    user_movie_matrix.index.tolist()
)
st.divider()

if st.button("Get Top Recommendations"):
    recs = recommend_movies(user_id, top_n=5)

    if not recs:
        st.warning("No Top Recommendations found for this user.")
    else:
        st.subheader(" Top (Recommendation Filtering)")
        for movie_id, score in recs:
            title = movies[movies['movieId'] == movie_id]['title'].values
            title = title[0] if len(title) > 0 else "Unknown Title"
            st.write(f"🎬 **{title}** — Predicted Rating: {score:.2f}")

st.divider()

# HYBRID RECOMMENDATION


if st.button("Get Hybrid Recommendations"):
    recs = hybrid_recommend_movies(user_id, top_n=5)

    if not recs:
        st.warning("No hybrid recommendations found for this user.")
    else:
        st.subheader(" Hybrid Recommendation (CF + Genres)")
        for movie_id, score in recs:
            title = movies[movies['movieId'] == movie_id]['title'].values
            title = title[0] if len(title) > 0 else "Unknown Title"
            st.write(f"🎬 **{title}** — Hybrid Score: {score:.2f}")
