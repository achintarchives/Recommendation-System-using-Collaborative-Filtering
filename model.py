# CineMatch: Movie Recommendation System
# Approach: Collaborative Filtering (Memory + Model Based)
# Dataset: MovieLens

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 2. LOAD DATA

# Load ratings and movies datasets
ratings = pd.read_csv(r"D:\movielens ds\ratings.csv")
movies = pd.read_csv(r"D:\movielens ds\movies.csv")

# Basic inspection
print("Ratings Columns:", ratings.columns)
print("\nRatings Preview:")
print(ratings.head())

print("\nRatings Info:")
print(ratings.info())

# 3. EXPLORATORY DATA ANALYSIS (EDA)

# 3.1 Rating Distribution
plt.figure(figsize=(6, 4))
sns.histplot(ratings['rating'], bins=10, kde=True)
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()

# 3.2 Ratings per User
user_activity = ratings.groupby('userId').size()
print("\nUser Rating Count Statistics:")
print(user_activity.describe())

# 4. USER–ITEM INTERACTION MATRIX

# Create User–Movie matrix
user_movie_matrix = ratings.pivot(
    index='userId',
    columns='movieId',
    values='rating'
)

# Calculate sparsity
total_possible_ratings = user_movie_matrix.size
actual_ratings = user_movie_matrix.notna().sum().sum()

sparsity = 1 - (actual_ratings / total_possible_ratings)
print("\nMatrix Sparsity:", sparsity)

# 5. MEMORY-BASED COLLABORATIVE FILTERING

# Fill missing ratings with 0 for similarity computation
user_movie_filled = user_movie_matrix.fillna(0)

# Compute Item–Item Cosine Similarity
item_similarity = cosine_similarity(user_movie_filled.T)

item_similarity_df = pd.DataFrame(
    item_similarity,
    index=user_movie_filled.columns,
    columns=user_movie_filled.columns
)

print("\nItem–Item Similarity Matrix Shape:", item_similarity_df.shape)

# SVD chosen due to strong performance on sparse user–item matrices


# 6. MODEL-BASED COLLABORATIVE FILTERING (SVD)

# Convert interaction matrix to NumPy array
R = user_movie_filled.values

# Apply Truncated SVD for Matrix Factorization
svd = TruncatedSVD(n_components=20, random_state=42)

# User latent factors
P = svd.fit_transform(R)

# Item latent factors (transposed)
Q_T = svd.components_

# Reconstruct predicted ratings matrix
R_hat = np.dot(P, Q_T)

print("\nPredicted Ratings Matrix Shape:", R_hat.shape)

# 7. TRAIN–TEST SPLIT FOR EVALUATION

train, test = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)

# Create index mappings
user_index = {user_id: idx for idx, user_id in enumerate(user_movie_matrix.index)}
movie_index = {movie_id: idx for idx, movie_id in enumerate(user_movie_matrix.columns)}

# 8. PREDICT TEST RATINGS

predicted_ratings = []

for row in test.itertuples():
    u_idx = user_index[row.userId]
    m_idx = movie_index[row.movieId]
    predicted_ratings.append(R_hat[u_idx, m_idx])

# 9. MODEL EVALUATION (RMSE)

rmse = np.sqrt(mean_squared_error(test['rating'], predicted_ratings))
print("\nModel RMSE:", rmse)


# 10. RECOMMENDATION FUNCTION

def recommend_movies(user_id, top_n=5):
    """
    Generate top-N movie recommendations for a given user
    using predicted ratings from matrix factorization.
    """
    user_idx = user_index[user_id]

    # Movies already rated by user
    user_ratings = user_movie_matrix.loc[user_id]

    # Identify unseen movies
    unseen_movies = user_ratings[user_ratings.isna()].index

    # Score unseen movies
    scores = {
        movie_id: R_hat[user_idx, movie_index[movie_id]]
        for movie_id in unseen_movies
    }

    # Sort movies by predicted rating
    top_recommendations = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return top_recommendations

# 11. EXAMPLE RECOMMENDATION OUTPUT

example_user = user_movie_matrix.index[0]
recommendations = recommend_movies(example_user, top_n=5)

print(f"\nTop Recommendations for User {example_user}:")

for movie_id, score in recommendations:
    title = movies[movies['movieId'] == movie_id]['title'].values
    title = title[0] if len(title) > 0 else "Unknown Title"
    print(f"{title} — Predicted Rating: {score:.2f}")


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

# GENRE MATRIX (CONTENT FEATURES)

# Split genres into lists
movies['genre_list'] = movies['genres'].apply(lambda x: x.split('|'))

# One-hot encode genres
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movies['genre_list'])

genre_df = pd.DataFrame(
    genre_matrix,
    index=movies['movieId'],
    columns=mlb.classes_
)

# Precompute genre similarity between movies
genre_similarity = cosine_similarity(genre_df)
genre_similarity_df = pd.DataFrame(
    genre_similarity,
    index=genre_df.index,
    columns=genre_df.index
)

def content_score(user_id, movie_id, threshold=4.0):
    """
    Computes content-based score for a user–movie pair
    based on genre similarity.
    """

    # Movies the user liked
    liked_movies = ratings[
        (ratings['userId'] == user_id) &
        (ratings['rating'] >= threshold)
    ]['movieId']

    if len(liked_movies) == 0:
        return 0.0

    # Average genre similarity
    similarities = [
        genre_similarity_df.loc[movie_id, liked_movie]
        for liked_movie in liked_movies
        if liked_movie in genre_similarity_df.columns
    ]

    if len(similarities) == 0:
        return 0.0

    return np.mean(similarities)

def hybrid_recommend_movies(user_id, top_n=5, alpha=0.7):
    """
    Hybrid recommender combining:
    - Collaborative Filtering (SVD)
    - Content-Based Filtering (Genres)
    """

    user_idx = user_index[user_id]
    user_ratings = user_movie_matrix.loc[user_id]

    # Unseen movies
    unseen_movies = user_ratings[user_ratings.isna()].index

    hybrid_scores = {}

    for movie_id in unseen_movies:
        # Collaborative score
        collab = R_hat[user_idx, movie_index[movie_id]]

        # Content score
        content = content_score(user_id, movie_id)

        # Hybrid score
        final_score = alpha * collab + (1 - alpha) * content

        hybrid_scores[movie_id] = final_score

    # Top-N movies
    top_movies = sorted(
        hybrid_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return top_movies

if __name__ == "__main__":
    sample_user = user_movie_matrix.index[0]
    recs = hybrid_recommend_movies(sample_user, top_n=5)

    print(f"\nHybrid Recommendations for User {sample_user}:")
    for movie_id, score in recs:
        title = movies[movies['movieId'] == movie_id]['title'].values[0]
        print(f"{title} — Score: {score:.3f}")



# 12.1. TRAIN–TEST SPLIT (CRITICAL)


train, test = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)


# 12.2. BUILD TRAIN-ONLY USER–ITEM MATRIX

train_matrix = train.pivot(
    index='userId',
    columns='movieId',
    values='rating'
)

train_matrix_filled = train_matrix.fillna(0)


# 12.3. RECOMMENDATION FUNCTION (TRAIN-BASED)


def recommend_movies_from_train(user_id, top_n=10):
    if user_id not in user_index:
        return []

    user_idx = user_index[user_id]
    user_ratings = train_matrix.loc[user_id]

    unseen_movies = user_ratings[user_ratings.isna()].index

    scores = {
        movie_id: R_hat[user_idx, movie_index[movie_id]]
        for movie_id in unseen_movies
        if movie_id in movie_index
    }

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]






# 12.4 Precision@K
# We consider a movie relevant if rating >= threshold

def recommend_movies(user_id, top_n=5):
    user_idx = user_index[user_id]
    user_ratings = user_movie_matrix.loc[user_id]

    unseen_movies = user_ratings[user_ratings.isna()].index

    scores = {
        movie_id: R_hat[user_idx, movie_index[movie_id]]
        for movie_id in unseen_movies
    }

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


def precision_at_k(test_df, K=10, threshold=3.0):
    precisions = []

    for user_id in test_df['userId'].unique():

        relevant_items = set(
            test_df[
                (test_df['userId'] == user_id) &
                (test_df['rating'] >= threshold)
            ]['movieId']
        )

        if len(relevant_items) == 0:
            continue

        recommended_items = [
            movie_id for movie_id, _ in recommend_movies(user_id, top_n=K)
        ]

        hits = len(set(recommended_items) & relevant_items)
        precisions.append(hits / K)

    if len(precisions) == 0:
        return 0.0

    return sum(precisions) / len(precisions)





