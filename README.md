# Recommendation-System-using-Collaborative-Filtering

# CineMatch — Movie Recommendation System

CineMatch is a **personalized movie recommendation engine** built using **Collaborative Filtering** techniques on the MovieLens dataset.
The system addresses user choice overload by recommending movies based on historical user–item interactions.

## Problem Statement

With thousands of movies available, users often struggle to decide what to watch.
CineMatch aims to reduce this overload by generating **personalized movie recommendations** using past user behavior.

## Approach

The project uses **Collaborative Filtering**, implemented in two stages:

### 1. Model-Based Collaborative Filtering

* Constructed a **User–Item Interaction Matrix**
* Applied **Matrix Factorization using Truncated SVD**
* Predicted missing ratings using:

[
R \approx P \times Q^T
]

where:

* `P` = user latent factors
* `Q` = item latent factors

### 2. Hybrid Recommendation (Optional Extension)

* Combined collaborative filtering with **content-based features (movie genres)**
* Used cosine similarity on genre vectors
* Final score computed as a weighted combination of CF and content similarity

## Dataset

* **MovieLens Dataset**

  * `ratings.csv`: userId, movieId, rating, timestamp
  * `movies.csv`: movieId, title, genres

## Exploratory Data Analysis (EDA)

* Rating distribution analysis
* User activity analysis
* Matrix sparsity calculation

  * Observed sparsity ≈ **98%**, motivating matrix factorization

## Evaluation

### RMSE (Root Mean Squared Error)

* Used to evaluate rating prediction accuracy
* Computed on a train–test split of the dataset

```text
RMSE: <printed during execution>
```

### Example Recommendation Output

Sample personalized recommendations generated for selected users, displaying:

* Movie title
* Predicted rating / score

*(Precision@K was explored as an additional metric but RMSE and recommendation outputs were the primary evaluation criteria.)*

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* (Optional) Streamlit for frontend demo

## How to Run

### Google Colab / Jupyter Notebook

1. Upload `ratings.csv` and `movies.csv`
2. Run the notebook cells sequentially
3. RMSE and recommendation outputs will be printed automatically

### Local (Optional)

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python main.py
```
## Limitations

* Cold start problem for new users/items
* High sparsity in the interaction matrix
* Static preferences (no temporal dynamics)
* Popularity bias toward frequently rated movies

## Future Improvements

* Incorporate implicit feedback (watch time, clicks)
* Temporal / sequence-aware recommendation models
* Neural Collaborative Filtering
* Real-time preference adaptation

## Final Notes

This project demonstrates:

* Practical use of collaborative filtering
* Application of matrix factorization for sparse data
* End-to-end recommendation pipeline from data to output

Built as an **academic + practical case study**, not a production system.


Just say it.
