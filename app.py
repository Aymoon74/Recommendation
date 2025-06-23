from flask import Flask, request, jsonify, Response
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scikit_surprise import SVD, Dataset, Reader
from joblib import load
import os

app = Flask(__name__)

# Load data and models
products_df = pd.read_csv('GRAD.products.csv')
interactions_df = pd.read_csv('user_interactions.csv') if os.path.exists('user_interactions.csv') else pd.DataFrame(columns=['user_id', 'product_id', 'rating'])
svd = load('svd_model.h5') if os.path.exists('svd_model.h5') else None

# Ensure product_id and user_id are strings
products_df['product_id'] = products_df['product_id'].astype(str)
if not interactions_df.empty:
    interactions_df['product_id'] = interactions_df['product_id'].astype(str)
    interactions_df['user_id'] = interactions_df['user_id'].astype(str)

# Recreate TF-IDF and cosine similarity matrix
products_df['combined_features'] = products_df['name'].fillna('') + ' ' + products_df['brand'].fillna('') + ' ' + products_df['categories'].fillna('')
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(products_df['combined_features'])
scaler = MinMaxScaler()
price_scaled = scaler.fit_transform(products_df[['price']].fillna(products_df['price'].mean()))
content_features = np.hstack([tfidf_matrix.toarray(), price_scaled])
cosine_sim = cosine_similarity(content_features)

# Recommendation Functions
def get_content_based_recommendations(product_id, cosine_sim=cosine_sim, products_df=products_df, top_n=20):
    """
    Generates content-based recommendations for a given product.
    Args:
        product_id (str): The ID of the product to find similar items for.
        cosine_sim (np.array): The cosine similarity matrix.
        products_df (pd.DataFrame): DataFrame containing product information.
        top_n (int): The number of recommendations to return.
    Returns:
        pd.DataFrame: DataFrame containing the top recommended products.
    """
    try:
        idx = products_df[products_df['product_id'] == str(product_id)].index
        if len(idx) == 0:
            print(f"Product ID {product_id} not found in products_df.")
            return pd.DataFrame()
        idx = idx[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        product_indices = [i[0] for i in sim_scores]
        return products_df.iloc[product_indices][['product_id', 'name', 'brand', 'categories', 'price']]
    except IndexError:
        print(f"Error occurred while getting content-based recommendations for product {product_id}.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error in content-based recommendations: {e}")
        return pd.DataFrame()

def get_collaborative_recommendations(user_id, model=svd, products_df=products_df, top_n=10):
    """
    Generates collaborative filtering recommendations for a given user.
    Args:
        user_id (str): The ID of the user to recommend items for.
        model (scikit_surprise.prediction_algorithms.svd.SVD): The trained SVD model.
        products_df (pd.DataFrame): DataFrame containing product information.
        top_n (int): The number of recommendations to return.
    Returns:
        pd.DataFrame: DataFrame containing the top recommended products.
    """
    if model is None or interactions_df.empty:
        print("Collaborative Filtering not available due to lack of model or interaction data.")
        return pd.DataFrame()
    
    product_ids = products_df['product_id'].unique()
    predictions = []
    for pid in product_ids:
        pred = model.predict(str(user_id), str(pid))
        predictions.append((pid, pred.est))
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_preds = predictions[:top_n]
    return products_df[products_df['product_id'].isin([p[0] for p in top_preds])][
        ['product_id', 'name', 'brand', 'categories', 'price']
    ]

def get_hybrid_recommendations(user_id, product_id=None, top_n=50, collab_weight=0.4, interactions_df=interactions_df, category=None):
    """
    Generates hybrid recommendations combining collaborative and content-based filtering.
    Args:
        user_id (str): The ID of the user.
        product_id (str, optional): A specific product ID for content-based recommendations.
        top_n (int): The number of recommendations to return.
        collab_weight (float): Base weight for collaborative filtering (0 to 1).
        interactions_df (pd.DataFrame): DataFrame containing user interactions.
        category (str, optional): A user-specified category for new users.
    Returns:
        pd.DataFrame: DataFrame containing the top recommended products.
    """
    print(f"Generating hybrid recommendations for user: {user_id}")
    user_interactions = interactions_df[interactions_df['user_id'] == str(user_id)]
    interaction_count = len(user_interactions)
    adjusted_collab_weight = min(0.7, collab_weight + (interaction_count * 0.05))
    adjusted_collab_weight = max(0.2, adjusted_collab_weight)
    content_weight = 1 - adjusted_collab_weight

    collab_recs_df = get_collaborative_recommendations(user_id, top_n=top_n*2)
    content_recs_df = pd.DataFrame()
    base_product_ids = []

    if product_id is not None:
        base_product_ids = [product_id]
        print(f"Using provided