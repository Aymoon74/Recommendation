from flask import Flask, request, jsonify, Response
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from surprise import SVD, Dataset, Reader
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
        model (surprise.prediction_algorithms.svd.SVD): The trained SVD model.
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
        print(f"Using provided product ID '{product_id}' for content-based filtering.")
    else:
        if not user_interactions.empty:
            base_product_ids = user_interactions['product_id'].unique().tolist()
            print(f"Using user's interaction products {base_product_ids} for content-based filtering.")
        else:
            # For new users, use category-based or popularity-based fallback
            if category is not None and category in products_df['categories'].values:
                print(f"Using user-specified category '{category}' for new user.")
                interacted_categories = [category]
            else:
                # Fallback to the most popular category based on product counts
                interacted_categories = products_df['categories'].value_counts().index[:1].tolist()
                print(f"New user with no interactions. Using popular category '{interacted_categories[0]}'.")

            # Check if 'rate' column exists; otherwise, use price or random selection
            if 'rate' in products_df.columns and not products_df['rate'].isnull().all():
                popular_products = products_df[products_df['categories'] == interacted_categories[0]].sort_values(by='rate', ascending=False).head(3)
            else:
                # Fallback to top products by price or random if 'rate' is unavailable
                popular_products = products_df[products_df['categories'] == interacted_categories[0]].sort_values(by='price', ascending=False).head(3)
                if popular_products.empty:
                    popular_products = products_df[products_df['categories'] == interacted_categories[0]].sample(n=min(3, len(products_df)), random_state=42)

            base_product_ids = popular_products['product_id'].tolist()
            print(f"Selected popular products: {base_product_ids}")

    content_scores = {}
    for base_pid in base_product_ids:
        temp_recs = get_content_based_recommendations(base_pid, top_n=top_n*2)
        if not temp_recs.empty:
            for i, row in temp_recs.reset_index(drop=True).iterrows():
                pid = row['product_id']
                score = (top_n * 2 - i) * content_weight / len(base_product_ids)
                content_scores[pid] = content_scores.get(pid, 0) + score

    if content_scores:
        content_recs_df = products_df[products_df['product_id'].isin(content_scores.keys())].copy()
        content_recs_df['content_score'] = content_recs_df['product_id'].map(content_scores)

    hybrid_scores = {}
    if not collab_recs_df.empty:
        for i, row in collab_recs_df.reset_index(drop=True).iterrows():
            pid = row['product_id']
            score = (top_n * 2 - i) * adjusted_collab_weight
            hybrid_scores[pid] = hybrid_scores.get(pid, 0) + score

    if not content_recs_df.empty:
        for i, row in content_recs_df.iterrows():
            pid = row['product_id']
            score = row['content_score']
            hybrid_scores[pid] = hybrid_scores.get(pid, 0) + score

    sorted_pids = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)
    user_interacted_products = user_interactions['product_id'].unique() if not user_interactions.empty else []
    final_recommended_pids = [pid for pid in sorted_pids if pid not in user_interacted_products]

    recommended_products = products_df[products_df['product_id'].isin(final_recommended_pids)].copy()
    recommended_products['hybrid_score'] = recommended_products['product_id'].apply(lambda x: hybrid_scores.get(x, 0))
    recommended_products = recommended_products.sort_values(by='hybrid_score', ascending=False).drop(columns='hybrid_score')

    return recommended_products.head(top_n)

# Middleware to handle CORS
@app.after_request
def add_cors_headers(response):
    """Add CORS headers"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

# Flask Routes
@app.route('/')
def api_docs():
    """API Documentation"""
    docs = {
        "api": "E-Commerce Recommendation Engine",
        "endpoints": {
            "/hybrid": {
                "method": "GET",
                "description": "Hybrid recommendations (collaborative + content-based)",
                "parameters": {
                    "user_id": "string (required)",
                    "product_id": "string (optional)",
                    "category": "string (optional, for new users to specify a preferred category)",
                    "top_n": "int (default: 30)"
                },
                "example": "/hybrid?user_id=68476733e0b3981f8a711165&top_n=5&category=Clothing"
            },
            "/content": {
                "method": "GET",
                "description": "Content-based recommendations",
                "parameters": {
                    "product_id": "string (required)",
                    "top_n": "int (default: 30)"
                },
                "example": "/content?product_id=683dc98cc029c78e3af4fb0d"
            }
        }
    }
    return jsonify(docs)

@app.route('/hybrid', methods=['GET'])
def hybrid_recommendations():
    try:
        user_id = request.args.get('user_id')
        product_id = request.args.get('product_id', None)
        category = request.args.get('category', None)
        top_n = int(request.args.get('top_n', 30))
        if not user_id:
            return jsonify({"error": "Missing required parameter: user_id"}), 400
        recommendations = get_hybrid_recommendations(user_id=user_id, product_id=product_id, category=category, top_n=top_n)
        if recommendations.empty:
            return jsonify({"status": "success", "count": 0, "recommendations": []}), 200
        
        # Ensure imageURLs is a string if it's a list
        if 'imageURLs' in recommendations.columns:
            recommendations['imageURLs'] = recommendations['imageURLs'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
        
        return jsonify({
            "status": "success",
            "user_id": user_id,
            "seed_product_id": product_id,
            "category": category,
            "count": len(recommendations),
            "recommendations": recommendations.to_dict(orient='records')
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/content', methods=['GET'])
def content_recommendations():
    try:
        product_id = request.args.get('product_id')
        top_n = int(request.args.get('top_n', 30))
        if not product_id:
            return jsonify({"error": "Missing required parameter: product_id"}), 400
        recommendations = get_content_based_recommendations(product_id=product_id, top_n=top_n)
        return jsonify({
            "status": "success",
            "seed_product_id": product_id,
            "count": len(recommendations),
            "recommendations": recommendations.to_dict(orient='records')
        })
    except Exception as e:
        print(f"Error in content_recommendations: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))