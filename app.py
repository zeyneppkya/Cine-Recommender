"""
Intelligent Movie Recommendation System
========================================
This Flask application implements three recommendation approaches:
1. Content-Based Filtering: Uses TF-IDF on movie genres with cosine similarity
2. Item-Based Collaborative Filtering: Uses user rating patterns
3. Hybrid System: Combines both approaches for better recommendations
"""

import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load movies and ratings data from CSV files."""
    movies_df = pd.read_csv('movies.csv')
    ratings_df = pd.read_csv('ratings.csv')
    return movies_df, ratings_df

# Load data at startup
movies_df, ratings_df = load_data()

# ============================================================================
# CONTENT-BASED FILTERING
# ============================================================================

def prepare_content_based_model(movies_df):
    """
    Prepare the content-based recommendation model using TF-IDF.
    
    This converts movie genres into TF-IDF vectors, allowing us to
    measure similarity between movies based on their genre composition.
    """
    # Replace pipe separators with spaces for TF-IDF processing
    movies_df['genres_processed'] = movies_df['genres'].str.replace('|', ' ', regex=False)
    
    # Create TF-IDF vectorizer for genre text
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres_processed'])
    
    # Compute cosine similarity matrix between all movies
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim, tfidf.get_feature_names_out()

# Prepare content-based model at startup
content_similarity, genre_features = prepare_content_based_model(movies_df)

def get_content_based_recommendations(movie_id, top_n=10):
    """
    Get content-based recommendations for a given movie.
    
    Returns movies with similar genres based on TF-IDF cosine similarity.
    """
    # Find the index of the movie
    idx = movies_df[movies_df['movieId'] == movie_id].index
    if len(idx) == 0:
        return []
    idx = idx[0]
    
    # Get similarity scores for all movies with this movie
    sim_scores = list(enumerate(content_similarity[idx]))
    
    # Sort movies by similarity score (excluding the movie itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]
    
    # Get movie details with similarity scores
    recommendations = []
    source_genres = movies_df.iloc[idx]['genres']
    
    for movie_idx, score in sim_scores:
        movie = movies_df.iloc[movie_idx]
        target_genres = movie['genres']
        
        # Find common genres for explanation
        source_set = set(source_genres.split('|'))
        target_set = set(target_genres.split('|'))
        common_genres = source_set.intersection(target_set)
        
        recommendations.append({
            'movieId': int(movie['movieId']),
            'title': movie['title'],
            'genres': movie['genres'],
            'score': round(float(score), 3),
            'reason': f"Shares genres: {', '.join(common_genres)}" if common_genres else "Similar genre profile",
            'method': 'content'
        })
    
    return recommendations

# ============================================================================
# ITEM-BASED COLLABORATIVE FILTERING
# ============================================================================

def prepare_collaborative_model(ratings_df, movies_df):
    """
    Prepare the item-based collaborative filtering model.
    
    Creates a movie-movie similarity matrix based on user rating patterns.
    Movies that are rated similarly by users are considered similar.
    """
    # Create user-item rating matrix
    rating_matrix = ratings_df.pivot_table(
        index='userId', 
        columns='movieId', 
        values='rating'
    ).fillna(0)
    
    # Transpose to get movie-movie similarity
    movie_matrix = rating_matrix.T
    
    # Calculate cosine similarity between movies based on user ratings
    if movie_matrix.shape[0] > 1:
        item_similarity = cosine_similarity(movie_matrix)
    else:
        item_similarity = np.array([[1.0]])
    
    return item_similarity, rating_matrix.columns.tolist()

# Prepare collaborative model at startup
item_similarity, rated_movie_ids = prepare_collaborative_model(ratings_df, movies_df)

def get_collaborative_recommendations(movie_id, top_n=10):
    """
    Get item-based collaborative filtering recommendations.
    
    Returns movies that have similar rating patterns from users.
    """
    if movie_id not in rated_movie_ids:
        return []
    
    # Find the index of the movie in the similarity matrix
    movie_idx = rated_movie_ids.index(movie_id)
    
    # Get similarity scores for all movies
    sim_scores = list(enumerate(item_similarity[movie_idx]))
    
    # Sort by similarity (excluding the movie itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != movie_idx and s[1] > 0][:top_n]
    
    recommendations = []
    for idx, score in sim_scores:
        rec_movie_id = rated_movie_ids[idx]
        movie_data = movies_df[movies_df['movieId'] == rec_movie_id]
        
        if len(movie_data) > 0:
            movie = movie_data.iloc[0]
            
            # Count users who rated both movies
            users_both = ratings_df[
                ratings_df['movieId'].isin([movie_id, rec_movie_id])
            ].groupby('userId').size()
            common_users = len(users_both[users_both == 2])
            
            recommendations.append({
                'movieId': int(movie['movieId']),
                'title': movie['title'],
                'genres': movie['genres'],
                'score': round(float(score), 3),
                'reason': f"Users who liked this also liked that ({common_users} users in common)",
                'method': 'collaborative'
            })
    
    return recommendations

# ============================================================================
# HYBRID RECOMMENDATION SYSTEM
# ============================================================================

def get_hybrid_recommendations(movie_id, top_n=10, content_weight=0.5):
    """
    Get hybrid recommendations combining content-based and collaborative filtering.
    
    This approach leverages the strengths of both methods:
    - Content-based: Good for finding movies with similar characteristics
    - Collaborative: Good for finding movies that appeal to similar audiences
    
    Args:
        movie_id: The source movie ID
        top_n: Number of recommendations to return
        content_weight: Weight for content-based scores (0-1)
    """
    collab_weight = 1 - content_weight
    
    # Get recommendations from both methods
    content_recs = get_content_based_recommendations(movie_id, top_n=20)
    collab_recs = get_collaborative_recommendations(movie_id, top_n=20)
    
    # Create a combined scoring dictionary
    combined_scores = {}
    
    # Add content-based scores
    for rec in content_recs:
        movie_id_rec = rec['movieId']
        combined_scores[movie_id_rec] = {
            'content_score': rec['score'],
            'collab_score': 0,
            'content_reason': rec['reason'],
            'collab_reason': '',
            'title': rec['title'],
            'genres': rec['genres']
        }
    
    # Add/update collaborative scores
    for rec in collab_recs:
        movie_id_rec = rec['movieId']
        if movie_id_rec in combined_scores:
            combined_scores[movie_id_rec]['collab_score'] = rec['score']
            combined_scores[movie_id_rec]['collab_reason'] = rec['reason']
        else:
            combined_scores[movie_id_rec] = {
                'content_score': 0,
                'collab_score': rec['score'],
                'content_reason': '',
                'collab_reason': rec['reason'],
                'title': rec['title'],
                'genres': rec['genres']
            }
    
    # Calculate hybrid scores
    recommendations = []
    for movie_id_rec, data in combined_scores.items():
        hybrid_score = (content_weight * data['content_score'] + 
                       collab_weight * data['collab_score'])
        
        # Build explanation
        reasons = []
        if data['content_score'] > 0:
            reasons.append(f"Genre similarity: {data['content_score']:.2f}")
        if data['collab_score'] > 0:
            reasons.append(f"User preference similarity: {data['collab_score']:.2f}")
        
        recommendations.append({
            'movieId': int(movie_id_rec),
            'title': data['title'],
            'genres': data['genres'],
            'score': round(hybrid_score, 3),
            'content_score': round(data['content_score'], 3),
            'collab_score': round(data['collab_score'], 3),
            'reason': ' | '.join(reasons) if reasons else 'Combined recommendation',
            'method': 'hybrid'
        })
    
    # Sort by hybrid score and return top N
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    return recommendations[:top_n]

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Render the main page with movie selection."""
    movies_list = movies_df[['movieId', 'title', 'genres']].to_dict('records')
    return render_template('index.html', movies=movies_list)

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """API endpoint to get recommendations for a selected movie."""
    data = request.get_json()
    movie_id = int(data.get('movie_id'))
    method = data.get('method', 'hybrid')
    
    # Get the source movie info
    source_movie = movies_df[movies_df['movieId'] == movie_id].iloc[0]
    
    # Get recommendations based on selected method
    if method == 'content':
        recommendations = get_content_based_recommendations(movie_id)
    elif method == 'collaborative':
        recommendations = get_collaborative_recommendations(movie_id)
    else:
        recommendations = get_hybrid_recommendations(movie_id)
    
    return jsonify({
        'source_movie': {
            'title': source_movie['title'],
            'genres': source_movie['genres']
        },
        'recommendations': recommendations,
        'method': method
    })

@app.route('/api/movies')
def get_movies():
    """API endpoint to get all movies."""
    movies_list = movies_df[['movieId', 'title', 'genres']].to_dict('records')
    return jsonify(movies_list)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
