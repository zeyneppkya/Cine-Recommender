"""
Akıllı Film Öneri Sistemi
=========================
Bu Flask uygulaması üç farklı öneri yaklaşımı kullanır:
1. İçerik Tabanlı Filtreleme: Film türlerinde TF-IDF ve cosine similarity kullanır
2. İşbirlikçi Filtreleme: Kullanıcı puanlama kalıplarını kullanır
3. Hibrit Sistem: Daha iyi öneriler için her iki yaklaşımı birleştirir
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
# VERİ YÜKLEME
# ============================================================================

def load_data():
    """CSV dosyalarından film ve puanlama verilerini yükle."""
    movies_df = pd.read_csv('movies.csv')
    ratings_df = pd.read_csv('ratings.csv')
    return movies_df, ratings_df

movies_df, ratings_df = load_data()

# Tür çevirileri (İngilizce -> Türkçe)
GENRE_TRANSLATIONS = {
    'Action': 'Aksiyon',
    'Adventure': 'Macera',
    'Animation': 'Animasyon',
    'Children': 'Çocuk',
    'Comedy': 'Komedi',
    'Crime': 'Suç',
    'Documentary': 'Belgesel',
    'Drama': 'Drama',
    'Fantasy': 'Fantastik',
    'Horror': 'Korku',
    'Musical': 'Müzikal',
    'Mystery': 'Gizem',
    'Romance': 'Romantik',
    'Sci-Fi': 'Bilim Kurgu',
    'Thriller': 'Gerilim',
    'War': 'Savaş',
    'Western': 'Kovboy'
}

def translate_genres(genres_str):
    """Türleri Türkçe'ye çevir."""
    genres = genres_str.split('|')
    translated = [GENRE_TRANSLATIONS.get(g, g) for g in genres]
    return ', '.join(translated)

# ============================================================================
# İÇERİK TABANLI FİLTRELEME (Content-Based Filtering)
# ============================================================================

def prepare_content_based_model(movies_df):
    """
    TF-IDF kullanarak içerik tabanlı öneri modelini hazırla.
    
    Film türlerini TF-IDF vektörlerine dönüştürür ve filmler arasındaki
    benzerliği tür kompozisyonlarına göre ölçer.
    """
    movies_df['genres_processed'] = movies_df['genres'].str.replace('|', ' ', regex=False)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres_processed'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim, tfidf.get_feature_names_out()

content_similarity, genre_features = prepare_content_based_model(movies_df)

def get_content_based_recommendations(movie_id, top_n=10):
    """
    Verilen film için içerik tabanlı öneriler al.
    
    TF-IDF cosine similarity'ye göre benzer türlerdeki filmleri döndürür.
    """
    idx = movies_df[movies_df['movieId'] == movie_id].index
    if len(idx) == 0:
        return []
    idx = idx[0]
    
    sim_scores = list(enumerate(content_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]
    
    recommendations = []
    source_movie = movies_df.iloc[idx]
    source_genres = set(source_movie['genres'].split('|'))
    source_genres_tr = translate_genres(source_movie['genres'])
    
    for movie_idx, score in sim_scores:
        movie = movies_df.iloc[movie_idx]
        target_genres = set(movie['genres'].split('|'))
        common_genres = source_genres.intersection(target_genres)
        common_genres_tr = [GENRE_TRANSLATIONS.get(g, g) for g in common_genres]
        
        if common_genres:
            explanation = f"Bu filmi {', '.join(common_genres_tr)} türündeki filmleri sevdiğiniz için öneriyoruz."
            detailed = f"Seçtiğiniz film ile {len(common_genres)} ortak tür paylaşıyor: {', '.join(common_genres_tr)}"
        else:
            explanation = "Bu film benzer bir tür profiline sahip."
            detailed = "TF-IDF analizi bu filmin içerik olarak benzer olduğunu gösteriyor."
        
        recommendations.append({
            'movieId': int(movie['movieId']),
            'title': movie['title'],
            'genres': movie['genres'],
            'score': round(float(score), 3),
            'explanation': explanation,
            'detailed_reason': detailed,
            'reason': f"Ortak türler: {', '.join(common_genres_tr)}" if common_genres else "Benzer tür profili",
            'method': 'content'
        })
    
    return recommendations

# ============================================================================
# İŞBİRLİKÇİ FİLTRELEME (Collaborative Filtering)
# ============================================================================

def prepare_collaborative_model(ratings_df, movies_df):
    """
    Film tabanlı işbirlikçi filtreleme modelini hazırla.
    
    Kullanıcı puanlama kalıplarına dayalı film-film benzerlik matrisi oluşturur.
    Kullanıcılar tarafından benzer şekilde puanlanan filmler benzer kabul edilir.
    """
    rating_matrix = ratings_df.pivot_table(
        index='userId', 
        columns='movieId', 
        values='rating'
    ).fillna(0)
    
    movie_matrix = rating_matrix.T
    
    if movie_matrix.shape[0] > 1:
        item_similarity = cosine_similarity(movie_matrix)
    else:
        item_similarity = np.array([[1.0]])
    
    return item_similarity, rating_matrix.columns.tolist()

item_similarity, rated_movie_ids = prepare_collaborative_model(ratings_df, movies_df)

def get_collaborative_recommendations(movie_id, top_n=10):
    """
    Film tabanlı işbirlikçi filtreleme önerileri al.
    
    Kullanıcılardan benzer puanlama kalıplarına sahip filmleri döndürür.
    """
    if movie_id not in rated_movie_ids:
        return []
    
    movie_idx = rated_movie_ids.index(movie_id)
    sim_scores = list(enumerate(item_similarity[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != movie_idx and s[1] > 0][:top_n]
    
    recommendations = []
    source_movie = movies_df[movies_df['movieId'] == movie_id].iloc[0]
    
    for idx, score in sim_scores:
        rec_movie_id = rated_movie_ids[idx]
        movie_data = movies_df[movies_df['movieId'] == rec_movie_id]
        
        if len(movie_data) > 0:
            movie = movie_data.iloc[0]
            
            users_source = set(ratings_df[ratings_df['movieId'] == movie_id]['userId'].tolist())
            users_target = set(ratings_df[ratings_df['movieId'] == rec_movie_id]['userId'].tolist())
            common_users = len(users_source.intersection(users_target))
            
            avg_rating = ratings_df[ratings_df['movieId'] == rec_movie_id]['rating'].mean()
            
            explanation = f"Bu filmi seven kullanıcılar, seçtiğiniz filmi de beğenmiş. {common_users} kullanıcı her iki filmi de izledi."
            detailed = f"Ortalama puan: {avg_rating:.1f}/5.0 - Benzer zevklere sahip kullanıcıların tercihi"
            
            recommendations.append({
                'movieId': int(movie['movieId']),
                'title': movie['title'],
                'genres': movie['genres'],
                'score': round(float(score), 3),
                'explanation': explanation,
                'detailed_reason': detailed,
                'reason': f"Benzer kullanıcılar tarafından beğenildi ({common_users} ortak kullanıcı)",
                'method': 'collaborative'
            })
    
    return recommendations

# ============================================================================
# HİBRİT ÖNERİ SİSTEMİ
# ============================================================================

def get_hybrid_recommendations(movie_id, top_n=10, content_weight=0.5):
    """
    İçerik tabanlı ve işbirlikçi filtrelemeyi birleştiren hibrit öneriler al.
    
    Bu yaklaşım her iki yöntemin güçlü yönlerini kullanır:
    - İçerik tabanlı: Benzer özelliklere sahip filmleri bulmak için iyi
    - İşbirlikçi: Benzer kitlelere hitap eden filmleri bulmak için iyi
    """
    collab_weight = 1 - content_weight
    
    content_recs = get_content_based_recommendations(movie_id, top_n=20)
    collab_recs = get_collaborative_recommendations(movie_id, top_n=20)
    
    combined_scores = {}
    
    for rec in content_recs:
        movie_id_rec = rec['movieId']
        combined_scores[movie_id_rec] = {
            'content_score': rec['score'],
            'collab_score': 0,
            'content_explanation': rec['explanation'],
            'collab_explanation': '',
            'title': rec['title'],
            'genres': rec['genres']
        }
    
    for rec in collab_recs:
        movie_id_rec = rec['movieId']
        if movie_id_rec in combined_scores:
            combined_scores[movie_id_rec]['collab_score'] = rec['score']
            combined_scores[movie_id_rec]['collab_explanation'] = rec['explanation']
        else:
            combined_scores[movie_id_rec] = {
                'content_score': 0,
                'collab_score': rec['score'],
                'content_explanation': '',
                'collab_explanation': rec['explanation'],
                'title': rec['title'],
                'genres': rec['genres']
            }
    
    recommendations = []
    for movie_id_rec, data in combined_scores.items():
        hybrid_score = (content_weight * data['content_score'] + 
                       collab_weight * data['collab_score'])
        
        explanations = []
        if data['content_score'] > 0 and data['collab_score'] > 0:
            explanation = "Bu film hem tür benzerliği hem de kullanıcı tercihleri açısından size uygun."
            detailed = f"İçerik benzerliği: %{data['content_score']*100:.0f} | Kullanıcı tercihi: %{data['collab_score']*100:.0f}"
        elif data['content_score'] > 0:
            movie_genres = data['genres'].split('|')
            genres_tr = [GENRE_TRANSLATIONS.get(g, g) for g in movie_genres]
            explanation = f"Bu filmi {', '.join(genres_tr[:2])} türündeki filmleri sevdiğiniz için öneriyoruz."
            detailed = f"İçerik benzerlik skoru: %{data['content_score']*100:.0f}"
        else:
            explanation = "Benzer zevklere sahip kullanıcılar bu filmi yüksek puanladı."
            detailed = f"Kullanıcı tercih skoru: %{data['collab_score']*100:.0f}"
        
        recommendations.append({
            'movieId': int(movie_id_rec),
            'title': data['title'],
            'genres': data['genres'],
            'score': round(hybrid_score, 3),
            'content_score': round(data['content_score'], 3),
            'collab_score': round(data['collab_score'], 3),
            'explanation': explanation,
            'detailed_reason': detailed,
            'reason': 'Hibrit öneri',
            'method': 'hybrid'
        })
    
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    return recommendations[:top_n]

# ============================================================================
# FLASK ROTLARI
# ============================================================================

@app.route('/')
def index():
    """Film seçimi ile ana sayfayı render et."""
    movies_list = movies_df[['movieId', 'title', 'genres']].to_dict('records')
    return render_template('index.html', movies=movies_list)

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Seçilen film için öneriler almak için API endpoint."""
    data = request.get_json()
    movie_id = int(data.get('movie_id'))
    method = data.get('method', 'hybrid')
    
    source_movie = movies_df[movies_df['movieId'] == movie_id].iloc[0]
    
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
    """Tüm filmleri almak için API endpoint."""
    movies_list = movies_df[['movieId', 'title', 'genres']].to_dict('records')
    return jsonify(movies_list)

# ============================================================================
# UYGULAMAYI ÇALIŞTIR
# ============================================================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
