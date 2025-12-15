# Intelligent Movie Recommendation System

## Overview
A Flask-based web application that provides intelligent movie recommendations using three different algorithms:
- **Content-Based Filtering**: Uses TF-IDF on movie genres with cosine similarity
- **Item-Based Collaborative Filtering**: Uses user rating patterns
- **Hybrid System**: Combines both approaches for better recommendations

## Project Structure
```
├── app.py                 # Main Flask application with recommendation engines
├── movies.csv             # Movie data (movieId, title, genres)
├── ratings.csv            # User ratings data (userId, movieId, rating)
├── templates/
│   └── index.html         # Main web interface
├── static/
│   └── style.css          # Styling
└── replit.md              # This documentation file
```

## How to Run
Click the Run button or execute `python app.py`. The application will start on port 5000.

## API Endpoints
- `GET /` - Main web interface
- `POST /api/recommend` - Get recommendations for a movie
- `GET /api/movies` - Get list of all movies

## Recommendation Algorithms

### Content-Based Filtering
Uses TF-IDF (Term Frequency-Inverse Document Frequency) to vectorize movie genres and calculates cosine similarity between movies.

### Collaborative Filtering
Creates a user-item rating matrix and calculates item-item similarity based on how users rate movies.

### Hybrid System
Combines both methods with weighted averaging (default 50% each) for more robust recommendations.

## Dependencies
- Flask
- pandas
- scikit-learn
- numpy

## Recent Changes
- December 2025: Initial implementation with all three recommendation systems
