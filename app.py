# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns # for data visualization
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from flask import Flask, render_template, jsonify
from scipy.sparse import csr_matrix

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('seaborn')

# Load the data into memory
user_df = pd.read_csv("users.csv")
game_df = pd.read_csv("games.csv")
recommendation_df = pd.read_csv("recommendations.csv")

# Convert date_release to datetime
game_df['date_release'] = pd.to_datetime(game_df['date_release'])
recommendation_df['date'] = pd.to_datetime(recommendation_df['date'])

# sayeed Ensure no missing values in key columns
recommendation_df = recommendation_df.dropna(subset=['user_id', 'hours', 'date', 'app_id', 'is_recommended'])

# sayeed Ensure 'date' is in datetime format and 'hours' is numeric
recommendation_df['date'] = pd.to_datetime(recommendation_df['date'], errors='coerce')
recommendation_df['hours'] = pd.to_numeric(recommendation_df['hours'], errors='coerce')


def prepare_improved_recommendation_data(user_df, game_df, recommendation_df, sample_size=1000):
    """Prepare the recommendation data with better filtering"""
    print("Sampling and preparing data...")
    
    # Sample recommendations
    recommendation_sample = recommendation_df.sample(n=sample_size, random_state=42)
    print(f"Initial sample size: {len(recommendation_sample)}")
    
    # Convert to integer and add positive/negative weight
    recommendation_sample['is_recommended'] = recommendation_sample['is_recommended'].astype(int)
    
    # Filter users and games
    user_interactions = recommendation_sample['user_id'].value_counts()
    game_interactions = recommendation_sample['app_id'].value_counts()
    
    min_user_interactions = 2
    min_game_interactions = 2
    
    valid_users = user_interactions[user_interactions >= min_user_interactions].index
    valid_games = game_interactions[game_interactions >= min_game_interactions].index
    
    filtered_recommendations = recommendation_sample[
        (recommendation_sample['user_id'].isin(valid_users)) &
        (recommendation_sample['app_id'].isin(valid_games))
    ]
    
    print(f"Filtered recommendations: {len(filtered_recommendations)}")
    print(f"Unique users: {len(valid_users)}")
    print(f"Unique games: {len(valid_games)}")
    
    # Create interaction matrix
    print("Creating interaction matrix...")
    interactions = filtered_recommendations.pivot(
        index='user_id',
        columns='app_id',
        values='is_recommended'
    ).fillna(0)
    
    return interactions.astype(np.int8)

def display_recommendations(user_id, recommender, interactions):
    """Display recommendations with improved formatting and error handling"""
    try:
        user_idx = interactions.index.get_loc(user_id)
        recommendations = recommender.get_recommendations(user_idx)
        
        if not recommendations:
            return "No recommendations found for this user."
        
        df = pd.DataFrame(recommendations)
        df = df[['title', 'rating', 'price', 'positive_ratio', 'user_reviews', 'score']]
        df['positive_ratio'] = df['positive_ratio'].apply(lambda x: f"{x:.1%}")
        df['score'] = df['score'].apply(lambda x: f"{x:.3f}")
        df['price'] = df['price'].apply(lambda x: f"${x:.2f}")
        return df
    
    except KeyError:
        return "User not found in the sample dataset"
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

class ImprovedGameRecommender:
    def __init__(self, interactions_matrix, game_df):
        self.interactions_matrix = interactions_matrix
        self.game_df = game_df
        
        # Create game mappings based on the columns in interactions_matrix
        self.game_indices = dict(enumerate(range(interactions_matrix.shape[1])))
        self.app_id_to_index = {v: k for k, v in self.game_indices.items()}
        
        # Compute game popularity (normalized)
        self.game_popularity = np.array(interactions_matrix.sum(axis=0)).flatten()
        self.game_popularity = (self.game_popularity - self.game_popularity.min()) / (
            self.game_popularity.max() - self.game_popularity.min() + 1e-10)
        
        # Use user-user similarity instead of item-item
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model.fit(interactions_matrix)
        
        print(f"Recommender initialized with {interactions_matrix.shape[0]} users and {interactions_matrix.shape[1]} games")
    
    def get_recommendations(self, user_idx, n_recommendations=5):
        user_profile = self.interactions_matrix[user_idx]
        
        if isinstance(user_profile, np.ndarray):
            user_profile = csr_matrix(user_profile)
        
        # Find similar users
        distances, indices = self.model.kneighbors(
            user_profile,
            n_neighbors=min(50, self.interactions_matrix.shape[0])
        )
        
        # Get games the user hasn't interacted with
        user_games = set(user_profile.nonzero()[1])
        all_games = set(range(self.interactions_matrix.shape[1]))
        candidate_games = list(all_games - user_games)
        
        if not candidate_games:
            return []
        
        # Calculate scores for candidate games
        scores = np.zeros(len(candidate_games))
        similar_users = indices[0][1:]  # Exclude the user themselves
        similar_distances = distances[0][1:]
        
        for i, game_idx in enumerate(candidate_games):
            # Weight recommendations by similarity
            game_ratings = self.interactions_matrix[similar_users, game_idx].toarray().flatten()
            if game_ratings.sum() > 0:
                similarity_score = np.dot(1 - similar_distances, game_ratings)
                popularity_score = self.game_popularity[game_idx]
                scores[i] = (0.7 * similarity_score) + (0.3 * popularity_score)
            else:
                scores[i] = 0.3 * self.game_popularity[game_idx]
        
        # Get top recommendations
        if scores.max() == 0:
            # If no scores, recommend popular games
            scores = self.game_popularity[candidate_games]
            
        top_game_indices = np.array(candidate_games)[np.argsort(-scores)[:n_recommendations]]
        
        recommendations = []
        for game_idx in top_game_indices:
            game_info = game_df.iloc[game_idx]
            recommendations.append({
                'app_id': game_info['app_id'],
                'title': game_info['title'],
                'rating': game_info['rating'],
                'price': game_info['price_final'],
                'positive_ratio': game_info['positive_ratio'],
                'user_reviews': game_info['user_reviews'],
                'score': float(scores[list(candidate_games).index(game_idx)])
            })
        
        return recommendations

class HybridGameRecommender(ImprovedGameRecommender):
    def __init__(self, interactions_matrix, game_df):
        super().__init__(interactions_matrix, game_df)
        
        # Initialize scalers
        self.price_scaler = MinMaxScaler()
        self.review_scaler = MinMaxScaler()
        
        # Create content features
        self.content_features = self._create_content_features()
        print(f"Content features shape: {self.content_features.shape}")
        
    def _create_content_features(self):
        # Create game features matrix
        features = pd.DataFrame(index=self.game_df.index)
        
        # Price features (normalized)
        features['price'] = self.price_scaler.fit_transform(
            self.game_df[['price_final']].fillna(0)
        ).flatten()
        
        # Rating features
        features['positive_ratio'] = self.game_df['positive_ratio'].fillna(0) / 100
        features['user_reviews'] = self.review_scaler.fit_transform(
            np.log1p(self.game_df[['user_reviews']].fillna(0))
        ).flatten()
        
        # Platform features
        features['win'] = self.game_df['win'].fillna(False).astype(float)
        features['mac'] = self.game_df['mac'].fillna(False).astype(float)
        features['linux'] = self.game_df['linux'].fillna(False).astype(float)
        
        # Rating categories (one-hot encoding)
        rating_dummies = pd.get_dummies(
            self.game_df['rating'].fillna('Unknown'), 
            prefix='rating'
        )
        features = pd.concat([features, rating_dummies], axis=1)
        
        # Steam Deck compatibility
        features['steam_deck'] = self.game_df['steam_deck'].fillna(False).astype(float)
        
        # Discount feature
        features['has_discount'] = (self.game_df['discount'] > 0).fillna(False).astype(float)
        
        # Convert all features to float
        features = features.astype(float)
        
        print("Feature columns:", features.columns.tolist())
        return features
    
    def _calculate_content_similarity(self, game_idx, user_idx):
        try:
            # Get user's played games
            user_games = self.interactions_matrix[user_idx].nonzero()[1]
            if len(user_games) == 0:
                return 0
            
            # Calculate average features of user's played games
            user_profile = self.content_features.iloc[user_games].mean().values
            game_profile = self.content_features.iloc[game_idx].values
            
            # Convert to numpy arrays and ensure they're float
            user_profile = np.array(user_profile, dtype=float)
            game_profile = np.array(game_profile, dtype=float)
            
            # Check for NaN values
            if np.any(np.isnan(user_profile)) or np.any(np.isnan(game_profile)):
                print(f"Warning: NaN values found for game_idx {game_idx}")
                return 0
            
            # Calculate cosine similarity
            if np.all(user_profile == 0) or np.all(game_profile == 0):
                return 0
                
            similarity = 1 - spatial.distance.cosine(user_profile, game_profile)
            return max(0, similarity)  # Ensure non-negative similarity
            
        except Exception as e:
            print(f"Error in _calculate_content_similarity for game_idx {game_idx}: {str(e)}")
            return 0
    
    def get_recommendations(self, user_idx, n_recommendations=5):
        try:
            # Get collaborative filtering recommendations
            cf_recommendations = super().get_recommendations(user_idx, n_recommendations * 2)
            
            if not cf_recommendations:
                return []
            
            # Calculate content similarity for recommended games
            for rec in cf_recommendations:
                game_idx = self.game_df[self.game_df['app_id'] == rec['app_id']].index[0]
                content_score = self._calculate_content_similarity(game_idx, user_idx)
                
                # Combine scores with weights
                cf_score = rec['score']
                popularity_score = min(1.0, rec['user_reviews'] / 1000)  # Cap at 1000 reviews
                
                # Final score combination
                rec['hybrid_score'] = {
                    'cf_score': cf_score,
                    'content_score': content_score,
                    'popularity_score': popularity_score
                }
                
                rec['score'] = (0.6 * cf_score + 
                              0.3 * content_score + 
                              0.1 * popularity_score)
            
            # Sort by combined score and return top N
            sorted_recs = sorted(cf_recommendations, key=lambda x: x['score'], reverse=True)
            
            # Add diversity
            diverse_recs = self._diversify_recommendations(sorted_recs, n_recommendations)
            
            return diverse_recs
            
        except Exception as e:
            print(f"Error in get_recommendations: {str(e)}")
            return []

    def _diversify_recommendations(self, recommendations, n):
        if len(recommendations) <= n:
            return recommendations
        
        diverse_recs = []
        price_ranges = set()
        ratings = set()
        
        for rec in recommendations:
            price = rec['price']
            rating = rec['rating']
            
            # Define price range
            if price < 5:
                price_range = 'budget'
            elif price < 15:
                price_range = 'mid'
            else:
                price_range = 'premium'
            
            # Add recommendation if it increases diversity
            if len(diverse_recs) < n and (
                len(diverse_recs) < n/2 or  # First half: best scores
                (price_range not in price_ranges or rating not in ratings)  # Second half: diversity
            ):
                diverse_recs.append(rec)
                price_ranges.add(price_range)
                ratings.add(rating)
                
            if len(diverse_recs) == n:
                break
                
        return diverse_recs

# Helper function to display more detailed recommendations
def display_detailed_recommendations(user_id, recommender, interactions):
    try:
        user_idx = interactions.index.get_loc(user_id)
        recommendations = recommender.get_recommendations(user_idx)
        
        if not recommendations:
            return "No recommendations found for this user."
        
        df = pd.DataFrame(recommendations)
        
        # Format columns
        if 'hybrid_score' in df.columns:
            df['cf_score'] = df['hybrid_score'].apply(lambda x: f"{x['cf_score']:.3f}")
            df['content_score'] = df['hybrid_score'].apply(lambda x: f"{x['content_score']:.3f}")
            df['popularity_score'] = df['hybrid_score'].apply(lambda x: f"{x['popularity_score']:.3f}")
            df = df.drop('hybrid_score', axis=1)
            
        df['positive_ratio'] = df['positive_ratio'].apply(lambda x: f"{x:.1%}")
        df['score'] = df['score'].apply(lambda x: f"{x:.3f}")
        df['price'] = df['price'].apply(lambda x: f"${x:.2f}")
        
        # Reorder columns
        columns = ['title', 'rating', 'price', 'positive_ratio', 'user_reviews', 'score']
        if 'cf_score' in df.columns:
            columns.extend(['cf_score', 'content_score', 'popularity_score'])
            
        return df[columns]
    
    except KeyError:
        return "User not found in the sample dataset"
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"


def get_user_data(user_id, user_df, recommendation_df, game_df):
    """
    Fetch user-specific data for the /user page.
    
    Parameters:
        user_id (int): The ID of the user.
        user_df (pd.DataFrame): DataFrame containing user information.
        recommendation_df (pd.DataFrame): DataFrame with user-game interactions.
        game_df (pd.DataFrame): DataFrame with game information.
    
    Returns:
        dict: User data including info, reviews, and games.
    """
    # Fetch user information
    user_info = user_df[user_df['user_id'] == user_id].to_dict(orient='records')
    if not user_info:
        return {"error": f"User with ID {user_id} not found."}

    user_info = user_info[0]  # Extract the single user's info

    # Fetch reviews and interactions
    user_reviews = recommendation_df[recommendation_df['user_id'] == user_id]
    
    # Join with game_df to include game details
    user_reviews = user_reviews.merge(
        game_df, 
        how='left', 
        left_on='app_id', 
        right_on='app_id'
    )
    
    # Format reviews into a list of dictionaries
    reviews_list = user_reviews[[
        'app_id', 'title', 'is_recommended', 'hours', 'helpful', 'funny'
    ]].to_dict(orient='records')

    # Construct the final output
    return {
        "user_info": {
            "user_id": user_info['user_id'],
            "total_reviews": user_info['reviews'],
            "total_products_interacted": user_info['products'],
        },
        "reviews": reviews_list
    }


def get_game_data(app_id, game_df, recommendation_df, user_df):
    """
    Fetch game-specific data for the /game page.
    
    Parameters:
        app_id (int): The ID of the game.
        game_df (pd.DataFrame): DataFrame containing game information.
        recommendation_df (pd.DataFrame): DataFrame with user-game interactions.
        user_df (pd.DataFrame): DataFrame with user information.
    
    Returns:
        dict: Game data including metadata, user interactions, and statistics.
    """
    # Fetch game metadata
    game_info = game_df[game_df['app_id'] == app_id].to_dict(orient='records')
    if not game_info:
        return {"error": f"Game with ID {app_id} not found."}
    
    game_info = game_info[0]  # Extract the single record as a dictionary

    # Fetch all interactions related to this game
    game_interactions = recommendation_df[recommendation_df['app_id'] == app_id]

    # Compute aggregated statistics
    total_reviews = len(game_interactions)
    recommended_count = game_interactions['is_recommended'].sum()
    recommendation_rate = recommended_count / total_reviews if total_reviews > 0 else 0
    avg_hours_played = game_interactions['hours'].mean() if total_reviews > 0 else 0

    # Fetch user reviews for the game
    user_reviews = game_interactions.merge(user_df, on='user_id', how='left')
    user_reviews = user_reviews[['user_id', 'hours', 'is_recommended', 'helpful', 'funny']]

    # Prepare the response
    game_data = {
        "game_info": game_info,
        "statistics": {
            "total_reviews": total_reviews,
            "recommended_count": recommended_count,
            "recommendation_rate": round(recommendation_rate, 2),
            "avg_hours_played": round(avg_hours_played, 2),
        },
        "user_reviews": user_reviews.to_dict(orient='records'),
    }
    
    return game_data

get_game_data(7000, game_df, recommendation_df, user_df)

# USERS

# sayeed 1. User's Review Distribution by Rating
def plot_user_rating_distribution(recommendation_df, user_id):
    user_reviews = recommendation_df[recommendation_df['user_id'] == user_id]
    
    # Ensure 'is_recommended' has no NaN values for this plot
    user_reviews['is_recommended'] = user_reviews['is_recommended'].fillna(0).astype(int)

    ratings = user_reviews['is_recommended'].value_counts()
    
    plt.figure(figsize=(8, 5))
    ratings.plot(kind='bar', color=['green', 'red'], alpha=0.7)
    plt.title(f"Rating Distribution for User {user_id}")
    plt.xlabel("Recommendation (1 = Recommended, 0 = Not Recommended)")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.show()


# 2. User's Playtime Distribution
def plot_user_playtime_distribution(recommendation_df, user_id):
    user_reviews = recommendation_df[recommendation_df['user_id'] == user_id]

    plt.figure(figsize=(8, 5))
    plt.hist(user_reviews['hours'], bins=10, color='blue', alpha=0.7)
    plt.title(f"Playtime Distribution for User {user_id}")
    plt.xlabel("Hours Played")
    plt.ylabel("Count")
    plt.show()

# 3. User's Review Activity Over Time
def plot_user_reviews_over_time(recommendation_df, user_id):
    user_reviews = recommendation_df[recommendation_df['user_id'] == user_id]
    user_reviews['date'] = pd.to_datetime(user_reviews['date'])
    reviews_by_date = user_reviews.groupby(user_reviews['date'].dt.date).size()

    plt.figure(figsize=(10, 6))
    reviews_by_date.plot(kind='line', marker='o')
    plt.title(f"Review Activity Over Time for User {user_id}")
    plt.xlabel("Date")
    plt.ylabel("Number of Reviews")
    plt.grid(True)
    plt.show()

# 4. Comparison of User's Average Playtime vs. Global Average
def plot_user_vs_global_playtime(recommendation_df, user_id):
    global_avg = recommendation_df['hours'].mean()
    user_avg = recommendation_df[recommendation_df['user_id'] == user_id]['hours'].mean()

    plt.figure(figsize=(6, 4))
    plt.bar(['Global Average', f'User {user_id}'], [global_avg, user_avg], color=['gray', 'blue'])
    plt.title("Average Playtime Comparison")
    plt.ylabel("Average Hours Played")
    plt.show()

# GAMES
# 1. Playtime Distribution for the Game
def plot_game_playtime_distribution(recommendation_df, app_id):
    game_reviews = recommendation_df[recommendation_df['app_id'] == app_id]

    plt.figure(figsize=(8, 5))
    plt.hist(game_reviews['hours'], bins=10, color='orange', alpha=0.7)
    plt.title(f"Playtime Distribution for Game {app_id}")
    plt.xlabel("Hours Played")
    plt.ylabel("Count")
    plt.show()

# 2. Game's Reviews Over Time
def plot_game_reviews_over_time(recommendation_df, app_id):
    game_reviews = recommendation_df[recommendation_df['app_id'] == app_id]
    game_reviews['date'] = pd.to_datetime(game_reviews['date'])
    reviews_by_date = game_reviews.groupby(game_reviews['date'].dt.date).size()

    plt.figure(figsize=(10, 6))
    reviews_by_date.plot(kind='line', marker='o', color='purple')
    plt.title(f"Review Activity Over Time for Game {app_id}")
    plt.xlabel("Date")
    plt.ylabel("Number of Reviews")
    plt.grid(True)
    plt.show()

# 3. User Ratings Distribution for the Game
def plot_game_user_rating_distribution(recommendation_df, app_id):
    game_reviews = recommendation_df[recommendation_df['app_id'] == app_id]

    plt.figure(figsize=(6, 4))
    plt.pie(
        game_reviews['is_recommended'].value_counts(),
        labels=['Recommended', 'Not Recommended'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['green', 'red']
    )
    plt.title(f"User Rating Distribution for Game {app_id}")
    plt.show()



# Initialize Flask app
app = Flask(__name__)

# Load your datasets
user_df = pd.read_csv("users.csv")
game_df = pd.read_csv("games.csv")
recommendation_df = pd.read_csv("recommendations.csv")

interactions = prepare_improved_recommendation_data(user_df, game_df, recommendation_df)
interactions_sparse = csr_matrix(interactions.values)
recommender = ImprovedGameRecommender(interactions_sparse, game_df)


# Flask Routes

app = Flask(__name__)

# Route for game-based recommendations
@app.route('/game/<int:game_id>', methods=['GET'])
def game_details(app_id):
    """
    Display detailed insights for a specific game, including:
    - Playtime distribution
    - Reviews over time
    """
    try:
        # Check if the game exists in the dataset
        if app_id not in game_df['app_id'].values:
            return jsonify({"error": f"Game with app_id {app_id} not found."}), 404

        # Generate plots for the game
        plot_game_playtime_distribution(recommendation_df, app_id)
        plot_game_reviews_over_time(recommendation_df, app_id)

        return jsonify({"message": f"Plots for Game {app_id} generated successfully. Check visual output."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Route for user-based recommendations
@app.route('/user/<int:user_id>', methods=['GET'])
def recommend_for_user(user_id):
    """
    Get recommendations for a specific user.
    """
    try:
        # Check if user_id exists
        if user_id not in user_df['user_id'].values:
            return jsonify({"error": "User ID not found"}), 404
        
        # Generate recommendations for the user
        recommendations = recommender.get_recommendations_for_user(user_id)
        
        # Optional: Use detailed display function for better output
        details = display_detailed_recommendations(user_id, recommender, recommendation_df)
        
        # Format the results
        result = {
            "user_id": user_id,
            "recommendations": recommendations,
            "details": details
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Flask app runner
if __name__ == '__main__':
    # Make sure the app runs on debug mode if needed
    app.run(debug=True)
