import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datetime import datetime


class MovieRecommendationSystem:
    def __init__(self):
        # Sample movie database with rich metadata
        self.movies = pd.DataFrame({
            'movie_id': range(1, 21),
            'title': [
                'The Shawshank Redemption', 'The Godfather', 'The Dark Knight',
                'Pulp Fiction', 'Forrest Gump', 'Inception', 'The Matrix',
                'Goodfellas', 'The Silence of the Lambs', 'Se7en',
                'The Prestige', 'Memento', 'Interstellar', 'The Departed',
                'Fight Club', 'The Green Mile', 'Gladiator', 'The Usual Suspects',
                'American Beauty', 'Parasite'
            ],
            'genre': [
                'Drama', 'Crime Drama', 'Action Thriller', 'Crime Drama', 'Drama',
                'Sci-Fi Thriller', 'Sci-Fi Action', 'Crime Drama', 'Thriller',
                'Crime Thriller', 'Mystery Thriller', 'Mystery Thriller', 'Sci-Fi Drama',
                'Crime Thriller', 'Drama', 'Drama', 'Action Drama', 'Mystery Thriller',
                'Drama', 'Thriller Drama'
            ],
            'director': [
                'Frank Darabont', 'Francis Ford Coppola', 'Christopher Nolan',
                'Quentin Tarantino', 'Robert Zemeckis', 'Christopher Nolan',
                'Wachowskis', 'Martin Scorsese', 'Jonathan Demme', 'David Fincher',
                'Christopher Nolan', 'Christopher Nolan', 'Christopher Nolan',
                'Martin Scorsese', 'David Fincher', 'Frank Darabont', 'Ridley Scott',
                'Bryan Singer', 'Sam Mendes', 'Bong Joon-ho'
            ],
            'year': [1994, 1972, 2008, 1994, 1994, 2010, 1999, 1990, 1991, 1995,
                     2006, 2000, 2014, 2006, 1999, 1999, 2000, 1995, 1999, 2019],
            'rating': [9.3, 9.2, 9.0, 8.9, 8.8, 8.8, 8.7, 8.7, 8.6, 8.6,
                       8.5, 8.4, 8.7, 8.5, 8.8, 8.6, 8.5, 8.5, 8.3, 8.5],
            'keywords': [
                'prison friendship hope redemption', 'mafia family power loyalty',
                'batman joker chaos justice hero', 'crime nonlinear dialogue coolness',
                'life journey innocence america', 'dreams reality heist mindbending',
                'simulation reality choice freedom', 'mafia crime loyalty violence',
                'serial killer FBI psychology', 'detective serial killer dark twisted',
                'magic rivalry obsession sacrifice', 'memory revenge puzzle nonlinear',
                'space time love relativity wormhole', 'undercover police betrayal boston',
                'consumerism identity anarchism twist', 'death row prison supernatural miracle',
                'rome gladiator revenge honor empire', 'crime twist mystery suspects',
                'suburban midlife crisis beauty', 'class inequality family thriller korean'
            ]
        })

        # User ratings (user_id, movie_id, rating)
        self.user_ratings = pd.DataFrame({
            'user_id': [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5],
            'movie_id': [1, 2, 3, 6, 1, 4, 5, 3, 6, 7, 11, 2, 8, 14, 3, 6, 11, 13],
            'rating': [5, 5, 4, 5, 5, 4, 5, 5, 5, 4, 5, 5, 5, 4, 4, 5, 4, 5]
        })

        # Create content-based features
        self.movies['combined_features'] = (
                self.movies['genre'] + ' ' +
                self.movies['director'] + ' ' +
                self.movies['keywords']
        )

        # TF-IDF Vectorization for content-based filtering
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies['combined_features'])

        # Compute similarity matrix
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def content_based_recommendations(self, movie_title, n=5):
        """Get recommendations based on movie content similarity"""
        try:
            idx = self.movies[self.movies['title'] == movie_title].index[0]
        except IndexError:
            return f"Movie '{movie_title}' not found in database."

        # Get similarity scores
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n + 1]  # Exclude the movie itself

        movie_indices = [i[0] for i in sim_scores]
        recommendations = self.movies.iloc[movie_indices][['title', 'genre', 'director', 'year', 'rating']]

        return recommendations

    def collaborative_filtering(self, user_id, n=5):
        """Get recommendations based on similar users' preferences"""
        if user_id not in self.user_ratings['user_id'].values:
            return "User not found. Try content-based recommendations instead."

        # Create user-item matrix
        user_item_matrix = self.user_ratings.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating'
        ).fillna(0)

        # Calculate user similarity
        user_similarity = cosine_similarity(user_item_matrix)
        user_similarity_df = pd.DataFrame(
            user_similarity,
            index=user_item_matrix.index,
            columns=user_item_matrix.index
        )

        # Find similar users
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:4]

        # Get movies rated by similar users but not by target user
        user_movies = set(self.user_ratings[self.user_ratings['user_id'] == user_id]['movie_id'])
        recommendations = []

        for sim_user in similar_users.index:
            sim_user_movies = self.user_ratings[self.user_ratings['user_id'] == sim_user]
            for _, row in sim_user_movies.iterrows():
                if row['movie_id'] not in user_movies and row['rating'] >= 4:
                    recommendations.append(row['movie_id'])

        # Get unique recommendations
        recommendations = list(set(recommendations))[:n]
        result = self.movies[self.movies['movie_id'].isin(recommendations)][
            ['title', 'genre', 'director', 'year', 'rating']]

        return result if not result.empty else "No recommendations found for this user."

    def hybrid_recommendations(self, user_id=None, movie_title=None, n=5):
        """Combine content-based and collaborative filtering"""
        recommendations = []

        if movie_title:
            content_recs = self.content_based_recommendations(movie_title, n)
            if isinstance(content_recs, pd.DataFrame):
                recommendations.append(("Content-Based", content_recs))

        if user_id:
            collab_recs = self.collaborative_filtering(user_id, n)
            if isinstance(collab_recs, pd.DataFrame):
                recommendations.append(("Collaborative Filtering", collab_recs))

        return recommendations

    def get_top_rated_movies(self, n=10, genre=None):
        """Get top rated movies, optionally filtered by genre"""
        movies = self.movies.copy()
        if genre:
            movies = movies[movies['genre'].str.contains(genre, case=False)]

        return movies.nlargest(n, 'rating')[['title', 'genre', 'director', 'year', 'rating']]

    def search_movies(self, keyword):
        """Search movies by keyword in title, genre, or director"""
        mask = (
                self.movies['title'].str.contains(keyword, case=False) |
                self.movies['genre'].str.contains(keyword, case=False) |
                self.movies['director'].str.contains(keyword, case=False) |
                self.movies['keywords'].str.contains(keyword, case=False)
        )
        return self.movies[mask][['title', 'genre', 'director', 'year', 'rating']]

    def add_user_rating(self, user_id, movie_title, rating):
        """Add a new user rating"""
        try:
            movie_id = self.movies[self.movies['title'] == movie_title]['movie_id'].values[0]
            new_rating = pd.DataFrame({
                'user_id': [user_id],
                'movie_id': [movie_id],
                'rating': [rating]
            })
            self.user_ratings = pd.concat([self.user_ratings, new_rating], ignore_index=True)
            return f"Rating added: {movie_title} - {rating}/5"
        except IndexError:
            return f"Movie '{movie_title}' not found."


# Demo usage
def main():
    print("=" * 60)
    print("🎬 AI-POWERED MOVIE RECOMMENDATION SYSTEM 🎬")
    print("=" * 60)

    recommender = MovieRecommendationSystem()

    # 1. Content-based recommendations
    print("\n1️⃣ CONTENT-BASED RECOMMENDATIONS")
    print("\nIf you liked 'Inception', you might also like:")
    print("-" * 60)
    recs = recommender.content_based_recommendations('Inception', n=5)
    print(recs.to_string(index=False))

    # 2. Collaborative filtering
    print("\n\n2️⃣ COLLABORATIVE FILTERING")
    print("\nRecommendations for User #1 based on similar users:")
    print("-" * 60)
    recs = recommender.collaborative_filtering(user_id=1, n=5)
    print(recs.to_string(index=False) if isinstance(recs, pd.DataFrame) else recs)

    # 3. Top rated movies
    print("\n\n3️⃣ TOP RATED MOVIES")
    print("-" * 60)
    top_movies = recommender.get_top_rated_movies(n=5)
    print(top_movies.to_string(index=False))

    # 4. Search functionality
    print("\n\n4️⃣ SEARCH MOVIES")
    print("\nSearching for 'Christopher Nolan' movies:")
    print("-" * 60)
    search_results = recommender.search_movies('Christopher Nolan')
    print(search_results.to_string(index=False))

    # 5. Hybrid recommendations
    print("\n\n5️⃣ HYBRID RECOMMENDATIONS")
    print("\nCombining multiple recommendation strategies:")
    print("-" * 60)
    hybrid = recommender.hybrid_recommendations(user_id=1, movie_title='The Dark Knight', n=3)
    for method, recs in hybrid:
        print(f"\n{method}:")
        print(recs.to_string(index=False))

    print("\n" + "=" * 60)
    print("System ready for personalized recommendations!")
    print("=" * 60)


if __name__ == "__main__":
    main()