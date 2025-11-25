# 🎬 AI-Powered Movie Recommendation System

An intelligent movie recommendation system that uses machine learning algorithms to suggest movies based on user preferences, viewing history, and content similarity. Available in both Python (backend/CLI) and HTML (interactive web interface) versions.

![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 📋 Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
  - [Python Version](#python-version)
  - [Web Interface](#web-interface)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## ✨ Features

### Core Functionality
- **Content-Based Filtering** - Recommends movies based on genre, director, and keywords using TF-IDF vectorization
- **Collaborative Filtering** - Suggests movies based on similar users' preferences
- **Hybrid Recommendations** - Combines multiple recommendation strategies for better accuracy
- **Smart Search** - Search movies by title, director, genre, or keywords
- **User Rating System** - Rate movies to get personalized recommendations
- **Top Rated Movies** - Browse highest-rated films with optional genre filtering

### Web Interface Features
- 🎨 Beautiful, modern UI with gradient design
- 📱 Fully responsive (works on mobile, tablet, desktop)
- 💾 Local storage for user preferences
- ⭐ Interactive 5-star rating system
- 🎯 Real-time movie recommendations
- 🔍 Advanced search and filtering
- 🎭 Genre-based discovery

## 🎥 Demo

### Python CLI Output
```
🎬 AI-POWERED MOVIE RECOMMENDATION SYSTEM 🎬
==========================================================

1️⃣ CONTENT-BASED RECOMMENDATIONS
If you liked 'Inception', you might also like:
------------------------------------------------------------
The Prestige | Mystery Thriller | Christopher Nolan | 2006 | ⭐ 8.5
Interstellar | Sci-Fi Drama | Christopher Nolan | 2014 | ⭐ 8.7
...
```

### Web Interface
![Screenshot Placeholder - Add your screenshot here]

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)
- Modern web browser (for HTML version)

### Python Version Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

### Web Interface Setup

1. **No installation required!** Simply open `index.html` in your web browser
2. Or serve it locally:
```bash
# Python 3
python -m http.server 8000

# Then visit http://localhost:8000
```

## 💻 Usage

### Python Version

Run the main script to see a demo of all features:

```bash
python movie_recommender.py
```

#### Using the API

```python
from movie_recommender import MovieRecommendationSystem

# Initialize the system
recommender = MovieRecommendationSystem()

# Get content-based recommendations
recommendations = recommender.content_based_recommendations('Inception', n=5)
print(recommendations)

# Get collaborative filtering recommendations
user_recs = recommender.collaborative_filtering(user_id=1, n=5)
print(user_recs)

# Search for movies
search_results = recommender.search_movies('Christopher Nolan')
print(search_results)

# Add a user rating
recommender.add_user_rating(user_id=1, movie_title='The Matrix', rating=5)

# Get top rated movies by genre
top_scifi = recommender.get_top_rated_movies(n=10, genre='Sci-Fi')
print(top_scifi)

# Hybrid recommendations
hybrid_recs = recommender.hybrid_recommendations(
    user_id=1, 
    movie_title='The Dark Knight', 
    n=5
)
```

### Web Interface

1. **Open `index.html`** in your browser
2. **Setup Profile** - Enter your name and select favorite genres
3. **Discover Movies** - Get personalized recommendations
4. **Find Similar** - Select a movie to find similar ones
5. **Rate Movies** - Rate movies you've watched
6. **Browse All** - Search through the entire database

## 🧠 How It Works

### Content-Based Filtering
Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization to analyze movie features:
- Genre
- Director
- Keywords/themes

Calculates **cosine similarity** between movies to find the most similar ones.

```
Movie Features → TF-IDF Vectors → Cosine Similarity → Recommendations
```

### Collaborative Filtering
Analyzes user rating patterns:
1. Creates a user-item matrix
2. Calculates similarity between users
3. Recommends movies that similar users enjoyed

```
User Ratings → Similarity Matrix → Similar Users → Recommendations
```

### Hybrid System
Combines both approaches for more accurate recommendations:
- Leverages content similarity when user data is limited
- Uses collaborative filtering when sufficient user ratings exist
- Provides diverse, personalized recommendations

## 📁 Project Structure

```
movie-recommendation-system/
│
├── movie_recommender.py       # Python backend/CLI version
├── index.html                 # Web interface (standalone)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
└── screenshots/               # Demo screenshots
   ├── web_interface.png
   └── cli_output.png
```
## 🛠️ Technologies Used

### Python Version
- **Python 3.7+**
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning algorithms
  - TfidfVectorizer - Text feature extraction
  - cosine_similarity - Similarity calculations

### Web Version
- **HTML5** - Structure
- **CSS3** - Styling with gradients and animations
- **Vanilla JavaScript** - Interactivity and logic
- **LocalStorage API** - Client-side data persistence

## 📊 Dataset

The system includes a curated dataset of 20 popular movies with:
- Title
- Genre
- Director
- Year
- Rating (IMDb-style)
- Keywords/themes

### Extending the Dataset

You can easily add more movies by editing the `movies` array in both versions:

**Python:**
```python
# In movie_recommender.py, add to self.movies DataFrame
```

**HTML:**
```javascript
// In index.html, add to the movies array
const movies = [
    { id: 21, title: 'New Movie', genre: 'Action', ... }
];
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Ideas for Contributions
- 🎬 Add more movies to the database
- 🌐 Add API integration (TMDB, OMDB)
- 🎨 Improve UI/UX design
- 📊 Add data visualization
- 🔐 Implement user authentication
- 🌍 Multi-language support
- 📱 Mobile app version
- 🎯 Improve recommendation algorithms

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

## 📧 Contact

**Your Name**
- GitHub: [@HirthikBalaji](https://github.com/HirthikBalaji)
- Email: hirthikbalaji2006@gmail.com
- LinkedIn: [Hirthik Balaji C](https://www.linkedin.com/in/hirthik-balaji-c-519b77229/)

## 🙏 Acknowledgments

- Inspired by Netflix and IMDb recommendation systems
- Movie data structure based on IMDb dataset format
- UI design inspired by modern streaming platforms
- Thanks to the scikit-learn team for excellent ML tools

## 🗺️ Roadmap

- [ ] Integration with TMDB API for real movie data
- [ ] User authentication and cloud storage
- [ ] Movie posters and trailers
- [ ] Advanced filtering (year, rating range, actors)
- [ ] Social features (share recommendations)
- [ ] Watchlist functionality
- [ ] Deep learning-based recommendations (Neural Collaborative Filtering)
- [ ] Mobile responsive improvements
- [ ] Dark/Light theme toggle
- [ ] Export recommendations as PDF

## 📸 Screenshots

### Web Interface
*Add your screenshots here*

### Python CLI
*Add your CLI screenshots here*

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ and 🎬

</div>