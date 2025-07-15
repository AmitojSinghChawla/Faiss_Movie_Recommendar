from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd

from config import get_data_path

csv_path = get_data_path("Raw_Data", "movies_cleaned.csv")

print(f"Resolved path: {csv_path}")  # ✅ Always works


# Load your cleaned dataset
tmdb_df = pd.read_csv(csv_path)


# # Display first few rows to verify dataset
# print("Dataset Loaded Successfully")
# print(tmdb_df.head())


# ---------------------------------------------
# 2. Combine Keywords and Genres Columns
# ---------------------------------------------

# Combine keywords and genres columns into one 'combined_features' column
tmdb_df['combined_features'] = tmdb_df['genres'] + ' ' + tmdb_df['keywords']

# Verify combined features column
print("/nCombined Features Sample")
print(tmdb_df[['id', 'combined_features']].head())


# ---------------------------------------------
# 3. Vectorize Combined Features with TF-IDF
# ---------------------------------------------

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Fit and transform the combined features to get the TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(tmdb_df['combined_features'])

# Check the shape of the resulting matrix
print("\nTF-IDF Matrix Shape:", tfidf_matrix.shape)
