from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os
import torch
from config import get_data_path

# Load model on GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
# ---------------------------------------------
# 1. Load the Cleaned Dataset
# ---------------------------------------------
csv_path = get_data_path("Raw_Data", "movies_cleaned.csv")
print(f"Resolved path: {csv_path}")

tmdb_df = pd.read_csv(csv_path)


# ---------------------------------------------
# 2. Combine Keywords and Genres Columns
# ---------------------------------------------

# Combine keywords and genres columns into one 'combined_features' column
tmdb_df['combined_features'] = tmdb_df['genres'] + ' ' + tmdb_df['keywords']

# Verify combined features column
print("/nCombined Features Sample")
print(tmdb_df[['id', 'combined_features']].head())


# 3. Encode with SentenceTransformer
# ---------------------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dimensional embeddings

print("\n⏳ Generating transformer embeddings...")
embeddings = model.encode(tmdb_df['combined_features'].tolist(), show_progress_bar=True)

# ---------------------------------------------
# 4. Save Embedding Matrix
# ---------------------------------------------
os.makedirs("Embeddings", exist_ok=True)
np.save("../Embeddings/transformer_embeddings.npy", embeddings)
print(f"\n✅ Embeddings saved at Embeddings/transformer_embeddings.npy")
print("Shape:", embeddings.shape)
