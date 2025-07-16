import os
import gdown
import pandas as pd
import numpy as np
import faiss

from recommendations import simple_recommendation

# === Resolve Project Root Dynamically ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def path_from_root(*parts):
    return os.path.join(PROJECT_ROOT, *parts)

# === File Paths ===
MOVIES_CSV_ID = "1Kfevxs885S4T3DN70CO41r-DbG4Mf0Pr"
transformer_embeddings_ID = "1spbHhNfJqrywokCEZKjcS4C17J7Tqz4n"
FAISS_INDEX_ID = "13c9iKRHfggY2214M2y4LtetJDRMWcDDa"

MOVIES_CSV_PATH = path_from_root("Raw_Data", "movies_cleaned.csv")
transformer_embeddings_PATH = path_from_root("Embeddings", "transformer_embeddings.npy")
FAISS_INDEX_PATH = path_from_root("Index", "faiss.index")

# === Ensure Necessary Folders Exist ===
for folder in ["Raw_Data", "Embeddings", "Index"]:
    os.makedirs(path_from_root(folder), exist_ok=True)

# === Download Helper ===
def download_from_drive(file_id, dest_path):
    if not os.path.exists(dest_path):
        print(f"⏬ Downloading {dest_path} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", dest_path, quiet=False)
        print(f"✅ Downloaded {dest_path}")
    else:
        print(f"✅ {dest_path} already exists. Skipping download.")

# === Download Files (only if missing) ===
download_from_drive(MOVIES_CSV_ID, MOVIES_CSV_PATH)
download_from_drive(transformer_embeddings_ID, transformer_embeddings_PATH)
download_from_drive(FAISS_INDEX_ID, FAISS_INDEX_PATH)

# === Load Assets ===
tmdb_df = pd.read_csv(MOVIES_CSV_PATH)
transformer_embeddings = np.load(transformer_embeddings_PATH).astype(np.float32)
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

# === Utility Functions ===
def get_movie_id_by_name(movie_name: str):
    result = tmdb_df[tmdb_df['title'].str.lower() == movie_name.strip().lower()]
    if result.empty:
        return None
    return int(result.iloc[0]['id'])

def get_recommendations(movie_name: str, top_n: int = 5):
    movie_id = get_movie_id_by_name(movie_name)
    if movie_id is None:
        return None, f"❌ Movie '{movie_name}' not found in the database."

    try:
        recommendations = simple_recommendation(
            movie_id=movie_id,
            tmdb_df=tmdb_df,
            transformer_embeddings=transformer_embeddings,
            faiss_index=faiss_index,
            top_n=top_n
        )
        return recommendations, None
    except Exception as e:
        return None, f"❌ Recommendation Error: {str(e)}"
