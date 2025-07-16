import faiss
import numpy as np
import pandas as pd

def simple_recommendation(movie_id, tmdb_df, transformer_embeddings, faiss_index, top_n=5):
    """
    Recommend movies based on a given movie ID using FAISS for similarity search.
    This function returns full metadata (poster, overview, language) for each result.

    Args:
    - movie_id (int): ID of the movie to base recommendations on.
    - tmdb_df (pd.DataFrame): DataFrame with movie metadata (must include 'id', 'title', etc.).
    - reduced_matrix (np.ndarray): SVD-reduced embedding matrix for all movies.
    - faiss_index (faiss.Index): Trained FAISS index for similarity search.
    - top_n (int): Number of recommendations to return.

    Returns:
    - List[Dict]: A list of recommended movies with full metadata.
    """

    try:
        # Step 1: Get the index of the movie in the DataFrame
        movie_idx = tmdb_df[tmdb_df['id'] == movie_id].index[0]
    except IndexError:
        raise ValueError(f"Movie ID {movie_id} not found in the dataset.")

    # Step 2: Get the embedding of the movie
    movie_embedding = np.array([transformer_embeddings[movie_idx]], dtype=np.float32)

    # Step 3: Query the FAISS index
    distances, indices = faiss_index.search(movie_embedding, top_n)

    # Step 4: Gather metadata for each result
    recommendations = []
    for i, idx in enumerate(indices[0]):
        movie_data = tmdb_df.iloc[idx]
        recommendations.append({
            'movie_id': int(movie_data['id']),
            'title': str(movie_data['title']),
            'poster_path': movie_data.get('poster_path', None),
            'overview': movie_data.get('overview', None),
            'original_language': movie_data.get('original_language', None),
            'similarity_score': float(distances[0][i])
        })

    return recommendations
