import faiss
import os
import numpy as np
from config import get_data_path

# === Load reduced matrix === #
transformer_embeddings_path = get_data_path("Embeddings/transformer_embeddings.npy")
transformer_embeddings = np.load(transformer_embeddings_path).astype(np.float32)

# === Define index path === #
index_path = get_data_path("Index/faiss.index")

def create_or_update_faiss_index(transformer_embeddings, index_path):
    """
    Creates a new FAISS index or updates an existing one based on user input.
    """
    if os.path.exists(index_path):
        print(f"⚠️ FAISS index already exists at {index_path}")
        user_input = input("🔁 Do you want to update the existing index? (yes/no): ").strip().lower()

        if user_input == 'yes':
            new_index_path = input("📦 Enter new name for the updated index (e.g., Index/faiss_index_2.index): ").strip()
            print(f"🔄 Updating FAISS index and saving as {new_index_path}...")
            index = faiss.read_index(index_path)
            index.add(transformer_embeddings)
            faiss.write_index(index, new_index_path)
            print(f"✅ Updated FAISS index saved as {new_index_path}")
        else:
            print("✅ The existing FAISS index will remain unchanged.")
    else:
        print(f"📂 No FAISS index found at {index_path}. Creating a new index...")
        dim = transformer_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(transformer_embeddings)
        faiss.write_index(index, index_path)
        print(f"✅ New FAISS index created and saved at {index_path}")

def load_faiss_index(index_path):
    """
    Loads a FAISS index from disk.
    """
    try:
        index = faiss.read_index(index_path)
        print(f"✅ FAISS index loaded from {index_path}")
        return index
    except Exception as e:
        print(f"❌ Error loading FAISS index: {e}")
        return None

# === Run the index creation/update process === #
if __name__ == "__main__":
    create_or_update_faiss_index(transformer_embeddings=transformer_embeddings, index_path=index_path)
