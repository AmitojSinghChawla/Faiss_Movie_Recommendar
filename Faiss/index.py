import faiss
import os
import numpy as np

# Load reduced matrix
reduced_matrix = np.load("Embeddings/reduced_matrix.npy").astype(np.float32)

index_path = "Index/faiss.index"  # Path to save the FAISS index

def create_or_update_faiss_index(reduced_matrix, index_path):
    if os.path.exists(index_path):
        print(f"FAISS index already exists at {index_path}")
        user_input = input("Do you want to update the existing index? (yes/no): ").lower()

        if user_input == 'yes':
            new_index_path = input("Enter new name for the updated index (e.g., Index/faiss_index_2.index): ")
            print(f"Updating FAISS index and saving as {new_index_path}...")
            index = faiss.read_index(index_path)
            index.add(reduced_matrix)
            faiss.write_index(index, new_index_path)
            print(f"Updated FAISS index saved as {new_index_path}")
        else:
            print("The existing FAISS index will remain unchanged.")
    else:
        print(f"No FAISS index found at {index_path}. Creating a new FAISS index.")
        dim = reduced_matrix.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(reduced_matrix)
        faiss.write_index(index, index_path)
        print(f"New FAISS index created and saved at {index_path}")

def load_faiss_index(index_path):
    try:
        index = faiss.read_index(index_path)
        print(f"FAISS index loaded from {index_path}")
        return index
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None
