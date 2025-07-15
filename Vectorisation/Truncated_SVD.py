from sklearn.decomposition import TruncatedSVD
from Tf_Idf import tfidf_matrix
import numpy as np


from config import get_data_path

index_path = get_data_path("Embeddings")

print(f"Resolved path: {index_path}")  # ✅ Always works


# ---------------------------------------------
# 4. Reduce Dimensionality Using Truncated SVD
# ---------------------------------------------

# Initialize the SVD with 100 components
svd = TruncatedSVD(n_components=100, random_state=42)

# Apply SVD to reduce the dimensionality of the TF-IDF matrix
reduced_matrix = svd.fit_transform(tfidf_matrix)

# Check the shape of the reduced matrix
print("\nReduced Matrix Shape:", reduced_matrix.shape)

# ---------------------------------------------
# 5. Save the Reduced Matrix for Future Use
# ---------------------------------------------

# Save the reduced matrix to a file for future use
np.save(f"{index_path}/reduced_matrix.npy", reduced_matrix)

# Confirm that the reduced matrix has been saved
print("\nReduced Matrix Saved Successfully!")