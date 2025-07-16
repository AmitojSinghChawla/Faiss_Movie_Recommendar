
# 🎬 FAISS-based Movie Recommendation System
---

A **content-based movie recommender system** powered by:
- `FAISS` for fast similarity search
- `Sentence Transformers` for semantic embeddings
- `Streamlit` for an interactive frontend

🚀 **Live App**: [Click here to try it out](https://your-deployment-link.streamlit.app) *(Coming Soon)*

---

## 📌 Features

- Input any movie title and get top **semantically similar** movies.
- Powered by **Transformer-based embeddings** for deep contextual understanding.
- FAISS handles fast approximate nearest-neighbor search on high-dimensional vectors.
- Built with modular, clean Python code and Streamlit UI.
- Automatically fetches necessary files (FAISS index, embeddings, movie metadata) from Google Drive when deployed.

---

## 🧠 Technologies Used

| Component            | Tech Stack                            |
|----------------------|----------------------------------------|
| Embeddings           | `sentence-transformers`                |
| Vector Similarity    | `FAISS (IndexFlatL2)`                  |
| Data Processing      | `Pandas`, `NumPy`                      |
| Vector Reduction     | `TruncatedSVD (optional)`              |
| UI                   | `Streamlit`                            |
| Deployment           | `Streamlit Cloud`, `Google Drive`      |

---

## 📁 Project Structure

```

Faiss\_Movie\_Recommender/
├── app/
│   ├── utils.py                # Handles downloading and recommendation logic
│   ├── recommendations.py      # Core recommendation function
├── embeddings/                 # FAISS matrix file (gitignored)
├── index/                      # FAISS index (gitignored)
├── raw\_data/                   # Cleaned metadata (gitignored)
├── config.py                   # File path helpers
├── main.py                     # Streamlit UI app
├── requirements.txt
├── .gitignore
└── README.md

````

---

## ⚙️ How It Works

1. Combine `genres + keywords` as movie features
2. Encode them using a **Transformer-based sentence model**
3. Build a FAISS index of the embeddings
4. On search, fetch the embedding of the input movie and find its top-N similar movies using FAISS
5. Show recommendations with **title, overview, language**, and **poster (if available)**

---

## 🧾 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
````

---

## 💻 Running Locally

```bash
streamlit run main.py
```

Make sure to place your data files like so (or let the app auto-download from GDrive):

```
Raw_Data/movies_cleaned.csv
Embeddings/reduced_matrix.npy
Index/faiss.index
```

---

## ☁️ Deployment

This project is deployed using [Streamlit Community Cloud](https://streamlit.io/cloud).
Since the model files are large, the app **downloads them at runtime** from Google Drive.

✅ Folders exist in Git, but large files are `.gitignore`d and downloaded only when needed.

---

## 📊 Example Result

```
🔍 Movie: Inception

1. Interstellar
   🌐 Language: en
   📖 Overview: A team of explorers travel through a wormhole...
   🔢 Similarity: 0.1721

2. The Matrix
   🌐 Language: en
   📖 Overview: A computer hacker learns about the true nature...
   🔢 Similarity: 0.1924
```

---

## 📌 TODOs

* [x] TF-IDF + SVD + FAISS version
* [x] Transformer embedding version
* [x] Streamlit frontend
* [x] GDrive integration for file loading
* [ ] Add MLflow for comparison
* [ ] Improve recommendation quality further using cast/director metadata

---

## 🧠 Author

**Amitoj Singh Chawla**
B.E. Data Science & AI
[GitHub Profile](https://github.com/AmitojSinghChawla)

---

