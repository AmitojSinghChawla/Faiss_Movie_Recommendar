import streamlit as st
from utils import get_recommendations

TMDB_BASE_POSTER_URL = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_POSTER = "https://via.placeholder.com/150x225?text=No+Image"

# === Page Setup === #
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# === Title === #
st.markdown("""
    <h1 style='text-align: center; color: #ff4b4b;'>🎥 Movie Recommendation System</h1>
    <p style='text-align: center;'>Powered by FAISS + Transformers — Find similar movies instantly!</p>
    <hr style='border: 1px solid #eee;' />
""", unsafe_allow_html=True)

# === Input === #
with st.container():
    st.markdown("#### 🔍 Search for a movie")
    movie_name = st.text_input("", placeholder="Type a movie name like *Inception*, *Titanic*, *Shrek*...")

# === If input is given === #
if movie_name:
    with st.spinner("Finding similar movies..."):
        recs, err = get_recommendations(movie_name, top_n=5)

    if err:
        st.error(err)
    else:
        # === Searched Movie === #
        st.markdown(f"### 🎯 You searched for: **{recs[0]['title']}**")
        with st.container():
            c1, c2 = st.columns([1, 2])
            with c1:
                poster_path = recs[0].get("poster_path")
                poster_url = (
                    TMDB_BASE_POSTER_URL + poster_path if isinstance(poster_path, str) and poster_path.startswith("/")
                    else poster_path if isinstance(poster_path, str) and poster_path.startswith("http")
                    else PLACEHOLDER_POSTER
                )
                st.image(poster_url, width=180)
            with c2:
                st.markdown(f"**🗣 Language:** `{recs[0]['original_language']}`")
                st.markdown(f"**📝 Overview:** {recs[0]['overview'] or '_No overview available._'}")

        st.markdown("---")
        st.markdown(f"### 🔁 Recommendations Based on **{movie_name.title()}**")

        # === Recommendations === #
        for rec in recs[1:]:
            with st.container():
                st.markdown(f"#### 🎬 {rec['title']}")
                c1, c2 = st.columns([1, 2])
                with c1:
                    poster_path = rec.get("poster_path")
                    poster_url = (
                        TMDB_BASE_POSTER_URL + poster_path if isinstance(poster_path, str) and poster_path.startswith("/")
                        else poster_path if isinstance(poster_path, str) and poster_path.startswith("http")
                        else PLACEHOLDER_POSTER
                    )
                    st.image(poster_url, width=160)
                with c2:
                    st.markdown(f"**🗣 Language:** `{rec['original_language']}`")
                    st.markdown(f"**🎯 Similarity Score:** `{rec['similarity_score']:.4f}`")
                    st.markdown(f"**📝 Overview:** {rec['overview'] or '_No overview available._'}")
            st.markdown("---")
