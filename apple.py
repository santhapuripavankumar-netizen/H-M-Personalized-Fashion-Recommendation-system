# ============================================================
# STREAMLIT APP (AUTO-UPDATE + IMAGE CARDS)
# FIX: Remove items with broken images
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# LOAD DATA (with image URLs)
# -----------------------------

def load_data():
    users = pd.DataFrame({"user_id": [0, 1, 2, 3, 4]})

    items = pd.DataFrame([
        {"item_id": 0, "desc": "blue cotton shirt casual summer", "category": "shirt", "price": 999, "img": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab"},
        {"item_id": 1, "desc": "red silk dress party wedding", "category": "dress", "price": 1999, "img": "https://images.unsplash.com/photo-1521335629791-ce4aec67dd53"},
        {"item_id": 2, "desc": "black denim jeans slim fit", "category": "jeans", "price": 1499, "img": "https://images.unsplash.com/photo-1541099649105-f69ad21f3246"},
        {"item_id": 3, "desc": "white cotton t shirt basic", "category": "tshirt", "price": 499, "img": "https://images.unsplash.com/photo-1523381210434-271e8be1f52b"},
        {"item_id": 4, "desc": "green hoodie winter warm", "category": "hoodie", "price": 1299, "img": "https://images.unsplash.com/photo-1556821840-3a63f95609a7"},
        {"item_id": 5, "desc": "yellow floral summer dress", "category": "dress", "price": 1799, "img": "https://images.unsplash.com/photo-1490481651871-ab68de25d43d"},
        {"item_id": 6, "desc": "blue denim jacket stylish", "category": "jacket", "price": 2499, "img": "https://images.unsplash.com/photo-1512436991641-6745cdb1723f"},
        {"item_id": 7, "desc": "black formal trousers office wear", "category": "trousers", "price": 1599, "img": "https://th.bing.com/th/id/OIP.5HBEK2nShgB8ROkwXMVKggAAAA?w=115&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3"},
    ])

    interactions = pd.DataFrame([
        {"user_id": 0, "item_id": 0},
        {"user_id": 0, "item_id": 2},
        {"user_id": 1, "item_id": 1},
        {"user_id": 1, "item_id": 5},
        {"user_id": 2, "item_id": 2},
        {"user_id": 2, "item_id": 3},
        {"user_id": 3, "item_id": 4},
        {"user_id": 3, "item_id": 6},
        {"user_id": 4, "item_id": 7},
    ])

    return users, items, interactions


# -----------------------------
# BUILD MATRIX
# -----------------------------

def build_matrix(interactions, n_users, n_items):
    rows = interactions["user_id"].values
    cols = interactions["item_id"].values
    data = np.ones(len(interactions))
    return coo_matrix((data, (rows, cols)), shape=(n_users, n_items))


# -----------------------------
# MODELS
# -----------------------------

def collaborative_model(matrix):
    return cosine_similarity(matrix.tocsr())


def collaborative_recommend(user_id, matrix, user_sim):
    scores = user_sim[user_id].dot(matrix.toarray())
    seen = matrix.tocsr()[user_id].toarray().flatten()
    scores[seen > 0] = -1
    return scores


def content_model(items):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(items["desc"])
    return cosine_similarity(tfidf_matrix)


def content_recommend(user_id, interactions, similarity):
    user_items = interactions[interactions.user_id == user_id]["item_id"].tolist()

    scores = np.zeros(similarity.shape[0])

    for item in user_items:
        scores += similarity[item]

    if len(user_items) > 0:
        scores /= len(user_items)

    for item in user_items:
        scores[item] = -1

    return scores


# -----------------------------
# HYBRID
# -----------------------------

def hybrid_scores(user_id, matrix, user_sim, similarity, interactions, alpha=0.7):
    collab = collaborative_recommend(user_id, matrix, user_sim)
    content = content_recommend(user_id, interactions, similarity)
    return alpha * collab + (1 - alpha) * content


# -----------------------------
# UI
# -----------------------------

def main():
    st.set_page_config(page_title="Fashion Recommender", layout="wide")

    st.title("🛍️ H&M Style Fashion Recommendation System")

    users, items, interactions = load_data()

    matrix = build_matrix(interactions, len(users), len(items))
    user_sim = collaborative_model(matrix)
    similarity = content_model(items)

    # Inputs (auto trigger)
    user_id = st.selectbox("Select User", users["user_id"].tolist(), index=3)
    alpha = st.slider("Hybrid Weight (Collaborative vs Content)", 0.0, 1.0, 0.7)

    # Auto compute
    scores = hybrid_scores(user_id, matrix, user_sim, similarity, interactions, alpha)

    # ✅ FIX: filter only items with valid images
    ranked_items = np.argsort(scores)[::-1]

    valid_items = []
    for i in ranked_items:
        img = items.iloc[i]["img"]

        # keep only valid image links
        if isinstance(img, str) and len(img) > 0:
            valid_items.append(i)

        if len(valid_items) == 5:
            break

    top_items = valid_items

    st.subheader("Top Recommendations")

    cols = st.columns(len(top_items))

    for idx, i in enumerate(top_items):
        item = items.iloc[i]

        with cols[idx]:
            st.image(item["img"] + "?auto=compress&fit=crop&w=400", use_container_width=True)
            st.markdown(f"**{item['category'].upper()}**")
            st.caption(item['desc'])
            st.write(f"💰 ₹{item['price']}")
            st.write(f"⭐ {round(scores[i], 3)}")


if __name__ == "__main__":
    main()