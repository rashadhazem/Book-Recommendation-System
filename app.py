import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import process

# ====================================
# 📚 Load Data
# ====================================
df = pd.read_csv('books.csv', on_bad_lines='skip')

# 🔑 Basic Preprocessing
df['title'] = df['title'].fillna('')
df['language_code'] = df['language_code'].fillna('unknown')

# ====================================
# ⚙️ Numeric Features
# ====================================
df2 = df.copy()
df2.loc[(df2['average_rating'] >= 0) & (df2['average_rating'] <= 1), 'rating_between'] = "between 0 and 1"
df2.loc[(df2['average_rating'] > 1) & (df2['average_rating'] <= 2), 'rating_between'] = "between 1 and 2"
df2.loc[(df2['average_rating'] > 2) & (df2['average_rating'] <= 3), 'rating_between'] = "between 2 and 3"
df2.loc[(df2['average_rating'] > 3) & (df2['average_rating'] <= 4), 'rating_between'] = "between 3 and 4"
df2.loc[(df2['average_rating'] > 4) & (df2['average_rating'] <= 5), 'rating_between'] = "between 4 and 5"

rating_df = pd.get_dummies(df2['rating_between'])
language_df = pd.get_dummies(df2['language_code'])
numeric_features = pd.concat([rating_df, language_df, df2['average_rating'], df2['ratings_count']], axis=1)

# Scale
scaler = MinMaxScaler()
numeric_features_scaled = scaler.fit_transform(numeric_features)

# ====================================
# ✏️ TF-IDF on Titles
# ====================================
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['title'])

# ====================================
# 🔗 Combine
# ====================================
from scipy.sparse import hstack
final_features = hstack([tfidf_matrix, numeric_features_scaled])

# ====================================
# 🔍 Nearest Neighbors
# ====================================
model = NearestNeighbors(n_neighbors=6, metric='cosine')
model.fit(final_features)

# ====================================
# 🚀 Streamlit App
# ====================================
st.title("📚 Book Recommendation System with TF-IDF")
st.write("🔍 Enter a keyword or book name:")

# ✅ User Input
book_input = st.text_input("Enter book name or keyword:")

# ✅ Session
if 'recommendations' not in st.session_state:
    st.session_state['recommendations'] = []
if 'details' not in st.session_state:
    st.session_state['details'] = None

# ✅ Recommend Button
if st.button("Recommend"):
    if book_input.strip() == '':
        st.warning("⚠️ Please enter a keyword or book name.")
    else:
        all_titles = df['title'].tolist()
        best_match, score = process.extractOne(book_input, all_titles)

        st.write(f"🔎 **Best Title Match:** {best_match} _(Fuzzy Match Score: {score})_")

        # Get vector for input
        input_vec = tfidf.transform([book_input])
        # Use average rating mean as default
        avg_rating_mean = np.mean(df['average_rating'])
        ratings_count_mean = np.mean(df['ratings_count'])
        # Dummy numeric features for query
        dummy_numeric = np.zeros((1, numeric_features_scaled.shape[1]))
        combined_input = hstack([input_vec, dummy_numeric])

        distances, indices = model.kneighbors(combined_input)

        recs = []
        for i in range(1, len(indices.flatten())):
            rec_id = indices.flatten()[i]
            dist = distances.flatten()[i]
            recs.append({
                'id': rec_id,
                'title': df.iloc[rec_id]['title'],
                'distance': dist
            })
        st.session_state['recommendations'] = recs
        st.session_state['details'] = None  # Reset

# ✅ Show Recommendations
if st.session_state['recommendations']:
    st.success("✅ Recommended books:")
    for rec in st.session_state['recommendations']:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"**{rec['title']}** _(Similarity Score: {1 - rec['distance']:.2f})_")
        with col2:
            if st.button(f"Show Details", key=f"details_{rec['id']}"):
                st.session_state['details'] = rec['id']

# ✅ Show Details
if st.session_state['details'] is not None:
    rec_id = st.session_state['details']
    book = df.iloc[rec_id]
    st.info(
        f"**📖 Title:** {book['title']}\n\n"
        f"**✍️ Author:** {book['authors']}\n\n"
        f"**⭐ Average Rating:** {book['average_rating']}\n\n"
        f"**📄 Pages:** {book['  num_pages']}\n\n"
        f"**🗳️ Ratings Count:** {book['ratings_count']}\n\n"
        f"**💬 Text Reviews Count:** {book['text_reviews_count']}\n\n"
        f"**📅 Publication Date:** {book['publication_date']}\n\n"
        f"**🌐 Language Code:** {book['language_code']}\n\n"
    )
