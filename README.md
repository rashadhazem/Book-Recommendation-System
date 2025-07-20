

# 📚 Book Recommendation System

A smart, interactive book recommendation engine built with **Python**, **Scikit-learn**, and **Streamlit**.
Users can search for any book title — even with typos or partial names — and get accurate, similar book suggestions based on a combination of content similarity and metadata.

---

## 🚀 Features

* **📖 Fuzzy search:** Type a keyword or partial title, the system finds the closest match automatically.
* **✅ Personalized suggestions:** Returns similar books based on title, average ratings, language, and popularity.
* **📊 Data-driven:** Uses a combination of NLP, one-hot encoding, and numeric features for better results.
* **🔍 Show details:** Each recommendation comes with a **Show Details** button to view author, average rating, pages, reviews, publication date, and more.
* **💻 Interactive UI:** Built with Streamlit for an easy, responsive web experience.

---

## 📂 Dataset

* Columns used:

  * `bookID`, `title`, `authors`, `average_rating`, `isbn`, `isbn13`
  * `language_code`, `num_pages`, `ratings_count`, `text_reviews_count`
  * `publication_date`, `publisher`, `year`
* Loaded from a CSV file.
* Handles parsing issues with `on_bad_lines='skip'`.

---

## ⚙️ How It Works

1. **Data Preparation:**

   * Cleans & encodes categorical features (e.g., rating bands, language).
   * Scales numerical features (e.g., average rating, ratings count).
   * Combines them into a single **feature matrix**.
2. **Model:**

   * Uses **`NearestNeighbors`** from Scikit-learn.
   * Finds the closest books in the feature space.
3. **Search:**

   * Uses **FuzzyWuzzy** to handle typos or partial matches.
   * Suggests the top-N similar books with distances.
4. **Frontend:**

   * Streamlit app with a text input, recommend button, and dynamic results.

---

## 🗂️ Project Structure

```
.
├── books.csv               # Your book dataset
├── app.py                  # Streamlit app code
├── README.md               # This file!
```

---

## ▶️ Getting Started

**1️⃣ Clone the repo**

```bash
git clone https://github.com/yourusername/book-recommendation-system.git
cd book-recommendation-system
```

**2️⃣ Install dependencies**

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
pandas
numpy
scikit-learn
streamlit
fuzzywuzzy
python-Levenshtein
```

**3️⃣ Run the app**

```bash
streamlit run app.py
```

**4️⃣ Open your browser**
Visit 👉 [http://localhost:8501](http://localhost:8501)

---

## 🎯 Example Usage

1. Enter a book name like `Programming` or `Harry Potter`.
2. Get the closest match + similar books.
3. Click **Show Details** for extra info on each recommendation.

---

## ✨ Improvements

* ✅ Add collaborative filtering for user-based suggestions.
* ✅ Integrate quizzes and point scoring.
* ✅ Award certificates to students who complete reading challenges.
* ✅ Deploy to **Streamlit Cloud** for public access.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📧 Contact

* **Author:** rashad hazem
* **LinkedIn:** linkedin.com/in/rashad-hazem-24623b286
* **Email:** [rashadhazem777@gmail.com](mailto:ashadhazem777@gmail.com)

---

## 📜 License

This project is open-source — feel free to adapt or expand it for your own learning or portfolio!

---


