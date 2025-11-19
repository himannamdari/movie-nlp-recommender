# Movie NLP Recommender (TF-IDF + Cosine Similarity)

This project is a simple **content-based movie recommendation system** built using
classic NLP techniques. Given a movie title, it recommends similar movies based
on their **title and genres**.

The goal of this repo is to demonstrate a clean, end-to-end NLP workflow that
can be easily understood, extended, and used as a portfolio project.

---

## How It Works

1. **Data**  
   Uses the open-source [MovieLens](https://grouplens.org/datasets/movielens/)
   dataset (`movies.csv` from the `ml-latest-small` release).

2. **Text Features**  
   For each movie, we create a simple text field by combining:
   - movie title
   - genres (e.g., "Animation Children Comedy")

3. **Vectorization (NLP)**  
   We use `TfidfVectorizer` from `scikit-learn` to convert the text into
   numerical vectors (TF-IDF). Each movie becomes a vector in a high-dimensional
   space where important words (for that movie) get higher weights.

4. **Similarity**  See: https://medium.com/@arjunprakash027/understanding-cosine-similarity-a-key-concept-in-data-science-72a0fcc57599
   We compute **cosine similarity** between all movie vectors. Two movies are
   considered "similar" if their TF-IDF vectors point in a similar direction.

5. **Recommendation**  
   Given a movie title, we:
   - find its index in the dataset  
   - look up its similarity scores with all other movies  
   - sort by similarity and return the top *N* most similar titles

---

## Project Structure

```text
movie-nlp-recommender/
├── notebooks/
│   └── movie_nlp_recommender.ipynb   # Colab / Jupyter notebook
├── src/
│   └── recommender.py                # (optional) Python class for reuse
├── data/
│   └── movies.csv                    # not tracked in git (see below)
├── requirements.txt
└── README.md
