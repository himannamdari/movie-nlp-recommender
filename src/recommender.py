# src/recommender.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class MovieRecommender:
    def __init__(self, movies_csv_path: str):
        self.movies_df = pd.read_csv(movies_csv_path)
        self._prepare_data()
        self._build_model()

    def _prepare_data(self):
        self.movies_df = self.movies_df.copy().reset_index(drop=True)
        self.movies_df['title'] = self.movies_df['title'].fillna('')
        self.movies_df['genres'] = self.movies_df['genres'].fillna('')
        self.movies_df['genres_clean'] = self.movies_df['genres'].str.replace('|', ' ', regex=False)
        self.movies_df['combined'] = self.movies_df['title'] + " " + self.movies_df['genres_clean']

    def _build_model(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df['combined'])
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        self.title_to_index = pd.Series(
            self.movies_df.index,
            index=self.movies_df['title'].str.lower()
        )

    def get_recommendations(self, title: str, n: int = 10) -> pd.DataFrame | None:
        title_lower = title.lower()
        if title_lower not in self.title_to_index:
            matches = self.movies_df[self.movies_df['title'].str.lower().str.contains(title_lower)]
            print(f"Title '{title}' not found as exact match.")
            if not matches.empty:
                print("Did you mean one of these?")
                print(matches['title'].head(10).to_string(index=False))
            return None

        idx = self.title_to_index[title_lower]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]

        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        results = self.movies_df.iloc[movie_indices][['title', 'genres']].copy()
        results['similarity'] = scores
        return results

