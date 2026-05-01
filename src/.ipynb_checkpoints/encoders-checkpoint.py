import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD  # Added for dimensionality reduction
from gensim.models import Word2Vec


class BoWEncoder:
    def __init__(self, articles_df):
        self.articles_df = articles_df
        self.dens_matrix = None
        self.terms = None
        self.norm_matrix = None

    def fit(self, n_components=100):
        vectorizer = CountVectorizer(max_df=0.50, min_df=3)
        # 1. Create sparse matrix
        sparse_matrix = vectorizer.fit_transform(self.articles_df)
        self.terms = vectorizer.get_feature_names_out()

        # 2. Dimensionality Reduction (LSA) - Helps clustering significantly
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.dens_matrix = svd.fit_transform(sparse_matrix)

        # 3. Normalize for Cosine Similarity during K-Means
        self.norm_matrix = normalize(self.dens_matrix, norm='l2')


class TFEncoder:
    def __init__(self, articles_df):
        self.articles_df = articles_df
        self.dens_matrix = None
        self.terms = None
        self.norm_matrix = None

    def fit(self, n_components=100):
        vectorizer = TfidfVectorizer(max_df=0.50, min_df=3)
        # 1. Create TF-IDF matrix
        tfidf_sparse = vectorizer.fit_transform(self.articles_df)
        self.terms = vectorizer.get_feature_names_out()

        # 2. Reduce dimensions (LSA) to remove noise
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.dens_matrix = svd.fit_transform(tfidf_sparse)

        # 3. Final normalization
        self.norm_matrix = normalize(self.dens_matrix, norm='l2')


class Word2VecEncoder:
    def __init__(self, articles_df):
        self.articles_df = articles_df
        self.dens_matrix = None
        self.terms = None
        self.norm_matrix = None
        self.model = None

    def fit(self, vector_size=50, window=5, min_count=3, max_df=0.50, min_df=5, workers=4):
        # 1. Use CountVectorizer just to find the "allowed" vocabulary based on DF
        # This mirrors your BoW/TF-IDF filtering logic
        cv = CountVectorizer(max_df=max_df, min_df=min_df)
        cv.fit(self.articles_df)
        allowed_vocab = set(cv.get_feature_names_out())

        # 2. Filter the sentences: Only keep words that passed the DF filter
        # We also lowercase and split here
        filtered_sentences = []
        for doc in self.articles_df:
            words = doc.lower().split()
            filtered_doc = [w for w in words if w in allowed_vocab]
            filtered_sentences.append(filtered_doc)

        # 3. Train Word2Vec on the filtered sentences
        self.model = Word2Vec(
            sentences=filtered_sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count, # Total frequency filter
            workers=workers,
            seed=42
        )

        self.terms = list(self.model.wv.index_to_key)

        # 4. Create document vectors (Mean Pooling)
        doc_vectors = []
        for doc in filtered_sentences:
            word_vecs = [self.model.wv[word] for word in doc if word in self.model.wv]
            if word_vecs:
                doc_vec = np.mean(word_vecs, axis=0)
            else:
                doc_vec = np.zeros(vector_size)
            doc_vectors.append(doc_vec)

        self.dens_matrix = np.array(doc_vectors)
        self.norm_matrix = normalize(self.dens_matrix, norm='l2')