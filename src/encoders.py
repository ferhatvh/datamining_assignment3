import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec

import numpy as np
from sklearn.preprocessing import normalize
import gensim.downloader as api



class BoWEncoder:
    def __init__(self, articles_df):
        self.articles_df = articles_df
        self.dens_matrix = None
        self.terms = None
        self.norm_matrix = None

    def fit(self, n_components=200, max_df=0.3):
        vectorizer = CountVectorizer(max_df=max_df, min_df=3)

        sparse_matrix = vectorizer.fit_transform(self.articles_df)
        self.terms = vectorizer.get_feature_names_out()

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.dens_matrix = svd.fit_transform(sparse_matrix)
        self.norm_matrix = normalize(self.dens_matrix, norm='l2')


class TFEncoder:
    def __init__(self, articles_df):
        self.articles_df = articles_df
        self.dens_matrix = None
        self.terms = None
        self.norm_matrix = None

    def fit(self, n_components=200, max_df=0.3):
        vectorizer = TfidfVectorizer(max_df=max_df, min_df=3)
        tfidf_sparse = vectorizer.fit_transform(self.articles_df)
        self.terms = vectorizer.get_feature_names_out()

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.dens_matrix = svd.fit_transform(tfidf_sparse)
        self.norm_matrix = normalize(self.dens_matrix, norm='l2')




class Word2VecEncoder:
    def __init__(self, articles_df):
        self.articles_df = articles_df
        self.dens_matrix = None
        self.norm_matrix = None
        self.model = None
        self.terms = None

    def fit(self, model_name="glove-wiki-gigaword-100"):
        self.model = api.load(model_name)
        vector_size = self.model.vector_size
        tokenized_docs = [
            doc.lower().split()
            for doc in self.articles_df
        ]

        self.terms = list(self.model.index_to_key)
        
        doc_vectors = []

        for doc in tokenized_docs:
            vecs = [
                self.model[word]
                for word in doc
                if word in self.model
            ]

            if len(vecs) > 0:
                doc_vec = np.mean(vecs, axis=0)
            else:
                doc_vec = np.zeros(vector_size)

            doc_vectors.append(doc_vec)

        self.dens_matrix = np.array(doc_vectors)
        self.norm_matrix = normalize(self.dens_matrix, norm='l2')