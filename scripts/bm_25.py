""" Modified from koreyou/bm25.py
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


class BM25(object):
    def __init__(self, b=0.75, k1=1.6):  # Lucene default: b=0.75, k1=1.2
        self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """Fit IDF to documents X"""
        self.vectorizer.fit(X)
        self.cache(X)
        self.avdl = self.last_len_X.mean()
        return self

    def cache(self, X):
        self.last_csc_X = super(TfidfVectorizer, self.vectorizer).transform(X).tocsc()
        self.last_len_X = self.last_csc_X.sum(1).A1
        return self

    def transform(self, q, X=None):
        """Calculate BM25 between query q and documents X"""
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        if X is not None:
            self.cache(X)
        X, len_X = self.last_csc_X, self.last_len_X
        (q,) = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.0
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1
