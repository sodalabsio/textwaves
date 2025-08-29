from typing import Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfRepresenter:
    def __init__(self, min_df=2, max_df=0.95, ngram_range=(1, 2)):
        self.vec = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)

    def fit_transform(self, texts) -> np.ndarray:
        return self.vec.fit_transform(texts).astype('float32')

    def transform(self, texts) -> np.ndarray:
        return self.vec.transform(texts).astype('float32')
