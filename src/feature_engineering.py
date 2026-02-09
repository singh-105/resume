from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache

class FeatureEngineer:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.sbert_model = None # Lazy load
        self.fitted = False

    def load_sbert(self):
        if self.sbert_model is None:
            # Using a lightweight model for speed
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    def fit_transform_tfidf(self, resume_texts):
        """
        Fits the TF-IDF vectorizer on the resume texts and returns the matrix.
        """
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(resume_texts)
        self.fitted = True
        return tfidf_matrix

    def transform_tfidf(self, text):
        if not self.fitted:
            raise ValueError("Vectorizer not fitted.")
        return self.tfidf_vectorizer.transform([text])
    
    @lru_cache(maxsize=32)
    def get_sbert_embedding(self, text):
        """
        Returns the SBERT embedding for a given text.
        """
        self.load_sbert()
        return self.sbert_model.encode([text])[0]

    def calculate_cosine_similarity(self, vec1, vec2):
        """
        Calculates cosine similarity between two vectors.
        vec1, vec2: 1D or 2D arrays.
        """
        # Ensure 2D for sklearn cosine_similarity
        if len(vec1.shape) == 1: vec1 = vec1.reshape(1, -1)
        if len(vec2.shape) == 1: vec2 = vec2.reshape(1, -1)
        
        return cosine_similarity(vec1, vec2)[0][0]
