import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

# Global cache for the model to ensure it's loaded only once
_model = None

def get_model():
    """
    Singleton pattern to load the Sentence Transformer model.
    """
    global _model
    if _model is None:
        # Using the lightweight model as requested
        _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _model

@lru_cache(maxsize=128)
def get_embedding(text):
    """
    Generates and caches embedding for a given text.
    LRU Cache ensures we don't re-compute for same resumes/JDs.
    """
    model = get_model()
    # model.encode returns a numpy array
    return model.encode([text])[0]

def calculate_semantic_similarity(resume_text, jd_text):
    """
    Computes the semantic similarity between resume and job description.
    
    Args:
        resume_text (str): Extracted text from resume.
        jd_text (str): Text of the Job Description.
        
    Returns:
        float: Similarity score between 0.0 and 1.0.
    """
    if not resume_text or not jd_text:
        return 0.0
        
    try:
        # 1. Get Embeddings (Cached)
        resume_emb = get_embedding(resume_text)
        jd_emb = get_embedding(jd_text)
        
        # 2. Reshape for sklearn (1, N)
        resume_emb = resume_emb.reshape(1, -1)
        jd_emb = jd_emb.reshape(1, -1)
        
        # 3. Compute Cosine Similarity
        # Returns [[score]]
        similarity = cosine_similarity(resume_emb, jd_emb)[0][0]
        
        # Ensure it's not negative (though usually 0-1 for SBERT)
        return float(max(0.0, similarity))
        
    except Exception as e:
        print(f"Error in semantic similarity: {e}")
        return 0.0

if __name__ == "__main__":
    # Test Example
    print("--- Semantic Similarity Test ---")
    
    resume = "I am an expert in Python, Machine Learning, and NLP."
    jd_good = "Looking for a Data Scientist with Python and NLP skills."
    jd_bad = "Looking for a HR Manager with recruitment experience."
    
    score_good = calculate_semantic_similarity(resume, jd_good)
    score_bad = calculate_semantic_similarity(resume, jd_bad)
    
    print(f"Resume: {resume}")
    print(f"JD (Target): {jd_good} -> Score: {score_good:.4f}")
    print(f"JD (Irrelevant): {jd_bad} -> Score: {score_bad:.4f}")
    
    assert score_good > score_bad, "Logic error: Irrelevant JD scored higher!"
    print("✅ Test Passed!")
