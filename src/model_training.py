import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
import pickle
import os
import random
from src.preprocessing import preprocess_text

# Note: We stick to Tfidf + RandomForest for the ML Component as requested.
# SBERT is used for the "Semantic Similarity" component in scoring.py, 
# not necessarily as input features here to avoid massive dimensionality increase 
# and training time for this phase.

def perturb_text(text, noise_level=0.3):
    """
    Randomly drops words to create a 'resume' that isn't a perfect match.
    """
    words = text.split()
    n_words = len(words)
    keep_count = int(n_words * (1 - noise_level))
    if keep_count < 1: keep_count = 1
    selected_words = random.sample(words, keep_count)
    return " ".join(selected_words)

def generate_synthetic_data(target_jd, other_jds, n_samples=50):
    data = []
    labels = []

    # Positive samples: Perturbed Target JD
    for _ in range(n_samples):
        resume_text = perturb_text(target_jd, noise_level=random.uniform(0.1, 0.4))
        data.append(resume_text)
        labels.append(1) # Good match

    # Negative samples: Perturbed Other JDs
    for _ in range(n_samples):
        other_jd = random.choice(other_jds)
        resume_text = perturb_text(other_jd, noise_level=random.uniform(0.1, 0.4))
        data.append(resume_text)
        labels.append(0) # Poor match

    return data, labels

def train_model(domain, jd_text, all_jds_dict):
    other_jds = [text for d, text in all_jds_dict.items() if d != domain]
    
    print(f"Generating synthetic data for {domain}...")
    X_text, y = generate_synthetic_data(jd_text, other_jds)
    
    # Create Pipeline with Calibration
    # CalibratedClassifierCV allows us to get better probability estimates
    base_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    calibrated_rf = CalibratedClassifierCV(base_rf, method='sigmoid')
    
    model = make_pipeline(
        TfidfVectorizer(stop_words='english', preprocessor=preprocess_text),
        calibrated_rf
    )
    
    print(f"Training calibrated model for {domain}...")
    
    # Optional: Print CV Score for verification
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X_text, y, cv=3)
    print(f"  -> CV Accuracy: {np.mean(cv_scores):.2f}")
    
    model.fit(X_text, y)
    
    # Save model
    model_path = f"models/{domain}_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return model

def load_model(domain):
    model_path = f"models/{domain}_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None
