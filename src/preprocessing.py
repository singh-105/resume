import spacy
import re
import sys

# Try to load the model, ignore if not found (will be handled by runner or user)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

def download_spacy_model():
    from spacy.cli import download
    download("en_core_web_sm")
    global nlp
    nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """
    Basic text cleaning: lowercase, remove URLs, special characters.
    """
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-z0-9\s]', '', text) # keep only alphanumeric and space
    text = re.sub(r'\s+', ' ', text).strip() # remove extra whitespace
    return text

def preprocess_text(text):
    """
    Advanced preprocessing: lemmatization, stopword removal using Spacy.
    """
    if nlp is None:
        # Fallback or attempt download
        try:
            download_spacy_model()
        except:
            return clean_text(text) # Fallback to basic cleaning if model fails

    cleaned = clean_text(text)
    doc = nlp(cleaned)
    # Remove stopwords and punctuation, and use lemmas
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)
