from src.model_training import load_model, train_model
from src.section_extraction import extract_sections
from src.feature_engineering import FeatureEngineer
from src.recommendations import detect_fresher, get_missing_skills
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Global Feature Engineer to share SBERT model and TFIDF
fe = FeatureEngineer()

def calculate_weighted_score(sections_dict, domain):
    """
    Calculates the weighted score using domain-specific ML model on sections.
    """
    # 1. Load Model
    model = load_model(domain)
    if model is None:
        # Fallback if model not found, though ideally shouldn't happen
        return {"final_score": 0, "section_scores": {}}

    is_fresher = sections_dict.get("is_fresher", False)
    
    # 2. Define Weights
    if is_fresher:
        weights = {
            "skills": 0.45,
            "projects": 0.30,
            "education": 0.15,
            "certifications": 0.10,
            "experience": 0.00
        }
    else:
        weights = {
            "skills": 0.40,
            "experience": 0.25,
            "projects": 0.20,
            "education": 0.10,
            "certifications": 0.05
        }
        
    section_scores = {}
    weighted_sum = 0
    total_weight_used = 0
    
    # 3. Score Each Section
    for section, weight in weights.items():
        if weight == 0:
            section_scores[section] = 0.0
            continue
            
        text = sections_dict.get(section, "")
        if not text.strip():
            # Empty section
            prob = 0.0
        else:
            try:
                # Predict probability for this section text
                # The model expects a list/iterable of strings
                prob = model.predict_proba([text])[0][1] # Probability of class 1 (Good Match)
            except Exception as e:
                # Fallback or error logging
                print(f"Error scoring section {section}: {e}")
                prob = 0.0
        
        section_scores[section] = round(prob, 2)
        weighted_sum += prob * weight
        total_weight_used += weight
        
    # Normalize if weights don't sum to 1 (they do in this case, but good practice)
    if total_weight_used > 0:
        final_score = (weighted_sum / total_weight_used) * 100
    else:
        final_score = 0
        
    return {
        "final_score": round(final_score, 1),
        "section_scores": section_scores
    }

def get_match_level(score):
    if score >= 81: return "Strong Match"
    if score >= 66: return "Good Match"
    if score >= 41: return "Moderate Match"
    return "Low Match"

def calculate_composite_score(resume_text, jd_text, domain):
    # 1. Load Model & Predict Global ML Probability (Overall feel)
    model = load_model(domain)
    if model is None:
        # Attempt to train if missing? Or just return 0
        return 0, "Model not trained", [], {}, False
        
    try:
        # Overall text score
        ml_proba = model.predict_proba([resume_text])[0][1] 
    except:
        ml_proba = 0

    # 2. Semantic Similarity (Full Text) using SBERT
    resume_emb = fe.get_sbert_embedding(resume_text)
    jd_emb = fe.get_sbert_embedding(jd_text)
    semantic_sim = fe.calculate_cosine_similarity(resume_emb, jd_emb) 
    
    # 3. Section Weighted Score (New Logic)
    sections = extract_sections(resume_text)
    # Note: extract_sections already returns 'is_fresher'. 
    # The dedicated detect_fresher function in recommendations might be redundant or complementary.
    # The user asked to use extract_sections output.
    weighted_result = calculate_weighted_score(sections, domain)
    section_final_score = weighted_result["final_score"] # 0-100 scale
    
    # Normalize section score to 0-1 for the composite formula
    section_score_norm = section_final_score / 100.0
    
    # 4. Final Composite Score
    # Formula: 0.4 * ML + 0.3 * Semantic + 0.3 * Section
    final_score = (0.4 * ml_proba) + (0.3 * semantic_sim) + (0.3 * section_score_norm)
    final_score = round(final_score * 100, 2)
    
    match_level = get_match_level(final_score)
    missing_skills = get_missing_skills(resume_text, jd_text)
    
    # We return 'sections' which is the raw text, maybe we want to attach valid scores to it for UI?
    # app.py expects 'sections' to be the dict of texts. 
    # We could enrich it, but app.py line 123 just does st.json(sections).
    # Let's verify if we should pass the detailed scores. 
    # The signature in app.py is: score, match_level, missing_skills, sections, is_fresher
    # 'sections' variable there is just the text dict.
    # If we want to show scores, we'd need to change app.py or pass a richer object.
    # For now, keeping contract same.
    
    return final_score, match_level, missing_skills, sections, sections.get("is_fresher", False)

if __name__ == "__main__":
    # Test
    sample_text = """
    JOHN DOE
    Fresher
    SKILLS
    Python, Java, SQL
    EDUCATION
    B.Tech Computer Science
    PROJECTS
    Resume Screening App
    """
    res = calculate_weighted_score(extract_sections(sample_text), "Data_Scientist")
    print("Test Result:", res)
