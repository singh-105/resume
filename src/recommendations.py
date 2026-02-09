import re

def detect_fresher(resume_text, sections):
    """
    Returns True if the candidate is likely a refresher.
    Logic: Keywords 'student', 'fresher', or empty/short Experience section.
    """
    text = resume_text.lower()
    if 'fresher' in text or 'recent graduate' in text or 'student' in text:
        return True
    
    # Check for numeric experience indications
    exp_text = sections.get("Experience", "")
    
    # Regex for "X years", "X+ years", "X yrs"
    # If NO numeric experience pattern found in employment history, likely fresher
    if not re.search(r'\d+\+?\s*(years|yrs|year|yr)', exp_text):
        return True
        
    return False

def get_missing_skills(resume_text, jd_text):
    """
    Simple skill gap analysis. 
    Extracts potential skills from JD (capitalized words or specific list) and checks if they are in Resume.
    """
    # In a real app, use a predefined database of 1000+ tech skills.
    # Here, we use a heuristic: extract keywords from JD that look like skills.
    # For better accuracy, we can load the 'Required Skills' line from our JD files.
    
    # Let's try to extract from "Required Skills:" line in JD if possible
    skills_needed = set()
    
    # Regex to find "Required Skills: ..."
    match = re.search(r"required skills:(.*?)(\n|$)", jd_text.lower())
    if match:
        raw_skills = match.group(1)
        # Split by comma
        tokens = [s.strip() for s in raw_skills.split(',')]
        for t in tokens:
            if t: skills_needed.add(t)
    else:
        # Fallback: simple Noun chunks or just specific keywords? 
        # Let's stick to the structured JD format we created.
        pass
        
    # Check what is missing
    resume_lower = resume_text.lower()
    missing = []
    for skill in skills_needed:
        if skill not in resume_lower:
            missing.append(skill.title())
            
    return missing

def recommend_better_domains(scores, current_domain):
    """
    scores: dict {domain: score}
    current_domain: str
    Returns list of (domain, score) tuples if better matches exist.
    """
    current_score = scores.get(current_domain, 0)
    better_options = []
    
    for domain, score in scores.items():
        if domain == current_domain:
            continue
        if score > current_score: # Any improvement
            better_options.append((domain, score))
            
    # Sort by score desc
    better_options.sort(key=lambda x: x[1], reverse=True)
    return better_options[:2] # Top 2
