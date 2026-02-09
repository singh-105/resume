import re

# Master list of common technical and professional skills
# In a real production system, this would be a large database or external file.
MASTER_SKILLS_DB = {
    # Programming Languages
    "python", "java", "c++", "c#", "javascript", "typescript", "scala", "r", "go", "swift", "kotlin",
    "php", "ruby", "perl", "bash", "shell", "matlab", "vb.net", 
    
    # Data Science & AI
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "pytorch", "keras",
    "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "nltk", "spacy", "hugging face",
    "data analysis", "data visualization", "big data", "hadoop", "spark", "tableau", "power bi",
    
    # Web Development
    "html", "css", "react", "angular", "vue", "node.js", "django", "flask", "fastapi", "spring boot",
    "asp.net", "laravel", "bootstrap", "tailwind", "jquery", "html5", "css3",
    
    # Database
    "sql", "mysql", "postgresql", "mongodb", "oracle", "sql server", "redis", "cassandra", "dynamodb",
    
    # Cloud & DevOps
    "aws", "azure", "google cloud", "gcp", "docker", "kubernetes", "jenkins", "git", "github", "gitlab",
    "ci/cd", "terraform", "ansible", "linux", "unix",
    
    # Soft Skills / Business
    "communication", "leadership", "project management", "agile", "scrum", "teamwork", "problem solving",
    "critical thinking", "time management", "sales", "marketing", "strategic planning"
}

def extract_skills(text):
    """
    Extracts known skills from text using simple keyword matching against the master DB.
    
    Args:
        text (str): Input text data (Resume or JD).
        
    Returns:
        set: A set of extracted skills (lowercase).
    """
    # 1. Clean and lowercase text
    text = text.lower()
    found_skills = set()
    
    # 2. Check for each skill in the text
    # This is a basic O(N*M) approach. For large DBs, use Aho-Corasick or Regex.
    for skill in MASTER_SKILLS_DB:
        # Regex boundary check to avoid partial matches (e.g., "java" in "javascript")
        # specific handling for c++ or c# to avoid regex errors
        escaped_skill = re.escape(skill)
        pattern = r'\b' + escaped_skill + r'\b'
        
        if re.search(pattern, text):
            found_skills.add(skill)
            
    return found_skills

def get_missing_skills(resume_text, jd_text):
    """
    Identifies skills present in JD but missing in Resume.
    
    Args:
        resume_text (str): Resume content.
        jd_text (str): Job Description content.
        
    Returns:
        list: List of missing skills.
    """
    # 1. Extract skills from both texts
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)
    
    # 2. Find difference
    missing = jd_skills - resume_skills
    
    return list(missing)

if __name__ == "__main__":
    # Test Example
    print("--- Skill Gap Analysis Test ---")
    
    jd = """
    Job Description: Data Scientist
    Required Skills: Python, Machine Learning, SQL, Deep Learning, AWS, Communication.
    """
    
    resume = """
    John Doe
    Python Developer
    Skills: Python, SQL, Flask, Django.
    """
    
    print(f"Resume Text:\n{resume.strip()}")
    print("-" * 20)
    print(f"JD Text:\n{jd.strip()}")
    print("-" * 20)
    
    missing = get_missing_skills(resume, jd)
    print(f"Missing Skills found: {missing}")
    
    expected_missing = {"machine learning", "deep learning", "aws", "communication"}
    
    # Check if we found at least the expected ones (set comparison needed strictly?)
    # Since our extractor uses the DB, it should find exactly what is int he text AND in DB.
    # In JD: Python, Machine Learning, SQL, Deep Learning, AWS, Communication
    # In Resume: Python, SQL
    # Expected Missing: Machine Learning, Deep Learning, AWS, Communication
    
    assert set(missing) == expected_missing, f"Test Failed! Got: {set(missing)}"
    print("✅ Test Passed!")
