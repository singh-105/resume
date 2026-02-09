import re

def extract_sections(resume_text):
    """
    Extracts logical sections from resume text and detects if the candidate is a fresher.
    
    Args:
        resume_text (str): The raw text of the resume.
        
    Returns:
        dict: A dictionary containing extracted text for 'skills', 'experience', 'projects',
              'education', 'certifications', and a boolean 'is_fresher'.
    """
    
    # 1. Convert to lowercase for case-insensitive matching
    text = resume_text.lower()
    
    # 2. Define regex patterns for headings
    # Using specific variants as requested
    patterns = {
        "skills": r"\b(skills|technical skills|core competencies|expertise)\b",
        "experience": r"\b(experience|work experience|professional experience|employment history)\b",
        "projects": r"\b(projects|academic projects|personal projects)\b",
        "education": r"\b(education|academic background|qualification)\b",
        "certifications": r"\b(certifications|certificates|licenses)\b"
    }
    
    # Initialize output dictionary
    extracted_sections = {
        "skills": "",
        "experience": "",
        "projects": "",
        "education": "",
        "certifications": "",
        "is_fresher": False
    }
    
    # 3. Search for headings and their positions
    matches = []
    for section, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            matches.append((match.start(), section))
            
    # Sort matches by position to process sections in order
    matches.sort()
    
    # 4. Capture text until next heading
    for i in range(len(matches)):
        start_pos, section_name = matches[i]
        
        # Determine the end position (start of next section or end of text)
        if i + 1 < len(matches):
            end_pos = matches[i+1][0]
        else:
            end_pos = len(text)
            
        # Extract content
        content = text[start_pos:end_pos]
        
        # Remove the heading itself from the content 
        # (This is a simple cleaner: split by first newline or just regex sub slightly safer?)
        # Let's simple split by newline if present, or just remove the matched pattern string.
        # However, to be robust to the User's "Search for headings... capture text", 
        # usually we want the *body* of the section.
        # Let's remove the first line which contains the header.
        lines = content.split('\n', 1)
        if len(lines) > 1:
            section_content = lines[1].strip()
        else:
            section_content = "" # Or maybe the header was inline? Let's assume header is on its own line usually.
            # If split failed (no newline), we might want to strip the specific matched word.
            # Re-matching specifically at start might be overkill but let's try to just strip.
            # For this simple implementation, if no newline, we might return empty or full line minus header.
            # Let's keep it simple: just strip whitespace.
            pass
            
        # Append to the section (handling duplicate headers if any, though usually one main block)
        # Verify if we should overwrite or append. "Store captured text in a dictionary". 
        # Appending is safer for split sections.
        current_val = extracted_sections[section_name]
        if current_val:
            extracted_sections[section_name] = current_val + "\n" + section_content
        else:
            extracted_sections[section_name] = section_content

    # 5. Fresher Detection Logic
    # Criterion A: Resume contains keywords "fresher", "student", "recent graduate"
    fresher_keywords = r"\b(fresher|student|recent graduate)\b"
    is_fresher_by_keyword = bool(re.search(fresher_keywords, text))
    
    # Criterion B: No numeric experience detected (e.g. "1 year", "2 years", "3 yrs")
    # Regex for year duration
    experience_duration_pattern = r"\b\d+\s*(years?|yrs?)\b"
    has_experience_duration = bool(re.search(experience_duration_pattern, text))
    
    # Mark as fresher if keyword found OR no experience duration found
    if is_fresher_by_keyword or not has_experience_duration:
        extracted_sections["is_fresher"] = True
    else:
        extracted_sections["is_fresher"] = False
        
    return extracted_sections

if __name__ == "__main__":
    # Small test example
    sample_resume = """
    John Doe
    Recent Graduate
    
    Technical Skills
    Python, SQL, Machine Learning, NLP
    
    Education
    B.Tech in Computer Science, 2024
    
    Academic Projects
    Resume Screening System: Built an NLP based system.
    
    Certifications
    AWS Certified Cloud Practitioner
    """
    
    print("--- Extracting Sections from Sample Resume ---")
    result = extract_sections(sample_resume)
    
    for key, value in result.items():
        print(f"{key.upper()}: {value}")
        
    print(f"\nIS FRESHER: {result['is_fresher']}")
