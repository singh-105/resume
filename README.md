# Resume Screening & Scoring System

An AI-powered resume screening system that analyzes resumes against job descriptions using Machine Learning and NLP.

## Features
- **Multi-Format Support**: extracts text from PDF, DOCX, and TXT files.
- **Domain Specific**: Scores resumes against specific job roles (Data Scientist, Business Analyst, etc.).
- **ML Scoring**: Uses TF-IDF and Random Forest (trained on synthetic data) to calculate match probability.
- **Visual Interface**: Simple and Clean UI using Streamlit.

## Setup Instructions

1. **Prerequisites**: Python 3.8+ installed.

2. **Installation**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Train Models**:
   Before running the app, you need to train the models (this generates synthetic training data).
   ```bash
   python train_models.py
   ```

4. **Run Application**:
   ```bash
   streamlit run app.py
   ```

## Project Structure
- `data/job_descriptions/`: Contains the job description text files.
- `src/`: Source code for loading, processing, and scoring.
- `models/`: Stores the trained ML models.

## How it works
1. **Extraction**: Text is extracted from the uploaded resume.
2. **Preprocessing**: Text is cleaned and lemmatized using Spacy.
3. **Training**: The system generates "synthetic" good/bad resumes based on the Job Description to train a Random Forest classifier.
4. **Scoring**: The resume is passed to the classifier to get a "Match Probability".
