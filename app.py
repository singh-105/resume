import streamlit as st
import os
from src.data_loader import load_resume
from src.scoring import calculate_composite_score
from src.recommendations import recommend_better_domains

# Set page config
st.set_page_config(page_title="Advanced Resume Screening", page_icon="🚀", layout="wide")

# Custom CSS for "Premium" look
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #ff4b4b , #feca57, #1dd1a1);
    }
</style>
""", unsafe_allow_html=True)

def load_jds():
    jd_dir = 'data/job_descriptions'
    jds = {}
    if not os.path.exists(jd_dir): return {}
    for filename in os.listdir(jd_dir):
        if filename.endswith('.txt'):
            domain = filename.replace('.txt', '')
            with open(os.path.join(jd_dir, filename), 'r', encoding='utf-8') as f:
                jds[domain] = f.read()
    return jds

def main():
    st.title("🚀 Smart Resume Screening System")
    st.subheader("AI-Powered Matching Engine (Phase 2)")

    # Sidebar
    st.sidebar.header("Configuration")
    jds = load_jds()
    
    if not jds:
        st.error("No Job Descriptions found!")
        return

    selected_domain = st.sidebar.selectbox("Select Target Role", list(jds.keys()))
    
    # File Upload
    uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=['pdf', 'docx', 'txt'])

    if uploaded_file and st.button("Analyze Profile"):
        with st.spinner("Running AI Analysis..."):
            # Save and load
            save_folder = "data/processed_resumes"
            os.makedirs(save_folder, exist_ok=True)
            temp_path = os.path.join(save_folder, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            resume_text = load_resume(temp_path)
            
            if not resume_text:
                st.error("Text extraction failed.")
                return
                
            # --- SCORING ENGINE ---
            jd_text = jds[selected_domain]
            score, match_level, missing_skills, sections, is_fresher = calculate_composite_score(resume_text, jd_text, selected_domain)
            
            # --- RECOMMENDATIONS ENGINE (Alt Domains) ---
            # We need to calculate scores for ALL domains to find better fits
            # This might be slow if we run full pipeline, so simplistic check:
            all_scores = {selected_domain: score}
            # For simplicity in demo, maybe we don't run full SBERT on all 10 domains unless requested
            # Let's run it for top 3 other random or just all (if small number)
            for d in jds:
                if d != selected_domain:
                    # Quick check using just ML model or simplified score? 
                    # Let's run full for accuracy as requested
                    s, _, _, _, _ = calculate_composite_score(resume_text, jds[d], d)
                    all_scores[d] = s
            
            better_domains = recommend_better_domains(all_scores, selected_domain)
            
            # --- UI LAYOUT ---
            st.divider()
            
            # Top Section: Score & Level
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Candidate Type:** {'Fresher' if is_fresher else 'Experienced'}")
            with col2:
                st.metric("Selection Probability", f"{score}%", delta_color="normal")
            with col3:
                color = "red"
                if "Good" in match_level: color = "orange"
                if "Strong" in match_level: color = "green"
                st.markdown(f"**Match Level:** :{color}[{match_level}]")
                
            st.progress(score / 100)
            
            # Middle Section: Analysis
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("⚠️ Missing Key Skills")
                if missing_skills:
                    for skill in missing_skills[:10]:
                        st.write(f"- ❌ {skill}")
                else:
                    st.success("No critical skill gaps detected!")
                    
            with c2:
                st.subheader("💡 Better Fit Roles")
                if better_domains:
                    for d, s in better_domains:
                        st.info(f"**{d}**: {s}% Match")
                else:
                    st.write("This role is the best fit for you!")
                    
            # Bottom: Section Details
            with st.expander("View Section-wise Analysis"):
                st.json(sections)
                
if __name__ == "__main__":
    main()
