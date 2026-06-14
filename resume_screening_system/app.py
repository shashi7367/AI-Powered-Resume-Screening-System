import os
import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Import custom modular components
from src.database import init_db, create_job_posting, save_candidate_result, get_candidates
from src.pdf_extractor import extract_text_from_pdf, PDFExtractionError
from src.skill_extractor import extract_resume_info, extract_jd_requirements
from src.matcher import compute_match_score
from src.ranker import rank_candidates, filter_by_min_score
from src.feedback_generator import generate_feedback

# Ensure database is initialized
init_db()

# --- Page Config & Styling ---
st.set_page_config(
    page_title="TalentRank | AI Resume Screener",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sleek metric card container */
    .dashboard-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #E2E8F0;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .card-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #3182CE;
        margin-bottom: 0.25rem;
    }
    
    .card-label {
        font-size: 0.85rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Feedback card */
    .feedback-box {
        background-color: #F7FAFC;
        border-left: 4px solid #3182CE;
        padding: 1rem;
        border-radius: 4px;
        font-style: italic;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title section
st.markdown("<h1 style='text-align: center; color: #1A365D;'>TalentRank AI 🤖</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #4A5568; font-size: 1.15rem; margin-bottom: 2rem;'>Intelligent, Database-Driven Candidate Screening & Ranking</p>", unsafe_allow_html=True)

# Initialize Session State
if "screened_posting_id" not in st.session_state:
    st.session_state.screened_posting_id = None
if "screened_df" not in st.session_state:
    st.session_state.screened_df = None

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.markdown("<h2 style='color: #1A365D; margin-top: 0;'>Settings Panel</h2>", unsafe_allow_html=True)

# 1. API Configuration
st.sidebar.subheader("1. API Configuration")
api_key_input = st.sidebar.text_input(
    "OpenAI API Key (Optional)",
    value=os.getenv("OPENAI_API_KEY", ""),
    type="password",
    help="Provide an API key to enable LLM-generated summaries. Otherwise, the app falls back to rule-based feedback."
)

if api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input.strip()

# 2. Match Scoring Weights
st.sidebar.subheader("2. Scoring Weights")
w_tfidf = st.sidebar.slider("TF-IDF Description Similarity", 0.0, 1.0, 0.6, 0.05)
w_skill = st.sidebar.slider("Skill Match Overlap", 0.0, 1.0, 0.4, 0.05)

# Normalise weights just in case
w_sum = w_tfidf + w_skill
if w_sum > 0:
    w_tfidf_norm = w_tfidf / w_sum
    w_skill_norm = w_skill / w_sum
else:
    w_tfidf_norm, w_skill_norm = 0.5, 0.5

st.sidebar.caption(f"Normalized Weights: TF-IDF ({w_tfidf_norm:.2f}) | Skill Match ({w_skill_norm:.2f})")

# 3. Job Description Input
st.sidebar.subheader("3. Job Description")
jd_source = st.sidebar.radio("JD Source", ["Paste Text", "Upload TXT File"])
jd_text = ""

if jd_source == "Paste Text":
    jd_text = st.sidebar.text_area(
        "Job Description Text",
        height=250,
        placeholder="Paste requirements here..."
    )
else:
    jd_file = st.sidebar.file_uploader("Upload Job Description (.txt)", type=["txt"])
    if jd_file:
        jd_text = jd_file.read().decode("utf-8")
        st.sidebar.success("JD File uploaded successfully!")

# --- MAIN DASHBOARD CONTENT ---
st.subheader("Batch Resume Upload")
uploaded_resumes = st.file_uploader(
    "Upload Candidate Resumes (PDF format)",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or multiple resumes to screen them against the JD requirements."
)

if st.button("Run Screening Pipeline", type="primary", use_container_width=True):
    if not jd_text.strip():
        st.error("Please provide a Job Description in the Sidebar to proceed.")
    elif not uploaded_resumes:
        st.error("Please upload at least one candidate resume PDF file.")
    else:
        with st.spinner("Processing JD requirements & initializing Job Posting..."):
            # Extract JD features
            jd_reqs = extract_jd_requirements(jd_text)
            jd_skills = jd_reqs["required_skills"]
            
            # Create job posting record in the DB
            job_id = create_job_posting(
                title="Dashboard Screening Run",
                description_text=jd_text,
                weight_tfidf=w_tfidf_norm,
                weight_skill=w_skill_norm
            )
            
        progress_bar = st.progress(0)
        total_files = len(uploaded_resumes)
        
        candidates_data = []
        
        for idx, pdf_file in enumerate(uploaded_resumes):
            with st.spinner(f"Parsing and evaluating {pdf_file.name}..."):
                try:
                    # 1. Parse text from PDF stream
                    raw_text = extract_text_from_pdf(pdf_file)
                    
                    # 2. Extract NLP profile
                    cand_info = extract_resume_info(raw_text)
                    
                    # 3. Match against JD
                    match_res = compute_match_score(
                        resume_text=raw_text,
                        jd_text=jd_text,
                        resume_skills=cand_info["skills"],
                        jd_required_skills=jd_skills,
                        weight_tfidf=w_tfidf_norm,
                        weight_skill=w_skill_norm
                    )
                    
                    # 4. Generate Feedback Summary
                    use_llm = bool(api_key_input.strip())
                    feedback = generate_feedback(cand_info, match_res, use_llm=use_llm)
                    
                    # Combine all extracted and computed results
                    cand_dict = {
                        "name": cand_info["name"],
                        "email": cand_info["email"],
                        "phone": cand_info["phone"],
                        "skills": cand_info["skills"],
                        "experience": str(cand_info["experience_years"]),
                        "education": cand_info["education"],
                        "tfidf_score": match_res["tfidf_score"],
                        "skill_overlap_score": match_res["skill_overlap_score"],
                        "final_score": match_res["final_score"],
                        "feedback": feedback,
                        "resume_filename": pdf_file.name,
                        "matched_skills": match_res["matched_skills"],
                        "missing_skills": match_res["missing_skills"]
                    }
                    
                    # Save to DB
                    save_candidate_result(job_id, cand_dict)
                    
                except PDFExtractionError as pe:
                    st.error(f"Failed to parse text from {pdf_file.name}: {pe}")
                except Exception as e:
                    st.error(f"Error processing {pdf_file.name}: {e}")
                    
            progress_bar.progress((idx + 1) / total_files)
            
        st.session_state.screened_posting_id = job_id
        # Fetch the results from the DB to guarantee data consistency
        st.session_state.screened_df = pd.DataFrame(get_candidates(job_id))
        st.success("Screening process completed! Results saved to database.")

# --- RESULTS SECTION ---
if st.session_state.screened_df is not None and not st.session_state.screened_df.empty:
    df_raw = st.session_state.screened_df.copy()
    
    # Dynamically compute matched and missing skills based on the Job Posting's JD text
    from src.database import get_job_postings
    from src.skill_extractor import extract_skills
    from src.matcher import compute_skill_overlap

    job_id = st.session_state.screened_posting_id
    postings = get_job_postings()
    posting = next((p for p in postings if p["id"] == job_id), None)
    
    if posting:
        jd_skills = extract_skills(posting["description_text"])
    else:
        jd_skills = []
        
    matched_skills_col = []
    missing_skills_col = []
    for idx, row in df_raw.iterrows():
        candidate_skills = row.get("skills", [])
        _, matched, missing = compute_skill_overlap(candidate_skills, jd_skills)
        matched_skills_col.append(matched)
        missing_skills_col.append(missing)
        
    df_raw["matched_skills"] = matched_skills_col
    df_raw["missing_skills"] = missing_skills_col

    
    st.markdown("---")
    st.subheader("Screening Analysis & Analytics")
    
    # 1. Summary Metrics Cards
    m_col1, m_col2, m_col3 = st.columns(3)
    
    total_screened = len(df_raw)
    avg_score = df_raw["final_score"].mean()
    top_cand = df_raw.iloc[0]["name"] if not df_raw.empty else "N/A"
    top_score = df_raw.iloc[0]["final_score"] if not df_raw.empty else 0.0
    
    with m_col1:
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="card-value">{total_screened}</div>
            <div class="card-label">Candidates Screened</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col2:
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="card-value">{avg_score:.1f}%</div>
            <div class="card-label">Average Match Score</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col3:
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="card-value" style="color: #38A169;">{top_cand} ({top_score:.1f}%)</div>
            <div class="card-label">Top Recommended Candidate</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Main Ranked Table
    st.markdown("### Candidate Match Ranking")
    
    # Minimum Match Score slider
    min_score_filter = st.slider("Filter by Minimum Match Score (%)", 0.0, 100.0, 0.0, 1.0)
    
    # Apply score filtering using ranker module function
    filtered_df = filter_by_min_score(df_raw, min_score_filter)
    
    # Re-apply ranking numbers on the filtered view
    ranked_display_df = rank_candidates(filtered_df.to_dict("records"))
    
    if ranked_display_df.empty:
        st.warning("No candidates match the minimum score threshold.")
    else:
        # Prepare readable display columns
        display_df = ranked_display_df.copy()
        
        # Format list values as strings
        display_df["Matched Skills"] = display_df["matched_skills"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        display_df["Missing Skills"] = display_df["missing_skills"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        display_df["Education"] = display_df["education"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        
        # Rename columns to match requested grid schema
        display_df = display_df.rename(columns={
            "rank": "Rank",
            "name": "Candidate Name",
            "final_score": "Match Score (%)",
            "experience": "Experience",
        })
        
        # Select target display columns
        grid_cols = ["Rank", "Candidate Name", "Match Score (%)", "Matched Skills", "Missing Skills", "Experience", "Education"]
        st.dataframe(
            display_df[grid_cols],
            use_container_width=True,
            hide_index=True
        )
        
        # Download Results as CSV button
        csv_buffer = io.StringIO()
        ranked_display_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"screening_results_job_{st.session_state.screened_posting_id}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # 3. Bar Chart of Scores
        st.markdown("### Match Scores Visualizer")
        chart_data = ranked_display_df[["name", "final_score"]].set_index("name")
        st.bar_chart(chart_data, y="final_score")
        
        # 4. Detailed Candidate Profile Cards (Expanders)
        st.markdown("### Detailed Candidate Reports")
        for idx, row in ranked_display_df.iterrows():
            with st.expander(f"Rank {row['rank']}: {row['name']} — Score: {row['final_score']:.1f}% ({row['resume_filename']})"):
                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown(f"**Email:** {row.get('email', 'N/A')}")
                    st.markdown(f"**Phone:** {row.get('phone', 'N/A')}")
                    st.markdown(f"**Experience:** {row.get('experience', 'N/A')} years")
                    st.markdown(f"**Education:** {', '.join(row.get('education', [])) if isinstance(row.get('education'), list) else row.get('education', 'N/A')}")
                with col_right:
                    st.markdown(f"**TF-IDF Similarity Score:** {row.get('tfidf_score', 0.0):.1f}%")
                    st.markdown(f"**Skills Overlap Score:** {row.get('skill_overlap_score', 0.0):.1f}%")
                    st.markdown(f"**Matched Skills:** {', '.join(row.get('matched_skills', []))}")
                    st.markdown(f"**Missing Skills:** {', '.join(row.get('missing_skills', [])) if row.get('missing_skills') else 'None'}")
                
                st.markdown("**Recruiter Evaluation Feedback:**")
                st.markdown(f"<div class='feedback-box'>{row.get('feedback', '')}</div>", unsafe_allow_html=True)
else:
    st.info("Upload PDF resumes and click 'Run Screening Pipeline' to see the candidate match report.")
