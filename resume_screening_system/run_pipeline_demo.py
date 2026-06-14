#!/usr/bin/env python3
"""
Pipeline Demo Script
Demonstrates end-to-end programmatic flow of the AI-Powered Resume Screening backend.
"""

import os
import sys
from pathlib import Path

# Add project src to system path if needed
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.database import init_db, create_job_posting, save_candidate_result, get_candidates
from src.pdf_extractor import extract_text_from_pdf
from src.skill_extractor import extract_resume_info, extract_skills
from src.ranker import rank_candidates
from src.feedback_generator import generate_feedback

def main():
    print("=" * 60)
    print("TalentRank AI-Powered Resume Screening Pipeline Demo")
    print("=" * 60)
    
    # 1. Initialize the SQLite database
    print("\n[1/5] Initializing database at 'data/resume_screener.db'...")
    init_db()
    
    # 2. Read and parse Job Description
    jd_file = project_root / "sample_data" / "sample_job_description.txt"
    if not jd_file.exists():
        print(f"Error: Job description file not found at {jd_file}")
        sys.exit(1)
        
    print(f"\n[2/5] Reading Job Description from '{jd_file.name}'...")
    with open(jd_file, "r") as f:
        jd_text = f.read()
        
    # Extract required skills from the job description
    jd_skills = extract_skills(jd_text)
    print(f"Parsed JD Skills: {jd_skills}")
    
    # Scoring weights
    weight_tfidf = 0.6
    weight_skill = 0.4
    
    # Create the job posting in database (weights are now stored alongside the JD)
    job_id = create_job_posting(
        title="Senior Python Backend Developer",
        description_text=jd_text,
        weight_tfidf=weight_tfidf,
        weight_skill=weight_skill,
    )
    print(f"Inserted Job Posting ID: {job_id}")
    
    # 3. Parse and extract candidates from sample resumes
    resume_dir = project_root / "sample_data" / "sample_resumes"
    if not resume_dir.exists():
        print(f"Error: Resume directory not found at {resume_dir}")
        sys.exit(1)
        
    print(f"\n[3/5] Parsing PDF resumes from '{resume_dir.name}'...")
    candidates = []
    
    for pdf_path in sorted(resume_dir.glob("*.pdf")):
        print(f" -> Parsing {pdf_path.name}...")
        try:
            raw_text = extract_text_from_pdf(pdf_path)
            candidate_info = extract_resume_info(raw_text)
            candidate_info["raw_text"] = raw_text
            candidate_info["filename"] = pdf_path.name
            candidates.append(candidate_info)
        except Exception as e:
            print(f"    Failed to parse {pdf_path.name}: {e}")
            
    if not candidates:
        print("No candidates successfully parsed. Exiting.")
        sys.exit(1)
        
    print(f"Successfully extracted {len(candidates)} candidates.")
    
    # 4. Score, rank, and generate feedback
    print(f"\n[4/5] Scoring, ranking and generating feedback for candidates...")
    
    from src.matcher import compute_match_score
    
    scored_candidates = []
    for cand in candidates:
        # Score using the matcher
        match_res = compute_match_score(
            resume_text=cand.get("raw_text", ""),
            jd_text=jd_text,
            resume_skills=cand.get("skills", []),
            jd_required_skills=jd_skills,
            weight_tfidf=weight_tfidf,
            weight_skill=weight_skill
        )
        
        # Merge scores into candidate dictionary
        scored_cand = cand.copy()
        scored_cand.update(match_res)
        
        # Generate feedback (falls back to rules if API key fails/quota exceeded)
        print(f" -> Generating feedback for {scored_cand['name']} (Score: {match_res['final_score']}%)...")
        feedback = generate_feedback(candidate_info=cand, match_result=match_res, use_llm=True)
        scored_cand["feedback"] = feedback
        scored_cand["resume_filename"] = scored_cand.get("filename", "")
        
        scored_candidates.append(scored_cand)
        
    # Rank candidates using the updated rank_candidates
    ranked_df = rank_candidates(scored_candidates)
    
    # Save the ranked candidates to the database
    for cand in ranked_df.to_dict('records'):
        save_candidate_result(job_posting_id=job_id, candidate_dict=cand)
        
    # 5. Fetch ranked results from database
    print(f"\n[5/5] Retrieving ranked candidates from database for Job ID {job_id}...")
    rows = get_candidates(job_id)
    
    print("\n" + "=" * 60)
    print("SCREENING RESULTS (Pre-sorted by Final Score Descending)")
    print("=" * 60)
    
    for rank, row in enumerate(rows, start=1):
        print(f"\nRank {rank}: {row['name']}")
        print(f"  Final Score      : {row['final_score']}%")
        print(f"  TF-IDF Score     : {row['tfidf_score']}%")
        print(f"  Skill Overlap    : {row['skill_overlap_score']}%")
        print(f"  Experience       : {row['experience']}")
        print(f"  Skills           : {row['skills']}")
        print(f"  Feedback         : {row['feedback']}")
        print("-" * 40)
        
    print(f"\nTotal candidates: {len(rows)}")
    print("Successfully finished pipeline demo.")

if __name__ == "__main__":
    main()
