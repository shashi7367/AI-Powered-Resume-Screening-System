"""
matcher.py — Core matching engine utilizing TF-IDF and Skill Overlap.

Exposes:
  • compute_match_score(...) - returns a composite match report between resume and JD.
"""

import re
import logging
from typing import List, Dict, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

def clean_for_tfidf(text: str) -> str:
    """
    Cleans text specifically for TF-IDF calculations by lowercasing,
    removing punctuation, and normalizing whitespace.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_tfidf_similarity(resume_text: str, jd_text: str) -> float:
    """
    Computes the cosine similarity between the resume text and job description
    using TF-IDF representation. Handles empty inputs and empty vocabulary cases.
    
    Returns:
        float: Similarity percentage (0.0 to 100.0)
    """
    cleaned_resume = clean_for_tfidf(resume_text)
    cleaned_jd = clean_for_tfidf(jd_text)
    
    # Check for empty text or very little content
    if not cleaned_resume or not cleaned_jd:
        return 0.0
        
    try:
        # Use English stop words to filter out non-informative words
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([cleaned_jd, cleaned_resume])
        
        # If vocabulary is empty, the matrix might be empty or raise an error
        if tfidf_matrix.shape[1] == 0:
            return 0.0
            
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(round(similarity * 100, 1))
    except Exception as e:
        logger.warning(f"Error computing TF-IDF similarity (possibly empty vocabulary): {e}")
        return 0.0

def compute_skill_overlap(resume_skills: List[str], jd_skills: List[str]) -> Tuple[float, List[str], List[str]]:
    """
    Compares resume skills against the job description requirements.
    
    Returns:
        - float: Skill match score percentage (0.0 to 100.0)
        - List[str]: Matched skills (intersection)
        - List[str]: Missing skills (requested in JD, missing in resume)
    """
    resume_set = {s.lower() for s in resume_skills}
    jd_set = {s.lower() for s in jd_skills}
    
    if not jd_set:
        return 100.0, list(resume_skills), []
        
    matched_set = resume_set.intersection(jd_set)
    missing_set = jd_set.difference(resume_set)
    
    r_map = {s.lower(): s for s in resume_skills}
    j_map = {s.lower(): s for s in jd_skills}
    
    matched_skills = sorted([j_map.get(s, r_map.get(s, s)) for s in matched_set])
    missing_skills = sorted([j_map[s] for s in missing_set])
    
    score = (len(matched_set) / len(jd_set)) * 100
    
    return float(round(score, 1)), matched_skills, missing_skills

def compute_composite_score(
    tfidf_score: float, 
    skill_score: float, 
    weights: Tuple[float, float] = (0.6, 0.4)
) -> float:
    """
    Combines the TF-IDF cosine similarity score and the Skill Overlap score
    using weights.
    """
    w1, w2 = weights
    total_w = w1 + w2
    if total_w != 1.0 and total_w > 0:
        w1 = w1 / total_w
        w2 = w2 / total_w
        
    composite = (tfidf_score * w1) + (skill_score * w2)
    return float(round(composite, 1))

def compute_match_score(
    resume_text: str,
    jd_text: str,
    resume_skills: List[str],
    jd_required_skills: List[str],
    weight_tfidf: float = 0.6,
    weight_skill: float = 0.4
) -> Dict[str, Any]:
    """
    Computes a composite match score between a resume and a job description.
    
    Args:
        resume_text: Raw or cleaned text of the resume.
        jd_text: Raw or cleaned text of the job description.
        resume_skills: List of skills extracted from the resume.
        jd_required_skills: List of skills required by the job description.
        weight_tfidf: Weight applied to TF-IDF score (default 0.6).
        weight_skill: Weight applied to skill overlap score (default 0.4).
        
    Returns:
        dict containing:
            - tfidf_score: 0-100, cosine similarity from TF-IDF
            - skill_overlap_score: 0-100, percentage of JD skills found in resume
            - final_score: weighted combination of tfidf_score and skill_overlap_score
            - matched_skills: list (intersection of resume and JD skills)
            - missing_skills: list (JD skills not found in resume)
    """
    tfidf_score = compute_tfidf_similarity(resume_text, jd_text)
    skill_overlap_score, matched_skills, missing_skills = compute_skill_overlap(resume_skills, jd_required_skills)
    
    # Normalize weights if they don't sum to 1.0 (and sum is positive)
    w_sum = weight_tfidf + weight_skill
    if w_sum != 1.0 and w_sum > 0:
        w_tfidf = weight_tfidf / w_sum
        w_skill = weight_skill / w_sum
    else:
        w_tfidf = weight_tfidf
        w_skill = weight_skill
        
    final_score = (tfidf_score * w_tfidf) + (skill_overlap_score * w_skill)
    final_score = float(round(final_score, 1))
    
    return {
        "tfidf_score": tfidf_score,
        "skill_overlap_score": skill_overlap_score,
        "final_score": final_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }
