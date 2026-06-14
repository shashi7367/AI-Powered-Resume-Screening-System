"""
feedback_generator.py — Generates technical recruiter evaluation feedback.

Supports both OpenAI LLM-based feedback and a rule-based template system
with multiple randomized/deterministic template variants.
"""

import os
import logging
from typing import Dict, Any, Optional

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

def _generate_rule_based_feedback(candidate_info: Dict[str, Any], match_result: Dict[str, Any]) -> str:
    """
    Generates structured, professional, rule-based feedback based on matching scores
    and extracted features, using 3 different sentence template variants per category.
    """
    name = candidate_info.get("name", "The candidate")
    exp_years = candidate_info.get("experience_years", 0)
    education = candidate_info.get("education", [])
    
    # Ensure education is formatted nicely
    if isinstance(education, list):
        education_str = ", ".join(education) if education else "degree information not specified"
    else:
        education_str = str(education)
        
    final_score = float(match_result.get("final_score", match_result.get("score", 0.0)))
    matched_skills_list = match_result.get("matched_skills", [])
    missing_skills_list = match_result.get("missing_skills", [])
    
    # Format list output for readability
    matched_str = ", ".join(matched_skills_list[:4]) if matched_skills_list else "None"
    missing_str = ", ".join(missing_skills_list[:4]) if missing_skills_list else "None"
    
    # Select template index based on name length to ensure diversity across candidates
    variant_idx = len(name) % 3
    
    # Determine the category and choose the variant
    if final_score >= 80.0:
        # Strong Fit
        variants = [
            f"{name} is an exceptional fit for this role, achieving a match score of {final_score}%. They possess strong alignment in core technologies such as {matched_str}, backed by {exp_years} years of relevant experience. With {education_str} and no critical gaps identified, they are highly recommended to advance to technical evaluation.",
            f"With a stellar match score of {final_score}%, {name} stands out as a highly qualified candidate. Their background features deep competence in {matched_str} and {exp_years} years of professional experience. No major skill gaps were found, making {name} a Strong Fit and an ideal choice for immediate technical screens.",
            f"An evaluation of {name}'s profile indicates a Strong Fit, scoring {final_score}% overall. Their expertise in {matched_str} matches the requirements remarkably well, complemented by {exp_years} years in the field and education in {education_str}. They are highly recommended for the hiring manager's review."
        ]
        return variants[variant_idx]
        
    elif final_score >= 50.0:
        # Moderate Fit
        variants = [
            f"{name} is a Moderate Fit for this position, demonstrating solid skills in {matched_str} and a match score of {final_score}%. While they offer {exp_years} years of experience, they lack direct mentions of {missing_str}. Recruiter review or a preliminary screening call is recommended to explore these areas.",
            f"Scoring {final_score}%, {name} shows reasonable alignment with requirements. They bring key strengths in {matched_str} and {exp_years} years of experience, though gaps in {missing_str} should be noted. They remain a viable candidate for initial recruiter screening.",
            f"The profile for {name} highlights moderate suitability with a composite score of {final_score}%. Their background includes {matched_str} and {exp_years} years of experience with {education_str}, but missing skills like {missing_str} are areas of concern. A follow-up chat is recommended to gauge overall technical fit."
        ]
        return variants[variant_idx]
        
    else:
        # Weak Fit
        variants = [
            f"{name} shows limited technical alignment for this opening, scoring {final_score}% on our matching index. Although they possess {exp_years} years of experience, they lack critical requested technologies like {missing_str}. We suggest holding on this application for now.",
            f"With a match score of {final_score}%, {name} is currently a Weak Fit for the requirements. Key skills such as {missing_str} are absent from the profile, and their background (including {matched_str}) does not sufficiently overlap with the role. Not recommended to advance.",
            f"Evaluation of {name}'s resume suggests a Weak Fit, scoring {final_score}% composite. Despite {exp_years} years of experience, there is insufficient overlap with the required stack (missing: {missing_str}). It is recommended to hold this candidate profile."
        ]
        return variants[variant_idx]

def generate_feedback(candidate_info: Dict[str, Any], match_result: Dict[str, Any], use_llm: bool = False) -> str:
    """
    Generates recruiter screening summary evaluation.
    
    Args:
        candidate_info: Dict containing name, email, phone, skills, experience_years, education.
        match_result: Dict containing tfidf_score, skill_overlap_score, final_score, matched_skills, missing_skills.
        use_llm: Whether to attempt LLM generation via OpenAI API.
        
    Returns:
        str: Feedback evaluation text. Evaluating overall suitability recommendation (Strong Fit / Moderate Fit / Weak Fit).
    """
    # Defensive/compat check: if caller passed a single combined dict or did not pass match_result properly
    if not isinstance(match_result, dict):
        logger.warning("generate_feedback was called with legacy signature. Attempting auto-recovery.")
        # Try to treat candidate_info as the combined dict and match_result as dummy/jd_text
        combined = candidate_info
        candidate_info = {
            "name": combined.get("name", "Unknown"),
            "experience_years": combined.get("experience_years", 0),
            "education": combined.get("education", [])
        }
        match_result = {
            "final_score": combined.get("score", combined.get("final_score", 0.0)),
            "tfidf_score": combined.get("tfidf_score", 0.0),
            "skill_overlap_score": combined.get("skill_score", combined.get("skill_overlap_score", 0.0)),
            "matched_skills": combined.get("matched_skills", []),
            "missing_skills": combined.get("missing_skills", [])
        }

    # If use_llm is true, try to use OpenAI API
    if use_llm:
        effective_key = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        if effective_key:
            effective_key = effective_key.strip()
            if "your_openai" in effective_key or effective_key == "":
                effective_key = None
                
        if not effective_key:
            logger.info("No valid OpenAI API Key found in environment. Falling back to rule-based evaluation.")
            return _generate_rule_based_feedback(candidate_info, match_result)
            
        final_score = match_result.get("final_score", match_result.get("score", 0.0))
        prompt = f"""
        You are an expert technical recruiter. Write a concise, professional 2-3 sentence candidate screening summary evaluation.
        
        Candidate Details:
        - Name: {candidate_info.get('name', 'Unknown')}
        - Extracted Experience: {candidate_info.get('experience_years', 0)} years
        - Education: {', '.join(candidate_info.get('education', []))}
        
        Matcher Result Details:
        - Overall Match Score: {final_score}%
        - Matching Skills: {', '.join(match_result.get('matched_skills', []))}
        - Missing Skills: {', '.join(match_result.get('missing_skills', []))}
        
        Instructions for feedback:
        1. Sentence 1: Summarize the candidate's primary strength/experience level and core tech skill alignment.
        2. Sentence 2: Address specific skill gaps or experience mismatches relative to the JD requirements.
        3. Sentence 3: State the overall suitability recommendation clearly using one of these exact strings: 'Strong Fit', 'Moderate Fit', or 'Weak Fit', followed by a recommended next step.
        Keep the feedback professional, constructive, and under 90 words. Do not output anything other than the feedback paragraph.
        """
        
        try:
            client = OpenAI(api_key=effective_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful and professional technical recruiter assistant. You output only clean candidate feedback summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            feedback_text = response.choices[0].message.content.strip()
            logger.info("Successfully generated feedback using OpenAI API.")
            return feedback_text
        except Exception as e:
            logger.error(f"OpenAI API call failed with error: {e}. Falling back to rule-based feedback.")
            return _generate_rule_based_feedback(candidate_info, match_result)
            
    # Default to rule-based template
    return _generate_rule_based_feedback(candidate_info, match_result)

# Alias for backwards compatibility
generate_candidate_feedback = generate_feedback
