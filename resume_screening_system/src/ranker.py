"""
ranker.py — Ranks candidate details sorted by final match score.

Exposes:
  • rank_candidates(candidate_results: list[dict]) -> pandas.DataFrame
  • filter_by_min_score(df, min_score) -> pandas.DataFrame
"""

import logging
import pandas as pd
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def rank_candidates(candidate_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Takes a list of candidate dictionaries (each containing fields from
    skill_extractor + matcher output, e.g. final_score, tfidf_score, etc.).
    
    Returns a pandas DataFrame sorted by final_score descending, with
    an added 'rank' column (1, 2, 3, ...).
    """
    if not candidate_results:
        return pd.DataFrame(columns=["rank", "name", "final_score"])
        
    # Create a copy of the candidate dicts to avoid mutating the inputs
    candidates_copy = [dict(c) for c in candidate_results]
    
    # Sort candidates by final_score descending
    # Default to 0.0 if final_score is missing
    sorted_candidates = sorted(
        candidates_copy, 
        key=lambda x: float(x.get("final_score", x.get("score", 0.0))), 
        reverse=True
    )
    
    # Add rank and standardize score key name if needed
    for index, cand in enumerate(sorted_candidates):
        cand["rank"] = index + 1
        # If 'score' is present but 'final_score' is not, populate 'final_score'
        if "final_score" not in cand and "score" in cand:
            cand["final_score"] = cand["score"]
            
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(sorted_candidates)
    
    # Ensure rank column is the first column for cleaner visualization
    if "rank" in df.columns:
        cols = ["rank"] + [col for col in df.columns if col != "rank"]
        df = df[cols]
        
    return df

def filter_by_min_score(df: pd.DataFrame, min_score: float) -> pd.DataFrame:
    """
    Filters candidates in the DataFrame by a minimum match score threshold.
    
    Args:
        df: The pandas DataFrame returned by rank_candidates.
        min_score: Minimum match score threshold (0-100).
        
    Returns:
        pandas.DataFrame: Filtered DataFrame (preserving sorted order).
    """
    if df.empty:
        return df
        
    # Check for 'final_score' first, fallback to 'score'
    score_col = "final_score" if "final_score" in df.columns else "score"
    if score_col not in df.columns:
        logger.warning("No score column found in DataFrame to filter by.")
        return df
        
    return df[df[score_col] >= min_score].copy()
