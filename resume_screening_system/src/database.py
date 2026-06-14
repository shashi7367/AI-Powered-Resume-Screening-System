"""
database.py — SQLite persistence layer for the Resume Screening System.

Manages two tables inside data/resume_screener.db:
  • job_postings  — stores JD text and configurable TF-IDF / skill weights
  • candidates    — stores per-candidate screening results linked to a posting

All SQL uses parameterized queries (? placeholders) to prevent injection.
The data/ directory and .db file are auto-created on first access.
"""

import json
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default database path (relative to the working directory, i.e. project root)
# ---------------------------------------------------------------------------
DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "resume_screener.db")


def _get_connection(db_path: str = None) -> sqlite3.Connection:
    """
    Returns a sqlite3 connection with foreign-key enforcement enabled.
    Auto-creates the parent directory if it doesn't exist yet.
    """
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON;")
    # Return rows as sqlite3.Row so we can convert to dicts easily
    conn.row_factory = sqlite3.Row
    return conn


# ===========================  INIT  =========================================

def init_db(db_path: str = None) -> None:
    """
    Creates the job_postings and candidates tables if they don't already exist.
    Safe to call repeatedly — uses CREATE TABLE IF NOT EXISTS.
    """
    conn = _get_connection(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS job_postings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            title           TEXT    NOT NULL,
            description_text TEXT   NOT NULL,
            weight_tfidf    REAL   NOT NULL DEFAULT 0.6,
            weight_skill    REAL   NOT NULL DEFAULT 0.4,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            job_posting_id      INTEGER NOT NULL,
            name                TEXT    NOT NULL,
            email               TEXT,
            phone               TEXT,
            skills              TEXT,       -- JSON-serialised list
            experience          TEXT,       -- free-form experience description / years
            education           TEXT,       -- free-form education info
            tfidf_score         REAL,
            skill_overlap_score REAL,
            final_score         REAL,
            feedback            TEXT,
            resume_filename     TEXT,
            screened_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_posting_id)
                REFERENCES job_postings (id) ON DELETE CASCADE
        );
    """)

    conn.commit()
    conn.close()
    logger.info("Database tables initialised at: %s", db_path or DB_PATH)


# ========================  JOB POSTINGS  ====================================

def create_job_posting(
    title: str,
    description_text: str,
    weight_tfidf: float = 0.6,
    weight_skill: float = 0.4,
    db_path: str = None,
) -> int:
    """
    Inserts a new job posting and returns its auto-generated ID.

    Args:
        title:            Short human-readable title for the role.
        description_text: Full job-description body text.
        weight_tfidf:     Weight applied to the TF-IDF similarity component (0-1).
        weight_skill:     Weight applied to the skill-overlap component (0-1).

    Returns:
        int — the new job_posting row ID.
    """
    init_db(db_path)
    conn = _get_connection(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO job_postings (title, description_text, weight_tfidf, weight_skill)
        VALUES (?, ?, ?, ?);
        """,
        (title, description_text, weight_tfidf, weight_skill),
    )

    posting_id = cur.lastrowid
    conn.commit()
    conn.close()
    logger.info("Created job posting '%s' → ID %d", title, posting_id)
    return posting_id


def get_job_postings(db_path: str = None) -> List[Dict[str, Any]]:
    """
    Returns every job posting as a list of dicts, newest first.
    """
    init_db(db_path)
    conn = _get_connection(db_path)
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM job_postings ORDER BY created_at DESC;"
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def delete_job_posting(job_posting_id: int, db_path: str = None) -> None:
    """
    Deletes a job posting **and** all associated candidate records
    (CASCADE enforced by the FK constraint + PRAGMA foreign_keys = ON).
    """
    init_db(db_path)
    conn = _get_connection(db_path)
    cur = conn.cursor()

    cur.execute("DELETE FROM job_postings WHERE id = ?;", (job_posting_id,))
    conn.commit()
    conn.close()
    logger.info("Deleted job posting ID %d (and its candidates)", job_posting_id)


# ==========================  CANDIDATES  ====================================

def save_candidate_result(
    job_posting_id: int,
    candidate_dict: Dict[str, Any],
    db_path: str = None,
) -> int:
    """
    Persists one screened candidate record.

    Expected keys in *candidate_dict* (all optional except 'name'):
        name, email, phone, skills (list → JSON), experience, education,
        tfidf_score, skill_overlap_score, final_score, feedback,
        resume_filename.

    Returns:
        int — the new candidate row ID.
    """
    init_db(db_path)
    conn = _get_connection(db_path)
    cur = conn.cursor()

    # Serialise the skills list to a JSON string for storage
    skills_raw = candidate_dict.get("skills", [])
    skills_json = json.dumps(skills_raw) if isinstance(skills_raw, (list, dict)) else str(skills_raw)

    # Normalise education — accept a list or a plain string
    edu_raw = candidate_dict.get("education", "")
    education_str = json.dumps(edu_raw) if isinstance(edu_raw, (list, dict)) else str(edu_raw)

    # Normalise experience — accept a number or a string
    exp_raw = candidate_dict.get("experience", candidate_dict.get("experience_years", ""))
    experience_str = str(exp_raw)

    cur.execute(
        """
        INSERT INTO candidates (
            job_posting_id, name, email, phone, skills,
            experience, education,
            tfidf_score, skill_overlap_score, final_score,
            feedback, resume_filename
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            job_posting_id,
            candidate_dict.get("name", "Unknown"),
            candidate_dict.get("email", ""),
            candidate_dict.get("phone", ""),
            skills_json,
            experience_str,
            education_str,
            float(candidate_dict.get("tfidf_score", 0.0)),
            float(candidate_dict.get("skill_overlap_score", candidate_dict.get("skill_score", 0.0))),
            float(candidate_dict.get("final_score", candidate_dict.get("score", 0.0))),
            candidate_dict.get("feedback", ""),
            candidate_dict.get("resume_filename", candidate_dict.get("filename", "")),
        ),
    )

    candidate_id = cur.lastrowid
    conn.commit()
    conn.close()
    logger.info("Saved candidate '%s' → ID %d", candidate_dict.get("name"), candidate_id)
    return candidate_id


def get_candidates(
    job_posting_id: Optional[int] = None,
    db_path: str = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves candidate records, sorted by final_score descending.

    Args:
        job_posting_id: If provided, only candidates for that posting are
                        returned. If None, **all** candidates are returned.

    Returns:
        list[dict] — each dict mirrors the candidates table columns, with
        the 'skills' field deserialised back into a Python list.
    """
    init_db(db_path)
    conn = _get_connection(db_path)
    cur = conn.cursor()

    if job_posting_id is not None:
        cur.execute(
            """
            SELECT * FROM candidates
            WHERE job_posting_id = ?
            ORDER BY final_score DESC;
            """,
            (job_posting_id,),
        )
    else:
        cur.execute(
            "SELECT * FROM candidates ORDER BY final_score DESC;"
        )

    rows = []
    for r in cur.fetchall():
        d = dict(r)
        # Deserialise JSON skills back to a Python list
        try:
            d["skills"] = json.loads(d["skills"]) if d.get("skills") else []
        except (json.JSONDecodeError, TypeError):
            d["skills"] = []
        # Deserialise education if stored as JSON
        try:
            parsed = json.loads(d["education"]) if d.get("education") else ""
            d["education"] = parsed
        except (json.JSONDecodeError, TypeError):
            pass  # keep as-is (plain string)
        rows.append(d)

    conn.close()
    return rows


def delete_candidates_for_posting(
    job_posting_id: int,
    db_path: str = None,
) -> None:
    """
    Removes all candidate records linked to a given job posting,
    without deleting the posting itself.
    """
    init_db(db_path)
    conn = _get_connection(db_path)
    cur = conn.cursor()

    cur.execute(
        "DELETE FROM candidates WHERE job_posting_id = ?;",
        (job_posting_id,),
    )
    conn.commit()
    conn.close()
    logger.info("Deleted all candidates for job posting ID %d", job_posting_id)
