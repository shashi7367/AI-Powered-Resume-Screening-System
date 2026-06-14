"""
skill_extractor.py — NLP-powered information extraction from resumes and JDs.

Public API
----------
extract_resume_info(resume_text)  → dict with name, email, phone, skills,
                                    experience_years, education
extract_jd_requirements(jd_text)  → dict with required_skills, experience_years
extract_skills(text)              → list[str]  (shared helper, also used directly
                                    by matcher / ranker / tests)

Internal implementation uses spaCy (en_core_web_sm) for NER-based name
detection and PhraseMatcher-based skill extraction, with regex fallbacks
for contact info, experience, and education.
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import spacy
import spacy.cli
from spacy.matcher import PhraseMatcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded spaCy model (singleton)
# ---------------------------------------------------------------------------
_nlp = None


def _get_nlp():
    """Load en_core_web_sm once; auto-download if missing."""
    global _nlp
    if _nlp is not None:
        return _nlp

    model = "en_core_web_sm"
    try:
        _nlp = spacy.load(model)
        logger.info("Loaded spaCy model: %s", model)
    except OSError:
        logger.info("spaCy model '%s' not found — attempting download…", model)
        try:
            spacy.cli.download(model)
            _nlp = spacy.load(model)
            logger.info("Downloaded and loaded '%s'", model)
        except Exception as exc:
            logger.warning("Could not download '%s' (%s) — using blank model.", model, exc)
            _nlp = spacy.blank("en")
    return _nlp


# ---------------------------------------------------------------------------
# Skills database loader
# ---------------------------------------------------------------------------

_skills_cache: Optional[Dict[str, List[str]]] = None


def load_skills_database() -> Dict[str, List[str]]:
    """
    Loads the canonical skill → aliases mapping from
    ``data/skills_database.json``.  The result is cached after the first call.

    Returns:
        dict  —  ``{ "Python": ["python", "py"], … }``
    """
    global _skills_cache
    if _skills_cache is not None:
        return _skills_cache

    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "skills_database.json",
        Path("data/skills_database.json"),
        Path("resume_screening_system/data/skills_database.json"),
    ]
    for path in candidates:
        if path.exists():
            try:
                with open(path, "r") as fh:
                    _skills_cache = json.load(fh)
                    logger.info("Loaded skills database from %s", path)
                    return _skills_cache
            except Exception as exc:
                logger.error("Failed to parse %s: %s", path, exc)

    logger.warning("skills_database.json not found — using minimal fallback.")
    _skills_cache = {
        "Python": ["python", "py"],
        "Java": ["java"],
        "JavaScript": ["javascript", "js"],
        "SQL": ["sql"],
        "Docker": ["docker"],
        "Git": ["git"],
    }
    return _skills_cache


# ---------------------------------------------------------------------------
# Name extraction
# ---------------------------------------------------------------------------

# Words that should never appear in a real person name
_NAME_BLACKLIST = frozenset({
    "resume", "curriculum", "vitae", "summary", "experience", "education",
    "skills", "work", "profile", "contact", "email", "phone", "engineer",
    "developer", "analyst", "manager", "lead", "architect", "senior",
    "junior", "certified", "professional", "learning", "deep", "machine",
    "neural", "networks", "data", "science", "pytorch", "tensorflow",
    "docker", "kubernetes", "cloud", "aws", "gcp", "azure", "sql",
    "objective", "references", "technical", "projects", "publications",
})


def _extract_name(text: str) -> str:
    """
    Best-effort candidate name extraction.

    Strategy:
      1. Run spaCy NER on the first ~800 chars and pick the first
         plausible PERSON entity (2–4 alphabetic tokens, none blacklisted).
      2. Fallback: take the first non-empty line that looks like a name
         (short, all-alpha words, not a section heading).
    """
    nlp = _get_nlp()
    doc = nlp(text[:800])

    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        # Clean stray newlines / pipes from the entity span
        cleaned = re.split(r"[\n|:]", ent.text)[0].strip()
        parts = cleaned.split()
        if 2 <= len(parts) <= 4 and all(
            p.isalpha() and len(p) > 1 and p.lower() not in _NAME_BLACKLIST
            for p in parts
        ):
            return cleaned

    # Fallback heuristic — first few non-empty lines
    for line in text.splitlines()[:5]:
        line = re.split(r"[\n|:]", line)[0].strip()
        parts = line.split()
        if (
            2 <= len(parts) <= 4
            and len(line) <= 40
            and all(p.isalpha() for p in parts)
            and not any(w in line.lower() for w in _NAME_BLACKLIST)
        ):
            return line

    return "Unknown Candidate"


# ---------------------------------------------------------------------------
# Contact info
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\-.\s]?)?\(?\d{3}\)?[\-.\s]?\d{3}[\-.\s]?\d{4}"
    r"|\b\d{10}\b"
)


def _extract_contact(text: str) -> Tuple[str, str]:
    """Return ``(email, phone)`` — empty strings if not found."""
    emails = _EMAIL_RE.findall(text)
    phones = _PHONE_RE.findall(text)
    return (emails[0] if emails else "", phones[0].strip() if phones else "")


# ---------------------------------------------------------------------------
# Skill extraction  (shared between resume and JD processing)
# ---------------------------------------------------------------------------

def extract_skills(text: str, skill_dict: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """
    Finds skills in *text* by matching against the skills database using
    spaCy's ``PhraseMatcher`` (case-insensitive).

    Returns a sorted list of canonical skill names (e.g. ``"JavaScript"``
    even if the text contained ``"js"``).
    """
    nlp = _get_nlp()
    if skill_dict is None:
        skill_dict = load_skills_database()

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    id_to_skill: Dict[str, str] = {}

    for canonical, aliases in skill_dict.items():
        # Build a match-ID that is a valid spaCy string-store key
        match_id = (
            canonical.upper()
            .replace("/", "_").replace(" ", "_")
            .replace(".", "_").replace("+", "P")
            .replace("#", "SHARP")
        )
        patterns = [nlp.make_doc(alias) for alias in aliases]
        matcher.add(match_id, patterns)
        id_to_skill[match_id] = canonical

    doc = nlp(text)
    found: set[str] = set()
    for match_id_hash, _start, _end in matcher(doc):
        match_str = nlp.vocab.strings[match_id_hash]
        found.add(id_to_skill.get(match_str, match_str))

    return sorted(found)


# ---------------------------------------------------------------------------
# Experience extraction
# ---------------------------------------------------------------------------

_EXP_PATTERNS = [
    # "8 years of experience", "5+ yrs professional experience"
    re.compile(r"\b(\d{1,2})\.?\d?\+?\s*(?:years?|yrs?)\b\s*(?:of\s+)?(?:experience|exp|working|professional)\b", re.I),
    # "experience: 5 years"
    re.compile(r"\b(?:experience|exp)\b\s*[:–\-]?\s*(\d{1,2})\.?\d?\+?\s*(?:years?|yrs?)\b", re.I),
    # "5+ years in backend"
    re.compile(r"\b(\d{1,2})\+?\s*years?\s+in\b", re.I),
]


def _extract_experience_years(text: str) -> int:
    """
    Extracts years-of-experience as an integer.

    Tries explicit phrases first (``"X years of experience"``).  If none
    are found, sums up date-range spans (``2018 – Present``).
    """
    # Explicit mentions
    explicit: List[int] = []
    for pat in _EXP_PATTERNS:
        for m in pat.finditer(text):
            try:
                explicit.append(int(float(m.group(1))))
            except (ValueError, IndexError):
                pass

    if explicit:
        return max(explicit)

    # Infer from date ranges (YYYY – YYYY | Present)
    range_re = re.compile(
        r"\b((?:19|20)\d{2})\s*[\-–—to]+\s*"
        r"((?:19|20)\d{2}|[Pp]resent|[Cc]urrent|[Nn]ow)\b"
    )
    total = 0
    current_year = 2026
    for m in range_re.finditer(text):
        start = int(m.group(1))
        end_str = m.group(2)
        end = current_year if end_str[0].isalpha() else int(end_str)
        total += max(0, end - start)

    return min(total, 40)  # cap at a sane max


# ---------------------------------------------------------------------------
# Education extraction
# ---------------------------------------------------------------------------

_DEGREE_KEYWORDS: List[str] = [
    # Doctorates
    "Ph.D.", "PhD", "Doctorate",
    # Masters
    "Master", "M.S.", "M.S", "MS", "M.Sc.", "M.Sc", "MSc",
    "M.B.A.", "MBA", "M.Tech", "MTech", "M.A.", "MA",
    # Bachelors
    "Bachelor", "B.S.", "B.S", "BS", "B.Sc.", "B.Sc", "BSc",
    "B.A.", "BA", "B.Tech", "BTech", "B.E.", "BE", "BCA",
    # Other
    "Associate Degree", "Diploma",
]


def _extract_education(text: str) -> List[str]:
    """
    Returns a list of detected degree keywords (e.g. ``["B.Tech", "MBA"]``).
    De-duplicated and capped at 4 entries.
    """
    found: list[str] = []
    seen_lower: set[str] = set()

    for kw in _DEGREE_KEYWORDS:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text, re.IGNORECASE):
            if kw.lower() not in seen_lower:
                seen_lower.add(kw.lower())
                found.append(kw)

    return found[:4]


# ===================================================================
#  PUBLIC API — Resume
# ===================================================================

def extract_resume_info(resume_text: str) -> dict:
    """
    Extracts structured candidate information from raw resume text.

    Returns:
        dict with keys:
            name             (str)   — best-effort candidate name
            email            (str)   — regex-extracted email
            phone            (str)   — regex-extracted phone number
            skills           (list)  — canonical skill names matched against
                                       ``data/skills_database.json``
            experience_years (int)   — years of experience
            education        (list)  — detected degree keywords, e.g.
                                       ``["B.Tech", "MBA"]``
    """
    name = _extract_name(resume_text)
    email, phone = _extract_contact(resume_text)
    skills = extract_skills(resume_text)
    experience_years = _extract_experience_years(resume_text)
    education = _extract_education(resume_text)

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": skills,
        "experience_years": experience_years,
        "education": education,
    }


# ===================================================================
#  PUBLIC API — Job Description
# ===================================================================

def extract_jd_requirements(jd_text: str) -> dict:
    """
    Extracts structured requirements from a job description.

    Returns:
        dict with keys:
            required_skills  (list)  — canonical skill names found in the JD
            experience_years (int)   — explicit experience requirement
                                       (0 if not stated)
    """
    required_skills = extract_skills(jd_text)
    experience_years = _extract_experience_years(jd_text)

    return {
        "required_skills": required_skills,
        "experience_years": experience_years,
    }


# ===================================================================
#  Legacy aliases (keep old callers working)
# ===================================================================

# These are used by existing tests and the demo script
extract_name = _extract_name
extract_contact_info = _extract_contact
extract_education = _extract_education


def extract_experience(text: str) -> Tuple[float, List[str]]:
    """
    Legacy wrapper — returns ``(years: float, history_highlights: list)``.
    Kept for backward-compat with test_pipeline.py assertions.
    """
    years = _extract_experience_years(text)

    # Extract a handful of job-title lines as "history highlights"
    _titles = [
        "Software Engineer", "Developer", "Data Scientist", "Product Manager",
        "Project Manager", "UX Designer", "DevOps Engineer", "Data Analyst",
        "Business Analyst", "Solutions Architect", "QA Engineer",
        "Full Stack Developer", "Backend Developer", "Frontend Developer",
        "Machine Learning Engineer", "Senior Software Engineer",
    ]
    highlights: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) > 120:
            continue
        for title in _titles:
            if re.search(r"\b" + re.escape(title) + r"\b", line, re.I):
                norm = re.sub(r"\s+", " ", line)
                if norm.lower() not in seen:
                    seen.add(norm.lower())
                    highlights.append(norm)
                break
    return float(years), highlights[:4]
