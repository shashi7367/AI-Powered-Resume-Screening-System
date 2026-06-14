# TalentRank AI Backend: AI-Powered Resume Screening System 💼🤖

TalentRank AI is a high-performance Python backend library designed to parse, analyze, score, persist, and rank candidate resumes against detailed job descriptions. 

This repository exposes a headless backend package (`src/`) that integrates with SQLite for persistence and exposes simple, programmatic APIs for PDF parsing, NLP skill extraction, textual similarity matching, and AI feedback generation.

---

## Project Layout

```
resume_screening_system/
├── requirements.txt            # Package dependencies
├── .env.example                # OpenAI API Key template
├── README.md                   # Setup, layout, and API instructions (this file)
├── run_pipeline_demo.py        # End-to-end programmatic demo script
├── data/
│   ├── skills_database.json    # Predefined JSON catalog of 100+ technical/soft skills
│   └── resume_screener.db      # SQLite database (auto-created)
├── sample_data/
│   ├── sample_resumes/         # Generated mock candidate PDF resumes for the demo
│   ├── generate_resumes.py     # Programmatic PDF resume compiler
│   └── sample_job_description.txt # Static sample Job Description text file
├── src/
│   ├── __init__.py             # Exports core APIs at the package level
│   ├── database.py             # SQLite persistence (job postings, candidates, cascades)
│   ├── pdf_extractor.py        # PDF text parser with BytesIO and PDFExtractionError
│   ├── skill_extractor.py      # spaCy PhraseMatcher loading JSON skills & details
│   ├── matcher.py              # TF-IDF cosine similarity & skill overlap calculation
│   ├── ranker.py               # DataFrame ranking and min_score threshold filtering
│   └── feedback_generator.py   # OpenAI summary evaluator & rule fallback engine
└── tests/
    └── test_pipeline.py        # Automated unittest suite validating all modules
```

---

## Backend API Specification

The backend components are exposed inside the `src` package and can be imported directly:

```python
from src.database import (
    init_db,
    create_job_posting,
    get_job_postings,
    delete_job_posting,
    save_candidate_result,
    get_candidates,
    delete_candidates_for_posting,
)
from src.pdf_extractor import extract_text_from_pdf, PDFExtractionError
from src.skill_extractor import extract_resume_info, extract_jd_requirements
from src.matcher import compute_match_score
from src.ranker import rank_candidates, filter_by_min_score
from src.feedback_generator import generate_feedback
```

### 1. Database Layer (`src/database.py`)
Provides SQLite persistence for job postings and candidate results under `data/resume_screener.db`:
- `init_db(db_path: str = None)`: Initializes the SQLite database.
- `create_job_posting(title: str, description_text: str, weight_tfidf: float = 0.6, weight_skill: float = 0.4, db_path: str = None) -> int`: Creates a new job posting with weights. Returns unique posting ID.
- `get_job_postings(db_path: str = None) -> list[dict]`: Retrieves all job postings.
- `delete_job_posting(job_posting_id: int, db_path: str = None)`: Deletes the posting and cascades deletion to all attached candidates.
- `save_candidate_result(job_posting_id: int, candidate_dict: dict, db_path: str = None) -> int`: Persists candidate scores, contact details, parsed skills, education history, and generated feedback.
- `get_candidates(job_posting_id: int = None, db_path: str = None) -> list[dict]`: Retrieves ranked candidate records for a job posting (or all candidates if None), sorted by final score descending.
- `delete_candidates_for_posting(job_posting_id: int, db_path: str = None)`: Deletes candidates associated with a job posting without deleting the posting itself.

### 2. PDF Parsing Module (`src/pdf_extractor.py`)
- `extract_text_from_pdf(pdf_file: Union[str, Path, bytes, BinaryIO]) -> str`: Extracts raw text from a PDF file path, bytes object, or file-like object. Uses `pdfplumber` with a fallback to `PyPDF2`. Raises `PDFExtractionError` on failure.

### 3. NLP Info Extractor (`src/skill_extractor.py`)
- `extract_resume_info(resume_text: str) -> dict`: Extracts details from resume text:
  ```python
  {
      "name": str,
      "email": str,
      "phone": str,
      "skills": list[str],
      "education": list[str],  # degree keywords, e.g. ["Bachelor", "Master"]
      "experience_years": int
  }
  ```
- `extract_jd_requirements(jd_text: str) -> dict`: Extracts required skills and explicit experience years from a job description text.

### 4. Text Matching Module (`src/matcher.py`)
- `compute_match_score(resume_text, jd_text, resume_skills, jd_required_skills, weight_tfidf=0.6, weight_skill=0.4) -> dict`: Computes similarity scores. Safe against division-by-zero or empty vocabularies:
  ```python
  {
      "tfidf_score": float,         # 0-100 cosine similarity
      "skill_overlap_score": float,  # 0-100 JD skills overlap
      "final_score": float,          # Weighted composite score
      "matched_skills": list[str],   # Skills present in both
      "missing_skills": list[str]    # JD skills missing in resume
  }
  ```

### 5. Candidate Ranking (`src/ranker.py`)
- `rank_candidates(candidate_results: list[dict]) -> pandas.DataFrame`: Formats candidate scores and metadata, sorting by `final_score` descending and adding a `rank` column (1, 2, 3, ...).
- `filter_by_min_score(df: pandas.DataFrame, min_score: float) -> pandas.DataFrame`: Filters the ranked candidates DataFrame by a score threshold.

### 6. Summary Feedback (`src/feedback_generator.py`)
- `generate_feedback(candidate_info: dict, match_result: dict, use_llm: bool = False) -> str`: Generates a short 2-3 sentence feedback summary paragraph. Uses the OpenAI API if `use_llm=True` and `OPENAI_API_KEY` is present. Otherwise, falls back to a randomized rule-based template (3 variants per fit category).

---

## Setup & Installation

### 1. Set Up Virtual Environment
On macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### 4. Configuration (Optional)
To use generative AI feedback, copy `.env.example` to `.env` and fill in your OpenAI key:
```bash
cp .env.example .env
```
Inside `.env`:
```ini
OPENAI_API_KEY=your_actual_openai_key_here
OPENAI_MODEL=gpt-4o-mini
```
*Note: If no API key is present or if it fails, the system automatically uses the rule-based feedback generator.*

---

## How to Run the System

### Step 1. Compile Sample PDF Resumes
Generate the mock candidate resumes (Alice, Bob, Charlie, Diana):
```bash
python sample_data/generate_resumes.py
```

### Step 2. Run the End-to-End Programmatic Pipeline
Execute the demo script to process the resumes, match them against the sample job description, save the results to SQLite, and print the ranked candidates list:
```bash
python run_pipeline_demo.py
```

### Step 3. Execute Automated Unit Tests
Run the unittest suite to verify module APIs, edge-cases, and database logic:
```bash
python -m unittest tests/test_pipeline.py
```
