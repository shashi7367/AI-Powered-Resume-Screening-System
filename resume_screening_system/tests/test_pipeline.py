import sys
import unittest
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.pdf_extractor import extract_text_from_pdf
from src.skill_extractor import (
    extract_name, extract_skills, extract_education,
    extract_experience, extract_contact_info,
    load_skills_database, extract_resume_info,
    extract_jd_requirements,
)
from src.matcher import (
    compute_tfidf_similarity, compute_skill_overlap,
    compute_composite_score, compute_match_score,
)
from src.ranker import rank_candidates
from src.feedback_generator import generate_feedback, generate_candidate_feedback
from src.database import (
    init_db, create_job_posting, get_job_postings,
    delete_job_posting, save_candidate_result,
    get_candidates, delete_candidates_for_posting,
)


class TestTalentRankPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pdf_dir = project_root / "sample_data" / "sample_resumes"
        cls.alice_pdf = cls.pdf_dir / "alice_devlin_senior_python_developer.pdf"
        cls.test_db = str(project_root / "tests" / "_test_screener.db")

        if not cls.alice_pdf.exists():
            raise FileNotFoundError(
                f"Setup failed. Mock resume not found at {cls.alice_pdf}. "
                "Run generate_resumes.py first."
            )

        cls.alice_text = extract_text_from_pdf(cls.alice_pdf)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db):
            try:
                os.remove(cls.test_db)
            except Exception:
                pass

    # ------------------------------------------------------------------ PDF
    def test_pdf_extraction(self):
        """Test PDF extraction yields structured content."""
        self.assertIsNotNone(self.alice_text)
        self.assertGreater(len(self.alice_text), 200)
        self.assertIn("Alice Devlin", self.alice_text)

    # ------------------------------------------------------------ Skills DB
    def test_skills_db_loading(self):
        """Test skills database JSON loader works."""
        db = load_skills_database()
        self.assertIsInstance(db, dict)
        self.assertIn("Python", db)
        self.assertIn("Docker", db)

    # -------------------------------------------------------------- NLP
    def test_nlp_extraction(self):
        """Test parsing of candidate personal details, skills, education, and experience."""
        name = extract_name(self.alice_text)
        email, phone = extract_contact_info(self.alice_text)
        skills = extract_skills(self.alice_text)
        edu = extract_education(self.alice_text)
        exp_years, history = extract_experience(self.alice_text)

        self.assertEqual(name, "Alice Devlin")
        self.assertEqual(email, "alice.devlin@email.com")
        self.assertEqual(phone, "+1-555-019-9123")
        self.assertIn("Python", skills)
        self.assertIn("AWS", skills)
        self.assertGreaterEqual(exp_years, 5.0)
        # extract_education now returns degree keywords, not institution names
        self.assertTrue(any("Bachelor" in entry for entry in edu))
        self.assertTrue(any("Senior Software Engineer" in hist for hist in history))

    # --------------------------------------------------------- Similarity
    def test_similarity_matching(self):
        """Test TF-IDF cosine similarity calculations and skill overlaps."""
        jd = "Required tech: Python, FastAPI, Docker, PostgreSQL. Need Senior Engineer."
        jd_skills = extract_skills(jd)

        tfidf = compute_tfidf_similarity(self.alice_text, jd)
        skill_score, matched, missing = compute_skill_overlap(
            extract_skills(self.alice_text), jd_skills
        )
        composite = compute_composite_score(tfidf, skill_score, weights=(0.5, 0.5))

        self.assertGreater(tfidf, 0.0)
        self.assertEqual(skill_score, 100.0)
        self.assertGreater(composite, 50.0)
        self.assertIn("Python", matched)
        self.assertEqual(len(missing), 0)

    # ------------------------------------------------ Unified match score
    def test_unified_match_score(self):
        """Test compute_match_score unified interface."""
        jd_skills = ["Python", "Django", "Docker"]
        resume_skills = ["Python", "Docker", "Git"]

        res = compute_match_score(
            resume_text=self.alice_text,
            jd_text="Senior Python Backend Developer. Must know Django and Docker.",
            resume_skills=resume_skills,
            jd_required_skills=jd_skills,
            weight_tfidf=0.6,
            weight_skill=0.4
        )
        for key in ("final_score", "tfidf_score", "skill_overlap_score",
                     "matched_skills", "missing_skills"):
            self.assertIn(key, res)

        self.assertIn("Python", res["matched_skills"])
        self.assertIn("Django", res["missing_skills"])

    def test_matcher_edge_cases(self):
        """Test matcher handles empty strings, short text, or empty vocabularies safely."""
        res = compute_match_score(
            resume_text="",
            jd_text="",
            resume_skills=[],
            jd_required_skills=[],
            weight_tfidf=0.6,
            weight_skill=0.4
        )
        self.assertEqual(res["tfidf_score"], 0.0)
        self.assertEqual(res["final_score"], 40.0)  # skill_overlap of empty is 100% * 0.4 = 40.0

        # test with only punctuation (empty vocabulary for TfidfVectorizer)
        res_empty_vocab = compute_match_score(
            resume_text="!!! ???",
            jd_text="...",
            resume_skills=["Python"],
            jd_required_skills=["Python"],
            weight_tfidf=0.6,
            weight_skill=0.4
        )
        self.assertEqual(res_empty_vocab["tfidf_score"], 0.0)
        self.assertEqual(res_empty_vocab["final_score"], 40.0) # skill score 100% * 0.4 = 40.0

    def test_matcher_comparative(self):
        """Test that a clearly-matching resume scores higher than a clearly-non-matching resume."""
        jd_text = "Senior Python Developer with AWS, Django, Docker, Kubernetes experience."
        jd_skills = ["Python", "AWS", "Django", "Docker", "Kubernetes"]

        matching_resume_text = "Expert software engineer. Python, Django, Kubernetes, AWS, Docker."
        matching_resume_skills = ["Python", "Django", "Kubernetes", "AWS", "Docker"]

        non_matching_resume_text = "Digital Marketing Specialist with SEO, Google Analytics, social media copywriting skills."
        non_matching_resume_skills = ["SEO", "Google Analytics", "Social Media", "Copywriting"]

        matching_score_res = compute_match_score(
            resume_text=matching_resume_text,
            jd_text=jd_text,
            resume_skills=matching_resume_skills,
            jd_required_skills=jd_skills,
            weight_tfidf=0.5,
            weight_skill=0.5
        )

        non_matching_score_res = compute_match_score(
            resume_text=non_matching_resume_text,
            jd_text=jd_text,
            resume_skills=non_matching_resume_skills,
            jd_required_skills=jd_skills,
            weight_tfidf=0.5,
            weight_skill=0.5
        )

        self.assertGreater(matching_score_res["final_score"], non_matching_score_res["final_score"])
        self.assertGreater(matching_score_res["tfidf_score"], non_matching_score_res["tfidf_score"])
        self.assertGreater(matching_score_res["skill_overlap_score"], non_matching_score_res["skill_overlap_score"])

    # ----------------------------------------------------------- Ranking
    def test_ranking(self):
        """Test ranker returns a DataFrame sorted by score descending."""
        jd = "Looking for Product Manager with agile scrum certifications."
        jd_skills = extract_skills(jd)

        bob_text = extract_text_from_pdf(
            self.pdf_dir / "bob_chen_data_scientist.pdf"
        )
        charlie_text = extract_text_from_pdf(
            self.pdf_dir / "charlie_miller_product_manager.pdf"
        )

        candidates = [
            {"name": "Bob Chen", "skills": extract_skills(bob_text), "raw_text": bob_text},
            {"name": "Charlie Miller", "skills": extract_skills(charlie_text), "raw_text": charlie_text},
        ]

        scored_candidates = []
        for cand in candidates:
            match_res = compute_match_score(
                resume_text=cand["raw_text"],
                jd_text=jd,
                resume_skills=cand["skills"],
                jd_required_skills=jd_skills,
                weight_tfidf=0.5,
                weight_skill=0.5
            )
            cand.update(match_res)
            scored_candidates.append(cand)

        ranked = rank_candidates(scored_candidates)

        self.assertEqual(ranked.iloc[0]["name"], "Charlie Miller")
        self.assertEqual(ranked.iloc[0]["rank"], 1)
        self.assertEqual(ranked.iloc[1]["name"], "Bob Chen")
        self.assertEqual(ranked.iloc[1]["rank"], 2)
        # Verify ranker can sort by final_score (mapped key)
        self.assertGreater(ranked.iloc[0]["final_score"], ranked.iloc[1]["final_score"])

    def test_filter_by_min_score(self):
        """Test the filter_by_min_score helper filters rows properly."""
        from src.ranker import filter_by_min_score
        import pandas as pd
        df = pd.DataFrame([
            {"rank": 1, "name": "Alice", "final_score": 85.0},
            {"rank": 2, "name": "Bob", "final_score": 60.0},
            {"rank": 3, "name": "Charlie", "final_score": 45.0}
        ])
        
        filtered = filter_by_min_score(df, 60.0)
        self.assertEqual(len(filtered), 2)
        self.assertNotIn("Charlie", filtered["name"].values)
        self.assertIn("Alice", filtered["name"].values)
        self.assertIn("Bob", filtered["name"].values)

    # --------------------------------------------------------- Feedback
    def test_jd_requirements_extraction(self):
        """Test extract_jd_requirements returns skills and experience."""
        jd = (
            "We need a Senior Python Backend Developer with 5+ years of experience. "
            "Must know Python, Django, Docker, PostgreSQL, and REST APIs."
        )
        result = extract_jd_requirements(jd)

        self.assertIn("required_skills", result)
        self.assertIn("experience_years", result)
        self.assertIn("Python", result["required_skills"])
        self.assertIn("Django", result["required_skills"])
        self.assertIn("Docker", result["required_skills"])
        self.assertGreaterEqual(result["experience_years"], 5)

    def test_feedback_generator(self):
        """Test rule-based fallback feedback generation."""
        cand_info = {
            "name": "Alice Devlin",
            "experience_years": 8,
            "education": ["Bachelor of Science in Computer Science"],
        }
        match_res = {
            "final_score": 85.0,
            "tfidf_score": 70.0,
            "skill_overlap_score": 100.0,
            "matched_skills": ["Python", "FastAPI"],
            "missing_skills": ["Kubernetes"],
        }

        # Test new signature
        feedback1 = generate_feedback(candidate_info=cand_info, match_result=match_res, use_llm=False)
        feedback2 = generate_candidate_feedback(candidate_info=cand_info, match_result=match_res, use_llm=False)

        self.assertIsNotNone(feedback1)
        self.assertEqual(feedback1, feedback2)
        self.assertIn("Alice Devlin", feedback1)
        # Alice Devlin length is 12. 12 % 3 = 0. So it picks variant index 0.
        # Variant 0 for Strong Fit has: "relevant experience", "highly recommended to advance"
        self.assertIn("8 years", feedback1)
        self.assertIn("highly recommended to advance", feedback1)

    # ======================================================= DATABASE TESTS
    def test_database_init_and_job_crud(self):
        """Test init_db, create/get/delete job postings."""
        # 1. init_db creates the file
        init_db(self.test_db)
        self.assertTrue(os.path.exists(self.test_db))

        # 2. create_job_posting
        jid = create_job_posting(
            title="Backend Engineer",
            description_text="Python, Django, REST APIs",
            weight_tfidf=0.5,
            weight_skill=0.5,
            db_path=self.test_db,
        )
        self.assertIsInstance(jid, int)

        # 3. get_job_postings returns it
        postings = get_job_postings(db_path=self.test_db)
        self.assertGreaterEqual(len(postings), 1)
        latest = postings[0]
        self.assertEqual(latest["title"], "Backend Engineer")
        self.assertEqual(latest["weight_tfidf"], 0.5)
        self.assertEqual(latest["weight_skill"], 0.5)

        # 4. delete_job_posting removes it
        delete_job_posting(jid, db_path=self.test_db)
        postings_after = get_job_postings(db_path=self.test_db)
        ids_after = [p["id"] for p in postings_after]
        self.assertNotIn(jid, ids_after)

    def test_database_candidate_crud(self):
        """Test save/get/delete candidate records."""
        init_db(self.test_db)

        # Create a job posting to attach candidates to
        jid = create_job_posting(
            title="Data Scientist",
            description_text="ML, Python, SQL",
            db_path=self.test_db,
        )

        # save_candidate_result
        cand = {
            "name": "Alice Devlin",
            "email": "alice@email.com",
            "phone": "555-1234",
            "skills": ["Python", "Docker"],
            "education": ["Stanford University"],
            "experience": "8 years",
            "tfidf_score": 72.5,
            "skill_overlap_score": 100.0,
            "final_score": 83.5,
            "feedback": "Strong Fit candidate.",
            "resume_filename": "alice_devlin.pdf",
        }
        cid = save_candidate_result(jid, cand, db_path=self.test_db)
        self.assertIsInstance(cid, int)

        # get_candidates with job_posting_id
        rows = get_candidates(jid, db_path=self.test_db)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["name"], "Alice Devlin")
        self.assertEqual(row["final_score"], 83.5)
        self.assertEqual(row["tfidf_score"], 72.5)
        self.assertEqual(row["skill_overlap_score"], 100.0)
        self.assertEqual(row["skills"], ["Python", "Docker"])
        self.assertEqual(row["resume_filename"], "alice_devlin.pdf")

        # get_candidates with no filter returns all
        all_rows = get_candidates(db_path=self.test_db)
        self.assertGreaterEqual(len(all_rows), 1)

        # delete_candidates_for_posting
        delete_candidates_for_posting(jid, db_path=self.test_db)
        after = get_candidates(jid, db_path=self.test_db)
        self.assertEqual(len(after), 0)

        # The posting itself should still exist
        postings = get_job_postings(db_path=self.test_db)
        ids = [p["id"] for p in postings]
        self.assertIn(jid, ids)

        # Clean up
        delete_job_posting(jid, db_path=self.test_db)

    def test_cascade_delete(self):
        """Deleting a job posting cascades to its candidates."""
        init_db(self.test_db)

        jid = create_job_posting(
            title="Temp Role",
            description_text="test",
            db_path=self.test_db,
        )
        save_candidate_result(jid, {"name": "Temp Candidate", "final_score": 50}, db_path=self.test_db)
        self.assertEqual(len(get_candidates(jid, db_path=self.test_db)), 1)

        delete_job_posting(jid, db_path=self.test_db)
        self.assertEqual(len(get_candidates(jid, db_path=self.test_db)), 0)


if __name__ == "__main__":
    unittest.main()
