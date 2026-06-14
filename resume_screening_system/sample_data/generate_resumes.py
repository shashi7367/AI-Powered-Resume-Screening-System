import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def create_resume_pdf(filename: str, name: str, email: str, phone: str, education: list, skills: list, experience: list):
    doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'ResumeTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'ResumeSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=15
    )
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'ResumeBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=4
    )
    
    bold_body_style = ParagraphStyle(
        'ResumeBoldBody',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    story = []
    story.append(Paragraph(name, title_style))
    story.append(Paragraph(f"Email: {email} | Phone: {phone} | Location: San Francisco, CA", subtitle_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("PROFESSIONAL SUMMARY", heading_style))
    summary_text = (
        f"Dedicated and results-oriented professional with extensive knowledge in engineering "
        f"and software components. Experienced in collaborative environments, focusing on quality development, "
        f"robust architectures, and efficient problem solving."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("TECHNICAL SKILLS", heading_style))
    skills_str = ", ".join(skills)
    story.append(Paragraph(skills_str, body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("WORK EXPERIENCE", heading_style))
    for job in experience:
        title_company = f"<b>{job['title']}</b> - {job['company']}"
        dates = f"<i>{job['dates']}</i>"
        story.append(Paragraph(f"{title_company} ({dates})", bold_body_style))
        for bullet in job['bullets']:
            story.append(Paragraph(f"• {bullet}", body_style))
        story.append(Spacer(1, 6))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("EDUCATION", heading_style))
    for edu in education:
        story.append(Paragraph(f"<b>{edu['degree']}</b> - {edu['school']} ({edu['year']})", body_style))
    
    doc.build(story)

def main():
    current_dir = Path(__file__).resolve().parent
    output_dir = current_dir / "sample_resumes"
    output_dir.mkdir(exist_ok=True)
    
    # 1. Alice Devlin (Senior Python Dev)
    create_resume_pdf(
        str(output_dir / "alice_devlin_senior_python_developer.pdf"),
        name="Alice Devlin",
        email="alice.devlin@email.com",
        phone="+1-555-019-9123",
        education=[
            {"degree": "Bachelor of Science in Computer Science", "school": "Stanford University", "year": "2018"}
        ],
        skills=["Python", "Django", "FastAPI", "PostgreSQL", "AWS", "Docker", "Kubernetes", "CI/CD", "Git", "Redis", "REST API", "System Design"],
        experience=[
            {
                "title": "Senior Software Engineer",
                "company": "TechCorp Solutions",
                "dates": "2021 - Present",
                "bullets": [
                    "Lead a team of 4 developers to build scalable Backend REST APIs using Python, FastAPI, and PostgreSQL.",
                    "Migrated legacy monolithic services to Microservices running on Docker and Kubernetes inside AWS.",
                    "Improved API response times by 40% through redis caching and database index tuning.",
                    "Implemented CI/CD pipelines using GitHub Actions to automate unit testing and container deployment."
                ]
            },
            {
                "title": "Python Developer",
                "company": "DevSoft Inc.",
                "dates": "2018 - 2021",
                "bullets": [
                    "Developed backend web modules for high-traffic Django web application.",
                    "Collaborated with frontend React developers to implement features and RESTful endpoints.",
                    "Analyzed databases and optimized queries, cutting SQL execution time in half."
                ]
            }
        ]
    )
    
    # 2. Bob Chen (Mid-Level Data Scientist)
    create_resume_pdf(
        str(output_dir / "bob_chen_data_scientist.pdf"),
        name="Bob Chen",
        email="bob.chen@email.com",
        phone="+1-555-014-4321",
        education=[
            {"degree": "Master of Science in Data Science", "school": "University of California, Berkeley", "year": "2020"},
            {"degree": "Bachelor of Science in Mathematics", "school": "University of Washington", "year": "2018"}
        ],
        skills=["Python", "SQL", "Pandas", "NumPy", "Scikit-Learn", "PyTorch", "TensorFlow", "Machine Learning", "Deep Learning", "Docker", "Git", "Data Analysis", "Tableau"],
        experience=[
            {
                "title": "Data Scientist",
                "company": "AnalyticsLabs",
                "dates": "2022 - Present",
                "bullets": [
                    "Designed and implemented Machine Learning predictive models using Scikit-Learn and PyTorch.",
                    "Extracted data from PostgreSQL databases and performed data analysis using Pandas and NumPy.",
                    "Deployed Deep Learning computer vision models in production containers using Docker.",
                    "Created business intelligence dashboards in Tableau for executive leadership."
                ]
            },
            {
                "title": "Junior Data Analyst",
                "company": "InfoSys Technologies",
                "dates": "2020 - 2022",
                "bullets": [
                    "Cleaned large unstructured datasets and performed statistical analysis to drive product decisions.",
                    "Wrote complex SQL queries to query analytics tables and compile reports."
                ]
            }
        ]
    )
    
    # 3. Charlie Miller (Junior Product Manager)
    create_resume_pdf(
        str(output_dir / "charlie_miller_product_manager.pdf"),
        name="Charlie Miller",
        email="charlie.miller@email.com",
        phone="+1-555-012-2345",
        education=[
            {"degree": "Master of Business Administration", "school": "Boston University", "year": "2022"},
            {"degree": "Bachelor of Arts in Business Administration", "school": "New York University", "year": "2020"}
        ],
        skills=["Project Management", "Agile", "Scrum", "Kanban", "Jira", "Communication", "Leadership", "Teamwork", "Collaboration"],
        experience=[
            {
                "title": "Product Owner",
                "company": "WebStart Inc.",
                "dates": "2024 - Present",
                "bullets": [
                    "Define product requirements and lead sprint planning meetings inside Agile and Scrum frameworks.",
                    "Manage project boards and backlogs in Jira, collaborating with designers and software developers.",
                    "Presented product roadmap updates to directors, ensuring cross-functional alignment."
                ]
            },
            {
                "title": "Associate Project Manager",
                "company": "DesignAgency",
                "dates": "2022 - 2024",
                "bullets": [
                    "Facilitated project updates and daily standups using Kanban methodologies.",
                    "Communicated directly with enterprise clients to coordinate software releases."
                ]
            }
        ]
    )
    
    # 4. Diana Prince (Digital Marketing Specialist)
    create_resume_pdf(
        str(output_dir / "diana_prince_digital_marketing_specialist.pdf"),
        name="Diana Prince",
        email="diana.prince@email.com",
        phone="+1-555-015-5678",
        education=[
            {"degree": "Bachelor of Arts in Communication", "school": "University of Southern California", "year": "2020"}
        ],
        skills=["SEO", "SEM", "Content Strategy", "Google Analytics", "Social Media", "Email Marketing", "Copywriting", "PPC", "Communication", "Teamwork", "Brand Management"],
        experience=[
            {
                "title": "Digital Marketing Manager",
                "company": "GrowthCo Media",
                "dates": "2022 - Present",
                "bullets": [
                    "Formulated content strategy and digital marketing campaigns, growing inbound leads by 50%.",
                    "Managed social media channels and PPC advertising budgets, monitoring web traffic through Google Analytics.",
                    "Coordinated email marketing campaigns, writing high-converting copywriting newsletters."
                ]
            },
            {
                "title": "Digital Copywriter",
                "company": "AdAgency",
                "dates": "2020 - 2022",
                "bullets": [
                    "Wrote copy for social media posts, search engine optimization (SEO), and pay-per-click (PPC) ads.",
                    "Collaborated with brand management teams to ensure brand voice consistency."
                ]
            }
        ]
    )
    
    print("Successfully generated mock resume PDFs in 'sample_data/sample_resumes/' folder!")

if __name__ == "__main__":
    main()
