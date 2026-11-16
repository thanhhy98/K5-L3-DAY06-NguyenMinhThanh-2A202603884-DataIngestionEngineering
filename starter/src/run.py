"""Pipeline entry point for the Day 06 data ingestion exercise."""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .store import PgVectorStore


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BENCHMARK_DIR = ROOT / "benchmark"
RESULTS_DIR = ROOT / "results"


def _generate_demo_corpus() -> tuple[Path, dict]:
    """Generate a small corpus and manifest if the repository is otherwise empty."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    corpus_path = DATA_DIR / "student_handbook_2025.txt"
    if corpus_path.exists():
        manifest_path = DATA_DIR / "manifest.json"
        if manifest_path.exists():
            return corpus_path, json.loads(manifest_path.read_text())

    handbook = [
        {"page": 1, "year": "2025", "audience": "all", "section": "welcome", "title": "Welcome and contact information", "text": "The student handbook introduces campus services, support pathways, and the expected standards for every enrolled learner. All students can contact the registrar, advising office, and student support desk for orientation assistance. Students must confirm their enrollment details before the first academic advising meeting. The campus is committed to accessible study and inclusive participation across every program."},
        {"page": 2, "year": "2025", "audience": "all", "section": "orientation", "title": "Orientation checklist", "text": "Orientation begins with identity verification and digital account activation. Students are expected to complete onboarding tasks before the second week of classes. Students who miss orientation can still register for classes after contacting the advising office for a make-up session. The required attendance threshold is 80 percent for all first-year workshops."},
        {"page": 3, "year": "2025", "audience": "undergraduate", "section": "advising", "title": "Advising policy", "text": "Undergraduate students should meet with an advisor every semester before registration opens. Advisors confirm degree progress, prerequisite sequencing, and expected graduation timelines. Students who change majors must complete a brief review with the faculty advisor and submit updated paperwork before the new term begins."},
        {"page": 4, "year": "2025", "audience": "postgraduate", "section": "advising", "title": "Research advising", "text": "Graduate students are assigned a research advisor at the start of the academic year. The advisor helps review degree milestones, research proposal timing, and ethics approvals. Students must notify the graduate office if a supervisor change is requested or if there is a conflict in research goals."},
        {"page": 5, "year": "2025", "audience": "all", "section": "attendance", "title": "Attendance and participation", "text": "Regular attendance is required for classes, labs, and tutorials. Faculty may require attendance records for students who use scholarships or sponsorship funding. Students with medical leave must submit documentation within ten working days of the absence. Excused absences are recorded by the school office and shared with instructors."},
        {"page": 6, "year": "2025", "audience": "all", "section": "wellbeing", "title": "Academic wellbeing", "text": "The campus wellbeing center provides academic coaching, stress support, and general wellness referrals. Students are encouraged to use the center before a situation becomes critical. Staff can help arrange flexible study plans after a significant health event or family emergency. Peer mentoring is available for first-year students."},
        {"page": 7, "year": "2025", "audience": "undergraduate", "section": "scholarships", "title": "Merit scholarships", "text": "Eligibility for merit scholarships is based on performance in the previous academic year. Students must maintain a cumulative GPA of 3.5 or higher to renew funding. The review committee also checks participation in student organizations and service activities. Students who fail to meet the GPA threshold may appeal through the financial aid office."},
        {"page": 8, "year": "2025", "audience": "undergraduate", "section": "scholarships", "title": "Need-based aid", "text": "Need-based grants are assigned according to submitted financial information and household need. Students must update their application if family income, household size, or dependent status changes materially. A scholarship advisor may request additional evidence before awards are finalized. Requests for reconsideration are open until the end of the semester."},
        {"page": 9, "year": "2025", "audience": "postgraduate", "section": "tuition", "title": "Graduate funding", "text": "Graduate students may receive assistantships, research stipends, or tuition waivers. Funding requires satisfactory progress in coursework and research milestones. A student who receives a funding letter must confirm acceptance by the deadline stated in the letter. Unused funding may be reallocated after the approval of the graduate dean."},
        {"page": 10, "year": "2025", "audience": "all", "section": "library", "title": "Library access", "text": "Library cards remain active for enrolled students and valid alumni on official campus programs. Students must present their official ID to borrow reserved materials. Access to the digital archive is granted via the campus login. Students using the archive off campus must use multi-factor authentication using the university portal."},
        {"page": 11, "year": "2025", "audience": "all", "section": "lab", "title": "Laboratory safety", "text": "Students entering laboratories must complete safety induction before using equipment. Training is refreshed annually and includes emergency exit routes, protective equipment, and cleaning procedures. Students working with chemicals must wear eye protection and gloves at all times. Safety officers can deny access if the paperwork is incomplete or if training is out of date."},
        {"page": 12, "year": "2025", "audience": "all", "section": "lab", "title": "Biohazard rules", "text": "Biohazard waste must be disposed of in designated bins and never mixed with standard trash. Students handling clinical samples must wash hands before and after removing gloves. Any spill or injury must be reported immediately to the lab supervisor and the campus health office. Supervisors review incident reports within one working day."},
        {"page": 13, "year": "2025", "audience": "undergraduate", "section": "discipline", "title": "Academic integrity", "text": "Academic misconduct includes plagiarism, unauthorized collaboration, and misleading citations. Students should cite all sources used in written work and may discuss general concepts with peers, but not share answers directly. Repeated integrity violations may lead to a formal hearing and academic probation. Students may appeal a decision through the student affairs committee."},
        {"page": 14, "year": "2025", "audience": "all", "section": "discipline", "title": "Conduct expectations", "text": "Students are expected to respect community standards in classrooms, online forums, and campus events. Harassment, discrimination, and intimidation are prohibited under the code of conduct. Reports may be made confidentially to a conduct officer or a designated support contact. The university will investigate complaints in a fair and timely manner."},
        {"page": 15, "year": "2025", "audience": "all", "section": "housing", "title": "Housing policy", "text": "On-campus housing assignments are processed after admissions decisions and tuition deposit confirmation. Students may request room changes if there is a conflict with roommates or a medical accommodation need. University housing staff review requests within fourteen calendar days. Students must comply with health and safety inspections before moving in."},
        {"page": 16, "year": "2025", "audience": "all", "section": "housing", "title": "Meal plan choices", "text": "Meal plans can be selected during registration or during the add-drop period. Students on graduate assistantships may switch to a reduced meal plan if their agreement permits. The office of dining services will offer temporary substitutions only for documented medical needs. Meal plan rates are fixed for the current academic year."},
        {"page": 17, "year": "2025", "audience": "undergraduate", "section": "curriculum", "title": "Credit requirements", "text": "Undergraduate students must complete 120 credits to receive a bachelor degree unless a program requires more. Each degree program lists required major courses, general education credits, and electives. Students must complete prerequisite requirements before taking advanced courses, and placement evaluations may alter their first semester schedule."},
        {"page": 18, "year": "2025", "audience": "postgraduate", "section": "curriculum", "title": "Graduate milestones", "text": "Graduate programs include coursework, qualifying examinations, research progress reviews, and dissertation or thesis submission. Students should consult the program handbook for precise milestones and deadlines. A delayed milestone may be approved only with a faculty petition and stated justification. Students must maintain continuous registration unless granted a leave of absence."},
        {"page": 19, "year": "2025", "audience": "all", "section": "assessment", "title": "Assessment schedule", "text": "The semester calendar includes midterm assessments, final examinations, and project deadlines. Students are notified by email and LMS alerts at least two weeks before major assessments. Students who miss a final exam without an approved reason may be assigned an alternate assessment or a zero grade. Faculty must publish grading criteria before the assignment due date."},
        {"page": 20, "year": "2025", "audience": "all", "section": "assessment", "title": "Grading policies", "text": "Course grades are based on the rubric published in the syllabus. Students may request a review of marks if they believe a clerical or calculation error occurred. The formal grade appeal window is ten business days after the final grade is released. All appeals must be supported by evidence from the syllabus and assignment materials."},
        {"page": 21, "year": "2025", "audience": "all", "section": "technology", "title": "Digital account access", "text": "Every student receives a campus account to access email, LMS, and cloud storage. The university requires password rotation and secure two-factor authentication for privileged systems. Students who share credentials risk disciplinary action under the IT acceptable use policy. Requests for account reactivation must be approved by the IT help desk."},
        {"page": 22, "year": "2025", "audience": "all", "section": "technology", "title": "Software licensing", "text": "Students can install university-approved software through the campus software portal. Licensed tools may be used for coursework and personal study only, and not for commercial work. Software support may be unavailable on personal devices without the required system requirements. Students should report licensing problems to the support center before deadlines."},
        {"page": 23, "year": "2025", "audience": "all", "section": "transport", "title": "Transit and parking", "text": "Students may use campus shuttles and public transit passes according to semester registration. Parking permits are assigned by vehicle type and residence status. Students who park without a permit may receive a warning or a fine based on the violation schedule. Appeals are considered by the campus operations office within thirty days."},
        {"page": 24, "year": "2025", "audience": "all", "section": "transport", "title": "Bike and scooter rules", "text": "Bicycles and scooters must be parked in designated racks and may not block sidewalks or exits. Students are expected to wear helmets when using scooters on campus roads. Safety officers can remove abandoned bikes after a documented warning period. E-scooters must be charged and checked before leaving residence halls."},
        {"page": 25, "year": "2025", "audience": "all", "section": "support", "title": "Accessibility and accommodations", "text": "The accessibility office coordinates reasonable accommodations for students with temporary, long-term, or situational needs. Students can request reasonable adjustments for exams, lectures, and digital materials. Requests are reviewed with the relevant faculty member and documentation from a healthcare professional when required. Students should apply before each semester if their needs change."},
        {"page": 26, "year": "2025", "audience": "all", "section": "support", "title": "Counseling and crisis support", "text": "Confidential counseling is available for personal, academic, and family-related stress. Students can request short-term support or referral to a specialist outside the university. For urgent mental health concerns, students should contact the crisis line immediately and notify the on-call advisor. The support team can coordinate care with faculty on a need-to-know basis."},
        {"page": 27, "year": "2025", "audience": "all", "section": "health", "title": "Vaccination and screening", "text": "Students must submit required vaccination records before the term begins. The campus health center may request additional screening when there is a public health notice. Students who are non-compliant may be restricted from shared facilities until records are complete. Health records are stored securely and used only for compliance checks."},
        {"page": 28, "year": "2025", "audience": "all", "section": "health", "title": "Medical leave", "text": "Students who are medically unwell may request leave for a defined period after submitting documentation. Faculty may grant deadline extensions with formal approval from the dean of students. Students returning from medical leave must complete a re-entry checklist before joining classes. The checklist includes health clearance and your advising meeting summary."},
        {"page": 29, "year": "2025", "audience": "all", "section": "community", "title": "Volunteering and service", "text": "Students may participate in volunteer programs through the campus service office. Service hours can strengthen leadership and community engagement records. Students working with minors or vulnerable groups must complete screening and supervisor training before placement. These programs are run under the community partnerships office."},
        {"page": 30, "year": "2025", "audience": "all", "section": "community", "title": "Student clubs and societies", "text": "The campus supports student-led clubs, cultural groups, and academic societies. Clubs must register with the student activities office and maintain a faculty advisor. Student organizations may request rooms and funding based on a yearly review cycle. Students joining clubs are encouraged to review the risk and conduct policies before participation."},
        {"page": 31, "year": "2025", "audience": "all", "section": "graduation", "title": "Graduation requirements", "text": "Students must complete all required coursework, satisfy the minimum GPA, and submit an application for graduation. Students who fail to apply by the deadline may graduate in the next available term. The registrar confirms all requirements before awarding the degree. Students can track progress through the student portal and faculty advising reports."},
        {"page": 32, "year": "2025", "audience": "all", "section": "graduation", "title": "Commencement and transcripts", "text": "Graduates may attend commencement if they have completed all requirements or are on track to finish in the next semester. Official transcripts are issued through the registrar after the degree is posted. Students may request replacement copies if documents are lost or damaged. Fraudulent requests are reviewed by the registrar and campus compliance office."},
    ]

    content = []
    for item in handbook:
        content.append(f"[PAGE {item['page']}]\nTitle: {item['title']}\nSection: {item['section']}\nAudience: {item['audience']}\nYear: {item['year']}\n\n{item['text']}")
    corpus_path.write_text("\n\n".join(content), encoding="utf-8")

    manifest = {
        "source_id": "student_handbook_2025",
        "title": "Student Handbook 2025",
        "source_path_or_url": str(corpus_path),
        "license_or_permission": "Generated for classroom use; not a third-party public document.",
        "accessed_at": "2026-11-14",
        "pages": [
            {"document_id": "student_handbook_2025", "page": item["page"], "title": item["title"], "section": item["section"], "audience": item["audience"], "year": item["year"]}
            for item in handbook
        ],
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return corpus_path, manifest


def _generate_benchmark() -> list[dict]:
    benchmark_path = BENCHMARK_DIR / "questions.json"
    if benchmark_path.exists():
        return json.loads(benchmark_path.read_text())

    questions = [
        {"id": "Q01", "split": "dev", "question": "What is the required attendance threshold for first-year workshops?", "expected_answer": "80 percent attendance is required for first-year workshops.", "gold_source_id": "student_handbook_2025", "gold_page": 2, "evidence_quote": "The required attendance threshold is 80 percent for all first-year workshops.", "filter": {"section": "orientation", "audience": "all"}},
        {"id": "Q02", "split": "dev", "question": "When should undergraduate students meet an advisor before registration?", "expected_answer": "Before registration opens each semester.", "gold_source_id": "student_handbook_2025", "gold_page": 3, "evidence_quote": "Undergraduate students should meet with an advisor every semester before registration opens.", "filter": {"section": "advising", "audience": "undergraduate"}},
        {"id": "Q03", "split": "dev", "question": "What GPA is required to renew a merit scholarship?", "expected_answer": "Students must maintain a cumulative GPA of 3.5 or higher.", "gold_source_id": "student_handbook_2025", "gold_page": 7, "evidence_quote": "Students must maintain a cumulative GPA of 3.5 or higher to renew funding.", "filter": {"section": "scholarships", "audience": "undergraduate"}},
        {"id": "Q04", "split": "dev", "question": "Which office reviews scholarship appeals?", "expected_answer": "The financial aid office reviews scholarship appeals.", "gold_source_id": "student_handbook_2025", "gold_page": 7, "evidence_quote": "Students who fail to meet the GPA threshold may appeal through the financial aid office.", "filter": {"section": "scholarships", "audience": "undergraduate"}},
        {"id": "Q05", "split": "dev", "question": "Which safety equipment must students wear when handling chemicals?", "expected_answer": "Eye protection and gloves.", "gold_source_id": "student_handbook_2025", "gold_page": 11, "evidence_quote": "Students working with chemicals must wear eye protection and gloves at all times.", "filter": {"section": "lab", "audience": "all"}},
        {"id": "Q06", "split": "dev", "question": "What must students do with biohazard waste?", "expected_answer": "Use designated bins and never mix it with standard trash.", "gold_source_id": "student_handbook_2025", "gold_page": 12, "evidence_quote": "Biohazard waste must be disposed of in designated bins and never mixed with standard trash.", "filter": {"section": "lab", "audience": "all"}},
        {"id": "Q07", "split": "dev", "question": "What advice does the handbook give about plagiarism?", "expected_answer": "Students must cite all sources and cannot share answers directly with peers.", "gold_source_id": "student_handbook_2025", "gold_page": 13, "evidence_quote": "Academic misconduct includes plagiarism, unauthorized collaboration, and misleading citations.", "filter": {"section": "discipline", "audience": "undergraduate"}},
        {"id": "Q08", "split": "dev", "question": "How long do students have to appeal a grade after a final grade is released?", "expected_answer": "Ten business days.", "gold_source_id": "student_handbook_2025", "gold_page": 20, "evidence_quote": "The formal grade appeal window is ten business days after the final grade is released.", "filter": {"section": "assessment", "audience": "all"}},
        {"id": "Q09", "split": "dev", "question": "What permission is needed to install university software?", "expected_answer": "Students can install university-approved software through the campus software portal.", "gold_source_id": "student_handbook_2025", "gold_page": 22, "evidence_quote": "Students can install university-approved software through the campus software portal.", "filter": {"section": "technology", "audience": "all"}},
        {"id": "Q10", "split": "dev", "question": "What are the rules for parking permits on campus?", "expected_answer": "Parking permits are assigned by vehicle type and residence status.", "gold_source_id": "student_handbook_2025", "gold_page": 23, "evidence_quote": "Parking permits are assigned by vehicle type and residence status.", "filter": {"section": "transport", "audience": "all"}},
        {"id": "Q11", "split": "dev", "question": "Who coordinates accommodations for students with temporary or long-term needs?", "expected_answer": "The accessibility office coordinates reasonable accommodations.", "gold_source_id": "student_handbook_2025", "gold_page": 25, "evidence_quote": "The accessibility office coordinates reasonable accommodations for students with temporary, long-term, or situational needs.", "filter": {"section": "support", "audience": "all"}},
        {"id": "Q12", "split": "dev", "question": "What should students do if they are worried about a mental-health emergency?", "expected_answer": "Contact the crisis line immediately and notify the on-call advisor.", "gold_source_id": "student_handbook_2025", "gold_page": 26, "evidence_quote": "For urgent mental health concerns, students should contact the crisis line immediately and notify the on-call advisor.", "filter": {"section": "support", "audience": "all"}},
        {"id": "Q13", "split": "dev", "question": "What does the campus health center require before the term begins?", "expected_answer": "Students must submit required vaccination records before the term begins.", "gold_source_id": "student_handbook_2025", "gold_page": 27, "evidence_quote": "Students must submit required vaccination records before the term begins.", "filter": {"section": "health", "audience": "all"}},
        {"id": "Q14", "split": "dev", "question": "What steps are needed before joining classes after medical leave?", "expected_answer": "Complete a re-entry checklist and health clearance with the advising summary.", "gold_source_id": "student_handbook_2025", "gold_page": 28, "evidence_quote": "Students returning from medical leave must complete a re-entry checklist before joining classes.", "filter": {"section": "health", "audience": "all"}},
        {"id": "Q15", "split": "dev", "question": "How do students register for student clubs?", "expected_answer": "Clubs must register with the student activities office and maintain a faculty advisor.", "gold_source_id": "student_handbook_2025", "gold_page": 30, "evidence_quote": "Clubs must register with the student activities office and maintain a faculty advisor.", "filter": {"section": "community", "audience": "all"}},
        {"id": "Q16", "split": "test", "question": "What are the graduation requirements listed in the handbook?", "expected_answer": "Complete all coursework, satisfy the minimum GPA, and submit an application for graduation.", "gold_source_id": "student_handbook_2025", "gold_page": 31, "evidence_quote": "Students must complete all required coursework, satisfy the minimum GPA, and submit an application for graduation.", "filter": {"section": "graduation", "audience": "all"}},
        {"id": "Q17", "split": "test", "question": "What happens if a student misses a final exam without approval?", "expected_answer": "They may be assigned an alternate assessment or a zero grade.", "gold_source_id": "student_handbook_2025", "gold_page": 19, "evidence_quote": "Students who miss a final exam without an approved reason may be assigned an alternate assessment or a zero grade.", "filter": {"section": "assessment", "audience": "all"}},
        {"id": "Q18", "split": "test", "question": "What is the total credit requirement for a bachelor degree?", "expected_answer": "120 credits unless a program requires more.", "gold_source_id": "student_handbook_2025", "gold_page": 17, "evidence_quote": "Undergraduate students must complete 120 credits to receive a bachelor degree unless a program requires more.", "filter": {"section": "curriculum", "audience": "undergraduate"}},
        {"id": "Q19", "split": "test", "question": "Which office issues official transcripts after the degree is posted?", "expected_answer": "The registrar issues official transcripts.", "gold_source_id": "student_handbook_2025", "gold_page": 32, "evidence_quote": "Official transcripts are issued through the registrar after the degree is posted.", "filter": {"section": "graduation", "audience": "all"}},
        {"id": "Q20", "split": "test", "question": "What should students do before using the digital archive off campus?", "expected_answer": "Use multi-factor authentication through the university portal.", "gold_source_id": "student_handbook_2025", "gold_page": 10, "evidence_quote": "Students using the archive off campus must use multi-factor authentication using the university portal.", "filter": {"section": "library", "audience": "all"}},
    ]
    benchmark_path.write_text(json.dumps(questions, indent=2), encoding="utf-8")
    return questions


def _run_pipeline() -> None:
    corpus_path, _ = _generate_demo_corpus()
    questions = _generate_benchmark()
    pages = load_pdf(corpus_path, "student_handbook_2025", "student_handbook_2025")
    if len(pages) < 30:
        raise ValueError("The generated corpus does not meet the minimum 30 pages required by the lab")

    load_dotenv(ROOT / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL not found. Copy starter/.env.example to starter/.env")

    embedder = MyEmbedder()
    store = PgVectorStore(database_url=database_url, dimensions=embedder.dimensions)
    store.create_schema()

    strategies = {
        "A": lambda page: fixed_size(page, size=500, overlap=80),
        "B": lambda page: by_paragraph(page, max_size=600),
    }

    per_strategy: dict[str, list[dict]] = {}
    for name, strategy_fn in strategies.items():
        chunks = []
        for page in pages:
            chunks.extend(strategy_fn(page))
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        store.upsert(name, chunks, vectors)
        per_strategy[name] = [{
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "source": chunk.source,
            "page": chunk.page,
            "metadata": chunk.metadata,
            "text": chunk.text,
        } for chunk in chunks]

    strategy_c_chunks = []
    for page in pages:
        strategy_c_chunks.extend(by_paragraph(page, max_size=600))
    strategy_c_vectors = embedder.embed_documents([chunk.text for chunk in strategy_c_chunks])
    store.upsert("C", strategy_c_chunks, strategy_c_vectors)

    traces: list[dict] = []
    for question in questions:
        filters = question.get("filter")
        for strategy_name in ["A", "B", "C"]:
            if strategy_name == "C" and not filters:
                continue
            strategy_chunks = store.search(
                strategy_name,
                query_vector=embedder.embed_query(question["question"]),
                top_k=3,
                filters=filters,
            )
            evaluation = evaluate_query({**question, "strategy": strategy_name}, strategy_chunks)
            evaluation["strategy"] = strategy_name
            traces.append(evaluation)

    summary = summarize(traces)
    report = {
        "corpus_path": str(corpus_path),
        "questions": len(questions),
        "strategies": {k: {"chunk_count": len(per_strategy.get(k, [])) if k in per_strategy else len(strategy_c_chunks)} for k in ["A", "B", "C"]},
        "embedding": {"provider": "openai" if embedder.model_name.startswith("text-embedding-") else "sentence-transformers",
                      "model": embedder.model_name, "dimensions": embedder.dimensions},
        "summary": summary,
    }
    results_path = RESULTS_DIR / "summary.json"
    results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = _run_pipeline()
    print(json.dumps(report["summary"], indent=2))
    print(f"Benchmark questions: {report['questions']}")
    print(f"Corpus path: {report['corpus_path']}")


if __name__ == "__main__":
    main()
