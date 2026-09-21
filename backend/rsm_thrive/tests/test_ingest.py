from pathlib import Path

import pytest

from rsm_thrive.models import Document, DocumentChunk
from rsm_thrive.services.embeddings import FakeEmbeddings
from rsm_thrive.services.resources.ingest import chunk_text, ingest_document

pytestmark = pytest.mark.django_db

FIXTURES = Path(__file__).parent / "fixtures" / "corpus"


class TestChunkText:
    def test_splits_on_headings(self):
        text = FIXTURES.joinpath("handbook-excerpt.md").read_text()
        chunks = chunk_text(text)
        assert [c["heading"] for c in chunks] == ["Dropping a course", "Laptop loans"]
        assert "week two" in chunks[0]["text"]

    def test_long_section_splits_with_overlap(self):
        paragraphs = "\n\n".join(f"Paragraph {i} " + "x" * 300 for i in range(10))
        chunks = chunk_text("# Big\n\n" + paragraphs, max_chars=800, overlap=100)
        assert len(chunks) > 1
        assert all(c["heading"] == "Big" for c in chunks)
        assert all(len(c["text"]) <= 800 for c in chunks)
        # the tail of one chunk reappears at the head of the next
        assert chunks[1]["text"][:50] in chunks[0]["text"]

    def test_preamble_before_first_heading_is_kept(self):
        chunks = chunk_text("intro line\n\n# H\n\nbody")
        assert chunks[0]["heading"] == "" and "intro" in chunks[0]["text"]

    def test_single_oversized_paragraph_is_hard_wrapped(self):
        # A single paragraph with no blank lines, longer than max_chars
        para = "x" * 4000  # 4000 chars, well over default max_chars=1400
        chunks = chunk_text("# Section\n\n" + para)
        assert len(chunks) > 1
        assert all(len(c["text"]) <= 1400 for c in chunks)
        assert all(c["heading"] == "Section" for c in chunks)
        # Verify no text is truncated: concatenation includes the final 50 chars
        full_text = "".join(c["text"] for c in chunks)
        assert full_text.endswith(para[-50:])


class TestIngestDocument:
    def test_creates_chunks_with_embeddings(self):
        text = FIXTURES.joinpath("handbook-excerpt.md").read_text()
        doc = ingest_document("test:handbook", "Handbook", "policy",
                              ["resources"], text, FakeEmbeddings())
        chunks = list(doc.chunks.all())
        assert len(chunks) == 2
        assert all(len(c.embedding) > 0 for c in chunks)
        assert chunks[0].seq == 0

    def test_rerun_replaces_not_duplicates(self):
        emb = FakeEmbeddings()
        ingest_document("test:h", "H", "policy", ["resources"], "# A\n\nfirst", emb)
        ingest_document("test:h", "H", "policy", ["resources"], "# B\n\nsecond", emb)
        assert Document.objects.filter(source="test:h").count() == 1
        assert DocumentChunk.objects.count() == 1
        assert DocumentChunk.objects.get().heading == "B"


class TestDestinationScoping:
    """Which bots can see a document is derived from its source host.

    The bug this guards: every crawled page was scoped to "resources" alone, so
    the career destination had an EMPTY corpus while 62 career.ucsd.edu and
    career.rady.ucsd.edu documents sat in the database. The career bot answered
    from the model's own knowledge and said so — "I don't have specific numbered
    context passages to cite" — about a real programme's real services.
    """

    def test_a_career_page_is_reachable_by_the_career_bot(self):
        from rsm_thrive.management.commands.ingest_corpus import destinations_for

        assert destinations_for(
            "policy", "https://career.ucsd.edu/guides/resume.pdf") == [
                "resources", "career"]
        assert destinations_for(
            "policy", "https://career.rady.ucsd.edu/resources/casecoach/") == [
                "resources", "career"]

    def test_a_career_page_stays_reachable_by_the_faq_bot(self):
        """Additive, not a move: "how do I write a resume" is a fair FAQ question."""
        from rsm_thrive.management.commands.ingest_corpus import destinations_for

        assert "resources" in destinations_for(
            "policy", "https://career.ucsd.edu/x")

    def test_an_ordinary_page_is_not_given_the_career_destination(self):
        from rsm_thrive.management.commands.ingest_corpus import destinations_for

        assert destinations_for(
            "policy", "https://students.ucsd.edu/academics/enroll/") == ["resources"]

    def test_a_syllabus_reaches_the_courses_bot_and_only_that(self):
        """It used to be ["resources", "courses"]. Full syllabus text is more
        chunks than the whole FAQ corpus and is dense with the words a student
        uses to ask about enrolment, so in "resources" it buried the pages that
        answer them -- see `TestSyllabiStayOutOfTheFaqCorpus` in
        test_catalog_build.py. The FAQ bot keeps a short `catalog` entry per
        course instead.
        """
        from rsm_thrive.management.commands.ingest_corpus import destinations_for

        assert destinations_for("syllabus", "") == ["courses"]

    def test_a_lookalike_host_is_not_matched(self):
        """Substring matching would catch "careers.example.com"; this is exact."""
        from rsm_thrive.management.commands.ingest_corpus import destinations_for

        assert destinations_for(
            "policy", "https://notcareer.ucsd.edu/x") == ["resources"]
        assert destinations_for(
            "policy", "https://career.ucsd.edu.evil.test/x") == ["resources"]
