"""courses.json is generated from corpus/syllabi/*.md and must not drift.

The markdown is the source a person edits; the JSON is what the planner counts
with. Two files holding the same facts is a standing invitation for them to
disagree, so the disagreement is a test failure rather than a surprise in a
plan six weeks later.
"""
import json
import pathlib
import re
from pathlib import Path

import pytest

from rsm_thrive.management.commands.build_catalog import (
    SYLLABI, build, frontmatter)
from rsm_thrive.services.courses import electives

CATALOG = Path(electives.__file__).resolve().parents[2] / "data/catalog/courses.json"


def markdown_files():
    return [p for p in sorted(SYLLABI.glob("*.md"))
            if not re.match(r".* \d+$", p.stem)]


class TestTheCatalogIsBuiltFromTheMarkdown:
    def test_it_is_up_to_date(self):
        """Run `python manage.py build_catalog` if this fails."""
        rendered = json.dumps(build(), indent=2, ensure_ascii=False) + "\n"
        assert CATALOG.read_text() == rendered, (
            "courses.json is out of date with corpus/syllabi/*.md")

    def test_every_catalog_course_has_a_markdown_file(self):
        ids = {frontmatter(p.read_text()).get("id") for p in markdown_files()}
        missing = [c["id"] for c in electives.load_catalog() if c["id"] not in ids]
        assert not missing, f"no markdown source for {missing}"

    def test_ids_are_unique(self):
        """Four courses share the code MGTA 495 and are different courses."""
        ids = [frontmatter(p.read_text()).get("id") for p in markdown_files()]
        assert len(ids) == len(set(ids))

    def test_shared_codes_survive(self):
        by_code = {}
        for course in electives.load_catalog():
            by_code.setdefault(course["code"], []).append(course["id"])
        assert len(by_code["MGTA 495"]) == 4, "special-topics variants collapsed"


class TestWhatIsSchedulable:
    def test_nothing_is_scheduled_without_a_unit_count(self):
        """The grounding rule applied to ourselves: a course whose unit count
        appears in none of the source material stays in the corpus, where it
        can be described, and out of the catalog, where it would be counted."""
        for path in markdown_files():
            meta = frontmatter(path.read_text())
            if not meta.get("units"):
                assert not meta.get("schedulable"), path.name

    def test_every_schedulable_course_has_units(self):
        for course in electives.load_catalog():
            assert isinstance(course["units"], int) and course["units"] > 0, \
                course["id"]

    def test_the_msba_core_matches_the_plan_of_study(self):
        """The 2026/2027 Plan of Study: six core courses totalling 22 units.

        MGTA 403 is an ELECTIVE there, and was flagged core here -- which made
        the bot tell students the core was seven courses and 24 units.
        """
        core = [c for c in electives.load_catalog() if c["is_core"]]
        assert sorted(c["code"] for c in core) == [
            "MGTA 444", "MGTA 451", "MGTA 452", "MGTA 453", "MGTA 454",
            "MGTA 455"]
        assert sum(c["units"] for c in core) == 22


class TestTheMarkdownIsReadable:
    @pytest.mark.parametrize("field", ["id", "code", "title", "department"])
    def test_every_file_carries_the_basics(self, field):
        for path in markdown_files():
            assert frontmatter(path.read_text()).get(field), f"{path.name}: {field}"

    def test_frontmatter_values_are_valid_json(self):
        """The flow style is the contract -- it is what lets `build_catalog`
        parse the frontmatter without a YAML dependency."""
        for path in markdown_files():
            block = re.match(r"\A---\n(.*?)\n---\n", path.read_text(), re.S)
            assert block, path.name
            for line in block.group(1).splitlines():
                key, _, raw = line.partition(":")
                json.loads(raw.strip())

    def test_the_body_carries_the_syllabus_for_courses_that_have_one(self):
        with_text = [p for p in markdown_files()
                     if frontmatter(p.read_text()).get("syllabi")]
        assert len(with_text) > 100
        for path in with_text:
            assert "## Syllabus text" in path.read_text(), path.name


class TestSyllabiStayOutOfTheFaqCorpus:
    """A syllabus reaches the courses bot alone.

    Full syllabus text is more chunks than the entire FAQ corpus and is dense
    with the words a student uses to ask about enrolment -- every syllabus says
    "class", most say "book". With them in "resources", "how to book a class"
    returned MGT 408 Finance and pushed the actual booking page off the results,
    and the bot answered that it had no steps for booking a class while holding
    the page that has them.
    """

    def test_a_syllabus_is_courses_only(self):
        from rsm_thrive.management.commands.ingest_corpus import destinations_for

        assert destinations_for("syllabus", "") == ["courses"]
        assert destinations_for("syllabus", "https://rady.ucsd.edu/x") == ["courses"]

    def test_everything_else_still_reaches_the_faq_bot(self):
        from rsm_thrive.management.commands.ingest_corpus import destinations_for

        assert destinations_for("policy", "") == ["resources"]
        assert "resources" in destinations_for("policy", "https://career.ucsd.edu/x")

    def test_the_career_destination_is_unaffected(self):
        from rsm_thrive.management.commands.ingest_corpus import destinations_for

        assert destinations_for("policy", "https://career.ucsd.edu/x") == [
            "resources", "career"]

    def test_the_faq_bot_keeps_a_course_summary(self):
        """It loses the syllabi, not course coverage: the `catalog` documents
        are one short entry per course and stay in "resources"."""
        from rsm_thrive.management.commands.ingest_corpus import destinations_for

        assert "resources" in destinations_for("catalog", "")

    def test_rescope_does_not_skip_a_syllabus_without_a_url(self):
        """The guard that protects fixture documents skipped the markdown
        syllabi, which carry no source URL -- so the rescope that was meant to
        fix this reported "0 documents" and changed nothing."""
        source = pathlib.Path(
            electives.__file__).resolve().parents[2] / \
            "management/commands/ingest_corpus.py"
        text = source.read_text()
        assert 'document.kind != "syllabus"' in text, (
            "rescope must not skip syllabi for lacking a source_url")
