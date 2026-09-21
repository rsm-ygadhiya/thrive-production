"""The one matcher: what a field asks for, against what this catalog teaches.

Both web-backed routes ran their own copy of this and the copies disagreed.
These cover the engine that replaced them — and specifically the four
properties the old ones did not have.
"""

import pytest

from rsm_thrive.services.courses import skill_match
from rsm_thrive.services.courses.electives import load_catalog


def course(code, **fields):
    base = {"code": code, "id": code, "title": code, "units": 4,
            "is_core": False, "skills": [], "topics": [], "tools": [],
            "description": ""}
    base.update(fields)
    return base


def codes(fits):
    return [fit.course["code"] for fit in fits]


# ---------------------------------------------------------------------------
# 1. A phrase matches a phrase
# ---------------------------------------------------------------------------

class TestARequirementMatchesOnePhrase:
    """`advisor._requirement_matches` required every token of the requirement
    to appear ANYWHERE in the course's text. "systems engineering" therefore
    matched a web-mining course, because "systems" is in "recommender systems"
    and "engineering" is in "data engineering" — two unrelated phrases."""

    SPLIT = course("MGTA 461", title="Web Mining and Recommender Systems",
                   skills=["data engineering pipelines"],
                   topics=["recommender systems"])

    def test_words_scattered_across_phrases_are_not_a_match(self):
        fits = skill_match.rank(["systems engineering", "recommender systems"],
                                catalog=[self.SPLIT])
        assert "systems engineering" not in skill_match.covered_by(fits)

    def test_the_phrase_that_is_really_there_still_matches(self):
        fits = skill_match.rank(["recommender systems", "data engineering"],
                                catalog=[self.SPLIT])
        assert skill_match.covered_by(fits) == {"recommender systems",
                                                "data engineering"}

    def test_half_a_two_word_requirement_is_not_a_match(self):
        """`role_lookup._overlap` accepted a match when every word longer than
        three characters was present, so "cpg analytics" matched on
        "analytics" alone and picked up four courses on that basis."""
        fits = skill_match.rank(
            ["cpg analytics", "sql querying"],
            catalog=[course("MGTA 1", title="Marketing Analytics",
                            skills=["sql querying", "customer analytics"])])
        assert "cpg analytics" not in skill_match.covered_by(fits)

    def test_most_of_a_longer_requirement_is_a_partial_match(self):
        """Tested at the phrase level: one match is not enough to be
        recommended (see `MIN_DISTINCT`), and what is under test here is how
        the match is graded, not whether the course is admitted."""
        requirement = "designing and running controlled experiments"
        tier, hits = skill_match._match_phrase(
            skill_match._words(requirement),
            skill_match._words("designing and analyzing experiments"))
        assert (tier, hits) == ("partial", 2)

    def test_a_partial_match_is_worth_less_than_a_whole_one(self):
        pool = [course("MGTA 1", skills=["designing and analyzing experiments"],
                       topics=["sql querying"]),
                course("MGTA 2", skills=["designing and running controlled experiments"],
                       topics=["sql querying"]),
                course("MGTA 3", skills=["poetry"])]
        fits = skill_match.rank(
            ["designing and running controlled experiments", "sql querying"],
            catalog=pool)
        assert codes(fits)[0] == "MGTA 2", "the exact phrase leads"


# ---------------------------------------------------------------------------
# 2. Words match on a stem, not on equality
# ---------------------------------------------------------------------------

class TestOrdinaryInflections:
    @pytest.mark.parametrize("left,right", [
        ("forecast", "forecasting"), ("design", "designing"),
        ("experiment", "experiments"), ("model", "modelling"),
        ("program", "programming"), ("statistics", "statistical"),
        ("optimisation", "optimization"), ("visualization", "visualisation"),
        ("risk", "risks"),
    ])
    def test_these_are_the_same_word(self, left, right):
        assert skill_match._same_word(left, right)
        assert skill_match._same_word(right, left), "and symmetrically"

    @pytest.mark.parametrize("left,right", [
        # The four false positives a bare shared-prefix rule produced on this
        # catalog. The first two were damaging: "tableau"/"table" credited the
        # SQL course with teaching Tableau, and "healthcare"/"health" made
        # Marketing Analytics a healthcare course.
        ("tableau", "table"), ("healthcare", "health"),
        ("communication", "community"), ("product", "produce"),
        ("data", "database"), ("sql", "sqlite"), ("analytics", "analysis"),
    ])
    def test_these_are_not(self, left, right):
        assert not skill_match._same_word(left, right)
        assert not skill_match._same_word(right, left)

    def test_a_short_word_must_match_exactly(self):
        """Measured on the real catalog, substring "r" appears in all 31
        courses and "ai" in 14 (chain, training, available)."""
        assert not skill_match._same_word("r", "reporting")
        assert not skill_match._same_word("ai", "chain")
        assert skill_match._same_word("sql", "sql")


# ---------------------------------------------------------------------------
# 3. A requirement everything satisfies is worth nothing
# ---------------------------------------------------------------------------

class TestDiscrimination:
    """The aerospace failure: every requirement was worth the same, so three
    of four results were carried by "python" — a word describing half this
    catalog, which tells a student nothing about which course to take."""

    def test_a_universal_requirement_weighs_zero(self):
        pool = [course(f"MGTA {n}", tools=["Python"]) for n in range(6)]
        assert skill_match.discriminations(["python"], catalog=pool)["python"] == 0.0

    def test_a_rare_requirement_weighs_most(self):
        pool = [course(f"MGTA {n}", tools=["Python"]) for n in range(6)]
        pool.append(course("MGTA 9", tools=["Python"],
                           skills=["reliability analysis"]))
        weights = skill_match.discriminations(
            ["python", "reliability analysis"], catalog=pool)
        assert weights["reliability analysis"] > weights["python"]

    def test_it_is_measured_on_the_real_catalog_too(self):
        """Not a synthetic property: on the shipped catalog "python" separates
        almost nothing and "cost effectiveness analysis" separates one course."""
        weights = skill_match.discriminations(
            ["python", "cost effectiveness analysis"])
        # Compared against the rare skill rather than against a fixed number.
        # The absolute value moves with the catalog -- it was under 1.0 across
        # 31 courses and is higher across ~90, because most of the courses
        # added teach no programming at all -- but the ORDER is the property
        # this is about, and it does not move.
        assert weights["python"] < weights["cost effectiveness analysis"]
        assert weights["cost effectiveness analysis"] > 2.0

    def test_the_ranking_follows_the_weights(self):
        pool = [course(f"MGTA {n}", tools=["Python"], skills=["general work"])
                for n in range(6)]
        pool.append(course("MGTA 9", title="Reliability Engineering",
                           tools=["Python"], skills=["reliability analysis"]))
        fits = skill_match.rank(["python", "reliability analysis"], catalog=pool)
        assert fits[0].course["code"] == "MGTA 9"

    def test_a_universal_match_still_counts_toward_admission(self):
        """It cannot help rank one course above another. It is still evidence
        the course teaches the thing, and skipping it made a course with two
        genuine matches look like a course with one."""
        pool = [course("MGTA 1", skills=["sql querying"], topics=["forecasting"]),
                course("MGTA 2", skills=["sql querying"])]
        fits = skill_match.rank(["sql querying", "forecasting"], catalog=pool)
        assert codes(fits) == ["MGTA 1"]


# ---------------------------------------------------------------------------
# 4. Tooling is not a recommendation
# ---------------------------------------------------------------------------

class TestContentBeatsTooling:
    """A healthcare requirement set admitted Marketing Analytics and Business
    Forecasting on "uses R" plus "uses Python" and nothing else — two real
    matches, neither of them a reason to take the course for healthcare."""

    def test_a_course_matched_only_on_software_is_dropped(self):
        fits = skill_match.rank(
            ["python", "r"],
            catalog=[course("MGTA 1", tools=["Python", "R"]),
                     course("MGTA 2", tools=["Python", "R"],
                            skills=["programming in python"])])
        assert codes(fits) == ["MGTA 2"]

    def test_a_course_about_the_field_is_exempt(self):
        fits = skill_match.rank(
            ["python"], domain="healthcare",
            catalog=[course("MGTA 495", title="Healthcare Analytics",
                            tools=["Python"]),
                     course("MGTA 1", tools=["Python"])])
        assert codes(fits) == ["MGTA 495"]

    def test_a_tie_records_the_content_match(self):
        """"excel" matches the skill "excel modelling" and the tool "Excel"
        equally well; which one is recorded decides whether the course counts
        as recommended for its content."""
        fits = skill_match.rank(
            ["excel", "python"],
            catalog=[course("MGTA 1", tools=["Excel", "Python"],
                            skills=["excel modelling"])])
        assert fits and any(match.field != "tools" for match in fits[0].matches)


# ---------------------------------------------------------------------------
# What is NOT covered
# ---------------------------------------------------------------------------

class TestCoverage:
    """Three courses with no caveat reads as "these will get you there"."""

    def test_it_names_what_nothing_teaches(self):
        pool = [course("MGTA 1", skills=["sql querying"], topics=["forecasting"])]
        wanted = ["sql querying", "forecasting", "reliability analysis",
                  "systems engineering"]
        cover = skill_match.coverage(
            wanted, skill_match.rank(wanted, catalog=pool), catalog=pool)
        assert set(cover["unmet"]) == {"reliability analysis", "systems engineering"}

    def test_the_share_is_weighted_not_counted(self):
        """Meeting "python" and missing "predictive maintenance" is not half a
        job done."""
        pool = [course(f"MGTA {n}", tools=["Python"]) for n in range(5)]
        pool.append(course("MGTA 9", tools=["Python"],
                           skills=["predictive maintenance"]))
        wanted = ["python", "reliability analysis"]
        cover = skill_match.coverage(
            wanted, skill_match.rank(wanted, catalog=pool), catalog=pool)
        assert cover["metShare"] == 0.0, "python is met and worth nothing"

    def test_a_thin_field_is_visibly_thin_on_the_real_catalog(self):
        """Aerospace, end to end. The catalog has the Excel, the forecasting
        and the visualisation, and nothing at all for what makes the field
        that field."""
        wanted = ["systems engineering", "reliability analysis",
                  "predictive maintenance", "statistical process control",
                  "aerospace manufacturing", "python", "excel", "tableau",
                  "forecasting", "data visualization"]
        fits = skill_match.rank(wanted, domain="aerospace")
        cover = skill_match.coverage(wanted, fits)
        assert cover["metShare"] < skill_match.THIN_COVERAGE
        assert "systems engineering" in cover["unmet"]
        assert "reliability analysis" in cover["unmet"]

    def test_a_well_served_role_is_visibly_well_served(self):
        wanted = ["experiment design", "a/b testing", "sql", "python",
                  "tableau", "forecasting", "recommendation systems",
                  "data storytelling"]
        fits = skill_match.rank(wanted)
        cover = skill_match.coverage(wanted, fits)
        assert cover["metShare"] > skill_match.THIN_COVERAGE
        assert fits, "and it recommends something"


# ---------------------------------------------------------------------------
# The guarantees the whole design rests on
# ---------------------------------------------------------------------------

class TestItCanOnlyEverNameARealCourse:
    def test_every_result_is_a_catalog_row(self):
        wanted = ["quantum cryptography", "sql", "python", "forecasting"]
        rows = {course["id"]: course for course in load_catalog()}
        for fit in skill_match.rank(wanted):
            assert rows[fit.course["id"]] is fit.course

    def test_every_quoted_phrase_is_the_catalogs_own_words(self):
        """A reason quotes the course, never the web. That is what makes it
        checkable against the course page."""
        wanted = ["demand forecasting", "sql", "experiment design"]
        for fit in skill_match.rank(wanted):
            haystack = " ".join(
                [fit.course["title"], fit.course.get("description") or ""]
                + (fit.course.get("skills") or [])
                + (fit.course.get("topics") or [])
                + (fit.course.get("tools") or []))
            for match in fit.matches:
                assert match.phrase in haystack, match.phrase

    def test_core_courses_are_not_recommended(self):
        """Everyone takes them; there is nothing to recommend."""
        fits = skill_match.rank(["sql", "python", "forecasting", "statistics"])
        assert not any(fit.course["is_core"] for fit in fits)

    def test_the_same_input_gives_the_same_ranking(self):
        wanted = ["sql", "python", "experiment design", "forecasting"]
        assert codes(skill_match.rank(wanted)) == codes(skill_match.rank(wanted))

    def test_nothing_relevant_returns_nothing(self):
        assert skill_match.rank(["viticulture", "oenology", "sommelier"]) == []
