"""Two gaps closed: listing the careers, and measuring the plan we shipped.

The coverage half is the subtler one. A recommendation quotes a figure computed
over the six courses it recommends; the PLAN is not those six courses, because
quarters have unit floors and ceilings and a course that does not fit is
dropped. So a student could be told "32% of what that career asks for" about a
set they were never shown.
"""

import pytest

from rsm_thrive.services.courses import electives, orchestrator, router
from rsm_thrive.services.llm import FakeLLM


class TestAskingWhichCareers:
    @pytest.mark.parametrize("question", [
        "what jobs do you have",
        "what jobs do you have?",
        "which careers do you cover",
        "what roles do you support",
        "list the careers",
        "what career paths are there",
        "what jobs are available",
    ])
    def test_it_is_ruled_without_a_model(self, question):
        route = router.rule_route(question)
        assert route is not None and route.name == router.CAREERS

    @pytest.mark.parametrize("question", [
        "what courses do you have for a data scientist",
        "what skills does a data scientist need",
        "i want to be a data scientist",
        "what jobs pay well",
        "does MGTA 452 have prerequisites",
    ])
    def test_a_question_about_one_job_is_not_the_list(self, question):
        route = router.rule_route(question)
        assert route is None or route.name != router.CAREERS

    def test_every_curated_career_is_named(self):
        reply = orchestrator._answer_careers(router.Route(router.CAREERS))
        for career in electives.load_careers().values():
            label = career.get("short_label") or career.get("label")
            assert f"**{label}**" in reply.body, f"{label} missing from the list"

    def test_the_count_matches_the_catalog(self):
        reply = orchestrator._answer_careers(router.Route(router.CAREERS))
        assert f"**{len(electives.load_careers())}**" in reply.body

    def test_it_says_an_unlisted_goal_is_still_worth_asking(self):
        """Fourteen bullets read as a menu; the uncurated path must be visible."""
        body = orchestrator._answer_careers(router.Route(router.CAREERS)).body
        assert "Not on the list?" in body

    def test_it_costs_no_model_call(self):
        llm = FakeLLM([])
        orchestrator._answer_careers(router.Route(router.CAREERS))
        assert llm.calls == []


class TestPlanCoverageForAnUncuratedCareer:
    SKILLS = ["sql", "dashboards", "forecasting", "sports performance analysis"]

    def _plan(self, courses):
        return {"quarters": [{"key": "fall", "courses": [
            {"courseId": cid} for cid in courses]}]}

    def _answers(self, placed, recommended):
        return {"unmatched_goal": "sports analytics manager",
                "goal_skills": self.SKILLS,
                "goal_courses": recommended}

    def _codes(self, n):
        return [c["id"] for c in electives.load_catalog()[:n]]

    def test_nothing_is_said_for_a_curated_goal(self):
        """Their bundle IS the answer; scoring it against a web page is wrong."""
        assert orchestrator._uncurated_coverage_note(
            {"goals": ["data-scientist"], "goal_skills": self.SKILLS},
            self._plan(self._codes(3))) == ""

    def test_nothing_is_said_without_skills(self):
        assert orchestrator._uncurated_coverage_note(
            {"unmatched_goal": "x"}, self._plan(self._codes(3))) == ""

    def test_nothing_is_said_without_a_plan(self):
        assert orchestrator._uncurated_coverage_note(
            self._answers([], []), None) == ""

    def test_it_reports_a_share_and_the_gap(self):
        codes = self._codes(4)
        note = orchestrator._uncurated_coverage_note(
            self._answers(codes, codes), self._plan(codes))
        assert "sports analytics manager" in note
        assert "of what that career asks for" in note
        assert "this plan covers about" in note

    def test_a_dropped_course_is_named(self):
        """The student saw it by name one turn ago. Losing it silently reads as
        the plan disagreeing with the recommendation for no stated reason."""
        codes = self._codes(4)
        note = orchestrator._uncurated_coverage_note(
            self._answers(codes[:2], codes), self._plan(codes[:2]))
        assert "didn't fit the unit limits" in note
        assert codes[2].split("-")[0] in note

    def test_nothing_is_said_about_drops_when_nothing_dropped(self):
        codes = self._codes(3)
        note = orchestrator._uncurated_coverage_note(
            self._answers(codes, codes), self._plan(codes))
        assert "didn't fit" not in note

    def test_the_share_is_measured_on_the_plan_not_the_recommendation(self):
        """The whole point: a smaller plan must not inherit the bigger set's
        coverage figure."""
        codes = self._codes(6)
        whole = orchestrator._uncurated_coverage_note(
            self._answers(codes, codes), self._plan(codes))
        part = orchestrator._uncurated_coverage_note(
            self._answers(codes[:1], codes), self._plan(codes[:1]))
        assert whole != part, "coverage did not change when the plan shrank"


class TestAskingWhatTheCatalogHolds:
    """A model handed 86 rows summarises them away.

    Asked "what classes do you have access to", it answered "86 courses, 6 core
    and 80 electives, all others are electives" and cited four courses at
    random. That is arithmetic the student could already see and none of the
    context they asked for, so the breakdown is deterministic.
    """

    @pytest.mark.parametrize("question", [
        "what courses do you have",
        "what classes do you have access to",
        "what electives are available",
        "what is in the catalog",
        "what courses can i take",
        "show me the catalog",
        "list all the courses",
        "what classes are there",
    ])
    def test_it_is_ruled_without_a_model(self, question):
        route = router.rule_route(question)
        assert route is not None and route.name == router.CATALOG

    @pytest.mark.parametrize("question", [
        "courses in finance",
        "courses about fraud",
        "what courses for a data scientist",
        "what electives help me work in healthcare",
        "does MGTA 452 have prerequisites",
        "what is MGTA 464 about",
        "what should i take",
        "what jobs do you have",
    ])
    def test_a_question_about_something_particular_is_not_the_list(self, question):
        route = router.rule_route(question)
        assert route is None or route.name != router.CATALOG

    def _body(self):
        return orchestrator._answer_catalog(router.Route(router.CATALOG)).body

    def test_every_programme_appears_with_its_terms(self):
        body = self._body()
        for programme in electives.departments_available():
            assert programme["name"] in body
            assert str(programme["total"]) in body

    def test_the_msba_is_listed_first_and_in_full(self):
        """The home programme is what a student chooses between most."""
        body = self._body()
        msba = [c for c in electives.load_catalog() if c["department"] == "MGTA"]
        assert body.index("Rady MSBA") < body.index("Rady MBA")
        for course in msba:
            assert course["title"] in body, course["id"]

    def test_the_core_is_marked_and_totalled(self):
        body = self._body()
        assert "22 units" in body
        for course in electives.load_catalog():
            if course["is_core"]:
                assert course["title"] in body

    def test_the_non_msba_cap_is_stated(self):
        assert f"{electives.NON_MSBA_UNIT_CAP} of your 28" in self._body()

    def test_no_title_is_shouted(self):
        """Two syllabi set their header in capitals, which read as shouting in
        a list of eighty."""
        for course in electives.load_catalog():
            letters = [c for c in course["title"] if c.isalpha()]
            assert any(c.islower() for c in letters), course["title"]

    def test_the_programme_note_keeps_its_acronym(self):
        """`str.capitalize` lowercases the tail, giving "pre-approved msba"."""
        assert "msba electives" not in self._body()

    def test_it_costs_no_model_call(self):
        llm = FakeLLM([])
        orchestrator._answer_catalog(router.Route(router.CATALOG))
        assert llm.calls == []
