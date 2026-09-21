"""What survives the interview's removal.

The four-step interview is gone (see `services/orchestrator.py`), and with it
the tests that drove it. What is here is the part that was never about the
interview: the planner's own arithmetic and the small deterministic reads the
orchestrator still relies on.

The routing, the plan and everything a student does with one are covered in
`test_orchestrator.py`.
"""

import pytest
from django.contrib.auth.models import User

from rsm_thrive.models import Conversation
from rsm_thrive.services.courses import planner

pytestmark = pytest.mark.django_db

# A complete set of answers, as the orchestrator assembles one: a track and a
# goal read straight off what the student typed, everything else assumed.
FULL = {
    "track": "11 month", "goals": ["data-scientist"],
    "skill_python": "comfortable", "skill_sql": "basic",
    "skill_stats": "comfortable", "skill_ml": "basic",
    "skill_communication": "basic", "workload": "moderate",
    "interests": ["machine-learning"],
}


@pytest.fixture
def user():
    return User.objects.create_user("stu")


@pytest.fixture
def conversation(user):
    return Conversation.objects.create(user=user, destination="courses",
                                       title="planning")


class TestAmbiguousCourseCodes:
    """The catalog has four Special Topics courses all coded MGTA 495."""

    def test_a_shared_code_is_disambiguated_in_the_plan(self, conversation):
        from rsm_thrive.services.courses import planner

        plan = planner.build_plan({**FULL, "goals": ["consultant"]})
        codes = [r["code"] for q in plan["quarters"] for r in q["courses"]
                 if r["code"]]
        assert len(codes) == len(set(codes)), codes

    def test_the_bare_code_is_still_available_for_matching(self, conversation):
        from rsm_thrive.services.courses import planner

        plan = planner.build_plan(FULL)
        for quarter in plan["quarters"]:
            for row in quarter["courses"]:
                if row["courseId"]:
                    assert row["baseCode"]

    def test_a_disambiguated_code_can_still_be_located(self, conversation):
        from rsm_thrive.services.courses import planner

        plan = planner.build_plan({**FULL, "goals": ["consultant"]})
        target = next((r for q in plan["quarters"] for r in q["courses"]
                       if r["code"] and "(" in r["code"]), None)
        if target is None:
            pytest.skip("this profile planned no ambiguous-coded course")
        assert planner.locate_code(plan, target["code"])[2] is not None
        assert planner.locate_code(plan, target["baseCode"])[2] is not None


class TestTargetingNote:
    """When a plan comes out mostly breadth, it says so — judged from the plan
    rather than from a catalog tag count that did not predict the outcome."""

    def test_a_mostly_breadth_plan_is_flagged(self):
        from rsm_thrive.services.courses import planner

        plan = planner.build_plan({**FULL, "goals": ["operations-supply-chain"]})
        note = plan["targeting"]
        assert note and note["onGoal"] < note["electiveSlots"]

    def test_a_well_targeted_plan_is_not_flagged(self):
        from rsm_thrive.services.courses import planner

        assert planner.build_plan({**FULL, "goals": ["data-scientist"]})["targeting"] is None

    def _on_goal(self, role):
        from rsm_thrive.services.courses import planner

        plan = planner.build_plan({**FULL, "goals": [role]})
        rows = [r for q in plan["quarters"] for r in q["courses"]
                if r["requirement"] == "Elective" and r["swappable"]]
        return sum(1 for r in rows if r.get("onGoal")), len(rows)

    def test_the_note_reports_the_plans_own_numbers(self):
        """Exact rather than derived from a proxy — this is the whole change."""
        from rsm_thrive.services.courses import planner

        plan = planner.build_plan({**FULL, "goals": ["operations-supply-chain"]})
        rows = [r for q in plan["quarters"] for r in q["courses"]
                if r["requirement"] == "Elective" and r["swappable"]]
        note = plan["targeting"]
        assert note["onGoal"] == sum(1 for r in rows if r.get("onGoal"))
        assert note["electiveSlots"] == len(rows)

    def test_more_tagged_electives_does_not_mean_a_more_targeted_plan(self):
        """The finding that retired the old threshold: the catalog count does
        not order the outcomes, so it cannot decide which roles are offered.

        Asserted as "such a pair exists" rather than naming two roles, because
        which pair it is moves with the student's skill levels — the fragility
        that made the first version of this test wrong.
        """
        from rsm_thrive.services.courses import planner
        from rsm_thrive.services.courses.electives import load_careers

        roles = sorted(load_careers())
        pairs = [
            (a, b) for a in roles for b in roles
            if planner.matched_elective_count(a) > planner.matched_elective_count(b)
            and self._on_goal(a)[0] <= self._on_goal(b)[0]
        ]
        assert pairs, "catalog count would have been a fair proxy after all"

    def test_the_note_names_advising_and_the_appointments_tab(self):
        from rsm_thrive.services.courses import planner

        text = planner.render_plan_markdown(
            planner.build_plan({**FULL, "goals": ["operations-supply-chain"]}))
        assert "Appointments" in text and "advising" in text.lower()

    def test_a_flagged_plan_is_still_complete(self):
        from rsm_thrive.services.courses import planner

        plan = planner.build_plan({**FULL, "goals": ["operations-supply-chain"]})
        assert plan["totals"]["total"] == 50 and plan["unfilled"] == []


class TestAQuestionIsNotAnAnswer:
    """The defect: asked "what if I switch to 17 month?", the extractor
    reported track="17 month" and the student's committed plan was rebuilt on
    the other track -- `CoursePlan` included, which is per-user, outlives the
    conversation and is what `/api/thrive/plan` serves. A what-if replaced the
    real plan of study.
    """

    def test_a_hypothetical_does_not_overwrite_an_answer_on_file(self):
        from rsm_thrive.services.courses import planner

        stored = {"track": "11 month", "goals": ["data-scientist"]}
        merged = planner.merge_intake(stored, {"track": "17 month"}, asking=True)
        assert merged["track"] == "11 month"

    def test_a_question_commits_nothing_not_even_into_a_blank(self):
        """Narrowed after measurement. Allowing a question to fill BLANKS was
        the original rule, on the theory that "what if I want to be a data
        scientist?" carries real information. It carries the same risk in a
        quieter form: probing every step of the interview, "is next quarter
        heavy?" wrote workload="heavy" from a turn that stated no preference.
        The interview asks again instead."""
        from rsm_thrive.services.courses import planner

        merged = planner.merge_intake({"track": "11 month"},
                                      {"goals": ["data-scientist"]}, asking=True)
        assert "goals" not in merged
        assert merged["track"] == "11 month", "what was on file stays on file"

    def test_a_plain_answer_still_overwrites(self):
        from rsm_thrive.services.courses import planner

        merged = planner.merge_intake({"track": "17 month"},
                                      {"track": "11 month"}, asking=False)
        assert merged["track"] == "11 month"

    @pytest.mark.parametrize("text", [
        "what if I switch to 17 month?", "should I have said data engineer?",
        "what electives are there?", "why do you need my track?",
        "can I change this later?",
    ])
    def test_these_read_as_questions(self, text):
        from rsm_thrive.services.courses import planner

        assert planner.is_question(text)

    @pytest.mark.parametrize("text", [
        "11 month", "data scientist", "moderate", "11 month?",
        "python 4 sql 3", "",
    ])
    def test_these_read_as_answers(self, text):
        from rsm_thrive.services.courses import planner

        # "11 month?" is a hedged ANSWER, not a question -- a trailing "?" alone
        # must not be enough, or an uncertain student stops being heard.
        assert not planner.is_question(text)


class TestThePlanIsNotReprintedAtEveryTurn:
    """The defect: `_plan_reply` was the unconditional fallthrough, so "ok",
    "thanks!" and "how do I enrol?" each reprinted the whole 3,680-character
    plan of study -- 24 of 25 probes."""

    @pytest.mark.parametrize("text", [
        "show me the plan", "print it again", "the plan again", "my plan",
    ])
    def test_these_ask_for_the_plan(self, text):
        from rsm_thrive.services.courses.replies import _wants_the_plan

        assert _wants_the_plan(text)

    @pytest.mark.parametrize("text", [
        "thanks!", "ok", "hmm", "how do I enrol?", "can you email this to me?",
    ])
    def test_these_do_not(self, text):
        from rsm_thrive.services.courses.replies import _wants_the_plan

        assert not _wants_the_plan(text)


class TestAnIncidentalWordIsNotAnAnswer:
    """Found by probing a question at every point in the interview: the word
    "heavy" inside "is next quarter heavy?" was recorded as the student's
    workload preference. The turn stated no preference at all."""

    @pytest.mark.parametrize("text", [
        "is next quarter heavy?", "is the workload heavy?",
        "are these courses light?", "does it get heavy later?",
        "will it be a light quarter?", "can I do 11 month?",
        "should I pick data scientist?", "was 17 month the longer one?",
    ])
    def test_these_read_as_questions(self, text):
        from rsm_thrive.services.courses import planner

        assert planner.is_question(text), text

    @pytest.mark.parametrize("text", [
        "heavy", "light", "11 month", "11 month?", "moderate",
        "data scientist", "python 4 sql 3",
    ])
    def test_answers_are_still_answers(self, text):
        # A hedged answer starts with its own value, never with an auxiliary.
        from rsm_thrive.services.courses import planner

        assert not planner.is_question(text), text

    def test_the_measured_case_end_to_end(self):
        from rsm_thrive.services.courses import planner

        extracted = planner.normalise_intake({"workload": "heavy"})
        assert extracted == {"workload": "heavy"}, "the extractor does report it"
        merged = planner.merge_intake({}, extracted,
                                      asking=planner.is_question("is next quarter heavy?"))
        assert merged == {}, "but a question must not turn it into an answer"
