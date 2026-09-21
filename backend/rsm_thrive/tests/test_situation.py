"""The situational route: where a student is, against what the degree requires.

The route that matters most and the one with the sharpest failure mode. If it
says someone needs 14 more units and it is wrong, they miss graduation — so the
tests here are mostly about the line between the arithmetic and the model:

* the arithmetic is done in `planner.build_plan` and `situation.ledger_for`,
  from `courses.json` and the published skeletons, and is asserted exactly;
* the model is handed the finished ledger and never computes anything, which is
  asserted by running the whole route with a model that returns nothing at all
  and checking every number is still there and still right.
"""

import json

import pytest
from django.contrib.auth.models import User

from rsm_thrive.models import Conversation
from rsm_thrive.services.courses import orchestrator, planner, router, situation
from rsm_thrive.services.llm import FakeLLM
from rsm_thrive.testing import enroll, make_course, make_student

pytestmark = pytest.mark.django_db

FULL = {"track": "11 month", "goals": ["data-scientist"]}


@pytest.fixture
def student():
    return make_student(username="mid", track="11 month")


@pytest.fixture
def conversation(student):
    return Conversation.objects.create(user=student.user, destination="courses",
                                       title="where am I")


def said(**overrides):
    blank = {"track": None, "previous_track": None, "current_quarter": None,
             "finish_on_time": False, "completed_codes": []}
    return json.dumps({**blank, **overrides})


# ---------------------------------------------------------------------------
# The planner change
# ---------------------------------------------------------------------------

class TestAPlanCanStartAnywhere:
    """`build_plan` always built from Summer, whatever the date. A student who
    switched tracks mid-programme, or who asks in week three of Winter, could
    only be answered by ignoring the first half of what came back."""

    def test_it_plans_only_the_quarters_that_remain(self):
        plan = planner.build_plan(FULL, start_from="winter")
        assert [q["label"] for q in plan["quarters"]] == ["Winter", "Spring"]
        assert plan["quartersRemaining"] == 2
        assert plan["startFrom"] == "winter"

    def test_a_plan_from_the_beginning_is_unchanged(self):
        """The parameter is additive: every plan built before it existed must
        come back byte-identical."""
        before = planner.build_plan(FULL)
        after = planner.build_plan(FULL, start_from="summer")
        assert [q["label"] for q in before["quarters"]] \
            == [q["label"] for q in after["quarters"]]
        assert before["startFrom"] is None and before["unscheduled"] == []
        assert before["totals"]["total"] == planner.TOTAL_UNITS

    def test_required_courses_behind_the_start_are_reported_not_dropped(self):
        """Cutting Summer and Fall removes five required courses from the
        plan. A plan that silently omits them reads as a plan that does not
        need them."""
        plan = planner.build_plan(FULL, start_from="winter")
        outstanding = {row["code"] for row in plan["unscheduled"]}
        assert outstanding == {"MGTA 451", "MGTA 403", "MGTA 464",
                               "MGTA 452", "MGTA 453"}
        assert all(row["quarterLabel"] for row in plan["unscheduled"])

    def test_a_course_already_taken_is_not_reported_as_outstanding(self):
        taken = {"MGTA 451", "MGTA 403"}
        plan = planner.build_plan(FULL, taken_ids=taken, start_from="winter")
        assert {row["code"] for row in plan["unscheduled"]} \
            == {"MGTA 464", "MGTA 452", "MGTA 453"}

    def test_the_ledger_adds_up(self):
        taken = {"MGTA 451", "MGTA 452", "MGTA 453", "MGTA 403", "MGTA 464"}
        plan = planner.build_plan(FULL, taken_ids=taken, start_from="winter")
        totals = plan["totals"]
        assert totals["completed"] == 16, "MGTA 451+403+464 (8u) plus 452+453 (8u)"
        assert totals["scheduled"] == totals["core"] + totals["elective"]
        assert totals["outstanding"] == (planner.TOTAL_UNITS
                                         - totals["completed"]
                                         - totals["scheduled"])

    def test_an_unknown_quarter_is_an_error_not_a_guess(self):
        with pytest.raises(ValueError) as raised:
            planner.build_plan(FULL, start_from="fall-two")
        assert "summer, fall, winter, spring" in str(raised.value)


class TestReadingWhichQuarterSomeoneMeans:
    @pytest.mark.parametrize("track,said,expected", [
        ("11 month", "winter", "winter"),
        ("11 month", "I'm already in Winter", "winter"),
        ("11 month", "Summer III", "summer"),
        ("17 month", "second fall", "fall-two"),
        ("17 month", "Fall (second year)", "fall-two"),
        ("17 month", "fall", "fall"),
    ])
    def test_these_resolve(self, track, said, expected):
        assert planner.resolve_quarter(track, said) == expected

    @pytest.mark.parametrize("track,said", [
        ("11 month", "fall two"),          # that track has no second Fall
        ("11 month", "my second one"),
        ("11 month", ""),
        ("17 month", "the middle bit"),
    ])
    def test_these_return_nothing_rather_than_guessing(self, track, said):
        """A plan built from the wrong quarter is wrong about every date in
        it. "I think you meant Winter" is a question worth asking."""
        assert planner.resolve_quarter(track, said) is None


# ---------------------------------------------------------------------------
# The arithmetic, and who does it
# ---------------------------------------------------------------------------

class TestTheModelNeverComputesAnything:
    def test_the_numbers_survive_a_model_that_says_nothing(self, student,
                                                           conversation):
        """`explain` returns "" on any failure. The ledger and the plan are the
        answer; losing the paragraph must never lose the numbers."""
        # One call for the extraction, then the explanation raises (exhausted).
        fake = FakeLLM([said(track="11 month", previous_track="17 month",
                             current_quarter="winter", finish_on_time=True)])
        body, asked, resolved = situation.answer(
            fake, student.user, "I switched to 11-month, I'm in winter")
        assert asked == ""
        assert resolved["start_from"] == "winter"
        assert "**Winter**" in body
        assert "2 quarters left" in body
        assert "MGTA 451" in body, "the outstanding requirements are still named"

    def test_the_explanation_is_given_the_ledger_and_told_it_is_all_it_has(self):
        fake = FakeLLM(["It's tight but it works."])
        ledger = {"unitsOutstanding": 4, "quartersRemaining": 2}
        assert situation.explain(fake, ledger, {}) == "It's tight but it works."
        system, messages, _json = fake.calls[0]
        assert "must come from the ledger" in system
        assert "Do NOT add, total, subtract or estimate anything" in system
        assert '"unitsOutstanding": 4' in messages[0]["content"]

    def test_a_shortfall_is_stated_plainly(self, student, conversation):
        fake = FakeLLM([said(track="11 month", current_quarter="spring"), ""])
        body, _asked, _resolved = situation.answer(fake, student.user,
                                                   "I'm in spring")
        assert "does not complete the degree" in body
        assert "advising" in body.lower()


class TestOnlyTheTranscriptAddsUnits:
    def test_completed_units_come_from_enrolments(self, student, conversation):
        course = make_course(id="c-451", code="MGTA 451", units=4)
        enroll(student, course, completed=True)
        position = {"track": "11 month", "start_from": "winter",
                    "previous_track": None}
        ledger, _plan = situation.ledger_for(student.user, position)
        assert ledger["unitsCompleted"] == 4
        assert ledger["completedCourses"] == ["MGTA 451"]

    def test_a_course_this_programme_does_not_carry_cannot_inflate_it(self, student):
        position = {"track": "11 month", "start_from": "winter",
                    "previous_track": None}
        ledger, _plan = situation.ledger_for(student.user, position,
                                             ["MGTA 451", "PHYS 999"])
        assert ledger["unitsCompleted"] == 4, "only the real course counted"
        assert ledger["completedCourses"] == ["MGTA 451"]

    def test_a_course_being_taken_now_is_not_banked_yet(self, student):
        """Enrolled and completed are different questions. Crediting units for
        a course still being taught is an error a student only finds out about
        at graduation."""
        course = make_course(id="c-452", code="MGTA 452", units=4)
        enroll(student, course, completed=False)
        assert planner.completed_course_ids(student.user) == set()
        assert "MGTA 452" in planner.taken_course_ids(student.user)


# ---------------------------------------------------------------------------
# The one question it is allowed to ask
# ---------------------------------------------------------------------------

class TestItAsksAtMostOneThing:
    def test_an_unreadable_quarter_is_asked_for_by_name(self, student):
        fake = FakeLLM([said(track="11 month", current_quarter="my second one")])
        body, asked, partial = situation.answer(
            fake, student.user, "I'm behind, what's left in my degree")
        assert asked == "quarter"
        # The half it DID work out is handed back, so the next turn -- which
        # will say only "winter" -- does not have to re-state the track.
        assert partial == {"track": "11 month"}
        assert "Summer III" in body and "Spring" in body
        assert "MGTA" not in body, "it does not answer before it knows"

    def test_the_track_falls_back_to_the_profile_rather_than_asking(self, student):
        """`StudentProfile.track` is on file. Asking for something already
        known is the interview behaviour being removed."""
        position, missing = situation.resolve_position(
            student.user, {"said_quarter": "winter"})
        assert missing == "" and position["track"] == "11 month"

    def test_a_switch_the_student_states_beats_the_profile(self, student):
        position, missing = situation.resolve_position(
            student.user, {"track": "17 month", "previous_track": "11 month",
                           "said_quarter": "winter"})
        assert missing == "" and position["track"] == "17 month"


# ---------------------------------------------------------------------------
# Through the orchestrator
# ---------------------------------------------------------------------------

class TestTheRouteIsReachedWithoutAModelCall:
    def test_the_question_from_the_brief_routes_here(self):
        route = router.rule_route(
            "I've switched from 17-month to 11-month and I'm already in "
            "Winter, what do I take to finish on time?")
        assert route.name == router.SITUATIONAL
        assert route.confidence is None

    def test_the_orchestrator_answers_it_end_to_end(self, student, conversation):
        fake = FakeLLM([said(track="11 month", previous_track="17 month",
                             current_quarter="winter", finish_on_time=True),
                        "You are carrying a lot into two quarters."])
        reply = orchestrator.answer(
            fake, conversation,
            "I've switched from 17-month to 11-month and I'm already in "
            "Winter, what do I take to finish on time?", [])
        assert reply.route == router.SITUATIONAL
        assert reply.model_note == "situation"
        assert "You are carrying a lot into two quarters." in reply.body
        assert "**Winter**" in reply.body
        assert reply.quick_replies == []

    def test_the_second_turn_does_not_re_ask_what_the_first_established(
            self, student, conversation):
        """Asked for a track and then for a quarter, the second turn states
        only the quarter. A route that cannot remember the first answer asks
        for the track again forever."""
        from rsm_thrive.models import StudentProfile

        StudentProfile.objects.filter(pk=student.pk).update(track="")
        first = FakeLLM([said(track="17 month")])
        reply = orchestrator.answer(first, conversation,
                                    "I'm on the 17 month one, am I behind?", [])
        assert "Which quarter are you in" in reply.body
        assert planner.load_session_situation(conversation)["track"] == "17 month"

        second = FakeLLM([said(current_quarter="winter"), "Tight, but doable."])
        reply = orchestrator.answer(second, conversation, "winter", [])
        assert "Which track" not in reply.body
        assert "**Winter**" in reply.body
