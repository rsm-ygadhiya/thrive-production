"""Where else a course could go, said at the moment it matters.

A swap used to speak only about the course coming in. The course going OUT is
the one the student may regret: dropped from the one quarter it runs in, it is
gone; dropped from one of three, it can be picked up later. And a student who
names a course they WANT was answered from the catalog row ("offered in
Winter") rather than from their plan ("your Winter, in place of MGTA 458").
"""

import pytest
from django.contrib.auth.models import User

from rsm_thrive.models import Conversation
from rsm_thrive.services.courses import orchestrator, planner
from rsm_thrive.services.courses.electives import load_catalog
from rsm_thrive.services.llm import FakeLLM

pytestmark = pytest.mark.django_db

ANSWERS = {
    "track": "11 month",
    "goals": ["data-scientist"],
    "skill_python": "comfortable",
    "skill_sql": "basic",
    "skill_stats": "comfortable",
    "skill_ml": "basic",
}


def _by_id():
    return {course["id"]: course for course in load_catalog()}


def _plan():
    return planner.build_for(ANSWERS)


def _row(plan, code):
    _q, _slot, row = planner.locate_code(plan, code)
    assert row is not None, f"{code} is not in the sample plan"
    return row


@pytest.fixture
def conversation():
    user = User.objects.create_user("avail")
    return Conversation.objects.create(user=user, destination="courses",
                                        title="planning")


def ask(conversation, *turns):
    reply = None
    for turn in turns:
        reply = orchestrator.answer(FakeLLM(["ok"] * 6), conversation, turn, [])
    return reply


# ---------------------------------------------------------------------------
# The facts underneath
# ---------------------------------------------------------------------------

class TestSeasons:
    def test_seasons_come_out_in_academic_order(self):
        course = {"offerings": [{"season": "SP"}, {"season": "FA"}, {"season": "WI"}]}
        assert planner.seasons_of(course) == ["FA", "WI", "SP"]

    def test_season_names_read_as_a_sentence(self):
        assert planner.season_names(["FA"]) == "Fall"
        assert planner.season_names(["FA", "WI"]) == "Fall and Winter"
        assert planner.season_names(["FA", "WI", "SP"]) == "Fall, Winter and Spring"
        assert planner.season_names([]) == ""

    def test_a_placement_needs_the_season_and_a_slot_of_the_same_size(self):
        """The two tests `apply_swap` applies — so nothing named here is a
        swap it would then refuse."""
        plan = _plan()
        by_id = _by_id()
        in_plan = {row["courseId"] for q in plan["quarters"] for row in q["courses"]}
        for course in load_catalog():
            if course["is_core"] or course["id"] in in_plan:
                continue
            for quarter, rows in planner.placements_for(plan, course):
                assert planner._offered_in(course, quarter["season"])
                for row in rows:
                    assert row["swappable"] and row["units"] == course["units"]
                    assert row["courseId"] != course["id"]
                    # And the swap itself agrees.
                    slot = quarter["courses"].index(row)
                    planner.apply_swap(ANSWERS, {}, quarter["key"], slot,
                                       course["id"])
                    assert by_id[course["id"]]  # the course is real


# ---------------------------------------------------------------------------
# The course going out
# ---------------------------------------------------------------------------

class TestTheCourseGoingOut:
    def test_a_course_offered_only_here_is_said_to_be_gone(self):
        plan = _plan()
        row = _row(plan, "MGTA 402")            # Fall only
        note = planner.displaced_note(plan, _by_id()[row["courseId"]], "fall")
        assert note.startswith("\n\n> ")
        assert "only offered in **Fall**" in note
        assert "no other quarter" in note
        assert "out of your plan" in note

    def test_a_course_offered_later_says_where_and_what_it_displaces(self):
        plan = _plan()
        row = _row(plan, "MGTA 457")            # in Fall; also Spring
        note = planner.displaced_note(plan, _by_id()[row["courseId"]], "fall")
        assert "also runs in Spring" in note
        assert "**Spring** (later," in note
        assert "in place of **MGTA 460**" in note
        assert "Say **swap MGTA 460 for MGTA 457**" in note

    def test_an_earlier_quarter_is_called_earlier(self):
        # Product Analyst's plan holds MGTA 457 in Spring; it also runs in Fall.
        plan = planner.build_for({**ANSWERS, "goals": ["product-analyst"]})
        row = _row(plan, "MGTA 457")
        note = planner.displaced_note(plan, _by_id()[row["courseId"]], "spring")
        assert "(earlier," in note
        assert "(later," not in note

    def test_a_season_the_catalog_only_knows_as_varies_is_hedged(self):
        # The CSE courses are the ones the catalog knows only as a pattern.
        plan = planner.build_for({**ANSWERS, "goals": ["ml-engineer"]})
        row = _row(plan, "CSE 251B")            # Winter here; Spring as "varies"
        note = planner.displaced_note(plan, _by_id()[row["courseId"]], "winter")
        assert "exact term varies" in note
        plan = _plan()
        row = _row(plan, "MGTA 457")            # Fall and Spring, real terms
        note = planner.displaced_note(plan, _by_id()[row["courseId"]], "fall")
        assert "exact term varies" not in note

    def test_offered_elsewhere_but_nowhere_to_put_it_says_so(self):
        """"Also offered in Winter" must not read as "you can still have it"
        when nothing in Winter can make room for it."""
        plan = _plan()
        course = {"id": "X 1", "code": "X 1", "units": 4, "is_core": False,
                  "offerings": [{"season": "FA", "term": "FA26"},
                                {"season": "SU", "term": "SU26"}]}
        # Summer has no swappable slot at all.
        note = planner.displaced_note(plan, course, "fall")
        assert "also runs in Summer" in note
        assert "no free 4-unit elective slot" in note
        assert "out of your plan for now" in note

    def test_an_unknown_quarter_yields_nothing(self):
        plan = _plan()
        assert planner.displaced_note(plan, _by_id()["MGTA 461"], "nope") == ""

    def test_a_swap_through_the_chat_carries_the_note(self, conversation):
        reply = ask(conversation, "data scientist, 11 month, moderate",
                    "swap MGTA 458 for MGT 404")
        assert reply.body.startswith("Swapped **MGTA 458** for **MGT 404**.")
        assert "**MGTA 458** is only offered in **Winter**" in reply.body
        assert reply.lead and reply.lead in reply.body
        assert "| " in reply.body, "and the plan is still printed under it"

    def test_mid_walk_through_the_note_survives_the_re_render(self, conversation):
        """The walk-through took only the first LINE of the swap reply as its
        lead; the note is a second paragraph and was being thrown away."""
        reply = ask(conversation, "data scientist, 11 month, moderate",
                    "walk me through it", "next", "swap MGTA 461 for MGTA 459")
        assert reply.model_note == "review"
        assert reply.body.startswith("Swapped **MGTA 461** for **MGTA 459**.")
        assert "**MGTA 461** is only offered in **Fall**" in reply.body
        assert "\n\n---\n\n" in reply.body


# ---------------------------------------------------------------------------
# The course wanted in
# ---------------------------------------------------------------------------

class TestTheCourseWantedIn:
    def test_it_says_the_seasons_and_the_plan_quarters_they_are(self):
        plan = _plan()
        note = planner.placement_note(plan, _by_id()["MGT 404"])   # Fall, Winter
        assert "**MGT 404** is offered in Fall and Winter" in note
        assert "**Fall** (in place of **MGTA 461**)" in note
        assert "**Winter** (in place of **MGTA 458**)" in note
        assert "Say **swap MGTA 461 for MGT 404**" in note

    def test_a_quarter_it_was_refused_from_is_named_first(self):
        plan = _plan()
        fall = next(q for q in plan["quarters"] if q["key"] == "fall")
        note = planner.placement_note(plan, _by_id()["MGTA 456"], missing_from=fall)
        assert note.lstrip("\n> ").startswith("**MGTA 456** isn't offered in **Fall**.")

    def test_a_core_course_is_not_offered_as_a_swap(self):
        note = planner.placement_note(_plan(), _by_id()["MGTA 452"])
        assert "core course" in note

    def test_a_course_already_taken_cannot_go_back_in(self):
        note = planner.placement_note(_plan(), _by_id()["MGTA 456"],
                                      taken_ids={"MGTA 456"})
        assert "already taken" in note

    def test_no_offering_on_file_is_said_rather_than_guessed(self):
        course = {"id": "X 2", "code": "X 2", "units": 4, "is_core": False,
                  "offerings": []}
        note = planner.placement_note(_plan(), course)
        assert "doesn't list when" in note
        assert "MSBA advising" in note

    def test_offered_but_no_slot_of_its_size_says_the_load_is_the_reason(self):
        course = {"id": "X 3", "code": "X 3", "units": 4, "is_core": False,
                  "offerings": [{"season": "SU", "term": "SU26"}]}
        note = planner.placement_note(_plan(), course)
        assert "In your plan that's Summer" in note
        assert "no free 4-unit elective slot" in note
        assert "Changing the load" in note

    @pytest.mark.parametrize("said", [
        "I want MGTA 456", "can I take MGTA 456?", "when is MGTA 456 offered",
        "MGTA 456", "is 456 available in winter", "I'd like to add 456",
    ])
    def test_wanting_a_course_reads_as_wanting_it(self, said):
        assert planner.wants_course(said)

    @pytest.mark.parametrize("said", [
        "does MGTA 456 have prerequisites?", "how many units is MGTA 461",
        "what does MGTA 458 cover?", "tell me about MGTA 456", "",
    ])
    def test_a_fact_about_a_course_is_not_a_want(self, said):
        assert not planner.wants_course(said)

    def test_wanting_a_course_not_in_the_plan_answers_from_the_plan(self, conversation):
        reply = ask(conversation, "data scientist, 11 month, moderate",
                    "I want MGTA 456")
        assert reply.model_note == "availability"
        assert reply.body.startswith("**MGTA 456 — ")
        assert "isn't in your plan yet" in reply.body
        assert "**MGTA 456** is offered in" in reply.body
        assert "in place of" in reply.body
        assert reply.quick_replies == []

    def test_asking_when_a_course_runs_answers_the_same_way(self, conversation):
        reply = ask(conversation, "data scientist, 11 month, moderate",
                    "when is CSE 251B offered?")
        assert reply.model_note == "availability"
        assert "offered in Winter and Spring" in reply.body
        assert "exact term varies" in reply.body

    def test_a_course_already_in_the_plan_is_not_claimed(self, conversation):
        """"I want MGTA 461" with 461 already scheduled is the change
        handler's turn: it shows alternatives, as before."""
        reply = ask(conversation, "data scientist, 11 month, moderate",
                    "I want MGTA 461")
        assert reply.model_note != "availability"
        assert "Other ways to fill that slot" in reply.body

    def test_a_fact_question_still_reaches_the_catalog_route(self, conversation):
        reply = ask(conversation, "data scientist, 11 month, moderate",
                    "does MGTA 456 have prerequisites?")
        assert reply.model_note != "availability"

    def test_a_swap_into_the_wrong_quarter_says_where_it_does_go(self, conversation):
        """MGTA 456 is not offered in Fall. The refusal used to stop at that."""
        reply = ask(conversation, "data scientist, 11 month, moderate",
                    "swap MGTA 461 for MGTA 456")
        assert "I can't put **MGTA 456** in that slot" in reply.body
        assert "**MGTA 456** isn't offered in **Fall**." in reply.body
        assert "**MGTA 456** is offered in" in reply.body
        assert "Say **swap" in reply.body

    def test_why_isnt_it_in_my_plan_says_where_it_could_be(self, conversation):
        reply = ask(conversation, "data scientist, 11 month, moderate",
                    "why isn't MGTA 456 in there")
        assert reply.model_note == "why"
        assert "isn't in your plan" in reply.body
        assert "**MGTA 456** is offered in" in reply.body
