"""Several conversations at once, and topics changed mid-conversation.

Two different questions that sound like one.

**Switching CHATS.** The rail lists saved conversations and a student clicks
between them. Every piece of planning state is scoped to a `Conversation` —
the intake, the walk-through position, which questions have been asked, the
situational position — and the whole point of that scoping is that opening a
second chat starts a second plan. If any of it leaked, a student would answer
"11 month" in one chat and find it applied to another, or reopen last week's
conversation and be handed this week's plan.

**Switching TOPIC.** Inside one conversation, going from a plan to a course
question to a career change to a quarter and back. Each of those routes writes
something; none may corrupt what another wrote.

`CoursePlan` is the deliberate exception and it is tested here too: it is keyed
to the USER, is what `GET /plan` serves, and outlives the chat that built it.
"""

import json

import pytest
from django.contrib.auth.models import User

from rsm_thrive.models import Conversation, CoursePlan, PlannerSession
from rsm_thrive.services.courses import orchestrator, planner, router
from rsm_thrive.services.llm import FakeLLM

pytestmark = pytest.mark.django_db


@pytest.fixture
def user():
    return User.objects.create_user("stu")


def chat(user, title):
    return Conversation.objects.create(user=user, destination="courses",
                                        title=title)


def llm():
    return FakeLLM(['{"route": "factual", "confidence": 0.9}'] * 3 + ["ok"] * 6)


def plan_through(conversation, role="data scientist", track="11 month",
                 skills="skip", load="moderate"):
    for turn in (f"{track}, {role}", skills, load):
        orchestrator.answer(llm(), conversation, turn, [])
    return planner.load_session_intake(conversation)


# ---------------------------------------------------------------------------
# Switching between chats
# ---------------------------------------------------------------------------

class TestTwoChatsAtOnce:
    def test_each_holds_its_own_intake(self, user):
        first, second = chat(user, "one"), chat(user, "two")
        plan_through(first, "data scientist", "11 month")
        plan_through(second, "pricing analyst", "17 month")

        assert planner.load_session_intake(first)["goals"] == ["data-scientist"]
        assert planner.load_session_intake(first)["track"] == "11 month"
        assert planner.load_session_intake(second)["goals"] == ["pricing-analyst"]
        assert planner.load_session_intake(second)["track"] == "17 month"

    def test_answering_in_one_does_not_answer_in_the_other(self, user):
        first, second = chat(user, "one"), chat(user, "two")
        orchestrator.answer(llm(), first, "data scientist", [])
        # `second` has been told nothing at all.
        reply = orchestrator.answer(llm(), second, "11 month", [])
        assert "aiming for" in reply.body, "it used the other chat's goal"
        assert not planner.load_session_intake(second).get("goals")

    def test_the_walk_through_position_is_per_chat(self, user):
        first, second = chat(user, "one"), chat(user, "two")
        plan_through(first)
        plan_through(second)
        orchestrator.answer(llm(), first, "walk me through it", [])
        orchestrator.answer(llm(), first, "next quarter", [])
        orchestrator.answer(llm(), first, "next quarter", [])
        orchestrator.answer(llm(), second, "walk me through it", [])

        assert planner.load_session_review(first)["index"] == 2
        assert planner.load_session_review(second)["index"] == 0

    def test_a_quarter_load_chosen_in_one_stays_there(self, user):
        first, second = chat(user, "one"), chat(user, "two")
        plan_through(first, track="17 month")
        plan_through(second, track="17 month")
        orchestrator.answer(llm(), first, "walk me through it", [])
        orchestrator.answer(llm(), first, "next quarter", [])
        orchestrator.answer(llm(), first, "next quarter", [])
        orchestrator.answer(llm(), first, "light", [])

        assert planner.load_session_intake(first).get("quarter_loads")
        assert not planner.load_session_intake(second).get("quarter_loads")

    def test_the_questions_asked_are_counted_per_chat(self, user):
        """`asked` is what stops a question repeating. Shared, it would stop it
        being asked in a chat where it never was."""
        first, second = chat(user, "one"), chat(user, "two")
        orchestrator.answer(llm(), first, "17 month, data scientist", [])
        assert planner.session_has_asked(first, "workload")
        assert not planner.session_has_asked(second, "workload")

        reply = orchestrator.answer(llm(), second, "17 month, data scientist", [])
        assert "spread across the quarters" in reply.body.lower()

    def test_the_situational_position_is_per_chat(self, user):
        first, second = chat(user, "one"), chat(user, "two")
        said = json.dumps({"track": "11 month", "previous_track": None,
                           "current_quarter": "winter", "finish_on_time": True,
                           "completed_codes": []})
        orchestrator.answer(FakeLLM([said, "Tight but doable."]), first,
                            "I switched to 11 month and I'm in winter", [])
        assert planner.load_session_situation(first).get("start_from") == "winter"
        assert planner.load_session_situation(second) == {}

    def test_a_goal_does_not_leak_between_chats(self, user):
        """A career named in one chat must not become the other chat's goal.

        This used to assert on `unmatched_goal`, which the removed web lookup
        stored for a role it had no profile for. Nothing stores that now -- an
        uncovered role is refused outright -- so the leak this guards against
        is tested with a goal that really is remembered."""
        first, second = chat(user, "one"), chat(user, "two")
        orchestrator.answer(llm(), first, "data scientist", [])
        assert planner.load_session_intake(first)["goals"] == ["data-scientist"]
        assert not planner.load_session_intake(second).get("unmatched_goal")

    def test_three_chats_interleaved_keep_three_plans(self, user):
        chats = [chat(user, f"c{n}") for n in range(3)]
        roles = ["data scientist", "ml engineer", "marketing analyst"]
        tracks = ["11 month", "17 month", "11 month"]
        # One turn each, round robin, rather than one chat at a time.
        for turn in range(3):
            for c, role, track in zip(chats, roles, tracks):
                text = [f"{track}, {role}", "skip", "moderate"][turn]
                orchestrator.answer(llm(), c, text, [])
        for c, role, track in zip(chats, roles, tracks):
            answers = planner.load_session_intake(c)
            assert answers["track"] == track, c.title
            assert answers["goals"] == [router.matched_role(role)], c.title
            plan = planner.build_for(answers, frozenset())
            assert plan["totals"]["total"] == planner.TOTAL_UNITS, c.title

    def test_deleting_one_chat_leaves_the_other_alone(self, user, client):
        first, second = chat(user, "one"), chat(user, "two")
        plan_through(first)
        plan_through(second, "pricing analyst")
        client.force_login(user)
        assert client.delete(
            f"/api/thrive/conversations/conv-{first.pk}").status_code == 200
        assert not PlannerSession.objects.filter(conversation=first).exists()
        assert planner.load_session_intake(second)["goals"] == ["pricing-analyst"]

    def test_the_committed_plan_is_the_users_and_survives(self, user):
        """`CoursePlan` is keyed to the USER on purpose: it is what
        `GET /plan` serves and it outlives the chat that built it."""
        first = chat(user, "one")
        plan_through(first, "data scientist")
        assert CoursePlan.objects.filter(user=user).count() == 1

        second = chat(user, "two")
        plan_through(second, "pricing analyst")
        assert CoursePlan.objects.filter(user=user).count() == 1
        assert CoursePlan.objects.get(user=user).intake["goals"] == ["pricing-analyst"]


class TestTwoStudentsAtOnce:
    def test_nothing_crosses_between_them(self, user):
        other = User.objects.create_user("other")
        mine, theirs = chat(user, "mine"), chat(other, "theirs")
        plan_through(mine, "data scientist", "11 month")
        plan_through(theirs, "ml engineer", "17 month")
        assert planner.load_session_intake(mine)["goals"] == ["data-scientist"]
        assert planner.load_session_intake(theirs)["goals"] == ["ml-engineer"]
        assert CoursePlan.objects.get(user=user).intake["track"] == "11 month"
        assert CoursePlan.objects.get(user=other).intake["track"] == "17 month"


# ---------------------------------------------------------------------------
# Switching topic inside one conversation
# ---------------------------------------------------------------------------

class TestChangingTheSubjectMidConversation:
    def test_a_course_question_does_not_disturb_the_plan(self, user):
        conversation = chat(user, "one")
        plan_through(conversation)
        before = planner.elective_codes(
            planner.build_for(planner.load_session_intake(conversation), frozenset()))

        orchestrator.answer(FakeLLM(["MGTA 464 is 2 units."]), conversation,
                            "how many units is MGTA 464", [])
        after = planner.elective_codes(
            planner.build_for(planner.load_session_intake(conversation), frozenset()))
        assert before == after

    def test_the_walk_through_survives_a_detour(self, user):
        conversation = chat(user, "one")
        plan_through(conversation)
        orchestrator.answer(llm(), conversation, "walk me through it", [])
        orchestrator.answer(llm(), conversation, "next quarter", [])
        at = planner.load_session_review(conversation)["index"]

        orchestrator.answer(FakeLLM(["MGTA 464 is 2 units."]), conversation,
                            "how many units is MGTA 464", [])
        assert planner.load_session_review(conversation)["index"] == at, \
            "a detour moved the walk-through"

        reply = orchestrator.answer(llm(), conversation, "next quarter", [])
        assert planner.load_session_review(conversation)["index"] == at + 1
        assert reply.body.strip()

    def test_changing_career_keeps_everything_else(self, user):
        conversation = chat(user, "one")
        plan_through(conversation, "data scientist", "11 month",
                     skills="python 5, sql 4", load="light")
        before = planner.load_session_intake(conversation)

        orchestrator.answer(llm(), conversation, "actually a pricing analyst", [])
        after = planner.load_session_intake(conversation)
        assert after["goals"] == ["pricing-analyst"]
        assert after["track"] == before["track"], "the track was lost"
        assert after["skill_python"] == 5, "the skills were lost"
        assert after["workload"] == before["workload"], "the load was lost"

    def test_a_full_round_trip_of_topics_holds(self, user):
        """Plan, course question, one quarter, career change, walk-through,
        load change, back to the plan."""
        conversation = chat(user, "one")
        plan_through(conversation)
        turns = [
            "does MGTA 464 have prerequisites?",
            "what should I take in winter?",
            "actually a consultant",
            "walk me through it",
            "next quarter",
            "light",
            "thanks",
            "show me the plan",
        ]
        for turn in turns:
            reply = orchestrator.answer(llm(), conversation, turn, [])
            assert reply.body.strip(), turn
            answers = planner.load_session_intake(conversation)
            plan = planner.build_for(answers, frozenset())
            assert plan["totals"]["total"] == planner.TOTAL_UNITS, turn
            assert plan["unfilled"] == [], turn
        assert planner.load_session_intake(conversation)["goals"] == ["consultant"]
        assert "MGTA" in reply.body

    def test_switching_track_mid_walk_through_and_carrying_on(self, user):
        conversation = chat(user, "one")
        plan_through(conversation, track="17 month")
        orchestrator.answer(llm(), conversation, "walk me through it", [])
        for _ in range(4):
            orchestrator.answer(llm(), conversation, "next quarter", [])
        assert planner.load_session_review(conversation)["index"] == 4

        orchestrator.answer(llm(), conversation, "11 month", [])
        assert planner.load_session_review(conversation) is None, \
            "the position pointed into a track with fewer quarters"

        reply = orchestrator.answer(llm(), conversation, "walk me through it", [])
        assert "Summer III" in reply.body
        assert planner.load_session_review(conversation)["index"] == 0
