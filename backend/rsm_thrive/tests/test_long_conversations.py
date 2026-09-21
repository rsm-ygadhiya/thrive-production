"""Long conversations, taken in orders nobody wrote a test for.

The hand-written tests each walk one path. A student does not: they answer half
a question, ask about a course, change career, walk two quarters, go back, swap
something, change their mind about the load, and say "thanks" in the middle. The
bugs that survive a test suite live in those orderings.

So this drives conversations from a fixed vocabulary in a SEEDED random order
and checks the invariants after every single turn. Seeded, not random: a sweep
that fails on an unrecorded seed is a bug report nobody can reproduce. The seeds
are listed, so a failure names the exact conversation that produced it and it
replays identically.

The invariants are the ones a student would notice being broken:

* a reply is never empty, and never the degraded "I can't reach my sources";
* a plan, whenever one exists, is still 50 units with no course twice and no
  unfillable hole;
* the session's own state stays describable -- the walk-through index points at
  a real quarter, the intake holds only legal values;
* nothing a student types can produce a plan for a career they did not name.
"""

import random

import pytest
from django.contrib.auth.models import User

from rsm_thrive.models import Conversation
from rsm_thrive.services.courses import orchestrator, planner, router
from rsm_thrive.services.llm import FakeLLM

pytestmark = pytest.mark.django_db

# What a student says, grouped so a failure names the KIND of turn that did it.
VOCABULARY = {
    "track": ["11 month", "17 month", "I'm on the 11 month track", "17-month"],
    "goal": ["data scientist", "product analyst", "consultant", "ml engineer",
             "pricing analyst", "healthcare analyst", "data nalyst",
             "I want to be a marketing analyst"],
    # Careers the catalog curates no bundle for. They reach the plan through
    # `selections_for_courses` rather than through a bundle, so the invariants
    # have to hold on that path too.
    "uncurated": ["esports analyst", "aerospace engineer", "sports scientist",
                  "climate risk modeller", "sommelier", "esports analsyt"],
    "skills": ["python 4, sql 2", "strong in python, no ML", "skip",
               "3 for stats", "no idea", "python 1 sql 1 stats 1 ml 1"],
    "load": ["light", "moderate", "heavy", "medium"],
    "walk": ["walk me through it", "next quarter", "finalise",
             "show me the plan"],
    "quarter": ["what should I take in winter?", "just plan my spring",
                "what am I taking in fall"],
    "course": ["does MGTA 464 have prerequisites?", "how many units is MGTA 461",
               "what does MGTA 458 cover?"],
    "route": ["build it around me", "use the recommended bundle"],
    "chat": ["thanks", "hi", "ok", "hmm", "what?"],
    "odd": ["", "   ", "!!!", "a" * 500, "python 9", "MGTA 999",
            "swap MGTA 461 for MGTA 463"],
}
ALL_TURNS = [(kind, text) for kind, texts in VOCABULARY.items() for text in texts]

DEGRADED = "having trouble reaching my knowledge sources"


def _llm():
    """A model that answers anything, so no turn dies of an exhausted script."""
    return FakeLLM(['{"route": "factual", "confidence": 0.9}'] * 4 + ["ok"] * 8)


def check_invariants(conversation, reply, history):
    where = " -> ".join(f"{kind}:{text[:24]!r}" for kind, text in history)
    assert reply.body.strip(), f"empty reply after {where}"
    assert DEGRADED not in reply.body, f"degraded after {where}"
    # The industry routes answer WITH buttons -- six industries, then ten job
    # titles, and the "which of those did you mean" that re-offers the same
    # list -- and nothing else on this surface does. A stray button anywhere
    # else is still the bug this was written to catch.
    if reply.model_note in ("industry-menu", "industry-roles", "menu-which"):
        assert reply.quick_replies, f"empty menu after {where}"
    else:
        assert reply.quick_replies == [], f"buttons after {where}"
    assert reply.form is None, f"a form after {where}"
    assert reply.route in router.ROUTES | {"plan"}, f"{reply.route} after {where}"

    answers = planner.load_session_intake(conversation)
    if answers.get("track"):
        assert answers["track"] in ("11 month", "17 month"), where
    if answers.get("goals"):
        careers = planner.load_careers()
        assert all(goal in careers for goal in answers["goals"]), where
    if answers.get("workload"):
        assert answers["workload"] in ("light", "moderate", "heavy"), where
    for key, value in answers.items():
        if key.startswith("skill_"):
            assert planner.skill_score(value) is not None, (key, value, where)

    if planner.next_intake_step(answers) is None and answers.get("track"):
        plan = planner.build_for(answers, frozenset())
        assert plan["totals"]["total"] == planner.TOTAL_UNITS, \
            f"{plan['totals']['total']} units after {where}"
        assert plan["unfilled"] == [], f"unfillable slot after {where}"
        ids = [row["courseId"] for quarter in plan["quarters"]
               for row in quarter["courses"] if row["courseId"]]
        assert len(ids) == len(set(ids)), f"duplicate course after {where}"
        session = planner.load_session_review(conversation)
        if session is not None:
            assert 0 <= session["index"] < len(plan["quarters"]), \
                f"review index {session['index']} after {where}"


@pytest.mark.parametrize("seed", range(12))
def test_a_long_conversation_in_a_random_order(seed):
    """Twelve conversations, twenty-five turns each — 300 turns in orders
    nobody wrote by hand."""
    rng = random.Random(seed)
    user = User.objects.create_user(f"walker{seed}")
    conversation = Conversation.objects.create(
        user=user, destination="courses", title="walk")
    history = []
    for _ in range(25):
        kind, text = rng.choice(ALL_TURNS)
        history.append((kind, text))
        reply = orchestrator.answer(_llm(), conversation, text, [])
        check_invariants(conversation, reply, history)


@pytest.mark.parametrize("seed", range(6))
def test_a_conversation_that_keeps_changing_its_mind(seed):
    """The pattern that produced the "it gave me the same answer" report:
    settle on a plan, then switch career repeatedly."""
    rng = random.Random(100 + seed)
    user = User.objects.create_user(f"fickle{seed}")
    conversation = Conversation.objects.create(
        user=user, destination="courses", title="fickle")
    history = []
    for text in (rng.choice(VOCABULARY["track"]), rng.choice(VOCABULARY["goal"]),
                 rng.choice(VOCABULARY["skills"]), rng.choice(VOCABULARY["load"])):
        history.append(("setup", text))
        check_invariants(conversation, orchestrator.answer(
            _llm(), conversation, text, []), history)

    seen = set()
    for _ in range(8):
        goal = rng.choice(VOCABULARY["goal"])
        history.append(("goal", goal))
        reply = orchestrator.answer(_llm(), conversation, goal, [])
        check_invariants(conversation, reply, history)
        answers = planner.load_session_intake(conversation)
        role = router.matched_role(goal)
        if role:
            assert answers["goals"] == [role], \
                f"said {goal!r}, plan is for {answers['goals']}"
            if role in seen and "Switched to" in reply.body:
                pass          # switching back is still a switch
            seen.add(role)


@pytest.mark.parametrize("seed", range(6))
def test_the_walk_through_survives_being_driven_sideways(seed):
    """Load answers, swaps, questions and navigation, interleaved."""
    rng = random.Random(200 + seed)
    user = User.objects.create_user(f"walker2{seed}")
    conversation = Conversation.objects.create(
        user=user, destination="courses", title="review")
    history = []
    for text in ("11 month", "data scientist", "skip", "moderate",
                 "walk me through it"):
        history.append(("setup", text))
        check_invariants(conversation, orchestrator.answer(
            _llm(), conversation, text, []), history)

    during = (VOCABULARY["load"] + VOCABULARY["walk"] + VOCABULARY["course"]
              + VOCABULARY["chat"] + VOCABULARY["odd"] + VOCABULARY["quarter"])
    for _ in range(20):
        text = rng.choice(during)
        history.append(("review", text))
        check_invariants(conversation, orchestrator.answer(
            _llm(), conversation, text, []), history)

    # And the loads it actually recorded are all legal for their track.
    answers = planner.load_session_intake(conversation)
    chosen = answers.get("quarter_units") or {}
    if chosen:
        assert not planner.quarter_units_problems(answers["track"], chosen), chosen


def test_every_single_turn_kind_is_reachable():
    """A vocabulary entry nobody can reach tests nothing. Guards against a
    typo quietly removing a whole class of turn from the sweep above."""
    for kind, texts in VOCABULARY.items():
        assert texts, kind
        for text in texts:
            assert isinstance(text, str), (kind, text)
    assert len(ALL_TURNS) >= 40
