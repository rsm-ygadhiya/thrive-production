"""Reading what a student meant by a job -- shorthand and the model's curated reading.

Found live, in one saved conversation: "i want to be a pm" was refused with
"I don't have a course recommendation for **pm**", and the student had to
type "product manager" to reach the profile that title already sat under.
The rules here are the ones that conversation broke, plus the guard rails the
fix must not loosen:

* Shorthand a profile lists resolves to it, without a model call, and is
  CONFIRMED back rather than acted on silently.
* Shorthand beside a digit is a time of day, not a job.
* The model may read a job as a curated profile only from the closed list, by
  id; anything else it says is thrown away and the turn is refused as before.
* A job the list does not cover is still refused, and still names no course.
"""

import json

import pytest
from django.contrib.auth.models import User

from rsm_thrive.models import Conversation
from rsm_thrive.services.courses import orchestrator, planner, router, understanding
from rsm_thrive.services.courses.electives import load_careers
from rsm_thrive.services.llm import FakeLLM

pytestmark = pytest.mark.django_db


@pytest.fixture
def user():
    return User.objects.create_user("stu")


@pytest.fixture
def conversation(user):
    return Conversation.objects.create(user=user, destination="courses",
                                       title="planning")


def ask(conversation, question, replies=None):
    """One turn. With no `replies`, FakeLLM raises if the flow consults it."""
    return orchestrator.answer(FakeLLM(list(replies or [])), conversation,
                               question, [])


def classified(route, role="", curated="", curated_confidence=0.0,
               confidence=0.9):
    return json.dumps({"route": route, "confidence": confidence, "role": role,
                       "industry": "", "curated": curated,
                       "curated_confidence": curated_confidence})


# ---------------------------------------------------------------------------
# Shorthand, read deterministically
# ---------------------------------------------------------------------------

class TestShorthandIsATitle:

    @pytest.mark.parametrize("said,expected", [
        ("i want to be a pm", "product-analyst"),
        ("I want to be a PM", "product-analyst"),
        ("17 month, pm", "product-analyst"),
        ("growth pm", "product-analyst"),
        ("product management", "product-analyst"),
        ("i wanna do ds", "data-scientist"),
        ("fp&a", "finance-quant"),
        ("a/b testing", "decision-scientist"),
        ("mle", "ml-engineer"),
    ])
    def test_shorthand_resolves_to_its_profile(self, said, expected):
        role_id, exact = router.role_match(said)
        assert role_id == expected, said
        assert exact is False, "shorthand is a reading, and must be confirmed"

    def test_every_listed_shorthand_reaches_its_own_profile(self):
        for role_id, role in load_careers().items():
            for short in role.get("shorthand") or []:
                assert router.matched_role(short) == role_id, short

    def test_shorthand_never_collides_with_another_profiles_title(self):
        """Titles may be shared on purpose ("growth analyst" sits under two
        profiles and is confirmed as a choice). Shorthand may not: it is
        matched before the fuzzy pass, so a clash would decide silently."""
        careers = load_careers()
        for role_id, role in careers.items():
            for short in role.get("shorthand") or []:
                for other_id, other in careers.items():
                    if other_id == role_id:
                        continue
                    theirs = {w.lower() for w in (other.get("titles") or [])
                              + (other.get("aliases") or [])
                              + (other.get("shorthand") or [])}
                    assert short.lower() not in theirs, (short, role_id, other_id)

    @pytest.mark.parametrize("said", [
        "class at 2 pm", "meet me at 2pm", "the 10:30 pm section",
        "is there a 4 pm class", "i have a BA in economics",
    ])
    def test_a_time_of_day_is_not_a_job(self, said):
        assert router.matched_role(said) == "", said

    def test_pm_rule_routes_without_a_model_call(self, conversation):
        ruled = router.rule_route("i want to be a pm")
        assert ruled is not None and ruled.name == router.ROLE
        assert ruled.role_id == "product-analyst"

    def test_pm_gets_the_product_recommendation_and_a_confirmation(self, conversation):
        reply = ask(conversation, "i want to be a pm")       # no replies: no model
        body = reply.body
        assert reply.model_note == "curated"
        assert not reply.refused
        assert "Reading **pm** as **Product Analyst" in body
        assert "meant something else" in body
        assert "MGTA 458" in body                              # the anchor course
        assert "don't have a course recommendation" not in body
        assert planner.load_session_intake(conversation).get("goals") == ["product-analyst"]

    def test_the_conversation_then_flows_on_to_a_plan(self, conversation):
        ask(conversation, "i want to be a pm")
        reply = ask(conversation, "17 months")
        assert "load" in reply.body.lower() or "plan of study" in reply.body
        reply = ask(conversation, "moderate")
        assert "plan of study" in reply.body
        assert "Product Analyst" in reply.body


# ---------------------------------------------------------------------------
# The model's curated reading
# ---------------------------------------------------------------------------

class TestTheModelReadsFromTheClosedList:

    def test_the_classifier_is_shown_the_roster(self, conversation):
        llm = FakeLLM([classified("out-of-scope")])
        orchestrator.answer(llm, conversation, "what colour is the sky", [])
        system = llm.calls[0][0]
        for role_id in load_careers():
            assert role_id in system, role_id
        assert "curated_confidence" in system

    def test_a_confident_reading_is_acted_on_and_confirmed(self, conversation):
        reply = ask(conversation, "i want to set airline ticket prices",
                    [classified("role", role="airline pricing",
                                curated="pricing-analyst", curated_confidence=0.9)])
        assert reply.model_note == "curated"
        assert "Reading **airline pricing** as **Pricing & Revenue Management**" in reply.body
        assert "meant something else" in reply.body
        assert planner.load_session_intake(conversation).get("goals") == ["pricing-analyst"]

    def test_a_reading_by_label_is_accepted_too(self):
        assert understanding.curated_reading(
            {"curated": "Pricing Analyst", "curated_confidence": 0.8}
        ) == ("pricing-analyst", 0.8)

    def test_an_id_not_in_the_table_is_thrown_away(self, conversation):
        reply = ask(conversation, "sommelier",
                    [classified("role", role="sommelier",
                                curated="wine-analyst", curated_confidence=0.99),
                     "unused"])
        assert reply.refused is True
        assert "sommelier" in reply.body
        assert not router.COURSE_CODE.search(reply.body)

    def test_a_hesitant_reading_is_not_acted_on(self, conversation):
        reply = ask(conversation, "something to do with numbers i guess",
                    [classified("role", role="numbers", curated="data-scientist",
                                curated_confidence=0.4), "unused"])
        assert reply.model_note != "curated"
        assert planner.load_session_intake(conversation).get("goals") in (None, [])

    def test_a_job_the_degree_cannot_serve_stays_refused(self, conversation):
        reply = ask(conversation, "firefighter",
                    [classified("role", role="firefighter", curated="",
                                curated_confidence=0.0), "unused"])
        assert reply.refused is True
        assert "advising" in reply.body.lower()

    def test_the_old_classifier_shape_still_works(self, conversation):
        """A verdict with no `curated` field at all -- every scripted test
        before this module -- behaves exactly as it did."""
        reply = ask(conversation, "beekeeper",
                    [json.dumps({"route": "role", "confidence": 0.9,
                                 "role": "beekeeper", "industry": ""}), "unused"])
        assert reply.refused is True


# ---------------------------------------------------------------------------
# The course the conversation then recommended
# ---------------------------------------------------------------------------

FULL = {"skill_python": "comfortable", "skill_sql": "basic",
        "skill_stats": "comfortable", "skill_ml": "basic",
        "skill_communication": "basic", "workload": "moderate"}


def placed(plan, code):
    return [q["label"] for q in plan["quarters"]
            for row in q["courses"] if row["code"] == code]


class TestConsumerBehaviorRunsInSpring:
    """The saved conversation put MGT 477 in Fall (second year) and told the
    student it was "offered FA, SP, WI". The scheduled section is MGT 477R,
    Spring only; the catalog said "varies" for every quarter."""

    def test_the_catalog_knows_it_as_a_spring_course(self):
        course = planner._catalog_by_id()["MGT 477"]
        assert planner.seasons_of(course) == ["SP"]
        assert all(o["term"] != "varies" for o in course["offerings"])

    @pytest.mark.parametrize("role_id", ["product-analyst", "marketing-analyst"])
    @pytest.mark.parametrize("track", ["11 month", "17 month"])
    def test_it_lands_in_spring_and_nowhere_else(self, role_id, track):
        plan = planner.build_for({**FULL, "track": track, "goals": [role_id]},
                                 frozenset())
        assert placed(plan, "MGT 477") == ["Spring"], (role_id, track)

    def test_the_recommendation_says_spring(self):
        body = orchestrator.curated_recommendation("product-analyst")
        assert "MGT 477 — Consumer Behavior** (4 units, offered SP)" in body

    def test_both_sections_never_share_a_plan(self):
        for role_id in load_careers():
            for track in ("11 month", "17 month"):
                plan = planner.build_for({**FULL, "track": track,
                                          "goals": [role_id]}, frozenset())
                titles = [row["title"] for q in plan["quarters"]
                          for row in q["courses"] if row["courseId"]]
                assert len(titles) == len(set(titles)), (role_id, track, titles)

    def test_the_twin_section_is_not_offered_as_a_swap(self):
        answers = {**FULL, "track": "11 month", "goals": ["product-analyst"]}
        plan = planner.build_for(answers, frozenset())
        spring = next(q for q in plan["quarters"] if q["key"] == "spring")
        slot = next(i for i, row in enumerate(spring["courses"])
                    if row["code"] == "MGT 477")
        options = planner.alternatives_for(plan, answers, "spring", slot)["options"]
        assert "MGT 477R" not in {o["courseId"] for o in options}

    def test_swapping_in_the_twin_is_refused(self):
        # The published shape (custom route) gives Spring two 4-unit slots.
        # Put MGT 477 in one by a swap, then ask the twin into the other.
        answers = {**FULL, "track": "11 month", "goals": ["product-analyst"],
                   "route": "custom"}
        plan = planner.build_for(answers, frozenset())
        spring = next(q for q in plan["quarters"] if q["key"] == "spring")
        first, second = [i for i, row in enumerate(spring["courses"])
                         if row["swappable"] and row["units"] == 4][:2]
        selections = planner.apply_swap(answers, {}, "spring", first, "MGT 477")
        assert selections["spring"][str(first)] == "MGT 477"
        with pytest.raises(ValueError, match="already in your Spring"):
            planner.apply_swap(answers, selections, "spring", second, "MGT 477R")

    def test_equivalents_are_symmetric(self):
        assert "MGT 477R" in planner.equivalents_of("MGT 477")
        assert "MGT 477" in planner.equivalents_of("MGT 477R")
        assert planner.equivalents_of("MGTA 458") == {"MGTA 458"}
