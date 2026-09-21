"""Routing, and what each route actually produces.

The four-step interview is gone. What replaces it is one classifier and one
dispatch table, so these tests are organised the same way: what the router
decides, then what each route does with the decision.

Two properties matter more than the rest and are asserted repeatedly:

* **The curated mapping stays authoritative.** "I want to be a data scientist"
  resolves through `careers.json` without a model call, and the reply is the
  bundle from `bundles.json`. That is the quality floor and it must not become
  probabilistic.
* **Nothing states a fact it cannot ground.** Where a route has no catalog
  match, no retrieval and no ledger, it refuses and names no course.
"""

import json

import pytest
from django.contrib.auth.models import User

from rsm_thrive.models import Conversation, PlannerSession
from rsm_thrive.services.courses import electives, orchestrator, planner, router
from rsm_thrive.services.llm import FakeLLM

pytestmark = pytest.mark.django_db

FULL = {
    "track": "11 month", "goals": ["data-scientist"],
    "skill_python": "comfortable", "skill_sql": "basic",
    "skill_stats": "comfortable", "skill_ml": "basic",
    "skill_communication": "basic", "workload": "moderate",
}


@pytest.fixture
def user():
    return User.objects.create_user("stu")


@pytest.fixture
def conversation(user):
    return Conversation.objects.create(user=user, destination="courses",
                                       title="planning")


def with_plan(conversation, **overrides):
    """A conversation that already holds a finished plan.

    `asked` matters as much as `intake`. A plan is delivered automatically
    exactly once, and the marker is what records that it happened -- a session
    with a complete intake and no marker is a session that has not been shown
    its plan yet, and it will be shown one on the next turn whatever that turn
    says. Seeding one without the other describes a state the app cannot reach.
    """
    PlannerSession.objects.update_or_create(
        conversation=conversation,
        defaults={"intake": {**FULL, **overrides},
                  "asked": ["workload", "plan"]})
    return conversation


def classified(route, confidence=0.9, **extra):
    return json.dumps({"route": route, "confidence": confidence,
                       "role": extra.get("role", ""),
                       "industry": extra.get("industry", "")})


# ---------------------------------------------------------------------------
# The router
# ---------------------------------------------------------------------------

class TestRulesDecideTheObviousCases:
    """A round trip to classify "prerequisites for MGTA 452" is a round trip
    spent learning something a regex already knew."""

    @pytest.mark.parametrize("question,expected", [
        ("does 464 have prerequisites", router.FACTUAL),
        ("prerequisites for MGTA 452", router.FACTUAL),
        ("how many units is MGTA 461", router.FACTUAL),
        ("I want to be a data scientist", router.ROLE),
        ("data scientist in the food industry", router.COMBINATION),
        ("I've switched from 17-month to 11-month and I'm already in Winter, "
         "what do I take to finish on time?", router.SITUATIONAL),
    ])
    def test_these_never_reach_the_model(self, question, expected):
        route = router.rule_route(question)
        assert route is not None and route.name == expected
        # A rule is not a probability. Recording it as 1.0 would make the two
        # indistinguishable in the trace view that exists to tell them apart.
        assert route.confidence is None

    @pytest.mark.parametrize("question", [
        "what electives suit the aerospace industry",
        "what is the weather in san diego",
        "should I do a summer internship or a capstone",
    ])
    def test_these_need_a_judgement(self, question):
        assert router.rule_route(question) is None

    def test_a_bare_number_only_counts_when_it_is_a_real_course(self):
        """"does 464" used to match a four-letter word next to a number."""
        assert router.names_a_course("does 464 have prerequisites")
        assert not router.names_a_course("how many units is week 101")

    def test_the_role_comes_from_the_curated_table_not_a_model(self):
        assert router.matched_role("I want to be a data scientist") == "data-scientist"
        assert router.matched_role("i'd like to do pricing") == ""

    def test_the_longest_title_wins(self):
        """"product data scientist" and "data scientist" are both curated."""
        role = router.matched_role("I want to be a product data scientist")
        assert role == "product-analyst"


class TestTheModelDecidesTheRest:
    def test_a_confident_route_is_taken(self):
        route = router.classify(FakeLLM([classified("industry", 0.9,
                                                    industry="aerospace")]),
                                "what should I take for aerospace")
        assert route.name == router.INDUSTRY
        assert route.confidence == 0.9 and route.industry == "aerospace"

    def test_low_confidence_becomes_unclear_rather_than_a_guess(self):
        route = router.classify(FakeLLM([classified("role", 0.2)]),
                                "something about money")
        assert route.name == router.UNCLEAR

    def test_out_of_scope_is_believed_even_when_unsure(self):
        """A low-confidence "not about the programme" still saw nothing it
        recognised. Asking a clarifying question about the weather is worse
        than saying plainly that it is not something this tool does."""
        route = router.classify(FakeLLM([classified("out-of-scope", 0.3)]),
                                "who won the world cup")
        assert route.name == router.OUT_OF_SCOPE

    def test_an_invented_route_is_not_acted_on(self):
        route = router.classify(FakeLLM([classified("banana", 0.99)]), "hi")
        assert route.name == router.UNCLEAR

    def test_a_failed_classifier_says_so_rather_than_guessing(self):
        """It used to default to FACTUAL, which during a real outage produced
        "I don't have catalog or syllabus material that answers that" on every
        turn — a content refusal blaming the corpus for a billing problem."""
        route = router.classify(FakeLLM([]), "anything")
        assert route.name == router.DEGRADED
        assert "could not be reached" in route.why

    def test_a_role_the_model_names_is_still_resolved_by_the_catalog(self):
        """The model may name a role; only `careers.json` may resolve one."""
        route = router.classify(
            FakeLLM([classified("role", 0.9, role="data scientist")]),
            "help me get into modelling work")
        assert route.role_id == "data-scientist"

    def test_a_role_the_catalog_does_not_carry_is_kept_as_words(self):
        route = router.classify(
            FakeLLM([classified("role", 0.9, role="esports analyst")]),
            "I want to work in esports")
        assert route.role_id == "" and route.unmatched_role == "esports analyst"


# ---------------------------------------------------------------------------
# Route: a curated role
# ---------------------------------------------------------------------------

class TestACuratedRoleIsAnsweredDeterministically:
    """The quality floor. Fourteen roles have a hand-authored mapping and it
    stays authoritative -- no model chooses a course on this path."""

    def test_the_bundle_is_returned_with_no_model_call_at_all(self, conversation):
        # An exhausted FakeLLM raises if anything asks it for anything.
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "I want to be a data scientist", [])
        assert reply.route == router.ROLE
        assert reply.model_note == "curated"
        assert "MGTA 461" in reply.body and "CSE 251A" in reply.body

    def test_every_course_it_names_exists_in_the_catalog(self, conversation):
        """Every RECOMMENDED code, not every code the text happens to contain:
        a title like "Web Mining (cross-listed CSE 258)" mentions a number that
        is not itself a catalog row, and quoting a course's own title is not
        recommending a course."""
        import re

        from rsm_thrive.services.courses.electives import load_catalog

        codes = {course["code"] for course in load_catalog()}
        recommended = re.compile(r"- \*\*([A-Z]{2,4} \d{3}[A-Z]?) —")
        for role_id in planner.load_careers():
            body = orchestrator.curated_recommendation(role_id)
            named = set(recommended.findall(body or ""))
            assert named, role_id
            assert named <= codes, role_id

    def test_every_curated_role_has_a_recommendation(self):
        for role_id in planner.load_careers():
            assert orchestrator.curated_recommendation(role_id), role_id

    def test_the_goal_is_remembered_so_the_track_alone_finishes_it(self, conversation):
        orchestrator.answer(FakeLLM([]), conversation,
                            "I want to be a data scientist", [])
        assert planner.load_session_intake(conversation)["goals"] == ["data-scientist"]

    def test_it_asks_for_the_track_rather_than_assuming_one(self, conversation):
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "I want to be a data scientist", [])
        assert "11-month" in reply.body and "17-month" in reply.body


class TestARoleTheCatalogDoesNotCurate:
    """The web lookup that used to answer these is gone.

    It worked -- it produced a grounded, catalog-only set of courses. What it
    could not do was tell a student which kind of answer they were looking at:
    a curated bundle traces to the reviewed design document, a web-matched set
    traced to whatever a model read that morning, and both arrived in the same
    formatting. Two sources of truth in one voice is the defect that removed
    it. Full coverage of the replacement is in `test_industry_flow.py`.
    """

    def test_it_says_it_has_no_recommendation_and_names_no_course(
            self, conversation):
        reply = orchestrator.answer(
            FakeLLM([json.dumps({"route": "role", "confidence": 0.9,
                                 "role": "sommelier", "industry": "",
                                 "quarter": ""})]),
            conversation, "what electives suit a sommelier", [])
        assert reply.refused and reply.model_note == "no-profile"
        assert "MGTA" not in reply.body

    def test_but_a_vague_area_is_pointed_at_what_we_do_have(self, conversation):
        """"Something in tech" is not a job, and guessing one is the thing
        being avoided. It names an industry we rank, so it gets that ranking."""
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "something in tech", [])
        assert reply.model_note == "industry-roles"
        assert "Product Analyst" in reply.body


class TestAnIndustryIsAnsweredFromTheTaxonomy:
    """Six industries, each with its profiles ranked by the design document.
    Nothing here consults a model: FakeLLM([]) raises if anything does."""

    def test_a_covered_industry_lists_its_roles_in_rank_order(self, conversation):
        reply = orchestrator.answer(FakeLLM([]), conversation, "banking", [])
        assert reply.model_note == "industry-roles"
        assert reply.body.index("Fraud Analyst") < reply.body.index("BI Analyst")

    def test_an_industry_we_do_not_rank_gets_the_six_we_do(self, conversation):
        reply = orchestrator.answer(
            FakeLLM([json.dumps({"route": "industry", "confidence": 0.9,
                                 "role": "", "industry": "agriculture",
                                 "quarter": ""})]),
            conversation, "electives for the agriculture industry", [])
        assert reply.model_note == "industry-menu"
        assert "MGTA" not in reply.body, "no course for an industry we do not rank"


class TestACombinationKeepsTheRoleDeterministic:
    """A curated role plus an industry. The role fixes the electives; the
    industry says who the student is competing with. The order is the point --
    the deterministic half leads."""

    def test_the_curated_spine_comes_first_and_whole(self, conversation):
        reply = orchestrator.answer(
            FakeLLM([]), conversation,
            "I want to be a data scientist in fintech", [])
        assert reply.route == router.COMBINATION
        assert "MGTA" in reply.body
        # The bundle, then the ranking -- not the other way round.
        assert reply.body.index("MGTA") < reply.body.index("Financial Services")

    def test_it_says_where_that_role_ranks_in_that_industry(self, conversation):
        reply = orchestrator.answer(
            FakeLLM([]), conversation,
            "I want to be a data scientist in fintech", [])
        assert "#2 of 7" in reply.body, "Data Scientist is second in that table"

    def test_a_role_that_industry_does_not_rank_keeps_its_spine(
            self, conversation):
        reply = orchestrator.answer(
            FakeLLM([]), conversation,
            "I want to be a decision scientist in healthcare", [])
        assert "MGTA" in reply.body, "the role's own bundle survives"
        assert "isn't one this industry" in reply.body


class TestAFactualQuestionIsAnsweredFromTheCatalog:
    def test_the_catalog_row_is_what_the_model_is_given(self, conversation):
        fake = FakeLLM(["MGTA 464 is a 2-unit course."])
        reply = orchestrator.answer(fake, conversation,
                                    "how many units is MGTA 464", [])
        assert reply.route == router.FACTUAL
        system = fake.calls[0][0]
        assert "MGTA 464" in system
        assert "Never state a unit count" in system

    def test_nothing_to_ground_it_means_a_refusal(self, conversation):
        """No catalog row, no retrieved passage, no answer. The route is given
        rather than classified, so this tests the handler and not the
        classifier's own outage behaviour."""
        reply = orchestrator._answer_factual(
            FakeLLM([]), router.Route(router.FACTUAL),
            "what are the prerequisites for PHYS 999", [])
        assert reply.refused is True
        assert reply.model_note == "refusal"
        assert "advising" in reply.body.lower()


# ---------------------------------------------------------------------------
# Route: out of scope, and unclear
# ---------------------------------------------------------------------------

class TestDecliningPlainly:
    def test_out_of_scope_says_what_it_does_instead(self, conversation):
        reply = orchestrator.answer(
            FakeLLM([classified("out-of-scope", 0.9)]), conversation,
            "what is the weather in san diego", [])
        assert reply.refused is True
        lowered = reply.body.lower()
        assert "course planner" in lowered
        # Not "ask advising" as the whole answer: advising cannot help with the
        # weather either.
        assert not lowered.startswith("i don't have material")

    def test_a_fresh_unclear_turn_re_asks_the_opening_questions(self, conversation):
        """The three-shapes menu is the right answer to a question that was
        unclear. It is a cold answer to "hi", and it stacked the menu on top of
        the offer to plan -- two invitations for one word."""
        reply = orchestrator.answer(FakeLLM([classified("role", 0.1)]),
                                    conversation, "hmm", [])
        assert reply.route == router.UNCLEAR
        assert reply.refused is False
        assert "what you're aiming for after the programme" in reply.body
        assert "Whenever you're ready" not in reply.body, "not the offer twice"

    def test_a_greeting_costs_no_model_call_at_all(self, conversation):
        """Measured at 1.8s on this backend to learn something a set
        membership test knows -- and it is the first thing many students
        type."""
        reply = orchestrator.answer(FakeLLM([]), conversation, "hi there", [])
        assert reply.route == router.UNCLEAR
        # Greeted back, then the three ways in -- see TestAGreetingIsGreetedBack.
        assert reply.model_note == "greeting" and "your track" in reply.body

    def test_unclear_offers_the_three_shapes_once_something_is_known(
            self, conversation):
        orchestrator.answer(FakeLLM([]), conversation, "11 month", [])
        reply = orchestrator.answer(FakeLLM([classified("role", 0.1)]),
                                    conversation, "hmm", [])
        assert reply.body.count("\n- ") == 3, "a course, a goal, or their own plan"

    def test_with_a_plan_on_screen_small_talk_is_small_talk(self, conversation):
        """"thanks" is not out of scope and not unclear."""
        with_plan(conversation)
        reply = orchestrator.answer(FakeLLM(["Glad it helped!"]), conversation,
                                    "thanks!", [])
        assert reply.body == "Glad it helped!"
        assert "MGTA" not in reply.body, "small talk must never reprint the plan"


# ---------------------------------------------------------------------------
# Free text into a plan, with no interview
# ---------------------------------------------------------------------------

class TestAPlanIsBuiltFromWhatWasTyped:
    def test_it_asks_about_the_spread_and_then_plans(self, conversation):
        """ONE question now. The self-rating question that used to come first
        is gone -- a student cannot rate their own machine learning before
        they have taken any, and a low rating steered them away from the very
        courses that would fix it."""
        first = orchestrator.answer(
            FakeLLM([]), conversation,
            "I'm on the 17 month track and I want to be a data scientist", [])
        assert "spread across the quarters" in first.body.lower()
        assert "starting from technically" not in first.body, \
            "nothing asks a student to rate themselves any more"
        assert "MGTA" not in first.body, "it does not plan before it asks"

        reply = orchestrator.answer(FakeLLM([]), conversation, "moderate", [])
        assert reply.route == "plan"
        assert "Summer III" in reply.body and "Spring" in reply.body

    def test_a_volunteered_rating_is_still_used(self, conversation):
        """Being asked and being told are different things. Only the asking
        went; a student who says where they are is still heard."""
        orchestrator.answer(
            FakeLLM([]), conversation,
            "11 month, data scientist, python 4, sql 2, no ML", [])
        stored = planner.load_session_intake(conversation)
        assert stored["skill_python"] == 4 and stored["skill_ml"] == 1

    def test_everything_in_one_sentence_asks_nothing(self, conversation):
        reply = orchestrator.answer(
            FakeLLM([]), conversation,
            "11 month, data scientist, python 4 sql 4, heavy", [])
        assert reply.route == "plan", "nothing left to ask"
        assert "MGTA" in reply.body
        stored = planner.load_session_intake(conversation)
        assert stored["workload"] == "heavy" and stored["skill_python"] == 4

    def test_the_track_after_a_role_carries_on(self, conversation):
        orchestrator.answer(FakeLLM([]), conversation,
                            "I want to be a data scientist", [])
        orchestrator.answer(FakeLLM([]), conversation, "11 month", [])
        orchestrator.answer(FakeLLM([]), conversation, "skip", [])
        reply = orchestrator.answer(FakeLLM([]), conversation, "light", [])
        assert reply.route == "plan"
        assert "Winter" in reply.body

    def test_the_question_is_not_asked_twice(self, conversation):
        """An interview with no exit is the thing being removed."""
        first = orchestrator.answer(
            FakeLLM([]), conversation, "17 month, data scientist", [])
        assert "spread across the quarters" in first.body.lower()
        # A reply that does not answer it. The plan arrives anyway, with a note.
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "actually just show me the plan", [])
        assert "MGTA" in reply.body
        assert "published plan does" in reply.body, "and says what it assumed"
        assert "spread across the quarters" not in reply.body.lower()

    def test_the_plan_does_not_open_by_listing_its_assumptions(self, conversation):
        """It used to, and it made sense while the interview asked: a student
        who had answered four of five deserved to know the fifth was guessed.

        Nothing asks now, so it fired on every plan and named all five areas
        every time -- forty words above the plan saying only that a question
        the student was never asked went unanswered."""
        orchestrator.answer(
            FakeLLM([]), conversation,
            "17 month track, aiming to be a data scientist", [])
        reply = orchestrator.answer(FakeLLM([]), conversation, "moderate", [])
        head = reply.body.split("plan of study")[0].lower()
        assert "i've assumed" not in head
        assert "since you haven't said" not in head

    def test_but_a_course_above_the_assumed_level_still_says_so(self, conversation):
        """The disclosure moved, it did not go. `_stretch_notes` marks a
        stretch on the course's own row, where it can be acted on."""
        for said in ("data scientist", "11 month", "moderate",
                     "walk me through it", "next quarter", "next quarter"):
            reply = orchestrator.answer(FakeLLM([]), conversation, said, [])
        assert "Heads up" in reply.body

    def test_a_track_with_no_goal_asks_the_one_thing_it_needs(self, conversation):
        reply = orchestrator.answer(FakeLLM([]), conversation, "11 month", [])
        assert "aiming for" in reply.body
        assert "Step 1 of" not in reply.body, "the interview is gone"

    def test_a_plan_does_not_leak_into_the_next_conversation(self, user, conversation):
        orchestrator.answer(FakeLLM([]), conversation,
                            "11 month, data scientist", [])
        fresh = Conversation.objects.create(user=user, destination="courses",
                                            title="another")
        assert planner.load_session_intake(fresh) == {}

    def test_no_reply_ever_carries_buttons(self, conversation):
        """The click-based flow is gone; every answer is free text."""
        for question in ("I want to be a data scientist", "11 month", "skip",
                         "moderate", "walk me through it", "light"):
            reply = orchestrator.answer(FakeLLM([]), conversation, question, [])
            assert reply.quick_replies == [], question
            assert reply.form is None, question
        assert planner.opening_prompt()["quickReplies"] == []
        assert planner.opening_prompt()["form"] is None


# ---------------------------------------------------------------------------
# What a student does with a plan they already have
# ---------------------------------------------------------------------------

class TestMaintainingAPlan:
    """These are operations, not questions, and they are checked before the
    router: "next quarter" names nothing a classifier could recognise, and
    "swap 461 for 463" names two codes and would be routed as a factual
    question about them."""

    def test_asking_to_see_it_reprints_it(self, conversation):
        with_plan(conversation)
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "show me the plan", [])
        assert "Summer III" in reply.body and reply.route == "plan"

    def test_the_walk_through_starts_and_advances(self, conversation):
        with_plan(conversation)
        first = orchestrator.answer(FakeLLM([]), conversation,
                                    "walk me through it", [])
        assert "Summer III" in first.body
        second = orchestrator.answer(FakeLLM([]), conversation, "next", [])
        assert "Fall" in second.body

    def test_naming_one_planned_course_offers_alternatives(self, conversation):
        with_plan(conversation)
        plan = planner.build_for(FULL, frozenset())
        code = next(row["code"] for quarter in plan["quarters"]
                    for row in quarter["courses"] if row["swappable"])
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    f"change {code}", [])
        assert code in reply.body

    def test_an_impossible_swap_explains_the_rule(self, conversation):
        with_plan(conversation)
        plan = planner.build_for(FULL, frozenset())
        code = next(row["code"] for quarter in plan["quarters"]
                    for row in quarter["courses"] if row["swappable"])
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    f"swap {code} for MGTA 402", [])
        assert "can't" in reply.body.lower()


class TestANonMsbaCourseCarriesItsEnrolmentTerms:
    """Recommending a course a student cannot enrol in is a bad failure, and a
    silent one — it looks identical to a good recommendation until enrolment is
    refused. Whether the catalog's CSE list is authoritative is an open
    question with the programme office; until it is answered, every non-MGTA
    course says on what terms it can be taken."""

    def test_a_bundle_reaching_outside_the_msba_says_so(self):
        body = orchestrator.curated_recommendation("data-scientist")
        assert "CSE 251A" in body
        assert "CSE majors have enrolment priority" in body
        assert "16 of the 28 elective units" in body
        assert "Confirm any non-MGTA course with MSBA advising" in body

    def test_an_all_msba_bundle_does_not_carry_the_caveat(self):
        """Only two of the fourteen bundles stay inside MGTA. They say
        nothing, because there is nothing to warn about."""
        body = orchestrator.curated_recommendation("analytics-engineer")
        assert "Before you plan around these" not in body

    def test_a_consent_programme_says_consent(self):
        body = orchestrator.curated_recommendation("business-data-analyst")
        assert "enrolment by consent" in body

    def test_mgtf_is_named_as_the_mqf_not_the_mfin(self):
        """Rady runs both, and MGTF is the MQF's prefix.

        Its own page lists "EMBA, Full-Time MBA, FlexEvening/FlexWeekend MBA,
        MPAc, MQF and MFin" as separate programmes; MGTF 416 is titled "MQF
        Professional Seminar" in Rady's catalog, and the current fee pages for
        this prefix are headed "Master of Quantitative Finance". We had it as
        the MFin, and `import_syllabi` had already stamped that into all 23
        MGTF syllabi, so the wrong name was sitting in the corpus looking
        sourced.
        """
        label = electives.DEPARTMENT_LABELS["MGTF"][0]
        assert "Quantitative Finance" in label
        assert "MFin" not in label

    def test_every_prefix_in_every_bundle_has_stated_terms(self):
        from rsm_thrive.services.courses.electives import DEPARTMENT_LABELS
        from rsm_thrive.services.courses import bundles

        for role_id in planner.load_careers():
            bundle = bundles.bundle_for(role_id) or {}
            for layer in ("anchor", "differentiator", "universal"):
                for code in bundle.get(layer) or []:
                    prefix = code.split()[0]
                    assert prefix in DEPARTMENT_LABELS, f"{role_id}: {code}"


# ---------------------------------------------------------------------------
# It opens by asking, and comes back to asking
# ---------------------------------------------------------------------------

class TestItOpensWithQuestions:
    def test_the_opening_names_all_three_ways_in(self):
        """It used to open on "what are you aiming for after the programme?",
        which assumes an answer exists. For a student in their first quarter
        it often does not, and being asked a question you cannot answer is a
        worse start than being shown the way in."""
        body = planner.opening_prompt()["body"]
        assert "know the job" in body, "the fast path for a settled student"
        assert "know the field, not the job" in body, "the industry path"
        assert "no idea yet" in body, "the path for a student with no target"
        assert "11 month, data scientist" in body, "one line is enough"

    def test_the_opening_does_not_demand_the_track_up_front(self):
        """The track is asked once the job is settled -- every curated
        recommendation closes by asking for it -- so demanding it here made
        the opening a two-field form and asked the same question twice."""
        body = planner.opening_prompt()["body"]
        assert "I'll need your track" not in body
        # Still SHOWN, so the one-line shortcut stays discoverable.
        assert "11 month, data scientist" in body

    def test_the_opening_does_not_promise_a_web_lookup(self):
        """It read "name an industry and I'll work from what that field is
        asking for" -- the removed lookup's promise, outliving the lookup."""
        body = planner.opening_prompt()["body"]
        assert "what that field is asking for" not in body
        assert "hires MSBA graduates into" in body

    def test_and_says_you_can_ask_something_else_first(self):
        body = planner.opening_prompt()["body"]
        assert "come back to this" in body

    def test_it_still_carries_no_buttons(self):
        assert planner.opening_prompt()["quickReplies"] == []
        assert planner.opening_prompt()["form"] is None


class TestADetourComesBackToThePlan:
    """The surface opened by asking two questions. A student who answered
    neither should not have to remember that it was waiting."""

    def test_a_factual_answer_offers_to_plan(self, conversation):
        fake = FakeLLM(["MGTA 464 is a 2-unit course."])
        reply = orchestrator.answer(fake, conversation,
                                    "how many units is MGTA 464", [])
        assert "MGTA 464" in reply.body, "the question is answered first"
        assert "which track" in reply.body, "and then the offer"

    def test_an_out_of_scope_refusal_offers_too(self, conversation):
        reply = orchestrator.answer(
            FakeLLM([classified("out-of-scope", 0.9)]), conversation,
            "what is the weather", [])
        assert reply.refused is True
        assert "whole plan of study" in reply.body

    def test_it_stops_after_twice(self, conversation):
        """Once is an offer; on every turn it is a leaflet."""
        bodies = []
        for _ in range(4):
            fake = FakeLLM(["MGTA 464 is a 2-unit course."])
            bodies.append(orchestrator.answer(
                fake, conversation, "how many units is MGTA 464", []).body)
        assert "Whenever you're ready" in bodies[0]
        assert "Still happy to build that plan" in bodies[1]
        assert "ready" not in bodies[2] and "Still happy" not in bodies[2]
        assert "Still happy" not in bodies[3]

    def test_a_conversation_with_a_plan_is_not_nudged(self, conversation):
        with_plan(conversation)
        fake = FakeLLM(["MGTA 464 is a 2-unit course."])
        reply = orchestrator.answer(fake, conversation,
                                    "how many units is MGTA 464", [])
        assert "Whenever you're ready" not in reply.body

    def test_the_role_route_is_not_nudged_because_it_already_asks(
            self, conversation):
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "I want to be a data scientist", [])
        assert "11-month" in reply.body, "it asks for the track itself"
        assert "Whenever you're ready" not in reply.body, "not twice"


# ---------------------------------------------------------------------------
# One quarter, on its own
# ---------------------------------------------------------------------------

class TestPlanningOneQuarter:
    @pytest.mark.parametrize("question,expected", [
        ("what should I take in Winter?", "winter"),
        ("just plan my spring", "spring"),
        ("what am I taking in fall", "fall"),
        ("plan my winter please", "winter"),
    ])
    def test_these_route_to_one_quarter(self, question, expected):
        route = router.rule_route(question)
        assert route is not None and route.name == router.QUARTER
        assert route.quarter == expected
        assert route.confidence is None

    @pytest.mark.parametrize("question", [
        "does MGTA 464 run in winter",
        "I've switched to 11 month and I'm already in winter, what's left",
    ])
    def test_these_do_not(self, question):
        """A catalog question about a term, and a position in the programme,
        are different questions from "plan this quarter"."""
        route = router.rule_route(question)
        assert route is None or route.name != router.QUARTER

    def test_it_shows_that_quarter_and_not_the_others(self, conversation):
        with_plan(conversation)
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "what should I take in winter?", [])
        assert reply.route == router.QUARTER
        assert "Here's **Winter** on its own" in reply.body
        assert "# Winter" in reply.body
        assert "# Spring" not in reply.body and "# Summer III" not in reply.body

    def test_the_whole_plan_is_still_built_underneath(self, conversation):
        """A quarter cannot be planned in isolation: which electives are left
        in Winter depends on what Fall and Spring took."""
        with_plan(conversation)
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "just plan my winter", [])
        plan = planner.build_for(planner.load_session_intake(conversation),
                                 frozenset())
        winter = next(q for q in plan["quarters"] if q["key"] == "winter")
        for row in winter["courses"]:
            if row["courseId"]:
                assert row["code"] in reply.body

    def test_it_carries_the_load_question_for_that_quarter(self, conversation):
        with_plan(conversation, track="17 month")
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "what should I take in winter?", [])
        assert "How heavy should Winter be?" in reply.body

    def test_but_not_on_the_11_month_track(self, conversation):
        """Every quarter there carries the published load; nothing to ask."""
        with_plan(conversation)
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "what should I take in winter?", [])
        assert "# Winter" in reply.body
        assert "How heavy should" not in reply.body

    def test_it_offers_the_rest_and_the_whole_plan(self, conversation):
        with_plan(conversation)
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "what should I take in winter?", [])
        assert "next quarter" in reply.body
        assert "show me the plan" in reply.body

    def test_and_continuing_from_there_works(self, conversation):
        with_plan(conversation)
        orchestrator.answer(FakeLLM([]), conversation,
                            "what should I take in winter?", [])
        reply = orchestrator.answer(FakeLLM([]), conversation, "next quarter", [])
        assert "# Spring" in reply.body

    def test_no_goal_yet_asks_for_one_and_nothing_else(self, conversation):
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "what should I take in winter?", [])
        assert "plan **Winter** on its own" in reply.body
        assert "aiming for" in reply.body
        assert "MGTA" not in reply.body

    def test_a_quarter_that_track_does_not_have_says_which_it_does(
            self, conversation):
        with_plan(conversation)      # 11 month: no second Fall
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "just plan my second fall", [])
        assert "couldn't tell which quarter" in reply.body
        assert "Summer III" in reply.body and "Spring" in reply.body

    def test_it_is_not_gated_behind_the_load_spread_question(self, conversation):
        """That question earns its place before a whole plan of study; in front
        of one quarter it is an interruption."""
        orchestrator.answer(FakeLLM([]), conversation,
                            "I want to be a data scientist", [])
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "11 month — what should I take in winter?", [])
        assert "spread across the quarters" not in reply.body.lower()
        assert "# Winter" in reply.body


class TestAnOutageSaysItIsAnOutage:
    """Both backends went down at once — a lapsed Codex entitlement and an
    exhausted TritonAI budget — and the app spent ten turns insisting it had no
    course material. The corpus was fine. It could not reach the model."""

    def test_it_names_the_real_problem(self, conversation):
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "I want to do somthign in esports", [])
        assert reply.route == router.DEGRADED
        assert "can't reach my language model" in reply.body
        assert "not with your question or with the course material" in reply.body

    def test_it_is_not_counted_as_a_content_gap(self, conversation):
        """`refused` is the content backlog — the list of material to write.
        An outage in it would be a phantom entry nobody can act on."""
        reply = orchestrator.answer(FakeLLM([]), conversation, "hmm what", [])
        assert reply.refused is False
        assert reply.model_note == "degraded"

    def test_it_says_what_still_works(self, conversation):
        """Most of this app needs no model at all, and a student staring at an
        error should be told which half is still theirs."""
        reply = orchestrator.answer(FakeLLM([]), conversation, "hmm what", [])
        assert "11 month, data scientist" in reply.body
        assert "quarter by quarter" in reply.body

    def test_and_the_deterministic_half_really_does_still_work(self, conversation):
        """An exhausted FakeLLM raises on any call. Everything below happens
        anyway."""
        for said, expected in (("I want to be a data scientist", "MGTA 461"),
                               ("17 month", "spread across the quarters"),
                               ("moderate", "plan of study"),
                               ("walk me through it", "Summer III"),
                               ("next quarter", "Fall")):
            reply = orchestrator.answer(FakeLLM([]), conversation, said, [])
            assert expected in reply.body, said


class TestAGreetingIsGreetedBack:
    """Seen live: "hi" was answered with the opening's own paragraph in other
    words, so the second message read as the first one repeating itself. A
    greeting is answered as a greeting, and the three ways in follow in a
    different voice."""

    def test_hi_gets_a_hello_and_the_three_ways_in(self, conversation):
        fake = FakeLLM([])
        reply = orchestrator.answer(fake, conversation, "hi", [])
        assert fake.calls == [], "a greeting never needs the model"
        assert reply.model_note == "greeting"
        assert reply.body.startswith("Hi! Good to have you here.")
        for way in ("job in mind", "field, not the job", "No idea yet"):
            assert way in reply.body
        assert "what you're aiming for after the programme" not in reply.body
        assert "Whenever you're ready" not in reply.body, "no nudge under a menu of its own"

    def test_how_are_you_is_answered(self, conversation):
        reply = orchestrator.answer(FakeLLM([]), conversation, "how's it going?", [])
        assert reply.body.startswith("Doing well, thanks for asking!")

    def test_with_a_goal_on_file_it_asks_for_the_track(self, conversation):
        PlannerSession.objects.create(conversation=conversation,
                                      intake={"goals": ["data-scientist"]})
        reply = orchestrator.answer(FakeLLM([]), conversation, "hello", [])
        assert reply.model_note == "greeting"
        assert "11-month" in reply.body and "job in mind" not in reply.body
