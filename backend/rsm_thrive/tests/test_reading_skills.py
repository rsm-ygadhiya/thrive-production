"""The reading skills: the model reads what the rules could not, and code decides.

Every regex on the course surface was grown from a live failure -- a phrasing
nobody had tested for, met with "I want to make sure I answer the right
question". These tests hold the layer that ends that: when the rules find
nothing, the `read_turn` skill reads the WHOLE turn against the question the
planner asked, and whatever it reads is validated against a closed set before
anything acts on it.

Three properties, asserted throughout:

* **Anything the rules already decide still never reaches the model.**
* **A value outside its closed set is dropped, never repaired or guessed.**
* **What the skill reads is acted on through the same code the rules reach**
  -- the same load question, the same plan, the same swap.
"""

import json

import pytest
from django.contrib.auth.models import User

from rsm_thrive.models import Conversation, PlannerSession
from rsm_thrive.services.courses import orchestrator, planner, router, situation, reading
from rsm_thrive.services.llm import FakeLLM

pytestmark = pytest.mark.django_db

FULL = {
    "track": "17 month", "goals": ["data-scientist"], "workload": "moderate",
}


@pytest.fixture
def user():
    return User.objects.create_user("stu")


@pytest.fixture
def conversation(user):
    return Conversation.objects.create(user=user, destination="courses",
                                       title="planning")


def said(**fields):
    """A `read_turn` answer, as the model would send it."""
    base = {"route": "unclear", "confidence": 0.8, "act": "answer",
            "role": "", "industry": "", "quarter": "", "curated": "",
            "curated_confidence": 0.0, "track": "", "load": "",
            "completed_codes": [], "skills": {}, "wants_menu": False,
            "plan_action": "", "courses": [], "route_choice": ""}
    return json.dumps({**base, **fields})


def ask(conversation, question, replies=None):
    return orchestrator.answer(FakeLLM(list(replies or [])), conversation,
                               question, [])


def with_plan(conversation, **overrides):
    PlannerSession.objects.update_or_create(
        conversation=conversation,
        defaults={"intake": {**FULL, **overrides}, "asked": ["workload", "plan"]})
    return conversation


# ---------------------------------------------------------------------------
# The skill files themselves
# ---------------------------------------------------------------------------

class TestTheSkillFiles:
    def test_both_readers_are_on_disk(self):
        assert {"read_turn", "answer_course_facts"} <= set(reading.available())

    def test_every_skill_loads_with_a_name_and_instructions(self):
        for name in reading.available():
            skill = reading.load(name)
            assert skill.name == name
            assert skill.description
            assert len(skill.body) > 200

    def test_every_prompt_either_bot_gives_the_model_is_a_skill_file(self):
        """Both bots keep their skills beside their code, and every one of
        them loads through the shared runner."""
        from rsm_thrive.services import skills as runner
        from rsm_thrive.services.resources import bot as resources_bot
        assert set(reading.available()) >= {
            "read_turn", "answer_course_facts",
            "explain_position", "small_talk"}
        assert set(runner.available(resources_bot.SKILLS_DIR)) == {
            "answer_question", "career_advice", "read_question"}
        for root in (reading.SKILLS_DIR, resources_bot.SKILLS_DIR):
            for name in runner.available(root):
                assert runner.load(root, name).name == name

    def test_the_factual_skill_takes_the_grounding_it_must_answer_from(self):
        skill = reading.load("answer_course_facts")
        assert skill.placeholders == {"catalog", "passages"}
        rendered = skill.prompt(catalog="Catalog entries:\n\nMGTA 464",
                                passages="Context passages: none matched.")
        assert rendered.rstrip().endswith("Context passages: none matched.")

    def test_read_turn_is_rendered_with_the_live_roster(self):
        from rsm_thrive.services.courses import understanding
        prompt = reading.load("read_turn").prompt(
            roster=understanding.roster(),
            industries=understanding.industry_roster())
        assert "<<" not in prompt
        assert "- data-scientist:" in prompt
        assert "- healthcare:" in prompt

    def test_a_placeholder_left_unrendered_is_an_error_not_a_prompt(self):
        """A model shown a literal `<<roster>>` answers from its own idea of
        what the profiles are, which is the source of careers this codebase
        refuses to have."""
        with pytest.raises(reading.SkillError):
            reading.load("read_turn").prompt()

    def test_an_unknown_skill_is_an_error(self):
        with pytest.raises(reading.SkillError):
            reading.load("read_minds")

    def test_the_context_is_labelled_as_ours_not_the_students(self):
        rendered = reading.render_context(
            {"question the planner asked last": "which track", "on file": ""})
        assert rendered.startswith("Context, from THRIVE's own records")
        assert "which track" in rendered
        assert "on file" not in rendered, "empty facts are left out"
        assert rendered.rstrip().endswith("The student's message:")

    def test_no_context_renders_nothing(self):
        assert reading.render_context({}) == ""
        assert reading.render_context(None) == ""


# ---------------------------------------------------------------------------
# Validators: the closed sets
# ---------------------------------------------------------------------------

class TestEveryValueIsCheckedAgainstAClosedSet:
    @pytest.mark.parametrize("said,expected", [
        ("17 month", "17 month"), ("17-month", "17 month"), ("17", "17 month"),
        ("seventeen months", "17 month"), ("11 month", "11 month"),
        ("the eleven month one", "11 month"),
        ("18 month", ""), ("", ""), (None, ""), ("a year", ""),
    ])
    def test_a_track_is_one_of_two(self, said, expected):
        assert reading.track_of(said) == expected

    @pytest.mark.parametrize("said,expected", [
        ("light", "light"), ("Moderate", "moderate"), ("HEAVY", "heavy"),
        ("lighter", ""), ("medium", ""), ("", ""), (None, ""),
    ])
    def test_a_load_is_one_of_three_and_is_not_repaired(self, said, expected):
        """"medium" is left to the skill to translate; the validator only
        accepts the three words the planner acts on."""
        assert reading.load_of(said) == expected

    def test_completed_codes_must_be_in_the_catalog(self):
        assert reading.codes_of(["MGTA 464", "MGTA 999", "banana"]) == ["MGTA 464"]
        assert reading.codes_of("mgta 464 and 402") == ["MGTA 464", "MGTA 402"]
        assert reading.codes_of(None) == []

    def test_a_course_named_by_title_resolves_only_when_unique(self):
        assert reading.courses_of(["Fraud Analytics"]) == ["MGTA 463"]
        assert reading.courses_of(["the fraud analytics one"]) == ["MGTA 463"]
        # "Business Analytics" is in several titles: a question, not a pick.
        assert reading.courses_of(["business analytics"]) == []
        assert reading.courses_of(["MGTA 463", "Fraud Analytics"]) == ["MGTA 463"]

    def test_skill_ratings_keep_only_known_areas_on_the_scale(self):
        assert reading.ratings_of({"python": 4, "sql": "1", "stats": "strong",
                                  "juggling": 5, "ml": 9}) == {
            "skill_python": 4, "skill_sql": 1, "skill_stats": 4}
        assert reading.ratings_of("python 4") == {}

    def test_confidence_is_clamped(self):
        assert reading.confidence_of("1.7") == 1.0
        assert reading.confidence_of(-2) == 0.0
        assert reading.confidence_of("high") == 0.0

    def test_flags_and_choices(self):
        assert reading.flag_of(True) and reading.flag_of("yes")
        assert not reading.flag_of("no") and not reading.flag_of(None)
        assert reading.route_choice_of("Fixed") == "fixed"
        assert reading.route_choice_of("bundle") == ""


# ---------------------------------------------------------------------------
# The router: a reading travels with the route
# ---------------------------------------------------------------------------

class TestTheRouteCarriesTheReading:
    def test_the_rules_still_decide_first_and_never_call_the_model(self):
        fake = FakeLLM([])
        route = router.classify(fake, "prerequisites for MGTA 452")
        assert route.name == router.FACTUAL and fake.calls == []

    def test_a_reading_puts_validated_values_on_the_route(self):
        route = router.classify(
            FakeLLM([said(track="the 17 month", load="heavy",
                             completed_codes=["MGTA 464", "XYZ 1"],
                             skills={"python": 4}, act="answer")]),
            "anything the rules cannot place")
        assert route.name == router.UNCLEAR
        assert route.act == "answer"
        assert route.learned == {"track": "17 month", "workload": "heavy",
                                 "completed_codes": ["MGTA 464"],
                                 "skill_python": 4}

    def test_values_outside_their_sets_are_dropped(self):
        route = router.classify(
            FakeLLM([said(track="18 month", load="lighter", act="shout",
                             plan_action="delete everything")]),
            "anything")
        assert route.learned == {} and route.action == {} and route.act == ""

    def test_the_context_reaches_the_model_above_the_students_words(self):
        fake = FakeLLM([said()])
        router.classify(fake, "17", {"question the planner asked last": "which track"})
        system, messages, json_mode = fake.calls[0]
        assert json_mode
        assert "curated profiles" in system
        assert messages[0]["content"].startswith("Context, from THRIVE's")
        assert messages[0]["content"].rstrip().endswith("17")

    def test_an_unreachable_model_is_reported_as_an_outage(self):
        route = router.classify(FakeLLM([]), "anything")
        assert route.name == router.DEGRADED

    def test_the_old_classifier_answers_still_parse(self):
        """Every existing test scripts the classifier with four keys. A reader
        that needed more would have broken all of them at once."""
        route = router.classify(
            FakeLLM([json.dumps({"route": "industry", "confidence": 0.9,
                                 "role": "", "industry": "aerospace"})]),
            "what electives suit the aerospace industry")
        assert route.name == router.INDUSTRY and route.industry == "aerospace"
        assert route.learned == {} and route.action == {}


# ---------------------------------------------------------------------------
# The orchestrator: acting on what was read
# ---------------------------------------------------------------------------

class TestAnAnswerTheRulesCannotReadStillLands:
    def test_the_longer_one_is_the_17_month_track(self, conversation):
        """The bot asked which track; the student did not say a number."""
        PlannerSession.objects.create(conversation=conversation,
                                      intake={"goals": ["data-scientist"]})
        reply = ask(conversation, "the longer one please",
                    [said(track="17 month", act="answer")])
        assert planner.load_session_intake(conversation)["track"] == "17 month"
        # And the flow carries on to the next question, as a typed "17" would.
        assert "how do you want the load spread" in reply.body

    def test_somewhere_in_between_is_a_moderate_load(self, conversation):
        PlannerSession.objects.create(
            conversation=conversation,
            intake={"goals": ["data-scientist"], "track": "17 month"},
            asked=["workload"])
        reply = ask(conversation, "somewhere in between i guess",
                    [said(load="moderate", act="answer")])
        assert planner.load_session_intake(conversation)["workload"] == "moderate"
        assert reply.route == "plan"
        assert "MGTA" in reply.body, "the plan was built"

    def test_not_too_heavy_is_light_not_heavy(self, conversation):
        """Found live, end to end: the loose word reader saw "heavy" in "not
        too heavy" and built a heavy plan. The skill reads the sentence, and
        its reading wins over the word match."""
        PlannerSession.objects.create(
            conversation=conversation,
            intake={"goals": ["data-scientist"], "track": "17 month"},
            asked=["workload"])
        reply = ask(conversation, "not too heavy please, i work part time",
                    [said(load="light", act="answer")])
        assert planner.load_session_intake(conversation)["workload"] == "light"
        assert "Spread **light**" in reply.body

    def test_the_word_match_is_still_the_fallback_when_the_model_is_down(
            self, conversation):
        PlannerSession.objects.create(
            conversation=conversation,
            intake={"goals": ["data-scientist"], "track": "17 month"},
            asked=["workload"])
        ask(conversation, "ok moderate load please", [])
        assert planner.load_session_intake(conversation)["workload"] == "moderate"

    def test_a_decline_is_an_answer_and_the_plan_follows(self, conversation):
        PlannerSession.objects.create(
            conversation=conversation,
            intake={"goals": ["data-scientist"], "track": "17 month"},
            asked=["workload"])
        reply = ask(conversation, "honestly i dont mind either way",
                    [said(act="decline")])
        assert reply.route == "plan" and "MGTA" in reply.body
        assert "workload" not in planner.load_session_intake(conversation)

    def test_a_question_states_nothing(self, conversation):
        """A what-if mentions a track without choosing it. Even if the skill
        reports one, a question must not put it on file."""
        PlannerSession.objects.create(conversation=conversation,
                                      intake={"goals": ["data-scientist"]})
        ask(conversation, "what if i went for the longer track?",
            [said(track="17 month", act="ask"), "It depends."])
        assert "track" not in planner.load_session_intake(conversation)

    def test_an_act_of_asking_is_refused_even_without_a_question_mark(
            self, conversation):
        PlannerSession.objects.create(conversation=conversation,
                                      intake={"goals": ["data-scientist"]})
        ask(conversation, "wondering about the longer track",
            [said(track="17 month", act="ask"), "It depends."])
        assert "track" not in planner.load_session_intake(conversation)

    def test_finished_courses_stated_in_passing_are_honoured(self, conversation):
        PlannerSession.objects.create(
            conversation=conversation,
            intake={"goals": ["data-scientist"], "track": "11 month"},
            asked=["workload"])
        # Named by TITLE, and by a title that is not also a career alias: a
        # turn with a code in it is the factual rule's ("464 is behind me"
        # has "is" and a course in it), and "fraud analytics" is a profile.
        # Both are the rules' to claim, and rightly so.
        reply = ask(conversation, "oh and i already did the web mining class",
                    [said(completed_codes=["web mining"], act="answer")])
        assert planner.load_session_intake(conversation)["completed_codes"] == ["MGTA 461"]
        assert "**MGTA 461**" in reply.body

    def test_i_dont_know_yet_gets_the_industry_menu(self, conversation):
        reply = ask(conversation, "idk really, still figuring it out",
                    [said(route="unclear", wants_menu=True, act="chat")])
        assert reply.model_note == "industry-menu"

    def test_but_not_when_a_goal_is_already_on_file(self, conversation):
        PlannerSession.objects.create(conversation=conversation,
                                      intake={"goals": ["data-scientist"]})
        reply = ask(conversation, "idk really",
                    [said(route="unclear", wants_menu=True, act="chat")])
        assert reply.model_note != "industry-menu"

    def test_a_situational_turn_leaves_its_track_to_the_situational_route(
            self, conversation):
        """That route reads its own position and syncs the track back. A
        second writer would race it."""
        fake = FakeLLM([said(route="situational", track="11 month",
                                confidence=0.9, act="answer"),
                        json.dumps({"track": "11 month", "previous_track": None,
                                    "current_quarter": "winter",
                                    "finish_on_time": True,
                                    "completed_codes": []}),
                        "Tight but doable."])
        reply = orchestrator.answer(fake, conversation,
                                    "i moved to the shorter programme and "
                                    "im mid way, will i make it", [])
        assert reply.route == router.SITUATIONAL


class TestThreeThingsALiveSweepFound:
    """Each of these was a real turn against the live model, with a plan or
    a pending question on screen, and each was answered with a rebuild the
    student had not asked for."""

    def test_a_what_if_over_a_finished_plan_changes_nothing(self, conversation):
        """"what if i had gone for the 17 month instead" moved an 11-month
        student to the 17-month track. The deterministic track reader had no
        question guard; now a question fills blanks and overwrites nothing."""
        with_plan(conversation, track="11 month")
        reply = ask(conversation, "what if i had gone for the 17 month instead",
                    [said(route="unclear", act="ask", track="17 month"),
                     "You'd have a second Fall to spread things over."])
        assert planner.load_session_intake(conversation)["track"] == "11 month"
        assert "17 month MSBA plan" not in reply.body

    def test_but_a_question_still_fills_a_blank(self, conversation):
        """"what should I take on the 11 month track as a data scientist" is
        a question that carries the whole intake. Blanks are filled."""
        reply = ask(conversation,
                    "what should i take on the 11 month track as a data scientist?")
        intake = planner.load_session_intake(conversation)
        assert intake["track"] == "11 month" and intake["goals"] == ["data-scientist"]

    def test_a_course_that_shares_its_name_with_a_profile_is_a_course(
            self, conversation):
        """"drop the web mining class and take fraud analytics instead"
        switched the career to Risk & Fraud, because "fraud analytics" is an
        alias in careers.json. With a plan on screen the skill decides, and
        a swap it reads wins over the goal switch."""
        with_plan(conversation)
        reply = ask(conversation,
                    "can i drop the web mining class and take fraud analytics instead",
                    [said(act="request", plan_action="swap",
                             courses=["MGTA 461", "MGTA 463"])])
        assert planner.load_session_intake(conversation)["goals"] == ["data-scientist"]
        assert "Switched to" not in reply.body
        assert reply.route == "plan"

    def test_and_a_plain_career_switch_still_never_calls_the_model(
            self, conversation):
        with_plan(conversation)
        reply = ask(conversation, "what about a consultant")
        assert "Switched to" in reply.body

    def test_a_question_about_the_load_options_is_answered_not_built_over(
            self, conversation):
        """"does heavy mean more classes?" was met with the plan, spread the
        assumed way. It is a question, and it has an answer."""
        PlannerSession.objects.create(
            conversation=conversation,
            intake={"goals": ["data-scientist"], "track": "17 month"},
            asked=["workload"])
        reply = ask(conversation, "does heavy mean more classes?",
                    [said(route="unclear", act="ask")])
        assert reply.body == orchestrator.LOAD_EXPLAINED
        assert "workload" not in planner.load_session_intake(conversation)
        assert not planner.session_has_asked(conversation, "plan"), \
            "the plan waits for the answer"

    def test_any_other_question_takes_its_route_and_the_plan_waits(
            self, conversation):
        PlannerSession.objects.create(
            conversation=conversation,
            intake={"goals": ["data-scientist"], "track": "17 month"},
            asked=["workload"])
        reply = ask(conversation, "is there a cover letter workshop?",
                    [said(route="out-of-scope", act="ask", confidence=0.9)])
        assert reply.route == router.OUT_OF_SCOPE
        assert not planner.session_has_asked(conversation, "plan")


class TestAnIndustryIsReadFromTheClosedListOfSix:
    def test_an_alias_on_disk_still_wins_without_the_reading(self, conversation):
        fake = FakeLLM([])
        reply = orchestrator.answer(fake, conversation, "i want to work in banking", [])
        assert reply.model_note == "industry-roles" and fake.calls == []
        assert "Financial Services" in reply.body
        assert "Reading **" not in reply.body, "an alias is not a reading"

    def test_words_no_alias_covers_are_read_and_confirmed_back(self, conversation):
        reply = ask(conversation, "id like to work at a hospital",
                    [said(route="industry", confidence=0.9,
                          industry="a hospital", curated_industry="healthcare",
                          curated_industry_confidence=0.9, act="request")])
        assert reply.model_note == "industry-roles"
        assert reply.body.startswith("Reading **a hospital** as **Healthcare")
        assert "say the word if you meant a different industry" in reply.body

    def test_a_hesitant_reading_gets_the_menu_not_a_guess(self, conversation):
        reply = ask(conversation, "id like to work at a hospital",
                    [said(route="industry", confidence=0.9,
                          industry="a hospital", curated_industry="healthcare",
                          curated_industry_confidence=0.4, act="request")])
        assert reply.model_note == "industry-menu"

    def test_an_id_not_in_the_table_is_thrown_away(self, conversation):
        reply = ask(conversation, "i want to work in agriculture",
                    [said(route="industry", confidence=0.9,
                          industry="agriculture", curated_industry="agri-food",
                          curated_industry_confidence=0.95, act="request")])
        assert reply.model_note == "industry-menu"

    def test_a_named_job_landing_on_the_catch_all_industry_is_refused(self, conversation):
        """The live sweep, on two models: "esports analyst" read as the
        industry route with "esports" placed in the catch-all, and answered
        with that list's ten titles. It names a job; the job is not covered;
        it is refused like any other."""
        reply = ask(conversation, "i want to be an esports analyst",
                    [said(route="industry", confidence=0.85, industry="esports",
                          curated_industry="other", curated_industry_confidence=0.9,
                          act="request")])
        assert reply.model_note == "no-profile" and reply.refused

    def test_but_a_job_function_in_a_real_industry_gets_that_industrys_roles(self, conversation):
        reply = ask(conversation, "i want to be an analyst at a hospital",
                    [said(route="industry", confidence=0.85, industry="a hospital",
                          curated_industry="healthcare", curated_industry_confidence=0.9,
                          act="request")])
        assert reply.model_note == "industry-roles" and "Healthcare" in reply.body

    def test_a_workplace_called_a_role_is_still_answered_as_an_industry(
            self, conversation):
        """Found live: "i want to work for a hedge fund" came back as the
        role route with no curated profile, and was refused as the career
        **hedge fund**. The same reading carried the industry; it wins."""
        reply = ask(conversation, "i want to work for a hedge fund",
                    [said(route="role", confidence=0.85, role="hedge fund",
                          industry="a hedge fund",
                          curated_industry="financial-services",
                          curated_industry_confidence=0.9, act="request")])
        assert reply.model_note == "industry-roles"
        assert "Financial Services" in reply.body
        assert not reply.refused

    def test_but_a_named_job_we_do_not_cover_is_still_refused(self, conversation):
        """The live sweep's rule: an uncovered JOB is refused, never redirected
        to the catch-all industry's titles. "esports analyst" names a job
        even when the model also reads "esports" as an industry."""
        reply = ask(conversation, "i want to be an esports analyst",
                    [said(route="role", confidence=0.9, role="esports analyst",
                          industry="esports", curated_industry="other",
                          curated_industry_confidence=0.9, act="request")])
        assert reply.model_note == "no-profile" and reply.refused
        assert "esports analyst" in reply.body

    def test_the_combination_route_uses_the_same_reading(self, conversation):
        """A role plus a field the aliases do not cover. The rules claim any
        turn naming a curated role, so the handler is exercised directly with
        the route the reading would have produced."""
        route = router.Route(router.COMBINATION, confidence=0.9,
                             role_id="data-scientist", industry="a hospital",
                             industry_id="healthcare")
        reply = orchestrator._answer_combination(
            FakeLLM([]), route, conversation, "data scientist at a hospital", [])
        assert reply.model_note == "curated+industry"
        assert "Healthcare" in reply.body and "#" in reply.body


class TestAMenuPickIsReadInAnyWords:
    def _show_roles(self, conversation, industry_id="healthcare"):
        industry = next(i for i in __import__("rsm_thrive.services.courses.electives",
                                              fromlist=["x"]).load_industries()
                        if i["id"] == industry_id)
        orchestrator.industry_roles_reply(conversation, industry)
        return industry

    def test_the_exact_shapes_never_reach_the_model(self, conversation):
        self._show_roles(conversation)
        fake = FakeLLM([])
        reply = orchestrator.answer(fake, conversation, "2", [])
        assert reply.model_note == "curated" and fake.calls == []

    def test_a_described_item_is_picked_by_position(self, conversation):
        from rsm_thrive.services.courses import electives
        self._show_roles(conversation)
        titles = electives.top_titles_for("healthcare", 10)
        reply = ask(conversation, "hmm the third one sounds like me",
                    [said(route="unclear", act="answer", menu_pick=[3])])
        assert reply.model_note == "curated"
        label = electives.load_careers()[titles[2][1]]["label"]
        assert label in reply.body

    def test_two_positions_and_a_compare_flag_compare_them(self, conversation):
        self._show_roles(conversation)
        reply = ask(conversation, "how do the first and the fourth differ really",
                    [said(route="unclear", act="ask", menu_pick=[1, 4],
                          menu_compare=True)])
        assert reply.model_note == "compare-roles"

    def test_industries_menu_too(self, conversation):
        orchestrator.industry_menu_reply(conversation)
        reply = ask(conversation, "the health one i think",
                    [said(route="unclear", act="answer", menu_pick=[4])])
        assert reply.model_note == "industry-roles"
        assert "Healthcare" in reply.body

    def test_a_position_off_the_end_is_dropped(self, conversation):
        orchestrator.industry_menu_reply(conversation)
        reply = ask(conversation, "the ninth one",
                    [said(route="unclear", act="answer", menu_pick=[9]),
                     "Which did you mean?"])
        assert reply.model_note != "industry-roles"

    def test_an_ambiguous_pick_asks_about_the_list_not_the_opening(
            self, conversation):
        """Found live: "the analyst one" on a list with four analyst titles
        read as unclear and fell through to the generic opening, which asked
        for a job title as though no list had been shown."""
        self._show_roles(conversation)
        reply = ask(conversation, "the analyst one i guess",
                    [said(route="unclear", confidence=0.3, act="answer")])
        assert reply.model_note == "menu-which"
        assert "more than one fits" in reply.body
        assert reply.quick_replies and "aiming for" not in reply.body

    def test_the_menu_is_numbered_in_the_context(self, conversation):
        self._show_roles(conversation)
        context = orchestrator.turn_context(
            conversation, planner.load_session_intake(conversation))
        assert context["menu on screen"].startswith("job titles in Healthcare")
        assert "1. " in context["menu on screen"] and "10. " in context["menu on screen"]

    def test_no_menu_no_pick(self, conversation):
        reply = ask(conversation, "the third one",
                    [said(route="unclear", act="answer", menu_pick=[3]),
                     "Nothing to pick from yet."])
        assert reply.model_note != "curated"


class TestTheResourcesBotsPromptsAreSkillFiles:
    def test_both_prompts_load_and_take_the_passages(self):
        from rsm_thrive.services import skills as runner
        from rsm_thrive.services.resources import bot
        assert runner.load(bot.SKILLS_DIR, "answer_question").placeholders == {"passages", "web"}
        assert runner.load(bot.SKILLS_DIR, "career_advice").placeholders == {"passages"}
        assert "system_prompt" not in __import__(
            "rsm_thrive.services.bot_config", fromlist=["x"]).bot_config("faq")

    def test_the_faq_answer_is_prompted_from_the_skill_file(self):
        from rsm_thrive.models import Document, DocumentChunk
        from rsm_thrive.services.embeddings import FakeEmbeddings
        from rsm_thrive.services.resources.bot import answer_faq
        doc = Document.objects.create(source="t:hb", title="MSBA Handbook",
                                      kind="policy", destinations=["resources"])
        [vector] = FakeEmbeddings().embed(["drop a course before week two"])
        DocumentChunk.objects.create(document=doc, seq=0, heading="Drops",
                                     text="drop a course before week two",
                                     embedding=vector)
        fake = FakeLLM(['{"in_scope": true, "search_terms": []}', "Before week two [1]."])
        answer_faq(fake, "drop a course before week two", [])
        system, _messages, _ = fake.calls[1]
        assert "using ONLY the material below" in system
        assert "[1] MSBA Handbook — Drops\ndrop a course before week two" in system
        assert "Web results: none were retrieved" in system


class TestAPlanOnScreenTakesInstructionsInAnyWords:
    def test_yeah_go_ahead_finalises(self, conversation):
        with_plan(conversation)
        reply = ask(conversation, "yeah go ahead, that's fine by me",
                    [said(act="accept", plan_action="finalise")])
        assert reply.model_note == "plan"
        assert reply.body == planner.finalised_markdown(
            planner.build_for(FULL, frozenset()))

    def test_lets_keep_going_advances_the_walk_through(self, conversation):
        with_plan(conversation)
        PlannerSession.objects.filter(conversation=conversation).update(
            review={"index": 0})
        reply = ask(conversation, "cool, lets keep going",
                    [said(act="request", plan_action="next")])
        assert planner.load_session_review(conversation)["index"] == 1
        assert reply.route == "plan"

    def test_make_fall_a_bit_easier_moves_that_quarter(self, conversation):
        with_plan(conversation)
        reply = ask(conversation, "could we make fall a bit easier",
                    [said(act="request", plan_action="set_load",
                             load="light", quarter="fall")])
        assert reply.route == "plan"
        assert "Fall" in reply.body

    def test_a_quarter_the_reading_missed_is_read_off_the_turn(self, conversation):
        """Measured live: the model returned the load and no quarter for "can
        we make fall a bit easier". The quarter is in the student's words and
        the names are a closed set, so the turn is checked before the whole
        plan is re-spread."""
        with_plan(conversation)
        reply = ask(conversation, "can we make fall a bit easier",
                    [said(act="request", plan_action="set_load", load="light")])
        assert "Fall" in reply.body
        assert planner.load_session_intake(conversation)["workload"] == "moderate", \
            "the overall preference is untouched; one quarter moved"

    def test_make_the_whole_thing_lighter_reseeds_the_spread(self, conversation):
        with_plan(conversation)
        reply = ask(conversation, "can you make the whole thing less intense",
                    [said(act="request", plan_action="set_load", load="light")])
        assert planner.load_session_intake(conversation)["workload"] == "light"
        assert reply.route == "plan"

    def test_a_course_named_by_title_is_explained(self, conversation):
        with_plan(conversation)
        plan = planner.build_for(FULL, frozenset())
        # A title with no course code in it, or the factual rule claims the
        # turn before any skill runs -- which is right, and not this test.
        row = next(r for q in plan["quarters"] for r in q["courses"]
                   if r.get("swappable") and r.get("code")
                   and not router.names_a_course(r["title"]))
        reply = ask(conversation, f"why is the {row['title'].lower()} one in there",
                    [said(act="ask", plan_action="explain",
                             courses=[row["title"]])])
        assert f"**{row['code']} — {row['title']}**" in reply.body

    def test_swapping_by_title_reaches_the_same_swap(self, conversation):
        with_plan(conversation)
        plan = planner.build_for(FULL, frozenset())
        quarter, row = next((q, r) for q in plan["quarters"] for r in q["courses"]
                            if r.get("swappable") and r.get("code")
                            and not router.names_a_course(r["title"]))
        slot = quarter["courses"].index(row)
        alternative = planner.alternatives_for(plan, FULL, quarter["key"], slot)
        options = alternative.get("options") or []
        if not options:
            pytest.skip("this slot has no alternative to swap in")
        incoming = options[0]["code"]
        reply = ask(conversation, f"swap out the {row['title'].lower()} one",
                    [said(act="request", plan_action="swap",
                             courses=[row["title"], incoming])])
        assert reply.body.startswith(f"Swapped **{row['code']}** for **{incoming}**")

    def test_no_plan_no_action(self, conversation):
        """With nothing on screen there is nothing to finalise; the turn
        falls through to the ordinary route."""
        reply = ask(conversation, "yeah go ahead",
                    [said(act="accept", plan_action="finalise")])
        assert reply.model_note != "plan"

    def test_a_turn_the_skill_does_not_place_is_still_small_talk(
            self, conversation):
        """The call count is unchanged: one reading, then the small-talk
        reply, exactly as before the skill existed."""
        with_plan(conversation)
        reply = ask(conversation, "my cat sat on the keyboard",
                    [said(route="out-of-scope", act="chat", confidence=0.9),
                     "Hope the cat is well."])
        assert reply.body == "Hope the cat is well."


# ---------------------------------------------------------------------------
# What the model is told about the conversation
# ---------------------------------------------------------------------------

class TestTheContextSaysWhatWasAsked:
    def test_nothing_on_file(self, conversation):
        context = orchestrator.turn_context(conversation, {})
        assert context["question the planner asked last"] == "none"
        assert context["on file"] == "nothing yet"
        assert context["plan on screen"] == "no"

    def test_a_goal_without_a_track_means_the_track_is_pending(self, conversation):
        context = orchestrator.turn_context(conversation, {"goals": ["data-scientist"]})
        assert "which track" in context["question the planner asked last"]
        assert "goal Data Scientist" in context["on file"]

    def test_the_load_question_is_pending_only_once_asked(self, conversation):
        answers = {"goals": ["data-scientist"], "track": "17 month"}
        before = orchestrator.turn_context(conversation, answers)
        assert "load" not in before["question the planner asked last"]
        planner.note_session_asked(conversation, "workload")
        after = orchestrator.turn_context(conversation, answers)
        assert "load spread" in after["question the planner asked last"]

    def test_a_plan_on_screen_is_listed_by_code(self, conversation):
        with_plan(conversation)
        context = orchestrator.turn_context(conversation, FULL)
        assert context["plan on screen"].startswith("yes: MGTA")
        assert "walk-through" not in context
        PlannerSession.objects.filter(conversation=conversation).update(
            review={"index": 1})
        context = orchestrator.turn_context(conversation, FULL)
        assert context["walk-through"].startswith("quarter on screen is")

    def test_a_situational_question_pending_is_named(self, conversation):
        planner.save_session_situation(conversation, {"awaiting": "quarter"})
        context = orchestrator.turn_context(conversation, {})
        assert "which quarter" in context["question the planner asked last"]


# ---------------------------------------------------------------------------
# read_position is a skill too
# ---------------------------------------------------------------------------

class TestThePositionIsReadByTheSameSkill:
    def test_a_position_is_read_off_the_turn_reading(self, user):
        """Was a second skill and a second model call on every situational
        turn; now the five fields sit on `read_turn`, validated the same way."""
        fake = FakeLLM([json.dumps({"track": "17", "previous_track": "11 month",
                                    "current_quarter": "winter",
                                    "finish_on_time": True,
                                    "completed_codes": ["MGTA 451"]})])
        said = situation.extract_position(fake, "im on 17 now, was 11, in winter",
                                          {"question the planner asked last": "x"})
        system, messages, _ = fake.calls[0]
        assert "curated profiles" in system, "the one planner reading skill"
        assert messages[0]["content"].startswith("Context, from THRIVE's")
        assert said["track"] == "17 month", "a bare number is still a track"
        assert said["previous_track"] == "11 month"
        assert said["said_quarter"] == "winter"
        assert said["completed_codes"] == ["MGTA 451"]

    def test_a_classified_situational_turn_reads_once(self, conversation):
        """The route that classified the turn already carries the position;
        the handler must not pay a second reading for it."""
        fake = FakeLLM([said(route="situational", confidence=0.9, act="answer",
                             track="11 month", current_quarter="winter",
                             finish_on_time=True),
                        "Tight but doable."])
        reply = orchestrator.answer(fake, conversation,
                                    "im on the 11 month, in winter now, will i finish?", [])
        assert reply.route == router.SITUATIONAL
        assert len(fake.calls) == 2, "one reading, one explanation"
        assert "Winter" in reply.body

    def test_an_unknown_track_is_dropped_not_guessed(self, user):
        fake = FakeLLM([json.dumps({"track": "18 month", "previous_track": None,
                                    "current_quarter": None,
                                    "finish_on_time": False,
                                    "completed_codes": []})])
        assert "track" not in situation.extract_position(fake, "whatever")

    def test_an_outage_reads_as_nothing_stated(self, user):
        assert situation.extract_position(FakeLLM([]), "anything") == {}

class TestThePlannerReadsOnTheMainModel:
    def test_read_turn_never_uses_the_fast_reader(self, conversation):
        """The live sweep caught gemini-3.5-flash reading "esports analyst"
        as a curated profile. Deciding which of fourteen careers a student
        gets stays on the main model; only opted-in skills use the reader."""
        class Split(FakeLLM):
            def __init__(self):
                super().__init__([said(route="out-of-scope", confidence=0.9, act="chat")])
                self.fast = FakeLLM([said(route="role", role="esports analyst",
                                          curated="data-scientist",
                                          curated_confidence=0.95)])
            def reader(self):
                return self.fast
        llm = Split()
        reply = orchestrator.answer(llm, conversation, "i want to be an esports analyst", [])
        assert llm.fast.calls == [] and len(llm.calls) == 1
        assert reply.route == router.OUT_OF_SCOPE


class TestACuratedMappingIsSaidNotAssumed:
    def test_a_title_filed_under_another_name_is_explained(self, conversation):
        """Seen live: "data engineer" opened "Analytics Engineer --" with no
        line between the two. The mapping is the design document's, not a
        guess, and it is said as such."""
        reply = ask(conversation, "data engineer")
        assert reply.model_note == "curated"
        assert reply.body.startswith(
            "**data engineer** roles are covered here by the **Analytics Engineer** profile")

    def test_the_profiles_own_name_needs_no_note(self, conversation):
        reply = ask(conversation, "data scientist")
        assert reply.model_note == "curated"
        assert "covered here by" not in reply.body and "Reading **" not in reply.body

    def test_a_misspelling_still_gets_the_reading_note_instead(self, conversation):
        reply = ask(conversation, "data scienist")
        assert reply.body.startswith("Reading **data scienist** as **Data Scientist**")


class TestAFieldWithNoJobIsAnIndustryNotAProfile:
    def test_a_curated_reading_with_no_job_words_is_not_acted_on(self, conversation):
        """Persona run: "i want to work in financial services" came back as
        the combination route with the Financial Analytics profile read off
        a field. No job was typed; the industry route answers."""
        reply = ask(conversation, "i want to work in financial services",
                    [said(route="combination", confidence=0.9, role="",
                          industry="financial services", curated="finance-quant",
                          curated_confidence=0.9, act="request")])
        assert reply.model_note == "industry-roles"
        assert "Financial Services" in reply.body
        assert "Financial Analytics" not in reply.body.split("\n")[0]

    def test_but_a_typed_job_still_gets_its_curated_reading(self, conversation):
        reply = ask(conversation, "id like to be an airline pricing person",
                    [said(route="role", confidence=0.9, role="airline pricing person",
                          curated="pricing-analyst", curated_confidence=0.9, act="request")])
        assert reply.model_note == "curated" and "Reading **" in reply.body


class TestBrokenJsonFromTheWriterIsSalvaged:
    def test_unescaped_quotes_in_the_reply_do_not_leak_braces(self):
        from rsm_thrive.services.llm import parse_llm_json
        raw = '{"answered": false, "reply": "Orientation Week happened [6], with the "Meet Your MSBA Team" resources [4]."}'
        parsed = parse_llm_json(raw)
        assert parsed["answered"] is False
        assert parsed["reply"].startswith("Orientation Week happened")
        assert '"Meet Your MSBA Team"' in parsed["reply"]

    def test_prose_is_still_prose(self):
        from rsm_thrive.services.llm import parse_llm_json
        assert parse_llm_json("just a sentence")["reply"] == "just a sentence"


class TestASectorWordInsideAJobIsNotAnIndustry:
    def test_sports_scientist_is_an_uncovered_job_not_the_catch_all_industry(self, conversation):
        """Persona run: "i want to be a sports scientist" was answered with the
        catch-all industry's roles, deterministically, because "sports" is one
        of its sectors. It names a job; the job is not covered; it is refused."""
        reply = ask(conversation, "hi! im an incoming international student, i want to be a sports scientist",
                    [said(route="role", confidence=0.9, role="sports scientist", act="request")])
        assert reply.model_note == "no-profile" and reply.refused

    def test_a_plain_sector_still_gets_the_catch_all_roles_without_the_model(self, conversation):
        fake = FakeLLM([])
        reply = orchestrator.answer(fake, conversation, "i want to work in sports", [])
        assert reply.model_note == "industry-roles" and fake.calls == []


class TestAOneWordAliasInAlifeStoryIsNotAGoal:
    def test_working_in_marketing_for_years_is_not_the_marketing_profile(self):
        q = "i've been working in marketing for 6 years and im switching into analytics"
        assert router.role_match(q) == ("", False)
        assert router.rule_route(q) is None, "left to the reading skill"

    def test_but_wanting_to_work_in_it_still_is(self):
        assert router.matched_role("i want to work in marketing") == "marketing-analyst"
        assert router.matched_role("i want to work in fintech") == "finance-quant"
        assert router.matched_role("marketing analytics please") == "marketing-analyst"

    def test_a_career_switch_is_not_a_track_switch(self):
        q = "i've been working in marketing for 6 years and im switching into analytics. is MGTA 479 offered in winter?"
        route = router.rule_route(q)
        assert route is not None and route.name == router.FACTUAL
        assert router.rule_route("i switched from 17 month to 11 month and im in winter").name == router.SITUATIONAL
        assert router.rule_route("im switching tracks to the 11 month, what's left").name == router.SITUATIONAL


class TestALoadWithAPlanActionMovesOneQuarterNotThePlan:
    def test_make_this_one_lighter_mid_walk_through(self, conversation):
        with_plan(conversation)
        PlannerSession.objects.filter(conversation=conversation).update(review={"index": 1})
        reply = ask(conversation, "make this one lighter",
                    [said(act="request", plan_action="set_load", load="light")])
        assert planner.load_session_intake(conversation)["workload"] == "moderate", \
            "the overall preference is untouched"
        assert reply.model_note == "review"


class TestAWorryIsNotAPosition:
    @pytest.mark.parametrize("q", [
        "ok so i might have messed up, i want to work in consulting? i really dont want to fall behind",
        "sorry i know i keep asking things but i want to be a aerospace engineer? i really dont want to fall behind",
    ])
    def test_fall_behind_is_not_a_season_plus_a_position(self, q):
        assert not router._is_situational(q)

    def test_a_course_question_with_a_worry_after_it_is_factual(self):
        for q in ["ok so i might have messed up, is MGTA 479 offered in winter? will that be a problem",
                  "ok so i might have messed up, is MGTA 479 offered in winter? i really dont want to fall behind",
                  "im kind of panicking, how is MGTA 459 graded? i really dont want to fall behind"]:
            assert router.rule_route(q).name == router.FACTUAL, q

    def test_a_course_inside_a_position_is_still_situational(self):
        for q in ["i've already done MGTA 464 and switched to 11 month, what's left",
                  "I switched to the 11-month track and I'm in Winter, does 464 still fit"]:
            assert router.rule_route(q).name == router.SITUATIONAL, q


class TestAnUnclearReadingThatNamesAJobIsAnUncoveredJob:
    def test_yoga_instructor_read_as_unclear_is_refused_by_name(self, conversation):
        reply = ask(conversation, "ok so i might have messed up, i want to be a yoga instructor? what happens if i get this wrong",
                    [said(route="unclear", confidence=0.4, role="yoga instructor", act="request")])
        assert reply.model_note == "no-profile" and reply.refused
        assert "yoga instructor" in reply.body

    def test_an_unclear_reading_with_no_job_still_asks(self, conversation):
        # ("not sure" would be the industry menu, deterministically; "hmm" alone reads as nothing.)
        reply = ask(conversation, "hmm well",
                    [said(route="unclear", confidence=0.3, act="chat")])
        assert reply.model_note == "unclear-opening"
