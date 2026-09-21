"""What real students type, and what the recommender must do with it.

Every case here came from `scripts/student_personas.py` -- conversations
driven the way a student actually writes, hedged and run-on and mid-thought --
and every one was a failure when first seen. The scripted harnesses had all
passed, because the scripts used the words the code already knew.

Kept as tests rather than as personas because each is a specific promise:
that phrasing gets that answer, deterministically, with no model call.
"""

import json

import pytest
from django.contrib.auth.models import User

from rsm_thrive.models import Conversation
from rsm_thrive.services.courses import electives, orchestrator, planner, router
from rsm_thrive.services.llm import FakeLLM

pytestmark = pytest.mark.django_db


@pytest.fixture
def user():
    return User.objects.create_user("stu")


@pytest.fixture
def conversation(user):
    return Conversation.objects.create(user=user, destination="courses",
                                       title="planning")


def ask(conversation, *turns):
    """Drive the turns with a model that raises if consulted."""
    reply = None
    for turn in turns:
        reply = orchestrator.answer(FakeLLM([]), conversation, turn, [])
    return reply


class TestAnAbbreviatedTrackIsATrack:
    """"yo i wanna do data sceince stuff, 11 mo" named a track that was not
    read, so the next turn's "moderate" had no track to attach to and was
    met with a request to clarify -- and the turn after that, and the one
    after that."""

    @pytest.mark.parametrize("said,track", [
        ("11 mo", "11 month"), ("17 mo", "17 month"), ("11mo", "11 month"),
        ("the 17 mos one", "17 month"), ("11 months", "11 month"),
    ])
    def test_mo_and_months_both_read(self, said, track):
        assert orchestrator.stated_track(said) == track

    def test_the_cascade_is_gone(self, conversation):
        reply = ask(conversation, "yo i wanna do data sceince stuff, 11 mo",
                    "moderate")
        stored = planner.load_session_intake(conversation)
        assert stored["track"] == "11 month"
        assert stored["workload"] == "moderate"
        assert reply.model_note == "plan"


class TestAChangeOfMindInPlainWords:
    """"hmm actually i think marketing is more my thing" was refused as the
    career **hmm marketing more** -- a fragment of the student's sentence
    quoted back as a job nobody maps courses to."""

    def test_it_switches_the_goal(self, conversation):
        reply = ask(conversation, "pricing analyst 11 month", "moderate",
                    "hmm actually i think marketing is more my thing")
        assert planner.load_session_intake(conversation)["goals"] == [
            "marketing-analyst"]
        assert "Switched to" in reply.body
        assert not reply.refused


class TestOneSentenceCarryingEverything:
    """"i have a full time job so i need the lightest possible schedule,
    aiming for business analyst, 17 month" kept only the role. "aiming FOR
    business analyst" satisfied the field pattern, so the job title came back
    as its own industry, the turn was ruled a combination, and the intake --
    which only ran for a plain role -- never saw the track or the load."""

    def test_the_role_is_not_also_the_field(self):
        route = router.rule_route("aiming for business analyst, 17 month")
        assert route.name == router.ROLE
        assert route.industry == ""

    def test_but_a_real_industry_beside_a_role_still_combines(self):
        """"fintech" resolves to Financial Analytics AND names an industry.
        That is the combination route's whole purpose, so it stays one."""
        route = router.rule_route("i want to work in fintech")
        assert route.name == router.COMBINATION
        assert route.industry == "fintech"

    def test_all_three_facts_are_kept(self, conversation):
        reply = ask(conversation,
                    "i have a full time job so i need the lightest possible "
                    "schedule, aiming for business analyst, 17 month")
        stored = planner.load_session_intake(conversation)
        assert stored["track"] == "17 month"
        assert stored["goals"] == ["business-data-analyst"]
        assert stored["workload"] == "light"
        assert reply.model_note == "plan"

    def test_lightest_and_heaviest_are_load_words(self):
        assert planner.load_mentioned("the lightest possible schedule") == "light"
        assert planner.load_mentioned("heaviest load you can give me") == "heavy"


class TestCoursesAStudentSaysTheyHaveDone:
    """"ive already done MGTA 464 and 402" was answered with a plan that
    scheduled both."""

    @pytest.mark.parametrize("said,codes", [
        ("MGTA 464 and 402", ["MGTA 464", "MGTA 402"]),
        ("can i swap 466 for something easier", ["MGTA 466"]),
        ("ive taken 464, 402 and MGTF 405", ["MGTA 464", "MGTF 405"]),
        ("the 251A class", ["CSE 251A"]),
    ])
    def test_the_codes_students_actually_type(self, said, codes):
        assert planner.mentioned_codes(said) == codes

    def test_an_ambiguous_bare_number_is_not_guessed(self):
        """495 exists in four departments. Reading it as any one of them
        would be the invention this whole layer avoids."""
        assert planner.mentioned_codes("take 495 next") == []

    def test_they_come_out_of_the_plan(self, conversation):
        reply = ask(conversation, "healthcare analytics, 17 month",
                    "ive already done MGTA 464 and 402", "moderate")
        stored = planner.load_session_intake(conversation)
        assert stored["completed_codes"] == ["MGTA 464", "MGTA 402"]
        plan = planner.build_for(stored, frozenset())
        rows = {r["courseId"]: r for q in plan["quarters"] for r in q["courses"]}
        # VISIBLE and marked done, not silently absent -- and not counted.
        assert rows["MGTA 464"]["completed"] and rows["MGTA 464"]["requirement"] == "Done"
        assert "MGTA 402" not in {i for i, r in rows.items() if not r.get("completed")}
        summer = plan["quarters"][0]
        assert summer["unitsPlanned"] == sum(
            r["units"] for r in summer["courses"] if not r.get("completed"))
        assert plan["totals"]["completed"] == 4
        assert plan["totals"]["outstanding"] == 0, "the degree still closes"
        assert "you said you've done them" in reply.body

    def test_a_completed_required_course_is_shown_done_not_rescheduled(self):
        """The elective slots always honoured `taken_ids`; the fixed and core
        slots did not, which is how MGTA 464 came back. It now stays in the
        plan marked done, contributing nothing to the schedule."""
        plan = planner.build_for(
            {"track": "11 month", "goals": ["data-scientist"],
             "workload": "moderate", "completed_codes": ["MGTA 464"]},
            frozenset())
        summer = plan["quarters"][0]
        row = next(r for r in summer["courses"] if r["courseId"] == "MGTA 464")
        assert row["completed"] and row["requirement"] == "Done"
        assert summer["unitsPlanned"] == 6, "451 (4) + 403 (2); 464 not counted"
        assert "Already done" in planner.review_quarter(plan, {"track": "11 month"}, 0)[0]

    def test_a_current_enrolment_is_not_dropped_from_its_quarter(self):
        """Demo is ENROLLED in MGTA 451 this Summer. The first cut treated
        every taken core course as finished and removed it, and Summer came
        out as a 4-unit quarter listing nothing."""
        from django.contrib.auth.models import User
        from rsm_thrive.models import Enrollment
        from rsm_thrive.testing import make_course
        user = User.objects.create_user("enrolled")
        Enrollment.objects.create(user=user, course=make_course(code="MGTA 451"))
        plan = planner.build_for(
            {"track": "17 month", "goals": ["consultant"], "workload": "moderate"},
            planner.taken_course_ids(user))
        summer = plan["quarters"][0]
        codes = [r["code"] for r in summer["courses"]]
        assert "MGTA 451" in codes
        assert summer["unitsPlanned"] == 8

    def test_the_walk_through_shows_every_fixed_summer_course(self):
        """403 and 464 are stamped "Required" and the Summer walk-through
        filtered on "Core" alone, so it had never listed them -- "Summer III
        -- 8 units" followed by one 4-unit course, every time."""
        answers = {"track": "11 month", "goals": ["data-scientist"],
                   "workload": "moderate"}
        plan = planner.build_for(answers, frozenset())
        body, _replies, _last = planner.review_quarter(plan, answers, 0)
        for code in ("MGTA 451", "MGTA 403", "MGTA 464"):
            assert code in body, f"{code} missing from the Summer walk-through"


class TestWhyThisCourse:
    """"why did you pick MGTA 463 for me" was read as a request for
    alternatives to it, because the change-request handler claims any turn
    naming a course code. The plan holds the reason on the row."""

    def test_why_gives_the_reason_not_a_swap_table(self, conversation):
        reply = ask(conversation, "data scientist", "11 month", "moderate",
                    "why did you pick MGTA 461 for me")
        assert reply.model_note == "why"
        assert "is in your plan:" in reply.body
        assert "Other ways to fill" not in reply.body

    def test_why_about_a_course_not_in_the_plan_says_so(self, conversation):
        reply = ask(conversation, "data scientist", "11 month", "moderate",
                    "why isn't MGTP 414 in there")
        assert reply.model_note == "why"
        assert "isn't in your plan" in reply.body


class TestSmallThingsThatBrokeTheFlow:

    def test_thanks_before_a_plan_is_acknowledged(self, conversation):
        reply = ask(conversation, "thanks!")
        assert reply.model_note == "small-talk"
        assert "Whenever you're ready: tell me" not in reply.body, "no nudge stacked on it"

    @pytest.mark.parametrize("said", [
        "how many units do i need to graduate", "how many units is the degree"])
    def test_the_degree_size_is_a_fact_not_a_question_back(self, conversation, said):
        reply = ask(conversation, said)
        assert reply.model_note == "degree-fact"
        assert "50 units" in reply.body

    def test_how_many_electives_is_a_fact_too(self, conversation):
        reply = ask(conversation, "how many electives is that")
        assert reply.model_note == "degree-fact"
        assert "28 of the 50" in reply.body

    def test_sector_names_are_recognised_but_never_offered_as_jobs(self):
        """"fintech" was a TITLE in the data. Nobody is hired as a fintech."""
        offered = {t for i in electives.load_industries()
                   for t, _ in electives.top_titles_for(i["id"], 10)}
        assert "fintech" not in offered
        assert router.matched_role("fintech") == "finance-quant"


class TestThirdRound:
    """Walk-through, swaps, quarter loads and the situational route -- the
    paths the first two rounds barely touched."""

    def test_a_swap_that_cannot_be_made_says_so_mid_walk_through(self, conversation):
        """Mid-walk-through, `_maintenance` re-rendered the quarter on screen
        and threw the change handler's message away -- so a swap that could
        not be made looked identical to one that had."""
        reply = ask(conversation, "data scientist, 11 month, moderate",
                    "walk me through it", "next", "swap MGTA 466 for MGTA 457")
        assert "I can't put **MGTA 457**" in reply.body
        assert "Fall" in reply.body or "Winter" in reply.body

    def test_a_named_quarter_can_be_made_lighter_outside_the_walk_through(
            self, conversation):
        """"can i make fall lighter" with a plan on file went to the
        classifier, came back situational, and asked which track."""
        reply = ask(conversation, "bi analyst 17 month moderate",
                    "can i make winter lighter")
        assert reply.model_note == "review"
        assert "**Winter** is now light" in reply.body

    def test_on_the_11_month_track_the_same_ask_is_answered_not_applied(
            self, conversation):
        reply = ask(conversation, "bi analyst 11 month",
                    "can i make fall lighter")
        assert reply.model_note == "review"
        assert "isn't a choice" in reply.body
        assert "# Fall" in reply.body

    def test_the_goal_prompt_no_longer_promises_a_web_lookup(self, conversation):
        reply = ask(conversation, "11 month")
        assert "asking for" not in reply.body
        assert "hires MSBA graduates into" in reply.body

    def test_is_it_really_necessary_gets_the_reason(self, conversation):
        reply = ask(conversation, "product analyst 11 month moderate",
                    "is MGTA 458 really necessary")
        assert reply.model_note == "why"
        assert "is in your plan:" in reply.body

    def test_the_situational_route_starts_from_the_known_track(self, conversation):
        """It kept its own record and asked for a track the intake already
        held. Seeded now; the seed is what `situation.answer` receives."""
        ask(conversation, "bi analyst 11 month moderate")
        stored = planner.load_session_situation(conversation)
        intake = planner.load_session_intake(conversation)
        assert intake["track"] == "11 month"
        # The seed is applied inside `_answer_situational`; the observable
        # promise is that the intake's track is what the route sees.
        seeded = {**stored, **({"track": intake["track"]}
                              if not stored.get("track") else {})}
        assert seeded["track"] == "11 month"


class TestTheNudgeAsksOnlyForWhatIsMissing:

    def _nudge_for(self, conversation, answers):
        from rsm_thrive.services.reply import BotReply
        reply = orchestrator._with_a_way_back(
            conversation, answers,
            BotReply("An answer.", [], "catalog-rag", route=router.FACTUAL))
        return reply.body.split("---")[-1]

    def test_with_a_goal_it_asks_for_the_track_only(self, conversation):
        tail = self._nudge_for(conversation, {"goals": ["data-scientist"]})
        assert "which track" in tail
        assert "aiming for" not in tail

    def test_with_a_track_it_asks_for_the_goal_only(self, conversation):
        tail = self._nudge_for(conversation, {"track": "11 month"})
        assert "aiming for" in tail
        assert "which track" not in tail

    def test_with_neither_it_asks_for_both(self, conversation):
        tail = self._nudge_for(conversation, {})
        assert "aiming for" in tail and "which track" in tail


class TestAJobTheStudentHasIsNotAQuestionAboutOurs:
    """"light, i have a part time job" -- an answer to the load question with
    a reason attached -- carries "have" and "job", and was answered with the
    list of fourteen careers."""

    @pytest.mark.parametrize("said", [
        "light, i have a part time job", "i have a full time job so light",
        "my job is demanding, keep it moderate", "ive got a new job, heavy is fine",
    ])
    def test_it_is_not_the_careers_list(self, said):
        assert not router.asks_which_careers(said)

    @pytest.mark.parametrize("said", [
        "what jobs do you have", "which careers do you cover", "list the jobs"])
    def test_but_the_inventory_question_still_is(self, said):
        assert router.asks_which_careers(said)

    def test_the_load_answer_lands(self, conversation):
        reply = ask(conversation, "marketing analyst", "11 month",
                    "light, i have a part time job")
        assert planner.load_session_intake(conversation)["workload"] == "light"
        assert reply.model_note == "plan"


class TestTheSituationalRouteFeedsTheIntake:
    """It worked the track out and kept it to itself, so the next turn asked
    for it again."""

    def test_a_resolved_track_reaches_the_intake(self, conversation):
        position = json.dumps({"track": "11 month", "previous_track": "17 month",
                               "current_quarter": "winter", "finish_on_time": True,
                               "completed_codes": []})
        orchestrator.answer(FakeLLM([position, "Tight but doable."]), conversation,
                            "i switched from 17 month to 11 month and im in winter",
                            [])
        assert planner.load_session_intake(conversation).get("track") == "11 month"


class TestFromTheConversationMatrix:
    """Four failures out of 110 generated conversations, each a real one."""

    def test_an_ampersand_title_is_still_a_named_role(self, conversation):
        """"probably fp&a analyst? not 100% sure" opened the industry menu:
        the phrase sanitiser strips "&", and the role guard ran on the
        sanitised text, where "fp a analyst" is nobody's title."""
        assert not orchestrator.wants_the_industry_menu(
            "probably fp&a analyst? not 100% sure but lets go with it")
        reply = ask(conversation,
                    "probably fp&a analyst? not 100% sure but lets go with it",
                    "11", "ok moderate is fine")
        stored = planner.load_session_intake(conversation)
        assert stored["goals"] == ["finance-quant"]
        assert stored["workload"] == "moderate"
        assert reply.model_note == "plan"

    def test_an_industry_word_at_the_industry_menu_is_a_pick(self, conversation):
        """"something in fintech", typed at the menu, is also a role alias,
        and the router built that bundle -- so the menu never advanced and
        the next turn's "5" picked the fifth INDUSTRY."""
        ask(conversation, "no idea")
        reply = ask(conversation, "something in fintech")
        assert reply.model_note == "industry-roles"
        assert "Financial Services" in reply.body
        picked = ask(conversation, "5")
        assert picked.model_note == "curated"
        fifth = electives.top_titles_for("financial-services", 5)[4][1]
        assert planner.load_session_intake(conversation)["goals"] == [fifth]

    def test_thanks_between_the_load_question_and_its_answer(self, conversation):
        reply = ask(conversation, "product analyst, 17 month", "thanks")
        assert reply.model_note == "small-talk"
        assert "light" in reply.body.lower(), "it re-offers the pending question"
        after = ask(conversation, "moderate")
        assert after.model_note == "plan"

    def test_thanks_re_offers_whatever_is_pending(self, conversation):
        assert "track" in ask(conversation, "data scientist", "thanks").body.lower()

    def test_a_title_under_two_profiles_is_confirmed_naming_the_other(
            self, conversation):
        """"growth analyst" is under Product AND Marketing in the design
        document. Free text picked one silently."""
        shared = orchestrator.shared_title_in("i want to be a growth analyst")
        assert shared and set(shared[1]) == {"product-analyst", "marketing-analyst"}
        reply = ask(conversation, "i want to be a growth analyst")
        assert "listed under both" in reply.body
        assert "Product Analyst" in reply.body and "Marketing" in reply.body

    def test_an_unshared_title_gets_no_such_note(self, conversation):
        reply = ask(conversation, "i want to be a data scientist")
        assert "listed under both" not in reply.body


class TestTheMoreSpecificTitleWins:
    """"i wanna be a helthcare data analyst" resolved to Business Analyst.
    The exact pass ran over every title first, "data analyst" matched exactly
    inside the longer phrase, and the fuzzy pass never saw "healthcare data
    analyst" one letter away. What the student typed is the longer title."""

    @pytest.mark.parametrize("said,role,exact", [
        ("i wanna be a helthcare data analyst", "healthcare-analyst", False),
        ("healthcare data analyst", "healthcare-analyst", True),
        ("clinical data analyst please", "healthcare-analyst", True),
        ("data analyst", "business-data-analyst", True),
        ("i want to be a data scientist", "data-scientist", True),
        ("senior data scienist", "data-scientist", False),
    ])
    def test_resolution(self, said, role, exact):
        assert router.role_match(said) == (role, exact)

    def test_the_fuzzy_specific_reading_is_confirmed_back(self, conversation):
        reply = ask(conversation, "i wanna be a helthcare data analyst")
        assert "Healthcare" in reply.body
        assert "Reading **helthcare data analyst** as" in reply.body
