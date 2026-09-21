"""The course recommender's front door: a field, or the industry menu.

A student arrives on this surface in one of two states, and the flow exists to
serve both:

* **They know what they want.** Name a role the mapping covers and they get
  its curated elective bundle, as before.
* **They do not.** "I don't know" is the commonest opening on an advising
  surface and the least useful thing to route as UNCLEAR. It now opens the
  industry menu: six industries, then that industry's ranked job titles, then
  the bundle for whichever they pick.

Both halves are curated. The web lookup that used to answer for roles and
industries the catalog had no profile for is GONE -- see
`test_no_profile_stops` below for what replaced it, and `advisor.py`'s
docstring for why. Everything a student is shown here traces to
`data/catalog/industries.json`, which carries Part III section 3.3 of the MSBA
Elective Recommender design document.
"""

import json
import re

import pytest
from django.contrib.auth.models import User

from rsm_thrive.models import Conversation
from rsm_thrive.services.courses import electives, orchestrator, planner, router
from rsm_thrive.services.llm import FakeLLM

pytestmark = pytest.mark.django_db

UNCOVERED = ["esports analyst", "sommelier", "wildlife biologist",
             "climate risk modeller"]


@pytest.fixture
def user():
    return User.objects.create_user("stu")


@pytest.fixture
def conversation(user):
    return Conversation.objects.create(user=user, destination="courses",
                                       title="planning")


def ask(conversation, question, replies=None):
    """One turn. With no `replies`, FakeLLM raises if the flow consults it --
    which is how the deterministic paths here prove they are deterministic."""
    llm = FakeLLM(replies=list(replies or []))
    return orchestrator.answer(llm, conversation, question, [])


def classified(route, **extra):
    """A classifier verdict, for the turns that genuinely need one.

    A job title nobody curates cannot be recognised without the model -- that
    is the whole reason the route exists -- so these tests supply the verdict
    rather than pretend the turn is free.
    """
    return json.dumps({"route": route, "confidence": 0.9, "role": "",
                       "industry": "", "quarter": "", **extra})


class TestTheTaxonomyIsWellFormed:
    """The data is the product here; a broken row is a broken recommendation."""

    def test_every_listed_role_exists_in_careers(self):
        careers = electives.load_careers()
        for industry in electives.load_industries():
            for role in industry["roles"]:
                assert role["id"] in careers, \
                    f"{industry['id']} lists unknown role {role['id']}"

    def test_every_curated_role_appears_in_some_industry(self):
        listed = {role["id"]
                  for industry in electives.load_industries()
                  for role in industry["roles"]}
        missing = set(electives.load_careers()) - listed
        assert not missing, f"unreachable from the menu: {sorted(missing)}"

    def test_no_industry_lists_the_same_role_twice(self):
        for industry in electives.load_industries():
            ids = [role["id"] for role in industry["roles"]]
            assert len(ids) == len(set(ids)), industry["id"]

    def test_every_industry_can_fill_a_top_ten(self):
        for industry in electives.load_industries():
            assert len(electives.top_titles_for(industry["id"], 10)) == 10, \
                industry["id"]


class TestTheTopTenSpansTheIndustry:
    """Depth-first would fill the list from one or two profiles. Technology's
    first five titles would then be product analyst, product data scientist,
    growth analyst, product manager and data scientist - product: one job
    written five ways, which tells a student nothing about the spread."""

    def test_the_first_titles_are_one_per_profile_in_rank_order(self):
        industry = electives.industry_by_id("technology-software")
        ranked = [role["id"] for role in industry["roles"]]
        got = [role_id for _, role_id
               in electives.top_titles_for("technology-software", len(ranked))]
        assert got == ranked

    def test_no_title_is_offered_twice(self):
        for industry in electives.load_industries():
            titles = [t for t, _ in electives.top_titles_for(industry["id"], 10)]
            assert len(titles) == len(set(titles)), industry["id"]

    def test_an_unknown_industry_offers_nothing_rather_than_guessing(self):
        assert electives.top_titles_for("agriculture") == []


class TestNotKnowingOpensTheMenu:
    """The answer to "I don't know" is a list, not a request to rephrase."""

    @pytest.mark.parametrize("said", [
        "i don't know", "no idea", "not sure", "show me the industries",
        "i have no idea what i want to do", "honestly i'm not sure yet",
        "dunno, haven't decided"])
    def test_it_shows_the_six_industries(self, conversation, said):
        reply = ask(conversation, said)
        for industry in electives.load_industries():
            assert industry["label"] in reply.body
        assert reply.model_note == "industry-menu"

    def test_the_industries_are_offered_as_quick_replies(self, conversation):
        reply = ask(conversation, "i don't know")
        assert [b["send"] for b in reply.quick_replies] == [
            i["label"] for i in electives.load_industries()]

    def test_it_does_not_need_the_model(self, conversation):
        """FakeLLM with no replies raises if consulted. The menu is data."""
        assert ask(conversation, "no idea").model_note == "industry-menu"


class TestPickingAnIndustryShowsItsRoles:

    def test_naming_an_industry_lists_its_titles_in_rank_order(self, conversation):
        reply = ask(conversation, "banking")
        expected = [orchestrator.title_case(t) for t, _
                    in electives.top_titles_for("financial-services", 10)]
        assert [b["send"] for b in reply.quick_replies] == expected
        assert reply.model_note == "industry-roles"
        # Ranked, and the ranking is the design document's.
        assert reply.body.index("Fraud Analyst") < reply.body.index("BI Analyst")

    def test_it_costs_no_classifier_call(self, conversation):
        """The six are a closed set, so recognising one is a lookup. FakeLLM
        with no replies raises if anything consults it."""
        assert ask(conversation, "retail").model_note == "industry-roles"

    def test_a_turn_naming_a_role_as_well_gets_both(self, conversation):
        """"Fintech" matches the Financial Analytics profile as well as the
        industry. That is the combination route's job, not the menu's: the
        role fixes the electives, the industry says who you compete with."""
        reply = ask(conversation, "i want to work in fintech")
        assert reply.route == router.COMBINATION
        assert "Financial Services" in reply.body

    @pytest.mark.parametrize("said,expected", [
        ("tech", "technology-software"),
        ("biotech in san diego", "healthcare"),
        ("consulting", "consulting"),
        ("e-commerce", "retail-cpg"),
        ("gaming", "other"),
    ])
    def test_the_words_students_actually_use_resolve(self, said, expected):
        assert (orchestrator.resolve_industry(said) or {}).get("id") == expected

    @pytest.mark.parametrize("said", [
        "esports analyst", "e-sports analyst", "e-sports", "bio-hacker"])
    def test_a_hyphen_does_not_end_a_word(self, said):
        """`\\b` treats a hyphen as a word boundary, so `\\bsports\\b` matched
        inside "e-sports" and a job we have no profile for came back as the
        catch-all industry -- and the hyphenated spelling is the commoner one."""
        assert orchestrator.resolve_industry(said) is None

    def test_but_a_real_sector_word_still_resolves(self):
        assert (orchestrator.resolve_industry("sports analytics") or {}).get("id") == "other"
        assert (orchestrator.resolve_industry("e-commerce") or {}).get("id") == "retail-cpg"

    def test_an_industry_we_do_not_cover_gets_the_menu(self, conversation):
        """Six is the whole of what the mapping covers, so the honest answer to
        a seventh is to say what the six are -- not to improvise."""
        reply = ask(conversation, "i want to go into agriculture",
                    [classified(router.INDUSTRY, industry="agriculture")])
        assert reply.model_note == "industry-menu"


class TestPickingARoleOffTheMenu:

    def test_it_builds_the_curated_bundle_for_that_role(self, conversation):
        ask(conversation, "i want to work in tech")
        reply = ask(conversation, "Analytics Engineer")
        assert reply.model_note == "curated"
        stored = planner.load_session_intake(conversation)
        assert stored.get("goals") == ["analytics-engineer"]

    def test_an_ambiguous_title_resolves_to_the_profile_that_offered_it(
            self, conversation):
        """"Growth analyst" is listed under BOTH Product and Marketing. Off the
        Retail menu it means Marketing, and free-text matching gets that wrong
        -- `router.matched_role` reaches Product first. The menu is the
        authority on what it offered."""
        assert router.matched_role("growth analyst") == "product-analyst"
        ask(conversation, "retail")
        assert orchestrator.role_from_industry_menu(
            conversation, "growth analyst") == "marketing-analyst"

    def test_a_title_is_only_honoured_after_a_menu_was_shown(self, conversation):
        assert orchestrator.role_from_industry_menu(
            conversation, "growth analyst") is None


class TestTheProfilesOwnNameIsRecognised:
    """A student who has read the programme material types the design
    document's name for a profile -- "product analytics", not "product
    analyst". Those matched nothing, so the shipped example prompt "Which
    electives suit product analytics?" came back as a refusal."""

    @pytest.mark.parametrize("said,expected", [
        ("product analytics", "product-analyst"),
        ("marketing analytics", "marketing-analyst"),
        ("fraud analytics", "fraud-risk-analyst"),
        ("business intelligence", "bi-analyst"),
        ("data science", "data-scientist"),
        ("financial analytics", "finance-quant"),
    ])
    def test_the_field_name_reaches_its_profile(self, said, expected):
        assert router.matched_role(said) == expected

    def test_every_alias_resolves_to_its_own_profile(self):
        for role_id, role in electives.load_careers().items():
            for alias in role.get("aliases") or []:
                assert router.matched_role(alias) == role_id, alias

    def test_aliases_are_recognised_but_never_offered(self):
        """`titles` is what the industry menu shows. "Product Analytics" is a
        field, not a job anyone is hired as, so it must not appear there."""
        offered = {t for industry in electives.load_industries()
                   for t, _ in electives.top_titles_for(industry["id"], 10)}
        for role in electives.load_careers().values():
            for alias in role.get("aliases") or []:
                assert alias not in offered, alias

    def test_the_shipped_example_prompt_gets_a_bundle(self, conversation):
        reply = ask(conversation, "Which electives suit product analytics?")
        assert reply.model_note == "curated"
        assert not reply.refused


class TestTheRefusalNamesTheJobNotTheQuestion:
    """It read "I don't have a course recommendation for **which electives
    suit product analytics**" -- the whole question handed back as though it
    were a job title."""

    @pytest.mark.parametrize("said,named", [
        ("which classes are best for an esports analyst", "esports analyst"),
        ("what courses should i take to become a sommelier", "sommelier"),
        ("i want to be a wildlife biologist", "wildlife biologist"),
    ])
    def test_only_the_job_survives_the_sentence(self, said, named):
        assert orchestrator._student_words(said) == named

    def test_no_course_word_reaches_the_refusal(self, conversation):
        """Asserted on the BOLDED name, not the sentence -- "I don't have a
        course recommendation for ..." says "course" itself, correctly."""
        reply = ask(conversation, "which electives should i take to be a sommelier",
                    [classified(router.ROLE, role="sommelier")])
        named = re.search(r"\*\*(.+?)\*\*", reply.body).group(1).lower()
        assert named == "sommelier", named


class TestEveryButtonIsRenderable:
    """The client keys its button row on `reply.send`.

    A list of bare strings therefore gave every button the key `undefined`,
    and Svelte aborts a keyed block on a duplicate key -- so the whole message
    list threw and the conversation rendered as an EMPTY PANE. The reply was
    correct the whole time; nothing between the orchestrator and the browser
    checked the shape. This is that check.
    """

    def _every_reply(self, conversation):
        return [ask(conversation, said) for said in
                ("i have no idea", "banking", "healthcare", "retail",
                 "consulting", "gaming", "tech")]

    def test_every_button_carries_a_label_and_something_to_send(
            self, conversation):
        for reply in self._every_reply(conversation):
            for button in reply.quick_replies:
                assert isinstance(button, dict), f"{button!r} is not a button"
                assert button.get("label"), button
                assert button.get("send"), button

    def test_no_two_buttons_share_a_send(self, conversation):
        """`send` is the key. Two buttons sharing one is the same crash."""
        for reply in self._every_reply(conversation):
            sends = [b["send"] for b in reply.quick_replies]
            assert len(sends) == len(set(sends)), sends

    def test_pressing_an_industry_button_lands_on_that_industry(
            self, conversation):
        """A shortened FACE must not cost the router the words it matches on."""
        for button in orchestrator.industry_buttons():
            resolved = orchestrator.resolve_industry(button["send"])
            assert resolved, button
            assert resolved["label"] == button["send"], button

    def test_pressing_a_role_button_lands_on_that_role(self, conversation):
        for industry in electives.load_industries():
            ask(conversation, industry["label"])
            for button in industry_role_buttons(conversation, industry):
                assert orchestrator.role_from_industry_menu(
                    conversation, button["send"]), button


def industry_role_buttons(conversation, industry):
    return orchestrator.industry_roles_reply(
        conversation, industry).quick_replies


class TestANumberedListTakesANumber:
    """The menu prints 1-6 and says "pick one". Measured live, a student typed
    "6" and got the generic "tell me what you're aiming for" fallback -- a
    list that will not take a number lied about being numbered."""

    @pytest.mark.parametrize("said", ["6", "#6", "6.", "number 6", "option 6"])
    def test_a_number_picks_that_industry(self, conversation, said):
        ask(conversation, "no idea")
        reply = ask(conversation, said)
        assert reply.model_note == "industry-roles"
        assert electives.load_industries()[5]["label"] in reply.body

    def test_then_a_number_picks_that_role(self, conversation):
        ask(conversation, "no idea")
        ask(conversation, "4")
        reply = ask(conversation, "1")
        assert reply.model_note == "curated"
        first = electives.top_titles_for("healthcare", 10)[0][1]
        assert planner.load_session_intake(conversation)["goals"] == [first]

    def test_a_number_off_the_end_is_not_a_pick(self, conversation):
        ask(conversation, "no idea")
        assert orchestrator.pick_by_number(conversation, "9") is None
        assert orchestrator.pick_by_number(conversation, "0") is None

    def test_a_number_with_no_menu_on_screen_means_nothing(self, conversation):
        """Otherwise "464" -- a course code -- would pick an industry."""
        assert orchestrator.pick_by_number(conversation, "2") is None

    def test_a_sentence_containing_a_number_is_not_a_pick(self, conversation):
        ask(conversation, "no idea")
        assert orchestrator.pick_by_number(
            conversation, "does MGTA 464 have prerequisites") is None


class TestAMenuAsksItsOwnQuestionOnly:
    """`_with_a_way_back` appends "tell me what you're aiming for and which
    track you're on" to detour routes. On a menu that is a SECOND, different
    question stacked under "pick one", and the student is left choosing which
    to answer. Measured live: the menu printed six industries, the nudge asked
    for a job title, and the next turn was neither."""

    def test_no_nudge_under_the_industry_menu(self, conversation):
        reply = ask(conversation, "no idea")
        assert "Whenever you're ready" not in reply.body
        assert "aiming for" not in reply.body

    def test_no_nudge_under_the_role_list(self, conversation):
        reply = ask(conversation, "banking")
        assert "Whenever you're ready" not in reply.body
        assert "aiming for" not in reply.body

    def test_the_menu_still_says_what_to_do_next(self, conversation):
        assert "Pick one" in ask(conversation, "no idea").body


class TestTheShortestAnswerToTheTrackQuestion:
    """Every curated recommendation closes with "which track you're on --
    11-month or 17-month", and the shortest honest answer is the number.

    Measured live: a student was asked exactly that, replied "17", and got
    "I want to make sure I answer the right question" -- the track patterns
    required the word "month" after the digits.
    """

    @pytest.mark.parametrize("said,track", [
        ("17", "17 month"), ("11", "11 month"), ("the 17", "17 month"),
        ("17.", "17 month"), ("seventeen", "17 month"), ("17mo", "17 month"),
        ("11-month", "11 month"), ("11 month, data scientist", "11 month"),
    ])
    def test_a_bare_number_is_the_track(self, said, track):
        assert orchestrator.stated_track(said) == track

    @pytest.mark.parametrize("said", [
        "I did 11 courses", "MGTA 464", "117", "1", "10", "does 464 run in fall"])
    def test_but_only_when_it_is_the_whole_answer(self, said):
        """A course code is three digits and a menu pick never exceeds ten, so
        nothing else on this surface is a lone 11 or 17."""
        assert orchestrator.stated_track(said) == ""

    def test_it_carries_through_to_the_plan(self, conversation):
        ask(conversation, "data scientist")
        reply = ask(conversation, "17")
        assert planner.load_session_intake(conversation)["track"] == "17 month"
        assert "spread across the quarters" in reply.body.lower()

    def test_it_works_after_the_industry_menu_too(self, conversation):
        ask(conversation, "no clue")
        ask(conversation, "4")
        ask(conversation, "1")
        ask(conversation, "17")
        assert planner.load_session_intake(conversation)["track"] == "17 month"


class TestComparingTwoThingsTheMenuOffered:
    """A student looking at ten numbered job titles asks how two of them
    differ before picking one. It used to reach the generic opening, which
    answers a question about two named things by asking what they are aiming
    for. Deterministic: the design document already says what each job is,
    what it gates on, and where it ranks."""

    def _to_roles(self, conversation):
        ask(conversation, "no idea")
        ask(conversation, "consulting")

    @pytest.mark.parametrize("said", [
        "whats the difference between the first two",
        "1 vs 2", "compare 1 and 2", "difference between 1 and 2",
        "how is analytics consultant different from business analyst",
    ])
    def test_the_shapes_students_use(self, conversation, said):
        self._to_roles(conversation)
        reply = ask(conversation, said)
        assert reply.model_note == "compare-roles", said

    def test_it_names_both_and_says_what_each_job_is(self, conversation):
        self._to_roles(conversation)
        body = ask(conversation, "difference between the first two").body
        assert "Consultant" in body and "Business Analyst" in body
        assert "The work" in body and "What it asks of you" in body

    def test_it_says_where_each_ranks_in_that_industry(self, conversation):
        self._to_roles(conversation)
        body = ask(conversation, "1 vs 2").body
        assert "#1 of 6" in body and "#2 of 6" in body

    def test_it_shows_where_the_electives_actually_differ(self, conversation):
        """Two careers that read differently can still share nine courses,
        and that is the thing worth knowing before agonising over them."""
        self._to_roles(conversation)
        body = ask(conversation, "1 vs 2").body
        assert "courses in common" in body
        assert "Only" in body

    def test_industries_compare_too(self, conversation):
        ask(conversation, "no idea")
        reply = ask(conversation, "whats the difference between the first two")
        assert reply.model_note == "compare-industries"
        assert "Technology" in reply.body and "Financial Services" in reply.body

    def test_a_bare_number_is_still_a_pick_not_a_comparison(self, conversation):
        self._to_roles(conversation)
        assert ask(conversation, "2").model_note == "curated"

    def test_too_vague_to_scope_is_left_alone(self, conversation):
        """"What's the difference" names nothing. Guessing which two would be
        exactly the invention this route exists to avoid."""
        self._to_roles(conversation)
        assert orchestrator.wants_a_comparison(
            conversation, "whats the difference") is None

    def test_a_comparison_with_no_menu_on_screen_means_nothing(self, conversation):
        assert orchestrator.wants_a_comparison(
            conversation, "whats the difference between the first two") is None

    def test_the_comparison_asks_one_question_not_two(self, conversation):
        self._to_roles(conversation)
        body = ask(conversation, "1 vs 2").body
        assert "Whenever you're ready" not in body


class TestNoProfileStopsRatherThanGuessing:
    """The web lookup that used to answer these is gone. It worked, but it
    answered a different question from the rest of this route: a curated
    bundle traces to the reviewed design document, a web-matched set traced to
    whatever a model read that morning, and both arrived in the same
    formatting. Two sources of truth in one voice is the defect."""

    @pytest.mark.parametrize("role", UNCOVERED)
    def test_it_says_plainly_that_it_has_no_recommendation(
            self, conversation, role):
        reply = ask(conversation, f"what should i take to become a {role}",
                    [classified(router.ROLE, role=role)])
        assert reply.refused
        assert reply.model_note == "no-profile"

    @pytest.mark.parametrize("role", UNCOVERED)
    def test_it_names_no_course(self, conversation, role):
        reply = ask(conversation, f"what should i take to become a {role}",
                    [classified(router.ROLE, role=role)])
        codes = [course["code"] for course in electives.load_catalog()]
        assert not [c for c in codes if c in reply.body], \
            "named a course for a role nothing maps courses to"

    @pytest.mark.parametrize("said", [
        "i want to be an esports analyst",
        "what electives should i take to become an esports analyst",
        "which courses should i take to become a sommelier",
    ])
    def test_a_purpose_clause_is_not_a_request_for_the_catalog(
            self, conversation, said):
        """"What should I take to BECOME X" asks about X, not about the shelf.

        It used to satisfy the catalog rule's two halves -- a course noun and
        "take" -- and answer a question about one job with all 86 courses. The
        role guard did not catch it, because that only knows the fourteen
        curated roles and this phrasing is commonest for the ones we do not.
        """
        reply = ask(conversation, said, [classified(router.ROLE)])
        assert reply.model_note != "catalog-overview", said

    def test_it_points_at_advising_and_at_the_menu(self, conversation):
        reply = ask(conversation, "what should i take to become a sommelier",
                    [classified(router.ROLE, role="sommelier")])
        assert "Appointments" in reply.body
        assert "show me the industries" in reply.body
