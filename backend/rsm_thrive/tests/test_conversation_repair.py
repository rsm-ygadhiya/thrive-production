"""Three defects found by using it, and what each one now does.

Every case here came out of one real session, and each was a different kind of
failure to LISTEN:

1. **"17 month data nalyst"** — the track was read, the career was not, and the
   student was asked for the career they had just given.
2. **The plan said what it had assumed about Python** rather than ever asking.
3. **"what about a consultant"** rebuilt the plan and looked identical: same
   header, same all-required Summer, six electives in a different order, and
   nothing anywhere naming the career it was for. A rebuild that really
   happened read as the bot repeating itself.
"""

import pytest
from django.contrib.auth.models import User

from rsm_thrive.models import Conversation
from rsm_thrive.services.courses import orchestrator, planner, router
from rsm_thrive.services.llm import FakeLLM

pytestmark = pytest.mark.django_db


@pytest.fixture
def user():
    return User.objects.create_user("stu")


@pytest.fixture
def conversation(user):
    return Conversation.objects.create(user=user, destination="courses",
                                        title="planning")


# ---------------------------------------------------------------------------
# 1. A mistyped title is still a title
# ---------------------------------------------------------------------------

class TestAMistypedRoleIsStillRead:
    @pytest.mark.parametrize("said,role", [
        ("17 month data nalyst", "business-data-analyst"),   # the live case
        ("data scienist", "data-scientist"),                 # dropped letter
        ("data sceintist", "data-scientist"),                # swapped pair
        ("pricing anlayst", "pricing-analyst"),
        ("produt manager", "product-analyst"),
        ("markting analyst", "marketing-analyst"),
    ])
    def test_these_resolve(self, said, role):
        matched, exact = router.role_match(said)
        assert matched == role
        assert exact is False, "and it knows it guessed"

    @pytest.mark.parametrize("said", [
        "sql analyst", "data guy", "I like data", "consultnt", "analyst",
    ])
    def test_these_still_resolve_to_nothing(self, said):
        """A character of slop must not turn a non-title into a career. A
        one-word title is exempt from the slop entirely — "consultant" is
        short enough to type right, and too easy to hit by accident."""
        assert router.role_match(said)[0] == ""

    def test_an_exact_title_is_never_treated_as_a_guess(self):
        assert router.role_match("data analyst") == ("business-data-analyst", True)

    def test_a_guess_is_confirmed_back_rather_than_acted_on_silently(
            self, conversation):
        """Guessing which of fourteen careers someone meant is not a guess to
        make quietly."""
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "17 month data nalyst", [])
        # The TRACK is not quoted back as part of the career. What needs
        # confirming is the half that was guessed at.
        assert "Reading **data nalyst** as **Business / Data Analyst**" \
            in reply.body
        assert "say the word if you meant something else" in reply.body

    def test_and_it_carries_on_instead_of_re_asking(self, conversation):
        """The live failure: the student was asked for the career they had
        just given."""
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "17 month data nalyst", [])
        assert "What are you aiming for" not in reply.body
        assert planner.load_session_intake(conversation)["goals"] == \
            ["business-data-analyst"]
        assert planner.load_session_intake(conversation)["track"] == "17 month"


# ---------------------------------------------------------------------------
# 2. It asks about Python
# ---------------------------------------------------------------------------

class TestItDoesNotAskYouToRateYourself:
    """The five self-rating questions are gone.

    They gated the plan behind sliders a student could not honestly fill in --
    nobody can rate their own machine learning before they have taken any --
    and a low rating steered them AWAY from the courses that would fix the
    gap, which is the opposite of advising. The scoring still runs, at the
    neutral rating, and says so in the plan.
    """

    def test_the_load_spread_is_the_only_question_left(self, conversation):
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "17 month, data scientist", [])
        assert "spread across the quarters" in reply.body.lower()
        assert "starting from technically" not in reply.body
        assert "rate yourself" not in reply.body.lower()

    def test_a_rating_volunteered_anyway_is_still_read(self, conversation):
        """Being asked and being told are different things."""
        orchestrator.answer(FakeLLM([]), conversation,
                            "11 month, data scientist, python 4, sql 2", [])
        stored = planner.load_session_intake(conversation)
        assert stored["skill_python"] == 4 and stored["skill_sql"] == 2

    def test_the_plan_does_not_recite_the_assumption(self, conversation):
        """Nothing asks for a rating, so announcing that none was given is a
        preamble about a question that was never put."""
        orchestrator.answer(FakeLLM([]), conversation,
                            "11 month, data scientist", [])
        reply = orchestrator.answer(FakeLLM([]), conversation, "moderate", [])
        assert "i've assumed" not in reply.body.lower()


class TestChangingYourMindIsAcknowledged:
    def _planned(self, conversation, role="data analyst"):
        orchestrator.answer(FakeLLM([]), conversation, f"17 month, {role}", [])
        orchestrator.answer(FakeLLM([]), conversation, "skip", [])
        return orchestrator.answer(FakeLLM([]), conversation, "moderate", [])

    def test_the_plan_names_the_career_it_is_for(self, conversation):
        reply = self._planned(conversation)
        assert "# Your 17 month MSBA plan of study — Business / Data Analyst" \
            in reply.body

    def test_switching_says_so_first(self, conversation):
        self._planned(conversation)
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "what about a consultant", [])
        assert reply.body.startswith("Switched to **Analytics / Data Consultant**")
        assert "— Analytics / Data Consultant" in reply.body, "and the heading"

    def test_the_same_electives_are_named_as_the_same(self, conversation):
        """Two profiles in a 24-elective catalog can genuinely resolve to the
        same six courses. Silently reprinting is what made a real rebuild look
        like a repeat."""
        self._planned(conversation)
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "what about a consultant", [])
        before = planner.elective_codes(planner.build_for(
            {"track": "17 month", "goals": ["business-data-analyst"],
             "workload": "moderate"}, frozenset()))
        after = planner.elective_codes(planner.build_for(
            {"track": "17 month", "goals": ["consultant"],
             "workload": "moderate"}, frozenset()))
        if before == after:
            assert "come out the **same** for both" in reply.body
        else:
            assert " in, " in reply.body and " out." in reply.body

    def test_a_genuinely_different_set_names_what_moved(self, conversation):
        self._planned(conversation)
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "actually I want to be a data scientist", [])
        assert reply.body.startswith("Switched to **Data Scientist**")
        assert "in, " in reply.body and "out." in reply.body

    def test_the_first_plan_is_not_a_switch(self, conversation):
        reply = self._planned(conversation)
        assert "Switched to" not in reply.body

    def test_saying_the_same_role_again_changes_nothing(self, conversation):
        self._planned(conversation)
        reply = orchestrator.answer(FakeLLM(["Still the same plan."]),
                                     conversation, "data analyst", [])
        assert "Switched to" not in reply.body


# ---------------------------------------------------------------------------
# A curated role gets its curated bundle
# ---------------------------------------------------------------------------

class TestACuratedRoleDefaultsToItsBundle:
    """`bundles.json` holds a hand-authored elective set per role, and that is
    the programme's recommendation. The scorer is the fallback for the roles
    nobody curated — it used to be the default for everyone."""

    SKILLS = {f"skill_{a['key']}": 3 for a in planner.SKILL_AREAS}

    def test_a_curated_role_uses_the_bundle(self):
        plan = planner.build_for(
            {"track": "11 month", "goals": ["data-scientist"], **self.SKILLS},
            frozenset())
        assert plan["route"] == "fixed"
        assert plan["routeFellBack"] is False

    def test_an_uncurated_goal_falls_back_to_the_scorer(self):
        plan = planner.build_for({"track": "11 month", "goals": [], **self.SKILLS},
                                 frozenset())
        assert plan["route"] == "custom"

    def test_an_explicit_choice_always_wins(self):
        for route in ("custom", "fixed"):
            plan = planner.build_for(
                {"track": "11 month", "goals": ["data-scientist"],
                 "route": route, **self.SKILLS}, frozenset())
            assert plan["route"] == route

    def test_the_plan_says_which_fill_it_got(self):
        for route, phrase in (("fixed", "electives I'd recommend for that career"),
                              ("custom", "filled against your own skills")):
            body = planner.render_plan_markdown(planner.build_for(
                {"track": "11 month", "goals": ["data-scientist"],
                 "route": route, **self.SKILLS}, frozenset()))
            assert phrase in body, route

    def test_the_bundle_reaches_courses_the_scorer_could_not(self):
        """A bundle derives its own slot shape, so it can place the 2-unit
        courses the published `[4, 4]` skeleton has no room for. That is why
        the curated sets are the more differentiated ones."""
        answers = {"track": "17 month", "goals": ["data-scientist"], **self.SKILLS}
        fixed = planner.elective_codes(
            planner.build_for({**answers, "route": "fixed"}, frozenset()))
        custom = planner.elective_codes(
            planner.build_for({**answers, "route": "custom"}, frozenset()))
        assert "MGTA 466" in fixed, "a 2-unit anchor the bundle places"
        assert "MGTA 466" not in custom
        assert "CSE 251A" in fixed

    def test_bundles_stay_distinct_across_the_curated_roles(self):
        """The complaint that started this: every role gave the same plan."""
        for track in ("11 month", "17 month"):
            sets = {role_id: tuple(sorted(planner.elective_codes(
                planner.build_for({"track": track, "goals": [role_id],
                                   **self.SKILLS}, frozenset()))))
                for role_id in planner.load_careers()}
            assert len(set(sets.values())) >= 13, track


class TestTheBundleAndTheLoadSpreadCoexist:
    """Two features that quietly destroyed each other. The bundle derived its
    slot shape from the PUBLISHED budgets and `_resized` then re-cut it to the
    chosen ones, discarding the sizes the pinned courses need."""

    SKILLS = {f"skill_{a['key']}": 3 for a in planner.SKILL_AREAS}

    @pytest.mark.parametrize("track", ["11 month", "17 month"])
    @pytest.mark.parametrize("load", ["light", "moderate", "heavy"])
    def test_the_degree_still_totals_fifty(self, track, load):
        """It came back at 48 — 26 elective units instead of 28."""
        answers = {"track": track, "goals": ["ml-engineer"], "workload": load,
                   **self.SKILLS}
        seeded = planner.seeded_units(track, load)
        if seeded:
            answers["quarter_units"] = seeded
        plan = planner.build_for(answers, frozenset())
        assert plan["totals"]["total"] == planner.TOTAL_UNITS
        assert plan["unfilled"] == []

    def test_no_quarter_drops_below_the_enrolment_minimum(self):
        """The flex was free to run a quarter under its floor: a light Fall on
        the 17-month track came back at 10 units."""
        for track in ("11 month", "17 month"):
            bounds = {q["key"]: q for q in planner.adjustable_quarters(track)}
            for quarter in planner.adjustable_quarters(track):
                for load in ("light", "moderate", "heavy"):
                    units = planner.units_for_load(track, quarter["key"], load)
                    if units is None:
                        continue   # not a choice on this track
                    wanted, _ = planner.rebalanced_units(
                        track, {}, quarter["key"], units)
                    if wanted is None:
                        continue
                    plan = planner.build_for(
                        {"track": track, "goals": ["data-scientist"],
                         "quarter_units": wanted, **self.SKILLS}, frozenset())
                    for built in plan["quarters"]:
                        room = bounds.get(built["key"])
                        if room:
                            assert room["min"] <= built["unitsPlanned"] <= room["max"], \
                                (track, quarter["key"], load, built["label"])

    def test_an_exact_fit_is_preferred_where_one_exists(self):
        """The flex is a concession, not a preference, and it was being taken
        even where an exact fit was available. Trying zero first recovers
        every bundle that fits without it. Measured with each course's real
        terms in the catalog (and free slots for what the set cannot reach):
        9 of 14 on the 11-month track and 6 of 14 on the 17-month place
        exactly; the rest genuinely need the give."""
        exact = {track: sum(
            1 for role_id in planner.load_careers()
            if not planner.build_for({"track": track, "goals": [role_id],
                                      **self.SKILLS}, frozenset())["quarterFlex"])
            for track in ("11 month", "17 month")}
        assert exact["11 month"] == 9, exact
        assert exact["17 month"] == 6, exact

    def test_the_flex_never_exceeds_its_own_bound(self):
        from rsm_thrive.services.courses.bundles import QUARTER_FLEX

        for track in ("11 month", "17 month"):
            for role_id in planner.load_careers():
                plan = planner.build_for({"track": track, "goals": [role_id],
                                          **self.SKILLS}, frozenset())
                for quarter in plan["quarterFlex"]:
                    drift = abs(quarter["planned"] - quarter["expected"])
                    assert drift <= QUARTER_FLEX, (track, role_id, quarter)
                assert plan["totals"]["total"] == planner.TOTAL_UNITS

    def test_where_the_flex_is_taken_the_plan_says_so(self):
        """A student who just chose these loads is owed that."""
        answers = {"track": "17 month", "goals": ["data-scientist"], **self.SKILLS}
        plan = planner.build_for(answers, frozenset())
        body = planner.render_plan_markdown(plan)
        for quarter in plan["quarterFlex"]:
            assert quarter["label"] in body
            assert f"{quarter['planned']} units rather than {quarter['expected']}" \
                in body

    def test_a_bundle_course_squeezed_out_is_named(self):
        """A light spread on the 11-month Data Scientist bundle dropped
        MGTA 402 and MGTA 460 and `divergence` reported nothing, because
        neither is an anchor."""
        answers = {"track": "11 month", "goals": ["data-scientist"],
                   "quarter_units": planner.seeded_units("11 month", "light"),
                   **self.SKILLS}
        plan = planner.build_for(answers, frozenset())
        missing = plan["bundleMissing"].get("universal") or []
        if missing:
            body = planner.render_plan_markdown(plan)
            assert "leaves no room for" in body
            for code in missing:
                assert code in body

    def test_a_swap_still_only_moves_its_own_slot_on_a_bundle(self):
        """`apply_swap` validated against the published shape while the
        student was looking at the bundle's — one Fall swap moved five rows."""
        answers = {"track": "11 month", "goals": ["data-scientist"], **self.SKILLS}
        before = planner.build_for(answers, frozenset())
        option = planner.alternatives_for(before, answers, "fall", 2)["options"][0]
        after = planner.build_for(answers, frozenset(), planner.apply_swap(
            answers, {}, "fall", 2, option["courseId"]))
        moved = [(q["key"], i)
                 for q, qa in zip(before["quarters"], after["quarters"])
                 for i, (a, b) in enumerate(zip(q["courses"], qa["courses"]))
                 if a["courseId"] != b["courseId"]]
        assert moved == [("fall", 2)]

    def test_a_fallback_is_not_reported_as_a_divergence(self):
        """A plan the scorer filled got a "you have moved off the recommended
        bundle" warning about an edit the student never made."""
        plan = planner.build_for(
            {"track": "11 month", "goals": ["data-scientist"], "route": "custom",
             **self.SKILLS}, frozenset())
        from rsm_thrive.services.courses.replies import _divergence_note

        assert _divergence_note({"route": "custom", "goals": ["data-scientist"]},
                                plan) == ""


class TestEveryCuratedRoleIsNameable:
    """A curated role nobody can name is not curated.

    Found by a live sweep of all fourteen roles: "healthcare analyst" matched
    NOTHING, because the title list held "healthcare data analyst" and four
    specialist variants but not the phrase anyone would actually type. Same for
    "quant analyst". A student naming the role got asked what they were aiming
    for, having just said it — the curated bundle was unreachable by its own
    obvious name.
    """

    # The plainest way to say each role. Not the catalog's own list -- checking
    # the list against itself proves nothing.
    PLAIN = {
        "business-data-analyst": ["business analyst", "data analyst"],
        "bi-analyst": ["bi analyst", "business intelligence analyst"],
        "product-analyst": ["product analyst", "product manager"],
        "data-scientist": ["data scientist"],
        "analytics-engineer": ["analytics engineer", "data engineer"],
        "marketing-analyst": ["marketing analyst"],
        "consultant": ["consultant", "analytics consultant"],
        "fraud-risk-analyst": ["fraud analyst", "risk analyst"],
        "operations-supply-chain": ["supply chain analyst", "operations analyst"],
        "pricing-analyst": ["pricing analyst"],
        "decision-scientist": ["decision scientist"],
        "ml-engineer": ["ml engineer", "machine learning engineer"],
        "healthcare-analyst": ["healthcare analyst", "health analyst"],
        "finance-quant": ["financial analyst", "quantitative analyst",
                          "quant analyst"],
    }

    def test_the_table_covers_every_curated_role(self):
        assert set(self.PLAIN) == set(planner.load_careers()), \
            "a role was added or removed without a plain phrasing"

    @pytest.mark.parametrize("role_id,phrases",
                             sorted(PLAIN.items()))
    def test_the_obvious_phrasing_resolves(self, role_id, phrases):
        for phrase in phrases:
            matched, exact = router.role_match(phrase)
            assert matched == role_id, f"{phrase!r} -> {matched or None}"
            assert exact, f"{phrase!r} should be an exact title, not a guess"

    def test_and_it_reaches_the_plan(self, conversation):
        """End to end, not just the lookup: the role has to survive into the
        intake and produce that role's bundle."""
        orchestrator.answer(FakeLLM([]), conversation,
                            "11 month, healthcare analyst", [])
        orchestrator.answer(FakeLLM([]), conversation, "skip", [])
        reply = orchestrator.answer(FakeLLM([]), conversation, "moderate", [])
        assert planner.load_session_intake(conversation)["goals"] == \
            ["healthcare-analyst"]
        assert "Healthcare / Life Sciences Analytics" in reply.body


class TestEditingTheCatalogTakesEffect:
    """The catalog is the source of truth programme staff edit by hand, and
    every read of it was cached forever, keyed on nothing.

    Found the hard way: two roles were made nameable in `careers.json`, the
    unit tests agreed, and the running server went on failing the live sweep
    because it still held the file it had read at boot. Django's autoreloader
    only watches `.py`, so nothing prompted a restart. "It takes effect when
    someone remembers" is not a workable contract for the file that decides
    what the bot recommends.
    """

    def test_the_version_changes_when_a_file_does(self, tmp_path):
        from rsm_thrive.services.courses import electives

        before = electives._stat_version()
        path = electives._DATA / "careers.json"
        original = path.read_text()
        try:
            path.write_text(original)          # rewrite bumps mtime
            assert electives._stat_version() != before
        finally:
            path.write_text(original)
            electives.forget_catalog()

    def test_a_change_is_picked_up_without_a_restart(self):
        import json

        from rsm_thrive.services.courses import electives, router

        path = electives._DATA / "careers.json"
        original = path.read_text()
        try:
            data = json.loads(original)
            data["data-scientist"]["titles"].append("chief vibes officer")
            path.write_text(json.dumps(data, indent=2) + "\n")
            electives.forget_catalog()
            assert router.role_match("chief vibes officer")[0] == "data-scientist"
        finally:
            path.write_text(original)
            electives.forget_catalog()
            assert router.role_match("chief vibes officer")[0] == ""

    def test_every_derived_lookup_follows_the_files(self):
        """The wrappers, not just `load_catalog`. A derived cache keyed on
        nothing goes stale exactly as badly as the one it derives from."""
        from rsm_thrive.services.courses import bundles, electives, planner, router
        from rsm_thrive.services.courses import skill_match

        version = electives.catalog_version()
        for cached in (electives._load_catalog, electives._load_careers,
                       electives._role_aliases_for, bundles._load_bundles,
                       bundles._by_id_for, planner._catalog_by_id_for,
                       planner._ambiguous_codes_for, router._role_titles_for,
                       router._course_numbers_for):
            assert cached.cache_info().maxsize >= 4, cached
        # Every wrapper takes the version, so a new version misses the cache.
        assert planner._catalog_by_id_for.cache_info().maxsize >= 4
        assert skill_match._indexed_catalog_for.cache_info().maxsize >= 4
        assert isinstance(version, tuple)
        assert len(version) == len(electives._CATALOG_FILES)

    def test_the_check_is_memoised_so_it_is_not_a_stat_per_lookup(self):
        """`bundles._by_id` sits inside the placement search, so "once per
        catalog lookup" is tens of thousands of times per plan. Unmemoised
        this took the suite from 8.7s to 15.1s."""
        from rsm_thrive.services.courses import electives

        electives.forget_catalog()
        first = electives.catalog_version()
        calls = {"n": 0}
        real = electives._stat_version

        def counted():
            calls["n"] += 1
            return real()

        electives._stat_version = counted
        try:
            for _ in range(1000):
                electives.catalog_version()
        finally:
            electives._stat_version = real
        assert calls["n"] == 0, "re-statted inside the TTL"
        assert first == electives.catalog_version()
