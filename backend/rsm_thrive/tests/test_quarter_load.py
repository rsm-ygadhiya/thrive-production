"""Light / moderate / heavy, asked twice and meaning two different things.

The original asked both at once, on a form, before the student had seen a
single course: five sliders distributing fifty units across four quarters. What
replaced it splits the question in two, because it always was two questions:

* **How demanding should the courses be?** Asked once, before the plan. It
  changes which electives get picked (`rank_electives` penalises a course whose
  own workload exceeds the preference) and changes no unit counts at all.
* **How many units should THIS quarter carry?** Asked during the walk-through,
  quarter by quarter, while the student is looking at the four courses the
  answer changes.

The constraint that shapes all of it: the degree is 50 units and Summer is
fixed, so a lighter quarter is a heavier one somewhere else. There is no such
thing as a globally light plan, and the tests below pin that down rather than
letting a 46-unit plan out of the door.
"""

import pytest
from django.contrib.auth.models import User

from rsm_thrive.models import Conversation, PlannerSession
from rsm_thrive.services.courses import electives, orchestrator, planner
from rsm_thrive.services.llm import FakeLLM

pytestmark = pytest.mark.django_db

FULL = {"track": "11 month", "goals": ["data-scientist"],
        "skill_python": "comfortable", "skill_sql": "basic",
        "skill_stats": "comfortable", "skill_ml": "basic",
        "skill_communication": "basic", "workload": "moderate"}


@pytest.fixture
def user():
    return User.objects.create_user("stu")


@pytest.fixture
def conversation(user):
    return Conversation.objects.create(user=user, destination="courses",
                                        title="planning")


def mid_walk(conversation, index=1, **overrides):
    """A conversation part-way through the walk-through."""
    PlannerSession.objects.update_or_create(
        conversation=conversation,
        defaults={"intake": {**FULL, **overrides},
                  "asked": ["workload", "plan"],
                  "review": {"index": index}})
    return conversation


# ---------------------------------------------------------------------------
# Reading the word
# ---------------------------------------------------------------------------

class TestReadingTheWord:
    @pytest.mark.parametrize("text,expected", [
        ("light", "light"), ("Heavy", "heavy"), ("medium", "moderate"),
        ("moderate", "moderate"), ("make it light", "light"),
        ("push me", "heavy"), ("normal", "moderate"),
    ])
    def test_these_are_answers(self, text, expected):
        assert planner.load_intent(text) == expected

    @pytest.mark.parametrize("text", [
        "is next quarter heavy?", "are these courses light?",
        "which one is the heavy option?", "swap MGTA 461 for MGTA 463",
        "next quarter", "finalise", "",
    ])
    def test_these_are_not(self, text):
        assert planner.load_intent(text) is None

    def test_a_bare_hedge_still_answers(self):
        """`is_question` treats "heavy?" as a hedged answer rather than a
        question, on the argued grounds that an answer starts with its own
        value where a question starts with an auxiliary. A student replying
        "heavy?" to "how heavy do you want Fall?" is answering it."""
        assert planner.load_intent("heavy?") == "heavy"

    def test_a_load_inside_an_intake_sentence_is_found(self):
        """"11 month, data scientist, heavy" names all three, and strict
        whole-message matching finds none of them."""
        assert planner.load_mentioned("11 month, data scientist, heavy") == "heavy"

    def test_but_only_where_the_turn_is_stating_intake_facts(self):
        """`load_mentioned` applied to any message would read "MGTA 456 is
        heavy" as a preference. `learned_from` only reaches for it on a turn
        that already stated a track or a goal."""
        assert orchestrator.learned_from("MGTA 456 is heavy") == {}
        assert orchestrator.learned_from("11 month and heavy") == {
            "track": "11 month", "workload": "heavy"}


# ---------------------------------------------------------------------------
# The arithmetic: 50 units, always
# ---------------------------------------------------------------------------

class TestTheDegreeTotalNeverMoves:
    def test_a_lighter_quarter_is_a_heavier_one_elsewhere(self):
        wanted, disturbed = planner.rebalanced_units("11 month", {}, "fall", 12)
        assert wanted == {"fall": 12, "winter": 16, "spring": 14}
        assert disturbed == []
        assert sum(wanted.values()) == planner.TOTAL_UNITS \
            - planner.fixed_quarter_units("11 month")

    def test_every_distribution_it_produces_is_schedulable(self):
        for track in ("11 month", "17 month"):
            for quarter in planner.adjustable_quarters(track):
                for load in ("light", "moderate", "heavy"):
                    units = planner.units_for_load(track, quarter["key"], load)
                    if units is None:
                        continue   # not a choice on this track
                    wanted, _ = planner.rebalanced_units(
                        track, {}, quarter["key"], units)
                    if wanted is None:
                        continue
                    assert not planner.quarter_units_problems(track, wanted), \
                        (track, quarter["key"], load, wanted)

    def test_units_move_two_at_a_time(self):
        """`_resized` cuts a budget into 4-unit slots plus a 2-unit remainder,
        so an odd budget leaves a hole no course in the catalog can fill."""
        for track in ("11 month", "17 month"):
            for quarter in planner.adjustable_quarters(track):
                units = planner.units_for_load(track, quarter["key"], "heavy")
                if units is None:
                    continue   # not a choice on this track
                wanted, _ = planner.rebalanced_units(
                    track, {}, quarter["key"], units)
                if wanted is None:
                    continue
                assert all(value % 2 == 0 for value in wanted.values()), wanted

    def test_an_impossible_load_is_refused_rather_than_approximated(self):
        """The 17-month second Fall cannot take 8 units: the three quarters
        before it are already at their 12-unit floors."""
        wanted, _ = planner.rebalanced_units("17 month", {}, "fall-two", 8)
        assert wanted is None

    def test_a_load_is_offered_at_the_heaviest_value_it_can_reach(self):
        """A quarter's own enrolment cap is not the real cap. On the 17-month
        track every quarter caps at 18 units and none can reach it: 18 in Fall
        leaves 24 for the other three, whose floors add up to 26. Reading the
        cap literally offered that Fall only "light", so the heaviest thing the
        student could have was the one option they were not shown."""
        assert planner.units_for_load("17 month", "fall", "heavy") == 16
        # And on the 11-month track there is no "heavy" to offer at all.
        assert planner.units_for_load("11 month", "fall", "heavy") is None
        offered = {option["value"]: option["units"]
                   for option in planner.load_options_for("17 month", "fall-two")}
        assert offered == {"light": 2, "moderate": 4, "heavy": 6}

    def test_a_quarter_with_no_lighter_version_offers_no_duplicate(self):
        """The 17-month Fall's published load IS its floor, so "light" and
        "moderate" are the same number and offering both is not a choice."""
        options = planner.load_options_for("17 month", "fall")
        assert len({option["units"] for option in options}) == len(options)

    def test_an_earlier_choice_is_not_silently_undone(self):
        """Answering "light" for Winter used to push Fall back off the "light"
        the student had picked one quarter earlier, with nothing saying so."""
        after_fall, _ = planner.rebalanced_units("11 month", {}, "fall", 12)
        after_winter, disturbed = planner.rebalanced_units(
            "11 month", after_fall, "winter", 12, pinned={"fall"})
        assert after_winter["fall"] == 12, "their Fall choice survived"
        assert disturbed == []

    def test_and_when_it_would_have_to_be_undone_it_is_refused(self):
        """All three light is 36 units against 42 required, so the third
        request cannot be honoured without moving the first two. Refused and
        named, rather than done silently."""
        chosen = {"fall": 12, "winter": 12, "spring": 18}
        wanted, blockers = planner.rebalanced_units(
            "11 month", chosen, "spring", 12, pinned={"fall", "winter"})
        assert wanted is None
        assert blockers == ["fall", "winter"], "and it names them"

    def test_a_quarter_fixed_by_earlier_choices_offers_nothing(self):
        chosen = {"fall": 12, "winter": 12, "spring": 18}
        assert planner.load_options_for("11 month", "spring", chosen,
                                        {"fall", "winter"}) == []


# ---------------------------------------------------------------------------
# Asked once, before the plan
# ---------------------------------------------------------------------------

class TestAPlacedCourseSaysWhoseProgrammeItIs:
    """The swap list flagged its options and the placed course did not, which
    is backwards: the alternatives are courses a student MIGHT take, and the
    placed one is the course they will."""

    def test_an_outside_course_is_flagged_on_its_own_row(self):
        answers = {"track": "11 month", "goals": ["bi-analyst"],
                   "workload": "moderate"}
        plan = planner.build_for(answers, frozenset())
        catalog = {c["id"]: c for c in electives.load_catalog()}
        for index, quarter in enumerate(plan["quarters"]):
            outside = [r for r in quarter["courses"]
                       if catalog.get(r.get("courseId"))
                       and not r["courseId"].startswith("MGTA")]
            if not outside:
                continue
            body, _replies, _last = planner.review_quarter(plan, answers, index)
            assert "outside the MSBA" in body, \
                f"{[r['code'] for r in outside]} shown with no flag"

    def test_an_all_msba_quarter_carries_no_such_note(self):
        answers = {"track": "11 month", "goals": ["healthcare-analyst"],
                   "workload": "moderate"}
        plan = planner.build_for(answers, frozenset())
        for index, quarter in enumerate(plan["quarters"]):
            if any(not r["courseId"].startswith("MGTA")
                   for r in quarter["courses"] if r.get("courseId")):
                continue
            body, _replies, _last = planner.review_quarter(plan, answers, index)
            assert "outside the MSBA" not in body


class TestACourseIsOnlyEverPlacedInATermItIsTaught:
    """The plan tells a student what to take in Fall. If the course is only
    taught in Spring, the plan is not a plan -- and nothing downstream would
    catch it, because the units still add to 50 and every slot is still full.

    Held across the whole grid rather than spot-checked: 14 roles x 2 tracks
    x 3 loads, every course in every quarter, plus every alternative offered
    for a swap.
    """

    def _seasons(self, course):
        return {o.get("season") for o in (course.get("offerings") or [])
                if o.get("season")}

    def _by_id(self):
        return {c["id"]: c for c in electives.load_catalog()}

    def _season_of(self, track):
        return {q["key"]: q["season"] for q in planner.TRACK_SKELETONS[track]}

    def test_every_catalog_course_says_when_it_runs(self):
        """The check below is vacuous for a course with no offerings."""
        missing = [c["code"] for c in electives.load_catalog()
                   if not self._seasons(c)]
        assert not missing, missing

    def test_no_placement_lands_in_the_wrong_term(self):
        catalog = self._by_id()
        wrong = []
        for goal in electives.load_careers():
            for track in ("11 month", "17 month"):
                seasons = self._season_of(track)
                for load in ("light", "moderate", "heavy"):
                    answers = {"track": track, "goals": [goal], "workload": load}
                    seeded = planner.seeded_units(track, load)
                    if seeded:
                        answers["quarter_units"] = seeded
                    plan = planner.build_for(answers, frozenset())
                    for quarter in plan["quarters"]:
                        want = seasons.get(quarter["key"])
                        for row in quarter["courses"]:
                            course = catalog.get(row.get("courseId"))
                            if not course:
                                continue
                            if want not in self._seasons(course):
                                wrong.append((course["code"], want, goal,
                                              track, load))
        assert not wrong, f"{len(wrong)} placements in the wrong term: {wrong[:3]}"

    def test_no_swap_is_offered_for_a_term_it_is_not_taught_in(self):
        catalog = self._by_id()
        wrong = []
        for goal in electives.load_careers():
            for track in ("11 month", "17 month"):
                seasons = self._season_of(track)
                answers = {"track": track, "goals": [goal],
                           "workload": "moderate"}
                plan = planner.build_for(answers, frozenset())
                for quarter in plan["quarters"]:
                    want = seasons.get(quarter["key"])
                    for slot, row in enumerate(quarter["courses"]):
                        if not (row.get("swappable") and row.get("courseId")):
                            continue
                        for option in planner.alternatives_for(
                                plan, answers, quarter["key"], slot,
                                frozenset(), 4)["options"]:
                            course = catalog.get(option["courseId"])
                            if course and want not in self._seasons(course):
                                wrong.append((course["code"], want, goal, track))
                            if option["units"] != row["units"]:
                                wrong.append((option["courseId"], "units",
                                              goal, track))
        assert not wrong, f"{len(wrong)} bad swap options: {wrong[:3]}"


class TestTheFinishingQuarterIsHeldByLoad:
    """The 17-month track ends on a second-year Fall, and the load rule holds
    its units: 4 on light and moderate, 2 on heavy. Before this, light
    floored every earlier quarter and let the finishing quarter absorb the
    rest, so the lightest spread ended on the heaviest term.

    The 11-month track has a finishing quarter (Spring) but no load to hold
    it by: the programme is compressed into four quarters and every one
    carries what the plan of study publishes. See `planner.load_is_a_choice`.
    """

    def _tail(self, track, load):
        seeded = planner.seeded_units(track, load)
        answers = {"track": track, "goals": ["data-scientist"], "workload": load}
        if seeded:
            answers["quarter_units"] = seeded
        plan = planner.build_for(answers, frozenset())
        return plan["quarters"][-1]["unitsPlanned"], plan

    @pytest.mark.parametrize("load,units", [
        ("light", 4), ("moderate", 4), ("heavy", 2)])
    def test_the_17_month_tail(self, load, units):
        tail, plan = self._tail("17 month", load)
        assert tail == units
        assert plan["totals"]["total"] == 50 and plan.get("route") == "fixed"

    def test_the_seed_itself_holds_the_rule(self):
        assert planner.seeded_units("17 month", "light")["fall-two"] == 4
        assert planner.seeded_units("17 month", "heavy")["fall-two"] == 2

    def test_each_track_knows_its_finishing_quarter(self):
        assert planner.tail_quarter_key("17 month") == "fall-two"
        assert planner.tail_quarter_key("11 month") == "spring"
        assert planner.tail_target("17 month", "light") == 4
        assert planner.tail_target("17 month", "heavy") == 2
        # A finishing quarter, but nothing to hold it by.
        assert planner.tail_target("11 month", "light") is None

    def test_light_lands_the_extra_units_late(self):
        """"The later quarters carry more" -- the two units the tail gave up
        go to Spring, not Fall."""
        seeded = planner.seeded_units("17 month", "light")
        assert seeded["spring"] == 14 and seeded["fall"] == 12

    def test_the_11_month_track_has_no_load_to_hold(self):
        """Compressed into four quarters, the 11-month plan of study IS the
        plan: light, moderate and heavy all seed nothing, Spring carries its
        published 14 units, and no plan on this track ever reports a short
        finishing quarter."""
        for load in ("light", "moderate", "heavy"):
            assert planner.seeded_units("11 month", load) == {}, load
            tail, plan = self._tail("11 month", load)
            assert tail == 14, load
            assert plan["tailShort"] is None
            assert plan["totals"]["total"] == 50 and plan.get("route") == "fixed"
        assert not planner.load_is_a_choice("11 month")
        assert planner.load_is_a_choice("17 month")

    def test_the_11_month_spring_keeps_the_full_time_floor(self):
        """Not a short tail: 12-18 like every other quarter, so the bundle
        placer's flex cannot run it under the graduate minimum."""
        spring = next(q for q in planner.adjustable_quarters("11 month")
                      if q["key"] == "spring")
        assert (spring["min"], spring["max"]) == (12, 18)
        assert planner.load_options_for("11 month", "spring") == []
        assert planner.quarter_units_form_for("11 month") is None

    def test_every_role_hits_the_tail_or_says_why_not(self):
        """The rule is held wherever the bundle allows and REPORTED where it
        cannot be. For bi-analyst a 4-unit final Fall would leave Spring under
        the 12-unit graduate minimum -- no differentiator combination fixes
        that -- so the tail carries 2 and the plan says so. Silently landing
        on 2 is the one outcome this forbids."""
        misses = []
        for goal in electives.load_careers():
            for load in ("light", "moderate", "heavy"):
                seeded = planner.seeded_units("17 month", load)
                answers = {"track": "17 month", "goals": [goal], "workload": load}
                if seeded:
                    answers["quarter_units"] = seeded
                plan = planner.build_for(answers, frozenset())
                assert plan["totals"]["total"] == 50, (goal, load)
                assert plan["unfilled"] == [], (goal, load)
                tail = plan["quarters"][-1]["unitsPlanned"]
                if tail == planner.tail_target("17 month", load):
                    assert plan["tailShort"] is None, (goal, load)
                else:
                    assert plan["tailShort"] == {
                        "wanted": planner.tail_target("17 month", load),
                        "actual": tail,
                        "label": plan["quarters"][-1]["label"]}, (goal, load)
                    misses.append((goal, load))
                # And the 11-month track, where the load is not a choice,
                # never has a short tail to report.
                eleven = planner.build_for(
                    {"track": "11 month", "goals": [goal], "workload": load},
                    frozenset())
                assert eleven["tailShort"] is None, (goal, load)
                assert eleven["totals"]["total"] == 50, (goal, load)
        # Heavy's 2-unit tail is reachable by every bundle; the misses are all
        # light/moderate bundles short of Spring-offered courses.
        assert not [m for m in misses if m[1] == "heavy"], misses
        assert len(misses) <= 16, misses

    def test_a_short_tail_is_said_out_loud(self):
        from django.contrib.auth.models import User
        from rsm_thrive.models import Conversation
        user = User.objects.create_user("tailer")
        conversation = Conversation.objects.create(user=user, destination="courses",
                                                   title="t")
        # Marketing Analyst: its Fall-taught courses are all needed earlier,
        # so the final Fall can only carry 2 of the 4 the load asks for.
        for said in ("marketing analyst", "17 month", "moderate"):
            reply = orchestrator.answer(FakeLLM([]), conversation, said, [])
        assert "carries **2 units** rather than the 4" in reply.body
        assert "taught in one term only" in reply.body


class TestTheBundleOutranksTheSpread:
    """A light or heavy spread moves units between quarters, and a bundle
    whose courses are taught in one term only often cannot be laid into the
    result. The whole curated bundle was then abandoned for the generic
    ranked fill -- silently.

    Measured across every role, track and load before the fix: 33 of 84
    combinations lost their bundle this way. `ml-engineer` on a light spread
    came back with 3 of its 9 electives serving the goal, against 8 of 9 on
    the published spread, and with none of its three CS courses.
    """

    def _plan(self, goal, track, load):
        answers = {"track": track, "goals": [goal], "workload": load}
        seeded = planner.seeded_units(track, load)
        if seeded:
            answers["quarter_units"] = seeded
        return planner.build_for(answers, frozenset())

    def test_no_role_track_load_combination_loses_its_bundle(self):
        lost = [(g, t, l)
                for g in electives.load_careers()
                for t in ("11 month", "17 month")
                for l in ("light", "moderate", "heavy")
                if self._plan(g, t, l).get("route") != "fixed"]
        assert not lost, f"{len(lost)} combinations fell back: {lost[:4]}"

    def test_every_combination_still_closes_at_fifty_units(self):
        for goal in electives.load_careers():
            for track in ("11 month", "17 month"):
                for load in ("light", "moderate", "heavy"):
                    plan = self._plan(goal, track, load)
                    assert plan["totals"]["total"] == 50, (goal, track, load)
                    assert plan["unfilled"] == [], (goal, track, load)

    def test_the_courses_still_serve_the_goal_on_every_load(self):
        catalog = {c["id"]: c for c in electives.load_catalog()}
        for goal, role in electives.load_careers().items():
            boosts = set(role.get("boost_courses") or {})
            for load in ("light", "moderate", "heavy"):
                plan = self._plan(goal, "11 month", load)
                chosen = [r["courseId"] for q in plan["quarters"]
                          for r in q["courses"] if r["requirement"] != "Core"]
                on_goal = sum(1 for c in chosen
                              if catalog[c]["code"] in boosts or c in boosts)
                assert on_goal / len(chosen) >= 0.5, \
                    f"{goal}/{load}: only {on_goal}/{len(chosen)} on-goal"

    def test_the_spread_no_longer_has_to_be_dropped(self):
        """This used to show bi-analyst on a light 17-month spread losing the
        spread to the bundle. With the catalog holding each course's real
        terms, `bundles._search` fills what the set cannot reach with free
        slots the scorer can, inside the chosen spread -- so no role, track
        and load combination has to give the spread up any more, and every
        plan still says which recommended courses were left out."""
        for goal in electives.load_careers():
            for track in ("11 month", "17 month"):
                for load in ("light", "moderate", "heavy"):
                    plan = self._plan(goal, track, load)
                    assert plan["spreadDropped"] is False, (goal, track, load)
                    assert isinstance(plan["bundleUnplaced"], list), (goal, track, load)


class TestThePreferenceIsAskedFirst:
    def _to_the_spread(self, conversation, opener="17 month, data scientist"):
        """A track and a goal is now all it takes.

        This used to answer the self-rating question first. That question is
        gone, so the spread is the FIRST and only thing asked -- and the extra
        turn would now answer the spread question itself.
        """
        return orchestrator.answer(FakeLLM([]), conversation, opener, [])

    def test_it_is_asked_before_any_course_is_named(self, conversation):
        reply = self._to_the_spread(conversation)
        assert "spread across the quarters" in reply.body.lower()
        assert "MGTA" not in reply.body

    def test_it_is_honest_that_the_total_does_not_move(self, conversation):
        """The degree is 50 units. A question that implied "light" meant less
        work would be promising something the plan cannot deliver."""
        reply = self._to_the_spread(conversation)
        assert "50 units either way" in reply.body
        assert "nothing here is final" in reply.body, "the walk-through refines it"

    def test_answering_it_builds_the_plan(self, conversation):
        orchestrator.answer(FakeLLM([]), conversation, "11 month, data scientist", [])
        reply = orchestrator.answer(FakeLLM([]), conversation, "light", [])
        assert "MGTA" in reply.body
        assert planner.load_session_intake(conversation)["workload"] == "light"

    def test_the_answer_becomes_a_real_distribution(self, conversation):
        self._to_the_spread(conversation)
        reply = orchestrator.answer(FakeLLM([]), conversation, "light", [])
        assert planner.load_session_intake(conversation)["quarter_units"] == {
            "fall": 12, "winter": 12, "spring": 14, "fall-two": 4}
        # The lead reports the spread the plan ACTUALLY has, not the seed:
        # the placer may sit a quarter two units off the seed to fit a course
        # taught in one term only. Saying "Winter 18 units" over a table
        # showing 16 was wrong in the first place a student looks.
        plan = planner.build_for(planner.load_session_intake(conversation),
                                 frozenset())
        for quarter in plan["quarters"][1:]:
            assert f"{quarter['label']} {quarter['unitsPlanned']} units" in reply.body
        assert "Fall (second year) 4 units" in reply.body, \
            "and says where the weight went"

    def test_heavy_front_loads_it_instead(self):
        assert planner.seeded_units("17 month", "heavy") == {
            "fall": 16, "winter": 12, "spring": 12, "fall-two": 2}

    def test_moderate_is_the_published_plan(self):
        """The published plan already ends on a 4-unit Fall, so moderate seeds
        nothing. And on the 11-month track NO load seeds anything: the plan of
        study is the plan."""
        assert planner.seeded_units("17 month", "moderate") == {}
        for load in ("light", "moderate", "heavy"):
            assert planner.seeded_units("11 month", load) == {}

    def test_the_11_month_track_is_not_asked(self, conversation):
        """Compressed into four quarters, it has no light or heavy version to
        offer -- so the plan arrives straight away, and says why the question
        did not come."""
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "11 month, data scientist", [])
        assert "spread across the quarters" not in reply.body.lower()
        assert "MGTA" in reply.body, "the plan, not a question"
        assert "haven't asked about light or heavy" in reply.body
        assert "compressed into four quarters" in reply.body
        # And the intake counts as complete: the walk-through starts on the
        # very next turn rather than being refused as an unfinished interview.
        assert planner.next_intake_step(planner.load_session_intake(conversation)) is None
        walk = orchestrator.answer(FakeLLM([]), conversation, "walk me through it", [])
        assert walk.model_note == "review" and "Summer III" in walk.body

    def test_a_load_said_on_the_11_month_track_is_answered_not_lost(
            self, conversation):
        orchestrator.answer(FakeLLM([]), conversation,
                            "11 month, data scientist", [])
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "light please, i have a part time job", [])
        assert reply.model_note == "plan"
        assert "there is no light or heavy version" in reply.body
        assert planner.load_session_intake(conversation)["workload"] == "light"
        assert "quarter_units" not in planner.load_session_intake(conversation)

    def test_it_acts_on_units_because_it_cannot_act_on_difficulty(self):
        """Documented rather than fixed. `rank_electives` does penalise a
        course heavier than the student asked for, but almost nothing in the
        catalog is graded light -- so a light preference has nothing lighter to
        pick and the picks come out identical. Asking about course difficulty
        and then delivering the same twelve courses would be worse than not
        asking.

        The bound was ONE light elective when the catalog held 31 courses. It
        is now two out of eighty, which is the same situation and not a
        different one; if it ever reaches a real share of the catalog, the
        question becomes worth asking and this test should fail so somebody
        revisits it.
        """
        from rsm_thrive.services.courses.electives import load_catalog

        electives = [c for c in load_catalog() if not c["is_core"]]
        light_courses = [c for c in electives if c.get("workload") == "light"]
        assert len(light_courses) / len(electives) < 0.1, (
            "light electives are no longer negligible -- a light preference "
            "may now be actionable, so revisit the question")

        picked = lambda plan: [row["code"] for quarter in plan["quarters"]
                               for row in quarter["courses"] if row["swappable"]]
        light = planner.build_for({**FULL, "workload": "light"}, frozenset())
        heavy = planner.build_for({**FULL, "workload": "heavy"}, frozenset())
        assert picked(light) == picked(heavy)

    def test_a_track_that_cannot_honour_it_says_so(self, conversation):
        """No quarter on the 17-month track can reach its 18-unit cap, so
        "heavy" there is a much smaller change than on the 11-month track."""
        seeded = planner.seeded_units("17 month", "heavy")
        assert seeded == {"fall": 16, "winter": 12, "spring": 12, "fall-two": 2}

    def test_a_light_preference_flags_what_it_could_not_honour(self):
        plan = planner.build_for({**FULL, "workload": "light"}, frozenset())
        stretches = [note for quarter in plan["quarters"]
                     for row in quarter["courses"]
                     for note in (row.get("stretch") or [])]
        assert any("light load you asked for" in note for note in stretches)

    def test_it_is_never_asked_twice(self, conversation):
        self._to_the_spread(conversation)
        reply = orchestrator.answer(FakeLLM([]), conversation,
                                    "actually just show me the plan", [])
        assert "spread across the quarters" not in reply.body.lower()
        assert "MGTA" in reply.body, "the plan arrives anyway"
        assert "published plan does" in reply.body, "and says what it assumed"


# ---------------------------------------------------------------------------
# Asked again, per quarter, during the walk-through
# ---------------------------------------------------------------------------

class TestEachQuarterIsAskedDuringTheWalkThrough:
    """On the 17-month track, where the load is a choice. Winter is the
    quarter under test: its published 14 can go to 12 or 16, so all three
    words mean different numbers there."""

    def test_the_quarter_carries_the_question_with_real_numbers(self, conversation):
        mid_walk(conversation, index=1, track="17 month")
        reply = orchestrator.answer(FakeLLM([]), conversation, "next quarter", [])
        assert "How heavy should Winter be?" in reply.body
        assert "12 units" in reply.body and "16 units" in reply.body

    def test_the_11_month_walk_through_asks_no_such_thing(self, conversation):
        """Every quarter carries the published load; the walk-through says so
        once, on the first quarter with elective room, and offers swaps."""
        mid_walk(conversation, index=0)
        reply = orchestrator.answer(FakeLLM([]), conversation, "next quarter", [])
        assert "# Fall" in reply.body
        assert "How heavy should" not in reply.body
        assert "no light or heavy version" in reply.body
        again = orchestrator.answer(FakeLLM([]), conversation, "next quarter", [])
        assert "# Winter" in again.body
        assert "no light or heavy version" not in again.body, "said once"
        assert "How heavy should" not in again.body

    def test_a_load_word_on_the_11_month_walk_through_is_answered(
            self, conversation):
        mid_walk(conversation)
        reply = orchestrator.answer(FakeLLM([]), conversation, "light", [])
        assert "isn't a choice" in reply.body
        assert "# Fall" in reply.body, "and the quarter is shown as it stands"
        assert planner.load_session_intake(conversation).get("quarter_loads") is None

    def test_answering_moves_the_units_and_says_where_they_went(self, conversation):
        mid_walk(conversation, index=2, track="17 month")
        reply = orchestrator.answer(FakeLLM([]), conversation, "light", [])
        assert "is now light — 12 units" in reply.body
        assert "absorbed it" in reply.body
        answers = planner.load_session_intake(conversation)
        # Winter 14 -> 12; the two units land in Fall, the first quarter in
        # published order with room.
        assert answers["quarter_units"] == {"fall": 14, "winter": 12,
                                            "spring": 12, "fall-two": 4}
        assert answers["quarter_loads"] == {"winter": "light"}

    def test_the_recommendation_moves_with_the_units(self, conversation):
        """A quarter asked to carry 16 units instead of 14 gets another
        elective slot, and the scorer fills it. That is the whole reason for
        asking here rather than on a form."""
        mid_walk(conversation, index=2, track="17 month")
        before = planner.build_for({**FULL, "track": "17 month"}, frozenset())
        orchestrator.answer(FakeLLM([]), conversation, "heavy", [])
        after = planner.build_for(planner.load_session_intake(conversation),
                                  frozenset())
        fall = lambda plan: next(q for q in plan["quarters"] if q["key"] == "winter")
        # The direction, not exact numbers: a curated bundle may sit up to
        # `bundles.QUARTER_FLEX` off its target to be schedulable at all, and
        # the plan says so where it does. See `planner.route_line`.
        assert fall(after)["unitsPlanned"] > fall(before)["unitsPlanned"]
        # Units, not row count. On a curated bundle the number of slots comes
        # from the sizes of the bundle's own courses, so 18 units can be four
        # rows or five; what must move is the load.
        elective_units = lambda plan: sum(
            row["units"] for row in fall(plan)["courses"]
            if row["requirement"] == "Elective")
        assert elective_units(after) > elective_units(before)
        assert after["totals"]["total"] == planner.TOTAL_UNITS

    def test_the_plan_still_totals_fifty_units(self, conversation):
        mid_walk(conversation, index=2, track="17 month")
        orchestrator.answer(FakeLLM([]), conversation, "heavy", [])
        plan = planner.build_for(planner.load_session_intake(conversation),
                                 frozenset())
        assert plan["totals"]["total"] == planner.TOTAL_UNITS

    def test_summer_says_plainly_that_it_cannot_change(self, conversation):
        mid_walk(conversation, index=0, track="17 month")
        reply = orchestrator.answer(FakeLLM([]), conversation, "heavy", [])
        assert "isn't adjustable" in reply.body
        assert "required courses" in reply.body
        assert planner.load_session_intake(conversation).get("quarter_units") is None

    def test_summer_on_the_11_month_track_gives_the_track_reason_instead(
            self, conversation):
        """Not "say next quarter and I'll show you one you can change" --
        on this track there is no such quarter."""
        mid_walk(conversation, index=0)
        reply = orchestrator.answer(FakeLLM([]), conversation, "heavy", [])
        assert "isn't a choice" in reply.body
        assert "one you can change" not in reply.body
        assert "# Summer III" in reply.body

    def test_a_load_outside_the_walk_through_is_the_overall_preference(
            self, conversation):
        """Same word, two meanings, disambiguated by whether a walk-through is
        running rather than by asking which they meant."""
        PlannerSession.objects.update_or_create(
            conversation=conversation,
            defaults={"intake": {**FULL, "track": "17 month"},
                      "asked": ["workload", "plan"], "review": None})
        orchestrator.answer(FakeLLM(["fine"]), conversation, "heavy", [])
        answers = planner.load_session_intake(conversation)
        assert answers["workload"] == "heavy"
        # It seeds the WHOLE spread, not one quarter -- which is exactly the
        # difference from the same word said mid-walk-through.
        assert answers["quarter_units"] == {"fall": 16, "winter": 12,
                                            "spring": 12, "fall-two": 2}
        assert "quarter_loads" not in answers

    def test_a_second_visit_states_the_answer_rather_than_re_asking(
            self, conversation):
        mid_walk(conversation, index=2, track="17 month")
        first = orchestrator.answer(FakeLLM([]), conversation, "light", [])
        assert "is now light" in first.body

        again = orchestrator.answer(FakeLLM([]), conversation, "light", [])
        assert "is already light — 12 units" in again.body
        assert "absorbed it" not in again.body, "nothing moved the second time"
        assert "Winter is light — 12 units**, as you asked" in again.body
        assert "How heavy should Winter be?" not in again.body

    def test_a_choice_already_made_is_never_moved_to_satisfy_a_later_one(
            self, conversation):
        """Refused and explained, rather than done silently. With Fall,
        Winter and the second Fall all decided, a lighter Spring has nowhere
        to send its two units."""
        mid_walk(conversation, index=3, track="17 month",
                 quarter_units={"fall": 12, "winter": 12, "spring": 14,
                                "fall-two": 4},
                 quarter_loads={"fall": "light", "winter": "light",
                                "fall-two": "moderate"})
        reply = orchestrator.answer(FakeLLM([]), conversation, "light", [])
        assert "without moving" in reply.body
        assert "**Fall**" in reply.body
        answers = planner.load_session_intake(conversation)
        assert answers["quarter_units"]["fall"] == 12, "their choice survived"

    def test_an_impossible_request_says_so_and_names_the_total(
            self, conversation):
        mid_walk(conversation, index=4, track="17 month",
                 quarter_units={"fall": 12, "winter": 12, "spring": 12,
                                "fall-two": 6},
                 quarter_loads={"fall": "light", "winter": "light",
                                "spring": "light"})
        reply = orchestrator.answer(FakeLLM([]), conversation, "light", [])
        assert "can't make" in reply.body.lower()
        assert "50 units" in reply.body

    def test_the_walk_through_still_carries_no_buttons(self, conversation):
        mid_walk(conversation)
        for turn in ("next quarter", "light", "heavy"):
            reply = orchestrator.answer(FakeLLM([]), conversation, turn, [])
            assert reply.quick_replies == [], turn
            assert reply.form is None, turn
            assert "_You can say:_" in reply.body, turn

    def test_the_spoken_line_keeps_navigation_ahead_of_swaps(self, conversation):
        """A quarter with two elective slots offers six swaps. A flat cap put
        all six in the line and dropped "next quarter" and the load answers off
        the end — the two things a student is most likely to want next."""
        mid_walk(conversation, index=2, track="17 month")
        reply = orchestrator.answer(FakeLLM([]), conversation, "next quarter", [])
        line = reply.body.split("_You can say:_")[1]
        assert "**light**" in line and "**heavy**" in line
        assert "**finalise**" in line or "**next quarter**" in line
        assert line.index("**light**") < line.index("swap"), "navigation first"
        assert line.count("swap ") <= 3, "and the swaps are the capped half"
