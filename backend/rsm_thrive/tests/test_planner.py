import pytest

from rsm_thrive.services.courses import planner
from rsm_thrive.services.courses.electives import load_catalog

ANSWERS = {
    "track": "11 month",
    "goals": ["data-scientist"],
    "skill_python": "comfortable",
    "skill_sql": "basic",
    "skill_stats": "comfortable",
    "skill_ml": "basic",
    "skill_communication": "basic",
    "workload": "moderate",
    "interests": ["machine-learning"],
}

BEGINNER = {**ANSWERS, "skill_python": "none", "skill_sql": "none",
            "skill_stats": "none", "skill_ml": "none",
            "goals": ["product-manager"], "workload": "light", "interests": []}


class TestIntake:
    def test_every_question_maps_to_a_scoring_input(self):
        keys = {q["key"] for q in planner.intake_questions()}
        assert "track" in keys and "goals" in keys and "workload" in keys

    def test_no_question_asks_a_student_to_rate_themselves(self):
        """Removed deliberately. Nobody can rate their own machine learning
        before they have taken any, and a low rating steered students away
        from the courses that would fix the gap."""
        keys = {q["key"] for q in planner.intake_questions()}
        assert not [k for k in keys if k.startswith("skill_")]

    def test_track_options_are_the_two_real_tracks(self):
        track = next(q for q in planner.intake_questions() if q["key"] == "track")
        assert {o["value"] for o in track["options"]} == set(planner.TRACK_SKELETONS)

    def test_validate_rejects_a_missing_track(self):
        answers = {**ANSWERS}
        del answers["track"]
        assert any("track" in p for p in planner.validate_intake(answers))

    def test_validate_rejects_an_unknown_goal(self):
        problems = planner.validate_intake({**ANSWERS, "goals": ["astronaut"]})
        assert any("goals" in p for p in problems)

    def test_validate_caps_the_number_of_goals(self):
        problems = planner.validate_intake(
            {**ANSWERS, "goals": ["data-scientist", "consultant",
                                  "marketing-analyst", "data-engineer"]})
        assert any("at most 3" in p for p in problems)

    def test_a_complete_intake_validates(self):
        assert planner.validate_intake(ANSWERS) == []

    def test_communication_does_not_inflate_technical_comfort(self):
        """A strong presenter is not therefore ready for PySpark."""
        quiet = planner.profile_from_intake(
            {**BEGINNER, "skill_communication": "advanced"})
        assert quiet["technical_comfort"] == planner.profile_from_intake(
            {**BEGINNER, "skill_communication": "none"})["technical_comfort"]

    def test_skill_answers_reach_the_scorer(self):
        low = planner.profile_from_intake(BEGINNER)["technical_comfort"]
        high = planner.profile_from_intake(
            {**ANSWERS, "skill_python": "advanced", "skill_sql": "advanced",
             "skill_stats": "advanced", "skill_ml": "advanced"})["technical_comfort"]
        assert low < high


class TestSkillRatings:
    """Nothing ASKS for a rating any more, but a rating a student volunteers
    is still read and still scored -- being asked and being told are different
    things. See `read_skills`."""

    def test_the_interview_never_reaches_a_rating_form(self):
        step = planner.next_intake_step({"track": "11 month",
                                         "goals": ["data-scientist"]})
        form = planner.rating_form_for(step)
        assert form is None or form["kind"] != "rating"

    def test_only_the_areas_still_missing_are_asked(self):
        step = {"key": "skills", "missing": ["skill_ml"]}
        form = planner.rating_form_for(step)
        assert [r["key"] for r in form["rows"]] == ["skill_ml"]

    def test_no_other_step_gets_a_form(self):
        for key in ("track", "goals", "workload"):
            assert planner.rating_form_for({"key": key, "missing": []}) is None

    def test_a_submitted_form_reads_like_a_person_wrote_it(self):
        said = planner.compose_rating_message(
            {"skill_python": 4, "skill_sql": 2})
        assert said == "Python programming 4, SQL and databases 2"

    @pytest.mark.parametrize("given,expected", [
        (4, 4), ("4", 4), ("comfortable", 4), ("none", 1), ("advanced", 5),
        ("expert", 5), ("beginner", 1), ("  Basic  ", 2),
    ])
    def test_a_rating_is_read_from_a_number_or_a_word(self, given, expected):
        """The buttons send numbers; people type words. Both are answers."""
        assert planner.skill_score(given) == expected

    @pytest.mark.parametrize("given", [0, 6, 9, -1, "", "banana", None, True])
    def test_an_unreadable_rating_is_rejected_not_guessed(self, given):
        assert planner.skill_score(given) is None

    def test_ratings_are_stored_as_numbers_whatever_arrived(self):
        answers = planner.normalise_intake(
            {"skill_python": "comfortable", "skill_sql": "2", "skill_ml": 5})
        assert answers == {"skill_python": 4, "skill_sql": 2, "skill_ml": 5}

    def test_an_out_of_range_rating_is_dropped(self):
        assert "skill_stats" not in planner.normalise_intake({"skill_stats": 9})

    def test_the_ratings_drive_technical_comfort_directly(self):
        low = planner.profile_from_intake(
            {**ANSWERS, "skill_python": 1, "skill_sql": 1,
             "skill_stats": 1, "skill_ml": 1})
        high = planner.profile_from_intake(
            {**ANSWERS, "skill_python": 5, "skill_sql": 5,
             "skill_stats": 5, "skill_ml": 5})
        assert low["technical_comfort"] == 1
        assert high["technical_comfort"] == 5

    def test_words_and_numbers_reach_the_same_comfort(self):
        by_word = planner.profile_from_intake(
            {**ANSWERS, "skill_python": "comfortable", "skill_sql": "comfortable",
             "skill_stats": "comfortable", "skill_ml": "comfortable"})
        by_number = planner.profile_from_intake(
            {**ANSWERS, "skill_python": 4, "skill_sql": 4,
             "skill_stats": 4, "skill_ml": 4})
        assert by_word["technical_comfort"] == by_number["technical_comfort"]

    def test_a_low_rating_still_raises_the_prerequisite_caution(self):
        from rsm_thrive.services.courses.electives import load_catalog

        course = next(c for c in load_catalog()
                      if "python" in (c.get("prerequisites") or "").lower())
        cautions = planner.prerequisite_cautions(
            course, {**ANSWERS, "skill_python": 1})
        assert cautions and "out of 5" in cautions[0]

    def test_a_high_rating_raises_no_caution(self):
        from rsm_thrive.services.courses.electives import load_catalog

        course = next(c for c in load_catalog()
                      if "python" in (c.get("prerequisites") or "").lower())
        assert planner.prerequisite_cautions(
            course, {**ANSWERS, "skill_python": 5, "skill_stats": 5,
                     "skill_ml": 5, "skill_sql": 5}) == []


class TestSkeletons:
    @pytest.mark.parametrize("track", sorted(planner.TRACK_SKELETONS))
    def test_every_track_totals_fifty_units(self, track):
        by_id = {c["id"]: c for c in load_catalog()}
        quarters = planner.TRACK_SKELETONS[track]
        core = sum(by_id[s["course_id"]]["units"] for q in quarters
                   for s in q["slots"] if s["kind"] == "core")
        fixed = sum(by_id[s["course_id"]]["units"] for q in quarters
                    for s in q["slots"] if s["kind"] == "fixed")
        open_slots = sum(s["units"] for q in quarters
                         for s in q["slots"] if s["kind"] == "elective")
        assert core == planner.CORE_UNITS
        assert fixed + open_slots == planner.ELECTIVE_UNITS
        assert core + fixed + open_slots == planner.TOTAL_UNITS

    @pytest.mark.parametrize("track", sorted(planner.TRACK_SKELETONS))
    def test_declared_quarter_units_match_their_slots(self, track):
        by_id = {c["id"]: c for c in load_catalog()}
        for quarter in planner.TRACK_SKELETONS[track]:
            total = sum(by_id[s["course_id"]]["units"] if s.get("course_id")
                        else s["units"] for s in quarter["slots"])
            assert total == quarter["units"], quarter["key"]


@pytest.mark.django_db
class TestBuildPlan:
    @pytest.mark.parametrize("track", sorted(planner.TRACK_SKELETONS))
    def test_plan_is_complete_and_adds_up(self, track):
        plan = planner.build_plan({**ANSWERS, "track": track})
        assert plan["unfilled"] == []
        assert plan["totals"]["total"] == planner.TOTAL_UNITS
        assert plan["totals"]["core"] == planner.CORE_UNITS
        assert plan["totals"]["elective"] == planner.ELECTIVE_UNITS
        for quarter in plan["quarters"]:
            assert quarter["unitsPlanned"] == quarter["unitsExpected"]

    def test_every_quarter_lists_core_and_elective_labelled(self):
        """The brief: show the core classes too, marked core vs elective.

        "Required" is the third label, and it means what neither of the others
        can say: Summer's MGTA 403 and MGTA 464 carry elective CREDIT and are
        mandatory anyway. See `_entry`.
        """
        plan = planner.build_plan(ANSWERS)
        for quarter in plan["quarters"]:
            assert quarter["courses"]
            for row in quarter["courses"]:
                assert row["requirement"] in ("Core", "Elective", "Required")
        cores = [r for q in plan["quarters"] for r in q["courses"]
                 if r["requirement"] == "Core"]
        assert {r["code"] for r in cores} == {
            "MGTA 451", "MGTA 452", "MGTA 453", "MGTA 455", "MGTA 444", "MGTA 454"}

    def test_summer_offers_no_choices_at_all(self):
        """Nothing in Summer is chooseable, on either track.

        The skeleton has never put an elective SLOT in Summer, but the two
        scheduled courses were labelled "Elective" in the plan a student reads,
        which invited them to look for a decision that does not exist.
        """
        for track in planner.TRACK_SKELETONS:
            plan = planner.build_plan({**ANSWERS, "track": track})
            summer = next(q for q in plan["quarters"] if q["key"] == "summer")

            assert not any(row["swappable"] for row in summer["courses"]), track
            assert "Elective" not in {row["requirement"] for row in summer["courses"]}, track
            assert {row["requirement"] for row in summer["courses"]} <= {
                "Core", "Required"}, track

    def test_summers_required_courses_still_count_as_elective_units(self):
        """The label changed; the arithmetic did not.

        MGTA 403 and MGTA 464 carry ELECTIVE credit in the published plan of
        study (both plan-of-study captures in the corpus say so). Relabelling
        them "Required" is about what a student can choose, not about what the
        registrar counts, and the 22/28 split has to survive it.
        """
        plan = planner.build_plan(ANSWERS)
        summer = next(q for q in plan["quarters"] if q["key"] == "summer")

        required = [r for r in summer["courses"] if r["requirement"] == "Required"]
        assert {r["code"] for r in required} == {"MGTA 403", "MGTA 464"}
        # `kind` is what the totals count, and it still says elective.
        assert all(r["kind"] == "elective" for r in required)
        assert plan["totals"]["core"] == planner.CORE_UNITS
        assert plan["totals"]["elective"] == planner.ELECTIVE_UNITS
        assert plan["totals"]["total"] == planner.TOTAL_UNITS

    def test_the_rendered_summer_says_there_is_nothing_to_choose(self):
        plan = planner.build_plan(ANSWERS)
        markdown = planner.render_plan_markdown(plan)

        assert "Everything this quarter is required" in markdown
        # And says why the totals still show elective units, so the table and
        # the summary above it do not look like they disagree.
        assert "count toward your elective units" in markdown

    def test_no_course_appears_twice(self):
        for track in planner.TRACK_SKELETONS:
            plan = planner.build_plan({**ANSWERS, "track": track})
            ids = [r["courseId"] for q in plan["quarters"] for r in q["courses"]
                   if r["courseId"]]
            assert len(ids) == len(set(ids)), track

    def test_electives_are_offered_in_the_quarter_they_are_placed_in(self):
        by_id = {c["id"]: c for c in load_catalog()}
        for track in planner.TRACK_SKELETONS:
            plan = planner.build_plan({**ANSWERS, "track": track})
            for quarter in plan["quarters"]:
                for row in quarter["courses"]:
                    if not row["courseId"]:
                        continue
                    seasons = {(o.get("season") or "").upper() for o
                               in by_id[row["courseId"]].get("offerings") or []}
                    assert quarter["season"] in seasons, (row["code"], quarter["key"])

    def test_core_slots_are_never_swappable(self):
        plan = planner.build_plan(ANSWERS)
        for quarter in plan["quarters"]:
            for row in quarter["courses"]:
                if row["requirement"] == "Core":
                    assert row["swappable"] is False

    def test_the_plan_changes_with_the_goal(self):
        """If two different students get the same electives, the intake is decoration."""
        ds = planner.build_plan(ANSWERS)
        pm = planner.build_plan(BEGINNER)
        def electives(plan):
            return {r["code"] for q in plan["quarters"] for r in q["courses"]
                    if r["requirement"] == "Elective" and r["swappable"]}
        assert electives(ds) != electives(pm)

    def test_a_beginner_is_warned_about_prerequisites(self):
        plan = planner.build_plan(BEGINNER)
        assert any(r["cautions"] for q in plan["quarters"] for r in q["courses"])

    def test_cautions_never_remove_a_course(self):
        """Advisory, not a filter — the student and their advisor decide."""
        plan = planner.build_plan(BEGINNER)
        assert plan["unfilled"] == []
        assert plan["totals"]["total"] == planner.TOTAL_UNITS

    def test_already_taken_courses_are_not_planned_again(self):
        plan = planner.build_plan(ANSWERS)
        first = next(r["courseId"] for q in plan["quarters"] for r in q["courses"]
                     if r["swappable"])
        again = planner.build_plan(ANSWERS, taken_ids={first})
        assert first not in {r["courseId"] for q in again["quarters"]
                             for r in q["courses"]}

    def test_the_plan_is_deterministic(self):
        assert planner.build_plan(ANSWERS) == planner.build_plan(ANSWERS)

    def test_unknown_track_is_an_error_not_a_silent_default(self):
        with pytest.raises(ValueError):
            planner.build_plan({**ANSWERS, "track": "4 month"})

    def test_plan_carries_booking_and_catalog_links(self):
        plan = planner.build_plan(ANSWERS)
        keys = {link["key"] for link in plan["links"]}
        assert {"tss", "schedule", "plans-drive"} <= keys
        for link in plan["links"]:
            assert link["url"].startswith("https://")


@pytest.mark.django_db
class TestAlternatives:
    def test_options_fill_the_same_hole(self):
        plan = planner.build_plan(ANSWERS)
        result = planner.alternatives_for(plan, ANSWERS, "fall", 2)
        assert result["options"]
        current = result["current"]
        for option in result["options"]:
            assert option["units"] == current["units"]

    def test_options_are_not_already_in_the_plan(self):
        plan = planner.build_plan(ANSWERS)
        in_plan = {r["courseId"] for q in plan["quarters"] for r in q["courses"]}
        for option in planner.alternatives_for(plan, ANSWERS, "fall", 2)["options"]:
            assert option["courseId"] not in in_plan

    def test_each_option_says_why_it_is_an_alternative(self):
        plan = planner.build_plan(ANSWERS)
        for option in planner.alternatives_for(plan, ANSWERS, "fall", 2)["options"]:
            assert option["sharedSkills"] or option["sharedFocus"] or option["reasons"]

    def test_options_are_ordered_by_similarity_to_the_current_course(self):
        plan = planner.build_plan(ANSWERS)
        sims = [o["similarity"] for o in
                planner.alternatives_for(plan, ANSWERS, "fall", 2)["options"]]
        assert sims == sorted(sims, reverse=True)

    def test_a_fixed_slot_offers_nothing_and_says_so(self):
        plan = planner.build_plan(ANSWERS)
        result = planner.alternatives_for(plan, ANSWERS, "summer", 0)
        assert result["swappable"] is False and result["options"] == []
        assert result["why"]

    def test_an_unknown_slot_is_an_error(self):
        plan = planner.build_plan(ANSWERS)
        with pytest.raises(ValueError):
            planner.alternatives_for(plan, ANSWERS, "nope", 0)


@pytest.mark.django_db
class TestSwap:
    """`build_for`, not `build_plan`, throughout.

    A swap acts on the plan the student is LOOKING at, and since curated roles
    default to their bundle that is no longer the same thing as the scorer's
    output. Driving the swap from one and reading the result off the other is
    what made a single Fall swap appear to move five rows.
    """

    def _first_option(self):
        plan = planner.build_for(ANSWERS)
        return planner.alternatives_for(plan, ANSWERS, "fall", 2)["options"][0]

    def test_a_swap_sticks_and_keeps_the_plan_valid(self):
        option = self._first_option()
        selections = planner.apply_swap(ANSWERS, {}, "fall", 2, option["courseId"])
        plan = planner.build_for(ANSWERS, selections=selections)
        assert plan["quarters"][1]["courses"][2]["courseId"] == option["courseId"]
        assert plan["totals"]["total"] == planner.TOTAL_UNITS
        assert plan["unfilled"] == []

    @pytest.mark.parametrize("route", ["fixed", "custom"])
    def test_a_swap_only_moves_the_slot_it_targets(self, route):
        answers = {**ANSWERS, "route": route}
        before = planner.build_for(answers)
        option = planner.alternatives_for(before, answers, "fall", 2)["options"][0]
        after = planner.build_for(
            answers, selections=planner.apply_swap(answers, {}, "fall", 2,
                                                   option["courseId"]))
        changed = [(q["key"], i) for q, qa in zip(before["quarters"], after["quarters"])
                   for i, (a, b) in enumerate(zip(q["courses"], qa["courses"]))
                   if a["courseId"] != b["courseId"]]
        assert changed == [("fall", 2)], route

    def test_a_swapped_course_is_marked_as_the_students_choice(self):
        option = self._first_option()
        plan = planner.build_for(
            ANSWERS, selections=planner.apply_swap(ANSWERS, {}, "fall", 2,
                                                   option["courseId"]))
        assert plan["quarters"][1]["courses"][2]["note"] == "your choice"

    def test_wrong_unit_size_is_refused(self):
        with pytest.raises(ValueError, match="units"):
            planner.apply_swap(ANSWERS, {}, "fall", 2, "MGTA 402")

    def test_a_course_not_offered_that_quarter_is_refused(self):
        with pytest.raises(ValueError, match="not offered"):
            planner.apply_swap(ANSWERS, {}, "fall", 2, "MGTA 456")

    def test_a_core_course_cannot_fill_an_elective_slot(self):
        with pytest.raises(ValueError, match="core"):
            planner.apply_swap(ANSWERS, {}, "fall", 2, "MGTA 452")

    def test_a_fixed_slot_cannot_be_swapped(self):
        with pytest.raises(ValueError, match="fixed"):
            planner.apply_swap(ANSWERS, {}, "summer", 1, "MGTA 457")

    def test_an_unknown_course_is_refused(self):
        with pytest.raises(ValueError, match="unknown course"):
            planner.apply_swap(ANSWERS, {}, "fall", 2, "MGTA 999")

    def test_a_course_already_taken_is_refused(self):
        option = self._first_option()
        with pytest.raises(ValueError, match="already taken"):
            planner.apply_swap(ANSWERS, {}, "fall", 2, option["courseId"],
                               taken_ids={option["courseId"]})

    def test_the_same_course_cannot_be_planned_twice(self):
        """Needs a course that is BOTH already in the plan elsewhere and legal
        for the target slot — otherwise an earlier rule refuses it first and the
        duplicate guard is never reached."""
        by_id = {c["id"]: c for c in load_catalog()}
        plan = planner.build_for(ANSWERS)
        # Any slot, and any course sitting in ANOTHER quarter that is legal
        # for it -- the pair depends on the timetable, not on the test.
        found = None
        for quarter in plan["quarters"]:
            for index, row in enumerate(quarter["courses"]):
                if not row["swappable"]:
                    continue
                for other in plan["quarters"]:
                    if other["key"] == quarter["key"]:
                        continue
                    for candidate in other["courses"]:
                        course = by_id.get(candidate["courseId"] or "")
                        if (candidate["swappable"] and course
                                and course["units"] == row["units"]
                                and planner._offered_in(course, quarter["season"])):
                            found = (quarter["key"], index, course["id"])
                if found:
                    break
            if found:
                break
        assert found, "no in-plan elective is also legal for another slot"
        key, slot, elsewhere = found
        with pytest.raises(ValueError, match="already in your"):
            planner.apply_swap(ANSWERS, {}, key, slot, elsewhere)


@pytest.mark.django_db
class TestRendering:
    def test_markdown_lists_every_quarter_and_marks_core_vs_elective(self):
        plan = planner.build_plan(ANSWERS)
        text = planner.render_plan_markdown(plan)
        for quarter in plan["quarters"]:
            assert f"## {quarter['label']}" in text
        assert "| Requirement |" in text
        assert "Core" in text and "Elective" in text

    def test_markdown_uses_tables_the_frontend_can_render(self):
        text = planner.render_plan_markdown(planner.build_plan(ANSWERS))
        assert "|---|---|---|---|" in text

    def test_markdown_carries_the_booking_links(self):
        text = planner.render_plan_markdown(planner.build_plan(ANSWERS))
        assert "https://sis.ucsd.edu/" in text
        assert "drive.google.com" in text

    def test_markdown_names_every_planned_course(self):
        plan = planner.build_plan(ANSWERS)
        text = planner.render_plan_markdown(plan)
        for quarter in plan["quarters"]:
            for row in quarter["courses"]:
                if row["code"]:
                    assert row["code"] in text

    def test_markdown_separates_goal_courses_from_breadth(self):
        text = planner.render_plan_markdown(planner.build_plan(ANSWERS))
        assert "Why these electives" in text

    def test_markdown_surfaces_a_beginners_warnings(self):
        text = planner.render_plan_markdown(planner.build_plan(BEGINNER))
        assert "Worth knowing before you book" in text

    def test_markdown_ends_with_the_advising_disclaimer(self):
        text = planner.render_plan_markdown(planner.build_plan(ANSWERS))
        assert "MSBA advising" in text


class TestSimilarity:
    def test_a_course_is_most_similar_to_itself(self):
        catalog = {c["id"]: c for c in load_catalog()}
        a = catalog["CSE 251A"]
        assert planner.similarity(a, a) == 1.0

    def test_unrelated_courses_score_lower_than_related_ones(self):
        catalog = {c["id"]: c for c in load_catalog()}
        ml = catalog["CSE 251A"]
        near = planner.similarity(ml, catalog["CSE 251B"])   # both ML
        far = planner.similarity(ml, catalog["MGTA 402"])    # presentation skills
        assert near > far

    def test_shared_skills_come_from_the_alternative(self):
        catalog = {c["id"]: c for c in load_catalog()}
        shared = planner.shared_skills(catalog["CSE 251A"], catalog["CSE 251B"])
        assert all(s in catalog["CSE 251B"]["skills"] for s in shared)


class TestAShortPlanSaysSo:
    """The defect: a plan that could not be filled reported the units it
    WANTED rather than the units it had.

    Fall publishes exactly two 2-unit electives (MGTA 402, MGTA 457), so a
    student who has already taken both leaves the Fall 2-unit slot unfillable.
    The row rendered "nothing available for this slot" -- and the header two
    lines above it still read "50 units ... which is what the degree requires",
    because every total summed the placeholder row's `units` alongside the real
    ones. A student could have planned a degree on a 48-unit plan that told
    them it was complete.

    The fixture pins the PUBLISHED spread (14 / 14 / 14): a moderate 11-month
    plan now holds Spring at 8 and cuts Fall as two 4-unit slots, which has
    no 2-unit slot to leave unfilled. The defect this guards is about the
    arithmetic of an unfillable slot, not about which spread produces one.
    """
    PUBLISHED = {**ANSWERS, "quarter_units": {"fall": 14, "winter": 14, "spring": 14}}

    # Every 2-unit Fall elective, computed rather than listed: the point of the
    # fixture is that NOTHING can fill Fall's 2-unit slot, and a hardcoded pair
    # stopped meaning that the moment the catalog grew past 31 courses.
    @staticmethod
    def _all_fall_two_unit():
        from rsm_thrive.services.courses.electives import load_catalog

        return frozenset(
            c["id"] for c in load_catalog()
            if not c["is_core"] and c["units"] == 2
            and any(o.get("season") == "FA" for o in c.get("offerings") or []))

    def test_totals_count_only_what_is_scheduled(self):
        plan = planner.build_plan(self.PUBLISHED, self._all_fall_two_unit())
        scheduled = sum(row["units"] for quarter in plan["quarters"]
                        for row in quarter["courses"] if row["courseId"])
        assert plan["unfilled"], "expected an unfillable slot for this fixture"
        assert plan["totals"]["total"] == scheduled == 48
        assert plan["totals"]["elective"] == 26

    def test_a_quarter_reports_the_units_it_holds(self):
        plan = planner.build_plan(self.PUBLISHED, self._all_fall_two_unit())
        for quarter in plan["quarters"]:
            held = sum(row["units"] for row in quarter["courses"] if row["courseId"])
            assert quarter["unitsPlanned"] == held, quarter["key"]

    def test_the_markdown_warns_instead_of_claiming_completeness(self):
        body = planner.render_plan_markdown(planner.build_plan(self.PUBLISHED, self._all_fall_two_unit()))
        assert "2 units short of the 50" in body
        assert "complete 50-unit plan" not in body

    def test_a_full_plan_still_reads_as_complete_and_never_warns(self):
        plan = planner.build_plan(ANSWERS)
        assert plan["unfilled"] == []
        assert plan["totals"]["total"] == 50
        assert "short of the" not in planner.render_plan_markdown(plan)


class TestTheExtractorKnowsJobTitles:
    """The defect: the taxonomy moved to fourteen profiles and took the job
    titles with it, so a student naming an ordinary title was told their
    career was not covered.

    `resolve_role` repairs a stored role id. It never sees free text, so the
    other half of the migration is that the extractor is handed the titles each
    profile is actually called by -- including the retired role names.
    """

    def test_every_profile_offers_the_titles_it_is_called_by(self):
        from rsm_thrive.services.courses.electives import load_careers

        for role_id, role in load_careers().items():
            assert role.get("titles"), f"{role_id} has no titles"

    def test_the_vocabulary_names_retired_roles_against_their_successor(self):
        vocabulary = planner.role_vocabulary()
        for said, role_id in [("product manager", "product-analyst"),
                              ("data engineer", "analytics-engineer"),
                              ("financial analyst", "finance-quant")]:
            line = next(l for l in vocabulary.splitlines() if l.startswith(role_id))
            assert said in line.lower(), f"{said!r} missing from {role_id}"

    def test_the_extract_prompt_carries_the_vocabulary(self):
        placeholders = planner.intake_extract_placeholders()
        assert "product manager" in placeholders["role_ids"].lower()
        # One line per role, so the model reads it as a lookup rather than prose.
        assert len(placeholders["role_ids"].splitlines()) == 14


class TestAQuestionIsNotAReviewCommand:
    """The defect: `review_intent` matched bare substrings, so ordinary
    questions asked during the walk-through were executed as commands.

    "can you confirm what MGTA 458 is?" contains "confirm" and FINALISED the
    plan. "what should I review?" restarted the walk-through, and "is next
    quarter heavy?" advanced it. A student asking about their plan was changing
    it, and `review_intent` runs before anything that could answer them.
    """

    @pytest.mark.parametrize("text", [
        "what's next?", "what should I review?",
        "can you confirm what MGTA 458 is?", "is next quarter heavy?",
        "what does review mean?", "why continue with this?",
        "does this look good to you?", "what comes next after winter?",
        "should I confirm this with advising?",
    ])
    def test_questions_are_not_commands(self, text):
        assert planner.review_intent(text) is None

    @pytest.mark.parametrize("text,intent", [
        ("next", "next"), ("Next", "next"), ("next quarter", "next"),
        ("continue", "next"), ("keep going", "next"),
        ("walk me through it", "start"), ("quarter by quarter", "start"),
        ("can you walk me through it?", "start"),
        ("finalise", "finalise"), ("looks good", "finalise"),
        ("Looks good, finalise it", "finalise"), ("done", "finalise"),
    ])
    def test_the_real_commands_still_work(self, text, intent):
        # The buttons send fixed strings; those must keep working exactly.
        assert planner.review_intent(text) == intent

    def test_every_button_send_is_understood(self):
        for button in planner.review_intro_replies():
            assert planner.review_intent(button["send"]) is not None, button


class TestTheCatalogAnswersCourseQuestions:
    """The course planner answers from `data/catalog/courses.json` rather than
    the document corpus. That file IS the answer to a course question -- units,
    seasons, prerequisites, tools and workload as fields -- where a retrieved
    prose chunk carries whatever a web page happened to say. Retrieval also
    answered questions this bot has no business answering: asked about tuition
    it found a fee page and replied, which belongs to the FAQ bot."""

    def test_a_named_course_returns_that_course_alone(self):
        from rsm_thrive.services.courses.electives import search_catalog

        hits = search_catalog("what is MGTA 458 about")
        assert [c["code"] for c in hits] == ["MGTA 458"]

    def test_a_tool_finds_the_course_that_teaches_it(self):
        from rsm_thrive.services.courses.electives import search_catalog

        codes = [c["code"] for c in search_catalog("which courses use Tableau")]
        assert "MGTA 457" in codes

    def test_a_season_is_searchable_by_name(self):
        """Students ask for "winter", never for "WI"."""
        from rsm_thrive.services.courses.electives import search_catalog

        hits = search_catalog("which electives are offered in winter")
        assert hits, "a season name must reach the catalog"

    @pytest.mark.parametrize("question", [
        "what are the tuition fees",
        "how do i set up zoom",
        "what's the weather in san diego",
        "when is orientation",
    ])
    def test_non_course_questions_reach_nothing(self, question):
        """The word "tuition" appears in a course description, so a bare term
        match answered a fee question from a course. A question has to be ABOUT
        courses before the catalog is consulted."""
        from rsm_thrive.services.courses.electives import search_catalog

        assert search_catalog(question) == []

    @pytest.mark.parametrize("question,expected", [
        ("what courses do you have", True),
        ("which electives have no prerequisites", True),
        ("how many units is it", True),
        ("what's the weather", False),
        ("how do i set up zoom", False),
    ])
    def test_course_questions_are_recognised(self, question, expected):
        from rsm_thrive.services.courses.electives import is_course_question

        assert is_course_question(question) is expected

    def test_the_overview_carries_enough_to_answer_from(self):
        """A code list cannot answer "which run in winter" or "which have no
        prerequisites", and that is what it used to return."""
        from rsm_thrive.services.courses.replies import _catalog_context

        context = _catalog_context("what courses do you have")
        assert context
        assert "prereq" in context and "offered" in context
        assert "MGTA 458" in context and "CSE 251A" in context
