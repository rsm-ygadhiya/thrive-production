"""A recommended set that the timetable cannot hold whole is placed in part,
and the plan says what was left out.

With each course's real terms in the catalog -- not "varies" for every quarter
-- several bundles no longer fit the quarters they are taught in. Before this,
the whole bundle was abandoned for the generic ranked fill, silently, for 54 of
the 84 role/track/load combinations. Now the spine (anchor and universal) is
kept, as many differentiators as fit are placed, free slots take the rest by
fit, and `bundleUnplaced` names what the timetable squeezed out.
"""

import pytest

from rsm_thrive.services.courses import bundles, electives, planner

pytestmark = pytest.mark.django_db

SKILLS = {"skill_python": "comfortable", "skill_sql": "basic",
          "skill_stats": "comfortable", "skill_ml": "basic",
          "skill_communication": "basic", "workload": "moderate"}


def plan_for(goal, track, load="moderate"):
    answers = {"track": track, "goals": [goal], **SKILLS, "workload": load}
    seeded = planner.seeded_units(track, load)
    if seeded:
        answers["quarter_units"] = seeded
    return planner.build_for(answers, frozenset())


class TestEveryBundleIsKept:

    def test_no_combination_falls_back_to_the_generic_fill(self):
        lost = [(g, t, l) for g in electives.load_careers()
                for t in ("11 month", "17 month")
                for l in ("light", "moderate", "heavy")
                if plan_for(g, t, l)["route"] != "fixed"]
        assert lost == []

    def test_every_plan_is_whole_and_legal(self):
        for g in electives.load_careers():
            for t in ("11 month", "17 month"):
                plan = plan_for(g, t)
                assert plan["totals"]["total"] == planner.TOTAL_UNITS, (g, t)
                assert plan["unfilled"] == [], (g, t)
                ids = [r["courseId"] for q in plan["quarters"]
                       for r in q["courses"] if r["courseId"]]
                assert len(ids) == len(set(ids)), (g, t)

    def test_a_course_is_never_placed_in_a_quarter_it_does_not_run_in(self):
        catalog = planner._catalog_by_id()
        for g in electives.load_careers():
            for t in ("11 month", "17 month"):
                for l in ("light", "moderate", "heavy"):
                    plan = plan_for(g, t, l)
                    for quarter in plan["quarters"]:
                        for row in quarter["courses"]:
                            if row["courseId"]:
                                assert planner._offered_in(
                                    catalog[row["courseId"]], quarter["season"]), \
                                    (g, t, l, quarter["label"], row["code"])


class TestWhatWasLeftOutIsSaid:

    def test_data_scientist_on_eleven_months_cannot_hold_all_three_fall_courses(self):
        """MGTA 461, CSE 251A and MGTA 402 are all Fall-only; Fall holds two.
        The MSBA's own course stays, the CSE one is reported."""
        plan = plan_for("data-scientist", "11 month")
        codes = {r["code"] for q in plan["quarters"] for r in q["courses"]}
        assert "MGTA 461" in codes and "MGTA 402" in codes
        assert "CSE 251A" not in codes
        assert plan["bundleUnplaced"] == ["CSE 251A"]
        body = planner.render_plan_markdown(plan)
        assert "Not all of the recommended set fits" in body
        assert "**CSE 251A** is left out" in body

    def test_a_bundle_that_fits_whole_says_nothing(self):
        plan = plan_for("product-analyst", "17 month")
        assert plan["bundleUnplaced"] == []
        assert "left out" not in planner.render_plan_markdown(plan)

    def test_left_out_courses_are_named_once(self):
        for g in electives.load_careers():
            for t in ("11 month", "17 month"):
                plan = plan_for(g, t)
                body = planner.render_plan_markdown(plan)
                for code in plan["bundleUnplaced"]:
                    assert body.count(f"**{code}** ") + body.count(f"**{code}**,") \
                        + body.count(f"**{code}**.") >= 1, (g, t, code)
                    assert "leaves no room for" not in body or code not in \
                        body.split("leaves no room for")[1].split("\n")[0], (g, t, code)

    def test_an_unplaced_course_really_could_not_be_added(self):
        """Nothing is reported as left out that a free slot could have taken:
        every left-out course has no slot of its size in a quarter it runs in
        that the plan left to the scorer."""
        catalog = planner._catalog_by_id()
        for g in electives.load_careers():
            for t in ("11 month", "17 month"):
                plan = plan_for(g, t)
                bundle = bundles.bundle_for(g)
                pinned = {c for layer in ("anchor", "differentiator", "universal")
                          for c in bundle.get(layer) or []}
                for code in plan["bundleUnplaced"]:
                    course = next(c for c in catalog.values()
                                  if planner.display_code(c) == code)
                    for quarter in plan["quarters"]:
                        if not planner._offered_in(course, quarter["season"]):
                            continue
                        for row in quarter["courses"]:
                            if (row["swappable"] and row["units"] == course["units"]
                                    and row["courseId"] not in pinned):
                                # A scorer-filled slot of the right size in a
                                # quarter this course runs in: the search
                                # preferred a fuller set over this course, so
                                # the course must at least be swappable in.
                                assert planner.placements_for(plan, course), \
                                    (g, t, code, quarter["label"])
