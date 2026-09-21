"""A hundred plans, checked for the invariants that make a plan legal.

Unit tests pin behaviour one case at a time. This sweeps the whole input space
the planner actually sees -- every track against every curated role against
every load answer, plus every single-quarter override on top -- and asserts the
properties that must hold for ALL of them.

Why a sweep rather than more unit tests: the light/moderate/heavy answers move
units between quarters, and the failure they risk is arithmetic. A plan that
totals 48 units, or leaves a 3-unit hole no course can fill, or schedules the
same course twice, is wrong in a way a student would only discover at
graduation -- and it would be wrong for one combination out of ninety while
every hand-written case passed.

The invariants, all of them checked on every plan:

1. **50 units**, 22 core and 28 elective -- the degree requirement.
2. **Every quarter within its published bounds**, and even, because
   `_resized` cuts a budget into 4-unit slots plus a 2-unit remainder.
3. **No course twice**, by catalog id.
4. **Every course real**, offered in the quarter it is placed in, and at the
   unit size of the slot it fills.
5. **No core course displaced** -- the published sequence is not negotiable.
"""

import itertools

import pytest

from rsm_thrive.services.courses import planner
from rsm_thrive.services.courses.electives import load_catalog

TRACKS = ("11 month", "17 month")
LOADS = ("light", "moderate", "heavy")
SKILLS = {f"skill_{area['key']}": "basic" for area in planner.SKILL_AREAS}


def _cases():
    """Every (track, role, load) triple, plus one per single-quarter override.

    Deliberately enumerated rather than sampled: the space is small enough to
    cover exhaustively, and a randomised sweep that fails on seed 7 is a bug
    report nobody can reproduce.
    """
    roles = sorted(planner.load_careers())
    for track, role, load in itertools.product(TRACKS, roles, LOADS):
        yield {"id": f"{track}/{role}/{load}", "track": track, "role": role,
               "load": load, "override": None}
    # And the walk-through's per-quarter answers, on top of a seeded spread.
    for track in TRACKS:
        for quarter in planner.adjustable_quarters(track):
            for load in LOADS:
                yield {"id": f"{track}/{quarter['key']}={load}",
                       "track": track, "role": "data-scientist",
                       "load": "moderate",
                       "override": (quarter["key"], load)}


CASES = list(_cases())


def _answers(case):
    answers = {"track": case["track"], "goals": [case["role"]],
               "workload": case["load"], **SKILLS}
    seeded = planner.seeded_units(case["track"], case["load"])
    if seeded:
        answers["quarter_units"] = seeded
    if case["override"]:
        key, load = case["override"]
        units = planner.units_for_load(case["track"], key, load,
                                       answers.get("quarter_units"))
        if units is not None:
            wanted, _blockers = planner.rebalanced_units(
                case["track"], answers.get("quarter_units") or {}, key, units)
            if wanted is not None:
                answers["quarter_units"] = wanted
                answers["quarter_loads"] = {key: load}
    return answers


def test_the_sweep_is_actually_a_hundred_cases():
    """If the catalog or the tracks change, this says so rather than quietly
    covering less."""
    assert len(CASES) >= 100, len(CASES)


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_the_plan_is_legal(case):
    answers = _answers(case)
    plan = planner.build_for(answers, frozenset())
    catalog = {course["id"]: course for course in load_catalog()}
    label = case["id"]

    # 1. The degree total.
    totals = plan["totals"]
    assert totals["core"] == planner.CORE_UNITS, label
    assert totals["elective"] == planner.ELECTIVE_UNITS, label
    assert totals["total"] == planner.TOTAL_UNITS, label
    assert plan["unfilled"] == [], f"{label}: unfillable slot {plan['unfilled']}"

    # 2. Quarter bounds, and even budgets.
    #
    # A quarter may sit `bundles.QUARTER_FLEX` units off its target on the
    # FIXED route and only there: exact per-quarter fill places 5 of 14
    # bundles on the 11-month track and 1 of 14 on the 17-month, so the
    # concession is what makes the curated sets schedulable at all. It is
    # tried at zero first, and wherever it is actually taken the plan has to
    # SAY so -- a student who just chose these loads is owed that.
    from rsm_thrive.services.courses.bundles import QUARTER_FLEX

    slack = QUARTER_FLEX if plan["route"] == "fixed" else 0
    bounds = {q["key"]: q for q in planner.adjustable_quarters(case["track"])}
    body = planner.render_plan_markdown(plan)
    for quarter in plan["quarters"]:
        drift = abs(quarter["unitsPlanned"] - quarter["unitsExpected"])
        assert drift <= slack, f"{label}: {quarter['label']} off by {drift}"
        if drift:
            assert quarter["label"] in body, \
                f"{label}: {quarter['label']} drifted silently"
            assert "rather than" in body, label
        assert quarter["unitsPlanned"] % 2 == 0, label
        room = bounds.get(quarter["key"])
        if room:
            assert room["min"] <= quarter["unitsPlanned"] <= room["max"], label

    rows = [row for quarter in plan["quarters"] for row in quarter["courses"]]

    # 3. Nothing scheduled twice.
    ids = [row["courseId"] for row in rows if row["courseId"]]
    assert len(ids) == len(set(ids)), f"{label}: duplicate {ids}"

    # 4. Every course real, offered then, and the right size.
    for quarter in plan["quarters"]:
        for row in quarter["courses"]:
            if not row["courseId"]:
                continue
            course = catalog.get(row["courseId"])
            assert course is not None, f"{label}: invented {row['courseId']}"
            assert course["units"] == row["units"], label
            seasons = {o["season"] for o in course.get("offerings") or []}
            assert quarter["season"] in seasons, \
                f"{label}: {row['code']} not offered in {quarter['label']}"

    # 5. The published core sequence is untouched.
    scheduled_core = [row["code"] for row in rows if row["kind"] == "core"]
    published = [catalog[slot["course_id"]]["code"]
                 for quarter in planner.TRACK_SKELETONS[case["track"]]
                 for slot in quarter["slots"] if slot["kind"] == "core"]
    assert scheduled_core == published, label


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_the_plan_renders_and_says_what_it_assumed(case):
    """A plan that cannot be rendered is not a plan. Cheap, and it catches the
    formatting crashes a unit test on one intake never would."""
    plan = planner.build_for(_answers(case), frozenset())
    body = planner.render_plan_markdown(plan)
    assert f"{planner.TOTAL_UNITS} units" in body, case["id"]
    assert "None" not in body, f"{case['id']}: a missing value reached the page"
    for quarter in plan["quarters"]:
        assert quarter["label"] in body, case["id"]


def test_every_load_answer_is_schedulable_on_every_quarter():
    """The walk-through must never offer a load it then refuses."""
    for track in TRACKS:
        for quarter in planner.adjustable_quarters(track):
            for option in planner.load_options_for(track, quarter["key"]):
                wanted, _ = planner.rebalanced_units(
                    track, {}, quarter["key"], option["units"])
                assert wanted is not None, (track, quarter["key"], option)
                assert not planner.quarter_units_problems(track, wanted), \
                    (track, quarter["key"], option, wanted)
