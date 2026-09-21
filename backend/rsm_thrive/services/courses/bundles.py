"""Fixed routes: the design document's per-profile elective bundles.

Two ways to fill 28 elective units, and this module is the first:

* **Fixed** -- take the bundle the design document specifies for a job profile
  (`data/catalog/bundles.json`), as an anchor / differentiator / universal set.
  Every student targeting that profile gets the same plan, which is what makes
  it defensible: "this IS the Product Analyst bundle".
* **Custom** -- `planner.build_plan`'s existing behaviour, where the scoring
  engine fills each slot against this student's skills, workload and interests.

## Why a bundle carries its own slot shape

`TRACK_SKELETONS` declares elective slots with fixed unit sizes -- `[4, 2]` in
Fall, `[4, 4]` in Winter -- and `build_plan` requires an exact unit match. That
shape is an invention: the published plans of study
(`data/corpus/program/`) give a UNIT TOTAL per quarter and never say how it
divides. Measured against the real shapes, 0 of 14 bundles could be placed on
the 17-month track and 6 of 14 on the 11-month, because every bundle's
universal layer is MGTA 402 (2u) + MGTA 460 (2u) and the 17-month track has no
2-unit slot at all.

Rather than rework the slot model for every plan, a fixed route derives its own
slots from the courses it already knows it contains, and hands them to
`build_plan` as an override. The custom route is untouched, so nothing about
today's behaviour changes for a student who does not choose a bundle.

The quarter's published unit budget is still the constraint -- a derived shape
must sum to exactly what `TRACK_SKELETONS` says that quarter carries.
"""

import itertools
import json
from functools import lru_cache
from pathlib import Path

from rsm_thrive.services.courses.electives import (catalog_version, load_catalog,
                                            resolve_role)

_DATA = Path(__file__).resolve().parents[2] / "data" / "catalog"

# The Summer slot is pre-placed by the published plan (MGTA 403 + MGTA 464), so
# a bundle never has to find room for 464 -- it is already in every plan.
PRE_PLACED = frozenset({"MGTA 403", "MGTA 464"})


@lru_cache(maxsize=4)
def _load_bundles(_version):
    raw = json.loads((_DATA / "bundles.json").read_text())
    return {key: value for key, value in raw.items() if not key.startswith("_")}


def load_bundles():
    """Profile id -> {anchor, differentiator, universal}. Keys starting with
    `_` are documentation, not profiles.

    Re-read when the file changes -- see `electives.catalog_version`."""
    return _load_bundles(catalog_version())


def bundle_for(role_id):
    """The bundle for a profile, resolving a retired role id forward."""
    resolved = resolve_role(role_id)
    return load_bundles().get(resolved) if resolved else None


@lru_cache(maxsize=4)
def _by_id_for(_version):
    return {course["id"]: course for course in load_catalog()}


def _by_id():
    return _by_id_for(catalog_version())


def _seasons(course_id):
    course = _by_id().get(course_id) or {}
    return {offering["season"] for offering in course.get("offerings") or []}


def _units(course_id):
    return (_by_id().get(course_id) or {}).get("units", 0)


def _quarter_budgets(skeleton, track):
    """Per quarter: its elective budget, and how far that may legally move.

    The budget is derived rather than read off the slots, because the slot
    shape is what this module replaces -- the budget is the part the plan of
    study actually states.

    `floor` and `ceiling` are the real limits on a quarter's ELECTIVE room,
    got by taking the quarter's own total bounds (`adjustable_quarters`, which
    encodes the 12-unit graduate minimum, the 18-unit cap and the short
    finishing term's own floor) and subtracting what its core courses cost.
    `QUARTER_FLEX` is then applied INSIDE them.

    Without the clamp the flex was free to run a quarter two units under its
    budget with nothing to stop it going under the enrolment minimum too:
    measured by the plan sweep, a light Fall on the 17-month track came back
    at 10 units, which is below the load a graduate student may enrol in.
    """
    from rsm_thrive.services.courses.planner import adjustable_quarters, tail_quarter_key

    bounds = {q["key"]: q for q in adjustable_quarters(track)}
    tail = tail_quarter_key(track)
    budgets = []
    for quarter in skeleton:
        spent = sum(_units(slot["course_id"]) for slot in quarter["slots"]
                    if slot["kind"] in ("core", "fixed"))
        room = bounds.get(quarter["key"])
        budgets.append({
            "key": quarter["key"], "season": quarter["season"],
            # The finishing quarter -- the one the load rule holds short.
            # Second-year Fall on the 17-month track, Spring on the 11-month.
            "tail": quarter["key"] == tail,
            "budget": quarter["units"] - spent,
            "floor": max(0, room["min"] - spent) if room else quarter["units"] - spent,
            "ceiling": (room["max"] - spent) if room else quarter["units"] - spent,
        })
    return budgets


# How far a quarter may sit from its published elective load. The TOTAL is
# still exact -- 24 units across the plan, which with the pre-placed Summer pair
# is the 28 the degree requires -- but a quarter may run 2 light or 2 heavy.
#
# Necessary, not cosmetic. Exact per-quarter fill places 5 of 14 bundles on the
# 11-month track and 1 of 14 on the 17-month, because the catalog does not
# offer the shapes it would need: Winter has exactly ONE 2-unit elective
# (MGTA 466), so a Winter budget of 8 can only ever be 4+4, and any bundle
# wanting 466 is unschedulable. With 2 units of give, all 14 place on both
# tracks.
#
# Defensible against the source, too: the plans of study are published as
# SAMPLES -- the app repeats their "confirm your schedule with MSBA advising"
# on every plan it renders -- and students do shift a course between quarters.
QUARTER_FLEX = 2


def _place(courses, budgets, flex=None, rigid=()):
    """Assign courses to quarters, or None if this set cannot be scheduled.

    The total must be spent exactly; each quarter may vary by `flex`, which
    defaults to `QUARTER_FLEX`, except the quarters named in `rigid`, which
    must hit their budget exactly. Most-constrained course first (fewest
    seasons, then largest), which keeps the search small enough to be
    exhaustive.

    `rigid` exists for the finishing quarter. This is a first-fit search, and
    with two units of give there are many valid assignments; the first one
    found fills the early quarters toward their ceilings and leaves the tail
    at its floor. Measured on the 17-month track: 11 of 14 bundles landed 2
    units in the final Fall on the PUBLISHED spread, whose budget there is 4
    -- not because 4 was impossible (Business Analyst has 457+402, or 459, or
    461 all offered then) but because the search never preferred it.
    """
    if flex is None:
        flex = QUARTER_FLEX
    rigid = set(rigid)
    order = sorted(courses, key=lambda c: (len(_seasons(c)), -_units(c)))
    assigned = {row["key"]: [] for row in budgets}
    load = {row["key"]: 0 for row in budgets}
    # The flex applies INSIDE each quarter's legal range, never outside it.
    give = {row["key"]: (0 if row["key"] in rigid else flex) for row in budgets}
    ceiling = {row["key"]: min(row["budget"] + give[row["key"]], row["ceiling"])
               for row in budgets}
    floor = {row["key"]: max(row["budget"] - give[row["key"]], row["floor"], 0)
             for row in budgets}
    season_of = {row["key"]: row["season"] for row in budgets}

    def recurse(index):
        if index == len(order):
            return all(floor[key] <= load[key] <= ceiling[key] for key in load)
        course = order[index]
        for key in load:
            if season_of[key] not in _seasons(course):
                continue
            if load[key] + _units(course) > ceiling[key]:
                continue
            load[key] += _units(course)
            assigned[key].append(course)
            if recurse(index + 1):
                return True
            assigned[key].pop()
            load[key] -= _units(course)
        return False

    return assigned if recurse(0) else None


def _fillable(budgets, exclude):
    """Per quarter: can a 4-unit / 2-unit elective from OUTSIDE this set fill it?

    A free slot is only worth cutting where the scorer can actually fill it --
    a 2-unit Winter slot has exactly one candidate in the catalog, and if the
    bundle already holds it the slot would come back "nothing available".
    """
    out = {}
    for row in budgets:
        sizes = {}
        for size in (4, 2):
            sizes[size] = any(
                course["units"] == size and not course["is_core"]
                and course["id"] not in exclude
                and any((o.get("season") or "").upper() == row["season"]
                        for o in course.get("offerings") or [])
                for course in load_catalog())
        out[row["key"]] = sizes
    return out


def _bounds(budgets, flex, rigid):
    """(floor, ceiling) per quarter, the flex applied INSIDE the legal range --
    exactly as `_place` does it, so a partial fill obeys the same limits."""
    rigid = set(rigid)
    give = {row["key"]: (0 if row["key"] in rigid else flex) for row in budgets}
    ceiling = {row["key"]: min(row["budget"] + give[row["key"]], row["ceiling"])
               for row in budgets}
    floor = {row["key"]: max(row["budget"] - give[row["key"]], row["floor"], 0)
             for row in budgets}
    return floor, ceiling


def _distribute(load, budgets, fill, target, floor, ceiling):
    """Free 4- and 2-unit slots that bring `load` up to `target` exactly.

    Published budget first, then up to each quarter's ceiling; every quarter
    must end at or above its floor. None when the leftover cannot be cut into
    fillable pieces.
    """
    leftover = target - sum(load.values())
    if leftover < 0:
        return None
    free = {row["key"]: [] for row in budgets}
    load = dict(load)
    by_room = sorted(budgets, key=lambda row: -(row["budget"] - load[row["key"]]))
    for limit in ("budget", "ceiling"):
        for size in (4, 2):
            for row in by_room:
                key = row["key"]
                cap = row["budget"] if limit == "budget" else ceiling[key]
                while (leftover >= size and load[key] + size <= cap
                       and fill[key][size]):
                    free[key].append(size)
                    load[key] += size
                    leftover -= size
    if leftover:
        return None
    if any(load[key] < floor[key] for key in load):
        return None
    return free


def _place_partial(courses, budgets, fill, target, flex, rigid=()):
    """Assign `courses` to quarters and fill what is left with free slots.

    (assigned, free) or None. Same per-quarter limits as `_place` -- the
    published budget give or take `flex`, inside the legal range, the `rigid`
    quarters exact -- so a plan built this way drifts no further from the
    chosen load than a fully placed bundle would. `_distribute` then has to
    close the total exactly with slots the scorer can fill, and every
    assignment is tried until one lets it.
    """
    order = sorted(courses, key=lambda c: (len(_seasons(c)), -_units(c)))
    assigned = {row["key"]: [] for row in budgets}
    load = {row["key"]: 0 for row in budgets}
    floor, ceiling = _bounds(budgets, flex, rigid)
    season_of = {row["key"]: row["season"] for row in budgets}
    found = {}

    def recurse(index):
        if index == len(order):
            free = _distribute(load, budgets, fill, target, floor, ceiling)
            if free is None:
                return False
            found["free"] = free
            return True
        course = order[index]
        for key in load:
            if season_of[key] not in _seasons(course):
                continue
            if load[key] + _units(course) > ceiling[key]:
                continue
            load[key] += _units(course)
            assigned[key].append(course)
            if recurse(index + 1):
                return True
            assigned[key].pop()
            load[key] -= _units(course)
        return False

    return (assigned, found["free"]) if recurse(0) else None


def _rebudgeted(skeleton, wanted_units):
    """The published skeleton with each quarter's TOTAL set to the chosen load.

    Only the totals move; the core and fixed slots are copied through
    untouched. `_quarter_budgets` then derives each quarter's elective room
    from the total the student actually chose rather than from the published
    one, which is what makes a bundle and a light or heavy spread coexist.
    """
    if not wanted_units:
        return skeleton
    return [{**quarter, "units": wanted_units.get(quarter["key"], quarter["units"])}
            for quarter in skeleton]


#: Prefixes whose anchors are swapped for an MSBA course when one exists.
#: CSE is NOT here, and that is the whole distinction: the design document
#: gates `data-scientist` and `ml-engineer` on real computer-science
#: coursework, and an all-MGTA "AI/ML Engineer" plan would be a plan that
#: does not reach the job. MGT and MGTF anchors are electives the MSBA can
#: cover from its own catalog.
SUBSTITUTABLE_PREFIXES = ("MGT ", "MGTF")


def _substitute_for(role_id, course_id, used):
    """The MSBA course that best serves THIS ROLE in that slot, or None.

    Chosen by the role's own ranking rather than by resemblance to the course
    being replaced. Measured, the resemblance route is weak -- the closest
    MGTA course to MGT 477 Consumer Behavior scores 0.27 on skill overlap, so
    "most similar" would be picking a barely-related course and implying it
    teaches the same thing. What the student actually wants in that slot is
    the best remaining course FOR THEIR GOAL, which the elective scorer
    already knows how to name.

    Same unit count, always: the bundle's arithmetic and the quarter budgets
    are built on it, and a 2-unit swap for a 4-unit anchor unbalances a plan
    that has to close at exactly 50.
    """
    from rsm_thrive.services.courses.electives import (load_careers, load_catalog,
                                               rank_electives)

    catalog = _by_id()
    original = catalog.get(course_id)
    if not original:
        return None
    ranked = rank_electives(
        load_catalog(),
        {"career_roles": [role_id], "career_tags": [], "technical_comfort": 3,
         "workload_preference": "moderate", "interests": []},
        load_careers())
    for row in ranked:
        candidate = row["course"]
        if not candidate["code"].startswith("MGTA"):
            continue
        if candidate["id"] in used or candidate["id"] in PRE_PLACED:
            continue
        if candidate.get("units") != original.get("units"):
            continue
        # And OFFERED when the original was. Matching units alone is not
        # enough: the placement engine has to put the course in a quarter, and
        # a substitute taught only in Spring cannot stand in for an anchor the
        # plan needs in Fall. Ignoring this returned course sets that no
        # arrangement could schedule, `choose_courses` gave up, and the plan
        # fell through to the generic scorer fill -- which is the same for
        # neighbouring roles, so Marketing and Pricing came out identical.
        if not (_seasons(original["id"]) <= _seasons(candidate["id"])):
            continue
        return candidate["id"]
    return None


def _prefer_msba(role_id, required, taken_ids):
    """Swap MGT/MGTF anchors for MSBA courses. Returns (courses, swaps)."""
    out, swaps = [], []
    used = set(required) | set(taken_ids)
    for course_id in required:
        if not course_id.startswith(SUBSTITUTABLE_PREFIXES):
            out.append(course_id)
            continue
        replacement = _substitute_for(role_id, course_id, used)
        if not replacement:
            out.append(course_id)
            continue
        used.discard(course_id)
        used.add(replacement)
        out.append(replacement)
        swaps.append((course_id, replacement))
    return out, swaps


def choose_courses(role_id, track, taken_ids=frozenset(), wanted_units=None):
    """The bundle's courses for one track, sized to fit and placeable.

    Anchor and universal are mandatory -- they are what makes the plan that
    profile rather than a neighbouring one. Differentiators are added until the
    elective budget is exactly met, and the design document expects that: most
    bundles list more differentiators than fit, with "drop one" as the
    instruction.

    Returns (courses, placement) or (None, None) when no combination fits,
    which is a real answer rather than a failure -- it means this bundle cannot
    be scheduled on this track once the student's completed courses are
    excluded.
    """
    from rsm_thrive.services.courses.planner import TRACK_SKELETONS

    bundle = bundle_for(role_id)
    skeleton = TRACK_SKELETONS.get(track)
    if not bundle or skeleton is None:
        return None, None, None

    budgets = _quarter_budgets(_rebudgeted(skeleton, wanted_units), track)
    target = sum(row["budget"] for row in budgets)

    def usable(course_id):
        return (course_id in _by_id() and course_id not in PRE_PLACED
                and course_id not in taken_ids)

    required = [c for c in bundle["anchor"] + bundle["universal"] if usable(c)]
    required = list(dict.fromkeys(required))
    return _search(role_id, required, bundle, budgets, target, taken_ids, usable)


def _search(role_id, required, bundle, budgets, target, taken_ids, usable):
    """The combination search, for one `required` spine.

    Returns (courses, placement, free): the bundle courses chosen, where each
    sits, and any free slots {quarter: [units, ...]} left for the scorer.
    `free` is empty whenever the whole set could be placed.
    """
    optional = [c for c in bundle["differentiator"]
                if usable(c) and c not in required]
    optional = list(dict.fromkeys(optional))
    needed = target - sum(_units(c) for c in required)
    if needed < 0:
        return None, None, None
    no_free = {row["key"]: [] for row in budgets}

    # Exact per-quarter loads FIRST, then with the flex.
    #
    # The flex exists because exact fill places only 5 of 14 bundles on the
    # 11-month track and 1 of 14 on the 17-month. But it is a concession, not a
    # preference: once the curated bundle became the default fill, every
    # student was getting a plan whose quarters could sit 2 units off the load
    # they had just chosen, including the ones where an exact fit was available
    # all along. Trying 0 first costs one extra search over a space small
    # enough to be exhaustive, and `flex_used` lets the plan say when the
    # concession was actually taken.
    #
    # Smallest addition first: the anchor is the profile, so a bundle that
    # closes on fewer differentiators is the more faithful one.
    # Fewest courses from OUTSIDE the MSBA first, within each size.
    #
    # A differentiator is chosen to make the units add up, and where several
    # combinations do that equally well the programme's own courses are the
    # ones a student can rely on registering for. So "MGTF 405 or MGTA 461,
    # both 4 units, both close the budget" resolves to the MGTA one rather
    # than to whichever `itertools` happened to emit first.
    #
    # This does NOT touch the anchor. An anchor is what makes the plan that
    # profile rather than a neighbouring one -- `ml-engineer` is three CSE
    # courses and dropping them would leave a bundle that is no longer AI/ML
    # Engineering. Those survive here and are flagged where they are shown.
    def outside_first(combination):
        return (sum(1 for c in combination if not c.startswith("MGTA")),
                combination)

    # THE LADDER. Exact everywhere first; then the FINISHING QUARTER exact
    # while the full quarters take the flex; then everything flexes. The
    # middle rung is what holds the tail at its budget -- 4 units on a light
    # or moderate spread, 2 on a heavy one -- and the last rung is what keeps
    # every bundle that placed before still placing.
    tail_keys = {row["key"] for row in budgets if row.get("tail")}
    rungs = [(0, ()), (QUARTER_FLEX, tail_keys), (QUARTER_FLEX, ())]
    if not tail_keys:
        rungs = [(0, ()), (QUARTER_FLEX, ())]
    for flex, rigid in rungs:
        for count in range(len(optional) + 1):
            for combination in sorted(
                    itertools.combinations(optional, count), key=outside_first):
                if sum(_units(c) for c in combination) != needed:
                    continue
                courses = required + list(combination)
                placement = _place(courses, budgets, flex, rigid)
                if placement is not None:
                    return courses, placement, no_free

    # THE PARTIAL RUNGS. The rungs above need the differentiators to close
    # the budget exactly, in the quarters everything is taught. With the
    # catalog holding each course's REAL terms rather than "varies" for every
    # quarter, that is often impossible -- Business Analyst's spine plus every
    # differentiator it lists leaves Spring six units short, because none of
    # them runs in Spring. The whole bundle was then abandoned for the generic
    # ranked fill, silently, for 54 of 84 role/track/load combinations.
    #
    # So: keep the spine and as many differentiators as the timetable holds
    # (most first, MSBA courses first among equals), and cut FREE slots for
    # the rest, which `build_plan` fills with the best fit for the goal. Only
    # if even the spine cannot be laid in are anchors given up, fewest first
    # -- the universal pair is never dropped. Whatever is left out is reported
    # on the plan (`bundleUnplaced`), never just missing.
    fill = _fillable(budgets, set(required) | set(optional) | set(taken_ids)
                     | PRE_PLACED)
    anchors = [c for c in required if c in (bundle.get("anchor") or [])]
    spines = [required]
    for drop_count in range(1, len(anchors) + 1):
        # Give up the courses from OUTSIDE the MSBA first. Data Scientist on
        # the 11-month track has MGTA 461, CSE 251A and MGTA 402 all taught in
        # Fall only, and Fall cannot hold the three; the programme's own
        # course is the one a student can rely on registering for.
        for dropped in sorted(itertools.combinations(anchors, drop_count),
                              key=lambda combo: (
                                  -sum(1 for c in combo if not c.startswith("MGTA")),
                                  combo)):
            spines.append([c for c in required if c not in dropped])
    for spine in spines:
        room = target - sum(_units(c) for c in spine)
        if room < 0:
            continue
        for flex, rigid in rungs:
            for count in range(len(optional), -1, -1):
                for combination in sorted(
                        itertools.combinations(optional, count), key=outside_first):
                    if sum(_units(c) for c in combination) > room:
                        continue
                    courses = spine + list(combination)
                    result = _place_partial(courses, budgets, fill, target,
                                            flex, rigid)
                    if result is not None:
                        placement, free = result
                        return courses, placement, free
    return None, None, None


def skeleton_for(role_id, track, taken_ids=frozenset(), wanted_units=None):
    """`TRACK_SKELETONS[track]`, with elective slots shaped to this bundle.

    Core and fixed slots are copied through untouched -- the published sequence
    is not this module's to change. Only the elective slots are replaced, by
    one slot per bundle course sized to that course.

    `wanted_units` is the student's chosen per-quarter load, and threading it
    through is what stops a bundle and a load spread destroying each other.
    Without it the shape was derived from the PUBLISHED budgets and then
    `planner._resized` re-cut it to the chosen ones -- which discards the sizes
    the pinned courses need. Measured by the plan sweep: a curated role with a
    heavy spread on the 17-month track came back with 26 elective units instead
    of 28, i.e. a 48-unit degree plan.

    Each shaped quarter also carries the chosen total, so `_resized` sees a
    skeleton already at the target and leaves it alone.
    """
    from rsm_thrive.services.courses.planner import TRACK_SKELETONS

    skeleton = TRACK_SKELETONS.get(track)
    if skeleton is None:
        return None, None, None
    skeleton = _rebudgeted(skeleton, wanted_units)
    courses, placement, free = choose_courses(role_id, track, taken_ids,
                                              wanted_units)
    if courses is None:
        return None, None, None

    shaped, selections = [], {}
    for quarter in skeleton:
        slots = [dict(slot) for slot in quarter["slots"]
                 if slot["kind"] in ("core", "fixed")]
        pinned = placement.get(quarter["key"], [])
        # Largest first, so a quarter reads big-course-then-small rather than
        # in whatever order the search happened to assign.
        pinned = sorted(pinned, key=lambda c: (-_units(c), c))
        for position, course_id in enumerate(pinned):
            slots.append({"kind": "elective", "units": _units(course_id)})
            selections.setdefault(quarter["key"], {})[
                str(len(slots) - 1)] = course_id
        # Free slots -- the part of the budget the bundle could not fill in
        # this quarter. Unpinned, so `build_plan` fills them by fit.
        for units in sorted((free or {}).get(quarter["key"], []), reverse=True):
            slots.append({"kind": "elective", "units": units})
        shaped.append({**quarter, "slots": slots})
    return shaped, selections, courses


def absent(plan, role_id):
    """Bundle courses this plan does NOT contain, by layer.

    `divergence` answers a narrower question -- which ANCHOR courses a student
    has swapped away from -- and answering only that hid a second way to lose
    part of a bundle. A light or heavy load spread re-cuts every quarter's
    elective slots (`planner._resized`), and a bundle whose universal layer is
    two 2-unit courses can come out of that with nowhere to put them:
    measured, a light spread on the 11-month Data Scientist bundle dropped
    MGTA 402 and MGTA 460 and filled the space with a scored pick, and
    `divergence` reported nothing because neither is an anchor.

    Returns {} for a bundle that is fully present, so a caller can treat it as
    "nothing to say".
    """
    bundle = bundle_for(role_id)
    if not bundle:
        return {}
    present = {row["courseId"] for quarter in plan["quarters"]
               for row in quarter["courses"] if row.get("courseId")}
    catalog = _by_id()
    out = {}
    for layer in ("anchor", "differentiator", "universal"):
        missing = [catalog[c]["code"] for c in (bundle.get(layer) or [])
                   if c not in present and c not in PRE_PLACED and c in catalog]
        if missing:
            out[layer] = missing
    # The differentiator layer is a MENU, not a checklist -- a bundle lists
    # more of them than a plan has room for, so some are always absent and
    # saying so every time would be noise.
    out.pop("differentiator", None)
    return out


def divergence(plan, role_id):
    """Bundle courses the student has swapped away from, by code.

    A fixed route the student has edited is no longer quite that bundle, and
    the design document is explicit that a recommendation which does not say
    what it gives up is not a recommendation. This is what the reply uses to
    say so.
    """
    bundle = bundle_for(role_id)
    if not bundle:
        return []
    present = {row["courseId"] for quarter in plan["quarters"]
               for row in quarter["courses"] if row.get("courseId")}
    catalog = _by_id()
    missing = [c for c in bundle["anchor"]
               if c not in present and c not in PRE_PLACED]
    return [catalog[c]["code"] for c in missing if c in catalog]
