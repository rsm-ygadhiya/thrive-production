"""Intake-driven course planner: interview the student, then build a full plan.

## Why this is not the electives bot

`services/electives.py` answers "which electives suit my career goal" in one
turn. It already accepts a rich profile — technical comfort, workload
tolerance, weighted career tags — but `recommend_for` hardcodes comfort 3 and
"moderate" workload, because a single chat turn has nowhere to ask. So the
engine was always able to personalise and never had the data.

This module supplies that data. It asks first (`intake_questions`), maps the
answers onto the profile the scorer already understands (`profile_from_intake`),
and then fills a real plan of study quarter by quarter rather than returning a
flat top-five.

## The plan is a skeleton with holes, not a free-for-all

The quarter structure, the core courses and the unit size of every elective
slot come from Rady's published plans of study (data/corpus/program/). A
student does not get to choose whether MGTA 451 is in their first summer, and
a 2-unit hole cannot be filled with a 4-unit course. Encoding the skeleton
means the planner cannot produce a schedule that would not be approved: the
core sequence is fixed, and each generated plan totals exactly 50 units —
22 core and 28 elective — which `tests/test_planner.py` asserts rather than
trusts.

## Deterministic, like the scorer it sits on

No LLM decides what goes in a slot. Same answers in, same plan out, which is
what makes a swap explainable ("these two courses teach these same skills")
and the whole thing testable. The LLM's job, if it is used at all, is to talk
about a plan this module produced.
"""
import json
import collections
import re
from functools import lru_cache

from rsm_thrive.models import Enrollment
from rsm_thrive.services.courses.electives import (DEPARTMENT_LABELS, WORKLOAD_LEVEL,
                                            catalog_version, is_msba,
                                            load_careers, load_catalog,
                                            rank_electives, resolve_role)

# ---------------------------------------------------------------------------
# The published skeletons
# ---------------------------------------------------------------------------

# `core` slots are the 22 required units and are never swappable. `fixed` slots
# are elective-labelled in the published plan but effectively determined:
# Summer is the only quarter with almost no elective offerings, so there is
# nothing to swap them for, and saying so is more honest than offering a choice
# that does not exist.
TRACK_SKELETONS = {
    "11 month": [
        {"key": "summer", "label": "Summer III", "season": "SU", "units": 8, "slots": [
            {"kind": "core", "course_id": "MGTA 451"},
            {"kind": "fixed", "course_id": "MGTA 403"},
            {"kind": "fixed", "course_id": "MGTA 464"},
        ]},
        {"key": "fall", "label": "Fall", "season": "FA", "units": 14, "slots": [
            {"kind": "core", "course_id": "MGTA 452"},
            {"kind": "core", "course_id": "MGTA 453"},
            {"kind": "elective", "units": 4},
            {"kind": "elective", "units": 2},
        ]},
        {"key": "winter", "label": "Winter", "season": "WI", "units": 14, "slots": [
            {"kind": "core", "course_id": "MGTA 455"},
            {"kind": "core", "course_id": "MGTA 444"},
            {"kind": "elective", "units": 4},
            {"kind": "elective", "units": 4},
        ]},
        {"key": "spring", "label": "Spring", "season": "SP", "units": 14, "slots": [
            {"kind": "core", "course_id": "MGTA 454"},
            {"kind": "elective", "units": 4},
            {"kind": "elective", "units": 4},
            {"kind": "elective", "units": 2},
        ]},
    ],
    "17 month": [
        {"key": "summer", "label": "Summer III", "season": "SU", "units": 8, "slots": [
            {"kind": "core", "course_id": "MGTA 451"},
            {"kind": "fixed", "course_id": "MGTA 403"},
            {"kind": "fixed", "course_id": "MGTA 464"},
        ]},
        {"key": "fall", "label": "Fall", "season": "FA", "units": 12, "slots": [
            {"kind": "core", "course_id": "MGTA 452"},
            {"kind": "core", "course_id": "MGTA 453"},
            {"kind": "elective", "units": 4},
        ]},
        {"key": "winter", "label": "Winter", "season": "WI", "units": 14, "slots": [
            {"kind": "core", "course_id": "MGTA 455"},
            {"kind": "core", "course_id": "MGTA 444"},
            {"kind": "elective", "units": 4},
            {"kind": "elective", "units": 4},
        ]},
        {"key": "spring", "label": "Spring", "season": "SP", "units": 12, "slots": [
            {"kind": "core", "course_id": "MGTA 454"},
            {"kind": "elective", "units": 4},
            {"kind": "elective", "units": 4},
        ]},
        {"key": "fall-two", "label": "Fall (second year)", "season": "FA", "units": 4, "slots": [
            {"kind": "elective", "units": 4},
        ]},
    ],
}

CORE_UNITS = 22
ELECTIVE_UNITS = 28
TOTAL_UNITS = 50

# Said wherever this app names a course. ONE string, used by the plan, the
# recommended set and the web-matched set alike, so the claim cannot be strong
# in one place and absent in another.
#
# It has to be evident rather than merely true. Every one of these replies is a
# recommendation and none of them is a decision -- the programme office and
# advising decide what a student may actually enrol in -- and a reply that
# lists eight courses under a confident heading reads as an instruction unless
# it says otherwise in the same breath.
RECOMMENDATION_ONLY = (
    "_This is a recommendation, not a decision. What you can actually enrol in "
    "is settled by the programme office and MSBA advising — confirm with them "
    "before you book._")

# Where a student goes to ACT on a plan. Every URL here is one the corpus
# already carries (crawled/Graduate Enrollment, crawled/Booking Your Classes,
# corpus/program) rather than one guessed from memory: a plan that sends a
# student to a wrong booking page is worse than one that sends them nowhere.
ACTION_LINKS = [
    {"key": "tss", "label": "Book these courses in TSS",
     "url": "https://sis.ucsd.edu/",
     "note": "Triton Student System — where graduate students book classes."},
    {"key": "webreg", "label": "WebReg (Summer)",
     "url": "https://act.ucsd.edu/webreg2",
     "note": "Summer sessions are booked in WebReg rather than TSS."},
    {"key": "schedule", "label": "Schedule of Classes",
     "url": "https://act.ucsd.edu/cgi-bin/tritonlink.pl/2/students/academic/classes/schedule_of_classes.pl",
     "note": "Confirm the day, time and seat count before you book."},
    {"key": "booking-help", "label": "How booking works in TSS",
     "url": "https://students.ucsd.edu/my-tritonlink/tools/tool-help/booking.html",
     "note": "Step-by-step help if a booking is refused."},
    {"key": "grad-enrollment", "label": "Graduate enrollment rules",
     "url": "https://students.ucsd.edu/academics/enroll/graduate-enrollment/index.html",
     "note": "Unit minimums, deadlines and holds."},
    {"key": "plans-drive", "label": "Official plans of study (Rady Drive)",
     "url": "https://drive.google.com/drive/folders/1NkOVx60spVY31IHWZ7KWlGXBiXLL6Xk6",
     "note": "The published plan this schedule follows. Sign in with your UCSD "
             "account — see THRIVE's syllabus guidance if you hit a permission wall."},
]

# ---------------------------------------------------------------------------
# The intake
# ---------------------------------------------------------------------------

# Declared skill, per area, on a scale the prerequisite text can be checked
# against. "comfortable" is 4 rather than 3 deliberately: a student who says
# they are comfortable with Python should not be steered away from the courses
# that use it, and the scorer penalises any course above the stated comfort.
# A 1-5 self-rating per area. Five points rather than the four named levels this
# started with, because `rank_electives` already scores `technical_comfort` on
# 1-5: a rating maps straight through instead of being translated by a table,
# and a student who wants the middle of the range now has one.
SKILL_SCALE = [
    {"value": 1, "label": "1", "help": "New to it"},
    {"value": 2, "label": "2", "help": "Some exposure"},
    {"value": 3, "label": "3", "help": "Working knowledge"},
    {"value": 4, "label": "4", "help": "Comfortable"},
    {"value": 5, "label": "5", "help": "Advanced"},
]

# Where the rating form starts before a student touches it. The middle, so
# adjusting means moving in whichever direction is true rather than starting
# from a claim the student has not made.
DEFAULT_SKILL_RATING = 3

# The words students actually type, kept as accepted input alongside the
# numbers. Dropping them would mean "python is comfortable" — which is how
# people answer this question in prose — stopped being understood the moment
# the buttons arrived.
SKILL_WORDS = {
    "none": 1, "beginner": 1, "new": 1,
    "basic": 2, "some": 2,
    "working": 3, "moderate": 3, "okay": 3, "ok": 3,
    "comfortable": 4, "good": 4, "strong": 4,
    "advanced": 5, "expert": 5,
}


def skill_score(value):
    """A rating as 1-5, from either a number or a word. None when unreadable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 1 <= value <= 5:
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text.isdigit() and 1 <= int(text) <= 5:
            return int(text)
        return SKILL_WORDS.get(text)
    return None


def skill_label(value):
    """How a rating reads back to the student."""
    score = skill_score(value)
    if score is None:
        return str(value)
    entry = next(s for s in SKILL_SCALE if s["value"] == score)
    return f"{score} ({entry['help'].lower()})"

# The areas the catalog's prerequisites actually talk about. Each carries the
# words used to spot that demand in a course's free-text prerequisite line.
SKILL_AREAS = [
    {"key": "python", "label": "Python programming",
     "prereq_words": ["python", "programming", "pyspark", "coding"]},
    {"key": "sql", "label": "SQL and databases",
     "prereq_words": ["sql", "database", "etl"]},
    {"key": "stats", "label": "Statistics and regression",
     "prereq_words": ["statistics", "statistical", "regression", "probability"]},
    {"key": "ml", "label": "Machine learning",
     "prereq_words": ["machine learning", "supervised", "deep learning"]},
    {"key": "communication", "label": "Presenting and storytelling",
     "prereq_words": []},
]

# Communication is a real skill and not a proxy for technical comfort, so it is
# excluded from the comfort average — otherwise a strong presenter reads as
# ready for PySpark.
TECHNICAL_AREAS = [area["key"] for area in SKILL_AREAS if area["prereq_words"]]

# ---------------------------------------------------------------------------
# Units per quarter, which replaced "light / moderate / heavy"
# ---------------------------------------------------------------------------
#
# The old question asked how hard a load the student wanted and answered it with
# a word. A word cannot be scheduled: it fed the scorer a tolerance and told the
# student nothing about what their year would look like. Units per quarter is
# the same question asked concretely -- it is the number a student actually
# feels, and it decides how many courses land in each term.
#
# The floor is what the university enforces rather than a preference. Full-time
# graduate standing is 12 units a quarter, so a slider that can go below it
# would offer something a student cannot take.
QUARTER_UNIT_MIN = 12
QUARTER_UNIT_MAX = 18

# The FINISHING quarter is the exception. Each track ends on one -- the
# 17-month track on a second-year Fall, the 11-month track on Spring -- and
# it is the quarter the light/moderate/heavy rule holds short (`TAIL_UNITS`),
# so the 12-unit floor cannot apply to it. What does apply is its own core
# load plus at least one elective: `FINAL_QUARTER_MIN` elective units.
#
# `FINAL_QUARTER_MAX` caps a tail whose PUBLISHED load is already short (the
# 17-month's 4-unit Fall). The 11-month Spring publishes at 14, a full
# quarter, and keeps the ordinary 18-unit cap so a student who wants a heavy
# Spring in the walk-through can still have one.
FINAL_QUARTER_MIN = 2
FINAL_QUARTER_MAX = 8


# Tracks where the load is a CHOICE. The 11-month track is not one: the
# programme is compressed into four quarters and every one of them carries
# what the plan of study publishes -- the degree has to be completed in the
# time, so there is no light or heavy version to offer, and asking would be a
# question with one answer. The 17-month track has the room, and keeps the
# question: its overall spread, the finishing-quarter rule and the per-quarter
# walk-through prompt all apply there only.
LOAD_IS_A_CHOICE = {"11 month": False, "17 month": True}


def load_is_a_choice(track):
    return LOAD_IS_A_CHOICE.get(track, True)


def tail_quarter_key(track):
    """The finishing quarter: the LAST one on the track with elective room.

    "Fall (second year)" on the 17-month track, "Spring" on the 11-month --
    which ends before any second-year Fall. It used to be found by published
    load ("under 12 units"), which found the 17-month tail and declared the
    11-month track had none; a rule about the finishing quarter has to find
    the finishing quarter on every track.
    """
    for quarter in reversed(TRACK_SKELETONS.get(track) or []):
        if any(slot["kind"] == "elective" for slot in quarter["slots"]):
            return quarter["key"]
    return None


def adjustable_quarters(track):
    """Quarters whose load the student may move, with their published default.

    Summer is excluded and cannot be adjusted: the published plan fills it
    entirely with required courses (MGTA 451, 403 and 464), so there is no
    elective room to move and nothing to decide.
    """
    skeleton = TRACK_SKELETONS.get(track) or []
    catalog = _catalog_by_id()
    tail = tail_quarter_key(track)
    out = []
    for quarter in skeleton:
        electives = [slot for slot in quarter["slots"] if slot["kind"] == "elective"]
        if not electives:
            continue
        spent = sum(catalog[slot["course_id"]]["units"] for slot in quarter["slots"]
                    if slot["kind"] in ("core", "fixed"))
        # The finishing quarter's floor is its core load plus one elective,
        # not the 12-unit full-time minimum: it is the quarter the load rule
        # holds at 4 (or 2) elective units. See `tail_quarter_key`.
        last = quarter["key"] == tail and load_is_a_choice(track)
        short = quarter["units"] < QUARTER_UNIT_MIN
        out.append({
            "key": quarter["key"], "label": quarter["label"],
            "default": quarter["units"],
            "min": spent + FINAL_QUARTER_MIN if last else max(QUARTER_UNIT_MIN, spent),
            "max": FINAL_QUARTER_MAX if short else QUARTER_UNIT_MAX,
            # A quarter can never go below what its own core courses cost.
            "core_units": spent,
        })
    return out


def fixed_quarter_units(track):
    """Units in quarters the student cannot move (Summer)."""
    adjustable = {q["key"] for q in adjustable_quarters(track)}
    return sum(q["units"] for q in (TRACK_SKELETONS.get(track) or [])
               if q["key"] not in adjustable)


def quarter_units_of(answers):
    """The student's chosen load per quarter, or the published default.

    Absent means the published plan, which is what every plan built before this
    question existed used.

    A stored distribution that does not WORK for this track is discarded here
    rather than trusted. The quarter keys differ between tracks -- only the
    17-month one has a second Fall -- so a spread chosen on one track and read
    on the other silently loses a quarter's worth of units: measured, a heavy
    17-month spread {fall 16, winter 12, spring 12, fall-two 2} read on the
    11-month track came to 40 against the 42 it needs, and produced a 48-unit
    degree plan.

    `orchestrator` also clears the spread when the track changes, which is the
    right place to fix the CAUSE. This is the floor under it: `quarter_units`
    is a plain dict on a JSONField that a saved plan, a `POST /plan` body or a
    half-finished conversation can all supply, and none of those routes should
    be able to produce a plan that does not add up.
    """
    track = (answers or {}).get("track") or "11 month"
    chosen = (answers or {}).get("quarter_units") or {}
    published = {q["key"]: q["default"] for q in adjustable_quarters(track)}
    if not chosen:
        # No per-quarter choice, but an overall load: the same distribution
        # the conversation seeds from that word (`seeded_units`), so a plan
        # built straight from {track, goals, workload} -- `POST /plan`, a
        # test, a session the orchestrator has not yet passed through -- is
        # the plan the student would see in the chat. Before this, the
        # 11-month "moderate" plan was the published 14/14/14 here and the
        # held-Spring 18/16/8 there, and the two disagreed about Spring.
        return seeded_units(track, (answers or {}).get("workload")) or published
    wanted = {key: int(units) if isinstance(units, (int, float))
              and not isinstance(units, bool) else default
              for key, default in published.items()
              for units in [chosen.get(key, default)]}
    return published if quarter_units_problems(track, wanted) else wanted


def quarter_units_problems(track, chosen):
    """Human-readable reasons a distribution cannot be scheduled, if any."""
    problems = []
    quarters = adjustable_quarters(track)
    if not quarters:
        return problems
    for quarter in quarters:
        value = chosen.get(quarter["key"])
        if not isinstance(value, int):
            problems.append(f"{quarter['label']}: give a number of units")
            continue
        if value < quarter["min"] or value > quarter["max"]:
            problems.append(
                f"{quarter['label']}: {quarter['min']}-{quarter['max']} units")
    total = sum(v for v in chosen.values() if isinstance(v, int))
    required = TOTAL_UNITS - fixed_quarter_units(track)
    if not problems and total != required:
        problems.append(
            f"those add up to {total + fixed_quarter_units(track)} units and the "
            f"degree is {TOTAL_UNITS} — move {abs(required - total)} "
            f"{'out of' if total > required else 'into'} a quarter")
    return problems


# ---------------------------------------------------------------------------
# Light / moderate / heavy
# ---------------------------------------------------------------------------

# What the three words mean, per quarter. Read off `adjustable_quarters`
# rather than written down, so a quarter's floor is always its own core load
# and its ceiling is always the enrolment cap -- neither is a number this
# module gets to invent.
LOAD_BOUNDS = {"light": "min", "moderate": "default", "heavy": "max"}

# The words a student uses for them. Whole-message matching, like
# `review_intent`, and for the same reason: these arrive as one-word answers
# to a question that was just asked, and substring matching would read them
# out of ordinary sentences -- "is next quarter heavy?" is a question about
# the plan, not an instruction to change it.
LOAD_WORDS = {
    "light": "light", "lighter": "light", "lightest": "light",
    "go light": "light", "make it light": "light", "keep it light": "light",
    "light please": "light", "a light one": "light", "light quarter": "light",
    "moderate": "moderate", "medium": "moderate", "normal": "moderate",
    "standard": "moderate", "as is": "moderate", "leave it": "moderate",
    "keep it as is": "moderate", "moderate please": "moderate",
    "heavy": "heavy", "heavier": "heavy", "heaviest": "heavy", "hard": "heavy",
    "go heavy": "heavy", "make it heavy": "heavy", "heavy please": "heavy",
    "a heavy one": "heavy", "heavy quarter": "heavy", "push me": "heavy",
}


def load_intent(text):
    """"light" / "moderate" / "heavy" from a whole message, or None.

    Whole-message matching only. Substring matching would read an instruction
    out of an ordinary sentence -- "is next quarter heavy?" is a question about
    the plan, and `is_question` catches that one because it opens with an
    auxiliary.

    A bare "heavy?" DOES count, and that is deliberate rather than an
    oversight: `is_question` already treats it as a hedged answer, on the
    argued grounds that an answer starts with its own value where a question
    starts with an auxiliary. A student replying "heavy?" to "how heavy do you
    want Fall?" is answering it.
    """
    lowered = re.sub(r"[^a-z0-9' ]+", " ", (text or "").strip().lower())
    lowered = " ".join(lowered.split())
    if not lowered or is_question(text):
        return None
    return LOAD_WORDS.get(lowered)


def load_mentioned(text):
    """A load word anywhere in a sentence, or None.

    The looser sibling of `load_intent`, for a turn that is already stating
    intake facts: "11 month, data scientist, heavy" names all three, and
    whole-message matching finds none of them because the message is not the
    word "heavy". Callers must only use this on a turn that has stated a track
    or a goal (see `orchestrator.learned_from`) -- applied to any message it
    would read "MGTA 456 is heavy" as a preference.

    Still refuses a question, so "are these courses light?" states nothing.
    """
    if is_question(text):
        return None
    # Every single-word key in LOAD_WORDS, so "lightest" and "heaviest" read.
    # The list was hand-typed and missed both: "the lightest possible
    # schedule" -- about as clear a preference as a student can state --
    # was read as no preference at all.
    words = "|".join(sorted((w for w in LOAD_WORDS if " " not in w), key=len,
                            reverse=True))
    match = re.search(rf"\b({words})\b", (text or "").lower())
    return LOAD_WORDS.get(match.group(1)) if match else None


def units_for_load(track, key, load, chosen=None, pinned=None):
    """The unit count `load` means for one quarter -- one that can be SCHEDULED.

    Clamped toward the quarter's published load until the rest of the degree
    can absorb it, because a quarter's own enrolment ceiling is not the real
    ceiling. On the 17-month track every quarter's cap is 18 units and NONE of
    them can reach it: 18 in Fall leaves 24 for the other three, whose floors
    add up to 26. Reading the cap literally offered that Fall only "light",
    when in fact it can carry 16 -- so the heaviest thing the student could
    have was the one option they were not shown.

    Returns None when even the published load is unreachable, which cannot
    happen for a distribution the published plan itself produces.
    """
    bound = LOAD_BOUNDS.get(load)
    if bound is None or not load_is_a_choice(track):
        return None
    quarter = next((q for q in adjustable_quarters(track) if q["key"] == key), None)
    if quarter is None:
        return None
    target, default = quarter[bound], quarter["default"]
    step = 2 if target < default else -2
    units = target
    while True:
        if rebalanced_units(track, chosen or {}, key, units, pinned)[0] is not None:
            return units
        if units == default:
            return None
        units += step
        # Never step past the published load: that is the moderate answer, and
        # walking through it would return a value the student did not ask for.
        if (step > 0 and units > default) or (step < 0 and units < default):
            return default


def load_options_for(track, key, chosen=None, pinned=None):
    """The choices for one quarter that can ACTUALLY be scheduled.

    Two things get filtered out, and both would otherwise be offers the plan
    cannot honour:

    * A load whose unit count another option already offers. A quarter whose
      published load is its own floor has no lighter version, and offering the
      same number twice under two names is not a choice.
    * A load the rest of the degree cannot absorb. The 17-month track's second
      Fall cannot go to 8 units, because the three quarters before it are
      already at their 12-unit floors and the degree is fixed at 50 -- so
      "heavy" is not on offer there, rather than being offered and then
      refused.

    The numbers are named because the words alone are not a choice: "light"
    means 12 units in Fall and 2 in that second Fall.
    """
    quarter = next((q for q in adjustable_quarters(track) if q["key"] == key), None)
    if quarter is None:
        return []
    seen, out = set(), []
    for load in LOAD_BOUNDS:
        units = units_for_load(track, key, load, chosen, pinned)
        if units is None or units in seen:
            continue
        seen.add(units)
        out.append({"value": load, "units": units,
                    "label": f"{load} — {units} units"})
    return out


def rebalanced_units(track, chosen, key, target, pinned=None):
    """(distribution, blockers) for moving one quarter to `target`.

    Three outcomes, and the middle one is the interesting one:

    * ``(dict, [])`` -- done, absorbed by quarters the student has not decided
      about.
    * ``(None, ["fall"])`` -- possible ONLY by moving a quarter they already
      chose a load for. Refused here and explained by the caller, because
      undoing their earlier decision to satisfy a later one is not a thing to
      do silently, and it is not a thing to do at all without asking.
    * ``(None, [])`` -- not possible at any price.

    THE constraint, and the reason a single global light/moderate/heavy does
    not exist: the degree is 50 units and Summer is fixed, so the adjustable
    quarters must always total exactly the same number. Making one quarter
    lighter does not make the degree lighter -- it moves work into another
    quarter. Saying so is the honest version of this question; silently
    producing a 46-unit plan is not, and `quarter_units_problems` would reject
    it anyway.

    `pinned` is the set of quarters the student has actually answered for, and
    it is passed in rather than read off `chosen`: `chosen` is the whole
    distribution this function itself produced last time, so every quarter in
    it looks decided. Keeping the two apart is what makes "you had asked for a
    light Fall" a true statement rather than an artefact of how the numbers are
    stored. See `quarter_loads` on the intake.

    Units move two at a time, because `_resized` cuts a quarter's budget into
    4-unit slots plus a 2-unit remainder -- an odd budget leaves a 1- or 3-unit
    hole that no course in the catalog can fill.
    """
    quarters = adjustable_quarters(track)
    if not quarters or not any(q["key"] == key for q in quarters):
        return None, []
    current = quarter_units_of({"track": track, "quarter_units": chosen})
    bounds = {q["key"]: q for q in quarters}
    limits = bounds[key]
    start = dict(current)
    start[key] = max(limits["min"], min(limits["max"], int(target)))
    required = TOTAL_UNITS - fixed_quarter_units(track)
    decided = {q for q in (pinned if pinned is not None else (chosen or {}))
               if q != key}
    order = [q["key"] for q in quarters if q["key"] != key]

    def absorb(absorbers):
        """Round-robin in published order, so one request gives one answer."""
        wanted = dict(start)
        while sum(wanted.values()) != required:
            gap = required - sum(wanted.values())
            step = 2 if gap > 0 else -2
            moved = False
            for other in absorbers:
                room = bounds[other]
                candidate = wanted[other] + step
                if room["min"] <= candidate <= room["max"]:
                    wanted[other] = candidate
                    moved = True
                    if sum(wanted.values()) == required:
                        break
            if not moved:
                return None
        return wanted

    undecided = [q for q in order if q not in decided]
    wanted = absorb(undecided) if undecided else None
    if wanted is not None:
        return wanted, []
    loose = absorb(order)
    if loose is None:
        return None, []
    return None, [q for q in order if q in decided and loose[q] != current[q]]


def seeded_units(track, load):
    """A starting per-quarter distribution for an overall light/heavy answer.

    ({} for "moderate", which IS the published plan.)

    The overall preference has to act on UNITS, because on this catalog it
    cannot act on anything else. `rank_electives` does penalise a course
    heavier than the student asked for, but the elective catalog holds 14
    moderate courses, 9 heavy ones and exactly ONE light one -- so a light
    preference has nothing lighter to pick and the plan comes back identical.
    Measured: for the data-scientist profile, light and heavy produce the same
    twelve courses. Leaving the question there would be asking something and
    then doing nothing with the answer.

    What it can do is decide WHERE THE WEIGHT SITS. The degree is fixed at 50
    units, so "light" cannot mean fewer -- it means the early quarters run at
    their floor and the units land later. This lightens each quarter in
    published order for as long as the total allows, and stops at the first one
    that would undo an earlier choice. For the 11-month track: light gives
    12 / 12 / 18, heavy gives 18 / 12 / 12, moderate gives the published
    14 / 14 / 14.

    It is a STARTING point. The walk-through then asks about each quarter
    individually, with the student looking at the courses the answer changes.
    """
    if not load_is_a_choice(track):
        return {}
    if load == "moderate":
        # The published plan, with the finishing quarter held at 4 elective
        # units. On the 17-month track the published plan already ends that
        # way and this stays {}; on the 11-month track Spring publishes at
        # 14, so moderate is a real distribution there.
        published = {q["key"]: q["default"] for q in adjustable_quarters(track)}
        held = _with_tail_rule(track, load, published)
        return {} if held == published else held
    if load not in ("light", "heavy"):
        return {}
    chosen, pinned = {}, set()
    for quarter in adjustable_quarters(track):
        target = units_for_load(track, quarter["key"], load)
        if target is None or target == quarter_units_of(
                {"track": track, "quarter_units": chosen}).get(quarter["key"]):
            # Already there (its published load is the bound), so pin it
            # without a rebalance and move on.
            chosen[quarter["key"]] = target
            pinned.add(quarter["key"])
            continue
        wanted, disturbed = rebalanced_units(track, chosen, quarter["key"],
                                             target, pinned)
        if wanted is None or disturbed:
            break
        chosen = dict(wanted)
        pinned.add(quarter["key"])
    if not chosen or quarter_units_problems(track, chosen):
        return {}
    return _with_tail_rule(track, load, chosen)


# The ELECTIVE units the finishing quarter carries, by overall load: light and
# moderate keep it at 4, heavy at 2. Elective units, because the finishing
# quarter's core is not negotiable -- on the 17-month track the second-year
# Fall has no core, so 4 elective units IS a 4-unit quarter; on the 11-month
# track Spring carries MGTA 454 (4 units), so it ends at 8 (or 6 on heavy).
#
# Before this, light put every earlier quarter at its 12-unit floor and let
# the finishing quarter absorb what was left -- 6 units on the 17-month, 18
# on the 11-month -- so the lightest spread ended on the heaviest term. The
# units it gives up land in the latest full quarters first, so light's own
# promise ("the later quarters carry more") still holds.
#
# The consequence on both tracks is that light and moderate coincide -- once
# the tail is fixed and the total is 50, there is only one way to lay the
# rest within the 12-18 range that puts the weight late. What separates them
# is the walk-through, where any single quarter can be moved.
TAIL_UNITS = {"light": 4, "moderate": 4, "heavy": 2}


def tail_target(track, load):
    """The finishing quarter's TOTAL under `load`: its core plus `TAIL_UNITS`.
    None where the track has no finishing quarter or the load has no rule."""
    units = TAIL_UNITS.get(load)
    tail = next((q for q in adjustable_quarters(track)
                 if q["key"] == tail_quarter_key(track)), None)
    if units is None or tail is None or not load_is_a_choice(track):
        return None
    return tail["core_units"] + units


def _with_tail_rule(track, load, chosen):
    """Hold the finishing quarter at `tail_target`, moving the difference into
    the latest full quarters two units at a time. Returns `chosen` untouched
    when the track has no tail, the load has no rule, or no legal spread
    holds it."""
    target = tail_target(track, load)
    quarters = adjustable_quarters(track)
    key = tail_quarter_key(track)
    tail = next((q for q in quarters if q["key"] == key), None)
    if target is None or tail is None or key not in chosen:
        return chosen
    if not (tail["min"] <= target <= tail["max"]):
        return chosen
    diff = chosen[key] - target
    if diff == 0:
        return chosen
    trial = {**chosen, key: target}
    step = 2 if diff > 0 else -2
    # Latest full quarter first: the units land late on a light spread and
    # come off late on a heavy one. Each quarter takes what it can within
    # its own bounds before the next one is touched.
    for quarter in reversed([q for q in quarters
                             if q["key"] != key and q["key"] in chosen]):
        while diff != 0:
            moved = trial[quarter["key"]] + step
            if not (quarter["min"] <= moved <= quarter["max"]):
                break
            trial[quarter["key"]] = moved
            diff -= step
        if diff == 0:
            break
    if diff != 0 or quarter_units_problems(track, trial):
        return chosen
    return trial


def load_of_quarter(track, key, chosen, pinned=None):
    """Which of the three words describes this quarter's current load.

    `pinned` is passed on for the same reason it exists on
    `rebalanced_units`: without it, `chosen` -- the whole distribution -- is
    read as a set of decisions, so every quarter looks locked, every load
    looks unreachable, and a quarter sitting at exactly 12 units reported no
    load at all.
    """
    units = quarter_units_of({"track": track, "quarter_units": chosen}).get(key)
    for load in LOAD_BOUNDS:
        if units_for_load(track, key, load, chosen, pinned) == units:
            return load
    return None


def quarter_units_form_for(track):
    """The slider form: one row per adjustable quarter, seeded with the plan.
    None where the load is not a choice (`load_is_a_choice`)."""
    quarters = adjustable_quarters(track)
    if not quarters or not load_is_a_choice(track):
        return None
    fixed = fixed_quarter_units(track)
    adjustable_keys = {q["key"] for q in quarters}
    # LOCKED quarters are listed too, greyed rather than hidden. Summer is
    # entirely required courses, so there is nothing to move -- but leaving it
    # out made the form add up to 42 with no visible reason, and a student
    # comparing that against the 50-unit degree has to be told where the other
    # 8 went rather than work it out from a parenthetical.
    locked = [{"key": q["key"], "label": q["label"], "units": q["units"],
               "locked": True}
              for q in (TRACK_SKELETONS.get(track) or [])
              if q["key"] not in adjustable_keys]
    return {
        "kind": "units",
        "rows": [{"key": q["key"], "label": q["label"], "min": q["min"],
                  "max": q["max"], "step": 2, "default": q["default"],
                  "locked": False}
                 for q in quarters],
        "lockedRows": locked,
        # Sliders move independently, so the running total is what tells a
        # student their plan still adds up. The submit is theirs to fix.
        "total": TOTAL_UNITS - fixed,
        "lockedTotal": fixed,
        "grandTotal": TOTAL_UNITS,
        "totalLabel": f"{TOTAL_UNITS - fixed} units to place",
        "submitLabel": "Use this load",
    }


WORKLOAD_CHOICES = [
    {"value": "light", "label": "Lighter — I have commitments outside class"},
    {"value": "moderate", "label": "Moderate — a normal full load"},
    {"value": "heavy", "label": "Heavy — I want to be pushed"},
]

ROUTE_CHOICES = [
    {"value": "fixed",
     "label": "Use the recommended bundle",
     "description": "The set of electives this career path is built from — the "
                    "same plan every student aiming at it gets"},
    {"value": "custom",
     "label": "Build it around me",
     "description": "Fill the electives against your own skills, workload and "
                    "interests, then swap anything you like"},
]

TRACK_CHOICES = [
    {"value": "11 month", "label": "11 month — Summer through Spring"},
    {"value": "17 month", "label": "17 month — finishes the following Fall"},
]


@lru_cache(maxsize=4)
def _catalog_by_id_for(_version):
    return {course["id"]: course for course in load_catalog()}



def _catalog_by_id():
    """Every course by id. Re-read when the catalog file changes."""
    return _catalog_by_id_for(catalog_version())

@lru_cache(maxsize=4)
def _ambiguous_codes_for(_version):
    """Codes shared by more than one distinct course.

    The catalog carries four Special Topics courses all coded MGTA 495, with
    different ids, titles, seasons and topics. Their ids keep the plan correct —
    measured over 80 generated plans, zero contained a duplicated course ID —
    but 25 of those plans PRINTED "MGTA 495" on two rows, which reads as the
    same course listed twice and makes "change MGTA 495" ambiguous.
    """
    counts = {}
    for course in load_catalog():
        counts[course["code"]] = counts.get(course["code"], 0) + 1
    return frozenset(code for code, n in counts.items() if n > 1)



def _ambiguous_codes():
    return _ambiguous_codes_for(catalog_version())

def display_code(course):
    """The code as a student should see it, disambiguated when it has to be."""
    code = course["code"]
    if code not in _ambiguous_codes():
        return code
    suffix = course["id"][len(code):].lstrip("- ").strip()
    return f"{code} ({suffix})" if suffix else code


def intake_questions():
    """The interview, as data the UI renders and the tests assert against.

    A fixed script rather than an LLM conversation: every answer feeds a
    specific scoring input, so a question with nowhere to land is a question
    not worth asking a student.
    """
    careers = load_careers()
    tags = sorted({tag for course in load_catalog()
                   for tag in (course.get("career_tags") or [])})
    return [
        {"key": "track", "kind": "single", "required": True,
         "prompt": "Which track are you on?",
         "help": "It decides how many quarters the plan has and which quarter each course lands in.",
         "options": TRACK_CHOICES},
        {"key": "goals", "kind": "multi", "required": True, "max": 3,
         "prompt": "What role are you aiming for after the program?",
         "help": "Pick the one closest to what you want.",
         "options": [{"value": role_id, "label": role["label"],
                      "description": role.get("description", "")}
                     for role_id, role in sorted(careers.items(),
                                                 key=lambda kv: kv[1]["label"])]},
        # Asked in the interview (it is in INTAKE_STEPS) but NOT required for a
        # valid intake. `POST /api/thrive/plan` takes an answers object
        # directly, and every plan built before this choice existed was a custom
        # one -- so an intake without a route is complete and means custom,
        # rather than a 400 for a field that did not exist yesterday.
        {"key": "route", "kind": "single", "required": False,
         "prompt": "How should I fill your electives?",
         "help": "You can change your mind later — either way you can swap any "
                 "course once you see the plan.",
         "options": ROUTE_CHOICES},
        # NO SELF-RATING. This used to ask five "rate yourself 1-5" questions
        # here. They are gone, deliberately: a student cannot rate their own
        # machine learning before they have taken any, the answers gated the
        # plan behind five sliders, and a low self-rating steered people AWAY
        # from the courses that would fix the gap -- the opposite of advising.
        #
        # The scoring machinery underneath is untouched and simply runs at the
        # neutral rating: `profile_from_intake` defaults every area to
        # DEFAULT_SKILL_RATING, and `prerequisite_cautions` returns nothing
        # when no rating exists. `read_skills` still reads a rating a student
        # VOLUNTEERS ("strong in python, never done ML"), because being told is
        # different from being asked.
        {"key": "quarter_units", "kind": "units", "required": False,
         "prompt": "How many units do you want in each quarter?",
         "help": "Starts on the published plan. Move them around if you would "
                 "rather front-load or finish light — the total stays at 50, "
                 "and full-time standing needs 12 a quarter.",
         "options": []},
        {"key": "workload", "kind": "single", "required": False,
         "prompt": "How heavy a course load do you want?",
         "options": WORKLOAD_CHOICES},
        {"key": "interests", "kind": "multi", "required": False,
         "prompt": "Anything you particularly want to work on?",
         "help": "Optional. These are matched against course topics and tools.",
         "options": [{"value": tag, "label": tag.replace("-", " ")} for tag in tags]},
    ]


def _is_allowed(value, allowed):
    """Membership test that survives whatever JSON a client actually sends.

    `allowed` is a set, so an unhashable value raised TypeError from inside
    `in` rather than failing validation. `POST /api/thrive/plan` takes an
    `answers` object straight from the caller, so `{"answers": {"track":
    {"a": 1}}}` -- a dict where a string belongs -- crashed the request and any
    logged-in student could turn a malformed body into a 500.

    An unhashable value is simply not one of the allowed strings, so it is
    invalid rather than exceptional, and the caller gets the 400 that says so.
    """
    try:
        return value in allowed
    except TypeError:
        return False


def validate_intake(answers):
    """Return a list of human-readable problems with a submitted intake."""
    problems = []
    for question in intake_questions():
        value = answers.get(question["key"])
        if question["key"].startswith("skill_"):
            # Accepts whatever `normalise_intake` accepts — a 1-5 number from
            # the rating form, or the words people type. Validating against the
            # option list alone would reject "comfortable", which is still a
            # perfectly good answer to a question asked in prose.
            if question["required"] and skill_score(value) is None:
                problems.append(f"{question['key']}: rate 1-5")
        elif question["kind"] == "single":
            allowed = {opt["value"] for opt in question["options"]}
            if question["required"] and not _is_allowed(value, allowed):
                problems.append(f"{question['key']}: pick one of {sorted(allowed)}")
        else:
            allowed = {opt["value"] for opt in question["options"]}
            chosen = value or []
            if not isinstance(chosen, list):
                problems.append(f"{question['key']}: expected a list")
                continue
            if question["required"] and not chosen:
                problems.append(f"{question['key']}: choose at least one")
            unknown = [c for c in chosen if not _is_allowed(c, allowed)]
            if unknown:
                problems.append(f"{question['key']}: unknown {unknown}")
            if question.get("max") and len(chosen) > question["max"]:
                problems.append(f"{question['key']}: at most {question['max']}")
    return problems


def workload_from_units(answers):
    """The old light/moderate/heavy, read off the load the student chose.

    `rank_electives` still wants a tolerance word, and asking for one separately
    would be asking the same question twice: someone who puts 18 units in a
    quarter has told us they will take a heavy term. An explicit `workload` on
    the intake still wins, so a plan saved before this question changed keeps
    the answer it was given.
    """
    stated = (answers or {}).get("workload")
    if stated:
        return stated
    chosen = (answers or {}).get("quarter_units")
    if not chosen:
        return "moderate"
    heaviest = max(quarter_units_of(answers).values(), default=14)
    if heaviest >= 16:
        return "heavy"
    return "light" if heaviest <= 12 else "moderate"


def profile_from_intake(answers):
    """Map interview answers onto the profile `rank_electives` already reads."""
    technical = [skill_score(answers.get(f"skill_{key}")) or DEFAULT_SKILL_RATING
                 for key in TECHNICAL_AREAS]
    comfort = round(sum(technical) / len(technical)) if technical else 3
    return {
        "career_roles": list(answers.get("goals") or []),
        "career_tags": list(answers.get("interests") or []),
        "technical_comfort": comfort,
        "workload_preference": workload_from_units(answers),
        "interests": [tag.replace("-", " ") for tag in (answers.get("interests") or [])],
        # The student's own words for a career the catalog curates no id for.
        # Carried so the plan can say what it is FOR: a heading that names the
        # career is the difference between a plan and a list of courses, and an
        # uncurated goal deserves it as much as a curated one.
        "stated_goal": str(answers.get("unmatched_goal") or ""),
    }


# ---------------------------------------------------------------------------
# Prerequisite cautions
# ---------------------------------------------------------------------------

def prerequisite_cautions(course, answers):
    """Areas a course's prerequisites demand that the student rated low.

    Advisory, never a filter. The prerequisite text is prose written for humans
    ("Solid Python programming experience required"), so matching words in it is
    a hint and not an entitlement to remove a course from a student's plan —
    that call belongs to the student and their advisor, and the wording is
    surfaced next to the warning so they can make it.
    """
    text = " ".join(filter(None, [course.get("prerequisites") or ""])).lower()
    if not text:
        return []
    cautions = []
    for area in SKILL_AREAS:
        if not area["prereq_words"]:
            continue
        if not any(word in text for word in area["prereq_words"]):
            continue
        level = answers.get(f"skill_{area['key']}")
        score = skill_score(level)
        if score is not None and score <= 2:
            cautions.append(
                f"expects {area['label'].lower()} and you rated yourself "
                f"{skill_label(level)} out of 5")
    return cautions


# ---------------------------------------------------------------------------
# Similarity, for "give me something else that teaches the same thing"
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")
# Words that appear in nearly every skill phrase and so carry no signal about
# whether two courses teach the same thing.
_SKILL_STOPWORDS = frozenset("""
a an the and or of to in on for with from as is are be using use used data
your you basic advanced introduction intro topics business analytics course
""".split())


def _phrase_words(phrases):
    words = set()
    for phrase in phrases or []:
        words |= {w for w in _WORD.findall(phrase.lower())
                  if len(w) > 2 and w not in _SKILL_STOPWORDS}
    return words


def _learning_words(course):
    """The vocabulary of what a course teaches: skills first, then topics/tools."""
    return (_phrase_words(course.get("skills"))
            | _phrase_words(course.get("topics"))
            | _phrase_words(course.get("tools")))


def similarity(left, right):
    """How much of the same learning two courses deliver, 0..1.

    Jaccard over the words of skills, topics and tools, plus a bump for shared
    career tags. Tags alone are too coarse — every marketing course shares the
    "marketing" tag while teaching completely different things — and words
    alone miss that two differently-worded courses serve the same career, so
    the two signals are combined rather than chosen between.
    """
    lw, rw = _learning_words(left), _learning_words(right)
    overlap = len(lw & rw) / len(lw | rw) if (lw | rw) else 0.0
    lt, rt = set(left.get("career_tags") or []), set(right.get("career_tags") or [])
    tag_overlap = len(lt & rt) / len(lt | rt) if (lt | rt) else 0.0
    return round(0.7 * overlap + 0.3 * tag_overlap, 4)


def shared_skills(left, right, limit=4):
    """Skill phrases from `right` that overlap what `left` teaches."""
    lw = _learning_words(left)
    scored = []
    for phrase in right.get("skills") or []:
        words = _phrase_words([phrase])
        if words & lw:
            scored.append((len(words & lw), phrase))
    scored.sort(key=lambda pair: -pair[0])
    return [phrase for _, phrase in scored[:limit]]


def distinct_skills(left, right, limit=3):
    """Skill phrases `left` teaches that `right` does not — what a swap loses."""
    rw = _learning_words(right)
    return [phrase for phrase in (left.get("skills") or [])
            if not (_phrase_words([phrase]) & rw)][:limit]


# ---------------------------------------------------------------------------
# Building the plan
# ---------------------------------------------------------------------------

def _offered_in(course, season):
    return any((offering.get("season") or "").upper() == season
               for offering in course.get("offerings") or [])


def equivalents_of(course_id, catalog=None):
    """Every catalog id that is the SAME course as `course_id`, itself included.

    Rady numbers a second section of an MBA course with a trailing R -- MGT 477
    and MGT 477R are both Consumer Behavior, and the catalog says "students may
    not earn credit for both". `also_known_as` carries the same fact under
    other numberings (MGT 203). A plan holding both twins would print Consumer
    Behavior twice and count 8 units for 4, which is what happened the moment
    both sections were scheduled for the same Spring.
    """
    catalog = catalog or _catalog_by_id()
    out = {course_id}
    course = catalog.get(course_id) or {}
    for other in course.get("also_known_as") or []:
        if other in catalog:
            out.add(other)
    twin = course_id[:-1] if course_id.endswith("R") else course_id + "R"
    if twin in catalog:
        out.add(twin)
    for other_id, other in catalog.items():
        if course_id in (other.get("also_known_as") or []):
            out.add(other_id)
    return out


def _clashes(course_id, held, catalog=None):
    """Is `course_id`, or a course equivalent to it, already in `held`?"""
    return bool(equivalents_of(course_id, catalog) & set(held))


def _within_workload(course, preference):
    return WORKLOAD_LEVEL.get(course.get("workload", "moderate"), 2) <= preference


def _within_comfort(course, comfort):
    return (course.get("technical_level") or 3) <= comfort


# A course this far above the student's stated comfort is a different kind of
# problem from one a single level above it. One level is a stretch; two is
# putting someone who reported no programming into PySpark.
SEVERE_STRETCH = 2


def _pick(candidates, profile):
    """Choose an elective: the ranked best, minus the severe mismatches.

    `rank_electives` already penalises a course that is heavier or more
    technical than the student asked for, and those penalties are calibrated.
    Measured across 100 synthetic students, overriding them with hard
    preference tiers cut electives-above-comfort from 214 to 102 but dropped
    goal alignment from 72% to 56% — it started filling career-critical slots
    with comfortable but irrelevant courses, which is a worse plan, not a
    safer one.

    So only a SEVERE technical mismatch (two levels or more above the declared
    comfort) is skipped while a ranked alternative exists. Workload is left
    entirely to the scorer: the catalog contains exactly one "light" elective,
    so a light-workload preference is not satisfiable and pretending otherwise
    would just push off-goal courses into the plan. Where a preference cannot
    be met, `_stretch_notes` says so on the row.
    """
    comfort = profile["technical_comfort"]
    gentle = [r for r in candidates
              if (r["course"].get("technical_level") or 3) - comfort < SEVERE_STRETCH]
    for tier in (gentle, candidates):
        if tier:
            return tier[0]["course"]
    return None


def _stretch_notes(course, profile):
    """Where this course exceeds what the student asked for."""
    notes = []
    preference = WORKLOAD_LEVEL.get(profile.get("workload_preference"), 2)
    if not _within_workload(course, preference):
        notes.append(
            f"{course.get('workload')} workload, above the "
            f"{profile.get('workload_preference')} load you asked for — nothing "
            f"lighter is offered for this slot")
    if not _within_comfort(course, profile["technical_comfort"]):
        notes.append(
            f"more technical (level {course.get('technical_level')}) than the "
            f"level {profile['technical_comfort']} you reported")
    return notes


def _entry(course, kind, *, swappable, reasons=None, cautions=None, note="",
           stretch=None, on_goal=None, requirement=None):
    """One row of a quarter, labelled core or elective as the student sees it.

    `requirement` overrides that label without touching `kind`, which is what
    the unit arithmetic counts. Summer's MGTA 403 and MGTA 464 are the reason:
    they carry ELECTIVE credit in the published plan of study, and they are
    also mandatory — every student takes them and there is nothing to swap
    them for. Labelling those rows "Elective" told a student they had a choice
    they do not have; labelling them "Core" would have contradicted the 22/28
    split printed directly underneath. They are labelled "Required" instead,
    and the Summer table says plainly which units they count toward.
    """
    return {
        "courseId": course["id"],
        "code": display_code(course),
        # The bare catalog code, for anything matching on identity rather than
        # display — `code` may carry a disambiguating suffix.
        "baseCode": course["code"],
        "title": course["title"],
        "units": course["units"],
        "kind": kind,                       # "core" | "elective"
        "requirement": requirement or ("Core" if kind == "core" else "Elective"),
        "swappable": swappable,
        "technicalLevel": course.get("technical_level"),
        "workload": course.get("workload"),
        "skills": (course.get("skills") or [])[:4],
        "reasons": reasons or [],
        "cautions": cautions or [],
        "note": note,
        # Where the course exceeds a limit the student stated. Separate from
        # `cautions`, which is about prerequisites: one is "this is harder than
        # you wanted", the other is "you may not be ready for this".
        "stretch": stretch or [],
        # 28 elective units have to be filled and few goals have 28 units of
        # on-target courses, so some electives are breadth by necessity. Saying
        # which is which stops the plan implying every choice serves the goal.
        "onGoal": on_goal,
    }


def _resized(skeleton, answers):
    """The skeleton with each quarter's elective slots matching the chosen load.

    The student sets units per quarter (see `adjustable_quarters`), and that has
    to change the PLAN, not just a number on a form. A quarter asked to carry 18
    units instead of 14 needs somewhere to put the extra four, so the elective
    slots are re-cut to the new budget: 4-unit slots first, with a 2-unit slot
    for a remainder that is not a multiple of four.

    Untouched when the student has chosen nothing and stated no load, so a
    plan built without this question is byte-identical to what it was before
    the question existed. (`quarter_units_of` supplies the load's seed when
    only the one-word answer is known.)
    """
    catalog = _catalog_by_id()
    wanted = quarter_units_of(answers)
    resized = []
    for quarter in skeleton:
        target = wanted.get(quarter["key"])
        if target is None or target == quarter["units"]:
            resized.append(quarter)
            continue
        keep = [dict(slot) for slot in quarter["slots"]
                if slot["kind"] in ("core", "fixed")]
        budget = target - sum(catalog[slot["course_id"]]["units"] for slot in keep)
        for _ in range(max(0, budget) // 4):
            keep.append({"kind": "elective", "units": 4})
        if max(0, budget) % 4:
            keep.append({"kind": "elective", "units": max(0, budget) % 4})
        resized.append({**quarter, "units": target, "slots": keep})
    return resized


# ---------------------------------------------------------------------------
# Starting somewhere other than the beginning
# ---------------------------------------------------------------------------

def quarter_keys(track):
    """The published quarter keys for a track, in order."""
    return [quarter["key"] for quarter in TRACK_SKELETONS.get(track) or []]


def quarter_label(track, key):
    for quarter in TRACK_SKELETONS.get(track) or []:
        if quarter["key"] == key:
            return quarter["label"]
    return key


# What a student calls the quarter they are in, mapped to the skeleton's keys.
# The second year's Fall needs its own vocabulary because "fall" alone is
# ambiguous on the 17-month track and a plan built from the wrong Fall is off
# by a year -- which is exactly the class of error the arithmetic here exists
# to make impossible.
QUARTER_WORDS = {
    "summer": "summer", "summer iii": "summer", "su": "summer",
    "fall": "fall", "autumn": "fall", "fa": "fall", "fall one": "fall",
    "first fall": "fall",
    "winter": "winter", "wi": "winter",
    "spring": "spring", "sp": "spring",
    "fall two": "fall-two", "second fall": "fall-two",
    "fall (second year)": "fall-two", "fall second year": "fall-two",
    "fall-two": "fall-two", "second-year fall": "fall-two",
}


def resolve_quarter(track, said):
    """The skeleton key for a quarter a student named, or None.

    Returns None rather than guessing. A plan built from the wrong quarter is
    wrong about every date and every unit count in it, and "I think you meant
    Winter" is a question worth asking rather than an assumption worth making.
    """
    text = " ".join((said or "").strip().lower().split())
    if not text:
        return None
    keys = quarter_keys(track)
    if text in keys:
        return text
    resolved = QUARTER_WORDS.get(text)
    if resolved is None:
        # A phrase rather than a bare word: "I'm already in winter". Longest
        # match first so "fall (second year)" beats "fall".
        for phrase in sorted(QUARTER_WORDS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(phrase)}\b", text):
                resolved = QUARTER_WORDS[phrase]
                break
    return resolved if resolved in keys else None


def _from_quarter(skeleton, start_from):
    """(the quarters still to plan, the quarters already behind the student).

    The whole of the mid-program change, and deliberately a slice rather than a
    filter: quarters run in a published order, and "everything from Winter" is
    that order's tail. A student who has switched tracks or fallen behind is
    planning the tail; the head is history, and history is not schedulable.
    """
    if not start_from:
        return skeleton, []
    keys = [quarter["key"] for quarter in skeleton]
    if start_from not in keys:
        raise ValueError(
            f"unknown quarter {start_from!r}; this track runs {', '.join(keys)}")
    index = keys.index(start_from)
    return skeleton[index:], skeleton[:index]


def _unscheduled_requirements(dropped, taken_ids):
    """Required courses that sit behind the start and are not on the transcript.

    THE reason `start_from` cannot be a simple slice and stop there. Cutting
    Summer and Fall off a plan removes MGTA 451, 403, 464, 452 and 453 from it
    -- and a plan that silently omits five required courses reads as a plan
    that does not need them. Every one that is not already taken comes back
    here, so the caller states it as outstanding rather than losing it.
    """
    catalog = _catalog_by_id()
    outstanding = []
    for quarter in dropped:
        for slot in quarter["slots"]:
            if slot["kind"] not in ("core", "fixed"):
                continue
            course = catalog[slot["course_id"]]
            if course["id"] in taken_ids:
                continue
            outstanding.append({
                "courseId": course["id"], "code": course["code"],
                "title": course["title"], "units": course["units"],
                "kind": slot["kind"],
                "quarter": quarter["key"], "quarterLabel": quarter["label"],
            })
    return outstanding


def build_plan(answers, taken_ids=frozenset(), selections=None, skeleton=None,
               start_from=None):
    """Build the plan of study for one student, from `start_from` onward.

    `selections` is {quarter_key: {slot_index: course_id}} and wins over the
    automatic pick, which is what makes a swap a persisted edit rather than a
    re-roll: the plan is rebuilt from the intake every time and the student's
    overrides are re-applied on top, so nothing else in the plan shifts
    underneath a single change.

    Every quarter lists its CORE courses as well as its electives, each row
    labelled, because a plan that shows only the electives does not tell a
    student what their quarter looks like.

    `start_from` is a published quarter key ("winter", "fall-two", ...). It
    plans only the quarters from there on, which is what a student who switched
    tracks mid-programme, or who is asking in week three of Winter, actually
    needs -- the plan used to start at Summer whatever the date, so the only
    honest thing to do with it was ignore the first half.

    Two things go with the cut, and both are arithmetic the caller must not
    have to redo:

    * `unscheduled` lists required courses that sit BEHIND the start and are
      not on the transcript. They do not vanish because their quarter did.
    * `totals` gains `completed`, `scheduled` and `outstanding`, so "how many
      units do I still need" is a number this function produced rather than one
      a language model added up.
    """
    catalog = _catalog_by_id()
    profile = profile_from_intake(answers)
    track = answers.get("track") or "11 month"
    # `skeleton` overrides the published shape, and only a FIXED route passes
    # one. A bundle is a known set of courses, so it can declare slots sized to
    # those courses instead of squeezing them into the generic `[4, 2]` /
    # `[4, 4]` shapes -- which fit 0 of 14 bundles on the 17-month track,
    # because every bundle needs two 2-unit courses and that track has no
    # 2-unit slot. See `services/bundles.py`. Core and fixed slots are the
    # published sequence either way; only the elective shape differs.
    skeleton = skeleton or TRACK_SKELETONS.get(track)
    if skeleton is None:
        raise ValueError(f"unknown track {track!r}")
    # The cut comes BEFORE the resize: resizing a quarter the student is
    # already past is work with no output, and `_resized` reads the load answer
    # per quarter, which for a dropped quarter is a load nobody can now choose.
    skeleton, dropped = _from_quarter(skeleton, start_from)
    unscheduled = _unscheduled_requirements(dropped, taken_ids)
    skeleton = _resized(skeleton, answers)

    ranked = rank_electives(load_catalog(), profile, load_careers())
    fit = {row["course"]["id"]: row for row in ranked}
    selections = selections or {}
    careers = load_careers()
    goal_tags = {tag for role in profile["career_roles"]
                 for tag in (careers.get(role, {}).get("career_tags") or [])}

    # Courses the student SAID they have finished -- not the wider `taken_ids`,
    # which also holds current enrolments. The first cut dropped every taken
    # core row, and for a student enrolled in MGTA 451 this quarter that
    # emptied Summer: "Summer III -- 4 units" listing nothing. A course being
    # taken now belongs in the plan; a course already finished belongs in it
    # too, marked done and counted in `completed`, rather than silently gone.
    stated_done = stated_completed_ids(answers)
    used, quarters, unfilled = set(), [], []
    for quarter in skeleton:
        rows = []
        for index, slot in enumerate(quarter["slots"]):
            if slot["kind"] in ("core", "fixed"):
                course = catalog[slot["course_id"]]
                used.add(course["id"])
                is_core = slot["kind"] == "core"
                if course["id"] in stated_done:
                    row = _entry(course, "core" if is_core else "elective",
                                 swappable=False, requirement="Done",
                                 reasons=["you said you've already done this"],
                                 note="already done — not counted again")
                    row["completed"] = True
                    rows.append(row)
                    continue
                rows.append(_entry(
                    course, "core" if is_core else "elective", swappable=False,
                    reasons=["required core course for the MSBA"] if is_core else
                            ["required — every student takes this, there is "
                             "nothing to choose here"],
                    cautions=prerequisite_cautions(course, answers),
                    note="" if is_core else ("required by the published plan of "
                                             "study; counts toward elective units"),
                    # Not "Elective": nothing about it is chooseable. See `_entry`.
                    requirement=None if is_core else "Required",
                ))
                continue

            chosen_id = (selections.get(quarter["key"]) or {}).get(str(index))
            candidates = [
                row for row in ranked
                if row["course"]["units"] == slot["units"]
                and _offered_in(row["course"], quarter["season"])
                and not _clashes(row["course"]["id"], used, catalog)
                and not _clashes(row["course"]["id"], taken_ids, catalog)
            ]
            course = None
            if (chosen_id and chosen_id in catalog
                    and not _clashes(chosen_id, used, catalog)):
                picked = catalog[chosen_id]
                # An override still has to be a legal course for this hole.
                if (picked["units"] == slot["units"]
                        and _offered_in(picked, quarter["season"])
                        and not _clashes(chosen_id, taken_ids, catalog)):
                    course = picked
            if course is None:
                course = _pick(candidates, profile)

            if course is None:
                unfilled.append({
                    "quarter": quarter["key"], "slot": index, "units": slot["units"],
                    "why": f"no {slot['units']}-unit elective is offered in "
                           f"{quarter['label']} that you have not already used",
                })
                rows.append({
                    "courseId": None, "code": None, "title": None,
                    "units": slot["units"], "kind": "elective",
                    "requirement": "Elective", "swappable": True,
                    "skills": [], "reasons": [], "cautions": [],
                    "note": "nothing available for this slot",
                })
                continue

            used.add(course["id"])
            rows.append(_entry(
                course, "elective", swappable=True,
                reasons=(fit.get(course["id"]) or {}).get("reasons", []),
                cautions=prerequisite_cautions(course, answers),
                note="your choice" if chosen_id == course["id"] else "",
                stretch=_stretch_notes(course, profile),
                on_goal=bool(set(course.get("career_tags") or []) & goal_tags),
            ))

        quarters.append({
            "key": quarter["key"], "label": quarter["label"],
            "season": quarter["season"],
            # SCHEDULED units, not slot capacity. An unfilled slot keeps its
            # `units` on the row so the table can show how big the hole is, but
            # counting it here made a plan claim units it does not contain: a
            # student who had already taken both of Fall's 2-unit electives got
            # a 48-unit plan whose header said 50. See `unfilled` below.
            "unitsPlanned": sum(r["units"] for r in rows
                                if r["courseId"] and not r.get("completed")),
            "unitsExpected": quarter["units"],
            "courses": rows,
        })

    core_units = sum(r["units"] for q in quarters for r in q["courses"]
                     if r["kind"] == "core" and r["courseId"]
                     and not r.get("completed"))
    elective_units = sum(r["units"] for q in quarters for r in q["courses"]
                         if r["kind"] == "elective" and r["courseId"]
                         and not r.get("completed"))
    # Units already on the transcript. Counted from the CATALOG rather than
    # from anything the caller says, so a course the student names but that
    # this programme does not carry cannot inflate the figure -- and every
    # number below is one this function computed.
    completed_units = sum(course["units"] for course in load_catalog()
                          if course["id"] in taken_ids)
    scheduled_units = core_units + elective_units
    return {
        "track": track,
        "quarters": quarters,
        # Null when the plan runs from the beginning, which is every plan built
        # before this parameter existed.
        "startFrom": start_from,
        "quartersRemaining": len(quarters),
        # Required courses whose quarter is behind the student and which are
        # not on their transcript. Empty for a plan built from the start.
        "unscheduled": unscheduled,
        "totals": {
            "core": core_units,
            "elective": elective_units,
            "total": scheduled_units,
            "coreRequired": CORE_UNITS,
            "electiveRequired": ELECTIVE_UNITS,
            "totalRequired": TOTAL_UNITS,
            # The mid-programme ledger. `outstanding` is what is left to earn
            # after this plan runs: zero means the plan finishes the degree,
            # and anything above zero is a shortfall the student has to be told
            # about rather than left to discover at graduation.
            "completed": completed_units,
            "scheduled": scheduled_units,
            "outstanding": max(0, TOTAL_UNITS - completed_units - scheduled_units),
        },
        "profile": profile,
        "unfilled": unfilled,
        "links": ACTION_LINKS,
        # Set when most of this plan's electives are breadth rather than aimed
        # at the goal. The plan is complete either way; saying so is what stops
        # a broad plan reading as a targeted one.
        "targeting": targeting_note(quarters, profile["career_roles"]),
        "disclaimer": ("This is a sample plan built from Rady's published plan of "
                       "study, and a recommendation rather than a decision — "
                       "confirm your schedule with MSBA advising before you "
                       "book, as course offerings and quarters can change."),
    }


# ---------------------------------------------------------------------------
# Swapping a course for one that teaches the same thing
# ---------------------------------------------------------------------------

def _locate(plan, quarter_key, slot):
    for quarter in plan["quarters"]:
        if quarter["key"] != quarter_key:
            continue
        if 0 <= slot < len(quarter["courses"]):
            return quarter, quarter["courses"][slot]
    return None, None


def alternatives_for(plan, answers, quarter_key, slot, taken_ids=frozenset(), limit=4):
    """Other courses that would fill the same hole and teach much the same thing.

    Ranked by similarity to the course currently in the slot FIRST and personal
    fit second. That order is deliberate: a student asking to change a course
    they dislike wants a different way to learn the same material, not the
    next-best course for their career — the second is what the planner already
    gave them.

    Each option carries what it shares with the current course and what the
    current course teaches that it does not, so the trade is visible instead of
    being taken on trust.
    """
    catalog = _catalog_by_id()
    quarter, current = _locate(plan, quarter_key, slot)
    if quarter is None:
        raise ValueError(f"no slot {slot} in quarter {quarter_key!r}")
    if not current["swappable"]:
        return {"quarter": quarter_key, "slot": slot, "current": current,
                "swappable": False, "options": [],
                "why": current.get("note") or "this slot is fixed"}

    in_plan = {row["courseId"] for q in plan["quarters"] for row in q["courses"]
               if row["courseId"]}
    profile = profile_from_intake(answers)
    fit = {row["course"]["id"]: row
           for row in rank_electives(load_catalog(), profile, load_careers())}
    current_course = catalog.get(current["courseId"]) if current["courseId"] else None

    options = []
    for course in load_catalog():
        if course["is_core"] or course["units"] != current["units"]:
            continue
        if not _offered_in(course, quarter["season"]):
            continue
        if _clashes(course["id"], in_plan, catalog) or _clashes(course["id"], taken_ids, catalog):
            continue
        sim = similarity(current_course, course) if current_course else 0.0
        row = fit.get(course["id"]) or {}
        options.append({
            "courseId": course["id"], "code": display_code(course),
            "baseCode": course["code"],
            "title": course["title"], "units": course["units"],
            "technicalLevel": course.get("technical_level"),
            "workload": course.get("workload"),
            "similarity": sim,
            "fitScore": row.get("score", 0.0),
            "sharedSkills": shared_skills(current_course, course) if current_course else [],
            "losesFromCurrent": distinct_skills(current_course, course) if current_course else [],
            # Two courses can serve the same career without sharing skill
            # wording — an NLP course and an ML-theory course overlap in what
            # they are FOR more than in how they describe themselves. Without
            # this, such an option arrives with an empty rationale and the
            # student has to guess why it was offered.
            "sharedFocus": sorted(
                set(course.get("career_tags") or [])
                & set((current_course or {}).get("career_tags") or [])),
            "reasons": row.get("reasons", []),
            "cautions": prerequisite_cautions(course, answers),
        })
    options.sort(key=lambda o: (-o["similarity"], -o["fitScore"], o["code"]))
    # "No options" is a fact about the timetable, not a failure, and a bare
    # empty list reads as the feature being broken. Say which constraint bit.
    why = ""
    if not options:
        why = (f"no other {current['units']}-unit elective is offered in "
               f"{quarter['label']} that is not already in your plan — this "
               f"slot is effectively fixed by the timetable")
    return {"quarter": quarter_key, "slot": slot, "current": current,
            "swappable": True, "options": options[:limit], "why": why}


def selections_for_courses(answers, course_ids, taken_ids=frozenset()):
    """Pin `course_ids` into the slots they legally fit, first fit wins.

    For a career the catalog curates no BUNDLE for. `bundles.skeleton_for`
    cannot help: it derives a slot shape from a known, complete set that spends
    the elective budget exactly, and a web-derived match is neither complete
    nor budget-shaped -- three or four courses against seven slots.

    So this does the weaker, always-possible thing: put each matched course in
    the first slot that can legally take it (right unit size, offered that
    season, not already used, not already taken), and leave the rest of the
    plan to the scorer. The result is a full 50-unit plan built AROUND what the
    job actually needs, rather than a list of courses with no plan attached.

    Deterministic: quarters in published order, slots in order, courses in the
    order they were matched (which is best-first, so the strongest match gets
    the earliest legal slot).
    """
    catalog = _catalog_by_id()
    skeleton = effective_skeleton(answers, taken_ids)
    wanted = [cid for cid in (course_ids or [])
              if cid in catalog and cid not in taken_ids
              and not catalog[cid]["is_core"]]
    selections, used = {}, set()
    for quarter in skeleton:
        for index, slot in enumerate(quarter["slots"]):
            if slot["kind"] != "elective":
                continue
            for course_id in wanted:
                course = catalog[course_id]
                if (course_id not in used
                        and course["units"] == slot["units"]
                        and _offered_in(course, quarter["season"])):
                    selections.setdefault(quarter["key"], {})[str(index)] = course_id
                    used.add(course_id)
                    break
    return selections


def effective_skeleton(answers, taken_ids=frozenset()):
    """The skeleton `build_plan` will ACTUALLY use for these answers.

    The published shape for a custom route, the bundle's derived shape for a
    fixed one, resized to the chosen load either way — the same three steps
    `build_for` and `build_plan` take between them.

    Exists because two callers were re-deriving it and getting it wrong.
    `apply_swap` validated a replacement against `TRACK_SKELETONS[track]` and
    pinned from `build_plan`, both of which ignore the route. That was
    invisible while custom was the default and became a real defect the moment
    the curated bundle became it: a bundle's derived shape has different slot
    SIZES at different INDICES, so a swap was checked against a hole that was
    not the one on screen, and the pinning wrote the scorer's picks into slots
    the bundle had filled with something else. Measured: one swap in Fall moved
    five rows the student never touched — precisely the "plan rearranges itself"
    failure `apply_swap` exists to prevent.
    """
    track = (answers or {}).get("track") or "11 month"
    skeleton, _pinned, _dropped, _unplaced = fixed_plan_inputs(answers, taken_ids)
    if skeleton is None:
        skeleton = TRACK_SKELETONS.get(track)
    if skeleton is None:
        raise ValueError(f"unknown track {track!r}")
    return _resized(skeleton, answers)


class NotOfferedError(ValueError):
    """The replacement exists and fits the slot's size, but not its season.

    Its own class because the caller's reply differs: every other refusal is
    final ("that slot is fixed", "you have already taken it"), while this one
    has a useful next line -- WHERE the course is offered, and which quarter
    of the plan that is. See `placement_note`.
    """


def apply_swap(answers, selections, quarter_key, slot, course_id, taken_ids=frozenset()):
    """Record a swap and hand back the updated selections.

    Validated against the skeleton rather than against the current plan: the
    replacement has to fit the hole (same units, offered that season, not
    already somewhere else in the plan), so a stale or hand-made request cannot
    write a schedule that would not be approved.

    Every auto-filled elective is PINNED as part of the swap, and that is the
    subtle part. Electives are picked greedily from a shared pool, so replacing
    one course frees the course it displaced and the later quarters re-pick
    around it: swapping a Fall elective for Deep Learning released Machine
    Learning back into the pool, which then displaced two Winter courses the
    student never touched. A plan that rearranges itself when you change one
    thing is not a plan you can reason about. Pinning turns the first swap into
    the moment the whole schedule becomes the student's own — after it, only the
    slot they act on moves.
    """
    catalog = _catalog_by_id()
    skeleton = effective_skeleton(answers, taken_ids)
    quarter = next((q for q in skeleton if q["key"] == quarter_key), None)
    if quarter is None:
        raise ValueError(f"unknown quarter {quarter_key!r}")
    if not (0 <= slot < len(quarter["slots"])):
        raise ValueError(f"no slot {slot} in {quarter_key!r}")
    if quarter["slots"][slot]["kind"] != "elective":
        raise ValueError("that slot is fixed by the published plan of study")

    course = catalog.get(course_id)
    if course is None:
        raise ValueError(f"unknown course {course_id!r}")
    if course["is_core"]:
        raise ValueError("a core course cannot fill an elective slot")
    if course["units"] != quarter["slots"][slot]["units"]:
        raise ValueError(
            f"{course['code']} is {course['units']} units and that slot is "
            f"{quarter['slots'][slot]['units']}")
    if not _offered_in(course, quarter["season"]):
        raise NotOfferedError(f"{course['code']} is not offered in {quarter['label']}")
    if _clashes(course_id, taken_ids, catalog):
        raise ValueError(f"you have already taken {course['code']}")

    updated = {key: dict(value) for key, value in (selections or {}).items()}

    # Pin what the planner chose automatically BEFORE changing anything, so the
    # rest of the schedule cannot shift in response to this edit.
    current_plan = build_for(answers, taken_ids, updated)
    for built_quarter, spec in zip(current_plan["quarters"], skeleton):
        for index, row in enumerate(built_quarter["courses"]):
            if spec["slots"][index]["kind"] != "elective" or not row["courseId"]:
                continue
            updated.setdefault(built_quarter["key"], {}).setdefault(
                str(index), row["courseId"])
    # The same course twice is the one duplicate a plan must never contain. If
    # the incoming course is pinned in another slot, that is the student asking
    # to move it, so say where it already is rather than silently relocating it.
    for other_key, slots in updated.items():
        for other_slot, other_id in list(slots.items()):
            if other_id not in equivalents_of(course_id, catalog):
                continue
            if other_key == quarter_key and other_slot == str(slot):
                continue
            label = next((q["label"] for q in skeleton if q["key"] == other_key),
                         other_key)
            held = (catalog.get(other_id) or {}).get("code", other_id)
            raise ValueError(f"{held} is already in your {label} quarter"
                             + ("" if other_id == course_id else
                                f" — the same course as {course['code']}"))

    updated.setdefault(quarter_key, {})[str(slot)] = course_id
    return updated


def route_of(answers):
    """"fixed" or "custom" — how the elective slots get filled.

    Defaults to FIXED for one of the fourteen curated roles, and custom for
    everything else. An explicit `route` on the intake always wins, so a
    student who asked to have it built around them keeps that.

    The default was custom, and flipping it is the point. `bundles.json` holds
    a hand-authored elective set per role and that is the programme's
    recommendation; the scorer is a fallback for the roles nobody curated. The
    curated sets are also far more sharply differentiated: Data Scientist gets
    CSE 251A, MGTA 461, 463 and 466 where Consultant gets MGT 451 and MGTA
    479, because a bundle derives its own slot shape (`bundles.skeleton_for`)
    and can therefore place the 2-unit courses the published skeleton has no
    room for. The scorer, working within the published `[4, 4]` shapes, could
    not reach them at all.
    """
    stated = (answers or {}).get("route")
    if stated in ("fixed", "custom"):
        return stated
    from rsm_thrive.services.courses import bundles

    goals = (answers or {}).get("goals") or []
    return "fixed" if goals and bundles.bundle_for(goals[0]) else "custom"


def fixed_plan_inputs(answers, taken_ids=frozenset()):
    """(skeleton, selections) for a fixed route, or (None, None).

    (None, None) whenever the route is custom, no goal is on file, or the
    bundle cannot be scheduled once the student's completed courses are taken
    out. Every one of those falls back to the ordinary ranked plan rather than
    failing, so a fixed route can never leave a student with no plan at all.
    """
    from rsm_thrive.services.courses import bundles

    if route_of(answers) != "fixed":
        return None, None, False, []
    goals = answers.get("goals") or []
    if not goals:
        return None, None, False, []
    track = answers.get("track") or "11 month"
    dropped = False
    wanted_units = quarter_units_of(
        # The load the student chose, so the bundle is placed inside the
        # quarters they actually asked for rather than inside the published
        # ones and then re-cut. See `bundles.skeleton_for`. The load WORD
        # goes too: with only "heavy" known, `quarter_units_of` seeds the
        # spread from it, and `build_plan` will re-cut to that same seed --
        # placing the bundle for the published spread here and re-cutting
        # to the heavy one there produced a 52-unit plan.
        {"track": track, "quarter_units": answers.get("quarter_units"),
         "workload": answers.get("workload")})
    skeleton, selections, courses = bundles.skeleton_for(
        goals[0], track, taken_ids, wanted_units)
    if skeleton is None and wanted_units:
        # THE BUNDLE OUTRANKS THE SPREAD.
        #
        # A light or heavy spread moves units between quarters, and a bundle
        # whose courses are only taught in particular terms often cannot be
        # laid into the result. When that happened the whole curated bundle
        # was abandoned and the plan fell through to the generic ranked fill
        # -- silently, and the student was never told their plan had stopped
        # being the one built for their career.
        #
        # Measured across every role, track and load: 33 of 84 combinations
        # lost their bundle this way. `ml-engineer` on a light spread came
        # back with 3 of 9 electives serving the goal, against 8 of 9 on the
        # published spread, and with none of its three CS courses.
        #
        # So the published spread is tried next, and only a bundle that
        # cannot be placed there at all falls back to the ranked plan. Which
        # of the two to give up is not a close call: the spread is a
        # preference the student can still change one quarter at a time in
        # the walk-through -- the load question says so in as many words --
        # while the bundle is the entire reason the plan is about their
        # career rather than about the catalog.
        skeleton, selections, courses = bundles.skeleton_for(
            goals[0], track, taken_ids, None)
        dropped = True
    # Recommended courses the timetable left no room for. `courses` is what
    # the bundle search could actually lay into the quarters; anything in the
    # bundle that is schedulable for this student and not in it was squeezed
    # out by the terms it runs in, and the plan says so.
    unplaced = []
    if skeleton is not None:
        bundle = bundles.bundle_for(goals[0]) or {}
        catalog = _catalog_by_id()
        for layer in ("anchor", "differentiator", "universal"):
            for course_id in bundle.get(layer) or []:
                if (course_id in catalog and course_id not in taken_ids
                        and course_id not in bundles.PRE_PLACED
                        and course_id not in (courses or [])):
                    unplaced.append(display_code(catalog[course_id]))
    return skeleton, selections, bool(skeleton is not None and dropped), unplaced


def _tail_shortfall(answers, plan):
    """{'wanted', 'actual', 'label'} when the finishing quarter missed the
    load's target (`tail_target`), else None."""
    track = answers.get("track") or "11 month"
    key = tail_quarter_key(track)
    wanted = tail_target(track, answers.get("workload") or "moderate")
    if key is None or wanted is None:
        return None
    quarter = next((q for q in plan["quarters"] if q["key"] == key), None)
    if quarter is None or quarter["unitsPlanned"] == wanted:
        return None
    return {"wanted": wanted, "actual": quarter["unitsPlanned"],
            "label": quarter["label"]}


def stated_completed_ids(answers):
    """Catalog ids for the codes a student said they had finished."""
    codes = {str(c).upper() for c in (answers or {}).get("completed_codes") or []}
    return frozenset(course["id"] for course in load_catalog()
                     if course["code"].upper() in codes)


def build_for(answers, taken_ids=frozenset(), selections=None, start_from=None):
    """Build this student's plan by whichever route they chose.

    The one place that decides. A fixed route hands `build_plan` a skeleton
    shaped to its bundle and the bundle pinned into it; a custom route calls
    `build_plan` exactly as everything did before.

    A student's own `selections` still win in both -- a swap is a swap, and on
    a fixed route it is also a DIVERGENCE, which `bundles.divergence` reports so
    the reply can say what was given up.

    `start_from` passes straight through to `build_plan`. On a FIXED route it
    also reshapes the bundle: `bundles.skeleton_for` sizes slots against the
    quarters a bundle can actually be placed in, and once some of those
    quarters are behind the student the bundle may no longer fit at all. When
    it does not, this falls back to the ranked plan rather than returning
    nothing -- the same fallback `fixed_plan_inputs` already makes for a bundle
    that cannot be scheduled around completed courses.
    """
    # Courses the student SAID they have finished, in this conversation. They
    # join the enrolment record here, in the one place every plan is built,
    # rather than at the dozen call sites that pass `taken_ids` in. A student
    # who types "I've already done MGTA 464 and 402" and is then handed a
    # plan with both of them in it has been ignored, and was.
    taken_ids = frozenset(taken_ids) | stated_completed_ids(answers)
    wanted = route_of(answers)
    skeleton, pinned, spread_dropped, unplaced = fixed_plan_inputs(answers, taken_ids)
    if skeleton is None:
        return _routed(build_plan(answers, taken_ids, selections,
                                  start_from=start_from), "custom", wanted)
    merged = {key: dict(value) for key, value in (pinned or {}).items()}
    for quarter_key, chosen in (selections or {}).items():
        merged.setdefault(quarter_key, {}).update(chosen)
    if spread_dropped:
        # And the chosen spread has to come OFF the answers too, not just out
        # of the skeleton. `build_plan` re-cuts a skeleton to
        # `answers["quarter_units"]`, so leaving it on put the bundle back
        # into the very distribution it could not be placed in -- measured,
        # `ml-engineer` on a light spread recovered only 6 of its 9 on-goal
        # electives instead of 8. Set to the PUBLISHED spread explicitly:
        # `quarter_units_of` seeds a missing spread from the load word, so
        # merely removing the key would put the dropped spread straight back.
        answers = {**answers, "quarter_units": {
            q["key"]: q["default"]
            for q in adjustable_quarters(answers.get("track") or "11 month")}}
    try:
        plan = _routed(build_plan(answers, taken_ids, merged, skeleton=skeleton,
                                  start_from=start_from), "fixed", wanted)
        # The student asked for a spread this bundle could not be laid into,
        # and the bundle won. Said out loud rather than left for them to
        # notice the quarters are not what they chose.
        plan["spreadDropped"] = spread_dropped
        # Recommended courses left out because of the terms they run in.
        plan["bundleUnplaced"] = unplaced
        # Whether the finishing quarter landed where the load says it should.
        # For some bundles it cannot: bi-analyst has 457, 402, 459 and MGTF
        # 405 offered in the final Fall, but sending 4 of those units there
        # leaves Spring short of its 12-unit graduate minimum, and no
        # differentiator combination changes that. Then the tail carries 2
        # and the plan SAYS so, rather than leaving a student who asked for a
        # light finish to notice the number on their own.
        plan["tailShort"] = _tail_shortfall(answers, plan)
        return plan
    except ValueError:
        # The bundle's derived skeleton omits a quarter the student says they
        # are in, so the bundle cannot describe where they actually are.
        return _routed(build_plan(answers, taken_ids, selections,
                                  start_from=start_from), "custom", wanted)


def _routed(plan, used, wanted):
    """Record which fill the plan ACTUALLY got, and whether that was the ask.

    A plan is the only thing that knows this. `route_of` says what was asked
    for; the bundle can still turn out to be unschedulable once the student's
    completed courses come out, and then the ranked fill runs instead. Reading
    the request and assuming it happened is how a plan filled by the scorer got
    a "you have moved off the recommended bundle" warning it had no business
    carrying.
    """
    plan["route"] = used
    plan["routeFellBack"] = used != wanted
    # Quarters the fill could not land on their target load, derived from the
    # plan rather than plumbed out of the search: `bundles` allows a quarter to
    # sit `QUARTER_FLEX` units off so that all fourteen bundles can be
    # scheduled at all, and where it takes that concession the student must be
    # told -- they have just chosen these loads.
    plan["quarterFlex"] = [
        {"label": quarter["label"], "planned": quarter["unitsPlanned"],
         "expected": quarter["unitsExpected"]}
        for quarter in plan["quarters"]
        if quarter["unitsPlanned"] != quarter["unitsExpected"]]
    if used == "fixed":
        from rsm_thrive.services.courses import bundles

        goals = plan.get("profile", {}).get("career_roles") or []
        plan["bundleMissing"] = bundles.absent(plan, goals[0]) if goals else {}
    else:
        plan["bundleMissing"] = {}
    return plan


def taken_course_ids(user):
    """Catalog ids for courses the student has already enrolled in."""
    codes = {enrollment.course.code for enrollment
             in Enrollment.objects.filter(user=user).select_related("course")}
    return {course["id"] for course in load_catalog() if course["code"] in codes}


def completed_course_ids(user):
    """Catalog ids for courses the student has actually FINISHED.

    Distinct from `taken_course_ids`, which counts anything enrolled in --
    including the quarter they are sitting in right now. Both are correct for
    their own question: "do not schedule this again" wants everything enrolled,
    and "how many units do you have" wants only what is banked. Using the wider
    set for the ledger would credit units for a course still being taught,
    which is the kind of arithmetic error a student only finds out about at
    graduation.
    """
    codes = {enrollment.course.code for enrollment
             in Enrollment.objects.filter(user=user, completed=True)
                                  .select_related("course")}
    return {course["id"] for course in load_catalog() if course["code"] in codes}


# ---------------------------------------------------------------------------
# Presenting a plan
# ---------------------------------------------------------------------------

def goal_labels(plan):
    """The career(s) this plan was built for, as a student would name them."""
    careers = load_careers()
    labels = [careers[role_id]["label"] for role_id
              in (plan.get("profile", {}).get("career_roles") or [])
              if role_id in careers]
    return ", ".join(labels) or plan.get("profile", {}).get("stated_goal", "")


def route_line(plan):
    """One line saying how the electives were filled, and what it cost.

    A plan that does not say where its electives came from is two different
    plans wearing the same header: the set recommended for this career, or a
    ranked fill against this student's own skills. Which one
    is on screen changes what "swap this" means and what the plan is evidence
    of, so it is stated rather than left to be inferred from the courses.
    """
    if not goal_labels(plan):
        return ""
    if plan.get("route") != "fixed":
        from rsm_thrive.services.courses import bundles

        # Only offer the switch where there is something to switch TO. A
        # career the catalog has no bundle for got "say use the recommended
        # bundle" underneath its plan -- an instruction that could not work,
        # for a set that does not exist. Same guard `route_switch_reply`
        # already applies to the button.
        roles = plan.get("profile", {}).get("career_roles") or []
        if roles and bundles.bundle_for(roles[0]):
            line = ("_Electives filled against your own skills, workload and "
                    "interests. Say **use the recommended bundle** for the set "
                    "I'd recommend for that career instead._")
        else:
            line = ("_There's no ready-made set for that career, so these "
                    "electives are filled against what the job asks for plus "
                    "your own skills and workload._")
        if plan.get("routeFellBack"):
            line = ("_The set I'd normally recommend for that career can't be "
                    "scheduled around the courses you've already taken, so "
                    "these electives are filled against your own skills and "
                    "workload instead._")
        return line
    line = ("_These are the electives I'd recommend for that career — a "
            "recommendation, not a requirement, and the same one everyone "
            "aiming at it gets. Say **build it around me** to fill them "
            "against your own skills and workload instead._")
    flexed = plan.get("quarterFlex") or []
    if flexed:
        line += ("\n\n> To fit the recommended set, "
                 + ", ".join(f"**{q['label']}** carries {q['planned']} units "
                             f"rather than {q['expected']}" for q in flexed)
                 + ". The degree total is unchanged. Say **build it around "
                   "me** to hold the loads exactly instead.")
    unplaced = plan.get("bundleUnplaced") or []
    if unplaced:
        # The timetable, not the student's load, is what left these out: the
        # bundle search could not lay them into the quarters they are taught
        # in, so their units went to the best fit for the goal instead.
        line += ("\n\n> Not all of the recommended set fits the quarters it "
                 "is taught in: "
                 + ", ".join(f"**{code}**" for code in unplaced)
                 + (" is" if len(unplaced) == 1 else " are")
                 + " left out, and those units go to the next-best fit for "
                   "your goal. Say **change** and a course in the walk-through "
                   "to swap one back in.")
    missing = plan.get("bundleMissing") or {}
    dropped = [code for layer in ("anchor", "universal")
               for code in missing.get(layer, []) if code not in unplaced]
    if dropped:
        # Say it here rather than let a student compare two lists. The load
        # spread re-cuts every quarter's elective slots, and a 2-unit course
        # can end up with nowhere to sit.
        line += ("\n\n> The load you chose leaves no room for "
                 + ", ".join(f"**{code}**" for code in dropped)
                 + ", which the recommended set includes. Say **moderate** for "
                 "that quarter, or **build it around me**, and they come back.")
    return line


def elective_codes(plan):
    """The swappable electives in a plan, as a set. For comparing two plans."""
    return {row["code"] for quarter in plan["quarters"]
            for row in quarter["courses"]
            if row["swappable"] and row["courseId"]}


def render_plan_markdown(plan):
    """A plan as Markdown the app already knows how to render.

    Tables and links rather than prose: a plan of study is a grid, and
    `RichMessage` renders Markdown tables and links as real `<table>` and `<a>`
    elements, so this arrives formatted instead of as a wall of pipes.

    Every quarter lists its CORE courses alongside its electives and marks
    which is which, because "what am I taking in Winter" is the question a plan
    exists to answer, and an elective-only list cannot answer it.
    """
    totals = plan["totals"]
    lines = [
        # The GOAL is in the heading, not just in the reasons under each row.
        # A student who changed their mind from data analyst to consultant got
        # a plan whose header was identical, whose Summer was identical (it is
        # all required), and which said nothing anywhere about which career it
        # was for -- so a rebuild that had genuinely happened looked like the
        # bot repeating itself.
        f"# Your {plan['track']} MSBA plan of study"
        + (f" — {goal_labels(plan)}" if goal_labels(plan) else ""),
        "",
        route_line(plan),
        "",
        f"**{totals['total']} units** — {totals['core']} core and "
        f"{totals['elective']} elective, against the "
        f"{totals['totalRequired']} the degree requires "
        f"({totals['coreRequired']} core, {totals['electiveRequired']} elective).",
        "",
    ]
    # A short plan must say so where the total is, not leave the reader to add
    # the table up. The rows already show "nothing available for this slot";
    # what was missing is that the HEADER used to claim the full 50 anyway.
    if plan["unfilled"]:
        short = totals["totalRequired"] - totals["total"]
        lines += [
            f"> ⚠️ **This plan is {short} units short of the {totals['totalRequired']} "
            f"you need to graduate.** "
            + " ".join(u["why"].capitalize() + "." for u in plan["unfilled"])
            + " Take this to MSBA advising — book them from the **Appointments** "
              "tab — so the remaining units can be filled from outside this "
              "catalog or by petition.",
            "",
        ]
    for quarter in plan["quarters"]:
        lines += [
            f"## {quarter['label']} — {quarter['unitsPlanned']} units",
            "",
            "| Course | Title | Units | Requirement |",
            "|---|---|---|---|",
        ]
        for row in quarter["courses"]:
            if not row["courseId"]:
                lines.append(f"| — | _{row['note']}_ | {row['units']} | "
                             f"{row['requirement']} |")
                continue
            lines.append(f"| **{row['code']}** | {row['title']} | "
                         f"{row['units']} | {row['requirement']} |")
        lines.append("")

        # A quarter with nothing to choose says so, rather than leaving a
        # student looking for the decision they are meant to make. Summer is
        # the case: every course in it is required. The second sentence is
        # there because two of those courses count as ELECTIVE units in the
        # published plan of study, so the totals above would otherwise look
        # like they disagreed with this table.
        if all(not row["swappable"] and row["courseId"] for row in quarter["courses"]):
            counts_as_elective = [row for row in quarter["courses"]
                                  if row["kind"] == "elective"]
            note = "Everything this quarter is required — there is nothing to choose."
            if counts_as_elective:
                codes = " and ".join(f"**{row['code']}**"
                                     for row in counts_as_elective)
                note += (f" {codes} still count toward your elective units, which "
                         f"is how the totals above add up.")
            lines += [note, ""]

        notes = []
        for row in quarter["courses"]:
            if row.get("stretch"):
                notes.append(f"- **{row['code']}** — {row['stretch'][0]}")
            for caution in row.get("cautions") or []:
                notes.append(f"- **{row['code']}** — {caution}")
        if notes:
            lines += ["Worth knowing before you book:", "", *notes, ""]

    goal_rows = [r for q in plan["quarters"] for r in q["courses"]
                 if r["requirement"] == "Elective" and r.get("onGoal")]
    breadth = [r for q in plan["quarters"] for r in q["courses"]
               if r["requirement"] == "Elective" and r.get("onGoal") is False]
    if goal_rows or breadth:
        lines += ["## Why these electives", ""]
        if goal_rows:
            lines.append("Chosen for your stated goal: "
                         + ", ".join(f"**{r['code']}**" for r in goal_rows) + ".")
        if breadth:
            lines.append("")
            lines.append("Filling the remaining elective units for breadth: "
                         + ", ".join(f"**{r['code']}**" for r in breadth)
                         + ". Any of these can be swapped for something closer "
                           "to your interests.")
        lines.append("")

    targeting = plan.get("targeting")
    if targeting:
        lines += [
            "## Worth a conversation with advising", "",
            f"Only **{targeting['onGoal']} of {targeting['electiveSlots']}** "
            f"elective slots line up with "
            f"{' and '.join(targeting['goals'])} — the rest are breadth, because "
            f"the courses aimed at that path are not all offered in the quarters "
            f"and sizes this plan needs. "
            # Only claim completeness when the plan IS complete. This sentence
            # used to hardcode "a complete 50-unit plan" and printed it directly
            # under a table containing an empty slot.
            + (f"It is still a complete {plan['totals']['totalRequired']}-unit plan, and "
               if not plan["unfilled"] else "")
            + f"MSBA advising can advise on getting closer to that goal — book them "
            f"from the **Appointments** tab.", ""]

    lines += ["## Booking and detail", ""]
    for link in plan["links"]:
        lines.append(f"- [{link['label']}]({link['url']}) — {link['note']}")
    lines += ["", f"_{plan['disclaimer']}_"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The interview, run as a conversation
# ---------------------------------------------------------------------------

# THE CHAT NO LONGER ASKS ANY OF THIS.
#
# `services/orchestrator.py` reads a track and a goal off what the student
# types (both closed sets, both deterministic), assumes the rest, and says what
# it assumed. `next_intake_step` survives as the test for "is this intake
# complete enough to build a plan from", which the orchestrator and
# `views/planner.py` both still need, and the question script below is still
# served by `GET /plan/intake` for a form-based planner surface. What is gone
# is the chat asking them one at a time: see `opening_prompt`.
#
# Asked in this order, and grouped so the student is not interrogated one field
# at a time. Track comes first because it decides how many quarters exist and
# which quarter every course lands in — asking anything else before it risks
# building toward a plan that then has to change shape.
INTAKE_STEPS = [
    {"key": "track", "fields": ["track"]},
    {"key": "goals", "fields": ["goals"]},
    {"key": "quarter_units", "fields": ["quarter_units"]},
]

# `interests` is deliberately absent above: it improves a plan and is not
# required to build one, so it is never a gate between a student and their plan.
# `route` joins `interests` here rather than becoming a fifth question. The
# interview is four steps and the design document is emphatic about keeping it
# short; more to the point, asking someone to choose between "the recommended
# bundle" and "built around me" BEFORE they have seen either is asking them to
# decide with nothing to look at. The plan is built the way every plan was
# built, and the other route is then offered as a button underneath it.
OPTIONAL_FIELDS = {"interests", "route", "workload"}


def _allowed_values():
    allowed = {}
    for question in intake_questions():
        allowed[question["key"]] = {opt["value"] for opt in question["options"]}
    # The skill areas are NOT questions any more -- nothing asks a student to
    # rate themselves -- but they are still valid ANSWERS, so they stay in the
    # allowlist. `read_skills` picks a rating out of "strong in python, never
    # done ML", and a student who volunteers that should have it used. Deriving
    # this map from the question list alone silently dropped exactly those
    # ratings on the floor: being asked and being told are different things,
    # and only the asking was meant to go.
    for area in SKILL_AREAS:
        allowed.setdefault(f"skill_{area['key']}",
                           {lvl["value"] for lvl in SKILL_SCALE})
    return allowed


# Below this share of a plan's elective slots actually targeting the student's
# goal, the plan is mostly breadth and should say so.
#
# Measured from the BUILT PLAN rather than from catalog tag counts, because the
# two disagree badly. Counting tagged electives said healthcare-analyst was
# well served (15 courses) and operations-supply-chain was not (5) — but the
# plans land at 4/7 and 3/7 on-goal respectively, nearly the same. What decides
# it is which of those courses are offered in the right quarter at the right
# unit size, not how many exist. So the old proxy excluded one role while
# shipping another that was barely better.
MIN_ON_GOAL_SHARE = 0.5


def matched_elective_count(role_id):
    """How many electives share a career tag with this role."""
    role = load_careers().get(role_id) or {}
    tags = set(role.get("career_tags") or [])
    if not tags:
        return 0
    return sum(1 for course in load_catalog()
               if not course["is_core"]
               and set(course.get("career_tags") or []) & tags)


def targeting_note(quarters, goals):
    """Whether this plan is mostly breadth, judged from the plan itself.

    Every role gets a plan — there are 28 elective units to fill and the catalog
    can always fill them. What varies is how much of that fill actually targets
    the student's goal, and that is knowable exactly once the plan exists: each
    elective row already records whether it matched a goal tag. No proxy, no
    threshold on a number that turned out not to predict the outcome.
    """
    electives = [row for quarter in quarters for row in quarter["courses"]
                 if row["requirement"] == "Elective" and row["swappable"]]
    if not electives or not goals:
        return None
    on_goal = sum(1 for row in electives if row.get("onGoal"))
    if on_goal / len(electives) >= MIN_ON_GOAL_SHARE:
        return None
    careers = load_careers()
    return {
        "onGoal": on_goal,
        "electiveSlots": len(electives),
        "goals": [careers.get(role_id, {}).get("label", role_id) for role_id in goals],
    }


def uncovered_career_reply(named):
    """What to say when the student's target job is not one the catalog serves.

    Deliberately NOT the list of ten roles. Answering "I want to be an esports
    analyst" with "pick one of these instead" spams a menu at someone whose
    actual question — does this degree get me there — has just been answered
    with no. The honest reply is that the MSBA electives are not built for it
    and that a human should weigh in, with the concrete way to reach one.

    One line does offer a way forward, because a dead end is its own failure,
    but it names no roles: a student who wants to proceed can say so and the
    interview picks up from there.
    """
    subject = f"**{named}**" if named else "that career"
    return (
        f"I don't have MSBA course material built around {subject}, so I'd be "
        f"guessing if I put a plan together for it — and a plan built on a guess "
        f"is worse than no plan.\n\n"
        f"**Please check with MSBA advising.** They can tell you whether the "
        f"programme supports that path and which electives would come closest. "
        f"You can book time with them from the **Appointments** tab.\n\n"
        f"If you'd rather I build a plan around a role the catalog does cover, "
        f"tell me which one you're closest to and I'll carry on from there."
    )


def unmatched_goal_of(raw):
    """The student's own words when they named a career we have no track for."""
    if not isinstance(raw, dict):
        return ""
    value = raw.get("unmatched_goal")
    return value.strip()[:60] if isinstance(value, str) and value.strip() else ""


def normalise_intake(raw):
    """Keep only the answers an extractor produced that are actually valid.

    The extractor is an LLM reading a free-text conversation, so it will
    occasionally return "17 months", a role that does not exist, or a skill
    level it invented. Dropping those rather than repairing them means the
    interview asks the question again, which is the right outcome: a guessed
    answer becomes a plan the student never agreed to.
    """
    if not isinstance(raw, dict):
        return {}
    allowed = _allowed_values()
    answers = {}
    for key, values in allowed.items():
        value = raw.get(key)
        if key == "goals":
            # Resolved, not just filtered: a stored plan or a mid-interview
            # extraction may name a role id from before the taxonomy moved to
            # the fourteen profiles. `resolve_role` maps those forward and
            # still returns None for a role nobody has ever offered, so an
            # invented goal is dropped and re-asked exactly as before.
            chosen, seen = [], set()
            for v in (value or []):
                resolved = resolve_role(v) if isinstance(v, str) else None
                if resolved and resolved not in seen:
                    seen.add(resolved)
                    chosen.append(resolved)
            if chosen:
                answers[key] = chosen[:3]
        elif key == "quarter_units":
            # A dict of {quarter: units}, kept only when it is a legal
            # distribution. A partial or impossible one is dropped so the step
            # asks again, exactly like every other answer the extractor gets
            # wrong -- a plan must never be built on a load that cannot be taken.
            track = raw.get("track") or answers.get("track") or "11 month"
            if isinstance(value, dict):
                cleaned = {k: int(v) for k, v in value.items()
                           if isinstance(v, (int, float)) and not isinstance(v, bool)}
                if cleaned and not quarter_units_problems(track, cleaned):
                    answers[key] = cleaned
        elif key == "interests":
            chosen = [v for v in (value or []) if isinstance(v, str) and v in values]
            if chosen:
                answers[key] = chosen
        elif key.startswith("skill_"):
            # Stored as 1-5 whatever it arrived as, so everything downstream
            # reads one type: an extractor may send 4, "4" or "comfortable".
            score = skill_score(value)
            if score is not None:
                answers[key] = score
        elif isinstance(value, str) and value in values:
            answers[key] = value
    return answers


# After this many attempts at the same step, assume a middling answer for what
# is still blank rather than asking again. Measured over 50 conversations: a
# student who answered "I've never written code, stats basic, I present well"
# left SQL and machine learning unstated, and the interview asked for them three
# times in a row and never produced a plan. An interview with no exit is worse
# than an assumption the student can see and correct.
MAX_STEP_ATTEMPTS = 2

# What an unanswered skill area is assumed to be: the NEUTRAL MIDDLE, the same
# value `DEFAULT_SKILL_RATING` gives the form.
#
# It was "basic" (2), on the argument that under-claiming is safer than
# over-claiming. Measured, that argument is wrong, and the cost is the thing a
# student notices first. `rank_electives` penalises a course 1.5 points per
# level above the stated comfort, and most electives sit at level 3 or 4, so an
# assumed 2 swamps the career-tag boost (about 3.0 for a course's primary tag)
# and every goal collapses onto the same gentle courses:
#
#     distinct elective sets across the 14 curated roles, custom route
#     assumed level    11-month    17-month
#     1 (new to it)      7/14        3/14
#     2 (some exposure) 10/14        6/14
#     3 (working)       14/14       13/14
#
# At 2 on the 17-month track, three different careers genuinely produced the
# same six courses -- which is exactly what a student reported, having watched
# "data analyst", "consultant" and "data scientist" return one plan.
#
# Under-claiming never made the plan safer either. Safety here is DISCLOSURE:
# `_stretch_notes` marks every course above the level the student gave, on the
# row, every time. Quietly degrading the recommendation is not a substitute for
# saying so. Since the self-rating questions were removed this is the NORMAL
# case rather than the exception: nothing asks for a rating, so every area sits
# at the assumed level unless a student volunteered one.
ASSUMED_SKILL = "working"


# The words a student uses for an area, beyond its own name. Read off
# `SKILL_AREAS` where possible and extended only where the label is not what
# anyone types: nobody writes "Presenting and storytelling".
SKILL_SAID_AS = {
    "python": ("python", "programming", "coding", "code", "pandas"),
    "sql": ("sql", "database", "databases", "queries", "querying", "etl"),
    "stats": ("stats", "statistics", "statistical", "regression",
              "probability", "econometrics"),
    "ml": ("ml", "machine learning", "modelling", "modeling", "deep learning",
           "ai"),
    "communication": ("communication", "communicating", "presenting",
                      "presentation", "presentations", "storytelling",
                      "speaking", "writing"),
}

# "skip", said in the ways people say it. A student who declines the question
# has answered it, and must not be asked again.
SKILL_DECLINES = frozenset({
    "skip", "skip it", "skip this", "no idea", "not sure", "dunno",
    "don't know", "dont know", "no preference", "whatever", "you decide",
    "just build it", "just go", "go ahead", "doesn't matter", "doesnt matter",
    "n/a", "na", "none of them", "pass",
})


def declines_skills(text):
    """Did the student decline to rate themselves?"""
    lowered = re.sub(r"[^a-z' /]+", " ", (text or "").strip().lower())
    return " ".join(lowered.split()) in SKILL_DECLINES


def read_skills(text):
    """{skill_<area>: 1-5} for every area this message rates. Deterministic.

    Reads the shapes people actually write -- "python 4, sql 2", "strong in
    python, no ML", "3 for stats" -- by finding each area's own vocabulary and
    then the nearest rating word or digit to it. No model call: the areas are a
    closed list of five and the ratings a closed list of five words plus five
    digits, so there is nothing here for a model to be better at, and a model
    reading it would put a round trip in front of the plan.

    A rating is only taken from WITHIN the same clause as the area it belongs
    to. "python 4, sql 2" splits on the comma; without that, "sql" would take
    the 4 from the phrase before it and the student would be handed a plan
    built on a level they did not claim.
    """
    ratings = {}
    lowered = " " + " ".join((text or "").lower().split()) + " "
    # Clause boundaries a person actually types when listing several ratings.
    clauses = re.split(r"[,;.]|\band\b|\bbut\b|/", lowered)
    for clause in clauses:
        found = [key for key, said_as in SKILL_SAID_AS.items()
                 if any(re.search(rf"\b{re.escape(word)}\b", clause)
                        for word in said_as)]
        if not found:
            continue
        # A STANDALONE 1-5. The lookarounds are the whole point: `\b([1-5])\b`
        # read "sql -3" as 3 and "python 3.5" as 3, inventing a rating out of a
        # minus sign and the integer part of a decimal. A student who types
        # either has not given a level on this scale, and guessing one for them
        # is how a plan gets built on a claim nobody made.
        digits = re.findall(r"(?<![\d.\-])([1-5])(?![\d.])", clause)
        words = [word for word in re.findall(r"[a-z']+", clause)
                 if word in SKILL_WORDS]
        # A negation reads as the bottom of the scale: "no ML" and "never
        # written python" are ratings, and the commonest way of giving one.
        if re.search(r"\b(no|not|never|zero|nothing|none|cant|can't)\b", clause):
            score = 1
        elif digits:
            score = int(digits[0])
        elif words:
            score = SKILL_WORDS[words[0]]
        else:
            continue
        for key in found:
            ratings[f"skill_{key}"] = score
    return ratings


def fill_assumed_skills(answers, missing):
    """Assume the unanswered skill areas, and report what was assumed."""
    filled = dict(answers)
    assumed = []
    for field in missing:
        if field.startswith("skill_") and not filled.get(field):
            filled[field] = ASSUMED_SKILL
            area = next(a for a in SKILL_AREAS if f"skill_{a['key']}" == field)
            assumed.append(area["label"])
    return filled, assumed


def next_intake_step(answers):
    """The next group of questions still unanswered, or None when complete.

    `unmatched_goal` satisfies the goals step. A student who says "esports
    analyst" HAS told us what they are aiming for -- it simply is not one of
    the fourteen ids `careers.json` curates, and treating that as "no goal"
    left them answering the question they had just answered, and never reaching
    a plan at all.
    """
    for step in INTAKE_STEPS:
        missing = [field for field in step["fields"]
                   if field not in OPTIONAL_FIELDS and not answers.get(field)]
        if step["key"] == "goals" and answers.get("unmatched_goal"):
            missing = []
        # The load question replaced "light / moderate / heavy", and an intake
        # answered before it existed carries the old word instead. That word is
        # the same information in coarser form, so a saved plan is COMPLETE and
        # keeps working -- without this, every plan built before the change
        # answered `GET /api/thrive/plan` with a 409 asking for a question its
        # student was never shown.
        if step["key"] == "quarter_units" and answers.get("workload"):
            continue
        # And on a track where the load is not a choice there is nothing to
        # answer: the intake is complete with a track and a goal. Without
        # this, an 11-month plan was delivered and then "walk me through it"
        # was refused as an unfinished interview, straight to the classifier.
        if step["key"] == "quarter_units" and not load_is_a_choice(
                answers.get("track") or "11 month"):
            continue
        if missing:
            return {"key": step["key"], "missing": missing}
    return None


def _question(key):
    return next(q for q in intake_questions() if q["key"] == key)


def step_position(step):
    """Where the question being asked sits in the interview, 1-based.

    The position of THIS step, not a count of finished ones. Counting finished
    steps looked equivalent and was not: a student who names a career before
    stating a track has answered step 2 and not step 1, so the header read
    "Step 2 of 4" above the track question.
    """
    for index, candidate in enumerate(INTAKE_STEPS, start=1):
        if candidate["key"] == step["key"]:
            return index
    return 1


def render_question(step, answers, unmatched_goal=""):
    """The next question, as Markdown the chat already renders.

    Bullet lists rather than prose: these are choices, and `RichMessage` turns
    `- ` lines into a real list. The step counter is there because an interview
    with no visible end is one students abandon.
    """
    lines = [f"**Step {step_position(step)} of {len(INTAKE_STEPS)}.**", ""]
    # A student who names a career the catalog has no track for should be told
    # that, not shown the same list again as though they had said nothing.
    if unmatched_goal and step["key"] == "goals":
        lines += [f"I don't have a track built around **{unmatched_goal}** — the "
                  f"roles below are the ones the elective scoring knows about. "
                  f"Pick whichever comes closest and I'll work from that.", ""]

    question = _question(step["missing"][0])
    lines.append(question["prompt"])
    if question.get("help"):
        lines += ["", f"_{question['help']}_"]

    # The options are printed ONLY when this step has no buttons to carry them.
    # Every step that does (`quick_replies_for`) now puts each option's
    # explanation on its own button, and printing the same list above them made
    # one question into five stacked blocks — a step counter, a prompt, a help
    # line, a bullet list, and a button row saying the list again.
    #
    # The fallback still matters: a client that ignores `quickReplies` has to
    # render a usable question, and this is the message body it renders.
    if not quick_replies_for(step):
        lines.append("")
        for option in question["options"]:
            label = option["label"]
            description = option.get("description")
            lines.append(f"- **{label}**" + (f" — {description}" if description else ""))
    return "\n".join(lines)


# A turn shaped like a question rather than an answer. Deterministic, for the
# same reason `review_intent` is: the decision has two outcomes and an extra
# model call would add a round trip and a failure mode to each one.
_QUESTION_OPENERS = (
    "what", "why", "how", "when", "where", "who", "which", "whose",
    "can i", "can you", "can we", "could i", "could you", "could we",
    "should i", "should we", "would it", "would i", "would you",
    "do i", "do you", "do we", "does it", "does this", "did you", "have you",
    "is it", "is this", "is there", "are you", "are there", "are we",
    "will it", "will you", "am i", "tell me", "explain", "show me what",
    # Bare auxiliaries, because a question does not have to name its subject in
    # the second word: "is next quarter heavy?" is one, and with only "is it" /
    # "is this" listed it read as an ANSWER -- so "heavy" was recorded as the
    # student's workload preference from a question that stated no preference.
    # Safe alongside the trailing-"?" requirement: a hedged answer starts with
    # its own value ("11 month?", "heavy?"), not with an auxiliary.
    "is ", "are ", "was ", "were ", "does ", "do ", "did ", "can ", "could ",
    "should ", "would ", "will ", "has ", "have ", "am ",
)
_HYPOTHETICAL = ("what if", "what happens if", "should i have", "would it be",
                 "instead of", "rather than", "suppose i", "if i switched",
                 "if i switch", "if i chose", "if i picked")


def is_hypothetical(text):
    """Is this turn a what-if -- a value mentioned to compare, not to choose?

    Narrower than `is_question`: "can I still finish on time on the 11 month
    track?" is a question that STATES a track, and "what if I had gone for
    the 17 month instead" is one that does not. Only the second must never
    move a value already on file, whichever reader saw it.
    """
    lowered = (text or "").strip().lower()
    return any(phrase in lowered for phrase in _HYPOTHETICAL)


def is_question(text):
    """Is this turn asking something rather than answering?

    Used for two different refusals, and it is deliberately the same test for
    both: a question must not overwrite an answer already on file
    (`merge_intake`), and a question must not be treated as a request to
    reprint the plan (`answer_electives`).

    Trailing "?" alone is not enough -- a student answering uncertainly types
    "11 month?" -- so the opener has to look interrogative too. That keeps a
    hedged ANSWER an answer while catching "what if I switch to 17 month?".
    """
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    if any(phrase in lowered for phrase in _HYPOTHETICAL):
        return True
    return lowered.endswith("?") and lowered.startswith(_QUESTION_OPENERS)


def merge_intake(stored, extracted, asking=False):
    """Accumulate interview answers across turns.

    Newly extracted values win, so a student correcting themselves ("actually
    the 11 month") is respected; anything the latest turn did not mention keeps
    the value already on file. This is the piece that makes the interview
    stateful without a state machine — and without trusting a language model to
    remember what it was told three messages ago.

    `asking` narrows that to filling BLANKS only. A question mentions values
    without choosing them, and the extractor cannot tell the difference: asked
    "what if I switch to 17 month?", it dutifully reported track="17 month" and
    the student's committed plan of study was rebuilt on the other track --
    `CoursePlan` included, which is per-user, outlives the conversation and is
    what `/api/thrive/plan` serves. A student asking what-if got their actual
    plan replaced.

    Blanks are still filled, because a question can carry genuinely new
    information ("what if I want to be a data scientist?" when no goal is on
    file yet is worth acting on). Only OVERWRITING is refused, which is where
    the harm was.
    """
    merged = dict(stored or {})
    for key, value in (extracted or {}).items():
        if value in (None, "", []):
            continue
        if asking:
            # A question commits NOTHING -- not even into a blank. Letting one
            # fill blanks looked harmless ("what if I want to be a data
            # scientist?" naming a goal) and is the same mistake in a quieter
            # form: measured live, "is next quarter heavy?" wrote
            # workload="heavy" from a turn that expressed no preference at all.
            # The interview asks again instead, which is what the design calls
            # for -- a value the student did not give must never become part of
            # a plan they are then shown.
            continue
        merged[key] = value
    return merged


def load_session_intake(conversation):
    """Answers gathered so far in THIS conversation, possibly none.

    Per-conversation on purpose: a new chat must start the interview over, both
    so it can be re-run and so a student's new goal is not silently overridden
    by an answer they gave last week.
    """
    from rsm_thrive.models import PlannerSession

    session = PlannerSession.objects.filter(conversation=conversation).first()
    return dict(session.intake) if session else {}


def interview_answers():
    """Every closed-set answer a student can send as a single message.

    Used to recognise a conversation TITLE that is really a button press. The
    title is taken from the student's first message, and on this destination
    that message is usually one tap — so the saved list filled up with rows
    called "17 month", several of them, none of which say what the plan was
    for. See `conversation_title`.
    """
    values = {choice["value"].lower() for choice in TRACK_CHOICES}
    values |= {choice["value"].lower() for choice in WORKLOAD_CHOICES}
    values |= {role["label"].lower() for role in load_careers().values()}
    return values


def conversation_title(answers):
    """What to call a saved plan conversation, or "" if it is too early to say.

    Named after the two answers that identify the plan — the goal it targets
    and the track it runs on — because those are what a student is looking for
    when they come back to a list of them a week later. The track alone is not
    enough: it is the first question, so titling on it would name every
    conversation before any of them had a subject.
    """
    goals = answers.get("goals") or []
    if not goals:
        return ""
    careers = load_careers()
    label = careers.get(goals[0], {}).get("label") or goals[0]
    track = answers.get("track")
    return f"Course plan — {label}" + (f", {track}" if track else "")


def supported_roles():
    """Every role a path can be built for, best-supported first.

    All of them, because a path can be built for all of them. An earlier version
    dropped a role whose tagged-elective count fell below a threshold — but
    measured against the plans that actually get built, that count does not
    predict how targeted the plan is: operations-supply-chain (5 tagged
    electives) lands 3 of 7 elective slots on-goal, and healthcare-analyst (15)
    lands 4 of 7. Excluding the first while offering the second was a
    distinction the data does not support, and it hid a job the courses can
    genuinely serve.

    Where a plan does come out mostly breadth, `targeting_note` says so on the
    plan itself — which is the honest place for it, because it is a fact about
    that plan rather than about the role.

    Ordering is by how many courses each role has boosted for it. That is NOT a
    claim about which jobs graduates most often take: this repo carries no
    placement or outcomes data, and inventing an order and calling it "most
    common" would be a statistic with nothing behind it. Course support is
    measurable and is a real signal of what the programme is built to serve.
    Swap the ordering for placement data the moment there is any.
    """
    careers = load_careers()
    return sorted(careers,
                  key=lambda rid: (-len(careers[rid].get("boost_courses") or {}),
                                   careers[rid]["label"]))


def rating_form_for(step, answers=None):
    """The form a step offers, if it has one.

    Two steps do. Skills asks about five areas at once, where twenty-five flat
    buttons would be a wall; the load question needs a slider per quarter and a
    running total, which no set of buttons can express. Everything else answers
    with words or a quick reply and gets None.

    `answers` is needed because the load form depends on the TRACK -- the two
    tracks have different quarters and different published defaults.
    """
    if not step:
        return None   # the interview is complete; nothing to ask
    if step.get("key") == "quarter_units":
        return quarter_units_form_for((answers or {}).get("track") or "11 month")
    return _rating_form_for(step)


def _rating_form_for(step):
    """A pre-filled 1-5 rating per area, so the skills step needs no typing.

    The one step with no useful quick replies: it asks about five areas at once,
    and a flat row of twenty-five buttons is a wall, not a shortcut. A form
    keeps it one control per area, starts every row at the middle so nothing is
    claimed on the student's behalf, and sends a single message when submitted.

    Returned as data rather than rendered here for the usual reason — the
    backend owns what is asked, the frontend owns how it looks.
    """
    if step["key"] != "skills":
        return None
    missing = set(step["missing"])
    rows = [{"key": f"skill_{area['key']}", "label": area["label"]}
            for area in SKILL_AREAS
            if f"skill_{area['key']}" in missing or not missing]
    if not rows:
        return None
    return {
        "kind": "rating",
        "rows": rows,
        "scale": [{"value": s["value"], "label": s["label"], "help": s["help"]}
                  for s in SKILL_SCALE],
        "default": DEFAULT_SKILL_RATING,
        # camelCase ON PURPOSE, and the one place Python emits it. This dict is
        # stored verbatim in ChatMessage.form and served straight through the
        # serializer, so the key IS the wire name the frontend reads. Renaming
        # it to snake_case "for consistency" would silently unlabel the button.
        "submitLabel": "Submit ratings",
    }


def compose_rating_message(ratings):
    """The message a submitted rating form sends, as a student would phrase it.

    NO PRODUCTION CALLER TODAY: the browser composes this same sentence in
    ChatWindow.svelte, because the form is submitted client-side. The two
    formats are character-identical and this one is the tested copy — keep
    them in step, and prefer calling this from any future server-side path.

    Words, not a payload: the transcript then reads like something a person
    said, and the extractor sees the same kind of input whether the student
    used the form or typed "python 4, sql 2".
    """
    labels = {f"skill_{area['key']}": area["label"] for area in SKILL_AREAS}
    parts = [f"{labels.get(key, key)} {value}"
             for key, value in ratings.items() if key in labels]
    return ", ".join(parts)


def _choice_button(option):
    """One closed-set option as a button, label split from its explanation.

    The published labels are written as "11 month — Summer through Spring":
    a name, an em dash, and what it means. The button face takes the name and
    the `description` carries the rest, so the explanation travels WITH the
    control instead of being repeated in a bullet list above it (see
    `render_question`, which no longer prints one).
    """
    label, _, description = option["label"].partition(" — ")
    return {"label": label, "send": option["value"],
            "description": description or option.get("description", "")}


def quick_replies_for(step):
    """Buttons to offer with a question, so a fixed choice need not be typed.

    Only for questions whose answers are a closed set. The skills step is left
    to free text on purpose: it asks about five areas at once, and thirty
    buttons is not a shortcut.
    """
    careers = load_careers()
    if step["key"] == "track":
        return [_choice_button(option) for option in TRACK_CHOICES]
    if step["key"] == "goals":
        # `short_label` comes from careers.json rather than being derived. A
        # slash means two different things in these labels — an alias in
        # "Marketing Analyst / Marketing Data Scientist", part of the name in
        # "Finance / Quantitative Analyst" — so splitting on it turned the
        # latter into a button reading "Finance". The button text is data.
        #
        # `send` stays the FULL label so the extractor still sees every alias it
        # might match on; only the button face is shortened.
        return [{"label": careers[role_id].get("short_label")
                          or careers[role_id]["label"],
                 "send": careers[role_id]["label"],
                 "description": careers[role_id].get("description", "")}
                for role_id in supported_roles()]
    if step["key"] == "workload":
        return [_choice_button(option) for option in WORKLOAD_CHOICES]
    return []


# THREE DOORS, not one. This used to open on "what are you aiming for after
# the programme?", which assumes an answer exists -- and for a large share of
# students in their first quarter it does not. Being asked a question you
# cannot answer is a worse start than being shown the way in, so the two
# paths that do not need a job title are named here rather than left to be
# discovered by a student who happens to type "I don't know".
#
# The old second half of door one read "name an industry and I'll work from
# what that field is asking for" -- the web lookup's promise, and it outlived
# the lookup. An industry now opens a ranked list of the roles it hires MSBA
# graduates into, which is a better answer and a different sentence.
OPENING = (
    "Hi — I'm the MSBA course planner. I'll lay out your whole plan of "
    "study — we can start from wherever you are:\n\n"
    "- **You know the job.** Name it with your track and we're done in one "
    "line: _\"11 month, data scientist\"_.\n"
    "- **You know the field, not the job.** Say _\"healthcare\"_, "
    "_\"fintech\"_, _\"consulting\"_ — I'll show you the roles it hires MSBA "
    "graduates into, most in demand first, and you pick from there.\n"
    "- **You have no idea yet.** Say so. I'll walk you through the six "
    "industries, then the roles inside whichever one interests you.\n\n"
    # THE TRACK IS NOT ASKED HERE. It used to be, as a standing requirement
    # ("I'll need your track either way"), which made the opening a form with
    # two fields before the student had said anything at all -- and it asked
    # twice, because every curated recommendation already closes by asking for
    # it, at the point where it is the only thing standing between the student
    # and a plan. A question asked before it is needed is a question asked
    # while the answer still costs the student something to give.
    #
    # The one-line shortcut still SHOWS a track, so the student who has both
    # answers ready can give them together and skip a turn.
    "And if something else is on your mind first — whether a course has "
    "prerequisites, what's left in your degree, what to take next quarter — "
    "just ask. I'll answer that and we can come back to this."
)


def opening_prompt():
    """What the courses surface shows before anything has been said.

    Two questions, asked warmly, with the escape hatch stated in the same
    breath. Deliberately not the interview's first step with its buttons, and
    deliberately not the bare list of examples it was before -- a surface that
    opens by describing its own capabilities makes the student compose the
    first move. Opening ON the two questions a plan cannot be built without
    means the common case is one line ("11 month, data scientist") and the
    conversation has already started.

    The four-step interview used to open here: "Which track are you on?" with
    two buttons, then a goal, then five skill sliders, then a unit split. It
    worked, and it made the bot feel like a form — a student who typed a real
    sentence was pushed back into the script, and three of the four questions
    were ones their sentence had already answered or did not need. What opens
    the surface now says what it can do and gets out of the way; everything
    it used to interrogate for is either extracted from what the student
    writes, taken from their record, or asked for once at the only point where
    proceeding would mean guessing.

    Still served from here rather than written into the frontend, for the
    original reason: the opening a student sees and the answers they get come
    from the same place and cannot drift apart.
    """
    return {"body": OPENING, "quickReplies": [], "form": None}


def load_session_review(conversation):
    """Where this conversation is in the walk-through, or None."""
    from rsm_thrive.models import PlannerSession

    session = PlannerSession.objects.filter(conversation=conversation).first()
    return session.review if session and session.review else None


def load_session_situation(conversation):
    """What this conversation has established about where the student is."""
    from rsm_thrive.models import PlannerSession

    session = PlannerSession.objects.filter(conversation=conversation).first()
    return dict(session.situation or {}) if session else {}


def clear_session_review(conversation):
    """Forget where the walk-through had got to.

    Called when the TRACK changes. The walk-through's position is an index into
    that track's sequence of quarters, and the two tracks do not have the same
    number of them -- so a student who had walked to the 17-month track's fifth
    quarter and then said "11 month" was left pointing at a quarter that no
    longer exists. `_review_reply` clamps before it renders, so nothing visibly
    broke; the stored position simply stopped describing anything, and every
    later read of it had to be defensive about a value that should never have
    been storable.
    """
    from rsm_thrive.models import PlannerSession

    PlannerSession.objects.filter(conversation=conversation).update(review=None)


def session_has_asked(conversation, key):
    from rsm_thrive.models import PlannerSession

    session = PlannerSession.objects.filter(conversation=conversation).first()
    return bool(session and key in (session.asked or []))


def note_session_asked(conversation, key):
    from rsm_thrive.models import PlannerSession

    session, _created = PlannerSession.objects.get_or_create(
        conversation=conversation, defaults={"intake": {}})
    asked = list(session.asked or [])
    if key not in asked:
        asked.append(key)
        session.asked = asked
        session.save(update_fields=["asked", "updated_at"])
    return session


def save_session_situation(conversation, position):
    from rsm_thrive.models import PlannerSession

    session, _created = PlannerSession.objects.get_or_create(
        conversation=conversation, defaults={"intake": {}})
    session.situation = position or None
    session.save(update_fields=["situation", "updated_at"])
    return session


def save_session_intake(conversation, answers):
    """Persist this conversation's in-progress interview.

    Accumulating rather than re-deriving keeps the interview steady if one
    extraction call comes back thin: a question already answered in this chat
    stays answered.
    """
    from rsm_thrive.models import PlannerSession

    session, created = PlannerSession.objects.get_or_create(
        conversation=conversation, defaults={"intake": answers})
    if not created:
        session.intake = answers
        session.save(update_fields=["intake", "updated_at"])
    return session


def save_intake(user, answers):
    """Persist a completed interview so the chat and the /plan API agree.

    Without this the two surfaces would each hold their own idea of the
    student's answers, and a swap made through one would be invisible to the
    other. Existing swaps are cleared because they were chosen against a plan
    built from different answers.
    """
    from rsm_thrive.models import CoursePlan

    record, created = CoursePlan.objects.get_or_create(
        user=user,
        defaults={"track": answers["track"], "intake": answers, "selections": {}},
    )
    if not created and record.intake != answers:
        record.track = answers["track"]
        record.intake = answers
        record.selections = {}
        record.save(update_fields=["track", "intake", "selections", "updated_at"])
    return record


def role_vocabulary():
    """Each role id with the job titles that mean it, for the extractor.

    The prompt tells the model to map a goal ONLY when the student named that
    job or an exact synonym of it. Handing it bare ids made that instruction
    bite: a student who said "product manager", "data engineer" or "financial
    analyst" -- all jobs this catalog serves -- was told their career was not
    covered, because none of those strings is a synonym of "Product Analyst /
    Product Data Scientist", "Analytics Engineer" or "Financial Analytics".

    The titles come from the taxonomy itself (`titles` on each profile), so the
    vocabulary a student is understood in and the vocabulary the profile is
    defined by cannot drift apart. Retired role names live in the same list,
    which is what makes the taxonomy change invisible to someone still using
    the old job title. `resolve_role` handles the same problem for stored ids;
    this is its free-text half.
    """
    careers = load_careers()
    lines = []
    for role_id in sorted(careers):
        titles = careers[role_id].get("titles") or []
        named = "; ".join(dict.fromkeys([careers[role_id]["label"], *titles]))
        lines.append(f"{role_id} (said as: {named})")
    return "\n".join(lines)


def intake_extract_placeholders():
    """The vocabularies the extractor prompt has to be told about."""
    allowed = _allowed_values()
    return {
        "tracks": ", ".join(sorted(allowed["track"])),
        "role_ids": role_vocabulary(),
        "levels": ("an integer 1-5, where "
                   + ", ".join(f"{lvl['value']} means {lvl['help'].lower()}"
                               for lvl in SKILL_SCALE)),
        "workloads": ", ".join(sorted(allowed["workload"])),
        "interest_tags": ", ".join(sorted(allowed["interests"])),
        "skill_keys": ", ".join(f"skill_{area['key']}" for area in SKILL_AREAS),
        "quarter_keys": ", ".join(sorted({
            quarter["key"] for skeleton in TRACK_SKELETONS.values()
            for quarter in skeleton})),
    }


# ---------------------------------------------------------------------------
# Talking about a plan that already exists
# ---------------------------------------------------------------------------

# "MGTA 451", "CSE 251A", "MGT 477", "MGTF 405" — the shapes the catalog uses.
COURSE_CODE = re.compile(r"\b([A-Z]{2,4})\s*(\d{3}[A-Z]?)\b", re.IGNORECASE)


def mentioned_codes(text):
    """Course codes named in a message, normalised to catalog spelling.

    Two shapes beyond the fully spelled code, because students write both:

    * **A bare number after a coded course inherits its department.** "MGTA
      464 and 402" names two courses; reading it as one dropped the second
      and built a plan that scheduled a course the student said they had
      finished.
    * **A bare number that is unambiguous in the catalog stands alone.**
      "can i swap 466 for something easier" names MGTA 466, because no other
      department has a 466. A number two departments share stays unread
      rather than guessed.
    """
    known = {course["code"].upper() for course in load_catalog()}
    by_number = collections.defaultdict(set)
    for code in known:
        department, _, number = code.partition(" ")
        by_number[number].add(department)
    found, last_department = [], None
    for token in re.finditer(r"\b(MGTA|MGTF|MGTP|MGT|CSE)\s*(\d{3}[A-Z]?)\b"
                             r"|\b(\d{3}[A-Z]?)\b", text or "", re.IGNORECASE):
        department, number, bare = token.groups()
        if department:
            last_department, number = department.upper(), number.upper()
        else:
            number = bare.upper()
            if last_department and f"{last_department} {number}" in known:
                department = last_department
            elif len(by_number.get(number, ())) == 1:
                department = next(iter(by_number[number]))
            else:
                continue
        code = f"{department.upper()} {number}"
        if code in known and code not in found:
            found.append(code)
    return found


def locate_code(plan, code):
    """Where a course sits in a plan: (quarter_key, slot, row).

    Matches the displayed code first and the bare catalog code second, so both
    "MGTA 495 (GENAI)" and "MGTA 495" find something. Returns
    (None, None, None) when nothing matches.
    """
    wanted = code.upper()
    for field in ("code", "baseCode"):
        for quarter in plan["quarters"]:
            for index, row in enumerate(quarter["courses"]):
                if (row.get(field) or "").upper() == wanted:
                    return quarter["key"], index, row
    return None, None, None


def rows_matching_code(plan, code):
    """Every plan row a bare code could mean — more than one when ambiguous."""
    wanted = code.upper()
    found = []
    for quarter in plan["quarters"]:
        for index, row in enumerate(quarter["courses"]):
            if wanted in {(row.get("code") or "").upper(),
                          (row.get("baseCode") or "").upper()}:
                found.append((quarter["key"], index, row))
    return found


def render_alternatives_markdown(result):
    """Alternatives as a table, with what each shares and what it costs."""
    current = result["current"]
    if not result["swappable"]:
        return (f"**{current['code']} — {current['title']}** cannot be changed: "
                f"{result['why']}.")
    if not result["options"]:
        return (f"There is no alternative for **{current['code']}**. "
                f"{result['why'].capitalize()}.")
    lines = [
        f"Other ways to fill that slot instead of **{current['code']} — "
        f"{current['title']}**:", "",
        "| Course | Title | Teaches some of the same | You would give up |",
        "|---|---|---|---|",
    ]
    for option in result["options"]:
        same = ", ".join(option["sharedSkills"][:2]) or \
            (", ".join(option["sharedFocus"]) or "a different angle on the same goal")
        lost = ", ".join(option["losesFromCurrent"][:2]) or "nothing important"
        lines.append(f"| **{option['code']}** | {option['title']} | {same} | {lost} |")
    lines += ["", "Reply with the code you want (for example "
              f"\"swap {current['code']} for {result['options'][0]['code']}\") "
              "and I'll update the plan."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Where else a course could go
# ---------------------------------------------------------------------------
#
# Two moments in a conversation turn on the same fact -- the seasons a course
# is offered in, read against the quarters of THIS plan:
#
# * A student swapping a course OUT wants to know whether that is goodbye. A
#   course offered only in Fall, dropped from Fall, is out of the plan for good;
#   one that also runs in Winter can be picked up later. The plan cannot say
#   which without being asked, so the swap reply says it unprompted.
# * A student naming a course they WANT wants to know where it fits. "Offered
#   in Winter and Spring" is the catalog's answer; "your Winter, in place of
#   MGTA 457 or MGTA 462" is the plan's, and it is the second one they can act
#   on.
#
# Both read `offerings` by SEASON only, exactly as `_offered_in` does when it
# builds the plan, so the note never contradicts the schedule it sits under.

SEASON_NAMES = {"SU": "Summer", "FA": "Fall", "WI": "Winter", "SP": "Spring"}
_SEASON_ORDER = ("SU", "FA", "WI", "SP")


def seasons_of(course):
    """The seasons a course runs in, in academic order, as codes."""
    have = {(o.get("season") or "").upper() for o in course.get("offerings") or []}
    return [season for season in _SEASON_ORDER if season in have]


def season_names(seasons):
    """"Fall, Winter and Spring" -- season codes said as a list."""
    names = [SEASON_NAMES.get(season, season) for season in seasons]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _term_varies(course, seasons):
    """Does the catalog only know these seasons as "varies"?

    A course listed for a season with a real term (FA26) is scheduled; one
    listed as "varies" is a pattern, not a promise. The note hedges the
    second, because "you could take it in Winter" is exactly the sentence a
    student plans around.
    """
    terms = [(o.get("term") or "").lower()
             for o in course.get("offerings") or []
             if (o.get("season") or "").upper() in set(seasons)]
    return bool(terms) and all(term == "varies" for term in terms)


def placements_for(plan, course, exclude_key=None):
    """The quarters of `plan` that could take `course`, and what it would displace.

    A quarter qualifies when the course is offered in its season AND it holds
    a swappable elective of the same unit size -- the same two tests
    `apply_swap` applies, so every placement named here is one it would accept.
    Returns [(quarter, [rows])] in plan order, skipping `exclude_key`.
    """
    placements = []
    for quarter in plan["quarters"]:
        if quarter["key"] == exclude_key:
            continue
        if not _offered_in(course, quarter["season"]):
            continue
        rows = [row for row in quarter["courses"]
                if row["swappable"] and row["courseId"]
                and row["courseId"] != course["id"]
                and row["units"] == course["units"]]
        if rows:
            placements.append((quarter, rows))
    return placements


def _in_place_of(rows):
    codes = [f"**{row['code']}**" for row in rows]
    if len(codes) == 1:
        return f"in place of {codes[0]}"
    return f"in place of {', '.join(codes[:-1])} or {codes[-1]}"


def _quarter_index(plan, key):
    return next((i for i, q in enumerate(plan["quarters"]) if q["key"] == key), None)


def displaced_note(plan, course, quarter_key):
    """What swapping `course` OUT of `quarter_key` means for it -- as a callout.

    Three honest outcomes, and the reply should say which:

    * It fits another quarter of the plan: name the quarters, say whether each
      is later or earlier than the one it is leaving, and give the swap that
      would put it there.
    * It runs in another season but nothing there can take it (no free slot of
      its size on this load): say so, so "also offered in Winter" is not read
      as "you can still have it".
    * It runs only in this season: say plainly that there is no other quarter
      for it. That is the sentence a student needs BEFORE they confirm the
      swap, not after.
    """
    here = next((q for q in plan["quarters"] if q["key"] == quarter_key), None)
    if here is None:
        return ""
    code = display_code(course)
    seasons = seasons_of(course)
    elsewhere = [season for season in seasons if season != here["season"]]
    if not elsewhere:
        if not seasons:
            return (f"\n\n> **{code}** has no listed offering in the catalog, so "
                    f"I can't say where else it could go — check with MSBA "
                    f"advising before you count on taking it later.")
        return (f"\n\n> **{code}** is only offered in "
                f"**{SEASON_NAMES.get(here['season'], here['season'])}**, so "
                f"there's no other quarter to take it in — with this swap it's "
                f"out of your plan.")

    placements = placements_for(plan, course, exclude_key=quarter_key)
    hedge = (" The exact term varies year to year, so confirm it on the "
             "Schedule of Classes before you count on it."
             if _term_varies(course, elsewhere) else "")
    if not placements:
        where = ("that quarter has" if len(elsewhere) == 1
                 else "none of those quarters has")
        return (f"\n\n> **{code}** also runs in {season_names(elsewhere)}, but "
                f"{where} no free {course['units']}-unit elective slot on your "
                f"current load, so it's out of your plan for now.{hedge}")

    current = _quarter_index(plan, quarter_key)
    parts = []
    for quarter, rows in placements:
        when = ("later" if _quarter_index(plan, quarter["key"]) > current
                else "earlier")
        parts.append(f"**{quarter['label']}** ({when}, {_in_place_of(rows)})")
    listed = (parts[0] if len(parts) == 1
              else ", ".join(parts[:-1]) + " or " + parts[-1])
    first_quarter, first_rows = placements[0]
    return (f"\n\n> **{code}** also runs in {season_names(elsewhere)}, so you "
            f"can still take it in {listed}. Say **swap {first_rows[0]['code']} "
            f"for {code}** and I'll move it there.{hedge}")


def placement_note(plan, course, taken_ids=frozenset(), *, missing_from=None):
    """Where a course the student WANTS could go in this plan -- as a callout.

    `missing_from` is the quarter they tried to put it in, when there was
    one; the note then opens by saying it is not offered there before saying
    where it is. Without it the note answers "I want MGTA 456" on its own.
    """
    code = display_code(course)
    seasons = seasons_of(course)
    opener = ""
    if missing_from is not None:
        opener = (f"**{code}** isn't offered in "
                  f"**{SEASON_NAMES.get(missing_from['season'], missing_from['season'])}**. ")
    if course["id"] in taken_ids:
        return f"\n\n> {opener}You've already taken **{code}**, so it can't go back in."
    if course["is_core"]:
        return (f"\n\n> {opener}**{code}** is a core course — every student takes "
                f"it, and it isn't something to swap into an elective slot.")
    if not seasons:
        return (f"\n\n> {opener}The catalog doesn't list when **{code}** is "
                f"offered, so I can't place it — check with MSBA advising.")

    placements = placements_for(plan, course)
    hedge = (" The exact term varies year to year, so confirm it on the "
             "Schedule of Classes before you count on it."
             if _term_varies(course, seasons) else "")
    offered = f"**{code}** is offered in {season_names(seasons)}."
    if not placements:
        in_plan = [q for q in plan["quarters"] if _offered_in(course, q["season"])]
        if not in_plan:
            return (f"\n\n> {opener}{offered} None of those falls in a quarter "
                    f"of your plan, so there's nowhere to put it.{hedge}")
        return (f"\n\n> {opener}{offered} In your plan that's "
                f"{season_names([q['season'] for q in in_plan])}, but there's no "
                f"free {course['units']}-unit elective slot in "
                f"{'that quarter' if len(in_plan) == 1 else 'those quarters'} on "
                f"your current load. Changing the load on one of them may open "
                f"one.{hedge}")

    parts = [f"**{quarter['label']}** ({_in_place_of(rows)})"
             for quarter, rows in placements]
    listed = (parts[0] if len(parts) == 1
              else ", ".join(parts[:-1]) + " or " + parts[-1])
    first_quarter, first_rows = placements[0]
    return (f"\n\n> {opener}{offered} In your plan it fits in {listed}. Say "
            f"**swap {first_rows[0]['code']} for {code}** and I'll make the "
            f"change.{hedge}")


# What a student says when they want a course IN, or want to know when it runs,
# as opposed to asking a fact about it ("does 456 have prerequisites") that the
# catalog route answers better.
_WANTS_COURSE = re.compile(
    r"\b(?:want|like|love|prefer|take|taking|add|put|fit|squeeze|include|get|"
    r"try|switch|move|instead|interested|when|which quarter|what quarter|"
    r"which term|what term|offered|offer|offers|offering|available|run|runs|"
    r"running|schedule|scheduled|enrol|enroll|register|sign up)\b",
    re.IGNORECASE)


def wants_course(text):
    """Is this turn asking to have a course, or asking when it runs?

    True for "I want MGTA 456", "can I take 456?", "when is 456 offered" and a
    bare "MGTA 456"; False for "does 456 have prerequisites", which names a
    course but asks the catalog something this note cannot answer.
    """
    said = (text or "").strip()
    if not said:
        return False
    if _WANTS_COURSE.search(said):
        return True
    # Nothing but the code(s): the shortest possible way to ask about one.
    return COURSE_CODE.sub("", said).strip(" ?.!,;:-") == ""


# ---------------------------------------------------------------------------
# Walking the plan, one quarter at a time
# ---------------------------------------------------------------------------

# Alternatives shown per slot during the review. Three is enough to make a real
# choice and few enough to read; the full list is always a question away.
REVIEW_ALTERNATIVES = 3


def _course_line(row, course):
    """One list row: the course, what it gives you, and whose programme it is.

    The brief comes from the catalog's own `skills`, so it costs nothing and
    cannot drift from what the course record says.

    A course from outside the MSBA says so ON THE ROW. Every one of these is
    pre-approved and none of them is a mistake, but they are not the
    programme's own courses: enrolment is by consent, and a student reading a
    plan should not have to recognise the prefix to know that MGTF 405 will
    need somebody's permission and MGTA 464 will not.
    """
    line = f"- **{row['code']}** {row['title']} ({row['units']} units)"
    if course:
        teaches = _teaches(course)
        if teaches:
            line += f" — {teaches}"
        if not is_msba(course):
            label = DEPARTMENT_LABELS.get(
                course.get("department"), ("outside the MSBA",))[0]
            line += f"\n  _{label} — outside the MSBA, enrolment by consent._"
    return line


def _teaches(course, limit=2):
    """A short line on what a course actually gives you."""
    skills = (course.get("skills") or [])[:limit]
    if skills:
        return "; ".join(skills)
    return (course.get("description") or "").split(".")[0]


LOAD_FIXED_LINE = (
    "_On the {track} track each quarter's load is what the plan of study "
    "publishes — the programme is compressed into four quarters, so there is "
    "no light or heavy version. You can still swap any elective._")


def review_quarter(plan, answers, index, taken_ids=frozenset()):
    """One quarter of the plan, with the alternatives for each elective slot.

    The plan arrives as a finished thing, which is the right way to arrive but
    the wrong way to be checked: a student handed twelve courses at once has no
    natural place to start disagreeing. Walking it a quarter at a time turns it
    into a sequence of small, answerable questions — here is what you are taking
    in Fall, here is why, here is what else would fit that slot and what each of
    those teaches instead.

    Returns (markdown, quick_replies, is_last).
    """
    quarters = plan["quarters"]
    index = max(0, min(index, len(quarters) - 1))
    quarter = quarters[index]
    catalog = _catalog_by_id()

    lines = [f"# {quarter['label']} — {quarter['unitsPlanned']} units",
             f"_Quarter {index + 1} of {len(quarters)}._", ""]

    done = [row for row in quarter["courses"] if row.get("completed")]
    if done:
        lines += ["**Already done — you told me, so these aren't counted again:**", ""]
        lines += [f"- **{row['code']}** {row['title']} ({row['units']} units)"
                  for row in done]
        lines.append("")
    # "Core" AND "Required". Summer's MGTA 403 and 464 are stamped "Required"
    # -- mandatory, but counting toward elective units -- and this list read
    # only "Core", so the walk-through had never shown them: "Summer III -- 8
    # units" followed by a single 4-unit course, every time.
    core = [row for row in quarter["courses"]
            if row["requirement"] in ("Core", "Required") and not row.get("completed")]
    if core:
        lines += ["**Required this quarter — these are fixed:**", ""]
        # A brief on the required courses too, not just the swappable ones.
        # A student walking their plan is deciding whether it is right, and
        # "MGTA 451 Business Analytics in Marketing, Finance & Operations"
        # tells them nothing they could not read off the code. They cannot
        # change these, but they can prepare for them.
        for row in core:
            lines.append(_course_line(row, catalog.get(row["courseId"])))
        lines.append("")

    replies = []
    swappable = [(slot, row) for slot, row in enumerate(quarter["courses"])
                 if row["swappable"] and row["courseId"]]
    fixed_electives = [row for row in quarter["courses"]
                       if row["requirement"] == "Elective" and not row["swappable"]]
    if fixed_electives:
        lines += ["**Scheduled electives — no alternative is offered this quarter:**",
                  ""]
        for row in fixed_electives:
            lines.append(_course_line(row, catalog.get(row["courseId"])))
        lines.append("")

    for slot, row in swappable:
        course = catalog.get(row["courseId"])
        lines += [f"## Your {row['units']}-unit elective: {row['code']} — "
                  f"{row['title']}", ""]
        if row.get("reasons"):
            # "Why this one:" rather than "Recommended because it ...". The
            # reasons are a mix of verb phrases ("builds your data-engineering
            # focus") and noun phrases ("directly relevant for Data
            # Scientist"), and the old template only agreed with the first
            # kind -- it rendered "Recommended because it directly relevant
            # for Data Scientist".
            lines.append(f"**Why this one:** {row['reasons'][0]}.")
        lines.append(f"**Teaches:** {_teaches(course)}." if course else "")
        # The PLACED course says whose programme it is, not only the
        # alternatives under it. The swap list carried the flag and this
        # heading did not, which is backwards: the alternatives are courses
        # the student might take, and this is the one they will.
        if course and not is_msba(course):
            label = DEPARTMENT_LABELS.get(course.get("department"),
                                          ("outside the MSBA",))[0]
            lines.append(f"> **{label}** — outside the MSBA's own courses, so "
                         f"enrolment is by consent. Confirm it with MSBA "
                         f"advising before you count on it.")
        for caution in row.get("cautions") or []:
            lines.append(f"> Heads up: it {caution}.")
        for stretch in row.get("stretch") or []:
            lines.append(f"> Heads up: {stretch}.")
        lines.append("")

        options = alternatives_for(plan, answers, quarter["key"], slot,
                                   taken_ids, REVIEW_ALTERNATIVES)["options"]
        if options:
            lines += ["Other courses that fit this slot:", ""]
            for option in options:
                alt = catalog.get(option["courseId"])
                shared = ", ".join(option["sharedSkills"][:1])
                line = (f"- **{option['code']}** {option['title']} — teaches "
                        f"{_teaches(alt, 1)}."
                        + (f" Shares *{shared}* with {row['code']}."
                           if shared else ""))
                # Same flag the placed rows carry. A swap list is where a
                # student is most likely to pick up an outside course without
                # noticing, because it reads as a menu of equals.
                if alt and not is_msba(alt):
                    line += " _Outside the MSBA — enrolment by consent._"
                lines.append(line)
                # The label names the swap, not just the course. A quarter with
                # two elective slots can offer the SAME alternative for both —
                # Spring offered "Take MGT 451" twice — and two identical
                # buttons doing different things is a coin toss, not a choice.
                replies.append({
                    "label": f"{option['code']} instead of {row['code']}",
                    "send": f"swap {row['code']} for {option['code']}"})
            lines.append("")
        else:
            lines += ["Nothing else is offered at this size in this quarter, so "
                      "this slot is effectively fixed.", ""]

    # How heavy this quarter should be, asked here rather than up front.
    #
    # It used to be one form at the end of the interview: five sliders, one per
    # quarter, answered before the student had seen a single course. Asked
    # DURING the walk-through it is a different question, because they are
    # looking at the four courses it would change -- and the answer re-cuts
    # this quarter's elective slots, so the recommendation moves with it.
    #
    # Only the loads that can actually be scheduled are offered: the degree is
    # 50 units and Summer is fixed, so a lighter quarter is a heavier one
    # somewhere else, and where that is impossible the option is not shown.
    # See `load_options_for`.
    track = (answers or {}).get("track") or "11 month"
    chosen = (answers or {}).get("quarter_units") or {}
    decided = set((answers or {}).get("quarter_loads") or {})
    options = load_options_for(track, quarter["key"], chosen, decided)
    first_open = next((q["key"] for q in adjustable_quarters(track)), None)
    if not load_is_a_choice(track):
        # Said once, on the first quarter with elective room, so a student
        # who expected the load question knows why it did not come.
        if quarter["key"] == first_open:
            lines += ["", LOAD_FIXED_LINE.format(track=track)]
    elif options:
        now = load_of_quarter(track, quarter["key"], chosen, decided)
        settled = quarter["key"] in decided
        # Once they have answered for this quarter, the same words become a
        # statement of where it stands plus the ways to change it. Re-asking a
        # question they just answered reads as not having heard them.
        if settled:
            lines += ["", f"**{quarter['label']} is {now} — "
                          f"{quarter['unitsPlanned']} units**, as you asked."]
            rest = [o for o in options if o["value"] != now]
            if rest:
                lines.append(
                    "Say " + " or ".join(f"**{o['value']}**" for o in rest)
                    + " to change it ("
                    + ", ".join(f"{o['units']} units" for o in rest) + ").")
        else:
            lines += ["", f"**How heavy should {quarter['label']} be?** It is "
                          f"{quarter['unitsPlanned']} units as it stands"
                          + (f" — {now}." if now else ".")]
            for option in options:
                lines.append(f"- **{option['value']}** — {option['units']} units")
            lines += ["", "_Changing it moves units into or out of the other "
                          "quarters; the degree is 50 either way._"]
        replies += [{"label": option["label"], "send": option["value"]}
                    for option in options if option["value"] != now or not settled]
    elif units_for_load(track, quarter["key"], "moderate", chosen, decided):
        # Adjustable in principle, and nothing left to offer: the loads already
        # chosen for the other quarters have fixed this one. Saying so beats
        # showing a quarter with no question under it and no reason why.
        earlier = ", ".join(f"**{quarter_label(track, key)}**" for key in decided
                            if key != quarter["key"])
        lines += ["", f"**{quarter['label']} is fixed at "
                      f"{quarter['unitsPlanned']} units** by the loads you "
                      f"chose for {earlier} — the degree is "
                      f"{TOTAL_UNITS} units, so this is what is left."
                  if earlier else
                  f"**{quarter['label']}** carries "
                  f"{quarter['unitsPlanned']} units and cannot move."]

    is_last = index >= len(quarters) - 1
    replies.append({"label": "Finalise my plan" if is_last else "Next quarter",
                    "send": "finalise" if is_last else "next quarter"})
    return "\n".join(line for line in lines if line is not None), replies, is_last


# The exact strings the review buttons send, plus the handful of ways a student
# types the same thing. Whole-message matches only -- see `review_intent`.
FINALISE_WORDS = frozenset({
    "finalise", "finalize", "looks good", "looks good finalise it", "that works",
    "i'm happy", "im happy", "confirm", "confirmed", "done", "perfect",
    "thats it", "that's it", "keep it", "yes", "yep", "sounds good",
})
NEXT_WORDS = frozenset({
    "next", "next quarter", "next one", "continue", "keep going", "go on",
    "carry on", "onwards", "then", "and then",
})


def review_intent(text):
    """What a message is asking of the review, if anything.

    Deterministic word matching rather than another model call: these arrive
    from buttons, and the two or three ways a student types them by hand are
    easy to list. An LLM here would add a round trip and a failure mode to a
    decision that has three outcomes.
    """
    lowered = re.sub(r"[^a-z0-9' ]+", "", (text or "").strip().lower()).strip()
    if not lowered:
        return None
    # WHOLE-message matching for the one-word intents, because these arrive
    # from buttons that send a fixed string. Substring matching read them out
    # of ordinary questions instead: "can you confirm what MGTA 458 is?"
    # finalised the plan, "what should I review?" restarted the walk-through,
    # and "is next quarter heavy?" advanced it. A student asking about the plan
    # was changing it.
    if lowered in FINALISE_WORDS:
        return "finalise"
    if lowered in NEXT_WORDS:
        return "next"
    # These stay substrings: they are several words long and unambiguous, so
    # "can you walk me through it?" is still a request to walk through it.
    if any(phrase in lowered for phrase in
           ("walk me through", "go through it", "one at a time",
            "quarter by quarter", "step through")):
        return "start"
    return None


# What a student types or presses to move between routes. Whole-message
# matching, like `review_intent`, because these arrive from buttons.
ROUTE_WORDS = {
    "use the recommended bundle": "fixed",
    "use the recommended electives": "fixed",
    "recommended bundle": "fixed",
    "use the bundle": "fixed",
    "build it around me": "custom",
    "build one around me": "custom",
    "tailor it to me": "custom",
    "custom": "custom",
}


def route_intent(text):
    """"fixed", "custom", or None -- a request to switch how electives are filled."""
    lowered = re.sub(r"[^a-z0-9' ]+", "", (text or "").strip().lower()).strip()
    return ROUTE_WORDS.get(lowered)


def route_switch_reply(answers):
    """The button offering the OTHER route, or nothing when there is no other.

    Only shown when a bundle actually exists for the student's goal -- offering
    "use the recommended bundle" to someone whose profile has none would be a
    button that cannot work.
    """
    from rsm_thrive.services.courses import bundles

    goals = answers.get("goals") or []
    if not goals or not bundles.bundle_for(goals[0]):
        return []
    if route_of(answers) == "fixed":
        return [{"label": "Build it around me", "send": "build it around me",
                 "description": "Fill the electives against your own skills, "
                                "workload and interests instead"}]
    return [{"label": "Use the recommended bundle",
             "send": "use the recommended bundle",
             "description": "The electives this career path is built from — the "
                            "same plan everyone aiming at it gets"}]


def review_intro_replies(answers=None):
    """Offered with a finished plan: walk it, switch route, or take it as it is."""
    return (route_switch_reply(answers or {})
            + [{"label": "Review it quarter by quarter",
                "send": "walk me through it",
                "description": "See each quarter's electives and what else fits"},
               {"label": "Looks good, finalise it", "send": "finalise",
                "description": "Keep the plan exactly as it is"}])


def finalised_markdown(plan):
    """The closing message once the student is happy with the plan."""
    return ("# Your plan is set\n\n"
            + render_plan_markdown(plan)
            + "\n\nNothing here is booked yet — use the links above when "
              "enrolment opens, and come back any time to change a course.")
