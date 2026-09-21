"""The course surface's one dispatcher.

`router.classify` decides WHAT kind of question this is; this decides what to
do about it and produces the reply. Nothing else routes: the regex that used to
live in `grounded_course_advisor`, the interview state machine in
`planner.next_intake_step`, and the chain of `if` branches in
`bots.answer_electives` are all replaced by the table in `answer` below.

## Free text first

There is no interview. A student who types a real sentence gets an answer to
that sentence, not step 1 of 4 — and the one place a question comes back is the
`unclear` route, plus the single question the situational route asks when it
genuinely does not know which quarter someone is standing in. A form that opens
by asking four questions makes a student answer three they did not need to.

The plan-of-study machinery survives untouched. Swapping a course, walking the
plan a quarter at a time and finalising it are all still here — they were never
the interview, they are what a student does with a plan once one exists, and
they are checked BEFORE the router runs so that "next quarter" and "swap 461
for 463" are not classified as though they were fresh questions.

## The rule this module enforces

**No fact about a course, a unit count, a prerequisite or a deadline is ever
stated unless it came from the catalog, the published skeletons or the
retrieved corpus.**

The model interprets and explains; it never asserts a number. Every handler
below holds that line in the same way: the deterministic layer produces the
facts, the prompt is given those facts and told they are the whole of what it
may say, and where a handler has no grounding it REFUSES rather than answering
from the model's own memory. `grounded_course_advisor` already worked this way;
this makes it the property of the whole surface.
"""

import re

from rsm_thrive.services.bot_config import bot_config
from rsm_thrive.services.courses import (bundles, electives, planner, reading,
                                         router, situation, skill_match)
from rsm_thrive.services.courses.electives import (NON_MSBA_UNIT_CAP,
                                                    load_careers, load_catalog)
from rsm_thrive.services.courses.replies import (
    _catalog_context, _course_availability, _explain_course, _finalise_reply,
    _handle_change_request, _plan_reply, _review_reply, _small_talk_reply,
    _wants_the_plan)
from rsm_thrive.services.reply import BotReply, append_sources, build_context
from rsm_thrive.services.resources.retrieval import retrieve


# ---------------------------------------------------------------------------
# What a turn states outright
# ---------------------------------------------------------------------------

# "11 month" / "17-month" / "the eleven month one". A closed set of two, so a
# pattern is the whole implementation and no model call is needed to read it.
# "11 mo" counts. Students abbreviate, and "yo i wanna do data science stuff,
# 11 mo" named a track that was not read -- so the next turn's "moderate" had
# no track to attach to and was met with a request to clarify.
_TRACK_WORDS = {
    "11 month": re.compile(r"\b(?:11|eleven)[\s-]?(?:months?|mos?)\b", re.IGNORECASE),
    "17 month": re.compile(r"\b(?:17|seventeen)[\s-]?(?:months?|mos?)\b", re.IGNORECASE),
}


# A BARE "11" or "17", as the whole message. Every curated recommendation
# closes by asking "which track you're on -- 11-month or 17-month", and the
# shortest honest answer to that is the number. It was not understood:
# measured live, a student was asked that question, replied "17", and got
# "I want to make sure I answer the right question".
#
# Safe as a whole-message test on this surface. Course codes are three digits,
# quarters are named words, and a menu pick is handled before this is reached
# and only ever offers 1-10 -- so a lone 11 or 17 has nothing else it could
# mean. Anything longer still needs the word, so "I did 11 courses" is not a
# track.
_BARE_TRACK = re.compile(
    r"^(?:the\s+)?(11|17|eleven|seventeen)(?:[\s-]?mo)?\.?$", re.IGNORECASE)
_BARE_TRACK_OF = {"11": "11 month", "eleven": "11 month",
                  "17": "17 month", "seventeen": "17 month"}


def stated_track(question):
    """The track this turn names, or "". Deterministic."""
    for track, pattern in _TRACK_WORDS.items():
        if pattern.search(question or ""):
            return track
    bare = _BARE_TRACK.match(" ".join((question or "").strip().split()))
    if bare:
        return _BARE_TRACK_OF[bare.group(1).lower()]
    return ""


_COMPLETION_VERB = re.compile(
    r"\b(already|ive|i've|have|had)\s+(done|taken|took|completed|finished|"
    r"passed)\b|\b(done|taken|took|completed|finished|passed)\s+(?:it|that|"
    r"those|them|these)?\s*(?:already|before|last)\b",
    re.IGNORECASE)


def learned_from(question):
    """Intake values this turn states outright, read without a model call.

    The whole of what the four-step interview used to extract, for the two
    fields a plan cannot be built without -- and both come from closed sets, so
    both are deterministic. The track is one of two strings; the goal is
    matched against `careers.json`'s own curated job titles by
    `router.matched_role`, which is the same lookup that makes the role route
    authoritative.

    Skills and per-quarter load are NOT read here and are never asked for. They
    refine a plan and are not required to build one, so under free-text-first
    they are assumed, stated as assumed, and correctable in a sentence. An
    interview that gates a plan behind five sliders is the thing being removed.
    """
    learned = {}
    track = stated_track(question)
    if track:
        learned["track"] = track
    # "I've already done MGTA 464 and 402." A completion verb beside course
    # codes is a fact about the transcript, and the plan has to honour it:
    # `planner.build_for` treats these exactly like enrolment rows.
    if _COMPLETION_VERB.search(question or ""):
        codes = planner.mentioned_codes(question)
        if codes:
            learned["completed_codes"] = codes
    role_id, _exact = router.role_match(question)
    if role_id:
        learned["goals"] = [role_id]
    learned.update(planner.read_skills(question))
    # Only outside the walk-through. During it, "light" is about the quarter on
    # screen rather than about the courses in general -- see
    # `_apply_quarter_load` -- and `_maintenance` runs first, so a review turn
    # never reaches here.
    #
    # The looser `load_mentioned` is used ONLY when this turn already stated a
    # track or a goal, because then it is plainly an intake turn: "11 month,
    # data scientist, heavy" names all three and the strict whole-message test
    # finds none of them. On any other turn the strict test governs, so a
    # passing remark that a course is heavy states no preference.
    load = (planner.load_mentioned(question) if learned
            else planner.load_intent(question))
    if load:
        learned["workload"] = load
    return learned


# Asked once, before the plan exists, and it is the ONE preference question
# that survived the interview's removal.
#
# It is not the old per-quarter unit form. That asked a student to distribute
# fifty units across four quarters before they had seen a single course, which
# is a spreadsheet question dressed as a conversation. This asks for one word
# and turns it into a starting distribution (`planner.seeded_units`), which the
# walk-through then refines one quarter at a time -- with the student looking
# at the courses each answer changes.
#
# The wording is about WHERE THE WEIGHT SITS rather than about how hard the
# courses are, and that is deliberate. The degree is fixed at 50 units, so
# "light" cannot mean fewer; and on this catalog it cannot mean easier either,
# because the elective list holds 14 moderate courses, 9 heavy and exactly one
# light, so a light preference has nothing lighter to pick. Promising an easier
# plan and then delivering the same twelve courses would be worse than not
# asking.
ASK_FOR_A_LOAD = (
    "Before I lay it out — how do you want the load spread across the "
    "quarters?\n\n"
    "- **light** — start easy; the later quarters carry more\n"
    "- **moderate** — the published plan, evenly spread\n"
    "- **heavy** — front-load it; get the weight done early\n\n"
    "The degree is 50 units either way, so this is about where they sit rather "
    "than how many there are. Say one of those — and I'll ask about each "
    "quarter individually when we walk through the plan, so nothing here is "
    "final.")

# The answer to a question ABOUT the load options, asked instead of picking
# one. Same facts as `ASK_FOR_A_LOAD`, said as an answer.
LOAD_EXPLAINED = (
    "None of them changes how many courses you take — the degree is **50 "
    "units** whichever you pick, so this is only about where the weight "
    "sits.\n\n"
    "- **light** — the early quarters carry less; the later ones carry more\n"
    "- **moderate** — the published plan, spread evenly\n"
    "- **heavy** — the early quarters carry more; you get the weight done "
    "early\n\n"
    "Say one of those and I'll lay the plan out — and you can still change "
    "any single quarter when we walk through it.")

ASSUMED_LOAD_NOTE = (
    "I've spread the units the way the published plan does, since you haven't "
    "said — say _light_ or _heavy_ any time and I'll redo it.\n\n")

LOAD_APPLIED_NOTE = (
    "Spread **{load}**: {spread}. The degree is still {total} units — a "
    "lighter quarter is a heavier one somewhere else.\n\n")

COMPLETED_NOTE = (
    "Leaving out {codes} \u2014 you said you've done them. That's **{units} "
    "units** already banked, so this plan schedules the remaining "
    "**{scheduled}**.\n\n")

TAIL_SHORT_NOTE = (
    "**{label}** carries **{actual} units** rather than the {wanted} a spread "
    "like this one usually ends on. Some of this path's courses are taught in "
    "one term only, and laying them out that way would push an earlier "
    "quarter past 18 units or under 12. The degree still closes at 50.\n\n")

SPREAD_DROPPED_NOTE = (
    "One thing about the spread: a **{load}** distribution cannot be laid "
    "around the courses this path needs — several of them are taught in one "
    "term only. I've kept the courses and used the published spread, because "
    "the courses are what make this plan about your career and the spread is "
    "something you can still change one quarter at a time below.\n\n")

# The 11-month track. See `planner.load_is_a_choice`.
LOAD_FIXED_TRACK_NOTE = (
    "On the **{track}** track I haven't asked about light or heavy: the "
    "programme is compressed into four quarters and each carries what the "
    "plan of study publishes. You can still swap any elective.\n\n")

LOAD_FIXED_TRACK_SAID_NOTE = (
    "You said **{load}** — on the {track} track there is no light or heavy "
    "version: the programme is compressed into four quarters and each carries "
    "what the plan of study publishes. Here it is as published; you can still "
    "swap any elective.\n\n")

LOAD_FIXED_ON_TRACK = (
    "**{label}** carries {units} units, and on the {track} track that isn't a "
    "choice — the programme is compressed into four quarters and each carries "
    "what the plan of study publishes. You can still swap electives in the "
    "quarters that have them.\n\n---\n\n")

LOAD_UNAVAILABLE_NOTE = (
    "You asked for a **{load}** spread, and on the {track} track the quarters "
    "are already close enough to their limits that there is nothing to move — "
    "so this is the published spread. You can still change any single quarter "
    "when we walk through it.\n\n")

# Asked before the load spread, because it changes WHICH courses are picked
# rather than where they sit -- `profile_from_intake` averages the technical
# areas into the comfort level `_pick` and `rank_electives` both read.
#
# It used to be five sliders on a form. Asked in prose it is one line back, and
# partial answers are fine: rating two areas and ignoring three is a real
# answer, and the rest are assumed and said to be assumed.
# Said when a title was matched with a character of slop. Confirmed rather
# than assumed: guessing which of fourteen careers someone meant is not a
# guess to make quietly.
# An exact title that the design document files under a differently named
# profile. Not a guess -- the mapping is curated -- but the student typed one
# name and is about to read another, and the line between them is owed.
MAPPED_TITLE_NOTE = (
    "**{title}** roles are covered here by the **{label}** profile \u2014 "
    "that's the elective set the programme curates for this job. Say the word "
    "if you meant something else.\n\n")

FUZZY_ROLE_NOTE = (
    "Reading **{said}** as **{label}** — say the word if you meant something "
    "else.\n\n")

ASK_FOR_A_GOAL = (
    "Good \u2014 **{track}** it is. What are you aiming for after the "
    "programme?\n\nAny job title works \u2014 \"data scientist\", "
    "\"pricing analyst\", \"product manager\" \u2014 or name an industry "
    "and I'll show you the roles it hires MSBA graduates into. Not sure yet? "
    "Say so and I'll walk you through the options.")


GOAL_SWITCHED = "Switched to **{label}**.\n\n"

ELECTIVES_CHANGED = (
    "That changes the electives: **{gained}** in, **{lost}** out.\n\n")

ELECTIVES_UNCHANGED = (
    "The recommended electives come out the **same** for both — in this "
    "catalog, {before} and {after} draw on the same set. What moves is the "
    "emphasis and the order, not the courses. Here it is again under the new "
    "heading so the plan says what it is for.\n\n")


def _goal_change_note(previous, answers, user):
    """What changed, when the goal did. "" when it did not.

    THE fix for a plan that looked like a repeat. A student who switched from
    data analyst to consultant got a plan with an identical header, an
    identical Summer (it is entirely required) and no mention anywhere of
    which career it was for -- so a rebuild that really had happened read as
    the bot saying the same thing twice.

    Saying the electives are unchanged, where they are, is the honest half of
    this. Two career profiles in a 24-elective catalog can genuinely resolve
    to the same six courses, and pretending otherwise -- or silently
    reprinting -- is worse than naming it.
    """
    before = list((previous or {}).get("goals") or [])
    after = list((answers or {}).get("goals") or [])
    if not after or before == after:
        return ""
    careers = load_careers()
    label = careers.get(after[0], {}).get("label", after[0])
    note = GOAL_SWITCHED.format(label=label)
    if not before:
        return note
    # The student's own enrolments, so the verdict below is about the plan
    # they would actually be shown rather than about a hypothetical one.
    taken = planner.taken_course_ids(user)
    old_plan = planner.build_for({**answers, "goals": before}, taken)
    new_plan = planner.build_for(answers, taken)
    was, now = planner.elective_codes(old_plan), planner.elective_codes(new_plan)
    if was == now:
        return note + ELECTIVES_UNCHANGED.format(
            before=careers.get(before[0], {}).get("label", before[0]),
            after=label)
    gained, lost = sorted(now - was), sorted(was - now)
    if gained or lost:
        return note + ELECTIVES_CHANGED.format(
            gained=", ".join(gained) or "nothing new",
            lost=", ".join(lost) or "nothing")
    return note


UNCURATED_COVERAGE = (
    "\n\n_On **{goal}**: this plan covers about **{share:.0%}** of what that "
    "career asks for. {gap}_")
UNCURATED_GAP = "It has nothing here for {unmet}."
UNCURATED_NO_GAP = "Everything I found it asking for is covered somewhere in it."
UNCURATED_DROPPED = (
    "\n\n_{dropped} matched your career but didn't fit the unit limits — ask "
    "me to swap {one} in and I'll show you what it would displace._")


def _uncurated_coverage_note(answers, plan):
    """How much of an uncurated career the FINISHED plan covers, or "".

    The recommendation already quotes a coverage figure, and that figure is
    about the six courses it recommended. The plan is not those six courses:
    quarters have unit floors and ceilings, and a course that does not fit is
    dropped. So the number a student was given could describe a set they were
    never actually shown -- stated to the digit, and quietly wrong.

    Recomputed here against the courses that survived scheduling. The
    discrimination weights still come from the WHOLE catalog, so the share
    stays comparable with the one quoted earlier; only the pool being measured
    narrows.

    Curated careers are not measured. Their bundle IS the answer to what the
    career needs -- a hand-authored mapping, not a match -- so scoring it
    against scraped requirements would be marking our own advising against a
    web page.
    """
    goal = answers.get("unmatched_goal")
    wanted = answers.get("goal_skills") or []
    if not goal or not wanted or not plan:
        return ""
    placed = {course["courseId"]: course
              for quarter in plan.get("quarters") or []
              for course in quarter.get("courses") or []}
    if not placed:
        return ""
    rows = [course for course in electives.load_catalog()
            if course["id"] in placed]
    if not rows:
        return ""
    fits = skill_match.rank(wanted, catalog=rows, limit=len(rows))
    cover = skill_match.coverage(wanted, fits)
    unmet = cover["unmet"][:4]
    note = UNCURATED_COVERAGE.format(
        goal=goal, share=cover["metShare"],
        gap=(UNCURATED_GAP.format(unmet=", ".join(unmet)) if unmet
             else UNCURATED_NO_GAP))
    # Naming what fell out matters more than the percentage: the student was
    # shown these courses by name one turn ago, and silently dropping one reads
    # as the plan disagreeing with the recommendation for no stated reason.
    dropped = [code for code in (answers.get("goal_courses") or [])
               if code not in placed]
    if dropped:
        shown = [planner.display_code({"code": code}) for code in dropped[:3]]
        note += UNCURATED_DROPPED.format(
            dropped=", ".join(f"**{code}**" for code in shown),
            one=f"**{shown[0]}**")
    return note


def _plan_now(conversation, answers, previous=None):
    """Build and show the plan.

    `fill_assumed_skills` supplies a low-middle level for every skill area the
    student has not mentioned. Deliberately low: under-claiming keeps courses
    in reach, where over-claiming puts someone in a course they cannot pass.
    Any course that lands above the assumed level is marked on its own row by
    `_stretch_notes`, which is where a student can do something about it.
    """
    # The assumed level is still FILLED -- the scorer needs a number for every
    # area -- but it is no longer announced. It used to be, and that made
    # sense while the interview asked: a student who had answered four of five
    # questions deserved to know the fifth had been guessed.
    #
    # Nothing asks now, so the note fired on every single plan and listed all
    # five areas every time: "I've assumed working knowledge for Python
    # programming, SQL and databases, Statistics and regression, Machine
    # learning, Presenting and storytelling since you haven't said". Forty
    # words of preamble, above the plan, saying only that a question the
    # student was never asked went unanswered.
    #
    # What the assumption actually affects is still disclosed where it can be
    # acted on: `_stretch_notes` marks every course above the assumed level on
    # the course's own row, in the walk-through, next to the course it is
    # about.
    filled, _assumed = planner.fill_assumed_skills(
        answers, [f"skill_{area['key']}" for area in planner.SKILL_AREAS])
    track = filled.get("track") or "11 month"
    load = filled.get("workload")
    lead = ""
    applied = False
    if not planner.load_is_a_choice(track):
        lead += (LOAD_FIXED_TRACK_SAID_NOTE.format(load=load, track=track)
                 if load and load != "moderate"
                 else LOAD_FIXED_TRACK_NOTE.format(track=track))
    elif not load:
        lead += ASSUMED_LOAD_NOTE
    elif not filled.get("quarter_units"):
        # Turn the one-word answer into a real distribution, once. A later
        # per-quarter answer wins, so this never overwrites a decision.
        seeded = planner.seeded_units(track, load)
        if seeded:
            filled = {**filled, "quarter_units": seeded}
            applied = True
        elif load != "moderate":
            lead += LOAD_UNAVAILABLE_NOTE.format(load=load, track=track)
    planner.save_session_intake(conversation, filled)
    # For an uncurated career there is no bundle, so what was matched from the
    # web IS the recommendation: pin it where it fits and let the scorer fill
    # the rest. See `planner.selections_for_courses`.
    pinned = planner.selections_for_courses(
        filled, filled.get("goal_courses") or [],
        planner.taken_course_ids(conversation.user))
    reply = _plan_reply(conversation.user, filled, pinned)
    tail = _uncurated_coverage_note(filled, reply.plan)
    if applied and reply.plan:
        # The spread the plan ACTUALLY has, read off the built plan rather
        # than the seed. The bundle placer may sit a quarter two units off
        # the seed to fit a course taught in one term only, and a lead that
        # said "Winter 18 units" over a table showing 16 was wrong in the
        # one place a student checks first.
        fixed = {q["key"] for q in planner.TRACK_SKELETONS.get(track) or []} - {
            q["key"] for q in planner.adjustable_quarters(track)}
        lead = LOAD_APPLIED_NOTE.format(
            load=load, total=planner.TOTAL_UNITS,
            spread=", ".join(
                f"{q['label']} {q['unitsPlanned']} units"
                for q in reply.plan["quarters"] if q["key"] not in fixed)) + lead
    lead = _goal_change_note(previous, filled, conversation.user) + lead
    if (reply.plan or {}).get("spreadDropped"):
        lead += SPREAD_DROPPED_NOTE.format(load=filled.get("workload") or "custom")
    short = (reply.plan or {}).get("tailShort")
    if short:
        lead += TAIL_SHORT_NOTE.format(**short)
    done = [c for c in (filled.get("completed_codes") or [])
            if planner.stated_completed_ids({"completed_codes": [c]})]
    if done:
        lead += COMPLETED_NOTE.format(
            codes=" and ".join(f"**{c}**" for c in done),
            units=(reply.plan or {}).get("totals", {}).get("completed", 0),
            scheduled=(reply.plan or {}).get("totals", {}).get("scheduled", 0))
    return spoken(BotReply(lead + reply.body + tail, [], "plan",
                           reply.quick_replies, route="plan"))


SHARED_TITLE_NOTE = (
    "**{title}** is listed under both **{chosen}** and {other} \u2014 I've gone "
    "with {chosen}. Say **{say}** if you meant the other.\n\n")


def shared_title_in(question):
    """(title, [role ids]) for a curated title this turn names that belongs
    to more than one profile, or None."""
    owners = {}
    for role_id, role in electives.load_careers().items():
        for title in (role.get("titles") or []):
            owners.setdefault(str(title).lower(), []).append(role_id)
    said = " ".join((question or "").lower().split())
    for title, role_ids in sorted(owners.items(), key=lambda kv: -len(kv[0])):
        if len(role_ids) > 1 and re.search(rf"\b{re.escape(title)}\b", said):
            return title, role_ids
    return None


def _confirmation(question):
    """A line confirming a role read with a character of slop, or "".

    Prefixed onto whatever question comes next, so the confirmation and the
    next step are one turn rather than two.
    """
    role_id, exact = router.role_match(question)
    if not role_id:
        return ""
    # Confirmed for two different readings, both of which put a career in front
    # of a student who did not quite name it: a MISSPELLED title, and an AREA
    # that matched a career off its short label. "Something in supply chain"
    # came back as the full Supply Chain / Operations Analytics set with
    # nothing saying that was a reading.
    shared = shared_title_in(question)
    if shared and role_id in shared[1]:
        # The design document lists this title under MORE THAN ONE profile --
        # "growth analyst" sits under Product and under Marketing -- so an
        # exact match still decided between two readings, and did so silently.
        # A student who meant the other one was handed the wrong bundle with
        # nothing marking it as a choice. Named here, with the way to switch.
        others = [electives.load_careers()[r] for r in shared[1] if r != role_id]
        chosen = electives.load_careers()[role_id]
        return SHARED_TITLE_NOTE.format(
            title=shared[0], chosen=chosen.get("short_label") or chosen["label"],
            other=" or ".join(f"**{o.get('short_label') or o['label']}**"
                              for o in others),
            say=(others[0].get("titles") or [others[0]["label"]])[0])
    role = load_careers().get(role_id) or {}
    label = role.get("label", role_id)
    if exact and router.names_a_job_function(question):
        # An exact title, but not the profile's OWN name: "data engineer" is
        # listed under Analytics Engineer, and the reply opened "Analytics
        # Engineer --" with nothing saying why. That read as an assumption,
        # though it is the design document's mapping. Say the mapping.
        title = router.matched_title(question)
        own = {str(label).lower(), str(role.get("short_label") or "").lower()}
        if title and title not in own and not any(
                title in name or name in title for name in own if name):
            return MAPPED_TITLE_NOTE.format(title=title, label=label)
        return ""
    # QUOTE WHAT THEY NAMED, not the first forty characters of their sentence.
    # Truncating the raw turn produced "Reading **im switching from consulting
    # into analyt** as **Analytics / Data Consultant**" -- a note whose whole
    # job is to show the student their own words back, showing them a
    # sentence cut off mid-syllable instead.
    # The TRACK comes out before the job is quoted. "17 month data nalyst"
    # names both, and quoting it whole reads as though the track were part of
    # the career -- "Reading **17 month data nalyst** as **Business / Data
    # Analyst**". What the student needs confirmed is the half we guessed at.
    without_track = question or ""
    for pattern in _TRACK_WORDS.values():
        without_track = pattern.sub(" ", without_track)
    without_track = _BARE_TRACK.sub(" ", " ".join(without_track.split()))
    said = _student_words(without_track) or _student_words(question)
    if not _looks_like_a_job_title(said):
        # Nothing quotable came out of it, and a confirmation that cannot
        # quote is just noise above an answer.
        return ""
    return FUZZY_ROLE_NOTE.format(said=said, label=label)


def named_only_a_track(question):
    """True when the track is the ONLY thing this turn says.

    The goal question is worth asking for "11 month" and insulting for "11
    month fintech". The test is what survives removing the track phrase and the
    words that wrap any answer -- `_ROLE_FILLER` already holds those, down to
    "industry" and "something", because it was written for exactly this job on
    the role path.

    Deliberately blind to WHAT is left over. Judging that is the router's work
    and it has a model to do it with; all this decides is whether there is
    anything there to judge.
    """
    remainder = question or ""
    for pattern in _TRACK_WORDS.values():
        remainder = pattern.sub(" ", remainder)
    return not [word for word in re.findall(r"[a-z0-9+#/-]+", remainder.lower())
                if word not in _ROLE_FILLER]


def _intake_progress(conversation, question, answers, learned=None):
    """A turn that moved the plan forward, or None.

    Runs before the router and produces no question of its own except one: a
    student who has named a track and nothing else has not said what the plan
    is FOR, and a plan of study built around no goal is just the published
    skeleton. That is the only genuinely blocking gap on this route, so it is
    the only thing asked.

    `learned` overrides the deterministic reading of the turn. It is supplied
    on the second pass, after the `read_turn` skill has read values off a
    turn the rules could not -- "the longer one" for a track, "somewhere in
    between" for a load -- and those values arrive already validated against
    their closed sets (see `router._learned_from_reading`).
    """
    if learned is None:
        learned = learned_from(question)
    if planner.is_question(question):
        # A QUESTION FILLS BLANKS AND OVERWRITES NOTHING -- the same rule
        # `planner.merge_intake` states for the extractor, applied to the
        # deterministic readers, which had no such guard. Measured live: "what
        # if i had gone for the 17 month instead", asked over a finished
        # 11-month plan, moved the student to the 17-month track and rebuilt
        # everything. A what-if mentions a value without choosing it.
        learned = {key: value for key, value in learned.items()
                   if not answers.get(key)}
    # A QUESTION WE JUST ASKED OWNS THE NEXT TURN'S LOAD WORD.
    #
    # `learned_from` reads a load with the strict whole-message test unless
    # the turn also named a track or a goal, which is right in general: a
    # passing remark that a course is heavy states no preference. It is wrong
    # directly after we have asked. Measured live: the bot asked how to spread
    # the load, the student answered "ok moderate load please", and got "I
    # want to make sure I answer the right question" -- for an answer to the
    # question it had just put.
    #
    # So the looser reading applies exactly when we are waiting on it: the
    # workload question has been asked, and no workload is on file yet.
    if (not learned.get("workload") and not answers.get("workload")
            and planner.session_has_asked(conversation, "workload")):
        said = planner.load_mentioned(question)
        if said:
            learned = {**learned, "workload": said}
    merged = {**answers, **learned}
    changed = any(merged.get(key) != answers.get(key) for key in learned)
    if learned.get("track") and answers.get("track") \
            and learned["track"] != answers["track"]:
        # The spread belongs to the OLD track's shape. Quarter keys differ
        # between the two -- only the 17-month track has a second Fall -- so
        # carrying it across loses a quarter's worth of units and the loads the
        # student chose no longer describe the quarters they now have.
        # Re-seeded below from the overall preference, which does still apply.
        merged.pop("quarter_units", None)
        merged.pop("quarter_loads", None)
        # And where the walk-through had got to: that index counts quarters in
        # the OLD track's sequence, and the two tracks have different numbers
        # of them.
        planner.clear_session_review(conversation)
        if merged.get("workload"):
            seeded = planner.seeded_units(learned["track"], merged["workload"])
            if seeded:
                merged["quarter_units"] = seeded
    if learned and changed:
        planner.save_session_intake(conversation, merged)

    if planner.declines_skills(question):
        # Declining IS an answer, and it must be recorded or the question comes
        # back on the next turn.
        planner.note_session_asked(conversation, "skills")
    # `unmatched_goal` counts. A career the catalog curates no id for is still
    # a career the student named -- see `planner.next_intake_step`.
    plannable = bool(merged.get("track")
                     and (merged.get("goals") or merged.get("unmatched_goal")))
    if plannable:
        # Both questions are asked once, and only once. A student who answers
        # with something else gets their plan on the NEXT turn with what was
        # assumed said out loud: an interview with no exit is the thing being
        # removed, so the exit is recorded rather than rediscovered.
        if (not merged.get("workload")
                and not planner.session_has_asked(conversation, "workload")):
            planner.note_session_asked(conversation, "workload")
            # On a track where the load is not a choice the question is
            # still RECORDED as asked -- and skipped. Recording it is what
            # lets a later "light, please" be read as a load and answered
            # ("there is no light version here") rather than sent to the
            # classifier as an unrelated remark.
            if planner.load_is_a_choice(merged["track"]):
                return BotReply(_confirmation(question) + ASK_FOR_A_LOAD, [],
                                "intake", route="plan")
        # Delivered automatically exactly once -- that is the answer to what
        # they asked for. After that it is reprinted only when this turn
        # changed something the plan is built from, so "thanks" and "how do I
        # enrol?" do not each reprint 3,700 characters.
        if not planner.session_has_asked(conversation, "plan"):
            planner.note_session_asked(conversation, "plan")
            return _plan_now(conversation, merged, answers)
        if changed:
            return _plan_now(conversation, merged, answers)
        return None
    if not learned or not changed:
        return None
    if merged.get("track") and "goals" not in learned:
        if named_only_a_track(question):
            return BotReply(ASK_FOR_A_GOAL.format(track=merged["track"]), [],
                            "intake", route="plan")
        # The turn named a track AND something else. Asking here would throw
        # the something else away and put back the question it answers: "11
        # month fintech" was met with "what are you aiming for? ... or name an
        # industry", which the student had just done. Falling through hands the
        # turn to the router, which can place "fintech" -- and the track is
        # already saved above, so nothing is lost either way.
        return None
    # A goal with no track yet: the role route answers it properly, with the
    # curated bundle, and asks for the track at the end of that.
    return None


# ---------------------------------------------------------------------------
# Route: a role the catalog curates
# ---------------------------------------------------------------------------

FIT_LINE = {
    "anchor": "the spine of this path — take these",
    "differentiator": "what separates a strong candidate from a competent one",
    "universal": "everyone takes these, whatever the path",
}


def curated_recommendation(role_id):
    """The curated bundle for a role, as Markdown. No model involved at all.

    THE quality floor. Fourteen roles have a hand-authored mapping in
    `bundles.json` and those stay authoritative — "I want to be a data
    scientist" gets our recommendation, not a model's guess at one. Every
    course named here is looked up in the catalog, so a code that does not
    exist cannot be printed.
    """
    bundle = bundles.bundle_for(role_id)
    if not bundle:
        return None
    role = load_careers().get(role_id) or {}
    catalog = {course["id"]: course for course in load_catalog()}
    lines = [f"**{role.get('label', role_id)}** — {role.get('work', '')}".rstrip(" —"),
             ""]
    if role.get("gate"):
        lines += [f"_{role['gate']}_", ""]
    # A RECOMMENDATION, said as one. It used to read "the elective set the
    # programme recommends for that path -- a curated mapping, not a guess",
    # which claims more standing than this has: the mapping is ours, it is
    # advice, and a student reading it as the programme's official position
    # would be reading it wrongly. What it IS -- deterministic, the same for
    # everyone with this goal, not a model's improvisation -- is a property
    # worth having and not a reason to sound authoritative about it.
    lines.append("Here's what I'd recommend for that path:")
    for layer in ("anchor", "differentiator", "universal"):
        codes = [code for code in (bundle.get(layer) or []) if code in catalog]
        if not codes:
            continue
        lines += ["", f"**{layer.title()}** — {FIT_LINE[layer]}"]
        for code in codes:
            course = catalog[code]
            seasons = sorted({o["season"] for o in course.get("offerings") or []})
            lines.append(
                f"- **{course['code']} — {course['title']}** "
                f"({course['units']} units"
                + (f", offered {', '.join(seasons)}" if seasons else "")
                + ")")
    lines += _enrolment_caveats(
        [code for layer in ("anchor", "differentiator", "universal")
         for code in (bundle.get(layer) or []) if code in catalog])
    lines += ["", planner.RECOMMENDATION_ONLY,
              "", "Tell me which track you're on — **11-month** or **17-month** "
                  "— and I'll lay this out quarter by quarter as a full plan of "
                  "study."]
    return "\n".join(lines)


def _enrolment_caveats(codes):
    """Say, per programme, on what terms a non-MSBA course can be taken.

    Recommending a course a student cannot actually enrol in is a bad failure,
    and it is a silent one: the recommendation looks identical to a good one
    right up until enrolment is refused. The catalog carries CSE, MBA (MGT) and
    MFin (MGTF) entries under `DEPARTMENT_LABELS`, which states the terms —
    consent for MGT/MGTF, and CSE majors having priority so CSE seats go as
    space permits — and that belongs beside the recommendation rather than in a
    data file nobody reads.

    Note that this states the CATALOG'S OWN terms, not a claim that the list is
    authoritative. Whether the CSE entries are the courses MSBA students may
    actually take is an open question with the programme office; until it is
    answered, saying "confirm before you plan around it" is the honest form of
    a recommendation we are not certain we can stand behind.
    """
    from rsm_thrive.services.courses.electives import DEPARTMENT_LABELS

    seen = []
    for code in codes:
        prefix = code.split()[0]
        if prefix != "MGTA" and prefix not in seen:
            seen.append(prefix)
    if not seen:
        return []
    lines = ["", "**Before you plan around these:**"]
    for prefix in seen:
        name, terms = DEPARTMENT_LABELS.get(prefix, (prefix, ""))
        lines.append(f"- {prefix} courses are {name} — {terms}."
                     if terms else f"- {prefix} courses are {name}.")
    lines.append(f"- Up to {NON_MSBA_UNIT_CAP} of the 28 elective units may "
                 f"come from outside the MSBA's own courses. Confirm any "
                 f"non-MGTA course with MSBA advising before you count on it.")
    return lines


UNCOVERED_ROLE_PREFIX = (
    "I don't have a set recommendation ready for **{role}**, so rather than "
    "guess I looked up what that job actually asks for and matched it against "
    "this catalog.\n\n")


def _answer_role(llm, route, conversation, question, history):
    body = curated_recommendation(route.role_id) if route.role_id else None
    if body:
        body = (_confirmation(question) or _reading_note(route)) + body
        # Remember the goal, so a follow-up naming a track builds the right
        # plan without asking the student to say the role again.
        stored = planner.load_session_intake(conversation)
        planner.save_session_intake(conversation, {**stored, "goals": [route.role_id]})
        return BotReply(body, [], "curated", route=router.ROLE,
                        route_confidence=route.confidence)
    if (route.industry_id and not route.role_id
            and not router.names_a_job_function(question)):
        # The model called it a role and could not name one, but DID read
        # the words as one of the six industries. Measured live: "i want to
        # work for a hedge fund" was refused as the career **hedge fund**,
        # when a hedge fund is a place to work and the financial-services
        # roles were the answer. An industry reading beats a refusal.
        #
        # Only for a WORKPLACE, though. "esports analyst" names a job, and a
        # job the fourteen profiles do not cover is refused, not redirected
        # to the catch-all industry's ten titles -- the same line
        # `_uncurated_role` has always held, and the live sweep checks it.
        return _answer_industry(llm, route, conversation, question)
    return _uncurated_role(llm, route, conversation, question)


def _reading_note(route):
    """The confirmation line for a role the MODEL read off the student's words.

    `_confirmation` covers what the router can see for itself -- a misspelled
    or shorthand title. A curated reading by the model (see
    `understanding.curated_reading`) leaves nothing on the turn for the router
    to match, so the note is built from what the route carried instead. Same
    words as the fuzzy note, deliberately: to the student it is the same
    event, a reading of theirs being checked.
    """
    if not route.reading or not route.role_id:
        return ""
    said = _student_words(route.reading) or " ".join(route.reading.split())
    if not _looks_like_a_job_title(said):
        said = " ".join(route.reading.split())[:60]
    if not said:
        return ""
    label = (load_careers().get(route.role_id) or {}).get("label", route.role_id)
    return FUZZY_ROLE_NOTE.format(said=said, label=label)


# Words that are ALWAYS in a question and never distinguish one job from
# another. What is left after these is what the student actually named.
# The words left after these are what the student actually named. The COURSE
# vocabulary in the second block matters as much as the filler in the first:
# without it "which electives suit product analytics" survived whole, and the
# reply read "I don't have a course recommendation for **which electives suit
# product analytics**" -- naming the question back to the student as though it
# were a job title.
_ROLE_FILLER = frozenset("""
a an the i im i'm want wants wanted like would love to be become work working
in into on at for with about something anything some kind sort area field
industry sector space world side of my me you what how do does can could
job role career position work maybe perhaps think thinking interested
it its them they there here part bit stuff thing things doing get getting
that this those these actually just really please yeah yes ok
which who whom whose where when why whether if
course courses class classes elective electives module modules unit units
take taking takes study studying learn learning enrol enroll
suit suits suited fit fits fitting good best right useful worth
recommend recommends recommended recommendation suggestion suggest
should shall need needs help helps helping tell show give
is are am was were been will did done any
switching switch moving transition transitioning from currently previously
background experience years year worked coming looking aiming
hmm hm um uh actually honestly think thinking guess maybe probably kinda kind
sorta sort prefer rather lean leaning side more thing things my me i
wanna gonna gotta tryna lemme dunno ya yeah yep nah
""".split())


def invented_job_function(said, role):
    """The job the model named that the student did not, or "".

    The guard against answering a question nobody asked. "What about something
    in esports" came back as a full recommendation for **esports operations
    coordinator** -- a job title the student had not said, followed by a list
    of "your target requirements" they had never given. The recommendation was
    perfectly good and it was for somebody else's career.

    It is the HEAD NOUN that decides, the last word of the title, because that
    is the job FUNCTION -- analyst, engineer, coordinator, biologist -- and
    inventing one is the whole failure. The modifiers are not the same thing: a
    student who typed "airscpae engineer" said "engineer" and merely misspelled
    the domain, and one who said "climate risk modeller" is not asking to be
    quizzed on whether they meant "modeler".

    Detecting invention rather than listing job nouns is deliberate. A list
    would have to know "biologist", "actuary", "modeller" and whatever comes
    next, and would be wrong about the one after that. "Did we make this up, or
    did they say it?" is exact, and it is the actual question.
    """
    import re as _re

    spoken = {word for word in _re.findall(r"[a-z]+", (said or "").lower())
              if word not in _ROLE_FILLER}
    proposed = [word for word in _re.findall(r"[a-z]+", (role or "").lower())
                if word not in _ROLE_FILLER]
    if not proposed:
        return ""
    head = proposed[-1]
    if any(router._within_one_edit(head, word) for word in spoken):
        return ""
    return head


def _student_words(question):
    """What the student actually named, with the sentence around it removed.

    Taken from the QUESTION rather than from `route.unmatched_role`, which is
    the model's extraction and can be thinner than what was said: scripted with
    a classifier that reported "esports" for a turn that read "esports
    analyst", the reply came back about **esports** -- dropping the very word
    the student had added to answer the question we had just asked them.
    """
    import re as _re

    words = [word for word in _re.findall(r"[a-z0-9+#/-]+", (question or "").lower())
             if word not in _ROLE_FILLER]
    return " ".join(words)[:60]


def curated_near(area):
    """Careers we DO have a ready-made set for that mention this area.

    Asked before falling back on the model's guess, because the model's guess
    is often poor and ours is always relevant: "I want to work in healthcare"
    was met with "I could read that as **healthcare worker**" -- an improvised
    title, for a programme that has a hand-written Healthcare / Life Sciences
    Analytics set sitting right there. Suggesting the curated one both answers
    better and lands the student on the stronger plan.
    """
    word = " ".join((area or "").lower().split())
    if not word:
        return []
    out = []
    for role in load_careers().values():
        haystack = " ".join(
            [role.get("label", ""), role.get("short_label", "")]
            + list(role.get("titles") or [])).lower()
        if word in haystack:
            out.append(role["label"])
    return out


ASK_WHAT_KIND_OF_WORK_CURATED = (
    "**{area}** — what kind of work do you want to do in it?\n\n"
    "The courses differ a lot between, say, analytics, operations and "
    "marketing, so a job title is enough. {closest} is the closest thing I "
    "have a ready-made set for — say that, or name a different job title.")

ASK_WHAT_KIND_OF_WORK = (
    "**{area}** — what kind of work do you want to do in it?\n\n"
    "I could read that as **{guess}**, but that is my reading rather than "
    "something you said, and the courses differ a lot between, say, analytics, "
    "operations and marketing. A job title is enough.\n\n"
    "Or just say **yes** and I'll go with that reading.")

# ---------------------------------------------------------------------------
# Route: a role the catalog has no profile for
# ---------------------------------------------------------------------------

NO_PROFILE_FOR_ROLE = (
    "I don't have a course recommendation for **{role}**.\n\n"
    "The MSBA elective mapping covers fourteen analytics job profiles, and that "
    "one isn't among them. I'm not going to assemble a plan around it by "
    "guessing which electives might transfer \u2014 a confident-sounding list of "
    "courses for a job nobody mapped them to is worse than no answer.\n\n"
    "**MSBA advising** can tell you whether the programme supports that "
    "direction; you can book time from the **Appointments** tab. If you'd like "
    "to look at what the mapping does cover, say **show me the industries**.")


# A job title is SHORT and has no digits in it. What is left of a sentence
# after the filler is stripped is not always a job: "i worked in marketing for
# 3 years and want to move more into data" reduces to "worked marketing 3
# years and move more data", which is a fragment of a life story, and naming
# it back to the student as the career we cannot cover is worse than useless.
_MAX_TITLE_WORDS = 4


def _looks_like_a_job_title(said):
    words = (said or "").split()
    if not words or len(words) > _MAX_TITLE_WORDS:
        return False
    return not any(any(ch.isdigit() for ch in word) for word in words)


def _uncurated_role(llm, route, conversation, question):
    """A job we have no profile for. Say so, and stop.

    This used to search the web for the role's skills and match them against
    the catalog with `role_lookup`. That is gone. The reason is not that it
    worked badly -- it worked -- but that it answered a DIFFERENT question from
    the one the rest of this route answers. Every curated profile's electives
    trace to the design document's bundle mapping, reviewed against the
    programme. A web-matched set traced to whatever a model read that morning,
    and arrived wearing the same formatting, so a student could not tell the
    two apart. Two sources of truth with one voice is the problem.

    `llm` and `route` are unused now and kept so the call sites do not have to
    care which kind of answer this is.
    """
    # The model's own extraction of the job comes first: it has already
    # stripped "hi! im an incoming international student, i want to be a"
    # from around "sports scientist", where the filler list leaves enough
    # of the sentence behind to look like no job at all.
    named = (route.unmatched_role or "").strip() or _student_words(question)
    if not _looks_like_a_job_title(named):
        # NOT A JOB, so not a refusal. A student who writes "i worked in
        # marketing for 3 years and want to move more into data" has told us
        # something useful and named no job at all -- and the reply was
        # "I don't have a course recommendation for **worked marketing 3
        # years and move more data**", which quotes a mangled fragment of
        # their own sentence back at them as though it were a career.
        #
        # Refusing is only honest when they actually named something we do
        # not cover. Otherwise the right answer is to ask.
        return industry_menu_reply(conversation, router.ROLE, route.confidence)
    return BotReply(NO_PROFILE_FOR_ROLE.format(role=named), [], "no-profile",
                    route=router.ROLE, route_confidence=route.confidence,
                    refused=True)


# ---------------------------------------------------------------------------
# Route: an industry
# ---------------------------------------------------------------------------

INDUSTRY_MENU_INTRO = (
    "No problem \u2014 that's what this is for. Analytics hiring splits along "
    "industry lines more than most people expect: the same degree points at "
    "quite different jobs depending on where you take it.\n\n"
    "**Which of these interests you most?**\n\n")

INDUSTRY_MENU_OUTRO = (
    "\n\nPick one and I'll show you the roles it hires for, ranked. You can "
    "change your mind at any point.")

INDUSTRY_ROLES_INTRO = (
    "**{label}.** {blurb}\n\nThe roles this industry hires MSBA graduates "
    "into, most in demand first:\n\n")

INDUSTRY_ROLES_OUTRO = (
    "\n\nSay which one you want to aim at and I'll build the elective plan "
    "around it \u2014 or say **show me the industries** to go back.")

# Asking outright, and admitting to having no target. Matched as PHRASES
# anywhere in the turn rather than as whole messages: "i have no idea what i
# want to do" is the commonest way a student says this and an exact-match list
# missed it, handing back the generic opening -- which is the least useful
# possible reply to somebody who has just said they are stuck.
_ASKS_FOR_INDUSTRIES = (
    "show me the industries", "show the industries", "list the industries",
    "show me industries", "list industries", "what industries",
)
_HAS_NO_TARGET = (
    "no idea", "not sure", "dont know", "don't know", "no clue", "unsure",
    "havent decided", "haven't decided", "not decided", "undecided",
    "no clue", "dunno",
)


_THANKS = re.compile(
    r"^(?:ok(?:ay)?[,!. ]*)?(?:thanks?|thank you|thx|ty|cheers|great|perfect|"
    r"awesome|cool|nice|got it|sounds good)(?:[,!. ]*(?:so much|a lot|this is "
    r"great|that helps))?[!. ]*$", re.IGNORECASE)
_DEGREE_UNITS = re.compile(
    r"\bhow many (?:units|credits)\b.*\b(?:graduate|degree|total|need|"
    r"program|programme|msba)\b|\bhow many (?:units|credits) (?:do i|is the)\b",
    re.IGNORECASE)
_DEGREE_ELECTIVES = re.compile(
    r"\bhow many electives?\b|\bhow many (?:of (?:those|them|these) are )?"
    r"electives?\b", re.IGNORECASE)
# "why", and the ways students question a pick without saying why: "is MGTA
# 458 really necessary", "do i actually need 466". Each, with a course code on
# the turn, is asking for the reason it is there.
_WHY = re.compile(
    r"\bwhy\b|\b(?:necessary|required|needed|essential|mandatory)\b|"
    r"\bdo i (?:really |actually )?(?:need|have to take)\b", re.IGNORECASE)

THANKS_THEN_LOAD = ("You're welcome. Whenever you're ready \u2014 **light**, "
                    "**moderate** or **heavy**?")
THANKS_THEN_TRACK = ("You're welcome. Which track are you on \u2014 the "
                     "**11-month** or the **17-month**?")
THANKS_THEN_GOAL = ("You're welcome. What are you aiming for after the "
                    "programme \u2014 a job, an industry, or not sure yet?")
THANKS_REPLY = (
    "You're welcome. Whenever you're ready, name a job or an industry -- or "
    "say you're not sure yet and I'll walk you through the options.")
DEGREE_UNITS_REPLY = (
    "The MSBA is **{total} units**: **{core}** of core that everyone takes, and "
    "**{elective}** of electives you choose. Full-time standing needs 12 a "
    "quarter. Name a job or an industry and I'll show you how the {elective} "
    "elective units fill in.")
DEGREE_ELECTIVES_REPLY = (
    "**{elective} of the {total} units are electives** -- the {core} core units "
    "are fixed. That is usually ten to twelve courses, depending on unit "
    "sizes. Name what you're aiming for and I'll pick them.")
# A colon, not "because it": the reasons mix verb phrases ("builds your
# data-engineering focus") with noun phrases ("directly relevant for Data
# Scientist"), and only the first kind agrees with "because it".
WHY_REPLY = "**{code} — {title}** is in your plan: {reasons}."
WHY_NOT_IN_PLAN = (
    "**{code}** isn't in your plan, so there's no pick to explain -- say "
    "**swap X for {code}** if you'd like it in.")


_QUARTER_LOAD_WORD = re.compile(
    r"\b(lighter|light|easier|heavier|heavy|harder|moderate|normal)\b",
    re.IGNORECASE)
_QUARTER_LOAD_OF = {"lighter": "light", "light": "light", "easier": "light",
                    "heavier": "heavy", "heavy": "heavy", "harder": "heavy",
                    "moderate": "moderate", "normal": "moderate"}


def _quarter_load_request(conversation, answers, said):
    """"Can I make Fall lighter?" with a plan on file -> that quarter, moved.

    The machinery existed (`_apply_quarter_load`) but was reachable only from
    inside the walk-through, keyed on the quarter on screen. Asked about a
    NAMED quarter with a plan already built, the turn went to the classifier,
    came back situational, and asked the student which track they were on.
    """
    if not (answers.get("track") and answers.get("goals")):
        return None
    word = _QUARTER_LOAD_WORD.search(said)
    if not word:
        return None
    track = answers.get("track") or "11 month"
    quarter_key = planner.resolve_quarter(track, said)
    if not quarter_key:
        return None
    keys = planner.quarter_keys(track)
    if quarter_key not in keys:
        return None
    return _apply_quarter_load(conversation, answers, keys.index(quarter_key),
                               _QUARTER_LOAD_OF[word.group(1).lower()])


def _direct_answer(conversation, answers, question):
    """Three answers that need no route, or None.

    * Thanks, before a plan exists. Afterwards `_small_talk_reply` handles it;
      before, it fell through to the generic opening, so "thanks!" was
      answered with "tell me what you're aiming for".
    * How many units, or how many electives, the degree is. Those are three
      constants, and "how many units do i need to graduate" was reaching the
      situational route and asking for the student's track in return.
    * Why a course is in the plan. The plan already holds the reason on the
      row; "why did you pick MGTA 463" was being read as a request for
      alternatives to it.
    """
    said = " ".join((question or "").strip().split())
    if _THANKS.match(said) and not planner.session_has_asked(conversation, "plan"):
        # Before the plan has been shown. After it, `_small_talk_reply` answers
        # in the model's own words; before it, this used to fire only with
        # nothing on file, so "thanks" between the load question and its
        # answer fell through to the classifier and came back as "I want to
        # make sure I answer the right question". The reply re-offers whatever
        # is pending, so the thank-you does not cost the student their place.
        if answers.get("goals") and answers.get("track"):
            body = THANKS_THEN_LOAD
        elif answers.get("goals"):
            body = THANKS_THEN_TRACK
        elif answers.get("track"):
            body = THANKS_THEN_GOAL
        else:
            body = THANKS_REPLY
        return BotReply(body, [], "small-talk", route=router.UNCLEAR)
    if not planner.mentioned_codes(said):
        if _DEGREE_UNITS.search(said):
            return BotReply(DEGREE_UNITS_REPLY.format(
                total=planner.TOTAL_UNITS, core=planner.CORE_UNITS,
                elective=planner.ELECTIVE_UNITS), [], "degree-fact",
                route=router.FACTUAL)
        if _DEGREE_ELECTIVES.search(said):
            return BotReply(DEGREE_ELECTIVES_REPLY.format(
                total=planner.TOTAL_UNITS, core=planner.CORE_UNITS,
                elective=planner.ELECTIVE_UNITS), [], "degree-fact",
                route=router.FACTUAL)
    quarter_load = _quarter_load_request(conversation, answers, said)
    if quarter_load is not None:
        return quarter_load
    if _WHY.search(said) and answers.get("track") and answers.get("goals"):
        codes = planner.mentioned_codes(said)
        if codes:
            plan = planner.build_for(
                answers, planner.taken_course_ids(conversation.user))
            _q, _slot, row = planner.locate_code(plan, codes[0])
            if row is None:
                # And WHERE it could go, since "swap X for it" is only useful
                # advice when there is an X in a quarter it runs in.
                course = next((c for c in load_catalog()
                               if c["code"].upper() == codes[0].upper()), None)
                note = (planner.placement_note(
                            plan, course, planner.taken_course_ids(conversation.user))
                        if course is not None else "")
                return BotReply(WHY_NOT_IN_PLAN.format(code=codes[0]) + note, [],
                                "why", route=router.FACTUAL)
            reasons = row.get("reasons") or ["fits the path you named"]
            joined = (reasons[0] if len(reasons) == 1
                      else ", ".join(reasons[:-1]) + " and " + reasons[-1])
            return BotReply(WHY_REPLY.format(code=row["code"], title=row["title"],
                                             reasons=joined), [], "why",
                            route=router.FACTUAL)
    return None


def wants_the_industry_menu(question):
    """Did this turn ask to see the industries, or admit to having no target?

    Deterministic rather than a model call: these are a small closed set of
    phrases, and paying a classifier round trip to be told that "no idea"
    means no idea is a second spent on something already known.

    A turn that NAMES something is never read as having no target, so "not
    sure whether to go into consulting or tech" keeps its industries rather
    than being treated as a blank.
    """
    lowered = " ".join(re.sub(r"[^a-z' ]+", " ", (question or "").lower()).split())
    if any(phrase in lowered for phrase in _ASKS_FOR_INDUSTRIES):
        return True
    if not any(phrase in lowered for phrase in _HAS_NO_TARGET):
        return False
    # The guards read the ORIGINAL text. The sanitiser above strips "&" for
    # phrase matching, and "probably fp&a analyst? not 100% sure" then failed
    # the role guard -- "fp a analyst" is not a title -- so a student who had
    # named a job was shown the industry menu, and the two turns after that
    # had no goal to attach to.
    return (not resolve_industry(question or "")
            and not router.matched_role(question or ""))


def industry_menu_body():
    lines = []
    for index, industry in enumerate(electives.load_industries(), 1):
        blurb = industry.get("blurb") or ""
        lines.append(f"{index}. **{industry['label']}**"
                     + (f" \u2014 {blurb}" if blurb else ""))
    return INDUSTRY_MENU_INTRO + "\n".join(lines) + INDUSTRY_MENU_OUTRO


def _button(text, label=None, description=""):
    """One quick reply, in the shape the client actually renders.

    `{label, send}`, NOT a bare string. The chat window keys its button row on
    `reply.send`; a list of strings gave every button the key `undefined`, and
    Svelte aborts a keyed block on a duplicate key -- so the whole message list
    threw and the conversation rendered as an empty pane. The reply itself was
    correct the entire time, which is exactly why this now has a test.

    `label` is the button face and `send` is what the student is taken to have
    said, so a long name can be shortened on screen without the router losing
    the words it matches on.
    """
    button = {"label": label or text, "send": text}
    if description:
        button["description"] = description
    return button


def industry_buttons():
    return [_button(industry["label"],
                    label=industry.get("short_label") or industry["label"])
            for industry in electives.load_industries()]


def industry_menu_reply(conversation, route=None, confidence=None):
    _remember_menu(conversation, "industries")
    return BotReply(industry_menu_body(), [], "industry-menu",
                    industry_buttons(),
                    route=route or router.INDUSTRY, route_confidence=confidence)


# Job titles are stored lowercase, and `str.title()` mangles every acronym in
# them -- "bi analyst" became "Bi Analyst", "heor analyst" "Heor Analyst". The
# catalog's own spelling of these is the one students see on postings.
_ACRONYMS = {
    "bi": "BI", "ml": "ML", "ai": "AI", "crm": "CRM", "heor": "HEOR",
    "aml": "AML", "sql": "SQL", "s&op": "S&OP", "fp&a": "FP&A",
    "cpg": "CPG", "genai": "GenAI",
}


def title_case(title):
    return " ".join(_ACRONYMS.get(word, word.title()) for word in title.split())


def _industry_roles_body(industry):
    """The industry's ten job titles, numbered, most in demand first.

    NO PROFILE NOTE beside the title. Each row used to carry the profile it
    maps to in brackets, and it read as "also known as" rather than "belongs
    to": the Healthcare list showed `BI Analyst (BI Developer)` at 3 and
    `BI Developer` at 9, so the note on one row named another row; and
    `Supply Chain Analyst (Supply Chain)` said nothing at all.

    Nothing is lost by dropping it. The mapping is self-evident one turn
    later, because a curated recommendation opens with the profile's own name
    -- pick "Applied Scientist" and the reply is headed "Data Scientist". What
    a student is scanning here is job titles, so this is job titles.
    """
    lines = []
    for index, (title, role_id) in enumerate(
            electives.top_titles_for(industry["id"], 10), 1):
        why = next((r.get("why") for r in industry["roles"]
                    if r["id"] == role_id), "") or ""
        tail = f" \u2014 {why}" if why else ""
        lines.append(f"{index}. **{title_case(title)}**{tail}")
    return (INDUSTRY_ROLES_INTRO.format(label=industry["label"],
                                        blurb=industry.get("blurb") or "")
            + "\n".join(lines) + INDUSTRY_ROLES_OUTRO)


def industry_roles_reply(conversation, industry, confidence=None):
    """The industry's top ten titles, and a note of which industry we are in.

    The industry is remembered because the titles are ambiguous on their own.
    "Growth analyst" is listed under BOTH Product and Marketing in the design
    document, so re-matching the student's pick as free text resolves it to
    whichever profile the matcher happens to reach first -- Product, even when
    the student picked it from the Retail list where it means Marketing. The
    menu knows which profile it offered; storing the industry lets the next
    turn ask the menu rather than guess.
    """
    _remember_menu(conversation, "roles", industry["id"])
    replies = [_button(title_case(title))
               for title, _ in electives.top_titles_for(industry["id"], 10)]
    return BotReply(_industry_roles_body(industry), [], "industry-roles", replies,
                    route=router.INDUSTRY, route_confidence=confidence)


# "6", "#6", "6.", "number 6", "option 3". A numbered list that will not take
# a number is a list that lied about being numbered -- measured live: the menu
# printed 1-6, said "Pick one", and a student who typed "6" was handed the
# generic "tell me what you're aiming for" fallback.
_JUST_A_NUMBER = re.compile(
    r"^(?:option|number|no\.?|#)?\s*#?\s*(\d{1,2})\s*[.)]?$", re.IGNORECASE)


def _said_a_number(question):
    match = _JUST_A_NUMBER.match(" ".join((question or "").strip().split()))
    return int(match.group(1)) if match else None


def pick_by_number(conversation, question):
    """What the student chose by typing a number, or None.

    Scoped to the menu THIS conversation was last shown, so "2" is the second
    industry after the industry menu and the second job title after the role
    list. A number typed with no menu on screen means nothing and falls
    through to ordinary routing rather than guessing at a list.
    """
    said = _said_a_number(question)
    if said is None or said < 1:
        return None
    stored = planner.load_session_intake(conversation) or {}
    menu = stored.get("menu")
    if menu == "industries":
        industries = electives.load_industries()
        if said <= len(industries):
            return ("industry", industries[said - 1])
    if menu == "roles" and stored.get("industry"):
        titles = electives.top_titles_for(stored["industry"], 10)
        if said <= len(titles):
            return ("role", titles[said - 1][1])
    return None


def _remember_menu(conversation, kind, industry_id=None):
    stored = planner.load_session_intake(conversation)
    updated = {**stored, "menu": kind}
    if industry_id:
        updated["industry"] = industry_id
    planner.save_session_intake(conversation, updated)


def role_from_industry_menu(conversation, question):
    """The role id behind a title the student just picked off the menu.

    None when this conversation has not been shown a menu, or when the turn is
    not one of the titles on it -- in which case the ordinary role matching
    runs and this never gets in the way.
    """
    industry_id = (planner.load_session_intake(conversation) or {}).get("industry")
    if not industry_id:
        return None
    said = " ".join((question or "").strip().lower().split())
    for title, role_id in electives.top_titles_for(industry_id, 10):
        if said == title.lower():
            return role_id
    return None


# "what's the difference between the first two", "1 vs 2", "compare analytics
# consultant and business analyst". A student looking at a numbered list of
# ten job titles asks this before picking one, and it used to reach the
# generic opening -- which answers a question about two named things by
# asking what they are aiming for.
_ASKS_TO_COMPARE = re.compile(
    r"\b(differ|differs|different|difference|differences|compare|comparison|"
    r"versus|vs)\b", re.IGNORECASE)

_ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3,
             "3rd": 3, "fourth": 4, "4th": 4, "fifth": 5, "5th": 5,
             "sixth": 6, "6th": 6, "seventh": 7, "7th": 7, "eighth": 8,
             "8th": 8, "ninth": 9, "9th": 9, "tenth": 10, "10th": 10}
_PAIR_WORDS = {"two": 2, "three": 3, "2": 2, "3": 3}


def _compared_positions(question, count):
    """The 1-based positions this turn is asking about, or []. Deterministic.

    Reads the three shapes people use: an ordinal run ("the first two"),
    explicit ordinals ("the first and the third"), and bare numbers ("1 vs 2").
    """
    said = " ".join((question or "").lower().split())
    run = re.search(r"\b(?:first|top|last)\s+(two|three|2|3)\b", said)
    if run:
        size = _PAIR_WORDS[run.group(1)]
        if "last" in run.group(0):
            return [n for n in range(count - size + 1, count + 1) if n >= 1]
        return list(range(1, min(size, count) + 1))
    picked = [_ORDINALS[w] for w in re.findall(r"[a-z0-9]+", said)
              if w in _ORDINALS]
    picked += [int(n) for n in re.findall(r"\b(\d{1,2})\b", said)
               if 1 <= int(n) <= count]
    ordered = list(dict.fromkeys(picked))
    return ordered[:3] if len(ordered) >= 2 else []


def wants_a_comparison(conversation, question):
    """(kind, [items]) for a comparison this conversation can answer, or None.

    Scoped to the menu that is on screen, exactly like `pick_by_number`: "the
    first two" means nothing without a list in front of it, and reading it as
    a comparison of something else would be a guess.
    """
    if not _ASKS_TO_COMPARE.search(question or ""):
        return None
    stored = planner.load_session_intake(conversation) or {}
    menu = stored.get("menu")
    if menu == "industries":
        industries = electives.load_industries()
        positions = _compared_positions(question, len(industries))
        named = [i for i in industries
                 if i["label"].lower() in (question or "").lower()]
        chosen = [industries[n - 1] for n in positions] or named
        return ("industries", chosen[:3]) if len(chosen) >= 2 else None
    if menu == "roles" and stored.get("industry"):
        titles = electives.top_titles_for(stored["industry"], 10)
        positions = _compared_positions(question, len(titles))
        chosen = [titles[n - 1][1] for n in positions]
        if len(chosen) < 2:
            said = (question or "").lower()
            chosen = [rid for title, rid in titles if title in said]
        chosen = list(dict.fromkeys(chosen))
        return ("roles", chosen[:3]) if len(chosen) >= 2 else None
    return None


COMPARE_ROLES_INTRO = "**{names}** — where they part company:\n\n"
COMPARE_ROLES_OUTRO = (
    "\n\nSay which one you want and I'll build the plan around it.")


def _role_comparison(conversation, role_ids):
    careers = electives.load_careers()
    stored = planner.load_session_intake(conversation) or {}
    industry = electives.industry_by_id(stored.get("industry") or "")
    ranked = [r["id"] for r in (industry or {}).get("roles", [])]
    rows = [careers[r] for r in role_ids]

    lines = [COMPARE_ROLES_INTRO.format(
        names=" vs ".join(r.get("short_label") or r["label"] for r in rows))]
    head = " | ".join([""] + [r.get("short_label") or r["label"] for r in rows])
    lines.append(f"|{head} |")
    lines.append("|" + "---|" * (len(rows) + 1))
    lines.append("| **The work** | "
                 + " | ".join(r.get("work") or r.get("description") or ""
                              for r in rows) + " |")
    lines.append("| **What it asks of you** | "
                 + " | ".join(r.get("gate") or "" for r in rows) + " |")
    if industry and all(r in ranked for r in role_ids):
        lines.append(f"| **Rank in {industry['label'].split(' (')[0]}** | "
                     + " | ".join(f"#{ranked.index(r) + 1} of {len(ranked)}"
                                  for r in role_ids) + " |")

    # WHERE THE ELECTIVES PART. Two careers that read differently can still
    # take nine of the same courses, and a student choosing between them
    # deserves to know that before they agonise over it.
    sets = {}
    for role_id in role_ids:
        bundle = bundles.bundle_for(role_id) or {}
        sets[role_id] = {c for key in ("anchor", "differentiator", "universal")
                         for c in (bundle.get(key) or [])}
    shared = set.intersection(*sets.values()) if sets else set()
    lines.append("\n**The electives, side by side.**\n")
    for role_id in role_ids:
        only = sorted(sets[role_id] - shared)
        label = careers[role_id].get("short_label") or careers[role_id]["label"]
        lines.append(f"- **Only {label}:** "
                     + (", ".join(only) if only else "nothing of its own"))
    lines.append(f"- **Both:** {len(shared)} courses in common"
                 + (f" — {', '.join(sorted(shared)[:4])}…" if shared else ""))
    return "".join(lines[:1]) + "\n".join(lines[1:]) + COMPARE_ROLES_OUTRO


def _industry_comparison(industries):
    lines = ["**" + " vs ".join(i["label"].split(" (")[0] for i in industries)
             + "** — where they part company:\n"]
    for industry in industries:
        top = [title_case(t) for t, _ in electives.top_titles_for(industry["id"], 3)]
        lines.append(f"\n**{industry['label'].split(' (')[0]}**"
                     + (f" — {industry['blurb']}" if industry.get("blurb") else ""))
        lines.append(f"  Hires most into: {', '.join(top)}.")
    lines.append("\n\nSay which one and I'll show you its roles, ranked.")
    return "\n".join(lines)


def comparison_reply(conversation, kind, items, confidence=None):
    body = (_role_comparison(conversation, items) if kind == "roles"
            else _industry_comparison(items))
    return BotReply(body, [], f"compare-{kind}", [],
                    route=router.INDUSTRY, route_confidence=confidence)


def resolve_industry(question):
    """Which of the six industries this turn names, or None.

    Matched on the industry's own label and on the words the label is made of,
    so "fintech", "biotech", "e-commerce" and "consulting" all land. Anything
    unrecognised returns None and gets the menu -- which is the honest answer,
    because these six are the whole of what the mapping covers.
    """
    said = (question or "").lower()
    best = None
    for industry in electives.load_industries():
        for word in _industry_words(industry):
            # WHOLE WORDS, and a HYPHEN DOES NOT END A WORD. A substring test
            # read "esports analyst" as the catch-all industry, because
            # "sports" is one of its sectors and sits inside "esports"; the
            # obvious repair, `\b...\b`, still matched "e-sports", because a
            # hyphen is a word boundary to `\b`. So a job we have no profile
            # for came back as a menu of six industries the student never
            # asked about, and the spelling with the hyphen is the commoner
            # one. The same trap waits in "e-commerce" and "bio-tech".
            if not re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", said):
                continue
            if best is None or len(word) > best[0]:
                best = (len(word), industry)
    return best[1] if best else None


def _industry_words(industry):
    label = industry["label"].lower()
    words = {label}
    # "Technology / Software" -> {"technology", "software"}; the parenthesised
    # tail of the catch-all industry is a list of its own sectors.
    for chunk in re.split(r"[/(),]| and | & ", label):
        chunk = chunk.strip()
        if len(chunk) > 3 and chunk != "other":
            words.add(chunk)
    words.update(_INDUSTRY_ALIASES.get(industry["id"], ()))
    return words


_INDUSTRY_ALIASES = {
    "technology-software": ("tech", "saas", "software", "startup", "startups"),
    "financial-services": ("finance", "fintech", "banking", "bank",
                           "insurance", "financial"),
    "consulting": ("consultancy", "advisory"),
    "healthcare": ("health", "biotech", "pharma", "life sciences", "medical",
                   "medtech", "device"),
    "retail-cpg": ("retail", "cpg", "ecommerce", "e-commerce", "consumer",
                   "grocery"),
    "other": ("media", "gaming", "games", "energy", "government", "defense",
              "defence", "sports", "telecom", "public sector"),
}


INDUSTRY_READING_NOTE = (
    "Reading **{said}** as **{label}** — say the word if you meant a different "
    "industry.\n\n")


def _answer_industry(llm, route, conversation, question):
    """An industry question, answered from the curated taxonomy only.

    This used to run a web search for what the field asks for and match the
    result against the catalog. That is gone for the same reason the web role
    lookup is: it produced a second, differently-sourced answer wearing the
    same clothes as the curated one.

    What replaced it is not a smaller answer. The design document ranks every
    profile WITHIN each of six industries, which is better than anything the
    search returned -- it is specific to this programme's fourteen profiles and
    it is reviewed. An industry we do not cover gets the menu of the six we do,
    rather than a plausible answer about a seventh.
    """
    industry = resolve_industry(question)
    if industry:
        return industry_roles_reply(conversation, industry, route.confidence)
    read = electives.industry_by_id(route.industry_id) if route.industry_id else None
    if (read and read["id"] == "other" and router.names_a_job_function(question)
            and not router.matched_role(question)):
        # "esports analyst" names a JOB, and the only industry the model could
        # place "esports" in is the catch-all. Answering with the catch-all's
        # ten titles would be the redirect the refusal rule forbids: a job
        # the fourteen profiles do not cover is refused, not steered to the
        # nearest list. Measured by the live sweep, twice, on two models.
        return _uncurated_role(llm, route, conversation, question)
    if read:
        # No alias on disk covered the words, and the `read_turn` skill read
        # them as one of the six -- by id, from the closed list, above the
        # confidence floor. Acted on AND confirmed back, exactly as a role
        # reading is: the student sees what was read and how to say no.
        reply = industry_roles_reply(conversation, read, route.confidence)
        said = " ".join((route.industry or "").split())[:60]
        if said and said.lower() != read["label"].lower():
            reply.body = INDUSTRY_READING_NOTE.format(
                said=said, label=read["label"]) + reply.body
        return reply
    return industry_menu_reply(conversation, router.INDUSTRY, route.confidence)


# ---------------------------------------------------------------------------
# Route: a role AND an industry
# ---------------------------------------------------------------------------

COMBINATION_JOIN = (
    "\n\n---\n\n**Now the {industry} part.** That role sits **#{rank} of "
    "{total}** in how much this industry hires MSBA graduates into it{why}.\n\n"
    "The electives above are still the spine \u2014 the role decides those. What "
    "the industry changes is who you are competing with, so here is the rest of "
    "the ranking for context:\n\n")

COMBINATION_NOT_RANKED = (
    "\n\n---\n\n_On the {industry} side: that role isn't one this industry "
    "is ranked for in the elective mapping, which usually means it is hired "
    "for elsewhere. The set above is still the right spine for the role._")


def _answer_combination(llm, route, conversation, question, history):
    """Role determinism for the spine, industry context on the electives.

    The order is the point. A role we curate has an authoritative answer and it
    goes first, in full; the industry lookup can only ever ADD colour to the
    elective choices around it. Letting the web half lead would put a
    probabilistic answer in front of a deterministic one for a question whose
    deterministic half we are sure about.
    """
    spine = _answer_role(llm, route, conversation, question, history)
    if spine.refused:
        # No curated spine to protect, so this is an industry question that
        # happened to mention a job title.
        return _answer_industry(llm, route, conversation, question)
    if not route.role_id:
        # The role path answered without a curated bundle -- a web-matched set,
        # or the question about which job they mean. Either way it is the
        # answer, and there is no spine to decorate.
        #
        # `spine.refused or not route.role_id` sent BOTH of those to the
        # industry path, which discarded a perfectly good clarifying question
        # and then re-ran the same lookup -- by which time `what-work` was
        # already marked as asked, so the second pass could only refuse.
        return spine
    industry = (resolve_industry(route.industry or question)
                or (electives.industry_by_id(route.industry_id)
                    if route.industry_id else None))
    label = (industry or {}).get("label") or route.industry or "industry"
    ranked = [r["id"] for r in industry["roles"]] if industry else []
    if industry and route.role_id in ranked:
        position = ranked.index(route.role_id)
        why = next((r.get("why") for r in industry["roles"]
                    if r["id"] == route.role_id), "") or ""
        careers = electives.load_careers()
        rest = "\n".join(
            f"{i}. **{(careers.get(r['id']) or {}).get('short_label') or r['id']}**"
            + (f" \u2014 {r['why']}" if r.get("why") else "")
            + ("  \u2190 you" if r["id"] == route.role_id else "")
            for i, r in enumerate(industry["roles"], 1))
        spine.body += COMBINATION_JOIN.format(
            industry=label, rank=position + 1, total=len(ranked),
            why=f" \u2014 {why.lower()}" if why else "") + rest
    else:
        spine.body += COMBINATION_NOT_RANKED.format(industry=label)
    spine.route = router.COMBINATION
    spine.model_note = "curated+industry"
    return spine


# ---------------------------------------------------------------------------
# Route: a factual question about a course
# ---------------------------------------------------------------------------

# The prompt lives in `skills/answer_course_facts/SKILL.md`. The catalog rows
# and retrieved passages are rendered into it, and the skill's rule is that
# they are the whole of what the model may say.
ANSWER_COURSE_FACTS = "answer_course_facts"

FACTUAL_REFUSAL = (
    "I don't have catalog or syllabus material that answers that, and I'd "
    "rather not guess — a wrong prerequisite or unit count is the kind of "
    "mistake you'd only find out about at enrolment.\n\n"
    "**MSBA advising** can confirm it; you can book time from the "
    "**Appointments** tab.")


def _answer_factual(llm, route, question, history):
    """Structured catalog first, syllabus corpus second, both or nothing.

    `courses.json` is the source of truth for units, seasons, prerequisites and
    workload because it carries them as FIELDS; a retrieved prose chunk carries
    whatever a page happened to say. The corpus is still retrieved alongside,
    because a syllabus answers "what does this course actually cover" in a way
    a catalog row does not.
    """
    config = bot_config("electives")
    catalog = _catalog_context(question)
    hits = retrieve(question, "courses", config["top_k"],
                    config["min_similarity"], config.get("lexical_min"),
                    config.get("lexical_floor", 0.0))
    if not catalog and not hits:
        return BotReply(FACTUAL_REFUSAL, [], "refusal", route=router.FACTUAL,
                        route_confidence=route.confidence, refused=True)
    messages = (history[-config["max_history_turns"]:]
                + [{"role": "user", "content": question}])
    body = reading.say(
        ANSWER_COURSE_FACTS, llm, messages,
        catalog=(f"Catalog entries:\n\n{catalog}" if catalog
                 else "Catalog entries: none matched."),
        passages=(f"Context passages:\n\n{build_context(hits)}" if hits
                  else "Context passages: none matched."))
    if not body:
        return BotReply(FACTUAL_REFUSAL, [], "refusal", route=router.FACTUAL,
                        route_confidence=route.confidence, refused=True)
    return BotReply(append_sources(body, hits),
                    [chunk.pk for chunk, _ in hits], "catalog-rag",
                    route=router.FACTUAL, route_confidence=route.confidence)


# ---------------------------------------------------------------------------
# Route: where this student actually is
# ---------------------------------------------------------------------------

def _answer_situational(llm, route, conversation, question):
    """Deterministic arithmetic, model-written interpretation. See `situation`."""
    stored = planner.load_session_situation(conversation)
    # Seeded from the intake. The situational session is its own record, so a
    # student who had given their track two turns earlier was asked for it
    # again -- "I need to know which track you're on now" -- by a route that
    # could have read it off the plan it was standing next to.
    intake = planner.load_session_intake(conversation) or {}
    if not stored.get("track") and intake.get("track"):
        stored = {**stored, "track": intake["track"]}
    # A route the model classified already carries the position it read
    # (`Route.position`); the rule-ruled route and the awaiting shortcut do
    # not, and read it once inside `situation.answer`.
    body, asked, resolved = situation.answer(
        llm, conversation.user, question, stored,
        context=turn_context(conversation, intake),
        said=route.position if route.confidence is not None else None)
    # Written whether or not anything was asked: a partial position is exactly
    # what the next turn needs, and a resolved one lets a follow-up ("what if I
    # take 16 units in Spring?") skip the extraction entirely. `awaiting` is
    # what brings a bare "winter" back here rather than to the classifier,
    # which has no way to know that one word is the answer to a question this
    # route asked one turn ago.
    planner.save_session_situation(conversation,
                                   {**stored, **resolved, "awaiting": asked})
    # And back the other way: a track this route worked out is a track the
    # intake has. Without it a student who had just been told "11 month
    # track, starting from Winter -- 2 quarters left" was asked which track
    # they were on by the very next turn.
    if resolved.get("track") in planner.TRACK_SKELETONS and not intake.get("track"):
        planner.save_session_intake(conversation,
                                    {**intake, "track": resolved["track"]})
    return BotReply(body, [], "situation", route=router.SITUATIONAL,
                    route_confidence=route.confidence)


# ---------------------------------------------------------------------------
# Route: one quarter, on its own
# ---------------------------------------------------------------------------

QUARTER_NEEDS_A_GOAL = (
    "I can plan **{label}** on its own — I just need to know what the plan is "
    "FOR. What are you aiming for after the programme? A job title is enough, "
    "or an industry.")

QUARTER_UNKNOWN = (
    "I couldn't tell which quarter you meant. On the {track} track they are: "
    "{quarters}. Say one of those and I'll plan just that one.")

QUARTER_LEAD = (
    "Here's **{label}** on its own. Say **next quarter** to carry on through "
    "the rest, or **show me the plan** for all {count} at once.")


def _answer_quarter(llm, route, conversation, question, answers):
    """One quarter, planned and shown by itself.

    A student asking "what should I take in Winter?" is asking a narrower
    question than "build my plan", and answering it with twelve courses across
    four quarters is answering a question they did not ask. The whole plan is
    still built underneath -- a quarter cannot be planned in isolation, because
    which electives are still available in Winter depends on what Spring and
    Fall took -- and only the quarter they asked about is shown.

    Deliberately NOT gated behind the load-spread question. That question earns
    its place before a whole plan of study; in front of one quarter it is an
    interruption, and the per-quarter load question is already in the output.
    """
    track = (answers.get("track") or stated_track(question)
             or _profile_track(conversation.user) or "11 month")
    key = planner.resolve_quarter(track, route.quarter or question)
    if key is None:
        return BotReply(
            QUARTER_UNKNOWN.format(
                track=track,
                quarters=", ".join(f"**{planner.quarter_label(track, k)}**"
                                   for k in planner.quarter_keys(track))),
            [], "quarter", route=router.QUARTER,
            route_confidence=route.confidence)

    label = planner.quarter_label(track, key)
    merged = {**answers, "track": track}
    if not merged.get("goals"):
        planner.save_session_intake(conversation, merged)
        return BotReply(QUARTER_NEEDS_A_GOAL.format(label=label), [], "intake",
                        route=router.QUARTER, route_confidence=route.confidence)

    filled, _assumed = planner.fill_assumed_skills(
        merged, [f"skill_{area['key']}" for area in planner.SKILL_AREAS])
    planner.save_session_intake(conversation, filled)
    plan = planner.build_for(filled, planner.taken_course_ids(conversation.user))
    index = next((position for position, quarter in enumerate(plan["quarters"])
                  if quarter["key"] == key), 0)
    reply = _review_reply(conversation, conversation.user, filled, index)
    lead = QUARTER_LEAD.format(label=label, count=len(plan["quarters"]))
    return spoken(BotReply(f"{lead}\n\n---\n\n{reply.body}", [], "quarter",
                           reply.quick_replies, route=router.QUARTER,
                           route_confidence=route.confidence))


def _profile_track(user):
    from rsm_thrive.models import StudentProfile

    profile = StudentProfile.objects.filter(user=user).first()
    return profile.track if profile else ""


# ---------------------------------------------------------------------------
# Routes that decline
# ---------------------------------------------------------------------------

OUT_OF_SCOPE = (
    "That's outside what I do — I'm the course planner, so I can answer "
    "questions about MSBA courses and electives, recommend courses for a "
    "career or an industry, and work out what's left in your plan of study.\n\n"
    "For anything else about the programme, **Ask THRIVE's Resources tab** "
    "answers from Rady's own material, and **MSBA advising** is bookable from "
    "the **Appointments** tab.")

DEGRADED = (
    "I can't reach my language model at the moment, so I can't work out what "
    "you're asking — that's a problem at my end, not with your question or "
    "with the course material.\n\n"
    "**What still works right now:** name a career and a track and I'll build "
    "the whole plan of study — _\"11 month, data scientist\"_ — then walk it "
    "quarter by quarter, change the load, or swap a course. None of that needs "
    "the model.\n\n"
    "What needs it back: questions about a specific course, industries and "
    "careers I have no ready-made set for, and working out where you are in "
    "the programme.")

UNCLEAR = (
    "I want to make sure I answer the right question. Are you asking about:\n\n"
    "- **a course** — what it covers, its units, when it runs;\n"
    "- **what to take** for a particular job or industry; or\n"
    "- **your own plan** — what's left, or how to finish on time?\n\n"
    "Tell me which and I'll go straight to it.")

# Said instead of the menu above when the conversation has established nothing
# yet -- a greeting, or a first message nobody could route. The three-shapes
# menu is the right answer to a question that was unclear; it is a cold answer
# to "hi", and it made the reply carry the menu AND the offer to plan
# underneath it, which is two invitations for one word.
UNCLEAR_OPENING = (
    "Happy to help — tell me **what you're aiming for after the programme** "
    "(a job title like \"data scientist\", or an industry) and **which track** "
    "you're on, the 11-month or the 17-month, and I'll lay out your whole plan "
    "of study.\n\n"
    "Or ask me something narrower — whether a course has prerequisites, what "
    "to take in one particular quarter — and I'll start there instead.")


MENU_WHICH = (
    "I couldn't tell which of those you meant — more than one fits. Say its "
    "number, or the title as it appears in the list.")


# A greeting gets greeted back, and THEN the three ways in -- in different
# words from the opening, which the student has just read. Measured live: "hi"
# was answered with the opening's own paragraph re-phrased, so the second
# message read as the first one repeating itself.
_HELLOS = frozenset({
    "hi", "hii", "hiya", "hey", "heya", "hello", "helo", "yo", "sup",
    "morning", "afternoon", "evening", "good morning", "good afternoon",
    "good evening", "hi there", "hey there", "hello there", "what's up",
    "whats up",
})
_HOW_ARE_YOU = frozenset({"how are you", "how's it going", "hows it going"})

GREETING_HELLO = "Hi! Good to have you here."
GREETING_HOW_ARE_YOU = "Doing well, thanks for asking!"
GREETING_ACK = "Great."

GREETING_WAYS_IN = (
    "So — where do you want to start?\n\n"
    "- **Got a job in mind?** Tell me the title and your track — _\"11 month, "
    "data scientist\"_ — and I'll build the whole plan.\n"
    "- **Know the field, not the job?** Name it — _healthcare_, _fintech_, "
    "_consulting_ — and I'll show you the roles it hires MSBA graduates into.\n"
    "- **No idea yet?** Completely fine — just say so and we'll go through the "
    "six industries together.\n\n"
    "And if there's a course question on your mind first, ask away.")

GREETING_THEN_TRACK = (
    "Which track are you on — the **11-month** or the **17-month**? Say it "
    "and I'll lay the plan out.")
GREETING_THEN_GOAL = (
    "What are you aiming for after the programme — a job title, an industry, "
    "or not sure yet? Any of those works.")
GREETING_THEN_LOAD = (
    "Whenever you're ready: **light**, **moderate** or **heavy** for the "
    "load, and the plan follows.")


def _greeting_reply(question, answers):
    """Greet back, then ask for the next thing -- never the opening again."""
    lowered = " ".join(re.sub(r"[^a-z' ]+", " ", (question or "").lower()).split())
    if lowered in _HOW_ARE_YOU:
        opener = GREETING_HOW_ARE_YOU
    elif lowered in _HELLOS:
        opener = GREETING_HELLO
    else:
        opener = GREETING_ACK
    if answers.get("goals") and answers.get("track"):
        body = GREETING_THEN_LOAD
    elif answers.get("goals"):
        body = GREETING_THEN_TRACK
    elif answers.get("track"):
        body = GREETING_THEN_GOAL
    else:
        body = GREETING_WAYS_IN
    return BotReply(f"{opener}\n\n{body}", [], "greeting", route=router.UNCLEAR)


def _decline(route, llm, conversation, question, answers):
    if route.name == router.DEGRADED:
        # Never `refused`: the corpus is fine, we could not reach the model.
        # Counting an outage as a content gap would put it straight into the
        # refusal report, which is meant to be the list of material to write.
        return BotReply(DEGRADED, [], "degraded", route=router.DEGRADED,
                        route_confidence=route.confidence)

    """Say plainly that this is not something we do, or ask which of three.

    With a plan already on screen there is a third case and it is the common
    one: "thanks", "ok", "that's helpful". Those are not out of scope and they
    are not unclear -- they are small talk, and answering them with a scope
    refusal or a menu of three question types reads as a bot that has stopped
    listening. `_small_talk_reply` answers in two sentences and never reprints
    the plan.
    """
    if planner.next_intake_step(answers) is None:
        reply = _small_talk_reply(llm, conversation.user, answers, question,
                                  planner.is_question(question))
        reply.route = route.name
        reply.route_confidence = route.confidence
        return reply
    if router.is_greeting(question):
        return _greeting_reply(question, answers)
    if (route.name == router.UNCLEAR and route.unmatched_role
            and _looks_like_a_job_title(route.unmatched_role)
            and router.names_a_job_function(route.unmatched_role)):
        # The reading was unclear about the ROUTE but read a job the profiles
        # do not cover. That is an uncovered job, refused as one -- not a
        # reason to re-ask what the student is aiming for.
        return _uncurated_role(llm, route, conversation, question)
    if route.name == router.OUT_OF_SCOPE:
        return BotReply(OUT_OF_SCOPE, [], "out-of-scope",
                        route=router.OUT_OF_SCOPE,
                        route_confidence=route.confidence, refused=True)
    menu = (planner.load_session_intake(conversation) or {}).get("menu")
    if menu in ("industries", "roles"):
        # A list is on screen and the turn could not be placed against it --
        # "the analyst one" on a list with four analyst titles. Measured
        # live, this fell through to the generic opening, which asks for a
        # job title and a track as though the list had never been shown.
        # The honest reply is about the list: which one.
        if menu == "industries":
            replies = industry_buttons()
        else:
            replies = [_button(title_case(title)) for title, _ in
                       electives.top_titles_for(answers.get("industry"), 10)]
        return BotReply(MENU_WHICH, [], "menu-which", replies,
                        route=router.UNCLEAR, route_confidence=route.confidence)
    if not answers.get("track") and not answers.get("goals"):
        # `model_note` is what stops `_with_a_way_back` adding the offer to
        # plan underneath: this reply already IS that offer.
        return BotReply(UNCLEAR_OPENING, [], "unclear-opening",
                        route=router.UNCLEAR, route_confidence=route.confidence)
    return BotReply(UNCLEAR, [], "unclear", route=router.UNCLEAR,
                    route_confidence=route.confidence)


# ---------------------------------------------------------------------------
# No buttons anywhere on this surface
# ---------------------------------------------------------------------------

def spoken(reply):
    """Turn a reply's buttons into a line of things the student can SAY.

    The click flow is gone, and that includes the shortcuts under a finished
    plan. They were never the interview, but they were still buttons, and a
    surface that answers free text and then hands back a row of chips is two
    interaction models in one conversation.

    The affordance is not dropped, only respoken. Every `send` string was
    already an ordinary sentence -- that was the point of them, so the
    transcript read as though the student had typed it -- and `route_intent`,
    `review_intent` and `mentioned_codes` all parse the same words whether
    they arrive from a button or a keyboard. Derived from the buttons
    themselves rather than written out again, so the two cannot drift.
    """
    if not reply.quick_replies:
        return reply
    sends = [choice["send"] for choice in reply.quick_replies if choice.get("send")]
    reply.quick_replies = []
    reply.form = None
    if not sends:
        return reply
    # Navigation and the load answers first, swaps after, and only the swaps
    # are capped. A quarter with two elective slots offers six swaps, and a
    # flat cap put all six in the line and dropped "next quarter" and the
    # light/moderate/heavy answers off the end -- the two things a student is
    # most likely to want next, crowded out by the ones they are least likely
    # to type verbatim.
    swaps = [send for send in sends if send.lower().startswith("swap ")]
    listed = [send for send in sends if send not in swaps] + swaps[:3]
    spoken_line = " · ".join(f"**{send}**" for send in listed)
    reply.body = f"{reply.body}\n\n_You can say:_ {spoken_line}"
    return reply


# ---------------------------------------------------------------------------
# Maintaining a plan that already exists
# ---------------------------------------------------------------------------

NOT_ADJUSTABLE = (
    "**{label}** isn't adjustable — the published plan fills it entirely with "
    "required courses, so there is no elective room to move. Say **next "
    "quarter** and I'll show you one you can change.")

BLOCKED_BY_A_CHOICE = (
    "I can't make **{label}** {load} without moving {blockers} off the load "
    "you already chose for {them}. The degree is {total} units, so the ones "
    "you have decided about are the only place these could come from.\n\n"
    "Tell me to make {blockers} moderate again and I'll do both.{available}")

CANNOT_REBALANCE = (
    "I can't make **{label}** {load} at all. The programme is {total} units "
    "and Summer is fixed, so units taken out of one quarter have to land in "
    "another — and the others are already at their limits.{available}")


def _available_note(options, load):
    rest = [option for option in options if option["value"] != load]
    if not rest:
        return ""
    return (" What I can do here: "
            + ", ".join(f"**{o['value']}** ({o['units']} units)" for o in rest)
            + ".")


def _apply_quarter_load(conversation, answers, index, load):
    """Move one quarter to a light / moderate / heavy load, and rebuild.

    The units change and so does the recommendation: `_resized` re-cuts the
    quarter's elective slots to the new budget, so a quarter asked to carry 18
    units instead of 14 gets another elective slot and the scorer fills it.
    That is the whole point of asking here rather than on a form -- the student
    is looking at the courses the answer changes.

    What is OFFERED and what is applied come from the same call, so the
    walk-through can never propose a load this refuses.
    """
    user = conversation.user
    plan = planner.build_for(answers, planner.taken_course_ids(user))
    quarters = plan["quarters"]
    index = max(0, min(index, len(quarters) - 1))
    quarter = quarters[index]
    track = answers.get("track") or "11 month"
    chosen = answers.get("quarter_units") or {}
    decided = dict(answers.get("quarter_loads") or {})
    options = planner.load_options_for(track, quarter["key"], chosen,
                                       set(decided))

    if not planner.load_is_a_choice(track):
        # Any quarter, Summer included: "moderate" said on the 11-month
        # walk-through is answered with why there is nothing to choose,
        # not with Summer's "say next quarter and I'll show you one you can
        # change" -- on this track there is no such quarter.
        reply = _review_reply(conversation, user, answers, index)
        return spoken(BotReply(
            LOAD_FIXED_ON_TRACK.format(label=quarter["label"],
                                       units=quarter["unitsPlanned"], track=track)
            + reply.body, [], "review", reply.quick_replies, route="plan"))
    if not planner.units_for_load(track, quarter["key"], "moderate"):
        return BotReply(NOT_ADJUSTABLE.format(label=quarter["label"]), [],
                        "review", route="plan")

    if decided.get(quarter["key"]) == load:
        # Said twice. Re-running the rebalance would produce the same numbers
        # and a reply claiming a change that did not happen.
        reply = _review_reply(conversation, user, answers, index)
        return spoken(BotReply(
            f"**{quarter['label']}** is already {load} — "
            f"{quarter['unitsPlanned']} units.\n\n---\n\n" + reply.body,
            [], "review", reply.quick_replies, route="plan"))

    offered = next((o for o in options if o["value"] == load), None)
    if offered is None:
        target = planner.LOAD_BOUNDS.get(load)
        bound = next((q for q in planner.adjustable_quarters(track)
                      if q["key"] == quarter["key"]), {})
        _wanted, blockers = planner.rebalanced_units(
            track, chosen, quarter["key"], bound.get(target, 0), set(decided))
        note = _available_note(options, load)
        if blockers:
            names = ", ".join(f"**{planner.quarter_label(track, key)}**"
                              for key in blockers)
            return BotReply(BLOCKED_BY_A_CHOICE.format(
                label=quarter["label"], load=load, blockers=names,
                them="it" if len(blockers) == 1 else "them",
                total=planner.TOTAL_UNITS, available=note),
                [], "review", route="plan")
        return BotReply(CANNOT_REBALANCE.format(
            label=quarter["label"], load=load, total=planner.TOTAL_UNITS,
            available=note), [], "review", route="plan")

    wanted, _blockers = planner.rebalanced_units(
        track, chosen, quarter["key"], offered["units"], set(decided))
    answers = {**answers, "quarter_units": wanted,
               "quarter_loads": {**decided, quarter["key"]: load}}
    planner.save_session_intake(conversation, answers)
    planner.save_intake(user, answers)

    # Report what the plan ACTUALLY carries, not the target it was given. The
    # lead said "Fall is now light -- 12 units" and the quarter rendered under
    # it said "Fall -- 14 units": the target was 12, but the recommended set
    # has no 2-unit course offered in Fall to come down to it, so the flex
    # left it at 14. A student reading both numbers on one screen has been
    # told two things, and the second one is the plan.
    rebuilt = planner.build_for(answers, planner.taken_course_ids(user))
    actual = {q["key"]: q["unitsPlanned"] for q in rebuilt["quarters"]}
    here = actual.get(quarter["key"], wanted[quarter["key"]])
    others = [key for key in wanted if key != quarter["key"]]
    first = f"**{quarter['label']}** is now {load} — {here} units."
    if here != wanted[quarter["key"]]:
        first += (f" (Asked for {wanted[quarter['key']]}; the courses this path "
                  f"needs land it at {here} — nothing smaller is offered there.)")
    lead = [first,
            "The other quarters absorbed it: "
            + ", ".join(f"{planner.quarter_label(track, key)} "
                        f"{actual.get(key, wanted[key])} units" for key in others)
            + f". The degree is still {planner.TOTAL_UNITS} units."]
    reply = _review_reply(conversation, user, answers, index)
    return spoken(BotReply("\n\n".join(lead) + "\n\n---\n\n" + reply.body,
                           [], "review", reply.quick_replies, route="plan"))


def _maintenance(llm, conversation, question, answers):
    """Acting on a plan already on screen, or None to route the question.

    Checked before the router because these are not questions about courses —
    they are operations on a plan. "next quarter" names nothing the classifier
    could recognise, and "swap 461 for 463" names two course codes and would be
    routed as a factual question about them.
    """
    user = conversation.user
    if planner.next_intake_step(answers) is not None:
        return None

    wanted_route = planner.route_intent(question)
    if wanted_route:
        answers = {**answers, "route": wanted_route}
        planner.save_session_intake(conversation, answers)
        reply = _plan_reply(user, answers)
        lead = ("Here it is built from the recommended bundle for that path."
                if wanted_route == "fixed"
                else "Here it is filled against your own skills and workload "
                     "instead.")
        return spoken(BotReply(f"{lead}\n\n{reply.body}", [], "plan",
                               reply.quick_replies, route="plan"))

    session = planner.load_session_review(conversation)

    # A load word, mid-walk-through, is about the quarter on screen. Checked
    # before `review_intent` because the two vocabularies do not overlap but
    # the ORDER states which reading wins, and before the change handler
    # because "light" names no course and would fall through to small talk.
    load = planner.load_intent(question)
    if session is not None and load:
        return _apply_quarter_load(conversation, answers, session["index"], load)

    intent = planner.review_intent(question)
    if intent == "finalise":
        return spoken(_finalise_reply(conversation, user, answers))
    if intent == "start":
        return spoken(_review_reply(conversation, user, answers, 0))
    if intent == "next" and session is not None:
        return spoken(_review_reply(conversation, user, answers, session["index"] + 1))

    if planner.is_question(question):
        explained = _explain_course(user, answers, question)
        if explained is not None:
            return explained

    change = _handle_change_request(user, answers, question)
    if change is not None:
        if session is not None and change.body.startswith("Swapped"):
            # Mid-walk-through, a successful swap re-renders the quarter on
            # screen -- but SAYS what it did first. The re-render alone was
            # returned before, so "swap MGTA 466 for MGTA 457" produced the
            # same quarter again with no acknowledgement, and a swap that
            # could NOT be made ("457 is already in your plan") was
            # indistinguishable from one that had.
            lead = change.lead or change.body.split("\n", 1)[0]
            review = _review_reply(conversation, user, answers, session["index"])
            return spoken(BotReply(f"{lead}\n\n---\n\n{review.body}", [],
                                   "review", review.quick_replies, route="plan"))
        return change

    # A course named that is in the catalog but not the plan, and wanted.
    # After the change handler, which owns every turn naming a course that IS
    # in the plan; before the router, which would answer from the catalog row
    # without knowing which quarter of this plan the answer points at.
    available = _course_availability(user, answers, question)
    if available is not None:
        return available

    if _wants_the_plan(question):
        return spoken(_plan_reply(user, answers))
    return None


# ---------------------------------------------------------------------------
# Coming back to the plan
# ---------------------------------------------------------------------------

# The routes that answer something OTHER than "build my plan". After one of
# these, and while no plan exists yet, the reply offers to get back to it --
# the surface opened by asking two questions, and a student who answered
# neither should not have to remember that it was waiting.
#
# `role` and `combination` are absent because they already end by asking for
# the track, and `quarter` because it produces a plan. Adding them would put
# two invitations in one reply.
_DETOURS = frozenset({router.FACTUAL, router.INDUSTRY, router.OUT_OF_SCOPE,
                      router.UNCLEAR})

# Capped at two. Once is a helpful offer; on every turn it is a leaflet, and a
# student asking a run of catalog questions is doing something legitimate that
# does not need interrupting.
NUDGES = (
    "\n\n---\n\nWhenever you're ready: tell me **what you're aiming for** "
    "and **which track** you're on, and I'll lay out your whole plan of "
    "study — or name one quarter and I'll do just that one.",
    "\n\n_Still happy to build that plan — one line like _\"11 month, data "
    "scientist\"_ is enough._",
)


NUDGE_TRACK_ONLY = (
    "\n\n---\n\nWhenever you're ready, tell me **which track** you're on "
    "\u2014 the 11-month or the 17-month \u2014 and I'll lay the plan out "
    "quarter by quarter.")
NUDGE_GOAL_ONLY = (
    "\n\n---\n\nWhenever you're ready, tell me **what you're aiming for** "
    "\u2014 a job title or an industry, or say you're not sure \u2014 and "
    "I'll lay out your whole plan of study.")


def _with_a_way_back(conversation, answers, reply):
    """Append the offer to plan, at most twice per conversation."""
    if reply.route not in _DETOURS or reply.model_note == "unclear-opening":
        return reply
    if reply.model_note in ("industry-menu", "industry-roles", "menu-which", "greeting",
                            "compare-roles", "compare-industries",
                            "small-talk", "degree-fact"):
        # A menu ends with its own instruction -- "pick one and I'll show you
        # the roles it hires for". Appending "tell me what you're aiming for
        # and which track you're on" under it asks a SECOND, different question
        # about the same turn, and the student is left choosing which one to
        # answer. Measured live: the menu printed six industries, the nudge
        # asked for a job title, and the next turn was neither.
        return reply
    if answers.get("track") and answers.get("goals"):
        return reply            # the plan exists; nothing to come back to
    for position, nudge in enumerate(NUDGES):
        key = f"nudge-{position}"
        if planner.session_has_asked(conversation, key):
            continue
        planner.note_session_asked(conversation, key)
        # Only the half that is missing. With a goal on file the nudge still
        # read "tell me what you're aiming for and which track" -- asking for
        # something the student had said two turns earlier.
        if position == 0:
            if answers.get("goals"):
                nudge = NUDGE_TRACK_ONLY
            elif answers.get("track"):
                nudge = NUDGE_GOAL_ONLY
        reply.body += nudge
        return reply
    return reply


# ---------------------------------------------------------------------------
# What the model is told about the conversation before it reads a turn
# ---------------------------------------------------------------------------

def turn_context(conversation, answers):
    """The planner's own records, for the `read_turn` skill.

    A one-word answer means nothing without the question it answers. The
    rules never had this: `load_intent` had to be told by its caller whether
    the load question was pending, and `_BARE_TRACK` reasons at length about
    why a lone "17" can only be a track. The skill is simply TOLD what was
    asked, what is on file and what is on screen, and reads the turn in that
    light. Every line is a fact from this module's own state, never anything
    the student typed, and it is labelled as such in the prompt.
    """
    answers = answers or {}
    context = {}
    awaiting = planner.load_session_situation(conversation).get("awaiting")
    if awaiting == "track":
        pending = "which track they are on now (11-month or 17-month)"
    elif awaiting == "quarter":
        pending = "which quarter they are in right now"
    elif (answers.get("track") and (answers.get("goals")
                                    or answers.get("unmatched_goal"))
          and not answers.get("workload")
          and planner.session_has_asked(conversation, "workload")):
        pending = "how they want the load spread (light, moderate or heavy)"
    elif answers.get("goals") and not answers.get("track"):
        pending = "which track they are on (11-month or 17-month)"
    elif answers.get("track") and not answers.get("goals") \
            and not answers.get("unmatched_goal"):
        pending = "what they are aiming for after the programme (a job title or an industry)"
    else:
        pending = ""
    context["question the planner asked last"] = pending or "none"
    on_file = []
    if answers.get("track"):
        on_file.append(f"track {answers['track']}")
    for role_id in answers.get("goals") or []:
        label = (load_careers().get(role_id) or {}).get("label")
        if label:
            on_file.append(f"goal {label}")
    if answers.get("unmatched_goal"):
        on_file.append(f"goal {answers['unmatched_goal']} (no curated profile)")
    if answers.get("workload"):
        on_file.append(f"load {answers['workload']}")
    context["on file"] = "; ".join(on_file) or "nothing yet"
    menu = answers.get("menu")
    if menu == "industries":
        context["menu on screen"] = "industries: " + "; ".join(
            f"{i}. {industry['label']}"
            for i, industry in enumerate(electives.load_industries(), 1))
    elif menu == "roles" and answers.get("industry"):
        industry = electives.industry_by_id(answers["industry"]) or {}
        context["menu on screen"] = (
            f"job titles in {industry.get('label', answers['industry'])}: "
            + "; ".join(f"{i}. {title_case(title)}" for i, (title, _role) in
                        enumerate(electives.top_titles_for(answers["industry"], 10), 1)))
    if planner.next_intake_step(answers) is None and planner.session_has_asked(
            conversation, "plan"):
        from rsm_thrive.models import CoursePlan

        # Read-only. `save_intake` would also hand back the swaps, but it
        # writes the committed plan as a side effect, and describing the
        # screen to the model must not change what is on it.
        record = CoursePlan.objects.filter(user=conversation.user).first()
        plan = planner.build_for(answers, planner.taken_course_ids(conversation.user),
                                 record.selections if record else None)
        rows = []
        for quarter in plan.get("quarters") or []:
            for row in quarter.get("courses") or []:
                if row.get("code"):
                    rows.append(f"{row['code']} ({quarter['label']}) {row.get('title', '')}".strip())
        context["plan on screen"] = "yes: " + "; ".join(rows) if rows else "yes"
        review = planner.load_session_review(conversation)
        if review is not None:
            index = min(review.get("index", 0), max(len(plan["quarters"]) - 1, 0))
            if plan.get("quarters"):
                context["walk-through"] = (
                    f"quarter on screen is {plan['quarters'][index]['label']}")
    else:
        context["plan on screen"] = "no"
    return context


# ---------------------------------------------------------------------------
# Acting on what the skill read
# ---------------------------------------------------------------------------

# The acts that STATE something. A question mentions values without choosing
# them (`planner.merge_intake` says why), a greeting states nothing, and a
# decline is recorded as an answer to the pending question rather than as a
# value for it.
_STATING_ACTS = frozenset({"answer", "request", "correct", "accept", "chat", ""})


def _stated_by(route, question):
    """The intake values a model-read turn actually STATED, or {}.

    Two guards on top of the validators. A question states nothing -- the
    act says so, or the turn's own shape does (`planner.is_question`). And
    the situational route reads its own position and syncs the track back
    itself, so its turn is left to it rather than written twice.

    The GOAL is deliberately never taken from here. A role the model read
    goes through `_answer_role`, which confirms the reading back before
    anything is built on it; putting it straight on file would build the plan
    on a reading the student never saw.
    """
    if route.act not in _STATING_ACTS or planner.is_question(question):
        return {}
    if route.name in (router.SITUATIONAL, router.OUT_OF_SCOPE, router.DEGRADED):
        return {}
    learned = dict(route.learned)
    if route.action.get("name") == "set_load" or route.action.get("load"):
        # "make this one lighter" mid-walk-through: the reading carried
        # both a load and a plan action, and writing the load to the intake
        # rebuilt the WHOLE plan light before the action could move one
        # quarter. A load that arrives with a plan action belongs to it.
        learned.pop("workload", None)
    return learned


def _learned_from_route(conversation, question, answers, route):
    """Put on file whatever the reading says the turn stated, or None.

    The rules already had one pass at this turn (`_intake_progress` with
    `learned_from`) and found nothing they were sure of. This is the second
    pass, with what the `read_turn` skill saw and the validators kept. Same
    machinery, same questions, same plan -- only the reader differs.
    """
    learned = _stated_by(route, question)
    if not learned:
        return None
    return _intake_progress(conversation, question, answers, learned=learned)


def _plan_is_imminent(conversation, answers):
    """Will the next turn deliver the plan whatever it says?

    True once a track and a goal are on file, the plan has not been shown,
    and nothing is left to ask -- the load is on file, or has been asked, or
    is not a choice on this track. `_intake_progress` treats any turn in this
    state as the cue to build, which is right (an interview with no exit is
    the thing being removed) and is also the one moment a remark the rules
    cannot read costs the student something: it is built into the plan or
    it is lost.
    """
    answers = answers or {}
    track = answers.get("track")
    if not track or not (answers.get("goals") or answers.get("unmatched_goal")):
        return False
    if planner.session_has_asked(conversation, "plan"):
        return False
    return bool(answers.get("workload")
                or planner.session_has_asked(conversation, "workload")
                or not planner.load_is_a_choice(track))


def _plan_action(llm, conversation, question, answers, route):
    """Do to the plan what the reading says the student asked, or None.

    Mirrors `_maintenance` branch for branch and calls the same functions --
    `_review_reply`, `_finalise_reply`, `_handle_change_request`,
    `_apply_quarter_load`. Nothing here builds a plan differently; it only
    reaches those functions from phrasings the word lists did not hold
    ("yeah go ahead", "let's keep going", "can we make fall a bit easier").
    Only with a plan on screen: without one there is nothing to act on.
    """
    action = route.action
    if not action or planner.next_intake_step(answers) is not None:
        return None
    user = conversation.user
    name = action["name"]
    session = planner.load_session_review(conversation)
    if name == "finalise":
        return spoken(_finalise_reply(conversation, user, answers))
    if name == "start_review":
        return spoken(_review_reply(conversation, user, answers, 0))
    if name == "next":
        index = (session["index"] + 1) if session is not None else 0
        return spoken(_review_reply(conversation, user, answers, index))
    if name == "show_plan":
        return spoken(_plan_reply(user, answers))
    if name == "switch_route" and action.get("route_choice"):
        said = ("use the recommended bundle" if action["route_choice"] == "fixed"
                else "build it around me")
        return _maintenance(llm, conversation, said, answers)
    if name == "set_load" and action.get("load"):
        track = answers.get("track") or "11 month"
        keys = planner.quarter_keys(track)
        # The quarter from the reading, or from the turn itself. Measured
        # live: "can we make fall a bit easier" came back with the load and
        # an empty quarter, and lightening the WHOLE plan for that is the
        # wrong answer to a question about one quarter. Quarter names are a
        # closed set, so the turn's own words are checked before the plan is.
        quarter_key = (planner.resolve_quarter(track, action.get("quarter"))
                       or planner.resolve_quarter(track, router.named_quarter(question)))
        if quarter_key is None and session is not None:
            quarter_key = keys[min(session["index"], len(keys) - 1)]
        if quarter_key in keys:
            return _apply_quarter_load(conversation, answers,
                                       keys.index(quarter_key), action["load"])
        if not planner.load_is_a_choice(track):
            return BotReply(LOAD_FIXED_TRACK_SAID_NOTE.format(
                load=action["load"], track=track).strip(), [], "plan",
                route="plan")
        # The whole plan: a new overall preference, re-seeded across the
        # quarters exactly as a first answer to the load question would be.
        merged = {**answers, "workload": action["load"]}
        merged.pop("quarter_units", None)
        merged.pop("quarter_loads", None)
        return _plan_now(conversation, merged, answers)
    codes = action.get("courses") or []
    if name in ("swap", "alternatives") and codes:
        changed = _handle_change_request(user, answers, " ".join(codes))
        if changed is not None:
            return changed
        available = _course_availability(user, answers, " ".join(codes))
        if available is not None:
            return available
    if name == "explain" and codes:
        return _explain_course(user, answers, codes[0])
    return None


def _menu_action(llm, conversation, question, history, answers, route):
    """Act on a menu pick the skill read, or None.

    `pick_by_number`, `role_from_industry_menu` and `wants_a_comparison` read
    the exact shapes -- a bare number, a title typed verbatim, an ordinal --
    and run before any model call. This is for the rest: "the analyst one",
    "the last two", "compare fintech and consulting". The skill saw the menu
    numbered in its context and reports positions; positions are checked
    against the list actually on screen, and the reply is the one the
    deterministic pick would have produced.
    """
    if not route.menu_pick:
        return None
    stored = planner.load_session_intake(conversation) or {}
    menu = stored.get("menu")
    if menu == "industries":
        industries = electives.load_industries()
        chosen = [industries[n - 1] for n in route.menu_pick if n <= len(industries)]
        if route.menu_compare and len(chosen) >= 2:
            return comparison_reply(conversation, "industries", chosen[:3],
                                    route.confidence)
        if len(chosen) == 1:
            return industry_roles_reply(conversation, chosen[0], route.confidence)
        return None
    if menu == "roles" and stored.get("industry"):
        titles = electives.top_titles_for(stored["industry"], 10)
        chosen = list(dict.fromkeys(
            titles[n - 1][1] for n in route.menu_pick if n <= len(titles)))
        if route.menu_compare and len(chosen) >= 2:
            return comparison_reply(conversation, "roles", chosen[:3],
                                    route.confidence)
        if len(chosen) == 1:
            return _answer_role(
                llm, router.Route(router.ROLE, role_id=chosen[0],
                                  confidence=route.confidence,
                                  why="picked from the menu; read by the skill"),
                conversation, question, history)
    return None


# ---------------------------------------------------------------------------
# The dispatcher
# ---------------------------------------------------------------------------

CAREERS_INTRO = (
    "I have ready-made pathways for **{count}** careers. Each one is a "
    "hand-built set of electives rather than something assembled on the "
    "spot:\n\n")
# The old second half promised "I'll look up what that job asks for, match it
# against this catalog" -- the removed web lookup, still advertised. What
# actually happens now is the industry menu, so that is what it says.
CAREERS_OUTRO = (
    "\n\nName any of them and I'll build the whole plan of study — say the "
    "track in the same breath (*\"11 month, data scientist\"*) and I'll go "
    "straight to it.\n\n**Not on the list?** Say **show me the industries** "
    "and I'll go the other way round: pick a field, and I'll show you the "
    "roles it hires MSBA graduates into, ranked.")


def _answer_careers(route):
    """The curated careers, listed. No model, no retrieval, no web.

    "What jobs do you have" is a closed question about our own data and it was
    getting a clarifying question back, which is the worst answer available: it
    asks the student to work out how to phrase something we could simply have
    told them.

    Read from `careers.json` rather than written out, so the list cannot drift
    from the bundles it promises. The last paragraph advertises the uncurated
    path on purpose -- a list of fourteen reads as a menu, and a student whose
    goal is not on it needs to know it is still a question worth asking.
    """
    careers = electives.load_careers()
    labels = sorted(
        (career.get("short_label") or career.get("label") or key)
        for key, career in careers.items())
    body = (CAREERS_INTRO.format(count=len(labels))
            + "\n".join(f"- **{label}**" for label in labels)
            + CAREERS_OUTRO)
    return BotReply(body, [], "careers", route=router.CAREERS,
                    route_confidence=route.confidence)


CATALOG_INTRO = (
    "I can plan across **{total} courses** — **{core} core** and "
    "**{electives} electives** — drawn from {programmes} programmes:\n\n")
CATALOG_CAP = (
    "\n**Up to {cap} of your 28 elective units** may come from outside the "
    "MSBA's own courses. The rest have to be MGTA.\n\n")
CATALOG_NEXT = (
    "Ask about any code and I'll tell you what it covers, what it needs first "
    "and when it runs — or name a career and a track (*\"11 month, data "
    "scientist\"*) and I'll build the whole plan around them.")


def _sentence(text):
    """First letter up, the rest untouched. `str.capitalize` lowercases the
    tail, which turns "pre-approved MSBA electives" into "msba"."""
    return text[:1].upper() + text[1:] if text else text


def _answer_catalog(route):
    """The catalog, broken down by programme. No model, no retrieval.

    A model handed 86 rows summarises them away: it answered "86 courses, 6
    core and 80 electives, all others are electives", which is arithmetic the
    student could already see and none of the context they asked for. The
    breakdown is the answer, and it comes from the catalog itself so it cannot
    drift from what the planner can actually schedule.

    Codes rather than titles for the other programmes. Eighty-six titles is a
    wall; the codes are scannable, grouped by what each programme IS, and any
    one of them can be asked about by name in the next turn -- which the
    closing line says.
    """
    overview = electives.catalog_overview()
    catalog = electives.load_catalog()
    body = CATALOG_INTRO.format(
        total=overview["total"], core=overview["core"],
        electives=overview["electives"], programmes=len(overview["programmes"]))

    for programme in sorted(overview["programmes"],
                            key=lambda p: (p["prefix"] != "MGTA", -p["total"])):
        rows = [c for c in catalog if c.get("department") == programme["prefix"]]
        body += (f"### {programme['name']} — {programme['total']} "
                 f"{'course' if programme['total'] == 1 else 'courses'}\n"
                 f"_{_sentence(programme['terms'])}._\n\n")
        core = [c for c in rows if c["is_core"]]
        if core:
            body += ("**Core** — everyone takes these, "
                     f"{sum(c['units'] for c in core)} units:\n\n")
            body += "".join(
                f"- **{planner.display_code(c)}** — {c['title']} ({c['units']}u)\n"
                for c in sorted(core, key=lambda c: c["code"]))
            body += "\n**Electives:**\n\n"
        electives_here = sorted((c for c in rows if not c["is_core"]),
                                key=lambda c: c["code"])
        if programme["prefix"] == "MGTA":
            # The home programme, listed in full: these are the courses a
            # student is choosing between most of the time.
            body += "".join(
                f"- **{planner.display_code(c)}** — {c['title']} ({c['units']}u)\n"
                for c in electives_here)
        else:
            body += ", ".join(f"**{planner.display_code(c)}** {c['title']}"
                              for c in electives_here) + "\n"
        body += "\n"

    body += CATALOG_CAP.format(cap=overview["nonMsbaCap"])
    body += CATALOG_NEXT
    return BotReply(body, [], "catalog-overview", route=router.CATALOG,
                    route_confidence=route.confidence)


HANDLERS = {
    router.CAREERS: lambda llm, route, conv, q, h: _answer_careers(route),
    router.CATALOG: lambda llm, route, conv, q, h: _answer_catalog(route),
    router.ROLE: lambda llm, route, conv, q, h: _answer_role(llm, route, conv, q, h),
    router.COMBINATION: lambda llm, route, conv, q, h: _answer_combination(llm, route, conv, q, h),
    router.INDUSTRY: lambda llm, route, conv, q, h: _answer_industry(
        llm, route, conv, q),
    router.FACTUAL: lambda llm, route, conv, q, h: _answer_factual(llm, route, q, h),
    router.SITUATIONAL: lambda llm, route, conv, q, h: _answer_situational(llm, route, conv, q),
    router.QUARTER: lambda llm, route, conv, q, h: _answer_quarter(
        llm, route, conv, q, planner.load_session_intake(conv)),
}


def answer(llm, conversation, question, history):
    """One turn on the courses surface.

    Order: plan maintenance, then a route, then the route's handler. The route
    and its confidence travel out on the `BotReply` so `ChatTurnLog` records
    which path an answer came down — without that, "the aerospace answer was
    bad" is a complaint about a black box rather than a finding about a route.
    """
    answers = planner.load_session_intake(conversation)

    # Three cheap answers that need no route, ahead of plan maintenance --
    # which claims any turn naming a course code, and was answering "why did
    # you pick MGTA 463" with a table of alternatives to it.
    direct = _direct_answer(conversation, answers, question)
    if direct is not None:
        return _with_a_way_back(conversation, answers, direct)

    maintained = _maintenance(llm, conversation, question, answers)
    if maintained is not None:
        if not maintained.route:
            maintained.route = "plan"
        return maintained

    # Reading a track or a goal off the turn happens only where it could not
    # displace a better answer. A rule route already decided is a stronger
    # signal than the two words it happened to contain: "I've switched from
    # 17-month to 11-month and I'm already in Winter" names a track, and
    # letting that reach `_intake_progress` answered a question about finishing
    # on time by asking what job they wanted. So the enrichment runs only when
    # nothing was ruled, or when the rule was ROLE -- which is the one route
    # `_intake_progress` can legitimately complete, because a role plus a track
    # IS the plan.
    # A title the industry menu just offered. Ahead of every route, because
    # the menu is the authority on what it meant: "growth analyst" appears
    # under two profiles, and free-text matching resolves it to the wrong one
    # when it was picked off the Retail list. Nothing happens here unless this
    # conversation has actually been shown a menu.
    # A comparison of two things the menu just offered. Before the numbered
    # pick, because "1 vs 2" contains numbers and is not a pick of either.
    comparing = wants_a_comparison(conversation, question)
    if comparing:
        return _with_a_way_back(conversation, answers, comparison_reply(
            conversation, comparing[0], comparing[1]))

    # An industry named while the INDUSTRY MENU is on screen is a pick from
    # it, whatever else the word could mean. "something in fintech", typed at
    # the menu, is an alias for the Financial Analytics profile as well as an
    # industry -- and the rule router, seeing the role, built that bundle. The
    # menu never advanced, so the student's next turn, "5", picked the fifth
    # INDUSTRY rather than the fifth role. Context settles it: they were
    # looking at a list of industries.
    if (planner.load_session_intake(conversation) or {}).get("menu") == "industries":
        named = resolve_industry(question)
        if named:
            return _with_a_way_back(conversation, answers,
                                    industry_roles_reply(conversation, named))

    numbered = pick_by_number(conversation, question)
    if numbered:
        kind, value = numbered
        if kind == "industry":
            return _with_a_way_back(conversation, answers,
                                    industry_roles_reply(conversation, value))
        return _with_a_way_back(conversation, answers, _answer_role(
            llm, router.Route(router.ROLE, role_id=value,
                              why="picked by number from the menu"),
            conversation, question, history))

    picked = role_from_industry_menu(conversation, question)
    if picked:
        return _with_a_way_back(conversation, answers, _answer_role(
            llm, router.Route(router.ROLE, role_id=picked,
                              why="picked from the industry menu"),
            conversation, question, history))

    # "I don't know" is an answer, and the one this surface exists to handle.
    # It is asked BEFORE classification because no classifier can place it --
    # the words carry no subject at all -- and the old behaviour was to route
    # it as UNCLEAR and ask the student to rephrase, which is the least useful
    # possible reply to somebody who has just said they are stuck.
    if wants_the_industry_menu(question):
        return _with_a_way_back(conversation, answers,
                                industry_menu_reply(conversation))

    ruled = router.rule_route(question)

    # One of the six industries, named outright, with no job title alongside
    # it. Deterministic on purpose: the six are a CLOSED SET, so recognising
    # one is a lookup rather than a judgement, and paying a classifier call to
    # be told "industry" about the word "fintech" is a second-long round trip
    # for an answer already on disk. A turn that names a role as well falls
    # through to the router, which has a combination route for exactly that.
    if ruled is None and not router.matched_role(question):
        named = resolve_industry(question)
        # A JOB whose words happen to contain a sector of the catch-all
        # industry is not an industry question. "sports scientist" carries
        # "sports", which lists under "Other (media, gaming, ... sports)",
        # and this shortcut answered a job we do not cover with that list's
        # ten titles -- the redirect the refusal rule forbids. A job function
        # plus the catch-all goes to the router, which refuses it honestly.
        if (named and not (named["id"] == "other"
                           and router.names_a_job_function(question))):
            return _with_a_way_back(
                conversation, answers,
                industry_roles_reply(conversation, named))

    # A turn that answers a question this surface asked one turn ago. Only the
    # situational route asks anything, and what comes back is usually a single
    # word -- "winter" -- which no classifier can place. A rule route that says
    # otherwise still wins: a student who answers a question with a different
    # question has changed the subject, and holding them to the old one is the
    # interview behaviour being removed.
    if (planner.load_session_situation(conversation).get("awaiting")
            and (ruled is None or ruled.name == router.SITUATIONAL)):
        return _answer_situational(
            llm, router.Route(router.SITUATIONAL,
                              why="answering a question this route asked"),
            conversation, question)

    route = None
    if (ruled is not None and ruled.name in (router.ROLE, router.COMBINATION)
            and planner.next_intake_step(answers) is None
            and reading.courses_of([router.matched_title(question)])):
        # A PROFILE ALIAS THAT IS ALSO A COURSE TITLE, with a plan on screen.
        # "can i drop the web mining class and take fraud analytics instead"
        # names MGTA 463 -- and "fraud analytics" is how `careers.json` spells
        # the Risk & Fraud profile, so the rule switched the student's career
        # and rebuilt the plan. The rule cannot tell a course from a career
        # here; the skill can, and is asked. A plan action wins; anything
        # else leaves the goal switch to run exactly as before.
        route = router.model_route(llm, question, turn_context(conversation, answers))
        acted = _plan_action(llm, conversation, question, answers, route)
        if acted is not None:
            if not acted.route:
                acted.route = "plan"
            return acted
        route = None
    if ruled is None or ruled.name in (router.ROLE, router.COMBINATION):
        learned = None
        if (ruled is None and _plan_is_imminent(conversation, answers)
                and not learned_from(question)):
            # THE NEXT TURN DELIVERS THE PLAN WHATEVER IT SAYS -- the load
            # question has been asked, and the exit from it is that any
            # answer produces the plan with what was assumed said out loud.
            # Read the turn before that happens, so what it states is the
            # load the plan is built on rather than a remark the plan ignored.
            #
            # The skill is consulted even when `load_mentioned` would find a
            # word, because a word is not a preference. Measured live: "not
            # too heavy please, i work part time" was built as HEAVY -- the
            # loose reader saw "heavy" and could not see "not". The skill
            # reads it as light; the word-match stays as the fallback inside
            # `_intake_progress` for when the skill reads nothing at all.
            route = router.classify(llm, question,
                                    turn_context(conversation, answers))
            learned = dict(_stated_by(route, question))
            if route.act == "ask":
                # A QUESTION IS NOT THE CUE TO BUILD. "does heavy mean more
                # classes?" was answered with the whole plan, spread the
                # assumed way, and the question went unanswered. About the
                # load options it gets the answer here; about anything else
                # it takes its route, and the plan waits one more turn.
                if _QUARTER_LOAD_WORD.search(question or ""):
                    return BotReply(LOAD_EXPLAINED, [], "intake", route="plan")
                learned = None
        if learned is not None or route is None:
            progressed = _intake_progress(conversation, question, answers,
                                          learned=learned)
            if progressed is not None:
                return progressed

    read_early = route is not None
    route = ruled or route or router.classify(llm, question,
                                              turn_context(conversation, answers))
    if ruled is None:
        # The model read the turn, and a reading carries more than a route.
        # Whatever it STATED goes on file, whatever it asked of the plan on
        # screen is done, and "I don't know yet" gets the menu -- each through
        # the same code the rules reach, from phrasings the rules did not
        # hold. The handler below still runs when none of these answers.
        acted = None
        if not read_early:
            acted = _learned_from_route(conversation, question, answers, route)
        if acted is None:
            acted = _plan_action(llm, conversation, question, answers, route)
        if acted is None:
            acted = _menu_action(llm, conversation, question, history, answers, route)
        if acted is not None:
            if not acted.route:
                acted.route = "plan"
            return acted
        if (route.wants_menu and route.name in (router.UNCLEAR, router.INDUSTRY,
                                                router.ROLE)
                and not route.role_id and not route.industry
                and not answers.get("goals")):
            return _with_a_way_back(conversation, answers,
                                    industry_menu_reply(conversation, route.name,
                                                        route.confidence))
    handler = HANDLERS.get(route.name)
    if handler is None:
        return _with_a_way_back(
            conversation, answers,
            _decline(route, llm, conversation, question, answers))
    reply = handler(llm, route, conversation, question, history)
    if not reply.route:
        reply.route = route.name
    return _with_a_way_back(conversation, answers, reply)
