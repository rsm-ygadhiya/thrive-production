"""Where a student actually is, against what the degree requires.

The question this answers is the one a form cannot: *I've switched from the
17-month to the 11-month track and I'm already in Winter — what do I take to
finish on time?* It is not a role question and not an industry question. It is
arithmetic about a specific person's position, and then a judgement about what
that arithmetic means.

## The line this module exists to hold

**The arithmetic is deterministic and the model never does any of it.**

Units remaining, quarters left, what is offered when, which requirements are
outstanding — every one of those is computed here, from `courses.json`,
`TRACK_SKELETONS` and the student's own enrolments, and handed to the model as
a finished ledger. The model's entire job is to explain the tradeoffs in the
ledger and to say what it would do about them.

If it says someone needs 14 more units and it is wrong, they miss graduation.
So `explain` is given the numbers and told, in the prompt and again by the
shape of what it receives, that it may not produce one of its own. `_facts`
renders the load-bearing figures into the reply itself, above the explanation,
so the numbers a student acts on are printed by this module whatever the model
does with its paragraph.

## Where the position comes from

Three sources, in order of authority:

1. **The transcript.** `Enrollment` rows say what has been completed. Nothing
   the student types can add units the record does not carry.
2. **What they said this turn.** A track switch and a current quarter are
   things only they know; both are extracted by the model and then RESOLVED
   here against the published skeletons, so an unrecognised quarter becomes a
   question rather than a guess.
3. **Their profile.** `StudentProfile.track` is the fallback when they have
   not said which track they are on.

Anything still unknown after those three is asked for. That is the one place
this route is allowed to ask a question, and it asks exactly one.
"""

from rsm_thrive.models import CoursePlan, StudentProfile
from rsm_thrive.services.courses import planner, reading
from rsm_thrive.services.courses.electives import load_catalog

TRACKS = ("11 month", "17 month")

# What the model is told when explaining a computed ledger:
# `skills/explain_position/SKILL.md`.
EXPLAIN_POSITION = "explain_position"

# The position is read by the planner's one reading skill (`read_turn`,
# section "where they are"); see `router.position_of`. What this module owns
# is checking it against the two tracks and the quarters of the track.


def extract_position(llm, question, context=None):
    """What the student said about where they are. Never what they implied."""
    from rsm_thrive.services.courses import router

    return router.read_position(llm, question, context)


def _profile_track(user):
    profile = StudentProfile.objects.filter(user=user).first()
    return profile.track if profile and profile.track in TRACKS else None


def resolve_position(user, said, stored=None):
    """(position, missing) — everything known about where they are.

    `missing` is the ONE thing still needed, or "". The route asks at most one
    question, because a student who has just described their situation in a
    sentence has earned an answer rather than an interview.
    """
    stored = stored or {}
    track = (said.get("track") or stored.get("track") or _profile_track(user))
    if track not in TRACKS:
        return {}, "track"
    quarter = (planner.resolve_quarter(track, said.get("said_quarter"))
               or stored.get("start_from"))
    if quarter not in planner.quarter_keys(track):
        return {"track": track}, "quarter"
    return {
        "track": track,
        "previous_track": said.get("previous_track") or stored.get("previous_track"),
        "start_from": quarter,
        "finish_on_time": bool(said.get("finish_on_time")),
    }, ""


ASK_TRACK = (
    "I can work out exactly what's left, but I need to know which track you're "
    "on now — the **11-month** or the **17-month**. Which is it?")

# The wording names the quarters rather than asking an open question, because
# "which quarter are you in" invites "my second one", and the second quarter of
# a 17-month track and of an 11-month track are the same quarter with different
# amounts of programme after them.
ASK_QUARTER_TEMPLATE = (
    "Which quarter are you in right now? On the {track} track that's one of: "
    "{quarters}.")


def ask_for(missing, track=None):
    if missing == "track":
        return ASK_TRACK
    quarters = ", ".join(
        f"**{planner.quarter_label(track, key)}**"
        for key in planner.quarter_keys(track or "11 month"))
    return ASK_QUARTER_TEMPLATE.format(track=track or "11 month", quarters=quarters)


# ---------------------------------------------------------------------------
# The arithmetic. Nothing below this line is ever done by a model.
# ---------------------------------------------------------------------------

def _codes_to_ids(codes):
    """Catalog ids for codes the student named. Unknown codes are dropped.

    Dropped, not trusted: a code that is not in this catalog is not a course
    this degree counts, and crediting it would inflate the completed-units
    figure with something the transcript does not carry.
    """
    wanted = {str(code).replace(" ", "").upper() for code in codes or []}
    return {course["id"] for course in load_catalog()
            if course["code"].replace(" ", "").upper() in wanted}


def ledger_for(user, position, extra_completed=()):
    """Every fact about this student's position, computed here.

    Returns (ledger, plan). The ledger is what the model is allowed to talk
    about; the plan is rendered by `planner.render_plan_markdown` exactly as
    every other plan is.
    """
    # The student's committed intake, when they have one, so a plan built here
    # is filled against the same goal and skill profile as the plan they
    # already have. The TRACK always comes from the position: they have just
    # told us they switched, and the saved intake is by definition out of date
    # about that.
    saved = CoursePlan.objects.filter(user=user).first()
    answers = {**((saved.intake if saved else None) or {}),
               "track": position["track"]}

    completed_ids = planner.completed_course_ids(user) | _codes_to_ids(extra_completed)
    # Everything enrolled in, so the plan does not schedule a course the
    # student is sitting in right now -- a different question from how many
    # units are banked, which is why the two sets are gathered separately.
    taken_ids = planner.taken_course_ids(user) | completed_ids

    plan = planner.build_for(answers, taken_ids, start_from=position["start_from"])
    catalog_units = {course["id"]: course["units"] for course in load_catalog()}
    completed_units = sum(catalog_units.get(cid, 0) for cid in completed_ids)
    scheduled = plan["totals"]["scheduled"]
    outstanding = max(0, planner.TOTAL_UNITS - completed_units - scheduled)

    return {
        "track": position["track"],
        "previousTrack": position.get("previous_track"),
        "startFrom": planner.quarter_label(position["track"], position["start_from"]),
        "quartersRemaining": plan["quartersRemaining"],
        "quarterLabels": [quarter["label"] for quarter in plan["quarters"]],
        "unitsRequired": planner.TOTAL_UNITS,
        "unitsCompleted": completed_units,
        "unitsScheduled": scheduled,
        "unitsOutstanding": outstanding,
        "finishesOnTime": outstanding == 0 and not plan["unscheduled"]
                          and not plan["unfilled"],
        "perQuarter": [{"label": quarter["label"],
                        "units": quarter["unitsPlanned"],
                        "courses": [row["code"] for row in quarter["courses"]
                                    if row["courseId"]]}
                       for quarter in plan["quarters"]],
        "unscheduledRequirements": [
            {"code": row["code"], "title": row["title"], "units": row["units"],
             "wouldHaveBeen": row["quarterLabel"]}
            for row in plan["unscheduled"]],
        "emptySlots": [{"quarter": row["quarter"], "units": row["units"],
                        "why": row["why"]} for row in plan["unfilled"]],
        "completedCourses": sorted(
            course["code"] for course in load_catalog()
            if course["id"] in completed_ids),
    }, plan


# ---------------------------------------------------------------------------
# Saying it
# ---------------------------------------------------------------------------

def _facts(ledger):
    """The numbers, printed by this module rather than by the model.

    Deliberately above the explanation and deliberately not paraphrased. If the
    model's paragraph and this block ever disagree, the student can see which
    one to act on -- and the one to act on is the one the catalog produced.
    """
    lines = [
        f"**Where you are.** {ledger['track']} track"
        + (f" (switched from {ledger['previousTrack']})" if ledger["previousTrack"] else "")
        + f", starting from **{ledger['startFrom']}** — "
        + f"{ledger['quartersRemaining']} quarter"
        + ("s" if ledger["quartersRemaining"] != 1 else "") + " left.",
        "",
        f"- Completed: **{ledger['unitsCompleted']} units**"
        + (f" ({', '.join(ledger['completedCourses'])})"
           if ledger["completedCourses"] else " on record"),
        f"- Scheduled below: **{ledger['unitsScheduled']} units**",
        f"- Still outstanding after this plan: **{ledger['unitsOutstanding']} units** "
        f"of the {ledger['unitsRequired']} the degree requires",
    ]
    if ledger["unscheduledRequirements"]:
        lines += ["", "**Required courses this plan cannot place**, because the "
                      "quarter they normally sit in is behind you:"]
        lines += [f"- **{row['code']} — {row['title']}** ({row['units']} units, "
                  f"normally {row['wouldHaveBeen']})"
                  for row in ledger["unscheduledRequirements"]]
        lines += ["", "You'll need MSBA advising to tell you when these can be "
                      "taken — that is a scheduling question this tool cannot "
                      "answer from the catalog."]
    for row in ledger["emptySlots"]:
        lines.append(f"- ⚠️ {row['units']}-unit slot in {row['quarter']}: {row['why']}")
    return "\n".join(lines)


def explain(llm, ledger, plan):
    """The tradeoffs, in the model's words, over this module's numbers."""
    import json

    payload = ("Ledger (the ONLY source of facts you may use):\n"
               + json.dumps(ledger, indent=1)
               + "\n\nExplain what this means for the student.")
    # An explanation is a courtesy; the ledger and the plan are the answer.
    # Losing the paragraph must never lose the numbers -- `say` returns ""
    # when the model fails, and the caller prints the ledger regardless.
    return reading.say(EXPLAIN_POSITION, llm,
                       [{"role": "user", "content": payload}])


NO_FINISH_NOTE = (
    "\n\n_This plan does not complete the degree in the quarters that remain. "
    "Please take it to MSBA advising — they can tell you what a petition, a "
    "summer session or an extra quarter would look like, and this tool cannot._")

DISCLAIMER = (
    "\n\n_Built from Rady's published plans of study and the MSBA course "
    "catalog. Every number above is computed from those; confirm your actual "
    "schedule with MSBA advising before you book._")


def answer(llm, user, question, stored=None, context=None, said=None):
    """(body, asked_for, resolved) for a situational question.

    `resolved` is everything the route worked out this turn, including a
    PARTIAL position when it had to ask for the rest. Handing it back is what
    stops the follow-up loop: asked for a track and then for a quarter, the
    second turn states only the quarter, and a route that cannot remember the
    first answer asks for the track again forever.

    `context` is what the planner knows about the conversation, handed to the
    reading so that a one-word answer to the question this route asked last
    turn is read as that answer. `said` is a position already read off this
    turn by the router (`Route.position`), so the route that classified the
    turn does not read it twice.
    """
    if said is None:
        said = extract_position(llm, question, context)
    if planner.is_hypothetical(question) and (stored or {}).get("track"):
        # "what if i had gone for the 17 month instead", asked by a student
        # whose track is on file, names the OTHER track -- and the reader
        # returned it as the one they are on, so the route asked which
        # quarter of the 17-month track they were in. A what-if compares; it
        # does not move them. The track on file stands.
        said = {key: value for key, value in said.items()
                if key not in ("track", "previous_track")}
    position, missing = resolve_position(user, said, stored)
    if missing:
        return ask_for(missing, position.get("track")), missing, position

    ledger, plan = ledger_for(user, position, said.get("completed_codes"))
    body = _facts(ledger)
    body += "\n\n" + planner.render_plan_markdown(plan)
    explanation = explain(llm, ledger, plan)
    if explanation:
        body += "\n\n" + explanation
    if not ledger["finishesOnTime"]:
        body += NO_FINISH_NOTE
    return body + DISCLAIMER, "", position
