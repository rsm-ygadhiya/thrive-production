"""The pieces a plan of study is maintained with, once one exists.

Swapping a course, walking the plan a quarter at a time, finalising it,
explaining why a course is there, and the two-sentence answer to anything
else once a plan is on screen. `orchestrator` dispatches to these after the
router has had its say; none of them is a route.

What went with the interview: `_extract_intake`, `_aside` and
`_retitle_if_button_pressed` existed only because the courses bot opened
with four scripted questions and had to cope with a student asking something
else during them. There is no script to interrupt now.
"""

from rsm_thrive.services.courses import planner, reading
from rsm_thrive.services.courses.electives import load_catalog
from rsm_thrive.services.reply import BotReply


def _catalog_entry(course):
    """One course as the model should see it: the fields, not prose about them."""
    seasons = sorted({o["season"] for o in course.get("offerings") or []})
    lines = [f"{course['code']} — {course['title']} ({course['units']} units, "
             f"{'core' if course['is_core'] else 'elective'})"]
    if seasons:
        lines.append(f"  offered: {', '.join(seasons)}")
    for label, key in (("prerequisites", "prerequisites"), ("workload", "workload"),
                       ("grading", "grading"), ("notes", "notes")):
        value = course.get(key)
        if value:
            lines.append(f"  {label}: {value}")
    for label, key in (("topics", "topics"), ("skills", "skills"), ("tools", "tools")):
        value = course.get(key)
        if value:
            lines.append(f"  {label}: {', '.join(value)}")
    if course.get("description"):
        lines.append(f"  about: {course['description']}")
    return "\n".join(lines)


def _catalog_context(question):
    """The catalog rows this question is about, or None if it is not about any.

    Grounded in `data/catalog/courses.json` rather than the document corpus,
    because that file IS the answer to a course question -- it carries units,
    seasons, prerequisites, tools and workload as fields, where a retrieved
    prose chunk carries whatever a web page happened to say. Retrieval also
    answered questions this bot has no business answering: asked about tuition
    it found a fee page and replied, which is the FAQ bot's job on a different
    tab.
    """
    from rsm_thrive.services.courses import electives

    courses = electives.search_catalog(question)
    if courses:
        return "\n\n".join(_catalog_entry(course) for course in courses)
    if not electives.is_course_question(question):
        return None
    # A course question that names nothing in particular. "What have you got?",
    # "which have no prerequisites" and "which run in winter" are all answered
    # by the WHOLE catalog rather than by six arbitrary rows -- and a code list
    # cannot answer any of them, which is what it used to return. Thirty-one
    # one-line entries is a small context and lets the model actually look.
    overview = electives.catalog_overview()
    # WHICH PROGRAMMES, not just which codes. Asked "what courses do you have
    # access to", a student wants to know they can reach past the MSBA's own
    # courses into CSE, MBA and MFin ones, and on what terms -- a code list
    # answers none of that.
    programmes = "\n".join(
        f"- {p['name']} ({p['prefix']}): {p['total']} courses "
        f"({p['core']} core, {p['electives']} elective) — {p['terms']}"
        for p in overview["programmes"])
    rows = []
    for course in electives.load_catalog():
        seasons = sorted({o["season"] for o in course.get("offerings") or []})
        rows.append(
            f"{course['code']} — {course['title']} | {course['units']}u | "
            f"{'core' if course['is_core'] else 'elective'} | "
            f"offered {', '.join(seasons) or 'unlisted'} | "
            f"prereq {course.get('prerequisites') or 'none'} | "
            f"workload {course.get('workload') or 'unlisted'}")
    return (f"The catalog holds {overview['total']} courses: "
            f"{overview['core']} core and {overview['electives']} electives, "
            f"drawn from these programmes:\n{programmes}\n"
            f"Up to {overview['nonMsbaCap']} of the 28 elective units may come "
            f"from outside the MSBA's own courses.\n\n"
            + "\n".join(rows))


def _divergence_note(answers, plan):
    """What a student on a fixed route has dropped out of their bundle.

    The design document is explicit that a recommendation which does not say
    what it gives up is not a recommendation. On a fixed route the bundle IS
    the recommendation, so swapping an anchor course out is exactly the moment
    to say so -- once, plainly, without refusing the swap.
    """
    from rsm_thrive.services.courses import bundles

    # The plan's OWN account of which fill it got, not the intake's request.
    # A bundle that turned out to be unschedulable falls back to the ranked
    # fill, and reading the request instead gave that plan a "you have moved
    # off the recommended bundle" warning about an edit the student never made.
    if plan.get("route") != "fixed":
        return ""
    goals = answers.get("goals") or []
    missing = bundles.divergence(plan, goals[0]) if goals else []
    if not missing:
        return ""
    names = ", ".join(f"**{code}**" for code in missing)
    return (f"\n\n> You have moved off the recommended bundle: {names} "
            f"{'is' if len(missing) == 1 else 'are'} part of what makes this a "
            f"plan for that path. Still your call — say the word and I'll put "
            f"{'it' if len(missing) == 1 else 'them'} back.")


def _plan_reply(user, answers, selections=None):
    """`selections` pins courses chosen outside the scorer -- the web-matched
    set for a career the catalog curates no bundle for."""
    record = planner.save_intake(user, answers)
    if selections:
        record.selections = selections
        record.save(update_fields=["selections", "updated_at"])
    plan = planner.build_for(answers, planner.taken_course_ids(user), selections)
    body = planner.render_plan_markdown(plan) + _divergence_note(answers, plan)
    body += ("\n\nWant to go through it a quarter at a time? I'll show what "
             "else fits each elective slot and what each one teaches, so you "
             "can swap before you finalise.")
    return BotReply(body, [], "plan", planner.review_intro_replies(answers),
                    plan=plan)


def _review_reply(conversation, user, answers, index):
    """Show one quarter of the review and remember where we are.

    The index is CLAMPED before it is stored. `review_quarter` clamps its own
    copy, so the right quarter was always rendered -- but the raw index went
    into the session, so a student pressing "next quarter" past the end walked
    the counter up for ever. Nothing visible broke; the state simply stopped
    describing anything, and every later read of it had to re-clamp to be safe.
    """
    taken = planner.taken_course_ids(user)
    record = planner.save_intake(user, answers)
    plan = planner.build_for(answers, taken, record.selections)
    index = max(0, min(index, len(plan["quarters"]) - 1))
    body, replies, _last = planner.review_quarter(plan, answers, index, taken)
    session = planner.save_session_intake(conversation, answers)
    session.review = {"index": index}
    session.save(update_fields=["review", "updated_at"])
    return BotReply(body, [], "review", replies)


def _finalise_reply(conversation, user, answers):
    taken = planner.taken_course_ids(user)
    record = planner.save_intake(user, answers)
    plan = planner.build_for(answers, taken, record.selections)
    session = planner.save_session_intake(conversation, answers)
    session.review = None
    session.save(update_fields=["review", "updated_at"])
    return BotReply(planner.finalised_markdown(plan), [], "plan")


def _handle_change_request(user, answers, question):
    """A message naming courses, once a plan already exists.

    Two shapes are useful and both are common: naming ONE course in the plan
    means "show me alternatives to this", and naming a second course as well
    means "make the swap". Anything else falls through to rebuilding the plan.
    """
    codes = planner.mentioned_codes(question)
    if not codes:
        return None

    record = planner.save_intake(user, answers)
    plan = planner.build_for(answers, planner.taken_course_ids(user),
                              record.selections)
    quarter_key, slot, row = None, None, None
    for code in codes:
        quarter_key, slot, row = planner.locate_code(plan, code)
        if row is not None:
            break
    if row is None:
        return None

    replacement = next((c for c in codes if c.upper() != (row["code"] or "").upper()),
                       None)
    if replacement:
        target = next((course["id"] for course in load_catalog()
                       if course["code"].upper() == replacement.upper()), None)
        taken = planner.taken_course_ids(user)
        by_id = {course["id"]: course for course in load_catalog()}
        try:
            record.selections = planner.apply_swap(
                answers, record.selections, quarter_key, slot, target, taken)
        except planner.NotOfferedError as exc:
            # Not offered THERE is not the end of the answer: say where it is,
            # and which quarter of this plan that is.
            here = next(q for q in plan["quarters"] if q["key"] == quarter_key)
            return BotReply(
                f"I can't put **{replacement}** in that slot: {exc}."
                + planner.placement_note(plan, by_id[target], taken,
                                         missing_from=here),
                [], "plan")
        except ValueError as exc:
            return BotReply(
                f"I can't put **{replacement}** in that slot: {exc}.", [], "plan")
        record.save(update_fields=["selections", "updated_at"])
        updated = planner.build_for(answers, taken, record.selections)
        # The course going OUT: is this goodbye, or can it be picked up later?
        # Said with the swap, because after it the student has no reason to ask.
        lead = f"Swapped **{row['code']}** for **{replacement}**."
        dropped = by_id.get(row["courseId"])
        if dropped is not None:
            lead += planner.displaced_note(updated, dropped, quarter_key)
        return BotReply(lead + "\n\n" + planner.render_plan_markdown(updated),
                        [], "plan", lead=lead)

    return BotReply(planner.render_alternatives_markdown(
        planner.alternatives_for(plan, answers, quarter_key, slot,
                                 planner.taken_course_ids(user))), [], "plan")



def _course_availability(user, answers, question):
    """A course the student wants that is NOT in the plan: say where it fits.

    "I want MGTA 456", "can I take CSE 251A", "when is 463 offered" -- each
    names a course, none of them is in the plan, so `_handle_change_request`
    and `_explain_course` both pass. Left alone the turn reached the catalog
    route, which answered "offered in Winter" from the catalog row: true, and
    not what a student with a plan on screen needs. The plan knows which
    quarter Winter IS and which course would have to make room.

    Only for a turn that reads as wanting the course or asking when it runs.
    "Does MGTA 456 have prerequisites?" names a course not in the plan too,
    and the catalog route answers that better than this can.
    """
    codes = planner.mentioned_codes(question)
    if not codes or not planner.wants_course(question):
        return None
    taken = planner.taken_course_ids(user)
    record = planner.save_intake(user, answers)
    plan = planner.build_for(answers, taken, record.selections)
    by_code = {course["code"].upper(): course for course in load_catalog()}
    wanted = [by_code[code.upper()] for code in codes
              if planner.locate_code(plan, code)[2] is None
              and code.upper() in by_code]
    if not wanted:
        return None
    notes = [planner.placement_note(plan, course, taken).strip()
             for course in wanted]
    body = "\n\n".join(notes)
    if len(wanted) == 1:
        course = wanted[0]
        body = (f"**{planner.display_code(course)} — {course['title']}** "
                f"({course['units']} units) isn't in your plan yet.\n\n{body}")
    return BotReply(body, [], "availability")


# Deterministic, like `review_intent`: a short closed list beats a model call.
_PLAN_REQUESTS = ("show me the plan", "show the plan", "show my plan", "print it",
                  "print the plan", "the plan again", "see it again", "again",
                  "full plan", "whole plan", "rebuild", "redo it", "start over",
                  "my plan")


def _wants_the_plan(question):
    """Did the student ask to SEE the plan of study?"""
    lowered = (question or "").strip().lower()
    return any(phrase in lowered for phrase in _PLAN_REQUESTS)


def _explain_course(user, answers, question):
    """Why a named course is in this plan, using the reasons already recorded.

    The engine writes a `reasons` list onto every elective row as it picks it,
    so the explanation is the scorer's own account rather than a second story
    told about it afterwards. Returns None when the question names no course in
    the plan, so the caller can fall through.
    """
    codes = planner.mentioned_codes(question)
    if not codes:
        return None
    record = planner.save_intake(user, answers)
    plan = planner.build_for(answers, planner.taken_course_ids(user),
                              record.selections)
    for code in codes:
        quarter_key, _slot, row = planner.locate_code(plan, code)
        if row is None:
            continue
        quarter = next(q for q in plan["quarters"] if q["key"] == quarter_key)
        lines = [f"**{row['code']} — {row['title']}** sits in "
                 f"**{quarter['label']}** ({row['units']} units)."]
        if row.get("reasons"):
            lines += [""] + [f"- {reason}" for reason in row["reasons"]]
        if row.get("cautions"):
            lines += [""] + [f"⚠️ {caution}" for caution in row["cautions"]]
        if row.get("swappable"):
            lines += ["", "Say the word and I'll show what else fits that slot."]
        else:
            lines += ["", "This one is required — there is nothing to choose here."]
        return BotReply("\n".join(lines), [], "plan")
    return None


# The two-sentence prompt lives in `skills/small_talk/SKILL.md`.
SMALL_TALK = "small_talk"


def _small_talk_reply(llm, user, answers, question, asking):
    """Anything else, once a plan exists: answer briefly, never reprint.

    The fallthrough used to be `_plan_reply`, so every unrecognised message --
    including "ok" and "thanks!" -- reprinted the entire plan of study.
    """
    body = reading.say(SMALL_TALK, llm, [{"role": "user", "content": question}])
    if not body:
        body = ("Your plan is set. You can ask me to swap a course, walk "
                "through it a quarter at a time, or show it again.")
    return BotReply(body, [], "plan")
