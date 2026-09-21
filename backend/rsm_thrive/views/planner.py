"""The course planner API: interview, plan, alternatives, swap, reset."""
import functools

from django.views.decorators.http import require_http_methods

from rsm_thrive.http import (BadRequest, api_login_required, json_error, json_ok,
                             parse_body, profile_required)
from rsm_thrive.models import CoursePlan
from rsm_thrive.services.courses import planner


def _bad_request_as_400(view):
    """Turn a `BadRequest` into a 400, the way the chat views do.

    A decorator rather than four try/except blocks: every route here validates
    input, and the error shape a client parses must not differ depending on
    which one refused it.
    """
    @functools.wraps(view)
    def wrapper(request, *args, **kwargs):
        try:
            return view(request, *args, **kwargs)
        except BadRequest as exc:
            return json_error("bad_request", str(exc), 400)
    return wrapper


def _label_choices(plan, baseline):
    """Say "your choice" only where the student actually differed.

    A swap pins every auto-filled elective so the rest of the schedule cannot
    shift (see `planner.apply_swap`), which means `selections` ends up holding
    courses the student never picked. Labelling all of them "your choice" tells
    a student they chose a course the planner chose for them. Comparing against
    the plan the intake alone would produce recovers the distinction without
    storing a second map.
    """
    for quarter, base in zip(plan["quarters"], baseline["quarters"]):
        for row, base_row in zip(quarter["courses"], base["courses"]):
            if not row["swappable"] or not row["courseId"]:
                continue
            row["note"] = ("your choice" if row["courseId"] != base_row["courseId"]
                           else "")
    return plan


def _plan_payload(record):
    taken = planner.taken_course_ids(record.user)
    plan = planner.build_for(record.intake, taken, record.selections)
    if record.selections:
        plan = _label_choices(plan, planner.build_for(record.intake, taken, {}))
    plan["intake"] = record.intake
    plan["updatedAt"] = record.updated_at.isoformat()
    # A ready-to-render Markdown view alongside the structured plan. The chat
    # surface and any export want formatted text, and deriving it here keeps one
    # definition of how a plan reads rather than one per client.
    plan["markdown"] = planner.render_plan_markdown(plan)
    return plan


@api_login_required
@profile_required
@require_http_methods(["GET"])
@_bad_request_as_400
def intake(request):
    """The interview script, plus whatever this student already answered.

    Sent as data rather than rendered server-side so the questions, their
    options and their validation all come from one place: `planner`. A question
    added there appears in the UI without a frontend change.
    """
    record = CoursePlan.objects.filter(user=request.user).first()
    return json_ok({
        "questions": planner.intake_questions(),
        "answers": record.intake if record else {},
        # The profile already knows the track. Offering it as a default rather
        # than skipping the question: a student can be on a different track from
        # the one on file, and the plan is wrong in a way they cannot see if we
        # assume.
        "suggested": {"track": request.thrive_profile.track},
        "hasPlan": record is not None,
        # What the chat should show before the student has said anything, so the
        # courses surface opens on the first question instead of a blank box.
        "starter": planner.opening_prompt(),
    })


@api_login_required
@profile_required
@require_http_methods(["GET", "POST", "DELETE"])
@_bad_request_as_400
def plan(request):
    if request.method == "GET":
        record = CoursePlan.objects.filter(user=request.user).first()
        if record is None:
            return json_error("no_plan", "No plan yet — answer the intake first.", 404)
        # The row also exists mid-interview, so "there is a row" is not "there
        # is a plan": building from a partial intake would fail on the track.
        step = planner.next_intake_step(record.intake)
        if step:
            return json_error("intake_incomplete",
                              f"Still need: {', '.join(step['missing'])}.", 409)
        return json_ok(_plan_payload(record))

    if request.method == "DELETE":
        CoursePlan.objects.filter(user=request.user).delete()
        return json_ok({"deleted": True})

    body = parse_body(request)
    answers = body.get("answers")
    if not isinstance(answers, dict):
        raise BadRequest("answers must be an object")
    problems = planner.validate_intake(answers)
    if problems:
        return json_error("invalid_intake", "; ".join(problems), 400)

    # A new intake invalidates old swaps: they were chosen against a different
    # plan, and silently carrying them over would put courses a student picked
    # for one goal into a plan built for another.
    record, _ = CoursePlan.objects.update_or_create(
        user=request.user,
        defaults={"track": answers["track"], "intake": answers, "selections": {}},
    )
    return json_ok(_plan_payload(record), status=201)


@api_login_required
@profile_required
@require_http_methods(["GET"])
@_bad_request_as_400
def alternatives(request):
    record = CoursePlan.objects.filter(user=request.user).first()
    if record is None:
        return json_error("no_plan", "No plan yet — answer the intake first.", 404)
    quarter = request.GET.get("quarter", "")
    try:
        slot = int(request.GET.get("slot", ""))
    except ValueError:
        raise BadRequest("slot must be an integer")
    built = planner.build_for(record.intake,
                               planner.taken_course_ids(record.user),
                               record.selections)
    try:
        return json_ok(planner.alternatives_for(
            built, record.intake, quarter, slot,
            planner.taken_course_ids(record.user)))
    except ValueError as exc:
        return json_error("bad_slot", str(exc), 400)


@api_login_required
@profile_required
@require_http_methods(["POST"])
@_bad_request_as_400
def swap(request):
    record = CoursePlan.objects.filter(user=request.user).first()
    if record is None:
        return json_error("no_plan", "No plan yet — answer the intake first.", 404)
    body = parse_body(request)
    quarter = body.get("quarter")
    course_id = body.get("courseId")
    if not isinstance(quarter, str) or not isinstance(course_id, str):
        raise BadRequest("quarter and courseId must be strings")
    try:
        slot = int(body.get("slot"))
    except (TypeError, ValueError):
        raise BadRequest("slot must be an integer")
    try:
        record.selections = planner.apply_swap(
            record.intake, record.selections, quarter, slot, course_id,
            planner.taken_course_ids(record.user))
    except ValueError as exc:
        # A refused swap is a normal outcome, not a server fault: the student
        # asked for something the published plan does not allow, and the message
        # says which rule stopped it.
        return json_error("swap_refused", str(exc), 400)
    record.save(update_fields=["selections", "updated_at"])
    return json_ok(_plan_payload(record))
