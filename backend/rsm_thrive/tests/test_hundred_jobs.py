"""A hundred job titles, and the rules that must hold for every one of them.

Fourteen careers are curated. Students name whatever they like, so the list
below is deliberately unbalanced towards what the app has NOT been built for:
analytics-adjacent jobs it can partly serve, jobs it cannot serve at all,
misspellings, and bare areas that name no job.

Three rules, checked on all hundred:

1. **Nothing is invented.** If the reply names a job, the student said it — or
   it is offered as a reading and asked about, not asserted.
2. **Nothing is recommended without saying it is a recommendation.** A reply
   that lists course codes under a confident heading reads as an instruction
   unless it says otherwise in the same breath.
3. **Nothing crashes, and every plan that appears is legal.**
"""

import json

import pytest
from django.contrib.auth.models import User

from rsm_thrive.models import Conversation
from rsm_thrive.services.courses import orchestrator, planner, router
from rsm_thrive.services.llm import FakeLLM

pytestmark = pytest.mark.django_db

# --- the hundred -----------------------------------------------------------

CURATED_SAID = [
    "business analyst", "data analyst", "senior analyst", "insights analyst",
    "reporting analyst", "bi analyst", "bi developer", "product analyst",
    "product manager", "data scientist", "analytics engineer", "data engineer",
    "marketing analyst", "growth analyst", "crm analyst", "consultant",
    "analytics consultant", "management consultant", "strategy consultant",
    "fraud analyst", "risk analyst", "supply chain analyst",
    "operations analyst", "demand planner", "pricing analyst",
    "decision scientist", "ml engineer", "machine learning engineer",
    "healthcare analyst", "clinical data analyst", "financial analyst",
    "quantitative analyst", "quant analyst", "s&op analyst",
]

ADJACENT = [
    "esports analyst", "sports scientist", "climate risk modeller",
    "actuary", "credit risk analyst", "revenue analyst", "people analyst",
    "hr analyst", "customer insights manager", "market research analyst",
    "business intelligence architect", "data steward", "solutions architect",
    "growth hacker", "trading analyst", "portfolio analyst",
    "logistics coordinator", "procurement analyst", "energy analyst",
    "policy analyst", "urban planner", "epidemiologist", "biostatistician",
    "clinical trial manager", "sports betting analyst", "game analyst",
    "media analyst", "advertising analyst", "seo specialist",
    "customer success manager",
]

UNSERVED = [
    "sommelier", "wildlife biologist", "chef", "firefighter", "architect",
    "veterinarian", "flight attendant", "novelist", "sculptor", "electrician",
    "paramedic", "translator", "beekeeper", "yoga instructor", "barista",
]

TYPOS = [
    "data nalyst", "data scienist", "data sceintist", "pricing anlayst",
    "produt manager", "markting analyst", "bussiness analyst",
    "consltant", "analytcs engineer", "esports analsyt", "airscpae engineer",
]

VAGUE = [
    "something in esports", "I want to work in healthcare",
    "something in finance", "anything in sports", "something with data",
    "work in tech", "the marketing side", "something in consulting",
    "I like numbers", "something in supply chain",
]

ALL_JOBS = CURATED_SAID + ADJACENT + UNSERVED + TYPOS + VAGUE


def test_there_really_are_a_hundred():
    assert len(ALL_JOBS) == 100, len(ALL_JOBS)
    assert len(set(ALL_JOBS)) == 100, "a duplicate would test one thing twice"


# --- scripted model replies ------------------------------------------------

def route_json(name="role", role="", industry="", confidence=0.85):
    return json.dumps({"route": name, "confidence": confidence, "role": role,
                       "industry": industry})


def job_json(role, known=True):
    return json.dumps({
        "known": known, "role": role, "summary": f"{role} work.",
        "skills": ["sql", "dashboards", "statistical analysis",
                   "data storytelling", "forecasting"],
        "tools": ["python", "tableau"], "topics": ["experiment design"]})


NOT_A_JOB = json.dumps({"known": False, "role": "", "summary": "", "skills": [],
                        "tools": [], "topics": []})


@pytest.fixture
def user():
    return User.objects.create_user("stu")


def fresh(user, name="c"):
    return Conversation.objects.create(user=user, destination="courses",
                                        title=name)


def scripted(said, model_role=None, known=True):
    """A model that routes to `role` and names `model_role` for the job."""
    role = model_role if model_role is not None else said
    return FakeLLM([route_json("role", role=said),
                    job_json(role, known) if known else NOT_A_JOB,
                    "Here is what the catalog offers…"] + ["ok"] * 4)


CODE = router.COURSE_CODE


def names_courses(body):
    return bool(CODE.search(body))


def says_it_is_only_a_recommendation(body):
    lowered = body.lower()
    return ("recommendation, not a decision" in lowered
            or "not a guarantee of employment" in lowered
            or "recommendation rather than a decision" in lowered)


# --- rule 1: nothing is invented -------------------------------------------

@pytest.mark.parametrize("said", ALL_JOBS)
def test_the_reply_never_asserts_a_job_the_student_did_not_name(said, user):
    """The model renames and re-invents freely. Whatever it returns, the reply
    must either use the student's words or ASK about its reading."""
    conversation = fresh(user, said)
    reply = orchestrator.answer(
        scripted(said, model_role="chief widget officer"), conversation, said, [])
    body = reply.body.lower()
    if "what kind of work" in body:
        assert "my reading rather than something you said" in body \
            or "ready-made set" in body, said
        return
    assert "chief widget officer" not in body, said
    stored = planner.load_session_intake(conversation).get("unmatched_goal", "")
    assert "widget" not in stored.lower(), said


@pytest.mark.parametrize("said", VAGUE)
def test_a_bare_area_is_never_silently_turned_into_a_career(said, user):
    """Three honest answers and one dishonest one.

    Naming an industry we cover and listing its roles is honest. Offering the
    menu is honest. Reading it as a career we curate and SAYING so is honest.
    Handing back a full recommendation with nothing marking it as a reading is
    not -- and that is still the only thing forbidden here.

    This used to require the words "what kind of work", from a follow-up that
    belonged to the removed web lookup. A bare area now lands on the industry
    taxonomy instead, which answers the same question with a ranked list of
    real jobs rather than another question.
    """
    conversation = fresh(user, said)
    reply = orchestrator.answer(
        scripted(said, model_role=f"{said} specialist coordinator"),
        conversation, said, [])
    lowered = reply.body.lower()
    if not names_courses(reply.body):
        assert (reply.model_note in ("industry-menu", "industry-roles",
                                     "no-profile")
                or "what kind of work" in lowered), said
        if reply.model_note == "no-profile":
            # Saying there is no recommendation is only honest if it also says
            # where to go next, which is advising or the industry menu.
            assert "show me the industries" in lowered, said
        return
    assert "reading" in lowered and "meant something else" in lowered, said


@pytest.mark.parametrize("said", CURATED_SAID + TYPOS)
def test_a_named_job_is_never_interrogated(said, user):
    """It must not start asking people who said what they meant."""
    conversation = fresh(user, said)
    reply = orchestrator.answer(scripted(said), conversation, said, [])
    assert "what kind of work" not in reply.body.lower(), said


# --- rule 2: it is evident that this is only a recommendation --------------

@pytest.mark.parametrize("said", ALL_JOBS)
def test_no_reply_names_a_course_without_saying_it_is_a_recommendation(said, user):
    conversation = fresh(user, said)
    reply = orchestrator.answer(scripted(said), conversation, said, [])
    if names_courses(reply.body):
        assert says_it_is_only_a_recommendation(reply.body), \
            f"{said}: named courses with no recommendation framing"


@pytest.mark.parametrize("said", ALL_JOBS)
def test_and_the_plan_it_leads_to_says_so_too(said, user):
    conversation = fresh(user, said)
    for turn in (said, "11 month", "skip", "moderate"):
        reply = orchestrator.answer(scripted(said), conversation, turn, [])
    if "plan of study" not in reply.body:
        return                      # it asked or refused; nothing to check
    assert says_it_is_only_a_recommendation(reply.body), said
    assert "advising" in reply.body.lower(), said


def test_every_curated_recommendation_says_it(user):
    for role_id in planner.load_careers():
        body = orchestrator.curated_recommendation(role_id)
        assert says_it_is_only_a_recommendation(body), role_id


# --- rule 3: nothing crashes, every plan is legal --------------------------

@pytest.mark.parametrize("said", ALL_JOBS)
def test_the_whole_flow_holds_together(said, user):
    conversation = fresh(user, said)
    last = None
    for turn in (said, "11 month", "python 4, sql 2", "moderate",
                 "walk me through it", "next quarter", "show me the plan"):
        last = orchestrator.answer(scripted(said), conversation, turn, [])
        assert last.body.strip(), f"{said}: empty reply to {turn!r}"
        # The industry routes answer WITH buttons -- six industries, then ten
        # job titles -- so quick replies are expected there and nowhere else.
        if last.model_note in ("industry-menu", "industry-roles"):
            assert last.quick_replies, said
        else:
            assert last.quick_replies == [], said
        assert last.form is None, said
        assert "None" not in last.body, f"{said}: 'None' reached the page"

    answers = planner.load_session_intake(conversation)
    if planner.next_intake_step(answers) is not None:
        return
    plan = planner.build_for(answers, frozenset())
    assert plan["totals"]["total"] == planner.TOTAL_UNITS, said
    assert plan["unfilled"] == [], said
    ids = [row["courseId"] for quarter in plan["quarters"]
           for row in quarter["courses"] if row["courseId"]]
    assert len(ids) == len(set(ids)), said


@pytest.mark.parametrize("said", UNSERVED)
def test_a_job_this_degree_cannot_serve_names_no_course(said, user):
    conversation = fresh(user, said)
    reply = orchestrator.answer(scripted(said, known=False), conversation, said, [])
    assert reply.refused is True, said
    assert not names_courses(reply.body), said
    assert "advising" in reply.body.lower(), said
