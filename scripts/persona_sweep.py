"""Five students, a hundred turns each, against the REAL model -- for both bots.

    cd backend && PYTHONPATH=$PWD uv run python ../scripts/persona_sweep.py \
        --bot resources|courses --persona newcomer|rusher|planner|switcher|worrier|all \
        [--out DIR] [--show]

The scripted sweeps use the words the code already knows. A student does not.
Each persona here writes the way one kind of student writes, and the same
hundred kinds of question are put through that voice:

* **newcomer**  -- an incoming international student, polite and hedged, asks
                   about arrival, visas, housing, IT setup, orientation.
* **rusher**    -- terse, lowercase, typos, one or two words, no question mark.
* **planner**   -- precise and long: dates, policies, where things are published.
* **switcher**  -- mid-career, career-first; unsure of the job, sure of the field.
* **worrier**   -- anxious run-ons and what-ifs; needs reassurance and a next step.

What is checked is what CAN be checked without a ground truth for a hundred
free-text answers: the promises each flow makes. Resources: always answers,
refuses only out of scope, says so when the web was used, lists only cited
sources, never hedges, a follow-up ends in an answer. Courses: every plan is
legal (50 units, no duplicate, nothing outside the catalog), the goal, track
and load on file are what was said, an uncovered job is refused and names no
course, a deterministic turn is never met with a clarification.

One JSON report per run in --out, every failure with the reply that produced
it. Exits non-zero on any failure.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.contrib.auth.models import User  # noqa: E402

from rsm_thrive.models import Conversation  # noqa: E402
from rsm_thrive.services.courses import electives, orchestrator, planner  # noqa: E402
from rsm_thrive.services.llm import get_llm  # noqa: E402
from rsm_thrive.services.resources.bot import answer_career, answer_faq  # noqa: E402

PERSONAS = ("newcomer", "rusher", "planner", "switcher", "worrier")
BUDGET_SECONDS = 60
CAT = {c["id"]: c for c in electives.load_catalog()}
DISPLAY = {c["code"] for c in electives.load_catalog()}
# Cross-listings and titles carry codes that are not catalog codes ("Web
# Mining and Recommender Systems (cross-listed CSE 258)"); they are the
# catalog's own words, not invented ones.
KNOWN_CODES = set(DISPLAY)
for c in electives.load_catalog():
    KNOWN_CODES.update(str(a).upper() for a in (c.get("also_known_as") or []))
    KNOWN_CODES.update(m.upper() for m in re.findall(
        r"\b(?:MGTA|MGTF|MGTP|MGT|CSE)\s?\d{3}[A-Z]?\b", str(c.get("title", ""))))
# A code the corpus itself mentions -- a prerequisite named in a syllabus, a
# cross-listing -- is grounded, not invented. Only a code found nowhere in
# our material counts as made up.
from rsm_thrive.models import DocumentChunk  # noqa: E402
for text in DocumentChunk.objects.values_list("text", flat=True).iterator():
    KNOWN_CODES.update(re.sub(r"\s+", " ", m.upper()) for m in re.findall(
        r"\b(?:MGTA|MGTF|MGTP|MGT|CSE)\s?\d{3}[A-Z]?\b", text or ""))
CAREERS = electives.load_careers()
INDUSTRIES = electives.load_industries()


# ---------------------------------------------------------------------------
# Resources: a hundred questions in each voice
# ---------------------------------------------------------------------------

# (kind, plain question) -- kinds: faq, career, out; faq2 pairs are added per persona
BASE_TOPICS = [
    ("faq", "how to register for classes"), ("faq", "when does enrollment open"),
    ("faq", "how do i drop a class"), ("faq", "what happens if i drop after week 2"),
    ("faq", "how do i wait-list a class"), ("faq", "what is RadyReady"),
    ("faq", "how many units is full time"), ("faq", "when can i see my grades"),
    ("faq", "how do i check my grades"), ("faq", "what gpa keeps me in good standing"),
    ("faq", "what is academic probation"), ("faq", "how do i get a transcript"),
    ("faq", "can i take a class pass no pass"), ("faq", "how do i appeal a grade"),
    ("faq", "when is tuition due"), ("faq", "how much is msba tuition"),
    ("faq", "how do i pay my bill"), ("faq", "is there a payment plan"),
    ("faq", "how do i clear a hold on my account"), ("faq", "how do i set up direct deposit"),
    ("faq", "what fees are there besides tuition"), ("faq", "can i get a refund if i leave"),
    ("faq", "are there scholarships for msba students"), ("faq", "how does financial aid work"),
    ("faq", "how do i raise an IT ticket"), ("faq", "how do i reset my password"),
    ("faq", "i am locked out of two step login"), ("faq", "how do i connect to campus wifi"),
    ("faq", "how do i print on campus"), ("faq", "what laptop do i need"),
    ("faq", "how do i use the vpn"), ("faq", "how do i get microsoft office"),
    ("faq", "where is the rady help desk"), ("faq", "canvas is not loading"),
    ("faq", "how do i get a ucsd email"), ("faq", "how do i use canvas"),
    ("faq", "what is tritonlink"), ("faq", "what is vmock"),
    ("faq", "how do i book a study room"), ("faq", "do i have zoom"),
    ("faq", "how do i get a student id card"), ("faq", "how do i get a parking permit"),
    ("faq", "is there a shuttle to campus"), ("faq", "where can i find housing"),
    ("faq", "how do i get a bus pass"), ("faq", "where is the gym"),
    ("faq", "how do i see a counsellor"), ("faq", "what health insurance do students get"),
    ("faq", "how do i get a social security number"), ("faq", "can i work on campus on an F-1"),
    ("faq", "how does OPT work"), ("faq", "what is CPT"), ("faq", "who do i contact about my I-20"),
    ("faq", "can i travel abroad during break"), ("faq", "what is the difference between the 11 and 17 month msba"),
    ("faq", "when does orientation start"), ("faq", "what is the summer course schedule"),
    ("faq", "how many electives are in the msba"), ("faq", "what is the capstone"),
    ("faq", "can i take a leave of absence"), ("faq", "how do i book msba advising"),
    ("faq", "who is my academic advisor"), ("faq", "what is the datacamp assignment"),
    ("faq", "is there math and stats prep"), ("faq", "how do i change my preferred name"),
    ("career", "how long should my resume be"), ("career", "how do i prepare for a data analyst interview"),
    ("career", "should i negotiate my first offer"), ("career", "how do i reach out to alumni on linkedin"),
    ("career", "what do i say about salary expectations"), ("career", "how do i explain a gap in my resume"),
    ("career", "is a cover letter necessary"), ("career", "how do i prepare a case interview"),
    ("career", "what should my linkedin headline say"), ("career", "how do i follow up after an interview"),
    ("career", "how do i find an internship"), ("career", "which analytics roles hire the most"),
    ("out", "who won the super bowl"), ("out", "what is the weather tomorrow"),
    ("out", "write my essay for me"), ("out", "tell me a joke"),
    ("out", "what is the capital of france"), ("out", "recommend a netflix series"),
    ("out", "integrate x squared from 0 to 3 for me"), ("out", "who is the president of france"),
]

FOLLOW_UPS = [  # (first turn, the answer to the bot's clarifying question)
    ("how do i reset my password", "the campus one"),
    ("i can't log in", "canvas"),
    ("how much does it cost", "tuition for the msba"),
    ("when is the deadline", "to drop a class"),
    ("how do i get help", "with my laptop"),
    ("where do i submit it", "my petition to drop after week 2"),
]

NEWCOMER_EXTRA = [
    ("faq", "i arrive in san diego in august, what do i need to do before classes start"),
    ("faq", "how do i get my student visa documents checked when i arrive"),
    ("faq", "is on campus housing available for grad students and how do i apply"),
    ("faq", "how do i open a us bank account as an international student"),
    ("faq", "do i need a us phone number for two step login"),
    ("faq", "how do i set up my ucsd account before i get to campus"),
    ("faq", "what should i bring to orientation"),
    ("faq", "how do i get from the airport to campus"),
    ("faq", "is there an international student orientation separate from msba orientation"),
    ("faq", "how do i enroll in the student health insurance plan"),
]


def _typo(text, rng):
    words = text.split()
    if len(words) < 3:
        return text
    i = rng.randrange(len(words))
    w = words[i]
    if len(w) > 4:
        j = rng.randrange(1, len(w) - 2)
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def voice(persona, kind, question, rng):
    """One question, in one student's voice."""
    q = question
    if kind == "out" and persona != "rusher":
        # An off-topic question stays off-topic. The planner's tail ("I want
        # to plan ahead for the whole year, so dates matter") glued to "tell
        # me a joke" made a programme question out of it, and the bot was
        # right to answer.
        return q + "?"
    if persona == "newcomer":
        openers = ["sorry if this is obvious, but ", "hi! im an incoming international student, ",
                   "quick question before i arrive - ", "hello, new msba student here. ", ""]
        return rng.choice(openers) + q + rng.choice(["?", "?", " please?", " :)"])
    if persona == "rusher":
        q = q.lower().replace("how do i ", "").replace("what is ", "").replace("where is ", "where ")
        q = q.replace("can i ", "").replace("is there ", "").replace(" the ", " ")
        q = re.sub(r"\b(please|i am|i)\b", "", q).strip()
        q = _typo(q, rng) if rng.random() < 0.4 else q
        return q
    if persona == "planner":
        tails = [" For the Fall quarter specifically, and where is that published?",
                 " Please include the exact deadline or policy and the page it comes from.",
                 " I want to plan ahead for the whole year, so dates matter.",
                 " If it differs between the 11-month and 17-month tracks, please say so."]
        return q[0].upper() + q[1:] + rng.choice(tails)
    if persona == "switcher":
        openers = ["i've been working in marketing for 6 years and im switching into analytics. ",
                   "coming back to school after a decade in industry - ", "career changer here: ", ""]
        return rng.choice(openers) + q + "?"
    if persona == "worrier":
        openers = ["im kind of panicking, ", "sorry i know i keep asking things but ",
                   "ok so i might have messed up, ", "this is probably nothing but "]
        tails = ["?? will that be a problem", "? i really dont want to fall behind",
                 "? what happens if i get this wrong", "?? please tell me its fixable"]
        return rng.choice(openers) + q + rng.choice(tails)
    return q


def resources_cases(persona):
    rng = random.Random(f"resources-{persona}")
    topics = list(BASE_TOPICS)
    if persona == "newcomer":
        topics = NEWCOMER_EXTRA + topics
    if persona == "switcher":
        # career-first: every career question twice, in two voices
        topics = [t for t in topics if t[0] == "career"] * 2 + [t for t in topics if t[0] != "career"]
    rng.shuffle(topics)
    cases = []
    # 94 single turns; a voice has more than one way to say a thing, so a
    # short topic list is walked twice with different openers and typos.
    while len(cases) < 94:
        kind, q = topics[len(cases) % len(topics)]
        cases.append((kind, voice(persona, kind, q, rng), None))
    for first, follow in FOLLOW_UPS:
        cases.append(("faq2", voice(persona, "faq", first, rng), follow))
    assert len(cases) == 100, len(cases)
    return cases


HEDGES = ("i don't have specific information", "i do not have specific information",
          "i don't have that information in the")


def check_resources(kind, reply, seconds, follow=None):
    body = reply.body or ""
    lowered = body.lower()
    problems = []
    if not body.strip():
        problems.append("empty body")
    if body.lstrip().startswith("{"):
        problems.append("raw JSON leaked")
    if seconds > BUDGET_SECONDS:
        problems.append(f"over budget: {seconds:.0f}s")
    if kind == "out":
        if not reply.refused:
            problems.append(f"answered instead of refusing ({reply.model_note})")
        return problems
    if reply.refused:
        problems.append("refused an in-scope question")
        return problems
    if kind == "faq2" and reply.model_note == "clarify":
        problems.append("asked again after the student answered")
    if reply.model_note in ("web", "llm+web") and "web search" not in lowered:
        problems.append("used the web without saying so")
    if reply.model_note in ("llm", "llm+web") and "source" not in lowered:
        problems.append("answered from the corpus without listing a source")
    if reply.model_note == "web" and "source:" in lowered:
        problems.append("a web answer cites Rady material")
    if reply.model_note != "clarify" and any(h in lowered for h in HEDGES):
        problems.append("hedged instead of answering or declining")
    if re.search(r"\[W\s?\d", body) and "read from the web" not in lowered:
        problems.append("a W marker with no web sources listed")
    return problems


def run_resources(persona, llm, show):
    results = []
    for index, (kind, question, follow) in enumerate(resources_cases(persona), 1):
        started = time.time()
        try:
            if kind == "career":
                reply = answer_career(llm, question, [])
            else:
                reply = answer_faq(llm, question, [])
                if kind == "faq2":
                    history = [{"role": "user", "content": question},
                               {"role": "assistant", "content": reply.body}]
                    started = time.time()
                    reply = answer_faq(llm, follow, history)
        except Exception as exc:  # noqa: BLE001
            results.append({"n": index, "kind": kind, "question": question, "follow": follow,
                            "problems": [f"crashed: {type(exc).__name__}: {exc}"],
                            "note": "-", "seconds": time.time() - started, "body": ""})
            print(f"[CRASH] {index:3} {kind:6} {question!r}: {exc}", flush=True)
            continue
        seconds = time.time() - started
        problems = check_resources(kind, reply, seconds, follow)
        results.append({"n": index, "kind": kind, "question": question, "follow": follow,
                        "problems": problems, "note": reply.model_note,
                        "seconds": round(seconds, 1), "body": reply.body[:1500]})
        status = "FAIL" if problems else "ok  "
        print(f"[{status}] {index:3} {seconds:5.1f}s {reply.model_note:8} {kind:5} {question[:70]!r}"
              + (f" -> {follow!r}" if follow else "")
              + (f"  :: {'; '.join(problems)}" if problems else ""), flush=True)
        if problems or show:
            print("      " + (reply.body or "")[:500].replace("\n", "\n      "), flush=True)
    return results


# ---------------------------------------------------------------------------
# Courses: a hundred conversations in each voice
# ---------------------------------------------------------------------------

DETERMINISTIC_OK = {"curated", "curated+industry", "intake", "plan", "review",
                    "industry-roles", "industry-menu", "compare-roles",
                    "compare-industries", "catalog-rag", "degree-fact", "why",
                    "greeting", "small-talk", "menu-which", "catalog-overview",
                    "careers-list", "situation", "quarter"}
BAD_NOTES = {"unclear", "degraded", "unclear-opening"}

UNCOVERED = ["esports analyst", "sommelier", "firefighter", "nurse practitioner",
             "race car driver", "sports scientist", "climate risk modeller",
             "aerospace engineer", "chef", "yoga instructor"]
FACTS = ["does MGTA 464 have prerequisites", "how many units is MGTA 461",
         "what does MGTA 458 cover", "when is MGTA 463 offered", "whats MGTA 456 about",
         "is MGTA 479 offered in winter", "who teaches MGTA 452", "how is MGTA 459 graded"]


def _title(role_id, rng):
    role = CAREERS[role_id]
    shared = set()
    owners = {}
    for rid, r in CAREERS.items():
        for t in r.get("titles") or []:
            owners.setdefault(t.lower(), []).append(rid)
    shared = {t for t, rids in owners.items() if len(rids) > 1}
    titles = [t for t in (role.get("titles") or []) if t.lower() not in shared] or role["titles"]
    return rng.choice(titles)


def role_turns(persona, role_id, track, load, rng):
    """A conversation that should end in a legal plan for `role_id`."""
    title = _title(role_id, rng)
    track_words = {"11 month": ["11 month", "the one year one", "11", "the shorter track", "11 mo"],
                   "17 month": ["17 month", "the longer one", "17", "seventeen months", "the 17-month track"]}
    load_words = {"light": ["light", "not too heavy please", "lighter", "take it easy"],
                  "moderate": ["moderate", "somewhere in the middle", "normal", "balanced is fine"],
                  "heavy": ["heavy", "front load it", "as much as possible early", "heavy please"]}
    t = rng.choice(track_words[track]); l = rng.choice(load_words[load])
    if persona == "newcomer":
        return [f"hi! im starting the msba this summer and i think i want to be a {title} eventually?",
                f"oh sorry, {t}", f"hmm {l} i think"]
    if persona == "rusher":
        # One word each, as a rusher types: the plain track number and the
        # plain load word -- never the first word of a longer phrase.
        return [title, rng.choice(["11", "11 mo", "11 month"] if track == "11 month"
                                  else ["17", "17 mo", "17 month"]), load]
    if persona == "planner":
        return [f"I'm on the {track} track and aiming to become a {title}. Please lay out the full plan.",
                l]
    if persona == "switcher":
        return [f"career changer here - after 6 years in operations i want to move into a {title} role",
                t, l]
    if persona == "worrier":
        worry = {"light": "i dont want to burn out", "moderate": "i think thats safest",
                 "heavy": "i want to get the hard part done early"}[load]
        return [f"ok so i think i want to be a {title} but im worried i dont have the background for it",
                f"im on the {t} one if that matters", f"maybe {l}? {worry}"]
    return [title, t, l]


def courses_cases(persona):
    rng = random.Random(f"courses-{persona}")
    cases = []
    roles = list(CAREERS)
    # 56 plans: every role on both tracks, four load words, persona voice
    for i, role_id in enumerate(roles * 4):
        if len(cases) >= 56:
            break
        track = "11 month" if i % 2 == 0 else "17 month"
        load = ["light", "moderate", "heavy"][i % 3]
        cases.append({"kind": "plan", "turns": role_turns(persona, role_id, track, load, rng),
                      "expect": {"goal": role_id, "track": track, "load": load},
                      "needs_model": persona != "rusher"})
    # 12 industries by name, in the voice
    for ind in INDUSTRIES * 2:
        if len(cases) >= 68:
            break
        # The plain field name: "Financial Services", not "Financial Services &
        # Fintech" -- "fintech" is a profile alias by design, so pasting the
        # full label names a role as well as a field.
        label = (ind.get("short_label") or ind["label"]).split(" (")[0].split(" / ")[0].split(" & ")[0]
        if ind["id"] == "other":
            label = "media"          # the catch-all's own name is not a field anyone names
        cases.append({"kind": "industry", "turns": [voice(persona, "faq", f"i want to work in {label.lower()}", rng).rstrip("?")],
                      "expect": {"note_in": ["industry-roles", "industry-menu"]},
                      "needs_model": True})
    # 10 uncovered jobs: refused, no course named
    for job in UNCOVERED:
        cases.append({"kind": "uncovered", "turns": [voice(persona, "faq", f"i want to be a {job}", rng).rstrip("?")],
                      "expect": {"refusal": True}, "needs_model": True})
    # 8 factual course questions
    for q in FACTS:
        cases.append({"kind": "factual", "turns": [voice(persona, "faq", q, rng)],
                      "expect": {"note_in": ["catalog-rag", "refusal"]}, "needs_model": True})
    # 8 situational / what-if / walk-through
    extras = [
        {"kind": "situation", "turns": ["i switched from 17 month to 11 month and im already in winter, can i finish on time?"],
         "expect": {"note_in": ["situation"]}, "needs_model": True},
        {"kind": "situation", "turns": ["im on the 11 month track, in spring now, whats left for me"],
         "expect": {"note_in": ["situation"]}, "needs_model": True},
        {"kind": "walk", "turns": ["data scientist", "17 month", "moderate", "walk me through it", "next quarter", "make this one lighter"],
         "expect": {"goal": "data-scientist", "track": "17 month", "load": "moderate"}, "needs_model": True},
        {"kind": "walk", "turns": ["product analyst, 11 month", "show me the plan", "why is MGTA 464 in there"],
         "expect": {"goal": "product-analyst", "track": "11 month"}, "needs_model": True},
        {"kind": "menu", "turns": ["no idea what i want to do yet", "the health one", "the analyst one"],
         "expect": {"note_in": ["curated", "menu-which", "industry-roles"]}, "needs_model": True},
        {"kind": "menu", "turns": ["not sure yet", "2", "whats the difference between the first two"],
         "expect": {"note_in": ["compare-roles", "curated", "industry-roles"]}, "needs_model": True},
        {"kind": "whatif", "turns": ["data scientist, 11 month", "moderate", "what if i had gone for the 17 month instead"],
         "expect": {"goal": "data-scientist", "track": "11 month"}, "needs_model": True},
        {"kind": "greeting", "turns": ["hi"], "expect": {"note_in": ["greeting"]}, "needs_model": False},
    ]
    for e in extras:
        e["turns"] = [voice(persona, "faq", t, rng).rstrip("?") if e["kind"] in ("situation",) else t for t in e["turns"]]
        cases.append(e)
    # 6 edge
    for t in ["", "!!!", "a" * 400, "MGTA 999", "python 9", "swap MGTA 461 for MGTA 463"]:
        cases.append({"kind": "edge", "turns": [t], "expect": {}, "needs_model": True})
    assert len(cases) == 100, len(cases)
    return cases


def sane(replies):
    for r in replies:
        if not (r.body or "").strip():
            return "empty reply"
        if re.search(r"(?:[:=(\[]\s*|\bis\s+)None\b|NoneType", r.body):
            return "a Python None reached the page"
        for b in r.quick_replies or []:
            if not isinstance(b, dict) or not b.get("send"):
                return f"bad button {b!r}"
    return None


def check_courses(case, replies, conversation):
    problems = []
    bad = sane(replies)
    if bad:
        problems.append(bad)
    expect = case["expect"]
    last = replies[-1]
    if expect.get("refusal"):
        if last.model_note not in ("no-profile", "industry-menu", "out-of-scope"):
            problems.append(f"uncovered job routed to {last.model_note}")
        named = [c for c in DISPLAY if c in last.body]
        if named:
            problems.append(f"named courses on refusal: {named[:2]}")
        return problems
    if expect.get("note_in") and last.model_note not in expect["note_in"]:
        problems.append(f"note {last.model_note!r} not in {expect['note_in']}")
    if not case["needs_model"]:
        for turn, reply in zip(case["turns"], replies):
            if reply.model_note in BAD_NOTES:
                problems.append(f"{reply.model_note!r} in reply to {turn!r}")
    stored = planner.load_session_intake(conversation) or {}
    if expect.get("goal") and stored.get("goals") != [expect["goal"]]:
        problems.append(f"goal {stored.get('goals')} != {[expect['goal']]}")
    if expect.get("track") and stored.get("track") != expect["track"]:
        problems.append(f"track {stored.get('track')!r} != {expect['track']!r}")
    if expect.get("load") and planner.load_is_a_choice(expect.get("track", "11 month")) \
            and stored.get("workload") != expect["load"]:
        problems.append(f"load {stored.get('workload')!r} != {expect['load']!r}")
    if expect.get("goal") and stored.get("goals") and stored.get("track"):
        plan = planner.build_for(stored, frozenset())
        total = plan["totals"]["total"] + plan["totals"]["completed"]
        if total != 50:
            problems.append(f"degree does not close: {total}")
        if plan["unfilled"]:
            problems.append(f"unfilled {plan['unfilled']}")
        ids = [r["courseId"] for q in plan["quarters"] for r in q["courses"] if r["courseId"]]
        if len(ids) != len(set(ids)):
            problems.append("duplicate course")
        if any(i not in CAT for i in ids):
            problems.append("course outside catalog")
    for reply in replies:
        for code in re.findall(r"\b(?:MGTA|MGTF|MGTP|MGT|CSE)\s?\d{3}[A-Z]?\b", reply.body or ""):
            plain = re.sub(r"\s+", " ", code.upper())
            if plain not in KNOWN_CODES and plain.rstrip("R") not in KNOWN_CODES:
                # `MGTA 999` is the student's own nonsense being quoted back
                if not any(code in t for t in case["turns"]):
                    problems.append(f"a course code not in the catalog: {code}")
                    break
    return problems


def run_courses(persona, llm, show):
    user, _ = User.objects.get_or_create(username=f"persona-{persona}")
    results = []
    for index, case in enumerate(courses_cases(persona), 1):
        conversation = Conversation.objects.create(user=user, destination="courses",
                                                   title=f"{persona} {index}")
        started = time.time()
        replies = []
        try:
            for turn in case["turns"]:
                replies.append(orchestrator.answer(llm, conversation, turn, []))
            problems = check_courses(case, replies, conversation)
        except Exception as exc:  # noqa: BLE001
            problems = [f"crashed: {type(exc).__name__}: {exc}"]
        finally:
            conversation.delete()
        seconds = time.time() - started
        if seconds > BUDGET_SECONDS * len(case["turns"]):
            problems.append(f"over budget: {seconds:.0f}s")
        note = replies[-1].model_note if replies else "-"
        results.append({"n": index, "kind": case["kind"], "turns": case["turns"],
                        "problems": problems, "note": note, "seconds": round(seconds, 1),
                        "body": (replies[-1].body[:1200] if replies else "")})
        status = "FAIL" if problems else "ok  "
        print(f"[{status}] {index:3} {seconds:5.1f}s {note:15} {case['kind']:9} {case['turns'][0][:60]!r}"
              + (f"  :: {'; '.join(problems)}" if problems else ""), flush=True)
        if problems or show:
            for turn, reply in zip(case["turns"], replies):
                print(f"      > {turn!r}\n      {(reply.body or '')[:300].replace(chr(10), ' ')}", flush=True)
    return results


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot", choices=["resources", "courses"], required=True)
    parser.add_argument("--persona", choices=PERSONAS + ("all",), required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    llm = get_llm()
    personas = PERSONAS if args.persona == "all" else (args.persona,)
    exit_code = 0
    for persona in personas:
        print(f"\n{'=' * 78}\n{args.bot.upper()} / {persona.upper()}\n{'=' * 78}", flush=True)
        started = time.time()
        results = (run_resources if args.bot == "resources" else run_courses)(persona, llm, args.show)
        failures = [r for r in results if r["problems"]]
        notes = {}
        for r in results:
            notes[r["note"]] = notes.get(r["note"], 0) + 1
        summary = {"bot": args.bot, "persona": persona, "passed": len(results) - len(failures),
                   "total": len(results), "seconds": round(time.time() - started),
                   "notes": notes, "results": results}
        print(f"\n{args.bot}/{persona}: {summary['passed']}/{summary['total']} passed "
              f"in {summary['seconds']}s; replies by kind: {notes}", flush=True)
        for r in failures:
            print(f"  - #{r['n']} [{r['kind']}] {(r.get('question') or r['turns'][0])[:70]!r}: "
                  f"{'; '.join(r['problems'])}", flush=True)
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            (Path(args.out) / f"{args.bot}_{persona}.json").write_text(json.dumps(summary, indent=1))
        if failures:
            exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
