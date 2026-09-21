"""100 Resources-bot questions against the REAL model, each checked for shape.

Not a pytest: every question costs live calls on the gateway budget. Run it
after changing the Resources flow, its skills, retrieval or the corpus:

    cd backend && PYTHONPATH=$PWD uv run python ../scripts/sweep_resources.py [--show]

There is no ground truth for a hundred free-text answers, so what is checked
is what CAN be checked without one -- the promises the flow makes:

* it always answers (no crash, no empty body, no raw JSON leaking through);
* an out-of-scope question is refused, an in-scope one is not;
* an answer that used the web says so and lists what it read; one that used
  the corpus lists the documents it cited and nothing it did not;
* no answer hedges with "I don't have specific information" -- the writer is
  told to answer or decline, never to pad;
* a two-turn exchange (a clarifying question, then the student's one-word
  answer) ends in an answer, not another question;
* every turn finishes inside the gateway's turn budget.

Bodies of every failure are printed so a person can read what happened.
Exits non-zero on any failure.
"""

import os
import re
import sys
import time

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from rsm_thrive.services.llm import get_llm  # noqa: E402
from rsm_thrive.services.resources.bot import answer_career, answer_faq  # noqa: E402

SHOW = "--show" in sys.argv
BUDGET_SECONDS = 60

# kind -> (question, [follow-up turn])
CASES = [
    # enrollment and registration
    ("faq", "how to register for classes"),
    ("faq", "when does enrollment open for fall"),
    ("faq", "can i add a class after the quarter starts"),
    ("faq", "how do i drop a class"),
    ("faq", "what happens if i drop a course after week 2"),
    ("faq", "how do i wait-list a class"),
    ("faq", "what is RadyReady and do i have to finish it"),
    ("faq", "how many units do i need to be full time"),
    ("faq", "can i take a class at another UC campus"),
    ("faq", "how do i get a W removed"),
    # grades and academics
    ("faq", "when can i get grades"),
    ("faq", "how do i check my grades"),
    ("faq", "what gpa do i need to stay in good standing"),
    ("faq", "what happens if i'm put on academic probation"),
    ("faq", "how do i get a transcript"),
    ("faq", "can i take a course pass/no pass"),
    ("faq", "what is the incomplete grade policy"),
    ("faq", "how do i appeal a grade"),
    # fees and money
    ("faq", "when is tuition due"),
    ("faq", "how much is tuition for the msba"),
    ("faq", "how do i pay my bill"),
    ("faq", "is there a payment plan"),
    ("faq", "what is a hold on my account and how do i clear it"),
    ("faq", "how do i set up direct deposit"),
    ("faq", "what fees am i charged besides tuition"),
    ("faq", "can i get a refund if i leave the program early"),
    ("faq", "are there fellowships or scholarships for msba students"),
    ("faq", "how does financial aid work for grad students"),
    # IT and accounts
    ("faq", "how to raise a IT ticket"),
    ("faq", "how do i reset my password"),
    ("faq", "i'm locked out of two-step login, what do i do"),
    ("faq", "how do i connect to the campus wifi"),
    ("faq", "how do i print on campus"),
    ("faq", "what laptop should i get for the msba"),
    ("faq", "how do i access the vpn from home"),
    ("faq", "how do i get microsoft office as a student"),
    ("faq", "where is the rady help desk"),
    ("faq", "my canvas isn't loading, who do i contact"),
    ("faq", "how do i get a ucsd email address"),
    ("faq", "how do i change my preferred name on canvas"),
    # tools and services
    ("faq", "how do i use canvas"),
    ("faq", "what is tritonlink"),
    ("faq", "what is vmock and how do i use it"),
    ("faq", "how do i book a study room"),
    ("faq", "do i have zoom as a student"),
    ("faq", "how do i get a student id card"),
    # campus life
    ("faq", "how do i get a parking permit"),
    ("faq", "is there a shuttle to campus"),
    ("faq", "where can i find housing near rady"),
    ("faq", "how do i get a free bus pass"),
    ("faq", "what food options are there near rady"),
    ("faq", "where is the gym and can grad students use it"),
    ("faq", "how do i see a counsellor"),
    ("faq", "what health insurance do students have"),
    # international
    ("faq", "how do i get a social security number as an international student"),
    ("faq", "can i work on campus on an F-1 visa"),
    ("faq", "how does OPT work"),
    ("faq", "what is CPT and how do i apply"),
    ("faq", "who do i contact about my I-20"),
    ("faq", "can i travel abroad during the break"),
    # programme specifics
    ("faq", "what's the difference between the 11 month and 17 month msba"),
    ("faq", "when does orientation start"),
    ("faq", "what is the summer course schedule for msba"),
    ("faq", "how many electives do i take in the msba"),
    ("faq", "what is the capstone"),
    ("faq", "can i take a leave of absence"),
    ("faq", "how do i book an appointment with msba advising"),
    ("faq", "who is my academic advisor"),
    ("faq", "what is the datacamp assignment before the program"),
    ("faq", "is there a math and statistics prep i need to complete"),
    # careers
    ("career", "how long should my resume be"),
    ("career", "how do i prepare for a data analyst interview"),
    ("career", "should i negotiate my first offer"),
    ("career", "how do i reach out to alumni on linkedin"),
    ("career", "what do i say when a recruiter asks about salary expectations"),
    ("career", "how do i explain a gap in my resume"),
    ("career", "is a cover letter necessary"),
    ("career", "how do i prepare a case interview for consulting"),
    ("career", "what should i put in my linkedin headline"),
    ("career", "how do i follow up after an interview"),
    # vague and short (should get an answer or one clarifying question, never the opening)
    ("faq", "fees?"),
    ("faq", "help with enrollment"),
    ("faq", "wifi"),
    ("faq", "i can't log in"),
    ("faq", "deadline"),
    ("faq", "where do i go for help"),
    # two-turn: a clarifying question, then a one-word answer
    ("faq2", "how do i reset my password", "the campus one"),
    ("faq2", "i can't log in", "canvas"),
    ("faq2", "how much does it cost", "tuition for the msba"),
    ("faq2", "when is the deadline", "to drop a class"),
    ("faq2", "how do i get help", "with my laptop"),
    ("faq2", "where do i submit it", "my petition to drop after week 2"),
    # out of scope: refused, plainly
    ("out", "who won the super bowl"),
    ("out", "what's the weather in san diego tomorrow"),
    ("out", "write my essay on supply chain analytics for me"),
    ("out", "tell me a joke"),
    ("out", "what is the capital of france"),
    ("out", "recommend a netflix series"),
    ("out", "solve this: integrate x^2 from 0 to 3"),
    ("out", "who is the president of the united states"),
]
assert len(CASES) == 100, len(CASES)

HEDGES = ("i don't have specific information", "i do not have specific information",
          "i don't have that information in the")


def check(kind, reply, seconds, follow=None):
    body = reply.body or ""
    problems = []
    if not body.strip():
        problems.append("empty body")
    if body.lstrip().startswith("{"):
        problems.append("raw JSON leaked")
    if seconds > BUDGET_SECONDS:
        problems.append(f"over budget: {seconds:.0f}s")
    lowered = body.lower()
    if kind == "out":
        if not reply.refused:
            problems.append(f"answered instead of refusing ({reply.model_note})")
        return problems
    if reply.refused:
        problems.append("refused an in-scope question")
        return problems
    if kind == "faq2" and follow is not None:
        if reply.model_note == "clarify":
            problems.append("asked again after the student answered")
    if reply.model_note in ("web", "llm+web"):
        if "web search" not in lowered:
            problems.append("used the web without saying so")
    if reply.model_note in ("llm", "llm+web"):
        if "source" not in lowered:
            problems.append("answered from the corpus without listing a source")
    if kind == "career" and reply.model_note == "advice" and "source" in lowered:
        problems.append("general advice cites a source it did not have")
    if reply.model_note == "web" and "source:" in lowered:
        problems.append("a web answer cites Rady material")
    if reply.model_note != "clarify" and any(h in lowered for h in HEDGES):
        problems.append("hedged instead of answering or declining")
    if re.search(r"\[W\s?\d", body) and "read from the web" not in lowered:
        problems.append("a W marker with no web sources listed")
    return problems


def main():
    llm = get_llm()
    failures, notes, slow = [], {}, []
    for index, case in enumerate(CASES, 1):
        kind, question = case[0], case[1]
        follow = case[2] if len(case) > 2 else None
        started = time.time()
        try:
            if kind == "career":
                reply = answer_career(llm, question, [])
            else:
                reply = answer_faq(llm, question, [])
                if kind == "faq2":
                    history = [{"role": "user", "content": question},
                               {"role": "assistant", "content": reply.body}]
                    started = time.time()          # the budget is per turn
                    reply = answer_faq(llm, follow, history)
        except Exception as exc:  # noqa: BLE001
            failures.append((kind, question, [f"crashed: {type(exc).__name__}: {exc}"], "", 0))
            print(f"[CRASH] {index:3} {kind:7} {question!r}: {exc}")
            continue
        seconds = time.time() - started
        problems = check(kind, reply, seconds, follow)
        notes[reply.model_note] = notes.get(reply.model_note, 0) + 1
        slow.append((seconds, question))
        status = "FAIL" if problems else "ok  "
        print(f"[{status}] {index:3} {seconds:5.1f}s {reply.model_note:8} {kind:7} {question!r}"
              + (f" -> {follow!r}" if follow else "")
              + (f"  :: {'; '.join(problems)}" if problems else ""))
        if problems or SHOW:
            print("      " + (reply.body or "")[:600].replace("\n", "\n      "))
        if problems:
            failures.append((kind, question, problems, reply.body, seconds))
    print("\n" + "=" * 78)
    print(f"{len(CASES) - len(failures)}/{len(CASES)} passed")
    print("by reply kind:", notes)
    slow.sort(reverse=True)
    print("slowest:", ", ".join(f"{q[:40]}={s:.1f}s" for s, q in slow[:3]))
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for kind, question, problems, _body, _s in failures:
            print(f"  - [{kind}] {question!r}: {'; '.join(problems)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
