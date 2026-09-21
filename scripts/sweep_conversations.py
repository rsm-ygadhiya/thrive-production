"""100 end-to-end course-recommender conversations against the REAL model.

Not a pytest: every turn costs a live call on the gateway budget, and the
suite must stay offline. Run this after changing routing, the taxonomy, the
bundles or the planner -- it exercises the whole surface at once.

    cd backend && PYTHONPATH=$PWD uv run python ../scripts/sweep_conversations.py

Covers all fourteen curated roles on BOTH tracks to a finished plan, the six
industries by number and by title through the menu, industries in a student's
own words, every load, change-of-mind mid-conversation, the quarter
walk-through, eight roles nobody curates, ten factual course questions, ten
edge cases, and eight resources-bot questions.

Plans are asserted, not eyeballed: 50 units exactly, no unfilled slot, no
duplicate, no course outside the catalog, and at least half the electives
drawn from courses that role explicitly boosts.

Two traps this has already caught, both worth knowing before adding a check:

* MGTA 495 is a Special Topics SHELL -- four sections (AIPA, GENAI, HC,
  MKTG) sharing one `code` across four `id`s. Plan rows carry ids, so
  validating them against codes reports every Special Topics placement as a
  course that does not exist.
* The degraded reply legitimately contains the English word "None" ("None of
  that needs the model"), so a bare substring test for a leaked Python value
  fires on correct prose.

Exits non-zero on any failure, so it can gate a release.
"""
import os, django, time, sys, re, collections
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()
from django.contrib.auth.models import User
from rsm_thrive.models import Conversation
from rsm_thrive.services.courses import orchestrator, planner, electives
from rsm_thrive.services.llm import get_llm

u, _ = User.objects.get_or_create(username="sweep100")
LLM = get_llm()
CAT = {c["id"]: c for c in electives.load_catalog()}
DISPLAY = {c["code"] for c in electives.load_catalog()}
CAREERS = electives.load_careers()
results = []

def run(name, turns, check, dest="courses"):
    c = Conversation.objects.create(user=u, destination=dest, title=name[:60])
    t0 = time.time(); rs = []
    try:
        for t in turns:
            rs.append(orchestrator.answer(LLM, c, t, []))
        problem = check(rs, c)
    except Exception as exc:
        problem = f"EXCEPTION {type(exc).__name__}: {str(exc)[:70]}"
    results.append((name, problem, time.time() - t0,
                    rs[-1].model_note if rs else "-"))
    c.delete()

def sane(rs, c):
    for r in rs:
        if not (r.body or "").strip(): return "empty reply"
        if re.search(r"(?:[:=(\[]\s*|\bis\s+)None\b|NoneType", r.body):
            return "a Python None reached the page"
        for b in r.quick_replies or []:
            if not isinstance(b, dict) or not b.get("send"): return f"bad button {b!r}"
    return None

def plan_ok(rs, c):
    bad = sane(rs, c)
    if bad: return bad
    a = planner.load_session_intake(c)
    if not a.get("goals"): return "no goal stored"
    p = planner.build_for(a, frozenset())
    if p["totals"]["total"] != 50: return f"units={p['totals']['total']}"
    if p["unfilled"]: return f"unfilled={p['unfilled']}"
    ids = [r["courseId"] for q in p["quarters"] for r in q["courses"]]
    if len(ids) != len(set(ids)): return "duplicate course"
    ghost = [i for i in ids if i not in CAT]
    if ghost: return f"not in catalog: {ghost}"
    # the classes must actually serve the goal
    goal = a["goals"][0]
    boosts = set(CAREERS[goal].get("boost_courses") or {})
    el = [i for i, r in zip(ids, [r for q in p["quarters"] for r in q["courses"]])
          if r["requirement"] != "Core"]
    on = sum(1 for i in el if CAT[i]["code"] in boosts or i in boosts)
    if el and on / len(el) < 0.5: return f"only {on}/{len(el)} on-goal"
    return None

def note_is(*ok):
    def f(rs, c):
        bad = sane(rs, c)
        return bad or (None if rs[-1].model_note in ok
                       else f"note={rs[-1].model_note}")
    return f

def refuses(rs, c):
    """Safety is the gate; the route is not.

    An uncovered role must never be answered with a course. Whether it is
    refused outright or handed the industry menu is a classifier's call, and
    measured at 17/18 -- so a hard gate here failed one run in six for a
    reply that was less direct but still honest. The graded eval scores that
    ratio; this only checks nothing was invented."""
    bad = sane(rs, c)
    if bad: return bad
    r = rs[-1]
    if r.model_note not in ("no-profile", "industry-menu", "out-of-scope"):
        return f"unexpected route for an uncovered role ({r.model_note})"
    named = [x for x in DISPLAY if x in r.body]
    return f"named courses: {named[:2]}" if named else None

ROLE_WORDS = ["data scientist", "product analyst", "bi analyst", "marketing analyst",
              "analytics consultant", "fraud analyst", "supply chain analyst",
              "pricing analyst", "decision scientist", "ml engineer",
              "healthcare data analyst", "financial analyst", "analytics engineer",
              "business analyst"]

# 1-28: every role, both tracks, to a finished plan
for role in ROLE_WORDS:
    for track in ("11 month", "17 month"):
        run(f"{role} / {track}", [role, track, "moderate"], plan_ok)

# 29-34: menu by NUMBER, industry then role, to a plan
for n in "123456":
    run(f"menu number {n}", ["no idea", n, "1", "11 month", "moderate"], plan_ok)

# 35-40: menu by TITLE rather than number
for ind in electives.load_industries():
    title = electives.top_titles_for(ind["id"], 1)[0][0]
    run(f"menu title: {ind['id']}", ["no idea", ind["label"],
                                     orchestrator.title_case(title),
                                     "11 month", "moderate"], plan_ok)

# 41-52: industries in a student's own words
for w in ["tech", "software", "banking", "consulting", "advisory",
          "biotech", "pharma", "retail", "cpg", "gaming", "energy"]:
    run(f"industry word: {w}", [w], note_is("industry-roles"))
# "fintech" matches the Financial Analytics profile as well as the industry,
# so it is a combination rather than a plain industry turn -- role first,
# industry ranking after. See test_industry_flow.
run("industry word: fintech", ["fintech"], note_is("curated+industry", "curated"))

# 53-58: loads and change of mind
for role, load in [("data scientist", "light"), ("bi analyst", "heavy"),
                   ("ml engineer", "light")]:
    run(f"{role}/{load}", [role, "11 month", load], plan_ok)
run("change role mid-chat", ["data scientist", "actually pricing analyst",
                             "11 month", "moderate"], plan_ok)
run("change track mid-chat", ["bi analyst", "11 month", "actually 17 month",
                              "moderate"], plan_ok)
run("two roles at once", ["data scientist and pricing analyst", "11 month",
                          "moderate"], plan_ok)

# 59-64: the walk-through
for role in ["data scientist", "healthcare data analyst", "ml engineer"]:
    run(f"walk: {role}", [role, "11 month", "moderate", "walk me through it"],
        note_is("review"))
    run(f"walk+next: {role}", [role, "11 month", "moderate",
                               "walk me through it", "next quarter"],
        note_is("review"))

# 65-72: roles nobody curates
for role in ["esports analyst", "sommelier", "wildlife biologist", "astronaut",
             "climate risk modeller", "yoga instructor", "race car driver",
             "underwater welder"]:
    run(f"uncovered: {role}", [f"i want to be a {role}"], refuses)

# 73-82: factual course questions
for q in ["does MGTA 464 have prerequisites?", "how many units is MGTA 461",
          "what does MGTA 458 cover?", "what courses do you have",
          "what electives are available", "is MGTA 495 offered in spring",
          "what is MGTF 405", "what are the core courses",
          "how many units is the degree", "what is MGTA 402 about"]:
    run(f"factual: {q[:34]}", [q], sane)

# 83-92: edge cases
EDGE = [
    ("blank", ["   "]), ("gibberish", ["zxcvbn qwerty asdf"]),
    ("emoji", ["🎓📊🤖"]), ("very long", ["i want " + "really " * 150 + "data scientist"]),
    ("bare number, no menu", ["4"]), ("huge number", ["999999"]),
    ("negative number", ["-3"]), ("injection", ["'; DROP TABLE courses; --"]),
    ("html", ["<script>alert(1)</script>"]), ("unicode rtl", ["‮malicious"]),
]
for name, turns in EDGE:
    run(f"edge: {name}", turns, sane)

# 93-100: the resources bot, which must not be disturbed by any of this
for q in ["what happens if I drop a course after week 2",
          "who has to approve a petition", "where do I request a laptop",
          "what is the tuition per unit", "how do I take a leave of absence",
          "what is VMock", "when is the add deadline",
          "what is the weather in san diego"]:
    run(f"resources: {q[:32]}", [q], sane, dest="resources")

fails = [r for r in results if r[1]]
print(f"\n{'='*80}\n{len(results)} CONVERSATIONS\n{'='*80}")
for name, problem, dt, note in results:
    if problem:
        print(f"[FAIL] {dt:5.1f}s {note:16s} {name:36s} {problem}")
by = collections.Counter(n for _, p, _, n in results if not p)
print(f"\npassing routes: {dict(by)}")
slow = sorted(results, key=lambda r: -r[2])[:3]
print(f"slowest: " + ", ".join(f"{n}={d:.1f}s" for n, _, d, _ in slow))
print(f"\n{len(results)-len(fails)}/{len(results)} passed")
sys.exit(1 if fails else 0)
