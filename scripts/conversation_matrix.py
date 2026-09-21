"""A generated matrix of full conversations, every one asserted at the end.

The personas found what the scripted harnesses could not, because they used
the words students use. This scales that: every curated role and every
industry, each in several natural phrasings, driven to a finished plan
through varied track and load answers, with realistic detours dropped in --
a factual question, a change of mind, a thank-you, a comparison.

    cd backend && PYTHONPATH=$PWD uv run python ../scripts/conversation_matrix.py [--show N]

Each conversation is checked at the END rather than per turn, because a
student is owed the right plan, not the right route on the way there:

* the goal on file is the profile the phrasing meant
* the track and load on file are what was said, however it was said
* a plan builds, closes the degree, repeats nothing, invents nothing
* no deterministic turn -- a role, a track, a load, a menu pick -- was met
  with a clarification request or a degraded reply
* no reply, anywhere, named a course for a role we do not cover

Anomalies print their full transcript so a person can read what happened.
Exits non-zero on any failure.
"""
import argparse
import itertools
import os
import random
import sys
import time

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.contrib.auth.models import User  # noqa: E402

from rsm_thrive.models import Conversation  # noqa: E402
from rsm_thrive.services.courses import electives, orchestrator, planner  # noqa: E402
from rsm_thrive.services.llm import get_llm  # noqa: E402

random.seed(11)
LLM = get_llm()
USER, _ = User.objects.get_or_create(username="matrix")
CAT = {c["id"]: c for c in electives.load_catalog()}
DISPLAY = {c["code"] for c in electives.load_catalog()}
CAREERS = electives.load_careers()

# ---------------------------------------------------------------- phrasings
def _shared_titles():
    owners = {}
    for rid, r in CAREERS.items():
        for t in (r.get("titles") or []):
            owners.setdefault(t.lower(), set()).add(rid)
    return {t for t, rids in owners.items() if len(rids) > 1}

SHARED = _shared_titles()


def role_phrasings(role_id, role):
    # Titles the design document lists under MORE THAN ONE profile ("growth
    # analyst" is under Product and Marketing) are not openers here: free text
    # resolving one to either owner is correct, so asserting a single owner
    # would assert on a coin. Those get their own check in the test suite,
    # which asserts the confirmation note instead.
    titles = [t for t in (role.get("titles") or []) if t.lower() not in SHARED] \
             or list(role.get("titles") or [])
    aliases = list(role.get("aliases") or [])
    t0 = titles[0]
    t1 = titles[1] if len(titles) > 1 else t0
    alias = aliases[0] if aliases else t0
    return [
        f"i want to be a {t0}",
        f"thinking about {t1}, 11 month",
        f"{alias} sounds like my thing",
        f"hey! aiming for {t0} after this, im on the 17 mo track",
        f"probably {t1}? not 100% sure but lets go with it",
    ]

INDUSTRY_PHRASINGS = {
    "technology-software": ["i want to work in tech", "software companies", "a startup ideally"],
    "financial-services":  ["banking", "i like finance", "something in fintech"],
    "consulting":          ["consulting", "id like to do advisory work", "management consulting probably"],
    "healthcare":          ["healthcare", "biotech, im in san diego after all", "pharma"],
    "retail-cpg":          ["retail", "cpg brands", "e-commerce"],
    "other":               ["gaming", "energy sector", "government or defense"],
}
TRACKS = ["11 month", "17 month", "11 mo", "the 17-month one", "11", "seventeen"]
TRACK_OF = {"11 month": "11 month", "17 month": "17 month", "11 mo": "11 month",
            "the 17-month one": "17 month", "11": "11 month", "seventeen": "17 month"}
LOADS = ["moderate", "light please", "heavy", "lets do light", "ok moderate is fine",
         "heavy, i want it done early", "light, i have a part time job"]
LOAD_OF = {"moderate": "moderate", "light please": "light", "heavy": "heavy",
           "lets do light": "light", "ok moderate is fine": "moderate",
           "heavy, i want it done early": "heavy", "light, i have a part time job": "light"}
DETOURS = [  # (turn, may need the model)
    ("whats MGTA 464 about", True), ("how many units is the degree", False),
    ("thanks", False), ("does MGTA 466 need anything before it", True),
    ("why did you pick that", False), ("walk me through it", False),
]
DETERMINISTIC_NOTES_OK = {"curated", "curated+industry", "intake", "plan", "review",
                          "industry-menu", "industry-roles", "compare-roles",
                          "compare-industries", "small-talk", "degree-fact",
                          "why", "quarter", "catalog-rag", "catalog-overview",
                          "careers", "situation", "no-profile", "out-of-scope"}
BAD_NOTES = {"unclear", "degraded", "unclear-opening"}

# ------------------------------------------------------------- the matrix
def _typo(title):
    """One deletion in the longest word of a multi-word title -- the slop
    `role_match` allows. One-word titles are exempt from slop by design, so
    they are left alone."""
    words = title.split()
    if len(words) < 2:
        return title
    i = max(range(len(words)), key=lambda k: len(words[k]))
    w = words[i]
    if len(w) < 5:
        return title
    words[i] = w[:2] + w[3:]
    return " ".join(words)


def build_matrix_v2():
    """A second phrasing family: openers as questions, every fact in one
    turn, typos, capitals, industries on the long track, and detours INTO the
    menu after a role has already been named."""
    convos = []
    for k, (role_id, role) in enumerate(CAREERS.items()):
        titles = [t for t in (role.get("titles") or []) if t.lower() not in SHARED] \
                 or list(role.get("titles") or [])
        t0 = titles[0]
        aliases = role.get("aliases") or []
        alias = aliases[0] if aliases else t0
        load = LOADS[k % len(LOADS)]
        variants = [
            (f"what should i take to become a {t0}", ["17 month", load], "17 month"),
            (f"{t0}, 11 month, {load.split(',')[0]}", [], "11 month"),
            (t0.upper(), ["11 month", load], "11 month"),
            (f"which electives for {alias}?", ["11", load], "11 month"),
        ]
        if _typo(t0) != t0:
            variants.append((f"i wanna be a {_typo(t0)}", ["17 month", load], "17 month"))
        for i, (opener, rest, track) in enumerate(variants):
            convos.append({"name": f"v2/role/{role_id}/{i}", "turns": [opener] + rest,
                           "expect_goal": role_id, "expect_track": track,
                           "expect_load": LOAD_OF[load if rest else load],
                           "needs_model": False})
    # industries on the long track, heavy and light, picked by title
    for j, (ind_id, phr) in enumerate(INDUSTRY_PHRASINGS.items()):
        titles = electives.top_titles_for(ind_id, 10)
        for i in (1, 3):
            title, role_id = titles[i]
            load = ["heavy", "light please"][i % 2]
            convos.append({"name": f"v2/industry/{ind_id}/{i}",
                           "turns": ["i have no idea", phr[i % len(phr)],
                                     orchestrator.title_case(title), "17 month", load],
                           "expect_goal": role_id, "expect_track": "17 month",
                           "expect_load": LOAD_OF[load], "needs_model": False})
    # a role named, then a detour INTO the menu, then a different pick
    for a_id, a_title in [("data-scientist", "data scientist"),
                          ("bi-analyst", "bi analyst"),
                          ("consultant", "analytics consultant")]:
        titles = electives.top_titles_for("technology-software", 10)
        title, b_id = titles[2]
        convos.append({"name": f"v2/detour-menu/{a_id}",
                       "turns": [f"{a_title} 11 month", "show me the industries",
                                 "tech", orchestrator.title_case(title), "moderate"],
                       "expect_goal": b_id, "expect_track": "11 month",
                       "expect_load": "moderate", "needs_model": False})
    # compare inside the menu, then pick one of the two
    for ind_id in ["consulting", "healthcare", "retail-cpg"]:
        _t, second = electives.top_titles_for(ind_id, 2)[1]
        convos.append({"name": f"v2/compare-then-pick/{ind_id}",
                       "turns": ["not sure", ind_id.split("-")[0], "1 vs 2", "2",
                                 "11 month", "moderate"],
                       "expect_goal": second, "expect_track": "11 month",
                       "expect_load": "moderate", "needs_model": False})
    return convos


def build_matrix():
    convos = []
    # 1) every role x several phrasings, varied track + load
    for role_id, role in CAREERS.items():
        for i, opener in enumerate(role_phrasings(role_id, role)):
            track = TRACKS[i % len(TRACKS)]
            load = LOADS[(i * 3 + len(role_id)) % len(LOADS)]
            turns = [opener]
            if TRACK_OF[track] not in opener and "11 month" not in opener and "17 mo" not in opener:
                turns.append(track)
            turns.append(load)
            expected_track = ("11 month" if "11 month" in opener else
                              "17 month" if "17 mo" in opener else TRACK_OF[track])
            convos.append({"name": f"role/{role_id}/{i}", "turns": turns,
                           "expect_goal": role_id, "expect_track": expected_track,
                           "expect_load": LOAD_OF[load], "needs_model": False})
    # 2) every industry x phrasing -> pick a role by number or title -> plan
    for ind_id, phrasings in INDUSTRY_PHRASINGS.items():
        titles = electives.top_titles_for(ind_id, 10)
        for i, opener in enumerate(phrasings):
            pick_n = (i * 2 + 1)  # 1, 3, 5
            title, role_id = titles[pick_n - 1]
            pick = str(pick_n) if i % 2 == 0 else orchestrator.title_case(title)
            track, load = TRACKS[i % len(TRACKS)], LOADS[i % len(LOADS)]
            convos.append({"name": f"industry/{ind_id}/{i}",
                           "turns": ["no idea", opener, pick, track, load],
                           "expect_goal": role_id, "expect_track": TRACK_OF[track],
                           "expect_load": LOAD_OF[load], "needs_model": False})
    # 3) detours dropped into role conversations
    for j, (role_id, role) in enumerate(list(CAREERS.items())[:12]):
        detour, needs_model = DETOURS[j % len(DETOURS)]
        t0 = (role.get("titles") or [role_id])[0]
        convos.append({"name": f"detour/{role_id}/{detour[:18]}",
                       "turns": [f"{t0}, 11 month", detour, "moderate"],
                       "expect_goal": role_id, "expect_track": "11 month",
                       "expect_load": "moderate", "needs_model": needs_model})
    # 4) change of mind, both directions
    pairs = [("data scientist", "ml-engineer", "actually ml engineer"),
             ("bi analyst", "consultant", "hmm consultant instead"),
             ("pricing analyst", "marketing-analyst", "marketing is more my thing"),
             ("fraud analyst", "finance-quant", "financial analyst actually")]
    for a, b_id, switch in pairs:
        convos.append({"name": f"switch/{a}->{b_id}",
                       "turns": [f"{a} 11 month moderate", switch],
                       "expect_goal": b_id, "expect_track": "11 month",
                       "expect_load": "moderate", "needs_model": False})
    # 5) uncovered roles, must refuse and name nothing
    for role in ["esports analyst", "sommelier", "astronaut", "yoga instructor",
                 "marine biologist", "film director"]:
        convos.append({"name": f"uncovered/{role}", "turns": [f"i want to be a {role}"],
                       "expect_refusal": True, "needs_model": True})
    return convos

# --------------------------------------------------------------- checking
def check(convo, replies, conversation):
    problems = []
    for turn, reply in zip(convo["turns"], replies):
        if not (reply.body or "").strip():
            problems.append(f"empty reply to {turn!r}")
        for b in reply.quick_replies or []:
            if not (isinstance(b, dict) and b.get("send")):
                problems.append(f"malformed button after {turn!r}")
    if convo.get("expect_refusal"):
        last = replies[-1]
        if last.model_note not in ("no-profile", "industry-menu", "out-of-scope"):
            problems.append(f"uncovered role routed to {last.model_note}")
        named = [c for c in DISPLAY if c in last.body]
        if named:
            problems.append(f"named courses on refusal: {named[:2]}")
        return problems
    for turn, reply in zip(convo["turns"], replies):
        if reply.model_note in BAD_NOTES and not convo["needs_model"]:
            problems.append(f"{reply.model_note!r} in reply to {turn!r}")
    stored = planner.load_session_intake(conversation)
    if stored.get("goals") != [convo["expect_goal"]]:
        problems.append(f"goal {stored.get('goals')} != {[convo['expect_goal']]}")
    if stored.get("track") != convo["expect_track"]:
        problems.append(f"track {stored.get('track')!r} != {convo['expect_track']!r}")
    # On the 11-month track the load is not a question (`load_is_a_choice`):
    # a load word said there is answered, and may or may not be recorded.
    load_matters = planner.load_is_a_choice(convo["expect_track"])
    if load_matters and stored.get("workload") != convo["expect_load"]:
        problems.append(f"load {stored.get('workload')!r} != {convo['expect_load']!r}")
    if not load_matters and stored.get("workload") not in (None, convo["expect_load"]):
        problems.append(f"load {stored.get('workload')!r} != {convo['expect_load']!r}")
    if stored.get("goals") and stored.get("track"):
        plan = planner.build_for(stored, planner.taken_course_ids(USER))
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
        if plan.get("route") != "fixed":
            problems.append(f"lost the curated bundle (route={plan.get('route')})")
    return problems

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=6, help="transcripts to print for failures")
    ap.add_argument("--variant", type=int, default=1, choices=(1, 2),
                    help="1: titles and industry words; 2: questions, typos, "
                         "one-turn intakes, detours into the menu")
    args = ap.parse_args()
    matrix = build_matrix_v2() if args.variant == 2 else build_matrix()
    failures, slow = [], []
    t_all = time.time()
    for convo in matrix:
        c = Conversation.objects.create(user=USER, destination="courses", title=convo["name"][:60])
        replies, timings = [], []
        try:
            for turn in convo["turns"]:
                t = time.time()
                replies.append(orchestrator.answer(LLM, c, turn, []))
                timings.append(time.time() - t)
            problems = check(convo, replies, c)
        except Exception as exc:  # noqa: BLE001
            problems = [f"EXCEPTION {type(exc).__name__}: {str(exc)[:90]}"]
        if not convo["needs_model"] and max(timings or [0]) > 1.5:
            slow.append((convo["name"], max(timings)))
        if problems:
            failures.append((convo, replies, problems))
        c.delete()
    print(f"\n{'=' * 78}\n{len(matrix)} conversations, {sum(len(c['turns']) for c in matrix)} turns, "
          f"{time.time() - t_all:.0f}s\n{'=' * 78}")
    for convo, replies, problems in failures[:args.show]:
        print(f"\n--- FAIL {convo['name']}: {'; '.join(problems)}")
        for turn, reply in zip(convo["turns"], replies):
            print(f"    STUDENT: {turn}")
            print(f"    [{reply.model_note}] {reply.body[:160].replace(chr(10), ' ')}")
    if len(failures) > args.show:
        print(f"\n... and {len(failures) - args.show} more:")
        for convo, _r, problems in failures[args.show:]:
            print(f"    {convo['name']}: {problems[0]}")
    if slow:
        print(f"\nslow deterministic turns (>1.5s, means a model call leaked in): {slow[:6]}")
    print(f"\n{len(matrix) - len(failures)}/{len(matrix)} passed")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
