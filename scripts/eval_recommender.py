"""A graded evaluation of the MSBA course recommender, component by component.

Different instrument from `sweep_conversations.py`. The sweep drives whole
conversations and answers "did anything break". This scores each PART of the
funnel separately and reports how well, so a regression shows up as a number
moving rather than as a test that happens to fail.

    cd backend && PYTHONPATH=$PWD uv run python ../scripts/eval_recommender.py
    ... --live     # also grade the routes that need a model call

Offline by default, because the funnel is deterministic almost end to end:
the taxonomy, the menus, role resolution, and every plan are dictionary
lookups and arithmetic. Only classification of an unrecognised sentence needs
the model, and `--live` grades that separately rather than hiding it inside
everything else.

Sections are GATES or SCORES. A gate is a property that must hold absolutely
-- a plan of 49 units is not 98% correct, it is wrong. A score is a quality
measure with a floor, where the number is the useful part and the floor only
catches collapse.
"""
import argparse
import collections
import json
import os
import sys

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from rsm_thrive.services.courses import (bundles, electives,  # noqa: E402
                                         orchestrator, planner, router)
from rsm_thrive.services.llm import FakeLLM  # noqa: E402

CAREERS = electives.load_careers()
CATALOG = {c["id"]: c for c in electives.load_catalog()}
BY_CODE = {c["code"] for c in electives.load_catalog()}
INDUSTRIES = electives.load_industries()
TRACKS = ("11 month", "17 month")
LOADS = ("light", "moderate", "heavy")

report = []


def gate(section, name, failures, detail=""):
    report.append(("gate", section, name, not failures,
                   detail or (f"{len(failures)} failing: {failures[:3]}"
                              if failures else "clean")))


def score(section, name, value, floor, detail=""):
    report.append(("score", section, name, value >= floor,
                   f"{value:.0%} (floor {floor:.0%}){' - ' + detail if detail else ''}"))


def seasons_of(course):
    return {o.get("season") for o in (course.get("offerings") or [])
            if o.get("season")}


def season_map(track):
    return {q["key"]: q["season"] for q in planner.TRACK_SKELETONS[track]}


def build(goal, track, load):
    answers = {"track": track, "goals": [goal], "workload": load}
    seeded = planner.seeded_units(track, load)
    if seeded:
        answers["quarter_units"] = seeded
    return answers, planner.build_for(answers, frozenset())


# ---------------------------------------------------------------------------
# 1. The data the whole thing stands on
# ---------------------------------------------------------------------------
def eval_taxonomy():
    s = "1. taxonomy"
    gate(s, "every industry role exists in careers.json",
         [r["id"] for i in INDUSTRIES for r in i["roles"] if r["id"] not in CAREERS])
    listed = {r["id"] for i in INDUSTRIES for r in i["roles"]}
    gate(s, "every career is reachable from some industry",
         sorted(set(CAREERS) - listed))
    gate(s, "no industry lists a role twice",
         [i["id"] for i in INDUSTRIES
          if len({r["id"] for r in i["roles"]}) != len(i["roles"])])
    gate(s, "every bundle course exists in the catalog",
         sorted({c for role in bundles.load_bundles().values()
                 for key in ("anchor", "differentiator", "universal")
                 for c in (role.get(key) or [])
                 if c not in CATALOG and c not in BY_CODE}))
    gate(s, "every course states when it runs",
         [c["code"] for c in electives.load_catalog() if not seasons_of(c)])
    gate(s, "every industry can fill a top ten",
         [i["id"] for i in INDUSTRIES
          if len(electives.top_titles_for(i["id"], 10)) != 10])


# ---------------------------------------------------------------------------
# 2. Naming a job, however the student spells it
# ---------------------------------------------------------------------------
def eval_role_resolution():
    s = "2. role resolution"
    titles = [(t, rid) for rid, role in CAREERS.items()
              for t in (role.get("titles") or [])]
    hits = sum(1 for t, rid in titles if router.matched_role(t) == rid)
    score(s, "every curated job title reaches its profile",
          hits / len(titles), 0.95, f"{hits}/{len(titles)}")

    aliases = [(a, rid) for rid, role in CAREERS.items()
               for a in (role.get("aliases") or [])]
    gate(s, "every alias reaches its own profile",
         [a for a, rid in aliases if router.matched_role(a) != rid])

    offered = {t for i in INDUSTRIES for t, _ in electives.top_titles_for(i["id"], 10)}
    gate(s, "no alias is ever offered as a job title",
         [a for a, _ in aliases if a in offered])

    typos = [("data scienist", "data-scientist"), ("markting analyst", "marketing-analyst"),
             ("pricing anlayst", "pricing-analyst"), ("bussiness analyst", "business-data-analyst"),
             ("consltant", "consultant"), ("produt manager", "product-analyst")]
    ok = sum(1 for said, want in typos if router.matched_role(said) == want)
    score(s, "one character of slop still resolves", ok / len(typos), 0.80,
          f"{ok}/{len(typos)}")

    not_jobs = ["sommelier", "astronaut", "wildlife biologist", "the weather",
                "MGTA 464", "what courses do you have", "117", ""]
    gate(s, "a non-job resolves to nothing",
         [q for q in not_jobs if router.matched_role(q)])


# ---------------------------------------------------------------------------
# 3. The two menus
# ---------------------------------------------------------------------------
def eval_menus():
    s = "3. menus"
    user = _user()
    convo = _conversation(user)

    pairs = [(i["id"], t, rid) for i in INDUSTRIES
             for t, rid in electives.top_titles_for(i["id"], 10)]
    wrong = []
    for industry_id, title, role_id in pairs:
        c = _conversation(user)
        orchestrator.industry_roles_reply(c, electives.industry_by_id(industry_id))
        if orchestrator.role_from_industry_menu(c, title) != role_id:
            wrong.append((industry_id, title))
        c.delete()
    gate(s, "every offered title resolves to the profile that offered it", wrong)

    orchestrator.industry_menu_reply(convo)
    # 1-indexed: 0 is out of range, not the first item.
    numbers = [(str(n), 1 <= n <= len(INDUSTRIES))
               for n in (1, 3, 6, 7, 0, 99)]
    gate(s, "menu numbers pick in range and nothing out of it",
         [n for n, want in numbers
          if bool(orchestrator.pick_by_number(convo, n)) is not want])

    buttons = orchestrator.industry_buttons()
    gate(s, "every button carries a label and a send",
         [b for b in buttons if not (b.get("label") and b.get("send"))])
    gate(s, "no two buttons share a send (the render key)",
         [] if len({b["send"] for b in buttons}) == len(buttons) else ["duplicate"])
    gate(s, "every button's send resolves back to its industry",
         [b["send"] for b in buttons
          if (orchestrator.resolve_industry(b["send"]) or {}).get("label") != b["send"]])

    words = {"tech": "technology-software", "banking": "financial-services",
             "fintech": "financial-services", "consulting": "consulting",
             "biotech": "healthcare", "pharma": "healthcare",
             "retail": "retail-cpg", "cpg": "retail-cpg",
             "gaming": "other", "energy": "other", "media": "other"}
    ok = sum(1 for w, want in words.items()
             if (orchestrator.resolve_industry(w) or {}).get("id") == want)
    score(s, "industry words students actually use", ok / len(words), 0.90,
          f"{ok}/{len(words)}")
    gate(s, "a word we do not cover resolves to nothing",
         [w for w in ["agriculture", "hospitality", "e-sports analyst", "law"]
          if orchestrator.resolve_industry(w)])
    convo.delete()


# ---------------------------------------------------------------------------
# 4. Is the plan a legal plan of study
# ---------------------------------------------------------------------------
def eval_plan_validity():
    s = "4. plan validity"
    units, unfilled, dupes, ghosts, terms, routes = [], [], [], [], [], []
    for goal in CAREERS:
        for track in TRACKS:
            seasons = season_map(track)
            for load in LOADS:
                _a, plan = build(goal, track, load)
                tag = (goal, track, load)
                if plan["totals"]["total"] != 50:
                    units.append(tag)
                if plan["unfilled"]:
                    unfilled.append(tag)
                if plan.get("route") != "fixed":
                    routes.append(tag)
                ids = [r["courseId"] for q in plan["quarters"] for r in q["courses"]]
                if len(ids) != len(set(ids)):
                    dupes.append(tag)
                ghosts += [i for i in ids if i not in CATALOG]
                for quarter in plan["quarters"]:
                    want = seasons.get(quarter["key"])
                    for row in quarter["courses"]:
                        course = CATALOG.get(row.get("courseId"))
                        if course and want not in seasons_of(course):
                            terms.append((course["code"], want) + tag)
    gate(s, "every plan closes at exactly 50 units", units)
    gate(s, "no plan has an unfilled slot", unfilled)
    gate(s, "no plan repeats a course", dupes)
    gate(s, "no plan names a course outside the catalog", sorted(set(ghosts)))
    gate(s, "no course is placed in a term it is not taught", terms)
    gate(s, "no load choice costs the curated bundle", routes)


# ---------------------------------------------------------------------------
# 5. Is the plan the RIGHT plan
# ---------------------------------------------------------------------------
def eval_plan_relevance():
    s = "5. plan relevance"
    on_goal, in_bundle, msba = [], [], []
    for goal, role in CAREERS.items():
        boosts = set(role.get("boost_courses") or {})
        spine = set()
        for key in ("anchor", "differentiator", "universal"):
            spine |= set((bundles.bundle_for(goal) or {}).get(key) or [])
        for track in TRACKS:
            _a, plan = build(goal, track, "moderate")
            chosen = [r["courseId"] for q in plan["quarters"]
                      for r in q["courses"] if r["requirement"] != "Core"]
            code = lambda c: CATALOG[c]["code"]
            on_goal.append(sum(1 for c in chosen
                               if code(c) in boosts or c in boosts) / len(chosen))
            in_bundle.append(sum(1 for c in chosen
                                 if code(c) in spine or c in spine) / len(chosen))
            msba.append(sum(1 for c in chosen if c.startswith("MGTA")) / len(chosen))
    score(s, "electives the professor weighted for that role",
          sum(on_goal) / len(on_goal), 0.70)
    score(s, "electives from that role's own bundle",
          sum(in_bundle) / len(in_bundle), 0.70)
    score(s, "electives that are the MSBA's own courses",
          sum(msba) / len(msba), 0.80)
    score(s, "worst single profile still serves its goal", min(on_goal), 0.50)

    for track in TRACKS:
        sets = {g: tuple(sorted(r["courseId"] for q in build(g, track, "moderate")[1]["quarters"]
                                for r in q["courses"] if r["requirement"] != "Core"))
                for g in CAREERS}
        score(s, f"profiles with a distinct plan ({track})",
              len(set(sets.values())) / len(sets), 0.85,
              f"{len(set(sets.values()))}/{len(sets)}")


# ---------------------------------------------------------------------------
# 6. What the student is actually shown
# ---------------------------------------------------------------------------
def eval_presentation():
    s = "6. presentation"
    user = _user()
    missing_brief, unflagged = [], []
    for goal in ("data-scientist", "ml-engineer", "healthcare-analyst",
                 "bi-analyst", "marketing-analyst"):
        answers, plan = build(goal, "11 month", "moderate")
        for index in range(len(plan["quarters"])):
            body, _replies, _last = planner.review_quarter(plan, answers, index)
            for quarter in [plan["quarters"][index]]:
                for row in quarter["courses"]:
                    course = CATALOG.get(row.get("courseId"))
                    if not course:
                        continue
                    if course["code"] not in body:
                        continue
                    if not planner._teaches(course):
                        missing_brief.append(course["code"])
                    if (not course["code"].startswith("MGTA")
                            and "outside the MSBA" not in body):
                        unflagged.append((goal, course["code"]))
    gate(s, "every shown course can say what it teaches", sorted(set(missing_brief)))
    gate(s, "a course outside the MSBA is flagged where it appears",
         sorted(set(unflagged)))

    conv = _conversation(user)
    menu = orchestrator.industry_menu_reply(conv).body
    gate(s, "the menu asks one question, not two",
         ["nudge"] if "aiming for" in menu else [])
    roles_body = orchestrator._industry_roles_body(INDUSTRIES[3])
    gate(s, "the role list is job titles, not annotated profiles",
         ["parenthetical"] if "_(" in roles_body else [])
    opening = planner.opening_prompt()["body"]
    gate(s, "the opening names all three ways in",
         [d for d in ("know the job", "know the field, not the job", "no idea yet")
          if d not in opening])
    conv.delete()


# ---------------------------------------------------------------------------
# 7. Saying no
# ---------------------------------------------------------------------------
def eval_refusals(live):
    s = "7. refusals"
    if not live:
        report.append(("skip", s, "needs a model call", True, "run with --live"))
        return
    from rsm_thrive.services.llm import get_llm
    llm, user = get_llm(), _user()
    roles = ("esports analyst", "sommelier", "wildlife biologist",
             "astronaut", "climate risk modeller", "underwater welder")
    repeats = 3
    leaked, refused, total = [], 0, 0
    for role in roles:
        for _ in range(repeats):
            c = _conversation(user)
            reply = orchestrator.answer(llm, c, f"i want to be a {role}", [])
            total += 1
            refused += bool(reply.refused)
            leaked += [x for x in BY_CODE if x in reply.body]
            c.delete()

    # SAFETY IS THE GATE. Whatever route an uncovered role takes, the one
    # thing that must never happen is a course being named for a job nothing
    # maps courses to. That is absolute and does not get a percentage.
    gate(s, "no course is ever named for an uncovered role", sorted(set(leaked)))

    # ROUTING IS A SCORE, because a classifier decides it and a classifier is
    # not deterministic. Measured at 17/18: the miss routed "sommelier" to the
    # industry menu instead of the refusal -- less direct, still honest, still
    # names no course. Sampled rather than asked once, so the number means
    # something; a hard gate here would fail roughly one run in six for a
    # behaviour that is not wrong.
    score(s, "an uncovered role is refused outright", refused / total, 0.80,
          f"{refused}/{total} over {repeats} runs each")


# ---------------------------------------------------------------------------
def _user():
    from django.contrib.auth.models import User
    u, _ = User.objects.get_or_create(username="evalbot")
    return u


def _conversation(user):
    from rsm_thrive.models import Conversation
    return Conversation.objects.create(user=user, destination="courses",
                                       title="eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="also grade the routes that need a model call")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    eval_taxonomy()
    eval_role_resolution()
    eval_menus()
    eval_plan_validity()
    eval_plan_relevance()
    eval_presentation()
    eval_refusals(args.live)

    if args.json:
        print(json.dumps([{"kind": k, "section": s, "name": n, "ok": ok,
                           "detail": d} for k, s, n, ok, d in report], indent=2))
        return 0

    width = max(len(n) for _, _, n, _, _ in report) + 2
    section = None
    for kind, sec, name, ok, detail in report:
        if sec != section:
            print(f"\n{sec}\n{'-' * 74}")
            section = sec
        mark = "SKIP" if kind == "skip" else ("PASS" if ok else "FAIL")
        print(f"  [{mark}] {name:<{width}} {detail}")

    graded = [r for r in report if r[0] != "skip"]
    failed = [r for r in graded if not r[3]]
    by = collections.Counter(r[0] for r in graded)
    print(f"\n{'=' * 74}")
    print(f"{len(graded) - len(failed)}/{len(graded)} checks passed "
          f"({by['gate']} gates, {by['score']} scores)")
    for _k, sec, name, _ok, detail in failed:
        print(f"  FAILED  {sec} / {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
