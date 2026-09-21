"""What the course planner trusts the model to read, and how it is checked.

The planner's facade over `services/skills.py`: `read`, `say`, `load` and
`available` here are bound to this bot's own `skills/` directory, and the
validators below know the closed sets the planner acts on -- two tracks,
three loads, the catalog's course codes, five skill areas on a 1-5 scale,
the quarters of a track. A reading is a claim about what the student SAID,
checked against what the planner can do; anything outside the set is
dropped, never repaired. "17 months" becomes "17 month" because that is the
same statement in a different spelling; "18 month" becomes "" because there
is no such track, and asking is cheaper than a plan built on one.

The skills themselves:

    skills/read_turn/SKILL.md          everything a turn states -- its route,
                                       the intake, and where the student is
    skills/explain_position/SKILL.md   the trade-offs in a computed ledger
    skills/answer_course_facts/SKILL.md a catalog question, from catalog rows
    skills/small_talk/SKILL.md         two sentences once a plan is on screen
"""

from __future__ import annotations

import re
from pathlib import Path

from rsm_thrive.services import skills
from rsm_thrive.services.skills import (Reading, Skill, SkillError,  # noqa: F401
                                        confidence_of, flag_of, one_of,
                                        render_context, words_of)

SKILLS_DIR = Path(__file__).resolve().parent / "skills"


def available():
    return skills.available(SKILLS_DIR)


def load(name):
    return skills.load(SKILLS_DIR, name)


def read(name, llm, text, context=None, **fill):
    return skills.read(SKILLS_DIR, name, llm, text, context, **fill)


def say(name, llm, messages, **fill):
    return skills.say(SKILLS_DIR, name, llm, messages, **fill)


# ---------------------------------------------------------------------------
# The planner's closed sets
# ---------------------------------------------------------------------------

_TRACK_OF = {"11": "11 month", "eleven": "11 month",
             "17": "17 month", "seventeen": "17 month"}


def track_of(value):
    """"11 month" / "17 month", from any of the ways it gets written, or ""."""
    said = str(value or "").lower()
    match = re.search(r"\b(11|17|eleven|seventeen)\b", said)
    return _TRACK_OF[match.group(1)] if match else ""


def load_of(value):
    return one_of(value, ("light", "moderate", "heavy"))


def route_choice_of(value):
    return one_of(value, ("fixed", "custom"))


def codes_of(value):
    """Catalog course codes, in catalog spelling. Anything else is dropped."""
    from rsm_thrive.services.courses import planner

    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value)
    return planner.mentioned_codes(str(value or ""))


def courses_of(value):
    """Catalog codes for courses named by code OR by title. Unknown ones drop.

    A student mid-walk-through says "swap out the consumer behaviour one", and
    the model is told to give the code from the plan it was shown; when it
    gives the title instead, the title is matched here against the catalog
    -- whole phrase inside a title, or every word of it -- and only a UNIQUE
    match counts. Two courses called something like it is a question to ask,
    not a course to pick.
    """
    from rsm_thrive.services.courses.electives import load_catalog

    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    catalog = load_catalog()
    codes = []
    for item in value:
        said = str(item or "")
        found = codes_of(said)
        if not found:
            phrase = " ".join(re.findall(r"[a-z0-9]+", said.lower()))
            words = [w for w in phrase.split() if len(w) >= 4]
            if phrase and words:
                hits = []
                for course in catalog:
                    title = " ".join(re.findall(r"[a-z0-9]+",
                                                str(course.get("title", "")).lower()))
                    if phrase in title or all(w in title.split() for w in words):
                        hits.append(course["code"])
                if len(hits) == 1:
                    found = hits
        for code in found:
            if code not in codes:
                codes.append(code)
    return codes


def ratings_of(value):
    """{skill_<area>: 1-5} for every area the model reported a level for."""
    from rsm_thrive.services.courses import planner

    if not isinstance(value, dict):
        return {}
    areas = {area["key"] for area in planner.SKILL_AREAS}
    out = {}
    for key, level in value.items():
        area = str(key).lower().strip().removeprefix("skill_")
        score = planner.skill_score(level)
        if area in areas and score is not None:
            out[f"skill_{area}"] = score
    return out


def quarter_of(track, value):
    """A quarter key on `track` for what the student called it, or None."""
    from rsm_thrive.services.courses import planner

    return planner.resolve_quarter(track or "11 month", words_of(value))
