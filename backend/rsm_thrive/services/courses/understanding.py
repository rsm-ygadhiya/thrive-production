"""Reading what a student MEANT by a job, against the closed list we curate.

The recommender's quality floor is that a career resolves through
`careers.json` and nothing else: the curated mapping is authoritative and
must not become probabilistic. That floor was also its blind spot. Measured
live, "i want to be a pm" was refused -- "I don't have a course recommendation
for **pm**" -- and the student had to type "product manager" before the very
same profile came back. The words were shorthand for a title the table holds;
the table just could not read shorthand.

This module is the reading layer, in two parts, and neither part decides
which courses a student gets:

* **Shorthand** -- `careers.json` carries, per profile, the abbreviations and
  field names students type for it ("pm", "ds", "fp&a", "a/b testing").
  Matched deterministically by the router, ahead of any model call, and
  REPORTED as a reading so the reply confirms it ("Reading **pm** as
  **Product Analyst / Product Data Scientist** -- say the word if you meant
  something else") rather than acting on it silently.

* **A curated reading by the model** -- when nothing on disk matches, the
  `read_turn` skill is shown the fourteen profiles and asked which one, if
  any, the student's words clearly denote. It may only answer with an id from that
  list, the code checks the id exists, and the reply confirms the reading
  back. The model interprets; the table still decides. A job the list does
  not cover -- sommelier, firefighter -- stays unmatched and is refused as
  before, because mapping it to the nearest analytics job is exactly the
  guess this codebase refuses to make.
"""

import re

from rsm_thrive.services.courses.electives import load_careers

# Below this the model is not sure enough for its reading to be acted on, even
# with a confirmation line. A wrong reading puts the wrong career's whole
# elective set in front of a student; asking once is cheaper.
MIN_READING_CONFIDENCE = 0.7

# Shorthand next to a digit is a time of day, not a job: "class at 2 pm",
# "2pm", "10:30 pm". Fixed-width lookbehinds, as Python requires.
_AFTER_A_DIGIT = r"(?<!\d)(?<!\d\s)(?<!\d\s\s)"


def shorthand_pattern(short):
    """A regex matching `short` as its own word(s), and never inside a time."""
    return re.compile(_AFTER_A_DIGIT + rf"\b{re.escape(short)}\b")


def industry_roster():
    """The six industries as the model is shown them: id, label."""
    from rsm_thrive.services.courses.electives import load_industries

    return "\n".join(f"- {industry['id']}: {industry['label']}"
                     for industry in load_industries())


def roster():
    """The curated profiles as the model is shown them: id, label, titles."""
    lines = []
    for role_id, role in load_careers().items():
        titles = ", ".join(str(t) for t in (role.get("titles") or []))
        lines.append(f"- {role_id}: {role.get('label', role_id)}"
                     + (f" ({titles})" if titles else ""))
    return "\n".join(lines)


# The instructions the model reads alongside the roster used to be a string
# here. They now live in `skills/read_turn/SKILL.md`, section 4, where the
# rest of the turn's reading is described and where staff can edit them; the
# roster is rendered into that file's `<<roster>>` placeholder per call.


def curated_reading(parsed):
    """(role id, confidence) the model read the job as, validated, or ("", 0.0).

    Only an id (or, forgivingly, a label) that `careers.json` actually holds
    comes back. Anything else the model says is discarded here, which is what
    keeps this a reading of the student's words rather than a source of
    careers.
    """
    if not isinstance(parsed, dict):
        return "", 0.0
    said = " ".join(str(parsed.get("curated") or "").lower().split())
    if not said:
        return "", 0.0
    careers = load_careers()
    role_id = ""
    if said in careers:
        role_id = said
    else:
        for candidate, role in careers.items():
            names = {str(role.get("label", "")).lower(),
                     str(role.get("short_label", "")).lower()}
            if said in names:
                role_id = candidate
                break
    if not role_id:
        return "", 0.0
    try:
        confidence = float(parsed.get("curated_confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    return role_id, max(0.0, min(1.0, confidence))


def curated_industry(parsed):
    """(industry id, confidence) the model read the field as, or ("", 0.0).

    Only an id (or, forgivingly, a label) that `industries.json` holds comes
    back -- the same rule as `curated_reading`, for the same reason.
    """
    from rsm_thrive.services.courses.electives import load_industries

    if not isinstance(parsed, dict):
        return "", 0.0
    said = " ".join(str(parsed.get("curated_industry") or "").lower().split())
    if not said:
        return "", 0.0
    industry_id = ""
    for industry in load_industries():
        names = {industry["id"], str(industry.get("label", "")).lower(),
                 str(industry.get("short_label", "")).lower()}
        if said in names:
            industry_id = industry["id"]
            break
    if not industry_id:
        return "", 0.0
    try:
        confidence = float(parsed.get("curated_industry_confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    return industry_id, max(0.0, min(1.0, confidence))
