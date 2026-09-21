"""Routing helpers for industry-shaped course questions.

WHAT THIS USED TO BE. A two-stage recommender: one web-backed model call asked
what a field currently hires for, then `skill_match` mapped the answer onto the
catalog. Both stages are gone. The industry answer now comes from the curated
taxonomy in `data/catalog/industries.json` -- the design document's own ranking
of every profile within each of six industries -- which is specific to this
programme and reviewed, where the web answer was neither, and which arrives
without the 8.3-second lookup this module measured.

WHAT IS LEFT is the half that never touched the web: reading a field out of a
student's sentence. `router` uses both of these to decide whether a question is
industry-shaped and what field it names, and that judgement is unchanged.
"""

from __future__ import annotations

import re

_COURSE_WORDS = frozenset(
    "course courses class classes elective electives msba program take teach "
    "study learn units credits prerequisite prerequisites"
    .split()
)
_TARGET_MARKERS = frozenset(
    "industry field sector domain career role job profession pathway"
    .split()
)
_GENERIC_TARGETS = frozenset(
    "industry field sector domain career role job profession pathway the msba "
    "program course courses class classes elective electives"
    .split()
)
_TARGET_PHRASE = re.compile(
    r"\b(?:in|for|toward|towards|within|support|assist(?:ing)?|help(?:ing)?)\s+"
    r"(?:the\s+)?([a-z][a-z0-9/& -]{2,70}?)(?=\s+(?:industry|field|sector|domain|career|role|job)\b|[?.!,]|$)",
    re.IGNORECASE,
)

# Keeps "c++", "c#" and ".net" in one piece rather than splitting them to noise.
_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.]*")

# "me work in healthcare analytics"; the field is the last two words.
_FIELD_STOPWORDS = frozenset(
    "a an the me my mine i we us our you your it its that this these those to "
    "into toward towards of on at in for with and or work working works "
    "get getting go going want wanting like want need needs help helping "
    "assist assisting support supporting be been being do doing does "
    "would could should can may might will after later someday"
    .split()
)


def is_industry_course_question(question: str) -> bool:
    """Return true for a course question that names any target field."""
    words = set(re.findall(r"[a-z]+", (question or "").lower()))
    if not words & _COURSE_WORDS:
        return False
    phrase = _TARGET_PHRASE.search(question or "")
    if phrase:
        target_words = set(re.findall(r"[a-z]+", phrase.group(1).lower()))
        if target_words - _GENERIC_TARGETS:
            return True
    # Also support concise forms such as "electives for healthcare" or
    # "courses, data science" while excluding a bare generic catalog request.
    if words & _TARGET_MARKERS:
        return len(words - _COURSE_WORDS - _GENERIC_TARGETS) >= 1
    return False


def _target_field(question: str) -> str | None:
    """The field the student named, e.g. "healthcare analytics". None if unsure.

    Every capture the pattern can make is considered, and the SHORTEST one that
    still names something survives. A single question yields several: "help me
    work in healthcare analytics" matches at both "help" and "in", giving "me
    work in healthcare analytics" and "healthcare analytics". The longer one is
    noise, and since the field is matched as a whole -- every token has to be
    present -- one stray word is enough to credit nothing at all.

    Only what the pattern captures is trusted. `is_industry_course_question` has
    a looser branch that routes on leftover words, and words like "work" do
    appear in course descriptions, so crediting them would hand a domain match
    to most of the catalog. When the field cannot be named confidently, no
    credit is awarded and behaviour is exactly what it was.

    The field comes from the student's own question, never from the web, so the
    module's grounding boundary is untouched.
    """
    candidates = []
    for phrase in _TARGET_PHRASE.finditer(question or ""):
        tokens = [token for token in _TOKEN.findall(phrase.group(1).lower())
                  if token not in _GENERIC_TARGETS and token not in _FIELD_STOPWORDS]
        if tokens:
            candidates.append(tokens)
    if not candidates:
        return None
    return " ".join(min(candidates, key=len))
