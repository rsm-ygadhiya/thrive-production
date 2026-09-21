"""Match what a job or an industry asks for against what this catalog teaches.

One matcher, shared by both web-backed routes. The web says what a field or a
role currently requires; THIS decides which of our courses teach it, and it is
the only thing allowed to name a course. The model never picks one, so it can
never name one that does not exist.

## Why this replaced two matchers

`grounded_course_advisor` and `role_lookup` each grew their own version of the
same job and they disagreed. Measured on identical input (a food-industry
requirement set): one returned four courses, the other six, with different
scores and three courses appearing in only one of them. Healthcare returned
five and two. Two implementations of one decision, neither reproducible from
the other.

Worse, both were doing something structurally wrong for this data. The
catalog's `skills` are natural-language PHRASES — "building and evaluating
forecasting models", "estimating ROI of analytics", 2 to 8 words, 118 of them —
and the web returns short phrases like "demand forecasting" or "experiment
design". Neither matcher compared phrases:

* `advisor._requirement_matches` required every token of the requirement to
  appear somewhere in the course's whole text. So "systems engineering" matched
  MGTA 461 because "systems" is in "recommender systems" and "engineering" is
  in "data engineering" — two unrelated phrases, one spurious match.
* `role_lookup._overlap` accepted a match when every word LONGER THAN THREE
  CHARACTERS was present. "cpg analytics" therefore matched on "analytics"
  alone, and picked up four more courses on that basis.

And both counted every requirement as worth the same. Measured on an aerospace
requirement set, the top result matched on "python, excel, forecasting" — three
words that describe half this catalog. Three of the four results were carried
by "python". That is the "plausible and unhelpful" failure: nothing in the
answer is false, and nothing in it is informative either.

## What this does instead

**1. A requirement matches ONE catalog phrase, not the union of a course's
text.** Words scattered across unrelated phrases are not evidence that a course
teaches the thing.

**2. Words match on a shared stem, not on equality.** "forecasting" /
"forecast", "designing" / "design", "experiments" / "experiment" are the same
requirement, and exact matching missed all three. See `_same_word`.

**3. A requirement every course satisfies is worth nothing.** Each requirement
is weighted by how much of the catalog it separates — the standard inverse
document frequency, over a fixed 31-row catalog, so it is exact rather than
estimated. "python" appears in most courses and scores near zero; "cost
effectiveness analysis" appears in one and dominates. This is the single change
that fixes the aerospace case, and it needs no threshold tuning: a requirement
that cannot discriminate contributes nothing by construction.

**4. Where it matched counts.** A course's `skills` and `tools` are a claim
about what it teaches; a word in its `description` is prose.

**5. Everything is evidence.** Every match records the requirement, the
catalog's own phrase, the field and the tier, so a recommendation can say
"teaches *building and evaluating forecasting models*" rather than assert a
course code and hope.

Deterministic throughout: same requirements in, same ranking out, and every
number here is computed from `courses.json`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field as dataclass_field
from functools import lru_cache
from rsm_thrive.services.courses.electives import catalog_version, load_catalog

# Keeps "c++", "c#", ".net" and "a/b" in one piece rather than splitting them
# to noise. Same tokeniser the advisor used, kept deliberately.
_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.]*")

# Words that carry no signal inside a skill phrase. "building and evaluating
# forecasting models" is about forecasting and models; "and" and "the" are
# there because it is English.
_FILLER = frozenset("""
a an the and or of in for with to from on at by via using use uses used
their its this that these those into onto over under about across through
your you we our it is are be been being as well other others including
""".split())

# Fields on a course, and how strong a claim each one is that the course
# actually teaches something. `skills` and `tools` are the course saying what
# you will be able to do; `description` is prose that may merely mention a word.
FIELD_WEIGHTS = {
    "title": 1.0,
    "skills": 1.0,
    "tools": 1.0,
    "topics": 0.8,
    "description": 0.4,
}

# How much a tier of match is worth. A partial match is real evidence and
# weaker evidence; it is not half a match by coincidence, it is a phrase that
# overlapped in most but not all of its content.
TIER_WEIGHTS = {"exact": 1.0, "phrase": 1.0, "partial": 0.5}

# A course whose subject IS the field the student named belongs in the answer
# whether or not it teaches the tooling current postings ask for. The catalog's
# Healthcare Analytics course matches almost none of the 24 requirements a
# healthcare search returns -- it teaches decision modelling, ICER and Markov
# cohorts, against postings that want SQL and Tableau -- so under a
# skills-only ranking the one course named for the field was the one course a
# student asking about it never saw.
DOMAIN_BONUS = 2.0

# Admission. Both gates are needed and they do different jobs.
#
# `MIN_DISTINCT` stops a course entering on one lucky word. `RELATIVE_FLOOR` is
# what keeps the answer SHORT: scores are IDF-weighted, so they are comparable
# within one query but not across queries, and an absolute cut would be
# generous on a query full of rare terms and brutal on one full of common ones.
# A fraction of the best score is the same judgement either way -- "is this
# course in the same league as the best match, or is it the tail?"
MIN_DISTINCT = 2
RELATIVE_FLOOR = 0.35
MAX_RESULTS = 6

# At least one match must be about CONTENT, not just tooling.
#
# A tool is how you work; a skill, topic or title is what you learn. A course
# recommended purely because it uses the same software as the field is not a
# recommendation about the field. Measured: a healthcare requirement set
# admitted Marketing Analytics and Business Forecasting on "uses R" plus "uses
# Python" and nothing else -- two real matches, both discriminating within the
# catalog, neither of them a reason to take the course for healthcare.
#
# A course that IS about the field is exempt, because being about it is a
# content claim of the strongest kind.
MIN_NON_TOOL = 1

# Below this share of the field's (discrimination-weighted) requirements met,
# the catalog covers the generic tooling and none of what makes the field that
# field. The recommendation is still worth giving; it must not be given as
# though it were a good fit. Callers change their framing on this.
THIN_COVERAGE = 0.5

# A reason worth less than this share of a course's best reason is not why the
# course is being recommended, and printing it makes the recommendation look
# thinner than it is. Applies only to the EXPLANATION -- the match still counts
# toward the score, it just does not get a line of its own.
REASON_FLOOR = 0.3

# A shared prefix shorter than this is a coincidence. Words shorter than this
# must match exactly, which is what keeps "r", "ai", "bi" and "etl" from
# matching inside longer words -- measured on this catalog, substring "r"
# appears in all 31 courses and "ai" in 14 (chain, training, available).
STEM_FLOOR = 4

# What may be left over once two words share a stem. Both remainders must be in
# here, which is the difference between an inflection and a coincidence.
#
# A bare shared-prefix rule was tried first and produced four false positives on
# this catalog, all of them damaging:
#
#   tableau     / table          -> credited MGTA 464 (SQL, common TABLE
#                                   expressions) with teaching Tableau
#   healthcare  / health         -> made Marketing Analytics a healthcare course
#   communication / community
#   product     / produce
#
# Every one is blocked here, because "au", "care", "ication" and "t" are not
# ways English inflects a word. Nothing legitimate was lost: forecast/
# forecasting, design/designing, experiment/experiments, statistics/
# statistical and optimisation/optimization all still match.
_INFLECTIONS = frozenset("""
s es ed d e ing ings er ers or ors ion ions tion tions ation ations
sation zation al ally ic ics ical y ies ment ments ness ive ives
""".split()) | {""}


@dataclass
class Match:
    """One requirement, met by one phrase the catalog actually contains."""
    requirement: str
    #: The catalog's own words. Never the web's -- this is what a reply quotes.
    phrase: str
    field: str
    tier: str
    #: How good the match itself is: tier times field. Used to pick the best
    #: evidence for one requirement.
    weight: float
    #: What this match is actually WORTH to the ranking: `weight` times how
    #: much the requirement discriminates. The two come apart badly and the
    #: difference is visible to students: "uses Python" is a perfect match
    #: (weight 1.0) and nearly worthless (contribution 0.45), because most of
    #: this catalog uses Python. Ordering the reasons by `weight` put "uses
    #: Python" at the top of an aerospace recommendation ahead of "covers data
    #: visualization", which is the "plausible and unhelpful" failure showing
    #: up again in the explanation after being fixed in the ranking.
    contribution: float = 0.0


@dataclass
class Fit:
    course: dict
    score: float
    #: Best evidence per requirement, strongest first.
    matches: list[Match] = dataclass_field(default_factory=list)
    #: The course is ABOUT the field the student named.
    domain: bool = False

    @property
    def covered(self):
        """The requirements this course meets, in the student's own words."""
        return [match.requirement for match in self.matches]


# ---------------------------------------------------------------------------
# Words and phrases
# ---------------------------------------------------------------------------

def _words(text):
    return [word for word in _TOKEN.findall((text or "").lower())
            if word not in _FILLER]


def _same_word(left, right):
    """Two words for the same thing: a shared stem plus an ordinary inflection.

    Exact for anything short. For anything longer, the two must share a stem of
    at least `STEM_FLOOR` characters AND what is left over on each side must be
    a way English inflects a word (`_INFLECTIONS`). Both halves are required:
    the prefix alone equated "tableau" with "table", and the suffix list alone
    would equate anything with anything.

    Deliberately NOT a real stemmer. A stemmer is a dependency, an irregular
    verb list and a second vocabulary to keep in step with this one; the whole
    benefit here is captured by a shared stem and thirty suffixes, and this
    version can be read in full and argued with.

    A doubled consonant before the suffix is allowed for, so "modelling" and
    "model" agree, as do "programming" and "program". English doubles the
    consonant before "-ing" and "-ed"; a rule that did not know this would
    treat the British spelling as a different word.
    """
    if left == right:
        return True
    if len(left) < STEM_FLOOR or len(right) < STEM_FLOOR:
        return False
    shared = 0
    for a, b in zip(left, right):
        if a != b:
            break
        shared += 1
    if shared < STEM_FLOOR:
        return False
    stem = left[:shared]
    remainders = []
    for word in (left, right):
        rest = word[shared:]
        if rest[:1] == stem[-1:]:
            rest = rest[1:]        # "model" + "l" + "ing"
        remainders.append(rest)
    return all(rest in _INFLECTIONS for rest in remainders)


def _match_phrase(requirement_words, phrase_words):
    """(tier, hits) for one requirement against ONE catalog phrase.

    Against one phrase, not against everything the course says. "systems
    engineering" is a requirement about systems engineering; finding "systems"
    in "recommender systems" and "engineering" in "data engineering" is finding
    two words, not the skill.

    A partial match needs at least TWO of the requirement's words, which is
    what stops a two-word requirement matching on its generic half: "cpg
    analytics" cannot enter on "analytics" alone.
    """
    hits = sum(1 for word in requirement_words
               if any(_same_word(word, other) for other in phrase_words))
    if not hits:
        return None, 0
    if hits == len(requirement_words):
        return "phrase", hits
    if hits >= 2 and hits * 2 >= len(requirement_words):
        return "partial", hits
    return None, 0


# ---------------------------------------------------------------------------
# The catalog, indexed once
# ---------------------------------------------------------------------------

def _phrases_of(course):
    """(field, phrase) for everything a course says about itself.

    The description is split into sentences rather than kept whole: a match
    against a 60-word paragraph is a match against whatever it happened to
    mention, where a match against one sentence is at least a match against one
    claim.
    """
    out = [("title", course.get("title") or "")]
    for field in ("skills", "tools", "topics"):
        out += [(field, str(value)) for value in (course.get(field) or [])]
    for sentence in re.split(r"(?<=[.;])\s+", course.get("description") or ""):
        if sentence.strip():
            out.append(("description", sentence.strip()))
    return [(field, phrase) for field, phrase in out if phrase]


def _index(pool):
    return [(course, [(field, phrase, _words(phrase))
                      for field, phrase in _phrases_of(course)])
            for course in pool]


@lru_cache(maxsize=8)
def _indexed_catalog_for(_version, core):
    return _index([course for course in load_catalog()
                   if core or not course.get("is_core")])


def _indexed_catalog(core):
    """The real catalog, indexed once per version of the file."""
    return _indexed_catalog_for(catalog_version(), core)


def _indexed(core, catalog=None):
    """[(course, [(field, phrase, words)])] for the pool being ranked.

    `catalog` overrides the shipped one. Only tests pass it, and they must be
    able to: every threshold here is relative to the POOL -- `discriminations`
    counts how many of these courses satisfy a requirement, and the relative
    floor is a share of the best score in this pool -- so a matcher that could
    only ever be exercised against 31 real rows could not be tested at the
    boundaries at all.
    """
    if catalog is None:
        return _indexed_catalog(core)
    return _index([course for course in catalog
                   if core or not course.get("is_core")])


# ---------------------------------------------------------------------------
# Matching one requirement across the pool
# ---------------------------------------------------------------------------

def _best_match(requirement, requirement_words, indexed_phrases):
    """The strongest evidence one course offers for one requirement, or None."""
    best = None
    for field, phrase, phrase_words in indexed_phrases:
        tier, _hits = _match_phrase(requirement_words, phrase_words)
        if tier is None:
            continue
        weight = TIER_WEIGHTS[tier] * FIELD_WEIGHTS.get(field, 0.5)
        if best is None or weight > best.weight or (
                # On a tie, content beats tooling. "excel" matching the skill
                # "excel modelling" and the tool "Excel" scores the same either
                # way, and which one is recorded decides whether the course
                # counts as recommended for its CONTENT -- see `MIN_NON_TOOL`.
                weight == best.weight
                and best.field == "tools" and field != "tools"):
            best = Match(requirement=requirement, phrase=phrase, field=field,
                         tier=tier, weight=weight)
    return best


def _normalise(requirements):
    """The requirement list, deduplicated, lowercased, empties dropped."""
    seen, out = set(), []
    for value in requirements or []:
        if not isinstance(value, (str, int, float)):
            continue
        text = " ".join(str(value).strip().lower().split())
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def discriminations(requirements, *, core=False, catalog=None):
    """How much each requirement separates this catalog. {requirement: weight}

    Inverse document frequency, computed exactly rather than estimated: the
    catalog is 31 fixed rows, so `df` is countable and there is nothing to
    approximate. A requirement every course satisfies scores 0 and contributes
    nothing to any ranking; one that only a single course satisfies dominates.

    This is what makes the difference between an answer and a shrug. Measured
    on an aerospace requirement set against the old matcher, three of four
    results were carried by "python" — a word that appears in most of this
    catalog and therefore tells a student nothing about which course to take.
    Here "python" is worth about 0.4 and "cost effectiveness analysis" about
    2.8, and no threshold had to be tuned to make that true.
    """
    indexed = _indexed(core, catalog)
    total = len(indexed)
    weights = {}
    for requirement in _normalise(requirements):
        words = _words(requirement)
        if not words:
            continue
        frequency = sum(
            1 for _course, phrases in indexed
            if _best_match(requirement, words, phrases) is not None)
        weights[requirement] = math.log((total + 1) / (1 + frequency))
    return weights


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def _content_matches(fit):
    """How many of a course's matches are about what it TEACHES."""
    return sum(1 for match in fit.matches if match.field != "tools")


def rank(requirements, *, domain=None, core=False, limit=MAX_RESULTS,
         catalog=None):
    """Courses that teach what was asked for, best first.

    `requirements` is everything the web returned — skills, tools and topics
    flattened into one list, because the distinction between them is the web's
    and the catalog does not keep it.

    `domain` is the field the student named, in their OWN words ("healthcare",
    "food"). It is matched like any other requirement, so a course about the
    field is credited for being about it.
    """
    indexed = _indexed(core, catalog)
    wanted = _normalise(requirements)
    weights = discriminations(wanted, core=core, catalog=catalog)
    domain_words = _words(domain) if domain else []

    scored = []
    for course, phrases in indexed:
        matches, score = [], 0.0
        for requirement in wanted:
            words = _words(requirement)
            if not words:
                continue
            best = _best_match(requirement, words, phrases)
            if best is None:
                continue
            # A requirement the WHOLE pool satisfies weighs nothing, and the
            # match is still recorded. The distinction is load-bearing: it
            # cannot help rank one course above another, but it is evidence the
            # course teaches the thing, so it should count toward admission
            # rather than silently make a course look thinner than it is.
            # Skipping it outright made a course with two genuine matches look
            # like a course with one.
            best.contribution = best.weight * weights.get(requirement, 0.0)
            matches.append(best)
            score += best.contribution
        is_domain = bool(domain_words) and _best_match(
            domain, domain_words, phrases) is not None
        matches.sort(key=lambda match: -match.contribution)
        scored.append(Fit(course=course, score=score, matches=matches,
                          domain=is_domain))

    # A "field" that fits most of the catalog is not naming a field.
    # "analytics" on its own would credit a third of these courses, and
    # admitting all of them on that basis is worse than admitting none.
    if sum(1 for fit in scored if fit.domain) > limit:
        for fit in scored:
            fit.domain = False
    domain_weight = (DOMAIN_BONUS * max(
        0.0, discriminations([domain], core=core, catalog=catalog)
        .get(" ".join((domain or "").lower().split()), 0.0)) if domain else 0.0)
    for fit in scored:
        if fit.domain:
            fit.score += domain_weight

    admitted = [fit for fit in scored
                if fit.domain or (len(fit.matches) >= MIN_DISTINCT
                                  and _content_matches(fit) >= MIN_NON_TOOL)]
    # No `score > 0` gate. A course whose every match is a requirement the
    # whole pool satisfies scores zero, and the relative floor below already
    # drops it whenever anything else scored at all. Gating on the score as
    # well meant that a pool where NOTHING discriminates returned nothing,
    # which is not "no good matches" — it is "every match is equally good",
    # and the two deserve different answers.
    if not admitted:
        return []
    best_score = max(fit.score for fit in admitted)
    if best_score > 0:
        admitted = [fit for fit in admitted
                    if fit.domain or fit.score >= RELATIVE_FLOOR * best_score]
    admitted.sort(key=lambda fit: (not fit.domain, -fit.score,
                                   fit.course.get("code", "")))
    return admitted[:limit]


# ---------------------------------------------------------------------------
# Saying why
# ---------------------------------------------------------------------------

def reasons(fit, limit=3):
    """Why this course is here, in the CATALOG's words and the student's.

    Both halves matter. The requirement is what the student asked about; the
    phrase is what this catalog actually promises, quoted rather than
    paraphrased. "matches sql" is an assertion — "teaches *querying and joining
    transactional data*, which covers **sql**" is checkable against the course
    page.
    """
    out = []
    if fit.domain:
        out.append("directly about this field")
    best = fit.matches[0].contribution if fit.matches else 0.0
    worth_saying = [match for match in fit.matches
                    if match.contribution >= REASON_FLOOR * best]
    for match in worth_saying[:limit]:
        if match.field == "tools":
            out.append(f"uses **{match.phrase}**")
        elif match.field == "description":
            # A whole sentence of prose, quoted back, reads as a quotation
            # rather than as a reason -- and it is the weakest kind of evidence
            # this matcher records. Say where it came from instead.
            out.append(f"its description covers **{match.requirement}**")
        elif match.tier == "partial":
            out.append(f"touches *{match.phrase}* — part of **{match.requirement}**")
        else:
            out.append(f"teaches *{match.phrase}* — covers **{match.requirement}**")
    return out


def coverage(requirements, fits, *, core=False, limit=6, catalog=None):
    """What the catalog met, and what it did not. Weighted by discrimination.

    The honest half of an industry answer, and the half that was missing. For
    aerospace this catalog matches "python", "excel", "tableau" and
    "forecasting" and matches NOTHING for systems engineering, reliability
    analysis, predictive maintenance, statistical process control or aerospace
    manufacturing — which is to say it matched the generic tooling and none of
    what makes the field that field. Returning three courses and saying nothing
    about the rest is how "plausible and unhelpful" survives a better matcher.

    `met_share` is weighted by `discriminations`, not counted, for the same
    reason the ranking is: meeting "python" and missing "predictive
    maintenance" is not half a job done.
    """
    weights = discriminations(requirements, core=core, catalog=catalog)
    met = covered_by(fits)
    unmet = sorted(((weight, requirement)
                    for requirement, weight in weights.items()
                    if requirement not in met and weight > 0),
                   reverse=True)
    total = sum(weights.values())
    met_weight = sum(weight for requirement, weight in weights.items()
                     if requirement in met)
    return {
        "met": sorted(met),
        "unmet": [requirement for _weight, requirement in unmet[:limit]],
        "metShare": (met_weight / total) if total else 0.0,
    }


def covered_by(fits):
    """Every requirement any recommended course meets. For "and what is not"."""
    return {match.requirement for fit in fits for match in fit.matches}
