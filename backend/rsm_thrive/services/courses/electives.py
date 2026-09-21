"""Deterministic elective scoring against a structured student profile.

Ported from the Rady Recommender prototype's ``matcher.py``. The scoring
arithmetic is kept identical to the original; only the output shape and
the Django-facing helpers (``load_catalog``, ``load_careers``,
``recommend_for``) are new.

Produces a ranked list with human-readable reasons; the LLM may present or
lightly adjust this ranking but the base ordering is reproducible.
"""
import json
import time
from functools import lru_cache
from pathlib import Path

from rsm_thrive.models import Enrollment

_DATA = Path(__file__).resolve().parents[2] / "data" / "catalog"

# The three files that ARE the catalog. Everything derived from them keys its
# cache on `catalog_version()`.
_CATALOG_FILES = ("courses.json", "careers.json", "bundles.json",
                  "industries.json")

WORKLOAD_LEVEL = {"light": 1, "moderate": 2, "heavy": 3}

# How long a version check is trusted. See `catalog_version`.
_VERSION_TTL = 1.0
_VERSION_MEMO = (0.0, None)


def catalog_version():
    """A token that changes when any catalog file does.

    Every lookup built on these files is cached -- the catalog is 31 rows read
    thousands of times a request -- and they were cached FOREVER, keyed on
    nothing. Editing `careers.json` therefore did nothing at all until someone
    restarted the process, and Django's autoreloader only watches `.py` files
    so nothing prompted them to. Found the hard way: two roles were made
    nameable, the change was verified by unit tests, and the running server
    went on failing on the live sweep because it still held the old file.

    The catalog is the source of truth that programme staff edit, so "it takes
    effect when someone remembers to restart" is not a workable contract.

    Modification time in NANOSECONDS, because two edits inside the same second
    are ordinary during a working session and a whole-second clock would miss
    the second one. A missing file reads as 0 rather than raising: the caller
    is about to open it and will produce a better error than this can.

    The stat itself is memoised for `_VERSION_TTL`. Three stats is 10
    microseconds, which reads as free until you notice where this is called
    from: `bundles._by_id` sits inside the combinatorial placement search, so
    "once per catalog lookup" is tens of thousands of times per plan. Measured,
    the unmemoised version took the test suite from 8.7s to 15.1s. A second of
    staleness is a fair trade for a file a human edits by hand; `forget_catalog`
    is there for anything that cannot wait.
    """
    now = time.monotonic()
    checked_at, version = _VERSION_MEMO
    if version is not None and now - checked_at < _VERSION_TTL:
        return version
    version = _stat_version()
    # A benign race: two threads may both stat and both assign. They compute
    # the same tuple, and the assignment is atomic, so no lock is needed for
    # something read on every catalog lookup.
    _set_version_memo(now, version)
    return version


def _stat_version():
    return tuple(
        (_DATA / name).stat().st_mtime_ns if (_DATA / name).exists() else 0
        for name in _CATALOG_FILES)


def _set_version_memo(checked_at, version):
    global _VERSION_MEMO
    _VERSION_MEMO = (checked_at, version)


def forget_catalog():
    """Drop every cached read of the catalog files. For tests, and for a
    process that has just written one and wants the change NOW rather than
    within `_VERSION_TTL`."""
    _set_version_memo(0.0, None)
    for cached in (_load_catalog, _load_careers, _role_aliases_for,
                   _load_industries):
        cached.cache_clear()


@lru_cache(maxsize=4)
def _load_catalog(_version):
    return json.loads((_DATA / "courses.json").read_text())


def load_catalog():
    return _load_catalog(catalog_version())


@lru_cache(maxsize=4)
def _load_careers(_version):
    return json.loads((_DATA / "careers.json").read_text())


def load_careers():
    return _load_careers(catalog_version())


@lru_cache(maxsize=4)
def _load_industries(_version):
    return json.loads((_DATA / "industries.json").read_text())


def load_industries():
    """The six industries, each with its roles in the design document's order.

    `industries.json` carries Part III section 3.3 of the MSBA Elective
    Recommender design document. The ORDER of each industry's role list is the
    ranking for that industry and is load-bearing -- it is what "top roles in
    <industry>" means -- so nothing here sorts it.
    """
    return _load_industries(catalog_version())["industries"]


def industry_by_id(industry_id):
    for industry in load_industries():
        if industry["id"] == industry_id:
            return industry
    return None


def top_titles_for(industry_id, limit=10):
    """The industry's best-known job titles, as (title, role_id) pairs.

    BREADTH FIRST across the profiles, not depth first. Taking titles in flat
    order would fill the list from the top one or two profiles -- Technology's
    first five would be "product analyst, product data scientist, growth
    analyst, product manager, data scientist - product", which is one job
    written five ways and tells a student nothing about the industry's spread.
    Round-robin takes each profile's primary title first, so the list spans
    every profile in the industry's ranking before it names a second title for
    anyone, and only then cycles for the remainder.

    Deduplicated across profiles, because a title like "growth analyst" is
    listed under both Product and Marketing, and the same word twice in a
    numbered list reads as a bug.
    """
    industry = industry_by_id(industry_id)
    if not industry:
        return []
    careers = load_careers()
    titles = [list(careers.get(role["id"], {}).get("titles") or [])
              for role in industry["roles"]]
    out, seen, depth = [], set(), 0
    while len(out) < limit and any(len(t) > depth for t in titles):
        for index, role_titles in enumerate(titles):
            if depth >= len(role_titles):
                continue
            title = role_titles[depth]
            if title in seen:
                continue
            seen.add(title)
            out.append((title, industry["roles"][index]["id"]))
            if len(out) >= limit:
                break
        depth += 1
    return out


@lru_cache(maxsize=4)
def _role_aliases_for(_version):
    """Old role id -> the profile that absorbed it.

    The taxonomy moved from ten roles to the design document's fourteen
    profiles, and a saved `CoursePlan.intake` or an in-flight conversation may
    still carry an old id. The alias is declared on the profile itself
    (`legacy_ids`), so the mapping is stated once, in the data, beside the
    thing it maps to -- rather than in a second table that can drift out of
    step with the first.
    """
    return {old: new
            for new, role in load_careers().items()
            for old in role.get("legacy_ids") or []}


def _role_aliases():
    return _role_aliases_for(catalog_version())


def resolve_role(role_id):
    """The current id for a role id that may predate the taxonomy change.

    Returns None for an id that is neither current nor a known alias, which is
    what keeps `normalise_intake` dropping an invented role rather than
    repairing it into something the student never said.
    """
    if role_id in load_careers():
        return role_id
    return _role_aliases().get(role_id)


# Words that make a question a COURSE question even when it names no course.
# "what can I take?" and "which classes are hands-on?" are both about the
# catalog and neither mentions a code or a title.
COURSE_WORDS = frozenset("""
course courses class classes elective electives core catalog curriculum
unit units credit credits prerequisite prerequisites prereq syllabus
offered offering quarter term seasons teach teaches taught instructor
workload difficulty hard easy technical topic topics skill skills tool tools
take taking enrol enroll enrolled study learn cover covers
""".split())

# Function words. Not course vocabulary, and not rare enough to be caught by
# `COMMON_TERM_SHARE` either -- "what" appears in exactly one description
# ("what-if analysis"), so a question made only of function words was returning
# that one course as though it were the answer. A closed set is the right shape
# here precisely because it IS closed: interrogatives and auxiliaries do not
# grow the way subject vocabulary does.
QUESTION_WORDS = frozenset("""
what which where when whos whose does have has had you your yours they them
their there here this that these those about with from into onto some many
much more most tell show list give offer offers access available able need
want like know anything everything something please thanks help
""".split())

_COURSE_CODE = __import__("re").compile(r"\b([A-Z]{2,4})\s*(\d{3}[A-Z]?)\b", __import__("re").IGNORECASE)


def _searchable(course):
    """Everything about a course a student might name it by."""
    seasons = {o.get("season") for o in course.get("offerings") or []}
    # Season NAMES as well as codes: a student asks "which electives run in
    # winter", never "which run in WI".
    spelled = {"SU": "summer", "FA": "fall autumn", "WI": "winter",
               "SP": "spring"}
    parts = [course.get("code", ""), course.get("title", ""),
             course.get("description", ""), course.get("department", ""),
             course.get("workload", "")]
    parts.extend(spelled.get(season, "") for season in seasons if season)
    # The OTHER names Rady's own pages use. A student reading the electives page
    # sees "CSE 250B" and types that; without this the catalog has no idea what
    # they mean, even though it carries the course under its new number.
    parts.extend(course.get("also_known_as") or [])
    for key in ("topics", "skills", "tools", "career_tags"):
        parts.extend(course.get(key) or [])
    return " ".join(parts).lower()


# A term appearing in more than this share of the catalog cannot distinguish
# one course from another, so it is dropped from a search rather than allowed
# to return six of whatever it matched first.
COMMON_TERM_SHARE = 0.25


def search_catalog(question, limit=6):
    """Courses this question is plausibly about, best first.

    Deterministic term overlap rather than embeddings: the catalog is ~90 rows,
    the fields are short and specific, and a student asking about "Tableau" or
    "fraud" is naming something that appears literally. A vector search over
    thirty items would add a dependency and a failure mode to a lookup.

    A course named by CODE always wins -- "what is MGTA 458" is not a fuzzy
    question and should not be answered with six neighbours.
    """
    text = (question or "").lower()
    catalog = load_catalog()

    named = set()
    for department, number in _COURSE_CODE.findall(question or ""):
        named.add(f"{department.upper()} {number.upper()}")
    if named:
        # Match on the course's own code OR any name it also goes by, so an old
        # or cross-listed number lands on the course that actually exists.
        hits = [c for c in catalog
                if c["code"].upper() in named
                or named & {a.upper() for a in (c.get("also_known_as") or [])}]
        if hits:
            return hits

    # Without a course code, the question has to use the vocabulary of the
    # catalog before a term match counts. "What are the tuition fees?" shares
    # the word "tuition" with a course description and was being answered from
    # that course -- a confident answer to a question this bot was not asked.
    if not is_course_question(question):
        return []

    terms = {word.strip(".,/-?!") for word in text.split()}
    terms = {t for t in terms
             if len(t) > 3 and t not in COURSE_WORDS
             and t not in QUESTION_WORDS}
    if not terms:
        return []

    haystacks = {course["code"]: _searchable(course) for course in catalog}

    # A term that matches most of the catalog is not a search term. "What
    # courses do you have" leaves {"what", "have"} after the stoplist, and both
    # appear in enough descriptions to return six arbitrary rows -- which is how
    # a question the WHOLE catalog answers started being answered by a handful
    # of it instead. Harmless while the catalog was 31 short rows; wrong the
    # moment it grew. A stoplist would be whack-a-mole, so the test is
    # proportional rather than lexical, and it is the same reasoning
    # `skill_match.discriminations` applies to skills.
    ceiling = max(2, int(len(catalog) * COMMON_TERM_SHARE))
    distinctive = {t for t in terms
                   if sum(1 for h in haystacks.values() if t in h) <= ceiling}
    # Only when something SURVIVES. A common term is worth dropping next to a
    # rarer one; on its own it is the whole question. "Which electives are
    # offered in winter" leaves {"winter"}, which is common precisely because
    # the student is asking about something many courses share -- dropping it
    # would answer a season question with nothing.
    terms = distinctive or terms

    scored = []
    for course in catalog:
        haystack = haystacks[course["code"]]
        hits = sum(1 for term in terms if term in haystack)
        if hits:
            scored.append((hits, course["code"], course))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [course for _hits, _code, course in scored[:limit]]


def is_course_question(question):
    """Does this ask about the catalog at all?

    True when it names a course, or uses the vocabulary of one. Everything else
    belongs to a different surface -- this bot plans courses and should say so
    rather than answer from whatever it can find.
    """
    text = (question or "").lower()
    if _COURSE_CODE.search(question or ""):
        return True
    words = {word.strip(".,/-?!") for word in text.split()}
    return bool(words & COURSE_WORDS)


# What each course prefix actually IS. Read off the catalog's own notes, which
# say it plainly -- "Pre-approved MSBA elective (MBA course -- enrollment by
# consent)" and so on. Asked "what courses do you have access to", a list of
# codes is not an answer: a student wants to know they can reach beyond the
# MSBA's own courses into CSE, MBA and MFin ones, and on what terms.
DEPARTMENT_LABELS = {
    "MGTA": ("Rady MSBA", "the programme's own courses — core and electives"),
    "CSE": ("UCSD Computer Science & Engineering",
            "pre-approved MSBA electives; CSE majors have enrolment priority, "
            "so seats go as space permits"),
    "MGT": ("Rady MBA",
            "pre-approved MSBA electives; enrolment by consent"),
    # MQF, not MFin. Rady runs both -- its own page lists "EMBA, Full-Time
    # MBA, FlexEvening/FlexWeekend MBA, MPAc, MQF and MFin" as separate
    # programmes -- and the MGTF courses are the MQF's: MGTF 416 is titled
    # "MQF Professional Seminar" in Rady's own catalog, and the current
    # registration-fee pages for this prefix's programme are headed "Master
    # of Quantitative Finance". The old label named the wrong degree, and
    # because `import_syllabi` stamps it into every MGTF syllabus, it had
    # already been ingested into the corpus as though it were sourced.
    "MGTF": ("Rady MQF (Master of Quantitative Finance)",
             "pre-approved MSBA electives; enrolment by consent"),
    "MGTP": ("Rady MPAc (Master of Professional Accountancy)",
             "pre-approved MSBA electives; enrolment by consent"),
}

# Up to 16 units of the 28 may come from outside the MSBA's own courses.
NON_MSBA_UNIT_CAP = 16


def departments_available():
    """Every programme this catalog reaches into, with counts and terms."""
    catalog = load_catalog()
    out = []
    for prefix in sorted({c.get("department", "") for c in catalog if c.get("department")}):
        rows = [c for c in catalog if c.get("department") == prefix]
        name, terms = DEPARTMENT_LABELS.get(prefix, (prefix, ""))
        out.append({
            "prefix": prefix, "name": name, "terms": terms,
            "total": len(rows),
            "core": sum(1 for r in rows if r["is_core"]),
            "electives": sum(1 for r in rows if not r["is_core"]),
            "codes": sorted(r["code"] for r in rows),
        })
    return out


def catalog_overview():
    """A summary for "what have you got?" -- a question no single course answers."""
    catalog = load_catalog()
    electives = [c for c in catalog if not c["is_core"]]
    departments = sorted({c.get("department", "") for c in catalog if c.get("department")})
    return {
        "total": len(catalog),
        "core": len(catalog) - len(electives),
        "electives": len(electives),
        "departments": departments,
        "programmes": departments_available(),
        "nonMsbaCap": NON_MSBA_UNIT_CAP,
        "codes": sorted(c["code"] for c in electives),
    }


#: What a course outside the MSBA carries in its `reasons`, so a plan, a
#: swap list and a walk-through can all say the same thing about it without
#: three of them re-deriving it from the course code.
OUTSIDE_THE_MSBA = "sits outside the MSBA — needs the programme's permission"


def is_msba(course):
    """The programme's own courses: the MGTA prefix, core and elective alike."""
    return str(course.get("code") or "").startswith("MGTA")


def rank_electives(catalog, profile, careers=None):
    """Return electives sorted by descending fit score.

    profile: {
        "career_roles": [..],           # role ids from careers.json (e.g. "product-manager")
        "career_tags": [..],            # from the catalog's career-tag vocabulary
        "technical_comfort": 1-5,
        "workload_preference": "light" | "moderate" | "heavy",
        "interests": [..],              # free-text keywords
    }
    careers: careers.json dict mapping role id -> {label, career_tags, boost_courses}.

    Each result is {"course": dict, "score": float, "reasons": [str]},
    sorted descending by score with ties broken by course code.
    """
    careers = careers or {}
    # roles expand into ordered tags + direct course boosts; the first role
    # is the student's primary objective
    roles = [r for r in (profile.get("career_roles") or []) if r in careers]
    ordered_tags = list(profile.get("career_tags") or [])
    course_boosts = {}   # course_id -> [(points, role label), ...]
    for pos, role_id in enumerate(roles):
        role = careers[role_id]
        role_w = 1.0 / (1 + 0.5 * pos)
        for tag in role.get("career_tags") or []:
            if tag not in ordered_tags:
                ordered_tags.append(tag)
        for cid, pts in (role.get("boost_courses") or {}).items():
            course_boosts.setdefault(cid, []).append((pts * role_w, role["label"]))

    # profile tag order matters: the first tag is the student's primary goal
    tag_weight = {
        tag: 1.0 / (1 + 0.5 * pos) for pos, tag in enumerate(ordered_tags)
    }
    career_tags = set(tag_weight)
    tech = profile.get("technical_comfort", 3)
    workload_pref = WORKLOAD_LEVEL.get(profile.get("workload_preference"), 2)
    interests = [i.lower() for i in (profile.get("interests") or [])]

    results = []
    for course in catalog:
        if course["is_core"]:
            continue
        score = 0.0
        reasons = []

        # career-tag matches weighted by position: a course's first tag is its
        # primary focus, later tags are secondary
        course_tags = course.get("career_tags") or []
        matched = [(pos, tag) for pos, tag in enumerate(course_tags) if tag in career_tags]
        if matched:
            score += sum(3.0 / (1 + pos) * tag_weight[tag] for pos, tag in matched)
            reasons.append(
                "builds your " + ", ".join(tag for _, tag in matched) + " focus"
            )

        for pts, label in course_boosts.get(course["id"], []):
            score += pts
            reasons.insert(0, f"directly relevant for {label}")

        gap = course.get("technical_level", 3) - tech
        if gap > 0:
            score -= 1.5 * gap
            reasons.append(f"more technical than your comfort level (+{gap})")
        else:
            score += 0.5
            reasons.append("matches your technical level")

        # workload preference is a tolerance cap: only penalize courses that
        # demand more than the student wants to take on
        wl_over = WORKLOAD_LEVEL[course.get("workload", "moderate")] - workload_pref
        if wl_over > 0:
            score -= 0.75 * wl_over
            reasons.append(f"workload is {course['workload']} — above your preference")
        else:
            reasons.append(f"{course['workload']} workload fits your preference")

        haystack = " ".join(
            [course.get("title", ""), course.get("description", "")]
            + (course.get("topics") or [])
            + (course.get("skills") or [])
            + (course.get("tools") or [])
        ).lower()
        hits = [kw for kw in interests if kw in haystack]
        if hits:
            score += 2.0 * len(hits)
            reasons.append("covers your interest in " + ", ".join(hits))

        if not is_msba(course):
            reasons.append(OUTSIDE_THE_MSBA)

        results.append({
            "course": course,
            "score": round(score, 2),
            "reasons": reasons,
        })

    # MSBA COURSES FIRST, and not as a tie-break -- as the first sort key, so
    # an MGTA course outranks every course outside the programme whatever the
    # scores say.
    #
    # This is a preference about ENROLMENT, not about quality, which is why a
    # score boost was the wrong instrument: a boost large enough to win
    # reliably distorts every comparison it touches, and one small enough not
    # to lets a marginally better-scoring outside course take a slot the
    # student may not be able to register for. MGTA courses are the
    # programme's own, they are what the design document's bundles are built
    # from, and they are the ones an MSBA student can rely on getting into.
    # Everything else needs permission and competes with the students that
    # programme belongs to.
    #
    # Nothing is EXCLUDED. An outside course still sorts, still scores, still
    # appears among the alternatives for a slot, and still lands in the plan
    # when no MSBA course can fill that slot -- it just never displaces one
    # that can. It also carries `OUTSIDE_THE_MSBA` in its reasons from here
    # on, so every surface that shows it can say so.
    results.sort(key=lambda r: (not is_msba(r["course"]),
                                -r["score"], r["course"]["code"]))
    return results


def recommend_for(user, career_roles, interests=None, limit=5):
    profile = {
        "career_roles": career_roles,
        "career_tags": [],
        "technical_comfort": 3,
        "workload_preference": "moderate",
        "interests": interests or [],
    }
    ranked = rank_electives(load_catalog(), profile, load_careers())
    taken = _taken_codes(user)
    ranked = [r for r in ranked if r["course"]["code"] not in taken]
    return ranked[:limit]


def _taken_codes(user):
    return {
        enrollment.course.code
        for enrollment in Enrollment.objects.filter(user=user)
                                             .select_related("course")
    }
