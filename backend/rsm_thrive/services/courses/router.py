"""One place that decides how a course question gets answered.

Routing used to be scattered: a regex in `grounded_course_advisor`, the
interview's own state machine in `planner.next_intake_step`, a vocabulary check
in `electives.is_course_question`, and a chain of `if` branches in
`bots.answer_electives` that each knew about the next. Nothing named the
decision, so nothing could be tested, logged or corrected as a decision.

This names it. `classify` returns one route and how it was arrived at; the
orchestrator dispatches on that and nothing else.

## Rules first, model second

Most questions do not need a round trip. "Prerequisites for MGTA 452" names a
course and asks a catalog question — that is a rule, it costs nothing, and it
cannot drift. So a small set of confident rules runs first, and the model is
asked only about what is left.

The two are kept distinguishable on purpose. A rule's `confidence` is None, not
1.0: a rule is not a probability, and recording it as one would make the two
indistinguishable in the trace view that exists to tell them apart — and it is
the model-decided routes whose accuracy anyone will ever want to measure.

## Unclear is a route

The one case that earns a follow-up question. Everything else answers; a
question comes back only when proceeding would mean guessing which of two very
different answers the student wanted. That is the whole difference between an
assistant and a form: the form asks first and always, this asks last and rarely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from rsm_thrive.services.courses.electives import (catalog_version, load_careers,
                                            load_catalog)
from rsm_thrive.services.courses.advisor import (
    _target_field, is_industry_course_question,
)
from rsm_thrive.services.courses import understanding, reading

ROLE = "role"
INDUSTRY = "industry"
FACTUAL = "factual"
COMBINATION = "combination"
SITUATIONAL = "situational"
QUARTER = "quarter"
CAREERS = "careers"
CATALOG = "catalog"
OUT_OF_SCOPE = "out-of-scope"
UNCLEAR = "unclear"
# Not a kind of question -- a statement that we could not reach the model to
# decide what kind it was. Kept as its own route so an outage is visible in
# the trace view instead of being counted as whatever it fell back to.
DEGRADED = "degraded"

ROUTES = frozenset({ROLE, INDUSTRY, FACTUAL, COMBINATION, SITUATIONAL,
                    QUARTER, OUT_OF_SCOPE, UNCLEAR, DEGRADED})

# Below this, the model is not confident enough to act on. A wrong route is not
# a slightly-worse answer -- it is the industry advisor answering a question
# about a prerequisite, or a refusal to a student who asked something we can
# answer well. Asking once is cheaper than either.
MIN_CONFIDENCE = 0.55


@dataclass
class Route:
    name: str
    #: None when a rule decided. A float only when the model did.
    confidence: float | None = None
    #: The curated `careers.json` id, when the question named a covered role.
    role_id: str = ""
    #: The student's own words for a role we have no bundle for.
    unmatched_role: str = ""
    #: The student's own words when `role_id` is a READING of them rather than
    #: a title they typed -- shorthand ("pm") or the model's curated reading.
    #: The reply confirms it back instead of acting on it silently.
    reading: str = ""
    #: The industry or field named, in the student's own words.
    industry: str = ""
    #: The student's own words for a single quarter they asked about.
    quarter: str = ""
    #: How the decision was reached, for the trace view.
    why: str = ""
    #: What the turn was DOING, when the model read it: "answer", "ask",
    #: "accept", "decline", "greet", ... (see `skills/read_turn/SKILL.md`).
    #: Empty when a rule decided.
    act: str = ""
    #: Intake values the turn stated that no rule could see -- a track, a
    #: load, finished courses, self-rated skills -- validated against their
    #: closed sets. The orchestrator puts them on file whatever the route.
    learned: dict = field(default_factory=dict)
    #: Something to do to a plan already on screen, when the model read one:
    #: {"name": "swap", "courses": [...]} and so on. Empty otherwise.
    action: dict = field(default_factory=dict)
    #: The student said they have no target yet, or asked for the options.
    wants_menu: bool = False
    #: The curated `industries.json` id the model read the field as, from the
    #: closed list of six, when no alias on disk matched the words.
    industry_id: str = ""
    #: 1-based positions on the menu on screen that the turn refers to, when
    #: the model read one -- "the analyst one", "the last two".
    menu_pick: list = field(default_factory=list)
    #: The turn asks how the picked items differ, rather than choosing one.
    menu_compare: bool = False
    #: Where the student says they are, when the turn described it -- the
    #: situational route's input, validated: track, previous_track,
    #: said_quarter, finish_on_time, completed_codes. See `position_of`.
    position: dict = field(default_factory=dict)


# A department prefix from the catalog itself, not any two-to-four letters.
# The loose form matched "does 464" -- "does" being a four-letter word next to
# a number -- which routed correctly by accident and would have routed "week
# 101" the same way. A bare number is handled separately and deterministically
# below, because "does 464 have prerequisites" is a real question a student
# asks and 464 IS a course in this catalog.
COURSE_CODE = re.compile(r"\b(MGTA|MGTF|MGT|CSE)\s*(\d{3}[A-Z]?)\b",
                         re.IGNORECASE)
_BARE_NUMBER = re.compile(r"\b(\d{3}[A-Z]?)\b", re.IGNORECASE)


@lru_cache(maxsize=4)
def _course_numbers_for(_version):
    """The numeric half of every catalog code, e.g. {"451", "464", "251A"}."""
    numbers = set()
    for course in load_catalog():
        parts = course["code"].split()
        if len(parts) == 2:
            numbers.add(parts[1].upper())
        for alias in course.get("also_known_as") or []:
            alias_parts = str(alias).split()
            if len(alias_parts) == 2:
                numbers.add(alias_parts[1].upper())
    return frozenset(numbers)


def _course_numbers():
    return _course_numbers_for(catalog_version())


def names_a_course(question):
    """Does this name a course, by code or by bare number?

    The bare-number half is checked against the CATALOG rather than against a
    pattern, so "464" resolves and "101" does not. That is the difference
    between reading a student's shorthand and pattern-matching any three
    digits in a sentence.
    """
    text = question or ""
    if COURSE_CODE.search(text):
        return True
    numbers = _course_numbers()
    return any(match.group(1).upper() in numbers
               for match in _BARE_NUMBER.finditer(text))

# Words that make a question about a course a FACTUAL one: it asks what the
# catalog or the syllabus says, not what to take.
# The words that turn a named course into a QUESTION ABOUT that course.
#
# The second block is the one that was missing, and the gap was not small: a
# student typed "whats MGTA 458 about" and, because "whats" is not "what" and
# "about" was not on the list, the turn was not ruled factual at all. It fell
# through to the intake, which had a track and a goal by then and answered a
# question about one course by printing the entire plan of study.
_FACTUAL_WORDS = frozenset("""
prerequisite prerequisites prereq prereqs corequisite units unit credits
credit offered offering offer run runs quarter quarters term terms season
syllabus grading graded assessment assessed workload textbook instructor
professor taught teaches meets schedule when what who how many
whats what's whos who's hows how's tell about cover covers covering
explain describe is are does do worth like good hard easy difficult
""".split())

# A student describing WHERE THEY ARE. Both halves are required: "I switched"
# on its own could be about a laptop, and "winter" on its own is a season.
# A switch counts only with track-shaped words after it ("switched to",
# "switching tracks"): "im switching into analytics. is MGTA 479 offered in
# winter?" carried "switching" and "winter" and was ruled situational, over
# a course code and a catalog word.
_POSITION_WORDS = re.compile(
    r"\b((?:switch(?:ed|ing)?|transferr?(?:ed|ing)?|moved)"
    r"(?=\s+(?:to|from|tracks?|programs?|programmes?|over|onto|between)\b)|"
    r"already in|currently in|"
    r"i'?m in|im in|behind|catch up|caught up|left|remaining|remain|finish|"
    r"finishing|graduat\w*|on time|drop(?:ped)?|withdrew|retake|retaking|"
    r"deferred|leave of absence|part[- ]time)\b", re.IGNORECASE)
# "fall" is a season only when it is not "fall behind": the worrier's "i
# really dont want to fall behind" carried both a season and a position word
# and made an industry question situational.
_PROGRAMME_WORDS = re.compile(
    r"\b(11[- ]month|17[- ]month|track|quarter|quarters|winter|spring|"
    r"fall(?!\s+behind)|summer|units|unit|degree|programme|program|plan of study)\b",
    re.IGNORECASE)

# Words that make a sentence about the student's OWN standing in the
# programme -- a track named, a switch, courses already done. With one of
# these, a course code in the sentence is part of the position ("I've done
# MGTA 464 and switched to 11 month, what's left"); without one, a course
# code plus a catalog word is a question about that course, whatever worry
# follows it ("is MGTA 479 offered in winter? will that be a problem").
# "behind" is deliberately absent: "i really dont want to fall behind" is a
# worry after a course question, not a position, and with a real season in
# the sentence ("offered in winter?") it made the question situational.
_OWN_STANDING = re.compile(
    r"\b(11[- ]?month|17[- ]?month|track|switch\w*|transferr?\w*|already|"
    r"i'?ve (?:done|taken|finished|completed)|completed|left|remaining)\b",
    re.IGNORECASE)


@lru_cache(maxsize=4)
def _role_titles_for(_version):
    """Every job title the catalog curates, longest first.

    Longest first because "product data scientist" and "data scientist" are
    both titles and both present in the same sentence; matching the short one
    first would file a Product Analyst question under Data Scientist.
    """
    titles = []
    for role_id, role in load_careers().items():
        for title in (role.get("titles") or []):
            titles.append((str(title).lower(), role_id))
        # `aliases` are RECOGNISED but never shown. They carry the design
        # document's own name for the profile -- "marketing analytics",
        # "fraud analytics", "product analytics" -- which is what a student
        # who has read the programme material types, and which was matching
        # nothing. They are kept out of `titles` because that list is also
        # what the industry menu offers, and "Product Analytics" is a field
        # rather than a job somebody is hired as.
        for alias in (role.get("aliases") or []):
            titles.append((str(alias).lower(), role_id))
        label = str(role.get("short_label") or "").lower()
        if label:
            titles.append((label, role_id))
    return sorted(titles, key=lambda row: -len(row[0]))


def _role_titles():
    return _role_titles_for(catalog_version())


@lru_cache(maxsize=4)
def _shorthand_for(_version):
    """What students TYPE for a profile, longest first: "pm", "ds", "fp&a".

    Kept apart from `titles` and `aliases` for two reasons. It is never shown
    -- nobody is hired as a "pm" -- and it is matched with a guard the others
    do not need: "pm" next to a digit is a time of day. Measured live, "i want
    to be a pm" was refused for want of this list, and the student had to type
    "product manager" to reach the profile that title already sat under.
    """
    out = []
    for role_id, role in load_careers().items():
        for short in (role.get("shorthand") or []):
            out.append((str(short).lower(), role_id))
    return sorted(out, key=lambda row: -len(row[0]))


def _shorthand():
    return _shorthand_for(catalog_version())


def shorthand_match(text):
    """The role id whose shorthand this text contains, or "".

    Longest first, so "quant research" is read before "quant" could be, and
    never inside a time of day -- see `understanding.shorthand_pattern`.
    """
    for short, role_id in _shorthand():
        if understanding.shorthand_pattern(short).search(text):
            return role_id
    return ""


# Phrasing that puts the next word in the student's PAST or present job,
# not their goal: "working in", "been in", "years in", "background in".
_BACKGROUND = re.compile(
    r"\b(?:worked|working|been|background|years?|experience|experienced|"
    r"coming|came|currently|previously|spent|career)\s+(?:in|from|of|as|at)"
    r"\s+(?:the\s+|a\s+)?$")


# A word this long may be off by one character and still be the same word.
# Below it, one edit is too much of the word to guess at: "sql" and "sal" are
# not the same thing, and neither are "ml" and "al".
_FUZZY_FLOOR = 4


def _within_one_edit(said, wanted):
    """Is `said` `wanted` with one character added, dropped, changed or swapped?

    A bounded check rather than a full edit-distance table: the answer is only
    ever needed for "is this the same word", the bound is 1, and the whole
    thing is a walk down two strings.

    Adjacent SWAPS count, which a plain edit distance would score as two
    changes. They are the commonest typo there is -- "anlayst" for "analyst",
    "sceintist" for "scientist" -- and excluding them left the most likely
    mistake as the one thing this could not read.
    """
    if said == wanted:
        return True
    if len(wanted) < _FUZZY_FLOOR or abs(len(said) - len(wanted)) > 1:
        return False
    if len(said) == len(wanted):
        differing = [i for i, (a, b) in enumerate(zip(said, wanted)) if a != b]
        if len(differing) == 1:
            return True
        return (len(differing) == 2 and differing[1] == differing[0] + 1
                and said[differing[0]] == wanted[differing[1]]
                and said[differing[1]] == wanted[differing[0]])
    shorter, longer = sorted((said, wanted), key=len)
    return any(longer[:i] + longer[i + 1:] == shorter for i in range(len(longer)))


def role_match(question):
    """(role id, was it exact?) for the curated role this question names.

    ("", False) when it names none.

    THE quality floor. "I want to be a data scientist" resolves here, off
    `careers.json`, and never reaches a model — the curated mapping is
    authoritative and must not become probabilistic.

    Exact matching runs first and alone decides whenever it can. A student who
    MISTYPES a title is still naming it, though: measured live, "17 month data
    nalyst" was read as a track and no goal at all, so the student was asked
    for the career they had just given. So a second pass allows one character
    of slop per word, and REPORTS that it did — the caller confirms a fuzzy
    match back rather than acting on it silently, because guessing which of
    fourteen careers someone meant is exactly the guess this module refuses to
    make quietly.

    One-word titles are exempt from the slop. "consultant" is too easy to hit
    by accident once a character is free, and it is short enough to type right.
    """
    text = " ".join((question or "").lower().split())
    if not text:
        return "", False
    titles = _role_titles()
    said = re.findall(r"[a-z&]+", text)

    def fuzzy(title):
        # A TYPED title with a slip in it -- so its words sit together, in
        # order. Matching each word anywhere in the sentence read "marketing
        # ... analytics" out of "working in marketing for 6 years and im
        # switching into analytics" as the title "marketing analytics".
        wanted = title.split()
        if len(wanted) < 2:
            return False
        return any(all(_within_one_edit(said[i + k], wanted[k]) for k in range(len(wanted)))
                   for i in range(len(said) - len(wanted) + 1))

    for title, role_id in titles:
        hit = re.search(rf"\b{re.escape(title)}\b", text)
        if hit and " " not in title and _BACKGROUND.search(text[:hit.start()]):
            # A ONE-WORD alias inside a life story is where they came FROM.
            # "i've been working in marketing for 6 years and im switching
            # into analytics" was ruled the Marketing profile on the word
            # "marketing". The same word after "i want to work in" is still
            # a goal and still rules; only the background phrasing is
            # skipped, and the reading skill sees the whole sentence.
            continue
        if hit:
            # THE MORE SPECIFIC READING WINS, even over an exact match.
            # "i wanna be a helthcare data analyst" contains "data analyst"
            # exactly, and that returned Business Analyst before the fuzzy
            # pass could see that "healthcare data analyst" matches with one
            # letter of slop. A longer title that fuzzy-matches AND contains
            # every word of the exact hit is what the student typed; the
            # exact hit is a substring of it. Reported as fuzzy so the
            # confirmation note shows them the reading.
            exact_words = set(title.split())
            for longer, longer_id in titles:
                if len(longer) <= len(title):
                    break
                if exact_words <= set(longer.split()) and fuzzy(longer):
                    return longer_id, False
            return role_id, True
    # SHORTHAND before slop. "pm" is one word and two letters, which the fuzzy
    # pass rightly refuses to guess at; it is not a guess when the profile
    # lists it. Reported as a reading, so the reply confirms "pm" back as the
    # profile it stands for rather than acting on it silently.
    short_id = shorthand_match(text)
    if short_id:
        return short_id, False
    for title, role_id in titles:
        if fuzzy(title):
            return role_id, False
    return "", False


def matched_role(question):
    """The curated role id this question names, or ""."""
    return role_match(question)[0]


def matched_title(question):
    """The exact words -- title, alias, short label or shorthand -- that
    `role_match` would resolve in this question, longest first, or "".

    For a caller that needs to know WHAT matched rather than what it resolved
    to: an alias that is also a course title ("fraud analytics") is a role to
    one reader and a course to another, and only the words themselves tell
    the caller to ask.
    """
    text = " ".join((question or "").lower().split())
    for title, _role_id in _role_titles():
        if re.search(rf"\b{re.escape(title)}\b", text):
            return title
    for short, _role_id in _shorthand():
        if understanding.shorthand_pattern(short).search(text):
            return short
    return ""


# Words that name a JOB rather than a domain. Used only to decide whether to
# CONFIRM a reading back, never to decide what a career is -- a miss here costs
# one extra line of confirmation, where a miss in `role_match` would cost a
# student the wrong career. That asymmetry is what makes a closed list the
# right tool here and the wrong one there.
_JOB_FUNCTIONS = frozenset("""
analyst analysts analytics scientist scientists engineer engineers manager
managers consultant consultants developer developers specialist specialists
coordinator coordinators associate associates lead leads director directors
officer officers strategist strategists modeller modeler statistician actuary
researcher researchers architect architects administrator advisor planner
planners intern trainee auditor accountant economist technologist
""".split())


# The shapes English gives a job: -ist, -eer, -ian, -ant, -or, -er. Enough on
# their own for the words no list will ever contain -- biologist, sommelier,
# archivist, statistician -- which is what stops this becoming the enumeration
# problem the list above nearly is. Five characters minimum, so "other" and
# "over" do not qualify as careers.
_JOB_SUFFIX = re.compile(r"(ist|eer|ian|ant|or|er)$")


def names_a_job_function(question):
    """Did the student name a job, or only an area?

    "supply chain analyst" names one; "something in supply chain" does not,
    and the second matched a curated career off its short label alone --
    handing back a full recommendation for a career the student had gestured
    at rather than chosen.

    Used only to decide whether to ASK or confirm, never to decide what a
    career is. A miss costs one extra turn; a miss in `role_match` would cost
    a student the wrong career, which is why a word list is the right tool
    here and the wrong one there.
    """
    words = {word for word in re.findall(r"[a-z]+", (question or "").lower())}
    if words & _JOB_FUNCTIONS:
        return True
    if any(len(word) >= 5 and _JOB_SUFFIX.search(word) for word in words):
        return True
    # And a misspelling of one. "esports analsyt" names a job as clearly as
    # "esports analyst" does, and neither the list nor the suffix sees it --
    # "analsyt" ends in "yt". Being asked to clarify a job you named and
    # merely mistyped is the interrogation this whole check exists to avoid.
    return any(_within_one_edit(word, function)
               for word in words if len(word) >= _FUZZY_FLOOR
               for function in _JOB_FUNCTIONS)


# One quarter, planned on its own. "What should I take in Winter?" is a
# planning request scoped to a term -- not a catalog question about which
# electives happen to run then, which is why a first-person planning phrase is
# required and not just the season's name.
_PLAN_ONE_QUARTER = re.compile(
    r"\b(?:"
    r"plan(?:ning)?\s+(?:my|out|just)|just\s+plan|"
    r"(?:what|which)\s+(?:should|do|am|would)\s+i\s+(?:be\s+)?tak\w+|"
    r"should\s+i\s+take|what\s+do\s+i\s+take|what\s+am\s+i\s+taking|"
    r"my\s+(?:summer|fall|winter|spring)|"
    r"(?:just|only)\s+(?:the\s+)?(?:summer|fall|winter|spring)|"
    r"(?:tak\w+|schedul\w+|fill)\s+(?:in|for)\s+(?:the\s+)?"
    r"(?:summer|fall|winter|spring)"
    r")\b", re.IGNORECASE)


# A greeting, and nothing else. Deterministic because it has to be: a model
# call to classify "hi" costs a round trip (measured at 1.8s on this backend)
# to learn something a set membership test knows, and it is the very first
# thing a lot of students type.
_GREETINGS = frozenset("""
hi hii hiya hey heya hello helo yo sup morning afternoon evening
thanks ta cheers ok okay k sure right cool nice
""".split()) | {
    "good morning", "good afternoon", "good evening", "how are you",
    "how's it going", "hows it going", "what's up", "whats up",
    "hi there", "hey there", "hello there", "thank you", "thanks a lot",
}


def is_greeting(question):
    """A whole message that is only a greeting or an acknowledgement."""
    lowered = re.sub(r"[^a-z' ]+", " ", (question or "").lower())
    lowered = " ".join(lowered.split())
    return bool(lowered) and lowered in _GREETINGS


def named_quarter(question):
    """The quarter this question is about, in the student's words, or "".

    Deliberately loose about the TRACK. Which quarter keys exist depends on it
    ("fall-two" only on the 17-month track) and the router does not know the
    student, so the words are captured here and resolved against the real
    track by `planner.resolve_quarter` in the handler -- which returns None
    rather than guessing when they do not fit.
    """
    match = re.search(
        r"\b(summer(?:\s+iii)?|fall(?:\s*\(?\s*second\s+year\s*\)?)?|"
        r"second\s+fall|fall\s+two|winter|spring)\b",
        question or "", re.IGNORECASE)
    return " ".join(match.group(1).split()).lower() if match else ""


def _words(question):
    return {word.strip(".,/-?!'\"") for word in (question or "").lower().split()}


def _is_situational(question):
    text = question or ""
    return bool(_POSITION_WORDS.search(text) and _PROGRAMME_WORDS.search(text))


# "what jobs do you have", "which careers do you cover", "list the roles".
#
# A closed question with a closed answer -- fourteen rows in `careers.json` --
# so it never needs a model, and asking one to place it produced a clarifying
# question instead of the list. The catch is that the words overlap almost
# entirely with a REQUEST ("what courses do you have for a data scientist"), so
# both halves have to match and nothing may name a role, a course or a field.
_CAREER_NOUNS = re.compile(
    r"\b(careers?|jobs?|roles?|professions?|job titles?|career paths?)\b",
    re.IGNORECASE)
_INVENTORY_VERBS = re.compile(
    r"\b(have|offer|support|cover|know|list|available|there|options?|"
    r"suggest|recommend|help with|do you do)\b", re.IGNORECASE)
# Words that turn "what jobs..." into a question about a PARTICULAR job, which
# is the role route's, not this one's.
# "become" earns its place here the same way "as" and "into" did: it opens a
# PURPOSE clause, and a question with one is about that purpose rather than
# about the shelf. Without it "what electives should I take to become an
# esports analyst" satisfied both halves of the inventory test -- "electives"
# and "take" -- and answered a question about one job with all 86 courses.
# The role guard above did not save it, because that only recognises the
# fourteen roles we curate, and the phrasing is commonest for the ones we do
# not. The docstring already promised this: false for "what should I take".
# "to be a <job>" is matched as a PHRASE, not on the word "be". Bare "be"
# would fire on "what courses can be taken", which is an inventory question
# and must keep its catalog answer; "to be a" only ever opens a purpose
# clause.
_NOT_INVENTORY = re.compile(
    r"(\bto\s+be\s+an?\b)|"
    r"\b(for|in|as|about|towards?|into|become|becoming|require|need|pay|"
    r"salary|skills?)\b",
    re.IGNORECASE)


_OWN_JOB = re.compile(
    r"\b(?:i|i've|ive|i'd|we)\s+(?:have|had|got|hold)\b|"
    r"\b(?:my|a|an)\s+(?:part[- ]time|full[- ]time|current|day|new)\s+job\b|"
    r"\bmy job\b", re.IGNORECASE)


def asks_which_careers(question):
    """True for "which careers do you know", false for "what does X need"."""
    text = question or ""
    if not (_CAREER_NOUNS.search(text) and _INVENTORY_VERBS.search(text)):
        return False
    # A job the STUDENT has is not a question about the jobs we cover. "light,
    # i have a part time job" -- an answer to the load question, with a reason
    # -- carries "have" and "job" and was answered with the list of fourteen
    # careers. First-person possession of a job is a fact about them, and the
    # inventory question is always about us ("do you have", "are there").
    if _OWN_JOB.search(text):
        return False
    if _NOT_INVENTORY.search(text) or names_a_course(text):
        return False
    return not matched_role(text)


# "what courses do you have", "what classes can I take", "what's in the
# catalog". Like `asks_which_careers`, a closed question with a closed answer,
# and one a model reliably UNDER-answers: handed 86 rows it replies "86 courses,
# 6 core and 80 electives, all others are electives", which is arithmetic the
# student can already see and none of the context they asked for.
_COURSE_NOUNS = re.compile(
    r"\b(courses?|classes|electives?|catalog|curriculum|offerings?)\b", re.IGNORECASE)
# Wider than the careers rule's, because a catalog gets asked about in more
# ways than a career list does -- "show me", "what can I take", "what have you
# got". "in" stays OUT of the exclusions' reach only here; "courses IN finance"
# is still excluded below, because it names a field.
_CATALOG_VERBS = re.compile(
    r"\b(have|has|is|are|offer|offers|support|cover|covers|know|list|available|there|"
    r"options?|show|see|browse|take|get|access|hold|holds|include|includes|"
    r"all|everything|full|entire|whole)\b", re.IGNORECASE)


# "in the catalog", "in the programme" -- a container, not a subject area.
_OUR_CONTAINER = re.compile(
    r"\bin\s+(?:the\s+)?(?:catalog|curriculum|programme|program|msba|"
    r"course\s+list)\b", re.IGNORECASE)


def asks_which_courses(question):
    """True for "what courses do you have", false for "what should I take"."""
    text = question or ""
    if not (_COURSE_NOUNS.search(text) and _CATALOG_VERBS.search(text)):
        return False
    # A course, a role or a field named makes this a question about THAT, which
    # the factual, role and industry routes answer better than a list does.
    if names_a_course(text) or matched_role(text):
        return False
    # `_NOT_INVENTORY` blocks "in", which is right for "courses IN finance" and
    # wrong for "what is IN the catalog" -- the second one names our own
    # container, not a field. Strip that reading before testing.
    without_container = _OUR_CONTAINER.sub(" ", text)
    if _NOT_INVENTORY.search(without_container):
        return False
    return True


# What may sit around a bare course code and still leave it a bare course
# code: a department prefix, digits, and the words people put either side of
# one when they are pointing at it.
_COURSE_ONLY_FILLER = frozenset("""
mgta mgtf mgt mgtp cse the a an this that course class about
""".split())


def rule_route(question):
    """A confident route, decided here, or None to ask the model.

    Order matters and is argued, not arbitrary:

    1. **Situational first.** "I switched to the 11-month track, what do I take
       to finish?" names no course and no role, but a later rule would happily
       read "take" as a catalog question. Where a student is standing changes
       what every other route would answer.
    2. **A named course with a catalog word** is factual. Nothing about "does
       MGTA 464 have prerequisites" is a recommendation request, and paying for
       a model call to learn that is the round trip this whole layer exists to
       avoid.
    3. **A curated role** is the deterministic path — with an industry named
       alongside it, the combination.

    Anything else goes to the model. In particular a bare industry is NOT ruled
    here even though `is_industry_course_question` would fire: that regex is
    loose enough to claim "what courses do you have", which is not an industry
    question at all, and letting it rule would reinstate the scattered routing
    this module replaces.
    """
    text = question or ""
    if is_greeting(text):
        return Route(UNCLEAR, why="a greeting; nothing to route yet")
    if asks_which_careers(text):
        return Route(CAREERS, why="asked which careers this planner knows")
    if asks_which_courses(text):
        return Route(CATALOG, why="asked what the catalog holds, overall")
    about_a_course = names_a_course(text) and bool(_words(text) & _FACTUAL_WORDS)
    if about_a_course and not _OWN_STANDING.search(text):
        # A course question first when nothing in the sentence is about the
        # student's own standing. "is MGTA 479 offered in winter? will that
        # be a problem" carried "winter" and read as situational.
        return Route(FACTUAL, why="named a course and asked a catalog question")
    if _is_situational(text):
        return Route(SITUATIONAL, why="named a position in the programme")
    if about_a_course:
        return Route(FACTUAL, why="named a course and asked a catalog question")
    # A turn that is ONLY a course code is about that course. "MGTA 458" on
    # its own carries no question word at all, and the rule above therefore
    # declined it -- leaving a bare code to be classified, or worse, to reach
    # an intake that would answer it with a plan.
    if names_a_course(text) and not (_words(text) - _COURSE_ONLY_FILLER
                                     - set(COURSE_CODE.findall(text.lower())[0]
                                           if COURSE_CODE.search(text) else [])):
        return Route(FACTUAL, why="named a course and nothing else")
    # A single quarter, asked about in the first person. After the factual rule
    # on purpose: "does MGTA 464 run in winter" names a course and asks what
    # the catalog says, which is a different question from "what should I take
    # in winter".
    quarter = named_quarter(text)
    if quarter and _PLAN_ONE_QUARTER.search(text):
        return Route(QUARTER, quarter=quarter,
                     why="asked what to take in one named quarter")
    role_id = matched_role(text)
    if role_id:
        # `_target_field`, not `is_industry_course_question`: the latter needs a
        # catalog word ("courses", "electives") before it will look at a field
        # at all, and "data scientist in the food industry" has none -- it was
        # being filed as a plain role question, which is the exact case the
        # combination route exists for.
        field = _target_field(text)
        # The ROLE is not also the FIELD. "aiming for business analyst"
        # satisfies the field pattern on the word "for", so the job title
        # itself came back as the industry and the turn was ruled a
        # combination of Business Analyst with the field "business analyst".
        # That misrouted it past the intake -- which only runs for ROLE -- and
        # a sentence carrying a role, a track AND a load kept only the role.
        titles = {str(t).lower() for t in
                  (load_careers().get(role_id) or {}).get("titles") or []}
        titles.add(str((load_careers().get(role_id) or {}).get("label", "")).lower())
        if field and field.lower() in titles:
            field = None
        if field or is_industry_course_question(text):
            return Route(COMBINATION, role_id=role_id, industry=field or "",
                         why="named a curated role and a field")
        return Route(ROLE, role_id=role_id, why="named a curated role")
    return None


# The classifier's instructions live in `skills/read_turn/SKILL.md`, not
# here. What used to be a string in this module asking for a route name is
# now a skill asking for a reading of the whole turn -- route AND every
# planning fact the turn states -- so that a phrasing nobody tested for still
# lands somewhere the code can act on. See `services/skills/__init__.py`.
READ_TURN = "read_turn"


def _learned_from_reading(turn, question):
    """Intake values a reading states outright, validated. Deterministic."""
    learned = {}
    track = reading.track_of(turn.get("track"))
    if track:
        learned["track"] = track
    load = reading.load_of(turn.get("load"))
    if load:
        learned["workload"] = load
    # By code or by title -- "I already did fraud analytics" names a course
    # as surely as "MGTA 463" does, and the validator only keeps a title that
    # matches exactly one catalog row.
    codes = reading.courses_of(turn.get("completed_codes"))
    if codes:
        learned["completed_codes"] = codes
    learned.update(reading.ratings_of(turn.get("skills")))
    return learned


PLAN_ACTIONS = frozenset({"next", "start_review", "finalise", "show_plan",
                          "swap", "alternatives", "explain", "set_load",
                          "switch_route"})


def _action_from_reading(turn):
    """What the reading asks to be done to a plan on screen, validated."""
    name = reading.one_of(turn.get("plan_action"), PLAN_ACTIONS)
    if not name:
        return {}
    action = {"name": name}
    courses = reading.courses_of(turn.get("courses"))
    if courses:
        action["courses"] = courses
    load = reading.load_of(turn.get("load"))
    if load:
        action["load"] = load
    quarter = reading.words_of(turn.get("quarter"), 40)
    if quarter:
        action["quarter"] = quarter
    choice = reading.route_choice_of(turn.get("route_choice"))
    if choice:
        action["route_choice"] = choice
    return action


ACTS = frozenset({"answer", "request", "ask", "correct", "accept", "decline",
                  "greet", "thank", "chat"})


def model_route(llm, question, context=None):
    """The `read_turn` skill's route for a question, with everything it read.

    Public for the one caller that needs a reading of a turn the rules HAVE
    placed -- see `orchestrator.answer`, where a profile alias that is also a
    course title is decided by the skill rather than by the rule.
    """
    turn = reading.read(READ_TURN, llm, question, context,
                       roster=understanding.roster(),
                       industries=understanding.industry_roster())
    if turn.degraded:
        # SAY the model is unreachable rather than picking a route without it.
        #
        # This used to default to FACTUAL, on the reasoning that a catalog
        # answer is the safest thing to do knowing nothing. What that produced
        # during a real outage was "I don't have catalog or syllabus material
        # that answers that" to every single turn -- a content refusal blaming
        # the corpus for a billing problem. Measured live: both backends went
        # down (a lapsed Codex entitlement and an exhausted TritonAI budget)
        # and the app spent ten turns insisting it had no material.
        return Route(DEGRADED, why="the language model could not be reached")
    parsed = turn.raw
    if not parsed:
        return Route(UNCLEAR, why="classifier returned nothing usable")
    # Everything the turn STATED travels with the route whatever the route
    # is, so a turn the classifier could not place still puts its track or
    # its load on file. Read before the route is judged, because an "unclear"
    # that carries the answer to the pending question is not unclear at all.
    act = reading.one_of(parsed.get("act"), ACTS)
    learned = _learned_from_reading(turn, question)
    action = _action_from_reading(turn)
    wants_menu = reading.flag_of(parsed.get("wants_menu"))
    position = position_of(parsed)
    menu_pick = _positions_of(parsed.get("menu_pick"))
    menu_compare = reading.flag_of(parsed.get("menu_compare"))
    name = str(parsed.get("route") or "").strip().lower()
    if name not in ROUTES:
        return Route(UNCLEAR, why=f"classifier returned unknown route {name!r}",
                     act=act, learned=learned, action=action,
                     wants_menu=wants_menu, menu_pick=menu_pick,
                     menu_compare=menu_compare, position=position)
    confidence = turn.confidence
    role_said = str(parsed.get("role") or "").strip()[:60]
    industry = str(parsed.get("industry") or "").strip()[:60]
    quarter = str(parsed.get("quarter") or "").strip()[:40]
    if name == UNCLEAR or (confidence < MIN_CONFIDENCE and name != OUT_OF_SCOPE):
        # Out of scope is exempt: a low-confidence "this isn't about the
        # programme" still means the model saw nothing it recognised, and
        # asking a clarifying question about the weather is worse than saying
        # plainly that it is not something this tool does.
        #
        # An unclear reading still carries the JOB the student named, when
        # they named one: "i want to be a yoga instructor? what happens if i
        # get this wrong" came back unclear, and the generic opening asked
        # what they were aiming for. `unmatched_role` lets the orchestrator
        # refuse it as the uncovered job it is.
        return Route(UNCLEAR, confidence,
                     why=("classified unclear" if name == UNCLEAR
                          else f"low confidence on {name!r}"),
                     unmatched_role=role_said if not matched_role(role_said) else "",
                     act=act, learned=learned, action=action,
                     wants_menu=wants_menu, menu_pick=menu_pick,
                     menu_compare=menu_compare, position=position)
    route = Route(name, confidence, industry=industry,
                  quarter=quarter or named_quarter(question), why="classified",
                  act=act, learned=learned, action=action,
                  wants_menu=wants_menu, menu_pick=menu_pick,
                  menu_compare=menu_compare, position=position)
    if name in (ROLE, COMBINATION):
        # The model may name a role; only `careers.json` may resolve one. It
        # is asked what the STUDENT said, and the curated table decides whether
        # that is a role we serve -- which is what keeps the deterministic path
        # deterministic even when a model chose the route into it.
        route.role_id = matched_role(role_said) or matched_role(question)
        if not route.role_id and role_said:
            # Nothing on disk matched the words. The model was ALSO asked
            # which curated profile, if any, those words denote -- from the
            # closed list, by id -- and `understanding.curated_reading` throws
            # away anything that is not one of ours. A confident reading is
            # acted on AND confirmed back (`Route.reading`), the way a
            # misspelled title is; anything less stays an uncovered role.
            #
            # Only when the student typed a JOB (`role_said`). "i want to
            # work in financial services" names a field and no job, and the
            # model still offered the Financial Analytics profile with
            # confidence -- a field mapped onto a career is the guess this
            # module refuses; the industry route has the right answer.
            read_id, read_confidence = understanding.curated_reading(parsed)
            if read_id and read_confidence >= understanding.MIN_READING_CONFIDENCE:
                route.role_id = read_id
                route.reading = role_said
                route.why = "classified; the model read the job as a curated profile"
            else:
                route.unmatched_role = role_said
        elif not route.role_id and (industry or understanding.curated_industry(parsed)[0]):
            route.name = INDUSTRY
            route.why = "classified as a role with no job named; a field was -- industry"
    if name in (INDUSTRY, COMBINATION, ROLE):
        # Same arrangement as the role: the model may say which of the six
        # curated industries the field belongs to, by id, from the closed
        # list, and `understanding.curated_industry` keeps only an id that
        # `industries.json` holds. The handler still tries the aliases on
        # disk first; this is the reading for the words no alias covers.
        read_id, read_confidence = understanding.curated_industry(parsed)
        if read_id and read_confidence >= understanding.MIN_READING_CONFIDENCE:
            route.industry_id = read_id
    return route


def position_of(parsed):
    """Where the student said they are, validated -- the situational route's
    input, read off the same turn reading as everything else.

    Used to be its own skill (`read_position`), a second model call on every
    situational turn asking for five fields `read_turn` already had room
    for. One reading, one call; the tracks are checked against the two that
    exist, the codes against the catalog, and the quarter is left in the
    student's words for `situation.resolve_position` to check against the
    track.
    """
    if not isinstance(parsed, dict):
        return {}
    out = {}
    for key in ("track", "previous_track"):
        value = reading.track_of(parsed.get(key))
        if value:
            out[key] = value
    said = parsed.get("current_quarter")
    if isinstance(said, str) and said.strip():
        out["said_quarter"] = said.strip()[:60]
    out["finish_on_time"] = reading.flag_of(parsed.get("finish_on_time"))
    codes = reading.courses_of(parsed.get("completed_codes"))
    if codes:
        out["completed_codes"] = codes
    return out


def read_position(llm, question, context=None):
    """The position alone, for the turn the situational route asks about.

    The awaiting path -- a bare "winter" answering "which quarter are you
    in" -- reaches the situational handler without going through
    `classify`, so it reads the turn here with the same skill and keeps
    the position fields.
    """
    turn = reading.read(READ_TURN, llm, question, context,
                        roster=understanding.roster(),
                        industries=understanding.industry_roster())
    if turn.degraded or not turn.raw:
        return {}
    return position_of(turn.raw)


def _positions_of(value):
    """1-based menu positions the model reported, as ints, or []."""
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value[:3]:
        try:
            position = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= position <= 10 and position not in out:
            out.append(position)
    return out


def classify(llm, question, context=None):
    """The route for this question. Rules first, then one model call.

    `context` is what the planner knows about the conversation -- the
    question it just asked, what is on file, whether a plan is on screen --
    rendered into the model's input by `reading.render_context`. The rules do
    not need it; the reading does, because "heavy" and "17" mean different
    things depending on what was asked.
    """
    ruled = rule_route(question)
    if ruled is not None:
        return ruled
    return model_route(llm, question, context)


_model_route = model_route
