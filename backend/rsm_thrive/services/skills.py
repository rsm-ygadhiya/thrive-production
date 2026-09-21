"""Skills: what the model is told to do, written down where staff can edit it.

## The problem this solves

Every reader on the course surface used to be a regular expression or a word
list, and every one of them was grown the same way: a student typed something
nobody had tested for, the bot said "I want to make sure I answer the right
question", and a phrase was added. A list can only ever hold what someone has
already seen a student type, so the untested phrasing is, by construction, the
one it cannot read. The rules are not wrong -- they are fast, free and
deterministic, and they stay in front. What changes is what happens when they
find nothing: the model is asked to READ the turn against an instruction kept
as a file. That instruction is the skill.

## What a skill is here

One directory holding a `SKILL.md`, under the bot it belongs to:

    services/courses/skills/<name>/SKILL.md      the course planner's
    services/resources/skills/<name>/SKILL.md    the Resources bot's

The file is the instruction the model receives, in prose, with `<<name>>`
placeholders for the closed lists rendered live from the catalog. Programme
staff can edit the wording without a deploy, exactly as they edit
`careers.json`; the file is re-read on every call for that reason, and it is
a few kilobytes.

This module is the runner, shared by both bots and knowing nothing about
either. Each bot has a thin facade that binds it to that bot's `skills/`
directory and adds that bot's own validators -- see
`services/courses/reading.py`.

## What a skill is NOT allowed to do

Decide anything. The model interprets; the tables decide. Every value a
reading skill returns passes through a validator that knows the closed set
the value must belong to, and anything outside it is dropped, not repaired.
A prose skill (`say`) is given the facts it may use and told they are the
whole of what it may say. That is the line `understanding.py` and
`situation.py` already held, made the property of every prompt at once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from rsm_thrive.services.llm import parse_llm_json

# `<<roster>>`, `<<quarters>>` -- a closed list rendered into the prompt at
# call time. Doubled angle brackets rather than braces, because the skill
# files show the model JSON examples, and JSON is made of braces.
_PLACEHOLDER = re.compile(r"<<([a-z_]+)>>")
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class SkillError(Exception):
    """A skill file is missing, malformed, or was rendered with a gap in it."""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    #: The instructions, with `<<placeholders>>` still in them.
    body: str

    @property
    def placeholders(self):
        return frozenset(_PLACEHOLDER.findall(self.body))

    def prompt(self, **fill):
        """The system prompt, with every placeholder rendered.

        Raises rather than sending a prompt with `<<roster>>` still in it:
        a model shown a literal placeholder where the list of profiles should
        be will answer from its own idea of what the profiles are, which is
        exactly the source of careers this codebase refuses to have.
        """
        missing = self.placeholders - set(fill)
        if missing:
            raise SkillError(
                f"skill {self.name!r} needs {sorted(missing)} rendered")
        return _PLACEHOLDER.sub(lambda m: str(fill[m.group(1)]), self.body)


def available(root):
    """Every skill under `root`, by name."""
    return sorted(path.parent.name for path in Path(root).glob("*/SKILL.md"))


def load(root, name):
    """The skill called `name` under `root`, read fresh from disk."""
    path = Path(root) / name / "SKILL.md"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SkillError(f"no skill named {name!r} under {root}") from None
    match = _FRONTMATTER.match(text)
    if not match:
        raise SkillError(f"skill {name!r} has no frontmatter")
    meta = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    if meta.get("name") != name:
        raise SkillError(f"skill {name!r} declares name {meta.get('name')!r}")
    body = text[match.end():].strip()
    if not body:
        raise SkillError(f"skill {name!r} has no instructions")
    return Skill(name, meta.get("description", ""), body)


# ---------------------------------------------------------------------------
# Running one
# ---------------------------------------------------------------------------

@dataclass
class Reading:
    """What a skill read off one turn. Validate before acting."""
    skill: str
    #: The model's answer as parsed. Callers read fields through validators,
    #: never raw -- see `courses/reading.py` for the shape of a well-behaved
    #: reader.
    raw: dict = field(default_factory=dict)
    confidence: float = 0.0
    #: True when the model could not be reached at all. Distinct from a
    #: reading that found nothing: an outage is reported as an outage.
    degraded: bool = False

    def get(self, key, default=None):
        return self.raw.get(key, default)


def render_context(context):
    """The bot's own records, as a preamble the model reads before the turn.

    "heavy" answers the load question and describes a course, depending on
    which question is pending. A reader that is not told what was asked can
    only guess, and guessing is what the regexes did. The context is written
    as facts about the conversation, labelled as ours, so the model does not
    mistake it for something the student said.
    """
    if not context:
        return ""
    lines = ["Context, from THRIVE's own records (not the student's words):"]
    for key, value in context.items():
        if value in (None, "", [], {}):
            continue
        lines.append(f"- {key}: {value}")
    if len(lines) == 1:
        return ""
    lines += ["", "The student's message:"]
    return "\n".join(lines) + "\n"


def read(root, name, llm, text, context=None, fast=False, **fill):
    """Run reading skill `name` over one turn. Never raises for a model failure.

    `fast=True` runs it on `llm.reader()` -- the smaller model, when one is
    configured. Opt-in per skill, because the two readers are not equal:
    measured 2026-09-20, gemini-3.5-flash read the Resources bot's
    `read_question` as well as claude-sonnet-5, but on the planner's
    `read_turn` it mapped "esports analyst" onto a curated profile with
    confidence, which the live sweep rightly fails. A reading that decides
    which of fourteen careers a student gets stays on the main model.
    """
    skill = load(root, name)
    system = skill.prompt(**fill)
    content = render_context(context) + str(text or "")
    model = llm.reader() if fast else llm
    try:
        raw = model.chat(system, [{"role": "user", "content": content}],
                         json_mode=True)
    except Exception:
        return Reading(name, degraded=True)
    parsed = parse_llm_json(raw or "")
    if not isinstance(parsed, dict):
        parsed = {}
    return Reading(name, parsed, confidence_of(parsed.get("confidence")))


def say(root, name, llm, messages, search=False, sources_out=None,
        search_query=None, **fill):
    """Run a prose skill: the model's reply as text, or "" if it failed.

    For the prompts that WRITE rather than read -- the Resources bot's
    answer from passages, the course planner's explanation of a position.
    The facts the reply may use are rendered into the skill's placeholders by
    the caller, which is the grounding rule made mechanical: what is not in
    the prompt is not available to say.

    `search=True` goes through `LLM.search_chat`, so a backend that cannot
    search natively has `websearch` do it and the pages land in the prompt;
    `sources_out` and `search_query` are passed through unchanged.
    """
    skill = load(root, name)
    system = skill.prompt(**fill)
    try:
        if search:
            raw = llm.search_chat(system, messages, sources_out=sources_out,
                                  search_query=search_query)
        else:
            raw = llm.chat(system, messages)
    except Exception:
        return ""
    return (raw or "").strip()


def say_json(root, name, llm, messages, search=False, sources_out=None,
             search_query=None, **fill):
    """Like `say`, but the skill answers in JSON and the dict comes back.

    {} when the model failed or said nothing parseable, so a caller can fail
    closed -- refuse, rather than pass prose through as though it had been
    judged. `sources_out` and `search_query` pass through to
    `LLM.search_chat` when `search` is on, so the caller can cite what was
    read and choose what is looked up.
    """
    skill = load(root, name)
    system = skill.prompt(**fill)
    try:
        if search:
            raw = llm.search_chat(system, messages, True, sources_out=sources_out,
                                  search_query=search_query)
        else:
            raw = llm.chat(system, messages, json_mode=True)
    except Exception:
        return {}
    parsed = parse_llm_json(raw or "")
    return parsed if isinstance(parsed, dict) else {}


# ---------------------------------------------------------------------------
# Validators every bot shares
# ---------------------------------------------------------------------------
#
# Each takes whatever the model said and returns a value the caller can act
# on, or the empty value. None of them repairs.

def confidence_of(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def one_of(value, allowed, default=""):
    """`value` if it is one of `allowed` (case- and space-insensitive)."""
    said = " ".join(str(value or "").lower().replace("-", " ").split())
    for option in allowed:
        if said == " ".join(str(option).lower().replace("-", " ").split()):
            return option
    return default


def words_of(value, limit=60):
    """The student's own words for something, trimmed, or ""."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:limit]


def flag_of(value):
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("true", "yes", "1")
